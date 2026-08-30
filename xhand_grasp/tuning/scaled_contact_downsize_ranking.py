"""Deterministic schema-v13 ranking with soft fingertip-slip tie breaks."""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
)
from ..artifacts import file_sha256, write_json
from ..config import ACTIVE_ACTUATORS, ACTIVE_FINGERS
from ..grasp_pose import canonical_sha256
from ..scene import build_model
from .actual_contact_manipulation import (
    manipulation_candidate_rank,
    rank_manipulation_candidates,
)


def _summary(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("summary", "result"):
        value = record.get(key)
        if isinstance(value, Mapping):
            return value
    return record


def _schema_version(
    record: Mapping[str, Any], config: Mapping[str, Any] | None
) -> int | None:
    candidate = record.get("config")
    if isinstance(candidate, Mapping):
        try:
            return int(candidate.get("schema_version", 1))
        except (TypeError, ValueError):
            return None
    if isinstance(config, Mapping):
        try:
            return int(config.get("schema_version", 1))
        except (TypeError, ValueError):
            return None
    return None


def v13_contact_slip_rank_evidence(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract a finite, comparison-ready soft slip summary from one result."""

    summary = _summary(record)
    metrics = summary.get("metrics", {})
    targeting = (
        metrics.get("contact_point_targeting", {})
        if isinstance(metrics, Mapping)
        else {}
    )
    slip = (
        targeting.get("contact_slip_from_grasp", {})
        if isinstance(targeting, Mapping)
        else {}
    )
    operation = slip.get("operation", {}) if isinstance(slip, Mapping) else {}
    per_finger = (
        operation.get("per_finger", {})
        if isinstance(operation, Mapping)
        else {}
    )
    baseline = slip.get("baseline", {}) if isinstance(slip, Mapping) else {}

    valid_duties: list[float] = []
    p95_values: list[float] = []
    max_values: list[float] = []
    baseline_valid = True
    for finger in ACTIVE_FINGERS:
        finger_baseline = (
            baseline.get(finger, {}) if isinstance(baseline, Mapping) else {}
        )
        baseline_valid = bool(
            baseline_valid
            and isinstance(finger_baseline, Mapping)
            and finger_baseline.get("valid") is True
        )
        values = per_finger.get(finger, {}) if isinstance(per_finger, Mapping) else {}
        if not isinstance(values, Mapping):
            values = {}
        try:
            duty = float(values.get("valid_duty"))
        except (TypeError, ValueError):
            duty = -math.inf
        try:
            p95 = float(values.get("tangent_slip_p95_m"))
        except (TypeError, ValueError):
            p95 = math.inf
        try:
            maximum = float(values.get("tangent_slip_max_m"))
        except (TypeError, ValueError):
            maximum = math.inf
        valid_duties.append(duty if math.isfinite(duty) else -math.inf)
        p95_values.append(p95 if math.isfinite(p95) else math.inf)
        max_values.append(maximum if math.isfinite(maximum) else math.inf)

    return {
        "soft_ranking_only": True,
        "all_grasp_baselines_valid": bool(baseline_valid),
        "operation_min_valid_duty": min(valid_duties, default=-math.inf),
        "operation_worst_finger_tangent_slip_p95_m": max(
            p95_values, default=math.inf
        ),
        "operation_worst_finger_tangent_slip_max_m": max(
            max_values, default=math.inf
        ),
    }


def annotate_v13_manipulation_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy carrying the exact slip evidence used by the sorter."""

    result = copy.deepcopy(dict(record))
    result["v13_contact_slip_rank_evidence"] = v13_contact_slip_rank_evidence(
        result
    )
    return result


def v13_manipulation_candidate_rank(
    record: Mapping[str, Any], config: Mapping[str, Any] | None
) -> tuple[Any, ...]:
    base = manipulation_candidate_rank(record, config=config)
    evidence = v13_contact_slip_rank_evidence(record)
    # Preserve every existing hard/margin priority and its candidate-ID final
    # tie break.  Slip is inserted only before failed-count/ID, so it can never
    # turn a hard failure into a winner over a full success.
    return (
        *base[:5],
        not bool(evidence["all_grasp_baselines_valid"]),
        -float(evidence["operation_min_valid_duty"]),
        float(evidence["operation_worst_finger_tangent_slip_p95_m"]),
        float(evidence["operation_worst_finger_tangent_slip_max_m"]),
        *base[5:],
    )


def rank_v13_manipulation_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Rank v13 records by hard success then soft slip, independent of workers.

    For every non-v13 input this is an exact delegation to the historical
    sorter, preserving the v1-v12 numerical and record-shape contract.
    """

    materialized = [copy.deepcopy(dict(record)) for record in records]
    if not materialized:
        return ()
    schemas = {_schema_version(record, config) for record in materialized}
    if schemas != {13}:
        return rank_manipulation_candidates(materialized, config=config)
    annotated = [annotate_v13_manipulation_record(record) for record in materialized]
    annotated.sort(
        key=lambda record: v13_manipulation_candidate_rank(record, config)
    )
    return tuple(annotated)


_COMPACTION_REPORT_SCHEMA_VERSION = 2
_COMPACTION_RANKING_POLICY = "schema_v13_hard_then_soft_contact_slip"


def _full_success(record: Mapping[str, Any]) -> bool:
    summary = _summary(record)
    stage = summary.get("stage_status", {})
    return bool(
        isinstance(stage, Mapping)
        and stage.get("full_success") is True
        and summary.get("passed") is True
    )


def _candidate_paths(record: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    result_path = Path(str(record["result_path"])).resolve()
    directory = result_path.parent
    return (
        directory / "resolved_config.json",
        result_path,
        directory / "trace.npz",
        directory / ".trace.npz.compacting",
    )


def _trace_evaluation_sha256(
    payload: Mapping[str, Any],
    *,
    trace_path: Path,
    tombstone_path: Path,
) -> tuple[str, bool]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError(f"manipulation result has no artifact mapping: {trace_path.parent}")
    hashes = artifacts.get("sha256")
    if not isinstance(hashes, Mapping):
        raise RuntimeError(f"manipulation result has no artifact hashes: {trace_path.parent}")
    retained = bool(artifacts.get("trace_retained", True))
    expected = (
        hashes.get("trace")
        if retained
        else artifacts.get("trace_sha256_at_evaluation")
    )
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise RuntimeError(
            f"manipulation result lost its evaluation trace SHA-256: {trace_path.parent}"
        )
    if retained:
        if artifacts.get("trace") is None or hashes.get("trace") != expected:
            raise RuntimeError(
                f"retained manipulation trace metadata is inconsistent: {trace_path.parent}"
            )
    elif artifacts.get("trace") is not None or "trace" in hashes:
        raise RuntimeError(
            f"compacted manipulation trace metadata is inconsistent: {trace_path.parent}"
        )
    if trace_path.is_file() and tombstone_path.exists():
        raise RuntimeError(
            f"manipulation trace and compaction tombstone both exist: {trace_path.parent}"
        )
    observed_path = trace_path if trace_path.is_file() else tombstone_path
    if observed_path.is_file() and file_sha256(observed_path) != expected:
        raise RuntimeError(
            f"manipulation trace state SHA-256 mismatch: {observed_path}"
        )
    return expected, retained


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_replace(source: Path, destination: Path) -> None:
    _fsync_file(source)
    source.replace(destination)
    _fsync_directory(destination.parent)


def _durable_unlink(path: Path) -> None:
    path.unlink()
    _fsync_directory(path.parent)


def _reconcile_trace_tombstone(
    payload: dict[str, Any],
    *,
    result_path: Path,
    trace_path: Path,
    tombstone_path: Path,
    successful: bool,
) -> dict[str, Any]:
    """Recover every durable state of the v13 trace-compaction transaction.

    The rename is the transaction intent marker.  A retained result plus a
    tombstone has not crossed the result commit point and is restored.  A
    compacted result plus a tombstone has crossed that point and only needs
    authenticated tombstone deletion.
    """

    expected, retained = _trace_evaluation_sha256(
        payload, trace_path=trace_path, tombstone_path=tombstone_path
    )
    if successful and not retained:
        raise RuntimeError(
            f"successful manipulation trace was already compacted: {trace_path}"
        )
    if not tombstone_path.exists():
        if retained and not trace_path.is_file():
            raise RuntimeError(
                f"manipulation trace disappeared before compaction: {trace_path}"
            )
        if not retained and trace_path.exists():
            raise RuntimeError(
                f"compacted manipulation result unexpectedly retained trace: {trace_path}"
            )
        return payload

    if retained:
        _durable_replace(tombstone_path, trace_path)
        if file_sha256(trace_path) != expected:
            raise RuntimeError(
                f"restored manipulation trace changed: {trace_path}"
            )
        return payload

    # result.json was already committed as compacted.  Complete that
    # transaction idempotently after authenticating the tombstone above.
    if file_sha256(tombstone_path) != expected:
        raise RuntimeError(
            f"manipulation tombstone changed before deletion: {tombstone_path}"
        )
    _durable_unlink(tombstone_path)
    return json.loads(result_path.read_text(encoding="utf-8"))


def _authenticate_record_binding(
    record: Mapping[str, Any], payload: Mapping[str, Any], result_path: Path
) -> str:
    semantic_sha256 = authenticate_candidate_result_semantic_sha256(
        payload, source=result_path
    )
    recorded_semantic = record.get("result_semantic_sha256")
    if recorded_semantic is not None and recorded_semantic != semantic_sha256:
        raise RuntimeError(
            f"manipulation candidate semantic evidence changed: {result_path}"
        )
    recorded_summary = record.get("summary")
    if isinstance(recorded_summary, Mapping) and canonical_sha256(
        recorded_summary
    ) != canonical_sha256(_summary(payload)):
        raise RuntimeError(
            f"manipulation candidate summary changed: {result_path}"
        )
    return semantic_sha256


def _persisted_candidate_evidence(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path, result_path, trace_path, tombstone_path = _candidate_paths(record)
    if not config_path.is_file() or not result_path.is_file():
        raise RuntimeError(f"manipulation candidate artifacts disappeared: {result_path.parent}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    semantic_sha256 = _authenticate_record_binding(record, payload, result_path)
    payload = _reconcile_trace_tombstone(
        payload,
        result_path=result_path,
        trace_path=trace_path,
        tombstone_path=tombstone_path,
        successful=_full_success(payload),
    )
    semantic_sha256 = _authenticate_record_binding(record, payload, result_path)
    trace_sha256, retained = _trace_evaluation_sha256(
        payload, trace_path=trace_path, tombstone_path=tombstone_path
    )
    if tombstone_path.exists():  # pragma: no cover - reconciliation is exhaustive.
        raise RuntimeError(f"unresolved manipulation compaction tombstone: {tombstone_path}")
    evidence = {
        "candidate_id": int(record["candidate_id"]),
        "config_sha256": file_sha256(config_path),
        "result_semantic_sha256": semantic_sha256,
        "trace_sha256_at_evaluation": trace_sha256,
        "full_success": _full_success(payload),
    }
    state = {
        **evidence,
        "trace_retained": retained,
        "result_file_sha256": file_sha256(result_path),
    }
    return evidence, state


def authenticate_v13_manipulation_compaction_report(
    records: Sequence[Mapping[str, Any]],
    report_path: str | Path,
    *,
    retain_failure_trace_count: int,
) -> tuple[dict[str, Any], ...]:
    """Authenticate a reusable compaction checkpoint against live artifacts.

    A later, wider compaction may monotonically change a failed trace from
    retained to compacted.  That transition is accepted only when config,
    semantic result evidence and the original trace digest still match.
    """

    path = Path(report_path).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("actual_contact_manipulation_compaction_schema_version")
        != _COMPACTION_REPORT_SCHEMA_VERSION
        or report.get("complete") is not True
        or report.get("candidate_ranking_policy") != _COMPACTION_RANKING_POLICY
        or int(report.get("retained_failure_trace_limit", -1))
        != int(retain_failure_trace_count)
    ):
        raise RuntimeError(f"schema-v13 manipulation compaction changed: {path}")
    materialized = [copy.deepcopy(dict(value)) for value in records]
    identifiers = sorted(int(value["candidate_id"]) for value in materialized)
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("schema-v13 compaction input has duplicate candidate IDs")
    if report.get("candidate_ids") != identifiers or report.get(
        "candidate_id_set_sha256"
    ) != canonical_sha256(identifiers):
        raise RuntimeError(f"schema-v13 compaction candidate set changed: {path}")
    raw_entries = report.get("candidate_records")
    if not isinstance(raw_entries, list) or len(raw_entries) != len(materialized):
        raise RuntimeError(f"schema-v13 compaction candidate records changed: {path}")
    entries = {int(value.get("candidate_id", -1)): value for value in raw_entries}
    if set(entries) != set(identifiers) or len(entries) != len(raw_entries):
        raise RuntimeError(f"schema-v13 compaction candidate records changed: {path}")

    refreshed: list[dict[str, Any]] = []
    invariant_evidence: list[dict[str, Any]] = []
    for value in sorted(materialized, key=lambda item: int(item["candidate_id"])):
        entry = entries[int(value["candidate_id"])]
        evidence, state = _persisted_candidate_evidence(value)
        expected_evidence = {
            key: entry.get(key)
            for key in (
                "candidate_id",
                "config_sha256",
                "result_semantic_sha256",
                "trace_sha256_at_evaluation",
                "full_success",
            )
        }
        if evidence != expected_evidence:
            raise RuntimeError(
                f"schema-v13 compaction candidate evidence changed: {path}"
            )
        report_retained = bool(entry.get("trace_retained", False))
        current_retained = bool(state["trace_retained"])
        if (not report_retained and current_retained) or (
            bool(entry.get("full_success", False)) and not current_retained
        ):
            raise RuntimeError(
                f"schema-v13 compaction trace state is not monotonic: {path}"
            )
        if current_retained == report_retained and entry.get(
            "result_file_sha256_after_compaction"
        ) != state["result_file_sha256"]:
            raise RuntimeError(
                f"schema-v13 compaction persisted result changed: {path}"
            )
        invariant_evidence.append(evidence)
        current = copy.deepcopy(value)
        current.update(
            {
                "trace_retained": current_retained,
                "trace_sha256_at_evaluation": state[
                    "trace_sha256_at_evaluation"
                ],
                "result_semantic_sha256": state["result_semantic_sha256"],
            }
        )
        refreshed.append(current)
    if report.get("candidate_persisted_evidence_sha256") != canonical_sha256(
        invariant_evidence
    ):
        raise RuntimeError(f"schema-v13 compaction evidence digest changed: {path}")
    return tuple(refreshed)


def compact_v13_manipulation_candidate_artifacts(
    records: Sequence[Mapping[str, Any]],
    *,
    retain_failure_trace_count: int = 1,
    report_path: Path | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Compact schema-v13 traces using the slip-aware deterministic rank.

    This intentionally mirrors the proven actual-contact compactor while
    keeping the v13 ranking extension out of the hash-bound legacy module.
    """

    if (
        not isinstance(retain_failure_trace_count, int)
        or isinstance(retain_failure_trace_count, bool)
        or retain_failure_trace_count < 0
    ):
        raise ValueError("retain_failure_trace_count must be a non-negative integer")
    from .actual_contact_grasp_pose import (
        _load_persisted_manipulation_candidate,
    )
    from .actual_contact_manipulation import manipulation_delta_bounds

    materialized = [copy.deepcopy(dict(value)) for value in records]

    failures = sorted(
        (value for value in materialized if not _full_success(value)),
        key=lambda value: v13_manipulation_candidate_rank(
            value, value.get("config")
        ),
    )
    retained_ids = {
        int(value["candidate_id"])
        for value in failures[:retain_failure_trace_count]
    }
    retained_ids.update(
        int(value["candidate_id"])
        for value in materialized
        if _full_success(value)
    )
    compacted: list[dict[str, Any]] = []
    for raw in materialized:
        identifier = int(raw["candidate_id"])
        config_path, result_path, trace_path, tombstone = _candidate_paths(raw)
        directory = result_path.parent
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        _authenticate_record_binding(raw, payload, result_path)
        payload = _reconcile_trace_tombstone(
            payload,
            result_path=result_path,
            trace_path=trace_path,
            tombstone_path=tombstone,
            successful=_full_success(raw),
        )
        artifacts = payload.setdefault("artifacts", {})
        hashes = artifacts.setdefault("sha256", {})
        keep = identifier in retained_ids
        trace_retained = bool(artifacts.get("trace_retained", True))
        if keep and not trace_retained:
            if _full_success(raw):
                raise RuntimeError(
                    f"successful manipulation trace was already compacted: {trace_path}"
                )
            keep = False
        if not keep and trace_retained:
            if not trace_path.is_file():
                raise RuntimeError(
                    f"manipulation trace disappeared before compaction: {trace_path}"
                )
            observed = file_sha256(trace_path)
            if hashes.get("trace") != observed:
                raise RuntimeError(
                    f"refusing to compact changed manipulation trace: {trace_path}"
                )
            _durable_replace(trace_path, tombstone)
            artifacts["trace"] = None
            artifacts["trace_retained"] = False
            artifacts["trace_sha256_at_evaluation"] = observed
            hashes.pop("trace", None)
            write_json(result_path, payload)
            _fsync_file(result_path)
            _fsync_directory(result_path.parent)
            if file_sha256(tombstone) != observed:
                raise RuntimeError(
                    f"manipulation tombstone changed before deletion: {tombstone}"
                )
            _durable_unlink(tombstone)
        elif keep:
            artifacts["trace_retained"] = True
            write_json(result_path, payload)
            _fsync_file(result_path)
            _fsync_directory(result_path.parent)
        expected_config = json.loads(config_path.read_text(encoding="utf-8"))
        loaded = _load_persisted_manipulation_candidate(
            directory,
            candidate_id=identifier,
            expected_config=expected_config,
        )
        if loaded is None:  # pragma: no cover
            raise RuntimeError("compacted manipulation candidate disappeared")
        compacted.append(loaded)
    compacted.sort(key=lambda value: int(value["candidate_id"]))

    ranked = sorted(
        compacted,
        key=lambda value: v13_manipulation_candidate_rank(
            value, value.get("config")
        ),
    )
    boundary_hits: list[str] = []
    if ranked and not _full_success(ranked[0]):
        best = ranked[0]
        config = best["config"]
        model, _ = build_model(copy.deepcopy(config))
        bounds = manipulation_delta_bounds(model, config)
        delta = best["manipulation_delta_rad"]
        boundary_hits = [
            name
            for name in ACTIVE_ACTUATORS
            if abs(float(delta[name]) - float(bounds[name][1])) <= 1e-9
            or abs(float(delta[name]) - float(bounds[name][0])) <= 1e-9
        ]
    persisted: list[tuple[dict[str, Any], dict[str, Any]]] = [
        _persisted_candidate_evidence(value)
        for value in compacted
    ]
    invariant_evidence = [value[0] for value in persisted]
    persisted_states = {
        int(value[1]["candidate_id"]): value[1] for value in persisted
    }
    candidate_ids = sorted(int(value["candidate_id"]) for value in compacted)
    report = {
        "actual_contact_manipulation_compaction_schema_version": (
            _COMPACTION_REPORT_SCHEMA_VERSION
        ),
        "complete": True,
        "candidate_ranking_policy": _COMPACTION_RANKING_POLICY,
        "candidate_ids": candidate_ids,
        "candidate_id_set_sha256": canonical_sha256(candidate_ids),
        "candidate_persisted_evidence_sha256": canonical_sha256(
            invariant_evidence
        ),
        "candidate_count": len(compacted),
        "full_success_count": sum(_full_success(value) for value in compacted),
        "retained_failure_trace_limit": int(retain_failure_trace_count),
        "retained_trace_count": sum(value["trace_retained"] for value in compacted),
        "compacted_trace_count": sum(
            not value["trace_retained"] for value in compacted
        ),
        "success_traces_always_retained": True,
        "boundary_limited": bool(boundary_hits),
        "best_candidate_boundary_hits": boundary_hits,
        "boundary_expansion": {
            "registered_bounds_changed": False,
            "status": "diagnostic_only_no_bound_change",
        },
        "candidate_records": [
            {
                "candidate_id": int(value["candidate_id"]),
                "full_success": _full_success(value),
                "config_sha256": persisted_states[int(value["candidate_id"])][
                    "config_sha256"
                ],
                "trace_retained": persisted_states[int(value["candidate_id"])][
                    "trace_retained"
                ],
                "trace_sha256_at_evaluation": persisted_states[
                    int(value["candidate_id"])
                ]["trace_sha256_at_evaluation"],
                "result_semantic_sha256": persisted_states[
                    int(value["candidate_id"])
                ]["result_semantic_sha256"],
                "result_file_sha256_after_compaction": persisted_states[
                    int(value["candidate_id"])
                ]["result_file_sha256"],
            }
            for value in sorted(compacted, key=lambda item: int(item["candidate_id"]))
        ],
    }
    if report_path is not None:
        if report_path.is_file():
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            if existing != report:
                raise RuntimeError(
                    f"schema-v13 compaction report changed: {report_path}"
                )
        else:
            write_json(report_path, report)
            _fsync_file(report_path)
            _fsync_directory(report_path.parent)
    return tuple(compacted), report


__all__ = [
    "annotate_v13_manipulation_record",
    "authenticate_v13_manipulation_compaction_report",
    "compact_v13_manipulation_candidate_artifacts",
    "rank_v13_manipulation_candidates",
    "v13_contact_slip_rank_evidence",
    "v13_manipulation_candidate_rank",
]
