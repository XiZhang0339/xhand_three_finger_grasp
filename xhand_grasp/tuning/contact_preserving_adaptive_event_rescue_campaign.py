"""Fourth-stage adaptive contact-event rescue for schema-v14 lifts.

The completed third-stage event campaign is immutable input.  This runner
authenticates that input (and its authenticated rescue-v2 ancestry),
materializes any summary-only Pareto center by a fresh full-reset rerun in the
*new* workspace, then executes three deterministic stages:

* 64 diagnostic projections when the numerical API exposes that stage;
* 512 projected exploration candidates; and
* 8 refinement centers with 64 candidates each.

Candidate physics, compact search artifacts, final full-reset reruns, MP4
rendering, and the power-loss-safe ledger are deliberately reused from the
production v14 runners.  This module only owns provenance, budget allocation,
adaptive routing, jerk-first ranking, and catalog selection.  It never writes
to the completed source campaign.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..actual_contact_grasp_pose_catalog import (
    authenticated_catalog_artifact_paths,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from ..artifacts import file_sha256, write_json
from ..config import load_config
from ..experiment import resolve_experiment
from ..grasp_pose import canonical_sha256
from .contact_preserving_candidate_artifacts import (
    authenticate_v14_candidate_artifacts,
)
from .contact_preserving_lift_rescue_campaign import (
    _commit_report,
    _execute_candidate_jobs,
    _load_committed_report,
    _publish_rescue_viewer_catalogs,
    _run_full_reset_candidate,
    _validate_phase_report,
)
from .contact_preserving_planned_lift_campaign import (
    build_contact_preserving_planned_lift_manifest,
)


ADAPTIVE_CAMPAIGN_SCHEMA_VERSION = 1
ADAPTIVE_SOURCE_AUDIT_SCHEMA_VERSION = 1
ADAPTIVE_PHASE_REPORT_SCHEMA_VERSION = 1
ADAPTIVE_RESULT_SCHEMA_VERSION = 1
DIAGNOSTIC_CANDIDATE_COUNT = 64
EXPLORATION_CANDIDATE_COUNT = 512
REFINEMENT_CENTER_COUNT = 8
REFINEMENT_CANDIDATES_PER_CENTER = 64
REFINEMENT_CANDIDATE_COUNT = (
    REFINEMENT_CENTER_COUNT * REFINEMENT_CANDIDATES_PER_CENTER
)
PUBLISHED_CANDIDATE_COUNT = 5
DEFAULT_SEED = 20260821
_JERK_CHECK = "smooth_motion_jerk_within_limit"


@dataclass(frozen=True, slots=True)
class _AdaptiveEventApi:
    source_authenticator: Callable[..., Any]
    center_discoverer: Callable[..., Sequence[Any]]
    center_selector: Callable[..., Sequence[Any]] | None
    center_bundle_authenticator: Callable[..., Any]
    source_type: type
    center_type: type
    descriptor_type: type
    polytope_type: type
    descriptor_builder: Callable[..., Any]
    descriptor_authenticator: Callable[..., Any]
    polytope_builder: Callable[..., Any]
    jobs_builder: Callable[..., Sequence[Mapping[str, Any]]]
    job_authenticator: Callable[..., Any]
    ranker: Callable[[Mapping[str, Any]], Any]
    budget_type: type


def _adaptive_event_api() -> _AdaptiveEventApi:
    """Load the numerical API only when the fourth-stage runner is invoked."""

    from . import contact_preserving_adaptive_event_rescue as adaptive

    required = {
        "authenticate_adaptive_event_source": "source_authenticator",
        "discover_jerk_pareto_centers": "center_discoverer",
        "authenticate_adaptive_center_bundle": "center_bundle_authenticator",
        "AdaptiveEventSource": "source_type",
        "AdaptiveEventCenter": "center_type",
        "AdaptiveEventDescriptor": "descriptor_type",
        "FeasibleEventPolytope": "polytope_type",
        "build_adaptive_event_descriptor": "descriptor_builder",
        "authenticate_adaptive_event_descriptor": "descriptor_authenticator",
        "derive_feasible_event_polytope": "polytope_builder",
        "build_adaptive_event_jobs": "jobs_builder",
        "authenticate_adaptive_event_job": "job_authenticator",
        "adaptive_event_candidate_rank": "ranker",
        "AdaptiveEventBudget": "budget_type",
    }
    missing = [name for name in required if not hasattr(adaptive, name)]
    if missing:
        raise RuntimeError(
            "adaptive-event numerical API is incomplete: " + ", ".join(missing)
        )
    values = {field: getattr(adaptive, name) for name, field in required.items()}
    values["center_selector"] = getattr(
        adaptive, "select_jerk_pareto_centers", None
    )
    return _AdaptiveEventApi(**values)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    for method_name in ("as_mapping", "to_dict", "descriptor"):
        method = getattr(value, method_name, None)
        if callable(method):
            payload = method()
            if isinstance(payload, Mapping):
                return copy.deepcopy(dict(payload))
    raise RuntimeError(f"{label} has no canonical mapping representation")


def _from_mapping(value_type: type, payload: Mapping[str, Any], *, label: str) -> Any:
    converter = getattr(value_type, "from_mapping", None)
    if callable(converter):
        return converter(copy.deepcopy(dict(payload)))
    try:
        return value_type(**copy.deepcopy(dict(payload)))
    except TypeError as exc:  # pragma: no cover - defensive API boundary.
        raise RuntimeError(f"{label} cannot be reconstructed from its mapping") from exc


def _attribute(value: Any, name: str, *, label: str) -> Any:
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, Mapping) and name in value:
        return value[name]
    raise RuntimeError(f"{label} lost required field {name}")


def _source_root(source: Any) -> Path:
    return Path(str(_attribute(source, "root", label="adaptive source"))).resolve()


def _source_id(source: Any) -> str:
    value = str(
        _attribute(
            source, "source_authentication_id", label="adaptive source"
        )
    )
    if len(value) != 64:
        raise RuntimeError("adaptive source authentication ID is not a SHA-256")
    return value


def _source_artifact_paths(source: Any) -> tuple[Path, ...]:
    raw = getattr(source, "artifact_paths", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RuntimeError("adaptive source artifact_paths must be a sequence")
    paths: list[Path] = []
    for value in raw:
        path = Path(value).expanduser().resolve()
        # The authenticated adaptive source intentionally includes its
        # immutable rescue-v2 ancestor, which is a sibling workspace rather
        # than a child of the completed event campaign.
        if not path.is_file():
            raise RuntimeError("adaptive source evidence is missing")
        paths.append(path)
    return tuple(dict.fromkeys(paths))


def _source_evidence_sha256(source: Any) -> dict[str, str]:
    root = _source_root(source)
    result: dict[str, str] = {}
    for path in _source_artifact_paths(source):
        key = (
            f"event_source/{path.relative_to(root)}"
            if path.is_relative_to(root)
            else f"authenticated_ancestor/{canonical_sha256(str(path))}/{path.name}"
        )
        result[key] = file_sha256(path)
    return dict(sorted(result.items()))


def _source_rerun_records(source: Any) -> tuple[Any, ...]:
    raw = getattr(source, "rerun_required_records", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RuntimeError("adaptive source rerun_required_records must be a sequence")
    return tuple(raw)


def _record_candidate_id(record: Any) -> int:
    value = _attribute(record, "candidate_id", label="adaptive rerun record")
    if isinstance(value, bool) or int(value) < 0:
        raise RuntimeError("adaptive rerun record has an invalid candidate ID")
    return int(value)


def _record_config(record: Any) -> dict[str, Any]:
    raw = getattr(record, "config", None)
    if raw is None and isinstance(record, Mapping):
        raw = record.get("config")
    if raw is None:
        config_path = Path(
            str(_attribute(record, "config_path", label="adaptive rerun record"))
        ).expanduser().resolve()
        if not config_path.is_file():
            raise RuntimeError("adaptive rerun record config_path is missing")
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or int(raw.get("schema_version", 0)) != 14:
        raise RuntimeError("adaptive rerun record lost its schema-v14 config")
    return copy.deepcopy(dict(raw))


def _center_id(center: Any) -> str:
    value = str(_attribute(center, "center_id", label="adaptive center"))
    if len(value) != 64:
        raise RuntimeError("adaptive center ID is not a SHA-256")
    return value


def _center_config(center: Any) -> dict[str, Any]:
    raw = _attribute(center, "config", label="adaptive center")
    if not isinstance(raw, Mapping) or int(raw.get("schema_version", 0)) != 14:
        raise RuntimeError("adaptive center lost its schema-v14 config")
    return copy.deepcopy(dict(raw))


def _center_trace_path(center: Any) -> Path:
    path = Path(
        str(_attribute(center, "trace_path", label="adaptive center"))
    ).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError("adaptive center retained trace is missing")
    expected = getattr(center, "trace_sha256", None)
    if expected is not None and str(expected) != file_sha256(path):
        raise RuntimeError("adaptive center retained trace SHA-256 changed")
    return path


def _center_candidate_id(center: Any) -> int:
    value = _attribute(center, "candidate_id", label="adaptive center")
    return int(value)


def _center_peak_jerk(center: Any) -> float:
    mapping = _mapping(center, label="adaptive center")
    candidates = (
        mapping.get("peak_abs_filtered_jerk_m_s3"),
        mapping.get("peak_jerk_m_s3"),
        mapping.get("operation_peak_abs_filtered_jerk_m_s3"),
    )
    summary = getattr(center, "summary", mapping.get("summary"))
    if isinstance(summary, Mapping):
        candidates = (
            *candidates,
            summary.get("metrics", {})
            .get("motion_smoothness", {})
            .get("operation_peak_abs_filtered_jerk_m_s3"),
        )
    for value in candidates:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return math.inf


def _fallback_select_centers(
    centers: Sequence[Any], *, max_centers: int
) -> tuple[Any, ...]:
    """Deterministic fallback; the numerical API normally owns Pareto policy."""

    unique: dict[str, Any] = {}
    for center in centers:
        mapping = _mapping(center, label="adaptive center")
        physical = str(
            mapping.get("physical_plan_sha256")
            or mapping.get("source_physical_plan_sha256")
            or _center_id(center)
        )
        incumbent = unique.get(physical)
        if incumbent is None or (
            _center_peak_jerk(center), _center_candidate_id(center)
        ) < (
            _center_peak_jerk(incumbent),
            _center_candidate_id(incumbent),
        ):
            unique[physical] = center
    return tuple(
        sorted(
            unique.values(),
            key=lambda value: (
                _center_peak_jerk(value),
                _center_candidate_id(value),
                _center_id(value),
            ),
        )[: int(max_centers)]
    )


def _select_centers(
    source: Any,
    materialized: Sequence[Any],
    api: _AdaptiveEventApi,
    *,
    max_centers: int = REFINEMENT_CENTER_COUNT,
) -> tuple[Any, ...]:
    try:
        discovered = tuple(
            api.center_discoverer(
                source,
                additional_centers=tuple(materialized),
                max_centers=int(max_centers),
            )
        )
        combined = discovered
    except TypeError:
        retained = tuple(api.center_discoverer(source, max_centers=int(max_centers)))
        combined = (*retained, *materialized)
    if not combined:
        raise RuntimeError("adaptive source contains no authenticated full-reset center")
    if api.center_selector is not None:
        selected = tuple(
            api.center_selector(combined, max_centers=int(max_centers))
        )
    else:
        selected = _fallback_select_centers(combined, max_centers=max_centers)
    ids = [_center_id(value) for value in selected]
    if not selected or len(ids) != len(set(ids)) or len(selected) > int(max_centers):
        raise RuntimeError("adaptive Pareto center selection is empty or non-unique")
    # Touch each trace/config now so no output stage starts from stale evidence.
    for center in selected:
        _center_config(center)
        _center_trace_path(center)
    return selected


def _authenticate_materialized_center(
    api: _AdaptiveEventApi,
    source: Any,
    record: Any,
    root: Path,
) -> tuple[Any, tuple[Path, ...]]:
    candidate_id = _record_candidate_id(record)
    config = _record_config(record)
    bundle = authenticate_v14_candidate_artifacts(
        root,
        expected_config=config,
        expected_candidate_id=candidate_id,
        require_retained_trace=True,
        expected_retain_grasp_success=False,
    )
    if bundle.trace_path is None:
        raise RuntimeError("adaptive center materialization lost trace.npz")
    source_result_path = Path(
        str(_attribute(record, "result_path", label="adaptive rerun record"))
    ).expanduser().resolve()
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if canonical_sha256(source_result.get("summary")) != canonical_sha256(
        bundle.result.get("summary")
    ):
        raise RuntimeError("adaptive center full-reset rerun changed its source summary")
    center = api.center_bundle_authenticator(
        root,
        source_record=record,
        source_authentication_id=_source_id(source),
    )
    _center_id(center)
    _center_trace_path(center)
    return center, bundle.artifact_paths


def _materialize_source_centers(
    source: Any,
    workspace: Path,
    api: _AdaptiveEventApi,
) -> tuple[tuple[Any, ...], Path]:
    stage = "adaptive_source_materialization"
    report_path = workspace / "adaptive_source_materialization" / "report.json"
    existing = _load_committed_report(workspace, stage, report_path)
    records = _source_rerun_records(source)
    if existing is not None:
        mappings = existing.get("centers")
        if not isinstance(mappings, list) or len(mappings) != len(records):
            raise RuntimeError("adaptive materialization report changed center count")
        centers: list[Any] = []
        for record, raw in zip(records, mappings, strict=True):
            if not isinstance(raw, Mapping):
                raise RuntimeError("adaptive materialized center mapping is malformed")
            root = (
                workspace
                / "adaptive_source_materialization"
                / "candidates"
                / f"candidate_{_record_candidate_id(record)}"
            ).resolve()
            center, _ = _authenticate_materialized_center(api, source, record, root)
            if canonical_sha256(_mapping(center, label="adaptive center")) != canonical_sha256(raw):
                raise RuntimeError("adaptive materialized center changed on resume")
            centers.append(center)
        return tuple(centers), report_path

    centers = []
    artifacts: list[Path] = []
    for record in records:
        candidate_id = _record_candidate_id(record)
        config = _record_config(record)
        root = (
            workspace
            / "adaptive_source_materialization"
            / "candidates"
            / f"candidate_{candidate_id}"
        )
        _run_full_reset_candidate(
            config,
            root,
            candidate_id,
            final_rerun=True,
        )
        center, paths = _authenticate_materialized_center(
            api, source, record, root.resolve()
        )
        centers.append(center)
        artifacts.extend(paths)
    payload = {
        "adaptive_source_materialization_schema_version": 1,
        "complete": True,
        "source_authentication_id": _source_id(source),
        "rerun_required_count": len(records),
        "centers": [_mapping(value, label="adaptive center") for value in centers],
        "records": [],
    }
    _commit_report(
        workspace,
        stage,
        report_path,
        payload,
        stage_input={
            "source_authentication_id": _source_id(source),
            "rerun_records_sha256": canonical_sha256(
                [_mapping(value, label="adaptive rerun record") for value in records]
            ),
        },
        artifacts=tuple(dict.fromkeys(artifacts)),
    )
    return tuple(centers), report_path


@dataclass(frozen=True, slots=True)
class _CenterContext:
    center: Any
    descriptor: Any
    polytope: Any

    @property
    def center_id(self) -> str:
        return _center_id(self.center)

    @property
    def descriptor_id(self) -> str:
        return str(
            _attribute(self.descriptor, "descriptor_id", label="adaptive descriptor")
        )

    @property
    def polytope_id(self) -> str:
        return str(
            _attribute(self.polytope, "polytope_id", label="adaptive polytope")
        )


def _build_contexts(
    centers: Sequence[Any], api: _AdaptiveEventApi
) -> tuple[_CenterContext, ...]:
    contexts: list[_CenterContext] = []
    for center in centers:
        config = _center_config(center)
        trace_path = _center_trace_path(center)
        with np.load(trace_path, allow_pickle=False) as trace:
            descriptor = api.descriptor_builder(center, trace)
            descriptor = _from_mapping(
                api.descriptor_type,
                _mapping(descriptor, label="adaptive descriptor"),
                label="adaptive descriptor",
            )
            api.descriptor_authenticator(descriptor, config, trace)
        polytope = api.polytope_builder(config, descriptor)
        polytope = _from_mapping(
            api.polytope_type,
            _mapping(polytope, label="adaptive polytope"),
            label="adaptive polytope",
        )
        contexts.append(_CenterContext(center, descriptor, polytope))
    descriptor_ids = [value.descriptor_id for value in contexts]
    if len(descriptor_ids) != len(set(descriptor_ids)):
        raise RuntimeError("adaptive centers produced duplicate descriptors")
    return tuple(contexts)


def _balanced_counts(total: int, count: int) -> tuple[int, ...]:
    if int(total) < 0 or int(count) <= 0:
        raise ValueError("adaptive budget count must be positive")
    quotient, remainder = divmod(int(total), int(count))
    values = tuple(quotient + (index < remainder) for index in range(int(count)))
    if min(values, default=0) <= 0 or sum(values) != int(total):
        raise RuntimeError("adaptive budget cannot be distributed across centers")
    return values


def _make_budget(
    api: _AdaptiveEventApi,
    *,
    stage: str,
    count: int,
    seed: int,
) -> Any:
    values = {
        "stage": stage,
        "total_candidate_count": int(count),
        "candidates_per_center": int(count),
        "max_center_count": 1,
        "seed": int(seed),
    }
    try:
        return api.budget_type(**values)
    except TypeError:
        # Early numerical prototypes used only stage/count/seed.  Keeping this
        # narrow compatibility path costs no provenance: the canonical budget
        # mapping is still embedded in every authenticated job payload.
        return api.budget_type(
            stage=stage,
            total_candidate_count=int(count),
            seed=int(seed),
        )


def _normalize_jobs(
    batches: Sequence[tuple[_CenterContext, Sequence[Mapping[str, Any]]]],
    *,
    api: _AdaptiveEventApi,
    expected_count: int,
    stage: str,
    excluded_physical_plan_sha256: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    jobs: list[dict[str, Any]] = []
    for context, raw_jobs in batches:
        for raw in raw_jobs:
            if not isinstance(raw, Mapping):
                raise RuntimeError(f"adaptive {stage} job is not a mapping")
            job = copy.deepcopy(dict(raw))
            try:
                api.job_authenticator(
                    job,
                    context.center,
                    context.descriptor,
                    context.polytope,
                    expected_excluded_physical_plan_sha256=tuple(
                        excluded_physical_plan_sha256
                    ),
                )
            except TypeError:
                if excluded_physical_plan_sha256:
                    raise
                api.job_authenticator(
                    job, context.center, context.descriptor, context.polytope
                )
            if (
                job.get("stage") != stage
                or job.get("full_reset_required") is not True
                or not isinstance(job.get("config"), Mapping)
                or not isinstance(job.get("projected_parameters"), Mapping)
            ):
                raise RuntimeError(
                    f"adaptive {stage} job lost stage, projection, config, or full-reset contract"
                )
            candidate_id = int(job.get("candidate_id", -1))
            local_index = int(job.get("local_index", -1))
            payload_sha = job.get("candidate_payload_sha256")
            if candidate_id < 0 or local_index < 0 or not isinstance(payload_sha, str) or len(payload_sha) != 64:
                raise RuntimeError(f"adaptive {stage} job has invalid identity evidence")
            metadata = {
                key: copy.deepcopy(value) for key, value in job.items() if key != "config"
            }
            metadata.update(
                {
                    "runner_stage": stage,
                    "source_center_id": context.center_id,
                    "job_sequence_index": local_index,
                    "full_reset_required": True,
                }
            )
            jobs.append(
                {
                    **job,
                    "candidate_sha256": payload_sha,
                    "job_sequence_index": local_index,
                    "job_metadata": metadata,
                }
            )
    if len(jobs) != int(expected_count):
        raise RuntimeError(
            f"adaptive {stage} built {len(jobs)} jobs, expected {expected_count}"
        )
    candidate_ids = [int(value["candidate_id"]) for value in jobs]
    payload_hashes = [str(value["candidate_payload_sha256"]) for value in jobs]
    local_indices = [int(value["local_index"]) for value in jobs]
    physical_hashes = [str(value.get("physical_plan_sha256", "")) for value in jobs]
    if any(len(value) != 64 for value in physical_hashes):
        raise RuntimeError(f"adaptive {stage} job lost its physical-plan SHA-256")
    for label, values in (
        ("candidate IDs", candidate_ids),
        ("candidate payload hashes", payload_hashes),
        ("local indices", local_indices),
        ("physical plans", physical_hashes),
    ):
        if len(values) != len(set(values)):
            raise RuntimeError(f"adaptive {stage} contains duplicate {label}")
    jobs.sort(key=lambda value: (int(value["local_index"]), int(value["candidate_id"])))
    if [int(value["local_index"]) for value in jobs] != list(range(expected_count)):
        raise RuntimeError(f"adaptive {stage} local indices are not contiguous")
    return tuple(jobs)


def _build_jobs_for_contexts(
    contexts: Sequence[_CenterContext],
    *,
    api: _AdaptiveEventApi,
    stage: str,
    total_count: int,
    seed: int,
    refinement_parameters: Sequence[Mapping[str, Any] | None] | None = None,
    excluded_physical_plan_sha256: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    if not contexts:
        raise RuntimeError(f"adaptive {stage} has no source center")
    counts = _balanced_counts(int(total_count), len(contexts))
    if refinement_parameters is None:
        parameters: tuple[Mapping[str, Any] | None, ...] = (None,) * len(contexts)
    else:
        parameters = tuple(refinement_parameters)
        if len(parameters) != len(contexts):
            raise RuntimeError("adaptive refinement parameters do not match centers")
    batches: list[tuple[_CenterContext, Sequence[Mapping[str, Any]]]] = []
    excluded = tuple(sorted(set(str(value) for value in excluded_physical_plan_sha256)))
    if any(len(value) != 64 for value in excluded):
        raise ValueError("adaptive excluded physical-plan identities are invalid")
    offset = 0
    for context, count, refinement in zip(contexts, counts, parameters, strict=True):
        budget = _make_budget(
            api,
            stage=stage,
            count=count,
            seed=int(seed),
        )
        raw = api.jobs_builder(
            context.center,
            context.descriptor,
            context.polytope,
            budget=budget,
            local_index_offset=offset,
            refinement_centers=(
                () if refinement is None else (copy.deepcopy(dict(refinement)),)
            ),
            excluded_physical_plan_sha256=excluded,
        )
        batches.append((context, tuple(raw)))
        offset += count
    normalized = _normalize_jobs(
        batches,
        api=api,
        expected_count=int(total_count),
        stage=stage,
        excluded_physical_plan_sha256=excluded,
    )
    overlap = {
        str(value["physical_plan_sha256"]) for value in normalized
    }.intersection(excluded)
    if overlap:
        raise RuntimeError(
            f"adaptive {stage} reused {len(overlap)} excluded physical plans"
        )
    return normalized


def _summary_failed_checks(record: Mapping[str, Any]) -> tuple[str, ...]:
    summary = record.get("summary")
    if not isinstance(summary, Mapping):
        return ("missing_summary",)
    failed = summary.get("failed_checks")
    if not isinstance(failed, list) or any(not isinstance(value, str) for value in failed):
        return ("missing_failed_checks",)
    return tuple(failed)


def _peak_jerk(record: Mapping[str, Any]) -> float:
    summary = record.get("summary")
    if not isinstance(summary, Mapping):
        return math.inf
    value = (
        summary.get("metrics", {})
        .get("motion_smoothness", {})
        .get("operation_peak_abs_filtered_jerk_m_s3")
    )
    return (
        float(value)
        if isinstance(value, (int, float)) and math.isfinite(float(value))
        else math.inf
    )


def _rank_key(record: Mapping[str, Any], api: _AdaptiveEventApi) -> tuple[Any, ...]:
    failed = _summary_failed_checks(record)
    non_jerk = tuple(value for value in failed if value != _JERK_CHECK)
    full_success = bool(record.get("full_success", False))
    if full_success and failed:
        group = 2
    elif full_success:
        group = 0
    elif not non_jerk and failed == (_JERK_CHECK,):
        group = 1
    else:
        group = 2
    numerical = api.ranker(record)
    if isinstance(numerical, list):
        numerical = tuple(numerical)
    elif not isinstance(numerical, tuple):
        numerical = (numerical,)
    candidate_id = int(record.get("candidate_id", 1 << 62))
    jerk = _peak_jerk(record)
    if group == 0:
        # A hard pass is selected by the established v14 contact/margin order;
        # jerk remains a tie-breaker within candidates already below its hard
        # threshold.
        return (0, *numerical, jerk, candidate_id)
    if group == 1:
        # For candidates whose only failure is jerk, this rescue exists
        # specifically to minimize that failure before secondary margins.
        return (1, jerk, *numerical, candidate_id)
    return (2, len(non_jerk), *numerical, jerk, candidate_id)


def _rank_records(
    records: Sequence[Mapping[str, Any]], api: _AdaptiveEventApi
) -> tuple[dict[str, Any], ...]:
    ranked = tuple(
        sorted(
            (copy.deepcopy(dict(value)) for value in records),
            key=lambda value: _rank_key(value, api),
        )
    )
    if {int(value["candidate_id"]) for value in ranked} != {
        int(value["candidate_id"]) for value in records
    }:
        raise RuntimeError("adaptive ranking lost candidates")
    return ranked


def _select_refinement_records(
    exploration_records: Sequence[Mapping[str, Any]],
    api: _AdaptiveEventApi,
    *,
    count: int = REFINEMENT_CENTER_COUNT,
) -> tuple[dict[str, Any], ...]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in _rank_records(exploration_records, api):
        metadata = record.get("rescue_job")
        if not isinstance(metadata, Mapping):
            raise RuntimeError("adaptive exploration record lost job provenance")
        parameters = metadata.get("projected_parameters")
        descriptor_id = metadata.get("descriptor_id")
        if not isinstance(parameters, Mapping) or not isinstance(descriptor_id, str):
            raise RuntimeError("adaptive exploration record lost projection evidence")
        digest = canonical_sha256(
            {"descriptor_id": descriptor_id, "projected_parameters": parameters}
        )
        if digest in seen:
            continue
        seen.add(digest)
        selected.append(copy.deepcopy(dict(record)))
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise RuntimeError("adaptive exploration did not retain eight unique refinement centers")
    return tuple(selected)


def _phase_payload(
    *,
    stage: str,
    records: Sequence[Mapping[str, Any]],
    source_authentication_id: str,
    contexts: Sequence[_CenterContext],
    refinement_centers: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    physical_hashes = _record_physical_plan_hashes(records)
    return {
        "adaptive_event_phase_report_schema_version": ADAPTIVE_PHASE_REPORT_SCHEMA_VERSION,
        "stage": stage,
        "complete": True,
        "declared_candidate_count": len(records),
        "candidate_count": len(records),
        "full_success_count": sum(bool(value.get("full_success", False)) for value in records),
        "physical_unique_candidate_count": len(set(physical_hashes)),
        "physical_duplicate_candidate_count": len(physical_hashes)
        - len(set(physical_hashes)),
        "physical_plan_set_sha256": canonical_sha256(sorted(set(physical_hashes))),
        "source_authentication_id": source_authentication_id,
        "center_ids": [value.center_id for value in contexts],
        "descriptor_ids": [value.descriptor_id for value in contexts],
        "polytope_ids": [value.polytope_id for value in contexts],
        "refinement_centers": [copy.deepcopy(dict(value)) for value in refinement_centers],
        "records": [copy.deepcopy(dict(value)) for value in records],
    }


def _record_physical_plan_hashes(
    records: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    result: list[str] = []
    for record in records:
        metadata = record.get("rescue_job")
        value = metadata.get("physical_plan_sha256") if isinstance(metadata, Mapping) else None
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError("adaptive result record lost physical-plan identity")
        result.append(value)
    return tuple(result)


def _execute_phase(
    workspace: Path,
    *,
    stage: str,
    directory: str,
    jobs: Sequence[Mapping[str, Any]],
    contexts: Sequence[_CenterContext],
    source_authentication_id: str,
    workers: int,
    global_rank_offset: int,
    refinement_centers: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], Path]:
    report_path = workspace / directory / "report.json"
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        _validate_phase_report(
            existing,
            expected_candidate_count=len(jobs),
            label=f"v14 adaptive {stage}",
        )
        return existing, report_path
    records, artifacts = _execute_candidate_jobs(
        jobs,
        workspace,
        phase_name=f"{directory}/candidates",
        workers=int(workers),
        global_rank_offset=int(global_rank_offset),
    )
    payload = _phase_payload(
        stage=stage,
        records=records,
        source_authentication_id=source_authentication_id,
        contexts=contexts,
        refinement_centers=refinement_centers,
    )
    committed = _commit_report(
        workspace,
        stage,
        report_path,
        payload,
        stage_input={
            "source_authentication_id": source_authentication_id,
            "jobs_sha256": canonical_sha256(jobs),
            "context_sha256": canonical_sha256(
                [
                    {
                        "center_id": value.center_id,
                        "descriptor_id": value.descriptor_id,
                        "polytope_id": value.polytope_id,
                    }
                    for value in contexts
                ]
            ),
            "refinement_centers_sha256": canonical_sha256(refinement_centers),
        },
        artifacts=artifacts,
    )
    _validate_phase_report(
        committed,
        expected_candidate_count=len(jobs),
        label=f"v14 adaptive {stage}",
    )
    return committed, report_path


def _selected_publication_records(
    records: Sequence[Mapping[str, Any]], api: _AdaptiveEventApi
) -> tuple[dict[str, Any], ...]:
    return _rank_records(records, api)[:PUBLISHED_CANDIDATE_COUNT]


def _materialize_publication_records(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    target_success_count: int,
) -> tuple[dict[str, Any], ...]:
    materialized: list[dict[str, Any]] = []
    for record in records:
        candidate_id = int(record["candidate_id"])
        source_root = (workspace / str(record["artifact_directory"])).resolve()
        if not source_root.is_relative_to(workspace):
            raise RuntimeError("adaptive catalog source escaped its workspace")
        bundle = authenticate_v14_candidate_artifacts(
            source_root,
            expected_candidate_id=candidate_id,
            expected_retain_grasp_success=False,
        )
        config = json.loads(bundle.config_path.read_text(encoding="utf-8"))
        final_root = (
            workspace
            / "catalog_source_reruns"
            / f"adaptive_target_{target_success_count}"
            / f"candidate_{candidate_id}"
        )
        rerun = _run_full_reset_candidate(
            config,
            final_root,
            candidate_id,
            final_rerun=True,
        )
        final_bundle = authenticate_v14_candidate_artifacts(
            final_root,
            expected_config=config,
            expected_candidate_id=candidate_id,
            require_retained_trace=True,
            expected_retain_grasp_success=False,
        )
        if canonical_sha256(rerun.get("summary")) != canonical_sha256(record.get("summary")):
            raise RuntimeError("adaptive catalog full-reset rerun changed its summary")
        if (
            bool(rerun.get("grasp_success", False)) != bool(record.get("grasp_success", False))
            or bool(rerun.get("full_success", False)) != bool(record.get("full_success", False))
        ):
            raise RuntimeError("adaptive catalog full-reset rerun changed its status")
        materialized.append(
            {
                **copy.deepcopy(dict(record)),
                **copy.deepcopy(final_bundle.result),
                "artifact_directory": str(final_root.relative_to(workspace)),
            }
        )
    return tuple(materialized)


def _catalog_stage(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    api: _AdaptiveEventApi,
    experiment_id: str,
    target_success_count: int,
    phase_report_hashes: Mapping[str, str],
) -> dict[str, Any]:
    stage = f"adaptive_catalog_target_{target_success_count}"
    root = workspace / "catalogs" / f"target_{target_success_count}"
    report_path = root / "report.json"
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        return existing
    selected = _selected_publication_records(records, api)
    materialized = _materialize_publication_records(
        selected,
        workspace,
        target_success_count=int(target_success_count),
    )
    catalogs = _publish_rescue_viewer_catalogs(
        materialized,
        workspace,
        root,
        experiment_id=experiment_id,
    )
    # The reused publisher owns byte-identical artifact and video semantics;
    # replace only the descriptive policy string with this runner's order.
    for relative in catalogs.values():
        path = workspace / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["selection_policy"] = (
            "non_jerk_hard_checks_then_peak_jerk_then_adaptive_event_rank"
        )
        payload["adaptive_event_rescue"] = True
        write_json(path, payload)
    artifacts: list[Path] = []
    for relative in catalogs.values():
        artifacts.extend(authenticated_catalog_artifact_paths(workspace / relative))
    full_count = sum(bool(value.get("full_success", False)) for value in records)
    payload = {
        "adaptive_event_catalog_report_schema_version": 1,
        "complete": True,
        "target_success_count": int(target_success_count),
        "full_success_count": full_count,
        "target_reached": full_count >= int(target_success_count),
        "published_candidate_count": len(materialized),
        "published_candidate_ids": [int(value["candidate_id"]) for value in materialized],
        "catalogs": catalogs,
        "selection_policy": "jerk_first_after_all_non_jerk_hard_checks",
        "video_policy": "independent_fresh_full_reset_ffprobe_and_full_decode",
        "records": [],
    }
    return _commit_report(
        workspace,
        stage,
        report_path,
        payload,
        stage_input={
            **copy.deepcopy(dict(phase_report_hashes)),
            "target_success_count": int(target_success_count),
            "selected_records_sha256": canonical_sha256(selected),
        },
        artifacts=tuple(dict.fromkeys(artifacts)),
    )


def _manifest(
    config_path: Path,
    source: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    base = build_contact_preserving_planned_lift_manifest(config_path, seed=int(seed))
    base.pop("campaign_input_sha256", None)
    root = _source_root(source)
    evidence = _source_evidence_sha256(source)
    base.update(
        {
            "contact_preserving_adaptive_event_campaign_schema_version": ADAPTIVE_CAMPAIGN_SCHEMA_VERSION,
            "campaign_kind": "contact_preserving_adaptive_event_rescue",
            "source_event_campaign_path": str(root),
            "source_event_authentication": _mapping(source, label="adaptive source"),
            "source_event_evidence_sha256": dict(sorted(evidence.items())),
            "adaptive_event_budget": {
                "diagnostic_candidate_count": DIAGNOSTIC_CANDIDATE_COUNT,
                "projected_exploration_candidate_count": EXPLORATION_CANDIDATE_COUNT,
                "local_refinement_center_count": REFINEMENT_CENTER_COUNT,
                "local_refinement_candidates_per_center": REFINEMENT_CANDIDATES_PER_CENTER,
                "local_refinement_candidate_count": REFINEMENT_CANDIDATE_COUNT,
                "publication_candidate_count": PUBLISHED_CANDIDATE_COUNT,
                "candidate_execution": "fresh_full_reset_free_dynamics",
                "catalog_execution": "independent_fresh_full_reset_rerun",
                "physical_plan_uniqueness": "global_across_all_search_stages",
                "ranking": "non_jerk_hard_checks_then_peak_jerk_then_adaptive_rank",
            },
        }
    )
    return {**base, "campaign_input_sha256": canonical_sha256(base)}


def build_contact_preserving_adaptive_event_rescue_manifest(
    config_path: str | Path,
    source_event_campaign: str | Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    api = _adaptive_event_api()
    source = api.source_authenticator(Path(source_event_campaign).expanduser().resolve())
    return _manifest(Path(config_path).expanduser().resolve(), source, seed=int(seed))


def run_contact_preserving_adaptive_event_rescue_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    source_event_campaign: str | Path,
    resume: bool,
    target_success_count: int,
    workers: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run/resume the fixed fourth-stage adaptive event rescue."""

    if int(workers) <= 0:
        raise ValueError("workers must be positive")
    if int(seed) != DEFAULT_SEED:
        raise ValueError("schema-v14 adaptive event rescue seed must be 20260821")
    if int(target_success_count) not in (1, 5):
        raise ValueError("target_success_count must be one or five")
    config_file = Path(config_path).expanduser().resolve()
    workspace = Path(output_dir).expanduser().resolve()
    source_root = Path(source_event_campaign).expanduser().resolve()
    if workspace == source_root or workspace.is_relative_to(source_root):
        raise ValueError("adaptive workspace must be outside its immutable event source")
    definition = resolve_experiment(load_config(config_file))
    if getattr(definition, "contact_preserving_planned_lift_campaign", None) is None:
        raise ValueError("adaptive runner requires the registered schema-v14 experiment")

    api = _adaptive_event_api()
    source = api.source_authenticator(source_root)
    if _source_root(source) != source_root:
        raise RuntimeError("adaptive source authenticator returned the wrong root")
    manifest = _manifest(config_file, source, seed=int(seed))
    initialize_or_resume_campaign(workspace, manifest, resume=bool(resume))

    materialized, materialization_path = _materialize_source_centers(
        source, workspace, api
    )
    centers = _select_centers(source, materialized, api)
    contexts = _build_contexts(centers, api)
    context_mapping = [
        {
            "center": _mapping(value.center, label="adaptive center"),
            "descriptor": _mapping(value.descriptor, label="adaptive descriptor"),
            "polytope": _mapping(value.polytope, label="adaptive polytope"),
        }
        for value in contexts
    ]

    audit_path = workspace / "adaptive_source_audit.json"
    audit_payload = {
        "adaptive_source_audit_schema_version": ADAPTIVE_SOURCE_AUDIT_SCHEMA_VERSION,
        "complete": True,
        "source": _mapping(source, label="adaptive source"),
        "source_authentication_id": _source_id(source),
        "materialization_report_sha256": file_sha256(materialization_path),
        "contexts": context_mapping,
        "evidence_sha256": _source_evidence_sha256(source),
        "records": [],
    }
    audit = _load_committed_report(workspace, "adaptive_source_audit", audit_path)
    if audit is None:
        audit = _commit_report(
            workspace,
            "adaptive_source_audit",
            audit_path,
            audit_payload,
            stage_input={
                "source_authentication_id": _source_id(source),
                "materialization_report_sha256": file_sha256(materialization_path),
                "contexts_sha256": canonical_sha256(context_mapping),
            },
        )
    elif canonical_sha256(audit) != canonical_sha256(audit_payload):
        raise RuntimeError("adaptive source or derived contexts changed on resume")

    diagnostics_jobs = _build_jobs_for_contexts(
        contexts,
        api=api,
        stage="diagnostic",
        total_count=DIAGNOSTIC_CANDIDATE_COUNT,
        seed=int(seed),
    )
    diagnostics, diagnostics_path = _execute_phase(
        workspace,
        stage="adaptive_diagnostics",
        directory="adaptive_diagnostics",
        jobs=diagnostics_jobs,
        contexts=contexts,
        source_authentication_id=_source_id(source),
        workers=int(workers),
        global_rank_offset=0,
    )

    exploration_jobs = _build_jobs_for_contexts(
        contexts,
        api=api,
        stage="exploration",
        total_count=EXPLORATION_CANDIDATE_COUNT,
        seed=int(seed),
        excluded_physical_plan_sha256=tuple(
            str(value["physical_plan_sha256"]) for value in diagnostics_jobs
        ),
    )
    exploration, exploration_path = _execute_phase(
        workspace,
        stage="adaptive_projected_exploration",
        directory="adaptive_projected_exploration",
        jobs=exploration_jobs,
        contexts=contexts,
        source_authentication_id=_source_id(source),
        workers=int(workers),
        global_rank_offset=DIAGNOSTIC_CANDIDATE_COUNT,
    )
    exploration_records = tuple(exploration["records"])
    refinement_records = _select_refinement_records(exploration_records, api)
    contexts_by_descriptor = {value.descriptor_id: value for value in contexts}
    refinement_contexts: list[_CenterContext] = []
    refinement_parameters: list[dict[str, Any]] = []
    refinement_evidence: list[dict[str, Any]] = []
    for record in refinement_records:
        metadata = record["rescue_job"]
        descriptor_id = str(metadata["descriptor_id"])
        context = contexts_by_descriptor.get(descriptor_id)
        if context is None:
            raise RuntimeError("adaptive refinement record references an unknown descriptor")
        parameters = copy.deepcopy(dict(metadata["projected_parameters"]))
        refinement_contexts.append(context)
        refinement_parameters.append(parameters)
        refinement_evidence.append(
            {
                "candidate_id": int(record["candidate_id"]),
                "source_center_id": context.center_id,
                "descriptor_id": descriptor_id,
                "projected_parameters": parameters,
            }
        )
    refinement_jobs = _build_jobs_for_contexts(
        refinement_contexts,
        api=api,
        stage="local_refinement",
        total_count=REFINEMENT_CANDIDATE_COUNT,
        seed=int(seed),
        refinement_parameters=refinement_parameters,
        excluded_physical_plan_sha256=tuple(
            str(value["physical_plan_sha256"])
            for value in (*diagnostics_jobs, *exploration_jobs)
        ),
    )
    refinement, refinement_path = _execute_phase(
        workspace,
        stage="adaptive_local_refinement",
        directory="adaptive_local_refinement",
        jobs=refinement_jobs,
        contexts=refinement_contexts,
        source_authentication_id=_source_id(source),
        workers=int(workers),
        global_rank_offset=DIAGNOSTIC_CANDIDATE_COUNT + EXPLORATION_CANDIDATE_COUNT,
        refinement_centers=refinement_evidence,
    )

    combined = _rank_records(
        (*diagnostics["records"], *exploration["records"], *refinement["records"]),
        api,
    )
    report_hashes = {
        "diagnostics_report_sha256": file_sha256(diagnostics_path),
        "exploration_report_sha256": file_sha256(exploration_path),
        "refinement_report_sha256": file_sha256(refinement_path),
    }
    catalog_report = _catalog_stage(
        combined,
        workspace,
        api=api,
        experiment_id=definition.experiment_id,
        target_success_count=int(target_success_count),
        phase_report_hashes=report_hashes,
    )
    full_count = sum(bool(value.get("full_success", False)) for value in combined)
    diagnostic_physical = _record_physical_plan_hashes(diagnostics["records"])
    exploration_physical = _record_physical_plan_hashes(exploration["records"])
    refinement_physical = _record_physical_plan_hashes(refinement["records"])
    combined_physical = (
        *diagnostic_physical,
        *exploration_physical,
        *refinement_physical,
    )
    if len(combined_physical) != len(set(combined_physical)):
        raise RuntimeError(
            "adaptive stages reused a physical plan despite the global exclusion contract"
        )
    result = {
        "contact_preserving_adaptive_event_rescue_result_schema_version": ADAPTIVE_RESULT_SCHEMA_VERSION,
        "complete": True,
        "experiment_id": definition.experiment_id,
        "source_event_campaign": str(source_root),
        "source_authentication_id": _source_id(source),
        "target_success_count": int(target_success_count),
        "diagnostic_candidate_count": len(diagnostics["records"]),
        "diagnostic_full_success_count": int(diagnostics["full_success_count"]),
        "diagnostic_physical_unique_candidate_count": len(
            set(diagnostic_physical)
        ),
        # Compatibility name consumed by the common tune CLI status printer.
        "exploration_candidate_count": len(exploration["records"]),
        "projected_exploration_candidate_count": len(exploration["records"]),
        "projected_exploration_full_success_count": int(exploration["full_success_count"]),
        "exploration_physical_unique_candidate_count": len(
            set(exploration_physical)
        ),
        "local_refinement_candidate_count": len(refinement["records"]),
        "local_refinement_full_success_count": int(refinement["full_success_count"]),
        "local_refinement_physical_unique_candidate_count": len(
            set(refinement_physical)
        ),
        "physical_unique_candidate_count": len(set(combined_physical)),
        "physical_duplicate_candidate_count": len(combined_physical)
        - len(set(combined_physical)),
        "physical_plan_set_sha256": canonical_sha256(
            sorted(set(combined_physical))
        ),
        "full_success_count": full_count,
        "target_reached": full_count >= int(target_success_count),
        "catalogs": copy.deepcopy(dict(catalog_report["catalogs"])),
        "published_candidate_ids": copy.deepcopy(
            list(catalog_report["published_candidate_ids"])
        ),
        "robustness": None,
        "fixed_mass_geometry_ablation": True,
        "stop_reason": (
            "target_full_success_count_reached"
            if full_count >= int(target_success_count)
            else "declared_adaptive_diagnostic_exploration_and_refinement_exhausted"
        ),
    }

    final_source = api.source_authenticator(source_root)
    if canonical_sha256(_mapping(final_source, label="adaptive source")) != canonical_sha256(
        _mapping(source, label="adaptive source")
    ):
        raise RuntimeError("immutable adaptive source changed during execution")
    final_manifest = _manifest(config_file, final_source, seed=int(seed))
    if canonical_sha256(final_manifest) != canonical_sha256(manifest):
        raise RuntimeError("adaptive code or inputs changed during execution")
    validate_stage_ledger(workspace)

    result_path = workspace / f"adaptive_event_rescue_result_target_{target_success_count}.json"
    result_stage = f"adaptive_result_target_{target_success_count}"
    existing = _load_committed_report(workspace, result_stage, result_path)
    if existing is None:
        result = _commit_report(
            workspace,
            result_stage,
            result_path,
            result,
            stage_input={
                **report_hashes,
                "catalog_report_sha256": file_sha256(
                    workspace / "catalogs" / f"target_{target_success_count}" / "report.json"
                ),
            },
        )
    elif canonical_sha256(existing) != canonical_sha256(result):
        raise RuntimeError("committed adaptive result changed on resume")
    return result


__all__ = [
    "DEFAULT_SEED",
    "DIAGNOSTIC_CANDIDATE_COUNT",
    "EXPLORATION_CANDIDATE_COUNT",
    "REFINEMENT_CANDIDATE_COUNT",
    "build_contact_preserving_adaptive_event_rescue_manifest",
    "run_contact_preserving_adaptive_event_rescue_campaign",
]
