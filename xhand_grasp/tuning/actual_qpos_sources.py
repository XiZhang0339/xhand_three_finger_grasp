"""Authenticated actual-contact joint-pose source manifests.

The schema-v9 campaign originally consumed a fixed list of 21 schema-v8
poses.  Larger-object campaigns need the same physical evidence contract but
must not depend on that historical cardinality (or silently promote a useful
near miss to success evidence).  This module therefore owns the versioned
source-manifest boundary while retaining the legacy names used by v9.

Version 1 is the historical ``pose_rescue_manifest_schema_version`` payload
with a non-empty ``poses`` list.  Version 2 is an
``actual_qpos_source_manifest_schema_version`` payload with a non-empty
``sources`` list.  Every v2 member binds repository-relative config, result,
and trace files by SHA-256 and is explicitly one of:

``authenticated_grasp_success``
    The result must declare grasp success and its trace must contain a
    continuous, all-gates-true interval lasting at least 250 ms.

``diagnostic_geometry_seed``
    The source may seed geometry but is permanently ineligible as success
    evidence.  Its files are still fully authenticated.  A finite actual
    joint vector is read from any available gate interval, a persisted
    measured vector, or finally the configured nominal contact pose.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ..artifacts import REPO_ROOT, file_sha256
from ..config import ACTIVE_ACTUATORS
from ..grasp_pose import canonical_sha256, controller_id, grasp_pose_id


ACTUAL_QPOS_SOURCE_MANIFEST_SCHEMA_VERSION = 2
AUTHENTICATED_GRASP_SUCCESS = "authenticated_grasp_success"
DIAGNOSTIC_GEOMETRY_SEED = "diagnostic_geometry_seed"
SOURCE_KINDS = frozenset(
    (AUTHENTICATED_GRASP_SUCCESS, DIAGNOSTIC_GEOMETRY_SEED)
)
MINIMUM_STABLE_WINDOW_S = 0.250
LEGACY_SAMPLE_PERIOD_S = 0.001
_DURATION_TOLERANCE_S = 1e-9


@dataclass(frozen=True, slots=True)
class ActualQposSource:
    """One content-addressed actual-qpos geometry source.

    The first thirteen fields intentionally match the former
    ``V8ActualQposSource`` constructor.  New provenance fields have defaults,
    so callers and tests that instantiate that legacy name remain compatible.
    """

    source_index: int
    pose_id: str
    config_path: str
    result_path: str
    trace_path: str
    config_sha256: str
    result_sha256: str
    trace_sha256: str
    stable_window_start_step: int
    stable_window_end_step: int
    stable_window_sample_count: int
    actual_joint_qpos_rad: tuple[float, ...]
    config: dict[str, Any]
    source_kind: str = AUTHENTICATED_GRASP_SUCCESS
    manifest_schema_version: int = 1
    stable_window_duration_s: float | None = None
    gate_evidence_authenticated: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_index, int)
            or isinstance(self.source_index, bool)
            or self.source_index < 0
        ):
            raise ValueError("actual-qpos source_index must be a non-negative integer")
        if not isinstance(self.pose_id, str) or not self.pose_id:
            raise ValueError("actual-qpos source pose_id must be non-empty")
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(f"unknown actual-qpos source_kind: {self.source_kind!r}")
        if self.manifest_schema_version not in (1, 2):
            raise ValueError("actual-qpos source manifest schema must be 1 or 2")
        actual = np.asarray(self.actual_joint_qpos_rad, dtype=np.float64)
        if actual.shape != (len(ACTIVE_ACTUATORS),):
            raise ValueError("actual-qpos source must contain eight active joints")
        if not np.isfinite(actual).all():
            raise ValueError("actual-qpos source must be finite")

        count = self.stable_window_sample_count
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("actual-qpos stable-window sample count is invalid")
        if count == 0:
            if (self.stable_window_start_step, self.stable_window_end_step) != (-1, -1):
                raise ValueError(
                    "an absent actual-qpos stable window must use indices -1, -1"
                )
        elif (
            self.stable_window_start_step < 0
            or self.stable_window_end_step < self.stable_window_start_step
            or self.stable_window_end_step - self.stable_window_start_step + 1
            != count
        ):
            raise ValueError("actual-qpos stable-window indices are inconsistent")

        duration = self.stable_window_duration_s
        if duration is None:
            duration = count * LEGACY_SAMPLE_PERIOD_S
            object.__setattr__(self, "stable_window_duration_s", duration)
        try:
            duration = float(duration)
        except (TypeError, ValueError) as error:
            raise ValueError("actual-qpos stable-window duration must be finite") from error
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("actual-qpos stable-window duration must be finite")

        eligible = self.source_kind == AUTHENTICATED_GRASP_SUCCESS
        if eligible and not self.gate_evidence_authenticated:
            raise ValueError("authenticated grasp source lacks authenticated gate evidence")
        if eligible and duration + _DURATION_TOLERANCE_S < MINIMUM_STABLE_WINDOW_S:
            raise ValueError(
                "authenticated grasp source contact window must last at least 250 ms"
            )

    @property
    def eligible_as_success_evidence(self) -> bool:
        """Whether this source can carry prior grasp-success evidence."""

        return bool(
            self.source_kind == AUTHENTICATED_GRASP_SUCCESS
            and self.gate_evidence_authenticated
            and float(self.stable_window_duration_s or 0.0)
            + _DURATION_TOLERANCE_S
            >= MINIMUM_STABLE_WINDOW_S
        )

    def generator_record(self) -> dict[str, Any]:
        source = copy.deepcopy(self.config)
        source["grasp_pose"] = {
            "nominal_joint_qpos_rad": {
                name: float(self.actual_joint_qpos_rad[index])
                for index, name in enumerate(ACTIVE_ACTUATORS)
            },
            "source_reference": (
                "v8_stable_contact_window_actual_qpos_median"
                if self.manifest_schema_version == 1
                else "stable_contact_window_actual_qpos_median"
            ),
        }
        # This metadata remains outside the source config consumed by the
        # geometry sampler.  In particular, a diagnostic seed can never
        # inherit or manufacture a success classification.
        return {
            "pose_id": self.pose_id,
            "source_index": self.source_index,
            "source_kind": self.source_kind,
            "eligible_as_success_evidence": self.eligible_as_success_evidence,
            "gate_evidence_authenticated": self.gate_evidence_authenticated,
            "config": source,
        }

    def report_record(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "pose_id": self.pose_id,
            "source_kind": self.source_kind,
            "eligible_as_success_evidence": self.eligible_as_success_evidence,
            "gate_evidence_authenticated": self.gate_evidence_authenticated,
            "manifest_schema_version": self.manifest_schema_version,
            "config_path": self.config_path,
            "result_path": self.result_path,
            "trace_path": self.trace_path,
            "config_sha256": self.config_sha256,
            "result_sha256": self.result_sha256,
            "trace_sha256": self.trace_sha256,
            "stable_window_start_step": self.stable_window_start_step,
            "stable_window_end_step": self.stable_window_end_step,
            "stable_window_sample_count": self.stable_window_sample_count,
            "stable_window_duration_s": self.stable_window_duration_s,
            "actual_joint_qpos_rad": {
                name: float(self.actual_joint_qpos_rad[index])
                for index, name in enumerate(ACTIVE_ACTUATORS)
            },
        }


# Public compatibility name retained for v9 imports and constructor users.
V8ActualQposSource = ActualQposSource


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("stable contact mask must be a non-empty vector")
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    if starts.size == 0:
        raise ValueError("source trace has no complete grasp-gate sample")
    lengths = ends - starts + 1
    maximum = int(np.max(lengths))
    selected = int(np.flatnonzero(lengths == maximum)[0])
    return int(starts[selected]), int(ends[selected])


def _scalar_index(trace: Mapping[str, Any], key: str) -> int | None:
    if key not in trace:
        return None
    value = np.asarray(trace[key])
    if value.size != 1:
        raise ValueError(f"{key} must be a scalar trace field")
    return int(value.reshape(-1)[0])


def _all_gate_mask(trace: Mapping[str, Any], trace_path: Path) -> np.ndarray:
    if "grasp_gate" not in trace:
        raise ValueError(f"actual-qpos source trace has no grasp_gate: {trace_path}")
    gate = np.asarray(trace["grasp_gate"], dtype=bool)
    if gate.ndim == 1:
        if gate.size == 0:
            raise ValueError(f"actual-qpos source grasp_gate is empty: {trace_path}")
        return gate
    if gate.ndim != 2 or gate.shape[0] == 0 or gate.shape[1] == 0:
        raise ValueError(f"actual-qpos source grasp_gate shape is invalid: {trace_path}")
    return np.all(gate, axis=1)


def _declared_stable_window(
    trace: Mapping[str, Any], all_gate: np.ndarray
) -> tuple[int, int] | None:
    pairs = (
        ("grasp_stable_window_start_step", "grasp_stable_window_end_step"),
        (
            "controller_grasp_stable_window_start_step",
            "controller_grasp_stable_window_end_step",
        ),
    )
    for start_key, end_key in pairs:
        start = _scalar_index(trace, start_key)
        end = _scalar_index(trace, end_key)
        if start is None and end is None:
            continue
        if start is None or end is None:
            raise ValueError("actual-qpos trace has an incomplete stable-window binding")
        if start < 0 or end < start or end >= all_gate.size:
            raise ValueError("actual-qpos trace stable-window indices are invalid")
        if not np.all(all_gate[start : end + 1]):
            raise ValueError("declared actual-qpos stable window contains a failed gate")
        lock = _scalar_index(trace, "grasp_lock_step")
        if lock is not None and lock >= 0 and lock != end:
            raise ValueError("actual-qpos stable-window end does not match grasp_lock_step")
        return start, end
    return None


def _window_duration_s(
    trace: Mapping[str, Any], start: int, end: int, *, require_time: bool
) -> float:
    count = end - start + 1
    if "time" not in trace:
        if require_time:
            raise ValueError("version-2 actual-qpos trace must contain time")
        return count * LEGACY_SAMPLE_PERIOD_S
    time = np.asarray(trace["time"], dtype=np.float64)
    if time.shape != (np.asarray(trace["grasp_gate"]).shape[0],):
        raise ValueError("actual-qpos trace time and grasp_gate lengths differ")
    if not np.isfinite(time).all():
        raise ValueError("actual-qpos trace time is not finite")
    if time.size < 2:
        if require_time:
            raise ValueError("version-2 actual-qpos trace has no sample interval")
        return count * LEGACY_SAMPLE_PERIOD_S
    deltas = np.diff(time)
    if not np.all(deltas > 0.0):
        raise ValueError("actual-qpos trace time must be strictly increasing")
    sample_period = float(np.median(deltas))
    if not np.allclose(deltas, sample_period, rtol=1e-7, atol=1e-12):
        raise ValueError("actual-qpos trace sample period is not uniform")
    return count * sample_period


def _joint_history(
    trace: Mapping[str, Any], trace_path: Path, sample_count: int
) -> np.ndarray | None:
    if "grasp_pose_actual_joint_qpos_rad" in trace:
        actual = np.asarray(
            trace["grasp_pose_actual_joint_qpos_rad"], dtype=np.float64
        )
        if actual.shape != (sample_count, len(ACTIVE_ACTUATORS)):
            raise ValueError(
                f"actual-qpos measured joint history has an invalid shape: {trace_path}"
            )
        return actual
    if not {"joint_qpos", "actuator_order"}.issubset(trace):
        return None
    joint_qpos = np.asarray(trace["joint_qpos"], dtype=np.float64)
    actuator_order = tuple(str(value) for value in trace["actuator_order"])
    if joint_qpos.ndim != 2 or joint_qpos.shape[0] != sample_count:
        raise ValueError(f"actual-qpos joint history has an invalid shape: {trace_path}")
    missing = set(ACTIVE_ACTUATORS) - set(actuator_order)
    if missing:
        raise ValueError(
            f"actual-qpos source actuator order is incomplete: {sorted(missing)}"
        )
    columns = [actuator_order.index(name) for name in ACTIVE_ACTUATORS]
    if max(columns) >= joint_qpos.shape[1]:
        raise ValueError(f"actual-qpos actuator columns exceed joint history: {trace_path}")
    return joint_qpos[:, columns]


def _configured_nominal(config: Mapping[str, Any]) -> np.ndarray | None:
    grasp_pose = config.get("grasp_pose")
    if not isinstance(grasp_pose, Mapping):
        return None
    nominal = grasp_pose.get("nominal_joint_qpos_rad")
    if not isinstance(nominal, Mapping) or set(nominal) != set(ACTIVE_ACTUATORS):
        return None
    try:
        values = np.asarray([nominal[name] for name in ACTIVE_ACTUATORS], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return values if np.isfinite(values).all() else None


def _extract_actual_qpos_evidence(
    trace_path: Path,
    config: Mapping[str, Any],
    *,
    source_kind: str,
    manifest_schema_version: int,
) -> tuple[int, int, int, float, np.ndarray, bool]:
    with np.load(trace_path, allow_pickle=False) as trace:
        all_gate = _all_gate_mask(trace, trace_path)
        declared = _declared_stable_window(trace, all_gate)
        gate_authenticated = True
        if declared is not None:
            start, end = declared
        else:
            try:
                start, end = _longest_true_run(all_gate)
            except ValueError:
                if source_kind == AUTHENTICATED_GRASP_SUCCESS:
                    raise ValueError(
                        f"authenticated source has no successful grasp gate: {trace_path}"
                    ) from None
                start = end = -1
                gate_authenticated = False
        count = 0 if start < 0 else end - start + 1
        duration = (
            0.0
            if count == 0
            else _window_duration_s(
                trace,
                start,
                end,
                require_time=manifest_schema_version >= 2,
            )
        )
        if duration + _DURATION_TOLERANCE_S < MINIMUM_STABLE_WINDOW_S:
            gate_authenticated = False
            if source_kind == AUTHENTICATED_GRASP_SUCCESS:
                raise ValueError(
                    f"authenticated source has no 250 ms stable contact window: {trace_path}"
                )

        history = _joint_history(trace, trace_path, all_gate.size)
        actual: np.ndarray | None = None
        if history is not None and count > 0:
            actual = np.median(history[start : end + 1], axis=0)
        if actual is None and "grasp_pose_actual_qpos_rad" in trace:
            persisted = np.asarray(
                trace["grasp_pose_actual_qpos_rad"], dtype=np.float64
            )
            if persisted.shape == (len(ACTIVE_ACTUATORS),) and np.isfinite(
                persisted
            ).all():
                actual = persisted
        if actual is None and source_kind == DIAGNOSTIC_GEOMETRY_SEED:
            actual = _configured_nominal(config)
        if (
            actual is None
            or actual.shape != (len(ACTIVE_ACTUATORS),)
            or not np.isfinite(actual).all()
        ):
            raise ValueError(f"actual-qpos source has no finite joint evidence: {trace_path}")

        # A v9 measured vector is a redundant binding to the same declared
        # stable interval.  Refuse a source if it disagrees with recomputation.
        if (
            source_kind == AUTHENTICATED_GRASP_SUCCESS
            and history is not None
            and declared is not None
            and "grasp_pose_actual_qpos_rad" in trace
        ):
            persisted = np.asarray(
                trace["grasp_pose_actual_qpos_rad"], dtype=np.float64
            )
            if persisted.shape != actual.shape or not np.array_equal(persisted, actual):
                raise ValueError(
                    f"persisted actual grasp qpos differs from stable-window median: {trace_path}"
                )
        return start, end, count, duration, actual, gate_authenticated


def _resolve_legacy_path(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def _resolve_v2_member(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"version-2 source {label} must be a repository-relative path")
    raw = PurePosixPath(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"version-2 source {label} must be a repository-relative path")
    path = (REPO_ROOT / Path(*raw.parts)).resolve()
    root = REPO_ROOT.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"version-2 source {label} escapes the repository")
    return path


def _verify_file_hash(path: Path, expected: Any, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"actual-qpos source has no valid {label} SHA-256")
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(f"actual-qpos source {label} hash changed: {path}")
    return observed


def _result_declares_grasp_success(result: Mapping[str, Any]) -> bool:
    summary = result.get("summary", result)
    if not isinstance(summary, Mapping):
        return False
    stage = summary.get("stage_status")
    if isinstance(stage, Mapping) and stage.get("grasp_success") is True:
        return True
    # Older grasp-only results predate stage_status but bind the same physical
    # contract through an explicit acquisition/rescue flag.
    return bool(
        summary.get("grasp_success") is True
        or result.get("grasp_success") is True
        or result.get("rescue_success") is True
    )


def _verify_result_bindings(
    config: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    config_hash: str,
    trace_hash: str,
    result_path: Path,
) -> None:
    artifacts = result.get("artifacts")
    if isinstance(artifacts, Mapping):
        hashes = artifacts.get("sha256")
        if isinstance(hashes, Mapping):
            expected_config = hashes.get("resolved_config")
            if expected_config is not None and expected_config != config_hash:
                raise RuntimeError(
                    f"actual-qpos result config binding changed: {result_path}"
                )
            expected_trace = hashes.get("trace")
            if expected_trace is not None and expected_trace != trace_hash:
                raise RuntimeError(
                    f"actual-qpos result trace binding changed: {result_path}"
                )
    semantic = canonical_sha256(config)
    if result.get("candidate_sha256") is not None and result["candidate_sha256"] != semantic:
        raise RuntimeError(f"actual-qpos result semantic config changed: {result_path}")
    if result.get("grasp_pose_id") is not None and result[
        "grasp_pose_id"
    ] != grasp_pose_id(config):
        raise RuntimeError(f"actual-qpos result grasp_pose_id changed: {result_path}")
    if result.get("controller_id") is not None and result[
        "controller_id"
    ] != controller_id(config):
        raise RuntimeError(f"actual-qpos result controller_id changed: {result_path}")


def _load_v1_source(source_index: int, raw: Mapping[str, Any]) -> ActualQposSource:
    config_path = _resolve_legacy_path(raw["source_config"])
    result_path = _resolve_legacy_path(raw["source_result"])
    config_hash = _verify_file_hash(
        config_path, raw["source_config_file_sha256"], "config"
    )
    result_hash = _verify_file_hash(
        result_path, raw["source_result_sha256"], "result"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if canonical_sha256(config) != str(raw["source_config_sha256"]):
        raise RuntimeError(f"legacy source config semantic hash changed: {config_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"legacy source result has no artifacts: {result_path}")
    trace_path = (result_path.parent / str(artifacts.get("trace", "trace.npz"))).resolve()
    expected_trace_hash = (
        artifacts.get("sha256", {}).get("trace")
        if isinstance(artifacts.get("sha256"), Mapping)
        else None
    )
    trace_hash = _verify_file_hash(trace_path, expected_trace_hash, "trace")
    start, end, count, duration, actual, authenticated = _extract_actual_qpos_evidence(
        trace_path,
        config,
        source_kind=AUTHENTICATED_GRASP_SUCCESS,
        manifest_schema_version=1,
    )
    return ActualQposSource(
        source_index=source_index,
        pose_id=str(raw["pose_id"]),
        config_path=str(config_path),
        result_path=str(result_path),
        trace_path=str(trace_path),
        config_sha256=config_hash,
        result_sha256=result_hash,
        trace_sha256=trace_hash,
        stable_window_start_step=start,
        stable_window_end_step=end,
        stable_window_sample_count=count,
        actual_joint_qpos_rad=tuple(float(value) for value in actual),
        config=copy.deepcopy(config),
        source_kind=AUTHENTICATED_GRASP_SUCCESS,
        manifest_schema_version=1,
        stable_window_duration_s=duration,
        gate_evidence_authenticated=authenticated,
    )


def _load_v2_source(source_index: int, raw: Mapping[str, Any]) -> ActualQposSource:
    kind = str(raw.get("source_kind", ""))
    if kind not in SOURCE_KINDS:
        raise ValueError(f"unknown actual-qpos source_kind: {kind!r}")
    hashes = raw.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("version-2 actual-qpos source must contain sha256 bindings")
    required_hashes = {"config", "result", "trace", "config_semantic"}
    if not required_hashes.issubset(hashes):
        missing = sorted(required_hashes - set(hashes))
        raise ValueError(f"version-2 actual-qpos source lacks hashes: {missing}")
    config_path = _resolve_v2_member(raw.get("config_path"), "config_path")
    result_path = _resolve_v2_member(raw.get("result_path"), "result_path")
    trace_path = _resolve_v2_member(raw.get("trace_path"), "trace_path")
    config_hash = _verify_file_hash(config_path, hashes["config"], "config")
    result_hash = _verify_file_hash(result_path, hashes["result"], "result")
    trace_hash = _verify_file_hash(trace_path, hashes["trace"], "trace")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    semantic = canonical_sha256(config)
    if semantic != str(hashes["config_semantic"]):
        raise RuntimeError(f"actual-qpos source semantic config changed: {config_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _verify_result_bindings(
        config,
        result,
        config_hash=config_hash,
        trace_hash=trace_hash,
        result_path=result_path,
    )
    if kind == AUTHENTICATED_GRASP_SUCCESS and not _result_declares_grasp_success(result):
        raise ValueError(
            f"authenticated actual-qpos source result is not grasp-success: {result_path}"
        )
    start, end, count, duration, actual, authenticated = _extract_actual_qpos_evidence(
        trace_path,
        config,
        source_kind=kind,
        manifest_schema_version=2,
    )
    expected_pose_id = grasp_pose_id(config)
    declared_pose_id = raw.get("pose_id")
    if declared_pose_id is not None and str(declared_pose_id) != expected_pose_id:
        raise RuntimeError(f"actual-qpos source pose_id changed: {config_path}")
    return ActualQposSource(
        source_index=source_index,
        pose_id=expected_pose_id,
        config_path=str(config_path),
        result_path=str(result_path),
        trace_path=str(trace_path),
        config_sha256=config_hash,
        result_sha256=result_hash,
        trace_sha256=trace_hash,
        stable_window_start_step=start,
        stable_window_end_step=end,
        stable_window_sample_count=count,
        actual_joint_qpos_rad=tuple(float(value) for value in actual),
        config=copy.deepcopy(config),
        source_kind=kind,
        manifest_schema_version=2,
        stable_window_duration_s=duration,
        gate_evidence_authenticated=authenticated,
    )


def resolve_registered_source_manifest(config: Mapping[str, Any]) -> Path:
    campaign = config.get("actual_contact_grasp_pose_campaign")
    if not isinstance(campaign, Mapping):
        raise ValueError("config has no actual-contact campaign")
    raw = campaign.get("source_pose_manifest")
    if not isinstance(raw, str) or not raw:
        raise ValueError("actual-contact campaign has no source_pose_manifest")
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def load_actual_qpos_sources(
    config: Mapping[str, Any],
    *,
    source_manifest_path: str | Path | None = None,
) -> tuple[ActualQposSource, ...]:
    """Load a non-empty, versioned source manifest with full authentication."""

    manifest_path = (
        resolve_registered_source_manifest(config)
        if source_manifest_path is None
        else Path(source_manifest_path).expanduser().resolve()
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("actual-qpos source manifest must be a JSON object")
    if "actual_qpos_source_manifest_schema_version" in payload:
        version = payload["actual_qpos_source_manifest_schema_version"]
        if version != ACTUAL_QPOS_SOURCE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported actual-qpos source manifest version: {version}")
        values = payload.get("sources")
        loader = _load_v2_source
    else:
        version = payload.get("pose_rescue_manifest_schema_version")
        if version != 1:
            raise ValueError("unrecognized actual-qpos source manifest schema")
        values = payload.get("poses")
        loader = _load_v1_source
    if not isinstance(values, list) or not values:
        raise ValueError("actual-qpos source manifest must contain a non-empty source list")
    records: list[ActualQposSource] = []
    for source_index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise ValueError("actual-qpos source manifest records must be mappings")
        records.append(loader(source_index, raw))
    if len({record.pose_id for record in records}) != len(records):
        raise ValueError("actual-qpos source manifest contains duplicate pose IDs")
    return tuple(records)


def load_v8_actual_qpos_sources(
    config: Mapping[str, Any],
    *,
    source_manifest_path: str | Path | None = None,
) -> tuple[ActualQposSource, ...]:
    """Compatibility wrapper for callers using the former v8-specific name."""

    return load_actual_qpos_sources(
        config, source_manifest_path=source_manifest_path
    )


__all__ = [
    "ACTUAL_QPOS_SOURCE_MANIFEST_SCHEMA_VERSION",
    "AUTHENTICATED_GRASP_SUCCESS",
    "DIAGNOSTIC_GEOMETRY_SEED",
    "SOURCE_KINDS",
    "ActualQposSource",
    "V8ActualQposSource",
    "load_actual_qpos_sources",
    "load_v8_actual_qpos_sources",
    "resolve_registered_source_manifest",
]
