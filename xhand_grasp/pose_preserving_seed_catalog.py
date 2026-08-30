"""Publish authenticated schema-v6 multi-seed grasp-acquisition catalogs.

The hand-pose retuning campaign writes one directory per source trajectory.
This module turns those independent runs into one Viewer-compatible catalog
without rerunning or reinterpreting them.  In particular, a source is only
advertised through an alias after its complete trace has been evaluated again
and the schema-v6 *sticky* pose-preservation contract passes through the grasp
acquisition sample (inclusive).

Expected source layout::

    campaign_runs/
      source_117060/
        resolved_config.json       # ``config.json`` is also accepted
        result.json                # direct summary or {"summary": summary}
        trace.npz
        artifact_hashes.json       # optional when result declares hashes
        trajectory.mp4             # optional

The dynamic tuner also writes ``best_config.json``, ``best_result.json`` and
``best_trace.npz`` beside ``source_result.json``.  That native layout is
accepted directly; callers do not need to rename or duplicate its artifacts.

``artifact_hashes.json`` is either a SHA-256 mapping or ``{"sha256": ...}``.
The source config and trace must have authenticated digests.  A result may
instead carry the usual ``artifacts.sha256`` mapping.  Publication is atomic;
the source directory is never modified.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import file_sha256, write_json
from .config import validate_config
from .controller import grasp_gate_order, quaternion_drift_deg
from .evaluation import evaluate_trace
from .scene import build_model
from .trajectory import _phase_steps
from .trajectory_catalog import object_physical_parameters


POSE_PRESERVING_SEED_CATALOG_SCHEMA_VERSION = 1
POSE_PRESERVING_GRASP_TRACE_SCHEMA_VERSION = 1
VALIDATION_SCOPE = "grasp_acquisition_pose_preserved"
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

_CONFIG_NAMES = ("resolved_config.json", "config.json")
_STANDARD_REQUIRED_NAMES = ("result.json", "trace.npz")
_DYNAMIC_SOURCE_RESULT_NAME = "source_result.json"
_DYNAMIC_BEST_NAMES = ("best_config.json", "best_result.json", "best_trace.npz")
_HASH_MANIFEST_NAME = "artifact_hashes.json"

# These are structural/numerical properties of the run, not manipulation or
# lifting goals.  Stage grasp_success already covers the schema-v3-v6 grasp
# gates; naming the integrity checks here makes the publication boundary easy
# to audit and prevents a future evaluator change from weakening it silently.
REQUIRED_ACQUISITION_CHECKS = (
    "stable_grasp_acquired",
    "grasp_acquisition_event_consistent",
    "grasp_gate_contiguous_window",
    "grasp_latch_remains_set",
    "grasp_gate_counter_is_consistent",
    "target_face_evidence_matches_raw_trace",
    "grasp_support_retained",
    "grasp_pose_is_stable",
    "no_early_object_lift",
    "v4_alignment_trace_matches_raw_contacts",
    "grasp_contact_height_alignment_contiguous",
    "v5_contact_exclusion_gate_matches_raw_trace",
    "v5_pad_fraction_trace_matches_raw_forces",
    "v5_pose_and_thumb_bend_trace_matches_raw_state",
    "v5_thumb_bend_command_matches_control_protocol",
    "v6_pose_preservation_trace_matches_raw_state",
    "v6_close_profile_trace_matches_config",
    "v6_first_distal_contact_steps_match_raw_trace",
    "v6_initial_joint_state_matches_pregrasp_config",
    "object_pose_preserved_until_grasp_acquisition",
    "support_retained_until_grasp_acquisition",
    "no_hand_cube_contact_during_settle",
    "hand_root_is_structurally_fixed",
    "hand_root_pose_did_not_move",
    "inactive_controls_are_exactly_zero",
    "inactive_joints_remain_open",
    "all_state_is_finite",
    "joint_limits_respected",
    "penetration_within_limit",
    "runtime_contact_friction_matches_cube",
    "no_palm_ring_or_pinky_contact",
)

_PREFIX_SCALAR_EVENTS = (
    "grasp_acquisition_step",
    "manipulation_start_step",
    "manipulation_end_step",
    "termination_step",
)
_STATIC_TRACE_FIELDS = (
    "face_order",
    "finger_order",
    "actuator_order",
    "grasp_gate_order",
)


@dataclass(frozen=True)
class _SourceRun:
    source_id: str
    directory: Path
    config_path: Path
    result_path: Path
    source_result_path: Path | None
    trace_path: Path
    grasp_trace_path: Path | None
    video_path: Path | None
    declared_hashes: dict[str, str]
    config: dict[str, Any]
    reported_summary: dict[str, Any]
    source_metadata: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _safe_source_id(value: object, *, directory: Path) -> str:
    source_id = str(value)
    if _SAFE_NAME.fullmatch(source_id) is None:
        raise ValueError(
            f"source id {source_id!r} from {directory} is not a safe catalog name"
        )
    return source_id


def _summary_from_result(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    candidate = payload.get("summary", payload)
    if not isinstance(candidate, Mapping):
        raise ValueError(f"result summary must be an object: {path}")
    summary = copy.deepcopy(dict(candidate))
    if not isinstance(summary.get("checks"), Mapping):
        raise ValueError(f"result summary is missing checks: {path}")
    if not isinstance(summary.get("stage_status"), Mapping):
        raise ValueError(f"result summary is missing stage_status: {path}")
    return summary


def _normalize_sha256(value: object, *, field: str, path: Path) -> str:
    digest = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"invalid SHA-256 for {field} in {path}")
    return digest


def _declared_source_hashes(
    directory: Path, result_payload: Mapping[str, Any]
) -> dict[str, str]:
    manifest_path = directory / _HASH_MANIFEST_NAME
    raw: object | None = None
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        raw = manifest.get("sha256", manifest)
        source_path = manifest_path
    else:
        artifacts = result_payload.get("artifacts")
        raw = artifacts.get("sha256") if isinstance(artifacts, Mapping) else None
        source_path = directory / "result.json"
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"source {directory.name!r} must authenticate config and trace with "
            f"{_HASH_MANIFEST_NAME} or result.artifacts.sha256"
        )
    hashes = {
        str(field): _normalize_sha256(value, field=str(field), path=source_path)
        for field, value in raw.items()
    }
    config_key = "resolved_config" if "resolved_config" in hashes else "config"
    missing = [
        logical
        for logical, present in (
            ("resolved_config", config_key in hashes),
            ("trace", "trace" in hashes),
        )
        if not present
    ]
    if missing:
        raise ValueError(
            f"source {directory.name!r} has no authenticated " + ", ".join(missing)
        )
    if config_key == "config":
        hashes["resolved_config"] = hashes.pop("config")
    return hashes


def _artifact_path_from_result(
    directory: Path,
    result_payload: Mapping[str, Any],
    logical_name: str,
    fallback_name: str,
) -> Path:
    artifacts = result_payload.get("artifacts")
    raw = artifacts.get(logical_name) if isinstance(artifacts, Mapping) else None
    relative = raw if isinstance(raw, str) and raw else fallback_name
    candidate = (directory / relative).resolve()
    try:
        candidate.relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError(
            f"source artifact {logical_name} escapes {directory}: {relative}"
        ) from exc
    return candidate


def _verify_digest(path: Path, expected: str, field: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"source {field} does not exist: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"source SHA-256 mismatch for {field}: expected {expected}, "
            f"observed {observed}: {path}"
        )


def _resolve_source_layout(
    directory: Path,
) -> tuple[Path, Path, Path, Path | None, dict[str, Any]] | None:
    """Resolve either a standalone run or the dynamic tuner's best-run view."""

    source_result_path = directory / _DYNAMIC_SOURCE_RESULT_NAME
    dynamic_paths = tuple(directory / name for name in _DYNAMIC_BEST_NAMES)
    dynamic_evidence = source_result_path.exists() or any(
        path.exists() for path in dynamic_paths
    )
    if dynamic_evidence:
        if not source_result_path.is_file() or not all(
            path.is_file() for path in dynamic_paths
        ):
            raise ValueError(
                f"partial dynamic source run {directory}: expected "
                f"{_DYNAMIC_SOURCE_RESULT_NAME} and {', '.join(_DYNAMIC_BEST_NAMES)}"
            )
        source_payload = _load_json(source_result_path)
        return (
            dynamic_paths[0],
            dynamic_paths[1],
            dynamic_paths[2],
            source_result_path,
            source_payload,
        )

    result_path = directory / "result.json"
    trace_path = directory / "trace.npz"
    config_candidates = [directory / name for name in _CONFIG_NAMES]
    config_path = next((path for path in config_candidates if path.is_file()), None)
    evidence = [
        result_path.exists(),
        trace_path.exists(),
        any(path.exists() for path in config_candidates),
    ]
    if not any(evidence):
        return None
    if config_path is None or not all(evidence[:2]):
        raise ValueError(
            f"partial source run {directory}: expected one config plus "
            f"{', '.join(_STANDARD_REQUIRED_NAMES)}"
        )
    return config_path, result_path, trace_path, None, {}


def _verify_dynamic_source_binding(
    directory: Path,
    source_payload: Mapping[str, Any],
    *,
    config_path: Path,
    result_path: Path,
    trace_path: Path,
) -> None:
    artifacts = source_payload.get("best_artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"dynamic source has no best_artifacts: {directory}")
    expected_paths = {
        "resolved_config": config_path,
        "result": result_path,
        "trace": trace_path,
    }
    hashes = artifacts.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"dynamic source has no best_artifacts.sha256: {directory}")
    for field, path in expected_paths.items():
        raw_path = artifacts.get(field)
        if raw_path != path.name:
            raise ValueError(
                f"dynamic source best_artifacts.{field} does not identify {path.name}"
            )
        expected = _normalize_sha256(
            hashes.get(field),
            field=f"best_artifacts.{field}",
            path=directory / _DYNAMIC_SOURCE_RESULT_NAME,
        )
        _verify_digest(path, expected, f"best_artifacts.{field}")


def _discover_sources(
    source_dir: Path, *, expected_count: int | None
) -> tuple[list[_SourceRun], str]:
    if not source_dir.is_dir():
        raise NotADirectoryError(f"source run directory does not exist: {source_dir}")
    child_dirs = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    runs: list[_SourceRun] = []
    experiment_id: str | None = None
    seen_ids: set[str] = set()
    for directory in child_dirs:
        layout = _resolve_source_layout(directory)
        if layout is None:
            continue
        (
            config_path,
            result_path,
            trace_path,
            source_result_path,
            source_payload,
        ) = layout

        result_payload = _load_json(result_path)
        if source_result_path is not None:
            _verify_dynamic_source_binding(
                directory,
                source_payload,
                config_path=config_path,
                result_path=result_path,
                trace_path=trace_path,
            )
        config = _load_json(config_path)
        validate_config(config)
        if int(config.get("schema_version", 0)) != 6:
            raise ValueError(f"source run must use schema version 6: {directory}")
        current_experiment = str(config.get("experiment_id", ""))
        if not current_experiment:
            raise ValueError(f"source config has no experiment_id: {config_path}")
        if experiment_id is None:
            experiment_id = current_experiment
        elif current_experiment != experiment_id:
            raise ValueError("all source runs must belong to one experiment")

        if source_result_path is not None:
            source_value = source_payload.get("source_candidate_id", directory.name)
        else:
            source_value = result_payload.get(
                "source_id", result_payload.get("label", directory.name)
            )
        source_id = _safe_source_id(source_value, directory=directory)
        if source_id in seen_ids:
            raise ValueError(f"duplicate source id: {source_id}")
        seen_ids.add(source_id)

        hashes = _declared_source_hashes(directory, result_payload)
        config_artifact = _artifact_path_from_result(
            directory, result_payload, "resolved_config", config_path.name
        )
        if config_artifact != config_path.resolve():
            raise ValueError(
                "result resolved_config does not identify the loaded config: "
                f"{directory}"
            )
        trace_artifact = _artifact_path_from_result(
            directory, result_payload, "trace", trace_path.name
        )
        if trace_artifact != trace_path.resolve():
            raise ValueError(
                f"result trace does not identify the loaded trace: {directory}"
            )
        _verify_digest(config_path, hashes["resolved_config"], "resolved_config")
        _verify_digest(trace_path, hashes["trace"], "trace")

        grasp_trace = directory / "grasp_trace.npz"
        video = directory / "trajectory.mp4"
        for logical, path in (("grasp_trace", grasp_trace), ("video", video)):
            if path.is_file() and logical in hashes:
                _verify_digest(path, hashes[logical], logical)
            elif logical in hashes and not path.is_file():
                raise FileNotFoundError(
                    f"source declares {logical} but file does not exist: {path}"
                )
        runs.append(
            _SourceRun(
                source_id=source_id,
                directory=directory.resolve(),
                config_path=config_path.resolve(),
                result_path=result_path.resolve(),
                source_result_path=(
                    source_result_path.resolve()
                    if source_result_path is not None
                    else None
                ),
                trace_path=trace_path.resolve(),
                grasp_trace_path=(
                    grasp_trace.resolve() if grasp_trace.is_file() else None
                ),
                video_path=video.resolve() if video.is_file() else None,
                declared_hashes=hashes,
                config=config,
                reported_summary=_summary_from_result(result_payload, result_path),
                source_metadata=copy.deepcopy(source_payload),
            )
        )

    if expected_count is not None and len(runs) != int(expected_count):
        raise ValueError(
            f"expected {int(expected_count)} source runs, discovered {len(runs)}"
        )
    if not runs:
        raise ValueError(f"no complete source runs found in {source_dir}")
    assert experiment_id is not None
    return runs, experiment_id


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            traces = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load authenticated trace: {path}") from exc
    time = np.asarray(traces.get("time"))
    if time.ndim != 1 or time.size == 0 or not np.isfinite(time).all():
        raise ValueError(f"trace requires a non-empty finite time axis: {path}")
    if time.size > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError(f"trace time must be strictly increasing: {path}")
    return traces


def _reported_summary_agrees(
    reported: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    disagreements: list[str] = []
    reported_stage = reported.get("stage_status")
    recomputed_stage = recomputed.get("stage_status")
    for key in ("grasp_success", "manipulation_success", "full_success"):
        left = reported_stage.get(key) if isinstance(reported_stage, Mapping) else None
        right = (
            recomputed_stage.get(key) if isinstance(recomputed_stage, Mapping) else None
        )
        if left is None or bool(left) != bool(right):
            disagreements.append(f"reported_stage_status.{key}")

    reported_metrics = reported.get("metrics")
    recomputed_metrics = recomputed.get("metrics")
    for key in _PREFIX_SCALAR_EVENTS:
        left = (
            reported_metrics.get(key)
            if isinstance(reported_metrics, Mapping)
            else None
        )
        right = (
            recomputed_metrics.get(key)
            if isinstance(recomputed_metrics, Mapping)
            else None
        )
        if left is None or right is None or int(left) != int(right):
            disagreements.append(f"reported_metrics.{key}")
    return not disagreements, disagreements


def _raw_prefix_validation(
    config: Mapping[str, Any], traces: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Independently audit the inclusive acquisition prefix and sticky bits."""

    required = (
        "time",
        "control_state",
        "grasp_gate_order",
        "grasp_gate",
        "grasp_gate_consecutive_steps",
        "grasp_acquired",
        "manipulation_progress",
        "grasp_acquisition_step",
        "cube_pos",
        "cube_quat",
        "initial_cube_pos_m",
        "initial_cube_quat",
        "support_contact",
        "hand_cube_contact",
        "pregrasp_pose_preserved_latched",
        "pregrasp_support_retained_latched",
        "settle_hand_contact_free_latched",
    )
    missing = [name for name in required if name not in traces]
    if missing:
        raise ValueError("schema-v6 trace is missing: " + ", ".join(missing))

    time = np.asarray(traces["time"], dtype=np.float64)
    total_steps = len(time)
    acquisition = int(np.asarray(traces["grasp_acquisition_step"]).reshape(()))
    event_in_range = 0 <= acquisition < total_steps
    states = np.asarray(traces["control_state"]).astype(str)
    if states.shape != (total_steps,):
        raise ValueError("control_state does not share the trace time axis")
    gate_order = tuple(str(value) for value in np.asarray(traces["grasp_gate_order"]))
    expected_order = grasp_gate_order(6)
    canonical_gate_axis = gate_order == expected_order
    gates = np.asarray(traces["grasp_gate"], dtype=bool)
    if gates.shape != (total_steps, len(expected_order)):
        raise ValueError("schema-v6 grasp_gate has a non-canonical shape")

    stable_steps = int(
        round(
            float(config["control_protocol"]["stable_window_s"])
            / float(np.median(np.diff(time)))
        )
    )
    window_start = acquisition - stable_steps + 1
    stable_window_in_range = bool(
        event_in_range and stable_steps > 0 and window_start >= 0
    )
    gate_window_passed = bool(
        stable_window_in_range
        and canonical_gate_axis
        and np.all(gates[window_start : acquisition + 1])
    )
    counters = np.asarray(traces["grasp_gate_consecutive_steps"], dtype=np.int64)
    acquired = np.asarray(traces["grasp_acquired"], dtype=bool)
    progress = np.asarray(traces["manipulation_progress"], dtype=np.float64)
    if counters.shape != (total_steps,) or acquired.shape != (total_steps,):
        raise ValueError("grasp counter/latch does not share the trace time axis")
    if progress.shape != (total_steps,):
        raise ValueError("manipulation_progress does not share the trace time axis")
    acquisition_event_consistent = bool(
        event_in_range
        and states[acquisition] == "VERIFY"
        and int(counters[acquisition]) == stable_steps
        and bool(acquired[acquisition])
        and not np.any(acquired[:acquisition])
        and np.all(acquired[acquisition:])
        and np.all(progress[: acquisition + 1] == 0.0)
    )

    initial_pos = np.asarray(traces["initial_cube_pos_m"], dtype=np.float64)
    initial_quat = np.asarray(traces["initial_cube_quat"], dtype=np.float64)
    positions = np.asarray(traces["cube_pos"], dtype=np.float64)
    quaternions = np.asarray(traces["cube_quat"], dtype=np.float64)
    if initial_pos.shape != (3,) or initial_quat.shape != (4,):
        raise ValueError("schema-v6 immutable cube pose has a non-canonical shape")
    if positions.shape != (total_steps, 3) or quaternions.shape != (total_steps, 4):
        raise ValueError("cube pose does not share the trace time axis")
    scope_stop = acquisition + 1 if event_in_range else total_steps
    translation = np.linalg.norm(positions[:scope_stop] - initial_pos, axis=1)
    orientation = np.asarray(
        [
            quaternion_drift_deg(initial_quat, value)
            for value in quaternions[:scope_stop]
        ],
        dtype=np.float64,
    )
    max_translation = float(np.max(translation))
    max_orientation = float(np.max(orientation))
    translation_limit = float(config["pose_preservation"]["max_translation_m"])
    orientation_limit = float(
        config["pose_preservation"]["max_orientation_drift_deg"]
    )
    raw_pose_preserved = bool(
        event_in_range
        and max_translation <= translation_limit + 1e-12
        and max_orientation <= orientation_limit + 1e-12
    )

    support = np.asarray(traces["support_contact"], dtype=bool)
    hand_contact = np.asarray(traces["hand_cube_contact"], dtype=bool)
    if support.shape != (total_steps,) or hand_contact.shape != (total_steps,):
        raise ValueError("support/hand contact does not share the trace time axis")
    raw_support_retained = bool(event_in_range and np.all(support[:scope_stop]))
    settle = states[:scope_stop] == "SETTLE"
    raw_settle_contact_free = bool(
        event_in_range and not np.any(hand_contact[:scope_stop][settle])
    )

    combined = np.asarray(traces["pregrasp_pose_preserved_latched"], dtype=bool)
    support_latch = np.asarray(
        traces["pregrasp_support_retained_latched"], dtype=bool
    )
    settle_latch = np.asarray(
        traces["settle_hand_contact_free_latched"], dtype=bool
    )
    for name, values in (
        ("pregrasp_pose_preserved_latched", combined),
        ("pregrasp_support_retained_latched", support_latch),
        ("settle_hand_contact_free_latched", settle_latch),
    ):
        if values.shape != (total_steps,):
            raise ValueError(f"{name} does not share the trace time axis")
    persisted_sticky_terminal_passed = bool(
        event_in_range
        and combined[acquisition]
        and support_latch[acquisition]
        and settle_latch[acquisition]
    )

    checks = {
        "event_in_range": event_in_range,
        "canonical_schema_v6_gate_axis": canonical_gate_axis,
        "continuous_full_gate_window": gate_window_passed,
        "acquisition_event_consistent": acquisition_event_consistent,
        "raw_pose_preserved_inclusive": raw_pose_preserved,
        "raw_support_retained_inclusive": raw_support_retained,
        "raw_settle_hand_contact_free": raw_settle_contact_free,
        "persisted_sticky_terminal_passed": persisted_sticky_terminal_passed,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "grasp_acquisition_step": acquisition,
        "grasp_acquisition_time_s": (
            float(time[acquisition]) if event_in_range else None
        ),
        "stable_window_s": float(config["control_protocol"]["stable_window_s"]),
        "stable_window_steps": stable_steps,
        "max_translation_before_acquisition_m": max_translation,
        "translation_limit_m": translation_limit,
        "max_orientation_before_acquisition_deg": max_orientation,
        "orientation_limit_deg": orientation_limit,
    }


def _recompute_summary(
    config: dict[str, Any], traces: dict[str, np.ndarray]
) -> dict[str, Any]:
    model, info = build_model(config)
    return evaluate_trace(model, info, config, _phase_steps(model, config), traces)


def _validation(
    run: _SourceRun,
    traces: dict[str, np.ndarray],
    recomputed: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _raw_prefix_validation(run.config, traces)
    checks = recomputed.get("checks")
    stage = recomputed.get("stage_status")
    recomputed_checks = {
        name: bool(checks.get(name, False)) if isinstance(checks, Mapping) else False
        for name in REQUIRED_ACQUISITION_CHECKS
    }
    recomputed_grasp_success = bool(
        isinstance(stage, Mapping) and stage.get("grasp_success", False)
    )
    summary_agrees, disagreements = _reported_summary_agrees(
        run.reported_summary, recomputed
    )
    passed = bool(
        raw["passed"]
        and recomputed_grasp_success
        and all(recomputed_checks.values())
        and summary_agrees
    )
    failed = list(raw["failed_checks"])
    failed.extend(name for name, value in recomputed_checks.items() if not value)
    failed.extend(disagreements)
    if not recomputed_grasp_success:
        failed.append("recomputed_stage_status.grasp_success")
    return {
        "validation_scope": VALIDATION_SCOPE,
        "passed": passed,
        "reported_grasp_success": bool(
            run.reported_summary["stage_status"].get("grasp_success", False)
        ),
        "recomputed_grasp_success": recomputed_grasp_success,
        "reported_summary_agrees": summary_agrees,
        "reported_summary_disagreements": disagreements,
        "raw_prefix_validation": raw,
        "required_recomputed_checks": recomputed_checks,
        "failed_checks": list(dict.fromkeys(failed)),
    }


def _grasp_prefix_payload(
    traces: Mapping[str, np.ndarray], validation: Mapping[str, Any], trace_hash: str
) -> dict[str, np.ndarray] | None:
    raw = validation["raw_prefix_validation"]
    acquisition = int(raw["grasp_acquisition_step"])
    total_steps = len(np.asarray(traces["time"]))
    if not 0 <= acquisition < total_steps:
        return None
    stop = acquisition + 1
    payload: dict[str, np.ndarray] = {}
    for name, source in traces.items():
        values = np.asarray(source)
        if name in _PREFIX_SCALAR_EVENTS:
            if name == "grasp_acquisition_step":
                payload[name] = np.asarray(acquisition, dtype=np.int64)
            continue
        if name == "video_frame_steps":
            frames = np.asarray(values, dtype=np.int64)
            payload[name] = frames[frames <= acquisition].copy()
        elif name in _STATIC_TRACE_FIELDS or values.shape == ():
            payload[name] = np.array(values, copy=True)
        elif values.shape[:1] == (total_steps,):
            payload[name] = np.array(values[:stop], copy=True)
        else:
            payload[name] = np.array(values, copy=True)
    payload.update(
        {
            "pose_preserving_grasp_trace_schema_version": np.asarray(
                POSE_PRESERVING_GRASP_TRACE_SCHEMA_VERSION, dtype=np.int64
            ),
            "validation_scope": np.asarray(VALIDATION_SCOPE, dtype=np.str_),
            "source_total_steps": np.asarray(total_steps, dtype=np.int64),
            "segment_start_step": np.asarray(0, dtype=np.int64),
            "segment_stop_step_inclusive": np.asarray(acquisition, dtype=np.int64),
            "source_trace_sha256": np.asarray(trace_hash, dtype=np.str_),
        }
    )
    return payload


def _best_sort_key(entry: Mapping[str, Any]) -> tuple[float, float, str]:
    raw = entry["grasp_validation"]["raw_prefix_validation"]
    translation_margin = 1.0 - float(
        raw["max_translation_before_acquisition_m"]
    ) / float(raw["translation_limit_m"])
    orientation_margin = 1.0 - float(
        raw["max_orientation_before_acquisition_deg"]
    ) / float(raw["orientation_limit_deg"])
    return (
        -min(translation_margin, orientation_margin),
        -translation_margin,
        str(entry["source_id"]),
    )


def export_pose_preserving_seed_catalog(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_count: int | None = 6,
    best_source_id: str | None = None,
) -> dict[str, Any]:
    """Authenticate, reevaluate and atomically publish all schema-v6 sources."""

    source = Path(source_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    if destination == source:
        raise ValueError("output_dir must not replace source_dir")
    runs, experiment_id = _discover_sources(source, expected_count=expected_count)
    destination.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}.staging."
    ) as staging_name:
        staging = Path(staging_name)
        for run in runs:
            traces = _load_trace(run.trace_path)
            recomputed = _recompute_summary(run.config, traces)
            validation = _validation(run, traces, recomputed)
            trajectory_id = f"pose_preserving_{run.source_id}"
            member = staging / trajectory_id
            member.mkdir()
            config_path = member / "resolved_config.json"
            trace_path = member / "trace.npz"
            shutil.copy2(run.config_path, config_path)
            shutil.copy2(run.trace_path, trace_path)
            trace_hash = file_sha256(trace_path)

            grasp_trace_path: Path | None = None
            prefix = _grasp_prefix_payload(traces, validation, trace_hash)
            if prefix is not None:
                grasp_trace_path = member / "grasp_trace.npz"
                np.savez_compressed(grasp_trace_path, **prefix)
            video_path: Path | None = None
            if run.video_path is not None:
                video_path = member / "trajectory.mp4"
                shutil.copy2(run.video_path, video_path)

            stage = recomputed.get("stage_status", {})
            classification = (
                "validated_pose_preserving_grasp_acquisition"
                if validation["passed"]
                else "failed_pose_preserving_grasp_acquisition"
            )
            physical = object_physical_parameters(run.config)
            result_payload = {
                "pose_preserving_seed_catalog_schema_version": (
                    POSE_PRESERVING_SEED_CATALOG_SCHEMA_VERSION
                ),
                "validation_scope": VALIDATION_SCOPE,
                "trajectory_id": trajectory_id,
                "source_id": run.source_id,
                "classification": classification,
                "grasp_validation": validation,
                "reported_summary": run.reported_summary,
                "recomputed_summary": recomputed,
                "config": run.config,
                "physical_parameters": physical,
                "source_provenance": {
                    "directory": str(run.directory),
                    "resolved_config_sha256": file_sha256(run.config_path),
                    "result_sha256": file_sha256(run.result_path),
                    "source_result_sha256": (
                        file_sha256(run.source_result_path)
                        if run.source_result_path is not None
                        else None
                    ),
                    "trace_sha256": file_sha256(run.trace_path),
                    "declared_sha256": run.declared_hashes,
                    "source_metadata": run.source_metadata,
                },
            }
            result_path = member / "result.json"
            write_json(result_path, result_payload)
            hashes = {
                "resolved_config": file_sha256(config_path),
                "result": file_sha256(result_path),
                "trace": trace_hash,
            }
            if grasp_trace_path is not None:
                hashes["grasp_trace"] = file_sha256(grasp_trace_path)
            if video_path is not None:
                hashes["video"] = file_sha256(video_path)

            metrics = recomputed.get("metrics", {})
            pose_metrics = (
                metrics.get("pose_preservation", {})
                if isinstance(metrics, Mapping)
                else {}
            )
            entry = {
                "trajectory_id": trajectory_id,
                "source_id": run.source_id,
                "label": run.source_id,
                "aliases": [],
                "classification": classification,
                "rerun_grasp_success": bool(validation["passed"]),
                "stage_status": copy.deepcopy(dict(stage))
                if isinstance(stage, Mapping)
                else {},
                "failed_checks": copy.deepcopy(validation["failed_checks"]),
                "grasp_validation": copy.deepcopy(validation),
                "physical_parameters": physical,
                "grasp_parameters": {
                    "thumb_bend_target_rad": float(
                        run.config["control"]["grasp_targets_rad"][
                            "left_hand_thumb_bend_joint_actuator"
                        ]
                    ),
                    "grasp_acquisition_step": int(
                        validation["raw_prefix_validation"][
                            "grasp_acquisition_step"
                        ]
                    ),
                    "max_translation_before_acquisition_m": pose_metrics.get(
                        "max_translation_m"
                    ),
                    "max_orientation_before_acquisition_deg": pose_metrics.get(
                        "max_orientation_drift_deg"
                    ),
                },
                "artifacts": {
                    "directory": trajectory_id,
                    "resolved_config": f"{trajectory_id}/{config_path.name}",
                    "result": f"{trajectory_id}/{result_path.name}",
                    "trace": f"{trajectory_id}/{trace_path.name}",
                    "grasp_trace": (
                        f"{trajectory_id}/{grasp_trace_path.name}"
                        if grasp_trace_path is not None
                        else None
                    ),
                    "video": (
                        f"{trajectory_id}/{video_path.name}"
                        if video_path is not None
                        else None
                    ),
                    "sha256": hashes,
                },
            }
            entries.append(entry)

        successful = [entry for entry in entries if entry["rerun_grasp_success"]]
        selected: dict[str, Any] | None = None
        if best_source_id is not None:
            matching = [
                entry
                for entry in entries
                if entry["source_id"] == best_source_id
            ]
            if len(matching) != 1:
                raise ValueError("best_source_id must identify exactly one source run")
            if not matching[0]["rerun_grasp_success"]:
                raise ValueError("best_source_id cannot identify a failed grasp")
            selected = matching[0]
        elif successful:
            selected = sorted(successful, key=_best_sort_key)[0]

        aliases: dict[str, str] = {}
        for entry in successful:
            alias = f"grasp_{entry['source_id']}"
            if alias in aliases:
                raise ValueError(f"duplicate generated alias: {alias}")
            entry["aliases"].append(alias)
            aliases[alias] = str(entry["trajectory_id"])
        if selected is not None:
            selected["aliases"].append("best_nominal")
            aliases["best_nominal"] = str(selected["trajectory_id"])

        catalog = {
            "pose_preserving_seed_catalog_schema_version": (
                POSE_PRESERVING_SEED_CATALOG_SCHEMA_VERSION
            ),
            "trajectory_catalog_schema_version": 1,
            "validation_scope": VALIDATION_SCOPE,
            "experiment_id": experiment_id,
            "source_directory": str(source),
            "trajectory_count": len(entries),
            "validated_grasp_count": len(successful),
            "failed_grasp_count": len(entries) - len(successful),
            "all_sources_authenticated": True,
            "campaign_has_validated_grasp": bool(successful),
            "aliases": aliases,
            "trajectories": entries,
        }
        write_json(staging / "catalog.json", catalog)
        staging.rename(destination)
    return catalog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a Viewer-compatible schema-v6 pose-preserving catalog"
    )
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-count", type=int, default=6)
    parser.add_argument("--best-source-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = export_pose_preserving_seed_catalog(
        args.source_dir,
        args.output_dir,
        expected_count=args.expected_count,
        best_source_id=args.best_source_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["campaign_has_validated_grasp"] else 2


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())


__all__ = [
    "POSE_PRESERVING_GRASP_TRACE_SCHEMA_VERSION",
    "POSE_PRESERVING_SEED_CATALOG_SCHEMA_VERSION",
    "REQUIRED_ACQUISITION_CHECKS",
    "VALIDATION_SCOPE",
    "export_pose_preserving_seed_catalog",
    "main",
]
