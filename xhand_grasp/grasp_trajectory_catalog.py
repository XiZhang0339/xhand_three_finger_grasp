"""Transactional catalogs for independently reproduced grasp acquisitions.

The schema-v5 controller already gives grasp acquisition a precise sample
boundary: the acquisition sample is produced by a ``VERIFY`` command and the
first possible ``MANIPULATE`` command is issued one sample later.  This module
publishes that prefix without changing the controller or the complete trace
used by the existing evaluator and live Viewer.

Catalog success is deliberately scoped to grasp acquisition.  It requires the
existing schema-v5 ``stage_status.grasp_success`` result plus the global model
and simulation-integrity checks below; manipulation and lift checks are not
silently reinterpreted as grasp checks.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .artifacts import file_sha256, resolved_run_config, run_metadata, write_json
from .config import ACTIVE_FINGERS, validate_config
from .controller import grasp_gate_order, quaternion_drift_deg
from .simulation import run_simulation
from .trajectory_catalog import object_physical_parameters


GRASP_TRAJECTORY_CATALOG_SCHEMA_VERSION = 1
GRASP_TRACE_SCHEMA_VERSION = 1
VALIDATION_SCOPE = "grasp_acquisition"
POST_ACQUISITION_DIAGNOSTIC_S = 1.0
_SAFE_LABEL = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# These are complete-run integrity properties, not manipulation objectives.
# Requiring them prevents a valid 0.25 s gate from blessing a structurally
# invalid or numerically corrupted run.  Lift, support-clearing and operation
# duty checks intentionally do not appear here.
GLOBAL_INTEGRITY_CHECKS = (
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

_EVENT_FIELDS = (
    "grasp_acquisition_step",
    "manipulation_start_step",
    "manipulation_end_step",
    "termination_step",
)
_STATIC_AXIS_FIELDS = (
    "face_order",
    "finger_order",
    "actuator_order",
    "grasp_gate_order",
)
_REQUIRED_TIMESERIES_SHAPES: dict[str, tuple[int, ...] | None] = {
    "time": (),
    "cube_pos": (3,),
    "cube_quat": (4,),
    "cube_velocity": (6,),
    "root_pos": (3,),
    "root_quat": (4,),
    "ctrl": None,
    "joint_qpos": None,
    "joint_qvel": None,
    "actuator_force": None,
    "forbidden_contact": (),
    "support_contact": (),
    "max_penetration": (),
    "finite": (),
    "distal_face_force_n": (3, 8),
    "active_nondistal_force_n": (3,),
    "target_face_effective": (3,),
    "control_state": (),
    "grasp_gate": None,
    "grasp_gate_consecutive_steps": (),
    "grasp_acquired": (),
    "manipulation_progress": (),
    "finger_down_tilt_deg": (),
    "target_face_contact_centroid_world_m": (3, 3),
    "target_face_contact_centroid_valid": (3,),
    "three_contact_height_spread_m": (),
    "three_contact_height_aligned": (),
    "root_cube_center_distance_m": (),
    "thumb_bend_command_rad": (),
    "thumb_bend_qpos_rad": (),
    "distal_pad_force_n": (3,),
    "distal_nonpad_force_n": (3,),
    "distal_pad_force_fraction": (3,),
    "distal_active_taxel_count": (3,),
}


def _summary_grasp_success(summary: Mapping[str, Any] | None) -> bool:
    if not isinstance(summary, Mapping):
        return False
    stage = summary.get("stage_status")
    return bool(isinstance(stage, Mapping) and stage.get("grasp_success", False))


def _summary_full_success(summary: Mapping[str, Any]) -> bool:
    stage = summary.get("stage_status")
    return bool(
        summary.get("passed", False)
        and isinstance(stage, Mapping)
        and stage.get("full_success", False)
    )


def _candidate_id(candidate: Mapping[str, Any], fallback: int) -> int:
    value = candidate.get("candidate_id", fallback)
    if isinstance(value, bool):
        raise ValueError("candidate_id must be a non-negative integer")
    try:
        identifier = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate_id must be a non-negative integer") from exc
    if identifier < 0 or identifier != value:
        raise ValueError("candidate_id must be a non-negative integer")
    return identifier


def _prepare_candidates(
    candidates: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    materialized = tuple(candidates)
    if not materialized:
        raise ValueError("at least one grasp candidate or config is required")

    prepared: list[dict[str, Any]] = []
    experiment_id: str | None = None
    seen_ids: set[int] = set()
    seen_labels: set[str] = set()
    for fallback, value in enumerate(materialized):
        if not isinstance(value, Mapping):
            raise ValueError("each grasp candidate must be a mapping")
        is_candidate = "config" in value
        if is_candidate:
            raw_config = value.get("config")
            if not isinstance(raw_config, Mapping):
                raise ValueError("candidate config must be a mapping")
            config = copy.deepcopy(dict(raw_config))
            reported = value.get("summary")
            if reported is not None and not isinstance(reported, Mapping):
                raise ValueError("candidate summary must be a mapping")
            identifier = _candidate_id(value, fallback)
        else:
            config = copy.deepcopy(dict(value))
            reported = None
            identifier = fallback

        if identifier in seen_ids:
            raise ValueError("candidate_id values must be unique")
        seen_ids.add(identifier)
        label_value = value.get("label", f"grasp_{identifier:06d}")
        if (
            not isinstance(label_value, str)
            or _SAFE_LABEL.fullmatch(label_value) is None
        ):
            raise ValueError(
                "candidate label must use lowercase letters, digits, '_' or '-'"
            )
        if label_value in seen_labels:
            raise ValueError("candidate labels must be unique")
        seen_labels.add(label_value)
        run_context = config.get("run_context")
        if run_context is not None and run_context != {
            "kind": "parameter_override_run"
        }:
            raise ValueError(
                "grasp catalogs accept only canonical configs or independently "
                "rerun parameter_override_run configs"
            )
        validate_config(config)
        if int(config.get("schema_version", 0)) != 5:
            raise ValueError("grasp acquisition catalogs require schema version 5")
        current_experiment = str(config.get("experiment_id", ""))
        if not current_experiment:
            raise ValueError("grasp candidate config has no experiment_id")
        if experiment_id is None:
            experiment_id = current_experiment
        elif current_experiment != experiment_id:
            raise ValueError("all grasp candidates must belong to one experiment")
        prepared.append(
            {
                "candidate_id": identifier,
                "label": label_value,
                "parameter_override_run": run_context is not None,
                "config": config,
                "reported_summary": (
                    copy.deepcopy(dict(reported))
                    if isinstance(reported, Mapping)
                    else None
                ),
                "reported_grasp_success": _summary_grasp_success(reported),
            }
        )

    assert experiment_id is not None
    return prepared, experiment_id


def _load_full_trace(path: Path, summary: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        traces = {name: np.array(archive[name], copy=True) for name in archive.files}
    time = traces.get("time")
    if time is None or time.ndim != 1 or len(time) == 0:
        raise ValueError("full grasp trace requires a non-empty time axis")
    total_steps = len(time)
    if not np.issubdtype(time.dtype, np.number) or not np.isfinite(time).all():
        raise ValueError("full grasp trace time must be finite and numeric")
    if total_steps > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError("full grasp trace time must be strictly increasing")

    for name, trailing_shape in _REQUIRED_TIMESERIES_SHAPES.items():
        if name not in traces:
            raise ValueError(f"full grasp trace is missing {name}")
        values = traces[name]
        if values.shape[:1] != (total_steps,):
            raise ValueError(f"full grasp trace {name} has a different frame count")
        if trailing_shape is not None and values.shape[1:] != trailing_shape:
            raise ValueError(
                f"full grasp trace {name} must have trailing shape {trailing_shape}"
            )
        if values.dtype.kind not in "bUS" and not np.isfinite(values).all():
            raise ValueError(f"full grasp trace {name} contains NaN or Inf")

    widths = {
        traces[name].shape[1]
        for name in ("ctrl", "joint_qpos", "joint_qvel", "actuator_force")
    }
    if len(widths) != 1 or next(iter(widths)) <= 0:
        raise ValueError("full grasp trace actuator arrays must share a positive width")
    missing_axes = [name for name in _STATIC_AXIS_FIELDS if name not in traces]
    if missing_axes:
        raise ValueError(
            "full grasp trace is missing static axes: " + ", ".join(missing_axes)
        )
    if tuple(str(value) for value in traces["finger_order"].tolist()) != tuple(
        ACTIVE_FINGERS
    ):
        raise ValueError("full grasp trace finger_order is not canonical")
    expected_gate_order = grasp_gate_order(5)
    if tuple(str(value) for value in traces["grasp_gate_order"].tolist()) != tuple(
        expected_gate_order
    ):
        raise ValueError("full grasp trace grasp_gate_order is not canonical schema-v5")
    if traces["grasp_gate"].shape[1:] != (len(expected_gate_order),):
        raise ValueError("full grasp trace grasp_gate width is not canonical")
    quaternion_norm = np.linalg.norm(traces["cube_quat"], axis=1)
    if not np.allclose(quaternion_norm, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("full grasp trace cube quaternions are not normalized")

    metrics = summary.get("metrics")
    for name in _EVENT_FIELDS:
        if name not in traces or np.asarray(traces[name]).shape != ():
            raise ValueError(f"full grasp trace is missing scalar event {name}")
        if isinstance(metrics, Mapping) and name in metrics:
            if int(np.asarray(traces[name])) != int(metrics[name]):
                raise ValueError(f"summary metric {name} disagrees with full trace")
    return traces


def _global_integrity(summary: Mapping[str, Any]) -> dict[str, bool]:
    checks = summary.get("checks")
    source = checks if isinstance(checks, Mapping) else {}
    return {name: bool(source.get(name, False)) for name in GLOBAL_INTEGRITY_CHECKS}


def _grasp_validation(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    traces: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    total_steps = len(np.asarray(traces["time"]))
    acquisition_step = int(np.asarray(traces["grasp_acquisition_step"]))
    stage_grasp_success = _summary_grasp_success(summary)
    event_in_range = 0 <= acquisition_step < total_steps
    event_is_verify = bool(
        event_in_range
        and str(np.asarray(traces["control_state"])[acquisition_step]) == "VERIFY"
    )
    acquisition_latch_consistent = bool(
        event_in_range
        and bool(np.asarray(traces["grasp_acquired"])[acquisition_step])
        and not np.any(np.asarray(traces["grasp_acquired"][:acquisition_step]))
    )
    manipulation_excluded_at_event = bool(
        event_in_range
        and float(np.asarray(traces["manipulation_progress"])[acquisition_step])
        == 0.0
    )
    global_checks = _global_integrity(summary)
    global_pass = all(global_checks.values())
    passed = bool(
        stage_grasp_success
        and event_in_range
        and event_is_verify
        and acquisition_latch_consistent
        and manipulation_excluded_at_event
        and global_pass
    )
    failed = []
    if not stage_grasp_success:
        failed.append("schema_v5_stage_grasp_success")
    if not event_in_range:
        failed.append("grasp_acquisition_event_in_range")
    if not event_is_verify:
        failed.append("grasp_acquisition_event_is_verify")
    if not acquisition_latch_consistent:
        failed.append("grasp_acquisition_latch_consistent")
    if not manipulation_excluded_at_event:
        failed.append("no_manipulation_at_grasp_acquisition")
    failed.extend(name for name, value in global_checks.items() if not value)
    return {
        "validation_scope": VALIDATION_SCOPE,
        "passed": passed,
        "stage_grasp_success": stage_grasp_success,
        "global_integrity_passed": global_pass,
        "global_integrity_checks": global_checks,
        "grasp_acquisition_step": acquisition_step,
        "grasp_acquisition_time_s": (
            float(np.asarray(traces["time"])[acquisition_step])
            if event_in_range
            else None
        ),
        "stable_window_s": float(config["control_protocol"]["stable_window_s"]),
        "event_in_range": event_in_range,
        "event_is_verify": event_is_verify,
        "acquisition_latch_consistent": acquisition_latch_consistent,
        "no_manipulation_at_acquisition": manipulation_excluded_at_event,
        "failed_checks": failed,
    }


def _post_acquisition_hold_diagnostic(
    traces: Mapping[str, np.ndarray], acquisition_step: int
) -> dict[str, Any]:
    """Report (but never gate on) one second immediately after acquisition."""

    time = np.asarray(traces["time"], dtype=np.float64)
    total_steps = len(time)
    if total_steps > 1:
        timestep = float(np.median(np.diff(time)))
    else:
        timestep = math.nan
    requested_steps = (
        int(round(POST_ACQUISITION_DIAGNOSTIC_S / timestep))
        if math.isfinite(timestep) and timestep > 0.0
        else 0
    )
    start = acquisition_step + 1
    stop = min(total_steps, start + requested_steps)
    available = 0 <= acquisition_step < total_steps and stop > start
    complete = bool(available and stop - start == requested_steps)
    base = {
        "scope": "diagnostic_only_not_an_acceptance_gate",
        "requested_duration_s": POST_ACQUISITION_DIAGNOSTIC_S,
        "requested_steps": requested_steps,
        "window_start_step": start if available else None,
        "window_stop_step_exclusive": stop if available else None,
        "sample_count": max(0, stop - start) if available else 0,
        "complete_window": complete,
    }
    if not available:
        return {
            **base,
            "target_face_topology_duty": None,
            "contact_height_aligned_duty": None,
            "support_contact_duty": None,
            "max_translation_from_acquisition_m": None,
            "max_orientation_drift_from_acquisition_deg": None,
            "max_linear_speed_m_s": None,
            "forbidden_contact_steps": None,
            "material_active_nondistal_violation_steps": None,
        }

    window = slice(start, stop)
    target_topology = np.all(
        np.asarray(traces["target_face_effective"], dtype=bool)[window], axis=1
    )
    aligned = np.asarray(traces["three_contact_height_aligned"], dtype=bool)[window]
    support = np.asarray(traces["support_contact"], dtype=bool)[window]
    position = np.asarray(traces["cube_pos"], dtype=np.float64)
    quaternion = np.asarray(traces["cube_quat"], dtype=np.float64)
    displacement = np.linalg.norm(position[window] - position[acquisition_step], axis=1)
    orientation = np.asarray(
        [
            quaternion_drift_deg(quaternion[acquisition_step], value)
            for value in quaternion[window]
        ],
        dtype=np.float64,
    )
    speed = np.linalg.norm(
        np.asarray(traces["cube_velocity"], dtype=np.float64)[window, :3], axis=1
    )
    gate_order = tuple(str(value) for value in traces["grasp_gate_order"].tolist())
    nondistal_index = gate_order.index("no_active_nondistal_contact")
    nondistal_ok = np.asarray(traces["grasp_gate"], dtype=bool)[
        window, nondistal_index
    ]
    return {
        **base,
        "target_face_topology_duty": float(np.mean(target_topology)),
        "contact_height_aligned_duty": float(np.mean(aligned)),
        "support_contact_duty": float(np.mean(support)),
        "max_translation_from_acquisition_m": float(np.max(displacement)),
        "max_orientation_drift_from_acquisition_deg": float(np.max(orientation)),
        "max_linear_speed_m_s": float(np.max(speed)),
        "forbidden_contact_steps": int(
            np.count_nonzero(np.asarray(traces["forbidden_contact"])[window])
        ),
        "material_active_nondistal_violation_steps": int(
            np.count_nonzero(~nondistal_ok)
        ),
    }


def _grasp_trace_payload(
    config: Mapping[str, Any],
    traces: Mapping[str, np.ndarray],
    acquisition_step: int,
    *,
    source_trace_sha256: str,
) -> dict[str, np.ndarray]:
    total_steps = len(np.asarray(traces["time"]))
    if not 0 <= acquisition_step < total_steps:
        raise ValueError("cannot extract a grasp trace without an acquisition event")
    stop = acquisition_step + 1
    stable_steps = int(
        round(
            float(config["control_protocol"]["stable_window_s"])
            / float(np.median(np.diff(np.asarray(traces["time"], dtype=np.float64))))
        )
    )
    window_start = acquisition_step - stable_steps + 1
    if window_start < 0:
        raise ValueError("grasp acquisition precedes its declared stable window")
    states = np.asarray(traces["control_state"]).astype(str)
    acquired = np.asarray(traces["grasp_acquired"], dtype=bool)
    progress = np.asarray(traces["manipulation_progress"], dtype=np.float64)
    counter = np.asarray(traces["grasp_gate_consecutive_steps"], dtype=np.int64)
    gate = np.asarray(traces["grasp_gate"], dtype=bool)
    if states[acquisition_step] != "VERIFY":
        raise ValueError("grasp trace must end on a VERIFY command sample")
    if np.any(np.isin(states[:stop], ("MANIPULATE", "HOLD"))):
        raise ValueError("grasp trace prefix contains an operation command")
    if np.any(progress[:stop] != 0.0):
        raise ValueError("grasp trace prefix contains manipulation progress")
    acquired_indices = np.flatnonzero(acquired[:stop])
    if not np.array_equal(acquired_indices, np.asarray([acquisition_step])):
        raise ValueError("grasp latch must first become true on the terminal sample")
    if int(counter[acquisition_step]) != stable_steps:
        raise ValueError("terminal grasp counter does not match stable_window_s")
    if not np.all(gate[window_start:stop]):
        raise ValueError("terminal grasp window does not pass every gate")

    payload: dict[str, np.ndarray] = {}
    for name, source in traces.items():
        values = np.asarray(source)
        if name in _EVENT_FIELDS:
            if name == "grasp_acquisition_step":
                payload[name] = np.asarray(acquisition_step, dtype=np.int64)
            continue
        if name == "video_frame_steps":
            frame_steps = np.asarray(values, dtype=np.int64)
            payload[name] = frame_steps[frame_steps <= acquisition_step].copy()
        elif name in _STATIC_AXIS_FIELDS or values.shape == ():
            payload[name] = np.array(values, copy=True)
        elif values.shape[:1] == (total_steps,):
            payload[name] = np.array(values[:stop], copy=True)
        else:
            payload[name] = np.array(values, copy=True)
    payload.update(
        {
            "grasp_trace_schema_version": np.asarray(
                GRASP_TRACE_SCHEMA_VERSION, dtype=np.int64
            ),
            "validation_scope": np.asarray(VALIDATION_SCOPE, dtype=np.str_),
            "source_total_steps": np.asarray(total_steps, dtype=np.int64),
            "segment_start_step": np.asarray(0, dtype=np.int64),
            "segment_stop_step_inclusive": np.asarray(
                acquisition_step, dtype=np.int64
            ),
            "source_trace_sha256": np.asarray(source_trace_sha256, dtype=np.str_),
        }
    )
    return payload


def _validated_metadata(config_path: Path) -> dict[str, Any]:
    metadata = run_metadata(config_path)
    if not isinstance(metadata, Mapping):
        raise RuntimeError("run_metadata must return a mapping")
    result = copy.deepcopy(dict(metadata))
    expected = file_sha256(config_path)
    if result.get("config_sha256") != expected:
        raise RuntimeError("run_metadata did not bind the resolved config SHA-256")
    return result


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_grasp_trajectory_catalog(
    candidates: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    video: bool = True,
    best_candidate_id: int | None = None,
) -> dict[str, Any]:
    """Independently rerun and atomically publish grasp-acquisition traces.

    Items may be search-result mappings containing ``candidate_id``, ``config``
    and an optional reported ``summary``, or bare schema-v5 config mappings.
    A reported grasp that fails its independent rerun aborts publication.
    """

    if not isinstance(video, bool):
        raise ValueError("video must be a boolean")
    if best_candidate_id is not None:
        raw_best_candidate_id = best_candidate_id
        if isinstance(raw_best_candidate_id, bool):
            raise ValueError("best_candidate_id must be a non-negative integer")
        try:
            best_candidate_id = int(raw_best_candidate_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("best_candidate_id must be a non-negative integer") from exc
        if best_candidate_id < 0 or best_candidate_id != raw_best_candidate_id:
            raise ValueError("best_candidate_id must be a non-negative integer")

    prepared, experiment_id = _prepare_candidates(candidates)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"output directory already exists: {destination}; choose a new output"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}.staging."
    ) as staging_name:
        staging = Path(staging_name)
        for selection in prepared:
            candidate_id = int(selection["candidate_id"])
            working = staging / f"candidate_{candidate_id:06d}"
            working.mkdir()
            full_trace_path = working / "trace.npz"
            video_path = working / "trajectory.mp4" if video else None
            summary = run_simulation(
                copy.deepcopy(selection["config"]),
                trace_path=full_trace_path,
                video_path=video_path,
            )
            if not isinstance(summary, dict):
                raise RuntimeError("run_simulation must return a summary mapping")
            traces = _load_full_trace(full_trace_path, summary)
            validation = _grasp_validation(selection["config"], summary, traces)
            if selection["reported_grasp_success"] and not validation["passed"]:
                raise RuntimeError(
                    f"reported grasp candidate {candidate_id} did not reproduce: "
                    f"failed_checks={validation['failed_checks']!r}"
                )
            classification = (
                "validated_grasp_acquisition"
                if validation["passed"]
                else "failed_grasp_acquisition_validation"
            )
            trajectory_id = (
                f"grasp_{candidate_id:06d}"
                if validation["passed"]
                else f"failed_grasp_{candidate_id:06d}"
            )
            final_directory = staging / trajectory_id
            if final_directory.exists():
                raise ValueError(f"duplicate trajectory identifier: {trajectory_id}")
            working.rename(final_directory)
            full_trace_path = final_directory / "trace.npz"
            video_path = final_directory / "trajectory.mp4" if video else None

            source_trace_sha256 = file_sha256(full_trace_path)
            grasp_trace_path: Path | None = None
            acquisition_step = int(validation["grasp_acquisition_step"])
            if 0 <= acquisition_step < len(traces["time"]):
                grasp_trace_path = final_directory / "grasp_trace.npz"
                payload = _grasp_trace_payload(
                    selection["config"],
                    traces,
                    acquisition_step,
                    source_trace_sha256=source_trace_sha256,
                )
                np.savez_compressed(grasp_trace_path, **payload)

            resolved = resolved_run_config(selection["config"], summary)
            config_path = final_directory / "resolved_config.json"
            write_json(config_path, resolved)
            metadata = _validated_metadata(config_path)
            physical = object_physical_parameters(resolved)
            hold_diagnostic = _post_acquisition_hold_diagnostic(
                traces, acquisition_step
            )
            local_hashes = {
                "resolved_config": file_sha256(config_path),
                "trace": source_trace_sha256,
            }
            if grasp_trace_path is not None:
                local_hashes["grasp_trace"] = file_sha256(grasp_trace_path)
            if video_path is not None:
                if not video_path.is_file():
                    raise RuntimeError("run_simulation did not create the requested MP4")
                local_hashes["video"] = file_sha256(video_path)
            artifacts = {
                "resolved_config": config_path.name,
                "result": "result.json",
                "trace": full_trace_path.name,
                "grasp_trace": (
                    grasp_trace_path.name if grasp_trace_path is not None else None
                ),
                "video": video_path.name if video_path is not None else None,
                "sha256": local_hashes,
            }
            full_success = _summary_full_success(summary)
            result = {
                "grasp_trajectory_catalog_schema_version": (
                    GRASP_TRAJECTORY_CATALOG_SCHEMA_VERSION
                ),
                "validation_scope": VALIDATION_SCOPE,
                "trajectory_id": trajectory_id,
                "classification": classification,
                "candidate_id": candidate_id,
                "label": str(selection["label"]),
                "parameter_override_run": bool(
                    selection["parameter_override_run"]
                ),
                "reported_grasp_success": bool(
                    selection["reported_grasp_success"]
                ),
                "rerun_grasp_success": bool(validation["passed"]),
                "rerun_full_success": full_success,
                "grasp_validation": validation,
                "post_acquisition_hold_diagnostic": hold_diagnostic,
                "target_faces": copy.deepcopy(
                    resolved["contact_topology"]["target_faces"]
                ),
                "physical_parameters": physical,
                "metadata": metadata,
                "summary": summary,
                "config": resolved,
                "experiment_status": resolved["experiment_status"],
                "artifacts": artifacts,
            }
            result_path = final_directory / "result.json"
            write_json(result_path, result)
            catalog_hashes = {
                **local_hashes,
                "result": _sha256_bytes(result_path),
            }
            entries.append(
                {
                    "trajectory_id": trajectory_id,
                    "classification": classification,
                    "candidate_id": candidate_id,
                    "label": str(selection["label"]),
                    "parameter_override_run": bool(
                        selection["parameter_override_run"]
                    ),
                    "reported_grasp_success": bool(
                        selection["reported_grasp_success"]
                    ),
                    "rerun_grasp_success": bool(validation["passed"]),
                    "rerun_full_success": full_success,
                    "stage_status": copy.deepcopy(
                        summary.get("stage_status", {})
                    ),
                    "grasp_validation": copy.deepcopy(validation),
                    "post_acquisition_hold_diagnostic": copy.deepcopy(
                        hold_diagnostic
                    ),
                    "target_faces": copy.deepcopy(
                        resolved["contact_topology"]["target_faces"]
                    ),
                    "physical_parameters": physical,
                    "artifacts": {
                        "directory": trajectory_id,
                        "resolved_config": f"{trajectory_id}/{config_path.name}",
                        "result": f"{trajectory_id}/{result_path.name}",
                        "trace": f"{trajectory_id}/{full_trace_path.name}",
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
                        "sha256": catalog_hashes,
                    },
                }
            )

        aliases: dict[str, str] = {}
        if best_candidate_id is not None:
            matching = [
                entry
                for entry in entries
                if int(entry["candidate_id"]) == best_candidate_id
            ]
            if len(matching) != 1:
                raise ValueError(
                    "best_candidate_id must identify exactly one published candidate"
                )
            if not matching[0]["rerun_grasp_success"]:
                raise RuntimeError(
                    "best_candidate_id cannot alias a failed grasp validation"
                )
            matching[0]["aliases"] = ["best_grasp"]
            aliases["best_grasp"] = str(matching[0]["trajectory_id"])

        catalog = {
            "grasp_trajectory_catalog_schema_version": (
                GRASP_TRAJECTORY_CATALOG_SCHEMA_VERSION
            ),
            "validation_scope": VALIDATION_SCOPE,
            "experiment_id": experiment_id,
            "trajectory_count": len(entries),
            "validated_grasp_count": sum(
                bool(entry["rerun_grasp_success"]) for entry in entries
            ),
            "failed_grasp_count": sum(
                not bool(entry["rerun_grasp_success"]) for entry in entries
            ),
            "parameter_override_trajectory_count": sum(
                bool(entry["parameter_override_run"]) for entry in entries
            ),
            "campaign_has_validated_grasp": any(
                bool(entry["rerun_grasp_success"]) for entry in entries
            ),
            "all_reported_grasps_reproduced": all(
                not bool(entry["reported_grasp_success"])
                or bool(entry["rerun_grasp_success"])
                for entry in entries
            ),
            # This is an honest observation only; it is never used as the
            # catalog's success predicate.
            "all_reruns_full_success": bool(entries)
            and all(bool(entry["rerun_full_success"]) for entry in entries),
            "aliases": aliases,
            "trajectories": entries,
        }
        write_json(staging / "catalog.json", catalog)
        if destination.exists():
            raise FileExistsError(
                f"output directory appeared during execution: {destination}; "
                "refusing to overwrite"
            )
        staging.rename(destination)
    return catalog


__all__ = [
    "GLOBAL_INTEGRITY_CHECKS",
    "GRASP_TRACE_SCHEMA_VERSION",
    "GRASP_TRAJECTORY_CATALOG_SCHEMA_VERSION",
    "VALIDATION_SCOPE",
    "export_grasp_trajectory_catalog",
]
