#!/usr/bin/env python3
"""Refine a measured v11 grasp by shifting the fixed root in cube-local -Y.

The source cube pose/physics and all controller values are immutable.  Each
candidate rebuilds the coupled hand pose once, runs fresh free-body dynamics,
and measures force-weighted contacts in the moving cube frame.  Only the best
candidate retains a full NPZ trace for deterministic Viewer inspection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xhand_grasp.artifacts import (
    file_sha256,
    json_compatible,
    resolved_run_config,
    write_json,
)
from xhand_grasp.config import load_config
from xhand_grasp.simulation import SimulationSession
from xhand_grasp.viewer import apply_viewer_overrides


DEFAULT_SOURCE = Path(
    "artifacts/left_opposed_face_palm_down_larger_relative_wrist_pose_"
    "actual_contact_smooth_vertical_lift/tune/"
    "negative_clockwise_orbit_adaptive_fine_sweep_from_4367930796723148114_v1/"
    "orbit_neg_1p1_deg/resolved_config.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/left_opposed_face_palm_down_larger_relative_wrist_pose_"
    "actual_contact_smooth_vertical_lift/tune/"
    "thumb_negative_y_root_translation_sweep_from_orbit_neg_1p1_v1"
)
FINGERS = ("thumb", "index", "mid")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _cube_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    cube = _mapping(config.get("cube"))
    return {
        key: copy.deepcopy(cube.get(key))
        for key in (
            "center_xy_m",
            "edge_m",
            "friction",
            "mass_kg",
            "rpy_deg",
            "solref_timeconst_s",
            "z_offset_m",
        )
    }


def _label(shift_mm: float) -> str:
    magnitude = f"{abs(float(shift_mm)):.1f}".replace(".", "p")
    return f"root_shift_neg_y_{magnitude}_mm"


def _local_contact_centroids(
    traces: Mapping[str, np.ndarray], start: int, end: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-finger median cube-local centroids and valid counts."""

    if start < 0 or end < start:
        return np.full((3, 3), np.nan), np.zeros(3, dtype=np.int64)
    positions = np.asarray(
        traces["target_face_contact_centroid_world_m"], dtype=np.float64
    )
    valid = np.asarray(
        traces["target_face_contact_centroid_valid"], dtype=bool
    )
    cube_pos = np.asarray(traces["cube_pos"], dtype=np.float64)
    cube_quat = np.asarray(traces["cube_quat"], dtype=np.float64)
    local = np.full((end - start + 1, 3, 3), np.nan, dtype=np.float64)
    for local_step, step in enumerate(range(start, end + 1)):
        rotation_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation_flat, cube_quat[step])
        rotation = rotation_flat.reshape(3, 3)
        local[local_step] = (
            rotation.T @ (positions[step] - cube_pos[step]).T
        ).T
    window_valid = valid[start : end + 1]
    medians = np.full((3, 3), np.nan, dtype=np.float64)
    counts = np.count_nonzero(window_valid, axis=0).astype(np.int64)
    for finger in range(3):
        if counts[finger]:
            medians[finger] = np.median(
                local[window_valid[:, finger], finger], axis=0
            )
    return medians, counts


def _stable_window(summary: Mapping[str, Any]) -> tuple[int, int]:
    metrics = _mapping(summary.get("metrics"))
    actual = _mapping(metrics.get("actual_grasp_pose"))
    events = _mapping(actual.get("events"))
    return (
        int(events.get("stable_window_start_step", -1)),
        int(events.get("stable_window_end_step", -1)),
    )


def _candidate_record(
    *,
    shift_mm: float,
    summary: Mapping[str, Any],
    traces: Mapping[str, np.ndarray],
    baseline_centroids_m: np.ndarray,
) -> dict[str, Any]:
    metrics = _mapping(summary.get("metrics"))
    stage = _mapping(summary.get("stage_status"))
    actual = _mapping(metrics.get("actual_grasp_pose"))
    alignment = _mapping(metrics.get("contact_alignment"))
    verify_alignment = _mapping(alignment.get("verify"))
    closure = _mapping(metrics.get("closure_alignment"))
    per_finger_closure = _mapping(closure.get("per_finger"))
    pose = _mapping(metrics.get("pose_preservation"))
    start, end = _stable_window(summary)
    centroids, counts = _local_contact_centroids(traces, start, end)
    if start >= 0 and end >= start:
        pad = np.asarray(traces["distal_pad_force_fraction"], dtype=np.float64)[
            start : end + 1
        ]
        pad_min = np.min(pad, axis=0)
    else:
        pad_min = np.zeros(3, dtype=np.float64)
    purity = _mapping(metrics.get("target_force_purity_min_on_contact"))
    face_forces = _mapping(metrics.get("verify_peak_target_face_force_n"))

    def closure_value(finger: str, name: str, fallback: float) -> float:
        value = _mapping(per_finger_closure.get(finger)).get(name, fallback)
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    actual_checks = _mapping(actual.get("checks"))
    actual_gate = bool(actual.get("passed", False)) and bool(actual_checks) and all(
        bool(value) for value in actual_checks.values()
    )
    index_mid_forward = all(
        (
            float(purity.get(finger, 0.0)) >= 0.95
            and float(pad_min[FINGERS.index(finger)]) >= 0.95
            and closure_value(finger, "angle_p95_deg", math.inf) <= 30.0
            and closure_value(finger, "minimum_inward_speed_m_s", 0.0) > 0.0
        )
        for finger in ("index", "mid")
    )
    strict_grasp = bool(stage.get("grasp_success", False))
    hard_pass = (
        strict_grasp
        and actual_gate
        and index_mid_forward
        and int(metrics.get("forbidden_contact_steps", 0)) == 0
        and int(metrics.get("verify_max_consecutive_all_gate_steps", 0)) >= 250
    )
    shifts = centroids - baseline_centroids_m
    return {
        "requested_root_shift_cube_y_mm": -abs(float(shift_mm)),
        "strict_grasp_success": strict_grasp,
        "actual_grasp_pose_gate_passed": actual_gate,
        "index_middle_forward_grasp": index_mid_forward,
        "hard_pass": hard_pass,
        "stable_window_start_step": start,
        "stable_window_end_step": end,
        "stable_window_valid_counts": {
            finger: int(counts[index]) for index, finger in enumerate(FINGERS)
        },
        "contact_centroid_cube_local_mm": {
            finger: (centroids[index] * 1000.0).tolist()
            for index, finger in enumerate(FINGERS)
        },
        "contact_centroid_shift_cube_local_mm": {
            finger: (shifts[index] * 1000.0).tolist()
            for index, finger in enumerate(FINGERS)
        },
        "index_middle_y_midpoint_mm": float(
            500.0 * (centroids[1, 1] + centroids[2, 1])
        ),
        "target_face_force_purity_min": {
            finger: float(purity.get(finger, 0.0)) for finger in FINGERS
        },
        "distal_pad_force_fraction_min": {
            finger: float(pad_min[index])
            for index, finger in enumerate(FINGERS)
        },
        "verify_peak_target_face_force_n": {
            finger: float(face_forces.get(finger, 0.0)) for finger in FINGERS
        },
        "closure_alignment_p95_deg": {
            finger: closure_value(finger, "angle_p95_deg", math.inf)
            for finger in FINGERS
        },
        "closure_minimum_inward_speed_m_s": {
            finger: closure_value(
                finger, "minimum_inward_speed_m_s", 0.0
            )
            for finger in FINGERS
        },
        "height_spread_max_mm": (
            None
            if verify_alignment.get("height_spread_max_m") is None
            else 1000.0 * float(verify_alignment["height_spread_max_m"])
        ),
        "pose_translation_mm": 1000.0 * float(
            pose.get("max_translation_m", math.inf)
        ),
        "pose_orientation_deg": float(
            pose.get("max_orientation_drift_deg", math.inf)
        ),
        "maximum_penetration_mm": 1000.0 * float(
            metrics.get("max_penetration_m", math.inf)
        ),
        "forbidden_contact_steps": int(
            metrics.get("forbidden_contact_steps", 0)
        ),
        "failed_checks": list(summary.get("failed_checks", [])),
    }


def _rank(record: Mapping[str, Any], target_shift_mm: float) -> tuple[Any, ...]:
    centroids = _mapping(record.get("contact_centroid_shift_cube_local_mm"))
    thumb = centroids.get("thumb", (math.inf, math.inf, math.inf))
    thumb_y = float(thumb[1]) if len(thumb) == 3 else math.inf
    forces = _mapping(record.get("verify_peak_target_face_force_n"))
    values = np.asarray(
        [float(forces.get(finger, 0.0)) for finger in FINGERS], dtype=np.float64
    )
    force_cv = (
        math.inf
        if float(np.mean(values)) <= 0.0
        else float(np.std(values) / np.mean(values))
    )
    return (
        int(bool(record.get("hard_pass"))),
        -abs(thumb_y + abs(float(target_shift_mm))),
        -abs(float(record.get("index_middle_y_midpoint_mm", math.inf))),
        -float(record.get("pose_translation_mm", math.inf)),
        -float(record.get("pose_orientation_deg", math.inf)),
        -force_cv,
        -abs(float(record.get("requested_root_shift_cube_y_mm", math.inf))),
    )


def _mark_override_status(
    resolved: dict[str, Any], summary: Mapping[str, Any]
) -> None:
    stage = _mapping(summary.get("stage_status"))
    resolved["experiment_status"] = {
        "classification": "parameter_override_run",
        "passed": bool(summary.get("passed", False)),
        "hard_constraints_passed": bool(summary.get("passed", False)),
        "grasp_success": bool(stage.get("grasp_success", False)),
        "manipulation_success": bool(stage.get("manipulation_success", False)),
        "full_success": bool(stage.get("full_success", False)),
        "failed_checks": list(summary.get("failed_checks", [])),
        "note": (
            "Fresh fixed-root cube-local -Y parameter override; source success "
            "evidence was not inherited."
        ),
    }


def _simulate(config: dict[str, Any]) -> tuple[dict[str, Any], Mapping[str, np.ndarray]]:
    session = SimulationSession(config)
    try:
        while not session.complete:
            session.advance_one()
        summary = session.finalize()
        return summary, session.traces
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--shifts-mm",
        nargs="+",
        type=float,
        default=(2.0, 3.0, 4.0, 5.0, 6.0),
        help="positive magnitudes applied along cube-local -Y",
    )
    parser.add_argument("--target-shift-mm", type=float, default=4.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="authenticate and reuse complete per-shift simulations",
    )
    args = parser.parse_args()

    source_path = args.source_config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    shifts = tuple(float(value) for value in args.shifts_mm)
    if (
        not shifts
        or len(set(shifts)) != len(shifts)
        or not all(math.isfinite(value) and value > 0.0 for value in shifts)
    ):
        raise ValueError("--shifts-mm must contain unique positive finite values")
    if not math.isfinite(args.target_shift_mm) or args.target_shift_mm <= 0.0:
        raise ValueError("--target-shift-mm must be positive and finite")
    if output.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {output}")

    source = load_config(source_path)
    source_cube = _cube_signature(source)
    source_control = copy.deepcopy(source["control"])
    relative = _mapping(
        _mapping(source.get("candidate_metadata")).get(
            "relative_wrist_pose_search"
        )
    )
    root_delta = np.asarray(relative["root_delta_cube_m"], dtype=np.float64)
    if root_delta.shape != (3,) or not np.isfinite(root_delta).all():
        raise ValueError("source has no finite v11 root_delta_cube_m")

    # Authenticate and measure the exact source trace instead of treating the
    # resolved config's success label as evidence.
    source_trace = source_path.parent / "trace.npz"
    if not source_trace.is_file():
        raise FileNotFoundError(f"source trace does not exist: {source_trace}")
    with np.load(source_trace, allow_pickle=False) as trace:
        source_start = int(np.asarray(trace["grasp_stable_window_start_step"]))
        source_end = int(np.asarray(trace["grasp_stable_window_end_step"]))
        baseline_centroids, baseline_counts = _local_contact_centroids(
            trace, source_start, source_end
        )
    if np.any(baseline_counts < 250) or not np.isfinite(baseline_centroids).all():
        raise RuntimeError("source has no authenticated 250-step contact window")

    output.mkdir(parents=True, exist_ok=args.resume)
    records: list[dict[str, Any]] = []
    configs: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for shift_mm in shifts:
        requested = root_delta.copy()
        requested[1] -= shift_mm / 1000.0
        config, changed = apply_viewer_overrides(
            source,
            root_delta_cube_mm=(requested * 1000.0).tolist(),
        )
        if not changed:
            raise RuntimeError("root -Y shift was not marked as an override")
        if _cube_signature(config) != source_cube:
            raise RuntimeError("root -Y refinement changed cube pose or physics")
        if config["control"] != source_control:
            raise RuntimeError("root -Y refinement changed controller values")
        label = _label(shift_mm)
        directory = output / label
        result_path = directory / "result.json"
        config_path = directory / "resolved_config.json"
        if args.resume and directory.is_dir():
            if not result_path.is_file() or not config_path.is_file():
                raise RuntimeError(f"incomplete resume artifacts: {directory}")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("source_config_sha256") != file_sha256(source_path):
                raise RuntimeError(f"resume source config hash changed: {result_path}")
            if payload.get("source_trace_sha256") != file_sha256(source_trace):
                raise RuntimeError(f"resume source trace hash changed: {result_path}")
            resolved = load_config(config_path)
            if _cube_signature(resolved) != source_cube:
                raise RuntimeError(f"resume cube inputs changed: {config_path}")
            if resolved["control"] != source_control:
                raise RuntimeError(f"resume controls changed: {config_path}")
            summary = payload.get("summary")
            record = payload.get("record")
            if not isinstance(summary, Mapping) or not isinstance(record, dict):
                raise RuntimeError(f"resume result is incomplete: {result_path}")
            if record.get("label") != label or not math.isclose(
                float(record.get("requested_root_shift_cube_y_mm", math.inf)),
                -shift_mm,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"resume shift identity changed: {result_path}")
        else:
            directory.mkdir()
            summary, traces = _simulate(config)
            record = _candidate_record(
                shift_mm=shift_mm,
                summary=summary,
                traces=traces,
                baseline_centroids_m=baseline_centroids,
            )
            record["label"] = label
            record["root_delta_cube_mm"] = (requested * 1000.0).tolist()
            resolved = resolved_run_config(config, summary)
            _mark_override_status(resolved, summary)
            write_json(config_path, resolved)
            write_json(
                result_path,
                {
                    "diagnostic_kind": "thumb_negative_y_fixed_root_refinement",
                    "source_config": str(source_path),
                    "source_config_sha256": file_sha256(source_path),
                    "source_trace_sha256": file_sha256(source_trace),
                    "cube_world_configuration_unchanged": True,
                    "controls_unchanged": True,
                    "record": record,
                    "summary": summary,
                },
            )
        record["artifacts"] = {
            "resolved_config": f"{label}/resolved_config.json",
            "result": f"{label}/result.json",
        }
        configs[label] = config
        summaries[label] = summary
        records.append(record)
        print(
            f"shift=-{shift_mm:g} mm hard={record['hard_pass']} "
            f"thumb_y={record['contact_centroid_cube_local_mm']['thumb'][1]:.3f} "
            f"index_y={record['contact_centroid_cube_local_mm']['index'][1]:.3f} "
            f"mid_y={record['contact_centroid_cube_local_mm']['mid'][1]:.3f}"
        )

    best = max(records, key=lambda value: _rank(value, args.target_shift_mm))
    best_label = str(best["label"])
    best_directory = output / best_label
    # Re-run the selected configuration and retain the exact trace used by the
    # published result.  Determinism is checked before replacing compact files.
    best_session = SimulationSession(configs[best_label])
    try:
        while not best_session.complete:
            best_session.advance_one()
        rerun_summary = best_session.finalize(
            trace_path=best_directory / "trace.npz"
        )
    finally:
        best_session.close()
    if json.dumps(json_compatible(rerun_summary), sort_keys=True) != json.dumps(
        json_compatible(summaries[best_label]), sort_keys=True
    ):
        raise RuntimeError("selected candidate was not deterministic on full rerun")
    best_resolved = resolved_run_config(configs[best_label], rerun_summary)
    _mark_override_status(best_resolved, rerun_summary)
    write_json(best_directory / "resolved_config.json", best_resolved)
    write_json(
        best_directory / "result.json",
        {
            "diagnostic_kind": "thumb_negative_y_fixed_root_refinement",
            "source_config": str(source_path),
            "source_config_sha256": file_sha256(source_path),
            "source_trace_sha256": file_sha256(source_trace),
            "cube_world_configuration_unchanged": True,
            "controls_unchanged": True,
            "record": best,
            "summary": rerun_summary,
        },
    )
    best["artifacts"]["trace"] = f"{best_label}/trace.npz"
    best["artifacts"]["sha256"] = {
        name: file_sha256(best_directory / filename)
        for name, filename in (
            ("resolved_config", "resolved_config.json"),
            ("result", "result.json"),
            ("trace", "trace.npz"),
        )
    }
    report = {
        "diagnostic_schema_version": 1,
        "diagnostic_kind": "thumb_negative_y_fixed_root_refinement",
        "status": "complete",
        "source_config": str(source_path),
        "source_config_sha256": file_sha256(source_path),
        "source_trace_sha256": file_sha256(source_trace),
        "cube_world_pose_size_mass_friction_unchanged": True,
        "controls_unchanged": True,
        "hand_root_fixed_during_each_simulation": True,
        "baseline_contact_centroid_cube_local_mm": {
            finger: (baseline_centroids[index] * 1000.0).tolist()
            for index, finger in enumerate(FINGERS)
        },
        "requested_target_shift_cube_y_mm": -abs(float(args.target_shift_mm)),
        "best_label": best_label,
        "best_record": best,
        "hard_pass_count": sum(bool(value["hard_pass"]) for value in records),
        "records": records,
    }
    write_json(output / "sweep_report.json", report)
    catalog = {
        "catalog_schema_version": 1,
        "catalog_kind": "thumb_negative_y_fixed_root_viewer_catalog",
        "aliases": {
            "best_nominal": best_label,
            "best_thumb_negative_y": best_label,
        },
        "trajectories": [
            {
                "label": best_label,
                "trajectory_id": best_label,
                "aliases": ["best_nominal", "best_thumb_negative_y"],
                "hard_pass": bool(best["hard_pass"]),
                "artifacts": copy.deepcopy(best["artifacts"]),
            }
        ],
    }
    write_json(output / "catalog.json", catalog)
    print(f"best={best_label} hard={best['hard_pass']}")
    print(output / "sweep_report.json")
    return 0 if best["hard_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
