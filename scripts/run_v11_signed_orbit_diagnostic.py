#!/usr/bin/env python3
"""Run signed cube-local-Z wrist-orbit diagnostics from one measured grasp.

This is deliberately a diagnostic runner, not an extension of the registered
v11 search policy.  Each generated configuration is passed through the Viewer
override path, which applies the coupled position/orientation transform from
the candidate's immutable anchor and marks the run as ``parameter_override_run``.
No source success evidence is inherited.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xhand_grasp.artifacts import file_sha256, resolved_run_config, write_json
from xhand_grasp.config import load_config
from xhand_grasp.simulation import run_simulation
from xhand_grasp.viewer import apply_viewer_overrides


DEFAULT_SOURCE = Path(
    "artifacts/left_opposed_face_palm_down_larger_relative_wrist_pose_"
    "actual_contact_smooth_vertical_lift/tune/"
    "formal_coordinated_thumb_manipulation_rescue_v1/dynamic/measured/"
    "candidate_4367930796723148114/resolved_config.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/left_opposed_face_palm_down_larger_relative_wrist_pose_"
    "actual_contact_smooth_vertical_lift/tune/"
    "negative_clockwise_orbit_sweep_from_4367930796723148114_v1"
)
DEFAULT_ANGLES = (-2.5, -5.0, -7.5, -10.0, -12.5, -15.0)


def _label(angle_deg: float) -> str:
    sign = "neg" if angle_deg < 0.0 else "pos"
    magnitude = f"{abs(angle_deg):.1f}".replace(".", "p")
    return f"orbit_{sign}_{magnitude}_deg"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _trace_inventory(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        names = tuple(sorted(payload.files))
        if not names:
            raise RuntimeError(f"trace has no arrays: {path}")
        shapes = {name: list(np.asarray(payload[name]).shape) for name in names}
    return {"array_count": len(names), "array_shapes": shapes}


def _result_record(
    *,
    index: int,
    angle_deg: float,
    label: str,
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    directory: Path,
) -> dict[str, Any]:
    metrics = _mapping(summary.get("metrics"))
    stage_status = _mapping(summary.get("stage_status"))
    actual = _mapping(metrics.get("actual_grasp_pose"))
    actual_checks = _mapping(actual.get("checks"))
    actual_metrics = _mapping(actual.get("metrics"))
    actual_events = _mapping(actual.get("events"))
    pose = _mapping(metrics.get("pose_preservation"))
    alignment = _mapping(metrics.get("contact_alignment"))
    verify_alignment = _mapping(alignment.get("verify"))
    closure = _mapping(metrics.get("closure_alignment"))
    per_finger_closure = _mapping(closure.get("per_finger"))
    relative = _mapping(metrics.get("relative_wrist_pose"))
    candidate_metadata = _mapping(config.get("candidate_metadata"))
    relative_metadata = _mapping(candidate_metadata.get("relative_wrist_pose_search"))
    run_context = _mapping(config.get("run_context"))
    all_actual_checks = bool(actual_checks) and all(
        bool(value) for value in actual_checks.values()
    )
    return {
        "index": index,
        "label": label,
        "clockwise_orbit_deg": float(angle_deg),
        "parameter_override_run": run_context.get("kind")
        == "parameter_override_run",
        "source_success_evidence_inherited": bool(
            relative_metadata.get("source_success_evidence_inherited", True)
        ),
        "coupled_transform": {
            "root_position_and_orientation_transformed_together": True,
            "clockwise_parameter_convention": "Rz(-alpha) in cube-local frame",
            "reported_clockwise_orbit_deg": relative.get(
                "clockwise_orbit_deg", angle_deg
            ),
        },
        "actual_grasp_pose_gate_passed": bool(actual.get("passed", False))
        and all_actual_checks,
        "actual_grasp_pose_checks": dict(actual_checks),
        "strict_grasp_success": bool(stage_status.get("grasp_success", False)),
        "manipulation_success": bool(
            stage_status.get("manipulation_success", False)
        ),
        "full_success": bool(stage_status.get("full_success", False)),
        "passed": bool(summary.get("passed", False)),
        "failed_checks": list(summary.get("failed_checks", [])),
        "verify_max_consecutive_all_gate_steps": int(
            metrics.get("verify_max_consecutive_all_gate_steps", 0)
        ),
        "verify_all_gate_duty": float(metrics.get("verify_all_gate_duty", 0.0)),
        "verify_gate_component_duty": dict(
            _mapping(metrics.get("verify_gate_component_duty"))
        ),
        "grasp_lock_step": int(actual_events.get("grasp_lock_step", -1)),
        "actual_thumb": dict(actual_metrics),
        "pose_preservation": dict(pose),
        "contact_alignment_verify": dict(verify_alignment),
        "closure_alignment": {
            "worst_finger": closure.get("worst_finger"),
            "worst_p95_angle_deg": closure.get("worst_p95_angle_deg"),
            "worst_minimum_inward_speed_m_s": closure.get(
                "worst_minimum_inward_speed_m_s"
            ),
            "per_finger": {
                str(name): {
                    "valid_count": _mapping(value).get("valid_count"),
                    "angle_p95_deg": _mapping(value).get("angle_p95_deg"),
                    "minimum_inward_speed_m_s": _mapping(value).get(
                        "minimum_inward_speed_m_s"
                    ),
                }
                for name, value in per_finger_closure.items()
            },
        },
        "max_penetration_m": float(metrics.get("max_penetration_m", math.inf)),
        "forbidden_contact_steps": int(
            metrics.get("forbidden_contact_steps", 0)
        ),
        "verify_peak_target_face_force_n": dict(
            _mapping(metrics.get("verify_peak_target_face_force_n"))
        ),
        "target_force_purity_min_on_contact": dict(
            _mapping(metrics.get("target_force_purity_min_on_contact"))
        ),
        "relative_wrist_pose": dict(relative),
        "hand_pose": copy.deepcopy(config["hand_pose"]),
        "artifacts": {
            "resolved_config": str(
                (directory / "resolved_config.json").as_posix()
            ),
            "result": str((directory / "result.json").as_posix()),
            "trace": str((directory / "trace.npz").as_posix()),
        },
    }


def _rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    pose = _mapping(record.get("pose_preservation"))
    alignment = _mapping(record.get("contact_alignment_verify"))

    def finite_or(value: object, fallback: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback
        return number if math.isfinite(number) else fallback

    return (
        int(bool(record.get("full_success"))),
        int(bool(record.get("strict_grasp_success"))),
        int(bool(record.get("actual_grasp_pose_gate_passed"))),
        int(record.get("verify_max_consecutive_all_gate_steps", 0)),
        float(record.get("verify_all_gate_duty", 0.0)),
        -finite_or(pose.get("max_translation_m"), math.inf),
        -finite_or(pose.get("max_orientation_drift_deg"), math.inf),
        -finite_or(alignment.get("height_spread_p95_m"), math.inf),
        float(record.get("clockwise_orbit_deg", -math.inf)),
    )


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


def _mark_parameter_override_status(
    resolved: dict[str, Any], summary: Mapping[str, Any]
) -> None:
    """Mirror Viewer live-output status without inheriting source claims."""

    stage = _mapping(summary.get("stage_status"))
    resolved["experiment_status"] = {
        "classification": "parameter_override_run",
        "passed": bool(summary.get("passed", False)),
        "hard_constraints_passed": bool(summary.get("passed", False)),
        "failed_checks": [str(value) for value in summary.get("failed_checks", [])],
        "note": (
            "This status was recomputed from a signed-orbit parameter override "
            "run; no source or catalog validation status was inherited."
        ),
        "grasp_success": bool(stage.get("grasp_success", False)),
        "manipulation_success": bool(stage.get("manipulation_success", False)),
        "full_success": bool(stage.get("full_success", False)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--angles-deg", nargs="+", type=float, default=list(DEFAULT_ANGLES)
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="authenticate and reuse complete per-angle artifacts",
    )
    args = parser.parse_args()

    source_path = args.source_config.resolve()
    output = args.output_dir.resolve()
    angles = tuple(float(value) for value in args.angles_deg)
    if not angles or not all(math.isfinite(value) for value in angles):
        raise ValueError("--angles-deg must contain finite values")
    if len(set(angles)) != len(angles):
        raise ValueError("--angles-deg values must be unique")
    if output.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {output}")

    source = load_config(source_path)
    source_config_sha = file_sha256(source_path)
    source_cube = _cube_signature(source)
    source_control = copy.deepcopy(source["control"])
    output.mkdir(parents=True, exist_ok=args.resume)
    records: list[dict[str, Any]] = []

    for index, angle in enumerate(angles, start=1):
        label = _label(angle)
        directory = output / label
        config, changed = apply_viewer_overrides(
            source, clockwise_orbit_deg=angle
        )
        if not changed:
            raise RuntimeError(f"orbit {angle} was not marked as an override")
        if _cube_signature(config) != source_cube:
            raise RuntimeError(f"orbit {angle} changed cube pose or physics")
        if config["control"] != source_control:
            raise RuntimeError(f"orbit {angle} changed controller values")
        relative = _mapping(
            _mapping(config.get("candidate_metadata")).get(
                "relative_wrist_pose_search"
            )
        )
        if relative.get("source_success_evidence_inherited") is not False:
            raise RuntimeError(f"orbit {angle} inherited source success evidence")

        trace_path = directory / "trace.npz"
        if args.resume and directory.is_dir():
            paths = tuple(
                directory / name
                for name in ("resolved_config.json", "result.json", "trace.npz")
            )
            if not all(path.is_file() for path in paths):
                raise RuntimeError(f"incomplete resume artifacts: {directory}")
            resolved = load_config(paths[0])
            payload = json.loads(paths[1].read_text(encoding="utf-8"))
            summary = payload.get("summary")
            if not isinstance(summary, Mapping):
                raise RuntimeError(f"resume result has no summary: {paths[1]}")
            if float(payload.get("angle_deg", math.nan)) != angle:
                raise RuntimeError(f"resume angle mismatch: {paths[1]}")
            if payload.get("source_config_sha256") != source_config_sha:
                raise RuntimeError(f"resume source hash mismatch: {paths[1]}")
            if _cube_signature(resolved) != source_cube:
                raise RuntimeError(f"resume orbit {angle} changed cube inputs")
            if resolved["control"] != source_control:
                raise RuntimeError(f"resume orbit {angle} changed controller values")
            _mark_parameter_override_status(resolved, summary)
            payload["run_kind"] = "parameter_override_run"
            payload["config"] = resolved
            write_json(paths[0], resolved)
            write_json(paths[1], payload)
        else:
            directory.mkdir()
            summary = run_simulation(copy.deepcopy(config), trace_path=trace_path)
            resolved = resolved_run_config(config, summary)
            _mark_parameter_override_status(resolved, summary)
            write_json(directory / "resolved_config.json", resolved)
            payload = {
                "run_kind": "parameter_override_run",
                "diagnostic_kind": "signed_clockwise_orbit_real_simulation",
                "angle_deg": angle,
                "source_candidate_id": "4367930796723148114",
                "source_config": str(source_path),
                "source_config_sha256": source_config_sha,
                "validation_scope": (
                    "fresh parameter_override_run; source success is not inherited"
                ),
                "cube_world_configuration_unchanged": True,
                "controls_unchanged": True,
                "config": resolved,
                "summary": summary,
            }
            write_json(directory / "result.json", payload)
        inventory = _trace_inventory(trace_path)
        record = _result_record(
            index=index,
            angle_deg=angle,
            label=label,
            summary=summary,
            config=resolved,
            directory=Path(label),
        )
        record["trace_inventory"] = inventory
        record["artifacts"]["sha256"] = {
            name: file_sha256(directory / filename)
            for name, filename in (
                ("resolved_config", "resolved_config.json"),
                ("result", "result.json"),
                ("trace", "trace.npz"),
            )
        }
        records.append(record)

    best = max(records, key=_rank)
    report = {
        "diagnostic_schema_version": 1,
        "diagnostic_kind": "negative_clockwise_orbit_real_simulation_sweep",
        "status": "complete_and_integrity_verified",
        "angles_deg": list(angles),
        "source_candidate_id": "4367930796723148114",
        "source_config": str(source_path),
        "source_config_sha256": source_config_sha,
        "cube_world_pose_size_mass_friction_unchanged": True,
        "controls_unchanged": True,
        "coupled_transform": True,
        "signed_convention": (
            "clockwise parameter alpha uses cube-local Rz(-alpha); negative "
            "alpha therefore rotates counter-clockwise from cube-local +Z"
        ),
        "parameter_override_only": True,
        "source_success_evidence_inherited": False,
        "strict_grasp_success_count": sum(
            bool(value["strict_grasp_success"]) for value in records
        ),
        "actual_grasp_pose_gate_pass_count": sum(
            bool(value["actual_grasp_pose_gate_passed"]) for value in records
        ),
        "full_success_count": sum(bool(value["full_success"]) for value in records),
        "best_angle_deg": best["clockwise_orbit_deg"],
        "best_label": best["label"],
        "records": records,
    }
    write_json(output / "sweep_report.json", report)
    report["sweep_report_sha256"] = file_sha256(output / "sweep_report.json")
    write_json(output / "sweep_summary.json", report)
    print(output / "sweep_summary.json")
    print(
        f"best={best['clockwise_orbit_deg']} "
        f"actual_gate={best['actual_grasp_pose_gate_passed']} "
        f"strict_grasp={best['strict_grasp_success']} "
        f"full={best['full_success']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
