#!/usr/bin/env python3
"""Full-reset local waypoint-shaping probes for a resolved v14 plan.

The terminal command, object, grasp pose, and feedback parameters remain
fixed.  Only small interior waypoint changes are applied around observed
contact-mode transitions.  The index-joint2 experiments are local at the
first three knots; its nearly saturated central knot increments are untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhand_grasp.artifacts import write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.experiment import ManipulationPlanParameters
from xhand_grasp.simulation import run_simulation


INDEX_J1 = "left_hand_index_joint1_actuator"
INDEX_J2 = "left_hand_index_joint2_actuator"
MID_J1 = "left_hand_mid_joint1_actuator"
MID_J2 = "left_hand_mid_joint2_actuator"


def _install_plan(config: dict[str, Any], waypoints: dict[str, np.ndarray]) -> None:
    old = config["manipulation_plan"]
    plan = ManipulationPlanParameters(
        schema_version=1,
        profile=str(old["profile"]),
        duration_s=float(old["duration_s"]),
        knot_times_s=tuple(float(value) for value in old["knot_times_s"]),
        actuator_waypoints_rad={
            name: tuple(float(value) for value in waypoints[name])
            for name in ACTIVE_ACTUATORS
        },
        desired_cube_position_delta_m=tuple(
            tuple(float(axis) for axis in value)
            for value in old["desired_cube_position_delta_m"]
        ),
        desired_cube_rotation_vector_rad=tuple(
            tuple(float(axis) for axis in value)
            for value in old["desired_cube_rotation_vector_rad"]
        ),
        max_knot_delta_rad=float(old["max_knot_delta_rad"]),
        trust_region_backtracks=int(old["trust_region_backtracks"]),
    )
    config["manipulation_plan"] = plan.as_config()


def _bump(values: np.ndarray, centre: int, amount: float) -> None:
    values[centre - 1 : centre + 2] += amount * np.asarray((0.5, 1.0, 0.5))


def _variant(
    source: dict[str, Any],
    *,
    early_index_j2_scale: float = 1.0,
    index_j1_bump_rad: float = 0.0,
    mid_j1_bump_rad: float = 0.0,
    mid_j2_bump_rad: float = 0.0,
) -> dict[str, Any]:
    config = copy.deepcopy(source)
    waypoints = {
        name: np.asarray(
            config["manipulation_plan"]["actuator_waypoints_rad"][name],
            dtype=np.float64,
        ).copy()
        for name in ACTIVE_ACTUATORS
    }
    # The first observed impulse occurs while leaving support at progress
    # ~0.083.  Scale only knots 1--3 and return to the original curve at knot
    # 4, leaving every near-limit central index-j2 increment untouched.
    waypoints[INDEX_J2][1:4] *= float(early_index_j2_scale)
    _bump(waypoints[INDEX_J1], 9, float(index_j1_bump_rad))
    _bump(waypoints[MID_J1], 14, float(mid_j1_bump_rad))
    _bump(waypoints[MID_J2], 14, float(mid_j2_bump_rad))
    _install_plan(config, waypoints)
    descriptor = {
        "schema_version": 1,
        "early_index_j2_scale": float(early_index_j2_scale),
        "index_j1_bump_rad": float(index_j1_bump_rad),
        "mid_j1_bump_rad": float(mid_j1_bump_rad),
        "mid_j2_bump_rad": float(mid_j2_bump_rad),
        "terminal_unchanged": True,
        "requires_full_reset_rerun": True,
    }
    config.setdefault("candidate_metadata", {})["waypoint_shape_probe"] = descriptor
    validate_config(config)
    return config


def _worker(payload: tuple[str, str]) -> dict[str, Any]:
    config_path, destination_path = payload
    destination = Path(destination_path)
    destination.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    write_json(destination / "resolved_config.json", config)
    result = run_simulation(config, trace_path=destination / "trace.npz")
    write_json(destination / "summary.json", result)
    metrics = result["metrics"]
    smooth = metrics["motion_smoothness"]
    contact = metrics["contact_preserving_planned_lift"]
    alignment = metrics["contact_alignment"]["operation"]
    return {
        "output": str(destination),
        "shape": config.get("candidate_metadata", {}).get(
            "waypoint_shape_probe",
            config.get("candidate_metadata", {}).get("progress_profile_probe", {}),
        ),
        "aborted": bool(metrics["controller_aborted"]),
        "simultaneous_duty": float(contact["simultaneous_target_face_effective_duty"]),
        "per_finger_duty": contact["target_face_effective_duty"],
        "median_lift_m": float(metrics["operation_median_lift_m"]),
        "minimum_lift_m": float(metrics["operation_minimum_lift_m"]),
        "lateral_m": float(smooth["operation_max_lateral_displacement_m"]),
        "orientation_deg": float(smooth["operation_max_orientation_drift_deg"]),
        "alignment_duty": float(alignment["aligned_duty"]),
        "jerk_m_s3": float(smooth["operation_peak_abs_filtered_jerk_m_s3"]),
        "failed_checks": result["failed_checks"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    source_path = Path(args.source).resolve()
    source = load_config(source_path)
    output = Path(args.output).resolve()
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    specifications = [
        ("base", {}),
        ("early_050", {"early_index_j2_scale": 0.50}),
        ("early_065", {"early_index_j2_scale": 0.65}),
        ("early_080", {"early_index_j2_scale": 0.80}),
        ("early_120", {"early_index_j2_scale": 1.20}),
        ("index_bump_neg", {"index_j1_bump_rad": -0.004}),
        ("index_bump_pos", {"index_j1_bump_rad": 0.004}),
        ("mid_j1_bump_neg", {"mid_j1_bump_rad": -0.004}),
        ("mid_j1_bump_pos", {"mid_j1_bump_rad": 0.004}),
        ("mid_j2_bump_neg", {"mid_j2_bump_rad": -0.004}),
        ("mid_j2_bump_pos", {"mid_j2_bump_rad": 0.004}),
    ]
    payloads = []
    for label, parameters in specifications:
        config = _variant(source, **parameters)
        config_path = inputs / f"{label}.json"
        write_json(config_path, config)
        payloads.append((str(config_path), str(output / "runs" / label)))

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: record["output"])
    report = {"schema_version": 1, "source": str(source_path), "records": records}
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
