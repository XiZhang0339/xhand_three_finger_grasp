#!/usr/bin/env python3
"""Deterministic diagnostic scan for upright index/middle v14 grasps.

This is deliberately an evidence-only scanner: generated configurations have
their publication identities removed and are never written as resolved run
artifacts.  A winning point must be finalized through the authenticated v14
candidate publisher before it can enter a Viewer catalog.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from xhand_grasp.config import ACTIVE_ACTUATORS
from xhand_grasp.experiment import ManipulationPlanParameters
from xhand_grasp.simulation import SimulationSession
from xhand_grasp.tuning.contact_preserving_time_warp import warp_knot_times


INDEX_BEND = "left_hand_index_bend_joint_actuator"
INDEX_JOINT2 = "left_hand_index_joint2_actuator"
MIDDLE_JOINT2 = "left_hand_mid_joint2_actuator"
IDENTITY_FIELDS = (
    "object_config_id",
    "grasp_pose_id",
    "grasp_object_pair_id",
    "planner_id",
    "controller_id",
)


def _numbers(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result or not np.isfinite(result).all():
        raise argparse.ArgumentTypeError("expected a comma-separated finite number list")
    return result


def _warps(value: str) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for item in value.split(","):
        left, separator, right = item.partition(":")
        if not separator:
            raise argparse.ArgumentTypeError("time warps must use a1:a2")
        result.append((float(left), float(right)))
    if not result or not np.isfinite(result).all():
        raise argparse.ArgumentTypeError("expected finite time-warp coefficients")
    return tuple(result)


def _interpolate_mapping(
    first: dict[str, float], second: dict[str, float], alpha: float
) -> dict[str, float]:
    if set(first) != set(second):
        raise ValueError("pose seeds disagree on actuator names")
    return {
        name: (1.0 - alpha) * float(first[name]) + alpha * float(second[name])
        for name in first
    }


def build_diagnostic_config(
    base: dict[str, Any],
    pose_seed: dict[str, Any],
    *,
    pose_alpha: float,
    index_bend_scale: float,
    index_joint2_scale: float,
    middle_joint2_scale: float,
    time_warp: tuple[float, float],
) -> dict[str, Any]:
    """Blend grasp geometry and alter only the three named lift profiles."""

    config = copy.deepcopy(base)
    alpha = float(pose_alpha)
    for key in ("translation_m", "rpy_deg"):
        first = np.asarray(base["hand_pose"][key], dtype=np.float64)
        second = np.asarray(pose_seed["hand_pose"][key], dtype=np.float64)
        config["hand_pose"][key] = ((1.0 - alpha) * first + alpha * second).tolist()
    config["grasp_pose"]["nominal_joint_qpos_rad"] = _interpolate_mapping(
        base["grasp_pose"]["nominal_joint_qpos_rad"],
        pose_seed["grasp_pose"]["nominal_joint_qpos_rad"],
        alpha,
    )
    for key in ("precontact_targets_rad", "contact_preload_targets_rad"):
        config["control"][key] = _interpolate_mapping(
            base["control"][key], pose_seed["control"][key], alpha
        )

    old = ManipulationPlanParameters.from_config(config["manipulation_plan"])
    scales = {
        INDEX_BEND: float(index_bend_scale),
        INDEX_JOINT2: float(index_joint2_scale),
        MIDDLE_JOINT2: float(middle_joint2_scale),
    }
    waypoints = {
        name: tuple(float(value) * scales.get(name, 1.0) for value in old.actuator_waypoints_rad[name])
        for name in ACTIVE_ACTUATORS
    }
    plan = ManipulationPlanParameters(
        schema_version=old.schema_version,
        profile=old.profile,
        duration_s=old.duration_s,
        knot_times_s=tuple(
            float(value)
            for value in warp_knot_times(old.knot_times_s, *time_warp)
        ),
        actuator_waypoints_rad=waypoints,
        desired_cube_position_delta_m=old.desired_cube_position_delta_m,
        desired_cube_rotation_vector_rad=old.desired_cube_rotation_vector_rad,
        max_knot_delta_rad=old.max_knot_delta_rad,
        trust_region_backtracks=old.trust_region_backtracks,
    )
    config["manipulation_plan"] = plan.as_config()
    config["control"]["manipulation_delta_rad"] = {
        name: float(waypoints[name][-1]) for name in ACTIVE_ACTUATORS
    }
    for name in IDENTITY_FIELDS:
        config.pop(name, None)
    config["candidate_metadata"] = {
        "diagnostic_upright_grasp_scan": {
            "pose_alpha": alpha,
            "index_bend_scale": float(index_bend_scale),
            "index_joint2_scale": float(index_joint2_scale),
            "middle_joint2_scale": float(middle_joint2_scale),
            "time_warp": [float(value) for value in time_warp],
            "cube_pose_sampled": False,
            "publication_forbidden": True,
        }
    }
    return config


def _evaluate(task: tuple[int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    sequence, descriptor, config = task
    session = SimulationSession(config)
    collision_frames = 0
    first_collision_step: int | None = None
    maximum_collision_force_n = 0.0
    contact_force = np.zeros(6, dtype=np.float64)
    try:
        while not session.complete:
            session.advance_one()
            step = session.step_index - 1
            frame_colliding = False
            for contact_index in range(session.data.ncon):
                contact = session.data.contact[contact_index]
                body1 = int(session.model.geom_bodyid[int(contact.geom1)])
                body2 = int(session.model.geom_bodyid[int(contact.geom2)])
                part1 = session.info.hand_body_parts.get(body1)
                part2 = session.info.hand_body_parts.get(body2)
                if {part1, part2} != {"index", "mid"} or int(contact.efc_address) < 0:
                    continue
                mujoco.mj_contactForce(
                    session.model, session.data, contact_index, contact_force
                )
                force = max(0.0, float(contact_force[0]))
                if force > 1e-8:
                    frame_colliding = True
                    maximum_collision_force_n = max(maximum_collision_force_n, force)
            if frame_colliding:
                collision_frames += 1
                if first_collision_step is None:
                    first_collision_step = step
        summary = session.finalize()
    finally:
        session.close()

    metrics = summary["metrics"]
    smooth = metrics["motion_smoothness"]
    contact = metrics["contact_preserving_planned_lift"]
    closure = metrics["closure_alignment"]["per_finger"]
    return {
        "sequence": sequence,
        **descriptor,
        "grasp_success": bool(summary["stage_status"]["grasp_success"]),
        "legacy_full_success": bool(summary["stage_status"]["full_success"]),
        "failed_checks": list(summary["failed_checks"]),
        "self_collision_free": collision_frames == 0,
        "index_middle_collision_frames": collision_frames,
        "first_collision_step": first_collision_step,
        "maximum_collision_force_n": maximum_collision_force_n,
        "simultaneous_contact_duty": float(
            contact["simultaneous_target_face_effective_duty"]
        ),
        "longest_contact_loss_s": float(
            contact["simultaneous_longest_contact_loss_s"]
        ),
        "operation_aborted": bool(contact["operation_aborted"]),
        "median_lift_m": float(metrics["operation_median_lift_m"]),
        "minimum_lift_m": float(metrics["operation_minimum_lift_m"]),
        "lateral_displacement_m": float(
            smooth["operation_max_lateral_displacement_m"]
        ),
        "orientation_drift_deg": float(
            smooth["operation_max_orientation_drift_deg"]
        ),
        "peak_jerk_m_s3": float(
            smooth["operation_peak_abs_filtered_jerk_m_s3"]
        ),
        "closure_p95_deg": {
            finger: float(closure[finger]["angle_p95_deg"])
            for finger in ("thumb", "index", "mid")
        },
    }


def _rank(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(record["grasp_success"]),
        not bool(record["self_collision_free"]),
        -float(record["simultaneous_contact_duty"]),
        float(record["longest_contact_loss_s"]),
        max(0.010 - float(record["minimum_lift_m"]), 0.0),
        float(record["peak_jerk_m_s3"]),
        max(float(record["closure_p95_deg"][name]) for name in ("index", "mid")),
        int(record["sequence"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--pose-seed-config", type=Path, required=True)
    parser.add_argument("--pose-alphas", type=_numbers, default=(0.0, 0.5, 1.0))
    parser.add_argument("--index-bend-scales", type=_numbers, default=(0.25, 0.3, 0.35, 0.4))
    parser.add_argument("--index-joint2-scales", type=_numbers, default=(1.0,))
    parser.add_argument("--middle-joint2-scales", type=_numbers, default=(1.0,))
    parser.add_argument("--time-warps", type=_warps, default=((0.0, 0.0),))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    pose_seed = json.loads(args.pose_seed_config.read_text(encoding="utf-8"))
    tasks = []
    grid = itertools.product(
        args.pose_alphas,
        args.index_bend_scales,
        args.index_joint2_scales,
        args.middle_joint2_scales,
        args.time_warps,
    )
    for sequence, (alpha, bend, index2, middle2, warp) in enumerate(grid):
        descriptor = {
            "pose_alpha": alpha,
            "index_bend_scale": bend,
            "index_joint2_scale": index2,
            "middle_joint2_scale": middle2,
            "time_warp": list(warp),
        }
        tasks.append(
            (
                sequence,
                descriptor,
                build_diagnostic_config(
                    base,
                    pose_seed,
                    pose_alpha=alpha,
                    index_bend_scale=bend,
                    index_joint2_scale=index2,
                    middle_joint2_scale=middle2,
                    time_warp=warp,
                ),
            )
        )
    if args.workers == 1:
        records = [_evaluate(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            records = list(executor.map(_evaluate, tasks))
    records.sort(key=_rank)
    payload = {"candidate_count": len(records), "records": records}
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
