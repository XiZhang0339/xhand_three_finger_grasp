#!/usr/bin/env python3
"""Generate deterministic v14 warm starts around the 79 mm v13 Pareto pair.

The grasp-local response is effectively zero while the cube remains supported,
so an all-zero checkpoint Jacobian cannot discover a useful lift.  This helper
materializes bounded, hash-addressed plan seeds around the two authenticated
79 mm terminals that already leave the support.  It only writes configs and a
manifest; it never runs MuJoCo and never mutates the v13 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v14_rescue_sweep import _v14_config
from xhand_grasp.artifacts import write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.experiment import ManipulationPlanParameters, get_experiment


DEFAULT_SOURCES = (
    "artifacts/left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift/"
    "tune/campaign_yaw_bounded_v2/manipulation/local_refinement/"
    "set_ee1d71c60ce224cf/candidates/candidate_109000000000008/resolved_config.json",
    "artifacts/left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift/"
    "tune/campaign_yaw_bounded_v2/manipulation/local_refinement/"
    "set_ee1d71c60ce224cf/candidates/candidate_109000000000009/resolved_config.json",
)

# Deterministic full-reset refinement of DEFAULT_SOURCES[0] (the authenticated
# 79 mm c008 source).  This is a command endpoint, not checkpoint evidence.
# It must always be materialized through ``_v14_config`` and rerun from the
# initial no-contact state before it can be ranked or published.
C3_TERMINAL_RAD = {
    "left_hand_index_bend_joint_actuator": -0.05,
    "left_hand_index_joint1_actuator": -0.1095,
    "left_hand_index_joint2_actuator": 0.535,
    "left_hand_mid_joint1_actuator": -0.09743618044100734,
    "left_hand_mid_joint2_actuator": 0.27082973148714146,
    "left_hand_thumb_bend_joint_actuator": -0.011031995765187281,
    "left_hand_thumb_rota_joint1_actuator": -0.09,
    "left_hand_thumb_rota_joint2_actuator": 0.46,
}


def c3_warm_start_terminal() -> dict[str, float]:
    """Return an isolated copy of the authenticated-c008 C3 endpoint."""

    return {name: float(C3_TERMINAL_RAD[name]) for name in ACTIVE_ACTUATORS}


def _canonical_id(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _minimum_jerk(value: np.ndarray) -> np.ndarray:
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _bounded_progress(terminal: float, knot_count: int, limit: float) -> np.ndarray:
    """Return a symmetric monotone S-curve whose command increments are bounded."""

    if abs(terminal) <= 1e-15:
        return np.zeros(knot_count, dtype=np.float64)
    u = np.linspace(0.0, 1.0, knot_count)
    increments = np.diff(3.0 * u**2 - 2.0 * u**3)
    maximum_progress_increment = limit / abs(terminal)
    if maximum_progress_increment * (knot_count - 1) < 1.0 - 1e-12:
        raise ValueError("terminal cannot satisfy the per-knot command limit")
    # Iteratively cap the largest increments and distribute their mass over the
    # remaining symmetric slots.  The result sums exactly to one.
    for _ in range(knot_count):
        over = increments > maximum_progress_increment + 1e-15
        if not np.any(over):
            break
        excess = float(np.sum(increments[over] - maximum_progress_increment))
        increments[over] = maximum_progress_increment
        free = ~over
        room = np.maximum(0.0, maximum_progress_increment - increments[free])
        if float(np.sum(room)) <= 0.0:
            raise ValueError("failed to redistribute bounded progress")
        increments[free] += excess * room / float(np.sum(room))
    progress = np.r_[0.0, np.cumsum(increments)]
    progress[-1] = 1.0
    return progress


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=np.float64)
    for column in range(dimensions):
        order = rng.permutation(count)
        result[:, column] = (order + rng.random(count)) / count
    return result


def _terminal_from_sample(source: dict[str, Any], sample: np.ndarray) -> dict[str, float]:
    terminal = {
        name: float(source["control"]["manipulation_delta_rad"][name])
        for name in ACTIVE_ACTUATORS
    }
    index_joint2 = 0.48 + 0.08 * sample[0]
    thumb_rota2 = 0.42 + 0.10 * sample[1]
    # The c008 full-reset contact-centroid probes measured approximately
    # dz_index/d(index_j2)=-32.23 and dz_index/d(index_j1)=-75.01 mm/rad.
    # This coupled term prevents the enlarged index-j2 lift command from simply
    # rolling the index witness off its tactile pad.
    index_joint1 = terminal["left_hand_index_joint1_actuator"] - 0.43 * (
        index_joint2 - terminal["left_hand_index_joint2_actuator"]
    ) + (-0.012 + 0.024 * sample[2])
    terminal.update(
        {
            "left_hand_index_bend_joint_actuator": -0.055 + 0.085 * sample[3],
            "left_hand_index_joint1_actuator": index_joint1,
            "left_hand_index_joint2_actuator": index_joint2,
            "left_hand_mid_joint1_actuator": terminal[
                "left_hand_mid_joint1_actuator"
            ] + (-0.004 + 0.018 * sample[4]),
            "left_hand_mid_joint2_actuator": terminal[
                "left_hand_mid_joint2_actuator"
            ] + (-0.025 + 0.075 * sample[5]),
            "left_hand_thumb_bend_joint_actuator": terminal[
                "left_hand_thumb_bend_joint_actuator"
            ] + (-0.015 + 0.030 * sample[6]),
            "left_hand_thumb_rota_joint1_actuator": terminal[
                "left_hand_thumb_rota_joint1_actuator"
            ] + 0.012 * sample[7],
            "left_hand_thumb_rota_joint2_actuator": thumb_rota2,
        }
    )
    return terminal


def _materialize(
    source: dict[str, Any], template: dict[str, Any], terminal: dict[str, float]
) -> dict[str, Any]:
    config = _v14_config(source, template)
    experiment = get_experiment(config["experiment_id"])
    bounds = experiment.search_bounds.manipulation_delta_rad
    assert bounds is not None
    for name, value in terminal.items():
        lower, upper = bounds[name]
        if not lower - 1e-12 <= value <= upper + 1e-12:
            raise ValueError(f"{name}={value} is outside the registered v14 range")
    times = np.asarray(config["manipulation_plan"]["knot_times_s"], dtype=np.float64)
    desired_progress = _minimum_jerk(times / times[-1])
    plan = ManipulationPlanParameters(
        schema_version=1,
        profile="piecewise_quintic_minimum_jerk",
        duration_s=float(times[-1]),
        knot_times_s=tuple(float(value) for value in times),
        actuator_waypoints_rad={
            name: tuple(
                float(terminal[name]) * value
                for value in _bounded_progress(float(terminal[name]), len(times), 0.04)
            )
            for name in ACTIVE_ACTUATORS
        },
        desired_cube_position_delta_m=tuple(
            (0.0, 0.0, 0.011 * float(value)) for value in desired_progress
        ),
        desired_cube_rotation_vector_rad=tuple((0.0, 0.0, 0.0) for _ in times),
        max_knot_delta_rad=0.04,
        trust_region_backtracks=4,
    )
    config["manipulation_plan"] = plan.as_config()
    config["control"]["manipulation_delta_rad"] = dict(terminal)
    config.setdefault("candidate_metadata", {})["v14_warm_start"] = {
        "schema_version": 1,
        "kind": "authenticated_79mm_terminal_contact_centroid_compensated",
        "requires_full_reset_rerun": True,
        "checkpoint_is_success_evidence": False,
        "requested_local_pose_refinement": {
            "root_delta_cube_m": [-0.0005, 0.0005],
            "wrist_local_rotvec_deg": [-0.5, 0.5],
            "must_refinalize_actual_grasp_qpos": True,
        },
    }
    validate_config(config)
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        default="grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json",
    )
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--count-per-source", type=int, default=128)
    parser.add_argument(
        "--mode",
        choices=("lhs", "c3", "c3-plus-lhs"),
        default="lhs",
        help=(
            "Generate only deterministic LHS seeds, the exact c008 C3 "
            "full-reset seed, or both."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sources = tuple(args.source) if args.source else DEFAULT_SOURCES
    template = load_config(args.template)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, Any]] = []
    if args.count_per_source <= 0:
        raise ValueError("--count-per-source must be positive")
    for source_index, source_path in enumerate(sources):
        source = load_config(source_path)
        terminals: list[tuple[str, int, dict[str, float]]] = []
        if args.mode in ("lhs", "c3-plus-lhs"):
            samples = _latin_hypercube(
                args.count_per_source, 8, args.seed + 1009 * source_index
            )
            terminals.extend(
                ("lhs", sample_index, _terminal_from_sample(source, sample))
                for sample_index, sample in enumerate(samples)
            )
        if args.mode in ("c3", "c3-plus-lhs") and source_index == 0:
            terminals.insert(0, ("c3_exact", -1, c3_warm_start_terminal()))
        for preset, sample_index, terminal in terminals:
            config = _materialize(source, template, terminal)
            config["candidate_metadata"]["v14_warm_start"].update(
                {
                    "preset": preset,
                    "source_family": "authenticated_79mm_c008"
                    if source_index == 0
                    else "authenticated_79mm_c009",
                }
            )
            candidate_id = _canonical_id(
                {
                    "source": str(source_path),
                    "preset": preset,
                    "terminal": terminal,
                    "plan_id": config["manipulation_plan"]["plan_id"],
                }
            )
            destination = output / f"candidate_{candidate_id[:16]}"
            destination.mkdir(parents=True, exist_ok=False)
            write_json(destination / "resolved_config.json", config)
            manifest_records.append(
                {
                    "candidate_id": candidate_id,
                    "source_config": str(source_path),
                    "preset": preset,
                    "sample_index": sample_index,
                    "resolved_config": str(destination / "resolved_config.json"),
                    "terminal_rad": terminal,
                    "plan_id": config["manipulation_plan"]["plan_id"],
                }
            )
    manifest = {
        "schema_version": 1,
        "complete": True,
        "seed": args.seed,
        "count": len(manifest_records),
        "records": sorted(manifest_records, key=lambda value: value["candidate_id"]),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "count": len(manifest_records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
