#!/usr/bin/env python3
"""Deterministic full-reset search over v14 manipulation time profiles.

This diagnostic keeps the grasp/object pair, terminal actuator command,
feedback gains, mass, friction, and cube pose fixed.  It only redistributes
the twenty bounded actuator increments in time.  The search is useful for
crossing collision-witness transitions at lower speed without weakening any
acceptance threshold.  Every candidate is persisted and rerun from the
initial no-contact state.
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

from scripts.v14_waypoint_shape_grid import _worker
from xhand_grasp.artifacts import write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.experiment import ManipulationPlanParameters


def _project_capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto ``sum(x)=1, 0<=x<=cap``."""

    if values.ndim != 1 or values.size == 0:
        raise ValueError("profile increments must be a non-empty vector")
    if cap * values.size < 1.0 - 1e-12:
        raise ValueError("per-knot cap cannot reach the terminal command")
    lo = float(np.min(values) - cap)
    hi = float(np.max(values))
    for _ in range(100):
        midpoint = 0.5 * (lo + hi)
        projected = np.clip(values - midpoint, 0.0, cap)
        if float(np.sum(projected)) > 1.0:
            lo = midpoint
        else:
            hi = midpoint
    result = np.clip(values - 0.5 * (lo + hi), 0.0, cap)
    result /= float(np.sum(result))
    if float(np.max(result)) > cap + 2e-12:
        raise RuntimeError("capped-simplex projection violated its cap")
    return result


def _profile(
    base: np.ndarray,
    sample: np.ndarray,
    *,
    cap: float,
) -> np.ndarray:
    """Create a monotone profile with local slow zones at observed events."""

    segment_u = (np.arange(base.size, dtype=np.float64) + 0.5) / base.size
    weights = base.copy()
    # Full-reset traces place the dominant impulses near controller progress
    # 0.083, 0.464, and 0.709.  Positive coefficients reduce local speed;
    # low-frequency tilt terms redistribute the saved progress smoothly.
    centres = (0.083, 0.464, 0.709)
    widths = (0.050, 0.075, 0.075)
    for index, (centre, width) in enumerate(zip(centres, widths, strict=True)):
        gaussian = np.exp(-0.5 * ((segment_u - centre) / width) ** 2)
        weights *= 1.0 - float(sample[index]) * gaussian
    weights *= np.exp(
        float(sample[3]) * np.sin(2.0 * np.pi * segment_u)
        + float(sample[4]) * np.sin(4.0 * np.pi * segment_u)
    )
    weights = np.maximum(weights, 1e-8)
    increments = _project_capped_simplex(weights, cap)
    progress = np.concatenate(([0.0], np.cumsum(increments)))
    progress[-1] = 1.0
    return progress


def _materialize(
    source: dict[str, Any],
    sample: np.ndarray,
    sample_index: int,
) -> dict[str, Any]:
    config = copy.deepcopy(source)
    terminal = {
        name: float(config["control"]["manipulation_delta_rad"][name])
        for name in ACTIVE_ACTUATORS
    }
    largest = max(abs(value) for value in terminal.values())
    cap = float(config["manipulation_plan"]["max_knot_delta_rad"]) / largest
    old_waypoints = config["manipulation_plan"]["actuator_waypoints_rad"]
    largest_name = max(ACTIVE_ACTUATORS, key=lambda name: abs(terminal[name]))
    previous = np.asarray(old_waypoints[largest_name], dtype=np.float64)
    base = np.abs(np.diff(previous / terminal[largest_name]))
    base /= float(np.sum(base))
    progress = _profile(base, sample, cap=cap)
    old = config["manipulation_plan"]
    plan = ManipulationPlanParameters(
        schema_version=1,
        profile=str(old["profile"]),
        duration_s=float(old["duration_s"]),
        knot_times_s=tuple(float(value) for value in old["knot_times_s"]),
        actuator_waypoints_rad={
            name: tuple(float(terminal[name] * value) for value in progress)
            for name in ACTIVE_ACTUATORS
        },
        desired_cube_position_delta_m=tuple(
            (0.0, 0.0, 0.011 * float(value)) for value in progress
        ),
        desired_cube_rotation_vector_rad=tuple(
            (0.0, 0.0, 0.0) for _ in progress
        ),
        max_knot_delta_rad=float(old["max_knot_delta_rad"]),
        trust_region_backtracks=int(old["trust_region_backtracks"]),
    )
    config["manipulation_plan"] = plan.as_config()
    config.setdefault("candidate_metadata", {})["progress_profile_probe"] = {
        "schema_version": 1,
        "sample_index": int(sample_index),
        "slow_zone_coefficients": [float(value) for value in sample[:3]],
        "redistribution_coefficients": [float(value) for value in sample[3:]],
        "progress": [float(value) for value in progress],
        "terminal_unchanged": True,
        "requires_full_reset_rerun": True,
    }
    validate_config(config)
    return config


def _samples(count: int, seed: int) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    lhs = np.empty((count, 5), dtype=np.float64)
    for column in range(lhs.shape[1]):
        lhs[:, column] = (rng.permutation(count) + rng.random(count)) / count
    # Slow-zone depth 0..0.9; signed smooth redistribution -0.45..0.45.
    lhs[:, :3] *= 0.9
    lhs[:, 3:] = 0.9 * (lhs[:, 3:] - 0.5)
    presets = np.asarray(
        [
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.90, 0.90, 0.0, 0.0),
            (0.0, 0.97, 0.90, 0.0, 0.0),
            (0.0, 0.90, 0.97, 0.0, 0.0),
            (0.0, 0.97, 0.97, 0.0, 0.0),
            (0.0, 0.995, 0.97, 0.0, 0.0),
            (0.0, 0.97, 0.995, 0.0, 0.0),
            (0.0, 0.995, 0.995, 0.0, 0.0),
            (0.0, 0.97, 0.97, -0.20, 0.20),
            (0.0, 0.97, 0.97, 0.20, -0.20),
            (0.0, 0.995, 0.995, -0.30, 0.0),
            (0.0, 0.995, 0.995, 0.30, 0.0),
        ],
        dtype=np.float64,
    )
    lhs[: min(count, presets.shape[0])] = presets[:count]
    return lhs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    source = load_config(source_path)
    output = Path(args.output).resolve()
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    payloads: list[tuple[str, str]] = []
    for sample_index, sample in enumerate(_samples(args.count, args.seed)):
        config = _materialize(source, sample, sample_index)
        config_path = inputs / f"candidate_{sample_index:04d}.json"
        write_json(config_path, config)
        payloads.append(
            (str(config_path), str(output / "runs" / f"candidate_{sample_index:04d}"))
        )

    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(
        key=lambda value: (
            len(value["failed_checks"]),
            float(value["jerk_m_s3"]),
            -float(value["simultaneous_duty"]),
            value["output"],
        )
    )
    report = {
        "schema_version": 1,
        "source": str(source_path),
        "count": int(args.count),
        "seed": int(args.seed),
        "records": records,
    }
    write_json(output / "report.json", report)
    print(json.dumps({"best": records[:10], "count": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
