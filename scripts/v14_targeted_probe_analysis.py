#!/usr/bin/env python3
"""Full-reset local terminal probes for a schema-v14 source pair.

This is an analysis helper, not campaign evidence.  It converts one retained
v13 manipulation config to the v14 controller, perturbs one terminal command
at a time, and persists fresh traces so a bounded local object-response model
can be fitted without relying on compacted v13 traces.
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

from scripts.v14_rescue_sweep import _v14_config
from scripts.v14_warm_start_grid import _bounded_progress
from xhand_grasp.artifacts import write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.experiment import ManipulationPlanParameters, get_experiment
from xhand_grasp.simulation import run_simulation


def _minimum_jerk(value: float) -> float:
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _set_terminal(config: dict[str, Any], terminal: np.ndarray) -> None:
    times = tuple(float(value) for value in config["manipulation_plan"]["knot_times_s"])
    desired_progress = tuple(_minimum_jerk(value / times[-1]) for value in times)
    plan = ManipulationPlanParameters(
        schema_version=1,
        profile="piecewise_quintic_minimum_jerk",
        duration_s=times[-1],
        knot_times_s=times,
        actuator_waypoints_rad={
            name: tuple(
                float(terminal[index]) * value
                for value in _bounded_progress(
                    float(terminal[index]), len(times), 0.04
                )
            )
            for index, name in enumerate(ACTIVE_ACTUATORS)
        },
        desired_cube_position_delta_m=tuple(
            (0.0, 0.0, 0.011 * value) for value in desired_progress
        ),
        desired_cube_rotation_vector_rad=tuple(
            (0.0, 0.0, 0.0) for _ in desired_progress
        ),
        max_knot_delta_rad=0.04,
        trust_region_backtracks=4,
    )
    config["manipulation_plan"] = plan.as_config()
    config["control"]["manipulation_delta_rad"] = {
        name: float(terminal[index]) for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    validate_config(config)


def _worker(payload: tuple[str, str, str, list[float]]) -> dict[str, Any]:
    source_path, template_path, output_path, terminal_values = payload
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    source = load_config(source_path)
    if int(source.get("schema_version", 0)) == 14:
        config = copy.deepcopy(source)
    else:
        config = _v14_config(source, load_config(template_path))
    terminal = np.asarray(terminal_values, dtype=np.float64)
    _set_terminal(config, terminal)
    write_json(output / "resolved_config.json", config)
    result = run_simulation(config, trace_path=output / "trace.npz")
    write_json(output / "summary.json", result)
    trace = np.load(output / "trace.npz", allow_pickle=False)
    lock = int(np.asarray(trace["grasp_lock_step"]).item())
    end = int(np.asarray(trace["manipulation_end_step"]).item())
    if lock >= 0 and end >= lock:
        delta = np.asarray(trace["cube_pos"][end] - trace["cube_pos"][lock])
    else:
        delta = np.full(3, np.nan)
    metrics = result.get("metrics", {})
    smooth = metrics.get("motion_smoothness", {})
    record = {
        "output": str(output),
        "terminal_rad": {name: float(terminal[index]) for index, name in enumerate(ACTIVE_ACTUATORS)},
        "grasp_lock_step": lock,
        "manipulation_end_step": end,
        "object_terminal_delta_m": delta.tolist(),
        "aborted": bool(metrics.get("controller_aborted", True)),
        "simultaneous_duty": float(metrics.get("operation_target_face_simultaneous_duty", 0.0)),
        "per_finger_duty": metrics.get("operation_target_face_contact_duty", {}),
        "median_lift_m": float(metrics.get("operation_median_lift_m", -1.0)),
        "minimum_lift_m": float(metrics.get("operation_minimum_lift_m", -1.0)),
        "lateral_m": float(smooth.get("operation_max_lateral_displacement_m", np.inf)),
        "orientation_deg": float(smooth.get("operation_max_orientation_drift_deg", np.inf)),
        "failed_checks": result.get("failed_checks", []),
    }
    write_json(output / "probe_record.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step-rad", type=float, default=0.015)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    source = load_config(args.source)
    base = np.asarray(
        [source["control"]["manipulation_delta_rad"][name] for name in ACTIVE_ACTUATORS],
        dtype=np.float64,
    )
    bounds = get_experiment(
        "left_opposed_face_palm_down_contact_preserving_planned_lift"
    ).search_bounds.manipulation_delta_rad
    assert bounds is not None
    probes: list[tuple[str, np.ndarray]] = [("zero", base.copy())]
    for index, name in enumerate(ACTIVE_ACTUATORS):
        for direction in (-1, 1):
            value = base.copy()
            value[index] += direction * float(args.step_rad)
            lower, upper = bounds[name]
            if value[index] < lower - 1e-12 or value[index] > upper + 1e-12:
                continue
            probes.append((f"{index:02d}_{'plus' if direction > 0 else 'minus'}_{name}", value))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payloads = [
        (str(Path(args.source).resolve()), str(Path(args.template).resolve()), str(output / label), value.tolist())
        for label, value in probes
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda value: value["output"])
    report = {"schema_version": 1, "source": str(Path(args.source).resolve()), "step_rad": args.step_rad, "records": records}
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
