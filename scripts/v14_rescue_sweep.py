#!/usr/bin/env python3
"""Run a bounded schema-v14 rescue sweep over retained v13 dynamics.

This is a diagnostic front-end for the formal resumable campaign.  It never
edits v13 evidence: selected source configs are converted in memory, rerun
from reset with the v14 controller, and written below a new output directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhand_grasp.artifacts import write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.experiment import ContactForceTargets, ManipulationPlanParameters
from xhand_grasp.simulation import run_simulation


REPLACED_FIELDS = (
    "schema_version",
    "experiment_id",
    "description",
    "control_protocol",
    "acceptance",
    "robustness",
    "search",
    "actual_contact_grasp_pose_campaign",
    "contact_preserving_planned_lift_campaign",
    "contact_alignment",
    "contact_feedback",
)


def _v14_config(source: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(source)
    for name in REPLACED_FIELDS:
        config[name] = copy.deepcopy(template[name])
    config.pop("scaled_contact_downsize_campaign", None)
    config.pop("scaled_contact_mapping", None)
    terminal = {
        name: float(source["control"]["manipulation_delta_rad"][name])
        for name in ACTIVE_ACTUATORS
    }
    times = tuple(float(value) for value in template["manipulation_plan"]["knot_times_s"])
    object_progress = tuple(
        10.0 * (value / 3.0) ** 3
        - 15.0 * (value / 3.0) ** 4
        + 6.0 * (value / 3.0) ** 5
        for value in times
    )
    # The v13 endpoint may be as large as 0.45 rad.  Space actuator knots
    # uniformly so every one of the 20 local moves remains well below the
    # v14 0.04 rad trust-region bound; the desired object path retains the
    # global minimum-jerk timing used for scoring.
    actuator_progress = tuple(value / 3.0 for value in times)
    plan = ManipulationPlanParameters(
        schema_version=1,
        profile="piecewise_quintic_minimum_jerk",
        duration_s=3.0,
        knot_times_s=times,
        actuator_waypoints_rad={
            name: tuple(terminal[name] * value for value in actuator_progress)
            for name in ACTIVE_ACTUATORS
        },
        desired_cube_position_delta_m=tuple(
            (0.0, 0.0, 0.011 * value) for value in object_progress
        ),
        desired_cube_rotation_vector_rad=tuple(
            (0.0, 0.0, 0.0) for _ in object_progress
        ),
        max_knot_delta_rad=0.04,
        trust_region_backtracks=4,
    )
    config["manipulation_plan"] = plan.as_config()
    config["control"]["manipulation_delta_rad"] = terminal
    config["contact_force_targets_n"] = ContactForceTargets(
        schema_version=1,
        source="verify_window_median_clamped",
        minimum_n=0.2,
        maximum_n=3.0,
        per_finger_n={"thumb": 0.2, "index": 0.2, "mid": 0.2},
    ).as_config()
    metadata = copy.deepcopy(config.get("candidate_metadata", {}))
    metadata.update(
        {
            "schema_version": 1,
            "v14_rescue_source_candidate_id": str(
                metadata.get("candidate_id", "unknown")
            ),
            "cube_pose_sampled": False,
            "hand_root_fixed_during_simulation": True,
        }
    )
    config["candidate_metadata"] = metadata
    validate_config(config)
    return config


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    smooth = metrics.get("motion_smoothness", {})
    return {
        "full_success": bool(result.get("stage_status", {}).get("full_success", False)),
        "grasp_success": bool(result.get("stage_status", {}).get("grasp_success", False)),
        "aborted": bool(metrics.get("controller_aborted", True)),
        "simultaneous_contact_duty": float(
            metrics.get("operation_target_face_simultaneous_duty", 0.0)
        ),
        "per_finger_contact_duty": copy.deepcopy(
            metrics.get("operation_target_face_contact_duty", {})
        ),
        "median_lift_m": float(metrics.get("operation_median_lift_m", -1.0)),
        "minimum_lift_m": float(metrics.get("operation_minimum_lift_m", -1.0)),
        "lateral_displacement_m": float(
            smooth.get("operation_max_lateral_displacement_m", float("inf"))
        ),
        "orientation_drift_deg": float(
            smooth.get("operation_max_orientation_drift_deg", float("inf"))
        ),
        "failed_checks": list(result.get("failed_checks", ())),
    }


def _worker(payload: tuple[str, str, str]) -> dict[str, Any]:
    source_path, template_path, destination = payload
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    config = _v14_config(load_config(source_path), load_config(template_path))
    write_json(output / "resolved_config.json", config)
    summary = run_simulation(config, trace_path=output / "trace.npz")
    write_json(output / "summary.json", summary)
    return {
        "source_config": source_path,
        "output_directory": str(output),
        "edge_mm": int(round(float(config["cube"]["edge_m"]) * 1000.0)),
        "metrics": _metrics(summary),
    }


def _selected_sources(campaign: Path, per_set: int) -> list[Path]:
    selected: list[Path] = []
    for directory in sorted((campaign / "manipulation" / "local_refinement").glob("set_*")):
        records: list[tuple[float, float, float, Path]] = []
        for result_path in sorted(directory.glob("candidates/candidate_*/result.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            metrics = result.get("summary", {}).get("metrics", {})
            records.append(
                (
                    float(metrics.get("operation_target_face_simultaneous_duty", 0.0)),
                    float(metrics.get("operation_median_lift_m", -1.0)),
                    -float(
                        metrics.get("motion_smoothness", {}).get(
                            "operation_max_lateral_displacement_m", float("inf")
                        )
                    ),
                    result_path.with_name("resolved_config.json"),
                )
            )
        # Preserve contact first, then include the best lift and lowest lateral
        # alternatives when they are distinct.
        orders = (
            sorted(records, key=lambda value: (-value[0], -value[1], -value[2])),
            sorted(records, key=lambda value: (-value[1], -value[0], -value[2])),
            sorted(records, key=lambda value: (-value[2], -value[0], -value[1])),
        )
        for order in orders:
            for record in order:
                if record[3] not in selected:
                    selected.append(record[3])
                    break
            if sum(path.is_relative_to(directory) for path in selected) >= per_set:
                break
    return selected


def _rank(record: dict[str, Any]) -> tuple[Any, ...]:
    value = record["metrics"]
    duties = value["per_finger_contact_duty"]
    return (
        not value["full_success"],
        value["aborted"],
        -float(value["simultaneous_contact_duty"]),
        -min(float(duties.get(name, 0.0)) for name in ("thumb", "index", "mid")),
        -float(value["median_lift_m"]),
        float(value["lateral_displacement_m"]),
        record["source_config"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--per-set", type=int, default=3)
    args = parser.parse_args()
    campaign = Path(args.campaign).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = _selected_sources(campaign, int(args.per_set))
    payloads = [
        (str(path), str(Path(args.template).resolve()), str(output / f"candidate_{index:03d}"))
        for index, path in enumerate(sources)
    ]
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {executor.submit(_worker, value): value for value in payloads}
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=_rank)
    report = {
        "schema_version": 1,
        "complete": True,
        "candidate_count": len(records),
        "records": records,
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
