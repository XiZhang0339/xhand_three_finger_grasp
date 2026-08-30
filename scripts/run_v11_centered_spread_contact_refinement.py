#!/usr/bin/env python3
"""Refine a centered thumb and wider index/middle contact pair.

"Center" is the cube-local Y centerline because this continuation follows the
prior Y-axis contact placement experiments.  Gravity-height alignment remains
an independent hard gate.  The cube is unchanged and free; every candidate is
rerun from the no-contact initial state before it can be catalogued as a grasp.
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
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.simulation import SimulationSession
from xhand_grasp.trajectory import preflight_config
from xhand_grasp.viewer import apply_viewer_overrides

from run_v11_thumb_negative_y_refinement import (  # noqa: E402
    FINGERS,
    _candidate_record,
    _cube_signature,
    _local_contact_centroids,
    _mark_override_status,
)


DEFAULT_SOURCE = Path(
    "artifacts/left_opposed_face_palm_down_larger_relative_wrist_pose_"
    "actual_contact_smooth_vertical_lift/tune/"
    "index_middle_positive_y_joint_pose_refinement_from_thumb_negative_y_best_v1/"
    "opposing_fingers_positive_y_thumb_fixed/resolved_config.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/left_opposed_face_palm_down_larger_relative_wrist_pose_"
    "actual_contact_smooth_vertical_lift/tune/"
    "centered_thumb_expanded_opposing_spread_from_positive_y_best_v1"
)

THUMB_CENTER_TOLERANCE_MM = 5.0
PAIR_MIDPOINT_TOLERANCE_MM = 3.0
MINIMUM_PAIR_SEPARATION_MM = 20.0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _candidate_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "label": "thumb_center_pose_stability_priority",
            "role": "thumb_center_pose_stability_priority",
            "target_points_cube_local_y_mm": {
                "thumb": 0.0,
                "index": -12.0,
                "mid": 12.0,
            },
            "local_root_delta_cube_mm": (
                -0.1855272,
                0.2970440,
                -1.2224036,
            ),
            "local_wrist_rotvec_deg": (
                -0.409827733,
                -0.258805397,
                0.324074013,
            ),
            "nominal_qpos_rad": (
                1.4078806878,
                0.1171247370,
                0.6668469987,
                -0.04908256395,
                0.51802661305,
                1.14542840475,
                0.56254041295,
                1.15374729060,
            ),
            "precontact_targets_rad": (
                1.41179844415,
                0.08428819570,
                0.65614351050,
                -0.05511232485,
                0.47903607520,
                1.14425353475,
                0.52221471325,
                1.15773118100,
            ),
            "preload_targets_rad": (
                1.41048689700,
                0.12407019740,
                0.67431704860,
                -0.04747880135,
                0.52154673485,
                1.14841673965,
                0.56515390275,
                1.15558349880,
            ),
            "search_evidence": {
                "method": "joint_pose_control_balancing_probe_refinement",
                "static_target_radius_m": 0.004,
                "dynamic_target_is_ranking_only": True,
            },
        },
        {
            "label": "thumb_center_priority",
            "role": "thumb_center_priority",
            "target_points_cube_local_y_mm": {
                "thumb": 0.0,
                "index": -12.0,
                "mid": 12.0,
            },
            "local_root_delta_cube_mm": (
                0.15223970251504782,
                0.8381312058116029,
                0.08900479697261514,
            ),
            "local_wrist_rotvec_deg": (
                -0.4864241100608292,
                -0.005817591972249429,
                0.08971919830907742,
            ),
            "nominal_qpos_rad": (
                1.4,
                0.10829595474736078,
                0.706693633716041,
                -0.036627198032889045,
                0.5147127099151037,
                1.1301596892076073,
                0.5586071038988463,
                1.1359186718955327,
            ),
            "precontact_targets_rad": (
                1.4040345139851247,
                0.07653751783864148,
                0.6970526895834259,
                -0.04297994863531771,
                0.47610982261774565,
                1.128647645785933,
                0.5187036478377398,
                1.1393498571258656,
            ),
            "preload_targets_rad": (
                1.4026062092,
                0.11524141514736078,
                0.7141636836160411,
                -0.03502343543288905,
                0.5182328317151035,
                1.1331480241076075,
                0.5612205936988461,
                1.1377548800955328,
            ),
            "search_evidence": {
                "method": "real_witness_14d_point_target_dls",
                "static_target_radius_m": 0.004,
                "dynamic_target_is_ranking_only": True,
            },
        },
        {
            "label": "balanced_center_priority",
            "role": "balanced_center_priority",
            "target_points_cube_local_y_mm": {
                "thumb": 0.0,
                "index": -12.0,
                "mid": 12.0,
            },
            "local_root_delta_cube_mm": (
                0.051005604,
                0.658578673,
                0.061076133,
            ),
            "local_wrist_rotvec_deg": (
                -0.571290864,
                0.022335673,
                0.186155298,
            ),
            "nominal_qpos_rad": (
                1.402248639219111,
                0.10737731058200828,
                0.7053178386714791,
                -0.044904663647751315,
                0.5141556524786804,
                1.129682504080079,
                0.5566395507099834,
                1.1394472394899777,
            ),
            "precontact_targets_rad": (
                1.4062601472,
                0.0757234383,
                0.6954676693,
                -0.0508511972,
                0.4755634731,
                1.1280411782,
                0.5117162682,
                1.1432366893,
            ),
            "preload_targets_rad": (
                1.4048548484,
                0.1143227710,
                0.7127878886,
                -0.0433009010,
                0.5176757743,
                1.1326708390,
                0.5592530405,
                1.1412834477,
            ),
            "search_evidence": {
                "method": "real_witness_dls_balanced_center_refinement",
                "static_target_radius_m": 0.004,
                "dynamic_target_is_ranking_only": True,
            },
        },
        {
            "label": "opposing_spread_priority",
            "role": "opposing_spread_priority",
            "target_points_cube_local_y_mm": {
                "thumb": 4.0,
                "index": -14.0,
                "mid": 8.0,
            },
            "local_root_delta_cube_mm": (
                0.1304541001,
                1.3896743204,
                0.0512551497,
            ),
            "local_wrist_rotvec_deg": (
                -0.453215676,
                -0.067368507,
                0.157522327,
            ),
            "nominal_qpos_rad": (
                1.4,
                0.108424816,
                0.703190205,
                -0.013075574,
                0.511576734,
                1.134653150,
                0.563259724,
                1.135481104,
            ),
            "precontact_targets_rad": (
                1.403999287,
                0.076724550,
                0.693498177,
                -0.020552534,
                0.473072065,
                1.133063568,
                0.523242810,
                1.139265329,
            ),
            "preload_targets_rad": (
                1.402606209,
                0.115370276,
                0.710660255,
                -0.011471811,
                0.515096856,
                1.137641485,
                0.565873214,
                1.137317312,
            ),
            "search_evidence": {
                "method": "staged_real_witness_14d_point_target_dls",
                "static_target_radius_m": 0.004,
                "dynamic_target_is_ranking_only": True,
            },
        },
    )


def _reanchored_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """Make local residuals absolute relative to this measured source pose."""

    result = copy.deepcopy(dict(source))
    metadata = result.setdefault("candidate_metadata", {})
    relative = metadata.setdefault("relative_wrist_pose_search", {})
    relative["anchor_hand_pose"] = copy.deepcopy(result["hand_pose"])
    relative["clockwise_orbit_deg"] = 0.0
    relative["root_delta_cube_m"] = [0.0, 0.0, 0.0]
    relative["wrist_local_rotvec_deg"] = [0.0, 0.0, 0.0]
    return result


def _set_active_values(
    config: dict[str, Any], field: str, values: object
) -> None:
    resolved = tuple(float(value) for value in values)  # type: ignore[arg-type]
    if len(resolved) != len(ACTIVE_ACTUATORS):
        raise ValueError(f"{field} must contain eight values")
    target = config["grasp_pose"] if field == "nominal_joint_qpos_rad" else config["control"]
    target[field] = dict(zip(ACTIVE_ACTUATORS, resolved, strict=True))


def _simulate(
    config: dict[str, Any], trace_path: Path
) -> tuple[dict[str, Any], Mapping[str, np.ndarray]]:
    session = SimulationSession(config)
    try:
        while not session.complete:
            session.advance_one()
        return session.finalize(trace_path=trace_path), session.traces
    finally:
        session.close()


def _center_spread_metrics(
    record: Mapping[str, Any], baseline_separation_mm: float
) -> dict[str, Any]:
    points = _mapping(record["contact_centroid_cube_local_mm"])
    thumb_y = float(points["thumb"][1])
    index_y = float(points["index"][1])
    middle_y = float(points["mid"][1])
    separation = middle_y - index_y
    midpoint = 0.5 * (middle_y + index_y)
    goal = (
        bool(record.get("hard_pass"))
        and abs(thumb_y) <= THUMB_CENTER_TOLERANCE_MM
        and abs(midpoint) <= PAIR_MIDPOINT_TOLERANCE_MM
        and separation >= MINIMUM_PAIR_SEPARATION_MM
    )
    return {
        "thumb_center_abs_error_mm": abs(thumb_y),
        "index_middle_midpoint_y_mm": midpoint,
        "index_middle_midpoint_abs_error_mm": abs(midpoint),
        "index_middle_separation_mm": separation,
        "separation_increase_from_source_mm": separation - baseline_separation_mm,
        "center_spread_goal_passed": goal,
    }


def _rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(bool(record.get("center_spread_goal_passed"))),
        int(bool(record.get("hard_pass"))),
        float(record.get("index_middle_separation_mm", -math.inf)),
        -float(record.get("thumb_center_abs_error_mm", math.inf)),
        -float(record.get("index_middle_midpoint_abs_error_mm", math.inf)),
        -float(record.get("pose_translation_mm", math.inf)),
        str(record.get("label", "")),
    )


def _mark_status(
    resolved: dict[str, Any], summary: Mapping[str, Any], record: Mapping[str, Any]
) -> None:
    _mark_override_status(resolved, summary)
    passed = bool(record.get("center_spread_goal_passed"))
    resolved["experiment_status"].update(
        {
            "classification": (
                "validated_fixed_160g_centered_spread_grasp_ablation"
                if passed
                else "centered_spread_grasp_near_miss"
            ),
            "passed": passed,
            "hard_constraints_passed": passed,
            "note": (
                "Fresh free-cube grasp. Center/spread is evaluated in cube-local "
                "Y; manipulation delta remains zero, so this is not a lift claim."
            ),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source_path = args.source_config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {output}")
    source = load_config(source_path)
    source_trace = source_path.parent / "trace.npz"
    if not source_trace.is_file():
        raise FileNotFoundError(f"source trace is missing: {source_trace}")
    source_config_sha = file_sha256(source_path)
    source_trace_sha = file_sha256(source_trace)
    source_cube = _cube_signature(source)
    with np.load(source_trace, allow_pickle=False) as trace:
        start = int(np.asarray(trace["grasp_stable_window_start_step"]))
        end = int(np.asarray(trace["grasp_stable_window_end_step"]))
        baseline, counts = _local_contact_centroids(trace, start, end)
    if np.any(counts < 250) or not np.isfinite(baseline).all():
        raise RuntimeError("source has no authenticated 250-step grasp window")
    baseline_separation_mm = float((baseline[2, 1] - baseline[1, 1]) * 1000.0)
    local_source = _reanchored_source(source)

    output.mkdir(parents=True, exist_ok=args.resume)
    records: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for spec in _candidate_specs():
        label = str(spec["label"])
        directory = output / label
        result_path = directory / "result.json"
        config_path = directory / "resolved_config.json"
        trace_path = directory / "trace.npz"
        if args.resume and all(
            path.is_file() for path in (result_path, config_path, trace_path)
        ):
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("source_config_sha256") != source_config_sha:
                raise RuntimeError(f"resume source config changed: {result_path}")
            if payload.get("source_trace_sha256") != source_trace_sha:
                raise RuntimeError(f"resume source trace changed: {result_path}")
            record = dict(payload["record"])
        else:
            directory.mkdir(exist_ok=args.resume)
            config, changed = apply_viewer_overrides(
                local_source,
                clockwise_orbit_deg=0.0,
                root_delta_cube_mm=spec["local_root_delta_cube_mm"],
                wrist_local_rotvec_deg=spec["local_wrist_rotvec_deg"],
            )
            if not changed:
                raise RuntimeError("candidate was not marked as a fresh override")
            _set_active_values(
                config, "nominal_joint_qpos_rad", spec["nominal_qpos_rad"]
            )
            _set_active_values(
                config, "precontact_targets_rad", spec["precontact_targets_rad"]
            )
            _set_active_values(
                config, "contact_preload_targets_rad", spec["preload_targets_rad"]
            )
            metadata = config.setdefault("candidate_metadata", {})
            metadata["centered_spread_contact_refinement"] = {
                "source_config_sha256": source_config_sha,
                "source_trace_sha256": source_trace_sha,
                "cube_pose_sampled": False,
                "hand_root_fixed_during_simulation": True,
                "center_axis": "cube_local_Y",
                "target_points_cube_local_y_mm": copy.deepcopy(
                    spec["target_points_cube_local_y_mm"]
                ),
                "search_evidence": copy.deepcopy(spec["search_evidence"]),
            }
            validate_config(config)
            preflight_config(config)
            if _cube_signature(config) != source_cube:
                raise RuntimeError("candidate changed cube pose or physics")
            summary, traces = _simulate(config, trace_path)
            record = _candidate_record(
                shift_mm=1.0,
                summary=summary,
                traces=traces,
                baseline_centroids_m=baseline,
            )
            record.pop("requested_root_shift_cube_y_mm", None)
            record.update(
                {
                    "label": label,
                    "role": str(spec["role"]),
                    "target_points_cube_local_y_mm": copy.deepcopy(
                        spec["target_points_cube_local_y_mm"]
                    ),
                    "hand_pose": copy.deepcopy(config["hand_pose"]),
                    "relative_wrist_pose": copy.deepcopy(
                        _mapping(config.get("candidate_metadata")).get(
                            "relative_wrist_pose_search"
                        )
                    ),
                    "nominal_joint_qpos_rad": copy.deepcopy(
                        config["grasp_pose"]["nominal_joint_qpos_rad"]
                    ),
                    "precontact_targets_rad": copy.deepcopy(
                        config["control"]["precontact_targets_rad"]
                    ),
                    "contact_preload_targets_rad": copy.deepcopy(
                        config["control"]["contact_preload_targets_rad"]
                    ),
                    "cube_world_pose_size_mass_friction_unchanged": True,
                }
            )
            record.update(_center_spread_metrics(record, baseline_separation_mm))
            resolved = resolved_run_config(config, summary)
            _mark_status(resolved, summary, record)
            write_json(config_path, resolved)
            write_json(
                result_path,
                {
                    "diagnostic_schema_version": 1,
                    "diagnostic_kind": "centered_spread_contact_refinement",
                    "source_config": str(source_path),
                    "source_config_sha256": source_config_sha,
                    "source_trace_sha256": source_trace_sha,
                    "record": record,
                    "summary": summary,
                },
            )
        artifacts = {
            "resolved_config": f"{label}/resolved_config.json",
            "result": f"{label}/result.json",
            "trace": f"{label}/trace.npz",
            "sha256": {
                "resolved_config": file_sha256(config_path),
                "result": file_sha256(result_path),
                "trace": file_sha256(trace_path),
            },
        }
        record["artifacts"] = copy.deepcopy(artifacts)
        records.append(record)
        entries.append(
            {
                "label": label,
                "trajectory_id": label,
                "aliases": [],
                "hard_pass": bool(record["hard_pass"]),
                "center_spread_goal_passed": bool(
                    record["center_spread_goal_passed"]
                ),
                "artifacts": artifacts,
            }
        )
        print(
            f"{label}: hard={record['hard_pass']} "
            f"goal={record['center_spread_goal_passed']} "
            f"thumb|y|={record['thumb_center_abs_error_mm']:.3f} mm "
            f"pair_mid={record['index_middle_midpoint_y_mm']:.3f} mm "
            f"separation={record['index_middle_separation_mm']:.3f} mm"
        )

    passing = [value for value in records if value["center_spread_goal_passed"]]
    selected = max(passing or records, key=_rank)
    center_best = min(
        (value for value in records if value["hard_pass"]),
        key=lambda value: (
            float(value["thumb_center_abs_error_mm"]),
            float(value["index_middle_midpoint_abs_error_mm"]),
            -float(value["index_middle_separation_mm"]),
        ),
    )
    spread_best = max(
        (value for value in records if value["hard_pass"]),
        key=lambda value: float(value["index_middle_separation_mm"]),
    )
    pair_center_best = min(
        (value for value in records if value["hard_pass"]),
        key=lambda value: (
            float(value["index_middle_midpoint_abs_error_mm"]),
            float(value["thumb_center_abs_error_mm"]),
            -float(value["index_middle_separation_mm"]),
        ),
    )
    aliases = {
        "best_nominal": str(selected["label"]),
        "best_centered_spread": str(selected["label"]),
        "best_thumb_center": str(center_best["label"]),
        "best_pair_center": str(pair_center_best["label"]),
        "best_opposing_spread": str(spread_best["label"]),
    }
    for entry in entries:
        entry["aliases"] = [
            alias for alias, target in aliases.items() if target == entry["label"]
        ]
    report = {
        "diagnostic_schema_version": 1,
        "diagnostic_kind": "centered_spread_contact_refinement",
        "status": "complete",
        "source_config": str(source_path),
        "source_config_sha256": source_config_sha,
        "source_trace_sha256": source_trace_sha,
        "cube_world_pose_size_mass_friction_unchanged": True,
        "hand_root_fixed_during_each_simulation": True,
        "runtime_time_varying_actuators": 8,
        "center_spread_acceptance": {
            "coordinate_frame": "cube_local",
            "center_axis": "Y",
            "thumb_center_abs_y_max_mm": THUMB_CENTER_TOLERANCE_MM,
            "index_middle_midpoint_abs_y_max_mm": PAIR_MIDPOINT_TOLERANCE_MM,
            "index_middle_separation_min_mm": MINIMUM_PAIR_SEPARATION_MM,
            "existing_grasp_hard_gates_also_required": True,
        },
        "baseline_contact_centroid_cube_local_mm": {
            finger: (baseline[index] * 1000.0).tolist()
            for index, finger in enumerate(FINGERS)
        },
        "baseline_index_middle_separation_mm": baseline_separation_mm,
        "candidate_count": len(records),
        "goal_pass_count": len(passing),
        "aliases": aliases,
        "records": records,
    }
    write_json(output / "refinement_report.json", report)
    write_json(
        output / "catalog.json",
        {
            "catalog_schema_version": 1,
            "catalog_kind": "centered_spread_contact_refinement_viewer_catalog",
            "aliases": aliases,
            "trajectories": entries,
        },
    )
    print(f"best={aliases['best_nominal']}")
    print(output / "refinement_report.json")
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())
