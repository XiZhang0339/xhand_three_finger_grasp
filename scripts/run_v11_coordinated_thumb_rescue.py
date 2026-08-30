#!/usr/bin/env python3
"""Resume v11 with a coordinated three-actuator thumb preload sweep.

The interrupted campaign showed that increasing thumb bend alone increases
normal force but rotates the commanded fingertip velocity away from the cube
normal.  This utility keeps each authenticated grasp pose fixed and scales the
two thumb-rotation closing components while sweeping the bend endpoint.  Every
candidate is rerun from the original free-body initial state.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xhand_grasp.config import validate_config
from xhand_grasp.tuning.actual_contact_grasp_pose import (
    _run_materialized_local_dynamic_stage,
)
from xhand_grasp.tuning.relative_wrist_pose_dynamic_guided import (
    build_dynamic_centroid_candidate_records,
)
from xhand_grasp.tuning.relative_wrist_pose_lift_manipulability import (
    _atomic_json,
    load_measured_grasp_source,
)


THUMB_BEND = "left_hand_thumb_bend_joint_actuator"
THUMB_ROTATION = (
    "left_hand_thumb_rota_joint1_actuator",
    "left_hand_thumb_rota_joint2_actuator",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", type=int, nargs="+", required=True)
    parser.add_argument(
        "--bend-target-rad",
        type=float,
        nargs="+",
        default=(1.415, 1.420, 1.425, 1.430),
    )
    parser.add_argument(
        "--rotation-scale",
        type=float,
        nargs="+",
        default=(1.25, 1.50, 2.00, 2.50),
    )
    parser.add_argument(
        "--strategy",
        choices=("endpoint-rotation-scale", "segment-shift"),
        default="endpoint-rotation-scale",
        help=(
            "scale the thumb rotation endpoint, or translate nominal/precontact/"
            "preload bend together to preserve the exact CLOSE direction"
        ),
    )
    parser.add_argument("--round-index", type=int, default=14)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def _source_directory(root: Path, candidate_id: int) -> Path:
    directory = root / f"candidate_{candidate_id}"
    required = tuple(directory / name for name in ("resolved_config.json", "result.json", "trace.npz"))
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"source candidate is incomplete: {candidate_id}")
    return directory


def _coordinated_config(
    source: dict[str, Any], *, bend_target: float, rotation_scale: float
) -> dict[str, Any]:
    config = copy.deepcopy(source)
    control = config["control"]
    precontact = control["precontact_targets_rad"]
    source_preload = control["contact_preload_targets_rad"]
    preload = copy.deepcopy(source_preload)
    preload[THUMB_BEND] = float(bend_target)
    for actuator in THUMB_ROTATION:
        preload[actuator] = float(
            precontact[actuator]
            + rotation_scale * (source_preload[actuator] - precontact[actuator])
        )
    control["contact_preload_targets_rad"] = preload
    control["manipulation_delta_rad"] = {
        name: 0.0 for name in control["manipulation_delta_rad"]
    }
    config.setdefault("candidate_metadata", {})["coordinated_thumb_preload"] = {
        "strategy": "bend_endpoint_with_scaled_rotation_closing_ray",
        "bend_target_rad": float(bend_target),
        "rotation_scale": float(rotation_scale),
        "source_preload_targets_rad": {
            name: float(source_preload[name])
            for name in (THUMB_BEND, *THUMB_ROTATION)
        },
    }
    validate_config(config)
    return config


def _segment_shift_config(
    source: dict[str, Any], *, bend_target: float
) -> dict[str, Any]:
    config = copy.deepcopy(source)
    old_target = float(
        config["control"]["contact_preload_targets_rad"][THUMB_BEND]
    )
    delta = float(bend_target - old_target)
    original_segment = {
        name: float(
            config["control"]["contact_preload_targets_rad"][name]
            - config["control"]["precontact_targets_rad"][name]
        )
        for name in config["control"]["contact_preload_targets_rad"]
    }
    config["grasp_pose"]["nominal_joint_qpos_rad"][THUMB_BEND] += delta
    config["control"]["precontact_targets_rad"][THUMB_BEND] += delta
    config["control"]["contact_preload_targets_rad"][THUMB_BEND] = float(
        bend_target
    )
    config["control"]["manipulation_delta_rad"] = {
        name: 0.0 for name in config["control"]["manipulation_delta_rad"]
    }
    shifted_segment = {
        name: float(
            config["control"]["contact_preload_targets_rad"][name]
            - config["control"]["precontact_targets_rad"][name]
        )
        for name in config["control"]["contact_preload_targets_rad"]
    }
    if shifted_segment != original_segment:
        raise RuntimeError("segment-shift changed the joint-space CLOSE vector")
    config.setdefault("candidate_metadata", {})["coordinated_thumb_preload"] = {
        "strategy": "translate_nominal_precontact_and_preload_bend_together",
        "bend_target_rad": float(bend_target),
        "bend_delta_rad": delta,
        "joint_space_close_segment_preserved": True,
    }
    validate_config(config)
    return config


def main() -> int:
    args = _parser().parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if any(not 1.40 <= value <= 1.60 for value in args.bend_target_rad):
        raise ValueError("bend targets must remain inside the v11 actual-qpos envelope")
    if any(value <= 0.0 for value in args.rotation_scale):
        raise ValueError("rotation scales must be positive")

    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    records: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for candidate_id in sorted(set(args.candidate_id)):
        directory = _source_directory(source_root, candidate_id)
        authenticated = load_measured_grasp_source(
            measured_config_path=directory / "resolved_config.json",
            measured_result_path=directory / "result.json",
            measured_trace_path=directory / "trace.npz",
        )
        source_hashes[str(candidate_id)] = authenticated.source_id
        if args.strategy == "segment-shift":
            proposals = [
                _segment_shift_config(
                    authenticated.config,
                    bend_target=bend_target,
                )
                for bend_target in sorted(set(args.bend_target_rad))
            ]
        else:
            proposals = [
                _coordinated_config(
                    authenticated.config,
                    bend_target=bend_target,
                    rotation_scale=rotation_scale,
                )
                for bend_target in sorted(set(args.bend_target_rad))
                for rotation_scale in sorted(set(args.rotation_scale))
            ]
        records.extend(
            build_dynamic_centroid_candidate_records(
                proposals,
                parent_candidate_id=candidate_id,
                round_index=args.round_index,
                kind=f"coordinated_thumb_preload_{args.strategy}",
                stage=f"lift_manip_coordinated_thumb_preload_{args.strategy}",
                parent_artifact_key=authenticated.source_id,
            )
        )

    execution = _run_materialized_local_dynamic_stage(
        records,
        output_dir,
        stage=f"lift_manip_coordinated_thumb_preload_{args.strategy}",
        workers=args.workers,
    )
    readiness: list[dict[str, Any]] = []
    for record in execution.records:
        summary = record.get("summary", {})
        status = summary.get("stage_status", {}) if isinstance(summary, dict) else {}
        if status.get("grasp_success") is not True:
            continue
        directory = output_dir / "dynamic" / "candidates" / f"candidate_{record['candidate_id']}"
        measured = load_measured_grasp_source(
            measured_config_path=directory / "resolved_config.json",
            measured_result_path=directory / "result.json",
            measured_trace_path=directory / "trace.npz",
        )
        metadata = measured.config["candidate_metadata"]["coordinated_thumb_preload"]
        readiness.append(
            {
                "candidate_id": int(record["candidate_id"]),
                "source_candidate_id": int(record["source_candidate_id"]),
                "bend_target_rad": float(metadata["bend_target_rad"]),
                "rotation_scale": (
                    float(metadata["rotation_scale"])
                    if "rotation_scale" in metadata
                    else None
                ),
                "readiness": measured.readiness.as_dict(),
                "artifact_directory": str(directory),
                "source_id": measured.source_id,
            }
        )
    readiness.sort(
        key=lambda value: (
            not bool(value["readiness"]["passed"]),
            float(value["readiness"]["normalized_max_violation"]),
            int(value["candidate_id"]),
        )
    )
    report = {
        "coordinated_thumb_rescue_schema_version": 1,
        "complete": True,
        "source_root": str(source_root),
        "source_ids": source_hashes,
        "strategy": args.strategy,
        "bend_targets_rad": sorted(set(args.bend_target_rad)),
        "rotation_scales": sorted(set(args.rotation_scale)),
        "materialized_candidate_count": len(records),
        "dynamic_candidate_count": len(execution.records),
        "grasp_success_count": len(readiness),
        "support_mode_ready_count": sum(
            bool(value["readiness"]["passed"]) for value in readiness
        ),
        "readiness_records": readiness,
    }
    _atomic_json(output_dir / "coordinated_thumb_rescue_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if readiness else 2


if __name__ == "__main__":
    raise SystemExit(main())
