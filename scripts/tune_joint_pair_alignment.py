#!/usr/bin/env python3
"""Publish one deterministic index/middle joint-base alignment refinement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import ACTIVE_ACTUATORS
from xhand_grasp.experiment import resolve_experiment
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.contact_preserving_candidate_artifacts import (
    run_or_resume_v14_candidate_artifacts,
)
from xhand_grasp.tuning.joint_pair_alignment_refinement import (
    JointPairAlignmentAdjustment,
    build_joint_pair_alignment_config,
    publish_joint_pair_alignment_catalog,
)
from xhand_grasp.tuning.upright_grasp_self_collision_rescue import (
    upright_grasp_audited_session_factory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-trace", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--experiment-id",
        help=(
            "migrate the immutable source geometry into a separately registered "
            "experiment before applying the refinement"
        ),
    )
    parser.add_argument("--wrist-local-rotvec-deg", nargs=3, type=float, required=True)
    parser.add_argument("--contact-pivot-world-m", nargs=3, type=float, required=True)
    parser.add_argument(
        "--root-delta-cube-mm", nargs=3, type=float, default=(0.0, 0.0, 0.0)
    )
    parser.add_argument("--plan-waypoint-scale", type=float, default=1.0)
    parser.add_argument(
        "--grasp-qpos-offset-rad",
        action="append",
        default=[],
        metavar="ACTUATOR=VALUE",
    )
    parser.add_argument(
        "--precontact-qpos-residual-rad",
        action="append",
        default=[],
        metavar="ACTUATOR=VALUE",
    )
    parser.add_argument(
        "--preload-target-residual-rad",
        action="append",
        default=[],
        metavar="ACTUATOR=VALUE",
    )
    parser.add_argument(
        "--close-profile-residual-fraction",
        action="append",
        default=[],
        metavar="FINGER=START,END",
        help="add independent start/end fraction residuals for thumb, index or mid",
    )
    parser.add_argument("--close-duration-s", type=float)
    parser.add_argument("--source-angle-p95-deg", type=float, required=True)
    parser.add_argument("--grasp-angle-p95-max-deg", type=float, required=True)
    parser.add_argument("--operation-angle-p95-max-deg", type=float, required=True)
    parser.add_argument("--minimum-improvement-deg", type=float, default=0.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    source_config_path = Path(args.source_config).expanduser().resolve()
    source_trace_path = Path(args.source_trace).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if args.experiment_id is not None:
        original_experiment_id = str(source_config["experiment_id"])
        original_semantic_sha256 = canonical_sha256(source_config)
        source_config["experiment_id"] = str(args.experiment_id)
        definition = resolve_experiment(source_config)
        source_config["description"] = definition.description
        metadata = source_config.setdefault("candidate_metadata", {})
        metadata["joint_pair_alignment_source_experiment_migration"] = {
            "schema_version": 1,
            "source_experiment_id": original_experiment_id,
            "target_experiment_id": str(args.experiment_id),
            "source_config_file_sha256": file_sha256(source_config_path),
            "source_config_semantic_sha256": original_semantic_sha256,
            "cube_and_controller_unchanged": True,
        }
    def actuator_values(items: list[str], option: str) -> dict[str, float]:
        values = {name: 0.0 for name in ACTIVE_ACTUATORS}
        for item in items:
            try:
                name, raw_value = item.split("=", 1)
                if name not in values:
                    raise ValueError
                values[name] = float(raw_value)
            except ValueError as exc:
                raise SystemExit(
                    f"invalid {option} {item!r}; expected known ACTUATOR=VALUE"
                ) from exc
        return values

    offsets = actuator_values(
        args.grasp_qpos_offset_rad, "--grasp-qpos-offset-rad"
    )
    precontact_residuals = actuator_values(
        args.precontact_qpos_residual_rad,
        "--precontact-qpos-residual-rad",
    )
    preload_residuals = actuator_values(
        args.preload_target_residual_rad,
        "--preload-target-residual-rad",
    )
    close_residuals = {
        "thumb": (0.0, 0.0),
        "index": (0.0, 0.0),
        "mid": (0.0, 0.0),
    }
    for item in args.close_profile_residual_fraction:
        try:
            name, raw_pair = item.split("=", 1)
            raw_start, raw_end = raw_pair.split(",", 1)
            if name not in close_residuals:
                raise ValueError
            close_residuals[name] = (float(raw_start), float(raw_end))
        except ValueError as exc:
            raise SystemExit(
                "invalid --close-profile-residual-fraction "
                f"{item!r}; expected thumb|index|mid=START,END"
            ) from exc
    adjustment = JointPairAlignmentAdjustment(
        wrist_local_rotvec_deg=tuple(args.wrist_local_rotvec_deg),
        contact_pivot_world_m=tuple(args.contact_pivot_world_m),
        root_delta_cube_m=tuple(
            float(value) * 1e-3 for value in args.root_delta_cube_mm
        ),
        fixed_grasp_qpos_offset_rad=tuple(
            offsets[name] for name in ACTIVE_ACTUATORS
        ),
        precontact_qpos_residual_rad=tuple(
            precontact_residuals[name] for name in ACTIVE_ACTUATORS
        ),
        preload_target_residual_rad=tuple(
            preload_residuals[name] for name in ACTIVE_ACTUATORS
        ),
        close_profile_start_residual_fraction=tuple(
            close_residuals[name][0] for name in ("thumb", "index", "mid")
        ),
        close_profile_end_residual_fraction=tuple(
            close_residuals[name][1] for name in ("thumb", "index", "mid")
        ),
        close_duration_s=args.close_duration_s,
        plan_waypoint_scale=args.plan_waypoint_scale,
        source_grasp_window_p95_deg=args.source_angle_p95_deg,
        grasp_window_p95_max_deg=args.grasp_angle_p95_max_deg,
        operation_p95_max_deg=args.operation_angle_p95_max_deg,
        minimum_improvement_deg=args.minimum_improvement_deg,
    )
    candidate_id, config = build_joint_pair_alignment_config(
        source_config,
        adjustment,
        source_config_sha256=file_sha256(source_config_path),
        source_trace_sha256=file_sha256(source_trace_path),
    )
    destination = output_root / "final" / f"candidate_{candidate_id}"
    bundle = run_or_resume_v14_candidate_artifacts(
        config,
        destination,
        candidate_id,
        final_rerun=True,
        session_factory=upright_grasp_audited_session_factory,
    )
    catalog = publish_joint_pair_alignment_catalog(bundle, output_root)
    report = {
        "complete": True,
        "candidate_id": candidate_id,
        "source_config": str(source_config_path),
        "source_trace": str(source_trace_path),
        "adjustment": adjustment.as_mapping(),
        "classification": bundle.result["classification"],
        "grasp_success": bundle.result["grasp_success"],
        "full_success": bundle.result["full_success"],
        "summary": bundle.result["summary"],
        "catalog": str(catalog),
    }
    write_json(output_root / "search_report.json", report)
    alignment = report["summary"].get("metrics", {}).get(
        "index_middle_joint_pair_alignment", {}
    )
    print(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "classification": bundle.result["classification"],
                "grasp_success": bundle.result["grasp_success"],
                "full_success": bundle.result["full_success"],
                "grasp_window_angle_p95_deg": alignment.get(
                    "grasp_window_angle_p95_deg"
                ),
                "operation_angle_p95_deg": alignment.get(
                    "operation_angle_p95_deg"
                ),
                "catalog": str(catalog),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
