#!/usr/bin/env python3
"""Run the resumable v11 6D wrist/contact-line geometry rescue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xhand_grasp.tuning.relative_wrist_pose_search import RelativeWristDLSSettings
from xhand_grasp.tuning.relative_wrist_pose_lift_manipulability import (
    LiftManipulabilityPolicy,
    _atomic_json,
    _physical_screen_input_sha256,
    load_measured_grasp_source,
    load_physical_screen_artifacts,
    physical_screen_lift_candidates,
    run_or_resume_generation_campaign,
    run_supported_dynamic_grasp_screen,
    write_physical_screen_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-config", type=Path, required=True)
    parser.add_argument("--measured-result", type=Path, required=True)
    parser.add_argument("--measured-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-cell", type=int, default=128)
    parser.add_argument("--retained-per-cell", type=int, default=4)
    parser.add_argument("--minimum-per-edge", type=int, default=8)
    parser.add_argument("--selected-total", type=int, default=32)
    parser.add_argument("--controller-seed-count", type=int, default=6)
    parser.add_argument("--dls-iterations", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    policy = LiftManipulabilityPolicy(
        samples_per_cell=args.samples_per_cell,
        retained_per_cell=args.retained_per_cell,
        minimum_per_edge=args.minimum_per_edge,
        selected_total=args.selected_total,
    )
    source = load_measured_grasp_source(
        measured_config_path=args.measured_config,
        measured_result_path=args.measured_result,
        measured_trace_path=args.measured_trace,
        policy=policy,
    )
    output = args.output_dir.expanduser().resolve()
    generation = run_or_resume_generation_campaign(
        source,
        output / "generation",
        policy=policy,
        resume=args.resume,
    )
    settings = RelativeWristDLSSettings(maximum_iterations=args.dls_iterations)
    physical_dir = output / "physical_screen"
    physical_input = _physical_screen_input_sha256(
        source, generation.selected_records, settings
    )
    if args.resume and (physical_dir / "physical_screen_manifest.json").is_file():
        physical = load_physical_screen_artifacts(
            physical_dir, expected_input_sha256=physical_input
        )
    else:
        physical = physical_screen_lift_candidates(
            generation.selected_records,
            settings=settings,
        )
        write_physical_screen_artifacts(
            physical,
            physical_dir,
            input_sha256=physical_input,
        )
    dynamic = run_supported_dynamic_grasp_screen(
        physical,
        output / "dynamic_screen",
        workers=args.workers,
        resume=args.resume,
        controller_seed_count=args.controller_seed_count,
        policy=policy,
    )
    report = {
        **dynamic.report,
        "complete": True,
        "source": source.identity(),
        "generation_manifest": generation.manifest,
        "physical_screen": physical.report,
        "candidate_records": list(dynamic.records),
        "ready_sources": [value.identity() for value in dynamic.ready_sources],
    }
    _atomic_json(output / "lift_geometry_rescue_report.json", report)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "generated_selected_count": len(generation.selected_records),
                "physical_dynamic_eligible_count": physical.report[
                    "dynamic_eligible_count"
                ],
                "dynamic_candidate_count": len(dynamic.records),
                "grasp_success_count": dynamic.report["grasp_success_count"],
                "support_mode_ready_count": len(dynamic.ready_sources),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if dynamic.ready_sources else 2


if __name__ == "__main__":
    raise SystemExit(main())
