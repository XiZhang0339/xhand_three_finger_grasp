#!/usr/bin/env python3
"""Reacquire selected v11 grasps, finalize measured qpos, and search lift."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xhand_grasp.tuning.actual_contact_grasp_pose import (
    _run_manipulation_stage,
    _run_materialized_local_dynamic_stage,
    _run_measured_grasp_pose_finalization_stage,
)
from xhand_grasp.tuning.relative_wrist_pose_dynamic_guided import (
    build_dynamic_centroid_candidate_records,
)
from xhand_grasp.tuning.relative_wrist_pose_lift_manipulability import (
    _atomic_json,
    load_measured_grasp_source,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round-index", type=int, default=17)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--target-success-count", type=int, default=1)
    parser.add_argument("--enable-local-refinement", action="store_true")
    parser.add_argument("--stage", default="formal_manipulation_rescue")
    return parser


def _summary_record(value: dict) -> dict:
    summary = value.get("summary", {})
    stage = summary.get("stage_status", {}) if isinstance(summary, dict) else {}
    return {
        "candidate_id": int(value["candidate_id"]),
        "source_candidate_id": int(value.get("source_candidate_id", -1)),
        "candidate_sha256": str(value.get("candidate_sha256", "")),
        "artifact_directory": value.get("artifact_directory"),
        "grasp_success": bool(stage.get("grasp_success", value.get("grasp_success", False))),
        "full_success": bool(stage.get("full_success", False)),
        "passed": bool(summary.get("passed", False)),
        "failed_checks": copy.deepcopy(summary.get("failed_checks", [])),
    }


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    materialized: list[dict] = []
    sources: list[dict] = []
    for raw_directory in args.candidate_dir:
        directory = raw_directory.expanduser().resolve()
        source = load_measured_grasp_source(
            measured_config_path=directory / "resolved_config.json",
            measured_result_path=directory / "result.json",
            measured_trace_path=directory / "trace.npz",
        )
        source_candidate_id = int(source.result["candidate_id"])
        records = build_dynamic_centroid_candidate_records(
            [source.config],
            parent_candidate_id=source_candidate_id,
            round_index=args.round_index,
            kind="formal_manipulation_rescue_reacquisition",
            stage="formal_manipulation_rescue_reacquisition",
            parent_artifact_key=source.source_id,
        )
        materialized.extend(records)
        sources.append(source.identity())

    dynamic = _run_materialized_local_dynamic_stage(
        materialized,
        output,
        stage="formal_manipulation_rescue_reacquisition",
        workers=args.workers,
    )
    measured = _run_measured_grasp_pose_finalization_stage(
        dynamic.records,
        output,
        stage="formal_manipulation_rescue_measured",
        workers=args.workers,
    )
    manipulation = _run_manipulation_stage(
        measured.records,
        output,
        target_success_count=args.target_success_count,
        seed=20260821,
        workers=args.workers,
        stage=args.stage,
        enable_local_refinement=args.enable_local_refinement,
    )
    full_count = sum(
        bool(value.get("summary", {}).get("passed", False))
        and bool(
            value.get("summary", {})
            .get("stage_status", {})
            .get("full_success", False)
        )
        for value in manipulation.records
    )
    report = {
        "v11_formal_manipulation_rescue_schema_version": 1,
        "complete": True,
        "sources": sources,
        "dynamic_reacquisition": {
            **dynamic.summary,
            "records": [_summary_record(dict(value)) for value in dynamic.records],
        },
        "measured_finalization": {
            **measured.summary,
            "records": [_summary_record(dict(value)) for value in measured.records],
        },
        "manipulation": {
            **manipulation.summary,
            "full_success_count": full_count,
            "records": [
                _summary_record(dict(value)) for value in manipulation.records
            ],
        },
    }
    _atomic_json(output / "formal_manipulation_rescue_report.json", report)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "dynamic_grasp_success_count": dynamic.summary.get(
                    "grasp_success_count", 0
                ),
                "measured_grasp_pose_success_count": measured.summary.get(
                    "measured_grasp_pose_success_count", 0
                ),
                "manipulation_candidate_count": len(manipulation.records),
                "full_success_count": full_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if full_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
