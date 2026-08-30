#!/usr/bin/env python3
"""Interpolate two same-grasp v11 manipulation deltas with full resets."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xhand_grasp.config import validate_config
from xhand_grasp.tuning.actual_contact_grasp_pose import (
    _load_persisted_manipulation_candidate,
    _run_persisted_manipulation_refinement_jobs,
)
from xhand_grasp.tuning.relative_wrist_pose_lift_manipulability import _atomic_json
from xhand_grasp.grasp_pose import canonical_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-candidate", type=Path, required=True)
    parser.add_argument("--aggressive-candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=19)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def _load(directory: Path) -> tuple[dict, dict]:
    config = json.loads((directory / "resolved_config.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    record = _load_persisted_manipulation_candidate(
        directory,
        candidate_id=int(result["candidate_id"]),
        expected_config=config,
    )
    if record is None:
        raise RuntimeError(f"candidate is absent: {directory}")
    return config, record


def _base_without_delta(config: dict) -> dict:
    value = copy.deepcopy(config)
    value["control"].pop("manipulation_delta_rad")
    value.pop("candidate_metadata", None)
    return value


def main() -> int:
    args = _parser().parse_args()
    if args.steps <= 0 or args.workers <= 0:
        raise ValueError("steps/workers must be positive")
    safe_config, safe_record = _load(args.safe_candidate.expanduser().resolve())
    aggressive_config, aggressive_record = _load(
        args.aggressive_candidate.expanduser().resolve()
    )
    if _base_without_delta(safe_config) != _base_without_delta(aggressive_config):
        raise RuntimeError("line-search endpoints do not share one grasp/controller base")
    names = tuple(sorted(safe_config["control"]["manipulation_delta_rad"]))
    if names != tuple(sorted(aggressive_config["control"]["manipulation_delta_rad"])):
        raise RuntimeError("line-search endpoints use different actuators")
    output = args.output_dir.expanduser().resolve()
    jobs = []
    for index in range(1, args.steps + 1):
        alpha = index / (args.steps + 1)
        config = copy.deepcopy(safe_config)
        config["control"]["manipulation_delta_rad"] = {
            name: float(
                (1.0 - alpha)
                * safe_config["control"]["manipulation_delta_rad"][name]
                + alpha
                * aggressive_config["control"]["manipulation_delta_rad"][name]
            )
            for name in names
        }
        config.setdefault("candidate_metadata", {})["manipulation_line_search"] = {
            "schema_version": 1,
            "safe_candidate_id": int(safe_record["candidate_id"]),
            "aggressive_candidate_id": int(aggressive_record["candidate_id"]),
            "safe_candidate_sha256": canonical_sha256(safe_config),
            "aggressive_candidate_sha256": canonical_sha256(aggressive_config),
            "alpha": float(alpha),
        }
        validate_config(config)
        candidate_id = 110_000_000_000_000 + index
        jobs.append(
            {
                "candidate_id": candidate_id,
                "config": config,
                "job_metadata": {
                    "refinement_root": str(output),
                    "discovery_index": 1216 + index - 1,
                    "parent_candidate_id": int(safe_record["candidate_id"]),
                    "parent_rank": 0,
                    "local_index": index - 1,
                },
            }
        )
    _run_persisted_manipulation_refinement_jobs(tuple(jobs), args.workers)
    records = []
    for job in jobs:
        directory = output / "candidates" / f"candidate_{job['candidate_id']}"
        record = _load_persisted_manipulation_candidate(
            directory,
            candidate_id=int(job["candidate_id"]),
            expected_config=job["config"],
        )
        if record is None:
            raise RuntimeError("line-search candidate disappeared")
        summary = record["summary"]
        metrics = summary.get("metrics", {})
        records.append(
            {
                "candidate_id": int(record["candidate_id"]),
                "alpha": float(
                    record["config"]["candidate_metadata"]
                    ["manipulation_line_search"]["alpha"]
                ),
                "full_success": bool(summary.get("passed"))
                and bool(summary.get("stage_status", {}).get("full_success")),
                "manipulation_completed": bool(
                    summary.get("checks", {}).get("manipulation_completed")
                ),
                "median_lift_m": metrics.get("median_lift_m"),
                "minimum_lift_m": metrics.get("minimum_lift_m"),
                "peak_lift_m": metrics.get("peak_lift_m"),
                "failed_checks": copy.deepcopy(summary.get("failed_checks", [])),
                "artifact_directory": str(directory),
            }
        )
    records.sort(
        key=lambda value: (
            not value["full_success"],
            not value["manipulation_completed"],
            -float(value["minimum_lift_m"] or -1e9),
            value["candidate_id"],
        )
    )
    report = {
        "v11_manipulation_line_search_schema_version": 1,
        "complete": True,
        "safe_candidate_id": int(safe_record["candidate_id"]),
        "aggressive_candidate_id": int(aggressive_record["candidate_id"]),
        "candidate_count": len(records),
        "full_success_count": sum(value["full_success"] for value in records),
        "records": records,
    }
    _atomic_json(output / "line_search_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["full_success_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
