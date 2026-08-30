#!/usr/bin/env python3
"""Promote one authenticated v15 grasp and tune v16 rolling-slip feedback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhand_grasp.tuning.rolling_slip_rescue_campaign import (
    DEFAULT_ALIGNMENT_GAINS,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_SLIP_RECOVERY_GAINS_RAD_PER_M,
    DEFAULT_SOURCE_CANDIDATE_ID,
    DEFAULT_SOURCE_CONFIG,
    run_rolling_slip_rescue_campaign,
)


def _positive_csv(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("gain list must be comma-separated numbers") from error
    if not result or any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("gain list must contain positive values")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", default=str(DEFAULT_SOURCE_CONFIG))
    parser.add_argument(
        "--source-trace",
        help="optional path whose SHA-256 must match the authenticated source trace",
    )
    parser.add_argument(
        "--source-candidate-id",
        type=int,
        default=DEFAULT_SOURCE_CANDIDATE_ID,
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument(
        "--alignment-gains",
        type=_positive_csv,
        default=DEFAULT_ALIGNMENT_GAINS,
        help="comma-separated gains; default includes source center 0.9325871364810551",
    )
    parser.add_argument(
        "--slip-gains",
        type=_positive_csv,
        default=DEFAULT_SLIP_RECOVERY_GAINS_RAD_PER_M,
        help="comma-separated rad/m gains; default includes 4",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    outcome = run_rolling_slip_rescue_campaign(
        args.source_config,
        args.output_dir,
        source_trace_path=args.source_trace,
        source_candidate_id=args.source_candidate_id,
        alignment_gains=args.alignment_gains,
        slip_recovery_gains_rad_per_m=args.slip_gains,
        workers=args.workers,
        resume=bool(args.resume),
    )
    report = outcome["report"]
    print(
        json.dumps(
            {
                "workspace": outcome["workspace"],
                "candidate_count": report["candidate_count"],
                "full_success_count": report["full_success_count"],
                "selected_candidate_id": report["selected_candidate_id"],
                "selected_grid_metrics": report["selected_grid_metrics"],
                "selected_final_metrics": report["selected_final_metrics"],
                "viewer_catalog": str(
                    Path(outcome["workspace"]) / report["viewer_catalog"]
                ),
                "aliases": report["aliases"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
