#!/usr/bin/env python3
"""Run an authenticated contact-environment ablation for one fixed grasp."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhand_grasp.tuning.contact_environment_campaign import (
    run_contact_environment_campaign,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    result = run_contact_environment_campaign(
        args.config,
        args.output_dir,
        resume=bool(args.resume),
        workers=args.workers,
        expected_source_sha256=args.expected_source_sha256,
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "workspace": result["workspace"],
                "case_count": report["case_count"],
                "aliases": report["aliases"],
                "classification_counts": report["classification_counts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
