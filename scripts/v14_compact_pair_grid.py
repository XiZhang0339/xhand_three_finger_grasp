#!/usr/bin/env python3
"""Run a compact full-reset terminal grid around one schema-v14 config.

This helper deliberately stays outside the formal campaign.  It is intended
for diagnosing two coupled actuator directions after a promising terminal
command has been found, while retaining a resolved config and raw trace for
every point in the grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v14_targeted_probe_analysis import _worker
from xhand_grasp.artifacts import write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config


DEFAULT_AXIS_A = "left_hand_index_joint1_actuator"
DEFAULT_AXIS_B = "left_hand_thumb_rota_joint1_actuator"


def _parse_offsets(value: str) -> tuple[float, ...]:
    offsets = tuple(float(item) for item in value.split(","))
    if not offsets:
        raise argparse.ArgumentTypeError("at least one offset is required")
    return offsets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--axis-a", default=DEFAULT_AXIS_A, choices=ACTIVE_ACTUATORS)
    parser.add_argument("--axis-b", default=DEFAULT_AXIS_B, choices=ACTIVE_ACTUATORS)
    parser.add_argument("--offsets-a", type=_parse_offsets, default=(-0.008, 0.0, 0.008))
    parser.add_argument("--offsets-b", type=_parse_offsets, default=(-0.008, 0.0, 0.008))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    template_path = Path(args.template).resolve()
    source = load_config(source_path)
    base = np.asarray(
        [source["control"]["manipulation_delta_rad"][name] for name in ACTIVE_ACTUATORS],
        dtype=np.float64,
    )
    index_a = ACTIVE_ACTUATORS.index(args.axis_a)
    index_b = ACTIVE_ACTUATORS.index(args.axis_b)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    payloads = []
    for row, offset_a in enumerate(args.offsets_a):
        for column, offset_b in enumerate(args.offsets_b):
            terminal = base.copy()
            terminal[index_a] += offset_a
            terminal[index_b] += offset_b
            label = f"r{row:02d}_c{column:02d}"
            payloads.append(
                (
                    str(source_path),
                    str(template_path),
                    str(output / label),
                    terminal.tolist(),
                )
            )

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: record["output"])
    report = {
        "schema_version": 1,
        "source": str(source_path),
        "axis_a": args.axis_a,
        "axis_b": args.axis_b,
        "offsets_a_rad": list(args.offsets_a),
        "offsets_b_rad": list(args.offsets_b),
        "records": records,
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
