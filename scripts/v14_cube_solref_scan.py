#!/usr/bin/env python3
"""Diagnose v14 contact smoothness across explicit cube solref values."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v14_targeted_probe_analysis import _worker
from xhand_grasp.artifacts import write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config


def _values(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(","))
    if not result or any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("solref values must be positive")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--timeconstants-s",
        type=_values,
        default=(0.004, 0.006, 0.008, 0.012, 0.016),
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    template_path = Path(args.template).resolve()
    output = Path(args.output).resolve()
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    source = load_config(source_path)
    terminal = [
        float(source["control"]["manipulation_delta_rad"][name])
        for name in ACTIVE_ACTUATORS
    ]
    payloads = []
    metadata = {}
    for timeconstant in args.timeconstants_s:
        label = f"solref_{timeconstant:.6f}s".replace(".", "p")
        config = copy.deepcopy(source)
        config["cube"]["solref_timeconst_s"] = float(timeconstant)
        config.setdefault("candidate_metadata", {})["cube_solref_probe"] = {
            "schema_version": 1,
            "timeconstant_s": float(timeconstant),
            "damping_ratio": 1.0,
            "cube_mass_kg": float(config["cube"]["mass_kg"]),
            "cube_friction": float(config["cube"]["friction"]),
            "cube_pose_unchanged": True,
            "requires_full_reset_rerun": True,
        }
        validate_config(config)
        input_path = inputs / f"{label}.json"
        write_json(input_path, config)
        destination = output / "runs" / label
        payloads.append(
            (str(input_path), str(template_path), str(destination), terminal)
        )
        metadata[str(destination)] = {
            "solref_timeconst_s": float(timeconstant),
            "solref_damping_ratio": 1.0,
        }

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            record = future.result()
            record.update(metadata[record["output"]])
            records.append(record)
    records.sort(key=lambda record: record["solref_timeconst_s"])
    report = {
        "schema_version": 1,
        "source": str(source_path),
        "cube_mass_kg": float(source["cube"]["mass_kg"]),
        "cube_friction": float(source["cube"]["friction"]),
        "cube_pose_unchanged": True,
        "records": records,
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
