#!/usr/bin/env python3
"""Full-reset v14 feedback-strength grid around one resolved plan.

The grid changes only the three proportional/integral force gains.  This is
useful after bidirectional unloading is enabled: a terminal command that was
safe with inward-only feedback can otherwise be over-released.  Each point is
stored as a fully validated resolved config plus raw trace.
"""

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
from xhand_grasp.experiment import ContactFeedbackParameters


def _scales(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(","))
    if not result or any(item < 0.0 for item in result):
        raise argparse.ArgumentTypeError("scales must be non-negative")
    return result


def _scaled_feedback(
    source: dict, *, kp_scale: float, ki_scale: float
) -> dict:
    base = source["contact_feedback"]
    return ContactFeedbackParameters(
        schema_version=1,
        strategy=str(base["strategy"]),
        filter_time_constant_s=float(base["filter_time_constant_s"]),
        kp_rad_per_n={
            finger: max(1e-12, float(value) * kp_scale)
            for finger, value in base["kp_rad_per_n"].items()
        },
        ki_rad_per_n_s={
            finger: float(value) * ki_scale
            for finger, value in base["ki_rad_per_n_s"].items()
        },
        integral_limit_n_s=float(base["integral_limit_n_s"]),
        correction_limit_rad=float(base["correction_limit_rad"]),
        rate_limit_rad_s=float(base["rate_limit_rad_s"]),
        acceleration_limit_rad_s2=float(base["acceleration_limit_rad_s2"]),
        force_risk_n=float(base["force_risk_n"]),
        freeze_on_risk=bool(base["freeze_on_risk"]),
        max_loss_s=float(base["max_loss_s"]),
        recovery_behavior=str(base["recovery_behavior"]),
        operation_contact_duty_min=float(base["operation_contact_duty_min"]),
    ).as_config()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kp-scales", type=_scales, default=(0.15, 0.30, 0.50, 0.65))
    parser.add_argument("--ki-scales", type=_scales, default=(0.0, 0.25, 0.50))
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
    for row, kp_scale in enumerate(args.kp_scales):
        for column, ki_scale in enumerate(args.ki_scales):
            label = f"kp_{kp_scale:.4f}_ki_{ki_scale:.4f}".replace(".", "p")
            config = copy.deepcopy(source)
            config["contact_feedback"] = _scaled_feedback(
                source, kp_scale=kp_scale, ki_scale=ki_scale
            )
            config.setdefault("candidate_metadata", {})["feedback_strength_probe"] = {
                "schema_version": 1,
                "kp_scale": kp_scale,
                "ki_scale": ki_scale,
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
                "kp_scale": kp_scale,
                "ki_scale": ki_scale,
            }

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            record = future.result()
            record.update(metadata[record["output"]])
            records.append(record)
    records.sort(key=lambda record: (record["kp_scale"], record["ki_scale"]))
    report = {
        "schema_version": 1,
        "source": str(source_path),
        "records": records,
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
