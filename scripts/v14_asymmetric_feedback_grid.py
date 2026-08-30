#!/usr/bin/env python3
"""Full-reset asymmetric per-finger v14 force-feedback scan."""

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


def _feedback(
    source: dict,
    *,
    thumb_index_kp_scale: float,
    mid_kp_scale: float,
    mid_ki_scale: float,
) -> dict:
    base = source["contact_feedback"]
    kp_scale = {
        "thumb": thumb_index_kp_scale,
        "index": thumb_index_kp_scale,
        "mid": mid_kp_scale,
    }
    ki_scale = {"thumb": 0.0, "index": 0.0, "mid": mid_ki_scale}
    return ContactFeedbackParameters(
        schema_version=1,
        strategy=str(base["strategy"]),
        filter_time_constant_s=float(base["filter_time_constant_s"]),
        kp_rad_per_n={
            finger: max(1e-12, float(base["kp_rad_per_n"][finger]) * kp_scale[finger])
            for finger in ("thumb", "index", "mid")
        },
        ki_rad_per_n_s={
            finger: float(base["ki_rad_per_n_s"][finger]) * ki_scale[finger]
            for finger in ("thumb", "index", "mid")
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
    parser.add_argument("--thumb-index-kp-scale", type=float, default=0.01)
    parser.add_argument("--mid-kp-scales", type=_scales, default=(0.05, 0.1, 0.2, 0.3, 0.5))
    parser.add_argument("--mid-ki-scales", type=_scales, default=(0.0, 0.1, 0.25))
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
    for mid_kp in args.mid_kp_scales:
        for mid_ki in args.mid_ki_scales:
            label = (
                f"ti_kp_{args.thumb_index_kp_scale:.4f}_mid_kp_{mid_kp:.4f}_"
                f"mid_ki_{mid_ki:.4f}"
            ).replace(".", "p")
            config = copy.deepcopy(source)
            config["contact_feedback"] = _feedback(
                source,
                thumb_index_kp_scale=args.thumb_index_kp_scale,
                mid_kp_scale=mid_kp,
                mid_ki_scale=mid_ki,
            )
            config.setdefault("candidate_metadata", {})["asymmetric_feedback_probe"] = {
                "schema_version": 1,
                "thumb_index_kp_scale": args.thumb_index_kp_scale,
                "mid_kp_scale": mid_kp,
                "mid_ki_scale": mid_ki,
                "thumb_index_ki_scale": 0.0,
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
                "thumb_index_kp_scale": args.thumb_index_kp_scale,
                "mid_kp_scale": mid_kp,
                "mid_ki_scale": mid_ki,
            }

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            record = future.result()
            record.update(metadata[record["output"]])
            records.append(record)
    records.sort(key=lambda record: (record["mid_kp_scale"], record["mid_ki_scale"]))
    report = {"schema_version": 1, "source": str(source_path), "records": records}
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
