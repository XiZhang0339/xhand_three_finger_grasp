#!/usr/bin/env python3
"""Run the authenticated 79-mm nonlinear warm starts with bounded feedback.

This is a small, resumable diagnostic boundary used before committing the
much larger formal schema-v14 campaign.  All candidates are regenerated from
the immutable v13 evidence and rerun from the original no-contact state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhand_grasp.artifacts import write_json
from xhand_grasp.config import load_config
from xhand_grasp.experiment import resolve_experiment
from xhand_grasp.scene import build_model
from xhand_grasp.tuning.actual_contact_manipulation import (
    manipulation_delta_bounds,
)
from xhand_grasp.tuning.contact_constrained_planner import (
    rank_contact_constrained_candidates,
)
from xhand_grasp.tuning.contact_preserving_planned_lift_campaign import (
    _default_candidate_runner,
    _warm_start_plan_records,
    authenticate_v13_grasp_sources,
    build_v14_source_pairs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--feedback-per-plan", type=int, default=1)
    parser.add_argument("--maximum-candidates", type=int, default=8)
    args = parser.parse_args()

    template = load_config(args.config)
    definition = resolve_experiment(template)
    bundle = authenticate_v13_grasp_sources(definition)
    pair = build_v14_source_pairs(template, definition, bundle)[0]
    if pair["priority_role"] != "primary_79mm":
        raise RuntimeError("authenticated primary pair ordering changed")
    model, _ = build_model(pair["config"])
    bounds = manipulation_delta_bounds(model, pair["config"])
    plans = _warm_start_plan_records(pair, pair["config"], bounds)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = _default_candidate_runner(
        plans,
        output,
        target_success_count=1,
        workers=int(args.workers),
        feedback_candidates_per_plan=int(args.feedback_per_plan),
        maximum_candidate_count=int(args.maximum_candidates),
    )
    ranked = rank_contact_constrained_candidates(records)
    report = {
        "schema_version": 1,
        "complete": True,
        "primary_source_candidate_id": int(pair["source_candidate_id"]),
        "feedback_candidates_per_plan": int(args.feedback_per_plan),
        "candidate_count": len(ranked),
        "full_success_count": sum(
            bool(value.get("full_success", False)) for value in ranked
        ),
        "records": list(ranked),
    }
    write_json(output / "report.json", report)
    # Candidate records intentionally contain NumPy scalar values from the
    # deterministic ranking path. ``write_json`` canonicalizes them, while a
    # direct ``json.dumps(report)`` would fail after all simulations had
    # already completed. Keep terminal output compact and natively serializable.
    print(
        json.dumps(
            {
                "output": str(output),
                "candidate_count": len(ranked),
                "full_success_count": report["full_success_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
