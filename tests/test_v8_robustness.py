from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import pytest

from xhand_grasp.artifacts import write_json
from xhand_grasp.config import load_config, validate_config
from xhand_grasp.tuning.normal_aligned_smooth_lift import (
    DEFAULT_V8_TEMPLATE,
    canonical_sha256,
)
from xhand_grasp.tuning.normal_aligned_smooth_lift_robustness import (
    generate_v8_perturbation_configs,
    run_v8_robustness_campaign,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v8_perturbations_are_deterministic_valid_and_inside_envelope():
    config = load_config(ROOT / DEFAULT_V8_TEMPLATE)
    before = copy.deepcopy(config)
    first = generate_v8_perturbation_configs(
        config,
        count=16,
        seed=20260821,
        source_candidate_id=89_000_000_000_000,
        family="per_exact_local_16",
    )
    second = generate_v8_perturbation_configs(
        config,
        count=16,
        seed=20260821,
        source_candidate_id=89_000_000_000_000,
        family="per_exact_local_16",
    )
    assert first == second
    assert config == before
    assert len({canonical_sha256(value) for value in first}) == 16
    for trial, case in enumerate(first):
        validate_config(case)
        assert case["run_context"] == {"kind": "robustness_trial"}
        resolved = case["candidate_metadata"]["robustness_trial"]
        assert resolved["trial"] == trial
        perturbation = resolved["resolved_perturbations"]
        assert all(abs(value) <= 0.0015 for value in perturbation["cube_center_xy_delta_m"])
        assert 0.0 <= perturbation["cube_gap_delta_m"] <= 0.0005
        assert all(abs(value) <= 3.0 for value in perturbation["cube_rpy_delta_deg"])
        assert 0.9 <= perturbation["mass_scale"] <= 1.1
        assert -0.1 <= perturbation["friction_delta"] <= 0.1


@dataclass
class _FakeCandidate:
    result_path: Path
    config_path: Path
    trace_path: Path
    result: dict
    config: dict
    candidate_id: str
    edge_mm: int
    rank: tuple
    full_success: bool = False
    stage: str = "lift_exact"
    pose_id: str = "pose"
    controller_id: str = "controller"
    candidate_sha256: str = "a" * 64
    thumb_target_rad: float = 1.25


def _summary(*, passed: bool) -> dict:
    return {
        "passed": passed,
        "failed_checks": [] if passed else ["synthetic_failure"],
        "checks": {"synthetic_failure": passed},
        "stage_status": {
            "grasp_success": passed,
            "manipulation_success": passed,
            "full_success": passed,
        },
        "metrics": {"operation_median_lift_m": 0.011 if passed else 0.0},
    }


def test_v8_robustness_runs_16_per_exact_plus_50_best_without_promotion(tmp_path):
    base = load_config(ROOT / DEFAULT_V8_TEMPLATE)
    candidates = []
    for index, edge_mm in enumerate((63, 61, 67)):
        config = copy.deepcopy(base)
        config["cube"]["edge_m"] = edge_mm / 1000.0
        validate_config(config)
        directory = tmp_path / f"source_{index}"
        directory.mkdir()
        config_path = directory / "resolved_config.json"
        result_path = directory / "result.json"
        trace_path = directory / "trace.npz"
        write_json(config_path, config)
        write_json(result_path, {"summary": _summary(passed=False)})
        trace_path.write_bytes(b"synthetic trace")
        candidates.append(
            _FakeCandidate(
                result_path=result_path,
                config_path=config_path,
                trace_path=trace_path,
                result={"summary": _summary(passed=False)},
                config=config,
                candidate_id=str(89_000_000_000_000 + index),
                edge_mm=edge_mm,
                rank=(index,),
                candidate_sha256=canonical_sha256(config),
                pose_id=f"pose_{index}",
                controller_id=f"controller_{index}",
            )
        )

    observed_jobs = []

    def runner(jobs, workers):
        assert workers == 3
        observed_jobs.extend(jobs)
        return [
            {
                "candidate_id": candidate_id,
                "config": copy.deepcopy(config),
                "summary": _summary(passed=True),
            }
            for candidate_id, config in jobs
        ]

    report = run_v8_robustness_campaign(
        [tmp_path],
        tmp_path / "report.json",
        workers=3,
        candidate_discoverer=lambda _roots: tuple(candidates),
        runner=runner,
    )

    assert len(observed_jobs) == 3 * 16 + 50
    assert report["complete"] is True
    assert report["source_exact_count"] == 3
    assert report["total_perturbation_count"] == 98
    assert report["registered_budget_complete"] is True
    assert [value["perturbation_count"] for value in report["per_exact"]] == [16] * 3
    assert report["best_robustness"]["perturbation_count"] == 50
    assert report["best_robustness"]["perturbation_passes"] == 50
    assert report["best_robustness"]["diagnostic_only_due_to_nominal_failure"] is True
    assert report["robust_passed"] is False
    assert report["stop_reason"] == "nominal_exact_candidate_failed_hard_constraints"
