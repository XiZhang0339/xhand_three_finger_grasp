from __future__ import annotations

from pathlib import Path
import copy

import pytest

import xhand_grasp.aligned_contacts_tuning as aligned_tuning
import xhand_grasp.search as search
from xhand_grasp.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift.json"
)


def test_public_tune_dispatches_schema_v4_to_executable_campaign(monkeypatch):
    config = load_config(CONFIG)
    captured = {}

    def fake_tuner(value, **kwargs):
        captured["config"] = value
        captured.update(kwargs)
        return {"campaign_kind": "aligned_contacts"}

    monkeypatch.setattr(aligned_tuning, "tune_aligned_contacts", fake_tuner)

    result = search.tune(
        config,
        samples=7,
        refine_top=3,
        refine_per=2,
        workers=4,
        seed=20260821,
        kinematic_samples_per_pitch=20,
        dynamic_candidate_count=10,
        local_refine_seed_count=5,
        local_refine_per_seed=2,
        final_candidate_count=5,
        perturbations_per_final=3,
        fallback_physics_count=0,
        fallback_kinematic_samples_per_pitch=11,
    )

    assert result == {"campaign_kind": "aligned_contacts"}
    assert captured["config"] is config
    assert captured["workers"] == 4
    assert captured["seed"] == 20260821
    assert captured["run_candidates"] is search._run_candidates
    assert captured["kinematic_samples_per_pitch"] == 20
    assert captured["dynamic_candidate_count"] == 10
    assert captured["local_refine_seed_count"] == 5
    assert captured["local_refine_per_seed"] == 2
    assert captured["final_candidate_count"] == 5
    assert captured["perturbations_per_final"] == 3
    assert captured["fallback_physics_count"] == 0
    assert captured["fallback_kinematic_samples_per_pitch"] == 11
    assert captured["legacy_parameters"] == {
        "samples": 7,
        "refine_top": 3,
        "refine_per": 2,
    }


def test_public_robustness_uses_v4_pose_and_material_envelope(monkeypatch):
    config = load_config(CONFIG)
    generated = []

    def fake_factory(parent, *, count, seed, definition):
        assert parent is config
        assert count == 50
        assert seed == 123
        assert definition.aligned_contact_campaign is not None
        for index in range(count):
            trial = copy.deepcopy(parent)
            trial["run_context"] = {"kind": "robustness_trial"}
            trial["candidate_metadata"] = {
                "resolved_perturbations": {"trial": index}
            }
            generated.append(trial)
        return generated

    def fake_runner(payloads, workers):
        assert workers == 3
        results = []
        for candidate_id, trial in payloads:
            passed = candidate_id == 0 or candidate_id <= 45
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": trial,
                    "summary": {
                        "passed": passed,
                        "failed_checks": [] if passed else ["synthetic"],
                        "checks": {"synthetic": passed},
                        "metrics": {"synthetic": float(passed)},
                        "stage_status": {
                            "grasp_success": passed,
                            "manipulation_success": passed,
                            "full_success": passed,
                        },
                    },
                }
            )
        return results

    monkeypatch.setattr(
        aligned_tuning, "generate_aligned_perturbation_configs", fake_factory
    )
    monkeypatch.setattr(search, "_run_candidates", fake_runner)

    result = search.robustness(config, workers=3, seed=123)

    assert result["campaign_kind"] == "aligned_contacts"
    assert result["grid_case_count"] == 0
    assert result["grid"] == []
    assert result["perturbation_trial_count"] == 50
    assert result["perturbation_passes"] == 45
    assert result["required_perturbation_passes"] == 45
    assert result["robustness_manifest"] == {
        "seed": 123,
        "perturbation_count": 50,
        "required_pass_count": 45,
        "perturbation_envelope": config["aligned_contact_campaign"][
            "perturbation_envelope"
        ],
    }
    assert result["nominal_passed"] is True
    assert result["robust_passed"] is True
    assert result["perturbations"][0]["resolved_perturbations"] == {"trial": 0}
    first_status = result["perturbations"][0]["resolved_trial_config"][
        "experiment_status"
    ]
    assert first_status["classification"] == "robustness_trial"
    assert first_status["robustness_passed"] is False
    assert result["perturbations"][-1]["passed"] is False


def test_robustness_cases_exposes_v4_perturbations_without_legacy_grid(monkeypatch):
    config = load_config(CONFIG)

    def fake_factory(parent, *, count, seed, definition):
        assert parent is config
        assert count == 50
        assert seed == 321
        assert definition.tuning_strategy == "aligned_contacts"
        return [copy.deepcopy(parent) for _ in range(count)]

    monkeypatch.setattr(
        aligned_tuning, "generate_aligned_perturbation_configs", fake_factory
    )

    grid, perturbations = search.robustness_cases(config, seed=321)

    assert grid == []
    assert len(perturbations) == 50


def test_v4_robustness_rejects_noncanonical_override_context(monkeypatch):
    config = load_config(CONFIG)
    config["run_context"] = {"kind": "parameter_override_run"}

    def forbidden_runner(payloads, workers):  # pragma: no cover - assertion path
        raise AssertionError("an override config must be rejected before simulation")

    monkeypatch.setattr(search, "_run_candidates", forbidden_runner)

    with pytest.raises(ValueError, match="requires a canonical config"):
        search.robustness(config, workers=1, seed=20260821)


def test_v4_robustness_requires_exactly_registered_case_count(monkeypatch):
    config = load_config(CONFIG)

    def short_factory(parent, *, count, seed, definition):
        return [copy.deepcopy(parent) for _ in range(count - 1)]

    monkeypatch.setattr(
        aligned_tuning, "generate_aligned_perturbation_configs", short_factory
    )

    with pytest.raises(RuntimeError, match="49 cases; expected 50"):
        search.robustness_cases(config, seed=20260821)

    def nominal_runner(payloads, workers):
        assert len(payloads) == 1
        candidate_id, trial = payloads[0]
        return [
            {
                "candidate_id": candidate_id,
                "config": trial,
                "summary": {
                    "passed": True,
                    "failed_checks": [],
                    "checks": {},
                    "metrics": {},
                    "stage_status": {
                        "grasp_success": True,
                        "manipulation_success": True,
                        "full_success": True,
                    },
                },
            }
        ]

    monkeypatch.setattr(search, "_run_candidates", nominal_runner)
    with pytest.raises(RuntimeError, match="49 cases; expected 50"):
        search.robustness(config, workers=1, seed=20260821)


def test_v4_robustness_requires_exactly_registered_runner_results(monkeypatch):
    config = load_config(CONFIG)

    def factory(parent, *, count, seed, definition):
        return [copy.deepcopy(parent) for _ in range(count)]

    calls = 0

    def short_runner(payloads, workers):
        nonlocal calls
        calls += 1
        selected = payloads if calls == 1 else payloads[:-1]
        return [
            {
                "candidate_id": candidate_id,
                "config": trial,
                "summary": {
                    "passed": True,
                    "failed_checks": [],
                    "checks": {},
                    "metrics": {},
                    "stage_status": {
                        "grasp_success": True,
                        "manipulation_success": True,
                        "full_success": True,
                    },
                },
            }
            for candidate_id, trial in selected
        ]

    monkeypatch.setattr(
        aligned_tuning, "generate_aligned_perturbation_configs", factory
    )
    monkeypatch.setattr(search, "_run_candidates", short_runner)

    with pytest.raises(RuntimeError, match="49 results; expected 50"):
        search.robustness(config, workers=1, seed=20260821)
