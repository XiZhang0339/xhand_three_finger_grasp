from __future__ import annotations

import copy
from pathlib import Path

import pytest

from xhand_grasp import search as search_module
from xhand_grasp.config import load_config
from xhand_grasp.experiment import resolve_experiment


ROOT = Path(__file__).resolve().parents[1]
LARGE_CONFIG = ROOT / "grasp_configs" / "left_opposed_face_palm_down_large_cube.json"


@pytest.fixture
def large_config() -> dict:
    return load_config(LARGE_CONFIG)


def test_large_cube_robustness_grid_has_75_density_and_25_control_cases(
    large_config,
):
    grid, perturbations = search_module.robustness_cases(large_config)
    definition = resolve_experiment(large_config)
    campaign = definition.size_campaign
    assert campaign is not None

    assert len(grid) == definition.robustness.grid_case_count == 100
    assert len(perturbations) == 50

    density_cases = grid[:75]
    control_cases = grid[75:]
    assert sorted({case["cube"]["edge_m"] for case in grid}) == pytest.approx(
        [0.038, 0.039, 0.040, 0.041, 0.042]
    )
    assert all(
        case["cube"]["mass_kg"]
        == pytest.approx(
            campaign.constant_density_mass_kg(case["cube"]["edge_m"])
            * scale
        )
        for edge_index in range(5)
        for scale_index, scale in enumerate((0.9, 1.0, 1.1))
        for friction_index in range(5)
        for case in [density_cases[edge_index * 15 + scale_index * 5 + friction_index]]
    )
    assert all(case["cube"]["mass_kg"] == pytest.approx(0.020) for case in control_cases)


@pytest.mark.parametrize(
    ("nominal_edge_m", "expected"),
    [
        (0.036, (0.036, 0.037, 0.038, 0.039, 0.040)),
        (0.040, (0.038, 0.039, 0.040, 0.041, 0.042)),
        (0.050, (0.046, 0.047, 0.048, 0.049, 0.050)),
    ],
)
def test_large_cube_robustness_edge_window_clamps_at_campaign_limits(
    large_config, nominal_edge_m, expected
):
    config = copy.deepcopy(large_config)
    config["cube"]["edge_m"] = nominal_edge_m
    grid, _ = search_module.robustness_cases(config)
    assert tuple(sorted({case["cube"]["edge_m"] for case in grid})) == pytest.approx(
        expected
    )


def test_large_cube_robustness_report_labels_case_families(
    large_config, monkeypatch
):
    call_sizes: list[int] = []

    def fake_run_candidates(payload, workers):
        assert workers == 1
        call_sizes.append(len(payload))
        return [
            {
                "candidate_id": candidate_id,
                "config": config,
                "summary": {
                    "passed": False,
                    "failed_checks": ["simulation_error"],
                    "checks": {},
                    "metrics": {},
                },
            }
            for candidate_id, config in payload
        ]

    monkeypatch.setattr(search_module, "_run_candidates", fake_run_candidates)
    result = search_module.robustness(large_config, workers=1)

    assert call_sizes == [1, 100, 50]
    assert result["grid_case_family_counts"] == {
        "constant_density": 75,
        "fixed_20g_control": 25,
    }
    assert [record["case_family"] for record in result["grid"][:75]] == [
        "constant_density"
    ] * 75
    assert [record["case_family"] for record in result["grid"][75:]] == [
        "fixed_20g_control"
    ] * 25
    assert all("edge_m" in record and "mass_kg" in record for record in result["grid"])


def test_fixed_20g_nominal_cannot_be_reported_as_robust(
    large_config, monkeypatch
):
    def run_once(config):
        call_index = 0

        def fake_run_candidates(payload, workers):
            nonlocal call_index
            current = call_index
            call_index += 1
            return [
                {
                    "candidate_id": candidate_id,
                    "config": candidate,
                    "summary": {
                        "passed": current != 1,
                        "failed_checks": [] if current != 1 else ["mock_grid_failure"],
                        "checks": {},
                        "metrics": {},
                    },
                }
                for candidate_id, candidate in payload
            ]

        monkeypatch.setattr(search_module, "_run_candidates", fake_run_candidates)
        return search_module.robustness(config, workers=1)

    fixed_mass = run_once(copy.deepcopy(large_config))
    assert fixed_mass["nominal_hard_constraints_passed"] is True
    assert fixed_mass["nominal_is_constant_density"] is False
    assert fixed_mass["perturbation_passes"] == 50
    assert fixed_mass["robust_passed"] is False

    density_config = copy.deepcopy(large_config)
    campaign = resolve_experiment(density_config).size_campaign
    assert campaign is not None
    density_config["cube"]["mass_kg"] = campaign.constant_density_mass_kg(
        density_config["cube"]["edge_m"]
    )
    density = run_once(density_config)
    assert density["nominal_is_constant_density"] is True
    assert density["nominal_friction_matches"] is True
    assert density["robust_passed"] is True

    density_config["cube"]["friction"] = 1.2
    non_nominal_friction = run_once(density_config)
    assert non_nominal_friction["nominal_is_constant_density"] is True
    assert non_nominal_friction["nominal_friction_matches"] is False
    assert non_nominal_friction["robust_passed"] is False


def test_hardest_passing_case_excludes_fixed_mass_control(
    large_config, monkeypatch
):
    call_index = 0

    def fake_run_candidates(payload, workers):
        nonlocal call_index
        current = call_index
        call_index += 1
        results = []
        for payload_index, (candidate_id, config) in enumerate(payload):
            passed = current == 1 and payload_index in (0, 75)
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": config,
                    "summary": {
                        "passed": passed,
                        "failed_checks": [] if passed else ["mock_failure"],
                        "checks": {},
                        "metrics": {"score": -1.0 if payload_index == 75 else 0.5},
                    },
                }
            )
        return results

    monkeypatch.setattr(search_module, "_run_candidates", fake_run_candidates)
    monkeypatch.setattr(
        search_module,
        "normalized_acceptance_margins",
        lambda metrics, acceptance, contact_topology: {"score": metrics["score"]},
    )

    result = search_module.robustness(large_config, workers=1)

    assert result["grid_case_family_passes"] == {
        "constant_density": 1,
        "fixed_20g_control": 1,
    }
    assert result["hardest_passing_grid_case"]["grid_index"] == 0
    assert result["hardest_passing_grid_case"]["case_family"] == "constant_density"
    assert result["hardest_passing_fixed_20g_control_grid_case"]["grid_index"] == 75
