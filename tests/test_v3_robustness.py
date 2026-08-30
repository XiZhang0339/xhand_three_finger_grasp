from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from xhand_grasp import search
from xhand_grasp.config import load_config
from xhand_grasp.experiment import resolve_experiment


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "grasp_configs" / "left_three_finger_cube.json"
V2_LARGE_CONFIG = (
    ROOT / "grasp_configs" / "left_opposed_face_palm_down_large_cube.json"
)
V3_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_grasp_then_lift.json"
)


def _failed_summary() -> dict[str, Any]:
    return {
        "passed": False,
        "failed_checks": ["synthetic_failure"],
        "checks": {
            "synthetic_failure": False,
            "synthetic_control_gate": True,
        },
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": False,
            "full_success": False,
        },
        "metrics": {"synthetic_metric": -1.0},
    }


def _fake_runner(call_sizes: list[int]):
    def run(
        payload: list[tuple[int, dict[str, Any]]], workers: int
    ) -> list[dict[str, Any]]:
        assert workers == 1
        call_sizes.append(len(payload))
        return [
            {
                "candidate_id": candidate_id,
                "config": copy.deepcopy(config),
                "summary": _failed_summary(),
            }
            for candidate_id, config in payload
        ]

    return run


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("fixed_mass", "constant-density nominal config"),
        ("non_nominal_friction", "nominal friction"),
    ],
)
def test_v3_robustness_rejects_noncanonical_nominal_before_simulation(
    mutation, message, monkeypatch
):
    config = load_config(V3_CONFIG)
    campaign = resolve_experiment(config).size_campaign
    assert campaign is not None
    if mutation == "non_nominal_friction":
        config["cube"]["mass_kg"] = campaign.constant_density_mass_kg(
            config["cube"]["edge_m"]
        )
        config["cube"]["friction"] = 0.6

    def must_not_run(*args, **kwargs):
        pytest.fail("robustness must reject the config before any simulation")

    monkeypatch.setattr(search, "_run_candidates", must_not_run)

    with pytest.raises(ValueError, match=message):
        search.robustness(config, workers=1, seed=20260821)


def test_v3_robustness_records_complete_checks_and_stage_status(monkeypatch):
    config = load_config(V3_CONFIG)
    campaign = resolve_experiment(config).size_campaign
    assert campaign is not None
    config["cube"]["mass_kg"] = campaign.constant_density_mass_kg(
        config["cube"]["edge_m"]
    )
    config["cube"]["friction"] = campaign.discovery_friction
    call_sizes: list[int] = []
    monkeypatch.setattr(search, "_run_candidates", _fake_runner(call_sizes))

    result = search.robustness(config, workers=1, seed=20260821)

    assert call_sizes == [1, 100, 50]
    assert len(result["grid"]) == 100
    assert len(result["perturbations"]) == 50
    for record in (*result["grid"], *result["perturbations"]):
        assert record["checks"] == {
            "synthetic_failure": False,
            "synthetic_control_gate": True,
        }
        assert record["stage_status"] == {
            "grasp_success": True,
            "manipulation_success": False,
            "full_success": False,
        }
        assert record["passed"] is False
        assert record["failed_checks"] == ["synthetic_failure"]
        assert record["metrics"] == {"synthetic_metric": -1.0}


@pytest.mark.parametrize("config_path", (V1_CONFIG, V2_LARGE_CONFIG))
def test_v1_v2_robustness_output_shape_remains_unchanged(
    config_path, monkeypatch
):
    config = load_config(config_path)
    call_sizes: list[int] = []
    monkeypatch.setattr(search, "_run_candidates", _fake_runner(call_sizes))

    result = search.robustness(config, workers=1, seed=20260821)

    assert call_sizes == [1, 100, 50]
    for record in (*result["grid"], *result["perturbations"]):
        assert "checks" not in record
        assert "stage_status" not in record
