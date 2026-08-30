from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.experiment import get_experiment, resolve_experiment
from xhand_grasp.experiments.opposed_face_palm_down_larger_cube_grasp_then_lift import (
    CONTROL_PROTOCOL,
    EXPERIMENT_ID,
    LARGER_CUBE_GRASP_THEN_LIFT,
    ROBUSTNESS_CASE_FAMILIES,
    SIZE_CAMPAIGN,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_grasp_then_lift.json"
)


def _read_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_v3_experiment_is_registered_and_config_is_version_locked():
    config = load_config(CONFIG_PATH)
    definition = resolve_experiment(config)

    assert config["schema_version"] == 3
    assert definition is LARGER_CUBE_GRASP_THEN_LIFT
    assert get_experiment(EXPERIMENT_ID) is definition
    assert definition.artifact_root == (
        "artifacts/left_opposed_face_palm_down_larger_cube_grasp_then_lift"
    )
    assert definition.control_protocol is CONTROL_PROTOCOL
    assert config["control_protocol"] == CONTROL_PROTOCOL.as_config()
    assert config["size_campaign"] == SIZE_CAMPAIGN.as_config()
    assert config["robustness"] == definition.robustness.as_config()
    assert config["experiment_status"]["full_success"] is False


def test_grasp_then_manipulate_protocol_has_fixed_timing_and_gate():
    protocol = CONTROL_PROTOCOL
    gate = protocol.grasp_gate

    assert protocol.strategy == "grasp_verify_then_manipulate"
    assert protocol.failure_behavior == "abort_hold_grasp_pose"
    assert protocol.settle_s == pytest.approx(0.5)
    assert protocol.close_s == pytest.approx(1.0)
    assert protocol.verify_timeout_s == pytest.approx(0.75)
    assert protocol.stable_window_s == pytest.approx(0.25)
    assert protocol.manipulate_s == pytest.approx(1.5)
    assert protocol.min_hold_s == pytest.approx(1.0)
    assert protocol.total_duration_s == pytest.approx(4.75)
    assert gate.min_target_face_force_n == pytest.approx(0.05)
    assert gate.min_target_force_fraction == pytest.approx(0.95)
    assert gate.require_touch is True
    assert gate.require_support_contact is True
    assert gate.max_translation_m == pytest.approx(0.0005)
    assert gate.max_orientation_drift_deg == pytest.approx(3.0)
    assert gate.max_linear_speed_m_s == pytest.approx(0.01)
    assert gate.max_early_lift_m == pytest.approx(0.001)


def test_v3_control_uses_one_grasp_pose_and_a_bounded_relative_delta():
    config = load_config(CONFIG_PATH)
    bounds = LARGER_CUBE_GRASP_THEN_LIFT.search_bounds
    control = config["control"]

    assert set(control) == {"grasp_targets_rad", "manipulation_delta_rad"}
    assert set(control["grasp_targets_rad"]) == set(ACTIVE_ACTUATORS)
    assert set(control["manipulation_delta_rad"]) == set(ACTIVE_ACTUATORS)
    assert bounds.final_target_delta_rad is None
    assert bounds.contains_targets(control["grasp_targets_rad"])
    assert bounds.contains_manipulation_delta(control["manipulation_delta_rad"])
    assert "timing" not in config


def test_larger_size_campaign_and_material_budgets_match_the_plan():
    assert SIZE_CAMPAIGN.coarse_edges_m == pytest.approx(
        (0.052, 0.054, 0.056, 0.058, 0.060, 0.062, 0.064)
    )
    assert SIZE_CAMPAIGN.coarse_samples_per_pitch == 5_000
    assert SIZE_CAMPAIGN.coarse_size_count == 3
    assert SIZE_CAMPAIGN.odd_samples_per_pitch == 10_000
    assert SIZE_CAMPAIGN.exact_size_count == 2
    assert SIZE_CAMPAIGN.fine_samples_per_pitch == 50_000
    assert SIZE_CAMPAIGN.dynamic_candidate_count == 512
    assert SIZE_CAMPAIGN.fixed_mass_local_refinement_count == 1_024
    assert SIZE_CAMPAIGN.manipulation_seed_count == 8
    assert SIZE_CAMPAIGN.manipulation_refine_per_seed == 128
    assert SIZE_CAMPAIGN.manipulation_local_refinement_count == 1_024
    assert SIZE_CAMPAIGN.constant_density_max_candidates == 16
    assert SIZE_CAMPAIGN.density_local_refinement_count == 512
    assert SIZE_CAMPAIGN.finalist_count == 16
    assert SIZE_CAMPAIGN.perturbations_per_final == 16
    assert SIZE_CAMPAIGN.target_sampling_policy == (
        "grasp_targets_plus_manipulation_delta"
    )
    assert SIZE_CAMPAIGN.density_kg_m3 == pytest.approx(740.7407407407407)
    assert SIZE_CAMPAIGN.constant_density_mass_kg(0.052) == pytest.approx(
        0.1041540740740741
    )
    assert SIZE_CAMPAIGN.constant_density_mass_kg(0.064) == pytest.approx(
        0.19418074074074077
    )


def test_larger_cube_robustness_family_is_75_plus_25_cases():
    families = ROBUSTNESS_CASE_FAMILIES

    assert families.edge_window_m(0.052) == pytest.approx(
        (0.052, 0.053, 0.054, 0.055, 0.056)
    )
    assert families.edge_window_m(0.058) == pytest.approx(
        (0.056, 0.057, 0.058, 0.059, 0.060)
    )
    assert families.edge_window_m(0.064) == pytest.approx(
        (0.060, 0.061, 0.062, 0.063, 0.064)
    )
    assert families.constant_density_case_count == 75
    assert families.fixed_mass_case_count == 25
    assert families.total_case_count == 100
    assert LARGER_CUBE_GRASP_THEN_LIFT.robustness.perturbation_count == 50
    assert LARGER_CUBE_GRASP_THEN_LIFT.robustness.required_pass_count == 45


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        (
            "control_protocol",
            lambda config: config["control_protocol"].__setitem__(
                "stable_window_s", 0.20
            ),
        ),
        (
            "size_campaign",
            lambda config: config["size_campaign"]["budget"].__setitem__(
                "manipulation_seed_count", 7
            ),
        ),
        (
            "robustness",
            lambda config: config["robustness"]["case_families"].__setitem__(
                "edge_limits_m", [0.050, 0.064]
            ),
        ),
    ],
)
def test_v3_versioned_blocks_reject_mutation(section, mutate):
    config = copy.deepcopy(_read_config())
    mutate(config)

    with pytest.raises(ValueError, match=section):
        validate_config(config)


def test_v3_rejects_legacy_control_and_out_of_bounds_delta():
    legacy_control = copy.deepcopy(_read_config())
    legacy_control["control"]["final_targets_rad"] = legacy_control["control"].pop(
        "manipulation_delta_rad"
    )
    with pytest.raises(ValueError, match="schema v3 control"):
        validate_config(legacy_control)

    invalid_delta = copy.deepcopy(_read_config())
    actuator = ACTIVE_ACTUATORS[0]
    invalid_delta["control"]["manipulation_delta_rad"][actuator] = 10.0
    with pytest.raises(ValueError, match="manipulation_delta_rad"):
        validate_config(invalid_delta)


def test_existing_v1_and_v2_configs_still_validate_unchanged():
    v1 = load_config(ROOT / "grasp_configs" / "left_three_finger_cube.json")
    v2 = load_config(ROOT / "grasp_configs" / "left_opposed_face_palm_down.json")
    large_v2 = load_config(
        ROOT / "grasp_configs" / "left_opposed_face_palm_down_large_cube.json"
    )

    assert v1["schema_version"] == 1
    assert v2["schema_version"] == 2
    assert large_v2["schema_version"] == 2
    assert "control_protocol" not in v1
    assert "control_protocol" not in v2
    assert "control_protocol" not in large_v2
    assert large_v2["size_campaign"] == resolve_experiment(
        large_v2
    ).size_campaign.as_config()


def test_feedback_gated_experiment_cannot_be_selected_from_schema_v2():
    config = copy.deepcopy(_read_config())
    config["schema_version"] = 2
    config["timing"] = {
        "settle_s": 0.5,
        "pregrasp_s": 1.0,
        "lift_s": 1.5,
        "hold_s": 1.0,
    }
    config["control"] = {
        "pregrasp_targets_rad": copy.deepcopy(
            _read_config()["control"]["grasp_targets_rad"]
        ),
        "final_targets_rad": copy.deepcopy(
            _read_config()["control"]["grasp_targets_rad"]
        ),
    }

    with pytest.raises(ValueError, match="schema_version 3"):
        resolve_experiment(config)
