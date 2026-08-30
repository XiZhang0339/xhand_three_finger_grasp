from __future__ import annotations

import copy
from pathlib import Path

import pytest

from xhand_grasp.config import load_config, validate_config
from xhand_grasp.experiment import get_experiment, resolve_experiment
from xhand_grasp.experiments.opposed_face_palm_down_high_thumb_variable_size_pose_preserving import (
    COARSE_EDGES_M,
    COARSE_THUMB_TARGETS_RAD,
    EXPERIMENT_DEFINITION,
    EXPERIMENT_ID,
    HIGH_THUMB_SIZE_CAMPAIGN,
    MANIPULATION_DELTA_BOUNDS_RAD,
    PREGRASP_TARGET_BOUNDS_RAD,
    THUMB_BEND_ACTUATOR,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_high_thumb_variable_size_"
    "pose_preserving_grasp_then_lift.json"
)


@pytest.fixture
def v7_config() -> dict:
    return load_config(CONFIG_PATH)


def test_v7_template_is_independently_registered(v7_config):
    definition = resolve_experiment(v7_config)

    assert v7_config["schema_version"] == 7
    assert v7_config["experiment_id"] == EXPERIMENT_ID
    assert definition is EXPERIMENT_DEFINITION
    assert get_experiment(EXPERIMENT_ID) is definition
    assert definition.tuning_strategy == (
        "high_thumb_variable_size_pose_preserving"
    )
    assert definition.far_hand_campaign is None
    assert definition.high_thumb_size_campaign is HIGH_THUMB_SIZE_CAMPAIGN
    assert v7_config["high_thumb_size_campaign"] == (
        HIGH_THUMB_SIZE_CAMPAIGN.as_config()
    )


def test_v7_campaign_declares_exact_grids_material_and_budget():
    campaign = HIGH_THUMB_SIZE_CAMPAIGN

    assert COARSE_EDGES_M == pytest.approx(
        (0.052, 0.054, 0.056, 0.058, 0.060, 0.062, 0.064, 0.066, 0.068, 0.070)
    )
    assert COARSE_THUMB_TARGETS_RAD == pytest.approx(
        (1.25, 1.30, 1.35, 1.40, 1.45)
    )
    assert campaign.edge_fine_step_m == pytest.approx(0.001)
    assert campaign.thumb_fine_step_rad == pytest.approx(0.01)
    assert campaign.fixed_mass_kg == pytest.approx(0.160)
    assert campaign.friction == pytest.approx(0.8)
    assert campaign.cube_center_xy_m == pytest.approx((0.071, -0.027))
    assert campaign.cell_count == 50
    assert campaign.static_sample_count == 1_000_000
    assert campaign.dynamic_candidate_count == 400
    assert campaign.local_refine_seed_count == 40
    assert campaign.local_refinement_count == 2_560
    assert campaign.fine_dynamic_candidate_count == 320
    assert campaign.exact_candidate_count == 48
    assert campaign.selected_grasp_count == 12
    assert campaign.selected_lift_seed_count == 6
    assert campaign.manipulation_refinement_count == 1_536


def test_v7_control_bounds_keep_high_thumb_and_separate_acquisition(v7_config):
    campaign = HIGH_THUMB_SIZE_CAMPAIGN
    bounds = EXPERIMENT_DEFINITION.search_bounds

    assert bounds.actuator_targets_rad[THUMB_BEND_ACTUATOR] == (1.25, 1.45)
    assert bounds.pregrasp_targets_rad == PREGRASP_TARGET_BOUNDS_RAD
    assert bounds.manipulation_delta_rad == MANIPULATION_DELTA_BOUNDS_RAD
    assert campaign.grasp_target_bounds_rad == bounds.actuator_targets_rad
    assert campaign.pregrasp_target_bounds_rad == bounds.pregrasp_targets_rad
    assert campaign.manipulation_delta_bounds_rad == (
        bounds.manipulation_delta_rad
    )
    assert v7_config["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR] == (
        pytest.approx(1.25)
    )
    assert set(v7_config["control"]["manipulation_delta_rad"].values()) == {
        0
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["cube"].__setitem__("edge_m", 0.051), "size range"),
        (lambda value: value["cube"].__setitem__("edge_m", 0.0525), "fine edge grid"),
        (lambda value: value["cube"].__setitem__("mass_kg", 0.161), "cube mass"),
        (lambda value: value["cube"].__setitem__("friction", 0.81), "cube friction"),
        (
            lambda value: value["cube"].__setitem__(
                "center_xy_m", [0.0711, -0.027]
            ),
            "center_xy_m",
        ),
        (lambda value: value["cube"].__setitem__("z_offset_m", 0.0001), "z_offset"),
    ],
)
def test_v7_rejects_cube_pose_or_material_sampling(v7_config, mutation, message):
    changed = copy.deepcopy(v7_config)
    mutation(changed)

    with pytest.raises(ValueError, match=message):
        validate_config(changed)


def test_v7_rejects_campaign_or_high_thumb_bound_mutation(v7_config):
    campaign = copy.deepcopy(v7_config)
    campaign["high_thumb_size_campaign"]["fixed_mass_kg"] = 0.170
    with pytest.raises(ValueError, match="high_thumb_size_campaign"):
        validate_config(campaign)

    target = copy.deepcopy(v7_config)
    target["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR] = 1.46
    with pytest.raises(ValueError, match="registered search bounds"):
        validate_config(target)

    pregrasp = copy.deepcopy(v7_config)
    pregrasp["control"]["pregrasp_targets_rad"][THUMB_BEND_ACTUATOR] = 1.31
    with pytest.raises(ValueError, match="pregrasp_targets_rad"):
        validate_config(pregrasp)


def test_v7_pose_preservation_is_sticky_and_not_a_v6_alias(v7_config):
    missing = copy.deepcopy(v7_config)
    del missing["pose_preservation"]
    with pytest.raises(ValueError, match="pose_preservation"):
        validate_config(missing)

    wrong_schema = copy.deepcopy(v7_config)
    wrong_schema["schema_version"] = 6
    with pytest.raises(ValueError, match="schema_version 6"):
        resolve_experiment(wrong_schema)
