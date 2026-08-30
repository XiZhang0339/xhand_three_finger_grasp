from __future__ import annotations

import copy
from pathlib import Path

import pytest

from xhand_grasp.config import (
    contact_preload_targets,
    load_config,
    precontact_targets,
    validate_config,
)
from xhand_grasp.experiment import (
    ACTIVE_ACTUATORS,
    ActualContactGraspPoseCampaignParameters,
    ActualContactGraspPoseSettings,
    get_experiment,
    resolve_experiment,
)
from xhand_grasp.experiments.opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift import (
    CAMPAIGN,
    EXPERIMENT_ID,
    GRASP_POSE,
)


ROOT = Path(__file__).resolve().parents[1]
V9_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift.json"
)
V8_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift.json"
)
THUMB = "left_hand_thumb_bend_joint_actuator"


@pytest.fixture
def v9_config() -> dict:
    return load_config(V9_CONFIG)


def test_v9_definition_and_budget_are_registered_and_versioned(v9_config):
    definition = get_experiment(EXPERIMENT_ID)

    assert resolve_experiment(v9_config) is definition
    assert definition.actual_contact_grasp_pose is GRASP_POSE
    assert definition.actual_contact_grasp_pose_campaign is CAMPAIGN
    assert definition.tuning_strategy == (
        "actual_contact_grasp_pose_smooth_vertical_lift"
    )
    assert CAMPAIGN.cell_count == 55
    assert CAMPAIGN.quick_static_sample_count == 110_000
    assert CAMPAIGN.full_static_sample_count == 550_000
    assert CAMPAIGN.maximum_dynamic_grasp_candidate_count == 1_760
    assert v9_config["search"]["budget"] == CAMPAIGN.budget_config()
    assert v9_config["actual_contact_grasp_pose_campaign"] == CAMPAIGN.as_config()
    assert CAMPAIGN.source_pose_manifest.endswith("source_pose_manifest.json")


def test_v9_separates_measured_grasp_qpos_from_controller_preload(v9_config):
    assert set(v9_config["control"]) == {
        "precontact_targets_rad",
        "contact_preload_targets_rad",
        "manipulation_delta_rad",
        "close_profile",
    }
    assert precontact_targets(v9_config) is v9_config["control"][
        "precontact_targets_rad"
    ]
    assert contact_preload_targets(v9_config) is v9_config["control"][
        "contact_preload_targets_rad"
    ]
    assert v9_config["grasp_pose"]["nominal_joint_qpos_rad"][THUMB] == 1.5
    assert contact_preload_targets(v9_config)[THUMB] == 1.6

    preload_only = copy.deepcopy(v9_config)
    preload_only["control"]["contact_preload_targets_rad"][THUMB] = 1.7
    validate_config(preload_only)
    assert preload_only["grasp_pose"]["nominal_joint_qpos_rad"][THUMB] == 1.5


def test_v9_rejects_command_aliases_and_incomplete_close_profile(v9_config):
    aliased = copy.deepcopy(v9_config)
    aliased["control"]["grasp_targets_rad"] = aliased["control"].pop(
        "contact_preload_targets_rad"
    )
    with pytest.raises(ValueError, match="schema v9 control must contain exactly"):
        validate_config(aliased)

    missing_profile = copy.deepcopy(v9_config)
    missing_profile["control"].pop("close_profile")
    with pytest.raises(ValueError, match="schema v9 control must contain exactly"):
        validate_config(missing_profile)

    incomplete_profile = copy.deepcopy(v9_config)
    incomplete_profile["control"]["close_profile"].pop(THUMB)
    with pytest.raises(ValueError, match="close_profile must contain exactly"):
        validate_config(incomplete_profile)


def test_v9_grasp_pose_requires_exact_measured_joint_contract(v9_config):
    missing_joint = copy.deepcopy(v9_config)
    missing_joint["grasp_pose"]["nominal_joint_qpos_rad"].pop(THUMB)
    with pytest.raises(ValueError, match="exactly the eight active actuators"):
        validate_config(missing_joint)

    nonfinite = copy.deepcopy(v9_config)
    nonfinite["grasp_pose"]["nominal_joint_qpos_rad"][THUMB] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        validate_config(nonfinite)

    outside_actual_range = copy.deepcopy(v9_config)
    outside_actual_range["grasp_pose"]["nominal_joint_qpos_rad"][THUMB] = 1.39
    with pytest.raises(ValueError, match="nominal thumb bend qpos"):
        validate_config(outside_actual_range)

    weakened = copy.deepcopy(v9_config)
    weakened["grasp_pose"]["max_nominal_joint_error_rad"] = 0.4
    with pytest.raises(ValueError, match="grasp_pose thresholds"):
        validate_config(weakened)


def test_v9_nominal_and_threshold_settings_have_closed_validation():
    settings = ActualContactGraspPoseSettings(
        thumb_actual_range_rad=(1.4, 1.6),
        max_nominal_joint_error_rad=0.04,
        max_joint_stability_span_rad=0.03,
        verify_continuous_s=0.25,
    )
    nominal = {name: 0.0 for name in ACTIVE_ACTUATORS}
    nominal[THUMB] = 1.5
    resolved = settings.resolved_config(nominal)
    assert resolved["nominal_joint_qpos_rad"][THUMB] == 1.5
    assert resolved["qpos_reference"] == "stable_contact_window_actual_qpos"

    with pytest.raises(ValueError, match="every stable-window thumb sample"):
        ActualContactGraspPoseSettings(
            thumb_actual_range_rad=(1.4, 1.6),
            max_nominal_joint_error_rad=0.04,
            max_joint_stability_span_rad=0.03,
            verify_continuous_s=0.25,
            require_all_thumb_samples_in_range=False,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config["cube"].__setitem__("edge_m", 0.0605), "edge"),
        (lambda config: config["cube"].__setitem__("mass_kg", 0.2), "mass"),
        (lambda config: config["cube"].__setitem__("friction", 0.9), "friction"),
        (
            lambda config: config["cube"]["rpy_deg"].__setitem__(2, 30.0),
            "yaw",
        ),
    ],
)
def test_v9_nominal_cube_policy_is_fixed(v9_config, mutate, message):
    changed = copy.deepcopy(v9_config)
    mutate(changed)
    with pytest.raises(ValueError, match=message):
        validate_config(changed)


def test_v9_campaign_class_rejects_inconsistent_quick_full_budgets():
    values = dict(CAMPAIGN.__dict__)
    values["quick_static_samples_per_cell"] = 10_000
    with pytest.raises(ValueError, match="quick static budget"):
        ActualContactGraspPoseCampaignParameters(**values)


def test_v8_behavior_and_control_helpers_remain_compatible():
    v8 = load_config(V8_CONFIG)
    assert v8["schema_version"] == 8
    assert precontact_targets(v8) is v8["control"]["pregrasp_targets_rad"]
    assert contact_preload_targets(v8) is v8["control"]["grasp_targets_rad"]

    wrong_version = copy.deepcopy(v8)
    wrong_version["schema_version"] = 9
    with pytest.raises(ValueError, match="schema v9 control must contain exactly"):
        validate_config(wrong_version)


def test_v9_definition_cannot_be_selected_by_an_older_schema(v9_config):
    wrong_version = copy.deepcopy(v9_config)
    wrong_version["schema_version"] = 8
    with pytest.raises(ValueError, match="schema v8 control must contain exactly"):
        validate_config(wrong_version)
