from __future__ import annotations

import copy
from pathlib import Path

import pytest

from xhand_grasp.config import (
    ACTIVE_ACTUATORS,
    load_config,
    resolved_pose_constraint_values,
    validate_config,
)
from xhand_grasp.experiment import (
    PosePreservationSettings,
    get_experiment,
    resolve_experiment,
)
from xhand_grasp.experiments.opposed_face_palm_down_pose_preserving_grasp import (
    EXPERIMENT_DEFINITION,
    EXPERIMENT_ID,
    FAR_HAND_CAMPAIGN,
    POSE_CONSTRAINTS,
    POSE_PRESERVATION,
    PREGRASP_TARGET_BOUNDS_RAD,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_pose_preserving_grasp.json"
)
V5_CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift.json"
)


@pytest.fixture
def v6_config() -> dict:
    return load_config(CONFIG_PATH)


def test_v6_template_resolves_to_its_registered_definition(v6_config):
    definition = resolve_experiment(v6_config)
    resolved = resolved_pose_constraint_values(v6_config)

    assert v6_config["schema_version"] == 6
    assert definition is EXPERIMENT_DEFINITION
    assert get_experiment(EXPERIMENT_ID) is definition
    assert definition.tuning_strategy == "pose_preserving_grasp"
    assert definition.pose_preservation is POSE_PRESERVATION
    assert v6_config["pose_preservation"] == POSE_PRESERVATION.as_config()
    assert v6_config["pose_constraints"] == POSE_CONSTRAINTS.as_config()
    assert v6_config["far_hand_campaign"] == FAR_HAND_CAMPAIGN.as_config()
    assert resolved["root_cube_distance_m"] == pytest.approx(
        0.15275895233063616
    )
    assert resolved["finger_down_tilt_deg"] == pytest.approx(
        31.823680536104167
    )
    assert resolved["palm_plane_ground_angle_deg"] == pytest.approx(
        31.83473317402027
    )


def test_v6_has_distinct_pregrasp_bounds_and_exact_close_profile(v6_config):
    bounds = EXPERIMENT_DEFINITION.search_bounds
    control = v6_config["control"]

    assert bounds.pregrasp_targets_rad == PREGRASP_TARGET_BOUNDS_RAD
    assert bounds.contains_pregrasp_targets(control["pregrasp_targets_rad"])
    assert bounds.contains_targets(control["grasp_targets_rad"])
    assert bounds.contains_manipulation_delta(
        control["manipulation_delta_rad"]
    )
    assert set(control["close_profile"]) == set(ACTIVE_ACTUATORS)
    assert control["close_profile"][
        "left_hand_mid_joint1_actuator"
    ]["start_fraction"] == 0.0
    assert control["close_profile"][
        "left_hand_index_joint1_actuator"
    ]["start_fraction"] == 0.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_translation_m": 0.0}, "max_translation_m"),
        ({"max_orientation_drift_deg": 181.0}, "max_orientation"),
        ({"reference": "first_contact_pose"}, "reference"),
        ({"scope": "verify_only"}, "scope"),
        ({"require_support_contact": 1}, "boolean"),
    ],
)
def test_pose_preservation_contract_rejects_weakened_semantics(
    changes, message
):
    values = {
        "max_translation_m": 0.0005,
        "max_orientation_drift_deg": 1.0,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        PosePreservationSettings(**values)


def test_v6_config_rejects_missing_or_changed_pose_contract(v6_config):
    missing = copy.deepcopy(v6_config)
    del missing["pose_preservation"]
    with pytest.raises(ValueError, match="pose_preservation"):
        validate_config(missing)

    changed = copy.deepcopy(v6_config)
    changed["pose_preservation"]["max_translation_m"] = 0.001
    with pytest.raises(ValueError, match="pose_preservation"):
        validate_config(changed)


def test_v6_config_rejects_invalid_pregrasp_and_close_profile(v6_config):
    pregrasp = copy.deepcopy(v6_config)
    pregrasp["control"]["pregrasp_targets_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] = 0.59
    with pytest.raises(ValueError, match="pregrasp_targets_rad"):
        validate_config(pregrasp)

    missing_actuator = copy.deepcopy(v6_config)
    del missing_actuator["control"]["close_profile"][ACTIVE_ACTUATORS[-1]]
    with pytest.raises(ValueError, match="eight active actuators"):
        validate_config(missing_actuator)

    invalid_interval = copy.deepcopy(v6_config)
    invalid_interval["control"]["close_profile"][ACTIVE_ACTUATORS[0]] = {
        "start_fraction": 0.7,
        "end_fraction": 0.7,
    }
    with pytest.raises(ValueError, match="start_fraction < end_fraction"):
        validate_config(invalid_interval)


def test_schema_v5_rejects_v6_fields_and_v6_rejects_v5_definition(v6_config):
    v5 = load_config(V5_CONFIG_PATH)
    with_pose_contract = copy.deepcopy(v5)
    with_pose_contract["pose_preservation"] = POSE_PRESERVATION.as_config()
    with pytest.raises(ValueError, match="only allowed in schema v6"):
        validate_config(with_pose_contract)

    wrong_definition = copy.deepcopy(v5)
    wrong_definition["schema_version"] = 6
    with pytest.raises(ValueError, match="pose-preserving"):
        resolve_experiment(wrong_definition)

    wrong_version = copy.deepcopy(v6_config)
    wrong_version["schema_version"] = 5
    with pytest.raises(ValueError, match="schema_version 5"):
        resolve_experiment(wrong_version)


@pytest.mark.parametrize(
    "filename",
    [
        "left_three_finger_cube.json",
        "left_opposed_face_palm_down.json",
        "left_opposed_face_palm_down_larger_cube_grasp_then_lift.json",
        "left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift.json",
        "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift.json",
    ],
)
def test_existing_schema_templates_still_validate(filename):
    config = load_config(ROOT / "grasp_configs" / filename)
    assert config["schema_version"] in (1, 2, 3, 4, 5)
