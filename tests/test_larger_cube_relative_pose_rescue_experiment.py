from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.experiment import get_experiment, resolve_experiment
from xhand_grasp.experiments.opposed_face_palm_down_larger_cube_grasp_then_lift import (
    LARGER_CUBE_GRASP_THEN_LIFT,
)
from xhand_grasp.experiments.opposed_face_palm_down_larger_cube_relative_pose_rescue import (
    EXPERIMENT_ID,
    LARGER_CUBE_RELATIVE_POSE_RESCUE,
)
from xhand_grasp.pose_rescue import DEFAULT_PLAN
from xhand_grasp.v2_search import _rpy_matrix


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_relative_pose_rescue.json"
)


def _raw_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_relative_pose_rescue_is_registered_and_reuses_v3_contracts():
    config = load_config(CONFIG_PATH)
    definition = resolve_experiment(config)
    source = LARGER_CUBE_GRASP_THEN_LIFT

    assert config["schema_version"] == 3
    assert definition is LARGER_CUBE_RELATIVE_POSE_RESCUE
    assert get_experiment(EXPERIMENT_ID) is definition
    assert definition.tuning_strategy == "relative_pose_rescue"
    assert definition.artifact_root == (
        "artifacts/left_opposed_face_palm_down_larger_cube_relative_pose_rescue"
    )
    assert definition.evaluation is source.evaluation
    assert definition.control_protocol is source.control_protocol
    assert definition.size_campaign is source.size_campaign
    assert definition.robustness is source.robustness
    assert config["control_protocol"] == source.control_protocol.as_config()
    assert config["size_campaign"] == source.size_campaign.as_config()
    assert config["robustness"] == source.robustness.as_config()


def test_relative_pose_rescue_bounds_are_focused_but_keep_full_target_range():
    bounds = LARGER_CUBE_RELATIVE_POSE_RESCUE.search_bounds
    source_bounds = LARGER_CUBE_GRASP_THEN_LIFT.search_bounds

    assert bounds.palm_pitch_values_deg == (87.0, 88.0, 89.0, 90.0)
    assert bounds.hand_roll_deg == (-3.0, 5.0)
    assert bounds.hand_yaw_deg == (-8.0, 1.0)
    assert dict(bounds.cube_position_in_root_m) == {
        "x": (0.080, 0.086),
        "y": (-0.034, -0.026),
        "z": (0.104, 0.114),
    }
    assert bounds.cube_yaw_deg == (24.0, 34.0)
    expected_targets = dict(source_bounds.actuator_targets_rad)
    expected_targets["left_hand_index_joint2_actuator"] = (0.3, 1.8)
    expected_targets["left_hand_mid_joint2_actuator"] = (0.3, 1.8)
    assert dict(bounds.actuator_targets_rad) == expected_targets
    expected_delta = dict(source_bounds.manipulation_delta_rad)
    expected_delta["left_hand_thumb_rota_joint2_actuator"] = (-0.05, 0.4)
    assert dict(bounds.manipulation_delta_rad) == expected_delta
    assert bounds.actuator_targets_rad[
        "left_hand_index_joint2_actuator"
    ] == pytest.approx((0.3, 1.8))
    assert bounds.actuator_targets_rad[
        "left_hand_mid_joint2_actuator"
    ] == pytest.approx((0.3, 1.8))
    assert bounds.final_target_delta_rad is None


def test_anchor_pose_targets_and_provenance_are_self_consistent():
    config = load_config(CONFIG_PATH)
    bounds = LARGER_CUBE_RELATIVE_POSE_RESCUE.search_bounds
    rescue = config["pose_rescue"]
    anchor = rescue["anchor"]

    assert rescue["schema_version"] == 1
    assert rescue["strategy"] == "relative_pose_rescue"
    assert rescue["source"] == {
        "experiment_id": "left_opposed_face_palm_down_larger_cube_grasp_then_lift",
        "search_seed": 20260821,
        "candidate_id": 309288,
        "candidate_kind": "linked_static_solution",
    }
    assert rescue["focus"]["priority_edges_m"] == [0.061, 0.062, 0.063]
    assert rescue["focus"] == {
        "priority_edges_m": [0.061, 0.062, 0.063],
        "cube_position_in_root_m": {
            "x": [0.08164, 0.08364],
            "y": [-0.03085, -0.02885],
            "z": [0.10736, 0.10936],
        },
        "hand_roll_deg": [-0.5, 2.5],
        "palm_pitch_values_deg": [88.5, 89.0, 89.5, 90.0],
        "hand_yaw_deg": [-5.8, -1.8],
        "cube_yaw_deg": [26.5, 30.5],
        "actuator_bounds_policy": (
            "measured_focus_within_one_shot_expanded_registered_envelope"
        ),
    }
    assert rescue["registered_outer_envelope"] == {
        "cube_position_in_root_m": {
            "x": [0.08, 0.086],
            "y": [-0.034, -0.026],
            "z": [0.104, 0.114],
        },
        "hand_roll_deg": [-3.0, 5.0],
        "palm_pitch_values_deg": [87.0, 88.0, 89.0, 90.0],
        "hand_yaw_deg": [-8.0, 1.0],
        "cube_yaw_deg": [24.0, 34.0],
        "purpose": "configuration validation envelope; not the default focused sampler",
    }
    assert rescue["budget_notes"] == {
        "broad_candidate_budget": 256,
        "local_parent_budget": 8,
        "local_samples_per_parent": 32,
        "dynamic_total_budget": 512,
        "registered_search_budget_is_outer_compatibility_metadata": True,
        "anchor_candidate_is_included_once_in_default_broad_batch": True,
        "relative_pose_is_refined_before_targets": True,
        "acceptance_thresholds_must_not_be_relaxed": True,
        "success_still_requires_constant_density_validation": True,
    }
    assert rescue["focus"]["cube_position_in_root_m"] == {
        name: list(value)
        for name, value in DEFAULT_PLAN.cube_position_in_root_m.items()
    }
    assert rescue["focus"]["palm_pitch_values_deg"] == list(
        DEFAULT_PLAN.hand_pitch_values_deg
    )
    assert rescue["budget_notes"]["broad_candidate_budget"] == (
        DEFAULT_PLAN.broad_candidate_budget
    )
    assert rescue["budget_notes"]["local_samples_per_parent"] == (
        DEFAULT_PLAN.local_samples_per_parent
    )
    assert rescue["budget_notes"]["dynamic_total_budget"] == (
        DEFAULT_PLAN.dynamic_candidate_budget
    )
    assert dict(DEFAULT_PLAN.actuator_limits_rad) == dict(bounds.actuator_targets_rad)

    assert anchor["edge_m"] == pytest.approx(0.062)
    assert anchor["hand_rpy_deg"] == pytest.approx([1.0, 90.0, -3.779])
    assert anchor["cube_position_in_root_m"] == pytest.approx(
        [0.082470, -0.029689, 0.108106]
    )
    assert anchor["cube_yaw_deg"] == pytest.approx(28.549)
    assert anchor["target_faces"] == {
        "thumb": "-X",
        "index": "+X",
        "mid": "+X",
    }
    assert config["contact_topology"]["target_faces"] == anchor["target_faces"]
    assert config["control"]["grasp_targets_rad"] == anchor["grasp_targets_rad"]
    assert set(anchor["grasp_targets_rad"]) == set(ACTIVE_ACTUATORS)
    assert bounds.contains_targets(anchor["grasp_targets_rad"])
    assert bounds.contains_manipulation_delta(
        config["control"]["manipulation_delta_rad"]
    )
    assert set(config["control"]["manipulation_delta_rad"].values()) == {0.0}
    assert bounds.contains_pitch(anchor["hand_rpy_deg"][1])
    assert bounds.contains_cube_position(anchor["cube_position_in_root_m"])

    cube_world = np.asarray(
        [
            *config["cube"]["center_xy_m"],
            config["scene"]["support_top_z_m"]
            + config["cube"]["edge_m"] / 2.0
            + config["cube"]["z_offset_m"],
        ],
        dtype=np.float64,
    )
    rotation = _rpy_matrix(anchor["hand_rpy_deg"])
    expected_translation = cube_world - rotation @ np.asarray(
        anchor["cube_position_in_root_m"], dtype=np.float64
    )
    np.testing.assert_allclose(
        expected_translation,
        [-0.03425669746601186, 0.0115923774280421, 0.19747],
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        config["hand_pose"]["translation_m"], expected_translation, atol=1e-12
    )


def test_tuning_strategy_is_versioned_and_invalid_values_are_rejected():
    assert LARGER_CUBE_GRASP_THEN_LIFT.tuning_strategy == "default"
    with pytest.raises(ValueError, match="tuning_strategy"):
        replace(
            LARGER_CUBE_RELATIVE_POSE_RESCUE,
            tuning_strategy="unknown_rescue_strategy",
        )


def test_relative_pose_rescue_config_rejects_registered_budget_or_target_drift():
    wrong_budget = copy.deepcopy(_raw_config())
    wrong_budget["search"]["budget"]["palm_pitch_values_deg"] = [87.0, 90.0]
    with pytest.raises(ValueError, match="search.budget"):
        validate_config(wrong_budget)

    wrong_target = copy.deepcopy(_raw_config())
    wrong_target["control"]["grasp_targets_rad"][
        "left_hand_index_joint2_actuator"
    ] = 2.0
    with pytest.raises(ValueError, match="grasp_targets_rad"):
        validate_config(wrong_target)
