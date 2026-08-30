from __future__ import annotations

import copy

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.experiment import ContactPointPlanParameters
from xhand_grasp.grasp_pose import controller_id, grasp_pose_id
from xhand_grasp.tuning.contact_point_manipulability import (
    ContactPointManipulabilityBudget,
    fit_contact_point_manipulability,
)


CONFIG_PATH = (
    "grasp_configs/"
    "left_opposed_face_palm_down_90mm_contact_point_targeted_actual_grasp_pose.json"
)


def _synthetic_probe_set() -> tuple[dict, ...]:
    matrix = np.zeros((6, len(ACTIVE_ACTUATORS)), dtype=np.float64)
    matrix[2, 0] = 0.1
    matrix[0, 1] = 0.02
    records = [
        {
            "probe": {
                "kind": "zero",
                "actuator": None,
                "direction": 0,
                "applied_delta_rad": {name: 0.0 for name in ACTIVE_ACTUATORS},
            },
            "response_6d": np.zeros(6).tolist(),
        }
    ]
    for column, actuator in enumerate(ACTIVE_ACTUATORS):
        for direction in (-1, 1):
            delta = direction * 0.01
            records.append(
                {
                    "probe": {
                        "kind": "single_actuator",
                        "actuator": actuator,
                        "direction": direction,
                        "applied_delta_rad": {
                            name: delta if name == actuator else 0.0
                            for name in ACTIVE_ACTUATORS
                        },
                    },
                    "response_6d": (matrix[:, column] * delta).tolist(),
                }
            )
    return tuple(records)


def test_point_plan_changes_grasp_identity_but_not_controller_identity() -> None:
    config = load_config(CONFIG_PATH)
    original_grasp = grasp_pose_id(config)
    original_controller = controller_id(config)
    plan = ContactPointPlanParameters.from_config(config["contact_point_plan"])
    moved = ContactPointPlanParameters(
        schema_version=plan.schema_version,
        cube_edge_m=plan.cube_edge_m,
        target_faces=plan.target_faces,
        target_face_yz_m={
            **dict(plan.target_face_yz_m),
            "thumb": (
                plan.target_face_yz_m["thumb"][0] + 0.001,
                plan.target_face_yz_m["thumb"][1],
            ),
        },
        target_radius_m=plan.target_radius_m,
    )
    changed = copy.deepcopy(config)
    changed["contact_point_plan"] = moved.as_config()
    validate_config(changed)
    assert grasp_pose_id(changed) != original_grasp
    assert controller_id(changed) == original_controller


def test_v12_controller_identity_excludes_diagnostic_manipulation_delta() -> None:
    config = load_config(CONFIG_PATH)
    original = controller_id(config)
    changed = copy.deepcopy(config)
    actuator = ACTIVE_ACTUATORS[0]
    changed["control"]["manipulation_delta_rad"][actuator] = 0.01
    validate_config(changed)
    assert controller_id(changed) == original

    changed["control"]["contact_preload_targets_rad"][actuator] += 0.01
    validate_config(changed)
    assert controller_id(changed) != original


def test_two_millimeter_manipulability_fit_is_bounded_and_diagnostic() -> None:
    bounds = {name: (-0.2, 0.2) for name in ACTIVE_ACTUATORS}
    result = fit_contact_point_manipulability(_synthetic_probe_set(), bounds)
    assert result["probe_count"] == 17
    assert result["success_evidence"] is False
    assert result["target_response_6d"] == [0.0, 0.0, 0.002, 0.0, 0.0, 0.0]
    assert max(abs(value) for value in result["solution_delta_rad"].values()) <= 0.05
    assert result["predicted_response_6d"][2] == pytest.approx(0.002, abs=2e-6)
    assert 0.0 < result["manipulability_score"] <= 1.0


def test_prescreen_budget_refuses_deltas_above_declared_limit() -> None:
    with pytest.raises(ValueError, match="0.05 rad"):
        ContactPointManipulabilityBudget(maximum_delta_rad=0.051)
