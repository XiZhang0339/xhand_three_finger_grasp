from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.scene import rpy_degrees_to_rotation_matrix
from xhand_grasp.tuning.joint_pair_alignment_refinement import (
    JointPairAlignmentAdjustment,
    build_joint_pair_alignment_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_pivoted_refinement_keeps_cube_and_contact_pivot_fixed() -> None:
    source = _config()
    pivot = np.asarray((0.08, -0.02, 0.14))
    adjustment = JointPairAlignmentAdjustment(
        wrist_local_rotvec_deg=(-2.0, 0.0, 0.0),
        contact_pivot_world_m=tuple(pivot),
        plan_waypoint_scale=1.0,
    )
    first_id, first = build_joint_pair_alignment_config(source, adjustment)
    second_id, second = build_joint_pair_alignment_config(source, adjustment)

    assert first_id == second_id
    assert canonical_sha256(first) == canonical_sha256(second)
    assert first["cube"] == source["cube"]
    source_rotation = rpy_degrees_to_rotation_matrix(source["hand_pose"]["rpy_deg"])
    resolved_rotation = rpy_degrees_to_rotation_matrix(first["hand_pose"]["rpy_deg"])
    source_local = source_rotation.T @ (
        pivot - np.asarray(source["hand_pose"]["translation_m"])
    )
    resolved_world = np.asarray(first["hand_pose"]["translation_m"]) + (
        resolved_rotation @ source_local
    )
    np.testing.assert_allclose(resolved_world, pivot, atol=1e-12)


def test_refinement_scales_plan_and_records_joint_pair_contract() -> None:
    source = _config()
    offsets = [0.0] * len(ACTIVE_ACTUATORS)
    offsets[ACTIVE_ACTUATORS.index("left_hand_thumb_bend_joint_actuator")] = 0.002
    adjustment = JointPairAlignmentAdjustment(
        wrist_local_rotvec_deg=(-3.1, 0.0, 0.0),
        contact_pivot_world_m=(0.083, -0.021, 0.135),
        root_delta_cube_m=(0.0, -0.0001, 0.0),
        fixed_grasp_qpos_offset_rad=tuple(offsets),
        plan_waypoint_scale=1.04,
        source_grasp_window_p95_deg=9.98,
        grasp_window_p95_max_deg=8.1,
        operation_p95_max_deg=8.1,
        minimum_improvement_deg=1.5,
    )
    _, resolved = build_joint_pair_alignment_config(source, adjustment)
    metadata = resolved["candidate_metadata"][
        "v14_index_middle_joint_pair_alignment_refinement"
    ]

    assert metadata["joint_names"] == [
        "left_hand_index_joint1",
        "left_hand_mid_joint1",
    ]
    assert metadata["minimum_improvement_deg"] == pytest.approx(1.5)
    assert resolved["grasp_pose"]["nominal_joint_qpos_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] == pytest.approx(
        source["grasp_pose"]["nominal_joint_qpos_rad"][
            "left_hand_thumb_bend_joint_actuator"
        ]
        + 0.002
    )
    for name, source_values in source["manipulation_plan"][
        "actuator_waypoints_rad"
    ].items():
        assert resolved["manipulation_plan"]["actuator_waypoints_rad"][name] == (
            pytest.approx([1.04 * float(value) for value in source_values])
        )
    assert resolved["control"]["manipulation_delta_rad"] == pytest.approx(
        {
            name: values[-1]
            for name, values in resolved["manipulation_plan"][
                "actuator_waypoints_rad"
            ].items()
        }
    )


def test_refinement_rejects_unsafe_plan_scale() -> None:
    with pytest.raises(ValueError, match="plan_waypoint_scale"):
        JointPairAlignmentAdjustment(
            wrist_local_rotvec_deg=(0.0, 0.0, 0.0),
            contact_pivot_world_m=(0.0, 0.0, 0.0),
            plan_waypoint_scale=1.2,
        )


def test_refinement_supports_independent_contact_control_residuals() -> None:
    source = _config()
    precontact = [0.0] * len(ACTIVE_ACTUATORS)
    preload = [0.0] * len(ACTIVE_ACTUATORS)
    index = ACTIVE_ACTUATORS.index("left_hand_index_joint1_actuator")
    precontact[index] = 0.012
    preload[index] = -0.007
    adjustment = JointPairAlignmentAdjustment(
        wrist_local_rotvec_deg=(0.0, 0.0, 0.0),
        contact_pivot_world_m=(0.083, -0.021, 0.135),
        precontact_qpos_residual_rad=tuple(precontact),
        preload_target_residual_rad=tuple(preload),
        close_profile_start_residual_fraction=(0.0, -0.05, 0.0),
        close_profile_end_residual_fraction=(0.0, -0.02, 0.0),
        close_duration_s=1.75,
    )
    assert adjustment.as_mapping()["schema_version"] == 2

    _, resolved = build_joint_pair_alignment_config(source, adjustment)
    name = "left_hand_index_joint1_actuator"
    assert resolved["grasp_pose"]["nominal_joint_qpos_rad"][name] == pytest.approx(
        source["grasp_pose"]["nominal_joint_qpos_rad"][name]
    )
    assert resolved["control"]["precontact_targets_rad"][name] == pytest.approx(
        source["control"]["precontact_targets_rad"][name] + 0.012
    )
    assert resolved["control"]["contact_preload_targets_rad"][name] == pytest.approx(
        source["control"]["contact_preload_targets_rad"][name] - 0.007
    )
    for actuator in ACTIVE_ACTUATORS[3:6]:
        assert resolved["control"]["close_profile"][actuator][
            "start_fraction"
        ] == pytest.approx(
            source["control"]["close_profile"][actuator]["start_fraction"] - 0.05
        )
        assert resolved["control"]["close_profile"][actuator][
            "end_fraction"
        ] == pytest.approx(
            source["control"]["close_profile"][actuator]["end_fraction"] - 0.02
        )
    assert resolved["control_protocol"]["close_s"] == pytest.approx(1.75)


def test_refinement_defaults_preserve_legacy_adjustment_semantics() -> None:
    source = _config()
    adjustment = JointPairAlignmentAdjustment(
        wrist_local_rotvec_deg=(-1.0, 0.0, 0.0),
        contact_pivot_world_m=(0.083, -0.021, 0.135),
    )
    _, resolved = build_joint_pair_alignment_config(source, adjustment)

    assert resolved["control"]["close_profile"] == source["control"]["close_profile"]
    assert resolved["control_protocol"]["close_s"] == source["control_protocol"][
        "close_s"
    ]
    assert adjustment.as_mapping()["schema_version"] == 1
    assert "precontact_qpos_residual_rad" not in adjustment.as_mapping()


def test_refinement_rejects_invalid_close_timing_after_materialization() -> None:
    source = _config()
    adjustment = JointPairAlignmentAdjustment(
        wrist_local_rotvec_deg=(0.0, 0.0, 0.0),
        contact_pivot_world_m=(0.083, -0.021, 0.135),
        close_profile_start_residual_fraction=(0.0, 0.4, 0.0),
    )
    with pytest.raises(ValueError, match="close_profile"):
        build_joint_pair_alignment_config(source, adjustment)

    unregistered = JointPairAlignmentAdjustment(
        wrist_local_rotvec_deg=(0.0, 0.0, 0.0),
        contact_pivot_world_m=(0.083, -0.021, 0.135),
        close_duration_s=1.6,
    )
    with pytest.raises(ValueError, match="control_protocol"):
        build_joint_pair_alignment_config(source, unregistered)
