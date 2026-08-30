from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.controller import (
    ControlState,
    OperationFeedback,
    TargetFaceEvidence,
)
from xhand_grasp.rolling_contact_slip import RollingAwareSlipEstimate
from xhand_grasp.scene import build_model
from xhand_grasp.simulation import (
    _allocate_traces,
    _rolling_contact_slip_summary,
    _v16_target_face_effective,
)


ROOT = Path(__file__).resolve().parents[1]
V15_CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift.json"
)


def _feedback(**updates: object) -> OperationFeedback:
    values: dict[str, object] = {
        "target_force_n": np.ones(3),
        "target_force_purity": np.ones(3),
        "target_face_effective": np.ones(3, dtype=bool),
        "material_off_target": np.zeros(3, dtype=bool),
        "material_active_nondistal": np.zeros(3, dtype=bool),
        "tactile_force_n": np.ones(3),
        "contact_centroid_world_m": np.zeros((3, 3)),
        "contact_centroid_valid": np.ones(3, dtype=bool),
        "cube_position_m": np.zeros(3),
        "cube_quaternion_wxyz": np.asarray((1.0, 0.0, 0.0, 0.0)),
        "cube_velocity": np.zeros(6),
        "joint_qpos_rad": np.zeros(12),
        "joint_qvel_rad_s": np.zeros(12),
        "forbidden_contact": False,
        "max_penetration_m": 0.0,
        "finite": True,
        "joint_limits_respected": True,
        "inactive_controls_zero": True,
    }
    values.update(updates)
    return OperationFeedback(**values)


def test_v16_operation_feedback_validates_signed_slip_and_live_jacobian() -> None:
    feedback = _feedback(
        rolling_signed_tangent_displacement_m=np.full((3, 2), -0.001),
        rolling_cumulative_irrecoverable_slip_m=np.full(3, 0.001),
        rolling_relative_tangent_velocity_m_s=np.full((3, 2), -0.002),
        rolling_contact_valid=np.ones(3, dtype=bool),
        rolling_contact_continuous=np.ones(3, dtype=bool),
        rolling_tangent_jacobian_m_per_rad=np.ones(
            (3, 2, len(ACTIVE_ACTUATORS))
        ),
        rolling_tangent_jacobian_valid=np.ones(3, dtype=bool),
    )
    assert feedback.rolling_signed_tangent_displacement_m.shape == (3, 2)
    assert feedback.rolling_tangent_jacobian_m_per_rad.shape == (3, 2, 8)

    with pytest.raises(ValueError, match="rolling cumulative slip"):
        _feedback(rolling_cumulative_irrecoverable_slip_m=-np.ones(3))
    with pytest.raises(ValueError, match="must have shape"):
        _feedback(rolling_tangent_jacobian_m_per_rad=np.zeros((3, 8)))


def test_v16_trace_axis_is_opt_in_and_keeps_v15_field_set_unchanged() -> None:
    model, _ = build_model(load_config(V15_CONFIG))
    v15 = _allocate_traces(model, 4, schema_version=15)
    v16 = _allocate_traces(model, 4, schema_version=16)

    assert not any(name.startswith("rolling_") for name in v15)
    assert set(v15) < set(v16)
    assert v16["rolling_signed_tangent_displacement_m"].shape == (4, 3, 2)
    assert v16["rolling_tangent_jacobian_m_per_rad"].shape == (4, 3, 2, 8)
    assert v16["rolling_patch_switch_count"].dtype == np.int64
    assert v16["native_tactile_target_face_effective"].shape == (4, 3)
    assert v16["rolling_aware_target_face_effective"].shape == (4, 3)
    controller_trace_fields = {
        "rolling_slip_feedback_source_step",
        "rolling_slip_filtered_velocity_m_s",
        "rolling_slip_predicted_displacement_m",
        "rolling_slip_predicted_magnitude_m",
        "rolling_slip_controller_cumulative_m",
        "rolling_slip_observation_valid",
        "rolling_slip_recovery_active",
        "rolling_slip_freeze_active",
        "rolling_slip_abort_risk",
        "rolling_slip_exit_run_steps",
        "rolling_slip_request_rad",
        "rolling_slip_correction_rad",
        "rolling_slip_correction_velocity_rad_s",
        "rolling_slip_active_jacobian_m_per_rad",
        "rolling_slip_feedback_saturated",
    }
    assert controller_trace_fields <= set(v16)
    assert controller_trace_fields.isdisjoint(v15)


def _rolling_estimate(valid: tuple[bool, bool, bool]) -> RollingAwareSlipEstimate:
    return RollingAwareSlipEstimate(
        time_s=1.0,
        target_faces=("-X", "+X", "+X"),
        signed_tangent_displacement_m=np.zeros((3, 2)),
        cumulative_irrecoverable_slip_m=np.zeros(3),
        relative_tangent_velocity_m_s=np.zeros((3, 2)),
        normal_force_n=np.where(valid, 0.2, 0.0),
        valid=np.asarray(valid),
        continuous=np.asarray(valid),
        rolling_detected=np.zeros(3, dtype=bool),
        rolling_force_fraction=np.zeros(3),
        patch_switch=np.zeros(3, dtype=bool),
        centroid_jump=np.zeros(3, dtype=bool),
        centroid_tangent_step_m=np.zeros(3),
        matched_patch_count=np.zeros(3, dtype=np.int64),
        new_patch_count=np.zeros(3, dtype=np.int64),
        dropped_patch_count=np.zeros(3, dtype=np.int64),
        patch_switch_count=np.zeros(3, dtype=np.int64),
    )


def test_v16_physical_distal_contact_only_rescues_operation_tactile_dropout() -> None:
    config = {
        "control_protocol": {
            "grasp_gate": {
                "min_target_face_force_n": 0.05,
                "min_target_force_fraction": 0.95,
            }
        }
    }
    native = np.asarray((True, False, False))
    target = TargetFaceEvidence(
        target_force_n=np.full(3, 0.2),
        total_distal_force_n=np.full(3, 0.2),
        target_force_purity=np.ones(3),
        target_face_effective=native,
        material_off_target=np.zeros(3, dtype=bool),
        material_active_nondistal=np.zeros(3, dtype=bool),
    )
    rolling = _rolling_estimate((False, True, True))

    np.testing.assert_array_equal(
        _v16_target_face_effective(config, ControlState.VERIFY, target, rolling),
        native,
    )
    np.testing.assert_array_equal(
        _v16_target_face_effective(
            config, ControlState.MANIPULATE, target, rolling
        ),
        (True, True, True),
    )

    unsafe = TargetFaceEvidence(
        target_force_n=np.full(3, 0.2),
        total_distal_force_n=np.full(3, 0.2),
        target_force_purity=np.asarray((1.0, 0.94, 1.0)),
        target_face_effective=np.ones(3, dtype=bool),
        material_off_target=np.asarray((False, False, True)),
        material_active_nondistal=np.asarray((True, False, False)),
    )
    np.testing.assert_array_equal(
        _v16_target_face_effective(
            config, ControlState.HOLD, unsafe, _rolling_estimate((True,) * 3)
        ),
        (False, False, False),
    )


def test_v16_summary_uses_only_manipulate_and_hold_samples() -> None:
    config = {
        "joint_pair_feedback": {
            "slip_freeze_threshold_m": 0.0015,
            "slip_abort_threshold_m": 0.0020,
        }
    }
    traces = {
        "control_state": np.asarray(
            ("VERIFY", "MANIPULATE", "MANIPULATE", "HOLD")
        ),
        "rolling_contact_target_faces": np.asarray(("-X", "+X", "+X")),
        "rolling_cumulative_irrecoverable_slip_m": np.asarray(
            (
                (0.009, 0.009, 0.009),
                (0.0, 0.0, 0.0),
                (0.001, 0.0002, 0.0003),
                (0.0012, 0.0003, 0.0004),
            )
        ),
        "rolling_signed_tangent_displacement_m": np.zeros((4, 3, 2)),
        "rolling_relative_tangent_velocity_m_s": np.full((4, 3, 2), 0.001),
        "rolling_contact_valid": np.ones((4, 3), dtype=bool),
        "rolling_contact_continuous": np.ones((4, 3), dtype=bool),
        "rolling_detected": np.zeros((4, 3), dtype=bool),
        "rolling_patch_switch": np.zeros((4, 3), dtype=bool),
        "rolling_centroid_jump": np.zeros((4, 3), dtype=bool),
        "rolling_centroid_tangent_step_m": np.zeros((4, 3)),
        "rolling_patch_switch_count": np.zeros((4, 3), dtype=np.int64),
        "rolling_tangent_jacobian_valid": np.ones((4, 3), dtype=bool),
        "rolling_slip_predicted_magnitude_m": np.zeros((4, 3)),
        "rolling_slip_recovery_active": np.zeros((4, 3), dtype=bool),
        "rolling_slip_freeze_active": np.zeros((4, 3), dtype=bool),
        "rolling_slip_abort_risk": np.zeros((4, 3), dtype=bool),
        "rolling_slip_feedback_saturated": np.zeros(4, dtype=bool),
        "native_tactile_target_face_effective": np.ones((4, 3), dtype=bool),
        "rolling_aware_target_face_effective": np.ones((4, 3), dtype=bool),
    }

    summary = _rolling_contact_slip_summary(config, traces)
    assert summary["operation_sample_count"] == 3
    assert summary["maximum_cumulative_irrecoverable_slip_m"] == pytest.approx(
        0.0012
    )
    assert summary["all_fingers_below_freeze_threshold"]
    assert summary["per_finger"]["thumb"]["valid_duty"] == 1.0
