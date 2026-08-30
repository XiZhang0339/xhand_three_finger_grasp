from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, load_config
from xhand_grasp.controller import (
    ControlState,
    GraspGateEvidence,
    JointPairAlignedContactPreservingPlannedLiftController,
    OperationFeedback,
    RollingSlipAwareJointPairAlignedContactPreservingPlannedLiftController,
    TargetFaceEvidence,
    build_grasp_controller,
    grasp_gate_order,
    merge_rolling_operation_target_face_effective,
)
from xhand_grasp.experiments.opposed_face_palm_down_joint_pair_near_zero_contact_preserving_planned_lift import (
    JOINT_PAIR_ALIGNMENT,
)
from xhand_grasp.experiments.opposed_face_palm_down_joint_pair_near_zero_rolling_slip_contact_preserving_planned_lift import (
    JOINT_PAIR_FEEDBACK,
)
from xhand_grasp.scene import build_model


ROOT = Path(__file__).resolve().parents[1]
V14_CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _phase_steps() -> dict[str, int]:
    return {
        "settle": 1,
        "close": 1,
        "verify": 750,
        "manipulate": 3000,
        "hold": 1000,
    }


def _model_config():
    source = load_config(V14_CONFIG)
    model, _ = build_model(source)
    config = copy.deepcopy(source)
    config["schema_version"] = 16
    config["control_protocol"]["strategy"] = (
        "grasp_verify_then_joint_pair_aligned_rolling_slip_"
        "contact_preserving_planned_lift"
    )
    config["joint_pair_alignment"] = JOINT_PAIR_ALIGNMENT.as_config()
    config["joint_pair_feedback"] = JOINT_PAIR_FEEDBACK.as_config()
    knot_count = len(config["manipulation_plan"]["knot_times_s"])
    active_count = len(ACTIVE_ACTUATORS)
    pair = np.zeros((knot_count, 2, active_count), dtype=np.float64)
    pair[:, 0, 0] = 1.0
    pair[:, 1, 1] = 1.0
    config["manipulation_plan"]["joint_pair_residual_jacobian_2x8"] = (
        pair.tolist()
    )
    config["manipulation_plan"]["object_response_jacobian_6x8"] = (
        np.zeros((knot_count, 6, active_count)).tolist()
    )
    config["manipulation_plan"]["target_force_jacobian_3x8"] = (
        np.zeros((knot_count, 3, active_count)).tolist()
    )
    return model, config


def _actual_qpos(model, config: dict) -> np.ndarray:
    values = np.zeros(model.nu, dtype=np.float64)
    for actuator in ACTIVE_ACTUATORS:
        values[model.actuator(actuator).id] = float(
            config["grasp_pose"]["nominal_joint_qpos_rad"][actuator]
        )
    return values


def _gate(config: dict) -> GraspGateEvidence:
    force = np.asarray(
        [
            config["contact_force_targets_n"]["per_finger_n"][finger]
            for finger in ACTIVE_FINGERS
        ],
        dtype=np.float64,
    )
    order = grasp_gate_order(16)
    return GraspGateEvidence(
        components=np.ones(len(order), dtype=bool),
        target_faces=TargetFaceEvidence(
            target_force_n=force,
            total_distal_force_n=force,
            target_force_purity=np.ones(3),
            target_face_effective=np.ones(3, dtype=bool),
            material_off_target=np.zeros(3, dtype=bool),
            material_active_nondistal=np.zeros(3, dtype=bool),
        ),
        hard_abort=False,
        gate_order=order,
    )


def _feedback(
    model,
    config: dict,
    *,
    signed: np.ndarray | None = None,
    cumulative: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    rolling_valid: np.ndarray | None = None,
    rolling_continuous: np.ndarray | None = None,
    jacobian: np.ndarray | None = None,
    jacobian_valid: np.ndarray | None = None,
    force: np.ndarray | None = None,
    legacy_slip_m: np.ndarray | None = None,
    rolling_detected: np.ndarray | None = None,
    patch_switch: np.ndarray | None = None,
) -> OperationFeedback:
    nominal_force = np.asarray(
        [
            config["contact_force_targets_n"]["per_finger_n"][finger]
            for finger in ACTIVE_FINGERS
        ],
        dtype=np.float64,
    )
    force_value = nominal_force if force is None else np.asarray(force)
    vector = np.asarray((0.0, 0.02, 0.0), dtype=np.float64)
    legacy = (
        np.zeros(3, dtype=np.float64)
        if legacy_slip_m is None
        else np.asarray(legacy_slip_m, dtype=np.float64)
    )
    return OperationFeedback(
        target_force_n=force_value,
        target_force_purity=np.ones(3),
        target_face_effective=np.ones(3, dtype=bool),
        material_off_target=np.zeros(3, dtype=bool),
        material_active_nondistal=np.zeros(3, dtype=bool),
        tactile_force_n=force_value,
        contact_centroid_world_m=np.zeros((3, 3)),
        contact_centroid_valid=np.ones(3, dtype=bool),
        contact_centroid_cube_local_m=np.zeros((3, 3)),
        tangent_slip_from_grasp_m=legacy,
        tangent_slip_valid=legacy > 0.0,
        cube_position_m=np.asarray((0.071, -0.027, 0.1)),
        cube_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
        cube_velocity=np.zeros(6),
        joint_qpos_rad=_actual_qpos(model, config),
        joint_qvel_rad_s=np.zeros(model.nu),
        forbidden_contact=False,
        max_penetration_m=0.0,
        finite=True,
        joint_limits_respected=True,
        inactive_controls_zero=True,
        joint_pair_vector_cube_m=vector,
        joint_pair_residual=np.zeros(2),
        joint_pair_angle_deg=0.0,
        joint_pair_length_m=0.02,
        joint_pair_positive_y=True,
        joint_pair_valid=True,
        rolling_signed_tangent_displacement_m=(
            np.zeros((3, 2)) if signed is None else signed
        ),
        rolling_cumulative_irrecoverable_slip_m=(
            np.zeros(3) if cumulative is None else cumulative
        ),
        rolling_relative_tangent_velocity_m_s=(
            np.zeros((3, 2)) if velocity is None else velocity
        ),
        rolling_contact_valid=(
            np.zeros(3, dtype=bool)
            if rolling_valid is None
            else rolling_valid
        ),
        rolling_contact_continuous=(
            np.zeros(3, dtype=bool)
            if rolling_continuous is None
            else rolling_continuous
        ),
        rolling_detected=(
            np.zeros(3, dtype=bool)
            if rolling_detected is None
            else rolling_detected
        ),
        rolling_patch_switch=(
            np.zeros(3, dtype=bool)
            if patch_switch is None
            else patch_switch
        ),
        rolling_tangent_jacobian_m_per_rad=(
            np.zeros((3, 2, len(ACTIVE_ACTUATORS)))
            if jacobian is None
            else jacobian
        ),
        rolling_tangent_jacobian_valid=(
            np.zeros(3, dtype=bool)
            if jacobian_valid is None
            else jacobian_valid
        ),
    )


def _controller():
    model, config = _model_config()
    controller = build_grasp_controller(model, config, _phase_steps())
    assert isinstance(
        controller,
        RollingSlipAwareJointPairAlignedContactPreservingPlannedLiftController,
    )
    return model, config, controller


def _acquire():
    model, config, controller = _controller()
    position = np.asarray((0.071, -0.027, 0.1))
    quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
    controller.latch_initial_pose(position, quaternion)
    gate = _gate(config)
    feedback = _feedback(model, config)
    acquisition = controller.close_end + controller.required_stable_steps - 1
    for step in range(acquisition + 1):
        controller.command(step)
        controller.observe(
            step,
            gate,
            position,
            quaternion,
            actual_joint_qpos_rad=_actual_qpos(model, config),
            operation_feedback=feedback,
        )
    assert controller.acquired
    return model, config, controller, acquisition + 1, gate, position, quaternion


def _orthogonal_thumb_tangent(controller) -> np.ndarray:
    inward = controller.inward_direction[0, controller._active_actuator_ids]
    basis = np.eye(len(ACTIVE_ACTUATORS), dtype=np.float64)
    candidates = basis - (
        (basis @ inward)[:, None] / float(inward @ inward)
    ) * inward[None, :]
    norms = np.linalg.norm(candidates, axis=1)
    tangent = candidates[int(np.argmax(norms))]
    return tangent / np.linalg.norm(tangent)


def test_v16_factory_selects_the_rolling_subclass() -> None:
    _, _, controller = _controller()
    assert isinstance(
        controller, JointPairAlignedContactPreservingPlannedLiftController
    )
    assert controller.SCHEMA_VERSION == 16
    assert controller.slip_feedback_enabled is False


def test_strict_rolling_contact_merge_requires_force_purity_and_safe_material() -> None:
    model, config, _ = _controller()
    base = _feedback(model, config)
    value = replace(
        base,
        target_face_effective=np.asarray((False, True, False)),
        tactile_force_n=np.zeros(3),
        rolling_contact_valid=np.ones(3, dtype=bool),
        target_force_n=np.asarray((0.05, 0.049, 0.05)),
        target_force_purity=np.asarray((0.95, 1.0, 0.949)),
    )
    rolling, merged = merge_rolling_operation_target_face_effective(
        value, minimum_force_n=0.05, minimum_purity=0.95
    )
    np.testing.assert_array_equal(rolling, (True, False, False))
    np.testing.assert_array_equal(merged, (True, True, False))

    off_target = replace(
        value,
        target_force_n=np.full(3, 0.05),
        target_force_purity=np.full(3, 0.95),
        material_off_target=np.asarray((True, False, False)),
        material_active_nondistal=np.asarray((False, True, False)),
    )
    rolling, _ = merge_rolling_operation_target_face_effective(
        off_target, minimum_force_n=0.05, minimum_purity=0.95
    )
    np.testing.assert_array_equal(rolling, (False, False, True))
    forbidden, _ = merge_rolling_operation_target_face_effective(
        replace(off_target, forbidden_contact=True),
        minimum_force_n=0.05,
        minimum_purity=0.95,
    )
    assert not np.any(forbidden)


def test_operation_can_use_strict_rolling_contact_without_native_tactile() -> None:
    model, config, controller, step, gate, position, quaternion = _acquire()
    controller.command(step)
    rolling_only = replace(
        _feedback(model, config),
        target_face_effective=np.zeros(3, dtype=bool),
        tactile_force_n=np.zeros(3),
        rolling_contact_valid=np.ones(3, dtype=bool),
        rolling_contact_continuous=np.ones(3, dtype=bool),
        rolling_tangent_jacobian_valid=np.ones(3, dtype=bool),
    )
    controller.observe(
        step,
        gate,
        position,
        quaternion,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=rolling_only,
    )
    assert not np.any(controller.rolling_native_target_face_effective)
    assert np.all(controller.rolling_physical_target_face_effective)
    assert np.all(controller.rolling_merged_target_face_effective)
    assert np.all(controller.rolling_contact_substitution_step_count == 1)
    assert np.all(controller.contact_loss_run_steps == 0)
    assert np.all(controller._last_feedback.target_face_effective)

    command = controller.command(step + 1)
    assert command.state is ControlState.MANIPULATE
    assert not controller.progress_frozen
    assert not controller.aborted


def test_verify_does_not_substitute_rolling_contact_for_native_touch() -> None:
    model, config, controller = _controller()
    position = np.asarray((0.071, -0.027, 0.1))
    quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
    controller.latch_initial_pose(position, quaternion)
    gate = _gate(config)
    safe = _feedback(model, config)
    for step in range(controller.close_end):
        controller.command(step)
        controller.observe(
            step,
            gate,
            position,
            quaternion,
            actual_joint_qpos_rad=_actual_qpos(model, config),
            operation_feedback=safe,
        )
    command = controller.command(controller.close_end)
    assert command.state is ControlState.VERIFY
    rolling_only = replace(
        safe,
        target_face_effective=np.zeros(3, dtype=bool),
        tactile_force_n=np.zeros(3),
        rolling_contact_valid=np.ones(3, dtype=bool),
    )
    controller.observe(
        controller.close_end,
        gate,
        position,
        quaternion,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=rolling_only,
    )
    assert not np.any(controller._last_feedback.target_face_effective)
    assert np.all(controller.rolling_physical_target_face_effective)
    assert not np.any(controller.rolling_merged_target_face_effective)
    assert not np.any(controller.rolling_contact_substitution_step_count)


def test_signed_material_slip_is_causal_and_reverses_the_recovery_request() -> None:
    model, config, controller, step, gate, position, quaternion = _acquire()
    before = controller.command(step)
    np.testing.assert_array_equal(controller.rolling_slip_request_rad, 0.0)
    assert before.state is ControlState.MANIPULATE

    tangent = _orthogonal_thumb_tangent(controller)
    jacobian = np.zeros((3, 2, len(ACTIVE_ACTUATORS)))
    jacobian[0, 0] = 0.05 * tangent
    common = {
        "rolling_valid": np.asarray((True, False, False)),
        "rolling_continuous": np.asarray((True, False, False)),
        "jacobian": jacobian,
        "jacobian_valid": np.asarray((True, False, False)),
    }
    positive = np.zeros((3, 2))
    positive[0, 0] = 0.0005
    controller.observe(
        step,
        gate,
        position,
        quaternion,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=_feedback(model, config, signed=positive, **common),
    )
    controller.command(step + 1)
    positive_request = controller.rolling_slip_request_rad[
        controller._active_actuator_ids
    ].copy()
    assert float(positive_request @ tangent) < 0.0
    assert not controller.progress_frozen

    negative = -positive
    controller.observe(
        step + 1,
        gate,
        position,
        quaternion,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=_feedback(model, config, signed=negative, **common),
    )
    controller.command(step + 2)
    negative_request = controller.rolling_slip_request_rad[
        controller._active_actuator_ids
    ]
    assert float(negative_request @ tangent) > 0.0
    assert np.all(
        controller.rolling_slip_request_rad[
            controller._inactive_actuator_ids
        ]
        == 0.0
    )


def test_recovery_and_progress_freeze_use_independent_hysteresis() -> None:
    model, config, controller = _controller()
    valid = np.asarray((True, False, False))

    def update(magnitude: float) -> None:
        signed = np.zeros((3, 2))
        signed[0, 0] = magnitude
        controller._update_rolling_slip_observation(
            _feedback(
                model,
                config,
                signed=signed,
                rolling_valid=valid,
                rolling_continuous=valid,
                jacobian_valid=valid,
            )
        )

    update(0.0005)
    assert controller.rolling_slip_recovery_active[0]
    assert not controller.rolling_slip_freeze_active[0]
    update(0.0003)
    assert controller.rolling_slip_recovery_active[0]

    for _ in range(controller.rolling_slip_exit_dwell_steps - 1):
        update(0.0002)
        assert controller.rolling_slip_recovery_active[0]
    update(0.0002)
    assert not controller.rolling_slip_recovery_active[0]

    update(0.0016)
    assert controller.rolling_slip_recovery_active[0]
    assert controller.rolling_slip_freeze_active[0]
    update(0.0013)
    assert controller.rolling_slip_freeze_active[0]
    update(0.0011)
    assert not controller.rolling_slip_freeze_active[0]
    assert controller.rolling_slip_recovery_active[0]


def test_pure_rolling_and_patch_switch_do_not_activate_material_recovery() -> None:
    model, config, controller = _controller()
    controller._update_rolling_slip_observation(
        _feedback(
            model,
            config,
            rolling_valid=np.ones(3, dtype=bool),
            rolling_continuous=np.zeros(3, dtype=bool),
            jacobian_valid=np.ones(3, dtype=bool),
            rolling_detected=np.ones(3, dtype=bool),
            patch_switch=np.ones(3, dtype=bool),
        )
    )
    assert not np.any(controller.rolling_slip_recovery_active)
    assert not np.any(controller.rolling_slip_freeze_active)
    assert not np.any(controller.rolling_slip_abort_risk)


def test_legacy_centroid_jump_cannot_freeze_or_recover_v16() -> None:
    model, config, controller, step, gate, position, quaternion = _acquire()
    controller.command(step)
    controller.observe(
        step,
        gate,
        position,
        quaternion,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=_feedback(
            model,
            config,
            legacy_slip_m=np.full(3, 0.0019),
        ),
    )
    command = controller.command(step + 1)
    assert command.state is ControlState.MANIPULATE
    assert not controller.progress_frozen
    assert not np.any(controller.joint_pair_slip_recovery_active)
    assert not np.any(controller.rolling_slip_recovery_active)
    assert not controller.aborted


def test_recovery_suppresses_outward_force_pi_and_preserves_inward_priority() -> None:
    _, _, controller = _controller()
    controller.rolling_slip_recovery_active[:] = (True, False, False)
    controller.filtered_force_n[:] = controller.force_targets_n + 1.0
    controller.force_integral_n_s[:] = -0.1
    correction = controller._update_feedback_correction()
    assert controller.force_error_n[0] >= 0.0
    assert controller.force_integral_n_s[0] >= 0.0
    assert controller.feedback_scalar_rad[0] >= 0.0
    inward = controller.inward_direction[0]
    assert float(correction @ inward) >= -1e-14


def test_cumulative_material_slip_aborts_after_the_observation() -> None:
    model, config, controller, step, gate, position, quaternion = _acquire()
    current = controller.command(step)
    assert current.state is ControlState.MANIPULATE
    cumulative = np.zeros(3)
    cumulative[0] = controller.rolling_slip_abort_threshold_m
    controller.observe(
        step,
        gate,
        position,
        quaternion,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=_feedback(model, config, cumulative=cumulative),
    )
    assert controller.aborted
    assert controller.rolling_slip_abort_reason == "rolling_material_slip"
    assert controller.command(step + 1).state is ControlState.ABORT


def test_v16_rejects_a_v15_feedback_contract() -> None:
    model, config = _model_config()
    config["joint_pair_feedback"]["schema_version"] = 1
    config["joint_pair_feedback"]["strategy"] = (
        "previous_frame_weighted_nullspace"
    )
    with pytest.raises(ValueError, match="feedback schema 2"):
        build_grasp_controller(model, config, _phase_steps())
