from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    INACTIVE_ACTUATORS,
    load_config,
)
from xhand_grasp.controller import (
    ContactPreservingPlannedLiftController,
    ControlState,
    GraspGateEvidence,
    OperationFeedback,
    TargetFaceEvidence,
    build_grasp_controller,
    grasp_gate_order,
)
from xhand_grasp.experiment import ContactForceTargets
from xhand_grasp.scene import build_model
from xhand_grasp.trajectory import (
    interpolate_quintic_c2,
    quintic_c2_knot_derivatives,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


@pytest.fixture(scope="module")
def model_config():
    config = load_config(CONFIG)
    model, _ = build_model(config)
    return model, config


def _phase_steps() -> dict[str, int]:
    # Keep the real 1 ms manipulation/hold durations while shortening phases
    # that do not participate in these pure-controller checks.
    return {
        "settle": 1,
        "close": 1,
        "verify": 750,
        "manipulate": 3000,
        "hold": 1000,
    }


def _gate(config: dict, forces: np.ndarray | None = None) -> GraspGateEvidence:
    target_force = (
        np.asarray(forces, dtype=np.float64)
        if forces is not None
        else np.asarray(
            [
                config["contact_force_targets_n"]["per_finger_n"][finger]
                for finger in ACTIVE_FINGERS
            ],
            dtype=np.float64,
        )
    )
    return GraspGateEvidence(
        components=np.ones(len(grasp_gate_order(14)), dtype=bool),
        target_faces=TargetFaceEvidence(
            target_force_n=target_force,
            total_distal_force_n=target_force,
            target_force_purity=np.ones(3),
            target_face_effective=np.ones(3, dtype=bool),
            material_off_target=np.zeros(3, dtype=bool),
            material_active_nondistal=np.zeros(3, dtype=bool),
        ),
        hard_abort=False,
        gate_order=grasp_gate_order(14),
    )


def _actual_qpos(model, config: dict) -> np.ndarray:
    result = np.zeros(model.nu, dtype=np.float64)
    for actuator in ACTIVE_ACTUATORS:
        result[model.actuator(actuator).id] = float(
            config["grasp_pose"]["nominal_joint_qpos_rad"][actuator]
        )
    return result


def _feedback(
    model,
    config: dict,
    *,
    forces: np.ndarray | None = None,
    purity: np.ndarray | None = None,
    effective: np.ndarray | None = None,
    off_target: np.ndarray | None = None,
    nondistal: np.ndarray | None = None,
    forbidden: bool = False,
) -> OperationFeedback:
    values = (
        np.asarray(forces, dtype=np.float64)
        if forces is not None
        else np.asarray(
            [
                config["contact_force_targets_n"]["per_finger_n"][finger]
                for finger in ACTIVE_FINGERS
            ],
            dtype=np.float64,
        )
    )
    return OperationFeedback(
        target_force_n=values,
        target_force_purity=(
            np.ones(3) if purity is None else np.asarray(purity, dtype=np.float64)
        ),
        target_face_effective=(
            np.ones(3, dtype=bool)
            if effective is None
            else np.asarray(effective, dtype=bool)
        ),
        material_off_target=(
            np.zeros(3, dtype=bool)
            if off_target is None
            else np.asarray(off_target, dtype=bool)
        ),
        material_active_nondistal=(
            np.zeros(3, dtype=bool)
            if nondistal is None
            else np.asarray(nondistal, dtype=bool)
        ),
        tactile_force_n=np.maximum(values, 0.1),
        contact_centroid_world_m=np.zeros((3, 3)),
        contact_centroid_valid=np.ones(3, dtype=bool),
        cube_position_m=np.asarray([0.071, -0.027, 0.1]),
        cube_quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        cube_velocity=np.zeros(6),
        joint_qpos_rad=_actual_qpos(model, config),
        joint_qvel_rad_s=np.zeros(model.nu),
        forbidden_contact=forbidden,
        max_penetration_m=0.0,
        finite=True,
        joint_limits_respected=True,
        inactive_controls_zero=True,
    )


def _acquired_controller(model, config: dict):
    controller = build_grasp_controller(model, config, _phase_steps())
    assert isinstance(controller, ContactPreservingPlannedLiftController)
    position = np.asarray([0.071, -0.027, 0.1])
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    controller.latch_initial_pose(position, quaternion)
    gate = _gate(config)
    feedback = _feedback(model, config)
    acquisition_step = controller.close_end + controller.required_stable_steps - 1
    for step in range(acquisition_step + 1):
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
    assert controller.manipulation_start_step == acquisition_step + 1
    return controller, acquisition_step + 1, gate, feedback


def _scaled_force_config(config: dict, scale: float) -> dict:
    result = copy.deepcopy(config)
    targets = ContactForceTargets.from_config(result["contact_force_targets_n"])
    result["contact_force_targets_n"] = ContactForceTargets(
        schema_version=targets.schema_version,
        source=targets.source,
        minimum_n=targets.minimum_n,
        maximum_n=targets.maximum_n,
        per_finger_n=targets.per_finger_n,
        operation_scale=scale,
    ).as_config()
    return result


def test_v14_legacy_force_target_path_retains_exact_controller_trace(
    model_config,
) -> None:
    model, config = model_config
    assert "operation_scale" not in config["contact_force_targets_n"]
    controller, start, gate, feedback = _acquired_controller(model, config)
    digest = hashlib.sha256()
    position = np.asarray([0.071, -0.027, 0.1])
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    for step in range(start, start + 128):
        command = controller.command(step)
        for value in (
            command.target,
            command.target_velocity_rad_s,
            controller.nominal_plan_target_rad,
            controller.feedback_correction_rad,
            controller.force_targets_n,
            controller.force_error_n,
            controller.force_integral_n_s,
        ):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        digest.update(command.state.value.encode())
        digest.update(np.float64(command.manipulation_progress).tobytes())
        controller.observe(
            step,
            gate,
            position,
            quaternion,
            actual_joint_qpos_rad=_actual_qpos(model, config),
            operation_feedback=feedback,
        )
    for name, value in sorted(controller.event_traces().items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    assert digest.hexdigest() == (
        "f38d930f8050e75d5a92d97c90e6584ca13be2b0c0234ea7640806f321b893ef"
    )


def test_v14_operation_force_scale_latches_then_preserves_causal_independent_pi(
    model_config,
) -> None:
    model, legacy = model_config
    config = _scaled_force_config(legacy, 0.70)
    controller, step, gate, _ = _acquired_controller(model, config)
    configured = np.asarray(
        [
            legacy["contact_force_targets_n"]["per_finger_n"][finger]
            for finger in ACTIVE_FINGERS
        ]
    )
    expected = np.clip(0.70 * configured, 0.2, 3.0)
    np.testing.assert_array_equal(controller.force_targets_n, expected)

    balanced = _feedback(model, config, forces=expected)
    controller._last_feedback = balanced
    controller.filtered_force_n[:] = expected
    controller.command(step)
    np.testing.assert_array_equal(controller.feedback_correction_rad, 0.0)
    before_correction = controller.feedback_correction_rad.copy()

    weak = expected.copy()
    weak[1] = 0.0
    controller.observe(
        step,
        gate,
        np.asarray([0.071, -0.027, 0.1]),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=_feedback(model, config, forces=weak),
    )
    # The current command preceded this observation; only the next sample may
    # react, preserving the controller's one-frame causal contract.
    np.testing.assert_array_equal(
        controller.feedback_correction_rad, before_correction
    )
    controller.command(step + 1)
    assert controller.feedback_scalar_rad[1] > 0.0
    np.testing.assert_array_equal(controller.feedback_scalar_rad[[0, 2]], 0.0)
    assert controller.force_integral_n_s[1] >= 0.0


def test_v14_factory_is_schema_gated_and_inactive_controls_remain_zero(
    model_config,
) -> None:
    model, config = model_config
    controller, step, _, _ = _acquired_controller(model, config)
    command = controller.command(step)
    assert command.state is ControlState.MANIPULATE
    inactive = [model.actuator(name).id for name in INACTIVE_ACTUATORS]
    np.testing.assert_array_equal(command.target[inactive], 0.0)
    for finger_index, finger_names in enumerate(
        (ACTIVE_ACTUATORS[:3], ACTIVE_ACTUATORS[3:6], ACTIVE_ACTUATORS[6:])
    ):
        owned = {model.actuator(name).id for name in finger_names}
        nonzero = set(np.flatnonzero(controller.inward_direction[finger_index]))
        assert nonzero <= owned


def test_v14_constructor_audits_real_ctrl_joint_and_continuous_plan_bounds(
    model_config,
) -> None:
    model, config = model_config
    controller = build_grasp_controller(model, config, _phase_steps())
    active_ids = np.asarray(
        [model.actuator(name).id for name in ACTIVE_ACTUATORS], dtype=np.int64
    )
    joint_ids = np.asarray(
        [model.actuator_trnid[actuator_id, 0] for actuator_id in active_ids],
        dtype=np.int64,
    )
    assert np.all(
        controller.nominal_plan_command_lower_rad[active_ids]
        >= model.actuator_ctrlrange[active_ids, 0] - 1e-12
    )
    assert np.all(
        controller.nominal_plan_command_upper_rad[active_ids]
        <= model.actuator_ctrlrange[active_ids, 1] + 1e-12
    )
    assert np.all(
        controller.nominal_plan_command_lower_rad[active_ids]
        >= model.jnt_range[joint_ids, 0] - 1e-12
    )
    assert np.all(
        controller.nominal_plan_command_upper_rad[active_ids]
        <= model.jnt_range[joint_ids, 1] + 1e-12
    )

    bad_precontact = copy.deepcopy(config)
    bad_precontact["control"]["precontact_targets_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] = -0.01
    with pytest.raises(ValueError, match="outside ctrlrange|ctrl range"):
        build_grasp_controller(model, bad_precontact, _phase_steps())

    bad_nominal = copy.deepcopy(config)
    bad_nominal["grasp_pose"]["nominal_joint_qpos_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] = 2.0
    with pytest.raises(ValueError, match="nominal grasp qpos.*joint range"):
        build_grasp_controller(model, bad_nominal, _phase_steps())

    bad_plan = copy.deepcopy(config)
    bad_plan["manipulation_plan"]["actuator_waypoints_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ][-1] = 1.0
    with pytest.raises(ValueError, match="C2 manipulation plan.*ctrl range"):
        build_grasp_controller(model, bad_plan, _phase_steps())


def test_v14_uses_previous_sample_and_true_freeze_resume_without_catchup(
    model_config,
) -> None:
    model, config = model_config
    controller, step, gate, good = _acquired_controller(model, config)
    position = good.cube_position_m
    quaternion = good.cube_quaternion_wxyz

    first = controller.command(step)
    first_progress = first.manipulation_progress
    weak = _feedback(
        model,
        config,
        forces=np.asarray([controller.force_targets_n[0], 0.0, controller.force_targets_n[2]]),
        effective=np.asarray([True, False, True]),
    )
    # This observation occurs after ``first`` was already emitted.  It may
    # therefore affect only the next command.
    controller.observe(
        step,
        gate,
        position,
        quaternion,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=weak,
    )

    frozen = controller.command(step + 1)
    assert frozen.manipulation_progress == pytest.approx(first_progress)
    assert controller.progress_frozen
    assert controller.recovery_active
    index_ids = [model.actuator(name).id for name in ACTIVE_ACTUATORS[3:6]]
    other_ids = [
        model.actuator(name).id
        for name in (*ACTIVE_ACTUATORS[:3], *ACTIVE_ACTUATORS[6:])
    ]
    assert np.linalg.norm(controller.feedback_correction_rad[index_ids]) > 0.0
    np.testing.assert_array_equal(
        controller.feedback_correction_rad[other_ids], 0.0
    )
    assert np.all(
        controller.feedback_correction_rad[index_ids]
        * controller.inward_direction[1, index_ids]
        >= -1e-15
    )
    assert np.max(np.abs(controller.feedback_scalar_velocity_rad_s)) <= (
        controller.correction_rate_limit_rad_s + 1e-12
    )
    assert np.max(np.abs(controller.feedback_scalar_velocity_rad_s)) <= (
        controller.correction_acceleration_limit_rad_s2
        * controller.timestep
        + 1e-12
    )

    controller.observe(
        step + 1,
        gate,
        position,
        quaternion,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=good,
    )
    recovered = controller.command(step + 2)
    assert not controller.progress_frozen
    assert not controller.recovery_active
    assert recovered.manipulation_progress == pytest.approx(
        first_progress + controller.timestep / controller.plan_duration_s
    )


def test_v14_purity_risk_requests_inward_recovery_even_with_high_force(
    model_config,
) -> None:
    model, config = model_config
    controller, step, gate, good = _acquired_controller(model, config)
    controller.command(step)
    high = controller.force_targets_n + 1.0
    bad_purity = _feedback(
        model,
        config,
        forces=high,
        purity=np.asarray([1.0, 0.90, 1.0]),
        effective=np.asarray([True, False, True]),
    )
    controller.observe(
        step,
        gate,
        good.cube_position_m,
        good.cube_quaternion_wxyz,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=bad_purity,
    )
    controller.command(step + 1)
    assert controller.progress_frozen
    assert controller.force_error_n[1] >= controller.force_risk_n
    assert controller.feedback_scalar_rad[1] > 0.0


def test_v14_anti_windup_unwinds_high_force_and_honors_ctrl_headroom(
    model_config,
) -> None:
    model, config = model_config
    controller, _, _, _ = _acquired_controller(model, config)
    controller.nominal_plan_target_rad[:] = controller.grasp_target
    controller.force_integral_n_s[:] = 0.2
    controller.filtered_force_n[:] = controller.force_targets_n + 10.0
    controller._last_feedback = _feedback(
        model, config, forces=controller.force_targets_n + 10.0
    )
    before = controller.force_integral_n_s.copy()
    controller._update_feedback_correction()
    assert np.all(controller.force_integral_n_s < before)
    assert np.all(controller.feedback_scalar_rad < 0.0)

    # Put one index actuator exactly on the bound in its inward direction.
    index = 1
    direction = controller.inward_direction[index]
    actuator_id = int(np.flatnonzero(np.abs(direction) > 1e-14)[0])
    controller.feedback_scalar_rad[:] = 0.0
    controller.feedback_scalar_velocity_rad_s[:] = 0.0
    controller.force_integral_n_s[:] = 0.0
    controller.nominal_plan_target_rad[:] = controller.grasp_target
    controller.nominal_plan_target_rad[actuator_id] = (
        controller.ctrl_upper[actuator_id]
        if direction[actuator_id] > 0.0
        else controller.ctrl_lower[actuator_id]
    )
    controller.filtered_force_n[:] = controller.force_targets_n
    weak = controller.force_targets_n.copy()
    weak[index] = 0.0
    controller._last_feedback = _feedback(
        model,
        config,
        forces=weak,
        effective=np.asarray([True, False, True]),
    )
    correction = controller._update_feedback_correction()
    assert controller._feedback_scalar_upper_bounds()[index] == pytest.approx(0.0)
    assert controller.force_integral_n_s[index] == pytest.approx(0.0)
    index_ids = [model.actuator(name).id for name in ACTIVE_ACTUATORS[3:6]]
    np.testing.assert_array_equal(correction[index_ids], 0.0)


def test_v14_feedback_position_trace_obeys_rate_acceleration_and_bounds(
    model_config,
) -> None:
    model, config = model_config
    controller, _, _, _ = _acquired_controller(model, config)
    controller.nominal_plan_target_rad[:] = controller.grasp_target
    low = np.zeros(3)
    controller._last_feedback = _feedback(
        model, config, forces=low, effective=np.zeros(3, dtype=bool)
    )
    samples = [controller.feedback_scalar_rad.copy()]
    for _ in range(100):
        controller.filtered_force_n[:] = low
        controller._update_feedback_correction()
        samples.append(controller.feedback_scalar_rad.copy())

    high = controller.force_targets_n + 10.0
    controller._last_feedback = _feedback(model, config, forces=high)
    for _ in range(500):
        controller.filtered_force_n[:] = high
        controller._update_feedback_correction()
        samples.append(controller.feedback_scalar_rad.copy())
    position = np.asarray(samples)
    velocity = np.diff(position, axis=0) / controller.timestep
    acceleration = np.diff(velocity, axis=0) / controller.timestep
    assert np.max(np.abs(velocity)) <= (
        controller.correction_rate_limit_rad_s + 1e-10
    )
    assert np.max(np.abs(acceleration)) <= (
        controller.correction_acceleration_limit_rad_s2 + 1e-8
    )
    assert np.min(position) < -1e-5
    assert np.min(position) >= -controller.correction_limit_rad - 1e-14
    assert np.max(position) <= controller.correction_limit_rad + 1e-14


def test_v14_bidirectional_pi_unloads_high_force_but_risk_recovers_inward(
    model_config,
) -> None:
    model, config = model_config
    controller, _, _, _ = _acquired_controller(model, config)
    controller.nominal_plan_target_rad[:] = controller.grasp_target

    high = controller.force_targets_n + 10.0
    controller._last_feedback = _feedback(model, config, forces=high)
    for _ in range(300):
        controller.filtered_force_n[:] = high
        controller._update_feedback_correction()
    unloaded = controller.feedback_scalar_rad.copy()
    assert np.all(unloaded < 0.0)

    lost = np.zeros(3)
    controller._last_feedback = _feedback(
        model,
        config,
        forces=lost,
        purity=np.zeros(3),
        effective=np.zeros(3, dtype=bool),
    )
    controller.filtered_force_n[:] = lost
    controller._update_feedback_correction()
    assert np.all(controller.force_error_n >= controller.force_risk_n)
    assert np.all(controller.force_integral_n_s >= 0.0)
    assert np.all(controller.feedback_scalar_velocity_rad_s > 0.0)
    assert np.all(controller.feedback_scalar_rad > unloaded)


def test_v14_bidirectional_feedback_honors_outward_ctrl_headroom(
    model_config,
) -> None:
    model, config = model_config
    controller, _, _, _ = _acquired_controller(model, config)
    finger = 0
    direction = controller.inward_direction[finger]
    actuator_id = int(np.flatnonzero(np.abs(direction) > 1e-14)[0])
    coefficient = float(direction[actuator_id])
    controller.nominal_plan_target_rad[:] = controller.grasp_target
    controller.nominal_plan_target_rad[actuator_id] = (
        controller.ctrl_lower[actuator_id]
        if coefficient > 0.0
        else controller.ctrl_upper[actuator_id]
    )
    scalar_lower, _ = controller._feedback_scalar_bounds()
    assert scalar_lower[finger] == pytest.approx(0.0)

    controller.filtered_force_n[:] = controller.force_targets_n + 10.0
    controller._last_feedback = _feedback(
        model, config, forces=controller.filtered_force_n
    )
    correction = controller._update_feedback_correction()
    owned = [model.actuator(name).id for name in ACTIVE_ACTUATORS[:3]]
    np.testing.assert_array_equal(correction[owned], 0.0)
    assert controller.force_integral_n_s[finger] == pytest.approx(0.0)


def test_v14_quintic_plan_is_c2_at_knots_and_stops_only_at_endpoints() -> None:
    times = np.asarray([0.0, 0.15, 0.30, 0.45])
    values = np.asarray(
        [
            [0.0, 0.0],
            [0.00055, 0.02],
            [0.00110, 0.03],
            [0.00165, 0.04],
        ]
    )
    velocities, accelerations = quintic_c2_knot_derivatives(times, values)
    np.testing.assert_array_equal(velocities[[0, -1]], 0.0)
    np.testing.assert_array_equal(accelerations[[0, -1]], 0.0)
    assert abs(velocities[1, 0]) > 0.0
    assert abs(velocities[2, 0]) > 0.0

    for knot in range(1, len(times) - 1):
        left = interpolate_quintic_c2(
            times,
            values,
            np.nextafter(times[knot], -np.inf),
            knot_velocities=velocities,
            knot_accelerations=accelerations,
        )
        right = interpolate_quintic_c2(
            times,
            values,
            times[knot],
            knot_velocities=velocities,
            knot_accelerations=accelerations,
        )
        np.testing.assert_allclose(left[0], right[0], atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(left[1], right[1], atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(left[2], right[2], atol=1e-10, rtol=0.0)

    start = interpolate_quintic_c2(
        times,
        values,
        times[0],
        knot_velocities=velocities,
        knot_accelerations=accelerations,
    )
    finish = interpolate_quintic_c2(
        times,
        values,
        times[-1],
        knot_velocities=velocities,
        knot_accelerations=accelerations,
    )
    np.testing.assert_array_equal(start[1], 0.0)
    np.testing.assert_array_equal(start[2], 0.0)
    np.testing.assert_allclose(finish[1], 0.0, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(finish[2], 0.0, atol=1e-13, rtol=0.0)


def test_v14_quintic_plan_matches_jerk_at_interior_knots() -> None:
    times = np.linspace(0.0, 3.0, 21)
    unit = times / times[-1]
    values = (3.0 * unit**2 - 2.0 * unit**3)[:, None] * np.asarray(
        [[0.535, -0.1495, 0.46]]
    )
    velocities, accelerations = quintic_c2_knot_derivatives(times, values)
    epsilon = 1e-5
    for knot in times[1:-1]:
        left_acceleration = interpolate_quintic_c2(
            times,
            values,
            knot - epsilon,
            knot_velocities=velocities,
            knot_accelerations=accelerations,
        )[2]
        centre_acceleration = interpolate_quintic_c2(
            times,
            values,
            knot,
            knot_velocities=velocities,
            knot_accelerations=accelerations,
        )[2]
        right_acceleration = interpolate_quintic_c2(
            times,
            values,
            knot + epsilon,
            knot_velocities=velocities,
            knot_accelerations=accelerations,
        )[2]
        left_jerk = (centre_acceleration - left_acceleration) / epsilon
        right_jerk = (right_acceleration - centre_acceleration) / epsilon
        np.testing.assert_allclose(left_jerk, right_jerk, atol=2e-3, rtol=2e-3)


def test_v14_extends_manipulate_into_saved_verify_slack_then_enters_hold(
    model_config,
) -> None:
    model, config = model_config
    controller = build_grasp_controller(model, config, _phase_steps())
    assert isinstance(controller, ContactPreservingPlannedLiftController)
    controller._acquired = True
    controller.manipulation_start_step = 0
    controller._next_step = controller.phase_steps["manipulate"]
    controller._planned_completed_steps = int(
        round(controller.plan_duration_s / controller.timestep)
    ) - 2
    controller._planned_elapsed_s = (
        controller.plan_duration_s - 2.0 * controller.timestep
    )
    controller._last_feedback = _feedback(model, config)
    controller.filtered_force_n[:] = controller.force_targets_n
    controller._force_target_latched = True
    controller._last_sent_target[:] = controller.grasp_target
    gate = _gate(config)
    position = np.asarray([0.071, -0.027, 0.1])
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])

    step = controller._next_step
    penultimate = controller.command(step)
    assert penultimate.state is ControlState.MANIPULATE
    assert penultimate.manipulation_progress < 1.0
    controller.observe(
        step, gate, position, quaternion, operation_feedback=_feedback(model, config)
    )

    final = controller.command(step + 1)
    assert final.state is ControlState.MANIPULATE
    assert final.manipulation_progress == pytest.approx(1.0)
    controller.observe(
        step + 1,
        gate,
        position,
        quaternion,
        operation_feedback=_feedback(model, config),
    )
    assert controller.manipulation_end_step == step + 1

    hold = controller.command(step + 2)
    assert hold.state is ControlState.HOLD
    assert hold.manipulation_progress == pytest.approx(1.0)


def test_v14_loss_allowance_and_immediate_safety_abort(model_config) -> None:
    model, config = model_config
    controller, step, gate, good = _acquired_controller(model, config)
    lost = _feedback(
        model,
        config,
        forces=np.zeros(3),
        effective=np.zeros(3, dtype=bool),
    )
    for offset in range(controller.maximum_loss_steps):
        controller.command(step + offset)
        controller.observe(
            step + offset,
            gate,
            good.cube_position_m,
            good.cube_quaternion_wxyz,
            actual_joint_qpos_rad=_actual_qpos(model, config),
            operation_feedback=lost,
        )
        assert not controller.aborted
    controller.command(step + controller.maximum_loss_steps)
    controller.observe(
        step + controller.maximum_loss_steps,
        gate,
        good.cube_position_m,
        good.cube_quaternion_wxyz,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=lost,
    )
    assert controller.aborted
    aborted = controller.command(step + controller.maximum_loss_steps + 1)
    assert aborted.state is ControlState.ABORT
    inactive = [model.actuator(name).id for name in INACTIVE_ACTUATORS]
    np.testing.assert_array_equal(aborted.target[inactive], 0.0)

    unsafe, unsafe_step, gate, good = _acquired_controller(model, config)
    unsafe.command(unsafe_step)
    off_target = np.asarray([False, True, False])
    unsafe.observe(
        unsafe_step,
        gate,
        good.cube_position_m,
        good.cube_quaternion_wxyz,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=_feedback(
            model, config, off_target=off_target
        ),
    )
    assert unsafe.aborted


def test_v14_aborts_before_consuming_the_mandatory_hold(model_config) -> None:
    model, config = model_config
    controller = build_grasp_controller(model, config, _phase_steps())
    assert isinstance(controller, ContactPreservingPlannedLiftController)
    controller._acquired = True
    controller.manipulation_start_step = 0
    controller._next_step = controller.plan_completion_deadline_step
    controller._planned_completed_steps = 100
    controller._planned_elapsed_s = 100 * controller.timestep
    controller._last_feedback = _feedback(model, config)
    controller._last_sent_target[:] = controller.grasp_target

    command = controller.command(controller.plan_completion_deadline_step)
    assert controller.aborted
    assert command.state is ControlState.ABORT
    assert controller.termination_step == controller.plan_completion_deadline_step
    np.testing.assert_array_equal(command.target, controller.grasp_target)


def test_v14_holds_last_valid_command_if_plan_removes_feedback_headroom(
    model_config,
) -> None:
    model, config = model_config
    controller = build_grasp_controller(model, config, _phase_steps())
    assert isinstance(controller, ContactPreservingPlannedLiftController)
    controller._acquired = True
    controller.manipulation_start_step = 0
    controller._next_step = controller.close_end
    controller._planned_completed_steps = 1
    controller._planned_elapsed_s = controller.timestep
    controller._last_feedback = _feedback(model, config)
    controller.filtered_force_n[:] = controller.force_targets_n
    controller._force_target_latched = True
    previous = controller.grasp_target.copy()
    controller._last_sent_target[:] = previous

    finger = 1
    direction = controller.inward_direction[finger]
    actuator_id = int(np.flatnonzero(np.abs(direction) > 1e-14)[0])
    controller.feedback_scalar_rad[finger] = 0.02
    controller.feedback_correction_rad[:] = (
        controller.feedback_scalar_rad[finger] * direction
    )
    # Make the feed-forward interpolation put the chosen actuator exactly at
    # its bound, removing all headroom while a non-zero bounded correction is
    # still active.
    offset = (
        controller.ctrl_upper[actuator_id]
        if direction[actuator_id] > 0.0
        else controller.ctrl_lower[actuator_id]
    ) - controller.grasp_target[actuator_id]
    controller.plan_waypoints[:, actuator_id] = offset
    controller.plan_waypoint_velocities_rad_s[:, actuator_id] = 0.0
    controller.plan_waypoint_accelerations_rad_s2[:, actuator_id] = 0.0

    command = controller.command(controller.close_end)
    assert command.state is ControlState.ABORT
    assert controller.feedback_headroom_abort_step == controller.close_end
    np.testing.assert_array_equal(command.target, previous)
    np.testing.assert_array_equal(command.target_velocity_rad_s, 0.0)
