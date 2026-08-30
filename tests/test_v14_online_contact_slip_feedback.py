from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    load_config,
    validate_config,
)
from xhand_grasp.controller import (
    ContactPreservingPlannedLiftController,
    GraspGateEvidence,
    OperationFeedback,
    TargetFaceEvidence,
    build_grasp_controller,
    grasp_gate_order,
)
from xhand_grasp.experiment import ContactFeedbackParameters
from xhand_grasp.scene import build_model
from xhand_grasp.simulation import SimulationSession, _allocate_traces


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "grasp_configs" / (
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


def _slip_config() -> tuple[object, dict]:
    config = load_config(CONFIG)
    legacy = ContactFeedbackParameters.from_config(config["contact_feedback"])
    feedback = ContactFeedbackParameters(
        schema_version=2,
        strategy=legacy.strategy,
        filter_time_constant_s=legacy.filter_time_constant_s,
        kp_rad_per_n=legacy.kp_rad_per_n,
        ki_rad_per_n_s=legacy.ki_rad_per_n_s,
        integral_limit_n_s=legacy.integral_limit_n_s,
        correction_limit_rad=legacy.correction_limit_rad,
        rate_limit_rad_s=legacy.rate_limit_rad_s,
        acceleration_limit_rad_s2=legacy.acceleration_limit_rad_s2,
        force_risk_n=legacy.force_risk_n,
        freeze_on_risk=legacy.freeze_on_risk,
        max_loss_s=legacy.max_loss_s,
        recovery_behavior=legacy.recovery_behavior,
        operation_contact_duty_min=legacy.operation_contact_duty_min,
        tangent_slip_freeze_threshold_m=0.001,
        tangent_slip_abort_threshold_m=0.003,
    )
    config["contact_feedback"] = feedback.as_config()
    model, _ = build_model(config)
    return model, config


def _actual_qpos(model, config: dict) -> np.ndarray:
    result = np.zeros(model.nu, dtype=np.float64)
    for actuator in ACTIVE_ACTUATORS:
        result[model.actuator(actuator).id] = float(
            config["grasp_pose"]["nominal_joint_qpos_rad"][actuator]
        )
    return result


def _gate(config: dict) -> GraspGateEvidence:
    force = np.asarray(
        [
            config["contact_force_targets_n"]["per_finger_n"][finger]
            for finger in ACTIVE_FINGERS
        ],
        dtype=np.float64,
    )
    return GraspGateEvidence(
        components=np.ones(len(grasp_gate_order(14)), dtype=bool),
        target_faces=TargetFaceEvidence(
            target_force_n=force,
            total_distal_force_n=force,
            target_force_purity=np.ones(3),
            target_face_effective=np.ones(3, dtype=bool),
            material_off_target=np.zeros(3, dtype=bool),
            material_active_nondistal=np.zeros(3, dtype=bool),
        ),
        hard_abort=False,
        gate_order=grasp_gate_order(14),
    )


def _feedback(
    model,
    config: dict,
    centroids: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    effective: np.ndarray | None = None,
) -> OperationFeedback:
    force = np.asarray(
        [
            config["contact_force_targets_n"]["per_finger_n"][finger]
            for finger in ACTIVE_FINGERS
        ],
        dtype=np.float64,
    )
    valid_array = (
        np.ones(3, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    )
    effective_array = (
        np.ones(3, dtype=bool)
        if effective is None
        else np.asarray(effective, dtype=bool)
    )
    return OperationFeedback(
        target_force_n=force,
        target_force_purity=np.ones(3),
        target_face_effective=effective_array,
        material_off_target=np.zeros(3, dtype=bool),
        material_active_nondistal=np.zeros(3, dtype=bool),
        tactile_force_n=force,
        contact_centroid_world_m=np.asarray(centroids, dtype=np.float64),
        contact_centroid_valid=valid_array,
        contact_centroid_cube_local_m=np.asarray(centroids, dtype=np.float64),
        cube_position_m=np.asarray([0.071, -0.027, 0.1]),
        cube_quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        cube_velocity=np.zeros(6),
        joint_qpos_rad=_actual_qpos(model, config),
        joint_qvel_rad_s=np.zeros(model.nu),
        forbidden_contact=False,
        max_penetration_m=0.0,
        finite=True,
        joint_limits_respected=True,
        inactive_controls_zero=True,
    )


def _acquire(model, config: dict, centroids: np.ndarray):
    controller = build_grasp_controller(model, config, _phase_steps())
    assert isinstance(controller, ContactPreservingPlannedLiftController)
    position = np.asarray([0.071, -0.027, 0.1])
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    controller.latch_initial_pose(position, quaternion)
    gate = _gate(config)
    feedback = _feedback(model, config, centroids)
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
    return controller, acquisition_step + 1, gate, feedback


def test_feedback_schema_two_is_versioned_without_changing_schema_one() -> None:
    raw = load_config(CONFIG)["contact_feedback"]
    legacy = ContactFeedbackParameters.from_config(raw)
    assert legacy.as_config() == raw
    model, config = _slip_config()
    del model
    parsed = ContactFeedbackParameters.from_config(config["contact_feedback"])
    assert parsed.schema_version == 2
    assert parsed.tangent_slip_freeze_threshold_m == pytest.approx(0.001)
    assert parsed.tangent_slip_abort_threshold_m == pytest.approx(0.003)
    assert parsed.feedback_id != legacy.feedback_id
    validate_config(config)

    tampered = copy.deepcopy(config["contact_feedback"])
    tampered["tangent_slip_freeze_threshold_m"] = 0.004
    with pytest.raises(ValueError, match="thresholds|feedback_id"):
        ContactFeedbackParameters.from_config(tampered)


def test_online_slip_is_causal_and_freezes_then_recovers() -> None:
    model, config = _slip_config()
    baseline = np.asarray(
        [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0], [0.04, 0.01, 0.0]]
    )
    controller, step, gate, good = _acquire(model, config, baseline)
    np.testing.assert_allclose(
        controller.grasp_contact_centroid_baseline_cube_local_m,
        baseline,
        rtol=0.0,
        atol=3e-16,
    )
    assert np.all(controller.grasp_contact_centroid_baseline_valid)

    first = controller.command(step)
    progress = first.manipulation_progress
    slipped_points = baseline.copy()
    # +X target faces have Y/Z tangent axes.  The observation is produced
    # after this command and therefore must not retroactively freeze it.
    slipped_points[1, 1] += 0.0015
    controller.observe(
        step,
        gate,
        good.cube_position_m,
        good.cube_quaternion_wxyz,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=_feedback(model, config, slipped_points),
    )
    assert not np.any(controller.tangent_slip_risk_active)

    frozen = controller.command(step + 1)
    assert frozen.manipulation_progress == pytest.approx(progress)
    np.testing.assert_array_equal(
        controller.tangent_slip_risk_active,
        np.asarray([False, True, False]),
    )
    assert controller.progress_frozen
    controller.observe(
        step + 1,
        gate,
        good.cube_position_m,
        good.cube_quaternion_wxyz,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=good,
    )

    resumed = controller.command(step + 2)
    assert not controller.progress_frozen
    assert not controller.recovery_active
    assert resumed.manipulation_progress > frozen.manipulation_progress


def test_online_slip_masks_missing_baseline_and_lost_contact() -> None:
    model, config = _slip_config()
    controller = build_grasp_controller(model, config, _phase_steps())
    assert isinstance(controller, ContactPreservingPlannedLiftController)
    points = np.full((3, 3), 0.1)

    without_baseline = controller._with_online_tangent_slip(
        _feedback(model, config, points)
    )
    np.testing.assert_array_equal(without_baseline.tangent_slip_valid, False)
    np.testing.assert_array_equal(without_baseline.tangent_slip_from_grasp_m, 0.0)

    controller.grasp_contact_centroid_baseline_valid[:] = True
    controller.grasp_contact_centroid_baseline_cube_local_m[:] = 0.0
    lost = controller._with_online_tangent_slip(
        _feedback(
            model,
            config,
            points,
            valid=np.asarray([True, False, True]),
            effective=np.asarray([True, True, False]),
        )
    )
    np.testing.assert_array_equal(lost.tangent_slip_valid, [True, False, False])
    assert lost.tangent_slip_from_grasp_m[1] == 0.0
    assert lost.tangent_slip_from_grasp_m[2] == 0.0


def test_abort_threshold_and_trace_fields_are_schema_version_gated() -> None:
    model, config = _slip_config()
    baseline = np.asarray(
        [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0], [0.04, 0.01, 0.0]]
    )
    controller, step, gate, good = _acquire(model, config, baseline)
    controller.command(step)
    unsafe_points = baseline.copy()
    unsafe_points[0, 2] += 0.0031
    controller.observe(
        step,
        gate,
        good.cube_position_m,
        good.cube_quaternion_wxyz,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=_feedback(model, config, unsafe_points),
    )
    assert controller.aborted
    np.testing.assert_array_equal(
        controller.tangent_slip_abort_risk, [True, False, False]
    )
    events = controller.event_traces()
    assert "online_grasp_contact_centroid_baseline_cube_local_m" in events

    v1 = _allocate_traces(model, 3, schema_version=14)
    v2 = _allocate_traces(
        model,
        3,
        schema_version=14,
        contact_feedback_schema_version=2,
    )
    for name in (
        "online_contact_tangent_slip_m",
        "online_contact_tangent_slip_valid",
        "contact_tangent_slip_freeze_risk",
        "contact_tangent_slip_abort_risk",
    ):
        assert name not in v1
        assert name in v2


def test_session_persists_recomputable_causal_slip_trace() -> None:
    _, config = _slip_config()
    session = SimulationSession(config)
    try:
        while not session.complete:
            session.advance_one()
        summary = session.finalize()
        checks = summary["checks"]
        assert checks["v14_online_tangent_slip_matches_raw_contacts"]
        assert checks["v14_tangent_slip_freeze_uses_previous_observation"]
        assert checks[
            "v14_tangent_slip_abort_matches_current_safety_observation"
        ]

        source = np.asarray(
            session.traces["operation_feedback_source_step"], dtype=np.int64
        )
        np.testing.assert_array_equal(source, np.arange(source.size) - 1)
        valid = np.asarray(
            session.traces["online_contact_tangent_slip_valid"], dtype=bool
        )
        acquisition = int(session.traces["grasp_acquisition_step"])
        assert not np.any(valid[:acquisition])
        assert np.all(valid[acquisition])
    finally:
        session.close()
