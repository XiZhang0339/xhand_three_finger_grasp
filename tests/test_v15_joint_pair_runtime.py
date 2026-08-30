from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, load_config
from xhand_grasp.controller import (
    ControlState,
    GraspGateEvidence,
    JointPairAlignedContactPreservingPlannedLiftController,
    OperationFeedback,
    TargetFaceEvidence,
    build_grasp_controller,
    grasp_gate_order,
)
from xhand_grasp.experiments.opposed_face_palm_down_joint_pair_near_zero_contact_preserving_planned_lift import (
    JOINT_PAIR_ALIGNMENT,
    JOINT_PAIR_FEEDBACK,
)
from xhand_grasp.evaluation import _stage_manipulation_label
from xhand_grasp.scene import build_model
from xhand_grasp.simulation import _allocate_traces, run_simulation
from xhand_grasp.viewer import joint_pair_overlay_status


ROOT = Path(__file__).resolve().parents[1]
V14_CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)
V15_CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift.json"
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
    config["schema_version"] = 15
    config["control_protocol"]["strategy"] = (
        "grasp_verify_then_joint_pair_aligned_contact_preserving_planned_lift"
    )
    config["joint_pair_alignment"] = JOINT_PAIR_ALIGNMENT.as_config()
    config["joint_pair_feedback"] = JOINT_PAIR_FEEDBACK.as_config()
    pair = np.zeros((21, 2, len(ACTIVE_ACTUATORS)), dtype=np.float64)
    pair[:, 0, 0] = 1.0
    pair[:, 1, 1] = 1.0
    config["manipulation_plan"]["joint_pair_residual_jacobian_2x8"] = (
        pair.tolist()
    )
    config["manipulation_plan"]["object_response_jacobian_6x8"] = (
        np.zeros((21, 6, len(ACTIVE_ACTUATORS))).tolist()
    )
    config["manipulation_plan"]["target_force_jacobian_3x8"] = (
        np.zeros((21, 3, len(ACTIVE_ACTUATORS))).tolist()
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
    order = grasp_gate_order(15)
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
    residual: tuple[float, float] = (0.0, 0.0),
    centroids: np.ndarray | None = None,
    angle_deg: float | None = None,
    positive_y: bool = True,
    valid: bool = True,
) -> OperationFeedback:
    force = np.asarray(
        [
            config["contact_force_targets_n"]["per_finger_n"][finger]
            for finger in ACTIVE_FINGERS
        ],
        dtype=np.float64,
    )
    residual_array = np.asarray(residual, dtype=np.float64)
    vector = np.asarray(
        (0.02 * residual_array[0], 0.02, 0.02 * residual_array[1]),
        dtype=np.float64,
    )
    if not positive_y:
        vector *= -1.0
    length = float(np.linalg.norm(vector))
    angle = (
        math.degrees(math.acos(float(np.clip(vector[1] / length, -1.0, 1.0))))
        if angle_deg is None
        else float(angle_deg)
    )
    centroid_values = (
        np.zeros((3, 3), dtype=np.float64)
        if centroids is None
        else np.asarray(centroids, dtype=np.float64)
    )
    return OperationFeedback(
        target_force_n=force,
        target_force_purity=np.ones(3),
        target_face_effective=np.ones(3, dtype=bool),
        material_off_target=np.zeros(3, dtype=bool),
        material_active_nondistal=np.zeros(3, dtype=bool),
        tactile_force_n=force,
        contact_centroid_world_m=centroid_values,
        contact_centroid_valid=np.ones(3, dtype=bool),
        contact_centroid_cube_local_m=centroid_values,
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
        joint_pair_residual=residual_array,
        joint_pair_angle_deg=angle,
        joint_pair_length_m=length,
        joint_pair_positive_y=positive_y,
        joint_pair_valid=valid,
    )


def _acquire():
    model, config = _model_config()
    controller = build_grasp_controller(model, config, _phase_steps())
    assert isinstance(
        controller, JointPairAlignedContactPreservingPlannedLiftController
    )
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
    assert controller.joint_pair_grasp_p95_deg == pytest.approx(0.0)
    return model, config, controller, acquisition + 1, gate, position, quaternion


def test_v15_gate_axis_is_append_only_and_trace_allocation_is_isolated() -> None:
    model, _ = _model_config()
    assert grasp_gate_order(15)[:-2] == grasp_gate_order(14)
    assert grasp_gate_order(15)[-2:] == (
        "joint_pair_alignment_safe",
        "no_active_finger_self_collision",
    )
    v14 = _allocate_traces(model, 3, schema_version=14)
    v15 = _allocate_traces(model, 3, schema_version=15)
    assert not any(name.startswith("joint_pair_") for name in v14)
    assert v15["joint_pair_residual"].shape == (3, 2)
    assert v15["joint_pair_plan_residual_jacobian_2x8"].shape == (21, 2, 8)


def test_v15_alignment_and_slip_recovery_use_only_previous_observation() -> None:
    model, config, controller, step, gate, position, quaternion = _acquire()
    first = controller.command(step)
    np.testing.assert_array_equal(controller.joint_pair_feedback_correction_rad, 0.0)

    centroids = np.zeros((3, 3), dtype=np.float64)
    centroids[1, 2] = 0.0016
    risky = _feedback(
        model,
        config,
        residual=(math.tan(math.radians(0.75)), 0.0),
        centroids=centroids,
        angle_deg=0.75,
    )
    controller.observe(
        step,
        gate,
        position,
        quaternion,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=risky,
    )
    assert first.state is ControlState.MANIPULATE
    assert not controller.aborted

    second = controller.command(step + 1)
    assert second.state is ControlState.MANIPULATE
    assert controller.progress_frozen
    assert controller.joint_pair_freeze_risk_active
    assert controller.joint_pair_slip_recovery_active[1]
    assert controller.joint_pair_alignment_request_rad[model.actuator(ACTIVE_ACTUATORS[0]).id] < 0.0
    assert np.any(controller.joint_pair_slip_recovery_correction_rad != 0.0)
    assert np.any(controller.joint_pair_feedback_correction_rad != 0.0)
    assert np.max(np.abs(controller.joint_pair_feedback_velocity_rad_s)) <= 0.2 + 1e-12


def test_v15_current_frame_abort_is_latched_for_the_next_command() -> None:
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
            residual=(0.02, 0.0),
            angle_deg=1.01,
        ),
    )
    assert controller.aborted
    assert controller.joint_pair_abort_reason == "alignment_angle"
    assert controller.command(step + 1).state is ControlState.ABORT


def test_v15_active_finger_self_collision_resets_verify_window() -> None:
    model, config = _model_config()
    controller = build_grasp_controller(model, config, _phase_steps())
    assert isinstance(
        controller, JointPairAlignedContactPreservingPlannedLiftController
    )
    position = np.asarray((0.071, -0.027, 0.1))
    quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
    controller.latch_initial_pose(position, quaternion)
    safe_gate = _gate(config)
    collision_components = safe_gate.components.copy()
    collision_components[
        safe_gate.gate_order.index("no_active_finger_self_collision")
    ] = False
    collision_gate = GraspGateEvidence(
        components=collision_components,
        target_faces=safe_gate.target_faces,
        hard_abort=False,
        gate_order=safe_gate.gate_order,
    )
    feedback = _feedback(model, config)
    collision_step = controller.close_end + controller.required_stable_steps - 1
    for step in range(collision_step + 1):
        controller.command(step)
        controller.observe(
            step,
            collision_gate if step == collision_step else safe_gate,
            position,
            quaternion,
            actual_joint_qpos_rad=_actual_qpos(model, config),
            operation_feedback=feedback,
        )
    assert not controller.acquired
    assert not controller.aborted

    for step in range(
        collision_step + 1,
        collision_step + controller.required_stable_steps + 1,
    ):
        controller.command(step)
        controller.observe(
            step,
            safe_gate,
            position,
            quaternion,
            actual_joint_qpos_rad=_actual_qpos(model, config),
            operation_feedback=feedback,
        )
    assert controller.acquired
    assert controller.grasp_stable_window_start_step == collision_step + 1


def test_v15_active_finger_self_collision_is_an_operation_hard_abort() -> None:
    model, config, controller, step, gate, position, quaternion = _acquire()
    controller.command(step)
    feedback = _feedback(model, config)
    feedback = OperationFeedback(
        **{
            **feedback.__dict__,
            "active_finger_self_collision": True,
        }
    )
    controller.observe(
        step,
        gate,
        position,
        quaternion,
        actual_joint_qpos_rad=_actual_qpos(model, config),
        operation_feedback=feedback,
    )
    assert controller.aborted
    assert controller.joint_pair_abort_reason == "active_finger_self_collision"
    assert controller.command(step + 1).state is ControlState.ABORT


def test_v15_viewer_status_prioritizes_abort_freeze_and_warning() -> None:
    config = {"schema_version": 15, "joint_pair_feedback": JOINT_PAIR_FEEDBACK.as_config()}
    traces = {
        "control_state": np.asarray(("MANIPULATE",) * 4),
        "joint_pair_angle_deg": np.asarray((0.1, 0.75, 0.2, 0.2)),
        "joint_pair_valid": np.ones(4, dtype=bool),
        "joint_pair_positive_y": np.ones(4, dtype=bool),
        "joint_pair_abort_risk": np.asarray((False, False, False, True)),
        "joint_pair_progress_frozen": np.asarray((False, False, True, True)),
    }
    assert joint_pair_overlay_status(traces, 0, config) == "safe"
    assert joint_pair_overlay_status(traces, 1, config) == "warning"
    assert joint_pair_overlay_status(traces, 2, config) == "frozen"
    assert joint_pair_overlay_status(traces, 3, config) == "abort"


@pytest.mark.parametrize(
    (
        "schema_version",
        "manipulation_success",
        "legacy_operation_executed",
        "manipulation_start_step",
        "operation_sample_count",
        "expected",
    ),
    (
        (15, True, True, 2250, 3500, "passed"),
        (15, False, False, 2250, 3500, "failed"),
        (15, False, False, -1, 0, "not_run"),
        # Sealed schema-v14 reports retain their legacy label semantics.
        (14, False, False, 2250, 3500, "not_run"),
    ),
)
def test_stage_manipulation_label_preserves_legacy_and_reports_v15_attempts(
    schema_version: int,
    manipulation_success: bool,
    legacy_operation_executed: bool,
    manipulation_start_step: int,
    operation_sample_count: int,
    expected: str,
) -> None:
    assert _stage_manipulation_label(
        schema_version=schema_version,
        manipulation_success=manipulation_success,
        legacy_operation_executed=legacy_operation_executed,
        manipulation_start_step=manipulation_start_step,
        operation_sample_count=operation_sample_count,
    ) == expected


def test_v15_abort_after_manipulate_is_reported_failed_not_not_run() -> None:
    summary = run_simulation(load_config(V15_CONFIG))
    assert summary["metrics"]["manipulation_start_step"] >= 0
    assert summary["metrics"]["contact_preserving_planned_lift"][
        "operation_sample_count"
    ] > 0
    # Keep the established hard-check semantics: an aborted/incomplete
    # operation is not valid success evidence.
    assert summary["checks"]["operation_executed"] is False
    assert summary["checks"]["manipulation_completed"] is False
    assert summary["stage_status"]["manipulation_success"] is False
    # Reporting now distinguishes a failed real attempt from no attempt.
    assert summary["stage_status"]["manipulation"] == "failed"
