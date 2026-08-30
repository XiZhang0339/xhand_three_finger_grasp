from __future__ import annotations

from pathlib import Path

import numpy as np

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.controller import (
    GraspGateEvidence,
    GraspVerifyThenManipulateController,
    TargetFaceEvidence,
    grasp_gate_order,
)
from xhand_grasp.scene import build_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)


def _phase_steps() -> dict[str, int]:
    return {
        "settle": 1,
        "close": 1,
        "verify": 260,
        "manipulate": 4,
        "hold": 1,
    }


def _passing_gate() -> GraspGateEvidence:
    order = grasp_gate_order(9)
    target = TargetFaceEvidence(
        target_force_n=np.full(3, 0.1),
        total_distal_force_n=np.full(3, 0.1),
        target_force_purity=np.ones(3),
        target_face_effective=np.ones(3, dtype=bool),
        material_off_target=np.zeros(3, dtype=bool),
        material_active_nondistal=np.zeros(3, dtype=bool),
    )
    return GraspGateEvidence(
        components=np.ones(len(order), dtype=bool),
        target_faces=target,
        hard_abort=False,
        gate_order=order,
    )


def _actual_vector(model, config: dict, *, thumb: float | None = None):
    result = np.zeros(model.nu, dtype=np.float64)
    for name in ACTIVE_ACTUATORS:
        result[model.actuator(name).id] = float(
            config["grasp_pose"]["nominal_joint_qpos_rad"][name]
        )
    if thumb is not None:
        result[
            model.actuator("left_hand_thumb_bend_joint_actuator").id
        ] = thumb
    return result


def _run_controller(actual_thumb: float | None = None):
    config = load_config(CONFIG)
    model, _ = build_model(config)
    controller = GraspVerifyThenManipulateController(
        model, config, _phase_steps()
    )
    position = np.asarray([0.071, -0.027, 0.1175])
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    controller.latch_initial_pose(position, quaternion)
    gate = _passing_gate()
    actual = _actual_vector(model, config, thumb=actual_thumb)
    commands = []
    for step in range(controller.total_steps):
        commands.append(controller.command(step))
        controller.observe(
            step,
            gate,
            position,
            quaternion,
            actual_joint_qpos_rad=actual,
        )
    controller.finish()
    return model, config, controller, commands


def test_v9_locks_the_measured_contact_pose_and_not_the_preload_command():
    model, config, controller, commands = _run_controller()
    assert controller.grasp_acquisition_step == 251
    assert controller.grasp_stable_window_start_step == 2
    assert controller.grasp_stable_window_end_step == 251
    nominal = np.asarray(
        [
            config["grasp_pose"]["nominal_joint_qpos_rad"][name]
            for name in ACTIVE_ACTUATORS
        ]
    )
    assert np.array_equal(controller.grasp_pose_actual_qpos_rad, nominal)
    thumb_id = model.actuator("left_hand_thumb_bend_joint_actuator").id
    assert commands[2].target[thumb_id] == 1.6
    assert controller.grasp_pose_actual_qpos_rad[0] == 1.5


def test_v9_rejects_high_preload_when_actual_thumb_remains_at_1p22():
    _, _, controller, _ = _run_controller(actual_thumb=1.22)
    assert controller.grasp_acquisition_step == -1
    assert controller.grasp_stable_window_start_step == -1
    assert controller.aborted
