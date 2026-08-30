from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, INACTIVE_ACTUATORS, load_config
from xhand_grasp.contacts import FACE_ORDER, Face
from xhand_grasp.controller import (
    V4_GRASP_GATE_ORDER,
    V6_GRASP_GATE_ORDER,
    ControlState,
    GraspVerifyThenManipulateController,
    compute_grasp_gate_evidence,
    grasp_gate_order,
)
from xhand_grasp.scene import build_model
from xhand_grasp.trajectory import smoothstep


ROOT = Path(__file__).resolve().parents[1]
V5_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift.json"
)


@pytest.fixture(scope="module")
def model_and_v5_config():
    config = load_config(V5_CONFIG)
    model, _ = build_model(config)
    return model, config


def _v6_config(source: dict) -> dict:
    config = copy.deepcopy(source)
    config["schema_version"] = 6
    config["pose_preservation"] = {
        "max_translation_m": 0.0005,
        "max_orientation_drift_deg": 1.0,
        "initialize_active_joints_at_pregrasp": False,
    }
    grasp = config["control"]["grasp_targets_rad"]
    config["control"]["pregrasp_targets_rad"] = {
        name: 0.25 * float(grasp[name]) for name in ACTIVE_ACTUATORS
    }
    config["control"]["close_profile"] = {
        name: {"start_fraction": 0.25, "end_fraction": 0.75}
        for name in ACTIVE_ACTUATORS
    }
    config["control"]["close_profile"][
        "left_hand_thumb_bend_joint_actuator"
    ] = {"start_fraction": 0.5, "end_fraction": 1.0}
    config["control"]["close_profile"][
        "left_hand_index_joint1_actuator"
    ] = {"start_fraction": 0.0, "end_fraction": 0.5}
    return config


def _phase_steps() -> dict[str, int]:
    return {
        "settle": 4,
        "close": 4,
        "verify": 250,
        "manipulate": 3,
        "hold": 2,
    }


def _gate_inputs() -> dict:
    force = np.zeros((3, len(FACE_ORDER)), dtype=np.float64)
    force[0, FACE_ORDER.index(Face.X_NEG)] = 0.1
    force[1, FACE_ORDER.index(Face.X_POS)] = 0.1
    force[2, FACE_ORDER.index(Face.X_POS)] = 0.1
    return {
        "distal_face_force_n": force,
        "active_nondistal_force_n": np.zeros(3),
        "tactile_force_n": np.full(3, 0.1),
        "forbidden_contact": False,
        "support_contact": True,
        "floor_contact": False,
        "cube_position": np.array([0.0, 0.0, 0.1]),
        "cube_quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        "reference_position": np.array([0.0, 0.0, 0.1]),
        "reference_quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        "cube_linear_speed_m_s": 0.0,
        "early_lift_m": 0.0,
        "palm_down_angle_deg": 10.0,
        "max_penetration_m": 0.0,
        "finite": True,
        "joint_limits_respected": True,
        "inactive_controls_zero": True,
    }


def _z_quaternion(degrees: float) -> np.ndarray:
    half = math.radians(degrees) / 2.0
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)])


def test_v6_gate_axis_is_append_only_and_requires_pose_history(model_and_v5_config):
    _, source = model_and_v5_config
    config = _v6_config(source)

    assert V6_GRASP_GATE_ORDER[:-1] == V4_GRASP_GATE_ORDER
    assert V6_GRASP_GATE_ORDER[-1] == "initial_pose_history_stable"
    assert grasp_gate_order(3) != V4_GRASP_GATE_ORDER
    assert grasp_gate_order(4) == V4_GRASP_GATE_ORDER
    assert grasp_gate_order(5) == V4_GRASP_GATE_ORDER
    assert grasp_gate_order(6) == V6_GRASP_GATE_ORDER

    with pytest.raises(ValueError, match="initial_pose_history_stable"):
        compute_grasp_gate_evidence(
            config, **_gate_inputs(), contact_height_aligned=True
        )

    passing = compute_grasp_gate_evidence(
        config,
        **_gate_inputs(),
        contact_height_aligned=True,
        initial_pose_history_stable=True,
    )
    assert passing.passed
    assert passing.gate_order == V6_GRASP_GATE_ORDER

    displaced = compute_grasp_gate_evidence(
        config,
        **_gate_inputs(),
        contact_height_aligned=True,
        initial_pose_history_stable=False,
    )
    assert not displaced.passed
    assert not displaced.hard_abort
    assert displaced.as_mapping()["initial_pose_history_stable"] is False


def test_v6_settle_and_per_actuator_delayed_close_targets(model_and_v5_config):
    model, source = model_and_v5_config
    controller = GraspVerifyThenManipulateController(
        model, _v6_config(source), _phase_steps()
    )

    commands = [controller.command(step) for step in range(9)]
    assert [command.state for command in commands[:4]] == [
        ControlState.SETTLE
    ] * 4
    assert [command.state for command in commands[4:8]] == [
        ControlState.CLOSE
    ] * 4
    assert commands[8].state is ControlState.VERIFY
    assert "PREGRASP" not in {state.value for state in ControlState}

    np.testing.assert_allclose(
        commands[0].target,
        smoothstep(0.25) * controller.pregrasp_target,
    )
    np.testing.assert_array_equal(commands[3].target, controller.pregrasp_target)

    thumb_id = model.actuator("left_hand_thumb_bend_joint_actuator").id
    index_id = model.actuator("left_hand_index_joint1_actuator").id
    ordinary_id = model.actuator("left_hand_mid_joint1_actuator").id
    first_close = commands[4].target

    # At global close fraction 0.25, thumb has not reached its delayed start,
    # index is halfway through [0, 0.5], and the ordinary actuator is exactly
    # at the start of [0.25, 0.75].
    assert first_close[thumb_id] == pytest.approx(
        controller.pregrasp_target[thumb_id]
    )
    assert first_close[index_id] == pytest.approx(
        controller.pregrasp_target[index_id]
        + 0.5
        * (
            controller.grasp_target[index_id]
            - controller.pregrasp_target[index_id]
        )
    )
    assert first_close[ordinary_id] == pytest.approx(
        controller.pregrasp_target[ordinary_id]
    )
    np.testing.assert_array_equal(commands[7].target, controller.grasp_target)
    np.testing.assert_array_equal(commands[8].target, controller.grasp_target)

    inactive_ids = [model.actuator(name).id for name in INACTIVE_ACTUATORS]
    for command in commands:
        np.testing.assert_array_equal(command.target[inactive_ids], 0.0)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: config["control"]["close_profile"].pop(
                ACTIVE_ACTUATORS[-1]
            ),
            "exactly the active actuators",
        ),
        (
            lambda config: config["control"]["close_profile"][
                ACTIVE_ACTUATORS[0]
            ].update(start_fraction=0.8, end_fraction=0.2),
            "0 <= start_fraction < end_fraction <= 1",
        ),
        (
            lambda config: config["control"]["close_profile"][
                ACTIVE_ACTUATORS[0]
            ].update(extra=0.5),
            "must contain exactly",
        ),
    ],
)
def test_v6_rejects_invalid_close_profiles(
    model_and_v5_config,
    mutate,
    message,
):
    model, source = model_and_v5_config
    config = _v6_config(source)
    mutate(config)
    with pytest.raises(ValueError, match=message):
        GraspVerifyThenManipulateController(model, config, _phase_steps())


def test_v6_initial_pose_history_is_explicit_fixed_and_sticky(
    model_and_v5_config,
):
    model, source = model_and_v5_config
    controller = GraspVerifyThenManipulateController(
        model, _v6_config(source), _phase_steps()
    )
    position = np.asarray([0.071, -0.027, 0.114])
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])

    with pytest.raises(RuntimeError, match="latched"):
        controller.initial_pose_reference()
    with pytest.raises(RuntimeError, match="latched"):
        controller.stability_reference(position, quaternion)

    controller.latch_initial_pose(position, quaternion)
    assert controller.initial_pose_latched
    with pytest.raises(RuntimeError, match="already been latched"):
        controller.latch_initial_pose(position, quaternion)

    reference_position, reference_quaternion = controller.initial_pose_reference()
    reference_position[:] = 100.0
    reference_quaternion[:] = 0.0
    fresh_position, fresh_quaternion = controller.initial_pose_reference()
    np.testing.assert_array_equal(fresh_position, position)
    np.testing.assert_array_equal(fresh_quaternion, quaternion)

    within = controller.observe_initial_pose_history(
        position + np.asarray([0.0004, 0.0, 0.0]), _z_quaternion(0.5)
    )
    assert within.passed
    assert within.maximum_translation_m == pytest.approx(0.0004)
    assert within.maximum_orientation_drift_deg == pytest.approx(0.5)

    violated = controller.observe_initial_pose_history(
        position + np.asarray([0.0006, 0.0, 0.0]), _z_quaternion(2.0)
    )
    assert not violated.passed
    assert not violated.translation_history_stable
    assert not violated.orientation_history_stable

    returned = controller.observe_initial_pose_history(position, quaternion)
    assert returned.current_translation_m == pytest.approx(0.0)
    assert returned.current_orientation_drift_deg == pytest.approx(0.0)
    assert returned.maximum_translation_m == pytest.approx(0.0006)
    assert returned.maximum_orientation_drift_deg == pytest.approx(2.0)
    assert not returned.passed

    stable_position, stable_quaternion = controller.stability_reference(
        position + 1.0, _z_quaternion(90.0)
    )
    np.testing.assert_array_equal(stable_position, position)
    np.testing.assert_array_equal(stable_quaternion, quaternion)


def test_v6_abort_latches_the_last_sent_target_without_a_command_jump(
    model_and_v5_config,
):
    model, source = model_and_v5_config
    config = _v6_config(source)
    controller = GraspVerifyThenManipulateController(
        model, config, _phase_steps()
    )
    position = np.asarray([0.071, -0.027, 0.114])
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    controller.latch_initial_pose(position, quaternion)

    sent = controller.command(0)
    kwargs = _gate_inputs()
    kwargs["forbidden_contact"] = True
    hard_abort = compute_grasp_gate_evidence(
        config,
        **kwargs,
        contact_height_aligned=True,
        initial_pose_history_stable=True,
    )
    assert hard_abort.hard_abort
    controller.observe(0, hard_abort, position, quaternion)

    first_abort = controller.command(1)
    second_abort = controller.command(2)
    assert first_abort.state is ControlState.ABORT
    assert second_abort.state is ControlState.ABORT
    np.testing.assert_array_equal(first_abort.target, sent.target)
    np.testing.assert_array_equal(second_abort.target, sent.target)
    assert not np.array_equal(sent.target, controller.grasp_target)


def test_v5_settle_and_close_commands_remain_numerically_identical(
    model_and_v5_config,
):
    model, source = model_and_v5_config
    controller = GraspVerifyThenManipulateController(
        model, source, _phase_steps()
    )
    commands = [controller.command(step) for step in range(8)]

    for command in commands[:4]:
        assert command.state is ControlState.SETTLE
        np.testing.assert_array_equal(command.target, 0.0)
    for local_step, command in enumerate(commands[4:8], start=1):
        assert command.state is ControlState.CLOSE
        np.testing.assert_allclose(
            command.target,
            smoothstep(local_step / 4) * controller.grasp_target,
            rtol=0.0,
            atol=0.0,
        )
