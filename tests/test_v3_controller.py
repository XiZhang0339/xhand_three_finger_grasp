from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.contacts import FACE_ORDER, Face
from xhand_grasp.controller import (
    GRASP_GATE_ORDER,
    ControlState,
    GraspGateEvidence,
    GraspVerifyThenManipulateController,
    TargetFaceEvidence,
    compute_target_face_evidence,
)
from xhand_grasp.scene import build_model
from xhand_grasp.trajectory import _phase_steps, preflight_config


ROOT = Path(__file__).resolve().parents[1]
V3_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_grasp_then_lift.json"
)


@pytest.fixture(scope="module")
def v3_model_config():
    config = load_config(V3_CONFIG)
    model, _ = build_model(config)
    return model, config


def _gate(*, passed: bool = True, hard_abort: bool = False) -> GraspGateEvidence:
    components = np.full(len(GRASP_GATE_ORDER), passed, dtype=bool)
    target = TargetFaceEvidence(
        target_force_n=np.full(3, 0.1),
        total_distal_force_n=np.full(3, 0.1),
        target_force_purity=np.ones(3),
        target_face_effective=np.full(3, passed, dtype=bool),
        material_off_target=np.zeros(3, dtype=bool),
        material_active_nondistal=np.zeros(3, dtype=bool),
    )
    return GraspGateEvidence(components, target, hard_abort)


def _step_controller(
    controller: GraspVerifyThenManipulateController,
    step: int,
    evidence: GraspGateEvidence,
):
    command = controller.command(step)
    controller.observe(
        step,
        evidence,
        np.array([0.0, 0.0, 0.1]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    return command


def test_v3_phase_plan_and_endpoint_preflight(v3_model_config):
    model, config = v3_model_config
    assert _phase_steps(model, config) == {
        "settle": 500,
        "close": 1000,
        "verify": 750,
        "manipulate": 1500,
        "hold": 1000,
    }
    assert sum(_phase_steps(model, config).values()) == 4750
    preflight_config(config)

    invalid = copy.deepcopy(config)
    actuator = ACTIVE_ACTUATORS[0]
    invalid["control"]["manipulation_delta_rad"][actuator] = 100.0
    # Configuration-domain validation may reject first; either way the actual
    # compiled ctrl range can never be bypassed.
    with pytest.raises(ValueError, match="manipulation_delta|ctrlrange|bounds"):
        preflight_config(invalid)


def test_target_face_evidence_keeps_wrong_and_edge_force_off_target(
    v3_model_config,
):
    _, config = v3_model_config
    force = np.zeros((3, len(FACE_ORDER)))
    target_indices = [
        FACE_ORDER.index(Face.X_NEG),
        FACE_ORDER.index(Face.X_POS),
        FACE_ORDER.index(Face.X_POS),
    ]
    for finger, face_index in enumerate(target_indices):
        force[finger, face_index] = 0.095
        force[finger, Face.EDGE_CORNER] = 0.005

    evidence = compute_target_face_evidence(
        config, force, np.zeros(3), np.full(3, 0.1)
    )
    np.testing.assert_allclose(evidence.target_force_purity, 0.95)
    assert evidence.target_face_effective.tolist() == [True, True, True]
    assert evidence.material_off_target.tolist() == [False, False, False]

    force[1, Face.EDGE_CORNER] += 1e-4
    contaminated = compute_target_face_evidence(
        config, force, np.zeros(3), np.full(3, 0.1)
    )
    assert contaminated.target_face_effective.tolist() == [True, False, True]

    no_touch = compute_target_face_evidence(
        config, force * 0.0 + np.eye(3, len(FACE_ORDER)) * 0.1,
        np.zeros(3),
        np.zeros(3),
    )
    assert not np.any(no_touch.target_face_effective)


def test_grasp_at_t_first_manipulates_at_t_plus_one_and_extends_hold(
    v3_model_config,
):
    model, config = v3_model_config
    controller = GraspVerifyThenManipulateController(
        model, config, _phase_steps(model, config)
    )
    states: list[ControlState] = []
    progress: list[float] = []
    for step in range(controller.total_steps):
        command = _step_controller(controller, step, _gate())
        states.append(command.state)
        progress.append(command.manipulation_progress)
    controller.finish()

    acquisition = 500 + 1000 + 250 - 1
    assert controller.grasp_acquisition_step == acquisition
    assert states[acquisition] is ControlState.VERIFY
    assert controller.manipulation_start_step == acquisition + 1
    assert states[acquisition + 1] is ControlState.MANIPULATE
    assert progress[acquisition] == 0.0
    assert progress[acquisition + 1] > 0.0
    assert controller.manipulation_end_step == acquisition + 1500
    assert states[controller.manipulation_end_step] is ControlState.MANIPULATE
    assert progress[controller.manipulation_end_step] == pytest.approx(1.0)
    assert states[controller.manipulation_end_step + 1] is ControlState.HOLD
    assert states.count(ControlState.HOLD) == 1500
    assert controller.termination_step == 4749


def test_failed_gate_resets_contiguous_window(v3_model_config):
    model, config = v3_model_config
    controller = GraspVerifyThenManipulateController(
        model, config, _phase_steps(model, config)
    )
    verify_start = 1500
    failure_step = verify_start + 100
    expected_acquisition = failure_step + 250
    for step in range(expected_acquisition + 1):
        evidence = _gate(passed=step != failure_step)
        _step_controller(controller, step, evidence)

    assert controller.grasp_acquisition_step == expected_acquisition
    assert controller.consecutive_steps == 250


def test_verify_timeout_enters_abort_and_never_sends_manipulation_target(
    v3_model_config,
):
    model, config = v3_model_config
    controller = GraspVerifyThenManipulateController(
        model, config, _phase_steps(model, config)
    )
    commands = []
    timeout_step = 500 + 1000 + 750 - 1
    for step in range(timeout_step + 2):
        commands.append(_step_controller(controller, step, _gate(passed=False)))

    assert commands[timeout_step].state is ControlState.VERIFY
    assert commands[timeout_step + 1].state is ControlState.ABORT
    np.testing.assert_array_equal(
        commands[timeout_step + 1].target, controller.grasp_target
    )
    assert all(command.manipulation_progress == 0.0 for command in commands)
    assert controller.grasp_acquisition_step == -1
    assert controller.manipulation_start_step == -1
    assert controller.manipulation_end_step == -1
    assert controller.termination_step == timeout_step


def test_hard_safety_failure_aborts_on_the_following_command(v3_model_config):
    model, config = v3_model_config
    controller = GraspVerifyThenManipulateController(
        model, config, _phase_steps(model, config)
    )
    first = _step_controller(controller, 0, _gate(hard_abort=True))
    second = _step_controller(controller, 1, _gate(passed=False))
    assert first.state is ControlState.SETTLE
    assert second.state is ControlState.ABORT
    np.testing.assert_array_equal(second.target, controller.grasp_target)
    assert controller.termination_step == 0
