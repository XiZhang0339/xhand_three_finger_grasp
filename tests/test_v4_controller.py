from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import load_config
from xhand_grasp.contacts import FACE_ORDER, Face
from xhand_grasp.controller import (
    GRASP_GATE_ORDER,
    V4_GRASP_GATE_ORDER,
    GraspVerifyThenManipulateController,
    compute_grasp_gate_evidence,
    grasp_gate_order,
)
from xhand_grasp.scene import build_model
from xhand_grasp.trajectory import _phase_steps


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_grasp_then_lift.json"
)


def _v4_config() -> dict:
    config = copy.deepcopy(load_config(SOURCE_CONFIG))
    config["schema_version"] = 4
    return config


def _gate_kwargs() -> dict:
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


def test_v4_gate_axis_is_append_only_and_requires_alignment_evidence():
    assert V4_GRASP_GATE_ORDER[:-1] == GRASP_GATE_ORDER
    assert V4_GRASP_GATE_ORDER[-1] == "contact_height_aligned"
    assert grasp_gate_order(3) == GRASP_GATE_ORDER
    assert grasp_gate_order(4) == V4_GRASP_GATE_ORDER

    with pytest.raises(ValueError, match="contact_height_aligned"):
        compute_grasp_gate_evidence(_v4_config(), **_gate_kwargs())

    evidence = compute_grasp_gate_evidence(
        _v4_config(), **_gate_kwargs(), contact_height_aligned=False
    )
    assert evidence.gate_order == V4_GRASP_GATE_ORDER
    assert not evidence.passed
    assert not evidence.hard_abort
    assert evidence.as_mapping()["contact_height_aligned"] is False


def test_v4_alignment_failure_resets_the_same_contiguous_grasp_window():
    source = load_config(SOURCE_CONFIG)
    model, _ = build_model(source)
    config = _v4_config()
    controller = GraspVerifyThenManipulateController(
        model, config, _phase_steps(model, source)
    )
    verify_start = 1500
    failure_step = verify_start + 100
    expected_acquisition = failure_step + 250
    for step in range(expected_acquisition + 1):
        command = controller.command(step)
        reference_position, reference_quaternion = controller.stability_reference(
            _gate_kwargs()["cube_position"], _gate_kwargs()["cube_quaternion"]
        )
        kwargs = _gate_kwargs()
        kwargs["reference_position"] = reference_position
        kwargs["reference_quaternion"] = reference_quaternion
        evidence = compute_grasp_gate_evidence(
            config,
            **kwargs,
            contact_height_aligned=step != failure_step,
        )
        controller.observe(
            step,
            evidence,
            kwargs["cube_position"],
            kwargs["cube_quaternion"],
        )

    assert command.state.value == "VERIFY"
    assert controller.grasp_acquisition_step == expected_acquisition
    assert controller.consecutive_steps == 250
