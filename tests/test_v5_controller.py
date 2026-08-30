from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import load_config
from xhand_grasp.contacts import FACE_ORDER, Face
from xhand_grasp.controller import (
    V4_GRASP_GATE_ORDER,
    compute_grasp_gate_evidence,
    grasp_gate_order,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_grasp_then_lift.json"
)


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


def test_v5_reuses_the_append_only_v4_alignment_gate_axis():
    config = copy.deepcopy(load_config(SOURCE_CONFIG))
    config["schema_version"] = 5

    assert grasp_gate_order(5) == V4_GRASP_GATE_ORDER
    with pytest.raises(ValueError, match="schema v5.*contact_height_aligned"):
        compute_grasp_gate_evidence(config, **_gate_inputs())

    evidence = compute_grasp_gate_evidence(
        config,
        **_gate_inputs(),
        contact_height_aligned=True,
    )
    assert evidence.passed
    assert evidence.gate_order == V4_GRASP_GATE_ORDER
