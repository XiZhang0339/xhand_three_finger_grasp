from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import load_config
from xhand_grasp.joint_pair_geometry import (
    joint_pair_telemetry,
    measure_joint_pair_geometry,
    resolve_joint_pair,
)
from xhand_grasp.scene import build_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_relative_pose_rescue_validated.json"
)


def _rotation_z(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_joint_pair_geometry_uses_rotated_cube_frame():
    cube_rotation = _rotation_z(63.0)
    expected_cube_vector = np.asarray([0.003, 0.021, -0.002])
    first = np.asarray([0.4, -0.2, 0.7])
    second = first + cube_rotation @ expected_cube_vector

    measured = measure_joint_pair_geometry(first, second, cube_rotation)

    np.testing.assert_allclose(
        measured["vector_cube_m"], expected_cube_vector, atol=1e-14
    )
    expected_angle = math.degrees(
        math.acos(
            abs(float(expected_cube_vector[1]))
            / float(np.linalg.norm(expected_cube_vector))
        )
    )
    assert measured["angle_to_cube_y_deg"] == pytest.approx(expected_angle)


def test_joint_pair_geometry_preserves_order_but_angle_is_undirected():
    rotation = _rotation_z(-31.0)
    first = np.asarray([0.1, 0.2, 0.3])
    cube_vector = np.asarray([0.004, -0.025, 0.001])
    second = first + rotation @ cube_vector

    forward = measure_joint_pair_geometry(first, second, rotation)
    reverse = measure_joint_pair_geometry(second, first, rotation)

    np.testing.assert_allclose(
        reverse["vector_cube_m"], -forward["vector_cube_m"], atol=1e-14
    )
    assert reverse["angle_to_cube_y_deg"] == pytest.approx(
        forward["angle_to_cube_y_deg"]
    )
    assert reverse["length_m"] == pytest.approx(forward["length_m"])


@pytest.mark.parametrize(
    ("first", "second", "rotation"),
    [
        (np.zeros(3), np.zeros(3), np.eye(3)),
        (np.zeros(3), np.asarray([np.nan, 0.0, 0.0]), np.eye(3)),
    ],
)
def test_joint_pair_geometry_rejects_zero_or_nonfinite_lines(
    first: np.ndarray,
    second: np.ndarray,
    rotation: np.ndarray,
):
    with pytest.raises(ValueError, match="finite line"):
        measure_joint_pair_geometry(first, second, rotation)


def test_joint_pair_resolution_rejects_invalid_joint_and_keeps_order():
    model, _ = build_model(load_config(CONFIG))
    names = ("left_hand_index_joint1", "left_hand_mid_joint1")

    forward = resolve_joint_pair(model, names)
    reverse = resolve_joint_pair(model, tuple(reversed(names)))
    assert forward is not None
    assert reverse is not None
    assert (forward.first_joint_id, forward.second_joint_id) == (
        reverse.second_joint_id,
        reverse.first_joint_id,
    )
    with pytest.raises(ValueError, match="unknown joint"):
        resolve_joint_pair(model, (names[0], "invalid_joint"))


def test_viewer_independent_telemetry_reports_resolved_names():
    model, _ = build_model(load_config(CONFIG))
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    binding = resolve_joint_pair(
        model, ("left_hand_index_joint1", "left_hand_mid_joint1")
    )
    assert binding is not None

    telemetry = joint_pair_telemetry(data, binding)

    assert telemetry["first_joint"] == "left_hand_index_joint1"
    assert telemetry["second_joint"] == "left_hand_mid_joint1"
    assert 0.0 <= float(telemetry["angle_to_cube_y_deg"]) <= 90.0
