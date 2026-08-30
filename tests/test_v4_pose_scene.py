from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

import xhand_grasp.scene as scene_module
from xhand_grasp.scene import (
    build_model,
    cube_vertical_half_extent_m,
    palm_plane_ground_angle_deg,
    rpy_degrees_to_rotation_matrix,
    signed_finger_down_tilt_deg,
    solve_press_depth_pose,
)


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "grasp_configs" / "left_three_finger_cube.json"
V2_CONFIG = ROOT / "grasp_configs" / "left_opposed_face_palm_down.json"
V3_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_grasp_then_lift.json"
)
GRAVITY = np.array([0.0, 0.0, -9.81])


def _read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("pitch_deg", "expected_tilt_deg"),
    [
        (0.0, -90.0),
        (80.0, -10.0),
        (90.0, 0.0),
        (100.0, 10.0),
        (180.0, 90.0),
    ],
)
def test_signed_finger_down_tilt_sign_and_range(pitch_deg, expected_tilt_deg):
    rotation = rpy_degrees_to_rotation_matrix([0.0, pitch_deg, 0.0])

    tilt = signed_finger_down_tilt_deg(rotation, GRAVITY)

    assert tilt == pytest.approx(expected_tilt_deg, abs=1e-12)
    assert -90.0 <= tilt <= 90.0


@pytest.mark.parametrize(
    ("pitch_deg", "expected_angle_deg"),
    [(90.0, 0.0), (80.0, 10.0), (0.0, 90.0), (-90.0, 180.0)],
)
def test_palm_plane_ground_angle_is_oriented(pitch_deg, expected_angle_deg):
    rotation = rpy_degrees_to_rotation_matrix([0.0, pitch_deg, 0.0])

    angle = palm_plane_ground_angle_deg(rotation, GRAVITY)

    assert angle == pytest.approx(expected_angle_deg, abs=1e-12)
    assert 0.0 <= angle <= 180.0


def test_pose_angles_use_the_direction_of_nonvertical_gravity():
    gravity = np.array([0.0, -9.81, -9.81])
    gravity_direction = gravity / np.linalg.norm(gravity)
    horizontal = np.array([1.0, 0.0, 0.0])
    tilt_rad = np.radians(27.5)
    local_z_world = (
        np.cos(tilt_rad) * horizontal
        + np.sin(tilt_rad) * gravity_direction
    )
    local_y_world = np.cross(local_z_world, gravity_direction)
    local_y_world /= np.linalg.norm(local_y_world)
    local_x_world = np.cross(local_y_world, local_z_world)
    rotation = np.column_stack(
        (local_x_world, local_y_world, local_z_world)
    )

    assert signed_finger_down_tilt_deg(rotation, gravity) == pytest.approx(
        27.5, abs=1e-12
    )
    assert palm_plane_ground_angle_deg(rotation, gravity) == pytest.approx(
        27.5, abs=1e-12
    )


@pytest.mark.parametrize(
    "measurement",
    [signed_finger_down_tilt_deg, palm_plane_ground_angle_deg],
)
def test_pose_angles_reject_zero_gravity(measurement):
    with pytest.raises(ValueError, match="non-zero"):
        measurement(np.eye(3), np.zeros(3))


def test_press_depth_solve_preserves_the_exact_rigid_transform():
    reference_root = np.array([0.13, -0.04, 0.212])
    cube_world = np.array([0.071, -0.027, 0.118])
    rotation = rpy_degrees_to_rotation_matrix([7.0, 76.0, -13.0])
    press_depth = 0.0065
    cube_y = -0.028
    cube_z = 0.094

    solved = solve_press_depth_pose(
        reference_root_translation_m=reference_root,
        press_depth_m=press_depth,
        rotation_world_from_root=rotation,
        cube_world_position_m=cube_world,
        cube_in_root_y_m=cube_y,
        cube_in_root_z_m=cube_z,
    )
    cube_in_root = np.asarray(solved.cube_in_root_m)
    root_translation = np.asarray(solved.root_translation_m)

    assert solved.cube_in_root_x_m == cube_in_root[0]
    np.testing.assert_allclose(cube_in_root[1:], [cube_y, cube_z], atol=0.0)
    np.testing.assert_allclose(
        root_translation,
        cube_world - rotation @ cube_in_root,
        rtol=0.0,
        atol=0.0,
    )
    assert root_translation[2] == pytest.approx(
        reference_root[2] - press_depth, abs=1e-15
    )


def test_press_depth_solve_rejects_a_near_singular_orientation():
    rotation = rpy_degrees_to_rotation_matrix([0.0, 1e-10, 0.0])

    with pytest.raises(ValueError, match="near-singular"):
        solve_press_depth_pose(
            reference_root_translation_m=[0.0, 0.0, 0.2],
            press_depth_m=0.005,
            rotation_world_from_root=rotation,
            cube_world_position_m=[0.0, 0.0, 0.1],
            cube_in_root_y_m=0.0,
            cube_in_root_z_m=0.1,
        )


def test_schema_v4_rotated_cube_places_its_lowest_vertex_on_support(
    monkeypatch,
):
    config = copy.deepcopy(_read_config(V3_CONFIG))
    config["schema_version"] = 4
    config["cube"]["rpy_deg"] = [23.0, -31.0, 17.0]
    config["cube"]["z_offset_m"] = 0.0017
    config["pose_constraints"] = {
        "reference_hand_translation_m": config["hand_pose"]["translation_m"],
        "finger_down_tilt_deg": 12.0,
        "palm_plane_ground_angle_deg": 14.0,
        "palm_press_depth_m": 0.006,
    }
    # This test isolates scene semantics from the independently versioned
    # schema validator.
    monkeypatch.setattr(scene_module, "validate_config", lambda _: None)

    model, info = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    half = float(config["cube"]["edge_m"]) / 2.0
    local_vertices = np.asarray(
        list(itertools.product((-half, half), repeat=3)), dtype=np.float64
    )
    cube_rotation = data.xmat[info.cube_body_id].reshape(3, 3)
    cube_center = data.xpos[info.cube_body_id]
    world_vertices = cube_center + local_vertices @ cube_rotation.T
    support_surface_z = (
        float(config["scene"]["support_top_z_m"])
        + float(config["cube"]["z_offset_m"])
    )

    assert np.min(world_vertices[:, 2]) == pytest.approx(
        support_surface_z, abs=1e-12
    )
    expected_extent = cube_vertical_half_extent_m(
        config["cube"]["edge_m"],
        rpy_degrees_to_rotation_matrix(config["cube"]["rpy_deg"]),
    )
    assert cube_center[2] == pytest.approx(
        float(config["scene"]["support_top_z_m"])
        + expected_extent
        + float(config["cube"]["z_offset_m"]),
        abs=1e-12,
    )
    assert model.body_jntnum[info.root_body_id] == 0
    assert model.body_mocapid[info.root_body_id] == -1
    np.testing.assert_allclose(
        data.xpos[info.root_body_id],
        config["hand_pose"]["translation_m"],
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("config_path", [V1_CONFIG, V2_CONFIG, V3_CONFIG])
def test_legacy_schema_cube_support_height_remains_edge_over_two(config_path):
    config = _read_config(config_path)
    config["cube"]["rpy_deg"] = [23.0, -31.0, 17.0]
    config["cube"]["z_offset_m"] = 0.0017

    model, info = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    legacy_center_z = (
        float(config["scene"]["support_top_z_m"])
        + float(config["cube"]["edge_m"]) / 2.0
        + float(config["cube"]["z_offset_m"])
    )
    assert data.xpos[info.cube_body_id, 2] == legacy_center_z
    rotated_extent = cube_vertical_half_extent_m(
        config["cube"]["edge_m"],
        rpy_degrees_to_rotation_matrix(config["cube"]["rpy_deg"]),
    )
    assert rotated_extent != pytest.approx(float(config["cube"]["edge_m"]) / 2.0)
