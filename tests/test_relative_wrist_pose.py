from __future__ import annotations

import numpy as np
import pytest

from xhand_grasp.relative_wrist_pose import (
    clockwise_orbit_rotation,
    rotation_matrix_to_rpy_degrees,
    rotvec_degrees_to_rotation_matrix,
    transform_relative_wrist_pose,
)
from xhand_grasp.scene import rpy_degrees_to_rotation_matrix


def test_positive_clockwise_orbit_uses_negative_cube_local_z_rotation():
    rotation = clockwise_orbit_rotation(90.0)

    np.testing.assert_allclose(
        rotation @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=1e-15
    )
    np.testing.assert_allclose(
        rotation @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], atol=1e-15
    )


def test_rpy_matrix_roundtrip_keeps_reference_branch_above_ninety_degrees():
    reference = np.asarray([-1.283447349945079, 120.00830140944369, 7.751648434489488])
    rotation = rpy_degrees_to_rotation_matrix(reference)

    recovered = rotation_matrix_to_rpy_degrees(
        rotation, reference_rpy_deg=reference
    )

    np.testing.assert_allclose(recovered, reference, atol=1e-11)
    np.testing.assert_allclose(
        rpy_degrees_to_rotation_matrix(recovered), rotation, atol=1e-12
    )


def test_coupled_transform_uses_cube_translation_and_hand_local_rotation():
    cube_position = np.asarray([1.0, 2.0, 3.0])
    cube_rotation = rpy_degrees_to_rotation_matrix([0.0, 0.0, 30.0])
    root_in_cube = np.asarray([0.2, 0.0, 0.1])
    root_position = cube_position + cube_rotation @ root_in_cube
    root_rotation = cube_rotation @ rpy_degrees_to_rotation_matrix(
        [10.0, 20.0, 30.0]
    )
    delta = np.asarray([0.01, -0.02, 0.03])
    orbit = clockwise_orbit_rotation(15.0)
    local = rotvec_degrees_to_rotation_matrix([1.0, 2.0, 3.0])

    result = transform_relative_wrist_pose(
        source_cube_world_position_m=cube_position,
        source_cube_world_rotation=cube_rotation,
        source_root_world_position_m=root_position,
        source_root_world_rotation=root_rotation,
        clockwise_orbit_deg=15.0,
        root_delta_cube_m=delta,
        wrist_local_rotvec_deg=[1.0, 2.0, 3.0],
    )

    expected_root_in_cube = orbit @ root_in_cube + delta
    expected_relative_rotation = (
        orbit @ (cube_rotation.T @ root_rotation) @ local
    )
    np.testing.assert_allclose(result.root_in_cube_m, expected_root_in_cube)
    np.testing.assert_allclose(
        result.root_world_position_m,
        cube_position + cube_rotation @ expected_root_in_cube,
    )
    np.testing.assert_allclose(
        result.root_world_rotation,
        cube_rotation @ expected_relative_rotation,
    )
    assert result.diagnostics.orbit_root_cube_distance_m == pytest.approx(
        np.linalg.norm(root_in_cube), abs=1e-12
    )
    assert result.diagnostics.result_root_cube_distance_m == pytest.approx(
        np.linalg.norm(expected_root_in_cube), abs=1e-12
    )


def test_transform_reexpresses_same_relation_in_rotated_target_cube():
    source_cube_rotation = rpy_degrees_to_rotation_matrix([0.0, 0.0, 20.0])
    target_cube_rotation = rpy_degrees_to_rotation_matrix([5.0, -3.0, 70.0])
    source_cube_position = np.asarray([0.1, -0.2, 0.3])
    target_cube_position = np.asarray([-0.4, 0.5, 0.6])
    root_in_cube = np.asarray([0.12, -0.03, 0.08])
    cube_from_root = rpy_degrees_to_rotation_matrix([2.0, 110.0, -4.0])

    result = transform_relative_wrist_pose(
        source_cube_world_position_m=source_cube_position,
        source_cube_world_rotation=source_cube_rotation,
        source_root_world_position_m=(
            source_cube_position + source_cube_rotation @ root_in_cube
        ),
        source_root_world_rotation=source_cube_rotation @ cube_from_root,
        target_cube_world_position_m=target_cube_position,
        target_cube_world_rotation=target_cube_rotation,
    )

    np.testing.assert_allclose(result.root_in_cube_m, root_in_cube)
    np.testing.assert_allclose(
        result.root_world_position_m,
        target_cube_position + target_cube_rotation @ root_in_cube,
    )
    np.testing.assert_allclose(
        result.root_world_rotation, target_cube_rotation @ cube_from_root
    )


def test_relative_pose_helpers_reject_nonfinite_or_nonrotation_inputs():
    with pytest.raises(ValueError, match="finite"):
        clockwise_orbit_rotation(float("nan"))
    with pytest.raises(ValueError, match="orthonormal"):
        rotation_matrix_to_rpy_degrees(np.ones((3, 3)))
    with pytest.raises(ValueError, match="three finite"):
        rotvec_degrees_to_rotation_matrix([0.0, 1.0])
