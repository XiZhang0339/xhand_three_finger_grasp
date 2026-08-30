from __future__ import annotations

import numpy as np
import pytest

from xhand_grasp.contacts import (
    FACE_COUNT,
    ContactForceSample,
    Face,
    FrameContactAggregate,
    aggregate_contact_frame,
    surface_witness,
    target_face_contact_centroids,
    three_finger_height_spread,
)


TARGET_FACES = (Face.Y_NEG, Face.Y_POS, Face.Y_POS)


def _target_samples() -> list[ContactForceSample]:
    return [
        ContactForceSample(
            0,
            1.0,
            Face.Y_NEG,
            surface_witness_position_world_m=np.array([0.10, -0.03, 0.20]),
        ),
        ContactForceSample(
            0,
            3.0,
            Face.Y_NEG,
            surface_witness_position_world_m=np.array([0.20, -0.03, 0.30]),
        ),
        ContactForceSample(
            1,
            2.0,
            Face.Y_POS,
            surface_witness_position_world_m=np.array([0.15, 0.03, 0.25]),
        ),
        ContactForceSample(
            2,
            4.0,
            Face.Y_POS,
            surface_witness_position_world_m=np.array([0.25, 0.03, 0.25]),
        ),
    ]


def test_multipoint_force_position_moment_and_target_centroids():
    aggregate = aggregate_contact_frame(_target_samples())

    np.testing.assert_allclose(
        aggregate.face_position_moment_n_m[0, Face.Y_NEG],
        1.0 * np.array([0.10, -0.03, 0.20])
        + 3.0 * np.array([0.20, -0.03, 0.30]),
    )
    centroids, valid = target_face_contact_centroids(
        aggregate.face_force_n,
        aggregate.face_position_moment_n_m,
        TARGET_FACES,
    )

    np.testing.assert_allclose(centroids[0], [0.175, -0.03, 0.275])
    np.testing.assert_allclose(centroids[1], [0.15, 0.03, 0.25])
    np.testing.assert_allclose(centroids[2], [0.25, 0.03, 0.25])
    np.testing.assert_array_equal(valid, [True, True, True])
    assert not aggregate.face_position_moment_n_m.flags.writeable
    assert not centroids.flags.writeable
    assert not valid.flags.writeable


@pytest.mark.parametrize("penetration_m", [0.0, 0.0002, 0.002])
def test_position_moment_is_invariant_to_cube_geom_order(penetration_m):
    surface_position = np.array([0.04, -0.012, 0.101])
    outward_normal = np.array([0.0, -1.0, 0.0])
    distance = -penetration_m

    witnesses = []
    for cube_is_geom1 in (True, False):
        geom_normal = outward_normal if cube_is_geom1 else -outward_normal
        contact_midpoint = surface_position + 0.5 * distance * outward_normal
        witnesses.append(
            surface_witness(
                contact_midpoint,
                distance,
                geom_normal,
                cube_is_geom1=cube_is_geom1,
            )
        )

    aggregates = [
        aggregate_contact_frame(
            [
                ContactForceSample(
                    0,
                    0.4,
                    Face.Y_NEG,
                    surface_witness_position_world_m=witness.position_world,
                )
            ]
        )
        for witness in witnesses
    ]
    np.testing.assert_allclose(
        aggregates[0].face_position_moment_n_m,
        aggregates[1].face_position_moment_n_m,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        aggregates[0].face_position_moment_n_m[0, Face.Y_NEG],
        0.4 * surface_position,
        atol=1e-15,
    )


def test_zero_target_force_returns_zero_centroid_and_false_mask():
    force = np.zeros((3, FACE_COUNT))
    moment = np.zeros((3, FACE_COUNT, 3))
    force[0, Face.Y_NEG] = 0.1
    moment[0, Face.Y_NEG] = 0.1 * np.array([1.0, 2.0, 3.0])

    centroids, valid = target_face_contact_centroids(force, moment, TARGET_FACES)

    np.testing.assert_allclose(centroids[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(centroids[1:], 0.0)
    np.testing.assert_array_equal(valid, [True, False, False])
    spread, spread_valid = three_finger_height_spread(
        centroids, valid, [0.0, 0.0, -9.81]
    )
    assert spread == 0.0
    assert spread_valid is False


def test_height_spread_uses_arbitrary_nonvertical_gravity_and_5mm_boundary():
    gravity = np.array([0.0, -9.81, -9.81])
    up = -gravity / np.linalg.norm(gravity)
    tangent = np.array([1.0, 0.0, 0.0])
    origin = np.array([0.02, -0.01, 0.10])
    centroids = np.vstack(
        [
            origin,
            origin + 0.002 * up + 0.03 * tangent,
            origin + 0.005 * up - 0.02 * tangent,
        ]
    )

    spread, valid = three_finger_height_spread(
        centroids, np.ones(3, dtype=bool), gravity
    )

    assert valid is True
    assert spread == pytest.approx(0.005, abs=1e-15)
    assert spread <= 0.005 + 1e-12


def test_legacy_aggregate_constructor_and_samples_default_to_zero_moments():
    aggregate = aggregate_contact_frame([ContactForceSample(1, 0.2, Face.X_POS)])
    direct = FrameContactAggregate(np.zeros((3, FACE_COUNT)), np.zeros(3))

    np.testing.assert_array_equal(aggregate.face_position_moment_n_m, 0.0)
    np.testing.assert_array_equal(direct.face_position_moment_n_m, 0.0)


def test_zero_gravity_and_invalid_shapes_are_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        three_finger_height_spread(np.zeros((3, 3)), np.ones(3), np.zeros(3))
    with pytest.raises(ValueError, match="face_position_moment_n_m"):
        target_face_contact_centroids(
            np.zeros((3, FACE_COUNT)), np.zeros((3, FACE_COUNT, 2)), TARGET_FACES
        )
