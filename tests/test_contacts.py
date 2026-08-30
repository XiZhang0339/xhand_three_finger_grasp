from __future__ import annotations

import math

import numpy as np
import pytest

from xhand_grasp.contacts import (
    DEFAULT_BOX_CONTACT_THRESHOLDS,
    FACE_COUNT,
    FACE_ORDER,
    BoxContactThresholds,
    ContactForceSample,
    Face,
    FaceTraceThresholds,
    aggregate_contact_frame,
    classify_box_contact,
    evaluate_face_trace,
    opposite_face,
    surface_witness,
    validate_three_finger_target_faces,
)


HALF_EXTENTS = np.array([0.015, 0.015, 0.015])
TARGET_FACES = (Face.Y_NEG, Face.Y_POS, Face.Y_POS)


def _rotation_matrix() -> np.ndarray:
    yaw = math.radians(37.0)
    pitch = math.radians(-19.0)
    rz = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ry = np.array(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ]
    )
    return rz @ ry


def _empty_trace(steps: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((steps, 3, FACE_COUNT)), np.zeros((steps, 3))


def _fill_target_contact(
    face_force: np.ndarray,
    tactile: np.ndarray,
    masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    *,
    force_n: float = 0.1,
) -> None:
    steps = face_force.shape[0]
    if masks is None:
        masks = tuple(np.ones(steps, dtype=bool) for _ in range(3))
    for finger_index, (face, mask) in enumerate(zip(TARGET_FACES, masks)):
        face_force[mask, finger_index, int(face)] = force_n
        tactile[mask, finger_index] = 1e-4


def test_face_order_and_opposites_are_stable():
    assert FACE_ORDER == (
        Face.X_POS,
        Face.X_NEG,
        Face.Y_POS,
        Face.Y_NEG,
        Face.Z_POS,
        Face.Z_NEG,
        Face.EDGE_CORNER,
        Face.UNKNOWN,
    )
    for positive, negative in (
        (Face.X_POS, Face.X_NEG),
        (Face.Y_POS, Face.Y_NEG),
        (Face.Z_POS, Face.Z_NEG),
    ):
        assert opposite_face(positive) is negative
        assert opposite_face(negative) is positive
    with pytest.raises(ValueError, match="does not have an opposite"):
        opposite_face(Face.EDGE_CORNER)


@pytest.mark.parametrize("penetration_m", [0.0, 0.0002, 0.002])
@pytest.mark.parametrize("cube_is_geom1", [True, False])
def test_surface_witness_is_rotation_penetration_and_geom_order_invariant(
    penetration_m, cube_is_geom1
):
    rotation = _rotation_matrix()
    center_world = np.array([0.071, -0.0295, 0.112])
    surface_local = np.array([HALF_EXTENTS[0], 0.002, -0.003])
    normal_local = Face.X_POS.outward_normal
    surface_world = center_world + rotation @ surface_local
    outward_world = rotation @ normal_local
    distance = -penetration_m
    contact_position = surface_world + 0.5 * distance * outward_world
    geom1_to_geom2_normal = outward_world if cube_is_geom1 else -outward_world

    witness = surface_witness(
        contact_position,
        distance,
        geom1_to_geom2_normal,
        cube_is_geom1=cube_is_geom1,
    )

    np.testing.assert_allclose(witness.position_world, surface_world, atol=1e-14)
    np.testing.assert_allclose(witness.outward_normal_world, outward_world, atol=1e-14)
    recovered_point = rotation.T @ (witness.position_world - center_world)
    recovered_normal = rotation.T @ witness.outward_normal_world
    result = classify_box_contact(recovered_point, recovered_normal, HALF_EXTENTS)
    assert result.face is Face.X_POS


@pytest.mark.parametrize("face", FACE_ORDER[:6])
def test_classify_all_six_clean_faces(face):
    point = np.zeros(3)
    point[face.axis] = face.sign * HALF_EXTENTS[face.axis]
    result = classify_box_contact(point, face.outward_normal, HALF_EXTENTS)
    assert result.face is face
    assert result.candidate_faces == (face,)
    assert result.surface_error_m == pytest.approx(0.0, abs=1e-15)
    assert result.normal_alignment == pytest.approx(1.0)


def test_edges_corners_and_edge_exclusion_band_are_never_assigned_to_a_face():
    edge = np.array([HALF_EXTENTS[0], HALF_EXTENTS[1], 0.0])
    edge_normal = np.array([1.0, 1.0, 0.0])
    result = classify_box_contact(edge, edge_normal, HALF_EXTENTS)
    assert result.face is Face.EDGE_CORNER
    assert set(result.candidate_faces) == {Face.X_POS, Face.Y_POS}

    corner = HALF_EXTENTS.copy()
    result = classify_box_contact(corner, [1.0, 1.0, 1.0], HALF_EXTENTS)
    assert result.face is Face.EDGE_CORNER
    assert set(result.candidate_faces) == {
        Face.X_POS,
        Face.Y_POS,
        Face.Z_POS,
    }

    margin = DEFAULT_BOX_CONTACT_THRESHOLDS.edge_margin_m
    inside_exclusion_band = np.array(
        [HALF_EXTENTS[0], HALF_EXTENTS[1] - margin + 1e-8, 0.0]
    )
    result = classify_box_contact(
        inside_exclusion_band, Face.X_POS.outward_normal, HALF_EXTENTS
    )
    assert result.face is Face.EDGE_CORNER
    assert result.candidate_faces == (Face.X_POS,)

    exactly_at_margin = np.array(
        [HALF_EXTENTS[0], HALF_EXTENTS[1] - margin, 0.0]
    )
    result = classify_box_contact(
        exactly_at_margin, Face.X_POS.outward_normal, HALF_EXTENTS
    )
    assert result.face is Face.X_POS


def test_surface_tolerance_and_signed_normal_alignment_boundaries_are_inclusive():
    tolerance = DEFAULT_BOX_CONTACT_THRESHOLDS.surface_tolerance_m
    point = np.array([HALF_EXTENTS[0] + tolerance, 0.0, 0.0])
    assert (
        classify_box_contact(point, Face.X_POS.outward_normal, HALF_EXTENTS).face
        is Face.X_POS
    )

    outside = point.copy()
    outside[0] += 2e-12
    assert (
        classify_box_contact(outside, Face.X_POS.outward_normal, HALF_EXTENTS).face
        is Face.UNKNOWN
    )

    tangent = math.sqrt(1.0 - 0.95**2)
    exact_normal = np.array([0.95, tangent, 0.0])
    assert (
        classify_box_contact(
            [HALF_EXTENTS[0], 0.0, 0.0], exact_normal, HALF_EXTENTS
        ).face
        is Face.X_POS
    )
    below_normal = np.array([0.95 - 2e-6, tangent, 0.0])
    below_normal /= np.linalg.norm(below_normal)
    assert (
        classify_box_contact(
            [HALF_EXTENTS[0], 0.0, 0.0], below_normal, HALF_EXTENTS
        ).face
        is Face.UNKNOWN
    )

    assert (
        classify_box_contact(
            [HALF_EXTENTS[0], 0.0, 0.0], Face.X_NEG.outward_normal, HALF_EXTENTS
        ).face
        is Face.UNKNOWN
    )


def test_frame_aggregation_sums_multipoint_force_without_splitting_edges():
    clean = classify_box_contact(
        [0.0, HALF_EXTENTS[1], 0.0], Face.Y_POS.outward_normal, HALF_EXTENTS
    )
    aggregate = aggregate_contact_frame(
        [
            ContactForceSample(1, 0.03, clean),
            ContactForceSample(1, 0.03, Face.Y_POS),
            ContactForceSample(1, 0.4, Face.EDGE_CORNER),
            ContactForceSample(0, 0.2, Face.UNKNOWN),
            ContactForceSample(2, 0.7, distal=False),
        ]
    )

    assert aggregate.face_force_n[1, Face.Y_POS] == pytest.approx(0.06)
    assert aggregate.face_force_n[1, Face.EDGE_CORNER] == pytest.approx(0.4)
    assert aggregate.face_force_n[1, Face.X_POS] == 0.0
    assert aggregate.face_force_n[0, Face.UNKNOWN] == pytest.approx(0.2)
    np.testing.assert_allclose(aggregate.active_nondistal_force_n, [0.0, 0.0, 0.7])
    assert not aggregate.face_force_n.flags.writeable


def test_target_topology_validation_requires_same_and_opposite_faces():
    assert validate_three_finger_target_faces(TARGET_FACES) == TARGET_FACES
    with pytest.raises(ValueError, match="identical"):
        validate_three_finger_target_faces(
            (Face.Y_NEG, Face.Y_POS, Face.X_POS)
        )
    with pytest.raises(ValueError, match="oppose"):
        validate_three_finger_target_faces(
            (Face.X_NEG, Face.Y_POS, Face.Y_POS)
        )
    with pytest.raises(ValueError, match="physical"):
        validate_three_finger_target_faces(
            (Face.EDGE_CORNER, Face.Y_POS, Face.Y_POS)
        )


def test_face_trace_accepts_exact_finger_and_simultaneous_duty_thresholds():
    steps = 100
    face_force, tactile = _empty_trace(steps)
    masks = (
        np.arange(steps) < 80,
        np.arange(steps) < 80,
        (np.arange(steps) < 70) | (np.arange(steps) >= 90),
    )
    _fill_target_contact(face_force, tactile, masks)

    result = evaluate_face_trace(
        face_force, tactile, TARGET_FACES, timestep_s=0.001
    )

    assert result.passed, result.failed_checks
    np.testing.assert_allclose(result.finger_target_duty, [0.8, 0.8, 0.8])
    assert result.simultaneous_target_duty == pytest.approx(0.7)


def test_individual_face_duties_do_not_replace_simultaneous_topology_duty():
    steps = 100
    face_force, tactile = _empty_trace(steps)
    indices = np.arange(steps)
    masks = (indices < 80, (indices >= 10) & (indices < 90), indices >= 20)
    _fill_target_contact(face_force, tactile, masks)

    result = evaluate_face_trace(
        face_force, tactile, TARGET_FACES, timestep_s=0.001
    )

    np.testing.assert_allclose(result.finger_target_duty, [0.8, 0.8, 0.8])
    assert result.simultaneous_target_duty == pytest.approx(0.6)
    assert not result.checks["simultaneous_target_face_duty"]


def test_face_switching_cannot_pass_by_posthoc_modal_face_selection():
    steps = 100
    face_force, tactile = _empty_trace(steps)
    tactile[:] = 1e-4
    alternate_faces = (Face.X_NEG, Face.X_POS, Face.X_POS)
    for finger_index, face in enumerate(alternate_faces):
        face_force[:50, finger_index, int(face)] = 0.1
    for finger_index, face in enumerate(TARGET_FACES):
        face_force[50:, finger_index, int(face)] = 0.1

    result = evaluate_face_trace(
        face_force, tactile, TARGET_FACES, timestep_s=0.001
    )

    np.testing.assert_allclose(result.finger_target_duty, [0.5, 0.5, 0.5])
    assert not result.passed
    assert not result.checks["thumb_target_face_duty"]


def test_edge_force_is_off_target_and_exact_95_percent_purity_is_accepted():
    face_force, tactile = _empty_trace(1)
    tactile[:] = 1e-4
    for finger_index, face in enumerate(TARGET_FACES):
        face_force[0, finger_index, int(face)] = 0.95
        face_force[0, finger_index, Face.EDGE_CORNER] = 0.05
    permissive_duration = FaceTraceThresholds(
        finger_contact_duty_min=1.0,
        simultaneous_contact_duty_min=1.0,
        max_off_target_duty=1.0,
        max_off_target_run_s=1.0,
        max_nondistal_duty=1.0,
        max_nondistal_run_s=1.0,
    )
    result = evaluate_face_trace(
        face_force,
        tactile,
        TARGET_FACES,
        timestep_s=0.001,
        thresholds=permissive_duration,
    )
    assert result.passed, result.failed_checks
    np.testing.assert_allclose(result.target_force_n, 0.95)
    np.testing.assert_allclose(result.off_target_force_n, 0.05)
    np.testing.assert_allclose(result.target_force_purity, 0.95)
    assert not np.any(result.material_off_target)

    face_force[:, 1, Face.Y_POS] = 0.949
    face_force[:, 1, Face.EDGE_CORNER] = 0.051
    result = evaluate_face_trace(
        face_force,
        tactile,
        TARGET_FACES,
        timestep_s=0.001,
        thresholds=permissive_duration,
    )
    assert not result.effective_contact[0, 1]
    assert result.material_off_target[0, 1]


def test_material_off_target_run_at_limit_passes_and_one_step_more_fails():
    steps = 100
    face_force, tactile = _empty_trace(steps)
    _fill_target_contact(face_force, tactile)
    thresholds = FaceTraceThresholds(
        max_off_target_duty=1.0,
        max_off_target_run_s=0.010,
    )

    face_force[:10, 0, Face.EDGE_CORNER] = 0.1
    result = evaluate_face_trace(
        face_force,
        tactile,
        TARGET_FACES,
        timestep_s=0.001,
        thresholds=thresholds,
    )
    assert result.longest_off_target_run_s == pytest.approx(0.010)
    assert result.checks["off_target_contact_run"]

    face_force[10, 0, Face.EDGE_CORNER] = 0.1
    result = evaluate_face_trace(
        face_force,
        tactile,
        TARGET_FACES,
        timestep_s=0.001,
        thresholds=thresholds,
    )
    assert result.longest_off_target_run_s == pytest.approx(0.011)
    assert not result.checks["off_target_contact_run"]


def test_material_off_target_duty_at_limit_passes_and_one_step_more_fails():
    steps = 1000
    face_force, tactile = _empty_trace(steps)
    _fill_target_contact(face_force, tactile)
    thresholds = FaceTraceThresholds(max_off_target_run_s=1.0)

    face_force[:10, 0, Face.UNKNOWN] = 0.1
    result = evaluate_face_trace(
        face_force,
        tactile,
        TARGET_FACES,
        timestep_s=0.001,
        thresholds=thresholds,
    )
    assert result.any_off_target_duty == pytest.approx(0.01)
    assert result.checks["off_target_contact_duty"]

    face_force[10, 0, Face.UNKNOWN] = 0.1
    result = evaluate_face_trace(
        face_force,
        tactile,
        TARGET_FACES,
        timestep_s=0.001,
        thresholds=thresholds,
    )
    assert result.any_off_target_duty == pytest.approx(0.011)
    assert not result.checks["off_target_contact_duty"]


def test_material_active_nondistal_contact_is_a_separate_hard_failure():
    steps = 1000
    face_force, tactile = _empty_trace(steps)
    _fill_target_contact(face_force, tactile)
    nondistal = np.zeros((steps, 3))
    nondistal[:11, 2] = 0.1

    result = evaluate_face_trace(
        face_force,
        tactile,
        TARGET_FACES,
        timestep_s=0.001,
        active_nondistal_force_n=nondistal,
    )

    assert np.all(result.effective_contact)
    assert result.any_nondistal_duty == pytest.approx(0.011)
    assert result.longest_nondistal_run_s == pytest.approx(0.011)
    assert not result.checks["active_nondistal_contact_duty"]
    assert not result.checks["active_nondistal_contact_run"]
    assert not result.passed


def test_invalid_geometry_force_and_trace_inputs_are_rejected():
    with pytest.raises(ValueError, match="smaller than edge_margin"):
        BoxContactThresholds(surface_tolerance_m=0.001, edge_margin_m=0.001)
    with pytest.raises(ValueError, match="positive"):
        classify_box_contact([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        aggregate_contact_frame([ContactForceSample(0, -0.1, Face.X_POS)])
    with pytest.raises(ValueError, match="shape"):
        evaluate_face_trace(
            np.zeros((10, 3, 6)),
            np.zeros((10, 3)),
            TARGET_FACES,
            timestep_s=0.001,
        )
