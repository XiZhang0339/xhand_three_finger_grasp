"""Cube-local fingertip slip diagnostics for contact-point experiments.

The frozen point target is a grasp-acquisition constraint.  Once manipulation
starts, a distal pad may roll on the correct face, so schema-v13 records this
motion as a soft ranking signal rather than reusing the point-region gate as a
hard operation constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import ACTIVE_FINGERS
from .contact_point_targeting import FACE_AXIS_AND_SIGN


@dataclass(frozen=True, slots=True)
class ContactTangentSlipTrace:
    """Force-weighted grasp baseline and per-step tangent-plane displacement."""

    baseline_centroid_cube_local_m: np.ndarray
    baseline_valid: np.ndarray
    baseline_force_sum_n: np.ndarray
    tangent_slip_from_grasp_m: np.ndarray
    tangent_slip_valid: np.ndarray


def contact_tangent_slip_from_grasp(
    centroid_cube_local_m: np.ndarray,
    centroid_valid: np.ndarray,
    target_face_effective: np.ndarray,
    target_face_force_n: np.ndarray,
    target_faces: Sequence[str],
    *,
    acquisition_start_step: int,
    acquisition_end_step: int,
) -> ContactTangentSlipTrace:
    """Measure pad rolling relative to the acquired grasp contact centroids.

    For each finger, the baseline is the target-face normal-force-weighted
    centroid over the inclusive stable acquisition window.  Displacements are
    expressed in the cube frame and projected onto that finger's target face,
    so cube rigid-body motion and solver motion along the face normal do not
    masquerade as tangential slip.  Invalid samples stay finite at zero and are
    disambiguated by the returned validity masks.
    """

    centroids = np.asarray(centroid_cube_local_m, dtype=np.float64)
    valid = np.asarray(centroid_valid, dtype=bool)
    effective = np.asarray(target_face_effective, dtype=bool)
    force = np.asarray(target_face_force_n, dtype=np.float64)
    if centroids.ndim != 3 or centroids.shape[1:] != (len(ACTIVE_FINGERS), 3):
        raise ValueError("centroid_cube_local_m must have shape (T, 3, 3)")
    expected = centroids.shape[:2]
    if valid.shape != expected:
        raise ValueError("centroid_valid must have shape (T, 3)")
    if effective.shape != expected:
        raise ValueError("target_face_effective must have shape (T, 3)")
    if force.shape != expected:
        raise ValueError("target_face_force_n must have shape (T, 3)")
    if not np.isfinite(centroids).all() or not np.isfinite(force).all():
        raise ValueError("contact centroids and target-face force must be finite")
    if np.any(force < 0.0):
        raise ValueError("target-face force must be non-negative")
    faces = tuple(str(face) for face in target_faces)
    if len(faces) != len(ACTIVE_FINGERS) or any(
        face not in FACE_AXIS_AND_SIGN for face in faces
    ):
        raise ValueError("target_faces must contain three canonical box faces")

    total_steps = centroids.shape[0]
    start = int(acquisition_start_step)
    end = int(acquisition_end_step)
    acquisition_available = 0 <= start <= end < total_steps
    baseline = np.zeros((len(ACTIVE_FINGERS), 3), dtype=np.float64)
    baseline_valid = np.zeros(len(ACTIVE_FINGERS), dtype=bool)
    force_sum = np.zeros(len(ACTIVE_FINGERS), dtype=np.float64)
    if acquisition_available:
        window = slice(start, end + 1)
        for finger_index in range(len(ACTIVE_FINGERS)):
            usable = (
                valid[window, finger_index]
                & effective[window, finger_index]
                & (force[window, finger_index] > 0.0)
            )
            weights = force[window, finger_index][usable]
            if weights.size and float(np.sum(weights)) > 0.0:
                force_sum[finger_index] = float(np.sum(weights))
                baseline[finger_index] = np.average(
                    centroids[window, finger_index][usable],
                    axis=0,
                    weights=weights,
                )
                baseline_valid[finger_index] = True

    slip = np.zeros(expected, dtype=np.float64)
    slip_valid = valid & effective & baseline_valid[None, :]
    for finger_index, face in enumerate(faces):
        normal_axis, _ = FACE_AXIS_AND_SIGN[face]
        tangent_axes = tuple(axis for axis in range(3) if axis != normal_axis)
        delta = centroids[:, finger_index] - baseline[finger_index]
        slip[:, finger_index] = np.linalg.norm(delta[:, tangent_axes], axis=1)
    slip[~slip_valid] = 0.0

    return ContactTangentSlipTrace(
        baseline_centroid_cube_local_m=baseline,
        baseline_valid=baseline_valid,
        baseline_force_sum_n=force_sum,
        tangent_slip_from_grasp_m=slip,
        tangent_slip_valid=slip_valid,
    )


__all__ = ["ContactTangentSlipTrace", "contact_tangent_slip_from_grasp"]
