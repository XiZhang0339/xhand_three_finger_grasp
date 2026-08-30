"""Rigid relative-wrist pose transforms used by search and Viewer overrides.

The convention is intentionally explicit.  A positive orbit angle is
clockwise when looking down the cube-local ``+Z`` axis, and is therefore the
mathematical rotation ``Rz(-angle)``.  Translation residuals are expressed in
the cube frame while rotation-vector residuals are post-multiplied in the
hand's local frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .scene import rpy_degrees_to_rotation_matrix


def _finite_vector3(values: Iterable[float], label: str) -> np.ndarray:
    try:
        vector = np.asarray(tuple(values), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain three finite values") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain three finite values")
    return vector


def _rotation_matrix(values: Iterable[Iterable[float]], label: str) -> np.ndarray:
    try:
        rotation = np.asarray(tuple(tuple(row) for row in values), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite 3x3 rotation matrix") from exc
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{label} must be a finite 3x3 rotation matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0.0):
        raise ValueError(f"{label} must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9):
        raise ValueError(f"{label} must have determinant +1")
    return rotation


def clockwise_orbit_rotation(clockwise_deg: float) -> np.ndarray:
    """Return cube-frame ``Rz(-clockwise_deg)``.

    Positive values move the hand clockwise when viewed from cube-local
    ``+Z`` toward the cube origin.
    """

    angle = float(clockwise_deg)
    if not math.isfinite(angle):
        raise ValueError("clockwise_deg must be finite")
    radians = math.radians(-angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotvec_degrees_to_rotation_matrix(rotvec_deg: Iterable[float]) -> np.ndarray:
    """Convert a degree-valued rotation vector to a proper rotation matrix."""

    vector = np.radians(_finite_vector3(rotvec_deg, "rotvec_deg"))
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-15:
        # First order is unnecessary at the exact identity and returning a
        # new array prevents callers from mutating shared module state.
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = (float(value) for value in axis)
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def _unwrap_degrees_near(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return values + 360.0 * np.round((reference - values) / 360.0)


def rotation_matrix_to_rpy_degrees(
    rotation_world_from_local: Iterable[Iterable[float]],
    *,
    reference_rpy_deg: Iterable[float] | None = None,
) -> np.ndarray:
    """Invert ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    When a reference is supplied, the equivalent Euler branch nearest that
    reference is returned.  This matters for XHAND poses whose pitch is above
    90 degrees: converting them to the principal branch would describe the
    same matrix but produce surprising roll/yaw jumps in resolved configs.
    """

    rotation = _rotation_matrix(rotation_world_from_local, "rotation")
    reference = (
        None
        if reference_rpy_deg is None
        else _finite_vector3(reference_rpy_deg, "reference_rpy_deg")
    )
    sin_pitch = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    cosine_pitch = math.cos(pitch)

    candidates: list[np.ndarray] = []
    if abs(cosine_pitch) > 1e-10:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        candidates.extend(
            (
                np.degrees([roll, pitch, yaw]),
                np.degrees([roll + math.pi, math.pi - pitch, yaw + math.pi]),
            )
        )
    else:
        # At gimbal lock choose the reference roll (or zero), then solve the
        # observable roll/yaw combination exactly.
        roll = 0.0 if reference is None else math.radians(float(reference[0]))
        if sin_pitch > 0.0:
            difference = math.atan2(
                float(rotation[0, 1]), float(rotation[1, 1])
            )
            yaw = roll - difference
        else:
            total = math.atan2(
                float(-rotation[0, 1]), float(rotation[1, 1])
            )
            yaw = total - roll
        candidates.append(np.degrees([roll, pitch, yaw]))

    if reference is None:
        selected = candidates[0]
    else:
        unwrapped = [
            _unwrap_degrees_near(candidate, reference) for candidate in candidates
        ]
        selected = min(
            unwrapped,
            key=lambda candidate: float(np.sum((candidate - reference) ** 2)),
        )
    reconstructed = rpy_degrees_to_rotation_matrix(selected)
    if not np.allclose(reconstructed, rotation, atol=2e-9, rtol=0.0):
        raise ValueError("rotation cannot be represented as a stable RPY triple")
    return np.asarray(selected, dtype=np.float64)


@dataclass(frozen=True)
class RelativeWristPoseDiagnostics:
    """Serializable diagnostics for one relative-pose transform."""

    clockwise_orbit_deg: float
    source_root_in_cube_m: tuple[float, float, float]
    orbit_root_in_cube_m: tuple[float, float, float]
    root_delta_cube_m: tuple[float, float, float]
    result_root_in_cube_m: tuple[float, float, float]
    wrist_local_rotvec_deg: tuple[float, float, float]
    wrist_local_rotvec_norm_deg: float
    source_root_cube_distance_m: float
    orbit_root_cube_distance_m: float
    result_root_cube_distance_m: float

    def as_dict(self) -> dict[str, object]:
        return {
            "clockwise_orbit_deg": self.clockwise_orbit_deg,
            "source_root_in_cube_m": list(self.source_root_in_cube_m),
            "orbit_root_in_cube_m": list(self.orbit_root_in_cube_m),
            "root_delta_cube_m": list(self.root_delta_cube_m),
            "result_root_in_cube_m": list(self.result_root_in_cube_m),
            "wrist_local_rotvec_deg": list(self.wrist_local_rotvec_deg),
            "wrist_local_rotvec_norm_deg": self.wrist_local_rotvec_norm_deg,
            "source_root_cube_distance_m": self.source_root_cube_distance_m,
            "orbit_root_cube_distance_m": self.orbit_root_cube_distance_m,
            "result_root_cube_distance_m": self.result_root_cube_distance_m,
        }


@dataclass(frozen=True)
class RelativeWristPoseResult:
    """World and cube-frame result of one coupled relative-pose transform."""

    root_world_position_m: tuple[float, float, float]
    root_world_rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    root_in_cube_m: tuple[float, float, float]
    cube_from_root_rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    diagnostics: RelativeWristPoseDiagnostics


def _tuple3(vector: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in vector)  # type: ignore[return-value]


def _tuple33(
    matrix: np.ndarray,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    return tuple(_tuple3(row) for row in matrix)  # type: ignore[return-value]


def transform_relative_wrist_pose(
    *,
    source_cube_world_position_m: Iterable[float],
    source_cube_world_rotation: Iterable[Iterable[float]],
    source_root_world_position_m: Iterable[float],
    source_root_world_rotation: Iterable[Iterable[float]],
    target_cube_world_position_m: Iterable[float] | None = None,
    target_cube_world_rotation: Iterable[Iterable[float]] | None = None,
    clockwise_orbit_deg: float = 0.0,
    root_delta_cube_m: Iterable[float] = (0.0, 0.0, 0.0),
    wrist_local_rotvec_deg: Iterable[float] = (0.0, 0.0, 0.0),
) -> RelativeWristPoseResult:
    """Apply the coupled orbit/translation/local-rotation transform.

    The source hand-to-cube relation is measured in the source cube frame.
    The transformed relation is then placed in the target cube frame.  The
    target defaults to the source, but accepting a distinct target lets Viewer
    edge/RPY overrides preserve the intended relative relation.
    """

    source_cube_position = _finite_vector3(
        source_cube_world_position_m, "source_cube_world_position_m"
    )
    source_root_position = _finite_vector3(
        source_root_world_position_m, "source_root_world_position_m"
    )
    source_cube_rotation = _rotation_matrix(
        source_cube_world_rotation, "source_cube_world_rotation"
    )
    source_root_rotation = _rotation_matrix(
        source_root_world_rotation, "source_root_world_rotation"
    )
    target_cube_position = (
        source_cube_position
        if target_cube_world_position_m is None
        else _finite_vector3(
            target_cube_world_position_m, "target_cube_world_position_m"
        )
    )
    target_cube_rotation = (
        source_cube_rotation
        if target_cube_world_rotation is None
        else _rotation_matrix(
            target_cube_world_rotation, "target_cube_world_rotation"
        )
    )
    delta = _finite_vector3(root_delta_cube_m, "root_delta_cube_m")
    rotvec = _finite_vector3(wrist_local_rotvec_deg, "wrist_local_rotvec_deg")
    orbit = clockwise_orbit_rotation(clockwise_orbit_deg)
    local_rotation = rotvec_degrees_to_rotation_matrix(rotvec)

    source_root_in_cube = source_cube_rotation.T @ (
        source_root_position - source_cube_position
    )
    source_cube_from_root_rotation = (
        source_cube_rotation.T @ source_root_rotation
    )
    orbit_root_in_cube = orbit @ source_root_in_cube
    result_root_in_cube = orbit_root_in_cube + delta
    result_cube_from_root_rotation = (
        orbit @ source_cube_from_root_rotation @ local_rotation
    )
    result_root_position = (
        target_cube_position + target_cube_rotation @ result_root_in_cube
    )
    result_root_rotation = target_cube_rotation @ result_cube_from_root_rotation

    diagnostics = RelativeWristPoseDiagnostics(
        clockwise_orbit_deg=float(clockwise_orbit_deg),
        source_root_in_cube_m=_tuple3(source_root_in_cube),
        orbit_root_in_cube_m=_tuple3(orbit_root_in_cube),
        root_delta_cube_m=_tuple3(delta),
        result_root_in_cube_m=_tuple3(result_root_in_cube),
        wrist_local_rotvec_deg=_tuple3(rotvec),
        wrist_local_rotvec_norm_deg=float(np.linalg.norm(rotvec)),
        source_root_cube_distance_m=float(np.linalg.norm(source_root_in_cube)),
        orbit_root_cube_distance_m=float(np.linalg.norm(orbit_root_in_cube)),
        result_root_cube_distance_m=float(np.linalg.norm(result_root_in_cube)),
    )
    return RelativeWristPoseResult(
        root_world_position_m=_tuple3(result_root_position),
        root_world_rotation=_tuple33(result_root_rotation),
        root_in_cube_m=_tuple3(result_root_in_cube),
        cube_from_root_rotation=_tuple33(result_cube_from_root_rotation),
        diagnostics=diagnostics,
    )


__all__ = [
    "RelativeWristPoseDiagnostics",
    "RelativeWristPoseResult",
    "clockwise_orbit_rotation",
    "rotation_matrix_to_rpy_degrees",
    "rotvec_degrees_to_rotation_matrix",
    "transform_relative_wrist_pose",
]
