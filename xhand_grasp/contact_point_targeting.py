"""Schema-v12 cube-local contact-point targeting primitives.

The simulation, offline evaluator, search code and Viewer all consume the
same frozen point plan.  Keeping the coordinate conversion here prevents a
world-frame cache (or a rotated cube) from silently changing the intended
surface locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .config import ACTIVE_FINGERS


FACE_AXIS_AND_SIGN: dict[str, tuple[int, float]] = {
    "+X": (0, 1.0),
    "-X": (0, -1.0),
    "+Y": (1, 1.0),
    "-Y": (1, -1.0),
    "+Z": (2, 1.0),
    "-Z": (2, -1.0),
}


@dataclass(frozen=True)
class ContactPointPlan:
    """Validated, immutable numeric view of ``contact_point_plan``."""

    point_plan_id: str
    target_points_cube_local_m: np.ndarray
    target_faces: tuple[str, str, str]
    target_radius_m: float

    def __post_init__(self) -> None:
        points = np.asarray(self.target_points_cube_local_m, dtype=np.float64)
        if points.shape != (len(ACTIVE_FINGERS), 3):
            raise ValueError("target contact points must have shape (3, 3)")
        if not np.isfinite(points).all():
            raise ValueError("target contact points must be finite")
        if len(self.target_faces) != len(ACTIVE_FINGERS) or any(
            face not in FACE_AXIS_AND_SIGN for face in self.target_faces
        ):
            raise ValueError("target contact faces must be canonical box faces")
        radius = float(self.target_radius_m)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("target contact radius must be positive and finite")
        if not isinstance(self.point_plan_id, str) or not self.point_plan_id:
            raise ValueError("contact point plan id must be a non-empty string")
        object.__setattr__(self, "target_points_cube_local_m", points.copy())
        object.__setattr__(self, "target_radius_m", radius)


@dataclass(frozen=True)
class ContactPointObservation:
    """One post-step contact-point observation in the cube frame."""

    centroid_cube_local_m: np.ndarray
    tangent_error_m: np.ndarray
    within_radius: np.ndarray

    @property
    def all_effective_within_radius(self) -> bool:
        return bool(np.all(self.within_radius))


def contact_point_plan_from_config(config: Mapping[str, Any]) -> ContactPointPlan:
    """Resolve the frozen schema-v12 face/face-YZ representation.

    Schema-v12 intentionally supports only the opposed ``+/-X`` faces used by
    this experiment.  For those faces, ``face_yz_m`` is unambiguous and the X
    coordinate is derived from the configured cube half-edge rather than
    copied from a potentially stale search artifact.
    """

    if int(config.get("schema_version", 1)) < 12:
        raise ValueError("contact point plans require schema v12+")
    raw_plan = config.get("contact_point_plan")
    if not isinstance(raw_plan, Mapping):
        raise ValueError("schema v12 requires contact_point_plan")
    # ``points``/``yz_m`` is the registered schema-v12 spelling.  Accept the
    # earlier planning spelling as a read-compatible alias so interrupted
    # search prefixes remain inspectable, while all traces use one canonical
    # expanded 3-D representation.
    points_mapping = raw_plan.get("points", raw_plan.get("target_points"))
    if not isinstance(points_mapping, Mapping) or set(points_mapping) != set(
        ACTIVE_FINGERS
    ):
        raise ValueError(
            "contact_point_plan.points must contain exactly thumb, index and mid"
        )

    edge_m = float(config["cube"]["edge_m"])
    declared_edge = float(raw_plan.get("cube_edge_m", edge_m))
    if (
        not np.isfinite(edge_m)
        or edge_m <= 0.0
        or not np.isfinite(declared_edge)
        or abs(declared_edge - edge_m) > 1e-12
    ):
        raise ValueError("contact point plan cube_edge_m must match cube.edge_m")
    if raw_plan.get("coordinate_frame") != "cube_local":
        raise ValueError("contact point plan coordinate_frame must be cube_local")
    if raw_plan.get("frozen") is not True:
        raise ValueError("contact point plan must be frozen before simulation")

    topology_faces = config["contact_topology"]["target_faces"]
    points = np.zeros((len(ACTIVE_FINGERS), 3), dtype=np.float64)
    faces: list[str] = []
    for finger_index, finger in enumerate(ACTIVE_FINGERS):
        entry = points_mapping[finger]
        if not isinstance(entry, Mapping):
            raise ValueError(f"target point for {finger} must be a mapping")
        face = str(entry.get("face"))
        if face not in {"+X", "-X"}:
            raise ValueError("schema-v12 face_yz_m targets support only +/-X faces")
        if str(topology_faces[finger]) != face:
            raise ValueError(
                f"target point face for {finger} must match contact topology"
            )
        face_yz = np.asarray(
            entry.get("yz_m", entry.get("face_yz_m")), dtype=np.float64
        )
        if face_yz.shape != (2,) or not np.isfinite(face_yz).all():
            raise ValueError(f"target point face_yz_m for {finger} must be finite [y,z]")
        _, sign = FACE_AXIS_AND_SIGN[face]
        points[finger_index] = (sign * edge_m / 2.0, face_yz[0], face_yz[1])
        faces.append(face)

    expanded = raw_plan.get("target_points_cube_local_m")
    if expanded is not None:
        if not isinstance(expanded, Mapping) or set(expanded) != set(
            ACTIVE_FINGERS
        ):
            raise ValueError(
                "contact_point_plan.target_points_cube_local_m must contain "
                "exactly thumb, index and mid"
            )
        declared = np.asarray(
            [expanded[finger] for finger in ACTIVE_FINGERS], dtype=np.float64
        )
        if (
            declared.shape != points.shape
            or not np.isfinite(declared).all()
            or not np.allclose(declared, points, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(
                "expanded target_points_cube_local_m does not match face/face_yz_m"
            )
        points = declared

    return ContactPointPlan(
        point_plan_id=str(raw_plan.get("point_plan_id", "")),
        target_points_cube_local_m=points,
        target_faces=tuple(faces),  # type: ignore[arg-type]
        target_radius_m=float(raw_plan.get("target_radius_m", 0.0)),
    )


def world_points_to_cube_local(
    points_world_m: np.ndarray,
    cube_position_world_m: np.ndarray,
    cube_rotation_world_from_local: np.ndarray,
) -> np.ndarray:
    """Transform three world points to cube-local coordinates."""

    points = np.asarray(points_world_m, dtype=np.float64)
    position = np.asarray(cube_position_world_m, dtype=np.float64)
    rotation = np.asarray(cube_rotation_world_from_local, dtype=np.float64)
    if points.shape != (len(ACTIVE_FINGERS), 3):
        raise ValueError("world contact centroids must have shape (3, 3)")
    if position.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("cube pose must contain a position and 3x3 rotation")
    if not all(np.isfinite(value).all() for value in (points, position, rotation)):
        raise ValueError("contact centroids and cube pose must be finite")
    return (rotation.T @ (points - position).T).T


def contact_point_observation(
    plan: ContactPointPlan,
    centroid_world_m: np.ndarray,
    centroid_valid: np.ndarray,
    target_face_effective: np.ndarray,
    cube_position_world_m: np.ndarray,
    cube_rotation_world_from_local: np.ndarray,
) -> ContactPointObservation:
    """Compute tangent-plane errors and the strict per-finger region gate."""

    valid = np.asarray(centroid_valid, dtype=bool)
    effective = np.asarray(target_face_effective, dtype=bool)
    if valid.shape != (len(ACTIVE_FINGERS),) or effective.shape != valid.shape:
        raise ValueError("contact validity/effectiveness must have shape (3,)")
    local = world_points_to_cube_local(
        centroid_world_m,
        cube_position_world_m,
        cube_rotation_world_from_local,
    )
    error = np.zeros(len(ACTIVE_FINGERS), dtype=np.float64)
    for finger_index, face in enumerate(plan.target_faces):
        normal_axis, _ = FACE_AXIS_AND_SIGN[face]
        tangent_axes = [axis for axis in range(3) if axis != normal_axis]
        delta = (
            local[finger_index] - plan.target_points_cube_local_m[finger_index]
        )
        error[finger_index] = float(np.linalg.norm(delta[tangent_axes]))
    within = (
        valid
        & effective
        & np.isfinite(error)
        & (error <= plan.target_radius_m + 1e-12)
    )
    # Invalid centroids are not point estimates.  Retaining a deterministic
    # zero in the numeric cache keeps NPZ arrays finite; ``within`` is the mask
    # that prevents those zeros from being interpreted as valid evidence.
    local = local.copy()
    local[~valid] = 0.0
    error[~valid] = 0.0
    return ContactPointObservation(local, error, within)


__all__ = [
    "ContactPointObservation",
    "ContactPointPlan",
    "FACE_AXIS_AND_SIGN",
    "contact_point_observation",
    "contact_point_plan_from_config",
    "world_points_to_cube_local",
]
