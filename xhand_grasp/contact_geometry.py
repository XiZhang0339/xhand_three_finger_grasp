"""MuJoCo adapters for static distal-contact geometry diagnostics.

The face classifier in :mod:`xhand_grasp.contacts` is intentionally pure
NumPy.  This module owns the MuJoCo-specific half of static screening: it asks
the collision engine for the actual closest points of the cube and distal
collision geoms, attributes the cube-side witness to a requested face, and can
repeat that query over a deterministic closure sweep.

No state produced by a sweep leaks back to its caller.  The full ``qpos`` and
``ctrl`` vectors are restored and ``mj_forward`` is run before returning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contacts import (
    DEFAULT_BOX_CONTACT_THRESHOLDS,
    BoxContactClassification,
    BoxContactThresholds,
    Face,
    classify_box_contact,
)


DEFAULT_GEOM_DISTANCE_MAX_M = 1.0
DEFAULT_STATIC_TARGET_GAP_M = (-0.0005, 0.003)
DEFAULT_STATIC_DISTAL_PRELOAD_CAP_M = 0.010
_NUMERIC_TOLERANCE = 1e-12


def _readonly(values: ArrayLike, *, dtype: type = np.float64) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _physical_face(value: Face | int) -> Face:
    face = Face(value)
    if not face.is_physical:
        raise ValueError("target face must be a physical cube face")
    return face


@dataclass(frozen=True, slots=True)
class NearestTaxelAssignment:
    """Nearest tactile-site assignment for one distal contact witness."""

    taxel_index: int | None
    distance_m: float

    def __post_init__(self) -> None:
        distance = float(self.distance_m)
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("distance_m must be finite and non-negative")
        object.__setattr__(self, "distance_m", distance)
        if self.taxel_index is not None and int(self.taxel_index) < 0:
            raise ValueError("taxel_index must be non-negative")

    @property
    def assigned(self) -> bool:
        return self.taxel_index is not None


def nearest_taxel_assignment(
    distal_witness_world_m: ArrayLike,
    taxel_positions_world_m: ArrayLike,
    *,
    max_assignment_distance_m: float = 0.006,
) -> NearestTaxelAssignment:
    """Assign one distal witness once to its nearest tactile site.

    The returned distance is always the true nearest-site distance, including
    when it exceeds the assignment radius and ``taxel_index`` is ``None``.
    The inclusive radius makes the configured 6 mm boundary deterministic.
    """

    witness = np.asarray(distal_witness_world_m, dtype=np.float64)
    sites = np.asarray(taxel_positions_world_m, dtype=np.float64)
    maximum = float(max_assignment_distance_m)
    if witness.shape != (3,) or not np.isfinite(witness).all():
        raise ValueError("distal_witness_world_m must contain three finite values")
    if sites.ndim != 2 or sites.shape[1] != 3 or sites.shape[0] == 0:
        raise ValueError("taxel_positions_world_m must have shape (taxels, 3)")
    if not np.isfinite(sites).all():
        raise ValueError("taxel_positions_world_m must be finite")
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_assignment_distance_m must be positive and finite")
    distances = np.linalg.norm(sites - witness, axis=1)
    index = int(np.argmin(distances))
    distance = float(distances[index])
    return NearestTaxelAssignment(
        taxel_index=(index if distance <= maximum + _NUMERIC_TOLERANCE else None),
        distance_m=distance,
    )


def _collidable_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    return bool(
        int(model.geom_contype[geom_id]) != 0
        or int(model.geom_conaffinity[geom_id]) != 0
    )


def distal_collision_geom_ids(
    model: mujoco.MjModel,
    distal_weld_ids: Mapping[str, int],
) -> dict[str, tuple[int, ...]]:
    """Return collidable geom ids rigidly attached to each distal body.

    Weld ids, rather than direct body ids, are used because imported mesh
    bodies can be joined by fixed bodies after MuJoCo compiles the model.
    Mapping iteration order is preserved in the result.
    """

    result: dict[str, tuple[int, ...]] = {}
    for finger, weld_id_value in distal_weld_ids.items():
        weld_id = int(weld_id_value)
        geom_ids = tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if _collidable_geom(model, geom_id)
            and int(model.body_weldid[int(model.geom_bodyid[geom_id])]) == weld_id
        )
        if not geom_ids:
            raise ValueError(f"no collidable distal geometry found for {finger}")
        result[str(finger)] = geom_ids
    return result


def active_nondistal_collision_geom_ids(
    model: mujoco.MjModel,
    hand_body_parts: Mapping[int, str],
    distal_weld_ids: Mapping[str, int],
) -> dict[str, tuple[int, ...]]:
    """Return active-finger collision geoms outside each distal weld."""

    result: dict[str, tuple[int, ...]] = {}
    for finger, distal_weld_value in distal_weld_ids.items():
        distal_weld = int(distal_weld_value)
        result[str(finger)] = tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if _collidable_geom(model, geom_id)
            and hand_body_parts.get(int(model.geom_bodyid[geom_id])) == finger
            and int(model.body_weldid[int(model.geom_bodyid[geom_id])]) != distal_weld
        )
    return result


@dataclass(frozen=True, slots=True)
class GeomDistanceWitness:
    """Closest-point evidence for one cube/distal collision-geom pair."""

    distal_geom_id: int
    signed_distance_m: float
    cube_point_world_m: NDArray[np.float64]
    distal_point_world_m: NDArray[np.float64]
    cube_point_local_m: NDArray[np.float64]
    classification: BoxContactClassification

    def __post_init__(self) -> None:
        distance = float(self.signed_distance_m)
        if not np.isfinite(distance):
            raise ValueError("signed_distance_m must be finite")
        object.__setattr__(self, "signed_distance_m", distance)
        for field in (
            "cube_point_world_m",
            "distal_point_world_m",
            "cube_point_local_m",
        ):
            values = np.asarray(getattr(self, field), dtype=np.float64)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{field} must contain three finite values")
            object.__setattr__(self, field, _readonly(values))

    @property
    def penetration_m(self) -> float:
        return max(0.0, -self.signed_distance_m)

    @property
    def face(self) -> Face:
        return self.classification.face


def geom_distance_witness(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    cube_geom_id: int,
    distal_geom_id: int,
    distance_max_m: float = DEFAULT_GEOM_DISTANCE_MAX_M,
    thresholds: BoxContactThresholds = DEFAULT_BOX_CONTACT_THRESHOLDS,
) -> GeomDistanceWitness | None:
    """Query one real collision pair with the cube ordered as ``geom1``.

    MuJoCo writes the cube and distal witnesses into ``fromto[:3]`` and
    ``fromto[3:]`` respectively.  If the pair is farther than ``distmax`` it
    returns ``distmax`` and no usable segment, represented here as ``None``.
    For a non-zero signed distance, ``(to-from)/distance`` consistently points
    along the cube-outward normal for both separation and penetration.
    """

    distance_max = float(distance_max_m)
    if not np.isfinite(distance_max) or distance_max <= 0.0:
        raise ValueError("distance_max_m must be positive and finite")
    cube_id = int(cube_geom_id)
    distal_id = int(distal_geom_id)
    if cube_id == distal_id:
        raise ValueError("cube and distal geom ids must differ")
    if not (0 <= cube_id < model.ngeom and 0 <= distal_id < model.ngeom):
        raise ValueError("geom id is outside the model")

    from_to = np.zeros(6, dtype=np.float64)
    distance = float(
        mujoco.mj_geomDistance(
            model,
            data,
            cube_id,
            distal_id,
            distance_max,
            from_to,
        )
    )
    if distance >= distance_max - _NUMERIC_TOLERANCE:
        return None

    cube_point = from_to[:3].copy()
    distal_point = from_to[3:].copy()
    cube_rotation = np.asarray(data.geom_xmat[cube_id], dtype=np.float64).reshape(3, 3)
    cube_position = np.asarray(data.geom_xpos[cube_id], dtype=np.float64)
    cube_point_local = cube_rotation.T @ (cube_point - cube_position)

    segment = distal_point - cube_point
    if abs(distance) > _NUMERIC_TOLERANCE and np.linalg.norm(segment) > _NUMERIC_TOLERANCE:
        outward_world = segment / distance
        outward_world /= np.linalg.norm(outward_world)
        outward_local = cube_rotation.T @ outward_world
    else:
        # At exact touching the closest-point segment has zero length.  Select
        # the nearest box plane only to supply the signed normal needed by the
        # strict classifier; plane/edge tolerances still decide attribution.
        half_extents = np.asarray(model.geom_size[cube_id], dtype=np.float64)
        plane_errors = []
        for face in tuple(Face)[:6]:
            plane_errors.append(
                abs(
                    float(cube_point_local[face.axis])
                    - face.sign * float(half_extents[face.axis])
                )
            )
        outward_local = tuple(Face)[int(np.argmin(plane_errors))].outward_normal

    classification = classify_box_contact(
        cube_point_local,
        outward_local,
        np.asarray(model.geom_size[cube_id], dtype=np.float64),
        thresholds=thresholds,
    )
    return GeomDistanceWitness(
        distal_geom_id=distal_id,
        signed_distance_m=distance,
        cube_point_world_m=cube_point,
        distal_point_world_m=distal_point,
        cube_point_local_m=cube_point_local,
        classification=classification,
    )


def nearest_distal_target_witness(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    cube_geom_id: int,
    distal_geom_ids: Sequence[int],
    target_face: Face | int,
    distance_max_m: float = DEFAULT_GEOM_DISTANCE_MAX_M,
    thresholds: BoxContactThresholds = DEFAULT_BOX_CONTACT_THRESHOLDS,
) -> GeomDistanceWitness | None:
    """Return the closest real distal witness attributable to ``target_face``."""

    target = _physical_face(target_face)
    candidates: list[GeomDistanceWitness] = []
    for geom_id in distal_geom_ids:
        witness = geom_distance_witness(
            model,
            data,
            cube_geom_id=cube_geom_id,
            distal_geom_id=int(geom_id),
            distance_max_m=distance_max_m,
            thresholds=thresholds,
        )
        if witness is not None and witness.face is target:
            candidates.append(witness)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            abs(item.signed_distance_m),
            item.signed_distance_m,
            item.distal_geom_id,
        ),
    )


def _flatten_geom_ids(values: Mapping[str, Sequence[int]] | Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, Mapping):
        flattened = [geom_id for geom_ids in values.values() for geom_id in geom_ids]
    else:
        flattened = list(values)
    return tuple(dict.fromkeys(int(geom_id) for geom_id in flattened))


def _maximum_cube_penetration_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
    geom_ids: Sequence[int],
    distance_max_m: float,
) -> float:
    maximum = 0.0
    for geom_id in geom_ids:
        distance = float(
            mujoco.mj_geomDistance(
                model,
                data,
                int(cube_geom_id),
                int(geom_id),
                float(distance_max_m),
                None,
            )
        )
        maximum = max(maximum, max(0.0, -distance))
    return maximum


@dataclass(frozen=True, slots=True)
class ClosureSweepSample:
    """Static closest-point evidence at one closure fraction."""

    alpha: float
    target_witnesses: tuple[GeomDistanceWitness | None, ...]
    target_gap_reached: NDArray[np.bool_]
    target_signed_gap_m: NDArray[np.float64]
    target_height_m: NDArray[np.float64]
    target_height_valid: NDArray[np.bool_]
    target_height_spread_m: float
    max_distal_preload_m: float
    max_active_nondistal_penetration_m: float
    max_forbidden_penetration_m: float

    def __post_init__(self) -> None:
        alpha = float(self.alpha)
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and within [0, 1]")
        object.__setattr__(self, "alpha", alpha)
        count = len(self.target_witnesses)
        for field, dtype in (
            ("target_gap_reached", np.bool_),
            ("target_signed_gap_m", np.float64),
            ("target_height_m", np.float64),
            ("target_height_valid", np.bool_),
        ):
            values = np.asarray(getattr(self, field), dtype=dtype)
            if values.shape != (count,):
                raise ValueError(f"{field} must match target_witnesses")
            object.__setattr__(self, field, _readonly(values, dtype=dtype))
        for field in (
            "target_height_spread_m",
            "max_distal_preload_m",
            "max_active_nondistal_penetration_m",
            "max_forbidden_penetration_m",
        ):
            value = float(getattr(self, field))
            if np.isnan(value) or value < 0.0:
                raise ValueError(f"{field} must be non-negative and not NaN")
            object.__setattr__(self, field, value)

    @property
    def all_targets_reached(self) -> bool:
        return bool(np.all(self.target_gap_reached))

    @property
    def active_nondistal_clear(self) -> bool:
        return self.max_active_nondistal_penetration_m <= _NUMERIC_TOLERANCE

    @property
    def forbidden_clear(self) -> bool:
        return self.max_forbidden_penetration_m <= _NUMERIC_TOLERANCE


@dataclass(frozen=True, slots=True)
class ClosureSweepResult:
    """Ordered samples returned by :func:`scan_distal_closure`."""

    samples: tuple[ClosureSweepSample, ...]
    target_gap_m: tuple[float, float]
    distal_preload_cap_m: float

    @property
    def eligible_samples(self) -> tuple[ClosureSweepSample, ...]:
        return tuple(
            sample
            for sample in self.samples
            if sample.all_targets_reached
            and sample.active_nondistal_clear
            and sample.forbidden_clear
            and sample.max_distal_preload_m
            <= self.distal_preload_cap_m + _NUMERIC_TOLERANCE
        )

    @property
    def first_all_target_alpha(self) -> float | None:
        for sample in self.samples:
            if sample.all_targets_reached:
                return sample.alpha
        return None


def scan_distal_closure(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    cube_geom_id: int,
    finger_order: Sequence[str],
    distal_geom_ids: Mapping[str, Sequence[int]],
    target_faces: Sequence[Face | int],
    actuator_qpos_addresses: Sequence[int],
    open_targets_rad: ArrayLike,
    closed_targets_rad: ArrayLike,
    alphas: Sequence[float],
    active_nondistal_geom_ids: Mapping[str, Sequence[int]] | Sequence[int] = (),
    forbidden_geom_ids: Sequence[int] = (),
    target_gap_m: tuple[float, float] = DEFAULT_STATIC_TARGET_GAP_M,
    distal_preload_cap_m: float = DEFAULT_STATIC_DISTAL_PRELOAD_CAP_M,
    distance_max_m: float = DEFAULT_GEOM_DISTANCE_MAX_M,
    thresholds: BoxContactThresholds = DEFAULT_BOX_CONTACT_THRESHOLDS,
) -> ClosureSweepResult:
    """Evaluate collision witnesses over a deterministic linear closure sweep.

    This function deliberately changes only actuator-owned qpos entries.  It
    does not step dynamics, alter the cube pose, or infer actuator ids from
    numerical coincidence.  Callers choose the closure fractions explicitly.
    """

    fingers = tuple(str(value) for value in finger_order)
    faces = tuple(_physical_face(value) for value in target_faces)
    if len(fingers) == 0 or len(fingers) != len(faces):
        raise ValueError("finger_order and target_faces must have equal non-zero length")
    if set(fingers) != set(distal_geom_ids):
        raise ValueError("distal_geom_ids must contain exactly finger_order")
    addresses = np.asarray(actuator_qpos_addresses, dtype=np.int64)
    opened = np.asarray(open_targets_rad, dtype=np.float64)
    closed = np.asarray(closed_targets_rad, dtype=np.float64)
    if addresses.ndim != 1 or opened.shape != addresses.shape or closed.shape != addresses.shape:
        raise ValueError("actuator qpos addresses and target arrays must have equal 1D shape")
    if not np.isfinite(opened).all() or not np.isfinite(closed).all():
        raise ValueError("closure targets must be finite")
    if np.any(addresses < 0) or np.any(addresses >= model.nq):
        raise ValueError("actuator_qpos_addresses contains an invalid qpos address")
    alpha_values = np.asarray(alphas, dtype=np.float64)
    if alpha_values.ndim != 1 or alpha_values.size == 0:
        raise ValueError("alphas must contain at least one value")
    if not np.isfinite(alpha_values).all() or np.any(alpha_values < 0.0) or np.any(alpha_values > 1.0):
        raise ValueError("alphas must be finite and within [0, 1]")
    if np.any(np.diff(alpha_values) < 0.0):
        raise ValueError("alphas must be monotonically non-decreasing")
    gap_lower, gap_upper = (float(value) for value in target_gap_m)
    if not np.isfinite([gap_lower, gap_upper]).all() or gap_lower > gap_upper:
        raise ValueError("target_gap_m must be a finite ordered pair")
    preload_cap = float(distal_preload_cap_m)
    if not np.isfinite(preload_cap) or preload_cap < 0.0:
        raise ValueError("distal_preload_cap_m must be finite and non-negative")

    distal_flat = _flatten_geom_ids(distal_geom_ids)
    nondistal_flat = _flatten_geom_ids(active_nondistal_geom_ids)
    forbidden_flat = _flatten_geom_ids(forbidden_geom_ids)
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm <= np.finfo(np.float64).eps:
        raise ValueError("closure sweep requires non-zero gravity")
    up = -gravity / gravity_norm

    saved_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    saved_ctrl = np.asarray(data.ctrl, dtype=np.float64).copy()
    samples: list[ClosureSweepSample] = []
    try:
        for alpha in alpha_values:
            data.qpos[addresses] = opened + float(alpha) * (closed - opened)
            mujoco.mj_forward(model, data)
            witnesses = tuple(
                nearest_distal_target_witness(
                    model,
                    data,
                    cube_geom_id=int(cube_geom_id),
                    distal_geom_ids=distal_geom_ids[finger],
                    target_face=face,
                    distance_max_m=distance_max_m,
                    thresholds=thresholds,
                )
                for finger, face in zip(fingers, faces)
            )
            gaps = np.asarray(
                [witness.signed_distance_m if witness is not None else np.inf for witness in witnesses],
                dtype=np.float64,
            )
            reached = np.isfinite(gaps) & (gaps >= gap_lower - _NUMERIC_TOLERANCE) & (
                gaps <= gap_upper + _NUMERIC_TOLERANCE
            )
            height_valid = np.asarray(
                [witness is not None for witness in witnesses], dtype=bool
            )
            heights = np.asarray(
                [
                    float(witness.cube_point_world_m @ up)
                    if witness is not None
                    else 0.0
                    for witness in witnesses
                ],
                dtype=np.float64,
            )
            height_spread = (
                float(np.ptp(heights)) if bool(np.all(height_valid)) else float("inf")
            )
            samples.append(
                ClosureSweepSample(
                    alpha=float(alpha),
                    target_witnesses=witnesses,
                    target_gap_reached=reached,
                    target_signed_gap_m=gaps,
                    target_height_m=heights,
                    target_height_valid=height_valid,
                    target_height_spread_m=height_spread,
                    max_distal_preload_m=_maximum_cube_penetration_m(
                        model, data, int(cube_geom_id), distal_flat, distance_max_m
                    ),
                    max_active_nondistal_penetration_m=_maximum_cube_penetration_m(
                        model, data, int(cube_geom_id), nondistal_flat, distance_max_m
                    ),
                    max_forbidden_penetration_m=_maximum_cube_penetration_m(
                        model, data, int(cube_geom_id), forbidden_flat, distance_max_m
                    ),
                )
            )
    finally:
        data.qpos[:] = saved_qpos
        data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(model, data)

    return ClosureSweepResult(
        samples=tuple(samples),
        target_gap_m=(gap_lower, gap_upper),
        distal_preload_cap_m=preload_cap,
    )


__all__ = [
    "DEFAULT_GEOM_DISTANCE_MAX_M",
    "DEFAULT_STATIC_DISTAL_PRELOAD_CAP_M",
    "DEFAULT_STATIC_TARGET_GAP_M",
    "ClosureSweepResult",
    "ClosureSweepSample",
    "GeomDistanceWitness",
    "NearestTaxelAssignment",
    "active_nondistal_collision_geom_ids",
    "distal_collision_geom_ids",
    "geom_distance_witness",
    "nearest_distal_target_witness",
    "nearest_taxel_assignment",
    "scan_distal_closure",
]
