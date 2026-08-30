"""Closure-direction diagnostics for real distal/cube contact witnesses.

The helpers in this module deliberately do not choose or re-orient a contact
normal.  A caller that reads an ``mjContact`` must first supply the cube-
outward world normal (for example, by using :func:`contacts.surface_witness`).
The desired finger-closing direction is then the negative of that normal.

The numerical layer is pure NumPy.  ``point_jacobian_command_velocity`` is a
small, read-only MuJoCo adapter which maps an actuator-target derivative into
the world velocity of a collision witness.  Keeping the two layers separate
allows static search, batch simulation and offline tests to share exactly the
same alignment semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray


_NUMERIC_EPSILON = 1e-12


def _vector3(values: ArrayLike, label: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain three finite values")
    return vector.copy()


def _readonly(values: ArrayLike, *, dtype: type = np.float64) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ClosureAlignmentSample:
    """Force-aggregated closure evidence for one finger in one frame.

    ``valid`` means the force, normal and non-zero command velocity make the
    alignment *measurable*.  It intentionally does not mean that the command
    points inward: an outward/tangential measurement remains valid so an
    evaluator cannot hide it by filtering on the validity mask.  Invalid
    samples use finite, conservative scalar sentinels (cosine ``-1`` and angle
    ``180 deg``), keeping complete NPZ/JSON-derived metric pipelines free of
    accidental NaN propagation.
    """

    total_normal_force_n: float
    command_velocity_world_m_s: NDArray[np.float64]
    cube_outward_normal_world: NDArray[np.float64]
    cosine: float
    angle_deg: float
    inward_speed_m_s: float
    tangent_speed_m_s: float
    valid: bool

    def __post_init__(self) -> None:
        force = float(self.total_normal_force_n)
        if not math.isfinite(force) or force < 0.0:
            raise ValueError("total_normal_force_n must be finite and non-negative")
        object.__setattr__(self, "total_normal_force_n", force)
        object.__setattr__(
            self,
            "command_velocity_world_m_s",
            _readonly(
                _vector3(
                    self.command_velocity_world_m_s,
                    "command_velocity_world_m_s",
                )
            ),
        )
        object.__setattr__(
            self,
            "cube_outward_normal_world",
            _readonly(
                _vector3(
                    self.cube_outward_normal_world,
                    "cube_outward_normal_world",
                )
            ),
        )
        for field in (
            "cosine",
            "angle_deg",
            "inward_speed_m_s",
            "tangent_speed_m_s",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
            object.__setattr__(self, field, value)
        if not -1.0 - _NUMERIC_EPSILON <= self.cosine <= 1.0 + _NUMERIC_EPSILON:
            raise ValueError("cosine must be within [-1, 1]")
        if not 0.0 <= self.angle_deg <= 180.0 + _NUMERIC_EPSILON:
            raise ValueError("angle_deg must be within [0, 180]")
        if self.tangent_speed_m_s < 0.0:
            raise ValueError("tangent_speed_m_s must be non-negative")
        if not isinstance(self.valid, (bool, np.bool_)):
            raise ValueError("valid must be boolean")
        object.__setattr__(self, "valid", bool(self.valid))


def invalid_closure_alignment(
    *,
    total_normal_force_n: float = 0.0,
    command_velocity_world_m_s: ArrayLike = (0.0, 0.0, 0.0),
    cube_outward_normal_world: ArrayLike = (0.0, 0.0, 0.0),
) -> ClosureAlignmentSample:
    """Return one finite worst-case sample with ``valid=False``."""

    return ClosureAlignmentSample(
        total_normal_force_n=float(total_normal_force_n),
        command_velocity_world_m_s=_vector3(
            command_velocity_world_m_s, "command_velocity_world_m_s"
        ),
        cube_outward_normal_world=_vector3(
            cube_outward_normal_world, "cube_outward_normal_world"
        ),
        cosine=-1.0,
        angle_deg=180.0,
        inward_speed_m_s=0.0,
        tangent_speed_m_s=0.0,
        valid=False,
    )


def closure_alignment_from_velocity(
    command_velocity_world_m_s: ArrayLike,
    cube_outward_normal_world: ArrayLike,
    *,
    normal_force_n: float = 1.0,
    minimum_normal_force_n: float = 0.0,
    speed_epsilon_m_s: float = _NUMERIC_EPSILON,
) -> ClosureAlignmentSample:
    """Evaluate one command velocity against one cube-outward normal.

    ``valid`` requires sufficient normal force plus non-zero command speed and
    normal.  Directional acceptance is intentionally separate: callers must
    require ``inward_speed_m_s > 0`` (or a configured positive threshold).
    Keeping outward/tangential samples measurable prevents them from being
    silently excluded from p95 and minimum-inward calculations.
    """

    velocity = _vector3(command_velocity_world_m_s, "command_velocity_world_m_s")
    outward = _vector3(cube_outward_normal_world, "cube_outward_normal_world")
    force = float(normal_force_n)
    minimum_force = float(minimum_normal_force_n)
    speed_epsilon = float(speed_epsilon_m_s)
    if not math.isfinite(force) or force < 0.0:
        raise ValueError("normal_force_n must be finite and non-negative")
    if not math.isfinite(minimum_force) or minimum_force < 0.0:
        raise ValueError("minimum_normal_force_n must be finite and non-negative")
    if not math.isfinite(speed_epsilon) or speed_epsilon <= 0.0:
        raise ValueError("speed_epsilon_m_s must be positive and finite")

    outward_norm = float(np.linalg.norm(outward))
    velocity_norm = float(np.linalg.norm(velocity))
    if (
        outward_norm <= speed_epsilon
        or velocity_norm <= speed_epsilon
        or force <= _NUMERIC_EPSILON
        or force + _NUMERIC_EPSILON < minimum_force
    ):
        return invalid_closure_alignment(
            total_normal_force_n=force,
            command_velocity_world_m_s=velocity,
            cube_outward_normal_world=(
                outward / outward_norm
                if outward_norm > speed_epsilon
                else np.zeros(3, dtype=np.float64)
            ),
        )

    outward_unit = outward / outward_norm
    desired_inward = -outward_unit
    inward_speed = float(velocity @ desired_inward)
    tangent_velocity = velocity - inward_speed * desired_inward
    tangent_speed = float(np.linalg.norm(tangent_velocity))
    cosine = float(
        np.clip(velocity @ desired_inward / velocity_norm, -1.0, 1.0)
    )
    angle = float(math.degrees(math.acos(cosine)))
    return ClosureAlignmentSample(
        total_normal_force_n=force,
        command_velocity_world_m_s=velocity,
        cube_outward_normal_world=outward_unit,
        cosine=cosine,
        angle_deg=angle,
        inward_speed_m_s=inward_speed,
        tangent_speed_m_s=tangent_speed,
        valid=True,
    )


def aggregate_force_weighted_closure_contacts(
    normal_force_n: ArrayLike,
    command_velocity_world_m_s: ArrayLike,
    cube_outward_normal_world: ArrayLike,
    *,
    minimum_total_force_n: float = 0.0,
    speed_epsilon_m_s: float = _NUMERIC_EPSILON,
) -> ClosureAlignmentSample:
    """Aggregate any number of same-finger contacts and evaluate alignment.

    Contact velocities and *unit* cube-outward normals are weighted by their
    non-negative normal forces.  Zero-force entries do not influence the
    result.  Supplying an empty collection is valid and returns an explicit
    invalid sample, which is useful for fixed-shape per-frame traces.
    """

    force = np.asarray(normal_force_n, dtype=np.float64)
    velocity = np.asarray(command_velocity_world_m_s, dtype=np.float64)
    outward = np.asarray(cube_outward_normal_world, dtype=np.float64)
    if force.ndim != 1:
        raise ValueError("normal_force_n must be one-dimensional")
    expected = (force.shape[0], 3)
    if velocity.shape != expected:
        raise ValueError(
            f"command_velocity_world_m_s must have shape {expected}"
        )
    if outward.shape != expected:
        raise ValueError(f"cube_outward_normal_world must have shape {expected}")
    if (
        not np.isfinite(force).all()
        or not np.isfinite(velocity).all()
        or not np.isfinite(outward).all()
        or np.any(force < 0.0)
    ):
        raise ValueError("contact forces and vectors must be finite and forces non-negative")
    minimum_force = float(minimum_total_force_n)
    if not math.isfinite(minimum_force) or minimum_force < 0.0:
        raise ValueError("minimum_total_force_n must be finite and non-negative")

    positive = force > _NUMERIC_EPSILON
    total_force = float(np.sum(force[positive]))
    if not np.any(positive) or total_force + _NUMERIC_EPSILON < minimum_force:
        return invalid_closure_alignment(total_normal_force_n=total_force)

    positive_normals = outward[positive]
    normal_norms = np.linalg.norm(positive_normals, axis=1)
    if np.any(normal_norms <= float(speed_epsilon_m_s)):
        return invalid_closure_alignment(total_normal_force_n=total_force)
    normal_units = positive_normals / normal_norms[:, np.newaxis]
    weights = force[positive, np.newaxis]
    weighted_velocity = np.sum(weights * velocity[positive], axis=0) / total_force
    weighted_outward = np.sum(weights * normal_units, axis=0) / total_force
    return closure_alignment_from_velocity(
        weighted_velocity,
        weighted_outward,
        normal_force_n=total_force,
        minimum_normal_force_n=minimum_force,
        speed_epsilon_m_s=speed_epsilon_m_s,
    )


def closure_direction_within_limits(
    sample: ClosureAlignmentSample,
    *,
    maximum_angle_deg: float,
    minimum_inward_speed_m_s: float = 0.0,
    require_positive_inward_speed: bool = True,
) -> bool:
    """Apply directional limits without conflating them with measurability.

    A zero configured minimum still means strictly positive motion when
    ``require_positive_inward_speed`` is true.  A positive configured minimum
    is inclusive up to numerical tolerance.  This mirrors the schema-v8 split
    between ``closure_alignment_valid`` and its angle/inward hard checks.
    """

    if not isinstance(sample, ClosureAlignmentSample):
        raise TypeError("sample must be a ClosureAlignmentSample")
    maximum_angle = float(maximum_angle_deg)
    minimum_inward = float(minimum_inward_speed_m_s)
    if not math.isfinite(maximum_angle) or not 0.0 < maximum_angle < 180.0:
        raise ValueError("maximum_angle_deg must be within (0, 180)")
    if not math.isfinite(minimum_inward) or minimum_inward < 0.0:
        raise ValueError(
            "minimum_inward_speed_m_s must be finite and non-negative"
        )
    if not isinstance(require_positive_inward_speed, bool):
        raise ValueError("require_positive_inward_speed must be boolean")
    if not sample.valid or sample.angle_deg > maximum_angle + _NUMERIC_EPSILON:
        return False
    if require_positive_inward_speed and sample.inward_speed_m_s <= 0.0:
        return False
    return bool(
        sample.inward_speed_m_s + _NUMERIC_EPSILON >= minimum_inward
    )


def point_jacobian_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    point_world_m: ArrayLike,
    body_id: int,
    generalized_velocity_rad_s: ArrayLike,
) -> NDArray[np.float64]:
    """Return the world linear velocity of a body-fixed point via ``mj_jac``."""

    point = _vector3(point_world_m, "point_world_m")
    body = int(body_id)
    if isinstance(body_id, bool) or body != body_id or not 0 <= body < model.nbody:
        raise ValueError("body_id is outside the model")
    qvel = np.asarray(generalized_velocity_rad_s, dtype=np.float64)
    if qvel.shape != (model.nv,) or not np.isfinite(qvel).all():
        raise ValueError(
            f"generalized_velocity_rad_s must contain {model.nv} finite values"
        )
    jacobian = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacobian, None, point, body)
    return jacobian @ qvel


def point_jacobian_command_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    point_world_m: ArrayLike,
    body_id: int,
    actuator_dof_addresses: Sequence[int] | NDArray[np.int_],
    actuator_target_velocity_rad_s: ArrayLike,
) -> NDArray[np.float64]:
    """Map actuator-target derivatives to one collision witness velocity.

    XHAND's position actuators each transmit with unit gear to one scalar joint.
    Callers provide that already-resolved actuator-to-DoF address map, avoiding
    any dependence on actuator numerical ordering inside this module.
    """

    addresses = np.asarray(actuator_dof_addresses, dtype=np.int64)
    velocity = np.asarray(actuator_target_velocity_rad_s, dtype=np.float64)
    if addresses.ndim != 1 or velocity.shape != addresses.shape:
        raise ValueError(
            "actuator_dof_addresses and actuator_target_velocity_rad_s must "
            "have the same one-dimensional shape"
        )
    if not np.isfinite(velocity).all():
        raise ValueError("actuator_target_velocity_rad_s must be finite")
    if np.any(addresses < 0) or np.any(addresses >= model.nv):
        raise ValueError("actuator_dof_addresses contains an invalid DoF address")
    if np.unique(addresses).size != addresses.size:
        raise ValueError("actuator_dof_addresses must not contain duplicates")
    generalized_velocity = np.zeros(model.nv, dtype=np.float64)
    generalized_velocity[addresses] = velocity
    return point_jacobian_velocity(
        model,
        data,
        point_world_m,
        body_id,
        generalized_velocity,
    )


__all__ = [
    "ClosureAlignmentSample",
    "aggregate_force_weighted_closure_contacts",
    "closure_alignment_from_velocity",
    "closure_direction_within_limits",
    "invalid_closure_alignment",
    "point_jacobian_command_velocity",
    "point_jacobian_velocity",
]
