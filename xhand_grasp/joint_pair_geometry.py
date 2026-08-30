"""Viewer-independent geometry for measuring a pair of MuJoCo joints.

The tuning and Viewer paths both need the same definition of whether the
line between two joint anchors is parallel to a cube axis.  Keeping that
calculation here prevents a search-time metric from depending on interactive
Viewer code and, more importantly, gives both paths identical ordering and
degenerate-line semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import mujoco
import numpy as np


DEFAULT_MINIMUM_JOINT_PAIR_SEPARATION_M = 0.010
"""Minimum index-to-middle anchor separation used by the v15 contract."""


@dataclass(frozen=True)
class JointPairBinding:
    """Resolved joint anchors and the body defining the local cube frame."""

    first_joint_name: str
    second_joint_name: str
    first_joint_id: int
    second_joint_id: int
    cube_body_id: int


def resolve_joint_pair(
    model: mujoco.MjModel,
    requested: Sequence[str] | None,
) -> JointPairBinding | None:
    """Resolve two distinct one-axis joints and the task cube body.

    ``None`` disables the diagnostic.  The error strings retain the Viewer
    option name because this function is also the public implementation of
    ``--show-joint-pair`` validation.
    """

    if requested is None:
        return None
    names = tuple(str(value) for value in requested)
    if len(names) != 2:
        raise ValueError("--show-joint-pair requires exactly two joint names")
    if not all(names) or names[0] == names[1]:
        raise ValueError("--show-joint-pair requires two distinct joint names")

    joint_ids: list[int] = []
    for name in names:
        try:
            joint_id = int(model.joint(name).id)
        except KeyError as exc:
            raise ValueError(
                f"--show-joint-pair names unknown joint {name!r}"
            ) from exc
        if int(model.jnt_type[joint_id]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise ValueError(
                f"--show-joint-pair joint {name!r} has no single display axis"
            )
        joint_ids.append(joint_id)

    try:
        cube_body_id = int(model.body("three_finger_cube").id)
    except KeyError as exc:
        raise ValueError(
            "--show-joint-pair requires the three_finger_cube body"
        ) from exc
    return JointPairBinding(
        first_joint_name=names[0],
        second_joint_name=names[1],
        first_joint_id=joint_ids[0],
        second_joint_id=joint_ids[1],
        cube_body_id=cube_body_id,
    )


def measure_joint_pair_geometry(
    first_anchor_world_m: Sequence[float] | np.ndarray,
    second_anchor_world_m: Sequence[float] | np.ndarray,
    cube_rotation_world: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, Any]:
    """Measure an ordered anchor vector in the cube-local frame.

    The returned angle is the *undirected* line angle to the cube Y axis.
    Reversing the joint order therefore negates ``vector_cube_m`` while
    preserving ``angle_to_cube_y_deg``.  A zero-length or non-finite line is
    rejected rather than being assigned an arbitrary angle.
    """

    first = np.asarray(first_anchor_world_m, dtype=np.float64)
    second = np.asarray(second_anchor_world_m, dtype=np.float64)
    cube_rotation = np.asarray(cube_rotation_world, dtype=np.float64)
    if first.shape != (3,) or second.shape != (3,):
        raise ValueError("joint-pair anchors must be three-dimensional")
    if cube_rotation.shape != (3, 3):
        raise ValueError("cube rotation must be a 3x3 matrix")

    vector_world = second - first
    vector_cube = cube_rotation.T @ vector_world
    length = float(np.linalg.norm(vector_cube))
    if (
        not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or not np.isfinite(cube_rotation).all()
        or not math.isfinite(length)
        or length <= 1e-12
    ):
        raise ValueError("joint-pair anchors do not define a finite line")
    parallel_error = math.degrees(
        math.acos(float(np.clip(abs(vector_cube[1]) / length, -1.0, 1.0)))
    )
    return {
        "first_anchor_world_m": first.copy(),
        "second_anchor_world_m": second.copy(),
        "vector_cube_m": vector_cube.copy(),
        "length_m": length,
        "angle_to_cube_y_deg": parallel_error,
    }


def joint_pair_telemetry(
    data: mujoco.MjData,
    binding: JointPairBinding,
) -> dict[str, Any]:
    """Measure a resolved pair using the current MuJoCo state."""

    geometry = measure_joint_pair_geometry(
        data.xanchor[binding.first_joint_id],
        data.xanchor[binding.second_joint_id],
        np.asarray(data.xmat[binding.cube_body_id], dtype=np.float64).reshape(3, 3),
    )
    return {
        "first_joint": binding.first_joint_name,
        "second_joint": binding.second_joint_name,
        **geometry,
    }


def joint_pair_signed_residual(
    vector_cube_m: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the differentiable residual ``(v_x / v_y, v_z / v_y)``.

    This residual is intentionally *directed*: the ordered vector must point
    toward cube-local ``+Y``.  Keeping direction validation here, rather than
    hiding it behind an absolute value, prevents a reversed index/middle pair
    from being accepted as aligned.  The returned array owns its data and is
    read-only so it can safely be embedded in hash-addressed planner evidence.
    """

    vector = np.asarray(vector_cube_m, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("joint-pair cube-local vector must contain three finite values")
    if float(vector[1]) <= 0.0:
        raise ValueError("ordered joint-pair vector must point toward cube-local +Y")
    residual = np.asarray(
        (float(vector[0] / vector[1]), float(vector[2] / vector[1])),
        dtype=np.float64,
    )
    if not np.isfinite(residual).all():
        raise ValueError("joint-pair signed residual is non-finite")
    residual.setflags(write=False)
    return residual


def joint_pair_angle_from_signed_residual_deg(
    residual: Sequence[float] | np.ndarray,
) -> float:
    """Convert a signed two-dimensional residual to its ``+Y`` angle."""

    value = np.asarray(residual, dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("joint-pair signed residual must contain two finite values")
    return math.degrees(math.atan(float(np.linalg.norm(value))))


def measure_oriented_joint_pair_geometry(
    index_anchor_world_m: Sequence[float] | np.ndarray,
    middle_anchor_world_m: Sequence[float] | np.ndarray,
    cube_rotation_world: Sequence[Sequence[float]] | np.ndarray,
    *,
    minimum_separation_m: float = DEFAULT_MINIMUM_JOINT_PAIR_SEPARATION_M,
) -> dict[str, Any]:
    """Measure the ordered index-to-middle vector required by schema v15.

    The legacy :func:`measure_joint_pair_geometry` deliberately treats the
    line as undirected for Viewer compatibility.  This new entry point layers
    the v15 manipulation contract on top without changing that numeric path:
    the vector is ``index -> middle`` in the cube frame, must point toward
    ``+Y``, must be at least ``minimum_separation_m`` long, and exposes the
    differentiable residual ``(v_x / v_y, v_z / v_y)``.
    """

    minimum = float(minimum_separation_m)
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("minimum_separation_m must be positive and finite")
    legacy = measure_joint_pair_geometry(
        index_anchor_world_m,
        middle_anchor_world_m,
        cube_rotation_world,
    )
    length = float(legacy["length_m"])
    if length + 1e-12 < minimum:
        raise ValueError(
            "ordered joint-pair anchors are below the minimum separation"
        )
    vector_cube = np.asarray(legacy["vector_cube_m"], dtype=np.float64)
    residual = joint_pair_signed_residual(vector_cube)
    angle = joint_pair_angle_from_signed_residual_deg(residual)
    vector_world = np.asarray(middle_anchor_world_m, dtype=np.float64) - np.asarray(
        index_anchor_world_m, dtype=np.float64
    )
    vector_world = vector_world.copy()
    vector_world.setflags(write=False)
    return {
        **legacy,
        "vector_world_m": vector_world,
        "ordered_from": "index",
        "ordered_to": "middle",
        "required_axis": "+Y",
        "points_toward_positive_cube_y": True,
        "minimum_separation_m": minimum,
        "minimum_separation_satisfied": True,
        "joint_pair_signed_residual": residual,
        "angle_to_cube_positive_y_deg": angle,
    }


def oriented_joint_pair_telemetry(
    data: mujoco.MjData,
    binding: JointPairBinding,
    *,
    minimum_separation_m: float = DEFAULT_MINIMUM_JOINT_PAIR_SEPARATION_M,
) -> dict[str, Any]:
    """Measure a resolved ordered pair with the strict schema-v15 semantics."""

    geometry = measure_oriented_joint_pair_geometry(
        data.xanchor[binding.first_joint_id],
        data.xanchor[binding.second_joint_id],
        np.asarray(data.xmat[binding.cube_body_id], dtype=np.float64).reshape(3, 3),
        minimum_separation_m=minimum_separation_m,
    )
    return {
        "first_joint": binding.first_joint_name,
        "second_joint": binding.second_joint_name,
        **geometry,
    }


# Descriptive alias retained for callers that name the measurement after the
# residual rather than the orientation contract.
measure_signed_joint_pair_geometry = measure_oriented_joint_pair_geometry


__all__ = [
    "DEFAULT_MINIMUM_JOINT_PAIR_SEPARATION_M",
    "JointPairBinding",
    "joint_pair_angle_from_signed_residual_deg",
    "joint_pair_signed_residual",
    "joint_pair_telemetry",
    "measure_joint_pair_geometry",
    "measure_oriented_joint_pair_geometry",
    "measure_signed_joint_pair_geometry",
    "oriented_joint_pair_telemetry",
    "resolve_joint_pair",
]
