"""Orientation-aware actual-contact search in the cube frame.

This module owns the geometry shared by the schema-v11 static search and by
viewer parameter overrides.  A proposal contains exactly thirteen variables:
the seven active joints other than thumb bend, a cube-frame hand-root
translation residual, and a hand-local rotation vector.  Thumb bend and the
cube world pose are immutable within one search stratum.

The clockwise orbit convention is intentionally explicit.  Looking down the
cube-local ``+Z`` axis, a positive requested orbit is the mathematical
rotation ``Rz(-angle)``.  It is applied to both cube-to-hand translation and
cube-to-hand orientation before the local wrist residual is right-multiplied::

    t_CH' = Rz(-angle) t_CH + delta_t_C
    R_CH' = Rz(-angle) R_CH Exp(delta_omega_H)

Static DLS evaluations reuse :func:`evaluate_direct_actual_contact_pose`.
Consequently a result remains a geometry proposal only: a free-cube dynamic
acquisition and measured-qpos finalization are still mandatory before it can
be published as grasp evidence.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import mujoco
import numpy as np

from ..config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, resolved_pose_constraint_values
from ..relative_wrist_pose import (
    clockwise_orbit_rotation,
    rotation_matrix_to_rpy_degrees,
    rotvec_degrees_to_rotation_matrix,
    transform_relative_wrist_pose,
)
from ..scene import build_model, rpy_degrees_to_rotation_matrix


THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
NON_THUMB_ACTUATORS = tuple(
    name for name in ACTIVE_ACTUATORS if name != THUMB_BEND_ACTUATOR
)
VARIABLE_NAMES = (
    *NON_THUMB_ACTUATORS,
    "root_delta_cube_m.x",
    "root_delta_cube_m.y",
    "root_delta_cube_m.z",
    "wrist_local_rotvec_rad.x",
    "wrist_local_rotvec_rad.y",
    "wrist_local_rotvec_rad.z",
)
VARIABLE_COUNT = 13
MEASUREMENT_NAMES = (
    "thumb_gap_m",
    "index_gap_m",
    "middle_gap_m",
    "index_minus_thumb_height_m",
    "middle_minus_thumb_height_m",
    "thumb_normal_alignment",
    "index_normal_alignment",
    "middle_normal_alignment",
)

_AXES = ("x", "y", "z")
_TOLERANCE = 1e-12
_TARGET_GAP_M = 0.00015


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain {length} finite values") from error
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return result.copy()


def _closed_range(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an ordered pair")
    result = tuple(_finite(item, label) for item in value)
    if len(result) != 2 or result[0] > result[1]:
        raise ValueError(f"{label} must be an ordered pair")
    return result


def _axis_ranges(value: Any, label: str) -> Mapping[str, tuple[float, float]]:
    if not isinstance(value, Mapping) or set(value) != set(_AXES):
        raise ValueError(f"{label} must contain exactly x, y and z")
    return MappingProxyType(
        {axis: _closed_range(value[axis], f"{label}.{axis}") for axis in _AXES}
    )


def _strictly_increasing_nonnegative(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a non-empty sequence")
    result = tuple(_finite(item, label) for item in value)
    if (
        not result
        or result[0] != 0.0
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(
            f"{label} must start at zero and be unique, increasing and non-negative"
        )
    return result


@dataclass(frozen=True, slots=True)
class RelativeWristPoseSearchPolicy:
    """Versioned cube-frame search envelope used by schema-v11."""

    clockwise_orbit_deg: tuple[float, ...]
    root_delta_cube_m: Mapping[str, tuple[float, float]]
    wrist_local_rotvec_deg: Mapping[str, tuple[float, float]]
    max_wrist_local_rotvec_norm_deg: float
    root_cube_distance_m: tuple[float, float]

    def __post_init__(self) -> None:
        orbit = _strictly_increasing_nonnegative(
            self.clockwise_orbit_deg, "clockwise_orbit_deg"
        )
        translation = _axis_ranges(self.root_delta_cube_m, "root_delta_cube_m")
        rotation = _axis_ranges(
            self.wrist_local_rotvec_deg, "wrist_local_rotvec_deg"
        )
        rotation_norm = _finite(
            self.max_wrist_local_rotvec_norm_deg,
            "max_wrist_local_rotvec_norm_deg",
        )
        distance = _closed_range(self.root_cube_distance_m, "root_cube_distance_m")
        if rotation_norm <= 0.0 or rotation_norm > 180.0:
            raise ValueError(
                "max_wrist_local_rotvec_norm_deg must lie within (0, 180]"
            )
        if distance[0] <= 0.0 or math.isclose(distance[0], distance[1]):
            raise ValueError("root_cube_distance_m must be a non-empty positive range")
        if any(not low < 0.0 < high for low, high in translation.values()):
            raise ValueError("root_delta_cube_m must straddle zero on every axis")
        if any(not low < 0.0 < high for low, high in rotation.values()):
            raise ValueError(
                "wrist_local_rotvec_deg must straddle zero on every axis"
            )
        largest_rotation_axis = max(
            max(abs(low), abs(high)) for low, high in rotation.values()
        )
        if rotation_norm + _TOLERANCE < largest_rotation_axis:
            raise ValueError(
                "max_wrist_local_rotvec_norm_deg must admit every axis bound"
            )
        object.__setattr__(self, "clockwise_orbit_deg", orbit)
        object.__setattr__(self, "root_delta_cube_m", translation)
        object.__setattr__(self, "wrist_local_rotvec_deg", rotation)
        object.__setattr__(self, "max_wrist_local_rotvec_norm_deg", rotation_norm)
        object.__setattr__(self, "root_cube_distance_m", distance)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RelativeWristPoseSearchPolicy":
        raw = config.get("relative_wrist_pose_search")
        if not isinstance(raw, Mapping):
            raise ValueError("config has no relative_wrist_pose_search policy")
        return cls(
            clockwise_orbit_deg=tuple(raw.get("clockwise_orbit_deg", ())),
            root_delta_cube_m=raw.get("root_delta_cube_m", {}),
            wrist_local_rotvec_deg=raw.get("wrist_local_rotvec_deg", {}),
            max_wrist_local_rotvec_norm_deg=raw.get(
                "max_wrist_local_rotvec_norm_deg"
            ),
            root_cube_distance_m=tuple(raw.get("root_cube_distance_m", ())),
        )

    def as_config(self) -> dict[str, Any]:
        return {
            "clockwise_orbit_deg": list(self.clockwise_orbit_deg),
            "root_delta_cube_m": {
                axis: list(self.root_delta_cube_m[axis]) for axis in _AXES
            },
            "wrist_local_rotvec_deg": {
                axis: list(self.wrist_local_rotvec_deg[axis]) for axis in _AXES
            },
            "max_wrist_local_rotvec_norm_deg": (
                self.max_wrist_local_rotvec_norm_deg
            ),
            "root_cube_distance_m": list(self.root_cube_distance_m),
        }


@dataclass(frozen=True, slots=True)
class RelativeWristSearchStratum:
    """One edge, actual-thumb and clockwise-orbit search stratum."""

    stratum_index: int
    edge_m: float
    thumb_actual_center_rad: float
    clockwise_orbit_deg: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stratum_index, int)
            or isinstance(self.stratum_index, bool)
            or self.stratum_index < 0
        ):
            raise ValueError("stratum_index must be a non-negative integer")
        edge = _finite(self.edge_m, "edge_m")
        thumb = _finite(self.thumb_actual_center_rad, "thumb_actual_center_rad")
        orbit = _finite(self.clockwise_orbit_deg, "clockwise_orbit_deg")
        if edge <= 0.0 or thumb <= 0.0 or orbit < 0.0:
            raise ValueError("edge/thumb must be positive and orbit non-negative")
        object.__setattr__(self, "edge_m", edge)
        object.__setattr__(self, "thumb_actual_center_rad", thumb)
        object.__setattr__(self, "clockwise_orbit_deg", orbit)

    @property
    def stratum_id(self) -> str:
        orbit = f"{self.clockwise_orbit_deg:g}".replace(".", "p")
        return (
            f"edge_{self.edge_m * 1000.0:.0f}mm_"
            f"thumb_actual_{self.thumb_actual_center_rad:.2f}rad_"
            f"clockwise_{orbit}deg"
        )

    # Actual-contact orchestration historically calls its search partitions
    # cells.  These aliases let a v11 stratum replace ActualContactSearchCell
    # without an adapter or an ambiguous second index.
    @property
    def cell_index(self) -> int:
        return self.stratum_index

    @property
    def cell_id(self) -> str:
        return self.stratum_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "stratum_index": self.stratum_index,
            "stratum_id": self.stratum_id,
            "cell_index": self.cell_index,
            "cell_id": self.cell_id,
            "edge_m": self.edge_m,
            "thumb_actual_center_rad": self.thumb_actual_center_rad,
            "clockwise_orbit_deg": self.clockwise_orbit_deg,
        }


def build_relative_wrist_strata(
    edges_m: Sequence[float],
    thumb_actual_centers_rad: Sequence[float],
    clockwise_orbit_deg: Sequence[float],
) -> tuple[RelativeWristSearchStratum, ...]:
    """Return deterministic edge-major, thumb-major, orbit-major strata."""

    edges = tuple(_finite(value, "edges_m") for value in edges_m)
    thumbs = tuple(
        _finite(value, "thumb_actual_centers_rad")
        for value in thumb_actual_centers_rad
    )
    orbits = _strictly_increasing_nonnegative(
        clockwise_orbit_deg, "clockwise_orbit_deg"
    )
    if (
        not edges
        or edges != tuple(sorted(set(edges)))
        or any(value <= 0.0 for value in edges)
    ):
        raise ValueError("edges_m must be positive, unique and increasing")
    if (
        not thumbs
        or thumbs != tuple(sorted(set(thumbs)))
        or any(value <= 0.0 for value in thumbs)
    ):
        raise ValueError(
            "thumb_actual_centers_rad must be positive, unique and increasing"
        )
    return tuple(
        RelativeWristSearchStratum(index, edge, thumb, orbit)
        for index, (edge, thumb, orbit) in enumerate(
            (edge, thumb, orbit)
            for edge in edges
            for thumb in thumbs
            for orbit in orbits
        )
    )


def rotation_vector_to_matrix(rotvec_rad: Sequence[float]) -> np.ndarray:
    """SO(3) exponential map for a radian-valued local rotation vector."""

    vector = _finite_vector(rotvec_rad, 3, "rotvec_rad")
    return rotvec_degrees_to_rotation_matrix(np.degrees(vector))


def clockwise_z_rotation_matrix(clockwise_deg: float) -> np.ndarray:
    """Return ``Rz(-angle)`` for the documented top-view convention."""

    angle = _finite(clockwise_deg, "clockwise_deg")
    if angle < 0.0:
        raise ValueError("clockwise_deg must be non-negative")
    return clockwise_orbit_rotation(angle)


def _cube_world_pose(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    # Keep the support-placement convention identical to the actual-contact
    # evaluator instead of maintaining another half-extent implementation.
    from .actual_contact_grasp_pose import _cube_world_position

    position = _cube_world_position(config)
    rotation = rpy_degrees_to_rotation_matrix(
        config["cube"].get("rpy_deg", (0.0, 0.0, 0.0))
    )
    return position, rotation


@dataclass(frozen=True, slots=True)
class RelativeWristTransform:
    hand_translation_world_m: tuple[float, float, float]
    hand_rotation_world_from_root: tuple[tuple[float, float, float], ...]
    cube_to_hand_translation_cube_m: tuple[float, float, float]
    cube_to_hand_rotation: tuple[tuple[float, float, float], ...]
    root_cube_distance_m: float


def apply_relative_wrist_transform(
    config: Mapping[str, Any],
    *,
    clockwise_orbit_deg: float,
    root_delta_cube_m: Sequence[float] = (0.0, 0.0, 0.0),
    wrist_local_rotvec_deg: Sequence[float] = (0.0, 0.0, 0.0),
) -> RelativeWristTransform:
    """Apply one orbit and 6D residual to a base hand/cube relation."""

    delta = _finite_vector(root_delta_cube_m, 3, "root_delta_cube_m")
    rotvec_deg = _finite_vector(
        wrist_local_rotvec_deg, 3, "wrist_local_rotvec_deg"
    )
    cube_position, cube_rotation = _cube_world_pose(config)
    hand_position = _finite_vector(
        config["hand_pose"]["translation_m"], 3, "hand_pose.translation_m"
    )
    hand_rotation = rpy_degrees_to_rotation_matrix(
        config["hand_pose"]["rpy_deg"]
    )
    shared = transform_relative_wrist_pose(
        source_cube_world_position_m=cube_position,
        source_cube_world_rotation=cube_rotation,
        source_root_world_position_m=hand_position,
        source_root_world_rotation=hand_rotation,
        clockwise_orbit_deg=clockwise_orbit_deg,
        root_delta_cube_m=delta,
        wrist_local_rotvec_deg=rotvec_deg,
    )
    return RelativeWristTransform(
        hand_translation_world_m=shared.root_world_position_m,
        hand_rotation_world_from_root=shared.root_world_rotation,
        cube_to_hand_translation_cube_m=shared.root_in_cube_m,
        cube_to_hand_rotation=shared.cube_from_root_rotation,
        root_cube_distance_m=shared.diagnostics.result_root_cube_distance_m,
    )


@dataclass(frozen=True, slots=True)
class RelativeWristVariables:
    """The fixed-order thirteen-variable optimization vector."""

    non_thumb_joint_qpos_rad: tuple[float, ...]
    root_delta_cube_m: tuple[float, float, float]
    wrist_local_rotvec_rad: tuple[float, float, float]

    def __post_init__(self) -> None:
        joints = _finite_vector(
            self.non_thumb_joint_qpos_rad,
            len(NON_THUMB_ACTUATORS),
            "non_thumb_joint_qpos_rad",
        )
        translation = _finite_vector(
            self.root_delta_cube_m, 3, "root_delta_cube_m"
        )
        rotation = _finite_vector(
            self.wrist_local_rotvec_rad, 3, "wrist_local_rotvec_rad"
        )
        object.__setattr__(
            self, "non_thumb_joint_qpos_rad", tuple(float(value) for value in joints)
        )
        object.__setattr__(
            self, "root_delta_cube_m", tuple(float(value) for value in translation)
        )
        object.__setattr__(
            self,
            "wrist_local_rotvec_rad",
            tuple(float(value) for value in rotation),
        )

    def as_array(self) -> np.ndarray:
        result = np.asarray(
            (
                *self.non_thumb_joint_qpos_rad,
                *self.root_delta_cube_m,
                *self.wrist_local_rotvec_rad,
            ),
            dtype=np.float64,
        )
        assert result.shape == (VARIABLE_COUNT,)
        return result

    @classmethod
    def from_array(cls, value: Sequence[float]) -> "RelativeWristVariables":
        vector = _finite_vector(value, VARIABLE_COUNT, "relative wrist variables")
        return cls(
            tuple(float(item) for item in vector[:7]),
            tuple(float(item) for item in vector[7:10]),
            tuple(float(item) for item in vector[10:13]),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RelativeWristVariables":
        nominal = config["grasp_pose"]["nominal_joint_qpos_rad"]
        relative = (
            config.get("candidate_metadata", {}).get(
                "relative_wrist_pose_search", {}
            )
            if isinstance(config.get("candidate_metadata"), Mapping)
            else {}
        )
        root_delta = relative.get("root_delta_cube_m", (0.0, 0.0, 0.0))
        rotvec_deg = relative.get(
            "wrist_local_rotvec_deg", (0.0, 0.0, 0.0)
        )
        return cls(
            tuple(float(nominal[name]) for name in NON_THUMB_ACTUATORS),
            tuple(float(value) for value in root_delta),
            tuple(float(value) for value in np.radians(rotvec_deg)),
        )


def materialize_relative_wrist_candidate(
    base_config: Mapping[str, Any],
    variables: RelativeWristVariables,
    *,
    clockwise_orbit_deg: float,
    synchronize_preload: bool = True,
) -> dict[str, Any]:
    """Create a proposal without changing thumb bend or the cube world pose."""

    policy = RelativeWristPoseSearchPolicy.from_config(base_config)
    result = copy.deepcopy(dict(base_config))
    source = copy.deepcopy(dict(base_config))
    existing_relative = (
        base_config.get("candidate_metadata", {}).get(
            "relative_wrist_pose_search", {}
        )
        if isinstance(base_config.get("candidate_metadata"), Mapping)
        else {}
    )
    anchor_hand_pose = existing_relative.get("anchor_hand_pose")
    if anchor_hand_pose is not None:
        if (
            not isinstance(anchor_hand_pose, Mapping)
            or set(anchor_hand_pose) != {"translation_m", "rpy_deg"}
        ):
            raise ValueError(
                "relative_wrist_pose_search.anchor_hand_pose must contain "
                "translation_m and rpy_deg"
            )
        source["hand_pose"] = {
            "translation_m": _finite_vector(
                anchor_hand_pose["translation_m"], 3, "anchor_hand_pose.translation_m"
            ).tolist(),
            "rpy_deg": _finite_vector(
                anchor_hand_pose["rpy_deg"], 3, "anchor_hand_pose.rpy_deg"
            ).tolist(),
        }
    else:
        anchor_hand_pose = copy.deepcopy(dict(base_config["hand_pose"]))
    transform = apply_relative_wrist_transform(
        source,
        clockwise_orbit_deg=clockwise_orbit_deg,
        root_delta_cube_m=variables.root_delta_cube_m,
        wrist_local_rotvec_deg=np.degrees(variables.wrist_local_rotvec_rad),
    )
    result["hand_pose"]["translation_m"] = list(
        transform.hand_translation_world_m
    )
    result["hand_pose"]["rpy_deg"] = rotation_matrix_to_rpy_degrees(
        transform.hand_rotation_world_from_root,
        reference_rpy_deg=base_config["hand_pose"]["rpy_deg"],
    ).tolist()
    nominal = result["grasp_pose"]["nominal_joint_qpos_rad"]
    thumb_before = float(nominal[THUMB_BEND_ACTUATOR])
    for index, name in enumerate(NON_THUMB_ACTUATORS):
        nominal[name] = float(variables.non_thumb_joint_qpos_rad[index])
    if not math.isclose(
        float(nominal[THUMB_BEND_ACTUATOR]), thumb_before, abs_tol=_TOLERANCE
    ):
        raise AssertionError("relative wrist materialization changed thumb bend")
    if synchronize_preload:
        preload = result.get("control", {}).get("contact_preload_targets_rad")
        if isinstance(preload, Mapping):
            result["control"]["contact_preload_targets_rad"] = {
                name: float(nominal[name]) for name in ACTIVE_ACTUATORS
            }
    metadata = result.setdefault("candidate_metadata", {})
    persisted_relative = copy.deepcopy(dict(existing_relative))
    persisted_relative.update({
        "clockwise_orbit_deg": float(clockwise_orbit_deg),
        "anchor_hand_pose": copy.deepcopy(dict(anchor_hand_pose)),
        "root_delta_cube_m": list(variables.root_delta_cube_m),
        "wrist_local_rotvec_deg": np.degrees(
            variables.wrist_local_rotvec_rad
        ).tolist(),
        "cube_to_hand_translation_cube_m": list(
            transform.cube_to_hand_translation_cube_m
        ),
        "cube_to_hand_rotation": [
            list(row) for row in transform.cube_to_hand_rotation
        ],
        "root_cube_distance_m": transform.root_cube_distance_m,
        "cube_pose_sampled": False,
        "hand_root_fixed_during_simulation": True,
        "policy": policy.as_config(),
    })
    metadata["relative_wrist_pose_search"] = persisted_relative
    return result


def model_joint_bounds(
    model: mujoco.MjModel,
    config: Mapping[str, Any],
) -> Mapping[str, tuple[float, float]]:
    """Intersect registered actuator bounds with true MuJoCo joint limits."""

    from ..experiment import resolve_experiment

    definition = resolve_experiment(dict(config))
    result: dict[str, tuple[float, float]] = {}
    for name in NON_THUMB_ACTUATORS:
        low, high = definition.search_bounds.actuator_targets_rad[name]
        actuator_id = model.actuator(name).id
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if bool(model.jnt_limited[joint_id]):
            low = max(float(low), float(model.jnt_range[joint_id, 0]))
            high = min(float(high), float(model.jnt_range[joint_id, 1]))
        if low > high:
            raise ValueError(f"empty registered/model joint intersection for {name}")
        result[name] = (float(low), float(high))
    return MappingProxyType(result)


def relative_wrist_boundary_violations(
    config: Mapping[str, Any],
    variables: RelativeWristVariables,
    policy: RelativeWristPoseSearchPolicy,
    *,
    clockwise_orbit_deg: float,
    joint_bounds: Mapping[str, tuple[float, float]],
    check_pose_constraints: bool = True,
) -> tuple[str, ...]:
    """Return stable reason codes; line search rejects any non-empty result."""

    reasons: list[str] = []
    if not any(
        math.isclose(clockwise_orbit_deg, value, abs_tol=_TOLERANCE)
        for value in policy.clockwise_orbit_deg
    ):
        reasons.append("clockwise_orbit_out_of_strata")
    for index, name in enumerate(NON_THUMB_ACTUATORS):
        if name not in joint_bounds:
            reasons.append(f"missing_joint_bound:{name}")
            continue
        value = variables.non_thumb_joint_qpos_rad[index]
        low, high = joint_bounds[name]
        if not low - _TOLERANCE <= value <= high + _TOLERANCE:
            reasons.append(f"joint_out_of_bounds:{name}")
    for index, axis in enumerate(_AXES):
        value = variables.root_delta_cube_m[index]
        low, high = policy.root_delta_cube_m[axis]
        if not low - _TOLERANCE <= value <= high + _TOLERANCE:
            reasons.append(f"root_delta_out_of_bounds:{axis}")
    rotvec_deg = np.degrees(variables.wrist_local_rotvec_rad)
    for index, axis in enumerate(_AXES):
        low, high = policy.wrist_local_rotvec_deg[axis]
        if not low - _TOLERANCE <= rotvec_deg[index] <= high + _TOLERANCE:
            reasons.append(f"wrist_rotvec_out_of_bounds:{axis}")
    if (
        float(np.linalg.norm(rotvec_deg))
        > policy.max_wrist_local_rotvec_norm_deg + _TOLERANCE
    ):
        reasons.append("wrist_rotvec_norm_out_of_bounds")
    candidate = materialize_relative_wrist_candidate(
        config,
        variables,
        clockwise_orbit_deg=clockwise_orbit_deg,
        synchronize_preload=False,
    )
    distance = float(
        candidate["candidate_metadata"]["relative_wrist_pose_search"][
            "root_cube_distance_m"
        ]
    )
    if not (
        policy.root_cube_distance_m[0] - _TOLERANCE
        <= distance
        <= policy.root_cube_distance_m[1] + _TOLERANCE
    ):
        reasons.append("root_cube_distance_out_of_bounds")
    if check_pose_constraints and isinstance(config.get("pose_constraints"), Mapping):
        pose = resolved_pose_constraint_values(candidate)
        registered = config["pose_constraints"]
        for key in (
            "finger_down_tilt_deg",
            "palm_plane_ground_angle_deg",
        ):
            if key in registered:
                low, high = _closed_range(registered[key], f"pose_constraints.{key}")
                if not low - _TOLERANCE <= float(pose[key]) <= high + _TOLERANCE:
                    reasons.append(f"pose_constraint_out_of_bounds:{key}")
        cube_in_root_bounds = registered.get("cube_position_in_root_m")
        if isinstance(cube_in_root_bounds, Mapping):
            cube_in_root = _finite_vector(
                pose["cube_position_in_root_m"],
                3,
                "resolved cube_position_in_root_m",
            )
            for index, axis in enumerate(_AXES):
                if axis not in cube_in_root_bounds:
                    reasons.append(
                        f"missing_pose_constraint:cube_position_in_root_m.{axis}"
                    )
                    continue
                low, high = _closed_range(
                    cube_in_root_bounds[axis],
                    f"pose_constraints.cube_position_in_root_m.{axis}",
                )
                if not (
                    low - _TOLERANCE
                    <= cube_in_root[index]
                    <= high + _TOLERANCE
                ):
                    reasons.append(
                        "pose_constraint_out_of_bounds:"
                        f"cube_position_in_root_m.{axis}"
                    )
        # Configuration validation also owns Euler roll/yaw bounds.  They are
        # representation-level constraints distinct from the physical palm
        # and finger angles above, so reject them here instead of generating
        # candidates that can never be promoted.
        from ..experiment import resolve_experiment

        search_bounds = resolve_experiment(dict(candidate)).search_bounds
        hand_rpy = _finite_vector(
            candidate["hand_pose"]["rpy_deg"], 3, "hand_pose.rpy_deg"
        )
        for value, (low, high), label in (
            (hand_rpy[0], search_bounds.hand_roll_deg, "hand_roll_deg"),
            (hand_rpy[2], search_bounds.hand_yaw_deg, "hand_yaw_deg"),
        ):
            if not low - _TOLERANCE <= value <= high + _TOLERANCE:
                reasons.append(f"search_bound_out_of_bounds:{label}")
        if not search_bounds.contains_cube_position(
            pose["cube_position_in_root_m"]
        ):
            reasons.append("search_bound_out_of_bounds:cube_position_in_root_m")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class RelativeWristTrialEvaluation:
    """Normalized evaluator output accepted by the injectable DLS core."""

    static_result: Any
    measurement: tuple[float, ...] | None
    safety_violations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.measurement is not None:
            measurement = _finite_vector(
                self.measurement, len(MEASUREMENT_NAMES), "measurement"
            )
            object.__setattr__(
                self, "measurement", tuple(float(value) for value in measurement)
            )
        object.__setattr__(
            self,
            "safety_violations",
            tuple(str(value) for value in self.safety_violations),
        )

    @property
    def safe(self) -> bool:
        return self.measurement is not None and not self.safety_violations


def actual_contact_trial_evaluation(
    result: Any,
    up_world: Sequence[float],
    *,
    maximum_penetration_m: float = 0.002,
    minimum_normal_alignment: float = 0.95,
    require_v11_safety_evidence: bool = False,
) -> RelativeWristTrialEvaluation:
    """Extract the DLS measurement and hard line-search safety gates.

    Schema v11 makes the extended collision proof mandatory.  Keeping the
    requirement explicit preserves the v9/v10 evaluator contract while the
    production v11 factory can fail closed on stale or synthetic evidence.
    """

    up = _finite_vector(up_world, 3, "up_world")
    norm = float(np.linalg.norm(up))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("up_world must be non-zero")
    up /= norm
    reasons: list[str] = []
    raw_witnesses = getattr(result, "target_witnesses", None)
    if raw_witnesses is None:
        reasons.append("missing_distal_witness_evidence")
        return RelativeWristTrialEvaluation(result, None, tuple(reasons))
    witnesses = tuple(raw_witnesses)
    if len(witnesses) != len(ACTIVE_FINGERS) or any(
        witness is None for witness in witnesses
    ):
        reasons.append("missing_distal_witness")
        return RelativeWristTrialEvaluation(result, None, tuple(reasons))
    resolved = tuple(witness for witness in witnesses if witness is not None)
    gaps = tuple(float(value.signed_gap_m) for value in resolved)
    heights = tuple(
        float(up @ np.asarray(value.cube_point_world_m, dtype=np.float64))
        for value in resolved
    )
    alignments = tuple(float(value.normal_alignment) for value in resolved)
    for finger, witness, gap, alignment in zip(
        ACTIVE_FINGERS, resolved, gaps, alignments
    ):
        if str(witness.finger) != finger:
            reasons.append(f"wrong_target_finger:{finger}")
        if gap < -maximum_penetration_m - _TOLERANCE:
            reasons.append(f"target_penetration_over_limit:{finger}")
        if alignment < minimum_normal_alignment - _TOLERANCE:
            reasons.append(f"target_normal_misaligned:{finger}")
    off_target_count = getattr(result, "off_target_distal_penetrating_count", None)
    if off_target_count is None:
        reasons.append("missing_off_target_distal_evidence")
    elif int(off_target_count) != 0:
        reasons.append("off_target_distal_penetration")
    active_nondistal_gap = getattr(
        result, "minimum_active_nondistal_gap_m", None
    )
    if active_nondistal_gap is None or not math.isfinite(
        float(active_nondistal_gap)
    ):
        reasons.append("missing_active_nondistal_gap_evidence")
    elif float(active_nondistal_gap) < -_TOLERANCE:
        reasons.append("active_nondistal_penetration")
    if getattr(result, "cube_freejoint_qpos_unchanged", None) is not True:
        reasons.append("cube_freejoint_changed")
    if require_v11_safety_evidence:
        forbidden_gap = getattr(
            result, "nominal_minimum_forbidden_hand_gap_m", None
        )
        if forbidden_gap is None or not math.isfinite(float(forbidden_gap)):
            reasons.append("missing_nominal_forbidden_hand_gap_evidence")
        elif float(forbidden_gap) < -_TOLERANCE:
            reasons.append("nominal_forbidden_hand_penetration")

        all_distal_penetration = getattr(
            result, "nominal_maximum_all_distal_penetration_m", None
        )
        if all_distal_penetration is None or not math.isfinite(
            float(all_distal_penetration)
        ):
            reasons.append("missing_nominal_all_distal_penetration_evidence")
        elif (
            float(all_distal_penetration)
            > float(maximum_penetration_m) + _TOLERANCE
        ):
            reasons.append("all_distal_penetration_over_limit")

        if getattr(result, "precontact_geometry_evaluated", None) is not True:
            # ``precontact_minimum_hand_gap_m`` has historically used a
            # finite negative sentinel when geometry was not reached.  The
            # explicit evaluated bit prevents that sentinel being mistaken
            # for a real query (or vice versa).
            reasons.append("precontact_geometry_not_evaluated")
        else:
            precontact_gap = getattr(
                result, "precontact_minimum_hand_gap_m", None
            )
            if precontact_gap is None or not math.isfinite(
                float(precontact_gap)
            ):
                reasons.append("missing_precontact_hand_gap_evidence")
            elif float(precontact_gap) < -_TOLERANCE:
                reasons.append("precontact_hand_penetration")
    measurement = (
        *gaps,
        heights[1] - heights[0],
        heights[2] - heights[0],
        *alignments,
    )
    return RelativeWristTrialEvaluation(result, measurement, tuple(reasons))


TrialEvaluator = Callable[[Mapping[str, Any]], RelativeWristTrialEvaluation]


def build_actual_contact_trial_evaluator(
    base_config: Mapping[str, Any],
) -> tuple[TrialEvaluator, Mapping[str, tuple[float, float]]]:
    """Compile once and expose the existing real-contact evaluator to DLS."""

    from .actual_contact_grasp_pose import (
        ActualContactStaticThresholds,
        evaluate_direct_actual_contact_pose,
    )

    model, info = build_model(copy.deepcopy(dict(base_config)))
    data = mujoco.MjData(model)
    thresholds = ActualContactStaticThresholds.from_config(base_config)
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm <= np.finfo(np.float64).eps:
        raise ValueError("orientation-aware DLS requires non-zero gravity")
    up = -gravity / gravity_norm

    def evaluate(config: Mapping[str, Any]) -> RelativeWristTrialEvaluation:
        result = evaluate_direct_actual_contact_pose(
            model,
            data,
            info,
            config,
            thresholds=thresholds,
        )
        return actual_contact_trial_evaluation(
            result,
            up,
            maximum_penetration_m=0.002,
            minimum_normal_alignment=thresholds.minimum_normal_alignment,
            require_v11_safety_evidence=(
                int(base_config.get("schema_version", 0)) == 11
            ),
        )

    return evaluate, model_joint_bounds(model, base_config)


@dataclass(frozen=True, slots=True)
class RelativeWristDLSSettings:
    maximum_iterations: int = 6
    damping: float = 0.025
    regularization_weight: float = 0.02
    contact_tolerance: float = 1e-3
    joint_finite_difference_rad: float = 2e-4
    translation_finite_difference_m: float = 2e-5
    rotation_finite_difference_rad: float = 2e-4
    joint_step_rad: float = 0.05
    translation_step_m: float = 0.002
    rotation_step_rad: float = math.radians(1.0)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.maximum_iterations, int)
            or isinstance(self.maximum_iterations, bool)
            or self.maximum_iterations <= 0
        ):
            raise ValueError("maximum_iterations must be a positive integer")
        for name in (
            "damping",
            "regularization_weight",
            "contact_tolerance",
            "joint_finite_difference_rad",
            "translation_finite_difference_m",
            "rotation_finite_difference_rad",
            "joint_step_rad",
            "translation_step_m",
            "rotation_step_rad",
        ):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class RelativeWristDLSResult:
    config: dict[str, Any]
    variables: RelativeWristVariables
    static_result: Any
    diagnostics: dict[str, Any]
    stop_reason: str


def _contact_target_and_scale() -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(
        (_TARGET_GAP_M,) * 3 + (0.0, 0.0) + (1.0,) * 3,
        dtype=np.float64,
    )
    scale = np.asarray((0.0005,) * 3 + (0.005,) * 2 + (0.05,) * 3)
    return target, scale


def _objective(
    measurement: np.ndarray,
    variables: np.ndarray,
    reference: np.ndarray,
    variable_scale: np.ndarray,
    settings: RelativeWristDLSSettings,
) -> tuple[float, float, float]:
    target, measurement_scale = _contact_target_and_scale()
    contact = float(np.linalg.norm((target - measurement) / measurement_scale))
    regularization = float(
        math.sqrt(settings.regularization_weight)
        * np.linalg.norm((variables - reference) / variable_scale)
    )
    return math.hypot(contact, regularization), contact, regularization


def solve_orientation_aware_dls(
    base_config: Mapping[str, Any],
    *,
    clockwise_orbit_deg: float,
    initial_variables: RelativeWristVariables | None = None,
    settings: RelativeWristDLSSettings | None = None,
    evaluator: TrialEvaluator | None = None,
    joint_bounds: Mapping[str, tuple[float, float]] | None = None,
    check_pose_constraints: bool = True,
) -> RelativeWristDLSResult:
    """Optimize the thirteen variables with hard-boundary line search.

    A caller may inject a pure evaluator for deterministic unit tests.  The
    production default compiles a MuJoCo model once and delegates every trial
    to the existing real collision-witness evaluator.
    """

    policy = RelativeWristPoseSearchPolicy.from_config(base_config)
    if not any(
        math.isclose(clockwise_orbit_deg, value, abs_tol=_TOLERANCE)
        for value in policy.clockwise_orbit_deg
    ):
        raise ValueError("clockwise_orbit_deg is not a registered stratum")
    resolved_settings = settings or RelativeWristDLSSettings()
    if evaluator is None:
        evaluator, model_bounds = build_actual_contact_trial_evaluator(base_config)
        if joint_bounds is None:
            joint_bounds = model_bounds
    if joint_bounds is None:
        raise ValueError("an injected evaluator requires explicit joint_bounds")
    if set(joint_bounds) != set(NON_THUMB_ACTUATORS):
        raise ValueError("joint_bounds must contain the seven non-thumb actuators")

    current_variables = initial_variables or RelativeWristVariables.from_config(
        base_config
    )
    current = current_variables.as_array()
    reference = current.copy()
    finite_difference = np.asarray(
        (*([resolved_settings.joint_finite_difference_rad] * 7),)
        + (*([resolved_settings.translation_finite_difference_m] * 3),)
        + (*([resolved_settings.rotation_finite_difference_rad] * 3),),
        dtype=np.float64,
    )
    variable_scale = np.asarray(
        (*([resolved_settings.joint_step_rad] * 7),)
        + (*([resolved_settings.translation_step_m] * 3),)
        + (*([resolved_settings.rotation_step_rad] * 3),),
        dtype=np.float64,
    )

    def boundary(value: np.ndarray) -> tuple[str, ...]:
        return relative_wrist_boundary_violations(
            base_config,
            RelativeWristVariables.from_array(value),
            policy,
            clockwise_orbit_deg=clockwise_orbit_deg,
            joint_bounds=joint_bounds,
            check_pose_constraints=check_pose_constraints,
        )

    def evaluate(value: np.ndarray) -> tuple[dict[str, Any], RelativeWristTrialEvaluation]:
        candidate = materialize_relative_wrist_candidate(
            base_config,
            RelativeWristVariables.from_array(value),
            clockwise_orbit_deg=clockwise_orbit_deg,
        )
        return candidate, evaluator(candidate)

    initial_boundary = boundary(current)
    if initial_boundary:
        raise ValueError(
            "initial relative-wrist variables violate boundaries: "
            + ", ".join(initial_boundary)
        )
    current_config, current_evaluation = evaluate(current)
    if current_evaluation.measurement is None:
        return RelativeWristDLSResult(
            current_config,
            RelativeWristVariables.from_array(current),
            current_evaluation.static_result,
            {
                "method": "orientation_aware_actual_contact_dls",
                "variable_names": list(VARIABLE_NAMES),
                "measurement_names": list(MEASUREMENT_NAMES),
                "clockwise_orbit_deg": float(clockwise_orbit_deg),
                "iterations": [],
                "initial_safety_violations": list(
                    current_evaluation.safety_violations
                ),
            },
            "missing_initial_distal_witness",
        )
    measurement = np.asarray(current_evaluation.measurement, dtype=np.float64)
    initial_measurement = measurement.copy()
    initial_safety = current_evaluation.safety_violations
    iteration_records: list[dict[str, Any]] = []
    stop_reason = "maximum_iterations"
    target, measurement_scale = _contact_target_and_scale()

    for iteration in range(resolved_settings.maximum_iterations):
        objective_before, contact_before, regularization_before = _objective(
            measurement,
            current,
            reference,
            variable_scale,
            resolved_settings,
        )
        if contact_before <= resolved_settings.contact_tolerance:
            stop_reason = "contact_target_converged"
            break
        jacobian = np.zeros((len(MEASUREMENT_NAMES), VARIABLE_COUNT))
        finite_difference_rejections: list[dict[str, Any]] = []
        for column in range(VARIABLE_COUNT):
            selected: tuple[np.ndarray, RelativeWristTrialEvaluation, float] | None = None
            for direction in (1.0, -1.0):
                trial = current.copy()
                trial[column] += direction * finite_difference[column]
                boundary_reasons = boundary(trial)
                if boundary_reasons:
                    finite_difference_rejections.append(
                        {
                            "variable": VARIABLE_NAMES[column],
                            "direction": direction,
                            "reasons": list(boundary_reasons),
                        }
                    )
                    continue
                _, evaluation = evaluate(trial)
                if not evaluation.safe:
                    finite_difference_rejections.append(
                        {
                            "variable": VARIABLE_NAMES[column],
                            "direction": direction,
                            "reasons": list(evaluation.safety_violations)
                            or ["unsafe_trial_evaluation"],
                        }
                    )
                    continue
                selected = (
                    np.asarray(evaluation.measurement, dtype=np.float64),
                    evaluation,
                    direction * finite_difference[column],
                )
                break
            if selected is not None:
                trial_measurement, _, actual_step = selected
                jacobian[:, column] = (
                    (trial_measurement - measurement) / actual_step
                ) / measurement_scale

        normalized_residual = (target - measurement) / measurement_scale
        scaled_jacobian = jacobian * variable_scale[np.newaxis, :]
        regularization_rows = (
            math.sqrt(resolved_settings.regularization_weight)
            * np.eye(VARIABLE_COUNT)
        )
        regularization_residual = (
            -math.sqrt(resolved_settings.regularization_weight)
            * (current - reference)
            / variable_scale
        )
        system = np.vstack((scaled_jacobian, regularization_rows))
        rhs = np.concatenate((normalized_residual, regularization_residual))
        normal = system.T @ system + (
            resolved_settings.damping**2 * np.eye(VARIABLE_COUNT)
        )
        unit_step = np.linalg.solve(normal, system.T @ rhs)
        maximum_unit_step = float(np.max(np.abs(unit_step)))
        if maximum_unit_step > 1.0:
            unit_step /= maximum_unit_step
        proposed_step = variable_scale * unit_step

        accepted: tuple[
            np.ndarray,
            dict[str, Any],
            RelativeWristTrialEvaluation,
            np.ndarray,
            float,
            float,
            float,
            float,
        ] | None = None
        line_search: list[dict[str, Any]] = []
        for line_scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = current + line_scale * proposed_step
            boundary_reasons = boundary(trial)
            if boundary_reasons:
                line_search.append(
                    {
                        "scale": line_scale,
                        "accepted": False,
                        "reasons": list(boundary_reasons),
                    }
                )
                continue
            trial_config, trial_evaluation = evaluate(trial)
            if not trial_evaluation.safe:
                line_search.append(
                    {
                        "scale": line_scale,
                        "accepted": False,
                        "reasons": list(trial_evaluation.safety_violations),
                    }
                )
                continue
            trial_measurement = np.asarray(
                trial_evaluation.measurement, dtype=np.float64
            )
            objective_after, contact_after, regularization_after = _objective(
                trial_measurement,
                trial,
                reference,
                variable_scale,
                resolved_settings,
            )
            improved = objective_after + 1e-12 < objective_before
            line_search.append(
                {
                    "scale": line_scale,
                    "accepted": improved,
                    "reasons": [] if improved else ["objective_not_improved"],
                    "objective": objective_after,
                }
            )
            if improved:
                accepted = (
                    trial,
                    trial_config,
                    trial_evaluation,
                    trial_measurement,
                    line_scale,
                    objective_after,
                    contact_after,
                    regularization_after,
                )
                break
        iteration_records.append(
            {
                "iteration": iteration,
                "objective_before": objective_before,
                "contact_objective_before": contact_before,
                "regularization_before": regularization_before,
                "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
                "finite_difference_rejections": finite_difference_rejections,
                "line_search": line_search,
                "accepted": accepted is not None,
            }
        )
        if accepted is None:
            stop_reason = "dls_no_safe_improvement"
            break
        (
            current,
            current_config,
            current_evaluation,
            measurement,
            accepted_scale,
            objective_after,
            contact_after,
            regularization_after,
        ) = accepted
        iteration_records[-1].update(
            {
                "accepted_scale": accepted_scale,
                "objective_after": objective_after,
                "contact_objective_after": contact_after,
                "regularization_after": regularization_after,
            }
        )

    final_variables = RelativeWristVariables.from_array(current)
    final_boundary = boundary(current)
    assert not final_boundary
    final_config, final_evaluation = evaluate(current)
    if bool(getattr(final_evaluation.static_result, "static_geometry_pass", False)):
        from .actual_contact_grasp_pose import apply_precontact_solution

        final_config = apply_precontact_solution(
            final_config, final_evaluation.static_result
        )
    final_objective, final_contact, final_regularization = _objective(
        np.asarray(final_evaluation.measurement, dtype=np.float64),
        current,
        reference,
        variable_scale,
        resolved_settings,
    )
    diagnostics = {
        "method": "orientation_aware_actual_contact_dls",
        "variable_names": list(VARIABLE_NAMES),
        "measurement_names": list(MEASUREMENT_NAMES),
        "clockwise_orbit_deg": float(clockwise_orbit_deg),
        "fixed_variable": THUMB_BEND_ACTUATOR,
        "fixed_thumb_actual_rad": float(
            base_config["grasp_pose"]["nominal_joint_qpos_rad"][
                THUMB_BEND_ACTUATOR
            ]
        ),
        "initial_variables": reference.tolist(),
        "final_variables": current.tolist(),
        "final_root_delta_cube_m": list(final_variables.root_delta_cube_m),
        "final_wrist_local_rotvec_deg": np.degrees(
            final_variables.wrist_local_rotvec_rad
        ).tolist(),
        "initial_measurement": initial_measurement.tolist(),
        "final_measurement": list(final_evaluation.measurement),
        "initial_safety_violations": list(initial_safety),
        "final_safety_violations": list(final_evaluation.safety_violations),
        "final_objective": final_objective,
        "final_contact_objective": final_contact,
        "final_regularization": final_regularization,
        "final_boundary_violations": list(final_boundary),
        "iterations": iteration_records,
        "boundary_rejection_count": sum(
            bool(trial["reasons"])
            and any("bounds" in reason for reason in trial["reasons"])
            for iteration in iteration_records
            for trial in iteration["line_search"]
        ),
        "safety_rejection_count": sum(
            bool(trial["reasons"])
            and any(
                "penetration" in reason
                or "witness" in reason
                or "freejoint" in reason
                for reason in trial["reasons"]
            )
            for iteration in iteration_records
            for trial in iteration["line_search"]
        ),
    }
    return RelativeWristDLSResult(
        final_config,
        final_variables,
        final_evaluation.static_result,
        diagnostics,
        stop_reason,
    )


def _stable_rank_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_stable_rank_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_stable_rank_value(item) for item in value)
    return value


def default_relative_wrist_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Rank static records without depending on worker completion order."""

    if "static_rank" in record:
        static_rank = _stable_rank_value(record["static_rank"])
    else:
        metrics = record.get("static_metrics", {})
        static_rank = (
            not bool(record.get("static_pass", False)),
            int(metrics.get("missing_target_witness_count", 99)),
            int(metrics.get("off_target_distal_penetrating_count", 99)),
            max(
                0.0,
                -float(metrics.get("minimum_active_nondistal_gap_m", -1.0)),
            ),
            float(metrics.get("contact_height_spread_m", 1.0)),
        )
    return (
        static_rank,
        int(record["candidate_id"]),
    )


@dataclass(frozen=True, slots=True)
class PerEdgeQuotaSelection:
    records: tuple[dict[str, Any], ...]
    per_edge_selected_count: Mapping[float, int]
    per_edge_orbit_coverage_deg: Mapping[float, tuple[float, ...]]
    deficient_edges_m: tuple[float, ...]
    quota_satisfied: bool


def retain_per_edge_orbit_quota(
    records: Sequence[Mapping[str, Any]],
    *,
    per_edge: int,
    rank_key: Callable[[Mapping[str, Any]], tuple[Any, ...]] = (
        default_relative_wrist_candidate_rank
    ),
) -> PerEdgeQuotaSelection:
    """Select per-edge quotas while covering every feasible orbit first.

    Each non-empty orbit bucket contributes its best record before remaining
    slots are filled globally by rank.  Candidate IDs are an unconditional
    final tie-break, so shuffling worker results cannot change selection.
    """

    if not isinstance(per_edge, int) or isinstance(per_edge, bool) or per_edge <= 0:
        raise ValueError("per_edge must be a positive integer")
    grouped: dict[float, dict[float, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen_candidates: set[int] = set()
    for raw in records:
        record = copy.deepcopy(dict(raw))
        candidate_id = int(record["candidate_id"])
        if candidate_id in seen_candidates:
            raise ValueError(f"duplicate candidate_id {candidate_id}")
        seen_candidates.add(candidate_id)
        edge = _finite(record["edge_m"], "edge_m")
        orbit = _finite(record["clockwise_orbit_deg"], "clockwise_orbit_deg")
        grouped[edge][orbit].append(record)

    selected: list[dict[str, Any]] = []
    counts: dict[float, int] = {}
    coverage: dict[float, tuple[float, ...]] = {}
    deficient: list[float] = []
    for edge in sorted(grouped):
        buckets = grouped[edge]
        for values in buckets.values():
            values.sort(key=lambda value: (rank_key(value), int(value["candidate_id"])))
        first_by_orbit = sorted(
            (values[0] for values in buckets.values()),
            key=lambda value: (rank_key(value), int(value["candidate_id"])),
        )
        edge_selected = first_by_orbit[:per_edge]
        selected_ids = {int(value["candidate_id"]) for value in edge_selected}
        remaining = sorted(
            (
                value
                for values in buckets.values()
                for value in values
                if int(value["candidate_id"]) not in selected_ids
            ),
            key=lambda value: (rank_key(value), int(value["candidate_id"])),
        )
        edge_selected.extend(remaining[: max(0, per_edge - len(edge_selected))])
        edge_selected.sort(
            key=lambda value: (rank_key(value), int(value["candidate_id"]))
        )
        selected.extend(edge_selected)
        counts[edge] = len(edge_selected)
        coverage[edge] = tuple(
            sorted({float(value["clockwise_orbit_deg"]) for value in edge_selected})
        )
        if len(edge_selected) < per_edge:
            deficient.append(edge)
    return PerEdgeQuotaSelection(
        records=tuple(selected),
        per_edge_selected_count=MappingProxyType(counts),
        per_edge_orbit_coverage_deg=MappingProxyType(coverage),
        deficient_edges_m=tuple(deficient),
        quota_satisfied=not deficient,
    )


__all__ = [
    "MEASUREMENT_NAMES",
    "NON_THUMB_ACTUATORS",
    "THUMB_BEND_ACTUATOR",
    "VARIABLE_COUNT",
    "VARIABLE_NAMES",
    "PerEdgeQuotaSelection",
    "RelativeWristDLSResult",
    "RelativeWristDLSSettings",
    "RelativeWristPoseSearchPolicy",
    "RelativeWristSearchStratum",
    "RelativeWristTransform",
    "RelativeWristTrialEvaluation",
    "RelativeWristVariables",
    "actual_contact_trial_evaluation",
    "apply_relative_wrist_transform",
    "build_actual_contact_trial_evaluator",
    "build_relative_wrist_strata",
    "clockwise_z_rotation_matrix",
    "default_relative_wrist_candidate_rank",
    "materialize_relative_wrist_candidate",
    "model_joint_bounds",
    "relative_wrist_boundary_violations",
    "retain_per_edge_orbit_quota",
    "rotation_vector_to_matrix",
    "solve_orientation_aware_dls",
]
