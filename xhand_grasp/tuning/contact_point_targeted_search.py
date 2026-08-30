"""Cube-local contact-point planning and point-targeted grasp-pose search.

Schema-v12 first freezes one three-point plan on a 90 mm cube and only then
optimizes the hand.  This module deliberately owns no multiprocessing or
artifact-writing policy.  It provides deterministic, side-effect-free
primitives that a campaign runner can resume and shard without changing the
numerical result:

* a prefix-stable, seeded Halton point-plan generator;
* immutable/hash-authenticated point plans;
* a fourteen-variable (eight actual qpos + six wrist pose) DLS layer using
  the existing real ``mj_geomDistance`` witness evaluator;
* stable ranking and a dynamic-runner-compatible static-record adapter.

The signed orbit convention matches :mod:`xhand_grasp.relative_wrist_pose`:
positive is clockwise when viewed from cube-local ``+Z``.  Negative values
therefore provide the counter-clockwise strata required by schema-v12.
Static geometry is proposal evidence only.  A free-cube dynamic acquisition
and measured-qpos finalization remain mandatory before publication.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import mujoco
import numpy as np

from ..config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, resolved_pose_constraint_values
from ..grasp_pose import canonical_sha256, controller_id, grasp_pose_id
from ..relative_wrist_pose import (
    rotation_matrix_to_rpy_degrees,
    transform_relative_wrist_pose,
)
from ..scene import build_model, rpy_degrees_to_rotation_matrix
from .pose_preserving_seed_campaign import canonical_sha256 as _campaign_sha256


POINT_PLAN_SCHEMA_VERSION = 1
POINT_SEARCH_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
DEFAULT_EDGE_M = 0.090
DEFAULT_SAMPLE_COUNT = 120_000
DEFAULT_RETAIN_POINT_PLAN_COUNT = 256
DEFAULT_RETAIN_STATIC_COUNT = 64
FINGER_ORDER = tuple(ACTIVE_FINGERS)
FACE_LABELS = ("-X", "+X")
REFERENCE_POINT_YZ_M = MappingProxyType(
    {
        "thumb": (0.010, 0.012),
        "index": (0.002, 0.012),
        "mid": (0.021, 0.012),
    }
)
REFERENCE_FACE = MappingProxyType(
    {"thumb": "-X", "index": "+X", "mid": "+X"}
)
SIGNED_ORBIT_DEG = (-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5)

VARIABLE_NAMES = (
    *ACTIVE_ACTUATORS,
    "root_delta_cube_m.x",
    "root_delta_cube_m.y",
    "root_delta_cube_m.z",
    "wrist_local_rotvec_rad.x",
    "wrist_local_rotvec_rad.y",
    "wrist_local_rotvec_rad.z",
)
VARIABLE_COUNT = 14
MEASUREMENT_NAMES = (
    "thumb_gap_m",
    "index_gap_m",
    "middle_gap_m",
    "thumb_y_error_m",
    "thumb_z_error_m",
    "index_y_error_m",
    "index_z_error_m",
    "middle_y_error_m",
    "middle_z_error_m",
    "thumb_normal_alignment",
    "index_normal_alignment",
    "middle_normal_alignment",
)

_AXES = ("x", "y", "z")
_PRIMES = (2, 3, 5, 7, 11, 13)
_TARGET_GAP_M = 0.00015
_TOLERANCE = 1e-12


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive integer") from error
    if result <= 0 or result != value:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _vector(value: Any, length: int, label: str) -> np.ndarray:
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
    values = tuple(_finite(item, label) for item in value)
    if len(values) != 2 or values[0] > values[1]:
        raise ValueError(f"{label} must be an ordered pair")
    return values


def _axis_ranges(value: Any, label: str) -> Mapping[str, tuple[float, float]]:
    if not isinstance(value, Mapping) or set(value) != set(_AXES):
        raise ValueError(f"{label} must contain exactly x, y and z")
    return MappingProxyType(
        {axis: _closed_range(value[axis], f"{label}.{axis}") for axis in _AXES}
    )


def _signed_orbits(value: Any) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("signed_orbit_deg must be a non-empty sequence")
    result = tuple(_finite(item, "signed_orbit_deg") for item in value)
    if not result or result != tuple(sorted(set(result))) or 0.0 not in result:
        raise ValueError(
            "signed_orbit_deg must be unique, increasing and contain zero"
        )
    return result


@dataclass(frozen=True, slots=True)
class CubeFaceContactPoint:
    """One target point represented as an X face plus cube-local ``(y,z)``."""

    face: str
    y_m: float
    z_m: float

    def __post_init__(self) -> None:
        face = str(self.face)
        if face not in FACE_LABELS:
            raise ValueError("contact points must lie on either -X or +X")
        object.__setattr__(self, "face", face)
        object.__setattr__(self, "y_m", _finite(self.y_m, "y_m"))
        object.__setattr__(self, "z_m", _finite(self.z_m, "z_m"))

    @property
    def sign(self) -> float:
        return -1.0 if self.face == "-X" else 1.0

    def local_xyz_m(self, edge_m: float) -> np.ndarray:
        edge = _finite(edge_m, "edge_m")
        if edge <= 0.0:
            raise ValueError("edge_m must be positive")
        return np.asarray((self.sign * 0.5 * edge, self.y_m, self.z_m))

    def as_config(self) -> dict[str, Any]:
        return {"face": self.face, "face_yz_m": [self.y_m, self.z_m]}

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "CubeFaceContactPoint":
        if not isinstance(value, Mapping) or set(value) != {"face", "face_yz_m"}:
            raise ValueError(
                "contact point must contain exactly face and face_yz_m"
            )
        yz = _vector(value["face_yz_m"], 2, "contact point face_yz_m")
        return cls(str(value["face"]), float(yz[0]), float(yz[1]))


def _plan_semantic(
    *,
    edge_m: float,
    target_radius_m: float,
    points: Mapping[str, CubeFaceContactPoint],
) -> dict[str, Any]:
    return {
        "schema_version": POINT_PLAN_SCHEMA_VERSION,
        "cube_edge_m": float(edge_m),
        "target_radius_m": float(target_radius_m),
        "target_points": {
            finger: points[finger].as_config() for finger in FINGER_ORDER
        },
        "frozen": True,
    }


@dataclass(frozen=True, slots=True)
class ContactPointPlan:
    """Immutable, content-addressed three-point plan."""

    edge_m: float
    target_radius_m: float
    points: Mapping[str, CubeFaceContactPoint]

    def __post_init__(self) -> None:
        edge = _finite(self.edge_m, "edge_m")
        radius = _finite(self.target_radius_m, "target_radius_m")
        if edge <= 0.0 or radius <= 0.0:
            raise ValueError("edge_m and target_radius_m must be positive")
        if not isinstance(self.points, Mapping) or set(self.points) != set(FINGER_ORDER):
            raise ValueError("points must contain exactly thumb, index and mid")
        resolved: dict[str, CubeFaceContactPoint] = {}
        for finger in FINGER_ORDER:
            point = self.points[finger]
            if not isinstance(point, CubeFaceContactPoint):
                raise ValueError(f"points.{finger} must be CubeFaceContactPoint")
            expected = REFERENCE_FACE[finger]
            if point.face != expected:
                raise ValueError(f"points.{finger}.face must be {expected}")
            local = point.local_xyz_m(edge)
            if np.any(np.abs(local) > 0.5 * edge + _TOLERANCE):
                raise ValueError(f"points.{finger} lies outside the cube")
            resolved[finger] = point
        object.__setattr__(self, "edge_m", edge)
        object.__setattr__(self, "target_radius_m", radius)
        object.__setattr__(self, "points", MappingProxyType(resolved))

    @property
    def point_plan_id(self) -> str:
        return canonical_sha256(
            _plan_semantic(
                edge_m=self.edge_m,
                target_radius_m=self.target_radius_m,
                points=self.points,
            )
        )

    def as_config(self) -> dict[str, Any]:
        semantic = _plan_semantic(
            edge_m=self.edge_m,
            target_radius_m=self.target_radius_m,
            points=self.points,
        )
        return {
            "schema_version": semantic["schema_version"],
            "point_plan_id": self.point_plan_id,
            "cube_edge_m": semantic["cube_edge_m"],
            "coordinate_frame": "cube_local",
            "target_points": semantic["target_points"],
            "target_points_cube_local_m": {
                finger: self.points[finger].local_xyz_m(self.edge_m).tolist()
                for finger in FINGER_ORDER
            },
            "target_radius_m": semantic["target_radius_m"],
            "frozen": True,
            "identity_binding": "grasp_pose_id_includes_point_plan_id",
        }

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "ContactPointPlan":
        if not isinstance(value, Mapping):
            raise ValueError("contact_point_plan must be a mapping")
        allowed = {
            "schema_version",
            "point_plan_id",
            "cube_edge_m",
            "coordinate_frame",
            "target_points",
            "target_points_cube_local_m",
            "target_radius_m",
            "frozen",
            "identity_binding",
        }
        if set(value) != allowed:
            raise ValueError("contact_point_plan has missing or unknown keys")
        if int(value["schema_version"]) != POINT_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported contact-point plan schema_version")
        if value["coordinate_frame"] != "cube_local":
            raise ValueError("contact_point_plan coordinate_frame must be cube_local")
        if value["identity_binding"] != "grasp_pose_id_includes_point_plan_id":
            raise ValueError("contact_point_plan has the wrong identity binding")
        if value["frozen"] is not True:
            raise ValueError("contact_point_plan must be frozen")
        raw_points = value["target_points"]
        if not isinstance(raw_points, Mapping) or set(raw_points) != set(FINGER_ORDER):
            raise ValueError("contact_point_plan.points must contain three fingers")
        plan = cls(
            edge_m=float(value["cube_edge_m"]),
            target_radius_m=float(value["target_radius_m"]),
            points={
                finger: CubeFaceContactPoint.from_config(raw_points[finger])
                for finger in FINGER_ORDER
            },
        )
        if str(value["point_plan_id"]) != plan.point_plan_id:
            raise ValueError("contact_point_plan point_plan_id does not match content")
        derived = value["target_points_cube_local_m"]
        if not isinstance(derived, Mapping) or set(derived) != set(FINGER_ORDER):
            raise ValueError(
                "target_points_cube_local_m must contain exactly three fingers"
            )
        for finger in FINGER_ORDER:
            persisted = _vector(
                derived[finger],
                3,
                f"target_points_cube_local_m.{finger}",
            )
            expected = plan.points[finger].local_xyz_m(plan.edge_m)
            if not np.allclose(persisted, expected, atol=_TOLERANCE, rtol=0.0):
                raise ValueError(
                    f"target_points_cube_local_m.{finger} is not the derived point"
                )
        return plan


@dataclass(frozen=True, slots=True)
class ContactPointSearchPolicy:
    """Versioned point generation, static gate and 6D wrist envelope."""

    seed: int = DEFAULT_SEED
    sample_count: int = DEFAULT_SAMPLE_COUNT
    retain_point_plan_count: int = DEFAULT_RETAIN_POINT_PLAN_COUNT
    reference_points: Mapping[str, CubeFaceContactPoint] | None = None
    half_width_m: float = 0.008
    minimum_edge_margin_m: float = 0.020
    maximum_height_spread_m: float = 0.005
    minimum_index_middle_separation_m: float = 0.010
    static_target_radius_m: float = 0.004
    target_radius_m: float = 0.002
    minimum_normal_alignment: float = 0.95
    maximum_penetration_m: float = 0.002
    signed_orbit_deg: tuple[float, ...] = SIGNED_ORBIT_DEG
    root_delta_cube_m: Mapping[str, tuple[float, float]] | None = None
    wrist_local_rotvec_deg: Mapping[str, tuple[float, float]] | None = None
    max_wrist_local_rotvec_norm_deg: float = 8.0
    root_cube_distance_m: tuple[float, float] = (0.135, 0.210)
    thumb_actual_range_rad: tuple[float, float] = (1.40, 1.60)
    registered_config: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        seed = int(self.seed)
        if isinstance(self.seed, bool) or seed != self.seed or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        count = _positive_int(self.sample_count, "sample_count")
        retain = _positive_int(
            self.retain_point_plan_count, "retain_point_plan_count"
        )
        if retain > count:
            raise ValueError("retain_point_plan_count cannot exceed sample_count")
        defaults = {
            finger: CubeFaceContactPoint(
                REFERENCE_FACE[finger], *REFERENCE_POINT_YZ_M[finger]
            )
            for finger in FINGER_ORDER
        }
        raw_reference = defaults if self.reference_points is None else self.reference_points
        if not isinstance(raw_reference, Mapping) or set(raw_reference) != set(FINGER_ORDER):
            raise ValueError("reference_points must contain exactly three fingers")
        reference: dict[str, CubeFaceContactPoint] = {}
        for finger in FINGER_ORDER:
            point = raw_reference[finger]
            if not isinstance(point, CubeFaceContactPoint):
                raise ValueError("reference_points must contain CubeFaceContactPoint values")
            if point.face != REFERENCE_FACE[finger]:
                raise ValueError(f"reference_points.{finger} has wrong face")
            reference[finger] = point
        root_defaults = {
            "x": (-0.012, 0.012),
            "y": (-0.012, 0.012),
            "z": (-0.015, 0.015),
        }
        rot_defaults = {axis: (-6.0, 6.0) for axis in _AXES}
        roots = _axis_ranges(
            root_defaults if self.root_delta_cube_m is None else self.root_delta_cube_m,
            "root_delta_cube_m",
        )
        rotations = _axis_ranges(
            rot_defaults
            if self.wrist_local_rotvec_deg is None
            else self.wrist_local_rotvec_deg,
            "wrist_local_rotvec_deg",
        )
        for field in (
            "half_width_m",
            "minimum_edge_margin_m",
            "maximum_height_spread_m",
            "minimum_index_middle_separation_m",
            "static_target_radius_m",
            "target_radius_m",
            "maximum_penetration_m",
        ):
            value = _finite(getattr(self, field), field)
            if value <= 0.0:
                raise ValueError(f"{field} must be positive")
            object.__setattr__(self, field, value)
        alignment = _finite(self.minimum_normal_alignment, "minimum_normal_alignment")
        if not 0.0 < alignment <= 1.0:
            raise ValueError("minimum_normal_alignment must lie within (0, 1]")
        if self.target_radius_m > self.static_target_radius_m:
            raise ValueError("target_radius_m cannot exceed static_target_radius_m")
        norm_limit = _finite(
            self.max_wrist_local_rotvec_norm_deg,
            "max_wrist_local_rotvec_norm_deg",
        )
        if norm_limit <= 0.0:
            raise ValueError("max_wrist_local_rotvec_norm_deg must be positive")
        largest_axis = max(
            max(abs(low), abs(high)) for low, high in rotations.values()
        )
        if norm_limit + _TOLERANCE < largest_axis:
            raise ValueError("rotvec norm limit must admit every axis bound")
        distance = _closed_range(self.root_cube_distance_m, "root_cube_distance_m")
        thumb = _closed_range(self.thumb_actual_range_rad, "thumb_actual_range_rad")
        if distance[0] <= 0.0 or distance[0] == distance[1]:
            raise ValueError("root_cube_distance_m must be a positive range")
        if thumb[0] <= 0.0 or thumb[0] == thumb[1]:
            raise ValueError("thumb_actual_range_rad must be a positive range")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "sample_count", count)
        object.__setattr__(self, "retain_point_plan_count", retain)
        object.__setattr__(self, "reference_points", MappingProxyType(reference))
        object.__setattr__(self, "minimum_normal_alignment", alignment)
        object.__setattr__(self, "signed_orbit_deg", _signed_orbits(self.signed_orbit_deg))
        object.__setattr__(self, "root_delta_cube_m", roots)
        object.__setattr__(self, "wrist_local_rotvec_deg", rotations)
        object.__setattr__(self, "max_wrist_local_rotvec_norm_deg", norm_limit)
        object.__setattr__(self, "root_cube_distance_m", distance)
        object.__setattr__(self, "thumb_actual_range_rad", thumb)
        if self.registered_config is not None:
            if not isinstance(self.registered_config, Mapping):
                raise ValueError("registered_config must be a mapping")
            object.__setattr__(
                self,
                "registered_config",
                MappingProxyType(copy.deepcopy(dict(self.registered_config))),
            )

    def as_config(self) -> dict[str, Any]:
        if self.registered_config is not None:
            return copy.deepcopy(dict(self.registered_config))
        assert self.reference_points is not None
        assert self.root_delta_cube_m is not None
        assert self.wrist_local_rotvec_deg is not None
        return {
            "schema_version": POINT_SEARCH_SCHEMA_VERSION,
            "seed": self.seed,
            "seed_point_plan_id": ContactPointPlan(
                DEFAULT_EDGE_M,
                self.target_radius_m,
                self.reference_points,
            ).point_plan_id,
            "seed_target_points": {
                finger: self.reference_points[finger].as_config()
                for finger in FINGER_ORDER
            },
            "face_yz_delta_m": [-self.half_width_m, self.half_width_m],
            "signed_clockwise_orbit_deg": list(self.signed_orbit_deg),
            "orbit_convention": {
                "axis": "cube_local_+Z",
                "positive_direction": "clockwise_viewed_from_cube_local_+Z",
                "internal_mathematical_angle_sign": -1,
                "translation_and_orientation_orbit_together": True,
            },
            "source_pose_candidate_ids": [
                4367930796723148114,
                4370955631788604117,
            ],
            "constraints": {
                "min_target_edge_margin_m": self.minimum_edge_margin_m,
                "max_target_height_spread_m": self.maximum_height_spread_m,
                "min_index_middle_separation_m": (
                    self.minimum_index_middle_separation_m
                ),
                "static_target_error_radius_m": self.static_target_radius_m,
                "dynamic_target_error_radius_m": self.target_radius_m,
                "min_witness_normal_alignment": self.minimum_normal_alignment,
                "max_penetration_m": self.maximum_penetration_m,
            },
            "optimization": {
                "method": "contact_point_orientation_aware_damped_least_squares",
                "variables": [
                    "eight_active_actual_qpos",
                    "root_translation_cube_frame_xyz",
                    "wrist_local_rotvec_xyz",
                ],
                "precontact_retreat_source": "contact_pose_point_jacobian",
                "cube_world_pose_is_not_sampled": True,
            },
            "budget": {
                "point_group_sample_count": self.sample_count,
                "retained_point_plan_count": self.retain_point_plan_count,
                "max_reachability_screen_count": 3584,
                "retained_static_pose_count": 64,
                "controllers_per_pose": 6,
                "maximum_dynamic_grasp_candidate_count": 384,
                "local_pose_count": 16,
                "local_refine_per_pose": 128,
                "maximum_local_refinement_count": 2048,
                "exact_candidate_count": 16,
                "selected_trajectory_count": 5,
                "perturbations_per_trajectory": 16,
            },
            "manipulability_prefilter": {
                "probe_count": 17,
                "max_active_delta_rad": 0.05,
                "virtual_translation_cube_world_m": [0.0, 0.0, 0.002],
                "success_evidence": False,
                "final_trajectories_require_full_reset_rerun": True,
            },
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ContactPointSearchPolicy":
        raw = config.get("contact_point_search")
        if not isinstance(raw, Mapping):
            raise ValueError("config has no contact_point_search mapping")
        if int(raw.get("schema_version", 0)) != POINT_SEARCH_SCHEMA_VERSION:
            raise ValueError("unsupported contact_point_search schema_version")
        references = raw.get("seed_target_points")
        if not isinstance(references, Mapping) or set(references) != set(FINGER_ORDER):
            raise ValueError("seed_target_points must contain three fingers")
        delta = _closed_range(raw.get("face_yz_delta_m"), "face_yz_delta_m")
        if not math.isclose(abs(delta[0]), abs(delta[1]), abs_tol=_TOLERANCE):
            raise ValueError("point generator requires a symmetric face_yz_delta_m")
        constraints = raw.get("constraints")
        budget = raw.get("budget")
        if not isinstance(constraints, Mapping) or not isinstance(budget, Mapping):
            raise ValueError("contact_point_search requires constraints and budget")
        thumb_range = config.get("grasp_pose", {}).get(
            "thumb_actual_range_rad", (1.40, 1.60)
        )
        pose_constraints = config.get("pose_constraints", {})
        root_distance = pose_constraints.get(
            "root_cube_distance_m", (0.135, 0.210)
        )
        seed_points = {
            finger: CubeFaceContactPoint.from_config(references[finger])
            for finger in FINGER_ORDER
        }
        cube = config.get("cube")
        if not isinstance(cube, Mapping):
            raise ValueError("config.cube must be a mapping")
        seed_plan = ContactPointPlan(
            _finite(cube.get("edge_m"), "cube.edge_m"),
            constraints["dynamic_target_error_radius_m"],
            seed_points,
        )
        if str(raw.get("seed_point_plan_id", "")) != seed_plan.point_plan_id:
            raise ValueError(
                "contact_point_search seed_point_plan_id does not match seed points"
            )
        return cls(
            seed=raw["seed"],
            sample_count=budget["point_group_sample_count"],
            retain_point_plan_count=budget["retained_point_plan_count"],
            reference_points=seed_points,
            half_width_m=abs(delta[1]),
            minimum_edge_margin_m=constraints["min_target_edge_margin_m"],
            maximum_height_spread_m=constraints["max_target_height_spread_m"],
            minimum_index_middle_separation_m=constraints[
                "min_index_middle_separation_m"
            ],
            static_target_radius_m=constraints["static_target_error_radius_m"],
            target_radius_m=constraints["dynamic_target_error_radius_m"],
            minimum_normal_alignment=constraints["min_witness_normal_alignment"],
            maximum_penetration_m=constraints["max_penetration_m"],
            signed_orbit_deg=tuple(raw["signed_clockwise_orbit_deg"]),
            root_cube_distance_m=tuple(root_distance),
            thumb_actual_range_rad=tuple(thumb_range),
            registered_config=raw,
        )


@dataclass(frozen=True, slots=True)
class ContactPointGeometryMetrics:
    minimum_edge_margin_m: float
    height_spread_m: float
    index_middle_separation_m: float
    reference_offset_norm_m: float
    hard_filter_pass: bool
    failure_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_edge_margin_m": self.minimum_edge_margin_m,
            "height_spread_m": self.height_spread_m,
            "index_middle_separation_m": self.index_middle_separation_m,
            "reference_offset_norm_m": self.reference_offset_norm_m,
            "hard_filter_pass": self.hard_filter_pass,
            "failure_reasons": list(self.failure_reasons),
        }


def evaluate_contact_point_plan_geometry(
    plan: ContactPointPlan,
    policy: ContactPointSearchPolicy,
) -> ContactPointGeometryMetrics:
    half = 0.5 * plan.edge_m
    points = [plan.points[finger] for finger in FINGER_ORDER]
    margins = [half - max(abs(point.y_m), abs(point.z_m)) for point in points]
    heights = [point.z_m for point in points]
    index = plan.points["index"]
    middle = plan.points["mid"]
    # The opposed fingers are ordered along cube-local Y.  Matching the
    # registered schema gate on that axis prevents a large Z offset from
    # disguising an unusably small index/middle lateral separation.
    separation = abs(index.y_m - middle.y_m)
    assert policy.reference_points is not None
    offset = math.sqrt(
        sum(
            (plan.points[finger].y_m - policy.reference_points[finger].y_m) ** 2
            + (plan.points[finger].z_m - policy.reference_points[finger].z_m) ** 2
            for finger in FINGER_ORDER
        )
    )
    reasons: list[str] = []
    if min(margins) < policy.minimum_edge_margin_m - _TOLERANCE:
        reasons.append("edge_margin_below_minimum")
    spread = max(heights) - min(heights)
    if spread > policy.maximum_height_spread_m + _TOLERANCE:
        reasons.append("contact_height_spread_over_limit")
    if separation < policy.minimum_index_middle_separation_m - _TOLERANCE:
        reasons.append("index_middle_separation_below_minimum")
    return ContactPointGeometryMetrics(
        minimum_edge_margin_m=float(min(margins)),
        height_spread_m=float(spread),
        index_middle_separation_m=float(separation),
        reference_offset_norm_m=float(offset),
        hard_filter_pass=not reasons,
        failure_reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class GeneratedContactPointPlan:
    sample_index: int
    plan: ContactPointPlan
    metrics: ContactPointGeometryMetrics

    def as_record(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "point_plan_id": self.plan.point_plan_id,
            "contact_point_plan": self.plan.as_config(),
            "geometry_metrics": self.metrics.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ContactPointGenerationResult:
    seed: int
    sample_count: int
    eligible_count: int
    retained: tuple[GeneratedContactPointPlan, ...]
    generator: str = "registered_seed_anchor_plus_cranley_shifted_halton_6d"

    def as_report(self) -> dict[str, Any]:
        return {
            "generator": self.generator,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "eligible_count": self.eligible_count,
            "retained_count": len(self.retained),
            "retained": [value.as_record() for value in self.retained],
        }


def _radical_inverse_sequence(count: int, base: int) -> np.ndarray:
    indices = np.arange(1, count + 1, dtype=np.int64)
    work = indices.copy()
    result = np.zeros(count, dtype=np.float64)
    factor = 1.0 / float(base)
    while bool(np.any(work)):
        result += factor * (work % base)
        work //= base
        factor /= float(base)
    return result


def _halton_6d(count: int, seed: int) -> np.ndarray:
    values = np.column_stack(
        [_radical_inverse_sequence(count, base) for base in _PRIMES]
    )
    shift = np.random.Generator(np.random.PCG64(seed)).random(len(_PRIMES))
    shifted = np.mod(values + shift[np.newaxis, :], 1.0)
    # The user's explicitly requested contact triplet is the registered
    # center of the search and must be screened, not merely approached by a
    # random low-discrepancy sample.  It occupies one member of the declared
    # 120k prefix; the remaining members retain their seeded Halton order.
    shifted[0, :] = 0.5
    return shifted


def generate_contact_point_plans(
    policy: ContactPointSearchPolicy | None = None,
    *,
    edge_m: float = DEFAULT_EDGE_M,
    sample_count: int | None = None,
    retain_count: int | None = None,
) -> ContactPointGenerationResult:
    """Generate the exact deterministic prefix and retain hard-filtered plans.

    The entire declared prefix is evaluated even though only the best plans
    are materialized.  This keeps the normal 120k campaign inexpensive while
    preserving an auditable ``sample_count`` and stable sample indices.
    """

    resolved = policy or ContactPointSearchPolicy()
    edge = _finite(edge_m, "edge_m")
    if edge <= 0.0:
        raise ValueError("edge_m must be positive")
    count = resolved.sample_count if sample_count is None else _positive_int(sample_count, "sample_count")
    retain = (
        resolved.retain_point_plan_count
        if retain_count is None
        else _positive_int(retain_count, "retain_count")
    )
    if retain > count:
        raise ValueError("retain_count cannot exceed sample_count")
    assert resolved.reference_points is not None
    unit = _halton_6d(count, resolved.seed)
    center = np.asarray(
        [
            coordinate
            for finger in FINGER_ORDER
            for coordinate in (
                resolved.reference_points[finger].y_m,
                resolved.reference_points[finger].z_m,
            )
        ],
        dtype=np.float64,
    )
    sampled = center + (2.0 * unit - 1.0) * resolved.half_width_m
    shaped = sampled.reshape(count, len(FINGER_ORDER), 2)
    half = 0.5 * edge
    edge_margin = half - np.max(np.abs(shaped), axis=2)
    minimum_edge_margin = np.min(edge_margin, axis=1)
    height_spread = np.ptp(shaped[:, :, 1], axis=1)
    index_middle_separation = np.abs(shaped[:, 1, 0] - shaped[:, 2, 0])
    reference_offset = np.linalg.norm(sampled - center[np.newaxis, :], axis=1)
    eligible = (
        (minimum_edge_margin >= resolved.minimum_edge_margin_m - _TOLERANCE)
        & (height_spread <= resolved.maximum_height_spread_m + _TOLERANCE)
        & (
            index_middle_separation
            >= resolved.minimum_index_middle_separation_m - _TOLERANCE
        )
    )
    eligible_indices = np.flatnonzero(eligible)
    # Primary key first in the documented rank; numpy.lexsort consumes keys
    # in reverse order.  The one-based Halton index is the final stable tie.
    ordered = eligible_indices[
        np.lexsort(
            (
                eligible_indices,
                -index_middle_separation[eligible_indices],
                -minimum_edge_margin[eligible_indices],
                reference_offset[eligible_indices],
                height_spread[eligible_indices],
            )
        )
    ]
    retained: list[GeneratedContactPointPlan] = []
    for index in ordered[:retain]:
        points = {
            finger: CubeFaceContactPoint(
                REFERENCE_FACE[finger],
                float(shaped[index, finger_index, 0]),
                float(shaped[index, finger_index, 1]),
            )
            for finger_index, finger in enumerate(FINGER_ORDER)
        }
        plan = ContactPointPlan(edge, resolved.target_radius_m, points)
        metrics = evaluate_contact_point_plan_geometry(plan, resolved)
        assert metrics.hard_filter_pass
        retained.append(GeneratedContactPointPlan(int(index), plan, metrics))
    return ContactPointGenerationResult(
        seed=resolved.seed,
        sample_count=count,
        eligible_count=int(np.count_nonzero(eligible)),
        retained=tuple(retained),
    )


def bind_frozen_contact_point_plan(
    config: Mapping[str, Any],
    plan: ContactPointPlan,
) -> dict[str, Any]:
    """Return a copy bound to exactly one plan; reject semantic replacement."""

    result = copy.deepcopy(dict(config))
    cube = result.get("cube")
    if not isinstance(cube, Mapping):
        raise ValueError("config.cube must be a mapping")
    edge = _finite(cube.get("edge_m"), "cube.edge_m")
    if not math.isclose(edge, plan.edge_m, abs_tol=_TOLERANCE):
        raise ValueError("contact point plan edge does not match cube.edge_m")
    topology = result.get("contact_topology")
    if not isinstance(topology, Mapping) or not isinstance(
        topology.get("target_faces"), Mapping
    ):
        raise ValueError("contact_topology.target_faces must be a mapping")
    target_faces = topology["target_faces"]
    expected_faces = {finger: plan.points[finger].face for finger in FINGER_ORDER}
    if dict(target_faces) != expected_faces:
        raise ValueError("contact point plan faces do not match contact topology")
    existing = result.get("contact_point_plan")
    if existing is not None:
        authenticated = ContactPointPlan.from_config(existing)
        if authenticated.point_plan_id != plan.point_plan_id:
            raise ValueError("a different contact_point_plan is already frozen")
    result["contact_point_plan"] = plan.as_config()
    return result


def assert_frozen_contact_point_plan(
    config: Mapping[str, Any],
    expected_point_plan_id: str | None = None,
) -> ContactPointPlan:
    raw = config.get("contact_point_plan")
    if not isinstance(raw, Mapping):
        raise ValueError("config has no frozen contact_point_plan")
    plan = ContactPointPlan.from_config(raw)
    if expected_point_plan_id is not None and plan.point_plan_id != str(
        expected_point_plan_id
    ):
        raise ValueError("contact_point_plan does not match expected plan id")
    cube = config.get("cube")
    if not isinstance(cube, Mapping) or not math.isclose(
        _finite(cube.get("edge_m"), "cube.edge_m"),
        plan.edge_m,
        abs_tol=_TOLERANCE,
    ):
        raise ValueError("frozen contact point plan no longer matches cube edge")
    topology = config.get("contact_topology", {}).get("target_faces", {})
    if dict(topology) != {
        finger: plan.points[finger].face for finger in FINGER_ORDER
    }:
        raise ValueError("frozen contact point plan no longer matches topology")
    return plan


def contact_point_world_positions(
    plan: ContactPointPlan,
    *,
    cube_world_position_m: Sequence[float],
    cube_world_rotation: Sequence[Sequence[float]],
) -> Mapping[str, tuple[float, float, float]]:
    position = _vector(cube_world_position_m, 3, "cube_world_position_m")
    rotation = np.asarray(cube_world_rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("cube_world_rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0.0):
        raise ValueError("cube_world_rotation must be orthonormal")
    return MappingProxyType(
        {
            finger: tuple(
                float(value)
                for value in position + rotation @ plan.points[finger].local_xyz_m(plan.edge_m)
            )
            for finger in FINGER_ORDER
        }
    )


@dataclass(frozen=True, slots=True)
class PointTargetVariables:
    actual_joint_qpos_rad: tuple[float, ...]
    root_delta_cube_m: tuple[float, float, float]
    wrist_local_rotvec_rad: tuple[float, float, float]

    def __post_init__(self) -> None:
        joints = _vector(
            self.actual_joint_qpos_rad, len(ACTIVE_ACTUATORS), "actual_joint_qpos_rad"
        )
        translation = _vector(self.root_delta_cube_m, 3, "root_delta_cube_m")
        rotation = _vector(
            self.wrist_local_rotvec_rad, 3, "wrist_local_rotvec_rad"
        )
        object.__setattr__(self, "actual_joint_qpos_rad", tuple(map(float, joints)))
        object.__setattr__(self, "root_delta_cube_m", tuple(map(float, translation)))
        object.__setattr__(self, "wrist_local_rotvec_rad", tuple(map(float, rotation)))

    def as_array(self) -> np.ndarray:
        result = np.asarray(
            (*self.actual_joint_qpos_rad, *self.root_delta_cube_m, *self.wrist_local_rotvec_rad),
            dtype=np.float64,
        )
        assert result.shape == (VARIABLE_COUNT,)
        return result

    @classmethod
    def from_array(cls, value: Sequence[float]) -> "PointTargetVariables":
        vector = _vector(value, VARIABLE_COUNT, "point-target variables")
        return cls(
            tuple(map(float, vector[:8])),
            tuple(map(float, vector[8:11])),
            tuple(map(float, vector[11:14])),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PointTargetVariables":
        nominal = config["grasp_pose"]["nominal_joint_qpos_rad"]
        metadata = config.get("candidate_metadata", {})
        point_metadata = (
            metadata.get("contact_point_target_search", {})
            if isinstance(metadata, Mapping)
            else {}
        )
        return cls(
            tuple(float(nominal[name]) for name in ACTIVE_ACTUATORS),
            tuple(float(value) for value in point_metadata.get("root_delta_cube_m", (0.0, 0.0, 0.0))),
            tuple(
                float(value)
                for value in np.radians(
                    point_metadata.get("wrist_local_rotvec_deg", (0.0, 0.0, 0.0))
                )
            ),
        )


def _cube_world_pose(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from .actual_contact_grasp_pose import _cube_world_position

    return (
        _cube_world_position(config),
        rpy_degrees_to_rotation_matrix(config["cube"].get("rpy_deg", (0.0, 0.0, 0.0))),
    )


def materialize_point_target_candidate(
    base_config: Mapping[str, Any],
    variables: PointTargetVariables,
    *,
    signed_orbit_deg: float,
    synchronize_preload: bool = True,
) -> dict[str, Any]:
    """Apply all fourteen variables without sampling or resetting cube pose."""

    orbit = _finite(signed_orbit_deg, "signed_orbit_deg")
    plan = assert_frozen_contact_point_plan(base_config)
    result = copy.deepcopy(dict(base_config))
    cube_before = copy.deepcopy(dict(result["cube"]))
    metadata = result.setdefault("candidate_metadata", {})
    existing = metadata.get("contact_point_target_search", {})
    if not isinstance(existing, Mapping):
        raise ValueError("candidate_metadata.contact_point_target_search must be a mapping")
    anchor = existing.get("anchor_hand_pose")
    if anchor is None:
        anchor = copy.deepcopy(dict(base_config["hand_pose"]))
    if not isinstance(anchor, Mapping) or set(anchor) != {"translation_m", "rpy_deg"}:
        raise ValueError("contact point anchor_hand_pose must contain translation_m and rpy_deg")
    cube_position, cube_rotation = _cube_world_pose(base_config)
    shared = transform_relative_wrist_pose(
        source_cube_world_position_m=cube_position,
        source_cube_world_rotation=cube_rotation,
        source_root_world_position_m=anchor["translation_m"],
        source_root_world_rotation=rpy_degrees_to_rotation_matrix(anchor["rpy_deg"]),
        clockwise_orbit_deg=orbit,
        root_delta_cube_m=variables.root_delta_cube_m,
        wrist_local_rotvec_deg=np.degrees(variables.wrist_local_rotvec_rad),
    )
    result["hand_pose"] = {
        "translation_m": list(shared.root_world_position_m),
        "rpy_deg": rotation_matrix_to_rpy_degrees(
            shared.root_world_rotation,
            reference_rpy_deg=base_config["hand_pose"]["rpy_deg"],
        ).tolist(),
    }
    nominal = result["grasp_pose"]["nominal_joint_qpos_rad"]
    for index, name in enumerate(ACTIVE_ACTUATORS):
        nominal[name] = float(variables.actual_joint_qpos_rad[index])
    if synchronize_preload:
        preload = result.get("control", {}).get("contact_preload_targets_rad")
        if isinstance(preload, Mapping):
            result["control"]["contact_preload_targets_rad"] = {
                name: float(nominal[name]) for name in ACTIVE_ACTUATORS
            }
    persisted = copy.deepcopy(dict(existing))
    persisted.update(
        {
            "signed_orbit_deg": orbit,
            "anchor_hand_pose": copy.deepcopy(dict(anchor)),
            "root_delta_cube_m": list(variables.root_delta_cube_m),
            "wrist_local_rotvec_deg": np.degrees(
                variables.wrist_local_rotvec_rad
            ).tolist(),
            "root_in_cube_m": list(shared.root_in_cube_m),
            "cube_from_root_rotation": [list(row) for row in shared.cube_from_root_rotation],
            "root_cube_distance_m": shared.diagnostics.result_root_cube_distance_m,
            "point_plan_id": plan.point_plan_id,
            "cube_pose_sampled": False,
            "hand_root_fixed_during_simulation": True,
        }
    )
    metadata["contact_point_target_search"] = persisted
    if result["cube"] != cube_before:
        raise AssertionError("point-target materialization changed cube pose")
    return result


@dataclass(frozen=True, slots=True)
class PointTargetTrialEvaluation:
    static_result: Any
    measurement: tuple[float, ...] | None
    point_error_yz_m: tuple[tuple[float, float], ...] | None
    point_distance_m: tuple[float, ...] | None
    safety_violations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.measurement is not None:
            values = _vector(self.measurement, len(MEASUREMENT_NAMES), "measurement")
            object.__setattr__(self, "measurement", tuple(map(float, values)))
        if self.point_error_yz_m is not None:
            values = np.asarray(self.point_error_yz_m, dtype=np.float64)
            if values.shape != (3, 2) or not np.isfinite(values).all():
                raise ValueError("point_error_yz_m must have shape (3, 2)")
            object.__setattr__(
                self,
                "point_error_yz_m",
                tuple(tuple(map(float, row)) for row in values),
            )
        if self.point_distance_m is not None:
            values = _vector(self.point_distance_m, 3, "point_distance_m")
            if np.any(values < 0.0):
                raise ValueError("point_distance_m must be non-negative")
            object.__setattr__(self, "point_distance_m", tuple(map(float, values)))
        object.__setattr__(
            self, "safety_violations", tuple(str(value) for value in self.safety_violations)
        )

    @property
    def safe(self) -> bool:
        return self.measurement is not None and not self.safety_violations


def point_target_trial_evaluation(
    result: Any,
    config: Mapping[str, Any],
    plan: ContactPointPlan | None = None,
    *,
    minimum_normal_alignment: float = 0.95,
    maximum_penetration_m: float = 0.002,
    require_extended_safety_evidence: bool = True,
) -> PointTargetTrialEvaluation:
    """Convert existing real witnesses to gap/YZ/normal DLS measurement."""

    from .relative_wrist_pose_search import actual_contact_trial_evaluation

    frozen = assert_frozen_contact_point_plan(config)
    if plan is not None and frozen.point_plan_id != plan.point_plan_id:
        raise ValueError("evaluation plan does not match frozen config plan")
    plan = frozen
    cube_position, cube_rotation = _cube_world_pose(config)
    gravity = np.asarray(config.get("scene", {}).get("gravity", (0.0, 0.0, -9.81)), dtype=np.float64)
    if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) <= np.finfo(float).eps:
        # The real model factory below supplies the actual gravity.  This
        # fallback only supports synthetic result tests and fails closed.
        gravity = np.asarray((0.0, 0.0, -1.0))
    base = actual_contact_trial_evaluation(
        result,
        -gravity / np.linalg.norm(gravity),
        maximum_penetration_m=maximum_penetration_m,
        minimum_normal_alignment=minimum_normal_alignment,
        require_v11_safety_evidence=require_extended_safety_evidence,
    )
    if base.measurement is None:
        return PointTargetTrialEvaluation(
            result, None, None, None, base.safety_violations
        )
    witnesses = tuple(getattr(result, "target_witnesses", ()) or ())
    if len(witnesses) != 3 or any(value is None for value in witnesses):
        return PointTargetTrialEvaluation(
            result,
            None,
            None,
            None,
            (*base.safety_violations, "missing_distal_witness"),
        )
    errors: list[tuple[float, float]] = []
    face_reasons: list[str] = []
    for finger, witness in zip(FINGER_ORDER, witnesses):
        world = _vector(
            getattr(witness, "cube_point_world_m"), 3, f"{finger} witness point"
        )
        local = cube_rotation.T @ (world - cube_position)
        target = plan.points[finger]
        errors.append((float(local[1] - target.y_m), float(local[2] - target.z_m)))
        witness_finger = str(getattr(witness, "finger", finger))
        if witness_finger != finger:
            face_reasons.append(f"wrong_target_finger:{finger}")
        raw_face = str(getattr(witness, "target_face", ""))
        accepted_names = {
            "-X": {"-X", "X_NEG"},
            "+X": {"+X", "X_POS"},
        }[target.face]
        if raw_face and raw_face not in accepted_names:
            face_reasons.append(f"wrong_target_face:{finger}")
    error_array = np.asarray(errors, dtype=np.float64)
    distances = np.linalg.norm(error_array, axis=1)
    gaps = tuple(float(value) for value in base.measurement[:3])
    normals = tuple(float(value) for value in base.measurement[5:8])
    measurement = (
        *gaps,
        *tuple(float(value) for value in error_array.reshape(-1)),
        *normals,
    )
    return PointTargetTrialEvaluation(
        result,
        measurement,
        tuple(errors),
        tuple(map(float, distances)),
        tuple(dict.fromkeys((*base.safety_violations, *face_reasons))),
    )


@dataclass(frozen=True, slots=True)
class PointTargetStaticAcceptance:
    passed: bool
    reasons: tuple[str, ...]
    maximum_point_distance_m: float
    point_distance_m: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "maximum_point_distance_m": self.maximum_point_distance_m,
            "point_distance_m": list(self.point_distance_m),
        }


def point_target_static_acceptance(
    evaluation: PointTargetTrialEvaluation,
    policy: ContactPointSearchPolicy,
) -> PointTargetStaticAcceptance:
    reasons = list(evaluation.safety_violations)
    distances = evaluation.point_distance_m or (math.inf, math.inf, math.inf)
    if evaluation.measurement is None:
        reasons.append("missing_point_target_measurement")
    if not bool(getattr(evaluation.static_result, "static_geometry_pass", False)):
        reasons.append("base_static_geometry_failed")
    for finger, distance in zip(FINGER_ORDER, distances):
        if distance > policy.static_target_radius_m + _TOLERANCE:
            reasons.append(f"point_target_radius_failed:{finger}")
    return PointTargetStaticAcceptance(
        passed=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        maximum_point_distance_m=float(max(distances)),
        point_distance_m=tuple(float(value) for value in distances),
    )


PointTargetEvaluator = Callable[[Mapping[str, Any]], PointTargetTrialEvaluation]


def model_active_joint_bounds(
    model: mujoco.MjModel, config: Mapping[str, Any]
) -> Mapping[str, tuple[float, float]]:
    from ..experiment import resolve_experiment

    definition = resolve_experiment(dict(config))
    bounds: dict[str, tuple[float, float]] = {}
    for name in ACTIVE_ACTUATORS:
        low, high = definition.search_bounds.actuator_targets_rad[name]
        actuator_id = model.actuator(name).id
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if bool(model.jnt_limited[joint_id]):
            low = max(float(low), float(model.jnt_range[joint_id, 0]))
            high = min(float(high), float(model.jnt_range[joint_id, 1]))
        if low > high:
            raise ValueError(f"empty registered/model joint intersection for {name}")
        bounds[name] = (float(low), float(high))
    return MappingProxyType(bounds)


def build_point_target_trial_evaluator(
    base_config: Mapping[str, Any],
    plan: ContactPointPlan | None = None,
) -> tuple[PointTargetEvaluator, Mapping[str, tuple[float, float]]]:
    from .actual_contact_grasp_pose import (
        ActualContactStaticThresholds,
        evaluate_direct_actual_contact_pose,
    )

    frozen = assert_frozen_contact_point_plan(base_config)
    if plan is not None and plan.point_plan_id != frozen.point_plan_id:
        raise ValueError("evaluator plan does not match config")
    model, info = build_model(copy.deepcopy(dict(base_config)))
    data = mujoco.MjData(model)
    thresholds = ActualContactStaticThresholds.from_config(base_config)

    def evaluate(config: Mapping[str, Any]) -> PointTargetTrialEvaluation:
        # The sealed v11 evaluator deliberately enables its extended
        # forbidden-hand/all-distal/precontact collision proof only for the
        # exact schema version that introduced it.  Keep that source file
        # byte-for-byte stable (its hash authenticates archived v11 evidence)
        # and invoke the same numerical capability through an evaluation-only
        # schema shadow.  The original schema-v12 config remains the source of
        # the frozen point plan, wrist transform and all campaign identity.
        evaluation_config = copy.deepcopy(dict(config))
        evaluation_config["schema_version"] = 11
        result = evaluate_direct_actual_contact_pose(
            model, data, info, evaluation_config, thresholds=thresholds
        )
        return point_target_trial_evaluation(
            result,
            config,
            frozen,
            minimum_normal_alignment=thresholds.minimum_normal_alignment,
            maximum_penetration_m=0.002,
            require_extended_safety_evidence=True,
        )

    return evaluate, model_active_joint_bounds(model, base_config)


def point_target_boundary_violations(
    base_config: Mapping[str, Any],
    variables: PointTargetVariables,
    policy: ContactPointSearchPolicy,
    *,
    signed_orbit_deg: float,
    joint_bounds: Mapping[str, tuple[float, float]],
    check_pose_constraints: bool = True,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not any(
        math.isclose(signed_orbit_deg, value, abs_tol=_TOLERANCE)
        for value in policy.signed_orbit_deg
    ):
        reasons.append("signed_orbit_out_of_strata")
    if set(joint_bounds) != set(ACTIVE_ACTUATORS):
        reasons.append("joint_bounds_incomplete")
    for index, name in enumerate(ACTIVE_ACTUATORS):
        if name not in joint_bounds:
            continue
        low, high = joint_bounds[name]
        value = variables.actual_joint_qpos_rad[index]
        if not low - _TOLERANCE <= value <= high + _TOLERANCE:
            reasons.append(f"joint_out_of_bounds:{name}")
    thumb = variables.actual_joint_qpos_rad[0]
    if not policy.thumb_actual_range_rad[0] - _TOLERANCE <= thumb <= policy.thumb_actual_range_rad[1] + _TOLERANCE:
        reasons.append("thumb_actual_out_of_bounds")
    assert policy.root_delta_cube_m is not None
    assert policy.wrist_local_rotvec_deg is not None
    for index, axis in enumerate(_AXES):
        low, high = policy.root_delta_cube_m[axis]
        if not low - _TOLERANCE <= variables.root_delta_cube_m[index] <= high + _TOLERANCE:
            reasons.append(f"root_delta_out_of_bounds:{axis}")
    rotvec_deg = np.degrees(variables.wrist_local_rotvec_rad)
    for index, axis in enumerate(_AXES):
        low, high = policy.wrist_local_rotvec_deg[axis]
        if not low - _TOLERANCE <= rotvec_deg[index] <= high + _TOLERANCE:
            reasons.append(f"wrist_rotvec_out_of_bounds:{axis}")
    if np.linalg.norm(rotvec_deg) > policy.max_wrist_local_rotvec_norm_deg + _TOLERANCE:
        reasons.append("wrist_rotvec_norm_out_of_bounds")
    candidate = materialize_point_target_candidate(
        base_config,
        variables,
        signed_orbit_deg=signed_orbit_deg,
        synchronize_preload=False,
    )
    distance = float(
        candidate["candidate_metadata"]["contact_point_target_search"][
            "root_cube_distance_m"
        ]
    )
    if not policy.root_cube_distance_m[0] - _TOLERANCE <= distance <= policy.root_cube_distance_m[1] + _TOLERANCE:
        reasons.append("root_cube_distance_out_of_bounds")
    if check_pose_constraints and isinstance(base_config.get("pose_constraints"), Mapping):
        resolved = resolved_pose_constraint_values(candidate)
        constraints = base_config["pose_constraints"]
        for key in ("finger_down_tilt_deg", "palm_plane_ground_angle_deg"):
            if key in constraints:
                low, high = _closed_range(constraints[key], f"pose_constraints.{key}")
                if not low - _TOLERANCE <= float(resolved[key]) <= high + _TOLERANCE:
                    reasons.append(f"pose_constraint_out_of_bounds:{key}")
        # Schema-v13 continues a source pose through 29 decreasing cube
        # sizes.  Bounding the local wrist rotvec alone is not sufficient:
        # composition is non-commutative, so an in-range local rotvec can
        # still move the resolved Euler roll/yaw outside the experiment's
        # registered search envelope.  Earlier schemas are intentionally
        # left byte-for-byte/numerically unchanged; v13 performs this extra
        # materialized-pose check during finite differences and line search.
        if int(base_config.get("schema_version", 1)) == 13:
            from ..experiment import resolve_experiment

            definition = resolve_experiment(candidate)
            search_bounds = definition.search_bounds
            hand_rpy = _vector(
                candidate["hand_pose"]["rpy_deg"], 3, "hand_pose.rpy_deg"
            )
            cube_rpy = _vector(
                candidate["cube"].get("rpy_deg", (0.0, 0.0, 0.0)),
                3,
                "cube.rpy_deg",
            )
            for value, declared, label in (
                (hand_rpy[0], search_bounds.hand_roll_deg, "hand_roll_deg"),
                (hand_rpy[2], search_bounds.hand_yaw_deg, "hand_yaw_deg"),
                (cube_rpy[2], search_bounds.cube_yaw_deg, "cube_yaw_deg"),
            ):
                if not declared[0] - _TOLERANCE <= value <= declared[1] + _TOLERANCE:
                    reasons.append(
                        f"registered_search_bound_out_of_bounds:{label}"
                    )
            far_constraints = definition.far_hand_pose_constraints
            if far_constraints is None or not far_constraints.contains_cube_position(
                resolved["cube_position_in_root_m"]
            ):
                reasons.append(
                    "registered_search_bound_out_of_bounds:cube_position_in_root_m"
                )
    return tuple(dict.fromkeys(reasons))


def project_point_target_variables(
    value: Sequence[float],
    policy: ContactPointSearchPolicy,
    joint_bounds: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    """Project a DLS proposal onto all registered 14-variable bounds.

    A joint at its limit is an active constraint, not a reason to discard the
    simultaneous root/wrist improvement.  Projection is deterministic and is
    applied only to line-search proposals; the hard boundary checker remains
    authoritative after projection.
    """

    result = _vector(value, VARIABLE_COUNT, "point-target variables")
    if set(joint_bounds) != set(ACTIVE_ACTUATORS):
        raise ValueError("joint_bounds must contain the eight active actuators")
    for index, name in enumerate(ACTIVE_ACTUATORS):
        lower, upper = joint_bounds[name]
        if index == 0:
            lower = max(float(lower), float(policy.thumb_actual_range_rad[0]))
            upper = min(float(upper), float(policy.thumb_actual_range_rad[1]))
        if lower > upper:
            raise ValueError(f"empty projected joint range for {name}")
        result[index] = float(np.clip(result[index], lower, upper))
    assert policy.root_delta_cube_m is not None
    for axis, index in zip(_AXES, range(8, 11)):
        result[index] = float(
            np.clip(result[index], *policy.root_delta_cube_m[axis])
        )
    assert policy.wrist_local_rotvec_deg is not None
    for axis, index in zip(_AXES, range(11, 14)):
        lower_deg, upper_deg = policy.wrist_local_rotvec_deg[axis]
        result[index] = float(
            np.clip(result[index], math.radians(lower_deg), math.radians(upper_deg))
        )
    rotvec = result[11:14]
    norm = float(np.linalg.norm(rotvec))
    norm_limit = math.radians(policy.max_wrist_local_rotvec_norm_deg)
    if norm > norm_limit:
        result[11:14] = rotvec * (norm_limit / norm)
    return result


@dataclass(frozen=True, slots=True)
class PointTargetDLSSettings:
    maximum_iterations: int = 8
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
        _positive_int(self.maximum_iterations, "maximum_iterations")
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
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class PointTargetDLSResult:
    config: dict[str, Any]
    variables: PointTargetVariables
    static_result: Any
    evaluation: PointTargetTrialEvaluation
    acceptance: PointTargetStaticAcceptance
    diagnostics: dict[str, Any]
    stop_reason: str


def _measurement_target_scale(policy: ContactPointSearchPolicy) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray((_TARGET_GAP_M,) * 3 + (0.0,) * 6 + (1.0,) * 3)
    scale = np.asarray((0.0005,) * 3 + (policy.static_target_radius_m,) * 6 + (0.05,) * 3)
    return target, scale


def _objective(
    measurement: np.ndarray,
    variables: np.ndarray,
    reference: np.ndarray,
    variable_scale: np.ndarray,
    policy: ContactPointSearchPolicy,
    settings: PointTargetDLSSettings,
) -> tuple[float, float, float]:
    target, scale = _measurement_target_scale(policy)
    contact = float(np.linalg.norm((target - measurement) / scale))
    regularization = float(
        math.sqrt(settings.regularization_weight)
        * np.linalg.norm((variables - reference) / variable_scale)
    )
    return math.hypot(contact, regularization), contact, regularization


def solve_point_target_dls(
    base_config: Mapping[str, Any],
    *,
    signed_orbit_deg: float,
    policy: ContactPointSearchPolicy | None = None,
    initial_variables: PointTargetVariables | None = None,
    settings: PointTargetDLSSettings | None = None,
    evaluator: PointTargetEvaluator | None = None,
    joint_bounds: Mapping[str, tuple[float, float]] | None = None,
    check_pose_constraints: bool = True,
) -> PointTargetDLSResult:
    """Run bounded DLS on real gap, point-YZ and normal measurements."""

    plan = assert_frozen_contact_point_plan(base_config)
    resolved_policy = policy or ContactPointSearchPolicy.from_config(base_config)
    if not any(
        math.isclose(signed_orbit_deg, value, abs_tol=_TOLERANCE)
        for value in resolved_policy.signed_orbit_deg
    ):
        raise ValueError("signed_orbit_deg is not a registered stratum")
    resolved_settings = settings or PointTargetDLSSettings()
    if evaluator is None:
        evaluator, model_bounds = build_point_target_trial_evaluator(base_config, plan)
        if joint_bounds is None:
            joint_bounds = model_bounds
    if joint_bounds is None or set(joint_bounds) != set(ACTIVE_ACTUATORS):
        raise ValueError("joint_bounds must contain the eight active actuators")
    current_variables = initial_variables or PointTargetVariables.from_config(base_config)
    current = current_variables.as_array()
    reference = current.copy()
    finite_difference = np.asarray(
        [resolved_settings.joint_finite_difference_rad] * 8
        + [resolved_settings.translation_finite_difference_m] * 3
        + [resolved_settings.rotation_finite_difference_rad] * 3
    )
    variable_scale = np.asarray(
        [resolved_settings.joint_step_rad] * 8
        + [resolved_settings.translation_step_m] * 3
        + [resolved_settings.rotation_step_rad] * 3
    )

    def boundary(value: np.ndarray) -> tuple[str, ...]:
        return point_target_boundary_violations(
            base_config,
            PointTargetVariables.from_array(value),
            resolved_policy,
            signed_orbit_deg=signed_orbit_deg,
            joint_bounds=joint_bounds,
            check_pose_constraints=check_pose_constraints,
        )

    def evaluate(value: np.ndarray) -> tuple[dict[str, Any], PointTargetTrialEvaluation]:
        candidate = materialize_point_target_candidate(
            base_config,
            PointTargetVariables.from_array(value),
            signed_orbit_deg=signed_orbit_deg,
        )
        return candidate, evaluator(candidate)

    initial_boundary = boundary(current)
    if initial_boundary:
        raise ValueError("initial point-target variables violate boundaries: " + ", ".join(initial_boundary))
    current_config, current_evaluation = evaluate(current)
    iterations: list[dict[str, Any]] = []
    if current_evaluation.measurement is None:
        acceptance = point_target_static_acceptance(current_evaluation, resolved_policy)
        return PointTargetDLSResult(
            current_config,
            PointTargetVariables.from_array(current),
            current_evaluation.static_result,
            current_evaluation,
            acceptance,
            {
                "method": "point_target_orientation_aware_dls",
                "variable_names": list(VARIABLE_NAMES),
                "measurement_names": list(MEASUREMENT_NAMES),
                "point_plan_id": plan.point_plan_id,
                "signed_orbit_deg": float(signed_orbit_deg),
                "iterations": [],
                "initial_safety_violations": list(current_evaluation.safety_violations),
            },
            "missing_initial_distal_witness",
        )
    measurement = np.asarray(current_evaluation.measurement, dtype=np.float64)
    initial_measurement = measurement.copy()
    target, measurement_scale = _measurement_target_scale(resolved_policy)
    stop_reason = "maximum_iterations"
    for iteration in range(resolved_settings.maximum_iterations):
        before, contact_before, regularization_before = _objective(
            measurement, current, reference, variable_scale, resolved_policy, resolved_settings
        )
        if contact_before <= resolved_settings.contact_tolerance:
            stop_reason = "contact_target_converged"
            break
        jacobian = np.zeros((len(MEASUREMENT_NAMES), VARIABLE_COUNT))
        fd_rejections: list[dict[str, Any]] = []
        for column in range(VARIABLE_COUNT):
            selected: tuple[np.ndarray, float] | None = None
            for direction in (1.0, -1.0):
                trial = current.copy()
                trial[column] += direction * finite_difference[column]
                reasons = boundary(trial)
                if reasons:
                    fd_rejections.append({"variable": VARIABLE_NAMES[column], "direction": direction, "reasons": list(reasons)})
                    continue
                _, trial_evaluation = evaluate(trial)
                if not trial_evaluation.safe:
                    fd_rejections.append({"variable": VARIABLE_NAMES[column], "direction": direction, "reasons": list(trial_evaluation.safety_violations) or ["unsafe_trial_evaluation"]})
                    continue
                selected = (np.asarray(trial_evaluation.measurement), direction * finite_difference[column])
                break
            if selected is not None:
                trial_measurement, actual_step = selected
                jacobian[:, column] = ((trial_measurement - measurement) / actual_step) / measurement_scale
        normalized_residual = (target - measurement) / measurement_scale
        scaled_jacobian = jacobian * variable_scale[np.newaxis, :]
        weight = math.sqrt(resolved_settings.regularization_weight)
        system = np.vstack((scaled_jacobian, weight * np.eye(VARIABLE_COUNT)))
        rhs = np.concatenate((normalized_residual, -weight * (current - reference) / variable_scale))
        normal = system.T @ system + resolved_settings.damping**2 * np.eye(VARIABLE_COUNT)
        unit_step = np.linalg.solve(normal, system.T @ rhs)
        maximum = float(np.max(np.abs(unit_step)))
        if maximum > 1.0:
            unit_step /= maximum
        step = variable_scale * unit_step
        accepted = None
        line_search: list[dict[str, Any]] = []
        for line_scale in (
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
        ):
            trial = project_point_target_variables(
                current + line_scale * step,
                resolved_policy,
                joint_bounds,
            )
            reasons = boundary(trial)
            if reasons:
                line_search.append({"scale": line_scale, "accepted": False, "reasons": list(reasons)})
                continue
            trial_config, trial_evaluation = evaluate(trial)
            if not trial_evaluation.safe:
                line_search.append({"scale": line_scale, "accepted": False, "reasons": list(trial_evaluation.safety_violations)})
                continue
            trial_measurement = np.asarray(trial_evaluation.measurement)
            after, contact_after, regularization_after = _objective(
                trial_measurement, trial, reference, variable_scale, resolved_policy, resolved_settings
            )
            improved = after + 1e-12 < before
            line_search.append({"scale": line_scale, "accepted": improved, "reasons": [] if improved else ["objective_not_improved"], "objective": after})
            if improved:
                accepted = (trial, trial_config, trial_evaluation, trial_measurement, line_scale, after, contact_after, regularization_after)
                break
        record = {
            "iteration": iteration,
            "objective_before": before,
            "contact_objective_before": contact_before,
            "regularization_before": regularization_before,
            "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
            "finite_difference_rejections": fd_rejections,
            "line_search": line_search,
            "accepted": accepted is not None,
        }
        iterations.append(record)
        if accepted is None:
            stop_reason = "dls_no_safe_improvement"
            break
        current, current_config, current_evaluation, measurement, accepted_scale, after, contact_after, regularization_after = accepted
        record.update({"accepted_scale": accepted_scale, "objective_after": after, "contact_objective_after": contact_after, "regularization_after": regularization_after})
    final_variables = PointTargetVariables.from_array(current)
    final_config, final_evaluation = evaluate(current)
    acceptance = point_target_static_acceptance(final_evaluation, resolved_policy)
    precontact_values = getattr(
        final_evaluation.static_result, "precontact_joint_qpos_rad", None
    )
    precontact_solution_applied = False
    if (
        acceptance.passed
        and isinstance(precontact_values, Sequence)
        and not isinstance(precontact_values, (str, bytes))
        and len(precontact_values) == len(ACTIVE_ACTUATORS)
    ):
        from .actual_contact_grasp_pose import apply_precontact_solution

        final_config = apply_precontact_solution(final_config, final_evaluation.static_result)
        precontact_solution_applied = True
    final_objective, final_contact, final_regularization = _objective(
        np.asarray(final_evaluation.measurement), current, reference, variable_scale, resolved_policy, resolved_settings
    )
    diagnostics = {
        "method": "point_target_orientation_aware_dls",
        "variable_names": list(VARIABLE_NAMES),
        "measurement_names": list(MEASUREMENT_NAMES),
        "point_plan_id": plan.point_plan_id,
        "signed_orbit_deg": float(signed_orbit_deg),
        "initial_variables": reference.tolist(),
        "final_variables": current.tolist(),
        "initial_measurement": initial_measurement.tolist(),
        "final_measurement": list(final_evaluation.measurement),
        "final_point_error_yz_m": [list(value) for value in (final_evaluation.point_error_yz_m or ())],
        "final_point_distance_m": list(final_evaluation.point_distance_m or ()),
        "final_safety_violations": list(final_evaluation.safety_violations),
        "final_objective": final_objective,
        "final_contact_objective": final_contact,
        "final_regularization": final_regularization,
        "final_static_acceptance": acceptance.as_dict(),
        "precontact_solution_applied": precontact_solution_applied,
        "iterations": iterations,
    }
    return PointTargetDLSResult(
        final_config,
        final_variables,
        final_evaluation.static_result,
        final_evaluation,
        acceptance,
        diagnostics,
        stop_reason,
    )


def default_point_plan_reachability_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Rank DLS reachability without worker completion-order dependence."""

    return (
        not bool(record.get("reachable", False)),
        -int(record.get("reachable_seed_count", 0)),
        -float(record.get("minimum_safety_margin_m", -math.inf)),
        float(record.get("height_spread_m", math.inf)),
        float(record.get("line_of_action_moment_nm", math.inf)),
        -float(record.get("minimum_edge_margin_m", -math.inf)),
        str(record["point_plan_id"]),
    )


def retain_ranked_point_plans(
    records: Sequence[Mapping[str, Any]], *, top_k: int = DEFAULT_RETAIN_POINT_PLAN_COUNT
) -> tuple[dict[str, Any], ...]:
    retain = _positive_int(top_k, "top_k")
    unique: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = copy.deepcopy(dict(raw))
        point_plan_id = str(record["point_plan_id"])
        if point_plan_id in unique:
            raise ValueError(f"duplicate point_plan_id {point_plan_id}")
        unique[point_plan_id] = record
    return tuple(sorted(unique.values(), key=default_point_plan_reachability_rank)[:retain])


def point_target_static_record(
    candidate_id: int,
    dls_result: PointTargetDLSResult,
    *,
    source_id: str | int | None = None,
) -> dict[str, Any]:
    """Adapt one result to ``run_actual_contact_dynamic_grasp_candidates``."""

    identifier = int(candidate_id)
    if isinstance(candidate_id, bool) or identifier != candidate_id or identifier < 0:
        raise ValueError("candidate_id must be a non-negative integer")
    config = copy.deepcopy(dls_result.config)
    point_plan = assert_frozen_contact_point_plan(config)
    metrics = dls_result.static_result.as_dict()
    metrics["point_target"] = {
        "point_plan_id": point_plan.point_plan_id,
        "point_error_yz_m": [
            list(value) for value in (dls_result.evaluation.point_error_yz_m or ())
        ],
        "point_distance_m": list(dls_result.evaluation.point_distance_m or ()),
        "static_acceptance": dls_result.acceptance.as_dict(),
        "dls_stop_reason": dls_result.stop_reason,
    }
    candidate_metadata = config.get("candidate_metadata", {})
    point_metadata = (
        candidate_metadata.get("contact_point_target_search", {})
        if isinstance(candidate_metadata, Mapping)
        else {}
    )
    orbit = float(
        point_metadata.get(
            "signed_orbit_deg",
            dls_result.diagnostics.get("signed_orbit_deg", 0.0),
        )
    )
    record = {
        "candidate_id": identifier,
        "config": config,
        "candidate_sha256": _campaign_sha256(config),
        "grasp_pose_id": grasp_pose_id(config),
        "controller_id": controller_id(config),
        "static_metrics": metrics,
        "static_pass": dls_result.acceptance.passed,
        "point_plan_id": point_plan.point_plan_id,
        "signed_orbit_deg": orbit,
        # Existing v11 dynamics/report code consumes the historical spelling.
        "clockwise_orbit_deg": orbit,
        "static_rank": default_point_target_static_rank_values(
            metrics, identifier
        ),
    }
    if source_id is not None:
        record["source_id"] = str(source_id)
    return record


def default_point_target_static_rank_values(
    metrics: Mapping[str, Any], candidate_id: int
) -> tuple[Any, ...]:
    target = metrics.get("point_target", {})
    acceptance = target.get("static_acceptance", {}) if isinstance(target, Mapping) else {}
    distances = tuple(float(value) for value in target.get("point_distance_m", (math.inf,) * 3))
    witnesses = metrics.get("target_witness", {})
    normal_deficit = 0.0
    gap_violation = 0.0
    for finger in FINGER_ORDER:
        witness = witnesses.get(finger) if isinstance(witnesses, Mapping) else None
        if not isinstance(witness, Mapping):
            normal_deficit += 1.0
            gap_violation += 0.020
            continue
        normal_deficit += max(0.0, 0.95 - float(witness["normal_alignment"]))
        gap = float(witness["signed_gap_m"])
        gap_violation += max(0.0, -0.0005 - gap, gap - 0.00025)
    retreats = metrics.get("retreat_evidence", {})
    max_closure_angle = max(
        (
            180.0
            if not isinstance(retreats, Mapping) or not isinstance(retreats.get(finger), Mapping)
            else float(retreats[finger]["closure_angle_deg"])
            for finger in FINGER_ORDER
        ),
        default=180.0,
    )
    return (
        not bool(acceptance.get("passed", False)),
        int(metrics.get("missing_target_witness_count", 99)),
        int(metrics.get("off_target_distal_penetrating_count", 99)),
        max(distances, default=math.inf),
        sum(distances),
        gap_violation,
        normal_deficit,
        max(0.0, -float(metrics.get("minimum_active_nondistal_gap_m", -1.0))),
        max_closure_angle,
        int(candidate_id),
    )


def default_point_target_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    if "static_rank" in record:
        return (*tuple(record["static_rank"]), int(record["candidate_id"]))
    return default_point_target_static_rank_values(
        record["static_metrics"], int(record["candidate_id"])
    )


def retain_point_target_static_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    top_k: int = DEFAULT_RETAIN_STATIC_COUNT,
    cover_signed_orbits: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Retain at most 64, covering every feasible signed orbit first."""

    retain = _positive_int(top_k, "top_k")
    values: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in records:
        record = copy.deepcopy(dict(raw))
        candidate_id = int(record["candidate_id"])
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate_id {candidate_id}")
        seen.add(candidate_id)
        values.append(record)
    values.sort(key=default_point_target_candidate_rank)
    selected: list[dict[str, Any]] = []
    if cover_signed_orbits:
        for orbit in sorted(
            {
                float(value.get("signed_orbit_deg", value.get("clockwise_orbit_deg")))
                for value in values
                if bool(value.get("static_pass", False))
            }
        ):
            candidate = next(
                value
                for value in values
                if bool(value.get("static_pass", False))
                and math.isclose(
                    float(value.get("signed_orbit_deg", value.get("clockwise_orbit_deg"))),
                    orbit,
                    abs_tol=_TOLERANCE,
                )
            )
            if len(selected) < retain:
                selected.append(candidate)
    selected_ids = {int(value["candidate_id"]) for value in selected}
    selected.extend(
        value
        for value in values
        if int(value["candidate_id"]) not in selected_ids
    )
    selected = selected[:retain]
    selected.sort(key=default_point_target_candidate_rank)
    return tuple(selected)


__all__ = [
    "DEFAULT_EDGE_M",
    "DEFAULT_RETAIN_POINT_PLAN_COUNT",
    "DEFAULT_RETAIN_STATIC_COUNT",
    "DEFAULT_SAMPLE_COUNT",
    "DEFAULT_SEED",
    "FACE_LABELS",
    "FINGER_ORDER",
    "MEASUREMENT_NAMES",
    "POINT_PLAN_SCHEMA_VERSION",
    "POINT_SEARCH_SCHEMA_VERSION",
    "REFERENCE_FACE",
    "REFERENCE_POINT_YZ_M",
    "SIGNED_ORBIT_DEG",
    "VARIABLE_COUNT",
    "VARIABLE_NAMES",
    "ContactPointGenerationResult",
    "ContactPointGeometryMetrics",
    "ContactPointPlan",
    "ContactPointSearchPolicy",
    "CubeFaceContactPoint",
    "GeneratedContactPointPlan",
    "PointTargetDLSResult",
    "PointTargetDLSSettings",
    "PointTargetStaticAcceptance",
    "PointTargetTrialEvaluation",
    "PointTargetVariables",
    "assert_frozen_contact_point_plan",
    "bind_frozen_contact_point_plan",
    "build_point_target_trial_evaluator",
    "contact_point_world_positions",
    "default_point_plan_reachability_rank",
    "default_point_target_candidate_rank",
    "evaluate_contact_point_plan_geometry",
    "generate_contact_point_plans",
    "materialize_point_target_candidate",
    "model_active_joint_bounds",
    "point_target_boundary_violations",
    "project_point_target_variables",
    "point_target_static_acceptance",
    "point_target_static_record",
    "point_target_trial_evaluation",
    "retain_point_target_static_candidates",
    "retain_ranked_point_plans",
    "solve_point_target_dls",
]
