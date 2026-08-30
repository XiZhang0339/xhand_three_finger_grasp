"""Pure planning helpers for the 61--63 mm relative-pose rescue.

This module deliberately owns no MuJoCo loop, CLI, or experiment registry
entry.  It turns one schema-v3 base configuration into deterministic candidate
configurations, provides the VERIFY continuation policy and result ordering,
and exposes a small injectable-runner boundary for a caller-owned simulator.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, validate_config
from .search import v3_verify_near_miss_rank


SOURCE_EXPERIMENT_ID = (
    "left_opposed_face_palm_down_larger_cube_grasp_then_lift"
)
RESCUE_STRATEGY = "relative_pose_rescue"
FIXED_SEED = 20260821
TARGET_FACES = MappingProxyType(
    {"thumb": "-X", "index": "+X", "mid": "+X"}
)

SOURCE_TUNE_RESULTS = (
    "artifacts/left_opposed_face_palm_down_larger_cube_grasp_then_lift/"
    "tune/tune_results.json"
)
SOURCE_TUNE_RESULTS_SHA256 = (
    "5c1cd21ac523904e06c2d2062c34fba0699323619f78b2c7f51108c6cb06a396"
)

THUMB_INDEX_ACTUATORS = ACTIVE_ACTUATORS[:6]
MIDDLE_JOINT1 = "left_hand_mid_joint1_actuator"
MIDDLE_JOINT2 = "left_hand_mid_joint2_actuator"


def _target_map(values: Sequence[float]) -> dict[str, float]:
    if len(values) != len(ACTIVE_ACTUATORS):
        raise ValueError("target values must follow the eight active actuators")
    return dict(zip(ACTIVE_ACTUATORS, (float(value) for value in values)))


_ANCHOR_TARGETS = _target_map(
    (0.95, 0.45, 0.50, 0.027, 0.348, 1.403, 0.65, 1.35)
)

_NARROW_TARGET_DELTAS = {
    "left_hand_thumb_bend_joint_actuator": (-0.08, 0.08),
    "left_hand_thumb_rota_joint1_actuator": (-0.08, 0.08),
    # The remaining ranges cover the measured 18/256 stable-grasp basin.
    "left_hand_thumb_rota_joint2_actuator": (-0.04, 0.08),
    "left_hand_index_bend_joint_actuator": (-0.047, 0.053),
    "left_hand_index_joint1_actuator": (-0.048, 0.132),
    "left_hand_index_joint2_actuator": (0.077, 0.297),
}

_ACTUATOR_LIMITS = {
    "left_hand_thumb_bend_joint_actuator": (0.15, 1.35),
    "left_hand_thumb_rota_joint1_actuator": (0.15, 1.35),
    "left_hand_thumb_rota_joint2_actuator": (0.40, 1.50),
    "left_hand_index_bend_joint_actuator": (-0.15, 0.15),
    "left_hand_index_joint1_actuator": (0.30, 1.70),
    # The rescue experiment consumed its one permitted 0.1 rad boundary
    # expansion on the two distal absolute-target axes.  Keep this audit copy
    # aligned with the registered safety envelope even though the measured
    # focused basin below currently tops out at 1.70 rad.
    "left_hand_index_joint2_actuator": (0.30, 1.80),
    MIDDLE_JOINT1: (0.30, 1.70),
    MIDDLE_JOINT2: (0.30, 1.80),
}


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _closed_range(values: Sequence[float], label: str) -> tuple[float, float]:
    materialized = tuple(values)
    if len(materialized) != 2:
        raise ValueError(f"{label} must contain exactly two bounds")
    lower = _finite(materialized[0], f"{label}[0]")
    upper = _finite(materialized[1], f"{label}[1]")
    if lower > upper:
        raise ValueError(f"{label} must be increasing")
    return (lower, upper)


def _range_map(
    values: Mapping[str, Sequence[float]],
    *,
    keys: Iterable[str],
    label: str,
) -> Mapping[str, tuple[float, float]]:
    expected = tuple(keys)
    if set(values) != set(expected):
        raise ValueError(f"{label} must contain exactly {expected!r}")
    return MappingProxyType(
        {name: _closed_range(values[name], f"{label}.{name}") for name in expected}
    )


def _finite_vector(
    values: Sequence[float], length: int, label: str
) -> tuple[float, ...]:
    result = tuple(_finite(value, label) for value in values)
    if len(result) != length:
        raise ValueError(f"{label} must contain {length} values")
    return result


@dataclass(frozen=True)
class PoseRescuePlan:
    """Versioned, source-anchored plan for a focused geometry rescue.

    Thumb and index targets use small deltas around the audited anchor.  The
    middle finger is intentionally parameterized by joint-1 and the *sum* of
    joints 1 and 2.  This permits a wide transfer of bend between its links
    while retaining a bounded total closure instead of drawing two unrelated
    absolute targets.
    """

    schema_version: int = 1
    strategy: str = RESCUE_STRATEGY
    seed: int = FIXED_SEED
    anchor_edge_m: float = 0.062
    priority_edges_m: tuple[float, ...] = (0.061, 0.062, 0.063)
    source_candidate_id: int = 309288
    anchor_cube_in_root_m: tuple[float, float, float] = (
        0.082470,
        -0.029689,
        0.108106,
    )
    anchor_hand_rpy_deg: tuple[float, float, float] = (1.0, 90.0, -3.779)
    anchor_cube_yaw_deg: float = 28.549
    anchor_targets_rad: Mapping[str, float] = field(
        default_factory=lambda: dict(_ANCHOR_TARGETS)
    )
    cube_position_in_root_m: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (0.08164, 0.08364),
            "y": (-0.03085, -0.02885),
            "z": (0.10736, 0.10936),
        }
    )
    hand_roll_deg: tuple[float, float] = (-0.5, 2.5)
    hand_pitch_values_deg: tuple[float, ...] = (88.5, 89.0, 89.5, 90.0)
    hand_yaw_deg: tuple[float, float] = (-5.8, -1.8)
    cube_yaw_deg: tuple[float, float] = (26.5, 30.5)
    narrow_target_delta_rad: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: dict(_NARROW_TARGET_DELTAS)
    )
    middle_joint1_rad: tuple[float, float] = (0.52, 0.78)
    middle_joint2_rad: tuple[float, float] = (1.32, 1.58)
    middle_joint_sum_rad: tuple[float, float] = (1.84, 2.36)
    actuator_limits_rad: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: dict(_ACTUATOR_LIMITS)
    )
    broad_candidate_budget: int = 256
    local_parent_budget: int = 8
    local_samples_per_parent: int = 32
    local_radius_fraction: float = 0.25
    dynamic_candidate_budget: int = 512
    continuation_candidate_budget: int = 128

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("pose-rescue schema_version must be 1")
        if self.strategy != RESCUE_STRATEGY:
            raise ValueError(f"strategy must be {RESCUE_STRATEGY!r}")
        if self.seed != FIXED_SEED:
            raise ValueError(f"pose-rescue seed is fixed at {FIXED_SEED}")
        if not math.isclose(
            _finite(self.anchor_edge_m, "anchor_edge_m"),
            0.062,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("pose-rescue anchor_edge_m is fixed at 0.062")
        priorities = tuple(
            _finite(value, "priority_edges_m") for value in self.priority_edges_m
        )
        if priorities != (0.061, 0.062, 0.063):
            raise ValueError("priority_edges_m must be (0.061, 0.062, 0.063)")
        object.__setattr__(self, "priority_edges_m", priorities)
        if self.source_candidate_id != 309288:
            raise ValueError("source_candidate_id must preserve audited candidate 309288")

        anchor_cube = _finite_vector(
            self.anchor_cube_in_root_m, 3, "anchor_cube_in_root_m"
        )
        anchor_rpy = _finite_vector(
            self.anchor_hand_rpy_deg, 3, "anchor_hand_rpy_deg"
        )
        object.__setattr__(self, "anchor_cube_in_root_m", anchor_cube)
        object.__setattr__(self, "anchor_hand_rpy_deg", anchor_rpy)
        object.__setattr__(
            self,
            "anchor_cube_yaw_deg",
            _finite(self.anchor_cube_yaw_deg, "anchor_cube_yaw_deg"),
        )

        cube_ranges = _range_map(
            self.cube_position_in_root_m,
            keys=("x", "y", "z"),
            label="cube_position_in_root_m",
        )
        object.__setattr__(self, "cube_position_in_root_m", cube_ranges)
        for axis, value in zip(("x", "y", "z"), anchor_cube):
            if not cube_ranges[axis][0] <= value <= cube_ranges[axis][1]:
                raise ValueError(f"anchor cube coordinate {axis} lies outside focus range")

        for label in ("hand_roll_deg", "hand_yaw_deg", "cube_yaw_deg"):
            object.__setattr__(
                self, label, _closed_range(getattr(self, label), label)
            )
        pitches = tuple(
            _finite(value, "hand_pitch_values_deg")
            for value in self.hand_pitch_values_deg
        )
        if pitches != (88.5, 89.0, 89.5, 90.0):
            raise ValueError(
                "hand_pitch_values_deg must be (88.5, 89, 89.5, 90)"
            )
        object.__setattr__(self, "hand_pitch_values_deg", pitches)
        roll, pitch, yaw = anchor_rpy
        if not self.hand_roll_deg[0] <= roll <= self.hand_roll_deg[1]:
            raise ValueError("anchor hand roll lies outside focus range")
        if pitch not in pitches:
            raise ValueError("anchor hand pitch must be one of the discrete values")
        if not self.hand_yaw_deg[0] <= yaw <= self.hand_yaw_deg[1]:
            raise ValueError("anchor hand yaw lies outside focus range")
        if not self.cube_yaw_deg[0] <= self.anchor_cube_yaw_deg <= self.cube_yaw_deg[1]:
            raise ValueError("anchor cube yaw lies outside focus range")

        anchor_targets = {
            str(name): _finite(value, f"anchor_targets_rad.{name}")
            for name, value in self.anchor_targets_rad.items()
        }
        if set(anchor_targets) != set(ACTIVE_ACTUATORS):
            raise ValueError("anchor_targets_rad must contain all active actuators")
        object.__setattr__(
            self, "anchor_targets_rad", MappingProxyType(anchor_targets)
        )
        narrow = _range_map(
            self.narrow_target_delta_rad,
            keys=THUMB_INDEX_ACTUATORS,
            label="narrow_target_delta_rad",
        )
        limits = _range_map(
            self.actuator_limits_rad,
            keys=ACTIVE_ACTUATORS,
            label="actuator_limits_rad",
        )
        object.__setattr__(self, "narrow_target_delta_rad", narrow)
        object.__setattr__(self, "actuator_limits_rad", limits)
        for name in THUMB_INDEX_ACTUATORS:
            low = anchor_targets[name] + narrow[name][0]
            high = anchor_targets[name] + narrow[name][1]
            if low < limits[name][0] - 1e-12 or high > limits[name][1] + 1e-12:
                raise ValueError(f"narrow target range for {name!r} exceeds limits")

        object.__setattr__(
            self,
            "middle_joint1_rad",
            _closed_range(self.middle_joint1_rad, "middle_joint1_rad"),
        )
        object.__setattr__(
            self,
            "middle_joint2_rad",
            _closed_range(self.middle_joint2_rad, "middle_joint2_rad"),
        )
        object.__setattr__(
            self,
            "middle_joint_sum_rad",
            _closed_range(self.middle_joint_sum_rad, "middle_joint_sum_rad"),
        )
        j1_low, j1_high = self.middle_joint1_rad
        j2_focus_low, j2_focus_high = self.middle_joint2_rad
        sum_low, sum_high = self.middle_joint_sum_rad
        j2_low, j2_high = limits[MIDDLE_JOINT2]
        if (
            j1_low < limits[MIDDLE_JOINT1][0] - 1e-12
            or j1_high > limits[MIDDLE_JOINT1][1] + 1e-12
            or j2_focus_low < j2_low - 1e-12
            or j2_focus_high > j2_high + 1e-12
            or sum_low > j1_low + j2_focus_low + 1e-12
            or sum_high < j1_high + j2_focus_high - 1e-12
        ):
            raise ValueError("correlated middle target ranges exceed actuator limits")
        anchor_sum = anchor_targets[MIDDLE_JOINT1] + anchor_targets[MIDDLE_JOINT2]
        if not j1_low <= anchor_targets[MIDDLE_JOINT1] <= j1_high:
            raise ValueError("anchor middle joint1 lies outside its correlated range")
        if not sum_low <= anchor_sum <= sum_high:
            raise ValueError("anchor middle joint sum lies outside its correlated range")

        for label in (
            "broad_candidate_budget",
            "local_parent_budget",
            "local_samples_per_parent",
            "dynamic_candidate_budget",
            "continuation_candidate_budget",
        ):
            _positive_integer(getattr(self, label), label)
        radius = _finite(self.local_radius_fraction, "local_radius_fraction")
        if not 0.0 < radius <= 1.0:
            raise ValueError("local_radius_fraction must lie within (0, 1]")
        object.__setattr__(self, "local_radius_fraction", radius)
        if (
            self.broad_candidate_budget
            + self.local_parent_budget * self.local_samples_per_parent
            != self.dynamic_candidate_budget
        ):
            raise ValueError(
                "dynamic_candidate_budget must cover broad plus local candidates"
            )
        if self.continuation_candidate_budget > self.dynamic_candidate_budget:
            raise ValueError(
                "continuation_candidate_budget cannot exceed dynamic_candidate_budget"
            )

    @property
    def target_faces(self) -> Mapping[str, str]:
        return TARGET_FACES

    @property
    def thumb_index_target_ranges_rad(self) -> Mapping[str, tuple[float, float]]:
        return MappingProxyType(
            {
                name: (
                    self.anchor_targets_rad[name]
                    + self.narrow_target_delta_rad[name][0],
                    self.anchor_targets_rad[name]
                    + self.narrow_target_delta_rad[name][1],
                )
                for name in THUMB_INDEX_ACTUATORS
            }
        )

    @property
    def lhs_candidate_count(self) -> int:
        return self.broad_candidate_budget

    @property
    def grid_candidate_count(self) -> int:
        return self.broad_candidate_budget

    @property
    def local_candidate_budget(self) -> int:
        return self.local_parent_budget * self.local_samples_per_parent

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "seed": self.seed,
            "anchor_edge_m": self.anchor_edge_m,
            "priority_edges_m": list(self.priority_edges_m),
            "source_candidate_id": self.source_candidate_id,
            "anchor": {
                "cube_in_root_m": list(self.anchor_cube_in_root_m),
                "hand_rpy_deg": list(self.anchor_hand_rpy_deg),
                "cube_yaw_deg": self.anchor_cube_yaw_deg,
                "targets_rad": dict(self.anchor_targets_rad),
                "target_faces": dict(self.target_faces),
            },
            "focus": {
                "cube_position_in_root_m": {
                    name: list(bounds)
                    for name, bounds in self.cube_position_in_root_m.items()
                },
                "hand_roll_deg": list(self.hand_roll_deg),
                "hand_pitch_values_deg": list(self.hand_pitch_values_deg),
                "hand_yaw_deg": list(self.hand_yaw_deg),
                "cube_yaw_deg": list(self.cube_yaw_deg),
            },
            "correlated_targets": {
                "narrow_thumb_index_delta_rad": {
                    name: list(bounds)
                    for name, bounds in self.narrow_target_delta_rad.items()
                },
                "thumb_index_absolute_rad": {
                    name: list(bounds)
                    for name, bounds in self.thumb_index_target_ranges_rad.items()
                },
                "middle_joint1_rad": list(self.middle_joint1_rad),
                "middle_joint2_rad": list(self.middle_joint2_rad),
                "middle_joint_sum_rad": list(self.middle_joint_sum_rad),
                "middle_joint2_formula": "middle_joint_sum_rad - middle_joint1_rad",
                "actuator_limits_rad": {
                    name: list(bounds)
                    for name, bounds in self.actuator_limits_rad.items()
                },
            },
            "local_refinement": {
                "parent_budget": self.local_parent_budget,
                "samples_per_parent": self.local_samples_per_parent,
                "radius_fraction": self.local_radius_fraction,
            },
        }


DEFAULT_PLAN = PoseRescuePlan()


def _scale(unit: float, bounds: Sequence[float]) -> float:
    value = _finite(unit, "unit sample")
    if not 0.0 <= value <= 1.0:
        raise ValueError("unit sample must lie within [0, 1]")
    if value == 0.0:
        return float(bounds[0])
    if value == 1.0:
        return float(bounds[1])
    return float(bounds[0] + value * (bounds[1] - bounds[0]))


def _rpy_matrix(rpy_deg: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = np.radians(_finite_vector(rpy_deg, 3, "rpy_deg"))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _cube_world_position(config: Mapping[str, Any]) -> np.ndarray:
    try:
        cube = config["cube"]
        scene = config["scene"]
        center_xy = cube["center_xy_m"]
        return np.asarray(
            [
                _finite(center_xy[0], "cube.center_xy_m[0]"),
                _finite(center_xy[1], "cube.center_xy_m[1]"),
                _finite(scene["support_top_z_m"], "scene.support_top_z_m")
                + _finite(cube["edge_m"], "cube.edge_m") / 2.0
                + _finite(cube.get("z_offset_m", 0.0), "cube.z_offset_m"),
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("base config requires cube and support geometry") from error


def cube_position_in_root(config: Mapping[str, Any]) -> np.ndarray:
    """Recover the generated relative position for audit and tests."""

    try:
        rpy = config["hand_pose"]["rpy_deg"]
        translation = np.asarray(
            config["hand_pose"]["translation_m"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("config requires a finite hand_pose") from error
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("hand_pose.translation_m must contain three finite values")
    return _rpy_matrix(rpy).T @ (_cube_world_position(config) - translation)


def _latin_hypercube(
    samples: int, dimensions: int, *, seed: int
) -> np.ndarray:
    count = _positive_integer(samples, "samples")
    dims = _positive_integer(dimensions, "dimensions")
    rng = np.random.default_rng(seed)
    matrix = np.empty((count, dims), dtype=np.float64)
    for dimension in range(dims):
        matrix[:, dimension] = (
            rng.permutation(count) + rng.random(count)
        ) / count
    return matrix


def _grid_samples(samples: int, dimensions: int, *, seed: int) -> np.ndarray:
    """Choose deterministic points from an endpoint-inclusive tensor grid.

    A coprime stride avoids constructing the potentially large Cartesian
    product and avoids the leading-axis bias caused by truncating it.
    """

    count = _positive_integer(samples, "samples")
    dims = _positive_integer(dimensions, "dimensions")
    if count == 1:
        return np.full((1, dims), 0.5, dtype=np.float64)
    levels = max(2, int(math.ceil(count ** (1.0 / dims))))
    capacity = levels**dims
    stride = 2 * (int(seed) % max(1, capacity // 2)) + 1
    while math.gcd(stride, capacity) != 1:
        stride += 2
    start = (int(seed) * 1_000_003) % capacity
    flat_indices = tuple((start + stride * index) % capacity for index in range(count))
    matrix = np.empty((count, dims), dtype=np.float64)
    for row, flat in enumerate(flat_indices):
        value = flat
        for dimension in range(dims):
            digit = value % levels
            value //= levels
            matrix[row, dimension] = digit / (levels - 1)
    # A semantic seed also scrambles which physical parameter receives each
    # base-N digit, while every coordinate remains on the same finite grid.
    permutation = np.random.default_rng(seed).permutation(dims)
    return matrix[:, permutation]


_SAMPLE_DIMENSIONS = 15


def _edge_seed(plan: PoseRescuePlan, edge_m: float, method: str) -> int:
    method_offset = {"lhs": 100_000_000, "grid": 200_000_000}[method]
    return plan.seed + method_offset + int(round(edge_m * 1_000_000.0)) * 1009


def _sample_parameters(
    row: np.ndarray, *, plan: PoseRescuePlan
) -> dict[str, Any]:
    values = np.asarray(row, dtype=np.float64)
    if values.shape != (_SAMPLE_DIMENSIONS,) or not np.isfinite(values).all():
        raise ValueError(f"pose-rescue sample must contain {_SAMPLE_DIMENSIONS} values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("pose-rescue sample must remain within [0, 1]")

    cursor = 0
    cube_in_root = np.asarray(
        [
            _scale(values[cursor + index], plan.cube_position_in_root_m[axis])
            for index, axis in enumerate(("x", "y", "z"))
        ],
        dtype=np.float64,
    )
    cursor += 3
    roll = _scale(values[cursor], plan.hand_roll_deg)
    cursor += 1
    pitch_index = min(
        int(values[cursor] * len(plan.hand_pitch_values_deg)),
        len(plan.hand_pitch_values_deg) - 1,
    )
    pitch = plan.hand_pitch_values_deg[pitch_index]
    cursor += 1
    hand_yaw = _scale(values[cursor], plan.hand_yaw_deg)
    cursor += 1
    cube_yaw = _scale(values[cursor], plan.cube_yaw_deg)
    cursor += 1

    targets: dict[str, float] = {}
    for name in THUMB_INDEX_ACTUATORS:
        delta = _scale(values[cursor], plan.narrow_target_delta_rad[name])
        targets[name] = float(plan.anchor_targets_rad[name] + delta)
        cursor += 1
    middle_j1 = _scale(values[cursor], plan.middle_joint1_rad)
    cursor += 1
    feasible_sum = (
        max(
            plan.middle_joint_sum_rad[0],
            middle_j1 + plan.middle_joint2_rad[0],
        ),
        min(
            plan.middle_joint_sum_rad[1],
            middle_j1 + plan.middle_joint2_rad[1],
        ),
    )
    middle_sum = _scale(values[cursor], feasible_sum)
    targets[MIDDLE_JOINT1] = middle_j1
    targets[MIDDLE_JOINT2] = middle_sum - middle_j1

    for name, value in targets.items():
        lower, upper = plan.actuator_limits_rad[name]
        if value < lower - 1e-12 or value > upper + 1e-12:
            raise AssertionError(f"generated target {name!r} escaped validated limits")
        targets[name] = float(min(max(value, lower), upper))
    return {
        "cube_in_root_m": cube_in_root,
        "hand_rpy_deg": (roll, pitch, hand_yaw),
        "cube_yaw_deg": cube_yaw,
        "grasp_targets_rad": targets,
    }


def _anchor_parameters(plan: PoseRescuePlan) -> dict[str, Any]:
    return {
        "cube_in_root_m": np.asarray(
            plan.anchor_cube_in_root_m, dtype=np.float64
        ),
        "hand_rpy_deg": plan.anchor_hand_rpy_deg,
        "cube_yaw_deg": plan.anchor_cube_yaw_deg,
        "grasp_targets_rad": dict(plan.anchor_targets_rad),
    }


Validator = Callable[[dict[str, Any]], Any]


def _materialize_candidate(
    base_config: Mapping[str, Any],
    *,
    edge_m: float,
    parameters: Mapping[str, Any],
    plan: PoseRescuePlan,
    validator: Validator | None,
) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(base_config))
    if int(candidate.get("schema_version", 0)) != 3:
        raise ValueError("pose rescue requires a schema-v3 base config")
    try:
        candidate["cube"]["edge_m"] = float(edge_m)
        rpy = [float(value) for value in parameters["hand_rpy_deg"]]
        cube_in_root = np.asarray(parameters["cube_in_root_m"], dtype=np.float64)
        if cube_in_root.shape != (3,) or not np.isfinite(cube_in_root).all():
            raise ValueError("cube_in_root_m must contain three finite values")
        candidate["hand_pose"]["rpy_deg"] = rpy
        candidate["cube"]["rpy_deg"] = [
            0.0,
            0.0,
            float(parameters["cube_yaw_deg"]),
        ]
        candidate["hand_pose"]["translation_m"] = (
            _cube_world_position(candidate) - _rpy_matrix(rpy) @ cube_in_root
        ).tolist()
        candidate["control"] = {
            "grasp_targets_rad": {
                name: float(parameters["grasp_targets_rad"][name])
                for name in ACTIVE_ACTUATORS
            },
            "manipulation_delta_rad": {
                name: 0.0 for name in ACTIVE_ACTUATORS
            },
        }
        candidate["contact_topology"]["target_faces"] = dict(plan.target_faces)
    except (KeyError, TypeError) as error:
        raise ValueError(
            "base config requires cube, hand_pose, control and contact_topology"
        ) from error
    if validator is not None:
        validator(candidate)
    return candidate


def make_anchor_candidate(
    base_config: Mapping[str, Any],
    *,
    plan: PoseRescuePlan = DEFAULT_PLAN,
    validator: Validator | None = None,
) -> dict[str, Any]:
    """Materialize the audited 62 mm source candidate from relative pose."""

    return _materialize_candidate(
        base_config,
        edge_m=plan.anchor_edge_m,
        parameters=_anchor_parameters(plan),
        plan=plan,
        validator=validator,
    )


def generate_pose_rescue_candidates(
    base_config: Mapping[str, Any],
    *,
    plan: PoseRescuePlan = DEFAULT_PLAN,
    method: str = "lhs",
    sample_count: int | None = None,
    samples_per_edge: int | None = None,
    include_anchor: bool = True,
    validator: Validator | None = None,
) -> list[dict[str, Any]]:
    """Generate configs in declared size-priority order.

    ``method='lhs'`` uses a seeded Latin hypercube.  ``method='grid'`` uses a
    deterministic finite tensor grid without materializing the full Cartesian
    product.  In either mode the first 62 mm slot is replaced by the exact
    source anchor when ``include_anchor`` is true; the declared count is not
    increased.  By default the 256-candidate broad budget is distributed in
    size-priority order.  ``samples_per_edge`` remains an explicit convenience
    for tests and caller-selected symmetric campaigns.
    """

    if method not in {"lhs", "grid"}:
        raise ValueError("method must be 'lhs' or 'grid'")
    if sample_count is not None and samples_per_edge is not None:
        raise ValueError("sample_count and samples_per_edge are mutually exclusive")
    if samples_per_edge is not None:
        per_edge = _positive_integer(samples_per_edge, "samples_per_edge")
        allocations = (per_edge,) * len(plan.priority_edges_m)
    else:
        total = _positive_integer(
            plan.broad_candidate_budget if sample_count is None else sample_count,
            "sample_count",
        )
        quotient, remainder = divmod(total, len(plan.priority_edges_m))
        allocations = tuple(
            quotient + int(index < remainder)
            for index in range(len(plan.priority_edges_m))
        )
    candidates: list[dict[str, Any]] = []
    for edge_m, count in zip(plan.priority_edges_m, allocations):
        if count == 0:
            continue
        seed = _edge_seed(plan, edge_m, method)
        if method == "lhs":
            rows = _latin_hypercube(count, _SAMPLE_DIMENSIONS, seed=seed)
        else:
            rows = _grid_samples(count, _SAMPLE_DIMENSIONS, seed=seed)
        for sample_index, row in enumerate(rows):
            if (
                include_anchor
                and sample_index == 0
                and math.isclose(
                    edge_m, plan.anchor_edge_m, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                parameters = _anchor_parameters(plan)
            else:
                parameters = _sample_parameters(row, plan=plan)
            candidates.append(
                _materialize_candidate(
                    base_config,
                    edge_m=edge_m,
                    parameters=parameters,
                    plan=plan,
                    validator=validator,
                )
            )
    return candidates


def _metrics(value: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = value.get("summary")
    if isinstance(summary, Mapping):
        metrics = summary.get("metrics")
        return metrics if isinstance(metrics, Mapping) else {}
    metrics = value.get("metrics")
    if isinstance(metrics, Mapping):
        return metrics
    return value


def _finger_values(
    metrics: Mapping[str, Any], name: str
) -> dict[str, float]:
    raw = metrics.get(name)
    if not isinstance(raw, Mapping):
        raw = {}
    values: dict[str, float] = {}
    for finger in ACTIVE_FINGERS:
        try:
            value = float(raw.get(finger, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        values[finger] = value if math.isfinite(value) else 0.0
    return values


def continuation_gate_evidence(
    result_or_metrics: Mapping[str, Any], *, timestep_s: float = 0.001
) -> dict[str, Any]:
    """Return the auditable rescue continuation decision and its evidence."""

    timestep = _finite(timestep_s, "timestep_s")
    if timestep <= 0.0:
        raise ValueError("timestep_s must be positive")
    metrics = _metrics(result_or_metrics)
    duties = _finger_values(metrics, "verify_target_face_effective_duty")
    forces = _finger_values(metrics, "verify_peak_target_face_force_n")
    duty_fingers = tuple(
        finger for finger in ACTIVE_FINGERS if duties[finger] >= 0.10
    )
    remaining = tuple(
        finger for finger in ACTIVE_FINGERS if finger not in duty_fingers
    )
    if len(duty_fingers) >= 2:
        third_force = (
            min(forces.values())
            if not remaining
            else min(forces[finger] for finger in remaining)
        )
    else:
        third_force = 0.0
    two_duty_plus_third_force = len(duty_fingers) >= 2 and third_force >= 0.02

    duration_value = metrics.get(
        "verify_max_consecutive_all_gate_s",
        metrics.get("verify_max_consecutive_gate_s"),
    )
    if duration_value is None:
        steps_value = metrics.get(
            "verify_max_consecutive_all_gate_steps",
            metrics.get("verify_max_consecutive_gate_steps", 0),
        )
        try:
            steps = max(0, int(steps_value))
        except (TypeError, ValueError):
            steps = 0
        all_gate_duration_s = steps * timestep
    else:
        try:
            all_gate_duration_s = float(duration_value)
        except (TypeError, ValueError):
            all_gate_duration_s = 0.0
        if not math.isfinite(all_gate_duration_s) or all_gate_duration_s < 0.0:
            all_gate_duration_s = 0.0
    all_gate_10ms = all_gate_duration_s + 1e-12 >= 0.010
    return {
        "continue": bool(two_duty_plus_third_force or all_gate_10ms),
        "two_duty_plus_third_force": bool(two_duty_plus_third_force),
        "all_gate_10ms": bool(all_gate_10ms),
        "duty_fingers": list(duty_fingers),
        "third_finger_peak_target_force_n": float(third_force),
        "all_gate_duration_s": float(all_gate_duration_s),
        "thresholds": {
            "verify_duty": 0.10,
            "third_finger_peak_target_force_n": 0.02,
            "all_gate_duration_s": 0.010,
        },
    }


def pose_rescue_continuation_gate(
    result_or_metrics: Mapping[str, Any], *, timestep_s: float = 0.001
) -> bool:
    return bool(
        continuation_gate_evidence(
            result_or_metrics, timestep_s=timestep_s
        )["continue"]
    )


def _stage_status(result: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        return {}
    status = summary.get("stage_status")
    return status if isinstance(status, Mapping) else {}


def _safe_metric(metrics: Mapping[str, Any], name: str, default: float) -> float:
    try:
        value = float(metrics.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def pose_rescue_candidate_rank(result: Mapping[str, Any]) -> tuple[float, ...]:
    """Worker-independent grasp rank headed by schema-v3 VERIFY evidence.

    Manipulation and full-pass booleans are deliberately absent: this rescue
    campaign stops at stable-grasp discovery and cannot promote later-stage
    claims even if an injected generic runner happens to populate them.
    """

    try:
        candidate_id = int(result["candidate_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("candidate result requires an integer candidate_id") from error
    status = _stage_status(result)
    grasp_success = bool(status.get("grasp_success", False))
    metrics = _metrics(result)
    duties = _finger_values(metrics, "verify_target_face_effective_duty")
    near_finger_count = sum(value >= 0.10 for value in duties.values())
    continuation = pose_rescue_continuation_gate(metrics)
    forbidden = _safe_metric(metrics, "forbidden_contact_steps", math.inf)
    force = _safe_metric(
        metrics, "peak_total_distal_contact_force_n", math.inf
    )
    saturation = _safe_metric(
        metrics, "actuator_saturation_fraction", math.inf
    )
    return (
        float(grasp_success),
        float(continuation),
        float(near_finger_count),
        *v3_verify_near_miss_rank(dict(metrics)),
        -forbidden,
        -force,
        -saturation,
        -float(candidate_id),
    )


def rank_pose_rescue_results(
    results: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    materialized = tuple(results)
    try:
        identifiers = tuple(int(item["candidate_id"]) for item in materialized)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("every result requires an integer candidate_id") from error
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate_id values must be unique")
    return tuple(
        sorted(materialized, key=pose_rescue_candidate_rank, reverse=True)
    )


def select_local_refinement_parents(
    results: Iterable[dict[str, Any]],
    *,
    plan: PoseRescuePlan = DEFAULT_PLAN,
) -> tuple[dict[str, Any], ...]:
    """Select only stable-grasp or continuation-worthy parents for local work."""

    eligible: list[dict[str, Any]] = []
    for result in results:
        status = _stage_status(result)
        if bool(status.get("grasp_success", False)) or pose_rescue_continuation_gate(
            result
        ):
            eligible.append(result)
    return rank_pose_rescue_results(eligible)[: plan.local_parent_budget]


def _clip(value: float, bounds: Sequence[float]) -> float:
    return float(min(max(value, float(bounds[0])), float(bounds[1])))


def _local_parameters(
    parent: Mapping[str, Any],
    row: np.ndarray,
    *,
    plan: PoseRescuePlan,
) -> dict[str, Any]:
    sampled = _sample_parameters(row, plan=plan)
    fraction = plan.local_radius_fraction
    parent_cube = cube_position_in_root(parent)
    cube = np.asarray(
        [
            _clip(
                (1.0 - fraction) * _clip(parent_cube[index], plan.cube_position_in_root_m[axis])
                + fraction * sampled["cube_in_root_m"][index],
                plan.cube_position_in_root_m[axis],
            )
            for index, axis in enumerate(("x", "y", "z"))
        ],
        dtype=np.float64,
    )
    try:
        parent_rpy = tuple(float(value) for value in parent["hand_pose"]["rpy_deg"])
        parent_cube_yaw = float(parent["cube"]["rpy_deg"][2])
        parent_targets = parent["control"]["grasp_targets_rad"]
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError("local parent requires a schema-v3 pose and grasp targets") from error
    if len(parent_rpy) != 3 or set(parent_targets) != set(ACTIVE_ACTUATORS):
        raise ValueError("local parent pose or grasp targets are incomplete")
    roll = (1.0 - fraction) * _clip(parent_rpy[0], plan.hand_roll_deg) + fraction * sampled[
        "hand_rpy_deg"
    ][0]
    pitch_continuous = (1.0 - fraction) * _clip(
        parent_rpy[1],
        (plan.hand_pitch_values_deg[0], plan.hand_pitch_values_deg[-1]),
    ) + fraction * sampled["hand_rpy_deg"][1]
    pitch = min(plan.hand_pitch_values_deg, key=lambda value: abs(value - pitch_continuous))
    yaw = (1.0 - fraction) * _clip(parent_rpy[2], plan.hand_yaw_deg) + fraction * sampled[
        "hand_rpy_deg"
    ][2]
    cube_yaw = (1.0 - fraction) * _clip(parent_cube_yaw, plan.cube_yaw_deg) + fraction * float(
        sampled["cube_yaw_deg"]
    )

    targets: dict[str, float] = {}
    for name in THUMB_INDEX_ACTUATORS:
        absolute_bounds = (
            plan.anchor_targets_rad[name] + plan.narrow_target_delta_rad[name][0],
            plan.anchor_targets_rad[name] + plan.narrow_target_delta_rad[name][1],
        )
        centre = _clip(float(parent_targets[name]), absolute_bounds)
        targets[name] = (1.0 - fraction) * centre + fraction * sampled[
            "grasp_targets_rad"
        ][name]
    for name, bounds in (
        (MIDDLE_JOINT1, plan.middle_joint1_rad),
        (MIDDLE_JOINT2, plan.middle_joint2_rad),
    ):
        centre = _clip(float(parent_targets[name]), bounds)
        targets[name] = (1.0 - fraction) * centre + fraction * sampled[
            "grasp_targets_rad"
        ][name]
    return {
        "cube_in_root_m": cube,
        "hand_rpy_deg": (roll, pitch, yaw),
        "cube_yaw_deg": cube_yaw,
        "grasp_targets_rad": targets,
    }


def generate_local_pose_rescue_candidates(
    results: Iterable[dict[str, Any]],
    *,
    plan: PoseRescuePlan = DEFAULT_PLAN,
    validator: Validator | None = None,
) -> list[dict[str, Any]]:
    """Generate the budgeted 32-point neighbourhood for each top parent."""

    candidates: list[dict[str, Any]] = []
    for parent_rank, result in enumerate(
        select_local_refinement_parents(results, plan=plan)
    ):
        try:
            candidate_id = int(result["candidate_id"])
            parent = result["config"]
            edge_m = _finite(parent["cube"]["edge_m"], "parent cube.edge_m")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("local result requires candidate_id and config") from error
        if not any(
            math.isclose(edge_m, declared, rel_tol=0.0, abs_tol=1e-12)
            for declared in plan.priority_edges_m
        ):
            raise ValueError("local parent edge must be one of the rescue priority sizes")
        seed = (
            plan.seed
            + 300_000_000
            + candidate_id * 1009
            + parent_rank * 1_000_003
        )
        rows = _latin_hypercube(
            plan.local_samples_per_parent, _SAMPLE_DIMENSIONS, seed=seed
        )
        for row in rows:
            candidates.append(
                _materialize_candidate(
                    parent,
                    edge_m=edge_m,
                    parameters=_local_parameters(parent, row, plan=plan),
                    plan=plan,
                    validator=validator,
                )
            )
    return candidates


CandidateRunner = Callable[
    [list[tuple[int, dict[str, Any]]], int], list[dict[str, Any]]
]


def run_pose_rescue_batch(
    configs: Sequence[dict[str, Any]],
    *,
    run_candidates: CandidateRunner,
    workers: int,
    first_candidate_id: int = 0,
) -> tuple[dict[str, Any], ...]:
    """Invoke a caller-owned runner once and enforce its ID/config contract."""

    if not callable(run_candidates):
        raise TypeError("run_candidates must be callable")
    worker_count = _positive_integer(workers, "workers")
    if (
        not isinstance(first_candidate_id, int)
        or isinstance(first_candidate_id, bool)
        or first_candidate_id < 0
    ):
        raise ValueError("first_candidate_id must be a non-negative integer")
    payload = [
        (first_candidate_id + index, copy.deepcopy(config))
        for index, config in enumerate(configs)
    ]
    expected = {candidate_id: config for candidate_id, config in payload}
    returned = run_candidates(payload, worker_count) if payload else []
    if not isinstance(returned, list):
        raise RuntimeError("pose-rescue runner must return a list")
    by_id: dict[int, dict[str, Any]] = {}
    for result in returned:
        if not isinstance(result, dict):
            raise RuntimeError("pose-rescue runner returned a non-mapping result")
        try:
            candidate_id = int(result["candidate_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "pose-rescue runner returned an invalid candidate_id"
            ) from error
        if candidate_id not in expected or candidate_id in by_id:
            raise RuntimeError("pose-rescue runner returned unknown or duplicate IDs")
        if result.get("config") != expected[candidate_id]:
            raise RuntimeError("pose-rescue runner rebound a candidate config")
        by_id[candidate_id] = result
    if set(by_id) != set(expected):
        raise RuntimeError("pose-rescue runner did not return every submitted ID")
    return tuple(by_id[candidate_id] for candidate_id in sorted(by_id))


def _explicit_grasp_success(result: Mapping[str, Any]) -> bool:
    status = _stage_status(result)
    value = status.get("grasp_success", False)
    return value is True or isinstance(value, np.bool_) and bool(value)


def _empty_probe(candidate_id: int) -> dict[str, Any]:
    return {
        "candidate_id": int(candidate_id),
        "passes": 0,
        "trial_count": 0,
        "trials": [],
    }


def _scoped_rescue_result(
    result: Mapping[str, Any], *, campaign_has_stable_grasp: bool
) -> dict[str, Any]:
    """Cap one generic simulation observation at the rescue campaign scope."""

    scoped = copy.deepcopy(dict(result))
    observed_summary = scoped.get("summary")
    if not isinstance(observed_summary, dict):
        observed_summary = {}
    grasp_success = _explicit_grasp_success(scoped)
    scoped["observed_grasp_evidence"] = {
        "grasp_success": grasp_success,
        "failed_checks": [
            str(value) for value in observed_summary.get("failed_checks", ())
        ],
        "metrics": copy.deepcopy(observed_summary.get("metrics", {})),
    }
    summary = copy.deepcopy(observed_summary)
    summary["passed"] = False
    failed_checks = [str(value) for value in summary.get("failed_checks", ())]
    scope_check = "pose_rescue_scope_excludes_manipulation_validation"
    if scope_check not in failed_checks:
        failed_checks.append(scope_check)
    summary["failed_checks"] = failed_checks
    checks = summary.get("checks")
    checks = copy.deepcopy(checks) if isinstance(checks, Mapping) else {}
    checks[scope_check] = False
    summary["checks"] = checks
    summary["stage_status"] = {
        "grasp_success": grasp_success,
        "manipulation_success": False,
        "full_success": False,
    }
    scoped["summary"] = summary
    scoped["material_policy"] = "fixed_20g_pose_rescue"
    scoped["local_perturbation_probe"] = _empty_probe(
        int(scoped["candidate_id"])
    )
    classification = (
        "pose_rescue_stable_grasp" if grasp_success else "not_validated"
    )
    scoped["config"]["experiment_status"] = {
        "classification": classification,
        "passed": False,
        "grasp_success": grasp_success,
        "manipulation_success": False,
        "full_success": False,
        "fixed_mass_discovery_passed": False,
        "constant_density_passed": False,
        "campaign_has_stable_grasp": bool(campaign_has_stable_grasp),
        "stop_reason": (
            "stable_grasp_found_rescue_scope_complete"
            if grasp_success
            else "candidate_did_not_acquire_stable_grasp"
        ),
        "note": (
            "A stable grasp was observed. This rescue campaign does not test "
            "manipulation or constant-density validation."
            if grasp_success
            else "This candidate did not acquire the declared stable grasp."
        ),
    }
    return scoped


def _effective_positive_count(
    value: int | None, default: int, label: str
) -> int:
    selected = default if value is None else value
    return _positive_integer(selected, label)


def tune_relative_pose_rescue(
    config: dict[str, Any],
    *,
    workers: int,
    seed: int,
    run_candidates: CandidateRunner,
    dynamic_candidate_count: int | None = None,
    local_refine_seed_count: int | None = None,
    local_refine_per_seed: int | None = None,
    kinematic_samples_per_pitch: int | None = None,
    final_candidate_count: int | None = None,
    perturbations_per_final: int | None = None,
    fallback_physics_count: int | None = None,
    fallback_kinematic_samples_per_pitch: int | None = None,
    legacy_parameters: Mapping[str, Any] | None = None,
    plan: PoseRescuePlan = DEFAULT_PLAN,
    method: str = "lhs",
) -> dict[str, Any]:
    """Run broad and continuation-driven local stable-grasp discovery only.

    The generic runner remains caller-owned.  Later manipulation, material and
    perturbation stages are intentionally not dispatched.  Overrides for those
    inapplicable stages are retained in the report instead of being silently
    reinterpreted as rescue work.
    """

    validate_config(config)
    worker_count = _positive_integer(workers, "workers")
    if seed != plan.seed:
        raise ValueError(
            f"relative pose rescue seed is fixed at {plan.seed}; received {seed}"
        )
    broad_count = _effective_positive_count(
        dynamic_candidate_count,
        plan.broad_candidate_budget,
        "dynamic_candidate_count",
    )
    local_parent_count = _effective_positive_count(
        local_refine_seed_count,
        plan.local_parent_budget,
        "local_refine_seed_count",
    )
    local_per_parent = _effective_positive_count(
        local_refine_per_seed,
        plan.local_samples_per_parent,
        "local_refine_per_seed",
    )
    maximum_simulations = broad_count + local_parent_count * local_per_parent
    effective_plan = replace(
        plan,
        broad_candidate_budget=broad_count,
        local_parent_budget=local_parent_count,
        local_samples_per_parent=local_per_parent,
        dynamic_candidate_budget=maximum_simulations,
        continuation_candidate_budget=min(
            plan.continuation_candidate_budget, maximum_simulations
        ),
    )

    broad_configs = generate_pose_rescue_candidates(
        config,
        plan=effective_plan,
        method=method,
        sample_count=broad_count,
        validator=validate_config,
    )
    broad_results = list(
        run_pose_rescue_batch(
            broad_configs,
            run_candidates=run_candidates,
            workers=worker_count,
            first_candidate_id=0,
        )
    )
    for result in broad_results:
        result["search_stage"] = "pose_rescue_broad"
        result["material_policy"] = "fixed_20g_pose_rescue"

    local_configs = generate_local_pose_rescue_candidates(
        broad_results,
        plan=effective_plan,
        validator=validate_config,
    )
    local_results = list(
        run_pose_rescue_batch(
            local_configs,
            run_candidates=run_candidates,
            workers=worker_count,
            first_candidate_id=len(broad_results),
        )
    )
    for result in local_results:
        result["search_stage"] = "pose_rescue_local"
        result["material_policy"] = "fixed_20g_pose_rescue"

    all_results = broad_results + local_results
    ranked = rank_pose_rescue_results(all_results)
    stable_results = [item for item in all_results if _explicit_grasp_success(item)]
    broad_stable_count = sum(
        _explicit_grasp_success(item) for item in broad_results
    )
    local_stable_count = sum(
        _explicit_grasp_success(item) for item in local_results
    )
    grasp_success = bool(stable_results)
    classification = (
        "pose_rescue_stable_grasp" if grasp_success else "not_validated"
    )
    stop_reason = (
        "stable_grasp_found_rescue_scope_complete"
        if grasp_success
        else "no_stable_grasp_acquired"
    )
    scoped_ranked = tuple(
        _scoped_rescue_result(
            item, campaign_has_stable_grasp=grasp_success
        )
        for item in ranked
    )
    best = copy.deepcopy(scoped_ranked[0])

    manifest = pose_rescue_manifest(effective_plan)
    manifest["effective_overrides"] = {
        "dynamic_candidate_count": dynamic_candidate_count,
        "local_refine_seed_count": local_refine_seed_count,
        "local_refine_per_seed": local_refine_per_seed,
    }
    manifest["not_applicable_overrides"] = {
        "kinematic_samples_per_pitch": kinematic_samples_per_pitch,
        "final_candidate_count": final_candidate_count,
        "perturbations_per_final": perturbations_per_final,
        "fallback_physics_count": fallback_physics_count,
        "fallback_kinematic_samples_per_pitch": (
            fallback_kinematic_samples_per_pitch
        ),
        "handling": (
            "reported_but_not_executed; pose rescue has no static screen, "
            "finalist probe, fallback physics, manipulation, or density stage"
        ),
    }
    manifest["legacy_parameters"] = {
        **({} if legacy_parameters is None else dict(legacy_parameters)),
        "handling": "reported_but_not_used_by_schema_v3_pose_rescue",
    }
    campaign_status = copy.deepcopy(best["config"]["experiment_status"])
    campaign_status.update(
        {
            "classification": classification,
            "grasp_success": grasp_success,
            "campaign_has_stable_grasp": grasp_success,
            "stop_reason": stop_reason,
        }
    )
    return {
        "experiment_id": str(config.get("experiment_id", "")),
        "campaign_kind": RESCUE_STRATEGY,
        "campaign_classification": classification,
        "campaign_status": campaign_status,
        "seed": int(seed),
        "workers": worker_count,
        "campaign_manifest": manifest,
        "candidate_count": len(all_results),
        "simulation_count": len(all_results),
        "perturbation_probe_count": 0,
        "passing_candidates": 0,
        "stable_grasp_candidate_count": len(stable_results),
        "broad_candidate_count": len(broad_results),
        "broad_stable_grasp_count": int(broad_stable_count),
        "local_candidate_count": len(local_results),
        "local_stable_grasp_count": int(local_stable_count),
        "grasp_success": grasp_success,
        "manipulation_success": False,
        "fixed_mass_success": False,
        "constant_density_success": False,
        "nominal_success": False,
        "stop_reason": stop_reason,
        "best_fixed_mass": None,
        "best": best,
        "top_candidates": [copy.deepcopy(item) for item in scoped_ranked[:20]],
        "local_perturbation_probes": [],
    }


def pose_rescue_manifest(
    plan: PoseRescuePlan = DEFAULT_PLAN,
) -> dict[str, Any]:
    """Return a JSON-safe source, invariant, continuation and budget manifest."""

    return {
        "schema_version": 1,
        "strategy": RESCUE_STRATEGY,
        "source_provenance": {
            "experiment_id": SOURCE_EXPERIMENT_ID,
            "artifact": SOURCE_TUNE_RESULTS,
            "artifact_sha256": SOURCE_TUNE_RESULTS_SHA256,
            "source_candidate_id": plan.source_candidate_id,
            "source_stage": "exact_62mm_static_screen",
            "source_stop_reason": "no_stable_grasp_acquired",
            "source_kinematic_sample_count": 1_295_000,
            "source_simulation_count": 1_792,
            "focused_probe": {
                "provenance_kind": "follow_up_joint_rescue_measurement",
                "anchor_candidate_id": plan.source_candidate_id,
                "candidate_count": 256,
                "stable_grasp_count": 18,
                "stable_grasp_rate": 18 / 256,
            },
        },
        "plan": plan.as_dict(),
        "budget": {
            "size_count": len(plan.priority_edges_m),
            "lhs_candidate_count": plan.lhs_candidate_count,
            "grid_candidate_count": plan.grid_candidate_count,
            "sampling_methods_are_alternatives": True,
            "broad_candidate_budget": plan.broad_candidate_budget,
            "local_parent_budget": plan.local_parent_budget,
            "local_samples_per_parent": plan.local_samples_per_parent,
            "local_radius_fraction": plan.local_radius_fraction,
            "local_candidate_budget": plan.local_candidate_budget,
            "dynamic_candidate_budget": plan.dynamic_candidate_budget,
            "continuation_candidate_budget": plan.continuation_candidate_budget,
        },
        "invariants": {
            "root_pose_runtime_static": True,
            "manipulation_delta_rad_zero": True,
            "target_faces": dict(plan.target_faces),
            "sizes_run_in_priority_order": True,
            "runner_is_injected": True,
            "simulation_loop_owned_elsewhere": True,
        },
        "continuation_policy": {
            "branch_a": {
                "minimum_fingers_with_verify_duty": 2,
                "minimum_verify_duty": 0.10,
                "remaining_finger_min_peak_target_force_n": 0.02,
            },
            "branch_b": {"minimum_all_gate_duration_s": 0.010},
            "operator": "branch_a OR branch_b",
        },
    }


# Small compatibility aliases keep later orchestration wiring unsurprising.
candidate_rank = pose_rescue_candidate_rank
continuation_gate = pose_rescue_continuation_gate
deterministic_rank_results = rank_pose_rescue_results
campaign_manifest = pose_rescue_manifest
generate_candidates = generate_pose_rescue_candidates


__all__ = [
    "CandidateRunner",
    "DEFAULT_PLAN",
    "FIXED_SEED",
    "MIDDLE_JOINT1",
    "MIDDLE_JOINT2",
    "PoseRescuePlan",
    "RESCUE_STRATEGY",
    "SOURCE_EXPERIMENT_ID",
    "SOURCE_TUNE_RESULTS",
    "SOURCE_TUNE_RESULTS_SHA256",
    "TARGET_FACES",
    "campaign_manifest",
    "candidate_rank",
    "continuation_gate",
    "continuation_gate_evidence",
    "cube_position_in_root",
    "deterministic_rank_results",
    "generate_candidates",
    "generate_local_pose_rescue_candidates",
    "generate_pose_rescue_candidates",
    "make_anchor_candidate",
    "pose_rescue_candidate_rank",
    "pose_rescue_continuation_gate",
    "pose_rescue_manifest",
    "rank_pose_rescue_results",
    "run_pose_rescue_batch",
    "select_local_refinement_parents",
    "tune_relative_pose_rescue",
]
