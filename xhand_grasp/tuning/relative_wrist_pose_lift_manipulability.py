"""Focused lift-manipulability search around authenticated schema-v11 grasps.

This module is deliberately independent of the historical v1--v11 campaign
orchestrators.  It never rewrites sealed evidence and it does not classify a
static pose as a grasp.  Its responsibilities are narrower:

* authenticate one measured grasp from a catalog or explicit artifact paths;
* quantify whether the supported grasp is ready to become a free-body grasp;
* generate deterministic, per-size/per-orbit relative-wrist neighborhoods;
* retain candidates with per-cell and per-size quotas; and
* expose an injectable bridge to the existing checkpoint-probe/full-reset
  manipulation runner.

Every published manipulation result must still be rerun from the initial
no-contact state by the existing runner.  Checkpoint branches remain search
evidence only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import mujoco
import numpy as np

from xhand_tactile import TactileReader

from ..actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
    authenticated_catalog_artifact_paths,
)
from ..artifacts import REPO_ROOT, file_sha256
from ..config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    INACTIVE_ACTUATORS,
    SCRIPT_DIR,
    load_config,
    validate_config,
)
from ..experiment import resolve_experiment
from ..grasp_pose import canonical_sha256, grasp_pose_id
from ..relative_wrist_pose import (
    rotation_matrix_to_rpy_degrees,
    transform_relative_wrist_pose,
)
from ..scene import build_model, rpy_degrees_to_rotation_matrix
from .actual_contact_grasp_pose_measured import extract_measured_grasp_qpos
from .pose_preserving_seed_campaign import canonical_sha256 as payload_sha256
from .relative_wrist_pose_search import (
    _TARGET_GAP_M as CANONICAL_TARGET_GAP_M,
    NON_THUMB_ACTUATORS,
    RelativeWristDLSResult,
    RelativeWristDLSSettings,
    RelativeWristPoseSearchPolicy,
    RelativeWristVariables,
    build_actual_contact_trial_evaluator,
    materialize_relative_wrist_candidate,
    model_joint_bounds,
    solve_orientation_aware_dls,
)


CAMPAIGN_KIND = "relative_wrist_pose_lift_manipulability"
MANIFEST_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1
SOURCE_SCHEMA_VERSION = 1
READINESS_SCHEMA_VERSION = 1
CANDIDATE_SCHEMA_VERSION = 1
GENERATION_SCHEMA_VERSION = 1
CANDIDATE_ID_BASE = 510_000_000_000_000
MAX_STATIC_CANDIDATE_ID = 576_460_752_303_422
DEFAULT_SEED = 20260821
DEFAULT_EDGES_M = (0.087, 0.089, 0.091, 0.093)
DEFAULT_ORBITS_DEG = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0)
_V11_TEMPLATE = (
    SCRIPT_DIR
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift.json"
)
_FACE_NORMAL = MappingProxyType(
    {
        "+X": np.asarray((1.0, 0.0, 0.0)),
        "-X": np.asarray((-1.0, 0.0, 0.0)),
        "+Y": np.asarray((0.0, 1.0, 0.0)),
        "-Y": np.asarray((0.0, -1.0, 0.0)),
        "+Z": np.asarray((0.0, 0.0, 1.0)),
        "-Z": np.asarray((0.0, 0.0, -1.0)),
    }
)
_EPS = 1e-12


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_tuple(value: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(_finite(item, label) for item in value)
    if not result or any(item <= 0.0 for item in result):
        raise ValueError(f"{label} must contain positive values")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must be unique and increasing")
    return result


def _orbit_tuple(value: Sequence[float]) -> tuple[float, ...]:
    result = tuple(_finite(item, "clockwise_orbits_deg") for item in value)
    if (
        not result
        or result[0] != 0.0
        or any(item < 0.0 for item in result)
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(
            "clockwise_orbits_deg must start at zero and be unique/increasing"
        )
    return result


def _vector3(value: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain three finite values")
    return result.copy()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _confined_artifact(base: Path, value: Any, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a confined relative path")
    resolved = (base / relative).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"{label} escapes its catalog directory")
    return resolved


@dataclass(frozen=True, slots=True)
class LiftManipulabilityPolicy:
    """Versioned deterministic geometry and readiness policy."""

    edges_m: tuple[float, ...] = DEFAULT_EDGES_M
    clockwise_orbits_deg: tuple[float, ...] = DEFAULT_ORBITS_DEG
    samples_per_cell: int = 128
    retained_per_cell: int = 4
    minimum_per_edge: int = 8
    selected_total: int = 32
    seed: int = DEFAULT_SEED
    non_thumb_joint_radius_rad: float = 0.06
    root_radius_m: tuple[float, float, float] = (0.004, 0.006, 0.006)
    wrist_radius_deg: tuple[float, float, float] = (2.5, 2.5, 2.0)
    minimum_force_balance_ratio: float = 0.70
    minimum_friction_margin_n: float = 0.0
    maximum_line_moment_n_m: float = 0.003
    hard_height_spread_m: float = 0.005
    soft_height_target_m: float = 0.002
    minimum_tactile_path_margin_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges_m", _positive_tuple(self.edges_m, "edges_m"))
        object.__setattr__(
            self,
            "clockwise_orbits_deg",
            _orbit_tuple(self.clockwise_orbits_deg),
        )
        for name in (
            "samples_per_cell",
            "retained_per_cell",
            "minimum_per_edge",
            "selected_total",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.retained_per_cell > self.samples_per_cell:
            raise ValueError("retained_per_cell cannot exceed samples_per_cell")
        available_per_edge = self.retained_per_cell * len(self.clockwise_orbits_deg)
        if self.minimum_per_edge > available_per_edge:
            raise ValueError("minimum_per_edge exceeds retained candidates per edge")
        if self.selected_total < self.minimum_per_edge * len(self.edges_m):
            raise ValueError("selected_total cannot satisfy every per-edge quota")
        available_total = available_per_edge * len(self.edges_m)
        if self.selected_total > available_total:
            raise ValueError("selected_total exceeds the retained candidate budget")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        for name in (
            "non_thumb_joint_radius_rad",
            "minimum_force_balance_ratio",
            "maximum_line_moment_n_m",
            "hard_height_spread_m",
            "soft_height_target_m",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.minimum_force_balance_ratio <= 1.0:
            raise ValueError("minimum_force_balance_ratio must lie in (0, 1]")
        if self.soft_height_target_m > self.hard_height_spread_m:
            raise ValueError("soft height target cannot exceed the hard threshold")
        _finite(self.minimum_friction_margin_n, "minimum_friction_margin_n")
        _finite(self.minimum_tactile_path_margin_m, "minimum_tactile_path_margin_m")
        root = _vector3(self.root_radius_m, "root_radius_m")
        wrist = _vector3(self.wrist_radius_deg, "wrist_radius_deg")
        if np.any(root <= 0.0) or np.any(wrist <= 0.0):
            raise ValueError("root/wrist search radii must be positive")
        object.__setattr__(self, "root_radius_m", tuple(float(v) for v in root))
        object.__setattr__(self, "wrist_radius_deg", tuple(float(v) for v in wrist))

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["edges_m"] = list(self.edges_m)
        value["clockwise_orbits_deg"] = list(self.clockwise_orbits_deg)
        value["root_radius_m"] = list(self.root_radius_m)
        value["wrist_radius_deg"] = list(self.wrist_radius_deg)
        return value


@dataclass(frozen=True, slots=True)
class SupportModeObservation:
    target_face_force_n: tuple[float, float, float]
    contact_centroid_cube_m: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    target_faces: tuple[str, str, str]
    height_spread_p95_m: float
    tactile_nearest_distance_p95_m: tuple[float, float, float]
    maximum_taxel_assignment_distance_m: float
    support_retained: bool

    def __post_init__(self) -> None:
        force = np.asarray(self.target_face_force_n, dtype=np.float64)
        centroid = np.asarray(self.contact_centroid_cube_m, dtype=np.float64)
        tactile = np.asarray(self.tactile_nearest_distance_p95_m, dtype=np.float64)
        if force.shape != (3,) or np.any(force < 0.0) or not np.isfinite(force).all():
            raise ValueError("target_face_force_n must contain three non-negative values")
        if centroid.shape != (3, 3) or not np.isfinite(centroid).all():
            raise ValueError("contact_centroid_cube_m must have shape (3, 3)")
        if tactile.shape != (3,) or np.any(tactile < 0.0) or not np.isfinite(tactile).all():
            raise ValueError("tactile nearest distances must be non-negative")
        if len(self.target_faces) != 3 or any(face not in _FACE_NORMAL for face in self.target_faces):
            raise ValueError("target_faces must contain three physical cube faces")
        if _finite(self.height_spread_p95_m, "height_spread_p95_m") < 0.0:
            raise ValueError("height_spread_p95_m must be non-negative")
        if _finite(
            self.maximum_taxel_assignment_distance_m,
            "maximum_taxel_assignment_distance_m",
        ) <= 0.0:
            raise ValueError("maximum taxel assignment distance must be positive")
        if not isinstance(self.support_retained, bool):
            raise ValueError("support_retained must be boolean")


@dataclass(frozen=True, slots=True)
class SupportModeReadiness:
    passed: bool
    checks: Mapping[str, bool]
    force_balance_ratio: float
    thumb_side_normal_force_n: float
    opposed_side_normal_force_n: float
    friction_capacity_n: float
    friction_margin_n: float
    line_of_action_moment_n_m: float
    height_spread_p95_m: float
    height_soft_excess_m: float
    tactile_path_margin_m: float
    normalized_max_violation: float
    policy: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.policy, Mapping):
            raise TypeError("support readiness policy must be a mapping")
        object.__setattr__(
            self,
            "policy",
            MappingProxyType(copy.deepcopy(dict(self.policy))),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "support_mode_readiness_schema_version": READINESS_SCHEMA_VERSION,
            "passed": self.passed,
            "checks": dict(self.checks),
            "force_balance_ratio": self.force_balance_ratio,
            "thumb_side_normal_force_n": self.thumb_side_normal_force_n,
            "opposed_side_normal_force_n": self.opposed_side_normal_force_n,
            "friction_capacity_n": self.friction_capacity_n,
            "friction_margin_n": self.friction_margin_n,
            "line_of_action_moment_n_m": self.line_of_action_moment_n_m,
            "height_spread_p95_m": self.height_spread_p95_m,
            "height_soft_excess_m": self.height_soft_excess_m,
            "tactile_path_margin_m": self.tactile_path_margin_m,
            "normalized_max_violation": self.normalized_max_violation,
            "policy": copy.deepcopy(dict(self.policy)),
            "policy_sha256": payload_sha256(dict(self.policy)),
        }


def evaluate_support_mode_readiness(
    observation: SupportModeObservation,
    *,
    mass_kg: float,
    sliding_friction: float,
    gravity_m_s2: Sequence[float],
    policy: LiftManipulabilityPolicy | None = None,
) -> SupportModeReadiness:
    """Evaluate whether a supported pinch has credible free-body margin."""

    resolved = policy or LiftManipulabilityPolicy()
    mass = _finite(mass_kg, "mass_kg")
    friction = _finite(sliding_friction, "sliding_friction")
    gravity = _vector3(gravity_m_s2, "gravity_m_s2")
    if mass <= 0.0 or friction <= 0.0 or np.linalg.norm(gravity) <= _EPS:
        raise ValueError("mass/friction/gravity must be positive and non-zero")
    forces = np.asarray(observation.target_face_force_n, dtype=np.float64)
    centroids = np.asarray(observation.contact_centroid_cube_m, dtype=np.float64)
    normals = np.asarray([_FACE_NORMAL[face] for face in observation.target_faces])
    if not np.allclose(normals[1], normals[2], atol=0.0, rtol=0.0):
        raise ValueError("index and middle target faces must be identical")
    if not np.allclose(normals[0], -normals[1], atol=0.0, rtol=0.0):
        raise ValueError("thumb target face must be opposite index/middle")
    thumb_force = float(forces[0])
    opposed_force = float(forces[1] + forces[2])
    strongest = max(thumb_force, opposed_force, _EPS)
    balance = min(thumb_force, opposed_force) / strongest
    capacity = 2.0 * friction * min(thumb_force, opposed_force)
    weight = mass * float(np.linalg.norm(gravity))
    friction_margin = capacity - weight
    # Contact force on the cube points opposite its outward face normal.
    force_on_cube = -forces[:, np.newaxis] * normals
    moment = float(np.linalg.norm(np.sum(np.cross(centroids, force_on_cube), axis=0)))
    tactile_margin = float(
        observation.maximum_taxel_assignment_distance_m
        - max(observation.tactile_nearest_distance_p95_m)
    )
    checks = {
        "support_retained": observation.support_retained,
        "opposed_normal_force_balanced": (
            balance + _EPS >= resolved.minimum_force_balance_ratio
        ),
        "friction_lift_margin": (
            friction_margin + _EPS >= resolved.minimum_friction_margin_n
        ),
        "line_of_action_moment_small": (
            moment <= resolved.maximum_line_moment_n_m + _EPS
        ),
        "contact_height_within_hard_limit": (
            observation.height_spread_p95_m
            <= resolved.hard_height_spread_m + _EPS
        ),
        "tactile_path_has_margin": (
            tactile_margin + _EPS >= resolved.minimum_tactile_path_margin_m
        ),
    }
    violations = (
        max(0.0, resolved.minimum_force_balance_ratio - balance)
        / resolved.minimum_force_balance_ratio,
        max(0.0, resolved.minimum_friction_margin_n - friction_margin)
        / max(weight, _EPS),
        max(0.0, moment - resolved.maximum_line_moment_n_m)
        / resolved.maximum_line_moment_n_m,
        max(0.0, observation.height_spread_p95_m - resolved.hard_height_spread_m)
        / resolved.hard_height_spread_m,
        max(0.0, resolved.minimum_tactile_path_margin_m - tactile_margin)
        / observation.maximum_taxel_assignment_distance_m,
        0.0 if observation.support_retained else 1.0,
    )
    return SupportModeReadiness(
        passed=all(checks.values()),
        checks=MappingProxyType(checks),
        force_balance_ratio=float(balance),
        thumb_side_normal_force_n=thumb_force,
        opposed_side_normal_force_n=opposed_force,
        friction_capacity_n=float(capacity),
        friction_margin_n=float(friction_margin),
        line_of_action_moment_n_m=moment,
        height_spread_p95_m=float(observation.height_spread_p95_m),
        height_soft_excess_m=max(
            0.0,
            float(observation.height_spread_p95_m) - resolved.soft_height_target_m,
        ),
        tactile_path_margin_m=tactile_margin,
        normalized_max_violation=float(max(violations)),
        policy=resolved.as_dict(),
    )


@dataclass(frozen=True, slots=True)
class MeasuredGraspSource:
    config_path: str
    result_path: str
    trace_path: str
    config_sha256: str
    result_sha256: str
    trace_sha256: str
    source_id: str
    stable_window_start_step: int
    stable_window_end_step: int
    actual_joint_qpos_rad: tuple[float, ...]
    config: dict[str, Any]
    result: dict[str, Any]
    observation: SupportModeObservation
    readiness: SupportModeReadiness
    catalog_path: str | None = None
    catalog_sha256: str | None = None
    trajectory_id: str | None = None

    def __post_init__(self) -> None:
        actual = np.asarray(self.actual_joint_qpos_rad, dtype=np.float64)
        if actual.shape != (len(ACTIVE_ACTUATORS),) or not np.isfinite(actual).all():
            raise ValueError("actual_joint_qpos_rad must contain eight finite values")
        if self.stable_window_start_step < 0 or (
            self.stable_window_end_step < self.stable_window_start_step
        ):
            raise ValueError("measured stable-window bounds are invalid")
        for name in (
            "config_sha256",
            "result_sha256",
            "trace_sha256",
            "source_id",
        ):
            digest = str(getattr(self, name))
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{name} must be a lower-case SHA-256 digest")
        expected_source_id = payload_sha256(
            {
                "config_sha256": self.config_sha256,
                "result_sha256": self.result_sha256,
                "trace_sha256": self.trace_sha256,
            }
        )
        if self.source_id != expected_source_id:
            raise ValueError("source_id does not bind the three measured artifacts")
        if self.catalog_sha256 is not None:
            digest = str(self.catalog_sha256)
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("catalog_sha256 must be a lower-case SHA-256 digest")
        if not isinstance(self.observation, SupportModeObservation):
            raise TypeError("observation must be SupportModeObservation")
        if not isinstance(self.readiness, SupportModeReadiness):
            raise TypeError("readiness must be SupportModeReadiness")
        object.__setattr__(self, "actual_joint_qpos_rad", tuple(float(v) for v in actual))
        object.__setattr__(self, "config", copy.deepcopy(dict(self.config)))
        object.__setattr__(self, "result", copy.deepcopy(dict(self.result)))

    def identity(self) -> dict[str, Any]:
        return {
            "measured_grasp_source_schema_version": SOURCE_SCHEMA_VERSION,
            "source_id": self.source_id,
            "config_path": self.config_path,
            "result_path": self.result_path,
            "trace_path": self.trace_path,
            "config_sha256": self.config_sha256,
            "result_sha256": self.result_sha256,
            "trace_sha256": self.trace_sha256,
            "config_semantic_sha256": canonical_sha256(self.config),
            "result_payload_sha256": payload_sha256(self.result),
            "stable_window_start_step": self.stable_window_start_step,
            "stable_window_end_step": self.stable_window_end_step,
            "actual_joint_qpos_rad": list(self.actual_joint_qpos_rad),
            "grasp_pose_id": grasp_pose_id(self.config),
            "catalog_path": self.catalog_path,
            "catalog_sha256": self.catalog_sha256,
            "trajectory_id": self.trajectory_id,
            "readiness": self.readiness.as_dict(),
        }


def _trace_scalar(trace: Mapping[str, Any], name: str) -> int:
    value = np.asarray(trace[name])
    if value.size != 1:
        raise ValueError(f"{name} must be a scalar trace field")
    return int(value.reshape(-1)[0])


def _quat_rotation_wxyz(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("cube quaternion must contain four finite values")
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = (float(item) for item in quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _source_observation_from_trace(
    config: Mapping[str, Any],
    trace_path: Path,
    *,
    model_and_info: tuple[mujoco.MjModel, Any] | None = None,
) -> SupportModeObservation:
    model, info = (
        build_model(dict(config))
        if model_and_info is None
        else model_and_info
    )
    data = mujoco.MjData(model)
    reader = TactileReader(model, data, "left")
    with np.load(trace_path, allow_pickle=False) as trace:
        start = _trace_scalar(trace, "grasp_stable_window_start_step")
        end = _trace_scalar(trace, "grasp_stable_window_end_step")
        if start < 0 or end < start:
            raise ValueError("measured trace has no stable grasp window")
        face_order = tuple(str(value) for value in np.asarray(trace["face_order"]))
        finger_order = tuple(str(value) for value in np.asarray(trace["finger_order"]))
        actuator_order = tuple(
            str(value) for value in np.asarray(trace["actuator_order"])
        )
        if finger_order != tuple(ACTIVE_FINGERS):
            raise ValueError("measured trace active-finger order changed")
        expected_actuators = tuple((*ACTIVE_ACTUATORS, *INACTIVE_ACTUATORS))
        if actuator_order != expected_actuators:
            raise ValueError("measured trace actuator order changed")
        target_mapping = config["contact_topology"]["target_faces"]
        target_faces = tuple(str(target_mapping[finger]) for finger in ACTIVE_FINGERS)
        target_indices = [face_order.index(face) for face in target_faces]
        face_force = np.asarray(trace["distal_face_force_n"], dtype=np.float64)
        force = np.asarray(
            [
                np.median(face_force[start : end + 1, index, target_indices[index]])
                for index in range(3)
            ],
            dtype=np.float64,
        )
        centroid_world = np.asarray(
            trace["target_face_contact_centroid_world_m"], dtype=np.float64
        )
        centroid_valid = np.asarray(
            trace["target_face_contact_centroid_valid"], dtype=bool
        )
        cube_pos = np.asarray(trace["cube_pos"], dtype=np.float64)
        cube_quat = np.asarray(trace["cube_quat"], dtype=np.float64)
        joint_qpos = np.asarray(trace["joint_qpos"], dtype=np.float64)
        if (
            joint_qpos.ndim != 2
            or joint_qpos.shape[1] != len(expected_actuators)
            or end >= joint_qpos.shape[0]
        ):
            raise ValueError("measured trace joint_qpos shape changed")
        distances: list[list[float]] = [[], [], []]
        local_positions: list[list[np.ndarray]] = [[], [], []]
        for step in range(start, end + 1):
            data.qpos[info.actuator_qpos_adrs] = joint_qpos[step]
            address = info.cube_qpos_adr
            data.qpos[address : address + 3] = cube_pos[step]
            data.qpos[address + 3 : address + 7] = cube_quat[step]
            mujoco.mj_forward(model, data)
            rotation = _quat_rotation_wxyz(cube_quat[step])
            for finger_index in range(3):
                if not centroid_valid[step, finger_index]:
                    continue
                witness = centroid_world[step, finger_index]
                local_positions[finger_index].append(
                    rotation.T @ (witness - cube_pos[step])
                )
                sites = data.site_xpos[reader.site_ids[finger_index]]
                distances[finger_index].append(
                    float(np.min(np.linalg.norm(sites - witness, axis=1)))
                )
        if any(not values for values in local_positions) or any(
            not values for values in distances
        ):
            raise ValueError("stable window lacks valid centroid/taxel evidence")
        local = np.asarray(
            [np.median(np.asarray(values), axis=0) for values in local_positions]
        )
        tactile_p95 = tuple(
            float(np.percentile(values, 95.0)) for values in distances
        )
        spread = np.asarray(trace["three_contact_height_spread_m"], dtype=np.float64)
        height_p95 = float(np.percentile(spread[start : end + 1], 95.0))
        support = bool(np.all(np.asarray(trace["support_contact"])[start : end + 1]))
    maximum_taxel = float(
        config.get("fingertip_contact_preferences", {}).get(
            "taxel_assignment_max_distance_m", 0.006
        )
    )
    return SupportModeObservation(
        target_face_force_n=tuple(float(value) for value in force),
        contact_centroid_cube_m=tuple(
            tuple(float(value) for value in row) for row in local
        ),
        target_faces=target_faces,
        height_spread_p95_m=height_p95,
        tactile_nearest_distance_p95_m=tactile_p95,
        maximum_taxel_assignment_distance_m=maximum_taxel,
        support_retained=support,
    )


def _catalog_paths(
    catalog_path: Path, trajectory: str | None
) -> tuple[Path, Path, Path, dict[str, str], str]:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not payload.get("complete", False):
        raise ValueError("trajectory catalog must be a complete JSON object")
    aliases = payload.get("aliases", {})
    requested = trajectory or "best_nominal"
    resolved = str(aliases.get(requested, requested)) if isinstance(aliases, Mapping) else requested
    records = payload.get("trajectories", ())
    selected = next(
        (
            record
            for record in records
            if isinstance(record, Mapping)
            and str(record.get("trajectory_id")) == resolved
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"trajectory is absent from catalog: {requested}")
    if not bool(selected.get("grasp_success", False)):
        raise ValueError("catalog trajectory is not a grasp success")
    artifacts = selected.get("artifacts", {})
    hashes = artifacts.get("sha256", {}) if isinstance(artifacts, Mapping) else {}
    if not isinstance(hashes, Mapping):
        raise ValueError("catalog trajectory has no artifact hashes")
    base = catalog_path.parent
    paths = tuple(
        _confined_artifact(base, artifacts[name], f"catalog {name}")
        for name in ("resolved_config", "result", "trace")
    )
    expected = {name: str(hashes[name]) for name in ("resolved_config", "result", "trace")}
    return paths[0], paths[1], paths[2], expected, resolved


def load_measured_grasp_source(
    *,
    measured_config_path: str | Path | None = None,
    measured_result_path: str | Path | None = None,
    measured_trace_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    trajectory: str | None = None,
    policy: LiftManipulabilityPolicy | None = None,
) -> MeasuredGraspSource:
    """Load and authenticate one measured grasp from explicit paths/catalog."""

    if (measured_config_path is None) == (catalog_path is None):
        raise ValueError("select exactly one of measured_config_path or catalog_path")
    expected: dict[str, str] = {}
    resolved_catalog: str | None = None
    catalog_sha256: str | None = None
    trajectory_id: str | None = None
    if catalog_path is not None:
        catalog = Path(catalog_path).expanduser().resolve()
        authenticated_catalog_artifact_paths(catalog)
        catalog_sha256 = file_sha256(catalog)
        config_path, result_path, trace_path, expected, trajectory_id = _catalog_paths(
            catalog, trajectory
        )
        resolved_catalog = str(catalog)
    else:
        config_path = Path(measured_config_path).expanduser().resolve()  # type: ignore[arg-type]
        result_path = (
            Path(measured_result_path).expanduser().resolve()
            if measured_result_path is not None
            else config_path.with_name("result.json")
        )
        trace_path = (
            Path(measured_trace_path).expanduser().resolve()
            if measured_trace_path is not None
            else config_path.with_name("trace.npz")
        )
    if not all(path.is_file() for path in (config_path, result_path, trace_path)):
        raise RuntimeError("measured grasp source is incomplete")
    actual_hash = {
        "resolved_config": file_sha256(config_path),
        "result": file_sha256(result_path),
        "trace": file_sha256(trace_path),
    }
    for name, digest in expected.items():
        if actual_hash[name] != digest:
            raise RuntimeError(f"catalog {name} hash changed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_config(config)
    if result.get("complete") is not True:
        raise RuntimeError("measured result is not a complete production artifact")
    authenticate_candidate_result_semantic_sha256(result, source=result_path)
    declared_candidate_sha = result.get("candidate_sha256")
    if not isinstance(declared_candidate_sha, str) or (
        declared_candidate_sha != canonical_sha256(config)
    ):
        raise RuntimeError("measured result does not bind the loaded config")
    result_artifacts = result.get("artifacts", {})
    result_hashes = (
        result_artifacts.get("sha256", {})
        if isinstance(result_artifacts, Mapping)
        else {}
    )
    if not isinstance(result_hashes, Mapping):
        raise RuntimeError("measured result has no artifact hash mapping")
    for name in ("resolved_config", "trace"):
        expected_digest = result_hashes.get(name)
        if not isinstance(expected_digest, str) or expected_digest != actual_hash[name]:
            raise RuntimeError(f"measured result {name} hash changed")
    actual = extract_measured_grasp_qpos(config, trace_path, result)
    with np.load(trace_path, allow_pickle=False) as trace:
        start = _trace_scalar(trace, "grasp_stable_window_start_step")
        end = _trace_scalar(trace, "grasp_stable_window_end_step")
    model_and_info = build_model(config)
    observation = _source_observation_from_trace(
        config,
        trace_path,
        model_and_info=model_and_info,
    )
    readiness = evaluate_support_mode_readiness(
        observation,
        mass_kg=float(config["cube"]["mass_kg"]),
        sliding_friction=float(config["cube"]["friction"]),
        gravity_m_s2=model_and_info[0].opt.gravity,
        policy=policy,
    )
    source_id = payload_sha256(
        {
            "config_sha256": actual_hash["resolved_config"],
            "result_sha256": actual_hash["result"],
            "trace_sha256": actual_hash["trace"],
        }
    )
    return MeasuredGraspSource(
        config_path=str(config_path),
        result_path=str(result_path),
        trace_path=str(trace_path),
        config_sha256=actual_hash["resolved_config"],
        result_sha256=actual_hash["result"],
        trace_sha256=actual_hash["trace"],
        source_id=source_id,
        stable_window_start_step=start,
        stable_window_end_step=end,
        actual_joint_qpos_rad=tuple(float(value) for value in actual),
        config=copy.deepcopy(config),
        result=copy.deepcopy(result),
        observation=observation,
        readiness=readiness,
        catalog_path=resolved_catalog,
        catalog_sha256=catalog_sha256,
        trajectory_id=trajectory_id,
    )


def reauthenticate_measured_grasp_source(
    source: MeasuredGraspSource,
    *,
    policy: LiftManipulabilityPolicy | None = None,
) -> MeasuredGraspSource:
    """Reload sealed evidence and reject post-load file or memory mutation."""

    stored_policy = copy.deepcopy(dict(source.readiness.policy))
    resolved_policy = (
        LiftManipulabilityPolicy(**stored_policy) if policy is None else policy
    )
    if resolved_policy.as_dict() != stored_policy:
        raise RuntimeError("readiness policy changed after source authentication")

    if source.catalog_path is not None:
        refreshed = load_measured_grasp_source(
            catalog_path=source.catalog_path,
            trajectory=source.trajectory_id,
            policy=resolved_policy,
        )
    else:
        refreshed = load_measured_grasp_source(
            measured_config_path=source.config_path,
            measured_result_path=source.result_path,
            measured_trace_path=source.trace_path,
            policy=resolved_policy,
        )
    if refreshed.source_id != source.source_id:
        raise RuntimeError("measured source artifacts changed after authentication")
    if canonical_sha256(refreshed.config) != canonical_sha256(source.config):
        raise RuntimeError("measured source config mutated after authentication")
    if payload_sha256(refreshed.result) != payload_sha256(source.result):
        raise RuntimeError("measured source result mutated after authentication")
    if refreshed.actual_joint_qpos_rad != source.actual_joint_qpos_rad:
        raise RuntimeError("measured source qpos evidence changed after authentication")
    if refreshed.readiness.as_dict() != source.readiness.as_dict():
        raise RuntimeError("measured source readiness changed after authentication")
    return refreshed


def _cube_world_pose(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    edge = float(config["cube"]["edge_m"])
    rotation = rpy_degrees_to_rotation_matrix(config["cube"]["rpy_deg"])
    half_vertical_extent = 0.5 * edge * float(np.sum(np.abs(rotation[2])))
    xy = np.asarray(config["cube"]["center_xy_m"], dtype=np.float64)
    z = (
        float(config["scene"]["support_top_z_m"])
        + half_vertical_extent
        + float(config["cube"].get("z_offset_m", 0.0))
    )
    return np.asarray((xy[0], xy[1], z), dtype=np.float64), rotation


def _pose_on_resized_cube(
    source_config: Mapping[str, Any], target_config: Mapping[str, Any], pose: Mapping[str, Any]
) -> dict[str, list[float]]:
    source_cube_position, source_cube_rotation = _cube_world_pose(source_config)
    target_cube_position, target_cube_rotation = _cube_world_pose(target_config)
    transformed = transform_relative_wrist_pose(
        source_cube_world_position_m=source_cube_position,
        source_cube_world_rotation=source_cube_rotation,
        source_root_world_position_m=pose["translation_m"],
        source_root_world_rotation=rpy_degrees_to_rotation_matrix(pose["rpy_deg"]),
        target_cube_world_position_m=target_cube_position,
        target_cube_world_rotation=target_cube_rotation,
    )
    return {
        "translation_m": list(transformed.root_world_position_m),
        "rpy_deg": rotation_matrix_to_rpy_degrees(
            transformed.root_world_rotation,
            reference_rpy_deg=pose["rpy_deg"],
        ).tolist(),
    }


def _registered_relative_wrist_base(source: MeasuredGraspSource) -> dict[str, Any]:
    """Promote an authenticated pre-v11 grasp without changing its evidence.

    The current grasp catalogs are schema-v10, while the power-recovery
    diagnostics are schema-v11.  Geometry generation needs the registered v11
    relative-wrist bounds in both cases.  Promotion creates a new in-memory
    proposal only; it never rewrites the source files or claims that the
    promoted configuration itself is measured evidence.
    """

    if isinstance(source.config.get("relative_wrist_pose_search"), Mapping):
        return copy.deepcopy(source.config)
    template = load_config(_V11_TEMPLATE)
    for field in (
        "cube",
        "hand_pose",
        "contact_topology",
        "control",
        "grasp_pose",
    ):
        template[field] = copy.deepcopy(source.config[field])
    nominal = template["grasp_pose"]["nominal_joint_qpos_rad"]
    for index, name in enumerate(ACTIVE_ACTUATORS):
        nominal[name] = float(source.actual_joint_qpos_rad[index])
    relative_policy = RelativeWristPoseSearchPolicy.from_config(template)
    metadata = copy.deepcopy(source.config.get("candidate_metadata", {}))
    metadata.update(
        {
            "promoted_from_measured_source_id": source.source_id,
            "source_schema_version": int(source.config["schema_version"]),
            "relative_wrist_pose_search": {
                "anchor_hand_pose": copy.deepcopy(source.config["hand_pose"]),
                "clockwise_orbit_deg": 0.0,
                "root_delta_cube_m": [0.0, 0.0, 0.0],
                "wrist_local_rotvec_deg": [0.0, 0.0, 0.0],
                "cube_pose_sampled": False,
                "hand_root_fixed_during_simulation": True,
                "policy": relative_policy.as_config(),
            },
        }
    )
    template["candidate_metadata"] = metadata
    return template


def resize_measured_source(
    source: MeasuredGraspSource, edge_m: float
) -> dict[str, Any]:
    """Resize only the registered cube while preserving its relative grasp."""

    edge = _finite(edge_m, "edge_m")
    search_base = _registered_relative_wrist_base(source)
    result = copy.deepcopy(search_base)
    result["cube"]["edge_m"] = edge
    result["cube"]["mass_kg"] = 0.160
    result["cube"]["friction"] = 0.8
    result["cube"]["z_offset_m"] = 0.0
    metadata = result.setdefault("candidate_metadata", {})
    relative = metadata.setdefault("relative_wrist_pose_search", {})
    anchor = relative.get("anchor_hand_pose", source.config["hand_pose"])
    relative["anchor_hand_pose"] = _pose_on_resized_cube(
        search_base, result, anchor
    )
    result["hand_pose"] = _pose_on_resized_cube(
        search_base, result, search_base["hand_pose"]
    )
    result["grasp_pose"]["nominal_joint_qpos_rad"] = {
        name: float(source.actual_joint_qpos_rad[index])
        for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    return result


def _prefix_stable_samples(count: int, dimensions: int, seed: int) -> np.ndarray:
    """Return counter-keyed samples whose first N rows never change at N+K."""

    values = np.empty((count, dimensions), dtype=np.float64)
    for index in range(count):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(index), 51_013_013])
        )
        values[index] = rng.uniform(-1.0, 1.0, size=dimensions)
    return values


def _candidate_id(
    source_id: str,
    edge: float,
    orbit: float,
    index: int,
    namespace_sha256: str,
) -> int:
    digest = _lift_candidate_sha256(
        source_id, edge, orbit, index, namespace_sha256
    )
    available = MAX_STATIC_CANDIDATE_ID - CANDIDATE_ID_BASE + 1
    return CANDIDATE_ID_BASE + int(digest[:16], 16) % available


def _lift_candidate_sha256(
    source_id: str,
    edge: float,
    orbit: float,
    index: int,
    namespace_sha256: str,
) -> str:
    return payload_sha256(
        {
            "source_id": source_id,
            "edge_m": edge,
            "clockwise_orbit_deg": orbit,
            "sample_index": index,
            "namespace_sha256": namespace_sha256,
        }
    )


def _preserve_controller_residuals(
    source: Mapping[str, Any],
    candidate: dict[str, Any],
    *,
    preserve_precontact: bool = True,
) -> None:
    definition = resolve_experiment(candidate)
    bounds = definition.search_bounds.actuator_targets_rad
    source_nominal = source["grasp_pose"]["nominal_joint_qpos_rad"]
    target_nominal = candidate["grasp_pose"]["nominal_joint_qpos_rad"]
    fields = ["contact_preload_targets_rad"]
    if preserve_precontact:
        fields.insert(0, "precontact_targets_rad")
    for field in fields:
        source_values = source.get("control", {}).get(field)
        if not isinstance(source_values, Mapping):
            continue
        candidate["control"][field] = {
            name: float(
                np.clip(
                    float(target_nominal[name])
                    + float(source_values[name])
                    - float(source_nominal[name]),
                    *bounds[name],
                )
            )
            for name in ACTIVE_ACTUATORS
        }
    candidate["control"]["manipulation_delta_rad"] = {
        name: 0.0 for name in ACTIVE_ACTUATORS
    }


CandidateEvaluator = Callable[[Mapping[str, Any]], SupportModeReadiness | Mapping[str, Any] | None]


def _readiness_dict(value: SupportModeReadiness | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, SupportModeReadiness):
        return value.as_dict()
    result = copy.deepcopy(dict(value))
    if "passed" not in result or "normalized_max_violation" not in result:
        raise ValueError("candidate evaluator must return readiness rank fields")
    return result


def lift_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    readiness = record.get("readiness")
    perturbation = float(record.get("normalized_perturbation_norm", 1.0))
    if not isinstance(readiness, Mapping):
        return (1, 1, perturbation, int(record["candidate_id"]))
    return (
        0,
        not bool(readiness.get("passed", False)),
        float(readiness.get("normalized_max_violation", 1e6)),
        float(readiness.get("height_soft_excess_m", 1e6)),
        float(readiness.get("line_of_action_moment_n_m", 1e6)),
        -float(readiness.get("friction_margin_n", -1e6)),
        -float(readiness.get("force_balance_ratio", 0.0)),
        -float(readiness.get("tactile_path_margin_m", -1e6)),
        perturbation,
        int(record["candidate_id"]),
    )


@dataclass(frozen=True, slots=True)
class LiftCandidateGeneration:
    records: tuple[dict[str, Any], ...]
    report: dict[str, Any]


DIAGNOSTIC_CLOCKWISE_ORBITS_DEG = (0.0, 0.25, 0.5)
DIAGNOSTIC_ROOT_X_OFFSETS_M = (0.00025, 0.00050, 0.00075)
DIAGNOSTIC_WRIST_LOCAL_ROTVECS_DEG = (
    (-5.020, 0.508, -0.359),
    (-5.50, -0.75, -0.80),
    (-6.0, -2.0, -1.3),
)


def generate_exact_lift_diagnostic_grid(
    source: MeasuredGraspSource,
    *,
    edge_m: float | None = None,
    clockwise_orbits_deg: Sequence[float] = DIAGNOSTIC_CLOCKWISE_ORBITS_DEG,
    root_x_offsets_m: Sequence[float] = DIAGNOSTIC_ROOT_X_OFFSETS_M,
    wrist_local_rotvecs_deg: Sequence[Sequence[float]] = (
        DIAGNOSTIC_WRIST_LOCAL_ROTVECS_DEG
    ),
) -> LiftCandidateGeneration:
    """Build the exact 27-point post-power diagnostic grid.

    Sub-degree diagnostic orbits are represented by rotating the persisted
    anchor and solving the registered zero-orbit stratum.  This preserves the
    schema-v11 registry verbatim while DLS still re-solves all seven non-thumb
    actual-contact qpos variables for every configuration-valid point.  Points
    outside the registered v11 pose envelope remain explicit diagnostics and
    are never physically evaluated or promoted.
    """

    edge = float(source.config["cube"]["edge_m"] if edge_m is None else edge_m)
    base = resize_measured_source(source, edge)
    base_relative = base.setdefault("candidate_metadata", {}).setdefault(
        "relative_wrist_pose_search", {}
    )
    anchor = copy.deepcopy(base_relative.get("anchor_hand_pose", base["hand_pose"]))
    source_variables = RelativeWristVariables.from_config(base)
    source_vector = source_variables.as_array()
    for index, name in enumerate(NON_THUMB_ACTUATORS):
        source_vector[index] = float(
            source.actual_joint_qpos_rad[ACTIVE_ACTUATORS.index(name)]
        )
    orbits = tuple(_finite(value, "diagnostic clockwise orbit") for value in clockwise_orbits_deg)
    offsets = tuple(_finite(value, "diagnostic root-x offset") for value in root_x_offsets_m)
    wrist_values = tuple(
        tuple(float(value) for value in _vector3(item, "diagnostic wrist rotvec"))
        for item in wrist_local_rotvecs_deg
    )
    if not orbits or not offsets or not wrist_values:
        raise ValueError("diagnostic grid axes must be non-empty")
    cube_position, cube_rotation = _cube_world_pose(base)
    anchor_rotation = rpy_degrees_to_rotation_matrix(anchor["rpy_deg"])
    namespace = payload_sha256(
        {
            "kind": "exact_post_power_lift_diagnostic_grid",
            "source_id": source.source_id,
            "edge_m": edge,
            "clockwise_orbits_deg": list(orbits),
            "root_x_offsets_m": list(offsets),
            "wrist_local_rotvecs_deg": [list(value) for value in wrist_values],
        }
    )
    records: list[dict[str, Any]] = []
    seen_candidate_ids: set[int] = set()
    sample_index = 0
    for orbit in orbits:
        orbited_anchor = transform_relative_wrist_pose(
            source_cube_world_position_m=cube_position,
            source_cube_world_rotation=cube_rotation,
            source_root_world_position_m=anchor["translation_m"],
            source_root_world_rotation=anchor_rotation,
            clockwise_orbit_deg=orbit,
        )
        orbit_base = copy.deepcopy(base)
        orbit_base["candidate_metadata"]["relative_wrist_pose_search"][
            "anchor_hand_pose"
        ] = {
            "translation_m": list(orbited_anchor.root_world_position_m),
            "rpy_deg": rotation_matrix_to_rpy_degrees(
                orbited_anchor.root_world_rotation,
                reference_rpy_deg=anchor["rpy_deg"],
            ).tolist(),
        }
        for root_x_offset in offsets:
            for wrist_deg in wrist_values:
                vector = source_vector.copy()
                vector[7] += root_x_offset
                vector[10:13] = np.radians(wrist_deg)
                variables = RelativeWristVariables.from_array(vector)
                candidate = materialize_relative_wrist_candidate(
                    orbit_base,
                    variables,
                    clockwise_orbit_deg=0.0,
                    synchronize_preload=False,
                )
                _preserve_controller_residuals(orbit_base, candidate)
                candidate_id = _candidate_id(
                    source.source_id,
                    edge,
                    orbit,
                    sample_index,
                    namespace,
                )
                if candidate_id in seen_candidate_ids:
                    raise RuntimeError("deterministic static candidate ID collision")
                seen_candidate_ids.add(candidate_id)
                lift_sha = _lift_candidate_sha256(
                    source.source_id,
                    edge,
                    orbit,
                    sample_index,
                    namespace,
                )
                metadata = candidate.setdefault("candidate_metadata", {})
                metadata.update(
                    {
                        "campaign_kind": CAMPAIGN_KIND,
                        "candidate_id": candidate_id,
                        "source_measured_grasp_id": source.source_id,
                        "lift_manipulability_search": {
                            "mode": "exact_post_power_diagnostic_grid",
                            "edge_m": edge,
                            "clockwise_orbit_deg": orbit,
                            "solver_clockwise_orbit_deg": 0.0,
                            "root_x_offset_from_source_m": root_x_offset,
                            "wrist_local_rotvec_deg": list(wrist_deg),
                            "variables": variables.as_array().tolist(),
                            "sample_index": sample_index,
                            "lift_candidate_sha256": lift_sha,
                            "static_candidate_id": candidate_id,
                            "cube_pose_sampled": False,
                            "hand_root_fixed_during_simulation": True,
                            "dls_resolves_seven_non_thumb_joints": True,
                        },
                    }
                )
                try:
                    validate_config(candidate)
                    configuration_valid = True
                    configuration_error = None
                except (TypeError, ValueError) as error:
                    configuration_valid = False
                    configuration_error = str(error)
                scale = np.asarray(
                    (0.06,) * 7 + (0.001, 0.001, 0.001) + tuple(np.radians((1.0, 1.0, 1.0)))
                )
                normalized = float(
                    np.linalg.norm((variables.as_array() - source_vector) / scale)
                    / math.sqrt(13.0)
                )
                records.append(
                    {
                        "lift_manipulability_candidate_schema_version": (
                            CANDIDATE_SCHEMA_VERSION
                        ),
                        "candidate_id": candidate_id,
                        "lift_candidate_sha256": lift_sha,
                        "source_id": source.source_id,
                        "edge_m": edge,
                        "clockwise_orbit_deg": orbit,
                        "solver_clockwise_orbit_deg": 0.0,
                        "sample_index": sample_index,
                        "normalized_perturbation_norm": normalized,
                        "candidate_sha256": canonical_sha256(candidate),
                        "configuration_valid": configuration_valid,
                        "configuration_error": configuration_error,
                        "readiness": None,
                        "evaluation_status": (
                            "proposal_only_requires_physical_screen"
                        ),
                        "eligible_for_dynamic_grasp": False,
                        "config": candidate,
                    }
                )
                sample_index += 1
    report = {
        "lift_candidate_generation_schema_version": GENERATION_SCHEMA_VERSION,
        "complete": True,
        "proposal_only": True,
        "physical_screen_required_before_dynamic_grasp": True,
        "mode": "exact_post_power_diagnostic_grid",
        "source_id": source.source_id,
        "edge_m": edge,
        "clockwise_orbits_deg": list(orbits),
        "root_x_offsets_m": list(offsets),
        "wrist_local_rotvecs_deg": [list(value) for value in wrist_values],
        "declared_sample_count": len(orbits) * len(offsets) * len(wrist_values),
        "retained_candidate_count": len(records),
    }
    return LiftCandidateGeneration(tuple(records), report)


@dataclass(frozen=True, slots=True)
class LiftPhysicalScreen:
    records: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def _physical_screen_input_sha256(
    source: MeasuredGraspSource,
    records: Sequence[Mapping[str, Any]],
    settings: RelativeWristDLSSettings,
) -> str:
    dependency_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "xhand_grasp" / "config.py",
        REPO_ROOT / "xhand_grasp" / "experiment.py",
        REPO_ROOT / "xhand_grasp" / "scene.py",
        REPO_ROOT / "xhand_grasp" / "tuning" / "relative_wrist_pose_search.py",
        REPO_ROOT / "xhand_grasp" / "tuning" / "actual_contact_grasp_pose.py",
        REPO_ROOT / "xhand_grasp" / "experiments" / "__init__.py",
        REPO_ROOT
        / "xhand_grasp"
        / "experiments"
        / "opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
        "smooth_vertical_lift.py",
        _V11_TEMPLATE,
    )
    return payload_sha256(
        {
            "source_id": source.source_id,
            "candidate_inputs": [
                {
                    "candidate_id": int(value["candidate_id"]),
                    "candidate_sha256": str(value["candidate_sha256"]),
                    "lift_candidate_sha256": str(value["lift_candidate_sha256"]),
                }
                for value in sorted(records, key=lambda item: int(item["candidate_id"]))
            ],
            "dls_settings": asdict(settings),
            "source_code_sha256": {
                str(path.relative_to(REPO_ROOT)): file_sha256(path)
                for path in dependency_paths
            },
            "model_sha256": file_sha256(REPO_ROOT / "xhand_left.xml"),
            "uv_lock_sha256": file_sha256(REPO_ROOT / "uv.lock"),
        }
    )


def write_physical_screen_artifacts(
    screen: LiftPhysicalScreen,
    output_dir: str | Path,
    *,
    input_sha256: str,
) -> dict[str, Any]:
    """Atomically publish configs and static evidence for legacy dynamic input.

    The persisted result explicitly records that collision-witness success is
    not grasp evidence.  Candidate IDs are retained unchanged so the legacy
    ``16*S+seed`` and ``1000*D+local`` namespaces remain safe.
    """

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for record in sorted(screen.records, key=lambda value: int(value["candidate_id"])):
        candidate_id = int(record["candidate_id"])
        if candidate_id > MAX_STATIC_CANDIDATE_ID:
            raise ValueError("static candidate ID exceeds downstream int64 budget")
        candidate_dir = output / "candidates" / f"candidate_{candidate_id}"
        config_path = candidate_dir / "resolved_config.json"
        result_path = candidate_dir / "physical_screen_result.json"
        config = copy.deepcopy(dict(record["config"]))
        config_digest = canonical_sha256(config)
        if config_digest != record["candidate_sha256"]:
            raise RuntimeError("physical-screen candidate config changed before publish")
        result = {key: copy.deepcopy(value) for key, value in record.items() if key != "config"}
        result.update(
            {
                "complete": True,
                "config_semantic_sha256": config_digest,
                "static_is_grasp_success_evidence": False,
                "grasp_success": False,
            }
        )
        _atomic_json(config_path, config)
        _atomic_json(result_path, result)
        entries.append(
            {
                "candidate_id": candidate_id,
                "lift_candidate_sha256": record["lift_candidate_sha256"],
                "eligible_for_dynamic_grasp": bool(record["eligible_for_dynamic_grasp"]),
                "resolved_config": str(config_path.relative_to(output)),
                "physical_screen_result": str(result_path.relative_to(output)),
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "physical_screen_result": file_sha256(result_path),
                },
            }
        )
    report_path = output / "physical_screen_report.json"
    report_payload = {**copy.deepcopy(screen.report), "candidates": entries}
    _atomic_json(report_path, report_payload)
    manifest = {
        "lift_physical_screen_manifest_schema_version": 1,
        "complete": True,
        "input_sha256": str(input_sha256),
        "report": str(report_path.relative_to(output)),
        "report_sha256": file_sha256(report_path),
        "candidate_count": len(entries),
        "dynamic_eligible_count": sum(
            bool(value["eligible_for_dynamic_grasp"]) for value in entries
        ),
        "static_is_grasp_success_evidence": False,
    }
    _atomic_json(output / "physical_screen_manifest.json", manifest)
    return manifest


def load_physical_screen_artifacts(
    output_dir: str | Path,
    *,
    expected_input_sha256: str | None = None,
) -> LiftPhysicalScreen:
    """Authenticate and load one published physical screen for dynamic use."""

    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "physical_screen_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise RuntimeError("physical-screen manifest is incomplete")
    if expected_input_sha256 is not None and manifest.get("input_sha256") != expected_input_sha256:
        raise RuntimeError("physical-screen resume input changed")
    report_path = _confined_artifact(output, manifest["report"], "physical screen report")
    if file_sha256(report_path) != manifest.get("report_sha256"):
        raise RuntimeError("physical-screen report SHA-256 mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for entry in report.get("candidates", []):
        config_path = _confined_artifact(output, entry["resolved_config"], "physical config")
        result_path = _confined_artifact(output, entry["physical_screen_result"], "physical result")
        for label, path in (("resolved_config", config_path), ("physical_screen_result", result_path)):
            if file_sha256(path) != entry["sha256"][label]:
                raise RuntimeError(f"physical-screen {label} SHA-256 mismatch")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if canonical_sha256(config) != result.get("config_semantic_sha256"):
            raise RuntimeError("physical-screen config semantic SHA-256 mismatch")
        if result.get("grasp_success") is not False:
            raise RuntimeError("static physical screen cannot claim grasp success")
        result.pop("config_semantic_sha256", None)
        result.pop("complete", None)
        result["config"] = config
        records.append(result)
    clean_report = {key: value for key, value in report.items() if key != "candidates"}
    return LiftPhysicalScreen(tuple(records), clean_report)


def dynamic_grasp_inputs_from_physical_screen(
    screen: LiftPhysicalScreen,
) -> tuple[dict[str, Any], ...]:
    """Return only real-static-pass configs in the legacy dynamic-runner shape."""

    result = []
    for record in sorted(screen.records, key=lambda value: int(value["candidate_id"])):
        if not bool(record.get("eligible_for_dynamic_grasp", False)):
            continue
        required_physical_evidence = (
            record.get("evaluation_status")
            == "physical_collision_witness_complete"
            and record.get("static_geometry_pass") is True
            and record.get("configuration_valid") is True
            and record.get("grasp_success") is False
            and record.get("static_is_grasp_success_evidence") is False
            and isinstance(record.get("static_result"), Mapping)
            and record["static_result"].get("static_geometry_pass") is True
        )
        if not required_physical_evidence:
            raise RuntimeError(
                "dynamic input lacks a completed physical collision-witness pass"
            )
        candidate_id = int(record["candidate_id"])
        if candidate_id > MAX_STATIC_CANDIDATE_ID:
            raise ValueError("static candidate ID exceeds downstream int64 budget")
        config = copy.deepcopy(dict(record["config"]))
        digest = canonical_sha256(config)
        if digest != record["candidate_sha256"]:
            raise RuntimeError("physical-screen config semantic hash changed")
        metadata = config.get("candidate_metadata", {})
        if not isinstance(metadata, Mapping) or (
            metadata.get("stage") != "physical_collision_witness_screen"
            or int(metadata.get("candidate_id", -1)) != candidate_id
            or metadata.get("lift_candidate_sha256")
            != record.get("lift_candidate_sha256")
            or metadata.get("proposal_candidate_sha256")
            != record.get("proposal_candidate_sha256")
        ):
            raise RuntimeError("dynamic input lost physical-stage provenance")
        result.append(
            {
                "candidate_id": candidate_id,
                "candidate_sha256": digest,
                "lift_candidate_sha256": record["lift_candidate_sha256"],
                "grasp_pose_id": grasp_pose_id(config),
                "config": config,
            }
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class LiftSupportedDynamicScreen:
    records: tuple[dict[str, Any], ...]
    ready_sources: tuple[MeasuredGraspSource, ...]
    report: dict[str, Any]


def _authenticate_dynamic_runner_outputs(
    inputs: Sequence[Mapping[str, Any]],
    raw_results: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    controller_seed_count: int,
) -> tuple[dict[str, Any], ...]:
    """Rebind every runner result to one exact physical-screen source config."""

    from .actual_contact_grasp_pose_dynamic import expand_controller_candidates

    expected_jobs = expand_controller_candidates(
        inputs, controller_seed_count=controller_seed_count
    )
    expected_by_id = {
        int(value["candidate_id"]): value for value in expected_jobs
    }
    if len(raw_results) != len(expected_by_id):
        raise RuntimeError("dynamic runner output cardinality changed")
    input_by_id = {int(value["candidate_id"]): value for value in inputs}
    observed_ids: set[int] = set()
    authenticated: list[dict[str, Any]] = []
    for raw in raw_results:
        record = copy.deepcopy(dict(raw))
        candidate_id = int(record.get("candidate_id", -1))
        if candidate_id in observed_ids:
            raise RuntimeError("dynamic runner returned duplicate candidate IDs")
        observed_ids.add(candidate_id)
        expected = expected_by_id.get(candidate_id)
        if expected is None:
            raise RuntimeError("dynamic runner returned an unknown candidate ID")
        source_id = int(record.get("source_candidate_id", -1))
        if source_id != int(expected["source_candidate_id"]):
            raise RuntimeError("dynamic runner changed source candidate ID")
        source = input_by_id.get(source_id)
        if source is None:
            raise RuntimeError("dynamic runner source is not an eligible static input")
        if int(record.get("controller_seed_index", -1)) != int(
            expected["controller_seed_index"]
        ):
            raise RuntimeError("dynamic runner changed controller seed index")
        config = record.get("config")
        if not isinstance(config, Mapping):
            raise RuntimeError("dynamic runner returned no resolved config")
        digest = canonical_sha256(config)
        if digest != expected["candidate_sha256"] or digest != record.get(
            "candidate_sha256"
        ):
            raise RuntimeError("dynamic runner rebound a different candidate config")
        if record.get("grasp_pose_id") != expected["grasp_pose_id"] or (
            grasp_pose_id(config) != source["grasp_pose_id"]
        ):
            raise RuntimeError("dynamic runner rebound a different grasp pose")
        if record.get("controller_id") != expected["controller_id"]:
            raise RuntimeError("dynamic runner rebound a different controller")
        metadata = config.get("candidate_metadata", {})
        if not isinstance(metadata, Mapping) or int(
            metadata.get("source_candidate_id", -1)
        ) != source_id:
            raise RuntimeError("dynamic config lost its static source binding")
        expected_lift = str(source["lift_candidate_sha256"])
        if str(metadata.get("lift_candidate_sha256")) != expected_lift:
            raise RuntimeError("dynamic config lost its lift candidate binding")
        artifact_value = record.get("artifact_directory")
        if not isinstance(artifact_value, str) or not artifact_value:
            raise RuntimeError("dynamic runner returned no artifact directory")
        try:
            directory = _confined_artifact(
                output, artifact_value, "dynamic artifact directory"
            )
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        if directory == output:
            raise RuntimeError("dynamic artifact directory must be below output")
        config_path = directory / "resolved_config.json"
        result_path = directory / "result.json"
        if not config_path.is_file() or not result_path.is_file():
            raise RuntimeError("dynamic runner did not persist config/result artifacts")
        persisted_config = json.loads(config_path.read_text(encoding="utf-8"))
        persisted_result = json.loads(result_path.read_text(encoding="utf-8"))
        authenticate_candidate_result_semantic_sha256(
            persisted_result, source=result_path
        )
        if canonical_sha256(persisted_config) != expected["candidate_sha256"]:
            raise RuntimeError("persisted dynamic config does not match expected job")
        for field, expected_value in (
            ("candidate_id", candidate_id),
            ("source_candidate_id", source_id),
            ("controller_seed_index", int(expected["controller_seed_index"])),
            ("candidate_sha256", expected["candidate_sha256"]),
            ("grasp_pose_id", expected["grasp_pose_id"]),
            ("controller_id", expected["controller_id"]),
        ):
            if persisted_result.get(field) != expected_value:
                raise RuntimeError(f"persisted dynamic {field} binding mismatch")
            if record.get(field) != persisted_result.get(field):
                raise RuntimeError(f"dynamic runner memory/disk {field} mismatch")
        if persisted_result.get("complete") is not True:
            raise RuntimeError("persisted dynamic result is incomplete")
        if record.get("summary") != persisted_result.get("summary"):
            raise RuntimeError("dynamic runner memory/disk summary mismatch")
        artifacts = persisted_result.get("artifacts", {})
        hashes = artifacts.get("sha256", {}) if isinstance(artifacts, Mapping) else {}
        if not isinstance(hashes, Mapping) or hashes.get(
            "resolved_config"
        ) != file_sha256(config_path):
            raise RuntimeError("persisted dynamic config file hash mismatch")
        declared_config_path = _confined_artifact(
            directory,
            artifacts.get("resolved_config"),
            "dynamic result config",
        )
        if declared_config_path != config_path:
            raise RuntimeError("dynamic result points at a different config")
        trace_path = directory / "trace.npz"
        trace_retained = bool(artifacts.get("trace_retained", True))
        if trace_retained:
            declared_trace_path = _confined_artifact(
                directory, artifacts.get("trace"), "dynamic result trace"
            )
            if declared_trace_path != trace_path or not trace_path.is_file():
                raise RuntimeError("dynamic result retained trace is missing")
            if hashes.get("trace") != file_sha256(trace_path):
                raise RuntimeError("persisted dynamic trace file hash mismatch")
        else:
            compacted_digest = artifacts.get("trace_sha256_at_evaluation")
            if trace_path.exists() or not isinstance(compacted_digest, str) or (
                len(compacted_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in compacted_digest
                )
            ):
                raise RuntimeError("dynamic compacted trace evidence is invalid")
        record["lift_candidate_sha256"] = expected_lift
        record["_resolved_artifact_directory"] = str(directory)
        authenticated.append(record)
    if observed_ids != set(expected_by_id):
        raise RuntimeError("dynamic runner did not preserve expected candidate IDs")
    authenticated.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(authenticated)


def run_supported_dynamic_grasp_screen(
    screen: LiftPhysicalScreen,
    output_dir: str | Path,
    *,
    workers: int = 1,
    resume: bool = False,
    controller_seed_count: int = 1,
    policy: LiftManipulabilityPolicy | None = None,
    dynamic_runner: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
) -> LiftSupportedDynamicScreen:
    """Run real acquisition, then enforce support-mode readiness before lift.

    This is the supported transition into the existing measured/manipulation
    runner.  Static witness success is never counted as grasp success.
    """

    production_runner_attested = dynamic_runner is None
    if dynamic_runner is None:
        from .actual_contact_grasp_pose_dynamic import (
            run_actual_contact_dynamic_grasp_candidates,
        )

        dynamic_runner = run_actual_contact_dynamic_grasp_candidates
    inputs = dynamic_grasp_inputs_from_physical_screen(screen)
    output = Path(output_dir).expanduser().resolve()
    raw_results = tuple(
        dynamic_runner(
            inputs,
            output,
            workers=workers,
            resume=resume,
            controller_seed_count=controller_seed_count,
        )
    )
    authenticated_results = _authenticate_dynamic_runner_outputs(
        inputs,
        raw_results,
        output,
        controller_seed_count=controller_seed_count,
    )
    readiness_policy = policy or LiftManipulabilityPolicy()
    records: list[dict[str, Any]] = []
    ready_sources: list[MeasuredGraspSource] = []
    for raw in authenticated_results:
        record = copy.deepcopy(dict(raw))
        summary = record.get("summary", {})
        stage = summary.get("stage_status", {}) if isinstance(summary, Mapping) else {}
        grasp_success = bool(
            isinstance(stage, Mapping) and stage.get("grasp_success") is True
        )
        if grasp_success and not production_runner_attested:
            raise RuntimeError(
                "an injected dynamic runner cannot promote grasp evidence"
            )
        readiness_payload: dict[str, Any] = {
            "status": "grasp_not_acquired",
            "passed": False,
        }
        if grasp_success:
            directory = Path(record["_resolved_artifact_directory"])
            source = load_measured_grasp_source(
                measured_config_path=directory / "resolved_config.json",
                measured_result_path=directory / "result.json",
                measured_trace_path=directory / "trace.npz",
                policy=readiness_policy,
            )
            if canonical_sha256(source.config) != record["candidate_sha256"]:
                raise RuntimeError(
                    "successful dynamic artifact does not match expected config"
                )
            for field, expected_value in (
                ("candidate_id", int(record["candidate_id"])),
                ("source_candidate_id", int(record["source_candidate_id"])),
                ("grasp_pose_id", record["grasp_pose_id"]),
                ("controller_id", record["controller_id"]),
            ):
                if source.result.get(field) != expected_value:
                    raise RuntimeError(
                        f"successful dynamic artifact {field} lineage mismatch"
                    )
            readiness_payload = source.readiness.as_dict()
            if source.readiness.passed:
                ready_sources.append(source)
        record["support_mode_readiness"] = readiness_payload
        record["eligible_for_manipulation"] = bool(
            grasp_success and readiness_payload.get("passed", False)
        )
        record.pop("_resolved_artifact_directory", None)
        records.append(record)
    report = {
        "lift_supported_dynamic_screen_schema_version": 1,
        "complete": True,
        "production_dynamic_runner_attested": production_runner_attested,
        "static_input_count": len(inputs),
        "dynamic_candidate_count": len(records),
        "grasp_success_count": sum(
            bool(
                isinstance(value.get("summary", {}).get("stage_status", {}), Mapping)
                and value["summary"]["stage_status"].get("grasp_success") is True
            )
            for value in records
        ),
        "support_mode_ready_count": len(ready_sources),
        "hard_thresholds": {
            "minimum_force_balance_ratio": readiness_policy.minimum_force_balance_ratio,
            "minimum_friction_margin_n": readiness_policy.minimum_friction_margin_n,
            "maximum_line_moment_n_m": readiness_policy.maximum_line_moment_n_m,
            "hard_height_spread_m": readiness_policy.hard_height_spread_m,
            "minimum_tactile_path_margin_m": (
                readiness_policy.minimum_tactile_path_margin_m
            ),
        },
        "soft_objectives": {
            "soft_height_target_m": readiness_policy.soft_height_target_m,
        },
        "readiness_policy": readiness_policy.as_dict(),
        "readiness_policy_sha256": payload_sha256(readiness_policy.as_dict()),
        "static_is_grasp_success_evidence": False,
        "ready_sources_feed_run_free_body_probe_search": True,
    }
    return LiftSupportedDynamicScreen(
        tuple(records),
        tuple(ready_sources),
        report,
    )


def solve_joint_only_contact_dls(
    base_config: Mapping[str, Any],
    *,
    clockwise_orbit_deg: float,
    initial_variables: RelativeWristVariables,
    settings: RelativeWristDLSSettings,
    evaluator: Callable[[Mapping[str, Any]], Any],
    joint_bounds: Mapping[str, tuple[float, float]],
) -> RelativeWristDLSResult:
    """DLS the seven non-thumb contact joints while holding the 6D wrist fixed."""

    reference = initial_variables.as_array()
    current = reference.copy()
    definition = resolve_experiment(base_config)
    campaign = definition.actual_contact_grasp_pose_campaign
    if campaign is None:
        raise ValueError("joint-only contact DLS requires an actual-contact campaign")
    # Use the same small positive separation as the production orientation-aware
    # DLS.  The campaign acceptance interval is asymmetric and its midpoint is
    # penetrating, so it is not a physically meaningful optimization target.
    gap_target = float(CANONICAL_TARGET_GAP_M)
    target = np.asarray((gap_target, gap_target, gap_target, 0.0, 0.0, 1.0, 1.0, 1.0))
    scale = np.asarray((0.0005, 0.0005, 0.0005, 0.005, 0.005, 0.05, 0.05, 0.05))
    finite_difference = float(settings.joint_finite_difference_rad)

    def materialize(vector: np.ndarray) -> tuple[dict[str, Any], Any]:
        variables = RelativeWristVariables.from_array(vector)
        config = materialize_relative_wrist_candidate(
            base_config,
            variables,
            clockwise_orbit_deg=clockwise_orbit_deg,
            synchronize_preload=False,
        )
        return config, evaluator(config)

    def objective(measurement: np.ndarray, vector: np.ndarray) -> float:
        contact = np.linalg.norm((target - measurement) / scale)
        regularization = math.sqrt(settings.regularization_weight) * np.linalg.norm(
            (vector[:7] - reference[:7]) / settings.joint_step_rad
        )
        return float(math.hypot(float(contact), float(regularization)))

    config, evaluation = materialize(current)
    diagnostics: dict[str, Any] = {
        "method": "orientation_aware_actual_contact_joint_only_dls",
        "fixed_root_delta_cube_m": current[7:10].tolist(),
        "fixed_wrist_local_rotvec_deg": np.degrees(current[10:13]).tolist(),
        "initial_variables": current.tolist(),
        "iterations": [],
    }
    if evaluation.measurement is None:
        diagnostics["initial_safety_violations"] = list(
            evaluation.safety_violations
        )
        diagnostics["final_variables"] = current.tolist()
        return RelativeWristDLSResult(
            config,
            RelativeWristVariables.from_array(current),
            evaluation.static_result,
            diagnostics,
            "missing_initial_distal_witness",
        )
    measurement = np.asarray(evaluation.measurement, dtype=np.float64)
    stop_reason = "maximum_iterations"
    for iteration in range(settings.maximum_iterations):
        before = objective(measurement, current)
        contact_objective = float(np.linalg.norm((target - measurement) / scale))
        if contact_objective <= settings.contact_tolerance:
            stop_reason = "contact_target_converged"
            break
        jacobian = np.zeros((len(target), 7), dtype=np.float64)
        for column, name in enumerate(NON_THUMB_ACTUATORS):
            for direction in (1.0, -1.0):
                trial = current.copy()
                trial[column] += direction * finite_difference
                low, high = joint_bounds[name]
                if not low <= trial[column] <= high:
                    continue
                _, trial_evaluation = materialize(trial)
                if not trial_evaluation.safe:
                    continue
                trial_measurement = np.asarray(
                    trial_evaluation.measurement, dtype=np.float64
                )
                jacobian[:, column] = (
                    (trial_measurement - measurement)
                    / (direction * finite_difference)
                    / scale
                )
                break
        residual = (target - measurement) / scale
        variable_scale = np.full(7, settings.joint_step_rad, dtype=np.float64)
        scaled_jacobian = jacobian * variable_scale[np.newaxis, :]
        regularization_rows = math.sqrt(settings.regularization_weight) * np.eye(7)
        regularization_residual = (
            -math.sqrt(settings.regularization_weight)
            * (current[:7] - reference[:7])
            / variable_scale
        )
        system = np.vstack((scaled_jacobian, regularization_rows))
        rhs = np.concatenate((residual, regularization_residual))
        normal = system.T @ system + settings.damping**2 * np.eye(7)
        unit_step = np.linalg.solve(normal, system.T @ rhs)
        largest = float(np.max(np.abs(unit_step)))
        if largest > 1.0:
            unit_step /= largest
        step = variable_scale * unit_step
        accepted = None
        line_search: list[dict[str, Any]] = []
        for line_scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = current.copy()
            trial[:7] += line_scale * step
            if any(
                not joint_bounds[name][0] <= trial[index] <= joint_bounds[name][1]
                for index, name in enumerate(NON_THUMB_ACTUATORS)
            ):
                line_search.append({"scale": line_scale, "accepted": False, "reason": "joint_bound"})
                continue
            trial_config, trial_evaluation = materialize(trial)
            if not trial_evaluation.safe:
                line_search.append(
                    {
                        "scale": line_scale,
                        "accepted": False,
                        "reason": "unsafe_contact_trial",
                        "violations": list(trial_evaluation.safety_violations),
                    }
                )
                continue
            trial_measurement = np.asarray(
                trial_evaluation.measurement, dtype=np.float64
            )
            after = objective(trial_measurement, trial)
            improved = after + 1e-12 < before
            line_search.append(
                {"scale": line_scale, "accepted": improved, "objective": after}
            )
            if improved:
                accepted = (
                    trial,
                    trial_config,
                    trial_evaluation,
                    trial_measurement,
                    after,
                )
                break
        diagnostics["iterations"].append(
            {
                "iteration": iteration,
                "objective_before": before,
                "contact_objective_before": contact_objective,
                "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
                "line_search": line_search,
                "accepted": accepted is not None,
            }
        )
        if accepted is None:
            stop_reason = "dls_no_safe_improvement"
            break
        current, config, evaluation, measurement, after = accepted
        diagnostics["iterations"][-1]["objective_after"] = after
    final_config, final_evaluation = materialize(current)
    if bool(getattr(final_evaluation.static_result, "static_geometry_pass", False)):
        from .actual_contact_grasp_pose import apply_precontact_solution

        final_config = apply_precontact_solution(
            final_config,
            final_evaluation.static_result,
        )
    diagnostics.update(
        {
            "final_variables": current.tolist(),
            "final_contact_objective": float(
                np.linalg.norm(
                    (target - np.asarray(final_evaluation.measurement)) / scale
                )
            )
            if final_evaluation.measurement is not None
            else 1e9,
            "fixed_pose_unchanged": bool(np.array_equal(current[7:], reference[7:])),
            "final_safety_violations": list(final_evaluation.safety_violations),
        }
    )
    if not np.array_equal(current[7:], reference[7:]):
        raise AssertionError("joint-only DLS changed the exact 6D wrist grid")
    return RelativeWristDLSResult(
        final_config,
        RelativeWristVariables.from_array(current),
        final_evaluation.static_result,
        diagnostics,
        stop_reason,
    )


def physical_screen_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = record.get("static_metrics", {})
    diagnostics = record.get("dls_diagnostics", {})
    return (
        not bool(record.get("static_geometry_pass", False)),
        int(metrics.get("missing_target_witness_count", 99)),
        int(metrics.get("off_target_distal_penetrating_count", 99)),
        max(0.0, -float(metrics.get("minimum_active_nondistal_gap_m", -1.0))),
        float(metrics.get("contact_height_spread_m", 1.0)),
        float(diagnostics.get("final_contact_objective", 1e9)),
        int(record["candidate_id"]),
    )


def _physical_model_family_sha256(config: Mapping[str, Any]) -> str:
    """Hash compile-time inputs while excluding DLS-supported mutable state."""

    payload = copy.deepcopy(dict(config))
    payload.pop("candidate_metadata", None)
    payload.pop("hand_pose", None)  # evaluator updates the fixed root per trial
    payload.pop("control", None)
    cube = payload.get("cube")
    if isinstance(cube, dict):
        # Edge is the one intentional compile-time variation; each edge still
        # receives its own evaluator below.
        cube["edge_m"] = "<per-edge>"
    grasp = payload.get("grasp_pose")
    if isinstance(grasp, dict):
        grasp.pop("nominal_joint_qpos_rad", None)
    return payload_sha256(payload)


def _authenticate_physical_proposals(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Fail closed on stale hashes, mixed sources and evaluator model drift."""

    ordered = tuple(
        sorted(
            (copy.deepcopy(dict(value)) for value in records),
            key=lambda value: int(value["candidate_id"]),
        )
    )
    identifiers = [int(value["candidate_id"]) for value in ordered]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("physical-screen candidate IDs must be unique")
    sources: set[str] = set()
    model_family: str | None = None
    for record in ordered:
        candidate_id = int(record["candidate_id"])
        if candidate_id < 0 or candidate_id > MAX_STATIC_CANDIDATE_ID:
            raise ValueError("physical-screen candidate ID exceeds static namespace")
        if record.get("evaluation_status") not in (
            "proposal_only_requires_physical_screen",
            "injected_evaluator_complete",
        ):
            raise ValueError("physical-screen input is not a proposal record")
        config = record.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("physical-screen proposal has no config")
        digest = canonical_sha256(config)
        if digest != record.get("candidate_sha256"):
            raise RuntimeError("physical-screen proposal config SHA-256 mismatch")
        metadata = config.get("candidate_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("physical-screen proposal has no candidate metadata")
        search = metadata.get("lift_manipulability_search")
        if not isinstance(search, Mapping):
            raise ValueError("physical-screen proposal has no lift-search metadata")
        source_id = str(record.get("source_id"))
        if len(source_id) != 64 or any(
            character not in "0123456789abcdef" for character in source_id
        ):
            raise ValueError("physical-screen proposal source_id is not SHA-256")
        if source_id != str(metadata.get("source_measured_grasp_id")):
            raise RuntimeError("physical-screen proposal source binding mismatch")
        sources.add(source_id)
        if int(metadata.get("candidate_id", -1)) != candidate_id:
            raise RuntimeError("physical-screen proposal candidate ID binding mismatch")
        lift_digest = str(record.get("lift_candidate_sha256"))
        if len(lift_digest) != 64 or any(
            character not in "0123456789abcdef" for character in lift_digest
        ):
            raise ValueError("physical-screen proposal lift digest is not SHA-256")
        if lift_digest != str(search.get("lift_candidate_sha256")):
            raise RuntimeError("physical-screen proposal lift SHA-256 binding mismatch")
        edge = float(record["edge_m"])
        if not math.isclose(
            edge, float(config["cube"]["edge_m"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("physical-screen proposal edge binding mismatch")
        if not math.isclose(
            edge, float(search["edge_m"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("physical-screen proposal search-edge mismatch")
        orbit = float(record["clockwise_orbit_deg"])
        if not math.isclose(
            orbit,
            float(search["clockwise_orbit_deg"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("physical-screen proposal orbit binding mismatch")
        solver_orbit = float(record.get("solver_clockwise_orbit_deg", orbit))
        expected_solver_orbit = float(
            search.get("solver_clockwise_orbit_deg", orbit)
        )
        if not math.isclose(
            solver_orbit, expected_solver_orbit, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("physical-screen proposal solver-orbit mismatch")
        if int(search.get("static_candidate_id", -1)) != candidate_id:
            raise RuntimeError("physical-screen proposal static ID binding mismatch")
        if "sample_index" in record and int(search.get("sample_index", -1)) != int(
            record["sample_index"]
        ):
            raise RuntimeError("physical-screen proposal sample binding mismatch")
        if record.get("configuration_valid") is not False:
            validate_config(dict(config))
        family = _physical_model_family_sha256(config)
        if model_family is None:
            model_family = family
        elif model_family != family:
            raise RuntimeError(
                "physical-screen proposals changed evaluator model inputs"
            )
    if len(sources) > 1:
        raise RuntimeError("physical-screen proposals must come from one measured source")
    return ordered


def physical_screen_lift_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    settings: RelativeWristDLSSettings | None = None,
    evaluator_factory: Callable[
        [Mapping[str, Any]],
        tuple[Callable[[Mapping[str, Any]], Any], Mapping[str, tuple[float, float]]],
    ] = build_actual_contact_trial_evaluator,
    solver: Callable[..., Any] | None = None,
) -> LiftPhysicalScreen:
    """Run real witness evaluation and orientation-aware DLS on proposals."""

    resolved_settings = settings or RelativeWristDLSSettings()
    readiness_policy = LiftManipulabilityPolicy()
    ordered = _authenticate_physical_proposals(records)
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for record in ordered:
        grouped[float(record["edge_m"])].append(record)
    screened: list[dict[str, Any]] = []
    for edge in sorted(grouped):
        evaluator: Callable[[Mapping[str, Any]], Any] | None = None
        joint_bounds: Mapping[str, tuple[float, float]] | None = None
        for proposal in grouped[edge]:
            if proposal.get("configuration_valid") is False:
                config = copy.deepcopy(dict(proposal["config"]))
                screened.append(
                    {
                        "lift_physical_screen_schema_version": 1,
                        "candidate_id": int(proposal["candidate_id"]),
                        "lift_candidate_sha256": proposal["lift_candidate_sha256"],
                        "proposal_candidate_sha256": proposal["candidate_sha256"],
                        "candidate_sha256": canonical_sha256(config),
                        "source_id": proposal["source_id"],
                        "edge_m": float(proposal["edge_m"]),
                        "clockwise_orbit_deg": float(proposal["clockwise_orbit_deg"]),
                        "solver_clockwise_orbit_deg": float(
                            proposal.get("solver_clockwise_orbit_deg", 0.0)
                        ),
                        "static_geometry_pass": False,
                        "static_is_grasp_success_evidence": False,
                        "grasp_success": False,
                        "support_mode_readiness": {
                            "status": "configuration_invalid_before_physical_screen",
                            "policy": readiness_policy.as_dict(),
                        },
                        "configuration_valid": False,
                        "configuration_error": proposal.get("configuration_error"),
                        "eligible_for_dynamic_grasp": False,
                        "evaluation_status": "configuration_invalid_not_evaluated",
                        "dls_stop_reason": "configuration_invalid",
                        "dls_diagnostics": {"physical_dls_executed": False},
                        "static_metrics": {},
                        "static_result": {
                            "static_geometry_pass": False,
                            "not_evaluated_reason": "configuration_invalid",
                        },
                        "config": config,
                    }
                )
                continue
            if evaluator is None or joint_bounds is None:
                evaluator, joint_bounds = evaluator_factory(proposal["config"])
            search = proposal["config"]["candidate_metadata"][
                "lift_manipulability_search"
            ]
            initial = RelativeWristVariables.from_array(search["variables"])
            solve_orbit = float(
                proposal.get(
                    "solver_clockwise_orbit_deg",
                    proposal["clockwise_orbit_deg"],
                )
            )
            diagnostic_mode = search.get("mode") == "exact_post_power_diagnostic_grid"
            resolved_solver = (
                solver
                if solver is not None
                else (
                    solve_joint_only_contact_dls
                    if diagnostic_mode
                    else solve_orientation_aware_dls
                )
            )
            solution = resolved_solver(
                proposal["config"],
                clockwise_orbit_deg=solve_orbit,
                initial_variables=initial,
                settings=resolved_settings,
                evaluator=evaluator,
                joint_bounds=joint_bounds,
            )
            refined = copy.deepcopy(dict(solution.config))
            # Both production solvers derive a collision-proven Jacobian retreat
            # for precontact.  Rebase only the preload residual here; restoring
            # the proposal precontact would silently discard that proof.
            _preserve_controller_residuals(
                proposal["config"], refined, preserve_precontact=False
            )
            refined_metadata = refined.setdefault("candidate_metadata", {})
            refined_metadata.update(
                {
                    "campaign_kind": CAMPAIGN_KIND,
                    "stage": "physical_collision_witness_screen",
                    "candidate_id": int(proposal["candidate_id"]),
                    "proposal_candidate_sha256": proposal["candidate_sha256"],
                    "lift_candidate_sha256": proposal["lift_candidate_sha256"],
                }
            )
            refined_search = refined_metadata.setdefault(
                "lift_manipulability_search", {}
            )
            refined_search["final_variables"] = solution.variables.as_array().tolist()
            refined_search["final_root_delta_cube_m"] = list(
                solution.variables.root_delta_cube_m
            )
            refined_search["final_wrist_local_rotvec_deg"] = np.degrees(
                solution.variables.wrist_local_rotvec_rad
            ).tolist()
            try:
                validate_config(refined)
                configuration_valid = True
                configuration_error = None
            except (TypeError, ValueError) as error:
                configuration_valid = False
                configuration_error = str(error)
            static_result = solution.static_result
            static_payload = (
                static_result.as_dict()
                if hasattr(static_result, "as_dict")
                else copy.deepcopy(dict(static_result))
            )
            static_pass = bool(static_payload.get("static_geometry_pass", False))
            static_metrics = {
                key: static_payload.get(key)
                for key in (
                    "missing_target_witness_count",
                    "off_target_distal_penetrating_count",
                    "minimum_active_nondistal_gap_m",
                    "contact_height_spread_m",
                    "nominal_minimum_forbidden_hand_gap_m",
                    "nominal_maximum_all_distal_penetration_m",
                    "precontact_minimum_hand_gap_m",
                )
            }
            screened.append(
                {
                    "lift_physical_screen_schema_version": 1,
                    "candidate_id": int(proposal["candidate_id"]),
                    "lift_candidate_sha256": proposal["lift_candidate_sha256"],
                    "proposal_candidate_sha256": proposal["candidate_sha256"],
                    "candidate_sha256": canonical_sha256(refined),
                    "source_id": proposal["source_id"],
                    "edge_m": float(proposal["edge_m"]),
                    "clockwise_orbit_deg": float(
                        proposal["clockwise_orbit_deg"]
                    ),
                    "solver_clockwise_orbit_deg": solve_orbit,
                    "static_geometry_pass": static_pass,
                    "static_is_grasp_success_evidence": False,
                    "grasp_success": False,
                    "support_mode_readiness": {
                        "status": "requires_dynamic_grasp_trace",
                        "hard_thresholds": {
                            "minimum_force_balance_ratio": (
                                readiness_policy.minimum_force_balance_ratio
                            ),
                            "minimum_friction_margin_n": (
                                readiness_policy.minimum_friction_margin_n
                            ),
                            "maximum_line_moment_n_m": (
                                readiness_policy.maximum_line_moment_n_m
                            ),
                            "hard_height_spread_m": (
                                readiness_policy.hard_height_spread_m
                            ),
                            "minimum_tactile_path_margin_m": (
                                readiness_policy.minimum_tactile_path_margin_m
                            ),
                        },
                        "policy": readiness_policy.as_dict(),
                        "policy_sha256": payload_sha256(
                            readiness_policy.as_dict()
                        ),
                    },
                    "configuration_valid": configuration_valid,
                    "configuration_error": configuration_error,
                    "eligible_for_dynamic_grasp": (
                        static_pass and configuration_valid
                    ),
                    "evaluation_status": "physical_collision_witness_complete",
                    "dls_stop_reason": str(solution.stop_reason),
                    "dls_diagnostics": copy.deepcopy(dict(solution.diagnostics)),
                    "static_metrics": static_metrics,
                    "static_result": static_payload,
                    "config": refined,
                }
            )
    screened.sort(key=physical_screen_rank)
    report = {
        "lift_physical_screen_report_schema_version": 1,
        "complete": True,
        "input_candidate_count": len(ordered),
        "screened_candidate_count": len(screened),
        "static_geometry_pass_count": sum(
            bool(value["static_geometry_pass"]) for value in screened
        ),
        "dynamic_eligible_count": sum(
            bool(value["eligible_for_dynamic_grasp"]) for value in screened
        ),
        "configuration_invalid_count": sum(
            not bool(value["configuration_valid"]) for value in screened
        ),
        "dls_evaluated_count": sum(
            value["evaluation_status"] == "physical_collision_witness_complete"
            for value in screened
        ),
        "grasp_success_count": 0,
        "static_is_grasp_success_evidence": False,
        "dynamic_grasp_required": True,
        "support_mode_readiness_required_before_manipulation": True,
    }
    return LiftPhysicalScreen(tuple(screened), report)


def run_exact_diagnostic_physical_screen(
    source: MeasuredGraspSource,
    output_dir: str | Path,
    *,
    settings: RelativeWristDLSSettings | None = None,
    resume: bool = False,
) -> LiftPhysicalScreen:
    """Classify all 27 points, DLS-screen the valid subset, and publish it."""

    resolved_settings = settings or RelativeWristDLSSettings()
    generation = generate_exact_lift_diagnostic_grid(source)
    input_sha = _physical_screen_input_sha256(
        source, generation.records, resolved_settings
    )
    manifest_path = Path(output_dir).expanduser().resolve() / "physical_screen_manifest.json"
    if resume and manifest_path.is_file():
        return load_physical_screen_artifacts(
            output_dir, expected_input_sha256=input_sha
        )
    if manifest_path.exists() and not resume:
        raise FileExistsError(
            f"physical screen already exists; pass resume=True: {manifest_path}"
        )
    screen = physical_screen_lift_candidates(
        generation.records, settings=resolved_settings
    )
    write_physical_screen_artifacts(screen, output_dir, input_sha256=input_sha)
    return screen


def generate_lift_manipulability_candidates(
    source: MeasuredGraspSource,
    *,
    policy: LiftManipulabilityPolicy | None = None,
    evaluator: CandidateEvaluator | None = None,
) -> LiftCandidateGeneration:
    """Generate prefix-stable cells and retain a deterministic local quota."""

    resolved = policy or LiftManipulabilityPolicy()
    search_base = _registered_relative_wrist_base(source)
    source_variables = RelativeWristVariables.from_config(search_base)
    source_vector = source_variables.as_array()
    for index, name in enumerate(NON_THUMB_ACTUATORS):
        source_vector[index] = float(
            source.actual_joint_qpos_rad[ACTIVE_ACTUATORS.index(name)]
        )
    source_relative_policy = RelativeWristPoseSearchPolicy.from_config(search_base)
    bounds_config = resize_measured_source(source, resolved.edges_m[0])
    source_model, _ = build_model(bounds_config)
    joint_bounds = model_joint_bounds(source_model, bounds_config)
    retained: list[dict[str, Any]] = []
    seen_candidate_ids: set[int] = set()
    cells: list[dict[str, Any]] = []
    candidate_namespace = payload_sha256(
        {
            "kind": CAMPAIGN_KIND,
            "seed": resolved.seed,
            "non_thumb_joint_radius_rad": resolved.non_thumb_joint_radius_rad,
            "root_radius_m": list(resolved.root_radius_m),
            "wrist_radius_deg": list(resolved.wrist_radius_deg),
        }
    )
    for edge in resolved.edges_m:
        resized = resize_measured_source(source, edge)
        for orbit in resolved.clockwise_orbits_deg:
            cell_seed = int(
                payload_sha256(
                    {
                        "seed": resolved.seed,
                        "edge_m_hex": float(edge).hex(),
                        "clockwise_orbit_deg_hex": float(orbit).hex(),
                        "namespace": "lift_manipulability_cell_v1",
                    }
                )[:8],
                16,
            )
            units = _prefix_stable_samples(resolved.samples_per_cell, 13, cell_seed)
            cell_records: list[dict[str, Any]] = []
            rejected = 0
            for sample_index, unit in enumerate(units):
                vector = source_vector.copy()
                vector[:7] += unit[:7] * resolved.non_thumb_joint_radius_rad
                for index, name in enumerate(NON_THUMB_ACTUATORS):
                    vector[index] = np.clip(vector[index], *joint_bounds[name])
                vector[7:10] += unit[7:10] * np.asarray(resolved.root_radius_m)
                for axis in range(3):
                    vector[7 + axis] = np.clip(
                        vector[7 + axis],
                        *source_relative_policy.root_delta_cube_m[("x", "y", "z")[axis]],
                    )
                wrist_deg = np.degrees(vector[10:13])
                wrist_deg += unit[10:13] * np.asarray(resolved.wrist_radius_deg)
                for axis in range(3):
                    wrist_deg[axis] = np.clip(
                        wrist_deg[axis],
                        *source_relative_policy.wrist_local_rotvec_deg[("x", "y", "z")[axis]],
                    )
                norm = float(np.linalg.norm(wrist_deg))
                if norm > source_relative_policy.max_wrist_local_rotvec_norm_deg:
                    wrist_deg *= source_relative_policy.max_wrist_local_rotvec_norm_deg / norm
                vector[10:13] = np.radians(wrist_deg)
                variables = RelativeWristVariables.from_array(vector)
                candidate = materialize_relative_wrist_candidate(
                    resized,
                    variables,
                    clockwise_orbit_deg=orbit,
                    synchronize_preload=False,
                )
                distance = float(
                    candidate["candidate_metadata"]["relative_wrist_pose_search"][
                        "root_cube_distance_m"
                    ]
                )
                if not (
                    source_relative_policy.root_cube_distance_m[0] - _EPS
                    <= distance
                    <= source_relative_policy.root_cube_distance_m[1] + _EPS
                ):
                    rejected += 1
                    continue
                _preserve_controller_residuals(resized, candidate)
                candidate_id = _candidate_id(
                    source.source_id,
                    edge,
                    orbit,
                    sample_index,
                    candidate_namespace,
                )
                if candidate_id in seen_candidate_ids:
                    raise RuntimeError("deterministic static candidate ID collision")
                seen_candidate_ids.add(candidate_id)
                lift_candidate_sha = _lift_candidate_sha256(
                    source.source_id,
                    edge,
                    orbit,
                    sample_index,
                    candidate_namespace,
                )
                candidate["candidate_metadata"].update(
                    {
                        "campaign_kind": CAMPAIGN_KIND,
                        "candidate_id": candidate_id,
                        "source_measured_grasp_id": source.source_id,
                        "lift_manipulability_search": {
                            "edge_m": edge,
                            "clockwise_orbit_deg": orbit,
                            "sample_index": sample_index,
                            "lift_candidate_sha256": lift_candidate_sha,
                            "static_candidate_id": candidate_id,
                            "downstream_id_limits": {
                                "maximum_static_candidate_id": (
                                    MAX_STATIC_CANDIDATE_ID
                                ),
                                "controller_seed_stride": 16,
                                "trust_candidate_stride": 1000,
                            },
                            "cube_pose_sampled": False,
                            "hand_root_fixed_during_simulation": True,
                            "variables": variables.as_array().tolist(),
                        },
                    }
                )
                try:
                    validate_config(candidate)
                except (TypeError, ValueError):
                    # Registered palm/finger orientation limits and the true
                    # actuator limits are part of the deterministic boundary,
                    # not a fatal error for the whole stratum.
                    rejected += 1
                    continue
                realized_scale = np.asarray(
                    (
                        *([resolved.non_thumb_joint_radius_rad] * 7),
                        *resolved.root_radius_m,
                        *np.radians(resolved.wrist_radius_deg),
                    ),
                    dtype=np.float64,
                )
                normalized = float(
                    np.linalg.norm((vector - source_vector) / realized_scale)
                    / math.sqrt(len(vector))
                )
                readiness = _readiness_dict(evaluator(candidate) if evaluator else None)
                record = {
                    "lift_manipulability_candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "lift_candidate_sha256": lift_candidate_sha,
                    "source_id": source.source_id,
                    "edge_m": edge,
                    "clockwise_orbit_deg": orbit,
                    "sample_index": sample_index,
                    "normalized_perturbation_norm": normalized,
                    "candidate_sha256": canonical_sha256(candidate),
                    "readiness": readiness,
                    "evaluation_status": (
                        "injected_evaluator_complete"
                        if readiness is not None
                        else "proposal_only_requires_physical_screen"
                    ),
                    "eligible_for_dynamic_grasp": False,
                    "config": candidate,
                }
                cell_records.append(record)
            ranked = sorted(cell_records, key=lift_candidate_rank)
            selected = ranked[: resolved.retained_per_cell]
            if len(selected) != resolved.retained_per_cell:
                raise RuntimeError(
                    "physical-boundary filtering left a deficient cell: "
                    f"edge={edge * 1000:g} mm orbit={orbit:g} deg retained="
                    f"{len(selected)}/{resolved.retained_per_cell}"
                )
            retained.extend(selected)
            cells.append(
                {
                    "edge_m": edge,
                    "clockwise_orbit_deg": orbit,
                    "declared_sample_count": resolved.samples_per_cell,
                    "generated_candidate_count": len(cell_records),
                    "boundary_rejected_count": rejected,
                    "retained_candidate_count": len(selected),
                    "retained_candidate_ids": [item["candidate_id"] for item in selected],
                }
            )
    report = {
        "lift_candidate_generation_schema_version": GENERATION_SCHEMA_VERSION,
        "complete": True,
        "source_id": source.source_id,
        "policy": resolved.as_dict(),
        "declared_cell_count": len(resolved.edges_m) * len(resolved.clockwise_orbits_deg),
        "declared_sample_count": (
            len(resolved.edges_m)
            * len(resolved.clockwise_orbits_deg)
            * resolved.samples_per_cell
        ),
        "generated_candidate_count": sum(item["generated_candidate_count"] for item in cells),
        "retained_candidate_count": len(retained),
        "proposal_only": evaluator is None,
        "physical_screen_required_before_dynamic_grasp": evaluator is None,
        "cells": cells,
    }
    return LiftCandidateGeneration(tuple(retained), report)


def select_lift_candidates_with_edge_quota(
    records: Sequence[Mapping[str, Any]],
    *,
    edges_m: Sequence[float],
    minimum_per_edge: int,
    selected_total: int,
) -> tuple[dict[str, Any], ...]:
    """Select each edge quota first, then fill by deterministic global rank."""

    edges = _positive_tuple(tuple(edges_m), "edges_m")
    if minimum_per_edge <= 0 or selected_total < minimum_per_edge * len(edges):
        raise ValueError("selection budget cannot satisfy edge quotas")
    unique: dict[str, dict[str, Any]] = {}
    for value in sorted(records, key=lift_candidate_rank):
        record = copy.deepcopy(dict(value))
        unique.setdefault(str(record["candidate_sha256"]), record)
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for record in unique.values():
        grouped[float(record["edge_m"])].append(record)
    selected: list[dict[str, Any]] = []
    for edge in edges:
        candidates = sorted(grouped.get(edge, ()), key=lift_candidate_rank)
        if len(candidates) < minimum_per_edge:
            raise RuntimeError(f"edge {edge * 1000:g} mm cannot satisfy its quota")
        selected.extend(candidates[:minimum_per_edge])
    selected_hashes = {str(value["candidate_sha256"]) for value in selected}
    for record in sorted(unique.values(), key=lift_candidate_rank):
        if len(selected) >= selected_total:
            break
        if str(record["candidate_sha256"]) not in selected_hashes:
            selected.append(record)
            selected_hashes.add(str(record["candidate_sha256"]))
    if len(selected) < selected_total:
        raise RuntimeError("candidate pool cannot satisfy selected_total")
    return tuple(selected[:selected_total])


@dataclass(frozen=True, slots=True)
class FreeBodySearchHooks:
    prepare_checkpoint: Callable[..., Any]
    manipulation_bounds: Callable[..., Mapping[str, Sequence[float]]]
    run_probes: Callable[..., Sequence[Mapping[str, Any]]]
    fit_response: Callable[..., Mapping[str, Any]]
    generate_deltas: Callable[..., Sequence[Mapping[str, float]]]
    run_full_reset: Callable[..., Mapping[str, Any]]
    full_reset_attested: bool = False

    @classmethod
    def existing_runner(cls) -> "FreeBodySearchHooks":
        from .actual_contact_manipulation import (
            fit_response_jacobian,
            generate_trust_region_deltas,
            manipulation_delta_bounds,
            prepare_grasp_checkpoint,
            run_checkpoint_probe_set,
            run_full_reset_candidates,
        )

        return cls(
            prepare_checkpoint=prepare_grasp_checkpoint,
            manipulation_bounds=manipulation_delta_bounds,
            run_probes=run_checkpoint_probe_set,
            fit_response=fit_response_jacobian,
            generate_deltas=generate_trust_region_deltas,
            run_full_reset=run_full_reset_candidates,
            full_reset_attested=True,
        )


def run_free_body_probe_search(
    source: MeasuredGraspSource,
    *,
    budget: Any,
    workers: int,
    hooks: FreeBodySearchHooks | None = None,
) -> dict[str, Any]:
    """Bridge one verified grasp into the existing probe/full-reset runner."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    source = reauthenticate_measured_grasp_source(source)
    resolved = hooks or FreeBodySearchHooks.existing_runner()
    if not source.readiness.passed:
        raise ValueError(
            "measured grasp is not support-mode ready for free-body manipulation"
        )
    checkpoint = resolved.prepare_checkpoint(
        source.config,
        source.trace_path,
        source.result,
    )
    bounds = resolved.manipulation_bounds(checkpoint.model, source.config)
    probes = tuple(resolved.run_probes(checkpoint, budget=budget))
    target = (0.0, 0.0, float(budget.target_upward_m), 0.0, 0.0, 0.0)
    response = resolved.fit_response(
        probes,
        bounds,
        config=source.config,
        target_response_6d=target,
        ridge=float(budget.ridge),
        inward_preload_weight=float(budget.inward_preload_weight),
    )
    deltas = tuple(
        resolved.generate_deltas(
            response,
            bounds,
            count=int(budget.trust_candidate_count),
            seed=int(budget.seed),
            trust_radius_fraction=float(budget.trust_radius_fraction),
            wide_candidate_fraction=float(budget.wide_candidate_fraction),
            wide_radius_fraction=float(budget.wide_radius_fraction),
        )
    )
    execution = resolved.run_full_reset(
        source.config,
        deltas,
        workers=workers,
        candidate_ids=range(len(deltas)),
    )
    return {
        "checkpoint_search_only": True,
        "source_id": source.source_id,
        "support_mode_readiness": source.readiness.as_dict(),
        "final_candidates_rerun_from_initial_state": bool(
            resolved.full_reset_attested
        ),
        "execution_attested": bool(resolved.full_reset_attested),
        "probe_count": len(probes),
        "response_model": copy.deepcopy(dict(response)),
        "candidate_count": len(deltas),
        "execution": execution,
    }


@dataclass(frozen=True, slots=True)
class LiftGenerationCampaign:
    output_dir: str
    manifest: dict[str, Any]
    ledger: dict[str, Any]
    retained_records: tuple[dict[str, Any], ...]
    selected_records: tuple[dict[str, Any], ...]


def _append_ledger_stage(
    ledger: dict[str, Any],
    *,
    name: str,
    artifacts: Sequence[Path],
    output: Path,
    summary: Mapping[str, Any],
) -> None:
    previous = ledger["stages"][-1]["stage_sha256"] if ledger["stages"] else None
    record = {
        "name": name,
        "previous_stage_sha256": previous,
        "artifacts": [
            {
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
            for path in artifacts
        ],
        "summary": copy.deepcopy(dict(summary)),
    }
    record["stage_sha256"] = payload_sha256(record)
    ledger["stages"].append(record)


def _authenticate_ledger(output: Path, ledger: Mapping[str, Any]) -> None:
    if ledger.get("lift_manipulability_ledger_schema_version") != LEDGER_SCHEMA_VERSION:
        raise RuntimeError("lift-manipulability ledger version changed")
    previous = None
    for raw in ledger.get("stages", ()):
        record = dict(raw)
        digest = record.pop("stage_sha256", None)
        if record.get("previous_stage_sha256") != previous:
            raise RuntimeError("lift-manipulability ledger chain changed")
        if digest != payload_sha256(record):
            raise RuntimeError("lift-manipulability ledger stage hash changed")
        for artifact in record["artifacts"]:
            path = _confined_artifact(output, artifact["path"], "ledger artifact")
            if not path.is_file() or file_sha256(path) != artifact["sha256"]:
                raise RuntimeError("lift-manipulability ledger artifact changed")
        previous = digest


def run_or_resume_generation_campaign(
    source: MeasuredGraspSource,
    output_dir: str | Path,
    *,
    policy: LiftManipulabilityPolicy | None = None,
    evaluator: CandidateEvaluator | None = None,
    resume: bool = False,
) -> LiftGenerationCampaign:
    """Atomically generate, resume, or authenticate a focused campaign.

    A stage becomes reusable only after its artifact hashes are committed to
    the ledger.  Files written immediately before a power loss are therefore
    harmless: an uncommitted stage is regenerated deterministically, whereas
    any change to a committed artifact is rejected.
    """

    resolved = policy or LiftManipulabilityPolicy()
    if evaluator is not None:
        raise ValueError(
            "persistent proposal campaigns do not accept an unversioned evaluator; "
            "run the physical-screen stage instead"
        )
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "campaign_manifest.json"
    ledger_path = output / "campaign_ledger.json"
    source_snapshot_path = output / "authenticated_source.json"
    retained_path = output / "static" / "retained_candidates.json"
    selection_path = output / "static" / "selection.json"
    dependency_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "xhand_grasp" / "config.py",
        REPO_ROOT / "xhand_grasp" / "experiment.py",
        REPO_ROOT / "xhand_grasp" / "scene.py",
        REPO_ROOT / "xhand_grasp" / "relative_wrist_pose.py",
        REPO_ROOT / "xhand_grasp" / "tuning" / "relative_wrist_pose_search.py",
        REPO_ROOT / "xhand_grasp" / "tuning" / "actual_contact_grasp_pose.py",
        REPO_ROOT / "xhand_grasp" / "actual_contact_grasp_pose_catalog.py",
        REPO_ROOT / "xhand_grasp" / "experiments" / "__init__.py",
        REPO_ROOT
        / "xhand_grasp"
        / "experiments"
        / "opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
        "smooth_vertical_lift.py",
        _V11_TEMPLATE,
    )
    input_payload = {
        "campaign_kind": CAMPAIGN_KIND,
        "source": source.identity(),
        "policy": resolved.as_dict(),
        "source_code_sha256": {
            str(path.relative_to(REPO_ROOT)): file_sha256(path)
            for path in dependency_paths
        },
        "model": {
            "path": str((REPO_ROOT / "xhand_left.xml").resolve()),
            "sha256": file_sha256(REPO_ROOT / "xhand_left.xml"),
        },
        "uv_lock": {
            "path": str((REPO_ROOT / "uv.lock").resolve()),
            "sha256": file_sha256(REPO_ROOT / "uv.lock"),
        },
    }
    input_sha = payload_sha256(input_payload)
    expected_stage_order = (
        "authenticated_source",
        "candidate_generation",
        "per_edge_selection",
        "complete_manifest",
    )
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("campaign exists; pass resume=True to authenticate it")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("lift_manipulability_manifest_schema_version")
            != MANIFEST_SCHEMA_VERSION
        ):
            raise RuntimeError("lift-manipulability manifest version changed")
        if manifest.get("input_sha256") != input_sha:
            raise RuntimeError("lift-manipulability resume input changed")
        ledger = (
            json.loads(ledger_path.read_text(encoding="utf-8"))
            if ledger_path.is_file()
            else {
                "lift_manipulability_ledger_schema_version": LEDGER_SCHEMA_VERSION,
                "input_sha256": input_sha,
                "stages": [],
            }
        )
    else:
        if output.exists():
            unexpected = [
                path
                for path in output.iterdir()
                if not (
                    path.is_file()
                    and path.name.startswith(".campaign_manifest.json.")
                    and path.name.endswith(".tmp")
                )
            ]
            if unexpected:
                raise FileExistsError(
                    "output directory is non-empty and has no campaign manifest"
                )
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "lift_manipulability_manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "complete": False,
            "input_sha256": input_sha,
            "input": input_payload,
        }
        _atomic_json(manifest_path, manifest)
        ledger = {
            "lift_manipulability_ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "input_sha256": input_sha,
            "stages": [],
        }
    if ledger.get("input_sha256") != input_sha:
        raise RuntimeError("lift-manipulability ledger input changed")
    _authenticate_ledger(output, ledger)
    stage_names = tuple(str(stage.get("name")) for stage in ledger["stages"])
    if stage_names != expected_stage_order[: len(stage_names)]:
        raise RuntimeError("lift-manipulability ledger stage order changed")

    if "authenticated_source" not in stage_names:
        _atomic_json(source_snapshot_path, source.identity())
        _append_ledger_stage(
            ledger,
            name="authenticated_source",
            artifacts=(source_snapshot_path,),
            output=output,
            summary={"source_id": source.source_id},
        )
        _atomic_json(ledger_path, ledger)
        stage_names = (*stage_names, "authenticated_source")

    if "candidate_generation" in stage_names:
        retained_payload = json.loads(retained_path.read_text(encoding="utf-8"))
        retained_records = tuple(copy.deepcopy(retained_payload["records"]))
        generation_report = copy.deepcopy(retained_payload["generation"])
    else:
        generated = generate_lift_manipulability_candidates(
            source, policy=resolved, evaluator=evaluator
        )
        retained_records = generated.records
        generation_report = generated.report
        retained_payload = {
            "complete": True,
            "generation": generation_report,
            "records": list(retained_records),
        }
        _atomic_json(retained_path, retained_payload)
        _append_ledger_stage(
            ledger,
            name="candidate_generation",
            artifacts=(retained_path,),
            output=output,
            summary={
                "retained_candidate_count": len(retained_records),
                "declared_sample_count": generation_report["declared_sample_count"],
            },
        )
        _atomic_json(ledger_path, ledger)
        stage_names = (*stage_names, "candidate_generation")

    if "per_edge_selection" in stage_names:
        selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = tuple(copy.deepcopy(selection_payload["records"]))
    else:
        selected = select_lift_candidates_with_edge_quota(
            retained_records,
            edges_m=resolved.edges_m,
            minimum_per_edge=resolved.minimum_per_edge,
            selected_total=resolved.selected_total,
        )
        selection_payload = {
            "complete": True,
            "minimum_per_edge": resolved.minimum_per_edge,
            "selected_total": resolved.selected_total,
            "selected_candidate_ids": [item["candidate_id"] for item in selected],
            "records": list(selected),
        }
        _atomic_json(selection_path, selection_payload)
        _append_ledger_stage(
            ledger,
            name="per_edge_selection",
            artifacts=(selection_path,),
            output=output,
            summary={"selected_candidate_count": len(selected)},
        )
        _atomic_json(ledger_path, ledger)
        stage_names = (*stage_names, "per_edge_selection")

    manifest = {
        "lift_manipulability_manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "complete": True,
        "input_sha256": input_sha,
        "input": input_payload,
        "retained_candidates_path": str(retained_path.relative_to(output)),
        "selection_path": str(selection_path.relative_to(output)),
        "ledger_path": str(ledger_path.relative_to(output)),
        "retained_candidate_count": len(retained_records),
        "selected_candidate_count": len(selected),
        "proposal_only": True,
        "physical_screen_required_before_dynamic_grasp": True,
    }
    _atomic_json(manifest_path, manifest)
    if "complete_manifest" not in stage_names:
        _append_ledger_stage(
            ledger,
            name="complete_manifest",
            artifacts=(manifest_path,),
            output=output,
            summary={"complete": True},
        )
        _atomic_json(ledger_path, ledger)
    return LiftGenerationCampaign(
        output_dir=str(output),
        manifest=manifest,
        ledger=ledger,
        retained_records=tuple(retained_records),
        selected_records=tuple(selected),
    )


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--measured-config")
    parser.add_argument("--measured-result")
    parser.add_argument("--measured-trace")
    parser.add_argument("--catalog")
    parser.add_argument("--trajectory", default="best_nominal")


def _load_cli_source(args: argparse.Namespace, policy: LiftManipulabilityPolicy) -> MeasuredGraspSource:
    return load_measured_grasp_source(
        measured_config_path=args.measured_config,
        measured_result_path=args.measured_result,
        measured_trace_path=args.measured_trace,
        catalog_path=args.catalog,
        trajectory=args.trajectory,
        policy=policy,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Focused schema-v11 lift-manipulability geometry search"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect-source")
    _source_arguments(inspect_parser)
    generate_parser = subparsers.add_parser("generate")
    _source_arguments(generate_parser)
    generate_parser.add_argument("--output-dir", required=True)
    generate_parser.add_argument("--samples-per-cell", type=int, default=128)
    generate_parser.add_argument("--retained-per-cell", type=int, default=4)
    generate_parser.add_argument("--minimum-per-edge", type=int, default=8)
    generate_parser.add_argument("--selected-total", type=int, default=32)
    generate_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate_parser.add_argument("--resume", action="store_true")
    physical_parser = subparsers.add_parser(
        "physical-screen",
        help=(
            "classify the exact 27-point grid and collision/DLS-screen its "
            "configuration-valid subset (not grasp evidence)"
        ),
    )
    _source_arguments(physical_parser)
    physical_parser.add_argument("--output-dir", required=True)
    physical_parser.add_argument("--dls-iterations", type=int, default=6)
    physical_parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if bool(args.measured_config) == bool(args.catalog):
        raise SystemExit("select exactly one of --measured-config or --catalog")
    policy = LiftManipulabilityPolicy(
        samples_per_cell=getattr(args, "samples_per_cell", 128),
        retained_per_cell=getattr(args, "retained_per_cell", 4),
        minimum_per_edge=getattr(args, "minimum_per_edge", 8),
        selected_total=getattr(args, "selected_total", 32),
        seed=getattr(args, "seed", DEFAULT_SEED),
    )
    source = _load_cli_source(args, policy)
    if args.command == "inspect-source":
        print(json.dumps(source.identity(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "physical-screen":
        screen = run_exact_diagnostic_physical_screen(
            source,
            args.output_dir,
            settings=RelativeWristDLSSettings(
                maximum_iterations=int(args.dls_iterations)
            ),
            resume=bool(args.resume),
        )
        print(
            json.dumps(
                {
                    "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                    "screened_candidate_count": len(screen.records),
                    "configuration_invalid_count": screen.report[
                        "configuration_invalid_count"
                    ],
                    "dls_evaluated_count": screen.report["dls_evaluated_count"],
                    "static_geometry_pass_count": screen.report[
                        "static_geometry_pass_count"
                    ],
                    "dynamic_eligible_count": screen.report[
                        "dynamic_eligible_count"
                    ],
                    "grasp_success_count": 0,
                    "next_stage": "run_supported_dynamic_grasp_screen",
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    campaign = run_or_resume_generation_campaign(
        source,
        args.output_dir,
        policy=policy,
        resume=bool(args.resume),
    )
    print(
        json.dumps(
            {
                "output_dir": campaign.output_dir,
                "retained_candidate_count": len(campaign.retained_records),
                "selected_candidate_count": len(campaign.selected_records),
                "complete": campaign.manifest["complete"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_KIND",
    "CANDIDATE_ID_BASE",
    "DEFAULT_EDGES_M",
    "DEFAULT_ORBITS_DEG",
    "DIAGNOSTIC_CLOCKWISE_ORBITS_DEG",
    "DIAGNOSTIC_ROOT_X_OFFSETS_M",
    "DIAGNOSTIC_WRIST_LOCAL_ROTVECS_DEG",
    "FreeBodySearchHooks",
    "LiftCandidateGeneration",
    "LiftGenerationCampaign",
    "LiftManipulabilityPolicy",
    "LiftPhysicalScreen",
    "LiftSupportedDynamicScreen",
    "MAX_STATIC_CANDIDATE_ID",
    "MeasuredGraspSource",
    "SupportModeObservation",
    "SupportModeReadiness",
    "build_argument_parser",
    "evaluate_support_mode_readiness",
    "dynamic_grasp_inputs_from_physical_screen",
    "generate_exact_lift_diagnostic_grid",
    "generate_lift_manipulability_candidates",
    "lift_candidate_rank",
    "load_measured_grasp_source",
    "load_physical_screen_artifacts",
    "main",
    "physical_screen_lift_candidates",
    "physical_screen_rank",
    "reauthenticate_measured_grasp_source",
    "resize_measured_source",
    "run_free_body_probe_search",
    "run_exact_diagnostic_physical_screen",
    "run_or_resume_generation_campaign",
    "run_supported_dynamic_grasp_screen",
    "select_lift_candidates_with_edge_quota",
    "solve_joint_only_contact_dls",
    "write_physical_screen_artifacts",
]
