"""Deterministic schema-v7 high-thumb, variable-size tuning campaign.

The campaign starts from the six independently validated schema-v6
pose-preserving acquisitions.  A source contributes hand/object geometry and
closure evidence, but never authorises a candidate result: every schema-v7
candidate is rebuilt and rerun.  For each requested cube size its initial
world pose is resolved once from ``center_xy``, source yaw, and the support
height.  Search code may then move only the fixed hand root and active-finger
commands; it must not sample or reset the free cube pose.

The module intentionally keeps generation, ranking, diversity selection and
resume validation pure (or filesystem-only) so they are cheap to unit test.
MuJoCo imports are confined to the default static and dynamic executors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import file_sha256, json_text, write_json
from ..config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, load_config, validate_config
from ..experiment import resolve_experiment
from ..scene import (
    cube_vertical_half_extent_m,
    rpy_degrees_to_quaternion,
    rpy_degrees_to_rotation_matrix,
)
from .pose_preserving_grasp import (
    acquisition_succeeded,
    grasp_succeeded,
    pose_preservation_succeeded,
)
from .pose_preserving_seed_campaign import canonical_sha256
from .pose_preserving_seed_dynamic import (
    CLOSE_GROUP_ACTUATORS,
    CLOSE_GROUP_ORDER,
    close_profile_from_group_starts,
)


EXPERIMENT_ID = (
    "left_opposed_face_palm_down_high_thumb_variable_size_"
    "pose_preserving_grasp_then_lift"
)
SOURCE_EXPERIMENT_ID = "left_opposed_face_palm_down_pose_preserving_grasp"
CAMPAIGN_KIND = "high_thumb_variable_size_pose_preserving_grasp_then_lift"
CAMPAIGN_SCHEMA_VERSION = 1
CANDIDATE_RESULT_SCHEMA_VERSION = 1
EXPECTED_SOURCE_COUNT = 6
THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
DEFAULT_SEED = 20260821
DEFAULT_SOURCE_CATALOG = Path(
    "artifacts/left_opposed_face_palm_down_pose_preserving_grasp/"
    "six_seed_dynamic_tune/trajectory_catalog/catalog.json"
)
DEFAULT_TEMPLATE = Path(
    "grasp_configs/"
    "left_opposed_face_palm_down_high_thumb_variable_size_"
    "pose_preserving_grasp_then_lift.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/"
    "left_opposed_face_palm_down_high_thumb_variable_size_pose_preserving"
)
DEFAULT_COARSE_EDGES_M = tuple(value / 1000.0 for value in range(52, 71, 2))
DEFAULT_COARSE_THUMB_TARGETS_RAD = (1.25, 1.30, 1.35, 1.40, 1.45)
DEFAULT_FIXED_MASS_KG = 0.160
DEFAULT_FRICTION = 0.8
DEFAULT_CENTER_XY_M = (0.071, -0.027)
DEFAULT_EDGE_FINE_STEP_M = 0.001
DEFAULT_THUMB_FINE_STEP_RAD = 0.01

_STATIC_STAGE_BASE = 71_000_000_000_000
_LOCAL_STAGE_BASE = 72_000_000_000_000
_FINE_STAGE_BASE = 73_000_000_000_000
_EXACT_STAGE_BASE = 74_000_000_000_000
_LIFT_STAGE_BASE = 75_000_000_000_000
_PERTURB_STAGE_BASE = 76_000_000_000_000
_ID_STRIDE = 1_000_000

_DEFAULT_HAND_RPY_RADIUS_DEG = np.asarray((0.75, 1.50, 0.75), dtype=np.float64)
_DEFAULT_CUBE_IN_ROOT_RADIUS_M = np.asarray((0.0020, 0.0020, 0.0020))
_DEFAULT_PREGRASP_RADIUS_RAD = 0.08
_DEFAULT_TERMINAL_RADIUS_RAD = 0.08
_DEFAULT_CLOSE_START_RADIUS = np.asarray((0.10, 0.08, 0.08))

StaticExecutor = Callable[[Sequence[Mapping[str, Any]], int], Sequence[Mapping[str, Any]]]
CandidateExecutor = Callable[
    [Sequence[Mapping[str, Any]], int], Sequence[Mapping[str, Any]]
]


@dataclass(frozen=True, slots=True)
class HighThumbCampaignBudget:
    """Versioned stage limits; defaults implement the declared full search."""

    static_samples_per_cell: int = 20_000
    static_retain_per_cell: int = 8
    dynamic_candidate_limit: int = 400
    local_refine_sizes_per_target: int = 4
    local_refine_seeds_per_size: int = 2
    local_refine_per_seed: int = 64
    fine_dynamic_limit: int = 320
    exact_reverify_limit: int = 48
    selected_grasp_count: int = 12
    selected_lift_seed_count: int = 6
    lift_candidates_per_seed: int = 256
    perturbations_per_grasp: int = 16

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"budget.{name} must be a positive integer")
        if self.static_samples_per_cell >= _ID_STRIDE:
            raise ValueError("static_samples_per_cell must be below the ID stride")

    @property
    def coarse_static_sample_count(self) -> int:
        return (
            len(DEFAULT_COARSE_EDGES_M)
            * len(DEFAULT_COARSE_THUMB_TARGETS_RAD)
            * self.static_samples_per_cell
        )


@dataclass(frozen=True, slots=True)
class HighThumbCampaignPolicy:
    coarse_edges_m: tuple[float, ...] = DEFAULT_COARSE_EDGES_M
    coarse_thumb_targets_rad: tuple[float, ...] = DEFAULT_COARSE_THUMB_TARGETS_RAD
    edge_fine_step_m: float = DEFAULT_EDGE_FINE_STEP_M
    thumb_fine_step_rad: float = DEFAULT_THUMB_FINE_STEP_RAD
    fixed_mass_kg: float = DEFAULT_FIXED_MASS_KG
    friction: float = DEFAULT_FRICTION
    cube_center_xy_m: tuple[float, float] = DEFAULT_CENTER_XY_M
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        edges = _strict_axis(self.coarse_edges_m, "coarse_edges_m", minimum=0.001)
        targets = _strict_axis(
            self.coarse_thumb_targets_rad,
            "coarse_thumb_targets_rad",
            minimum=1.25,
            maximum=1.45,
        )
        object.__setattr__(self, "coarse_edges_m", edges)
        object.__setattr__(self, "coarse_thumb_targets_rad", targets)
        for name in ("edge_fine_step_m", "thumb_fine_step_rad", "fixed_mass_kg", "friction"):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        center = _finite_vector(self.cube_center_xy_m, 2, "cube_center_xy_m")
        object.__setattr__(self, "cube_center_xy_m", tuple(center.tolist()))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class DiversitySelection:
    selected: tuple[dict[str, Any], ...]
    requested_count: int
    satisfied: bool
    distinct_edges: int
    distinct_seed_families: int
    bend_band_counts: Mapping[str, int]
    deficiencies: tuple[str, ...]


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _finite_vector(values: Any, length: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain {length} finite values") from error
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return vector.copy()


def _strict_axis(
    values: Iterable[Any],
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, ...]:
    axis = tuple(_finite(value, label) for value in values)
    if not axis or tuple(sorted(set(axis))) != axis:
        raise ValueError(f"{label} must be non-empty, unique and increasing")
    if minimum is not None and axis[0] < minimum - 1e-12:
        raise ValueError(f"{label} is below {minimum}")
    if maximum is not None and axis[-1] > maximum + 1e-12:
        raise ValueError(f"{label} is above {maximum}")
    return axis


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _safe_catalog_member(catalog: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"catalog {label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"catalog {label} must be a safe relative path")
    root = catalog.parent.resolve()
    member = (root / relative).resolve()
    try:
        member.relative_to(root)
    except ValueError as error:
        raise ValueError(f"catalog {label} escapes its directory") from error
    if not member.is_file():
        raise FileNotFoundError(f"catalog {label} does not exist: {member}")
    return member


def _verified_artifact(
    catalog: Path,
    artifacts: Mapping[str, Any],
    field: str,
) -> tuple[Path, str]:
    member = _safe_catalog_member(catalog, artifacts.get(field), field)
    hashes = artifacts.get("sha256")
    expected = hashes.get(field) if isinstance(hashes, Mapping) else None
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"catalog has no valid SHA-256 for {field}")
    actual = file_sha256(member)
    if actual != expected.lower():
        raise ValueError(f"catalog SHA-256 mismatch for {field}: {member}")
    return member, actual


def cube_world_pose_for_size(
    source_config: Mapping[str, Any],
    edge_m: float,
    center_xy_m: Sequence[float] = DEFAULT_CENTER_XY_M,
) -> dict[str, list[float]]:
    """Resolve the immutable initial pose for one source-family/size pair."""

    edge = _finite(edge_m, "edge_m")
    if edge <= 0.0:
        raise ValueError("edge_m must be positive")
    center = _finite_vector(center_xy_m, 2, "center_xy_m")
    cube = source_config["cube"]
    rpy = _finite_vector(cube.get("rpy_deg", (0.0, 0.0, 0.0)), 3, "cube.rpy_deg")
    rotation = rpy_degrees_to_rotation_matrix(rpy)
    vertical = cube_vertical_half_extent_m(edge, rotation)
    support = _finite(source_config["scene"]["support_top_z_m"], "support_top_z_m")
    z_offset = _finite(cube.get("z_offset_m", 0.0), "cube.z_offset_m")
    return {
        "position_m": [float(center[0]), float(center[1]), support + vertical + z_offset],
        "quaternion_wxyz": rpy_degrees_to_quaternion(rpy).tolist(),
        "rpy_deg": rpy.tolist(),
    }


def _trace_acquisition_qpos(config: Mapping[str, Any], trace_path: Path) -> dict[str, float]:
    with np.load(trace_path, allow_pickle=False) as trace:
        required = {"actuator_order", "joint_qpos", "grasp_acquisition_step"}
        missing = required - set(trace.files)
        if missing:
            raise ValueError("source trace is missing: " + ", ".join(sorted(missing)))
        order = tuple(str(value) for value in np.asarray(trace["actuator_order"]))
        qpos = np.asarray(trace["joint_qpos"], dtype=np.float64)
        step = int(np.asarray(trace["grasp_acquisition_step"]).reshape(()))
    if len(order) != len(set(order)) or not set(ACTIVE_ACTUATORS).issubset(order):
        raise ValueError("source trace actuator_order is invalid")
    if qpos.ndim != 2 or qpos.shape[1] != len(order) or not np.isfinite(qpos).all():
        raise ValueError("source trace joint_qpos is invalid")
    if not 0 <= step < qpos.shape[0]:
        raise ValueError("source grasp acquisition step is out of range")
    return {name: float(qpos[step, order.index(name)]) for name in ACTIVE_ACTUATORS}


def load_authenticated_v6_sources(
    catalog_path: str | Path,
    *,
    expected_count: int = EXPECTED_SOURCE_COUNT,
) -> tuple[dict[str, Any], ...]:
    """Authenticate and load exactly the six validated schema-v6 sources."""

    expected = _positive_int(expected_count, "expected_count")
    catalog = Path(catalog_path).expanduser().resolve()
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("source catalog must be a mapping")
    if payload.get("pose_preserving_seed_catalog_schema_version") != 1:
        raise ValueError("unsupported pose-preserving source catalog schema")
    if payload.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("source catalog belongs to the wrong experiment")
    if payload.get("validation_scope") != "grasp_acquisition_pose_preserved":
        raise ValueError("source catalog has the wrong validation scope")
    if payload.get("all_sources_authenticated") is not True:
        raise ValueError("source catalog did not authenticate every source")
    entries = payload.get("trajectories")
    if not isinstance(entries, list) or len(entries) != expected:
        raise ValueError(f"source catalog must contain exactly {expected} trajectories")
    if payload.get("validated_grasp_count") != expected:
        raise ValueError("every v6 source must be a validated grasp")

    catalog_sha = file_sha256(catalog)
    result: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    seen_trajectories: set[str] = set()
    for order, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError("source trajectory entries must be mappings")
        entry = dict(raw_entry)
        trajectory_id = str(entry.get("trajectory_id", ""))
        family = str(entry.get("source_id", ""))
        if not trajectory_id or not family:
            raise ValueError("source trajectory is missing its IDs")
        if family in seen_families or trajectory_id in seen_trajectories:
            raise ValueError("source family and trajectory IDs must be unique")
        seen_families.add(family)
        seen_trajectories.add(trajectory_id)
        validation = entry.get("grasp_validation")
        if (
            entry.get("classification") != "validated_pose_preserving_grasp_acquisition"
            or entry.get("rerun_grasp_success") is not True
            or not isinstance(validation, Mapping)
            or validation.get("passed") is not True
        ):
            raise ValueError(f"source {trajectory_id} is not a validated v6 grasp")
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("source trajectory has no artifact mapping")
        config_path, config_sha = _verified_artifact(catalog, artifacts, "resolved_config")
        trace_path, trace_sha = _verified_artifact(catalog, artifacts, "trace")
        result_path, result_sha = _verified_artifact(catalog, artifacts, "result")
        config = load_config(config_path)
        if int(config.get("schema_version", 0)) != 6:
            raise ValueError("source resolved config must use schema version 6")
        if config.get("experiment_id") != SOURCE_EXPERIMENT_ID:
            raise ValueError("source resolved config belongs to the wrong experiment")
        if config["control"]["manipulation_delta_rad"] != {
            name: 0.0 for name in ACTIVE_ACTUATORS
        }:
            raise ValueError("source grasp must have zero manipulation delta")
        acquisition_qpos = _trace_acquisition_qpos(config, trace_path)
        initial_pose = cube_world_pose_for_size(
            config,
            float(config["cube"]["edge_m"]),
            config["cube"]["center_xy_m"],
        )
        source_payload = {
            "source_order": order,
            "source_family_id": family,
            "source_trajectory_id": trajectory_id,
            "config": copy.deepcopy(config),
            "cube": copy.deepcopy(config["cube"]),
            "scene": copy.deepcopy(config["scene"]),
            "contact_topology": copy.deepcopy(config["contact_topology"]),
            "hand_pose": copy.deepcopy(config["hand_pose"]),
            "control": copy.deepcopy(config["control"]),
            "acquisition_qpos_rad": acquisition_qpos,
            "initial_cube_world_pose": initial_pose,
            "provenance": {
                "catalog": str(catalog),
                "catalog_sha256": catalog_sha,
                "resolved_config": str(config_path),
                "resolved_config_sha256": config_sha,
                "trace": str(trace_path),
                "trace_sha256": trace_sha,
                "result": str(result_path),
                "result_sha256": result_sha,
            },
        }
        source_payload["source_sha256"] = canonical_sha256(
            {key: value for key, value in source_payload.items() if key != "config"}
        )
        result.append(source_payload)
    return tuple(result)


def policy_from_config(config: Mapping[str, Any]) -> HighThumbCampaignPolicy:
    """Read the canonical registered campaign block (with legacy aliases)."""

    block = config.get("high_thumb_size_campaign")
    if not isinstance(block, Mapping):
        raise ValueError("schema-v7 config has no high_thumb_size_campaign")
    return HighThumbCampaignPolicy(
        coarse_edges_m=tuple(block.get("coarse_edges_m", DEFAULT_COARSE_EDGES_M)),
        coarse_thumb_targets_rad=tuple(
            block.get("coarse_thumb_targets_rad", DEFAULT_COARSE_THUMB_TARGETS_RAD)
        ),
        edge_fine_step_m=float(
            block.get("edge_fine_step_m", block.get("fine_edge_step_m", DEFAULT_EDGE_FINE_STEP_M))
        ),
        thumb_fine_step_rad=float(
            block.get(
                "thumb_fine_step_rad",
                block.get("fine_thumb_step_rad", DEFAULT_THUMB_FINE_STEP_RAD),
            )
        ),
        fixed_mass_kg=float(block.get("fixed_mass_kg", DEFAULT_FIXED_MASS_KG)),
        friction=float(block.get("friction", DEFAULT_FRICTION)),
        cube_center_xy_m=tuple(
            block.get(
                "cube_center_xy_m",
                block.get("cube_pose_policy", {}).get("center_xy_m", DEFAULT_CENTER_XY_M),
            )
        ),
        seed=int(block.get("seed", config.get("search", {}).get("seed", DEFAULT_SEED))),
    )


def coarse_size_target_cells(
    policy: HighThumbCampaignPolicy,
) -> tuple[dict[str, Any], ...]:
    """Return the stable edge-major, thumb-minor coarse search cells."""

    cells = []
    for edge_index, edge in enumerate(policy.coarse_edges_m):
        for target_index, target in enumerate(policy.coarse_thumb_targets_rad):
            cells.append(
                {
                    "cell_index": len(cells),
                    "edge_index": edge_index,
                    "target_index": target_index,
                    "edge_m": edge,
                    "thumb_target_rad": target,
                    "cell_id": f"edge_{edge * 1000.0:.3f}_thumb_{target:.3f}",
                }
            )
    return tuple(cells)


def _source_cube_in_root(source: Mapping[str, Any]) -> np.ndarray:
    config = source["config"]
    rotation = rpy_degrees_to_rotation_matrix(config["hand_pose"]["rpy_deg"])
    cube_world = _finite_vector(
        source["initial_cube_world_pose"]["position_m"], 3, "source cube pose"
    )
    root = _finite_vector(config["hand_pose"]["translation_m"], 3, "source root")
    return rotation.T @ (cube_world - root)


def _target_mapping(values: Mapping[str, Any], label: str) -> dict[str, float]:
    if set(values) != set(ACTIVE_ACTUATORS):
        raise ValueError(f"{label} must contain exactly the active actuators")
    return {name: _finite(values[name], f"{label}.{name}") for name in ACTIVE_ACTUATORS}


def _close_group_starts(config: Mapping[str, Any]) -> dict[str, float]:
    profile = config["control"]["close_profile"]
    result: dict[str, float] = {}
    for group in CLOSE_GROUP_ORDER:
        values = {float(profile[name]["start_fraction"]) for name in CLOSE_GROUP_ACTUATORS[group]}
        if len(values) != 1:
            raise ValueError("source close profile is not grouped")
        result[group] = values.pop()
    return result


def _bounds(template: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    definition = resolve_experiment(dict(template))
    search = definition.search_bounds
    if search.pregrasp_targets_rad is None or search.manipulation_delta_rad is None:
        raise ValueError("schema-v7 search bounds are incomplete")
    return search, search.pregrasp_targets_rad, search.actuator_targets_rad


def _contains_local_pose(definition: Any, local: np.ndarray) -> bool:
    if not definition.search_bounds.contains_cube_position(local):
        return False
    constraints = definition.far_hand_pose_constraints
    if constraints is None:
        return True
    distance = float(np.linalg.norm(local))
    return (
        constraints.root_cube_distance_m[0] - 1e-12
        <= distance
        <= constraints.root_cube_distance_m[1] + 1e-12
    )


def _full_envelope_pose(
    template: Mapping[str, Any],
    rng: np.random.Generator,
    definition: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the full registered hand/object envelope by deterministic rejection."""

    from .far_hand_fingertip import (
        _feasible_tilt_for_roll_deg,
        root_pitch_for_finger_down_tilt_deg,
    )

    definition = resolve_experiment(dict(template)) if definition is None else definition
    search = definition.search_bounds
    constraints = definition.far_hand_pose_constraints
    for _ in range(128):
        roll = float(rng.uniform(*search.hand_roll_deg))
        desired_tilt = float(rng.uniform(search.palm_pitch_deg[0], search.palm_pitch_deg[1]))
        tilt = _feasible_tilt_for_roll_deg(
            desired_tilt, roll_deg=roll, constraints=constraints
        )
        pitch = root_pitch_for_finger_down_tilt_deg(roll, tilt)
        yaw = float(rng.uniform(*search.hand_yaw_deg))
        local = np.asarray(
            [
                rng.uniform(*search.cube_position_in_root_m[axis])
                for axis in ("x", "y", "z")
            ],
            dtype=np.float64,
        )
        if _contains_local_pose(definition, local):
            return np.asarray((roll, pitch, yaw), dtype=np.float64), local
    # This should be unreachable for the registered v7 ranges.  Keeping a
    # deterministic midpoint fallback makes malformed narrow test definitions
    # fail through validation instead of looping forever.
    local = np.asarray(
        [
            sum(search.cube_position_in_root_m[axis]) * 0.5
            for axis in ("x", "y", "z")
        ],
        dtype=np.float64,
    )
    if constraints is not None and not _contains_local_pose(definition, local):
        raise ValueError("registered v7 cube-in-root envelope has no feasible sample")
    roll = sum(search.hand_roll_deg) * 0.5
    tilt = sum(search.palm_pitch_deg) * 0.5
    return (
        np.asarray(
            (roll, root_pitch_for_finger_down_tilt_deg(roll, tilt), sum(search.hand_yaw_deg) * 0.5),
            dtype=np.float64,
        ),
        local,
    )


def _finger_down_tilt_from_rpy(rpy_deg: Sequence[float]) -> float:
    roll, pitch, _ = np.radians(_finite_vector(rpy_deg, 3, "hand RPY"))
    sine = float(np.clip(-math.cos(roll) * math.cos(pitch), -1.0, 1.0))
    return float(math.degrees(math.asin(sine)))


def _local_rpy_sample(
    template: Mapping[str, Any],
    base_rpy: np.ndarray,
    unit: np.ndarray,
    definition: Any,
    radius_deg: Sequence[float] = _DEFAULT_HAND_RPY_RADIUS_DEG,
) -> np.ndarray:
    from .far_hand_fingertip import (
        _feasible_tilt_for_roll_deg,
        root_pitch_for_finger_down_tilt_deg,
    )

    search = definition.search_bounds
    constraints = definition.far_hand_pose_constraints
    radius = _finite_vector(radius_deg, 3, "RPY sampling radius")
    roll = float(np.clip(base_rpy[0] + unit[0] * radius[0], *search.hand_roll_deg))
    desired_tilt = _finger_down_tilt_from_rpy(base_rpy) + float(unit[1] * radius[1])
    if constraints is None:
        tilt = float(np.clip(desired_tilt, *search.palm_pitch_deg))
    else:
        tilt = _feasible_tilt_for_roll_deg(
            desired_tilt, roll_deg=roll, constraints=constraints
        )
    yaw = float(np.clip(base_rpy[2] + unit[2] * radius[2], *search.hand_yaw_deg))
    return np.asarray(
        (roll, root_pitch_for_finger_down_tilt_deg(roll, tilt), yaw),
        dtype=np.float64,
    )


def _clipped_local_pose(
    template: Mapping[str, Any],
    proposed: np.ndarray,
    fallback: np.ndarray,
    definition: Any | None = None,
) -> np.ndarray:
    definition = resolve_experiment(dict(template)) if definition is None else definition
    search = definition.search_bounds
    clipped = np.asarray(
        [
            np.clip(proposed[index], *search.cube_position_in_root_m[axis])
            for index, axis in enumerate(("x", "y", "z"))
        ],
        dtype=np.float64,
    )
    if _contains_local_pose(definition, clipped):
        return clipped
    if _contains_local_pose(definition, fallback):
        return fallback.copy()
    raise ValueError("source cube-in-root relation is outside the registered v7 envelope")


def assert_pose_preserving_candidate_invariants(
    config: Mapping[str, Any],
    *,
    expected_pose: Mapping[str, Any],
    expected_mass_kg: float,
    expected_friction: float,
    acquisition_stage: bool,
) -> None:
    """Reject any candidate that moved or constrained the free cube."""

    cube = config["cube"]
    actual_pose = cube_world_pose_for_size(config, cube["edge_m"], cube["center_xy_m"])
    if actual_pose != expected_pose:
        raise ValueError("candidate changed its fixed per-size initial cube pose")
    if not math.isclose(float(cube["mass_kg"]), expected_mass_kg, abs_tol=1e-15):
        raise ValueError("candidate changed the fixed campaign mass")
    if not math.isclose(float(cube["friction"]), expected_friction, abs_tol=1e-15):
        raise ValueError("candidate changed the fixed campaign friction")
    if acquisition_stage and config["control"]["manipulation_delta_rad"] != {
        name: 0.0 for name in ACTIVE_ACTUATORS
    }:
        raise ValueError("acquisition candidate must have zero manipulation delta")
    metadata = config.get("candidate_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("candidate metadata is missing")
    if metadata.get("cube_pose_sampled") is not False:
        raise ValueError("candidate must explicitly declare cube_pose_sampled=false")
    if metadata.get("free_cube_pose_reset_during_run") is not False:
        raise ValueError("candidate must not reset the free cube pose")


def materialize_high_thumb_candidate(
    source: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    edge_m: float,
    thumb_target_rad: float,
    hand_rpy_deg: Sequence[float] | None = None,
    cube_in_root_m: Sequence[float] | None = None,
    pregrasp_targets_rad: Mapping[str, Any] | None = None,
    grasp_targets_rad: Mapping[str, Any] | None = None,
    close_group_start_fractions: Mapping[str, Any] | None = None,
    manipulation_delta_rad: Mapping[str, Any] | None = None,
    candidate_id: int,
    stage: str = "acquisition",
    candidate_metadata: Mapping[str, Any] | None = None,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> dict[str, Any]:
    """Build one schema-v7 candidate while holding its cube pose immutable."""

    if int(template.get("schema_version", 0)) != 7:
        raise ValueError("high-thumb variable-size template must use schema version 7")
    if template.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("high-thumb variable-size template has the wrong experiment_id")
    policy = policy_from_config(template)
    edge = _finite(edge_m, "edge_m")
    thumb = _finite(thumb_target_rad, "thumb_target_rad")
    if not 1.25 - 1e-12 <= thumb <= 1.45 + 1e-12:
        raise ValueError("thumb_target_rad must lie within [1.25, 1.45]")
    config = copy.deepcopy(dict(template))
    config["scene"] = copy.deepcopy(source["scene"])
    config["contact_topology"] = copy.deepcopy(source["contact_topology"])
    config["cube"] = copy.deepcopy(source["cube"])
    config["cube"].update(
        {
            "edge_m": edge,
            "mass_kg": policy.fixed_mass_kg,
            "friction": policy.friction,
            "center_xy_m": list(policy.cube_center_xy_m),
            "z_offset_m": 0.0,
        }
    )
    expected_pose = cube_world_pose_for_size(config, edge, policy.cube_center_xy_m)
    rpy = _finite_vector(
        source["hand_pose"]["rpy_deg"] if hand_rpy_deg is None else hand_rpy_deg,
        3,
        "hand_rpy_deg",
    )
    local = _finite_vector(
        _source_cube_in_root(source) if cube_in_root_m is None else cube_in_root_m,
        3,
        "cube_in_root_m",
    )
    cube_world = np.asarray(expected_pose["position_m"], dtype=np.float64)
    config["hand_pose"] = {
        "rpy_deg": rpy.tolist(),
        "translation_m": (
            cube_world - rpy_degrees_to_rotation_matrix(rpy) @ local
        ).tolist(),
    }

    pregrasp = _target_mapping(
        source["control"]["pregrasp_targets_rad"]
        if pregrasp_targets_rad is None
        else pregrasp_targets_rad,
        "pregrasp_targets_rad",
    )
    targets = _target_mapping(
        source["control"]["grasp_targets_rad"]
        if grasp_targets_rad is None
        else grasp_targets_rad,
        "grasp_targets_rad",
    )
    targets[THUMB_BEND_ACTUATOR] = thumb
    if close_group_start_fractions is None:
        starts = _close_group_starts(source["config"])
    else:
        if set(close_group_start_fractions) != set(CLOSE_GROUP_ORDER):
            raise ValueError("close group starts must contain thumb, index and mid")
        starts = {
            group: _finite(close_group_start_fractions[group], f"close start {group}")
            for group in CLOSE_GROUP_ORDER
        }
    close_profile = close_profile_from_group_starts(template, starts)
    if manipulation_delta_rad is None:
        manipulation = {name: 0.0 for name in ACTIVE_ACTUATORS}
    else:
        manipulation = _target_mapping(manipulation_delta_rad, "manipulation_delta_rad")
    config["control"] = {
        "pregrasp_targets_rad": pregrasp,
        "grasp_targets_rad": targets,
        "manipulation_delta_rad": manipulation,
        "close_profile": close_profile,
    }
    config.pop("run_context", None)
    metadata = copy.deepcopy(dict(candidate_metadata or {}))
    metadata.update(
        {
            "campaign_kind": CAMPAIGN_KIND,
            "candidate_id": int(candidate_id),
            "stage": stage,
            "seed_family": str(source["source_family_id"]),
            "source_family_id": str(source["source_family_id"]),
            "source_trajectory_id": str(source["source_trajectory_id"]),
            "source_sha256": str(source["source_sha256"]),
            "edge_m": edge,
            "thumb_target_rad": thumb,
            "fixed_cube_initial_pose": copy.deepcopy(expected_pose),
            "candidate_cube_in_root_m": local.tolist(),
            "cube_pose_sampled": False,
            "free_cube_pose_reset_during_run": False,
            "fixed_mass_kg": policy.fixed_mass_kg,
            "fixed_friction": policy.friction,
        }
    )
    config["candidate_metadata"] = metadata
    assert_pose_preserving_candidate_invariants(
        config,
        expected_pose=expected_pose,
        expected_mass_kg=policy.fixed_mass_kg,
        expected_friction=policy.friction,
        acquisition_stage=stage != "lift",
    )
    if validator is not None:
        validator(config)
    return config


def _clip_mapping(
    base: Mapping[str, float],
    offsets: Mapping[str, float],
    bounds: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    return {
        name: float(np.clip(float(base[name]) + float(offsets[name]), *bounds[name]))
        for name in ACTIVE_ACTUATORS
    }


def generate_high_thumb_candidates(
    sources: Sequence[Mapping[str, Any]],
    template: Mapping[str, Any],
    *,
    edge_m: float,
    thumb_target_rad: float,
    count: int,
    seed: int,
    cell_index: int,
    start_index: int = 0,
    stage: str = "coarse_static",
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    """Generate a prefix-stable, source-balanced search cell."""

    candidate_count = _positive_int(count, "count")
    if (
        not isinstance(start_index, int)
        or isinstance(start_index, bool)
        or start_index < 0
        or start_index + candidate_count > _ID_STRIDE
    ):
        raise ValueError("start_index + count must stay within the candidate-ID stride")
    if len(sources) != EXPECTED_SOURCE_COUNT:
        raise ValueError("candidate generation requires all six v6 sources")
    if not isinstance(cell_index, int) or isinstance(cell_index, bool) or cell_index < 0:
        raise ValueError("cell_index must be a non-negative integer")
    search, pregrasp_bounds, target_bounds = _bounds(template)
    definition = resolve_experiment(dict(template))
    stage_base = {
        "coarse_static": _STATIC_STAGE_BASE,
        "local_refine": _LOCAL_STAGE_BASE,
        "fine": _FINE_STAGE_BASE,
        "exact": _EXACT_STAGE_BASE,
    }.get(stage)
    if stage_base is None:
        raise ValueError(f"unsupported acquisition stage {stage!r}")
    records: list[dict[str, Any]] = []
    for local_index in range(start_index, start_index + candidate_count):
        source = sources[local_index % len(sources)]
        family_occurrence = local_index // len(sources)
        base_rpy = _finite_vector(source["hand_pose"]["rpy_deg"], 3, "source RPY")
        base_local = _source_cube_in_root(source)
        base_pregrasp = _target_mapping(
            source["control"]["pregrasp_targets_rad"], "source pregrasp"
        )
        base_targets = _target_mapping(
            source["control"]["grasp_targets_rad"], "source targets"
        )
        base_starts = _close_group_starts(source["config"])
        if family_occurrence == 0:
            unit = np.zeros(24, dtype=np.float64)
            sample_seed = None
            sample_mode = "source_anchor"
            rng = None
        else:
            sequence = np.random.SeedSequence(
                [
                    int(seed),
                    int(cell_index),
                    int(local_index),
                    int(round(float(edge_m) * 1e6)),
                    int(round(float(thumb_target_rad) * 1e6)),
                    7_100_001,
                ]
            )
            rng = np.random.default_rng(sequence)
            unit = 2.0 * rng.random(24) - 1.0
            sample_seed = int(seed)
            mode_draw = float(rng.random())
            low_pitch_bias = bool(
                str(source["source_family_id"]) == "117061"
                and 0.062 - 1e-12 <= float(edge_m) <= 0.064 + 1e-12
                and float(thumb_target_rad) <= 1.30 + 1e-12
            )
            diagnostic_anchor = bool(
                family_occurrence == 1
                and str(source["source_family_id"]) == "117061"
                and (
                    (
                        math.isclose(float(edge_m), 0.062, abs_tol=1e-12)
                        and math.isclose(
                            float(thumb_target_rad), 1.30, abs_tol=1e-12
                        )
                    )
                    or (
                        math.isclose(float(edge_m), 0.060, abs_tol=1e-12)
                        and math.isclose(
                            float(thumb_target_rad), 1.25, abs_tol=1e-12
                        )
                    )
                )
            )
            if diagnostic_anchor:
                sample_mode = "validated_dynamic_near_miss_anchor"
            elif mode_draw < 0.40:
                sample_mode = "full_registered_envelope"
            elif low_pitch_bias and mode_draw < 0.70:
                sample_mode = "family_117061_low_pitch_bias"
            else:
                sample_mode = "bend_scaled_source_local"
        if sample_mode == "full_registered_envelope":
            assert rng is not None
            rpy, local = _full_envelope_pose(template, rng, definition)
        elif sample_mode == "validated_dynamic_near_miss_anchor":
            rpy = base_rpy.copy()
            rpy[1] -= 1.9 if math.isclose(float(edge_m), 0.062) else 1.5
            local = base_local.copy()
        else:
            bend_fraction = float(
                np.clip((float(thumb_target_rad) - 1.25) / (1.45 - 1.25), 0.0, 1.0)
            )
            local_radius = np.asarray(
                (
                    0.003 + 0.005 * bend_fraction,
                    0.003 + 0.003 * bend_fraction,
                    0.003 + 0.003 * bend_fraction,
                ),
                dtype=np.float64,
            )
            rpy = _local_rpy_sample(template, base_rpy, unit[:3], definition)
            if sample_mode == "family_117061_low_pitch_bias":
                assert rng is not None
                rpy[1] = float(rng.uniform(120.1, 121.0))
            local = _clipped_local_pose(
                template,
                base_local + unit[3:6] * local_radius,
                base_local,
                definition,
            )
        if sample_mode == "full_registered_envelope":
            assert rng is not None
            pregrasp = {
                name: float(rng.uniform(*pregrasp_bounds[name]))
                for name in ACTIVE_ACTUATORS
            }
        elif sample_mode == "validated_dynamic_near_miss_anchor":
            pregrasp = dict(base_pregrasp)
        else:
            pre_offsets = {
                name: float(unit[6 + index] * _DEFAULT_PREGRASP_RADIUS_RAD)
                for index, name in enumerate(ACTIVE_ACTUATORS)
            }
            pregrasp = _clip_mapping(base_pregrasp, pre_offsets, pregrasp_bounds)
        targets = dict(base_targets)
        non_thumb = [name for name in ACTIVE_ACTUATORS if name != THUMB_BEND_ACTUATOR]
        for index, name in enumerate(non_thumb):
            if sample_mode == "full_registered_envelope":
                assert rng is not None
                targets[name] = float(rng.uniform(*target_bounds[name]))
            elif sample_mode == "validated_dynamic_near_miss_anchor":
                targets[name] = float(base_targets[name])
            else:
                local_radius = _DEFAULT_TERMINAL_RADIUS_RAD
                if name in {
                    "left_hand_thumb_rota_joint1_actuator",
                    "left_hand_thumb_rota_joint2_actuator",
                }:
                    bend_fraction = float(
                        np.clip(
                            (float(thumb_target_rad) - 1.25) / (1.45 - 1.25),
                            0.0,
                            1.0,
                        )
                    )
                    local_radius += 0.12 * bend_fraction
                targets[name] = float(
                    np.clip(
                        targets[name] + unit[14 + index] * local_radius,
                        *target_bounds[name],
                    )
                )
        targets[THUMB_BEND_ACTUATOR] = float(thumb_target_rad)
        starts: dict[str, float] = {}
        for index, group in enumerate(CLOSE_GROUP_ORDER):
            end = min(
                float(template["control"]["close_profile"][name]["end_fraction"])
                for name in CLOSE_GROUP_ACTUATORS[group]
            )
            if sample_mode == "full_registered_envelope":
                assert rng is not None
                starts[group] = float(rng.uniform(0.0, end - 0.02))
            elif sample_mode == "validated_dynamic_near_miss_anchor":
                starts[group] = float(base_starts[group])
                if group == "index" and math.isclose(float(edge_m), 0.062):
                    starts[group] = 0.10
            else:
                starts[group] = float(
                    np.clip(
                        base_starts[group] + unit[21 + index] * _DEFAULT_CLOSE_START_RADIUS[index],
                        0.0,
                        end - 0.02,
                    )
                )
        candidate_id = stage_base + cell_index * _ID_STRIDE + local_index
        config = materialize_high_thumb_candidate(
            source,
            template,
            edge_m=edge_m,
            thumb_target_rad=thumb_target_rad,
            hand_rpy_deg=rpy,
            cube_in_root_m=local,
            pregrasp_targets_rad=pregrasp,
            grasp_targets_rad=targets,
            close_group_start_fractions=starts,
            candidate_id=candidate_id,
            stage=stage,
            candidate_metadata={
                "cell_index": cell_index,
                "local_index": local_index,
                "seed": sample_seed,
                "sampled_fields": [
                    "hand_pose",
                    "control.pregrasp_targets_rad",
                    "control.grasp_targets_rad_except_thumb_bend",
                    "control.close_profile.group_start_fraction",
                ],
                "pose_sample_mode": sample_mode,
            },
            validator=validator,
        )
        records.append(
            {
                "candidate_id": candidate_id,
                "candidate_sha256": canonical_sha256(config),
                "stage": stage,
                "cell_index": cell_index,
                "local_index": local_index,
                "source_family_id": str(source["source_family_id"]),
                "source_trajectory_id": str(source["source_trajectory_id"]),
                "edge_m": float(edge_m),
                "thumb_target_rad": float(thumb_target_rad),
                "acquisition_qpos_reference_rad": copy.deepcopy(
                    source["acquisition_qpos_rad"]
                ),
                "config": config,
            }
        )
    return tuple(records)


_FACE_BY_LABEL: dict[str, Any] | None = None


def _face_by_label() -> Mapping[str, Any]:
    global _FACE_BY_LABEL
    if _FACE_BY_LABEL is None:
        from ..contacts import Face

        _FACE_BY_LABEL = {
            "+X": Face.X_POS,
            "-X": Face.X_NEG,
            "+Y": Face.Y_POS,
            "-Y": Face.Y_NEG,
            "+Z": Face.Z_POS,
            "-Z": Face.Z_NEG,
        }
    return _FACE_BY_LABEL


def _static_candidate_score(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    gaps = tuple(float(value) for value in metrics["target_signed_gap_m"])
    gap_violation = sum(max(0.0, -0.0005 - value) + max(0.0, value - 0.003) for value in gaps)
    height = float(metrics["contact_height_spread_m"])
    nondistal = float(metrics["minimum_active_nondistal_gap_m"])
    witness_missing = int(metrics["missing_distal_witness_count"])
    score = (
        10_000.0 * witness_missing
        + 1_000.0 * gap_violation
        + 1_000.0 * max(0.0, height - 0.005)
        + 1_000.0 * max(0.0, -nondistal)
        + 10.0 * sum(abs(value - 0.00125) for value in gaps if math.isfinite(value))
        + (height if math.isfinite(height) else 10.0)
    )
    return (
        witness_missing > 0,
        gap_violation > 0.0,
        height > 0.005,
        nondistal < 0.0,
        score,
    )


def screen_static_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    retain: int,
    _runtime_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use real collision geoms to rank a materialised search cell.

    Each source family is compiled only once.  The fixed hand root and active
    joint qpos are then changed before ``mj_forward``; ``mj_step`` is never
    called, so this stage cannot be confused with dynamic grasp validation.
    """

    retain_count = _positive_int(retain, "retain")
    if not candidates:
        raise ValueError("static screen candidates must not be empty")
    import mujoco

    from ..contact_geometry import (
        active_nondistal_collision_geom_ids,
        distal_collision_geom_ids,
        nearest_distal_target_witness,
    )
    from ..scene import build_model

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        groups[str(candidate["source_family_id"])].append(candidate)
    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    runtime_cache = {} if _runtime_cache is None else _runtime_cache
    for family in sorted(groups):
        family_candidates = groups[family]
        if family not in runtime_cache:
            model, info = build_model(copy.deepcopy(dict(family_candidates[0]["config"])))
            data = mujoco.MjData(model)
            active_ids = np.asarray(
                [model.actuator(name).id for name in ACTIVE_ACTUATORS], dtype=np.int64
            )
            qpos_addresses = info.actuator_qpos_adrs[active_ids]
            distal = distal_collision_geom_ids(model, info.distal_weld_ids)
            nondistal = active_nondistal_collision_geom_ids(
                model, info.hand_body_parts, info.distal_weld_ids
            )
            gravity = np.asarray(model.opt.gravity, dtype=np.float64)
            gravity_norm = float(np.linalg.norm(gravity))
            if gravity_norm <= np.finfo(np.float64).eps:
                raise ValueError("static screen requires non-zero gravity")
            runtime_cache[family] = (
                model,
                info,
                data,
                qpos_addresses,
                distal,
                nondistal,
                -gravity / gravity_norm,
                family_candidates[0]["config"]["contact_topology"]["target_faces"],
                None,
            )
        (
            model,
            info,
            data,
            qpos_addresses,
            distal,
            nondistal,
            up,
            target_faces,
            cube_qpos_reference,
        ) = runtime_cache[family]
        for candidate in family_candidates:
            config = candidate["config"]
            root_rpy = _finite_vector(config["hand_pose"]["rpy_deg"], 3, "hand RPY")
            model.body_pos[info.root_body_id] = _finite_vector(
                config["hand_pose"]["translation_m"], 3, "hand translation"
            )
            model.body_quat[info.root_body_id] = rpy_degrees_to_quaternion(root_rpy)
            mujoco.mj_resetData(model, data)
            pregrasp = np.asarray(
                [
                    float(config["control"]["pregrasp_targets_rad"][name])
                    for name in ACTIVE_ACTUATORS
                ],
                dtype=np.float64,
            )
            targets = np.asarray(
                [
                    float(config["control"]["grasp_targets_rad"][name])
                    for name in ACTIVE_ACTUATORS
                ],
                dtype=np.float64,
            )
            profile = config["control"]["close_profile"]

            def observe(qpos: np.ndarray) -> dict[str, Any]:
                data.qpos[qpos_addresses] = qpos
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                nonlocal cube_qpos_reference
                cube_qpos = np.asarray(
                    data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 7],
                    dtype=np.float64,
                ).copy()
                if cube_qpos_reference is None:
                    cube_qpos_reference = cube_qpos
                    cached = list(runtime_cache[family])
                    cached[-1] = cube_qpos_reference
                    runtime_cache[family] = tuple(cached)
                elif not np.array_equal(cube_qpos, cube_qpos_reference):
                    raise RuntimeError("static closure scan changed the fixed cube qpos")

                gaps: list[float] = []
                heights: list[float] = []
                witnesses: dict[str, Any] = {}
                individually_clean: dict[str, bool] = {}
                missing = 0
                for finger in ACTIVE_FINGERS:
                    witness = nearest_distal_target_witness(
                        model,
                        data,
                        cube_geom_id=info.cube_geom_id,
                        distal_geom_ids=distal[finger],
                        target_face=_face_by_label()[target_faces[finger]],
                        distance_max_m=0.020,
                    )
                    if witness is None:
                        missing += 1
                        gaps.append(0.1)
                        heights.append(0.1)
                        witnesses[finger] = None
                        individually_clean[finger] = False
                    else:
                        gap = float(witness.signed_distance_m)
                        alignment = float(witness.classification.normal_alignment)
                        clearance = float(witness.classification.edge_clearance_m)
                        gaps.append(gap)
                        heights.append(float(up @ witness.cube_point_world_m))
                        witnesses[finger] = {
                            "distal_geom_id": int(witness.distal_geom_id),
                            "signed_distance_m": gap,
                            "penetration_m": float(witness.penetration_m),
                            "normal_alignment": alignment,
                            "edge_clearance_m": clearance,
                            "cube_point_world_m": witness.cube_point_world_m.tolist(),
                            "distal_point_world_m": witness.distal_point_world_m.tolist(),
                        }
                        individually_clean[finger] = bool(
                            -0.0005 <= gap <= 0.003
                            and alignment >= 0.95
                            and clearance >= 0.0005
                        )
                nondistal_min = 0.1
                segment = np.empty(6, dtype=np.float64)
                for finger in ACTIVE_FINGERS:
                    for geom_id in nondistal[finger]:
                        nondistal_min = min(
                            nondistal_min,
                            float(
                                mujoco.mj_geomDistance(
                                    model,
                                    data,
                                    info.cube_geom_id,
                                    int(geom_id),
                                    1.0,
                                    segment,
                                )
                            ),
                        )
                hand_cube_collision = False
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    if float(contact.dist) > 0.0:
                        continue
                    if int(contact.geom1) == info.cube_geom_id:
                        other = int(contact.geom2)
                    elif int(contact.geom2) == info.cube_geom_id:
                        other = int(contact.geom1)
                    else:
                        continue
                    if int(model.geom_bodyid[other]) in info.hand_body_parts:
                        hand_cube_collision = True
                        break
                height_spread = (
                    float(max(heights) - min(heights)) if not missing else 0.1
                )
                metrics = {
                    "missing_distal_witness_count": missing,
                    "target_signed_gap_m": gaps,
                    "contact_height_m": heights,
                    "contact_height_spread_m": height_spread,
                    "minimum_active_nondistal_gap_m": nondistal_min,
                    "distal_witness": witnesses,
                    "finger_target_window": individually_clean,
                    "hand_cube_collision": hand_cube_collision,
                }
                metrics["three_finger_clean"] = bool(
                    all(individually_clean.values())
                    and nondistal_min >= 0.0
                    and height_spread <= 0.005
                )
                return metrics

            alpha_values = np.linspace(0.0, 1.0, 17)
            observations: list[tuple[float, dict[str, Any]]] = []
            first_window_alpha: dict[str, float | None] = {
                finger: None for finger in ACTIVE_FINGERS
            }
            for alpha in alpha_values:
                command = np.empty(len(ACTIVE_ACTUATORS), dtype=np.float64)
                for actuator_index, name in enumerate(ACTIVE_ACTUATORS):
                    start = float(profile[name]["start_fraction"])
                    end = float(profile[name]["end_fraction"])
                    progress = float(np.clip((alpha - start) / (end - start), 0.0, 1.0))
                    smooth = progress * progress * (3.0 - 2.0 * progress)
                    command[actuator_index] = pregrasp[actuator_index] + smooth * (
                        targets[actuator_index] - pregrasp[actuator_index]
                    )
                metrics_at_alpha = observe(command)
                observations.append((float(alpha), metrics_at_alpha))
                for finger in ACTIVE_FINGERS:
                    if (
                        first_window_alpha[finger] is None
                        and metrics_at_alpha["finger_target_window"][finger]
                    ):
                        first_window_alpha[finger] = float(alpha)
            pregrasp_metrics = observations[0][1]
            clean_observations = [
                item for item in observations if item[1]["three_finger_clean"]
            ]
            candidate_observations = clean_observations or observations
            best_alpha, metrics = min(
                candidate_observations,
                key=lambda item: _static_candidate_score(item[1]) + (item[0],),
            )
            reference = _target_mapping(
                candidate["acquisition_qpos_reference_rad"],
                "acquisition_qpos_reference_rad",
            )
            reference_metrics = observe(
                np.asarray([reference[name] for name in ACTIVE_ACTUATORS], dtype=np.float64)
            )
            onsets = [
                value for value in first_window_alpha.values() if value is not None
            ]
            onset_span = (
                float(max(onsets) - min(onsets))
                if len(onsets) == len(ACTIVE_FINGERS)
                else 1.0
            )
            metrics = {
                **metrics,
                "closure_scan_alpha_values": alpha_values.tolist(),
                "selected_closure_alpha": best_alpha,
                "clean_closure_alpha_count": len(clean_observations),
                "first_target_window_alpha": first_window_alpha,
                "contact_onset_alpha_span": onset_span,
                "pregrasp_hand_cube_contact": bool(pregrasp_metrics["hand_cube_collision"]),
                "pregrasp_minimum_active_nondistal_gap_m": float(
                    pregrasp_metrics["minimum_active_nondistal_gap_m"]
                ),
                "acquisition_qpos_reference": reference_metrics,
            }
            score_key = (
                bool(metrics["pregrasp_hand_cube_contact"]),
                not bool(clean_observations),
                onset_span,
            ) + _static_candidate_score(metrics)
            record = {
                **copy.deepcopy(dict(candidate)),
                "static_pass": bool(
                    not metrics["pregrasp_hand_cube_contact"]
                    and clean_observations
                    and onset_span <= 0.25
                ),
                "static_score": float(_static_candidate_score(metrics)[-1] + onset_span),
                "static_metrics": metrics,
            }
            ranked.append((score_key + (int(candidate["candidate_id"]),), record))
    ranked.sort(key=lambda item: item[0])
    ordered = [record for _, record in ranked]
    retained = _select_static_family_anchors(ordered, retain_count)
    return {
        "evaluated_count": len(candidates),
        "static_pass_count": sum(bool(record["static_pass"]) for _, record in ranked),
        "retained_count": len(retained),
        "retained": retained,
    }


def _static_record_rank(value: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = value["static_metrics"]
    metadata = value.get("config", {}).get("candidate_metadata", {})
    dynamic_near_miss_anchor = bool(
        isinstance(metadata, Mapping)
        and metadata.get("pose_sample_mode")
        == "validated_dynamic_near_miss_anchor"
    )
    return (
        not dynamic_near_miss_anchor,
        not bool(value.get("static_pass", False)),
        bool(metrics.get("pregrasp_hand_cube_contact", True)),
        int(metrics.get("clean_closure_alpha_count", 0)) <= 0,
        float(metrics.get("contact_onset_alpha_span", 1.0)),
        float(value.get("static_score", 1e9)),
        int(value["candidate_id"]),
    )


def _select_static_family_anchors(
    values: Sequence[Mapping[str, Any]], retain: int
) -> tuple[dict[str, Any], ...]:
    """Keep each source family represented before filling by global rank."""

    limit = min(_positive_int(retain, "retain"), len(values))
    ordered = sorted(
        (copy.deepcopy(dict(value)) for value in values), key=_static_record_rank
    )
    if limit == 0:
        return ()
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    families = sorted({str(value["source_family_id"]) for value in ordered})
    if limit >= len(families):
        for family in families:
            best = next(value for value in ordered if str(value["source_family_id"]) == family)
            selected.append(best)
            selected_ids.add(int(best["candidate_id"]))
    for value in ordered:
        if len(selected) >= limit:
            break
        if int(value["candidate_id"]) not in selected_ids:
            selected.append(value)
            selected_ids.add(int(value["candidate_id"]))
    selected.sort(key=_static_record_rank)
    return tuple(selected)


def execute_static_cell_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Generate, screen, and atomically commit one independently resumable cell."""

    output = Path(str(job["output_directory"]))
    if output.exists():
        raise FileExistsError(f"static cell output already exists: {output}")
    sample_count = int(job["sample_count"])
    retain_count = int(job["retain_count"])
    # Keep peak memory bounded even at the declared 20k samples/cell.  The
    # compiled six-family models are cached across chunks, while only each
    # chunk's top-k can possibly belong to the cell-wide top-k.
    chunk_size = min(512, sample_count)
    cache: dict[str, Any] = {}
    retained_pool: list[dict[str, Any]] = []
    static_pass_count = 0
    for start in range(0, sample_count, chunk_size):
        candidates = generate_high_thumb_candidates(
            job["sources"],
            job["template"],
            edge_m=float(job["edge_m"]),
            thumb_target_rad=float(job["thumb_target_rad"]),
            count=min(chunk_size, sample_count - start),
            start_index=start,
            seed=int(job["seed"]),
            cell_index=int(job["cell_index"]),
            validator=None,
        )
        chunk = screen_static_candidates(
            candidates, retain=retain_count, _runtime_cache=cache
        )
        static_pass_count += int(chunk["static_pass_count"])
        retained_pool.extend(copy.deepcopy(dict(value)) for value in chunk["retained"])
    retained = _select_static_family_anchors(retained_pool, retain_count)
    screened = {
        "evaluated_count": sample_count,
        "static_pass_count": static_pass_count,
        "retained_count": len(retained),
        "retained": retained,
    }
    # Only the handful of candidates that can enter dynamics pay the complete
    # schema validation cost.  Static materialisation already enforces the
    # fixed-object invariant for every sample.
    for retained in screened["retained"]:
        validate_config(retained["config"])
    payload = {
        "static_cell_schema_version": 1,
        "complete": True,
        "cell_input_sha256": str(job["cell_input_sha256"]),
        "cell_index": int(job["cell_index"]),
        "cell_id": str(job["cell_id"]),
        "edge_m": float(job["edge_m"]),
        "thumb_target_rad": float(job["thumb_target_rad"]),
        "sample_count": int(job["sample_count"]),
        "evaluated_count": int(screened["evaluated_count"]),
        "static_pass_count": int(screened["static_pass_count"]),
        "retained_count": int(screened["retained_count"]),
        "retained": [copy.deepcopy(dict(value)) for value in screened["retained"]],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as staging:
        staging_path = Path(staging)
        write_json(staging_path / "result.json", payload)
        staging_path.rename(output)
    return payload


def run_static_cell_jobs(
    jobs: Sequence[Mapping[str, Any]], workers: int
) -> tuple[dict[str, Any], ...]:
    count = _positive_int(workers, "workers")
    if not jobs:
        return ()
    if count == 1:
        results = [execute_static_cell_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=count, mp_context=context) as executor:
            results = list(executor.map(execute_static_cell_job, jobs, chunksize=1))
    results.sort(key=lambda value: int(value["cell_index"]))
    return tuple(results)


def _load_reusable_static_cell(job: Mapping[str, Any]) -> dict[str, Any] | None:
    directory = Path(str(job["output_directory"]))
    if not directory.exists():
        return None
    path = directory / "result.json"
    if not path.is_file():
        raise RuntimeError(f"incomplete static cell directory: {directory}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise RuntimeError(f"static cell is not complete: {path}")
    if payload.get("cell_input_sha256") != job["cell_input_sha256"]:
        raise RuntimeError(f"static cell inputs changed: {path}")
    if int(payload.get("cell_index", -1)) != int(job["cell_index"]):
        raise RuntimeError(f"static cell index changed: {path}")
    return payload


def _nested(mapping: Mapping[str, Any], path: Sequence[str], default: Any) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _rank_number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _result_flags(result: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    if all(name in result for name in ("acquisition_success", "pose_preservation_success", "lift_success")):
        return (
            bool(result["acquisition_success"]),
            bool(result["pose_preservation_success"]),
            bool(result["lift_success"]),
        )
    summary = result.get("summary", {})
    ranking = {"candidate_id": int(result.get("candidate_id", 0)), "summary": summary}
    stage_status = summary.get("stage_status", {}) if isinstance(summary, Mapping) else {}
    return (
        acquisition_succeeded(ranking),
        pose_preservation_succeeded(ranking),
        bool(stage_status.get("full_success", summary.get("passed", False))),
    )


def candidate_rank_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    """Extract stable scalar rank evidence from either a run or fake test result."""

    summary = result.get("summary", {})
    metrics = summary.get("metrics", {}) if isinstance(summary, Mapping) else {}
    pose = metrics.get("pose_preservation", {}) if isinstance(metrics, Mapping) else {}
    translation = _rank_number(pose.get("max_translation_m"), math.inf)
    translation_limit = _rank_number(pose.get("translation_limit_m"), 0.0005)
    orientation = _rank_number(pose.get("max_orientation_drift_deg"), math.inf)
    orientation_limit = _rank_number(pose.get("orientation_limit_deg"), 1.0)
    translation_margin = (
        (translation_limit - translation) / translation_limit
        if translation_limit > 0.0
        else -math.inf
    )
    orientation_margin = (
        (orientation_limit - orientation) / orientation_limit
        if orientation_limit > 0.0
        else -math.inf
    )
    alignment = _nested(metrics, ("contact_alignment", "verify"), {})
    pad = _nested(metrics, ("fingertip_contact", "verify", "force_weighted_pad_fraction"), {})
    if not isinstance(pad, Mapping):
        pad = {}
    duties = metrics.get("verify_target_face_effective_duty", {})
    if not isinstance(duties, Mapping):
        duties = {}
    declared_rank = result.get("rank_metrics", {})
    if not isinstance(declared_rank, Mapping):
        declared_rank = {}
    trace_metrics = result.get("thumb_bend_trace_metrics", {})
    if not isinstance(trace_metrics, Mapping):
        trace_metrics = {}
    return {
        "pose_min_normalized_margin": min(translation_margin, orientation_margin),
        "translation_margin": translation_margin,
        "orientation_margin": orientation_margin,
        "grasp_stability_margin": _rank_number(metrics.get("grasp_stability_margin"), -math.inf),
        "verify_gate_steps": _rank_number(metrics.get("verify_max_consecutive_all_gate_steps"), 0.0),
        "all_state_gate_steps": _rank_number(
            declared_rank.get(
                "all_state_gate_steps",
                trace_metrics.get("max_consecutive_all_gate_steps_close_verify"),
            ),
            0.0,
        ),
        "alignment_duty": _rank_number(
            alignment.get("aligned_duty") if isinstance(alignment, Mapping) else None,
            0.0,
        ),
        "alignment_height_p95_m": _rank_number(
            alignment.get("height_spread_p95_m") if isinstance(alignment, Mapping) else None,
            math.inf,
        ),
        "minimum_pad_force_fraction": min(
            (_rank_number(pad.get(finger), 0.0) for finger in ACTIVE_FINGERS),
            default=0.0,
        ),
        "minimum_target_face_duty": min(
            (_rank_number(duties.get(finger), 0.0) for finger in ACTIVE_FINGERS),
            default=0.0,
        ),
        "peak_total_distal_contact_force_n": _rank_number(
            metrics.get("peak_total_distal_contact_force_n"), math.inf
        ),
        "actuator_saturation_fraction": _rank_number(
            metrics.get("actuator_saturation_fraction"), math.inf
        ),
        "median_lift_m": _rank_number(metrics.get("median_lift_m"), -math.inf),
        "minimum_lift_m": _rank_number(metrics.get("minimum_lift_m"), -math.inf),
    }


def high_thumb_candidate_rank(result: Mapping[str, Any]) -> tuple[Any, ...]:
    acquisition, pose, lift = _result_flags(result)
    stage = str(result.get("stage", "acquisition"))
    rank = candidate_rank_metrics(result)
    hard_grasp = acquisition and pose
    if stage == "lift":
        return (
            not lift,
            not hard_grasp,
            -rank["all_state_gate_steps"],
            -rank["pose_min_normalized_margin"],
            -_rank_number(result.get("thumb_target_rad"), -math.inf),
            -rank["grasp_stability_margin"],
            -rank["alignment_duty"],
            rank["alignment_height_p95_m"],
            -rank["minimum_pad_force_fraction"],
            -rank["minimum_target_face_duty"],
            rank["peak_total_distal_contact_force_n"],
            rank["actuator_saturation_fraction"],
            int(result.get("candidate_id", 2**63 - 1)),
        )
    if not hard_grasp:
        # A failed high-target command must not outrank a quantitatively much
        # closer lower-target grasp.  This branch is used only to choose what
        # deserves local refinement; it never turns a near miss into evidence.
        return (
            True,
            not acquisition,
            not pose,
            -rank["all_state_gate_steps"],
            -rank["pose_min_normalized_margin"],
            -rank["alignment_duty"],
            rank["alignment_height_p95_m"],
            -rank["minimum_pad_force_fraction"],
            -rank["minimum_target_face_duty"],
            -_rank_number(result.get("thumb_target_rad"), -math.inf),
            rank["peak_total_distal_contact_force_n"],
            rank["actuator_saturation_fraction"],
            int(result.get("candidate_id", 2**63 - 1)),
        )
    return (
        False,
        False,
        False,
        -_rank_number(result.get("thumb_target_rad"), -math.inf),
        -rank["pose_min_normalized_margin"],
        -rank["grasp_stability_margin"],
        -rank["alignment_duty"],
        rank["alignment_height_p95_m"],
        -rank["minimum_pad_force_fraction"],
        -rank["minimum_target_face_duty"],
        rank["peak_total_distal_contact_force_n"],
        rank["actuator_saturation_fraction"],
        int(result.get("candidate_id", 2**63 - 1)),
    )


def rank_high_thumb_results(
    results: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    materialized = [copy.deepcopy(dict(value)) for value in results]
    materialized.sort(key=high_thumb_candidate_rank)
    return tuple(materialized)


def _thumb_trace_metrics(trace_path: Path) -> dict[str, Any]:
    with np.load(trace_path, allow_pickle=False) as trace:
        required = {
            "thumb_bend_command_rad",
            "thumb_bend_qpos_rad",
            "actuator_force",
            "actuator_order",
            "grasp_acquisition_step",
        }
        if not required.issubset(trace.files):
            return {"available": False}
        command = np.asarray(trace["thumb_bend_command_rad"], dtype=np.float64)
        qpos = np.asarray(trace["thumb_bend_qpos_rad"], dtype=np.float64)
        force = np.asarray(trace["actuator_force"], dtype=np.float64)
        order = tuple(str(value) for value in np.asarray(trace["actuator_order"]))
        step = int(np.asarray(trace["grasp_acquisition_step"]).reshape(()))
        gate = (
            np.asarray(trace["grasp_gate"], dtype=bool)
            if "grasp_gate" in trace.files
            else None
        )
        states = (
            np.asarray(trace["control_state"]).astype(str)
            if "control_state" in trace.files
            else None
        )
    if command.ndim != 1 or qpos.shape != command.shape or not np.isfinite(command).all() or not np.isfinite(qpos).all():
        return {"available": False}
    terminal = len(command) - 1 if step < 0 else min(step, len(command) - 1)
    prefix = slice(0, terminal + 1)
    try:
        force_values = force[prefix, order.index(THUMB_BEND_ACTUATOR)]
    except (ValueError, IndexError):
        force_values = np.asarray([], dtype=np.float64)
    error = command[prefix] - qpos[prefix]
    max_gate_run = 0
    if (
        gate is not None
        and states is not None
        and gate.ndim == 2
        and gate.shape[0] == command.shape[0]
        and states.shape == command.shape
    ):
        close_or_verify = np.isin(states, ("CLOSE", "VERIFY"))
        all_gate = np.all(gate, axis=1) & close_or_verify
        current_run = 0
        for passed in all_gate:
            current_run = current_run + 1 if bool(passed) else 0
            max_gate_run = max(max_gate_run, current_run)
    return {
        "available": True,
        "scope_end_step": terminal,
        "command_rad": {
            "min": float(np.min(command[prefix])),
            "p50": float(np.median(command[prefix])),
            "max": float(np.max(command[prefix])),
        },
        "actual_qpos_rad": {
            "min": float(np.min(qpos[prefix])),
            "p50": float(np.median(qpos[prefix])),
            "max": float(np.max(qpos[prefix])),
        },
        "tracking_error_rad": {
            "abs_p50": float(np.median(np.abs(error))),
            "abs_max": float(np.max(np.abs(error))),
        },
        "max_consecutive_all_gate_steps_close_verify": int(max_gate_run),
        "actuator_force": {
            "abs_p50": float(np.median(np.abs(force_values))) if force_values.size else None,
            "abs_max": float(np.max(np.abs(force_values))) if force_values.size else None,
        },
    }


def _candidate_classification(stage: str, acquisition: bool, pose: bool, lift: bool) -> str:
    if stage == "lift" and lift:
        return "validated_pose_preserving_high_thumb_lift"
    if acquisition and pose:
        return "validated_pose_preserving_high_thumb_grasp"
    if acquisition:
        return "high_thumb_grasp_pose_preservation_failed"
    if pose:
        return "high_thumb_pose_preserved_grasp_not_acquired"
    return "high_thumb_pose_preserving_near_miss"


def _candidate_result_payload(
    job: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    config_file_sha256: str,
    trace_file_sha256: str,
    thumb_trace_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "candidate_id": int(job["candidate_id"]),
        "summary": summary,
    }
    acquisition = acquisition_succeeded(record)
    pose = pose_preservation_succeeded(record)
    stage_status = summary.get("stage_status", {})
    lift = bool(stage_status.get("full_success", summary.get("passed", False)))
    search_stage = str(job["stage"])
    stage = "lift" if search_stage == "lift" else "acquisition"
    provisional = {
        **copy.deepcopy(dict(job)),
        "summary": summary,
        "acquisition_success": acquisition,
        "pose_preservation_success": pose,
        "lift_success": lift,
        "thumb_bend_trace_metrics": copy.deepcopy(dict(thumb_trace_metrics)),
    }
    provisional.pop("config", None)
    provisional.pop("output_directory", None)
    provisional.pop("artifact_directory", None)
    rank_metrics = candidate_rank_metrics(provisional)
    return {
        "candidate_result_schema_version": CANDIDATE_RESULT_SCHEMA_VERSION,
        "complete": True,
        # Dedicated refiners may reuse the atomic trajectory writer while
        # keeping their own provenance namespace.  Existing v7 jobs omit the
        # field and therefore retain the historical campaign kind.
        "campaign_kind": str(job.get("campaign_kind", CAMPAIGN_KIND)),
        "stage": stage,
        "search_stage": search_stage,
        "candidate_id": int(job["candidate_id"]),
        "candidate_sha256": str(job["candidate_sha256"]),
        "source_family_id": str(job["source_family_id"]),
        "source_trajectory_id": str(job["source_trajectory_id"]),
        "edge_m": float(job["edge_m"]),
        "thumb_target_rad": float(job["thumb_target_rad"]),
        "classification": _candidate_classification(stage, acquisition, pose, lift),
        "grasp_success": grasp_succeeded(record),
        "pose_preservation_success": pose,
        "acquisition_success": acquisition,
        "lift_success": lift,
        "rank_metrics": rank_metrics,
        "thumb_bend_trace_metrics": copy.deepcopy(dict(thumb_trace_metrics)),
        "summary": copy.deepcopy(dict(summary)),
        "artifacts": {
            "resolved_config": "resolved_config.json",
            "trace": "trace.npz",
            "video": None,
            "sha256": {
                "resolved_config": config_file_sha256,
                "trace": trace_file_sha256,
                "video": None,
            },
        },
    }


def execute_high_thumb_candidate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Run one complete MuJoCo trajectory and atomically bind its artifacts."""

    from ..simulation import run_simulation

    output = Path(str(job["output_directory"]))
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as staging:
        staging_path = Path(staging)
        config_path = staging_path / "resolved_config.json"
        trace_path = staging_path / "trace.npz"
        write_json(config_path, job["config"])
        summary = run_simulation(copy.deepcopy(dict(job["config"])), trace_path=trace_path)
        if not trace_path.is_file():
            raise RuntimeError("run_simulation did not create trace.npz")
        payload = _candidate_result_payload(
            job,
            summary,
            config_file_sha256=file_sha256(config_path),
            trace_file_sha256=file_sha256(trace_path),
            thumb_trace_metrics=_thumb_trace_metrics(trace_path),
        )
        write_json(staging_path / "result.json", payload)
        staging_path.rename(output)
    return {
        **copy.deepcopy(payload),
        "config": copy.deepcopy(dict(job["config"])),
        "artifact_directory": str(job["artifact_directory"]),
        "reused": False,
    }


def run_high_thumb_candidate_jobs(
    jobs: Sequence[Mapping[str, Any]], workers: int
) -> tuple[dict[str, Any], ...]:
    count = _positive_int(workers, "workers")
    if not jobs:
        return ()
    if count == 1:
        results = [execute_high_thumb_candidate_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=count, mp_context=context) as executor:
            results = list(executor.map(execute_high_thumb_candidate_job, jobs, chunksize=1))
    results.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(results)


def _load_reusable_candidate(job: Mapping[str, Any]) -> dict[str, Any] | None:
    directory = Path(str(job["output_directory"]))
    if not directory.exists():
        return None
    result_path = directory / "result.json"
    config_path = directory / "resolved_config.json"
    trace_path = directory / "trace.npz"
    if not all(path.is_file() for path in (result_path, config_path, trace_path)):
        raise RuntimeError(f"incomplete candidate artifact directory: {directory}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise RuntimeError(f"candidate result is incomplete: {result_path}")
    if int(payload.get("candidate_id", -1)) != int(job["candidate_id"]):
        raise RuntimeError(f"candidate ID mismatch: {result_path}")
    if payload.get("candidate_sha256") != job["candidate_sha256"]:
        raise RuntimeError(f"candidate semantic digest mismatch: {result_path}")
    persisted = load_config(config_path)
    if canonical_sha256(persisted) != job["candidate_sha256"]:
        raise RuntimeError(f"persisted candidate config changed: {config_path}")
    hashes = payload.get("artifacts", {}).get("sha256", {})
    if hashes.get("resolved_config") != file_sha256(config_path):
        raise RuntimeError(f"candidate config file hash mismatch: {config_path}")
    if hashes.get("trace") != file_sha256(trace_path):
        raise RuntimeError(f"candidate trace file hash mismatch: {trace_path}")
    try:
        with np.load(trace_path, allow_pickle=False) as trace:
            if not trace.files:
                raise RuntimeError(f"candidate trace is empty: {trace_path}")
    except (OSError, ValueError) as error:
        raise RuntimeError(f"candidate trace is unreadable: {trace_path}") from error
    return {
        **copy.deepcopy(dict(payload)),
        "config": copy.deepcopy(dict(job["config"])),
        "artifact_directory": str(job["artifact_directory"]),
        "reused": True,
    }


def _run_or_resume_candidate_jobs(
    candidates: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    workers: int,
    resume: bool,
    executor: CandidateExecutor,
) -> tuple[dict[str, Any], ...]:
    jobs = []
    for candidate in candidates:
        relative = Path("candidates") / f"candidate_{int(candidate['candidate_id'])}"
        jobs.append(
            {
                **copy.deepcopy(dict(candidate)),
                "artifact_directory": str(relative),
                "output_directory": str(output / relative),
            }
        )
    complete: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for job in jobs:
        reusable = _load_reusable_candidate(job) if resume else None
        (complete if reusable is not None else pending).append(reusable or job)
    executed = tuple(executor(tuple(pending), workers)) if pending else ()
    expected = {int(job["candidate_id"]): job for job in pending}
    received = [int(value.get("candidate_id", -1)) for value in executed]
    if len(received) != len(expected) or set(received) != set(expected):
        raise RuntimeError("candidate executor did not preserve candidate IDs")
    if len(received) != len(set(received)):
        raise RuntimeError("candidate executor returned duplicate candidate IDs")
    for value in executed:
        result = copy.deepcopy(dict(value))
        job = expected[int(result["candidate_id"])]
        if result.get("candidate_sha256") != job["candidate_sha256"]:
            raise RuntimeError("candidate executor rebound a candidate config")
        if result.get("config") != job["config"]:
            raise RuntimeError("candidate executor returned a different config")
        complete.append(result)
    complete.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(complete)


def thumb_selection_band(
    target_rad: float,
    bands: Sequence[Sequence[float]] = ((1.25, 1.31), (1.31, 1.38), (1.38, 1.45)),
) -> str:
    """Classify with the registered lower-open-except-first policy."""

    target = _finite(target_rad, "target_rad")
    for index, raw in enumerate(bands):
        lower, upper = (_finite(raw[0], "band lower"), _finite(raw[1], "band upper"))
        inside = lower - 1e-12 <= target <= upper + 1e-12 if index == 0 else lower + 1e-12 < target <= upper + 1e-12
        if inside:
            return f"band_{index}"
    raise ValueError(f"thumb target {target} is outside the selection bands")


def select_diverse_grasps(
    results: Iterable[Mapping[str, Any]],
    *,
    count: int = 12,
    minimum_distinct_edges: int = 4,
    minimum_seed_families: int = 3,
    minimum_per_thumb_band: int = 3,
    maximum_per_edge_target_pair: int = 2,
    bands: Sequence[Sequence[float]] = ((1.25, 1.31), (1.31, 1.38), (1.38, 1.45)),
) -> DiversitySelection:
    """Select successful grasps with deterministic coverage quotas.

    Band quotas are filled first.  Within each quota, new source families and
    edges are preferred before the normal physical rank.  This makes the
    output independent of executor order while avoiding a combinatorial
    optimiser in the million-sample path.
    """

    requested = _positive_int(count, "count")
    min_edges = _positive_int(minimum_distinct_edges, "minimum_distinct_edges")
    min_families = _positive_int(minimum_seed_families, "minimum_seed_families")
    min_band = _positive_int(minimum_per_thumb_band, "minimum_per_thumb_band")
    max_pair = _positive_int(maximum_per_edge_target_pair, "maximum_per_edge_target_pair")
    passing = [
        copy.deepcopy(dict(value))
        for value in results
        if _result_flags(value)[0] and _result_flags(value)[1]
    ]
    passing.sort(key=high_thumb_candidate_rank)
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[int] = set()
    pair_counts: Counter[tuple[float, float]] = Counter()

    def permitted(value: Mapping[str, Any]) -> bool:
        pair = (round(float(value["edge_m"]), 12), round(float(value["thumb_target_rad"]), 12))
        return int(value["candidate_id"]) not in chosen_ids and pair_counts[pair] < max_pair

    def add(value: Mapping[str, Any]) -> None:
        record = copy.deepcopy(dict(value))
        chosen.append(record)
        chosen_ids.add(int(record["candidate_id"]))
        pair_counts[(round(float(record["edge_m"]), 12), round(float(record["thumb_target_rad"]), 12))] += 1

    for band_index in range(len(bands) - 1, -1, -1):
        label = f"band_{band_index}"
        for _ in range(min_band):
            candidates = [
                value
                for value in passing
                if permitted(value)
                and thumb_selection_band(float(value["thumb_target_rad"]), bands) == label
            ]
            if not candidates:
                break
            current_edges = {round(float(value["edge_m"]), 12) for value in chosen}
            current_families = {str(value["source_family_id"]) for value in chosen}
            candidates.sort(
                key=lambda value: (
                    round(float(value["edge_m"]), 12) in current_edges,
                    str(value["source_family_id"]) in current_families,
                    high_thumb_candidate_rank(value),
                )
            )
            add(candidates[0])

    while len(chosen) < requested:
        candidates = [value for value in passing if permitted(value)]
        if not candidates:
            break
        current_edges = {round(float(value["edge_m"]), 12) for value in chosen}
        current_families = {str(value["source_family_id"]) for value in chosen}
        candidates.sort(
            key=lambda value: (
                len(current_edges) >= min_edges or round(float(value["edge_m"]), 12) in current_edges,
                len(current_families) >= min_families or str(value["source_family_id"]) in current_families,
                high_thumb_candidate_rank(value),
            )
        )
        add(candidates[0])

    selected = tuple(chosen[:requested])
    edge_count = len({round(float(value["edge_m"]), 12) for value in selected})
    family_count = len({str(value["source_family_id"]) for value in selected})
    band_counts = Counter(
        thumb_selection_band(float(value["thumb_target_rad"]), bands) for value in selected
    )
    deficiencies: list[str] = []
    if len(selected) < requested:
        deficiencies.append(f"selected_count={len(selected)}<{requested}")
    if edge_count < min_edges:
        deficiencies.append(f"distinct_edges={edge_count}<{min_edges}")
    if family_count < min_families:
        deficiencies.append(f"seed_families={family_count}<{min_families}")
    for index in range(len(bands)):
        label = f"band_{index}"
        if band_counts[label] < min_band:
            deficiencies.append(f"{label}={band_counts[label]}<{min_band}")
    return DiversitySelection(
        selected=selected,
        requested_count=requested,
        satisfied=not deficiencies,
        distinct_edges=edge_count,
        distinct_seed_families=family_count,
        bend_band_counts=dict(band_counts),
        deficiencies=tuple(deficiencies),
    )


def select_lift_seeds(
    selected_grasps: Sequence[Mapping[str, Any]], count: int = 6
) -> tuple[dict[str, Any], ...]:
    requested = _positive_int(count, "count")
    ranked = list(rank_high_thumb_results(selected_grasps))
    selected: list[dict[str, Any]] = []
    while ranked and len(selected) < requested:
        edges = {round(float(value["edge_m"]), 12) for value in selected}
        families = {str(value["source_family_id"]) for value in selected}
        ranked.sort(
            key=lambda value: (
                round(float(value["edge_m"]), 12) in edges,
                str(value["source_family_id"]) in families,
                high_thumb_candidate_rank(value),
            )
        )
        selected.append(ranked.pop(0))
    return tuple(selected)


def generate_lift_candidates(
    selected_grasps: Sequence[Mapping[str, Any]],
    template: Mapping[str, Any],
    *,
    count_per_seed: int,
    seed: int,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    """Generate bounded relative manipulation commands from acquired grasps."""

    per_seed = _positive_int(count_per_seed, "count_per_seed")
    search = resolve_experiment(dict(template)).search_bounds
    bounds = search.manipulation_delta_rad
    if bounds is None:
        raise ValueError("schema-v7 experiment has no manipulation delta bounds")
    generated: list[dict[str, Any]] = []
    for seed_index, grasp in enumerate(selected_grasps):
        base_config = copy.deepcopy(dict(grasp["config"]))
        for local_index in range(per_seed):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [seed, int(grasp["candidate_id"]), local_index, 7_500_001]
                )
            )
            manipulation = {
                name: float(rng.uniform(*bounds[name])) for name in ACTIVE_ACTUATORS
            }
            candidate_id = _LIFT_STAGE_BASE + seed_index * _ID_STRIDE + local_index
            config = copy.deepcopy(base_config)
            config["control"]["manipulation_delta_rad"] = manipulation
            metadata = copy.deepcopy(config.get("candidate_metadata", {}))
            metadata.update(
                {
                    "campaign_kind": CAMPAIGN_KIND,
                    "candidate_id": candidate_id,
                    "stage": "lift",
                    "parent_acquisition_candidate_id": int(grasp["candidate_id"]),
                    "local_index": local_index,
                    "seed": seed,
                    "cube_pose_sampled": False,
                    "free_cube_pose_reset_during_run": False,
                }
            )
            config["candidate_metadata"] = metadata
            if validator is not None:
                validator(config)
            generated.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_sha256": canonical_sha256(config),
                    "stage": "lift",
                    "search_stage": "lift",
                    "source_family_id": str(grasp["source_family_id"]),
                    "source_trajectory_id": str(grasp["source_trajectory_id"]),
                    "edge_m": float(grasp["edge_m"]),
                    "thumb_target_rad": float(grasp["thumb_target_rad"]),
                    "parent_acquisition_candidate_id": int(grasp["candidate_id"]),
                    "local_index": local_index,
                    "config": config,
                }
            )
    generated.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(generated)


def _local_from_config(config: Mapping[str, Any]) -> np.ndarray:
    pose = cube_world_pose_for_size(config, config["cube"]["edge_m"], config["cube"]["center_xy_m"])
    rpy = _finite_vector(config["hand_pose"]["rpy_deg"], 3, "candidate hand RPY")
    root = _finite_vector(config["hand_pose"]["translation_m"], 3, "candidate hand root")
    return rpy_degrees_to_rotation_matrix(rpy).T @ (
        np.asarray(pose["position_m"], dtype=np.float64) - root
    )


def generate_local_refinement_candidates(
    seed_results: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    template: Mapping[str, Any],
    *,
    count_per_seed: int,
    seed: int,
    stage: str = "local_refine",
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    """Refine hand-only variables around ranked dynamic candidates."""

    per_seed = _positive_int(count_per_seed, "count_per_seed")
    source_by_family = {str(value["source_family_id"]): value for value in sources}
    _, pregrasp_bounds, target_bounds = _bounds(template)
    definition = resolve_experiment(dict(template))
    stage_base = _LOCAL_STAGE_BASE if stage == "local_refine" else _FINE_STAGE_BASE
    generated: list[dict[str, Any]] = []
    for seed_index, result in enumerate(seed_results):
        family = str(result["source_family_id"])
        if family not in source_by_family:
            raise ValueError(f"unknown source family {family}")
        source = source_by_family[family]
        base = result["config"]
        base_rpy = _finite_vector(base["hand_pose"]["rpy_deg"], 3, "seed RPY")
        base_local = _local_from_config(base)
        base_pregrasp = _target_mapping(base["control"]["pregrasp_targets_rad"], "seed pregrasp")
        base_targets = _target_mapping(base["control"]["grasp_targets_rad"], "seed targets")
        base_starts = _close_group_starts(base)
        for local_index in range(per_seed):
            if local_index == 0:
                unit = np.zeros(21, dtype=np.float64)
                sample_seed = None
            else:
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, int(result["candidate_id"]), local_index, 7_200_001]
                    )
                )
                unit = 2.0 * rng.random(21) - 1.0
                sample_seed = seed
            rpy = _local_rpy_sample(
                template,
                base_rpy,
                unit[:3],
                definition,
                radius_deg=(0.35, 0.50, 0.35),
            )
            local = _clipped_local_pose(
                template,
                base_local + unit[3:6] * np.asarray((0.0015, 0.0015, 0.0015)),
                base_local,
                definition,
            )
            pregrasp = _clip_mapping(
                base_pregrasp,
                {
                    name: float(unit[6 + index] * 0.03)
                    for index, name in enumerate(ACTIVE_ACTUATORS)
                },
                pregrasp_bounds,
            )
            targets = dict(base_targets)
            non_thumb = [name for name in ACTIVE_ACTUATORS if name != THUMB_BEND_ACTUATOR]
            for index, name in enumerate(non_thumb):
                targets[name] = float(
                    np.clip(targets[name] + unit[14 + index] * 0.04, *target_bounds[name])
                )
            target = float(result["thumb_target_rad"])
            targets[THUMB_BEND_ACTUATOR] = target
            starts = {}
            for group_index, group in enumerate(CLOSE_GROUP_ORDER):
                end = min(
                    float(template["control"]["close_profile"][name]["end_fraction"])
                    for name in CLOSE_GROUP_ACTUATORS[group]
                )
                # Reuse the final three dimensions cyclically.  Closure timing
                # is a low-dimensional nuisance parameter, not a new pose.
                starts[group] = float(
                    np.clip(
                        base_starts[group] + unit[18 + group_index] * 0.04,
                        0.0,
                        end - 0.02,
                    )
                )
            candidate_id = stage_base + seed_index * _ID_STRIDE + local_index
            config = materialize_high_thumb_candidate(
                source,
                template,
                edge_m=float(result["edge_m"]),
                thumb_target_rad=target,
                hand_rpy_deg=rpy,
                cube_in_root_m=local,
                pregrasp_targets_rad=pregrasp,
                grasp_targets_rad=targets,
                close_group_start_fractions=starts,
                candidate_id=candidate_id,
                stage=stage,
                candidate_metadata={
                    "parent_candidate_id": int(result["candidate_id"]),
                    "local_index": local_index,
                    "seed": sample_seed,
                },
                validator=validator,
            )
            generated.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_sha256": canonical_sha256(config),
                    "stage": stage,
                    "source_family_id": family,
                    "source_trajectory_id": str(result["source_trajectory_id"]),
                    "edge_m": float(result["edge_m"]),
                    "thumb_target_rad": target,
                    "parent_candidate_id": int(result["candidate_id"]),
                    "local_index": local_index,
                    "config": config,
                }
            )
    return tuple(generated)


def local_refinement_seeds(
    results: Iterable[Mapping[str, Any]],
    *,
    sizes_per_target: int,
    seeds_per_size: int,
) -> tuple[dict[str, Any], ...]:
    """Choose the registered 5 x 4 x 2 deterministic local seeds."""

    size_count = _positive_int(sizes_per_target, "sizes_per_target")
    seed_count = _positive_int(seeds_per_size, "seeds_per_size")
    ranked = rank_high_thumb_results(results)
    selected: list[dict[str, Any]] = []
    targets = sorted({round(float(value["thumb_target_rad"]), 12) for value in ranked})
    for target in targets:
        target_group = [value for value in ranked if round(float(value["thumb_target_rad"]), 12) == target]
        edge_best: dict[float, dict[str, Any]] = {}
        for value in target_group:
            edge_best.setdefault(round(float(value["edge_m"]), 12), value)
        chosen_edges = sorted(edge_best, key=lambda edge: high_thumb_candidate_rank(edge_best[edge]))[:size_count]
        for edge in chosen_edges:
            group = [
                value
                for value in target_group
                if round(float(value["edge_m"]), 12) == edge
            ]
            selected.extend(group[:seed_count])
    return tuple(selected)


def fine_size_target_cells(
    results: Iterable[Mapping[str, Any]],
    policy: HighThumbCampaignPolicy,
) -> tuple[dict[str, float], ...]:
    """Refine one coarse step around passing or best near-miss geometry."""

    ranked = rank_high_thumb_results(results)
    if not ranked:
        return ()
    passing = [value for value in ranked if _result_flags(value)[0] and _result_flags(value)[1]]
    anchors = passing if passing else list(ranked[:8])
    edge_min, edge_max = policy.coarse_edges_m[0], policy.coarse_edges_m[-1]
    thumb_min, thumb_max = policy.coarse_thumb_targets_rad[0], policy.coarse_thumb_targets_rad[-1]
    cells: set[tuple[float, float]] = set()
    for anchor in anchors[:16]:
        edge = float(anchor["edge_m"])
        target = float(anchor["thumb_target_rad"])
        for edge_delta in (-policy.edge_fine_step_m, 0.0, policy.edge_fine_step_m):
            for target_delta in (-policy.thumb_fine_step_rad, 0.0, policy.thumb_fine_step_rad):
                fine_edge = round(edge + edge_delta, 12)
                fine_target = round(target + target_delta, 12)
                if edge_min - 1e-12 <= fine_edge <= edge_max + 1e-12 and thumb_min - 1e-12 <= fine_target <= thumb_max + 1e-12:
                    cells.add((fine_edge, fine_target))
    return tuple(
        {"edge_m": edge, "thumb_target_rad": target}
        for edge, target in sorted(cells)
    )


def generate_fine_candidates(
    ranked_results: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    template: Mapping[str, Any],
    *,
    limit: int,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    """Transfer the nearest ranked hand relation to each 1 mm / .01 rad cell."""

    requested = _positive_int(limit, "limit")
    source_by_family = {str(value["source_family_id"]): value for value in sources}
    generated: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        edge = float(cell["edge_m"])
        target = float(cell["thumb_target_rad"])
        seeds = sorted(
            ranked_results,
            key=lambda value: (
                abs(float(value["edge_m"]) - edge),
                abs(float(value["thumb_target_rad"]) - target),
                high_thumb_candidate_rank(value),
            ),
        )
        for seed_result in seeds[:2]:
            family = str(seed_result["source_family_id"])
            source = source_by_family[family]
            base = seed_result["config"]
            candidate_id = _FINE_STAGE_BASE + cell_index * _ID_STRIDE + len(generated) % _ID_STRIDE
            targets = dict(base["control"]["grasp_targets_rad"])
            targets[THUMB_BEND_ACTUATOR] = target
            config = materialize_high_thumb_candidate(
                source,
                template,
                edge_m=edge,
                thumb_target_rad=target,
                hand_rpy_deg=base["hand_pose"]["rpy_deg"],
                cube_in_root_m=_local_from_config(base),
                pregrasp_targets_rad=base["control"]["pregrasp_targets_rad"],
                grasp_targets_rad=targets,
                close_group_start_fractions=_close_group_starts(base),
                candidate_id=candidate_id,
                stage="fine",
                candidate_metadata={"parent_candidate_id": int(seed_result["candidate_id"])},
                validator=validator,
            )
            generated.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_sha256": canonical_sha256(config),
                    "stage": "fine",
                    "source_family_id": family,
                    "source_trajectory_id": str(seed_result["source_trajectory_id"]),
                    "edge_m": edge,
                    "thumb_target_rad": target,
                    "parent_candidate_id": int(seed_result["candidate_id"]),
                    "config": config,
                }
            )
            if len(generated) >= requested:
                return tuple(generated)
    return tuple(generated)


def generate_exact_reverify_candidates(
    ranked_results: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> tuple[dict[str, Any], ...]:
    requested = _positive_int(limit, "limit")
    generated = []
    for index, result in enumerate(ranked_results[:requested]):
        config = copy.deepcopy(dict(result["config"]))
        candidate_id = _EXACT_STAGE_BASE + index
        metadata = copy.deepcopy(config.get("candidate_metadata", {}))
        metadata.update(
            {
                "candidate_id": candidate_id,
                "stage": "exact",
                "parent_candidate_id": int(result["candidate_id"]),
                "locked_timestep_s": 0.001,
            }
        )
        config["candidate_metadata"] = metadata
        generated.append(
            {
                "candidate_id": candidate_id,
                "candidate_sha256": canonical_sha256(config),
                "stage": "exact",
                "source_family_id": str(result["source_family_id"]),
                "source_trajectory_id": str(result["source_trajectory_id"]),
                "edge_m": float(result["edge_m"]),
                "thumb_target_rad": float(result["thumb_target_rad"]),
                "parent_candidate_id": int(result["candidate_id"]),
                "config": config,
            }
        )
    return tuple(generated)


def budget_from_config(config: Mapping[str, Any]) -> HighThumbCampaignBudget:
    campaign = resolve_experiment(dict(config)).high_thumb_size_campaign
    if campaign is None:
        raise ValueError("schema-v7 experiment has no high-thumb campaign")
    return HighThumbCampaignBudget(
        static_samples_per_cell=campaign.static_samples_per_cell,
        static_retain_per_cell=campaign.static_retain_per_cell,
        dynamic_candidate_limit=campaign.dynamic_candidate_count,
        local_refine_sizes_per_target=campaign.local_sizes_per_thumb_band,
        local_refine_seeds_per_size=campaign.local_seeds_per_size,
        local_refine_per_seed=campaign.local_refine_per_seed,
        fine_dynamic_limit=campaign.fine_dynamic_candidate_count,
        exact_reverify_limit=campaign.exact_candidate_count,
        selected_grasp_count=campaign.selected_grasp_count,
        selected_lift_seed_count=campaign.selected_lift_seed_count,
        lift_candidates_per_seed=campaign.manipulation_candidates_per_lift_seed,
        perturbations_per_grasp=campaign.perturbations_per_grasp,
    )


def _round_robin_static_retained(
    cell_results: Sequence[Mapping[str, Any]], limit: int
) -> tuple[dict[str, Any], ...]:
    requested = _positive_int(limit, "dynamic candidate limit")
    by_cell = [
        sorted(
            (copy.deepcopy(dict(value)) for value in result["retained"]),
            key=lambda value: (
                not bool(value.get("static_pass", False)),
                float(value.get("static_score", math.inf)),
                int(value["candidate_id"]),
            ),
        )
        for result in sorted(cell_results, key=lambda value: int(value["cell_index"]))
    ]
    selected = []
    depth = 0
    while len(selected) < requested:
        added = False
        for group in by_cell:
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) >= requested:
                    break
        if not added:
            break
        depth += 1
    return tuple(selected)


def _perturbation_candidates(
    selected: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    template: Mapping[str, Any],
    *,
    count_per_grasp: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    local = generate_local_refinement_candidates(
        selected,
        sources,
        template,
        count_per_seed=count_per_grasp,
        seed=seed,
        stage="local_refine",
    )
    result = []
    for index, candidate in enumerate(local):
        record = copy.deepcopy(candidate)
        candidate_id = _PERTURB_STAGE_BASE + index
        config = record["config"]
        config["candidate_metadata"]["candidate_id"] = candidate_id
        config["candidate_metadata"]["stage"] = "perturbation"
        config["candidate_metadata"]["perturbation_of_candidate_id"] = int(
            record["parent_candidate_id"]
        )
        record.update(
            {
                "candidate_id": candidate_id,
                "candidate_sha256": canonical_sha256(config),
                "stage": "perturbation",
            }
        )
        result.append(record)
    return tuple(result)


def _candidate_manifest_record(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": int(result["candidate_id"]),
        "candidate_sha256": str(result["candidate_sha256"]),
        "stage": str(result["stage"]),
        "search_stage": str(result.get("search_stage", result["stage"])),
        "source_family_id": str(result["source_family_id"]),
        "edge_m": float(result["edge_m"]),
        "thumb_target_rad": float(result["thumb_target_rad"]),
    }


def _candidate_result_binding(output: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    relative = Path(str(result["artifact_directory"])) / "result.json"
    path = output / relative
    if not path.is_file():
        raise RuntimeError(f"completed candidate has no result artifact: {path}")
    return {
        **_candidate_manifest_record(result),
        "classification": str(result["classification"]),
        "acquisition_success": bool(result["acquisition_success"]),
        "pose_preservation_success": bool(result["pose_preservation_success"]),
        "lift_success": bool(result["lift_success"]),
        "result": str(relative),
        "result_sha256": file_sha256(path),
    }


def run_high_thumb_variable_size_campaign(
    source_catalog_path: str | Path = DEFAULT_SOURCE_CATALOG,
    template_path: str | Path = DEFAULT_TEMPLATE,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    workers: int = 1,
    resume: bool = False,
    dry_run: bool = False,
    policy: HighThumbCampaignPolicy | None = None,
    budget: HighThumbCampaignBudget | None = None,
    acquisition_only: bool = False,
    static_executor: StaticExecutor = run_static_cell_jobs,
    candidate_executor: CandidateExecutor = run_high_thumb_candidate_jobs,
) -> dict[str, Any]:
    """Run or resume the complete coarse/refine/exact/grasp/lift campaign."""

    worker_count = _positive_int(workers, "workers")
    catalog = Path(source_catalog_path).expanduser().resolve()
    template_file = Path(template_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    sources = load_authenticated_v6_sources(catalog)
    template = load_config(template_file)
    resolved_policy = policy or policy_from_config(template)
    resolved_budget = budget or budget_from_config(template)
    cells = coarse_size_target_cells(resolved_policy)
    input_payload = {
        "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        "source_catalog_sha256": file_sha256(catalog),
        "template_file_sha256": file_sha256(template_file),
        "template_semantic_sha256": canonical_sha256(template),
        "tuner_source_sha256": file_sha256(Path(__file__)),
        "source_sha256": [str(value["source_sha256"]) for value in sources],
        "policy": asdict(resolved_policy),
        "budget": asdict(resolved_budget),
        "acquisition_only": bool(acquisition_only),
    }
    input_sha = canonical_sha256(input_payload)
    plan_payload = {
        **input_payload,
        "campaign_input_sha256": input_sha,
        "source_count": len(sources),
        "coarse_cell_count": len(cells),
        "coarse_static_sample_count": len(cells) * resolved_budget.static_samples_per_cell,
        "maximum_coarse_dynamic_count": resolved_budget.dynamic_candidate_limit,
        "maximum_local_dynamic_count": (
            len(resolved_policy.coarse_thumb_targets_rad)
            * resolved_budget.local_refine_sizes_per_target
            * resolved_budget.local_refine_seeds_per_size
            * resolved_budget.local_refine_per_seed
        ),
        "maximum_fine_dynamic_count": resolved_budget.fine_dynamic_limit,
        "maximum_exact_count": resolved_budget.exact_reverify_limit,
        "maximum_perturbation_count": (
            resolved_budget.selected_grasp_count * resolved_budget.perturbations_per_grasp
        ),
        "maximum_lift_count": (
            0
            if acquisition_only
            else resolved_budget.selected_lift_seed_count * resolved_budget.lift_candidates_per_seed
        ),
    }
    if dry_run:
        return {**plan_payload, "dry_run": True, "output_directory_created": False}

    manifest_path = output / "campaign_manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output}; pass --resume")
        if not manifest_path.is_file():
            raise RuntimeError("resume output has no campaign_manifest.json")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("campaign_input_sha256") != input_sha:
            raise RuntimeError("resume campaign inputs do not match existing manifest")
    else:
        output.mkdir(parents=True)
        write_json(
            manifest_path,
            {
                **plan_payload,
                "source_catalog": str(catalog),
                "template": str(template_file),
                "output_directory": str(output),
                "candidates": [],
            },
        )

    static_jobs = []
    for cell in cells:
        cell_input = {
            "campaign_input_sha256": input_sha,
            "cell": cell,
            "sample_count": resolved_budget.static_samples_per_cell,
            "retain_count": resolved_budget.static_retain_per_cell,
        }
        relative = Path("static") / f"cell_{int(cell['cell_index']):02d}"
        static_jobs.append(
            {
                **copy.deepcopy(cell),
                "cell_input_sha256": canonical_sha256(cell_input),
                "sources": sources,
                "template": template,
                "sample_count": resolved_budget.static_samples_per_cell,
                "retain_count": resolved_budget.static_retain_per_cell,
                "seed": resolved_policy.seed,
                "output_directory": str(output / relative),
            }
        )
    static_results = []
    pending_static = []
    for job in static_jobs:
        reusable = _load_reusable_static_cell(job) if resume else None
        (static_results if reusable is not None else pending_static).append(reusable or job)
    executed_static = tuple(static_executor(tuple(pending_static), worker_count)) if pending_static else ()
    static_results.extend(copy.deepcopy(dict(value)) for value in executed_static)
    static_results.sort(key=lambda value: int(value["cell_index"]))
    if len(static_results) != len(cells):
        raise RuntimeError("static executor did not return every search cell")

    coarse_candidates = _round_robin_static_retained(
        static_results, resolved_budget.dynamic_candidate_limit
    )
    all_results: list[dict[str, Any]] = list(
        _run_or_resume_candidate_jobs(
            coarse_candidates,
            output,
            workers=worker_count,
            resume=resume,
            executor=candidate_executor,
        )
    )
    local_seeds = local_refinement_seeds(
        all_results,
        sizes_per_target=resolved_budget.local_refine_sizes_per_target,
        seeds_per_size=resolved_budget.local_refine_seeds_per_size,
    )
    local_candidates = generate_local_refinement_candidates(
        local_seeds,
        sources,
        template,
        count_per_seed=resolved_budget.local_refine_per_seed,
        seed=resolved_policy.seed,
    )
    all_results.extend(
        _run_or_resume_candidate_jobs(
            local_candidates,
            output,
            workers=worker_count,
            resume=resume,
            executor=candidate_executor,
        )
    )
    ranked_acquisition = rank_high_thumb_results(all_results)
    fine_cells = fine_size_target_cells(ranked_acquisition, resolved_policy)
    fine_candidates = generate_fine_candidates(
        ranked_acquisition,
        fine_cells,
        sources,
        template,
        limit=resolved_budget.fine_dynamic_limit,
    )
    all_results.extend(
        _run_or_resume_candidate_jobs(
            fine_candidates,
            output,
            workers=worker_count,
            resume=resume,
            executor=candidate_executor,
        )
    )
    exact_candidates = generate_exact_reverify_candidates(
        rank_high_thumb_results(all_results), limit=resolved_budget.exact_reverify_limit
    )
    exact_results = _run_or_resume_candidate_jobs(
        exact_candidates,
        output,
        workers=worker_count,
        resume=resume,
        executor=candidate_executor,
    )
    all_results.extend(exact_results)

    registered = resolve_experiment(template).high_thumb_size_campaign
    assert registered is not None
    selection = select_diverse_grasps(
        exact_results,
        count=resolved_budget.selected_grasp_count,
        minimum_distinct_edges=registered.minimum_distinct_edges,
        minimum_seed_families=registered.minimum_seed_families,
        minimum_per_thumb_band=registered.minimum_per_thumb_band,
        maximum_per_edge_target_pair=registered.maximum_per_edge_target_pair,
        bands=registered.thumb_selection_bands_rad,
    )
    perturbation_results: tuple[dict[str, Any], ...] = ()
    lift_seed_results: tuple[dict[str, Any], ...] = ()
    lift_results: tuple[dict[str, Any], ...] = ()
    if selection.selected:
        perturbations = _perturbation_candidates(
            selection.selected,
            sources,
            template,
            count_per_grasp=resolved_budget.perturbations_per_grasp,
            seed=resolved_policy.seed,
        )
        perturbation_results = _run_or_resume_candidate_jobs(
            perturbations,
            output,
            workers=worker_count,
            resume=resume,
            executor=candidate_executor,
        )
        all_results.extend(perturbation_results)
    if selection.satisfied and not acquisition_only:
        lift_seed_results = select_lift_seeds(
            selection.selected, resolved_budget.selected_lift_seed_count
        )
        lift_candidates = generate_lift_candidates(
            lift_seed_results,
            template,
            count_per_seed=resolved_budget.lift_candidates_per_seed,
            seed=resolved_policy.seed,
        )
        lift_results = _run_or_resume_candidate_jobs(
            lift_candidates,
            output,
            workers=worker_count,
            resume=resume,
            executor=candidate_executor,
        )
        all_results.extend(lift_results)

    all_results.sort(key=lambda value: int(value["candidate_id"]))
    manifest_candidates = [_candidate_manifest_record(value) for value in all_results]
    manifest = {
        **plan_payload,
        "source_catalog": str(catalog),
        "template": str(template_file),
        "output_directory": str(output),
        "candidates": manifest_candidates,
    }
    write_json(manifest_path, manifest)
    bindings = [_candidate_result_binding(output, value) for value in all_results]
    acquisition_passes = [value for value in exact_results if _result_flags(value)[0] and _result_flags(value)[1]]
    lift_passes = [value for value in lift_results if _result_flags(value)[2]]
    perturbation_by_parent: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for value in perturbation_results:
        parent = int(value["config"]["candidate_metadata"]["perturbation_of_candidate_id"])
        perturbation_by_parent[parent].append(value)
    campaign_results = {
        "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        # Written only after every externally hashed result binding exists.
        "complete": True,
        "campaign_input_sha256": input_sha,
        "workers": worker_count,
        "static": {
            "cell_count": len(static_results),
            "sample_count": sum(int(value["sample_count"]) for value in static_results),
            "pass_count": sum(int(value["static_pass_count"]) for value in static_results),
            "retained_count": sum(int(value["retained_count"]) for value in static_results),
        },
        "completed_dynamic_candidate_count": len(all_results),
        "exact_acquisition_pass_count": len(acquisition_passes),
        "selected_grasp_count": len(selection.selected),
        "selection_satisfied": selection.satisfied,
        "selection_deficiencies": list(selection.deficiencies),
        "selected_grasps": [
            {
                "candidate_id": int(value["candidate_id"]),
                "source_family_id": str(value["source_family_id"]),
                "edge_m": float(value["edge_m"]),
                "thumb_target_rad": float(value["thumb_target_rad"]),
                "perturbation_pass_count": sum(
                    _result_flags(item)[0] and _result_flags(item)[1]
                    for item in perturbation_by_parent[int(value["candidate_id"])]
                ),
                "perturbation_count": len(
                    perturbation_by_parent[int(value["candidate_id"])]
                ),
            }
            for value in selection.selected
        ],
        "selected_lift_seed_ids": [int(value["candidate_id"]) for value in lift_seed_results],
        "lift_pass_count": len(lift_passes),
        "highest_passing_thumb_target_rad": (
            max(float(value["thumb_target_rad"]) for value in acquisition_passes)
            if acquisition_passes
            else None
        ),
        "stop_reason": (
            "completed_grasp_and_lift_search"
            if selection.satisfied and not acquisition_only
            else "completed_acquisition_only"
            if selection.satisfied
            else "insufficient_hard_passing_diverse_grasps"
        ),
        "candidate_results": bindings,
    }
    write_json(output / "campaign_results.json", campaign_results)
    return campaign_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the schema-v7 high-thumb variable-size pose-preserving "
            "grasp and lift campaign."
        )
    )
    parser.add_argument("--source-catalog", default=str(DEFAULT_SOURCE_CATALOG))
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--acquisition-only", action="store_true")
    parser.add_argument("--edges-mm", type=float, nargs="+", default=None)
    parser.add_argument("--thumb-targets-rad", type=float, nargs="+", default=None)
    parser.add_argument("--static-samples-per-cell", type=int, default=None)
    parser.add_argument("--static-retain-per-cell", type=int, default=None)
    parser.add_argument("--dynamic-candidate-limit", type=int, default=None)
    parser.add_argument("--local-sizes-per-target", type=int, default=None)
    parser.add_argument("--local-seeds-per-size", type=int, default=None)
    parser.add_argument("--local-refine-per-seed", type=int, default=None)
    parser.add_argument("--fine-dynamic-limit", type=int, default=None)
    parser.add_argument("--exact-reverify-limit", type=int, default=None)
    parser.add_argument("--selected-grasp-count", type=int, default=None)
    parser.add_argument("--selected-lift-seed-count", type=int, default=None)
    parser.add_argument("--lift-candidates-per-seed", type=int, default=None)
    parser.add_argument("--perturbations-per-grasp", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    template = load_config(args.template)
    base_policy = policy_from_config(template)
    edges = (
        base_policy.coarse_edges_m
        if args.edges_mm is None
        else tuple(sorted(set(float(value) / 1000.0 for value in args.edges_mm)))
    )
    targets = (
        base_policy.coarse_thumb_targets_rad
        if args.thumb_targets_rad is None
        else tuple(sorted(set(float(value) for value in args.thumb_targets_rad)))
    )
    registered_edges = policy_from_config(template).coarse_edges_m
    if edges[0] < registered_edges[0] - 1e-12 or edges[-1] > registered_edges[-1] + 1e-12:
        raise ValueError("--edges-mm must stay within the registered 52--70 mm range")
    policy = HighThumbCampaignPolicy(
        coarse_edges_m=edges,
        coarse_thumb_targets_rad=targets,
        edge_fine_step_m=base_policy.edge_fine_step_m,
        thumb_fine_step_rad=base_policy.thumb_fine_step_rad,
        fixed_mass_kg=base_policy.fixed_mass_kg,
        friction=base_policy.friction,
        cube_center_xy_m=base_policy.cube_center_xy_m,
        seed=base_policy.seed,
    )
    base_budget_values = asdict(budget_from_config(template))
    overrides = {
        "static_samples_per_cell": args.static_samples_per_cell,
        "static_retain_per_cell": args.static_retain_per_cell,
        "dynamic_candidate_limit": args.dynamic_candidate_limit,
        "local_refine_sizes_per_target": args.local_sizes_per_target,
        "local_refine_seeds_per_size": args.local_seeds_per_size,
        "local_refine_per_seed": args.local_refine_per_seed,
        "fine_dynamic_limit": args.fine_dynamic_limit,
        "exact_reverify_limit": args.exact_reverify_limit,
        "selected_grasp_count": args.selected_grasp_count,
        "selected_lift_seed_count": args.selected_lift_seed_count,
        "lift_candidates_per_seed": args.lift_candidates_per_seed,
        "perturbations_per_grasp": args.perturbations_per_grasp,
    }
    base_budget_values.update(
        {name: value for name, value in overrides.items() if value is not None}
    )
    budget = HighThumbCampaignBudget(**base_budget_values)
    result = run_high_thumb_variable_size_campaign(
        args.source_catalog,
        args.template,
        args.output_dir,
        workers=args.workers,
        resume=args.resume,
        dry_run=args.dry_run,
        acquisition_only=args.acquisition_only,
        policy=policy,
        budget=budget,
    )
    print(json_text(result))
    if args.dry_run:
        return 0
    return 0 if bool(result["selection_satisfied"]) else 2


__all__ = [
    "CAMPAIGN_KIND",
    "CAMPAIGN_SCHEMA_VERSION",
    "DEFAULT_OUTPUT",
    "DEFAULT_SOURCE_CATALOG",
    "DEFAULT_TEMPLATE",
    "DiversitySelection",
    "EXPERIMENT_ID",
    "HighThumbCampaignBudget",
    "HighThumbCampaignPolicy",
    "assert_pose_preserving_candidate_invariants",
    "budget_from_config",
    "build_parser",
    "candidate_rank_metrics",
    "coarse_size_target_cells",
    "cube_world_pose_for_size",
    "execute_high_thumb_candidate_job",
    "execute_static_cell_job",
    "fine_size_target_cells",
    "generate_exact_reverify_candidates",
    "generate_fine_candidates",
    "generate_high_thumb_candidates",
    "generate_lift_candidates",
    "generate_local_refinement_candidates",
    "high_thumb_candidate_rank",
    "load_authenticated_v6_sources",
    "local_refinement_seeds",
    "main",
    "materialize_high_thumb_candidate",
    "policy_from_config",
    "rank_high_thumb_results",
    "run_high_thumb_candidate_jobs",
    "run_high_thumb_variable_size_campaign",
    "run_static_cell_jobs",
    "screen_static_candidates",
    "select_diverse_grasps",
    "select_lift_seeds",
    "thumb_selection_band",
]
