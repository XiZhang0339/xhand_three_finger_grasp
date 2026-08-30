"""Deterministic schema-v8 static discovery of new hand/object poses.

This module owns only the geometric stage of the new-pose campaign.  It never
calls ``mj_step`` and never rewrites the cube freejoint.  Each cell fixes the
cube's world pose, size and material; generated candidates may vary only the
fixed hand-root pose and the pregrasp/grasp controller values.  Real collision
geoms, ``mj_geomDistance`` witnesses and point Jacobians provide the promotion
evidence consumed by a later (separate) dynamics scheduler.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from ..closure_alignment import (
    closure_alignment_from_velocity,
    closure_direction_within_limits,
    point_jacobian_command_velocity,
)
from ..config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    resolved_pose_constraint_values,
)
from ..contact_geometry import (
    active_nondistal_collision_geom_ids,
    distal_collision_geom_ids,
    geom_distance_witness,
    nearest_distal_target_witness,
)
from ..contacts import Face
from ..experiment import resolve_experiment
from ..scene import (
    build_model,
    cube_vertical_half_extent_m,
    rpy_degrees_to_quaternion,
    rpy_degrees_to_rotation_matrix,
)
from .pose_preserving_seed_campaign import canonical_sha256


EXPERIMENT_ID = (
    "left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift"
)
CAMPAIGN_KIND = "normal_aligned_new_pose_static_search"
STATIC_RESULT_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
DEFAULT_EDGES_M = tuple(value / 1000.0 for value in range(60, 71))
DEFAULT_THUMB_TARGETS_RAD = (1.25, 1.30, 1.35, 1.40, 1.45)
DEFAULT_TARGET_GAP_M = (-0.0005, 0.003)
DEFAULT_ALPHA_COUNT = 33
DEFAULT_COARSE_ALPHA_COUNT = 7
DEFAULT_COARSE_NEAR_MARGIN_M = 0.008
_CANDIDATE_BASE = 91_000_000_000_000
_CELL_STRIDE = 1_000_000

_FACE_BY_LABEL = {
    "+X": Face.X_POS,
    "-X": Face.X_NEG,
    "+Y": Face.Y_POS,
    "-Y": Face.Y_NEG,
    "+Z": Face.Z_POS,
    "-Z": Face.Z_NEG,
}


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _vector(values: Any, length: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain {length} finite values") from error
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return vector.copy()


def _positive_int(value: Any, label: str) -> int:
    integer = int(value)
    if isinstance(value, bool) or integer != value or integer <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return integer


def _source_config(value: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = value.get("config", value)
    if not isinstance(candidate, Mapping):
        raise ValueError("existing pose must be a config or contain config")
    for key in ("hand_pose", "cube", "scene", "control"):
        if key not in candidate:
            raise ValueError(f"existing pose config is missing {key}")
    return candidate


def _target_mapping(values: Mapping[str, Any], label: str) -> dict[str, float]:
    if set(values) != set(ACTIVE_ACTUATORS):
        raise ValueError(f"{label} must contain exactly the active actuators")
    return {
        name: _finite(values[name], f"{label}.{name}")
        for name in ACTIVE_ACTUATORS
    }


@dataclass(frozen=True, slots=True)
class PoseSearchCell:
    cell_index: int
    edge_m: float
    thumb_target_rad: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cell_index, int)
            or isinstance(self.cell_index, bool)
            or self.cell_index < 0
        ):
            raise ValueError("cell_index must be a non-negative integer")
        edge = _finite(self.edge_m, "edge_m")
        thumb = _finite(self.thumb_target_rad, "thumb_target_rad")
        if not 0.060 - 1e-12 <= edge <= 0.070 + 1e-12:
            raise ValueError("edge_m must lie within [0.060, 0.070]")
        if not 1.25 - 1e-12 <= thumb <= 1.45 + 1e-12:
            raise ValueError("thumb_target_rad must lie within [1.25, 1.45]")
        object.__setattr__(self, "edge_m", edge)
        object.__setattr__(self, "thumb_target_rad", thumb)

    @property
    def cell_id(self) -> str:
        return (
            f"edge_{self.edge_m * 1000.0:.0f}mm_"
            f"thumb_{self.thumb_target_rad:.2f}rad"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_index": self.cell_index,
            "cell_id": self.cell_id,
            "edge_m": self.edge_m,
            "thumb_target_rad": self.thumb_target_rad,
        }


@dataclass(frozen=True, slots=True)
class StaticPoseSearchBudget:
    samples_per_cell: int = 10_000
    retain_per_cell: int = 4

    def __post_init__(self) -> None:
        _positive_int(self.samples_per_cell, "samples_per_cell")
        _positive_int(self.retain_per_cell, "retain_per_cell")
        if self.samples_per_cell >= _CELL_STRIDE:
            raise ValueError("samples_per_cell must be smaller than the cell stride")

    def as_dict(self, *, cell_count: int) -> dict[str, int]:
        count = _positive_int(cell_count, "cell_count")
        return {
            "cell_count": count,
            "samples_per_cell": self.samples_per_cell,
            "declared_sample_count": count * self.samples_per_cell,
            "retain_per_cell": self.retain_per_cell,
            "maximum_retained_count": count * self.retain_per_cell,
        }


def pose_search_cells(template: Mapping[str, Any]) -> tuple[PoseSearchCell, ...]:
    """Return the registered 11-by-5 grid in stable edge-major order."""

    definition = resolve_experiment(dict(template))
    campaign = definition.normal_aligned_smooth_lift_campaign
    if int(template.get("schema_version", 0)) != 8 or campaign is None:
        raise ValueError("template must resolve to the registered schema-v8 campaign")
    cells: list[PoseSearchCell] = []
    for edge in campaign.edges_m:
        for thumb in campaign.thumb_targets_rad:
            cells.append(PoseSearchCell(len(cells), edge, thumb))
    return tuple(cells)


def cube_initial_world_pose(config: Mapping[str, Any]) -> dict[str, list[float]]:
    """Resolve the support-placed free cube pose without compiling a model."""

    cube = config["cube"]
    edge = _finite(cube["edge_m"], "cube.edge_m")
    centre = _vector(cube["center_xy_m"], 2, "cube.center_xy_m")
    rpy = _vector(cube.get("rpy_deg", (0.0, 0.0, 0.0)), 3, "cube.rpy_deg")
    rotation = rpy_degrees_to_rotation_matrix(rpy)
    support = _finite(config["scene"]["support_top_z_m"], "support_top_z_m")
    offset = _finite(cube.get("z_offset_m", 0.0), "cube.z_offset_m")
    return {
        "position_m": [
            float(centre[0]),
            float(centre[1]),
            support + cube_vertical_half_extent_m(edge, rotation) + offset,
        ],
        "quaternion_wxyz": rpy_degrees_to_quaternion(rpy).tolist(),
        "rpy_deg": rpy.tolist(),
    }


def pose_id_for_config(config: Mapping[str, Any]) -> str:
    """Hash only geometry/topology; controller changes cannot alter this ID."""

    return canonical_sha256(
        {
            "cube": config["cube"],
            "hand_pose": config["hand_pose"],
            "contact_topology": config["contact_topology"],
        }
    )


def controller_id_for_config(config: Mapping[str, Any]) -> str:
    """Hash only the close/manipulation command law, never object/hand pose."""

    protocol = config["control_protocol"]
    return canonical_sha256(
        {
            "control": config["control"],
            "close_s": protocol["close_s"],
            "manipulation_profile": protocol.get("manipulation_profile"),
        }
    )


def _cube_position_in_root(config: Mapping[str, Any]) -> np.ndarray:
    cube_world = _vector(
        cube_initial_world_pose(config)["position_m"], 3, "cube world position"
    )
    root = _vector(config["hand_pose"]["translation_m"], 3, "hand translation")
    rotation = rpy_degrees_to_rotation_matrix(config["hand_pose"]["rpy_deg"])
    return rotation.T @ (cube_world - root)


def _contains_local(definition: Any, local: np.ndarray) -> bool:
    if not definition.search_bounds.contains_cube_position(local):
        return False
    constraints = definition.far_hand_pose_constraints
    if constraints is None:
        return True
    lower, upper = constraints.root_cube_distance_m
    distance = float(np.linalg.norm(local))
    return lower - 1e-12 <= distance <= upper + 1e-12


def _full_pose_sample(definition: Any, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    from .far_hand_fingertip import (
        _feasible_tilt_for_roll_deg,
        root_pitch_for_finger_down_tilt_deg,
    )

    search = definition.search_bounds
    constraints = definition.far_hand_pose_constraints
    for _ in range(128):
        roll = float(rng.uniform(*search.hand_roll_deg))
        tilt = float(rng.uniform(*search.palm_pitch_deg))
        if constraints is not None:
            tilt = _feasible_tilt_for_roll_deg(
                tilt, roll_deg=roll, constraints=constraints
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
        if _contains_local(definition, local):
            return np.asarray((roll, pitch, yaw)), local
    raise ValueError("registered pose envelope has no feasible sample")


def _local_pose_sample(
    definition: Any,
    base_config: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb a seed while retaining the registered tilt/distance envelope."""

    from .far_hand_fingertip import (
        _feasible_tilt_for_roll_deg,
        root_pitch_for_finger_down_tilt_deg,
    )

    base_rpy = _vector(base_config["hand_pose"]["rpy_deg"], 3, "source hand RPY")
    base_local = _cube_position_in_root(base_config)
    base_tilt = float(
        resolved_pose_constraint_values(dict(base_config))["finger_down_tilt_deg"]
    )
    constraints = definition.far_hand_pose_constraints
    if constraints is None:
        raise ValueError("schema-v8 local pose search requires pose constraints")
    for _ in range(64):
        roll = float(
            np.clip(
                base_rpy[0] + rng.uniform(-1.0, 1.0),
                *definition.search_bounds.hand_roll_deg,
            )
        )
        yaw = float(
            np.clip(
                base_rpy[2] + rng.uniform(-1.0, 1.0),
                *definition.search_bounds.hand_yaw_deg,
            )
        )
        tilt = _feasible_tilt_for_roll_deg(
            float(
                np.clip(
                    base_tilt + rng.uniform(-1.5, 1.5),
                    *constraints.finger_down_tilt_deg,
                )
            ),
            roll_deg=roll,
            constraints=constraints,
        )
        proposed_rpy = np.asarray(
            (roll, root_pitch_for_finger_down_tilt_deg(roll, tilt), yaw),
            dtype=np.float64,
        )
        local = base_local + rng.uniform(-0.003, 0.003, size=3)
        if _contains_local(definition, local):
            return proposed_rpy, local
    return _full_pose_sample(definition, rng)


def _sample_targets(
    source: Mapping[str, Any],
    template: Mapping[str, Any],
    definition: Any,
    rng: np.random.Generator | None,
    *,
    thumb_target_rad: float,
    full_envelope: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    source_control = source["control"]
    source_pregrasp = _target_mapping(
        source_control.get(
            "pregrasp_targets_rad",
            template["control"]["pregrasp_targets_rad"],
        ),
        "source pregrasp",
    )
    source_grasp = _target_mapping(
        source_control["grasp_targets_rad"], "source grasp"
    )
    pregrasp_bounds = definition.search_bounds.pregrasp_targets_rad
    grasp_bounds = definition.search_bounds.actuator_targets_rad
    if pregrasp_bounds is None:
        raise ValueError("schema-v8 search requires pregrasp target bounds")
    pregrasp: dict[str, float] = {}
    grasp: dict[str, float] = {}
    for name in ACTIVE_ACTUATORS:
        if full_envelope:
            assert rng is not None
            pregrasp[name] = float(rng.uniform(*pregrasp_bounds[name]))
            grasp[name] = float(rng.uniform(*grasp_bounds[name]))
        elif rng is None:
            pregrasp[name] = float(
                np.clip(source_pregrasp[name], *pregrasp_bounds[name])
            )
            grasp[name] = float(np.clip(source_grasp[name], *grasp_bounds[name]))
        else:
            pregrasp[name] = float(
                np.clip(
                    source_pregrasp[name] + rng.uniform(-0.08, 0.08),
                    *pregrasp_bounds[name],
                )
            )
            grasp[name] = float(
                np.clip(
                    source_grasp[name] + rng.uniform(-0.10, 0.10),
                    *grasp_bounds[name],
                )
            )
    grasp[THUMB_BEND_ACTUATOR] = float(thumb_target_rad)
    return pregrasp, grasp


def _cell_cube_config(
    template: Mapping[str, Any], cell: PoseSearchCell, definition: Any
) -> dict[str, Any]:
    campaign = definition.normal_aligned_smooth_lift_campaign
    assert campaign is not None
    cube = copy.deepcopy(dict(template["cube"]))
    cube.update(
        {
            "edge_m": cell.edge_m,
            "mass_kg": campaign.fixed_mass_kg,
            "friction": campaign.friction,
            "center_xy_m": list(campaign.cube_center_xy_m),
            "z_offset_m": 0.0,
        }
    )
    return cube


def generate_pose_cell_candidates(
    template: Mapping[str, Any],
    existing_poses: Sequence[Mapping[str, Any]],
    cell: PoseSearchCell,
    *,
    count: int,
    start_index: int = 0,
    seed: int | None = None,
    validator: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Generate a prefix-stable, seed-balanced static cell.

    The first occurrence of every supplied pose is an exact hand/control
    anchor transferred to the cell cube.  Later occurrences deterministically
    mix local perturbations with full registered-envelope samples.
    """

    candidate_count = _positive_int(count, "count")
    if (
        not isinstance(start_index, int)
        or isinstance(start_index, bool)
        or start_index < 0
        or start_index + candidate_count > _CELL_STRIDE
    ):
        raise ValueError("start_index + count must stay within the cell stride")
    if not existing_poses:
        raise ValueError("existing_poses must contain at least one pose")
    definition = resolve_experiment(dict(template))
    campaign = definition.normal_aligned_smooth_lift_campaign
    if int(template.get("schema_version", 0)) != 8 or campaign is None:
        raise ValueError("template must be the registered schema-v8 campaign")
    cells = pose_search_cells(template)
    if cell not in cells:
        raise ValueError("cell is not part of the registered schema-v8 grid")
    resolved_seed = campaign.seed if seed is None else int(seed)
    if resolved_seed < 0:
        raise ValueError("seed must be non-negative")
    sources = tuple(_source_config(value) for value in existing_poses)
    source_records = tuple(existing_poses)
    fixed_cube = _cell_cube_config(template, cell, definition)
    fixed_pose_probe = copy.deepcopy(dict(template))
    fixed_pose_probe["cube"] = copy.deepcopy(fixed_cube)
    fixed_cube_world_pose = cube_initial_world_pose(fixed_pose_probe)

    records: list[dict[str, Any]] = []
    for local_index in range(start_index, start_index + candidate_count):
        source_index = local_index % len(sources)
        occurrence = local_index // len(sources)
        source = sources[source_index]
        source_record = source_records[source_index]
        rng: np.random.Generator | None = None
        mode = "source_anchor"
        full_envelope = False
        if occurrence > 0:
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        resolved_seed,
                        cell.cell_index,
                        local_index,
                        int(round(cell.edge_m * 1e6)),
                        int(round(cell.thumb_target_rad * 1e6)),
                        8_100_001,
                    ]
                )
            )
            full_envelope = bool(rng.random() < 0.25)
            mode = "full_registered_envelope" if full_envelope else "seed_local"

        if full_envelope:
            assert rng is not None
            hand_rpy, cube_in_root = _full_pose_sample(definition, rng)
        elif rng is None:
            hand_rpy = _vector(
                source["hand_pose"]["rpy_deg"], 3, "source hand RPY"
            )
            cube_in_root = _cube_position_in_root(source)
            if not _contains_local(definition, cube_in_root):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [resolved_seed, cell.cell_index, local_index, 8_100_002]
                    )
                )
                hand_rpy, cube_in_root = _full_pose_sample(definition, rng)
                mode = "source_anchor_out_of_bounds_fallback"
                full_envelope = True
        else:
            hand_rpy, cube_in_root = _local_pose_sample(definition, source, rng)

        pregrasp, grasp = _sample_targets(
            source,
            template,
            definition,
            rng,
            thumb_target_rad=cell.thumb_target_rad,
            full_envelope=full_envelope,
        )
        config = copy.deepcopy(dict(template))
        config["cube"] = copy.deepcopy(fixed_cube)
        cube_world = _vector(
            fixed_cube_world_pose["position_m"], 3, "fixed cube world position"
        )
        root_rotation = rpy_degrees_to_rotation_matrix(hand_rpy)
        config["hand_pose"] = {
            "rpy_deg": hand_rpy.tolist(),
            "translation_m": (cube_world - root_rotation @ cube_in_root).tolist(),
        }
        source_profile = source["control"].get("close_profile")
        if not isinstance(source_profile, Mapping) or set(source_profile) != set(
            ACTIVE_ACTUATORS
        ):
            source_profile = template["control"]["close_profile"]
        config["control"] = {
            "pregrasp_targets_rad": pregrasp,
            "grasp_targets_rad": grasp,
            "manipulation_delta_rad": {
                name: 0.0 for name in ACTIVE_ACTUATORS
            },
            "close_profile": copy.deepcopy(dict(source_profile)),
        }
        config.pop("run_context", None)
        candidate_id = (
            _CANDIDATE_BASE + cell.cell_index * _CELL_STRIDE + local_index
        )
        source_pose_id = (
            str(source_record.get("pose_id"))
            if isinstance(source_record, Mapping) and source_record.get("pose_id")
            else pose_id_for_config(source)
        )
        config["candidate_metadata"] = {
            "campaign_kind": CAMPAIGN_KIND,
            "stage": "new_pose_static",
            "candidate_id": candidate_id,
            "cell_index": cell.cell_index,
            "cell_id": cell.cell_id,
            "local_index": local_index,
            "seed": resolved_seed,
            "source_pose_id": source_pose_id,
            "sample_mode": mode,
            "sampled_fields": [
                "hand_pose",
                "control.pregrasp_targets_rad",
                "control.grasp_targets_rad",
            ],
            "fixed_cube_initial_world_pose": copy.deepcopy(fixed_cube_world_pose),
            "cube_pose_sampled": False,
            "free_cube_pose_reset_during_scan": False,
        }
        assert_static_candidate_scope(
            config,
            template=template,
            cell=cell,
            expected_cube_world_pose=fixed_cube_world_pose,
        )
        if validator is not None:
            validator(config)
        records.append(
            {
                "candidate_id": candidate_id,
                "cell_index": cell.cell_index,
                "cell_id": cell.cell_id,
                "edge_m": cell.edge_m,
                "thumb_target_rad": cell.thumb_target_rad,
                "source_pose_id": source_pose_id,
                "pose_id": pose_id_for_config(config),
                "controller_id": controller_id_for_config(config),
                "candidate_sha256": canonical_sha256(config),
                "config": config,
            }
        )
    return tuple(records)


def assert_static_candidate_scope(
    config: Mapping[str, Any],
    *,
    template: Mapping[str, Any],
    cell: PoseSearchCell,
    expected_cube_world_pose: Mapping[str, Any] | None = None,
) -> None:
    """Reject object/material mutation or undeclared static-search fields."""

    definition = resolve_experiment(dict(template))
    expected_cube = _cell_cube_config(template, cell, definition)
    if config["cube"] != expected_cube:
        raise ValueError("candidate changed the fixed per-cell cube configuration")
    expected_pose = (
        cube_initial_world_pose(config)
        if expected_cube_world_pose is None
        else dict(expected_cube_world_pose)
    )
    if cube_initial_world_pose(config) != expected_pose:
        raise ValueError("candidate changed the fixed per-cell cube world pose")
    manipulation = config["control"]["manipulation_delta_rad"]
    if manipulation != {name: 0.0 for name in ACTIVE_ACTUATORS}:
        raise ValueError("static acquisition candidates require zero manipulation delta")
    metadata = config.get("candidate_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("candidate metadata is missing")
    if metadata.get("cube_pose_sampled") is not False:
        raise ValueError("candidate must declare cube_pose_sampled=false")
    if metadata.get("free_cube_pose_reset_during_scan") is not False:
        raise ValueError("static search must not reset the cube freejoint")
    allowed = {
        "hand_pose",
        "control.pregrasp_targets_rad",
        "control.grasp_targets_rad",
    }
    if set(metadata.get("sampled_fields", ())) != allowed:
        raise ValueError("static search sampled fields exceed hand pose/control scope")


def _profiled_command(
    model: mujoco.MjModel,
    config: Mapping[str, Any],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return forced qpos targets and analytic target derivatives."""

    pregrasp = config["control"]["pregrasp_targets_rad"]
    grasp = config["control"]["grasp_targets_rad"]
    profile = config["control"]["close_profile"]
    close_s = _finite(config["control_protocol"]["close_s"], "close_s")
    target = np.zeros(model.nu, dtype=np.float64)
    target_velocity = np.zeros(model.nu, dtype=np.float64)
    fraction = float(np.clip(alpha, 0.0, 1.0))
    for name in ACTIVE_ACTUATORS:
        actuator_id = model.actuator(name).id
        start = _finite(profile[name]["start_fraction"], f"{name}.start_fraction")
        end = _finite(profile[name]["end_fraction"], f"{name}.end_fraction")
        local = (fraction - start) / (end - start)
        clipped = float(np.clip(local, 0.0, 1.0))
        progress = clipped * clipped * (3.0 - 2.0 * clipped)
        delta = float(grasp[name]) - float(pregrasp[name])
        target[actuator_id] = float(pregrasp[name]) + progress * delta
        if 0.0 < local < 1.0:
            target_velocity[actuator_id] = (
                delta
                * 6.0
                * clipped
                * (1.0 - clipped)
                / (close_s * (end - start))
            )
    return target, target_velocity


def coarse_closure_alphas(
    config: Mapping[str, Any], *, uniform_count: int = DEFAULT_COARSE_ALPHA_COUNT
) -> tuple[float, ...]:
    """Return the deterministic inexpensive closure grid.

    A plain uniform grid can step over a short actuator motion window.  The
    coarse stage therefore always includes every profile start/end breakpoint
    in addition to a small uniform grid.  The final promotion decision uses a
    deliberately expanded geometric near margin; only the subsequent 33-point
    scan is accepted as static-pass evidence.
    """

    count = _positive_int(uniform_count, "uniform_count")
    if count < 3:
        raise ValueError("uniform_count must be at least three")
    profile = config["control"]["close_profile"]
    if not isinstance(profile, Mapping) or set(profile) != set(ACTIVE_ACTUATORS):
        raise ValueError("close_profile must contain exactly the active actuators")
    values = {float(value) for value in np.linspace(0.0, 1.0, count)}
    for name in ACTIVE_ACTUATORS:
        start = _finite(profile[name]["start_fraction"], f"{name}.start_fraction")
        end = _finite(profile[name]["end_fraction"], f"{name}.end_fraction")
        if not 0.0 <= start < end <= 1.0:
            raise ValueError("close profile fractions must satisfy 0 <= start < end <= 1")
        values.update((start, end))
    return tuple(sorted(values))


def _minimum_geom_gap(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
    geom_ids: Sequence[int],
    *,
    distance_max_m: float = 0.020,
) -> float:
    if not geom_ids:
        return float(distance_max_m)
    segment = np.zeros(6, dtype=np.float64)
    return min(
        float(
            mujoco.mj_geomDistance(
                model,
                data,
                int(cube_geom_id),
                int(geom_id),
                float(distance_max_m),
                segment,
            )
        )
        for geom_id in geom_ids
    )


def _witness_outward_normal(
    witness: Any,
    cube_rotation: np.ndarray,
    target_face: Face,
) -> np.ndarray:
    segment = witness.distal_point_world_m - witness.cube_point_world_m
    distance = float(witness.signed_distance_m)
    norm = float(np.linalg.norm(segment))
    if abs(distance) > 1e-12 and norm > 1e-12:
        outward = segment / distance
        return outward / np.linalg.norm(outward)
    return cube_rotation @ target_face.outward_normal


def _contact_minimum_gap(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
    geom_ids: Sequence[int],
    *,
    clear_value_m: float = 0.020,
) -> tuple[float, tuple[int, ...]]:
    """Read already-generated cube contacts without extra distance queries."""

    selected = {int(value) for value in geom_ids}
    minimum = float(clear_value_m)
    penetrating: set[int] = set()
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        other = -1
        if geom1 == cube_geom_id and geom2 in selected:
            other = geom2
        elif geom2 == cube_geom_id and geom1 in selected:
            other = geom1
        if other < 0:
            continue
        distance = float(contact.dist)
        minimum = min(minimum, distance)
        if distance < -1e-12:
            penetrating.add(other)
    return minimum, tuple(sorted(penetrating))


def _coarse_snapshot_rank(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    gaps = tuple(float(value) for value in metrics["promotion_signed_gap_m"])
    angles = tuple(float(value) for value in metrics["closure_angle_deg"])
    inward = tuple(float(value) for value in metrics["closure_inward_speed_m_s"])
    margin = float(metrics.get("near_margin_m", DEFAULT_COARSE_NEAR_MARGIN_M))
    expanded_gap_violation = sum(
        max(0.0, DEFAULT_TARGET_GAP_M[0] - margin - value)
        + max(0.0, value - DEFAULT_TARGET_GAP_M[1] - margin)
        for value in gaps
    )
    return (
        not bool(metrics["coarse_sampled_pass_hint"]),
        not bool(metrics["safe_promotion_eligible"]),
        int(metrics["missing_target_witness_count"]),
        int(metrics["off_target_penetrating_count"]),
        max(0.0, -float(metrics["minimum_active_nondistal_gap_m"])),
        expanded_gap_violation,
        sum(abs(value) for value in gaps),
        sum(max(0.0, value - 45.0) for value in angles),
        -min(inward),
        max(angles),
        float(metrics["contact_height_spread_m"]),
        float(metrics["closure_alpha"]),
    )


def _coarse_record_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return _coarse_snapshot_rank(record["coarse_metrics"]) + (
        str(record["pose_id"]),
        str(record["controller_id"]),
        int(record["candidate_id"]),
    )


def _evaluate_candidate_coarse(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: Any,
    record: Mapping[str, Any],
    *,
    distal_geom_ids: Mapping[str, Sequence[int]],
    active_nondistal_geom_ids: Mapping[str, Sequence[int]],
    all_hand_geom_ids: Sequence[int],
    uniform_alpha_count: int,
    near_margin_m: float,
) -> dict[str, Any]:
    """Cheap, conservative promotion scan; never supplies pass evidence.

    Every alpha uses real ``mj_geomDistance`` witnesses and real point
    Jacobians.  Expensive all-geom distance checks are replaced by MuJoCo's
    already generated contact list.  A profile-breakpoint grid and an expanded
    gap envelope keep candidates whose best closure lies between coarse
    samples eligible for the exact scan.
    """

    config = record["config"]
    model.body_pos[info.root_body_id] = _vector(
        config["hand_pose"]["translation_m"], 3, "hand translation"
    )
    model.body_quat[info.root_body_id] = rpy_degrees_to_quaternion(
        config["hand_pose"]["rpy_deg"]
    )
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    cube_qpos = np.asarray(
        data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 7],
        dtype=np.float64,
    ).copy()
    active_ids = np.asarray(
        [model.actuator(name).id for name in ACTIVE_ACTUATORS], dtype=np.int64
    )
    active_qpos_addresses = info.actuator_qpos_adrs[active_ids]
    target_faces = {
        finger: _FACE_BY_LABEL[
            str(config["contact_topology"]["target_faces"][finger])
        ]
        for finger in ACTIVE_FINGERS
    }
    thresholds = config["closure_alignment"]
    maximum_angle = float(thresholds["static_max_angle_deg"])
    minimum_inward = float(thresholds["min_inward_speed_m_s"])
    height_limit = float(config["contact_alignment"]["max_height_spread_m"])
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm <= 1e-12:
        raise ValueError("static closure search requires non-zero gravity")
    up = -gravity / gravity_norm

    pregrasp_target, _ = _profiled_command(model, config, 0.0)
    data.qpos[active_qpos_addresses] = pregrasp_target[active_ids]
    mujoco.mj_forward(model, data)
    pregrasp_minimum_hand_gap, pregrasp_penetrating = _contact_minimum_gap(
        model, data, info.cube_geom_id, all_hand_geom_ids
    )
    if not np.array_equal(
        data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 7], cube_qpos
    ):
        raise RuntimeError("static coarse pregrasp evaluation changed the cube qpos")

    alphas = coarse_closure_alphas(config, uniform_count=uniform_alpha_count)
    # A pregrasp penetration is a hard failure independent of the closure
    # trajectory, so it is safe to avoid every remaining geometry query.
    if pregrasp_penetrating:
        return {
            "scan_kind": "coarse_promotion_only",
            "alpha_sample_count": 0,
            "declared_alpha_values": list(alphas),
            "closure_alpha": 0.0,
            "coarse_sampled_pass_hint": False,
            "safe_promotion_eligible": False,
            "safe_early_reject_reason": "pregrasp_penetration",
            "near_margin_m": float(near_margin_m),
            "missing_target_witness_count": len(ACTIVE_FINGERS),
            "off_target_penetrating_count": 0,
            "target_signed_gap_m": [0.035] * len(ACTIVE_FINGERS),
            "promotion_signed_gap_m": [0.035] * len(ACTIVE_FINGERS),
            "closure_angle_deg": [180.0] * len(ACTIVE_FINGERS),
            "closure_inward_speed_m_s": [0.0] * len(ACTIVE_FINGERS),
            "closure_tangent_speed_m_s": [0.0] * len(ACTIVE_FINGERS),
            "contact_height_spread_m": 0.020,
            "minimum_active_nondistal_gap_m": 0.020,
            "pregrasp_minimum_hand_gap_m": pregrasp_minimum_hand_gap,
            "target_witness": {finger: None for finger in ACTIVE_FINGERS},
            "cube_freejoint_qpos_unchanged": True,
        }

    nondistal_flat = tuple(
        geom_id
        for finger in ACTIVE_FINGERS
        for geom_id in active_nondistal_geom_ids[finger]
    )
    observations: list[dict[str, Any]] = []
    distance_max = max(0.035, DEFAULT_TARGET_GAP_M[1] + near_margin_m + 0.010)
    for alpha in alphas:
        target, target_velocity = _profiled_command(model, config, alpha)
        data.qpos[active_qpos_addresses] = target[active_ids]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if not np.array_equal(
            data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 7], cube_qpos
        ):
            raise RuntimeError("static coarse closure scan changed the cube qpos")
        cube_rotation = np.asarray(
            data.geom_xmat[info.cube_geom_id], dtype=np.float64
        ).reshape(3, 3)
        gaps: list[float] = []
        promotion_gaps: list[float] = []
        angles: list[float] = []
        inward_speeds: list[float] = []
        tangent_speeds: list[float] = []
        heights: list[float] = []
        witnesses: dict[str, Any] = {}
        missing = 0
        off_target_penetrating = 0
        clean: list[bool] = []
        for finger in ACTIVE_FINGERS:
            face = target_faces[finger]
            generic = [
                witness
                for geom_id in distal_geom_ids[finger]
                if (
                    witness := geom_distance_witness(
                        model,
                        data,
                        cube_geom_id=info.cube_geom_id,
                        distal_geom_id=int(geom_id),
                        distance_max_m=distance_max,
                    )
                )
                is not None
            ]
            target_candidates = [value for value in generic if value.face is face]
            target_witness = (
                min(
                    target_candidates,
                    key=lambda value: (
                        abs(value.signed_distance_m),
                        value.signed_distance_m,
                        value.distal_geom_id,
                    ),
                )
                if target_candidates
                else None
            )
            promotion_witness = target_witness
            if promotion_witness is None and generic:
                promotion_witness = min(
                    generic,
                    key=lambda value: (
                        abs(value.signed_distance_m),
                        value.signed_distance_m,
                        value.distal_geom_id,
                    ),
                )
            if target_witness is None:
                missing += 1
            if any(
                value.signed_distance_m < -1e-12 and value.face is not face
                for value in generic
            ):
                off_target_penetrating += 1
            if promotion_witness is None:
                gaps.append(distance_max)
                promotion_gaps.append(distance_max)
                angles.append(180.0)
                inward_speeds.append(0.0)
                tangent_speeds.append(0.0)
                heights.append(0.0)
                clean.append(False)
                witnesses[finger] = None
                continue
            outward = cube_rotation @ face.outward_normal
            distal_body = int(model.geom_bodyid[promotion_witness.distal_geom_id])
            point_velocity = point_jacobian_command_velocity(
                model,
                data,
                promotion_witness.distal_point_world_m,
                distal_body,
                info.actuator_dof_adrs,
                target_velocity,
            )
            alignment = closure_alignment_from_velocity(point_velocity, outward)
            target_gap = (
                float(target_witness.signed_distance_m)
                if target_witness is not None
                else distance_max
            )
            promotion_gap = float(promotion_witness.signed_distance_m)
            direction_ok = closure_direction_within_limits(
                alignment,
                maximum_angle_deg=maximum_angle,
                minimum_inward_speed_m_s=minimum_inward,
                require_positive_inward_speed=bool(
                    thresholds["require_positive_inward_speed"]
                ),
            )
            gap_ok = bool(
                target_witness is not None
                and DEFAULT_TARGET_GAP_M[0] - 1e-12
                <= target_gap
                <= DEFAULT_TARGET_GAP_M[1] + 1e-12
            )
            gaps.append(target_gap)
            promotion_gaps.append(promotion_gap)
            angles.append(alignment.angle_deg)
            inward_speeds.append(alignment.inward_speed_m_s)
            tangent_speeds.append(alignment.tangent_speed_m_s)
            heights.append(float(up @ promotion_witness.cube_point_world_m))
            clean.append(gap_ok and direction_ok)
            witnesses[finger] = {
                "distal_geom_id": int(promotion_witness.distal_geom_id),
                "signed_distance_m": promotion_gap,
                "cube_point_world_m": promotion_witness.cube_point_world_m.tolist(),
                "distal_point_world_m": promotion_witness.distal_point_world_m.tolist(),
                "observed_face": promotion_witness.face.name,
                "target_face": face.name,
                "target_face_match": bool(target_witness is not None),
                "closure_command_velocity_world_m_s": point_velocity.tolist(),
                "cube_outward_normal_world": outward.tolist(),
            }
        minimum_nondistal_gap, _ = _contact_minimum_gap(
            model, data, info.cube_geom_id, nondistal_flat
        )
        height_spread = float(np.ptp(heights)) if all(witnesses.values()) else 0.020
        safe_near = bool(
            all(value is not None for value in witnesses.values())
            and all(
                DEFAULT_TARGET_GAP_M[0] - near_margin_m - 1e-12
                <= value
                <= DEFAULT_TARGET_GAP_M[1] + near_margin_m + 1e-12
                for value in promotion_gaps
            )
            and minimum_nondistal_gap >= -near_margin_m
            and height_spread <= height_limit + 2.0 * near_margin_m + 1e-12
        )
        pass_hint = bool(
            missing == 0
            and all(clean)
            and off_target_penetrating == 0
            and minimum_nondistal_gap >= -1e-12
            and height_spread <= height_limit + 1e-12
        )
        observations.append(
            {
                "scan_kind": "coarse_promotion_only",
                "alpha_sample_count": len(alphas),
                "declared_alpha_values": list(alphas),
                "closure_alpha": float(alpha),
                "coarse_sampled_pass_hint": pass_hint,
                "safe_promotion_eligible": safe_near,
                "safe_early_reject_reason": None,
                "near_margin_m": float(near_margin_m),
                "missing_target_witness_count": missing,
                "off_target_penetrating_count": off_target_penetrating,
                "target_signed_gap_m": gaps,
                "promotion_signed_gap_m": promotion_gaps,
                "closure_angle_deg": angles,
                "closure_inward_speed_m_s": inward_speeds,
                "closure_tangent_speed_m_s": tangent_speeds,
                "contact_height_spread_m": height_spread,
                "minimum_active_nondistal_gap_m": minimum_nondistal_gap,
                "pregrasp_minimum_hand_gap_m": pregrasp_minimum_hand_gap,
                "target_witness": witnesses,
                "cube_freejoint_qpos_unchanged": True,
            }
        )
    return min(observations, key=_coarse_snapshot_rank)


def _snapshot_rank(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    gaps = tuple(float(value) for value in metrics["target_signed_gap_m"])
    angles = tuple(float(value) for value in metrics["closure_angle_deg"])
    inward = tuple(float(value) for value in metrics["closure_inward_speed_m_s"])
    gap_violation = sum(
        max(0.0, DEFAULT_TARGET_GAP_M[0] - value)
        + max(0.0, value - DEFAULT_TARGET_GAP_M[1])
        for value in gaps
    )
    angle_violation = sum(max(0.0, value - 45.0) for value in angles)
    return (
        not bool(metrics["static_geometry_pass"]),
        int(metrics["missing_target_witness_count"]),
        int(metrics["off_target_penetrating_count"]),
        max(0.0, -float(metrics["minimum_active_nondistal_gap_m"])),
        gap_violation,
        angle_violation,
        -min(inward),
        max(angles),
        float(metrics["contact_height_spread_m"]),
        float(metrics["closure_alpha"]),
    )


def _static_record_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = record["static_metrics"]
    return _snapshot_rank(metrics) + (
        str(record["pose_id"]),
        str(record["controller_id"]),
        int(record["candidate_id"]),
    )


def _evaluate_candidate_static(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: Any,
    record: Mapping[str, Any],
    *,
    distal_geom_ids: Mapping[str, Sequence[int]],
    active_nondistal_geom_ids: Mapping[str, Sequence[int]],
    all_hand_geom_ids: Sequence[int],
    alpha_count: int,
    alpha_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    config = record["config"]
    model.body_pos[info.root_body_id] = _vector(
        config["hand_pose"]["translation_m"], 3, "hand translation"
    )
    model.body_quat[info.root_body_id] = rpy_degrees_to_quaternion(
        config["hand_pose"]["rpy_deg"]
    )
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    cube_qpos = np.asarray(
        data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 7],
        dtype=np.float64,
    ).copy()
    if int(model.jnt_type[info.cube_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError("schema-v8 static search requires a freejoint cube")
    active_ids = np.asarray(
        [model.actuator(name).id for name in ACTIVE_ACTUATORS], dtype=np.int64
    )
    active_qpos_addresses = info.actuator_qpos_adrs[active_ids]
    target_faces = {
        finger: _FACE_BY_LABEL[
            str(config["contact_topology"]["target_faces"][finger])
        ]
        for finger in ACTIVE_FINGERS
    }
    thresholds = config["closure_alignment"]
    maximum_angle = float(thresholds["static_max_angle_deg"])
    minimum_inward = float(thresholds["min_inward_speed_m_s"])
    height_limit = float(config["contact_alignment"]["max_height_spread_m"])
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm <= 1e-12:
        raise ValueError("static closure search requires non-zero gravity")
    up = -gravity / gravity_norm

    pregrasp_target, _ = _profiled_command(model, config, 0.0)
    data.qpos[active_qpos_addresses] = pregrasp_target[active_ids]
    mujoco.mj_forward(model, data)
    pregrasp_minimum_hand_gap = _minimum_geom_gap(
        model, data, info.cube_geom_id, all_hand_geom_ids
    )
    if not np.array_equal(
        data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 7], cube_qpos
    ):
        raise RuntimeError("static pregrasp evaluation changed the cube qpos")

    resolved_alphas = (
        tuple(float(value) for value in np.linspace(0.0, 1.0, alpha_count))
        if alpha_values is None
        else tuple(float(value) for value in alpha_values)
    )
    if not resolved_alphas or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in resolved_alphas
    ):
        raise ValueError("alpha_values must contain finite fractions within [0, 1]")
    observations: list[dict[str, Any]] = []
    for alpha in resolved_alphas:
        target, target_velocity = _profiled_command(model, config, float(alpha))
        data.qpos[active_qpos_addresses] = target[active_ids]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if not np.array_equal(
            data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 7], cube_qpos
        ):
            raise RuntimeError("static closure scan changed the cube qpos")

        cube_rotation = np.asarray(
            data.geom_xmat[info.cube_geom_id], dtype=np.float64
        ).reshape(3, 3)
        gaps: list[float] = []
        angles: list[float] = []
        inward_speeds: list[float] = []
        tangent_speeds: list[float] = []
        heights: list[float] = []
        witness_records: dict[str, Any] = {}
        clean: list[bool] = []
        missing = 0
        off_target_penetrating = 0
        for finger in ACTIVE_FINGERS:
            face = target_faces[finger]
            witness = nearest_distal_target_witness(
                model,
                data,
                cube_geom_id=info.cube_geom_id,
                distal_geom_ids=distal_geom_ids[finger],
                target_face=face,
                distance_max_m=0.020,
            )
            for geom_id in distal_geom_ids[finger]:
                any_witness = geom_distance_witness(
                    model,
                    data,
                    cube_geom_id=info.cube_geom_id,
                    distal_geom_id=int(geom_id),
                    distance_max_m=0.020,
                )
                if (
                    any_witness is not None
                    and any_witness.signed_distance_m < -1e-12
                    and any_witness.face is not face
                ):
                    off_target_penetrating += 1
            if witness is None:
                missing += 1
                gaps.append(0.020)
                angles.append(180.0)
                inward_speeds.append(0.0)
                tangent_speeds.append(0.0)
                heights.append(0.0)
                clean.append(False)
                witness_records[finger] = None
                continue
            distal_body = int(model.geom_bodyid[witness.distal_geom_id])
            point_velocity = point_jacobian_command_velocity(
                model,
                data,
                witness.distal_point_world_m,
                distal_body,
                info.actuator_dof_adrs,
                target_velocity,
            )
            outward = _witness_outward_normal(witness, cube_rotation, face)
            alignment = closure_alignment_from_velocity(point_velocity, outward)
            gap = float(witness.signed_distance_m)
            direction_ok = closure_direction_within_limits(
                alignment,
                maximum_angle_deg=maximum_angle,
                minimum_inward_speed_m_s=minimum_inward,
                require_positive_inward_speed=bool(
                    thresholds["require_positive_inward_speed"]
                ),
            )
            gap_ok = bool(
                DEFAULT_TARGET_GAP_M[0] - 1e-12
                <= gap
                <= DEFAULT_TARGET_GAP_M[1] + 1e-12
            )
            gaps.append(gap)
            angles.append(alignment.angle_deg)
            inward_speeds.append(alignment.inward_speed_m_s)
            tangent_speeds.append(alignment.tangent_speed_m_s)
            heights.append(float(up @ witness.cube_point_world_m))
            clean.append(gap_ok and direction_ok)
            witness_records[finger] = {
                "distal_geom_id": int(witness.distal_geom_id),
                "signed_distance_m": gap,
                "cube_point_world_m": witness.cube_point_world_m.tolist(),
                "distal_point_world_m": witness.distal_point_world_m.tolist(),
                "face": face.name,
                "normal_alignment": float(
                    witness.classification.normal_alignment
                ),
                "edge_clearance_m": float(
                    witness.classification.edge_clearance_m
                ),
                "closure_command_velocity_world_m_s": point_velocity.tolist(),
                "cube_outward_normal_world": outward.tolist(),
            }

        nondistal_flat = tuple(
            geom_id
            for finger in ACTIVE_FINGERS
            for geom_id in active_nondistal_geom_ids[finger]
        )
        minimum_nondistal_gap = _minimum_geom_gap(
            model, data, info.cube_geom_id, nondistal_flat
        )
        height_spread = (
            float(np.ptp(heights)) if missing == 0 else 0.020
        )
        passed = bool(
            missing == 0
            and all(clean)
            and off_target_penetrating == 0
            and minimum_nondistal_gap >= -1e-12
            and height_spread <= height_limit + 1e-12
            and pregrasp_minimum_hand_gap >= -1e-12
        )
        observations.append(
            {
                "closure_alpha": float(alpha),
                "static_geometry_pass": passed,
                "missing_target_witness_count": missing,
                "off_target_penetrating_count": off_target_penetrating,
                "target_signed_gap_m": gaps,
                "closure_angle_deg": angles,
                "closure_inward_speed_m_s": inward_speeds,
                "closure_tangent_speed_m_s": tangent_speeds,
                "contact_height_spread_m": height_spread,
                "minimum_active_nondistal_gap_m": minimum_nondistal_gap,
                "pregrasp_minimum_hand_gap_m": pregrasp_minimum_hand_gap,
                "target_witness": witness_records,
                "cube_freejoint_qpos_unchanged": True,
            }
        )
    best = min(observations, key=_snapshot_rank)
    return {
        **best,
        "scan_kind": "full_static_evidence",
        "alpha_sample_count": len(resolved_alphas),
        "declared_alpha_values": list(resolved_alphas),
    }


def screen_static_pose_chunk_two_stage(
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 4,
    alpha_count: int = DEFAULT_ALPHA_COUNT,
    coarse_alpha_count: int = DEFAULT_COARSE_ALPHA_COUNT,
    promotion_count: int = 16,
    coarse_near_margin_m: float = DEFAULT_COARSE_NEAR_MARGIN_M,
) -> dict[str, Any]:
    """Coarse-screen all candidates and fully verify only a promotion pool.

    ``evaluated_count`` and ``coarse_scan_count`` include every generated
    candidate.  ``static_pass_count``, final ranking and every retained record
    are derived exclusively from the complete ``alpha_count`` scan.  This is
    intentionally a chunk-level primitive so memory and full-scan work remain
    bounded for a 10,000-sample cell.
    """

    retain = _positive_int(top_k, "top_k")
    full_count = _positive_int(alpha_count, "alpha_count")
    coarse_count = _positive_int(coarse_alpha_count, "coarse_alpha_count")
    promote = _positive_int(promotion_count, "promotion_count")
    margin = _finite(coarse_near_margin_m, "coarse_near_margin_m")
    if full_count < 3:
        raise ValueError("alpha_count must be at least three")
    if coarse_count < 3:
        raise ValueError("coarse_alpha_count must be at least three")
    if promote < retain:
        raise ValueError("promotion_count must be at least top_k")
    if margin < 0.0:
        raise ValueError("coarse_near_margin_m must be non-negative")
    if not candidates:
        raise ValueError("candidates must not be empty")
    cell_indices = {int(value["cell_index"]) for value in candidates}
    if len(cell_indices) != 1:
        raise ValueError("two-stage screening requires exactly one cell")
    first_config = copy.deepcopy(dict(candidates[0]["config"]))
    fixed_cube_hash = canonical_sha256(first_config["cube"])
    for candidate in candidates:
        if canonical_sha256(candidate["config"]["cube"]) != fixed_cube_hash:
            raise ValueError("all candidates in a cell must share one fixed cube")

    model, info = build_model(first_config)
    data = mujoco.MjData(model)
    distal = distal_collision_geom_ids(model, info.distal_weld_ids)
    nondistal = active_nondistal_collision_geom_ids(
        model, info.hand_body_parts, info.distal_weld_ids
    )
    all_hand_geoms = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in info.hand_body_parts
        and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    )

    coarse_records: list[dict[str, Any]] = []
    for candidate in candidates:
        record = copy.deepcopy(dict(candidate))
        record["coarse_metrics"] = _evaluate_candidate_coarse(
            model,
            data,
            info,
            record,
            distal_geom_ids=distal,
            active_nondistal_geom_ids=nondistal,
            all_hand_geom_ids=all_hand_geoms,
            uniform_alpha_count=coarse_count,
            near_margin_m=margin,
        )
        record["coarse_rank"] = list(
            _coarse_snapshot_rank(record["coarse_metrics"])
        )
        coarse_records.append(record)
    coarse_ordered = sorted(coarse_records, key=_coarse_record_rank)

    # Sampled-pass hints are never dropped.  The remaining slots first consume
    # the conservative expanded-near envelope, then deterministic rank
    # fallbacks.  This preserves intermediate-contact candidates even if no
    # exact coarse grid point happens to meet the hard target-gap interval.
    mandatory = [
        value
        for value in coarse_ordered
        if bool(value["coarse_metrics"]["coarse_sampled_pass_hint"])
    ]
    selected_ids = {int(value["candidate_id"]) for value in mandatory}
    target_size = min(len(coarse_ordered), max(promote, len(mandatory)))
    promoted = list(mandatory)
    for eligible_only in (True, False):
        for value in coarse_ordered:
            candidate_id = int(value["candidate_id"])
            if candidate_id in selected_ids:
                continue
            if eligible_only and not bool(
                value["coarse_metrics"]["safe_promotion_eligible"]
            ):
                continue
            promoted.append(value)
            selected_ids.add(candidate_id)
            if len(promoted) >= target_size:
                break
        if len(promoted) >= target_size:
            break
    promoted.sort(key=_coarse_record_rank)

    fully_evaluated: list[dict[str, Any]] = []
    for record in promoted:
        metrics = _evaluate_candidate_static(
            model,
            data,
            info,
            record,
            distal_geom_ids=distal,
            active_nondistal_geom_ids=nondistal,
            all_hand_geom_ids=all_hand_geoms,
            alpha_count=full_count,
        )
        record["static_pass"] = bool(metrics["static_geometry_pass"])
        record["static_metrics"] = metrics
        record["static_rank"] = list(_snapshot_rank(metrics))
        record["static_evidence"] = {
            "scan_kind": "full_static_evidence",
            "alpha_sample_count": full_count,
            "coarse_scan_used_for_pass": False,
        }
        if bool(record["coarse_metrics"]["coarse_sampled_pass_hint"]):
            reason = "coarse_sampled_pass_hint"
        elif bool(record["coarse_metrics"]["safe_promotion_eligible"]):
            reason = "expanded_near_envelope"
        else:
            reason = "deterministic_rank_fallback"
        record["promotion_reason"] = reason
        fully_evaluated.append(record)
    ordered = sorted(fully_evaluated, key=_static_record_rank)
    retained = tuple(ordered[: min(retain, len(ordered))])
    if any(
        value["static_metrics"].get("scan_kind") != "full_static_evidence"
        for value in retained
    ):
        raise RuntimeError("retained static candidates must have full-scan evidence")
    return {
        "static_result_schema_version": STATIC_RESULT_SCHEMA_VERSION,
        "cell_index": next(iter(cell_indices)),
        "cell_id": str(candidates[0]["cell_id"]),
        "evaluated_count": len(coarse_records),
        "coarse_scan_count": len(coarse_records),
        "full_scan_count": len(fully_evaluated),
        "not_promoted_count": len(coarse_records) - len(fully_evaluated),
        "coarse_sampled_pass_hint_count": sum(
            bool(value["coarse_metrics"]["coarse_sampled_pass_hint"])
            for value in coarse_records
        ),
        "coarse_safe_eligible_count": sum(
            bool(value["coarse_metrics"]["safe_promotion_eligible"])
            for value in coarse_records
        ),
        "static_pass_count": sum(
            bool(value["static_pass"]) for value in fully_evaluated
        ),
        "retained_count": len(retained),
        "retained": retained,
        "ranking_policy": (
            "full_scan_only_hard_pass_missing_offtarget_nondistal_gap_angle_"
            "inward_height_pose_controller_candidate_id"
        ),
    }


def screen_static_pose_cell(
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 4,
    alpha_count: int = DEFAULT_ALPHA_COUNT,
) -> dict[str, Any]:
    """Evaluate and deterministically retain one cell's best static poses."""

    retain = _positive_int(top_k, "top_k")
    alphas = _positive_int(alpha_count, "alpha_count")
    if alphas < 3:
        raise ValueError("alpha_count must be at least three")
    if not candidates:
        raise ValueError("candidates must not be empty")
    cell_indices = {int(value["cell_index"]) for value in candidates}
    if len(cell_indices) != 1:
        raise ValueError("screen_static_pose_cell requires exactly one cell")
    first_config = copy.deepcopy(dict(candidates[0]["config"]))
    fixed_cube_hash = canonical_sha256(first_config["cube"])
    for candidate in candidates:
        if canonical_sha256(candidate["config"]["cube"]) != fixed_cube_hash:
            raise ValueError("all candidates in a cell must share one fixed cube")
    model, info = build_model(first_config)
    data = mujoco.MjData(model)
    distal = distal_collision_geom_ids(model, info.distal_weld_ids)
    nondistal = active_nondistal_collision_geom_ids(
        model, info.hand_body_parts, info.distal_weld_ids
    )
    all_hand_geoms = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in info.hand_body_parts
        and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    )
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        record = copy.deepcopy(dict(candidate))
        metrics = _evaluate_candidate_static(
            model,
            data,
            info,
            record,
            distal_geom_ids=distal,
            active_nondistal_geom_ids=nondistal,
            all_hand_geom_ids=all_hand_geoms,
            alpha_count=alphas,
        )
        record["static_pass"] = bool(metrics["static_geometry_pass"])
        record["static_metrics"] = metrics
        record["static_rank"] = list(_snapshot_rank(metrics))
        record["static_evidence"] = {
            "scan_kind": "full_static_evidence",
            "alpha_sample_count": alphas,
            "coarse_scan_used_for_pass": False,
        }
        evaluated.append(record)
    ordered = sorted(evaluated, key=_static_record_rank)
    retained = tuple(ordered[: min(retain, len(ordered))])
    return {
        "static_result_schema_version": STATIC_RESULT_SCHEMA_VERSION,
        "cell_index": next(iter(cell_indices)),
        "cell_id": str(candidates[0]["cell_id"]),
        "evaluated_count": len(evaluated),
        "static_pass_count": sum(bool(value["static_pass"]) for value in evaluated),
        "retained_count": len(retained),
        "retained": retained,
        "ranking_policy": (
            "hard_pass_missing_offtarget_nondistal_gap_angle_inward_height_"
            "pose_controller_candidate_id"
        ),
    }


def retain_top_k_per_cell(
    records: Sequence[Mapping[str, Any]], *, top_k: int
) -> dict[int, tuple[dict[str, Any], ...]]:
    """Pure worker-order-independent top-k selection for screened records."""

    limit = _positive_int(top_k, "top_k")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for value in records:
        if "static_metrics" not in value:
            raise ValueError("screened record is missing static_metrics")
        grouped[int(value["cell_index"])].append(copy.deepcopy(dict(value)))
    return {
        cell: tuple(sorted(values, key=_static_record_rank)[:limit])
        for cell, values in sorted(grouped.items())
    }


def pose_search_trigger_report(
    template: Mapping[str, Any],
    *,
    rescued_full_pass_count: int,
    best_worst_closure_angle_deg: float,
    force_thumb_band_diversity_search: bool = True,
) -> dict[str, Any]:
    """Explain exactly why the independent new-pose stage should run."""

    definition = resolve_experiment(dict(template))
    campaign = definition.normal_aligned_smooth_lift_campaign
    alignment = definition.closure_alignment
    if campaign is None or alignment is None:
        raise ValueError("template has no schema-v8 new-pose campaign")
    pass_count = int(rescued_full_pass_count)
    if (
        isinstance(rescued_full_pass_count, bool)
        or pass_count != rescued_full_pass_count
        or pass_count < 0
    ):
        raise ValueError("rescued_full_pass_count must be a non-negative integer")
    angle = _finite(best_worst_closure_angle_deg, "best_worst_closure_angle_deg")
    reasons: list[str] = []
    if pass_count < campaign.selected_trajectory_count:
        reasons.append("rescued_full_pass_count_below_required")
    if angle > alignment.optimization_target_max_angle_deg + 1e-12:
        reasons.append("closure_optimization_target_not_met")
    if force_thumb_band_diversity_search:
        reasons.append("thumb_band_diversity_search_declared")
    return {
        "triggered": bool(reasons),
        "reasons": reasons,
        "rescued_full_pass_count": pass_count,
        "required_full_pass_count": campaign.selected_trajectory_count,
        "best_worst_closure_angle_deg": angle,
        "optimization_target_max_angle_deg": (
            alignment.optimization_target_max_angle_deg
        ),
        "force_thumb_band_diversity_search": bool(
            force_thumb_band_diversity_search
        ),
    }


def static_pose_search_budget_report(
    template: Mapping[str, Any],
    *,
    completed_cell_results: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Report declared/evaluated static work without starting dynamics."""

    definition = resolve_experiment(dict(template))
    campaign = definition.normal_aligned_smooth_lift_campaign
    if campaign is None:
        raise ValueError("template has no schema-v8 new-pose campaign")
    cells = pose_search_cells(template)
    completed_indices: set[int] = set()
    evaluated = 0
    coarse_scanned = 0
    fully_scanned = 0
    not_promoted = 0
    passed = 0
    retained = 0
    for result in completed_cell_results:
        index = int(result["cell_index"])
        if index in completed_indices:
            raise ValueError("completed_cell_results contains a duplicate cell")
        if not 0 <= index < len(cells):
            raise ValueError("completed cell index is outside the registered grid")
        completed_indices.add(index)
        result_evaluated = int(result["evaluated_count"])
        result_coarse = int(result.get("coarse_scan_count", result_evaluated))
        result_full = int(result.get("full_scan_count", result_evaluated))
        if result_coarse != result_evaluated or not 0 <= result_full <= result_evaluated:
            raise ValueError("completed static scan accounting is inconsistent")
        evaluated += result_evaluated
        coarse_scanned += result_coarse
        fully_scanned += result_full
        not_promoted += result_evaluated - result_full
        passed += int(result["static_pass_count"])
        retained += int(result["retained_count"])
    declared = StaticPoseSearchBudget(
        samples_per_cell=campaign.static_samples_per_cell,
        retain_per_cell=campaign.static_retain_per_cell,
    ).as_dict(cell_count=len(cells))
    return {
        "budget_schema_version": 1,
        "declared": declared,
        "completed_cell_count": len(completed_indices),
        "remaining_cell_count": len(cells) - len(completed_indices),
        "evaluated_candidate_count": evaluated,
        "coarse_scan_count": coarse_scanned,
        "full_scan_count": fully_scanned,
        "not_promoted_count": not_promoted,
        "static_pass_count": passed,
        "retained_candidate_count": retained,
        "complete": len(completed_indices) == len(cells),
        "dynamics_scheduled": False,
    }


__all__ = [
    "CAMPAIGN_KIND",
    "DEFAULT_ALPHA_COUNT",
    "DEFAULT_COARSE_ALPHA_COUNT",
    "DEFAULT_COARSE_NEAR_MARGIN_M",
    "DEFAULT_EDGES_M",
    "DEFAULT_SEED",
    "DEFAULT_TARGET_GAP_M",
    "DEFAULT_THUMB_TARGETS_RAD",
    "EXPERIMENT_ID",
    "PoseSearchCell",
    "StaticPoseSearchBudget",
    "assert_static_candidate_scope",
    "controller_id_for_config",
    "coarse_closure_alphas",
    "cube_initial_world_pose",
    "generate_pose_cell_candidates",
    "pose_id_for_config",
    "pose_search_cells",
    "pose_search_trigger_report",
    "retain_top_k_per_cell",
    "screen_static_pose_cell",
    "screen_static_pose_chunk_two_stage",
    "static_pose_search_budget_report",
]
