"""Static geometry utilities for schema-v9 *measured* grasp poses.

Unlike the earlier closure-sweep screen, this module treats the proposed
contact joint vector as geometry.  It writes that vector to the eight active
joint ``qpos`` entries exactly once, calls :func:`mujoco.mj_forward`, and then
queries the real collision geoms with :func:`mujoco.mj_geomDistance`.  A
command target is never accepted as evidence for the contact pose.

The second half of the screen constructs a genuinely separated pre-contact
pose.  For each finger, the point Jacobian at its distal witness is inverted
to move the pad 2--4 mm along the cube-outward normal.  The reverse motion is
then checked to point inward within 30 degrees.  No ``mj_step`` call, cube
weld, or freejoint reset is used here; dynamic acquisition remains a separate
and mandatory campaign stage.
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..actual_contact_capability import (
    ACTUAL_CONTACT_SCHEMA_VERSIONS,
    LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID,
)
from ..closure_alignment import (
    closure_alignment_from_velocity,
    closure_direction_within_limits,
    point_jacobian_command_velocity,
)
from ..actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
    authenticated_catalog_artifact_paths,
    bind_candidate_result_semantic_sha256,
    build_campaign_manifest,
    commit_campaign_stage,
    export_actual_contact_grasp_pose_catalog,
    export_actual_contact_manipulation_catalog,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from ..actual_contact_selection import select_actual_contact_candidates
from ..artifacts import file_sha256, write_json
from ..config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    load_config,
    resolved_pose_constraint_values,
    validate_config,
)
from ..contact_geometry import (
    active_nondistal_collision_geom_ids,
    distal_collision_geom_ids,
    geom_distance_witness,
    nearest_distal_target_witness,
)
from ..contacts import BoxContactThresholds, Face
from ..experiment import resolve_experiment
from ..grasp_pose import controller_id, grasp_pose_id
from ..scene import (
    build_model,
    cube_vertical_half_extent_m,
    rpy_degrees_to_quaternion,
    rpy_degrees_to_rotation_matrix,
)
from .actual_qpos_sources import (
    ActualQposSource,
    V8ActualQposSource,
    load_actual_qpos_sources,
    load_v8_actual_qpos_sources,
    resolve_registered_source_manifest as _resolve_registered_source_manifest,
)
from .pose_preserving_seed_campaign import canonical_sha256


EXPERIMENT_ID = LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID
CAMPAIGN_KIND = "actual_contact_grasp_pose_static_search"
STATIC_RESULT_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
DEFAULT_DISTANCE_MAX_M = 0.020
_NUMERIC_TOLERANCE = 1e-12
_CANDIDATE_BASE = 101_000_000_000_000
_CELL_STRIDE = 1_000_000
_EVIDENCE_ANCHOR_BASE = 9_000_000
_UNIFORM_REFINE_ID_OFFSET = 800_000
_UNIFORM_TARGET_GAP_M = 0.00015
_LOCAL_REFINEMENT_BASE = 202_000_000_000_000

# Obtain the physical-face witness first, then apply the campaign's strict
# edge/normal thresholds explicitly in ``FingerContactEvidence``.  Using the
# strict classifier inside ``nearest_distal_target_witness`` would collapse a
# 0.49 mm edge miss into ``None`` and lose the diagnostic margin.
_PERMISSIVE_FACE_THRESHOLDS = BoxContactThresholds(
    surface_tolerance_m=1e-7,
    edge_margin_m=1e-6,
    normal_alignment_min=1e-6,
)

_FACE_BY_LABEL = {
    "+X": Face.X_POS,
    "-X": Face.X_NEG,
    "+Y": Face.Y_POS,
    "-Y": Face.Y_NEG,
    "+Z": Face.Z_POS,
    "-Z": Face.Z_NEG,
}

FINGER_ACTUATORS = {
    "thumb": ACTIVE_ACTUATORS[0:3],
    "index": ACTIVE_ACTUATORS[3:6],
    "mid": ACTIVE_ACTUATORS[6:8],
}


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _resolve_actual_contact_definition(
    config: Mapping[str, Any], context: str
) -> Any:
    """Resolve through the module hook retained by legacy static-test fixtures."""

    if int(config.get("schema_version", 0)) not in ACTUAL_CONTACT_SCHEMA_VERSIONS:
        raise ValueError(f"{context} requires an actual-contact schema")
    definition = resolve_experiment(dict(config))
    if definition.actual_contact_grasp_pose_campaign is None:
        raise ValueError(f"{context} requires an actual-contact campaign")
    return definition


def _vector(values: Any, length: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain {length} finite values") from error
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return result.copy()


def _positive_int(value: Any, label: str) -> int:
    result = int(value)
    if isinstance(value, bool) or result != value or result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _target_mapping(values: Mapping[str, Any], label: str) -> dict[str, float]:
    if set(values) != set(ACTIVE_ACTUATORS):
        raise ValueError(f"{label} must contain exactly the eight active actuators")
    return {
        name: _finite(values[name], f"{label}.{name}")
        for name in ACTIVE_ACTUATORS
    }


def _target_faces(config: Mapping[str, Any]) -> dict[str, Face]:
    values = config["contact_topology"]["target_faces"]
    if set(values) != set(ACTIVE_FINGERS):
        raise ValueError("target_faces must contain exactly thumb, index and mid")
    try:
        return {finger: _FACE_BY_LABEL[str(values[finger])] for finger in ACTIVE_FINGERS}
    except KeyError as error:
        raise ValueError("target_faces contains an unsupported face") from error


@dataclass(frozen=True, slots=True)
class ActualContactSearchCell:
    """One edge/actual-thumb cell in the registered 11-by-5 campaign."""

    cell_index: int
    edge_m: float
    thumb_actual_center_rad: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cell_index, int)
            or isinstance(self.cell_index, bool)
            or self.cell_index < 0
        ):
            raise ValueError("cell_index must be a non-negative integer")
        edge = _finite(self.edge_m, "edge_m")
        thumb = _finite(self.thumb_actual_center_rad, "thumb_actual_center_rad")
        if edge <= 0.0:
            raise ValueError("edge_m must be positive")
        if thumb <= 0.0:
            raise ValueError("thumb_actual_center_rad must be positive")
        object.__setattr__(self, "edge_m", edge)
        object.__setattr__(self, "thumb_actual_center_rad", thumb)

    @property
    def cell_id(self) -> str:
        return (
            f"edge_{self.edge_m * 1000.0:.0f}mm_"
            f"thumb_actual_{self.thumb_actual_center_rad:.2f}rad"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_index": self.cell_index,
            "cell_id": self.cell_id,
            "edge_m": self.edge_m,
            "thumb_actual_center_rad": self.thumb_actual_center_rad,
        }


@dataclass(frozen=True, slots=True)
class ActualContactStaticThresholds:
    """Hard static promotion thresholds, all expressed in SI units."""

    signed_gap_m: tuple[float, float] = (-0.0005, 0.00025)
    minimum_normal_alignment: float = 0.95
    minimum_edge_margin_m: float = 0.0005
    maximum_height_spread_m: float = 0.005
    retreat_m: tuple[float, float] = (0.002, 0.004)
    maximum_closure_angle_deg: float = 30.0
    distance_max_m: float = DEFAULT_DISTANCE_MAX_M

    def __post_init__(self) -> None:
        gap = tuple(_finite(value, "signed_gap_m") for value in self.signed_gap_m)
        retreat = tuple(_finite(value, "retreat_m") for value in self.retreat_m)
        if len(gap) != 2 or gap[0] > gap[1]:
            raise ValueError("signed_gap_m must be an ordered pair")
        if len(retreat) != 2 or retreat[0] <= 0.0 or retreat[0] > retreat[1]:
            raise ValueError("retreat_m must be a positive ordered pair")
        alignment = _finite(self.minimum_normal_alignment, "minimum_normal_alignment")
        edge = _finite(self.minimum_edge_margin_m, "minimum_edge_margin_m")
        spread = _finite(self.maximum_height_spread_m, "maximum_height_spread_m")
        angle = _finite(self.maximum_closure_angle_deg, "maximum_closure_angle_deg")
        distance = _finite(self.distance_max_m, "distance_max_m")
        if not 0.0 <= alignment <= 1.0:
            raise ValueError("minimum_normal_alignment must lie within [0, 1]")
        if edge < 0.0 or spread < 0.0:
            raise ValueError("edge margin and height spread must be non-negative")
        if not 0.0 < angle < 180.0:
            raise ValueError("maximum_closure_angle_deg must lie within (0, 180)")
        if distance <= 0.0 or distance <= retreat[1]:
            raise ValueError("distance_max_m must exceed the maximum retreat")
        object.__setattr__(self, "signed_gap_m", gap)
        object.__setattr__(self, "retreat_m", retreat)
        object.__setattr__(self, "minimum_normal_alignment", alignment)
        object.__setattr__(self, "minimum_edge_margin_m", edge)
        object.__setattr__(self, "maximum_height_spread_m", spread)
        object.__setattr__(self, "maximum_closure_angle_deg", angle)
        object.__setattr__(self, "distance_max_m", distance)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ActualContactStaticThresholds":
        definition = _resolve_actual_contact_definition(
            config, "actual-contact static thresholds"
        )
        campaign = definition.actual_contact_grasp_pose_campaign
        if campaign is None:
            raise ValueError("config does not resolve an actual-contact campaign")
        closure = config["closure_alignment"]
        return cls(
            signed_gap_m=campaign.witness_signed_gap_m,
            minimum_normal_alignment=campaign.min_witness_normal_alignment,
            minimum_edge_margin_m=campaign.min_witness_edge_margin_m,
            maximum_height_spread_m=campaign.max_contact_height_spread_m,
            retreat_m=campaign.precontact_retreat_m,
            maximum_closure_angle_deg=float(closure["dynamic_p95_max_angle_deg"]),
        )


@dataclass(frozen=True, slots=True)
class FingerContactEvidence:
    finger: str
    target_face: str
    distal_geom_id: int
    signed_gap_m: float
    normal_alignment: float
    edge_margin_m: float
    cube_point_world_m: tuple[float, float, float]
    distal_point_world_m: tuple[float, float, float]
    cube_outward_normal_world: tuple[float, float, float]
    gap_ok: bool
    normal_ok: bool
    edge_ok: bool

    @property
    def passed(self) -> bool:
        return self.gap_ok and self.normal_ok and self.edge_ok

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


@dataclass(frozen=True, slots=True)
class FingerRetreatEvidence:
    finger: str
    requested_retreat_m: float
    predicted_outward_retreat_m: float
    measured_outward_retreat_m: float
    closure_velocity_world_m_s: tuple[float, float, float]
    closure_angle_deg: float
    inward_speed_m_s: float
    tangent_speed_m_s: float
    joint_delta_rad: tuple[float, ...]
    retreat_ok: bool
    direction_ok: bool

    @property
    def passed(self) -> bool:
        return self.retreat_ok and self.direction_ok

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


@dataclass(frozen=True, slots=True)
class ActualContactStaticResult:
    """Complete direct-pose evidence for one candidate."""

    static_geometry_pass: bool
    nominal_joint_qpos_rad: tuple[float, ...]
    precontact_joint_qpos_rad: tuple[float, ...]
    target_witnesses: tuple[FingerContactEvidence | None, ...]
    retreat_evidence: tuple[FingerRetreatEvidence | None, ...]
    contact_height_spread_m: float
    minimum_active_nondistal_gap_m: float
    precontact_minimum_hand_gap_m: float
    off_target_distal_penetrating_count: int
    missing_target_witness_count: int
    cube_freejoint_qpos_unchanged: bool
    direct_contact_qpos_forward_count: int = 1
    evaluation_mode: str = "direct_actual_contact_qpos"
    # Schema-v11 safety evidence is optional at the data-model boundary so
    # schema-v9/v10 records keep their byte-for-byte serialization contract.
    # The v11 production DLS evaluator requires every field and fails closed
    # when an injected/legacy result does not carry it.
    nominal_minimum_forbidden_hand_gap_m: float | None = None
    nominal_maximum_all_distal_penetration_m: float | None = None
    precontact_geometry_evaluated: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "static_geometry_pass": self.static_geometry_pass,
            "nominal_joint_qpos_rad": list(self.nominal_joint_qpos_rad),
            "precontact_joint_qpos_rad": list(self.precontact_joint_qpos_rad),
            "target_witness": {
                finger: (
                    None if witness is None else witness.as_dict()
                )
                for finger, witness in zip(ACTIVE_FINGERS, self.target_witnesses)
            },
            "retreat_evidence": {
                finger: (None if value is None else value.as_dict())
                for finger, value in zip(ACTIVE_FINGERS, self.retreat_evidence)
            },
            "contact_height_spread_m": self.contact_height_spread_m,
            "minimum_active_nondistal_gap_m": self.minimum_active_nondistal_gap_m,
            "precontact_minimum_hand_gap_m": self.precontact_minimum_hand_gap_m,
            "off_target_distal_penetrating_count": (
                self.off_target_distal_penetrating_count
            ),
            "missing_target_witness_count": self.missing_target_witness_count,
            "cube_freejoint_qpos_unchanged": self.cube_freejoint_qpos_unchanged,
            "direct_contact_qpos_forward_count": self.direct_contact_qpos_forward_count,
            "evaluation_mode": self.evaluation_mode,
        }
        if self.nominal_minimum_forbidden_hand_gap_m is not None:
            result["nominal_minimum_forbidden_hand_gap_m"] = (
                self.nominal_minimum_forbidden_hand_gap_m
            )
        if self.nominal_maximum_all_distal_penetration_m is not None:
            result["nominal_maximum_all_distal_penetration_m"] = (
                self.nominal_maximum_all_distal_penetration_m
            )
        if self.precontact_geometry_evaluated is not None:
            result["precontact_geometry_evaluated"] = (
                self.precontact_geometry_evaluated
            )
        return result


def actual_contact_search_cells(
    template: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Return registered partitions in deterministic edge-major order.

    Schema v9/v10 partitions only by cube edge and measured thumb band.  A
    schema-v11 relative-wrist campaign adds the registered clockwise orbit as
    a third, explicit stratum dimension.  Keeping the dispatch here makes the
    streaming/resume machinery operate on 700 real strata without changing
    any legacy candidate IDs or numerical paths.
    """

    definition = _resolve_actual_contact_definition(
        template, "actual-contact search cells"
    )
    campaign = definition.actual_contact_grasp_pose_campaign
    if campaign is None:  # Kept explicit for static type narrowing.
        raise ValueError("template has no actual-contact campaign")
    relative = definition.relative_wrist_pose_search
    if relative is not None:
        from .relative_wrist_pose_search import build_relative_wrist_strata

        return build_relative_wrist_strata(
            campaign.edges_m,
            campaign.thumb_actual_centers_rad,
            relative.clockwise_orbit_deg,
        )
    return tuple(
        ActualContactSearchCell(index, edge, thumb)
        for index, (edge, thumb) in enumerate(
            (edge, thumb)
            for edge in campaign.edges_m
            for thumb in campaign.thumb_actual_centers_rad
        )
    )


def _cube_world_position(config: Mapping[str, Any]) -> np.ndarray:
    cube = config["cube"]
    edge = _finite(cube["edge_m"], "cube.edge_m")
    rotation = rpy_degrees_to_rotation_matrix(cube.get("rpy_deg", (0.0, 0.0, 0.0)))
    center = _vector(cube["center_xy_m"], 2, "cube.center_xy_m")
    z = (
        _finite(config["scene"]["support_top_z_m"], "support_top_z_m")
        + cube_vertical_half_extent_m(edge, rotation)
        + _finite(cube.get("z_offset_m", 0.0), "cube.z_offset_m")
    )
    return np.asarray((center[0], center[1], z), dtype=np.float64)


def _cube_in_root(config: Mapping[str, Any]) -> np.ndarray:
    rotation = rpy_degrees_to_rotation_matrix(config["hand_pose"]["rpy_deg"])
    root = _vector(config["hand_pose"]["translation_m"], 3, "hand translation")
    return rotation.T @ (_cube_world_position(config) - root)


def _source_config(value: Mapping[str, Any]) -> Mapping[str, Any]:
    result = value.get("config", value)
    if not isinstance(result, Mapping):
        raise ValueError("source pose must be a config or contain config")
    return result


def _source_nominal(config: Mapping[str, Any], template: Mapping[str, Any]) -> dict[str, float]:
    grasp_pose = config.get("grasp_pose")
    if isinstance(grasp_pose, Mapping) and isinstance(
        grasp_pose.get("nominal_joint_qpos_rad"), Mapping
    ):
        return _target_mapping(
            grasp_pose["nominal_joint_qpos_rad"], "source nominal grasp qpos"
        )
    control = config.get("control", {})
    for key in ("contact_preload_targets_rad", "grasp_targets_rad"):
        if isinstance(control.get(key), Mapping):
            return _target_mapping(control[key], f"source control.{key}")
    return _target_mapping(
        template["grasp_pose"]["nominal_joint_qpos_rad"], "template nominal grasp qpos"
    )


def _local_pose_is_registered(definition: Any, local: np.ndarray) -> bool:
    bounds = definition.search_bounds.cube_position_in_root_m
    if any(
        not bounds[axis][0] - _NUMERIC_TOLERANCE
        <= float(local[index])
        <= bounds[axis][1] + _NUMERIC_TOLERANCE
        for index, axis in enumerate(("x", "y", "z"))
    ):
        return False
    constraints = definition.far_hand_pose_constraints
    if constraints is None:
        return True
    distance = float(np.linalg.norm(local))
    return bool(
        constraints.root_cube_distance_m[0] - _NUMERIC_TOLERANCE
        <= distance
        <= constraints.root_cube_distance_m[1] + _NUMERIC_TOLERANCE
    )


def _sample_registered_local_pose(
    definition: Any,
    source: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    allow_exact_anchor: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Perturb one seed without leaving its signed-tilt/distance envelope."""

    from .far_hand_fingertip import (
        _feasible_tilt_for_roll_deg,
        root_pitch_for_finger_down_tilt_deg,
    )

    constraints = definition.far_hand_pose_constraints
    if constraints is None:
        raise ValueError("schema-v9 static search requires far-hand constraints")
    source_rpy = _vector(source["hand_pose"]["rpy_deg"], 3, "source hand RPY")
    source_local = _cube_in_root(source)
    try:
        source_tilt = float(
            resolved_pose_constraint_values(dict(source))["finger_down_tilt_deg"]
        )
    except (KeyError, ValueError):
        source_tilt = float(sum(constraints.finger_down_tilt_deg) * 0.5)
    exact_rpy_ok = bool(
        definition.search_bounds.hand_roll_deg[0] - _NUMERIC_TOLERANCE
        <= source_rpy[0]
        <= definition.search_bounds.hand_roll_deg[1] + _NUMERIC_TOLERANCE
        and definition.search_bounds.hand_yaw_deg[0] - _NUMERIC_TOLERANCE
        <= source_rpy[2]
        <= definition.search_bounds.hand_yaw_deg[1] + _NUMERIC_TOLERANCE
        and constraints.finger_down_tilt_deg[0] - _NUMERIC_TOLERANCE
        <= source_tilt
        <= constraints.finger_down_tilt_deg[1] + _NUMERIC_TOLERANCE
    )
    if allow_exact_anchor and exact_rpy_ok and _local_pose_is_registered(
        definition, source_local
    ):
        return source_rpy, source_local, "source_contact_anchor"
    for _ in range(128):
        roll = float(
            np.clip(
                source_rpy[0] + rng.uniform(-1.0, 1.0),
                *definition.search_bounds.hand_roll_deg,
            )
        )
        yaw = float(
            np.clip(
                source_rpy[2] + rng.uniform(-1.0, 1.0),
                *definition.search_bounds.hand_yaw_deg,
            )
        )
        tilt = float(
            np.clip(
                source_tilt + rng.uniform(-1.5, 1.5),
                *constraints.finger_down_tilt_deg,
            )
        )
        tilt = _feasible_tilt_for_roll_deg(
            tilt, roll_deg=roll, constraints=constraints
        )
        hand_rpy = np.asarray(
            (roll, root_pitch_for_finger_down_tilt_deg(roll, tilt), yaw),
            dtype=np.float64,
        )
        local = source_local + rng.uniform(-0.003, 0.003, size=3)
        bounds = definition.search_bounds.cube_position_in_root_m
        local = np.clip(
            local,
            [bounds[axis][0] for axis in ("x", "y", "z")],
            [bounds[axis][1] for axis in ("x", "y", "z")],
        )
        if _local_pose_is_registered(definition, local):
            return hand_rpy, local, "source_contact_local"
    raise ValueError("source pose has no feasible schema-v9 local perturbation")


def generate_actual_contact_pose_candidates(
    template: Mapping[str, Any],
    source_poses: Sequence[Mapping[str, Any]],
    cell: Any,
    *,
    count: int,
    start_index: int = 0,
    seed: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Generate prefix-stable direct contact-pose candidates.

    The cube world pose is fixed by the cell.  Only the hand root and nominal
    contact qpos are sampled.  ``precontact_targets_rad`` is deliberately not
    sampled here; it is replaced only after a Jacobian retreat has passed.
    """

    candidate_count = _positive_int(count, "count")
    if (
        not isinstance(start_index, int)
        or isinstance(start_index, bool)
        or start_index < 0
        or start_index + candidate_count > _CELL_STRIDE
    ):
        raise ValueError("start_index + count must stay within the cell stride")
    if not source_poses:
        raise ValueError("source_poses must not be empty")
    definition = _resolve_actual_contact_definition(
        template, "actual-contact static candidate generation"
    )
    campaign = definition.actual_contact_grasp_pose_campaign
    if campaign is None:
        raise ValueError("template has no actual-contact campaign")
    if cell not in actual_contact_search_cells(template):
        raise ValueError("cell is not registered by the actual-contact campaign")
    resolved_seed = campaign.seed if seed is None else int(seed)
    if resolved_seed < 0:
        raise ValueError("seed must be non-negative")
    sources = tuple(_source_config(value) for value in source_poses)
    new_cube = copy.deepcopy(dict(template["cube"]))
    new_cube.update(
        {
            "edge_m": cell.edge_m,
            "mass_kg": campaign.fixed_mass_kg,
            "friction": campaign.friction,
            "center_xy_m": list(campaign.cube_center_xy_m),
            "rpy_deg": [0.0, 0.0, campaign.cube_yaw_deg],
            "z_offset_m": 0.0,
        }
    )
    fixed_probe = copy.deepcopy(dict(template))
    fixed_probe["cube"] = copy.deepcopy(new_cube)
    new_cube_world = _cube_world_position(fixed_probe)
    bounds = definition.search_bounds.actuator_targets_rad
    relative_search = definition.relative_wrist_pose_search
    results: list[dict[str, Any]] = []
    for local_index in range(start_index, start_index + candidate_count):
        if relative_search is None:
            source_index = local_index % len(sources)
            occurrence = local_index // len(sources)
        else:
            # The authenticated source manifest stores the declared primary
            # candidate first.  A ten-sample deterministic cycle realizes the
            # registered 70/30 mix exactly, while round-robin selection keeps
            # all certified 82--84 mm neighbors represented in the 30% tail.
            if len(sources) < 2:
                raise ValueError(
                    "relative-wrist search requires a primary and at least "
                    "one certified neighboring source"
                )
            cycle, slot = divmod(local_index, 10)
            if slot < 7:
                source_index = 0
            else:
                source_index = 1 + ((cycle * 3 + slot - 7) % (len(sources) - 1))
            occurrence = cycle
        source = sources[source_index]
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    resolved_seed,
                    cell.cell_index,
                    local_index,
                    int(round(cell.edge_m * 1e6)),
                    int(round(cell.thumb_actual_center_rad * 1e6)),
                    9_100_001,
                ]
            )
        )
        nominal = _source_nominal(source, template)
        for name in ACTIVE_ACTUATORS:
            if name == THUMB_BEND_ACTUATOR:
                nominal[name] = cell.thumb_actual_center_rad
            elif occurrence > 0:
                nominal[name] = float(
                    np.clip(nominal[name] + rng.uniform(-0.10, 0.10), *bounds[name])
                )
            else:
                nominal[name] = float(np.clip(nominal[name], *bounds[name]))
        config = copy.deepcopy(dict(template))
        config["cube"] = copy.deepcopy(new_cube)
        config["grasp_pose"]["nominal_joint_qpos_rad"] = dict(nominal)
        relative_metadata: dict[str, Any] | None = None
        if relative_search is None:
            hand_rpy, local, mode = _sample_registered_local_pose(
                definition,
                source,
                rng,
                allow_exact_anchor=occurrence == 0,
            )
            root_rotation = rpy_degrees_to_rotation_matrix(hand_rpy)
            config["hand_pose"] = {
                "rpy_deg": hand_rpy.tolist(),
                "translation_m": (new_cube_world - root_rotation @ local).tolist(),
            }
        else:
            from ..relative_wrist_pose import (
                rotation_matrix_to_rpy_degrees,
                transform_relative_wrist_pose,
            )
            from .relative_wrist_pose_search import (
                NON_THUMB_ACTUATORS,
                RelativeWristPoseSearchPolicy,
                RelativeWristVariables,
                materialize_relative_wrist_candidate,
                relative_wrist_boundary_violations,
            )

            policy = RelativeWristPoseSearchPolicy.from_config(config)
            source_cube_position = _cube_world_position(source)
            source_cube_rotation = rpy_degrees_to_rotation_matrix(
                source["cube"].get("rpy_deg", (0.0, 0.0, 0.0))
            )
            target_cube_rotation = rpy_degrees_to_rotation_matrix(
                new_cube.get("rpy_deg", (0.0, 0.0, 0.0))
            )
            source_root_rotation = rpy_degrees_to_rotation_matrix(
                source["hand_pose"]["rpy_deg"]
            )
            transferred = transform_relative_wrist_pose(
                source_cube_world_position_m=source_cube_position,
                source_cube_world_rotation=source_cube_rotation,
                source_root_world_position_m=source["hand_pose"]["translation_m"],
                source_root_world_rotation=source_root_rotation,
                target_cube_world_position_m=new_cube_world,
                target_cube_world_rotation=target_cube_rotation,
            )
            anchor_hand_pose = {
                "translation_m": list(transferred.root_world_position_m),
                "rpy_deg": rotation_matrix_to_rpy_degrees(
                    transferred.root_world_rotation,
                    reference_rpy_deg=source["hand_pose"]["rpy_deg"],
                ).tolist(),
            }
            config["hand_pose"] = copy.deepcopy(anchor_hand_pose)
            config.setdefault("candidate_metadata", {})[
                "relative_wrist_pose_search"
            ] = {"anchor_hand_pose": copy.deepcopy(anchor_hand_pose)}

            joint_bounds = {
                name: tuple(float(value) for value in bounds[name])
                for name in NON_THUMB_ACTUATORS
            }
            selected_variables: RelativeWristVariables | None = None
            selected_candidate: dict[str, Any] | None = None
            boundary_hits: dict[str, int] = defaultdict(int)
            for attempt in range(128):
                exact = local_index == 0 and attempt == 0
                delta = np.zeros(3, dtype=np.float64)
                rotvec_deg = np.zeros(3, dtype=np.float64)
                if not exact:
                    delta = np.asarray(
                        [
                            rng.uniform(*policy.root_delta_cube_m[axis])
                            for axis in ("x", "y", "z")
                        ],
                        dtype=np.float64,
                    )
                    for _ in range(64):
                        rotvec_deg = np.asarray(
                            [
                                rng.uniform(*policy.wrist_local_rotvec_deg[axis])
                                for axis in ("x", "y", "z")
                            ],
                            dtype=np.float64,
                        )
                        if (
                            np.linalg.norm(rotvec_deg)
                            <= policy.max_wrist_local_rotvec_norm_deg
                            + _NUMERIC_TOLERANCE
                        ):
                            break
                    else:  # pragma: no cover - rejection probability is tiny.
                        raise RuntimeError("could not sample a bounded wrist rotvec")
                variables = RelativeWristVariables(
                    tuple(float(nominal[name]) for name in NON_THUMB_ACTUATORS),
                    tuple(float(value) for value in delta),
                    tuple(float(value) for value in np.radians(rotvec_deg)),
                )
                violations = relative_wrist_boundary_violations(
                    config,
                    variables,
                    policy,
                    clockwise_orbit_deg=float(cell.clockwise_orbit_deg),
                    joint_bounds=joint_bounds,
                )
                if violations:
                    for reason in violations:
                        boundary_hits[reason] += 1
                    continue
                selected_variables = variables
                selected_candidate = materialize_relative_wrist_candidate(
                    config,
                    variables,
                    clockwise_orbit_deg=float(cell.clockwise_orbit_deg),
                )
                break
            if selected_variables is None or selected_candidate is None:
                raise RuntimeError(
                    "relative-wrist stratum has no feasible sampled pose: "
                    f"{cell.cell_id}; boundary_hits={dict(sorted(boundary_hits.items()))}"
                )
            config = selected_candidate
            relative_metadata = copy.deepcopy(
                config["candidate_metadata"]["relative_wrist_pose_search"]
            )
            relative_metadata.update(
                {
                    "source_index": source_index,
                    "source_role": (
                        "primary_anchor"
                        if source_index == 0
                        else "certified_neighbor"
                    ),
                    "sampling_fraction": (
                        relative_search.primary_anchor_fraction
                        if source_index == 0
                        else relative_search.certified_neighbor_fraction
                    ),
                    "boundary_rejection_counts": dict(sorted(boundary_hits.items())),
                }
            )
            config["candidate_metadata"][
                "relative_wrist_pose_search"
            ] = copy.deepcopy(relative_metadata)
            mode = (
                "primary_anchor_relative_6d"
                if source_index == 0
                else "certified_neighbor_relative_6d"
            )
        config["grasp_pose"]["nominal_joint_qpos_rad"] = dict(nominal)
        # Preload is a controller quantity.  Starting it at the measured-pose
        # proposal is neutral and, crucially, does not make it static evidence.
        preload = dict(nominal)
        preload[THUMB_BEND_ACTUATOR] = float(
            np.clip(
                max(
                    nominal[THUMB_BEND_ACTUATOR],
                    float(config["control"]["contact_preload_targets_rad"][THUMB_BEND_ACTUATOR]),
                ),
                *bounds[THUMB_BEND_ACTUATOR],
            )
        )
        config["control"]["contact_preload_targets_rad"] = preload
        config["control"]["manipulation_delta_rad"] = {
            name: 0.0 for name in ACTIVE_ACTUATORS
        }
        candidate_id = _CANDIDATE_BASE + cell.cell_index * _CELL_STRIDE + local_index
        prior_metadata = copy.deepcopy(config.get("candidate_metadata", {}))
        config["candidate_metadata"] = {
            **prior_metadata,
            "campaign_kind": CAMPAIGN_KIND,
            "stage": "direct_actual_contact_static",
            "candidate_id": candidate_id,
            "cell_index": cell.cell_index,
            "cell_id": cell.cell_id,
            "local_index": local_index,
            "seed": resolved_seed,
            "source_index": source_index,
            "sample_mode": mode,
            "sampled_fields": [
                "hand_pose",
                "grasp_pose.nominal_joint_qpos_rad",
            ],
            "cube_pose_sampled": False,
            "free_cube_pose_reset_during_scan": False,
            "static_filter_is_success_evidence": False,
            "contact_pose_interpolation_used": False,
        }
        if relative_metadata is not None:
            config["candidate_metadata"]["sampled_fields"] = [
                "hand_pose.cube_frame_translation",
                "hand_pose.hand_local_rotation_vector",
                "grasp_pose.nominal_joint_qpos_rad",
            ]
            config["candidate_metadata"][
                "relative_wrist_pose_search"
            ] = copy.deepcopy(relative_metadata)
        results.append(
            {
                "candidate_id": candidate_id,
                **cell.as_dict(),
                "local_index": local_index,
                "grasp_pose_id": grasp_pose_id(config),
                "controller_id": controller_id(config),
                "candidate_sha256": canonical_sha256(config),
                "config": config,
            }
        )
    return tuple(results)


def _collidable_hand_geom_ids(model: mujoco.MjModel, info: Any) -> tuple[int, ...]:
    return tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in info.hand_body_parts
        and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    )


def _forbidden_hand_geom_ids(
    model: mujoco.MjModel, info: Any
) -> tuple[int, ...]:
    """Return collidable palm, ring and pinky geoms for v11 safety proof."""

    forbidden_parts = {"palm", "ring", "pinky"}
    return tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if info.hand_body_parts.get(int(model.geom_bodyid[geom_id]))
        in forbidden_parts
        and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    )


def _minimum_geom_gap(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
    geom_ids: Sequence[int],
    distance_max_m: float,
) -> float:
    if not geom_ids:
        return float(distance_max_m)
    return min(
        float(
            mujoco.mj_geomDistance(
                model,
                data,
                int(cube_geom_id),
                int(geom_id),
                float(distance_max_m),
                None,
            )
        )
        for geom_id in geom_ids
    )


def _maximum_geom_penetration_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
    geom_ids: Sequence[int],
    distance_max_m: float,
) -> float:
    """Return the deepest penetration over all requested collision geoms."""

    return max(
        0.0,
        -_minimum_geom_gap(
            model, data, cube_geom_id, geom_ids, distance_max_m
        ),
    )


def _joint_limited_qpos(
    model: mujoco.MjModel,
    info: Any,
    active_ids: np.ndarray,
    proposed: np.ndarray,
) -> np.ndarray:
    result = proposed.copy()
    for local_index, actuator_id in enumerate(active_ids):
        joint_id = int(model.actuator_trnid[int(actuator_id), 0])
        if bool(model.jnt_limited[joint_id]):
            result[local_index] = float(
                np.clip(result[local_index], *model.jnt_range[joint_id])
            )
    return result


def _witness_evidence(
    witness: Any,
    *,
    finger: str,
    face: Face,
    cube_rotation: np.ndarray,
    thresholds: ActualContactStaticThresholds,
) -> FingerContactEvidence:
    outward = cube_rotation @ face.outward_normal
    gap = float(witness.signed_distance_m)
    alignment = float(witness.classification.normal_alignment)
    edge = float(witness.classification.edge_clearance_m)
    return FingerContactEvidence(
        finger=finger,
        target_face=face.name,
        distal_geom_id=int(witness.distal_geom_id),
        signed_gap_m=gap,
        normal_alignment=alignment,
        edge_margin_m=edge,
        cube_point_world_m=tuple(float(value) for value in witness.cube_point_world_m),
        distal_point_world_m=tuple(float(value) for value in witness.distal_point_world_m),
        cube_outward_normal_world=tuple(float(value) for value in outward),
        gap_ok=bool(
            thresholds.signed_gap_m[0] - _NUMERIC_TOLERANCE
            <= gap
            <= thresholds.signed_gap_m[1] + _NUMERIC_TOLERANCE
        ),
        normal_ok=alignment + _NUMERIC_TOLERANCE >= thresholds.minimum_normal_alignment,
        edge_ok=edge + _NUMERIC_TOLERANCE >= thresholds.minimum_edge_margin_m,
    )


def _solve_precontact_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: Any,
    active_ids: np.ndarray,
    contact_qpos: np.ndarray,
    witnesses: Mapping[str, FingerContactEvidence],
    requested_retreat_m: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, tuple[float, ...]]]:
    """One damped point-Jacobian inverse per independent finger group."""

    proposed = contact_qpos.copy()
    predicted: dict[str, float] = {}
    deltas: dict[str, tuple[float, ...]] = {}
    active_index = {name: index for index, name in enumerate(ACTIVE_ACTUATORS)}
    for finger in ACTIVE_FINGERS:
        evidence = witnesses[finger]
        local_indices = np.asarray(
            [active_index[name] for name in FINGER_ACTUATORS[finger]], dtype=np.int64
        )
        actuator_ids = active_ids[local_indices]
        dof_addresses = np.asarray(info.actuator_dof_adrs[actuator_ids], dtype=np.int64)
        jacobian = np.zeros((3, model.nv), dtype=np.float64)
        body_id = int(model.geom_bodyid[evidence.distal_geom_id])
        point = np.asarray(evidence.distal_point_world_m, dtype=np.float64)
        mujoco.mj_jac(model, data, jacobian, None, point, body_id)
        matrix = jacobian[:, dof_addresses]
        outward = np.asarray(evidence.cube_outward_normal_world, dtype=np.float64)
        desired = float(requested_retreat_m) * outward
        # SVD/pseudoinverse gives the minimum-norm group displacement and is
        # deterministic even for the two-DoF middle finger.
        delta = np.linalg.pinv(matrix, rcond=1e-9) @ desired
        proposed[local_indices] += delta
        deltas[finger] = tuple(float(value) for value in delta)
        predicted[finger] = float(outward @ (matrix @ delta))
    limited = _joint_limited_qpos(model, info, active_ids, proposed)
    return limited, predicted, deltas


def _refine_precontact_retreat(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: Any,
    active_ids: np.ndarray,
    contact_qpos: np.ndarray,
    proposed_qpos: np.ndarray,
    witnesses: Mapping[str, FingerContactEvidence],
    faces: Mapping[str, Face],
    distal: Mapping[str, Sequence[int]],
    thresholds: ActualContactStaticThresholds,
) -> np.ndarray:
    """Line-search each independent finger group through real geometry.

    The point-Jacobian inverse is a local linear estimate.  Thumb kinematics
    are nonlinear enough that a nominal 3 mm step can produce almost no real
    separation (or overshoot).  This deterministic refinement scales only the
    already-computed outward joint direction and chooses a real witness gap in
    the configured 2--4 mm band; it never changes the contact pose itself.
    """

    active_index = {name: index for index, name in enumerate(ACTIVE_ACTUATORS)}
    refined = contact_qpos.copy()
    qpos_addresses = np.asarray(info.actuator_qpos_adrs[active_ids], dtype=np.int64)
    target = 0.5 * (thresholds.retreat_m[0] + thresholds.retreat_m[1])
    # Include small and large scales; sorting by true gap error below removes
    # any dependence on this enumeration other than deterministic tie-breaks.
    scales = np.concatenate(
        (
            np.linspace(0.25, 2.0, 15),
            np.linspace(2.25, 5.0, 12),
        )
    )
    for finger in ACTIVE_FINGERS:
        local_indices = np.asarray(
            [active_index[name] for name in FINGER_ACTUATORS[finger]],
            dtype=np.int64,
        )
        base_delta = proposed_qpos[local_indices] - contact_qpos[local_indices]
        choices: list[tuple[tuple[float, ...], np.ndarray]] = []
        for scale in scales:
            test = contact_qpos.copy()
            test[local_indices] += float(scale) * base_delta
            test = _joint_limited_qpos(model, info, active_ids, test)
            data.qpos[qpos_addresses] = test
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            witness = nearest_distal_target_witness(
                model,
                data,
                cube_geom_id=info.cube_geom_id,
                distal_geom_ids=distal[finger],
                target_face=faces[finger],
                distance_max_m=thresholds.distance_max_m,
                thresholds=_PERMISSIVE_FACE_THRESHOLDS,
            )
            measured = (
                -thresholds.distance_max_m
                if witness is None
                else float(witness.signed_distance_m)
                - witnesses[finger].signed_gap_m
            )
            in_band = (
                thresholds.retreat_m[0] - _NUMERIC_TOLERANCE
                <= measured
                <= thresholds.retreat_m[1] + _NUMERIC_TOLERANCE
            )
            choices.append(
                (
                    (
                        0.0 if in_band else 1.0,
                        abs(measured - target),
                        abs(float(scale) - 1.0),
                    ),
                    test[local_indices].copy(),
                )
            )
        choices.sort(key=lambda value: value[0])
        refined[local_indices] = choices[0][1]
    data.qpos[qpos_addresses] = contact_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return refined


def evaluate_direct_actual_contact_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: Any,
    config: Mapping[str, Any],
    *,
    nominal_joint_qpos_rad: Mapping[str, Any] | None = None,
    thresholds: ActualContactStaticThresholds | None = None,
    requested_retreat_m: float | None = None,
    close_duration_s: float = 1.0,
) -> ActualContactStaticResult:
    """Evaluate one direct actual-qpos proposal and derive its pre-contact pose.

    All touched model/data state is restored before return.  The cube qpos is
    snapshotted only for an invariance assertion; it is never overwritten.
    """

    if int(config.get("schema_version", 0)) not in ACTUAL_CONTACT_SCHEMA_VERSIONS:
        raise ValueError(
            "direct actual-contact screening requires an actual-contact schema"
        )
    limits = thresholds or ActualContactStaticThresholds.from_config(config)
    requested = (
        0.5 * (limits.retreat_m[0] + limits.retreat_m[1])
        if requested_retreat_m is None
        else _finite(requested_retreat_m, "requested_retreat_m")
    )
    if not limits.retreat_m[0] <= requested <= limits.retreat_m[1]:
        raise ValueError("requested_retreat_m must lie within configured retreat_m")
    close_duration = _finite(close_duration_s, "close_duration_s")
    if close_duration <= 0.0:
        raise ValueError("close_duration_s must be positive")
    nominal_mapping = _target_mapping(
        config["grasp_pose"]["nominal_joint_qpos_rad"]
        if nominal_joint_qpos_rad is None
        else nominal_joint_qpos_rad,
        "nominal_joint_qpos_rad",
    )
    nominal = np.asarray(
        [nominal_mapping[name] for name in ACTIVE_ACTUATORS], dtype=np.float64
    )
    active_ids = np.asarray(
        [model.actuator(name).id for name in ACTIVE_ACTUATORS], dtype=np.int64
    )
    qpos_addresses = np.asarray(info.actuator_qpos_adrs[active_ids], dtype=np.int64)
    faces = _target_faces(config)
    distal = distal_collision_geom_ids(model, info.distal_weld_ids)
    nondistal = active_nondistal_collision_geom_ids(
        model, info.hand_body_parts, info.distal_weld_ids
    )
    all_hand = _collidable_hand_geom_ids(model, info)
    schema_v11_safety = int(config.get("schema_version", 0)) == 11
    # Do not even issue these additional geometry queries for v9/v10: their
    # archived static metrics and numerical path are part of the compatibility
    # contract.  A v11 result, in contrast, must carry all three proof fields.
    forbidden_hand = (
        _forbidden_hand_geom_ids(model, info) if schema_v11_safety else ()
    )
    all_distal = (
        tuple(
            geom_id
            for finger in ACTIVE_FINGERS
            for geom_id in distal[finger]
        )
        if schema_v11_safety
        else ()
    )
    saved_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    saved_qvel = np.asarray(data.qvel, dtype=np.float64).copy()
    saved_ctrl = np.asarray(data.ctrl, dtype=np.float64).copy()
    saved_root_pos = np.asarray(model.body_pos[info.root_body_id], dtype=np.float64).copy()
    saved_root_quat = np.asarray(model.body_quat[info.root_body_id], dtype=np.float64).copy()
    try:
        model.body_pos[info.root_body_id] = _vector(
            config["hand_pose"]["translation_m"], 3, "hand translation"
        )
        model.body_quat[info.root_body_id] = rpy_degrees_to_quaternion(
            config["hand_pose"]["rpy_deg"]
        )
        mujoco.mj_resetData(model, data)
        cube_qpos = np.asarray(
            data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 7],
            dtype=np.float64,
        ).copy()
        data.qpos[qpos_addresses] = nominal
        data.qvel[:] = 0.0
        # This is the only forward evaluation of the proposed contact qpos.
        mujoco.mj_forward(model, data)
        cube_rotation = np.asarray(
            data.geom_xmat[info.cube_geom_id], dtype=np.float64
        ).reshape(3, 3)
        gravity = np.asarray(model.opt.gravity, dtype=np.float64)
        gravity_norm = float(np.linalg.norm(gravity))
        if gravity_norm <= np.finfo(np.float64).eps:
            raise ValueError("actual-contact static search requires non-zero gravity")
        up = -gravity / gravity_norm
        contact_evidence: list[FingerContactEvidence | None] = []
        evidence_by_finger: dict[str, FingerContactEvidence] = {}
        off_target = 0
        heights: list[float] = []
        for finger in ACTIVE_FINGERS:
            face = faces[finger]
            witness = nearest_distal_target_witness(
                model,
                data,
                cube_geom_id=info.cube_geom_id,
                distal_geom_ids=distal[finger],
                target_face=face,
                distance_max_m=limits.distance_max_m,
                thresholds=_PERMISSIVE_FACE_THRESHOLDS,
            )
            if witness is None:
                contact_evidence.append(None)
            else:
                evidence = _witness_evidence(
                    witness,
                    finger=finger,
                    face=face,
                    cube_rotation=cube_rotation,
                    thresholds=limits,
                )
                contact_evidence.append(evidence)
                evidence_by_finger[finger] = evidence
                heights.append(float(up @ np.asarray(evidence.cube_point_world_m)))
            for geom_id in distal[finger]:
                any_witness = geom_distance_witness(
                    model,
                    data,
                    cube_geom_id=info.cube_geom_id,
                    distal_geom_id=int(geom_id),
                    distance_max_m=limits.distance_max_m,
                )
                if (
                    any_witness is not None
                    and any_witness.signed_distance_m < -_NUMERIC_TOLERANCE
                    and any_witness.face is not face
                ):
                    off_target += 1
        missing = len(ACTIVE_FINGERS) - len(evidence_by_finger)
        nondistal_flat = tuple(
            geom_id
            for finger in ACTIVE_FINGERS
            for geom_id in nondistal[finger]
        )
        minimum_nondistal = _minimum_geom_gap(
            model,
            data,
            info.cube_geom_id,
            nondistal_flat,
            limits.distance_max_m,
        )
        nominal_minimum_forbidden_hand_gap: float | None = None
        nominal_maximum_all_distal_penetration: float | None = None
        precontact_geometry_evaluated: bool | None = None
        if schema_v11_safety:
            nominal_minimum_forbidden_hand_gap = _minimum_geom_gap(
                model,
                data,
                info.cube_geom_id,
                forbidden_hand,
                limits.distance_max_m,
            )
            nominal_maximum_all_distal_penetration = (
                _maximum_geom_penetration_m(
                    model,
                    data,
                    info.cube_geom_id,
                    all_distal,
                    limits.distance_max_m,
                )
            )
            precontact_geometry_evaluated = False
        # Missing evidence uses finite worst-case sentinels so result JSON can
        # remain strict (``allow_nan=False``) without making absence look good.
        height_spread = (
            float(np.ptp(heights)) if missing == 0 else limits.distance_max_m
        )
        precontact = nominal.copy()
        retreat_values: list[FingerRetreatEvidence | None] = [None] * len(ACTIVE_FINGERS)
        precontact_minimum_hand_gap = -limits.distance_max_m
        extended_nominal_geometry_clear = bool(
            not schema_v11_safety
            or (
                nominal_minimum_forbidden_hand_gap is not None
                and nominal_minimum_forbidden_hand_gap >= -_NUMERIC_TOLERANCE
                and nominal_maximum_all_distal_penetration is not None
                and nominal_maximum_all_distal_penetration
                <= 0.002 + _NUMERIC_TOLERANCE
            )
        )
        legacy_contact_geometry_ready = bool(
            missing == 0
            and all(
                value is not None and value.passed
                for value in contact_evidence
            )
            and height_spread
            <= limits.maximum_height_spread_m + _NUMERIC_TOLERANCE
            and minimum_nondistal >= -_NUMERIC_TOLERANCE
            and off_target == 0
            and extended_nominal_geometry_clear
        )
        # A v11 DLS proposal must be allowed to optimize target gap and height
        # errors.  Those objective terms therefore cannot gate the safety
        # query that supplies its finite differences.  As long as all three
        # physical target witnesses have usable face normal/edge evidence and
        # every independent collision gate is clear, derive the retreat and
        # query the complete precontact hand gap.  Final static acceptance
        # below still requires the strict gap, height and retreat conditions.
        precontact_geometry_ready = bool(
            legacy_contact_geometry_ready
            if not schema_v11_safety
            else (
                missing == 0
                and all(
                    value is not None and value.normal_ok and value.edge_ok
                    for value in contact_evidence
                )
                and minimum_nondistal >= -_NUMERIC_TOLERANCE
                and off_target == 0
                and extended_nominal_geometry_clear
            )
        )
        if precontact_geometry_ready:
            precontact, predicted, joint_deltas = _solve_precontact_qpos(
                model,
                data,
                info,
                active_ids,
                nominal,
                evidence_by_finger,
                requested,
            )
            precontact = _refine_precontact_retreat(
                model,
                data,
                info,
                active_ids,
                nominal,
                precontact,
                evidence_by_finger,
                faces,
                distal,
                limits,
            )
            joint_deltas = {
                finger: tuple(
                    float(
                        precontact[
                            ACTIVE_ACTUATORS.index(name)
                        ]
                        - nominal[ACTIVE_ACTUATORS.index(name)]
                    )
                    for name in FINGER_ACTUATORS[finger]
                )
                for finger in ACTIVE_FINGERS
            }
            data.qpos[qpos_addresses] = precontact
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            if schema_v11_safety:
                precontact_geometry_evaluated = True
            precontact_gaps: dict[str, float] = {}
            for finger in ACTIVE_FINGERS:
                pre_witness = nearest_distal_target_witness(
                    model,
                    data,
                    cube_geom_id=info.cube_geom_id,
                    distal_geom_ids=distal[finger],
                    target_face=faces[finger],
                    distance_max_m=limits.distance_max_m,
                    thresholds=_PERMISSIVE_FACE_THRESHOLDS,
                )
                precontact_gaps[finger] = (
                    -limits.distance_max_m
                    if pre_witness is None
                    else float(pre_witness.signed_distance_m)
                )
            precontact_minimum_hand_gap = _minimum_geom_gap(
                model,
                data,
                info.cube_geom_id,
                all_hand,
                limits.distance_max_m,
            )
            # Return to the exact contact pose only to evaluate the commanded
            # closure Jacobian there.  This is not an interpolation or search.
            data.qpos[qpos_addresses] = nominal
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            qdot = (nominal - precontact) / close_duration
            actuator_velocity = np.zeros(model.nu, dtype=np.float64)
            actuator_velocity[active_ids] = qdot
            for finger_index, finger in enumerate(ACTIVE_FINGERS):
                evidence = evidence_by_finger[finger]
                body_id = int(model.geom_bodyid[evidence.distal_geom_id])
                velocity = point_jacobian_command_velocity(
                    model,
                    data,
                    evidence.distal_point_world_m,
                    body_id,
                    info.actuator_dof_adrs,
                    actuator_velocity,
                )
                outward = np.asarray(evidence.cube_outward_normal_world, dtype=np.float64)
                alignment = closure_alignment_from_velocity(velocity, outward)
                measured = precontact_gaps[finger] - evidence.signed_gap_m
                retreat_ok = bool(
                    limits.retreat_m[0] - _NUMERIC_TOLERANCE
                    <= measured
                    <= limits.retreat_m[1] + _NUMERIC_TOLERANCE
                )
                direction_ok = closure_direction_within_limits(
                    alignment,
                    maximum_angle_deg=limits.maximum_closure_angle_deg,
                    minimum_inward_speed_m_s=0.0,
                    require_positive_inward_speed=True,
                )
                retreat_values[finger_index] = FingerRetreatEvidence(
                    finger=finger,
                    requested_retreat_m=requested,
                    predicted_outward_retreat_m=predicted[finger],
                    measured_outward_retreat_m=measured,
                    closure_velocity_world_m_s=tuple(float(value) for value in velocity),
                    closure_angle_deg=alignment.angle_deg,
                    inward_speed_m_s=alignment.inward_speed_m_s,
                    tangent_speed_m_s=alignment.tangent_speed_m_s,
                    joint_delta_rad=joint_deltas[finger],
                    retreat_ok=retreat_ok,
                    direction_ok=direction_ok,
                )
        cube_unchanged = bool(
            np.array_equal(
                data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 7], cube_qpos
            )
        )
        passed = bool(
            missing == 0
            and all(value is not None and value.passed for value in contact_evidence)
            and all(value is not None and value.passed for value in retreat_values)
            and height_spread <= limits.maximum_height_spread_m + _NUMERIC_TOLERANCE
            and minimum_nondistal >= -_NUMERIC_TOLERANCE
            and precontact_minimum_hand_gap >= -_NUMERIC_TOLERANCE
            and off_target == 0
            and cube_unchanged
            and extended_nominal_geometry_clear
            and (
                not schema_v11_safety
                or precontact_geometry_evaluated is True
            )
        )
        result = ActualContactStaticResult(
            static_geometry_pass=passed,
            nominal_joint_qpos_rad=tuple(float(value) for value in nominal),
            precontact_joint_qpos_rad=tuple(float(value) for value in precontact),
            target_witnesses=tuple(contact_evidence),
            retreat_evidence=tuple(retreat_values),
            contact_height_spread_m=height_spread,
            minimum_active_nondistal_gap_m=minimum_nondistal,
            precontact_minimum_hand_gap_m=precontact_minimum_hand_gap,
            off_target_distal_penetrating_count=off_target,
            missing_target_witness_count=missing,
            cube_freejoint_qpos_unchanged=cube_unchanged,
            nominal_minimum_forbidden_hand_gap_m=(
                nominal_minimum_forbidden_hand_gap
            ),
            nominal_maximum_all_distal_penetration_m=(
                nominal_maximum_all_distal_penetration
            ),
            precontact_geometry_evaluated=precontact_geometry_evaluated,
        )
    finally:
        model.body_pos[info.root_body_id] = saved_root_pos
        model.body_quat[info.root_body_id] = saved_root_quat
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(model, data)
    return result


def apply_precontact_solution(
    config: Mapping[str, Any], result: ActualContactStaticResult
) -> dict[str, Any]:
    """Persist a passed Jacobian retreat without changing the nominal pose."""

    if not result.static_geometry_pass:
        raise ValueError("only a passed static result has a promotable precontact pose")
    resolved = copy.deepcopy(dict(config))
    resolved["control"]["precontact_targets_rad"] = {
        name: float(result.precontact_joint_qpos_rad[index])
        for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    metadata = resolved.setdefault("candidate_metadata", {})
    metadata["precontact_derived_from_contact_point_jacobian"] = True
    metadata["contact_pose_interpolation_used"] = False
    metadata["static_filter_is_success_evidence"] = False
    return resolved


def _static_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = record["static_metrics"]
    witnesses = tuple(
        metrics["target_witness"].get(finger) for finger in ACTIVE_FINGERS
    )
    retreats = tuple(
        metrics["retreat_evidence"].get(finger) for finger in ACTIVE_FINGERS
    )
    gap_low, gap_high = (-0.0005, 0.00025)
    gap_violation = sum(
        0.020
        if value is None
        else max(0.0, gap_low - float(value["signed_gap_m"]))
        + max(0.0, float(value["signed_gap_m"]) - gap_high)
        for value in witnesses
    )
    return (
        not bool(metrics["static_geometry_pass"]),
        int(metrics["missing_target_witness_count"]),
        int(metrics["off_target_distal_penetrating_count"]),
        max(0.0, -float(metrics["minimum_active_nondistal_gap_m"])),
        max(0.0, -float(metrics["precontact_minimum_hand_gap_m"])),
        gap_violation,
        max(
            (
                180.0 if value is None else float(value["closure_angle_deg"])
                for value in retreats
            ),
            default=180.0,
        ),
        float(metrics["contact_height_spread_m"]),
        str(record["grasp_pose_id"]),
        int(record["candidate_id"]),
    )


def screen_actual_contact_pose_cell(
    candidates: Sequence[Mapping[str, Any]], *, top_k: int = 4
) -> dict[str, Any]:
    """Directly screen a same-cube cell with one compiled MuJoCo model."""

    retain = _positive_int(top_k, "top_k")
    if not candidates:
        raise ValueError("candidates must not be empty")
    cells = {int(record["cell_index"]) for record in candidates}
    cube_hashes = {canonical_sha256(record["config"]["cube"]) for record in candidates}
    if len(cells) != 1 or len(cube_hashes) != 1:
        raise ValueError("screen_actual_contact_pose_cell requires one fixed-cube cell")
    model, info = build_model(copy.deepcopy(dict(candidates[0]["config"])))
    data = mujoco.MjData(model)
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        record = copy.deepcopy(dict(candidate))
        result = evaluate_direct_actual_contact_pose(
            model, data, info, record["config"]
        )
        record["static_pass"] = result.static_geometry_pass
        record["static_metrics"] = result.as_dict()
        if result.static_geometry_pass:
            promoted = apply_precontact_solution(record["config"], result)
            try:
                validate_config(promoted)
            except ValueError as error:
                # A geometric retreat outside the registered precontact
                # controller bounds cannot enter dynamics.  Keep it as a
                # ranked near miss instead of aborting the entire cell.
                record["static_pass"] = False
                record["static_metrics"]["static_geometry_pass"] = False
                record["static_metrics"]["promotion_config_valid"] = False
                record["static_metrics"]["promotion_config_error"] = str(error)
            else:
                record["config"] = promoted
                record["controller_id"] = controller_id(record["config"])
                record["candidate_sha256"] = canonical_sha256(record["config"])
                record["static_metrics"]["promotion_config_valid"] = True
        record["static_rank"] = list(_static_rank(record))
        evaluated.append(record)
    ordered = sorted(evaluated, key=_static_rank)
    retained = tuple(ordered[: min(retain, len(ordered))])
    return {
        "static_result_schema_version": STATIC_RESULT_SCHEMA_VERSION,
        "cell_index": next(iter(cells)),
        "cell_id": str(candidates[0]["cell_id"]),
        "evaluated_count": len(evaluated),
        "static_pass_count": sum(bool(record["static_pass"]) for record in evaluated),
        "retained_count": len(retained),
        "retained": retained,
        "ranking_policy": (
            "hard_pass_missing_offtarget_nondistal_precontact_gap_"
            "target_gap_closure_angle_height_grasp_pose_candidate_id"
        ),
    }


def retain_top_actual_contact_candidates(
    records: Sequence[Mapping[str, Any]], *, top_k: int
) -> dict[int, tuple[dict[str, Any], ...]]:
    """Worker-order-independent per-cell reduction of screened records."""

    limit = _positive_int(top_k, "top_k")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if "static_metrics" not in record:
            raise ValueError("screened record is missing static_metrics")
        grouped[int(record["cell_index"])].append(copy.deepcopy(dict(record)))
    return {
        cell: tuple(sorted(values, key=_static_rank)[:limit])
        for cell, values in sorted(grouped.items())
    }


def _uniform_contact_measurement(
    result: ActualContactStaticResult,
    up: np.ndarray,
) -> np.ndarray | None:
    witnesses = result.target_witnesses
    if any(value is None for value in witnesses):
        return None
    resolved = tuple(value for value in witnesses if value is not None)
    gaps = [float(value.signed_gap_m) for value in resolved]
    heights = [
        float(up @ np.asarray(value.cube_point_world_m, dtype=np.float64))
        for value in resolved
    ]
    # ACTIVE_FINGERS is thumb, index, mid.
    return np.asarray(
        (*gaps, heights[1] - heights[0], heights[2] - heights[0]),
        dtype=np.float64,
    )


def refine_uniform_gap_equal_height_contact_pose(
    record: Mapping[str, Any],
    *,
    maximum_iterations: int = 4,
    target_gap_m: float = _UNIFORM_TARGET_GAP_M,
    refined_candidate_id: int | None = None,
) -> dict[str, Any]:
    """Deterministically balance three real distal witnesses around one pose.

    The thumb bend is held exactly at its registered cell center.  A damped
    five-output inverse solves three witness gaps plus index/thumb and
    middle/thumb contact-height differences using the other seven active
    joint values and the fixed hand root's XYZ.  This is only a geometric
    proposal; the returned candidate must still acquire the free cube in the
    full dynamic controller stage.
    """

    iterations = _positive_int(maximum_iterations, "maximum_iterations")
    target_gap = _finite(target_gap_m, "target_gap_m")
    if not -0.0005 <= target_gap <= 0.00025:
        raise ValueError("target_gap_m must lie within the registered witness band")
    source = copy.deepcopy(dict(record))
    config = copy.deepcopy(dict(source["config"]))
    if not bool(source.get("static_pass", False)):
        raise ValueError("uniform contact refinement requires a static hard pass")
    model, info = build_model(config)
    data = mujoco.MjData(model)
    limits = ActualContactStaticThresholds.from_config(config)
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    up = -gravity / float(np.linalg.norm(gravity))
    active_ids = np.asarray(
        [model.actuator(name).id for name in ACTIVE_ACTUATORS], dtype=np.int64
    )
    nonthumb_indices = tuple(
        index
        for index, name in enumerate(ACTIVE_ACTUATORS)
        if name != THUMB_BEND_ACTUATOR
    )
    thumb_index = ACTIVE_ACTUATORS.index(THUMB_BEND_ACTUATOR)
    thumb_center = float(source["thumb_actual_center_rad"])
    nominal = np.asarray(
        [
            config["grasp_pose"]["nominal_joint_qpos_rad"][name]
            for name in ACTIVE_ACTUATORS
        ],
        dtype=np.float64,
    )
    nominal[thumb_index] = thumb_center
    root = _vector(config["hand_pose"]["translation_m"], 3, "hand translation")
    target = np.asarray((target_gap, target_gap, target_gap, 0.0, 0.0))
    variable_scales = np.asarray((*([0.05] * 7), 0.002, 0.002, 0.002))
    finite_steps = np.asarray((*([2e-4] * 7), 2e-5, 2e-5, 2e-5))

    def materialize(values: np.ndarray, root_xyz: np.ndarray) -> dict[str, Any]:
        candidate = copy.deepcopy(config)
        candidate["hand_pose"]["translation_m"] = root_xyz.tolist()
        candidate["grasp_pose"]["nominal_joint_qpos_rad"] = {
            name: float(values[index])
            for index, name in enumerate(ACTIVE_ACTUATORS)
        }
        return candidate

    def evaluate(values: np.ndarray, root_xyz: np.ndarray) -> tuple[
        ActualContactStaticResult, np.ndarray | None
    ]:
        candidate = materialize(values, root_xyz)
        result = evaluate_direct_actual_contact_pose(
            model,
            data,
            info,
            candidate,
            thresholds=limits,
        )
        return result, _uniform_contact_measurement(result, up)

    current_result, current = evaluate(nominal, root)
    if current is None:
        raise RuntimeError("static hard pass lost its three target witnesses")
    initial = current.copy()
    iteration_records: list[dict[str, Any]] = []
    for iteration in range(iterations):
        residual = target - current
        if float(np.max(np.abs(residual))) <= 5e-7:
            break
        jacobian = np.zeros((5, 10), dtype=np.float64)
        for column in range(10):
            perturbed_qpos = nominal.copy()
            perturbed_root = root.copy()
            if column < 7:
                local_index = nonthumb_indices[column]
                perturbed_qpos[local_index] += finite_steps[column]
                perturbed_qpos = _joint_limited_qpos(
                    model, info, active_ids, perturbed_qpos
                )
                actual_step = perturbed_qpos[local_index] - nominal[local_index]
            else:
                axis = column - 7
                perturbed_root[axis] += finite_steps[column]
                actual_step = finite_steps[column]
            _, measured = evaluate(perturbed_qpos, perturbed_root)
            if measured is None or abs(float(actual_step)) <= _NUMERIC_TOLERANCE:
                continue
            jacobian[:, column] = (measured - current) / float(actual_step)
        scaled = jacobian * variable_scales[np.newaxis, :]
        damping_m = 2.5e-5
        normalized_step = scaled.T @ np.linalg.solve(
            scaled @ scaled.T + damping_m**2 * np.eye(5), residual
        )
        maximum_normalized = float(np.max(np.abs(normalized_step)))
        if maximum_normalized > 1.0:
            normalized_step /= maximum_normalized
        proposed_step = variable_scales * normalized_step
        accepted: tuple[
            float, np.ndarray, np.ndarray, ActualContactStaticResult, np.ndarray
        ] | None = None
        current_score = float(np.linalg.norm(residual))
        for line_scale in (1.0, 0.5, 0.25, 0.125):
            trial_qpos = nominal.copy()
            for column, local_index in enumerate(nonthumb_indices):
                trial_qpos[local_index] += line_scale * proposed_step[column]
            trial_qpos[thumb_index] = thumb_center
            trial_qpos = _joint_limited_qpos(
                model, info, active_ids, trial_qpos
            )
            trial_root = root + line_scale * proposed_step[7:10]
            trial_result, trial = evaluate(trial_qpos, trial_root)
            if trial is None:
                continue
            score = float(np.linalg.norm(target - trial))
            if score + 1e-12 < current_score:
                accepted = (
                    line_scale,
                    trial_qpos,
                    trial_root,
                    trial_result,
                    trial,
                )
                break
        iteration_records.append(
            {
                "iteration": iteration,
                "measurement_before_m": current.tolist(),
                "residual_norm_before_m": current_score,
                "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
                "accepted": accepted is not None,
                "line_scale": None if accepted is None else accepted[0],
            }
        )
        if accepted is None:
            break
        _, nominal, root, current_result, current = accepted

    refined_config = materialize(nominal, root)
    refined_config["control"]["contact_preload_targets_rad"] = {
        name: float(nominal[index])
        for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    final_result, final_measurement = evaluate(nominal, root)
    if final_measurement is None:
        raise RuntimeError("uniform refinement lost its final target witnesses")
    refined_id = (
        int(source["candidate_id"]) + _UNIFORM_REFINE_ID_OFFSET
        if refined_candidate_id is None
        else int(refined_candidate_id)
    )
    if refined_id < 0:
        raise ValueError("refined_candidate_id must be non-negative")
    metadata = refined_config.setdefault("candidate_metadata", {})
    metadata.update(
        {
            "stage": "uniform_gap_equal_height_local_refinement",
            "candidate_id": refined_id,
            "parent_candidate_id": int(source["candidate_id"]),
            "uniform_target_gap_m": target_gap,
            "thumb_bend_fixed_at_cell_center": True,
            "static_filter_is_success_evidence": False,
        }
    )
    static_pass = bool(final_result.static_geometry_pass)
    promotion_error: str | None = None
    if static_pass:
        refined_config = apply_precontact_solution(refined_config, final_result)
        try:
            validate_config(refined_config)
        except ValueError as error:
            static_pass = False
            promotion_error = str(error)
    result = {
        **{key: copy.deepcopy(value) for key, value in source.items() if key != "config"},
        "candidate_id": refined_id,
        "parent_candidate_id": int(source["candidate_id"]),
        "grasp_pose_id": grasp_pose_id(refined_config),
        "controller_id": controller_id(refined_config),
        "candidate_sha256": canonical_sha256(refined_config),
        "config": refined_config,
        "static_pass": static_pass,
        "static_metrics": final_result.as_dict(),
        "uniform_refinement": {
            "method": "damped_least_squares_real_distal_witness",
            "target_gap_m": target_gap,
            "optimized_variables": [
                *(ACTIVE_ACTUATORS[index] for index in nonthumb_indices),
                "hand_pose.translation_m.x",
                "hand_pose.translation_m.y",
                "hand_pose.translation_m.z",
            ],
            "fixed_variable": THUMB_BEND_ACTUATOR,
            "initial_measurement_m": initial.tolist(),
            "final_measurement_m": final_measurement.tolist(),
            "initial_residual_norm_m": float(np.linalg.norm(target - initial)),
            "final_residual_norm_m": float(
                np.linalg.norm(target - final_measurement)
            ),
            "iterations": iteration_records,
            "improved": bool(
                np.linalg.norm(target - final_measurement)
                < np.linalg.norm(target - initial)
            ),
            "promotion_config_error": promotion_error,
        },
    }
    result["static_metrics"]["static_geometry_pass"] = static_pass
    result["static_metrics"]["uniform_refinement"] = copy.deepcopy(
        result["uniform_refinement"]
    )
    result["static_rank"] = list(_static_rank(result))
    return result


# ---------------------------------------------------------------------------
# Resumable production campaign orchestration


@dataclass(frozen=True, slots=True)
class CampaignStageExecution:
    """One stage's in-memory records plus its immutable ledger artifacts."""

    records: tuple[dict[str, Any], ...]
    artifacts: tuple[Path, ...]
    summary: dict[str, Any]


def _evidence_anchor_source_files(
    value: str | Path,
) -> tuple[Path, Path, Path]:
    """Resolve one externally verified grasp directory without guessing files."""

    path = Path(value).expanduser().resolve()
    directory = path if path.is_dir() else path.parent
    config_path = directory / "resolved_config.json"
    result_path = directory / "result.json"
    trace_path = directory / "trace.npz"
    missing = [
        candidate.name
        for candidate in (config_path, result_path, trace_path)
        if not candidate.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"evidence grasp anchor {directory} is missing {', '.join(missing)}"
        )
    return config_path, result_path, trace_path


def _evidence_anchor_source_descriptor(value: str | Path) -> dict[str, Any]:
    config_path, result_path, trace_path = _evidence_anchor_source_files(value)
    return {
        "source_directory": str(config_path.parent),
        "config_sha256": file_sha256(config_path),
        "result_sha256": file_sha256(result_path),
        "trace_sha256": file_sha256(trace_path),
    }


def _load_persisted_evidence_anchor(
    dynamic_root: Path,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    directory = dynamic_root / Path(str(raw["artifact_directory"]))
    config_path = directory / "resolved_config.json"
    result_path = directory / "result.json"
    trace_path = directory / "trace.npz"
    if not all(path.is_file() for path in (config_path, result_path, trace_path)):
        raise RuntimeError(f"evidence grasp anchor disappeared: {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    authenticate_candidate_result_semantic_sha256(payload, source=result_path)
    hashes = payload.get("artifacts", {}).get("sha256", {})
    if hashes.get("resolved_config") != file_sha256(config_path):
        raise RuntimeError(f"evidence anchor config changed: {config_path}")
    if hashes.get("trace") != file_sha256(trace_path):
        raise RuntimeError(f"evidence anchor trace changed: {trace_path}")
    identifier = int(raw["candidate_id"])
    if int(payload.get("candidate_id", -1)) != identifier:
        raise RuntimeError(f"evidence anchor candidate ID changed: {result_path}")
    if payload.get("candidate_sha256") != canonical_sha256(config):
        raise RuntimeError(f"evidence anchor semantic config changed: {config_path}")
    if payload.get("grasp_pose_id") != grasp_pose_id(config):
        raise RuntimeError(f"evidence anchor grasp_pose_id changed: {result_path}")
    if payload.get("controller_id") != controller_id(config):
        raise RuntimeError(f"evidence anchor controller_id changed: {result_path}")
    return {
        **payload,
        "config": config,
        "artifact_directory": str(raw["artifact_directory"]),
        "evidence_anchor": True,
        "evidence_anchor_priority": int(raw["evidence_anchor_priority"]),
        "reused": True,
    }


def _prepare_evidence_grasp_anchors(
    output_dir: Path,
    paths: Sequence[str | Path],
) -> CampaignStageExecution:
    """Authenticate and import externally reproduced grasp successes.

    Imported grasps do not bypass any manipulation acceptance condition.  They
    merely skip rediscovering an already fully simulated 250 ms grasp window;
    every manipulation candidate still starts from the configured no-contact
    reset and is rerun through the ordinary schema-v9 evaluator.
    """

    from .actual_contact_grasp_pose_dynamic import dynamic_grasp_rank_evidence
    from .actual_contact_manipulation import validate_grasp_success_source

    dynamic_root = output_dir / "dynamic"
    report_path = output_dir / "evidence_anchors" / "source_bundle.json"
    supplied = tuple(paths)
    supplied_descriptors = tuple(
        _evidence_anchor_source_descriptor(value) for value in supplied
    )
    supplied_sha = canonical_sha256(supplied_descriptors)
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("source_bundle_sha256") != supplied_sha:
            raise RuntimeError("evidence grasp anchor inputs changed on resume")
        records = tuple(
            _load_persisted_evidence_anchor(dynamic_root, raw)
            for raw in report.get("anchors", ())
        )
        artifacts = [report_path]
        for record in records:
            artifacts.extend(
                _dynamic_artifact_paths(record, dynamic_root)
            )
        return CampaignStageExecution(
            records=records,
            artifacts=tuple(artifacts),
            summary={
                "evidence_grasp_anchor_count": len(records),
                "source_bundle_sha256": report["source_bundle_sha256"],
            },
        )
    if not supplied:
        report = {
            "actual_contact_evidence_anchor_bundle_schema_version": 1,
            "complete": True,
            "source_bundle_sha256": supplied_sha,
            "evidence_grasp_anchor_count": 0,
            "selection_policy": "authenticated_anchors_before_discovered_grasps",
            "anchors": [],
        }
        _write_or_verify_json(report_path, report)
        return CampaignStageExecution(
            records=(),
            artifacts=(report_path,),
            summary={
                "evidence_grasp_anchor_count": 0,
                "source_bundle_sha256": supplied_sha,
            },
        )

    records: list[dict[str, Any]] = []
    report_records: list[dict[str, Any]] = []
    seen_evidence: set[str] = set()
    for priority, value in enumerate(supplied):
        config_path, source_result_path, trace_path = _evidence_anchor_source_files(
            value
        )
        descriptor = supplied_descriptors[priority]
        evidence_sha = canonical_sha256(descriptor)
        if evidence_sha in seen_evidence:
            raise ValueError("duplicate evidence grasp anchor")
        seen_evidence.add(evidence_sha)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_config(config)
        source_result = json.loads(
            source_result_path.read_text(encoding="utf-8")
        )
        validate_grasp_success_source(config, trace_path, source_result)
        summary_value = source_result.get("summary", source_result)
        if not isinstance(summary_value, Mapping):
            raise ValueError("evidence grasp anchor has no simulation summary")
        summary = copy.deepcopy(dict(summary_value))
        identifier = _EVIDENCE_ANCHOR_BASE + priority
        relative = Path("evidence_anchors") / f"candidate_{identifier}"
        destination = dynamic_root / relative
        candidate_sha = canonical_sha256(config)
        evidence = dynamic_grasp_rank_evidence(
            {"candidate_id": identifier, "config": config, "summary": summary}
        )
        actual = summary.get("metrics", {}).get("actual_grasp_pose", {})
        payload = {
            "candidate_result_schema_version": 1,
            "complete": True,
            "campaign_kind": CAMPAIGN_KIND,
            "stage": "authenticated_evidence_grasp_anchor",
            "candidate_id": identifier,
            "source_candidate_id": identifier,
            "controller_seed_index": -1,
            "candidate_sha256": candidate_sha,
            "grasp_pose_id": grasp_pose_id(config),
            "controller_id": controller_id(config),
            "grasp_success": True,
            "classification": "actual_contact_grasp_pose_acquired",
            "rank_evidence": evidence,
            "actual_grasp_pose": copy.deepcopy(actual),
            "summary": summary,
            "evidence_anchor": True,
            "evidence_anchor_priority": priority,
            "source_evidence": copy.deepcopy(descriptor),
            "artifacts": {
                "resolved_config": "resolved_config.json",
                "trace": "trace.npz",
                "trace_retained": True,
                "sha256": {},
            },
        }
        if destination.exists():
            raise RuntimeError(f"partial evidence anchor exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=destination.parent, prefix=f".{destination.name}."
        ) as staging_name:
            staging = Path(staging_name)
            shutil.copy2(config_path, staging / "resolved_config.json")
            shutil.copy2(trace_path, staging / "trace.npz")
            payload["artifacts"]["sha256"] = {
                "resolved_config": file_sha256(staging / "resolved_config.json"),
                "trace": file_sha256(staging / "trace.npz"),
            }
            payload = bind_candidate_result_semantic_sha256(payload)
            write_json(staging / "result.json", payload)
            staging.rename(destination)
        record = {
            **copy.deepcopy(payload),
            "config": copy.deepcopy(config),
            "artifact_directory": str(relative),
            "reused": False,
        }
        records.append(record)
        report_records.append(
            {
                "candidate_id": identifier,
                "evidence_anchor_priority": priority,
                "artifact_directory": str(relative),
                "source_evidence": copy.deepcopy(descriptor),
            }
        )
    report = {
        "actual_contact_evidence_anchor_bundle_schema_version": 1,
        "complete": True,
        "source_bundle_sha256": supplied_sha,
        "evidence_grasp_anchor_count": len(records),
        "selection_policy": "authenticated_anchors_before_discovered_grasps",
        "anchors": report_records,
    }
    _write_or_verify_json(report_path, report)
    artifacts = [report_path]
    for record in records:
        artifacts.extend(_dynamic_artifact_paths(record, dynamic_root))
    return CampaignStageExecution(
        records=tuple(records),
        artifacts=tuple(artifacts),
        summary={
            "evidence_grasp_anchor_count": len(records),
            "source_bundle_sha256": supplied_sha,
        },
    )


def _write_or_verify_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministic artifact or authenticate an identical retry."""

    expected = copy.deepcopy(dict(payload))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != canonical_sha256(expected):
            raise RuntimeError(f"resumable artifact input/output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, expected)


def _run_size_continuation_stage(
    template: Mapping[str, Any],
    sources: Sequence[ActualQposSource],
    output_dir: Path,
    *,
    source_bundle_sha256: str,
) -> CampaignStageExecution | None:
    """Persist registered 1 mm DLS continuation proposals, when enabled.

    The returned records contain only publishable campaign sizes.  Hidden
    bridge/pre-bridge records remain in the hash-bound report so they can be
    audited, but they can never enter dynamics or a Viewer catalog.
    """

    if "size_continuation" not in template:
        return None
    from .actual_contact_size_continuation import continue_actual_qpos_sources

    continuation_input_sha256 = canonical_sha256(
        {
            "experiment_id": template.get("experiment_id"),
            "source_bundle_sha256": source_bundle_sha256,
            "policy": template["size_continuation"],
        }
    )
    report_path = output_dir / "sources" / "size_continuation.json"
    if report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            payload.get("complete") is not True
            or payload.get("continuation_input_sha256")
            != continuation_input_sha256
        ):
            raise RuntimeError(
                f"size continuation resume input changed: {report_path}"
            )
        published = tuple(
            _authenticate_static_record(value)
            for value in payload.get("published_records", ())
        )
        bridge = tuple(
            _authenticate_static_record(value)
            for value in payload.get("internal_records", ())
        )
        report = payload.get("report")
        if not isinstance(report, Mapping):
            raise RuntimeError("size continuation report is missing")
    else:
        execution = continue_actual_qpos_sources(template, sources)
        published = execution.published_records
        bridge = execution.bridge_records
        report = execution.report
        payload = {
            "actual_contact_size_continuation_stage_schema_version": 1,
            "complete": True,
            "experiment_id": template.get("experiment_id"),
            "continuation_input_sha256": continuation_input_sha256,
            "source_bundle_sha256": source_bundle_sha256,
            "report": copy.deepcopy(report),
            "published_records": list(published),
            "internal_records": list(bridge),
        }
        _write_or_verify_json(report_path, payload)
    if any(not bool(value.get("size_continuation", {}).get("publishable")) for value in published):
        raise RuntimeError("size continuation exposed a non-publishable record")
    if any(bool(value.get("size_continuation", {}).get("publishable")) for value in bridge):
        raise RuntimeError("size continuation mislabeled an internal record")
    return CampaignStageExecution(
        records=published,
        artifacts=(report_path,),
        summary={
            "continuation_input_sha256": continuation_input_sha256,
            "published_record_count": len(published),
            "published_static_pass_count": sum(
                bool(value.get("static_pass")) for value in published
            ),
            "internal_record_count": len(bridge),
            "source_success_evidence_inherited": False,
        },
    )


def _static_cell_input_sha256(
    *,
    campaign_input_sha256: str,
    source_bundle_sha256: str,
    stage: str,
    cell: ActualContactSearchCell,
    start_index: int,
    sample_count: int,
    retain_count: int,
    seed: int,
    prior_pool_sha256: str | None = None,
) -> str:
    return canonical_sha256(
        {
            "campaign_input_sha256": campaign_input_sha256,
            "source_bundle_sha256": source_bundle_sha256,
            "stage": stage,
            "cell": cell.as_dict(),
            "start_index": int(start_index),
            "sample_count": int(sample_count),
            "retain_count": int(retain_count),
            "seed": int(seed),
            "prior_pool_sha256": prior_pool_sha256,
            "screen": {
                "mode": "direct_actual_contact_qpos",
                "interpolation": False,
                "static_filter_is_success_evidence": False,
            },
        }
    )


def _authenticate_static_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    config = result.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("persisted static candidate has no config")
    if canonical_sha256(config) != str(result.get("candidate_sha256")):
        raise RuntimeError("persisted static candidate semantic hash changed")
    if grasp_pose_id(config) != str(result.get("grasp_pose_id")):
        raise RuntimeError("persisted static candidate grasp_pose_id changed")
    return result


def _screen_static_candidate_chunks(
    template: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    cell: ActualContactSearchCell,
    *,
    start_index: int,
    sample_count: int,
    seed: int,
    retain_pool_count: int,
    chunk_size: int = 256,
) -> tuple[tuple[dict[str, Any], ...], int, int]:
    """Stream one cell while retaining a globally deterministic small pool."""

    pool: list[dict[str, Any]] = []
    evaluated = 0
    passed = 0
    for offset in range(0, sample_count, chunk_size):
        count = min(chunk_size, sample_count - offset)
        generated = generate_actual_contact_pose_candidates(
            template,
            sources,
            cell,
            count=count,
            start_index=start_index + offset,
            seed=seed,
        )
        screened = screen_actual_contact_pose_cell(
            generated, top_k=min(retain_pool_count, count)
        )
        evaluated += int(screened["evaluated_count"])
        passed += int(screened["static_pass_count"])
        pool.extend(copy.deepcopy(dict(value)) for value in screened["retained"])
        pool = list(
            retain_top_actual_contact_candidates(
                pool, top_k=retain_pool_count
            )[cell.cell_index]
        )
    if evaluated != sample_count:
        raise RuntimeError("static cell did not evaluate its declared sample count")
    return tuple(pool), evaluated, passed


def _screen_static_cell_job(
    job: tuple[
        Mapping[str, Any],
        Sequence[Mapping[str, Any]],
        Any,
        int,
        int,
        int,
        int,
    ],
) -> tuple[int, tuple[dict[str, Any], ...], int, int]:
    """Spawn-safe worker for one complete prefix-stable search stratum."""

    (
        template,
        sources,
        cell,
        start_index,
        sample_count,
        seed,
        retain_pool_count,
    ) = job
    pool, evaluated, passed = _screen_static_candidate_chunks(
        template,
        sources,
        cell,
        start_index=start_index,
        sample_count=sample_count,
        seed=seed,
        retain_pool_count=retain_pool_count,
    )
    return int(cell.cell_index), pool, evaluated, passed


def _run_static_stage(
    template: Mapping[str, Any],
    sources: Sequence[V8ActualQposSource],
    output_dir: Path,
    *,
    stage: str,
    campaign_input_sha256: str,
    seed: int,
    start_index: int,
    sample_count_per_cell: int,
    selected_retain_per_cell: int,
    pool_retain_per_cell: int,
    prior_pool_by_cell: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
    workers: int = 1,
) -> CampaignStageExecution:
    """Run/reuse one quick or expanded direct-pose static stage."""

    resolved_workers = _positive_int(workers, "workers")
    cells = actual_contact_search_cells(template)
    generator_sources = tuple(source.generator_record() for source in sources)
    source_report = [source.report_record() for source in sources]
    source_bundle_sha256 = canonical_sha256(source_report)
    stage_root = output_dir / "static" / stage
    retained_all: list[dict[str, Any]] = []
    full_pool_by_cell: dict[str, list[dict[str, Any]]] = {}
    cell_paths: list[Path] = []
    evaluated_total = 0
    pass_total = 0
    prior_record_total = 0
    prior_static_pass_total = 0
    specifications: list[dict[str, Any]] = []
    missing_jobs: list[
        tuple[
            Mapping[str, Any],
            Sequence[Mapping[str, Any]],
            Any,
            int,
            int,
            int,
            int,
        ]
    ] = []
    resolved_new_pools: dict[
        int, tuple[tuple[dict[str, Any], ...], int, int]
    ] = {}
    for cell in cells:
        prior_values = tuple(
            copy.deepcopy(dict(value))
            for value in (prior_pool_by_cell or {}).get(cell.cell_index, ())
        )
        prior_pool_sha256 = (
            canonical_sha256(
                [
                    {
                        "candidate_id": int(value["candidate_id"]),
                        "candidate_sha256": str(value["candidate_sha256"]),
                        "static_pass": bool(value.get("static_pass")),
                    }
                    for value in prior_values
                ]
            )
            if prior_values
            else None
        )
        input_sha = _static_cell_input_sha256(
            campaign_input_sha256=campaign_input_sha256,
            source_bundle_sha256=source_bundle_sha256,
            stage=stage,
            cell=cell,
            start_index=start_index,
            sample_count=sample_count_per_cell,
            retain_count=pool_retain_per_cell,
            seed=seed,
            prior_pool_sha256=prior_pool_sha256,
        )
        cell_path = stage_root / f"cell_{cell.cell_index:02d}.json"
        specification = {
            "cell": cell,
            "prior_values": prior_values,
            "input_sha": input_sha,
            "cell_path": cell_path,
        }
        specifications.append(specification)
        if cell_path.is_file():
            payload = json.loads(cell_path.read_text(encoding="utf-8"))
            if payload.get("cell_input_sha256") != input_sha or payload.get("complete") is not True:
                raise RuntimeError(f"static cell resume input changed: {cell_path}")
            new_pool = tuple(
                _authenticate_static_record(value)
                for value in payload.get("new_pool", ())
            )
            evaluated = int(payload["evaluated_count"])
            passed = int(payload["static_pass_count"])
            resolved_new_pools[int(cell.cell_index)] = (
                new_pool,
                evaluated,
                passed,
            )
        else:
            missing_jobs.append(
                (
                    copy.deepcopy(dict(template)),
                    generator_sources,
                    cell,
                    int(start_index),
                    int(sample_count_per_cell),
                    int(seed),
                    int(pool_retain_per_cell),
                )
            )

    specification_by_cell = {
        int(value["cell"].cell_index): value for value in specifications
    }

    def persist_screened(
        screened: tuple[int, tuple[dict[str, Any], ...], int, int]
    ) -> None:
        cell_index, new_pool, evaluated, passed = screened
        specification = specification_by_cell[int(cell_index)]
        cell = specification["cell"]
        cell_path = Path(specification["cell_path"])
        payload = {
            "actual_contact_static_cell_schema_version": 1,
            "complete": True,
            "stage": stage,
            **cell.as_dict(),
            "cell_input_sha256": str(specification["input_sha"]),
            "start_index": int(start_index),
            "evaluated_count": int(evaluated),
            "static_pass_count": int(passed),
            "new_pool": list(new_pool),
        }
        _write_or_verify_json(cell_path, payload)
        resolved_new_pools[int(cell_index)] = (
            new_pool,
            int(evaluated),
            int(passed),
        )

    if missing_jobs:
        if resolved_workers == 1 or len(missing_jobs) == 1:
            for job in missing_jobs:
                persist_screened(_screen_static_cell_job(job))
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=min(resolved_workers, len(missing_jobs)),
                mp_context=context,
            ) as executor:
                for screened in executor.map(_screen_static_cell_job, missing_jobs):
                    persist_screened(screened)

    for specification in specifications:
        cell = specification["cell"]
        prior_values = specification["prior_values"]
        input_sha = str(specification["input_sha"])
        cell_path = Path(specification["cell_path"])
        new_pool, evaluated, passed = resolved_new_pools[int(cell.cell_index)]
        if not cell_path.is_file():  # pragma: no cover - persisted above.
            raise RuntimeError(f"static cell result was not persisted: {cell_path}")
        combined = list(prior_values)
        combined.extend(copy.deepcopy(dict(value)) for value in new_pool)
        pool = retain_top_actual_contact_candidates(
            combined, top_k=pool_retain_per_cell
        ).get(cell.cell_index, ())
        selected = tuple(pool[:selected_retain_per_cell])
        retained_all.extend(copy.deepcopy(dict(value)) for value in selected)
        full_pool_by_cell[str(cell.cell_index)] = [
            copy.deepcopy(dict(value)) for value in pool
        ]
        evaluated_total += evaluated
        pass_total += passed
        prior_record_total += len(prior_values)
        prior_static_pass_total += sum(
            bool(value.get("static_pass")) for value in prior_values
        )
        cell_paths.append(cell_path)
    retained_all.sort(key=lambda value: int(value["candidate_id"]))
    report = {
        "actual_contact_static_stage_schema_version": 1,
        "complete": True,
        "stage": stage,
        "campaign_input_sha256": campaign_input_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "cell_count": len(cells),
        "sample_start_index": int(start_index),
        "samples_per_cell": int(sample_count_per_cell),
        "evaluated_count": evaluated_total,
        "static_pass_observation_count": pass_total,
        "prior_pool_record_count": prior_record_total,
        "prior_pool_static_pass_count": prior_static_pass_total,
        "selected_retain_per_cell": int(selected_retain_per_cell),
        "pool_retain_per_cell": int(pool_retain_per_cell),
        "retained_count": len(retained_all),
        "retained_static_pass_count": sum(
            bool(value.get("static_pass")) for value in retained_all
        ),
        "retained": retained_all,
        "pool_by_cell": full_pool_by_cell,
    }
    report_path = stage_root / "report.json"
    _write_or_verify_json(report_path, report)
    return CampaignStageExecution(
        records=tuple(retained_all),
        artifacts=tuple((*cell_paths, report_path)),
        summary={
            "evaluated_count": evaluated_total,
            "retained_count": len(retained_all),
            "retained_static_pass_count": report["retained_static_pass_count"],
        },
    )


def _run_uniform_refinement_stage(
    static_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    stage: str,
    top_count: int = 20,
    maximum_iterations: int = 4,
) -> CampaignStageExecution:
    """Run/reuse deterministic uniform-gap IK on the best static hard passes."""

    limit = _positive_int(top_count, "top_count")
    iterations = _positive_int(maximum_iterations, "maximum_iterations")
    selected = tuple(
        sorted(
            (value for value in static_records if bool(value.get("static_pass"))),
            key=_static_rank,
        )[:limit]
    )
    input_sha = canonical_sha256(
        {
            "stage": stage,
            "top_count": limit,
            "maximum_iterations": iterations,
            "target_gap_m": _UNIFORM_TARGET_GAP_M,
            "source_candidates": [
                {
                    "candidate_id": int(value["candidate_id"]),
                    "candidate_sha256": str(value["candidate_sha256"]),
                }
                for value in selected
            ],
        }
    )
    report_path = output_dir / "static" / stage / "uniform_refinement.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("refinement_input_sha256") != input_sha:
            raise RuntimeError(f"uniform refinement resume input changed: {report_path}")
        records = tuple(
            _authenticate_static_record(value)
            for value in report.get("refined_candidates", ())
        )
    else:
        records = tuple(
            refine_uniform_gap_equal_height_contact_pose(
                value,
                maximum_iterations=iterations,
            )
            for value in selected
        )
        report = {
            "actual_contact_uniform_refinement_schema_version": 1,
            "complete": True,
            "stage": stage,
            "refinement_input_sha256": input_sha,
            "top_count_budget": limit,
            "maximum_iterations": iterations,
            "target_gap_m": _UNIFORM_TARGET_GAP_M,
            "selected_source_count": len(selected),
            "refined_static_pass_count": sum(
                bool(value.get("static_pass")) for value in records
            ),
            "refined_candidates": list(records),
        }
        _write_or_verify_json(report_path, report)
    return CampaignStageExecution(
        records=records,
        artifacts=(report_path,),
        summary={
            "selected_source_count": len(selected),
            "refined_candidate_count": len(records),
            "refined_static_pass_count": sum(
                bool(value.get("static_pass")) for value in records
            ),
            "top_count_budget": limit,
            "maximum_iterations": iterations,
        },
    )


def _run_relative_wrist_refinement_stage(
    static_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    stage: str,
    per_edge: int,
    maximum_iterations: int = 6,
) -> CampaignStageExecution:
    """Apply the schema-v11 13-variable DLS under a strict per-edge quota.

    Static sampling retains a small prefix-stable pool in every
    edge/thumb/orbit stratum.  This reducer first covers feasible orbit bands
    within each edge, then fills exactly the registered per-edge quota by
    deterministic rank.  That prevents the easier 85 mm strata from
    consuming the dynamic budget intended for 86--104 mm.
    """

    from .relative_wrist_pose_search import (
        RelativeWristDLSSettings,
        retain_per_edge_orbit_quota,
        solve_orientation_aware_dls,
    )

    edge_quota = _positive_int(per_edge, "per_edge")
    iterations = _positive_int(maximum_iterations, "maximum_iterations")
    selection = retain_per_edge_orbit_quota(
        static_records, per_edge=edge_quota
    )
    input_sha = canonical_sha256(
        {
            "stage": stage,
            "per_edge": edge_quota,
            "maximum_iterations": iterations,
            "method": "orientation_aware_actual_contact_dls_13_variables",
            "source_candidates": [
                {
                    "candidate_id": int(value["candidate_id"]),
                    "candidate_sha256": str(value["candidate_sha256"]),
                    "edge_m": float(value["edge_m"]),
                    "clockwise_orbit_deg": float(value["clockwise_orbit_deg"]),
                }
                for value in selection.records
            ],
        }
    )
    report_path = (
        output_dir
        / "static"
        / stage
        / "relative_wrist_orientation_refinement.json"
    )
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("refinement_input_sha256") != input_sha:
            raise RuntimeError(
                f"relative-wrist refinement resume input changed: {report_path}"
            )
        records = tuple(
            _authenticate_static_record(value)
            for value in report.get("refined_candidates", ())
        )
    else:
        records_list: list[dict[str, Any]] = []
        dls_records: list[dict[str, Any]] = []
        for source in selection.records:
            source_record = copy.deepcopy(dict(source))
            orbit = float(source_record["clockwise_orbit_deg"])
            try:
                solved = solve_orientation_aware_dls(
                    source_record["config"],
                    clockwise_orbit_deg=orbit,
                    settings=RelativeWristDLSSettings(
                        maximum_iterations=iterations
                    ),
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                # Preserve a fully ranked near miss.  DLS is a geometry
                # optimizer, never success evidence, and one singular source
                # must not erase the quota for its size.
                source_record.setdefault("relative_wrist_dls", {})
                source_record["relative_wrist_dls"] = {
                    "executed": False,
                    "error": str(error),
                    "source_candidate_id": int(source_record["candidate_id"]),
                }
                records_list.append(source_record)
                dls_records.append(copy.deepcopy(source_record["relative_wrist_dls"]))
                continue

            refined = copy.deepcopy(source_record)
            refined_id = int(source_record["candidate_id"]) + _UNIFORM_REFINE_ID_OFFSET
            refined["candidate_id"] = refined_id
            refined["parent_candidate_id"] = int(source_record["candidate_id"])
            refined["config"] = copy.deepcopy(solved.config)
            refined["config"].setdefault("candidate_metadata", {}).update(
                {
                    "stage": "relative_wrist_orientation_refinement",
                    "candidate_id": refined_id,
                    "parent_candidate_id": int(source_record["candidate_id"]),
                    "orientation_aware_dls": True,
                }
            )
            static_result = solved.static_result
            refined["static_pass"] = bool(static_result.static_geometry_pass)
            refined["static_metrics"] = static_result.as_dict()
            if refined["static_pass"]:
                promoted = apply_precontact_solution(
                    refined["config"], static_result
                )
                try:
                    validate_config(promoted)
                except ValueError as error:
                    refined["static_pass"] = False
                    refined["static_metrics"]["static_geometry_pass"] = False
                    refined["static_metrics"]["promotion_config_valid"] = False
                    refined["static_metrics"]["promotion_config_error"] = str(error)
                else:
                    refined["config"] = promoted
                    refined["static_metrics"]["promotion_config_valid"] = True
            refined["relative_wrist_dls"] = {
                "executed": True,
                "source_candidate_id": int(source_record["candidate_id"]),
                "stop_reason": solved.stop_reason,
                "diagnostics": copy.deepcopy(solved.diagnostics),
            }
            refined["grasp_pose_id"] = grasp_pose_id(refined["config"])
            refined["controller_id"] = controller_id(refined["config"])
            refined["candidate_sha256"] = canonical_sha256(refined["config"])
            refined["static_rank"] = list(_static_rank(refined))

            # Never replace a better already-passed source by a degraded DLS
            # child.  Both choices remain hash-bound in the diagnostic report.
            chosen = min((source_record, refined), key=_static_rank)
            records_list.append(copy.deepcopy(dict(chosen)))
            dls_records.append(
                {
                    "source_candidate_id": int(source_record["candidate_id"]),
                    "refined_candidate_id": refined_id,
                    "selected_candidate_id": int(chosen["candidate_id"]),
                    "source_static_pass": bool(source_record.get("static_pass")),
                    "refined_static_pass": bool(refined.get("static_pass")),
                    "stop_reason": solved.stop_reason,
                    "diagnostics": copy.deepcopy(solved.diagnostics),
                }
            )
        records = tuple(
            sorted(records_list, key=lambda value: int(value["candidate_id"]))
        )
        report = {
            "relative_wrist_refinement_schema_version": 1,
            "complete": True,
            "stage": stage,
            "refinement_input_sha256": input_sha,
            "method": "orientation_aware_actual_contact_dls_13_variables",
            "variable_count": 13,
            "maximum_iterations": iterations,
            "per_edge_quota": edge_quota,
            "quota_satisfied": bool(selection.quota_satisfied),
            "deficient_edges_m": list(selection.deficient_edges_m),
            "per_edge_selected_count": {
                str(edge): int(count)
                for edge, count in selection.per_edge_selected_count.items()
            },
            "per_edge_orbit_coverage_deg": {
                str(edge): list(values)
                for edge, values in selection.per_edge_orbit_coverage_deg.items()
            },
            "selected_source_count": len(selection.records),
            "selected_static_pass_count": sum(
                bool(value.get("static_pass")) for value in records
            ),
            "dls_records": dls_records,
            "refined_candidates": list(records),
        }
        _write_or_verify_json(report_path, report)
    return CampaignStageExecution(
        records=records,
        artifacts=(report_path,),
        summary={
            "selected_source_count": len(records),
            "refined_static_pass_count": sum(
                bool(value.get("static_pass")) for value in records
            ),
            "per_edge_quota": edge_quota,
            "quota_satisfied": bool(report.get("quota_satisfied", False)),
            "deficient_edges_m": copy.deepcopy(
                report.get("deficient_edges_m", [])
            ),
        },
    )


def _replace_static_with_uniform_refinements(
    static_records: Sequence[Mapping[str, Any]],
    refined_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Use a successful refined child in place of its parent, never in addition."""

    replacements = {
        int(value["parent_candidate_id"]): copy.deepcopy(dict(value))
        for value in refined_records
        if bool(value.get("static_pass", False))
    }
    merged = [
        replacements.get(int(value["candidate_id"]), copy.deepcopy(dict(value)))
        for value in static_records
    ]
    identifiers = [int(value["candidate_id"]) for value in merged]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("uniform refinement introduced a candidate ID collision")
    return tuple(merged)


def _local_latin_hypercube(
    count: int, dimensions: int, *, seed: int
) -> np.ndarray:
    """Prefix-independent centered LHS with row zero reserved for the anchor."""

    if count <= 0 or dimensions <= 0:
        raise ValueError("local LHS dimensions and count must be positive")
    if count == 1:
        return np.zeros((1, dimensions), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    remaining = count - 1
    values = np.empty((remaining, dimensions), dtype=np.float64)
    for column in range(dimensions):
        permutation = rng.permutation(remaining)
        values[:, column] = (
            (permutation + rng.random(remaining)) / remaining * 2.0 - 1.0
        )
    return np.vstack((np.zeros((1, dimensions), dtype=np.float64), values))


def _authenticate_materialized_dynamic_record(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    record = copy.deepcopy(dict(raw))
    config = record.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("local refinement candidate has no config")
    if canonical_sha256(config) != str(record.get("candidate_sha256")):
        raise RuntimeError("local refinement config semantic hash changed")
    if grasp_pose_id(config) != str(record.get("grasp_pose_id")):
        raise RuntimeError("local refinement grasp_pose_id changed")
    if controller_id(config) != str(record.get("controller_id")):
        raise RuntimeError("local refinement controller_id changed")
    return record


def select_dynamic_local_refinement_parents(
    dynamic_records: Sequence[Mapping[str, Any]], *, top_count: int
) -> tuple[dict[str, Any], ...]:
    """Select local-search parents from authoritative dynamic rank evidence."""

    from .actual_contact_grasp_pose_dynamic import rank_dynamic_grasp_results

    limit = _positive_int(top_count, "top_count")
    ranked = rank_dynamic_grasp_results(dynamic_records)
    if not ranked:
        return ()
    first_definition = resolve_experiment(dict(ranked[0]["config"]))
    relative = first_definition.relative_wrist_pose_search
    if relative is None:
        return ranked[:limit]

    # Schema-v11 reserves the same local-search budget for every requested
    # object size.  Select the two authoritative best parents within each
    # edge before truncating, rather than allowing global rank to collapse the
    # 2,560 trials onto only the smallest cubes.
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for value in ranked:
        definition = resolve_experiment(dict(value["config"]))
        if definition.experiment_id != first_definition.experiment_id:
            raise ValueError("relative-wrist local parents mix experiments")
        edge = float(value["config"]["cube"]["edge_m"])
        grouped[edge].append(copy.deepcopy(dict(value)))
    selected = [
        value
        for edge in sorted(grouped)
        for value in grouped[edge][: relative.local_seed_poses_per_edge]
    ]
    return tuple(selected[:limit])


def _run_joint_controller_local_refinement_stage(
    dynamic_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    stage: str,
    top_count: int,
    candidates_per_pose: int,
    seed: int,
    maximum_ik_iterations: int = 4,
) -> CampaignStageExecution:
    """Generate the declared top-20 x 64 joint pose/controller candidates."""

    pose_count = _positive_int(top_count, "top_count")
    per_pose = _positive_int(candidates_per_pose, "candidates_per_pose")
    ik_iterations = _positive_int(maximum_ik_iterations, "maximum_ik_iterations")
    selected = select_dynamic_local_refinement_parents(
        dynamic_records, top_count=pose_count
    )
    relative_local_mode = bool(
        selected
        and resolve_experiment(dict(selected[0]["config"]))
        .relative_wrist_pose_search
        is not None
    )
    registered_close_option_sets = {
        tuple(
            float(value)
            for value in resolve_experiment(record["config"])
            .control_protocol.close_duration_options_s
        )
        for record in selected
    }
    if len(registered_close_option_sets) > 1:
        raise ValueError(
            "local refinement parents use different close-duration grids"
        )
    registered_close_options = (
        registered_close_option_sets.pop()
        if registered_close_option_sets
        else tuple()
    )
    input_sha = canonical_sha256(
        {
            "stage": stage,
            "seed": int(seed),
            "top_count": pose_count,
            "candidates_per_pose": per_pose,
            "maximum_ik_iterations": ik_iterations,
            "source_candidates": [
                {
                    "candidate_id": int(value["candidate_id"]),
                    "candidate_sha256": str(value["candidate_sha256"]),
                    "summary_sha256": canonical_sha256(value.get("summary", {})),
                }
                for value in selected
            ],
            "joint_half_width_rad": 0.015,
            "root_half_width_m": 0.0003,
            **(
                {
                    "wrist_local_rotvec_half_width_deg": 0.3,
                    "variable_dimension": 27,
                    "parent_selection": "per_edge_rank_top_2",
                }
                if relative_local_mode
                else {}
            ),
            "preload_half_width_rad": 0.02,
            "profile_start_half_width": 0.08,
            "profile_end_half_width": 0.05,
            "close_duration_options_s": list(registered_close_options),
        }
    )
    report_path = output_dir / "static" / stage / "joint_controller_local_refinement.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("refinement_input_sha256") != input_sha:
            raise RuntimeError(f"local refinement resume input changed: {report_path}")
        dynamic_candidates = tuple(
            _authenticate_materialized_dynamic_record(value)
            for value in report.get("dynamic_candidates", ())
        )
    else:
        dynamic_records: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for pose_rank, parent in enumerate(selected):
            controller_definition = resolve_experiment(parent["config"])
            relative_mode = controller_definition.relative_wrist_pose_search is not None
            units = _local_latin_hypercube(
                per_pose,
                27 if relative_mode else 24,
                seed=int(
                    np.random.SeedSequence(
                        [int(seed), int(parent["candidate_id"]), pose_rank, 9_064]
                    ).generate_state(1)[0]
                ),
            )
            parent_config = copy.deepcopy(dict(parent["config"]))
            parent_nominal = np.asarray(
                [
                    parent_config["grasp_pose"]["nominal_joint_qpos_rad"][name]
                    for name in ACTIVE_ACTUATORS
                ],
                dtype=np.float64,
            )
            parent_root = _vector(
                parent_config["hand_pose"]["translation_m"],
                3,
                "hand translation",
            )
            thumb_index = ACTIVE_ACTUATORS.index(THUMB_BEND_ACTUATOR)
            nonthumb = tuple(
                index for index in range(len(ACTIVE_ACTUATORS)) if index != thumb_index
            )
            parent_metadata = parent_config.get("candidate_metadata", {})
            thumb_center = float(parent_nominal[thumb_index])
            preload_bounds = controller_definition.search_bounds.actuator_targets_rad
            parent_preload = parent_config["control"][
                "contact_preload_targets_rad"
            ]
            parent_profile = parent_config["control"]["close_profile"]
            close_options = tuple(
                float(value)
                for value in parent_config["control_protocol"].get(
                    "close_duration_options_s", ()
                )
            )
            if close_options != registered_close_options:
                raise ValueError(
                    "local refinement close durations must match the registered "
                    "actual-contact experiment"
                )
            parent_close = float(parent_config["control_protocol"]["close_s"])
            ordered_close_options = tuple(
                sorted(close_options, key=lambda value: (abs(value - parent_close), value))
            )
            for local_index, unit in enumerate(units):
                local_static_id = (
                    _LOCAL_REFINEMENT_BASE
                    + pose_rank * per_pose
                    + local_index
                )
                proposal_config = copy.deepcopy(parent_config)
                nominal = parent_nominal.copy()
                for column, actuator_index in enumerate(nonthumb):
                    nominal[actuator_index] += 0.015 * float(unit[column])
                nominal[thumb_index] = thumb_center
                proposal_config["grasp_pose"]["nominal_joint_qpos_rad"] = {
                    name: float(nominal[index])
                    for index, name in enumerate(ACTIVE_ACTUATORS)
                }
                if relative_mode:
                    from .relative_wrist_pose_search import (
                        RelativeWristDLSSettings,
                        RelativeWristVariables,
                        materialize_relative_wrist_candidate,
                        solve_orientation_aware_dls,
                    )

                    parent_variables = RelativeWristVariables.from_config(
                        parent_config
                    )
                    proposal_variables = RelativeWristVariables(
                        tuple(
                            float(nominal[index])
                            for index in nonthumb
                        ),
                        tuple(
                            float(parent_variables.root_delta_cube_m[index])
                            + 0.0003 * float(unit[7 + index])
                            for index in range(3)
                        ),
                        tuple(
                            float(parent_variables.wrist_local_rotvec_rad[index])
                            + math.radians(0.3) * float(unit[10 + index])
                            for index in range(3)
                        ),
                    )
                    relative_metadata = parent_config["candidate_metadata"][
                        "relative_wrist_pose_search"
                    ]
                    proposal_config = materialize_relative_wrist_candidate(
                        proposal_config,
                        proposal_variables,
                        clockwise_orbit_deg=float(
                            relative_metadata["clockwise_orbit_deg"]
                        ),
                    )
                else:
                    proposal_config["hand_pose"]["translation_m"] = (
                        parent_root + 0.0003 * unit[7:10]
                    ).tolist()
                proposal = {
                    "candidate_id": local_static_id,
                    "cell_index": int(parent_metadata.get("cell_index", -1)),
                    "cell_id": str(parent_metadata.get("cell_id", "dynamic_parent")),
                    "edge_m": float(parent_config["cube"]["edge_m"]),
                    "thumb_actual_center_rad": thumb_center,
                    "parent_candidate_id": int(parent["candidate_id"]),
                    "config": proposal_config,
                    # Geometry is re-evaluated below; this only authorizes the
                    # projection helper to start from the promoted parent.
                    "static_pass": True,
                }
                if relative_mode:
                    proposal["clockwise_orbit_deg"] = float(
                        proposal_config["candidate_metadata"][
                            "relative_wrist_pose_search"
                        ]["clockwise_orbit_deg"]
                    )
                try:
                    if relative_mode:
                        solved = solve_orientation_aware_dls(
                            proposal_config,
                            clockwise_orbit_deg=float(
                                proposal["clockwise_orbit_deg"]
                            ),
                            initial_variables=RelativeWristVariables.from_config(
                                proposal_config
                            ),
                            settings=RelativeWristDLSSettings(
                                maximum_iterations=ik_iterations
                            ),
                        )
                        refined_config = copy.deepcopy(solved.config)
                        static_result = solved.static_result
                        static_pass = bool(static_result.static_geometry_pass)
                        if static_pass:
                            refined_config = apply_precontact_solution(
                                refined_config, static_result
                            )
                            validate_config(refined_config)
                        refined = {
                            **copy.deepcopy(proposal),
                            "config": refined_config,
                            "static_pass": static_pass,
                            "static_metrics": static_result.as_dict(),
                            "relative_wrist_dls": {
                                "executed": True,
                                "stop_reason": solved.stop_reason,
                                "diagnostics": copy.deepcopy(solved.diagnostics),
                            },
                        }
                    else:
                        refined = refine_uniform_gap_equal_height_contact_pose(
                            proposal,
                            maximum_iterations=ik_iterations,
                            refined_candidate_id=local_static_id,
                        )
                    if not bool(refined.get("static_pass", False)):
                        diagnostics.append(
                            {
                                "candidate_id": local_static_id,
                                "parent_candidate_id": int(parent["candidate_id"]),
                                "static_pass": False,
                                "geometry_refinement": copy.deepcopy(
                                    refined.get(
                                        "relative_wrist_dls",
                                        refined.get("uniform_refinement", {}),
                                    )
                                ),
                            }
                        )
                        continue
                    controller_config = copy.deepcopy(dict(refined["config"]))
                    controller_config["control"]["contact_preload_targets_rad"] = {
                        name: float(
                            np.clip(
                                float(parent_preload[name])
                                + 0.02
                                * float(unit[(13 if relative_mode else 10) + index]),
                                *preload_bounds[name],
                            )
                        )
                        for index, name in enumerate(ACTIVE_ACTUATORS)
                    }
                    profile: dict[str, dict[str, float]] = {}
                    for group_index, finger in enumerate(ACTIVE_FINGERS):
                        names = FINGER_ACTUATORS[finger]
                        base_start = float(parent_profile[names[0]]["start_fraction"])
                        base_end = float(parent_profile[names[0]]["end_fraction"])
                        start = float(
                            np.clip(
                                base_start
                                + 0.08
                                * float(unit[(21 if relative_mode else 18) + group_index]),
                                0.0,
                                0.95,
                            )
                        )
                        end = float(
                            np.clip(
                                base_end
                                + 0.05
                                * float(unit[(24 if relative_mode else 21) + group_index]),
                                max(start + 0.01, 0.01),
                                1.0,
                            )
                        )
                        for name in names:
                            profile[name] = {
                                "start_fraction": start,
                                "end_fraction": end,
                            }
                    controller_config["control"]["close_profile"] = profile
                    controller_config["control_protocol"]["close_s"] = float(
                        ordered_close_options[local_index % len(ordered_close_options)]
                    )
                    controller_config["control"]["manipulation_delta_rad"] = {
                        name: 0.0 for name in ACTIVE_ACTUATORS
                    }
                    dynamic_id = local_static_id * 16 + local_index % 8
                    metadata = controller_config.setdefault("candidate_metadata", {})
                    metadata.update(
                        {
                            "campaign_kind": CAMPAIGN_KIND,
                            "stage": "dynamic_joint_controller_local_refinement",
                            "candidate_id": dynamic_id,
                            "source_candidate_id": int(parent["candidate_id"]),
                            "local_refinement_pose_rank": pose_rank,
                            "local_refinement_index": local_index,
                            "local_refinement_parent_candidate_id": int(
                                parent["candidate_id"]
                            ),
                            "local_refinement_joint_root_lhs": unit.tolist(),
                            "local_refinement_ik_projection": (
                                "orientation_aware_actual_contact_dls_13_variables"
                                if relative_mode
                                else "uniform_gap_equal_height_damped_least_squares"
                            ),
                            "local_refinement_parent_dynamic_summary_sha256": (
                                canonical_sha256(parent.get("summary", {}))
                            ),
                            "manipulation_delta_is_zero": True,
                        }
                    )
                    validate_config(controller_config)
                    materialized = {
                        "campaign_kind": CAMPAIGN_KIND,
                        "stage": "dynamic_joint_controller_local_refinement",
                        "candidate_id": dynamic_id,
                        "source_candidate_id": int(parent["candidate_id"]),
                        "controller_seed_index": local_index,
                        "grasp_pose_id": grasp_pose_id(controller_config),
                        "controller_id": controller_id(controller_config),
                        "candidate_sha256": canonical_sha256(controller_config),
                        "config": controller_config,
                    }
                    dynamic_records.append(materialized)
                    diagnostics.append(
                        {
                            "candidate_id": local_static_id,
                            "dynamic_candidate_id": int(
                                materialized["candidate_id"]
                            ),
                            "parent_candidate_id": int(parent["candidate_id"]),
                            "parent_dynamic_rank": pose_rank,
                            "static_pass": True,
                            "controller_local_index": local_index,
                            "geometry_refinement": copy.deepcopy(
                                refined.get(
                                    "relative_wrist_dls",
                                    refined.get("uniform_refinement", {}),
                                )
                            ),
                        }
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                    diagnostics.append(
                        {
                            "candidate_id": local_static_id,
                            "parent_candidate_id": int(parent["candidate_id"]),
                            "static_pass": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
        dynamic_records.sort(key=lambda value: int(value["candidate_id"]))
        dynamic_candidates = tuple(dynamic_records)
        report = {
            "actual_contact_joint_controller_local_refinement_schema_version": 1,
            "complete": True,
            "stage": stage,
            "refinement_input_sha256": input_sha,
            "top_count_budget": pose_count,
            "candidates_per_pose_budget": per_pose,
            "declared_candidate_budget": pose_count * per_pose,
            "selected_pose_count": len(selected),
            "selected_parent_candidate_ids": [
                int(value["candidate_id"]) for value in selected
            ],
            "parent_selection": (
                "per_edge_rank_top_2"
                if relative_local_mode
                else "rank_dynamic_grasp_results"
            ),
            "generated_candidate_count": len(selected) * per_pose,
            "dynamic_promoted_count": len(dynamic_candidates),
            "maximum_ik_iterations": ik_iterations,
            "joint_half_width_rad": 0.015,
            "root_half_width_m": 0.0003,
            **(
                {
                    "wrist_local_rotvec_half_width_deg": 0.3,
                    "variable_dimension": 27,
                }
                if relative_local_mode
                else {}
            ),
            "preload_half_width_rad": 0.02,
            "profile_start_half_width": 0.08,
            "profile_end_half_width": 0.05,
            "controller_assignment": "parent_dynamic_control_lhs_refinement",
            "diagnostics": diagnostics,
            "dynamic_candidates": list(dynamic_candidates),
        }
        _write_or_verify_json(report_path, report)
    return CampaignStageExecution(
        records=dynamic_candidates,
        artifacts=(report_path,),
        summary={
            "selected_pose_count": len(selected),
            "selected_parent_candidate_ids": [
                int(value["candidate_id"]) for value in selected
            ],
            "parent_selection": (
                "per_edge_rank_top_2"
                if relative_local_mode
                else "rank_dynamic_grasp_results"
            ),
            "declared_candidate_budget": pose_count * per_pose,
            "generated_candidate_count": len(selected) * per_pose,
            "dynamic_promoted_count": len(dynamic_candidates),
        },
    )


def _run_dynamic_stage(
    static_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    stage: str,
    workers: int,
    seed: int,
) -> CampaignStageExecution:
    """Run/reuse eight controller variants for each hard static promotion."""

    from .actual_contact_grasp_pose_dynamic import (
        grasp_stage_succeeded,
        run_actual_contact_dynamic_grasp_candidates,
    )

    promotable_static = tuple(
        copy.deepcopy(dict(value))
        for value in static_records
        if bool(value.get("static_pass", False))
    )
    controller_seed_count = (
        8
        if not static_records
        else int(
            resolve_experiment(dict(static_records[0]["config"]))
            .actual_contact_grasp_pose_campaign.controller_seeds_per_pose
        )
    )
    dynamic_root = output_dir / "dynamic"
    dynamic_resume = dynamic_root.exists()
    records = run_actual_contact_dynamic_grasp_candidates(
        promotable_static,
        dynamic_root,
        workers=workers,
        resume=dynamic_resume,
        seed=seed,
        controller_seed_count=controller_seed_count,
    )
    report = {
        "actual_contact_dynamic_stage_schema_version": 1,
        "complete": True,
        "stage": stage,
        "static_candidate_ids": [
            int(value["candidate_id"]) for value in promotable_static
        ],
        "dynamic_candidate_count": len(records),
        "grasp_success_count": sum(grasp_stage_succeeded(value) for value in records),
        "candidate_records": [
            {
                "candidate_id": int(value["candidate_id"]),
                "source_candidate_id": int(value["source_candidate_id"]),
                "candidate_sha256": str(value["candidate_sha256"]),
                "grasp_pose_id": str(value["grasp_pose_id"]),
                "controller_id": str(value["controller_id"]),
                "grasp_success": grasp_stage_succeeded(value),
                "artifact_directory": str(value["artifact_directory"]),
            }
            for value in records
        ],
    }
    report_path = output_dir / "dynamic" / f"{stage}_report.json"
    _write_or_verify_json(report_path, report)
    return CampaignStageExecution(
        records=tuple(copy.deepcopy(dict(value)) for value in records),
        artifacts=(report_path,),
        summary={
            "dynamic_candidate_count": len(records),
            "grasp_success_count": report["grasp_success_count"],
        },
    )


def _run_measured_grasp_pose_finalization_stage(
    dynamic_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    stage: str,
    workers: int,
) -> CampaignStageExecution:
    """Rebind every dynamic success to its measured 250 ms actual qpos.

    The finalizer persists failures as diagnostics, but only exact fixed-point
    reacquisitions are returned as manipulation/catalog inputs.
    """

    from .actual_contact_grasp_pose_measured import (
        measured_grasp_pose_succeeded,
        run_or_resume_measured_grasp_finalization,
    )
    from .actual_contact_grasp_pose_dynamic import (
        grasp_stage_succeeded,
        rank_dynamic_grasp_results,
    )

    finalization_input = tuple(
        copy.deepcopy(dict(value)) for value in dynamic_records
    )
    if finalization_input:
        definition = resolve_experiment(dict(finalization_input[0]["config"]))
        relative = definition.relative_wrist_pose_search
        if relative is not None:
            successful = [
                value
                for value in rank_dynamic_grasp_results(finalization_input)
                if grasp_stage_succeeded(value)
            ]
            grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
            for value in successful:
                grouped[float(value["config"]["cube"]["edge_m"])].append(
                    copy.deepcopy(dict(value))
                )
            finalization_input = tuple(
                value
                for edge in sorted(grouped)
                for value in grouped[edge][: relative.exact_candidates_per_edge]
            )

    dynamic_root = output_dir / "dynamic"
    measured_root = dynamic_root / "measured"
    records = run_or_resume_measured_grasp_finalization(
        finalization_input,
        dynamic_root,
        workers=workers,
        resume=measured_root.exists(),
    )
    successes = tuple(
        copy.deepcopy(dict(value))
        for value in records
        if measured_grasp_pose_succeeded(value)
    )
    relative_finalization = bool(
        finalization_input
        and resolve_experiment(dict(finalization_input[0]["config"]))
        .relative_wrist_pose_search
        is not None
    )
    report = {
        "actual_contact_measured_grasp_pose_stage_schema_version": 1,
        "complete": True,
        "stage": stage,
        "input_dynamic_candidate_count": len(dynamic_records),
        "authoritative_dynamic_grasp_success_count": len(records),
        "measured_grasp_pose_success_count": len(successes),
        "measured_grasp_pose_failure_count": len(records) - len(successes),
        "workers": int(workers),
        "selection_policy": (
            "persist_all_diagnostics_promote_only_exact_actual_median_fixed_points"
        ),
        "candidate_records": [
            {
                "candidate_id": int(value["candidate_id"]),
                "source_candidate_id": int(value["source_candidate_id"]),
                "candidate_sha256": str(value["candidate_sha256"]),
                "grasp_pose_id": str(value["grasp_pose_id"]),
                "controller_id": str(value["controller_id"]),
                "measured_grasp_pose_success": measured_grasp_pose_succeeded(
                    value
                ),
                "artifact_directory": str(value["artifact_directory"]),
                "initial_state_source": value.get("initial_state_source"),
                "checkpoint_used": value.get("checkpoint_used"),
                "source_input_sha256": value.get("source_provenance", {}).get(
                    "source_input_sha256"
                ),
            }
            for value in records
        ],
    }
    if relative_finalization:
        report["selected_exact_candidate_count"] = len(finalization_input)
        report["exact_candidate_selection_policy"] = (
            "per_edge_exact_candidate_quota"
        )
    measured_root.mkdir(parents=True, exist_ok=True)
    report_path = measured_root / f"{stage}_report.json"
    _write_or_verify_json(report_path, report)
    artifacts: list[Path] = [report_path]
    for value in records:
        directory = dynamic_root / str(value["artifact_directory"])
        artifacts.extend(
            (
                directory / "resolved_config.json",
                directory / "result.json",
                directory / "trace.npz",
            )
        )
    return CampaignStageExecution(
        records=successes,
        artifacts=tuple(artifacts),
        summary={
            "authoritative_dynamic_grasp_success_count": len(records),
            "measured_grasp_pose_success_count": len(successes),
            "measured_grasp_pose_failure_count": len(records) - len(successes),
            "workers": int(workers),
        },
    )


def _with_evidence_grasp_anchors(
    execution: CampaignStageExecution,
    evidence: CampaignStageExecution,
) -> CampaignStageExecution:
    """Prepend authenticated anchors while rejecting candidate-ID collisions."""

    records = [
        *(copy.deepcopy(dict(value)) for value in evidence.records),
        *(copy.deepcopy(dict(value)) for value in execution.records),
    ]
    identifiers = [int(value["candidate_id"]) for value in records]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("evidence/discovered dynamic candidate ID collision")
    grasp_success_count = sum(
        bool(
            value.get(
                "grasp_success",
                value.get("summary", {})
                .get("stage_status", {})
                .get("grasp_success", False),
            )
        )
        for value in records
    )
    return CampaignStageExecution(
        records=tuple(records),
        artifacts=execution.artifacts,
        summary={
            **copy.deepcopy(execution.summary),
            "evidence_grasp_anchor_count": len(evidence.records),
            "dynamic_candidate_count_including_evidence": len(records),
            "grasp_success_count_including_evidence": grasp_success_count,
        },
    )


def _run_materialized_local_dynamic_stage(
    candidates: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    stage: str,
    workers: int,
) -> CampaignStageExecution:
    """Run one real controller per joint/root/preload/timing local proposal."""

    from .actual_contact_grasp_pose_dynamic import (
        compact_dynamic_candidate_artifacts,
        grasp_stage_succeeded,
        run_or_resume_dynamic_candidates,
    )

    dynamic_root = output_dir / "dynamic"
    records = run_or_resume_dynamic_candidates(
        tuple(copy.deepcopy(dict(value)) for value in candidates),
        dynamic_root,
        workers=workers,
        resume=dynamic_root.exists(),
    )
    records = compact_dynamic_candidate_artifacts(
        records,
        dynamic_root,
        retain_failure_trace_count=24,
    )
    report = {
        "actual_contact_local_dynamic_stage_schema_version": 1,
        "complete": True,
        "stage": stage,
        "materialized_candidate_count": len(candidates),
        "dynamic_candidate_count": len(records),
        "grasp_success_count": sum(grasp_stage_succeeded(value) for value in records),
        "workers": int(workers),
        "candidate_records": [
            {
                "candidate_id": int(value["candidate_id"]),
                "source_candidate_id": int(value["source_candidate_id"]),
                "candidate_sha256": str(value["candidate_sha256"]),
                "grasp_pose_id": str(value["grasp_pose_id"]),
                "controller_id": str(value["controller_id"]),
                "grasp_success": grasp_stage_succeeded(value),
                "artifact_directory": str(value["artifact_directory"]),
            }
            for value in records
        ],
    }
    report_path = output_dir / "dynamic" / f"{stage}_local_refinement_report.json"
    _write_or_verify_json(report_path, report)
    return CampaignStageExecution(
        records=tuple(copy.deepcopy(dict(value)) for value in records),
        artifacts=(report_path,),
        summary={
            "dynamic_candidate_count": len(records),
            "grasp_success_count": report["grasp_success_count"],
            "workers": int(workers),
        },
    )


def _merge_dynamic_executions(
    *executions: CampaignStageExecution,
) -> CampaignStageExecution:
    records: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    for execution in executions:
        records.extend(copy.deepcopy(dict(value)) for value in execution.records)
        artifacts.extend(execution.artifacts)
    identifiers = [int(value["candidate_id"]) for value in records]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("merged dynamic stages contain duplicate candidate IDs")
    grasp_success_count = sum(
        bool(
            value.get(
                "grasp_success",
                value.get("summary", {})
                .get("stage_status", {})
                .get("grasp_success", False),
            )
        )
        for value in records
    )
    return CampaignStageExecution(
        records=tuple(records),
        artifacts=tuple(artifacts),
        summary={
            "dynamic_candidate_count": len(records),
            "grasp_success_count": grasp_success_count,
        },
    )


def _dynamic_artifact_paths(
    record: Mapping[str, Any], dynamic_root: Path
) -> tuple[Path, Path, Path]:
    directory = dynamic_root / Path(str(record["artifact_directory"]))
    return (
        directory / "resolved_config.json",
        directory / "result.json",
        directory / "trace.npz",
    )


def _dynamic_catalog_candidates(
    records: Sequence[Mapping[str, Any]], output_dir: Path
) -> tuple[dict[str, Any], ...]:
    """Bind retained dynamic artifacts for the grasp-pose Viewer catalog."""

    from .actual_contact_grasp_pose_dynamic import rank_dynamic_grasp_results

    dynamic_root = output_dir / "dynamic"
    ordered_by_discovery = sorted(records, key=lambda value: int(value["candidate_id"]))
    discovery = {
        int(value["candidate_id"]): index for index, value in enumerate(ordered_by_discovery)
    }
    selected: list[Mapping[str, Any]] = [
        value
        for value in ordered_by_discovery
        if bool(value.get("grasp_success", value.get("summary", {}).get("stage_status", {}).get("grasp_success", False)))
    ]
    # Catalog publication requires a trace for its diagnostic.  Compaction
    # guarantees that the globally best failures retain one.
    for value in rank_dynamic_grasp_results(records):
        config_path, result_path, trace_path = _dynamic_artifact_paths(value, dynamic_root)
        if value not in selected and trace_path.is_file():
            selected.append(value)
            break
    candidates: list[dict[str, Any]] = []
    for value in selected:
        config_path, result_path, trace_path = _dynamic_artifact_paths(value, dynamic_root)
        if not all(path.is_file() for path in (config_path, result_path, trace_path)):
            if bool(value.get("grasp_success", False)):
                raise RuntimeError("successful dynamic grasp lost a catalog artifact")
            continue
        identifier = int(value["candidate_id"])
        candidates.append(
            {
                "candidate_id": identifier,
                "discovery_index": discovery[identifier],
                "config_path": config_path,
                "result_path": result_path,
                "trace_path": trace_path,
                "summary": copy.deepcopy(dict(value["summary"])),
            }
        )
    return tuple(candidates)


def _load_persisted_manipulation_candidate(
    directory: Path,
    *,
    candidate_id: int,
    expected_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not directory.exists():
        return None
    config_path = directory / "resolved_config.json"
    result_path = directory / "result.json"
    trace_path = directory / "trace.npz"
    if not all(path.is_file() for path in (config_path, result_path)):
        raise RuntimeError(f"incomplete manipulation candidate: {directory}")
    persisted_config = json.loads(config_path.read_text(encoding="utf-8"))
    if canonical_sha256(persisted_config) != canonical_sha256(expected_config):
        raise RuntimeError(f"manipulation candidate config changed: {config_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    authenticate_candidate_result_semantic_sha256(payload, source=result_path)
    if int(payload.get("candidate_id", -1)) != int(candidate_id):
        raise RuntimeError(f"manipulation candidate ID changed: {result_path}")
    if payload.get("candidate_sha256") != canonical_sha256(persisted_config):
        raise RuntimeError(f"manipulation candidate semantic config changed: {config_path}")
    artifacts = payload.get("artifacts", {})
    hashes = artifacts.get("sha256", {}) if isinstance(artifacts, Mapping) else {}
    if hashes.get("resolved_config") != file_sha256(config_path):
        raise RuntimeError(f"manipulation config hash changed: {config_path}")
    trace_retained = bool(artifacts.get("trace_retained", True))
    tombstone = directory / ".trace.npz.compacting"
    if tombstone.exists():
        if trace_retained:
            if trace_path.exists():
                raise RuntimeError(f"ambiguous manipulation trace recovery: {directory}")
            tombstone.replace(trace_path)
        else:
            expected = artifacts.get("trace_sha256_at_evaluation")
            if expected != file_sha256(tombstone):
                raise RuntimeError(f"changed manipulation trace tombstone: {tombstone}")
            tombstone.unlink()
    if trace_retained:
        if not trace_path.is_file() or hashes.get("trace") != file_sha256(trace_path):
            raise RuntimeError(f"manipulation trace hash changed: {trace_path}")
    elif trace_path.exists():
        raise RuntimeError(f"compacted manipulation candidate retained trace: {trace_path}")
    elif not isinstance(artifacts.get("trace_sha256_at_evaluation"), str):
        raise RuntimeError(f"compacted manipulation candidate lost trace digest: {result_path}")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise RuntimeError(f"manipulation candidate has no summary: {result_path}")
    return {
        "candidate_id": candidate_id,
        "config_path": config_path,
        "result_path": result_path,
        "trace_path": trace_path,
        "trace_retained": trace_retained,
        "trace_sha256_at_evaluation": (
            hashes.get("trace")
            if trace_retained
            else artifacts.get("trace_sha256_at_evaluation")
        ),
        "result_semantic_sha256": payload["result_semantic_sha256"],
        "candidate_sha256": payload.get("candidate_sha256"),
        "config": copy.deepcopy(persisted_config),
        "manipulation_delta_rad": copy.deepcopy(
            persisted_config["control"]["manipulation_delta_rad"]
        ),
        "discovery_index": int(payload.get("discovery_index", 0)),
        "source_candidate_id": int(payload.get("source_candidate_id", -1)),
        "summary": copy.deepcopy(dict(summary)),
        "initial_state_source": payload.get("initial_state_source"),
        "checkpoint_used": payload.get("checkpoint_used"),
    }


def _persist_or_run_manipulation_candidate(
    config: Mapping[str, Any],
    directory: Path,
    *,
    candidate_id: int,
    discovery_index: int,
    source_candidate_id: int,
    search_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reusable = _load_persisted_manipulation_candidate(
        directory,
        candidate_id=candidate_id,
        expected_config=config,
    )
    if reusable is not None:
        return reusable
    from ..simulation import run_simulation

    if directory.exists():
        raise RuntimeError(f"partial manipulation candidate exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.TemporaryDirectory(
        dir=directory.parent, prefix=f".{directory.name}."
    ) as staging_name:
        staging = Path(staging_name)
        config_path = staging / "resolved_config.json"
        trace_path = staging / "trace.npz"
        write_json(config_path, config)
        summary = run_simulation(copy.deepcopy(dict(config)), trace_path=trace_path)
        if not trace_path.is_file():
            raise RuntimeError("manipulation full rerun did not create trace.npz")
        stage_status = summary.get("stage_status", {})
        full_success = bool(
            isinstance(stage_status, Mapping)
            and stage_status.get("full_success") is True
            and summary.get("passed") is True
        )
        payload = bind_candidate_result_semantic_sha256({
            "actual_contact_manipulation_candidate_schema_version": 1,
            "complete": True,
            "candidate_id": int(candidate_id),
            "discovery_index": int(discovery_index),
            "source_candidate_id": int(source_candidate_id),
            "candidate_sha256": canonical_sha256(config),
            "grasp_pose_id": grasp_pose_id(config),
            "controller_id": controller_id(config),
            "initial_state_source": "configured_no_contact_reset",
            "checkpoint_used": False,
            "checkpoint_usage": "search_probes_only_not_final_evidence",
            "full_success": full_success,
            "search_metadata": copy.deepcopy(dict(search_metadata or {})),
            "summary": summary,
            "artifacts": {
                "resolved_config": "resolved_config.json",
                "trace": "trace.npz",
                "trace_retained": True,
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "trace": file_sha256(trace_path),
                },
            },
        })
        write_json(staging / "result.json", payload)
        staging.rename(directory)
    persisted = _load_persisted_manipulation_candidate(
        directory,
        candidate_id=candidate_id,
        expected_config=config,
    )
    if persisted is None:  # pragma: no cover - atomic rename guarantees this.
        raise RuntimeError("persisted manipulation candidate disappeared")
    return persisted


def _run_persisted_manipulation_candidate_job(
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Spawn-safe full-reset simulation with an atomically persisted trace."""

    local_id = int(job["candidate_id"])
    metadata = job.get("job_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("persisted manipulation job requires metadata")
    source_id = int(metadata["source_candidate_id"])
    source_discovery = int(metadata["source_discovery_index"])
    trust_count = int(metadata["trust_candidate_count"])
    source_root = Path(str(metadata["source_root"]))
    candidate_id = source_id * 1000 + local_id
    discovery_index = source_discovery * trust_count + local_id
    record = _persist_or_run_manipulation_candidate(
        job["config"],
        source_root / "candidates" / f"candidate_{candidate_id}",
        candidate_id=candidate_id,
        discovery_index=discovery_index,
        source_candidate_id=source_id,
    )
    return {
        "candidate_id": local_id,
        "summary": copy.deepcopy(dict(record["summary"])),
        "persisted_candidate_id": candidate_id,
        "discovery_index": discovery_index,
    }


def _run_persisted_manipulation_candidate_jobs(
    jobs: Sequence[Mapping[str, Any]], workers: int
) -> tuple[dict[str, Any], ...]:
    """Run trace-persisting full resets in deterministic spawn-map order."""

    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not jobs:
        return ()
    if workers == 1:
        records = [_run_persisted_manipulation_candidate_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            records = list(
                executor.map(
                    _run_persisted_manipulation_candidate_job,
                    jobs,
                    chunksize=1,
                )
            )
    records.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(records)


def compact_manipulation_candidate_artifacts(
    records: Sequence[Mapping[str, Any]],
    *,
    retain_failure_trace_count: int = 24,
    report_path: Path | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Safely keep every success trace and only the global best failed traces."""

    if (
        not isinstance(retain_failure_trace_count, int)
        or isinstance(retain_failure_trace_count, bool)
        or retain_failure_trace_count < 0
    ):
        raise ValueError("retain_failure_trace_count must be a non-negative integer")
    from .actual_contact_manipulation import (
        manipulation_candidate_rank,
        manipulation_delta_bounds,
    )

    materialized = [copy.deepcopy(dict(value)) for value in records]

    def succeeded(value: Mapping[str, Any]) -> bool:
        summary = value.get("summary", {})
        stage = summary.get("stage_status", {}) if isinstance(summary, Mapping) else {}
        return bool(
            isinstance(stage, Mapping)
            and stage.get("full_success") is True
            and summary.get("passed") is True
        )

    failures = sorted(
        (value for value in materialized if not succeeded(value)),
        key=lambda value: manipulation_candidate_rank(
            value, config=value.get("config")
        ),
    )
    retained_ids = {
        int(value["candidate_id"])
        for value in failures[:retain_failure_trace_count]
    }
    retained_ids.update(
        int(value["candidate_id"]) for value in materialized if succeeded(value)
    )
    compacted: list[dict[str, Any]] = []
    for raw in materialized:
        identifier = int(raw["candidate_id"])
        directory = Path(raw["result_path"]).resolve().parent
        config_path = directory / "resolved_config.json"
        result_path = directory / "result.json"
        trace_path = directory / "trace.npz"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        authenticate_candidate_result_semantic_sha256(payload, source=result_path)
        artifacts = payload.setdefault("artifacts", {})
        hashes = artifacts.setdefault("sha256", {})
        keep = identifier in retained_ids
        trace_retained = bool(artifacts.get("trace_retained", True))
        if keep and not trace_retained:
            if succeeded(raw):
                raise RuntimeError(
                    f"successful manipulation trace was already compacted: {trace_path}"
                )
            # Failure compaction is intentionally irreversible.  A later
            # superset may have removed a trace that this subset would place
            # in its local top-N; never claim that such a trace was restored.
            keep = False
        if not keep and trace_retained:
            if not trace_path.is_file():
                raise RuntimeError(f"manipulation trace disappeared before compaction: {trace_path}")
            observed = file_sha256(trace_path)
            if hashes.get("trace") != observed:
                raise RuntimeError(f"refusing to compact changed manipulation trace: {trace_path}")
            tombstone = directory / ".trace.npz.compacting"
            if tombstone.exists():
                raise RuntimeError(f"stale manipulation compaction tombstone: {tombstone}")
            trace_path.replace(tombstone)
            artifacts["trace"] = None
            artifacts["trace_retained"] = False
            artifacts["trace_sha256_at_evaluation"] = observed
            hashes.pop("trace", None)
            write_json(result_path, payload)
            if file_sha256(tombstone) != observed:
                raise RuntimeError(f"manipulation tombstone changed before deletion: {tombstone}")
            tombstone.unlink()
        elif keep:
            artifacts["trace_retained"] = True
            write_json(result_path, payload)
        expected_config = json.loads(config_path.read_text(encoding="utf-8"))
        loaded = _load_persisted_manipulation_candidate(
            directory,
            candidate_id=identifier,
            expected_config=expected_config,
        )
        if loaded is None:  # pragma: no cover
            raise RuntimeError("compacted manipulation candidate disappeared")
        compacted.append(loaded)
    compacted.sort(key=lambda value: int(value["candidate_id"]))

    ranked = sorted(
        compacted,
        key=lambda value: manipulation_candidate_rank(
            value, config=value.get("config")
        ),
    )
    boundary_hits: list[str] = []
    if ranked and not succeeded(ranked[0]):
        best = ranked[0]
        config = best["config"]
        model, _ = build_model(copy.deepcopy(config))
        bounds = manipulation_delta_bounds(model, config)
        delta = best["manipulation_delta_rad"]
        boundary_hits = [
            name
            for name in ACTIVE_ACTUATORS
            if abs(float(delta[name]) - float(bounds[name][1])) <= 1e-9
            or abs(float(delta[name]) - float(bounds[name][0])) <= 1e-9
        ]
    report = {
        "actual_contact_manipulation_compaction_schema_version": 1,
        "complete": True,
        "candidate_count": len(compacted),
        "full_success_count": sum(succeeded(value) for value in compacted),
        "retained_failure_trace_limit": int(retain_failure_trace_count),
        "retained_trace_count": sum(value["trace_retained"] for value in compacted),
        "compacted_trace_count": sum(not value["trace_retained"] for value in compacted),
        "success_traces_always_retained": True,
        "boundary_limited": bool(boundary_hits),
        "best_candidate_boundary_hits": boundary_hits,
        "boundary_expansion": {
            "registered_bounds_changed": False,
            "status": "diagnostic_only_no_bound_change",
        },
        "candidate_records": [
            {
                "candidate_id": int(value["candidate_id"]),
                "full_success": succeeded(value),
                "trace_retained": bool(value["trace_retained"]),
                "trace_sha256_at_evaluation": value["trace_sha256_at_evaluation"],
                "result_semantic_sha256": value["result_semantic_sha256"],
                "result_file_sha256_after_compaction": file_sha256(value["result_path"]),
            }
            for value in compacted
        ],
    }
    if report_path is not None:
        _write_or_verify_json(report_path, report)
    return tuple(compacted), report


def _load_manipulation_source_report(
    report_path: Path, *, expected_input_sha256: str
) -> tuple[dict[str, Any], ...] | None:
    if not report_path.is_file():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        payload.get("complete") is not True
        or payload.get("source_input_sha256") != expected_input_sha256
    ):
        raise RuntimeError(f"manipulation source resume input changed: {report_path}")
    records: list[dict[str, Any]] = []
    for raw in payload.get("candidate_records", ()):
        directory = report_path.parent / str(raw["artifact_directory"])
        config_path = directory / "resolved_config.json"
        expected_config = json.loads(config_path.read_text(encoding="utf-8"))
        record = _load_persisted_manipulation_candidate(
            directory,
            candidate_id=int(raw["candidate_id"]),
            expected_config=expected_config,
        )
        if record is None:
            raise RuntimeError(f"manipulation source artifact disappeared: {directory}")
        if record["result_semantic_sha256"] != raw.get("result_semantic_sha256"):
            raise RuntimeError(f"manipulation source result evidence changed: {directory}")
        record["discovery_index"] = int(raw["discovery_index"])
        records.append(record)
    return tuple(records)


def _committed_manipulation_result_hashes(
    output_dir: Path, ledger: Mapping[str, Any]
) -> dict[int, set[str]]:
    """Return result hashes authorized by committed compaction reports."""

    authorized: dict[int, set[str]] = defaultdict(set)
    for stage in ledger.get("stages", {}).values():
        artifacts = stage.get("artifacts", {}) if isinstance(stage, Mapping) else {}
        if not isinstance(artifacts, Mapping):
            continue
        for relative in artifacts:
            path = Path(str(relative))
            if not path.name.endswith("_compaction.json"):
                continue
            resolved = (output_dir / path).resolve()
            if not resolved.is_relative_to(output_dir) or not resolved.is_file():
                raise RuntimeError("committed compaction report escaped the campaign")
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            if int(
                payload.get("actual_contact_manipulation_compaction_schema_version", 0)
            ) != 1:
                continue
            for raw in payload.get("candidate_records", ()):
                if not isinstance(raw, Mapping):
                    raise RuntimeError("committed compaction record is not an object")
                digest = raw.get("result_file_sha256_after_compaction")
                if not isinstance(digest, str):
                    raise RuntimeError("committed compaction record lost its result hash")
                authorized[int(raw["candidate_id"])].add(digest)
    return authorized


def _committed_campaign_path(output_dir: Path, value: Any, field: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise RuntimeError(f"committed {field} must be campaign-relative")
    resolved = (output_dir / relative).resolve()
    if not resolved.is_relative_to(output_dir):
        raise RuntimeError(f"committed {field} escaped the campaign")
    return resolved


def _load_committed_manipulation_stage(
    output_dir: Path,
    *,
    stage: str,
    target_success_count: int,
) -> CampaignStageExecution | None:
    """Authenticate and reuse a manipulation stage already in the ledger.

    A later global compaction may legitimately remove failures that an earlier
    quick stage once retained.  Replaying that quick compaction is therefore
    neither idempotent nor safe; the committed report plus a later committed
    compaction hash is the authoritative resume source.
    """

    stage_name = f"{stage}_manipulation_{target_success_count}"
    ledger = validate_stage_ledger(output_dir)
    entry = ledger.get("stages", {}).get(stage_name)
    if not isinstance(entry, Mapping):
        return None
    report_relative = Path(
        "manipulation"
    ) / f"{stage}_target_{target_success_count}_report.json"
    artifacts = entry.get("artifacts", {})
    if not isinstance(artifacts, Mapping) or str(report_relative) not in artifacts:
        raise RuntimeError(f"committed {stage_name} lost its stage report")
    report_path = (output_dir / report_relative).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report.get("actual_contact_manipulation_stage_schema_version", 0)) != 1:
        # Unit/integration adapters predating the production persisted-stage
        # schema remain runnable; only authenticated production reports are
        # eligible for the committed-stage fast path.
        return None
    if (
        report.get("complete") is not True
        or int(report.get("target_success_count", -1)) != target_success_count
    ):
        raise RuntimeError(f"committed {stage_name} report changed identity")
    authorized = _committed_manipulation_result_hashes(output_dir, ledger)
    records: list[dict[str, Any]] = []
    for raw in report.get("candidate_records", ()):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"committed {stage_name} candidate is not an object")
        identifier = int(raw["candidate_id"])
        config_path = _committed_campaign_path(
            output_dir, raw.get("config_path"), "config_path"
        )
        result_path = _committed_campaign_path(
            output_dir, raw.get("result_path"), "result_path"
        )
        trace_path = _committed_campaign_path(
            output_dir, raw.get("trace_path"), "trace_path"
        )
        if result_path.parent != config_path.parent or trace_path.parent != result_path.parent:
            raise RuntimeError(f"committed {stage_name} candidate paths diverged")
        if file_sha256(result_path) not in authorized.get(identifier, set()):
            raise RuntimeError(
                f"current result state is not authorized by a committed compaction: {result_path}"
            )
        expected_config = json.loads(config_path.read_text(encoding="utf-8"))
        record = _load_persisted_manipulation_candidate(
            result_path.parent,
            candidate_id=identifier,
            expected_config=expected_config,
        )
        if record is None:  # pragma: no cover - result_path was authenticated.
            raise RuntimeError(f"committed {stage_name} candidate disappeared")
        if record["result_semantic_sha256"] != raw.get("result_semantic_sha256"):
            raise RuntimeError(
                f"committed {stage_name} candidate evidence changed: {result_path}"
            )
        record["discovery_index"] = int(raw["discovery_index"])
        records.append(record)
    if len(records) != int(report.get("candidate_count", -1)):
        raise RuntimeError(f"committed {stage_name} candidate count changed")
    records.sort(key=lambda value: int(value["candidate_id"]))
    return CampaignStageExecution(
        records=tuple(records),
        artifacts=tuple(output_dir / relative for relative in artifacts),
        summary=copy.deepcopy(dict(entry.get("summary", {}))),
    )


def _run_manipulation_source(
    source: Mapping[str, Any],
    output_dir: Path,
    *,
    source_discovery_index: int,
    seed: int,
    workers: int,
) -> tuple[dict[str, Any], ...]:
    """Checkpoint-probe one grasp and persist every canonical full reset rerun."""

    from .actual_contact_manipulation import (
        ManipulationSearchBudget,
        run_checkpoint_guided_manipulation,
    )

    source_id = int(source["candidate_id"])
    dynamic_root = output_dir / "dynamic"
    config_path, result_path, trace_path = _dynamic_artifact_paths(source, dynamic_root)
    if not all(path.is_file() for path in (config_path, result_path, trace_path)):
        raise RuntimeError("promoted grasp source is missing config/result/trace")
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    source_result = json.loads(result_path.read_text(encoding="utf-8"))
    definition = resolve_experiment(source_config)
    campaign = definition.actual_contact_grasp_pose_campaign
    if campaign is None:
        raise ValueError("manipulation source has no actual-contact campaign")
    budget = ManipulationSearchBudget(
        seed=int(seed),
        trust_candidate_count=int(campaign.manipulation_candidates_per_pose),
    )
    source_input_sha256 = canonical_sha256(
        {
            "source_candidate_id": source_id,
            "candidate_sha256": source.get("candidate_sha256"),
            "grasp_pose_id": source.get("grasp_pose_id"),
            "controller_id": source.get("controller_id"),
            "config_sha256": file_sha256(config_path),
            "result_sha256": file_sha256(result_path),
            "trace_sha256": file_sha256(trace_path),
            "budget": asdict(budget),
        }
    )
    source_root = output_dir / "manipulation" / "sources" / f"source_{source_id}"
    source_report_path = source_root / "search_report.json"
    reusable = _load_manipulation_source_report(
        source_report_path, expected_input_sha256=source_input_sha256
    )
    if reusable is not None:
        return reusable

    metadata = tuple(
        {
            "source_candidate_id": source_id,
            "source_discovery_index": int(source_discovery_index),
            "trust_candidate_count": budget.trust_candidate_count,
            "source_root": str(source_root),
        }
        for _ in range(budget.trust_candidate_count)
    )
    search = run_checkpoint_guided_manipulation(
        source_config,
        trace_path,
        source_result,
        budget=budget,
        workers=workers,
        full_reset_executor=_run_persisted_manipulation_candidate_jobs,
        full_reset_job_metadata=metadata,
    )
    reruns = search["full_reruns"]["candidates"]
    persisted: list[dict[str, Any]] = []
    for rerun in reruns:
        local_id = int(rerun["candidate_id"])
        candidate_id = source_id * 1000 + local_id
        discovery_index = (
            source_discovery_index * budget.trust_candidate_count + local_id
        )
        record = _load_persisted_manipulation_candidate(
            source_root / "candidates" / f"candidate_{candidate_id}",
            candidate_id=candidate_id,
            expected_config=rerun["config"],
        )
        if record is None:
            raise RuntimeError("parallel manipulation candidate was not persisted")
        record["discovery_index"] = discovery_index
        persisted.append(record)
    persisted.sort(key=lambda value: int(value["candidate_id"]))
    if len(reruns) != len(persisted):
        raise RuntimeError("manipulation search/persistence candidate counts diverged")
    if any(
        value.get("initial_state_source") != "configured_no_contact_reset"
        or value.get("checkpoint_used") is not False
        for value in reruns
    ):
        raise RuntimeError("final manipulation evidence did not use a full reset")
    with np.load(trace_path, allow_pickle=False) as trace:
        source_lock_step = int(np.asarray(trace["grasp_lock_step"]).reshape(-1)[0])
        source_actual = np.asarray(
            trace["grasp_pose_actual_qpos_rad"], dtype=np.float64
        ).reshape(-1)
    search_actual = np.asarray(search["actual_grasp_qpos_rad"], dtype=np.float64)
    lock_matches = source_lock_step == int(search["grasp_lock_step"])
    actual_matches = bool(np.array_equal(source_actual, search_actual))
    if not lock_matches or not actual_matches:
        raise RuntimeError("manipulation checkpoint reacquisition changed source grasp evidence")
    source_trace_consistency = {
        "source_grasp_lock_step": source_lock_step,
        "reacquired_grasp_lock_step": int(search["grasp_lock_step"]),
        "lock_step_matches": lock_matches,
        "source_actual_grasp_qpos_rad": source_actual.tolist(),
        "reacquired_actual_grasp_qpos_rad": search_actual.tolist(),
        "stable_window_median_matches": actual_matches,
        "lock_sample_joint_qpos_rad": copy.deepcopy(
            search["lock_sample_joint_qpos_rad"]
        ),
    }
    # Bind the final full-reset evidence itself, not only its search report, to
    # the checkpoint/source audit.  Catalog publication copies these result
    # files verbatim and exposes the same fields in catalog.json below.
    for value in persisted:
        result_file = Path(value["result_path"])
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        payload["initial_state_source"] = "configured_no_contact_reset"
        payload["checkpoint_used"] = False
        payload["source_trace_consistency"] = copy.deepcopy(
            source_trace_consistency
        )
        payload = bind_candidate_result_semantic_sha256(payload)
        write_json(result_file, payload)
        value["result_semantic_sha256"] = payload["result_semantic_sha256"]
    report = {
        "actual_contact_manipulation_source_schema_version": 1,
        "complete": True,
        "source_input_sha256": source_input_sha256,
        "source_candidate_id": source_id,
        "source_discovery_index": int(source_discovery_index),
        "source_trace_consistency": source_trace_consistency,
        "checkpoint_policy": {
            "search_probes_use_grasp_lock_checkpoint": True,
            "final_initial_state_source": "configured_no_contact_reset",
            "final_checkpoint_used": False,
        },
        "budget": copy.deepcopy(search["budget"]),
        "workers_requested": int(workers),
        "workers_used_for_full_reset": min(int(workers), len(persisted)),
        "full_reset_executor": "spawn_atomic_trace_persistence",
        "probe_count": int(search["probe_count"]),
        "probes": copy.deepcopy(search["probes"]),
        "response_model": copy.deepcopy(search["response_model"]),
        "candidate_count": len(persisted),
        "full_success_count": sum(
            bool(value["summary"].get("stage_status", {}).get("full_success"))
            and bool(value["summary"].get("passed"))
            for value in persisted
        ),
        "candidate_records": [
            {
                "candidate_id": int(value["candidate_id"]),
                "discovery_index": int(value["discovery_index"]),
                "artifact_directory": str(
                    Path(value["result_path"]).parent.relative_to(source_root)
                ),
                "initial_state_source": value["initial_state_source"],
                "checkpoint_used": value["checkpoint_used"],
                "result_semantic_sha256": value["result_semantic_sha256"],
            }
            for value in persisted
        ],
    }
    _write_or_verify_json(source_report_path, report)
    return tuple(persisted)


def _existing_manipulation_source_count(output_dir: Path) -> int:
    roots = output_dir / "manipulation" / "sources"
    if not roots.is_dir():
        return 0
    indices = []
    for path in roots.glob("source_*/search_report.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("complete") is True:
            indices.append(int(payload["source_discovery_index"]))
    return max(indices, default=-1) + 1


def _run_persisted_manipulation_refinement_job(
    job: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = job.get("job_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("manipulation refinement job requires metadata")
    identifier = int(job["candidate_id"])
    root = Path(str(metadata["refinement_root"]))
    record = _persist_or_run_manipulation_candidate(
        job["config"],
        root / "candidates" / f"candidate_{identifier}",
        candidate_id=identifier,
        discovery_index=int(metadata["discovery_index"]),
        source_candidate_id=int(metadata["parent_candidate_id"]),
        search_metadata={
            "stage": "global_top8_local_refinement",
            "parent_candidate_id": int(metadata["parent_candidate_id"]),
            "parent_rank": int(metadata["parent_rank"]),
            "local_index": int(metadata["local_index"]),
        },
    )
    return {
        "candidate_id": identifier,
        "summary": copy.deepcopy(record["summary"]),
    }


def _run_persisted_manipulation_refinement_jobs(
    jobs: Sequence[Mapping[str, Any]], workers: int
) -> tuple[dict[str, Any], ...]:
    if not jobs:
        return ()
    if workers == 1:
        values = [_run_persisted_manipulation_refinement_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            values = list(
                executor.map(
                    _run_persisted_manipulation_refinement_job,
                    jobs,
                    chunksize=1,
                )
            )
    values.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(values)


def _run_manipulation_local_refinement_stage(
    parent_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    seed: int,
    workers: int,
    budget: Any | None = None,
) -> CampaignStageExecution:
    """Run the declared global top-eight by 128 full-reset refinement."""

    from .actual_contact_manipulation import (
        LocalRefinementBudget,
        generate_multiconfig_local_refinement_candidates,
        manipulation_candidate_rank_evidence,
        materialize_manipulation_config,
        rank_manipulation_candidates,
    )

    resolved_budget = (
        LocalRefinementBudget(seed=int(seed)) if budget is None else budget
    )
    ranked = rank_manipulation_candidates(parent_records)
    if len(ranked) < int(resolved_budget.parent_count):
        raise RuntimeError(
            "manipulation local refinement requires eight full-reset parents"
        )
    parents = tuple(ranked[: int(resolved_budget.parent_count)])
    generated = generate_multiconfig_local_refinement_candidates(
        parents,
        budget=resolved_budget,
    )
    parent_by_id = {int(value["candidate_id"]): value for value in parents}
    parent_evidence = [
        {
            "candidate_id": int(value["candidate_id"]),
            "candidate_sha256": value["candidate_sha256"],
            "result_semantic_sha256": value["result_semantic_sha256"],
            "manipulation_delta_rad": copy.deepcopy(value["manipulation_delta_rad"]),
        }
        for value in parents
    ]
    refinement_input_sha256 = canonical_sha256(
        {
            "parents": parent_evidence,
            "budget": asdict(resolved_budget),
            "seed": int(seed),
        }
    )
    root = (
        output_dir
        / "manipulation"
        / "local_refinement"
        / f"set_{refinement_input_sha256[:16]}"
    )
    generated_path = root / "generated_candidates.json"
    generated_report = {
        "complete": True,
        "refinement_input_sha256": refinement_input_sha256,
        "budget": asdict(resolved_budget),
        "parent_records": parent_evidence,
        "generated_candidate_count": len(generated),
        "candidates": list(generated),
    }
    _write_or_verify_json(generated_path, generated_report)
    discovery_start = max(
        (int(value.get("discovery_index", -1)) for value in parent_records),
        default=-1,
    ) + 1
    all_records: list[dict[str, Any]] = []
    batch_paths: list[Path] = []
    batch_size = int(resolved_budget.batch_size)
    for batch_index, start in enumerate(range(0, len(generated), batch_size)):
        proposals = generated[start : start + batch_size]
        batch_path = root / "batches" / f"batch_{batch_index:03d}.json"
        jobs = []
        expected_configs: dict[int, dict[str, Any]] = {}
        for offset, proposal in enumerate(proposals):
            identifier = int(proposal["candidate_id"])
            parent = parent_by_id[int(proposal["parent_candidate_id"])]
            config = materialize_manipulation_config(
                parent["config"], proposal["manipulation_delta_rad"]
            )
            expected_configs[identifier] = config
            jobs.append(
                {
                    "candidate_id": identifier,
                    "config": config,
                    "job_metadata": {
                        "refinement_root": str(root),
                        "discovery_index": discovery_start + start + offset,
                        "parent_candidate_id": int(proposal["parent_candidate_id"]),
                        "parent_rank": int(proposal["parent_rank"]),
                        "local_index": int(proposal["local_index"]),
                    },
                }
            )
        if batch_path.is_file():
            batch_payload = json.loads(batch_path.read_text(encoding="utf-8"))
            if (
                batch_payload.get("complete") is not True
                or batch_payload.get("refinement_input_sha256")
                != refinement_input_sha256
                or batch_payload.get("candidate_ids")
                != [int(value["candidate_id"]) for value in proposals]
            ):
                raise RuntimeError(f"manipulation refinement batch changed: {batch_path}")
        else:
            _run_persisted_manipulation_refinement_jobs(tuple(jobs), workers)
            batch_payload = {
                "complete": True,
                "refinement_input_sha256": refinement_input_sha256,
                "batch_index": batch_index,
                "candidate_ids": [int(value["candidate_id"]) for value in proposals],
            }
        batch_records = []
        for job in jobs:
            identifier = int(job["candidate_id"])
            record = _load_persisted_manipulation_candidate(
                root / "candidates" / f"candidate_{identifier}",
                candidate_id=identifier,
                expected_config=expected_configs[identifier],
            )
            if record is None:
                raise RuntimeError("refinement candidate was not persisted")
            batch_records.append(record)
        batch_payload["result_semantic_sha256"] = [
            value["result_semantic_sha256"] for value in batch_records
        ]
        _write_or_verify_json(batch_path, batch_payload)
        all_records.extend(batch_records)
        batch_paths.append(batch_path)
    ranked_results = rank_manipulation_candidates(all_records)
    report = {
        "actual_contact_manipulation_local_refinement_schema_version": 1,
        "complete": True,
        "refinement_input_sha256": refinement_input_sha256,
        "declared_parent_count": int(resolved_budget.parent_count),
        "declared_candidates_per_parent": int(
            resolved_budget.candidates_per_parent
        ),
        "declared_candidate_budget": int(resolved_budget.maximum_candidate_count),
        "executed_candidate_count": len(all_records),
        "full_success_count": sum(
            bool(value["summary"].get("passed"))
            and bool(value["summary"].get("stage_status", {}).get("full_success"))
            for value in all_records
        ),
        "stop_reason": (
            "full_success_found"
            if any(
                bool(value["summary"].get("passed"))
                and bool(value["summary"].get("stage_status", {}).get("full_success"))
                for value in all_records
            )
            else "declared_1024_candidate_budget_exhausted"
        ),
        "best_candidate_id": int(ranked_results[0]["candidate_id"]),
        "best_rank_evidence": manipulation_candidate_rank_evidence(
            ranked_results[0], config=ranked_results[0]["config"]
        ),
    }
    report_path = root / "search_report.json"
    _write_or_verify_json(report_path, report)
    return CampaignStageExecution(
        records=tuple(all_records),
        artifacts=tuple((generated_path, *batch_paths, report_path)),
        summary=copy.deepcopy(report),
    )


def _run_manipulation_stage(
    dynamic_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    target_success_count: int,
    seed: int,
    workers: int,
    stage: str = "default",
    enable_local_refinement: bool = True,
) -> CampaignStageExecution:
    """Run 64-proposal sources, then the declared global top8x128 refinement."""

    from .actual_contact_grasp_pose_dynamic import promotable_grasp_results

    ranked = promotable_grasp_results(dynamic_records)
    anchors = sorted(
        (value for value in ranked if bool(value.get("evidence_anchor", False))),
        key=lambda value: (
            int(value.get("evidence_anchor_priority", 2**31 - 1)),
            int(value["candidate_id"]),
        ),
    )
    ordinary = [value for value in ranked if not bool(value.get("evidence_anchor", False))]
    relative = (
        None
        if not ranked
        else resolve_experiment(dict(ranked[0]["config"])).relative_wrist_pose_search
    )
    if relative is None:
        promoted = tuple((*anchors, *ordinary))[:24]
        maximum_sources = 24
        source_selection_policy = "authenticated_anchors_then_global_dynamic_rank"
    else:
        maximum_sources = int(relative.max_manipulation_pose_count)
        grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for value in (*anchors, *ordinary):
            grouped[float(value["config"]["cube"]["edge_m"])].append(
                copy.deepcopy(dict(value))
            )
        first_per_edge = [grouped[edge][0] for edge in sorted(grouped)]
        selected_ids = {int(value["candidate_id"]) for value in first_per_edge}
        remaining = [
            value
            for value in (*anchors, *ordinary)
            if int(value["candidate_id"]) not in selected_ids
        ]
        promoted = tuple((*first_per_edge, *remaining))[:maximum_sources]
        source_selection_policy = (
            "one_best_measured_grasp_per_successful_edge_then_global_rank"
        )
    all_records: list[dict[str, Any]] = []
    next_source_discovery = _existing_manipulation_source_count(output_dir)
    source_report_paths: list[Path] = []
    for source in promoted:
        source_id = int(source["candidate_id"])
        report_path = (
            output_dir
            / "manipulation"
            / "sources"
            / f"source_{source_id}"
            / "search_report.json"
        )
        if report_path.is_file():
            source_discovery = int(
                json.loads(report_path.read_text(encoding="utf-8"))[
                    "source_discovery_index"
                ]
            )
        else:
            source_discovery = next_source_discovery
            next_source_discovery += 1
        records = _run_manipulation_source(
            source,
            output_dir,
            source_discovery_index=source_discovery,
            seed=seed,
            workers=workers,
        )
        all_records.extend(copy.deepcopy(dict(value)) for value in records)
        source_report_paths.append(report_path)
        target_selection = select_actual_contact_candidates(
            all_records,
            kind="manipulation",
            selected_count=target_success_count,
        )
        if target_selection.metadata["target_reached"]:
            break
    all_records.sort(key=lambda value: int(value["discovery_index"]))
    full_count = sum(
        bool(value["summary"].get("passed"))
        and bool(value["summary"].get("stage_status", {}).get("full_success"))
        for value in all_records
    )
    target_selection = select_actual_contact_candidates(
        all_records,
        kind="manipulation",
        selected_count=target_success_count,
    )
    refinement: CampaignStageExecution | None = None
    if (
        enable_local_refinement
        and not target_selection.metadata["target_reached"]
        and len(all_records) >= 8
    ):
        refinement = _run_manipulation_local_refinement_stage(
            all_records,
            output_dir,
            seed=seed,
            workers=workers,
        )
        all_records.extend(
            copy.deepcopy(dict(value)) for value in refinement.records
        )
        all_records.sort(key=lambda value: int(value["discovery_index"]))
        full_count = sum(
            bool(value["summary"].get("passed"))
            and bool(value["summary"].get("stage_status", {}).get("full_success"))
            for value in all_records
        )
        target_selection = select_actual_contact_candidates(
            all_records,
            kind="manipulation",
            selected_count=target_success_count,
        )
    compaction_path = (
        output_dir
        / "manipulation"
        / f"{stage}_target_{target_success_count}_compaction.json"
    )
    compacted, compaction = compact_manipulation_candidate_artifacts(
        all_records,
        retain_failure_trace_count=24,
        report_path=compaction_path,
    )
    all_records = list(compacted)
    report = {
        "actual_contact_manipulation_stage_schema_version": 1,
        "complete": True,
        "target_success_count": int(target_success_count),
        "promotable_grasp_count": len(promoted),
        "maximum_processed_grasp_sources": maximum_sources,
        "evidence_grasp_anchor_count": len(anchors),
        "evidence_anchors_processed_first": True,
        "workers_requested": int(workers),
        "processed_source_count": len(source_report_paths),
        "candidate_count": len(all_records),
        "full_success_count": full_count,
        "target_reached": target_selection.metadata["target_reached"],
        "selection": copy.deepcopy(target_selection.metadata),
        "diversity": copy.deepcopy(target_selection.metadata["diversity"]),
        "stop_reason": (
            "diverse_target_reached"
            if target_selection.metadata["target_reached"]
            else (
                f"maximum_{maximum_sources}_grasp_sources_processed_with_diversity_deficit"
                if len(source_report_paths) >= len(promoted)
                else "declared_search_budget_exhausted_with_diversity_deficit"
            )
        ),
        "local_refinement": (
            {
                "executed": True,
                **copy.deepcopy(refinement.summary),
            }
            if refinement is not None
            else {"executed": False}
        ),
        "trace_compaction": {
            key: copy.deepcopy(value)
            for key, value in compaction.items()
            if key != "candidate_records"
        },
        "candidate_records": [
            {
                "candidate_id": int(value["candidate_id"]),
                "discovery_index": int(value["discovery_index"]),
                "config_path": str(Path(value["config_path"]).relative_to(output_dir)),
                "result_path": str(Path(value["result_path"]).relative_to(output_dir)),
                "trace_path": str(Path(value["trace_path"]).relative_to(output_dir)),
                "initial_state_source": value["initial_state_source"],
                "checkpoint_used": value["checkpoint_used"],
                "trace_retained": value["trace_retained"],
                "trace_sha256_at_evaluation": value[
                    "trace_sha256_at_evaluation"
                ],
                "result_semantic_sha256": value["result_semantic_sha256"],
            }
            for value in all_records
        ],
    }
    if relative is not None:
        report["source_selection_policy"] = source_selection_policy
    stage_report = (
        output_dir
        / "manipulation"
        / f"{stage}_target_{target_success_count}_report.json"
    )
    _write_or_verify_json(stage_report, report)
    refinement_artifacts = () if refinement is None else refinement.artifacts
    return CampaignStageExecution(
        records=tuple(all_records),
        artifacts=tuple(
            (*source_report_paths, *refinement_artifacts, compaction_path, stage_report)
        ),
        summary={
            "processed_source_count": len(source_report_paths),
            "candidate_count": len(all_records),
            "full_success_count": full_count,
            "target_reached": target_selection.metadata["target_reached"],
            "selection": copy.deepcopy(target_selection.metadata),
            "diversity": copy.deepcopy(target_selection.metadata["diversity"]),
            "stop_reason": (
                "diverse_target_reached"
                if target_selection.metadata["target_reached"]
                else "search_budget_exhausted_with_diversity_deficit"
            ),
            "local_refinement_executed": refinement is not None,
            "local_refinement_candidate_count": (
                0 if refinement is None else len(refinement.records)
            ),
            "boundary_limited": bool(compaction["boundary_limited"]),
            "retained_trace_count": int(compaction["retained_trace_count"]),
        },
    )


def _commit_execution_stage(
    output_dir: Path,
    stage: str,
    execution: CampaignStageExecution,
    *,
    stage_input: Mapping[str, Any],
) -> None:
    commit_campaign_stage(
        output_dir,
        stage,
        stage_input=stage_input,
        artifacts=execution.artifacts,
        summary=execution.summary,
    )


def _publish_campaign_catalogs(
    output_dir: Path,
    dynamic_records: Sequence[Mapping[str, Any]],
    manipulation_records: Sequence[Mapping[str, Any]],
    *,
    target_success_count: int,
    experiment_id: str,
) -> tuple[dict[str, str], tuple[Path, ...]]:
    """Publish immutable, target-versioned Viewer catalogs."""

    catalog_root = output_dir / "catalogs" / f"target_{target_success_count}"
    result: dict[str, str] = {}
    artifacts: list[Path] = []
    grasp_candidates = _dynamic_catalog_candidates(dynamic_records, output_dir)
    if grasp_candidates:
        destination = catalog_root / "grasp_pose"
        catalog_path = destination / "catalog.json"
        if destination.exists():
            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            if payload.get("experiment_id") != experiment_id:
                raise RuntimeError("persisted grasp-pose catalog changed experiment")
        else:
            export_actual_contact_grasp_pose_catalog(
                grasp_candidates,
                destination,
                selected_count=target_success_count,
            )
        result["grasp_pose"] = str(catalog_path.relative_to(output_dir))
        artifacts.extend(authenticated_catalog_artifact_paths(catalog_path))
    if manipulation_records:
        destination = catalog_root / "manipulation"
        catalog_path = destination / "catalog.json"
        if destination.exists():
            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            if payload.get("experiment_id") != experiment_id:
                raise RuntimeError("persisted manipulation catalog changed experiment")
        else:
            export_actual_contact_manipulation_catalog(
                manipulation_records,
                destination,
                selected_count=target_success_count,
            )
        catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        source_by_id = {
            str(value["candidate_id"]): value for value in manipulation_records
        }
        for entry in catalog_payload.get("trajectories", ()):
            result_relative = entry.get("artifacts", {}).get("result")
            if not isinstance(result_relative, str):
                continue
            persisted_result = json.loads(
                (destination / result_relative).read_text(encoding="utf-8")
            )
            source_record = source_by_id.get(str(entry.get("candidate_id")), {})
            entry["initial_state_source"] = persisted_result.get(
                "initial_state_source",
                source_record.get("initial_state_source"),
            )
            entry["checkpoint_used"] = persisted_result.get(
                "checkpoint_used", source_record.get("checkpoint_used")
            )
            entry["source_trace_consistency"] = copy.deepcopy(
                persisted_result.get("source_trace_consistency", {})
            )
        write_json(catalog_path, catalog_payload)
        result["manipulation"] = str(catalog_path.relative_to(output_dir))
        artifacts.extend(authenticated_catalog_artifact_paths(catalog_path))
    return result, tuple(dict.fromkeys(artifacts))


def _pool_by_cell_from_static_report(report_path: Path) -> dict[int, tuple[dict[str, Any], ...]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    pools = payload.get("pool_by_cell")
    if not isinstance(pools, Mapping):
        raise RuntimeError("quick static report has no full-prefix pool")
    return {
        int(cell): tuple(_authenticate_static_record(value) for value in values)
        for cell, values in pools.items()
    }


def run_actual_contact_grasp_pose_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool,
    target_success_count: int,
    workers: int,
    seed: int,
    evidence_anchor_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Execute or resume a measured grasp/manipulation campaign.

    A target of one stops immediately after the first canonical full-reset
    manipulation success has been persisted as ``best_first``.  A later
    ``--resume --target-success-count 5`` invocation authenticates every
    prior input and artifact, then continues the same deterministic streams.
    """

    if target_success_count not in (1, 5):
        raise ValueError("target_success_count must be 1 or 5")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    resolved_config_path = Path(config_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    template = load_config(resolved_config_path)
    validate_config(template)
    definition = _resolve_actual_contact_definition(
        template, "actual-contact campaign runner"
    )
    campaign = definition.actual_contact_grasp_pose_campaign
    if campaign is None:
        raise ValueError("experiment has no actual-contact campaign")
    # Authenticate every external source before creating or resuming any
    # workspace.  A missing, truncated, or hash-mismatched v9 trace therefore
    # leaves the new v10 artifact tree untouched.
    sources = load_actual_qpos_sources(template)
    if definition.relative_wrist_pose_search is not None:
        relative_policy = definition.relative_wrist_pose_search
        primary_id = int(
            sources[0].config.get("candidate_metadata", {}).get(
                "candidate_id", -1
            )
        )
        if primary_id != relative_policy.primary_anchor_candidate_id:
            raise RuntimeError(
                "relative-wrist source manifest does not place the registered "
                "primary anchor first"
            )
        secondary_edges = {
            float(value.config["cube"]["edge_m"]) for value in sources[1:]
        }
        if not secondary_edges or not secondary_edges.issubset(
            set(relative_policy.certified_neighbor_edges_m)
        ):
            raise RuntimeError(
                "relative-wrist secondary sources are outside the certified "
                "neighbor edge set"
            )
    manifest = build_campaign_manifest(resolved_config_path, seed=seed)
    workspace = initialize_or_resume_campaign(output, manifest, resume=resume)
    campaign_input_sha256 = str(manifest["campaign_input_sha256"])

    source_schema_versions = sorted(
        {int(value.manifest_schema_version) for value in sources}
    )
    source_kind_counts = {
        kind: sum(value.source_kind == kind for value in sources)
        for kind in sorted({value.source_kind for value in sources})
    }
    authenticated_source_count = sum(
        bool(value.eligible_as_success_evidence) for value in sources
    )
    diagnostic_source_count = len(sources) - authenticated_source_count
    source_report = {
        "actual_contact_source_bundle_schema_version": 2,
        "complete": True,
        "experiment_id": definition.experiment_id,
        "source_manifest_path": str(_resolve_registered_source_manifest(template)),
        "source_manifest_sha256": file_sha256(
            _resolve_registered_source_manifest(template)
        ),
        "source_count": len(sources),
        "source_manifest_schema_versions": source_schema_versions,
        "source_kind_counts": source_kind_counts,
        "authenticated_source_count": authenticated_source_count,
        "diagnostic_source_count": diagnostic_source_count,
        "diagnostic_sources_are_success_evidence": False,
        "qpos_reference": (
            "stable_contact_window_actual_qpos_median"
            if source_schema_versions == [2]
            else "versioned_actual_contact_window_qpos_median"
        ),
        "sources": [value.report_record() for value in sources],
    }
    if definition.relative_wrist_pose_search is not None:
        source_report["relative_wrist_anchor_sampling"] = {
            "primary_candidate_id": (
                definition.relative_wrist_pose_search.primary_anchor_candidate_id
            ),
            "primary_source_index": 0,
            "primary_fraction": (
                definition.relative_wrist_pose_search.primary_anchor_fraction
            ),
            "certified_neighbor_fraction": (
                definition.relative_wrist_pose_search.certified_neighbor_fraction
            ),
            "certified_neighbor_edges_m": list(
                definition.relative_wrist_pose_search.certified_neighbor_edges_m
            ),
        }
    source_report["source_bundle_sha256"] = canonical_sha256(source_report["sources"])
    source_report_path = workspace / "sources" / "source_bundle.json"
    _write_or_verify_json(source_report_path, source_report)
    source_execution = CampaignStageExecution(
        records=tuple(value.generator_record() for value in sources),
        artifacts=(source_report_path,),
        summary={
            "source_count": len(sources),
            "authenticated_source_count": authenticated_source_count,
            "diagnostic_source_count": diagnostic_source_count,
            "source_bundle_sha256": source_report["source_bundle_sha256"],
        },
    )
    _commit_execution_stage(
        workspace,
        "source_bundle",
        source_execution,
        stage_input={
            "campaign_input_sha256": campaign_input_sha256,
            "source_manifest_sha256": source_report["source_manifest_sha256"],
        },
    )

    size_continuation = _run_size_continuation_stage(
        template,
        sources,
        workspace,
        source_bundle_sha256=source_report["source_bundle_sha256"],
    )
    continuation_pool_by_cell: dict[int, tuple[dict[str, Any], ...]] = {}
    if size_continuation is not None:
        grouped_continuation: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in size_continuation.records:
            cell_index = int(record["cell_index"])
            if cell_index < 0:
                raise RuntimeError(
                    "publishable size continuation record has no campaign cell"
                )
            grouped_continuation[cell_index].append(
                copy.deepcopy(dict(record))
            )
        continuation_pool_by_cell = {
            cell: tuple(values)
            for cell, values in sorted(grouped_continuation.items())
        }
        _commit_execution_stage(
            workspace,
            "size_continuation",
            size_continuation,
            stage_input={
                "campaign_input_sha256": campaign_input_sha256,
                "source_bundle_sha256": source_report[
                    "source_bundle_sha256"
                ],
                "continuation_input_sha256": size_continuation.summary[
                    "continuation_input_sha256"
                ],
                "publication_policy": (
                    f"registered_{campaign.edges_m[0] * 1000.0:.0f}_"
                    f"{campaign.edges_m[-1] * 1000.0:.0f}_mm_only"
                ),
            },
        )

    evidence_anchors = _prepare_evidence_grasp_anchors(
        workspace, evidence_anchor_paths
    )
    _commit_execution_stage(
        workspace,
        "evidence_grasp_anchors",
        evidence_anchors,
        stage_input={
            "campaign_input_sha256": campaign_input_sha256,
            "source_bundle_sha256": evidence_anchors.summary[
                "source_bundle_sha256"
            ],
            "selection_policy": "authenticated_anchors_before_discovered_grasps",
        },
    )

    quick_static = _run_static_stage(
        template,
        sources,
        workspace,
        stage="quick",
        campaign_input_sha256=campaign_input_sha256,
        seed=seed,
        start_index=0,
        sample_count_per_cell=campaign.quick_static_samples_per_cell,
        selected_retain_per_cell=campaign.quick_static_retain_per_cell,
        pool_retain_per_cell=campaign.full_static_retain_per_cell,
        prior_pool_by_cell=continuation_pool_by_cell or None,
        workers=(workers if definition.relative_wrist_pose_search is not None else 1),
    )
    _commit_execution_stage(
        workspace,
        "quick_static",
        quick_static,
        stage_input={
            "campaign_input_sha256": campaign_input_sha256,
            "source_bundle_sha256": source_report["source_bundle_sha256"],
            "sample_count": (
                len(actual_contact_search_cells(template))
                * campaign.quick_static_samples_per_cell
            ),
            "retain_per_cell": campaign.quick_static_retain_per_cell,
            "size_continuation_sha256": (
                None
                if size_continuation is None
                else size_continuation.summary["continuation_input_sha256"]
            ),
        },
    )
    if definition.relative_wrist_pose_search is None:
        quick_refinement = _run_uniform_refinement_stage(
            quick_static.records,
            workspace,
            stage="quick",
            top_count=campaign.local_pose_count,
        )
        _commit_execution_stage(
            workspace,
            "quick_uniform_refinement",
            quick_refinement,
            stage_input={
                "campaign_input_sha256": campaign_input_sha256,
                "static_candidate_sha256": canonical_sha256(
                    [value["candidate_sha256"] for value in quick_static.records]
                ),
                "top_count": campaign.local_pose_count,
                "method": "uniform_gap_equal_height_damped_least_squares",
            },
        )
        quick_dynamic_input = _replace_static_with_uniform_refinements(
            quick_static.records, quick_refinement.records
        )
    else:
        relative_policy = definition.relative_wrist_pose_search
        quick_refinement = _run_relative_wrist_refinement_stage(
            quick_static.records,
            workspace,
            stage="quick",
            per_edge=relative_policy.retained_poses_per_edge,
        )
        _commit_execution_stage(
            workspace,
            "quick_relative_wrist_orientation_refinement",
            quick_refinement,
            stage_input={
                "campaign_input_sha256": campaign_input_sha256,
                "static_candidate_sha256": canonical_sha256(
                    [value["candidate_sha256"] for value in quick_static.records]
                ),
                "per_edge_quota": relative_policy.retained_poses_per_edge,
                "maximum_dynamic_grasp_candidate_count": (
                    len(campaign.edges_m)
                    * relative_policy.retained_poses_per_edge
                    * relative_policy.controllers_per_pose
                ),
                "method": "orientation_aware_actual_contact_dls_13_variables",
            },
        )
        quick_dynamic_input = quick_refinement.records
    quick_dynamic = _with_evidence_grasp_anchors(
        _run_dynamic_stage(
            quick_dynamic_input,
            workspace,
            stage="quick",
            workers=workers,
            seed=seed,
        ),
        evidence_anchors,
    )
    _commit_execution_stage(
        workspace,
        "quick_dynamic",
        quick_dynamic,
        stage_input={
            "campaign_input_sha256": campaign_input_sha256,
            "static_candidate_sha256": canonical_sha256(
                [value["candidate_sha256"] for value in quick_dynamic_input]
            ),
            "controller_seeds_per_pose": campaign.controller_seeds_per_pose,
        },
    )
    quick_measured = _run_measured_grasp_pose_finalization_stage(
        quick_dynamic.records,
        workspace,
        stage="quick",
        workers=workers,
    )
    _commit_execution_stage(
        workspace,
        "quick_measured_grasp_pose_finalization",
        quick_measured,
        stage_input={
            "campaign_input_sha256": campaign_input_sha256,
            "dynamic_grasp_evidence_sha256": canonical_sha256(
                [
                    {
                        "candidate_id": int(value["candidate_id"]),
                        "candidate_sha256": value["candidate_sha256"],
                        "summary_sha256": canonical_sha256(
                            value.get("summary", {})
                        ),
                    }
                    for value in quick_dynamic.records
                    if bool(
                        value.get("summary", {})
                        .get("stage_status", {})
                        .get("grasp_success", False)
                    )
                ]
            ),
            "maximum_fixed_point_iterations": 4,
            "required_actual_nominal_relation": "exact_elementwise_equality",
        },
    )
    quick_manipulation = _load_committed_manipulation_stage(
        workspace,
        stage="quick",
        target_success_count=target_success_count,
    ) or _run_manipulation_stage(
        quick_measured.records,
        workspace,
        target_success_count=target_success_count,
        seed=seed,
        workers=workers,
        stage="quick",
        enable_local_refinement=False,
    )
    _commit_execution_stage(
        workspace,
        f"quick_manipulation_{target_success_count}",
        quick_manipulation,
        stage_input={
            "campaign_input_sha256": campaign_input_sha256,
            "dynamic_candidate_sha256": canonical_sha256(
                [value["candidate_sha256"] for value in quick_measured.records]
            ),
            "target_success_count": target_success_count,
        },
    )
    final_static = quick_static
    final_dynamic = quick_measured
    final_manipulation = quick_manipulation
    local_refinement_execution: CampaignStageExecution | None = None
    local_dynamic_execution: CampaignStageExecution | None = None
    quick_target_selection = select_actual_contact_candidates(
        quick_manipulation.records,
        kind="manipulation",
        selected_count=target_success_count,
    )
    if not bool(quick_target_selection.metadata["target_reached"]):
        quick_pool = _pool_by_cell_from_static_report(
            workspace / "static" / "quick" / "report.json"
        )
        expanded_static = _run_static_stage(
            template,
            sources,
            workspace,
            stage="expanded",
            campaign_input_sha256=campaign_input_sha256,
            seed=seed,
            start_index=campaign.quick_static_samples_per_cell,
            sample_count_per_cell=(
                campaign.full_static_samples_per_cell
                - campaign.quick_static_samples_per_cell
            ),
            selected_retain_per_cell=campaign.full_static_retain_per_cell,
            pool_retain_per_cell=campaign.full_static_retain_per_cell,
            prior_pool_by_cell=quick_pool,
            workers=(workers if definition.relative_wrist_pose_search is not None else 1),
        )
        _commit_execution_stage(
            workspace,
            "expanded_static",
            expanded_static,
            stage_input={
                "campaign_input_sha256": campaign_input_sha256,
                "source_bundle_sha256": source_report["source_bundle_sha256"],
                "prefix_start": campaign.quick_static_samples_per_cell,
                "full_sample_count": (
                    len(actual_contact_search_cells(template))
                    * campaign.full_static_samples_per_cell
                ),
                "retain_per_cell": campaign.full_static_retain_per_cell,
            },
        )
        if definition.relative_wrist_pose_search is None:
            expanded_refinement = _run_uniform_refinement_stage(
                expanded_static.records,
                workspace,
                stage="expanded",
                top_count=campaign.local_pose_count,
            )
            _commit_execution_stage(
                workspace,
                "expanded_uniform_refinement",
                expanded_refinement,
                stage_input={
                    "campaign_input_sha256": campaign_input_sha256,
                    "static_candidate_sha256": canonical_sha256(
                        [
                            value["candidate_sha256"]
                            for value in expanded_static.records
                        ]
                    ),
                    "top_count": campaign.local_pose_count,
                    "method": "uniform_gap_equal_height_damped_least_squares",
                },
            )
            expanded_dynamic_input = _replace_static_with_uniform_refinements(
                expanded_static.records, expanded_refinement.records
            )
        else:
            relative_policy = definition.relative_wrist_pose_search
            expanded_refinement = _run_relative_wrist_refinement_stage(
                expanded_static.records,
                workspace,
                stage="expanded",
                per_edge=relative_policy.retained_poses_per_edge,
            )
            _commit_execution_stage(
                workspace,
                "expanded_relative_wrist_orientation_refinement",
                expanded_refinement,
                stage_input={
                    "campaign_input_sha256": campaign_input_sha256,
                    "static_candidate_sha256": canonical_sha256(
                        [
                            value["candidate_sha256"]
                            for value in expanded_static.records
                        ]
                    ),
                    "per_edge_quota": relative_policy.retained_poses_per_edge,
                    "maximum_dynamic_grasp_candidate_count": (
                        len(campaign.edges_m)
                        * relative_policy.retained_poses_per_edge
                        * relative_policy.controllers_per_pose
                    ),
                    "method": "orientation_aware_actual_contact_dls_13_variables",
                },
            )
            expanded_dynamic_input = expanded_refinement.records
        expanded_base_dynamic = _with_evidence_grasp_anchors(
            _run_dynamic_stage(
                expanded_dynamic_input,
                workspace,
                stage="expanded",
                workers=workers,
                seed=seed,
            ),
            evidence_anchors,
        )
        _commit_execution_stage(
            workspace,
            "expanded_dynamic",
            expanded_base_dynamic,
            stage_input={
                "campaign_input_sha256": campaign_input_sha256,
                "static_candidate_sha256": canonical_sha256(
                    [value["candidate_sha256"] for value in expanded_dynamic_input]
                ),
                "maximum_dynamic_candidate_count": (
                    campaign.maximum_dynamic_grasp_candidate_count
                    if definition.relative_wrist_pose_search is None
                    else len(campaign.edges_m)
                    * definition.relative_wrist_pose_search.retained_poses_per_edge
                    * definition.relative_wrist_pose_search.controllers_per_pose
                ),
            },
        )
        local_refinement = _run_joint_controller_local_refinement_stage(
            expanded_base_dynamic.records,
            workspace,
            stage="expanded",
            top_count=campaign.local_pose_count,
            candidates_per_pose=campaign.local_refine_per_pose,
            seed=seed,
        )
        local_refinement_execution = local_refinement
        _commit_execution_stage(
            workspace,
            "expanded_joint_controller_local_refinement",
            local_refinement,
            stage_input={
                "campaign_input_sha256": campaign_input_sha256,
                "dynamic_parent_evidence_sha256": canonical_sha256(
                    [
                        {
                            "candidate_sha256": value["candidate_sha256"],
                            "summary_sha256": canonical_sha256(
                                value.get("summary", {})
                            ),
                        }
                        for value in expanded_base_dynamic.records
                    ]
                ),
                "top_count": campaign.local_pose_count,
                "candidates_per_pose": campaign.local_refine_per_pose,
                "joint_root_ik_projection": (
                    "orientation_aware_actual_contact_dls_13_variables"
                    if definition.relative_wrist_pose_search is not None
                    else "uniform_gap_equal_height_damped_least_squares"
                ),
                "controller_fields": [
                    "contact_preload_targets_rad",
                    "close_profile",
                    "control_protocol.close_s",
                ],
                "parent_selection": (
                    "per_edge_rank_top_2"
                    if definition.relative_wrist_pose_search is not None
                    else "rank_dynamic_grasp_results"
                ),
            },
        )
        local_dynamic = _run_materialized_local_dynamic_stage(
            local_refinement.records,
            workspace,
            stage="expanded",
            workers=workers,
        )
        local_dynamic_execution = local_dynamic
        _commit_execution_stage(
            workspace,
            "expanded_local_refinement_dynamic",
            local_dynamic,
            stage_input={
                "campaign_input_sha256": campaign_input_sha256,
                "local_candidate_sha256": canonical_sha256(
                    [value["candidate_sha256"] for value in local_refinement.records]
                ),
                "declared_candidate_budget": (
                    campaign.local_pose_count * campaign.local_refine_per_pose
                ),
                "workers": workers,
            },
        )
        expanded_dynamic = _merge_dynamic_executions(
            expanded_base_dynamic, local_dynamic
        )
        expanded_measured = _run_measured_grasp_pose_finalization_stage(
            expanded_dynamic.records,
            workspace,
            stage="expanded",
            workers=workers,
        )
        _commit_execution_stage(
            workspace,
            "expanded_measured_grasp_pose_finalization",
            expanded_measured,
            stage_input={
                "campaign_input_sha256": campaign_input_sha256,
                "dynamic_grasp_evidence_sha256": canonical_sha256(
                    [
                        {
                            "candidate_id": int(value["candidate_id"]),
                            "candidate_sha256": value["candidate_sha256"],
                            "summary_sha256": canonical_sha256(
                                value.get("summary", {})
                            ),
                        }
                        for value in expanded_dynamic.records
                        if bool(
                            value.get("summary", {})
                            .get("stage_status", {})
                            .get("grasp_success", False)
                        )
                    ]
                ),
                "maximum_fixed_point_iterations": 4,
                "required_actual_nominal_relation": (
                    "exact_elementwise_equality"
                ),
            },
        )
        expanded_manipulation = _load_committed_manipulation_stage(
            workspace,
            stage="expanded",
            target_success_count=target_success_count,
        ) or _run_manipulation_stage(
            expanded_measured.records,
            workspace,
            target_success_count=target_success_count,
            seed=seed,
            workers=workers,
            stage="expanded",
            enable_local_refinement=True,
        )
        _commit_execution_stage(
            workspace,
            f"expanded_manipulation_{target_success_count}",
            expanded_manipulation,
            stage_input={
                "campaign_input_sha256": campaign_input_sha256,
                "dynamic_candidate_sha256": canonical_sha256(
                    [
                        value["candidate_sha256"]
                        for value in expanded_measured.records
                    ]
                ),
                "target_success_count": target_success_count,
            },
        )
        final_static = expanded_static
        final_dynamic = expanded_measured
        final_manipulation = expanded_manipulation

    catalogs, catalog_artifacts = _publish_campaign_catalogs(
        workspace,
        final_dynamic.records,
        final_manipulation.records,
        target_success_count=target_success_count,
        experiment_id=definition.experiment_id,
    )
    catalog_execution = CampaignStageExecution(
        records=(),
        artifacts=catalog_artifacts,
        summary={"catalogs": catalogs},
    )
    _commit_execution_stage(
        workspace,
        f"catalogs_{target_success_count}",
        catalog_execution,
        stage_input={
            "campaign_input_sha256": campaign_input_sha256,
            "target_success_count": target_success_count,
            "dynamic_candidate_count": len(final_dynamic.records),
            "manipulation_candidate_count": len(final_manipulation.records),
        },
    )
    ledger = validate_stage_ledger(workspace)
    from .actual_contact_grasp_pose_dynamic import promotable_grasp_results

    grasp_success_count = len(promotable_grasp_results(final_dynamic.records))
    full_success_count = sum(
        bool(value["summary"].get("passed"))
        and bool(value["summary"].get("stage_status", {}).get("full_success"))
        for value in final_manipulation.records
    )
    final_selection = select_actual_contact_candidates(
        final_manipulation.records,
        kind="manipulation",
        selected_count=target_success_count,
    )
    manipulation_source_consistency = []
    for path in sorted(
        (workspace / "manipulation" / "sources").glob(
            "source_*/search_report.json"
        )
        if (workspace / "manipulation" / "sources").is_dir()
        else ()
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        manipulation_source_consistency.append(
            {
                "source_candidate_id": int(payload["source_candidate_id"]),
                **copy.deepcopy(dict(payload["source_trace_consistency"])),
            }
        )
    relative_campaign_report: dict[str, Any] | None = None
    if definition.relative_wrist_pose_search is not None:
        relative_policy = definition.relative_wrist_pose_search
        grasp_by_edge = {
            f"{edge * 1000.0:.0f}": sum(
                math.isclose(
                    float(value["config"]["cube"]["edge_m"]), edge, abs_tol=1e-12
                )
                for value in final_dynamic.records
            )
            for edge in campaign.edges_m
        }
        full_by_edge = {
            f"{edge * 1000.0:.0f}": sum(
                bool(value["summary"].get("passed"))
                and bool(
                    value["summary"].get("stage_status", {}).get("full_success")
                )
                and math.isclose(
                    float(value["config"]["cube"]["edge_m"]),
                    edge,
                    abs_tol=1e-12,
                )
                for value in final_manipulation.records
            )
            for edge in campaign.edges_m
        }
        relative_campaign_report = {
            "enabled": True,
            "search_schema_version": relative_policy.schema_version,
            "budget": relative_policy.budget_config(
                edge_count=len(campaign.edges_m),
                thumb_band_count=len(campaign.thumb_actual_centers_rad),
            ),
            "orbit_convention": {
                "axis": "cube_local_+Z",
                "positive": "clockwise_viewed_from_cube_local_+Z",
                "mathematical_angle_sign": -1,
                "translation_and_orientation_rotated_together": True,
            },
            "fixed_mass_kg": campaign.fixed_mass_kg,
            "fixed_mass_classification": (
                campaign.validation_labels["manipulation"]
                if full_success_count > 0 and campaign.validation_labels is not None
                else (
                    campaign.validation_labels["grasp"]
                    if grasp_success_count > 0 and campaign.validation_labels is not None
                    else "not_validated"
                )
            ),
            "grasp_success_count_by_edge_mm": grasp_by_edge,
            "full_success_count_by_edge_mm": full_by_edge,
            "constant_density_revalidation": {
                "density_kg_m3": relative_policy.density_revalidation_kg_m3,
                "status": (
                    "pending_after_fixed_mass_full_success"
                    if full_success_count > 0
                    else "not_run_without_fixed_mass_full_success"
                ),
                "fixed_mass_result_not_overwritten": True,
            },
            "robustness": {
                "required_pass_count": 45,
                "perturbation_count": 50,
                "status": (
                    "pending_after_fixed_mass_full_success"
                    if full_success_count > 0
                    else "not_run_without_fixed_mass_full_success"
                ),
            },
        }
    result = {
        "actual_contact_grasp_pose_campaign_result_schema_version": 1,
        "complete": True,
        "experiment_id": definition.experiment_id,
        "campaign_input_sha256": campaign_input_sha256,
        "resume_supported": True,
        "target_success_count": target_success_count,
        "target_reached": final_selection.metadata["target_reached"],
        "source_count": len(sources),
        "source_authentication": {
            "manifest_schema_versions": source_schema_versions,
            "source_kind_counts": source_kind_counts,
            "authenticated_source_count": authenticated_source_count,
            "diagnostic_source_count": diagnostic_source_count,
            "diagnostic_sources_are_success_evidence": False,
        },
        "size_continuation": (
            {
                "executed": False,
                "reason": "experiment_has_no_size_continuation_policy",
            }
            if size_continuation is None
            else {
                "executed": True,
                **copy.deepcopy(size_continuation.summary),
                "bridge_records_publishable": False,
            }
        ),
        "evidence_grasp_anchor_count": len(evidence_anchors.records),
        "static_retained_count": len(final_static.records),
        "static_pass_count": sum(
            bool(value.get("static_pass")) for value in final_static.records
        ),
        "dynamic_candidate_count": len(final_dynamic.records),
        "grasp_success_count": grasp_success_count,
        "measured_grasp_pose_finalization": {
            "executed": True,
            "ledger_stages": [
                name
                for name in (
                    "quick_measured_grasp_pose_finalization",
                    "expanded_measured_grasp_pose_finalization",
                )
                if name in ledger["stages"]
            ],
            "qpos_reference": (
                "persisted_250ms_actual_contact_window_median"
            ),
            "exact_actual_nominal_required": True,
            "maximum_fixed_point_iterations": 4,
            "initial_state_source": "configured_no_contact_reset",
            "checkpoint_used": False,
            "authoritative_dynamic_grasp_success_count": int(
                final_dynamic.summary[
                    "authoritative_dynamic_grasp_success_count"
                ]
            ),
            "success_count": int(
                final_dynamic.summary["measured_grasp_pose_success_count"]
            ),
            "failure_diagnostic_count": int(
                final_dynamic.summary["measured_grasp_pose_failure_count"]
            ),
        },
        "manipulation_candidate_count": len(final_manipulation.records),
        "full_success_count": full_success_count,
        "selection": copy.deepcopy(final_selection.metadata),
        "diversity": copy.deepcopy(final_selection.metadata["diversity"]),
        "joint_controller_local_refinement": {
            "executed": local_refinement_execution is not None,
            "ledger_stages": [
                "expanded_joint_controller_local_refinement",
                "expanded_local_refinement_dynamic",
            ],
            "top_pose_budget": campaign.local_pose_count,
            "candidates_per_pose_budget": campaign.local_refine_per_pose,
            "declared_candidate_budget": (
                campaign.local_pose_count * campaign.local_refine_per_pose
            ),
            "generated_candidate_count": (
                0
                if local_refinement_execution is None
                else int(
                    local_refinement_execution.summary[
                        "generated_candidate_count"
                    ]
                )
            ),
            "parent_selection": (
                "per_edge_rank_top_2"
                if definition.relative_wrist_pose_search is not None
                else "rank_dynamic_grasp_results"
            ),
            "selected_parent_candidate_ids": (
                []
                if local_refinement_execution is None
                else copy.deepcopy(
                    local_refinement_execution.summary[
                        "selected_parent_candidate_ids"
                    ]
                )
            ),
            "geometry_ik_promoted_count": (
                0
                if local_refinement_execution is None
                else int(
                    local_refinement_execution.summary[
                        "dynamic_promoted_count"
                    ]
                )
            ),
            "full_dynamic_candidate_count": (
                0
                if local_dynamic_execution is None
                else int(
                    local_dynamic_execution.summary["dynamic_candidate_count"]
                )
            ),
            "jointly_varied_fields": [
                (
                    "hand_pose.cube_frame_translation_and_local_rotvec"
                    if definition.relative_wrist_pose_search is not None
                    else "hand_pose.translation_m"
                ),
                "grasp_pose.nominal_joint_qpos_rad",
                "control.contact_preload_targets_rad",
                "control.close_profile",
                "control_protocol.close_s",
            ],
            "per_proposal_geometry_projection": (
                "orientation_aware_actual_contact_dls_13_variables"
                if definition.relative_wrist_pose_search is not None
                else "uniform_gap_equal_height_damped_least_squares"
            ),
        },
        "catalogs": catalogs,
        "checkpoint_policy": {
            "search_probes_may_restore_latched_free_dynamics": True,
            "final_initial_state_source": "configured_no_contact_reset",
            "final_checkpoint_used": False,
        },
        "source_trace_consistency": manipulation_source_consistency,
        "committed_stages": sorted(
            set(ledger["stages"]) | {f"campaign_result_{target_success_count}"}
        ),
    }
    if relative_campaign_report is not None:
        result["relative_wrist_pose_search"] = relative_campaign_report
    result_path = workspace / f"campaign_result_target_{target_success_count}.json"
    _write_or_verify_json(result_path, result)
    commit_campaign_stage(
        workspace,
        f"campaign_result_{target_success_count}",
        stage_input={
            "campaign_input_sha256": campaign_input_sha256,
            "target_success_count": target_success_count,
            "catalogs": catalogs,
        },
        artifacts=(result_path,),
        summary={
            "grasp_success_count": grasp_success_count,
            "full_success_count": full_success_count,
            "target_reached": final_selection.metadata["target_reached"],
            "selection": copy.deepcopy(final_selection.metadata),
            "diversity": copy.deepcopy(final_selection.metadata["diversity"]),
        },
    )
    return result


__all__ = [
    "CAMPAIGN_KIND",
    "DEFAULT_SEED",
    "EXPERIMENT_ID",
    "FINGER_ACTUATORS",
    "STATIC_RESULT_SCHEMA_VERSION",
    "ActualContactSearchCell",
    "ActualContactStaticResult",
    "ActualContactStaticThresholds",
    "ActualQposSource",
    "CampaignStageExecution",
    "FingerContactEvidence",
    "FingerRetreatEvidence",
    "V8ActualQposSource",
    "actual_contact_search_cells",
    "apply_precontact_solution",
    "evaluate_direct_actual_contact_pose",
    "generate_actual_contact_pose_candidates",
    "load_actual_qpos_sources",
    "load_v8_actual_qpos_sources",
    "refine_uniform_gap_equal_height_contact_pose",
    "retain_top_actual_contact_candidates",
    "run_actual_contact_grasp_pose_campaign",
    "screen_actual_contact_pose_cell",
    "select_dynamic_local_refinement_parents",
]
