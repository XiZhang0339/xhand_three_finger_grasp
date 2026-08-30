"""Deterministic upright-grasp rescue with active-finger collision auditing.

The schema-v14 lift search historically treated hand--hand contacts as
ordinary MuJoCo contacts.  That is appropriate for the sealed v1--v13 paths,
but it lets an otherwise good lift hide an index/middle collision.  This
module adds the stricter rule only to explicitly requested rescue runs:

* an authenticated grasp artifact supplies hand root pose, measured grasp
  qpos, precontact command, preload command and close timing;
* an authenticated manipulation artifact supplies the multi-knot plan,
  force targets and feedback controller;
* the index side-sway (``index_bend``) plan is scaled and hard limited;
* small deterministic root/joint perturbations form independent candidates;
* a session decorator records cross-finger contacts and turns any such
  contact into a hard evaluation failure.

Candidate execution is still delegated to
``run_or_resume_v14_candidate_artifacts``.  The decorator never changes the
normal :class:`~xhand_grasp.simulation.SimulationSession`, so legacy schemas
and existing v14 campaigns retain their exact traces and evaluation results.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np

from ..config import ACTIVE_ACTUATORS, validate_config
from ..experiment import ManipulationPlanParameters, resolve_experiment
from ..grasp_pose import canonical_sha256
from ..joint_pair_geometry import (
    JointPairBinding,
    joint_pair_telemetry,
    resolve_joint_pair,
)
from ..relative_wrist_pose import (
    rotation_matrix_to_rpy_degrees,
    rotvec_degrees_to_rotation_matrix,
)
from ..scene import rpy_degrees_to_rotation_matrix
from ..simulation import SimulationSession, SimulationStep
from ..v14_identity import (
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
    v14_time_warp_controller_id,
    v14_upright_rescue_resolved_payload_sha256,
)
from .contact_preserving_candidate_artifacts import (
    V14CandidateArtifactBundle,
    authenticate_v14_candidate_artifacts,
    run_or_resume_v14_candidate_artifacts,
)
from .contact_preserving_joint_refinement import (
    JointRefinementLimits,
    resolve_joint_refinement_limits,
)


UPRIGHT_GRASP_RESCUE_SCHEMA_VERSION = 3
DEFAULT_SEED = 20260821
MAXIMUM_CANDIDATE_COUNT = 256
EXPERIMENT_ID = "left_opposed_face_palm_down_contact_preserving_planned_lift"
INDEX_BEND_ACTUATOR = "left_hand_index_bend_joint_actuator"
INDEX_JOINT2_ACTUATOR = "left_hand_index_joint2_actuator"
SELF_COLLISION_CHECK = "no_active_finger_self_collision"
UPRIGHT_CLOSURE_CHECK = "index_middle_upright_closure_within_limit"
INDEX_BEND_EXCURSION_CHECK = "index_bend_operation_excursion_within_limit"
JOINT_PAIR_ALIGNMENT_CHECK = "index_middle_joint1_line_aligned_with_cube_y"
DEFAULT_UPRIGHT_CLOSURE_P95_MAX_DEG = 20.0
DEFAULT_INDEX_BEND_OPERATION_EXCURSION_MAX_RAD = 0.050
_ACTIVE_FINGERS = ("thumb", "index", "mid")
_OPERATION_STATES = frozenset(("MANIPULATE", "HOLD", "ABORT"))
_GRASP_STATES = frozenset(("SETTLE", "CLOSE", "VERIFY"))


def _require_finite_tuple(
    values: Sequence[float], length: int, label: str, *, nonnegative: bool = False
) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain {length} finite values")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{label} must contain non-negative values")
    return tuple(float(value) for value in array)


@dataclass(frozen=True, slots=True)
class UprightGraspRescueBudget:
    """Bounded deterministic search around one geometry/plan pairing."""

    candidate_count: int = 64
    seed: int = DEFAULT_SEED
    grasp_geometry_blend_fraction: float = 1.0
    use_grasp_close_timing: bool = True
    index_joint2_plan_scale: float = 1.0
    fixed_wrist_local_rotvec_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixed_nominal_offset_rad: tuple[float, ...] = (0.0,) * len(ACTIVE_ACTUATORS)
    fixed_precontact_residual_rad: tuple[float, ...] = (0.0,) * len(ACTIVE_ACTUATORS)
    fixed_preload_residual_rad: tuple[float, ...] = (0.0,) * len(ACTIVE_ACTUATORS)
    index_bend_plan_scale_range: tuple[float, float] = (0.15, 0.35)
    index_bend_max_abs_delta_rad: float = 0.025
    root_translation_radius_cube_m: tuple[float, float, float] = (
        0.003,
        0.004,
        0.003,
    )
    wrist_local_rotvec_radius_deg: tuple[float, float, float] = (2.0, 2.0, 3.0)
    wrist_local_rotvec_norm_limit_deg: float = 4.0
    nominal_qpos_radius_rad: float = 0.005
    index_bend_nominal_radius_rad: float = 0.040
    precontact_residual_radius_rad: float = 0.003
    preload_residual_radius_rad: float = 0.005

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_count, bool)
            or not 1 <= int(self.candidate_count) <= MAXIMUM_CANDIDATE_COUNT
        ):
            raise ValueError(
                f"candidate_count must lie within [1, {MAXIMUM_CANDIDATE_COUNT}]"
            )
        if isinstance(self.seed, bool) or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        blend = float(self.grasp_geometry_blend_fraction)
        if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
            raise ValueError("grasp_geometry_blend_fraction must lie within [0, 1]")
        object.__setattr__(self, "grasp_geometry_blend_fraction", blend)
        if not isinstance(self.use_grasp_close_timing, bool):
            raise TypeError("use_grasp_close_timing must be boolean")
        index_joint2_scale = float(self.index_joint2_plan_scale)
        if not math.isfinite(index_joint2_scale) or not 0.5 <= index_joint2_scale <= 1.25:
            raise ValueError("index_joint2_plan_scale must lie within [0.5, 1.25]")
        object.__setattr__(self, "index_joint2_plan_scale", index_joint2_scale)
        scales = _require_finite_tuple(
            self.index_bend_plan_scale_range,
            2,
            "index_bend_plan_scale_range",
            nonnegative=True,
        )
        if scales[0] > scales[1] or scales[1] > 1.0:
            raise ValueError("index bend scale range must be ordered within [0, 1]")
        object.__setattr__(self, "index_bend_plan_scale_range", scales)
        for name in (
            "root_translation_radius_cube_m",
            "wrist_local_rotvec_radius_deg",
        ):
            object.__setattr__(
                self,
                name,
                _require_finite_tuple(
                    getattr(self, name), 3, name, nonnegative=True
                ),
            )
        object.__setattr__(
            self,
            "fixed_wrist_local_rotvec_deg",
            _require_finite_tuple(
                self.fixed_wrist_local_rotvec_deg,
                3,
                "fixed_wrist_local_rotvec_deg",
            ),
        )
        for name in (
            "fixed_nominal_offset_rad",
            "fixed_precontact_residual_rad",
            "fixed_preload_residual_rad",
        ):
            object.__setattr__(
                self,
                name,
                _require_finite_tuple(
                    getattr(self, name), len(ACTIVE_ACTUATORS), name
                ),
            )
        for name in (
            "index_bend_max_abs_delta_rad",
            "wrist_local_rotvec_norm_limit_deg",
            "nominal_qpos_radius_rad",
            "index_bend_nominal_radius_rad",
            "precontact_residual_radius_rad",
            "preload_residual_radius_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if (
            np.linalg.norm(self.fixed_wrist_local_rotvec_deg)
            > self.wrist_local_rotvec_norm_limit_deg + 1e-12
        ):
            raise ValueError(
                "fixed_wrist_local_rotvec_deg exceeds the wrist rotation norm limit"
            )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": UPRIGHT_GRASP_RESCUE_SCHEMA_VERSION,
            "candidate_count": int(self.candidate_count),
            "maximum_candidate_count": MAXIMUM_CANDIDATE_COUNT,
            "seed": int(self.seed),
            "grasp_geometry_blend_fraction": (
                self.grasp_geometry_blend_fraction
            ),
            "use_grasp_close_timing": self.use_grasp_close_timing,
            "index_joint2_plan_scale": self.index_joint2_plan_scale,
            "fixed_wrist_local_rotvec_deg": list(
                self.fixed_wrist_local_rotvec_deg
            ),
            "fixed_nominal_offset_rad": list(self.fixed_nominal_offset_rad),
            "fixed_precontact_residual_rad": list(
                self.fixed_precontact_residual_rad
            ),
            "fixed_preload_residual_rad": list(
                self.fixed_preload_residual_rad
            ),
            "active_actuator_order": list(ACTIVE_ACTUATORS),
            "index_bend_plan_scale_range": list(
                self.index_bend_plan_scale_range
            ),
            "index_bend_max_abs_delta_rad": self.index_bend_max_abs_delta_rad,
            "root_translation_radius_cube_m": list(
                self.root_translation_radius_cube_m
            ),
            "wrist_local_rotvec_radius_deg": list(
                self.wrist_local_rotvec_radius_deg
            ),
            "wrist_local_rotvec_norm_limit_deg": (
                self.wrist_local_rotvec_norm_limit_deg
            ),
            "nominal_qpos_radius_rad": self.nominal_qpos_radius_rad,
            "index_bend_nominal_radius_rad": self.index_bend_nominal_radius_rad,
            "precontact_residual_radius_rad": (
                self.precontact_residual_radius_rad
            ),
            "preload_residual_radius_rad": self.preload_residual_radius_rad,
        }


@dataclass(frozen=True, slots=True)
class UprightGraspRescueSource:
    """Authenticated grasp geometry and manipulation-controller seed."""

    grasp_root: Path
    plan_root: Path
    grasp_candidate_id: int
    plan_candidate_id: int
    grasp_config: dict[str, Any]
    plan_config: dict[str, Any]
    grasp_result_semantic_sha256: str
    plan_result_semantic_sha256: str
    grasp_trace_sha256: str
    plan_trace_sha256: str
    source_authentication_id: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": UPRIGHT_GRASP_RESCUE_SCHEMA_VERSION,
            "grasp_root": str(self.grasp_root),
            "plan_root": str(self.plan_root),
            "grasp_candidate_id": int(self.grasp_candidate_id),
            "plan_candidate_id": int(self.plan_candidate_id),
            "grasp_config_sha256": canonical_sha256(self.grasp_config),
            "plan_config_sha256": canonical_sha256(self.plan_config),
            "grasp_result_semantic_sha256": self.grasp_result_semantic_sha256,
            "plan_result_semantic_sha256": self.plan_result_semantic_sha256,
            "grasp_trace_sha256": self.grasp_trace_sha256,
            "plan_trace_sha256": self.plan_trace_sha256,
            "source_authentication_id": self.source_authentication_id,
        }


def _load_bundle_config(bundle: V14CandidateArtifactBundle) -> dict[str, Any]:
    value = json.loads(bundle.config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("authenticated candidate config is not a JSON object")
    return value


def _bundle_trace_sha(bundle: V14CandidateArtifactBundle) -> str:
    value = bundle.result.get("artifacts", {}).get("sha256", {}).get("trace")
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError("authenticated rescue source lost its retained trace digest")
    return value


def _require_compatible_sources(
    grasp_config: Mapping[str, Any], plan_config: Mapping[str, Any]
) -> None:
    if grasp_config.get("experiment_id") != EXPERIMENT_ID or plan_config.get(
        "experiment_id"
    ) != EXPERIMENT_ID:
        raise ValueError("upright rescue requires two registered v14 experiment sources")
    for key in (
        "cube",
        "scene",
        "contact_topology",
        "contact_point_plan",
        "acceptance",
        "pose_constraints",
        "motion_smoothness",
    ):
        if canonical_sha256(grasp_config.get(key)) != canonical_sha256(
            plan_config.get(key)
        ):
            raise ValueError(f"upright rescue sources disagree on fixed field {key}")


def authenticate_upright_grasp_rescue_source(
    grasp_candidate_root: str | Path,
    plan_candidate_root: str | Path,
) -> UprightGraspRescueSource:
    """Authenticate and bind one grasp-geometry / manipulation-plan pair."""

    grasp_bundle = authenticate_v14_candidate_artifacts(
        grasp_candidate_root, require_retained_trace=True
    )
    plan_bundle = authenticate_v14_candidate_artifacts(
        plan_candidate_root, require_retained_trace=True
    )
    if grasp_bundle.result.get("grasp_success") is not True:
        raise RuntimeError("grasp geometry source is not a measured grasp success")
    if plan_bundle.result.get("grasp_success") is not True:
        raise RuntimeError("manipulation source is not a measured grasp success")
    grasp_config = _load_bundle_config(grasp_bundle)
    plan_config = _load_bundle_config(plan_bundle)
    _require_compatible_sources(grasp_config, plan_config)
    payload = {
        "schema_version": UPRIGHT_GRASP_RESCUE_SCHEMA_VERSION,
        "kind": "authenticated_upright_grasp_self_collision_rescue_sources",
        "grasp_candidate_id": grasp_bundle.candidate_id,
        "plan_candidate_id": plan_bundle.candidate_id,
        "grasp_config_sha256": canonical_sha256(grasp_config),
        "plan_config_sha256": canonical_sha256(plan_config),
        "grasp_result_semantic_sha256": grasp_bundle.result[
            "result_semantic_sha256"
        ],
        "plan_result_semantic_sha256": plan_bundle.result[
            "result_semantic_sha256"
        ],
        "grasp_trace_sha256": _bundle_trace_sha(grasp_bundle),
        "plan_trace_sha256": _bundle_trace_sha(plan_bundle),
    }
    return UprightGraspRescueSource(
        grasp_root=grasp_bundle.destination,
        plan_root=plan_bundle.destination,
        grasp_candidate_id=grasp_bundle.candidate_id,
        plan_candidate_id=plan_bundle.candidate_id,
        grasp_config=grasp_config,
        plan_config=plan_config,
        grasp_result_semantic_sha256=str(
            grasp_bundle.result["result_semantic_sha256"]
        ),
        plan_result_semantic_sha256=str(
            plan_bundle.result["result_semantic_sha256"]
        ),
        grasp_trace_sha256=_bundle_trace_sha(grasp_bundle),
        plan_trace_sha256=_bundle_trace_sha(plan_bundle),
        source_authentication_id=canonical_sha256(payload),
    )


def _interpolate_actuator_mapping(
    first: Mapping[str, Any], second: Mapping[str, Any], fraction: float
) -> dict[str, float]:
    if set(first) != set(second):
        raise ValueError("upright rescue source actuator mappings disagree")
    return {
        name: (1.0 - fraction) * float(first[name])
        + fraction * float(second[name])
        for name in first
    }


def _base_combined_config(
    source: UprightGraspRescueSource,
    *,
    grasp_geometry_blend_fraction: float = 1.0,
    use_grasp_close_timing: bool = True,
) -> dict[str, Any]:
    """Combine only the declared geometry and controller domains."""

    grasp = source.grasp_config
    config = copy.deepcopy(source.plan_config)
    fraction = float(grasp_geometry_blend_fraction)
    config["hand_pose"] = {
        key: (
            (1.0 - fraction)
            * np.asarray(source.plan_config["hand_pose"][key], dtype=np.float64)
            + fraction * np.asarray(grasp["hand_pose"][key], dtype=np.float64)
        ).tolist()
        for key in ("translation_m", "rpy_deg")
    }
    config["grasp_pose"] = copy.deepcopy(source.plan_config["grasp_pose"])
    config["grasp_pose"]["nominal_joint_qpos_rad"] = (
        _interpolate_actuator_mapping(
            source.plan_config["grasp_pose"]["nominal_joint_qpos_rad"],
            grasp["grasp_pose"]["nominal_joint_qpos_rad"],
            fraction,
        )
    )
    for key in ("precontact_targets_rad", "contact_preload_targets_rad"):
        config["control"][key] = _interpolate_actuator_mapping(
            source.plan_config["control"][key],
            grasp["control"][key],
            fraction,
        )
    if use_grasp_close_timing and "close_profile" in grasp["control"]:
        config["control"]["close_profile"] = copy.deepcopy(
            grasp["control"]["close_profile"]
        )
    if use_grasp_close_timing:
        config["control_protocol"]["close_s"] = float(
            grasp["control_protocol"]["close_s"]
        )
    # Old rescue metadata can select a different canonical planner lineage.
    # Replace it with one self-contained provenance block below.
    config["candidate_metadata"] = {
        "schema_version": 14,
        "cube_pose_sampled": False,
        "hand_root_fixed_during_simulation": True,
    }
    for key in (
        "object_config_id",
        "grasp_pose_id",
        "grasp_object_pair_id",
        "planner_id",
        "controller_id",
        "experiment_status",
        "run_context",
    ):
        config.pop(key, None)
    return config


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    result = np.empty((count, dimensions), dtype=np.float64)
    for axis in range(dimensions):
        order = rng.permutation(count)
        result[:, axis] = (order + rng.random(count)) / count * 2.0 - 1.0
    return result


def _structured_samples(count: int, seed: int) -> tuple[np.ndarray, ...]:
    """Start with bend-scale ablations, then deterministic 31-D coverage."""

    result: list[np.ndarray] = []
    # Sample layout: root(3), rotation(3), nominal(8), precontact(8),
    # preload(8), bend-scale(1).  Zero geometry perturbations isolate the
    # known collision-causing plan axis first.
    for normalized_scale in (0.0, -1.0, -0.5, 0.5, 1.0):
        sample = np.zeros(31, dtype=np.float64)
        sample[-1] = normalized_scale
        result.append(sample)
    if count > len(result):
        result.extend(_latin_hypercube(count - len(result), 31, seed))
    return tuple(value.copy() for value in result[:count])


def _rotation_matrix_to_rotvec_degrees(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    sine = math.sin(angle)
    if abs(sine) <= 1e-10:
        raise ValueError("upright rescue rotation is numerically singular")
    axis = np.asarray(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ),
        dtype=np.float64,
    ) / (2.0 * sine)
    return np.degrees(axis * angle)


def _preserve_closing_rays(
    preload: dict[str, float],
    precontact: dict[str, float],
    base_preload: Mapping[str, float],
    base_precontact: Mapping[str, float],
) -> None:
    for names in (
        ACTIVE_ACTUATORS[0:3],
        ACTIVE_ACTUATORS[3:6],
        ACTIVE_ACTUATORS[6:8],
    ):
        original = np.asarray(
            [float(base_preload[name]) - float(base_precontact[name]) for name in names]
        )
        proposed = np.asarray(
            [float(preload[name]) - float(precontact[name]) for name in names]
        )
        if float(original @ proposed) <= 0.0 or np.linalg.norm(proposed) <= 1e-6:
            for name in names:
                preload[name] = float(base_preload[name])
                precontact[name] = float(base_precontact[name])


def _scaled_plan(
    plan_config: Mapping[str, Any],
    *,
    scale: float,
    maximum_abs_delta_rad: float,
    index_joint2_scale: float,
) -> ManipulationPlanParameters:
    original = ManipulationPlanParameters.from_config(plan_config)
    waypoints = {
        name: tuple(float(value) for value in original.actuator_waypoints_rad[name])
        for name in ACTIVE_ACTUATORS
    }
    waypoints[INDEX_BEND_ACTUATOR] = tuple(
        float(
            np.clip(
                float(value) * float(scale),
                -float(maximum_abs_delta_rad),
                float(maximum_abs_delta_rad),
            )
        )
        for value in original.actuator_waypoints_rad[INDEX_BEND_ACTUATOR]
    )
    waypoints[INDEX_JOINT2_ACTUATOR] = tuple(
        float(value) * float(index_joint2_scale)
        for value in original.actuator_waypoints_rad[INDEX_JOINT2_ACTUATOR]
    )
    return ManipulationPlanParameters(
        schema_version=original.schema_version,
        profile=original.profile,
        duration_s=original.duration_s,
        knot_times_s=original.knot_times_s,
        actuator_waypoints_rad=waypoints,
        desired_cube_position_delta_m=original.desired_cube_position_delta_m,
        desired_cube_rotation_vector_rad=(
            original.desired_cube_rotation_vector_rad
        ),
        max_knot_delta_rad=original.max_knot_delta_rad,
        trust_region_backtracks=original.trust_region_backtracks,
    )


def _candidate_identity(
    source: UprightGraspRescueSource,
    budget: UprightGraspRescueBudget,
    sequence: int,
    normalized_sample: np.ndarray,
) -> tuple[int, str]:
    digest = canonical_sha256(
        {
            "schema_version": UPRIGHT_GRASP_RESCUE_SCHEMA_VERSION,
            "source_authentication_id": source.source_authentication_id,
            "budget": budget.as_mapping(),
            "sequence": int(sequence),
            "normalized_sample": normalized_sample.tolist(),
        }
    )
    identifier = 15_800_000_000_000_000 + int(digest[:13], 16) % 100_000_000_000_000
    return identifier, digest


def _materialize_candidate(
    source: UprightGraspRescueSource,
    budget: UprightGraspRescueBudget,
    sequence: int,
    normalized: np.ndarray,
    limits: JointRefinementLimits,
    *,
    validate: bool,
) -> dict[str, Any]:
    config = _base_combined_config(
        source,
        grasp_geometry_blend_fraction=budget.grasp_geometry_blend_fraction,
        use_grasp_close_timing=budget.use_grasp_close_timing,
    )
    fixed_cube_sha = canonical_sha256(config["cube"])
    base_hand = copy.deepcopy(config["hand_pose"])
    base_nominal = copy.deepcopy(config["grasp_pose"]["nominal_joint_qpos_rad"])
    base_precontact = copy.deepcopy(config["control"]["precontact_targets_rad"])
    base_preload = copy.deepcopy(config["control"]["contact_preload_targets_rad"])

    translation_delta_cube = normalized[0:3] * np.asarray(
        budget.root_translation_radius_cube_m
    )
    cube_rotation = rpy_degrees_to_rotation_matrix(config["cube"]["rpy_deg"])
    base_translation = np.asarray(base_hand["translation_m"], dtype=np.float64)
    config["hand_pose"]["translation_m"] = (
        base_translation + cube_rotation @ translation_delta_cube
    ).tolist()

    local_rotvec = np.asarray(
        budget.fixed_wrist_local_rotvec_deg, dtype=np.float64
    ) + normalized[3:6] * np.asarray(budget.wrist_local_rotvec_radius_deg)
    norm = float(np.linalg.norm(local_rotvec))
    if norm > budget.wrist_local_rotvec_norm_limit_deg:
        local_rotvec *= budget.wrist_local_rotvec_norm_limit_deg / norm
    base_rpy = np.asarray(base_hand["rpy_deg"], dtype=np.float64)
    base_rotation = rpy_degrees_to_rotation_matrix(base_rpy)
    if float(np.linalg.norm(local_rotvec)) <= 1e-15:
        proposed_rpy = base_rpy.copy()
        effective_rotvec = np.zeros(3, dtype=np.float64)
    else:
        proposed_rotation = base_rotation @ rotvec_degrees_to_rotation_matrix(
            local_rotvec
        )
        proposed_rpy = rotation_matrix_to_rpy_degrees(
            proposed_rotation, reference_rpy_deg=base_rpy
        )
        definition = resolve_experiment(config)
        proposed_rpy[0] = np.clip(
            proposed_rpy[0], *definition.search_bounds.hand_roll_deg
        )
        proposed_rpy[2] = np.clip(
            proposed_rpy[2], *definition.search_bounds.hand_yaw_deg
        )
        effective_rotation = rpy_degrees_to_rotation_matrix(proposed_rpy)
        effective_rotvec = _rotation_matrix_to_rotvec_degrees(
            base_rotation.T @ effective_rotation
        )
        effective_norm = float(np.linalg.norm(effective_rotvec))
        if effective_norm > budget.wrist_local_rotvec_norm_limit_deg:
            effective_rotvec *= (
                budget.wrist_local_rotvec_norm_limit_deg / effective_norm
            )
            effective_rotation = base_rotation @ rotvec_degrees_to_rotation_matrix(
                effective_rotvec
            )
            proposed_rpy = rotation_matrix_to_rpy_degrees(
                effective_rotation, reference_rpy_deg=base_rpy
            )
    config["hand_pose"]["rpy_deg"] = proposed_rpy.tolist()

    nominal_units = normalized[6:14]
    precontact_units = normalized[14:22]
    preload_units = normalized[22:30]
    nominal: dict[str, float] = {}
    precontact: dict[str, float] = {}
    preload: dict[str, float] = {}
    for index, name in enumerate(ACTIVE_ACTUATORS):
        nominal_radius = (
            budget.index_bend_nominal_radius_rad
            if name == INDEX_BEND_ACTUATOR
            else budget.nominal_qpos_radius_rad
        )
        low, high = limits.preload_target_rad[name]
        if name == "left_hand_thumb_bend_joint_actuator":
            low, high = max(low, 1.40), min(high, 1.60)
        nominal[name] = float(
            np.clip(
                float(base_nominal[name])
                + float(budget.fixed_nominal_offset_rad[index])
                + nominal_units[index] * nominal_radius,
                low,
                high,
            )
        )
        nominal_delta = nominal[name] - float(base_nominal[name])
        command_low, command_high = limits.command_target_rad[name]
        precontact[name] = float(
            np.clip(
                float(base_precontact[name])
                + nominal_delta
                + float(budget.fixed_precontact_residual_rad[index])
                + precontact_units[index] * budget.precontact_residual_radius_rad,
                command_low,
                command_high,
            )
        )
        preload[name] = float(
            np.clip(
                float(base_preload[name])
                + nominal_delta
                + float(budget.fixed_preload_residual_rad[index])
                + preload_units[index] * budget.preload_residual_radius_rad,
                low,
                high,
            )
        )
    _preserve_closing_rays(preload, precontact, base_preload, base_precontact)
    config["grasp_pose"]["nominal_joint_qpos_rad"] = nominal
    config["control"]["precontact_targets_rad"] = precontact
    config["control"]["contact_preload_targets_rad"] = preload

    lower_scale, upper_scale = budget.index_bend_plan_scale_range
    scale = lower_scale + (float(normalized[-1]) + 1.0) * 0.5 * (
        upper_scale - lower_scale
    )
    plan = _scaled_plan(
        source.plan_config["manipulation_plan"],
        scale=scale,
        maximum_abs_delta_rad=budget.index_bend_max_abs_delta_rad,
        index_joint2_scale=budget.index_joint2_plan_scale,
    )
    config["manipulation_plan"] = plan.as_config()
    config["control"]["manipulation_delta_rad"] = {
        name: float(plan.actuator_waypoints_rad[name][-1])
        for name in ACTIVE_ACTUATORS
    }

    candidate_id, candidate_sha = _candidate_identity(
        source, budget, sequence, normalized
    )
    config["object_config_id"] = v14_object_config_id(config)
    config["grasp_pose_id"] = v14_grasp_pose_id(config)
    config["grasp_object_pair_id"] = v14_grasp_object_pair_id(config)
    source_plan_id = str(source.plan_config["manipulation_plan"]["plan_id"])
    planner_payload = {
        "schema_version": UPRIGHT_GRASP_RESCUE_SCHEMA_VERSION,
        "kind": "v14_upright_grasp_self_collision_rescue_plan",
        "source_authentication_id": source.source_authentication_id,
        "grasp_source_candidate_id": source.grasp_candidate_id,
        "plan_source_candidate_id": source.plan_candidate_id,
        "source_plan_id": source_plan_id,
        "resolved_plan_id": plan.plan_id,
        "grasp_object_pair_id": config["grasp_object_pair_id"],
        "candidate_sha256": candidate_sha,
    }
    resolved_payload_sha = v14_upright_rescue_resolved_payload_sha256(config)
    planner_payload["resolved_candidate_payload_sha256"] = resolved_payload_sha
    config["planner_id"] = canonical_sha256(planner_payload)
    config["controller_id"] = v14_time_warp_controller_id(config)
    metadata = config["candidate_metadata"]
    metadata["candidate_id"] = candidate_id
    metadata["v14_upright_grasp_self_collision_rescue"] = {
        "schema_version": UPRIGHT_GRASP_RESCUE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha,
        "sequence": int(sequence),
        "source_authentication_id": source.source_authentication_id,
        "grasp_source_candidate_id": source.grasp_candidate_id,
        "plan_source_candidate_id": source.plan_candidate_id,
        "source_plan_id": source_plan_id,
        "budget": budget.as_mapping(),
        "normalized_sample": normalized.tolist(),
        "resolved_candidate_payload_sha256": resolved_payload_sha,
        "root_translation_delta_cube_m": translation_delta_cube.tolist(),
        "wrist_local_rotvec_deg": effective_rotvec.tolist(),
        "index_bend_plan_scale": float(scale),
        "grasp_geometry_blend_fraction": (
            budget.grasp_geometry_blend_fraction
        ),
        "use_grasp_close_timing": budget.use_grasp_close_timing,
        "index_bend_max_abs_delta_rad": (
            budget.index_bend_max_abs_delta_rad
        ),
        "nominal_qpos_delta_rad": {
            name: nominal[name] - float(base_nominal[name])
            for name in ACTIVE_ACTUATORS
        },
        "precontact_residual_rad": {
            name: precontact[name]
            - float(base_precontact[name])
            - (nominal[name] - float(base_nominal[name]))
            for name in ACTIVE_ACTUATORS
        },
        "preload_residual_rad": {
            name: preload[name]
            - float(base_preload[name])
            - (nominal[name] - float(base_nominal[name]))
            for name in ACTIVE_ACTUATORS
        },
        "active_finger_self_collision_is_hard_failure": True,
        "upright_closure_p95_max_deg": DEFAULT_UPRIGHT_CLOSURE_P95_MAX_DEG,
        "index_bend_operation_excursion_max_rad": (
            DEFAULT_INDEX_BEND_OPERATION_EXCURSION_MAX_RAD
        ),
        "cube_pose_sampled": False,
        "full_reset_required": True,
    }
    if canonical_sha256(config["cube"]) != fixed_cube_sha:
        raise AssertionError("upright rescue changed the cube")
    if validate:
        validate_config(config)
    return {
        "upright_grasp_rescue_job_schema_version": (
            UPRIGHT_GRASP_RESCUE_SCHEMA_VERSION
        ),
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha,
        "sequence": int(sequence),
        "config": config,
        "job_metadata": copy.deepcopy(
            metadata["v14_upright_grasp_self_collision_rescue"]
        ),
    }


def build_upright_grasp_rescue_jobs(
    source: UprightGraspRescueSource,
    *,
    budget: UprightGraspRescueBudget = UprightGraspRescueBudget(),
    validate_configs: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Create worker-order-independent candidate configs for the rescue."""

    _require_compatible_sources(source.grasp_config, source.plan_config)
    base = _base_combined_config(
        source,
        grasp_geometry_blend_fraction=budget.grasp_geometry_blend_fraction,
        use_grasp_close_timing=budget.use_grasp_close_timing,
    )
    limits = resolve_joint_refinement_limits(base)
    jobs_list: list[dict[str, Any]] = []
    for sequence, raw_sample in enumerate(
        _structured_samples(int(budget.candidate_count), int(budget.seed))
    ):
        accepted: dict[str, Any] | None = None
        # A certified source can lie close to a registered palm/tilt or
        # root-distance boundary.  Keep the sampled joint/controller values,
        # but deterministically retract only the 6-D root perturbation until
        # the registered pose contract is restored.
        for pose_scale in (1.0, 0.75, 0.5, 0.25, 0.125, 0.0):
            normalized = raw_sample.copy()
            normalized[:6] *= pose_scale
            candidate = _materialize_candidate(
                source,
                budget,
                sequence,
                normalized,
                limits,
                validate=False,
            )
            if validate_configs:
                try:
                    validate_config(candidate["config"])
                except ValueError as error:
                    message = str(error)
                    if not any(
                        token in message
                        for token in (
                            "pose constraints",
                            "root_cube_distance",
                            "hand roll",
                            "hand yaw",
                        )
                    ):
                        raise
                    continue
            candidate["job_metadata"]["root_pose_retraction_scale"] = float(
                pose_scale
            )
            candidate["config"]["candidate_metadata"][
                "v14_upright_grasp_self_collision_rescue"
            ]["root_pose_retraction_scale"] = float(pose_scale)
            accepted = candidate
            break
        if accepted is None:
            raise RuntimeError(
                "upright rescue could not retract a sampled root pose into bounds"
            )
        jobs_list.append(accepted)
    jobs = tuple(jobs_list)
    identifiers = [int(value["candidate_id"]) for value in jobs]
    hashes = [canonical_sha256(value["config"]) for value in jobs]
    if len(set(identifiers)) != len(identifiers) or len(set(hashes)) != len(hashes):
        raise RuntimeError("upright rescue generated duplicate candidates")
    return jobs


@dataclass(frozen=True, slots=True)
class ActiveFingerCollisionContact:
    first_finger: str
    second_finger: str
    first_geom: str
    second_geom: str
    normal_force_n: float
    penetration_m: float

    @property
    def pair_label(self) -> str:
        return f"{self.first_geom}|{self.second_geom}"


@dataclass(frozen=True, slots=True)
class ActiveFingerSelfCollisionStep:
    contacts: tuple[ActiveFingerCollisionContact, ...] = ()

    @property
    def active(self) -> bool:
        return bool(self.contacts)

    @property
    def total_normal_force_n(self) -> float:
        return float(sum(value.normal_force_n for value in self.contacts))

    @property
    def max_penetration_m(self) -> float:
        return float(max((value.penetration_m for value in self.contacts), default=0.0))


def _active_finger_from_geom_name(name: str) -> str | None:
    for finger in _ACTIVE_FINGERS:
        if f"left_hand_{finger}_" in name:
            return finger
    return None


def active_finger_self_collision_snapshot(
    model: mujoco.MjModel, data: mujoco.MjData
) -> ActiveFingerSelfCollisionStep:
    """Return all active contacts between two different controlled fingers."""

    contacts: list[ActiveFingerCollisionContact] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        first_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
        ) or ""
        second_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
        ) or ""
        first_finger = _active_finger_from_geom_name(first_name)
        second_finger = _active_finger_from_geom_name(second_name)
        if (
            first_finger is None
            or second_finger is None
            or first_finger == second_finger
        ):
            continue
        force = np.zeros(6, dtype=np.float64)
        if int(contact.efc_address) >= 0:
            mujoco.mj_contactForce(model, data, contact_index, force)
        normal_force = max(0.0, float(force[0]))
        penetration = max(0.0, -float(contact.dist))
        if normal_force <= 1e-8 and penetration <= 0.0:
            continue
        if (first_finger, first_name) > (second_finger, second_name):
            first_finger, second_finger = second_finger, first_finger
            first_name, second_name = second_name, first_name
        contacts.append(
            ActiveFingerCollisionContact(
                first_finger=first_finger,
                second_finger=second_finger,
                first_geom=first_name,
                second_geom=second_name,
                normal_force_n=normal_force,
                penetration_m=penetration,
            )
        )
    contacts.sort(key=lambda value: value.pair_label)
    return ActiveFingerSelfCollisionStep(tuple(contacts))


class _SessionLike(Protocol):
    model: Any
    data: Any

    @property
    def complete(self) -> bool: ...

    def advance_one(self) -> Any: ...

    def finalize(self, *, trace_path: str | Path | None = None) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


CollisionDetector = Callable[[_SessionLike], ActiveFingerSelfCollisionStep]


def _default_collision_detector(
    session: _SessionLike,
) -> ActiveFingerSelfCollisionStep:
    return active_finger_self_collision_snapshot(session.model, session.data)


def _longest_true_run(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


class ActiveFingerSelfCollisionAuditedSession:
    """Session decorator that makes cross-active-finger collision a hard check."""

    def __init__(
        self,
        inner: _SessionLike,
        *,
        detector: CollisionDetector = _default_collision_detector,
    ) -> None:
        self.inner = inner
        self.detector = detector
        self._steps: list[ActiveFingerSelfCollisionStep] = []
        self._states: list[str] = []
        metadata = getattr(inner, "config", {}).get("candidate_metadata", {})
        alignment = (
            metadata.get("v14_index_middle_joint_pair_alignment_refinement")
            if isinstance(metadata, Mapping)
            else None
        )
        self._joint_pair_alignment = (
            copy.deepcopy(dict(alignment))
            if isinstance(alignment, Mapping)
            else None
        )
        self._joint_pair_binding: JointPairBinding | None = None
        self._joint_pair_vectors: list[np.ndarray] = []
        self._joint_pair_angles: list[float] = []
        if self._joint_pair_alignment is not None:
            requested = self._joint_pair_alignment.get(
                "joint_names",
                ("left_hand_index_joint1", "left_hand_mid_joint1"),
            )
            self._joint_pair_binding = resolve_joint_pair(inner.model, requested)
            if self._joint_pair_binding is None:  # pragma: no cover - guarded input.
                raise ValueError("joint-pair alignment requires two named joints")
        self._summary: dict[str, Any] | None = None

    @property
    def complete(self) -> bool:
        return self.inner.complete

    def __getattr__(self, name: str) -> Any:
        """Expose the wrapped physics session to Viewer/display consumers."""

        return getattr(self.inner, name)

    def reset(self) -> None:
        self.inner.reset()
        self._steps.clear()
        self._states.clear()
        self._joint_pair_vectors.clear()
        self._joint_pair_angles.clear()
        self._summary = None

    def advance_one(self) -> Any:
        result = self.inner.advance_one()
        snapshot = self.detector(self.inner)
        if not isinstance(snapshot, ActiveFingerSelfCollisionStep):
            raise TypeError("self-collision detector returned the wrong type")
        self._steps.append(snapshot)
        state = getattr(result, "control_state", None)
        self._states.append("" if state is None else str(state))
        if self._joint_pair_binding is not None:
            pair = joint_pair_telemetry(self.inner.data, self._joint_pair_binding)
            self._joint_pair_vectors.append(
                np.asarray(pair["vector_cube_m"], dtype=np.float64).copy()
            )
            self._joint_pair_angles.append(float(pair["angle_to_cube_y_deg"]))
        return result

    def _install_trace_fields(self) -> None:
        traces = getattr(self.inner, "traces", None)
        if not isinstance(traces, dict):
            return
        traces["active_finger_self_collision"] = np.asarray(
            [value.active for value in self._steps], dtype=bool
        )
        traces["active_finger_self_collision_contact_count"] = np.asarray(
            [len(value.contacts) for value in self._steps], dtype=np.int64
        )
        traces["active_finger_self_collision_normal_force_n"] = np.asarray(
            [value.total_normal_force_n for value in self._steps], dtype=np.float64
        )
        traces["active_finger_self_collision_max_penetration_m"] = np.asarray(
            [value.max_penetration_m for value in self._steps], dtype=np.float64
        )
        traces["active_finger_self_collision_pairs"] = np.asarray(
            [";".join(contact.pair_label for contact in value.contacts) for value in self._steps],
            dtype=np.str_,
        )
        if self._joint_pair_binding is not None:
            traces["index_middle_joint1_line_cube_m"] = np.asarray(
                self._joint_pair_vectors, dtype=np.float64
            )
            traces["index_middle_joint1_line_angle_to_cube_y_deg"] = np.asarray(
                self._joint_pair_angles, dtype=np.float64
            )

    def _augment_summary(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        summary = copy.deepcopy(dict(raw))
        active = np.asarray([value.active for value in self._steps], dtype=bool)
        collision_free = not bool(np.any(active))
        model = getattr(self.inner, "model", None)
        option = getattr(model, "opt", None)
        timestep = float(getattr(option, "timestep", 0.0))
        pair_counts: dict[str, int] = {}
        state_counts: dict[str, int] = {}
        for state, snapshot in zip(self._states, self._steps):
            if snapshot.active:
                state_counts[state] = state_counts.get(state, 0) + 1
            for pair_label in {
                contact.pair_label for contact in snapshot.contacts
            }:
                pair_counts[pair_label] = pair_counts.get(pair_label, 0) + 1
        first_step = int(np.flatnonzero(active)[0]) if np.any(active) else -1
        metrics = summary.setdefault("metrics", {})
        metrics["active_finger_self_collision"] = {
            "collision_free": collision_free,
            "sample_count": len(self._steps),
            "collision_frame_count": int(np.count_nonzero(active)),
            "collision_duty": float(np.mean(active)) if active.size else 0.0,
            "first_collision_step": first_step,
            "longest_consecutive_collision_steps": _longest_true_run(active),
            "longest_consecutive_collision_s": (
                _longest_true_run(active) * timestep
            ),
            "maximum_total_normal_force_n": float(
                max((value.total_normal_force_n for value in self._steps), default=0.0)
            ),
            "maximum_penetration_m": float(
                max((value.max_penetration_m for value in self._steps), default=0.0)
            ),
            "pair_frame_counts": dict(sorted(pair_counts.items())),
            "state_frame_counts": dict(sorted(state_counts.items())),
        }
        checks = summary.setdefault("checks", {})
        checks[SELF_COLLISION_CHECK] = collision_free
        metadata = getattr(self.inner, "config", {}).get(
            "candidate_metadata", {}
        )
        upright = (
            metadata.get("v14_upright_grasp_self_collision_rescue", {})
            if isinstance(metadata, Mapping)
            else {}
        )
        closure_limit = float(
            upright.get(
                "upright_closure_p95_max_deg",
                DEFAULT_UPRIGHT_CLOSURE_P95_MAX_DEG,
            )
        )
        closure = metrics.get("closure_alignment", {})
        per_finger = (
            closure.get("per_finger", {})
            if isinstance(closure, Mapping)
            else {}
        )
        try:
            closure_values = tuple(
                float(per_finger[finger]["angle_p95_deg"])
                for finger in ("index", "mid")
            )
            closure_aligned = bool(
                np.isfinite(closure_values).all()
                and max(closure_values) <= closure_limit + 1e-12
            )
        except (KeyError, TypeError, ValueError):
            closure_values = (math.inf, math.inf)
            closure_aligned = False
        metrics["upright_grasp"] = {
            "closure_p95_deg": {
                "index": closure_values[0],
                "mid": closure_values[1],
            },
            "closure_p95_max_deg": closure_limit,
            "closure_aligned": closure_aligned,
        }
        checks[UPRIGHT_CLOSURE_CHECK] = closure_aligned

        traces = getattr(self.inner, "traces", {})
        operation_indices = np.asarray(
            [state in _OPERATION_STATES for state in self._states], dtype=bool
        )
        bend_limit = float(
            upright.get(
                "index_bend_operation_excursion_max_rad",
                DEFAULT_INDEX_BEND_OPERATION_EXCURSION_MAX_RAD,
            )
        )
        bend_excursion = 0.0
        bend_evidence_available = False
        if (
            isinstance(traces, Mapping)
            and "joint_qpos" in traces
            and np.any(operation_indices)
        ):
            joint_qpos = np.asarray(traces["joint_qpos"], dtype=np.float64)
            model = getattr(self.inner, "model", None)
            if model is not None and joint_qpos.shape[0] == operation_indices.size:
                actuator_id = int(model.actuator(INDEX_BEND_ACTUATOR).id)
                values = joint_qpos[operation_indices, actuator_id]
                if values.size and np.isfinite(values).all():
                    bend_evidence_available = True
                    bend_excursion = float(np.max(np.abs(values - values[0])))
        bend_within_limit = bool(
            not bend_evidence_available
            or bend_excursion <= bend_limit + 1e-12
        )
        metrics["upright_grasp"][
            "index_bend_operation_excursion_rad"
        ] = bend_excursion
        metrics["upright_grasp"][
            "index_bend_operation_excursion_max_rad"
        ] = bend_limit
        metrics["upright_grasp"][
            "index_bend_operation_evidence_available"
        ] = bend_evidence_available
        checks[INDEX_BEND_EXCURSION_CHECK] = bend_within_limit
        pair_aligned = True
        if self._joint_pair_alignment is not None:
            traces = getattr(self.inner, "traces", {})
            start = int(np.asarray(traces.get("grasp_stable_window_start_step", -1)))
            end = int(np.asarray(traces.get("grasp_stable_window_end_step", -1)))
            lock = int(np.asarray(traces.get("grasp_lock_step", -1)))
            angles = np.asarray(self._joint_pair_angles, dtype=np.float64)
            window_valid = bool(
                0 <= start <= end < angles.size
                and lock == end
                and np.isfinite(angles[start : end + 1]).all()
            )
            grasp_values = angles[start : end + 1] if window_valid else np.empty(0)
            operation_mask = np.asarray(
                [state in _OPERATION_STATES for state in self._states], dtype=bool
            )
            operation_values = (
                angles[operation_mask]
                if angles.shape == operation_mask.shape and np.any(operation_mask)
                else np.empty(0)
            )
            grasp_limit = float(
                self._joint_pair_alignment.get("grasp_window_p95_max_deg", math.inf)
            )
            operation_limit = float(
                self._joint_pair_alignment.get("operation_p95_max_deg", math.inf)
            )
            source_p95 = float(
                self._joint_pair_alignment.get("source_grasp_window_p95_deg", math.inf)
            )
            minimum_improvement = float(
                self._joint_pair_alignment.get("minimum_improvement_deg", 0.0)
            )
            grasp_p95 = (
                float(np.percentile(grasp_values, 95))
                if grasp_values.size
                else math.inf
            )
            operation_p95 = (
                float(np.percentile(operation_values, 95))
                if operation_values.size
                else math.inf
            )
            pair_aligned = bool(
                window_valid
                and grasp_p95 <= grasp_limit + 1e-12
                and source_p95 - grasp_p95 >= minimum_improvement - 1e-12
                and operation_p95 <= operation_limit + 1e-12
            )
            metrics["index_middle_joint_pair_alignment"] = {
                "joint_names": list(
                    self._joint_pair_alignment.get("joint_names", ())
                ),
                "grasp_window_start_step": start,
                "grasp_window_end_step": end,
                "grasp_lock_step": lock,
                "grasp_window_sample_count": int(grasp_values.size),
                "grasp_window_angle_p50_deg": (
                    float(np.percentile(grasp_values, 50))
                    if grasp_values.size
                    else math.inf
                ),
                "grasp_window_angle_p95_deg": grasp_p95,
                "grasp_window_angle_max_deg": (
                    float(np.max(grasp_values)) if grasp_values.size else math.inf
                ),
                "grasp_lock_angle_deg": (
                    float(angles[lock]) if window_valid else math.inf
                ),
                "operation_angle_p95_deg": operation_p95,
                "operation_angle_max_deg": (
                    float(np.max(operation_values))
                    if operation_values.size
                    else math.inf
                ),
                "source_grasp_window_p95_deg": source_p95,
                "minimum_improvement_deg": minimum_improvement,
                "grasp_window_p95_max_deg": grasp_limit,
                "operation_p95_max_deg": operation_limit,
                "aligned": pair_aligned,
            }
            checks[JOINT_PAIR_ALIGNMENT_CHECK] = pair_aligned
        failed = [str(value) for value in summary.get("failed_checks", ())]
        for check, passed in (
            (SELF_COLLISION_CHECK, collision_free),
            (UPRIGHT_CLOSURE_CHECK, closure_aligned),
            (INDEX_BEND_EXCURSION_CHECK, bend_within_limit),
            (JOINT_PAIR_ALIGNMENT_CHECK, pair_aligned),
        ):
            if not passed and check not in failed:
                failed.append(check)
        summary["failed_checks"] = failed
        stage = summary.setdefault("stage_status", {})
        collision_in_grasp = any(
            snapshot.active and state in _GRASP_STATES
            for state, snapshot in zip(self._states, self._steps)
        )
        collision_in_operation = any(
            snapshot.active and state in _OPERATION_STATES
            for state, snapshot in zip(self._states, self._steps)
        )
        if collision_in_grasp:
            stage["grasp_success"] = False
        if collision_in_grasp or collision_in_operation or not closure_aligned or not bend_within_limit or not pair_aligned:
            stage["manipulation_success"] = False
            stage["full_success"] = False
        summary["passed"] = bool(summary.get("passed", False)) and all(
            (collision_free, closure_aligned, bend_within_limit, pair_aligned)
        )
        if not collision_free or not closure_aligned or not bend_within_limit or not pair_aligned:
            stage["full_success"] = False
        return summary

    def finalize(self, *, trace_path: str | Path | None = None) -> Mapping[str, Any]:
        if not self.complete:
            raise RuntimeError("cannot finalize an incomplete audited session")
        self._install_trace_fields()
        if self._summary is None:
            self._summary = self._augment_summary(self.inner.finalize())
        if trace_path is not None:
            self.inner.finalize(trace_path=trace_path)
        return copy.deepcopy(self._summary)

    def close(self) -> None:
        self.inner.close()


def upright_grasp_audited_session_factory(
    config: dict[str, Any]
) -> ActiveFingerSelfCollisionAuditedSession:
    """Default session factory for the existing atomic candidate runner."""

    return ActiveFingerSelfCollisionAuditedSession(SimulationSession(config))


def run_upright_grasp_rescue_job(
    job: Mapping[str, Any],
    output_root: str | Path,
    *,
    final_rerun: bool = False,
    retain_grasp_success: bool = False,
) -> V14CandidateArtifactBundle:
    """Run/resume one generated job through the shared artifact machinery."""

    candidate_id = int(job["candidate_id"])
    destination = Path(output_root) / "candidates" / f"candidate_{candidate_id}"
    return run_or_resume_v14_candidate_artifacts(
        job["config"],
        destination,
        candidate_id,
        final_rerun=bool(final_rerun),
        retain_grasp_success=bool(retain_grasp_success),
        session_factory=upright_grasp_audited_session_factory,
    )


def _metric(record: Mapping[str, Any], path: Sequence[str], default: float) -> float:
    value: Any = record
    try:
        for key in path:
            value = value[key]
        result = float(value)
    except (KeyError, TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def upright_grasp_rescue_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Rank collision freedom and contact preservation before lift distance."""

    summary = record.get("summary", {})
    checks = summary.get("checks", {}) if isinstance(summary, Mapping) else {}
    collision_free = bool(checks.get(SELF_COLLISION_CHECK, False))
    closure_aligned = bool(checks.get(UPRIGHT_CLOSURE_CHECK, False))
    bend_excursion_ok = bool(checks.get(INDEX_BEND_EXCURSION_CHECK, False))
    contact_duty = _metric(
        record,
        (
            "summary",
            "metrics",
            "contact_preserving_planned_lift",
            "simultaneous_target_face_effective_duty",
        ),
        -math.inf,
    )
    lift = _metric(
        record,
        ("summary", "metrics", "operation_median_lift_m"),
        -math.inf,
    )
    jerk = _metric(
        record,
        (
            "summary",
            "metrics",
            "motion_smoothness",
            "operation_peak_abs_filtered_jerk_m_s3",
        ),
        math.inf,
    )
    closure_index = _metric(
        record,
        (
            "summary",
            "metrics",
            "closure_alignment",
            "per_finger",
            "index",
            "angle_p95_deg",
        ),
        math.inf,
    )
    closure_mid = _metric(
        record,
        (
            "summary",
            "metrics",
            "closure_alignment",
            "per_finger",
            "mid",
            "angle_p95_deg",
        ),
        math.inf,
    )
    return (
        0 if bool(record.get("full_success", False)) else 1,
        0 if collision_free else 1,
        0 if bool(record.get("grasp_success", False)) else 1,
        0 if contact_duty >= 0.99 - 1e-12 else 1,
        0 if closure_aligned else 1,
        0 if bend_excursion_ok else 1,
        max(closure_index, closure_mid),
        -contact_duty,
        -lift,
        jerk,
        int(record.get("candidate_id", 2**63 - 1)),
    )


def rank_upright_grasp_rescue_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        copy.deepcopy(dict(value))
        for value in sorted(records, key=upright_grasp_rescue_candidate_rank)
    )


__all__ = [
    "ActiveFingerCollisionContact",
    "ActiveFingerSelfCollisionAuditedSession",
    "ActiveFingerSelfCollisionStep",
    "DEFAULT_SEED",
    "INDEX_BEND_ACTUATOR",
    "INDEX_BEND_EXCURSION_CHECK",
    "JOINT_PAIR_ALIGNMENT_CHECK",
    "MAXIMUM_CANDIDATE_COUNT",
    "SELF_COLLISION_CHECK",
    "UPRIGHT_CLOSURE_CHECK",
    "UPRIGHT_GRASP_RESCUE_SCHEMA_VERSION",
    "UprightGraspRescueBudget",
    "UprightGraspRescueSource",
    "active_finger_self_collision_snapshot",
    "authenticate_upright_grasp_rescue_source",
    "build_upright_grasp_rescue_jobs",
    "rank_upright_grasp_rescue_records",
    "run_upright_grasp_rescue_job",
    "upright_grasp_audited_session_factory",
    "upright_grasp_rescue_candidate_rank",
]
