"""Executable orchestration for the schema-v4 aligned-contact campaign.

The experiment definition and ranking policy live in their versioned modules;
this module owns candidate materialization, parallel static geometry screening,
the hard stage boundaries, and command-compatible reporting.  Every expensive
boundary is injectable so unit tests can exercise the complete state machine
with tiny deterministic budgets.
"""

from __future__ import annotations

import copy
import heapq
import math
import multiprocessing
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .aligned_contacts_search import (
    aligned_contact_candidate_rank,
    budget_manifest,
    candidate_tilt_band_deg,
)
from .config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    DISTAL_BODY_NAMES,
    resolved_pose_constraint_values,
    validate_config,
)
from .contacts import BoxContactThresholds
from .evaluation import face_from_label
from .experiment import (
    AlignedContactCampaignParameters,
    AlignedContactPerturbationEnvelope,
    ExperimentDefinition,
    OpposedFaceAssignment,
    resolve_experiment,
)
from .experiments.opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift import (
    ALIGNED_CONTACT_CAMPAIGN,
    EXPERIMENT_ID,
)
from .larger_cube_grasp_search import (
    full_succeeded,
    grasp_succeeded,
    sample_manipulation_delta_candidates,
)
from .scene import (
    build_model,
    cube_vertical_half_extent_m,
    palm_plane_ground_angle_deg,
    rpy_degrees_to_rotation_matrix,
    signed_finger_down_tilt_deg,
    solve_press_depth_pose,
)
from .search import normalized_acceptance_margins
from .v2_search import _static_score as _mujoco_static_score


CandidateRunner = Callable[
    [list[tuple[int, dict[str, Any]]], int], list[dict[str, Any]]
]
StaticRunner = Callable[..., Any]
Validator = Callable[[dict[str, Any]], None]

GRAVITY_WORLD_M_S2 = (0.0, 0.0, -9.81)
EXACT_TIMESTEP_S = 0.001

_STAGE_SEED_OFFSETS = {
    "static": 100_000_000,
    "grasp_refinement": 200_000_000,
    "manipulation_refinement": 300_000_000,
    "robustness": 400_000_000,
}


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _scale(unit: float, bounds: Sequence[float]) -> float:
    return float(float(bounds[0]) + float(unit) * (float(bounds[1]) - float(bounds[0])))


def _latin_hypercube(
    samples: int, dimensions: int, rng: np.random.Generator
) -> np.ndarray:
    _positive_integer(samples, "samples")
    _positive_integer(dimensions, "dimensions")
    result = np.empty((samples, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        result[:, dimension] = (
            rng.permutation(samples) + rng.random(samples)
        ) / samples
    return result


@dataclass(frozen=True)
class AlignedTuningBudget:
    """Effective per-band work budget used by one tuning invocation."""

    static_samples_per_edge_band: int
    dynamic_candidates_per_band: int
    grasp_refine_seed_count_per_band: int
    grasp_refine_per_seed: int
    manipulation_seed_count_per_band: int
    manipulation_refine_per_seed: int
    exact_candidates_per_band: int
    perturbation_count: int

    def __post_init__(self) -> None:
        for name in (
            "static_samples_per_edge_band",
            "dynamic_candidates_per_band",
            "grasp_refine_seed_count_per_band",
            "grasp_refine_per_seed",
            "manipulation_seed_count_per_band",
            "manipulation_refine_per_seed",
            "exact_candidates_per_band",
        ):
            _positive_integer(getattr(self, name), name)
        _nonnegative_integer(self.perturbation_count, "perturbation_count")
        if self.dynamic_candidates_per_band < 4:
            raise ValueError(
                "dynamic_candidates_per_band must retain all four face assignments"
            )

    @classmethod
    def from_campaign(
        cls,
        campaign: AlignedContactCampaignParameters,
        *,
        perturbation_count: int,
    ) -> "AlignedTuningBudget":
        return cls(
            static_samples_per_edge_band=campaign.static_samples_per_edge_band,
            dynamic_candidates_per_band=campaign.dynamic_candidates_per_band,
            grasp_refine_seed_count_per_band=(
                campaign.grasp_refine_seed_count_per_band
            ),
            grasp_refine_per_seed=campaign.grasp_refine_per_seed,
            manipulation_seed_count_per_band=(
                campaign.manipulation_seed_count_per_band
            ),
            manipulation_refine_per_seed=(
                campaign.manipulation_refine_per_seed
            ),
            exact_candidates_per_band=campaign.exact_candidates_per_band,
            perturbation_count=perturbation_count,
        )

    def as_dict(self, *, edge_count: int, band_count: int) -> dict[str, Any]:
        return {
            "edge_count": int(edge_count),
            "band_count": int(band_count),
            "static_samples_per_edge_band": self.static_samples_per_edge_band,
            "static_sample_count": (
                edge_count * band_count * self.static_samples_per_edge_band
            ),
            "dynamic_candidates_per_band": self.dynamic_candidates_per_band,
            "dynamic_candidate_count": (
                band_count * self.dynamic_candidates_per_band
            ),
            "grasp_refine_seed_count_per_band": (
                self.grasp_refine_seed_count_per_band
            ),
            "grasp_refine_per_seed": self.grasp_refine_per_seed,
            "grasp_refinement_count": (
                band_count
                * self.grasp_refine_seed_count_per_band
                * self.grasp_refine_per_seed
            ),
            "manipulation_seed_count_per_band": (
                self.manipulation_seed_count_per_band
            ),
            "manipulation_refine_per_seed": (
                self.manipulation_refine_per_seed
            ),
            "manipulation_refinement_count": (
                band_count
                * self.manipulation_seed_count_per_band
                * self.manipulation_refine_per_seed
            ),
            "exact_candidates_per_band": self.exact_candidates_per_band,
            "exact_candidate_count": (
                band_count * self.exact_candidates_per_band
            ),
            "perturbation_count": self.perturbation_count,
        }


# Compatibility names retained for callers that imported the former
# tuner-local type.  The source of truth now lives in the versioned campaign.
AlignedPerturbationRanges = AlignedContactPerturbationEnvelope
ALIGNED_PERTURBATION_RANGES = ALIGNED_CONTACT_CAMPAIGN.perturbation_envelope


@dataclass(frozen=True)
class AlignedStaticJob:
    job_id: int
    edge_m: float
    tilt_band_center_deg: float
    samples: int
    seed: int
    face_sample_counts: tuple[int, int, int, int]


@dataclass(frozen=True)
class StaticRetainedCandidate:
    static_candidate_id: int
    face_index: int
    tilt_band_center_deg: float
    score: tuple[float, ...]
    config: dict[str, Any]
    diagnostic: dict[str, Any]


@dataclass(frozen=True)
class StaticScreenOutcome:
    sample_count: int
    retained_by_band: dict[float, tuple[dict[str, Any], ...]]
    diagnostics_by_band: dict[float, tuple[dict[str, Any], ...]]
    near_miss_by_band: dict[float, dict[str, Any] | None]
    job_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _StaticJobOutcome:
    job: AlignedStaticJob
    retained: tuple[StaticRetainedCandidate, ...]
    valid_pose_count: int
    eligible_count: int
    clean_three_count: int
    near_three_count: int
    forbidden_count: int
    penetration_exceeded_count: int
    face_gate_counts: tuple[dict[str, Any], ...]
    best_near_miss: StaticRetainedCandidate | None


def aligned_static_candidate_advances(
    diagnostic: Mapping[str, Any],
    *,
    max_penetration_m: float,
    near_signed_distance_m: tuple[float, float] = (-0.002, 0.003),
) -> bool:
    """Apply the schema-v4 geometry-only advancement gate.

    Clean contacts and the estimated contact-height alignment are intentionally
    absent from this predicate: both are useful static ranking signals, while
    the dynamic CLOSE+VERIFY stages own those hard decisions.
    """

    maximum_penetration = _finite(max_penetration_m, "max_penetration_m")
    near_lower = _finite(near_signed_distance_m[0], "near_signed_distance_m[0]")
    near_upper = _finite(near_signed_distance_m[1], "near_signed_distance_m[1]")
    if near_lower > near_upper:
        raise ValueError("near_signed_distance_m must be ordered")
    if bool(diagnostic.get("forbidden_contact", True)):
        return False
    try:
        penetration = float(diagnostic.get("max_penetration_m", math.inf))
    except (TypeError, ValueError):
        return False
    if (
        not math.isfinite(penetration)
        or penetration > maximum_penetration + 1e-12
    ):
        return False
    try:
        near_count = int(diagnostic.get("near_target_face_count", 0))
    except (TypeError, ValueError):
        return False
    if near_count != len(ACTIVE_FINGERS):
        return False
    distances = diagnostic.get("target_site_signed_distance_m")
    if distances is None or isinstance(distances, (str, bytes)):
        return False
    try:
        values = tuple(float(value) for value in distances)
    except (TypeError, ValueError):
        return False
    return len(values) == len(ACTIVE_FINGERS) and all(
        math.isfinite(value)
        and near_lower - 1e-12 <= value <= near_upper + 1e-12
        for value in values
    )


def root_pitch_for_finger_down_tilt_deg(
    roll_deg: float, finger_down_tilt_deg: float
) -> float:
    """Solve pitch for the requested signed local-+Z tilt under world -Z gravity."""

    roll = math.radians(_finite(roll_deg, "roll_deg"))
    tilt = math.radians(_finite(finger_down_tilt_deg, "finger_down_tilt_deg"))
    cosine_roll = math.cos(roll)
    if abs(cosine_roll) <= 1e-9:
        raise ValueError("finger-down pitch solve is near-singular at this roll")
    cosine_pitch = -math.sin(tilt) / cosine_roll
    if cosine_pitch < -1.0 - 1e-12 or cosine_pitch > 1.0 + 1e-12:
        raise ValueError("requested finger-down tilt is infeasible at this roll")
    return float(math.degrees(math.acos(np.clip(cosine_pitch, -1.0, 1.0))))


def _feasible_tilt_and_pitch(
    *,
    roll_deg: float,
    requested_tilt_deg: float,
    definition: ExperimentDefinition,
) -> tuple[float, float]:
    constraints = definition.pose_constraints
    if constraints is None:
        raise ValueError("aligned-contact experiment is missing pose constraints")
    tilt_lower, tilt_upper = constraints.finger_down_tilt_deg
    palm_lower, palm_upper = constraints.palm_plane_ground_angle_deg
    requested = float(np.clip(requested_tilt_deg, tilt_lower, tilt_upper))
    pitch = root_pitch_for_finger_down_tilt_deg(roll_deg, requested)
    rotation = rpy_degrees_to_rotation_matrix([roll_deg, pitch, 0.0])
    palm_angle = palm_plane_ground_angle_deg(rotation, GRAVITY_WORLD_M_S2)
    if palm_angle > palm_upper + 1e-12:
        # At the upper endpoint a non-zero roll makes exact tilt require a
        # slightly steeper palm.  Keep the declared palm boundary exact; the
        # resulting tilt remains just inside its band and global bounds.
        pitch = 90.0 + palm_upper
        rotation = rpy_degrees_to_rotation_matrix([roll_deg, pitch, 0.0])
        requested = signed_finger_down_tilt_deg(rotation, GRAVITY_WORLD_M_S2)
    elif palm_angle < palm_lower - 1e-12:
        pitch = root_pitch_for_finger_down_tilt_deg(roll_deg, tilt_lower)
        rotation = rpy_degrees_to_rotation_matrix([roll_deg, pitch, 0.0])
        requested = signed_finger_down_tilt_deg(rotation, GRAVITY_WORLD_M_S2)
    if not tilt_lower - 1e-12 <= requested <= tilt_upper + 1e-12:
        raise ValueError("roll and palm bounds have no feasible finger-down tilt")
    return float(requested), float(pitch)


def cube_world_position_m(config: Mapping[str, Any]) -> np.ndarray:
    """Resolve the schema-aware initial cube centre in world coordinates."""

    cube = config["cube"]
    rotation = rpy_degrees_to_rotation_matrix(
        cube.get("rpy_deg", [0.0, 0.0, 0.0])
    )
    extent = cube_vertical_half_extent_m(float(cube["edge_m"]), rotation)
    return np.asarray(
        [
            float(cube["center_xy_m"][0]),
            float(cube["center_xy_m"][1]),
            float(config["scene"]["support_top_z_m"])
            + extent
            + float(cube.get("z_offset_m", 0.0)),
        ],
        dtype=np.float64,
    )


def _assignment(
    value: OpposedFaceAssignment | Mapping[str, str],
) -> OpposedFaceAssignment:
    if isinstance(value, OpposedFaceAssignment):
        return value
    return OpposedFaceAssignment.from_mapping(dict(value))


def materialize_aligned_candidate(
    base: Mapping[str, Any],
    *,
    edge_m: float,
    tilt_band_center_deg: float,
    roll_deg: float,
    yaw_deg: float,
    press_depth_m: float,
    cube_in_root_y_m: float,
    cube_in_root_z_m: float,
    cube_yaw_deg: float,
    cube_roll_deg: float = 0.0,
    cube_pitch_deg: float = 0.0,
    grasp_targets_rad: Mapping[str, float],
    target_assignment: OpposedFaceAssignment | Mapping[str, str],
    manipulation_delta_rad: Mapping[str, float] | None = None,
    finger_down_tilt_deg: float | None = None,
    static_candidate_id: int | None = None,
    allow_pose_constraint_perturbation: bool = False,
    validator: Validator | None = validate_config,
) -> dict[str, Any]:
    """Create one fully resolved schema-v4 candidate without a movable root."""

    candidate = copy.deepcopy(dict(base))
    # Search candidates are fresh canonical campaign inputs.  An inherited
    # Viewer/robustness context would relax validation and could promote an
    # exploratory result to a nominal claim.
    candidate.pop("experiment_status", None)
    candidate.pop("run_context", None)
    definition = resolve_experiment(candidate)
    campaign = definition.aligned_contact_campaign
    constraints = definition.pose_constraints
    if (
        definition.experiment_id != EXPERIMENT_ID
        or campaign is None
        or constraints is None
    ):
        raise ValueError("materialization requires the registered v4 experiment")
    edge = _finite(edge_m, "edge_m")
    if not any(
        math.isclose(edge, declared, rel_tol=0.0, abs_tol=1e-12)
        for declared in campaign.edges_m
    ):
        raise ValueError("edge_m must be one of the registered 59--64 mm edges")
    grasp = {name: _finite(grasp_targets_rad[name], name) for name in ACTIVE_ACTUATORS}
    if set(grasp_targets_rad) != set(ACTIVE_ACTUATORS):
        raise ValueError("grasp_targets_rad must contain exactly eight actuators")
    if manipulation_delta_rad is None:
        manipulation = {name: 0.0 for name in ACTIVE_ACTUATORS}
    else:
        if set(manipulation_delta_rad) != set(ACTIVE_ACTUATORS):
            raise ValueError(
                "manipulation_delta_rad must contain exactly eight actuators"
            )
        manipulation = {
            name: _finite(manipulation_delta_rad[name], name)
            for name in ACTIVE_ACTUATORS
        }

    candidate["cube"]["edge_m"] = edge
    candidate["cube"]["mass_kg"] = campaign.constant_density_mass_kg(edge)
    candidate["cube"]["friction"] = campaign.friction
    candidate["cube"]["rpy_deg"] = [
        _finite(cube_roll_deg, "cube_roll_deg"),
        _finite(cube_pitch_deg, "cube_pitch_deg"),
        _finite(cube_yaw_deg, "cube_yaw_deg"),
    ]
    target_tilt = (
        float(tilt_band_center_deg)
        if finger_down_tilt_deg is None
        else _finite(finger_down_tilt_deg, "finger_down_tilt_deg")
    )
    finite_roll = _finite(roll_deg, "roll_deg")
    if allow_pose_constraint_perturbation:
        resolved_tilt = target_tilt
        pitch = root_pitch_for_finger_down_tilt_deg(finite_roll, target_tilt)
    else:
        resolved_tilt, pitch = _feasible_tilt_and_pitch(
            roll_deg=finite_roll,
            requested_tilt_deg=target_tilt,
            definition=definition,
        )
    rpy = [float(roll_deg), pitch, _finite(yaw_deg, "yaw_deg")]
    rotation = rpy_degrees_to_rotation_matrix(rpy)
    cube_world = cube_world_position_m(candidate)
    solved = solve_press_depth_pose(
        reference_root_translation_m=(
            constraints.reference_hand_translation_m
        ),
        press_depth_m=_finite(press_depth_m, "press_depth_m"),
        rotation_world_from_root=rotation,
        cube_world_position_m=cube_world,
        cube_in_root_y_m=_finite(cube_in_root_y_m, "cube_in_root_y_m"),
        cube_in_root_z_m=_finite(cube_in_root_z_m, "cube_in_root_z_m"),
    )
    candidate["hand_pose"] = {
        "translation_m": list(solved.root_translation_m),
        "rpy_deg": rpy,
    }
    candidate["control"] = {
        "grasp_targets_rad": grasp,
        "manipulation_delta_rad": manipulation,
    }
    candidate["contact_topology"]["target_faces"] = _assignment(
        target_assignment
    ).as_dict()
    metadata = dict(candidate.get("candidate_metadata", {}))
    metadata.update(
        {
            "tilt_band_center_deg": float(tilt_band_center_deg),
            "resolved_finger_down_tilt_deg": resolved_tilt,
            "palm_plane_ground_angle_deg": palm_plane_ground_angle_deg(
                rotation, GRAVITY_WORLD_M_S2
            ),
            "palm_press_depth_m": float(press_depth_m),
            "cube_in_root_m": list(solved.cube_in_root_m),
        }
    )
    if static_candidate_id is not None:
        metadata["static_candidate_id"] = int(static_candidate_id)
    candidate["candidate_metadata"] = metadata
    if validator is not None:
        validator(candidate)
    return candidate


def _effective_static_jobs(
    campaign: AlignedContactCampaignParameters,
    *,
    samples_per_edge_band: int,
    seed: int,
) -> tuple[AlignedStaticJob, ...]:
    samples = _positive_integer(samples_per_edge_band, "samples_per_edge_band")
    quotient, remainder = divmod(samples, 4)
    face_counts = tuple(
        quotient + int(index < remainder) for index in range(4)
    )
    jobs: list[AlignedStaticJob] = []
    band_count = len(campaign.tilt_band_centers_deg)
    for edge_index, edge in enumerate(campaign.edges_m):
        for band_index, band in enumerate(campaign.tilt_band_centers_deg):
            job_id = edge_index * band_count + band_index
            jobs.append(
                AlignedStaticJob(
                    job_id=job_id,
                    edge_m=float(edge),
                    tilt_band_center_deg=float(band),
                    samples=samples,
                    seed=int(seed) + _STAGE_SEED_OFFSETS["static"] + job_id,
                    face_sample_counts=face_counts,
                )
            )
    return tuple(jobs)


def _distal_geometry_ids(model: mujoco.MjModel, info: Any) -> dict[str, tuple[int, ...]]:
    return {
        finger: tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.body_weldid[int(model.geom_bodyid[geom_id])])
            == info.distal_weld_ids[finger]
            and (
                int(model.geom_contype[geom_id]) != 0
                or int(model.geom_conaffinity[geom_id]) != 0
            )
        )
        for finger in ACTIVE_FINGERS
    }


def _distal_site_ids(model: mujoco.MjModel, info: Any) -> dict[str, np.ndarray]:
    return {
        finger: np.asarray(
            [
                site_id
                for site_id in range(model.nsite)
                if int(model.body_weldid[int(model.site_bodyid[site_id])])
                == info.distal_weld_ids[finger]
            ],
            dtype=int,
        )
        for finger in ACTIVE_FINGERS
    }


def _nearest_target_geometry_estimates(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: Any,
    assignment: OpposedFaceAssignment,
    distal_geom_ids: Mapping[str, Sequence[int]],
    thresholds: BoxContactThresholds,
) -> tuple[list[float], list[float], list[float | None], float]:
    cube_rotation = data.geom_xmat[info.cube_geom_id].reshape(3, 3)
    cube_position = data.geom_xpos[info.cube_geom_id]
    half_extent = model.geom_size[info.cube_geom_id]
    signed_distances: list[float] = []
    absolute_distances: list[float] = []
    heights: list[float | None] = []
    for finger, label in assignment.as_dict().items():
        face = face_from_label(label)
        tangential = [axis for axis in range(3) if axis != face.axis]
        best: tuple[float, float, float] | None = None
        for geom_id in distal_geom_ids[finger]:
            from_to = np.zeros(6, dtype=np.float64)
            distance = float(
                mujoco.mj_geomDistance(
                    model,
                    data,
                    info.cube_geom_id,
                    int(geom_id),
                    1.0,
                    from_to,
                )
            )
            cube_point = from_to[:3]
            local = cube_rotation.T @ (cube_point - cube_position)
            surface_error = abs(
                local[face.axis] - face.sign * half_extent[face.axis]
            )
            edge_clearance = float(
                np.min(half_extent[tangential] - np.abs(local[tangential]))
            )
            if (
                surface_error <= thresholds.surface_tolerance_m + 1e-9
                and edge_clearance + 1e-12 >= thresholds.edge_margin_m
            ):
                candidate = (abs(distance), distance, float(cube_point[2]))
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            signed_distances.append(math.inf)
            absolute_distances.append(math.inf)
            heights.append(None)
        else:
            absolute_distances.append(best[0])
            signed_distances.append(best[1])
            heights.append(best[2])
    finite_heights = [height for height in heights if height is not None]
    spread = (
        float(max(finite_heights) - min(finite_heights))
        if len(finite_heights) == len(ACTIVE_FINGERS)
        else math.inf
    )
    return signed_distances, absolute_distances, heights, spread


def _static_parameter_pose(
    base: Mapping[str, Any],
    *,
    definition: ExperimentDefinition,
    band: float,
    roll: float,
    yaw: float,
    press: float,
    cube_y: float,
    cube_z: float,
) -> tuple[list[float], tuple[float, float, float]] | None:
    resolved_tilt, pitch = _feasible_tilt_and_pitch(
        roll_deg=roll,
        requested_tilt_deg=band,
        definition=definition,
    )
    rpy = [roll, pitch, yaw]
    rotation = rpy_degrees_to_rotation_matrix(rpy)
    solved = solve_press_depth_pose(
        reference_root_translation_m=(
            definition.pose_constraints.reference_hand_translation_m  # type: ignore[union-attr]
        ),
        press_depth_m=press,
        rotation_world_from_root=rotation,
        cube_world_position_m=cube_world_position_m(base),
        cube_in_root_y_m=cube_y,
        cube_in_root_z_m=cube_z,
    )
    bounds = definition.search_bounds
    if not bounds.contains_cube_position(solved.cube_in_root_m):
        return None
    palm = palm_plane_ground_angle_deg(rotation, GRAVITY_WORLD_M_S2)
    constraints = definition.pose_constraints
    assert constraints is not None
    if not (
        constraints.finger_down_tilt_deg[0] - 1e-12
        <= resolved_tilt
        <= constraints.finger_down_tilt_deg[1] + 1e-12
        and constraints.palm_plane_ground_angle_deg[0] - 1e-12
        <= palm
        <= constraints.palm_plane_ground_angle_deg[1] + 1e-12
    ):
        return None
    return rpy, solved.cube_in_root_m


def _screen_static_job(
    payload: tuple[dict[str, Any], AlignedStaticJob, int]
) -> _StaticJobOutcome:
    base, job, retain_per_band = payload
    definition = resolve_experiment(base)
    campaign = definition.aligned_contact_campaign
    if campaign is None:
        raise ValueError("static job requires aligned_contact_campaign")
    job_base = copy.deepcopy(base)
    job_base["cube"]["edge_m"] = job.edge_m
    job_base["cube"]["mass_kg"] = campaign.constant_density_mass_kg(job.edge_m)
    job_base["cube"]["friction"] = campaign.friction
    validate_config(job_base)
    model, info = build_model(job_base)
    data = mujoco.MjData(model)
    topology = job_base["contact_topology"]
    thresholds = BoxContactThresholds(
        surface_tolerance_m=float(topology["surface_tolerance_m"]),
        edge_margin_m=float(topology["edge_margin_m"]),
        normal_alignment_min=float(topology["min_normal_alignment"]),
    )
    distal_geom_ids = _distal_geometry_ids(model, info)
    distal_site_ids = _distal_site_ids(model, info)
    if any(not values for values in distal_geom_ids.values()) or any(
        values.size == 0 for values in distal_site_ids.values()
    ):
        raise ValueError("active distal geometry/tactile mapping is incomplete")

    bounds = definition.search_bounds
    constraints = definition.pose_constraints
    alignment = definition.contact_alignment
    assert constraints is not None and alignment is not None
    dimensions = 6 + len(ACTIVE_ACTUATORS)
    matrix = _latin_hypercube(
        job.samples, dimensions, np.random.default_rng(job.seed)
    )
    assignments = definition.candidate_faces
    capacities = tuple(
        retain_per_band // len(assignments)
        + int(index < retain_per_band % len(assignments))
        for index in range(len(assignments))
    )
    heaps: list[list[tuple[Any, ...]]] = [[] for _ in assignments]
    valid_pose_count = 0
    eligible_count = 0
    clean_three_count = 0
    near_three_count = 0
    forbidden_count = 0
    penetration_exceeded_count = 0
    face_gate_counts = [
        {
            "face_index": face_index,
            "target_faces": assignments[face_index].as_dict(),
            "sample_count": int(job.face_sample_counts[face_index]),
            "valid_pose_count": 0,
            "clean_three_count": 0,
            "near_three_count": 0,
            "forbidden_count": 0,
            "penetration_exceeded_count": 0,
            "eligible_count": 0,
        }
        for face_index in range(len(assignments))
    ]
    max_penetration_m = float(job_base["acceptance"]["max_penetration_m"])
    best_near_miss: tuple[Any, ...] | None = None
    for row_index, row in enumerate(matrix):
        static_id = job.job_id * job.samples + row_index
        face_index = row_index % len(assignments)
        cursor = 0
        roll = _scale(row[cursor], bounds.hand_roll_deg)
        cursor += 1
        yaw = _scale(row[cursor], bounds.hand_yaw_deg)
        cursor += 1
        press = _scale(row[cursor], constraints.palm_press_depth_m)
        cursor += 1
        cube_y = _scale(row[cursor], bounds.cube_position_in_root_m["y"])
        cursor += 1
        cube_z = _scale(row[cursor], bounds.cube_position_in_root_m["z"])
        cursor += 1
        cube_yaw = _scale(row[cursor], bounds.cube_yaw_deg)
        cursor += 1
        targets = {
            name: _scale(row[cursor + index], bounds.actuator_targets_rad[name])
            for index, name in enumerate(ACTIVE_ACTUATORS)
        }
        pose = _static_parameter_pose(
            job_base,
            definition=definition,
            band=job.tilt_band_center_deg,
            roll=roll,
            yaw=yaw,
            press=press,
            cube_y=cube_y,
            cube_z=cube_z,
        )
        if pose is None:
            continue
        valid_pose_count += 1
        face_gate_counts[face_index]["valid_pose_count"] += 1
        rpy, cube_in_root = pose
        assignment = assignments[face_index]
        parameters = {
            "candidate_id": static_id,
            "hand_rpy_deg": rpy,
            "cube_in_root_m": np.asarray(cube_in_root, dtype=np.float64),
            "cube_yaw_deg": cube_yaw,
            "pregrasp_targets_rad": targets,
            "target_assignment": assignment,
        }
        legacy_score, diagnostic = _mujoco_static_score(
            model,
            data,
            info,
            job_base,
            parameters,
            thresholds,
            distal_geom_ids,
            distal_site_ids,
        )
        (
            signed_distances,
            absolute_distances,
            heights,
            height_spread,
        ) = _nearest_target_geometry_estimates(
            model,
            data,
            info,
            assignment,
            distal_geom_ids,
            thresholds,
        )
        finite_distance = sum(
            distance if math.isfinite(distance) else 1.0
            for distance in absolute_distances
        )
        aligned = height_spread <= alignment.max_height_spread_m + 1e-12
        score = (
            float(not bool(diagnostic.get("forbidden_contact", True))),
            float(
                float(diagnostic.get("max_penetration_m", math.inf))
                <= float(job_base["acceptance"]["max_penetration_m"]) + 1e-12
            ),
            float(diagnostic.get("clean_target_contact_count", 0)),
            float(aligned),
            float(diagnostic.get("near_target_face_count", 0)),
            -height_spread,
            -finite_distance,
            *(float(value) for value in legacy_score[:-1]),
            -float(static_id),
        )
        diagnostic = dict(diagnostic)
        diagnostic.update(
            {
                "static_candidate_id": static_id,
                "static_job_id": job.job_id,
                "edge_m": job.edge_m,
                "tilt_band_center_deg": job.tilt_band_center_deg,
                "target_nearest_signed_distance_m": signed_distances,
                "target_nearest_absolute_distance_m": absolute_distances,
                "target_contact_height_estimate_m": heights,
                "contact_height_spread_estimate_m": height_spread,
                "contact_height_aligned_estimate": aligned,
                "static_rank": score,
            }
        )
        clean_three = int(diagnostic.get("clean_target_contact_count", 0)) >= len(
            ACTIVE_FINGERS
        )
        near_three = int(diagnostic.get("near_target_face_count", 0)) == len(
            ACTIVE_FINGERS
        )
        forbidden = bool(diagnostic.get("forbidden_contact", True))
        try:
            penetration = float(diagnostic.get("max_penetration_m", math.inf))
        except (TypeError, ValueError):
            penetration = math.inf
        penetration_exceeded = (
            not math.isfinite(penetration)
            or penetration > max_penetration_m + 1e-12
        )
        clean_three_count += int(clean_three)
        near_three_count += int(near_three)
        forbidden_count += int(forbidden)
        penetration_exceeded_count += int(penetration_exceeded)
        face_gate_counts[face_index]["clean_three_count"] += int(clean_three)
        face_gate_counts[face_index]["near_three_count"] += int(near_three)
        face_gate_counts[face_index]["forbidden_count"] += int(forbidden)
        face_gate_counts[face_index]["penetration_exceeded_count"] += int(
            penetration_exceeded
        )
        near_miss_item = (
            score,
            -static_id,
            face_index,
            parameters,
            diagnostic,
            press,
            cube_y,
            cube_z,
        )
        if best_near_miss is None or near_miss_item[:2] > best_near_miss[:2]:
            best_near_miss = near_miss_item
        eligible = aligned_static_candidate_advances(
            diagnostic,
            max_penetration_m=max_penetration_m,
        )
        diagnostic["eligible_for_dynamic"] = bool(eligible)
        if not eligible:
            continue
        eligible_count += 1
        face_gate_counts[face_index]["eligible_count"] += 1
        item = (
            score,
            -static_id,
            parameters,
            diagnostic,
            press,
            cube_y,
            cube_z,
        )
        heap = heaps[face_index]
        capacity = capacities[face_index]
        if capacity == 0:
            continue
        if len(heap) < capacity:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)

    def materialize_item(
        item: tuple[Any, ...], face_index: int
    ) -> StaticRetainedCandidate:
        score, neg_id, parameters, diagnostic, press, cube_y, cube_z = item
        static_id = -int(neg_id)
        config = materialize_aligned_candidate(
            job_base,
            edge_m=job.edge_m,
            tilt_band_center_deg=job.tilt_band_center_deg,
            roll_deg=float(parameters["hand_rpy_deg"][0]),
            yaw_deg=float(parameters["hand_rpy_deg"][2]),
            press_depth_m=press,
            cube_in_root_y_m=cube_y,
            cube_in_root_z_m=cube_z,
            cube_yaw_deg=float(parameters["cube_yaw_deg"]),
            grasp_targets_rad=parameters["pregrasp_targets_rad"],
            target_assignment=parameters["target_assignment"],
            static_candidate_id=static_id,
        )
        config["candidate_metadata"].update(
            {
                "static_job_id": job.job_id,
                "static_face_index": face_index,
                "static_score": list(score),
            }
        )
        return StaticRetainedCandidate(
            static_candidate_id=static_id,
            face_index=face_index,
            tilt_band_center_deg=job.tilt_band_center_deg,
            score=tuple(score),
            config=config,
            diagnostic=diagnostic,
        )

    retained: list[StaticRetainedCandidate] = []
    for face_index, heap in enumerate(heaps):
        for item in sorted(heap, key=lambda value: value[:2], reverse=True):
            retained.append(materialize_item(item, face_index))
    materialized_near_miss = None
    if best_near_miss is not None:
        (
            score,
            neg_id,
            face_index,
            parameters,
            diagnostic,
            press,
            cube_y,
            cube_z,
        ) = best_near_miss
        materialized_near_miss = materialize_item(
            (score, neg_id, parameters, diagnostic, press, cube_y, cube_z),
            int(face_index),
        )
    retained.sort(key=lambda item: (item.score, -item.static_candidate_id), reverse=True)
    return _StaticJobOutcome(
        job=job,
        retained=tuple(retained),
        valid_pose_count=valid_pose_count,
        eligible_count=eligible_count,
        clean_three_count=clean_three_count,
        near_three_count=near_three_count,
        forbidden_count=forbidden_count,
        penetration_exceeded_count=penetration_exceeded_count,
        face_gate_counts=tuple(face_gate_counts),
        best_near_miss=materialized_near_miss,
    )


def run_parallel_static_screen(
    config: dict[str, Any],
    *,
    jobs: Sequence[AlignedStaticJob],
    retain_per_band: int,
    workers: int,
    definition: ExperimentDefinition | None = None,
) -> StaticScreenOutcome:
    """Run edge×band MuJoCo jobs with spawn and merge by deterministic IDs."""

    worker_count = _positive_integer(workers, "workers")
    retain = _positive_integer(retain_per_band, "retain_per_band")
    selected_definition = resolve_experiment(config) if definition is None else definition
    assignments = selected_definition.candidate_faces
    if len(assignments) != 4:
        raise ValueError("aligned-contact static screening requires four faces")
    ordered_jobs = tuple(sorted(jobs, key=lambda job: job.job_id))
    if len({job.job_id for job in ordered_jobs}) != len(ordered_jobs):
        raise ValueError("static job IDs must be unique")
    payloads = [(copy.deepcopy(config), job, retain) for job in ordered_jobs]
    if worker_count <= 1 or len(payloads) <= 1:
        outcomes = [_screen_static_job(payload) for payload in payloads]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(worker_count, len(payloads)),
            mp_context=context,
        ) as executor:
            outcomes = list(executor.map(_screen_static_job, payloads, chunksize=1))
    outcomes.sort(key=lambda outcome: outcome.job.job_id)

    bands = tuple(
        float(value)
        for value in selected_definition.aligned_contact_campaign.tilt_band_centers_deg  # type: ignore[union-attr]
    )
    retained_by_band: dict[float, tuple[dict[str, Any], ...]] = {}
    diagnostics_by_band: dict[float, tuple[dict[str, Any], ...]] = {}
    near_miss_by_band: dict[float, dict[str, Any] | None] = {}
    face_capacities = tuple(
        retain // len(assignments) + int(index < retain % len(assignments))
        for index in range(len(assignments))
    )
    for band in bands:
        selected_records: list[StaticRetainedCandidate] = []
        for face_index, capacity in enumerate(face_capacities):
            pool = [
                retained
                for outcome in outcomes
                if math.isclose(
                    outcome.job.tilt_band_center_deg,
                    band,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for retained in outcome.retained
                if retained.face_index == face_index
            ]
            pool.sort(
                key=lambda item: (item.score, -item.static_candidate_id),
                reverse=True,
            )
            selected_records.extend(pool[:capacity])
        selected_records.sort(
            key=lambda item: (item.score, -item.static_candidate_id), reverse=True
        )
        retained_by_band[band] = tuple(
            copy.deepcopy(item.config) for item in selected_records
        )
        diagnostics_by_band[band] = tuple(
            copy.deepcopy(item.diagnostic) for item in selected_records
        )
        near_pool = [
            outcome.best_near_miss
            for outcome in outcomes
            if math.isclose(
                outcome.job.tilt_band_center_deg,
                band,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and outcome.best_near_miss is not None
        ]
        near_pool.sort(
            key=lambda item: (
                item.score,
                -item.static_candidate_id,
            ),
            reverse=True,
        )
        near_miss_by_band[band] = (
            {
                "static_candidate_id": near_pool[0].static_candidate_id,
                "config": copy.deepcopy(near_pool[0].config),
                "diagnostic": copy.deepcopy(near_pool[0].diagnostic),
            }
            if near_pool
            else None
        )
    selected_keys = {
        (
            int(config["candidate_metadata"]["static_job_id"]),
            int(config["candidate_metadata"]["static_candidate_id"]),
        )
        for configs in retained_by_band.values()
        for config in configs
    }
    job_records_list: list[dict[str, Any]] = []
    for outcome in outcomes:
        selected_for_job = {
            static_id
            for job_id, static_id in selected_keys
            if job_id == outcome.job.job_id
        }
        local_retained_by_face = [0] * len(assignments)
        selected_by_face = [0] * len(assignments)
        for retained in outcome.retained:
            local_retained_by_face[retained.face_index] += 1
            if retained.static_candidate_id in selected_for_job:
                selected_by_face[retained.face_index] += 1
        face_gate_records: list[dict[str, Any]] = []
        for face_index, counts in enumerate(outcome.face_gate_counts):
            record = dict(counts)
            record["local_reservoir_count"] = local_retained_by_face[face_index]
            record["retained_count"] = selected_by_face[face_index]
            record["selected_for_dynamic_count"] = selected_by_face[face_index]
            face_gate_records.append(record)
        selected_count = len(selected_for_job)
        if selected_count:
            stop_reason = "eligible_candidates_retained"
        elif outcome.eligible_count:
            stop_reason = "eligible_candidates_not_selected_by_band_cap"
        else:
            stop_reason = "no_static_candidate_passed_geometry_gate"
        job_records_list.append(
            {
                "job_id": outcome.job.job_id,
                "edge_m": outcome.job.edge_m,
                "edge_mm": outcome.job.edge_m * 1000.0,
                "tilt_band_center_deg": outcome.job.tilt_band_center_deg,
                "seed": outcome.job.seed,
                "sample_count": outcome.job.samples,
                "face_sample_counts": list(outcome.job.face_sample_counts),
                "face_assignments": [
                    assignment.as_dict() for assignment in assignments
                ],
                "valid_pose_count": outcome.valid_pose_count,
                "clean_three_count": outcome.clean_three_count,
                "near_three_count": outcome.near_three_count,
                "forbidden_count": outcome.forbidden_count,
                "penetration_exceeded_count": (
                    outcome.penetration_exceeded_count
                ),
                "eligible_count": outcome.eligible_count,
                "local_reservoir_count": len(outcome.retained),
                "retained_count": selected_count,
                "selected_for_dynamic_count": selected_count,
                "face_gate_counts": face_gate_records,
                "stop_reason": stop_reason,
            }
        )
    job_records = tuple(job_records_list)
    return StaticScreenOutcome(
        sample_count=sum(outcome.job.samples for outcome in outcomes),
        retained_by_band=retained_by_band,
        diagnostics_by_band=diagnostics_by_band,
        near_miss_by_band=near_miss_by_band,
        job_records=job_records,
    )


def _config_band(config: Mapping[str, Any]) -> float:
    try:
        return candidate_tilt_band_deg({"config": config, "candidate_id": 0})
    except ValueError:
        definition = resolve_experiment(dict(config))
        campaign = definition.aligned_contact_campaign
        if campaign is None:
            raise
        actual_tilt = float(
            resolved_pose_constraint_values(dict(config))["finger_down_tilt_deg"]
        )
        distances = [
            (abs(actual_tilt - float(band)), float(band))
            for band in campaign.tilt_band_centers_deg
        ]
        minimum_distance = min(distance for distance, _ in distances)
        nearest = [
            band
            for distance, band in distances
            if math.isclose(
                distance, minimum_distance, rel_tol=0.0, abs_tol=1e-12
            )
        ]
        if len(nearest) != 1:
            raise ValueError(
                "resolved finger-down tilt is equidistant between declared bands; "
                "persist tilt_band_center_deg explicitly"
            )
        return nearest[0]


def _coerce_static_outcome(
    value: Any,
    *,
    bands: Sequence[float],
) -> StaticScreenOutcome:
    """Accept the public outcome or concise mapping/sequence test doubles."""

    if isinstance(value, StaticScreenOutcome):
        outcome = value
    else:
        sample_count = 0
        job_records: tuple[dict[str, Any], ...] = ()
        diagnostics: dict[float, tuple[dict[str, Any], ...]] = {
            float(band): () for band in bands
        }
        near_misses: dict[float, dict[str, Any] | None] = {
            float(band): None for band in bands
        }
        if isinstance(value, Mapping) and "retained_by_band" in value:
            raw_groups = value["retained_by_band"]
            if not isinstance(raw_groups, Mapping):
                raise TypeError("static retained_by_band must be a mapping")
            sample_count = int(value.get("sample_count", 0))
            raw_records = value.get("job_records", ())
            job_records = tuple(copy.deepcopy(tuple(raw_records)))
            raw_diagnostics = value.get("diagnostics_by_band", {})
            if isinstance(raw_diagnostics, Mapping):
                diagnostics = {
                    float(band): tuple(copy.deepcopy(raw_diagnostics.get(band, ())))
                    for band in bands
                }
            raw_near_misses = value.get("near_miss_by_band", {})
            if isinstance(raw_near_misses, Mapping):
                near_misses = {
                    float(band): copy.deepcopy(
                        raw_near_misses.get(band, raw_near_misses.get(str(band)))
                    )
                    for band in bands
                }
        elif isinstance(value, Mapping):
            raw_groups = value
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            raw_groups_dict: dict[float, list[dict[str, Any]]] = {
                float(band): [] for band in bands
            }
            for config in value:
                if not isinstance(config, Mapping):
                    raise TypeError("static candidate sequence must contain configs")
                raw_groups_dict[_config_band(config)].append(copy.deepcopy(dict(config)))
            raw_groups = raw_groups_dict
        else:
            raise TypeError("static_runner returned an unsupported outcome")
        groups: dict[float, tuple[dict[str, Any], ...]] = {}
        for band in bands:
            raw = raw_groups.get(band, raw_groups.get(str(band), ()))
            configs: list[dict[str, Any]] = []
            for config in raw:
                if not isinstance(config, Mapping):
                    raise TypeError("static retained candidates must be configs")
                candidate = copy.deepcopy(dict(config))
                metadata = dict(candidate.get("candidate_metadata", {}))
                metadata.setdefault("tilt_band_center_deg", float(band))
                candidate["candidate_metadata"] = metadata
                configs.append(candidate)
            groups[float(band)] = tuple(configs)
        outcome = StaticScreenOutcome(
            sample_count=sample_count,
            retained_by_band=groups,
            diagnostics_by_band=diagnostics,
            near_miss_by_band=near_misses,
            job_records=job_records,
        )
    declared = tuple(float(band) for band in bands)
    unknown = set(float(value) for value in outcome.retained_by_band) - set(declared)
    if unknown:
        raise ValueError(f"static outcome contains undeclared tilt bands: {unknown}")
    normalized_groups: dict[float, tuple[dict[str, Any], ...]] = {}
    normalized_diagnostics: dict[float, tuple[dict[str, Any], ...]] = {}
    for band in declared:
        candidates = tuple(
            copy.deepcopy(candidate)
            for candidate in outcome.retained_by_band.get(band, ())
        )
        for candidate in candidates:
            actual_band = _config_band(candidate)
            if not math.isclose(actual_band, band, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("static candidate is assigned to the wrong tilt band")
        normalized_groups[band] = candidates
        normalized_diagnostics[band] = tuple(
            copy.deepcopy(item)
            for item in outcome.diagnostics_by_band.get(band, ())
        )
    return StaticScreenOutcome(
        sample_count=int(outcome.sample_count),
        retained_by_band=normalized_groups,
        diagnostics_by_band=normalized_diagnostics,
        near_miss_by_band={
            float(band): copy.deepcopy(outcome.near_miss_by_band.get(float(band)))
            for band in declared
        },
        job_records=tuple(copy.deepcopy(outcome.job_records)),
    )


def _nested_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalize_alignment_metrics(result: dict[str, Any]) -> None:
    """Expose canonical flat rank aliases from the v4 nested evaluator data."""

    summary = _nested_mapping(result.get("summary"))
    metrics_value = summary.get("metrics")
    if not isinstance(metrics_value, dict):
        return
    metrics = metrics_value
    alignment = _nested_mapping(metrics.get("contact_alignment"))
    operation = _nested_mapping(alignment.get("operation"))
    aligned_duty = operation.get("aligned_duty")
    p95_spread = operation.get("height_spread_p95_m")
    if aligned_duty is not None:
        metrics.setdefault("operation_aligned_contact_duty", aligned_duty)
    if p95_spread is not None:
        metrics.setdefault("operation_contact_height_spread_p95_m", p95_spread)
    config = result.get("config")
    if not isinstance(config, dict):
        return
    try:
        margins = normalized_acceptance_margins(
            metrics,
            config["acceptance"],
            contact_topology=config.get("contact_topology"),
            contact_alignment=config.get("contact_alignment"),
            pose_constraints=config.get("pose_constraints"),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return
    if margins:
        metrics.setdefault("normalized_acceptance_margins", margins)
        metrics.setdefault("minimum_normalized_margin", min(margins.values()))


def _tag_stage_config(
    config: Mapping[str, Any], *, stage: str, material_policy: str
) -> dict[str, Any]:
    tagged = copy.deepcopy(dict(config))
    metadata = dict(tagged.get("candidate_metadata", {}))
    metadata["search_stage"] = stage
    metadata["material_policy"] = material_policy
    if stage == "exact_1ms_confirmation":
        metadata["physics_timestep_s"] = EXACT_TIMESTEP_S
    tagged["candidate_metadata"] = metadata
    return tagged


def _run_stage(
    configs: Sequence[Mapping[str, Any]],
    *,
    next_candidate_id: int,
    workers: int,
    run_candidates: CandidateRunner,
    stage: str,
    material_policy: str,
    validator: Validator = validate_config,
) -> tuple[list[dict[str, Any]], int]:
    tagged = [
        _tag_stage_config(config, stage=stage, material_policy=material_policy)
        for config in configs
    ]
    for config in tagged:
        validator(config)
    payloads = [
        (next_candidate_id + index, copy.deepcopy(config))
        for index, config in enumerate(tagged)
    ]
    submitted = {
        candidate_id: copy.deepcopy(config) for candidate_id, config in payloads
    }
    raw_results = run_candidates(payloads, workers) if payloads else []
    results = [copy.deepcopy(dict(result)) for result in raw_results]
    expected_ids = tuple(candidate_id for candidate_id, _ in payloads)
    try:
        received_ids = tuple(int(result["candidate_id"]) for result in results)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{stage} runner returned an invalid candidate_id") from error
    if (
        len(results) != len(payloads)
        or len(set(received_ids)) != len(received_ids)
        or set(received_ids) != set(expected_ids)
    ):
        raise RuntimeError(
            f"{stage} runner result IDs must exactly match submitted IDs; "
            f"expected={expected_ids}, received={received_ids}"
        )
    for result in results:
        candidate_id = int(result["candidate_id"])
        if result.get("config") != submitted[candidate_id]:
            raise RuntimeError(
                f"{stage} runner rebound candidate_id={candidate_id} to a "
                "different configuration"
            )
        result["search_stage"] = stage
        result["material_policy"] = material_policy
        result["tilt_band_center_deg"] = _config_band(result["config"])
        _normalize_alignment_metrics(result)
    results.sort(key=lambda result: int(result["candidate_id"]))
    return results, next_candidate_id + len(payloads)


def _ranked(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized = list(results)
    ids = [int(result["candidate_id"]) for result in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be unique")
    return sorted(materialized, key=aligned_contact_candidate_rank, reverse=True)


def _select_per_band(
    results: Iterable[dict[str, Any]],
    *,
    bands: Sequence[float],
    limit: int,
    predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[float, tuple[dict[str, Any], ...]]:
    selected: dict[float, tuple[dict[str, Any], ...]] = {}
    ranked = _ranked(results)
    for band in bands:
        group = [
            result
            for result in ranked
            if math.isclose(
                candidate_tilt_band_deg(result),
                float(band),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and (predicate is None or predicate(result))
        ]
        selected[float(band)] = tuple(group[:limit])
    return selected


def _flatten_band_groups(
    groups: Mapping[float, Sequence[Mapping[str, Any]]],
    bands: Sequence[float],
) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(dict(item))
        for band in bands
        for item in groups.get(float(band), ())
    ]


def _local_grasp_candidates(
    parent: Mapping[str, Any],
    *,
    parent_candidate_id: int,
    count: int,
    seed: int,
    definition: ExperimentDefinition,
    validator: Validator = validate_config,
) -> list[dict[str, Any]]:
    sample_count = _positive_integer(count, "count")
    rng = np.random.default_rng(seed)
    bounds = definition.search_bounds
    constraints = definition.pose_constraints
    assert constraints is not None
    parent_config = parent["config"] if "config" in parent else parent
    if not isinstance(parent_config, Mapping):
        raise TypeError("grasp parent must contain a config")
    resolved = resolved_pose_constraint_values(dict(parent_config))
    parent_rpy = np.asarray(parent_config["hand_pose"]["rpy_deg"], dtype=np.float64)
    parent_press = float(resolved["palm_press_depth_m"])
    parent_cube = np.asarray(resolved["cube_position_in_root_m"], dtype=np.float64)
    parent_cube_yaw = float(parent_config["cube"]["rpy_deg"][2])
    parent_targets = parent_config["control"]["grasp_targets_rad"]
    assignment = OpposedFaceAssignment.from_mapping(
        parent_config["contact_topology"]["target_faces"]
    )
    band = _config_band(parent_config)
    edge = float(parent_config["cube"]["edge_m"])
    candidates: list[dict[str, Any]] = []
    for sample_index in range(sample_count):
        roll = float(np.clip(parent_rpy[0] + rng.normal(0.0, 0.75), *bounds.hand_roll_deg))
        yaw = float(np.clip(parent_rpy[2] + rng.normal(0.0, 0.75), *bounds.hand_yaw_deg))
        press = float(
            np.clip(
                parent_press + rng.normal(0.0, 0.00075),
                *constraints.palm_press_depth_m,
            )
        )
        cube_y = float(
            np.clip(
                parent_cube[1] + rng.normal(0.0, 0.00125),
                *bounds.cube_position_in_root_m["y"],
            )
        )
        cube_z = float(
            np.clip(
                parent_cube[2] + rng.normal(0.0, 0.00125),
                *bounds.cube_position_in_root_m["z"],
            )
        )
        cube_yaw = float(
            np.clip(parent_cube_yaw + rng.normal(0.0, 1.5), *bounds.cube_yaw_deg)
        )
        targets = {}
        for name in ACTIVE_ACTUATORS:
            lower, upper = bounds.actuator_targets_rad[name]
            span = float(upper - lower)
            targets[name] = float(
                np.clip(
                    float(parent_targets[name]) + rng.normal(0.0, 0.02 * span),
                    lower,
                    upper,
                )
            )
        try:
            candidate = materialize_aligned_candidate(
                parent_config,
                edge_m=edge,
                tilt_band_center_deg=band,
                roll_deg=roll,
                yaw_deg=yaw,
                press_depth_m=press,
                cube_in_root_y_m=cube_y,
                cube_in_root_z_m=cube_z,
                cube_yaw_deg=cube_yaw,
                grasp_targets_rad=targets,
                target_assignment=assignment,
                validator=validator,
            )
        except ValueError:
            # A corner draw can solve X just outside its declared envelope.
            # Keep the sampled grasp controls but fall back to the known-valid
            # parent pose so every declared local trial is materialized.
            candidate = copy.deepcopy(dict(parent_config))
            candidate["control"]["grasp_targets_rad"] = targets
            candidate["control"]["manipulation_delta_rad"] = {
                name: 0.0 for name in ACTIVE_ACTUATORS
            }
            validator(candidate)
        metadata = dict(candidate.get("candidate_metadata", {}))
        metadata.update(
            {
                "tilt_band_center_deg": band,
                "parent_candidate_id": int(parent_candidate_id),
                "local_grasp_sample_index": sample_index,
            }
        )
        candidate["candidate_metadata"] = metadata
        candidates.append(candidate)
    return candidates


def _manipulation_candidates(
    parent: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    definition: ExperimentDefinition,
    validator: Validator = validate_config,
) -> list[dict[str, Any]]:
    parent_config = parent["config"]
    delta_bounds = definition.search_bounds.manipulation_delta_rad
    if delta_bounds is None:
        raise ValueError("aligned-contact experiment is missing manipulation bounds")
    candidates = sample_manipulation_delta_candidates(
        parent_config,
        count=count,
        seed=seed,
        delta_bounds_rad=delta_bounds,
        local_radius_rad=0.10,
        absolute_target_bounds_rad=definition.search_bounds.actuator_targets_rad,
        validator=validator,
    )
    band = candidate_tilt_band_deg(parent)
    for index, candidate in enumerate(candidates):
        metadata = dict(candidate.get("candidate_metadata", {}))
        metadata.update(
            {
                "tilt_band_center_deg": band,
                "parent_candidate_id": int(parent["candidate_id"]),
                "local_manipulation_sample_index": index,
            }
        )
        candidate["candidate_metadata"] = metadata
    return candidates


def generate_aligned_perturbation_configs(
    parent_config: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    definition: ExperimentDefinition | None = None,
    ranges: AlignedPerturbationRanges | None = None,
    validator: Validator = validate_config,
) -> list[dict[str, Any]]:
    """Generate the 50-case local pose/material robustness catalog."""

    sample_count = _nonnegative_integer(count, "count")
    if sample_count == 0:
        return []
    selected_definition = (
        resolve_experiment(dict(parent_config)) if definition is None else definition
    )
    campaign = selected_definition.aligned_contact_campaign
    constraints = selected_definition.pose_constraints
    if campaign is None or constraints is None:
        raise ValueError("robustness generation requires the v4 campaign")
    selected_ranges = campaign.perturbation_envelope if ranges is None else ranges
    parent_pose = resolved_pose_constraint_values(dict(parent_config))
    parent_rpy = np.asarray(parent_config["hand_pose"]["rpy_deg"], dtype=np.float64)
    parent_cube = np.asarray(parent_pose["cube_position_in_root_m"], dtype=np.float64)
    parent_cube_rpy = np.asarray(parent_config["cube"]["rpy_deg"], dtype=np.float64)
    parent_tilt = float(parent_pose["finger_down_tilt_deg"])
    parent_press = float(parent_pose["palm_press_depth_m"])
    parent_xy = np.asarray(parent_config["cube"]["center_xy_m"], dtype=np.float64)
    parent_candidate_metadata = copy.deepcopy(
        dict(parent_config.get("candidate_metadata", {}))
    )
    assignment = OpposedFaceAssignment.from_mapping(
        parent_config["contact_topology"]["target_faces"]
    )
    band = _config_band(parent_config)
    dimensions = selected_ranges.dimensions
    matrix = _latin_hypercube(
        sample_count, dimensions, np.random.default_rng(int(seed))
    )
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(matrix):
        edge = float(parent_config["cube"]["edge_m"])
        roll = float(
            parent_rpy[0]
            + _scale(row[0], selected_ranges.hand_roll_yaw_delta_deg)
        )
        yaw = float(
            parent_rpy[2]
            + _scale(row[1], selected_ranges.hand_roll_yaw_delta_deg)
        )
        tilt = float(
            parent_tilt
            + _scale(row[2], selected_ranges.finger_down_tilt_delta_deg)
        )
        press = float(
            parent_press
            + _scale(row[3], selected_ranges.palm_press_depth_delta_m)
        )
        cube_y = float(parent_cube[1])
        cube_z = float(parent_cube[2])
        cube_rpy = parent_cube_rpy + np.asarray(
            [
                _scale(row[4 + axis], selected_ranges.cube_rpy_delta_deg)
                for axis in range(3)
            ]
        )
        cube_xy = (
            parent_xy
            + np.asarray(
                [
                    _scale(row[7], selected_ranges.cube_center_xy_delta_m),
                    _scale(row[8], selected_ranges.cube_center_xy_delta_m),
                ]
            )
        )
        trial_base = copy.deepcopy(dict(parent_config))
        # A robustness trial is a fresh physical observation.  Never carry a
        # nominal or previous-campaign claim into the generated input; the
        # caller derives a new status from this trial's actual summary.
        trial_base.pop("experiment_status", None)
        trial_base.pop("candidate_metadata", None)
        # Solve the hand against the nominal cube pose.  Independent cube XY,
        # gap and RPY perturbations are applied afterward so none of those
        # dimensions is cancelled by translating the fixed root with the cube.
        trial_base["cube"]["center_xy_m"] = parent_xy.tolist()
        trial_base["cube"]["z_offset_m"] = float(
            parent_config["cube"].get("z_offset_m", 0.0)
        )
        candidate = materialize_aligned_candidate(
            trial_base,
            edge_m=edge,
            tilt_band_center_deg=band,
            finger_down_tilt_deg=tilt,
            roll_deg=roll,
            yaw_deg=yaw,
            press_depth_m=press,
            cube_in_root_y_m=cube_y,
            cube_in_root_z_m=cube_z,
            cube_roll_deg=float(parent_cube_rpy[0]),
            cube_pitch_deg=float(parent_cube_rpy[1]),
            cube_yaw_deg=float(parent_cube_rpy[2]),
            grasp_targets_rad=parent_config["control"]["grasp_targets_rad"],
            manipulation_delta_rad=parent_config["control"]["manipulation_delta_rad"],
            target_assignment=assignment,
            allow_pose_constraint_perturbation=True,
            validator=None,
        )
        candidate["cube"]["center_xy_m"] = cube_xy.tolist()
        candidate["cube"]["z_offset_m"] = _scale(
            row[9], selected_ranges.cube_gap_m
        )
        candidate["cube"]["rpy_deg"] = cube_rpy.tolist()
        density_scale = _scale(row[10], selected_ranges.mass_scale)
        friction_delta = _scale(row[11], selected_ranges.friction_delta)
        candidate["cube"]["mass_kg"] = (
            campaign.constant_density_mass_kg(edge) * density_scale
        )
        candidate["cube"]["friction"] = campaign.friction + friction_delta
        candidate["run_context"] = {"kind": "robustness_trial"}
        actual_pose = resolved_pose_constraint_values(candidate)
        metadata = dict(candidate.get("candidate_metadata", {}))
        metadata["resolved_finger_down_tilt_deg"] = float(
            actual_pose["finger_down_tilt_deg"]
        )
        metadata["palm_plane_ground_angle_deg"] = float(
            actual_pose["palm_plane_ground_angle_deg"]
        )
        metadata["palm_press_depth_m"] = float(
            actual_pose["palm_press_depth_m"]
        )
        metadata["cube_in_root_m"] = list(
            actual_pose["cube_position_in_root_m"]
        )
        metadata.update(
            {
                "tilt_band_center_deg": band,
                "robustness_case_index": index,
                "robustness_seed": int(seed),
                "search_stage": "robustness_trial",
                "material_policy": "local_pose_and_material_perturbation",
                "parent_candidate_metadata": copy.deepcopy(
                    parent_candidate_metadata
                ),
                "density_scale": density_scale,
                "reference_density_kg_m3": campaign.density_kg_m3,
                "resolved_perturbations": {
                    "cube_center_xy_delta_m": (
                        cube_xy - parent_xy
                    ).tolist(),
                    "cube_rpy_delta_deg": (
                        cube_rpy - parent_cube_rpy
                    ).tolist(),
                    "cube_gap_m": float(candidate["cube"]["z_offset_m"]),
                    "hand_roll_delta_deg": float(roll - parent_rpy[0]),
                    "hand_yaw_delta_deg": float(yaw - parent_rpy[2]),
                    "finger_down_tilt_delta_deg": float(
                        actual_pose["finger_down_tilt_deg"] - parent_tilt
                    ),
                    "palm_press_depth_delta_m": float(
                        actual_pose["palm_press_depth_m"] - parent_press
                    ),
                    "mass_scale": density_scale,
                    "friction_delta": friction_delta,
                    "cube_in_root_delta_m": (
                        np.asarray(actual_pose["cube_position_in_root_m"])
                        - parent_cube
                    ).tolist(),
                },
            }
        )
        candidate["candidate_metadata"] = metadata
        validator(candidate)
        candidates.append(candidate)
    return candidates


def _per_band_override(
    value: int | None,
    *,
    band_count: int,
    default: int,
    label: str,
) -> int:
    if value is None:
        return default
    total = _positive_integer(value, label)
    quotient, remainder = divmod(total, band_count)
    if remainder:
        raise ValueError(f"{label} must be divisible by the five tilt bands")
    return _positive_integer(quotient, f"{label} per band")


def _resolve_budget(
    campaign: AlignedContactCampaignParameters,
    definition: ExperimentDefinition,
    *,
    budget: AlignedTuningBudget | None,
    static_samples_per_edge_band: int | None,
    dynamic_candidates_per_band: int | None,
    grasp_refine_seed_count_per_band: int | None,
    grasp_refine_per_seed: int | None,
    manipulation_seed_count_per_band: int | None,
    manipulation_refine_per_seed: int | None,
    exact_candidates_per_band: int | None,
    perturbation_count: int | None,
    kinematic_samples_per_pitch: int | None,
    dynamic_candidate_count: int | None,
    local_refine_seed_count: int | None,
    local_refine_per_seed: int | None,
    final_candidate_count: int | None,
    perturbations_per_final: int | None,
) -> AlignedTuningBudget:
    direct_values = (
        static_samples_per_edge_band,
        dynamic_candidates_per_band,
        grasp_refine_seed_count_per_band,
        grasp_refine_per_seed,
        manipulation_seed_count_per_band,
        manipulation_refine_per_seed,
        exact_candidates_per_band,
        perturbation_count,
        kinematic_samples_per_pitch,
        dynamic_candidate_count,
        local_refine_seed_count,
        local_refine_per_seed,
        final_candidate_count,
        perturbations_per_final,
    )
    if budget is not None:
        if any(value is not None for value in direct_values):
            raise ValueError("budget cannot be combined with individual overrides")
        return budget
    band_count = len(campaign.tilt_band_centers_deg)

    def choose(
        direct: int | None,
        compatibility: int | None,
        default: int,
        label: str,
    ) -> int:
        if direct is not None and compatibility is not None:
            raise ValueError(f"provide only one {label} override")
        selected = direct if direct is not None else compatibility
        return default if selected is None else _positive_integer(selected, label)

    static_samples = choose(
        static_samples_per_edge_band,
        kinematic_samples_per_pitch,
        campaign.static_samples_per_edge_band,
        "static_samples_per_edge_band",
    )
    dynamic_per_band = (
        _positive_integer(dynamic_candidates_per_band, "dynamic_candidates_per_band")
        if dynamic_candidates_per_band is not None
        else _per_band_override(
            dynamic_candidate_count,
            band_count=band_count,
            default=campaign.dynamic_candidates_per_band,
            label="dynamic_candidate_count",
        )
    )
    grasp_seeds_per_band = (
        _positive_integer(
            grasp_refine_seed_count_per_band,
            "grasp_refine_seed_count_per_band",
        )
        if grasp_refine_seed_count_per_band is not None
        else _per_band_override(
            local_refine_seed_count,
            band_count=band_count,
            default=campaign.grasp_refine_seed_count_per_band,
            label="local_refine_seed_count",
        )
    )
    grasp_samples = choose(
        grasp_refine_per_seed,
        local_refine_per_seed,
        campaign.grasp_refine_per_seed,
        "grasp_refine_per_seed",
    )
    exact_per_band = (
        _positive_integer(exact_candidates_per_band, "exact_candidates_per_band")
        if exact_candidates_per_band is not None
        else _per_band_override(
            final_candidate_count,
            band_count=band_count,
            default=campaign.exact_candidates_per_band,
            label="final_candidate_count",
        )
    )
    if perturbation_count is not None and perturbations_per_final is not None:
        raise ValueError("provide only one perturbation-count override")
    perturbations = (
        perturbation_count
        if perturbation_count is not None
        else perturbations_per_final
        if perturbations_per_final is not None
        else definition.robustness.perturbation_count
    )
    return AlignedTuningBudget(
        static_samples_per_edge_band=static_samples,
        dynamic_candidates_per_band=dynamic_per_band,
        grasp_refine_seed_count_per_band=grasp_seeds_per_band,
        grasp_refine_per_seed=grasp_samples,
        manipulation_seed_count_per_band=(
            campaign.manipulation_seed_count_per_band
            if manipulation_seed_count_per_band is None
            else _positive_integer(
                manipulation_seed_count_per_band,
                "manipulation_seed_count_per_band",
            )
        ),
        manipulation_refine_per_seed=(
            campaign.manipulation_refine_per_seed
            if manipulation_refine_per_seed is None
            else _positive_integer(
                manipulation_refine_per_seed,
                "manipulation_refine_per_seed",
            )
        ),
        exact_candidates_per_band=exact_per_band,
        perturbation_count=_nonnegative_integer(
            perturbations, "perturbation_count"
        ),
    )


def _failure_result(config: Mapping[str, Any], *, band: float) -> dict[str, Any]:
    fallback = copy.deepcopy(dict(config))
    metadata = dict(fallback.get("candidate_metadata", {}))
    metadata["tilt_band_center_deg"] = float(band)
    fallback["candidate_metadata"] = metadata
    return {
        "candidate_id": -1,
        "config": fallback,
        "summary": {
            "passed": False,
            "failed_checks": ["no_dynamic_candidate"],
            "checks": {"no_dynamic_candidate": False},
            "metrics": {},
            "stage_status": {
                "grasp_success": False,
                "manipulation_success": False,
                "full_success": False,
            },
        },
        "search_stage": "static_screen",
        "material_policy": "constant_density",
        "tilt_band_center_deg": float(band),
    }


def _diagnostic_seed_config(
    config: Mapping[str, Any],
    *,
    band: float,
    definition: ExperimentDefinition,
) -> dict[str, Any]:
    """Resolve a valid per-band seed used only for catalog diagnostics."""

    pose = resolved_pose_constraint_values(dict(config))
    rpy = config["hand_pose"]["rpy_deg"]
    cube_in_root = pose["cube_position_in_root_m"]
    candidate = materialize_aligned_candidate(
        config,
        edge_m=float(config["cube"]["edge_m"]),
        tilt_band_center_deg=band,
        roll_deg=float(rpy[0]),
        yaw_deg=float(rpy[2]),
        press_depth_m=float(pose["palm_press_depth_m"]),
        cube_in_root_y_m=float(cube_in_root[1]),
        cube_in_root_z_m=float(cube_in_root[2]),
        cube_yaw_deg=float(config["cube"]["rpy_deg"][2]),
        grasp_targets_rad=config["control"]["grasp_targets_rad"],
        target_assignment=OpposedFaceAssignment.from_mapping(
            config["contact_topology"]["target_faces"]
        ),
    )
    candidate["candidate_metadata"]["diagnostic_seed"] = True
    return candidate


def _compact_near_miss(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    if "summary" not in result:
        return copy.deepcopy(dict(result))
    summary = _nested_mapping(result.get("summary"))
    return {
        "candidate_id": int(result.get("candidate_id", -1)),
        "search_stage": result.get("search_stage"),
        "tilt_band_center_deg": candidate_tilt_band_deg(result),
        "failed_checks": copy.deepcopy(summary.get("failed_checks", [])),
        "stage_status": copy.deepcopy(summary.get("stage_status", {})),
        "metrics": copy.deepcopy(summary.get("metrics", {})),
    }


def tune_aligned_contacts(
    config: dict[str, Any],
    *,
    workers: int,
    seed: int,
    run_candidates: CandidateRunner,
    static_runner: StaticRunner = run_parallel_static_screen,
    perturbation_factory: Callable[..., list[dict[str, Any]]] = (
        generate_aligned_perturbation_configs
    ),
    budget: AlignedTuningBudget | None = None,
    static_samples_per_edge_band: int | None = None,
    dynamic_candidates_per_band: int | None = None,
    grasp_refine_seed_count_per_band: int | None = None,
    grasp_refine_per_seed: int | None = None,
    manipulation_seed_count_per_band: int | None = None,
    manipulation_refine_per_seed: int | None = None,
    exact_candidates_per_band: int | None = None,
    perturbation_count: int | None = None,
    # Compatibility names used by search.tune / command_tune routing.
    kinematic_samples_per_pitch: int | None = None,
    dynamic_candidate_count: int | None = None,
    local_refine_seed_count: int | None = None,
    local_refine_per_seed: int | None = None,
    final_candidate_count: int | None = None,
    perturbations_per_final: int | None = None,
    fallback_physics_count: int | None = None,
    fallback_kinematic_samples_per_pitch: int | None = None,
    legacy_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the complete v4 search with independent per-band hard gates."""

    if config.get("run_context") is not None:
        raise ValueError(
            "schema-v4 campaign tuning requires a canonical config; "
            "parameter_override_run and robustness_trial configs are not eligible"
        )
    validate_config(config)
    definition = resolve_experiment(config)
    if (
        definition.experiment_id != EXPERIMENT_ID
        or definition.tuning_strategy != "aligned_contacts"
        or definition.aligned_contact_campaign is None
        or definition.pose_constraints is None
        or definition.contact_alignment is None
    ):
        raise ValueError("selected experiment is not the registered schema-v4 campaign")
    worker_count = _positive_integer(workers, "workers")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    campaign = definition.aligned_contact_campaign
    bands = tuple(float(value) for value in campaign.tilt_band_centers_deg)
    effective_budget = _resolve_budget(
        campaign,
        definition,
        budget=budget,
        static_samples_per_edge_band=static_samples_per_edge_band,
        dynamic_candidates_per_band=dynamic_candidates_per_band,
        grasp_refine_seed_count_per_band=grasp_refine_seed_count_per_band,
        grasp_refine_per_seed=grasp_refine_per_seed,
        manipulation_seed_count_per_band=manipulation_seed_count_per_band,
        manipulation_refine_per_seed=manipulation_refine_per_seed,
        exact_candidates_per_band=exact_candidates_per_band,
        perturbation_count=perturbation_count,
        kinematic_samples_per_pitch=kinematic_samples_per_pitch,
        dynamic_candidate_count=dynamic_candidate_count,
        local_refine_seed_count=local_refine_seed_count,
        local_refine_per_seed=local_refine_per_seed,
        final_candidate_count=final_candidate_count,
        perturbations_per_final=perturbations_per_final,
    )

    jobs = _effective_static_jobs(
        campaign,
        samples_per_edge_band=effective_budget.static_samples_per_edge_band,
        seed=seed,
    )
    static_raw = static_runner(
        copy.deepcopy(config),
        jobs=jobs,
        retain_per_band=effective_budget.dynamic_candidates_per_band,
        workers=worker_count,
        definition=definition,
    )
    static = _coerce_static_outcome(static_raw, bands=bands)
    dynamic_configs = [
        copy.deepcopy(candidate)
        for band in bands
        for candidate in static.retained_by_band.get(band, ())[
            : effective_budget.dynamic_candidates_per_band
        ]
    ]

    next_candidate_id = 0
    dynamic_results, next_candidate_id = _run_stage(
        dynamic_configs,
        next_candidate_id=next_candidate_id,
        workers=worker_count,
        run_candidates=run_candidates,
        stage="close_verify",
        material_policy="constant_density",
    )
    grasp_parent_groups = _select_per_band(
        dynamic_results,
        bands=bands,
        limit=effective_budget.grasp_refine_seed_count_per_band,
    )
    grasp_configs: list[dict[str, Any]] = []
    for band_index, band in enumerate(bands):
        for parent_index, parent in enumerate(grasp_parent_groups[band]):
            grasp_configs.extend(
                _local_grasp_candidates(
                    parent,
                    parent_candidate_id=int(parent["candidate_id"]),
                    count=effective_budget.grasp_refine_per_seed,
                    seed=(
                        seed
                        + _STAGE_SEED_OFFSETS["grasp_refinement"]
                        + band_index * 1_000_003
                        + parent_index * 10_007
                    ),
                    definition=definition,
                )
            )
    grasp_results, next_candidate_id = _run_stage(
        grasp_configs,
        next_candidate_id=next_candidate_id,
        workers=worker_count,
        run_candidates=run_candidates,
        stage="pose_grasp_refinement",
        material_policy="constant_density",
    )
    grasp_pool = dynamic_results + grasp_results
    qualified_grasp_groups = _select_per_band(
        grasp_pool,
        bands=bands,
        limit=effective_budget.manipulation_seed_count_per_band,
        predicate=grasp_succeeded,
    )
    manipulation_configs: list[dict[str, Any]] = []
    for band_index, band in enumerate(bands):
        for parent_index, parent in enumerate(qualified_grasp_groups[band]):
            manipulation_configs.extend(
                _manipulation_candidates(
                    parent,
                    count=effective_budget.manipulation_refine_per_seed,
                    seed=(
                        seed
                        + _STAGE_SEED_OFFSETS["manipulation_refinement"]
                        + band_index * 1_000_003
                        + parent_index * 10_007
                    ),
                    definition=definition,
                )
            )
    manipulation_results, next_candidate_id = _run_stage(
        manipulation_configs,
        next_candidate_id=next_candidate_id,
        workers=worker_count,
        run_candidates=run_candidates,
        stage="manipulation_refinement",
        material_policy="constant_density",
    )

    exact_parent_groups = _select_per_band(
        manipulation_results,
        bands=bands,
        limit=effective_budget.exact_candidates_per_band,
    )
    exact_configs = [
        copy.deepcopy(parent["config"])
        for band in bands
        for parent in exact_parent_groups[band]
    ]
    exact_results, next_candidate_id = _run_stage(
        exact_configs,
        next_candidate_id=next_candidate_id,
        workers=worker_count,
        run_candidates=run_candidates,
        stage="exact_1ms_confirmation",
        material_policy="constant_density",
    )
    exact_groups = _select_per_band(
        exact_results,
        bands=bands,
        limit=effective_budget.exact_candidates_per_band,
    )
    finalist_by_band: dict[float, dict[str, Any] | None] = {}
    for band in bands:
        hard_passes = [result for result in exact_groups[band] if full_succeeded(result)]
        finalist_by_band[band] = (
            copy.deepcopy(_ranked(hard_passes)[0]) if hard_passes else None
        )
    finalists = [
        finalist_by_band[band]
        for band in bands
        if finalist_by_band[band] is not None
    ]

    completed_before_diagnostics = (
        dynamic_results + grasp_results + manipulation_results + exact_results
    )
    completed_bands = {
        candidate_tilt_band_deg(result) for result in completed_before_diagnostics
    }
    diagnostic_configs: list[dict[str, Any]] = []
    for band in bands:
        if band in completed_bands:
            continue
        static_near = static.near_miss_by_band.get(band)
        if isinstance(static_near, Mapping) and isinstance(
            static_near.get("config"), Mapping
        ):
            diagnostic_configs.append(copy.deepcopy(dict(static_near["config"])))
        else:
            diagnostic_configs.append(
                _diagnostic_seed_config(config, band=band, definition=definition)
            )
    diagnostic_results, next_candidate_id = _run_stage(
        diagnostic_configs,
        next_candidate_id=next_candidate_id,
        workers=worker_count,
        run_candidates=run_candidates,
        stage="catalog_diagnostic_rerun",
        material_policy="constant_density_diagnostic_only",
    )

    selected_band_candidates: list[dict[str, Any]] = []
    for band in bands:
        if finalist_by_band[band] is not None:
            selected_band_candidates.append(copy.deepcopy(finalist_by_band[band]))
            continue
        pool = [
            result
            for result in completed_before_diagnostics + diagnostic_results
            if math.isclose(
                candidate_tilt_band_deg(result),
                band,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ]
        if not pool:
            raise RuntimeError(
                f"tilt band {band:g} has no fully simulated catalog candidate"
            )
        selected_band_candidates.append(copy.deepcopy(_ranked(pool)[0]))

    advanced_pool = (
        exact_results
        if exact_results
        else manipulation_results
        if manipulation_results
        else grasp_pool
    )
    if finalists:
        best_source = _ranked(finalists)[0]
    elif advanced_pool:
        best_source = _ranked(advanced_pool)[0]
    elif diagnostic_results:
        best_source = _ranked(diagnostic_results)[0]
    else:
        best_source = _failure_result(config, band=bands[0])

    catalog_candidates: list[dict[str, Any]] = []
    perturbation_results: list[dict[str, Any]] = []
    perturbation_seed = seed + _STAGE_SEED_OFFSETS["robustness"]
    if (
        int(best_source["candidate_id"]) >= 0
        and effective_budget.perturbation_count
    ):
        catalog_candidates = perturbation_factory(
            best_source["config"],
            count=effective_budget.perturbation_count,
            seed=perturbation_seed,
            definition=definition,
        )
        if len(catalog_candidates) != effective_budget.perturbation_count:
            raise RuntimeError(
                "perturbation_factory must return exactly the requested count"
            )
        perturbation_results, next_candidate_id = _run_stage(
            catalog_candidates,
            next_candidate_id=next_candidate_id,
            workers=worker_count,
            run_candidates=run_candidates,
            stage="best_overall_robustness",
            material_policy="local_pose_and_material_perturbation",
        )
    perturbation_passes = sum(full_succeeded(result) for result in perturbation_results)
    required_perturbation_passes = (
        math.ceil(
            effective_budget.perturbation_count
            * definition.robustness.required_pass_count
            / definition.robustness.perturbation_count
        )
        if effective_budget.perturbation_count
        else 0
    )
    nominal_success = bool(finalists)
    robustness_success = bool(
        nominal_success
        and perturbation_results
        and perturbation_passes >= required_perturbation_passes
    )
    grasp_success = any(grasp_succeeded(result) for result in grasp_pool)
    manipulation_success = any(full_succeeded(result) for result in manipulation_results)

    if robustness_success:
        classification = "validated_aligned_contacts_robust"
        stop_reason = "best_overall_passed_robustness"
    elif nominal_success:
        classification = "aligned_contacts_nominal_hard_pass"
        stop_reason = "nominal_pass_found_robustness_not_passed"
    elif manipulation_results:
        classification = "aligned_contacts_manipulation_near_miss"
        stop_reason = "no_exact_hard_pass"
    elif grasp_success:
        classification = "aligned_contacts_stable_grasp_only"
        stop_reason = "no_manipulation_candidate_passed"
    elif dynamic_results:
        classification = "aligned_contacts_not_validated"
        stop_reason = "no_qualified_grasp"
    else:
        classification = "aligned_contacts_static_gate_empty"
        stop_reason = "no_static_candidate_passed_geometry_gate"

    probe_record = {
        "candidate_id": int(best_source["candidate_id"]),
        "diagnostic_only": not nominal_success,
        "seed": int(perturbation_seed),
        "passes": int(perturbation_passes),
        "required_passes": int(required_perturbation_passes),
        "trial_count": len(perturbation_results),
        "trials": copy.deepcopy(perturbation_results),
    }
    best = copy.deepcopy(best_source)
    best["local_perturbation_probe"] = copy.deepcopy(probe_record)
    best["config"]["experiment_status"] = {
        "classification": classification,
        "passed": nominal_success,
        "grasp_success": grasp_success,
        "manipulation_success": manipulation_success,
        "full_success": nominal_success,
        "constant_density_passed": nominal_success,
        "robustness_passed": robustness_success,
        "stop_reason": stop_reason,
        "note": (
            "A per-band exact candidate and the best-overall perturbation gate passed."
            if robustness_success
            else "At least one band produced an exact hard pass; robustness remains unproven."
            if nominal_success
            else "No exact schema-v4 hard pass was found."
        ),
    }

    per_band: dict[str, dict[str, Any]] = {}
    near_misses: dict[str, dict[str, Any] | None] = {}
    for band in bands:
        static_band_records = [
            record
            for record in static.job_records
            if math.isclose(
                float(record.get("tilt_band_center_deg", math.inf)),
                band,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ]
        static_face_samples = [
            sum(
                int(record.get("face_sample_counts", [0, 0, 0, 0])[face_index])
                for record in static_band_records
            )
            for face_index in range(4)
        ]
        dynamic_band = [
            result for result in dynamic_results if candidate_tilt_band_deg(result) == band
        ]
        grasp_band = [
            result for result in grasp_results if candidate_tilt_band_deg(result) == band
        ]
        manipulation_band = [
            result
            for result in manipulation_results
            if candidate_tilt_band_deg(result) == band
        ]
        exact_band = list(exact_groups[band])
        qualified_count = sum(
            grasp_succeeded(result) for result in dynamic_band + grasp_band
        )
        if finalist_by_band[band] is not None:
            band_stop = "hard_pass_selected"
            near = None
        elif exact_band:
            band_stop = "exact_candidates_failed_hard_checks"
            near = _compact_near_miss(_ranked(exact_band)[0])
        elif manipulation_band:
            band_stop = "no_exact_candidate_available"
            near = _compact_near_miss(_ranked(manipulation_band)[0])
        elif qualified_count:
            band_stop = "qualified_grasp_but_no_manipulation_result"
            qualified = [
                result
                for result in dynamic_band + grasp_band
                if grasp_succeeded(result)
            ]
            near = _compact_near_miss(_ranked(qualified)[0])
        elif dynamic_band or grasp_band:
            band_stop = "no_qualified_grasp"
            near = _compact_near_miss(_ranked(dynamic_band + grasp_band)[0])
        else:
            band_stop = "no_static_candidate_passed_geometry_gate"
            diagnostic_band = [
                result
                for result in diagnostic_results
                if math.isclose(
                    candidate_tilt_band_deg(result),
                    band,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ]
            near = (
                _compact_near_miss(_ranked(diagnostic_band)[0])
                if diagnostic_band
                else copy.deepcopy(static.near_miss_by_band.get(band))
            )
        key = f"{band:g}"
        per_band[key] = {
            "tilt_band_center_deg": band,
            "static_sample_count": sum(
                int(record.get("sample_count", 0)) for record in static_band_records
            ),
            "static_face_sample_counts": static_face_samples,
            "static_face_sample_allocation": [
                {
                    "face_index": face_index,
                    "target_faces": definition.candidate_faces[
                        face_index
                    ].as_dict(),
                    "sample_count": static_face_samples[face_index],
                }
                for face_index in range(4)
            ],
            "static_valid_pose_count": sum(
                int(record.get("valid_pose_count", 0))
                for record in static_band_records
            ),
            "static_clean_three_count": sum(
                int(record.get("clean_three_count", 0))
                for record in static_band_records
            ),
            "static_near_three_count": sum(
                int(record.get("near_three_count", 0))
                for record in static_band_records
            ),
            "static_forbidden_count": sum(
                int(record.get("forbidden_count", 0))
                for record in static_band_records
            ),
            "static_penetration_exceeded_count": sum(
                int(record.get("penetration_exceeded_count", 0))
                for record in static_band_records
            ),
            "static_eligible_count": sum(
                int(record.get("eligible_count", 0))
                for record in static_band_records
            ),
            "static_local_retained_count": sum(
                int(record.get("local_reservoir_count", 0))
                for record in static_band_records
            ),
            "static_retained_count": len(static.retained_by_band.get(band, ())),
            "dynamic_count": len(dynamic_band),
            "grasp_parent_count": len(grasp_parent_groups[band]),
            "grasp_refinement_count": len(grasp_band),
            "qualified_grasp_count": int(qualified_count),
            "manipulation_parent_count": len(qualified_grasp_groups[band]),
            "manipulation_refinement_count": len(manipulation_band),
            "exact_parent_count": len(exact_parent_groups[band]),
            "exact_confirmation_count": len(exact_band),
            "exact_hard_pass_count": sum(full_succeeded(item) for item in exact_band),
            "selected_finalist_candidate_id": (
                None
                if finalist_by_band[band] is None
                else int(finalist_by_band[band]["candidate_id"])
            ),
            "stop_reason": band_stop,
        }
        near_misses[key] = near

    def result_has_edge(result: Mapping[str, Any], edge_m: float) -> bool:
        try:
            result_edge = float(result["config"]["cube"]["edge_m"])
        except (KeyError, TypeError, ValueError):
            return False
        return math.isclose(result_edge, edge_m, rel_tol=0.0, abs_tol=1e-12)

    per_size: dict[str, dict[str, Any]] = {}
    for edge in (float(value) for value in campaign.edges_m):
        static_edge_records = [
            record
            for record in static.job_records
            if math.isclose(
                float(record.get("edge_m", math.inf)),
                edge,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        dynamic_edge = [
            result for result in dynamic_results if result_has_edge(result, edge)
        ]
        grasp_edge = [
            result for result in grasp_results if result_has_edge(result, edge)
        ]
        manipulation_edge = [
            result
            for result in manipulation_results
            if result_has_edge(result, edge)
        ]
        exact_edge = [
            result for result in exact_results if result_has_edge(result, edge)
        ]
        edge_finalists = [
            result for result in finalists if result_has_edge(result, edge)
        ]
        qualified_edge = [
            result
            for result in dynamic_edge + grasp_edge
            if grasp_succeeded(result)
        ]
        if edge_finalists:
            edge_stop = "hard_pass_selected"
        elif exact_edge:
            edge_stop = "exact_candidates_failed_hard_checks"
        elif manipulation_edge:
            edge_stop = "no_exact_hard_pass"
        elif qualified_edge:
            edge_stop = "qualified_grasp_but_no_manipulation_result"
        elif dynamic_edge or grasp_edge:
            edge_stop = "no_qualified_grasp"
        else:
            edge_stop = "no_static_candidate_selected_for_dynamic"
        edge_key = f"{edge * 1000.0:g}"
        per_size[edge_key] = {
            "edge_m": edge,
            "edge_mm": edge * 1000.0,
            "static_sample_count": sum(
                int(record.get("sample_count", 0))
                for record in static_edge_records
            ),
            "static_face_sample_counts": [
                sum(
                    int(
                        record.get("face_sample_counts", [0, 0, 0, 0])[
                            face_index
                        ]
                    )
                    for record in static_edge_records
                )
                for face_index in range(4)
            ],
            "static_valid_pose_count": sum(
                int(record.get("valid_pose_count", 0))
                for record in static_edge_records
            ),
            "static_clean_three_count": sum(
                int(record.get("clean_three_count", 0))
                for record in static_edge_records
            ),
            "static_near_three_count": sum(
                int(record.get("near_three_count", 0))
                for record in static_edge_records
            ),
            "static_forbidden_count": sum(
                int(record.get("forbidden_count", 0))
                for record in static_edge_records
            ),
            "static_penetration_exceeded_count": sum(
                int(record.get("penetration_exceeded_count", 0))
                for record in static_edge_records
            ),
            "static_eligible_count": sum(
                int(record.get("eligible_count", 0))
                for record in static_edge_records
            ),
            "static_local_retained_count": sum(
                int(record.get("local_reservoir_count", 0))
                for record in static_edge_records
            ),
            "static_retained_count": len(dynamic_edge),
            "dynamic_count": len(dynamic_edge),
            "dynamic_grasp_pass_count": sum(
                grasp_succeeded(result) for result in dynamic_edge
            ),
            "dynamic_hard_pass_count": sum(
                full_succeeded(result) for result in dynamic_edge
            ),
            "grasp_refinement_count": len(grasp_edge),
            "grasp_refinement_grasp_pass_count": sum(
                grasp_succeeded(result) for result in grasp_edge
            ),
            "grasp_refinement_hard_pass_count": sum(
                full_succeeded(result) for result in grasp_edge
            ),
            "manipulation_refinement_count": len(manipulation_edge),
            "manipulation_grasp_pass_count": sum(
                grasp_succeeded(result) for result in manipulation_edge
            ),
            "manipulation_hard_pass_count": sum(
                full_succeeded(result) for result in manipulation_edge
            ),
            "exact_confirmation_count": len(exact_edge),
            "exact_grasp_pass_count": sum(
                grasp_succeeded(result) for result in exact_edge
            ),
            "exact_hard_pass_count": sum(
                full_succeeded(result) for result in exact_edge
            ),
            "selected_finalist_count": len(edge_finalists),
            "stop_reason": edge_stop,
        }

    all_main_results = dynamic_results + grasp_results + manipulation_results + exact_results
    ranked_advanced = _ranked(advanced_pool) if advanced_pool else [best_source]
    top_candidates: list[dict[str, Any]] = []
    for result in ranked_advanced[:20]:
        item = copy.deepcopy(result)
        if int(item["candidate_id"]) == int(best_source["candidate_id"]):
            item["local_perturbation_probe"] = copy.deepcopy(probe_record)
        top_candidates.append(item)
    effective_manifest = budget_manifest(campaign)
    effective_manifest["effective_budget"] = effective_budget.as_dict(
        edge_count=len(campaign.edges_m), band_count=len(bands)
    )
    effective_manifest["static_execution"] = {
        "job_count": len(jobs),
        "spawn_parallel": True,
        "worker_count": worker_count,
        "hard_geometry_gate_required": True,
        "hard_geometry_gate": {
            "forbidden_contact": False,
            "max_penetration_m": float(config["acceptance"]["max_penetration_m"]),
            "near_target_face_count": len(ACTIVE_FINGERS),
            "target_site_signed_distance_m": [-0.002, 0.003],
            "clean_target_contact_count_is_ranking_only": True,
            "estimated_contact_height_is_ranking_only": True,
        },
    }
    effective_manifest["perturbation_envelope"] = (
        campaign.perturbation_envelope.as_config()
    )
    effective_manifest["stage_seed_offsets"] = dict(_STAGE_SEED_OFFSETS)
    effective_manifest["best_overall_perturbation_seed"] = int(
        perturbation_seed
    )
    effective_manifest["compatibility_overrides_not_used"] = {
        "fallback_physics_count": fallback_physics_count,
        "fallback_kinematic_samples_per_pitch": (
            fallback_kinematic_samples_per_pitch
        ),
        "legacy_parameters": (
            {} if legacy_parameters is None else dict(legacy_parameters)
        ),
    }
    stage_counts = {
        "static_sample_count": static.sample_count,
        "static_retained_count": len(dynamic_configs),
        "dynamic_close_verify_count": len(dynamic_results),
        "grasp_refinement_count": len(grasp_results),
        "manipulation_refinement_count": len(manipulation_results),
        "exact_1ms_confirmation_count": len(exact_results),
        "catalog_diagnostic_rerun_count": len(diagnostic_results),
        "per_band_finalist_count": len(finalists),
        "best_overall_perturbation_count": len(perturbation_results),
    }
    return {
        "experiment_id": definition.experiment_id,
        "campaign_kind": "aligned_contacts",
        "campaign_classification": classification,
        "campaign_status": copy.deepcopy(best["config"]["experiment_status"]),
        "seed": int(seed),
        "workers": worker_count,
        "campaign_manifest": effective_manifest,
        "static_jobs": [copy.deepcopy(record) for record in static.job_records],
        "static_diagnostics_by_band": {
            f"{band:g}": [copy.deepcopy(item) for item in static.diagnostics_by_band[band]]
            for band in bands
        },
        "stage_counts": stage_counts,
        "per_band": per_band,
        "per_size": per_size,
        "finalists": [copy.deepcopy(item) for item in finalists],
        "selected_band_candidates": copy.deepcopy(selected_band_candidates),
        "near_misses": near_misses,
        "catalog_candidates": copy.deepcopy(catalog_candidates),
        "catalog_results": copy.deepcopy(perturbation_results),
        "candidate_count": len(all_main_results),
        "diagnostic_simulation_count": len(diagnostic_results),
        "kinematic_sample_count": static.sample_count,
        "perturbation_probe_count": len(perturbation_results),
        "perturbation_seed": int(perturbation_seed),
        "simulation_count": (
            len(all_main_results)
            + len(diagnostic_results)
            + len(perturbation_results)
        ),
        "passing_candidates": len(finalists),
        "stable_grasp_candidate_count": sum(
            grasp_succeeded(result) for result in grasp_pool
        ),
        "grasp_success": grasp_success,
        "manipulation_success": manipulation_success,
        "fixed_mass_success": False,
        "constant_density_success": nominal_success,
        "nominal_success": nominal_success,
        "robustness_success": robustness_success,
        "stop_reason": stop_reason,
        "best_fixed_mass": None,
        "best": best,
        "top_candidates": top_candidates,
        "local_perturbation_probes": [probe_record] if perturbation_results else [],
    }


__all__ = [
    "ALIGNED_PERTURBATION_RANGES",
    "AlignedPerturbationRanges",
    "AlignedStaticJob",
    "AlignedTuningBudget",
    "CandidateRunner",
    "EXACT_TIMESTEP_S",
    "StaticRetainedCandidate",
    "StaticRunner",
    "StaticScreenOutcome",
    "aligned_static_candidate_advances",
    "cube_world_position_m",
    "generate_aligned_perturbation_configs",
    "materialize_aligned_candidate",
    "root_pitch_for_finger_down_tilt_deg",
    "run_parallel_static_screen",
    "tune_aligned_contacts",
]
