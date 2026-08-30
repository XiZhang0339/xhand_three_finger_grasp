"""Deterministic schema-v5 far-hand fingertip search and robustness.

The static stage deliberately uses MuJoCo collision geoms, not tactile-site
plane distances.  It scans a closure path before dynamics so a compliant final
pose may carry distal preload without allowing proximal or palm penetration.
All dynamic stages are injectable, keeping the orchestration deterministic and
cheap to unit test.
"""

from __future__ import annotations

import copy
import heapq
import json
import math
import multiprocessing
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from ..config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    resolved_pose_constraint_values,
    validate_config,
)
from ..contact_geometry import (
    active_nondistal_collision_geom_ids,
    distal_collision_geom_ids,
    nearest_taxel_assignment,
    scan_distal_closure,
)
from ..contacts import BoxContactThresholds
from ..evaluation import face_from_label
from ..experiment import (
    ExperimentDefinition,
    FarHandFingertipCampaignParameters,
    OpposedFaceAssignment,
    resolve_experiment,
)
from ..experiments.opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift import (
    EXPERIMENT_ID,
    FAR_HAND_CAMPAIGN,
    THUMB_BEND_ACTUATOR,
)
from ..larger_cube_grasp_search import (
    full_succeeded,
    grasp_succeeded,
    manipulation_succeeded,
    sample_manipulation_delta_candidates,
)
from ..scene import (
    ModelInfo,
    build_model,
    cube_vertical_half_extent_m,
    palm_plane_ground_angle_deg,
    rpy_degrees_to_quaternion,
    rpy_degrees_to_rotation_matrix,
    signed_finger_down_tilt_deg,
)


CandidateRunner = Callable[
    [list[tuple[int, dict[str, Any]]], int], list[dict[str, Any]]
]
StaticRunner = Callable[..., Any]
Validator = Callable[[dict[str, Any]], None]

GRAVITY_WORLD_M_S2 = (0.0, 0.0, -9.81)
EXACT_TIMESTEP_S = 0.001
_STAGE_SEED_OFFSET = {
    "static": 510_000_000,
    "grasp": 520_000_000,
    "boundary": 530_000_000,
    "manipulation": 540_000_000,
    "robustness": 550_000_000,
}


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _scale(unit: float, bounds: Sequence[float]) -> float:
    return float(bounds[0] + unit * (bounds[1] - bounds[0]))


def _latin_hypercube(
    samples: int, dimensions: int, rng: np.random.Generator
) -> np.ndarray:
    _positive_int(samples, "samples")
    _positive_int(dimensions, "dimensions")
    values = np.empty((samples, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        values[:, dimension] = (
            rng.permutation(samples) + rng.random(samples)
        ) / samples
    return values


def _tilt_band_bounds_deg(
    band: float,
    *,
    bands: Sequence[float],
    limits: Sequence[float],
) -> tuple[float, float]:
    """Return a disjoint resolved-tilt interval around one catalog band.

    A band label is a reporting category, not a requirement that every pose
    use exactly the centre value.  In particular, a non-zero root roll at the
    20 degree centre makes the palm-plane angle slightly greater than the
    declared 20 degree limit.  Midpoint partitions keep every sample assigned
    to exactly one band while leaving the edge bands a non-zero feasible
    interval.
    """

    ordered = tuple(float(value) for value in bands)
    selected = float(band)
    if selected not in ordered:
        raise ValueError(f"undeclared tilt band: {selected:g}")
    index = ordered.index(selected)
    lower = (
        float(limits[0])
        if index == 0
        else 0.5 * (ordered[index - 1] + selected)
    )
    upper = (
        float(limits[1])
        if index == len(ordered) - 1
        else 0.5 * (selected + ordered[index + 1])
    )
    if lower > upper:
        raise ValueError("tilt bands and limits are inconsistent")
    return lower, upper


def _cube_world_position_m(config: Mapping[str, Any]) -> np.ndarray:
    cube = config["cube"]
    rotation = rpy_degrees_to_rotation_matrix(
        cube.get("rpy_deg", [0.0, 0.0, 0.0])
    )
    return np.asarray(
        [
            float(cube["center_xy_m"][0]),
            float(cube["center_xy_m"][1]),
            float(config["scene"]["support_top_z_m"])
            + cube_vertical_half_extent_m(float(cube["edge_m"]), rotation)
            + float(cube.get("z_offset_m", 0.0)),
        ],
        dtype=np.float64,
    )


def root_pitch_for_finger_down_tilt_deg(
    roll_deg: float, tilt_deg: float
) -> float:
    """Resolve the same signed local-+Z tilt convention as schema v4."""

    roll = math.radians(_finite(roll_deg, "roll_deg"))
    tilt = math.radians(_finite(tilt_deg, "tilt_deg"))
    cosine_roll = math.cos(roll)
    if abs(cosine_roll) <= 1e-9:
        raise ValueError("finger-down pitch solve is singular")
    cosine_pitch = -math.sin(tilt) / cosine_roll
    if not -1.0 - 1e-12 <= cosine_pitch <= 1.0 + 1e-12:
        raise ValueError("requested finger-down tilt is infeasible")
    return float(math.degrees(math.acos(np.clip(cosine_pitch, -1.0, 1.0))))


def _feasible_tilt_for_roll_deg(
    desired_tilt_deg: float,
    *,
    roll_deg: float,
    constraints: Any,
) -> float:
    """Clip a resolved finger tilt to the coupled palm-angle envelope."""

    tilt_lower, tilt_upper = constraints.finger_down_tilt_deg
    palm_lower, palm_upper = constraints.palm_plane_ground_angle_deg

    def palm_angle(tilt: float) -> float:
        rpy = [
            float(roll_deg),
            root_pitch_for_finger_down_tilt_deg(roll_deg, tilt),
            0.0,
        ]
        return palm_plane_ground_angle_deg(
            rpy_degrees_to_rotation_matrix(rpy), GRAVITY_WORLD_M_S2
        )

    feasible_lower = float(tilt_lower)
    feasible_upper = float(tilt_upper)
    if palm_angle(feasible_upper) > palm_upper + 1e-12:
        low, high = feasible_lower, feasible_upper
        for _ in range(60):
            middle = 0.5 * (low + high)
            if palm_angle(middle) <= palm_upper:
                low = middle
            else:
                high = middle
        feasible_upper = low
    if palm_angle(feasible_lower) < palm_lower - 1e-12:
        low, high = feasible_lower, feasible_upper
        for _ in range(60):
            middle = 0.5 * (low + high)
            if palm_angle(middle) < palm_lower:
                low = middle
            else:
                high = middle
        feasible_lower = high
    if feasible_lower > feasible_upper + 1e-10:
        raise ValueError("root roll has no feasible v5 tilt/palm-angle solution")
    return float(np.clip(desired_tilt_deg, feasible_lower, feasible_upper))


def _resolved_root_pose(
    *,
    config: Mapping[str, Any],
    roll_deg: float,
    tilt_deg: float,
    yaw_deg: float,
    cube_in_root_m: Sequence[float],
) -> tuple[list[float], list[float]]:
    rpy = [
        _finite(roll_deg, "roll_deg"),
        root_pitch_for_finger_down_tilt_deg(roll_deg, tilt_deg),
        _finite(yaw_deg, "yaw_deg"),
    ]
    rotation = rpy_degrees_to_rotation_matrix(rpy)
    relative = np.asarray(cube_in_root_m, dtype=np.float64)
    if relative.shape != (3,) or not np.isfinite(relative).all():
        raise ValueError("cube_in_root_m must contain three finite values")
    translation = _cube_world_position_m(config) - rotation @ relative
    return rpy, translation.tolist()


@dataclass(frozen=True)
class FarHandTuningBudget:
    primary_static_samples_per_band: int
    fallback_static_samples_per_band: int
    static_retain_per_band: int
    dynamic_candidates_per_band: int
    grasp_refine_seed_count_per_band: int
    grasp_refine_per_seed: int
    manipulation_seed_count_per_band: int
    manipulation_refine_per_seed: int
    exact_candidates_per_band: int
    perturbation_count: int

    def __post_init__(self) -> None:
        for label in (
            "primary_static_samples_per_band",
            "fallback_static_samples_per_band",
            "static_retain_per_band",
            "dynamic_candidates_per_band",
            "grasp_refine_seed_count_per_band",
            "grasp_refine_per_seed",
            "manipulation_seed_count_per_band",
            "manipulation_refine_per_seed",
            "exact_candidates_per_band",
        ):
            _positive_int(getattr(self, label), label)
        _nonnegative_int(self.perturbation_count, "perturbation_count")
        if self.dynamic_candidates_per_band > self.static_retain_per_band:
            raise ValueError("dynamic candidates cannot exceed static retention")

    @classmethod
    def from_campaign(
        cls,
        campaign: FarHandFingertipCampaignParameters,
        *,
        perturbation_count: int,
    ) -> "FarHandTuningBudget":
        return cls(
            primary_static_samples_per_band=(
                campaign.primary_static_samples_per_band
            ),
            fallback_static_samples_per_band=(
                campaign.fallback_static_samples_per_band
            ),
            static_retain_per_band=campaign.static_retain_per_band,
            dynamic_candidates_per_band=campaign.dynamic_candidates_per_band,
            grasp_refine_seed_count_per_band=(
                campaign.grasp_refine_seed_count_per_band
            ),
            grasp_refine_per_seed=campaign.grasp_refine_per_seed,
            manipulation_seed_count_per_band=(
                campaign.manipulation_seed_count_per_band
            ),
            manipulation_refine_per_seed=campaign.manipulation_refine_per_seed,
            exact_candidates_per_band=campaign.exact_candidates_per_band,
            perturbation_count=perturbation_count,
        )

    def as_dict(self, *, band_count: int) -> dict[str, int]:
        per_band = (
            self.primary_static_samples_per_band
            + self.fallback_static_samples_per_band
        )
        return {
            "band_count": int(band_count),
            "static_samples_per_band": per_band,
            "static_sample_count": int(band_count * per_band),
            "static_retain_per_band": self.static_retain_per_band,
            "dynamic_candidate_count": (
                band_count * self.dynamic_candidates_per_band
            ),
            "grasp_refinement_count": (
                band_count
                * self.grasp_refine_seed_count_per_band
                * self.grasp_refine_per_seed
            ),
            "manipulation_refinement_count": (
                band_count
                * self.manipulation_seed_count_per_band
                * self.manipulation_refine_per_seed
            ),
            "exact_candidate_count": (
                band_count * self.exact_candidates_per_band
            ),
            "perturbation_count": self.perturbation_count,
        }


@dataclass(frozen=True)
class FarHandStaticJob:
    job_id: int
    first_candidate_id: int
    tilt_band_center_deg: float
    topology_kind: str
    target_assignment: OpposedFaceAssignment
    samples: int
    seed: int


@dataclass(frozen=True)
class FarHandStaticOutcome:
    sample_count: int
    retained_by_band: dict[float, tuple[dict[str, Any], ...]]
    diagnostics_by_band: dict[float, tuple[dict[str, Any], ...]]
    near_miss_by_band: dict[float, dict[str, Any] | None]
    job_records: tuple[dict[str, Any], ...]


def materialize_far_hand_candidate(
    base: Mapping[str, Any],
    *,
    tilt_band_center_deg: float,
    roll_deg: float,
    yaw_deg: float,
    cube_in_root_m: Sequence[float],
    cube_yaw_deg: float,
    grasp_targets_rad: Mapping[str, float],
    target_assignment: OpposedFaceAssignment | Mapping[str, str],
    manipulation_delta_rad: Mapping[str, float] | None = None,
    finger_down_tilt_deg: float | None = None,
    static_candidate_id: int | None = None,
    static_diagnostic: Mapping[str, Any] | None = None,
    boundary_expansion: Mapping[str, Any] | None = None,
    validator: Validator | None = validate_config,
) -> dict[str, Any]:
    """Resolve one immutable-root schema-v5 candidate."""

    candidate = copy.deepcopy(dict(base))
    candidate.pop("experiment_status", None)
    candidate.pop("run_context", None)
    definition = resolve_experiment(candidate)
    campaign = definition.far_hand_campaign
    if definition.experiment_id != EXPERIMENT_ID or campaign is None:
        raise ValueError("materialization requires the registered schema-v5 campaign")
    if set(grasp_targets_rad) != set(ACTIVE_ACTUATORS):
        raise ValueError("grasp_targets_rad must contain exactly eight actuators")
    grasp = {
        name: _finite(grasp_targets_rad[name], f"grasp_targets_rad.{name}")
        for name in ACTIVE_ACTUATORS
    }
    if manipulation_delta_rad is None:
        manipulation = {name: 0.0 for name in ACTIVE_ACTUATORS}
    else:
        if set(manipulation_delta_rad) != set(ACTIVE_ACTUATORS):
            raise ValueError(
                "manipulation_delta_rad must contain exactly eight actuators"
            )
        manipulation = {
            name: _finite(
                manipulation_delta_rad[name], f"manipulation_delta_rad.{name}"
            )
            for name in ACTIVE_ACTUATORS
        }
    assignment = (
        target_assignment
        if isinstance(target_assignment, OpposedFaceAssignment)
        else OpposedFaceAssignment.from_mapping(target_assignment)
    )
    if assignment not in definition.candidate_faces:
        raise ValueError("target assignment is outside the v5 face policy")
    relative = np.asarray(cube_in_root_m, dtype=np.float64)
    if relative.shape != (3,) or not np.isfinite(relative).all():
        raise ValueError("cube_in_root_m must contain three finite values")

    candidate["cube"]["edge_m"] = campaign.nominal_edge_m
    candidate["cube"]["mass_kg"] = campaign.nominal_mass_kg
    candidate["cube"]["friction"] = campaign.friction
    candidate["cube"]["rpy_deg"] = [
        0.0,
        0.0,
        _finite(cube_yaw_deg, "cube_yaw_deg"),
    ]
    tilt = (
        _finite(tilt_band_center_deg, "tilt_band_center_deg")
        if finger_down_tilt_deg is None
        else _finite(finger_down_tilt_deg, "finger_down_tilt_deg")
    )
    rpy, translation = _resolved_root_pose(
        config=candidate,
        roll_deg=roll_deg,
        tilt_deg=tilt,
        yaw_deg=yaw_deg,
        cube_in_root_m=relative,
    )
    candidate["hand_pose"] = {
        "translation_m": translation,
        "rpy_deg": rpy,
    }
    candidate["control"] = {
        "grasp_targets_rad": grasp,
        "manipulation_delta_rad": manipulation,
    }
    candidate["contact_topology"]["target_faces"] = assignment.as_dict()
    metadata = dict(candidate.get("candidate_metadata", {}))
    metadata.update(
        {
            "tilt_band_center_deg": float(tilt_band_center_deg),
            "resolved_finger_down_tilt_deg": float(tilt),
            "cube_in_root_m": relative.tolist(),
            "root_cube_distance_m": float(np.linalg.norm(relative)),
            "legacy_palm_press_depth_m": float(
                definition.far_hand_pose_constraints.legacy_press_reference_translation_m[2]
                - translation[2]
            ),
        }
    )
    if static_candidate_id is not None:
        metadata["static_candidate_id"] = int(static_candidate_id)
    if static_diagnostic is not None:
        metadata["static_closure_diagnostic"] = copy.deepcopy(
            dict(static_diagnostic)
        )
    if boundary_expansion is not None:
        metadata["boundary_expansion"] = copy.deepcopy(dict(boundary_expansion))
    candidate["candidate_metadata"] = metadata
    if validator is not None:
        validator(candidate)
    return candidate


def far_hand_static_jobs(
    *,
    campaign: FarHandFingertipCampaignParameters = FAR_HAND_CAMPAIGN,
    seed: int | None = None,
    primary_samples_per_band: int | None = None,
    fallback_samples_per_band: int | None = None,
) -> tuple[FarHandStaticJob, ...]:
    """Return five primary and five fallback jobs with disjoint IDs/seeds."""

    effective_seed = 20260821 if seed is None else int(seed)
    if effective_seed < 0:
        raise ValueError("seed must be non-negative")
    primary_count = _positive_int(
        campaign.primary_static_samples_per_band
        if primary_samples_per_band is None
        else primary_samples_per_band,
        "primary_samples_per_band",
    )
    fallback_count = _positive_int(
        campaign.fallback_static_samples_per_band
        if fallback_samples_per_band is None
        else fallback_samples_per_band,
        "fallback_samples_per_band",
    )
    jobs: list[FarHandStaticJob] = []
    first_id = 5  # IDs 0--4 are reserved for evidence anchors.
    for band_index, band in enumerate(campaign.tilt_band_centers_deg):
        for topology_index, (kind, assignment, samples) in enumerate(
            (
                ("primary", campaign.primary_face, primary_count),
                ("fallback", campaign.fallback_face, fallback_count),
            )
        ):
            job_id = 2 * band_index + topology_index
            jobs.append(
                FarHandStaticJob(
                    job_id=job_id,
                    first_candidate_id=first_id,
                    tilt_band_center_deg=float(band),
                    topology_kind=kind,
                    target_assignment=assignment,
                    samples=samples,
                    seed=effective_seed + _STAGE_SEED_OFFSET["static"] + job_id,
                )
            )
            first_id += samples
    return tuple(jobs)


def _collision_geom_sets(
    model: mujoco.MjModel, info: ModelInfo
) -> tuple[
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, ...]],
    tuple[int, ...],
]:
    distal = distal_collision_geom_ids(model, info.distal_weld_ids)
    active_nondistal = active_nondistal_collision_geom_ids(
        model, info.hand_body_parts, info.distal_weld_ids
    )
    claimed = {
        geom_id
        for values in tuple(distal.values()) + tuple(active_nondistal.values())
        for geom_id in values
    }
    forbidden: list[int] = []
    for geom_id in range(model.ngeom):
        if (
            int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
        ):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        part = info.hand_body_parts.get(body_id)
        if part is None:
            continue
        if geom_id not in claimed:
            # Palm and inactive fingers are forbidden.  Active proximal links
            # are passed separately so diagnostics preserve the distinction.
            forbidden.append(geom_id)
    return distal, active_nondistal, tuple(forbidden)


def _distal_site_ids(
    model: mujoco.MjModel, info: ModelInfo
) -> dict[str, np.ndarray]:
    sites = {
        finger: np.asarray(
            [
                site_id
                for site_id in range(model.nsite)
                if int(model.body_weldid[int(model.site_bodyid[site_id])])
                == info.distal_weld_ids[finger]
            ],
            dtype=np.intp,
        )
        for finger in ACTIVE_FINGERS
    }
    if any(values.size == 0 for values in sites.values()):
        raise ValueError("active distal tactile site mapping is incomplete")
    return sites


def _geom_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
    other_geom_id: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    from_to = np.zeros(6, dtype=np.float64)
    distance = float(
        mujoco.mj_geomDistance(
            model,
            data,
            int(cube_geom_id),
            int(other_geom_id),
            1.0,
            from_to,
        )
    )
    return distance, from_to[:3].copy(), from_to[3:].copy()


def _target_witness(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: ModelInfo,
    *,
    finger: str,
    target_label: str,
    distal_geom_ids: Mapping[str, Sequence[int]],
    thresholds: BoxContactThresholds,
    site_ids: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    face = face_from_label(target_label)
    cube_rotation = data.geom_xmat[info.cube_geom_id].reshape(3, 3)
    cube_position = data.geom_xpos[info.cube_geom_id]
    half_extent = model.geom_size[info.cube_geom_id]
    tangential = [axis for axis in range(3) if axis != face.axis]
    best: tuple[float, int, dict[str, Any]] | None = None
    for geom_id in distal_geom_ids[finger]:
        distance, cube_point, finger_point = _geom_distance(
            model, data, info.cube_geom_id, int(geom_id)
        )
        local = cube_rotation.T @ (cube_point - cube_position)
        surface_error = abs(
            float(local[face.axis]) - face.sign * float(half_extent[face.axis])
        )
        edge_clearance = float(
            np.min(half_extent[tangential] - np.abs(local[tangential]))
        )
        separation = finger_point - cube_point
        separation_norm = float(np.linalg.norm(separation))
        if separation_norm > 1e-12:
            normal_local = cube_rotation.T @ (separation / separation_norm)
            normal_alignment = float(normal_local @ face.outward_normal)
        else:
            normal_alignment = 1.0
        if (
            surface_error > thresholds.surface_tolerance_m + 1e-9
            or edge_clearance + 1e-12 < thresholds.edge_margin_m
            or normal_alignment + 1e-12 < thresholds.normal_alignment_min
        ):
            continue
        tactile_positions = data.site_xpos[site_ids[finger]]
        tactile_distances = np.linalg.norm(
            tactile_positions - finger_point[np.newaxis, :], axis=1
        )
        nearest_taxel_distance = float(np.min(tactile_distances))
        record = {
            "signed_gap_m": distance,
            "cube_witness_world_m": cube_point.tolist(),
            "finger_witness_world_m": finger_point.tolist(),
            "surface_error_m": surface_error,
            "edge_clearance_m": edge_clearance,
            "normal_alignment": normal_alignment,
            "nearest_taxel_distance_m": nearest_taxel_distance,
            "geom_id": int(geom_id),
        }
        key = (abs(distance), int(geom_id))
        if best is None or key < best[:2]:
            best = (key[0], key[1], record)
    if best is None:
        return {
            "signed_gap_m": math.inf,
            "cube_witness_world_m": None,
            "finger_witness_world_m": None,
            "surface_error_m": math.inf,
            "edge_clearance_m": -math.inf,
            "normal_alignment": -math.inf,
            "nearest_taxel_distance_m": math.inf,
            "geom_id": None,
        }
    return best[2]


def far_hand_static_candidate_advances(
    diagnostic: Mapping[str, Any],
    *,
    target_signed_gap_m: tuple[float, float] = (-0.0005, 0.003),
    max_static_distal_preload_m: float = 0.010,
) -> bool:
    """Pure v5 static gate used by workers and regression tests."""

    lower, upper = (
        _finite(target_signed_gap_m[0], "target_signed_gap_m[0]"),
        _finite(target_signed_gap_m[1], "target_signed_gap_m[1]"),
    )
    if lower > upper:
        raise ValueError("target_signed_gap_m must be ordered")
    try:
        gaps = tuple(float(value) for value in diagnostic["selected_signed_gap_m"])
        preload = float(diagnostic["full_target_distal_preload_m"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        not diagnostic.get("forbidden_penetration", True)
        and len(gaps) == len(ACTIVE_FINGERS)
        and all(
            math.isfinite(value) and lower - 1e-12 <= value <= upper + 1e-12
            for value in gaps
        )
        and math.isfinite(preload)
        and preload <= max_static_distal_preload_m + 1e-12
    )


def closure_sweep_diagnostic(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: ModelInfo,
    config: Mapping[str, Any],
    *,
    hand_rpy_deg: Sequence[float],
    cube_in_root_m: Sequence[float],
    cube_yaw_deg: float,
    grasp_targets_rad: Mapping[str, float],
    target_assignment: OpposedFaceAssignment,
    distal_geom_ids: Mapping[str, Sequence[int]] | None = None,
    active_nondistal_geom_ids: Mapping[str, Sequence[int]] | None = None,
    forbidden_geom_ids: Sequence[int] | None = None,
    distal_site_ids: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Inspect a candidate over the registered 80--100% closure path."""

    definition = resolve_experiment(config)
    campaign = definition.far_hand_campaign
    preferences = definition.fingertip_contact_preferences
    if campaign is None or preferences is None:
        raise ValueError("closure sweep requires the schema-v5 campaign")
    if set(grasp_targets_rad) != set(ACTIVE_ACTUATORS):
        raise ValueError("grasp_targets_rad must contain active actuators")
    if (
        distal_geom_ids is None
        or active_nondistal_geom_ids is None
        or forbidden_geom_ids is None
    ):
        derived_distal, derived_nondistal, derived_forbidden = (
            _collision_geom_sets(model, info)
        )
        distal_geom_ids = derived_distal
        active_nondistal_geom_ids = derived_nondistal
        forbidden_geom_ids = derived_forbidden
    if distal_site_ids is None:
        distal_site_ids = _distal_site_ids(model, info)
    topology = config["contact_topology"]
    thresholds = BoxContactThresholds(
        surface_tolerance_m=float(topology["surface_tolerance_m"]),
        edge_margin_m=float(topology["edge_margin_m"]),
        normal_alignment_min=float(topology["min_normal_alignment"]),
    )
    relative = np.asarray(cube_in_root_m, dtype=np.float64)
    rotation = rpy_degrees_to_rotation_matrix(hand_rpy_deg)
    root_translation = _cube_world_position_m(config) - rotation @ relative
    cube_world = _cube_world_position_m(config)
    labels = target_assignment.as_dict()
    mujoco.mj_resetData(model, data)
    model.body_pos[info.root_body_id] = root_translation
    model.body_quat[info.root_body_id] = rpy_degrees_to_quaternion(hand_rpy_deg)
    data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 3] = cube_world
    data.qpos[info.cube_qpos_adr + 3 : info.cube_qpos_adr + 7] = (
        rpy_degrees_to_quaternion([0.0, 0.0, cube_yaw_deg])
    )
    mujoco.mj_forward(model, data)
    actuator_ids = [model.actuator(name).id for name in ACTIVE_ACTUATORS]
    sweep = scan_distal_closure(
        model,
        data,
        cube_geom_id=info.cube_geom_id,
        finger_order=ACTIVE_FINGERS,
        distal_geom_ids=distal_geom_ids,
        target_faces=[face_from_label(labels[finger]) for finger in ACTIVE_FINGERS],
        actuator_qpos_addresses=[
            int(info.actuator_qpos_adrs[actuator_id])
            for actuator_id in actuator_ids
        ],
        open_targets_rad=np.zeros(len(ACTIVE_ACTUATORS), dtype=np.float64),
        closed_targets_rad=np.asarray(
            [grasp_targets_rad[name] for name in ACTIVE_ACTUATORS],
            dtype=np.float64,
        ),
        alphas=campaign.closure_alpha_values,
        active_nondistal_geom_ids=active_nondistal_geom_ids,
        forbidden_geom_ids=forbidden_geom_ids,
        target_gap_m=campaign.target_signed_gap_m,
        distal_preload_cap_m=campaign.max_static_distal_preload_m,
        thresholds=thresholds,
    )
    eligible = list(sweep.eligible_samples)
    selected_sample = (
        min(
            eligible,
            key=lambda sample: (
                sample.target_height_spread_m,
                float(np.sum(np.abs(sample.target_signed_gap_m))),
                sample.alpha,
            ),
        )
        if eligible
        else None
    )
    selected_witnesses = (
        selected_sample.target_witnesses
        if selected_sample is not None
        else (None,) * len(ACTIVE_FINGERS)
    )
    saved_qpos = data.qpos.copy()
    saved_ctrl = data.ctrl.copy()
    try:
        if selected_sample is not None:
            for name, actuator_id in zip(ACTIVE_ACTUATORS, actuator_ids):
                qpos_adr = int(info.actuator_qpos_adrs[actuator_id])
                target = selected_sample.alpha * float(grasp_targets_rad[name])
                data.qpos[qpos_adr] = target
                data.ctrl[actuator_id] = target
            mujoco.mj_forward(model, data)
        taxel_assignments = [
            (
                None
                if witness is None
                else nearest_taxel_assignment(
                    witness.distal_point_world_m,
                    data.site_xpos[distal_site_ids[finger]],
                    max_assignment_distance_m=(
                        preferences.taxel_assignment_max_distance_m
                    ),
                )
            )
            for finger, witness in zip(ACTIVE_FINGERS, selected_witnesses)
        ]
    finally:
        data.qpos[:] = saved_qpos
        data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(model, data)
    full_target_preload = float(sweep.samples[-1].max_distal_preload_m)
    forbidden_penetration = any(
        not sample.active_nondistal_clear or not sample.forbidden_clear
        for sample in sweep.samples
    )
    diagnostic = {
        "closure_alpha_values": list(campaign.closure_alpha_values),
        "selected_closure_alpha": (
            None if selected_sample is None else selected_sample.alpha
        ),
        "selected_signed_gap_m": (
            [math.inf] * 3
            if selected_sample is None
            else selected_sample.target_signed_gap_m.tolist()
        ),
        "target_witness_world_m": [
            None if witness is None else witness.cube_point_world_m.tolist()
            for witness in selected_witnesses
        ],
        "contact_height_spread_estimate_m": (
            math.inf
            if selected_sample is None
            else selected_sample.target_height_spread_m
        ),
        "contact_height_aligned_estimate": bool(
            selected_sample is not None
            and selected_sample.target_height_spread_m
            <= definition.contact_alignment.max_height_spread_m + 1e-12
        ),
        "nearest_taxel_distance_m": [
            math.inf if item is None else item.distance_m
            for item in taxel_assignments
        ],
        "pad_assigned_count": (
            sum(item is not None and item.assigned for item in taxel_assignments)
        ),
        "minimum_edge_clearance_m": (
            -math.inf
            if selected_sample is None
            else min(
                witness.classification.edge_clearance_m
                for witness in selected_witnesses
                if witness is not None
            )
        ),
        "minimum_normal_alignment": (
            -math.inf
            if selected_sample is None
            else min(
                witness.classification.normal_alignment
                for witness in selected_witnesses
                if witness is not None
            )
        ),
        "forbidden_penetration": bool(forbidden_penetration),
        "max_active_nondistal_penetration_m": max(
            sample.max_active_nondistal_penetration_m for sample in sweep.samples
        ),
        "max_forbidden_penetration_m": max(
            sample.max_forbidden_penetration_m for sample in sweep.samples
        ),
        "full_target_distal_preload_m": float(full_target_preload),
        "closure_frames": [
            {
                "closure_alpha": sample.alpha,
                "signed_gap_m": sample.target_signed_gap_m.tolist(),
                "height_spread_m": sample.target_height_spread_m,
                "all_targets_reached": sample.all_targets_reached,
                "max_distal_preload_m": sample.max_distal_preload_m,
                "max_active_nondistal_penetration_m": (
                    sample.max_active_nondistal_penetration_m
                ),
                "max_forbidden_penetration_m": (
                    sample.max_forbidden_penetration_m
                ),
            }
            for sample in sweep.samples
        ],
    }
    diagnostic["eligible_for_dynamic"] = far_hand_static_candidate_advances(
        diagnostic,
        target_signed_gap_m=campaign.target_signed_gap_m,
        max_static_distal_preload_m=campaign.max_static_distal_preload_m,
    )
    return diagnostic


def _static_score(
    diagnostic: Mapping[str, Any], candidate_id: int
) -> tuple[float, ...]:
    gaps = diagnostic.get("selected_signed_gap_m", (math.inf,) * 3)
    try:
        gap_error = sum(abs(float(value)) for value in gaps)
    except (TypeError, ValueError):
        gap_error = math.inf
    return (
        float(bool(diagnostic.get("eligible_for_dynamic", False))),
        float(bool(diagnostic.get("contact_height_aligned_estimate", False))),
        float(diagnostic.get("pad_assigned_count", 0)),
        -float(diagnostic.get("contact_height_spread_estimate_m", math.inf)),
        float(diagnostic.get("minimum_edge_clearance_m", -math.inf)),
        float(diagnostic.get("minimum_normal_alignment", -math.inf)),
        -gap_error,
        -float(diagnostic.get("full_target_distal_preload_m", math.inf)),
        -float(candidate_id),
    )


def _sample_targets(
    row: np.ndarray,
    *,
    row_index: int,
    sample_count: int,
    definition: ExperimentDefinition,
) -> dict[str, float]:
    bounds = definition.search_bounds.actuator_targets_rad
    targets: dict[str, float] = {}
    high_count = round(
        definition.far_hand_campaign.concentrated_thumb_bend_fraction
        * sample_count
    )
    for index, name in enumerate(ACTIVE_ACTUATORS):
        selected_bounds = bounds[name]
        if name == THUMB_BEND_ACTUATOR and row_index < high_count:
            selected_bounds = (
                definition.far_hand_campaign.concentrated_thumb_bend_rad
            )
        targets[name] = _scale(float(row[index]), selected_bounds)
    return targets


def _screen_job(
    payload: tuple[dict[str, Any], FarHandStaticJob, int]
) -> tuple[FarHandStaticJob, tuple[dict[str, Any], ...], dict[str, Any] | None, dict[str, Any]]:
    base, job, retain = payload
    definition = resolve_experiment(base)
    campaign = definition.far_hand_campaign
    constraints = definition.far_hand_pose_constraints
    if campaign is None or constraints is None:
        raise ValueError("far-hand static screen requires the v5 definition")
    model, info = build_model(base)
    data = mujoco.MjData(model)
    distal, active_nondistal, forbidden = _collision_geom_sets(model, info)
    sites = _distal_site_ids(model, info)
    bounds = definition.search_bounds
    # roll, yaw, resolved tilt, cube-in-root XYZ, cube yaw and eight targets.
    # The resolved tilt is sampled inside the band cell instead of being fixed
    # to its label.  This preserves a feasible 20-degree band under non-zero
    # roll while metadata continues to use the canonical band centre.
    matrix = _latin_hypercube(
        job.samples, 7 + len(ACTIVE_ACTUATORS), np.random.default_rng(job.seed)
    )
    tilt_bounds = _tilt_band_bounds_deg(
        job.tilt_band_center_deg,
        bands=campaign.tilt_band_centers_deg,
        limits=constraints.finger_down_tilt_deg,
    )
    heap: list[tuple[Any, ...]] = []
    best: tuple[Any, ...] | None = None
    valid_pose_count = 0
    eligible_count = 0
    aligned_count = 0
    near_three_count = 0
    forbidden_count = 0
    preload_exceeded_count = 0
    for row_index, row in enumerate(matrix):
        candidate_id = job.first_candidate_id + row_index
        roll = _scale(row[0], bounds.hand_roll_deg)
        yaw = _scale(row[1], bounds.hand_yaw_deg)
        resolved_tilt = _scale(row[2], tilt_bounds)
        relative = np.asarray(
            [
                _scale(row[3], constraints.cube_position_in_root_m["x"]),
                _scale(row[4], constraints.cube_position_in_root_m["y"]),
                _scale(row[5], constraints.cube_position_in_root_m["z"]),
            ],
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(relative))
        if not (
            constraints.root_cube_distance_m[0] - 1e-12
            <= distance
            <= constraints.root_cube_distance_m[1] + 1e-12
        ):
            continue
        cube_yaw = _scale(row[6], bounds.cube_yaw_deg)
        targets = _sample_targets(
            row[7:],
            row_index=row_index,
            sample_count=job.samples,
            definition=definition,
        )
        rpy, _ = _resolved_root_pose(
            config=base,
            roll_deg=roll,
            tilt_deg=resolved_tilt,
            yaw_deg=yaw,
            cube_in_root_m=relative,
        )
        palm_angle = palm_plane_ground_angle_deg(
            rpy_degrees_to_rotation_matrix(rpy), GRAVITY_WORLD_M_S2
        )
        if not (
            constraints.palm_plane_ground_angle_deg[0] - 1e-12
            <= palm_angle
            <= constraints.palm_plane_ground_angle_deg[1] + 1e-12
        ):
            continue
        valid_pose_count += 1
        diagnostic = closure_sweep_diagnostic(
            model,
            data,
            info,
            base,
            hand_rpy_deg=rpy,
            cube_in_root_m=relative,
            cube_yaw_deg=cube_yaw,
            grasp_targets_rad=targets,
            target_assignment=job.target_assignment,
            distal_geom_ids=distal,
            active_nondistal_geom_ids=active_nondistal,
            forbidden_geom_ids=forbidden,
            distal_site_ids=sites,
        )
        score = _static_score(diagnostic, candidate_id)
        diagnostic.update(
            {
                "static_candidate_id": candidate_id,
                "static_job_id": job.job_id,
                "tilt_band_center_deg": job.tilt_band_center_deg,
                "resolved_finger_down_tilt_deg": resolved_tilt,
                "topology_kind": job.topology_kind,
                "target_faces": job.target_assignment.as_dict(),
                "root_cube_distance_m": distance,
                "thumb_bend_target_rad": targets[THUMB_BEND_ACTUATOR],
                "static_rank": score,
            }
        )
        near_three = diagnostic.get("selected_closure_alpha") is not None
        eligible = bool(diagnostic["eligible_for_dynamic"])
        aligned = bool(diagnostic["contact_height_aligned_estimate"])
        forbidden_hit = bool(diagnostic["forbidden_penetration"])
        preload_exceeded = bool(
            float(diagnostic["full_target_distal_preload_m"])
            > campaign.max_static_distal_preload_m + 1e-12
        )
        eligible_count += int(eligible)
        aligned_count += int(aligned)
        near_three_count += int(near_three)
        forbidden_count += int(forbidden_hit)
        preload_exceeded_count += int(preload_exceeded)
        parameters = {
            "tilt_band_center_deg": job.tilt_band_center_deg,
            "finger_down_tilt_deg": resolved_tilt,
            "roll_deg": roll,
            "yaw_deg": yaw,
            "cube_in_root_m": relative.tolist(),
            "cube_yaw_deg": cube_yaw,
            "grasp_targets_rad": targets,
            "target_assignment": job.target_assignment,
            "static_candidate_id": candidate_id,
        }
        item = (score, -candidate_id, parameters, diagnostic)
        if best is None or item[:2] > best[:2]:
            best = item
        if not eligible:
            continue
        if len(heap) < retain:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)

    retained_items = sorted(heap, key=lambda item: item[:2], reverse=True)
    retained = tuple(
        materialize_far_hand_candidate(
            base,
            **parameters,
            static_diagnostic=diagnostic,
        )
        for _, _, parameters, diagnostic in retained_items
    )
    near_miss = None
    if best is not None:
        _, _, parameters, diagnostic = best
        near_miss = {
            "config": materialize_far_hand_candidate(
                base,
                **parameters,
                static_diagnostic=diagnostic,
            ),
            "diagnostic": copy.deepcopy(diagnostic),
        }
    record = {
        "job_id": job.job_id,
        "tilt_band_center_deg": job.tilt_band_center_deg,
        "topology_kind": job.topology_kind,
        "target_faces": job.target_assignment.as_dict(),
        "resolved_tilt_bounds_deg": list(tilt_bounds),
        "sample_count": job.samples,
        "valid_pose_count": valid_pose_count,
        "eligible_count": eligible_count,
        "aligned_count": aligned_count,
        "near_three_count": near_three_count,
        "forbidden_count": forbidden_count,
        "preload_exceeded_count": preload_exceeded_count,
        "retained_count": len(retained),
        "stop_reason": (
            "eligible_candidates_retained"
            if retained
            else "no_static_candidate_passed_real_geom_closure_gate"
        ),
    }
    return job, retained, near_miss, record


def _evidence_anchor_configs(
    base: Mapping[str, Any], definition: ExperimentDefinition
) -> tuple[dict[str, Any], ...]:
    """Materialize the transformed v3 and aligned-static evidence anchors."""

    source_targets = {
        name: float(base["control"]["grasp_targets_rad"][name])
        for name in ACTIVE_ACTUATORS
    }
    source_manipulation = {
        name: float(base["control"]["manipulation_delta_rad"][name])
        for name in ACTIVE_ACTUATORS
    }
    optimized_targets = dict(
        zip(
            ACTIVE_ACTUATORS,
            (
                0.9562314579814923,
                0.3762493202900651,
                1.0070913503882977,
                -0.024299181483503213,
                0.6940773793455135,
                1.35,
                0.8789227411679765,
                1.1,
            ),
        )
    )
    optimized_manipulation = dict(
        zip(
            ACTIVE_ACTUATORS,
            (
                -0.2,
                0.08631773474672418,
                0.3762606369265243,
                -0.01652957378127996,
                -0.1141601500613772,
                0.08411430586565397,
                0.07055759969302872,
                0.38491932948956076,
            ),
        )
    )
    anchors = (
        {
            "anchor_name": "transformed_v3_dynamic_contact",
            "candidate_id": 0,
            "band": 15.0,
            "roll": 0.0,
            "yaw": -2.6325407810459214,
            "relative": [0.084, -0.02886063167725714, 0.10872576454433164],
            "cube_yaw": 30.589880839844653,
            "targets": source_targets,
            "manipulation": source_manipulation,
        },
        {
            "anchor_name": "static_aligned_geometry_seed",
            "candidate_id": 1,
            "band": 17.5,
            "roll": 0.0,
            "yaw": -2.5,
            "relative": [0.09094, -0.02817, 0.10436],
            "cube_yaw": 25.92,
            "targets": dict(
                zip(
                    ACTIVE_ACTUATORS,
                    (0.998, 0.260, 0.932, -0.034, 0.747, 1.139, 0.959, 1.029),
                )
            ),
            "manipulation": source_manipulation,
        },
        {
            "anchor_name": "independently_reproduced_full_success_seed",
            "candidate_id": 2,
            "band": 20.0,
            "resolved_tilt": 19.9872968445099,
            "roll": -2.0,
            "yaw": -1.7892512430329526,
            "relative": [
                0.08976189614101181,
                -0.03028179757966007,
                0.11088123020658476,
            ],
            "cube_yaw": 29.42029479457431,
            "targets": optimized_targets,
            "manipulation": optimized_manipulation,
        },
        {
            "anchor_name": (
                "independently_reproduced_interior_best_r24_full_success_seed"
            ),
            "candidate_id": 3,
            "band": 17.5,
            "resolved_tilt": 18.17800289033549,
            "roll": -0.7954427593631117,
            "yaw": -1.4451798949000747,
            "relative": [
                0.08976189614101182,
                -0.027792003386342752,
                0.11023613143893961,
            ],
            "cube_yaw": 27.974813355652397,
            "targets": optimized_targets,
            "manipulation": optimized_manipulation,
        },
        {
            "anchor_name": (
                "independently_reproduced_interior_grid_full_success_seed"
            ),
            "candidate_id": 4,
            "band": 17.5,
            "resolved_tilt": 18.49708020401807,
            "roll": -1.0,
            "yaw": -1.7892512430329526,
            "relative": [
                0.08976189614101183,
                -0.03128179757966006,
                0.11188123020658478,
            ],
            "cube_yaw": 29.42029479457431,
            "targets": optimized_targets,
            "manipulation": optimized_manipulation,
        },
    )
    result: list[dict[str, Any]] = []
    for anchor in anchors:
        clipped_targets = {
            name: float(
                np.clip(
                    anchor["targets"][name],
                    definition.search_bounds.actuator_targets_rad[name][0],
                    definition.search_bounds.actuator_targets_rad[name][1],
                )
            )
            for name in ACTIVE_ACTUATORS
        }
        candidate = materialize_far_hand_candidate(
            base,
            tilt_band_center_deg=anchor["band"],
            finger_down_tilt_deg=anchor.get(
                "resolved_tilt", anchor["band"]
            ),
            roll_deg=anchor["roll"],
            yaw_deg=anchor["yaw"],
            cube_in_root_m=anchor["relative"],
            cube_yaw_deg=anchor["cube_yaw"],
            grasp_targets_rad=clipped_targets,
            manipulation_delta_rad=anchor["manipulation"],
            target_assignment=definition.far_hand_campaign.primary_face,
            static_candidate_id=anchor["candidate_id"],
        )
        candidate["candidate_metadata"]["evidence_anchor"] = anchor[
            "anchor_name"
        ]
        candidate["candidate_metadata"]["evidence_anchor_force_dynamic"] = True
        if anchor["anchor_name"].startswith("independently_reproduced_"):
            candidate["candidate_metadata"][
                "independently_reproduced_full_success_seed"
            ] = True
        result.append(candidate)
    return tuple(result)


def _diagnostic_seed_config(
    base: Mapping[str, Any],
    *,
    band: float,
    definition: ExperimentDefinition,
) -> dict[str, Any]:
    """Materialize a valid per-band fallback for a real dynamics rerun."""

    resolved = resolved_pose_constraint_values(dict(base))
    base_rpy = base["hand_pose"]["rpy_deg"]
    candidate = materialize_far_hand_candidate(
        base,
        tilt_band_center_deg=float(band),
        finger_down_tilt_deg=float(band),
        roll_deg=0.0,
        yaw_deg=float(
            np.clip(
                float(base_rpy[2]),
                *definition.search_bounds.hand_yaw_deg,
            )
        ),
        cube_in_root_m=resolved["cube_position_in_root_m"],
        cube_yaw_deg=float(
            np.clip(
                float(base["cube"]["rpy_deg"][2]),
                *definition.search_bounds.cube_yaw_deg,
            )
        ),
        grasp_targets_rad=base["control"]["grasp_targets_rad"],
        target_assignment=definition.far_hand_campaign.primary_face,
    )
    candidate["candidate_metadata"]["diagnostic_seed"] = (
        "canonical_v5_template_projected_to_tilt_band"
    )
    return candidate


def _mark_as_anchor_descendant(candidate: dict[str, Any]) -> None:
    """Remove proof identity while retaining explicit deterministic lineage."""

    metadata = dict(candidate.get("candidate_metadata", {}))
    anchor_name = metadata.pop("evidence_anchor", None)
    for key in (
        "evidence_anchor_force_dynamic",
        "static_gate_passed",
        "independently_reproduced_full_success_seed",
        "static_candidate_id",
        "static_closure_diagnostic",
    ):
        metadata.pop(key, None)
    if anchor_name is not None:
        metadata["parent_evidence_anchor"] = str(anchor_name)
    candidate["candidate_metadata"] = metadata


def run_parallel_far_hand_static_screen(
    base: dict[str, Any],
    *,
    jobs: Sequence[FarHandStaticJob] | None = None,
    retain_per_band: int | None = None,
    workers: int = 1,
    definition: ExperimentDefinition | None = None,
) -> FarHandStaticOutcome:
    """Run collision-geom closure jobs and merge them per tilt band."""

    validate_config(base)
    definition = resolve_experiment(base) if definition is None else definition
    campaign = definition.far_hand_campaign
    if campaign is None or definition.experiment_id != EXPERIMENT_ID:
        raise ValueError("static screen requires the schema-v5 experiment")
    worker_count = _positive_int(workers, "workers")
    retain = _positive_int(
        campaign.static_retain_per_band
        if retain_per_band is None
        else retain_per_band,
        "retain_per_band",
    )
    effective_jobs = tuple(far_hand_static_jobs(campaign=campaign) if jobs is None else jobs)
    payloads = [(copy.deepcopy(base), job, retain) for job in effective_jobs]
    if worker_count == 1:
        outcomes = [_screen_job(payload) for payload in payloads]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context
        ) as executor:
            outcomes = list(executor.map(_screen_job, payloads, chunksize=1))
    outcomes.sort(key=lambda item: item[0].job_id)
    bands = tuple(float(value) for value in campaign.tilt_band_centers_deg)
    grouped: dict[float, list[dict[str, Any]]] = {band: [] for band in bands}
    diagnostics: dict[float, list[dict[str, Any]]] = {band: [] for band in bands}
    near_misses: dict[float, list[dict[str, Any]]] = {band: [] for band in bands}
    records: list[dict[str, Any]] = []
    for job, retained, near_miss, record in outcomes:
        grouped[job.tilt_band_center_deg].extend(retained)
        diagnostics[job.tilt_band_center_deg].extend(
            candidate["candidate_metadata"]["static_closure_diagnostic"]
            for candidate in retained
        )
        if near_miss is not None:
            near_misses[job.tilt_band_center_deg].append(near_miss)
        records.append(record)

    # Anchors are always evaluated in the real-geometry screen; they are not
    # counted as random samples and receive their reserved deterministic IDs.
    anchor_model, anchor_info = build_model(base)
    anchor_data = mujoco.MjData(anchor_model)
    distal, nondistal, forbidden = _collision_geom_sets(anchor_model, anchor_info)
    sites = _distal_site_ids(anchor_model, anchor_info)
    for anchor in _evidence_anchor_configs(base, definition):
        metadata = anchor["candidate_metadata"]
        resolved = resolved_pose_constraint_values(anchor)
        assignment = OpposedFaceAssignment.from_mapping(
            anchor["contact_topology"]["target_faces"]
        )
        diagnostic = closure_sweep_diagnostic(
            anchor_model,
            anchor_data,
            anchor_info,
            anchor,
            hand_rpy_deg=anchor["hand_pose"]["rpy_deg"],
            cube_in_root_m=resolved["cube_position_in_root_m"],
            cube_yaw_deg=anchor["cube"]["rpy_deg"][2],
            grasp_targets_rad=anchor["control"]["grasp_targets_rad"],
            target_assignment=assignment,
            distal_geom_ids=distal,
            active_nondistal_geom_ids=nondistal,
            forbidden_geom_ids=forbidden,
            distal_site_ids=sites,
        )
        diagnostic.update(
            {
                "static_candidate_id": metadata["static_candidate_id"],
                "tilt_band_center_deg": metadata["tilt_band_center_deg"],
                "topology_kind": "primary_anchor",
                "target_faces": assignment.as_dict(),
                "root_cube_distance_m": resolved["root_cube_distance_m"],
                "thumb_bend_target_rad": anchor["control"][
                    "grasp_targets_rad"
                ][THUMB_BEND_ACTUATOR],
            }
        )
        diagnostic["static_rank"] = _static_score(
            diagnostic, int(metadata["static_candidate_id"])
        )
        anchor["candidate_metadata"]["static_closure_diagnostic"] = copy.deepcopy(
            diagnostic
        )
        anchor["candidate_metadata"]["evidence_anchor_force_dynamic"] = True
        anchor["candidate_metadata"]["static_gate_passed"] = bool(
            diagnostic["eligible_for_dynamic"]
        )
        band = float(metadata["tilt_band_center_deg"])
        diagnostics[band].append(diagnostic)
        near_misses[band].append(
            {"config": copy.deepcopy(anchor), "diagnostic": diagnostic}
        )
        # The two declared evidence anchors always receive a real dynamics
        # evaluation.  A failed static gate remains explicitly recorded and
        # never counts as success; this exception only prevents the strongest
        # known clean three-finger region from being discarded before the
        # compliant simulation can evaluate it.
        grouped[band].append(anchor)

    retained_by_band: dict[float, tuple[dict[str, Any], ...]] = {}
    best_by_band: dict[float, dict[str, Any] | None] = {}
    for band in bands:
        anchors = sorted(
            (
                item
                for item in grouped[band]
                if item.get("candidate_metadata", {}).get(
                    "evidence_anchor_force_dynamic"
                )
                is True
            ),
            key=lambda item: int(
                item["candidate_metadata"]["static_candidate_id"]
            ),
        )
        ordinary = sorted(
            (
                item
                for item in grouped[band]
                if item.get("candidate_metadata", {}).get(
                    "evidence_anchor_force_dynamic"
                )
                is not True
            ),
            key=lambda item: tuple(
                item["candidate_metadata"]["static_closure_diagnostic"][
                    "static_rank"
                ]
            ),
            reverse=True,
        )
        # Evidence anchors are a declared addition to the random reservoir and
        # must not be dropped by a deliberately tiny diagnostic override.
        candidates = anchors + ordinary[: max(0, retain - len(anchors))]
        retained_by_band[band] = tuple(candidates)
        attempts = near_misses[band]
        best_by_band[band] = (
            copy.deepcopy(
                max(
                    attempts,
                    key=lambda item: tuple(item["diagnostic"]["static_rank"]),
                )
            )
            if attempts
            else None
        )
    return FarHandStaticOutcome(
        sample_count=sum(job.samples for job in effective_jobs),
        retained_by_band=retained_by_band,
        diagnostics_by_band={
            band: tuple(copy.deepcopy(diagnostics[band])) for band in bands
        },
        near_miss_by_band=best_by_band,
        job_records=tuple(records),
    )


def _summary(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = result.get("summary", {})
    return value if isinstance(value, Mapping) else {}


def _metrics(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _summary(result).get("metrics", {})
    return value if isinstance(value, Mapping) else {}


def _metric(
    result: Mapping[str, Any], names: Sequence[str], *, default: float
) -> float:
    containers: list[Mapping[str, Any]] = [
        _metrics(result),
        _summary(result),
        result,
    ]
    for container in containers:
        for name in names:
            value = container.get(name)
            if isinstance(value, Mapping):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return default


def _minimum_margin(result: Mapping[str, Any]) -> float:
    metrics = _metrics(result)
    direct = _metric(
        result,
        ("minimum_normalized_margin", "minimum_acceptance_margin"),
        default=-math.inf,
    )
    if math.isfinite(direct):
        return direct
    margins = metrics.get("normalized_acceptance_margins")
    if isinstance(margins, Mapping) and margins:
        try:
            values = [float(value) for value in margins.values()]
        except (TypeError, ValueError):
            return -math.inf
        if values and all(math.isfinite(value) for value in values):
            return min(values)
    config = result.get("config")
    if isinstance(config, Mapping):
        try:
            # Local import avoids a module cycle: search dispatches to this
            # tuner, while this fallback reuses its canonical margin formula.
            from ..search import normalized_acceptance_margins

            kwargs: dict[str, Any] = {
                "contact_topology": config.get("contact_topology"),
                "contact_alignment": config.get("contact_alignment"),
                "pose_constraints": config.get("pose_constraints"),
            }
            computed = normalized_acceptance_margins(
                dict(metrics), dict(config["acceptance"]), **kwargs
            )
            values = [float(value) for value in computed.values()]
            if values and all(math.isfinite(value) for value in values):
                return min(values)
        except (KeyError, TypeError, ValueError):
            pass
    return -math.inf


def _grasp_margin(result: Mapping[str, Any]) -> float:
    direct = _metric(
        result,
        (
            "grasp_stability_margin",
            "minimum_grasp_gate_margin",
            "grasp_minimum_normalized_margin",
        ),
        default=-math.inf,
    )
    if math.isfinite(direct):
        return direct
    metrics = _metrics(result)
    margins = metrics.get("grasp_gate_margins")
    if isinstance(margins, Mapping) and margins:
        try:
            values = [float(value) for value in margins.values()]
        except (TypeError, ValueError):
            values = []
        if values and all(math.isfinite(value) for value in values):
            return min(values)
    reached = _metric(
        result,
        (
            "verify_max_consecutive_all_gate_steps",
            "verify_max_consecutive_gate_steps",
            "grasp_gate_final_consecutive_steps",
        ),
        default=-math.inf,
    )
    required = _metric(
        result, ("grasp_stable_window_steps",), default=-math.inf
    )
    if math.isfinite(reached) and math.isfinite(required) and required > 0.0:
        return (reached - required) / required
    return -math.inf


def _pad_rank_metrics(result: Mapping[str, Any]) -> tuple[float, float]:
    metrics = _metrics(result)
    fingertip = metrics.get("fingertip_contact")
    if isinstance(fingertip, Mapping):
        operation = fingertip.get("operation", {})
        verify = fingertip.get("verify", {})
        selected = (
            operation
            if isinstance(operation, Mapping)
            and int(operation.get("sample_count", 0)) > 0
            else verify
        )
        if isinstance(selected, Mapping):
            nested_fraction = selected.get("force_weighted_pad_fraction")
            nested_taxels = selected.get("max_active_taxel_count")
            if isinstance(nested_fraction, Mapping) and isinstance(
                nested_taxels, Mapping
            ):
                return (
                    min(
                        float(nested_fraction.get(finger, 0.0))
                        for finger in ACTIVE_FINGERS
                    ),
                    sum(
                        float(nested_taxels.get(finger, 0.0))
                        for finger in ACTIVE_FINGERS
                    ),
                )
    fraction = metrics.get("distal_pad_force_fraction")
    if isinstance(fraction, Mapping):
        values = [
            float(fraction.get(finger, 0.0)) for finger in ACTIVE_FINGERS
        ]
        pad_fraction = min(values)
    elif isinstance(fraction, Sequence) and not isinstance(fraction, (str, bytes)):
        values = [float(value) for value in fraction]
        pad_fraction = min(values) if values else -math.inf
    else:
        pad_fraction = _metric(
            result,
            ("minimum_distal_pad_force_fraction", "pad_force_fraction"),
            default=-math.inf,
        )
    coverage = metrics.get("distal_pad_active_taxel_count")
    if isinstance(coverage, Mapping):
        taxel_coverage = sum(float(coverage.get(finger, 0.0)) for finger in ACTIVE_FINGERS)
    elif isinstance(coverage, Sequence) and not isinstance(coverage, (str, bytes)):
        taxel_coverage = sum(float(value) for value in coverage)
    else:
        taxel_coverage = _metric(
            result,
            ("active_pad_taxel_count", "pad_taxel_coverage"),
            default=-math.inf,
        )
    return float(pad_fraction), float(taxel_coverage)


def _finite_rank_number(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _balanced_verify_metric(
    metrics: Mapping[str, Any], name: str
) -> tuple[float, ...]:
    values = metrics.get(name, {})
    if not isinstance(values, Mapping):
        values = {}
    return tuple(
        sorted(
            _finite_rank_number(values.get(finger, 0.0))
            for finger in ACTIVE_FINGERS
        )
    )


def _verify_near_miss_rank(result: Mapping[str, Any]) -> tuple[float, ...]:
    """Rank raw three-finger VERIFY evidence weakest-finger first.

    This key is intentionally independent of the soft fingertip-pad score.
    A one-finger proximal press with a high pad ratio must never outrank a
    candidate that is close to acquiring the complete three-finger gate.
    """

    metrics = _metrics(result)
    component_duty = metrics.get("verify_gate_component_duty", {})
    if isinstance(component_duty, Mapping) and component_duty:
        components = tuple(
            _finite_rank_number(value) for value in component_duty.values()
        )
        component_minimum = min(components)
        component_mean = sum(components) / len(components)
    else:
        component_minimum = 0.0
        component_mean = 0.0
    return (
        _finite_rank_number(metrics.get("verify_effective_finger_count", 0)),
        _finite_rank_number(
            metrics.get("verify_max_simultaneous_effective_finger_count", 0)
        ),
        *_balanced_verify_metric(metrics, "verify_target_face_effective_duty"),
        _finite_rank_number(
            metrics.get("verify_target_face_simultaneous_duty", 0.0)
        ),
        *_balanced_verify_metric(metrics, "verify_peak_target_face_force_n"),
        *_balanced_verify_metric(metrics, "verify_peak_tactile_n"),
        _finite_rank_number(
            metrics.get(
                "verify_max_consecutive_all_gate_steps",
                metrics.get("verify_max_consecutive_gate_steps", 0),
            )
        ),
        _finite_rank_number(metrics.get("verify_all_gate_duty", 0.0)),
        component_minimum,
        component_mean,
    )


def far_hand_candidate_rank(result: Mapping[str, Any]) -> tuple[float, ...]:
    """Canonical worker-independent v5 dynamic ranking."""

    try:
        candidate_id = int(result["candidate_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("candidate result requires an integer candidate_id") from error
    full = full_succeeded(result)
    grasp = grasp_succeeded(result)
    manipulation = manipulation_succeeded(result)
    grasp_margin = _grasp_margin(result)
    metrics = _metrics(result)
    alignment = metrics.get("contact_alignment", {})
    operation = (
        alignment.get("operation", {}) if isinstance(alignment, Mapping) else {}
    )
    # Empty/failed contact windows are serialized as JSON ``null`` rather
    # than omitting the percentile fields.  They are valid near-miss results
    # and must sort behind measured windows instead of aborting the campaign.
    aligned_duty = (
        _finite_rank_number(
            operation.get("aligned_duty"), default=-math.inf
        )
        if isinstance(operation, Mapping)
        else -math.inf
    )
    p95_spread = (
        _finite_rank_number(
            operation.get("height_spread_p95_m"), default=math.inf
        )
        if isinstance(operation, Mapping)
        else math.inf
    )
    pad_fraction, taxel_coverage = _pad_rank_metrics(result)
    topology = _metric(
        result,
        (
            "operation_target_face_simultaneous_duty",
            "target_face_simultaneous_duty",
        ),
        default=-math.inf,
    )
    force = _metric(
        result,
        ("peak_total_distal_contact_force_n", "peak_contact_force_n"),
        default=math.inf,
    )
    drift = _metric(
        result,
        ("orientation_drift_deg", "operation_orientation_drift_deg"),
        default=math.inf,
    )
    saturation = _metric(
        result,
        ("actuator_saturation_fraction", "saturation_fraction"),
        default=math.inf,
    )
    if not grasp:
        # The explicit stage bits are deliberately first.  VERIFY evidence is
        # then more important than pad preference for a failed grasp, so an
        # apparently good one-finger pad contact cannot win the near-miss rank.
        return (
            float(full),
            float(grasp),
            float(manipulation),
            *_verify_near_miss_rank(result),
            _minimum_margin(result),
            pad_fraction,
            taxel_coverage,
            -force,
            -saturation,
            -float(candidate_id),
        )
    return (
        float(full),
        float(grasp),
        float(manipulation),
        _minimum_margin(result),
        grasp_margin,
        aligned_duty,
        -p95_spread,
        pad_fraction,
        taxel_coverage,
        topology,
        -force,
        -drift,
        -saturation,
        -float(candidate_id),
    )


def deterministic_rank_far_hand_results(
    results: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    materialized = tuple(results)
    ids = tuple(int(result["candidate_id"]) for result in materialized)
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be unique")
    return tuple(
        sorted(materialized, key=far_hand_candidate_rank, reverse=True)
    )


def _physical_candidate_key(result: Mapping[str, Any]) -> str:
    """Canonical key for physics-equivalent pose/control candidates."""

    config = result.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("candidate result is missing config")
    payload = {
        "hand_pose": config.get("hand_pose"),
        "cube": config.get("cube"),
        "control": config.get("control"),
        "target_faces": config.get("contact_topology", {}).get(
            "target_faces"
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _deduplicate_physical_results(
    results: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Keep the best deterministic result for every physical trajectory."""

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in deterministic_rank_far_hand_results(results):
        key = _physical_candidate_key(result)
        if key in seen:
            continue
        seen.add(key)
        unique.append(copy.deepcopy(result))
    return tuple(unique)


def _candidate_band(result: Mapping[str, Any]) -> float:
    if "tilt_band_center_deg" in result:
        return float(result["tilt_band_center_deg"])
    config = result.get("config", result)
    if not isinstance(config, Mapping):
        raise ValueError("candidate is missing config")
    metadata = config.get("candidate_metadata", {})
    if not isinstance(metadata, Mapping) or "tilt_band_center_deg" not in metadata:
        raise ValueError("candidate is missing tilt_band_center_deg")
    return float(metadata["tilt_band_center_deg"])


def _select_per_band(
    results: Iterable[dict[str, Any]],
    *,
    bands: Sequence[float],
    limit: int,
    predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[float, tuple[dict[str, Any], ...]]:
    ranked = deterministic_rank_far_hand_results(results)
    return {
        float(band): tuple(
            copy.deepcopy(result)
            for result in ranked
            if math.isclose(
                _candidate_band(result), float(band), abs_tol=1e-9
            )
            and (predicate is None or predicate(result))
        )[:limit]
        for band in bands
    }


def _tag_config(
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
    material_policy: str = "constant_density_nominal",
) -> tuple[list[dict[str, Any]], int]:
    tagged = [
        _tag_config(config, stage=stage, material_policy=material_policy)
        for config in configs
    ]
    for config in tagged:
        validate_config(config)
    payloads = [
        (next_candidate_id + index, copy.deepcopy(config))
        for index, config in enumerate(tagged)
    ]
    submitted = {candidate_id: config for candidate_id, config in payloads}
    raw = run_candidates(payloads, workers) if payloads else []
    results = [copy.deepcopy(dict(result)) for result in raw]
    expected = set(submitted)
    received = {int(result.get("candidate_id", -1)) for result in results}
    if len(results) != len(payloads) or received != expected:
        raise RuntimeError(f"{stage} runner did not preserve candidate IDs")
    for result in results:
        candidate_id = int(result["candidate_id"])
        if result.get("config") != submitted[candidate_id]:
            raise RuntimeError(f"{stage} runner rebound candidate configuration")
        result["search_stage"] = stage
        result["material_policy"] = material_policy
        result["tilt_band_center_deg"] = _candidate_band(result["config"])
    results.sort(key=lambda result: int(result["candidate_id"]))
    return results, next_candidate_id + len(payloads)


def _clamp_relative_to_nominal_domain(
    relative: np.ndarray, definition: ExperimentDefinition
) -> np.ndarray:
    constraints = definition.far_hand_pose_constraints
    assert constraints is not None
    raw = np.asarray(relative, dtype=np.float64)
    if raw.shape != (3,) or not np.all(np.isfinite(raw)):
        raise ValueError("relative pose must be a finite three-vector")
    axis_lower = np.asarray(
        [
            constraints.cube_position_in_root_m[axis][0]
            for axis in ("x", "y", "z")
        ],
        dtype=np.float64,
    )
    axis_upper = np.asarray(
        [
            constraints.cube_position_in_root_m[axis][1]
            for axis in ("x", "y", "z")
        ],
        dtype=np.float64,
    )
    result = np.clip(raw, axis_lower, axis_upper)
    lower, upper = constraints.root_cube_distance_m
    norm = float(np.linalg.norm(result))
    if lower <= norm <= upper:
        return result

    # Project along the clipped candidate's ray.  In the v5 domain all three
    # axis intervals have fixed signs, so ||clip(scale * result, box)|| is
    # monotone in ``scale``.  Bisection therefore finds the exact intersection
    # with the distance shell without the clip/rescale oscillation that a
    # fixed number of iterations can leave just outside the valid domain.
    def scaled_and_clipped(scale: float) -> np.ndarray:
        return np.clip(result * scale, axis_lower, axis_upper)

    target = lower if norm < lower else upper
    if norm < lower:
        scale_low = 1.0
        scale_high = 2.0
        projected_high = scaled_and_clipped(scale_high)
        while float(np.linalg.norm(projected_high)) < target:
            scale_high *= 2.0
            if scale_high > 2.0**20:
                raise RuntimeError(
                    "v5 pose box does not intersect the lower distance shell"
                )
            projected_high = scaled_and_clipped(scale_high)
    else:
        scale_low = 0.0
        scale_high = 1.0
        if float(np.linalg.norm(scaled_and_clipped(scale_low))) > target:
            raise RuntimeError(
                "v5 pose box does not intersect the upper distance shell"
            )

    for _ in range(64):
        scale_mid = 0.5 * (scale_low + scale_high)
        projected_mid = scaled_and_clipped(scale_mid)
        if float(np.linalg.norm(projected_mid)) < target:
            scale_low = scale_mid
        else:
            scale_high = scale_mid
    # Pick the shell-feasible side of the final bracket.  The 64-step bracket
    # is far tighter than FarHandPoseConstraints' 1e-12 membership tolerance.
    result = scaled_and_clipped(
        scale_high if norm < lower else scale_low
    )
    if not constraints.contains_cube_position(result):
        raise RuntimeError("unable to clamp local refinement into v5 pose domain")
    return result


def _local_grasp_candidates(
    parent: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    definition: ExperimentDefinition,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    parent_config = parent.get("config", parent)
    if not isinstance(parent_config, Mapping):
        raise TypeError("parent must contain a config")
    pose = resolved_pose_constraint_values(dict(parent_config))
    parent_relative = np.asarray(pose["cube_position_in_root_m"], dtype=np.float64)
    parent_rpy = np.asarray(parent_config["hand_pose"]["rpy_deg"], dtype=np.float64)
    parent_tilt = float(pose["finger_down_tilt_deg"])
    band = float(parent_config["candidate_metadata"]["tilt_band_center_deg"])
    bounds = definition.search_bounds
    result: list[dict[str, Any]] = []
    for index in range(count):
        roll = float(np.clip(parent_rpy[0] + rng.normal(0.0, 0.5), *bounds.hand_roll_deg))
        yaw = float(np.clip(parent_rpy[2] + rng.normal(0.0, 0.6), *bounds.hand_yaw_deg))
        constraints = definition.far_hand_pose_constraints
        tilt = _feasible_tilt_for_roll_deg(
            float(
                np.clip(
                    parent_tilt + rng.normal(0.0, 0.5),
                    *constraints.finger_down_tilt_deg,
                )
            ),
            roll_deg=roll,
            constraints=constraints,
        )
        relative = _clamp_relative_to_nominal_domain(
            parent_relative + rng.normal(0.0, [0.0015, 0.0012, 0.0015]),
            definition,
        )
        cube_yaw = float(
            np.clip(
                float(parent_config["cube"]["rpy_deg"][2])
                + rng.normal(0.0, 1.0),
                *bounds.cube_yaw_deg,
            )
        )
        targets: dict[str, float] = {}
        for name in ACTIVE_ACTUATORS:
            lower, upper = bounds.actuator_targets_rad[name]
            span = upper - lower
            targets[name] = float(
                np.clip(
                    float(parent_config["control"]["grasp_targets_rad"][name])
                    + rng.normal(0.0, 0.035 * span),
                    lower,
                    upper,
                )
            )
        candidate = materialize_far_hand_candidate(
            parent_config,
            tilt_band_center_deg=band,
            finger_down_tilt_deg=tilt,
            roll_deg=roll,
            yaw_deg=yaw,
            cube_in_root_m=relative,
            cube_yaw_deg=cube_yaw,
            grasp_targets_rad=targets,
            target_assignment=parent_config["contact_topology"]["target_faces"],
        )
        _mark_as_anchor_descendant(candidate)
        candidate["candidate_metadata"]["local_sample_index"] = index
        result.append(candidate)
    return result


def boundary_expansion_decision(
    config: Mapping[str, Any], definition: ExperimentDefinition | None = None
) -> dict[str, Any]:
    """Return the deterministic one-shot v5 expansion decision."""

    definition = resolve_experiment(config) if definition is None else definition
    campaign = definition.far_hand_campaign
    constraints = definition.far_hand_pose_constraints
    if campaign is None or constraints is None:
        raise ValueError("boundary expansion requires the v5 experiment")
    existing = config.get("candidate_metadata", {}).get("boundary_expansion", {})
    if isinstance(existing, Mapping) and existing.get("applied") is True:
        return {
            "applied": False,
            "count": 1,
            "distance_expanded": False,
            "thumb_bend_expanded": False,
            "reason": "one_shot_expansion_already_consumed",
        }
    resolved = resolved_pose_constraint_values(dict(config))
    distance = float(resolved["root_cube_distance_m"])
    thumb = float(config["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR])
    policy = campaign.boundary_expansion
    distance_hit = distance >= (
        constraints.root_cube_distance_m[1] - policy.distance_tolerance_m - 1e-12
    )
    thumb_hit = thumb >= (
        definition.search_bounds.actuator_targets_rad[THUMB_BEND_ACTUATOR][1]
        - policy.thumb_bend_tolerance_rad
        - 1e-12
    )
    return {
        "applied": bool(distance_hit or thumb_hit),
        "count": int(distance_hit or thumb_hit),
        "distance_expanded": bool(distance_hit),
        "thumb_bend_expanded": bool(thumb_hit),
        "reason": (
            "nominal_upper_boundary_hit"
            if distance_hit or thumb_hit
            else "no_nominal_upper_boundary_hit"
        ),
    }


def _expanded_grasp_candidates(
    parent: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    definition: ExperimentDefinition,
    decision: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if decision.get("applied") is not True or decision.get("count") != 1:
        return []
    rng = np.random.default_rng(seed)
    parent_config = parent.get("config", parent)
    pose = resolved_pose_constraint_values(dict(parent_config))
    parent_relative = np.asarray(pose["cube_position_in_root_m"], dtype=np.float64)
    parent_rpy = parent_config["hand_pose"]["rpy_deg"]
    band = float(parent_config["candidate_metadata"]["tilt_band_center_deg"])
    policy = definition.far_hand_campaign.boundary_expansion
    constraints = definition.far_hand_pose_constraints
    bounds = definition.search_bounds
    configs: list[dict[str, Any]] = []
    for index in range(count):
        relative = parent_relative + rng.normal(0.0, [0.0012, 0.0008, 0.0008])
        x_upper = (
            policy.expanded_cube_in_root_x_max_m
            if decision.get("distance_expanded")
            else constraints.cube_position_in_root_m["x"][1]
        )
        relative[0] = np.clip(
            relative[0], constraints.cube_position_in_root_m["x"][0], x_upper
        )
        relative[1] = np.clip(
            relative[1], *constraints.cube_position_in_root_m["y"]
        )
        relative[2] = np.clip(
            relative[2], *constraints.cube_position_in_root_m["z"]
        )
        maximum_distance = (
            policy.expanded_root_cube_distance_max_m
            if decision.get("distance_expanded")
            else constraints.root_cube_distance_m[1]
        )
        norm = float(np.linalg.norm(relative))
        if norm > maximum_distance:
            relative *= maximum_distance / norm
        minimum_distance = constraints.root_cube_distance_m[0]
        norm = float(np.linalg.norm(relative))
        if norm < minimum_distance:
            relative *= minimum_distance / norm

        targets = {
            name: float(parent_config["control"]["grasp_targets_rad"][name])
            for name in ACTIVE_ACTUATORS
        }
        for name in ACTIVE_ACTUATORS:
            lower, upper = bounds.actuator_targets_rad[name]
            if name == THUMB_BEND_ACTUATOR and decision.get("thumb_bend_expanded"):
                upper += policy.thumb_bend_expand_rad
            targets[name] = float(
                np.clip(
                    targets[name] + rng.normal(0.0, 0.015 * (upper - lower)),
                    lower,
                    upper,
                )
            )
        roll = float(
            np.clip(
                parent_rpy[0] + rng.normal(0.0, 0.35),
                *bounds.hand_roll_deg,
            )
        )
        tilt = _feasible_tilt_for_roll_deg(
            float(
                np.clip(
                    float(pose["finger_down_tilt_deg"])
                    + rng.normal(0.0, 0.35),
                    *constraints.finger_down_tilt_deg,
                )
            ),
            roll_deg=roll,
            constraints=constraints,
        )
        candidate = materialize_far_hand_candidate(
            parent_config,
            tilt_band_center_deg=band,
            finger_down_tilt_deg=tilt,
            roll_deg=roll,
            yaw_deg=float(
                np.clip(parent_rpy[2] + rng.normal(0.0, 0.4), *bounds.hand_yaw_deg)
            ),
            cube_in_root_m=relative,
            cube_yaw_deg=float(
                np.clip(
                    parent_config["cube"]["rpy_deg"][2] + rng.normal(0.0, 0.7),
                    *bounds.cube_yaw_deg,
                )
            ),
            grasp_targets_rad=targets,
            target_assignment=parent_config["contact_topology"]["target_faces"],
            boundary_expansion=decision,
        )
        _mark_as_anchor_descendant(candidate)
        candidate["candidate_metadata"]["boundary_sample_index"] = index
        configs.append(candidate)
    return configs


def _manipulation_candidates(
    parent: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    definition: ExperimentDefinition,
) -> list[dict[str, Any]]:
    parent_config = parent.get("config", parent)
    if not isinstance(parent_config, Mapping):
        raise TypeError("parent must contain a config")
    absolute_limits = {
        "left_hand_thumb_bend_joint_actuator": (0.75, 1.47),
        "left_hand_thumb_rota_joint1_actuator": (-0.698, 1.57),
        "left_hand_thumb_rota_joint2_actuator": (0.0, 1.57),
        "left_hand_index_bend_joint_actuator": (-0.174, 0.174),
        "left_hand_index_joint1_actuator": (0.0, 1.919),
        "left_hand_index_joint2_actuator": (0.0, 1.919),
        "left_hand_mid_joint1_actuator": (0.0, 1.919),
        "left_hand_mid_joint2_actuator": (0.0, 1.919),
    }
    candidates = sample_manipulation_delta_candidates(
        parent_config,
        count=count,
        seed=seed,
        delta_bounds_rad=definition.search_bounds.manipulation_delta_rad,
        local_radius_rad=0.10,
        absolute_target_bounds_rad=absolute_limits,
        validator=validate_config,
    )
    for candidate in candidates:
        _mark_as_anchor_descendant(candidate)
        validate_config(candidate)
    return candidates


def _set_relative_pose(
    config: dict[str, Any],
    *,
    cube_in_root_m: Sequence[float],
    finger_down_tilt_deg: float,
    hand_roll_deg: float,
    hand_yaw_deg: float,
) -> None:
    rpy, translation = _resolved_root_pose(
        config=config,
        roll_deg=hand_roll_deg,
        tilt_deg=finger_down_tilt_deg,
        yaw_deg=hand_yaw_deg,
        cube_in_root_m=cube_in_root_m,
    )
    config["hand_pose"] = {"translation_m": translation, "rpy_deg": rpy}


def generate_far_hand_size_cases(
    config: Mapping[str, Any],
    *,
    definition: ExperimentDefinition | None = None,
) -> list[dict[str, Any]]:
    """Generate the declared 59--64 mm constant-density post-success sweep."""

    definition = resolve_experiment(config) if definition is None else definition
    campaign = definition.far_hand_campaign
    if campaign is None:
        raise ValueError("size cases require far_hand_campaign")
    pose = resolved_pose_constraint_values(dict(config))
    relative = pose["cube_position_in_root_m"]
    tilt = pose["finger_down_tilt_deg"]
    rpy = config["hand_pose"]["rpy_deg"]
    cases: list[dict[str, Any]] = []
    for edge in campaign.post_success_edges_m:
        case = copy.deepcopy(dict(config))
        case.pop("experiment_status", None)
        case["run_context"] = {"kind": "robustness_trial"}
        case["cube"]["edge_m"] = float(edge)
        case["cube"]["mass_kg"] = campaign.constant_density_mass_kg(edge)
        case["cube"]["friction"] = campaign.friction
        _set_relative_pose(
            case,
            cube_in_root_m=relative,
            finger_down_tilt_deg=tilt,
            hand_roll_deg=float(rpy[0]),
            hand_yaw_deg=float(rpy[2]),
        )
        metadata = dict(case.get("candidate_metadata", {}))
        metadata.update(
            {
                "search_stage": "post_success_size_sweep",
                "material_policy": "constant_density",
                "size_sweep_edge_m": float(edge),
            }
        )
        case["candidate_metadata"] = metadata
        validate_config(case)
        cases.append(case)
    return cases


def generate_far_hand_perturbation_configs(
    config: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    definition: ExperimentDefinition | None = None,
) -> list[dict[str, Any]]:
    """Generate 12-D fixed-seed v5 pose/material perturbations."""

    definition = resolve_experiment(config) if definition is None else definition
    campaign = definition.far_hand_campaign
    if campaign is None:
        raise ValueError("far-hand perturbations require far_hand_campaign")
    sample_count = _positive_int(count, "count")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    envelope = campaign.perturbation_envelope
    matrix = _latin_hypercube(
        sample_count, envelope.dimensions, np.random.default_rng(seed)
    )
    parent_pose = resolved_pose_constraint_values(dict(config))
    parent_relative = np.asarray(
        parent_pose["cube_position_in_root_m"], dtype=np.float64
    )
    parent_distance = float(np.linalg.norm(parent_relative))
    parent_direction = parent_relative / parent_distance
    parent_hand_rpy = np.asarray(config["hand_pose"]["rpy_deg"], dtype=np.float64)
    parent_cube_rpy = np.asarray(config["cube"]["rpy_deg"], dtype=np.float64)
    parent_xy = np.asarray(config["cube"]["center_xy_m"], dtype=np.float64)
    result: list[dict[str, Any]] = []
    for index, row in enumerate(matrix):
        cursor = 0
        xy_delta = np.asarray(
            [
                _scale(row[cursor], envelope.cube_center_xy_delta_m),
                _scale(row[cursor + 1], envelope.cube_center_xy_delta_m),
            ]
        )
        cursor += 2
        gap = _scale(row[cursor], envelope.cube_gap_m)
        cursor += 1
        cube_rpy_delta = np.asarray(
            [
                _scale(row[cursor + axis], envelope.cube_rpy_delta_deg)
                for axis in range(3)
            ]
        )
        cursor += 3
        hand_roll_delta = _scale(row[cursor], envelope.hand_roll_yaw_delta_deg)
        hand_yaw_delta = _scale(row[cursor + 1], envelope.hand_roll_yaw_delta_deg)
        cursor += 2
        tilt_delta = _scale(row[cursor], envelope.finger_down_tilt_delta_deg)
        cursor += 1
        distance_delta = _scale(
            row[cursor], envelope.root_cube_distance_delta_m
        )
        cursor += 1
        mass_scale = _scale(row[cursor], envelope.mass_scale)
        cursor += 1
        friction_delta = _scale(row[cursor], envelope.friction_delta)

        case = copy.deepcopy(dict(config))
        case.pop("experiment_status", None)
        case["run_context"] = {"kind": "robustness_trial"}
        case["cube"]["center_xy_m"] = (parent_xy + xy_delta).tolist()
        case["cube"]["z_offset_m"] = float(gap)
        case["cube"]["rpy_deg"] = (parent_cube_rpy + cube_rpy_delta).tolist()
        case["cube"]["mass_kg"] = (
            campaign.constant_density_mass_kg(float(case["cube"]["edge_m"]))
            * mass_scale
        )
        case["cube"]["friction"] = campaign.friction + friction_delta
        relative = parent_direction * (parent_distance + distance_delta)
        _set_relative_pose(
            case,
            cube_in_root_m=relative,
            finger_down_tilt_deg=parent_pose["finger_down_tilt_deg"] + tilt_delta,
            hand_roll_deg=parent_hand_rpy[0] + hand_roll_delta,
            hand_yaw_deg=parent_hand_rpy[2] + hand_yaw_delta,
        )
        metadata = dict(case.get("candidate_metadata", {}))
        metadata.update(
            {
                "search_stage": "robustness_trial",
                "material_policy": "local_pose_and_material_perturbation",
                "robustness_seed": int(seed),
                "robustness_trial": index,
                "resolved_perturbations": {
                    "cube_center_xy_delta_m": xy_delta.tolist(),
                    "cube_gap_m": gap,
                    "cube_rpy_delta_deg": cube_rpy_delta.tolist(),
                    "hand_roll_deg": hand_roll_delta,
                    "hand_yaw_deg": hand_yaw_delta,
                    "finger_down_tilt_delta_deg": tilt_delta,
                    "root_cube_distance_delta_m": distance_delta,
                    "mass_scale": mass_scale,
                    "friction_delta": friction_delta,
                },
            }
        )
        case["candidate_metadata"] = metadata
        validate_config(case)
        result.append(case)
    return result


def _coerce_static_outcome(
    value: Any, *, bands: Sequence[float]
) -> FarHandStaticOutcome:
    if isinstance(value, FarHandStaticOutcome):
        return value
    if isinstance(value, Mapping):
        retained_source = value.get("retained_by_band", value)
        if not isinstance(retained_source, Mapping):
            raise TypeError("static retained_by_band must be a mapping")
        near_source = value.get("near_miss_by_band", {})
        diagnostics_source = value.get("diagnostics_by_band", {})
        if not isinstance(near_source, Mapping):
            raise TypeError("static near_miss_by_band must be a mapping")
        if not isinstance(diagnostics_source, Mapping):
            raise TypeError("static diagnostics_by_band must be a mapping")
        retained = {
            float(band): tuple(
                copy.deepcopy(dict(candidate))
                for candidate in retained_source.get(
                    band, retained_source.get(str(band), ())
                )
            )
            for band in bands
        }
        return FarHandStaticOutcome(
            sample_count=int(value.get("sample_count", 0)),
            retained_by_band=retained,
            diagnostics_by_band={
                float(band): tuple(
                    copy.deepcopy(item)
                    for item in diagnostics_source.get(
                        band, diagnostics_source.get(str(band), ())
                    )
                )
                for band in bands
            },
            near_miss_by_band={
                float(band): copy.deepcopy(
                    near_source.get(band, near_source.get(str(band)))
                )
                for band in bands
            },
            job_records=tuple(copy.deepcopy(value.get("job_records", ()))),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        grouped = {float(band): [] for band in bands}
        for candidate in value:
            grouped[_candidate_band(candidate)].append(copy.deepcopy(candidate))
        return FarHandStaticOutcome(
            sample_count=0,
            retained_by_band={
                band: tuple(candidates) for band, candidates in grouped.items()
            },
            diagnostics_by_band={float(band): () for band in bands},
            near_miss_by_band={float(band): None for band in bands},
            job_records=(),
        )
    raise TypeError("static runner returned an unsupported outcome")


def _resolve_budget(
    campaign: FarHandFingertipCampaignParameters,
    definition: ExperimentDefinition,
    *,
    budget: FarHandTuningBudget | None,
    kinematic_samples_per_pitch: int | None,
    dynamic_candidate_count: int | None,
    local_refine_seed_count: int | None,
    local_refine_per_seed: int | None,
    final_candidate_count: int | None,
    perturbations_per_final: int | None,
    fallback_kinematic_samples_per_pitch: int | None,
) -> FarHandTuningBudget:
    if budget is not None:
        return budget
    band_count = len(campaign.tilt_band_centers_deg)

    def per_band(total: int | None, default: int, label: str) -> int:
        if total is None:
            return default
        value = _positive_int(total, label)
        return max(1, math.ceil(value / band_count))

    primary = (
        campaign.primary_static_samples_per_band
        if kinematic_samples_per_pitch is None
        else _positive_int(
            kinematic_samples_per_pitch, "kinematic_samples_per_pitch"
        )
    )
    fallback = (
        campaign.fallback_static_samples_per_band
        if fallback_kinematic_samples_per_pitch is None
        else _positive_int(
            fallback_kinematic_samples_per_pitch,
            "fallback_kinematic_samples_per_pitch",
        )
    )
    return FarHandTuningBudget(
        primary_static_samples_per_band=primary,
        fallback_static_samples_per_band=fallback,
        static_retain_per_band=campaign.static_retain_per_band,
        dynamic_candidates_per_band=per_band(
            dynamic_candidate_count,
            campaign.dynamic_candidates_per_band,
            "dynamic_candidate_count",
        ),
        grasp_refine_seed_count_per_band=per_band(
            local_refine_seed_count,
            campaign.grasp_refine_seed_count_per_band,
            "local_refine_seed_count",
        ),
        grasp_refine_per_seed=(
            campaign.grasp_refine_per_seed
            if local_refine_per_seed is None
            else _positive_int(local_refine_per_seed, "local_refine_per_seed")
        ),
        manipulation_seed_count_per_band=(
            campaign.manipulation_seed_count_per_band
        ),
        manipulation_refine_per_seed=campaign.manipulation_refine_per_seed,
        exact_candidates_per_band=per_band(
            final_candidate_count,
            campaign.exact_candidates_per_band,
            "final_candidate_count",
        ),
        perturbation_count=(
            definition.robustness.perturbation_count
            if perturbations_per_final is None
            else _nonnegative_int(
                perturbations_per_final, "perturbations_per_final"
            )
        ),
    )


def tune_far_hand_fingertip(
    config: dict[str, Any],
    *,
    workers: int,
    seed: int,
    run_candidates: CandidateRunner,
    static_runner: StaticRunner = run_parallel_far_hand_static_screen,
    budget: FarHandTuningBudget | None = None,
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
    """Execute the complete v5 static, grasp, operation and robustness gates."""

    del fallback_physics_count  # v5 owns an explicit topology fallback budget.
    if config.get("run_context") is not None:
        raise ValueError("v5 tuning requires a canonical nominal config")
    validate_config(config)
    definition = resolve_experiment(config)
    campaign = definition.far_hand_campaign
    if (
        definition.experiment_id != EXPERIMENT_ID
        or definition.tuning_strategy != "far_hand_fingertip"
        or campaign is None
    ):
        raise ValueError("selected experiment is not the schema-v5 campaign")
    worker_count = _positive_int(workers, "workers")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    bands = tuple(float(value) for value in campaign.tilt_band_centers_deg)
    effective = _resolve_budget(
        campaign,
        definition,
        budget=budget,
        kinematic_samples_per_pitch=kinematic_samples_per_pitch,
        dynamic_candidate_count=dynamic_candidate_count,
        local_refine_seed_count=local_refine_seed_count,
        local_refine_per_seed=local_refine_per_seed,
        final_candidate_count=final_candidate_count,
        perturbations_per_final=perturbations_per_final,
        fallback_kinematic_samples_per_pitch=(
            fallback_kinematic_samples_per_pitch
        ),
    )
    jobs = far_hand_static_jobs(
        campaign=campaign,
        seed=seed,
        primary_samples_per_band=effective.primary_static_samples_per_band,
        fallback_samples_per_band=effective.fallback_static_samples_per_band,
    )
    static_raw = static_runner(
        copy.deepcopy(config),
        jobs=jobs,
        retain_per_band=effective.static_retain_per_band,
        workers=worker_count,
        definition=definition,
    )
    static = _coerce_static_outcome(static_raw, bands=bands)
    dynamic_configs: list[dict[str, Any]] = []
    diagnostic_fallback_bands: list[float] = []
    for band in bands:
        retained = tuple(static.retained_by_band.get(band, ()))
        anchors = tuple(
            candidate
            for candidate in retained
            if candidate.get("candidate_metadata", {}).get(
                "evidence_anchor_force_dynamic"
            )
            is True
        )
        ordinary = tuple(
            candidate
            for candidate in retained
            if candidate.get("candidate_metadata", {}).get(
                "evidence_anchor_force_dynamic"
            )
            is not True
        )
        selected = anchors + ordinary[
            : max(0, effective.dynamic_candidates_per_band - len(anchors))
        ]
        if selected:
            dynamic_configs.extend(copy.deepcopy(candidate) for candidate in selected)
            continue

        # A static hard-gate miss is still useful diagnosis, but only after a
        # real simulation.  Always run exactly one best static near miss (or a
        # canonical projected seed when the worker produced no valid pose) so
        # every declared band can publish an honest best_attempt trajectory.
        near = static.near_miss_by_band.get(band)
        near_config = near.get("config") if isinstance(near, Mapping) else None
        fallback = (
            copy.deepcopy(dict(near_config))
            if isinstance(near_config, Mapping)
            else _diagnostic_seed_config(
                config, band=band, definition=definition
            )
        )
        metadata = dict(fallback.get("candidate_metadata", {}))
        metadata.update(
            {
                "tilt_band_center_deg": float(band),
                "static_gate_fallback": True,
                "static_gate_fallback_reason": (
                    "best_static_near_miss"
                    if isinstance(near_config, Mapping)
                    else "no_static_pose_available"
                ),
            }
        )
        fallback["candidate_metadata"] = metadata
        validate_config(fallback)
        dynamic_configs.append(fallback)
        diagnostic_fallback_bands.append(float(band))

    next_id = 0
    dynamic_results, next_id = _run_stage(
        dynamic_configs,
        next_candidate_id=next_id,
        workers=worker_count,
        run_candidates=run_candidates,
        stage="close_verify",
    )
    grasp_parents = _select_per_band(
        dynamic_results,
        bands=bands,
        limit=effective.grasp_refine_seed_count_per_band,
    )
    grasp_configs: list[dict[str, Any]] = []
    for band_index, band in enumerate(bands):
        for parent_index, parent in enumerate(grasp_parents[band]):
            grasp_configs.extend(
                _local_grasp_candidates(
                    parent,
                    count=effective.grasp_refine_per_seed,
                    seed=(
                        seed
                        + _STAGE_SEED_OFFSET["grasp"]
                        + band_index * 1_000_003
                        + parent_index * 10_007
                    ),
                    definition=definition,
                )
            )
    grasp_results, next_id = _run_stage(
        grasp_configs,
        next_candidate_id=next_id,
        workers=worker_count,
        run_candidates=run_candidates,
        stage="pose_grasp_refinement",
    )
    grasp_pool = dynamic_results + grasp_results

    # Consume the one-shot upper-bound extension only for a band's current
    # best result and only when its nominal pose or thumb target is truly close.
    expansion_configs: list[dict[str, Any]] = []
    expansion_records: list[dict[str, Any]] = []
    best_by_band = _select_per_band(grasp_pool, bands=bands, limit=1)
    for band_index, band in enumerate(bands):
        if not best_by_band[band]:
            continue
        parent = best_by_band[band][0]
        decision = boundary_expansion_decision(parent["config"], definition)
        expansion_records.append({"tilt_band_center_deg": band, **decision})
        if decision["applied"]:
            expansion_configs.extend(
                _expanded_grasp_candidates(
                    parent,
                    count=effective.grasp_refine_per_seed,
                    seed=seed + _STAGE_SEED_OFFSET["boundary"] + band_index,
                    definition=definition,
                    decision=decision,
                )
            )
    expansion_results, next_id = _run_stage(
        expansion_configs,
        next_candidate_id=next_id,
        workers=worker_count,
        run_candidates=run_candidates,
        stage="one_shot_boundary_refinement",
    )
    grasp_pool.extend(expansion_results)

    qualified = _select_per_band(
        grasp_pool,
        bands=bands,
        limit=effective.manipulation_seed_count_per_band,
        predicate=grasp_succeeded,
    )
    manipulation_configs: list[dict[str, Any]] = []
    for band_index, band in enumerate(bands):
        for parent_index, parent in enumerate(qualified[band]):
            manipulation_configs.extend(
                _manipulation_candidates(
                    parent,
                    count=effective.manipulation_refine_per_seed,
                    seed=(
                        seed
                        + _STAGE_SEED_OFFSET["manipulation"]
                        + band_index * 1_000_003
                        + parent_index * 10_007
                    ),
                    definition=definition,
                )
            )
    manipulation_results, next_id = _run_stage(
        manipulation_configs,
        next_candidate_id=next_id,
        workers=worker_count,
        run_candidates=run_candidates,
        stage="manipulation_refinement",
    )
    exact_parent_pool = _deduplicate_physical_results(
        manipulation_results
        + [item for item in grasp_pool if full_succeeded(item)]
    )
    exact_parents = _select_per_band(
        exact_parent_pool,
        bands=bands,
        limit=effective.exact_candidates_per_band,
    )
    # Pin the independently reproduced seed when its initial dynamics rerun
    # passes.  Manipulation LHS samples are relative perturbations and do not
    # include the parent delta exactly, so omitting this step could throw away
    # an already-proven trajectory before the exact confirmation stage.
    for band in bands:
        reproduced = deterministic_rank_far_hand_results(
            item
            for item in exact_parent_pool
            if math.isclose(_candidate_band(item), band, abs_tol=1e-9)
            and full_succeeded(item)
            and item.get("config", {})
            .get("candidate_metadata", {})
            .get("independently_reproduced_full_success_seed")
            is True
        )
        if not reproduced:
            continue
        current = list(exact_parents[band])
        pinned = [copy.deepcopy(item) for item in reproduced]
        pinned_keys = {_physical_candidate_key(item) for item in pinned}
        current = pinned + [
            item
            for item in current
            if _physical_candidate_key(item) not in pinned_keys
        ]
        # The formal limit is 16 and currently contains at most two proven
        # seeds in one band.  For a tiny test/debug override, proof seeds remain
        # mandatory even when that means exceeding the requested diagnostic
        # count; formal campaign budgets are unchanged.
        current = current[: max(effective.exact_candidates_per_band, len(pinned))]
        exact_parents[band] = tuple(current)
    exact_configs = [
        copy.deepcopy(parent["config"])
        for band in bands
        for parent in exact_parents[band]
    ]
    exact_results, next_id = _run_stage(
        exact_configs,
        next_candidate_id=next_id,
        workers=worker_count,
        run_candidates=run_candidates,
        stage="exact_1ms_confirmation",
    )
    all_dynamic = (
        dynamic_results
        + grasp_results
        + expansion_results
        + manipulation_results
        + exact_results
    )
    ranked_all = list(deterministic_rank_far_hand_results(all_dynamic))

    # Publish exactly one genuinely simulated candidate per declared band.
    # Exact confirmations take precedence; when a band never reaches that
    # stage its most advanced real dynamics result is the honest near miss.
    finalists: list[dict[str, Any]] = []
    selected_band_candidates: list[dict[str, Any]] = []
    for band in bands:
        exact_band = [
            item
            for item in exact_results
            if math.isclose(_candidate_band(item), band, abs_tol=1e-9)
        ]
        hard_band = [item for item in exact_band if full_succeeded(item)]
        if hard_band:
            finalist = copy.deepcopy(
                deterministic_rank_far_hand_results(hard_band)[0]
            )
            finalists.append(finalist)
            selected_band_candidates.append(copy.deepcopy(finalist))
            continue
        manipulation_band = [
            item
            for item in manipulation_results
            if math.isclose(_candidate_band(item), band, abs_tol=1e-9)
        ]
        grasp_band = [
            item
            for item in grasp_pool
            if math.isclose(_candidate_band(item), band, abs_tol=1e-9)
        ]
        pool = exact_band or manipulation_band or grasp_band
        if not pool:
            raise RuntimeError(
                f"tilt band {band:g} has no real dynamics result for catalog"
            )
        selected_band_candidates.append(
            copy.deepcopy(deterministic_rank_far_hand_results(pool)[0])
        )

    nominal_success = bool(finalists)
    best_source = copy.deepcopy(
        deterministic_rank_far_hand_results(
            finalists if finalists else selected_band_candidates
        )[0]
    )

    size_results: list[dict[str, Any]] = []
    perturbation_results: list[dict[str, Any]] = []
    if nominal_success:
        size_results, next_id = _run_stage(
            generate_far_hand_size_cases(
                best_source["config"], definition=definition
            ),
            next_candidate_id=next_id,
            workers=worker_count,
            run_candidates=run_candidates,
            stage="post_success_size_sweep",
            material_policy="constant_density_size_sweep",
        )
        if effective.perturbation_count:
            perturbation_results, next_id = _run_stage(
                generate_far_hand_perturbation_configs(
                    best_source["config"],
                    count=effective.perturbation_count,
                    seed=seed + _STAGE_SEED_OFFSET["robustness"],
                    definition=definition,
                ),
                next_candidate_id=next_id,
                workers=worker_count,
                run_candidates=run_candidates,
                stage="robustness_trial",
                material_policy="local_pose_and_material_perturbation",
            )
    perturbation_passes = sum(
        full_succeeded(result) for result in perturbation_results
    )
    required_perturbation_passes = (
        math.ceil(
            effective.perturbation_count
            * definition.robustness.required_pass_count
            / definition.robustness.perturbation_count
        )
        if effective.perturbation_count
        else 0
    )
    robust_passed = bool(
        nominal_success
        and effective.perturbation_count
        == definition.robustness.perturbation_count
        and perturbation_passes >= required_perturbation_passes
    )

    grasp_success = any(grasp_succeeded(item) for item in grasp_pool)
    manipulation_success = any(
        manipulation_succeeded(item)
        for item in manipulation_results + exact_results
    )
    if robust_passed:
        classification = "validated_far_hand_fingertip_robust"
        stop_reason = "validated_nominal_and_robust"
    elif nominal_success:
        classification = "far_hand_fingertip_nominal_hard_pass"
        stop_reason = "validated_nominal_not_robust"
    elif manipulation_results:
        classification = "far_hand_fingertip_manipulation_near_miss"
        stop_reason = "no_exact_hard_pass"
    elif grasp_success:
        classification = "far_hand_fingertip_stable_grasp_only"
        stop_reason = "no_manipulation_candidate_passed"
    else:
        classification = "far_hand_fingertip_not_validated"
        stop_reason = "no_qualified_grasp"

    perturbation_seed = seed + _STAGE_SEED_OFFSET["robustness"]
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
        "robustness_passed": robust_passed,
        "stop_reason": stop_reason,
        "note": (
            "An exact 1 ms candidate and the 45/50 perturbation gate passed."
            if robust_passed
            else "An exact 1 ms constant-density candidate passed; robustness remains unproven."
            if nominal_success
            else "No exact schema-v5 hard pass was found; this is the best simulated near miss."
        ),
    }
    # The alias source must be one of the exact objects sent to the catalog.
    for index, item in enumerate(selected_band_candidates):
        if int(item["candidate_id"]) == int(best["candidate_id"]):
            selected_band_candidates[index] = copy.deepcopy(best)
            break
    else:  # pragma: no cover - guarded by selection construction above.
        raise RuntimeError("best candidate is not a published band selection")

    boundary_limited = False
    expansion = best["config"].get("candidate_metadata", {}).get(
        "boundary_expansion", {}
    )
    if isinstance(expansion, Mapping) and expansion.get("applied") is True:
        resolved = resolved_pose_constraint_values(best["config"])
        policy = campaign.boundary_expansion
        boundary_limited = bool(
            expansion.get("distance_expanded")
            and float(resolved["root_cube_distance_m"])
            >= policy.expanded_root_cube_distance_max_m
            - policy.distance_tolerance_m
            or expansion.get("thumb_bend_expanded")
            and float(
                best["config"]["control"]["grasp_targets_rad"][
                    THUMB_BEND_ACTUATOR
                ]
            )
            >= definition.search_bounds.actuator_targets_rad[
                THUMB_BEND_ACTUATOR
            ][1]
            + policy.thumb_bend_expand_rad
            - policy.thumb_bend_tolerance_rad
        )

    per_band = {}
    for band in bands:
        band_results = [
            result for result in all_dynamic if math.isclose(_candidate_band(result), band)
        ]
        band_ranked = deterministic_rank_far_hand_results(band_results)
        selected = next(
            item
            for item in selected_band_candidates
            if math.isclose(_candidate_band(item), band, abs_tol=1e-9)
        )
        per_band[str(band)] = {
            "static_retained": len(static.retained_by_band.get(band, ())),
            "static_gate_fallback": band in diagnostic_fallback_bands,
            "dynamic_count": len(band_results),
            "grasp_success_count": sum(grasp_succeeded(item) for item in band_results),
            "full_success_count": sum(full_succeeded(item) for item in band_results),
            "selected_candidate_id": int(selected["candidate_id"]),
            "best_candidate_id": int(selected["candidate_id"]),
            "best_static_near_miss": copy.deepcopy(
                static.near_miss_by_band.get(band)
            ),
        }
    top_candidates = [copy.deepcopy(best)]
    top_candidates.extend(
        copy.deepcopy(item)
        for item in ranked_all
        if int(item["candidate_id"]) != int(best["candidate_id"])
    )
    top_candidates = top_candidates[:20]
    return {
        "campaign_kind": "far_hand_fingertip",
        "experiment_id": EXPERIMENT_ID,
        "seed": int(seed),
        "workers": worker_count,
        "formal_budget": campaign.budget_config(),
        "effective_budget": effective.as_dict(band_count=len(bands)),
        "legacy_parameters": copy.deepcopy(dict(legacy_parameters or {})),
        "static_screen": {
            "sample_count": static.sample_count,
            "job_records": copy.deepcopy(static.job_records),
            "diagnostic_fallback_bands_deg": diagnostic_fallback_bands,
        },
        "stage_counts": {
            "close_verify": len(dynamic_results),
            "pose_grasp_refinement": len(grasp_results),
            "one_shot_boundary_refinement": len(expansion_results),
            "manipulation_refinement": len(manipulation_results),
            "exact_1ms_confirmation": len(exact_results),
            "post_success_size_sweep": len(size_results),
            "robustness_trial": len(perturbation_results),
        },
        "candidate_count": len(all_dynamic),
        "simulation_count": (
            len(all_dynamic) + len(size_results) + len(perturbation_results)
        ),
        "diagnostic_simulation_count": len(diagnostic_fallback_bands),
        "passing_candidates": len(finalists),
        "stable_grasp_candidate_count": sum(
            grasp_succeeded(item) for item in grasp_pool
        ),
        "grasp_success": grasp_success,
        "manipulation_success": manipulation_success,
        "fixed_mass_success": False,
        "constant_density_success": nominal_success,
        "nominal_success": nominal_success,
        "robustness_success": robust_passed,
        "nominal_passed": nominal_success,
        "robust_passed": robust_passed,
        "perturbation_passes": perturbation_passes,
        "required_perturbation_passes": required_perturbation_passes,
        "perturbation_probe_count": len(perturbation_results),
        "perturbation_seed": int(perturbation_seed),
        "boundary_expansion": expansion_records,
        "boundary_limited": boundary_limited,
        "per_band": per_band,
        "finalists": [copy.deepcopy(item) for item in finalists],
        "selected_band_candidates": copy.deepcopy(selected_band_candidates),
        "best": best,
        "top_candidates": top_candidates,
        "size_sweep": size_results,
        "perturbations": perturbation_results,
        "local_perturbation_probes": (
            [copy.deepcopy(probe_record)] if perturbation_results else []
        ),
        "stop_reason": stop_reason,
    }


def run_far_hand_robustness(
    config: dict[str, Any],
    *,
    workers: int,
    seed: int,
    run_candidates: CandidateRunner,
) -> dict[str, Any]:
    """Run nominal first; size and perturbation cases never mask its failure."""

    if config.get("run_context") is not None:
        raise ValueError("v5 robustness requires a canonical nominal config")
    validate_config(config)
    definition = resolve_experiment(config)
    campaign = definition.far_hand_campaign
    if campaign is None or definition.experiment_id != EXPERIMENT_ID:
        raise ValueError("far-hand robustness requires the v5 experiment")
    worker_count = _positive_int(workers, "workers")
    nominal = run_candidates([(0, copy.deepcopy(config))], worker_count)
    if len(nominal) != 1 or int(nominal[0].get("candidate_id", -1)) != 0:
        raise RuntimeError("nominal robustness runner did not preserve ID 0")
    nominal_result = nominal[0]
    nominal_passed = full_succeeded(nominal_result)
    size_results: list[dict[str, Any]] = []
    perturbation_results: list[dict[str, Any]] = []
    if nominal_passed:
        size_configs = generate_far_hand_size_cases(config, definition=definition)
        size_results = run_candidates(
            [
                (index + 1, candidate)
                for index, candidate in enumerate(size_configs)
            ],
            worker_count,
        )
        perturbation_configs = generate_far_hand_perturbation_configs(
            config,
            count=definition.robustness.perturbation_count,
            seed=seed,
            definition=definition,
        )
        offset = 1 + len(size_results)
        perturbation_results = run_candidates(
            [
                (offset + index, candidate)
                for index, candidate in enumerate(perturbation_configs)
            ],
            worker_count,
        )
    size_records = [
        {
            "edge_m": float(result["config"]["cube"]["edge_m"]),
            "mass_kg": float(result["config"]["cube"]["mass_kg"]),
            "friction": float(result["config"]["cube"]["friction"]),
            "passed": full_succeeded(result),
            "failed_checks": copy.deepcopy(
                result["summary"].get("failed_checks", [])
            ),
            "stage_status": copy.deepcopy(
                result["summary"].get("stage_status", {})
            ),
            "metrics": copy.deepcopy(result["summary"].get("metrics", {})),
        }
        for result in size_results
    ]
    perturbation_records = [
        {
            "trial": index,
            "passed": full_succeeded(result),
            "failed_checks": copy.deepcopy(
                result["summary"].get("failed_checks", [])
            ),
            "checks": copy.deepcopy(result["summary"].get("checks", {})),
            "stage_status": copy.deepcopy(
                result["summary"].get("stage_status", {})
            ),
            "metrics": copy.deepcopy(result["summary"].get("metrics", {})),
            "cube": copy.deepcopy(result["config"]["cube"]),
            "hand_pose": copy.deepcopy(result["config"]["hand_pose"]),
            "resolved_perturbations": copy.deepcopy(
                result["config"].get("candidate_metadata", {}).get(
                    "resolved_perturbations", {}
                )
            ),
        }
        for index, result in enumerate(perturbation_results)
    ]
    perturbation_passes = sum(item["passed"] for item in perturbation_records)
    robust_passed = bool(
        nominal_passed
        and len(perturbation_records) == definition.robustness.perturbation_count
        and perturbation_passes >= definition.robustness.required_pass_count
    )
    return {
        "campaign_kind": "far_hand_fingertip",
        "seed": int(seed),
        "nominal_passed": nominal_passed,
        "nominal_hard_constraints_passed": bool(
            nominal_result["summary"].get("passed", False)
        ),
        "nominal_full_success": nominal_passed,
        "nominal_is_constant_density": math.isclose(
            float(config["cube"]["mass_kg"]),
            campaign.constant_density_mass_kg(float(config["cube"]["edge_m"])),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "nominal_friction_matches": math.isclose(
            float(config["cube"]["friction"]), campaign.friction, abs_tol=1e-12
        ),
        "nominal_summary": copy.deepcopy(nominal_result["summary"]),
        "grid_case_count": len(size_records),
        "grid_passes": sum(item["passed"] for item in size_records),
        "grid": size_records,
        "hardest_passing_grid_case": None,
        "hardest_passing_constant_density_grid_case": None,
        "hardest_passing_fixed_20g_control_grid_case": None,
        "perturbation_trial_count": len(perturbation_records),
        "perturbation_passes": perturbation_passes,
        "required_perturbation_passes": definition.robustness.required_pass_count,
        "robust_passed": robust_passed,
        "perturbations": perturbation_records,
        "stop_reason": (
            "completed"
            if nominal_passed
            else "nominal_failed_post_success_campaign_skipped"
        ),
    }


__all__ = [
    "EXACT_TIMESTEP_S",
    "FarHandStaticJob",
    "FarHandStaticOutcome",
    "FarHandTuningBudget",
    "boundary_expansion_decision",
    "closure_sweep_diagnostic",
    "deterministic_rank_far_hand_results",
    "far_hand_candidate_rank",
    "far_hand_static_candidate_advances",
    "far_hand_static_jobs",
    "generate_far_hand_perturbation_configs",
    "generate_far_hand_size_cases",
    "materialize_far_hand_candidate",
    "root_pitch_for_finger_down_tilt_deg",
    "run_far_hand_robustness",
    "run_parallel_far_hand_static_screen",
    "tune_far_hand_fingertip",
]
