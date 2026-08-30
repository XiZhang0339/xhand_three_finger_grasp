"""Deterministic global top-eight joint/controller refinement for schema v14.

This module is intentionally independent of the campaign runner.  It turns
already-evaluated candidate records into immutable full-reset job
specifications; execution/persistence remain the runner's responsibility.
No random draw depends on input order, worker count, or completion order.
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

from ..config import (
    ACTIVE_ACTUATORS,
    contact_preload_targets,
    precontact_targets,
    validate_config,
)
from ..experiment import (
    ContactFeedbackParameters,
    ManipulationPlanParameters,
    resolve_experiment,
)
from ..grasp_pose import canonical_sha256
from ..scene import build_model
from ..trajectory import minimum_jerk, quintic_c2_knot_derivatives
from ..v14_identity import (
    V14_PAIR_ID_FIELDS,
    v14_joint_refined_controller_id,
)
from .contact_constrained_planner import rank_contact_constrained_candidates


DEFAULT_SEED = 20260821
GLOBAL_PARENT_COUNT = 8
CANDIDATES_PER_PARENT = 128
ACTIVE_DIMENSION = len(ACTIVE_ACTUATORS)
_FINGERS = ("thumb", "index", "mid")
_FINGER_ACTUATORS = (
    ACTIVE_ACTUATORS[0:3],
    ACTIVE_ACTUATORS[3:6],
    ACTIVE_ACTUATORS[6:8],
)
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class JointRefinementBudget:
    """Versioned local-search budget with the required 8 x 128 default."""

    parent_count: int = GLOBAL_PARENT_COUNT
    candidates_per_parent: int = CANDIDATES_PER_PARENT
    seed: int = DEFAULT_SEED
    plan_radius_fraction: float = 0.06
    preload_radius_rad: float = 0.01
    feedback_kp_scale: tuple[float, float] = (0.5, 1.5)
    feedback_ki_scale: tuple[float, float] = (0.0, 1.5)

    def __post_init__(self) -> None:
        for name in ("parent_count", "candidates_per_parent"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        radius = float(self.plan_radius_fraction)
        preload = float(self.preload_radius_rad)
        if not math.isfinite(radius) or not 0.0 < radius <= 1.0:
            raise ValueError("plan_radius_fraction must lie within (0, 1]")
        if not math.isfinite(preload) or preload <= 0.0:
            raise ValueError("preload_radius_rad must be positive and finite")
        object.__setattr__(self, "plan_radius_fraction", radius)
        object.__setattr__(self, "preload_radius_rad", preload)
        for name in ("feedback_kp_scale", "feedback_ki_scale"):
            lower, upper = (float(value) for value in getattr(self, name))
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower < 0.0
                or lower > 1.0
                or upper < 1.0
            ):
                raise ValueError(f"{name} must bracket the exact-parent scale 1")
            object.__setattr__(self, name, (lower, upper))

    @property
    def maximum_candidate_count(self) -> int:
        return self.parent_count * self.candidates_per_parent

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "parent_count": self.parent_count,
            "candidates_per_parent": self.candidates_per_parent,
            "maximum_candidate_count": self.maximum_candidate_count,
            "seed": self.seed,
            "plan_radius_fraction": self.plan_radius_fraction,
            "preload_radius_rad": self.preload_radius_rad,
            "feedback_kp_scale": list(self.feedback_kp_scale),
            "feedback_ki_scale": list(self.feedback_ki_scale),
        }


def _named_bounds(
    values: Mapping[str, Sequence[float]], label: str
) -> MappingProxyType:
    if set(values) != set(ACTIVE_ACTUATORS):
        raise ValueError(f"{label} must contain exactly the active actuators")
    normalized: dict[str, tuple[float, float]] = {}
    for name in ACTIVE_ACTUATORS:
        pair = values[name]
        if len(pair) != 2:
            raise ValueError(f"{label}.{name} must contain two values")
        lower, upper = (float(value) for value in pair)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"{label}.{name} is invalid")
        normalized[name] = (lower, upper)
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class JointRefinementLimits:
    """Physical command, registered preload, and manipulation bounds."""

    command_target_rad: Mapping[str, Sequence[float]]
    preload_target_rad: Mapping[str, Sequence[float]]
    registered_plan_delta_rad: Mapping[str, Sequence[float]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command_target_rad",
            _named_bounds(self.command_target_rad, "command_target_rad"),
        )
        object.__setattr__(
            self,
            "preload_target_rad",
            _named_bounds(self.preload_target_rad, "preload_target_rad"),
        )
        object.__setattr__(
            self,
            "registered_plan_delta_rad",
            _named_bounds(
                self.registered_plan_delta_rad, "registered_plan_delta_rad"
            ),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "command_target_rad": {
                name: list(self.command_target_rad[name])
                for name in ACTIVE_ACTUATORS
            },
            "preload_target_rad": {
                name: list(self.preload_target_rad[name])
                for name in ACTIVE_ACTUATORS
            },
            "registered_plan_delta_rad": {
                name: list(self.registered_plan_delta_rad[name])
                for name in ACTIVE_ACTUATORS
            },
        }


def resolve_joint_refinement_limits(
    config: Mapping[str, Any],
) -> JointRefinementLimits:
    """Compile one resolved pair and intersect ctrl, joint, and search bounds."""

    if int(config.get("schema_version", 0)) != 14:
        raise ValueError("contact-preserving joint refinement requires schema v14")
    model, _ = build_model(copy.deepcopy(dict(config)))
    definition = resolve_experiment(config)
    registered = definition.search_bounds.manipulation_delta_rad
    if registered is None or set(registered) != set(ACTIVE_ACTUATORS):
        raise ValueError("the v14 experiment must register manipulation bounds")
    command: dict[str, tuple[float, float]] = {}
    preload_target: dict[str, tuple[float, float]] = {}
    plan: dict[str, tuple[float, float]] = {}
    preload = contact_preload_targets(copy.deepcopy(dict(config)))
    for name in ACTIVE_ACTUATORS:
        actuator_id = int(model.actuator(name).id)
        if not bool(model.actuator_ctrllimited[actuator_id]):
            raise ValueError(f"{name} has no finite ctrl range")
        lower, upper = (
            float(value) for value in model.actuator_ctrlrange[actuator_id]
        )
        if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError(f"{name} does not use a joint transmission")
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0 or not bool(model.jnt_limited[joint_id]):
            raise ValueError(f"{name} does not target a limited joint")
        lower = max(lower, float(model.jnt_range[joint_id, 0]))
        upper = min(upper, float(model.jnt_range[joint_id, 1]))
        if lower > upper:
            raise ValueError(f"{name} has no common ctrl/joint range")
        command[name] = (lower, upper)
        registered_preload_lower, registered_preload_upper = (
            float(value)
            for value in definition.search_bounds.actuator_targets_rad[name]
        )
        preload_target[name] = (
            max(lower, registered_preload_lower),
            min(upper, registered_preload_upper),
        )
        if preload_target[name][0] > preload_target[name][1]:
            raise ValueError(f"{name} has no registered real preload range")
        plan[name] = (
            max(float(registered[name][0]), lower - float(preload[name])),
            min(float(registered[name][1]), upper - float(preload[name])),
        )
        if plan[name][0] > plan[name][1]:
            raise ValueError(f"{name} has no registered real manipulation range")
    return JointRefinementLimits(command, preload_target, plan)


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=np.float64)
    for column in range(dimensions):
        order = rng.permutation(count)
        result[:, column] = (order + rng.random(count)) / float(count)
    result = 2.0 * result - 1.0
    # Every basin contains its exact parent as a deterministic anchor.
    result[0] = 0.0
    return result


def _parent_seed(seed: int, parent_identity: str) -> int:
    digest = bytes.fromhex(parent_identity)
    words = np.frombuffer(digest[:16], dtype="<u4").astype(np.uint32)
    return int(
        np.random.SeedSequence([int(seed), *(int(value) for value in words)])
        .generate_state(1)[0]
    )


def _scale_from_unit(unit: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    value = float(unit)
    return (
        1.0 + value * (upper - 1.0)
        if value >= 0.0
        else 1.0 + value * (1.0 - lower)
    )


def _plan_profile(config: Mapping[str, Any], name: str) -> np.ndarray:
    waypoints = np.asarray(
        config["manipulation_plan"]["actuator_waypoints_rad"][name],
        dtype=np.float64,
    )
    terminal = float(config["control"]["manipulation_delta_rad"][name])
    if abs(terminal) > 1e-10:
        profile = waypoints / terminal
    elif np.max(np.abs(waypoints), initial=0.0) <= 1e-12:
        profile = np.asarray(
            [minimum_jerk(index / (waypoints.size - 1)) for index in range(waypoints.size)],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"{name} has non-zero waypoints with a zero terminal")
    if (
        not np.isfinite(profile).all()
        or abs(float(profile[0])) > 1e-12
        or abs(float(profile[-1]) - 1.0) > 1e-10
    ):
        raise ValueError(f"{name} manipulation profile is invalid")
    return profile


def _bezier_controls(times: np.ndarray, profile: np.ndarray) -> np.ndarray:
    velocities, accelerations = quintic_c2_knot_derivatives(times, profile)
    result = []
    for index, duration in enumerate(np.diff(times)):
        value0 = profile[index]
        value1 = profile[index + 1]
        velocity0 = velocities[index]
        velocity1 = velocities[index + 1]
        acceleration0 = accelerations[index]
        acceleration1 = accelerations[index + 1]
        result.extend(
            (
                value0,
                value0 + duration * velocity0 / 5.0,
                value0 + 2.0 * duration * velocity0 / 5.0 + duration**2 * acceleration0 / 20.0,
                value1 - 2.0 * duration * velocity1 / 5.0 + duration**2 * acceleration1 / 20.0,
                value1 - duration * velocity1 / 5.0,
                value1,
            )
        )
    return np.asarray(result, dtype=np.float64)


def _conditional_plan_bounds(
    *,
    registered: Sequence[float],
    command: Sequence[float],
    preload: float,
    controls: np.ndarray,
    maximum_profile_increment: float,
    max_knot_delta_rad: float,
) -> tuple[float, float]:
    lower, upper = (float(value) for value in registered)
    command_lower, command_upper = (float(value) for value in command)
    for coefficient in controls:
        if coefficient > _EPSILON:
            lower = max(lower, (command_lower - preload) / coefficient)
            upper = min(upper, (command_upper - preload) / coefficient)
        elif coefficient < -_EPSILON:
            lower = max(lower, (command_upper - preload) / coefficient)
            upper = min(upper, (command_lower - preload) / coefficient)
        elif not command_lower - _EPSILON <= preload <= command_upper + _EPSILON:
            raise ValueError("preload lies outside the real command range")
    if maximum_profile_increment > _EPSILON:
        amplitude = float(max_knot_delta_rad) / maximum_profile_increment
        lower = max(lower, -amplitude)
        upper = min(upper, amplitude)
    if lower > upper + _EPSILON:
        raise ValueError("preload leaves no safe manipulation-plan interval")
    return lower, upper


@dataclass(frozen=True, slots=True)
class _PreparedParent:
    limits: JointRefinementLimits
    times: np.ndarray
    maximum_knot_delta_rad: float
    profiles: Mapping[str, np.ndarray]
    bezier_controls: Mapping[str, np.ndarray]
    maximum_profile_increment: Mapping[str, float]
    config_sha256: str


def _prepare_parent(
    config: Mapping[str, Any], limits: JointRefinementLimits
) -> _PreparedParent:
    times = np.asarray(
        config["manipulation_plan"]["knot_times_s"], dtype=np.float64
    )
    profiles = {name: _plan_profile(config, name) for name in ACTIVE_ACTUATORS}
    controls = {
        name: _bezier_controls(times, profiles[name]) for name in ACTIVE_ACTUATORS
    }
    increments = {
        name: float(np.max(np.abs(np.diff(profiles[name])), initial=0.0))
        for name in ACTIVE_ACTUATORS
    }
    times.setflags(write=False)
    for values in (*profiles.values(), *controls.values()):
        values.setflags(write=False)
    return _PreparedParent(
        limits=limits,
        times=times,
        maximum_knot_delta_rad=float(
            config["manipulation_plan"]["max_knot_delta_rad"]
        ),
        profiles=MappingProxyType(profiles),
        bezier_controls=MappingProxyType(controls),
        maximum_profile_increment=MappingProxyType(increments),
        config_sha256=canonical_sha256(config),
    )


def _preserve_closing_rays(
    proposed: dict[str, float],
    base: Mapping[str, float],
    precontact: Mapping[str, float],
) -> dict[str, float]:
    result = dict(proposed)
    for names in _FINGER_ACTUATORS:
        base_direction = np.asarray(
            [float(base[name]) - float(precontact[name]) for name in names]
        )
        candidate_direction = np.asarray(
            [float(result[name]) - float(precontact[name]) for name in names]
        )
        if (
            np.max(np.abs(candidate_direction), initial=0.0) <= 1e-6
            or float(candidate_direction @ base_direction) <= 0.0
        ):
            for name in names:
                result[name] = float(base[name])
    return result


def _feedback_config(
    base: Mapping[str, Any], kp_units: np.ndarray, ki_units: np.ndarray, budget: JointRefinementBudget
) -> dict[str, Any]:
    return ContactFeedbackParameters(
        schema_version=int(base.get("schema_version", 1)),
        strategy=str(base["strategy"]),
        filter_time_constant_s=float(base["filter_time_constant_s"]),
        kp_rad_per_n={
            finger: float(base["kp_rad_per_n"][finger])
            * _scale_from_unit(kp_units[index], budget.feedback_kp_scale)
            for index, finger in enumerate(_FINGERS)
        },
        ki_rad_per_n_s={
            finger: float(base["ki_rad_per_n_s"][finger])
            * _scale_from_unit(ki_units[index], budget.feedback_ki_scale)
            for index, finger in enumerate(_FINGERS)
        },
        integral_limit_n_s=float(base["integral_limit_n_s"]),
        correction_limit_rad=float(base["correction_limit_rad"]),
        rate_limit_rad_s=float(base["rate_limit_rad_s"]),
        acceleration_limit_rad_s2=float(base["acceleration_limit_rad_s2"]),
        force_risk_n=float(base["force_risk_n"]),
        freeze_on_risk=bool(base["freeze_on_risk"]),
        max_loss_s=float(base["max_loss_s"]),
        recovery_behavior=str(base["recovery_behavior"]),
        operation_contact_duty_min=float(base["operation_contact_duty_min"]),
        tangent_slip_freeze_threshold_m=(
            float(base["tangent_slip_freeze_threshold_m"])
            if int(base.get("schema_version", 1)) >= 2
            else None
        ),
        tangent_slip_abort_threshold_m=(
            float(base["tangent_slip_abort_threshold_m"])
            if int(base.get("schema_version", 1)) >= 2
            else None
        ),
    ).as_config()


def _materialize_candidate(
    parent: Mapping[str, Any],
    prepared: _PreparedParent,
    units: np.ndarray,
    *,
    budget: JointRefinementBudget,
    parent_rank: int,
    local_index: int,
    validate: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(parent["config"]))
    if int(config.get("schema_version", 0)) != 14:
        raise ValueError("every refinement parent config must use schema v14")
    limits = prepared.limits
    base_preload = contact_preload_targets(copy.deepcopy(config))
    precontact = precontact_targets(copy.deepcopy(config))
    preload: dict[str, float] = {}
    for index, name in enumerate(ACTIVE_ACTUATORS):
        lower, upper = limits.preload_target_rad[name]
        preload[name] = float(
            np.clip(
                float(base_preload[name])
                + units[ACTIVE_DIMENSION + index] * budget.preload_radius_rad,
                lower,
                upper,
            )
        )
    preload = _preserve_closing_rays(preload, base_preload, precontact)

    times = prepared.times
    maximum_knot_delta = prepared.maximum_knot_delta_rad
    terminal: dict[str, float] = {}
    profiles: dict[str, np.ndarray] = {}
    for index, name in enumerate(ACTIVE_ACTUATORS):
        profile = prepared.profiles[name]
        profiles[name] = profile
        lower, upper = _conditional_plan_bounds(
            registered=limits.registered_plan_delta_rad[name],
            command=limits.command_target_rad[name],
            preload=preload[name],
            controls=prepared.bezier_controls[name],
            maximum_profile_increment=prepared.maximum_profile_increment[name],
            max_knot_delta_rad=maximum_knot_delta,
        )
        base_value = float(config["control"]["manipulation_delta_rad"][name])
        radius = budget.plan_radius_fraction * (upper - lower)
        terminal[name] = float(
            np.clip(base_value + units[index] * radius, lower, upper)
        )

    old_plan = config["manipulation_plan"]
    plan = ManipulationPlanParameters(
        schema_version=1,
        profile=str(old_plan["profile"]),
        duration_s=float(old_plan["duration_s"]),
        knot_times_s=tuple(float(value) for value in times),
        actuator_waypoints_rad={
            name: tuple(float(value) for value in profiles[name] * terminal[name])
            for name in ACTIVE_ACTUATORS
        },
        desired_cube_position_delta_m=tuple(
            tuple(float(axis) for axis in value)
            for value in old_plan["desired_cube_position_delta_m"]
        ),
        desired_cube_rotation_vector_rad=tuple(
            tuple(float(axis) for axis in value)
            for value in old_plan["desired_cube_rotation_vector_rad"]
        ),
        max_knot_delta_rad=maximum_knot_delta,
        trust_region_backtracks=int(old_plan["trust_region_backtracks"]),
    )
    config["control"]["contact_preload_targets_rad"] = preload
    config["control"]["manipulation_delta_rad"] = terminal
    config["manipulation_plan"] = plan.as_config()
    feedback_offset = 2 * ACTIVE_DIMENSION
    config["contact_feedback"] = _feedback_config(
        config["contact_feedback"],
        units[feedback_offset : feedback_offset + 3],
        units[feedback_offset + 3 : feedback_offset + 6],
        budget,
    )
    parent_id = int(parent["candidate_id"])
    parent_config_sha = prepared.config_sha256
    identity = {
        "schema_version": 1,
        "kind": "v14_global_joint_local_refinement",
        "seed": budget.seed,
        "parent_candidate_id": parent_id,
        "parent_config_sha256": parent_config_sha,
        "local_index": local_index,
        "plan_id": config["manipulation_plan"]["plan_id"],
        "feedback_id": config["contact_feedback"]["feedback_id"],
        "contact_preload_targets_rad": preload,
    }
    digest = canonical_sha256(identity)
    candidate_id = 14 * 10**15 + int(digest[:12], 16) % 10**14
    # Unit-level/pre-planning refinement can start from the ID-free tune
    # template.  Do not manufacture a lone controller ID for that search
    # intermediate.  A real planned parent already carries the pair/planner
    # chain and receives the canonical refined controller ID.
    if all(config.get(name) for name in (*V14_PAIR_ID_FIELDS, "planner_id")):
        config["controller_id"] = v14_joint_refined_controller_id(config)
    else:
        config.pop("controller_id", None)
    config.setdefault("candidate_metadata", {})["v14_joint_local_refinement"] = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "candidate_sha256": digest,
        "parent_candidate_id": parent_id,
        "parent_rank": parent_rank,
        "local_index": local_index,
        "seed": budget.seed,
        "full_reset_required": True,
    }
    if validate:
        validate_config(config)
    sequence_index = local_index * budget.parent_count + parent_rank
    return {
        "joint_refinement_job_schema_version": 1,
        "candidate_id": candidate_id,
        "candidate_sha256": digest,
        "parent_candidate_id": parent_id,
        "parent_rank": parent_rank,
        "local_index": local_index,
        "job_sequence_index": sequence_index,
        "config": config,
        "job_metadata": {
            "stage": "v14_global_top8_joint_local_refinement",
            "parent_candidate_id": parent_id,
            "parent_rank": parent_rank,
            "local_index": local_index,
            "job_sequence_index": sequence_index,
            "seed": budget.seed,
        },
    }


LimitsResolver = Callable[[Mapping[str, Any]], JointRefinementLimits]


def build_joint_refinement_job_specs(
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    budget: JointRefinementBudget = JointRefinementBudget(),
    limits_resolver: LimitsResolver | None = None,
    validate_configs: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Return interleaved immutable jobs for the globally ranked top eight."""

    ranked = rank_contact_constrained_candidates(candidate_records)
    parents = tuple(ranked[: budget.parent_count])
    if len(parents) != budget.parent_count:
        raise ValueError(
            f"joint refinement requires {budget.parent_count} candidate records"
        )
    resolver = resolve_joint_refinement_limits if limits_resolver is None else limits_resolver
    per_parent: list[tuple[_PreparedParent, np.ndarray]] = []
    dimensions = 2 * ACTIVE_DIMENSION + 6
    for parent in parents:
        if "candidate_id" not in parent or not isinstance(parent.get("config"), Mapping):
            raise ValueError("each refinement parent requires candidate_id and config")
        identity = canonical_sha256(
            {
                "candidate_id": int(parent["candidate_id"]),
                "config_sha256": canonical_sha256(parent["config"]),
            }
        )
        limits = resolver(parent["config"])
        per_parent.append(
            (
                _prepare_parent(parent["config"], limits),
                _latin_hypercube(
                    budget.candidates_per_parent,
                    dimensions,
                    _parent_seed(budget.seed, identity),
                ),
            )
        )

    jobs = []
    for local_index in range(budget.candidates_per_parent):
        for parent_rank, parent in enumerate(parents):
            prepared, units = per_parent[parent_rank]
            jobs.append(
                _materialize_candidate(
                    parent,
                    prepared,
                    units[local_index],
                    budget=budget,
                    parent_rank=parent_rank,
                    local_index=local_index,
                    validate=validate_configs,
                )
            )
    # The nested loop already emits the balanced deterministic runner order.
    # Avoid deep-copying 1,024 resolved configs merely to sort an already
    # monotonic sequence; the public sorter remains available for arbitrary
    # worker-completion records.
    result = tuple(jobs)
    if len(result) != budget.maximum_candidate_count:
        raise AssertionError("joint-refinement generation violated its exact budget")
    identifiers = [int(value["candidate_id"]) for value in result]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("joint-refinement candidate ID collision")
    return result


def stable_sort_joint_refinement_jobs(
    jobs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Restore deterministic interleaved order after arbitrary worker order."""

    values = [copy.deepcopy(dict(value)) for value in jobs]
    values.sort(
        key=lambda value: (
            int(value["job_sequence_index"]),
            int(value["candidate_id"]),
        )
    )
    return tuple(values)


def rank_joint_refinement_results(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Apply the v14 contact-first total order to worker-completion results."""

    return rank_contact_constrained_candidates(records)


__all__ = [
    "CANDIDATES_PER_PARENT",
    "DEFAULT_SEED",
    "GLOBAL_PARENT_COUNT",
    "JointRefinementBudget",
    "JointRefinementLimits",
    "build_joint_refinement_job_specs",
    "rank_joint_refinement_results",
    "resolve_joint_refinement_limits",
    "stable_sort_joint_refinement_jobs",
]
