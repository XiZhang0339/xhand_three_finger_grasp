"""Deterministic time-warp rescue candidates for schema-v14 plans.

The contact-preserving controller already consumes a 21-knot C2 quintic
trajectory.  This module changes *when* the existing knots are reached without
changing the runtime controller or any legacy schema.  Two bounded, zero-mean
Gaussian log-duration bases provide a compact search over the part of the lift
where contact loss and the largest jerk have historically occurred.

The public job builder is intentionally independent of the campaign runner.
It emits authenticated, full-reset job specifications which a runner may
persist and evaluate in any worker order.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import (
    ACTIVE_ACTUATORS,
    contact_preload_targets,
    precontact_targets,
    validate_config,
)
from ..experiment import ManipulationPlanParameters
from ..grasp_pose import canonical_sha256
from ..trajectory import quintic_c2_knot_derivatives
from .contact_constrained_planner import (
    rank_contact_constrained_candidates,
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
)
from .contact_preserving_joint_refinement import (
    JointRefinementLimits,
    resolve_joint_refinement_limits,
)


TIME_WARP_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
KNOT_COUNT = 21
SEGMENT_COUNT = 20
DURATION_S = 3.0
GAUSSIAN_CENTERS = (0.47, 0.70)
GAUSSIAN_SIGMA = 0.07
COEFFICIENT_BOUNDS = (-0.45, 0.45)
MIN_SEGMENT_DURATION_S = 0.090
MAX_SEGMENT_DURATION_S = 0.240
DEFAULT_TOTAL_CANDIDATES = 256
DEFAULT_MAX_PARENTS = 4
DEFAULT_TERMINAL_RADIUS_RAD = 0.010
DEFAULT_PRELOAD_RADIUS_RAD = 0.006
_FINGER_ACTUATORS = (
    ACTIVE_ACTUATORS[0:3],
    ACTIVE_ACTUATORS[3:6],
    ACTIVE_ACTUATORS[6:8],
)
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class TimeWarpBudget:
    """Versioned fixed-budget rescue policy.

    With four parents the split is exactly 32 warp-only, 16 warp+terminal and
    16 warp+preload candidates per parent.  With fewer parents, the three
    global quotas (128/64/64) are distributed independently by parent rank, so
    the total remains exactly 256 and the allocation is explicit in the
    manifest and every job.
    """

    total_candidate_count: int = DEFAULT_TOTAL_CANDIDATES
    max_parent_count: int = DEFAULT_MAX_PARENTS
    seed: int = DEFAULT_SEED
    terminal_radius_rad: float = DEFAULT_TERMINAL_RADIUS_RAD
    preload_radius_rad: float = DEFAULT_PRELOAD_RADIUS_RAD

    def __post_init__(self) -> None:
        for name in ("total_candidate_count", "max_parent_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.total_candidate_count != DEFAULT_TOTAL_CANDIDATES:
            raise ValueError("schema-v14 time-warp rescue budget must be exactly 256")
        if self.max_parent_count != DEFAULT_MAX_PARENTS:
            raise ValueError("schema-v14 time-warp rescue supports exactly four parent slots")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("terminal_radius_rad", "preload_radius_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": TIME_WARP_SCHEMA_VERSION,
            "total_candidate_count": self.total_candidate_count,
            "max_parent_count": self.max_parent_count,
            "seed": self.seed,
            "terminal_radius_rad": self.terminal_radius_rad,
            "preload_radius_rad": self.preload_radius_rad,
            "category_totals": {
                "warp_only": self.total_candidate_count // 2,
                "warp_terminal": self.total_candidate_count // 4,
                "warp_preload": self.total_candidate_count // 4,
            },
        }


@dataclass(frozen=True, slots=True)
class _MaterializedTimeWarp:
    """One safe candidate and the *actual* independently applied scales.

    Joint backoff is attempted first.  A parent may, however, sit exactly on
    a waypoint, preload, joint, or continuous Bezier-hull boundary.  In that
    case an arbitrarily small terminal/preload perturbation can remain
    infeasible even though the requested time warp itself is safe.  Keeping
    the three scales separate prevents persisted evidence from claiming that
    an offset was applied when it was clipped or reverted to the parent.
    """

    config: dict[str, Any]
    coefficient_scale: float
    terminal_scale: float
    preload_scale: float
    fallback_reason: str


def zero_mean_gaussian_bases() -> np.ndarray:
    """Return the immutable 2 x 20 log-duration basis matrix."""

    segment_centers = (np.arange(SEGMENT_COUNT, dtype=np.float64) + 0.5) / float(
        SEGMENT_COUNT
    )
    bases = np.stack(
        [
            np.exp(
                -0.5
                * ((segment_centers - center) / float(GAUSSIAN_SIGMA)) ** 2
            )
            for center in GAUSSIAN_CENTERS
        ]
    )
    bases -= np.mean(bases, axis=1, keepdims=True)
    bases.setflags(write=False)
    return bases


_GAUSSIAN_BASES = zero_mean_gaussian_bases()


def warp_knot_times(
    parent_knot_times_s: Sequence[float],
    a1: float,
    a2: float,
) -> np.ndarray:
    """Apply the two-basis softmax duration warp.

    The zero-coefficient branch is special-cased so it reproduces the parent
    knot times bit-for-bit rather than merely within floating-point tolerance.
    """

    parent = np.asarray(parent_knot_times_s, dtype=np.float64)
    if (
        parent.shape != (KNOT_COUNT,)
        or not np.isfinite(parent).all()
        or parent[0] != 0.0
        or parent[-1] != DURATION_S
        or np.any(np.diff(parent) <= 0.0)
    ):
        raise ValueError("parent plan must contain 21 increasing knots over exactly 3 s")
    coefficients = np.asarray((a1, a2), dtype=np.float64)
    if not np.isfinite(coefficients).all() or np.any(
        coefficients < COEFFICIENT_BOUNDS[0] - _EPSILON
    ) or np.any(coefficients > COEFFICIENT_BOUNDS[1] + _EPSILON):
        raise ValueError("time-warp coefficients must lie within [-0.45, 0.45]")
    parent_durations = np.diff(parent)
    if np.any(parent_durations < MIN_SEGMENT_DURATION_S - _EPSILON) or np.any(
        parent_durations > MAX_SEGMENT_DURATION_S + _EPSILON
    ):
        raise ValueError("parent segment duration lies outside 90--240 ms")
    if float(a1) == 0.0 and float(a2) == 0.0:
        result = parent.copy()
        result.setflags(write=False)
        return result

    logits = np.log(parent_durations / float(np.sum(parent_durations)))
    logits += coefficients @ _GAUSSIAN_BASES
    weights = np.exp(logits - float(np.max(logits)))
    durations = DURATION_S * weights / float(np.sum(weights))
    if np.any(durations < MIN_SEGMENT_DURATION_S - _EPSILON) or np.any(
        durations > MAX_SEGMENT_DURATION_S + _EPSILON
    ):
        raise ValueError("warped segment duration lies outside 90--240 ms")
    result = np.concatenate(([0.0], np.cumsum(durations)))
    # Remove accumulated roundoff while preserving every positive segment.
    result[-1] = DURATION_S
    result.setflags(write=False)
    return result


def quintic_bezier_controls(
    knot_times_s: Sequence[float], knot_values: Sequence[Sequence[float]] | np.ndarray
) -> np.ndarray:
    """Convert the shared clamped C4 quintic to per-segment Bezier controls."""

    times = np.asarray(knot_times_s, dtype=np.float64)
    values = np.asarray(knot_values, dtype=np.float64)
    if times.shape != (KNOT_COUNT,) or values.shape[0] != KNOT_COUNT:
        raise ValueError("quintic hull requires 21 matching knots")
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("quintic knot values must be a finite vector or matrix")
    velocities, accelerations = quintic_c2_knot_derivatives(times, values)
    controls = np.empty((SEGMENT_COUNT, 6, values.shape[1]), dtype=np.float64)
    for index, duration in enumerate(np.diff(times)):
        value0 = values[index]
        value1 = values[index + 1]
        velocity0 = velocities[index]
        velocity1 = velocities[index + 1]
        acceleration0 = accelerations[index]
        acceleration1 = accelerations[index + 1]
        controls[index] = np.stack(
            (
                value0,
                value0 + duration * velocity0 / 5.0,
                value0
                + 2.0 * duration * velocity0 / 5.0
                + duration**2 * acceleration0 / 20.0,
                value1
                - 2.0 * duration * velocity1 / 5.0
                + duration**2 * acceleration1 / 20.0,
                value1 - duration * velocity1 / 5.0,
                value1,
            )
        )
    controls.setflags(write=False)
    return controls


def actuator_bezier_hull(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return exact Bezier controls and conservative relative-command hulls."""

    plan = ManipulationPlanParameters.from_config(config["manipulation_plan"])
    values = np.stack(
        [plan.actuator_waypoints_rad[name] for name in ACTIVE_ACTUATORS], axis=1
    )
    controls = quintic_bezier_controls(plan.knot_times_s, values)
    return {
        "controls_rad": controls,
        "relative_lower_rad": np.min(controls, axis=(0, 1)),
        "relative_upper_rad": np.max(controls, axis=(0, 1)),
    }


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


def validate_time_warped_config(
    config: Mapping[str, Any],
    *,
    limits: JointRefinementLimits | None = None,
    validate_schema: bool = True,
) -> dict[str, Any]:
    """Fail closed on the 3 s/21-knot contract and continuous command limits."""

    if int(config.get("schema_version", 0)) != 14:
        raise ValueError("contact-preserving time warp requires schema v14")
    plan = ManipulationPlanParameters.from_config(config.get("manipulation_plan", {}))
    times = np.asarray(plan.knot_times_s, dtype=np.float64)
    durations = np.diff(times)
    if len(times) != KNOT_COUNT or not math.isclose(
        plan.duration_s, DURATION_S, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("time-warp plan must contain 21 knots over exactly 3 s")
    if np.any(durations < MIN_SEGMENT_DURATION_S - _EPSILON) or np.any(
        durations > MAX_SEGMENT_DURATION_S + _EPSILON
    ):
        raise ValueError("time-warp segment duration lies outside 90--240 ms")
    control = config.get("control")
    if not isinstance(control, Mapping):
        raise ValueError("schema-v14 control block is missing")
    terminal = control.get("manipulation_delta_rad")
    if not isinstance(terminal, Mapping) or set(terminal) != set(ACTIVE_ACTUATORS):
        raise ValueError("manipulation_delta_rad must name all active actuators")
    for name in ACTIVE_ACTUATORS:
        if not math.isclose(
            float(terminal[name]),
            float(plan.actuator_waypoints_rad[name][-1]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("terminal control mirror disagrees with the plan")

    resolved_limits = (
        resolve_joint_refinement_limits(config) if limits is None else limits
    )
    preload = contact_preload_targets(copy.deepcopy(dict(config)))
    precontact = precontact_targets(copy.deepcopy(dict(config)))
    hull = actuator_bezier_hull(config)
    relative_lower = np.asarray(hull["relative_lower_rad"], dtype=np.float64)
    relative_upper = np.asarray(hull["relative_upper_rad"], dtype=np.float64)
    for index, name in enumerate(ACTIVE_ACTUATORS):
        command_lower, command_upper = resolved_limits.command_target_rad[name]
        preload_lower, preload_upper = resolved_limits.preload_target_rad[name]
        plan_lower, plan_upper = resolved_limits.registered_plan_delta_rad[name]
        if not command_lower - _EPSILON <= float(precontact[name]) <= command_upper + _EPSILON:
            raise ValueError(f"{name} precontact target exceeds real command limits")
        if not preload_lower - _EPSILON <= float(preload[name]) <= preload_upper + _EPSILON:
            raise ValueError(f"{name} preload target exceeds registered limits")
        if not plan_lower - _EPSILON <= float(terminal[name]) <= plan_upper + _EPSILON:
            raise ValueError(f"{name} terminal delta exceeds registered limits")
        continuous_lower = float(preload[name]) + float(relative_lower[index])
        continuous_upper = float(preload[name]) + float(relative_upper[index])
        if continuous_lower < command_lower - _EPSILON or continuous_upper > command_upper + _EPSILON:
            raise ValueError(f"{name} quintic Bezier hull exceeds real command limits")
    if validate_schema:
        validate_config(dict(config))
    return {
        "schema_version": TIME_WARP_SCHEMA_VERSION,
        "duration_s": plan.duration_s,
        "knot_count": len(times),
        "segment_duration_min_s": float(np.min(durations)),
        "segment_duration_max_s": float(np.max(durations)),
        "relative_lower_rad": {
            name: float(relative_lower[index])
            for index, name in enumerate(ACTIVE_ACTUATORS)
        },
        "relative_upper_rad": {
            name: float(relative_upper[index])
            for index, name in enumerate(ACTIVE_ACTUATORS)
        },
    }


def _time_warp_planner_id(
    *,
    parent_config_sha256: str,
    parent_plan_id: str,
    coefficients: Sequence[float],
    category: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": TIME_WARP_SCHEMA_VERSION,
            "kind": "v14_contact_preserving_two_gaussian_time_warp",
            "parent_config_sha256": parent_config_sha256,
            "parent_plan_id": parent_plan_id,
            "coefficients": [float(value) for value in coefficients],
            "category": category,
            "duration_bounds_s": [
                MIN_SEGMENT_DURATION_S,
                MAX_SEGMENT_DURATION_S,
            ],
        }
    )


def _time_warp_controller_id(config: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": TIME_WARP_SCHEMA_VERSION,
            "kind": "v14_contact_preserving_time_warp_controller",
            "grasp_object_pair_id": config["grasp_object_pair_id"],
            "plan_id": config["manipulation_plan"]["plan_id"],
            "target_id": config["contact_force_targets_n"]["target_id"],
            "feedback_id": config["contact_feedback"]["feedback_id"],
            "planner_id": config["planner_id"],
            "contact_preload_targets_rad": config["control"][
                "contact_preload_targets_rad"
            ],
        }
    )


def apply_time_warp_to_config(
    parent_config: Mapping[str, Any],
    *,
    a1: float,
    a2: float,
    terminal_offset_rad: Mapping[str, float] | None = None,
    preload_offset_rad: Mapping[str, float] | None = None,
    category: str = "warp_only",
    limits: JointRefinementLimits | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Install one time warp and optional small terminal/preload perturbation."""

    if category not in {"warp_only", "warp_terminal", "warp_preload"}:
        raise ValueError("unknown time-warp candidate category")
    if int(parent_config.get("schema_version", 0)) != 14:
        raise ValueError("time-warp parent must use schema v14")
    config = copy.deepcopy(dict(parent_config))
    old_plan = ManipulationPlanParameters.from_config(config["manipulation_plan"])
    resolved_limits = (
        resolve_joint_refinement_limits(config) if limits is None else limits
    )
    terminal_offsets = {
        name: 0.0 if terminal_offset_rad is None else float(terminal_offset_rad[name])
        for name in ACTIVE_ACTUATORS
    }
    preload_offsets = {
        name: 0.0 if preload_offset_rad is None else float(preload_offset_rad[name])
        for name in ACTIVE_ACTUATORS
    }
    if terminal_offset_rad is not None and set(terminal_offset_rad) != set(ACTIVE_ACTUATORS):
        raise ValueError("terminal_offset_rad must name all active actuators")
    if preload_offset_rad is not None and set(preload_offset_rad) != set(ACTIVE_ACTUATORS):
        raise ValueError("preload_offset_rad must name all active actuators")
    if not np.isfinite(tuple(terminal_offsets.values())).all() or not np.isfinite(
        tuple(preload_offsets.values())
    ).all():
        raise ValueError("time-warp perturbations must be finite")

    base_preload = contact_preload_targets(copy.deepcopy(config))
    precontact = precontact_targets(copy.deepcopy(config))
    preload = {
        name: float(
            np.clip(
                float(base_preload[name]) + preload_offsets[name],
                *resolved_limits.preload_target_rad[name],
            )
        )
        for name in ACTIVE_ACTUATORS
    }
    preload = _preserve_closing_rays(preload, base_preload, precontact)
    config["control"]["contact_preload_targets_rad"] = preload

    new_waypoints: dict[str, tuple[float, ...]] = {}
    terminal: dict[str, float] = {}
    exact_terminal = all(value == 0.0 for value in terminal_offsets.values())
    for name in ACTIVE_ACTUATORS:
        old_values = np.asarray(old_plan.actuator_waypoints_rad[name], dtype=np.float64)
        old_terminal = float(old_values[-1])
        if exact_terminal:
            # Preserve every parent float (including signed zero) exactly.  In
            # particular a zero warp must not acquire a different plan_id due
            # solely to ``-0.0`` being normalized to ``0.0``.
            new_terminal = old_terminal
            values = old_values.copy()
        else:
            lower, upper = resolved_limits.registered_plan_delta_rad[name]
            new_terminal = float(
                np.clip(old_terminal + terminal_offsets[name], lower, upper)
            )
            if abs(old_terminal) > 1e-12:
                values = old_values * (new_terminal / old_terminal)
            else:
                progress = np.asarray(old_plan.desired_cube_position_delta_m)[:, 2]
                if abs(float(progress[-1])) > 1e-12:
                    progress = progress / float(progress[-1])
                else:
                    progress = np.linspace(0.0, 1.0, KNOT_COUNT)
                values = progress * new_terminal
            values[0] = 0.0
            values[-1] = new_terminal
        new_waypoints[name] = tuple(float(value) for value in values)
        terminal[name] = new_terminal

    new_times = warp_knot_times(old_plan.knot_times_s, a1, a2)
    plan = ManipulationPlanParameters(
        schema_version=old_plan.schema_version,
        profile=old_plan.profile,
        duration_s=DURATION_S,
        knot_times_s=tuple(float(value) for value in new_times),
        actuator_waypoints_rad=new_waypoints,
        desired_cube_position_delta_m=old_plan.desired_cube_position_delta_m,
        desired_cube_rotation_vector_rad=old_plan.desired_cube_rotation_vector_rad,
        max_knot_delta_rad=old_plan.max_knot_delta_rad,
        trust_region_backtracks=old_plan.trust_region_backtracks,
    )
    config["manipulation_plan"] = plan.as_config()
    config["control"]["manipulation_delta_rad"] = terminal
    config["object_config_id"] = v14_object_config_id(config)
    config["grasp_pose_id"] = v14_grasp_pose_id(config)
    config["grasp_object_pair_id"] = v14_grasp_object_pair_id(config)
    parent_sha = canonical_sha256(parent_config)
    config["planner_id"] = _time_warp_planner_id(
        parent_config_sha256=parent_sha,
        parent_plan_id=old_plan.plan_id,
        coefficients=(a1, a2),
        category=category,
    )
    config["controller_id"] = _time_warp_controller_id(config)
    # Continuous-time safety is never optional.  ``validate`` only controls
    # the comparatively expensive full repository schema validation.
    validate_time_warped_config(
        config, limits=resolved_limits, validate_schema=validate
    )
    return config


def _distribute(total: int, count: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, count)
    return tuple(quotient + int(index < remainder) for index in range(count))


def _category_allocations(parent_count: int) -> tuple[dict[str, int], ...]:
    if not 1 <= parent_count <= DEFAULT_MAX_PARENTS:
        raise ValueError("time-warp rescue requires one to four parents")
    global_counts = {
        "warp_only": DEFAULT_TOTAL_CANDIDATES // 2,
        "warp_terminal": DEFAULT_TOTAL_CANDIDATES // 4,
        "warp_preload": DEFAULT_TOTAL_CANDIDATES // 4,
    }
    distributed = {
        category: _distribute(total, parent_count)
        for category, total in global_counts.items()
    }
    return tuple(
        {category: distributed[category][rank] for category in global_counts}
        for rank in range(parent_count)
    )


def _seed_for(*values: Any) -> int:
    digest = canonical_sha256(values)
    words = np.frombuffer(bytes.fromhex(digest[:32]), dtype="<u4")
    return int(np.random.SeedSequence([int(value) for value in words]).generate_state(1)[0])


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        order = rng.permutation(count)
        values[:, dimension] = (order + rng.random(count)) / float(count)
    return 2.0 * values - 1.0


def _candidate_id(identity_sha256: str) -> int:
    return 14 * 10**15 + int(identity_sha256[:12], 16) % 10**14


def _materialize_with_backoff(
    parent_config: Mapping[str, Any],
    *,
    coefficients: np.ndarray,
    terminal_offset: dict[str, float],
    preload_offset: dict[str, float],
    category: str,
    limits: JointRefinementLimits,
    validate_configs: bool,
) -> _MaterializedTimeWarp:
    # Continuous-spline headroom can be narrower than knot headroom.  Shrink
    # deterministically toward the exact authenticated parent, never clip an
    # interior Bezier control point silently.

    parent_terminal = {
        name: float(parent_config["control"]["manipulation_delta_rad"][name])
        for name in ACTIVE_ACTUATORS
    }
    parent_preload = contact_preload_targets(copy.deepcopy(dict(parent_config)))

    def materialize(
        coefficient_scale: float,
        terminal_scale: float,
        preload_scale: float,
    ) -> dict[str, Any] | None:
        expected_terminal = {
            name: terminal_offset[name] * terminal_scale
            for name in ACTIVE_ACTUATORS
        }
        expected_preload = {
            name: preload_offset[name] * preload_scale
            for name in ACTIVE_ACTUATORS
        }
        try:
            config = apply_time_warp_to_config(
                parent_config,
                a1=float(coefficients[0] * coefficient_scale),
                a2=float(coefficients[1] * coefficient_scale),
                terminal_offset_rad=expected_terminal,
                preload_offset_rad=expected_preload,
                category=category,
                limits=limits,
                validate=validate_configs,
            )
        except ValueError:
            return None

        # ``apply_time_warp_to_config`` also serves interactive callers and
        # therefore clips to physical limits and may restore a closing ray.
        # Search jobs must be stricter: reject any such silent change so the
        # persisted scale and offset evidence describes the generated config
        # exactly.
        actual_terminal = {
            name: float(config["control"]["manipulation_delta_rad"][name])
            - parent_terminal[name]
            for name in ACTIVE_ACTUATORS
        }
        actual_preload = {
            name: float(config["control"]["contact_preload_targets_rad"][name])
            - float(parent_preload[name])
            for name in ACTIVE_ACTUATORS
        }
        if any(
            not math.isclose(
                actual_terminal[name],
                expected_terminal[name],
                rel_tol=0.0,
                abs_tol=2e-12,
            )
            for name in ACTIVE_ACTUATORS
        ) or any(
            not math.isclose(
                actual_preload[name],
                expected_preload[name],
                rel_tol=0.0,
                abs_tol=2e-12,
            )
            for name in ACTIVE_ACTUATORS
        ):
            return None
        return config

    backoff_scales = tuple(0.5**backoff for backoff in range(17))

    # Prefer the original joint backoff: it retains the requested direction
    # in the combined coefficient/terminal/preload space.
    for backoff, scale in enumerate(backoff_scales):
        config = materialize(scale, scale, scale)
        if config is not None:
            return _MaterializedTimeWarp(
                config=config,
                coefficient_scale=scale,
                terminal_scale=scale if category == "warp_terminal" else 0.0,
                preload_scale=scale if category == "warp_preload" else 0.0,
                fallback_reason="none" if backoff == 0 else "joint_backoff",
            )

    # At a boundary, shrinking every component together can fail forever:
    # e.g. a parent already has a 0.04-rad adjacent waypoint step, so every
    # positive terminal scale violates the same hard limit.  Find the largest
    # independently safe time warp first, then try the optional perturbation.
    coefficient_scale = 0.0
    coefficient_config: dict[str, Any] | None = None
    for scale in (*backoff_scales, 0.0):
        candidate = materialize(scale, 0.0, 0.0)
        if candidate is not None:
            coefficient_scale = scale
            coefficient_config = candidate
            break
    if coefficient_config is None:
        raise ValueError(
            "authenticated time-warp parent is not safe even as the exact parent"
        )

    if category == "warp_terminal":
        for scale in backoff_scales:
            candidate = materialize(coefficient_scale, scale, 0.0)
            if candidate is not None:
                return _MaterializedTimeWarp(
                    config=candidate,
                    coefficient_scale=coefficient_scale,
                    terminal_scale=scale,
                    preload_scale=0.0,
                    fallback_reason=(
                        "independent_terminal_backoff"
                        if coefficient_scale == 1.0
                        else "coefficient_and_independent_terminal_backoff"
                    ),
                )
        return _MaterializedTimeWarp(
            config=coefficient_config,
            coefficient_scale=coefficient_scale,
            terminal_scale=0.0,
            preload_scale=0.0,
            fallback_reason="terminal_offset_zero_no_safe_headroom",
        )

    if category == "warp_preload":
        for scale in backoff_scales:
            candidate = materialize(coefficient_scale, 0.0, scale)
            if candidate is not None:
                return _MaterializedTimeWarp(
                    config=candidate,
                    coefficient_scale=coefficient_scale,
                    terminal_scale=0.0,
                    preload_scale=scale,
                    fallback_reason=(
                        "independent_preload_backoff"
                        if coefficient_scale == 1.0
                        else "coefficient_and_independent_preload_backoff"
                    ),
                )
        return _MaterializedTimeWarp(
            config=coefficient_config,
            coefficient_scale=coefficient_scale,
            terminal_scale=0.0,
            preload_scale=0.0,
            fallback_reason="preload_offset_zero_no_safe_headroom",
        )

    # warp_only reaches here only when the independent coefficient search had
    # to fall all the way back to the parent.  It remains a correctly labelled
    # exact-parent anchor for this deterministic stratum.
    return _MaterializedTimeWarp(
        config=coefficient_config,
        coefficient_scale=coefficient_scale,
        terminal_scale=0.0,
        preload_scale=0.0,
        fallback_reason=(
            "coefficient_backoff" if coefficient_scale > 0.0 else "exact_parent"
        ),
    )


LimitsResolver = Callable[[Mapping[str, Any]], JointRefinementLimits]


def build_contact_preserving_time_warp_jobs(
    parent_records: Sequence[Mapping[str, Any]],
    *,
    budget: TimeWarpBudget = TimeWarpBudget(),
    limits_resolver: LimitsResolver | None = None,
    validate_configs: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Build exactly 256 deterministic, worker-order-independent rescue jobs."""

    if not parent_records:
        raise ValueError("time-warp rescue requires at least one parent")
    ranked = rank_contact_constrained_candidates(parent_records)
    parents = tuple(ranked[: budget.max_parent_count])
    allocations = _category_allocations(len(parents))
    resolver = resolve_joint_refinement_limits if limits_resolver is None else limits_resolver
    jobs: list[dict[str, Any]] = []
    sequence_index = 0
    for parent_rank, (parent, allocation) in enumerate(zip(parents, allocations)):
        if "candidate_id" not in parent or not isinstance(parent.get("config"), Mapping):
            raise ValueError("each time-warp parent requires candidate_id and config")
        parent_id = int(parent["candidate_id"])
        parent_config = copy.deepcopy(dict(parent["config"]))
        if int(parent_config.get("schema_version", 0)) != 14:
            raise ValueError("every time-warp parent config must use schema v14")
        parent_sha = canonical_sha256(parent_config)
        limits = resolver(parent_config)
        for category in ("warp_only", "warp_terminal", "warp_preload"):
            count = allocation[category]
            seed = _seed_for(budget.seed, parent_id, parent_sha, category)
            units = _latin_hypercube(count, 2 + len(ACTIVE_ACTUATORS), seed)
            if category == "warp_only":
                units[0] = 0.0
            for category_index in range(count):
                coefficients = units[category_index, :2] * COEFFICIENT_BOUNDS[1]
                terminal_offset = {name: 0.0 for name in ACTIVE_ACTUATORS}
                preload_offset = {name: 0.0 for name in ACTIVE_ACTUATORS}
                if category == "warp_terminal":
                    terminal_offset = {
                        name: float(units[category_index, 2 + index])
                        * budget.terminal_radius_rad
                        for index, name in enumerate(ACTIVE_ACTUATORS)
                    }
                elif category == "warp_preload":
                    preload_offset = {
                        name: float(units[category_index, 2 + index])
                        * budget.preload_radius_rad
                        for index, name in enumerate(ACTIVE_ACTUATORS)
                    }
                materialized = _materialize_with_backoff(
                    parent_config,
                    coefficients=coefficients,
                    terminal_offset=terminal_offset,
                    preload_offset=preload_offset,
                    category=category,
                    limits=limits,
                    validate_configs=validate_configs,
                )
                config = materialized.config
                applied_coefficients = [
                    float(value * materialized.coefficient_scale)
                    for value in coefficients
                ]
                applied_terminal = {
                    name: float(
                        config["control"]["manipulation_delta_rad"][name]
                        - parent_config["control"]["manipulation_delta_rad"][name]
                    )
                    for name in ACTIVE_ACTUATORS
                }
                parent_preload = contact_preload_targets(copy.deepcopy(parent_config))
                applied_preload = {
                    name: float(
                        config["control"]["contact_preload_targets_rad"][name]
                        - parent_preload[name]
                    )
                    for name in ACTIVE_ACTUATORS
                }
                relevant_scales = [materialized.coefficient_scale]
                if category == "warp_terminal":
                    relevant_scales.append(materialized.terminal_scale)
                elif category == "warp_preload":
                    relevant_scales.append(materialized.preload_scale)
                legacy_joint_scale = (
                    relevant_scales[0]
                    if all(
                        value == relevant_scales[0] for value in relevant_scales[1:]
                    )
                    else None
                )
                identity = {
                    "schema_version": TIME_WARP_SCHEMA_VERSION,
                    "kind": "v14_contact_preserving_time_warp_candidate",
                    "seed": budget.seed,
                    "parent_candidate_id": parent_id,
                    "parent_config_sha256": parent_sha,
                    "parent_rank": parent_rank,
                    "category": category,
                    "category_index": category_index,
                    "requested_coefficients": coefficients.tolist(),
                    "requested_terminal_offset_rad": copy.deepcopy(terminal_offset),
                    "requested_preload_offset_rad": copy.deepcopy(preload_offset),
                    "applied_coefficients": applied_coefficients,
                    # Kept for readers of schema-version 1 jobs.  ``None`` is
                    # deliberate when independent fallback means no single
                    # truthful joint scale exists.
                    "applied_backoff_scale": legacy_joint_scale,
                    "applied_scales": {
                        "coefficient": materialized.coefficient_scale,
                        "terminal": materialized.terminal_scale,
                        "preload": materialized.preload_scale,
                    },
                    "fallback_reason": materialized.fallback_reason,
                    "parent_terminal_rad": {
                        name: float(
                            parent_config["control"]["manipulation_delta_rad"][name]
                        )
                        for name in ACTIVE_ACTUATORS
                    },
                    "parent_preload_targets_rad": {
                        name: float(parent_preload[name])
                        for name in ACTIVE_ACTUATORS
                    },
                    "terminal_offset_rad": applied_terminal,
                    "preload_offset_rad": applied_preload,
                    "plan_id": config["manipulation_plan"]["plan_id"],
                    "controller_id": config["controller_id"],
                }
                digest = canonical_sha256(identity)
                candidate_id = _candidate_id(digest)
                metadata = {
                    **identity,
                    "candidate_id": candidate_id,
                    "candidate_sha256": digest,
                    "job_sequence_index": sequence_index,
                    "parent_allocation": copy.deepcopy(allocation),
                    "full_reset_required": True,
                }
                config.setdefault("candidate_metadata", {})[
                    "v14_contact_preserving_time_warp"
                ] = copy.deepcopy(metadata)
                job = {
                    "time_warp_job_schema_version": TIME_WARP_SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "candidate_sha256": digest,
                    "config_sha256": canonical_sha256(config),
                    "parent_candidate_id": parent_id,
                    "parent_rank": parent_rank,
                    "local_index": category_index,
                    "job_sequence_index": sequence_index,
                    "config": config,
                    "job_metadata": metadata,
                }
                jobs.append(job)
                sequence_index += 1
    if len(jobs) != budget.total_candidate_count:
        raise RuntimeError("time-warp job builder did not produce exactly 256 jobs")
    for label, values in (
        ("candidate IDs", [job["candidate_id"] for job in jobs]),
        ("candidate hashes", [job["candidate_sha256"] for job in jobs]),
        ("config hashes", [job["config_sha256"] for job in jobs]),
    ):
        if len(set(values)) != budget.total_candidate_count:
            raise RuntimeError(f"time-warp job builder did not produce 256 unique {label}")
    return tuple(jobs)


def time_warp_campaign_manifest(
    parent_records: Sequence[Mapping[str, Any]],
    budget: TimeWarpBudget = TimeWarpBudget(),
) -> dict[str, Any]:
    """Describe the exact selected parents and fixed candidate allocation."""

    if not parent_records:
        raise ValueError("time-warp rescue requires at least one parent")
    parents = tuple(
        rank_contact_constrained_candidates(parent_records)[: budget.max_parent_count]
    )
    allocations = _category_allocations(len(parents))
    payload = {
        "time_warp_campaign_manifest_schema_version": TIME_WARP_SCHEMA_VERSION,
        "kind": "v14_contact_preserving_time_warp_rescue",
        "budget": budget.as_mapping(),
        "received_parent_count": len(parent_records),
        "selected_parent_count": len(parents),
        "parent_selection": "contact_first_then_stable_candidate_id_top4",
        "fewer_parent_allocation": (
            "global_128_64_64_category_quotas_evenly_distributed_by_parent_rank"
        ),
        "parents": [
            {
                "parent_rank": rank,
                "parent_candidate_id": int(parent["candidate_id"]),
                "parent_config_sha256": canonical_sha256(parent["config"]),
                "allocation": copy.deepcopy(allocations[rank]),
                "total": int(sum(allocations[rank].values())),
            }
            for rank, parent in enumerate(parents)
        ],
    }
    if sum(value["total"] for value in payload["parents"]) != 256:
        raise RuntimeError("time-warp manifest allocation is not exactly 256")
    return {**payload, "manifest_id": canonical_sha256(payload)}


def authenticate_time_warp_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate a generated job without trusting its persisted summary."""

    expected = {
        "time_warp_job_schema_version",
        "candidate_id",
        "candidate_sha256",
        "config_sha256",
        "parent_candidate_id",
        "parent_rank",
        "local_index",
        "job_sequence_index",
        "config",
        "job_metadata",
    }
    if set(job) != expected or int(job["time_warp_job_schema_version"]) != 1:
        raise ValueError("time-warp job contains unexpected or missing fields")
    config = job["config"]
    metadata = job["job_metadata"]
    if not isinstance(config, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("time-warp job config and metadata must be mappings")
    persisted = config.get("candidate_metadata", {}).get(
        "v14_contact_preserving_time_warp"
    )
    if persisted != metadata:
        raise ValueError("time-warp job metadata does not match its config")
    identity_fields = {
        "schema_version",
        "kind",
        "seed",
        "parent_candidate_id",
        "parent_config_sha256",
        "parent_rank",
        "category",
        "category_index",
        "requested_coefficients",
        "requested_terminal_offset_rad",
        "requested_preload_offset_rad",
        "applied_coefficients",
        "applied_backoff_scale",
        "applied_scales",
        "fallback_reason",
        "parent_terminal_rad",
        "parent_preload_targets_rad",
        "terminal_offset_rad",
        "preload_offset_rad",
        "plan_id",
        "controller_id",
    }
    if not identity_fields.issubset(metadata):
        raise ValueError("time-warp job identity metadata is incomplete")
    identity = {name: copy.deepcopy(metadata[name]) for name in identity_fields}
    digest = canonical_sha256(identity)
    candidate_id = _candidate_id(digest)
    if digest != job["candidate_sha256"] or digest != metadata.get("candidate_sha256"):
        raise ValueError("time-warp candidate identity hash mismatch")
    if candidate_id != int(job["candidate_id"]) or candidate_id != int(
        metadata.get("candidate_id", -1)
    ):
        raise ValueError("time-warp candidate ID mismatch")
    for outer, inner in (
        ("parent_candidate_id", "parent_candidate_id"),
        ("parent_rank", "parent_rank"),
        ("local_index", "category_index"),
        ("job_sequence_index", "job_sequence_index"),
    ):
        if int(job[outer]) != int(metadata[inner]):
            raise ValueError(f"time-warp {outer} mismatch")
    if canonical_sha256(config) != job["config_sha256"]:
        raise ValueError("time-warp config hash mismatch")

    category = metadata["category"]
    if category not in {"warp_only", "warp_terminal", "warp_preload"}:
        raise ValueError("time-warp category is invalid")
    scales = metadata["applied_scales"]
    if not isinstance(scales, Mapping) or set(scales) != {
        "coefficient",
        "terminal",
        "preload",
    }:
        raise ValueError("time-warp applied scales are invalid")
    normalized_scales = {name: float(value) for name, value in scales.items()}
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in normalized_scales.values()
    ):
        raise ValueError("time-warp applied scale lies outside [0, 1]")

    requested_coefficients = np.asarray(
        metadata["requested_coefficients"], dtype=np.float64
    )
    applied_coefficients = np.asarray(
        metadata["applied_coefficients"], dtype=np.float64
    )
    if requested_coefficients.shape != (2,) or applied_coefficients.shape != (2,):
        raise ValueError("time-warp coefficient metadata is invalid")
    if not np.allclose(
        applied_coefficients,
        requested_coefficients * normalized_scales["coefficient"],
        rtol=0.0,
        atol=2e-12,
    ):
        raise ValueError("time-warp applied coefficient scale is inaccurate")

    def verify_offsets(
        *,
        requested_key: str,
        applied_key: str,
        parent_key: str,
        config_values: Mapping[str, Any],
        scale_name: str,
    ) -> None:
        requested = metadata[requested_key]
        applied = metadata[applied_key]
        parent = metadata[parent_key]
        if not all(
            isinstance(value, Mapping) and set(value) == set(ACTIVE_ACTUATORS)
            for value in (requested, applied, parent)
        ):
            raise ValueError(f"time-warp {scale_name} offset metadata is invalid")
        scale = normalized_scales[scale_name]
        for name in ACTIVE_ACTUATORS:
            requested_value = float(requested[name])
            applied_value = float(applied[name])
            actual_value = float(config_values[name]) - float(parent[name])
            if not math.isclose(
                applied_value,
                requested_value * scale,
                rel_tol=0.0,
                abs_tol=2e-12,
            ) or not math.isclose(
                actual_value, applied_value, rel_tol=0.0, abs_tol=2e-12
            ):
                raise ValueError(
                    f"time-warp applied {scale_name} scale or offset is inaccurate"
                )

    verify_offsets(
        requested_key="requested_terminal_offset_rad",
        applied_key="terminal_offset_rad",
        parent_key="parent_terminal_rad",
        config_values=config["control"]["manipulation_delta_rad"],
        scale_name="terminal",
    )
    verify_offsets(
        requested_key="requested_preload_offset_rad",
        applied_key="preload_offset_rad",
        parent_key="parent_preload_targets_rad",
        config_values=config["control"]["contact_preload_targets_rad"],
        scale_name="preload",
    )

    if category != "warp_terminal" and normalized_scales["terminal"] != 0.0:
        raise ValueError("time-warp terminal scale is nonzero outside its category")
    if category != "warp_preload" and normalized_scales["preload"] != 0.0:
        raise ValueError("time-warp preload scale is nonzero outside its category")
    relevant_scales = [normalized_scales["coefficient"]]
    if category == "warp_terminal":
        relevant_scales.append(normalized_scales["terminal"])
    elif category == "warp_preload":
        relevant_scales.append(normalized_scales["preload"])
    expected_legacy_scale = (
        relevant_scales[0]
        if all(value == relevant_scales[0] for value in relevant_scales[1:])
        else None
    )
    if metadata["applied_backoff_scale"] != expected_legacy_scale:
        raise ValueError("legacy joint-backoff scale misrepresents component scales")
    fallback_reason = metadata["fallback_reason"]
    allowed_fallback_reasons = {
        "none",
        "joint_backoff",
        "coefficient_backoff",
        "independent_terminal_backoff",
        "coefficient_and_independent_terminal_backoff",
        "terminal_offset_zero_no_safe_headroom",
        "independent_preload_backoff",
        "coefficient_and_independent_preload_backoff",
        "preload_offset_zero_no_safe_headroom",
        "exact_parent",
    }
    if fallback_reason not in allowed_fallback_reasons:
        raise ValueError("time-warp fallback reason is invalid")
    if fallback_reason == "terminal_offset_zero_no_safe_headroom" and not (
        category == "warp_terminal" and normalized_scales["terminal"] == 0.0
    ):
        raise ValueError("terminal fallback reason disagrees with actual scale")
    if fallback_reason == "preload_offset_zero_no_safe_headroom" and not (
        category == "warp_preload" and normalized_scales["preload"] == 0.0
    ):
        raise ValueError("preload fallback reason disagrees with actual scale")
    plan = ManipulationPlanParameters.from_config(config["manipulation_plan"])
    if plan.plan_id != metadata["plan_id"]:
        raise ValueError("time-warp plan binding mismatch")
    if config.get("controller_id") != metadata["controller_id"] or config.get(
        "controller_id"
    ) != _time_warp_controller_id(config):
        raise ValueError("time-warp controller binding mismatch")
    return {
        "candidate_id": candidate_id,
        "candidate_sha256": digest,
        "config_sha256": job["config_sha256"],
        "plan_id": plan.plan_id,
        "controller_id": config["controller_id"],
    }


def stable_sort_time_warp_jobs(
    jobs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return deep-copied jobs in deterministic runner sequence order."""

    result = [copy.deepcopy(dict(job)) for job in jobs]
    result.sort(
        key=lambda job: (int(job["job_sequence_index"]), int(job["candidate_id"]))
    )
    if [int(job["job_sequence_index"]) for job in result] != list(range(len(result))):
        raise ValueError("time-warp jobs do not form one contiguous sequence")
    return tuple(result)


__all__ = [
    "COEFFICIENT_BOUNDS",
    "DEFAULT_TOTAL_CANDIDATES",
    "DURATION_S",
    "GAUSSIAN_CENTERS",
    "GAUSSIAN_SIGMA",
    "KNOT_COUNT",
    "MAX_SEGMENT_DURATION_S",
    "MIN_SEGMENT_DURATION_S",
    "SEGMENT_COUNT",
    "TimeWarpBudget",
    "actuator_bezier_hull",
    "apply_time_warp_to_config",
    "authenticate_time_warp_job",
    "build_contact_preserving_time_warp_jobs",
    "quintic_bezier_controls",
    "stable_sort_time_warp_jobs",
    "time_warp_campaign_manifest",
    "validate_time_warped_config",
    "warp_knot_times",
    "zero_mean_gaussian_bases",
]
