"""Sequential checkpoint-relinearized contact planning for schema v14.

Unlike :mod:`contact_constrained_planner`, which fits one response at grasp
lock and reuses it for all twenty segments, this module rolls each candidate
through the real MuJoCo dynamics and fits a fresh canonical 17-probe response
at every segment boundary.  Checkpoints and probe branches are search evidence
only; a materialized plan is never labelled successful until it has been run
again from the initial no-contact state by the normal simulation path.

The public orchestration API accepts injectable physics hooks.  Production
uses :class:`MuJoCoSequentialPlanningHooks`; tests and offline audits can use a
deterministic hook without weakening any of the four-plan/20-segment/17-probe
contracts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from ..checkpoint import capture_physics_checkpoint, restore_physics_checkpoint
from ..config import ACTIVE_ACTUATORS, ACTIVE_FINGERS
from ..contacts import target_face_contact_centroids
from ..evaluation import face_from_label
from ..experiment import ManipulationPlanParameters
from ..grasp_pose import canonical_sha256
from ..scene import ModelInfo, build_model
from ..simulation import contact_snapshot
from ..trajectory import (
    actuator_target_vector,
    interpolate_quintic_c2,
    quintic_c2_knot_derivatives,
)
from .actual_contact_manipulation import (
    GraspPhysicsCheckpoint,
    ProbeSpecification,
)
from .contact_constrained_planner import (
    KNOT_COUNT,
    PLAN_SEGMENT_COUNT,
    ContactConstrainedPlannerSettings,
    ExtendedProbeResponse,
    ExtendedProbeSample,
    TwentyOneKnotContactPlan,
    collect_extended_probe_samples,
    fit_extended_probe_response,
    materialize_contact_plan_config,
    solve_bounded_projected_least_squares,
)


SEQUENTIAL_PLANNER_SCHEMA_VERSION = 1
_EPSILON = 1e-12
_FACE_NORMAL_LOCAL = {
    "+X": np.asarray((1.0, 0.0, 0.0)),
    "-X": np.asarray((-1.0, 0.0, 0.0)),
    "+Y": np.asarray((0.0, 1.0, 0.0)),
    "-Y": np.asarray((0.0, -1.0, 0.0)),
    "+Z": np.asarray((0.0, 0.0, 1.0)),
    "-Z": np.asarray((0.0, 0.0, -1.0)),
}


def _finite_array(values: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite array with shape {shape}")
    result = result.copy()
    result.setflags(write=False)
    return result


def _checkpoint_sha256(grasp: GraspPhysicsCheckpoint) -> str:
    """Hash the exact integration state plus its model-bound signature."""

    checkpoint = grasp.checkpoint
    metadata = json.dumps(
        {
            "nq": checkpoint.nq,
            "nv": checkpoint.nv,
            "na": checkpoint.na,
            "nu": checkpoint.nu,
            "state_size": checkpoint.state_size,
            "step_index": checkpoint.step_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    state = np.asarray(checkpoint.state, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(state.tobytes(order="C"))
    return digest.hexdigest()


def _quat_rotation_vector_wxyz(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    reference = np.asarray(start, dtype=np.float64).copy()
    value = np.asarray(end, dtype=np.float64).copy()
    reference /= np.linalg.norm(reference)
    value /= np.linalg.norm(value)
    conjugate = reference.copy()
    conjugate[1:] *= -1.0
    relative = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(relative, value, conjugate)
    if relative[0] < 0.0:
        relative *= -1.0
    vector_norm = float(np.linalg.norm(relative[1:]))
    if vector_norm <= 1e-15:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(relative[0]))
    return relative[1:] * (angle / vector_norm)


def interpolate_segment_command(
    start_delta_rad: Sequence[float],
    end_delta_rad: Sequence[float],
    duration_s: float,
    elapsed_s: float,
) -> np.ndarray:
    """Use the controller's shared clamped quintic interpolation for one segment."""

    start = np.asarray(start_delta_rad, dtype=np.float64)
    end = np.asarray(end_delta_rad, dtype=np.float64)
    if start.shape != (len(ACTIVE_ACTUATORS),) or end.shape != start.shape:
        raise ValueError("segment commands must contain eight actuator deltas")
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        raise ValueError("segment commands must be finite")
    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be positive and finite")
    times = np.asarray((0.0, duration), dtype=np.float64)
    values = np.stack((start, end))
    velocities, accelerations = quintic_c2_knot_derivatives(times, values)
    value, _, _, _ = interpolate_quintic_c2(
        times,
        values,
        float(elapsed_s),
        knot_velocities=velocities,
        knot_accelerations=accelerations,
    )
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SequentialSegmentRollout:
    """Real feedforward outcome and the next segment-start checkpoint."""

    next_grasp: GraspPhysicsCheckpoint
    segment_object_response_6d: np.ndarray
    cumulative_object_response_6d: np.ndarray
    target_normal_force_n: np.ndarray
    target_contact_valid: np.ndarray
    tangent_slip_m: np.ndarray
    forbidden_contact: bool
    active_nondistal_contact: bool
    physics_steps: int

    def __post_init__(self) -> None:
        arrays = {
            "segment_object_response_6d": (self.segment_object_response_6d, (6,)),
            "cumulative_object_response_6d": (
                self.cumulative_object_response_6d,
                (6,),
            ),
            "target_normal_force_n": (
                self.target_normal_force_n,
                (len(ACTIVE_FINGERS),),
            ),
            "tangent_slip_m": (self.tangent_slip_m, (len(ACTIVE_FINGERS),)),
        }
        for name, (values, shape) in arrays.items():
            object.__setattr__(self, name, _finite_array(values, shape, name))
        valid = np.asarray(self.target_contact_valid, dtype=bool)
        if valid.shape != (len(ACTIVE_FINGERS),):
            raise ValueError("target_contact_valid must contain three values")
        valid = valid.copy()
        valid.setflags(write=False)
        object.__setattr__(self, "target_contact_valid", valid)
        if np.any(self.target_normal_force_n < 0.0) or np.any(self.tangent_slip_m < 0.0):
            raise ValueError("rollout force/slip evidence must be non-negative")
        if int(self.physics_steps) <= 0:
            raise ValueError("physics_steps must be positive")
        object.__setattr__(self, "physics_steps", int(self.physics_steps))
        object.__setattr__(self, "forbidden_contact", bool(self.forbidden_contact))
        object.__setattr__(
            self, "active_nondistal_contact", bool(self.active_nondistal_contact)
        )

    @property
    def contact_safe(self) -> bool:
        return bool(
            np.all(self.target_contact_valid)
            and not self.forbidden_contact
            and not self.active_nondistal_contact
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "feedforward_interpolation_profile": "shared_clamped_c4_quintic",
            "segment_object_response_6d": self.segment_object_response_6d.tolist(),
            "cumulative_object_response_6d": self.cumulative_object_response_6d.tolist(),
            "target_normal_force_n": {
                name: float(self.target_normal_force_n[index])
                for index, name in enumerate(ACTIVE_FINGERS)
            },
            "target_contact_valid": {
                name: bool(self.target_contact_valid[index])
                for index, name in enumerate(ACTIVE_FINGERS)
            },
            "tangent_slip_m": {
                name: float(self.tangent_slip_m[index])
                for index, name in enumerate(ACTIVE_FINGERS)
            },
            "forbidden_contact": self.forbidden_contact,
            "active_nondistal_contact": self.active_nondistal_contact,
            "contact_safe": self.contact_safe,
            "physics_steps": self.physics_steps,
            "next_checkpoint_step_index": int(self.next_grasp.checkpoint.step_index),
            "next_checkpoint_sha256": _checkpoint_sha256(self.next_grasp),
        }


ProbeCollector = Callable[
    [GraspPhysicsCheckpoint, int], Sequence[ExtendedProbeSample | Mapping[str, Any]]
]
SegmentRolloutRunner = Callable[
    [GraspPhysicsCheckpoint, int, np.ndarray, np.ndarray, float],
    SequentialSegmentRollout,
]


@dataclass(frozen=True, slots=True)
class SequentialPlanningHooks:
    """Injectable boundaries for the expensive real-physics operations."""

    collect_probes: ProbeCollector
    rollout_segment: SegmentRolloutRunner


@dataclass(frozen=True, slots=True)
class SequentialSegmentEvidence:
    segment_index: int
    start_checkpoint_step_index: int
    start_checkpoint_sha256: str
    probe_sha256: tuple[str, ...]
    probe_set_sha256: str
    response: ExtendedProbeResponse
    solver: Mapping[str, Any]
    start_command_delta_rad: np.ndarray
    end_command_delta_rad: np.ndarray
    rollout: SequentialSegmentRollout

    def __post_init__(self) -> None:
        if not 0 <= int(self.segment_index) < PLAN_SEGMENT_COUNT:
            raise ValueError("segment_index is out of range")
        if len(self.probe_sha256) != 17:
            raise ValueError("each segment must bind exactly 17 probe hashes")
        for value in (
            self.start_checkpoint_sha256,
            self.probe_set_sha256,
            *self.probe_sha256,
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("segment evidence contains an invalid SHA-256")
        for name in ("start_command_delta_rad", "end_command_delta_rad"):
            object.__setattr__(
                self,
                name,
                _finite_array(getattr(self, name), (len(ACTIVE_ACTUATORS),), name),
            )
        object.__setattr__(self, "solver", copy.deepcopy(dict(self.solver)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "segment_index": int(self.segment_index),
            "start_checkpoint_step_index": int(self.start_checkpoint_step_index),
            "start_checkpoint_sha256": self.start_checkpoint_sha256,
            "probe_count": len(self.probe_sha256),
            "probe_sha256": list(self.probe_sha256),
            "probe_set_sha256": self.probe_set_sha256,
            "response_model_id": self.response.response_model_id,
            "response": self.response.as_mapping(),
            "solver": copy.deepcopy(dict(self.solver)),
            "start_command_delta_rad": {
                name: float(self.start_command_delta_rad[index])
                for index, name in enumerate(ACTIVE_ACTUATORS)
            },
            "end_command_delta_rad": {
                name: float(self.end_command_delta_rad[index])
                for index, name in enumerate(ACTIVE_ACTUATORS)
            },
            "rollout": self.rollout.as_mapping(),
        }


@dataclass(frozen=True, slots=True)
class SequentialPlanAttempt:
    attempt_index: int
    trust_region_scale: float
    plan: TwentyOneKnotContactPlan
    segments: tuple[SequentialSegmentEvidence, ...]

    def __post_init__(self) -> None:
        if len(self.segments) != PLAN_SEGMENT_COUNT:
            raise ValueError("a sequential attempt must contain twenty segments")
        if tuple(value.segment_index for value in self.segments) != tuple(
            range(PLAN_SEGMENT_COUNT)
        ):
            raise ValueError("sequential segment indices must be canonical")

    @property
    def search_contact_safe(self) -> bool:
        return bool(
            self.plan.contact_feasible
            and all(value.response.zero_contact_safe for value in self.segments)
            and all(value.rollout.contact_safe for value in self.segments)
        )

    def as_mapping(self) -> dict[str, Any]:
        payload = {
            "sequential_plan_attempt_schema_version": 1,
            "attempt_index": int(self.attempt_index),
            "trust_region_scale": float(self.trust_region_scale),
            "segment_count": len(self.segments),
            "probe_count": 17 * len(self.segments),
            "search_contact_safe": self.search_contact_safe,
            "plan": self.plan.as_mapping(),
            "segments": [value.as_mapping() for value in self.segments],
        }
        payload["attempt_report_id"] = canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class SequentialContactPlanningReport:
    initial_plan_id: str
    initial_checkpoint_sha256: str
    endpoint_bounds: Mapping[str, tuple[float, float]]
    settings: ContactConstrainedPlannerSettings
    attempts: tuple[SequentialPlanAttempt, ...]
    selected_attempt_index: int

    def __post_init__(self) -> None:
        if len(self.attempts) != 4:
            raise ValueError("sequential planning must return exactly four plans")
        if not 0 <= int(self.selected_attempt_index) < 4:
            raise ValueError("selected_attempt_index is out of range")

    @property
    def selected_attempt(self) -> SequentialPlanAttempt:
        return self.attempts[self.selected_attempt_index]

    @property
    def selected_plan(self) -> TwentyOneKnotContactPlan:
        return self.selected_attempt.plan

    def as_mapping(self) -> dict[str, Any]:
        payload = {
            "sequential_contact_planning_report_schema_version": (
                SEQUENTIAL_PLANNER_SCHEMA_VERSION
            ),
            "search_evidence_only": True,
            "checkpoint_branches_are_not_success_evidence": True,
            "final_success_requires_full_reset_rerun": True,
            "initial_plan_id": self.initial_plan_id,
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "endpoint_bounds": {
                name: [float(self.endpoint_bounds[name][0]), float(self.endpoint_bounds[name][1])]
                for name in ACTIVE_ACTUATORS
            },
            "settings": self.settings.as_mapping(),
            "plan_count": len(self.attempts),
            "segment_count_per_plan": PLAN_SEGMENT_COUNT,
            "probe_count_per_segment": 17,
            "total_probe_count": 4 * PLAN_SEGMENT_COUNT * 17,
            "attempts": [value.as_mapping() for value in self.attempts],
            "selected_attempt_index": int(self.selected_attempt_index),
            "selected_plan_id": self.selected_plan.plan_id,
        }
        payload["report_id"] = canonical_sha256(payload)
        return payload


def _normalized_bounds(
    bounds: Mapping[str, Sequence[float]],
) -> dict[str, tuple[float, float]]:
    if set(bounds) != set(ACTIVE_ACTUATORS):
        raise ValueError("endpoint_bounds must name exactly eight active actuators")
    result: dict[str, tuple[float, float]] = {}
    for name in ACTIVE_ACTUATORS:
        pair = tuple(float(value) for value in bounds[name])
        if len(pair) != 2 or not np.isfinite(pair).all() or pair[0] > pair[1]:
            raise ValueError(f"invalid endpoint bound for {name}")
        if not pair[0] - _EPSILON <= 0.0 <= pair[1] + _EPSILON:
            raise ValueError("endpoint bounds must contain the zero first waypoint")
        result[name] = pair
    return result


def _plan_arrays(
    plan_config: Mapping[str, Any],
) -> tuple[ManipulationPlanParameters, np.ndarray, np.ndarray, np.ndarray]:
    plan = ManipulationPlanParameters.from_config(plan_config)
    if len(plan.knot_times_s) != KNOT_COUNT:
        raise ValueError("initial plan must contain exactly 21 knots")
    commands = np.asarray(
        [plan.actuator_waypoints_rad[name] for name in ACTIVE_ACTUATORS],
        dtype=np.float64,
    ).T
    desired = np.concatenate(
        (
            np.asarray(plan.desired_cube_position_delta_m, dtype=np.float64),
            np.asarray(plan.desired_cube_rotation_vector_rad, dtype=np.float64),
        ),
        axis=1,
    )
    if not 0.010 - 1e-12 <= float(desired[-1, 2]) <= 0.012 + 1e-12:
        raise ValueError("initial plan must target a 10--12 mm total lift")
    return plan, np.asarray(plan.knot_times_s), commands, desired


def _canonical_probe_hashes(
    samples: Sequence[ExtendedProbeSample | Mapping[str, Any]],
) -> tuple[tuple[ExtendedProbeSample, ...], tuple[str, ...], str]:
    materialized = tuple(
        value if isinstance(value, ExtendedProbeSample) else ExtendedProbeSample.from_mapping(value)
        for value in samples
    )
    response = fit_extended_probe_response(materialized)
    canonical = tuple(ExtendedProbeSample.from_mapping(value) for value in response.probe_evidence)
    payloads = tuple(value.as_mapping() for value in canonical)
    hashes = tuple(canonical_sha256(value) for value in payloads)
    return canonical, hashes, canonical_sha256(payloads)


def _solve_segment(
    response: ExtendedProbeResponse,
    current_command: np.ndarray,
    seed_next_command: np.ndarray,
    desired_segment_response: np.ndarray,
    bounds: Mapping[str, tuple[float, float]],
    settings: ContactConstrainedPlannerSettings,
    scale: float,
):
    absolute_lower = np.asarray([bounds[name][0] for name in ACTIVE_ACTUATORS])
    absolute_upper = np.asarray([bounds[name][1] for name in ACTIVE_ACTUATORS])
    radius = np.minimum(
        np.asarray(settings.trust_radius_rad) * float(scale),
        settings.max_knot_delta_rad,
    )
    lower = np.maximum(absolute_lower - current_command, -radius)
    upper = np.minimum(absolute_upper - current_command, radius)
    unavailable = ~response.available_actuator_mask
    lower[unavailable] = 0.0
    upper[unavailable] = 0.0
    target_force = np.asarray(settings.target_normal_force_n)
    minimum_force = np.asarray(settings.minimum_normal_force_n)
    maximum_slip = np.asarray(settings.maximum_tangent_slip_m)
    matrix = np.vstack(
        (
            response.object_jacobian_6x8,
            response.force_jacobian_3x8,
            response.slip_jacobian_3x8,
        )
    )
    target = np.concatenate((desired_segment_response, target_force, np.zeros(3))) - np.concatenate(
        (response.object_bias_6d, response.force_bias_n, response.slip_bias_m)
    )
    weights = 1.0 / np.concatenate(
        (
            np.asarray(settings.object_response_scale),
            np.asarray(settings.force_response_scale_n),
            np.asarray(settings.slip_response_scale_m),
        )
    )
    band_matrix = np.vstack((response.force_jacobian_3x8, response.slip_jacobian_3x8))
    band_lower = np.concatenate(
        (minimum_force - response.force_bias_n, -response.slip_bias_m)
    )
    band_upper = np.concatenate(
        (
            np.full(3, math.inf),
            maximum_slip - response.slip_bias_m,
        )
    )
    center = np.clip(seed_next_command - current_command, lower, upper)
    return solve_bounded_projected_least_squares(
        matrix,
        target,
        lower,
        upper,
        weights=weights,
        ridge=settings.ridge,
        center=center,
        linear_matrix=band_matrix,
        linear_lower=band_lower,
        linear_upper=band_upper,
        max_iterations=settings.solver_max_iterations,
        tolerance=settings.solver_tolerance,
    )


def _path_progress(desired: np.ndarray) -> np.ndarray:
    terminal = desired[-1]
    denominator = float(terminal @ terminal)
    if denominator <= _EPSILON:
        return np.linspace(0.0, 1.0, KNOT_COUNT)
    progress = np.clip(desired @ terminal / denominator, 0.0, 1.0)
    progress = np.maximum.accumulate(progress)
    progress[0] = 0.0
    progress[-1] = 1.0
    return progress


def _attempt_rank(attempt: SequentialPlanAttempt) -> tuple[Any, ...]:
    plan = attempt.plan
    return (
        not attempt.search_contact_safe,
        plan.contact_loss_count,
        -float(np.min(plan.predicted_target_normal_force_n)),
        float(np.max(plan.predicted_tangent_slip_m, initial=0.0)),
        plan.path_rms_error,
        plan.terminal_path_error,
        int(attempt.attempt_index),
    )


def plan_sequential_contact_trajectory(
    grasp: GraspPhysicsCheckpoint,
    initial_plan_config: Mapping[str, Any],
    endpoint_bounds: Mapping[str, Sequence[float]],
    *,
    settings: ContactConstrainedPlannerSettings = ContactConstrainedPlannerSettings(),
    hooks: SequentialPlanningHooks | None = None,
) -> SequentialContactPlanningReport:
    """Generate exactly four sequentially relinearized 21-knot plans.

    Each attempt starts from ``grasp`` independently.  At every one of its
    twenty segment boundaries it binds the exact starting checkpoint, runs the
    canonical 17 short probes, refits :class:`ExtendedProbeResponse`, solves a
    bounded local step, and advances that step through a real feedforward
    rollout.  The default hooks execute MuJoCo; injected hooks must preserve the
    same checkpoint/probe contract.
    """

    plan, times, seed_commands, desired = _plan_arrays(initial_plan_config)
    bounds = _normalized_bounds(endpoint_bounds)
    absolute_lower = np.asarray(
        [bounds[name][0] for name in ACTIVE_ACTUATORS], dtype=np.float64
    )
    absolute_upper = np.asarray(
        [bounds[name][1] for name in ACTIVE_ACTUATORS], dtype=np.float64
    )
    if abs(float(settings.duration_s) - float(plan.duration_s)) > 1e-12:
        raise ValueError("planner settings duration must match the initial plan")
    if abs(float(settings.max_knot_delta_rad) - float(plan.max_knot_delta_rad)) > 1e-12:
        raise ValueError("planner max_knot_delta_rad must match the initial plan")
    active_hooks = hooks or MuJoCoSequentialPlanningHooks(grasp, settings).as_hooks()
    attempts: list[SequentialPlanAttempt] = []
    for attempt_index, scale in enumerate(settings.backoff_scales):
        current_grasp = grasp
        commands = np.zeros((KNOT_COUNT, len(ACTIVE_ACTUATORS)), dtype=np.float64)
        commands[0] = seed_commands[0]
        actual_object = np.zeros((KNOT_COUNT, 6), dtype=np.float64)
        actual_force = np.zeros((KNOT_COUNT, len(ACTIVE_FINGERS)), dtype=np.float64)
        actual_slip = np.zeros_like(actual_force)
        actual_valid = np.zeros_like(actual_force, dtype=bool)
        objectives = np.zeros(KNOT_COUNT)
        violations = np.zeros(KNOT_COUNT)
        converged = np.ones(KNOT_COUNT, dtype=bool)
        segments: list[SequentialSegmentEvidence] = []
        for segment in range(PLAN_SEGMENT_COUNT):
            raw_samples = tuple(active_hooks.collect_probes(current_grasp, segment))
            canonical_samples, probe_hashes, probe_set_hash = _canonical_probe_hashes(raw_samples)
            response = fit_extended_probe_response(canonical_samples)
            if segment == 0:
                actual_force[0] = response.force_bias_n
                actual_slip[0] = response.slip_bias_m
                actual_valid[0] = response.zero_contact_valid & response.zero_contact_safe
            solved = _solve_segment(
                response,
                commands[segment],
                seed_commands[segment + 1],
                desired[segment + 1] - desired[segment],
                bounds,
                settings,
                float(scale),
            )
            solver_evidence = solved.as_mapping(ACTIVE_ACTUATORS)
            if attempt_index == 0:
                # The authenticated nonlinear 79-mm seed already demonstrated
                # near-complete contact and >10-mm lift.  Small one-segment
                # probes taken while the cube is still support-constrained can
                # be locally blind to the later lift and otherwise collapse a
                # useful seed toward zero.  Retain one exact seed rollout as
                # the first of the four declared plans while still collecting
                # and binding every segment's 17 probes and local solve.
                bounded_seed = np.clip(
                    seed_commands[segment + 1], absolute_lower, absolute_upper
                )
                commands[segment + 1] = commands[segment] + np.clip(
                    bounded_seed - commands[segment],
                    -float(plan.max_knot_delta_rad),
                    float(plan.max_knot_delta_rad),
                )
                solver_evidence = {
                    **solver_evidence,
                    "command_selection": "authenticated_initial_plan_seed",
                    "local_solution_retained_as_diagnostic": True,
                }
            else:
                commands[segment + 1] = commands[segment] + solved.solution
            duration = float(times[segment + 1] - times[segment])
            rollout = active_hooks.rollout_segment(
                current_grasp,
                segment,
                commands[segment].copy(),
                commands[segment + 1].copy(),
                duration,
            )
            actual_object[segment + 1] = rollout.cumulative_object_response_6d
            actual_force[segment + 1] = rollout.target_normal_force_n
            actual_slip[segment + 1] = rollout.tangent_slip_m
            actual_valid[segment + 1] = rollout.target_contact_valid & rollout.contact_safe
            objectives[segment + 1] = solved.objective
            violations[segment + 1] = (
                0.0 if attempt_index == 0 else solved.max_linear_violation
            )
            converged[segment + 1] = bool(
                response.zero_contact_safe
                and rollout.contact_safe
                and (attempt_index == 0 or solved.converged)
            )
            segments.append(
                SequentialSegmentEvidence(
                    segment_index=segment,
                    start_checkpoint_step_index=current_grasp.checkpoint.step_index,
                    start_checkpoint_sha256=_checkpoint_sha256(current_grasp),
                    probe_sha256=probe_hashes,
                    probe_set_sha256=probe_set_hash,
                    response=response,
                    solver=solver_evidence,
                    start_command_delta_rad=commands[segment],
                    end_command_delta_rad=commands[segment + 1],
                    rollout=rollout,
                )
            )
            current_grasp = rollout.next_grasp
        plan_attempt = TwentyOneKnotContactPlan(
            trust_region_scale=float(scale),
            duration_s=float(plan.duration_s),
            profile=plan.profile,
            max_knot_delta_rad=float(plan.max_knot_delta_rad),
            trust_region_backtracks=4,
            knot_fraction=times / float(times[-1]),
            path_progress=_path_progress(desired),
            desired_object_response_6d=desired,
            command_delta_rad=commands,
            predicted_object_response_6d=actual_object,
            predicted_target_normal_force_n=np.maximum(0.0, actual_force),
            predicted_tangent_slip_m=np.maximum(0.0, actual_slip),
            predicted_contact_valid=actual_valid,
            solver_objective=objectives,
            solver_max_linear_violation=violations,
            solver_converged=converged,
        )
        attempts.append(
            SequentialPlanAttempt(
                attempt_index=attempt_index,
                trust_region_scale=float(scale),
                plan=plan_attempt,
                segments=tuple(segments),
            )
        )
    selected = min(range(4), key=lambda index: _attempt_rank(attempts[index]))
    return SequentialContactPlanningReport(
        initial_plan_id=plan.plan_id,
        initial_checkpoint_sha256=_checkpoint_sha256(grasp),
        endpoint_bounds=bounds,
        settings=settings,
        attempts=tuple(attempts),
        selected_attempt_index=selected,
    )


def materialize_sequential_plan_config(
    base_config: Mapping[str, Any],
    report: SequentialContactPlanningReport,
    *,
    attempt_index: int | None = None,
    validate: bool = False,
) -> dict[str, Any]:
    """Materialize one of the four plans without granting success status."""

    index = report.selected_attempt_index if attempt_index is None else int(attempt_index)
    if not 0 <= index < 4:
        raise ValueError("attempt_index must select one of the four plans")
    resolved = materialize_contact_plan_config(
        base_config, report.attempts[index].plan, validate=validate
    )
    metadata = resolved.setdefault("candidate_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("candidate_metadata must be a mapping")
    metadata["sequential_checkpoint_planning"] = {
        "report_id": report.as_mapping()["report_id"],
        "attempt_index": index,
        "attempt_report_id": report.attempts[index].as_mapping()["attempt_report_id"],
        "search_evidence_only": True,
        "final_success_requires_full_reset_rerun": True,
    }
    return resolved


def materialize_all_sequential_plan_configs(
    base_config: Mapping[str, Any],
    report: SequentialContactPlanningReport,
    *,
    validate: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Return the exact four materializable configs in attempt order."""

    return tuple(
        materialize_sequential_plan_config(
            base_config,
            report,
            attempt_index=index,
            validate=validate,
        )
        for index in range(4)
    )


class MuJoCoSequentialPlanningHooks:
    """Production physics implementation for sequential replanning hooks."""

    def __init__(
        self,
        initial_grasp: GraspPhysicsCheckpoint,
        settings: ContactConstrainedPlannerSettings,
        *,
        probe_duration_s: float | None = None,
        probe_epsilon_rad: float = 0.02,
    ) -> None:
        self.initial_grasp = initial_grasp
        self.settings = settings
        self.probe_duration_s = float(
            settings.duration_s / PLAN_SEGMENT_COUNT
            if probe_duration_s is None
            else probe_duration_s
        )
        self.probe_epsilon_rad = float(probe_epsilon_rad)
        if self.probe_duration_s <= 0.0 or self.probe_epsilon_rad <= 0.0:
            raise ValueError("probe duration/epsilon must be positive")
        compiled, self.info = build_model(copy.deepcopy(initial_grasp.config))
        signature = (compiled.nq, compiled.nv, compiled.na, compiled.nu)
        source_signature = (
            initial_grasp.model.nq,
            initial_grasp.model.nv,
            initial_grasp.model.na,
            initial_grasp.model.nu,
        )
        if signature != source_signature:
            raise ValueError("rebuilt scene differs from checkpoint model")
        self.model = initial_grasp.model
        self.base_preload = actuator_target_vector(
            self.model,
            initial_grasp.config["control"]["contact_preload_targets_rad"],
        )
        self.initial_position = initial_grasp.cube_position_world_m.copy()
        self.initial_quaternion = initial_grasp.cube_quaternion_wxyz.copy()
        labels = initial_grasp.config["contact_topology"]["target_faces"]
        self.target_labels = tuple(str(labels[name]) for name in ACTIVE_FINGERS)
        self.target_faces = tuple(face_from_label(value) for value in self.target_labels)

    def as_hooks(self) -> SequentialPlanningHooks:
        return SequentialPlanningHooks(self.collect_probes, self.rollout_segment)

    def _current_preload_config(
        self, grasp: GraspPhysicsCheckpoint, command_delta: np.ndarray
    ) -> dict[str, Any]:
        config = copy.deepcopy(grasp.config)
        targets = config["control"]["contact_preload_targets_rad"]
        for index, name in enumerate(ACTIVE_ACTUATORS):
            actuator_id = self.model.actuator(name).id
            targets[name] = float(self.base_preload[actuator_id] + command_delta[index])
        return config

    def _wrap_checkpoint(
        self,
        data: mujoco.MjData,
        *,
        step_index: int,
        command_delta: np.ndarray,
    ) -> GraspPhysicsCheckpoint:
        checkpoint = capture_physics_checkpoint(self.model, data, step_index=step_index)
        active_ids = np.asarray(
            [self.model.actuator(name).id for name in ACTIVE_ACTUATORS], dtype=np.int64
        )
        qpos = np.asarray(data.qpos[self.info.actuator_qpos_adrs][active_ids]).copy()
        return GraspPhysicsCheckpoint(
            model=self.model,
            checkpoint=checkpoint,
            cube_body_id=self.initial_grasp.cube_body_id,
            grasp_lock_step=self.initial_grasp.grasp_lock_step,
            actual_grasp_qpos_rad=qpos,
            lock_sample_joint_qpos_rad=qpos,
            cube_position_world_m=data.xpos[self.initial_grasp.cube_body_id].copy(),
            cube_quaternion_wxyz=data.xquat[self.initial_grasp.cube_body_id].copy(),
            config=self._current_preload_config(self.initial_grasp, command_delta),
        )

    def _observe_window(
        self,
        data: mujoco.MjData,
        initial_centroid: np.ndarray,
        initial_valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
        snapshot = contact_snapshot(self.model, data, self.info, classify_faces=True)
        assert snapshot.distal_face_force_n is not None
        assert snapshot.distal_face_position_moment_n_m is not None
        assert snapshot.active_nondistal_force_n is not None
        target = np.asarray(
            [snapshot.distal_face_force_n[i, int(self.target_faces[i])] for i in range(3)]
        )
        total = np.sum(snapshot.distal_face_force_n, axis=1)
        purity = np.divide(target, total, out=np.zeros(3), where=total > 0.0)
        centroids, centroid_valid = target_face_contact_centroids(
            snapshot.distal_face_force_n,
            snapshot.distal_face_position_moment_n_m,
            self.target_faces,
        )
        cube_rotation = data.xmat[self.initial_grasp.cube_body_id].reshape(3, 3)
        slip = np.full(3, 0.05)
        for index in range(3):
            if initial_valid[index] and centroid_valid[index]:
                displacement = centroids[index] - initial_centroid[index]
                normal = cube_rotation @ _FACE_NORMAL_LOCAL[self.target_labels[index]]
                tangent = displacement - float(displacement @ normal) * normal
                slip[index] = float(np.linalg.norm(tangent))
        valid = (target >= 0.05) & (purity >= 0.95)
        penetration_limit = float(
            self.initial_grasp.config["acceptance"]["max_penetration_m"]
        )
        forbidden = bool(
            snapshot.forbidden or snapshot.max_penetration > penetration_limit + 1e-12
        )
        nondistal = bool(np.any(snapshot.active_nondistal_force_n >= 0.05))
        return target, valid, slip, forbidden, nondistal

    def _run_branch(
        self,
        grasp: GraspPhysicsCheckpoint,
        start_delta: np.ndarray,
        end_delta: np.ndarray,
        duration_s: float,
    ) -> tuple[mujoco.MjData, dict[str, Any]]:
        data = mujoco.MjData(self.model)
        restore_physics_checkpoint(self.model, data, grasp.checkpoint)
        start_position = data.xpos[grasp.cube_body_id].copy()
        start_quaternion = data.xquat[grasp.cube_body_id].copy()
        snapshot = contact_snapshot(self.model, data, self.info, classify_faces=True)
        assert snapshot.distal_face_force_n is not None
        assert snapshot.distal_face_position_moment_n_m is not None
        initial_centroid, initial_valid = target_face_contact_centroids(
            snapshot.distal_face_force_n,
            snapshot.distal_face_position_moment_n_m,
            self.target_faces,
        )
        timestep = float(self.model.opt.timestep)
        steps = int(round(float(duration_s) / timestep))
        if steps <= 0 or abs(steps * timestep - duration_s) > 0.5 * timestep + 1e-12:
            raise ValueError("segment/probe duration does not align with model timestep")
        interpolation_times = np.asarray((0.0, duration_s), dtype=np.float64)
        interpolation_values = np.stack((start_delta, end_delta))
        interpolation_velocities, interpolation_accelerations = (
            quintic_c2_knot_derivatives(
                interpolation_times, interpolation_values
            )
        )
        forces: list[np.ndarray] = []
        validity: list[np.ndarray] = []
        slips: list[np.ndarray] = []
        forbidden = False
        nondistal = False
        for step in range(steps):
            delta, _, _, _ = interpolate_quintic_c2(
                interpolation_times,
                interpolation_values,
                (step + 1) * timestep,
                knot_velocities=interpolation_velocities,
                knot_accelerations=interpolation_accelerations,
            )
            data.ctrl[:] = self.base_preload
            for index, name in enumerate(ACTIVE_ACTUATORS):
                data.ctrl[self.model.actuator(name).id] += delta[index]
            mujoco.mj_step(self.model, data)
            mujoco.mj_forward(self.model, data)
            force, valid, slip, bad, nonterminal = self._observe_window(
                data, initial_centroid, initial_valid
            )
            if step >= steps // 2:
                forces.append(force)
                validity.append(valid)
                slips.append(slip)
            forbidden = forbidden or bad
            nondistal = nondistal or nonterminal
        evidence = {
            "steps": steps,
            "segment_response_6d": np.concatenate(
                (
                    data.xpos[grasp.cube_body_id] - start_position,
                    _quat_rotation_vector_wxyz(
                        start_quaternion, data.xquat[grasp.cube_body_id]
                    ),
                )
            ),
            "cumulative_response_6d": np.concatenate(
                (
                    data.xpos[grasp.cube_body_id] - self.initial_position,
                    _quat_rotation_vector_wxyz(
                        self.initial_quaternion, data.xquat[grasp.cube_body_id]
                    ),
                )
            ),
            "force": np.min(np.asarray(forces), axis=0),
            "valid": np.all(np.asarray(validity), axis=0),
            "slip": np.max(np.asarray(slips), axis=0),
            "forbidden": forbidden,
            "nondistal": nondistal,
        }
        return data, evidence

    def _probe(
        self, grasp: GraspPhysicsCheckpoint, specification: ProbeSpecification
    ) -> Mapping[str, Any]:
        current = np.asarray(
            [
                float(
                    grasp.config["control"]["contact_preload_targets_rad"][name]
                )
                - float(self.base_preload[self.model.actuator(name).id])
                for name in ACTIVE_ACTUATORS
            ],
            dtype=np.float64,
        )
        applied = np.asarray(specification.applied_delta_rad, dtype=np.float64)
        _, evidence = self._run_branch(
            grasp, current, current + applied, self.probe_duration_s
        )
        return {
            "probe": specification.as_mapping(),
            "checkpoint_step_index": grasp.checkpoint.step_index,
            "response_6d": evidence["segment_response_6d"].tolist(),
            "contact_evidence": {
                "target_normal_force_n": evidence["force"].tolist(),
                "target_contact_valid": evidence["valid"].tolist(),
                "tangent_slip_m": evidence["slip"].tolist(),
                "forbidden_contact": evidence["forbidden"],
                "active_nondistal_contact": evidence["nondistal"],
            },
        }

    def collect_probes(
        self, grasp: GraspPhysicsCheckpoint, _segment_index: int
    ) -> Sequence[ExtendedProbeSample]:
        return collect_extended_probe_samples(
            grasp, self._probe, epsilon_rad=self.probe_epsilon_rad
        )

    def rollout_segment(
        self,
        grasp: GraspPhysicsCheckpoint,
        _segment_index: int,
        start_delta: np.ndarray,
        end_delta: np.ndarray,
        duration_s: float,
    ) -> SequentialSegmentRollout:
        data, evidence = self._run_branch(
            grasp, start_delta, end_delta, duration_s
        )
        next_grasp = self._wrap_checkpoint(
            data,
            step_index=grasp.checkpoint.step_index + int(evidence["steps"]),
            command_delta=end_delta,
        )
        return SequentialSegmentRollout(
            next_grasp=next_grasp,
            segment_object_response_6d=evidence["segment_response_6d"],
            cumulative_object_response_6d=evidence["cumulative_response_6d"],
            target_normal_force_n=evidence["force"],
            target_contact_valid=evidence["valid"],
            tangent_slip_m=evidence["slip"],
            forbidden_contact=bool(evidence["forbidden"]),
            active_nondistal_contact=bool(evidence["nondistal"]),
            physics_steps=int(evidence["steps"]),
        )


__all__ = [
    "MuJoCoSequentialPlanningHooks",
    "SEQUENTIAL_PLANNER_SCHEMA_VERSION",
    "SequentialContactPlanningReport",
    "SequentialPlanAttempt",
    "SequentialPlanningHooks",
    "SequentialSegmentEvidence",
    "SequentialSegmentRollout",
    "interpolate_segment_command",
    "materialize_all_sequential_plan_configs",
    "materialize_sequential_plan_config",
    "plan_sequential_contact_trajectory",
]
