"""Pair-aware local grasp-pose search helpers for schema v15.

The generic point-target DLS used by schemas v12--v14 deliberately knows
nothing about the directed index-to-middle joint line.  Schema v15 needs both
properties at once: all three real distal witnesses must stay on their frozen
target points *and* the ordered joint line must stay close to cube-local
``+Y``.  This module layers a small, bounded null-space refinement on top of
the existing point-target solver without changing any older numerical path.

The functions here own no multiprocessing, artifact, or campaign-ledger
policy.  A production backend can compile one evaluator per worker, call
``solve_pair_aware_static_candidate`` for each deterministic
``StaticPerturbation``, and then apply the six canonical grasp-controller
variants.  Static results remain search evidence only.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from ..config import ACTIVE_ACTUATORS
from ..joint_pair_geometry import (
    measure_oriented_joint_pair_geometry,
    resolve_joint_pair,
)
from ..scene import build_model, rpy_degrees_to_quaternion
from ..simulation import _active_finger_self_collision_snapshot
from .actual_contact_grasp_pose import (
    apply_precontact_solution,
    evaluate_direct_actual_contact_pose,
)
from .contact_point_targeted_search import (
    ContactPointPlan,
    ContactPointSearchPolicy,
    PointTargetDLSResult,
    PointTargetDLSSettings,
    PointTargetTrialEvaluation,
    PointTargetVariables,
    assert_frozen_contact_point_plan,
    materialize_point_target_candidate,
    model_active_joint_bounds,
    point_target_boundary_violations,
    point_target_static_acceptance,
    point_target_trial_evaluation,
    project_point_target_variables,
    solve_point_target_dls,
)
from .joint_pair_near_zero_campaign import (
    EXPERIMENT_ID,
    GraspControlVariant,
    StaticPerturbation,
)


_EPSILON = 1e-12
_VARIABLE_COUNT = 14
_PAIR_NAMES = ("left_hand_index_joint1", "left_hand_mid_joint1")


def _finite_vector(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must have shape {shape} and be finite")
    return result


def _require_v15(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 15:
        raise ValueError("pair-aware pose search requires schema version 15")
    if str(config.get("experiment_id", "")) != EXPERIMENT_ID:
        raise ValueError("pair-aware pose search requires the registered v15 experiment")
    cube = config.get("cube")
    if not isinstance(cube, Mapping):
        raise ValueError("config.cube must be a mapping")
    if not (
        math.isclose(float(cube.get("edge_m", -1.0)), 0.079, abs_tol=1e-12)
        and math.isclose(float(cube.get("mass_kg", -1.0)), 0.160, abs_tol=1e-12)
        and math.isclose(float(cube.get("friction", -1.0)), 0.8, abs_tol=1e-12)
    ):
        raise ValueError("v15 pair-aware pose search requires the fixed 79 mm cube")


@dataclass(frozen=True, slots=True)
class JointPairStaticObservation:
    """One real MuJoCo joint-line and active-finger collision observation."""

    vector_cube_m: tuple[float, float, float]
    signed_residual: tuple[float, float]
    length_m: float
    angle_deg: float
    positive_y: bool
    active_finger_self_collision: bool
    self_collision_contact_count: int = 0
    self_collision_max_penetration_m: float = 0.0

    def __post_init__(self) -> None:
        vector = _finite_vector(self.vector_cube_m, (3,), "vector_cube_m")
        residual = _finite_vector(self.signed_residual, (2,), "signed_residual")
        length = float(self.length_m)
        angle = float(self.angle_deg)
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("length_m must be positive and finite")
        if not math.isfinite(angle) or angle < 0.0:
            raise ValueError("angle_deg must be non-negative and finite")
        if bool(self.positive_y) != bool(vector[1] > 0.0):
            raise ValueError("positive_y disagrees with vector_cube_m")
        if self.positive_y:
            expected = np.asarray((vector[0] / vector[1], vector[2] / vector[1]))
            if not np.allclose(expected, residual, rtol=1e-10, atol=1e-12):
                raise ValueError("signed_residual disagrees with vector_cube_m")
        if int(self.self_collision_contact_count) < 0:
            raise ValueError("self_collision_contact_count must be non-negative")
        penetration = float(self.self_collision_max_penetration_m)
        if not math.isfinite(penetration) or penetration < 0.0:
            raise ValueError("self_collision_max_penetration_m must be non-negative")

    @property
    def safe(self) -> bool:
        return bool(self.positive_y and not self.active_finger_self_collision)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "vector_cube_m": list(self.vector_cube_m),
            "signed_residual": list(self.signed_residual),
            "length_m": self.length_m,
            "angle_deg": self.angle_deg,
            "positive_y": self.positive_y,
            "active_finger_self_collision": self.active_finger_self_collision,
            "self_collision_contact_count": self.self_collision_contact_count,
            "self_collision_max_penetration_m": (
                self.self_collision_max_penetration_m
            ),
        }


PairEvaluator = Callable[[Mapping[str, Any]], JointPairStaticObservation]
ContactEvaluator = Callable[[Mapping[str, Any]], PointTargetTrialEvaluation]


@dataclass(frozen=True, slots=True)
class PairNullspaceRefinementSettings:
    """Small post-DLS trust region used to drive the pair residual to zero."""

    maximum_iterations: int = 4
    target_angle_deg: float = 0.05
    minimum_angle_improvement_deg: float = 1e-5
    pair_residual_scale_deg: float = 0.5
    pair_weight: float = 1.0
    contact_preservation_weight: float = 0.35
    damping: float = 0.05
    joint_finite_difference_rad: float = 2e-4
    translation_finite_difference_m: float = 2e-5
    rotation_finite_difference_rad: float = 2e-4
    joint_step_rad: float = 0.01
    translation_step_m: float = 0.0005
    rotation_step_rad: float = math.radians(0.25)

    def __post_init__(self) -> None:
        if isinstance(self.maximum_iterations, bool) or int(self.maximum_iterations) <= 0:
            raise ValueError("maximum_iterations must be a positive integer")
        for name in (
            "target_angle_deg",
            "minimum_angle_improvement_deg",
            "pair_residual_scale_deg",
            "pair_weight",
            "contact_preservation_weight",
            "damping",
            "joint_finite_difference_rad",
            "translation_finite_difference_m",
            "rotation_finite_difference_rad",
            "joint_step_rad",
            "translation_step_m",
            "rotation_step_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.target_angle_deg >= self.pair_residual_scale_deg:
            raise ValueError("target angle must be below the residual scale")


@dataclass(frozen=True, slots=True)
class PairNullspaceRefinementResult:
    config: dict[str, Any]
    variables: PointTargetVariables
    contact_evaluation: PointTargetTrialEvaluation
    pair_observation: JointPairStaticObservation
    stop_reason: str
    iterations: tuple[dict[str, Any], ...]

    @property
    def improved(self) -> bool:
        return any(bool(value.get("accepted", False)) for value in self.iterations)


@dataclass(frozen=True, slots=True)
class PairAwareStaticCandidate:
    perturbation: StaticPerturbation
    point_dls: PointTargetDLSResult
    pair_refinement: PairNullspaceRefinementResult
    static_pass: bool

    @property
    def config(self) -> dict[str, Any]:
        return copy.deepcopy(self.pair_refinement.config)

    @property
    def candidate_id(self) -> int:
        return self.perturbation.candidate_id

    @property
    def maximum_point_distance_m(self) -> float:
        distances = self.pair_refinement.contact_evaluation.point_distance_m
        return math.inf if distances is None else float(max(distances))

    @property
    def rank(self) -> tuple[Any, ...]:
        # Pair alignment precedes point error once every real witness and hard
        # collision condition passed.  Source distance breaks near ties and
        # makes the known measured grasp the natural conservative anchor.
        value = self.pair_refinement.variables.as_array()
        source = self.point_dls.variables.as_array()
        scale = np.asarray(
            (0.03,) * 8
            + (0.0015,) * 3
            + (math.radians(1.5),) * 3,
            dtype=np.float64,
        )
        local_motion = float(np.linalg.norm((value - source) / scale))
        return (
            not self.static_pass,
            self.pair_refinement.pair_observation.angle_deg,
            self.maximum_point_distance_m,
            local_motion,
            self.candidate_id,
        )


def static_perturbation_variables(
    seed_config: Mapping[str, Any], perturbation: StaticPerturbation
) -> PointTargetVariables:
    """Resolve all 14 offsets around the measured schema-v15 seed."""

    _require_v15(seed_config)
    base = PointTargetVariables.from_config(seed_config)
    joints = np.asarray(base.actual_joint_qpos_rad) + np.asarray(
        perturbation.joint_qpos_offset_rad
    )
    translation = np.asarray(base.root_delta_cube_m) + np.asarray(
        perturbation.root_delta_cube_m
    )
    rotation = np.asarray(base.wrist_local_rotvec_rad) + np.radians(
        perturbation.wrist_local_rotvec_deg
    )
    return PointTargetVariables(tuple(joints), tuple(translation), tuple(rotation))


def materialize_static_perturbation(
    seed_config: Mapping[str, Any],
    perturbation: StaticPerturbation,
    *,
    synchronize_preload: bool = True,
) -> dict[str, Any]:
    """Materialize a 14-variable start while preserving the cube byte-for-byte."""

    variables = static_perturbation_variables(seed_config, perturbation)
    result = materialize_point_target_candidate(
        seed_config,
        variables,
        signed_orbit_deg=0.0,
        synchronize_preload=synchronize_preload,
    )
    if result["cube"] != seed_config["cube"]:
        raise AssertionError("v15 static perturbation changed the fixed cube")
    metadata = result.setdefault("candidate_metadata", {})
    metadata["v15_static_perturbation"] = {
        **perturbation.as_mapping(),
        "cube_pose_sampled": False,
        "static_filter_is_success_evidence": False,
    }
    return result


def materialize_grasp_control_variant(
    candidate_config: Mapping[str, Any],
    variant: GraspControlVariant,
    *,
    reference_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one canonical close-duration/profile variant to a static pose.

    Point-target DLS derives a collision-free precontact pose and initially
    synchronizes preload to the proposed actual qpos.  ``original`` transports
    the authenticated source's *preload residual* to that new qpos, while
    ``synchronized_preload`` keeps the exact qpos target and synchronizes the
    command timing.  This distinction is essential: merely changing timing
    would otherwise make the declared six-controller grid contain duplicate
    preload policies.
    """

    _require_v15(candidate_config)
    result = copy.deepcopy(dict(candidate_config))
    result["control_protocol"]["close_s"] = float(variant.close_s)
    if variant.mode == "synchronized_preload":
        nominal = result["grasp_pose"]["nominal_joint_qpos_rad"]
        result["control"]["contact_preload_targets_rad"] = {
            name: float(nominal[name]) for name in ACTIVE_ACTUATORS
        }
        result["control"]["close_profile"] = {
            name: {"start_fraction": 0.0, "end_fraction": 1.0}
            for name in ACTIVE_ACTUATORS
        }
    elif variant.mode == "original" and reference_config is not None:
        _require_v15(reference_config)
        if candidate_config["cube"] != reference_config["cube"]:
            raise ValueError("preload reference must use the same fixed cube")
        candidate_nominal = result["grasp_pose"]["nominal_joint_qpos_rad"]
        reference_nominal = reference_config["grasp_pose"][
            "nominal_joint_qpos_rad"
        ]
        reference_preload = reference_config["control"][
            "contact_preload_targets_rad"
        ]
        result["control"]["contact_preload_targets_rad"] = {
            name: float(
                candidate_nominal[name]
                + reference_preload[name]
                - reference_nominal[name]
            )
            for name in ACTIVE_ACTUATORS
        }
        result["control"]["close_profile"] = copy.deepcopy(
            reference_config["control"]["close_profile"]
        )
    elif variant.mode != "original":  # pragma: no cover - dataclass validates
        raise ValueError(f"unsupported close mode {variant.mode}")
    metadata = result.setdefault("candidate_metadata", {})
    metadata["v15_grasp_control_variant"] = variant.as_mapping()
    return result


def build_pair_aware_static_evaluators(
    seed_config: Mapping[str, Any],
) -> tuple[
    ContactEvaluator,
    PairEvaluator,
    Mapping[str, tuple[float, float]],
]:
    """Compile one MuJoCo model and return reusable real-geometry callbacks."""

    _require_v15(seed_config)
    plan = assert_frozen_contact_point_plan(seed_config)
    model, info = build_model(copy.deepcopy(dict(seed_config)))
    data = mujoco.MjData(model)
    binding = resolve_joint_pair(model, _PAIR_NAMES)
    if binding is None:  # pragma: no cover - fixed names are always supplied
        raise RuntimeError("v15 joint-pair binding could not be resolved")
    minimum_length = float(seed_config["joint_pair_alignment"]["minimum_length_m"])
    active_ids = np.asarray(
        [model.actuator(name).id for name in ACTIVE_ACTUATORS], dtype=np.int64
    )
    qpos_addresses = np.asarray(info.actuator_qpos_adrs, dtype=np.int64)[active_ids]

    def contact(config: Mapping[str, Any]) -> PointTargetTrialEvaluation:
        static = evaluate_direct_actual_contact_pose(model, data, info, config)
        return point_target_trial_evaluation(
            static,
            config,
            plan,
            minimum_normal_alignment=0.95,
            maximum_penetration_m=0.002,
            require_extended_safety_evidence=False,
        )

    def pair(config: Mapping[str, Any]) -> JointPairStaticObservation:
        model.body_pos[info.root_body_id] = np.asarray(
            config["hand_pose"]["translation_m"], dtype=np.float64
        )
        model.body_quat[info.root_body_id] = rpy_degrees_to_quaternion(
            config["hand_pose"]["rpy_deg"]
        )
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_addresses] = np.asarray(
            [
                config["grasp_pose"]["nominal_joint_qpos_rad"][name]
                for name in ACTIVE_ACTUATORS
            ],
            dtype=np.float64,
        )
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        geometry = measure_oriented_joint_pair_geometry(
            data.xanchor[binding.first_joint_id],
            data.xanchor[binding.second_joint_id],
            np.asarray(data.xmat[binding.cube_body_id]).reshape(3, 3),
            minimum_separation_m=minimum_length,
        )
        collision = _active_finger_self_collision_snapshot(model, data, info)
        return JointPairStaticObservation(
            vector_cube_m=tuple(float(v) for v in geometry["vector_cube_m"]),
            signed_residual=tuple(
                float(v) for v in geometry["joint_pair_signed_residual"]
            ),
            length_m=float(geometry["length_m"]),
            angle_deg=float(geometry["angle_to_cube_positive_y_deg"]),
            positive_y=True,
            active_finger_self_collision=bool(collision["active"]),
            self_collision_contact_count=int(collision["contact_count"]),
            self_collision_max_penetration_m=float(
                collision["max_penetration_m"]
            ),
        )

    return contact, pair, model_active_joint_bounds(model, seed_config)


def build_v15_local_contact_policy(
    config: Mapping[str, Any],
) -> ContactPointSearchPolicy:
    """Return the exact +/- local pose envelope registered by schema v15."""

    _require_v15(config)
    plan = assert_frozen_contact_point_plan(config)
    root = np.asarray(config["hand_pose"]["translation_m"], dtype=np.float64)
    # The local search envelope is tied to the measured anchor rather than a
    # broad historical v12 range.  The actual distance is rechecked by the
    # generic boundary function after every coupled wrist transform.
    from .actual_contact_grasp_pose import _cube_world_position

    distance = float(np.linalg.norm(root - _cube_world_position(config)))
    return ContactPointSearchPolicy(
        seed=20260821,
        sample_count=1,
        retain_point_plan_count=1,
        reference_points={name: plan.points[name] for name in plan.points},
        half_width_m=0.008,
        minimum_edge_margin_m=0.0005,
        maximum_height_spread_m=0.005,
        minimum_index_middle_separation_m=0.010,
        static_target_radius_m=0.004,
        target_radius_m=float(plan.target_radius_m),
        minimum_normal_alignment=0.95,
        maximum_penetration_m=0.002,
        signed_orbit_deg=(0.0,),
        root_delta_cube_m={axis: (-0.0015, 0.0015) for axis in "xyz"},
        wrist_local_rotvec_deg={axis: (-1.5, 1.5) for axis in "xyz"},
        max_wrist_local_rotvec_norm_deg=2.0,
        root_cube_distance_m=(distance - 0.010, distance + 0.010),
        thumb_actual_range_rad=(1.40, 1.60),
    )


def refine_pair_alignment_in_contact_nullspace(
    base_config: Mapping[str, Any],
    start_variables: PointTargetVariables,
    *,
    contact_evaluator: ContactEvaluator,
    pair_evaluator: PairEvaluator,
    policy: ContactPointSearchPolicy,
    joint_bounds: Mapping[str, tuple[float, float]],
    settings: PairNullspaceRefinementSettings = PairNullspaceRefinementSettings(),
) -> PairNullspaceRefinementResult:
    """Reduce pair residual while keeping the real point-contact gate passed."""

    _require_v15(base_config)
    current = start_variables.as_array()
    finite_difference = np.asarray(
        (settings.joint_finite_difference_rad,) * 8
        + (settings.translation_finite_difference_m,) * 3
        + (settings.rotation_finite_difference_rad,) * 3
    )
    variable_scale = np.asarray(
        (settings.joint_step_rad,) * 8
        + (settings.translation_step_m,) * 3
        + (settings.rotation_step_rad,) * 3
    )
    plan = assert_frozen_contact_point_plan(base_config)
    contact_scale = np.asarray(
        (0.0005,) * 3 + (0.004,) * 6 + (0.05,) * 3,
        dtype=np.float64,
    )
    pair_scale = math.tan(math.radians(settings.pair_residual_scale_deg))

    def boundary(value: np.ndarray) -> tuple[str, ...]:
        return point_target_boundary_violations(
            base_config,
            PointTargetVariables.from_array(value),
            policy,
            signed_orbit_deg=0.0,
            joint_bounds=joint_bounds,
            check_pose_constraints=True,
        )

    def evaluate(
        value: np.ndarray,
    ) -> tuple[dict[str, Any], PointTargetTrialEvaluation, JointPairStaticObservation]:
        candidate = materialize_point_target_candidate(
            base_config,
            PointTargetVariables.from_array(value),
            signed_orbit_deg=0.0,
            synchronize_preload=True,
        )
        return candidate, contact_evaluator(candidate), pair_evaluator(candidate)

    initial_boundary = boundary(current)
    if initial_boundary:
        raise ValueError(
            "initial pair-refinement variables violate boundaries: "
            + ", ".join(initial_boundary)
        )
    config, contact, pair = evaluate(current)
    acceptance = point_target_static_acceptance(contact, policy)
    if not acceptance.passed or not pair.safe:
        return PairNullspaceRefinementResult(
            config,
            PointTargetVariables.from_array(current),
            contact,
            pair,
            "initial_contact_or_pair_gate_failed",
            (),
        )
    iterations: list[dict[str, Any]] = []
    stop_reason = "maximum_iterations"
    for iteration in range(settings.maximum_iterations):
        if pair.angle_deg <= settings.target_angle_deg + _EPSILON:
            stop_reason = "pair_target_converged"
            break
        base_measurement = _finite_vector(contact.measurement, (12,), "contact measurement")
        base_residual = _finite_vector(pair.signed_residual, (2,), "pair residual")
        contact_jacobian = np.zeros((12, _VARIABLE_COUNT), dtype=np.float64)
        pair_jacobian = np.zeros((2, _VARIABLE_COUNT), dtype=np.float64)
        usable = np.zeros(_VARIABLE_COUNT, dtype=bool)
        rejections: list[dict[str, Any]] = []
        for column in range(_VARIABLE_COUNT):
            for direction in (1.0, -1.0):
                trial = current.copy()
                trial[column] += direction * finite_difference[column]
                reasons = boundary(trial)
                if reasons:
                    rejections.append(
                        {"column": column, "direction": direction, "reasons": list(reasons)}
                    )
                    continue
                _, trial_contact, trial_pair = evaluate(trial)
                trial_acceptance = point_target_static_acceptance(trial_contact, policy)
                if not trial_acceptance.passed or not trial_pair.safe:
                    rejections.append(
                        {
                            "column": column,
                            "direction": direction,
                            "reasons": ["real_contact_or_pair_gate_failed"],
                        }
                    )
                    continue
                actual_step = direction * finite_difference[column]
                contact_jacobian[:, column] = (
                    _finite_vector(trial_contact.measurement, (12,), "trial contact")
                    - base_measurement
                ) / actual_step
                pair_jacobian[:, column] = (
                    _finite_vector(trial_pair.signed_residual, (2,), "trial pair")
                    - base_residual
                ) / actual_step
                usable[column] = True
                break
        normalized_contact = (
            contact_jacobian * variable_scale[np.newaxis, :]
        ) / contact_scale[:, np.newaxis]
        normalized_pair = (
            pair_jacobian * variable_scale[np.newaxis, :]
        ) / pair_scale
        active = np.flatnonzero(usable)
        if active.size == 0 or np.linalg.matrix_rank(normalized_pair[:, active]) == 0:
            iterations.append(
                {
                    "iteration": iteration,
                    "accepted": False,
                    "reason": "no_pair_sensitive_safe_variable",
                    "finite_difference_rejections": rejections,
                }
            )
            stop_reason = "no_pair_sensitive_safe_variable"
            break
        identity = np.eye(_VARIABLE_COUNT, dtype=np.float64)
        system = np.vstack(
            (
                settings.pair_weight * normalized_pair,
                settings.contact_preservation_weight * normalized_contact,
                settings.damping * identity,
            )
        )
        rhs = np.concatenate(
            (
                -settings.pair_weight * base_residual / pair_scale,
                np.zeros(12, dtype=np.float64),
                np.zeros(_VARIABLE_COUNT, dtype=np.float64),
            )
        )
        unit_step = np.linalg.lstsq(system, rhs, rcond=None)[0]
        unit_step[~usable] = 0.0
        peak = float(np.max(np.abs(unit_step)))
        if peak > 1.0:
            unit_step /= peak
        proposed_step = variable_scale * unit_step
        accepted: tuple[
            np.ndarray,
            dict[str, Any],
            PointTargetTrialEvaluation,
            JointPairStaticObservation,
            float,
        ] | None = None
        line_search: list[dict[str, Any]] = []
        for line_scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            trial = project_point_target_variables(
                current + line_scale * proposed_step,
                policy,
                joint_bounds,
            )
            reasons = boundary(trial)
            if reasons:
                line_search.append(
                    {"scale": line_scale, "accepted": False, "reasons": list(reasons)}
                )
                continue
            trial_config, trial_contact, trial_pair = evaluate(trial)
            trial_acceptance = point_target_static_acceptance(trial_contact, policy)
            improved = bool(
                trial_acceptance.passed
                and trial_pair.safe
                and trial_pair.angle_deg
                <= pair.angle_deg - settings.minimum_angle_improvement_deg
            )
            line_search.append(
                {
                    "scale": line_scale,
                    "accepted": improved,
                    "pair_angle_deg": trial_pair.angle_deg,
                    "maximum_point_distance_m": trial_acceptance.maximum_point_distance_m,
                }
            )
            if improved:
                accepted = (
                    trial,
                    trial_config,
                    trial_contact,
                    trial_pair,
                    line_scale,
                )
                break
        iterations.append(
            {
                "iteration": iteration,
                "angle_before_deg": pair.angle_deg,
                "jacobian_rank": int(np.linalg.matrix_rank(normalized_pair[:, active])),
                "usable_variable_count": int(active.size),
                "finite_difference_rejections": rejections,
                "line_search": line_search,
                "accepted": accepted is not None,
            }
        )
        if accepted is None:
            stop_reason = "no_safe_pair_improvement"
            break
        current, config, contact, pair, accepted_scale = accepted
        iterations[-1]["accepted_scale"] = accepted_scale
        iterations[-1]["angle_after_deg"] = pair.angle_deg
    else:
        stop_reason = "maximum_iterations"

    final_acceptance = point_target_static_acceptance(contact, policy)
    if final_acceptance.passed and bool(
        getattr(contact.static_result, "static_geometry_pass", False)
    ):
        config = apply_precontact_solution(config, contact.static_result)
    metadata = config.setdefault("candidate_metadata", {})
    metadata["v15_pair_nullspace_refinement"] = {
        "schema_version": 1,
        "point_plan_id": plan.point_plan_id,
        "stop_reason": stop_reason,
        "iteration_count": len(iterations),
        "final_angle_deg": pair.angle_deg,
        "static_filter_is_success_evidence": False,
    }
    return PairNullspaceRefinementResult(
        config,
        PointTargetVariables.from_array(current),
        contact,
        pair,
        stop_reason,
        tuple(iterations),
    )


def solve_pair_aware_static_candidate(
    seed_config: Mapping[str, Any],
    perturbation: StaticPerturbation,
    *,
    contact_evaluator: ContactEvaluator | None = None,
    pair_evaluator: PairEvaluator | None = None,
    joint_bounds: Mapping[str, tuple[float, float]] | None = None,
    point_settings: PointTargetDLSSettings = PointTargetDLSSettings(
        maximum_iterations=12
    ),
    pair_settings: PairNullspaceRefinementSettings = PairNullspaceRefinementSettings(),
) -> PairAwareStaticCandidate:
    """Run real point DLS, pair-nullspace refinement and all static hard gates."""

    _require_v15(seed_config)
    if contact_evaluator is None or pair_evaluator is None or joint_bounds is None:
        if not (
            contact_evaluator is None
            and pair_evaluator is None
            and joint_bounds is None
        ):
            raise ValueError("supply all evaluators/bounds together or none of them")
        contact_evaluator, pair_evaluator, joint_bounds = (
            build_pair_aware_static_evaluators(seed_config)
        )
    assert contact_evaluator is not None
    assert pair_evaluator is not None
    assert joint_bounds is not None
    policy = build_v15_local_contact_policy(seed_config)
    initial = static_perturbation_variables(seed_config, perturbation)
    point = solve_point_target_dls(
        seed_config,
        signed_orbit_deg=0.0,
        policy=policy,
        initial_variables=initial,
        settings=point_settings,
        evaluator=contact_evaluator,
        joint_bounds=joint_bounds,
        check_pose_constraints=True,
    )
    pair = refine_pair_alignment_in_contact_nullspace(
        seed_config,
        point.variables,
        contact_evaluator=contact_evaluator,
        pair_evaluator=pair_evaluator,
        policy=policy,
        joint_bounds=joint_bounds,
        settings=pair_settings,
    )
    acceptance = point_target_static_acceptance(pair.contact_evaluation, policy)
    alignment = seed_config["joint_pair_alignment"]
    static_pass = bool(
        acceptance.passed
        and pair.pair_observation.safe
        and pair.pair_observation.length_m
        >= float(alignment["minimum_length_m"]) - _EPSILON
        and pair.pair_observation.angle_deg
        <= float(alignment["grasp_max_deg"]) + _EPSILON
    )
    config = pair.config
    metadata = config.setdefault("candidate_metadata", {})
    metadata["v15_static_perturbation"] = {
        **perturbation.as_mapping(),
        "cube_pose_sampled": False,
        "static_filter_is_success_evidence": False,
    }
    pair = PairNullspaceRefinementResult(
        config,
        pair.variables,
        pair.contact_evaluation,
        pair.pair_observation,
        pair.stop_reason,
        pair.iterations,
    )
    return PairAwareStaticCandidate(perturbation, point, pair, static_pass)


def retain_pair_aware_static_candidates(
    candidates: Sequence[PairAwareStaticCandidate], *, top_k: int = 64
) -> tuple[PairAwareStaticCandidate, ...]:
    if isinstance(top_k, bool) or int(top_k) <= 0:
        raise ValueError("top_k must be a positive integer")
    identifiers = [value.candidate_id for value in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate IDs must be unique")
    passed = [value for value in candidates if value.static_pass]
    return tuple(sorted(passed, key=lambda value: value.rank)[: int(top_k)])


__all__ = [
    "ContactEvaluator",
    "JointPairStaticObservation",
    "PairAwareStaticCandidate",
    "PairEvaluator",
    "PairNullspaceRefinementResult",
    "PairNullspaceRefinementSettings",
    "build_pair_aware_static_evaluators",
    "build_v15_local_contact_policy",
    "materialize_grasp_control_variant",
    "materialize_static_perturbation",
    "refine_pair_alignment_in_contact_nullspace",
    "retain_pair_aware_static_candidates",
    "solve_pair_aware_static_candidate",
    "static_perturbation_variables",
]
