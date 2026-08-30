"""Projected active-set DLS for schema-v11 relative wrist candidates.

This solver is deliberately separate from the frozen v11 campaign solver.  It
uses the same materializer and the same fail-closed MuJoCo witness evaluator,
but treats the registered contact gates as intervals rather than pulling every
contact towards one arbitrary point inside those intervals.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import mujoco

from .actual_contact_grasp_pose import (
    ActualContactStaticThresholds,
    apply_precontact_solution,
)
from ..config import ACTIVE_ACTUATORS, validate_config
from ..experiment import resolve_experiment
from ..scene import build_model, rpy_degrees_to_quaternion
from .relative_wrist_pose_search import (
    MEASUREMENT_NAMES,
    NON_THUMB_ACTUATORS,
    THUMB_BEND_ACTUATOR,
    VARIABLE_COUNT,
    VARIABLE_NAMES,
    RelativeWristDLSResult,
    RelativeWristDLSSettings,
    RelativeWristPoseSearchPolicy,
    RelativeWristTrialEvaluation,
    RelativeWristVariables,
    TrialEvaluator,
    build_actual_contact_trial_evaluator,
    materialize_relative_wrist_candidate,
    relative_wrist_boundary_violations,
)


_EPS = 1e-12


def classify_full_scene_penetrations(
    contacts: Any,
    *,
    cube_geom_id: int,
    support_geom_id: int,
    maximum_penetration_m: float = 0.002,
) -> tuple[dict[str, Any], ...]:
    """Return deep contacts, exempting only the cube/support geom pair."""

    allowed = frozenset((int(cube_geom_id), int(support_geom_id)))
    return tuple(
        copy.deepcopy(dict(contact))
        for contact in contacts
        if frozenset((int(contact["geom1_id"]), int(contact["geom2_id"])))
        != allowed
        and float(contact["distance_m"]) < -float(maximum_penetration_m) - _EPS
    )


class CachedFullScenePenetrationGate:
    """Check nominal/precontact contacts with one cached MuJoCo model/data."""

    def __init__(
        self,
        base_config: Mapping[str, Any],
        *,
        maximum_penetration_m: float = 0.002,
    ) -> None:
        self.model, self.info = build_model(copy.deepcopy(dict(base_config)))
        self.data = mujoco.MjData(self.model)
        self.maximum_penetration_m = float(maximum_penetration_m)
        if not math.isfinite(self.maximum_penetration_m) or self.maximum_penetration_m <= 0.0:
            raise ValueError("maximum_penetration_m must be positive and finite")
        active_ids = np.asarray(self.info.active_actuator_ids, dtype=np.int64)
        self._qpos_addresses = np.asarray(
            self.info.actuator_qpos_adrs[active_ids], dtype=np.int64
        )
        self._allowed_pair = frozenset(
            (int(self.info.cube_geom_id), int(self.info.support_geom_id))
        )

    def _geom_name(self, geom_id: int) -> str:
        return str(self.model.geom(geom_id).name or f"geom_{geom_id}")

    def evaluate(
        self,
        config: Mapping[str, Any],
        *,
        nominal_joint_qpos_rad: Any,
        precontact_joint_qpos_rad: Any,
    ) -> dict[str, Any]:
        nominal = np.asarray(nominal_joint_qpos_rad, dtype=np.float64)
        precontact = np.asarray(precontact_joint_qpos_rad, dtype=np.float64)
        expected = (len(ACTIVE_ACTUATORS),)
        if (
            nominal.shape != expected
            or precontact.shape != expected
            or not np.isfinite(nominal).all()
            or not np.isfinite(precontact).all()
        ):
            return {
                "enabled": True,
                "passed": False,
                "maximum_allowed_penetration_m": self.maximum_penetration_m,
                "phases": {},
                "violations": [
                    {
                        "phase": "input",
                        "reason": "invalid_nominal_or_precontact_qpos",
                        # Finite sentinel keeps strict JSON artifact writers
                        # usable while still ranking the malformed input worst.
                        "penetration_m": 1.0,
                    }
                ],
            }

        self.model.body_pos[self.info.root_body_id] = np.asarray(
            config["hand_pose"]["translation_m"], dtype=np.float64
        )
        self.model.body_quat[self.info.root_body_id] = rpy_degrees_to_quaternion(
            config["hand_pose"]["rpy_deg"]
        )
        phases: dict[str, Any] = {}
        violations: list[dict[str, Any]] = []
        for phase, qpos in (("nominal", nominal), ("precontact", precontact)):
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self._qpos_addresses] = qpos
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            phase_contacts: list[dict[str, Any]] = []
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                geom1, geom2 = (int(contact.geom[0]), int(contact.geom[1]))
                distance = float(contact.dist)
                record = {
                    "phase": phase,
                    "contact_index": contact_index,
                    "geom1_id": geom1,
                    "geom1_name": self._geom_name(geom1),
                    "geom1_body_name": str(
                        self.model.body(int(self.model.geom_bodyid[geom1])).name
                    ),
                    "geom2_id": geom2,
                    "geom2_name": self._geom_name(geom2),
                    "geom2_body_name": str(
                        self.model.body(int(self.model.geom_bodyid[geom2])).name
                    ),
                    "distance_m": distance,
                    "penetration_m": max(0.0, -distance),
                }
                phase_contacts.append(record)
            phase_violations = list(
                classify_full_scene_penetrations(
                    phase_contacts,
                    cube_geom_id=int(self.info.cube_geom_id),
                    support_geom_id=int(self.info.support_geom_id),
                    maximum_penetration_m=self.maximum_penetration_m,
                )
            )
            violations.extend(phase_violations)
            disallowed_contacts = [
                record
                for record in phase_contacts
                if frozenset((record["geom1_id"], record["geom2_id"]))
                != self._allowed_pair
            ]
            deepest = max(
                disallowed_contacts,
                key=lambda item: float(item["penetration_m"]),
                default=None,
            )
            phases[phase] = {
                "disallowed_contact_count": len(disallowed_contacts),
                "deep_violation_count": len(phase_violations),
                "maximum_disallowed_penetration_m": (
                    0.0 if deepest is None else float(deepest["penetration_m"])
                ),
                "deepest_geom_pair": (
                    None
                    if deepest is None
                    else [deepest["geom1_name"], deepest["geom2_name"]]
                ),
            }
        return {
            "enabled": True,
            "passed": not violations,
            "maximum_allowed_penetration_m": self.maximum_penetration_m,
            "allowed_cube_support_pair": sorted(self._allowed_pair),
            "phases": phases,
            "maximum_disallowed_penetration_m": max(
                (
                    float(phase["maximum_disallowed_penetration_m"])
                    for phase in phases.values()
                ),
                default=0.0,
            ),
            "violations": violations,
        }


def _full_scene_safety_reason(record: Mapping[str, Any]) -> str:
    if record.get("reason"):
        return f"full_scene_penetration:{record['phase']}:{record['reason']}"
    return (
        f"full_scene_penetration:{record['phase']}:"
        f"{record['geom1_name']}|{record['geom2_name']}:"
        f"{float(record['penetration_m']):.9g}m"
    )


@dataclass(frozen=True, slots=True)
class ActiveSetEvaluationContext:
    """Reusable compiled evaluator state for one immutable scene/config.

    The MuJoCo closures and penetration gate are stateful and intentionally
    shared only by sequential solves in one worker.  Spawn workers must build
    one context each; callers must not share a context between threads.
    """

    evaluator: TrialEvaluator
    joint_bounds: Mapping[str, tuple[float, float]]
    full_scene_gate: CachedFullScenePenetrationGate | None

    def __post_init__(self) -> None:
        if not callable(self.evaluator):
            raise ValueError("evaluation context evaluator must be callable")
        if set(self.joint_bounds) != set(NON_THUMB_ACTUATORS):
            raise ValueError(
                "evaluation context joint_bounds must contain the seven "
                "non-thumb actuators"
            )
        resolved: dict[str, tuple[float, float]] = {}
        for name in NON_THUMB_ACTUATORS:
            low, high = self.joint_bounds[name]
            low_value, high_value = float(low), float(high)
            if (
                not math.isfinite(low_value)
                or not math.isfinite(high_value)
                or low_value > high_value
            ):
                raise ValueError(f"invalid evaluation context bound for {name}")
            resolved[name] = (low_value, high_value)
        object.__setattr__(self, "joint_bounds", MappingProxyType(resolved))


def build_active_set_evaluation_context(
    config: Mapping[str, Any],
) -> ActiveSetEvaluationContext:
    """Compile the target-witness evaluator and full-scene gate exactly once."""

    evaluator, joint_bounds = build_actual_contact_trial_evaluator(config)
    return ActiveSetEvaluationContext(
        evaluator=evaluator,
        joint_bounds=joint_bounds,
        full_scene_gate=CachedFullScenePenetrationGate(config),
    )


def evaluate_active_set_candidate(
    context: ActiveSetEvaluationContext,
    config: Mapping[str, Any],
) -> tuple[RelativeWristTrialEvaluation, dict[str, Any]]:
    """Freshly evaluate one candidate, including both full-scene phases.

    This is the public final-check path for recovery pipelines.  It never
    reuses a prior report: the base evaluator runs again, followed by fresh
    nominal and precontact ``mj_forward`` calls on the cached gate model.
    """

    evaluation = context.evaluator(config)
    if context.full_scene_gate is None:
        return evaluation, {"enabled": False, "reason": "no_full_scene_gate"}
    static = evaluation.static_result
    report = context.full_scene_gate.evaluate(
        config,
        nominal_joint_qpos_rad=getattr(static, "nominal_joint_qpos_rad", ()),
        precontact_joint_qpos_rad=getattr(static, "precontact_joint_qpos_rad", ()),
    )
    if not report["passed"]:
        evaluation = RelativeWristTrialEvaluation(
            static,
            evaluation.measurement,
            (
                *evaluation.safety_violations,
                *(
                    _full_scene_safety_reason(record)
                    for record in report["violations"]
                ),
            ),
        )
    return evaluation, report


def _variable_bounds(
    policy: RelativeWristPoseSearchPolicy,
    joint_bounds: Mapping[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    lower = [float(joint_bounds[name][0]) for name in NON_THUMB_ACTUATORS]
    upper = [float(joint_bounds[name][1]) for name in NON_THUMB_ACTUATORS]
    for axis in ("x", "y", "z"):
        low, high = policy.root_delta_cube_m[axis]
        lower.append(float(low))
        upper.append(float(high))
    for axis in ("x", "y", "z"):
        low, high = policy.wrist_local_rotvec_deg[axis]
        lower.append(math.radians(float(low)))
        upper.append(math.radians(float(high)))
    return np.asarray(lower), np.asarray(upper)


def _safe_number(value: Any, fallback: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return fallback
    return resolved if math.isfinite(resolved) else fallback


def _gate_residual_and_violations(
    evaluation: RelativeWristTrialEvaluation,
    thresholds: ActualContactStaticThresholds,
    precontact_bounds: Mapping[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return smooth normalized residuals and non-negative hard violations."""

    if evaluation.measurement is None:
        raise ValueError("gate residual requires a complete distal measurement")
    measurement = np.asarray(evaluation.measurement, dtype=np.float64)
    gap_low, gap_high = thresholds.signed_gap_m
    gap_mid = 0.5 * (gap_low + gap_high)
    gap_half = max(0.5 * (gap_high - gap_low), 1e-9)
    height_scale = max(thresholds.maximum_height_spread_m, 1e-9)
    normal_scale = max(1.0 - thresholds.minimum_normal_alignment, 1e-6)

    smooth: list[float] = []
    violations: list[float] = []
    for value in measurement[:3]:
        smooth.append((float(value) - gap_mid) / gap_half)
        violations.append(max(gap_low - value, value - gap_high, 0.0) / gap_half)
    for value in measurement[3:5]:
        smooth.append(float(value) / height_scale)
    # Pairwise height spread is max(0, h1-h0, h2-h0) - min(...).
    relative_heights = np.asarray((0.0, measurement[3], measurement[4]))
    height_violation = max(
        float(np.ptp(relative_heights)) - thresholds.maximum_height_spread_m,
        0.0,
    ) / height_scale
    violations.append(height_violation)
    for value in measurement[5:8]:
        smooth.append((float(value) - 1.0) / normal_scale)
        violations.append(
            max(thresholds.minimum_normal_alignment - value, 0.0) / normal_scale
        )

    static = evaluation.static_result
    witnesses = tuple(getattr(static, "target_witnesses", ()) or ())
    edge_values: list[float] = []
    for index in range(3):
        witness = witnesses[index] if index < len(witnesses) else None
        edge = _safe_number(getattr(witness, "edge_margin_m", None), -1.0)
        edge_values.append(edge)
        scale = max(thresholds.minimum_edge_margin_m, 1e-6)
        # Once inside the admissible region, do not pull the witness farther
        # from the edge at the expense of another hard gate.
        smooth.append(min((edge - thresholds.minimum_edge_margin_m) / scale, 0.0))
        violations.append(max(thresholds.minimum_edge_margin_m - edge, 0.0) / scale)

    retreats = tuple(getattr(static, "retreat_evidence", ()) or ())
    retreat_low, retreat_high = thresholds.retreat_m
    retreat_mid = 0.5 * (retreat_low + retreat_high)
    retreat_half = max(0.5 * (retreat_high - retreat_low), 1e-9)
    retreat_values: list[float] = []
    closure_angles: list[float] = []
    inward_speeds: list[float] = []
    for index in range(3):
        evidence = retreats[index] if index < len(retreats) else None
        retreat = _safe_number(
            getattr(evidence, "measured_outward_retreat_m", None), -1.0
        )
        angle = _safe_number(getattr(evidence, "closure_angle_deg", None), 180.0)
        inward = _safe_number(getattr(evidence, "inward_speed_m_s", None), -1.0)
        retreat_values.append(retreat)
        closure_angles.append(angle)
        inward_speeds.append(inward)
        smooth.append((retreat - retreat_mid) / retreat_half)
        violations.append(
            max(retreat_low - retreat, retreat - retreat_high, 0.0) / retreat_half
        )
        smooth.append(max(angle, 0.0) / thresholds.maximum_closure_angle_deg)
        violations.append(
            max(angle - thresholds.maximum_closure_angle_deg, 0.0)
            / thresholds.maximum_closure_angle_deg
        )
        # Direction is a hard boolean gate; use a dimensionless one-sided
        # residual without inventing a positive-speed acceptance threshold.
        direction_bad = 1.0 if inward <= 0.0 else 0.0
        smooth.append(direction_bad)
        violations.append(direction_bad)

    precontact = tuple(
        float(value)
        for value in (getattr(static, "precontact_joint_qpos_rad", ()) or ())
    )
    precontact_violations: dict[str, float] = {}
    for index, name in enumerate(ACTIVE_ACTUATORS):
        low, high = precontact_bounds[name]
        width = max(float(high) - float(low), 1e-6)
        if index >= len(precontact) or not math.isfinite(precontact[index]):
            signed = -1.0
            violation = 1.0
        else:
            value = precontact[index]
            if value < low:
                signed = (value - low) / width
            elif value > high:
                signed = (value - high) / width
            else:
                signed = 0.0
            violation = abs(signed)
        smooth.append(signed)
        violations.append(violation)
        precontact_violations[name] = violation

    diagnostics = {
        "gaps_m": measurement[:3].tolist(),
        "height_spread_m": float(np.ptp(relative_heights)),
        "normal_alignment": measurement[5:8].tolist(),
        "edge_margin_m": edge_values,
        "retreat_m": retreat_values,
        "closure_angle_deg": closure_angles,
        "inward_speed_m_s": inward_speeds,
        "precontact_target_normalized_violation": precontact_violations,
    }
    return np.asarray(smooth), np.asarray(violations), diagnostics


def _merit(
    evaluation: RelativeWristTrialEvaluation,
    thresholds: ActualContactStaticThresholds,
    variables: np.ndarray,
    reference: np.ndarray,
    variable_scale: np.ndarray,
    regularization_weight: float,
    precontact_bounds: Mapping[str, tuple[float, float]],
    promotion_config_valid: bool,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    residual, violations, gate = _gate_residual_and_violations(
        evaluation, thresholds, precontact_bounds
    )
    static_pass = bool(
        getattr(evaluation.static_result, "static_geometry_pass", False)
    )
    hard_pass = static_pass and promotion_config_valid
    regularization = float(
        math.sqrt(regularization_weight)
        * np.linalg.norm((variables - reference) / variable_scale)
    )
    # Lexicographic hard-pass and maximum-gate priority prevents a large but
    # already-admissible gap/retreat residual masking the last failing finger.
    key = (
        0.0 if hard_pass else 1.0,
        float(np.max(violations, initial=0.0)),
        float(np.linalg.norm(violations)),
        float(np.linalg.norm(residual)),
        regularization,
    )
    return key, {
        **gate,
        "static_geometry_pass": static_pass,
        "promotion_config_valid": promotion_config_valid,
        "hard_pass": hard_pass,
        "maximum_normalized_gate_violation": key[1],
        "normalized_gate_violation_l2": key[2],
        "normalized_residual_l2": key[3],
        "regularization": regularization,
        "merit_key": list(key),
    }


def _maximum_feasible_alpha(
    current: np.ndarray,
    step: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rotvec_radius_rad: float,
) -> float:
    alpha = 1.0
    for value, delta, low, high in zip(current, step, lower, upper):
        if delta > 0.0:
            alpha = min(alpha, (high - value) / delta)
        elif delta < 0.0:
            alpha = min(alpha, (low - value) / delta)
    r = current[10:13]
    d = step[10:13]
    a = float(d @ d)
    b = 2.0 * float(r @ d)
    c = float(r @ r) - rotvec_radius_rad**2
    if a > _EPS:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            positive = (-b + math.sqrt(discriminant)) / (2.0 * a)
            if positive >= 0.0:
                alpha = min(alpha, positive)
    return max(0.0, min(1.0, float(alpha)))


def _active_set_step(
    jacobian: np.ndarray,
    residual: np.ndarray,
    current: np.ndarray,
    reference: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    variable_scale: np.ndarray,
    settings: RelativeWristDLSSettings,
    rotvec_radius_rad: float,
) -> tuple[np.ndarray, tuple[int, ...], bool]:
    """Solve, freeze outward box coordinates, then apply sphere tangency."""

    free = np.ones(VARIABLE_COUNT, dtype=bool)
    active: set[int] = set()
    unit_step = np.zeros(VARIABLE_COUNT)
    scaled_jacobian = jacobian * variable_scale[np.newaxis, :]
    reg = math.sqrt(settings.regularization_weight)
    for _ in range(VARIABLE_COUNT + 1):
        indices = np.flatnonzero(free)
        if not len(indices):
            break
        system = np.vstack(
            (scaled_jacobian[:, indices], reg * np.eye(len(indices)))
        )
        rhs = np.concatenate(
            (-residual, -reg * (current[indices] - reference[indices]) / variable_scale[indices])
        )
        normal = system.T @ system + settings.damping**2 * np.eye(len(indices))
        solved = np.linalg.solve(normal, system.T @ rhs)
        unit_step[:] = 0.0
        # Component-wise trust region: one saturated coordinate cannot shrink
        # all other useful coordinates as in global infinity-norm scaling.
        unit_step[indices] = np.clip(solved, -1.0, 1.0)
        step = variable_scale * unit_step
        newly_active = []
        for index in indices:
            at_low = current[index] <= lower[index] + 1e-10
            at_high = current[index] >= upper[index] - 1e-10
            if (at_low and step[index] < 0.0) or (at_high and step[index] > 0.0):
                newly_active.append(int(index))
        if not newly_active:
            break
        for index in newly_active:
            free[index] = False
            active.add(index)

    step = variable_scale * unit_step
    rotvec = current[10:13]
    radius_active = float(np.linalg.norm(rotvec)) >= rotvec_radius_rad - 1e-10
    tangent_projected = False
    if radius_active and float(rotvec @ step[10:13]) > 0.0:
        normal = rotvec / max(float(np.linalg.norm(rotvec)), _EPS)
        step[10:13] -= normal * float(normal @ step[10:13])
        tangent_projected = True
    return step, tuple(sorted(active)), tangent_projected


def solve_orientation_aware_active_set_dls(
    base_config: Mapping[str, Any],
    *,
    clockwise_orbit_deg: float,
    initial_variables: RelativeWristVariables | None = None,
    settings: RelativeWristDLSSettings | None = None,
    evaluator: TrialEvaluator | None = None,
    joint_bounds: Mapping[str, tuple[float, float]] | None = None,
    evaluation_context: ActiveSetEvaluationContext | None = None,
    check_pose_constraints: bool = True,
    promotion_validator: Callable[[dict[str, Any]], None] | None = None,
) -> RelativeWristDLSResult:
    """Solve the v11 contact gates with projected active-set DLS."""

    policy = RelativeWristPoseSearchPolicy.from_config(base_config)
    if not any(math.isclose(clockwise_orbit_deg, value, abs_tol=_EPS) for value in policy.clockwise_orbit_deg):
        raise ValueError("clockwise_orbit_deg is not a registered stratum")
    resolved = settings or RelativeWristDLSSettings(maximum_iterations=12)
    if evaluation_context is not None and (
        evaluator is not None or joint_bounds is not None
    ):
        raise ValueError(
            "evaluation_context is mutually exclusive with evaluator/joint_bounds"
        )
    if evaluation_context is None:
        if evaluator is None:
            evaluation_context = build_active_set_evaluation_context(base_config)
            if joint_bounds is not None:
                evaluation_context = ActiveSetEvaluationContext(
                    evaluation_context.evaluator,
                    joint_bounds,
                    evaluation_context.full_scene_gate,
                )
        else:
            if joint_bounds is None:
                raise ValueError("an injected evaluator requires explicit joint_bounds")
            evaluation_context = ActiveSetEvaluationContext(
                evaluator, joint_bounds, None
            )
    evaluator = evaluation_context.evaluator
    joint_bounds = evaluation_context.joint_bounds
    full_scene_gate = evaluation_context.full_scene_gate
    full_scene_by_result_id: dict[int, dict[str, Any]] = {}
    thresholds = ActualContactStaticThresholds.from_config(base_config)
    search_bounds = resolve_experiment(dict(base_config)).search_bounds
    if search_bounds.pregrasp_targets_rad is None:
        raise ValueError("active-set actual-contact DLS requires pregrasp bounds")
    precontact_bounds = search_bounds.pregrasp_targets_rad
    resolved_promotion_validator = promotion_validator or validate_config
    lower, upper = _variable_bounds(policy, joint_bounds)
    current = (initial_variables or RelativeWristVariables.from_config(base_config)).as_array()
    reference = current.copy()
    fd = np.asarray(
        [resolved.joint_finite_difference_rad] * 7
        + [resolved.translation_finite_difference_m] * 3
        + [resolved.rotation_finite_difference_rad] * 3
    )
    scale = np.asarray(
        [resolved.joint_step_rad] * 7
        + [resolved.translation_step_m] * 3
        + [resolved.rotation_step_rad] * 3
    )
    rot_radius = math.radians(policy.max_wrist_local_rotvec_norm_deg)

    def boundary(value: np.ndarray) -> tuple[str, ...]:
        return relative_wrist_boundary_violations(
            base_config,
            RelativeWristVariables.from_array(value),
            policy,
            clockwise_orbit_deg=clockwise_orbit_deg,
            joint_bounds=joint_bounds,
            check_pose_constraints=check_pose_constraints,
        )

    def evaluate(value: np.ndarray) -> tuple[dict[str, Any], RelativeWristTrialEvaluation]:
        config = materialize_relative_wrist_candidate(
            base_config,
            RelativeWristVariables.from_array(value),
            clockwise_orbit_deg=clockwise_orbit_deg,
        )
        trial_evaluation, report = evaluate_active_set_candidate(
            evaluation_context, config
        )
        full_scene_by_result_id[id(trial_evaluation.static_result)] = report
        return config, trial_evaluation

    def full_scene_diagnostics(
        evaluation: RelativeWristTrialEvaluation,
    ) -> dict[str, Any]:
        return full_scene_by_result_id.get(
            id(evaluation.static_result),
            {"enabled": True, "passed": False, "reason": "missing_gate_report"},
        )

    def promotion(
        config: dict[str, Any], evaluation: RelativeWristTrialEvaluation
    ) -> tuple[dict[str, Any], bool, list[str]]:
        if not evaluation.safe:
            return config, False, list(evaluation.safety_violations)
        if not bool(
            getattr(evaluation.static_result, "static_geometry_pass", False)
        ):
            return config, False, ["static_geometry_not_passed"]
        try:
            promoted = apply_precontact_solution(config, evaluation.static_result)
            resolved_promotion_validator(promoted)
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            return config, False, [f"{type(error).__name__}: {error}"]
        return promoted, True, []

    initial_boundary = boundary(current)
    if initial_boundary:
        raise ValueError("initial relative-wrist variables violate boundaries: " + ", ".join(initial_boundary))
    current_config, current_eval = evaluate(current)
    diagnostics: dict[str, Any] = {
        "method": "orientation_aware_actual_contact_active_set_dls",
        "variable_names": list(VARIABLE_NAMES),
        "measurement_names": list(MEASUREMENT_NAMES),
        "clockwise_orbit_deg": float(clockwise_orbit_deg),
        "initial_variables": reference.tolist(),
        "iterations": [],
        "central_difference_count": 0,
        "one_sided_difference_count": 0,
        "initial_full_scene_contact_safety": full_scene_diagnostics(current_eval),
    }
    if current_eval.measurement is None:
        diagnostics["initial_safety_violations"] = list(current_eval.safety_violations)
        diagnostics["promotion_config_valid"] = False
        diagnostics["promotion_validation_errors"] = [
            "missing_initial_distal_witness"
        ]
        return RelativeWristDLSResult(
            current_config,
            RelativeWristVariables.from_array(current),
            current_eval.static_result,
            diagnostics,
            "missing_initial_distal_witness",
        )
    _, current_promotion_valid, current_promotion_errors = promotion(
        current_config, current_eval
    )
    residual, _, _ = _gate_residual_and_violations(
        current_eval, thresholds, precontact_bounds
    )
    initial_measurement = list(current_eval.measurement)
    initial_safety = list(current_eval.safety_violations)
    current_merit, current_gate = _merit(
        current_eval,
        thresholds,
        current,
        reference,
        scale,
        resolved.regularization_weight,
        precontact_bounds,
        current_promotion_valid,
    )
    stop_reason = "maximum_iterations"

    for iteration in range(resolved.maximum_iterations):
        if current_promotion_valid:
            stop_reason = "static_geometry_pass"
            break
        jacobian = np.zeros((len(residual), VARIABLE_COUNT))
        fd_records: list[dict[str, Any]] = []
        for column in range(VARIABLE_COUNT):
            samples: dict[int, tuple[np.ndarray, RelativeWristTrialEvaluation]] = {}
            for direction in (1, -1):
                trial = current.copy()
                trial[column] += direction * fd[column]
                reasons = boundary(trial)
                if reasons:
                    fd_records.append({"variable": VARIABLE_NAMES[column], "direction": direction, "reasons": list(reasons)})
                    continue
                _, trial_eval = evaluate(trial)
                if not trial_eval.safe:
                    fd_records.append({"variable": VARIABLE_NAMES[column], "direction": direction, "reasons": list(trial_eval.safety_violations) or ["unsafe_trial_evaluation"]})
                    continue
                trial_residual, _, _ = _gate_residual_and_violations(
                    trial_eval, thresholds, precontact_bounds
                )
                samples[direction] = (trial_residual, trial_eval)
            if 1 in samples and -1 in samples:
                jacobian[:, column] = (samples[1][0] - samples[-1][0]) / (2.0 * fd[column])
                diagnostics["central_difference_count"] += 1
            elif 1 in samples:
                jacobian[:, column] = (samples[1][0] - residual) / fd[column]
                diagnostics["one_sided_difference_count"] += 1
            elif -1 in samples:
                jacobian[:, column] = (residual - samples[-1][0]) / fd[column]
                diagnostics["one_sided_difference_count"] += 1

        proposed, active, tangent = _active_set_step(
            jacobian, residual, current, reference, lower, upper, scale, resolved, rot_radius
        )
        feasible_alpha = _maximum_feasible_alpha(current, proposed, lower, upper, rot_radius)
        line_search: list[dict[str, Any]] = []
        accepted = None
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            alpha = feasible_alpha * fraction
            if alpha <= _EPS:
                continue
            trial = current + alpha * proposed
            reasons = boundary(trial)
            if reasons:
                line_search.append({"alpha": alpha, "accepted": False, "reasons": list(reasons)})
                continue
            trial_config, trial_eval = evaluate(trial)
            if not trial_eval.safe:
                line_search.append({"alpha": alpha, "accepted": False, "reasons": list(trial_eval.safety_violations) or ["unsafe_trial_evaluation"]})
                continue
            _, trial_promotion_valid, trial_promotion_errors = promotion(
                trial_config, trial_eval
            )
            trial_merit, trial_gate = _merit(
                trial_eval,
                thresholds,
                trial,
                reference,
                scale,
                resolved.regularization_weight,
                precontact_bounds,
                trial_promotion_valid,
            )
            improved = trial_merit < current_merit
            line_search.append({
                "alpha": alpha,
                "accepted": improved,
                "reasons": [] if improved else ["gate_merit_not_improved"],
                "merit_key": list(trial_merit),
                "promotion_config_valid": trial_promotion_valid,
                "promotion_validation_errors": trial_promotion_errors,
            })
            if improved:
                accepted = (
                    trial,
                    trial_config,
                    trial_eval,
                    trial_merit,
                    trial_gate,
                    trial_promotion_valid,
                    trial_promotion_errors,
                )
                break
        coordinate_polish: dict[str, Any] = {
            "executed": accepted is None,
            "trials": [],
            "accepted": False,
            "hard_pass_early_stop": False,
        }
        accepted_source = "line_search" if accepted is not None else None
        if accepted is None:
            best_polish = None
            hard_pass_found = False
            for column in range(VARIABLE_COUNT):
                for multiplier in (1, 2, 4):
                    for direction in (1, -1):
                        delta = direction * multiplier * fd[column]
                        trial = current.copy()
                        trial[column] += delta
                        trial_record: dict[str, Any] = {
                            "variable": VARIABLE_NAMES[column],
                            "direction": direction,
                            "multiplier": multiplier,
                            "delta": float(delta),
                            "accepted": False,
                        }
                        reasons = boundary(trial)
                        if reasons:
                            trial_record["reasons"] = list(reasons)
                            coordinate_polish["trials"].append(trial_record)
                            continue
                        trial_config, trial_eval = evaluate(trial)
                        if not trial_eval.safe:
                            trial_record["reasons"] = (
                                list(trial_eval.safety_violations)
                                or ["unsafe_trial_evaluation"]
                            )
                            coordinate_polish["trials"].append(trial_record)
                            continue
                        (
                            _,
                            trial_promotion_valid,
                            trial_promotion_errors,
                        ) = promotion(trial_config, trial_eval)
                        trial_merit, trial_gate = _merit(
                            trial_eval,
                            thresholds,
                            trial,
                            reference,
                            scale,
                            resolved.regularization_weight,
                            precontact_bounds,
                            trial_promotion_valid,
                        )
                        improved = trial_merit < current_merit
                        trial_record.update(
                            {
                                "reasons": (
                                    []
                                    if improved
                                    else ["gate_merit_not_improved"]
                                ),
                                "strictly_improved": improved,
                                "merit_key": list(trial_merit),
                                "promotion_config_valid": (
                                    trial_promotion_valid
                                ),
                                "promotion_validation_errors": (
                                    trial_promotion_errors
                                ),
                            }
                        )
                        coordinate_polish["trials"].append(trial_record)
                        if not improved:
                            continue
                        proposal = (
                            trial,
                            trial_config,
                            trial_eval,
                            trial_merit,
                            trial_gate,
                            trial_promotion_valid,
                            trial_promotion_errors,
                        )
                        if best_polish is None or trial_merit < best_polish[3]:
                            best_polish = proposal
                            coordinate_polish["best_trial_index"] = (
                                len(coordinate_polish["trials"]) - 1
                            )
                        if trial_promotion_valid:
                            # Every promotable hard pass outranks every near
                            # miss.  Stop the deterministic pattern at the
                            # first hard pass instead of spending or cycling.
                            best_polish = proposal
                            hard_pass_found = True
                            coordinate_polish["best_trial_index"] = (
                                len(coordinate_polish["trials"]) - 1
                            )
                            break
                    if hard_pass_found:
                        break
                if hard_pass_found:
                    break
            if best_polish is not None:
                accepted = best_polish
                accepted_source = "coordinate_pattern_polish"
                coordinate_polish["accepted"] = True
                coordinate_polish["hard_pass_early_stop"] = hard_pass_found
                coordinate_polish["trials"][
                    coordinate_polish["best_trial_index"]
                ]["accepted"] = True
        record = {
            "iteration": iteration,
            "gate_before": current_gate,
            "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
            "finite_difference_rejections": fd_records,
            "active_box_variables": [VARIABLE_NAMES[index] for index in active],
            "rotvec_tangent_projected": tangent,
            "maximum_feasible_alpha": feasible_alpha,
            "proposed_step": proposed.tolist(),
            "line_search": line_search,
            "coordinate_pattern_polish": coordinate_polish,
            "accepted": accepted is not None,
            "accepted_source": accepted_source,
        }
        diagnostics["iterations"].append(record)
        if accepted is None:
            stop_reason = "active_set_no_safe_gate_improvement"
            break
        (
            current,
            current_config,
            current_eval,
            current_merit,
            current_gate,
            current_promotion_valid,
            current_promotion_errors,
        ) = accepted
        residual, _, _ = _gate_residual_and_violations(
            current_eval, thresholds, precontact_bounds
        )
        record["gate_after"] = current_gate
        if current_promotion_valid:
            stop_reason = "static_geometry_pass"
            break

    final_variables = RelativeWristVariables.from_array(current)
    final_boundary = boundary(current)
    final_config, final_eval = evaluate(current)
    promoted_config, final_promotion_valid, final_promotion_errors = promotion(
        final_config, final_eval
    )
    if final_promotion_valid:
        final_config = promoted_config
    final_merit, final_gate = _merit(
        final_eval,
        thresholds,
        current,
        reference,
        scale,
        resolved.regularization_weight,
        precontact_bounds,
        final_promotion_valid,
    )
    diagnostics.update({
        "fixed_variable": THUMB_BEND_ACTUATOR,
        "fixed_thumb_actual_rad": float(base_config["grasp_pose"]["nominal_joint_qpos_rad"][THUMB_BEND_ACTUATOR]),
        "initial_measurement": initial_measurement,
        "final_measurement": list(final_eval.measurement),
        "initial_safety_violations": initial_safety,
        "final_safety_violations": list(final_eval.safety_violations),
        "final_variables": current.tolist(),
        "final_gate": final_gate,
        "promotion_config_valid": final_promotion_valid,
        "promotion_validation_errors": final_promotion_errors,
        "final_merit_key": list(final_merit),
        "final_boundary_violations": list(final_boundary),
        "full_scene_contact_safety": full_scene_diagnostics(final_eval),
    })
    return RelativeWristDLSResult(
        final_config, final_variables, final_eval.static_result, diagnostics, stop_reason
    )


__all__ = [
    "ActiveSetEvaluationContext",
    "CachedFullScenePenetrationGate",
    "build_active_set_evaluation_context",
    "classify_full_scene_penetrations",
    "evaluate_active_set_candidate",
    "solve_orientation_aware_active_set_dls",
]
