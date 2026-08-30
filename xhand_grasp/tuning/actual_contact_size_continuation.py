"""Deterministic 1 mm size continuation for actual-contact grasp poses.

The continuation is deliberately a geometry proposal stage.  It starts from
an authenticated measured-qpos source, changes only the registered cube size
and fixed hand geometry, and uses a damped least-squares solve to recover the
three real distal witnesses.  Bridge sizes are retained for provenance but
are never returned as publishable candidates.

The free cube is not welded, made mocap, or reset here.  As in the parent
actual-contact static screen, ``mj_forward`` is used only to evaluate a direct
contact-pose proposal; every promoted candidate must still acquire the free
cube in the full dynamic state machine.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from ..config import ACTIVE_ACTUATORS, validate_config
from ..experiment import resolve_experiment
from ..grasp_pose import controller_id, grasp_pose_id
from ..scene import rpy_degrees_to_rotation_matrix
from .pose_preserving_seed_campaign import canonical_sha256


THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
CONTINUATION_METHOD = "damped_least_squares_distal_witness"
CONTINUATION_RESULT_SCHEMA_VERSION = 1
_CANDIDATE_BASE = 304_000_000_000_000
_TOLERANCE = 1e-12
_TARGET_GAP_M = 0.00015


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _edge_tuple(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a non-empty sequence")
    result = tuple(_finite(item, label) for item in value)
    if not result or any(edge <= 0.0 for edge in result):
        raise ValueError(f"{label} must contain positive sizes")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{label} must be strictly increasing")
    return result


@dataclass(frozen=True, slots=True)
class SizeContinuationPolicy:
    """Validated experiment-owned continuation policy."""

    step_m: float
    bridge_edges_m: tuple[float, ...]
    published_edges_m: tuple[float, ...]
    adjust_hand_root_pose: bool
    adjust_non_thumb_bend_active_joints: bool
    hold_thumb_bend_at_cell_center: bool
    publish_bridge_results: bool

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SizeContinuationPolicy":
        raw = config.get("size_continuation")
        if not isinstance(raw, Mapping):
            raise ValueError("config has no registered size_continuation policy")
        if raw.get("method") != CONTINUATION_METHOD:
            raise ValueError("unsupported size_continuation method")
        step = _finite(raw.get("step_m"), "size_continuation.step_m")
        if step <= 0.0:
            raise ValueError("size_continuation.step_m must be positive")
        bridge = _edge_tuple(
            raw.get("bridge_edges_m"), "size_continuation.bridge_edges_m"
        )
        published = _edge_tuple(
            raw.get("published_edges_m"),
            "size_continuation.published_edges_m",
        )
        all_edges = (*bridge, *published)
        if tuple(sorted(set(all_edges))) != all_edges:
            raise ValueError("bridge and published continuation edges must not overlap")
        for left, right in zip(all_edges, all_edges[1:]):
            if not math.isclose(right - left, step, abs_tol=1e-12):
                raise ValueError("size_continuation edges must use the declared step")
        flags = {
            name: raw.get(name)
            for name in (
                "adjust_hand_root_pose",
                "adjust_non_thumb_bend_active_joints",
                "hold_thumb_bend_at_cell_center",
                "publish_bridge_results",
            )
        }
        if any(not isinstance(value, bool) for value in flags.values()):
            raise ValueError("size_continuation policy flags must be boolean")
        if not flags["adjust_hand_root_pose"]:
            raise ValueError("DLS size continuation must adjust the fixed hand root")
        if not flags["adjust_non_thumb_bend_active_joints"]:
            raise ValueError("DLS size continuation must adjust seven active joints")
        if not flags["hold_thumb_bend_at_cell_center"]:
            raise ValueError("DLS size continuation must hold thumb bend")
        if flags["publish_bridge_results"]:
            raise ValueError("bridge continuation results must not be publishable")

        definition = resolve_experiment(dict(config))
        campaign = definition.actual_contact_grasp_pose_campaign
        if campaign is None:
            raise ValueError("size continuation requires an actual-contact campaign")
        if len(campaign.edges_m) != len(published) or not np.allclose(
            campaign.edges_m, published, atol=1e-12, rtol=0.0
        ):
            raise ValueError("published continuation edges must equal campaign edges")
        return cls(
            step_m=step,
            bridge_edges_m=bridge,
            published_edges_m=published,
            adjust_hand_root_pose=flags["adjust_hand_root_pose"],
            adjust_non_thumb_bend_active_joints=flags[
                "adjust_non_thumb_bend_active_joints"
            ],
            hold_thumb_bend_at_cell_center=flags[
                "hold_thumb_bend_at_cell_center"
            ],
            publish_bridge_results=flags["publish_bridge_results"],
        )

    @property
    def first_bridge_edge_m(self) -> float:
        return self.bridge_edges_m[0]

    @property
    def last_published_edge_m(self) -> float:
        return self.published_edges_m[-1]

    def is_published(self, edge_m: float) -> bool:
        return any(
            math.isclose(edge_m, value, abs_tol=1e-12)
            for value in self.published_edges_m
        )


@dataclass(frozen=True, slots=True)
class SizeContinuationExecution:
    """Continuation records split at the publication boundary."""

    published_records: tuple[dict[str, Any], ...]
    bridge_records: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def continuation_edge_schedule(
    policy: SizeContinuationPolicy, source_edge_m: float
) -> tuple[tuple[float, bool], ...]:
    """Return a prefix-stable 1 mm schedule after ``source_edge_m``.

    Sources below the declared bridge are stepped through hidden pre-bridge
    sizes as well.  This avoids a discontinuous jump from a 63--67 mm
    measured source to 68 mm while still ensuring only 72--90 mm records are
    publishable.
    """

    source = _finite(source_edge_m, "source_edge_m")
    if source >= policy.last_published_edge_m - _TOLERANCE:
        return ()
    source_units = int(round(source / policy.step_m))
    if not math.isclose(source_units * policy.step_m, source, abs_tol=1e-9):
        raise ValueError("source edge must align to the continuation step")
    end_units = int(round(policy.last_published_edge_m / policy.step_m))
    return tuple(
        (
            unit * policy.step_m,
            policy.is_published(unit * policy.step_m),
        )
        for unit in range(source_units + 1, end_units + 1)
    )


def _cube_world_position(config: Mapping[str, Any]) -> np.ndarray:
    from .actual_contact_grasp_pose import _cube_world_position as implementation

    return implementation(config)


def _cube_in_root(config: Mapping[str, Any]) -> np.ndarray:
    rotation = rpy_degrees_to_rotation_matrix(config["hand_pose"]["rpy_deg"])
    root = np.asarray(config["hand_pose"]["translation_m"], dtype=np.float64)
    return rotation.T @ (_cube_world_position(config) - root)


def _source_config(source: Any) -> Mapping[str, Any]:
    if hasattr(source, "config"):
        value = source.config
    elif isinstance(source, Mapping):
        value = source.get("config", source)
    else:
        raise ValueError("continuation source must contain a config")
    if not isinstance(value, Mapping):
        raise ValueError("continuation source config must be a mapping")
    return value


def _source_actual_qpos(source: Any, config: Mapping[str, Any]) -> np.ndarray:
    if hasattr(source, "actual_joint_qpos_rad"):
        raw = source.actual_joint_qpos_rad
    elif isinstance(source, Mapping) and "actual_joint_qpos_rad" in source:
        raw = source["actual_joint_qpos_rad"]
    else:
        raw = config["grasp_pose"]["nominal_joint_qpos_rad"]
        if isinstance(raw, Mapping):
            raw = [raw[name] for name in ACTIVE_ACTUATORS]
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (len(ACTIVE_ACTUATORS),) or not np.isfinite(values).all():
        raise ValueError("continuation source actual qpos must contain eight values")
    return values


def _source_field(source: Any, name: str, default: Any) -> Any:
    if hasattr(source, name):
        return getattr(source, name)
    if isinstance(source, Mapping):
        return source.get(name, default)
    return default


def _clamp_local_pose(definition: Any, local: np.ndarray) -> np.ndarray:
    bounds = definition.search_bounds.cube_position_in_root_m
    low = np.asarray([bounds[axis][0] for axis in ("x", "y", "z")])
    high = np.asarray([bounds[axis][1] for axis in ("x", "y", "z")])
    value = np.clip(np.asarray(local, dtype=np.float64), low, high)
    constraints = definition.far_hand_pose_constraints
    if constraints is None:
        return value
    distance_low, distance_high = constraints.root_cube_distance_m
    for _ in range(4):
        distance = float(np.linalg.norm(value))
        if distance < distance_low - _TOLERANCE:
            value = np.clip(value * (distance_low / max(distance, 1e-12)), low, high)
        elif distance > distance_high + _TOLERANCE:
            value = np.clip(value * (distance_high / distance), low, high)
        else:
            return value
    distance = float(np.linalg.norm(value))
    if not distance_low - 1e-9 <= distance <= distance_high + 1e-9:
        raise ValueError("source hand--cube relation cannot enter registered bounds")
    return value


def prepare_continuation_config(
    template: Mapping[str, Any],
    source: Any,
    target_edge_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map one measured source to a fixed-pose, fixed-mass target size.

    Cube XY/yaw/support placement comes only from the registered campaign.
    The function never samples or copies a source cube world pose.
    """

    policy = SizeContinuationPolicy.from_config(template)
    edge = _finite(target_edge_m, "target_edge_m")
    scheduled = continuation_edge_schedule(
        policy, float(_source_config(source)["cube"]["edge_m"])
    )
    if not any(math.isclose(edge, item[0], abs_tol=1e-12) for item in scheduled):
        raise ValueError("target edge is not on the forward continuation schedule")
    definition = resolve_experiment(dict(template))
    campaign = definition.actual_contact_grasp_pose_campaign
    if campaign is None:
        raise ValueError("size continuation requires an actual-contact campaign")
    source_config = _source_config(source)
    source_qpos = _source_actual_qpos(source, source_config)
    bounds = definition.search_bounds.actuator_targets_rad
    limited = source_qpos.copy()
    for index, name in enumerate(ACTIVE_ACTUATORS):
        limited[index] = float(np.clip(limited[index], *bounds[name]))
    thumb_index = ACTIVE_ACTUATORS.index(THUMB_BEND_ACTUATOR)
    thumb_before = float(source_qpos[thumb_index])
    actual_range = tuple(float(value) for value in template["grasp_pose"]["thumb_actual_range_rad"])
    clipped_thumb = float(np.clip(thumb_before, *actual_range))
    # A continuation record belongs to one registered search cell.  Snap the
    # measured source to the nearest declared actual-angle center once, then
    # hold that exact value throughout every 1 mm DLS step in the chain.
    limited[thumb_index] = float(
        min(
            campaign.thumb_actual_centers_rad,
            key=lambda value: (abs(float(value) - clipped_thumb), float(value)),
        )
    )

    result = copy.deepcopy(dict(template))
    result["cube"].update(
        {
            "edge_m": edge,
            "mass_kg": campaign.fixed_mass_kg,
            "friction": campaign.friction,
            "center_xy_m": list(campaign.cube_center_xy_m),
            "rpy_deg": [0.0, 0.0, campaign.cube_yaw_deg],
            "z_offset_m": 0.0,
        }
    )
    result["hand_pose"]["rpy_deg"] = copy.deepcopy(
        source_config["hand_pose"]["rpy_deg"]
    )
    local = _clamp_local_pose(definition, _cube_in_root(source_config))
    rotation = rpy_degrees_to_rotation_matrix(result["hand_pose"]["rpy_deg"])
    result["hand_pose"]["translation_m"] = (
        _cube_world_position(result) - rotation @ local
    ).tolist()
    nominal = {
        name: float(limited[index]) for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    result["grasp_pose"]["nominal_joint_qpos_rad"] = nominal
    preload = dict(nominal)
    preload_bounds = definition.search_bounds.actuator_targets_rad
    source_preload = source_config.get("control", {}).get(
        "contact_preload_targets_rad", {}
    )
    for name in ACTIVE_ACTUATORS:
        if isinstance(source_preload, Mapping) and name in source_preload:
            preload[name] = float(np.clip(source_preload[name], *preload_bounds[name]))
    preload[THUMB_BEND_ACTUATOR] = float(
        np.clip(
            max(preload[THUMB_BEND_ACTUATOR], nominal[THUMB_BEND_ACTUATOR]),
            *preload_bounds[THUMB_BEND_ACTUATOR],
        )
    )
    result["control"]["contact_preload_targets_rad"] = preload
    result["control"]["manipulation_delta_rad"] = {
        name: 0.0 for name in ACTIVE_ACTUATORS
    }
    diagnostics = {
        "source_edge_m": float(source_config["cube"]["edge_m"]),
        "target_edge_m": edge,
        "cube_in_root_m": local.tolist(),
        "thumb_actual_source_rad": thumb_before,
        "thumb_actual_clipped_to_range_rad": clipped_thumb,
        "thumb_actual_continuation_rad": float(limited[thumb_index]),
        "thumb_actual_clipped": not math.isclose(
            thumb_before, float(limited[thumb_index]), abs_tol=1e-12
        ),
        "cube_pose_sampled": False,
        "cube_world_pose_source": "registered_center_yaw_and_support_height",
        "fixed_mass_kg": campaign.fixed_mass_kg,
        "friction": campaign.friction,
    }
    return result, diagnostics


def _clip_qpos(model: mujoco.MjModel, info: Any, definition: Any, value: np.ndarray) -> np.ndarray:
    result = value.copy()
    for index, name in enumerate(ACTIVE_ACTUATORS):
        actuator_id = model.actuator(name).id
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        low, high = definition.search_bounds.actuator_targets_rad[name]
        if bool(model.jnt_limited[joint_id]):
            low = max(low, float(model.jnt_range[joint_id, 0]))
            high = min(high, float(model.jnt_range[joint_id, 1]))
        result[index] = float(np.clip(result[index], low, high))
    return result


def _materialize(config: Mapping[str, Any], qpos: np.ndarray, root: np.ndarray) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result["hand_pose"]["translation_m"] = root.tolist()
    result["grasp_pose"]["nominal_joint_qpos_rad"] = {
        name: float(qpos[index]) for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    return result


def _dls_step(
    config: Mapping[str, Any],
    *,
    maximum_iterations: int,
) -> tuple[dict[str, Any], Any, dict[str, Any], str | None]:
    # Imports stay local so this independent module can be imported by the
    # parent static-search module without creating an import cycle.
    from .actual_contact_grasp_pose import (
        ActualContactStaticThresholds,
        _uniform_contact_measurement,
        apply_precontact_solution,
        evaluate_direct_actual_contact_pose,
    )
    from ..scene import build_model

    definition = resolve_experiment(dict(config))
    model_config = copy.deepcopy(dict(config))
    # Bridge edges are intentionally outside the published campaign grid.
    # Use the existing non-nominal trial validation context only for model
    # compilation; never persist it in the candidate returned to the caller.
    if not any(
        math.isclose(float(config["cube"]["edge_m"]), edge, abs_tol=1e-12)
        for edge in definition.actual_contact_grasp_pose_campaign.edges_m
    ):
        model_config["run_context"] = {"kind": "robustness_trial"}
    model, info = build_model(model_config)
    data = mujoco.MjData(model)
    limits = ActualContactStaticThresholds.from_config(config)
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    up = -gravity / float(np.linalg.norm(gravity))
    qpos = np.asarray(
        [config["grasp_pose"]["nominal_joint_qpos_rad"][name] for name in ACTIVE_ACTUATORS],
        dtype=np.float64,
    )
    qpos = _clip_qpos(model, info, definition, qpos)
    root = np.asarray(config["hand_pose"]["translation_m"], dtype=np.float64)
    thumb_index = ACTIVE_ACTUATORS.index(THUMB_BEND_ACTUATOR)
    fixed_thumb = float(qpos[thumb_index])
    nonthumb = tuple(index for index in range(len(ACTIVE_ACTUATORS)) if index != thumb_index)
    rotation = rpy_degrees_to_rotation_matrix(config["hand_pose"]["rpy_deg"])
    target = np.asarray((_TARGET_GAP_M,) * 3 + (0.0, 0.0), dtype=np.float64)
    scales = np.asarray((*([0.05] * 7), 0.002, 0.002, 0.002), dtype=np.float64)
    eps = np.asarray((*([2e-4] * 7), 2e-5, 2e-5, 2e-5), dtype=np.float64)

    def evaluate(q: np.ndarray, xyz: np.ndarray) -> tuple[Any, np.ndarray | None]:
        candidate = _materialize(config, q, xyz)
        value = evaluate_direct_actual_contact_pose(
            model, data, info, candidate, thresholds=limits
        )
        return value, _uniform_contact_measurement(value, up)

    result, measured = evaluate(qpos, root)
    if measured is None:
        return dict(config), result, {"iterations": [], "initial_measurement_m": None}, "missing_distal_witness"
    initial = measured.copy()
    iterations: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for iteration in range(maximum_iterations):
        residual = target - measured
        if float(np.max(np.abs(residual))) <= 5e-7:
            break
        jacobian = np.zeros((5, 10), dtype=np.float64)
        for column in range(10):
            trial_qpos = qpos.copy()
            trial_root = root.copy()
            if column < 7:
                index = nonthumb[column]
                trial_qpos[index] += eps[column]
                trial_qpos[thumb_index] = fixed_thumb
                trial_qpos = _clip_qpos(model, info, definition, trial_qpos)
                actual_step = float(trial_qpos[index] - qpos[index])
            else:
                axis = column - 7
                trial_root[axis] += eps[column]
                local = rotation.T @ (_cube_world_position(config) - trial_root)
                local = _clamp_local_pose(definition, local)
                trial_root = _cube_world_position(config) - rotation @ local
                actual_step = float(trial_root[axis] - root[axis])
            _, trial = evaluate(trial_qpos, trial_root)
            if trial is not None and abs(actual_step) > _TOLERANCE:
                jacobian[:, column] = (trial - measured) / actual_step
        scaled = jacobian * scales[np.newaxis, :]
        damping = 2.5e-5
        step = scaled.T @ np.linalg.solve(
            scaled @ scaled.T + damping**2 * np.eye(5), residual
        )
        maximum = float(np.max(np.abs(step)))
        if maximum > 1.0:
            step /= maximum
        proposed = scales * step
        accepted = None
        before = float(np.linalg.norm(residual))
        for line_scale in (1.0, 0.5, 0.25, 0.125):
            trial_qpos = qpos.copy()
            for column, index in enumerate(nonthumb):
                trial_qpos[index] += line_scale * proposed[column]
            trial_qpos[thumb_index] = fixed_thumb
            trial_qpos = _clip_qpos(model, info, definition, trial_qpos)
            trial_root = root + line_scale * proposed[7:10]
            local = rotation.T @ (_cube_world_position(config) - trial_root)
            local = _clamp_local_pose(definition, local)
            trial_root = _cube_world_position(config) - rotation @ local
            trial_result, trial = evaluate(trial_qpos, trial_root)
            if trial is not None and float(np.linalg.norm(target - trial)) + 1e-12 < before:
                accepted = (trial_qpos, trial_root, trial_result, trial, line_scale)
                break
        iterations.append(
            {
                "iteration": iteration,
                "residual_norm_before_m": before,
                "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
                "accepted": accepted is not None,
                "line_scale": None if accepted is None else accepted[4],
            }
        )
        if accepted is None:
            stop_reason = "dls_no_improvement"
            break
        qpos, root, result, measured, _ = accepted

    final_config = _materialize(config, qpos, root)
    final_config["control"]["contact_preload_targets_rad"] = {
        name: float(qpos[index]) for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    result, measured = evaluate(qpos, root)
    details = {
        "method": CONTINUATION_METHOD,
        "optimized_variables": [
            *(name for name in ACTIVE_ACTUATORS if name != THUMB_BEND_ACTUATOR),
            "hand_pose.translation_m.x",
            "hand_pose.translation_m.y",
            "hand_pose.translation_m.z",
        ],
        "fixed_variable": THUMB_BEND_ACTUATOR,
        "fixed_thumb_actual_rad": fixed_thumb,
        "initial_measurement_m": initial.tolist(),
        "final_measurement_m": None if measured is None else measured.tolist(),
        "initial_residual_norm_m": float(np.linalg.norm(target - initial)),
        "final_residual_norm_m": (
            None if measured is None else float(np.linalg.norm(target - measured))
        ),
        "iterations": iterations,
    }
    if result.static_geometry_pass:
        promoted = apply_precontact_solution(final_config, result)
        final_config = promoted
    return final_config, result, details, stop_reason


def continue_actual_qpos_sources(
    template: Mapping[str, Any],
    sources: Sequence[Any],
    *,
    maximum_iterations_per_step: int = 4,
) -> SizeContinuationExecution:
    """Continue measured sources through bridge sizes into 72--90 mm cells."""

    if not sources:
        raise ValueError("size continuation requires a non-empty source set")
    if maximum_iterations_per_step <= 0:
        raise ValueError("maximum_iterations_per_step must be positive")
    policy = SizeContinuationPolicy.from_config(template)
    definition = resolve_experiment(dict(template))
    campaign = definition.actual_contact_grasp_pose_campaign
    assert campaign is not None
    cells = {
        (round(edge, 12), center): edge_index * len(campaign.thumb_actual_centers_rad) + thumb_index
        for edge_index, edge in enumerate(campaign.edges_m)
        for thumb_index, center in enumerate(campaign.thumb_actual_centers_rad)
    }
    published_records: list[dict[str, Any]] = []
    bridge_records: list[dict[str, Any]] = []
    chains: list[dict[str, Any]] = []
    for ordinal, source in enumerate(sources):
        source_config = _source_config(source)
        source_edge = float(source_config["cube"]["edge_m"])
        source_index = int(_source_field(source, "source_index", ordinal))
        pose_id = str(_source_field(source, "pose_id", grasp_pose_id(dict(source_config))))
        source_kind = str(_source_field(source, "source_kind", "authenticated_grasp_success"))
        source_eligible = bool(_source_field(source, "eligible_as_success_evidence", False))
        schedule = continuation_edge_schedule(policy, source_edge)
        current: Any = source
        step_records: list[dict[str, Any]] = []
        chain_stop = "reached_90_mm" if schedule else "source_at_or_above_last_edge"
        for edge, publishable in schedule:
            candidate_id = _CANDIDATE_BASE + ordinal * 100_000 + int(round(edge * 1000.0))
            try:
                proposed, preparation = prepare_continuation_config(template, current, edge)
                final_config, result, dls, step_stop = _dls_step(
                    proposed, maximum_iterations=maximum_iterations_per_step
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                chain_stop = f"continuation_error:{type(error).__name__}:{error}"
                step_records.append(
                    {
                        "candidate_id": candidate_id,
                        "source_index": source_index,
                        "source_pose_id": pose_id,
                        "edge_m": edge,
                        "publishable": publishable,
                        "status": "stopped",
                        "stop_reason": chain_stop,
                    }
                )
                break
            actual_thumb = float(
                final_config["grasp_pose"]["nominal_joint_qpos_rad"][THUMB_BEND_ACTUATOR]
            )
            thumb_center = min(
                campaign.thumb_actual_centers_rad,
                key=lambda value: (abs(value - actual_thumb), value),
            )
            cell_index = cells.get((round(edge, 12), thumb_center), -1)
            cell_id = (
                f"edge_{edge * 1000.0:.0f}mm_thumb_actual_{thumb_center:.2f}rad"
                if publishable
                else f"internal_bridge_{edge * 1000.0:.0f}mm"
            )
            metadata = final_config.setdefault("candidate_metadata", {})
            metadata.update(
                {
                    "stage": "actual_contact_size_continuation",
                    "candidate_id": candidate_id,
                    "parent_source_index": source_index,
                    "parent_pose_id": pose_id,
                    "source_kind": source_kind,
                    "source_success_evidence_inherited": False,
                    "edge_m": edge,
                    "publishable": publishable,
                    "cube_pose_sampled": False,
                    "free_cube_pose_reset_during_scan": False,
                    "thumb_bend_fixed_during_dls": True,
                }
            )
            static_pass = bool(result.static_geometry_pass)
            promotion_error = None
            if static_pass and publishable:
                try:
                    validate_config(final_config)
                except ValueError as error:
                    static_pass = False
                    promotion_error = str(error)
            metrics = result.as_dict()
            metrics["static_geometry_pass"] = static_pass
            metrics["continuation"] = copy.deepcopy(dls)
            if promotion_error is not None:
                metrics["promotion_config_error"] = promotion_error
            record = {
                "candidate_id": candidate_id,
                "parent_source_index": source_index,
                "source_pose_id": pose_id,
                "source_kind": source_kind,
                "source_was_eligible_as_success_evidence": source_eligible,
                "source_success_evidence_inherited": False,
                "cell_index": cell_index,
                "cell_id": cell_id,
                "edge_m": edge,
                "thumb_actual_center_rad": thumb_center,
                "grasp_pose_id": grasp_pose_id(final_config),
                "controller_id": controller_id(final_config),
                "candidate_sha256": canonical_sha256(final_config),
                "config": final_config,
                "static_pass": static_pass,
                "static_metrics": metrics,
                "size_continuation": {
                    **preparation,
                    **dls,
                    "publishable": publishable,
                    "source_success_evidence_inherited": False,
                },
                "status": "continued",
                "stop_reason": step_stop,
            }
            # Importing the rank locally keeps the public module independent
            # while making its records directly consumable by the static pool.
            from .actual_contact_grasp_pose import _static_rank

            record["static_rank"] = list(_static_rank(record))
            step_records.append(
                {
                    "candidate_id": candidate_id,
                    "edge_m": edge,
                    "publishable": publishable,
                    "static_pass": static_pass,
                    "status": "continued",
                    "stop_reason": step_stop,
                }
            )
            (published_records if publishable else bridge_records).append(record)
            # Continue from the geometrically adjusted pose, never from a
            # source success label.  A lost witness or stalled DLS ends this
            # deterministic chain rather than jumping across a size.
            current = {
                "config": final_config,
                "actual_joint_qpos_rad": [
                    final_config["grasp_pose"]["nominal_joint_qpos_rad"][name]
                    for name in ACTIVE_ACTUATORS
                ],
            }
            # A locally stalled line search is diagnostic, not a broken
            # continuation: the next 1 mm object expansion changes the
            # witness residual and can restore a descent direction.  Losing
            # a distal witness is the hard geometric discontinuity.
            if step_stop == "missing_distal_witness" and not static_pass:
                chain_stop = step_stop
                break
        chains.append(
            {
                "source_index": source_index,
                "source_pose_id": pose_id,
                "source_kind": source_kind,
                "source_edge_m": source_edge,
                "step_count": len(step_records),
                "published_step_count": sum(
                    bool(value.get("publishable")) for value in step_records
                ),
                "stop_reason": chain_stop,
                "steps": step_records,
            }
        )
    published_records.sort(key=lambda value: (value["cell_index"], value["candidate_id"]))
    bridge_records.sort(key=lambda value: (value["edge_m"], value["candidate_id"]))
    report = {
        "actual_contact_size_continuation_result_schema_version": (
            CONTINUATION_RESULT_SCHEMA_VERSION
        ),
        "experiment_id": str(template["experiment_id"]),
        "method": CONTINUATION_METHOD,
        "step_m": policy.step_m,
        "source_count": len(sources),
        "published_edge_range_m": [
            policy.published_edges_m[0],
            policy.published_edges_m[-1],
        ],
        "bridge_edges_m": list(policy.bridge_edges_m),
        "publish_bridge_results": False,
        "published_record_count": len(published_records),
        "bridge_record_count": len(bridge_records),
        "published_static_pass_count": sum(
            bool(value["static_pass"]) for value in published_records
        ),
        "chains": chains,
        "cube_pose_sampled": False,
        "source_success_evidence_inherited": False,
    }
    return SizeContinuationExecution(
        published_records=tuple(published_records),
        bridge_records=tuple(bridge_records),
        report=report,
    )


__all__ = [
    "CONTINUATION_METHOD",
    "CONTINUATION_RESULT_SCHEMA_VERSION",
    "SizeContinuationExecution",
    "SizeContinuationPolicy",
    "continuation_edge_schedule",
    "continue_actual_qpos_sources",
    "prepare_continuation_config",
]
