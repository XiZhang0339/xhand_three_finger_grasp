"""Deterministic schema-v6 pose-preserving grasp acquisition search.

This module is deliberately a small orchestration layer.  It searches locally
around an evidence-backed resolved configuration and delegates every dynamics
decision to an injected candidate runner.  Candidate IDs, materialization and
ranking are independent of runner return order, which keeps ``spawn`` worker
counts from changing the selected trajectory.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    resolved_pose_constraint_values,
    validate_config,
)
from ..experiment import ExperimentDefinition, resolve_experiment


EXPERIMENT_ID = "left_opposed_face_palm_down_pose_preserving_grasp"
CandidateRunner = Callable[
    [list[tuple[int, dict[str, Any]]], int], list[dict[str, Any]]
]

_POSE_CHECKS = (
    "object_pose_preserved_until_grasp_acquisition",
    "support_retained_until_grasp_acquisition",
    "no_hand_cube_contact_during_settle",
    "v6_pose_preservation_trace_matches_raw_state",
    "v6_close_profile_trace_matches_config",
)


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _schema_v6_definition(base: Mapping[str, Any]) -> ExperimentDefinition:
    candidate = copy.deepcopy(dict(base))
    validate_config(candidate)
    if int(candidate.get("schema_version", 0)) != 6:
        raise ValueError("pose-preserving tuning requires a schema-v6 config")
    definition = resolve_experiment(candidate)
    if (
        definition.experiment_id != EXPERIMENT_ID
        or definition.tuning_strategy != "pose_preserving_grasp"
        or definition.pose_preservation is None
    ):
        raise ValueError(
            "selected experiment is not the registered pose-preserving grasp"
        )
    return definition


def _target_mapping(values: Mapping[str, Any], label: str) -> dict[str, float]:
    if set(values) != set(ACTIVE_ACTUATORS):
        raise ValueError(f"{label} must contain exactly the eight active actuators")
    return {
        name: _finite(values[name], f"{label}.{name}")
        for name in ACTIVE_ACTUATORS
    }


def _close_profile_mapping(
    values: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    if set(values) != set(ACTIVE_ACTUATORS):
        raise ValueError(
            "close_profile must contain exactly the eight active actuators"
        )
    result: dict[str, dict[str, float]] = {}
    for name in ACTIVE_ACTUATORS:
        interval = values[name]
        if not isinstance(interval, Mapping) or set(interval) != {
            "start_fraction",
            "end_fraction",
        }:
            raise ValueError(
                f"close_profile.{name} must contain exactly start_fraction "
                "and end_fraction"
            )
        start = _finite(
            interval["start_fraction"],
            f"close_profile.{name}.start_fraction",
        )
        end = _finite(
            interval["end_fraction"],
            f"close_profile.{name}.end_fraction",
        )
        if not 0.0 <= start < end <= 1.0:
            raise ValueError(
                f"close_profile.{name} must satisfy "
                "0 <= start_fraction < end_fraction <= 1"
            )
        result[name] = {"start_fraction": start, "end_fraction": end}
    return result


def _hand_pose_mapping(values: Mapping[str, Any]) -> dict[str, list[float]]:
    if set(values) != {"translation_m", "rpy_deg"}:
        raise ValueError(
            "hand_pose must contain exactly translation_m and rpy_deg"
        )
    result: dict[str, list[float]] = {}
    for name in ("translation_m", "rpy_deg"):
        sequence = tuple(values[name])
        if len(sequence) != 3:
            raise ValueError(f"hand_pose.{name} must contain three values")
        result[name] = [
            _finite(value, f"hand_pose.{name}") for value in sequence
        ]
    return result


def materialize_pose_preserving_candidate(
    base: Mapping[str, Any],
    *,
    hand_pose: Mapping[str, Any] | None = None,
    pregrasp_targets_rad: Mapping[str, Any] | None = None,
    grasp_targets_rad: Mapping[str, Any] | None = None,
    close_profile: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> dict[str, Any]:
    """Build one resolved v6 candidate without mutating its source config."""

    definition = _schema_v6_definition(base)
    candidate = copy.deepcopy(dict(base))
    candidate.pop("experiment_status", None)
    candidate.pop("run_context", None)
    source_control = base["control"]
    candidate["hand_pose"] = _hand_pose_mapping(
        base["hand_pose"] if hand_pose is None else hand_pose
    )
    candidate["control"] = {
        "pregrasp_targets_rad": _target_mapping(
            source_control["pregrasp_targets_rad"]
            if pregrasp_targets_rad is None
            else pregrasp_targets_rad,
            "pregrasp_targets_rad",
        ),
        "grasp_targets_rad": _target_mapping(
            source_control["grasp_targets_rad"]
            if grasp_targets_rad is None
            else grasp_targets_rad,
            "grasp_targets_rad",
        ),
        "manipulation_delta_rad": _target_mapping(
            source_control["manipulation_delta_rad"],
            "manipulation_delta_rad",
        ),
        "close_profile": _close_profile_mapping(
            source_control["close_profile"]
            if close_profile is None
            else close_profile
        ),
    }
    candidate_metadata = copy.deepcopy(
        dict(candidate.get("candidate_metadata", {}))
    )
    if metadata is not None:
        candidate_metadata.update(copy.deepcopy(dict(metadata)))
    candidate["candidate_metadata"] = candidate_metadata
    if validator is not None:
        validator(candidate)

    # A catalog label is derived from the physical pose, never sampled as an
    # independent value that could disagree with it.
    resolved = resolved_pose_constraint_values(candidate)
    bands = definition.far_hand_campaign.tilt_band_centers_deg
    tilt = float(resolved["finger_down_tilt_deg"])
    band = min(
        (float(value) for value in bands),
        key=lambda value: (abs(value - tilt), value),
    )
    candidate["candidate_metadata"].update(
        {
            "resolved_finger_down_tilt_deg": tilt,
            "tilt_band_center_deg": band,
            "cube_in_root_m": list(resolved["cube_position_in_root_m"]),
            "root_cube_distance_m": float(resolved["root_cube_distance_m"]),
        }
    )
    return candidate


def _latin_hypercube(
    samples: int, dimensions: int, rng: np.random.Generator
) -> np.ndarray:
    if samples <= 0 or dimensions <= 0:
        raise ValueError("samples and dimensions must be positive")
    result = np.empty((samples, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        result[:, dimension] = (
            rng.permutation(samples) + rng.random(samples)
        ) / samples
    return result


def _perturbed_candidate(
    base: Mapping[str, Any],
    unit: np.ndarray,
    *,
    amplitude_scale: float,
    local_sample_index: int,
) -> dict[str, Any]:
    definition = resolve_experiment(base)
    bounds = definition.search_bounds
    signed = 2.0 * np.asarray(unit, dtype=np.float64) - 1.0
    if signed.shape != (6 + 4 * len(ACTIVE_ACTUATORS),):
        raise ValueError("local search sample has the wrong dimension")
    cursor = 0
    source_pose = base["hand_pose"]
    translation_radius = (0.0015, 0.0012, 0.0015)
    rpy_radius = (0.4, 0.6, 0.5)
    hand_pose = {"translation_m": [], "rpy_deg": []}
    for index, radius in enumerate(translation_radius):
        hand_pose["translation_m"].append(
            float(source_pose["translation_m"][index])
            + amplitude_scale * radius * signed[cursor]
        )
        cursor += 1
    for index, radius in enumerate(rpy_radius):
        hand_pose["rpy_deg"].append(
            float(source_pose["rpy_deg"][index])
            + amplitude_scale * radius * signed[cursor]
        )
        cursor += 1

    pregrasp: dict[str, float] = {}
    grasp: dict[str, float] = {}
    pregrasp_bounds = bounds.pregrasp_targets_rad
    assert pregrasp_bounds is not None
    for name in ACTIVE_ACTUATORS:
        lower, upper = pregrasp_bounds[name]
        radius = 0.05 * (upper - lower)
        pregrasp[name] = float(
            np.clip(
                float(base["control"]["pregrasp_targets_rad"][name])
                + amplitude_scale * radius * signed[cursor],
                lower,
                upper,
            )
        )
        cursor += 1
    for name in ACTIVE_ACTUATORS:
        lower, upper = bounds.actuator_targets_rad[name]
        radius = 0.035 * (upper - lower)
        grasp[name] = float(
            np.clip(
                float(base["control"]["grasp_targets_rad"][name])
                + amplitude_scale * radius * signed[cursor],
                lower,
                upper,
            )
        )
        cursor += 1

    profile: dict[str, dict[str, float]] = {}
    for name in ACTIVE_ACTUATORS:
        source = base["control"]["close_profile"][name]
        start = float(source["start_fraction"]) + (
            amplitude_scale * 0.08 * signed[cursor]
        )
        cursor += 1
        end = float(source["end_fraction"]) + (
            amplitude_scale * 0.08 * signed[cursor]
        )
        cursor += 1
        start = float(np.clip(start, 0.0, 0.98))
        end = float(np.clip(end, start + 0.02, 1.0))
        profile[name] = {"start_fraction": start, "end_fraction": end}

    return materialize_pose_preserving_candidate(
        base,
        hand_pose=hand_pose,
        pregrasp_targets_rad=pregrasp,
        grasp_targets_rad=grasp,
        close_profile=profile,
        metadata={
            "search_stage": "local_acquisition_search",
            "local_sample_index": int(local_sample_index),
            "local_amplitude_scale": float(amplitude_scale),
        },
    )


def generate_pose_preserving_candidates(
    base: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    amplitude_scale: float = 1.0,
) -> tuple[dict[str, Any], ...]:
    """Return ``count`` stable-order candidates, including the base first."""

    _schema_v6_definition(base)
    candidate_count = _positive_int(count, "count")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    amplitude = _finite(amplitude_scale, "amplitude_scale")
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError("amplitude_scale must be within [0, 1]")
    configs = [
        materialize_pose_preserving_candidate(
            base,
            metadata={
                "search_stage": "local_acquisition_search",
                "local_sample_index": 0,
                "local_amplitude_scale": 0.0,
            },
        )
    ]
    if candidate_count == 1:
        return tuple(configs)
    dimensions = 6 + 4 * len(ACTIVE_ACTUATORS)
    matrix = _latin_hypercube(
        candidate_count - 1,
        dimensions,
        np.random.default_rng(seed),
    )
    for index, row in enumerate(matrix, start=1):
        # A direct root-pose perturbation can cross the coupled root/cube shell
        # at its edge.  Deterministically shrink toward the valid source pose
        # instead of dropping a sample and changing IDs or worker batches.
        for shrink in (1.0, 0.5, 0.25, 0.125, 0.0):
            try:
                candidate = _perturbed_candidate(
                    base,
                    row,
                    amplitude_scale=amplitude * shrink,
                    local_sample_index=index,
                )
            except ValueError:
                continue
            configs.append(candidate)
            break
        else:  # pragma: no cover - the zero-amplitude source is validated.
            raise RuntimeError("unable to materialize a valid local candidate")
    return tuple(configs)


def _summary(result: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = result.get("summary", {})
    return summary if isinstance(summary, Mapping) else {}


def _metrics(result: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = _summary(result).get("metrics", {})
    return metrics if isinstance(metrics, Mapping) else {}


def _rank_number(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def grasp_succeeded(result: Mapping[str, Any]) -> bool:
    stage = _summary(result).get("stage_status", {})
    return bool(
        isinstance(stage, Mapping) and stage.get("grasp_success", False)
    )


def pose_preservation_succeeded(result: Mapping[str, Any]) -> bool:
    checks = _summary(result).get("checks", {})
    return bool(
        isinstance(checks, Mapping)
        and all(bool(checks.get(name, False)) for name in _POSE_CHECKS)
    )


def acquisition_succeeded(result: Mapping[str, Any]) -> bool:
    return grasp_succeeded(result) and pose_preservation_succeeded(result)


def _pose_margins(result: Mapping[str, Any]) -> tuple[float, float]:
    metrics = _metrics(result)
    pose = metrics.get("pose_preservation", {})
    config = result.get("config", {})
    settings = (
        config.get("pose_preservation", {})
        if isinstance(config, Mapping)
        else {}
    )
    if not isinstance(pose, Mapping) or not isinstance(settings, Mapping):
        return (-math.inf, -math.inf)
    translation_limit = _rank_number(
        pose.get("translation_limit_m", settings.get("max_translation_m")),
        math.inf,
    )
    orientation_limit = _rank_number(
        pose.get(
            "orientation_limit_deg",
            settings.get("max_orientation_drift_deg"),
        ),
        math.inf,
    )
    translation = _rank_number(pose.get("max_translation_m"), math.inf)
    orientation = _rank_number(
        pose.get("max_orientation_drift_deg"), math.inf
    )
    translation_margin = (
        (translation_limit - translation) / translation_limit
        if math.isfinite(translation_limit) and translation_limit > 0.0
        else -math.inf
    )
    orientation_margin = (
        (orientation_limit - orientation) / orientation_limit
        if math.isfinite(orientation_limit) and orientation_limit > 0.0
        else -math.inf
    )
    return (translation_margin, orientation_margin)


def pose_preserving_candidate_rank(
    result: Mapping[str, Any],
) -> tuple[float, ...]:
    """Worker-independent acquisition rank with weakest evidence first."""

    try:
        candidate_id = int(result["candidate_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("candidate result requires an integer candidate_id") from error
    summary = _summary(result)
    metrics = _metrics(result)
    stage = summary.get("stage_status", {})
    manipulation = bool(
        isinstance(stage, Mapping)
        and stage.get("manipulation_success", False)
    )
    full = bool(summary.get("passed", False))
    duties = metrics.get("verify_target_face_effective_duty", {})
    balanced_duty = tuple(
        sorted(
            _rank_number(
                duties.get(finger, 0.0)
                if isinstance(duties, Mapping)
                else 0.0,
                0.0,
            )
            for finger in ACTIVE_FINGERS
        )
    )
    translation_margin, orientation_margin = _pose_margins(result)
    pose_metrics = metrics.get("pose_preservation", {})
    onset_span = (
        _rank_number(
            pose_metrics.get("distal_contact_onset_span_s"), math.inf
        )
        if isinstance(pose_metrics, Mapping)
        else math.inf
    )
    stage_name = result.get("search_stage")
    if not isinstance(stage_name, str):
        config = result.get("config", {})
        metadata = (
            config.get("candidate_metadata", {})
            if isinstance(config, Mapping)
            else {}
        )
        stage_name = (
            metadata.get("search_stage", "")
            if isinstance(metadata, Mapping)
            else ""
        )
    stage_priority = {
        "acquisition_discovery": 0.0,
        "acquisition_local_refinement": 1.0,
        "acquisition_confirmation": 2.0,
        "acquisition_local_probe": 3.0,
    }.get(str(stage_name), 0.0)
    return (
        float(acquisition_succeeded(result)),
        float(grasp_succeeded(result)),
        float(pose_preservation_succeeded(result)),
        float(full),
        float(manipulation),
        min(translation_margin, orientation_margin),
        translation_margin,
        orientation_margin,
        _rank_number(metrics.get("verify_effective_finger_count"), 0.0),
        _rank_number(
            metrics.get("verify_max_simultaneous_effective_finger_count"),
            0.0,
        ),
        *balanced_duty,
        _rank_number(
            metrics.get("verify_max_consecutive_all_gate_steps"), 0.0
        ),
        -onset_span,
        -_rank_number(
            metrics.get("peak_total_distal_contact_force_n"), math.inf
        ),
        -_rank_number(
            metrics.get("actuator_saturation_fraction"), math.inf
        ),
        stage_priority,
        -float(candidate_id),
    )


def deterministic_rank_pose_preserving_results(
    results: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    materialized = tuple(results)
    ids = tuple(int(result["candidate_id"]) for result in materialized)
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be unique")
    return tuple(
        sorted(
            materialized,
            key=pose_preserving_candidate_rank,
            reverse=True,
        )
    )


@dataclass(frozen=True)
class PosePreservingTuningBudget:
    candidate_count: int
    refine_seed_count: int
    refine_per_seed: int
    final_candidate_count: int
    perturbation_count: int

    def __post_init__(self) -> None:
        _positive_int(self.candidate_count, "candidate_count")
        _nonnegative_int(self.refine_seed_count, "refine_seed_count")
        _nonnegative_int(self.refine_per_seed, "refine_per_seed")
        _positive_int(self.final_candidate_count, "final_candidate_count")
        _nonnegative_int(self.perturbation_count, "perturbation_count")

    def as_dict(self) -> dict[str, int]:
        return {
            "candidate_count": self.candidate_count,
            "refine_seed_count": self.refine_seed_count,
            "refine_per_seed": self.refine_per_seed,
            "final_candidate_count": self.final_candidate_count,
            "perturbation_count": self.perturbation_count,
        }


def _resolve_budget(
    *,
    samples: int,
    refine_top: int,
    refine_per: int,
    dynamic_candidate_count: int | None,
    local_refine_seed_count: int | None,
    local_refine_per_seed: int | None,
    final_candidate_count: int | None,
    perturbations_per_final: int | None,
) -> PosePreservingTuningBudget:
    return PosePreservingTuningBudget(
        candidate_count=_positive_int(
            samples if dynamic_candidate_count is None else dynamic_candidate_count,
            "dynamic_candidate_count",
        ),
        refine_seed_count=_nonnegative_int(
            refine_top
            if local_refine_seed_count is None
            else local_refine_seed_count,
            "local_refine_seed_count",
        ),
        refine_per_seed=_nonnegative_int(
            refine_per
            if local_refine_per_seed is None
            else local_refine_per_seed,
            "local_refine_per_seed",
        ),
        final_candidate_count=_positive_int(
            max(1, refine_top)
            if final_candidate_count is None
            else final_candidate_count,
            "final_candidate_count",
        ),
        perturbation_count=_nonnegative_int(
            0 if perturbations_per_final is None else perturbations_per_final,
            "perturbations_per_final",
        ),
    )


def _tag_config(config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    tagged = copy.deepcopy(dict(config))
    metadata = dict(tagged.get("candidate_metadata", {}))
    metadata["search_stage"] = stage
    tagged["candidate_metadata"] = metadata
    validate_config(tagged)
    return tagged


def _run_stage(
    configs: Sequence[Mapping[str, Any]],
    *,
    next_candidate_id: int,
    workers: int,
    run_candidates: CandidateRunner,
    stage: str,
) -> tuple[list[dict[str, Any]], int]:
    tagged = [_tag_config(config, stage) for config in configs]
    payloads = [
        (next_candidate_id + index, config)
        for index, config in enumerate(tagged)
    ]
    submitted = {candidate_id: config for candidate_id, config in payloads}
    raw = run_candidates(payloads, workers) if payloads else []
    results = [copy.deepcopy(dict(result)) for result in raw]
    received = [int(result.get("candidate_id", -1)) for result in results]
    if len(results) != len(payloads) or set(received) != set(submitted):
        raise RuntimeError(f"{stage} runner did not preserve candidate IDs")
    if len(received) != len(set(received)):
        raise RuntimeError(f"{stage} runner returned duplicate candidate IDs")
    for result in results:
        candidate_id = int(result["candidate_id"])
        if result.get("config") != submitted[candidate_id]:
            raise RuntimeError(f"{stage} runner rebound candidate configuration")
        result["search_stage"] = stage
        result["material_policy"] = "constant_density_nominal"
    results.sort(key=lambda result: int(result["candidate_id"]))
    return results, next_candidate_id + len(payloads)


def tune_pose_preserving_grasp(
    config: dict[str, Any],
    *,
    workers: int,
    seed: int,
    run_candidates: CandidateRunner,
    samples: int,
    refine_top: int,
    refine_per: int,
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
    """Search locally for a stable grasp that preserves the initial cube pose."""

    if config.get("run_context") is not None:
        raise ValueError("schema-v6 tuning requires a canonical nominal config")
    definition = _schema_v6_definition(config)
    worker_count = _positive_int(workers, "workers")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    effective = _resolve_budget(
        samples=samples,
        refine_top=refine_top,
        refine_per=refine_per,
        dynamic_candidate_count=dynamic_candidate_count,
        local_refine_seed_count=local_refine_seed_count,
        local_refine_per_seed=local_refine_per_seed,
        final_candidate_count=final_candidate_count,
        perturbations_per_final=perturbations_per_final,
    )
    next_id = 0
    discovery_configs = generate_pose_preserving_candidates(
        config,
        count=effective.candidate_count,
        seed=seed,
    )
    discovery_results, next_id = _run_stage(
        discovery_configs,
        next_candidate_id=next_id,
        workers=worker_count,
        run_candidates=run_candidates,
        stage="acquisition_discovery",
    )

    discovery_ranked = deterministic_rank_pose_preserving_results(
        discovery_results
    )
    refine_configs: list[dict[str, Any]] = []
    refine_parent_count = min(
        effective.refine_seed_count,
        len(discovery_ranked),
    )
    if effective.refine_per_seed:
        for parent_index, parent in enumerate(
            discovery_ranked[:refine_parent_count]
        ):
            refine_configs.extend(
                generate_pose_preserving_candidates(
                    parent["config"],
                    count=effective.refine_per_seed,
                    seed=seed + 610_000_000 + parent_index * 10_007,
                    amplitude_scale=0.4,
                )
            )
    refine_results, next_id = _run_stage(
        refine_configs,
        next_candidate_id=next_id,
        workers=worker_count,
        run_candidates=run_candidates,
        stage="acquisition_local_refinement",
    )

    prefinal = tuple(discovery_results + refine_results)
    ranked_prefinal = deterministic_rank_pose_preserving_results(prefinal)
    finalist_count = min(effective.final_candidate_count, len(ranked_prefinal))
    final_configs = [
        copy.deepcopy(result["config"])
        for result in ranked_prefinal[:finalist_count]
    ]
    final_results, next_id = _run_stage(
        final_configs,
        next_candidate_id=next_id,
        workers=worker_count,
        run_candidates=run_candidates,
        stage="acquisition_confirmation",
    )
    all_results = discovery_results + refine_results + final_results
    ranked = list(deterministic_rank_pose_preserving_results(all_results))
    if not ranked:  # pragma: no cover - candidate_count is positive.
        raise RuntimeError("pose-preserving tuning produced no dynamics results")
    best = copy.deepcopy(ranked[0])
    near_candidates = [
        result for result in ranked if not acquisition_succeeded(result)
    ]
    near_miss = copy.deepcopy(near_candidates[0]) if near_candidates else None

    probe_results: list[dict[str, Any]] = []
    if effective.perturbation_count:
        probe_configs = generate_pose_preserving_candidates(
            best["config"],
            count=effective.perturbation_count,
            seed=seed + 620_000_000,
            amplitude_scale=0.2,
        )
        probe_results, next_id = _run_stage(
            probe_configs,
            next_candidate_id=next_id,
            workers=worker_count,
            run_candidates=run_candidates,
            stage="acquisition_local_probe",
        )
    probe_passes = sum(acquisition_succeeded(item) for item in probe_results)
    probe_record = {
        "candidate_id": int(best["candidate_id"]),
        "passes": int(probe_passes),
        "trial_count": len(probe_results),
        "seed": int(seed + 620_000_000),
        "trials": copy.deepcopy(probe_results),
    }
    best["local_perturbation_probe"] = copy.deepcopy(probe_record)
    stable_acquisition = acquisition_succeeded(best)
    full_success = bool(_summary(best).get("passed", False))
    manipulation_success = bool(
        _summary(best).get("stage_status", {}).get(
            "manipulation_success", False
        )
    )
    if full_success:
        classification = "pose_preserving_full_success"
        stop_reason = "full_hard_pass_found"
    elif stable_acquisition:
        classification = "pose_preserving_stable_grasp_only"
        stop_reason = "stable_pose_preserving_grasp_found"
    else:
        classification = "pose_preserving_grasp_near_miss"
        stop_reason = "no_pose_preserving_stable_grasp"
    best["config"]["experiment_status"] = {
        "classification": classification,
        "passed": full_success,
        "grasp_success": grasp_succeeded(best),
        "pose_preservation_success": pose_preservation_succeeded(best),
        "manipulation_success": manipulation_success,
        "full_success": full_success,
        "robustness_passed": False,
        "stop_reason": stop_reason,
        "note": (
            "A stable grasp was acquired without changing the initial cube pose."
            if stable_acquisition
            else "No stable pose-preserving grasp was found; this is the best simulated near miss."
        ),
    }
    top_candidates = [copy.deepcopy(best)]
    top_candidates.extend(
        copy.deepcopy(result)
        for result in ranked
        if int(result["candidate_id"]) != int(best["candidate_id"])
    )
    top_candidates = top_candidates[:20]
    legacy = copy.deepcopy(dict(legacy_parameters or {}))
    legacy.update(
        {
            "samples": int(samples),
            "refine_top": int(refine_top),
            "refine_per": int(refine_per),
        }
    )
    return {
        "campaign_kind": "pose_preserving_grasp",
        "experiment_id": EXPERIMENT_ID,
        "seed": int(seed),
        "workers": worker_count,
        "formal_budget": definition.far_hand_campaign.budget_config(),
        "effective_budget": {
            **effective.as_dict(),
            "kinematic_samples_per_pitch": kinematic_samples_per_pitch,
            "fallback_physics_count": fallback_physics_count,
            "fallback_kinematic_samples_per_pitch": (
                fallback_kinematic_samples_per_pitch
            ),
        },
        "legacy_parameters": legacy,
        "stage_counts": {
            "acquisition_discovery": len(discovery_results),
            "acquisition_local_refinement": len(refine_results),
            "acquisition_confirmation": len(final_results),
            "acquisition_local_probe": len(probe_results),
        },
        "candidate_count": len(all_results),
        "simulation_count": len(all_results) + len(probe_results),
        "perturbation_probe_count": len(probe_results),
        "passing_candidates": sum(
            acquisition_succeeded(result) for result in all_results
        ),
        "full_passing_candidates": sum(
            bool(_summary(result).get("passed", False))
            for result in all_results
        ),
        "grasp_success": stable_acquisition,
        "pose_preservation_success": pose_preservation_succeeded(best),
        "manipulation_success": manipulation_success,
        "nominal_success": full_success,
        "robustness_success": False,
        "nominal_passed": full_success,
        "robust_passed": False,
        "perturbation_passes": int(probe_passes),
        "required_perturbation_passes": len(probe_results),
        "best": best,
        "best_attempt": copy.deepcopy(best),
        "near_miss": near_miss,
        "top_candidates": top_candidates,
        "selected_band_candidates": [copy.deepcopy(best)],
        "local_perturbation_probes": (
            [copy.deepcopy(probe_record)] if probe_results else []
        ),
        "stop_reason": stop_reason,
    }


__all__ = [
    "EXPERIMENT_ID",
    "PosePreservingTuningBudget",
    "acquisition_succeeded",
    "deterministic_rank_pose_preserving_results",
    "generate_pose_preserving_candidates",
    "grasp_succeeded",
    "materialize_pose_preserving_candidate",
    "pose_preservation_succeeded",
    "pose_preserving_candidate_rank",
    "tune_pose_preserving_grasp",
]
