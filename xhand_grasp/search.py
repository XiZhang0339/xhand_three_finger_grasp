"""Deterministic tuning and robustness search for the cube-lift task."""

from __future__ import annotations

import copy
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from .config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    SEARCH_TARGET_BOUNDS,
    validate_config,
)
from .experiment import resolve_experiment
from .simulation import run_simulation


def _finite_rank_number(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _balanced_finger_metric(metrics: dict[str, Any], name: str) -> tuple[float, ...]:
    values = metrics.get(name, {})
    if not isinstance(values, dict):
        values = {}
    return tuple(
        sorted(
            _finite_rank_number(values.get(finger, 0.0))
            for finger in ACTIVE_FINGERS
        )
    )


def v3_verify_near_miss_rank(metrics: dict[str, Any]) -> tuple[float, ...]:
    """Rank raw VERIFY evidence without rewarding an absence of contact.

    Per-finger values are sorted weakest-first, so adding evidence for a
    missing finger is more valuable than concentrating force on one finger.
    Fixed-size aggregate gate terms keep the key independent of mapping order.
    """

    component_duty = metrics.get("verify_gate_component_duty", {})
    if isinstance(component_duty, dict) and component_duty:
        component_values = tuple(
            _finite_rank_number(value) for value in component_duty.values()
        )
        component_minimum = min(component_values)
        component_mean = sum(component_values) / len(component_values)
    else:
        component_minimum = 0.0
        component_mean = 0.0
    return (
        _finite_rank_number(metrics.get("verify_effective_finger_count", 0)),
        _finite_rank_number(
            metrics.get("verify_max_simultaneous_effective_finger_count", 0)
        ),
        *_balanced_finger_metric(metrics, "verify_target_face_effective_duty"),
        _finite_rank_number(
            metrics.get("verify_target_face_simultaneous_duty", 0.0)
        ),
        *_balanced_finger_metric(metrics, "verify_peak_target_face_force_n"),
        *_balanced_finger_metric(metrics, "verify_peak_tactile_n"),
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


def candidate_rank(result: dict[str, Any]) -> tuple[float, ...]:
    summary = result["summary"]
    metrics = summary["metrics"]
    candidate_id = int(result["candidate_id"])
    stage_status = summary.get("stage_status")
    if isinstance(stage_status, dict):
        grasp_success = bool(stage_status.get("grasp_success", False))
        manipulation_success = bool(
            stage_status.get("manipulation_success", False)
        )
        if not grasp_success:
            return (
                float(summary["passed"]),
                0.0,
                float(manipulation_success),
                *v3_verify_near_miss_rank(metrics),
                float(sum(summary.get("checks", {}).values())),
                -float(metrics.get("material_active_nondistal_duty", math.inf)),
                float(metrics.get("median_lift_m", -math.inf)),
                -float(metrics.get("peak_total_distal_contact_force_n", math.inf)),
                -float(metrics.get("actuator_saturation_fraction", math.inf)),
                -candidate_id,
            )
        operation_duty = metrics.get("operation_target_face_contact_duty", {})
        balanced_operation_duty = sorted(
            float(operation_duty.get(finger, 0.0)) for finger in ACTIVE_FINGERS
        )
        minimum_margin = -math.inf
        if summary["passed"]:
            margin_kwargs: dict[str, Any] = {
                "contact_topology": result["config"].get("contact_topology")
            }
            if result["config"].get("contact_alignment") is not None:
                margin_kwargs["contact_alignment"] = result["config"][
                    "contact_alignment"
                ]
            if result["config"].get("pose_constraints") is not None:
                margin_kwargs["pose_constraints"] = result["config"][
                    "pose_constraints"
                ]
            margins = normalized_acceptance_margins(
                metrics,
                result["config"]["acceptance"],
                **margin_kwargs,
            )
            minimum_margin = min(margins.values())
        return (
            float(summary["passed"]),
            1.0,
            float(manipulation_success),
            float(metrics.get("grasp_gate_final_consecutive_steps", 0)),
            *balanced_operation_duty,
            float(metrics.get("operation_target_face_simultaneous_duty", 0.0)),
            float(minimum_margin),
            float(sum(summary.get("checks", {}).values())),
            -float(metrics.get("material_active_nondistal_duty", math.inf)),
            float(metrics.get("median_lift_m", -math.inf)),
            -float(metrics.get("peak_total_distal_contact_force_n", math.inf)),
            -float(metrics.get("actuator_saturation_fraction", math.inf)),
            -candidate_id,
        )
    if summary["passed"]:
        topology = result.get("config", {}).get("contact_topology")
        v2_rank_metrics = {
            "minimum_lift_m",
            "orientation_drift_deg",
            "end_linear_speed_m_s",
            "max_penetration_m",
            "inactive_joint_max_abs_rad",
            "simultaneous_contact_duty",
            "peak_tactile_n",
            "max_palm_down_angle_deg",
            "target_face_contact_duty",
            "target_face_simultaneous_duty",
            "material_off_target_duty",
            "material_off_target_longest_run_s",
            "material_active_nondistal_duty",
            "material_active_nondistal_longest_run_s",
        }
        if topology is not None and v2_rank_metrics <= metrics.keys():
            margin_kwargs = {"contact_topology": topology}
            if result["config"].get("contact_alignment") is not None:
                margin_kwargs["contact_alignment"] = result["config"][
                    "contact_alignment"
                ]
            if result["config"].get("pose_constraints") is not None:
                margin_kwargs["pose_constraints"] = result["config"][
                    "pose_constraints"
                ]
            margins = normalized_acceptance_margins(
                metrics,
                result["config"]["acceptance"],
                **margin_kwargs,
            )
            return (
                1.0,
                min(margins.values()),
                -metrics["peak_total_distal_contact_force_n"],
                -metrics["orientation_drift_deg"],
                -metrics["actuator_saturation_fraction"],
                -metrics["hold_height_span_m"],
                metrics["median_lift_m"],
                -candidate_id,
            )
        return (
            1.0,
            -metrics["peak_total_distal_contact_force_n"],
            -metrics["actuator_saturation_fraction"],
            -metrics["hold_height_span_m"],
            metrics["median_lift_m"],
            -candidate_id,
        )
    if "target_face_contact_duty" in metrics:
        balanced_duties = sorted(
            float(value) for value in metrics["target_face_contact_duty"].values()
        )
        target_peaks = sorted(
            float(value) for value in metrics["peak_target_face_force_n"].values()
        )
        return (
            0.0,
            float(sum(summary["checks"].values())),
            *balanced_duties,
            float(metrics["target_face_simultaneous_duty"]),
            -float(metrics["material_off_target_duty"]),
            -float(metrics["material_active_nondistal_duty"]),
            *target_peaks,
            metrics["median_lift_m"],
            -float(metrics["forbidden_contact_steps"]),
            -candidate_id,
        )
    return (
        0.0,
        float(sum(summary["checks"].values())),
        metrics["median_lift_m"],
        min(metrics["contact_duty"].values()),
        -float(metrics["forbidden_contact_steps"]),
        -candidate_id,
    )


def latin_hypercube(
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


def _sample_candidate(
    base: dict[str, Any],
    unit: np.ndarray,
    *,
    include_physics: bool,
) -> dict[str, Any]:
    candidate = copy.deepcopy(base)
    cursor = 0
    translation_ranges = ((-0.006, 0.006), (-0.005, 0.005), (-0.005, 0.005))
    for index, (lower_delta, upper_delta) in enumerate(translation_ranges):
        base_value = float(base["hand_pose"]["translation_m"][index])
        candidate["hand_pose"]["translation_m"][index] = (
            base_value + lower_delta + unit[cursor] * (upper_delta - lower_delta)
        )
        cursor += 1
    for index in range(3):
        base_value = float(base["hand_pose"]["rpy_deg"][index])
        candidate["hand_pose"]["rpy_deg"][index] = (
            base_value - 5.0 + 10.0 * unit[cursor]
        )
        cursor += 1
    for field in ("pregrasp_targets_rad", "final_targets_rad"):
        for name in ACTIVE_ACTUATORS:
            lower, upper = SEARCH_TARGET_BOUNDS[name]
            candidate["control"][field][name] = lower + unit[cursor] * (
                upper - lower
            )
            cursor += 1
    if include_physics:
        candidate["cube"]["edge_m"] = 0.026 + unit[cursor] * 0.008
        cursor += 1
        candidate["cube"]["mass_kg"] = 0.010 + unit[cursor] * 0.020
        cursor += 1
        candidate["cube"]["friction"] = 0.4 + unit[cursor] * 0.8
        cursor += 1
    validate_config(candidate)
    return candidate


def _perturbed_cube_cases(
    config: dict[str, Any], *, count: int, seed: int
) -> list[dict[str, Any]]:
    """Return deterministic local initial-state and material perturbations.

    A separate generator is used for vertical clearance so adding that axis does
    not silently alter the established x/y, attitude, mass, and friction cases.
    """

    if count < 0:
        raise ValueError("perturbation count must be non-negative")
    parameters = resolve_experiment(config).robustness
    if min(parameters.mass_scale) <= 0.0:
        raise ValueError("robustness mass_scale must remain positive")
    if (
        float(config["cube"]["friction"])
        + min(parameters.friction_delta)
        <= 0.0
    ):
        raise ValueError("robustness friction perturbations must remain positive")
    rng = np.random.default_rng(seed)
    vertical_rng = np.random.default_rng(seed + 1_000_003)
    cases: list[dict[str, Any]] = []
    for _ in range(count):
        case = copy.deepcopy(config)
        case["cube"]["center_xy_m"][0] += float(
            rng.uniform(*parameters.position_xy_delta_m)
        )
        case["cube"]["center_xy_m"][1] += float(
            rng.uniform(*parameters.position_xy_delta_m)
        )
        case["cube"]["rpy_deg"][0] += float(
            rng.uniform(*parameters.rpy_delta_deg)
        )
        case["cube"]["rpy_deg"][1] += float(
            rng.uniform(*parameters.rpy_delta_deg)
        )
        case["cube"]["rpy_deg"][2] += float(
            rng.uniform(*parameters.rpy_delta_deg)
        )
        case["cube"]["mass_kg"] *= float(rng.uniform(*parameters.mass_scale))
        case["cube"]["friction"] = (
            float(case["cube"]["friction"])
            + float(rng.uniform(*parameters.friction_delta))
        )
        case["cube"]["z_offset_m"] = float(
            case["cube"].get("z_offset_m", 0.0)
        ) + float(vertical_rng.uniform(*parameters.z_offset_delta_m))
        validate_config(case)
        cases.append(case)
    return cases


def _run_candidate(payload: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    """Run one candidate; kept at module scope for spawn pickling."""

    candidate_id, config = payload
    try:
        summary = run_simulation(config)
    except Exception as error:  # Search must preserve failed candidates and diagnostics.
        summary = {
            "passed": False,
            "failed_checks": ["simulation_error"],
            "checks": {"simulation_error": False},
            "metrics": {
                "median_lift_m": -math.inf,
                "contact_duty": {finger: 0.0 for finger in ACTIVE_FINGERS},
                "forbidden_contact_steps": math.inf,
                "peak_total_distal_contact_force_n": math.inf,
                "actuator_saturation_fraction": 1.0,
                "hold_height_span_m": math.inf,
            },
            "error": f"{type(error).__name__}: {error}",
        }
    return {"candidate_id": candidate_id, "config": config, "summary": summary}


def _run_candidates(
    candidates: list[tuple[int, dict[str, Any]]], workers: int
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_run_candidate(candidate) for candidate in candidates]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        return list(executor.map(_run_candidate, candidates, chunksize=1))


def tune(
    config: dict[str, Any],
    *,
    samples: int,
    refine_top: int,
    refine_per: int,
    workers: int,
    seed: int,
    kinematic_samples_per_pitch: int | None = None,
    dynamic_candidate_count: int | None = None,
    local_refine_seed_count: int | None = None,
    local_refine_per_seed: int | None = None,
    final_candidate_count: int | None = None,
    perturbations_per_final: int | None = None,
    fallback_physics_count: int | None = None,
    fallback_kinematic_samples_per_pitch: int | None = None,
) -> dict[str, Any]:
    if int(config.get("schema_version", 1)) >= 2:
        definition = resolve_experiment(config)
        if getattr(definition, "tuning_strategy", "default") == (
            "pose_preserving_grasp"
        ):
            from .tuning.pose_preserving_grasp import (
                tune_pose_preserving_grasp,
            )

            return tune_pose_preserving_grasp(
                config,
                workers=workers,
                seed=seed,
                run_candidates=_run_candidates,
                samples=samples,
                refine_top=refine_top,
                refine_per=refine_per,
                kinematic_samples_per_pitch=kinematic_samples_per_pitch,
                dynamic_candidate_count=dynamic_candidate_count,
                local_refine_seed_count=local_refine_seed_count,
                local_refine_per_seed=local_refine_per_seed,
                final_candidate_count=final_candidate_count,
                perturbations_per_final=perturbations_per_final,
                fallback_physics_count=fallback_physics_count,
                fallback_kinematic_samples_per_pitch=(
                    fallback_kinematic_samples_per_pitch
                ),
                legacy_parameters={
                    "samples": samples,
                    "refine_top": refine_top,
                    "refine_per": refine_per,
                },
            )
        if getattr(definition, "tuning_strategy", "default") == "far_hand_fingertip":
            from .tuning.far_hand_fingertip import tune_far_hand_fingertip

            return tune_far_hand_fingertip(
                config,
                workers=workers,
                seed=seed,
                run_candidates=_run_candidates,
                kinematic_samples_per_pitch=kinematic_samples_per_pitch,
                dynamic_candidate_count=dynamic_candidate_count,
                local_refine_seed_count=local_refine_seed_count,
                local_refine_per_seed=local_refine_per_seed,
                final_candidate_count=final_candidate_count,
                perturbations_per_final=perturbations_per_final,
                fallback_physics_count=fallback_physics_count,
                fallback_kinematic_samples_per_pitch=(
                    fallback_kinematic_samples_per_pitch
                ),
                legacy_parameters={
                    "samples": samples,
                    "refine_top": refine_top,
                    "refine_per": refine_per,
                },
            )
        if getattr(definition, "tuning_strategy", "default") == "aligned_contacts":
            from .aligned_contacts_tuning import tune_aligned_contacts

            return tune_aligned_contacts(
                config,
                workers=workers,
                seed=seed,
                run_candidates=_run_candidates,
                kinematic_samples_per_pitch=kinematic_samples_per_pitch,
                dynamic_candidate_count=dynamic_candidate_count,
                local_refine_seed_count=local_refine_seed_count,
                local_refine_per_seed=local_refine_per_seed,
                final_candidate_count=final_candidate_count,
                perturbations_per_final=perturbations_per_final,
                fallback_physics_count=fallback_physics_count,
                fallback_kinematic_samples_per_pitch=(
                    fallback_kinematic_samples_per_pitch
                ),
                legacy_parameters={
                    "samples": samples,
                    "refine_top": refine_top,
                    "refine_per": refine_per,
                },
            )
        if getattr(definition, "tuning_strategy", "default") == "relative_pose_rescue":
            from .pose_rescue import tune_relative_pose_rescue

            return tune_relative_pose_rescue(
                config,
                workers=workers,
                seed=seed,
                run_candidates=_run_candidates,
                kinematic_samples_per_pitch=kinematic_samples_per_pitch,
                dynamic_candidate_count=dynamic_candidate_count,
                local_refine_seed_count=local_refine_seed_count,
                local_refine_per_seed=local_refine_per_seed,
                final_candidate_count=final_candidate_count,
                perturbations_per_final=perturbations_per_final,
                fallback_physics_count=fallback_physics_count,
                fallback_kinematic_samples_per_pitch=(
                    fallback_kinematic_samples_per_pitch
                ),
                legacy_parameters={
                    "samples": samples,
                    "refine_top": refine_top,
                    "refine_per": refine_per,
                },
            )
        if definition.size_campaign is not None:
            if definition.control_protocol is not None:
                from .larger_cube_grasp_search import (
                    tune_larger_cube_grasp_then_lift,
                )

                return tune_larger_cube_grasp_then_lift(
                    config,
                    workers=workers,
                    seed=seed,
                    run_candidates=_run_candidates,
                    rank_candidate=candidate_rank,
                    kinematic_samples_per_pitch=kinematic_samples_per_pitch,
                    dynamic_candidate_count=dynamic_candidate_count,
                    local_refine_seed_count=local_refine_seed_count,
                    local_refine_per_seed=local_refine_per_seed,
                    final_candidate_count=final_candidate_count,
                    perturbations_per_final=perturbations_per_final,
                    fallback_physics_count=fallback_physics_count,
                    fallback_kinematic_samples_per_pitch=(
                        fallback_kinematic_samples_per_pitch
                    ),
                    perturb_cases=_perturbed_cube_cases,
                )
            from .large_cube_search import tune_large_cube

            return tune_large_cube(
                config,
                workers=workers,
                seed=seed,
                run_candidates=_run_candidates,
                rank_candidate=candidate_rank,
                kinematic_samples_per_pitch=kinematic_samples_per_pitch,
                dynamic_candidate_count=dynamic_candidate_count,
                local_refine_seed_count=local_refine_seed_count,
                local_refine_per_seed=local_refine_per_seed,
                final_candidate_count=final_candidate_count,
                perturbations_per_final=perturbations_per_final,
                fallback_physics_count=fallback_physics_count,
                fallback_kinematic_samples_per_pitch=(
                    fallback_kinematic_samples_per_pitch
                ),
                perturb_cases=_perturbed_cube_cases,
            )
        from .v2_search import tune_opposed_face

        return tune_opposed_face(
            config,
            workers=workers,
            seed=seed,
            run_candidates=_run_candidates,
            rank_candidate=candidate_rank,
            kinematic_samples_per_pitch=kinematic_samples_per_pitch,
            dynamic_candidate_count=dynamic_candidate_count,
            local_refine_seed_count=local_refine_seed_count,
            local_refine_per_seed=local_refine_per_seed,
            final_candidate_count=final_candidate_count,
            perturbations_per_final=perturbations_per_final,
            fallback_physics_count=fallback_physics_count,
            fallback_kinematic_samples_per_pitch=(
                fallback_kinematic_samples_per_pitch
            ),
            perturb_cases=_perturbed_cube_cases,
        )
    if refine_top <= 0 or refine_per <= 0:
        raise ValueError("refine_top and refine_per must be positive")
    rng = np.random.default_rng(seed)
    dimensions = 6 + 2 * len(ACTIVE_ACTUATORS)
    matrix = latin_hypercube(samples, dimensions, rng)
    candidates: list[tuple[int, dict[str, Any]]] = [(0, copy.deepcopy(config))]
    candidates.extend(
        (index + 1, _sample_candidate(config, row, include_physics=False))
        for index, row in enumerate(matrix)
    )
    results = _run_candidates(candidates, workers)
    next_id = len(candidates)

    if not any(result["summary"]["passed"] for result in results):
        physics_dimensions = dimensions + 3
        physics_matrix = latin_hypercube(samples, physics_dimensions, rng)
        physics_candidates = [
            (next_id + index, _sample_candidate(config, row, include_physics=True))
            for index, row in enumerate(physics_matrix)
        ]
        results.extend(_run_candidates(physics_candidates, workers))
        next_id += len(physics_candidates)

    ranked = sorted(results, key=candidate_rank, reverse=True)
    probe_parents = ranked[: min(refine_top, len(ranked))]
    probe_payloads: list[tuple[int, dict[str, Any]]] = []
    probe_owner: dict[int, int] = {}
    for parent in probe_parents:
        cases = _perturbed_cube_cases(parent["config"], count=refine_per, seed=seed)
        for case in cases:
            probe_owner[next_id] = int(parent["candidate_id"])
            probe_payloads.append((next_id, case))
            next_id += 1
    probe_runs = _run_candidates(probe_payloads, workers)

    probes_by_parent: dict[int, list[dict[str, Any]]] = {
        int(parent["candidate_id"]): [] for parent in probe_parents
    }
    for probe in probe_runs:
        probes_by_parent[probe_owner[int(probe["candidate_id"])]].append(probe)

    probe_results: list[dict[str, Any]] = []
    for parent in probe_parents:
        parent_id = int(parent["candidate_id"])
        trials = probes_by_parent[parent_id]
        probe_results.append(
            {
                "candidate_id": parent_id,
                "passes": sum(trial["summary"]["passed"] for trial in trials),
                "trial_count": len(trials),
                "trials": [
                    {
                        "passed": trial["summary"]["passed"],
                        "failed_checks": trial["summary"]["failed_checks"],
                        "cube": trial["config"]["cube"],
                        "metrics": trial["summary"]["metrics"],
                    }
                    for trial in trials
                ],
            }
        )
    probe_lookup = {record["candidate_id"]: record for record in probe_results}

    def selection_rank(parent: dict[str, Any]) -> tuple[float, ...]:
        summary = parent["summary"]
        metrics = summary["metrics"]
        probe = probe_lookup.get(int(parent["candidate_id"]), {"passes": -1})
        return (
            float(summary["passed"]),
            float(probe["passes"]),
            -float(metrics.get("peak_total_distal_contact_force_n", math.inf)),
            -float(metrics.get("orientation_drift_deg", math.inf)),
            -float(metrics.get("hold_height_span_m", math.inf)),
            -float(metrics.get("actuator_saturation_fraction", math.inf)),
            float(metrics.get("median_lift_m", -math.inf)),
            -int(parent["candidate_id"]),
        )

    best_parent = max(ranked, key=selection_rank)
    best = copy.deepcopy(best_parent)
    best["local_perturbation_probe"] = probe_lookup.get(
        int(best_parent["candidate_id"]),
        {
            "candidate_id": int(best_parent["candidate_id"]),
            "passes": 0,
            "trial_count": 0,
        },
    )
    return {
        "seed": seed,
        "samples": samples,
        "refine_top": refine_top,
        "refine_per": refine_per,
        "workers": workers,
        "candidate_count": len(results),
        "perturbation_probe_count": len(probe_runs),
        "simulation_count": len(results) + len(probe_runs),
        "passing_candidates": sum(result["summary"]["passed"] for result in results),
        "best": best,
        "top_candidates": ranked[: min(20, len(ranked))],
        "local_perturbation_probes": probe_results,
    }


def robustness_cases(
    config: dict[str, Any], seed: int | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    definition = resolve_experiment(config)
    parameters = definition.robustness
    effective_seed = parameters.seed if seed is None else int(seed)
    if getattr(definition, "tuning_strategy", "default") == "far_hand_fingertip":
        from .tuning.far_hand_fingertip import (
            generate_far_hand_perturbation_configs,
            generate_far_hand_size_cases,
        )

        grid = generate_far_hand_size_cases(config, definition=definition)
        perturbations = generate_far_hand_perturbation_configs(
            config,
            count=parameters.perturbation_count,
            seed=effective_seed,
            definition=definition,
        )
        return grid, perturbations
    if getattr(definition, "tuning_strategy", "default") == "aligned_contacts":
        from .aligned_contacts_tuning import generate_aligned_perturbation_configs

        perturbations = generate_aligned_perturbation_configs(
            config,
            count=parameters.perturbation_count,
            seed=effective_seed,
            definition=definition,
        )
        if len(perturbations) != parameters.perturbation_count:
            raise RuntimeError(
                "aligned-contact robustness generator returned "
                f"{len(perturbations)} cases; expected "
                f"{parameters.perturbation_count}"
            )
        return [], perturbations
    grid, _ = _robustness_grid_cases(config, definition)
    perturbations = _perturbed_cube_cases(
        config,
        count=parameters.perturbation_count,
        seed=effective_seed,
    )
    return grid, perturbations


def _robustness_grid_cases(
    config: dict[str, Any], definition: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize a grid and parallel, JSON-safe case-family metadata.

    Legacy experiments retain their Cartesian edge/mass/friction grid.  A
    registered size campaign instead receives the declared 75 constant-density
    cases followed by the 25 fixed-mass geometry controls.  Keeping metadata
    parallel to the runnable configs avoids leaking reporting-only keys into a
    config later selected as ``hardest_passing_config``.
    """

    parameters = definition.robustness
    families = getattr(parameters, "case_families", None)
    campaign = getattr(definition, "size_campaign", None)
    grid: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []

    if families is None:
        for edge_m in parameters.edge_m:
            for mass_kg in parameters.mass_kg:
                for friction in parameters.friction:
                    case = copy.deepcopy(config)
                    case["cube"]["edge_m"] = float(edge_m)
                    case["cube"]["mass_kg"] = float(mass_kg)
                    case["cube"]["friction"] = float(friction)
                    grid.append(case)
                    metadata.append({})
        return grid, metadata

    if campaign is None:
        raise ValueError("robustness case families require a size_campaign")
    edge_window = families.edge_window_m(float(config["cube"]["edge_m"]))

    for edge_m in edge_window:
        reference_mass = campaign.constant_density_mass_kg(edge_m)
        for density_scale in families.density_mass_scales:
            for friction in families.friction:
                mass_kg = reference_mass * float(density_scale)
                case = copy.deepcopy(config)
                case["cube"]["edge_m"] = float(edge_m)
                case["cube"]["mass_kg"] = mass_kg
                case["cube"]["friction"] = float(friction)
                validate_config(case)
                grid.append(case)
                metadata.append(
                    {
                        "case_family": "constant_density",
                        "density_scale": float(density_scale),
                        "reference_density_kg_m3": campaign.density_kg_m3,
                    }
                )

    for edge_m in edge_window:
        for friction in families.friction:
            case = copy.deepcopy(config)
            case["cube"]["edge_m"] = float(edge_m)
            case["cube"]["mass_kg"] = float(families.fixed_mass_kg)
            case["cube"]["friction"] = float(friction)
            validate_config(case)
            grid.append(case)
            metadata.append(
                {
                    "case_family": families.fixed_mass_case_label,
                    "density_scale": None,
                    "reference_density_kg_m3": campaign.density_kg_m3,
                }
            )

    if len(grid) != families.total_case_count:
        raise AssertionError(
            f"generated {len(grid)} robustness cases, expected "
            f"{families.total_case_count}"
        )
    return grid, metadata


def normalized_acceptance_margins(
    metrics: dict[str, Any],
    acceptance: dict[str, Any],
    *,
    contact_topology: dict[str, Any] | None = None,
    contact_alignment: dict[str, Any] | None = None,
    pose_constraints: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return dimensionless distance to every quantitative hard threshold."""

    lower_bounds = {
        "median_lift_m": "median_lift_m",
        "minimum_lift_m": "minimum_lift_m",
    }
    upper_bounds = {
        "hold_height_span_m": "max_height_span_m",
        "orientation_drift_deg": "max_orientation_drift_deg",
        "end_linear_speed_m_s": "max_end_linear_speed_m_s",
        "max_penetration_m": "max_penetration_m",
        "inactive_joint_max_abs_rad": "inactive_joint_abs_max_rad",
    }
    margins = {
        metric_name: (float(metrics[metric_name]) - float(acceptance[threshold_name]))
        / float(acceptance[threshold_name])
        for metric_name, threshold_name in lower_bounds.items()
    }
    margins.update(
        {
            metric_name: (
                float(acceptance[threshold_name]) - float(metrics[metric_name])
            )
            / float(acceptance[threshold_name])
            for metric_name, threshold_name in upper_bounds.items()
        }
    )
    duty_threshold = float(acceptance["finger_contact_duty"])
    duty_scale = max(1e-12, 1.0 - duty_threshold)
    for finger in ACTIVE_FINGERS:
        margins[f"{finger}_contact_duty"] = (
            float(metrics["contact_duty"][finger]) - duty_threshold
        ) / duty_scale
    simultaneous_threshold = float(acceptance["simultaneous_contact_duty"])
    margins["simultaneous_contact_duty"] = (
        float(metrics["simultaneous_contact_duty"]) - simultaneous_threshold
    ) / max(1e-12, 1.0 - simultaneous_threshold)
    touch_threshold = float(acceptance["touch_force_min_n"])
    for finger in ACTIVE_FINGERS:
        margins[f"{finger}_tactile"] = (
            float(metrics["peak_tactile_n"][finger]) - touch_threshold
        ) / touch_threshold
    if "max_palm_down_angle_deg" in acceptance and "max_palm_down_angle_deg" in metrics:
        threshold = float(acceptance["max_palm_down_angle_deg"])
        margins["max_palm_down_angle_deg"] = (
            threshold - float(metrics["max_palm_down_angle_deg"])
        ) / max(threshold, 1e-12)
    if "target_face_contact_duty" in metrics:
        for finger in ACTIVE_FINGERS:
            margins[f"{finger}_target_face_contact_duty"] = (
                float(metrics["target_face_contact_duty"][finger]) - duty_threshold
            ) / duty_scale
        margins["target_face_simultaneous_duty"] = (
            float(metrics["target_face_simultaneous_duty"])
            - simultaneous_threshold
        ) / max(1e-12, 1.0 - simultaneous_threshold)
    if "operation_target_face_contact_duty" in metrics:
        for finger in ACTIVE_FINGERS:
            margins[f"operation_{finger}_target_face_contact_duty"] = (
                float(metrics["operation_target_face_contact_duty"][finger])
                - duty_threshold
            ) / duty_scale
        margins["operation_target_face_simultaneous_duty"] = (
            float(metrics["operation_target_face_simultaneous_duty"])
            - simultaneous_threshold
        ) / max(1e-12, 1.0 - simultaneous_threshold)
    if contact_topology is not None:
        maximum_duty = float(contact_topology["max_material_off_target_duty"])
        maximum_run = float(contact_topology["max_material_off_target_run_s"])

        def upper_margin(metric_name: str, maximum: float) -> None:
            margins[metric_name] = (
                maximum - float(metrics[metric_name])
            ) / max(abs(maximum), 1e-12)

        upper_margin("material_off_target_duty", maximum_duty)
        upper_margin("material_off_target_longest_run_s", maximum_run)
        if bool(contact_topology["forbid_active_nondistal"]):
            upper_margin("material_active_nondistal_duty", maximum_duty)
            upper_margin("material_active_nondistal_longest_run_s", maximum_run)
    if contact_alignment is not None and "contact_alignment" in metrics:
        required_duty = float(contact_alignment["operation_aligned_duty"])
        observed_duty = float(
            metrics["contact_alignment"]["operation"]["aligned_duty"]
        )
        margins["operation_contact_height_aligned_duty"] = (
            observed_duty - required_duty
        ) / max(1e-12, 1.0 - required_duty)
    if pose_constraints is not None:
        pose_metric_names = {
            "finger_down_tilt_deg": "finger_down_tilt_deg",
            "palm_plane_ground_angle_deg": "palm_plane_ground_angle_deg",
            "palm_press_depth_m": "palm_press_depth_m",
        }
        for metric_name, constraint_name in pose_metric_names.items():
            if metric_name not in metrics:
                continue
            lower, upper = (
                float(value) for value in pose_constraints[constraint_name]
            )
            scale = max(abs(lower), abs(upper), upper - lower, 1e-12)
            observed = metrics[metric_name]
            margins[f"{metric_name}_lower"] = (
                float(observed["min"]) - lower
            ) / scale
            margins[f"{metric_name}_upper"] = (
                upper - float(observed["max"])
            ) / scale
        if (
            "root_cube_distance_m" in pose_constraints
            and "root_cube_center_distance_m" in metrics
        ):
            lower, upper = (
                float(value)
                for value in pose_constraints["root_cube_distance_m"]
            )
            observed_distance = float(
                metrics["root_cube_center_distance_m"]["initial"]
            )
            scale = max(abs(lower), abs(upper), upper - lower, 1e-12)
            margins["initial_root_cube_distance_lower"] = (
                observed_distance - lower
            ) / scale
            margins["initial_root_cube_distance_upper"] = (
                upper - observed_distance
            ) / scale
    return margins


def _aligned_contact_robustness(
    config: dict[str, Any],
    *,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    """Run the schema-v4 nominal case and its registered 50-case envelope.

    The aligned-contact campaign has no Cartesian material grid.  Its declared
    robustness contract is the independent object-pose, hand-pose and material
    perturbation envelope owned by the v4 experiment implementation.
    """

    from .aligned_contacts_tuning import generate_aligned_perturbation_configs
    from .larger_cube_grasp_search import full_succeeded

    # ``parameter_override_run`` deliberately relaxes the registered material
    # and pose envelope so that the Viewer can investigate arbitrary nearby
    # configurations.  Such a run is useful diagnostics, but it must never be
    # promoted to the campaign's canonical robustness claim.  Removing the
    # context makes validation enforce the registered v4 envelope again, so
    # callers can explicitly save a newly canonicalized resolved config first.
    if config.get("run_context") is not None:
        raise ValueError(
            "schema-v4 campaign robustness requires a canonical config; "
            "parameter_override_run and robustness_trial configs are not eligible"
        )
    validate_config(config)

    definition = resolve_experiment(config)
    campaign = definition.aligned_contact_campaign
    if campaign is None:
        raise ValueError("aligned-contact robustness requires an aligned campaign")

    nominal_result = _run_candidates([(0, copy.deepcopy(config))], workers)[0]
    nominal_summary = nominal_result["summary"]
    perturbation_configs = generate_aligned_perturbation_configs(
        config,
        count=definition.robustness.perturbation_count,
        seed=seed,
        definition=definition,
    )
    expected_count = int(definition.robustness.perturbation_count)
    if len(perturbation_configs) != expected_count:
        raise RuntimeError(
            "aligned-contact robustness generator returned "
            f"{len(perturbation_configs)} cases; expected {expected_count}"
        )
    perturbation_results = _run_candidates(
        [
            (index + 1, perturbation)
            for index, perturbation in enumerate(perturbation_configs)
        ],
        workers,
    )
    if len(perturbation_results) != expected_count:
        raise RuntimeError(
            "aligned-contact robustness runner returned "
            f"{len(perturbation_results)} results; expected {expected_count}"
        )

    perturbation_records: list[dict[str, Any]] = []
    from .artifacts import resolved_run_config

    for trial, result in enumerate(perturbation_results):
        summary = result["summary"]
        trial_config = result["config"]
        full_success = full_succeeded(result)
        perturbation_records.append(
            {
                "trial": trial,
                "candidate_id": int(result["candidate_id"]),
                "passed": full_success,
                "hard_constraints_passed": bool(summary.get("passed", False)),
                "failed_checks": copy.deepcopy(summary.get("failed_checks", [])),
                "error": summary.get("error"),
                "checks": copy.deepcopy(summary.get("checks", {})),
                "stage_status": copy.deepcopy(summary.get("stage_status", {})),
                "metrics": copy.deepcopy(summary.get("metrics", {})),
                "cube": copy.deepcopy(trial_config["cube"]),
                "hand_pose": copy.deepcopy(trial_config["hand_pose"]),
                "resolved_perturbations": copy.deepcopy(
                    trial_config.get("candidate_metadata", {}).get(
                        "resolved_perturbations", {}
                    )
                ),
                "resolved_trial_config": resolved_run_config(
                    trial_config, summary
                ),
            }
        )

    nominal_hard_constraints_passed = bool(nominal_summary.get("passed", False))
    nominal_full_success = full_succeeded(nominal_result)
    nominal_edge = float(config["cube"]["edge_m"])
    nominal_is_constant_density = math.isclose(
        float(config["cube"]["mass_kg"]),
        campaign.constant_density_mass_kg(nominal_edge),
        rel_tol=1e-12,
        abs_tol=1e-15,
    )
    nominal_friction_matches = math.isclose(
        float(config["cube"]["friction"]),
        campaign.friction,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    nominal_passed = bool(
        nominal_hard_constraints_passed
        and nominal_full_success
        and nominal_is_constant_density
        and nominal_friction_matches
    )
    perturbation_passes = sum(
        bool(record["passed"]) for record in perturbation_records
    )
    required_passes = int(definition.robustness.required_pass_count)
    return {
        "campaign_kind": "aligned_contacts",
        "seed": int(seed),
        "robustness_manifest": {
            "seed": int(seed),
            "perturbation_count": expected_count,
            "required_pass_count": int(
                definition.robustness.required_pass_count
            ),
            "perturbation_envelope": campaign.perturbation_envelope.as_config(),
        },
        "nominal_passed": nominal_passed,
        "nominal_hard_constraints_passed": nominal_hard_constraints_passed,
        "nominal_full_success": nominal_full_success,
        "nominal_is_constant_density": nominal_is_constant_density,
        "nominal_friction_matches": nominal_friction_matches,
        "nominal_summary": nominal_summary,
        "grid_case_count": 0,
        "grid_passes": 0,
        "hardest_passing_grid_case": None,
        "hardest_passing_constant_density_grid_case": None,
        "hardest_passing_fixed_20g_control_grid_case": None,
        "grid": [],
        "perturbation_trial_count": len(perturbation_records),
        "perturbation_passes": perturbation_passes,
        "required_perturbation_passes": required_passes,
        "robust_passed": bool(
            nominal_passed and perturbation_passes >= required_passes
        ),
        "perturbations": perturbation_records,
    }


def robustness(
    config: dict[str, Any], *, workers: int, seed: int | None = None
) -> dict[str, Any]:
    definition = resolve_experiment(config)
    parameters = definition.robustness
    effective_seed = parameters.seed if seed is None else int(seed)
    if getattr(definition, "tuning_strategy", "default") == "far_hand_fingertip":
        from .tuning.far_hand_fingertip import run_far_hand_robustness

        return run_far_hand_robustness(
            config,
            workers=workers,
            seed=effective_seed,
            run_candidates=_run_candidates,
        )
    if getattr(definition, "tuning_strategy", "default") == "aligned_contacts":
        return _aligned_contact_robustness(
            config,
            workers=workers,
            seed=effective_seed,
        )
    campaign = getattr(definition, "size_campaign", None)
    schema_version = int(config.get("schema_version", 1))
    if schema_version >= 3 and campaign is not None:
        edge_m = float(config["cube"]["edge_m"])
        mass_kg = float(config["cube"]["mass_kg"])
        expected_mass_kg = campaign.constant_density_mass_kg(edge_m)
        if not math.isclose(
            mass_kg, expected_mass_kg, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "schema-v3 campaign robustness requires a constant-density "
                f"nominal config: cube.mass_kg={mass_kg!r}, expected "
                f"{expected_mass_kg!r} for cube.edge_m={edge_m!r}"
            )
        friction = float(config["cube"]["friction"])
        if not math.isclose(
            friction,
            campaign.discovery_friction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "schema-v3 campaign robustness requires nominal friction: "
                f"cube.friction={friction!r}, expected "
                f"{campaign.discovery_friction!r}"
            )
    nominal_result = _run_candidates([(0, copy.deepcopy(config))], workers)[0]
    nominal_summary = nominal_result["summary"]

    grid, grid_metadata = _robustness_grid_cases(config, definition)
    perturbations = _perturbed_cube_cases(
        config,
        count=parameters.perturbation_count,
        seed=effective_seed,
    )
    grid_results = _run_candidates(list(enumerate(grid, start=1)), workers)
    offset = 1 + len(grid_results)
    perturbation_results = _run_candidates(
        [(offset + index, case) for index, case in enumerate(perturbations)], workers
    )
    grid_summary = []
    for grid_index, result in enumerate(grid_results):
        cube = result["config"]["cube"]
        record = {
            "grid_index": grid_index,
            "edge_mm": round(cube["edge_m"] * 1000),
            "mass_g": round(cube["mass_kg"] * 1000),
            "edge_m": float(cube["edge_m"]),
            "mass_kg": float(cube["mass_kg"]),
            "friction": cube["friction"],
            "passed": result["summary"]["passed"],
            "failed_checks": result["summary"]["failed_checks"],
            "metrics": result["summary"]["metrics"],
            "normalized_acceptance_margin": None,
            "limiting_metric": None,
        }
        if schema_version >= 3:
            record["checks"] = copy.deepcopy(
                result["summary"].get("checks", {})
            )
            record["stage_status"] = copy.deepcopy(
                result["summary"].get("stage_status", {})
            )
        record.update(grid_metadata[grid_index])
        if record["passed"]:
            margin_kwargs = {"contact_topology": config.get("contact_topology")}
            if config.get("contact_alignment") is not None:
                margin_kwargs["contact_alignment"] = config["contact_alignment"]
            if config.get("pose_constraints") is not None:
                margin_kwargs["pose_constraints"] = config["pose_constraints"]
            margins = normalized_acceptance_margins(
                result["summary"]["metrics"],
                config["acceptance"],
                **margin_kwargs,
            )
            limiting_metric = min(margins, key=margins.get)
            record["normalized_acceptance_margin"] = margins[limiting_metric]
            record["limiting_metric"] = limiting_metric
        grid_summary.append(record)
    perturbation_summary = []
    for index, result in enumerate(perturbation_results):
        record = {
            "trial": index,
            "passed": result["summary"]["passed"],
            "failed_checks": result["summary"]["failed_checks"],
            "cube": result["config"]["cube"],
            "metrics": result["summary"]["metrics"],
        }
        if schema_version >= 3:
            record["checks"] = copy.deepcopy(
                result["summary"].get("checks", {})
            )
            record["stage_status"] = copy.deepcopy(
                result["summary"].get("stage_status", {})
            )
        perturbation_summary.append(record)
    perturbation_passes = sum(result["passed"] for result in perturbation_summary)
    passing_grid = [record for record in grid_summary if record["passed"]]
    density_passing_grid = [
        record
        for record in passing_grid
        if record.get("case_family", "constant_density") == "constant_density"
    ]
    fixed_mass_label = (
        parameters.case_families.fixed_mass_case_label
        if parameters.case_families is not None
        else "fixed_mass_control"
    )
    control_passing_grid = [
        record
        for record in passing_grid
        if record.get("case_family") == fixed_mass_label
    ]
    hardest_passing = (
        min(
            density_passing_grid,
            key=lambda record: (
                record["normalized_acceptance_margin"],
                record["grid_index"],
            ),
        )
        if density_passing_grid
        else None
    )
    hardest_control = (
        min(
            control_passing_grid,
            key=lambda record: (
                record["normalized_acceptance_margin"],
                record["grid_index"],
            ),
        )
        if control_passing_grid
        else None
    )
    nominal_hard_constraints_passed = bool(nominal_summary["passed"])
    nominal_is_constant_density = bool(
        campaign is None
        or math.isclose(
            float(config["cube"]["mass_kg"]),
            campaign.constant_density_mass_kg(float(config["cube"]["edge_m"])),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    nominal_friction_matches = bool(
        campaign is None
        or math.isclose(
            float(config["cube"]["friction"]),
            campaign.discovery_friction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    nominal_passed = bool(
        nominal_hard_constraints_passed
        and nominal_is_constant_density
        and nominal_friction_matches
    )
    robust_passed = bool(
        nominal_passed
        and perturbation_passes >= parameters.required_pass_count
    )
    result = {
        "seed": effective_seed,
        "nominal_passed": nominal_passed,
        "nominal_hard_constraints_passed": nominal_hard_constraints_passed,
        "nominal_is_constant_density": nominal_is_constant_density,
        "nominal_friction_matches": nominal_friction_matches,
        "nominal_summary": nominal_summary,
        "grid_case_count": len(grid_summary),
        "grid_passes": sum(result["passed"] for result in grid_summary),
        "hardest_passing_grid_case": hardest_passing,
        "hardest_passing_constant_density_grid_case": hardest_passing,
        "hardest_passing_fixed_20g_control_grid_case": hardest_control,
        "grid": grid_summary,
        "perturbation_trial_count": len(perturbation_summary),
        "perturbation_passes": perturbation_passes,
        "required_perturbation_passes": parameters.required_pass_count,
        "robust_passed": robust_passed,
        "perturbations": perturbation_summary,
    }
    if parameters.case_families is not None:
        family_counts = {
            family: sum(
                record.get("case_family") == family for record in grid_summary
            )
            for family in ("constant_density", fixed_mass_label)
        }
        result["grid_case_family_counts"] = family_counts
        result["grid_case_family_passes"] = {
            "constant_density": len(density_passing_grid),
            fixed_mass_label: len(control_passing_grid),
        }
        result["robustness_case_families"] = parameters.case_families.as_config()
    return result
