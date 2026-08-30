"""Deterministic coarse-to-fine search for the large-cube campaign.

The module deliberately owns only the campaign orchestration.  Geometry,
simulation, ranking, and perturbation generation remain injectable package
services so the full 36--50 mm budget can be tested without running MuJoCo in
unit tests.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .config import ACTIVE_ACTUATORS, validate_config
from .experiment import ExperimentDefinition, SearchBounds, resolve_experiment
from .scene import build_model
from .v2_search import (
    KinematicScreenResult,
    _cube_position_in_root,
    _cube_world_position,
    _local_candidates,
    _rpy_matrix,
    kinematic_screen,
)


CandidateRunner = Callable[
    [list[tuple[int, dict[str, Any]]], int], list[dict[str, Any]]
]
CandidateRank = Callable[[dict[str, Any]], tuple[float, ...]]
PerturbationFactory = Callable[..., list[dict[str, Any]]]
KinematicScreener = Callable[..., KinematicScreenResult]
LocalCandidateFactory = Callable[..., list[dict[str, Any]]]


_STAGE_SEED_OFFSETS = {
    "coarse": 100_000_000,
    "odd": 200_000_000,
    "fine": 300_000_000,
}


def _edge_key(edge_m: float) -> int:
    """Return an exact, readable integer key for the declared millimetre grid."""

    return int(round(float(edge_m) * 1_000_000.0))


def _screen_seed(seed: int, stage: str, edge_m: float) -> int:
    return int(seed) + _STAGE_SEED_OFFSETS[stage] + _edge_key(edge_m) * 1009


def _face_key(config: Mapping[str, Any]) -> tuple[str, str, str]:
    target = config["contact_topology"]["target_faces"]
    return (str(target["thumb"]), str(target["index"]), str(target["mid"]))


def _fixed_material_config(
    config: dict[str, Any],
    *,
    edge_m: float,
    mass_kg: float,
    friction: float,
) -> dict[str, Any]:
    """Change cube material while preserving its pose relative to the hand.

    Increasing the edge raises the cube centre because its bottom must stay on
    the support.  Recomputing the hand translation from the preserved
    cube-in-root position prevents that support correction from silently
    changing the geometry being screened.
    """

    candidate = copy.deepcopy(config)
    cube_in_root = _cube_position_in_root(candidate)
    candidate["cube"]["edge_m"] = float(edge_m)
    candidate["cube"]["mass_kg"] = float(mass_kg)
    candidate["cube"]["friction"] = float(friction)
    candidate["cube"]["z_offset_m"] = 0.0
    rotation = _rpy_matrix(candidate["hand_pose"]["rpy_deg"])
    candidate["hand_pose"]["translation_m"] = (
        _cube_world_position(candidate) - rotation @ cube_in_root
    ).tolist()
    validate_config(candidate)
    return candidate


def static_candidate_advances(
    diagnostic: Mapping[str, Any], *, max_penetration_m: float
) -> bool:
    """Apply the declared clean-three / clean-two-near-three static gate."""

    if bool(diagnostic.get("forbidden_contact", False)):
        return False
    penetration = float(diagnostic.get("max_penetration_m", math.inf))
    if not math.isfinite(penetration) or penetration > max_penetration_m + 1e-12:
        return False
    clean = int(diagnostic.get("clean_target_contact_count", 0))
    if clean >= 3:
        return True
    if clean < 2 or int(diagnostic.get("near_target_face_count", 0)) < 3:
        return False
    distances = diagnostic.get("target_site_signed_distance_m")
    if distances is None:
        return False
    values = tuple(float(value) for value in distances)
    return len(values) == 3 and all(
        math.isfinite(value) and -0.002 - 1e-12 <= value <= 0.003 + 1e-12
        for value in values
    )


def _static_rank(diagnostic: Mapping[str, Any]) -> tuple[float, ...]:
    distances = diagnostic.get("target_site_signed_distance_m", ())
    finite_distance = sum(
        abs(float(value)) if math.isfinite(float(value)) else 1.0
        for value in distances
    )
    score = tuple(float(value) for value in diagnostic.get("score", ()))
    return (
        float(not bool(diagnostic.get("forbidden_contact", False))),
        float(diagnostic.get("clean_target_contact_count", 0)),
        float(diagnostic.get("near_target_face_count", 0)),
        -finite_distance,
        *score,
        -float(diagnostic.get("candidate_id", math.inf)),
    )


def _face_sample_counts(
    sample_count: int, definition: ExperimentDefinition
) -> dict[str, int]:
    quotient, remainder = divmod(sample_count, len(definition.candidate_faces))
    return {
        assignment.index: quotient + int(index < remainder)
        for index, assignment in enumerate(definition.candidate_faces)
    }


def _screen_record(
    *,
    stage: str,
    edge_m: float,
    samples_per_pitch: int,
    screen: KinematicScreenResult,
    definition: ExperimentDefinition,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    eligible: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    clean_histogram = {str(value): 0 for value in range(4)}
    near_histogram = {str(value): 0 for value in range(4)}
    retained_face_counts = {assignment.index: 0 for assignment in definition.candidate_faces}
    eligible_face_counts = dict(retained_face_counts)
    for candidate, diagnostic in pairs:
        clean = max(0, min(3, int(diagnostic.get("clean_target_contact_count", 0))))
        near = max(0, min(3, int(diagnostic.get("near_target_face_count", 0))))
        clean_histogram[str(clean)] += 1
        near_histogram[str(near)] += 1
        retained_face_counts[_face_key(candidate)[1]] += 1
    for candidate, _ in eligible:
        eligible_face_counts[_face_key(candidate)[1]] += 1
    return {
        "stage": stage,
        "edge_m": float(edge_m),
        "edge_mm": float(edge_m) * 1000.0,
        "seed": int(screen.seed),
        "pitch_values_deg": list(definition.search_bounds.palm_pitch_values_deg),
        "samples_per_pitch": int(samples_per_pitch),
        "sample_count": int(screen.sample_count),
        "face_sample_counts": _face_sample_counts(screen.sample_count, definition),
        "retained_count": len(pairs),
        "eligible_count": len(eligible),
        "retained_clean_count_histogram": clean_histogram,
        "retained_near_count_histogram": near_histogram,
        "retained_face_counts": retained_face_counts,
        "eligible_face_counts": eligible_face_counts,
        "top_diagnostics": [pair[1] for pair in pairs[:20]],
    }


def _run_size_screen(
    config: dict[str, Any],
    *,
    edge_m: float,
    stage: str,
    samples_per_pitch: int,
    retain: int,
    seed: int,
    definition: ExperimentDefinition,
    screen_candidates: KinematicScreener,
) -> dict[str, Any]:
    campaign = definition.size_campaign
    assert campaign is not None
    base = _fixed_material_config(
        config,
        edge_m=edge_m,
        mass_kg=campaign.discovery_mass_kg,
        friction=campaign.discovery_friction,
    )
    effective_seed = _screen_seed(seed, stage, edge_m)
    screen = screen_candidates(
        base,
        samples_per_pitch=samples_per_pitch,
        retain=retain,
        seed=effective_seed,
        definition=definition,
    )
    candidate_count = len(screen.candidates)
    diagnostic_count = len(screen.diagnostics)
    if (
        candidate_count == 0
        or candidate_count != diagnostic_count
        or int(screen.retained_count) != candidate_count
    ):
        raise RuntimeError(
            f"{stage} screen at edge={edge_m:.6f} m returned inconsistent "
            f"retention: retained_count={screen.retained_count}, "
            f"candidates={candidate_count}, diagnostics={diagnostic_count}"
        )
    pairs = [
        (copy.deepcopy(candidate), dict(diagnostic))
        for candidate, diagnostic in zip(screen.candidates, screen.diagnostics)
    ]
    pairs.sort(key=lambda item: _static_rank(item[1]), reverse=True)
    maximum_penetration = float(config["acceptance"]["max_penetration_m"])
    eligible = [
        pair
        for pair in pairs
        if static_candidate_advances(pair[1], max_penetration_m=maximum_penetration)
    ]
    return {
        "edge_m": float(edge_m),
        "pairs": pairs,
        "eligible": eligible,
        "record": _screen_record(
            stage=stage,
            edge_m=edge_m,
            samples_per_pitch=samples_per_pitch,
            screen=screen,
            definition=definition,
            pairs=pairs,
            eligible=eligible,
        ),
    }


def _size_rank(bundle: Mapping[str, Any]) -> tuple[Any, ...]:
    eligible = bundle["eligible"]
    pairs = eligible if eligible else bundle["pairs"]
    best_key = _static_rank(pairs[0][1]) if pairs else (-math.inf,)
    sample_count = max(1, int(bundle["record"]["sample_count"]))
    return (
        float(bool(eligible)),
        best_key,
        len(eligible) / sample_count,
        len(eligible),
        -float(bundle["edge_m"]),
    )


def adjacent_odd_edges_m(
    selected_edges_m: Iterable[float], *, lower_m: float, upper_m: float
) -> tuple[float, ...]:
    """Return sorted unique +/-1 mm neighbours inside the declared range."""

    result: set[float] = set()
    for edge in selected_edges_m:
        for delta in (-0.001, 0.001):
            candidate = round(float(edge) + delta, 12)
            if lower_m - 1e-12 <= candidate <= upper_m + 1e-12:
                result.add(candidate)
    return tuple(sorted(result))


def diagnose_boundary(
    config: Mapping[str, Any],
    bounds: SearchBounds,
    *,
    position_tolerance_m: float,
    actuator_tolerance_rad: float,
) -> dict[str, Any]:
    """Describe named lower/upper search-bound hits for one candidate."""

    cube_position = _cube_position_in_root(dict(config))
    position_hits: list[dict[str, Any]] = []
    for axis, value in zip(("x", "y", "z"), cube_position):
        lower, upper = bounds.cube_position_in_root_m[axis]
        if float(value) - lower <= position_tolerance_m + 1e-12:
            position_hits.append(
                {"axis": axis, "side": "lower", "value": float(value), "bound": lower}
            )
        if upper - float(value) <= position_tolerance_m + 1e-12:
            position_hits.append(
                {"axis": axis, "side": "upper", "value": float(value), "bound": upper}
            )

    actuator_hits: list[dict[str, Any]] = []
    for phase in ("pregrasp_targets_rad", "final_targets_rad"):
        targets = config["control"][phase]
        for name in ACTIVE_ACTUATORS:
            value = float(targets[name])
            lower, upper = bounds.actuator_targets_rad[name]
            if value - lower <= actuator_tolerance_rad + 1e-12:
                actuator_hits.append(
                    {
                        "actuator": name,
                        "phase": phase,
                        "side": "lower",
                        "value": value,
                        "bound": lower,
                    }
                )
            if upper - value <= actuator_tolerance_rad + 1e-12:
                actuator_hits.append(
                    {
                        "actuator": name,
                        "phase": phase,
                        "side": "upper",
                        "value": value,
                        "bound": upper,
                    }
                )
    return {
        "near_boundary": bool(position_hits or actuator_hits),
        "cube_position_in_root_m": cube_position.tolist(),
        "position_hits": position_hits,
        "actuator_hits": actuator_hits,
    }


def _model_actuator_hard_limits(config: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Intersect MuJoCo actuator and driven-joint limits by actuator name."""

    model, _ = build_model(config)
    result: dict[str, tuple[float, float]] = {}
    for name in ACTIVE_ACTUATORS:
        actuator_id = model.actuator(name).id
        lower, upper = -math.inf, math.inf
        if bool(model.actuator_ctrllimited[actuator_id]):
            lower = max(lower, float(model.actuator_ctrlrange[actuator_id, 0]))
            upper = min(upper, float(model.actuator_ctrlrange[actuator_id, 1]))
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id >= 0 and bool(model.jnt_limited[joint_id]):
            lower = max(lower, float(model.jnt_range[joint_id, 0]))
            upper = min(upper, float(model.jnt_range[joint_id, 1]))
        result[name] = (lower, upper)
    return result


def expand_definition_once(
    definition: ExperimentDefinition,
    diagnostic: Mapping[str, Any],
    *,
    actuator_hard_limits: Mapping[str, tuple[float, float]],
) -> tuple[ExperimentDefinition, dict[str, Any]]:
    """Expand only touched named sides, capped by actual model limits."""

    campaign = definition.size_campaign
    if campaign is None:
        raise ValueError("boundary expansion requires a size campaign")
    policy = campaign.boundary
    bounds = definition.search_bounds
    cube_ranges = {
        axis: list(value) for axis, value in bounds.cube_position_in_root_m.items()
    }
    actuator_ranges = {
        name: list(value) for name, value in bounds.actuator_targets_rad.items()
    }
    applied: list[dict[str, Any]] = []
    for hit in diagnostic.get("position_hits", ()):
        axis = str(hit["axis"])
        side = str(hit["side"])
        index = 0 if side == "lower" else 1
        before = cube_ranges[axis][index]
        cube_ranges[axis][index] += (
            -policy.position_expand_m if side == "lower" else policy.position_expand_m
        )
        applied.append(
            {
                "kind": "cube_position",
                "name": axis,
                "side": side,
                "before": before,
                "after": cube_ranges[axis][index],
                "capped": False,
            }
        )

    actuator_sides = {
        (str(hit["actuator"]), str(hit["side"]))
        for hit in diagnostic.get("actuator_hits", ())
    }
    for name, side in sorted(actuator_sides):
        if name not in actuator_hard_limits:
            raise ValueError(f"missing model hard limits for actuator {name!r}")
        index = 0 if side == "lower" else 1
        before = actuator_ranges[name][index]
        proposed = before + (
            -policy.actuator_expand_rad if side == "lower" else policy.actuator_expand_rad
        )
        hard_lower, hard_upper = actuator_hard_limits[name]
        after = max(hard_lower, proposed) if side == "lower" else min(hard_upper, proposed)
        actuator_ranges[name][index] = after
        applied.append(
            {
                "kind": "actuator",
                "name": name,
                "side": side,
                "before": before,
                "after": after,
                "model_limit": hard_lower if side == "lower" else hard_upper,
                "capped": not math.isclose(after, proposed, rel_tol=0.0, abs_tol=1e-12),
            }
        )

    expanded_bounds = replace(
        bounds,
        cube_position_in_root_m=cube_ranges,
        actuator_targets_rad=actuator_ranges,
    )
    return replace(definition, search_bounds=expanded_bounds), {
        "applied": bool(applied),
        "expansion_count": int(bool(applied)),
        "changes": applied,
    }


def _top_results(
    results: Iterable[dict[str, Any]], count: int, rank: CandidateRank
) -> list[dict[str, Any]]:
    """Return the global deterministic top-N without imposing face quotas."""

    return sorted(results, key=rank, reverse=True)[:count]


def _annotate_results(
    results: Iterable[dict[str, Any]], *, stage: str, material_policy: str
) -> list[dict[str, Any]]:
    ordered = sorted(results, key=lambda item: int(item["candidate_id"]))
    for result in ordered:
        result["search_stage"] = stage
        result["material_policy"] = material_policy
    return ordered


def _has_local_refinement_signal(
    result: Mapping[str, Any], *, minimum_target_fingers: int
) -> bool:
    """Return whether an initial run has enough target-face refinement signal.

    Production v2 summaries always expose ``target_face_simultaneous_duty``.
    Missing or non-finite metrics are not evidence of contact and simulation
    errors must never unlock the expensive local-refinement stage.
    """

    summary = result.get("summary", {})
    if "simulation_error" in summary.get("failed_checks", ()):
        return False
    metrics = summary.get("metrics", {})
    if "target_face_simultaneous_duty" not in metrics:
        return False
    try:
        duty = float(metrics["target_face_simultaneous_duty"])
    except (TypeError, ValueError):
        return False
    if math.isfinite(duty) and duty > 0.0:
        return True
    per_finger = metrics.get("target_face_contact_duty")
    if not isinstance(per_finger, Mapping):
        return False
    try:
        duties = tuple(
            float(per_finger.get(finger, 0.0))
            for finger in ("thumb", "index", "mid")
        )
    except (TypeError, ValueError):
        return False
    return sum(math.isfinite(value) and value > 0.0 for value in duties) >= int(
        minimum_target_fingers
    )


def _run_stage(
    candidates: Iterable[dict[str, Any]],
    *,
    next_id: int,
    workers: int,
    run_candidates: CandidateRunner,
    stage: str,
    material_policy: str,
) -> tuple[list[dict[str, Any]], int]:
    payload: list[tuple[int, dict[str, Any]]] = []
    for candidate in candidates:
        payload.append((next_id, candidate))
        next_id += 1
    if not payload:
        return [], next_id
    results = run_candidates(payload, workers)
    expected_ids = tuple(candidate_id for candidate_id, _ in payload)
    try:
        received_ids = tuple(int(result["candidate_id"]) for result in results)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{stage} runner returned an invalid candidate_id") from error
    if (
        len(results) != len(payload)
        or len(set(received_ids)) != len(received_ids)
        or set(received_ids) != set(expected_ids)
    ):
        raise RuntimeError(
            f"{stage} runner result IDs must exactly match submitted IDs; "
            f"expected={expected_ids}, received={received_ids}"
        )
    submitted = {candidate_id: candidate for candidate_id, candidate in payload}
    for result in results:
        candidate_id = int(result["candidate_id"])
        if result.get("config") != submitted[candidate_id]:
            raise RuntimeError(
                f"{stage} runner rebound candidate_id={candidate_id} to a "
                "different configuration"
            )
    return _annotate_results(
        results, stage=stage, material_policy=material_policy
    ), next_id


def _probe_size_boundaries(
    parent: dict[str, Any],
    *,
    definition: ExperimentDefinition,
    next_id: int,
    workers: int,
    run_candidates: CandidateRunner,
) -> tuple[list[dict[str, Any]], int]:
    """Walk by 0.5 mm from a density pass until failure or range limit."""

    campaign = definition.size_campaign
    families = definition.robustness.case_families
    if campaign is None or families is None:
        return [], next_id
    nominal_edge = float(parent["config"]["cube"]["edge_m"])
    lower, upper = families.edge_limits_m
    step = families.boundary_probe_step_m
    records: list[dict[str, Any]] = []
    for direction, sign in (("smaller", -1.0), ("larger", 1.0)):
        trials: list[dict[str, Any]] = []
        step_index = 1
        while True:
            edge_m = round(nominal_edge + sign * step_index * step, 12)
            if edge_m < lower - 1e-12 or edge_m > upper + 1e-12:
                stop = "declared_range_limit"
                break
            candidate = _fixed_material_config(
                parent["config"],
                edge_m=edge_m,
                mass_kg=campaign.constant_density_mass_kg(edge_m),
                friction=float(parent["config"]["cube"]["friction"]),
            )
            results, next_id = _run_stage(
                [candidate],
                next_id=next_id,
                workers=workers,
                run_candidates=run_candidates,
                stage="constant_density_size_boundary_probe",
                material_policy="constant_density",
            )
            result = results[0]
            trials.append(
                {
                    "step_index": step_index,
                    "edge_m": edge_m,
                    "edge_mm": edge_m * 1000.0,
                    "mass_kg": float(candidate["cube"]["mass_kg"]),
                    "passed": bool(result["summary"]["passed"]),
                    "failed_checks": list(result["summary"].get("failed_checks", ())),
                    "metrics": result["summary"].get("metrics", {}),
                    "candidate_id": int(result["candidate_id"]),
                }
            )
            if not result["summary"]["passed"]:
                stop = "first_hard_failure"
                break
            step_index += 1
        passing = [trial for trial in trials if trial["passed"]]
        failure = next((trial for trial in trials if not trial["passed"]), None)
        records.append(
            {
                "direction": direction,
                "step_m": step,
                "stop_reason": stop,
                "trial_count": len(trials),
                "last_passing_edge_m": (
                    passing[-1]["edge_m"] if passing else nominal_edge
                ),
                "first_failing_edge_m": (
                    failure["edge_m"] if failure is not None else None
                ),
                "trials": trials,
            }
        )
    return records, next_id


def _candidate_counts_by_edge(
    results: Iterable[dict[str, Any]], rank: CandidateRank
) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        buckets.setdefault(_edge_key(result["config"]["cube"]["edge_m"]), []).append(result)
    records: list[dict[str, Any]] = []
    for key in sorted(buckets):
        items = buckets[key]
        ranked = sorted(items, key=rank, reverse=True)
        records.append(
            {
                "edge_m": float(ranked[0]["config"]["cube"]["edge_m"]),
                "edge_mm": float(ranked[0]["config"]["cube"]["edge_m"]) * 1000.0,
                "candidate_count": len(items),
                "passing_candidates": sum(bool(item["summary"]["passed"]) for item in items),
                "best_candidate_id": int(ranked[0]["candidate_id"]),
                "best_failed_checks": list(ranked[0]["summary"].get("failed_checks", ())),
            }
        )
    return records


def _effective_count(override: int | None, default: int, label: str, *, zero_ok: bool = False) -> int:
    value = default if override is None else int(override)
    if value < 0 or (value == 0 and not zero_ok):
        qualifier = "non-negative" if zero_ok else "positive"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def tune_large_cube(
    config: dict[str, Any],
    *,
    workers: int,
    seed: int,
    run_candidates: CandidateRunner,
    rank_candidate: CandidateRank,
    perturb_cases: PerturbationFactory | None,
    kinematic_samples_per_pitch: int | None = None,
    dynamic_candidate_count: int | None = None,
    local_refine_seed_count: int | None = None,
    local_refine_per_seed: int | None = None,
    final_candidate_count: int | None = None,
    perturbations_per_final: int | None = None,
    fallback_physics_count: int | None = None,
    fallback_kinematic_samples_per_pitch: int | None = None,
    screen_candidates: KinematicScreener | None = None,
    local_candidate_factory: LocalCandidateFactory | None = None,
    actuator_hard_limits: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Execute the versioned 36--50 mm fixed-mass/density campaign."""

    validate_config(config)
    definition = resolve_experiment(config)
    campaign = definition.size_campaign
    if campaign is None:
        raise ValueError("the selected experiment does not define a size campaign")
    screen_candidates = kinematic_screen if screen_candidates is None else screen_candidates
    local_candidate_factory = (
        _local_candidates if local_candidate_factory is None else local_candidate_factory
    )

    common_screen_override = kinematic_samples_per_pitch
    coarse_samples = _effective_count(
        common_screen_override, campaign.coarse_samples_per_pitch, "coarse samples"
    )
    secondary_override = (
        fallback_kinematic_samples_per_pitch
        if fallback_kinematic_samples_per_pitch is not None
        else common_screen_override
    )
    odd_samples = _effective_count(
        secondary_override, campaign.odd_samples_per_pitch, "odd samples"
    )
    fine_samples = _effective_count(
        secondary_override, campaign.fine_samples_per_pitch, "fine samples"
    )
    dynamic_total = _effective_count(
        dynamic_candidate_count,
        campaign.dynamic_candidate_count,
        "dynamic candidate count",
    )
    local_seed_total = _effective_count(
        local_refine_seed_count,
        campaign.exact_size_count * campaign.local_seed_count_per_size,
        "local refine seed count",
    )
    local_per_seed = _effective_count(
        local_refine_per_seed,
        campaign.local_refine_per_seed,
        "local refine per seed",
    )
    finalist_count = _effective_count(
        final_candidate_count, campaign.finalist_count, "final candidate count"
    )
    perturbation_count = _effective_count(
        perturbations_per_final,
        campaign.perturbations_per_final,
        "perturbations per final",
    )
    density_max = _effective_count(
        fallback_physics_count,
        campaign.constant_density_max_candidates,
        "constant-density candidate count",
        zero_ok=True,
    )
    static_retain = max(campaign.fine_dynamic_per_size, math.ceil(dynamic_total / 2))
    # The generic screener divides ``retain`` evenly among face assignments.
    # Fine screening therefore retains one full per-size dynamic capacity for
    # *each* face before applying the gate and a global rank.  Otherwise a
    # single viable opposed-face orientation could be capped at one quarter of
    # the declared 256 candidates per exact size.
    fine_retain = static_retain * len(definition.candidate_faces)

    coarse_bundles = [
        _run_size_screen(
            config,
            edge_m=edge,
            stage="coarse",
            samples_per_pitch=coarse_samples,
            retain=static_retain,
            seed=seed,
            definition=definition,
            screen_candidates=screen_candidates,
        )
        for edge in campaign.coarse_edges_m
    ]
    selected_coarse = sorted(coarse_bundles, key=_size_rank, reverse=True)[
        : campaign.coarse_size_count
    ]
    odd_edges = adjacent_odd_edges_m(
        (bundle["edge_m"] for bundle in selected_coarse),
        lower_m=campaign.coarse_edges_m[0],
        upper_m=campaign.coarse_edges_m[-1],
    )
    odd_bundles = [
        _run_size_screen(
            config,
            edge_m=edge,
            stage="odd",
            samples_per_pitch=odd_samples,
            retain=static_retain,
            seed=seed,
            definition=definition,
            screen_candidates=screen_candidates,
        )
        for edge in odd_edges
    ]
    exact_pool = selected_coarse + odd_bundles
    selected_exact = sorted(exact_pool, key=_size_rank, reverse=True)[
        : campaign.exact_size_count
    ]
    if len(selected_exact) != campaign.exact_size_count:
        raise RuntimeError("large-cube search could not select the declared exact sizes")

    static_best_pairs = [
        (bundle["eligible"] or bundle["pairs"])[0]
        for bundle in selected_exact
        if bundle["eligible"] or bundle["pairs"]
    ]
    static_best = max(static_best_pairs, key=lambda pair: _static_rank(pair[1]))
    boundary_policy = campaign.boundary
    pre_fine_boundary = diagnose_boundary(
        static_best[0],
        definition.search_bounds,
        position_tolerance_m=boundary_policy.position_tolerance_m,
        actuator_tolerance_rad=boundary_policy.actuator_tolerance_rad,
    )
    effective_definition = definition
    expansion = {
        "applied": False,
        "expansion_count": 0,
        "changes": [],
        "trigger_stage": None,
    }

    initial_fine_bundles = [
        _run_size_screen(
            config,
            edge_m=float(bundle["edge_m"]),
            stage="fine",
            samples_per_pitch=fine_samples,
            retain=fine_retain,
            seed=seed,
            definition=effective_definition,
            screen_candidates=screen_candidates,
        )
        for bundle in selected_exact
    ]
    fine_static_pairs = [
        pair
        for bundle in initial_fine_bundles
        for pair in (bundle["eligible"] or bundle["pairs"][:1])
    ]
    fine_static_best = max(
        fine_static_pairs, key=lambda pair: _static_rank(pair[1])
    )
    fine_static_boundary = diagnose_boundary(
        fine_static_best[0],
        effective_definition.search_bounds,
        position_tolerance_m=boundary_policy.position_tolerance_m,
        actuator_tolerance_rad=boundary_policy.actuator_tolerance_rad,
    )
    # Static scoring evaluates the pregrasp only.  Do not spend the one-shot
    # expansion on an unscored terminal target; the first full dynamic best
    # below is the authoritative expansion trigger.
    fine_bundles = initial_fine_bundles

    quotient, remainder = divmod(dynamic_total, len(fine_bundles))
    dynamic_candidates: list[dict[str, Any]] = []
    no_gate_candidates = True
    for index, bundle in enumerate(fine_bundles):
        capacity = quotient + int(index < remainder)
        eligible = bundle["eligible"]
        if eligible:
            no_gate_candidates = False
        selected = sorted(
            eligible, key=lambda pair: _static_rank(pair[1]), reverse=True
        )[:capacity]
        dynamic_candidates.extend(pair[0] for pair in selected)
        selected_count = len(selected)
        bundle["record"]["dynamic_selection"] = {
            "requested_count": capacity,
            "eligible_available_count": len(eligible),
            "selected_count": selected_count,
            "shortfall_count": capacity - selected_count,
            "underfilled_reason": (
                None
                if selected_count == capacity
                else "insufficient_candidates_meeting_static_gate"
            ),
        }
    # Preserve one auditable dynamic near miss if every fine-screen candidate
    # missed the declared static gate.  It is labelled diagnostic and cannot
    # be mistaken for an advanced campaign candidate.
    diagnostic_only = False
    if not dynamic_candidates:
        available = [pair for bundle in fine_bundles for pair in bundle["pairs"]]
        if not available:
            raise RuntimeError("large-cube fine screens retained no candidates")
        dynamic_candidates = [max(available, key=lambda pair: _static_rank(pair[1]))[0]]
        diagnostic_only = True

    next_id = 0
    fixed_dynamic, next_id = _run_stage(
        dynamic_candidates,
        next_id=next_id,
        workers=workers,
        run_candidates=run_candidates,
        stage=("fixed_mass_gate_near_miss" if diagnostic_only else "fixed_mass_dynamic"),
        material_policy="fixed_20g_control",
    )

    fixed_dynamic_error_count = sum(
        "simulation_error" in result["summary"].get("failed_checks", ())
        for result in fixed_dynamic
    )
    all_fixed_dynamic_errors = bool(
        fixed_dynamic
        and fixed_dynamic_error_count == len(fixed_dynamic)
    )
    initial_topology_signal_count = sum(
        _has_local_refinement_signal(
            result,
            minimum_target_fingers=(
                campaign.local_refinement_min_target_fingers
            ),
        )
        for result in fixed_dynamic
    )
    dynamic_boundary = diagnose_boundary(
        max(fixed_dynamic, key=rank_candidate)["config"],
        effective_definition.search_bounds,
        position_tolerance_m=boundary_policy.position_tolerance_m,
        actuator_tolerance_rad=boundary_policy.actuator_tolerance_rad,
    )
    if (
        not expansion["applied"]
        and not all_fixed_dynamic_errors
        and dynamic_boundary["near_boundary"]
    ):
        limits = (
            _model_actuator_hard_limits(config)
            if actuator_hard_limits is None
            else dict(actuator_hard_limits)
        )
        effective_definition, expansion = expand_definition_once(
            effective_definition, dynamic_boundary, actuator_hard_limits=limits
        )
        expansion["trigger_stage"] = "fixed_mass_dynamic_best"

    fixed_local_candidates: list[dict[str, Any]] = []
    continue_fixed_refinement = bool(
        not all_fixed_dynamic_errors
        and (not diagnostic_only or expansion["applied"])
        and (initial_topology_signal_count or expansion["applied"])
    )
    if continue_fixed_refinement:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for result in fixed_dynamic:
            grouped.setdefault(
                _edge_key(result["config"]["cube"]["edge_m"]), []
            ).append(result)
        seed_quotient, seed_remainder = divmod(local_seed_total, len(fine_bundles))
        for edge_index, bundle in enumerate(fine_bundles):
            edge_results = grouped.get(_edge_key(bundle["edge_m"]), [])
            seed_capacity = seed_quotient + int(edge_index < seed_remainder)
            parents = _top_results(edge_results, seed_capacity, rank_candidate)
            for parent_index, parent in enumerate(parents):
                fixed_local_candidates.extend(
                    local_candidate_factory(
                        parent["config"],
                        count=local_per_seed,
                        seed=(
                            seed
                            + 400_000_000
                            + edge_index * 1_000_003
                            + parent_index
                        ),
                        definition=effective_definition,
                    )
                )
    fixed_local, next_id = _run_stage(
        fixed_local_candidates,
        next_id=next_id,
        workers=workers,
        run_candidates=run_candidates,
        stage="fixed_mass_local_refinement",
        material_policy="fixed_20g_control",
    )
    fixed_results = fixed_dynamic + fixed_local
    ranked_fixed = sorted(fixed_results, key=rank_candidate, reverse=True)
    post_local_boundary = diagnose_boundary(
        ranked_fixed[0]["config"],
        effective_definition.search_bounds,
        position_tolerance_m=boundary_policy.position_tolerance_m,
        actuator_tolerance_rad=boundary_policy.actuator_tolerance_rad,
    )
    fixed_boundary_local: list[dict[str, Any]] = []
    if (
        not expansion["applied"]
        and not all_fixed_dynamic_errors
        and post_local_boundary["near_boundary"]
    ):
        limits = (
            _model_actuator_hard_limits(config)
            if actuator_hard_limits is None
            else dict(actuator_hard_limits)
        )
        effective_definition, expansion = expand_definition_once(
            effective_definition,
            post_local_boundary,
            actuator_hard_limits=limits,
        )
        expansion["trigger_stage"] = "fixed_mass_local_best"
        boundary_candidates = local_candidate_factory(
            ranked_fixed[0]["config"],
            count=local_per_seed,
            seed=seed + 450_000_000,
            definition=effective_definition,
        )
        fixed_boundary_local, next_id = _run_stage(
            boundary_candidates,
            next_id=next_id,
            workers=workers,
            run_candidates=run_candidates,
            stage="fixed_mass_boundary_refinement",
            material_policy="fixed_20g_control",
        )
        fixed_local.extend(fixed_boundary_local)
        fixed_results.extend(fixed_boundary_local)
        ranked_fixed = sorted(fixed_results, key=rank_candidate, reverse=True)
    fixed_passes = [item for item in ranked_fixed if item["summary"]["passed"]]
    fixed_mass_success = bool(fixed_passes)
    best_fixed_mass = copy.deepcopy(fixed_passes[0]) if fixed_passes else None

    density_candidates: list[dict[str, Any]] = []
    if fixed_mass_success and density_max:
        for parent in _top_results(
            fixed_passes, min(density_max, len(fixed_passes)), rank_candidate
        ):
            candidate = copy.deepcopy(parent["config"])
            edge_m = float(candidate["cube"]["edge_m"])
            candidate["cube"]["mass_kg"] = campaign.constant_density_mass_kg(edge_m)
            validate_config(candidate)
            density_candidates.append(candidate)
    density_reevaluation, next_id = _run_stage(
        density_candidates,
        next_id=next_id,
        workers=workers,
        run_candidates=run_candidates,
        stage="constant_density_reevaluation",
        material_policy="constant_density",
    )

    density_local_candidates: list[dict[str, Any]] = []
    density_reevaluation_error_count = sum(
        "simulation_error" in item["summary"].get("failed_checks", ())
        for item in density_reevaluation
    )
    evaluable_density_reevaluation = [
        item
        for item in density_reevaluation
        if "simulation_error" not in item["summary"].get("failed_checks", ())
    ]
    all_density_reevaluation_errors = bool(
        density_reevaluation
        and density_reevaluation_error_count == len(density_reevaluation)
    )
    if evaluable_density_reevaluation and not any(
        item["summary"]["passed"] for item in density_reevaluation
    ):
        parents = _top_results(
            evaluable_density_reevaluation,
            min(
                campaign.density_refine_seed_count,
                len(evaluable_density_reevaluation),
            ),
            rank_candidate,
        )
        for parent_index, parent in enumerate(parents):
            density_local_candidates.extend(
                local_candidate_factory(
                    parent["config"],
                    count=campaign.density_refine_per_seed,
                    seed=seed + 500_000_000 + parent_index,
                    definition=effective_definition,
                )
            )
    density_local, next_id = _run_stage(
        density_local_candidates,
        next_id=next_id,
        workers=workers,
        run_candidates=run_candidates,
        stage="constant_density_local_refinement",
        material_policy="constant_density",
    )
    density_results = density_reevaluation + density_local
    ranked_density = sorted(density_results, key=rank_candidate, reverse=True)
    density_boundary_local: list[dict[str, Any]] = []
    if (
        ranked_density
        and not all_density_reevaluation_errors
        and not expansion["applied"]
    ):
        density_boundary = diagnose_boundary(
            ranked_density[0]["config"],
            effective_definition.search_bounds,
            position_tolerance_m=boundary_policy.position_tolerance_m,
            actuator_tolerance_rad=boundary_policy.actuator_tolerance_rad,
        )
        if density_boundary["near_boundary"]:
            limits = (
                _model_actuator_hard_limits(config)
                if actuator_hard_limits is None
                else dict(actuator_hard_limits)
            )
            effective_definition, expansion = expand_definition_once(
                effective_definition,
                density_boundary,
                actuator_hard_limits=limits,
            )
            expansion["trigger_stage"] = "constant_density_best"
            boundary_candidates = local_candidate_factory(
                ranked_density[0]["config"],
                count=campaign.density_refine_per_seed,
                seed=seed + 550_000_000,
                definition=effective_definition,
            )
            density_boundary_local, next_id = _run_stage(
                boundary_candidates,
                next_id=next_id,
                workers=workers,
                run_candidates=run_candidates,
                stage="constant_density_boundary_refinement",
                material_policy="constant_density",
            )
            density_local.extend(density_boundary_local)
            density_results.extend(density_boundary_local)
            ranked_density = sorted(
                density_results, key=rank_candidate, reverse=True
            )
    density_passes = [item for item in ranked_density if item["summary"]["passed"]]
    constant_density_success = bool(density_passes)

    final_pool = ranked_density if ranked_density else ranked_fixed
    hard_finalists = [item for item in final_pool if item["summary"]["passed"]]
    finalist_parents = _top_results(
        hard_finalists if hard_finalists else final_pool,
        min(finalist_count, len(hard_finalists if hard_finalists else final_pool)),
        rank_candidate,
    )
    probe_results: list[dict[str, Any]] = []
    if perturb_cases is not None:
        for parent_index, parent in enumerate(finalist_parents):
            cases = perturb_cases(
                parent["config"],
                count=perturbation_count,
                seed=seed + 600_000_000 + parent_index,
            )
            if len(cases) != perturbation_count:
                raise RuntimeError(
                    "finalist perturbation factory must return exactly "
                    f"{perturbation_count} cases; received {len(cases)} for "
                    f"candidate_id={parent['candidate_id']}"
                )
            trials, next_id = _run_stage(
                cases,
                next_id=next_id,
                workers=workers,
                run_candidates=run_candidates,
                stage="finalist_local_perturbation",
                material_policy="local_material_and_pose_perturbation",
            )
            for trial in trials:
                trial["parent_material_policy"] = str(parent["material_policy"])
            probe_results.append(
                {
                    "candidate_id": int(parent["candidate_id"]),
                    "parent_material_policy": str(parent["material_policy"]),
                    "passes": sum(bool(trial["summary"]["passed"]) for trial in trials),
                    "trial_count": len(trials),
                    "trials": trials,
                }
            )
    probe_lookup = {record["candidate_id"]: record for record in probe_results}

    def selection_key(item: dict[str, Any]) -> tuple[Any, ...]:
        probe = probe_lookup.get(int(item["candidate_id"]), {"passes": -1})
        return (
            float(item["summary"]["passed"]),
            float(probe["passes"]),
            *rank_candidate(item),
            -int(item["candidate_id"]),
        )

    best = copy.deepcopy(max(final_pool, key=selection_key))
    best["local_perturbation_probe"] = copy.deepcopy(
        probe_lookup.get(
            int(best["candidate_id"]),
            {
                "candidate_id": int(best["candidate_id"]),
                "passes": 0,
                "trial_count": 0,
                "trials": [],
            },
        )
    )
    if constant_density_success:
        campaign_classification = "validated_constant_density"
        config_classification = "validated_constant_density"
        stop_reason = "constant_density_hard_pass_found"
    elif fixed_mass_success and ranked_density:
        campaign_classification = "validated_fixed_mass_ablation_only"
        config_classification = (
            "constant_density_simulation_error"
            if all_density_reevaluation_errors
            else "best_constant_density_near_miss"
        )
        stop_reason = (
            "all_constant_density_reevaluations_failed_to_simulate"
            if all_density_reevaluation_errors
            else "no_constant_density_hard_pass"
        )
    elif fixed_mass_success:
        campaign_classification = "validated_fixed_mass_ablation_only"
        config_classification = "validated_fixed_mass_ablation"
        stop_reason = "constant_density_stage_disabled"
    else:
        campaign_classification = "not_validated"
        config_classification = (
            "fixed_mass_simulation_error"
            if all_fixed_dynamic_errors
            else "best_near_miss"
        )
        stop_reason = (
            "all_fixed_mass_dynamics_failed_to_simulate"
            if all_fixed_dynamic_errors
            else "no_kinematic_candidate_met_gate"
            if no_gate_candidates
            else "no_initial_local_refinement_signal"
            if not continue_fixed_refinement
            else "no_fixed_mass_hard_pass"
        )
    campaign_status = {
        "classification": campaign_classification,
        "passed": constant_density_success,
        "fixed_mass_discovery_passed": fixed_mass_success,
        "constant_density_passed": constant_density_success,
        "stop_reason": stop_reason,
        "note": (
            "A constant-density candidate passed all declared hard constraints."
            if constant_density_success
            else "The fixed-mass geometry passed, but every constant-density reevaluation failed to simulate."
            if fixed_mass_success and all_density_reevaluation_errors
            else "Only the fixed-20 g geometry ablation passed; constant-density validation failed."
            if fixed_mass_success and ranked_density
            else "The fixed-20 g geometry ablation passed, but the constant-density stage was disabled."
            if fixed_mass_success
            else "Every fixed-mass dynamic candidate ended in a simulation error."
            if all_fixed_dynamic_errors
            else "No fixed-20 g candidate passed all declared hard constraints within the declared budget."
        ),
    }
    best["config"]["experiment_status"] = {
        "classification": config_classification,
        "passed": constant_density_success,
        "fixed_mass_ablation_passed": fixed_mass_success,
        "constant_density_passed": constant_density_success,
        "note": campaign_status["note"],
    }
    if best_fixed_mass is not None:
        best_fixed_mass["config"]["experiment_status"] = {
            "classification": "validated_fixed_mass_ablation",
            "passed": False,
            "hard_constraints_passed_at_fixed_mass": True,
            "constant_density_passed": False,
            "campaign_constant_density_passed": constant_density_success,
            "note": "This is a 20 g geometry-control pass, not constant-density validation.",
        }

    size_boundary_probes: list[dict[str, Any]] = []
    if constant_density_success:
        size_boundary_probes, next_id = _probe_size_boundaries(
            best,
            definition=definition,
            next_id=next_id,
            workers=workers,
            run_candidates=run_candidates,
        )

    post_search_boundary = diagnose_boundary(
        best["config"],
        effective_definition.search_bounds,
        position_tolerance_m=boundary_policy.position_tolerance_m,
        actuator_tolerance_rad=boundary_policy.actuator_tolerance_rad,
    )
    boundary_limited = bool(
        expansion["expansion_count"] >= boundary_policy.max_expansions
        and post_search_boundary["near_boundary"]
    )
    expansion["initial_diagnostic"] = dynamic_boundary
    expansion["pre_fine_static_diagnostic"] = pre_fine_boundary
    expansion["fine_static_diagnostic"] = fine_static_boundary
    expansion["post_local_diagnostic"] = post_local_boundary
    expansion["final_diagnostic"] = post_search_boundary
    expansion["boundary_limited"] = boundary_limited

    all_main_results = fixed_results + density_results
    ranked_all = sorted(all_main_results, key=rank_candidate, reverse=True)
    screen_records = {
        "coarse": [bundle["record"] for bundle in coarse_bundles],
        "odd": [bundle["record"] for bundle in odd_bundles],
        "fine": [bundle["record"] for bundle in fine_bundles],
    }
    size_campaign_diagnostics = {
        "schema_version": campaign.schema_version,
        "declared": campaign.as_config(),
        "effective_budget": {
            "coarse_samples_per_pitch": coarse_samples,
            "odd_samples_per_pitch": odd_samples,
            "fine_samples_per_pitch": fine_samples,
            "fine_retain_total_per_size": fine_retain,
            "fine_retain_capacity_per_face": static_retain,
            "dynamic_candidate_count": dynamic_total,
            "local_refine_seed_count": local_seed_total,
            "local_refine_per_seed": local_per_seed,
            "constant_density_max_candidates": density_max,
            "finalist_count": finalist_count,
            "perturbations_per_final": perturbation_count,
        },
        "kinematic_gate": {
            "clean_three": True,
            "alternative_min_clean": 2,
            "alternative_near_count": 3,
            "target_site_signed_distance_m": [-0.002, 0.003],
            "forbidden_contact": False,
            "max_penetration_m": float(config["acceptance"]["max_penetration_m"]),
        },
        "continuation_policy": {
            "local_refinement_min_target_fingers": (
                campaign.local_refinement_min_target_fingers
            ),
            "boundary_hit_also_continues": True,
        },
        "selected_coarse_edges_m": [float(bundle["edge_m"]) for bundle in selected_coarse],
        "odd_edges_m": list(odd_edges),
        "selected_exact_edges_m": [float(bundle["edge_m"]) for bundle in selected_exact],
        "stages": screen_records,
        "boundary_expansion": expansion,
    }
    perturbation_probe_count = sum(record["trial_count"] for record in probe_results)
    size_boundary_probe_count = sum(
        record["trial_count"] for record in size_boundary_probes
    )
    return {
        "experiment_id": definition.experiment_id,
        "campaign_kind": "large_cube_size",
        "campaign_classification": campaign_classification,
        "campaign_status": campaign_status,
        "seed": int(seed),
        "workers": int(workers),
        "size_campaign_diagnostics": size_campaign_diagnostics,
        "size_stages": screen_records,
        "selected_exact_edges_m": size_campaign_diagnostics["selected_exact_edges_m"],
        "kinematic_sample_count": sum(
            record["sample_count"] for records in screen_records.values() for record in records
        ),
        "kinematic_retained_count": sum(
            record["retained_count"] for records in screen_records.values() for record in records
        ),
        "kinematic_top_diagnostics": [
            pair[1]
            for bundle in fine_bundles
            for pair in bundle["pairs"][:10]
        ][:20],
        "initial_dynamic_count": len(fixed_dynamic),
        "local_refinement_count": len(fixed_local) + len(density_local),
        "fallback_physics_count": len(density_reevaluation),
        "fallback_kinematic_samples_per_pitch": fine_samples,
        "fixed_mass_dynamic_count": len(fixed_dynamic),
        "fixed_mass_dynamic_simulation_error_count": fixed_dynamic_error_count,
        "initial_three_finger_topology_signal_count": (
            initial_topology_signal_count
        ),
        "initial_local_refinement_signal_count": initial_topology_signal_count,
        "fixed_mass_refinement_continued": continue_fixed_refinement,
        "fixed_mass_local_refinement_count": len(fixed_local),
        "fixed_mass_boundary_refinement_count": len(fixed_boundary_local),
        "constant_density_reevaluation_count": len(density_reevaluation),
        "constant_density_reevaluation_simulation_error_count": (
            density_reevaluation_error_count
        ),
        "constant_density_local_refinement_count": len(density_local),
        "constant_density_boundary_refinement_count": len(
            density_boundary_local
        ),
        "candidate_count": len(all_main_results),
        "perturbation_probe_count": perturbation_probe_count,
        "size_boundary_probe_count": size_boundary_probe_count,
        "simulation_count": (
            len(all_main_results)
            + perturbation_probe_count
            + size_boundary_probe_count
        ),
        "passing_candidates": len(density_passes),
        "fixed_mass_passing_candidates": len(fixed_passes),
        "constant_density_passing_candidates": len(density_passes),
        "fixed_mass_success": fixed_mass_success,
        "constant_density_success": constant_density_success,
        "nominal_success": constant_density_success,
        "alternative_physics_success": False,
        "stop_reason": stop_reason,
        "boundary_diagnostics": expansion,
        "boundary_limited": boundary_limited,
        "size_summaries": _candidate_counts_by_edge(all_main_results, rank_candidate),
        "size_boundary_probes": size_boundary_probes,
        "best_fixed_mass": best_fixed_mass,
        "best": best,
        "top_candidates": ranked_all[:20],
        "local_perturbation_probes": probe_results,
    }


__all__ = [
    "adjacent_odd_edges_m",
    "diagnose_boundary",
    "expand_definition_once",
    "static_candidate_advances",
    "tune_large_cube",
]
