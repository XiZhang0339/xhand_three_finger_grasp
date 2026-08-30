"""Pure, deterministic planning helpers for the schema-v4 campaign.

This module intentionally has no MuJoCo dependency.  Search runners can use the
jobs, ranking and per-band selection here without coupling orchestration to the
experiment registry or to worker completion order.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from .config import validate_config
from .experiment import AlignedContactCampaignParameters
from .experiment import resolve_experiment
from .experiments.opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift import (
    ALIGNED_CONTACT_CAMPAIGN,
    EXPERIMENT_ID,
)


DEFAULT_CAMPAIGN = ALIGNED_CONTACT_CAMPAIGN
DEFAULT_PLAN = DEFAULT_CAMPAIGN


def _finite_metric(
    containers: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    *,
    default: float,
) -> float:
    for container in containers:
        for name in names:
            if name not in container:
                continue
            try:
                number = float(container[name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return default


def _summary(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = result.get("summary", {})
    return value if isinstance(value, Mapping) else {}


def _metrics(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _summary(result).get("metrics", {})
    return value if isinstance(value, Mapping) else {}


def _candidate_id(result: Mapping[str, Any]) -> int:
    value = result.get("candidate_id")
    if isinstance(value, bool):
        raise ValueError("candidate result requires an integer candidate_id")
    try:
        candidate_id = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "candidate result requires an integer candidate_id"
        ) from error
    if isinstance(value, float) and value != candidate_id:
        raise ValueError("candidate result requires an integer candidate_id")
    return candidate_id


def _hard_pass(result: Mapping[str, Any]) -> bool:
    summary = _summary(result)
    stage_status = summary.get("stage_status")
    if isinstance(stage_status, Mapping) and "full_success" in stage_status:
        return bool(stage_status["full_success"])
    if "full_success" in summary:
        return bool(summary["full_success"])
    return bool(summary.get("passed", result.get("passed", False)))


def _minimum_margin(result: Mapping[str, Any]) -> float:
    summary = _summary(result)
    metrics = _metrics(result)
    direct = _finite_metric(
        (metrics, summary, result),
        (
            "minimum_normalized_margin",
            "min_normalized_margin",
            "minimum_acceptance_margin",
            "minimum_margin",
        ),
        default=-math.inf,
    )
    if math.isfinite(direct):
        return direct
    for container in (metrics, summary, result):
        for name in (
            "normalized_acceptance_margins",
            "acceptance_margins",
            "hard_constraint_margins",
        ):
            margins = container.get(name)
            if not isinstance(margins, Mapping) or not margins:
                continue
            values: list[float] = []
            for value in margins.values():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    values = []
                    break
                if not math.isfinite(number):
                    values = []
                    break
                values.append(number)
            if values:
                return min(values)
    config = result.get("config")
    if isinstance(config, Mapping):
        acceptance = config.get("acceptance")
        if isinstance(acceptance, dict):
            try:
                # Local import avoids making the pure planning module part of
                # search.py's import graph while still ranking real evaluator
                # summaries by every registered quantitative hard threshold.
                from .search import normalized_acceptance_margins

                margins = normalized_acceptance_margins(
                    dict(metrics),
                    acceptance,
                    contact_topology=(
                        config.get("contact_topology")
                        if isinstance(config.get("contact_topology"), dict)
                        else None
                    ),
                    contact_alignment=(
                        config.get("contact_alignment")
                        if isinstance(config.get("contact_alignment"), dict)
                        else None
                    ),
                    pose_constraints=(
                        config.get("pose_constraints")
                        if isinstance(config.get("pose_constraints"), dict)
                        else None
                    ),
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                pass
            else:
                finite = [
                    float(value)
                    for value in margins.values()
                    if math.isfinite(float(value))
                ]
                if finite:
                    return min(finite)
    return -math.inf


def aligned_contact_candidate_rank(
    result: Mapping[str, Any],
) -> tuple[float, ...]:
    """Return the canonical descending v4 candidate rank.

    Order is exactly: hard pass, minimum hard-constraint margin, aligned duty,
    lower p95 height spread, topology retention, lower peak force, lower
    orientation drift, lower actuator saturation, then lower candidate ID.
    """

    summary = _summary(result)
    metrics = _metrics(result)
    containers = (metrics, summary, result)
    alignment = metrics.get("contact_alignment")
    operation_alignment = (
        alignment.get("operation", {}) if isinstance(alignment, Mapping) else {}
    )
    aligned_duty = _finite_metric(
        (operation_alignment, *containers),
        (
            "aligned_duty",
            "operation_aligned_contact_duty",
            "operation_contacts_aligned_duty",
            "operation_aligned_duty",
            "aligned_contact_duty",
        ),
        default=-math.inf,
    )
    p95_spread = _finite_metric(
        (operation_alignment, *containers),
        (
            "height_spread_p95_m",
            "operation_contact_height_spread_p95_m",
            "operation_height_spread_p95_m",
            "contact_height_spread_p95_m",
            "aligned_height_spread_p95_m",
        ),
        default=math.inf,
    )
    topology = _finite_metric(
        containers,
        (
            "operation_target_face_simultaneous_duty",
            "target_face_simultaneous_duty",
            "topology_duty",
        ),
        default=-math.inf,
    )
    force = _finite_metric(
        containers,
        (
            "peak_total_distal_contact_force_n",
            "peak_grasp_contact_force_n",
            "peak_contact_force_n",
        ),
        default=math.inf,
    )
    drift = _finite_metric(
        containers,
        (
            "orientation_drift_deg",
            "operation_orientation_drift_deg",
            "max_orientation_drift_deg",
        ),
        default=math.inf,
    )
    saturation = _finite_metric(
        containers,
        ("actuator_saturation_fraction", "saturation_fraction"),
        default=math.inf,
    )
    return (
        float(_hard_pass(result)),
        _minimum_margin(result),
        aligned_duty,
        -p95_spread,
        topology,
        -force,
        -drift,
        -saturation,
        -float(_candidate_id(result)),
    )


# Concise public alias matching earlier search modules.
candidate_rank = aligned_contact_candidate_rank


def deterministic_rank_results(
    results: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Return a worker-order-independent total order with unique IDs."""

    materialized = tuple(results)
    ids = tuple(_candidate_id(result) for result in materialized)
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be unique")
    return tuple(
        sorted(materialized, key=aligned_contact_candidate_rank, reverse=True)
    )


rank_candidates = deterministic_rank_results


def candidate_tilt_band_deg(result: Mapping[str, Any]) -> float:
    """Read a candidate's explicitly persisted tilt-band centre."""

    containers: list[Mapping[str, Any]] = [result]
    for outer in (result.get("metadata"), result.get("search_metadata")):
        if isinstance(outer, Mapping):
            containers.append(outer)
    config = result.get("config")
    if isinstance(config, Mapping):
        for outer in (
            config,
            config.get("metadata"),
            config.get("search_metadata"),
            config.get("candidate_metadata"),
        ):
            if isinstance(outer, Mapping):
                containers.append(outer)
    value = _finite_metric(
        containers,
        (
            "tilt_band_center_deg",
            "tilt_band_deg",
            "finger_down_tilt_center_deg",
            "finger_down_tilt_deg",
        ),
        default=math.nan,
    )
    if not math.isfinite(value):
        raise ValueError("candidate result requires a finite tilt band centre")
    return value


def select_per_band(
    results: Iterable[Mapping[str, Any]],
    per_band: int | None = None,
    *,
    count_per_band: int | None = None,
    band_centers_deg: Sequence[float] | None = None,
    campaign: AlignedContactCampaignParameters = DEFAULT_CAMPAIGN,
) -> dict[float, tuple[Mapping[str, Any], ...]]:
    """Select independently within each band and retain explicit empty bands.

    No candidate from a productive band is ever used to fill a missing or
    undersubscribed band.  The returned mapping follows declared band order.
    """

    if per_band is not None and count_per_band is not None:
        raise ValueError("provide only one of per_band and count_per_band")
    limit = per_band if per_band is not None else count_per_band
    if limit is None:
        limit = campaign.exact_candidates_per_band
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("per_band must be a positive integer")
    centers = tuple(
        float(value)
        for value in (
            campaign.tilt_band_centers_deg
            if band_centers_deg is None
            else band_centers_deg
        )
    )
    if (
        not centers
        or any(not math.isfinite(value) for value in centers)
        or len(set(centers)) != len(centers)
    ):
        raise ValueError("band_centers_deg must be finite and unique")
    ranked = deterministic_rank_results(results)
    selected: dict[float, tuple[Mapping[str, Any], ...]] = {}
    for center in centers:
        matching = tuple(
            result
            for result in ranked
            if math.isclose(
                candidate_tilt_band_deg(result),
                center,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
        selected[center] = matching[:limit]
    return selected


def select_candidates_per_band(
    results: Iterable[Mapping[str, Any]],
    per_band: int | None = None,
    *,
    count_per_band: int | None = None,
    band_centers_deg: Sequence[float] | None = None,
    campaign: AlignedContactCampaignParameters = DEFAULT_CAMPAIGN,
) -> tuple[Mapping[str, Any], ...]:
    """Flatten :func:`select_per_band` in declared band order."""

    groups = select_per_band(
        results,
        per_band,
        count_per_band=count_per_band,
        band_centers_deg=band_centers_deg,
        campaign=campaign,
    )
    return tuple(result for group in groups.values() for result in group)


@dataclass(frozen=True)
class StaticScreenJob:
    edge_m: float
    tilt_band_center_deg: float
    samples: int
    seed: int
    face_sample_counts: tuple[int, int, int, int]


def static_screen_jobs(
    campaign: AlignedContactCampaignParameters = DEFAULT_CAMPAIGN,
    *,
    seed: int = 20260821,
) -> tuple[StaticScreenJob, ...]:
    """Build all 6 x 5 static jobs with balanced lateral-face allocation."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    samples = campaign.static_samples_per_edge_band
    quotient, remainder = divmod(samples, 4)
    face_counts = tuple(
        quotient + (1 if index < remainder else 0) for index in range(4)
    )
    jobs: list[StaticScreenJob] = []
    for edge_index, edge_m in enumerate(campaign.edges_m):
        for band_index, center in enumerate(campaign.tilt_band_centers_deg):
            jobs.append(
                StaticScreenJob(
                    edge_m=edge_m,
                    tilt_band_center_deg=center,
                    samples=samples,
                    seed=seed + edge_index * len(campaign.tilt_band_centers_deg) + band_index,
                    face_sample_counts=face_counts,
                )
            )
    return tuple(jobs)


def budget_manifest(
    campaign: AlignedContactCampaignParameters = DEFAULT_CAMPAIGN,
) -> dict[str, Any]:
    """Return the exact declared v4 work budget and per-band invariants."""

    jobs = static_screen_jobs(campaign)
    return {
        "edges_m": list(campaign.edges_m),
        "tilt_band_centers_deg": list(campaign.tilt_band_centers_deg),
        "stage_order": [
            "static_screen",
            "dynamic_screen",
            "local_grasp_refinement",
            "local_manipulation_refinement",
            "exact_selection",
            "robustness_perturbation",
        ],
        "selection_policy": {
            "bands_are_independent": True,
            "fill_missing_bands": False,
        },
        "per_band": {
            "static_sample_count": (
                len(campaign.edges_m)
                * campaign.static_samples_per_edge_band
            ),
            "dynamic_candidate_count": campaign.dynamic_candidates_per_band,
            "grasp_refine_seed_count": (
                campaign.grasp_refine_seed_count_per_band
            ),
            "grasp_refine_per_seed": campaign.grasp_refine_per_seed,
            "grasp_refinement_count": (
                campaign.grasp_refine_seed_count_per_band
                * campaign.grasp_refine_per_seed
            ),
            "manipulation_seed_count": (
                campaign.manipulation_seed_count_per_band
            ),
            "manipulation_refine_per_seed": (
                campaign.manipulation_refine_per_seed
            ),
            "manipulation_refinement_count": (
                campaign.manipulation_seed_count_per_band
                * campaign.manipulation_refine_per_seed
            ),
            "exact_candidate_count": campaign.exact_candidates_per_band,
        },
        "nominal_counts": {
            "static_job_count": len(jobs),
            "static_sample_count": campaign.static_sample_count,
            "dynamic_candidate_count": campaign.dynamic_candidate_count,
            "grasp_refinement_count": campaign.grasp_refinement_count,
            "manipulation_refinement_count": (
                campaign.manipulation_refinement_count
            ),
            "exact_candidate_count": campaign.exact_candidate_count,
        },
    }


campaign_manifest = budget_manifest


RunCandidates = Callable[
    [list[tuple[int, dict[str, Any]]], int], list[dict[str, Any]]
]
ScreenCandidates = Callable[
    [dict[str, Any], StaticScreenJob, int], Sequence[dict[str, Any]]
]
RefineCandidates = Callable[..., Sequence[dict[str, Any]]]
PerturbCases = Callable[..., Sequence[dict[str, Any]]]


def run_candidate_stage(
    configs: Sequence[dict[str, Any]],
    *,
    next_id: int,
    workers: int,
    run_candidates: RunCandidates,
    stage: str,
) -> tuple[list[dict[str, Any]], int]:
    """Execute one stage while preserving exact candidate-ID/config binding."""

    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    payloads = [
        (next_id + index, copy.deepcopy(config))
        for index, config in enumerate(configs)
    ]
    submitted = {
        candidate_id: copy.deepcopy(config) for candidate_id, config in payloads
    }
    results = list(run_candidates(payloads, workers)) if payloads else []
    expected_ids = tuple(candidate_id for candidate_id, _ in payloads)
    try:
        received_ids = tuple(_candidate_id(result) for result in results)
    except ValueError as error:
        raise RuntimeError(f"{stage} runner returned an invalid candidate_id") from error
    if (
        len(results) != len(payloads)
        or len(received_ids) != len(set(received_ids))
        or set(received_ids) != set(expected_ids)
    ):
        raise RuntimeError(
            f"{stage} runner result IDs must exactly match submitted IDs; "
            f"expected={expected_ids}, received={received_ids}"
        )
    for result in results:
        candidate_id = _candidate_id(result)
        if result.get("config") != submitted[candidate_id]:
            raise RuntimeError(
                f"{stage} runner rebound candidate_id={candidate_id} to a "
                "different configuration"
            )
        result["search_stage"] = stage
    return (
        sorted(results, key=_candidate_id),
        next_id + len(payloads),
    )


def _with_band_metadata(
    config: Mapping[str, Any], band_center_deg: float
) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    metadata = result.setdefault("search_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("candidate search_metadata must be a mapping")
    declared = metadata.get("tilt_band_center_deg")
    if declared is not None and not math.isclose(
        float(declared), band_center_deg, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("screen candidate changed its tilt band")
    metadata["tilt_band_center_deg"] = float(band_center_deg)
    return result


def _stage_success(result: Mapping[str, Any], name: str) -> bool:
    summary = _summary(result)
    status = summary.get("stage_status")
    if isinstance(status, Mapping) and name in status:
        return bool(status[name])
    return bool(summary.get(name, False))


def _positive_override(value: int | None, default: int, label: str) -> int:
    result = default if value is None else value
    if not isinstance(result, int) or isinstance(result, bool) or result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def tune_aligned_contacts(
    config: dict[str, Any],
    *,
    workers: int,
    run_candidates: RunCandidates,
    screen_candidates: ScreenCandidates,
    refine_candidates: RefineCandidates,
    perturb_cases: PerturbCases,
    seed: int | None = None,
    dynamic_candidates_per_band: int | None = None,
    grasp_refine_seed_count_per_band: int | None = None,
    grasp_refine_per_seed: int | None = None,
    manipulation_seed_count_per_band: int | None = None,
    manipulation_refine_per_seed: int | None = None,
    exact_candidates_per_band: int | None = None,
    perturbations_per_final: int | None = None,
) -> dict[str, Any]:
    """Execute the band-preserving schema-v4 campaign through injected engines.

    The callbacks isolate MuJoCo-specific static screening, local sampling and
    perturbation construction.  This function owns the safety-critical parts:
    fixed stage order, exact ID/config binding, grasp gating, deterministic
    ranking, and independent per-band quotas.
    """

    validate_config(config)
    definition = resolve_experiment(config)
    campaign = definition.aligned_contact_campaign
    if definition.experiment_id != EXPERIMENT_ID or campaign is None:
        raise ValueError("selected experiment is not the schema-v4 aligned campaign")
    effective_seed = definition.search_bounds.seed if seed is None else seed
    if (
        not isinstance(effective_seed, int)
        or isinstance(effective_seed, bool)
        or effective_seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    dynamic_per_band = _positive_override(
        dynamic_candidates_per_band,
        campaign.dynamic_candidates_per_band,
        "dynamic_candidates_per_band",
    )
    grasp_parents_per_band = _positive_override(
        grasp_refine_seed_count_per_band,
        campaign.grasp_refine_seed_count_per_band,
        "grasp_refine_seed_count_per_band",
    )
    grasp_per_seed = _positive_override(
        grasp_refine_per_seed,
        campaign.grasp_refine_per_seed,
        "grasp_refine_per_seed",
    )
    manipulation_parents_per_band = _positive_override(
        manipulation_seed_count_per_band,
        campaign.manipulation_seed_count_per_band,
        "manipulation_seed_count_per_band",
    )
    manipulation_per_seed = _positive_override(
        manipulation_refine_per_seed,
        campaign.manipulation_refine_per_seed,
        "manipulation_refine_per_seed",
    )
    exact_per_band = _positive_override(
        exact_candidates_per_band,
        campaign.exact_candidates_per_band,
        "exact_candidates_per_band",
    )
    perturbation_count = _positive_override(
        perturbations_per_final,
        definition.robustness.perturbation_count,
        "perturbations_per_final",
    )

    static_by_edge_band: dict[tuple[float, float], list[dict[str, Any]]] = {
        (edge, center): []
        for edge in campaign.edges_m
        for center in campaign.tilt_band_centers_deg
    }
    for job in static_screen_jobs(campaign, seed=effective_seed):
        screened = screen_candidates(
            copy.deepcopy(config), job, dynamic_per_band
        )
        static_by_edge_band[(job.edge_m, job.tilt_band_center_deg)].extend(
            _with_band_metadata(candidate, job.tilt_band_center_deg)
            for candidate in screened
        )
    dynamic_configs_list: list[dict[str, Any]] = []
    for center in campaign.tilt_band_centers_deg:
        edge_groups = [
            static_by_edge_band[(edge, center)] for edge in campaign.edges_m
        ]
        depth = 0
        band_selected_count = 0
        while band_selected_count < dynamic_per_band:
            advanced = False
            for group in edge_groups:
                if depth < len(group):
                    dynamic_configs_list.append(group[depth])
                    band_selected_count += 1
                    advanced = True
                    if band_selected_count == dynamic_per_band:
                        break
            if not advanced:
                break
            depth += 1
    dynamic_configs = tuple(dynamic_configs_list)

    next_id = 0
    dynamic_results, next_id = run_candidate_stage(
        dynamic_configs,
        next_id=next_id,
        workers=workers,
        run_candidates=run_candidates,
        stage="dynamic_screen",
    )
    grasp_parents = select_candidates_per_band(
        dynamic_results,
        grasp_parents_per_band,
        campaign=campaign,
    )
    grasp_configs: list[dict[str, Any]] = []
    for parent_index, parent in enumerate(grasp_parents):
        band = candidate_tilt_band_deg(parent)
        refined = refine_candidates(
            copy.deepcopy(parent["config"]),
            count=grasp_per_seed,
            band_center_deg=band,
            seed=effective_seed + 1_000_000 + parent_index,
            stage="local_grasp_refinement",
        )
        grasp_configs.extend(_with_band_metadata(item, band) for item in refined)
    grasp_results, next_id = run_candidate_stage(
        grasp_configs,
        next_id=next_id,
        workers=workers,
        run_candidates=run_candidates,
        stage="local_grasp_refinement",
    )

    verified_grasps = tuple(
        result for result in grasp_results if _stage_success(result, "grasp_success")
    )
    manipulation_parents = select_candidates_per_band(
        verified_grasps,
        manipulation_parents_per_band,
        campaign=campaign,
    )
    manipulation_configs: list[dict[str, Any]] = []
    for parent_index, parent in enumerate(manipulation_parents):
        band = candidate_tilt_band_deg(parent)
        refined = refine_candidates(
            copy.deepcopy(parent["config"]),
            count=manipulation_per_seed,
            band_center_deg=band,
            seed=effective_seed + 2_000_000 + parent_index,
            stage="local_manipulation_refinement",
        )
        manipulation_configs.extend(
            _with_band_metadata(item, band) for item in refined
        )
    manipulation_results, next_id = run_candidate_stage(
        manipulation_configs,
        next_id=next_id,
        workers=workers,
        run_candidates=run_candidates,
        stage="local_manipulation_refinement",
    )

    full_passes = tuple(result for result in manipulation_results if _hard_pass(result))
    finalists = select_candidates_per_band(
        full_passes,
        exact_per_band,
        campaign=campaign,
    )
    perturbation_configs: list[dict[str, Any]] = []
    perturbation_owner_ids: list[int] = []
    for finalist_index, finalist in enumerate(finalists):
        cases = tuple(
            perturb_cases(
                copy.deepcopy(finalist["config"]),
                count=perturbation_count,
                seed=effective_seed + 3_000_000 + finalist_index,
            )
        )
        if len(cases) != perturbation_count:
            raise RuntimeError(
                "perturb_cases must return exactly the requested count"
            )
        for case in cases:
            materialized = copy.deepcopy(case)
            materialized["run_context"] = {"kind": "robustness_trial"}
            validate_config(materialized)
            perturbation_configs.append(materialized)
            perturbation_owner_ids.append(_candidate_id(finalist))
    perturbation_results, next_id = run_candidate_stage(
        perturbation_configs,
        next_id=next_id,
        workers=workers,
        run_candidates=run_candidates,
        stage="robustness_perturbation",
    )
    passes_by_finalist = {_candidate_id(finalist): 0 for finalist in finalists}
    for owner_id, result in zip(
        perturbation_owner_ids, perturbation_results, strict=True
    ):
        passes_by_finalist[owner_id] += int(_hard_pass(result))
    robust_finalists = tuple(
        finalist
        for finalist in finalists
        if passes_by_finalist[_candidate_id(finalist)]
        >= definition.robustness.required_pass_count
    )
    return {
        "schema_version": 1,
        "experiment_id": definition.experiment_id,
        "seed": effective_seed,
        "declared_budget": budget_manifest(campaign),
        "actual_counts": {
            "dynamic_candidate_count": len(dynamic_results),
            "grasp_refinement_count": len(grasp_results),
            "manipulation_refinement_count": len(manipulation_results),
            "exact_candidate_count": len(finalists),
            "perturbation_count": len(perturbation_results),
        },
        "dynamic_results": dynamic_results,
        "grasp_results": grasp_results,
        "manipulation_results": manipulation_results,
        "finalists": finalists,
        "perturbation_results": perturbation_results,
        "perturbation_passes_by_finalist": passes_by_finalist,
        "robust_finalists": robust_finalists,
        "campaign_success": bool(robust_finalists),
        "next_candidate_id": next_id,
    }


tune = tune_aligned_contacts


__all__ = [
    "DEFAULT_CAMPAIGN",
    "DEFAULT_PLAN",
    "StaticScreenJob",
    "aligned_contact_candidate_rank",
    "budget_manifest",
    "campaign_manifest",
    "candidate_rank",
    "candidate_tilt_band_deg",
    "deterministic_rank_results",
    "rank_candidates",
    "select_candidates_per_band",
    "select_per_band",
    "static_screen_jobs",
    "run_candidate_stage",
    "tune",
    "tune_aligned_contacts",
]
