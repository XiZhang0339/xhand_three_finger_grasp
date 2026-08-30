"""Deterministic perturbation audit for schema-v8 exact lift candidates."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import file_sha256, json_text, write_json
from ..config import validate_config
from ..experiment import resolve_experiment
from ..normal_aligned_smooth_lift_catalog import (
    CatalogCandidate,
    discover_catalog_candidates,
)
from ..search import _run_candidates
from .normal_aligned_smooth_lift import (
    DEFAULT_SEED,
    EXPERIMENT_ID,
    canonical_sha256,
)


ROBUSTNESS_REPORT_SCHEMA_VERSION = 1
DEFAULT_SEARCH_ROOT = Path(
    "artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_"
    "smooth_vertical_lift/tune/lift"
)
DEFAULT_OUTPUT = Path(
    "artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_"
    "smooth_vertical_lift/robustness/perturbation_report.json"
)
FINAL_PERTURBATIONS_PER_EXACT = 16
BEST_PERTURBATION_COUNT = 50

CandidateRunner = Callable[
    [list[tuple[int, dict[str, Any]]], int], list[dict[str, Any]]
]
CandidateDiscoverer = Callable[
    [Sequence[str | Path]], tuple[CatalogCandidate, ...]
]


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    _positive_int(count, "count")
    _positive_int(dimensions, "dimensions")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    values = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        values[:, dimension] = (
            rng.permutation(count) + rng.random(count)
        ) / count
    return values


def _scale(unit: float, bounds: Sequence[float]) -> float:
    return float(float(bounds[0]) + unit * (float(bounds[1]) - float(bounds[0])))


def _derived_seed(seed: int, candidate_id: str, family: int) -> int:
    try:
        identifier = int(candidate_id)
    except (TypeError, ValueError) as error:
        raise ValueError("exact candidate ID must be an integer") from error
    state = np.random.SeedSequence(
        [int(seed), identifier & 0xFFFFFFFF, int(family)]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


def generate_v8_perturbation_configs(
    config: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    source_candidate_id: str | int,
    family: str,
) -> list[dict[str, Any]]:
    """Generate the registered 8-D cube-pose/material perturbation envelope."""

    sample_count = _positive_int(count, "count")
    if family not in {"per_exact_local_16", "best_pose_material_50"}:
        raise ValueError("unknown v8 perturbation family")
    base = copy.deepcopy(dict(config))
    if base.get("run_context") is not None:
        raise ValueError("v8 robustness requires a canonical exact configuration")
    validate_config(base)
    definition = resolve_experiment(base)
    if (
        definition.experiment_id != EXPERIMENT_ID
        or definition.normal_aligned_smooth_lift_campaign is None
    ):
        raise ValueError("v8 robustness requires the normal-aligned experiment")
    parameters = definition.robustness
    matrix = _latin_hypercube(sample_count, 8, seed)
    base_xy = np.asarray(base["cube"]["center_xy_m"], dtype=np.float64)
    base_rpy = np.asarray(base["cube"].get("rpy_deg", [0.0, 0.0, 0.0]))
    base_gap = float(base["cube"].get("z_offset_m", 0.0))
    base_mass = float(base["cube"]["mass_kg"])
    base_friction = float(base["cube"]["friction"])
    cases: list[dict[str, Any]] = []
    for trial, row in enumerate(matrix):
        xy_delta = np.asarray(
            [
                _scale(row[0], parameters.position_xy_delta_m),
                _scale(row[1], parameters.position_xy_delta_m),
            ],
            dtype=np.float64,
        )
        gap_delta = _scale(row[2], parameters.z_offset_delta_m)
        rpy_delta = np.asarray(
            [_scale(row[3 + axis], parameters.rpy_delta_deg) for axis in range(3)],
            dtype=np.float64,
        )
        mass_scale = _scale(row[6], parameters.mass_scale)
        friction_delta = _scale(row[7], parameters.friction_delta)
        case = copy.deepcopy(base)
        case.pop("experiment_status", None)
        case["run_context"] = {"kind": "robustness_trial"}
        case["cube"]["center_xy_m"] = (base_xy + xy_delta).tolist()
        case["cube"]["z_offset_m"] = base_gap + gap_delta
        case["cube"]["rpy_deg"] = (base_rpy + rpy_delta).tolist()
        case["cube"]["mass_kg"] = base_mass * mass_scale
        case["cube"]["friction"] = base_friction + friction_delta
        resolved = {
            "cube_center_xy_delta_m": xy_delta.tolist(),
            "cube_gap_delta_m": float(gap_delta),
            "cube_rpy_delta_deg": rpy_delta.tolist(),
            "mass_scale": float(mass_scale),
            "friction_delta": float(friction_delta),
        }
        metadata = copy.deepcopy(dict(case.get("candidate_metadata", {})))
        metadata["robustness_trial"] = {
            "family": family,
            "seed": int(seed),
            "trial": int(trial),
            "source_candidate_id": str(source_candidate_id),
            "resolved_perturbations": resolved,
        }
        case["candidate_metadata"] = metadata
        validate_config(case)
        cases.append(case)
    return cases


def _hard_pass(summary: Mapping[str, Any]) -> bool:
    stage = summary.get("stage_status", {})
    return bool(
        summary.get("passed", False)
        and isinstance(stage, Mapping)
        and stage.get("grasp_success", False)
        and stage.get("manipulation_success", False)
        and stage.get("full_success", False)
    )


def _source_record(candidate: CatalogCandidate) -> dict[str, Any]:
    summary = candidate.result.get("summary", {})
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_sha256": candidate.candidate_sha256,
        "pose_id": candidate.pose_id,
        "controller_id": candidate.controller_id,
        "stage": candidate.stage,
        "edge_mm": candidate.edge_mm,
        "thumb_target_rad": candidate.thumb_target_rad,
        "nominal_full_success": _hard_pass(summary),
        "nominal_failed_checks": copy.deepcopy(summary.get("failed_checks", [])),
        "artifacts": {
            "result": str(candidate.result_path),
            "config": str(candidate.config_path),
            "trace": str(candidate.trace_path) if candidate.trace_path else None,
            "sha256": {
                "result": file_sha256(candidate.result_path),
                "config": file_sha256(candidate.config_path),
                "trace": (
                    file_sha256(candidate.trace_path)
                    if candidate.trace_path is not None
                    else None
                ),
            },
        },
    }


def _trial_record(
    result: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    summary = result.get("summary", {})
    config = result.get("config", {})
    robustness = config.get("candidate_metadata", {}).get("robustness_trial", {})
    return {
        "trial": int(metadata["trial"]),
        "run_candidate_id": int(result["candidate_id"]),
        "source_candidate_id": str(metadata["source_candidate_id"]),
        "family": str(metadata["family"]),
        "seed": int(metadata["seed"]),
        "passed": _hard_pass(summary),
        "config_sha256": canonical_sha256(config),
        "failed_checks": copy.deepcopy(summary.get("failed_checks", [])),
        "checks": copy.deepcopy(summary.get("checks", {})),
        "stage_status": copy.deepcopy(summary.get("stage_status", {})),
        "metrics": copy.deepcopy(summary.get("metrics", {})),
        "error": summary.get("error"),
        "cube": copy.deepcopy(config.get("cube", {})),
        "resolved_perturbations": copy.deepcopy(
            robustness.get("resolved_perturbations", {})
        ),
    }


def run_v8_robustness_campaign(
    search_roots: Sequence[str | Path],
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    workers: int,
    seed: int = DEFAULT_SEED,
    final_perturbations: int = FINAL_PERTURBATIONS_PER_EXACT,
    best_perturbations: int = BEST_PERTURBATION_COUNT,
    candidate_discoverer: CandidateDiscoverer | None = None,
    runner: CandidateRunner | None = None,
) -> dict[str, Any]:
    """Run 16-per-exact and 50-best perturbations without promoting near misses."""

    worker_count = _positive_int(workers, "workers")
    final_count = _positive_int(final_perturbations, "final_perturbations")
    best_count = _positive_int(best_perturbations, "best_perturbations")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    discoverer = discover_catalog_candidates if candidate_discoverer is None else candidate_discoverer
    execute = _run_candidates if runner is None else runner
    roots = tuple(Path(value).expanduser().resolve() for value in search_roots)
    candidates = tuple(sorted(discoverer(roots), key=lambda value: value.rank))
    if not candidates:
        raise ValueError("no authenticated lift_exact candidates were found")
    if any(candidate.stage != "lift_exact" for candidate in candidates):
        raise ValueError("v8 robustness accepts only lift_exact candidates")

    jobs: list[tuple[int, dict[str, Any]]] = []
    job_metadata: dict[int, dict[str, Any]] = {}
    per_exact_seeds: dict[str, int] = {}
    for source_index, candidate in enumerate(candidates):
        local_seed = _derived_seed(seed, candidate.candidate_id, 16)
        per_exact_seeds[candidate.candidate_id] = local_seed
        configs = generate_v8_perturbation_configs(
            candidate.config,
            count=final_count,
            seed=local_seed,
            source_candidate_id=candidate.candidate_id,
            family="per_exact_local_16",
        )
        for trial, config in enumerate(configs):
            run_id = (source_index + 1) * 1_000_000 + trial
            jobs.append((run_id, config))
            job_metadata[run_id] = {
                "trial": trial,
                "source_candidate_id": candidate.candidate_id,
                "family": "per_exact_local_16",
                "seed": local_seed,
            }

    best = candidates[0]
    best_seed = _derived_seed(seed, best.candidate_id, 50)
    best_configs = generate_v8_perturbation_configs(
        best.config,
        count=best_count,
        seed=best_seed,
        source_candidate_id=best.candidate_id,
        family="best_pose_material_50",
    )
    for trial, config in enumerate(best_configs):
        run_id = 900_000_000 + trial
        jobs.append((run_id, config))
        job_metadata[run_id] = {
            "trial": trial,
            "source_candidate_id": best.candidate_id,
            "family": "best_pose_material_50",
            "seed": best_seed,
        }

    raw_results = execute(jobs, worker_count)
    expected_ids = {candidate_id for candidate_id, _ in jobs}
    actual_ids = {int(result.get("candidate_id", -1)) for result in raw_results}
    if actual_ids != expected_ids or len(raw_results) != len(jobs):
        raise RuntimeError("robustness runner did not preserve the complete job set")
    records = {
        int(result["candidate_id"]): _trial_record(
            result, job_metadata[int(result["candidate_id"])]
        )
        for result in raw_results
    }

    per_exact: list[dict[str, Any]] = []
    for candidate in candidates:
        trials = sorted(
            (
                record
                for record in records.values()
                if record["family"] == "per_exact_local_16"
                and record["source_candidate_id"] == candidate.candidate_id
            ),
            key=lambda record: record["trial"],
        )
        source = _source_record(candidate)
        per_exact.append(
            {
                **source,
                "perturbation_seed": per_exact_seeds[candidate.candidate_id],
                "perturbation_count": len(trials),
                "perturbation_passes": sum(record["passed"] for record in trials),
                "trials": trials,
            }
        )

    best_trials = sorted(
        (
            record
            for record in records.values()
            if record["family"] == "best_pose_material_50"
        ),
        key=lambda record: record["trial"],
    )
    best_source = _source_record(best)
    best_passes = sum(record["passed"] for record in best_trials)
    required_passes = int(resolve_experiment(best.config).robustness.required_pass_count)
    registered_budget_complete = bool(
        final_count == FINAL_PERTURBATIONS_PER_EXACT
        and best_count == BEST_PERTURBATION_COUNT
        and all(
            record["perturbation_count"] == FINAL_PERTURBATIONS_PER_EXACT
            for record in per_exact
        )
        and len(best_trials) == BEST_PERTURBATION_COUNT
    )
    robust_passed = bool(
        best_source["nominal_full_success"]
        and registered_budget_complete
        and best_passes >= required_passes
    )
    report = {
        "v8_robustness_report_schema_version": ROBUSTNESS_REPORT_SCHEMA_VERSION,
        "complete": True,
        "experiment_id": EXPERIMENT_ID,
        "seed": int(seed),
        "workers": worker_count,
        "search_roots": [str(value) for value in roots],
        "source_exact_count": len(candidates),
        "source_exact_candidate_ids": [value.candidate_id for value in candidates],
        "nominal_success_count": sum(
            _hard_pass(candidate.result.get("summary", {})) for candidate in candidates
        ),
        "final_perturbations_per_exact": final_count,
        "best_perturbation_count": best_count,
        "registered_budget_complete": registered_budget_complete,
        "total_perturbation_count": len(records),
        "total_perturbation_passes": sum(record["passed"] for record in records.values()),
        "per_exact": per_exact,
        "best_robustness": {
            **best_source,
            "perturbation_seed": best_seed,
            "perturbation_count": len(best_trials),
            "perturbation_passes": best_passes,
            "required_perturbation_passes": required_passes,
            "robust_passed": robust_passed,
            "diagnostic_only_due_to_nominal_failure": not bool(
                best_source["nominal_full_success"]
            ),
            "trials": best_trials,
        },
        "robust_passed": robust_passed,
        "stop_reason": (
            "robust_passed"
            if robust_passed
            else (
                "nominal_exact_candidate_failed_hard_constraints"
                if not best_source["nominal_full_success"]
                else (
                    "registered_perturbation_budget_incomplete"
                    if not registered_budget_complete
                    else "perturbation_pass_count_below_requirement"
                )
            )
        ),
        "provenance": {
            "implementation_sha256": file_sha256(Path(__file__)),
            "model_sha256": file_sha256(Path(__file__).resolve().parents[2] / "xhand_left.xml"),
            "uv_lock_sha256": file_sha256(Path(__file__).resolve().parents[2] / "uv.lock"),
        },
    }
    output = Path(output_path).expanduser().resolve()
    write_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run schema-v8 exact-candidate perturbation diagnostics"
    )
    parser.add_argument(
        "search_roots",
        nargs="*",
        default=[str(DEFAULT_SEARCH_ROOT)],
        help="directories containing authenticated lift_exact candidates",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--final-perturbations",
        type=int,
        default=FINAL_PERTURBATIONS_PER_EXACT,
    )
    parser.add_argument(
        "--best-perturbations", type=int, default=BEST_PERTURBATION_COUNT
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_v8_robustness_campaign(
        args.search_roots,
        args.output,
        workers=args.workers,
        seed=args.seed,
        final_perturbations=args.final_perturbations,
        best_perturbations=args.best_perturbations,
    )
    print(
        json_text(
            {
                key: value
                for key, value in report.items()
                if key not in {"per_exact", "best_robustness"}
            }
        )
    )
    return 0 if report["robust_passed"] else 2


__all__ = [
    "BEST_PERTURBATION_COUNT",
    "DEFAULT_OUTPUT",
    "DEFAULT_SEARCH_ROOT",
    "FINAL_PERTURBATIONS_PER_EXACT",
    "ROBUSTNESS_REPORT_SCHEMA_VERSION",
    "build_parser",
    "generate_v8_perturbation_configs",
    "main",
    "run_v8_robustness_campaign",
]
