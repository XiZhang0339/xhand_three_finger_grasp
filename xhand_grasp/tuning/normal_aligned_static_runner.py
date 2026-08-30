"""Resumable cell scheduler for the schema-v8 static pose search.

The geometric evaluator lives in :mod:`normal_aligned_pose_search`; this
module only binds its declared 55-cell campaign to authenticated v7 source
poses, spawn workers, and content-addressed artifacts.  Every cell commits one
complete ``result.json`` atomically.  A resumed run re-authenticates all inputs
and reuses a cell only when its exact input hash still matches.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..artifacts import file_sha256, write_json
from ..config import load_config, validate_config
from .normal_aligned_pose_search import (
    CAMPAIGN_KIND,
    DEFAULT_ALPHA_COUNT,
    DEFAULT_COARSE_ALPHA_COUNT,
    DEFAULT_COARSE_NEAR_MARGIN_M,
    DEFAULT_SEED,
    EXPERIMENT_ID,
    PoseSearchCell,
    generate_pose_cell_candidates,
    pose_search_cells,
    retain_top_k_per_cell,
    screen_static_pose_chunk_two_stage,
)
from .normal_aligned_smooth_lift import (
    DEFAULT_V7_CAMPAIGN_RESULTS,
    DEFAULT_V8_TEMPLATE,
    build_pose_rescue_manifest,
)
from .pose_preserving_seed_campaign import canonical_sha256


STATIC_RUNNER_SCHEMA_VERSION = 1
STATIC_CELL_RESULT_SCHEMA_VERSION = 1
STATIC_SEARCH_REPORT_SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(
    "artifacts/"
    "left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/"
    "tune/new_pose_static"
)


def _positive_int(value: Any, label: str) -> int:
    integer = int(value)
    if isinstance(value, bool) or integer != value or integer <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return integer


@dataclass(frozen=True, slots=True)
class StaticSearchRunnerBudget:
    """Execution details around the registered 550,000-sample campaign."""

    samples_per_cell: int = 10_000
    retain_per_cell: int = 4
    alpha_count: int = DEFAULT_ALPHA_COUNT
    coarse_alpha_count: int = DEFAULT_COARSE_ALPHA_COUNT
    promotion_per_chunk: int = 16
    coarse_near_margin_m: float = DEFAULT_COARSE_NEAR_MARGIN_M
    chunk_size: int = 256

    def __post_init__(self) -> None:
        for name in (
            "samples_per_cell",
            "retain_per_cell",
            "alpha_count",
            "coarse_alpha_count",
            "promotion_per_chunk",
            "chunk_size",
        ):
            value = getattr(self, name)
            _positive_int(value, name)
        if self.alpha_count < 3:
            raise ValueError("alpha_count must be at least three")
        if self.coarse_alpha_count < 3:
            raise ValueError("coarse_alpha_count must be at least three")
        if self.promotion_per_chunk < self.retain_per_cell:
            raise ValueError("promotion_per_chunk must be at least retain_per_cell")
        margin = float(self.coarse_near_margin_m)
        if not 0.0 <= margin < 0.050:
            raise ValueError("coarse_near_margin_m must lie within [0, 0.050)")
        object.__setattr__(self, "coarse_near_margin_m", margin)
        if self.samples_per_cell >= 1_000_000:
            raise ValueError("samples_per_cell must be smaller than the cell stride")

    def report(
        self, *, registered_cell_count: int, selected_cell_count: int
    ) -> dict[str, Any]:
        registered = _positive_int(registered_cell_count, "registered_cell_count")
        selected = _positive_int(selected_cell_count, "selected_cell_count")
        return {
            **asdict(self),
            "registered_cell_count": registered,
            "selected_cell_count": selected,
            "registered_declared_sample_count": registered * self.samples_per_cell,
            "selected_declared_sample_count": selected * self.samples_per_cell,
            "selected_maximum_retained_count": selected * self.retain_per_cell,
        }


def _source_records(pose_manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    poses = pose_manifest.get("poses")
    if not isinstance(poses, list) or not poses:
        raise ValueError("authenticated pose manifest contains no source poses")
    records: list[dict[str, Any]] = []
    for raw in poses:
        if not isinstance(raw, Mapping):
            raise ValueError("pose manifest records must be mappings")
        records.append(
            {
                "pose_id": str(raw["pose_id"]),
                "source_config": str(raw["source_config"]),
                "source_config_file_sha256": str(
                    raw["source_config_file_sha256"]
                ),
                "source_config_sha256": str(raw["source_config_sha256"]),
            }
        )
    if len({value["pose_id"] for value in records}) != len(records):
        raise ValueError("authenticated pose manifest contains duplicate pose IDs")
    return tuple(records)


def _load_source_configs(
    source_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Re-authenticate source files inside each spawned worker."""

    sources: list[dict[str, Any]] = []
    for record in source_records:
        path = Path(str(record["source_config"])).expanduser().resolve()
        if file_sha256(path) != str(record["source_config_file_sha256"]):
            raise RuntimeError(f"source config file hash changed: {path}")
        config = load_config(path)
        if canonical_sha256(config) != str(record["source_config_sha256"]):
            raise RuntimeError(f"source config semantic hash changed: {path}")
        sources.append({"pose_id": str(record["pose_id"]), "config": config})
    return tuple(sources)


def _cell_from_job(job: Mapping[str, Any]) -> PoseSearchCell:
    return PoseSearchCell(
        int(job["cell_index"]),
        float(job["edge_m"]),
        float(job["thumb_target_rad"]),
    )


def execute_static_search_cell(job: Mapping[str, Any]) -> dict[str, Any]:
    """Generate, screen, and atomically commit one independent search cell."""

    result_path = Path(str(job["result_path"])).expanduser().resolve()
    if result_path.is_file():
        raise FileExistsError(f"static cell result already exists: {result_path}")
    budget = StaticSearchRunnerBudget(**dict(job["budget"]))
    cell = _cell_from_job(job)
    template = copy.deepcopy(dict(job["template"]))
    sources = _load_source_configs(job["source_records"])

    retained_pool: list[dict[str, Any]] = []
    evaluated_count = 0
    coarse_scan_count = 0
    full_scan_count = 0
    not_promoted_count = 0
    coarse_sampled_pass_hint_count = 0
    coarse_safe_eligible_count = 0
    static_pass_count = 0
    chunk_count = 0
    for start in range(0, budget.samples_per_cell, budget.chunk_size):
        count = min(budget.chunk_size, budget.samples_per_cell - start)
        candidates = generate_pose_cell_candidates(
            template,
            sources,
            cell,
            count=count,
            start_index=start,
            seed=int(job["seed"]),
        )
        screened = screen_static_pose_chunk_two_stage(
            candidates,
            top_k=min(count, budget.retain_per_cell),
            alpha_count=budget.alpha_count,
            coarse_alpha_count=budget.coarse_alpha_count,
            promotion_count=min(count, budget.promotion_per_chunk),
            coarse_near_margin_m=budget.coarse_near_margin_m,
        )
        evaluated_count += int(screened["evaluated_count"])
        coarse_scan_count += int(screened["coarse_scan_count"])
        full_scan_count += int(screened["full_scan_count"])
        not_promoted_count += int(screened["not_promoted_count"])
        coarse_sampled_pass_hint_count += int(
            screened["coarse_sampled_pass_hint_count"]
        )
        coarse_safe_eligible_count += int(
            screened["coarse_safe_eligible_count"]
        )
        static_pass_count += int(screened["static_pass_count"])
        retained_pool.extend(
            copy.deepcopy(dict(value)) for value in screened["retained"]
        )
        chunk_count += 1

    retained_by_cell = retain_top_k_per_cell(
        retained_pool, top_k=budget.retain_per_cell
    )
    retained = retained_by_cell.get(cell.cell_index, ())
    for record in retained:
        validate_config(record["config"])
    if evaluated_count != budget.samples_per_cell:
        raise RuntimeError("static cell did not evaluate its declared sample budget")
    if coarse_scan_count != evaluated_count:
        raise RuntimeError("every generated candidate must receive one coarse scan")
    if full_scan_count + not_promoted_count != evaluated_count:
        raise RuntimeError("two-stage static scan accounting is inconsistent")

    payload = {
        "static_cell_result_schema_version": STATIC_CELL_RESULT_SCHEMA_VERSION,
        "complete": True,
        "campaign_input_sha256": str(job["campaign_input_sha256"]),
        "cell_input_sha256": str(job["cell_input_sha256"]),
        **cell.as_dict(),
        "seed": int(job["seed"]),
        "budget": asdict(budget),
        "chunk_count": chunk_count,
        "evaluated_count": evaluated_count,
        "coarse_scan_count": coarse_scan_count,
        "full_scan_count": full_scan_count,
        "not_promoted_count": not_promoted_count,
        "coarse_sampled_pass_hint_count": coarse_sampled_pass_hint_count,
        "coarse_safe_eligible_count": coarse_safe_eligible_count,
        "static_pass_count": static_pass_count,
        "static_pass_count_basis": "promoted_candidates_full_scan_only",
        "retained_count": len(retained),
        "retained": [copy.deepcopy(dict(value)) for value in retained],
        "ranking_policy": (
            "coarse_promotion_then_full_scan_only_hard_pass_missing_offtarget_"
            "nondistal_gap_angle_inward_height_pose_controller_candidate_id"
        ),
        "dynamics_scheduled": False,
    }
    write_json(result_path, payload)
    return payload


def run_static_search_cell_jobs(
    jobs: Sequence[Mapping[str, Any]], workers: int
) -> tuple[dict[str, Any], ...]:
    """Execute cells serially or with a deterministic spawn process pool."""

    worker_count = _positive_int(workers, "workers")
    if not jobs:
        return ()
    if worker_count == 1:
        results = [execute_static_search_cell(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context
        ) as executor:
            results = list(
                executor.map(execute_static_search_cell, jobs, chunksize=1)
            )
    results.sort(key=lambda value: int(value["cell_index"]))
    return tuple(results)


def _load_reusable_cell(job: Mapping[str, Any]) -> dict[str, Any] | None:
    path = Path(str(job["result_path"])).expanduser().resolve()
    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError(f"static cell result is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"static cell result is unreadable: {path}") from error
    if not isinstance(payload, Mapping) or payload.get("complete") is not True:
        raise RuntimeError(f"static cell result is incomplete: {path}")
    exact = {
        "campaign_input_sha256": str(job["campaign_input_sha256"]),
        "cell_input_sha256": str(job["cell_input_sha256"]),
        "cell_index": int(job["cell_index"]),
        "cell_id": str(job["cell_id"]),
        "evaluated_count": int(job["budget"]["samples_per_cell"]),
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"static cell {key} changed: {path}")
    retained = payload.get("retained")
    if not isinstance(retained, list):
        raise RuntimeError(f"static cell retained records are missing: {path}")
    if int(payload.get("retained_count", -1)) != len(retained):
        raise RuntimeError(f"static cell retained count is inconsistent: {path}")
    evaluated = int(payload["evaluated_count"])
    if int(payload.get("coarse_scan_count", -1)) != evaluated:
        raise RuntimeError(f"static cell coarse scan count is inconsistent: {path}")
    full = int(payload.get("full_scan_count", -1))
    not_promoted = int(payload.get("not_promoted_count", -1))
    if full < 0 or not_promoted < 0 or full + not_promoted != evaluated:
        raise RuntimeError(f"static cell full scan count is inconsistent: {path}")
    for record in retained:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("config"), Mapping
        ):
            raise RuntimeError(f"static cell retained config is missing: {path}")
        if canonical_sha256(record["config"]) != record.get("candidate_sha256"):
            raise RuntimeError(f"static cell retained config hash changed: {path}")
    return copy.deepcopy(dict(payload))


def _selected_cells(
    registered: Sequence[PoseSearchCell], cell_indices: Sequence[int] | None
) -> tuple[PoseSearchCell, ...]:
    if cell_indices is None:
        return tuple(registered)
    indices = tuple(int(value) for value in cell_indices)
    if not indices:
        raise ValueError("cell_indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("cell_indices must not contain duplicates")
    by_index = {cell.cell_index: cell for cell in registered}
    try:
        return tuple(by_index[index] for index in sorted(indices))
    except KeyError as error:
        raise ValueError(f"cell index is outside the registered grid: {error.args[0]}") from error


def _retained_config_bindings(
    output: Path, cell_results: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for result in sorted(cell_results, key=lambda value: int(value["cell_index"])):
        for rank, record in enumerate(result["retained"]):
            candidate_id = int(record["candidate_id"])
            relative = (
                Path("retained_configs")
                / f"cell_{int(result['cell_index']):02d}"
                / f"rank_{rank:02d}_candidate_{candidate_id}"
                / "resolved_config.json"
            )
            path = output / relative
            write_json(path, record["config"])
            bindings.append(
                {
                    "cell_index": int(result["cell_index"]),
                    "cell_id": str(result["cell_id"]),
                    "rank": rank,
                    "candidate_id": candidate_id,
                    "candidate_sha256": str(record["candidate_sha256"]),
                    "pose_id": str(record["pose_id"]),
                    "controller_id": str(record["controller_id"]),
                    "static_pass": bool(record["static_pass"]),
                    "config_path": str(relative),
                    "config_file_sha256": file_sha256(path),
                }
            )
    return bindings


def run_normal_aligned_static_search(
    campaign_results_path: str | Path = DEFAULT_V7_CAMPAIGN_RESULTS,
    template_path: str | Path = DEFAULT_V8_TEMPLATE,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    workers: int = 1,
    resume: bool = False,
    dry_run: bool = False,
    seed: int = DEFAULT_SEED,
    budget: StaticSearchRunnerBudget | None = None,
    cell_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run or resume the authenticated schema-v8 static pose campaign."""

    worker_count = _positive_int(workers, "workers")
    resolved_budget = budget or StaticSearchRunnerBudget()
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    source_campaign = Path(campaign_results_path).expanduser().resolve()
    template_file = Path(template_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    template = load_config(template_file)
    if (
        int(template.get("schema_version", 0)) != 8
        or template.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("template must be the registered schema-v8 experiment")
    pose_manifest = build_pose_rescue_manifest(source_campaign)
    source_records = _source_records(pose_manifest)
    registered = pose_search_cells(template)
    selected = _selected_cells(registered, cell_indices)

    input_payload = {
        "static_runner_schema_version": STATIC_RUNNER_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        "stage": "new_pose_static",
        "seed": int(seed),
        "budget": resolved_budget.report(
            registered_cell_count=len(registered), selected_cell_count=len(selected)
        ),
        "selected_cells": [cell.as_dict() for cell in selected],
        "source_campaign_results_sha256": file_sha256(source_campaign),
        "pose_manifest_sha256": canonical_sha256(pose_manifest),
        "source_config_hashes": [
            {
                "pose_id": value["pose_id"],
                "file_sha256": value["source_config_file_sha256"],
                "semantic_sha256": value["source_config_sha256"],
            }
            for value in source_records
        ],
        "template_file_sha256": file_sha256(template_file),
        "template_semantic_sha256": canonical_sha256(template),
        "search_core_source_sha256": file_sha256(
            Path(__file__).with_name("normal_aligned_pose_search.py")
        ),
        "runner_source_sha256": file_sha256(Path(__file__)),
    }
    campaign_input_sha256 = canonical_sha256(input_payload)
    plan = {
        **input_payload,
        "campaign_input_sha256": campaign_input_sha256,
        "source_pose_count": len(source_records),
        "output_directory": str(output),
    }
    if dry_run:
        return {**plan, "dry_run": True, "output_directory_created": False}

    manifest_path = output / "campaign_manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"output directory already exists: {output}; pass --resume"
            )
        if not manifest_path.is_file():
            raise RuntimeError("resume output has no campaign_manifest.json")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("campaign_input_sha256") != campaign_input_sha256:
            raise RuntimeError(
                "resume campaign inputs do not match the existing manifest"
            )
    else:
        output.mkdir(parents=True)
        write_json(output / "source_pose_manifest.json", pose_manifest)
        write_json(
            manifest_path,
            {
                **plan,
                "source_campaign_results": str(source_campaign),
                "template": str(template_file),
                "complete": False,
            },
        )

    jobs: list[dict[str, Any]] = []
    for cell in selected:
        relative = Path("cells") / f"cell_{cell.cell_index:02d}" / "result.json"
        cell_input = {
            "campaign_input_sha256": campaign_input_sha256,
            "cell": cell.as_dict(),
            "seed": int(seed),
            "budget": asdict(resolved_budget),
        }
        jobs.append(
            {
                **cell.as_dict(),
                "campaign_input_sha256": campaign_input_sha256,
                "cell_input_sha256": canonical_sha256(cell_input),
                "seed": int(seed),
                "budget": asdict(resolved_budget),
                "template": template,
                "source_records": source_records,
                "result_path": str(output / relative),
                "result_relative_path": str(relative),
            }
        )

    complete: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for job in jobs:
        reusable = _load_reusable_cell(job) if resume else None
        (complete if reusable is not None else pending).append(reusable or job)
    executed = run_static_search_cell_jobs(tuple(pending), worker_count)
    complete.extend(copy.deepcopy(dict(value)) for value in executed)
    complete.sort(key=lambda value: int(value["cell_index"]))
    expected_indices = {cell.cell_index for cell in selected}
    if {int(value["cell_index"]) for value in complete} != expected_indices:
        raise RuntimeError("static cell executor did not return every selected cell")

    jobs_by_index = {int(job["cell_index"]): job for job in jobs}
    cell_bindings: list[dict[str, Any]] = []
    for result in complete:
        index = int(result["cell_index"])
        job = jobs_by_index[index]
        path = Path(str(job["result_path"])).resolve()
        if result.get("cell_input_sha256") != job["cell_input_sha256"]:
            raise RuntimeError("static cell executor rebound a different cell input")
        cell_bindings.append(
            {
                "cell_index": index,
                "cell_id": str(result["cell_id"]),
                "result_path": str(job["result_relative_path"]),
                "result_sha256": file_sha256(path),
                "evaluated_count": int(result["evaluated_count"]),
                "coarse_scan_count": int(result["coarse_scan_count"]),
                "full_scan_count": int(result["full_scan_count"]),
                "not_promoted_count": int(result["not_promoted_count"]),
                "coarse_sampled_pass_hint_count": int(
                    result["coarse_sampled_pass_hint_count"]
                ),
                "coarse_safe_eligible_count": int(
                    result["coarse_safe_eligible_count"]
                ),
                "static_pass_count": int(result["static_pass_count"]),
                "retained_count": int(result["retained_count"]),
            }
        )

    retained_bindings = _retained_config_bindings(output, complete)
    report = {
        "static_search_report_schema_version": STATIC_SEARCH_REPORT_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        "stage": "new_pose_static",
        "complete": True,
        "campaign_input_sha256": campaign_input_sha256,
        "budget": resolved_budget.report(
            registered_cell_count=len(registered), selected_cell_count=len(selected)
        ),
        "source_pose_count": len(source_records),
        "completed_cell_count": len(complete),
        "evaluated_count": sum(
            int(value["evaluated_count"]) for value in complete
        ),
        "coarse_scan_count": sum(
            int(value["coarse_scan_count"]) for value in complete
        ),
        "full_scan_count": sum(
            int(value["full_scan_count"]) for value in complete
        ),
        "not_promoted_count": sum(
            int(value["not_promoted_count"]) for value in complete
        ),
        "coarse_sampled_pass_hint_count": sum(
            int(value["coarse_sampled_pass_hint_count"]) for value in complete
        ),
        "coarse_safe_eligible_count": sum(
            int(value["coarse_safe_eligible_count"]) for value in complete
        ),
        "static_pass_count": sum(
            int(value["static_pass_count"]) for value in complete
        ),
        "static_pass_count_basis": "promoted_candidates_full_scan_only",
        "retained_count": len(retained_bindings),
        "cell_results": cell_bindings,
        "retained_configs": retained_bindings,
        "dynamics_scheduled": False,
        "stop_reason": "selected_static_cells_complete",
    }
    report_path = output / "search_report.json"
    write_json(report_path, report)
    write_json(
        manifest_path,
        {
            **plan,
            "source_campaign_results": str(source_campaign),
            "template": str(template_file),
            "complete": True,
            "search_report": "search_report.json",
            "search_report_sha256": file_sha256(report_path),
            "completed_cell_count": len(complete),
        },
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the resumable schema-v8 normal-aligned static search."
    )
    parser.add_argument(
        "--campaign-results", type=Path, default=DEFAULT_V7_CAMPAIGN_RESULTS
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_V8_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--samples-per-cell", type=int, default=10_000)
    parser.add_argument("--retain-per-cell", type=int, default=4)
    parser.add_argument("--alpha-count", type=int, default=DEFAULT_ALPHA_COUNT)
    parser.add_argument(
        "--coarse-alpha-count", type=int, default=DEFAULT_COARSE_ALPHA_COUNT
    )
    parser.add_argument("--promotion-per-chunk", type=int, default=16)
    parser.add_argument(
        "--coarse-near-margin-mm",
        type=float,
        default=DEFAULT_COARSE_NEAR_MARGIN_M * 1000.0,
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--cell-index",
        type=int,
        action="append",
        help="Run only this registered cell (repeatable; default: all 55).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    budget = StaticSearchRunnerBudget(
        samples_per_cell=arguments.samples_per_cell,
        retain_per_cell=arguments.retain_per_cell,
        alpha_count=arguments.alpha_count,
        coarse_alpha_count=arguments.coarse_alpha_count,
        promotion_per_chunk=arguments.promotion_per_chunk,
        coarse_near_margin_m=arguments.coarse_near_margin_mm / 1000.0,
        chunk_size=arguments.chunk_size,
    )
    result = run_normal_aligned_static_search(
        arguments.campaign_results,
        arguments.template,
        arguments.output,
        workers=arguments.workers,
        resume=arguments.resume,
        dry_run=arguments.dry_run,
        seed=arguments.seed,
        budget=budget,
        cell_indices=arguments.cell_index,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI script.
    sys.exit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "STATIC_CELL_RESULT_SCHEMA_VERSION",
    "STATIC_RUNNER_SCHEMA_VERSION",
    "STATIC_SEARCH_REPORT_SCHEMA_VERSION",
    "StaticSearchRunnerBudget",
    "execute_static_search_cell",
    "main",
    "run_normal_aligned_static_search",
    "run_static_search_cell_jobs",
]
