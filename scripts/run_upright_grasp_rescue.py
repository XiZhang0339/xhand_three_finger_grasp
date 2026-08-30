#!/usr/bin/env python3
"""Run/resume the authenticated v14 upright-grasp rescue campaign."""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.actual_contact_grasp_pose_catalog import (
    authenticated_catalog_artifact_paths,
)
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.contact_preserving_candidate_artifacts import (
    run_or_resume_v14_candidate_artifacts,
)
from xhand_grasp.tuning.contact_preserving_planned_lift_campaign import (
    publish_viewer_catalogs,
)
from xhand_grasp.tuning.upright_grasp_self_collision_rescue import (
    UprightGraspRescueBudget,
    authenticate_upright_grasp_rescue_source,
    build_upright_grasp_rescue_jobs,
    rank_upright_grasp_rescue_records,
    run_upright_grasp_rescue_job,
    upright_grasp_audited_session_factory,
)


def _run(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    job, output_root = task
    bundle = run_upright_grasp_rescue_job(
        job,
        output_root,
        final_rerun=False,
        retain_grasp_success=False,
    )
    return copy.deepcopy(bundle.result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp-source", type=Path, required=True)
    parser.add_argument("--plan-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--grasp-geometry-blend", type=float)
    parser.add_argument(
        "--keep-plan-close-timing",
        action="store_true",
        help="retain the manipulation source close timing instead of the grasp source timing",
    )
    parser.add_argument(
        "--directed-thumb-branch-best",
        action="store_true",
        help=(
            "reproduce the bounded no-self-collision candidate with a 25%% "
            "upright grasp blend, limited index side-sway and the best tested "
            "thumb contact-branch offset"
        ),
    )
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = authenticate_upright_grasp_rescue_source(
        args.grasp_source, args.plan_source
    )
    if args.directed_thumb_branch_best:
        if args.grasp_geometry_blend is not None or args.keep_plan_close_timing:
            parser.error(
                "--directed-thumb-branch-best cannot be combined with "
                "--grasp-geometry-blend or --keep-plan-close-timing"
            )
        budget = UprightGraspRescueBudget(
            candidate_count=(
                1 if args.candidate_count is None else args.candidate_count
            ),
            seed=args.seed,
            grasp_geometry_blend_fraction=0.25,
            use_grasp_close_timing=False,
            index_joint2_plan_scale=0.9,
            fixed_wrist_local_rotvec_deg=(0.0, 0.0, -0.1),
            fixed_nominal_offset_rad=(
                0.0,
                -0.002,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            fixed_precontact_residual_rad=(
                -0.002,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            index_bend_plan_scale_range=(0.4, 0.4),
        )
    else:
        budget = UprightGraspRescueBudget(
            candidate_count=(
                64 if args.candidate_count is None else args.candidate_count
            ),
            seed=args.seed,
            grasp_geometry_blend_fraction=(
                1.0
                if args.grasp_geometry_blend is None
                else args.grasp_geometry_blend
            ),
            use_grasp_close_timing=not args.keep_plan_close_timing,
        )
    jobs = build_upright_grasp_rescue_jobs(source, budget=budget)
    manifest_core = {
        "manifest_schema_version": 1,
        "campaign_kind": "upright_grasp_self_collision_rescue",
        "source": source.as_mapping(),
        "budget": budget.as_mapping(),
        "candidate_ids": [int(job["candidate_id"]) for job in jobs],
    }
    campaign_id = canonical_sha256(manifest_core)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("campaign_id") != campaign_id:
            raise RuntimeError(
                "upright rescue output contains an incompatible campaign manifest"
            )
    else:
        write_json(
            manifest_path,
            {**manifest_core, "campaign_id": campaign_id, "complete": False},
        )
    tasks = tuple((job, str(output)) for job in jobs)
    if args.workers == 1:
        records = tuple(_run(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            records = tuple(executor.map(_run, tasks))
    ranked = rank_upright_grasp_rescue_records(records)
    best_id = int(ranked[0]["candidate_id"])
    best_job = next(job for job in jobs if int(job["candidate_id"]) == best_id)
    final_root = output / "final" / f"candidate_{best_id}"
    final = run_or_resume_v14_candidate_artifacts(
        best_job["config"],
        final_root,
        best_id,
        final_rerun=True,
        retain_grasp_success=False,
        session_factory=upright_grasp_audited_session_factory,
    )
    if final.result.get("summary_sha256") != ranked[0].get("summary_sha256"):
        raise RuntimeError(
            "upright rescue final rerun disagrees with the selected search summary"
        )
    catalog_destination = output / "catalogs" / "target_1"
    expected_catalogs = {
        "grasp_pose": catalog_destination / "grasp_pose" / "catalog.json",
        "manipulation": catalog_destination / "manipulation" / "catalog.json",
    }
    if all(path.is_file() for path in expected_catalogs.values()):
        catalogs = {
            name: str(path.relative_to(output))
            for name, path in expected_catalogs.items()
        }
    elif any(path.parent.exists() for path in expected_catalogs.values()):
        raise RuntimeError("upright rescue contains a partial Viewer catalog")
    else:
        catalog_record = {
            **copy.deepcopy(final.result),
            "artifact_directory": str(final.destination.relative_to(output)),
        }
        catalogs = publish_viewer_catalogs(
            (catalog_record,),
            output,
            catalog_destination,
            experiment_id=str(final.result["experiment_id"]),
            render_videos=False,
        )
    catalog_hashes: dict[str, str] = {}
    for name, relative in catalogs.items():
        catalog_path = output / relative
        authenticated_catalog_artifact_paths(catalog_path)
        catalog_hashes[name] = file_sha256(catalog_path)
    report = {
        "complete": True,
        "campaign_kind": "upright_grasp_self_collision_rescue",
        "source_authentication_id": source.source_authentication_id,
        "candidate_count": len(ranked),
        "ranked_candidate_ids": [int(value["candidate_id"]) for value in ranked],
        "records": list(ranked),
        "best_candidate_id": best_id,
        "best_final_artifacts": {
            "root": str(final.destination),
            "resolved_config": str(final.config_path),
            "result": str(final.result_path),
            "trace": str(final.trace_path),
        },
        "best_final_result": final.result,
        "viewer_catalogs": catalogs,
        "viewer_catalog_sha256": catalog_hashes,
    }
    report_path = output / "search_report.json"
    write_json(report_path, report)
    write_json(
        manifest_path,
        {
            **manifest_core,
            "campaign_id": campaign_id,
            "complete": True,
            "search_report": report_path.name,
            "search_report_sha256": file_sha256(report_path),
        },
    )
    print(report_path)


if __name__ == "__main__":
    main()
