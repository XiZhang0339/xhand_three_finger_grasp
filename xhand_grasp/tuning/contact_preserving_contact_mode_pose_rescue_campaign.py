"""Atomic campaign runner for the bounded schema-v14 contact-mode rescue.

Every search candidate is a retained full-reset run because active-taxel
branch transitions only exist in the raw trace.  The source candidate remains
immutable, every committed stage is ledger authenticated, and the best five
are independently rerun before Viewer catalogs are published.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..actual_contact_grasp_pose_catalog import (
    authenticated_catalog_artifact_paths,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from ..artifacts import file_sha256, write_json
from ..config import load_config
from ..experiment import resolve_experiment
from ..grasp_pose import canonical_sha256
from .contact_preserving_candidate_artifacts import (
    authenticate_v14_candidate_artifacts,
)
from .contact_preserving_contact_mode_pose_rescue import (
    CONTACT_MODE_POSE_RESCUE_SCHEMA_VERSION,
    MAXIMUM_CANDIDATE_COUNT,
    ContactModePoseRescueBudget,
    ContactModePoseSource,
    authenticate_contact_mode_pose_source,
    build_contact_mode_pose_rescue_jobs,
    contact_mode_physical_config_sha256,
    contact_mode_trace_diagnostics,
    rank_contact_mode_pose_records,
    run_contact_mode_pose_rescue_jobs,
)
from .contact_preserving_lift_rescue_campaign import (
    _commit_report,
    _load_committed_report,
    _publish_rescue_viewer_catalogs,
    _run_full_reset_candidate,
)
from .contact_preserving_planned_lift_campaign import (
    build_contact_preserving_planned_lift_manifest,
)


CONTACT_MODE_POSE_CAMPAIGN_SCHEMA_VERSION = 1
CONTACT_MODE_POSE_RESULT_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
PUBLISHED_CANDIDATE_COUNT = 5


def _source_evidence(source: ContactModePoseSource) -> dict[str, str]:
    return {
        name: file_sha256(source.root / name)
        for name in ("resolved_config.json", "result.json", "trace.npz")
    }


def _manifest(
    config_path: Path,
    source: ContactModePoseSource,
    budget: ContactModePoseRescueBudget,
) -> dict[str, Any]:
    base = build_contact_preserving_planned_lift_manifest(
        config_path, seed=int(budget.seed)
    )
    base.pop("campaign_input_sha256", None)
    implementation_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("contact_preserving_contact_mode_pose_rescue.py"),
    )
    base.update(
        {
            "contact_mode_pose_campaign_schema_version": (
                CONTACT_MODE_POSE_CAMPAIGN_SCHEMA_VERSION
            ),
            "campaign_kind": "contact_preserving_contact_mode_pose_rescue",
            "source_candidate_directory": str(source.root),
            "source_authentication": source.as_mapping(),
            "source_evidence_sha256": _source_evidence(source),
            "budget": budget.as_mapping(),
            "implementation_sha256": {
                path.name: file_sha256(path) for path in implementation_paths
            },
            "execution_contract": {
                "search": "fresh_full_reset_retained_trace",
                "ranking": "grasp_then_thumb_taxel_branch_then_jerk",
                "catalog": "independent_fresh_full_reset_rerun",
                "cube_pose_changed": False,
                "manipulation_plan_changed": False,
            },
        }
    )
    return {**base, "campaign_input_sha256": canonical_sha256(base)}


def build_contact_mode_pose_rescue_manifest(
    config_path: str | Path,
    source_candidate_directory: str | Path,
    *,
    candidate_count: int = MAXIMUM_CANDIDATE_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    source = authenticate_contact_mode_pose_source(source_candidate_directory)
    budget = ContactModePoseRescueBudget(
        candidate_count=int(candidate_count), seed=int(seed)
    )
    return _manifest(Path(config_path).expanduser().resolve(), source, budget)


def _assert_stage_input(
    workspace: Path, stage: str, stage_input: Mapping[str, Any]
) -> None:
    record = validate_stage_ledger(workspace).get("stages", {}).get(stage)
    if not isinstance(record, Mapping):
        raise RuntimeError(f"committed contact-mode stage {stage} disappeared")
    if record.get("stage_input_sha256") != canonical_sha256(dict(stage_input)):
        raise RuntimeError(f"committed contact-mode stage {stage} input changed")


def _validate_search_report(
    payload: Mapping[str, Any], *, expected_count: int
) -> None:
    records = payload.get("records")
    if (
        payload.get("complete") is not True
        or int(payload.get("candidate_count", -1)) != int(expected_count)
        or not isinstance(records, list)
        or len(records) != int(expected_count)
    ):
        raise RuntimeError("contact-mode search report is incomplete")
    identifiers = [int(value.get("candidate_id", -1)) for value in records]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("contact-mode search report contains duplicate candidates")
    for rank, record in enumerate(records):
        diagnostics = record.get("contact_mode_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise RuntimeError("contact-mode search record lost trace diagnostics")
        if int(record.get("contact_mode_rank", -1)) != rank:
            raise RuntimeError("contact-mode search rank changed")
    physical = [str(value.get("physical_config_sha256", "")) for value in records]
    if any(len(value) != 64 for value in physical) or len(physical) != len(
        set(physical)
    ):
        raise RuntimeError("contact-mode search lost physical uniqueness")
    baselines = [
        value
        for value in records
        if bool(value.get("is_exact_parent_reproduction_baseline", False))
    ]
    if len(baselines) != 1:
        raise RuntimeError("contact-mode search lost its exact-parent baseline")


def _run_search_stage(
    workspace: Path,
    source: ContactModePoseSource,
    budget: ContactModePoseRescueBudget,
    *,
    workers: int,
) -> tuple[dict[str, Any], Path]:
    stage = "contact_mode_pose_search"
    report_path = workspace / stage / "report.json"
    jobs = build_contact_mode_pose_rescue_jobs(source, budget=budget)
    stage_input = {
        "source_authentication_id": source.source_authentication_id,
        "budget_sha256": canonical_sha256(budget.as_mapping()),
        "jobs_sha256": canonical_sha256(jobs),
    }
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        _assert_stage_input(workspace, stage, stage_input)
        _validate_search_report(existing, expected_count=budget.candidate_count)
        return existing, report_path
    records = run_contact_mode_pose_rescue_jobs(
        jobs, workspace / stage / "candidates", workers=int(workers)
    )
    artifacts: list[Path] = []
    normalized: list[dict[str, Any]] = []
    for rank, raw in enumerate(records):
        record = copy.deepcopy(dict(raw))
        root = Path(str(record["artifact_directory"])).resolve()
        if not root.is_relative_to(workspace):
            raise RuntimeError("contact-mode candidate escaped its workspace")
        bundle = authenticate_v14_candidate_artifacts(
            root,
            expected_candidate_id=int(record["candidate_id"]),
            require_retained_trace=True,
            expected_retain_grasp_success=False,
        )
        artifacts.extend(bundle.artifact_paths)
        record["artifact_directory"] = str(root.relative_to(workspace))
        record["contact_mode_rank"] = rank
        normalized.append(record)
    payload = {
        "contact_mode_pose_search_report_schema_version": 1,
        "complete": True,
        "source_authentication_id": source.source_authentication_id,
        "candidate_count": len(normalized),
        "grasp_success_count": sum(
            bool(value.get("grasp_success", False)) for value in normalized
        ),
        "full_success_count": sum(
            bool(value.get("full_success", False)) for value in normalized
        ),
        "new_full_success_count": sum(
            bool(value.get("full_success", False))
            and not bool(value.get("is_exact_parent_reproduction_baseline", False))
            for value in normalized
        ),
        "source_reproduction_candidate_count": 1,
        "new_unique_candidate_count": len(normalized) - 1,
        "physical_unique_trial_count": len(
            {str(value["physical_config_sha256"]) for value in normalized}
        ),
        "physical_duplicate_against_source_count": sum(
            str(value["physical_config_sha256"])
            == contact_mode_physical_config_sha256(source.config)
            for value in normalized
        ),
        "trace_retained_candidate_count": len(normalized),
        "records": normalized,
    }
    committed = _commit_report(
        workspace,
        stage,
        report_path,
        payload,
        stage_input=stage_input,
        artifacts=tuple(dict.fromkeys(artifacts)),
    )
    _validate_search_report(committed, expected_count=budget.candidate_count)
    return committed, report_path


def _materialize_publications(
    records: Sequence[Mapping[str, Any]], workspace: Path, *, target_success_count: int
) -> tuple[dict[str, Any], ...]:
    materialized: list[dict[str, Any]] = []
    for record in records[:PUBLISHED_CANDIDATE_COUNT]:
        candidate_id = int(record["candidate_id"])
        search_root = (workspace / str(record["artifact_directory"])).resolve()
        bundle = authenticate_v14_candidate_artifacts(
            search_root,
            expected_candidate_id=candidate_id,
            require_retained_trace=True,
            expected_retain_grasp_success=False,
        )
        config = json.loads(bundle.config_path.read_text(encoding="utf-8"))
        root = (
            workspace
            / "catalog_source_reruns"
            / f"contact_mode_target_{target_success_count}"
            / f"candidate_{candidate_id}"
        )
        rerun = _run_full_reset_candidate(
            config, root, candidate_id, final_rerun=True
        )
        final = authenticate_v14_candidate_artifacts(
            root,
            expected_config=config,
            expected_candidate_id=candidate_id,
            require_retained_trace=True,
            expected_retain_grasp_success=False,
        )
        if canonical_sha256(rerun.get("summary")) != canonical_sha256(
            record.get("summary")
        ):
            raise RuntimeError("contact-mode independent rerun changed its summary")
        assert final.trace_path is not None
        import numpy as np

        with np.load(final.trace_path, allow_pickle=False) as trace:
            diagnostics = contact_mode_trace_diagnostics(trace)
        if canonical_sha256(diagnostics) != canonical_sha256(
            record.get("contact_mode_diagnostics")
        ):
            raise RuntimeError(
                "contact-mode independent rerun changed its taxel diagnostics"
            )
        materialized.append(
            {
                **copy.deepcopy(dict(record)),
                **copy.deepcopy(final.result),
                "artifact_directory": str(root.relative_to(workspace)),
                "contact_mode_diagnostics": diagnostics,
            }
        )
    return tuple(materialized)


def _restrict_aliases(catalog_path: Path, hard_pass_ids: Sequence[int]) -> None:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries = payload.get("trajectories")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("contact-mode publisher produced an empty catalog")
    hard = {int(value) for value in hard_pass_ids}
    successes = [
        value for value in entries if int(value.get("candidate_id", -1)) in hard
    ]
    chosen = successes[0] if successes else entries[0]
    alias = "best_nominal" if successes else "best_attempt"
    payload["aliases"] = {alias: chosen["trajectory_id"]}
    payload["best_grasp_object_pairs"] = {
        alias: {
            key: chosen.get(key)
            for key in (
                "candidate_id",
                "edge_m",
                "object_config_id",
                "grasp_pose_id",
                "grasp_object_pair_id",
                "planner_id",
                "controller_id",
                "validation_label",
            )
        }
    }
    payload["selection_policy"] = (
        "hard_success_then_grasp_then_thumb_taxel_branch_then_jerk"
    )
    payload["contact_mode_pose_rescue"] = True
    write_json(catalog_path, payload)


def _catalog_stage(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    experiment_id: str,
    target_success_count: int,
    search_report_path: Path,
) -> dict[str, Any]:
    stage = f"contact_mode_pose_catalog_target_{target_success_count}"
    root = workspace / "catalogs" / f"target_{target_success_count}"
    report_path = root / "report.json"
    selected = tuple(
        copy.deepcopy(dict(value))
        for value in records[:PUBLISHED_CANDIDATE_COUNT]
    )
    stage_input = {
        "search_report_sha256": file_sha256(search_report_path),
        "target_success_count": int(target_success_count),
        "selected_records_sha256": canonical_sha256(selected),
    }
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        _assert_stage_input(workspace, stage, stage_input)
        return existing
    materialized = _materialize_publications(
        selected, workspace, target_success_count=target_success_count
    )
    catalogs = _publish_rescue_viewer_catalogs(
        materialized, workspace, root, experiment_id=experiment_id
    )
    hard_ids = tuple(
        int(value["candidate_id"])
        for value in materialized
        if bool(value.get("full_success", False))
        and not bool(value.get("is_exact_parent_reproduction_baseline", False))
    )
    artifacts: list[Path] = []
    for relative in catalogs.values():
        catalog_path = workspace / relative
        _restrict_aliases(catalog_path, hard_ids)
        artifacts.extend(authenticated_catalog_artifact_paths(catalog_path))
    full_count = sum(
        bool(value.get("full_success", False))
        and not bool(value.get("is_exact_parent_reproduction_baseline", False))
        for value in records
    )
    payload = {
        "contact_mode_pose_catalog_report_schema_version": 1,
        "complete": True,
        "target_success_count": int(target_success_count),
        "target_reached": full_count >= int(target_success_count),
        "full_success_count": full_count,
        "published_candidate_count": len(materialized),
        "published_candidate_ids": [
            int(value["candidate_id"]) for value in materialized
        ],
        "catalogs": catalogs,
        "records": [],
    }
    return _commit_report(
        workspace,
        stage,
        report_path,
        payload,
        stage_input=stage_input,
        artifacts=tuple(dict.fromkeys(artifacts)),
    )


def run_contact_mode_pose_rescue_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    source_candidate_directory: str | Path,
    resume: bool,
    target_success_count: int,
    workers: int,
    seed: int = DEFAULT_SEED,
    candidate_count: int = MAXIMUM_CANDIDATE_COUNT,
) -> dict[str, Any]:
    """Run/resume one fixed-budget, trace-ranked contact-mode search."""

    if isinstance(workers, bool) or int(workers) <= 0:
        raise ValueError("workers must be positive")
    if int(seed) != DEFAULT_SEED:
        raise ValueError("contact-mode pose rescue seed must be 20260821")
    if int(target_success_count) not in (1, 5):
        raise ValueError("target_success_count must be one or five")
    config_file = Path(config_path).expanduser().resolve()
    workspace = Path(output_dir).expanduser().resolve()
    source_root = Path(source_candidate_directory).expanduser().resolve()
    if workspace == source_root or workspace.is_relative_to(source_root):
        raise ValueError("contact-mode rescue workspace must be outside its source")
    definition = resolve_experiment(load_config(config_file))
    if getattr(definition, "contact_preserving_planned_lift_campaign", None) is None:
        raise ValueError("contact-mode rescue requires the registered v14 experiment")
    source = authenticate_contact_mode_pose_source(source_root)
    budget = ContactModePoseRescueBudget(
        candidate_count=int(candidate_count), seed=int(seed)
    )
    manifest = _manifest(config_file, source, budget)
    initialize_or_resume_campaign(workspace, manifest, resume=bool(resume))

    audit_path = workspace / "contact_mode_pose_source_audit.json"
    audit_payload = {
        "contact_mode_pose_source_audit_schema_version": 1,
        "complete": True,
        "source": source.as_mapping(),
        "source_evidence_sha256": _source_evidence(source),
        "records": [],
    }
    audit_input = {
        "source_authentication_id": source.source_authentication_id,
        "source_evidence_sha256": _source_evidence(source),
    }
    audit = _load_committed_report(
        workspace, "contact_mode_pose_source_audit", audit_path
    )
    if audit is None:
        _commit_report(
            workspace,
            "contact_mode_pose_source_audit",
            audit_path,
            audit_payload,
            stage_input=audit_input,
        )
    else:
        _assert_stage_input(workspace, "contact_mode_pose_source_audit", audit_input)
        if canonical_sha256(audit) != canonical_sha256(audit_payload):
            raise RuntimeError("contact-mode source audit changed on resume")

    search, search_path = _run_search_stage(
        workspace, source, budget, workers=int(workers)
    )
    ranked = rank_contact_mode_pose_records(search["records"])
    # Ranks are persisted and must already represent the pure total order.
    if [int(value["candidate_id"]) for value in ranked] != [
        int(value["candidate_id"]) for value in search["records"]
    ]:
        raise RuntimeError("contact-mode committed search order changed")
    catalog = _catalog_stage(
        ranked,
        workspace,
        experiment_id=definition.experiment_id,
        target_success_count=int(target_success_count),
        search_report_path=search_path,
    )
    full_count = int(search["new_full_success_count"])
    result = {
        "contact_mode_pose_rescue_result_schema_version": (
            CONTACT_MODE_POSE_RESULT_SCHEMA_VERSION
        ),
        "complete": True,
        "experiment_id": definition.experiment_id,
        "source_candidate_directory": str(source_root),
        "source_candidate_id": source.candidate_id,
        "source_authentication_id": source.source_authentication_id,
        "candidate_count": int(search["candidate_count"]),
        "source_reproduction_candidate_count": int(
            search["source_reproduction_candidate_count"]
        ),
        "new_unique_candidate_count": int(search["new_unique_candidate_count"]),
        "physical_unique_trial_count": int(search["physical_unique_trial_count"]),
        "physical_duplicate_against_source_count": int(
            search["physical_duplicate_against_source_count"]
        ),
        "grasp_success_count": int(search["grasp_success_count"]),
        "full_success_count": full_count,
        "target_success_count": int(target_success_count),
        "target_reached": full_count >= int(target_success_count),
        "published_candidate_ids": copy.deepcopy(
            list(catalog["published_candidate_ids"])
        ),
        "catalogs": copy.deepcopy(dict(catalog["catalogs"])),
        "fixed_mass_geometry_ablation": True,
        "robustness": None,
        "exit_code": 0 if full_count else 2,
        "stop_reason": (
            "target_full_success_count_reached"
            if full_count >= int(target_success_count)
            else "declared_contact_mode_pose_budget_exhausted"
        ),
    }
    final_source = authenticate_contact_mode_pose_source(source_root)
    if final_source.source_authentication_id != source.source_authentication_id:
        raise RuntimeError("immutable contact-mode source changed during execution")
    if canonical_sha256(_manifest(config_file, final_source, budget)) != canonical_sha256(
        manifest
    ):
        raise RuntimeError("contact-mode implementation or inputs changed")
    validate_stage_ledger(workspace)

    result_path = workspace / f"contact_mode_pose_rescue_result_target_{target_success_count}.json"
    stage = f"contact_mode_pose_result_target_{target_success_count}"
    stage_input = {
        "search_report_sha256": file_sha256(search_path),
        "catalog_report_sha256": file_sha256(
            workspace / "catalogs" / f"target_{target_success_count}" / "report.json"
        ),
    }
    existing = _load_committed_report(workspace, stage, result_path)
    if existing is None:
        result = _commit_report(
            workspace,
            stage,
            result_path,
            result,
            stage_input=stage_input,
        )
    else:
        _assert_stage_input(workspace, stage, stage_input)
        if canonical_sha256(existing) != canonical_sha256(result):
            raise RuntimeError("committed contact-mode result changed on resume")
        result = existing
    return result


__all__ = [
    "DEFAULT_SEED",
    "MAXIMUM_CANDIDATE_COUNT",
    "build_contact_mode_pose_rescue_manifest",
    "run_contact_mode_pose_rescue_campaign",
]
