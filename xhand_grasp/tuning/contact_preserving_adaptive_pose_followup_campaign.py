"""Atomic two-stage campaign for the v14 adaptive grasp-pose follow-up.

The immutable source is a *completed* 256-candidate contact-mode pose search.
Exactly 64 one-axis finite-difference probes run first.  Sparse 2--4D
combinations are generated and run only when no probe is already a hard pass.
Every candidate uses the existing full-reset atomic runner and retains its raw
trace.  The best five are independently rerun before Viewer catalogs and MP4s
are published.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..actual_contact_grasp_pose_catalog import (
    authenticated_catalog_artifact_paths,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from ..artifacts import file_sha256, write_json
from ..config import load_config
from ..experiment import resolve_experiment
from ..grasp_pose import canonical_sha256
from .contact_preserving_adaptive_pose_followup import (
    ADAPTIVE_POSE_FOLLOWUP_SCHEMA_VERSION,
    DEFAULT_SEED,
    EXPECTED_FIRST_STAGE_CANDIDATE_COUNT,
    MAXIMUM_FOLLOWUP_CANDIDATE_COUNT,
    MAXIMUM_PROBE_CANDIDATE_COUNT,
    AdaptiveContactModePoseSource,
    AdaptivePoseFollowupBudget,
    build_adaptive_pose_combination_jobs,
    build_adaptive_pose_probe_jobs,
    build_adaptive_pose_source,
)
from .contact_preserving_candidate_artifacts import (
    V14CandidateArtifactBundle,
    authenticate_v14_candidate_artifacts,
)
from .contact_preserving_contact_mode_pose_rescue import (
    contact_mode_physical_config_sha256,
    contact_mode_trace_diagnostics,
    rank_contact_mode_pose_records,
    run_contact_mode_pose_rescue_jobs,
)
from .contact_preserving_contact_mode_pose_rescue_campaign import (
    _materialize_publications,
)
from .contact_preserving_lift_rescue_campaign import (
    _commit_report,
    _load_committed_report,
    _publish_rescue_viewer_catalogs,
)
from .contact_preserving_planned_lift_campaign import (
    build_contact_preserving_planned_lift_manifest,
)


ADAPTIVE_POSE_CAMPAIGN_SCHEMA_VERSION = 1
ADAPTIVE_POSE_RESULT_SCHEMA_VERSION = 1
PUBLISHED_CANDIDATE_COUNT = 5


@dataclass(frozen=True, slots=True)
class AuthenticatedAdaptivePoseCampaignSource:
    root: Path
    report_path: Path
    numerical_source: AdaptiveContactModePoseSource
    artifact_paths: tuple[Path, ...]
    source_authentication_id: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_POSE_CAMPAIGN_SCHEMA_VERSION,
            "root": str(self.root),
            "report_path": str(self.report_path),
            "report_sha256": file_sha256(self.report_path),
            "numerical_source": self.numerical_source.as_mapping(),
            "artifact_count": len(self.artifact_paths),
            "source_authentication_id": self.source_authentication_id,
        }


def _resolve_source_report(root: Path) -> Path:
    formal = root / "contact_mode_pose_search" / "report.json"
    sealed_direct = root / "report.json"
    if formal.is_file():
        validate_stage_ledger(root)
        return formal
    if sealed_direct.is_file():
        return sealed_direct
    raise FileNotFoundError(
        f"completed contact-mode pose report not found under {root}"
    )


def _candidate_root(source_root: Path, record: Mapping[str, Any]) -> Path:
    raw = Path(str(record.get("artifact_directory", ""))).expanduser()
    candidate = raw.resolve() if raw.is_absolute() else (source_root / raw).resolve()
    if not candidate.is_relative_to(source_root):
        raise RuntimeError("contact-mode source candidate escaped its campaign")
    return candidate


def authenticate_adaptive_pose_campaign_source(
    source_campaign: str | Path,
) -> AuthenticatedAdaptivePoseCampaignSource:
    """Authenticate all 256 source bundles and select strict jerk-only centres."""

    root = Path(source_campaign).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    report_path = _resolve_source_report(root)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise RuntimeError("contact-mode source report is not a JSON object")
    records = report.get("records")
    if (
        report.get("complete") is not True
        or int(report.get("candidate_count", -1))
        != EXPECTED_FIRST_STAGE_CANDIDATE_COUNT
        or not isinstance(records, list)
        or len(records) != EXPECTED_FIRST_STAGE_CANDIDATE_COUNT
        or int(report.get("full_success_count", -1)) != 0
        or any(bool(value.get("full_success", False)) for value in records)
    ):
        raise RuntimeError(
            "adaptive pose follow-up requires a complete zero-success 256-candidate source"
        )
    identifiers = [int(value.get("candidate_id", -1)) for value in records]
    if min(identifiers, default=-1) < 0 or len(identifiers) != len(set(identifiers)):
        raise RuntimeError("contact-mode source report contains duplicate candidates")

    configs: dict[int, dict[str, Any]] = {}
    traces: dict[int, str] = {}
    bundles: dict[int, V14CandidateArtifactBundle] = {}
    artifact_paths: list[Path] = [report_path]
    for name in ("campaign_manifest.json", "stage_ledger.json"):
        candidate = root / name
        if candidate.is_file():
            artifact_paths.append(candidate)
    for record in records:
        candidate_id = int(record["candidate_id"])
        candidate_root = _candidate_root(root, record)
        bundle = authenticate_v14_candidate_artifacts(
            candidate_root,
            expected_candidate_id=candidate_id,
            require_retained_trace=True,
        )
        if bundle.trace_path is None:
            raise RuntimeError("contact-mode source lost a retained trace")
        config = json.loads(bundle.config_path.read_text(encoding="utf-8"))
        if canonical_sha256(config) != str(record.get("config_semantic_sha256", "")):
            raise RuntimeError("contact-mode source report/config hash changed")
        if canonical_sha256(bundle.result.get("summary")) != canonical_sha256(
            record.get("summary")
        ):
            raise RuntimeError("contact-mode source report/result summary changed")
        configs[candidate_id] = config
        traces[candidate_id] = file_sha256(bundle.trace_path)
        bundles[candidate_id] = bundle
        artifact_paths.extend(bundle.artifact_paths)
    numerical = build_adaptive_pose_source(
        records,
        configs,
        traces,
        search_report_sha256=file_sha256(report_path),
        expected_candidate_count=EXPECTED_FIRST_STAGE_CANDIDATE_COUNT,
    )
    records_by_id = {int(value["candidate_id"]): value for value in records}
    for center in numerical.centers:
        bundle = bundles[center.candidate_id]
        assert bundle.trace_path is not None
        with np.load(bundle.trace_path, allow_pickle=False) as trace:
            diagnostics = contact_mode_trace_diagnostics(trace)
        if canonical_sha256(diagnostics) != canonical_sha256(
            records_by_id[center.candidate_id].get("contact_mode_diagnostics")
        ):
            raise RuntimeError("contact-mode top-centre trace diagnostics changed")
    unique_artifacts = tuple(dict.fromkeys(path.resolve() for path in artifact_paths))
    identity = canonical_sha256(
        {
            "schema_version": ADAPTIVE_POSE_CAMPAIGN_SCHEMA_VERSION,
            "root": str(root),
            "report_path": str(report_path),
            "numerical_source": numerical.as_mapping(),
            "artifact_count": len(unique_artifacts),
        }
    )
    return AuthenticatedAdaptivePoseCampaignSource(
        root=root,
        report_path=report_path,
        numerical_source=numerical,
        artifact_paths=unique_artifacts,
        source_authentication_id=identity,
    )


def _manifest(
    config_path: Path,
    source: AuthenticatedAdaptivePoseCampaignSource,
    budget: AdaptivePoseFollowupBudget,
) -> dict[str, Any]:
    base = build_contact_preserving_planned_lift_manifest(
        config_path, seed=int(budget.seed)
    )
    base.pop("campaign_input_sha256", None)
    implementation_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("contact_preserving_adaptive_pose_followup.py"),
    )
    base.update(
        {
            "adaptive_pose_campaign_schema_version": ADAPTIVE_POSE_CAMPAIGN_SCHEMA_VERSION,
            "campaign_kind": "contact_preserving_adaptive_pose_followup",
            "source_campaign": str(source.root),
            "source_authentication": source.as_mapping(),
            "budget": budget.as_mapping(),
            "implementation_sha256": {
                path.name: file_sha256(path) for path in implementation_paths
            },
            "execution_contract": {
                "probe": "64_one_axis_fresh_full_reset_retained_trace",
                "sparse_gate": "only_when_probe_full_success_count_is_zero",
                "sparse": "64_safe_direction_2_to_4d_fresh_full_reset",
                "publication": "top5_independent_full_reset_mp4",
                "cube_pose_changed": False,
                "manipulation_plan_changed": False,
                "acceptance_thresholds_changed": False,
            },
        }
    )
    return {**base, "campaign_input_sha256": canonical_sha256(base)}


def build_adaptive_pose_followup_manifest(
    config_path: str | Path,
    source_campaign: str | Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    source = authenticate_adaptive_pose_campaign_source(source_campaign)
    budget = AdaptivePoseFollowupBudget(
        candidate_count=MAXIMUM_FOLLOWUP_CANDIDATE_COUNT, seed=int(seed)
    )
    return _manifest(Path(config_path).expanduser().resolve(), source, budget)


def _assert_stage_input(
    workspace: Path, stage: str, stage_input: Mapping[str, Any]
) -> None:
    record = validate_stage_ledger(workspace).get("stages", {}).get(stage)
    if not isinstance(record, Mapping):
        raise RuntimeError(f"committed adaptive pose stage {stage} disappeared")
    if record.get("stage_input_sha256") != canonical_sha256(dict(stage_input)):
        raise RuntimeError(f"committed adaptive pose stage {stage} input changed")


def _validate_stage_report(
    report: Mapping[str, Any], *, expected_count: int, label: str
) -> None:
    records = report.get("records")
    if (
        report.get("complete") is not True
        or int(report.get("candidate_count", -1)) != int(expected_count)
        or not isinstance(records, list)
        or len(records) != int(expected_count)
    ):
        raise RuntimeError(f"{label} did not exhaust its registered budget")
    identifiers = [int(value.get("candidate_id", -1)) for value in records]
    physical = [str(value.get("physical_config_sha256", "")) for value in records]
    if (
        min(identifiers, default=-1) < 0
        or len(identifiers) != len(set(identifiers))
        or any(len(value) != 64 for value in physical)
        or len(physical) != len(set(physical))
    ):
        raise RuntimeError(f"{label} lost candidate or physical uniqueness")
    observed_full = sum(bool(value.get("full_success", False)) for value in records)
    if observed_full != int(report.get("full_success_count", -1)):
        raise RuntimeError(f"{label} full-success count changed")
    if int(report.get("physical_unique_candidate_count", len(physical))) != len(
        physical
    ):
        raise RuntimeError(f"{label} physical-unique count changed")


def _sparse_stage_required(probe_report: Mapping[str, Any]) -> bool:
    """Authenticate the full probe budget before applying the hard-pass gate."""

    _validate_stage_report(
        probe_report,
        expected_count=MAXIMUM_PROBE_CANDIDATE_COUNT,
        label="adaptive_pose_probe",
    )
    observed = sum(
        bool(value.get("full_success", False))
        for value in probe_report["records"]
    )
    if observed != int(probe_report.get("full_success_count", -1)):
        raise RuntimeError("adaptive probe hard-pass count changed")
    return observed == 0


def _run_job_stage(
    workspace: Path,
    stage: str,
    jobs: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    stage_input_extra: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    report_path = workspace / stage / "report.json"
    stage_input = {
        **copy.deepcopy(dict(stage_input_extra)),
        "candidate_count": len(jobs),
        "jobs_sha256": canonical_sha256(jobs),
    }
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        _assert_stage_input(workspace, stage, stage_input)
        _validate_stage_report(existing, expected_count=len(jobs), label=stage)
        return existing, report_path
    job_by_id = {int(value["candidate_id"]): value for value in jobs}
    if len(job_by_id) != len(jobs):
        raise RuntimeError(f"{stage} jobs contain duplicate candidate IDs")
    raw_records = run_contact_mode_pose_rescue_jobs(
        jobs, workspace / stage / "candidates", workers=int(workers)
    )
    artifacts: list[Path] = []
    normalized: list[dict[str, Any]] = []
    for stage_rank, raw in enumerate(raw_records):
        record = copy.deepcopy(dict(raw))
        candidate_id = int(record["candidate_id"])
        job = job_by_id[candidate_id]
        root = Path(str(record["artifact_directory"])).resolve()
        if not root.is_relative_to(workspace):
            raise RuntimeError(f"{stage} candidate escaped its workspace")
        bundle = authenticate_v14_candidate_artifacts(
            root,
            expected_config=job["config"],
            expected_candidate_id=candidate_id,
            require_retained_trace=True,
            expected_retain_grasp_success=False,
        )
        if str(record.get("physical_config_sha256")) != str(
            job["physical_config_sha256"]
        ):
            raise RuntimeError(f"{stage} candidate physical identity changed")
        artifacts.extend(bundle.artifact_paths)
        record["artifact_directory"] = str(root.relative_to(workspace))
        record["adaptive_stage"] = str(
            job["job_metadata"]["adaptive_stage"]
        )
        record["adaptive_stage_rank"] = stage_rank
        normalized.append(record)
    payload = {
        "adaptive_pose_stage_report_schema_version": 1,
        "complete": True,
        "stage": stage,
        "candidate_count": len(normalized),
        "grasp_success_count": sum(
            bool(value.get("grasp_success", False)) for value in normalized
        ),
        "full_success_count": sum(
            bool(value.get("full_success", False)) for value in normalized
        ),
        "physical_unique_candidate_count": len(
            {str(value["physical_config_sha256"]) for value in normalized}
        ),
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
    _validate_stage_report(committed, expected_count=len(jobs), label=stage)
    return committed, report_path


def _rewrite_catalog_metadata(catalog_path: Path) -> None:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    payload["selection_policy"] = (
        "hard_success_then_contact_mode_stability_then_filtered_jerk"
    )
    payload["adaptive_pose_followup"] = True
    payload["adaptive_two_stage_policy"] = (
        "64_one_axis_then_if_zero_success_64_safe_sparse"
    )
    write_json(catalog_path, payload)


def _catalog_stage(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    experiment_id: str,
    target_success_count: int,
    source_report_hashes: Mapping[str, str],
) -> dict[str, Any]:
    stage = f"adaptive_pose_catalog_target_{target_success_count}"
    root = workspace / "catalogs" / f"target_{target_success_count}"
    report_path = root / "report.json"
    selected = tuple(copy.deepcopy(dict(value)) for value in records[:PUBLISHED_CANDIDATE_COUNT])
    stage_input = {
        "source_report_hashes": copy.deepcopy(dict(source_report_hashes)),
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
    artifacts: list[Path] = []
    for relative in catalogs.values():
        catalog_path = workspace / relative
        _rewrite_catalog_metadata(catalog_path)
        artifacts.extend(authenticated_catalog_artifact_paths(catalog_path))
    full_count = sum(bool(value.get("full_success", False)) for value in records)
    payload = {
        "adaptive_pose_catalog_report_schema_version": 1,
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


def run_contact_preserving_adaptive_pose_followup_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    source_contact_mode_campaign: str | Path,
    resume: bool,
    target_success_count: int,
    workers: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run/resume the conditional 64 + 64 adaptive pose follow-up."""

    if isinstance(workers, bool) or int(workers) <= 0:
        raise ValueError("workers must be positive")
    if int(seed) != DEFAULT_SEED:
        raise ValueError("adaptive pose follow-up seed must be 20260821")
    if int(target_success_count) not in (1, 5):
        raise ValueError("target_success_count must be one or five")
    config_file = Path(config_path).expanduser().resolve()
    workspace = Path(output_dir).expanduser().resolve()
    source_root = Path(source_contact_mode_campaign).expanduser().resolve()
    if workspace == source_root or workspace.is_relative_to(source_root):
        raise ValueError("adaptive pose workspace must be outside its source")
    definition = resolve_experiment(load_config(config_file))
    if getattr(definition, "contact_preserving_planned_lift_campaign", None) is None:
        raise ValueError("adaptive pose runner requires the registered v14 experiment")
    source = authenticate_adaptive_pose_campaign_source(source_root)
    budget = AdaptivePoseFollowupBudget(
        candidate_count=MAXIMUM_FOLLOWUP_CANDIDATE_COUNT, seed=int(seed)
    )
    manifest = _manifest(config_file, source, budget)
    initialize_or_resume_campaign(workspace, manifest, resume=bool(resume))

    audit_path = workspace / "adaptive_pose_source_audit.json"
    audit_payload = {
        "adaptive_pose_source_audit_schema_version": 1,
        "complete": True,
        "source": source.as_mapping(),
        "records": [],
    }
    audit_input = {
        "source_authentication_id": source.source_authentication_id,
        "source_report_sha256": file_sha256(source.report_path),
    }
    audit = _load_committed_report(workspace, "adaptive_pose_source_audit", audit_path)
    if audit is None:
        _commit_report(
            workspace,
            "adaptive_pose_source_audit",
            audit_path,
            audit_payload,
            stage_input=audit_input,
        )
    else:
        _assert_stage_input(workspace, "adaptive_pose_source_audit", audit_input)
        if canonical_sha256(audit) != canonical_sha256(audit_payload):
            raise RuntimeError("adaptive pose source audit changed on resume")

    probe_jobs = build_adaptive_pose_probe_jobs(
        source.numerical_source, budget=budget
    )
    probe, probe_path = _run_job_stage(
        workspace,
        "adaptive_pose_probe",
        probe_jobs,
        workers=int(workers),
        stage_input_extra={
            "source_authentication_id": source.source_authentication_id,
            "budget_sha256": canonical_sha256(budget.as_mapping()),
        },
    )
    sparse: dict[str, Any] | None = None
    sparse_path: Path | None = None
    if _sparse_stage_required(probe):
        sparse_jobs = build_adaptive_pose_combination_jobs(
            source.numerical_source,
            probe_jobs,
            probe["records"],
            budget=budget,
        )
        if len(sparse_jobs) != MAXIMUM_FOLLOWUP_CANDIDATE_COUNT - MAXIMUM_PROBE_CANDIDATE_COUNT:
            raise RuntimeError("adaptive sparse generator returned the wrong budget")
        sparse, sparse_path = _run_job_stage(
            workspace,
            "adaptive_pose_sparse",
            sparse_jobs,
            workers=int(workers),
            stage_input_extra={
                "source_authentication_id": source.source_authentication_id,
                "probe_report_sha256": file_sha256(probe_path),
                "budget_sha256": canonical_sha256(budget.as_mapping()),
            },
        )
    combined = [copy.deepcopy(dict(value)) for value in probe["records"]]
    if sparse is not None:
        combined.extend(copy.deepcopy(dict(value)) for value in sparse["records"])
    combined_physical = [str(value["physical_config_sha256"]) for value in combined]
    if len(combined_physical) != len(set(combined_physical)):
        raise RuntimeError("adaptive probe and sparse stages reused a physical candidate")
    ranked = list(rank_contact_mode_pose_records(combined))
    for rank, record in enumerate(ranked):
        record["plan_rank"] = rank
    report_hashes = {"probe": file_sha256(probe_path)}
    if sparse_path is not None:
        report_hashes["sparse"] = file_sha256(sparse_path)
    catalog = _catalog_stage(
        ranked,
        workspace,
        experiment_id=definition.experiment_id,
        target_success_count=int(target_success_count),
        source_report_hashes=report_hashes,
    )
    full_count = sum(bool(value.get("full_success", False)) for value in ranked)
    result = {
        "adaptive_pose_followup_result_schema_version": ADAPTIVE_POSE_RESULT_SCHEMA_VERSION,
        "complete": True,
        "experiment_id": definition.experiment_id,
        "source_contact_mode_campaign": str(source_root),
        "source_authentication_id": source.source_authentication_id,
        "probe_candidate_count": int(probe["candidate_count"]),
        "probe_full_success_count": int(probe["full_success_count"]),
        "sparse_stage_executed": sparse is not None,
        "sparse_candidate_count": 0 if sparse is None else int(sparse["candidate_count"]),
        "sparse_full_success_count": 0 if sparse is None else int(sparse["full_success_count"]),
        "physical_unique_candidate_count": len(
            {str(value["physical_config_sha256"]) for value in ranked}
        ),
        "full_success_count": full_count,
        "target_success_count": int(target_success_count),
        "target_reached": full_count >= int(target_success_count),
        "published_candidate_ids": copy.deepcopy(catalog["published_candidate_ids"]),
        "catalogs": copy.deepcopy(dict(catalog["catalogs"])),
        "fixed_mass_geometry_ablation": True,
        "robustness": None,
        "exit_code": 0 if full_count else 2,
        "stop_reason": (
            "probe_hard_pass_sparse_stage_skipped"
            if int(probe["full_success_count"]) > 0
            else (
                "target_full_success_count_reached"
                if full_count >= int(target_success_count)
                else "declared_adaptive_pose_budget_exhausted"
            )
        ),
    }
    final_source = authenticate_adaptive_pose_campaign_source(source_root)
    if final_source.source_authentication_id != source.source_authentication_id:
        raise RuntimeError("immutable adaptive pose source changed during execution")
    if canonical_sha256(_manifest(config_file, final_source, budget)) != canonical_sha256(manifest):
        raise RuntimeError("adaptive pose implementation or inputs changed")
    validate_stage_ledger(workspace)

    result_path = workspace / f"adaptive_pose_followup_result_target_{target_success_count}.json"
    stage = f"adaptive_pose_followup_result_target_{target_success_count}"
    stage_input = {
        "source_report_hashes": report_hashes,
        "catalog_report_sha256": file_sha256(
            workspace / "catalogs" / f"target_{target_success_count}" / "report.json"
        ),
    }
    existing = _load_committed_report(workspace, stage, result_path)
    if existing is None:
        result = _commit_report(
            workspace, stage, result_path, result, stage_input=stage_input
        )
    else:
        _assert_stage_input(workspace, stage, stage_input)
        if canonical_sha256(existing) != canonical_sha256(result):
            raise RuntimeError("committed adaptive pose result changed on resume")
        result = existing
    return result


__all__ = [
    "ADAPTIVE_POSE_CAMPAIGN_SCHEMA_VERSION",
    "AuthenticatedAdaptivePoseCampaignSource",
    "build_adaptive_pose_followup_manifest",
    "authenticate_adaptive_pose_campaign_source",
    "run_contact_preserving_adaptive_pose_followup_campaign",
]
