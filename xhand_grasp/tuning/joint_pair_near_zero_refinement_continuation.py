"""Authenticated refinement continuation for the schema-v15 campaign.

The expensive static, grasp, sequential-probe and feedback-grid stages are
immutable search evidence once their stage ledger has been committed.  This
module permits a corrected refinement policy to consume that evidence without
editing the source workspace or pretending that a changed implementation is a
normal in-place resume.

The continuation owns a new manifest, ledger and candidate namespace.  Its
manifest binds the complete upstream manifest/ledger/feedback report plus the
current implementation hash.  Every published candidate is still rerun from
the initial no-contact state by the production physics backend.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..actual_contact_grasp_pose_catalog import (
    commit_campaign_stage,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from ..artifacts import (
    REPO_ROOT,
    aggregate_source_sha256,
    file_sha256,
    implementation_paths,
    write_json,
)
from ..config import load_config
from ..grasp_pose import canonical_sha256
from .joint_pair_near_zero_campaign import (
    EXPERIMENT_ID,
    JointPairNearZeroBudget,
    SEED,
    rank_near_zero_candidates,
)
from .joint_pair_near_zero_campaign_runner import (
    V15CampaignBackend,
    V15CampaignJob,
    _record_id,
    _run_stage,
    build_exact_rerun_jobs,
    build_feedback_jobs,
    build_perturbation_jobs,
    build_refinement_jobs,
    build_robustness_jobs,
    build_v15_campaign_manifest,
    publish_v15_viewer_catalog,
)
from .joint_pair_near_zero_candidate_artifacts import (
    authenticate_v15_candidate_artifacts,
)
from .joint_pair_near_zero_physics_backend import (
    create_joint_pair_near_zero_campaign_backend,
)


CONTINUATION_SCHEMA_VERSION = 1
SOURCE_STAGE = "feedback_grid"


@dataclass(frozen=True, slots=True)
class AuthenticatedFeedbackSource:
    workspace: Path
    manifest_path: Path
    ledger_path: Path
    report_path: Path
    manifest: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    binding_payload: dict[str, Any]

    @property
    def binding(self) -> dict[str, Any]:
        # Authentication freezes these hashes as one snapshot.  Recomputing
        # them lazily would permit the source ledger to change between bundle
        # verification and continuation-manifest construction.
        return copy.deepcopy(self.binding_payload)


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON mapping: {path}")
    return value


def _committed_stage_report(
    root: Path,
    ledger: Mapping[str, Any],
    stage: str,
) -> tuple[dict[str, Any], Mapping[str, Any], Path]:
    stages = ledger.get("stages")
    committed = stages.get(stage) if isinstance(stages, Mapping) else None
    if not isinstance(committed, Mapping) or committed.get("complete") is not True:
        raise RuntimeError(f"{stage} is not committed in the source ledger")
    report_path = root / "stages" / stage / "report.json"
    if not report_path.is_file():
        raise RuntimeError(f"committed {stage} report is missing")
    relative = str(report_path.relative_to(root))
    artifacts = committed.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or artifacts.get(relative) != file_sha256(report_path)
    ):
        raise RuntimeError(f"{stage} report differs from its source ledger")
    report = _load_mapping(report_path)
    records = report.get("records")
    if (
        report.get("joint_pair_near_zero_stage_report_schema_version") != 1
        or report.get("complete") is not True
        or report.get("stage") != stage
        or not isinstance(records, list)
        or int(report.get("job_count", -1)) != len(records)
    ):
        raise RuntimeError(f"committed {stage} report is incomplete or malformed")
    return report, committed, report_path


def authenticate_feedback_source(
    workspace: str | Path,
    *,
    config_path: str | Path | None = None,
    repository_root: str | Path = REPO_ROOT,
    budget: JointPairNearZeroBudget = JointPairNearZeroBudget(),
) -> AuthenticatedFeedbackSource:
    """Authenticate a complete v15 feedback stage and every candidate bundle."""

    root = Path(workspace).expanduser().resolve()
    manifest_path = root / "campaign_manifest.json"
    ledger_path = root / "stage_ledger.json"
    if not (manifest_path.is_file() and ledger_path.is_file()):
        raise RuntimeError("continuation source lacks a committed feedback stage")
    manifest_file_sha256 = file_sha256(manifest_path)
    ledger_file_sha256 = file_sha256(ledger_path)
    manifest = _load_mapping(manifest_path)
    manifest_bound = copy.deepcopy(manifest)
    observed_campaign_input = manifest_bound.pop("campaign_input_sha256", None)
    if (
        manifest.get("joint_pair_near_zero_campaign_manifest_schema_version") != 1
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or observed_campaign_input != canonical_sha256(manifest_bound)
    ):
        raise RuntimeError("continuation source has a malformed campaign manifest")

    registered_config_path = (
        Path(str(manifest.get("config_path"))).expanduser().resolve()
        if config_path is None
        else Path(config_path).expanduser().resolve()
    )
    current_manifest = build_v15_campaign_manifest(
        registered_config_path,
        repository_root=repository_root,
        budget=budget,
    )
    # A corrected continuation implementation is intentionally allowed to
    # have a different source_sha256.  Physical inputs, source evidence and
    # the declared budget are not migration points.
    immutable_fields = (
        "experiment_id",
        "seed",
        "config_sha256",
        "model_sha256",
        "uv_lock_sha256",
        "actual_qpos_source_manifest_sha256",
        "source_id",
        "source_artifact_sha256",
        "budget",
    )
    mismatches = [
        field
        for field in immutable_fields
        if manifest.get(field) != current_manifest.get(field)
    ]
    if mismatches:
        raise RuntimeError(
            "feedback source physical/campaign inputs differ from the current "
            "continuation: " + ", ".join(mismatches)
        )

    ledger = validate_stage_ledger(root)
    expected_stage_counts = {
        "static_filter": budget.static_start_count,
        "dynamic_grasp": budget.dynamic_grasp_count,
        "sequential_planning": budget.sequential_plan_count,
        SOURCE_STAGE: budget.feedback_grid_count,
    }
    stages = ledger.get("stages")
    if not isinstance(stages, Mapping) or "source_migration" not in stages:
        raise RuntimeError("feedback source lost its committed prerequisite stages")
    for stage, count in expected_stage_counts.items():
        record = stages.get(stage)
        if (
            not isinstance(record, Mapping)
            or record.get("complete") is not True
            or record.get("summary", {}).get("job_count") != count
        ):
            raise RuntimeError(
                f"feedback source has the wrong committed {stage} budget"
            )

    plan_report, _, _ = _committed_stage_report(
        root, ledger, "sequential_planning"
    )
    plan_records = plan_report["records"]
    if len(plan_records) != budget.sequential_plan_count:
        raise RuntimeError("feedback source has the wrong sequential-plan count")
    expected_jobs = build_feedback_jobs(plan_records, budget)
    expected_job_sha256 = canonical_sha256(
        [job.descriptor() for job in expected_jobs]
    )

    report, committed, report_path = _committed_stage_report(
        root, ledger, SOURCE_STAGE
    )
    report_file_sha256 = file_sha256(report_path)
    raw_records = report["records"]
    if (
        len(raw_records) != budget.feedback_grid_count
        or report.get("job_sha256") != expected_job_sha256
    ):
        raise RuntimeError("feedback report differs from the current canonical jobs")
    expected_ids = [job.candidate_id for job in expected_jobs]
    observed_order = [_record_id(record) for record in raw_records]
    if observed_order != expected_ids:
        raise RuntimeError("feedback report candidate order differs from canonical jobs")

    committed_artifacts = committed.get("artifacts")
    if not isinstance(committed_artifacts, Mapping):
        raise RuntimeError("feedback stage has no committed artifact mapping")
    records: list[dict[str, Any]] = []
    observed_ids: set[int] = set()
    candidate_evidence: list[dict[str, Any]] = []
    for raw, job in zip(raw_records, expected_jobs, strict=True):
        if not isinstance(raw, Mapping):
            raise RuntimeError("feedback stage contains a non-mapping record")
        record = copy.deepcopy(dict(raw))
        candidate_id = _record_id(record)
        if candidate_id in observed_ids:
            raise RuntimeError("feedback stage contains duplicate candidate IDs")
        observed_ids.add(candidate_id)
        config_path = Path(str(record.get("config_path"))).expanduser().resolve()
        result_path = Path(str(record.get("result_path"))).expanduser().resolve()
        candidate_root = (
            root
            / "physics"
            / SOURCE_STAGE
            / f"candidate_{candidate_id}"
        )
        if (
            config_path != candidate_root / "resolved_config.json"
            or result_path != candidate_root / "result.json"
            or record.get("stage") != SOURCE_STAGE
            or int(record.get("parent_candidate_id", -1))
            != int(job.parent_candidate_id or -1)
        ):
            raise RuntimeError("feedback record differs from its canonical job paths")
        bundle = authenticate_v15_candidate_artifacts(
            candidate_root,
            expected_candidate_id=candidate_id,
        )
        expected_trace = record.get("trace_path")
        observed_trace = str(bundle.trace_path) if bundle.trace_path is not None else None
        if expected_trace != observed_trace:
            raise RuntimeError("feedback record trace path changed")
        for path in bundle.artifact_paths:
            relative = str(path.relative_to(root))
            if committed_artifacts.get(relative) != file_sha256(path):
                raise RuntimeError("feedback bundle is not bound by its committed stage")
        for field in ("grasp_success", "full_success", "summary"):
            if canonical_sha256(record.get(field)) != canonical_sha256(
                bundle.result.get(field)
            ):
                raise RuntimeError(
                    f"feedback report changed authenticated candidate field {field}"
                )
        resolved = _load_mapping(bundle.config_path)
        metadata = resolved.get("candidate_metadata")
        campaign_job = (
            metadata.get("v15_campaign_job")
            if isinstance(metadata, Mapping)
            else None
        )
        if (
            not isinstance(campaign_job, Mapping)
            or int(metadata.get("candidate_id", -1)) != candidate_id
            or campaign_job.get("stage") != SOURCE_STAGE
            or int(campaign_job.get("index", -1)) != job.index
            or int(campaign_job.get("parent_candidate_id", -1))
            != int(job.parent_candidate_id or -1)
            or campaign_job.get("job_sha256")
            != canonical_sha256(job.descriptor())
        ):
            raise RuntimeError("feedback candidate lost its canonical job descriptor")
        evidence_paths = {}
        for path in bundle.artifact_paths:
            relative = str(path.relative_to(root))
            evidence_paths[path.name] = {
                "path": relative,
                "sha256": str(committed_artifacts[relative]),
            }
        candidate_evidence.append(
            {
                "candidate_id": candidate_id,
                "parent_candidate_id": job.parent_candidate_id,
                "job_sha256": canonical_sha256(job.descriptor()),
                "artifacts": evidence_paths,
            }
        )
        records.append(record)

    # Detect a writer racing this read-only authentication pass.
    if (
        file_sha256(manifest_path) != manifest_file_sha256
        or file_sha256(ledger_path) != ledger_file_sha256
        or file_sha256(report_path) != report_file_sha256
    ):
        raise RuntimeError("feedback source changed during authentication")
    binding_payload = {
        "workspace": str(root),
        "campaign_input_sha256": observed_campaign_input,
        "upstream_source_sha256": manifest.get("source_sha256"),
        "campaign_manifest_sha256": manifest_file_sha256,
        "stage_ledger_sha256": ledger_file_sha256,
        "feedback_stage_record_sha256": canonical_sha256(committed),
        "feedback_stage_input_sha256": committed.get("stage_input_sha256"),
        "feedback_report_sha256": report_file_sha256,
        "feedback_job_sha256": expected_job_sha256,
        "feedback_job_count": len(records),
        "feedback_record_sha256": canonical_sha256(records),
        "candidate_evidence_sha256": canonical_sha256(candidate_evidence),
        "fully_reauthenticated_against_current_jobs": True,
        "read_only": True,
    }
    binding_payload["feedback_source_id"] = canonical_sha256(binding_payload)
    return AuthenticatedFeedbackSource(
        root,
        manifest_path,
        ledger_path,
        report_path,
        manifest,
        tuple(records),
        binding_payload,
    )


def build_continuation_manifest(
    source: AuthenticatedFeedbackSource,
    *,
    config_path: str | Path,
    repository_root: str | Path = REPO_ROOT,
    budget: JointPairNearZeroBudget = JointPairNearZeroBudget(),
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    if (
        int(config.get("schema_version", 0)) != 15
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("continuation requires the registered schema-v15 config")
    bound = {
        "joint_pair_near_zero_refinement_continuation_schema_version": (
            CONTINUATION_SCHEMA_VERSION
        ),
        "experiment_id": EXPERIMENT_ID,
        "continuation_kind": "post_feedback_manipulation_viable_refinement",
        "seed": int(budget.seed),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "model_sha256": file_sha256(root / "xhand_left.xml"),
        "uv_lock_sha256": file_sha256(root / "uv.lock"),
        "actual_qpos_source_manifest_sha256": source.manifest.get(
            "actual_qpos_source_manifest_sha256"
        ),
        "source_sha256": aggregate_source_sha256(implementation_paths()),
        "upstream_feedback_source": source.binding,
        "budget": budget.as_mapping(),
    }
    return {**bound, "campaign_input_sha256": canonical_sha256(bound)}


def _expected_stage_input(
    jobs: Sequence[V15CampaignJob], context: Mapping[str, Any]
) -> tuple[str, dict[str, str], str]:
    job_sha256 = canonical_sha256([job.descriptor() for job in jobs])
    stage_input = {
        "job_sha256": job_sha256,
        "context_sha256": canonical_sha256(context),
    }
    return job_sha256, stage_input, canonical_sha256(stage_input)


def _load_committed_continuation_stage(
    workspace: Path,
    stage: str,
    jobs: Sequence[V15CampaignJob],
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    ledger = validate_stage_ledger(workspace)
    if stage not in ledger["stages"]:
        return None
    report, committed, _ = _committed_stage_report(workspace, ledger, stage)
    expected_job_sha, _, expected_stage_input_sha = _expected_stage_input(
        jobs, context
    )
    records = report["records"]
    if (
        committed.get("stage_input_sha256") != expected_stage_input_sha
        or report.get("job_sha256") != expected_job_sha
        or int(report.get("job_count", -1)) != len(jobs)
        or [_record_id(record) for record in records]
        != [job.candidate_id for job in jobs]
    ):
        raise RuntimeError(
            f"committed continuation stage {stage} differs from current jobs/input"
        )
    return tuple(copy.deepcopy(dict(record)) for record in records)


def _run_continuation_stage(
    workspace: Path,
    stage: str,
    jobs: Sequence[V15CampaignJob],
    backend: V15CampaignBackend,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Run one stage or strictly authenticate its committed descriptor."""

    committed = _load_committed_continuation_stage(
        workspace, stage, jobs, context
    )
    if committed is not None:
        return committed
    _run_stage(workspace, stage, jobs, backend, context)
    persisted = _load_committed_continuation_stage(
        workspace, stage, jobs, context
    )
    if persisted is None:  # Defensive: _run_stage must commit before return.
        raise RuntimeError(f"continuation stage {stage} returned without a commit")
    return persisted


def _remove_uncommitted_catalog_tree(path: Path, *, expected_parent: Path) -> None:
    resolved_parent = expected_parent.resolve()
    if path.is_symlink() or not path.is_dir() or path.resolve().parent != resolved_parent:
        raise RuntimeError(f"unsafe uncommitted catalog artifact: {path}")
    shutil.rmtree(path)


def _validate_catalog(
    catalog_root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    robust_candidate_id: int | None,
) -> Path:
    path = catalog_root / "catalog.json"
    payload = _load_mapping(path)
    ranked = list(rank_near_zero_candidates(records)[:5])
    expected_ids = [_record_id(record) for record in ranked]
    trajectories = payload.get("trajectories")
    if (
        payload.get("trajectory_catalog_schema_version") != 1
        or payload.get("joint_pair_near_zero_viewer_catalog_schema_version") != 1
        or payload.get("complete") is not True
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("catalog_kind") != "manipulation"
        or not isinstance(trajectories, list)
        or [int(entry.get("candidate_id", -1)) for entry in trajectories]
        != expected_ids
    ):
        raise RuntimeError("continuation catalog is incomplete or non-canonical")

    successes: list[int] = []
    for rank, (entry, record) in enumerate(zip(trajectories, ranked, strict=True), 1):
        candidate_id = _record_id(record)
        trajectory_id = f"candidate_{candidate_id}"
        full_success = bool(record.get("full_success"))
        if full_success:
            successes.append(candidate_id)
        if (
            entry.get("trajectory_id") != trajectory_id
            or int(entry.get("rank", -1)) != rank
            or bool(entry.get("grasp_success"))
            != bool(record.get("grasp_success"))
            or bool(entry.get("full_success")) != full_success
            or entry.get("classification")
            != ("success" if full_success else "diagnostic")
        ):
            raise RuntimeError("continuation catalog entry changed its classification")
        artifacts = entry.get("artifacts")
        hashes = artifacts.get("sha256") if isinstance(artifacts, Mapping) else None
        if not isinstance(hashes, Mapping):
            raise RuntimeError("continuation catalog entry has no artifact hashes")
        for field, filename in (
            ("config_path", "resolved_config.json"),
            ("result_path", "result.json"),
            ("trace_path", "trace.npz"),
        ):
            source_raw = record.get(field)
            if source_raw is None:
                raise RuntimeError("final exact-rerun catalog candidate lost an artifact")
            source = Path(str(source_raw)).expanduser().resolve()
            key = field.removesuffix("_path")
            relative = f"{trajectory_id}/{filename}"
            target = catalog_root / relative
            if (
                artifacts.get(key) != relative
                or not target.is_file()
                or hashes.get(key) != file_sha256(target)
                or file_sha256(source) != file_sha256(target)
            ):
                raise RuntimeError("continuation catalog artifact binding changed")

    expected_aliases = {
        f"pair_rank_{index:02d}": f"candidate_{candidate_id}"
        for index, candidate_id in enumerate(expected_ids, 1)
    }
    if successes:
        expected_aliases["best_first"] = f"candidate_{successes[0]}"
        expected_aliases["best_nominal"] = f"candidate_{successes[0]}"
    elif expected_ids:
        expected_aliases["best_attempt"] = f"candidate_{expected_ids[0]}"
    if robust_candidate_id is not None and robust_candidate_id in successes:
        expected_aliases["best_robust"] = f"candidate_{robust_candidate_id}"
    if (
        payload.get("aliases") != expected_aliases
        or int(payload.get("success_count", -1)) != len(successes)
    ):
        raise RuntimeError("continuation catalog aliases changed")
    for entry in trajectories:
        expected = sorted(
            alias
            for alias, target in expected_aliases.items()
            if target == entry["trajectory_id"]
        )
        if entry.get("aliases") != expected:
            raise RuntimeError("continuation catalog entry aliases changed")
    return path


def _publish_committed_catalog(
    workspace: Path,
    *,
    target_success_count: int,
    records: Sequence[Mapping[str, Any]],
    robust_candidate_id: int | None,
    stage_input: Mapping[str, Any],
) -> Path:
    stage = f"catalog_target_{target_success_count}"
    catalog_root = (
        workspace
        / "catalogs"
        / f"target_{target_success_count}"
        / "manipulation"
    )
    ledger = validate_stage_ledger(workspace)
    committed = ledger["stages"].get(stage)
    if committed is not None:
        if committed.get("stage_input_sha256") != canonical_sha256(stage_input):
            raise RuntimeError("committed catalog input changed")
        return _validate_catalog(
            catalog_root, records, robust_candidate_id=robust_candidate_id
        )

    staging = catalog_root.parent / ".manipulation.v15-staging"
    if staging.exists() or staging.is_symlink():
        _remove_uncommitted_catalog_tree(
            staging, expected_parent=catalog_root.parent
        )
    if catalog_root.exists() or catalog_root.is_symlink():
        try:
            _validate_catalog(
                catalog_root, records, robust_candidate_id=robust_candidate_id
            )
        except (FileNotFoundError, RuntimeError, ValueError, TypeError):
            _remove_uncommitted_catalog_tree(
                catalog_root, expected_parent=catalog_root.parent
            )
    if not catalog_root.exists():
        catalog_root.parent.mkdir(parents=True, exist_ok=True)
        publish_v15_viewer_catalog(
            records,
            staging,
            robust_candidate_id=robust_candidate_id,
        )
        _validate_catalog(
            staging, records, robust_candidate_id=robust_candidate_id
        )
        staging.rename(catalog_root)
    catalog_path = _validate_catalog(
        catalog_root, records, robust_candidate_id=robust_candidate_id
    )
    commit_campaign_stage(
        workspace,
        stage,
        stage_input=stage_input,
        artifacts=tuple(
            path for path in catalog_root.rglob("*") if path.is_file()
        ),
    )
    return catalog_path


def _commit_target_result(
    workspace: Path,
    *,
    target_success_count: int,
    result: Mapping[str, Any],
    stage_input: Mapping[str, Any],
) -> dict[str, Any]:
    stage = f"continuation_result_target_{target_success_count}"
    path = workspace / f"continuation_result_target_{target_success_count}.json"
    ledger = validate_stage_ledger(workspace)
    committed = ledger["stages"].get(stage)
    expected = copy.deepcopy(dict(result))
    if committed is not None:
        if committed.get("stage_input_sha256") != canonical_sha256(stage_input):
            raise RuntimeError("committed continuation result input changed")
        observed = _load_mapping(path)
        if canonical_sha256(observed) != canonical_sha256(expected):
            raise RuntimeError("committed continuation result changed")
        return observed
    write_json(path, expected)
    commit_campaign_stage(
        workspace,
        stage,
        stage_input=stage_input,
        artifacts=(path,),
        summary={
            "full_success_count": int(expected["full_success_count"]),
            "target_reached": bool(expected["target_reached"]),
        },
    )
    return expected


def run_refinement_continuation(
    source_workspace: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path,
    resume: bool,
    target_success_count: int,
    backend: V15CampaignBackend,
    repository_root: str | Path = REPO_ROOT,
    budget: JointPairNearZeroBudget = JointPairNearZeroBudget(),
) -> dict[str, Any]:
    """Run corrected refinement and all success-evidence stages."""

    if target_success_count not in (1, 5):
        raise ValueError("target_success_count must be one or five")
    source_root = Path(source_workspace).expanduser().resolve()
    requested_workspace = Path(output_dir).expanduser().resolve()
    if (
        requested_workspace == source_root
        or requested_workspace.is_relative_to(source_root)
        or source_root.is_relative_to(requested_workspace)
    ):
        raise ValueError(
            "continuation output must be disjoint from its immutable source workspace"
        )
    source = authenticate_feedback_source(
        source_root,
        config_path=config_path,
        repository_root=repository_root,
        budget=budget,
    )
    manifest = build_continuation_manifest(
        source,
        config_path=config_path,
        repository_root=repository_root,
        budget=budget,
    )
    workspace = initialize_or_resume_campaign(
        requested_workspace, manifest, resume=resume
    )
    imported_path = workspace / "source" / "authenticated_feedback_source.json"
    ledger = validate_stage_ledger(workspace)
    source_payload = {
        "joint_pair_near_zero_feedback_source_schema_version": 1,
        "complete": True,
        "experiment_id": EXPERIMENT_ID,
        **source.binding,
    }
    if "source_import" not in ledger["stages"]:
        imported_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(imported_path, source_payload)
        commit_campaign_stage(
            workspace,
            "source_import",
            stage_input=source.binding,
            artifacts=(imported_path,),
            summary={"feedback_record_count": len(source.records)},
        )
    else:
        committed_import = ledger["stages"]["source_import"]
        if (
            committed_import.get("stage_input_sha256")
            != canonical_sha256(source.binding)
            or canonical_sha256(_load_mapping(imported_path))
            != canonical_sha256(source_payload)
        ):
            raise RuntimeError("committed continuation source import changed")

    context: dict[str, Any] = {
        "seed_config_path": str(
            source.workspace / "source" / "resolved_seed_config.json"
        ),
        # Target one versus five changes publication only, never the
        # deterministic physics stream or its committed stage descriptors.
        "feedback_source_id": source.binding["feedback_source_id"],
        # Refinement contains 1,024 full-reset grasp near misses.  They retain
        # authenticated config/result evidence while exact/final reruns below
        # still force the trace from their own session.
        "retain_grasp_trace": False,
    }
    refine_jobs = build_refinement_jobs(source.records, budget)
    refined = _run_continuation_stage(
        workspace,
        "feedback_refinement",
        refine_jobs,
        backend,
        {**context, "parents": source.records},
    )
    exact_parents: Sequence[Mapping[str, Any]] = (*source.records, *refined)
    exact_jobs = build_exact_rerun_jobs(exact_parents, budget)
    exact = _run_continuation_stage(
        workspace,
        "exact_rerun",
        exact_jobs,
        backend,
        {**context, "parents": exact_parents},
    )
    perturb_jobs = build_perturbation_jobs(exact, budget)
    perturb = _run_continuation_stage(
        workspace,
        "local_perturbation",
        perturb_jobs,
        backend,
        {**context, "parents": exact},
    )
    pass_counts: dict[int, int] = {}
    for record in perturb:
        parent = int(record.get("parent_candidate_id", -1))
        pass_counts[parent] = pass_counts.get(parent, 0) + int(
            bool(record.get("full_success"))
        )
    ranked_exact: list[dict[str, Any]] = []
    for record in exact:
        materialized = copy.deepcopy(dict(record))
        materialized["perturbation_pass_count"] = pass_counts.get(
            _record_id(record), 0
        )
        ranked_exact.append(materialized)
    robust_jobs = build_robustness_jobs(ranked_exact, budget)
    robustness = _run_continuation_stage(
        workspace,
        "robustness",
        robust_jobs,
        backend,
        {**context, "parents": ranked_exact},
    )
    robust_parent = None
    if (
        robustness
        and sum(bool(value.get("full_success")) for value in robustness)
        >= budget.robustness_required_passes
    ):
        robust_parent = robust_jobs[0].parent_candidate_id

    stage_report_sha256 = {
        stage: file_sha256(workspace / "stages" / stage / "report.json")
        for stage in (
            "feedback_refinement",
            "exact_rerun",
            "local_perturbation",
            "robustness",
        )
    }
    catalog_stage_input = {
        "target_success_count": int(target_success_count),
        "feedback_source_id": source.binding["feedback_source_id"],
        "upstream_feedback_report_sha256": source.binding[
            "feedback_report_sha256"
        ],
        "stage_report_sha256": stage_report_sha256,
        "ranked_exact_sha256": canonical_sha256(ranked_exact),
        "robust_candidate_id": robust_parent,
    }
    catalog_path = _publish_committed_catalog(
        workspace,
        target_success_count=target_success_count,
        records=ranked_exact,
        robust_candidate_id=robust_parent,
        stage_input=catalog_stage_input,
    )

    successes = [value for value in ranked_exact if bool(value.get("full_success"))]
    result = {
        "joint_pair_near_zero_refinement_continuation_result_schema_version": 1,
        "complete": True,
        "experiment_id": EXPERIMENT_ID,
        "workspace": str(workspace),
        "source_workspace": str(source.workspace),
        "target_success_count": int(target_success_count),
        "full_success_count": len(successes),
        "target_reached": len(successes) >= target_success_count,
        "robust_candidate_id": robust_parent,
        "catalog_path": str(catalog_path),
        "stage_counts": {
            "imported_feedback": len(source.records),
            "feedback_refinement": len(refined),
            "exact_rerun": len(exact),
            "local_perturbation": len(perturb),
            "robustness": len(robustness),
        },
    }
    result_stage_input = {
        **catalog_stage_input,
        "catalog_sha256": file_sha256(catalog_path),
    }
    return _commit_target_result(
        workspace,
        target_success_count=target_success_count,
        result=result,
        stage_input=result_stage_input,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume schema-v15 after an authenticated feedback-grid stage"
    )
    parser.add_argument("--source-workspace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config",
        default=str(
            REPO_ROOT
            / "grasp_configs"
            / "left_opposed_face_palm_down_joint_pair_near_zero_"
            "contact_preserving_planned_lift.json"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target-success-count", type=int, choices=(1, 5), default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    budget = JointPairNearZeroBudget(seed=int(args.seed))
    result = run_refinement_continuation(
        args.source_workspace,
        args.output_dir,
        config_path=args.config,
        resume=bool(args.resume),
        target_success_count=int(args.target_success_count),
        backend=create_joint_pair_near_zero_campaign_backend(
            workers=int(args.workers), seed=int(args.seed)
        ),
        budget=budget,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["target_reached"] else 2)


if __name__ == "__main__":
    main()


__all__ = [
    "AuthenticatedFeedbackSource",
    "CONTINUATION_SCHEMA_VERSION",
    "authenticate_feedback_source",
    "build_continuation_manifest",
    "run_refinement_continuation",
]
