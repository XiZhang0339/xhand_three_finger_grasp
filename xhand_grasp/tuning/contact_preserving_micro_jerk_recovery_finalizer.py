"""Fail-closed finalizer for a completed, source-drifted micro-jerk run.

The original 1,024 simulations are immutable execution evidence.  They may
finish successfully while the normal campaign runner correctly refuses final
publication because the repository source tree changed during execution.
This module never resumes or edits that workspace.  Instead it authenticates
the old manifest, ledger, complete phase report and every candidate bundle,
then creates a disjoint recovery workspace bound to both source snapshots.

Only the ranked top five are rerun under the current source tree.  A single
semantic-summary difference stops publication.  Checkpoint or summary-only
evidence is never accepted in place of those independent full-reset reruns.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..actual_contact_grasp_pose_catalog import (
    authenticated_catalog_artifact_paths,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from ..artifacts import (
    aggregate_source_sha256,
    file_sha256,
    implementation_paths,
)
from ..grasp_pose import canonical_sha256
from .contact_preserving_candidate_artifacts import authenticate_v14_candidate_artifacts
from .contact_preserving_force_debias_rescue_campaign import _restrict_catalog_aliases
from .contact_preserving_lift_rescue_campaign import (
    _commit_report,
    _load_committed_report,
    _publish_rescue_viewer_catalogs,
    _run_full_reset_candidate,
)
from .contact_preserving_micro_jerk_rescue import (
    MICRO_CANDIDATE_COUNT,
    authenticate_micro_jerk_source,
    micro_jerk_candidate_rank,
)


RECOVERY_FINALIZER_SCHEMA_VERSION = 1
RECOVERY_RESULT_SCHEMA_VERSION = 1
PUBLISHED_CANDIDATE_COUNT = 5
EXPERIMENT_ID = "left_opposed_face_palm_down_contact_preserving_planned_lift"
SOURCE_STAGE = "micro_jerk_refinement"
SOURCE_REPORT_STAGE = "micro_refinement"


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _manifest_semantic_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: copy.deepcopy(value) for key, value in payload.items() if key != "campaign_input_sha256"}
    )


def _source_file_paths(manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    raw = manifest.get("source_files")
    if not isinstance(raw, list) or not raw or not all(isinstance(value, str) for value in raw):
        raise RuntimeError("old micro manifest lost its execution source-file list")
    # These paths describe the old snapshot only.  Their current contents are
    # intentionally not compared with the old aggregate digest.
    return tuple(Path(value) for value in raw)


def current_source_snapshot() -> dict[str, Any]:
    paths = tuple(implementation_paths())
    return {
        "source_sha256": aggregate_source_sha256(paths),
        "source_files": [
            str(path.relative_to(Path(__file__).resolve().parents[2]))
            for path in paths
        ],
    }


@dataclass(frozen=True, slots=True)
class AuthenticatedMicroJerkExecution:
    root: Path
    manifest_path: Path
    ledger_path: Path
    report_path: Path
    manifest: dict[str, Any]
    ledger: dict[str, Any]
    report: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    artifact_paths: tuple[Path, ...]
    old_execution_source_sha256: str
    old_execution_source_files: tuple[str, ...]
    execution_authentication_id: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "recovery_finalizer_schema_version": RECOVERY_FINALIZER_SCHEMA_VERSION,
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": file_sha256(self.manifest_path),
            "ledger_path": str(self.ledger_path),
            "ledger_sha256": file_sha256(self.ledger_path),
            "report_path": str(self.report_path),
            "report_sha256": file_sha256(self.report_path),
            "candidate_count": len(self.records),
            "candidate_artifact_set_sha256": canonical_sha256(
                {
                    str(path.relative_to(self.root)): file_sha256(path)
                    for path in self.artifact_paths
                    if path.is_relative_to(self.root)
                }
            ),
            "old_execution_source_sha256": self.old_execution_source_sha256,
            "old_execution_source_files": list(self.old_execution_source_files),
            "execution_authentication_id": self.execution_authentication_id,
        }


def _validate_phase_report_shape(
    report: Mapping[str, Any], *, expected_source_authentication_id: str
) -> tuple[dict[str, Any], ...]:
    records = report.get("records")
    if (
        report.get("complete") is not True
        or report.get("stage") != SOURCE_REPORT_STAGE
        or int(report.get("declared_candidate_count", -1)) != MICRO_CANDIDATE_COUNT
        or int(report.get("candidate_count", -1)) != MICRO_CANDIDATE_COUNT
        or report.get("source_authentication_id") != expected_source_authentication_id
        or not isinstance(records, list)
        or len(records) != MICRO_CANDIDATE_COUNT
    ):
        raise RuntimeError("old micro refinement report is incomplete or incompatible")
    normalized = tuple(copy.deepcopy(dict(value)) for value in records if isinstance(value, Mapping))
    if len(normalized) != MICRO_CANDIDATE_COUNT:
        raise RuntimeError("old micro refinement report contains a malformed record")
    candidate_ids = tuple(int(value.get("candidate_id", -1)) for value in normalized)
    if min(candidate_ids) < 0 or len(set(candidate_ids)) != MICRO_CANDIDATE_COUNT:
        raise RuntimeError("old micro refinement candidate identities are not unique")
    physical: list[str] = []
    for value in normalized:
        metadata = value.get("rescue_job")
        digest = metadata.get("physical_plan_sha256") if isinstance(metadata, Mapping) else None
        if not _is_sha(digest):
            raise RuntimeError("old micro record lost physical-plan provenance")
        physical.append(str(digest))
    if len(set(physical)) != MICRO_CANDIDATE_COUNT:
        raise RuntimeError("old micro refinement reused a physical plan")
    if int(report.get("physical_unique_candidate_count", -1)) != MICRO_CANDIDATE_COUNT:
        raise RuntimeError("old micro report physical uniqueness count changed")
    if int(report.get("physical_duplicate_candidate_count", -1)) != 0:
        raise RuntimeError("old micro report declares duplicate physical plans")
    if report.get("physical_plan_set_sha256") != canonical_sha256(sorted(physical)):
        raise RuntimeError("old micro report physical-plan set digest changed")
    recomputed_success = sum(bool(value.get("full_success", False)) for value in normalized)
    if int(report.get("full_success_count", -1)) != recomputed_success:
        raise RuntimeError("old micro report success count changed")
    return normalized


def authenticate_completed_micro_jerk_execution(
    source_workspace: str | Path,
) -> AuthenticatedMicroJerkExecution:
    root = Path(source_workspace).expanduser().resolve()
    manifest_path = root / "campaign_manifest.json"
    ledger_path = root / "stage_ledger.json"
    report_path = root / SOURCE_STAGE / "report.json"
    if not root.is_dir() or not all(path.is_file() for path in (manifest_path, ledger_path, report_path)):
        raise RuntimeError("old micro workspace is not a completed execution source")
    manifest = _load_object(manifest_path, "old micro manifest")
    if (
        manifest.get("campaign_kind") != "contact_preserving_bounded_micro_jerk_rescue"
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("campaign_input_sha256") != _manifest_semantic_sha256(manifest)
        or int(manifest.get("micro_jerk_budget", {}).get("candidate_count", -1))
        != MICRO_CANDIDATE_COUNT
        or not _is_sha(manifest.get("source_sha256"))
    ):
        raise RuntimeError("old micro manifest is incompatible or changed")
    old_source_files = tuple(str(value) for value in _source_file_paths(manifest))
    for name in ("config", "model", "uv_lock"):
        path = Path(str(manifest.get(f"{name}_path", ""))).expanduser().resolve()
        expected = manifest.get(f"{name}_sha256")
        if not path.is_file() or not _is_sha(expected) or file_sha256(path) != expected:
            raise RuntimeError(f"old micro immutable {name} input changed")
    source_campaign = Path(str(manifest.get("source_force_debias_campaign_path", ""))).resolve()
    authenticated_parent = authenticate_micro_jerk_source(source_campaign)
    if canonical_sha256(authenticated_parent.as_mapping()) != canonical_sha256(
        manifest.get("source_authentication")
    ):
        raise RuntimeError("old micro authenticated parent evidence changed")
    ledger = validate_stage_ledger(root)
    stage = ledger.get("stages", {}).get(SOURCE_STAGE)
    if not isinstance(stage, Mapping) or stage.get("complete") is not True:
        raise RuntimeError("old micro refinement stage is not atomically committed")
    artifacts = stage.get("artifacts")
    report_relative = str(report_path.relative_to(root))
    if not isinstance(artifacts, Mapping) or artifacts.get(report_relative) != file_sha256(report_path):
        raise RuntimeError("old micro refinement report is not bound by its ledger")
    report = _load_object(report_path, "old micro refinement report")
    source_id = str(manifest["source_authentication"]["source_authentication_id"])
    records = _validate_phase_report_shape(
        report, expected_source_authentication_id=source_id
    )
    artifact_paths: list[Path] = [manifest_path, ledger_path, report_path]
    for record in records:
        relative = record.get("artifact_directory")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RuntimeError("old micro candidate path is unsafe")
        destination = (root / relative).resolve()
        if not destination.is_relative_to(root):
            raise RuntimeError("old micro candidate escaped its workspace")
        bundle = authenticate_v14_candidate_artifacts(
            destination,
            expected_candidate_id=int(record["candidate_id"]),
            expected_retain_grasp_success=False,
        )
        if canonical_sha256(bundle.result.get("summary")) != canonical_sha256(
            record.get("summary")
        ):
            raise RuntimeError("old micro candidate summary changed")
        if record.get("config_semantic_sha256") != bundle.result.get(
            "config_semantic_sha256"
        ):
            raise RuntimeError("old micro candidate config identity changed")
        artifact_paths.extend(bundle.artifact_paths)
    immutable = tuple(dict.fromkeys(path.resolve() for path in artifact_paths))
    authentication_payload = {
        "recovery_finalizer_schema_version": RECOVERY_FINALIZER_SCHEMA_VERSION,
        "root": str(root),
        "manifest_sha256": file_sha256(manifest_path),
        "ledger_sha256": file_sha256(ledger_path),
        "report_sha256": file_sha256(report_path),
        "candidate_artifact_set_sha256": canonical_sha256(
            {
                str(path.relative_to(root)): file_sha256(path)
                for path in immutable
                if path.is_relative_to(root)
            }
        ),
        "old_execution_source_sha256": manifest["source_sha256"],
        "old_execution_source_files": list(old_source_files),
    }
    return AuthenticatedMicroJerkExecution(
        root=root,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        report_path=report_path,
        manifest=copy.deepcopy(manifest),
        ledger=copy.deepcopy(ledger),
        report=copy.deepcopy(report),
        records=records,
        artifact_paths=immutable,
        old_execution_source_sha256=str(manifest["source_sha256"]),
        old_execution_source_files=old_source_files,
        execution_authentication_id=canonical_sha256(authentication_payload),
    )


def _rank_records(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        sorted(
            (copy.deepcopy(dict(value)) for value in records),
            key=lambda value: (*micro_jerk_candidate_rank(value), int(value["candidate_id"])),
        )
    )


def build_micro_jerk_recovery_manifest(
    source: AuthenticatedMicroJerkExecution,
) -> dict[str, Any]:
    current = current_source_snapshot()
    selected = _rank_records(source.records)[:PUBLISHED_CANDIDATE_COUNT]
    base = {
        "campaign_manifest_schema_version": 1,
        "micro_jerk_recovery_finalizer_schema_version": RECOVERY_FINALIZER_SCHEMA_VERSION,
        "campaign_kind": "contact_preserving_micro_jerk_recovery_finalizer",
        "experiment_id": EXPERIMENT_ID,
        "seed": int(source.manifest["seed"]),
        "config_path": str(source.manifest["config_path"]),
        "config_sha256": str(source.manifest["config_sha256"]),
        "model_path": str(source.manifest["model_path"]),
        "model_sha256": str(source.manifest["model_sha256"]),
        "uv_lock_path": str(source.manifest["uv_lock_path"]),
        "uv_lock_sha256": str(source.manifest["uv_lock_sha256"]),
        "actual_qpos_source_manifest_path": str(
            source.manifest["actual_qpos_source_manifest_path"]
        ),
        "actual_qpos_source_manifest_sha256": str(
            source.manifest["actual_qpos_source_manifest_sha256"]
        ),
        "source_workspace": str(source.root),
        "source_execution_authentication": source.as_mapping(),
        "old_execution_source_sha256": source.old_execution_source_sha256,
        "old_execution_source_files": list(source.old_execution_source_files),
        "source_sha256": str(current["source_sha256"]),
        "source_files": copy.deepcopy(current["source_files"]),
        "selected_candidate_ids": [int(value["candidate_id"]) for value in selected],
        "selected_records_sha256": canonical_sha256(selected),
        "recovery_contract": {
            "source_candidate_count": MICRO_CANDIDATE_COUNT,
            "publication_candidate_count": PUBLISHED_CANDIDATE_COUNT,
            "rerun": "current_source_fresh_full_reset",
            "summary_match": "exact_canonical_required",
            "source_workspace_mutation": False,
        },
    }
    return {**base, "campaign_input_sha256": canonical_sha256(base)}


def _assert_stage_input(
    workspace: Path, stage: str, stage_input: Mapping[str, Any]
) -> None:
    stage_record = validate_stage_ledger(workspace).get("stages", {}).get(stage)
    if not isinstance(stage_record, Mapping):
        raise RuntimeError(f"committed recovery stage {stage} disappeared")
    if stage_record.get("stage_input_sha256") != canonical_sha256(dict(stage_input)):
        raise RuntimeError(f"committed recovery stage {stage} input changed")


def _validate_top_five_report(
    payload: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], workspace: Path
) -> tuple[dict[str, Any], ...]:
    records = payload.get("records")
    expected_ids = [int(value["candidate_id"]) for value in selected]
    if (
        payload.get("complete") is not True
        or int(payload.get("candidate_count", -1)) != PUBLISHED_CANDIDATE_COUNT
        or not isinstance(records, list)
        or [int(value.get("candidate_id", -1)) for value in records] != expected_ids
    ):
        raise RuntimeError("micro recovery top-five report changed")
    normalized: list[dict[str, Any]] = []
    for record, source_record in zip(records, selected, strict=True):
        destination = (workspace / str(record["artifact_directory"])).resolve()
        bundle = authenticate_v14_candidate_artifacts(
            destination,
            expected_candidate_id=int(record["candidate_id"]),
            require_retained_trace=True,
            expected_retain_grasp_success=False,
        )
        if canonical_sha256(bundle.result.get("summary")) != canonical_sha256(
            source_record.get("summary")
        ):
            raise RuntimeError("micro recovery rerun summary differs from old execution")
        normalized.append(copy.deepcopy(dict(record)))
    return tuple(normalized)


def _top_five_stage(
    workspace: Path,
    source: AuthenticatedMicroJerkExecution,
) -> tuple[tuple[dict[str, Any], ...], Path]:
    selected = _rank_records(source.records)[:PUBLISHED_CANDIDATE_COUNT]
    stage = "micro_recovery_top_five_rerun"
    report_path = workspace / stage / "report.json"
    stage_input = {
        "execution_authentication_id": source.execution_authentication_id,
        "current_source_sha256": current_source_snapshot()["source_sha256"],
        "selected_records_sha256": canonical_sha256(selected),
    }
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        _assert_stage_input(workspace, stage, stage_input)
        return _validate_top_five_report(existing, selected, workspace), report_path
    records: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    for source_record in selected:
        candidate_id = int(source_record["candidate_id"])
        old_root = (source.root / str(source_record["artifact_directory"])).resolve()
        old_bundle = authenticate_v14_candidate_artifacts(
            old_root,
            expected_candidate_id=candidate_id,
            expected_retain_grasp_success=False,
        )
        config = _load_object(old_bundle.config_path, "old selected candidate config")
        destination = workspace / stage / "candidates" / f"candidate_{candidate_id}"
        rerun = _run_full_reset_candidate(
            config, destination, candidate_id, final_rerun=True
        )
        bundle = authenticate_v14_candidate_artifacts(
            destination,
            expected_config=config,
            expected_candidate_id=candidate_id,
            require_retained_trace=True,
            expected_retain_grasp_success=False,
        )
        if canonical_sha256(rerun.get("summary")) != canonical_sha256(
            source_record.get("summary")
        ):
            raise RuntimeError(
                f"micro recovery candidate {candidate_id} changed under current source"
            )
        artifacts.extend(bundle.artifact_paths)
        records.append(
            {
                **copy.deepcopy(dict(source_record)),
                **copy.deepcopy(bundle.result),
                "artifact_directory": str(destination.relative_to(workspace)),
                "old_execution_summary_sha256": canonical_sha256(
                    source_record["summary"]
                ),
                "current_rerun_summary_sha256": canonical_sha256(
                    bundle.result["summary"]
                ),
            }
        )
    payload = {
        "micro_jerk_recovery_top_five_report_schema_version": 1,
        "complete": True,
        "candidate_count": len(records),
        "full_success_count": sum(bool(value.get("full_success", False)) for value in records),
        "execution_authentication_id": source.execution_authentication_id,
        "current_source_sha256": stage_input["current_source_sha256"],
        "records": records,
    }
    committed = _commit_report(
        workspace,
        stage,
        report_path,
        payload,
        stage_input=stage_input,
        artifacts=tuple(dict.fromkeys(artifacts)),
    )
    return _validate_top_five_report(committed, selected, workspace), report_path


def _catalog_stage(
    workspace: Path,
    records: Sequence[Mapping[str, Any]],
    report_path: Path,
) -> dict[str, Any]:
    stage = "micro_recovery_catalog"
    root = workspace / "catalogs" / "target_1"
    output = root / "report.json"
    stage_input = {
        "top_five_report_sha256": file_sha256(report_path),
        "records_sha256": canonical_sha256(records),
    }
    existing = _load_committed_report(workspace, stage, output)
    if existing is not None:
        _assert_stage_input(workspace, stage, stage_input)
        return existing
    catalogs = _publish_rescue_viewer_catalogs(
        records, workspace, root, experiment_id=EXPERIMENT_ID
    )
    hard_ids = tuple(
        int(value["candidate_id"])
        for value in records
        if bool(value.get("full_success", False))
    )
    artifacts: list[Path] = []
    for relative in catalogs.values():
        path = workspace / relative
        _restrict_catalog_aliases(path, hard_pass_candidate_ids=hard_ids)
        artifacts.extend(authenticated_catalog_artifact_paths(path))
    payload = {
        "micro_jerk_recovery_catalog_report_schema_version": 1,
        "complete": True,
        "published_candidate_count": len(records),
        "published_candidate_ids": [int(value["candidate_id"]) for value in records],
        "full_success_count": len(hard_ids),
        "catalogs": catalogs,
        "records": [],
    }
    return _commit_report(
        workspace,
        stage,
        output,
        payload,
        stage_input=stage_input,
        artifacts=tuple(dict.fromkeys(artifacts)),
    )


def finalize_completed_micro_jerk_campaign(
    source_workspace: str | Path,
    output_dir: str | Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    source = authenticate_completed_micro_jerk_execution(source_workspace)
    workspace = Path(output_dir).expanduser().resolve()
    if workspace == source.root or workspace.is_relative_to(source.root):
        raise ValueError("micro recovery workspace must be disjoint from old execution")
    manifest = build_micro_jerk_recovery_manifest(source)
    initialize_or_resume_campaign(workspace, manifest, resume=bool(resume))

    audit_path = workspace / "micro_recovery_source_audit.json"
    audit_payload = {
        "micro_jerk_recovery_source_audit_schema_version": 1,
        "complete": True,
        "source_execution": source.as_mapping(),
        "current_source_snapshot": current_source_snapshot(),
        "records": [],
    }
    audit_input = {
        "execution_authentication_id": source.execution_authentication_id,
        "current_source_sha256": manifest["source_sha256"],
    }
    audit = _load_committed_report(
        workspace, "micro_recovery_source_audit", audit_path
    )
    if audit is None:
        _commit_report(
            workspace,
            "micro_recovery_source_audit",
            audit_path,
            audit_payload,
            stage_input=audit_input,
        )
    else:
        _assert_stage_input(workspace, "micro_recovery_source_audit", audit_input)
        if canonical_sha256(audit) != canonical_sha256(audit_payload):
            raise RuntimeError("micro recovery source audit changed on resume")

    records, rerun_report = _top_five_stage(workspace, source)
    catalog = _catalog_stage(workspace, records, rerun_report)
    full_count = sum(bool(value.get("full_success", False)) for value in records)
    result = {
        "micro_jerk_recovery_result_schema_version": RECOVERY_RESULT_SCHEMA_VERSION,
        "complete": True,
        "experiment_id": EXPERIMENT_ID,
        "source_workspace": str(source.root),
        "execution_authentication_id": source.execution_authentication_id,
        "old_execution_source_sha256": source.old_execution_source_sha256,
        "current_source_sha256": manifest["source_sha256"],
        "source_candidate_count": MICRO_CANDIDATE_COUNT,
        "rerun_candidate_count": len(records),
        "full_success_count": full_count,
        "catalogs": copy.deepcopy(dict(catalog["catalogs"])),
        "published_candidate_ids": copy.deepcopy(list(catalog["published_candidate_ids"])),
        "exit_code": 0 if full_count else 2,
        "stop_reason": "recovered_current_source_top_five_published",
    }

    final_source = authenticate_completed_micro_jerk_execution(source.root)
    if final_source.execution_authentication_id != source.execution_authentication_id:
        raise RuntimeError("old micro execution changed during recovery")
    if current_source_snapshot()["source_sha256"] != manifest["source_sha256"]:
        raise RuntimeError("current source tree changed during recovery")
    validate_stage_ledger(workspace)
    result_path = workspace / "micro_jerk_recovery_result.json"
    stage = "micro_recovery_result"
    stage_input = {
        "top_five_report_sha256": file_sha256(rerun_report),
        "catalog_report_sha256": file_sha256(
            workspace / "catalogs" / "target_1" / "report.json"
        ),
    }
    existing = _load_committed_report(workspace, stage, result_path)
    if existing is None:
        return _commit_report(
            workspace, stage, result_path, result, stage_input=stage_input
        )
    _assert_stage_input(workspace, stage, stage_input)
    if canonical_sha256(existing) != canonical_sha256(result):
        raise RuntimeError("committed micro recovery result changed on resume")
    return existing


__all__ = [
    "AuthenticatedMicroJerkExecution",
    "authenticate_completed_micro_jerk_execution",
    "build_micro_jerk_recovery_manifest",
    "current_source_snapshot",
    "finalize_completed_micro_jerk_campaign",
]
