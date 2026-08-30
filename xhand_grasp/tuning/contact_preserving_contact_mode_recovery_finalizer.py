"""Fail-closed finalization of an interrupted contact-mode rescue campaign.

The original contact-mode workspace is immutable execution evidence.  A
source-tree change after its search and catalog stages were committed makes a
normal resume correctly fail, but does not invalidate those already sealed
physics artifacts.  This module therefore:

* authenticates the old manifest, ledger, all 256 retained candidates and
  both published Top-5 catalogs without rebuilding the old manifest;
* binds the complete authenticated evidence digest and the *old manifest file
  digest* into a separate recovery workspace whose manifest covers the
  current source tree;
* independently reruns the published Top-5 from their initial no-contact
  state with current code and requires exact summaries and exact NPZ physical
  fields (apart from the renderer-only ``video_frame_steps`` field); and
* publishes new Viewer catalogs and a final result only after every comparison
  succeeds.

No file in the source workspace is written.  A mismatch at any boundary is a
hard error and no later recovery stage can be committed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
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
from .contact_preserving_candidate_artifacts import (
    V14CandidateArtifactBundle,
    authenticate_v14_candidate_artifacts,
)
from .contact_preserving_contact_mode_pose_rescue import (
    MAXIMUM_CANDIDATE_COUNT,
    contact_mode_physical_config_sha256,
    contact_mode_trace_diagnostics,
)
from .contact_preserving_contact_mode_pose_rescue_campaign import (
    PUBLISHED_CANDIDATE_COUNT,
    _validate_search_report,
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


RECOVERY_FINALIZER_SCHEMA_VERSION = 1
RECOVERY_RESULT_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
_IGNORED_TRACE_FIELDS = frozenset({"video_frame_steps"})
_REQUIRED_SOURCE_STAGES = frozenset(
    {
        "contact_mode_pose_source_audit",
        "contact_mode_pose_search",
        "contact_mode_pose_catalog_target_1",
    }
)
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class _RecoveryPolicy:
    search_candidate_count: int = MAXIMUM_CANDIDATE_COUNT
    published_candidate_count: int = PUBLISHED_CANDIDATE_COUNT

    def __post_init__(self) -> None:
        if isinstance(self.search_candidate_count, bool) or int(
            self.search_candidate_count
        ) <= 0:
            raise ValueError("search_candidate_count must be positive")
        if isinstance(self.published_candidate_count, bool) or int(
            self.published_candidate_count
        ) <= 0:
            raise ValueError("published_candidate_count must be positive")
        if int(self.published_candidate_count) > int(self.search_candidate_count):
            raise ValueError("published count cannot exceed the search count")


PRODUCTION_POLICY = _RecoveryPolicy()


@dataclass(frozen=True)
class ContactModeRecoverySource:
    """Fully authenticated, read-only source campaign evidence."""

    root: Path
    manifest: dict[str, Any]
    ledger: dict[str, Any]
    search_report: dict[str, Any]
    catalog_report: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    top_records: tuple[dict[str, Any], ...]
    configs: Mapping[int, dict[str, Any]]
    search_trace_paths: Mapping[int, Path]
    reference_trace_paths: Mapping[int, Path]
    catalog_paths: Mapping[str, Path]
    evidence: dict[str, Any]
    evidence_sha256: str

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": RECOVERY_FINALIZER_SCHEMA_VERSION,
            "root": str(self.root),
            "experiment_id": self.manifest["experiment_id"],
            "source_manifest_sha256": self.evidence["source_manifest_sha256"],
            "source_campaign_input_sha256": self.manifest[
                "campaign_input_sha256"
            ],
            "source_ledger_sha256": self.evidence["source_ledger_sha256"],
            "search_report_sha256": self.evidence["search_report_sha256"],
            "catalog_report_sha256": self.evidence["catalog_report_sha256"],
            "candidate_count": len(self.records),
            "published_candidate_ids": [
                int(value["candidate_id"]) for value in self.top_records
            ],
            "ledger_artifact_count": int(
                self.evidence["ledger_artifact_count"]
            ),
            "complete_workspace_evidence_sha256": self.evidence_sha256,
            "old_manifest_is_immutable_execution_evidence": True,
            "read_only": True,
        }


CandidateRunner = Callable[[Mapping[str, Any], Path, int], Mapping[str, Any]]
CatalogPublisher = Callable[
    [Sequence[Mapping[str, Any]], Path, Path, str], Mapping[str, str]
]


@dataclass(frozen=True)
class _RecoveryBackend:
    backend_id: str
    candidate_runner: CandidateRunner
    catalog_publisher: CatalogPublisher


def _production_candidate_runner(
    config: Mapping[str, Any], destination: Path, candidate_id: int
) -> Mapping[str, Any]:
    return _run_full_reset_candidate(
        config, destination, int(candidate_id), final_rerun=True
    )


def _production_catalog_publisher(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    destination: Path,
    experiment_id: str,
) -> Mapping[str, str]:
    return _publish_rescue_viewer_catalogs(
        records, workspace, destination, experiment_id=experiment_id
    )


DEFAULT_BACKEND = _RecoveryBackend(
    backend_id="production_full_reset_and_video_v1",
    candidate_runner=_production_candidate_runner,
    catalog_publisher=_production_catalog_publisher,
)


def _without_video(summary: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(summary))
    value.pop("video", None)
    return value


def _array_bytes(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise RuntimeError("object arrays are forbidden in authenticated traces")
    return np.ascontiguousarray(array).tobytes(order="C")


def _trace_field_digest(names: Sequence[str], archive: Any) -> str:
    digest = hashlib.sha256()
    for name in names:
        value = np.asarray(archive[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(_array_bytes(value))
        digest.update(b"\0")
    return digest.hexdigest()


def compare_contact_mode_physical_traces(
    reference_path: str | Path, current_path: str | Path
) -> dict[str, Any]:
    """Require exact physical fields while ignoring renderer frame binding.

    This is intentionally stronger than comparing a handful of aggregate
    metrics: every NPZ array present in both physical traces must have the same
    name, shape, dtype and byte representation.  ``video_frame_steps`` is the
    sole exception because a catalog render legitimately populates it while a
    non-rendering full-reset rerun leaves it empty.
    """

    reference = Path(reference_path).expanduser().resolve()
    current = Path(current_path).expanduser().resolve()
    if not reference.is_file() or not current.is_file():
        raise FileNotFoundError("both trace paths must exist")
    with np.load(reference, allow_pickle=False) as left, np.load(
        current, allow_pickle=False
    ) as right:
        left_names = set(left.files) - _IGNORED_TRACE_FIELDS
        right_names = set(right.files) - _IGNORED_TRACE_FIELDS
        if left_names != right_names:
            raise RuntimeError(
                "recovery trace schema changed: "
                f"missing={sorted(left_names - right_names)}, "
                f"added={sorted(right_names - left_names)}"
            )
        names = tuple(sorted(left_names))
        for name in names:
            lhs = np.asarray(left[name])
            rhs = np.asarray(right[name])
            if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
                raise RuntimeError(
                    f"recovery trace field {name} changed shape or dtype"
                )
            if _array_bytes(lhs) != _array_bytes(rhs):
                raise RuntimeError(f"recovery trace field {name} changed values")
        left_digest = _trace_field_digest(names, left)
        right_digest = _trace_field_digest(names, right)
    if left_digest != right_digest:  # Defensive: byte comparison above implies this.
        raise RuntimeError("recovery trace aggregate digest changed")
    return {
        "schema_version": RECOVERY_FINALIZER_SCHEMA_VERSION,
        "physical_fields_exact": True,
        "compared_field_count": len(names),
        "compared_field_names_sha256": canonical_sha256(list(names)),
        "physical_fields_sha256": left_digest,
        "ignored_fields": sorted(_IGNORED_TRACE_FIELDS),
        "reference_trace_file_sha256": file_sha256(reference),
        "current_trace_file_sha256": file_sha256(current),
    }


def _stage_artifacts(ledger: Mapping[str, Any], stage: str) -> dict[str, str]:
    record = ledger.get("stages", {}).get(stage)
    if not isinstance(record, Mapping) or record.get("complete") is not True:
        raise RuntimeError(f"source recovery stage {stage} is not committed")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError(f"source recovery stage {stage} lost its artifacts")
    return {str(key): str(value) for key, value in artifacts.items()}


def _require_stage_artifact(
    source: Path, artifacts: Mapping[str, str], path: Path
) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(source):
        raise RuntimeError("source artifact escaped the immutable workspace")
    relative = str(resolved.relative_to(source))
    if artifacts.get(relative) != file_sha256(resolved):
        raise RuntimeError(
            f"source ledger does not bind the expected artifact: {relative}"
        )


def _record_matches_bundle(
    record: Mapping[str, Any], bundle: V14CandidateArtifactBundle
) -> None:
    for key, value in bundle.result.items():
        if key not in record or canonical_sha256(record[key]) != canonical_sha256(
            value
        ):
            raise RuntimeError(
                f"source search report changed candidate {bundle.candidate_id} field {key}"
            )


def _catalog_member_path(
    catalog_path: Path, entry: Mapping[str, Any], name: str
) -> Path:
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("source catalog trajectory lost its artifact mapping")
    relative = artifacts.get(name)
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise RuntimeError(f"source catalog trajectory has no safe {name} path")
    path = (catalog_path.parent / relative).resolve()
    if not path.is_relative_to(catalog_path.parent.resolve()) or not path.is_file():
        raise RuntimeError(f"source catalog {name} artifact is missing")
    return path


def _authenticate_source_workspace(
    source_workspace: str | Path,
    *,
    policy: _RecoveryPolicy,
) -> ContactModeRecoverySource:
    root = Path(source_workspace).expanduser().resolve()
    manifest_path = root / "campaign_manifest.json"
    ledger_path = root / "stage_ledger.json"
    search_report_path = root / "contact_mode_pose_search" / "report.json"
    catalog_report_path = root / "catalogs" / "target_1" / "report.json"
    audit_path = root / "contact_mode_pose_source_audit.json"
    for path in (
        manifest_path,
        ledger_path,
        search_report_path,
        catalog_report_path,
        audit_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"source recovery evidence is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise RuntimeError("source campaign manifest is malformed")
    manifest = copy.deepcopy(dict(manifest))
    declared_input = manifest.get("campaign_input_sha256")
    manifest_body = {
        key: value for key, value in manifest.items() if key != "campaign_input_sha256"
    }
    if (
        manifest.get("campaign_kind")
        != "contact_preserving_contact_mode_pose_rescue"
        or int(manifest.get("contact_mode_pose_campaign_schema_version", 0)) != 1
        or declared_input != canonical_sha256(manifest_body)
    ):
        raise RuntimeError("source contact-mode manifest failed self-authentication")
    budget = manifest.get("budget")
    if not isinstance(budget, Mapping) or int(budget.get("candidate_count", -1)) != int(
        policy.search_candidate_count
    ):
        raise RuntimeError("source contact-mode manifest has the wrong fixed budget")

    ledger = validate_stage_ledger(root)
    stages = ledger.get("stages", {})
    if set(stages) != set(_REQUIRED_SOURCE_STAGES):
        raise RuntimeError(
            "source contact-mode ledger must contain exactly the three committed "
            "pre-finalization stages"
        )

    audit_artifacts = _stage_artifacts(ledger, "contact_mode_pose_source_audit")
    _require_stage_artifact(root, audit_artifacts, audit_path)
    if set(audit_artifacts) != {str(audit_path.relative_to(root))}:
        raise RuntimeError("source audit stage contains unexpected artifacts")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not isinstance(audit, Mapping)
        or audit.get("complete") is not True
        or canonical_sha256(audit.get("source"))
        != canonical_sha256(manifest.get("source_authentication"))
        or canonical_sha256(audit.get("source_evidence_sha256"))
        != canonical_sha256(manifest.get("source_evidence_sha256"))
    ):
        raise RuntimeError("source contact-mode audit differs from its manifest")

    search = json.loads(search_report_path.read_text(encoding="utf-8"))
    if not isinstance(search, Mapping):
        raise RuntimeError("source contact-mode search report is malformed")
    search = copy.deepcopy(dict(search))
    _validate_search_report(
        search, expected_count=int(policy.search_candidate_count)
    )
    if search.get("source_authentication_id") != manifest.get(
        "source_authentication", {}
    ).get("source_authentication_id"):
        raise RuntimeError("source search report changed its input authentication")
    search_artifacts = _stage_artifacts(ledger, "contact_mode_pose_search")
    _require_stage_artifact(root, search_artifacts, search_report_path)

    configs: dict[int, dict[str, Any]] = {}
    search_traces: dict[int, Path] = {}
    candidate_evidence: list[dict[str, Any]] = []
    expected_search_artifacts = {str(search_report_path.relative_to(root))}
    candidate_roots: set[Path] = set()
    records = tuple(copy.deepcopy(dict(value)) for value in search["records"])
    for rank, record in enumerate(records):
        candidate_id = int(record.get("candidate_id", -1))
        relative = record.get("artifact_directory")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise RuntimeError("source search candidate has an unsafe artifact path")
        candidate_root = (root / relative).resolve()
        allowed_root = (root / "contact_mode_pose_search" / "candidates").resolve()
        if (
            not candidate_root.is_relative_to(allowed_root)
            or candidate_root in candidate_roots
        ):
            raise RuntimeError("source search candidate escaped or duplicated its root")
        candidate_roots.add(candidate_root)
        bundle = authenticate_v14_candidate_artifacts(
            candidate_root,
            expected_candidate_id=candidate_id,
            require_retained_trace=True,
            expected_retain_grasp_success=False,
        )
        _record_matches_bundle(record, bundle)
        config = json.loads(bundle.config_path.read_text(encoding="utf-8"))
        if str(record.get("physical_config_sha256")) != (
            contact_mode_physical_config_sha256(config)
        ):
            raise RuntimeError("source search physical config digest changed")
        assert bundle.trace_path is not None
        with np.load(bundle.trace_path, allow_pickle=False) as trace:
            diagnostics = contact_mode_trace_diagnostics(trace)
        if canonical_sha256(diagnostics) != canonical_sha256(
            record.get("contact_mode_diagnostics")
        ):
            raise RuntimeError("source search trace diagnostics changed")
        for path in bundle.artifact_paths:
            _require_stage_artifact(root, search_artifacts, path)
            expected_search_artifacts.add(str(path.relative_to(root)))
        configs[candidate_id] = config
        search_traces[candidate_id] = bundle.trace_path
        candidate_evidence.append(
            {
                "rank": rank,
                "candidate_id": candidate_id,
                "artifact_directory": str(candidate_root.relative_to(root)),
                "config_file_sha256": file_sha256(bundle.config_path),
                "config_semantic_sha256": bundle.result[
                    "config_semantic_sha256"
                ],
                "result_file_sha256": file_sha256(bundle.result_path),
                "result_semantic_sha256": bundle.result[
                    "result_semantic_sha256"
                ],
                "trace_file_sha256": file_sha256(bundle.trace_path),
                "record_sha256": canonical_sha256(record),
            }
        )
    if set(search_artifacts) != expected_search_artifacts:
        raise RuntimeError("source search ledger has missing or unexpected artifacts")

    catalog_report = json.loads(catalog_report_path.read_text(encoding="utf-8"))
    if not isinstance(catalog_report, Mapping):
        raise RuntimeError("source contact-mode catalog report is malformed")
    catalog_report = copy.deepcopy(dict(catalog_report))
    expected_ids = [
        int(value["candidate_id"])
        for value in records[: int(policy.published_candidate_count)]
    ]
    if (
        catalog_report.get("complete") is not True
        or int(catalog_report.get("published_candidate_count", -1))
        != int(policy.published_candidate_count)
        or [int(value) for value in catalog_report.get("published_candidate_ids", ())]
        != expected_ids
    ):
        raise RuntimeError("source catalog report changed its ranked Top-5")
    catalogs = catalog_report.get("catalogs")
    if not isinstance(catalogs, Mapping) or set(catalogs) != {
        "grasp_pose",
        "manipulation",
    }:
        raise RuntimeError("source catalog report lost its two Viewer catalogs")
    catalog_stage_artifacts = _stage_artifacts(
        ledger, "contact_mode_pose_catalog_target_1"
    )
    _require_stage_artifact(root, catalog_stage_artifacts, catalog_report_path)
    expected_catalog_artifacts = {str(catalog_report_path.relative_to(root))}
    catalog_paths: dict[str, Path] = {}
    reference_traces: dict[int, Path] = {}
    catalog_evidence: list[dict[str, Any]] = []
    records_by_id = {int(value["candidate_id"]): value for value in records}
    for kind in ("grasp_pose", "manipulation"):
        relative = catalogs[kind]
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise RuntimeError("source catalog report contains an unsafe path")
        catalog_path = (root / relative).resolve()
        if not catalog_path.is_relative_to(root):
            raise RuntimeError("source Viewer catalog escaped its workspace")
        authenticated = authenticated_catalog_artifact_paths(catalog_path)
        for path in authenticated:
            _require_stage_artifact(root, catalog_stage_artifacts, path)
            expected_catalog_artifacts.add(str(path.relative_to(root)))
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        trajectories = payload.get("trajectories")
        if not isinstance(trajectories, list) or [
            int(value.get("candidate_id", -1)) for value in trajectories
        ] != expected_ids:
            raise RuntimeError("source Viewer catalog changed its Top-5 order")
        entry_evidence: list[dict[str, Any]] = []
        for entry in trajectories:
            candidate_id = int(entry["candidate_id"])
            record = records_by_id[candidate_id]
            config_path = _catalog_member_path(catalog_path, entry, "resolved_config")
            result_path = _catalog_member_path(catalog_path, entry, "result")
            trace_path = _catalog_member_path(catalog_path, entry, "trace")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if canonical_sha256(config) != canonical_sha256(configs[candidate_id]):
                raise RuntimeError("source catalog changed a Top-5 resolved config")
            if canonical_sha256(_without_video(result.get("summary", {}))) != (
                canonical_sha256(_without_video(record.get("summary", {})))
            ):
                raise RuntimeError("source catalog changed a Top-5 summary")
            trace_comparison = compare_contact_mode_physical_traces(
                search_traces[candidate_id], trace_path
            )
            if kind == "manipulation":
                reference_traces[candidate_id] = trace_path
            entry_evidence.append(
                {
                    "candidate_id": candidate_id,
                    "config_sha256": file_sha256(config_path),
                    "result_sha256": file_sha256(result_path),
                    "trace_sha256": file_sha256(trace_path),
                    "trace_physical_fields_sha256": trace_comparison[
                        "physical_fields_sha256"
                    ],
                }
            )
        catalog_paths[kind] = catalog_path
        catalog_evidence.append(
            {
                "kind": kind,
                "catalog_path": str(catalog_path.relative_to(root)),
                "catalog_sha256": file_sha256(catalog_path),
                "authenticated_artifact_count": len(authenticated),
                "entries": entry_evidence,
            }
        )
    if set(catalog_stage_artifacts) != expected_catalog_artifacts:
        raise RuntimeError("source catalog ledger has missing or unexpected artifacts")
    if set(reference_traces) != set(expected_ids):
        raise RuntimeError("source manipulation catalog lost a Top-5 reference trace")

    ledger_artifact_count = sum(
        len(value.get("artifacts", {})) for value in ledger["stages"].values()
    )
    evidence = {
        "schema_version": RECOVERY_FINALIZER_SCHEMA_VERSION,
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_campaign_input_sha256": declared_input,
        "source_ledger_sha256": file_sha256(ledger_path),
        "source_ledger_semantic_sha256": canonical_sha256(ledger),
        "ledger_artifact_count": ledger_artifact_count,
        "ledger_stages_sha256": canonical_sha256(ledger["stages"]),
        "source_audit_sha256": file_sha256(audit_path),
        "search_report_sha256": file_sha256(search_report_path),
        "catalog_report_sha256": file_sha256(catalog_report_path),
        "candidate_evidence": candidate_evidence,
        "catalog_evidence": catalog_evidence,
    }
    evidence_sha256 = canonical_sha256(evidence)
    return ContactModeRecoverySource(
        root=root,
        manifest=manifest,
        ledger=copy.deepcopy(dict(ledger)),
        search_report=search,
        catalog_report=catalog_report,
        records=records,
        top_records=records[: int(policy.published_candidate_count)],
        configs=configs,
        search_trace_paths=search_traces,
        reference_trace_paths=reference_traces,
        catalog_paths=catalog_paths,
        evidence=evidence,
        evidence_sha256=evidence_sha256,
    )


def authenticate_contact_mode_recovery_source(
    source_workspace: str | Path,
) -> ContactModeRecoverySource:
    """Authenticate the fixed 256-candidate/Top-5 production workspace."""

    return _authenticate_source_workspace(
        source_workspace, policy=PRODUCTION_POLICY
    )


def _assert_current_inputs_match_old_execution(
    current: Mapping[str, Any], source: ContactModeRecoverySource
) -> None:
    for key in (
        "experiment_id",
        "config_sha256",
        "model_sha256",
        "uv_lock_sha256",
        "actual_qpos_source_manifest_sha256",
        "v13_source_catalog_sha256",
        "v13_measured_grasp_report_sha256",
        "v13_static_downsize_report_sha256",
    ):
        if current.get(key) != source.manifest.get(key):
            raise RuntimeError(
                f"current recovery input {key} differs from the old execution"
            )


def _build_recovery_manifest_from_source(
    config_path: Path,
    source: ContactModeRecoverySource,
    *,
    backend_id: str,
) -> dict[str, Any]:
    base = build_contact_preserving_planned_lift_manifest(
        config_path, seed=DEFAULT_SEED
    )
    _assert_current_inputs_match_old_execution(base, source)
    base.pop("campaign_input_sha256", None)
    implementation = Path(__file__).resolve()
    base.update(
        {
            "contact_mode_recovery_finalizer_schema_version": (
                RECOVERY_FINALIZER_SCHEMA_VERSION
            ),
            "campaign_kind": "contact_mode_pose_recovery_finalizer",
            "source_workspace": str(source.root),
            "source_workspace_authentication": source.descriptor(),
            "source_workspace_evidence_sha256": source.evidence_sha256,
            "immutable_old_manifest_sha256": source.evidence[
                "source_manifest_sha256"
            ],
            "immutable_old_campaign_input_sha256": source.manifest[
                "campaign_input_sha256"
            ],
            "recovery_implementation_sha256": file_sha256(implementation),
            "recovery_backend_id": str(backend_id),
            "execution_contract": {
                "old_workspace_written": False,
                "old_manifest_rebuilt_with_current_source": False,
                "old_manifest_is_execution_evidence": True,
                "top5_current_source_full_reset_rerun": True,
                "summary_comparison": "canonical_exact_without_video_metadata",
                "trace_comparison": "all_npz_fields_byte_exact_except_video_frame_steps",
                "difference_policy": "fail_closed_before_publication",
            },
        }
    )
    return {**base, "campaign_input_sha256": canonical_sha256(base)}


def build_contact_mode_recovery_finalizer_manifest(
    config_path: str | Path, source_workspace: str | Path
) -> dict[str, Any]:
    source = authenticate_contact_mode_recovery_source(source_workspace)
    return _build_recovery_manifest_from_source(
        Path(config_path).expanduser().resolve(),
        source,
        backend_id=DEFAULT_BACKEND.backend_id,
    )


def _assert_stage_input(
    workspace: Path, stage: str, stage_input: Mapping[str, Any]
) -> None:
    ledger = validate_stage_ledger(workspace)
    record = ledger.get("stages", {}).get(stage)
    if not isinstance(record, Mapping):
        raise RuntimeError(f"committed recovery stage {stage} disappeared")
    if record.get("stage_input_sha256") != canonical_sha256(dict(stage_input)):
        raise RuntimeError(f"committed recovery stage {stage} input changed")


def _candidate_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _production_candidate_runner(
        payload["config"],
        Path(str(payload["destination"])),
        int(payload["candidate_id"]),
    )
    return copy.deepcopy(dict(result))


def _execute_current_reruns(
    payloads: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    backend: _RecoveryBackend,
) -> tuple[dict[str, Any], ...]:
    if backend is DEFAULT_BACKEND and int(workers) > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(payloads)), mp_context=context
        ) as pool:
            return tuple(pool.map(_candidate_worker, payloads))
    return tuple(
        copy.deepcopy(
            dict(
                backend.candidate_runner(
                    payload["config"],
                    Path(str(payload["destination"])),
                    int(payload["candidate_id"]),
                )
            )
        )
        for payload in payloads
    )


def _compare_rerun_bundle(
    source: ContactModeRecoverySource,
    source_record: Mapping[str, Any],
    bundle: V14CandidateArtifactBundle,
) -> dict[str, Any]:
    candidate_id = int(source_record["candidate_id"])
    if canonical_sha256(bundle.result.get("summary")) != canonical_sha256(
        source_record.get("summary")
    ):
        raise RuntimeError(
            f"current-source rerun changed candidate {candidate_id} summary"
        )
    for key in ("grasp_success", "full_success", "classification"):
        if bundle.result.get(key) != source_record.get(key):
            raise RuntimeError(
                f"current-source rerun changed candidate {candidate_id} {key}"
            )
    assert bundle.trace_path is not None
    trace_comparison = compare_contact_mode_physical_traces(
        source.reference_trace_paths[candidate_id], bundle.trace_path
    )
    with np.load(bundle.trace_path, allow_pickle=False) as trace:
        diagnostics = contact_mode_trace_diagnostics(trace)
    if canonical_sha256(diagnostics) != canonical_sha256(
        source_record.get("contact_mode_diagnostics")
    ):
        raise RuntimeError(
            f"current-source rerun changed candidate {candidate_id} contact diagnostics"
        )
    return {
        "summary_exact": True,
        "status_exact": True,
        "contact_mode_diagnostics_exact": True,
        "trace": trace_comparison,
    }


def _validate_rerun_report(
    payload: Mapping[str, Any],
    source: ContactModeRecoverySource,
    workspace: Path,
) -> None:
    records = payload.get("records")
    expected_ids = [int(value["candidate_id"]) for value in source.top_records]
    if (
        payload.get("complete") is not True
        or payload.get("all_exact") is not True
        or not isinstance(records, list)
        or [int(value.get("candidate_id", -1)) for value in records] != expected_ids
    ):
        raise RuntimeError("recovery Top-5 rerun report is incomplete")
    source_by_id = {
        int(value["candidate_id"]): value for value in source.top_records
    }
    for record in records:
        candidate_id = int(record["candidate_id"])
        relative = record.get("artifact_directory")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise RuntimeError("recovery rerun artifact path is unsafe")
        root = (workspace / relative).resolve()
        if not root.is_relative_to((workspace / "top5_current_source_reruns").resolve()):
            raise RuntimeError("recovery rerun escaped its workspace")
        bundle = authenticate_v14_candidate_artifacts(
            root,
            expected_config=source.configs[candidate_id],
            expected_candidate_id=candidate_id,
            require_retained_trace=True,
            expected_retain_grasp_success=False,
        )
        comparison = _compare_rerun_bundle(
            source, source_by_id[candidate_id], bundle
        )
        if canonical_sha256(comparison) != canonical_sha256(
            record.get("recovery_comparison")
        ):
            raise RuntimeError("committed recovery comparison evidence changed")


def _run_top5_stage(
    source: ContactModeRecoverySource,
    workspace: Path,
    *,
    workers: int,
    backend: _RecoveryBackend,
) -> tuple[dict[str, Any], Path]:
    stage = "contact_mode_recovery_top5_rerun"
    report_path = workspace / "top5_current_source_reruns" / "report.json"
    selected = tuple(source.top_records)
    stage_input = {
        "source_workspace_evidence_sha256": source.evidence_sha256,
        "selected_candidate_records_sha256": canonical_sha256(selected),
        "current_source_sha256": json.loads(
            (workspace / "campaign_manifest.json").read_text(encoding="utf-8")
        )["source_sha256"],
        "backend_id": backend.backend_id,
    }
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        _assert_stage_input(workspace, stage, stage_input)
        _validate_rerun_report(existing, source, workspace)
        return existing, report_path

    payloads = []
    for record in selected:
        candidate_id = int(record["candidate_id"])
        payloads.append(
            {
                "candidate_id": candidate_id,
                "config": copy.deepcopy(source.configs[candidate_id]),
                "destination": str(
                    workspace
                    / "top5_current_source_reruns"
                    / f"candidate_{candidate_id}"
                ),
            }
        )
    raw = _execute_current_reruns(
        payloads, workers=int(workers), backend=backend
    )
    records: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    for source_record, returned, job in zip(selected, raw, payloads, strict=True):
        candidate_id = int(source_record["candidate_id"])
        if int(returned.get("candidate_id", -1)) != candidate_id:
            raise RuntimeError("current-source recovery rerun returned the wrong ID")
        root = Path(job["destination"]).resolve()
        bundle = authenticate_v14_candidate_artifacts(
            root,
            expected_config=source.configs[candidate_id],
            expected_candidate_id=candidate_id,
            require_retained_trace=True,
            expected_retain_grasp_success=False,
        )
        comparison = _compare_rerun_bundle(source, source_record, bundle)
        artifacts.extend(bundle.artifact_paths)
        records.append(
            {
                **copy.deepcopy(bundle.result),
                "artifact_directory": str(root.relative_to(workspace)),
                "source_contact_mode_rank": int(
                    source_record["contact_mode_rank"]
                ),
                "recovery_comparison": comparison,
            }
        )
    payload = {
        "contact_mode_recovery_top5_report_schema_version": 1,
        "complete": True,
        "all_exact": True,
        "source_workspace_evidence_sha256": source.evidence_sha256,
        "candidate_count": len(records),
        "candidate_ids": [int(value["candidate_id"]) for value in records],
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
    _validate_rerun_report(committed, source, workspace)
    return committed, report_path


def _annotate_recovery_catalog(
    path: Path, source: ContactModeRecoverySource, current_source_sha256: str
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("recovery publisher produced a malformed catalog")
    updated = copy.deepcopy(dict(payload))
    updated["selection_policy"] = "immutable_authenticated_source_top5_order"
    updated["contact_mode_recovery_finalizer"] = {
        "schema_version": RECOVERY_FINALIZER_SCHEMA_VERSION,
        "source_workspace_evidence_sha256": source.evidence_sha256,
        "immutable_old_manifest_sha256": source.evidence[
            "source_manifest_sha256"
        ],
        "current_source_sha256": current_source_sha256,
        "top5_current_source_rerun_exact": True,
    }
    write_json(path, updated)


def _validate_recovery_catalog_report(
    report: Mapping[str, Any], source: ContactModeRecoverySource, workspace: Path
) -> None:
    if (
        report.get("complete") is not True
        or report.get("top5_current_source_rerun_exact") is not True
        or [int(value) for value in report.get("published_candidate_ids", ())]
        != [int(value["candidate_id"]) for value in source.top_records]
    ):
        raise RuntimeError("recovery catalog report is incomplete")
    catalogs = report.get("catalogs")
    if not isinstance(catalogs, Mapping) or set(catalogs) != {
        "grasp_pose",
        "manipulation",
    }:
        raise RuntimeError("recovery catalog report lost its two catalogs")
    for relative in catalogs.values():
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise RuntimeError("recovery catalog path is unsafe")
        path = (workspace / relative).resolve()
        if not path.is_relative_to(workspace):
            raise RuntimeError("recovery catalog escaped its workspace")
        authenticated_catalog_artifact_paths(path)


def _run_catalog_stage(
    rerun: Mapping[str, Any],
    rerun_report_path: Path,
    source: ContactModeRecoverySource,
    workspace: Path,
    *,
    backend: _RecoveryBackend,
) -> tuple[dict[str, Any], Path]:
    stage = "contact_mode_recovery_catalog"
    root = workspace / "catalogs" / "recovered_top5"
    report_path = root / "report.json"
    manifest = json.loads(
        (workspace / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    stage_input = {
        "source_workspace_evidence_sha256": source.evidence_sha256,
        "rerun_report_sha256": file_sha256(rerun_report_path),
        "current_source_sha256": manifest["source_sha256"],
        "backend_id": backend.backend_id,
    }
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        _assert_stage_input(workspace, stage, stage_input)
        _validate_recovery_catalog_report(existing, source, workspace)
        return existing, report_path
    records = tuple(copy.deepcopy(dict(value)) for value in rerun["records"])
    catalogs = dict(
        backend.catalog_publisher(
            records,
            workspace,
            root,
            str(source.manifest["experiment_id"]),
        )
    )
    artifacts: list[Path] = []
    for relative in catalogs.values():
        path = (workspace / relative).resolve()
        _annotate_recovery_catalog(path, source, str(manifest["source_sha256"]))
        artifacts.extend(authenticated_catalog_artifact_paths(path))
    payload = {
        "contact_mode_recovery_catalog_report_schema_version": 1,
        "complete": True,
        "source_workspace_evidence_sha256": source.evidence_sha256,
        "top5_current_source_rerun_exact": True,
        "published_candidate_count": len(records),
        "published_candidate_ids": [
            int(value["candidate_id"]) for value in records
        ],
        "catalogs": catalogs,
        "records": [],
    }
    committed = _commit_report(
        workspace,
        stage,
        report_path,
        payload,
        stage_input=stage_input,
        artifacts=tuple(dict.fromkeys(artifacts)),
    )
    _validate_recovery_catalog_report(committed, source, workspace)
    return committed, report_path


def _run_recovery_finalizer(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    source_workspace: str | Path,
    resume: bool,
    workers: int,
    policy: _RecoveryPolicy,
    backend: _RecoveryBackend,
) -> dict[str, Any]:
    if isinstance(workers, bool) or int(workers) <= 0:
        raise ValueError("workers must be positive")
    config_file = Path(config_path).expanduser().resolve()
    workspace = Path(output_dir).expanduser().resolve()
    source_root = Path(source_workspace).expanduser().resolve()
    if workspace == source_root or workspace.is_relative_to(source_root):
        raise ValueError("recovery workspace must be outside the immutable source")
    definition = resolve_experiment(load_config(config_file))
    if getattr(definition, "contact_preserving_planned_lift_campaign", None) is None:
        raise ValueError("contact-mode recovery requires the registered v14 experiment")

    source = _authenticate_source_workspace(source_root, policy=policy)
    manifest = _build_recovery_manifest_from_source(
        config_file, source, backend_id=backend.backend_id
    )
    initialize_or_resume_campaign(workspace, manifest, resume=bool(resume))

    audit_stage = "contact_mode_recovery_source_audit"
    audit_path = workspace / "source_workspace_audit.json"
    audit_input = {
        "source_workspace_evidence_sha256": source.evidence_sha256,
        "immutable_old_manifest_sha256": source.evidence[
            "source_manifest_sha256"
        ],
    }
    audit_payload = {
        "contact_mode_recovery_source_audit_schema_version": 1,
        "complete": True,
        "source_workspace": source.descriptor(),
        "complete_authenticated_evidence": source.evidence,
        "records": [],
    }
    existing_audit = _load_committed_report(
        workspace, audit_stage, audit_path
    )
    if existing_audit is None:
        _commit_report(
            workspace,
            audit_stage,
            audit_path,
            audit_payload,
            stage_input=audit_input,
        )
    else:
        _assert_stage_input(workspace, audit_stage, audit_input)
        if canonical_sha256(existing_audit) != canonical_sha256(audit_payload):
            raise RuntimeError("committed recovery source audit changed")

    rerun, rerun_path = _run_top5_stage(
        source, workspace, workers=int(workers), backend=backend
    )
    catalog, catalog_path = _run_catalog_stage(
        rerun, rerun_path, source, workspace, backend=backend
    )

    # Reauthenticate both sides immediately before final commitment.  This
    # detects source tampering or a code change during a long Top-5/video run.
    final_source = _authenticate_source_workspace(source_root, policy=policy)
    if final_source.evidence_sha256 != source.evidence_sha256:
        raise RuntimeError("immutable source workspace changed during recovery")
    final_manifest = _build_recovery_manifest_from_source(
        config_file, final_source, backend_id=backend.backend_id
    )
    if canonical_sha256(final_manifest) != canonical_sha256(manifest):
        raise RuntimeError("current source or recovery inputs changed during execution")

    full_count = sum(
        bool(value.get("full_success", False)) for value in rerun["records"]
    )
    grasp_count = sum(
        bool(value.get("grasp_success", False)) for value in rerun["records"]
    )
    result = {
        "contact_mode_recovery_finalizer_result_schema_version": (
            RECOVERY_RESULT_SCHEMA_VERSION
        ),
        "complete": True,
        "experiment_id": definition.experiment_id,
        "source_workspace": str(source_root),
        "source_workspace_evidence_sha256": source.evidence_sha256,
        "immutable_old_manifest_sha256": source.evidence[
            "source_manifest_sha256"
        ],
        "immutable_old_campaign_input_sha256": source.manifest[
            "campaign_input_sha256"
        ],
        "current_source_sha256": manifest["source_sha256"],
        "top5_current_source_rerun_exact": True,
        "published_candidate_ids": copy.deepcopy(
            list(catalog["published_candidate_ids"])
        ),
        "grasp_success_count": grasp_count,
        "full_success_count": full_count,
        "catalogs": copy.deepcopy(dict(catalog["catalogs"])),
        "recovered_without_modifying_source_workspace": True,
        "fixed_mass_geometry_ablation": True,
        "exit_code": 0 if full_count else 2,
        "stop_reason": (
            "recovered_top5_contains_full_success"
            if full_count
            else "recovered_completed_budget_has_no_full_success"
        ),
    }
    stage = "contact_mode_recovery_result"
    result_path = workspace / "contact_mode_recovery_finalizer_result.json"
    stage_input = {
        "source_workspace_evidence_sha256": source.evidence_sha256,
        "rerun_report_sha256": file_sha256(rerun_path),
        "catalog_report_sha256": file_sha256(catalog_path),
        "current_source_sha256": manifest["source_sha256"],
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
            raise RuntimeError("committed recovery final result changed")
        result = existing
    validate_stage_ledger(workspace)
    return result


def run_contact_mode_recovery_finalizer(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    source_workspace: str | Path,
    resume: bool,
    workers: int = 1,
) -> dict[str, Any]:
    """Recover the fixed production campaign into a new authenticated root."""

    return _run_recovery_finalizer(
        config_path,
        output_dir,
        source_workspace=source_workspace,
        resume=resume,
        workers=workers,
        policy=PRODUCTION_POLICY,
        backend=DEFAULT_BACKEND,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed finalization of an immutable contact-mode campaign "
            "whose committed stages predate the current source-tree hash."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-workspace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_contact_mode_recovery_finalizer(
        args.config,
        args.output_dir,
        source_workspace=args.source_workspace,
        resume=bool(args.resume),
        workers=int(args.workers),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return int(result["exit_code"])


if __name__ == "__main__":  # pragma: no cover - exercised as a module CLI.
    raise SystemExit(main())


__all__ = [
    "ContactModeRecoverySource",
    "authenticate_contact_mode_recovery_source",
    "build_contact_mode_recovery_finalizer_manifest",
    "compare_contact_mode_physical_traces",
    "run_contact_mode_recovery_finalizer",
]
