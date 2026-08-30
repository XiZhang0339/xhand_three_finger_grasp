"""Third-stage event-aware rescue for a completed schema-v14 rescue campaign.

This runner is deliberately separate from both the production v14 campaign
and the two-stage refinement/time-warp rescue.  Its input is an immutable,
completed rescue-v2 workspace.  Before creating any output it authenticates
the source ledger, every phase-two candidate, the contact-first Viewer catalog
and the retained full-reset trace selected as the event source.

The physics budget is fixed and exhaustive:

* 512 event-aware exploration candidates;
* 512 local-refinement candidates around contact-first exploration results;
* every candidate starts from the initial no-contact state; and
* every catalog entry is independently rerun from that state before MP4
  publication.

Candidate synthesis is provided by :mod:`contact_preserving_event_rescue` and
loaded lazily.  Keeping that numerical module independent lets its event
detector and perturbation model evolve without weakening this file's artifact,
resume, or provenance boundaries.  No CLI is registered here; callers invoke
``run_contact_preserving_event_rescue_campaign`` explicitly until the third
stage has production evidence.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
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
from .contact_preserving_lift_rescue_campaign import (
    _commit_report,
    _execute_candidate_jobs,
    _load_committed_report,
    _materialize_rescue_catalog_traces,
    _publish_rescue_viewer_catalogs,
    _rank_phase_one,
    _validate_phase_report,
)
from .contact_preserving_planned_lift_campaign import (
    build_contact_preserving_planned_lift_manifest,
)
from .contact_preserving_time_warp import authenticate_time_warp_job


EVENT_CAMPAIGN_SCHEMA_VERSION = 1
EVENT_SOURCE_SCHEMA_VERSION = 1
EVENT_PHASE_REPORT_SCHEMA_VERSION = 1
EVENT_RESULT_SCHEMA_VERSION = 1
EVENT_EXPLORATION_CANDIDATE_COUNT = 512
EVENT_REFINEMENT_CANDIDATE_COUNT = 512
EVENT_REFINEMENT_CENTER_COUNT = 8
DEFAULT_SEED = 20260821
_SOURCE_CATALOG_RELATIVE = Path("catalogs/target_1/manipulation/catalog.json")
_SOURCE_TIME_WARP_REPORT_RELATIVE = Path("time_warp_rescue/report.json")
_SOURCE_RESULT_RELATIVE = Path("rescue_result_target_1.json")


@dataclass(frozen=True, slots=True)
class AuthenticatedEventRescueSource:
    """Immutable phase-two parent plus its retained full-reset trace."""

    root: Path
    parent_candidate_id: int
    parent_config: dict[str, Any]
    parent_summary: dict[str, Any]
    parent_record: dict[str, Any]
    phase_two_bundle: V14CandidateArtifactBundle
    rerun_bundle: V14CandidateArtifactBundle
    catalog_path: Path
    phase_two_report_path: Path
    result_path: Path
    manifest_path: Path
    ledger_path: Path
    source_trace_sha256: str
    source_parent_peak_jerk_m_s3: float
    phase_two_candidate_count: int
    phase_two_candidate_set_sha256: str
    source_authentication_id: str

    @property
    def trace_path(self) -> Path:
        if self.rerun_bundle.trace_path is None:
            raise RuntimeError("event rescue source lost its retained trace")
        return self.rerun_bundle.trace_path

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.manifest_path,
                    self.ledger_path,
                    self.phase_two_report_path,
                    self.result_path,
                    self.catalog_path,
                    *self.phase_two_bundle.artifact_paths,
                    *self.rerun_bundle.artifact_paths,
                )
            )
        )

    def descriptor(self) -> dict[str, Any]:
        payload = {
            "event_rescue_source_schema_version": EVENT_SOURCE_SCHEMA_VERSION,
            "source_root": str(self.root),
            "source_campaign_input_sha256": json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )["campaign_input_sha256"],
            "source_manifest_sha256": file_sha256(self.manifest_path),
            "source_ledger_sha256": file_sha256(self.ledger_path),
            "source_result_sha256": file_sha256(self.result_path),
            "phase_two_report_sha256": file_sha256(self.phase_two_report_path),
            "source_catalog_sha256": file_sha256(self.catalog_path),
            "parent_candidate_id": self.parent_candidate_id,
            "parent_config_semantic_sha256": canonical_sha256(self.parent_config),
            "parent_result_semantic_sha256": self.rerun_bundle.result[
                "result_semantic_sha256"
            ],
            "source_trace_sha256": self.source_trace_sha256,
            "source_parent_peak_jerk_m_s3": self.source_parent_peak_jerk_m_s3,
            "source_parent_selection": (
                "retained_catalog_jerk_only_minimum_peak_jerk"
            ),
            "phase_two_candidate_count": self.phase_two_candidate_count,
            "phase_two_candidate_set_sha256": self.phase_two_candidate_set_sha256,
            "read_only": True,
        }
        observed = canonical_sha256(payload)
        if observed != self.source_authentication_id:
            raise RuntimeError("event-rescue source descriptor changed after authentication")
        return {**payload, "source_authentication_id": observed}


@dataclass(frozen=True, slots=True)
class _EventRescueApi:
    descriptor_type: type
    descriptor_builder: Callable[..., Any]
    descriptor_authenticator: Callable[..., Any]
    exploration_builder: Callable[..., Sequence[Mapping[str, Any]]]
    refinement_builder: Callable[..., Sequence[Mapping[str, Any]]]
    job_authenticator: Callable[..., Any]
    budget_type: type


def _event_rescue_api() -> _EventRescueApi:
    """Load the numerical API only when a third-stage run actually starts."""

    from . import contact_preserving_event_rescue as event

    required = {
        "EventRescueDescriptor": "descriptor_type",
        "build_event_rescue_descriptor": "descriptor_builder",
        "authenticate_event_rescue_descriptor": "descriptor_authenticator",
        "build_event_rescue_jobs": "exploration_builder",
        "build_event_rescue_refinement_jobs": "refinement_builder",
        "authenticate_event_rescue_job": "job_authenticator",
        "EventRescueBudget": "budget_type",
    }
    missing = [name for name in required if not hasattr(event, name)]
    if missing:
        raise RuntimeError(
            "event-rescue numerical API is incomplete: " + ", ".join(missing)
        )
    return _EventRescueApi(
        **{field: getattr(event, name) for name, field in required.items()}
    )


def _mapping_descriptor(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    for converter_name in ("as_mapping", "to_dict"):
        converter = getattr(value, converter_name, None)
        if callable(converter):
            payload = converter()
            if isinstance(payload, Mapping):
                return copy.deepcopy(dict(payload))
    raise RuntimeError("event-rescue descriptor has no authenticated mapping form")


def _reconstruct_time_warp_job(
    record: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    metadata = record.get("rescue_job")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("phase-two candidate lost time-warp job metadata")
    return {
        "time_warp_job_schema_version": 1,
        "candidate_id": int(record["candidate_id"]),
        "candidate_sha256": str(metadata["candidate_sha256"]),
        "config_sha256": canonical_sha256(config),
        "parent_candidate_id": int(metadata["parent_candidate_id"]),
        "parent_rank": int(metadata["parent_rank"]),
        "local_index": int(metadata["category_index"]),
        "job_sequence_index": int(metadata["job_sequence_index"]),
        "config": copy.deepcopy(dict(config)),
        "job_metadata": copy.deepcopy(dict(metadata)),
    }


def _record_matches_bundle(
    record: Mapping[str, Any], bundle: V14CandidateArtifactBundle
) -> None:
    for key, value in bundle.result.items():
        if key not in record or canonical_sha256(record[key]) != canonical_sha256(
            value
        ):
            raise RuntimeError(
                f"phase-two report changed candidate {bundle.candidate_id} field {key}"
            )


def _catalog_trace_matches_source_dynamics(
    source_trace: Path, catalog_trace: Path
) -> bool:
    """Compare every dynamics field while allowing rendered frame indices.

    The catalog renderer reuses the same full-reset simulation but fills the
    previously empty ``video_frame_steps`` field.  Consequently the NPZ byte
    hash is intentionally different even though all 150 physical trace arrays
    are equal.  Event detection is bound to the unrendered source trace; this
    check prevents the video path from masking any dynamics difference.
    """

    with np.load(source_trace, allow_pickle=False) as source, np.load(
        catalog_trace, allow_pickle=False
    ) as rendered:
        if set(source.files) != set(rendered.files):
            return False
        for name in source.files:
            if name == "video_frame_steps":
                if source[name].size != 0 or rendered[name].ndim != 1:
                    return False
                continue
            left = source[name]
            right = rendered[name]
            if left.shape != right.shape or left.dtype != right.dtype:
                return False
            if left.dtype.kind in "fc":
                equal = np.array_equal(left, right, equal_nan=True)
            else:
                equal = np.array_equal(left, right)
            if not equal:
                return False
    return True


def authenticate_completed_event_rescue_source(
    source_campaign: str | Path,
) -> AuthenticatedEventRescueSource:
    """Authenticate a completed zero-success rescue-v2 without writing to it."""

    root = Path(source_campaign).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = root / "campaign_manifest.json"
    ledger_path = root / "stage_ledger.json"
    report_path = root / _SOURCE_TIME_WARP_REPORT_RELATIVE
    result_path = root / _SOURCE_RESULT_RELATIVE
    catalog_path = root / _SOURCE_CATALOG_RELATIVE
    if not all(
        value.is_file()
        for value in (
            manifest_path,
            ledger_path,
            report_path,
            result_path,
            catalog_path,
        )
    ):
        raise RuntimeError("event-rescue source lost required completed evidence")

    ledger = validate_stage_ledger(root)
    required_stages = {
        "refinement_rescue",
        "time_warp_rescue",
        "rescue_catalog_target_1",
        "rescue_result_target_1",
    }
    if not required_stages.issubset(ledger.get("stages", {})):
        raise RuntimeError("event-rescue source has not completed both rescue phases")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    phase_two = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        manifest.get("campaign_kind") != "contact_preserving_post_campaign_rescue"
        or result.get("complete") is not True
        or result.get("time_warp_triggered") is not True
        or int(result.get("time_warp_candidate_count", -1)) != 256
        or int(result.get("time_warp_full_success_count", -1)) != 0
        or int(result.get("full_success_count", -1)) != 0
    ):
        raise RuntimeError("event-rescue source is not a completed zero-success rescue")
    _validate_phase_report(
        phase_two,
        expected_candidate_count=256,
        label="event-rescue phase-two source",
    )
    records = phase_two["records"]
    authenticated_records: list[dict[str, Any]] = []
    bundles: dict[int, V14CandidateArtifactBundle] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise RuntimeError("phase-two source record is not a mapping")
        record = copy.deepcopy(dict(raw))
        candidate_id = int(record.get("candidate_id", -1))
        relative = record.get("artifact_directory")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise RuntimeError("phase-two source candidate path is unsafe")
        candidate_root = (root / relative).resolve()
        expected_root = (root / "time_warp_rescue" / "candidates").resolve()
        if not candidate_root.is_relative_to(expected_root):
            raise RuntimeError("phase-two source candidate escaped its workspace")
        bundle = authenticate_v14_candidate_artifacts(
            candidate_root,
            expected_candidate_id=candidate_id,
            expected_retain_grasp_success=False,
        )
        _record_matches_bundle(record, bundle)
        config = json.loads(bundle.config_path.read_text(encoding="utf-8"))
        authenticate_time_warp_job(_reconstruct_time_warp_job(record, config))
        bundles[candidate_id] = bundle
        authenticated_records.append({**record, "config": config})

    ranked = tuple(_rank_phase_one(authenticated_records))
    if {int(value["candidate_id"]) for value in ranked} != set(bundles):
        raise RuntimeError("contact-first source ranking lost phase-two candidates")
    contact_first = ranked[0]
    contact_first_id = int(contact_first["candidate_id"])

    # The committed catalog contains the production full-reset rerun and MP4.
    # Authenticate all catalog artifacts, then require its best-attempt alias
    # to agree with a fresh contact-first ranking of all 256 phase-two records.
    authenticated_catalog_artifact_paths(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    aliases = catalog.get("aliases")
    trajectories = catalog.get("trajectories")
    if not isinstance(aliases, Mapping) or not isinstance(trajectories, list):
        raise RuntimeError("event-rescue source catalog is malformed")
    alias = aliases.get("best_attempt")
    contact_first_catalog = next(
        (
            value
            for value in trajectories
            if isinstance(value, Mapping) and value.get("trajectory_id") == alias
        ),
        None,
    )
    if (
        contact_first_catalog is None
        or int(contact_first_catalog.get("candidate_id", -1)) != contact_first_id
    ):
        raise RuntimeError("source catalog best_attempt is not contact-first phase two")

    # Contact-first ranking remains the publication order, but event rescue is
    # specifically a jerk-only third stage.  Among the five authenticated
    # retained catalog reruns, select the lowest-jerk candidate whose *only*
    # failed check is the jerk threshold.  This avoids using the contact-first
    # 12.07 m/s^3 item when an equally contact-safe 7.68 m/s^3 retained trace
    # is already available.  Summary-only candidates are never treated as
    # event sources because they cannot authenticate event timing.
    records_by_id = {
        int(value["candidate_id"]): value for value in authenticated_records
    }
    retained_options: list[
        tuple[float, int, Mapping[str, Any], dict[str, Any], V14CandidateArtifactBundle]
    ] = []
    for trajectory in trajectories:
        if not isinstance(trajectory, Mapping):
            raise RuntimeError("source catalog trajectory is not a mapping")
        candidate_id = int(trajectory.get("candidate_id", -1))
        record = records_by_id.get(candidate_id)
        if record is None:
            raise RuntimeError("source catalog contains a non-phase-two candidate")
        rerun_root = (
            root
            / "catalog_source_reruns"
            / "rescue_target_1"
            / f"candidate_{candidate_id}"
        ).resolve()
        rerun_option = authenticate_v14_candidate_artifacts(
            rerun_root,
            expected_config=record["config"],
            expected_candidate_id=candidate_id,
            require_retained_trace=True,
            expected_retain_grasp_success=False,
        )
        if canonical_sha256(rerun_option.result["summary"]) != canonical_sha256(
            record["summary"]
        ):
            raise RuntimeError("source full-reset rerun changed phase-two summary")
        catalog_artifacts = trajectory.get("artifacts")
        if not isinstance(catalog_artifacts, Mapping):
            raise RuntimeError("source catalog candidate lost artifact metadata")
        catalog_trace = (
            catalog_path.parent / str(catalog_artifacts.get("trace"))
        ).resolve()
        catalog_trace_sha = catalog_artifacts.get("sha256", {}).get("trace")
        if (
            not catalog_trace.is_relative_to(catalog_path.parent)
            or not catalog_trace.is_file()
            or file_sha256(catalog_trace) != catalog_trace_sha
            or not _catalog_trace_matches_source_dynamics(
                rerun_option.trace_path, catalog_trace
            )
        ):
            raise RuntimeError(
                "source catalog dynamics differ from its full-reset rerun"
            )
        summary = rerun_option.result["summary"]
        failed = summary.get("failed_checks")
        motion = summary.get("metrics", {}).get("motion_smoothness", {})
        jerk = motion.get("operation_peak_abs_filtered_jerk_m_s3")
        if failed == ["smooth_motion_jerk_within_limit"] and isinstance(
            jerk, (int, float)
        ) and np.isfinite(float(jerk)):
            retained_options.append(
                (float(jerk), candidate_id, trajectory, record, rerun_option)
            )
    if not retained_options:
        raise RuntimeError("source catalog has no retained jerk-only event parent")
    parent_jerk, best_id, selected, best, rerun = min(
        retained_options, key=lambda value: (value[0], value[1])
    )
    trace_sha = file_sha256(rerun.trace_path)

    candidate_set_sha = canonical_sha256(
        [
            {
                "candidate_id": int(value["candidate_id"]),
                "config_semantic_sha256": value["config_semantic_sha256"],
                "result_semantic_sha256": value["result_semantic_sha256"],
            }
            for value in sorted(
                authenticated_records, key=lambda item: int(item["candidate_id"])
            )
        ]
    )

    source_payload = {
        "event_rescue_source_schema_version": EVENT_SOURCE_SCHEMA_VERSION,
        "source_root": str(root),
        "source_campaign_input_sha256": manifest.get("campaign_input_sha256"),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_ledger_sha256": file_sha256(ledger_path),
        "phase_two_report_sha256": file_sha256(report_path),
        "source_result_sha256": file_sha256(result_path),
        "source_catalog_sha256": file_sha256(catalog_path),
        "parent_candidate_id": best_id,
        "parent_config_semantic_sha256": canonical_sha256(best["config"]),
        "parent_result_semantic_sha256": rerun.result["result_semantic_sha256"],
        "source_trace_sha256": trace_sha,
        "source_parent_peak_jerk_m_s3": parent_jerk,
        "source_parent_selection": (
            "retained_catalog_jerk_only_minimum_peak_jerk"
        ),
        "phase_two_candidate_count": len(authenticated_records),
        "phase_two_candidate_set_sha256": candidate_set_sha,
        "read_only": True,
    }
    source_id = canonical_sha256(source_payload)
    return AuthenticatedEventRescueSource(
        root=root,
        parent_candidate_id=best_id,
        parent_config=copy.deepcopy(best["config"]),
        parent_summary=copy.deepcopy(best["summary"]),
        parent_record=copy.deepcopy(best),
        phase_two_bundle=bundles[best_id],
        rerun_bundle=rerun,
        catalog_path=catalog_path,
        phase_two_report_path=report_path,
        result_path=result_path,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        source_trace_sha256=trace_sha,
        source_parent_peak_jerk_m_s3=parent_jerk,
        phase_two_candidate_count=len(authenticated_records),
        phase_two_candidate_set_sha256=candidate_set_sha,
        source_authentication_id=source_id,
    )


def _build_event_descriptor(source: AuthenticatedEventRescueSource, api: _EventRescueApi):
    with np.load(source.trace_path, allow_pickle=False) as trace:
        built = api.descriptor_builder(
            source.parent_config,
            trace,
            source_trace_sha256=source.source_trace_sha256,
            source_candidate_id=source.parent_candidate_id,
        )
        descriptor = api.descriptor_type.from_mapping(_mapping_descriptor(built))
        api.descriptor_authenticator(
            descriptor,
            source.parent_config,
            trace,
            source_trace_sha256=source.source_trace_sha256,
            source_candidate_id=source.parent_candidate_id,
        )
    return descriptor


def _normalize_event_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    descriptor: Any,
    api: _EventRescueApi,
    expected_count: int,
    stage: str,
) -> tuple[dict[str, Any], ...]:
    if len(jobs) != int(expected_count):
        raise RuntimeError(f"event {stage} did not build exactly {expected_count} jobs")
    result: list[dict[str, Any]] = []
    identifiers: list[int] = []
    hashes: list[str] = []
    for fallback_index, raw in enumerate(jobs):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"event {stage} job is not a mapping")
        job = copy.deepcopy(dict(raw))
        api.job_authenticator(job, descriptor)
        candidate_id = int(job.get("candidate_id", -1))
        config = job.get("config")
        parameters = job.get("parameters")
        if candidate_id < 0 or not isinstance(config, Mapping) or not isinstance(
            parameters, Mapping
        ):
            raise RuntimeError(f"event {stage} job lost identity, config, or parameters")
        payload_sha = job.get("candidate_payload_sha256")
        if not isinstance(payload_sha, str) or len(payload_sha) != 64:
            raise RuntimeError(f"event {stage} job lost its candidate payload SHA-256")
        local_index = int(job.get("local_index", fallback_index))
        metadata = {
            key: copy.deepcopy(value)
            for key, value in job.items()
            if key != "config"
        }
        metadata.update(
            {
                "job_sequence_index": local_index,
                "full_reset_required": True,
                "runner_stage": stage,
            }
        )
        result.append(
            {
                **job,
                "candidate_sha256": payload_sha,
                "job_sequence_index": local_index,
                "job_metadata": metadata,
            }
        )
        identifiers.append(candidate_id)
        hashes.append(payload_sha)
    for label, values in (
        ("candidate IDs", identifiers),
        ("candidate payload hashes", hashes),
    ):
        if len(set(values)) != expected_count:
            raise RuntimeError(f"event {stage} does not contain {expected_count} unique {label}")
    result.sort(key=lambda value: (value["job_sequence_index"], value["candidate_id"]))
    if [int(value["job_sequence_index"]) for value in result] != list(
        range(expected_count)
    ):
        raise RuntimeError(f"event {stage} job sequence is not contiguous")
    return tuple(result)


def _rank_event_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Use the established fail-closed contact-first rescue order."""

    ranked = tuple(_rank_phase_one(records))
    if {int(value["candidate_id"]) for value in ranked} != {
        int(value["candidate_id"]) for value in records
    }:
        raise RuntimeError("event rescue contact-first ranking lost candidates")
    return ranked


def _select_refinement_centers(
    exploration_records: Sequence[Mapping[str, Any]],
    *,
    count: int = EVENT_REFINEMENT_CENTER_COUNT,
) -> tuple[dict[str, Any], ...]:
    """Select unique event parameter centers in contact-first order."""

    centers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in _rank_event_records(exploration_records):
        metadata = record.get("rescue_job")
        parameters = metadata.get("parameters") if isinstance(metadata, Mapping) else None
        if not isinstance(parameters, Mapping):
            raise RuntimeError("event exploration record lost its parameter evidence")
        payload = copy.deepcopy(dict(parameters))
        digest = canonical_sha256(payload)
        if digest in seen:
            continue
        seen.add(digest)
        centers.append(payload)
        if len(centers) == int(count):
            break
    if len(centers) != int(count):
        raise RuntimeError("event exploration did not retain eight unique local centers")
    return tuple(centers)


def _event_manifest(
    config_path: Path,
    source: AuthenticatedEventRescueSource,
    *,
    seed: int,
) -> dict[str, Any]:
    base = build_contact_preserving_planned_lift_manifest(config_path, seed=seed)
    base.pop("campaign_input_sha256", None)
    source_descriptor = source.descriptor()
    evidence = {
        str(path.relative_to(source.root)): file_sha256(path)
        for path in source.artifact_paths
        if path.is_relative_to(source.root)
    }
    base.update(
        {
            "contact_preserving_event_campaign_schema_version": (
                EVENT_CAMPAIGN_SCHEMA_VERSION
            ),
            "campaign_kind": "contact_preserving_event_aware_rescue",
            "source_rescue_campaign_path": str(source.root),
            "source_rescue_authentication": source_descriptor,
            "source_rescue_evidence_sha256": dict(sorted(evidence.items())),
            "event_rescue_budget": {
                "source_parent_count": 1,
                "source_parent_selection": (
                    "retained_catalog_jerk_only_minimum_peak_jerk"
                ),
                "exploration_candidate_count": EVENT_EXPLORATION_CANDIDATE_COUNT,
                "local_refinement_candidate_count": (
                    EVENT_REFINEMENT_CANDIDATE_COUNT
                ),
                "local_refinement_center_count": EVENT_REFINEMENT_CENTER_COUNT,
                "candidate_execution": "fresh_full_reset_free_dynamics",
                "catalog_execution": "independent_fresh_full_reset_rerun",
                "ranking": "full_success_then_contact_first_then_smoothness",
            },
        }
    )
    return {**base, "campaign_input_sha256": canonical_sha256(base)}


def build_contact_preserving_event_rescue_manifest(
    config_path: str | Path,
    source_rescue_campaign: str | Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    source = authenticate_completed_event_rescue_source(source_rescue_campaign)
    api = _event_rescue_api()
    descriptor = _build_event_descriptor(source, api)
    return _bind_descriptor_to_manifest(
        _event_manifest(
            Path(config_path).expanduser().resolve(), source, seed=int(seed)
        ),
        _mapping_descriptor(descriptor),
    )


def _bind_descriptor_to_manifest(
    manifest: Mapping[str, Any], descriptor_mapping: Mapping[str, Any]
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(manifest))
    payload.pop("campaign_input_sha256", None)
    payload["event_descriptor"] = copy.deepcopy(dict(descriptor_mapping))
    return {**payload, "campaign_input_sha256": canonical_sha256(payload)}


def _phase_payload(
    *,
    stage: str,
    records: Sequence[Mapping[str, Any]],
    descriptor_mapping: Mapping[str, Any],
    source_authentication_id: str,
    centers: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "event_rescue_phase_report_schema_version": EVENT_PHASE_REPORT_SCHEMA_VERSION,
        "stage": stage,
        "complete": True,
        "declared_candidate_count": len(records),
        "candidate_count": len(records),
        "full_success_count": sum(
            bool(value.get("full_success", False)) for value in records
        ),
        "source_authentication_id": source_authentication_id,
        "event_descriptor": copy.deepcopy(dict(descriptor_mapping)),
        "refinement_centers": [copy.deepcopy(dict(value)) for value in centers],
        "records": [copy.deepcopy(dict(value)) for value in records],
    }


def _catalog_stage(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    experiment_id: str,
    target_success_count: int,
    exploration_report_path: Path,
    refinement_report_path: Path,
) -> dict[str, Any]:
    stage = f"event_catalog_target_{target_success_count}"
    root = workspace / "catalogs" / f"target_{target_success_count}"
    report_path = root / "report.json"
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        return existing
    ranked = _rank_event_records(records)
    materialized = _materialize_rescue_catalog_traces(
        ranked, workspace, target_success_count=target_success_count
    )
    catalogs = _publish_rescue_viewer_catalogs(
        materialized,
        workspace,
        root,
        experiment_id=experiment_id,
    )
    artifacts: list[Path] = []
    for relative in catalogs.values():
        artifacts.extend(authenticated_catalog_artifact_paths(workspace / relative))
    full_count = sum(bool(value.get("full_success", False)) for value in ranked)
    payload = {
        "event_rescue_catalog_report_schema_version": 1,
        "complete": True,
        "target_success_count": int(target_success_count),
        "full_success_count": full_count,
        "target_reached": full_count >= int(target_success_count),
        "catalogs": catalogs,
        "selection_policy": "contact_first_event_rescue_top_five",
        "video_policy": "fresh_full_reset_rerun_ffprobe_and_full_decode",
        "records": [],
    }
    return _commit_report(
        workspace,
        stage,
        report_path,
        payload,
        stage_input={
            "exploration_report_sha256": file_sha256(exploration_report_path),
            "refinement_report_sha256": file_sha256(refinement_report_path),
            "target_success_count": int(target_success_count),
        },
        artifacts=tuple(dict.fromkeys(artifacts)),
    )


def run_contact_preserving_event_rescue_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    source_rescue_campaign: str | Path,
    resume: bool,
    target_success_count: int,
    workers: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run the fixed 512 + 512 event-aware third-stage rescue."""

    if int(workers) <= 0:
        raise ValueError("workers must be positive")
    if int(seed) != DEFAULT_SEED:
        raise ValueError("schema-v14 event rescue seed must be 20260821")
    if int(target_success_count) not in (1, 5):
        raise ValueError("target_success_count must be one or five")
    config_file = Path(config_path).expanduser().resolve()
    workspace = Path(output_dir).expanduser().resolve()
    source_root = Path(source_rescue_campaign).expanduser().resolve()
    if workspace == source_root or workspace.is_relative_to(source_root):
        raise ValueError("event-rescue workspace must be outside its immutable source")
    definition = resolve_experiment(load_config(config_file))
    if getattr(definition, "contact_preserving_planned_lift_campaign", None) is None:
        raise ValueError("event-rescue runner requires the registered v14 experiment")

    # Authenticate all source evidence before output creation, then bind the
    # retained trace into both the source descriptor and campaign manifest.
    source = authenticate_completed_event_rescue_source(source_root)
    api = _event_rescue_api()
    descriptor = _build_event_descriptor(source, api)
    descriptor_mapping = _mapping_descriptor(descriptor)
    manifest = _bind_descriptor_to_manifest(
        _event_manifest(config_file, source, seed=int(seed)), descriptor_mapping
    )
    initialize_or_resume_campaign(workspace, manifest, resume=bool(resume))

    audit_path = workspace / "event_source_audit.json"
    audit = _load_committed_report(workspace, "event_source_audit", audit_path)
    audit_payload = {
        "event_source_audit_schema_version": EVENT_SOURCE_SCHEMA_VERSION,
        "complete": True,
        "source": source.descriptor(),
        "event_descriptor": descriptor_mapping,
        "evidence_sha256": {
            str(path.relative_to(source.root)): file_sha256(path)
            for path in source.artifact_paths
            if path.is_relative_to(source.root)
        },
        "records": [],
    }
    if audit is None:
        audit = _commit_report(
            workspace,
            "event_source_audit",
            audit_path,
            audit_payload,
            stage_input={
                "source_authentication_id": source.source_authentication_id,
                "event_descriptor": descriptor_mapping,
            },
        )
    elif canonical_sha256(audit) != canonical_sha256(audit_payload):
        raise RuntimeError("event-rescue source changed on resume")

    exploration_path = workspace / "event_exploration" / "report.json"
    exploration = _load_committed_report(
        workspace, "event_exploration", exploration_path
    )
    if exploration is None:
        budget = api.budget_type(
            stage="exploration",
            total_candidate_count=EVENT_EXPLORATION_CANDIDATE_COUNT,
            seed=int(seed),
        )
        raw_jobs = api.exploration_builder(
            source.parent_config,
            descriptor,
            budget=budget,
            validate_configs=True,
        )
        jobs = _normalize_event_jobs(
            raw_jobs,
            descriptor=descriptor,
            api=api,
            expected_count=EVENT_EXPLORATION_CANDIDATE_COUNT,
            stage="exploration",
        )
        records, artifacts = _execute_candidate_jobs(
            jobs,
            workspace,
            phase_name="event_exploration/candidates",
            workers=int(workers),
            global_rank_offset=0,
        )
        ranked = _rank_event_records(records)
        exploration = _commit_report(
            workspace,
            "event_exploration",
            exploration_path,
            _phase_payload(
                stage="exploration",
                records=ranked,
                descriptor_mapping=descriptor_mapping,
                source_authentication_id=source.source_authentication_id,
            ),
            stage_input={
                "source_audit_sha256": file_sha256(audit_path),
                "candidate_count": EVENT_EXPLORATION_CANDIDATE_COUNT,
                "jobs_sha256": canonical_sha256(jobs),
            },
            artifacts=artifacts,
        )
    _validate_phase_report(
        exploration,
        expected_candidate_count=EVENT_EXPLORATION_CANDIDATE_COUNT,
        label="v14 event exploration",
    )
    exploration_records = tuple(exploration["records"])
    centers = _select_refinement_centers(exploration_records)

    refinement_path = workspace / "event_local_refinement" / "report.json"
    refinement = _load_committed_report(
        workspace, "event_local_refinement", refinement_path
    )
    if refinement is None:
        budget = api.budget_type(
            stage="local_refinement",
            total_candidate_count=EVENT_REFINEMENT_CANDIDATE_COUNT,
            seed=int(seed),
        )
        raw_jobs = api.refinement_builder(
            source.parent_config,
            descriptor,
            centers,
            budget=budget,
            validate_configs=True,
        )
        jobs = _normalize_event_jobs(
            raw_jobs,
            descriptor=descriptor,
            api=api,
            expected_count=EVENT_REFINEMENT_CANDIDATE_COUNT,
            stage="local_refinement",
        )
        records, artifacts = _execute_candidate_jobs(
            jobs,
            workspace,
            phase_name="event_local_refinement/candidates",
            workers=int(workers),
            global_rank_offset=EVENT_EXPLORATION_CANDIDATE_COUNT,
        )
        ranked = _rank_event_records(records)
        refinement = _commit_report(
            workspace,
            "event_local_refinement",
            refinement_path,
            _phase_payload(
                stage="local_refinement",
                records=ranked,
                descriptor_mapping=descriptor_mapping,
                source_authentication_id=source.source_authentication_id,
                centers=centers,
            ),
            stage_input={
                "exploration_report_sha256": file_sha256(exploration_path),
                "candidate_count": EVENT_REFINEMENT_CANDIDATE_COUNT,
                "centers_sha256": canonical_sha256(centers),
                "jobs_sha256": canonical_sha256(jobs),
            },
            artifacts=artifacts,
        )
    _validate_phase_report(
        refinement,
        expected_candidate_count=EVENT_REFINEMENT_CANDIDATE_COUNT,
        label="v14 event local refinement",
    )
    refinement_records = tuple(refinement["records"])

    combined = _rank_event_records((*exploration_records, *refinement_records))
    catalog_report = _catalog_stage(
        combined,
        workspace,
        experiment_id=definition.experiment_id,
        target_success_count=int(target_success_count),
        exploration_report_path=exploration_path,
        refinement_report_path=refinement_path,
    )
    full_count = sum(bool(value.get("full_success", False)) for value in combined)
    result = {
        "contact_preserving_event_rescue_result_schema_version": (
            EVENT_RESULT_SCHEMA_VERSION
        ),
        "complete": True,
        "experiment_id": definition.experiment_id,
        "source_rescue_campaign": str(source.root),
        "source_authentication_id": source.source_authentication_id,
        "source_parent_candidate_id": source.parent_candidate_id,
        "event_descriptor": descriptor_mapping,
        "target_success_count": int(target_success_count),
        "exploration_candidate_count": len(exploration_records),
        "exploration_full_success_count": int(exploration["full_success_count"]),
        "local_refinement_candidate_count": len(refinement_records),
        "local_refinement_full_success_count": int(
            refinement["full_success_count"]
        ),
        "full_success_count": full_count,
        "target_reached": full_count >= int(target_success_count),
        "catalogs": copy.deepcopy(dict(catalog_report["catalogs"])),
        "robustness": None,
        "fixed_mass_geometry_ablation": True,
        "stop_reason": (
            "target_full_success_count_reached"
            if full_count >= int(target_success_count)
            else "declared_event_exploration_and_local_refinement_exhausted"
        ),
    }

    # Re-authenticate the immutable source and numerical descriptor after all
    # physics and rendering, then require the complete code/source manifest to
    # remain unchanged before the final atomic result commit.
    final_source = authenticate_completed_event_rescue_source(source.root)
    if final_source.source_authentication_id != source.source_authentication_id:
        raise RuntimeError("immutable event-rescue source changed during execution")
    final_descriptor = _build_event_descriptor(final_source, api)
    if canonical_sha256(_mapping_descriptor(final_descriptor)) != canonical_sha256(
        descriptor_mapping
    ):
        raise RuntimeError("event detector changed during campaign execution")
    final_manifest = _bind_descriptor_to_manifest(
        _event_manifest(config_file, final_source, seed=int(seed)),
        _mapping_descriptor(final_descriptor),
    )
    if canonical_sha256(final_manifest) != canonical_sha256(manifest):
        raise RuntimeError("event-rescue code or inputs changed during execution")
    validate_stage_ledger(workspace)

    result_path = workspace / f"event_rescue_result_target_{target_success_count}.json"
    result_stage = f"event_result_target_{target_success_count}"
    existing = _load_committed_report(workspace, result_stage, result_path)
    if existing is None:
        result = _commit_report(
            workspace,
            result_stage,
            result_path,
            result,
            stage_input={
                "catalog_report_sha256": file_sha256(
                    workspace
                    / "catalogs"
                    / f"target_{target_success_count}"
                    / "report.json"
                ),
                "exploration_report_sha256": file_sha256(exploration_path),
                "refinement_report_sha256": file_sha256(refinement_path),
            },
        )
    elif canonical_sha256(existing) != canonical_sha256(result):
        raise RuntimeError("committed event-rescue result changed on resume")
    return result


__all__ = [
    "AuthenticatedEventRescueSource",
    "DEFAULT_SEED",
    "EVENT_EXPLORATION_CANDIDATE_COUNT",
    "EVENT_REFINEMENT_CANDIDATE_COUNT",
    "authenticate_completed_event_rescue_source",
    "build_contact_preserving_event_rescue_manifest",
    "run_contact_preserving_event_rescue_campaign",
]
