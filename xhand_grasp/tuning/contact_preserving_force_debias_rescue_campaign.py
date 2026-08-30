"""Authenticated force-debias rescue campaign for schema-v14 lifts.

The campaign consumes a *completed* adaptive-event campaign as immutable
evidence.  It runs a fixed 160-candidate force discovery followed by eight
64-candidate local refinements, then independently full-reset reruns and
renders the best five candidates.  Search candidates are globally unique by
both candidate identity and physical-plan hash.

Numerical construction lives in :mod:`contact_preserving_force_debias_rescue`.
Keeping the orchestration here deliberately small makes the power-loss and
provenance rules independently testable from MuJoCo physics.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
from .contact_preserving_lift_rescue_campaign import (
    _commit_report,
    _execute_candidate_jobs,
    _load_committed_report,
    _publish_rescue_viewer_catalogs,
    _run_full_reset_candidate,
    _validate_phase_report,
)
from .contact_preserving_candidate_artifacts import (
    authenticate_v14_candidate_artifacts,
)
from .contact_preserving_planned_lift_campaign import (
    _bind_catalog_entry_aliases,
    build_contact_preserving_planned_lift_manifest,
)


FORCE_DEBIAS_CAMPAIGN_SCHEMA_VERSION = 1
FORCE_DEBIAS_PHASE_REPORT_SCHEMA_VERSION = 1
FORCE_DEBIAS_RESULT_SCHEMA_VERSION = 1
DISCOVERY_CENTER_COUNT = 5
DISCOVERY_CANDIDATES_PER_CENTER = 32
DISCOVERY_CANDIDATE_COUNT = DISCOVERY_CENTER_COUNT * DISCOVERY_CANDIDATES_PER_CENTER
REFINEMENT_CENTER_COUNT = 8
REFINEMENT_CANDIDATES_PER_CENTER = 64
REFINEMENT_CANDIDATE_COUNT = REFINEMENT_CENTER_COUNT * REFINEMENT_CANDIDATES_PER_CENTER
PUBLISHED_CANDIDATE_COUNT = 5
DEFAULT_SEED = 20260821


@dataclass(frozen=True, slots=True)
class _ForceDebiasApi:
    source_authenticator: Callable[..., Any]
    jobs_builder: Callable[..., Sequence[Mapping[str, Any]]]
    job_authenticator: Callable[..., Any]
    ranker: Callable[[Mapping[str, Any]], Any]


def _force_debias_api() -> _ForceDebiasApi:
    """Load the numerical layer only when this campaign is requested."""

    from . import contact_preserving_force_debias_rescue as numerical

    required = {
        "authenticate_force_debias_source": "source_authenticator",
        "build_force_debias_jobs": "jobs_builder",
        "authenticate_force_debias_job": "job_authenticator",
        "force_debias_candidate_rank": "ranker",
    }
    missing = [name for name in required if not hasattr(numerical, name)]
    if missing:
        raise RuntimeError(
            "force-debias numerical API is incomplete: " + ", ".join(missing)
        )
    return _ForceDebiasApi(
        **{field: getattr(numerical, name) for name, field in required.items()}
    )


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    for method_name in ("as_mapping", "to_dict", "descriptor"):
        method = getattr(value, method_name, None)
        if callable(method):
            payload = method()
            if isinstance(payload, Mapping):
                return copy.deepcopy(dict(payload))
    raise RuntimeError(f"{label} has no canonical mapping representation")


def _attribute(value: Any, name: str, *, label: str) -> Any:
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, Mapping) and name in value:
        return value[name]
    raise RuntimeError(f"{label} lost required field {name}")


def _source_root(source: Any) -> Path:
    return Path(str(_attribute(source, "root", label="force-debias source"))).resolve()


def _source_id(source: Any) -> str:
    value = str(
        _attribute(
            source, "source_authentication_id", label="force-debias source"
        )
    )
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("force-debias source authentication ID is invalid")
    return value


def _source_artifact_paths(source: Any) -> tuple[Path, ...]:
    raw = _attribute(source, "artifact_paths", label="force-debias source")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RuntimeError("force-debias source artifact_paths must be a sequence")
    result: list[Path] = []
    for value in raw:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"force-debias source evidence is missing: {path}")
        result.append(path)
    if not result:
        raise RuntimeError("force-debias source has no authenticated evidence")
    return tuple(dict.fromkeys(result))


def _source_centers(source: Any) -> tuple[Any, ...]:
    raw = _attribute(source, "centers", label="force-debias source")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RuntimeError("force-debias source centers must be a sequence")
    centers = tuple(raw)
    if len(centers) != DISCOVERY_CENTER_COUNT:
        raise RuntimeError("force-debias source must authenticate exactly five centers")
    identities: list[str] = []
    for center in centers:
        mapping = _mapping(center, label="force-debias center")
        identity = mapping.get("center_id")
        trace_path = mapping.get("trace_path")
        if not isinstance(identity, str) or len(identity) != 64:
            raise RuntimeError("force-debias center identity is invalid")
        path = Path(str(trace_path)).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError("force-debias center lost its full-reset trace")
        declared = mapping.get("trace_sha256")
        if isinstance(declared, str) and declared != file_sha256(path):
            raise RuntimeError("force-debias center trace SHA-256 changed")
        identities.append(identity)
    if len(identities) != len(set(identities)):
        raise RuntimeError("force-debias source centers are not unique")
    return centers


def _source_evidence_sha256(source: Any) -> dict[str, str]:
    root = _source_root(source)
    result: dict[str, str] = {}
    for path in _source_artifact_paths(source):
        key = (
            str(path.relative_to(root))
            if path.is_relative_to(root)
            else f"ancestor/{canonical_sha256(str(path))}/{path.name}"
        )
        result[key] = file_sha256(path)
    return dict(sorted(result.items()))


def _job_physical_hash(job: Mapping[str, Any]) -> str:
    value = job.get("physical_plan_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError("force-debias job lost physical-plan SHA-256")
    return value


def _operation_scale(job: Mapping[str, Any]) -> float:
    feedback = job.get("feedback_parameters")
    projected = job.get("projected_parameters")
    config = job.get("config")
    if not isinstance(feedback, Mapping) or not isinstance(projected, Mapping) or not isinstance(config, Mapping):
        raise RuntimeError("force-debias job lost operation-scale provenance")
    value = feedback.get("operation_scale")
    projected_value = projected.get("feedback:operation_scale")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError("force-debias operation scale is invalid")
    scale = float(value)
    if not isinstance(projected_value, (int, float)) or float(projected_value) != scale:
        raise RuntimeError("force-debias projected operation scale changed")
    if scale < 0.70 - 1e-12 or scale > 1.00 + 1e-12:
        raise RuntimeError("force-debias operation scale left the registered range")
    targets = config.get("contact_force_targets_n")
    if not isinstance(targets, Mapping):
        raise RuntimeError("force-debias config lost force targets")
    target_id = targets.get("target_id")
    if not isinstance(target_id, str) or len(target_id) != 64:
        raise RuntimeError("force-debias operation scale is not bound into target_id")
    if abs(scale - 1.0) > 1e-12 and float(targets.get("operation_scale", math.nan)) != scale:
        raise RuntimeError("force-debias non-unit operation scale is absent from config")
    return scale


def _normalize_jobs(
    raw_jobs: Sequence[Mapping[str, Any]],
    *,
    api: _ForceDebiasApi,
    source: Any,
    stage: str,
    expected_count: int,
    excluded_physical_plan_sha256: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    excluded = tuple(sorted(set(str(value) for value in excluded_physical_plan_sha256)))
    if any(len(value) != 64 for value in excluded):
        raise ValueError("force-debias excluded physical identity is invalid")
    jobs: list[dict[str, Any]] = []
    for raw in raw_jobs:
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"force-debias {stage} job is malformed")
        job = copy.deepcopy(dict(raw))
        api.job_authenticator(
            job,
            source,
            expected_stage=stage,
            expected_excluded_physical_plan_sha256=excluded,
        )
        if (
            job.get("stage") != stage
            or job.get("full_reset_required") is not True
            or not isinstance(job.get("config"), Mapping)
            or int(job["config"].get("schema_version", 0)) != 14
        ):
            raise RuntimeError(
                f"force-debias {stage} job lost stage/config/full-reset contract"
            )
        candidate_id = int(job.get("candidate_id", -1))
        local_index = int(job.get("local_index", -1))
        payload_hash = job.get("candidate_payload_sha256")
        if (
            candidate_id < 0
            or local_index < 0
            or not isinstance(payload_hash, str)
            or len(payload_hash) != 64
        ):
            raise RuntimeError(f"force-debias {stage} job has invalid identity")
        physical = _job_physical_hash(job)
        _operation_scale(job)
        if physical in excluded:
            raise RuntimeError(f"force-debias {stage} reused an excluded physical plan")
        metadata = {
            key: copy.deepcopy(value) for key, value in job.items() if key != "config"
        }
        metadata.update(
            {
                "runner_stage": stage,
                "job_sequence_index": local_index,
                "global_exclusion_set_sha256": canonical_sha256(excluded),
                "full_reset_required": True,
            }
        )
        jobs.append(
            {
                **job,
                "candidate_sha256": payload_hash,
                "job_sequence_index": local_index,
                "job_metadata": metadata,
            }
        )
    if len(jobs) != int(expected_count):
        raise RuntimeError(
            f"force-debias {stage} built {len(jobs)} jobs, expected {expected_count}"
        )
    jobs.sort(key=lambda value: (int(value["local_index"]), int(value["candidate_id"])))
    identities = {
        "candidate IDs": [int(value["candidate_id"]) for value in jobs],
        "candidate payloads": [str(value["candidate_payload_sha256"]) for value in jobs],
        "local indices": [int(value["local_index"]) for value in jobs],
        "physical plans": [_job_physical_hash(value) for value in jobs],
    }
    for label, values in identities.items():
        if len(values) != len(set(values)):
            raise RuntimeError(f"force-debias {stage} contains duplicate {label}")
    if identities["local indices"] != list(range(int(expected_count))):
        raise RuntimeError(f"force-debias {stage} local indices are not contiguous")
    if stage == "discovery":
        scales = tuple(_operation_scale(value) for value in jobs)
        if min(scales) > 0.70 + 1e-12 or max(scales) < 1.00 - 1e-12:
            raise RuntimeError(
                "force-debias discovery did not cover operation scale 0.70 through 1.00"
            )
    return tuple(jobs)


def _build_jobs(
    api: _ForceDebiasApi,
    source: Any,
    *,
    stage: str,
    total_count: int,
    seed: int,
    refinement_records: Sequence[Mapping[str, Any]] = (),
    excluded_physical_plan_sha256: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    raw = api.jobs_builder(
        source,
        stage=stage,
        total_count=int(total_count),
        seed=int(seed),
        local_index_offset=0,
        refinement_records=tuple(copy.deepcopy(dict(value)) for value in refinement_records),
        excluded_physical_plan_sha256=tuple(
            sorted(set(str(value) for value in excluded_physical_plan_sha256))
        ),
    )
    return _normalize_jobs(
        tuple(raw),
        api=api,
        source=source,
        stage=stage,
        expected_count=int(total_count),
        excluded_physical_plan_sha256=excluded_physical_plan_sha256,
    )


def _rank_key(record: Mapping[str, Any], api: _ForceDebiasApi) -> tuple[Any, ...]:
    rank = api.ranker(record)
    if isinstance(rank, list):
        rank = tuple(rank)
    elif not isinstance(rank, tuple):
        rank = (rank,)
    return (*rank, int(record.get("candidate_id", 1 << 62)))


def _rank_records(
    records: Sequence[Mapping[str, Any]], api: _ForceDebiasApi
) -> tuple[dict[str, Any], ...]:
    ranked = tuple(
        sorted(
            (copy.deepcopy(dict(value)) for value in records),
            key=lambda value: _rank_key(value, api),
        )
    )
    if len(ranked) != len(records):
        raise RuntimeError("force-debias ranking lost candidates")
    return ranked


def _record_physical_hash(record: Mapping[str, Any]) -> str:
    metadata = record.get("rescue_job")
    value = metadata.get("physical_plan_sha256") if isinstance(metadata, Mapping) else None
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError("force-debias result lost physical-plan identity")
    return value


def _assert_global_record_uniqueness(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    candidate_ids = tuple(int(value["candidate_id"]) for value in records)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("force-debias stages reused a candidate ID")
    physical = tuple(_record_physical_hash(value) for value in records)
    if len(physical) != len(set(physical)):
        raise RuntimeError("force-debias stages reused a physical plan")
    return candidate_ids, physical


def _select_refinement_records(
    discovery_records: Sequence[Mapping[str, Any]],
    api: _ForceDebiasApi,
) -> tuple[dict[str, Any], ...]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in _rank_records(discovery_records, api):
        physical = _record_physical_hash(record)
        if physical in seen:
            continue
        seen.add(physical)
        selected.append(copy.deepcopy(dict(record)))
        if len(selected) == REFINEMENT_CENTER_COUNT:
            break
    if len(selected) != REFINEMENT_CENTER_COUNT:
        raise RuntimeError("force-debias discovery retained fewer than eight unique centers")
    return tuple(selected)


def _phase_payload(
    *,
    stage: str,
    records: Sequence[Mapping[str, Any]],
    source_authentication_id: str,
    refinement_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    physical = tuple(_record_physical_hash(value) for value in records)
    return {
        "force_debias_phase_report_schema_version": FORCE_DEBIAS_PHASE_REPORT_SCHEMA_VERSION,
        "stage": stage,
        "complete": True,
        "declared_candidate_count": len(records),
        "candidate_count": len(records),
        "full_success_count": sum(bool(value.get("full_success", False)) for value in records),
        "physical_unique_candidate_count": len(set(physical)),
        "physical_duplicate_candidate_count": len(physical) - len(set(physical)),
        "physical_plan_set_sha256": canonical_sha256(sorted(set(physical))),
        "source_authentication_id": source_authentication_id,
        "refinement_records": [
            {
                "candidate_id": int(value["candidate_id"]),
                "physical_plan_sha256": _record_physical_hash(value),
            }
            for value in refinement_records
        ],
        "records": [copy.deepcopy(dict(value)) for value in records],
    }


def _assert_committed_stage_input(
    workspace: Path, stage: str, stage_input: Mapping[str, Any]
) -> None:
    ledger = validate_stage_ledger(workspace)
    record = ledger.get("stages", {}).get(stage)
    if not isinstance(record, Mapping):
        raise RuntimeError(f"committed force-debias stage {stage} disappeared")
    if record.get("stage_input_sha256") != canonical_sha256(dict(stage_input)):
        raise RuntimeError(f"committed force-debias stage {stage} input changed on resume")


def _execute_phase(
    workspace: Path,
    *,
    stage: str,
    directory: str,
    jobs: Sequence[Mapping[str, Any]],
    source_authentication_id: str,
    workers: int,
    global_rank_offset: int,
    refinement_records: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], Path]:
    report_path = workspace / directory / "report.json"
    stage_input = {
        "source_authentication_id": source_authentication_id,
        "jobs_sha256": canonical_sha256(jobs),
        "refinement_records_sha256": canonical_sha256(refinement_records),
    }
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        _assert_committed_stage_input(workspace, stage, stage_input)
        _validate_phase_report(
            existing,
            expected_candidate_count=len(jobs),
            label=f"v14 force-debias {stage}",
        )
        return existing, report_path
    records, artifacts = _execute_candidate_jobs(
        jobs,
        workspace,
        phase_name=f"{directory}/candidates",
        workers=int(workers),
        global_rank_offset=int(global_rank_offset),
    )
    payload = _phase_payload(
        stage=stage,
        records=records,
        source_authentication_id=source_authentication_id,
        refinement_records=refinement_records,
    )
    committed = _commit_report(
        workspace,
        stage,
        report_path,
        payload,
        stage_input=stage_input,
        artifacts=artifacts,
    )
    _validate_phase_report(
        committed,
        expected_candidate_count=len(jobs),
        label=f"v14 force-debias {stage}",
    )
    return committed, report_path


def _materialize_publication_records(
    records: Sequence[Mapping[str, Any]], workspace: Path, *, target_success_count: int
) -> tuple[dict[str, Any], ...]:
    materialized: list[dict[str, Any]] = []
    for record in records[:PUBLISHED_CANDIDATE_COUNT]:
        candidate_id = int(record["candidate_id"])
        source_root = (workspace / str(record["artifact_directory"])).resolve()
        if not source_root.is_relative_to(workspace):
            raise RuntimeError("force-debias catalog source escaped its workspace")
        bundle = authenticate_v14_candidate_artifacts(
            source_root,
            expected_candidate_id=candidate_id,
            expected_retain_grasp_success=False,
        )
        config = json.loads(bundle.config_path.read_text(encoding="utf-8"))
        final_root = (
            workspace
            / "catalog_source_reruns"
            / f"force_debias_target_{target_success_count}"
            / f"candidate_{candidate_id}"
        )
        rerun = _run_full_reset_candidate(config, final_root, candidate_id, final_rerun=True)
        final_bundle = authenticate_v14_candidate_artifacts(
            final_root,
            expected_config=config,
            expected_candidate_id=candidate_id,
            require_retained_trace=True,
            expected_retain_grasp_success=False,
        )
        if canonical_sha256(rerun.get("summary")) != canonical_sha256(record.get("summary")):
            raise RuntimeError("force-debias catalog full-reset rerun changed its summary")
        if bool(rerun.get("full_success", False)) != bool(record.get("full_success", False)):
            raise RuntimeError("force-debias catalog full-reset rerun changed success status")
        materialized.append(
            {
                **copy.deepcopy(dict(record)),
                **copy.deepcopy(final_bundle.result),
                "artifact_directory": str(final_root.relative_to(workspace)),
            }
        )
    return tuple(materialized)


def _restrict_catalog_aliases(
    catalog_path: Path, *, hard_pass_candidate_ids: Sequence[int]
) -> None:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries = payload.get("trajectories")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("force-debias publisher produced an empty catalog")
    hard_ids = {int(value) for value in hard_pass_candidate_ids}
    successes = [
        value for value in entries if int(value.get("candidate_id", -1)) in hard_ids
    ]
    chosen = successes[0] if successes else entries[0]
    alias = "best_nominal" if successes else "best_attempt"
    payload["aliases"] = {alias: chosen["trajectory_id"]}
    _bind_catalog_entry_aliases(entries, payload["aliases"])
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
        "hard_success_then_force_debias_rank; best_nominal_only_on_hard_pass"
    )
    payload["force_debias_rescue"] = True
    write_json(catalog_path, payload)


def _catalog_stage(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    experiment_id: str,
    target_success_count: int,
    report_hashes: Mapping[str, str],
) -> dict[str, Any]:
    stage = f"force_debias_catalog_target_{target_success_count}"
    root = workspace / "catalogs" / f"target_{target_success_count}"
    report_path = root / "report.json"
    selected = tuple(copy.deepcopy(dict(value)) for value in records[:PUBLISHED_CANDIDATE_COUNT])
    stage_input = {
        **copy.deepcopy(dict(report_hashes)),
        "target_success_count": int(target_success_count),
        "selected_records_sha256": canonical_sha256(selected),
    }
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        _assert_committed_stage_input(workspace, stage, stage_input)
        return existing
    materialized = _materialize_publication_records(
        selected, workspace, target_success_count=int(target_success_count)
    )
    catalogs = _publish_rescue_viewer_catalogs(
        materialized, workspace, root, experiment_id=experiment_id
    )
    hard_pass_ids = tuple(
        int(value["candidate_id"])
        for value in materialized
        if bool(value.get("full_success", False))
    )
    for relative in catalogs.values():
        _restrict_catalog_aliases(
            workspace / relative, hard_pass_candidate_ids=hard_pass_ids
        )
    artifacts: list[Path] = []
    for relative in catalogs.values():
        artifacts.extend(authenticated_catalog_artifact_paths(workspace / relative))
    full_count = sum(bool(value.get("full_success", False)) for value in records)
    payload = {
        "force_debias_catalog_report_schema_version": 1,
        "complete": True,
        "target_success_count": int(target_success_count),
        "full_success_count": full_count,
        "target_reached": full_count >= int(target_success_count),
        "published_candidate_count": len(materialized),
        "published_candidate_ids": [int(value["candidate_id"]) for value in materialized],
        "catalogs": catalogs,
        "alias_policy": "best_attempt_on_failure_best_nominal_only_on_hard_pass",
        "video_policy": "independent_fresh_full_reset_ffprobe_and_full_decode",
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


def _manifest(config_path: Path, source: Any, *, seed: int) -> dict[str, Any]:
    centers = _source_centers(source)
    base = build_contact_preserving_planned_lift_manifest(config_path, seed=int(seed))
    base.pop("campaign_input_sha256", None)
    base.update(
        {
            "contact_preserving_force_debias_campaign_schema_version": FORCE_DEBIAS_CAMPAIGN_SCHEMA_VERSION,
            "campaign_kind": "contact_preserving_force_debias_rescue",
            "source_adaptive_campaign_path": str(_source_root(source)),
            "source_adaptive_authentication": _mapping(source, label="force-debias source"),
            "source_adaptive_evidence_sha256": _source_evidence_sha256(source),
            "source_center_set_sha256": canonical_sha256(
                [_mapping(value, label="force-debias center") for value in centers]
            ),
            "force_debias_budget": {
                "discovery_center_count": DISCOVERY_CENTER_COUNT,
                "discovery_candidates_per_center": DISCOVERY_CANDIDATES_PER_CENTER,
                "discovery_candidate_count": DISCOVERY_CANDIDATE_COUNT,
                "refinement_center_count": REFINEMENT_CENTER_COUNT,
                "refinement_candidates_per_center": REFINEMENT_CANDIDATES_PER_CENTER,
                "refinement_candidate_count": REFINEMENT_CANDIDATE_COUNT,
                "publication_candidate_count": PUBLISHED_CANDIDATE_COUNT,
                "candidate_execution": "fresh_full_reset_free_dynamics",
                "catalog_execution": "independent_fresh_full_reset_rerun",
                "physical_plan_uniqueness": "global_across_discovery_and_refinement",
            },
        }
    )
    return {**base, "campaign_input_sha256": canonical_sha256(base)}


def build_contact_preserving_force_debias_rescue_manifest(
    config_path: str | Path,
    source_adaptive_campaign: str | Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    api = _force_debias_api()
    source = api.source_authenticator(Path(source_adaptive_campaign).expanduser().resolve())
    return _manifest(Path(config_path).expanduser().resolve(), source, seed=int(seed))


def run_contact_preserving_force_debias_rescue_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    source_adaptive_campaign: str | Path,
    resume: bool,
    target_success_count: int,
    workers: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run or resume the fixed force-debias discovery/refinement campaign."""

    if int(workers) <= 0:
        raise ValueError("workers must be positive")
    if int(seed) != DEFAULT_SEED:
        raise ValueError("schema-v14 force-debias seed must be 20260821")
    if int(target_success_count) not in (1, 5):
        raise ValueError("target_success_count must be one or five")
    config_file = Path(config_path).expanduser().resolve()
    workspace = Path(output_dir).expanduser().resolve()
    source_root = Path(source_adaptive_campaign).expanduser().resolve()
    if workspace == source_root or workspace.is_relative_to(source_root):
        raise ValueError("force-debias workspace must be outside its immutable source")
    definition = resolve_experiment(load_config(config_file))
    if getattr(definition, "contact_preserving_planned_lift_campaign", None) is None:
        raise ValueError("force-debias runner requires the registered schema-v14 experiment")

    api = _force_debias_api()
    source = api.source_authenticator(source_root)
    if _source_root(source) != source_root:
        raise RuntimeError("force-debias authenticator returned the wrong source root")
    _source_centers(source)
    manifest = _manifest(config_file, source, seed=int(seed))
    initialize_or_resume_campaign(workspace, manifest, resume=bool(resume))

    audit_path = workspace / "force_debias_source_audit.json"
    audit_payload = {
        "force_debias_source_audit_schema_version": 1,
        "complete": True,
        "source": _mapping(source, label="force-debias source"),
        "source_authentication_id": _source_id(source),
        "evidence_sha256": _source_evidence_sha256(source),
        "records": [],
    }
    audit_stage_input = {
        "source_authentication_id": _source_id(source),
        "source_sha256": canonical_sha256(_mapping(source, label="force-debias source")),
    }
    audit = _load_committed_report(workspace, "force_debias_source_audit", audit_path)
    if audit is None:
        audit = _commit_report(
            workspace,
            "force_debias_source_audit",
            audit_path,
            audit_payload,
            stage_input=audit_stage_input,
        )
    else:
        _assert_committed_stage_input(workspace, "force_debias_source_audit", audit_stage_input)
        if canonical_sha256(audit) != canonical_sha256(audit_payload):
            raise RuntimeError("force-debias source audit changed on resume")

    discovery_jobs = _build_jobs(
        api,
        source,
        stage="discovery",
        total_count=DISCOVERY_CANDIDATE_COUNT,
        seed=int(seed),
    )
    discovery, discovery_path = _execute_phase(
        workspace,
        stage="force_debias_discovery",
        directory="force_debias_discovery",
        jobs=discovery_jobs,
        source_authentication_id=_source_id(source),
        workers=int(workers),
        global_rank_offset=0,
    )
    refinement_centers = _select_refinement_records(discovery["records"], api)
    excluded = tuple(_job_physical_hash(value) for value in discovery_jobs)
    refinement_jobs = _build_jobs(
        api,
        source,
        stage="refinement",
        total_count=REFINEMENT_CANDIDATE_COUNT,
        seed=int(seed),
        refinement_records=refinement_centers,
        excluded_physical_plan_sha256=excluded,
    )
    refinement, refinement_path = _execute_phase(
        workspace,
        stage="force_debias_refinement",
        directory="force_debias_refinement",
        jobs=refinement_jobs,
        source_authentication_id=_source_id(source),
        workers=int(workers),
        global_rank_offset=DISCOVERY_CANDIDATE_COUNT,
        refinement_records=refinement_centers,
    )
    combined = _rank_records((*discovery["records"], *refinement["records"]), api)
    combined_candidate_ids, combined_physical = _assert_global_record_uniqueness(combined)
    report_hashes = {
        "discovery_report_sha256": file_sha256(discovery_path),
        "refinement_report_sha256": file_sha256(refinement_path),
    }
    catalog_report = _catalog_stage(
        combined,
        workspace,
        experiment_id=definition.experiment_id,
        target_success_count=int(target_success_count),
        report_hashes=report_hashes,
    )
    full_count = sum(bool(value.get("full_success", False)) for value in combined)
    result = {
        "contact_preserving_force_debias_rescue_result_schema_version": FORCE_DEBIAS_RESULT_SCHEMA_VERSION,
        "complete": True,
        "experiment_id": definition.experiment_id,
        "source_adaptive_campaign": str(source_root),
        "source_authentication_id": _source_id(source),
        "target_success_count": int(target_success_count),
        "discovery_candidate_count": len(discovery["records"]),
        "discovery_full_success_count": int(discovery["full_success_count"]),
        "refinement_candidate_count": len(refinement["records"]),
        "refinement_full_success_count": int(refinement["full_success_count"]),
        "physical_unique_candidate_count": len(set(combined_physical)),
        "physical_duplicate_candidate_count": len(combined_physical) - len(set(combined_physical)),
        "candidate_id_unique_count": len(set(combined_candidate_ids)),
        "candidate_id_duplicate_count": len(combined_candidate_ids)
        - len(set(combined_candidate_ids)),
        "physical_plan_set_sha256": canonical_sha256(sorted(set(combined_physical))),
        "full_success_count": full_count,
        "target_reached": full_count >= int(target_success_count),
        "catalogs": copy.deepcopy(dict(catalog_report["catalogs"])),
        "published_candidate_ids": copy.deepcopy(list(catalog_report["published_candidate_ids"])),
        "robustness": None,
        "fixed_mass_geometry_ablation": True,
        "exit_code": 0 if full_count else 2,
        "stop_reason": (
            "target_full_success_count_reached"
            if full_count >= int(target_success_count)
            else "declared_force_debias_discovery_and_refinement_exhausted"
        ),
    }

    final_source = api.source_authenticator(source_root)
    _source_centers(final_source)
    if canonical_sha256(_mapping(final_source, label="force-debias source")) != canonical_sha256(
        _mapping(source, label="force-debias source")
    ):
        raise RuntimeError("immutable force-debias source changed during execution")
    if canonical_sha256(_manifest(config_file, final_source, seed=int(seed))) != canonical_sha256(manifest):
        raise RuntimeError("force-debias implementation or inputs changed during execution")
    validate_stage_ledger(workspace)

    result_path = workspace / f"force_debias_rescue_result_target_{target_success_count}.json"
    result_stage = f"force_debias_result_target_{target_success_count}"
    result_input = {
        **report_hashes,
        "catalog_report_sha256": file_sha256(
            workspace / "catalogs" / f"target_{target_success_count}" / "report.json"
        ),
    }
    existing = _load_committed_report(workspace, result_stage, result_path)
    if existing is None:
        result = _commit_report(
            workspace,
            result_stage,
            result_path,
            result,
            stage_input=result_input,
        )
    else:
        _assert_committed_stage_input(workspace, result_stage, result_input)
        if canonical_sha256(existing) != canonical_sha256(result):
            raise RuntimeError("committed force-debias result changed on resume")
        result = existing
    return result


__all__ = [
    "DEFAULT_SEED",
    "DISCOVERY_CANDIDATE_COUNT",
    "REFINEMENT_CANDIDATE_COUNT",
    "build_contact_preserving_force_debias_rescue_manifest",
    "run_contact_preserving_force_debias_rescue_campaign",
]
