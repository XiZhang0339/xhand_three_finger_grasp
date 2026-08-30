"""Authenticated fixed-budget campaign for the v14 bounded micro-jerk rescue.

The numerical design is intentionally isolated in
``contact_preserving_micro_jerk_rescue``.  This module owns only immutable
source authentication, fresh-reset execution, atomic resume, ranking, final
trace materialization and Viewer publication.  It is not wired into the
public CLI: the formal caller selects it explicitly after preflight.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
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
from ..artifacts import file_sha256
from ..config import load_config
from ..experiment import resolve_experiment
from ..grasp_pose import canonical_sha256
from .contact_preserving_candidate_artifacts import authenticate_v14_candidate_artifacts
from .contact_preserving_force_debias_rescue_campaign import _restrict_catalog_aliases
from .contact_preserving_lift_rescue_campaign import (
    _commit_report,
    _execute_candidate_jobs,
    _load_committed_report,
    _publish_rescue_viewer_catalogs,
    _run_full_reset_candidate,
    _validate_phase_report,
)
from .contact_preserving_planned_lift_campaign import (
    build_contact_preserving_planned_lift_manifest,
)


MICRO_JERK_CAMPAIGN_SCHEMA_VERSION = 1
MICRO_JERK_PHASE_REPORT_SCHEMA_VERSION = 1
MICRO_JERK_RESULT_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
MICRO_CANDIDATE_COUNT = 1024
MICRO_CENTER_COUNT = 8
SENSITIVITY_COUNT_PER_CENTER = 32
TRUST_RADII = (0.015, 0.03, 0.06)
TRUST_COUNT_PER_RADIUS_PER_CENTER = 32
PUBLISHED_CANDIDATE_COUNT = 5


@dataclass(frozen=True, slots=True)
class _MicroJerkApi:
    source_authenticator: Callable[..., Any]
    jobs_builder: Callable[..., Sequence[Mapping[str, Any]]]
    job_authenticator: Callable[..., Any]
    ranker: Callable[[Mapping[str, Any]], Any]


def _micro_jerk_api() -> _MicroJerkApi:
    from . import contact_preserving_micro_jerk_rescue as numerical

    required = {
        "authenticate_micro_jerk_source": "source_authenticator",
        "build_micro_jerk_jobs": "jobs_builder",
        "authenticate_micro_jerk_job": "job_authenticator",
        "micro_jerk_candidate_rank": "ranker",
    }
    missing = [name for name in required if not hasattr(numerical, name)]
    if missing:
        raise RuntimeError("micro-jerk numerical API is incomplete: " + ", ".join(missing))
    return _MicroJerkApi(
        **{field: getattr(numerical, name) for name, field in required.items()}
    )


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    method = getattr(value, "as_mapping", None)
    if callable(method):
        payload = method()
        if isinstance(payload, Mapping):
            return copy.deepcopy(dict(payload))
    raise RuntimeError(f"{label} has no canonical mapping")


def _attribute(value: Any, name: str, *, label: str) -> Any:
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, Mapping) and name in value:
        return value[name]
    raise RuntimeError(f"{label} lost {name}")


def _source_root(source: Any) -> Path:
    return Path(str(_attribute(source, "root", label="micro source"))).resolve()


def _source_id(source: Any) -> str:
    value = str(_attribute(source, "source_authentication_id", label="micro source"))
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("micro source authentication ID is invalid")
    return value


def _source_artifact_paths(source: Any) -> tuple[Path, ...]:
    raw = _attribute(source, "artifact_paths", label="micro source")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RuntimeError("micro source artifact paths are malformed")
    paths = tuple(dict.fromkeys(Path(value).expanduser().resolve() for value in raw))
    if not paths or any(not path.is_file() for path in paths):
        raise RuntimeError("micro source evidence is missing")
    return paths


def _source_centers(source: Any) -> tuple[Any, ...]:
    raw = _attribute(source, "centers", label="micro source")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RuntimeError("micro source centers are malformed")
    centers = tuple(raw)
    if len(centers) != MICRO_CENTER_COUNT:
        raise RuntimeError("micro source must contain exactly eight centers")
    mappings = tuple(_mapping(value, label="micro center") for value in centers)
    identities = tuple(str(value.get("center_id", "")) for value in mappings)
    if any(len(value) != 64 for value in identities) or len(set(identities)) != len(identities):
        raise RuntimeError("micro source center identity changed")
    kinds = Counter(str(value.get("center_kind")) for value in mappings)
    if kinds != {"jerk_only": 6, "contact_boundary": 2}:
        raise RuntimeError("micro source lost its six-plus-two center design")
    widths = {float(value["event_half_width_progress"]) for value in mappings}
    source_candidates = {int(value["source_candidate_id"]) for value in mappings}
    if not {0.18, 0.22}.issubset(widths) or len(source_candidates) < 2:
        raise RuntimeError("micro source lost width/source diversity")
    return centers


def _prior_physical(source: Any) -> tuple[str, ...]:
    raw = _attribute(source, "prior_physical_plan_sha256", label="micro source")
    values = tuple(sorted(set(str(value) for value in raw)))
    if len(values) < 672 or any(len(value) != 64 for value in values):
        raise RuntimeError("micro source physical exclusion set changed")
    return values


def _source_evidence_sha256(source: Any) -> dict[str, str]:
    root = _source_root(source)
    result: dict[str, str] = {}
    for path in _source_artifact_paths(source):
        key = (
            str(path.relative_to(root))
            if path.is_relative_to(root)
            else f"external/{canonical_sha256(str(path))}/{path.name}"
        )
        result[key] = file_sha256(path)
    return dict(sorted(result.items()))


def _normalize_jobs(
    raw_jobs: Sequence[Mapping[str, Any]], *, api: _MicroJerkApi, source: Any
) -> tuple[dict[str, Any], ...]:
    prior = _prior_physical(source)
    prior_set = set(prior)
    jobs: list[dict[str, Any]] = []
    for raw in raw_jobs:
        if not isinstance(raw, Mapping):
            raise RuntimeError("micro-refinement job is malformed")
        job = copy.deepcopy(dict(raw))
        api.job_authenticator(
            job,
            source,
            expected_excluded_physical_plan_sha256=prior,
        )
        config = job.get("config")
        if (
            job.get("stage") != "micro_refinement"
            or job.get("full_reset_required") is not True
            or not isinstance(config, Mapping)
            or int(config.get("schema_version", 0)) != 14
        ):
            raise RuntimeError("micro job lost stage/config/full-reset contract")
        physical = str(job.get("physical_plan_sha256", ""))
        if len(physical) != 64 or physical in prior_set:
            raise RuntimeError("micro job reused excluded physical evidence")
        local_index = int(job.get("local_index", -1))
        candidate_id = int(job.get("candidate_id", -1))
        payload_hash = str(job.get("candidate_payload_sha256", ""))
        if local_index < 0 or candidate_id < 0 or len(payload_hash) != 64:
            raise RuntimeError("micro job identity is invalid")
        metadata = {key: copy.deepcopy(value) for key, value in job.items() if key != "config"}
        metadata.update(
            runner_stage="micro_refinement",
            job_sequence_index=local_index,
            global_exclusion_set_sha256=canonical_sha256(prior),
            full_reset_required=True,
        )
        jobs.append(
            {
                **job,
                "candidate_sha256": payload_hash,
                "job_sequence_index": local_index,
                "job_metadata": metadata,
            }
        )
    jobs.sort(key=lambda value: (int(value["local_index"]), int(value["candidate_id"])))
    if len(jobs) != MICRO_CANDIDATE_COUNT:
        raise RuntimeError("micro campaign did not build exactly 1024 candidates")
    identities = {
        "candidate IDs": [int(value["candidate_id"]) for value in jobs],
        "payloads": [str(value["candidate_payload_sha256"]) for value in jobs],
        "physical plans": [str(value["physical_plan_sha256"]) for value in jobs],
    }
    for label, values in identities.items():
        if len(values) != len(set(values)):
            raise RuntimeError(f"micro campaign contains duplicate {label}")
    if [int(value["local_index"]) for value in jobs] != list(range(MICRO_CANDIDATE_COUNT)):
        raise RuntimeError("micro campaign local indices are not contiguous")
    per_center: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in jobs:
        per_center[str(value["source_center_id"])].append(value)
    if len(per_center) != MICRO_CENTER_COUNT:
        raise RuntimeError("micro campaign did not preserve eight center budgets")
    for values in per_center.values():
        modes = Counter(str(value["sampling_mode"]) for value in values)
        if modes != {"deterministic_sensitivity": 32, "trust_shell": 96}:
            raise RuntimeError("micro per-center sensitivity/trust budget changed")
        radii = Counter(float(value["trust_radius"]) for value in values if value["sampling_mode"] == "trust_shell")
        if radii != {0.015: 32, 0.03: 32, 0.06: 32}:
            raise RuntimeError("micro per-center trust-radius budget changed")
    return tuple(jobs)


def _rank_records(records: Sequence[Mapping[str, Any]], api: _MicroJerkApi) -> tuple[dict[str, Any], ...]:
    def key(value: Mapping[str, Any]) -> tuple[Any, ...]:
        raw = api.ranker(value)
        rank = raw if isinstance(raw, tuple) else tuple(raw) if isinstance(raw, list) else (raw,)
        return (*rank, int(value.get("candidate_id", 1 << 62)))

    return tuple(sorted((copy.deepcopy(dict(value)) for value in records), key=key))


def _record_physical(record: Mapping[str, Any]) -> str:
    metadata = record.get("rescue_job")
    value = metadata.get("physical_plan_sha256") if isinstance(metadata, Mapping) else None
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError("micro result lost physical-plan identity")
    return value


def _phase_payload(records: Sequence[Mapping[str, Any]], source_id: str) -> dict[str, Any]:
    physical = tuple(_record_physical(value) for value in records)
    return {
        "micro_jerk_phase_report_schema_version": MICRO_JERK_PHASE_REPORT_SCHEMA_VERSION,
        "stage": "micro_refinement",
        "complete": True,
        "declared_candidate_count": MICRO_CANDIDATE_COUNT,
        "candidate_count": len(records),
        "full_success_count": sum(bool(value.get("full_success", False)) for value in records),
        "physical_unique_candidate_count": len(set(physical)),
        "physical_duplicate_candidate_count": len(physical) - len(set(physical)),
        "physical_plan_set_sha256": canonical_sha256(sorted(set(physical))),
        "source_authentication_id": source_id,
        "records": [copy.deepcopy(dict(value)) for value in records],
    }


def _assert_committed_stage_input(workspace: Path, stage: str, stage_input: Mapping[str, Any]) -> None:
    record = validate_stage_ledger(workspace).get("stages", {}).get(stage)
    if not isinstance(record, Mapping):
        raise RuntimeError(f"committed micro stage {stage} disappeared")
    if record.get("stage_input_sha256") != canonical_sha256(dict(stage_input)):
        raise RuntimeError(f"committed micro stage {stage} input changed on resume")


def _execute_phase(
    workspace: Path,
    *,
    jobs: Sequence[Mapping[str, Any]],
    source_id: str,
    workers: int,
) -> tuple[dict[str, Any], Path]:
    stage = "micro_jerk_refinement"
    report_path = workspace / stage / "report.json"
    stage_input = {"source_authentication_id": source_id, "jobs_sha256": canonical_sha256(jobs)}
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        _assert_committed_stage_input(workspace, stage, stage_input)
        _validate_phase_report(existing, expected_candidate_count=MICRO_CANDIDATE_COUNT, label="v14 micro-jerk refinement")
        return existing, report_path
    records, artifacts = _execute_candidate_jobs(
        jobs,
        workspace,
        phase_name=f"{stage}/candidates",
        workers=int(workers),
        global_rank_offset=0,
    )
    payload = _phase_payload(records, source_id)
    committed = _commit_report(
        workspace,
        stage,
        report_path,
        payload,
        stage_input=stage_input,
        artifacts=artifacts,
    )
    _validate_phase_report(committed, expected_candidate_count=MICRO_CANDIDATE_COUNT, label="v14 micro-jerk refinement")
    return committed, report_path


def branch_stability_from_trace(trace_path: str | Path) -> dict[str, Any]:
    """Compute the soft contact-branch tie-break from one retained trace."""

    with np.load(Path(trace_path), allow_pickle=False) as archive:
        taxels = np.asarray(archive["distal_active_taxel_count"], dtype=np.int64)
        centroids = np.asarray(archive["target_face_contact_centroid_cube_local_m"], dtype=np.float64)
        valid = np.asarray(archive["target_face_contact_centroid_valid"], dtype=bool)
        start = int(np.asarray(archive["manipulation_start_step"]).reshape(-1)[0])
        end = int(np.asarray(archive["manipulation_end_step"]).reshape(-1)[0])
    if taxels.ndim != 2 or taxels.shape[1] != 3 or centroids.shape != (taxels.shape[0], 3, 3) or valid.shape != taxels.shape:
        raise RuntimeError("micro branch-stability trace shapes changed")
    if not 1 <= start < end <= len(taxels) or not np.isfinite(centroids).all():
        raise RuntimeError("micro branch-stability trace bounds are invalid")
    switches = np.count_nonzero(np.diff(taxels[start - 1 : end], axis=0), axis=0)
    maximum_jump = 0.0
    per_finger_jump: list[float] = []
    for finger in range(3):
        local = 0.0
        for step in range(start, end):
            if valid[step - 1, finger] and valid[step, finger]:
                local = max(local, float(np.linalg.norm(centroids[step, finger] - centroids[step - 1, finger])))
        per_finger_jump.append(local)
        maximum_jump = max(maximum_jump, local)
    return {
        "finger_order": ["thumb", "index", "mid"],
        "taxel_switch_count_per_finger": [int(value) for value in switches],
        "total_taxel_switch_count": int(np.sum(switches)),
        "thumb_taxel_switch_count": int(switches[0]),
        "maximum_witness_jump_per_finger_m": per_finger_jump,
        "maximum_witness_jump_m": maximum_jump,
        "manipulation_start_step": start,
        "manipulation_end_step": end,
    }


def _materialize_top_five(
    ranked: Sequence[Mapping[str, Any]], workspace: Path, api: _MicroJerkApi, *, target_success_count: int
) -> tuple[dict[str, Any], ...]:
    materialized: list[dict[str, Any]] = []
    for record in ranked[:PUBLISHED_CANDIDATE_COUNT]:
        candidate_id = int(record["candidate_id"])
        source_root = (workspace / str(record["artifact_directory"])).resolve()
        if not source_root.is_relative_to(workspace):
            raise RuntimeError("micro publication source escaped workspace")
        bundle = authenticate_v14_candidate_artifacts(
            source_root,
            expected_candidate_id=candidate_id,
            expected_retain_grasp_success=False,
        )
        config = json.loads(bundle.config_path.read_text(encoding="utf-8"))
        final_root = workspace / "catalog_source_reruns" / f"micro_jerk_target_{target_success_count}" / f"candidate_{candidate_id}"
        rerun = _run_full_reset_candidate(config, final_root, candidate_id, final_rerun=True)
        final_bundle = authenticate_v14_candidate_artifacts(
            final_root,
            expected_config=config,
            expected_candidate_id=candidate_id,
            require_retained_trace=True,
            expected_retain_grasp_success=False,
        )
        if canonical_sha256(rerun.get("summary")) != canonical_sha256(record.get("summary")):
            raise RuntimeError("micro publication full-reset rerun changed summary")
        if bool(rerun.get("full_success", False)) != bool(record.get("full_success", False)):
            raise RuntimeError("micro publication full-reset rerun changed success")
        branch = branch_stability_from_trace(final_bundle.trace_path)
        materialized.append(
            {
                **copy.deepcopy(dict(record)),
                **copy.deepcopy(final_bundle.result),
                "artifact_directory": str(final_root.relative_to(workspace)),
                "branch_stability": branch,
            }
        )
    return _rank_records(materialized, api)


def _catalog_stage(
    ranked: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    api: _MicroJerkApi,
    experiment_id: str,
    target_success_count: int,
    refinement_report_sha256: str,
) -> dict[str, Any]:
    stage = f"micro_jerk_catalog_target_{target_success_count}"
    root = workspace / "catalogs" / f"target_{target_success_count}"
    report_path = root / "report.json"
    selected = tuple(copy.deepcopy(dict(value)) for value in ranked[:PUBLISHED_CANDIDATE_COUNT])
    stage_input = {
        "refinement_report_sha256": refinement_report_sha256,
        "target_success_count": int(target_success_count),
        "selected_records_sha256": canonical_sha256(selected),
    }
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        _assert_committed_stage_input(workspace, stage, stage_input)
        return existing
    materialized = _materialize_top_five(selected, workspace, api, target_success_count=target_success_count)
    catalogs = _publish_rescue_viewer_catalogs(materialized, workspace, root, experiment_id=experiment_id)
    hard_ids = tuple(int(value["candidate_id"]) for value in materialized if bool(value.get("full_success", False)))
    for relative in catalogs.values():
        _restrict_catalog_aliases(workspace / relative, hard_pass_candidate_ids=hard_ids)
    artifacts: list[Path] = []
    for relative in catalogs.values():
        artifacts.extend(authenticated_catalog_artifact_paths(workspace / relative))
    payload = {
        "micro_jerk_catalog_report_schema_version": 1,
        "complete": True,
        "target_success_count": int(target_success_count),
        "full_success_count": sum(bool(value.get("full_success", False)) for value in ranked),
        "target_reached": sum(bool(value.get("full_success", False)) for value in ranked) >= int(target_success_count),
        "published_candidate_count": len(materialized),
        "published_candidate_ids": [int(value["candidate_id"]) for value in materialized],
        "published_branch_stability": [
            {"candidate_id": int(value["candidate_id"]), **copy.deepcopy(value["branch_stability"])}
            for value in materialized
        ],
        "catalogs": catalogs,
        "alias_policy": "best_attempt_on_zero_success_best_nominal_only_on_hard_pass",
        "video_policy": "top_five_independent_full_reset_ffprobe_and_full_decode",
        "records": [],
    }
    return _commit_report(workspace, stage, report_path, payload, stage_input=stage_input, artifacts=tuple(dict.fromkeys(artifacts)))


def _manifest(config_path: Path, source: Any, *, seed: int) -> dict[str, Any]:
    base = build_contact_preserving_planned_lift_manifest(config_path, seed=int(seed))
    base.pop("campaign_input_sha256", None)
    base.update(
        {
            "contact_preserving_micro_jerk_campaign_schema_version": MICRO_JERK_CAMPAIGN_SCHEMA_VERSION,
            "campaign_kind": "contact_preserving_bounded_micro_jerk_rescue",
            "source_force_debias_campaign_path": str(_source_root(source)),
            "source_authentication": _mapping(source, label="micro source"),
            "source_evidence_sha256": _source_evidence_sha256(source),
            "source_center_set_sha256": canonical_sha256([_mapping(value, label="micro center") for value in _source_centers(source)]),
            "prior_physical_plan_set_sha256": canonical_sha256(_prior_physical(source)),
            "micro_jerk_budget": {
                "center_count": MICRO_CENTER_COUNT,
                "sensitivity_per_center": SENSITIVITY_COUNT_PER_CENTER,
                "trust_radii": list(TRUST_RADII),
                "trust_per_radius_per_center": TRUST_COUNT_PER_RADIUS_PER_CENTER,
                "candidate_count": MICRO_CANDIDATE_COUNT,
                "publication_candidate_count": PUBLISHED_CANDIDATE_COUNT,
                "candidate_execution": "fresh_full_reset_free_dynamics",
                "catalog_execution": "independent_top_five_fresh_full_reset_rerun",
                "global_physical_exclusion": True,
            },
        }
    )
    return {**base, "campaign_input_sha256": canonical_sha256(base)}


def build_contact_preserving_micro_jerk_rescue_manifest(
    config_path: str | Path, source_force_debias_campaign: str | Path, *, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    api = _micro_jerk_api()
    source = api.source_authenticator(Path(source_force_debias_campaign).expanduser().resolve())
    return _manifest(Path(config_path).expanduser().resolve(), source, seed=int(seed))


def run_contact_preserving_micro_jerk_rescue_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    source_force_debias_campaign: str | Path,
    resume: bool,
    target_success_count: int,
    workers: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if int(workers) <= 0:
        raise ValueError("workers must be positive")
    if int(seed) != DEFAULT_SEED:
        raise ValueError("micro-jerk campaign seed must be 20260821")
    if int(target_success_count) not in (1, 5):
        raise ValueError("target_success_count must be one or five")
    config_file = Path(config_path).expanduser().resolve()
    workspace = Path(output_dir).expanduser().resolve()
    source_root = Path(source_force_debias_campaign).expanduser().resolve()
    if workspace == source_root or workspace.is_relative_to(source_root):
        raise ValueError("micro workspace must be outside immutable source")
    definition = resolve_experiment(load_config(config_file))
    if getattr(definition, "contact_preserving_planned_lift_campaign", None) is None:
        raise ValueError("micro runner requires registered schema-v14 experiment")

    api = _micro_jerk_api()
    source = api.source_authenticator(source_root)
    _source_centers(source)
    manifest = _manifest(config_file, source, seed=int(seed))
    initialize_or_resume_campaign(workspace, manifest, resume=bool(resume))

    audit_path = workspace / "micro_jerk_source_audit.json"
    audit_payload = {
        "micro_jerk_source_audit_schema_version": 1,
        "complete": True,
        "source": _mapping(source, label="micro source"),
        "source_authentication_id": _source_id(source),
        "evidence_sha256": _source_evidence_sha256(source),
        "records": [],
    }
    audit_input = {
        "source_authentication_id": _source_id(source),
        "source_sha256": canonical_sha256(_mapping(source, label="micro source")),
    }
    audit = _load_committed_report(workspace, "micro_jerk_source_audit", audit_path)
    if audit is None:
        _commit_report(workspace, "micro_jerk_source_audit", audit_path, audit_payload, stage_input=audit_input)
    else:
        _assert_committed_stage_input(workspace, "micro_jerk_source_audit", audit_input)
        if canonical_sha256(audit) != canonical_sha256(audit_payload):
            raise RuntimeError("micro source audit changed on resume")

    jobs = _normalize_jobs(api.jobs_builder(source, total_count=MICRO_CANDIDATE_COUNT, seed=int(seed), local_index_offset=0, excluded_physical_plan_sha256=_prior_physical(source)), api=api, source=source)
    phase, phase_path = _execute_phase(workspace, jobs=jobs, source_id=_source_id(source), workers=int(workers))
    records = tuple(copy.deepcopy(dict(value)) for value in phase["records"])
    candidate_ids = tuple(int(value["candidate_id"]) for value in records)
    physical = tuple(_record_physical(value) for value in records)
    if len(set(candidate_ids)) != len(candidate_ids) or len(set(physical)) != len(physical):
        raise RuntimeError("micro phase lost global candidate/physical uniqueness")
    ranked = _rank_records(records, api)
    catalog = _catalog_stage(
        ranked,
        workspace,
        api=api,
        experiment_id=definition.experiment_id,
        target_success_count=int(target_success_count),
        refinement_report_sha256=file_sha256(phase_path),
    )
    full_count = sum(bool(value.get("full_success", False)) for value in ranked)
    result = {
        "contact_preserving_micro_jerk_rescue_result_schema_version": MICRO_JERK_RESULT_SCHEMA_VERSION,
        "complete": True,
        "experiment_id": definition.experiment_id,
        "source_force_debias_campaign": str(source_root),
        "source_authentication_id": _source_id(source),
        "target_success_count": int(target_success_count),
        "candidate_count": len(records),
        "candidate_id_unique_count": len(set(candidate_ids)),
        "physical_unique_candidate_count": len(set(physical)),
        "physical_duplicate_candidate_count": len(physical) - len(set(physical)),
        "physical_plan_set_sha256": canonical_sha256(sorted(set(physical))),
        "full_success_count": full_count,
        "target_reached": full_count >= int(target_success_count),
        "catalogs": copy.deepcopy(dict(catalog["catalogs"])),
        "published_candidate_ids": copy.deepcopy(list(catalog["published_candidate_ids"])),
        "fixed_mass_geometry_ablation": True,
        "exit_code": 0 if full_count else 2,
        "stop_reason": "target_full_success_count_reached" if full_count >= int(target_success_count) else "declared_1024_micro_jerk_budget_exhausted",
    }

    final_source = api.source_authenticator(source_root)
    if canonical_sha256(_mapping(final_source, label="micro source")) != canonical_sha256(_mapping(source, label="micro source")):
        raise RuntimeError("immutable micro source changed during execution")
    if canonical_sha256(_manifest(config_file, final_source, seed=int(seed))) != canonical_sha256(manifest):
        raise RuntimeError("micro implementation or inputs changed during execution")
    validate_stage_ledger(workspace)
    result_path = workspace / f"micro_jerk_rescue_result_target_{target_success_count}.json"
    result_stage = f"micro_jerk_result_target_{target_success_count}"
    result_input = {
        "refinement_report_sha256": file_sha256(phase_path),
        "catalog_report_sha256": file_sha256(workspace / "catalogs" / f"target_{target_success_count}" / "report.json"),
    }
    existing = _load_committed_report(workspace, result_stage, result_path)
    if existing is None:
        result = _commit_report(workspace, result_stage, result_path, result, stage_input=result_input)
    else:
        _assert_committed_stage_input(workspace, result_stage, result_input)
        if canonical_sha256(existing) != canonical_sha256(result):
            raise RuntimeError("committed micro result changed on resume")
        result = existing
    return result


__all__ = [
    "MICRO_CANDIDATE_COUNT",
    "build_contact_preserving_micro_jerk_rescue_manifest",
    "branch_stability_from_trace",
    "run_contact_preserving_micro_jerk_rescue_campaign",
]
