"""Authenticated post-campaign rescue for the schema-v14 planned lift.

The completed formal campaign is immutable input.  This runner authenticates
that workspace, creates a separate hash-bound campaign, executes the fixed
1,024-candidate refinement rescue, and only when that phase has no full
success executes the fixed 256-candidate time-warp rescue.  Every physics
candidate uses the same atomic full-reset artifact boundary as the original
v14 campaign; selected catalog entries are rerun from the initial no-contact
state before publication.
"""

from __future__ import annotations

import copy
import json
import multiprocessing
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ..actual_contact_grasp_pose_catalog import (
    authenticated_catalog_artifact_paths,
    commit_campaign_stage,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from ..artifacts import REPO_ROOT, file_sha256, write_json
from ..config import load_config
from ..experiment import resolve_experiment
from ..grasp_pose import canonical_sha256
from ..rendering import probe_video
from ..simulation import run_simulation
from .contact_preserving_candidate_artifacts import (
    authenticate_v14_candidate_artifacts,
)
from .contact_preserving_planned_lift_campaign import (
    CATALOG_SCHEMA_VERSION,
    _bind_catalog_entry_aliases,
    _catalog_entry_from_candidate,
    _copy_or_link,
    _default_robustness_runner,
    _publish_robust_alias_catalog,
    _run_full_reset_candidate,
    build_contact_preserving_planned_lift_manifest,
)


RESCUE_CAMPAIGN_SCHEMA_VERSION = 1
RESCUE_SOURCE_AUDIT_SCHEMA_VERSION = 1
RESCUE_PHASE_REPORT_SCHEMA_VERSION = 1
RESCUE_RESULT_SCHEMA_VERSION = 1
PHASE_ONE_CANDIDATE_COUNT = 1024
TIME_WARP_CANDIDATE_COUNT = 256
DEFAULT_SEED = 20260821


def _authenticate_source(source_campaign: Path):
    from .contact_preserving_refinement_rescue import (
        authenticate_formal_v2_rescue_source,
    )

    return authenticate_formal_v2_rescue_source(source_campaign)


def _select_phase_one_parents(source: Any):
    from .contact_preserving_refinement_rescue import (
        select_refinement_rescue_parents,
    )

    return select_refinement_rescue_parents(source, maximum_parent_count=8)


def _build_phase_one_jobs(parents: Sequence[Any]):
    from .contact_preserving_refinement_rescue import (
        RefinementRescueBudget,
        build_refinement_rescue_job_specs,
    )

    budget = RefinementRescueBudget(
        total_candidates=PHASE_ONE_CANDIDATE_COUNT,
        seed=DEFAULT_SEED,
    )
    return build_refinement_rescue_job_specs(parents, budget=budget)


def _rank_phase_one(records: Sequence[Mapping[str, Any]]):
    from .contact_preserving_refinement_rescue import (
        rank_refinement_rescue_results,
    )

    return rank_refinement_rescue_results(records)


def _build_time_warp_jobs(parent_records: Sequence[Mapping[str, Any]]):
    from .contact_preserving_time_warp import (
        TimeWarpBudget,
        authenticate_time_warp_job,
        build_contact_preserving_time_warp_jobs,
        stable_sort_time_warp_jobs,
    )

    jobs = build_contact_preserving_time_warp_jobs(
        parent_records,
        budget=TimeWarpBudget(
            total_candidate_count=TIME_WARP_CANDIDATE_COUNT,
            seed=DEFAULT_SEED,
        ),
    )
    ordered = stable_sort_time_warp_jobs(jobs)
    for job in ordered:
        authenticate_time_warp_job(job)
    return ordered


def _time_warp_manifest(parent_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from .contact_preserving_time_warp import TimeWarpBudget, time_warp_campaign_manifest

    return time_warp_campaign_manifest(
        parent_records,
        TimeWarpBudget(
            total_candidate_count=TIME_WARP_CANDIDATE_COUNT,
            seed=DEFAULT_SEED,
        ),
    )


def _source_descriptor(source: Any) -> dict[str, Any]:
    descriptor = getattr(source, "descriptor", None)
    if callable(descriptor):
        payload = descriptor()
        if not isinstance(payload, Mapping):
            raise RuntimeError("v14 rescue source descriptor is not a mapping")
        return copy.deepcopy(dict(payload))
    authentication_id = getattr(source, "source_authentication_id", None)
    if not isinstance(authentication_id, str):
        raise RuntimeError("v14 rescue source lost source_authentication_id")
    return {
        "source_authentication_id": authentication_id,
        "record_count": int(getattr(source, "record_count")),
        "manifest_sha256": str(getattr(source, "manifest_sha256")),
        "stage_ledger_sha256": str(getattr(source, "stage_ledger_sha256")),
    }


def _source_evidence_paths(source_campaign: Path, source: Any) -> tuple[Path, ...]:
    declared = getattr(source, "artifact_paths", ())
    paths: list[Path] = []
    if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
        paths.extend(Path(value).expanduser().resolve() for value in declared)
    for relative in (
        "campaign_manifest.json",
        "stage_ledger.json",
        "candidate_search/target_1.json",
        "joint_refinement/target_1.json",
        "campaign_result_target_1.json",
    ):
        candidate = (source_campaign / relative).resolve()
        if candidate.is_file():
            paths.append(candidate)
    unique = tuple(dict.fromkeys(paths))
    if not unique or any(not value.is_file() for value in unique):
        raise RuntimeError("v14 rescue source evidence paths are incomplete")
    return unique


def _manifest_from_authenticated_source(
    config_path: Path,
    source_campaign: Path,
    source: Any,
    *,
    seed: int,
    refinement_reuse: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base = build_contact_preserving_planned_lift_manifest(config_path, seed=seed)
    base.pop("campaign_input_sha256", None)
    descriptor = _source_descriptor(source)
    evidence = _source_evidence_paths(source_campaign, source)
    source_files = {
        str(path.relative_to(source_campaign)): file_sha256(path)
        for path in evidence
        if path.is_relative_to(source_campaign)
    }
    base.update(
        {
            "contact_preserving_rescue_manifest_schema_version": (
                RESCUE_CAMPAIGN_SCHEMA_VERSION
            ),
            "campaign_kind": "contact_preserving_post_campaign_rescue",
            "source_campaign_path": str(source_campaign),
            "source_campaign_authentication": descriptor,
            "source_campaign_evidence_sha256": source_files,
            "rescue_budget": {
                "phase_one_candidate_count": PHASE_ONE_CANDIDATE_COUNT,
                "time_warp_candidate_count": TIME_WARP_CANDIDATE_COUNT,
                "time_warp_only_when_phase_one_full_success_count_is_zero": True,
            },
        }
    )
    if refinement_reuse is not None:
        base["authenticated_refinement_reuse"] = copy.deepcopy(
            dict(refinement_reuse)
        )
    return {**base, "campaign_input_sha256": canonical_sha256(base)}


def build_contact_preserving_lift_rescue_manifest(
    config_path: str | Path,
    source_campaign: str | Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Authenticate and bind the immutable source into a new rescue manifest."""

    source_root = Path(source_campaign).expanduser().resolve()
    source = _authenticate_source(source_root)
    return _manifest_from_authenticated_source(
        Path(config_path).expanduser().resolve(),
        source_root,
        source,
        seed=int(seed),
    )


def _load_committed_report(
    workspace: Path, stage: str, path: Path
) -> dict[str, Any] | None:
    ledger = validate_stage_ledger(workspace)
    if stage not in ledger["stages"]:
        return None
    if not path.is_file():
        raise RuntimeError(f"committed rescue stage {stage} lost its report")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("complete") is not True:
        raise RuntimeError(f"committed rescue stage {stage} is incomplete")
    return copy.deepcopy(dict(payload))


def _validate_phase_report(
    report: Mapping[str, Any], *, expected_candidate_count: int, label: str
) -> None:
    records = report.get("records")
    if (
        report.get("complete") is not True
        or int(report.get("declared_candidate_count", -1))
        != int(expected_candidate_count)
        or int(report.get("candidate_count", -1)) != int(expected_candidate_count)
        or not isinstance(records, list)
        or len(records) != int(expected_candidate_count)
    ):
        raise RuntimeError(f"{label} did not exhaust its registered budget")
    identifiers = [int(value.get("candidate_id", -1)) for value in records]
    if min(identifiers, default=-1) < 0 or len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"{label} candidate identities are invalid")
    observed_full = sum(bool(value.get("full_success", False)) for value in records)
    if int(report.get("full_success_count", -1)) != observed_full:
        raise RuntimeError(f"{label} full-success count changed")


def _authenticate_refinement_reuse(
    reuse_root: Path,
    *,
    formal_source_root: Path,
    formal_source: Any,
    expected_jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate an old completed phase against freshly rebuilt job specs.

    The old report is never trusted as simulation evidence on its own.  Its
    committed ledger is checked first, then every persisted candidate bundle
    is authenticated against the newly generated candidate ID and complete
    resolved config.  This permits a source-code migration to reuse expensive
    immutable physics evidence without weakening resume hash checks.
    """

    root = reuse_root.expanduser().resolve()
    if not root.is_dir() or root == formal_source_root or root.is_relative_to(
        formal_source_root
    ):
        raise ValueError("refinement reuse must be a separate rescue workspace")
    ledger = validate_stage_ledger(root)
    if "refinement_rescue" not in ledger.get("stages", {}):
        raise RuntimeError("refinement reuse has no committed refinement stage")
    manifest_path = root / "campaign_manifest.json"
    ledger_path = root / "stage_ledger.json"
    report_path = root / "refinement_rescue" / "report.json"
    if not all(path.is_file() for path in (manifest_path, ledger_path, report_path)):
        raise RuntimeError("refinement reuse lost its manifest, ledger, or report")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_phase_report(
        report,
        expected_candidate_count=PHASE_ONE_CANDIDATE_COUNT,
        label="reused v14 refinement rescue",
    )
    source_descriptor = _source_descriptor(formal_source)
    old_source = manifest.get("source_campaign_authentication")
    if (
        manifest.get("campaign_kind") != "contact_preserving_post_campaign_rescue"
        or manifest.get("experiment_id") != source_descriptor.get("experiment_id")
        or not isinstance(old_source, Mapping)
        or old_source.get("source_authentication_id")
        != source_descriptor.get("source_authentication_id")
        or Path(str(manifest.get("source_campaign_path", ""))).resolve()
        != formal_source_root
    ):
        raise RuntimeError("refinement reuse is bound to a different formal source")
    if len(expected_jobs) != PHASE_ONE_CANDIDATE_COUNT:
        raise RuntimeError("current refinement job builder changed its fixed budget")
    jobs_by_id = {int(job["candidate_id"]): job for job in expected_jobs}
    if len(jobs_by_id) != PHASE_ONE_CANDIDATE_COUNT:
        raise RuntimeError("current refinement jobs contain duplicate identities")
    records_by_id = {int(record["candidate_id"]): record for record in report["records"]}
    if set(records_by_id) != set(jobs_by_id):
        raise RuntimeError("reused refinement candidate IDs differ from current jobs")

    bundles: dict[int, Any] = {}
    evidence: list[dict[str, Any]] = []
    for candidate_id in sorted(jobs_by_id):
        job = jobs_by_id[candidate_id]
        record = records_by_id[candidate_id]
        relative = record.get("artifact_directory")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise RuntimeError("reused refinement artifact path is unsafe")
        candidate_root = (root / relative).resolve()
        if not candidate_root.is_relative_to(root / "refinement_rescue" / "candidates"):
            raise RuntimeError("reused refinement artifact escaped its candidate root")
        bundle = authenticate_v14_candidate_artifacts(
            candidate_root,
            expected_config=job["config"],
            expected_candidate_id=candidate_id,
            expected_retain_grasp_success=False,
        )
        for key, value in bundle.result.items():
            if key not in record or canonical_sha256(record[key]) != canonical_sha256(value):
                raise RuntimeError(
                    f"reused refinement report changed candidate {candidate_id} field {key}"
                )
        bundles[candidate_id] = bundle
        evidence.append(
            {
                "candidate_id": candidate_id,
                "config_sha256": canonical_sha256(job["config"]),
                "result_semantic_sha256": bundle.result["result_semantic_sha256"],
            }
        )
    descriptor_payload = {
        "refinement_reuse_schema_version": 1,
        "source_workspace": str(root),
        "source_campaign_input_sha256": manifest.get("campaign_input_sha256"),
        "source_code_sha256": manifest.get("source_sha256"),
        "formal_source_authentication_id": source_descriptor.get(
            "source_authentication_id"
        ),
        "manifest_sha256": file_sha256(manifest_path),
        "ledger_sha256": file_sha256(ledger_path),
        "refinement_report_sha256": file_sha256(report_path),
        "candidate_count": len(evidence),
        "candidate_evidence_sha256": canonical_sha256(evidence),
        "fully_reauthenticated_against_current_jobs": True,
        "read_only": True,
    }
    return {
        "root": root,
        "descriptor": {
            **descriptor_payload,
            "refinement_reuse_id": canonical_sha256(descriptor_payload),
        },
        "bundles": bundles,
    }


def _import_reused_refinement_candidates(
    reuse: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    workspace: Path,
) -> None:
    """Atomically materialize authenticated bundles in a new workspace."""

    bundles = reuse.get("bundles")
    if not isinstance(bundles, Mapping):
        raise RuntimeError("authenticated refinement reuse lost its bundles")
    destination_root = workspace / "refinement_rescue" / "candidates"
    destination_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        candidate_id = int(job["candidate_id"])
        source_bundle = bundles.get(candidate_id)
        if source_bundle is None:
            raise RuntimeError(f"refinement reuse lost candidate {candidate_id}")
        destination = destination_root / f"candidate_{candidate_id}"
        if destination.exists():
            authenticate_v14_candidate_artifacts(
                destination,
                expected_config=job["config"],
                expected_candidate_id=candidate_id,
                expected_retain_grasp_success=False,
            )
            continue
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".candidate_{candidate_id}.reuse-staging.",
                dir=destination_root,
            )
        )
        try:
            for source_path in source_bundle.artifact_paths:
                target = staging / source_path.name
                try:
                    os.link(source_path, target)
                except OSError:
                    shutil.copy2(source_path, target)
            authenticate_v14_candidate_artifacts(
                staging,
                expected_config=job["config"],
                expected_candidate_id=candidate_id,
                expected_retain_grasp_success=False,
            )
            staging.rename(destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _time_warp_required(phase_one_report: Mapping[str, Any]) -> bool:
    """Return the registered gate after authenticating the whole first phase."""

    _validate_phase_report(
        phase_one_report,
        expected_candidate_count=PHASE_ONE_CANDIDATE_COUNT,
        label="v14 refinement rescue",
    )
    return int(phase_one_report["full_success_count"]) == 0


def _commit_report(
    workspace: Path,
    stage: str,
    path: Path,
    payload: Mapping[str, Any],
    *,
    stage_input: Mapping[str, Any],
    artifacts: Sequence[Path] = (),
) -> dict[str, Any]:
    write_json(path, payload)
    commit_campaign_stage(
        workspace,
        stage,
        stage_input=stage_input,
        artifacts=(path, *artifacts),
        summary={
            "complete": True,
            "record_count": len(payload.get("records", ())),
            "full_success_count": int(payload.get("full_success_count", 0)),
        },
    )
    return copy.deepcopy(dict(payload))


def _run_candidate_job(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _run_full_reset_candidate(
        payload["config"],
        Path(str(payload["destination"])),
        int(payload["candidate_id"]),
        retain_grasp_success=False,
    )
    return copy.deepcopy(dict(result))


def _job_sequence(job: Mapping[str, Any], fallback: int) -> int:
    if "job_sequence_index" in job:
        return int(job["job_sequence_index"])
    metadata = job.get("job_metadata", {})
    if isinstance(metadata, Mapping) and "job_sequence_index" in metadata:
        return int(metadata["job_sequence_index"])
    return int(fallback)


def _execute_candidate_jobs(
    jobs: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    phase_name: str,
    workers: int,
    global_rank_offset: int,
) -> tuple[tuple[dict[str, Any], ...], tuple[Path, ...]]:
    normalized = [copy.deepcopy(dict(value)) for value in jobs]
    identifiers = [int(value["candidate_id"]) for value in normalized]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"{phase_name} candidate IDs are not unique")
    ordered = sorted(
        enumerate(normalized),
        key=lambda pair: (_job_sequence(pair[1], pair[0]), int(pair[1]["candidate_id"])),
    )
    payloads: list[dict[str, Any]] = []
    ordered_jobs: list[dict[str, Any]] = []
    for fallback, job in ordered:
        candidate_id = int(job["candidate_id"])
        config = job.get("config")
        if not isinstance(config, Mapping) or int(config.get("schema_version", 0)) != 14:
            raise RuntimeError(f"{phase_name} job {candidate_id} lost its v14 config")
        destination = workspace / phase_name / f"candidate_{candidate_id}"
        payloads.append(
            {
                "config": copy.deepcopy(dict(config)),
                "candidate_id": candidate_id,
                "destination": str(destination),
            }
        )
        ordered_jobs.append(job)
    if int(workers) == 1:
        raw = tuple(_run_candidate_job(value) for value in payloads)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as pool:
            raw = tuple(pool.map(_run_candidate_job, payloads))
    records: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    for sequence, (job, result, payload) in enumerate(
        zip(ordered_jobs, raw, payloads, strict=True)
    ):
        candidate_id = int(job["candidate_id"])
        if int(result.get("candidate_id", -1)) != candidate_id:
            raise RuntimeError(f"{phase_name} returned the wrong candidate ID")
        root = Path(payload["destination"]).resolve()
        bundle = authenticate_v14_candidate_artifacts(
            root,
            expected_config=job["config"],
            expected_candidate_id=candidate_id,
            expected_retain_grasp_success=False,
        )
        artifacts.extend(bundle.artifact_paths)
        metadata = copy.deepcopy(dict(job.get("job_metadata", {})))
        records.append(
            {
                **copy.deepcopy(bundle.result),
                "artifact_directory": str(root.relative_to(workspace)),
                "rescue_stage": phase_name,
                "rescue_job": metadata,
                "plan_rank": int(global_rank_offset + sequence),
                "feedback_index": 0,
                "feedback_id": str(
                    job.get("candidate_sha256", canonical_sha256(metadata))
                ),
            }
        )
    return tuple(records), tuple(dict.fromkeys(artifacts))


def _records_with_configs(
    records: Sequence[Mapping[str, Any]], workspace: Path
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for record in records:
        root = (workspace / str(record["artifact_directory"])).resolve()
        if not root.is_relative_to(workspace):
            raise RuntimeError("rescue candidate path escaped its workspace")
        bundle = authenticate_v14_candidate_artifacts(
            root,
            expected_candidate_id=int(record["candidate_id"]),
            expected_retain_grasp_success=False,
        )
        config = json.loads(bundle.config_path.read_text(encoding="utf-8"))
        result.append({**copy.deepcopy(dict(record)), "config": config})
    return tuple(result)


def _select_time_warp_parent_records(
    records: Sequence[Mapping[str, Any]], workspace: Path
) -> tuple[dict[str, Any], ...]:
    """Fence the legacy time-warp builder to the rescue-ranked top four."""

    return tuple(_rank_phase_one(_records_with_configs(records, workspace))[:4])


def _selected_rescue_catalog_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Select top five using the rescue order, retaining chronological first pass."""

    ranked = list(_rank_phase_one(records))
    first_success = min(
        (value for value in records if bool(value.get("full_success", False))),
        key=lambda value: (
            int(value.get("plan_rank", 1 << 30)),
            int(value.get("candidate_id", 1 << 62)),
        ),
        default=None,
    )
    selected = [copy.deepcopy(dict(value)) for value in ranked[:5]]
    if first_success is not None and all(
        int(value["candidate_id"]) != int(first_success["candidate_id"])
        for value in selected
    ):
        if selected:
            selected[-1] = copy.deepcopy(dict(first_success))
        else:
            selected.append(copy.deepcopy(dict(first_success)))
        selected = list(_rank_phase_one(selected))
    return tuple(selected)


def _materialize_rescue_catalog_traces(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    target_success_count: int,
) -> tuple[dict[str, Any], ...]:
    """Full-reset rerun exactly the rescue-ranked records selected for publication."""

    materialized: list[dict[str, Any]] = []
    for record in _selected_rescue_catalog_records(records):
        candidate_id = int(record["candidate_id"])
        root = (workspace / str(record["artifact_directory"])).resolve()
        trace_name = record.get("artifacts", {}).get("trace")
        if isinstance(trace_name, str) and (root / trace_name).is_file():
            materialized.append(copy.deepcopy(dict(record)))
            continue
        config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
        final_root = (
            workspace
            / "catalog_source_reruns"
            / f"rescue_target_{target_success_count}"
            / f"candidate_{candidate_id}"
        )
        rerun = _run_full_reset_candidate(
            config, final_root, candidate_id, final_rerun=True
        )
        if canonical_sha256(rerun.get("summary")) != canonical_sha256(
            record.get("summary")
        ):
            raise RuntimeError("rescue catalog full-reset rerun changed its summary")
        if (
            bool(rerun.get("grasp_success", False))
            != bool(record.get("grasp_success", False))
            or bool(rerun.get("full_success", False))
            != bool(record.get("full_success", False))
        ):
            raise RuntimeError("rescue catalog full-reset rerun changed its status")
        materialized.append(
            {
                **copy.deepcopy(dict(record)),
                **rerun,
                "artifact_directory": str(final_root.relative_to(workspace)),
            }
        )
    return tuple(materialized)


def _best_pair_fields(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: entry.get(key)
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


def _publish_rescue_viewer_catalogs(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    destination: Path,
    *,
    experiment_id: str,
) -> dict[str, str]:
    """Publish rescue-ranked Viewer catalogs without changing legacy ranking."""

    catalog_root = destination / "manipulation"
    if catalog_root.exists():
        if catalog_root.is_symlink() or catalog_root.resolve().parent != destination.resolve():
            raise RuntimeError("unsafe stale rescue manipulation catalog")
        shutil.rmtree(catalog_root)
    catalog_root.mkdir(parents=True, exist_ok=False)
    entries = [
        _catalog_entry_from_candidate(
            record,
            workspace,
            catalog_root,
            index + 1,
            render_video=True,
            simulation_runner=run_simulation,
            video_probe=probe_video,
        )
        for index, record in enumerate(records)
    ]
    aliases = {
        f"pair_rank_{index + 1:02d}": entry["trajectory_id"]
        for index, entry in enumerate(entries)
    }
    successes = [value for value in entries if value["classification"] == "success"]
    first_success_record = min(
        (value for value in records if bool(value.get("full_success", False))),
        key=lambda value: (
            int(value.get("plan_rank", 1 << 30)),
            int(value["candidate_id"]),
        ),
        default=None,
    )
    if successes:
        first = next(
            (
                value
                for value in successes
                if first_success_record is not None
                and int(value["candidate_id"])
                == int(first_success_record["candidate_id"])
            ),
            successes[0],
        )
        aliases["best_first"] = first["trajectory_id"]
        aliases["best_nominal"] = successes[0]["trajectory_id"]
    elif entries:
        aliases["best_attempt"] = entries[0]["trajectory_id"]
    _bind_catalog_entry_aliases(entries, aliases)
    by_id = {value["trajectory_id"]: value for value in entries}
    catalog = {
        "contact_preserving_viewer_catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "trajectory_catalog_schema_version": 1,
        "catalog_kind": "manipulation",
        "complete": True,
        "experiment_id": experiment_id,
        "selection_policy": (
            "full_success_then_refinement_rescue_contact_and_smoothness_rank"
        ),
        "production_trajectory_video_policy": (
            "deterministic_full_reset_rerun_ffprobe_and_full_decode"
        ),
        "success_count": len(successes),
        "aliases": aliases,
        "best_grasp_object_pairs": {
            alias: _best_pair_fields(by_id[trajectory])
            for alias, trajectory in aliases.items()
            if alias in {"best_first", "best_nominal", "best_attempt"}
        },
        "trajectories": entries,
    }
    manipulation_path = catalog_root / "catalog.json"
    write_json(manipulation_path, catalog)

    grasp_entries = [copy.deepcopy(value) for value in entries if value["grasp_success"]]
    grasp_success_count = len(grasp_entries)
    if not grasp_entries and entries:
        grasp_entries = [copy.deepcopy(entries[0])]
    grasp_root = destination / "grasp_pose"
    if grasp_root.exists():
        if grasp_root.is_symlink() or grasp_root.resolve().parent != destination.resolve():
            raise RuntimeError("unsafe stale rescue grasp catalog")
        shutil.rmtree(grasp_root)
    grasp_root.mkdir(parents=True, exist_ok=False)
    for entry in grasp_entries:
        entry["classification"] = "success" if entry["grasp_success"] else "diagnostic"
        for artifact_name in ("resolved_config", "result", "trace", "video"):
            relative = entry["artifacts"].get(artifact_name)
            if relative is not None:
                _copy_or_link(catalog_root / relative, grasp_root / relative)
    grasp_aliases = {
        f"pair_rank_{index + 1:02d}": entry["trajectory_id"]
        for index, entry in enumerate(grasp_entries)
    }
    if grasp_success_count:
        grasp_aliases["best_first"] = grasp_entries[0]["trajectory_id"]
        grasp_aliases["best_nominal"] = grasp_entries[0]["trajectory_id"]
    elif grasp_entries:
        grasp_aliases["best_attempt"] = grasp_entries[0]["trajectory_id"]
    _bind_catalog_entry_aliases(grasp_entries, grasp_aliases)
    grasp_path = grasp_root / "catalog.json"
    write_json(
        grasp_path,
        {
            "contact_preserving_viewer_catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "trajectory_catalog_schema_version": 1,
            "catalog_kind": "grasp_pose",
            "complete": True,
            "experiment_id": experiment_id,
            "production_trajectory_video_policy": catalog[
                "production_trajectory_video_policy"
            ],
            "success_count": grasp_success_count,
            "aliases": grasp_aliases,
            "trajectories": grasp_entries,
        },
    )
    return {
        "grasp_pose": str(grasp_path.relative_to(workspace)),
        "manipulation": str(manipulation_path.relative_to(workspace)),
    }


def _catalog_stage(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    experiment_id: str,
    target_success_count: int,
    source_report_hashes: Mapping[str, str],
) -> dict[str, Any]:
    stage = f"rescue_catalog_target_{target_success_count}"
    report_path = workspace / "catalogs" / f"target_{target_success_count}" / "report.json"
    existing = _load_committed_report(workspace, stage, report_path)
    if existing is not None:
        return existing
    materialized = _materialize_rescue_catalog_traces(
        records, workspace, target_success_count=target_success_count
    )
    catalogs = _publish_rescue_viewer_catalogs(
        materialized,
        workspace,
        report_path.parent,
        experiment_id=experiment_id,
    )
    artifacts: list[Path] = []
    for relative in catalogs.values():
        artifacts.extend(authenticated_catalog_artifact_paths(workspace / relative))
    full_count = sum(bool(value.get("full_success", False)) for value in records)
    payload = {
        "rescue_catalog_report_schema_version": 1,
        "complete": True,
        "target_success_count": target_success_count,
        "full_success_count": full_count,
        "target_reached": full_count >= target_success_count,
        "catalogs": catalogs,
    }
    return _commit_report(
        workspace,
        stage,
        report_path,
        payload,
        stage_input={
            **dict(source_report_hashes),
            "target_success_count": target_success_count,
        },
        artifacts=tuple(dict.fromkeys(artifacts)),
    )


def run_contact_preserving_lift_rescue_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    source_campaign: str | Path,
    reuse_refinement_from: str | Path | None = None,
    resume: bool,
    target_success_count: int,
    workers: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run/resume the fixed two-phase v14 rescue and publish final evidence."""

    if int(workers) <= 0:
        raise ValueError("workers must be positive")
    if int(seed) != DEFAULT_SEED:
        raise ValueError("schema-v14 rescue seed must be 20260821")
    if int(target_success_count) not in (1, 5):
        raise ValueError("target_success_count must be one or five")
    config_file = Path(config_path).expanduser().resolve()
    workspace = Path(output_dir).expanduser().resolve()
    source_root = Path(source_campaign).expanduser().resolve()
    if workspace == source_root or workspace.is_relative_to(source_root):
        raise ValueError("rescue workspace must be outside its immutable source campaign")
    reuse_root = (
        None
        if reuse_refinement_from is None
        else Path(reuse_refinement_from).expanduser().resolve()
    )
    if reuse_root is not None and (
        workspace == reuse_root or workspace.is_relative_to(reuse_root)
    ):
        raise ValueError("rescue workspace must be outside its refinement reuse source")
    template = load_config(config_file)
    definition = resolve_experiment(template)
    if getattr(definition, "contact_preserving_planned_lift_campaign", None) is None:
        raise ValueError("rescue runner requires the registered schema-v14 experiment")

    # Re-authenticate immutable source artifacts before every fresh or resumed
    # execution.  No output directory is created until this succeeds.
    source = _authenticate_source(source_root)
    prebuilt_phase_one_jobs: tuple[dict[str, Any], ...] | None = None
    refinement_reuse: dict[str, Any] | None = None
    if reuse_root is not None:
        parents = _select_phase_one_parents(source)
        prebuilt_phase_one_jobs = tuple(_build_phase_one_jobs(parents))
        refinement_reuse = _authenticate_refinement_reuse(
            reuse_root,
            formal_source_root=source_root,
            formal_source=source,
            expected_jobs=prebuilt_phase_one_jobs,
        )
    refinement_reuse_descriptor = (
        None
        if refinement_reuse is None
        else copy.deepcopy(dict(refinement_reuse["descriptor"]))
    )
    manifest = _manifest_from_authenticated_source(
        config_file,
        source_root,
        source,
        seed=int(seed),
        refinement_reuse=refinement_reuse_descriptor,
    )
    initialize_or_resume_campaign(workspace, manifest, resume=bool(resume))
    source_descriptor = _source_descriptor(source)
    source_audit_path = workspace / "source_campaign_audit.json"
    source_audit = _load_committed_report(
        workspace, "rescue_source_audit", source_audit_path
    )
    source_payload = {
        "rescue_source_audit_schema_version": RESCUE_SOURCE_AUDIT_SCHEMA_VERSION,
        "complete": True,
        "source_campaign": str(source_root),
        "authentication": source_descriptor,
        "evidence_sha256": {
            str(value.relative_to(source_root)): file_sha256(value)
            for value in _source_evidence_paths(source_root, source)
            if value.is_relative_to(source_root)
        },
    }
    if source_audit is None:
        source_audit = _commit_report(
            workspace,
            "rescue_source_audit",
            source_audit_path,
            source_payload,
            stage_input={"source_authentication": source_descriptor},
        )
    elif canonical_sha256(source_audit) != canonical_sha256(source_payload):
        raise RuntimeError("immutable rescue source authentication changed on resume")

    phase_one_path = workspace / "refinement_rescue" / "report.json"
    phase_one = _load_committed_report(
        workspace, "refinement_rescue", phase_one_path
    )
    if phase_one is None:
        if prebuilt_phase_one_jobs is None:
            parents = _select_phase_one_parents(source)
            jobs = tuple(_build_phase_one_jobs(parents))
        else:
            jobs = prebuilt_phase_one_jobs
        if len(jobs) != PHASE_ONE_CANDIDATE_COUNT:
            raise RuntimeError("v14 refinement rescue did not build exactly 1024 jobs")
        if refinement_reuse is not None:
            _import_reused_refinement_candidates(
                refinement_reuse, jobs, workspace
            )
        records, artifacts = _execute_candidate_jobs(
            jobs,
            workspace,
            phase_name="refinement_rescue/candidates",
            workers=int(workers),
            global_rank_offset=0,
        )
        ranked = tuple(_rank_phase_one(records))
        if {int(value["candidate_id"]) for value in ranked} != {
            int(value["candidate_id"]) for value in records
        }:
            raise RuntimeError("v14 refinement rescue ranking lost candidates")
        phase_one = _commit_report(
            workspace,
            "refinement_rescue",
            phase_one_path,
            {
                "refinement_rescue_report_schema_version": RESCUE_PHASE_REPORT_SCHEMA_VERSION,
                "complete": True,
                "declared_candidate_count": PHASE_ONE_CANDIDATE_COUNT,
                "candidate_count": len(ranked),
                "full_success_count": sum(
                    bool(value.get("full_success", False)) for value in ranked
                ),
                "source_authentication_id": source_descriptor.get(
                    "source_authentication_id"
                ),
                "records": list(ranked),
            },
            stage_input={
                "source_audit_sha256": file_sha256(source_audit_path),
                "candidate_count": PHASE_ONE_CANDIDATE_COUNT,
                "jobs_sha256": canonical_sha256(jobs),
                "authenticated_refinement_reuse": refinement_reuse_descriptor,
            },
            artifacts=artifacts,
        )
    phase_one_records = tuple(phase_one["records"])
    # Candidate bundle objects carry full semantic summaries.  The newly
    # committed phase and its ledger are now authoritative inside this
    # workspace, so release the import cache before the physics-heavy warp
    # stage.  The immutable source is authenticated again before final commit.
    refinement_reuse = None
    _validate_phase_report(
        phase_one,
        expected_candidate_count=PHASE_ONE_CANDIDATE_COUNT,
        label="v14 refinement rescue",
    )
    phase_one_full = int(phase_one["full_success_count"])

    time_warp: dict[str, Any] | None = None
    time_warp_records: tuple[Mapping[str, Any], ...] = ()
    time_warp_path = workspace / "time_warp_rescue" / "report.json"
    if _time_warp_required(phase_one):
        time_warp = _load_committed_report(
            workspace, "time_warp_rescue", time_warp_path
        )
        if time_warp is None:
            parents = _select_time_warp_parent_records(
                phase_one_records, workspace
            )
            warp_manifest = _time_warp_manifest(parents)
            jobs = tuple(_build_time_warp_jobs(parents))
            if len(jobs) != TIME_WARP_CANDIDATE_COUNT:
                raise RuntimeError("v14 time-warp rescue did not build exactly 256 jobs")
            records, artifacts = _execute_candidate_jobs(
                jobs,
                workspace,
                phase_name="time_warp_rescue/candidates",
                workers=int(workers),
                global_rank_offset=PHASE_ONE_CANDIDATE_COUNT,
            )
            ranked = _rank_phase_one(records)
            time_warp = _commit_report(
                workspace,
                "time_warp_rescue",
                time_warp_path,
                {
                    "time_warp_rescue_report_schema_version": RESCUE_PHASE_REPORT_SCHEMA_VERSION,
                    "complete": True,
                    "declared_candidate_count": TIME_WARP_CANDIDATE_COUNT,
                    "candidate_count": len(ranked),
                    "full_success_count": sum(
                        bool(value.get("full_success", False)) for value in ranked
                    ),
                    "time_warp_manifest": warp_manifest,
                    "records": list(ranked),
                },
                stage_input={
                    "phase_one_report_sha256": file_sha256(phase_one_path),
                    "time_warp_manifest": warp_manifest,
                    "jobs_sha256": canonical_sha256(jobs),
                },
                artifacts=artifacts,
            )
        time_warp_records = tuple(time_warp["records"])
        _validate_phase_report(
            time_warp,
            expected_candidate_count=TIME_WARP_CANDIDATE_COUNT,
            label="v14 time-warp rescue",
        )
    elif "time_warp_rescue" in validate_stage_ledger(workspace)["stages"]:
        raise RuntimeError(
            "time-warp stage exists although phase one already has a full success"
        )

    combined = _rank_phase_one(
        (*phase_one_records, *time_warp_records)
    )
    full_count = sum(bool(value.get("full_success", False)) for value in combined)
    catalog_report = _catalog_stage(
        combined,
        workspace,
        experiment_id=definition.experiment_id,
        target_success_count=int(target_success_count),
        source_report_hashes={
            "phase_one_report_sha256": file_sha256(phase_one_path),
            "time_warp_report_sha256": (
                file_sha256(time_warp_path) if time_warp is not None else "not_run"
            ),
        },
    )
    catalogs = copy.deepcopy(dict(catalog_report["catalogs"]))

    robustness: dict[str, Any] | None = None
    if full_count:
        robustness_stage = f"rescue_robustness_target_{target_success_count}"
        robustness_path = (
            workspace
            / "robustness"
            / f"target_{target_success_count}"
            / "perturbation_report.json"
        )
        robustness = _load_committed_report(
            workspace, robustness_stage, robustness_path
        )
        if robustness is None:
            raw = _default_robustness_runner(
                workspace / catalogs["manipulation"],
                robustness_path,
                workers=int(workers),
                seed=int(seed),
                campaign=definition.contact_preserving_planned_lift_campaign,
            )
            payload = {"complete": True, **copy.deepcopy(dict(raw))}
            artifacts = tuple(
                value
                for value in sorted(robustness_path.parent.rglob("*"))
                if value.is_file() and value != robustness_path
            )
            robustness = _commit_report(
                workspace,
                robustness_stage,
                robustness_path,
                payload,
                stage_input={
                    "catalog_sha256": file_sha256(
                        workspace / catalogs["manipulation"]
                    ),
                    "required_passes": int(
                        definition.contact_preserving_planned_lift_campaign.robustness_required_passes
                    ),
                },
                artifacts=artifacts,
            )
        if robustness.get("robust_success") is True:
            robust_stage = f"rescue_robust_catalog_target_{target_success_count}"
            robust_report_path = (
                workspace / "catalogs" / f"target_{target_success_count}" / "robust_report.json"
            )
            robust_report = _load_committed_report(
                workspace, robust_stage, robust_report_path
            )
            if robust_report is None:
                nominal = workspace / catalogs["manipulation"]
                robust_catalog = _publish_robust_alias_catalog(
                    nominal,
                    nominal.parent.parent / "robust_manipulation",
                    robust_candidate_id=str(robustness["best_50_candidate_id"]),
                )
                robust_report = _commit_report(
                    workspace,
                    robust_stage,
                    robust_report_path,
                    {
                        "complete": True,
                        "catalog": str(robust_catalog.relative_to(workspace)),
                        "best_robust_alias": "best_robust",
                        "records": [],
                    },
                    stage_input={
                        "nominal_catalog_sha256": file_sha256(nominal),
                        "robustness_report_sha256": file_sha256(robustness_path),
                    },
                    artifacts=authenticated_catalog_artifact_paths(robust_catalog),
                )
            catalogs["robust"] = robust_report["catalog"]

    result = {
        "contact_preserving_rescue_result_schema_version": RESCUE_RESULT_SCHEMA_VERSION,
        "complete": True,
        "experiment_id": definition.experiment_id,
        "source_campaign": str(source_root),
        "source_authentication_id": source_descriptor.get("source_authentication_id"),
        "authenticated_refinement_reuse": refinement_reuse_descriptor,
        "target_success_count": int(target_success_count),
        "phase_one_candidate_count": len(phase_one_records),
        "phase_one_full_success_count": phase_one_full,
        "time_warp_triggered": phase_one_full == 0,
        "time_warp_candidate_count": len(time_warp_records),
        "time_warp_full_success_count": sum(
            bool(value.get("full_success", False)) for value in time_warp_records
        ),
        "full_success_count": full_count,
        "target_reached": full_count >= int(target_success_count),
        "catalogs": catalogs,
        "robustness": copy.deepcopy(robustness),
        "fixed_mass_geometry_ablation": True,
        "stop_reason": (
            "target_full_success_count_reached"
            if full_count >= int(target_success_count)
            else "declared_refinement_and_time_warp_rescue_exhausted"
        ),
    }
    if reuse_root is not None:
        if prebuilt_phase_one_jobs is None:
            raise RuntimeError("refinement reuse lost its rebuilt job set")
        final_reuse = _authenticate_refinement_reuse(
            reuse_root,
            formal_source_root=source_root,
            formal_source=source,
            expected_jobs=prebuilt_phase_one_jobs,
        )
        if canonical_sha256(final_reuse["descriptor"]) != canonical_sha256(
            refinement_reuse_descriptor
        ):
            raise RuntimeError("immutable refinement reuse changed during execution")
    final_manifest = _manifest_from_authenticated_source(
        config_file,
        source_root,
        source,
        seed=int(seed),
        refinement_reuse=refinement_reuse_descriptor,
    )
    if canonical_sha256(final_manifest) != canonical_sha256(manifest):
        raise RuntimeError("v14 rescue implementation or source changed during execution")
    validate_stage_ledger(workspace)
    result_path = workspace / f"rescue_result_target_{target_success_count}.json"
    result_stage = f"rescue_result_target_{target_success_count}"
    existing = _load_committed_report(workspace, result_stage, result_path)
    if existing is None:
        result = _commit_report(
            workspace,
            result_stage,
            result_path,
            result,
            stage_input={
                "catalog_report_sha256": file_sha256(
                    workspace / "catalogs" / f"target_{target_success_count}" / "report.json"
                ),
                "robustness_sha256": (
                    canonical_sha256(robustness) if robustness is not None else "not_run"
                ),
            },
        )
    elif canonical_sha256(existing) != canonical_sha256(result):
        raise RuntimeError("committed v14 rescue result changed on resume")
    return result


__all__ = [
    "DEFAULT_SEED",
    "PHASE_ONE_CANDIDATE_COUNT",
    "RESCUE_CAMPAIGN_SCHEMA_VERSION",
    "TIME_WARP_CANDIDATE_COUNT",
    "build_contact_preserving_lift_rescue_manifest",
    "run_contact_preserving_lift_rescue_campaign",
]
