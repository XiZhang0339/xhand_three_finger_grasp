"""Resumable schema-v13 fixed-scale-hand / smaller-cube campaign.

The numerical point continuation is owned by :mod:`contact_point_downsize`.
This module owns the part that must remain boring and auditable: deterministic
strata, authenticated source evidence, power-loss-safe stage commits, dynamic
grasp/finalization dispatch, manipulation dispatch, and two independent Viewer
catalogs.  Expensive boundaries are injectable so orchestration tests do not
replace any evidence checks with synthetic success in production code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..actual_contact_grasp_pose_catalog import (
    authenticated_catalog_artifact_paths,
    build_campaign_manifest,
    commit_campaign_stage,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from ..actual_contact_selection import (
    final_candidate_rank,
    full_success as catalog_full_success,
    grasp_success as catalog_grasp_success,
)
from ..artifacts import REPO_ROOT, file_sha256, implementation_paths, write_json
from ..config import load_config, validate_config
from ..experiment import ExperimentDefinition, resolve_experiment
from ..grasp_pose import canonical_sha256, controller_id, grasp_pose_id
from .actual_contact_grasp_pose_dynamic import (
    grasp_stage_succeeded,
    rank_dynamic_grasp_results,
    run_actual_contact_dynamic_grasp_candidates,
)
from .actual_contact_grasp_pose_measured import (
    measured_grasp_pose_succeeded,
    run_or_resume_measured_grasp_finalization,
)
from .scaled_contact_downsize_catalog import (
    export_scaled_contact_downsize_grasp_catalog,
    export_scaled_contact_downsize_manipulation_catalog,
)
from .scaled_contact_downsize_ranking import v13_manipulation_candidate_rank


CAMPAIGN_RESULT_SCHEMA_VERSION = 1
SOURCE_AUDIT_SCHEMA_VERSION = 1
STATIC_REPORT_SCHEMA_VERSION = 1
CATALOG_METADATA_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821


def _campaign(definition: ExperimentDefinition):
    value = getattr(definition, "scaled_contact_downsize_campaign", None)
    if value is None:
        raise ValueError("experiment has no scaled_contact_downsize_campaign")
    return value


def _repo_json_path(value: str, label: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if not path.is_relative_to(REPO_ROOT / "artifacts") or not path.is_file():
        raise FileNotFoundError(f"{label} is missing below artifacts/: {path}")
    return path


def _catalog_member(catalog: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or Path(value).is_absolute():
        raise RuntimeError(f"source catalog has no safe {label} member")
    path = (catalog.parent / value).resolve()
    if not path.is_relative_to(catalog.parent) or not path.is_file():
        raise RuntimeError(f"source catalog lost {label}: {value}")
    return path


def _audit_source_catalog_files(
    catalog_path: Path, aliases: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    """Fail closed on aliases and all three source evidence members.

    This cheap audit deliberately duplicates only catalog/path/hash handling;
    stable-window physics is rechecked by ``audit_downsize_source`` before DLS.
    """

    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("trajectories"), list
    ):
        raise RuntimeError("scaled-contact source catalog is malformed")
    alias_map = payload.get("aliases")
    if not isinstance(alias_map, Mapping):
        raise RuntimeError("scaled-contact source catalog has no alias mapping")
    trajectories = {
        str(value.get("trajectory_id")): value
        for value in payload["trajectories"]
        if isinstance(value, Mapping)
    }
    records: list[dict[str, Any]] = []
    for alias in aliases:
        trajectory_id = alias_map.get(alias)
        if not isinstance(trajectory_id, str) or trajectory_id not in trajectories:
            raise RuntimeError(f"source alias {alias!r} is missing or ambiguous")
        entry = trajectories[trajectory_id]
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise RuntimeError(f"source alias {alias!r} has no artifact mapping")
        digests = artifacts.get("sha256")
        if not isinstance(digests, Mapping):
            raise RuntimeError(f"source alias {alias!r} has no SHA-256 mapping")
        members: dict[str, Any] = {}
        for name in ("resolved_config", "result", "trace"):
            path = _catalog_member(catalog_path, artifacts.get(name), name)
            expected = digests.get(name)
            actual = file_sha256(path)
            if not isinstance(expected, str) or expected != actual:
                raise RuntimeError(
                    f"source alias {alias!r} {name} SHA-256 mismatch"
                )
            members[f"{name}_path"] = str(path)
            members[f"{name}_sha256"] = actual
        records.append(
            {
                "alias": alias,
                "trajectory_id": trajectory_id,
                **members,
            }
        )
    return tuple(records)


def build_scaled_contact_downsize_manifest(
    config_path: str | Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Bind the v13 source catalog *and* its explicit source manifest.

    The generic actual-contact manifest already binds the model, lockfile,
    registered actual-qpos manifest, config and implementation tree.  The
    schema-v13 continuation adds the diagnostic source catalog, because those
    measured centroids are immutable campaign input too.
    """

    config = Path(config_path).expanduser().resolve()
    definition = resolve_experiment(load_config(config))
    campaign = _campaign(definition)
    if int(seed) != int(campaign.seed):
        raise ValueError(
            "schema-v13 seed must match scaled_contact_downsize_campaign.seed"
        )
    source_catalog = _repo_json_path(campaign.source_catalog, "source_catalog")
    source_manifest = _repo_json_path(campaign.source_manifest, "source_manifest")
    source_records = _audit_source_catalog_files(
        source_catalog, campaign.source_aliases
    )
    audited_sources = _default_source_loader(campaign)
    _cross_validate_source_manifest(campaign, audited_sources)
    sources = tuple(implementation_paths())
    base = build_campaign_manifest(config, seed=int(seed), source_paths=sources)
    base.pop("campaign_input_sha256", None)
    base.update(
        {
            "scaled_contact_downsize_manifest_schema_version": 1,
            "scaled_contact_downsize_campaign": campaign.as_config(),
            "scaled_contact_source_catalog_path": str(
                source_catalog.relative_to(REPO_ROOT)
            ),
            "scaled_contact_source_catalog_sha256": file_sha256(source_catalog),
            "scaled_contact_source_manifest_path": str(
                source_manifest.relative_to(REPO_ROOT)
            ),
            "scaled_contact_source_manifest_sha256": file_sha256(source_manifest),
            "scaled_contact_source_records": list(source_records),
        }
    )
    return {**base, "campaign_input_sha256": canonical_sha256(base)}


def campaign_strata(campaign: Any) -> tuple[dict[str, Any], ...]:
    """Return the registered 174 strata in prefix-stable continuation order."""

    values: list[dict[str, Any]] = []
    index = 0
    for edge_m in sorted(campaign.edges_m, reverse=True):
        for source_alias in campaign.source_aliases:
            for mapping_mode in campaign.mapping_modes:
                values.append(
                    {
                        "stratum_index": index,
                        "edge_m": float(edge_m),
                        "edge_mm": int(round(float(edge_m) * 1000.0)),
                        "source_alias": str(source_alias),
                        "mapping_mode": str(mapping_mode),
                    }
                )
                index += 1
    if len(values) != int(campaign.stratum_count):
        raise RuntimeError("registered scaled-contact stratum count changed")
    return tuple(values)


def _candidate_identifier(*parts: Any) -> int:
    digest = hashlib.sha256(
        json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    # Dynamic expansion appends a small controller suffix; remain in int64.
    return 13 * 10**15 + int.from_bytes(digest[:6], "big") % 10**14


def _source_descriptor(value: Any, alias: str) -> dict[str, Any]:
    fields = (
        "catalog_path",
        "catalog_sha256",
        "trajectory_id",
        "config_path",
        "result_path",
        "trace_path",
        "config_sha256",
        "result_sha256",
        "trace_sha256",
    )
    result = {"source_alias": alias}
    for field in fields:
        raw = getattr(value, field, None)
        if raw is not None:
            result[field] = str(raw) if isinstance(raw, Path) else copy.deepcopy(raw)
    stable = getattr(value, "stable_window", None)
    if stable is not None:
        result["stable_window"] = {
            name: copy.deepcopy(getattr(stable, name))
            for name in (
                "start_step",
                "end_step",
                "sample_count",
                "centroid_cube_local_m",
                "valid_counts",
                "force_weight_sum_n",
            )
            if hasattr(stable, name)
        }
    return result


def _default_source_loader(campaign: Any) -> Mapping[str, Any]:
    from .contact_point_downsize import audit_downsize_source

    catalog = _repo_json_path(campaign.source_catalog, "source_catalog")
    return {
        alias: audit_downsize_source(catalog, trajectory=alias)
        for alias in campaign.source_aliases
    }


def _cross_validate_source_manifest(
    campaign: Any, sources: Mapping[str, Any]
) -> None:
    """Cross-check manifest values against freshly recomputed NPZ evidence."""

    manifest_path = _repo_json_path(campaign.source_manifest, "source_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("sources") if isinstance(manifest, Mapping) else None
    if not isinstance(entries, list):
        raise RuntimeError("scaled-contact source manifest has no source list")
    by_alias = {
        str(value.get("alias")): value
        for value in entries
        if isinstance(value, Mapping)
    }
    if tuple(by_alias) != tuple(campaign.source_aliases):
        raise RuntimeError("source manifest aliases/order differ from the campaign")
    catalog = manifest.get("source_catalog", {})
    catalog_path = _repo_json_path(campaign.source_catalog, "source_catalog")
    if (
        not isinstance(catalog, Mapping)
        or catalog.get("path") != campaign.source_catalog
        or catalog.get("sha256") != file_sha256(catalog_path)
    ):
        raise RuntimeError("source manifest catalog binding changed")
    for alias in campaign.source_aliases:
        evidence = sources[alias]
        entry = by_alias[alias]
        expected = entry.get("reference_contact_evidence")
        if not isinstance(expected, Mapping):
            raise RuntimeError(f"source manifest {alias!r} has no contact evidence")
        fresh_payload = evidence.reference_contact_payload(alias)
        fresh = {
            "reference_contact_evidence_sha256": (
                evidence.reference_contact_evidence_sha256(alias)
            ),
            "reference_target_face_yz_m": fresh_payload[
                "reference_target_face_yz_m"
            ],
            "stable_window_start_step": evidence.stable_window.start_step,
            "stable_window_end_step": evidence.stable_window.end_step,
        }
        if canonical_sha256(expected) != canonical_sha256(fresh):
            raise RuntimeError(
                f"source manifest {alias!r} disagrees with fresh trace centroids"
            )
        artifacts = entry.get("artifacts", {})
        hashes = artifacts.get("sha256", {}) if isinstance(artifacts, Mapping) else {}
        if (
            not isinstance(hashes, Mapping)
            or hashes.get("resolved_config") != evidence.config_sha256
            or hashes.get("result") != evidence.result_sha256
            or hashes.get("trace") != evidence.trace_sha256
        ):
            raise RuntimeError(
                f"source manifest {alias!r} artifact hashes disagree with audit"
            )


def _static_initialization_branch(
    start_index: int, *, has_previous_edge: bool
) -> tuple[str, int, bool]:
    """Map the four campaign starts onto the declared initialization basins.

    The campaign-level start index is deliberately not passed through to the
    DLS solver.  Start zero is the descending-size continuation (when one is
    available), start one is an unperturbed remap of the authenticated 89-mm
    source, and starts two/three are the first two deterministic local basins
    around that direct remap.  At 88 mm there is no previous edge, so start
    zero is a second direct solve with its own damping schedule.
    """

    if isinstance(start_index, bool) or start_index not in range(4):
        raise ValueError("static campaign start_index must be in [0, 3]")
    if start_index == 0:
        if has_previous_edge:
            return "previous_edge_continuation", 0, True
        return "reference_edge_direct_alternate_damping", 0, False
    if start_index == 1:
        return "reference_edge_direct", 0, False
    return f"reference_edge_local_branch_{start_index - 1}", start_index - 1, False


def _solve_static_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Spawn-safe real-witness DLS worker for one deterministic start."""

    from .contact_point_downsize import solve_downsize_static_candidate
    from .contact_point_targeted_search import PointTargetDLSSettings

    start_index = int(job["start_index"])
    initialization_branch, solver_start_index, use_previous_edge = (
        _static_initialization_branch(
            start_index,
            has_previous_edge=job.get("previous_edge_config") is not None,
        )
    )
    damping = (0.0125, 0.025, 0.05, 0.075)[start_index]
    regularization = (0.01, 0.02, 0.04, 0.08)[start_index]
    evidence = job["evidence"]
    stratum = job["stratum"]
    identifier = int(job["candidate_id"])
    try:
        solved = solve_downsize_static_candidate(
            evidence.config,
            target_edge_m=float(stratum["edge_m"]),
            source_centroids_cube_local_m=(
                evidence.stable_window.centroid_cube_local_m
            ),
            mapping_mode=str(stratum["mapping_mode"]),
            source_id=str(stratum["source_alias"]),
            source_evidence=evidence,
            target_template=job["template"],
            target_radius_m=float(job["target_radius_m"]),
            edge_guard_m=float(job["edge_guard_m"]),
            seed=int(job["seed"]),
            start_index=solver_start_index,
            max_iterations=int(job["max_iterations"]),
            previous_edge_config=(
                job.get("previous_edge_config") if use_previous_edge else None
            ),
            settings=PointTargetDLSSettings(
                damping=damping,
                regularization_weight=regularization,
            ),
        )
        record = solved.as_dynamic_record(
            identifier, source_id=str(stratum["source_alias"])
        )
        # A static pass is allowed to enter controller expansion only when the
        # exact materialized config satisfies the same registered constraints
        # used by the dynamic runner.  This is deliberately inside the worker
        # try block: a future numerical inconsistency becomes an inspectable
        # near miss instead of aborting the campaign after earlier batches.
        validate_config(record["config"])
    except (RuntimeError, ValueError, ArithmeticError) as error:
        # A numerical DLS failure is a near-miss record, not permission to
        # persist an internally inconsistent template (for example an 88-mm
        # point plan beside a 70-mm cube).  Materialize the direct mapped seed
        # so the report remains a valid, inspectable schema-v13 near miss.  It
        # stays ``static_pass=False`` and is never sent to dynamic simulation.
        from .contact_point_downsize import (
            build_downsize_contact_point_plan,
            map_downsize_seed_config,
        )
        from .contact_point_targeted_search import bind_frozen_contact_point_plan

        plan = build_downsize_contact_point_plan(
            evidence.source_edge_m,
            float(stratum["edge_m"]),
            evidence.stable_window.centroid_cube_local_m,
            mapping_mode=str(stratum["mapping_mode"]),
            target_radius_m=float(job["target_radius_m"]),
            edge_guard_m=float(job["edge_guard_m"]),
        )
        mapped = map_downsize_seed_config(
            evidence.config,
            float(stratum["edge_m"]),
            target_template=job["template"],
        )
        config = copy.deepcopy(mapped.config)
        config.pop("contact_point_plan", None)
        config["scaled_contact_mapping"] = evidence.scaled_contact_mapping(
            str(stratum["mapping_mode"]),
            plan,
            str(stratum["source_alias"]),
        )
        config = bind_frozen_contact_point_plan(config, plan)
        candidate_metadata = config.setdefault("candidate_metadata", {})
        candidate_metadata["contact_point_downsize"] = {
            "source_id": str(stratum["source_alias"]),
            "source_alias": str(stratum["source_alias"]),
            "source_edge_m": float(evidence.source_edge_m),
            "target_edge_m": float(stratum["edge_m"]),
            "edge_m": float(stratum["edge_m"]),
            "mapping_mode": str(stratum["mapping_mode"]),
            "point_plan_id": plan.point_plan_id,
            "static_filter_is_success_evidence": False,
            "initialization_branch": (
                f"dls_error_{initialization_branch}_fallback"
            ),
        }
        record = {
            "candidate_id": identifier,
            "source_id": str(stratum["source_alias"]),
            "config": config,
            "candidate_sha256": canonical_sha256(config),
            "static_pass": False,
            "static_rank": [True, 99, 99, 1.0, 1.0, identifier],
            "static_metrics": {
                "dls_job_error": f"{type(error).__name__}: {error}"
            },
        }
        # The direct mapped fallback must itself remain a legal schema-v13
        # configuration even though it can never be promoted to dynamics.
        validate_config(config)
    metadata = record.setdefault("downsize_metadata", {})
    metadata.update(copy.deepcopy(stratum))
    metadata["dls_start_index"] = start_index
    metadata["solver_start_index"] = solver_start_index
    metadata["initialization_branch"] = initialization_branch
    metadata["previous_edge_continuation"] = use_previous_edge
    config = record.get("config")
    if isinstance(config, dict):
        candidate_metadata = config.setdefault("candidate_metadata", {})
        downsize_metadata = candidate_metadata.setdefault(
            "contact_point_downsize", {}
        )
        if isinstance(downsize_metadata, dict):
            downsize_metadata.update(
                {
                    "campaign_start_index": start_index,
                    "solver_start_index": solver_start_index,
                    "initialization_branch": initialization_branch,
                    "previous_edge_continuation": use_previous_edge,
                }
            )
            record["candidate_sha256"] = canonical_sha256(config)
    return record


def _authenticate_static_progress(
    path: Path,
    *,
    expected_input_sha256: str,
    retained_count: int,
    expected_stratum: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if (
        payload.get("complete") is not True
        or payload.get("stratum_input_sha256") != expected_input_sha256
        or not isinstance(records, list)
        or len(records) != retained_count
        or payload.get("records_semantic_sha256") != canonical_sha256(records)
    ):
        raise RuntimeError(f"static progress changed: {path}")
    identifiers = [int(value.get("candidate_id", -1)) for value in records]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"static progress candidate IDs changed: {path}")
    for value in records:
        if canonical_sha256(value.get("config", {})) != value.get("candidate_sha256"):
            raise RuntimeError(f"static progress candidate hash changed: {path}")
        metadata = value.get("downsize_metadata", {})
        if any(
            metadata.get(field) != expected_stratum.get(field)
            for field in (
                "stratum_index",
                "edge_m",
                "edge_mm",
                "source_alias",
                "mapping_mode",
            )
        ):
            raise RuntimeError(f"static progress stratum metadata changed: {path}")
    expected_order = sorted(
        records,
        key=lambda value: (
            not bool(value.get("static_pass", False)),
            tuple(value.get("static_rank", ())),
            int(value["candidate_id"]),
        ),
    )
    if [int(value["candidate_id"]) for value in expected_order] != identifiers:
        raise RuntimeError(f"static progress retained ordering changed: {path}")
    return tuple(copy.deepcopy(records))


def _default_static_runner(
    template: Mapping[str, Any],
    campaign: Any,
    sources: Mapping[str, Any],
    *,
    workspace: Path,
    resume: bool,
    seed: int,
    workers: int,
) -> tuple[dict[str, Any], ...]:
    """Run all strata with spawn workers and per-stratum atomic checkpoints."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    progress_root = workspace / "static_progress"
    progress_root.mkdir(parents=True, exist_ok=True)
    strata = campaign_strata(campaign)
    by_edge: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for stratum in strata:
        by_edge[float(stratum["edge_m"])].append(stratum)
    prior: dict[tuple[str, str], dict[str, Any]] = {}
    retained_all: list[dict[str, Any]] = []
    executor = None
    if workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=int(workers), mp_context=multiprocessing.get_context("spawn")
        )
    try:
        for edge in sorted(by_edge, reverse=True):
            missing_jobs: list[dict[str, Any]] = []
            loaded: dict[int, tuple[dict[str, Any], ...]] = {}
            input_hashes: dict[int, str] = {}
            for stratum in by_edge[edge]:
                key = (stratum["source_alias"], stratum["mapping_mode"])
                previous = prior.get(key)
                input_sha = canonical_sha256(
                    {
                        "stratum": stratum,
                        "source_config_sha256": sources[key[0]].config_sha256,
                        "source_result_sha256": sources[key[0]].result_sha256,
                        "source_trace_sha256": sources[key[0]].trace_sha256,
                        "previous_candidate_sha256": (
                            None
                            if previous is None
                            else canonical_sha256(previous)
                        ),
                        "seed": int(seed),
                        "dls_starts": int(campaign.dls_starts_per_stratum),
                        "max_iterations": int(campaign.max_dls_iterations),
                    }
                )
                input_hashes[int(stratum["stratum_index"])] = input_sha
                path = progress_root / f"stratum_{stratum['stratum_index']:03d}.json"
                if path.is_file():
                    if not resume:
                        raise FileExistsError(path)
                    loaded[int(stratum["stratum_index"])] = (
                        _authenticate_static_progress(
                            path,
                            expected_input_sha256=input_sha,
                            retained_count=int(campaign.static_retain_per_stratum),
                            expected_stratum=stratum,
                        )
                    )
                    continue
                for start_index in range(int(campaign.dls_starts_per_stratum)):
                    missing_jobs.append(
                        {
                            "candidate_id": _candidate_identifier(
                                stratum["stratum_index"], start_index, seed
                            ),
                            "stratum": stratum,
                            "start_index": start_index,
                            "seed": int(seed),
                            "evidence": sources[key[0]],
                            "template": copy.deepcopy(dict(template)),
                            "target_radius_m": float(
                                campaign.contact_target_radius_m
                            ),
                            "edge_guard_m": float(campaign.minimum_edge_guard_m),
                            "max_iterations": int(campaign.max_dls_iterations),
                            "previous_edge_config": copy.deepcopy(previous),
                        }
                    )
            if missing_jobs:
                results = (
                    list(map(_solve_static_job, missing_jobs))
                    if executor is None
                    else list(executor.map(_solve_static_job, missing_jobs, chunksize=1))
                )
                grouped_results: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for value in results:
                    grouped_results[int(value["downsize_metadata"]["stratum_index"])].append(value)
                for stratum in by_edge[edge]:
                    index = int(stratum["stratum_index"])
                    if index in loaded:
                        continue
                    candidates = grouped_results[index]
                    candidates.sort(
                        key=lambda value: (
                            not bool(value.get("static_pass", False)),
                            tuple(value.get("static_rank", ())),
                            int(value["candidate_id"]),
                        )
                    )
                    retained = tuple(
                        candidates[: int(campaign.static_retain_per_stratum)]
                    )
                    continuation = next(
                        (
                            value
                            for value in retained
                            if "dls_job_error"
                            not in value.get("static_metrics", {})
                        ),
                        None,
                    )
                    path = progress_root / f"stratum_{index:03d}.json"
                    write_json(
                        path,
                        {
                            "complete": True,
                            "stratum": stratum,
                            "stratum_input_sha256": input_hashes[index],
                            "executed_start_count": len(candidates),
                            "retained_count": len(retained),
                            "continuation_candidate_id": (
                                None
                                if continuation is None
                                else int(continuation["candidate_id"])
                            ),
                            "continuation_skipped_reason": (
                                None
                                if continuation is not None
                                else "all_retained_dls_jobs_raised_before_legal_v13_config"
                            ),
                            "records_semantic_sha256": canonical_sha256(retained),
                            "records": list(retained),
                        },
                    )
                    loaded[index] = retained
            for stratum in by_edge[edge]:
                index = int(stratum["stratum_index"])
                records = loaded[index]
                retained_all.extend(copy.deepcopy(value) for value in records)
                continuation = next(
                    (
                        value
                        for value in records
                        if "dls_job_error"
                        not in value.get("static_metrics", {})
                    ),
                    None,
                )
                if continuation is not None:
                    key = (stratum["source_alias"], stratum["mapping_mode"])
                    # A real-witness DLS near miss is still the required
                    # previous-size continuation seed.  Only a worker failure
                    # lacking a legal schema-v13 mapping may be skipped.
                    prior[key] = copy.deepcopy(dict(continuation["config"]))
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    identifiers = [int(value["candidate_id"]) for value in retained_all]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("scaled-contact static candidate ID collision")
    return tuple(retained_all)


def _metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    direct = record.get("downsize_metadata")
    if isinstance(direct, Mapping):
        return copy.deepcopy(dict(direct))
    downsize = record.get("downsize")
    if isinstance(downsize, Mapping):
        value = copy.deepcopy(dict(downsize))
        value.setdefault("edge_m", value.get("target_edge_m"))
        value.setdefault("source_alias", value.get("source_id"))
        return value
    config = record.get("config")
    if isinstance(config, Mapping):
        scaled = config.get("scaled_contact_mapping")
        if isinstance(scaled, Mapping):
            value = copy.deepcopy(dict(scaled))
            value.setdefault("edge_m", config.get("cube", {}).get("edge_m"))
            return value
        candidate = config.get("candidate_metadata")
        if isinstance(candidate, Mapping):
            for key in (
                "scaled_contact_downsize",
                "contact_point_downsize",
                "scaled_contact_downsize_campaign",
            ):
                value = candidate.get(key)
                if isinstance(value, Mapping):
                    return copy.deepcopy(dict(value))
    return {}


def _record_edge_m(record: Mapping[str, Any]) -> float:
    metadata = _metadata(record)
    raw = metadata.get("edge_m")
    if raw is None:
        config = record.get("config", {})
        raw = config.get("cube", {}).get("edge_m") if isinstance(config, Mapping) else None
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise RuntimeError("candidate has no downsize edge metadata") from error
    if not math.isfinite(value):
        raise RuntimeError("candidate edge metadata is non-finite")
    return value


def _default_manipulation_runner(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    campaign: Any,
    target_success_count: int,
    seed: int,
    workers: int,
    resume: bool,
) -> tuple[dict[str, Any], ...]:
    """Probe one finalized grasp per passing edge/mapping, then refine."""

    del resume
    if (
        not isinstance(target_success_count, int)
        or isinstance(target_success_count, bool)
        or target_success_count <= 0
    ):
        raise ValueError("target_success_count must be a positive integer")
    from .actual_contact_grasp_pose import (
        _run_manipulation_local_refinement_stage,
        _run_manipulation_source,
    )
    from .actual_contact_manipulation import (
        LocalRefinementBudget,
    )
    from .scaled_contact_downsize_ranking import (
        authenticate_v13_manipulation_compaction_report,
        compact_v13_manipulation_candidate_artifacts,
        rank_v13_manipulation_candidates,
        v13_manipulation_candidate_rank,
    )

    compaction_root = output_dir / "manipulation" / "v13_compaction"

    def compact_batch(
        values: Sequence[Mapping[str, Any]], report_path: Path
    ) -> tuple[dict[str, Any], ...]:
        # A later edge/global compaction can legitimately remove a trace kept
        # by an earlier report.  On resume, candidate loaders authenticate the
        # current result state; an existing committed local report therefore
        # marks this destructive step complete and must not be replayed.
        if report_path.is_file():
            return authenticate_v13_manipulation_compaction_report(
                values,
                report_path,
                retain_failure_trace_count=1,
            )
        compacted, _ = compact_v13_manipulation_candidate_artifacts(
            values,
            retain_failure_trace_count=1,
            report_path=report_path,
        )
        return compacted

    successes = [
        copy.deepcopy(dict(value))
        for value in rank_dynamic_grasp_results(records)
        if measured_grasp_pose_succeeded(value) and grasp_stage_succeeded(value)
    ]
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for value in successes:
        metadata = _metadata(value)
        grouped[
            (_record_edge_m(value), str(metadata.get("mapping_mode", "unknown")))
        ].append(value)
    promoted = [grouped[key][0] for key in sorted(grouped)]
    all_records: list[dict[str, Any]] = []

    exhaustive_registered_campaign = int(target_success_count) >= int(
        campaign.selected_grasp_count
    )

    def target_reached(values: Sequence[Mapping[str, Any]]) -> bool:
        # The top-level v13 source stage passes the registered publication
        # capacity (87), which is an explicit request to execute every passing
        # edge/mapping's 17+64 probes and both refinement tiers.  Direct
        # target-1/5 invocations may still stop early and resume as originally
        # designed by the manipulation runner.
        return bool(
            not exhaustive_registered_campaign
            and _success_counts(values)[1] >= int(target_success_count)
        )

    for discovery, source in enumerate(promoted):
        source_records = _run_manipulation_source(
            source,
            output_dir,
            source_discovery_index=discovery,
            seed=int(seed),
            workers=int(workers),
        )
        all_records.extend(
            compact_batch(
                source_records,
                compaction_root
                / f"source_{int(source['candidate_id'])}_compaction.json",
            )
        )
        # Each source search and its compaction report are already atomic.
        # Stop only manipulation expansion (never the completed 88->60 grasp
        # scan) as soon as this invocation's requested full-success target is
        # available.  A target-5 resume deterministically reloads these source
        # reports before continuing with the next source.
        if target_reached(all_records):
            break

    # One edge-local 32-run pass prevents a globally easy edge from consuming
    # every local-refinement slot.
    by_edge: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for value in all_records:
        by_edge[_record_edge_m(value)].append(value)
    edge_compacted: list[dict[str, Any]] = []
    edge_order = [] if target_reached(all_records) else sorted(by_edge)
    for edge_index, edge in enumerate(edge_order):
        parent = rank_v13_manipulation_candidates(by_edge[edge])[:1]
        if not parent:
            continue
        execution = _run_manipulation_local_refinement_stage(
            parent,
            output_dir,
            seed=int(seed) + int(round(edge * 1_000_000)),
            workers=int(workers),
            budget=LocalRefinementBudget(
                parent_count=1,
                candidates_per_parent=int(campaign.manipulation_edge_refine_per_best),
                batch_size=min(
                    32, int(campaign.manipulation_edge_refine_per_best)
                ),
                seed=int(seed) + int(round(edge * 1_000_000)),
            ),
        )
        edge_compacted.extend(
            compact_batch(
                (*by_edge[edge], *execution.records),
                compaction_root
                / f"edge_{int(round(edge * 1000.0)):02d}_compaction.json",
            )
        )
        if target_reached(edge_compacted):
            for remaining_edge in edge_order[edge_index + 1 :]:
                edge_compacted.extend(by_edge[remaining_edge])
            break
    if edge_compacted:
        all_records = edge_compacted

    ranked = rank_v13_manipulation_candidates(all_records)
    parent_count = min(int(campaign.global_refine_pose_count), len(ranked))
    if parent_count and not target_reached(all_records):
        execution = _run_manipulation_local_refinement_stage(
            ranked[:parent_count],
            output_dir,
            seed=int(seed) + 13,
            workers=int(workers),
            budget=LocalRefinementBudget(
                parent_count=parent_count,
                candidates_per_parent=int(campaign.global_refine_per_pose),
                seed=int(seed) + 13,
            ),
        )
        all_records.extend(execution.records)
    all_records = list(
        compact_batch(
            all_records,
            compaction_root
            / f"global_target_{int(target_success_count)}_compaction.json",
        )
    )
    return tuple(rank_v13_manipulation_candidates(all_records))


def _dynamic_group_key(record: Mapping[str, Any]) -> tuple[float, str]:
    metadata = _metadata(record)
    return (
        _record_edge_m(record),
        str(metadata.get("mapping_mode", record.get("mapping_mode", "unknown"))),
    )


def _default_batched_dynamic_runner(
    static_records: Sequence[Mapping[str, Any]],
    dynamic_root: Path,
    *,
    workers: int,
    resume: bool,
    seed: int,
    controller_seed_count: int,
) -> tuple[dict[str, Any], ...]:
    """Execute one edge/mapping group at a time and compact immediately.

    One group contains at most three source aliases x two retained poses x six
    controllers (36 traces).  Keeping one ranked failure trace per group gives
    every edge/mapping a Viewer diagnostic without ever materializing all
    2,088 traces at once.
    """

    # Preflight the complete promoted set before creating the first dynamic
    # artifact.  This prevents a late edge from exposing an invalid static
    # pose only after earlier edge/mapping batches have already run.
    promoted = [
        value for value in static_records if bool(value.get("static_pass", False))
    ]
    for value in promoted:
        try:
            validate_config(value["config"])
        except (KeyError, TypeError, ValueError) as error:
            metadata = _metadata(value)
            raise RuntimeError(
                "schema-v13 dynamic preflight rejected static-pass candidate "
                f"{value.get('candidate_id')} "
                f"(stratum={metadata.get('stratum_index')}, "
                f"edge_mm={metadata.get('edge_mm')}, "
                f"mapping={metadata.get('mapping_mode')}): {error}"
            ) from error

    grouped: dict[tuple[float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for value in promoted:
        grouped[_dynamic_group_key(value)].append(value)
    all_records: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: (-value[0], value[1])):
        records = run_actual_contact_dynamic_grasp_candidates(
            grouped[key],
            dynamic_root,
            workers=int(workers),
            resume=bool(resume) or dynamic_root.exists(),
            seed=int(seed),
            controller_seed_count=int(controller_seed_count),
            retain_failure_trace_count=1,
        )
        all_records.extend(copy.deepcopy(dict(value)) for value in records)
        if records and not any(grasp_stage_succeeded(value) for value in records):
            retained = _candidate_artifacts(records, dynamic_root)
            if not any(path.name == "trace.npz" for path in retained):
                raise RuntimeError(
                    "dynamic compaction lost the best edge/mapping near-miss trace"
                )
    identifiers = [int(value["candidate_id"]) for value in all_records]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("batched dynamic runner produced duplicate candidate IDs")
    return tuple(sorted(all_records, key=lambda value: int(value["candidate_id"])))


def _default_local_grasp_runner(
    dynamic_records: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    campaign: Any,
    workers: int,
    resume: bool,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Run the registered top-two x 32 local search in bounded batches."""

    del resume
    from .actual_contact_grasp_pose import (
        _run_joint_controller_local_refinement_stage,
    )
    from .actual_contact_grasp_pose_dynamic import (
        compact_dynamic_candidate_artifacts,
        run_or_resume_dynamic_candidates,
    )

    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for value in rank_dynamic_grasp_results(dynamic_records):
        grouped[_dynamic_group_key(value)].append(copy.deepcopy(dict(value)))
    all_records: list[dict[str, Any]] = []
    dynamic_root = workspace / "dynamic"
    for group_index, key in enumerate(
        sorted(grouped, key=lambda value: (-value[0], value[1]))
    ):
        parents = grouped[key][: int(campaign.local_poses_per_edge_mapping)]
        if not parents:
            continue
        edge_mm = int(round(key[0] * 1000.0))
        safe_mode = "proportional" if key[1].startswith("proportional") else "absolute"
        stage = f"v13_edge_{edge_mm}_{safe_mode}"
        generated = _run_joint_controller_local_refinement_stage(
            parents,
            workspace,
            stage=stage,
            top_count=int(campaign.local_poses_per_edge_mapping),
            candidates_per_pose=int(campaign.local_refine_per_pose),
            seed=int(seed) + group_index,
            maximum_ik_iterations=min(4, int(campaign.max_dls_iterations)),
        )
        remapped: list[dict[str, Any]] = []
        for local_index, raw in enumerate(generated.records):
            value = copy.deepcopy(dict(raw))
            identifier = _candidate_identifier(
                "local_grasp", edge_mm, safe_mode, group_index, local_index
            )
            config = copy.deepcopy(dict(value["config"]))
            metadata = config.setdefault("candidate_metadata", {})
            metadata["candidate_id"] = identifier
            metadata["stage"] = "scaled_contact_local_grasp_refinement"
            metadata["scaled_contact_local_refinement"] = {
                "edge_m": key[0],
                "mapping_mode": key[1],
                "local_index": local_index,
                "group_index": group_index,
            }
            value.update(
                {
                    "candidate_id": identifier,
                    "config": config,
                    "candidate_sha256": canonical_sha256(config),
                    "grasp_pose_id": grasp_pose_id(config),
                    "controller_id": controller_id(config),
                }
            )
            remapped.append(value)
        if not remapped:
            continue
        executed = run_or_resume_dynamic_candidates(
            tuple(remapped),
            dynamic_root,
            workers=int(workers),
            resume=dynamic_root.exists(),
        )
        compacted = compact_dynamic_candidate_artifacts(
            executed,
            dynamic_root,
            retain_failure_trace_count=1,
        )
        all_records.extend(copy.deepcopy(dict(value)) for value in compacted)
    identifiers = [int(value["candidate_id"]) for value in all_records]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("local grasp refinement produced duplicate IDs")
    return tuple(sorted(all_records, key=lambda value: int(value["candidate_id"])))


def _default_grasp_robustness_runner(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    workers: int,
    seed: int,
    campaign: Any,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """Run local-16 and best-50 perturbations using grasp success semantics."""

    from .contact_point_grasp_robustness import (
        discover_v12_grasp_robustness_sources,
        run_v12_grasp_perturbation_audit,
        v12_grasp_hard_success,
    )

    if catalog_path is not None:
        successful = list(
            discover_v12_grasp_robustness_sources(
                (catalog_path,),
                maximum_nominal_grasps=int(campaign.selected_grasp_count),
            )
        )
    else:
        successful = [
            copy.deepcopy(dict(value))
            for value in rank_dynamic_grasp_results(records)
            if measured_grasp_pose_succeeded(value)
            and grasp_stage_succeeded(value)
            and v12_grasp_hard_success(value.get("summary", {}))
        ][: int(campaign.selected_grasp_count)]
    if not successful:
        raise ValueError("grasp robustness requires a measured grasp success")

    sources = (
        successful
        if catalog_path is not None
        else [
            {
                **value,
                "best": index == 0,
                "discovery_index": int(value.get("discovery_index", index)),
            }
            for index, value in enumerate(successful)
        ]
    )
    result = run_v12_grasp_perturbation_audit(
        sources,
        output_path,
        workers=int(workers),
        seed=int(seed),
        local_perturbations=int(campaign.perturbations_per_published),
        best_perturbations=int(campaign.robustness_trials),
        maximum_nominal_grasps=int(campaign.selected_grasp_count),
    )
    # The registered ``robust`` label denotes robust *full lift* success.  A
    # grasp-only 45/50 audit must not inherit that stronger claim.
    if result.get("robust_passed") is True:
        result["validation_label"] = resolve_experiment(
            (
                successful[0].config
                if catalog_path is not None
                else successful[0]["config"]
            )
        ).actual_contact_grasp_pose_campaign.validation_labels["grasp"]
    return {
        **result,
        "scaled_contact_grasp_robustness_schema_version": 1,
        "success_metric": "grasp_success_not_full_lift_success",
        "full_lift_robustness_claimed": False,
    }


@dataclass(frozen=True, slots=True)
class CampaignBackend:
    """Numerical/artifact boundaries used by production and focused tests."""

    source_loader: Callable[[Any], Mapping[str, Any]] = _default_source_loader
    static_runner: Callable[..., tuple[dict[str, Any], ...]] = _default_static_runner
    dynamic_runner: Callable[..., tuple[dict[str, Any], ...]] = (
        _default_batched_dynamic_runner
    )
    local_grasp_runner: Callable[..., tuple[dict[str, Any], ...]] = (
        _default_local_grasp_runner
    )
    measured_runner: Callable[..., tuple[dict[str, Any], ...]] = (
        run_or_resume_measured_grasp_finalization
    )
    manipulation_runner: Callable[..., tuple[dict[str, Any], ...]] = (
        _default_manipulation_runner
    )
    grasp_catalog_exporter: Callable[..., dict[str, Any]] = (
        export_scaled_contact_downsize_grasp_catalog
    )
    manipulation_catalog_exporter: Callable[..., dict[str, Any]] = (
        export_scaled_contact_downsize_manipulation_catalog
    )
    grasp_robustness_runner: Callable[..., dict[str, Any]] = (
        _default_grasp_robustness_runner
    )
    robustness_runner: Callable[..., dict[str, Any]] | None = None


DEFAULT_BACKEND = CampaignBackend()


def _safe_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise RuntimeError(f"committed v13 report is malformed: {path}")
    return payload


def _existing_stage(workspace: Path, stage: str, report: Path) -> dict[str, Any] | None:
    ledger = validate_stage_ledger(workspace)
    if stage not in ledger["stages"]:
        return None
    if not report.is_file():
        raise RuntimeError(f"committed stage {stage} lost {report}")
    return _safe_report(report)


def _commit_report(
    workspace: Path,
    stage: str,
    report_path: Path,
    payload: Mapping[str, Any],
    *,
    stage_input: Mapping[str, Any],
    extra_artifacts: Sequence[Path] = (),
) -> dict[str, Any]:
    write_json(report_path, payload)
    commit_campaign_stage(
        workspace,
        stage,
        stage_input=stage_input,
        artifacts=(report_path, *extra_artifacts),
        summary={
            key: copy.deepcopy(payload[key])
            for key in (
                "source_count",
                "stratum_count",
                "candidate_count",
                "grasp_success_count",
                "full_success_count",
                "target_reached",
            )
            if key in payload
        },
    )
    return copy.deepcopy(dict(payload))


def _candidate_artifacts(
    records: Sequence[Mapping[str, Any]], root: Path
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for value in records:
        directory = value.get("artifact_directory")
        if directory is not None:
            candidate_root = (root / str(directory)).resolve()
            for name in ("resolved_config.json", "result.json", "trace.npz"):
                path = candidate_root / name
                if path.is_file():
                    paths.append(path)
        for field in ("config_path", "result_path", "trace_path"):
            raw = value.get(field)
            if raw is not None and Path(str(raw)).is_file():
                paths.append(Path(str(raw)).resolve())
    return tuple(dict.fromkeys(paths))


def _success_counts(records: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    grasp = sum(
        measured_grasp_pose_succeeded(value) and grasp_stage_succeeded(value)
        for value in records
    )
    full = sum(
        bool(value.get("summary", {}).get("passed", False))
        and bool(
            value.get("summary", {})
            .get("stage_status", {})
            .get("full_success", False)
        )
        for value in records
    )
    return int(grasp), int(full)


def _catalog_candidates(
    records: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    *,
    kind: str = "grasp_pose",
) -> tuple[dict[str, Any], ...]:
    # Enforce the registered per-edge publication cap before the generic
    # actual-contact exporter applies its global target count.  Prefer a new
    # mapping/source family first, then fill remaining slots by input rank.
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for value in records:
        grouped[_record_edge_m(value)].append(value)
    if kind not in {"grasp_pose", "manipulation"}:
        raise ValueError("catalog kind must be grasp_pose or manipulation")
    success_predicate = (
        catalog_grasp_success if kind == "grasp_pose" else catalog_full_success
    )

    def rank_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
        if kind == "grasp_pose":
            return final_candidate_rank(value)
        config = value.get("config")
        if not isinstance(config, Mapping):
            directory = value.get("artifact_directory")
            config_path = (
                (artifact_root / str(directory) / "resolved_config.json").resolve()
                if directory is not None
                else Path(str(value.get("config_path", ""))).resolve()
            )
            if not config_path.is_file():
                raise FileNotFoundError(
                    "schema-v13 manipulation rank lost resolved config: "
                    f"candidate_{value.get('candidate_id')}"
                )
            config = load_config(config_path)
        return v13_manipulation_candidate_rank(value, config)

    publication_order: list[Mapping[str, Any]] = []
    for edge in sorted(grouped):
        # Upstream worker completion order is never publication order.  Use
        # the experiment-specific rank first, then apply the per-edge family
        # coverage rule below.  Manipulation must not lose schema-v13's soft
        # contact-slip ordering at this publication boundary.
        values = sorted(grouped[edge], key=rank_key)
        selected: list[Mapping[str, Any]] = []
        used: set[tuple[str, str]] = set()

        # A failed family representative must never displace a hard pass from
        # the registered three-per-edge publication allowance.  First take
        # diverse successes, then remaining successes, and only then retain
        # near misses for a possible diagnostic entry.
        pools = (
            [value for value in values if success_predicate(value)],
            [value for value in values if not success_predicate(value)],
        )
        for pool in pools:
            for value in pool:
                metadata = _metadata(value)
                family = (
                    str(metadata.get("mapping_mode", "unknown")),
                    str(
                        metadata.get(
                            "source_alias", metadata.get("source_id", "unknown")
                        )
                    ),
                )
                if family in used:
                    continue
                selected.append(value)
                used.add(family)
                if len(selected) == 3:
                    break
            if len(selected) == 3:
                break
            selected_ids = {int(value["candidate_id"]) for value in selected}
            for value in pool:
                if int(value["candidate_id"]) in selected_ids:
                    continue
                selected.append(value)
                selected_ids.add(int(value["candidate_id"]))
                if len(selected) == 3:
                    break
            if len(selected) == 3:
                break
        publication_order.extend(selected[:3])
    result: list[dict[str, Any]] = []
    for discovery, value in enumerate(publication_order):
        directory = value.get("artifact_directory")
        if directory is not None:
            root = (artifact_root / str(directory)).resolve()
            config_path = root / "resolved_config.json"
            result_path = root / "result.json"
            trace_path = root / "trace.npz"
        else:
            config_path = Path(str(value.get("config_path", ""))).resolve()
            result_path = Path(str(value.get("result_path", ""))).resolve()
            trace_path = Path(str(value.get("trace_path", ""))).resolve()
        if not all(path.is_file() for path in (config_path, result_path, trace_path)):
            if success_predicate(value):
                raise FileNotFoundError(
                    "successful schema-v13 catalog candidate lost an artifact: "
                    f"candidate_{value.get('candidate_id')}"
                )
            continue
        result.append(
            {
                "candidate_id": int(value["candidate_id"]),
                "discovery_index": int(value.get("discovery_index", discovery)),
                "config_path": config_path,
                "result_path": result_path,
                "trace_path": trace_path,
                "summary": copy.deepcopy(dict(value["summary"])),
            }
        )
    return tuple(result)


def _annotate_catalog(
    catalog_path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    kind: str,
) -> dict[str, Any]:
    """Add deterministic edge/mapping/source metadata and useful aliases."""

    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    metadata_by_id = {
        str(value["candidate_id"]): {
            **_metadata(value),
            "edge_m": _record_edge_m(value),
            "edge_mm": int(round(_record_edge_m(value) * 1000.0)),
        }
        for value in records
    }
    aliases = copy.deepcopy(dict(payload.get("aliases", {})))
    successful_by_edge: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in payload.get("trajectories", ()):
        metadata = metadata_by_id.get(str(entry.get("candidate_id")), {})
        entry["campaign_metadata"] = metadata
        for field in ("edge_m", "edge_mm", "mapping_mode", "source_alias"):
            if field in metadata:
                entry[field] = metadata[field]
        if entry.get("classification") == "success" and "edge_mm" in metadata:
            successful_by_edge[int(metadata["edge_mm"])].append(entry)
    over_cap = {
        edge_mm: len(entries)
        for edge_mm, entries in successful_by_edge.items()
        if len(entries) > 3
    }
    if over_cap:
        raise RuntimeError(
            "schema-v13 catalog exceeded three successes per edge: "
            + ", ".join(
                f"{edge_mm}mm={count}" for edge_mm, count in sorted(over_cap.items())
            )
        )
    predicate = (
        catalog_grasp_success if kind == "grasp_pose" else catalog_full_success
    )
    expected_success_edges = {
        int(round(_record_edge_m(value) * 1000.0))
        for value in records
        if predicate(value)
    }
    missing_edges = sorted(expected_success_edges - set(successful_by_edge))
    if missing_edges:
        raise RuntimeError(
            "schema-v13 Viewer catalog omitted successful edges: "
            + ", ".join(str(value) for value in missing_edges)
        )
    for edge_mm, entries in sorted(successful_by_edge.items()):
        entry = entries[0]
        alias = f"edge_{edge_mm}_best"
        aliases[alias] = entry["trajectory_id"]
        if alias not in entry.setdefault("aliases", []):
            entry["aliases"].append(alias)
    if successful_by_edge:
        smallest = min(successful_by_edge)
        entry = successful_by_edge[smallest][0]
        smallest_alias = (
            "smallest_grasp_pass" if kind == "grasp_pose" else "smallest_lift_pass"
        )
        aliases[smallest_alias] = entry["trajectory_id"]
        if smallest_alias not in entry.setdefault("aliases", []):
            entry["aliases"].append(smallest_alias)
        if "best_nominal" not in aliases:
            best = next(
                entry
                for entry in payload.get("trajectories", ())
                if entry.get("classification") == "success"
            )
            aliases["best_nominal"] = best["trajectory_id"]
            if "best_nominal" not in best.setdefault("aliases", []):
                best["aliases"].append("best_nominal")
    payload["catalog_metadata_schema_version"] = CATALOG_METADATA_SCHEMA_VERSION
    payload["metadata_fields"] = ["edge_mm", "mapping_mode", "source_alias"]
    payload["publication_cap_per_edge"] = 3
    payload["successful_edges_mm"] = sorted(successful_by_edge)
    payload["per_edge_success_counts"] = {
        str(edge_mm): len(entries)
        for edge_mm, entries in sorted(successful_by_edge.items())
    }
    payload["aliases"] = aliases
    write_json(catalog_path, payload)
    return payload


def _publish_catalog(
    records: Sequence[Mapping[str, Any]],
    source_root: Path,
    destination: Path,
    *,
    target_success_count: int,
    kind: str,
    exporter: Callable[..., dict[str, Any]],
) -> Path | None:
    candidates = _catalog_candidates(records, source_root, kind=kind)
    if not candidates:
        return None
    predicate = (
        catalog_grasp_success if kind == "grasp_pose" else catalog_full_success
    )
    # The CLI target (one or five) is a campaign stopping/reporting target,
    # not a license to discard successful per-edge evidence.  Publish every
    # hard pass surviving the registered three-per-edge cap; retain one slot
    # when there is no pass so the generic exporter can publish best_attempt.
    selected_count = max(1, sum(predicate(value) for value in candidates))
    catalog_path = destination / "catalog.json"
    if not catalog_path.is_file():
        exporter(candidates, destination, selected_count=selected_count)
    _annotate_catalog(catalog_path, records, kind=kind)
    return catalog_path


def _default_robustness_runner(
    catalog_path: Path,
    output_path: Path,
    *,
    workers: int,
    seed: int,
    campaign: Any,
) -> dict[str, Any]:
    from .actual_contact_grasp_pose_robustness import (
        discover_v9_robustness_sources,
        run_v9_robustness_campaign,
    )

    # Generic v9 calls the extra-50 source ``best_first`` for historical
    # reasons.  Schema-v13 explicitly requires the best complete lift, which
    # is the catalog's final-rank ``best_nominal`` alias rather than its
    # chronological first success.  Rebind only the in-memory robustness
    # source marker; the authenticated catalog and member evidence are not
    # modified.
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    aliases = catalog.get("aliases", {})
    trajectories = {
        str(value.get("trajectory_id")): value
        for value in catalog.get("trajectories", ())
        if isinstance(value, Mapping)
    }
    best_trajectory = aliases.get("best_nominal") if isinstance(aliases, Mapping) else None
    best_entry = trajectories.get(str(best_trajectory))
    if not isinstance(best_entry, Mapping):
        raise RuntimeError("schema-v13 lift catalog has no best_nominal trajectory")
    best_candidate_id = str(best_entry.get("candidate_id"))
    discovered = discover_v9_robustness_sources((catalog_path,))
    if not any(value.candidate_id == best_candidate_id for value in discovered):
        raise RuntimeError("best_nominal is absent from authenticated robustness sources")
    rebound = tuple(
        replace(value, best_first=value.candidate_id == best_candidate_id)
        for value in discovered
    )

    result = run_v9_robustness_campaign(
        (catalog_path,),
        output_path,
        workers=int(workers),
        seed=int(seed),
        local_perturbations=int(campaign.perturbations_per_published),
        best_perturbations=int(campaign.robustness_trials),
        max_nominal_trajectories=int(campaign.selected_grasp_count),
        candidate_discoverer=lambda _roots: rebound,
    )
    return {
        **result,
        "scaled_contact_lift_robustness_schema_version": 1,
        "best_50_selection_alias": "best_nominal",
        "best_50_candidate_id": best_candidate_id,
    }


def run_scaled_contact_downsize_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool,
    target_success_count: int,
    workers: int,
    seed: int = DEFAULT_SEED,
    evidence_anchor_paths: Sequence[str | Path] = (),
    backend: CampaignBackend = DEFAULT_BACKEND,
) -> dict[str, Any]:
    """Execute or resume the registered schema-v13 campaign."""

    if evidence_anchor_paths:
        raise ValueError(
            "schema-v13 sources are fixed by source_aliases; "
            "--evidence-grasp-anchor is not accepted"
        )
    config_file = Path(config_path).expanduser().resolve()
    workspace = Path(output_dir).expanduser().resolve()
    template = load_config(config_file)
    definition = resolve_experiment(template)
    campaign = _campaign(definition)
    if int(seed) != int(campaign.seed):
        raise ValueError("schema-v13 tune seed differs from the registered seed")
    manifest = build_scaled_contact_downsize_manifest(config_file, seed=int(seed))
    initialize_or_resume_campaign(workspace, manifest, resume=bool(resume))

    source_report_path = workspace / "source_audit.json"
    source_report = _existing_stage(workspace, "source_audit", source_report_path)
    # Sources are re-audited even when the report exists.  A stage ledger only
    # protects copied output; this protects the external immutable inputs.
    sources = backend.source_loader(campaign)
    if tuple(sources) != tuple(campaign.source_aliases):
        raise RuntimeError("source loader did not preserve registered source aliases")
    descriptors = [
        _source_descriptor(sources[alias], alias) for alias in campaign.source_aliases
    ]
    fresh_source = {
        "scaled_contact_source_audit_schema_version": SOURCE_AUDIT_SCHEMA_VERSION,
        "complete": True,
        "source_count": len(descriptors),
        "source_aliases": list(campaign.source_aliases),
        "sources": descriptors,
    }
    if source_report is None:
        source_report = _commit_report(
            workspace,
            "source_audit",
            source_report_path,
            fresh_source,
            stage_input={
                "source_catalog_sha256": manifest[
                    "scaled_contact_source_catalog_sha256"
                ],
                "source_manifest_sha256": manifest[
                    "scaled_contact_source_manifest_sha256"
                ],
            },
        )
    elif canonical_sha256(source_report) != canonical_sha256(fresh_source):
        raise RuntimeError("schema-v13 source audit changed on resume")

    static_report_path = workspace / "static_downsize_report.json"
    static_report = _existing_stage(
        workspace, "static_downsize_scan", static_report_path
    )
    if static_report is None:
        static_records = backend.static_runner(
            template,
            campaign,
            sources,
            workspace=workspace,
            resume=bool(resume),
            seed=int(seed),
            workers=int(workers),
        )
        expected_max = (
            int(campaign.stratum_count) * int(campaign.static_retain_per_stratum)
        )
        if len(static_records) > expected_max:
            raise RuntimeError("static runner exceeded the registered retained budget")
        static_report = _commit_report(
            workspace,
            "static_downsize_scan",
            static_report_path,
            {
                "scaled_contact_static_report_schema_version": STATIC_REPORT_SCHEMA_VERSION,
                "complete": True,
                "stratum_count": int(campaign.stratum_count),
                "dls_start_count": (
                    int(campaign.stratum_count)
                    * int(campaign.dls_starts_per_stratum)
                ),
                "candidate_count": len(static_records),
                "static_pass_count": sum(
                    bool(value.get("static_pass", False)) for value in static_records
                ),
                "records": list(static_records),
            },
            stage_input={
                "campaign": campaign.as_config(),
                "source_audit_sha256": file_sha256(source_report_path),
            },
            extra_artifacts=tuple(sorted((workspace / "static_progress").glob("*.json"))),
        )
    static_records = tuple(copy.deepcopy(static_report.get("records", ())))

    dynamic_root = workspace / "dynamic"
    dynamic_report_path = workspace / "dynamic_grasp_report.json"
    dynamic_report = _existing_stage(workspace, "dynamic_grasp", dynamic_report_path)
    if dynamic_report is None:
        dynamic_records = backend.dynamic_runner(
            static_records,
            dynamic_root,
            workers=int(workers),
            resume=bool(resume),
            seed=int(seed),
            controller_seed_count=int(campaign.controllers_per_pose),
        )
        dynamic_report = _commit_report(
            workspace,
            "dynamic_grasp",
            dynamic_report_path,
            {
                "complete": True,
                "candidate_count": len(dynamic_records),
                "raw_grasp_success_count": sum(
                    grasp_stage_succeeded(value) for value in dynamic_records
                ),
                "records": list(dynamic_records),
            },
            stage_input={
                "static_report_sha256": file_sha256(static_report_path),
                "controllers_per_pose": int(campaign.controllers_per_pose),
            },
            extra_artifacts=_candidate_artifacts(dynamic_records, dynamic_root),
        )
    dynamic_records = tuple(copy.deepcopy(dynamic_report.get("records", ())))

    local_report_path = workspace / "local_grasp_refinement_report.json"
    local_report = _existing_stage(
        workspace, "local_grasp_refinement", local_report_path
    )
    if local_report is None:
        local_records = backend.local_grasp_runner(
            dynamic_records,
            workspace,
            campaign=campaign,
            workers=int(workers),
            resume=bool(resume),
            seed=int(seed),
        )
        declared_local_budget = (
            len(campaign.edges_m)
            * len(campaign.mapping_modes)
            * int(campaign.local_poses_per_edge_mapping)
            * int(campaign.local_refine_per_pose)
        )
        if len(local_records) > declared_local_budget:
            raise RuntimeError("local grasp runner exceeded its registered budget")
        generated_reports = tuple(
            sorted((workspace / "static").glob("v13_edge_*/*.json"))
        )
        local_report = _commit_report(
            workspace,
            "local_grasp_refinement",
            local_report_path,
            {
                "complete": True,
                "declared_parent_count": (
                    len(campaign.edges_m)
                    * len(campaign.mapping_modes)
                    * int(campaign.local_poses_per_edge_mapping)
                ),
                "declared_candidate_budget": declared_local_budget,
                "candidate_count": len(local_records),
                "grasp_success_count": sum(
                    grasp_stage_succeeded(value) for value in local_records
                ),
                "records": list(local_records),
                "stop_reason": (
                    "all_edge_mapping_local_budgets_processed"
                    if local_records
                    else "no_local_candidate_survived_real_geometry_projection"
                ),
            },
            stage_input={
                "dynamic_report_sha256": file_sha256(dynamic_report_path),
                "local_poses_per_edge_mapping": int(
                    campaign.local_poses_per_edge_mapping
                ),
                "local_refine_per_pose": int(campaign.local_refine_per_pose),
            },
            extra_artifacts=tuple(
                (
                    *_candidate_artifacts(local_records, workspace / "dynamic"),
                    *generated_reports,
                )
            ),
        )
    local_records = tuple(copy.deepcopy(local_report.get("records", ())))
    combined_dynamic_records = tuple((*dynamic_records, *local_records))

    measured_report_path = workspace / "measured_grasp_report.json"
    measured_report = _existing_stage(
        workspace, "measured_grasp_finalization", measured_report_path
    )
    if measured_report is None:
        promotable = tuple(
            value for value in combined_dynamic_records if grasp_stage_succeeded(value)
        )
        measured_records = backend.measured_runner(
            promotable,
            dynamic_root,
            workers=int(workers),
            resume=bool(resume),
        )
        grasp_count, _ = _success_counts(measured_records)
        measured_report = _commit_report(
            workspace,
            "measured_grasp_finalization",
            measured_report_path,
            {
                "complete": True,
                "candidate_count": len(measured_records),
                "grasp_success_count": grasp_count,
                "records": list(measured_records),
            },
            stage_input={
                "dynamic_report_sha256": file_sha256(dynamic_report_path),
                "local_report_sha256": file_sha256(local_report_path),
            },
            extra_artifacts=_candidate_artifacts(measured_records, dynamic_root),
        )
    measured_records = tuple(copy.deepcopy(measured_report.get("records", ())))

    # Manipulation search is a target-independent, registered full campaign.
    # Keep its expensive candidate evidence in one authenticated source report,
    # then bind a cheap target-specific report for the target-1 and target-5
    # publication/resume flows.  This prevents a completed target-1 catalog
    # from being mistaken for the target-5 stage while avoiding duplicate
    # probes and local refinements.
    manipulation_source_path = workspace / "manipulation" / "source_report.json"
    manipulation_source = _existing_stage(
        workspace, "manipulation_source", manipulation_source_path
    )
    if manipulation_source is None:
        manipulation_records = backend.manipulation_runner(
            measured_records,
            workspace,
            campaign=campaign,
            target_success_count=int(campaign.selected_grasp_count),
            seed=int(seed),
            workers=int(workers),
            resume=bool(resume),
        )
        _, full_count = _success_counts(manipulation_records)
        manipulation_source = _commit_report(
            workspace,
            "manipulation_source",
            manipulation_source_path,
            {
                "complete": True,
                "search_scope": "registered_full_downsize_campaign",
                "candidate_count": len(manipulation_records),
                "full_success_count": full_count,
                "records": list(manipulation_records),
            },
            stage_input={
                "measured_report_sha256": file_sha256(measured_report_path),
                "budget": campaign.budget_config(),
            },
            extra_artifacts=_candidate_artifacts(manipulation_records, workspace),
        )
    manipulation_records = tuple(
        copy.deepcopy(manipulation_source.get("records", ()))
    )
    manipulation_stage = f"manipulation_target_{int(target_success_count)}"
    manipulation_report_path = (
        workspace
        / "manipulation"
        / f"target_{int(target_success_count)}_report.json"
    )
    manipulation_report = _existing_stage(
        workspace, manipulation_stage, manipulation_report_path
    )
    if manipulation_report is None:
        _, full_count = _success_counts(manipulation_records)
        manipulation_report = _commit_report(
            workspace,
            manipulation_stage,
            manipulation_report_path,
            {
                "complete": True,
                "candidate_count": len(manipulation_records),
                "full_success_count": full_count,
                "target_success_count": int(target_success_count),
                "target_reached": full_count >= int(target_success_count),
                "source_report": str(manipulation_source_path.relative_to(workspace)),
                "source_report_sha256": file_sha256(manipulation_source_path),
            },
            stage_input={
                "manipulation_source_report_sha256": file_sha256(
                    manipulation_source_path
                ),
                "target_success_count": int(target_success_count),
            },
        )

    catalog_stage = f"catalog_target_{int(target_success_count)}"
    catalog_report_path = workspace / "catalogs" / f"target_{target_success_count}" / "report.json"
    catalog_report = _existing_stage(workspace, catalog_stage, catalog_report_path)
    if catalog_report is None:
        root = catalog_report_path.parent
        grasp_catalog = _publish_catalog(
            rank_dynamic_grasp_results(measured_records),
            dynamic_root,
            root / "grasp_pose",
            target_success_count=int(target_success_count),
            kind="grasp_pose",
            exporter=backend.grasp_catalog_exporter,
        )
        manipulation_catalog = _publish_catalog(
            manipulation_records,
            workspace,
            root / "manipulation",
            target_success_count=int(target_success_count),
            kind="manipulation",
            exporter=backend.manipulation_catalog_exporter,
        )
        catalogs = {
            key: str(path.relative_to(workspace))
            for key, path in (
                ("grasp_pose", grasp_catalog),
                ("manipulation", manipulation_catalog),
            )
            if path is not None
        }
        catalog_artifacts: list[Path] = []
        for path in (grasp_catalog, manipulation_catalog):
            if path is not None:
                catalog_artifacts.extend(authenticated_catalog_artifact_paths(path))
        grasp_count, _ = _success_counts(measured_records)
        _, full_count = _success_counts(manipulation_records)
        catalog_report = _commit_report(
            workspace,
            catalog_stage,
            catalog_report_path,
            {
                "complete": True,
                "grasp_success_count": grasp_count,
                "full_success_count": full_count,
                "target_success_count": int(target_success_count),
                "target_reached": full_count >= int(target_success_count),
                "catalogs": catalogs,
            },
            stage_input={
                "measured_report_sha256": file_sha256(measured_report_path),
                "manipulation_target_report_sha256": file_sha256(
                    manipulation_report_path
                ),
                "target_success_count": int(target_success_count),
            },
            extra_artifacts=tuple(dict.fromkeys(catalog_artifacts)),
        )

    catalogs = copy.deepcopy(dict(catalog_report.get("catalogs", {})))
    grasp_count = int(catalog_report.get("grasp_success_count", 0))
    full_count = int(catalog_report.get("full_success_count", 0))
    robustness: dict[str, Any] = {}
    grasp_catalog_relative = catalogs.get("grasp_pose")
    if grasp_count and grasp_catalog_relative:
        grasp_robustness_path = (
            workspace
            / "robustness"
            / f"target_{int(target_success_count)}"
            / "grasp_perturbation_report.json"
        )
        grasp_stage = f"grasp_robustness_target_{int(target_success_count)}"
        grasp_robustness = _existing_stage(
            workspace, grasp_stage, grasp_robustness_path
        )
        if grasp_robustness is None:
            grasp_robustness = backend.grasp_robustness_runner(
                measured_records,
                grasp_robustness_path,
                workers=int(workers),
                seed=int(seed),
                campaign=campaign,
                catalog_path=workspace / grasp_catalog_relative,
            )
            grasp_robustness = _commit_report(
                workspace,
                grasp_stage,
                grasp_robustness_path,
                {"complete": True, **copy.deepcopy(dict(grasp_robustness))},
                stage_input={
                    "catalog_sha256": file_sha256(
                        workspace / grasp_catalog_relative
                    ),
                    "local_trials": int(campaign.perturbations_per_published),
                    "best_trials": int(campaign.robustness_trials),
                    "required_passes": int(campaign.robustness_required_passes),
                    "success_metric": "grasp_success",
                },
            )
        robustness["grasp"] = grasp_robustness
    manipulation_catalog_relative = catalogs.get("manipulation")
    if full_count and manipulation_catalog_relative:
        robustness_path = (
            workspace
            / "robustness"
            / f"target_{int(target_success_count)}"
            / "perturbation_report.json"
        )
        robustness_stage = f"robustness_target_{int(target_success_count)}"
        lift_robustness = _existing_stage(
            workspace, robustness_stage, robustness_path
        )
        if lift_robustness is None:
            robustness_runner = backend.robustness_runner or _default_robustness_runner
            lift_robustness = robustness_runner(
                workspace / manipulation_catalog_relative,
                robustness_path,
                workers=int(workers),
                seed=int(seed),
                campaign=campaign,
            )
            robustness_payload = {
                "complete": True,
                **copy.deepcopy(dict(lift_robustness)),
            }
            lift_robustness = _commit_report(
                workspace,
                robustness_stage,
                robustness_path,
                robustness_payload,
                stage_input={
                    "catalog_sha256": file_sha256(
                        workspace / manipulation_catalog_relative
                    ),
                    "local_trials": int(campaign.perturbations_per_published),
                    "best_trials": int(campaign.robustness_trials),
                    "required_passes": int(campaign.robustness_required_passes),
                    "success_metric": "full_success",
                },
            )
        robustness["lift"] = lift_robustness

    return {
        "scaled_contact_downsize_campaign_result_schema_version": (
            CAMPAIGN_RESULT_SCHEMA_VERSION
        ),
        "complete": True,
        "experiment_id": definition.experiment_id,
        "resume": bool(resume),
        "target_success_count": int(target_success_count),
        "grasp_success_count": grasp_count,
        "full_success_count": full_count,
        "target_reached": full_count >= int(target_success_count),
        "fixed_mass_geometry_ablation": True,
        "catalogs": catalogs,
        "robustness": robustness or None,
        "stop_reason": (
            "target_full_success_count_reached"
            if full_count >= int(target_success_count)
            else "declared_budget_exhausted_before_target_full_success_count"
        ),
    }


__all__ = [
    "CAMPAIGN_RESULT_SCHEMA_VERSION",
    "CATALOG_METADATA_SCHEMA_VERSION",
    "DEFAULT_BACKEND",
    "DEFAULT_SEED",
    "CampaignBackend",
    "build_scaled_contact_downsize_manifest",
    "campaign_strata",
    "run_scaled_contact_downsize_campaign",
]
