"""Resumable post-validation for the schema-v11 relative-wrist campaign.

The search campaign and post-validation campaign deliberately have separate
workspaces.  This runner authenticates an immutable v11 manipulation catalog,
reruns every canonical fixed-160-g success from the configured no-contact
state, revalidates each reproduced success at the registered material density,
and gives only the catalog's reproduced ``best_first`` trajectory the exact
50-case pose/friction audit.

Every simulation is committed as an independent directory rename.  A resume
therefore either authenticates and reuses a complete job or fails closed; it
never trusts a partial directory and never silently replaces evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
    authenticated_catalog_artifact_paths,
)
from ..artifacts import REPO_ROOT, file_sha256, json_text, write_json
from ..config import validate_config
from ..grasp_pose import canonical_sha256
from .relative_wrist_pose_validation import (
    DEFAULT_SEED,
    FIXED_MASS_KG,
    apply_pose_friction_perturbation_case,
    classify_pose_friction_robustness,
    generate_pose_friction_perturbation_cases,
    materialize_constant_density_revalidation,
    validation_label,
)
from ..experiments.opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_smooth_vertical_lift import (
    ARTIFACT_ROOT,
    EXPERIMENT_ID,
)


POST_VALIDATION_SCHEMA_VERSION = 1
POST_VALIDATION_RESULT_SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(ARTIFACT_ROOT) / "post_validation"
_SHA256_FIELD = "post_validation_result_sha256"

SimulationRunner = Callable[..., dict[str, Any]]


def _positive_workers(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("workers must be a positive integer")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def _hard_full_success(summary: Mapping[str, Any]) -> bool:
    status = summary.get("stage_status")
    return bool(
        summary.get("passed") is True
        and isinstance(status, Mapping)
        and status.get("grasp_success") is True
        and status.get("manipulation_success") is True
        and status.get("full_success") is True
    )


def _physics_summary_digest(summary: Mapping[str, Any]) -> str:
    """Hash physics evidence while ignoring renderer-only probe metadata."""

    payload = copy.deepcopy(dict(summary))
    payload.pop("video", None)
    return _digest(payload)


def _bind_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    bound = copy.deepcopy(dict(payload))
    bound.pop(_SHA256_FIELD, None)
    bound[_SHA256_FIELD] = _digest(bound)
    return bound


def _authenticate_result(payload: Mapping[str, Any], *, source: Path) -> str:
    expected = payload.get(_SHA256_FIELD)
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"post-validation result has no valid hash: {source}")
    semantic = copy.deepcopy(dict(payload))
    semantic.pop(_SHA256_FIELD, None)
    actual = _digest(semantic)
    if actual != expected:
        raise RuntimeError(f"post-validation result hash mismatch: {source}")
    return actual


def _safe_member(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise RuntimeError(f"catalog has no safe {label} member")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError(f"catalog {label} member is missing or escapes its root")
    return path


@dataclass(frozen=True, slots=True)
class CanonicalManipulationSource:
    candidate_id: str
    trajectory_id: str
    discovery_index: int
    best_first: bool
    config: dict[str, Any]
    summary: dict[str, Any]
    config_path: Path
    result_path: Path
    trace_path: Path
    source_hashes: dict[str, str]


def load_canonical_fixed_160g_sources(
    catalog_path: str | Path,
) -> tuple[CanonicalManipulationSource, ...]:
    """Authenticate and return only canonical v11 full-success trajectories."""

    catalog = Path(catalog_path).expanduser().resolve()
    authenticated_catalog_artifact_paths(catalog)
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("post-validation requires the registered v11 experiment")
    if payload.get("catalog_kind") != "manipulation" or payload.get("complete") is not True:
        raise ValueError("post-validation requires a complete manipulation catalog")
    aliases = payload.get("aliases", {})
    if not isinstance(aliases, Mapping):
        raise ValueError("catalog aliases must be a mapping")
    best_first_target = aliases.get("best_first")
    entries = payload.get("trajectories")
    if not isinstance(entries, list):
        raise ValueError("catalog trajectories must be a list")
    root = catalog.parent.resolve()
    sources: list[CanonicalManipulationSource] = []
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise ValueError("catalog trajectory entries must be objects")
        # Diagnostic and failed entries remain authenticated by the call above,
        # but they are never eligible for post-validation.
        if raw.get("classification") != "success" or raw.get("full_success") is not True:
            continue
        artifacts = raw.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("catalog success has no artifact map")
        config_path = _safe_member(root, artifacts.get("resolved_config"), "config")
        result_path = _safe_member(root, artifacts.get("result"), "result")
        trace_path = _safe_member(root, artifacts.get("trace"), "trace")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(result, dict):
            raise ValueError("catalog success artifacts must be JSON objects")
        if int(result.get("actual_contact_manipulation_candidate_schema_version", 0)) != 1:
            raise ValueError("catalog success is not production manipulation evidence")
        authenticate_candidate_result_semantic_sha256(result, source=result_path)
        summary = result.get("summary")
        if not isinstance(summary, Mapping) or not _hard_full_success(summary):
            raise ValueError("catalog success does not reproduce full-success semantics")
        if result.get("complete") is not True or result.get("full_success") is not True:
            raise ValueError("catalog success result is incomplete")
        if config.get("schema_version") != 11 or config.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError("catalog success is not a schema-v11 configuration")
        if config.get("run_context") is not None:
            raise ValueError("catalog success must be a canonical nominal configuration")
        validate_config(config)
        if not math.isclose(float(config["cube"]["mass_kg"]), FIXED_MASS_KG, abs_tol=1e-12):
            raise ValueError("catalog success is not a fixed-160-g trajectory")
        if not math.isclose(float(config["cube"]["friction"]), 0.8, abs_tol=1e-12):
            raise ValueError("catalog success does not use the nominal friction")
        result_candidate_id = result.get("candidate_id")
        if isinstance(result_candidate_id, bool):
            raise ValueError("catalog candidate ID must be a non-negative integer")
        try:
            candidate_number = int(result_candidate_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "catalog candidate ID must be a non-negative integer"
            ) from exc
        if candidate_number < 0:
            raise ValueError("catalog candidate ID must be a non-negative integer")
        candidate_id = str(candidate_number)
        if str(raw.get("candidate_id", candidate_id)) != candidate_id:
            raise ValueError("catalog and result candidate IDs disagree")
        trajectory_id = str(raw.get("trajectory_id", ""))
        if not trajectory_id:
            raise ValueError("catalog success has no trajectory_id")
        sources.append(
            CanonicalManipulationSource(
                candidate_id=candidate_id,
                trajectory_id=trajectory_id,
                discovery_index=int(raw.get("discovery_index", index)),
                best_first=trajectory_id == best_first_target,
                config=copy.deepcopy(config),
                summary=copy.deepcopy(dict(summary)),
                config_path=config_path,
                result_path=result_path,
                trace_path=trace_path,
                source_hashes={
                    "resolved_config": file_sha256(config_path),
                    "result": file_sha256(result_path),
                    "trace": file_sha256(trace_path),
                },
            )
        )
    if not sources:
        raise ValueError("catalog contains no canonical fixed-160-g full success")
    if sum(source.best_first for source in sources) != 1:
        raise ValueError("eligible catalog successes must contain exactly one best_first")
    identifiers = [source.candidate_id for source in sources]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("eligible catalog candidate IDs must be unique")
    return tuple(sorted(sources, key=lambda source: (source.discovery_index, source.candidate_id)))


def _manifest(
    catalog_path: Path,
    sources: Sequence[CanonicalManipulationSource],
    *,
    seed: int,
    workers: int,
    render_density_videos: bool,
) -> dict[str, Any]:
    input_paths = authenticated_catalog_artifact_paths(catalog_path)
    provenance = {
        str(path): file_sha256(path)
        for path in input_paths
    }
    for optional in (REPO_ROOT / "uv.lock", REPO_ROOT / "xhand_left.xml"):
        if optional.is_file():
            provenance[str(optional.resolve())] = file_sha256(optional)
    for module in (Path(__file__), Path(__file__).with_name("relative_wrist_pose_validation.py")):
        provenance[str(module.resolve())] = file_sha256(module)
    payload = {
        "post_validation_manifest_schema_version": POST_VALIDATION_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "catalog_path": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "seed": int(seed),
        "workers": int(workers),
        "render_density_videos": bool(render_density_videos),
        "source_candidate_ids": [source.candidate_id for source in sources],
        "best_first_candidate_id": next(
            source.candidate_id for source in sources if source.best_first
        ),
        "inputs_sha256": dict(sorted(provenance.items())),
    }
    payload["manifest_sha256"] = _digest(payload)
    return payload


def _initialize_workspace(output: Path, manifest: Mapping[str, Any], *, resume: bool) -> None:
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        path = output / "post_validation_manifest.json"
        if not path.is_file():
            raise RuntimeError("post-validation workspace has no manifest")
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != manifest:
            raise RuntimeError("post-validation manifest/input hash mismatch")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.staging."
    ) as staging_name:
        staging = Path(staging_name)
        write_json(staging / "post_validation_manifest.json", manifest)
        write_json(
            staging / "progress.json",
            {
                "complete": False,
                "committed_job_count": 0,
                "committed_jobs": [],
            },
        )
        staging.rename(output)


def _default_runner(
    config: dict[str, Any],
    *,
    trace_path: Path | None,
    video_path: Path | None,
) -> dict[str, Any]:
    from ..simulation import run_simulation

    return run_simulation(config, trace_path=trace_path, video_path=video_path)


def _run(
    runner: SimulationRunner,
    config: dict[str, Any],
    *,
    trace_path: Path | None,
    video_path: Path | None,
) -> dict[str, Any]:
    summary = runner(
        copy.deepcopy(config), trace_path=trace_path, video_path=video_path
    )
    if not isinstance(summary, dict):
        raise RuntimeError("simulation runner must return a summary object")
    return summary


def _load_job(directory: Path, *, expected: Mapping[str, Any]) -> dict[str, Any]:
    result_path = directory / "result.json"
    config_path = directory / "resolved_config.json"
    trace_path = directory / "trace.npz"
    if not all(path.is_file() for path in (result_path, config_path, trace_path)):
        raise RuntimeError(f"partial post-validation job exists: {directory}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"post-validation result is malformed: {result_path}")
    _authenticate_result(payload, source=result_path)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"post-validation job identity changed: {directory}")
    artifacts = payload.get("artifacts")
    hashes = artifacts.get("sha256") if isinstance(artifacts, Mapping) else None
    if not isinstance(hashes, Mapping):
        raise RuntimeError(f"post-validation job has no artifact hashes: {directory}")
    for name, path in (("resolved_config", config_path), ("trace", trace_path)):
        if hashes.get(name) != file_sha256(path):
            raise RuntimeError(f"post-validation {name} hash mismatch: {directory}")
    video_path = directory / "trajectory.mp4"
    declared_video = artifacts.get("video") if isinstance(artifacts, Mapping) else None
    if declared_video is not None:
        if declared_video != video_path.name or not video_path.is_file():
            raise RuntimeError(f"post-validation video is missing: {directory}")
        if hashes.get("video") != file_sha256(video_path):
            raise RuntimeError(f"post-validation video hash mismatch: {directory}")
    elif video_path.exists():
        raise RuntimeError(f"post-validation job has an undeclared video: {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if canonical_sha256(config) != payload.get("config_sha256"):
        raise RuntimeError(f"post-validation config semantic hash mismatch: {directory}")
    return {
        **copy.deepcopy(payload),
        "directory": directory,
        "config_path": config_path,
        "result_path": result_path,
        "trace_path": trace_path,
        "video_path": video_path if declared_video is not None else None,
        "config": config,
    }


@dataclass(frozen=True, slots=True)
class _PostValidationJob:
    directory: Path
    config: dict[str, Any]
    job_id: str
    family: str
    source: CanonicalManipulationSource
    validation_label_on_success: str | None
    render_video_on_success: bool = False
    trial: int | None = None

    @property
    def expected(self) -> dict[str, Any]:
        return {
            "post_validation_result_schema_version": (
                POST_VALIDATION_RESULT_SCHEMA_VERSION
            ),
            "job_id": self.job_id,
            "family": self.family,
            "source_candidate_id": self.source.candidate_id,
            "trial": self.trial,
            "config_sha256": canonical_sha256(self.config),
        }


def _job(
    directory: Path,
    config: Mapping[str, Any],
    *,
    job_id: str,
    family: str,
    source: CanonicalManipulationSource,
    validation_label_on_success: str | None,
    render_video_on_success: bool = False,
    trial: int | None = None,
) -> _PostValidationJob:
    return _PostValidationJob(
        directory=directory.resolve(),
        config=copy.deepcopy(dict(config)),
        job_id=str(job_id),
        family=str(family),
        source=source,
        validation_label_on_success=validation_label_on_success,
        render_video_on_success=bool(render_video_on_success),
        trial=trial,
    )


def _write_job_staging(
    staging: Path,
    job: _PostValidationJob,
    *,
    runner: SimulationRunner,
) -> None:
    expected = job.expected
    config_path = staging / "resolved_config.json"
    trace_path = staging / "trace.npz"
    write_json(config_path, job.config)
    summary = _run(
        runner,
        copy.deepcopy(job.config),
        trace_path=trace_path,
        video_path=None,
    )
    if not trace_path.is_file():
        raise RuntimeError("post-validation simulation did not create trace.npz")
    full_success = _hard_full_success(summary)
    video_path: Path | None = None
    if full_success and job.render_video_on_success:
        video_path = staging / "trajectory.mp4"
        video_summary = _run(
            runner,
            copy.deepcopy(job.config),
            trace_path=None,
            video_path=video_path,
        )
        if _physics_summary_digest(video_summary) != _physics_summary_digest(summary):
            raise RuntimeError("video rerun changed the post-validation summary")
        if not video_path.is_file():
            raise RuntimeError("video rerun did not create trajectory.mp4")
    artifacts: dict[str, Any] = {
        "resolved_config": config_path.name,
        "trace": trace_path.name,
        "video": video_path.name if video_path is not None else None,
        "sha256": {
            "resolved_config": file_sha256(config_path),
            "trace": file_sha256(trace_path),
        },
    }
    if video_path is not None:
        artifacts["sha256"]["video"] = file_sha256(video_path)
    payload = _bind_result(
        {
            **expected,
            "complete": True,
            "full_reset_rerun": True,
            "initial_state_source": "configured_no_contact_reset",
            "checkpoint_used": False,
            "full_success": full_success,
            "validation_label": (
                job.validation_label_on_success if full_success else None
            ),
            "source": {
                "trajectory_id": job.source.trajectory_id,
                "best_first": job.source.best_first,
                "source_hashes": copy.deepcopy(job.source.source_hashes),
            },
            "summary": summary,
            "artifacts": artifacts,
        }
    )
    write_json(staging / "result.json", payload)


def _persist_job(
    job: _PostValidationJob,
    *,
    runner: SimulationRunner,
) -> dict[str, Any]:
    expected = job.expected
    directory = job.directory
    if directory.exists():
        return _load_job(directory, expected=expected)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=directory.parent, prefix=f".{directory.name}.staging."
    ) as staging_name:
        staging = Path(staging_name)
        _write_job_staging(staging, job, runner=runner)
        staging.rename(directory)
    return _load_job(directory, expected=expected)


def _spawn_job_worker(payload: Mapping[str, Any]) -> dict[str, str]:
    """Run one isolated simulation; the parent process performs the commit."""

    job = payload.get("job")
    staging_value = payload.get("staging")
    runner = payload.get("runner")
    if not isinstance(job, _PostValidationJob) or not isinstance(staging_value, str):
        raise ValueError("spawned post-validation job payload is malformed")
    if not callable(runner):
        raise ValueError("spawned post-validation job has no runner")
    staging = Path(staging_value).resolve()
    _write_job_staging(staging, job, runner=runner)
    return {"job_id": job.job_id, "staging": str(staging)}


def _run_stage(
    jobs: Sequence[_PostValidationJob],
    *,
    workers: int,
    runner: SimulationRunner,
    on_commit: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    """Run one dependency stage and return records in deterministic job order."""

    worker_count = _positive_workers(workers)
    ordered = tuple(jobs)
    identifiers = [job.job_id for job in ordered]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("post-validation stage job IDs must be unique")
    records: dict[str, dict[str, Any]] = {}
    missing: list[_PostValidationJob] = []
    for job in ordered:
        if job.directory.exists():
            record = _load_job(job.directory, expected=job.expected)
            records[job.job_id] = record
            on_commit(record)
        else:
            missing.append(job)
    if not missing:
        return [records[job.job_id] for job in ordered]
    if worker_count == 1:
        for job in missing:
            record = _persist_job(job, runner=runner)
            records[job.job_id] = record
            on_commit(record)
        return [records[job.job_id] for job in ordered]

    temporaries: list[tempfile.TemporaryDirectory[str]] = []
    payloads: list[dict[str, Any]] = []
    try:
        for job in missing:
            job.directory.parent.mkdir(parents=True, exist_ok=True)
            temporary = tempfile.TemporaryDirectory(
                dir=job.directory.parent,
                prefix=f".{job.directory.name}.staging.",
            )
            temporaries.append(temporary)
            payloads.append(
                {
                    "job": job,
                    "staging": str(Path(temporary.name).resolve()),
                    "runner": runner,
                }
            )
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            # executor.map yields in input order even when workers finish out
            # of order.  Only the parent performs each atomic directory rename.
            completed = executor.map(_spawn_job_worker, payloads, chunksize=1)
            for job, temporary, result in zip(missing, temporaries, completed):
                if result != {
                    "job_id": job.job_id,
                    "staging": str(Path(temporary.name).resolve()),
                }:
                    raise RuntimeError("spawned post-validation job identity changed")
                staging = Path(temporary.name)
                if job.directory.exists():
                    raise RuntimeError(
                        f"post-validation job appeared before parent commit: {job.directory}"
                    )
                staging.rename(job.directory)
                record = _load_job(job.directory, expected=job.expected)
                records[job.job_id] = record
                on_commit(record)
    finally:
        for temporary in temporaries:
            temporary.cleanup()
    return [records[job.job_id] for job in ordered]


def _record_path(output: Path, record: Mapping[str, Any], name: str) -> str:
    return str(Path(record[name]).resolve().relative_to(output))


def _catalog_entry(
    output: Path,
    record: Mapping[str, Any],
    *,
    trajectory_id: str,
    aliases: Sequence[str],
) -> dict[str, Any]:
    artifacts = record["artifacts"]
    video_path = record.get("video_path")
    return {
        "trajectory_id": trajectory_id,
        "label": trajectory_id,
        "aliases": list(aliases),
        "classification": "success" if record["full_success"] else "diagnostic",
        "candidate_id": record["job_id"],
        "source_candidate_id": record["source_candidate_id"],
        "full_success": bool(record["full_success"]),
        "validation_label": record.get("validation_label"),
        "failed_checks": copy.deepcopy(record["summary"].get("failed_checks", [])),
        "artifacts": {
            "resolved_config": _record_path(output, record, "config_path"),
            "result": _record_path(output, record, "result_path"),
            "trace": _record_path(output, record, "trace_path"),
            "video": (
                str(Path(video_path).resolve().relative_to(output))
                if video_path is not None
                else None
            ),
            "sha256": copy.deepcopy(artifacts["sha256"]),
        },
    }


def _build_catalog(
    output: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    family: str,
    success_label: str | None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    success_index = 0
    for index, record in enumerate(records, start=1):
        trajectory_id = f"{family}_{index:02d}_{record['job_id']}"
        entry_aliases: list[str] = []
        if record["full_success"]:
            success_index += 1
            alias = f"{family}_success_{success_index}"
            entry_aliases.append(alias)
            aliases[alias] = trajectory_id
            if success_index == 1:
                first_alias = (
                    "first_passing_trial"
                    if family == "pose_friction"
                    else "best_nominal"
                )
                entry_aliases.append(first_alias)
                aliases[first_alias] = trajectory_id
            source = record.get("source", {})
            if family == "fixed_160g" and isinstance(source, Mapping) and source.get("best_first"):
                entry_aliases.append("best_first")
                aliases["best_first"] = trajectory_id
        entries.append(
            _catalog_entry(
                output,
                record,
                trajectory_id=trajectory_id,
                aliases=entry_aliases,
            )
        )
    if success_index == 0 and entries:
        entries[0]["aliases"].append("best_attempt")
        aliases["best_attempt"] = entries[0]["trajectory_id"]
    return {
        "trajectory_catalog_schema_version": 1,
        "post_validation_catalog_schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "catalog_kind": f"post_validation_{family}",
        "complete": True,
        "family": family,
        "validation_label": success_label if success_index else None,
        "success_count": success_index,
        "failure_count": len(entries) - success_index,
        "aliases": aliases,
        "trajectories": entries,
    }


def _write_or_verify(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise RuntimeError(f"persisted post-validation artifact changed: {path}")
    else:
        write_json(path, payload)


def _write_progress(output: Path, records: Sequence[Mapping[str, Any]], *, complete: bool) -> None:
    jobs = sorted({str(record["job_id"]) for record in records})
    write_json(
        output / "progress.json",
        {
            "complete": bool(complete),
            "committed_job_count": len(jobs),
            "committed_jobs": jobs,
        },
    )


def run_relative_wrist_pose_post_validation(
    catalog_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    resume: bool = False,
    seed: int = DEFAULT_SEED,
    workers: int = 1,
    render_density_videos: bool = False,
    simulation_runner: SimulationRunner | None = None,
) -> dict[str, Any]:
    """Execute and atomically persist the registered schema-v11 validation."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    worker_count = _positive_workers(workers)
    catalog = Path(catalog_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    sources = load_canonical_fixed_160g_sources(catalog)
    manifest = _manifest(
        catalog,
        sources,
        seed=seed,
        workers=worker_count,
        render_density_videos=render_density_videos,
    )
    _initialize_workspace(output, manifest, resume=resume)
    execute = _default_runner if simulation_runner is None else simulation_runner
    committed: dict[str, dict[str, Any]] = {}

    def record_commit(record: dict[str, Any]) -> None:
        committed[str(record["job_id"])] = record
        _write_progress(output, tuple(committed.values()), complete=False)

    fixed_jobs = [
        _job(
            output / "jobs" / "fixed_160g" / f"candidate_{source.candidate_id}",
            source.config,
            job_id=f"fixed_160g_{source.candidate_id}",
            family="fixed_160g_full_reset",
            source=source,
            validation_label_on_success=validation_label(
                "fixed_160g", "manipulation"
            ),
        )
        for source in sources
    ]
    fixed_records = _run_stage(
        fixed_jobs,
        workers=worker_count,
        runner=execute,
        on_commit=record_commit,
    )

    density_jobs: list[_PostValidationJob] = []
    for source, fixed in zip(sources, fixed_records):
        if not fixed["full_success"]:
            continue
        density_config = materialize_constant_density_revalidation(
            source.config, source_candidate_id=source.candidate_id
        )
        density_jobs.append(
            _job(
                output
                / "jobs"
                / "constant_density"
                / f"candidate_{source.candidate_id}",
                density_config,
                job_id=f"constant_density_{source.candidate_id}",
                family="constant_density_full_reset",
                source=source,
                validation_label_on_success=validation_label(
                    "constant_density", "manipulation"
                ),
                render_video_on_success=render_density_videos,
            )
        )
    density_records = _run_stage(
        density_jobs,
        workers=worker_count,
        runner=execute,
        on_commit=record_commit,
    )

    best_source = next(source for source in sources if source.best_first)
    fixed_by_id = {
        str(record["source_candidate_id"]): record for record in fixed_records
    }
    robustness_jobs: list[_PostValidationJob] = []
    if fixed_by_id[best_source.candidate_id]["full_success"]:
        for case in generate_pose_friction_perturbation_cases(seed=seed):
            trial_config = apply_pose_friction_perturbation_case(
                best_source.config,
                case,
                source_candidate_id=best_source.candidate_id,
            )
            robustness_jobs.append(
                _job(
                    output
                    / "jobs"
                    / "pose_friction"
                    / f"trial_{case.trial:02d}",
                    trial_config,
                    job_id=(
                        f"pose_friction_{best_source.candidate_id}_{case.trial:02d}"
                    ),
                    family="best_first_pose_friction_50",
                    source=best_source,
                    validation_label_on_success=None,
                    trial=case.trial,
                )
            )
    robustness_records = _run_stage(
        robustness_jobs,
        workers=worker_count,
        runner=execute,
        on_commit=record_commit,
    )

    robustness = classify_pose_friction_robustness(
        [bool(record["full_success"]) for record in robustness_records],
        nominal_full_success=bool(
            fixed_by_id[best_source.candidate_id]["full_success"]
        ),
        family="fixed_160g",
    ).as_dict()
    catalog_specs = (
        (
            output / "fixed_160g_catalog.json",
            fixed_records,
            "fixed_160g",
            validation_label("fixed_160g", "manipulation"),
        ),
        (
            output / "constant_density_catalog.json",
            density_records,
            "constant_density",
            validation_label("constant_density", "manipulation"),
        ),
        (
            output / "pose_friction_catalog.json",
            robustness_records,
            "pose_friction",
            robustness.get("validation_label"),
        ),
    )
    catalogs: dict[str, dict[str, Any]] = {}
    for path, records, family, label in catalog_specs:
        payload = _build_catalog(
            output, records, family=family, success_label=label
        )
        _write_or_verify(path, payload)
        catalogs[family] = {
            "path": str(path.relative_to(output)),
            "sha256": file_sha256(path),
            "success_count": payload["success_count"],
            "failure_count": payload["failure_count"],
        }

    report = {
        "relative_wrist_pose_post_validation_report_schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "workers": worker_count,
        "source_catalog": str(catalog),
        "source_success_count": len(sources),
        "best_first_candidate_id": best_source.candidate_id,
        "fixed_160g": {
            "attempted": len(fixed_records),
            "full_success_count": sum(bool(record["full_success"]) for record in fixed_records),
            "validation_label": (
                validation_label("fixed_160g", "manipulation")
                if any(record["full_success"] for record in fixed_records)
                else None
            ),
        },
        "constant_density": {
            "eligible_after_fixed_rerun": sum(
                bool(record["full_success"]) for record in fixed_records
            ),
            "attempted": len(density_records),
            "full_success_count": sum(bool(record["full_success"]) for record in density_records),
            "validation_label": (
                validation_label("constant_density", "manipulation")
                if any(record["full_success"] for record in density_records)
                else None
            ),
        },
        "best_first_pose_friction": robustness,
        "catalogs": catalogs,
        "failure_is_never_promoted": True,
    }
    report_path = output / "post_validation_report.json"
    _write_or_verify(report_path, report)
    _write_progress(output, tuple(committed.values()), complete=True)
    return copy.deepcopy(report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable schema-v11 fixed/density/pose-friction post-validation"
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="positive number of spawn workers used within each validation stage",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--render-density-videos",
        action="store_true",
        help="rerun passing equal-density jobs with the existing MP4 renderer",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_relative_wrist_pose_post_validation(
        args.catalog,
        args.output_dir,
        resume=args.resume,
        seed=args.seed,
        workers=args.workers,
        render_density_videos=args.render_density_videos,
    )
    print(json_text(report))
    return 0 if report["fixed_160g"]["full_success_count"] > 0 else 2


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI.
    raise SystemExit(main())


__all__ = [
    "CanonicalManipulationSource",
    "DEFAULT_OUTPUT",
    "load_canonical_fixed_160g_sources",
    "main",
    "run_relative_wrist_pose_post_validation",
]
