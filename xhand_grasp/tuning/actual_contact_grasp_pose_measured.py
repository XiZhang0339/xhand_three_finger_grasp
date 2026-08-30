"""Finalize schema-v9 grasp identity from the measured contact window.

Dynamic acquisition starts with a geometrically proposed nominal qpos.  A
successful run proves contact, but its authoritative grasp shape is the
persisted median over the 250 ms stable window.  This module writes that
measured vector back into ``grasp_pose.nominal_joint_qpos_rad`` and performs a
fresh, complete no-contact-reset simulation.  Only a reacquired grasp whose
new measured median is *exactly* the configured nominal vector is promotable.

The nominal qpos is deliberately absent from ``controller_id``.  Consequently
this finalization may change grasp identity while the approach, preload and
manipulation controller remain byte-for-byte identical.
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..actual_contact_capability import (
    LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID,
    resolve_actual_contact_definition,
)
from ..actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
    bind_candidate_result_semantic_sha256,
)
from ..artifacts import file_sha256, json_compatible, write_json
from ..config import ACTIVE_ACTUATORS, validate_config
from ..grasp_pose import (
    canonical_sha256,
    continuous_window_steps,
    controller_id,
    grasp_pose_id,
)


EXPERIMENT_ID = LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID
CAMPAIGN_KIND = "actual_contact_grasp_pose_measured_finalization"
MEASURED_RESULT_SCHEMA_VERSION = 1
LOCKED_TIMESTEP_S = 0.001
MAX_FIXED_POINT_ITERATIONS = 4


MeasuredExecutor = Callable[
    [Sequence[Mapping[str, Any]], int], Sequence[Mapping[str, Any]]
]
SimulationRunner = Callable[..., Mapping[str, Any]]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _hard_grasp_success(summary: Mapping[str, Any]) -> bool:
    return bool(_mapping(summary.get("stage_status")).get("grasp_success", False))


def _trace_scalar(trace: Mapping[str, Any], name: str) -> int:
    value = np.asarray(trace[name])
    if value.size != 1:
        raise ValueError(f"{name} must be a scalar trace field")
    return int(value.reshape(-1)[0])


def _load_trace(
    trace_or_path: Mapping[str, Any] | str | Path,
) -> tuple[Mapping[str, Any], Any]:
    if isinstance(trace_or_path, Mapping):
        return trace_or_path, None
    archive = np.load(Path(trace_or_path), allow_pickle=False)
    return archive, archive


def extract_measured_grasp_qpos(
    config: Mapping[str, Any],
    trace_or_path: Mapping[str, Any] | str | Path,
    summary_or_result: Mapping[str, Any],
) -> np.ndarray:
    """Authenticate and recompute the persisted 250 ms actual-qpos median."""

    resolve_actual_contact_definition(
        config, context="measured actual-contact finalization"
    )
    summary_value = summary_or_result.get("summary", summary_or_result)
    if not isinstance(summary_value, Mapping) or not _hard_grasp_success(summary_value):
        raise ValueError("measured finalization source is not a grasp success")
    trace, closeable = _load_trace(trace_or_path)
    try:
        required = continuous_window_steps(
            LOCKED_TIMESTEP_S,
            float(config["grasp_pose"]["verify_continuous_s"]),
        )
        start = _trace_scalar(trace, "grasp_stable_window_start_step")
        end = _trace_scalar(trace, "grasp_stable_window_end_step")
        lock = _trace_scalar(trace, "grasp_lock_step")
        if start < 0 or end != lock or end - start + 1 != required:
            raise ValueError("source trace has no exact declared stable grasp window")
        if "grasp_pose_actual_joint_qpos_rad" not in trace:
            raise ValueError("source trace has no raw actual-qpos history")
        history = np.asarray(
            trace["grasp_pose_actual_joint_qpos_rad"], dtype=np.float64
        )
        if (
            history.ndim != 2
            or history.shape[1] != len(ACTIVE_ACTUATORS)
            or end >= history.shape[0]
            or not np.isfinite(history).all()
        ):
            raise ValueError("source actual-qpos history has an invalid shape")
        recomputed = np.percentile(history[start : end + 1], 50.0, axis=0)
        persisted = np.asarray(
            trace["grasp_pose_actual_qpos_rad"], dtype=np.float64
        )
        if (
            persisted.shape != (len(ACTIVE_ACTUATORS),)
            or not np.isfinite(persisted).all()
            or not np.array_equal(recomputed, persisted)
        ):
            raise ValueError("persisted measured qpos does not equal the raw-window median")
        return persisted.copy()
    finally:
        if closeable is not None:
            closeable.close()


def materialize_measured_grasp_config(
    source_config: Mapping[str, Any],
    measured_qpos_rad: Sequence[float] | np.ndarray,
    *,
    source_candidate_id: int,
    source_candidate_sha256: str,
    source_trace_sha256: str,
    fixed_point_iteration: int = 0,
) -> dict[str, Any]:
    """Bind a measured qpos without allowing actuator commands to rebind it."""

    original = copy.deepcopy(dict(source_config))
    before = canonical_sha256(original)
    values = np.asarray(measured_qpos_rad, dtype=np.float64)
    if values.shape != (len(ACTIVE_ACTUATORS),) or not np.isfinite(values).all():
        raise ValueError("measured_qpos_rad must contain eight finite values")
    if not isinstance(source_candidate_id, int) or isinstance(source_candidate_id, bool):
        raise ValueError("source_candidate_id must be an integer")
    if not isinstance(fixed_point_iteration, int) or fixed_point_iteration < 0:
        raise ValueError("fixed_point_iteration must be a non-negative integer")
    source_controller = controller_id(original)
    source_pose = grasp_pose_id(original)
    resolved = copy.deepcopy(original)
    resolved["grasp_pose"]["nominal_joint_qpos_rad"] = {
        name: float(values[index]) for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    metadata = resolved.setdefault("candidate_metadata", {})
    metadata["campaign_kind"] = CAMPAIGN_KIND
    metadata["stage"] = "measured_grasp_pose_finalization"
    metadata["candidate_id"] = int(source_candidate_id)
    metadata["measured_grasp_pose_finalization"] = {
        "source_candidate_id": int(source_candidate_id),
        "source_candidate_sha256": str(source_candidate_sha256),
        "source_grasp_pose_id": source_pose,
        "source_controller_id": source_controller,
        "source_trace_sha256": str(source_trace_sha256),
        "qpos_reference": "persisted_250ms_actual_contact_window_median",
        "preload_command_used_as_pose_evidence": False,
        "fixed_point_iteration": int(fixed_point_iteration),
    }
    if controller_id(resolved) != source_controller:
        raise RuntimeError("measured qpos finalization changed controller_id")
    validate_config(resolved)
    if canonical_sha256(original) != before:
        raise RuntimeError("measured qpos finalization mutated its source config")
    return resolved


def _confined_source_directory(
    dynamic_root: Path, artifact_directory: Any
) -> Path:
    relative = Path(str(artifact_directory))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("dynamic artifact_directory must be confined and relative")
    directory = (dynamic_root / relative).resolve()
    if not directory.is_relative_to(dynamic_root.resolve()):
        raise ValueError("dynamic artifact_directory escapes dynamic_root")
    return directory


def _authenticate_source(
    record: Mapping[str, Any], dynamic_root: Path
) -> dict[str, Any]:
    directory = _confined_source_directory(dynamic_root, record["artifact_directory"])
    config_path = directory / "resolved_config.json"
    result_path = directory / "result.json"
    trace_path = directory / "trace.npz"
    if not all(path.is_file() for path in (config_path, result_path, trace_path)):
        raise RuntimeError(f"measured finalization source is incomplete: {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    authenticate_candidate_result_semantic_sha256(
        result, source=result_path
    )
    identifier = int(record["candidate_id"])
    if int(result.get("candidate_id", -1)) != identifier:
        raise RuntimeError(f"source candidate ID changed: {result_path}")
    candidate_sha = canonical_sha256(config)
    if result.get("candidate_sha256") != candidate_sha:
        raise RuntimeError(f"source candidate semantic hash changed: {result_path}")
    source_pose = grasp_pose_id(config)
    source_controller = controller_id(config)
    if result.get("grasp_pose_id") != source_pose:
        raise RuntimeError(f"source grasp_pose_id changed: {result_path}")
    if result.get("controller_id") != source_controller:
        raise RuntimeError(f"source controller_id changed: {result_path}")
    hashes = _mapping(_mapping(result.get("artifacts")).get("sha256"))
    config_sha = file_sha256(config_path)
    trace_sha = file_sha256(trace_path)
    if hashes.get("resolved_config") != config_sha or hashes.get("trace") != trace_sha:
        raise RuntimeError(f"source artifact hash changed: {directory}")
    actual = extract_measured_grasp_qpos(config, trace_path, result)
    provenance = {
        "source_candidate_id": identifier,
        "source_artifact_directory": str(Path(str(record["artifact_directory"]))),
        "source_config_path": str(config_path),
        "source_result_path": str(result_path),
        "source_trace_path": str(trace_path),
        "source_config_file_sha256": config_sha,
        "source_result_sha256": file_sha256(result_path),
        "source_trace_sha256": trace_sha,
        "source_candidate_sha256": candidate_sha,
        "source_grasp_pose_id": source_pose,
        "source_controller_id": source_controller,
        "evidence_anchor": bool(record.get("evidence_anchor", False)),
        "evidence_anchor_priority": record.get("evidence_anchor_priority"),
    }
    provenance["source_input_sha256"] = canonical_sha256(provenance)
    return {
        "candidate_id": identifier,
        "config": config,
        "measured_qpos_rad": actual.tolist(),
        "source_provenance": provenance,
    }


def prepare_measured_finalization_jobs(
    records: Sequence[Mapping[str, Any]],
    dynamic_root: str | Path,
    *,
    max_iterations: int = MAX_FIXED_POINT_ITERATIONS,
) -> tuple[dict[str, Any], ...]:
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or not 1 <= max_iterations <= MAX_FIXED_POINT_ITERATIONS
    ):
        raise ValueError(
            f"max_iterations must lie within [1, {MAX_FIXED_POINT_ITERATIONS}]"
        )
    root = Path(dynamic_root).expanduser().resolve()
    source_records = [
        record
        for record in records
        if _hard_grasp_success(_mapping(record.get("summary")))
    ]
    identifiers = [int(record["candidate_id"]) for record in source_records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("authoritative dynamic source IDs must be unique")
    jobs: list[dict[str, Any]] = []
    for record in sorted(source_records, key=lambda value: int(value["candidate_id"])):
        source = _authenticate_source(record, root)
        identifier = int(source["candidate_id"])
        relative = Path("measured") / f"candidate_{identifier}"
        jobs.append(
            {
                **source,
                "artifact_directory": str(relative),
                "output_directory": str(root / relative),
                "max_iterations": int(max_iterations),
            }
        )
    return tuple(jobs)


def measured_grasp_pose_succeeded(record: Mapping[str, Any]) -> bool:
    return bool(record.get("measured_grasp_pose_success", False))


def _full_reset_evidence(
    trace_path: Path, summary: Mapping[str, Any]
) -> dict[str, bool]:
    with np.load(trace_path, allow_pickle=False) as trace:
        initialized = bool(np.asarray(trace["initialized_at_pregrasp"]).reshape(-1)[0])
    checks = _mapping(summary.get("checks"))
    return {
        "initialized_at_configured_precontact": initialized,
        "initial_joint_state_matches_precontact": bool(
            checks.get("v6_initial_joint_state_matches_pregrasp_config", False)
        ),
        "settle_has_no_hand_cube_contact": bool(
            checks.get("no_hand_cube_contact_during_settle", False)
        ),
    }


def evaluate_measured_finalization_job(
    job: Mapping[str, Any],
    *,
    simulation_runner: SimulationRunner | None = None,
) -> dict[str, Any]:
    """Run one bounded measured-qpos fixed point and commit it atomically."""

    if simulation_runner is None:
        from ..simulation import run_simulation

        simulation_runner = run_simulation
    output = Path(str(job["output_directory"]))
    if output.exists():
        raise FileExistsError(f"measured candidate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_config = copy.deepcopy(dict(job["config"]))
    source_before = canonical_sha256(source_config)
    provenance = copy.deepcopy(dict(job["source_provenance"]))
    identifier = int(job["candidate_id"])
    measured = np.asarray(job["measured_qpos_rad"], dtype=np.float64)
    history: list[dict[str, Any]] = []
    final_config: dict[str, Any] | None = None
    final_summary: dict[str, Any] | None = None
    final_actual: np.ndarray | None = None
    exact = False
    reacquired = False
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}."
    ) as staging_name:
        staging = Path(staging_name)
        trace_path = staging / "trace.npz"
        for iteration in range(int(job["max_iterations"])):
            config = materialize_measured_grasp_config(
                source_config,
                measured,
                source_candidate_id=identifier,
                source_candidate_sha256=str(provenance["source_candidate_sha256"]),
                source_trace_sha256=str(provenance["source_trace_sha256"]),
                fixed_point_iteration=iteration,
            )
            summary_value = simulation_runner(
                copy.deepcopy(config), trace_path=trace_path, video_path=None
            )
            if not isinstance(summary_value, Mapping) or not trace_path.is_file():
                raise RuntimeError("measured finalization runner produced no summary/trace")
            summary = copy.deepcopy(dict(json_compatible(summary_value)))
            reacquired = _hard_grasp_success(summary)
            actual = (
                extract_measured_grasp_qpos(config, trace_path, summary)
                if reacquired
                else None
            )
            exact = bool(actual is not None and np.array_equal(actual, measured))
            history.append(
                {
                    "iteration": iteration,
                    "configured_nominal_qpos_rad": {
                        name: float(measured[index])
                        for index, name in enumerate(ACTIVE_ACTUATORS)
                    },
                    "measured_actual_qpos_rad": (
                        {
                            name: float(actual[index])
                            for index, name in enumerate(ACTIVE_ACTUATORS)
                        }
                        if actual is not None
                        else None
                    ),
                    "grasp_success": reacquired,
                    "exact_nominal_actual_match": exact,
                    "grasp_pose_id": grasp_pose_id(config),
                    "candidate_sha256": canonical_sha256(config),
                    "summary_sha256": canonical_sha256(summary),
                }
            )
            final_config, final_summary, final_actual = config, summary, actual
            if reacquired and exact:
                break
            if not reacquired:
                break
            assert actual is not None
            measured = actual
        assert final_config is not None and final_summary is not None
        reset = _full_reset_evidence(trace_path, final_summary)
        controller_unchanged = bool(
            controller_id(final_config) == provenance["source_controller_id"]
        )
        full_reset = all(reset.values())
        success = bool(reacquired and exact and controller_unchanged and full_reset)
        if canonical_sha256(source_config) != source_before:
            raise RuntimeError("measured finalization mutated the source config")
        config_path = staging / "resolved_config.json"
        write_json(config_path, final_config)
        candidate_sha = canonical_sha256(final_config)
        payload = bind_candidate_result_semantic_sha256({
            "measured_grasp_pose_result_schema_version": (
                MEASURED_RESULT_SCHEMA_VERSION
            ),
            "candidate_result_schema_version": 1,
            "complete": True,
            "campaign_kind": CAMPAIGN_KIND,
            "stage": "measured_grasp_pose_finalization",
            "candidate_id": identifier,
            "source_candidate_id": identifier,
            "candidate_sha256": candidate_sha,
            "grasp_pose_id": grasp_pose_id(final_config),
            "controller_id": controller_id(final_config),
            "measured_grasp_pose_success": success,
            "grasp_success": success,
            "classification": (
                "measured_actual_contact_grasp_pose_finalized"
                if success
                else "measured_actual_contact_grasp_pose_rejected"
            ),
            "initial_state_source": "configured_no_contact_reset",
            "checkpoint_used": False,
            "source_provenance": provenance,
            "fixed_point": {
                "maximum_iterations": int(job["max_iterations"]),
                "iterations_executed": len(history),
                "converged_exactly": exact,
                "history": history,
            },
            "finalization_checks": {
                "grasp_reacquired": reacquired,
                "actual_median_exactly_matches_configured_nominal": exact,
                "controller_id_unchanged": controller_unchanged,
                "full_reset_reacquisition": full_reset,
                **reset,
            },
            "actual_grasp_pose_qpos_rad": (
                {
                    name: float(final_actual[index])
                    for index, name in enumerate(ACTIVE_ACTUATORS)
                }
                if final_actual is not None
                else None
            ),
            "summary": final_summary,
            "evidence_anchor": bool(provenance.get("evidence_anchor", False)),
            "evidence_anchor_priority": provenance.get("evidence_anchor_priority"),
            "artifacts": {
                "resolved_config": "resolved_config.json",
                "trace": "trace.npz",
                "trace_retained": True,
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "trace": file_sha256(trace_path),
                },
            },
        })
        write_json(staging / "result.json", payload)
        staging.rename(output)
    return {
        **payload,
        "config": copy.deepcopy(final_config),
        "artifact_directory": str(job["artifact_directory"]),
        "reused": False,
    }


def run_measured_finalization_jobs(
    jobs: Sequence[Mapping[str, Any]], workers: int
) -> tuple[dict[str, Any], ...]:
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not jobs:
        return ()
    if workers == 1:
        records = [evaluate_measured_finalization_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            records = list(executor.map(_evaluate_job, jobs, chunksize=1))
    records.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(records)


def _evaluate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_measured_finalization_job(job)


def _load_reusable_measured_candidate(job: Mapping[str, Any]) -> dict[str, Any] | None:
    directory = Path(str(job["output_directory"]))
    if not directory.exists():
        return None
    config_path = directory / "resolved_config.json"
    result_path = directory / "result.json"
    trace_path = directory / "trace.npz"
    if not all(path.is_file() for path in (config_path, result_path, trace_path)):
        raise RuntimeError(f"incomplete measured finalization candidate: {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    authenticate_candidate_result_semantic_sha256(
        result, source=result_path
    )
    if (
        result.get("complete") is not True
        or result.get("campaign_kind") != CAMPAIGN_KIND
        or int(result.get("candidate_id", -1)) != int(job["candidate_id"])
    ):
        raise RuntimeError(f"measured finalization identity changed: {result_path}")
    provenance = _mapping(result.get("source_provenance"))
    expected = _mapping(job.get("source_provenance"))
    if provenance.get("source_input_sha256") != expected.get("source_input_sha256"):
        raise RuntimeError(f"measured finalization source changed: {result_path}")
    hashes = _mapping(_mapping(result.get("artifacts")).get("sha256"))
    if (
        hashes.get("resolved_config") != file_sha256(config_path)
        or hashes.get("trace") != file_sha256(trace_path)
    ):
        raise RuntimeError(f"measured finalization artifact changed: {directory}")
    if result.get("candidate_sha256") != canonical_sha256(config):
        raise RuntimeError(f"measured candidate hash changed: {result_path}")
    if result.get("grasp_pose_id") != grasp_pose_id(config):
        raise RuntimeError(f"measured grasp_pose_id changed: {result_path}")
    if result.get("controller_id") != controller_id(config):
        raise RuntimeError(f"measured controller_id changed: {result_path}")
    success = bool(result.get("measured_grasp_pose_success", False))
    if success:
        actual = extract_measured_grasp_qpos(config, trace_path, result)
        nominal = np.asarray(
            [
                config["grasp_pose"]["nominal_joint_qpos_rad"][name]
                for name in ACTIVE_ACTUATORS
            ],
            dtype=np.float64,
        )
        if not np.array_equal(actual, nominal):
            raise RuntimeError(
                f"measured grasp pose no longer equals its trace: {result_path}"
            )
    return {
        **result,
        "config": config,
        "artifact_directory": str(job["artifact_directory"]),
        "reused": True,
    }


def run_or_resume_measured_grasp_finalization(
    records: Sequence[Mapping[str, Any]],
    dynamic_root: str | Path,
    *,
    workers: int = 1,
    resume: bool = False,
    max_iterations: int = MAX_FIXED_POINT_ITERATIONS,
    executor: MeasuredExecutor = run_measured_finalization_jobs,
) -> tuple[dict[str, Any], ...]:
    jobs = prepare_measured_finalization_jobs(
        records, dynamic_root, max_iterations=max_iterations
    )
    complete: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for job in jobs:
        reusable = _load_reusable_measured_candidate(job) if resume else None
        (complete if reusable is not None else pending).append(reusable or job)
    executed = tuple(executor(tuple(pending), workers)) if pending else ()
    expected = {int(job["candidate_id"]): job for job in pending}
    if (
        {int(value.get("candidate_id", -1)) for value in executed} != set(expected)
        or len(executed) != len(expected)
    ):
        raise RuntimeError("measured executor did not preserve candidate IDs")
    for raw in executed:
        value = copy.deepcopy(dict(raw))
        job = expected[int(value["candidate_id"])]
        if _mapping(value.get("source_provenance")).get(
            "source_input_sha256"
        ) != _mapping(job.get("source_provenance")).get("source_input_sha256"):
            raise RuntimeError("measured executor rebound a different dynamic source")
        value.setdefault("artifact_directory", str(job["artifact_directory"]))
        complete.append(value)
    complete.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(complete)


__all__ = [
    "CAMPAIGN_KIND",
    "EXPERIMENT_ID",
    "LOCKED_TIMESTEP_S",
    "MAX_FIXED_POINT_ITERATIONS",
    "MEASURED_RESULT_SCHEMA_VERSION",
    "MeasuredExecutor",
    "evaluate_measured_finalization_job",
    "extract_measured_grasp_qpos",
    "materialize_measured_grasp_config",
    "measured_grasp_pose_succeeded",
    "prepare_measured_finalization_jobs",
    "run_measured_finalization_jobs",
    "run_or_resume_measured_grasp_finalization",
]
