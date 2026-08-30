"""Deterministic schema-v16 rolling-slip rescue around one sealed v15 pose.

The source candidate is authenticated and treated as immutable evidence.  A
candidate changes only the registered v16 feedback gains and the identities
that necessarily bind the new controller semantics.  Every grid point is run
from the initial free-dynamics state; the selected point is run once more with
a complete NPZ trace for direct Viewer use.
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import file_sha256, json_compatible, write_json
from ..config import load_config, validate_config
from ..experiment import JointPairFeedbackParameters
from ..experiments.opposed_face_palm_down_joint_pair_near_zero_contact_preserving_planned_lift import (
    EXPERIMENT_ID as V15_EXPERIMENT_ID,
)
from ..experiments.opposed_face_palm_down_joint_pair_near_zero_rolling_slip_contact_preserving_planned_lift import (
    EXPERIMENT_DEFINITION as V16_EXPERIMENT,
    EXPERIMENT_ID as V16_EXPERIMENT_ID,
    JOINT_PAIR_FEEDBACK as V16_DEFAULT_FEEDBACK,
)
from ..grasp_pose import canonical_sha256
from ..simulation import run_simulation
from ..v16_identity import install_v16_top_level_identities
from .joint_pair_near_zero_candidate_artifacts import (
    authenticate_v15_candidate_artifacts,
)


ROLLING_SLIP_RESCUE_CAMPAIGN_SCHEMA_VERSION = 1
ROLLING_SLIP_RESCUE_RESULT_SCHEMA_VERSION = 1
ROLLING_SLIP_RESCUE_CATALOG_SCHEMA_VERSION = 1

DEFAULT_SOURCE_CANDIDATE_ID = 15204225421299876
DEFAULT_SOURCE_DIRECTORY = Path(
    "artifacts/left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift/tune/"
    "formal_campaign_v15_4_refinement_continuation_v1/catalogs/target_1/"
    "manipulation/candidate_15204225421299876"
)
DEFAULT_SOURCE_CONFIG = DEFAULT_SOURCE_DIRECTORY / "resolved_config.json"
DEFAULT_SOURCE_TRACE = DEFAULT_SOURCE_DIRECTORY / "trace.npz"
DEFAULT_OUTPUT_DIRECTORY = Path(
    "artifacts/left_opposed_face_palm_down_joint_pair_near_zero_rolling_slip_"
    "contact_preserving_planned_lift/tune/"
    "rolling_slip_rescue_from_15204225421299876_v1"
)
DEFAULT_ALIGNMENT_GAINS = (
    0.25,
    0.50,
    0.75,
    0.9325871364810551,
    1.00,
)
DEFAULT_SLIP_RECOVERY_GAINS_RAD_PER_M = (2.0, 4.0, 6.0, 8.0)


SimulationRunner = Callable[..., Mapping[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    expected = json_compatible(value)
    if path.exists():
        if not path.is_file() or _read_json(path) != expected:
            raise RuntimeError(f"committed rescue artifact changed: {path}")
        return
    write_json(path, expected)


def _normalize_positive_grid(
    values: Sequence[float],
    *,
    label: str,
    bounds: tuple[float, float],
) -> tuple[float, ...]:
    result = tuple(sorted({float(value) for value in values}))
    if not result or any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{label} must contain finite positive values")
    if result[0] < bounds[0] - 1e-12 or result[-1] > bounds[1] + 1e-12:
        raise ValueError(f"{label} is outside the registered search range")
    return result


def _validate_trace_archive(path: Path, *, source: bool) -> None:
    if not path.is_file():
        raise RuntimeError(f"trace is missing: {path}")
    with np.load(path, allow_pickle=False) as trace:
        required = {"time", "control_state", "cube_pos", "manipulation_progress"}
        if source:
            missing = sorted(required - set(trace.files))
            if missing:
                raise RuntimeError(
                    "authenticated source trace is missing fields: "
                    + ", ".join(missing)
                )
            count = int(np.asarray(trace["time"]).size)
            if count <= 0 or any(
                int(np.asarray(trace[name]).shape[0]) != count
                for name in required - {"time"}
            ):
                raise RuntimeError("authenticated source trace has inconsistent lengths")
        elif not trace.files:
            raise RuntimeError("selected rescue trace is empty")


def authenticate_source_evidence(
    source_config_path: str | Path = DEFAULT_SOURCE_CONFIG,
    source_trace_path: str | Path | None = None,
    *,
    source_candidate_id: int = DEFAULT_SOURCE_CANDIDATE_ID,
) -> dict[str, Any]:
    """Authenticate and describe one sealed v15 config/result/trace bundle."""

    config_path = Path(source_config_path).expanduser().resolve()
    bundle = authenticate_v15_candidate_artifacts(
        config_path.parent,
        expected_candidate_id=source_candidate_id,
        require_retained_trace=True,
    )
    if bundle.config_path != config_path:
        raise RuntimeError("source config must be the authenticated resolved_config.json")
    assert bundle.trace_path is not None
    requested_trace = (
        bundle.trace_path
        if source_trace_path is None
        else Path(source_trace_path).expanduser().resolve()
    )
    if not requested_trace.is_file():
        raise FileNotFoundError(requested_trace)
    if file_sha256(requested_trace) != file_sha256(bundle.trace_path):
        raise RuntimeError("source trace does not match the authenticated candidate")
    _validate_trace_archive(requested_trace, source=True)
    config = load_config(config_path)
    if int(config.get("schema_version", 0)) != 15:
        raise RuntimeError("rolling-slip rescue source must use schema v15")
    if config.get("experiment_id") != V15_EXPERIMENT_ID:
        raise RuntimeError("rolling-slip rescue source has the wrong experiment")
    return {
        "candidate_id": int(source_candidate_id),
        "config": config,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "config_semantic_sha256": canonical_sha256(config),
        "result_path": str(bundle.result_path),
        "result_sha256": file_sha256(bundle.result_path),
        "trace_path": str(requested_trace),
        "trace_sha256": file_sha256(requested_trace),
    }


def _candidate_key(
    evidence: Mapping[str, Any],
    alignment_gain: float,
    slip_recovery_gain_rad_per_m: float,
) -> str:
    return canonical_sha256(
        {
            "identity_schema_version": 1,
            "domain": "v16_rolling_slip_rescue_candidate",
            "source_candidate_id": evidence["candidate_id"],
            "source_config_sha256": evidence["config_sha256"],
            "source_trace_sha256": evidence["trace_sha256"],
            "alignment_gain": float(alignment_gain),
            "slip_recovery_gain_rad_per_m": float(
                slip_recovery_gain_rad_per_m
            ),
        }
    )


def promote_v15_source_config(
    source_config: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    alignment_gain: float,
    slip_recovery_gain_rad_per_m: float,
) -> dict[str, Any]:
    """Return a validated v16 config without mutating the source mapping."""

    source_snapshot = canonical_sha256(source_config)
    config = copy.deepcopy(dict(source_config))
    key = _candidate_key(
        evidence, alignment_gain, slip_recovery_gain_rad_per_m
    )
    candidate_id = int(key[:15], 16)
    feedback = JointPairFeedbackParameters(
        **{
            **{
                name: getattr(V16_DEFAULT_FEEDBACK, name)
                for name in V16_DEFAULT_FEEDBACK.__dataclass_fields__
            },
            "alignment_gain": float(alignment_gain),
            "slip_recovery_gain_rad_per_m": float(
                slip_recovery_gain_rad_per_m
            ),
        }
    )

    config["schema_version"] = 16
    config["experiment_id"] = V16_EXPERIMENT_ID
    config["description"] = V16_EXPERIMENT.description
    protocol = V16_EXPERIMENT.control_protocol.as_config()
    source_protocol = source_config.get("control_protocol", {})
    if isinstance(source_protocol, Mapping) and "close_s" in source_protocol:
        protocol["close_s"] = float(source_protocol["close_s"])
    config["control_protocol"] = protocol
    config["actual_contact_grasp_pose_campaign"] = (
        V16_EXPERIMENT.actual_contact_grasp_pose_campaign.as_config()
    )
    config["contact_preserving_planned_lift_campaign"] = (
        V16_EXPERIMENT.contact_preserving_planned_lift_campaign.as_config()
    )
    config["joint_pair_alignment"] = (
        V16_EXPERIMENT.joint_pair_alignment.as_config()
    )
    config["joint_pair_feedback"] = feedback.as_config()
    metadata = copy.deepcopy(config.get("candidate_metadata", {}))
    metadata.update(
        {
            "candidate_id": candidate_id,
            "schema_version": 16,
            "v16_rolling_slip_rescue": {
                "schema_version": 1,
                "candidate_key": key,
                "source_candidate_id": int(evidence["candidate_id"]),
                "source_config_sha256": str(evidence["config_sha256"]),
                "source_trace_sha256": str(evidence["trace_sha256"]),
                "alignment_gain": float(alignment_gain),
                "slip_recovery_gain_rad_per_m": float(
                    slip_recovery_gain_rad_per_m
                ),
                "from_initial_no_contact_state": True,
                "source_artifacts_read_only": True,
            },
        }
    )
    config["candidate_metadata"] = metadata
    install_v16_top_level_identities(config)
    validate_config(config)
    if canonical_sha256(source_config) != source_snapshot:
        raise RuntimeError("v15 source config was mutated during promotion")
    return config


def _metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    stage = summary.get("stage_status", {})
    values = summary.get("metrics", {})
    if not isinstance(stage, Mapping) or not isinstance(values, Mapping):
        raise RuntimeError("simulation summary is missing stage_status/metrics")
    planned = values.get("contact_preserving_planned_lift", {})
    if not isinstance(planned, Mapping):
        planned = {}
    per_finger = planned.get("target_face_effective_duty", {})
    if not isinstance(per_finger, Mapping):
        per_finger = values.get("operation_target_face_contact_duty", {})
    duties = [
        float(value)
        for value in per_finger.values()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ] if isinstance(per_finger, Mapping) else []
    simultaneous = planned.get(
        "simultaneous_target_face_effective_duty",
        values.get("operation_target_face_simultaneous_duty", 0.0),
    )
    try:
        duties.append(float(simultaneous))
    except (TypeError, ValueError):
        duties.append(0.0)
    contact_score = min(duties) if duties else 0.0

    def finite_or(value: Any, default: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    rolling = summary.get("rolling_contact_slip", {})
    rolling_fingers = (
        rolling.get("per_finger", {}) if isinstance(rolling, Mapping) else {}
    )
    rolling_maxima = []
    if isinstance(rolling_fingers, Mapping):
        for record in rolling_fingers.values():
            if isinstance(record, Mapping):
                rolling_maxima.append(
                    finite_or(
                        record.get("maximum_cumulative_irrecoverable_slip_m"),
                        0.0,
                    )
                )
    full = bool(summary.get("passed", False)) and bool(
        stage.get("full_success", False)
    )
    return {
        "full_success": full,
        "grasp_success": bool(stage.get("grasp_success", False)),
        "contact_score": float(contact_score),
        "maximum_plan_progress": finite_or(
            planned.get("maximum_plan_progress"), 0.0
        ),
        "final_plan_progress": finite_or(
            planned.get("final_plan_progress"), 0.0
        ),
        "median_lift_m": finite_or(
            values.get("operation_median_lift_m", values.get("median_lift_m")),
            -1.0,
        ),
        "minimum_lift_m": finite_or(
            values.get("operation_minimum_lift_m", values.get("minimum_lift_m")),
            -1.0,
        ),
        "maximum_rolling_material_slip_m": (
            max(rolling_maxima) if rolling_maxima else 0.0
        ),
        "operation_aborted": bool(planned.get("operation_aborted", False)),
        "failed_checks": [str(value) for value in summary.get("failed_checks", [])],
    }


def _rank_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = record["metrics"]
    return (
        not bool(metrics["full_success"]),
        -float(metrics["contact_score"]),
        -float(metrics["maximum_plan_progress"]),
        -float(metrics["median_lift_m"]),
        -float(metrics["minimum_lift_m"]),
        float(record["alignment_gain"]),
        float(record["slip_recovery_gain_rad_per_m"]),
        int(record["candidate_id"]),
    )


def _case_result_semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("result_semantic_sha256", None)
    return canonical_sha256(payload)


def _authenticate_case(
    case_dir: Path,
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    config_path = case_dir / "resolved_config.json"
    result_path = case_dir / "result.json"
    if not config_path.is_file() or not result_path.is_file():
        raise RuntimeError(f"incomplete committed v16 rescue case: {case_dir}")
    config = _read_json(config_path)
    if canonical_sha256(config) != canonical_sha256(expected_config):
        raise RuntimeError(f"v16 rescue case config changed: {case_dir.name}")
    validate_config(config)
    result = _read_json(result_path)
    if result.get("result_semantic_sha256") != _case_result_semantic_sha256(result):
        raise RuntimeError(f"v16 rescue case result digest changed: {case_dir.name}")
    checks = {
        "candidate_id": config["candidate_metadata"]["candidate_id"],
        "config_sha256": file_sha256(config_path),
        "config_semantic_sha256": canonical_sha256(config),
        "summary_sha256": canonical_sha256(result.get("summary")),
    }
    for name, expected in checks.items():
        if result.get(name) != expected:
            raise RuntimeError(f"v16 rescue case {name} changed: {case_dir.name}")
    return result


def _execute_case(
    payload: Mapping[str, Any],
    *,
    simulation_runner: SimulationRunner = run_simulation,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(payload["config"]))
    case_dir = Path(payload["case_dir"])
    case_dir.mkdir(parents=True, exist_ok=True)
    config_path = case_dir / "resolved_config.json"
    if config_path.exists():
        if _read_json(config_path) != json_compatible(config):
            raise RuntimeError(f"v16 rescue partial config changed: {case_dir.name}")
    else:
        write_json(config_path, config)
    if (case_dir / "result.json").is_file():
        return _authenticate_case(case_dir, config)
    summary = json_compatible(dict(simulation_runner(copy.deepcopy(config))))
    result: dict[str, Any] = {
        "rolling_slip_rescue_result_schema_version": (
            ROLLING_SLIP_RESCUE_RESULT_SCHEMA_VERSION
        ),
        "candidate_id": int(payload["candidate_id"]),
        "candidate_key": str(payload["candidate_key"]),
        "alignment_gain": float(payload["alignment_gain"]),
        "slip_recovery_gain_rad_per_m": float(
            payload["slip_recovery_gain_rad_per_m"]
        ),
        "config_sha256": file_sha256(config_path),
        "config_semantic_sha256": canonical_sha256(config),
        "summary_sha256": canonical_sha256(summary),
        "metrics": _metrics(summary),
        "summary": summary,
    }
    result["result_semantic_sha256"] = _case_result_semantic_sha256(result)
    write_json(case_dir / "result.json", result)
    return _authenticate_case(case_dir, config)


def _spawn_case(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _execute_case(payload)


def _manifest(
    evidence: Mapping[str, Any],
    alignment_gains: Sequence[float],
    slip_gains: Sequence[float],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    body = {
        "rolling_slip_rescue_campaign_schema_version": (
            ROLLING_SLIP_RESCUE_CAMPAIGN_SCHEMA_VERSION
        ),
        "experiment_id": V16_EXPERIMENT_ID,
        "source": {
            key: copy.deepcopy(evidence[key])
            for key in (
                "candidate_id",
                "config_path",
                "config_sha256",
                "config_semantic_sha256",
                "result_path",
                "result_sha256",
                "trace_path",
                "trace_sha256",
            )
        },
        "grid": {
            "alignment_gain": list(alignment_gains),
            "slip_recovery_gain_rad_per_m": list(slip_gains),
            "candidate_count": len(candidates),
            "candidate_keys": [str(value["candidate_key"]) for value in candidates],
        },
        "selection_policy": (
            "full_success_then_contact_then_plan_progress_then_lift_v1"
        ),
        "source_artifacts_read_only": True,
    }
    return {**body, "campaign_input_sha256": canonical_sha256(body)}


def _initialize_workspace(
    workspace: Path,
    manifest: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    manifest_path = workspace / "campaign_manifest.json"
    if workspace.exists():
        if not resume:
            raise FileExistsError(
                f"output directory already exists: {workspace}; use --resume"
            )
        if not manifest_path.is_file() or _read_json(manifest_path) != json_compatible(manifest):
            raise RuntimeError("v16 rolling-slip rescue resume inputs changed")
        return
    workspace.mkdir(parents=True)
    write_json(manifest_path, manifest)


def _atomic_trace_name(directory: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=directory, prefix=".trace.", suffix=".npz"
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _authenticate_final(
    directory: Path,
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    config_path = directory / "resolved_config.json"
    result_path = directory / "result.json"
    trace_path = directory / "trace.npz"
    if not all(path.is_file() for path in (config_path, result_path, trace_path)):
        raise RuntimeError("committed v16 rescue final rerun is incomplete")
    if canonical_sha256(_read_json(config_path)) != canonical_sha256(expected_config):
        raise RuntimeError("v16 rescue final config changed")
    _validate_trace_archive(trace_path, source=False)
    result = _read_json(result_path)
    if result.get("result_semantic_sha256") != _case_result_semantic_sha256(result):
        raise RuntimeError("v16 rescue final result digest changed")
    checks = {
        "config_sha256": file_sha256(config_path),
        "trace_sha256": file_sha256(trace_path),
        "summary_sha256": canonical_sha256(result.get("summary")),
    }
    for name, expected in checks.items():
        if result.get(name) != expected:
            raise RuntimeError(f"v16 rescue final {name} changed")
    return result


def _run_or_resume_final(
    workspace: Path,
    selected: Mapping[str, Any],
    *,
    simulation_runner: SimulationRunner,
) -> dict[str, Any]:
    directory = workspace / "best"
    directory.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(dict(selected["config"]))
    config_path = directory / "resolved_config.json"
    if config_path.exists():
        if _read_json(config_path) != json_compatible(config):
            raise RuntimeError("v16 rescue final partial config changed")
    else:
        write_json(config_path, config)
    result_path = directory / "result.json"
    trace_path = directory / "trace.npz"
    if result_path.exists():
        return _authenticate_final(directory, config)
    if trace_path.exists():
        raise RuntimeError(
            "incomplete final rerun has an uncommitted trace; use a new output directory"
        )
    temporary_trace = _atomic_trace_name(directory)
    try:
        summary = json_compatible(
            dict(
                simulation_runner(
                    copy.deepcopy(config), trace_path=temporary_trace
                )
            )
        )
        _validate_trace_archive(temporary_trace, source=False)
        os.replace(temporary_trace, trace_path)
        result: dict[str, Any] = {
            "rolling_slip_rescue_result_schema_version": (
                ROLLING_SLIP_RESCUE_RESULT_SCHEMA_VERSION
            ),
            "final_full_reset_rerun": True,
            "candidate_id": int(selected["candidate_id"]),
            "candidate_key": str(selected["candidate_key"]),
            "alignment_gain": float(selected["alignment_gain"]),
            "slip_recovery_gain_rad_per_m": float(
                selected["slip_recovery_gain_rad_per_m"]
            ),
            "config_sha256": file_sha256(config_path),
            "config_semantic_sha256": canonical_sha256(config),
            "trace_sha256": file_sha256(trace_path),
            "summary_sha256": canonical_sha256(summary),
            "selection_metrics": copy.deepcopy(selected["metrics"]),
            "metrics": _metrics(summary),
            "summary": summary,
        }
        result["result_semantic_sha256"] = _case_result_semantic_sha256(result)
        write_json(result_path, result)
        return _authenticate_final(directory, config)
    finally:
        temporary_trace.unlink(missing_ok=True)


def _viewer_catalog(
    workspace: Path,
    selected: Mapping[str, Any],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    trajectory_id = f"candidate_{int(selected['candidate_id'])}"
    aliases = {"best_attempt": trajectory_id, "best_rescue": trajectory_id}
    if bool(final["metrics"]["full_success"]):
        aliases["best_nominal"] = trajectory_id
    entry_aliases = sorted(aliases)
    return {
        "trajectory_catalog_schema_version": 1,
        "rolling_slip_rescue_viewer_catalog_schema_version": (
            ROLLING_SLIP_RESCUE_CATALOG_SCHEMA_VERSION
        ),
        "experiment_id": V16_EXPERIMENT_ID,
        "catalog_kind": "manipulation",
        "complete": True,
        "selection_policy": (
            "full_success_then_contact_then_plan_progress_then_lift_v1"
        ),
        "success_count": int(bool(final["metrics"]["full_success"])),
        "aliases": aliases,
        "trajectories": [
            {
                "trajectory_id": trajectory_id,
                "candidate_id": int(selected["candidate_id"]),
                "label": "best_rescue",
                "aliases": entry_aliases,
                "rank": 1,
                "classification": (
                    "validated" if final["metrics"]["full_success"] else "diagnostic"
                ),
                "grasp_success": bool(final["metrics"]["grasp_success"]),
                "full_success": bool(final["metrics"]["full_success"]),
                "artifacts": {
                    "resolved_config": "best/resolved_config.json",
                    "result": "best/result.json",
                    "trace": "best/trace.npz",
                    "video": None,
                    "sha256": {
                        "resolved_config": file_sha256(
                            workspace / "best/resolved_config.json"
                        ),
                        "result": file_sha256(workspace / "best/result.json"),
                        "trace": file_sha256(workspace / "best/trace.npz"),
                    },
                },
            }
        ],
    }


def run_rolling_slip_rescue_campaign(
    source_config_path: str | Path = DEFAULT_SOURCE_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    source_trace_path: str | Path | None = None,
    source_candidate_id: int = DEFAULT_SOURCE_CANDIDATE_ID,
    alignment_gains: Sequence[float] = DEFAULT_ALIGNMENT_GAINS,
    slip_recovery_gains_rad_per_m: Sequence[float] = (
        DEFAULT_SLIP_RECOVERY_GAINS_RAD_PER_M
    ),
    resume: bool = False,
    workers: int = 1,
    simulation_runner: SimulationRunner | None = None,
) -> dict[str, Any]:
    """Run or authenticate a deterministic feedback-gain rescue campaign."""

    if isinstance(workers, bool) or int(workers) != workers or int(workers) < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    if simulation_runner is not None and workers != 1:
        raise ValueError("a custom simulation_runner requires workers=1")
    campaign = V16_EXPERIMENT.contact_preserving_planned_lift_campaign
    assert campaign is not None
    assert campaign.alignment_gain_options is not None
    assert campaign.slip_recovery_gain_options_rad_per_m is not None
    normalized_alignment = _normalize_positive_grid(
        alignment_gains,
        label="alignment_gains",
        bounds=(
            campaign.alignment_gain_options[0],
            campaign.alignment_gain_options[-1],
        ),
    )
    normalized_slip = _normalize_positive_grid(
        slip_recovery_gains_rad_per_m,
        label="slip_recovery_gains_rad_per_m",
        bounds=(
            campaign.slip_recovery_gain_options_rad_per_m[0],
            campaign.slip_recovery_gain_options_rad_per_m[-1],
        ),
    )
    evidence = authenticate_source_evidence(
        source_config_path,
        source_trace_path,
        source_candidate_id=source_candidate_id,
    )
    source_config = evidence["config"]
    candidates: list[dict[str, Any]] = []
    for alignment_gain in normalized_alignment:
        for slip_gain in normalized_slip:
            config = promote_v15_source_config(
                source_config,
                evidence,
                alignment_gain=alignment_gain,
                slip_recovery_gain_rad_per_m=slip_gain,
            )
            rescue = config["candidate_metadata"]["v16_rolling_slip_rescue"]
            candidates.append(
                {
                    "candidate_id": int(config["candidate_metadata"]["candidate_id"]),
                    "candidate_key": str(rescue["candidate_key"]),
                    "alignment_gain": alignment_gain,
                    "slip_recovery_gain_rad_per_m": slip_gain,
                    "config": config,
                }
            )
    candidates.sort(
        key=lambda value: (
            value["alignment_gain"],
            value["slip_recovery_gain_rad_per_m"],
            value["candidate_id"],
        )
    )
    manifest = _manifest(
        evidence, normalized_alignment, normalized_slip, candidates
    )
    workspace = Path(output_dir).expanduser().resolve()
    _initialize_workspace(workspace, manifest, resume=bool(resume))

    payloads = [
        {
            **candidate,
            "case_dir": str(
                workspace / "cases" / f"candidate_{candidate['candidate_id']}"
            ),
        }
        for candidate in candidates
    ]
    runner = run_simulation if simulation_runner is None else simulation_runner
    if workers == 1:
        results = [
            _execute_case(payload, simulation_runner=runner) for payload in payloads
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            results = list(executor.map(_spawn_case, payloads))

    # Refuse to publish a selection if the authenticated source changed while
    # MuJoCo was running.
    if file_sha256(evidence["config_path"]) != evidence["config_sha256"]:
        raise RuntimeError("v15 source config changed during rescue campaign")
    if file_sha256(evidence["trace_path"]) != evidence["trace_sha256"]:
        raise RuntimeError("v15 source trace changed during rescue campaign")

    by_candidate = {value["candidate_id"]: value for value in candidates}
    ranked = sorted(results, key=_rank_key)
    selected_result = ranked[0]
    selected = {
        **by_candidate[int(selected_result["candidate_id"])],
        "metrics": copy.deepcopy(selected_result["metrics"]),
    }
    final = _run_or_resume_final(
        workspace,
        selected,
        simulation_runner=runner,
    )
    catalog = _viewer_catalog(workspace, selected, final)
    _write_or_verify_json(workspace / "catalog.json", catalog)
    aliases = copy.deepcopy(catalog["aliases"])
    report_body = {
        "rolling_slip_rescue_campaign_report_schema_version": 1,
        "campaign_input_sha256": manifest["campaign_input_sha256"],
        "experiment_id": V16_EXPERIMENT_ID,
        "source": copy.deepcopy(manifest["source"]),
        "candidate_count": len(ranked),
        "full_success_count": sum(
            bool(value["metrics"]["full_success"]) for value in ranked
        ),
        "selection_policy": manifest["selection_policy"],
        "selected_candidate_id": int(selected["candidate_id"]),
        "selected_grid_metrics": copy.deepcopy(selected["metrics"]),
        "selected_final_metrics": copy.deepcopy(final["metrics"]),
        "aliases": aliases,
        "viewer_catalog": "catalog.json",
        "ranked_candidates": [
            {
                "rank": rank,
                "candidate_id": int(value["candidate_id"]),
                "candidate_key": str(value["candidate_key"]),
                "alignment_gain": float(value["alignment_gain"]),
                "slip_recovery_gain_rad_per_m": float(
                    value["slip_recovery_gain_rad_per_m"]
                ),
                "metrics": copy.deepcopy(value["metrics"]),
                "artifacts": {
                    "resolved_config": str(
                        Path("cases")
                        / f"candidate_{value['candidate_id']}"
                        / "resolved_config.json"
                    ),
                    "result": str(
                        Path("cases")
                        / f"candidate_{value['candidate_id']}"
                        / "result.json"
                    ),
                },
            }
            for rank, value in enumerate(ranked, start=1)
        ],
        "best_artifacts": {
            "resolved_config": "best/resolved_config.json",
            "result": "best/result.json",
            "trace": "best/trace.npz",
            "sha256": {
                "resolved_config": file_sha256(
                    workspace / "best/resolved_config.json"
                ),
                "result": file_sha256(workspace / "best/result.json"),
                "trace": file_sha256(workspace / "best/trace.npz"),
            },
        },
    }
    report = {**report_body, "report_sha256": canonical_sha256(report_body)}
    _write_or_verify_json(workspace / "campaign_report.json", report)
    return {
        "workspace": str(workspace),
        "manifest": copy.deepcopy(manifest),
        "report": copy.deepcopy(report),
        "catalog": copy.deepcopy(catalog),
    }


__all__ = [
    "DEFAULT_ALIGNMENT_GAINS",
    "DEFAULT_OUTPUT_DIRECTORY",
    "DEFAULT_SLIP_RECOVERY_GAINS_RAD_PER_M",
    "DEFAULT_SOURCE_CANDIDATE_ID",
    "DEFAULT_SOURCE_CONFIG",
    "DEFAULT_SOURCE_TRACE",
    "ROLLING_SLIP_RESCUE_CAMPAIGN_SCHEMA_VERSION",
    "ROLLING_SLIP_RESCUE_CATALOG_SCHEMA_VERSION",
    "ROLLING_SLIP_RESCUE_RESULT_SCHEMA_VERSION",
    "authenticate_source_evidence",
    "promote_v15_source_config",
    "run_rolling_slip_rescue_campaign",
]
