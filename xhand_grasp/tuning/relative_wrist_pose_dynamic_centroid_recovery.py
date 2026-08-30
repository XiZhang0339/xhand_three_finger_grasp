"""Authenticated out-of-tree dynamic-centroid recovery for schema-v11.

This orchestrator deliberately treats the completed active-set v3 directory as
immutable evidence.  It authenticates the complete hash chain and every one of
the 51 dynamic candidates, runs a small force/contact-centroid controller
ladder first, and only then delegates to the bounded centroid-guided pose
runner.  Any measured grasp and manipulation evidence is produced in a new
directory.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
    bind_candidate_result_semantic_sha256,
)
from ..artifacts import file_sha256
from ..config import validate_config
from ..grasp_pose import grasp_pose_id
from .actual_contact_grasp_pose import apply_precontact_solution
from .pose_preserving_seed_campaign import canonical_sha256
from .relative_wrist_pose_active_set import (
    build_active_set_evaluation_context,
    evaluate_active_set_candidate,
)
from .relative_wrist_pose_dynamic_guided import (
    GuidedRefinementPolicy,
    authenticate_dynamic_record,
    build_dynamic_centroid_candidate_records,
    deduplicate_guided_candidates,
    extract_dynamic_centroid_diagnostic,
    extract_compacted_dynamic_contact_observation,
    generate_controller_balance_proposals,
    run_dynamic_contact_centroid_guided_refinement,
    select_dynamic_centroid_parents,
)


EXPERIMENT_ID = (
    "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift"
)
RECOVERY_SCHEMA_VERSION = 1
EXPECTED_PARENT_STAGE_NAMES = (
    "authenticated_parent",
    "run_target_1_workers_12",
    "fresh_full_scene_source_filter",
    "safe_merit_selection",
    "active_set_batch_1",
    "active_set_batch_2",
    "recovery_dynamic",
    "recovery_local_refinement_decision_1",
    "recovery_joint_controller_local_refinement",
    "recovery_local_refinement_dynamic",
    "recovery_dynamic_merge_1",
    "recovery_measured_grasp_pose_finalization_1",
    "recovery_manipulation_1",
    "recovery_catalogs_1",
    "final_report_1",
)
EXPECTED_DYNAMIC_COUNT = 51
EXPECTED_RETAINED_TRACE_COUNT = 30
EXPECTED_TOMBSTONE_COUNT = 21
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class AuthenticatedV3Recovery:
    parent: Path
    experiment_id: str
    recovery_input_sha256: str
    terminal_stage_sha256: str
    static_source: dict[str, Any]
    dynamic_records: tuple[dict[str, Any], ...]
    authenticated_dynamic: tuple[dict[str, Any], ...]
    retained_trace_count: int
    tombstone_count: int
    evidence_snapshot: dict[str, Any]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _manifest_semantic_sha256(payload: Mapping[str, Any], field: str) -> str:
    value = copy.deepcopy(dict(payload))
    value.pop(field, None)
    return canonical_sha256(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _validate_parent_ledger(parent: Path) -> dict[str, Any]:
    payload = _load_object(parent / "recovery_stage_ledger.json", "v3 stage ledger")
    if int(payload.get("recovery_stage_ledger_schema_version", 0)) != 1:
        raise RuntimeError("v3 stage ledger schema changed")
    stages = payload.get("stages")
    if not isinstance(stages, list) or tuple(
        str(value.get("name")) for value in stages if isinstance(value, Mapping)
    ) != EXPECTED_PARENT_STAGE_NAMES:
        raise RuntimeError("v3 stage ledger is incomplete or has unexpected stages")
    previous: str | None = None
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise RuntimeError("v3 stage ledger contains a malformed stage")
        expected = canonical_sha256(
            {key: value for key, value in stage.items() if key != "stage_sha256"}
        )
        if stage.get("stage_sha256") != expected:
            raise RuntimeError("v3 stage ledger hash chain changed")
        if stage.get("previous_stage_sha256") != previous:
            raise RuntimeError("v3 stage ledger predecessor changed")
        for artifact in stage.get("artifacts", ()):
            relative = Path(str(artifact.get("path")))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("v3 ledger contains an unsafe artifact path")
            path = (parent / relative).resolve()
            if not path.is_relative_to(parent) or not path.is_file():
                raise RuntimeError(f"v3 ledger artifact is missing: {relative}")
            if file_sha256(path) != artifact.get("sha256"):
                raise RuntimeError(f"v3 ledger artifact SHA-256 changed: {relative}")
        previous = str(stage["stage_sha256"])
    return payload


def _source_files_still_match(manifest: Mapping[str, Any]) -> None:
    source_code = manifest.get("source_code")
    if not isinstance(source_code, Mapping) or not source_code:
        raise RuntimeError("v3 manifest has no source-code binding")
    for name, raw in source_code.items():
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"v3 source binding is malformed: {name}")
        path = Path(str(raw.get("path"))).expanduser().resolve()
        if not path.is_file() or file_sha256(path) != raw.get("sha256"):
            raise RuntimeError(f"v3 bound source changed: {name}")


def _load_static_source(parent: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    expected_counts = {1: (145, 0), 2: (113, 1)}
    for batch, (candidate_count, pass_count) in expected_counts.items():
        payload = _load_object(
            parent / "static" / f"active_set_batch_{batch}.json",
            f"v3 static batch {batch}",
        )
        candidates = payload.get("candidates")
        if (
            payload.get("complete") is not True
            or int(payload.get("batch", -1)) != batch
            or not isinstance(candidates, list)
            or len(candidates) != candidate_count
            or sum(bool(value.get("static_pass")) for value in candidates) != pass_count
        ):
            raise RuntimeError(f"v3 static batch {batch} contract changed")
        identifiers = [int(value["candidate_id"]) for value in candidates]
        if len(identifiers) != len(set(identifiers)):
            raise RuntimeError(f"v3 static batch {batch} has duplicate candidate IDs")
        records.extend(
            copy.deepcopy(dict(value))
            for value in candidates
            if bool(value.get("static_pass"))
        )
    if len(records) != 1:
        raise RuntimeError("v3 must contain exactly one static promotion")
    source = records[0]
    config = source.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("v3 static promotion lost its config")
    if canonical_sha256(config) != source.get("candidate_sha256"):
        raise RuntimeError("v3 static promotion semantic hash changed")
    if grasp_pose_id(config) != source.get("grasp_pose_id"):
        raise RuntimeError("v3 static promotion grasp-pose identity changed")
    if not all(
        bool(source.get(field))
        for field in (
            "static_pass",
            "recovery_static_promotable",
            "initial_contact_safety_pass",
        )
    ):
        raise RuntimeError("v3 static promotion lost fresh-safety evidence")
    return source


def _dynamic_report_records(parent: Path) -> tuple[dict[str, Any], ...]:
    dynamic = parent / "dynamic"
    report_paths = (
        dynamic / "recovery_report.json",
        dynamic / "recovery_active_set_local_local_refinement_report.json",
    )
    expected = (6, 45)
    result: list[dict[str, Any]] = []
    for path, count in zip(report_paths, expected):
        report = _load_object(path, "v3 dynamic stage report")
        records = report.get("candidate_records")
        if (
            report.get("complete") is not True
            or int(report.get("dynamic_candidate_count", -1)) != count
            or int(report.get("grasp_success_count", -1)) != 0
            or not isinstance(records, list)
            or len(records) != count
        ):
            raise RuntimeError(f"v3 dynamic report contract changed: {path}")
        result.extend(copy.deepcopy(dict(value)) for value in records)
    identifiers = [int(value["candidate_id"]) for value in result]
    directories = [str(value["artifact_directory"]) for value in result]
    if len(result) != EXPECTED_DYNAMIC_COUNT or len(set(identifiers)) != len(result):
        raise RuntimeError("v3 dynamic candidate IDs changed")
    if len(set(directories)) != len(result):
        raise RuntimeError("v3 dynamic artifact directories are not unique")
    merged = _load_object(dynamic / "merged" / "target_1.json", "v3 dynamic merge")
    if (
        merged.get("complete") is not True
        or int(merged.get("merged_dynamic_candidate_count", -1)) != len(result)
        or list(merged.get("merged_candidate_ids", ())) != identifiers
        or int(merged.get("merged_grasp_success_count", -1)) != 0
    ):
        raise RuntimeError("v3 merged dynamic report changed")
    return tuple(result)


def _authenticate_dynamic_bundle(
    parent: Path, records: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    dynamic_root = parent / "dynamic"
    expected_directories: set[Path] = set()
    authenticated: list[dict[str, Any]] = []
    retained = 0
    tombstones = 0
    for raw in records:
        relative = Path(str(raw["artifact_directory"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("v3 dynamic artifact path is not confined")
        directory = (dynamic_root / relative).resolve()
        if not directory.is_relative_to(dynamic_root):
            raise RuntimeError("v3 dynamic artifact escaped its root")
        if directory.name != f"candidate_{int(raw['candidate_id'])}":
            raise RuntimeError("v3 dynamic artifact directory and candidate ID differ")
        expected_directories.add(directory)
        value = authenticate_dynamic_record(raw, dynamic_root)
        result_path = Path(value["result_path"])
        authenticate_candidate_result_semantic_sha256(
            value["result"], source=result_path
        )
        for field in (
            "candidate_id",
            "candidate_sha256",
            "grasp_pose_id",
            "controller_id",
            "grasp_success",
        ):
            if value["result"].get(field) != raw.get(field):
                raise RuntimeError(f"v3 dynamic report/result mismatch: {field}")
        if value["trace_retained"]:
            retained += 1
            allowed = {"resolved_config.json", "result.json", "trace.npz"}
            with np.load(value["trace_path"], allow_pickle=False) as trace:
                for name in trace.files:
                    np.asarray(trace[name])
        else:
            tombstones += 1
            allowed = {"resolved_config.json", "result.json"}
            artifacts = value["result"].get("artifacts", {})
            digest = str(artifacts.get("trace_sha256_at_evaluation", ""))
            if artifacts.get("trace") is not None or _SHA256.fullmatch(digest) is None:
                raise RuntimeError("v3 compacted trace tombstone is malformed")
            if bool(value["result"].get("grasp_success", False)):
                raise RuntimeError("v3 compacted a successful dynamic trace")
        observed = {path.name for path in directory.iterdir() if path.is_file()}
        nested = [path.name for path in directory.iterdir() if path.is_dir()]
        if observed != allowed or nested:
            raise RuntimeError("v3 dynamic candidate contains partial or extra artifacts")
        authenticated.append(value)
    actual_directories = {
        path.resolve()
        for path in (dynamic_root / "candidates").iterdir()
        if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise RuntimeError("v3 contains missing or orphan dynamic candidate directories")
    if retained != EXPECTED_RETAINED_TRACE_COUNT or tombstones != EXPECTED_TOMBSTONE_COUNT:
        raise RuntimeError("v3 retained/compacted dynamic trace counts changed")
    return tuple(authenticated)


def authenticate_completed_v3_recovery(
    parent_dir: str | Path,
) -> AuthenticatedV3Recovery:
    parent = Path(parent_dir).expanduser().resolve()
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    manifest = _load_object(parent / "recovery_manifest.json", "v3 recovery manifest")
    if (
        int(manifest.get("relative_wrist_active_set_recovery_manifest_schema_version", 0))
        != 2
        or manifest.get("complete") is not True
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or _manifest_semantic_sha256(manifest, "recovery_input_sha256")
        != manifest.get("recovery_input_sha256")
    ):
        raise RuntimeError("v3 recovery manifest is incompatible or changed")
    _source_files_still_match(manifest)
    ledger = _validate_parent_ledger(parent)
    final = _load_object(parent / "recovery_reports" / "target_1.json", "v3 final report")
    if (
        final.get("complete") is not True
        or final.get("recovery_input_sha256") != manifest.get("recovery_input_sha256")
        or int(final.get("static_pass_count", -1)) != 1
        or int(final.get("grasp_success_count", -1)) != 0
        or int(final.get("full_success_count", -1)) != 0
        or final.get("stop_reason") != "dynamic_grasp_not_verified"
    ):
        raise RuntimeError("v3 final report is not the exhausted zero-grasp run")
    static_source = _load_static_source(parent)
    records = _dynamic_report_records(parent)
    authenticated = _authenticate_dynamic_bundle(parent, records)
    descriptors = [
        {
            "candidate_id": int(value["candidate_id"]),
            "candidate_sha256": str(value["candidate_sha256"]),
            "artifact_key": str(value["artifact_key"]),
            "result_file_sha256": str(value["result_file_sha256"]),
            "trace_retained": bool(value["trace_retained"]),
            "trace_file_sha256": value["trace_file_sha256"],
            "trace_sha256_at_evaluation": value["result"]
            .get("artifacts", {})
            .get("trace_sha256_at_evaluation"),
        }
        for value in authenticated
    ]
    snapshot = {
        "dynamic_centroid_parent_snapshot_schema_version": 1,
        "complete": True,
        "parent": str(parent),
        "experiment_id": EXPERIMENT_ID,
        "parent_recovery_input_sha256": str(manifest["recovery_input_sha256"]),
        "parent_terminal_stage_sha256": str(ledger["stages"][-1]["stage_sha256"]),
        "static_source_candidate_id": int(static_source["candidate_id"]),
        "static_source_candidate_sha256": str(static_source["candidate_sha256"]),
        "dynamic_candidate_count": len(descriptors),
        "retained_trace_count": sum(value["trace_retained"] for value in authenticated),
        "tombstone_count": sum(not value["trace_retained"] for value in authenticated),
        "dynamic_evidence": descriptors,
    }
    snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
    return AuthenticatedV3Recovery(
        parent=parent,
        experiment_id=EXPERIMENT_ID,
        recovery_input_sha256=str(manifest["recovery_input_sha256"]),
        terminal_stage_sha256=str(ledger["stages"][-1]["stage_sha256"]),
        static_source=copy.deepcopy(static_source),
        dynamic_records=records,
        authenticated_dynamic=authenticated,
        retained_trace_count=int(snapshot["retained_trace_count"]),
        tombstone_count=int(snapshot["tombstone_count"]),
        evidence_snapshot=snapshot,
    )


def validate_recovery_paths(
    parent_dir: str | Path, output_dir: str | Path
) -> tuple[Path, Path]:
    parent = Path(parent_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if parent == output or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("dynamic-centroid recovery output and parent must be disjoint")
    return parent, output


def recovery_source_hashes() -> dict[str, dict[str, str]]:
    names = (
        "xhand_grasp.tuning.relative_wrist_pose_dynamic_centroid_recovery",
        "xhand_grasp.tuning.relative_wrist_pose_dynamic_guided",
        "xhand_grasp.tuning.relative_wrist_pose_active_set",
        "xhand_grasp.tuning.actual_contact_grasp_pose",
        "xhand_grasp.tuning.actual_contact_grasp_pose_dynamic",
        "xhand_grasp.tuning.actual_contact_grasp_pose_measured",
        "xhand_grasp.tuning.actual_contact_manipulation",
        "xhand_grasp.actual_contact_grasp_pose_catalog",
        "xhand_grasp.simulation",
    )
    result: dict[str, dict[str, str]] = {}
    for name in names:
        module = importlib.import_module(name)
        raw = inspect.getsourcefile(module)
        if raw is None:
            raise RuntimeError(f"recovery dependency has no source file: {name}")
        path = Path(raw).resolve()
        result[name] = {"path": str(path), "sha256": file_sha256(path)}
    return result


def build_recovery_manifest(
    authenticated: AuthenticatedV3Recovery, *, seed: int
) -> dict[str, Any]:
    payload = {
        "dynamic_centroid_recovery_manifest_schema_version": RECOVERY_SCHEMA_VERSION,
        "complete": True,
        "parent": str(authenticated.parent),
        "experiment_id": authenticated.experiment_id,
        "parent_recovery_input_sha256": authenticated.recovery_input_sha256,
        "parent_terminal_stage_sha256": authenticated.terminal_stage_sha256,
        "parent_snapshot_sha256": authenticated.evidence_snapshot["snapshot_sha256"],
        "seed": int(seed),
        "policy": asdict(GuidedRefinementPolicy()),
        "execution_order": (
            "controller_balance_ladder_then_centroid_guided_then_measured_"
            "then_manipulation"
        ),
        "source_code": recovery_source_hashes(),
    }
    payload["recovery_input_sha256"] = canonical_sha256(payload)
    return payload


def _write_or_authenticate(path: Path, payload: Mapping[str, Any], *, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(f"recovery artifact already exists: {path}")
        existing = _load_object(path, "recovery resume artifact")
        if canonical_sha256(existing) != canonical_sha256(payload):
            raise RuntimeError(f"recovery resume input changed: {path}")
        return
    _atomic_json(path, payload)


def _artifact_descriptor(output: Path, path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(output) or not resolved.is_file():
        raise RuntimeError(f"recovery artifact is not a file inside output: {path}")
    return {"path": str(resolved.relative_to(output)), "sha256": file_sha256(resolved)}


def _load_ledger(output: Path) -> dict[str, Any]:
    path = output / "dynamic_centroid_stage_ledger.json"
    if not path.is_file():
        return {"dynamic_centroid_stage_ledger_schema_version": 1, "stages": []}
    payload = _load_object(path, "dynamic-centroid stage ledger")
    if int(payload.get("dynamic_centroid_stage_ledger_schema_version", 0)) != 1:
        raise RuntimeError("unsupported dynamic-centroid stage ledger")
    previous: str | None = None
    for stage in payload.get("stages", ()):
        expected = canonical_sha256(
            {key: value for key, value in stage.items() if key != "stage_sha256"}
        )
        if stage.get("stage_sha256") != expected or stage.get("previous_stage_sha256") != previous:
            raise RuntimeError("dynamic-centroid stage ledger hash chain changed")
        for artifact in stage.get("artifacts", ()):
            path = output / str(artifact["path"])
            if not path.is_file() or file_sha256(path) != artifact.get("sha256"):
                raise RuntimeError(f"dynamic-centroid stage artifact changed: {path}")
        previous = str(stage["stage_sha256"])
    return payload


def _stage_input_without_workers(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("workers", None)
    return result


def _commit_stage(
    output: Path,
    name: str,
    *,
    stage_input: Mapping[str, Any],
    artifacts: Sequence[Path],
    summary: Mapping[str, Any],
) -> None:
    ledger = _load_ledger(output)
    stages = list(ledger["stages"])
    descriptors = sorted(
        (_artifact_descriptor(output, path) for path in set(artifacts)),
        key=lambda value: value["path"],
    )
    existing = next((value for value in stages if value.get("name") == name), None)
    if existing is not None:
        if (
            _stage_input_without_workers(existing.get("stage_input", {}))
            != _stage_input_without_workers(stage_input)
            or existing.get("artifacts") != descriptors
        ):
            raise RuntimeError(f"dynamic-centroid stage resume input changed: {name}")
        return
    record = {
        "name": name,
        "stage_input_sha256": canonical_sha256(stage_input),
        "stage_input": copy.deepcopy(dict(stage_input)),
        "artifacts": descriptors,
        "summary": copy.deepcopy(dict(summary)),
        "previous_stage_sha256": None if not stages else stages[-1]["stage_sha256"],
    }
    record["stage_sha256"] = canonical_sha256(record)
    stages.append(record)
    _atomic_json(
        output / "dynamic_centroid_stage_ledger.json",
        {"dynamic_centroid_stage_ledger_schema_version": 1, "stages": stages},
    )


def result_only_parent_rank(value: Mapping[str, Any]) -> tuple[Any, ...]:
    result = value["result"]
    metrics = result.get("summary", {}).get("metrics", {})
    gate = metrics.get("verify_gate_component_duty", {})
    alignment = metrics.get("contact_alignment", {}).get("verify", {})
    pose = metrics.get("pose_preservation", {})
    effective = int(metrics.get("verify_max_simultaneous_effective_finger_count", 0))
    topology = float(metrics.get("verify_target_face_simultaneous_duty", 0.0))
    height = float(alignment.get("height_spread_p95_m") or math.inf)
    translation = float(pose.get("max_translation_m", math.inf))
    orientation = float(pose.get("max_orientation_drift_deg", math.inf))
    thumb = float(gate.get("thumb_actual_qpos_within_range", 0.0))
    return (
        not bool(result.get("grasp_success", False)),
        max(0, 3 - effective),
        1.0 - topology,
        max(
            0.0,
            height / 0.005 - 1.0,
            translation / 0.0005 - 1.0,
            orientation - 1.0,
            1.0 - thumb,
        ),
        height,
        translation,
        orientation,
        -int(metrics.get("verify_max_consecutive_gate_steps", 0)),
        str(value["artifact_key"]),
    )


def select_parent_evidence(
    authenticated: Sequence[Mapping[str, Any]], *, count: int
) -> tuple[dict[str, Any], ...]:
    if count <= 0:
        raise ValueError("parent count must be positive")
    ranked = sorted(authenticated, key=result_only_parent_rank)
    return tuple(copy.deepcopy(dict(value)) for value in ranked[:count])


def select_guided_parent_evidence(
    authenticated: Sequence[Mapping[str, Any]], *, count: int
) -> tuple[dict[str, Any], ...]:
    """Mirror the production runner's coverage selection before replay."""

    observations = []
    by_artifact: dict[str, Mapping[str, Any]] = {}
    for value in authenticated:
        by_artifact[str(value["artifact_key"])] = value
        observations.append(
            extract_dynamic_centroid_diagnostic(value)
            if value["trace_retained"]
            else extract_compacted_dynamic_contact_observation(
                value["config"],
                value["result"],
                artifact_key=str(value["artifact_key"]),
            )
        )
    selected = select_dynamic_centroid_parents(observations, top_count=count)
    return tuple(
        copy.deepcopy(dict(by_artifact[value.artifact_key])) for value in selected
    )


def _replay_record(
    value: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exactly reproduce one compacted parent trace without touching v3."""

    from ..simulation import run_simulation

    identifier = int(value["candidate_id"])
    destination = output / "parent_trace_replays" / f"candidate_{identifier}"
    original = value["result"]
    evaluation_sha = str(
        original.get("artifacts", {}).get("trace_sha256_at_evaluation", "")
    )
    record = {
        "candidate_id": identifier,
        "candidate_sha256": str(value["candidate_sha256"]),
        "grasp_pose_id": str(original["grasp_pose_id"]),
        "controller_id": str(original["controller_id"]),
        "grasp_success": bool(original.get("grasp_success", False)),
        "artifact_directory": str(destination),
    }
    if destination.is_dir():
        authenticated = authenticate_dynamic_record(record, output)
        authenticate_candidate_result_semantic_sha256(
            authenticated["result"], source=authenticated["result_path"]
        )
        if authenticated["trace_file_sha256"] != evaluation_sha:
            raise RuntimeError("persisted exact replay trace differs from tombstone")
        return record, {
            "candidate_id": identifier,
            "reused": True,
            "trace_sha256": evaluation_sha,
            "summary_sha256": canonical_sha256(original["summary"]),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    ) as temporary:
        staging = Path(temporary)
        config = copy.deepcopy(dict(value["config"]))
        config_path = staging / "resolved_config.json"
        trace_path = staging / "trace.npz"
        _atomic_json(config_path, config)
        summary = _json_safe(
            run_simulation(copy.deepcopy(config), trace_path=trace_path)
        )
        if canonical_sha256(summary) != canonical_sha256(original["summary"]):
            raise RuntimeError("exact replay summary differs from compacted evidence")
        trace_sha = file_sha256(trace_path)
        if trace_sha != evaluation_sha:
            raise RuntimeError("exact replay trace differs from compacted evidence")
        replay_result = copy.deepcopy(dict(original))
        replay_result["artifacts"] = {
            "resolved_config": "resolved_config.json",
            "trace": "trace.npz",
            "trace_retained": True,
            "sha256": {
                "resolved_config": file_sha256(config_path),
                "trace": trace_sha,
            },
            "exact_replay_of_compacted_parent": True,
            "parent_result_file_sha256": str(value["result_file_sha256"]),
        }
        replay_result = bind_candidate_result_semantic_sha256(replay_result)
        _atomic_json(staging / "result.json", replay_result)
        staging.rename(destination)
    return record, {
        "candidate_id": identifier,
        "reused": False,
        "trace_sha256": evaluation_sha,
        "summary_sha256": canonical_sha256(original["summary"]),
    }


def prepare_guidance_evidence(
    authenticated: AuthenticatedV3Recovery,
    output: Path,
    *,
    parent_count: int,
    replay: Callable[[Mapping[str, Any], Path], tuple[dict[str, Any], dict[str, Any]]] = _replay_record,
    selector: Callable[..., tuple[dict[str, Any], ...]] = select_guided_parent_evidence,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], Path]:
    selected = selector(
        authenticated.authenticated_dynamic, count=parent_count
    )
    retained_by_id = {
        int(value["candidate_id"]): copy.deepcopy(dict(record))
        for value, record in zip(
            authenticated.authenticated_dynamic, authenticated.dynamic_records
        )
        if bool(value["trace_retained"])
    }
    replay_diagnostics: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    for value in selected:
        identifier = int(value["candidate_id"])
        if value["trace_retained"]:
            selected_records.append(copy.deepcopy(retained_by_id[identifier]))
        else:
            record, diagnostic = replay(value, output)
            selected_records.append(record)
            replay_diagnostics.append(diagnostic)
    # The ridge model benefits from every authenticated retained trace.  Only
    # compacted candidates that were actually selected are replayed.
    guidance = list(retained_by_id.values())
    existing = {int(value["candidate_id"]) for value in guidance}
    guidance.extend(
        copy.deepcopy(value)
        for value in selected_records
        if int(value["candidate_id"]) not in existing
    )
    report = {
        "dynamic_centroid_parent_trace_replay_schema_version": 1,
        "complete": True,
        "selected_parent_candidate_ids": [int(value["candidate_id"]) for value in selected],
        "selected_tombstone_count": sum(not value["trace_retained"] for value in selected),
        "exact_replay_count": len(replay_diagnostics),
        "replays": replay_diagnostics,
        "guidance_record_count": len(guidance),
    }
    report_path = output / "parent_trace_replays" / "report.json"
    _write_or_authenticate(report_path, report, resume=report_path.exists())
    return tuple(guidance), tuple(selected_records), report_path


def _dynamic_record_artifacts(output: Path, execution: Any) -> tuple[Path, ...]:
    paths: set[Path] = {Path(value).resolve() for value in execution.artifacts}
    dynamic_root = output / "dynamic"
    for value in execution.records:
        directory_value = value.get("artifact_directory")
        if directory_value is not None:
            directory = Path(str(directory_value))
            if not directory.is_absolute():
                directory = dynamic_root / directory
            directory = directory.resolve()
            if directory.is_relative_to(output):
                for name in ("resolved_config.json", "result.json", "trace.npz"):
                    path = directory / name
                    if path.is_file():
                        paths.add(path)
        for field in ("config_path", "result_path", "trace_path"):
            raw = value.get(field)
            if raw is not None:
                path = Path(str(raw)).resolve()
                if path.is_relative_to(output) and path.is_file():
                    paths.add(path)
    return tuple(sorted(paths))


def _run_controller_ladder(
    authenticated: AuthenticatedV3Recovery,
    selected_records: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    workers: int,
) -> Any:
    from .actual_contact_grasp_pose import _run_materialized_local_dynamic_stage

    context = build_active_set_evaluation_context(authenticated.static_source["config"])
    candidates: list[dict[str, Any]] = []
    for raw in selected_records:
        source_root = authenticated.parent / "dynamic"
        value = authenticate_dynamic_record(raw, source_root)
        observation = extract_dynamic_centroid_diagnostic(value)
        proposals = generate_controller_balance_proposals(
            value["config"], observation, count=8
        )
        safe: list[dict[str, Any]] = []
        for config in proposals:
            fresh, gate = evaluate_active_set_candidate(context, config)
            if not (
                fresh.safe
                and fresh.static_result.static_geometry_pass
                and gate.get("passed", False)
            ):
                continue
            promoted = apply_precontact_solution(config, fresh.static_result)
            validate_config(promoted)
            safe.append(promoted)
        candidates.extend(
            build_dynamic_centroid_candidate_records(
                safe,
                parent_candidate_id=int(value["candidate_id"]),
                parent_artifact_key=str(value["artifact_key"]),
                round_index=8,
                kind="controller_balance_ladder",
                stage="dynamic_centroid_controller_ladder",
            )
        )
    materialized = deduplicate_guided_candidates(candidates)
    return _run_materialized_local_dynamic_stage(
        materialized,
        output,
        stage="dynamic_centroid_controller_ladder",
        workers=workers,
    )


def _execution_has_grasp(execution: Any) -> bool:
    return int(execution.summary.get("grasp_success_count", 0)) > 0


def recovery_stop_reason(
    *, grasp_count: int, measured_count: int, full_count: int
) -> str:
    if grasp_count == 0:
        return "dynamic_centroid_grasp_not_verified"
    if measured_count == 0:
        return "measured_grasp_pose_not_verified"
    if full_count == 0:
        return "manipulation_full_success_not_verified"
    return "manipulation_full_success_verified"


def _best_attempt_report(executions: Sequence[Any]) -> dict[str, Any]:
    records = [value for execution in executions for value in execution.records]
    ranked = sorted(
        records,
        key=lambda value: (
            not bool(value.get("grasp_success", False)),
            -int(
                value.get("summary", {})
                .get("metrics", {})
                .get("verify_max_simultaneous_effective_finger_count", 0)
            ),
            -float(
                value.get("summary", {})
                .get("metrics", {})
                .get("verify_target_face_simultaneous_duty", 0.0)
            ),
            -int(
                value.get("summary", {})
                .get("metrics", {})
                .get("verify_max_consecutive_gate_steps", 0)
            ),
            int(value["candidate_id"]),
        ),
    )
    return {
        "dynamic_centroid_best_attempt_report_schema_version": 1,
        "complete": True,
        "candidate_count": len(records),
        "grasp_success_count": sum(bool(value.get("grasp_success")) for value in records),
        "best_attempts": [
            {
                "candidate_id": int(value["candidate_id"]),
                "candidate_sha256": str(value["candidate_sha256"]),
                "grasp_pose_id": str(value["grasp_pose_id"]),
                "controller_id": str(value["controller_id"]),
                "grasp_success": bool(value.get("grasp_success", False)),
                "artifact_directory": str(value.get("artifact_directory", "")),
                "rank_evidence": copy.deepcopy(value.get("rank_evidence", {})),
            }
            for value in ranked[:10]
        ],
    }


def run_dynamic_centroid_recovery(
    parent_dir: str | Path,
    output_dir: str | Path,
    *,
    workers: int,
    resume: bool,
    target_success_count: int,
    seed: int = 20260821,
) -> dict[str, Any]:
    if workers <= 0 or isinstance(workers, bool):
        raise ValueError("workers must be a positive integer")
    if target_success_count not in (1, 5):
        raise ValueError("target_success_count must be 1 or 5")
    if seed < 0 or isinstance(seed, bool):
        raise ValueError("seed must be a non-negative integer")
    parent, output = validate_recovery_paths(parent_dir, output_dir)
    authenticated = authenticate_completed_v3_recovery(parent)
    manifest = build_recovery_manifest(authenticated, seed=seed)
    run_manifest = {
        "dynamic_centroid_recovery_run_manifest_schema_version": 1,
        "complete": True,
        "recovery_input_sha256": manifest["recovery_input_sha256"],
        "target_success_count": target_success_count,
        "workers": workers,
        "worker_count_changes_numerical_result": False,
    }
    run_manifest["run_input_sha256"] = canonical_sha256(run_manifest)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "parent_snapshot.json"
    manifest_path = output / "recovery_manifest.json"
    run_path = output / "run_manifests" / f"target_{target_success_count}" / f"workers_{workers}.json"
    _write_or_authenticate(snapshot_path, authenticated.evidence_snapshot, resume=resume)
    _write_or_authenticate(manifest_path, manifest, resume=resume)
    _write_or_authenticate(run_path, run_manifest, resume=resume)
    _commit_stage(
        output,
        "authenticated_parent",
        stage_input={"recovery_input_sha256": manifest["recovery_input_sha256"]},
        artifacts=(snapshot_path, manifest_path),
        summary={
            "dynamic_candidate_count": len(authenticated.dynamic_records),
            "retained_trace_count": authenticated.retained_trace_count,
            "tombstone_count": authenticated.tombstone_count,
        },
    )
    _commit_stage(
        output,
        f"run_target_{target_success_count}_workers_{workers}",
        stage_input=run_manifest,
        artifacts=(run_path,),
        summary={"target_success_count": target_success_count, "workers": workers},
    )

    policy = GuidedRefinementPolicy()
    guidance_records, selected_records, replay_report = prepare_guidance_evidence(
        authenticated, output, parent_count=policy.parent_count
    )
    replay_artifacts = [replay_report]
    for raw in selected_records:
        path = Path(str(raw["artifact_directory"]))
        if path.is_absolute() and path.is_relative_to(output):
            replay_artifacts.extend(
                value for value in path.iterdir() if value.is_file()
            )
    _commit_stage(
        output,
        "parent_trace_selection_and_exact_replay",
        stage_input={
            "recovery_input_sha256": manifest["recovery_input_sha256"],
            "parent_count": policy.parent_count,
        },
        artifacts=tuple(replay_artifacts),
        summary=_load_object(replay_report, "parent replay report"),
    )

    ladder = _run_controller_ladder(
        authenticated, selected_records, output, workers=workers
    )
    ladder_artifacts = _dynamic_record_artifacts(output, ladder)
    _commit_stage(
        output,
        "controller_balance_ladder",
        stage_input={
            "recovery_input_sha256": manifest["recovery_input_sha256"],
            "workers": workers,
            "selected_parent_sha256": canonical_sha256(
                [value["candidate_sha256"] for value in selected_records]
            ),
        },
        artifacts=ladder_artifacts,
        summary=ladder.summary,
    )
    executions = [ladder]
    if not _execution_has_grasp(ladder):
        guided = run_dynamic_contact_centroid_guided_refinement(
            guidance_records,
            authenticated.static_source,
            authenticated.parent / "dynamic",
            output,
            workers=workers,
            seed=seed,
            stage="dynamic_centroid_guided",
            policy=policy,
        )
        executions.append(guided)
        guided_report = output / "dynamic" / "dynamic_centroid_guided_guided_report.json"
        guided_artifacts = set(_dynamic_record_artifacts(output, guided))
        if guided_report.is_file():
            guided_artifacts.add(guided_report)
        _commit_stage(
            output,
            "centroid_guided_pose_and_controller",
            stage_input={
                "recovery_input_sha256": manifest["recovery_input_sha256"],
                "workers": workers,
                "guidance_candidate_sha256": canonical_sha256(
                    [value["candidate_sha256"] for value in guidance_records]
                ),
            },
            artifacts=tuple(guided_artifacts),
            summary=guided.summary,
        )

    from .actual_contact_grasp_pose import (
        CampaignStageExecution,
        _publish_campaign_catalogs,
        _run_manipulation_stage,
        _run_measured_grasp_pose_finalization_stage,
    )

    dynamic_records = tuple(
        copy.deepcopy(dict(value))
        for execution in executions
        for value in execution.records
    )
    combined = CampaignStageExecution(
        records=dynamic_records,
        artifacts=tuple(
            path for execution in executions for path in execution.artifacts
        ),
        summary={
            "dynamic_candidate_count": len(dynamic_records),
            "grasp_success_count": sum(
                bool(value.get("grasp_success")) for value in dynamic_records
            ),
        },
    )
    best_report = _best_attempt_report(executions)
    best_path = output / "diagnostics" / "best_attempts.json"
    _write_or_authenticate(best_path, best_report, resume=best_path.exists())
    _commit_stage(
        output,
        "dynamic_best_attempts",
        stage_input={"recovery_input_sha256": manifest["recovery_input_sha256"]},
        artifacts=(best_path,),
        summary={
            "candidate_count": best_report["candidate_count"],
            "grasp_success_count": best_report["grasp_success_count"],
        },
    )

    grasp_count = int(combined.summary["grasp_success_count"])
    measured_records: tuple[dict[str, Any], ...] = ()
    manipulation_records: tuple[dict[str, Any], ...] = ()
    measured_count = 0
    full_count = 0
    catalogs: dict[str, str] = {}
    if grasp_count:
        measured = _run_measured_grasp_pose_finalization_stage(
            combined.records,
            output,
            stage=f"dynamic_centroid_target_{target_success_count}",
            workers=workers,
        )
        measured_records = tuple(measured.records)
        measured_count = len(measured_records)
        _commit_stage(
            output,
            f"measured_grasp_pose_finalization_{target_success_count}",
            stage_input={
                "recovery_input_sha256": manifest["recovery_input_sha256"],
                "workers": workers,
                "dynamic_candidate_sha256": canonical_sha256(
                    [value["candidate_sha256"] for value in combined.records]
                ),
            },
            artifacts=_dynamic_record_artifacts(output, measured),
            summary=measured.summary,
        )
    if measured_count:
        manipulation = _run_manipulation_stage(
            measured_records,
            output,
            target_success_count=target_success_count,
            seed=seed,
            workers=workers,
            stage="dynamic_centroid",
            enable_local_refinement=True,
        )
        manipulation_records = tuple(manipulation.records)
        full_count = int(manipulation.summary.get("full_success_count", 0))
        _commit_stage(
            output,
            f"manipulation_{target_success_count}",
            stage_input={
                "recovery_input_sha256": manifest["recovery_input_sha256"],
                "workers": workers,
                "target_success_count": target_success_count,
            },
            artifacts=_dynamic_record_artifacts(output, manipulation),
            summary=manipulation.summary,
        )
        catalogs, catalog_artifacts = _publish_campaign_catalogs(
            output,
            measured_records,
            manipulation_records,
            target_success_count=target_success_count,
            experiment_id=authenticated.experiment_id,
        )
        _commit_stage(
            output,
            f"catalogs_{target_success_count}",
            stage_input={
                "recovery_input_sha256": manifest["recovery_input_sha256"],
                "target_success_count": target_success_count,
            },
            artifacts=catalog_artifacts,
            summary={"catalogs": catalogs},
        )

    stop_reason = recovery_stop_reason(
        grasp_count=grasp_count,
        measured_count=measured_count,
        full_count=full_count,
    )
    report = {
        "dynamic_centroid_recovery_report_schema_version": 1,
        "complete": True,
        "recovery_input_sha256": manifest["recovery_input_sha256"],
        "target_success_count": target_success_count,
        "parent_dynamic_candidate_count": len(authenticated.dynamic_records),
        "parent_retained_trace_count": authenticated.retained_trace_count,
        "parent_tombstone_count": authenticated.tombstone_count,
        "controller_ladder_candidate_count": len(ladder.records),
        "centroid_guided_executed": len(executions) > 1,
        "new_dynamic_candidate_count": len(dynamic_records),
        "grasp_success_count": grasp_count,
        "measured_grasp_pose_count": measured_count,
        "manipulation_candidate_count": len(manipulation_records),
        "full_success_count": full_count,
        "target_reached": full_count >= target_success_count,
        "catalogs": catalogs,
        "best_attempt_report": str(best_path.relative_to(output)),
        "stop_reason": stop_reason,
    }
    report_path = output / "recovery_reports" / f"target_{target_success_count}.json"
    _write_or_authenticate(report_path, report, resume=report_path.exists())
    _commit_stage(
        output,
        f"final_report_{target_success_count}",
        stage_input={
            "recovery_input_sha256": manifest["recovery_input_sha256"],
            "target_success_count": target_success_count,
            "workers": workers,
        },
        artifacts=(report_path,),
        summary=report,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticated dynamic-centroid recovery for completed v3 evidence"
    )
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-success-count", type=int, choices=(1, 5), default=1)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_dynamic_centroid_recovery(
        arguments.parent,
        arguments.output_dir,
        workers=arguments.workers,
        resume=arguments.resume,
        target_success_count=arguments.target_success_count,
        seed=arguments.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if int(report["full_success_count"]) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
