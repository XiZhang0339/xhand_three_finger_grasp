"""Atomic, trace-sparing artifacts for schema-v15 search candidates.

This is intentionally additive: the sealed schema-v14 artifact protocol and
its error/retention semantics are not imported or modified.  Search failures
retain only their authenticated semantic summary; grasp/full successes and
explicit final reruns retain the trace produced by that exact simulation.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
    bind_candidate_result_semantic_sha256,
)
from ..artifacts import file_sha256, json_compatible, write_json
from ..config import validate_config
from ..grasp_pose import canonical_sha256
from ..v15_identity import validate_v15_top_level_identities
from .joint_pair_near_zero_campaign import EXPERIMENT_ID


CANDIDATE_ARTIFACT_SCHEMA_VERSION = 1
# Keep this public name bound to the sealed policy so callers which do not opt
# into compaction produce byte-for-byte equivalent retention metadata.
TRACE_RETENTION_POLICY = "v15_grasp_or_full_success_or_explicit_final_rerun_v1"
COMPACT_TRACE_RETENTION_POLICY = (
    "v15_full_success_or_explicit_final_rerun_compact_grasp_v2"
)
_CANDIDATE_DIRECTORY = re.compile(r"^candidate_(\d+)$")


class CandidateSession(Protocol):
    @property
    def complete(self) -> bool: ...
    def advance_one(self) -> Any: ...
    def finalize(self, *, trace_path: str | Path | None = None) -> Mapping[str, Any]: ...
    def close(self) -> None: ...


SessionFactory = Callable[[dict[str, Any]], CandidateSession]


@dataclass(frozen=True, slots=True)
class V15CandidateArtifactBundle:
    candidate_id: int
    destination: Path
    config_path: Path
    result_path: Path
    trace_path: Path | None
    result: dict[str, Any]
    reused: bool

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return tuple(
            value
            for value in (self.config_path, self.result_path, self.trace_path)
            if value is not None
        )


def _validate(config: Mapping[str, Any]) -> str:
    if int(config.get("schema_version", 0)) != 15:
        raise ValueError("v15 candidate artifacts require schema_version 15")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("v15 candidate artifacts require the near-zero experiment")
    protocol = config.get("control_protocol")
    if not isinstance(protocol, Mapping) or protocol.get("strategy") != (
        "grasp_verify_then_joint_pair_aligned_contact_preserving_planned_lift"
    ):
        raise ValueError("v15 candidate has the wrong control strategy")
    validate_v15_top_level_identities(config)
    return EXPERIMENT_ID


def _status(summary: Mapping[str, Any]) -> tuple[bool, bool]:
    stage = summary.get("stage_status")
    grasp = bool(isinstance(stage, Mapping) and stage.get("grasp_success", False))
    full = bool(
        summary.get("passed", False)
        and isinstance(stage, Mapping)
        and stage.get("full_success", False)
    )
    if full and not grasp:
        raise RuntimeError("v15 full success cannot omit grasp success")
    return grasp, full


def _retention(
    grasp: bool,
    full: bool,
    final: bool,
    *,
    retain_grasp_trace: bool = True,
) -> tuple[bool, str]:
    if final:
        return True, "explicit_final_rerun"
    if full:
        return True, "full_success"
    if grasp and retain_grasp_trace:
        return True, "grasp_success"
    if grasp:
        return False, "grasp_success_summary_only"
    return False, "near_miss_summary_only"


def _retention_contract(
    retention: Mapping[str, Any],
) -> tuple[bool, bool]:
    """Return ``(retain_grasp_trace, final_rerun)`` for a strict contract.

    Version one is the already-sealed campaign format.  Version two is an
    explicit opt-in compact format: it never silently changes the meaning of
    an existing directory and carries enough information to recompute the
    retention decision without guessing from the reason string.
    """

    schema_version = int(retention.get("schema_version", 0))
    policy = retention.get("policy")
    if schema_version == 1 and policy == TRACE_RETENTION_POLICY:
        if set(retention) != {"schema_version", "policy", "retained", "reason"}:
            raise RuntimeError("v15 v1 retention contract has unexpected fields")
        return True, retention.get("reason") == "explicit_final_rerun"
    if schema_version == 2 and policy == COMPACT_TRACE_RETENTION_POLICY:
        if set(retention) != {
            "schema_version",
            "policy",
            "retain_grasp_trace",
            "final_rerun",
            "retained",
            "reason",
        }:
            raise RuntimeError("v15 compact retention contract has unexpected fields")
        if retention.get("retain_grasp_trace") is not False:
            raise RuntimeError("v15 compact retention contract changed")
        final_rerun = retention.get("final_rerun")
        if not isinstance(final_rerun, bool):
            raise RuntimeError("v15 compact retention final-rerun flag changed")
        return False, final_rerun
    raise RuntimeError("v15 candidate retention contract changed")


def _default_factory(config: dict[str, Any]) -> CandidateSession:
    from ..simulation import SimulationSession
    return SimulationSession(config)


def _fsync(path: Path, *, directory: bool = False) -> None:
    flags = os.O_RDONLY | (getattr(os, "O_DIRECTORY", 0) if directory else 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _staging_prefix(destination: Path) -> str:
    return f".{destination.name}.v15-staging."


def _remove_stale_staging(destination: Path) -> None:
    parent = destination.parent.resolve()
    prefix = _staging_prefix(destination)
    for raw in parent.glob(f"{prefix}*"):
        path = raw.resolve()
        if raw.is_symlink() or not raw.is_dir() or path.parent != parent:
            raise RuntimeError(f"unsafe v15 staging artifact: {raw}")
        shutil.rmtree(raw)


def authenticate_v15_candidate_artifacts(
    destination: str | Path,
    *,
    expected_config: Mapping[str, Any] | None = None,
    expected_candidate_id: int | None = None,
    require_retained_trace: bool = False,
    expected_retain_grasp_trace: bool | None = None,
) -> V15CandidateArtifactBundle:
    if expected_retain_grasp_trace is not None and not isinstance(
        expected_retain_grasp_trace, bool
    ):
        raise TypeError("expected_retain_grasp_trace must be a boolean or None")
    root = Path(destination).expanduser().resolve()
    config_path, result_path, trace_path = (
        root / "resolved_config.json",
        root / "result.json",
        root / "trace.npz",
    )
    if not config_path.is_file() or not result_path.is_file():
        raise RuntimeError(f"incomplete v15 candidate directory: {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _validate(config)
    if expected_config is not None and canonical_sha256(config) != canonical_sha256(expected_config):
        raise RuntimeError("resumed v15 candidate config changed")
    if result.get("complete") is not True or int(result.get("joint_pair_near_zero_candidate_schema_version", 0)) != 1:
        raise RuntimeError("v15 candidate result is incomplete")
    candidate_id = int(result.get("candidate_id", -1))
    match = _CANDIDATE_DIRECTORY.fullmatch(root.name)
    if match and int(match.group(1)) != candidate_id:
        raise RuntimeError("v15 candidate directory ID changed")
    if expected_candidate_id is not None and candidate_id != int(expected_candidate_id):
        raise RuntimeError("resumed v15 candidate ID changed")
    authenticate_candidate_result_semantic_sha256(result, source=result_path)
    summary = result.get("summary")
    if not isinstance(summary, Mapping) or result.get("summary_sha256") != canonical_sha256(summary):
        raise RuntimeError("v15 candidate summary digest mismatch")
    if result.get("config_semantic_sha256") != canonical_sha256(config):
        raise RuntimeError("v15 candidate config digest mismatch")
    grasp, full = _status(summary)
    if bool(result.get("grasp_success")) != grasp or bool(result.get("full_success")) != full:
        raise RuntimeError("v15 candidate classification changed")
    retention = result.get("trace_retention")
    if not isinstance(retention, Mapping):
        raise RuntimeError("v15 candidate retention contract changed")
    retain_grasp_trace, final_rerun = _retention_contract(retention)
    if (
        expected_retain_grasp_trace is not None
        and retain_grasp_trace != expected_retain_grasp_trace
    ):
        raise RuntimeError("resumed v15 candidate trace-retention policy changed")
    retained_value = retention.get("retained")
    if not isinstance(retained_value, bool):
        raise RuntimeError("v15 candidate retained flag changed")
    retained = retained_value
    expected = _retention(
        grasp,
        full,
        final_rerun,
        retain_grasp_trace=retain_grasp_trace,
    )
    if (retained, retention.get("reason")) != expected:
        raise RuntimeError("v15 candidate retention decision is inconsistent")
    if require_retained_trace and not retained:
        raise RuntimeError("summary-only candidate cannot satisfy final rerun")
    artifacts = result.get("artifacts")
    hashes = artifacts.get("sha256") if isinstance(artifacts, Mapping) else None
    if not isinstance(hashes, Mapping) or hashes.get("resolved_config") != file_sha256(config_path):
        raise RuntimeError("v15 candidate config file digest mismatch")
    if retained:
        if not trace_path.is_file() or hashes.get("trace") != file_sha256(trace_path):
            raise RuntimeError("v15 retained trace digest mismatch")
    elif trace_path.exists():
        raise RuntimeError("v15 summary-only candidate unexpectedly has a trace")
    allowed = {"resolved_config.json", "result.json"} | ({"trace.npz"} if retained else set())
    if {p.name for p in root.iterdir()} != allowed:
        raise RuntimeError("v15 candidate directory contains extra artifacts")
    return V15CandidateArtifactBundle(
        candidate_id, root, config_path, result_path, trace_path if retained else None,
        copy.deepcopy(result), True,
    )


def run_or_resume_v15_candidate_artifacts(
    config: Mapping[str, Any],
    destination: str | Path,
    candidate_id: int,
    *,
    final_rerun: bool = False,
    retain_grasp_trace: bool = True,
    session_factory: SessionFactory | None = None,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> V15CandidateArtifactBundle:
    if isinstance(candidate_id, bool) or int(candidate_id) < 0:
        raise ValueError("candidate_id must be a non-negative integer")
    if not isinstance(retain_grasp_trace, bool):
        raise TypeError("retain_grasp_trace must be a boolean")
    resolved = copy.deepcopy(dict(config))
    experiment_id = _validate(resolved)
    if validator is not None:
        validator(resolved)
    root = Path(destination).expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    _remove_stale_staging(root)
    if root.exists():
        return authenticate_v15_candidate_artifacts(
            root, expected_config=resolved, expected_candidate_id=candidate_id,
            require_retained_trace=final_rerun,
            expected_retain_grasp_trace=retain_grasp_trace,
        )
    staging = Path(tempfile.mkdtemp(dir=root.parent, prefix=_staging_prefix(root)))
    committed = False
    try:
        config_path, trace_path = staging / "resolved_config.json", staging / "trace.npz"
        write_json(config_path, resolved)
        session = (session_factory or _default_factory)(copy.deepcopy(resolved))
        try:
            while not session.complete:
                session.advance_one()
            summary = json_compatible(copy.deepcopy(dict(session.finalize())))
            grasp, full = _status(summary)
            retained, reason = _retention(
                grasp,
                full,
                bool(final_rerun),
                retain_grasp_trace=retain_grasp_trace,
            )
            if retained:
                repeated = json_compatible(session.finalize(trace_path=trace_path))
                if canonical_sha256(repeated) != canonical_sha256(summary):
                    raise RuntimeError("trace persistence changed v15 summary")
        finally:
            session.close()
        hashes = {"resolved_config": file_sha256(config_path)}
        if retained:
            hashes["trace"] = file_sha256(trace_path)
        retention_record = (
            {
                "schema_version": 1,
                "policy": TRACE_RETENTION_POLICY,
                "retained": retained,
                "reason": reason,
            }
            if retain_grasp_trace
            else {
                "schema_version": 2,
                "policy": COMPACT_TRACE_RETENTION_POLICY,
                "retain_grasp_trace": False,
                "final_rerun": bool(final_rerun),
                "retained": retained,
                "reason": reason,
            }
        )
        record = bind_candidate_result_semantic_sha256({
            "joint_pair_near_zero_candidate_schema_version": 1,
            "complete": True,
            "candidate_id": int(candidate_id),
            "experiment_id": experiment_id,
            "classification": "success" if full else "grasp_success_manipulation_near_miss" if grasp else "near_miss",
            "grasp_success": grasp,
            "full_success": full,
            "config_semantic_sha256": canonical_sha256(resolved),
            "summary_sha256": canonical_sha256(summary),
            "summary": summary,
            "trace_retention": retention_record,
            "artifacts": {"resolved_config": config_path.name, "trace": trace_path.name if retained else None, "trace_retained": retained, "sha256": hashes},
        })
        result_path = staging / "result.json"
        write_json(result_path, record)
        _fsync(config_path); _fsync(result_path)
        if retained: _fsync(trace_path)
        _fsync(staging, directory=True)
        staging.rename(root); _fsync(root.parent, directory=True)
        committed = True
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)
    bundle = authenticate_v15_candidate_artifacts(
        root, expected_config=resolved, expected_candidate_id=candidate_id,
        require_retained_trace=final_rerun,
        expected_retain_grasp_trace=retain_grasp_trace,
    )
    return V15CandidateArtifactBundle(
        bundle.candidate_id, bundle.destination, bundle.config_path,
        bundle.result_path, bundle.trace_path, bundle.result, False,
    )


__all__ = [
    "COMPACT_TRACE_RETENTION_POLICY", "TRACE_RETENTION_POLICY",
    "V15CandidateArtifactBundle",
    "authenticate_v15_candidate_artifacts", "run_or_resume_v15_candidate_artifacts",
]
