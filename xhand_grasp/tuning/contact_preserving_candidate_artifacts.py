"""Atomic, trace-sparing candidate artifacts for the schema-v14 search.

The v14 controller search evaluates thousands of full-reset runs.  Most are
near misses whose ranking evidence is completely represented by the resolved
configuration and the semantic evaluation summary.  Writing a compressed NPZ
for every such run needlessly multiplies the campaign's disk footprint.

This module deliberately makes the retention decision *after* evaluation but
before any trace is materialized:

* a full success always retains the complete in-memory trace;
* a grasp-only success is retained by default, but high-volume refinement may
  explicitly keep only its semantic summary;
* an explicitly requested final rerun retains the complete trace even when it
  fails; and
* every other near miss commits only config + authenticated result summary.

Candidate directories are published with one same-filesystem rename.  Resume
authenticates the semantic result, exact requested config, all file hashes and
the retention contract.  The helper rejects every schema other than v14, so it
cannot alter the artifact or numerical paths of schema <= 13 experiments.
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
from ..actual_contact_capability import (
    resolve_contact_preserving_planned_lift_definition,
)
from ..artifacts import file_sha256, json_compatible, write_json
from ..config import validate_config
from ..grasp_pose import canonical_sha256


CANDIDATE_ARTIFACT_SCHEMA_VERSION = 1
TRACE_RETENTION_SCHEMA_VERSION = 1
TRACE_RETENTION_POLICY = (
    "v14_configurable_grasp_or_full_success_or_explicit_final_rerun_v2"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_DIRECTORY = re.compile(r"^candidate_(\d+)$")


class _CandidateSession(Protocol):
    """Small subset of :class:`SimulationSession` used by this module."""

    @property
    def complete(self) -> bool: ...

    def advance_one(self) -> Any: ...

    def finalize(self, *, trace_path: str | Path | None = None) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


SessionFactory = Callable[[dict[str, Any]], _CandidateSession]
ConfigValidator = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class V14CandidateArtifactBundle:
    """Authenticated paths and semantic record for one immutable candidate."""

    candidate_id: int
    destination: Path
    config_path: Path
    result_path: Path
    trace_path: Path | None
    trace_retained: bool
    trace_retention_reason: str
    result: dict[str, Any]
    reused: bool

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        values = [self.config_path, self.result_path]
        if self.trace_path is not None:
            values.append(self.trace_path)
        return tuple(values)


def _default_session_factory(config: dict[str, Any]) -> _CandidateSession:
    # Imported lazily so artifact-only resume/authentication never initializes
    # MuJoCo or allocates a simulation trace.
    from ..simulation import SimulationSession

    return SimulationSession(config)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_v14_config(config: Mapping[str, Any]) -> str:
    if int(config.get("schema_version", 0)) != 14:
        raise ValueError("trace-sparing candidate artifacts are restricted to schema v14")
    try:
        definition = resolve_contact_preserving_planned_lift_definition(
            config,
            context="trace-sparing candidate artifacts",
        )
    except ValueError as error:
        raise ValueError(
            "trace-sparing candidate artifacts require the registered v14 experiment"
        ) from error
    protocol = config.get("control_protocol")
    if not isinstance(protocol, Mapping) or protocol.get("strategy") != (
        "grasp_verify_then_contact_preserving_planned_lift"
    ):
        raise ValueError("v14 candidate has the wrong control strategy")
    return definition.experiment_id


def _stage_status(summary: Mapping[str, Any]) -> tuple[bool, bool]:
    stage = summary.get("stage_status")
    if not isinstance(stage, Mapping):
        return False, False
    grasp_success = bool(stage.get("grasp_success", False))
    full_success = bool(summary.get("passed", False)) and bool(
        stage.get("full_success", False)
    )
    if full_success and not grasp_success:
        raise RuntimeError("v14 full success cannot omit grasp success")
    return grasp_success, full_success


def _retention_decision(
    *,
    grasp_success: bool,
    full_success: bool,
    final_rerun: bool,
    retain_grasp_success: bool,
) -> tuple[bool, str]:
    if final_rerun:
        return True, "explicit_final_rerun"
    if full_success:
        return True, "full_success"
    if grasp_success and retain_grasp_success:
        return True, "grasp_success"
    if grasp_success:
        return False, "grasp_success_summary_only"
    return False, "near_miss_summary_only"


def _retention_record(
    *, retained: bool, reason: str, retain_grasp_success: bool
) -> dict[str, Any]:
    return {
        "schema_version": TRACE_RETENTION_SCHEMA_VERSION,
        "policy": TRACE_RETENTION_POLICY,
        "retained": bool(retained),
        "reason": str(reason),
        "retain_grasp_success": bool(retain_grasp_success),
        "trace_materialized": bool(retained),
        "summary_only": not bool(retained),
    }


def _classification(grasp_success: bool, full_success: bool) -> str:
    if full_success:
        return "success"
    if grasp_success:
        return "grasp_success_manipulation_near_miss"
    return "near_miss"


def _safe_staging_prefix(destination: Path) -> str:
    return f".{destination.name}.v14-staging."


def _remove_owned_staging_directories(destination: Path) -> None:
    """Remove only stale staging directories owned by this exact candidate.

    A machine power loss can leave a large retained-success NPZ in staging.
    The final candidate directory is still absent or already atomically
    complete.  Confining cleanup by resolved parent, exact prefix and symlink
    rejection avoids treating arbitrary workspace content as disposable.
    """

    parent = destination.parent.resolve()
    prefix = _safe_staging_prefix(destination)
    for value in parent.glob(f"{prefix}*"):
        candidate = value.resolve()
        if (
            value.is_symlink()
            or not value.is_dir()
            or candidate.parent != parent
            or not value.name.startswith(prefix)
        ):
            raise RuntimeError(f"unsafe v14 staging artifact encountered: {value}")
        shutil.rmtree(value)
    if parent.is_dir():
        _fsync_directory(parent)


def _allowed_candidate_files(retained: bool) -> set[str]:
    result = {"resolved_config.json", "result.json"}
    if retained:
        result.add("trace.npz")
    return result


def authenticate_v14_candidate_artifacts(
    destination: str | Path,
    *,
    expected_config: Mapping[str, Any] | None = None,
    expected_candidate_id: int | None = None,
    require_retained_trace: bool = False,
    expected_retain_grasp_success: bool | None = None,
) -> V14CandidateArtifactBundle:
    """Load one complete candidate and fail closed on any resume mismatch."""

    root = Path(destination).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    config_path = root / "resolved_config.json"
    result_path = root / "result.json"
    if not config_path.is_file() or not result_path.is_file():
        raise RuntimeError(f"incomplete v14 candidate directory: {root}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(result, dict):
        raise RuntimeError(f"v14 candidate artifacts are not JSON objects: {root}")
    experiment_id = _validate_v14_config(config)
    if expected_config is not None and canonical_sha256(config) != canonical_sha256(
        expected_config
    ):
        raise RuntimeError("resumed v14 candidate requested config changed")
    if int(result.get("contact_preserving_candidate_schema_version", 0)) != (
        CANDIDATE_ARTIFACT_SCHEMA_VERSION
    ) or result.get("complete") is not True:
        raise RuntimeError(f"v14 candidate result is incomplete or incompatible: {root}")
    identifier = int(result.get("candidate_id", -1))
    directory_match = _CANDIDATE_DIRECTORY.fullmatch(root.name)
    if (
        directory_match is not None
        and int(directory_match.group(1)) != identifier
    ):
        raise RuntimeError(
            "v14 candidate directory ID disagrees with artifact candidate ID"
        )
    if expected_candidate_id is not None and identifier != int(expected_candidate_id):
        raise RuntimeError(
            f"resumed v14 candidate {expected_candidate_id} has the wrong candidate ID"
        )
    if result.get("experiment_id") != experiment_id:
        raise RuntimeError("v14 candidate result has the wrong experiment ID")
    authenticate_candidate_result_semantic_sha256(result, source=result_path)

    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        raise RuntimeError("v14 candidate result lost its semantic summary")
    summary_sha = result.get("summary_sha256")
    if summary_sha != canonical_sha256(summary):
        raise RuntimeError("v14 candidate summary SHA-256 mismatch")
    if result.get("config_semantic_sha256") != canonical_sha256(config):
        raise RuntimeError("v14 candidate config semantic SHA-256 mismatch")
    grasp_success, full_success = _stage_status(summary)
    if bool(result.get("grasp_success", False)) != grasp_success:
        raise RuntimeError("v14 candidate grasp classification changed")
    if bool(result.get("full_success", False)) != full_success:
        raise RuntimeError("v14 candidate full-success classification changed")
    if result.get("classification") != _classification(grasp_success, full_success):
        raise RuntimeError("v14 candidate classification changed")

    retention = result.get("trace_retention")
    if not isinstance(retention, Mapping):
        raise RuntimeError("v14 candidate lost trace_retention evidence")
    retained = bool(retention.get("retained", False))
    reason = str(retention.get("reason", ""))
    retain_grasp_success = retention.get("retain_grasp_success")
    if not isinstance(retain_grasp_success, bool):
        raise RuntimeError("v14 candidate lost its grasp trace-retention policy")
    if (
        expected_retain_grasp_success is not None
        and retain_grasp_success is not bool(expected_retain_grasp_success)
    ):
        raise RuntimeError("resumed v14 candidate trace-retention policy changed")
    expected_retention = _retention_record(
        retained=retained,
        reason=reason,
        retain_grasp_success=retain_grasp_success,
    )
    if dict(retention) != expected_retention:
        raise RuntimeError("v14 candidate trace_retention contract changed")
    expected_decision = _retention_decision(
        grasp_success=grasp_success,
        full_success=full_success,
        final_rerun=reason == "explicit_final_rerun",
        retain_grasp_success=retain_grasp_success,
    )
    if (retained, reason) != expected_decision:
        raise RuntimeError("v14 candidate trace retention is inconsistent with its result")
    if require_retained_trace and not retained:
        raise RuntimeError(
            "existing v14 summary-only candidate cannot satisfy a final-rerun request"
        )

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("v14 candidate lost its artifact mapping")
    hashes = artifacts.get("sha256")
    if not isinstance(hashes, Mapping):
        raise RuntimeError("v14 candidate lost its artifact SHA-256 mapping")
    if artifacts.get("resolved_config") != config_path.name:
        raise RuntimeError("v14 candidate config artifact path changed")
    if hashes.get("resolved_config") != file_sha256(config_path):
        raise RuntimeError("v14 candidate config file SHA-256 mismatch")
    if bool(artifacts.get("trace_retained", False)) != retained:
        raise RuntimeError("v14 candidate trace retention metadata disagrees")

    trace_path = root / "trace.npz"
    if retained:
        if artifacts.get("trace") != trace_path.name or not trace_path.is_file():
            raise RuntimeError("v14 retained candidate lost trace.npz")
        trace_sha = hashes.get("trace")
        if not isinstance(trace_sha, str) or _SHA256.fullmatch(trace_sha) is None:
            raise RuntimeError("v14 retained candidate lost its trace SHA-256")
        if file_sha256(trace_path) != trace_sha:
            raise RuntimeError("v14 retained trace SHA-256 mismatch")
        if artifacts.get("trace_sha256_at_evaluation") != trace_sha:
            raise RuntimeError("v14 retained trace evaluation SHA-256 changed")
    else:
        if trace_path.exists() or artifacts.get("trace") is not None:
            raise RuntimeError("v14 summary-only candidate unexpectedly retained a trace")
        if "trace" in hashes or artifacts.get("trace_sha256_at_evaluation") is not None:
            raise RuntimeError("v14 never-materialized trace has a fabricated digest")

    observed_files = {value.name for value in root.iterdir() if value.is_file()}
    observed_directories = [value.name for value in root.iterdir() if value.is_dir()]
    if observed_files != _allowed_candidate_files(retained) or observed_directories:
        raise RuntimeError("v14 candidate contains partial or extra artifacts")
    return V14CandidateArtifactBundle(
        candidate_id=identifier,
        destination=root,
        config_path=config_path,
        result_path=result_path,
        trace_path=trace_path if retained else None,
        trace_retained=retained,
        trace_retention_reason=reason,
        result=copy.deepcopy(result),
        reused=True,
    )


def run_or_resume_v14_candidate_artifacts(
    config: Mapping[str, Any],
    destination: str | Path,
    candidate_id: int,
    *,
    final_rerun: bool = False,
    retain_grasp_success: bool = True,
    session_factory: SessionFactory | None = None,
    validator: ConfigValidator | None = validate_config,
) -> V14CandidateArtifactBundle:
    """Evaluate and atomically publish one trace-sparing v14 candidate.

    The complete trace remains in the live session until the semantic summary
    determines retention.  Consequently, an ordinary failed search candidate
    never creates a temporary NPZ, while retained evidence uses the exact same
    run that produced its summary.  ``retain_grasp_success=False`` is intended
    for high-volume controller/refinement batches; full success and explicit
    final reruns remain unconditionally retained.
    """

    if isinstance(candidate_id, bool) or int(candidate_id) < 0:
        raise ValueError("candidate_id must be a non-negative integer")
    resolved = copy.deepcopy(dict(config))
    experiment_id = _validate_v14_config(resolved)
    if validator is not None:
        validator(resolved)
    root = Path(destination).expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    _remove_owned_staging_directories(root)
    if root.exists():
        return authenticate_v14_candidate_artifacts(
            root,
            expected_config=resolved,
            expected_candidate_id=int(candidate_id),
            require_retained_trace=bool(final_rerun),
            expected_retain_grasp_success=bool(retain_grasp_success),
        )

    factory = session_factory or _default_session_factory
    staging = Path(
        tempfile.mkdtemp(
            dir=root.parent,
            prefix=_safe_staging_prefix(root),
        )
    )
    committed = False
    try:
        config_path = staging / "resolved_config.json"
        trace_path = staging / "trace.npz"
        write_json(config_path, resolved)
        session = factory(copy.deepcopy(resolved))
        try:
            while not session.complete:
                session.advance_one()
            raw_summary = session.finalize()
            if not isinstance(raw_summary, Mapping):
                raise RuntimeError("v14 candidate simulation returned no summary mapping")
            summary = json_compatible(copy.deepcopy(dict(raw_summary)))
            if not isinstance(summary, dict):  # pragma: no cover - guarded above.
                raise RuntimeError("v14 candidate summary is not JSON compatible")
            grasp_success, full_success = _stage_status(summary)
            retain, reason = _retention_decision(
                grasp_success=grasp_success,
                full_success=full_success,
                final_rerun=bool(final_rerun),
                retain_grasp_success=bool(retain_grasp_success),
            )
            if retain:
                repeated = json_compatible(
                    session.finalize(trace_path=trace_path)
                )
                if canonical_sha256(repeated) != canonical_sha256(summary):
                    raise RuntimeError("trace persistence changed the v14 evaluation summary")
                if not trace_path.is_file():
                    raise RuntimeError("retained v14 candidate did not create trace.npz")
            elif trace_path.exists():
                raise RuntimeError("summary-only v14 candidate materialized an unexpected trace")
        finally:
            session.close()

        config_sha = file_sha256(config_path)
        trace_sha = file_sha256(trace_path) if retain else None
        artifact_hashes = {"resolved_config": config_sha}
        if trace_sha is not None:
            artifact_hashes["trace"] = trace_sha
        record = {
            "contact_preserving_candidate_schema_version": (
                CANDIDATE_ARTIFACT_SCHEMA_VERSION
            ),
            "complete": True,
            "candidate_id": int(candidate_id),
            "experiment_id": experiment_id,
            "classification": _classification(grasp_success, full_success),
            "full_success": full_success,
            "grasp_success": grasp_success,
            "config_semantic_sha256": canonical_sha256(resolved),
            "summary_sha256": canonical_sha256(summary),
            "summary": summary,
            "trace_retention": _retention_record(
                retained=retain,
                reason=reason,
                retain_grasp_success=bool(retain_grasp_success),
            ),
            "artifacts": {
                "resolved_config": config_path.name,
                "trace": trace_path.name if retain else None,
                "trace_retained": retain,
                "trace_sha256_at_evaluation": trace_sha,
                "sha256": artifact_hashes,
            },
        }
        record = bind_candidate_result_semantic_sha256(record)
        result_path = staging / "result.json"
        write_json(result_path, record)
        for path in (config_path, result_path):
            _fsync_file(path)
        if retain:
            _fsync_file(trace_path)
        _fsync_directory(staging)
        staging.rename(root)
        _fsync_directory(root.parent)
        committed = True
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)

    bundle = authenticate_v14_candidate_artifacts(
        root,
        expected_config=resolved,
        expected_candidate_id=int(candidate_id),
        require_retained_trace=bool(final_rerun),
        expected_retain_grasp_success=bool(retain_grasp_success),
    )
    return V14CandidateArtifactBundle(
        candidate_id=bundle.candidate_id,
        destination=bundle.destination,
        config_path=bundle.config_path,
        result_path=bundle.result_path,
        trace_path=bundle.trace_path,
        trace_retained=bundle.trace_retained,
        trace_retention_reason=bundle.trace_retention_reason,
        result=copy.deepcopy(bundle.result),
        reused=False,
    )


__all__ = [
    "CANDIDATE_ARTIFACT_SCHEMA_VERSION",
    "TRACE_RETENTION_POLICY",
    "TRACE_RETENTION_SCHEMA_VERSION",
    "V14CandidateArtifactBundle",
    "authenticate_v14_candidate_artifacts",
    "run_or_resume_v14_candidate_artifacts",
]
