"""Authenticated, deterministic refinement rescue for schema-v14 near misses.

The formal v14 campaign is immutable input.  This module has no output-side
filesystem operations: it authenticates a completed formal campaign read-only,
selects only candidates that completed manipulation while preserving all three
contacts, and materializes a balanced 1,024-job local refinement schedule.
The campaign runner owns atomic persistence and resume.

The rescue differs intentionally from the original joint refinement:

* parent eligibility is fail-closed and requires completed, non-aborted
  operation with the registered 99% / 10 ms contact guarantees;
* semantically identical resolved configurations are de-duplicated;
* every parent is evaluated exactly once before perturbation;
* perturbations cover eight independent parameter blocks and five radii using
  deterministic, group-local Latin hypercubes; and
* a zero integral gain is an invariant, not a value that sampling may awaken.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..actual_contact_grasp_pose_catalog import validate_stage_ledger
from ..artifacts import file_sha256
from ..config import (
    ACTIVE_ACTUATORS,
    contact_preload_targets,
    precontact_targets,
    validate_config,
)
from ..experiment import ContactFeedbackParameters, ManipulationPlanParameters
from ..experiments.opposed_face_palm_down_contact_preserving_planned_lift import (
    EXPERIMENT_ID,
)
from ..grasp_pose import canonical_sha256
from ..trajectory import minimum_jerk, quintic_c2_knot_derivatives
from .contact_preserving_candidate_artifacts import (
    V14CandidateArtifactBundle,
    authenticate_v14_candidate_artifacts,
)
from .contact_preserving_joint_refinement import (
    JointRefinementLimits,
    resolve_joint_refinement_limits,
)


FORMAL_V2_SOURCE_SCHEMA_VERSION = 1
REFINEMENT_RESCUE_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
DEFAULT_TOTAL_CANDIDATES = 1024
MAXIMUM_PARENT_COUNT = 8
RADIUS_FRACTIONS = (0.20, 0.40, 0.60, 0.80, 1.00)
PARAMETER_BLOCKS = (
    "thumb_plan",
    "index_plan",
    "middle_plan",
    "thumb_preload",
    "index_preload",
    "middle_preload",
    "feedback_kp",
    "feedback_ki",
)
_FINGERS = ("thumb", "index", "mid")
_FINGER_ACTUATORS = {
    "thumb": ACTIVE_ACTUATORS[0:3],
    "index": ACTIVE_ACTUATORS[3:6],
    "middle": ACTIVE_ACTUATORS[6:8],
}
_REPORT_SPECS = (
    (
        "candidate_search",
        Path("candidate_search/target_1.json"),
        "contact_preserving_candidate_report_schema_version",
        64,
        frozenset(
            {
                "artifact_directory",
                "feedback_id",
                "feedback_index",
                "plan_candidate_id",
                "plan_rank",
            }
        ),
    ),
    (
        "joint_refinement",
        Path("joint_refinement/target_1.json"),
        "v14_joint_refinement_report_schema_version",
        1024,
        frozenset(
            {
                "artifact_directory",
                "job_sequence_index",
                "joint_refinement",
                "local_index",
                "parent_candidate_id",
                "parent_rank",
            }
        ),
    ),
)
_REQUIRED_LEDGER_STAGES = frozenset(
    {
        "source_audit",
        "grasp_pose_rescue",
        "grasp_object_pairs",
        "offline_planning",
        "candidate_search_target_1",
        "joint_refinement_target_1",
        "catalog_target_1",
    }
)
_PARENT_CHECKS = (
    "operation_executed",
    "manipulation_completed",
    "v14_operation_did_not_abort",
    "v14_plan_progress_reached_one",
    "v14_thumb_contact_duty_at_least_99_percent",
    "v14_index_contact_duty_at_least_99_percent",
    "v14_middle_contact_duty_at_least_99_percent",
    "v14_simultaneous_contact_duty_at_least_99_percent",
    "v14_thumb_contact_loss_within_limit",
    "v14_index_contact_loss_within_limit",
    "v14_middle_contact_loss_within_limit",
    "v14_simultaneous_contact_loss_within_limit",
    "no_palm_ring_or_pinky_contact",
    "active_nondistal_contacts_within_limit",
)
_EPSILON = 1e-12


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(value: Any, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result >= 0 else default


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def _confined(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise RuntimeError(f"formal v2 {label} is not a safe relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise RuntimeError(f"formal v2 {label} escaped or is missing")
    return path


@dataclass(frozen=True, slots=True)
class AuthenticatedRescueCandidate:
    """One report record whose immutable candidate bundle was authenticated."""

    stage: str
    candidate_id: int
    artifact_directory: str
    config_semantic_sha256: str
    record: Mapping[str, Any]
    bundle: V14CandidateArtifactBundle

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", MappingProxyType(copy.deepcopy(dict(self.record))))

    @property
    def config_path(self) -> Path:
        return self.bundle.config_path

    @property
    def result_path(self) -> Path:
        return self.bundle.result_path

    def load_config(self) -> dict[str, Any]:
        config = _load_json_object(self.config_path, "formal v2 candidate config")
        if canonical_sha256(config) != self.config_semantic_sha256:
            raise RuntimeError("formal v2 candidate config semantic hash changed")
        return config

    def descriptor(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "candidate_id": self.candidate_id,
            "artifact_directory": self.artifact_directory,
            "config_semantic_sha256": self.config_semantic_sha256,
            "result_semantic_sha256": self.bundle.result[
                "result_semantic_sha256"
            ],
            "config_file_sha256": file_sha256(self.config_path),
            "result_file_sha256": file_sha256(self.result_path),
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedFormalV2RescueSource:
    """Read-only authenticated formal-v2 campaign input."""

    root: Path
    manifest_path: Path
    ledger_path: Path
    candidate_report_path: Path
    refinement_report_path: Path
    campaign_result_path: Path
    manifest_sha256: str
    ledger_sha256: str
    candidate_report_sha256: str
    refinement_report_sha256: str
    campaign_result_sha256: str
    source_authentication_id: str
    records: tuple[AuthenticatedRescueCandidate, ...]

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        """Small immutable source surface; the ledger binds every candidate."""

        return (
            self.manifest_path,
            self.ledger_path,
            self.candidate_report_path,
            self.refinement_report_path,
            self.campaign_result_path,
        )

    def descriptor(self) -> dict[str, Any]:
        by_stage: dict[str, int] = {}
        for record in self.records:
            by_stage[record.stage] = by_stage.get(record.stage, 0) + 1
        return {
            "formal_v2_source_schema_version": FORMAL_V2_SOURCE_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "source_root": str(self.root),
            "source_authentication_id": self.source_authentication_id,
            "manifest_sha256": self.manifest_sha256,
            "ledger_sha256": self.ledger_sha256,
            "candidate_report_sha256": self.candidate_report_sha256,
            "refinement_report_sha256": self.refinement_report_sha256,
            "campaign_result_sha256": self.campaign_result_sha256,
            "candidate_count": len(self.records),
            "candidate_count_by_stage": dict(sorted(by_stage.items())),
            "candidate_semantic_set_sha256": canonical_sha256(
                [record.descriptor() for record in self.records]
            ),
            "read_only": True,
        }


@dataclass(frozen=True, slots=True)
class RefinementRescueParent:
    rank: int
    candidate_id: int
    config_semantic_sha256: str
    source_stage: str
    source_artifact_directory: str
    source_authentication_id: str
    config: Mapping[str, Any]
    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", MappingProxyType(copy.deepcopy(dict(self.config))))
        object.__setattr__(self, "record", MappingProxyType(copy.deepcopy(dict(self.record))))

    def descriptor(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "config_semantic_sha256": self.config_semantic_sha256,
            "source_stage": self.source_stage,
            "source_artifact_directory": self.source_artifact_directory,
            "source_authentication_id": self.source_authentication_id,
        }


@dataclass(frozen=True, slots=True)
class RefinementRescueBudget:
    total_candidates: int = DEFAULT_TOTAL_CANDIDATES
    maximum_parent_count: int = MAXIMUM_PARENT_COUNT
    seed: int = DEFAULT_SEED
    radius_fractions: tuple[float, ...] = RADIUS_FRACTIONS
    parameter_blocks: tuple[str, ...] = PARAMETER_BLOCKS
    maximum_plan_radius_fraction: float = 0.10
    maximum_preload_radius_rad: float = 0.02
    maximum_feedback_fraction: float = 0.75

    def __post_init__(self) -> None:
        for name in ("total_candidates", "maximum_parent_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.total_candidates != DEFAULT_TOTAL_CANDIDATES:
            raise ValueError("formal v14 refinement rescue budget must be exactly 1024")
        if not 1 <= self.maximum_parent_count <= MAXIMUM_PARENT_COUNT:
            raise ValueError("maximum_parent_count must lie within [1, 8]")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        radii = tuple(float(value) for value in self.radius_fractions)
        if radii != RADIUS_FRACTIONS:
            raise ValueError("rescue requires the registered five radii")
        if tuple(self.parameter_blocks) != PARAMETER_BLOCKS:
            raise ValueError("rescue requires the registered eight parameter blocks")
        for name in (
            "maximum_plan_radius_fraction",
            "maximum_preload_radius_rad",
            "maximum_feedback_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "radius_fractions", radii)
        object.__setattr__(self, "parameter_blocks", tuple(self.parameter_blocks))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": REFINEMENT_RESCUE_SCHEMA_VERSION,
            "total_candidates": self.total_candidates,
            "maximum_parent_count": self.maximum_parent_count,
            "seed": self.seed,
            "radius_fractions": list(self.radius_fractions),
            "parameter_blocks": list(self.parameter_blocks),
            "maximum_plan_radius_fraction": self.maximum_plan_radius_fraction,
            "maximum_preload_radius_rad": self.maximum_preload_radius_rad,
            "maximum_feedback_fraction": self.maximum_feedback_fraction,
        }


def _authenticate_wrapper(
    persisted: Mapping[str, Any],
    wrapper: Mapping[str, Any],
    allowed_metadata: frozenset[str],
) -> None:
    persisted_keys = frozenset(persisted)
    wrapper_keys = frozenset(wrapper)
    missing = sorted(persisted_keys - wrapper_keys)
    unknown = sorted(wrapper_keys - persisted_keys - allowed_metadata)
    if missing:
        raise RuntimeError("formal v2 report dropped persisted fields: " + ", ".join(missing))
    if unknown:
        raise RuntimeError("formal v2 report added unknown fields: " + ", ".join(unknown))
    projection = {key: wrapper[key] for key in persisted_keys}
    if canonical_sha256(projection) != canonical_sha256(persisted):
        raise RuntimeError("formal v2 report changed persisted candidate evidence")


def _authenticate_bound_manifest_files(root: Path, manifest: Mapping[str, Any]) -> None:
    fields = (
        ("config_path", "config_sha256"),
        ("model_path", "model_sha256"),
        ("uv_lock_path", "uv_lock_sha256"),
    )
    for path_field, hash_field in fields:
        raw = manifest.get(path_field)
        expected = manifest.get(hash_field)
        if not isinstance(raw, str) or not _sha256(expected):
            raise RuntimeError(f"formal v2 manifest lost {path_field}/{hash_field}")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"formal v2 manifest-bound file changed: {path_field}")
    raw_source = manifest.get("actual_qpos_source_manifest_path")
    expected_source = manifest.get("actual_qpos_source_manifest_sha256")
    if not isinstance(raw_source, str) or not _sha256(expected_source):
        raise RuntimeError("formal v2 manifest lost actual-qpos source binding")
    source_path = Path(raw_source).expanduser()
    if not source_path.is_absolute():
        # These manifests use repository-relative paths.  Walking upward from
        # the formal campaign until the bound relative path exists avoids
        # coupling this read-only verifier to a mutable process cwd.
        candidates = (root, *root.parents)
        source_path = next(
            ((parent / source_path).resolve() for parent in candidates if (parent / source_path).is_file()),
            (root / source_path).resolve(),
        )
    else:
        source_path = source_path.resolve()
    if not source_path.is_file() or file_sha256(source_path) != expected_source:
        raise RuntimeError("formal v2 actual-qpos source manifest changed")


def authenticate_formal_v2_rescue_source(
    source_campaign: str | Path,
) -> AuthenticatedFormalV2RescueSource:
    """Authenticate a completed formal-v2 campaign without modifying it."""

    root = Path(source_campaign).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = root / "campaign_manifest.json"
    ledger_path = root / "stage_ledger.json"
    result_path = root / "campaign_result_target_1.json"
    manifest = _load_json_object(manifest_path, "formal v2 manifest")
    if (
        manifest.get("campaign_manifest_schema_version") != 1
        or manifest.get("contact_preserving_campaign_manifest_schema_version") != 1
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or not _sha256(manifest.get("campaign_input_sha256"))
    ):
        raise RuntimeError("formal v2 manifest is incompatible")
    _authenticate_bound_manifest_files(root, manifest)
    ledger = validate_stage_ledger(root)
    stages = _mapping(ledger.get("stages"))
    missing_stages = sorted(_REQUIRED_LEDGER_STAGES - set(stages))
    if missing_stages:
        raise RuntimeError("formal v2 ledger is incomplete: " + ", ".join(missing_stages))
    result = _load_json_object(result_path, "formal v2 campaign result")
    if (
        result.get("complete") is not True
        or result.get("experiment_id") != EXPERIMENT_ID
        or int(result.get("base_feedback_candidate_count", -1)) != 64
        or int(result.get("joint_refinement_candidate_count", -1)) != 1024
        or int(result.get("full_success_count", -1)) != 0
        or result.get("stop_reason") != "declared_contact_first_search_exhausted"
    ):
        raise RuntimeError("formal v2 campaign result is incomplete or incompatible")

    authenticated: list[AuthenticatedRescueCandidate] = []
    seen_ids: set[int] = set()
    report_paths: dict[str, Path] = {}
    for stage, relative, schema_field, expected_count, allowed_metadata in _REPORT_SPECS:
        report_path = (root / relative).resolve()
        report_paths[stage] = report_path
        report = _load_json_object(report_path, f"formal v2 {stage} report")
        records = report.get("records")
        if (
            report.get("complete") is not True
            or int(report.get(schema_field, 0)) != 1
            or not isinstance(records, list)
            or int(report.get("candidate_count", -1)) != len(records)
            or len(records) != expected_count
        ):
            raise RuntimeError(f"formal v2 {stage} report is incomplete")
        for raw in records:
            if not isinstance(raw, Mapping):
                raise RuntimeError(f"formal v2 {stage} contains a non-object record")
            candidate_id = int(raw.get("candidate_id", -1))
            if candidate_id < 0 or candidate_id in seen_ids:
                raise RuntimeError("formal v2 candidate IDs are invalid or duplicated")
            seen_ids.add(candidate_id)
            directory = _confined(root, raw.get("artifact_directory"), "candidate artifact")
            bundle = authenticate_v14_candidate_artifacts(
                directory,
                expected_candidate_id=candidate_id,
            )
            _authenticate_wrapper(bundle.result, raw, allowed_metadata)
            config_hash = str(raw.get("config_semantic_sha256", ""))
            if config_hash != bundle.result.get("config_semantic_sha256") or not _sha256(config_hash):
                raise RuntimeError("formal v2 candidate config identity changed")
            authenticated.append(
                AuthenticatedRescueCandidate(
                    stage=stage,
                    candidate_id=candidate_id,
                    artifact_directory=str(directory.relative_to(root)),
                    config_semantic_sha256=config_hash,
                    record=raw,
                    bundle=bundle,
                )
            )
    authenticated.sort(key=lambda value: (value.stage, value.candidate_id))
    candidate_report = report_paths["candidate_search"]
    refinement_report = report_paths["joint_refinement"]
    hashes = {
        "manifest": file_sha256(manifest_path),
        "ledger": file_sha256(ledger_path),
        "candidate_report": file_sha256(candidate_report),
        "refinement_report": file_sha256(refinement_report),
        "campaign_result": file_sha256(result_path),
    }
    source_id = canonical_sha256(
        {
            "schema_version": FORMAL_V2_SOURCE_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "hashes": hashes,
            "candidate_count": len(authenticated),
            "candidate_bindings": [
                {
                    "candidate_id": value.candidate_id,
                    "config_semantic_sha256": value.config_semantic_sha256,
                    "result_semantic_sha256": value.bundle.result[
                        "result_semantic_sha256"
                    ],
                }
                for value in authenticated
            ],
        }
    )
    return AuthenticatedFormalV2RescueSource(
        root=root,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        candidate_report_path=candidate_report,
        refinement_report_path=refinement_report,
        campaign_result_path=result_path,
        manifest_sha256=hashes["manifest"],
        ledger_sha256=hashes["ledger"],
        candidate_report_sha256=hashes["candidate_report"],
        refinement_report_sha256=hashes["refinement_report"],
        campaign_result_sha256=hashes["campaign_result"],
        source_authentication_id=source_id,
        records=tuple(authenticated),
    )


def refinement_rescue_parent_eligible(record: Mapping[str, Any]) -> bool:
    """Return true only for complete contact-preserving operation evidence."""

    summary = _mapping(record.get("summary"))
    stage = _mapping(summary.get("stage_status"))
    checks = _mapping(summary.get("checks"))
    metrics = _mapping(summary.get("metrics"))
    planned = _mapping(metrics.get("contact_preserving_planned_lift"))
    duties = _mapping(planned.get("target_face_effective_duty"))
    losses = _mapping(planned.get("longest_contact_loss_steps"))
    required_duty = _finite(planned.get("required_contact_duty"), math.inf)
    allowed_loss = _integer(planned.get("allowed_contact_loss_steps"), -1)
    progress = _finite(planned.get("final_plan_progress"), -math.inf)
    maximum_progress = _finite(planned.get("maximum_plan_progress"), -math.inf)
    forbidden_steps = _integer(metrics.get("forbidden_contact_steps"), 10**9)
    nondistal_duty = _finite(metrics.get("material_active_nondistal_duty"), math.inf)
    return bool(
        stage.get("grasp_success") is True
        and all(checks.get(name) is True for name in _PARENT_CHECKS)
        and planned.get("operation_aborted") is False
        and metrics.get("controller_aborted") is False
        and progress >= 1.0 - _EPSILON
        and maximum_progress >= 1.0 - _EPSILON
        and 0.0 < required_duty <= 1.0
        and allowed_loss >= 0
        and all(
            _finite(duties.get(finger), -math.inf) >= required_duty - _EPSILON
            for finger in _FINGERS
        )
        and _finite(
            planned.get("simultaneous_target_face_effective_duty"), -math.inf
        )
        >= required_duty - _EPSILON
        and all(
            _integer(losses.get(finger), 10**9) <= allowed_loss
            for finger in _FINGERS
        )
        and _integer(
            planned.get("simultaneous_longest_contact_loss_steps"), 10**9
        )
        <= allowed_loss
        and forbidden_steps == 0
        and nondistal_duty <= 0.01 + _EPSILON
    )


def _smoothness_evidence(metrics: Mapping[str, Any]) -> tuple[dict[str, float], bool]:
    smooth = _mapping(metrics.get("motion_smoothness"))
    raw = {
        "median_lift_m": _finite(metrics.get("operation_median_lift_m"), math.nan),
        "minimum_lift_m": _finite(metrics.get("operation_minimum_lift_m"), math.nan),
        "lateral_m": _finite(smooth.get("operation_max_lateral_displacement_m"), math.nan),
        "orientation_deg": _finite(smooth.get("operation_max_orientation_drift_deg"), math.nan),
        "backtrack_m": _finite(smooth.get("operation_cumulative_height_backtrack_m"), math.nan),
        "upward_speed_m_s": _finite(smooth.get("operation_peak_filtered_upward_speed_m_s"), math.nan),
        "acceleration_m_s2": _finite(smooth.get("operation_peak_abs_filtered_acceleration_m_s2"), math.nan),
        "jerk_m_s3": _finite(smooth.get("operation_peak_abs_filtered_jerk_m_s3"), math.nan),
        "hold_entry_speed_m_s": _finite(smooth.get("operation_hold_entry_linear_speed_m_s"), math.nan),
    }
    return raw, all(math.isfinite(value) for value in raw.values())


def refinement_rescue_rank_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract complete rescue metrics; missing evidence ranks fail-closed."""

    summary = _mapping(record.get("summary", record))
    stage = _mapping(summary.get("stage_status"))
    metrics = _mapping(summary.get("metrics"))
    planned = _mapping(metrics.get("contact_preserving_planned_lift"))
    duties = _mapping(planned.get("target_face_effective_duty"))
    losses = _mapping(planned.get("longest_contact_loss_steps"))
    contact_duties = [
        _finite(duties.get(finger), -math.inf) for finger in _FINGERS
    ] + [
        _finite(planned.get("simultaneous_target_face_effective_duty"), -math.inf)
    ]
    contact_losses = [
        _integer(losses.get(finger), 10**9) for finger in _FINGERS
    ] + [
        _integer(planned.get("simultaneous_longest_contact_loss_steps"), 10**9)
    ]
    smooth, smooth_complete = _smoothness_evidence(metrics)
    margins_without_jerk = (
        smooth["median_lift_m"] / 0.010 - 1.0,
        smooth["minimum_lift_m"] / 0.008 - 1.0,
        1.0 - smooth["lateral_m"] / 0.002,
        1.0 - smooth["orientation_deg"] / 10.0,
        1.0 - smooth["backtrack_m"] / 0.0002,
        1.0 - smooth["upward_speed_m_s"] / 0.020,
        1.0 - smooth["acceleration_m_s2"] / 0.12,
        1.0 - smooth["hold_entry_speed_m_s"] / 0.005,
    ) if smooth_complete else (-math.inf,)
    failed_checks = summary.get("failed_checks")
    if not isinstance(failed_checks, list) or any(
        not isinstance(value, str) for value in failed_checks
    ):
        non_jerk_failure_count = 10**9
    else:
        non_jerk_failure_count = sum(
            value != "smooth_motion_jerk_within_limit"
            for value in failed_checks
        )
    identifier = record.get("candidate_id", "")
    stable_id = (0, int(identifier)) if isinstance(identifier, int) and not isinstance(identifier, bool) else (1, str(identifier))
    contact_complete = bool(
        all(math.isfinite(value) for value in contact_duties)
        and all(value < 10**9 for value in contact_losses)
    )
    return {
        "full_success": bool(summary.get("passed") is True and stage.get("full_success") is True),
        "parent_eligible": refinement_rescue_parent_eligible(record),
        "contact_metrics_complete": contact_complete,
        "minimum_contact_duty": min(contact_duties, default=-math.inf),
        "maximum_contact_loss_steps": max(contact_losses, default=10**9),
        "overall_metrics_complete": smooth_complete,
        "non_jerk_failure_count": non_jerk_failure_count,
        "overall_minimum_normalized_margin": min(margins_without_jerk),
        "peak_jerk_m_s3": smooth["jerk_m_s3"] if smooth_complete else math.inf,
        "median_lift_m": smooth["median_lift_m"] if smooth_complete else -math.inf,
        "minimum_lift_m": smooth["minimum_lift_m"] if smooth_complete else -math.inf,
        "lateral_m": smooth["lateral_m"] if smooth_complete else math.inf,
        "orientation_deg": smooth["orientation_deg"] if smooth_complete else math.inf,
        "actuator_saturation_fraction": _finite(metrics.get("actuator_saturation_fraction"), math.inf),
        "stable_candidate_id": stable_id,
    }


def refinement_rescue_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    evidence = refinement_rescue_rank_evidence(record)
    return (
        not evidence["full_success"],
        not evidence["parent_eligible"],
        not evidence["contact_metrics_complete"],
        -evidence["minimum_contact_duty"],
        evidence["maximum_contact_loss_steps"],
        not evidence["overall_metrics_complete"],
        evidence["non_jerk_failure_count"],
        -evidence["overall_minimum_normalized_margin"],
        evidence["peak_jerk_m_s3"],
        -evidence["median_lift_m"],
        -evidence["minimum_lift_m"],
        evidence["lateral_m"],
        evidence["orientation_deg"],
        evidence["actuator_saturation_fraction"],
        evidence["stable_candidate_id"],
    )


def rank_refinement_rescue_results(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    materialized = [copy.deepcopy(dict(value)) for value in records]
    materialized.sort(key=refinement_rescue_candidate_rank)
    return tuple(materialized)


def select_refinement_rescue_parents(
    source: AuthenticatedFormalV2RescueSource,
    *,
    maximum_parent_count: int = MAXIMUM_PARENT_COUNT,
) -> tuple[RefinementRescueParent, ...]:
    if not 1 <= int(maximum_parent_count) <= MAXIMUM_PARENT_COUNT:
        raise ValueError("maximum_parent_count must lie within [1, 8]")
    eligible = [value for value in source.records if refinement_rescue_parent_eligible(value.record)]
    eligible.sort(key=lambda value: refinement_rescue_candidate_rank(value.record))
    selected: list[RefinementRescueParent] = []
    seen_configs: set[str] = set()
    for candidate in eligible:
        if candidate.config_semantic_sha256 in seen_configs:
            continue
        seen_configs.add(candidate.config_semantic_sha256)
        config = candidate.load_config()
        selected.append(
            RefinementRescueParent(
                rank=len(selected),
                candidate_id=candidate.candidate_id,
                config_semantic_sha256=candidate.config_semantic_sha256,
                source_stage=candidate.stage,
                source_artifact_directory=candidate.artifact_directory,
                source_authentication_id=source.source_authentication_id,
                config=config,
                record=candidate.record,
            )
        )
        if len(selected) == int(maximum_parent_count):
            break
    if not selected:
        raise RuntimeError("formal v2 contains no eligible refinement-rescue parent")
    return tuple(selected)


def allocate_refinement_rescue_budget(parent_count: int, total: int = 1024) -> tuple[int, ...]:
    if not isinstance(parent_count, int) or isinstance(parent_count, bool) or not 1 <= parent_count <= 8:
        raise ValueError("parent_count must lie within [1, 8]")
    if total != DEFAULT_TOTAL_CANDIDATES:
        raise ValueError("refinement rescue total must be exactly 1024")
    quotient, remainder = divmod(total, parent_count)
    allocation = tuple(quotient + int(index < remainder) for index in range(parent_count))
    if sum(allocation) != total or min(allocation) <= 0:
        raise AssertionError("invalid rescue budget allocation")
    return allocation


def _lhs(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=np.float64)
    for column in range(dimensions):
        permutation = rng.permutation(count)
        result[:, column] = (permutation + rng.random(count)) / count
    return 2.0 * result - 1.0


def _group_seed(seed: int, parent_hash: str, block: str, radius_index: int) -> int:
    digest = canonical_sha256(
        {
            "seed": seed,
            "parent_config_semantic_sha256": parent_hash,
            "parameter_block": block,
            "radius_index": radius_index,
        }
    )
    words = np.frombuffer(bytes.fromhex(digest[:32]), dtype="<u4")
    return int(np.random.SeedSequence([seed, *(int(value) for value in words)]).generate_state(1)[0])


def _plan_profile(config: Mapping[str, Any], name: str) -> np.ndarray:
    waypoints = np.asarray(config["manipulation_plan"]["actuator_waypoints_rad"][name], dtype=np.float64)
    terminal = float(config["control"]["manipulation_delta_rad"][name])
    if abs(terminal) > 1e-10:
        profile = waypoints / terminal
    elif np.max(np.abs(waypoints), initial=0.0) <= _EPSILON:
        profile = np.asarray([minimum_jerk(index / (waypoints.size - 1)) for index in range(waypoints.size)], dtype=np.float64)
    else:
        raise ValueError(f"{name} has nonzero waypoints with zero terminal delta")
    if not np.isfinite(profile).all() or abs(float(profile[0])) > _EPSILON or abs(float(profile[-1]) - 1.0) > 1e-10:
        raise ValueError(f"{name} manipulation profile is invalid")
    return profile


def _bezier_controls(times: np.ndarray, profile: np.ndarray) -> np.ndarray:
    velocities, accelerations = quintic_c2_knot_derivatives(times, profile)
    values: list[float] = []
    for index, duration in enumerate(np.diff(times)):
        p0, p1 = profile[index : index + 2]
        v0, v1 = velocities[index : index + 2]
        a0, a1 = accelerations[index : index + 2]
        values.extend((p0, p0 + duration * v0 / 5.0, p0 + 2.0 * duration * v0 / 5.0 + duration**2 * a0 / 20.0, p1 - 2.0 * duration * v1 / 5.0 + duration**2 * a1 / 20.0, p1 - duration * v1 / 5.0, p1))
    return np.asarray(values, dtype=np.float64)


def _safe_terminal_bounds(config: Mapping[str, Any], limits: JointRefinementLimits, name: str, preload: float, profile: np.ndarray) -> tuple[float, float]:
    lower, upper = limits.registered_plan_delta_rad[name]
    command_lower, command_upper = limits.command_target_rad[name]
    controls = _bezier_controls(np.asarray(config["manipulation_plan"]["knot_times_s"], dtype=np.float64), profile)
    for coefficient in controls:
        if coefficient > _EPSILON:
            lower = max(lower, (command_lower - preload) / coefficient)
            upper = min(upper, (command_upper - preload) / coefficient)
        elif coefficient < -_EPSILON:
            lower = max(lower, (command_upper - preload) / coefficient)
            upper = min(upper, (command_lower - preload) / coefficient)
    increment = float(np.max(np.abs(np.diff(profile)), initial=0.0))
    if increment > _EPSILON:
        amplitude = float(config["manipulation_plan"]["max_knot_delta_rad"]) / increment
        lower, upper = max(lower, -amplitude), min(upper, amplitude)
    if lower > upper + _EPSILON:
        raise ValueError(f"{name} has no safe refinement interval")
    return float(lower), float(upper)


def _preserve_closing_ray(proposed: dict[str, float], base: Mapping[str, float], precontact: Mapping[str, float], finger: str) -> None:
    names = _FINGER_ACTUATORS[finger]
    base_direction = np.asarray([float(base[name]) - float(precontact[name]) for name in names])
    new_direction = np.asarray([float(proposed[name]) - float(precontact[name]) for name in names])
    if np.max(np.abs(new_direction), initial=0.0) <= 1e-6 or float(new_direction @ base_direction) <= 0.0:
        for name in names:
            proposed[name] = float(base[name])


def _feedback_from_units(config: Mapping[str, Any], block: str, units: np.ndarray, radius: float, budget: RefinementRescueBudget) -> dict[str, Any]:
    base = config["contact_feedback"]
    kp = {finger: float(base["kp_rad_per_n"][finger]) for finger in _FINGERS}
    ki = {finger: float(base["ki_rad_per_n_s"][finger]) for finger in _FINGERS}
    field = kp if block == "feedback_kp" else ki
    for index, finger in enumerate(_FINGERS):
        original = field[finger]
        if block == "feedback_ki" and original == 0.0:
            field[finger] = 0.0
        else:
            field[finger] = original * (1.0 + float(units[index]) * radius * budget.maximum_feedback_fraction)
    return ContactFeedbackParameters(
        schema_version=int(base.get("schema_version", 1)),
        strategy=str(base["strategy"]),
        filter_time_constant_s=float(base["filter_time_constant_s"]),
        kp_rad_per_n=kp,
        ki_rad_per_n_s=ki,
        integral_limit_n_s=float(base["integral_limit_n_s"]),
        correction_limit_rad=float(base["correction_limit_rad"]),
        rate_limit_rad_s=float(base["rate_limit_rad_s"]),
        acceleration_limit_rad_s2=float(base["acceleration_limit_rad_s2"]),
        force_risk_n=float(base["force_risk_n"]),
        freeze_on_risk=bool(base["freeze_on_risk"]),
        max_loss_s=float(base["max_loss_s"]),
        recovery_behavior=str(base["recovery_behavior"]),
        operation_contact_duty_min=float(base["operation_contact_duty_min"]),
        tangent_slip_freeze_threshold_m=(
            float(base["tangent_slip_freeze_threshold_m"])
            if int(base.get("schema_version", 1)) >= 2
            else None
        ),
        tangent_slip_abort_threshold_m=(
            float(base["tangent_slip_abort_threshold_m"])
            if int(base.get("schema_version", 1)) >= 2
            else None
        ),
    ).as_config()


def _materialize_job(parent: RefinementRescueParent, *, units: np.ndarray | None, block: str, radius_index: int, radius: float, local_index: int, job_sequence_index: int, budget: RefinementRescueBudget, rescue_id: str, limits: JointRefinementLimits, validate: bool) -> dict[str, Any]:
    config = copy.deepcopy(dict(parent.config))
    exact = units is None
    if not exact:
        base_preload = contact_preload_targets(config)
        preload = dict(base_preload)
        if block.endswith("_preload"):
            finger = block.removesuffix("_preload")
            for index, name in enumerate(_FINGER_ACTUATORS[finger]):
                lower, upper = limits.preload_target_rad[name]
                preload[name] = float(np.clip(base_preload[name] + float(units[index]) * radius * budget.maximum_preload_radius_rad, lower, upper))
            _preserve_closing_ray(preload, base_preload, precontact_targets(config), finger)
            config["control"]["contact_preload_targets_rad"] = preload

        if block.endswith("_plan"):
            finger = block.removesuffix("_plan")
            terminal = copy.deepcopy(dict(config["control"]["manipulation_delta_rad"]))
            profiles = {name: _plan_profile(config, name) for name in ACTIVE_ACTUATORS}
            for index, name in enumerate(_FINGER_ACTUATORS[finger]):
                lower, upper = _safe_terminal_bounds(config, limits, name, float(preload[name]), profiles[name])
                span = upper - lower
                terminal[name] = float(np.clip(float(terminal[name]) + float(units[index]) * radius * budget.maximum_plan_radius_fraction * span, lower, upper))
            old = config["manipulation_plan"]
            plan = ManipulationPlanParameters(
                schema_version=1,
                profile=str(old["profile"]),
                duration_s=float(old["duration_s"]),
                knot_times_s=tuple(float(value) for value in old["knot_times_s"]),
                actuator_waypoints_rad={name: tuple(float(value) for value in profiles[name] * float(terminal[name])) for name in ACTIVE_ACTUATORS},
                desired_cube_position_delta_m=tuple(tuple(float(axis) for axis in value) for value in old["desired_cube_position_delta_m"]),
                desired_cube_rotation_vector_rad=tuple(tuple(float(axis) for axis in value) for value in old["desired_cube_rotation_vector_rad"]),
                max_knot_delta_rad=float(old["max_knot_delta_rad"]),
                trust_region_backtracks=int(old["trust_region_backtracks"]),
            )
            config["control"]["manipulation_delta_rad"] = terminal
            config["manipulation_plan"] = plan.as_config()
        elif block in ("feedback_kp", "feedback_ki"):
            config["contact_feedback"] = _feedback_from_units(config, block, units, radius, budget)

    identity = {
        "schema_version": REFINEMENT_RESCUE_SCHEMA_VERSION,
        "kind": "v14_contact_preserving_refinement_rescue_candidate",
        "rescue_id": rescue_id,
        "source_authentication_id": parent.source_authentication_id,
        "parent_candidate_id": parent.candidate_id,
        "parent_config_semantic_sha256": parent.config_semantic_sha256,
        "parent_rank": parent.rank,
        "local_index": local_index,
        "parameter_block": block,
        "radius_index": radius_index,
        "radius_fraction": radius,
        "lhs_units": None if units is None else [float(value) for value in units],
        "exact_parent": exact,
        "seed": budget.seed,
    }
    digest = canonical_sha256(identity)
    candidate_id = 14 * 10**15 + int(digest[:12], 16) % 10**14
    if not exact:
        config["controller_id"] = canonical_sha256(
            {
                "schema_version": 1,
                "kind": "v14_contact_preserving_refinement_rescue_controller",
                "grasp_object_pair_id": config.get("grasp_object_pair_id"),
                "planner_id": config.get("planner_id"),
                "plan_id": config["manipulation_plan"]["plan_id"],
                "target_id": config["contact_force_targets_n"]["target_id"],
                "feedback_id": config["contact_feedback"]["feedback_id"],
                "contact_preload_targets_rad": config["control"]["contact_preload_targets_rad"],
            }
        )
    metadata = config.setdefault("candidate_metadata", {})
    metadata["v14_contact_preserving_refinement_rescue"] = {
        **identity,
        "candidate_id": candidate_id,
        "candidate_sha256": digest,
        "job_sequence_index": job_sequence_index,
        "full_reset_required": True,
    }
    if validate:
        validate_config(config)
    return {
        "refinement_rescue_job_schema_version": REFINEMENT_RESCUE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "candidate_sha256": digest,
        "config": config,
        "job_metadata": copy.deepcopy(metadata["v14_contact_preserving_refinement_rescue"]),
    }


def refinement_rescue_id(parents: Sequence[RefinementRescueParent], budget: RefinementRescueBudget) -> str:
    if not parents:
        raise ValueError("refinement rescue requires at least one parent")
    return canonical_sha256(
        {
            "schema_version": REFINEMENT_RESCUE_SCHEMA_VERSION,
            "kind": "v14_contact_preserving_refinement_rescue",
            "source_authentication_id": parents[0].source_authentication_id,
            "parents": [parent.descriptor() for parent in parents],
            "budget": budget.as_mapping(),
        }
    )


def build_refinement_rescue_job_specs(
    parents: Sequence[RefinementRescueParent],
    *,
    budget: RefinementRescueBudget = RefinementRescueBudget(),
    limits_resolver: Any = None,
    validate_configs: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Build exactly 1,024 worker-independent, block-balanced full-reset jobs."""

    ordered = tuple(sorted(parents, key=lambda value: value.rank))
    if not ordered or len(ordered) > budget.maximum_parent_count:
        raise ValueError("rescue parent count is outside the registered budget")
    if tuple(parent.rank for parent in ordered) != tuple(range(len(ordered))):
        raise ValueError("rescue parent ranks must be contiguous from zero")
    if len({parent.config_semantic_sha256 for parent in ordered}) != len(ordered):
        raise ValueError("rescue parents must be config-semantically unique")
    if len({parent.source_authentication_id for parent in ordered}) != 1:
        raise ValueError("all rescue parents must share one authenticated source")
    allocation = allocate_refinement_rescue_budget(len(ordered), budget.total_candidates)
    rescue_id = refinement_rescue_id(ordered, budget)
    resolver = resolve_joint_refinement_limits if limits_resolver is None else limits_resolver
    limits = [resolver(parent.config) for parent in ordered]
    jobs_by_parent: list[list[dict[str, Any]]] = []
    for parent, count, parent_limits in zip(ordered, allocation, limits):
        assignments = [
            (
                budget.parameter_blocks[(local - 1) % len(budget.parameter_blocks)],
                ((local - 1) // len(budget.parameter_blocks)) % len(budget.radius_fractions),
            )
            for local in range(1, count)
        ]
        group_counts: dict[tuple[str, int], int] = {}
        for assignment in assignments:
            group_counts[assignment] = group_counts.get(assignment, 0) + 1
        group_units: dict[tuple[str, int], np.ndarray] = {}
        group_offsets = {key: 0 for key in group_counts}
        for (block, radius_index), group_count in group_counts.items():
            dimension = 3 if block in ("thumb_plan", "index_plan", "thumb_preload", "index_preload", "feedback_kp", "feedback_ki") else 2
            group_units[(block, radius_index)] = _lhs(group_count, dimension, _group_seed(budget.seed, parent.config_semantic_sha256, block, radius_index))
        parent_jobs = [
            _materialize_job(parent, units=None, block="exact_parent", radius_index=-1, radius=0.0, local_index=0, job_sequence_index=parent.rank, budget=budget, rescue_id=rescue_id, limits=parent_limits, validate=validate_configs)
        ]
        for local_index, (block, radius_index) in enumerate(assignments, start=1):
            key = (block, radius_index)
            offset = group_offsets[key]
            group_offsets[key] += 1
            parent_jobs.append(
                _materialize_job(
                    parent,
                    units=group_units[key][offset],
                    block=block,
                    radius_index=radius_index,
                    radius=budget.radius_fractions[radius_index],
                    local_index=local_index,
                    job_sequence_index=local_index * len(ordered) + parent.rank,
                    budget=budget,
                    rescue_id=rescue_id,
                    limits=parent_limits,
                    validate=validate_configs,
                )
            )
        jobs_by_parent.append(parent_jobs)
    jobs = [
        jobs_by_parent[parent_rank][local_index]
        for local_index in range(max(allocation))
        for parent_rank in range(len(ordered))
        if local_index < allocation[parent_rank]
    ]
    if len(jobs) != budget.total_candidates:
        raise AssertionError("refinement rescue violated its exact 1,024-job budget")
    identifiers = [int(job["candidate_id"]) for job in jobs]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("refinement rescue candidate ID collision")
    return tuple(jobs)


def prepare_formal_v2_refinement_rescue(
    source_campaign: str | Path,
    *,
    budget: RefinementRescueBudget = RefinementRescueBudget(),
    validate_configs: bool = True,
) -> tuple[AuthenticatedFormalV2RescueSource, tuple[RefinementRescueParent, ...], tuple[dict[str, Any], ...]]:
    """Convenience entry point for a runner's authenticated source stage."""

    source = authenticate_formal_v2_rescue_source(source_campaign)
    parents = select_refinement_rescue_parents(source, maximum_parent_count=budget.maximum_parent_count)
    jobs = build_refinement_rescue_job_specs(parents, budget=budget, validate_configs=validate_configs)
    return source, parents, jobs


__all__ = [
    "AuthenticatedFormalV2RescueSource",
    "AuthenticatedRescueCandidate",
    "DEFAULT_SEED",
    "DEFAULT_TOTAL_CANDIDATES",
    "MAXIMUM_PARENT_COUNT",
    "PARAMETER_BLOCKS",
    "RADIUS_FRACTIONS",
    "RefinementRescueBudget",
    "RefinementRescueParent",
    "allocate_refinement_rescue_budget",
    "authenticate_formal_v2_rescue_source",
    "build_refinement_rescue_job_specs",
    "prepare_formal_v2_refinement_rescue",
    "rank_refinement_rescue_results",
    "refinement_rescue_candidate_rank",
    "refinement_rescue_id",
    "refinement_rescue_parent_eligible",
    "refinement_rescue_rank_evidence",
    "select_refinement_rescue_parents",
]
