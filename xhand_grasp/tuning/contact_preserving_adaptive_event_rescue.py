"""Adaptive, constraint-projected event rescue for schema-v14 lifts.

This module is deliberately numerical and provenance focused.  It does not
run simulations or write campaign artifacts.  A runner may authenticate a
completed event campaign, materialize the jerk-Pareto full-reset centers that
are still summary-only, and then use the APIs below to build deterministic
full-reset jobs.

Unlike the first event rescue, every sampled perturbation is projected into
the *actual* linear feasible set of the parent 21-knot plan.  The set includes
the adjacent-knot contract, registered terminal bounds, and the continuous
quintic Bezier command hull.  Candidate identity uses a physical-plan hash
which intentionally ignores planner/controller/metadata identifiers.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..actual_contact_grasp_pose_catalog import validate_stage_ledger
from ..artifacts import file_sha256
from ..config import ACTIVE_ACTUATORS, contact_preload_targets, validate_config
from ..experiment import ManipulationPlanParameters
from ..grasp_pose import canonical_sha256
from .contact_constrained_planner import (
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
)
from .contact_preserving_candidate_artifacts import (
    V14CandidateArtifactBundle,
    authenticate_v14_candidate_artifacts,
)
from .contact_preserving_event_rescue import (
    ContactSwitchEvent,
    EventJacobianDirections,
    _checkpoint_jacobian_directions,
    _face_tangent,
    compact_c2_event_bump,
    trace_content_sha256,
)
from .contact_preserving_event_rescue_campaign import (
    authenticate_completed_event_rescue_source,
)
from .contact_preserving_joint_refinement import resolve_joint_refinement_limits
from .contact_preserving_time_warp import (
    _time_warp_controller_id,
    actuator_bezier_hull,
    quintic_bezier_controls,
    validate_time_warped_config,
)


ADAPTIVE_EVENT_SCHEMA_VERSION = 1
EXPERIMENT_ID = "left_opposed_face_palm_down_contact_preserving_planned_lift"
DEFAULT_SEED = 20260821
DEFAULT_MAX_CENTERS = 8
_FINGERS = ("thumb", "index", "mid")
_SHA_LENGTH = 64
_EPS = 1e-12


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence is not an object: {path}")
    return value


def _finite(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _integer(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _zero_normalized(value: Any) -> Any:
    """Canonicalize signed zero and recursively remove non-physical IDs."""

    if isinstance(value, Mapping):
        ignored = {
            "candidate_metadata",
            "controller_id",
            "planner_id",
            "plan_id",
            "feedback_id",
            "object_config_id",
            "grasp_pose_id",
            "grasp_object_pair_id",
        }
        return {
            str(key): _zero_normalized(item)
            for key, item in value.items()
            if str(key) not in ignored
        }
    if isinstance(value, (list, tuple)):
        return [_zero_normalized(item) for item in value]
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


def physical_plan_sha256(config: Mapping[str, Any]) -> str:
    """Hash all physically relevant plan/controller fields, never metadata IDs."""

    if int(config.get("schema_version", 0)) != 14:
        raise ValueError("adaptive event rescue requires schema v14")
    keys = (
        "experiment_id",
        "side",
        "cube",
        "scene",
        "hand_pose",
        "pose_constraints",
        "contact_topology",
        "contact_point_plan",
        "grasp_pose",
        "control_protocol",
        "control",
        "contact_force_targets_n",
        "contact_feedback",
        "manipulation_plan",
    )
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"physical plan is missing fields: {missing}")
    return canonical_sha256(
        {key: _zero_normalized(config[key]) for key in keys}
    )


@dataclass(frozen=True, slots=True)
class AdaptiveEventBudget:
    stage: str = "exploration"
    total_candidate_count: int = 64
    candidates_per_center: int = 64
    max_center_count: int = DEFAULT_MAX_CENTERS
    seed: int = DEFAULT_SEED
    resampling_multiplier: int = 48
    refinement_radius_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.stage not in {"diagnostic", "exploration", "local_refinement"}:
            raise ValueError("unknown adaptive event stage")
        for name in (
            "total_candidate_count",
            "candidates_per_center",
            "max_center_count",
            "resampling_multiplier",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        radius = float(self.refinement_radius_fraction)
        if not math.isfinite(radius) or not 0.0 < radius <= 1.0:
            raise ValueError("refinement_radius_fraction must lie in (0, 1]")
        object.__setattr__(self, "refinement_radius_fraction", radius)

    @classmethod
    def diagnostic(cls, *, center_count: int = 8, seed: int = DEFAULT_SEED):
        count = max(1, int(center_count))
        return cls(
            stage="diagnostic",
            total_candidate_count=8 * count,
            candidates_per_center=8,
            max_center_count=count,
            seed=seed,
        )

    @classmethod
    def exploration(
        cls, *, center_count: int = 8, candidates_per_center: int = 64,
        seed: int = DEFAULT_SEED,
    ):
        count = max(1, int(center_count))
        per = int(candidates_per_center)
        return cls(
            stage="exploration",
            total_candidate_count=count * per,
            candidates_per_center=per,
            max_center_count=count,
            seed=seed,
        )

    @classmethod
    def refinement(
        cls, *, center_count: int = 8, candidates_per_center: int = 64,
        seed: int = DEFAULT_SEED,
    ):
        count = max(1, int(center_count))
        per = int(candidates_per_center)
        return cls(
            stage="local_refinement",
            total_candidate_count=count * per,
            candidates_per_center=per,
            max_center_count=count,
            seed=seed,
        )

    def as_mapping(self) -> dict[str, Any]:
        return {"schema_version": ADAPTIVE_EVENT_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AdaptiveEventBudget":
        payload = dict(raw)
        if int(payload.pop("schema_version", -1)) != ADAPTIVE_EVENT_SCHEMA_VERSION:
            raise ValueError("adaptive budget schema is invalid")
        return cls(**payload)


def allocate_adaptive_event_budget(
    center_count: int, budget: AdaptiveEventBudget
) -> tuple[int, ...]:
    if not 1 <= int(center_count) <= budget.max_center_count:
        raise ValueError("center_count lies outside the adaptive budget")
    total = int(budget.total_candidate_count)
    quotient, remainder = divmod(total, int(center_count))
    allocation = tuple(quotient + int(index < remainder) for index in range(center_count))
    if min(allocation) <= 0:
        raise ValueError("adaptive budget cannot allocate a positive center quota")
    return allocation


@dataclass(frozen=True, slots=True)
class AdaptiveRerunRecord:
    candidate_id: int
    source_stage: str
    source_root: Path
    artifact_directory: Path
    config_path: Path
    result_path: Path
    config_semantic_sha256: str
    result_semantic_sha256: str
    summary_sha256: str
    physical_plan_sha256: str
    peak_jerk_m_s3: float
    source_record_sha256: str
    source_authentication_id: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_EVENT_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "source_stage": self.source_stage,
            "source_root": str(self.source_root),
            "artifact_directory": str(self.artifact_directory),
            "config_path": str(self.config_path),
            "result_path": str(self.result_path),
            "config_semantic_sha256": self.config_semantic_sha256,
            "result_semantic_sha256": self.result_semantic_sha256,
            "summary_sha256": self.summary_sha256,
            "physical_plan_sha256": self.physical_plan_sha256,
            "peak_jerk_m_s3": self.peak_jerk_m_s3,
            "source_record_sha256": self.source_record_sha256,
            "source_authentication_id": self.source_authentication_id,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AdaptiveRerunRecord":
        payload = dict(raw)
        if int(payload.pop("schema_version", -1)) != ADAPTIVE_EVENT_SCHEMA_VERSION:
            raise ValueError("adaptive rerun record schema is invalid")
        for key in ("source_root", "artifact_directory", "config_path", "result_path"):
            payload[key] = Path(payload[key]).expanduser().resolve()
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class AdaptiveEventCenter:
    candidate_id: int
    source_stage: str
    source_authentication_id: str
    source_record_sha256: str
    config: dict[str, Any]
    result: dict[str, Any]
    summary: dict[str, Any]
    config_path: Path
    result_path: Path
    trace_path: Path
    config_semantic_sha256: str
    result_semantic_sha256: str
    trace_sha256: str
    trace_content_sha256: str
    physical_plan_sha256: str
    center_id: str

    def as_mapping(self, *, include_payloads: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": ADAPTIVE_EVENT_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "source_stage": self.source_stage,
            "source_authentication_id": self.source_authentication_id,
            "source_record_sha256": self.source_record_sha256,
            "config_path": str(self.config_path),
            "result_path": str(self.result_path),
            "trace_path": str(self.trace_path),
            "config_semantic_sha256": self.config_semantic_sha256,
            "result_semantic_sha256": self.result_semantic_sha256,
            "trace_sha256": self.trace_sha256,
            "trace_content_sha256": self.trace_content_sha256,
            "physical_plan_sha256": self.physical_plan_sha256,
            "center_id": self.center_id,
        }
        if include_payloads:
            payload.update(
                config=copy.deepcopy(self.config),
                result=copy.deepcopy(self.result),
                summary=copy.deepcopy(self.summary),
            )
        return payload

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AdaptiveEventCenter":
        payload = dict(raw)
        if int(payload.pop("schema_version", -1)) != ADAPTIVE_EVENT_SCHEMA_VERSION:
            raise ValueError("adaptive center schema is invalid")
        for key in ("config_path", "result_path", "trace_path"):
            payload[key] = Path(payload[key]).expanduser().resolve()
        for key in ("config", "result", "summary"):
            if key not in payload:
                path = payload["config_path"] if key == "config" else payload["result_path"]
                loaded = _load_json(path)
                payload[key] = loaded if key != "summary" else copy.deepcopy(loaded["summary"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class AdaptiveEventSource:
    root: Path
    source_authentication_id: str
    manifest_path: Path
    ledger_path: Path
    exploration_report_path: Path
    refinement_report_path: Path
    result_path: Path
    catalog_path: Path
    artifact_paths: tuple[Path, ...]
    retained_centers: tuple[AdaptiveEventCenter, ...]
    rerun_required_records: tuple[AdaptiveRerunRecord, ...]
    source_record_count: int
    ancestor_source_authentication_id: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_EVENT_SCHEMA_VERSION,
            "root": str(self.root),
            "source_authentication_id": self.source_authentication_id,
            "manifest_path": str(self.manifest_path),
            "ledger_path": str(self.ledger_path),
            "exploration_report_path": str(self.exploration_report_path),
            "refinement_report_path": str(self.refinement_report_path),
            "result_path": str(self.result_path),
            "catalog_path": str(self.catalog_path),
            "artifact_paths": [str(path) for path in self.artifact_paths],
            "retained_centers": [value.as_mapping() for value in self.retained_centers],
            "rerun_required_records": [value.as_mapping() for value in self.rerun_required_records],
            "source_record_count": self.source_record_count,
            "ancestor_source_authentication_id": self.ancestor_source_authentication_id,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AdaptiveEventSource":
        payload = dict(raw)
        if int(payload.pop("schema_version", -1)) != ADAPTIVE_EVENT_SCHEMA_VERSION:
            raise ValueError("adaptive source schema is invalid")
        for key in (
            "root", "manifest_path", "ledger_path", "exploration_report_path",
            "refinement_report_path", "result_path", "catalog_path",
        ):
            payload[key] = Path(payload[key]).expanduser().resolve()
        payload["artifact_paths"] = tuple(Path(value).expanduser().resolve() for value in payload["artifact_paths"])
        payload["retained_centers"] = tuple(AdaptiveEventCenter.from_mapping(value) for value in payload["retained_centers"])
        payload["rerun_required_records"] = tuple(AdaptiveRerunRecord.from_mapping(value) for value in payload["rerun_required_records"])
        return cls(**payload)


def _summary_peak_jerk(summary: Mapping[str, Any]) -> float:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        return math.inf
    smoothness = metrics.get("motion_smoothness")
    if not isinstance(smoothness, Mapping):
        return math.inf
    return _finite(
        smoothness.get("operation_peak_abs_filtered_jerk_m_s3"), math.inf
    )


def _summary_nonjerk_failures(summary: Mapping[str, Any]) -> tuple[str, ...]:
    raw = summary.get("failed_checks", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ("malformed_failed_checks",)
    return tuple(
        sorted(
            str(value)
            for value in raw
            if str(value) != "smooth_motion_jerk_within_limit"
        )
    )


def _summary_grasp_success(summary: Mapping[str, Any]) -> bool:
    stage = summary.get("stage_status")
    return isinstance(stage, Mapping) and bool(stage.get("grasp_success", False))


def _report_records(path: Path, *, expected_count: int) -> list[dict[str, Any]]:
    report = _load_json(path)
    records = report.get("records")
    if (
        report.get("complete") is not True
        or not isinstance(records, list)
        or len(records) != expected_count
        or int(report.get("candidate_count", -1)) != expected_count
    ):
        raise RuntimeError(f"adaptive source report is incomplete: {path}")
    if not all(isinstance(value, Mapping) for value in records):
        raise RuntimeError(f"adaptive source report contains a malformed record: {path}")
    return [copy.deepcopy(dict(value)) for value in records]


def _record_candidate_paths(
    source_root: Path, record: Mapping[str, Any]
) -> tuple[Path, Path, Path]:
    relative = record.get("artifact_directory")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuntimeError("adaptive source candidate has an unsafe artifact directory")
    candidate_root = (source_root / relative).resolve()
    if not candidate_root.is_relative_to(source_root):
        raise RuntimeError("adaptive source candidate escaped its campaign")
    return candidate_root, candidate_root / "resolved_config.json", candidate_root / "result.json"


def _make_rerun_record(
    *,
    source_root: Path,
    source_stage: str,
    record: Mapping[str, Any],
    source_authentication_id: str,
) -> AdaptiveRerunRecord:
    candidate_root, config_path, result_path = _record_candidate_paths(source_root, record)
    if not config_path.is_file() or not result_path.is_file():
        raise RuntimeError("adaptive rerun source lost config/result evidence")
    config = _load_json(config_path)
    result = _load_json(result_path)
    candidate_id = int(record.get("candidate_id", -1))
    if int(result.get("candidate_id", -2)) != candidate_id:
        raise RuntimeError("adaptive rerun source candidate identity changed")
    config_sha = canonical_sha256(config)
    if config_sha != record.get("config_semantic_sha256") or config_sha != result.get(
        "config_semantic_sha256"
    ):
        raise RuntimeError("adaptive rerun source config semantic digest changed")
    result_sha = str(result.get("result_semantic_sha256", ""))
    if result_sha != record.get("result_semantic_sha256") or not _is_sha(result_sha):
        raise RuntimeError("adaptive rerun source result semantic digest changed")
    summary = result.get("summary")
    if not isinstance(summary, Mapping) or canonical_sha256(summary) != result.get("summary_sha256"):
        raise RuntimeError("adaptive rerun source summary digest changed")
    if record.get("summary_sha256") != result.get("summary_sha256"):
        raise RuntimeError("adaptive report and candidate summary disagree")
    physical = physical_plan_sha256(config)
    record_payload = {
        "candidate_id": candidate_id,
        "source_stage": source_stage,
        "config_semantic_sha256": config_sha,
        "result_semantic_sha256": result_sha,
        "summary_sha256": result["summary_sha256"],
        "physical_plan_sha256": physical,
        "peak_jerk_m_s3": _summary_peak_jerk(summary),
    }
    return AdaptiveRerunRecord(
        candidate_id=candidate_id,
        source_stage=source_stage,
        source_root=source_root,
        artifact_directory=candidate_root,
        config_path=config_path,
        result_path=result_path,
        config_semantic_sha256=config_sha,
        result_semantic_sha256=result_sha,
        summary_sha256=str(result["summary_sha256"]),
        physical_plan_sha256=physical,
        peak_jerk_m_s3=_summary_peak_jerk(summary),
        source_record_sha256=canonical_sha256(record_payload),
        source_authentication_id=source_authentication_id,
    )


def _pareto_rerun_records(
    records: Sequence[AdaptiveRerunRecord], *, max_records: int = DEFAULT_MAX_CENTERS
) -> tuple[AdaptiveRerunRecord, ...]:
    """Return jerk-Pareto summary records, de-duplicated by physical plan."""

    if max_records <= 0:
        raise ValueError("max_records must be positive")
    best_by_physical: dict[str, AdaptiveRerunRecord] = {}
    for record in records:
        result = _load_json(record.result_path)
        summary = result.get("summary")
        if (
            not isinstance(summary, Mapping)
            or not _summary_grasp_success(summary)
            or _summary_nonjerk_failures(summary)
            or not math.isfinite(record.peak_jerk_m_s3)
        ):
            continue
        previous = best_by_physical.get(record.physical_plan_sha256)
        key = (record.peak_jerk_m_s3, record.candidate_id)
        if previous is None or key < (previous.peak_jerk_m_s3, previous.candidate_id):
            best_by_physical[record.physical_plan_sha256] = record
    ranked = sorted(
        best_by_physical.values(),
        key=lambda value: (value.peak_jerk_m_s3, value.candidate_id),
    )
    return tuple(ranked[:max_records])


def authenticate_adaptive_center_bundle(
    destination: str | Path,
    *,
    source_record: AdaptiveRerunRecord | Mapping[str, Any],
    source_authentication_id: str,
) -> AdaptiveEventCenter:
    """Authenticate one full-reset trace materialized for a summary-only center."""

    record = (
        source_record
        if isinstance(source_record, AdaptiveRerunRecord)
        else AdaptiveRerunRecord.from_mapping(source_record)
    )
    if record.source_authentication_id != source_authentication_id:
        raise RuntimeError("adaptive center record belongs to another source audit")
    source_config = _load_json(record.config_path)
    bundle = authenticate_v14_candidate_artifacts(
        destination,
        expected_config=source_config,
        expected_candidate_id=record.candidate_id,
        require_retained_trace=True,
    )
    if bundle.trace_path is None:
        raise RuntimeError("adaptive center has no full-reset trace")
    config = _load_json(bundle.config_path)
    result = copy.deepcopy(bundle.result)
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        raise RuntimeError("adaptive center result has no summary")
    if physical_plan_sha256(config) != record.physical_plan_sha256:
        raise RuntimeError("adaptive center rerun changed the physical plan")
    trace_sha = file_sha256(bundle.trace_path)
    with np.load(bundle.trace_path, allow_pickle=False) as loaded:
        trace = {name: loaded[name] for name in loaded.files}
    content_sha = trace_content_sha256(trace)
    payload = {
        "candidate_id": record.candidate_id,
        "source_authentication_id": source_authentication_id,
        "source_record_sha256": record.source_record_sha256,
        "config_semantic_sha256": canonical_sha256(config),
        "result_semantic_sha256": result.get("result_semantic_sha256"),
        "trace_sha256": trace_sha,
        "trace_content_sha256": content_sha,
        "physical_plan_sha256": record.physical_plan_sha256,
    }
    return AdaptiveEventCenter(
        candidate_id=record.candidate_id,
        source_stage=record.source_stage,
        source_authentication_id=source_authentication_id,
        source_record_sha256=record.source_record_sha256,
        config=config,
        result=result,
        summary=copy.deepcopy(dict(summary)),
        config_path=bundle.config_path,
        result_path=bundle.result_path,
        trace_path=bundle.trace_path,
        config_semantic_sha256=canonical_sha256(config),
        result_semantic_sha256=str(result.get("result_semantic_sha256", "")),
        trace_sha256=trace_sha,
        trace_content_sha256=content_sha,
        physical_plan_sha256=record.physical_plan_sha256,
        center_id=canonical_sha256(payload),
    )


def authenticate_adaptive_event_source(
    source_campaign: str | Path,
) -> AdaptiveEventSource:
    """Read-only audit of the completed event campaign and its rescue-v2 parent."""

    root = Path(source_campaign).expanduser().resolve()
    manifest_path = root / "campaign_manifest.json"
    ledger_path = root / "stage_ledger.json"
    exploration_path = root / "event_exploration" / "report.json"
    refinement_path = root / "event_local_refinement" / "report.json"
    result_path = root / "event_rescue_result_target_1.json"
    catalog_path = root / "catalogs" / "target_1" / "manipulation" / "catalog.json"
    required = (manifest_path, ledger_path, exploration_path, refinement_path, result_path, catalog_path)
    if not root.is_dir() or not all(path.is_file() for path in required):
        raise RuntimeError("adaptive source lost completed event-campaign evidence")
    ledger = validate_stage_ledger(root)
    required_stages = {
        "event_source_audit", "event_exploration", "event_local_refinement",
        "event_catalog_target_1", "event_result_target_1",
    }
    if not required_stages.issubset(ledger.get("stages", {})):
        raise RuntimeError("adaptive source event campaign is incomplete")
    manifest = _load_json(manifest_path)
    result = _load_json(result_path)
    if (
        manifest.get("campaign_kind") != "contact_preserving_event_aware_rescue"
        or result.get("complete") is not True
        or int(result.get("full_success_count", -1)) != 0
    ):
        raise RuntimeError("adaptive source is not the completed zero-success event campaign")
    exploration = _report_records(exploration_path, expected_count=512)
    refinement = _report_records(refinement_path, expected_count=512)
    ancestor_path = Path(str(manifest.get("source_rescue_campaign_path", ""))).expanduser().resolve()
    ancestor = authenticate_completed_event_rescue_source(ancestor_path)
    ancestor_report = _load_json(ancestor.phase_two_report_path)
    ancestor_records = ancestor_report.get("records")
    if not isinstance(ancestor_records, list) or len(ancestor_records) != 256:
        raise RuntimeError("adaptive source ancestor lost its 256 phase-two records")

    raw_groups = (
        (root, "event_exploration", exploration),
        (root, "event_local_refinement", refinement),
        (ancestor.root, "rescue_v2_time_warp", ancestor_records),
    )
    # Bind the immutable evidence first; records receive this identifier below.
    auth_payload = {
        "schema_version": ADAPTIVE_EVENT_SCHEMA_VERSION,
        "root": str(root),
        "manifest_sha256": file_sha256(manifest_path),
        "ledger_sha256": file_sha256(ledger_path),
        "exploration_report_sha256": file_sha256(exploration_path),
        "refinement_report_sha256": file_sha256(refinement_path),
        "result_sha256": file_sha256(result_path),
        "catalog_sha256": file_sha256(catalog_path),
        "ancestor_source_authentication_id": ancestor.source_authentication_id,
        "candidate_record_set_sha256": canonical_sha256(
            [
                {
                    "stage": stage,
                    "candidate_id": int(record.get("candidate_id", -1)),
                    "config_semantic_sha256": record.get("config_semantic_sha256"),
                    "result_semantic_sha256": record.get("result_semantic_sha256"),
                    "summary_sha256": record.get("summary_sha256"),
                }
                for _, stage, records in raw_groups
                for record in records
            ]
        ),
    }
    authentication_id = canonical_sha256(auth_payload)
    authenticated: list[AdaptiveRerunRecord] = []
    for source_root, stage, records in raw_groups:
        for record in records:
            authenticated.append(
                _make_rerun_record(
                    source_root=source_root,
                    source_stage=stage,
                    record=record,
                    source_authentication_id=authentication_id,
                )
            )
    rerun_required = _pareto_rerun_records(
        authenticated, max_records=DEFAULT_MAX_CENTERS
    )
    records_by_id = {value.candidate_id: value for value in authenticated}
    retained: list[AdaptiveEventCenter] = []
    rerun_root = root / "catalog_source_reruns" / "rescue_target_1"
    if rerun_root.is_dir():
        for destination in sorted(rerun_root.glob("candidate_*")):
            if not destination.is_dir():
                continue
            try:
                candidate_id = int(destination.name.removeprefix("candidate_"))
            except ValueError as exc:
                raise RuntimeError("adaptive source retained an unsafe candidate name") from exc
            record = records_by_id.get(candidate_id)
            if record is None:
                raise RuntimeError("adaptive source retained an unknown candidate")
            retained.append(
                authenticate_adaptive_center_bundle(
                    destination,
                    source_record=record,
                    source_authentication_id=authentication_id,
                )
            )
    return AdaptiveEventSource(
        root=root,
        source_authentication_id=authentication_id,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        exploration_report_path=exploration_path,
        refinement_report_path=refinement_path,
        result_path=result_path,
        catalog_path=catalog_path,
        artifact_paths=tuple(dict.fromkeys((*required, *ancestor.artifact_paths))),
        retained_centers=tuple(retained),
        rerun_required_records=rerun_required,
        source_record_count=len(authenticated),
        ancestor_source_authentication_id=ancestor.source_authentication_id,
    )


def discover_jerk_pareto_records(
    source: AdaptiveEventSource, *, max_records: int = DEFAULT_MAX_CENTERS
) -> tuple[AdaptiveRerunRecord, ...]:
    if not _is_sha(source.source_authentication_id):
        raise ValueError("adaptive source authentication ID is invalid")
    return tuple(source.rerun_required_records[:max_records])


def select_jerk_pareto_centers(
    centers: Sequence[AdaptiveEventCenter], *, max_centers: int = DEFAULT_MAX_CENTERS
) -> tuple[AdaptiveEventCenter, ...]:
    if max_centers <= 0:
        raise ValueError("max_centers must be positive")
    best: dict[str, AdaptiveEventCenter] = {}
    for center in centers:
        if not _is_sha(center.center_id) or not _is_sha(center.physical_plan_sha256):
            raise ValueError("adaptive center identity is invalid")
        if _summary_nonjerk_failures(center.summary) or not _summary_grasp_success(center.summary):
            continue
        previous = best.get(center.physical_plan_sha256)
        key = (_summary_peak_jerk(center.summary), center.candidate_id, center.center_id)
        if previous is None or key < (
            _summary_peak_jerk(previous.summary), previous.candidate_id, previous.center_id
        ):
            best[center.physical_plan_sha256] = center
    return tuple(
        sorted(
            best.values(),
            key=lambda value: (
                _summary_peak_jerk(value.summary), value.candidate_id, value.center_id
            ),
        )[:max_centers]
    )


def discover_jerk_pareto_centers(
    source: AdaptiveEventSource,
    *,
    additional_centers: Sequence[AdaptiveEventCenter] = (),
    max_centers: int = DEFAULT_MAX_CENTERS,
) -> tuple[AdaptiveEventCenter, ...]:
    values = (*source.retained_centers, *additional_centers)
    if any(value.source_authentication_id != source.source_authentication_id for value in values):
        raise RuntimeError("adaptive centers belong to another source audit")
    return select_jerk_pareto_centers(values, max_centers=max_centers)


@dataclass(frozen=True, slots=True)
class AdaptiveEventDetectionSettings:
    tangent_jump_threshold_m: float = 0.00015
    force_jump_threshold_n: float = 0.15
    cluster_window_steps: int = 10
    checkpoint_lead_steps: int = 50
    centered_filter_half_width_steps: int = 25
    jerk_association_extra_steps: int = 5
    minimum_peak_abs_jerk_m_s3: float = 2.5
    global_max_events: int = 64
    selected_fingers: tuple[str, ...] = _FINGERS

    def __post_init__(self) -> None:
        for name in ("tangent_jump_threshold_m", "force_jump_threshold_n"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        for name in (
            "cluster_window_steps", "checkpoint_lead_steps",
            "centered_filter_half_width_steps", "jerk_association_extra_steps",
            "global_max_events",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.cluster_window_steps <= 0 or self.checkpoint_lead_steps <= 0 or self.global_max_events <= 0:
            raise ValueError("cluster, checkpoint and event-count settings must be positive")
        jerk = float(self.minimum_peak_abs_jerk_m_s3)
        if not math.isfinite(jerk) or jerk < 0.0:
            raise ValueError("minimum jerk must be finite and non-negative")
        object.__setattr__(self, "minimum_peak_abs_jerk_m_s3", jerk)
        fingers = tuple(str(value) for value in self.selected_fingers)
        if not fingers or len(set(fingers)) != len(fingers) or any(value not in _FINGERS for value in fingers):
            raise ValueError("selected fingers are invalid")
        object.__setattr__(self, "selected_fingers", fingers)

    def as_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["selected_fingers"] = list(self.selected_fingers)
        return value

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AdaptiveEventDetectionSettings":
        payload = dict(raw)
        payload["selected_fingers"] = tuple(payload["selected_fingers"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class AdaptiveContactEvent:
    finger: str
    finger_index: int
    event_step: int
    checkpoint_step: int
    manipulation_progress: float
    tangent_jump_cube_local_m: tuple[float, float, float]
    tangent_jump_m: float
    taxel_count_before: int
    taxel_count_after: int
    target_force_before_n: float
    target_force_after_n: float
    force_jump_n: float
    force_load_transfer_n: tuple[float, float, float]
    associated_jerk_peak_step: int
    associated_peak_abs_jerk_m_s3: float
    association_radius_steps: int
    event_id: str

    def as_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AdaptiveContactEvent":
        payload = dict(raw)
        payload["tangent_jump_cube_local_m"] = tuple(payload["tangent_jump_cube_local_m"])
        payload["force_load_transfer_n"] = tuple(payload["force_load_transfer_n"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class AdaptiveEventDescriptor:
    schema_version: int
    experiment_id: str
    center_id: str
    source_candidate_id: int
    source_config_semantic_sha256: str
    source_trace_sha256: str
    source_trace_content_sha256: str
    manipulation_start_step: int
    manipulation_end_step: int
    detection_settings: dict[str, Any]
    events: tuple[AdaptiveContactEvent, ...]
    directions: tuple[EventJacobianDirections, ...]
    descriptor_id: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "center_id": self.center_id,
            "source_candidate_id": self.source_candidate_id,
            "source_config_semantic_sha256": self.source_config_semantic_sha256,
            "source_trace_sha256": self.source_trace_sha256,
            "source_trace_content_sha256": self.source_trace_content_sha256,
            "manipulation_start_step": self.manipulation_start_step,
            "manipulation_end_step": self.manipulation_end_step,
            "detection_settings": copy.deepcopy(self.detection_settings),
            "events": [value.as_mapping() for value in self.events],
            "directions": [value.as_mapping() for value in self.directions],
            "descriptor_id": self.descriptor_id,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AdaptiveEventDescriptor":
        payload = dict(raw)
        payload["events"] = tuple(AdaptiveContactEvent.from_mapping(value) for value in payload["events"])
        payload["directions"] = tuple(EventJacobianDirections(**dict(value)) for value in payload["directions"])
        descriptor = cls(**payload)
        _validate_adaptive_event_descriptor(descriptor)
        return descriptor


def _trace_value(trace: Mapping[str, Any], name: str, ndim: int | None = None) -> np.ndarray:
    if name not in trace:
        raise ValueError(f"adaptive event trace is missing {name}")
    value = np.asarray(trace[name])
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"adaptive event trace {name} must have {ndim} dimensions")
    if value.dtype.kind in "fc" and not np.isfinite(value).all():
        raise ValueError(f"adaptive event trace {name} contains non-finite values")
    return value


def _trace_scalar_step(trace: Mapping[str, Any], name: str) -> int:
    values = _trace_value(trace, name)
    if values.size != 1:
        raise ValueError(f"adaptive event trace {name} must be scalar")
    return int(values.reshape(-1)[0])


def detect_adaptive_contact_events(
    config: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    settings: AdaptiveEventDetectionSettings = AdaptiveEventDetectionSettings(),
) -> tuple[AdaptiveContactEvent, ...]:
    """Re-detect all force/taxel/witness events in one retained full trace."""

    if int(config.get("schema_version", 0)) != 14 or config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("adaptive event detection requires the registered v14 experiment")
    finger_order = tuple(str(value) for value in _trace_value(trace, "finger_order").tolist())
    if finger_order != _FINGERS:
        raise ValueError("adaptive event trace finger order changed")
    centroids = _trace_value(trace, "target_face_contact_centroid_cube_local_m", 3).astype(float, copy=False)
    valid = _trace_value(trace, "target_face_contact_centroid_valid", 2).astype(bool, copy=False)
    effective = _trace_value(trace, "target_face_effective", 2).astype(bool, copy=False)
    taxels = _trace_value(trace, "distal_active_taxel_count", 2).astype(np.int64, copy=False)
    force = _trace_value(trace, "contact_force_filtered_n", 2).astype(float, copy=False)
    progress = _trace_value(trace, "manipulation_progress", 1).astype(float, copy=False)
    jerk = _trace_value(trace, "operation_vertical_jerk_filtered_m_s3", 1).astype(float, copy=False)
    length = len(progress)
    expected_2d = (length, len(_FINGERS))
    if centroids.shape != (length, len(_FINGERS), 3) or any(
        value.shape != expected_2d for value in (valid, effective, taxels, force)
    ) or jerk.shape != (length,):
        raise ValueError("adaptive event arrays have inconsistent shapes")
    start = _trace_scalar_step(trace, "manipulation_start_step")
    end = _trace_scalar_step(trace, "manipulation_end_step")
    if not 1 <= start < end <= length:
        raise ValueError("adaptive event manipulation bounds are invalid")
    target_faces = config.get("contact_topology", {}).get("target_faces", {})
    radius = settings.centered_filter_half_width_steps + settings.jerk_association_extra_steps
    raw: list[AdaptiveContactEvent] = []
    for finger in settings.selected_fingers:
        index = _FINGERS.index(finger)
        face = str(target_faces.get(finger, ""))
        for step in range(max(start, 1), end):
            both_valid = bool(valid[step - 1, index] and valid[step, index])
            both_effective = bool(effective[step - 1, index] and effective[step, index])
            tangent = np.zeros(3, dtype=float)
            if both_valid:
                tangent = _face_tangent(centroids[step, index] - centroids[step - 1, index], face)
            tangent_norm = float(np.linalg.norm(tangent))
            taxel_changed = int(taxels[step, index]) != int(taxels[step - 1, index])
            force_jump = float(force[step, index] - force[step - 1, index])
            force_changed = abs(force_jump) >= settings.force_jump_threshold_n
            if not both_effective or not both_valid or not (
                tangent_norm >= settings.tangent_jump_threshold_m or taxel_changed or force_changed
            ):
                continue
            lower = max(start, step - radius)
            upper = min(end, step + radius + 1)
            local = np.abs(jerk[lower:upper])
            peak_offset = int(np.argmax(local))
            peak_step = lower + peak_offset
            peak = float(local[peak_offset])
            if peak < settings.minimum_peak_abs_jerk_m_s3:
                continue
            transfer = force[step] - force[step - 1]
            payload = {
                "schema_version": ADAPTIVE_EVENT_SCHEMA_VERSION,
                "finger": finger,
                "event_step": step,
                "checkpoint_step": max(start, step - settings.checkpoint_lead_steps),
                "tangent_jump_cube_local_m": [float(value) for value in tangent],
                "taxel_count_transition": [int(taxels[step - 1, index]), int(taxels[step, index])],
                "force_jump_n": force_jump,
                "associated_jerk_peak_step": peak_step,
            }
            raw.append(
                AdaptiveContactEvent(
                    finger=finger,
                    finger_index=index,
                    event_step=step,
                    checkpoint_step=max(start, step - settings.checkpoint_lead_steps),
                    manipulation_progress=float(progress[step]),
                    tangent_jump_cube_local_m=tuple(float(value) for value in tangent),
                    tangent_jump_m=tangent_norm,
                    taxel_count_before=int(taxels[step - 1, index]),
                    taxel_count_after=int(taxels[step, index]),
                    target_force_before_n=float(force[step - 1, index]),
                    target_force_after_n=float(force[step, index]),
                    force_jump_n=force_jump,
                    force_load_transfer_n=tuple(float(value) for value in transfer),
                    associated_jerk_peak_step=peak_step,
                    associated_peak_abs_jerk_m_s3=peak,
                    association_radius_steps=radius,
                    event_id=canonical_sha256(payload),
                )
            )

    # Cluster only near-identical events of the same finger.  Unlike v1 this
    # does not impose a two-per-finger cap; every physical cluster survives.
    clustered: list[AdaptiveContactEvent] = []
    for finger in settings.selected_fingers:
        values = sorted((value for value in raw if value.finger == finger), key=lambda value: value.event_step)
        groups: list[list[AdaptiveContactEvent]] = []
        for value in values:
            if not groups or value.event_step - groups[-1][-1].event_step > settings.cluster_window_steps:
                groups.append([value])
            else:
                groups[-1].append(value)
        for group in groups:
            clustered.append(
                max(
                    group,
                    key=lambda value: (
                        value.associated_peak_abs_jerk_m_s3,
                        value.tangent_jump_m,
                        abs(value.force_jump_n),
                        -value.event_step,
                    ),
                )
            )
    clustered.sort(key=lambda value: (value.event_step, value.finger_index, value.event_id))
    if not clustered:
        raise RuntimeError("adaptive detector found no jerk-associated contact events")
    if len(clustered) > settings.global_max_events:
        raise RuntimeError("adaptive detector event count exceeds its declared bound")
    return tuple(clustered)


def _descriptor_identity_payload(descriptor: AdaptiveEventDescriptor) -> dict[str, Any]:
    payload = descriptor.as_mapping()
    payload.pop("descriptor_id")
    return payload


def _validate_adaptive_event_descriptor(descriptor: AdaptiveEventDescriptor) -> None:
    if descriptor.schema_version != ADAPTIVE_EVENT_SCHEMA_VERSION or descriptor.experiment_id != EXPERIMENT_ID:
        raise ValueError("adaptive descriptor version or experiment is invalid")
    for value in (
        descriptor.center_id, descriptor.source_config_semantic_sha256,
        descriptor.source_trace_sha256, descriptor.source_trace_content_sha256,
        descriptor.descriptor_id,
    ):
        if not _is_sha(value):
            raise ValueError("adaptive descriptor contains an invalid SHA-256")
    settings = AdaptiveEventDetectionSettings.from_mapping(descriptor.detection_settings)
    if settings.as_mapping() != descriptor.detection_settings:
        raise ValueError("adaptive descriptor settings are not canonical")
    if not descriptor.events or len(descriptor.events) != len(descriptor.directions):
        raise ValueError("adaptive descriptor must bind one direction per event")
    if tuple(value.event_id for value in descriptor.events) != tuple(value.event_id for value in descriptor.directions):
        raise ValueError("adaptive descriptor event/direction ordering changed")
    for event in descriptor.events:
        event_payload = {
            "schema_version": ADAPTIVE_EVENT_SCHEMA_VERSION,
            "finger": event.finger,
            "event_step": event.event_step,
            "checkpoint_step": event.checkpoint_step,
            "tangent_jump_cube_local_m": list(event.tangent_jump_cube_local_m),
            "taxel_count_transition": [event.taxel_count_before, event.taxel_count_after],
            "force_jump_n": event.force_jump_n,
            "associated_jerk_peak_step": event.associated_jerk_peak_step,
        }
        if event.event_id != canonical_sha256(event_payload):
            raise ValueError("adaptive contact-event identity changed")
    for direction in descriptor.directions:
        if (
            direction.finger not in _FINGERS
            or not _is_sha(direction.directions_id)
            or len(direction.tangent_direction_rad_unit) != len(ACTIVE_ACTUATORS)
            or len(direction.normal_unload_direction_rad_unit) != len(ACTIVE_ACTUATORS)
            or not np.isfinite(direction.tangent_direction_rad_unit).all()
            or not np.isfinite(direction.normal_unload_direction_rad_unit).all()
        ):
            raise ValueError("adaptive event direction evidence is invalid")
    if canonical_sha256(_descriptor_identity_payload(descriptor)) != descriptor.descriptor_id:
        raise ValueError("adaptive descriptor identity changed")


def build_adaptive_event_descriptor(
    center: AdaptiveEventCenter,
    trace: Mapping[str, Any] | None = None,
    *,
    settings: AdaptiveEventDetectionSettings = AdaptiveEventDetectionSettings(),
) -> AdaptiveEventDescriptor:
    if file_sha256(center.trace_path) != center.trace_sha256:
        raise RuntimeError("adaptive center trace file changed")
    if trace is None:
        with np.load(center.trace_path, allow_pickle=False) as loaded:
            trace_values = {name: loaded[name] for name in loaded.files}
    else:
        trace_values = {name: np.asarray(trace[name]) for name in trace}
    if trace_content_sha256(trace_values) != center.trace_content_sha256:
        raise RuntimeError("adaptive center trace content changed")
    events = detect_adaptive_contact_events(center.config, trace_values, settings=settings)
    directions: list[EventJacobianDirections] = []
    for event in events:
        legacy = ContactSwitchEvent(
            finger=event.finger,
            finger_index=event.finger_index,
            event_step=event.event_step,
            checkpoint_step=event.checkpoint_step,
            manipulation_progress=event.manipulation_progress,
            tangent_jump_cube_local_m=event.tangent_jump_cube_local_m,
            tangent_jump_m=event.tangent_jump_m,
            local_peak_abs_jerk_m_s3=event.associated_peak_abs_jerk_m_s3,
            event_id=event.event_id,
        )
        directions.append(_checkpoint_jacobian_directions(center.config, trace_values, legacy))
    start = _trace_scalar_step(trace_values, "manipulation_start_step")
    end = _trace_scalar_step(trace_values, "manipulation_end_step")
    base = AdaptiveEventDescriptor(
        schema_version=ADAPTIVE_EVENT_SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        center_id=center.center_id,
        source_candidate_id=center.candidate_id,
        source_config_semantic_sha256=center.config_semantic_sha256,
        source_trace_sha256=center.trace_sha256,
        source_trace_content_sha256=center.trace_content_sha256,
        manipulation_start_step=start,
        manipulation_end_step=end,
        detection_settings=settings.as_mapping(),
        events=events,
        directions=tuple(directions),
        descriptor_id="0" * 64,
    )
    descriptor = replace(base, descriptor_id=canonical_sha256(_descriptor_identity_payload(base)))
    _validate_adaptive_event_descriptor(descriptor)
    return descriptor


def authenticate_adaptive_event_descriptor(
    descriptor: AdaptiveEventDescriptor | Mapping[str, Any],
    center_or_config: AdaptiveEventCenter | Mapping[str, Any],
    trace: Mapping[str, Any] | None = None,
) -> AdaptiveEventDescriptor:
    value = descriptor if isinstance(descriptor, AdaptiveEventDescriptor) else AdaptiveEventDescriptor.from_mapping(descriptor)
    _validate_adaptive_event_descriptor(value)
    if isinstance(center_or_config, AdaptiveEventCenter):
        center = center_or_config
        if (
            value.center_id != center.center_id
            or value.source_candidate_id != center.candidate_id
            or value.source_config_semantic_sha256 != center.config_semantic_sha256
            or value.source_trace_sha256 != center.trace_sha256
            or value.source_trace_content_sha256 != center.trace_content_sha256
        ):
            raise RuntimeError("adaptive descriptor does not authenticate its center")
    else:
        config = dict(center_or_config)
        if canonical_sha256(config) != value.source_config_semantic_sha256:
            raise RuntimeError("adaptive descriptor config digest changed")
        if trace is None or trace_content_sha256(trace) != value.source_trace_content_sha256:
            raise RuntimeError("adaptive descriptor trace content changed")
    return value


@dataclass(frozen=True, slots=True)
class FeasibleEventPolytope:
    schema_version: int
    experiment_id: str
    descriptor_id: str
    parent_physical_plan_sha256: str
    parameter_names: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    constraint_matrix: tuple[tuple[float, ...], ...]
    constraint_upper: tuple[float, ...]
    constraint_labels: tuple[str, ...]
    base_waypoints_rad: tuple[tuple[float, ...], ...]
    basis_waypoints_rad: tuple[tuple[tuple[float, ...], ...], ...]
    bump_half_width_progress: float
    projection_tolerance: float
    projection_max_iterations: int
    polytope_id: str

    def __post_init__(self) -> None:
        dimension = len(self.parameter_names)
        if dimension == 0 or len(set(self.parameter_names)) != dimension:
            raise ValueError("adaptive polytope parameter names are empty or duplicated")
        if len(self.lower_bounds) != dimension or len(self.upper_bounds) != dimension:
            raise ValueError("adaptive polytope bounds have the wrong dimension")
        if any(not math.isfinite(value) for value in (*self.lower_bounds, *self.upper_bounds)) or any(
            lower > upper for lower, upper in zip(self.lower_bounds, self.upper_bounds)
        ):
            raise ValueError("adaptive polytope bounds are invalid")
        if not (
            len(self.constraint_matrix) == len(self.constraint_upper) == len(self.constraint_labels)
            and all(len(row) == dimension for row in self.constraint_matrix)
        ):
            raise ValueError("adaptive polytope constraints have inconsistent shapes")
        base = np.asarray(self.base_waypoints_rad, dtype=float)
        basis = np.asarray(self.basis_waypoints_rad, dtype=float)
        if base.ndim != 2 or base.shape[1] != len(ACTIVE_ACTUATORS) or basis.shape != (dimension, *base.shape):
            raise ValueError("adaptive polytope waypoint basis has the wrong shape")
        if not np.isfinite(base).all() or not np.isfinite(basis).all():
            raise ValueError("adaptive polytope waypoint basis is non-finite")
        if not _is_sha(self.descriptor_id) or not _is_sha(self.parent_physical_plan_sha256) or not _is_sha(self.polytope_id):
            raise ValueError("adaptive polytope contains an invalid SHA-256")
        if self.schema_version != ADAPTIVE_EVENT_SCHEMA_VERSION or self.experiment_id != EXPERIMENT_ID:
            raise ValueError("adaptive polytope version or experiment is invalid")
        if self.projection_tolerance <= 0.0 or self.projection_max_iterations <= 0:
            raise ValueError("adaptive polytope projection settings are invalid")
        if canonical_sha256(self._identity_payload()) != self.polytope_id:
            raise ValueError("adaptive polytope identity changed")
        if not self.contains(np.zeros(dimension), tolerance=max(self.projection_tolerance, 1e-9)):
            raise ValueError("adaptive polytope unexpectedly excludes its parent")

    def _identity_payload(self) -> dict[str, Any]:
        payload = self.as_mapping()
        payload.pop("polytope_id")
        return payload

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "descriptor_id": self.descriptor_id,
            "parent_physical_plan_sha256": self.parent_physical_plan_sha256,
            "parameter_names": list(self.parameter_names),
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
            "constraint_matrix": [list(row) for row in self.constraint_matrix],
            "constraint_upper": list(self.constraint_upper),
            "constraint_labels": list(self.constraint_labels),
            "base_waypoints_rad": [list(row) for row in self.base_waypoints_rad],
            "basis_waypoints_rad": [[list(row) for row in matrix] for matrix in self.basis_waypoints_rad],
            "bump_half_width_progress": self.bump_half_width_progress,
            "projection_tolerance": self.projection_tolerance,
            "projection_max_iterations": self.projection_max_iterations,
            "polytope_id": self.polytope_id,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FeasibleEventPolytope":
        payload = dict(raw)
        for name in ("parameter_names", "lower_bounds", "upper_bounds", "constraint_upper", "constraint_labels"):
            payload[name] = tuple(payload[name])
        payload["constraint_matrix"] = tuple(tuple(row) for row in payload["constraint_matrix"])
        payload["base_waypoints_rad"] = tuple(tuple(row) for row in payload["base_waypoints_rad"])
        payload["basis_waypoints_rad"] = tuple(
            tuple(tuple(row) for row in matrix) for matrix in payload["basis_waypoints_rad"]
        )
        return cls(**payload)

    @property
    def dimension(self) -> int:
        return len(self.parameter_names)

    def contains(self, parameters: Sequence[float], *, tolerance: float | None = None) -> bool:
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.dimension,) or not np.isfinite(values).all():
            return False
        tol = self.projection_tolerance if tolerance is None else float(tolerance)
        if np.any(values < np.asarray(self.lower_bounds) - tol) or np.any(values > np.asarray(self.upper_bounds) + tol):
            return False
        if self.constraint_matrix:
            matrix = np.asarray(self.constraint_matrix)
            upper = np.asarray(self.constraint_upper)
            if np.any(matrix @ values > upper + tol):
                return False
        return True

    def project(self, parameters: Sequence[float]) -> np.ndarray:
        """Deterministic cyclic projection onto the box and all halfspaces."""

        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.dimension,) or not np.isfinite(values).all():
            raise ValueError("adaptive projection input is invalid")
        lower = np.asarray(self.lower_bounds)
        upper = np.asarray(self.upper_bounds)
        matrix = np.asarray(self.constraint_matrix, dtype=float)
        bounds = np.asarray(self.constraint_upper, dtype=float)
        current = np.clip(values, lower, upper)
        inward = max(5e-14, 0.25 * self.projection_tolerance)
        for _ in range(self.projection_max_iterations):
            previous = current.copy()
            for row, bound in zip(matrix, bounds, strict=True):
                violation = float(row @ current - bound)
                norm_squared = float(row @ row)
                if violation > 0.0 and norm_squared > _EPS:
                    current -= ((violation + inward) / norm_squared) * row
                    current = np.clip(current, lower, upper)
            if np.max(np.abs(current - previous), initial=0.0) <= self.projection_tolerance and self.contains(
                current, tolerance=5.0 * self.projection_tolerance
            ):
                break
        # A final exact-bound cleanup is intentionally stricter than the
        # public membership tolerance.  ManipulationPlanParameters admits at
        # most 1e-12 at an adjacent-knot boundary, so serialized candidates
        # must not rely on an outward feasibility epsilon.
        for _ in range(64):
            changed = False
            for row, bound in zip(matrix, bounds, strict=True):
                violation = float(row @ current - bound)
                norm_squared = float(row @ row)
                if violation > 0.0 and norm_squared > _EPS:
                    current -= ((violation + inward) / norm_squared) * row
                    current = np.clip(current, lower, upper)
                    changed = True
            if not changed:
                break
        if not self.contains(current, tolerance=1e-8):
            # The origin is authenticated feasible.  Bisection on the ray to
            # it is a deterministic fail-closed fallback for ill-conditioned
            # cyclic projections, not a constraint backoff.
            candidate = current.copy()
            low, high = 0.0, 1.0
            for _ in range(80):
                scale = (low + high) / 2.0
                trial = scale * candidate
                if self.contains(trial, tolerance=1e-10):
                    low = scale
                else:
                    high = scale
            current = low * candidate
        if not self.contains(current, tolerance=1e-8):
            raise RuntimeError("adaptive feasible-polytope projection failed")
        return current

    def parameter_mapping(self, parameters: Sequence[float]) -> dict[str, float]:
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.dimension,) or not np.isfinite(values).all():
            raise ValueError("adaptive parameter vector is invalid")
        return {name: float(value) for name, value in zip(self.parameter_names, values, strict=True)}

    def parameter_vector(self, parameters: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
        if isinstance(parameters, Mapping):
            if set(parameters) != set(self.parameter_names):
                raise ValueError("adaptive parameter mapping keys changed")
            values = np.asarray([float(parameters[name]) for name in self.parameter_names])
        else:
            values = np.asarray(parameters, dtype=float)
        if values.shape != (self.dimension,) or not np.isfinite(values).all():
            raise ValueError("adaptive parameter vector is invalid")
        return values


def _append_linear_constraint(
    rows: list[np.ndarray], bounds: list[float], labels: list[str],
    row: np.ndarray, bound: float, label: str,
    lower: np.ndarray, upper: np.ndarray,
) -> None:
    maximum = float(np.where(row >= 0.0, row * upper, row * lower).sum())
    if maximum <= float(bound) + 1e-14:
        return
    if float(row @ np.zeros_like(row)) > float(bound) + 1e-9:
        raise RuntimeError(f"adaptive parent violates declared constraint {label}")
    rows.append(row.astype(float, copy=True))
    bounds.append(float(bound))
    labels.append(label)


def derive_feasible_event_polytope(
    config: Mapping[str, Any],
    descriptor: AdaptiveEventDescriptor | Mapping[str, Any],
    *,
    bump_half_width_progress: float = 0.13,
    tangent_radius_rad: float = 0.015,
    normal_unload_radius_rad: float = 0.006,
    terminal_shrink_upper: float = 0.02,
    projection_tolerance: float = 1e-12,
    projection_max_iterations: int = 512,
) -> FeasibleEventPolytope:
    value = descriptor if isinstance(descriptor, AdaptiveEventDescriptor) else AdaptiveEventDescriptor.from_mapping(descriptor)
    _validate_adaptive_event_descriptor(value)
    if canonical_sha256(config) != value.source_config_semantic_sha256:
        raise RuntimeError("adaptive polytope config does not match descriptor")
    plan = ManipulationPlanParameters.from_config(config["manipulation_plan"])
    times = np.asarray(plan.knot_times_s, dtype=float)
    base = np.stack(
        [np.asarray(plan.actuator_waypoints_rad[name], dtype=float) for name in ACTIVE_ACTUATORS], axis=1
    )
    names = ["terminal_shrink"]
    lower = [0.0]
    upper = [float(terminal_shrink_upper)]
    basis = [-base]
    directions = {item.event_id: item for item in value.directions}
    for event in value.events:
        bump = compact_c2_event_bump(times / plan.duration_s, event.manipulation_progress, bump_half_width_progress)
        direction = directions[event.event_id]
        names.extend((f"event:{event.event_id}:tangent", f"event:{event.event_id}:unload"))
        lower.extend((-float(tangent_radius_rad), 0.0))
        upper.extend((float(tangent_radius_rad), float(normal_unload_radius_rad)))
        basis.extend(
            (
                bump[:, None] * np.asarray(direction.tangent_direction_rad_unit, dtype=float)[None, :],
                bump[:, None] * np.asarray(direction.normal_unload_direction_rad_unit, dtype=float)[None, :],
            )
        )
    basis_array = np.asarray(basis, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    labels: list[str] = []

    for knot in range(base.shape[0] - 1):
        for actuator, name in enumerate(ACTIVE_ACTUATORS):
            base_delta = float(base[knot + 1, actuator] - base[knot, actuator])
            coefficients = basis_array[:, knot + 1, actuator] - basis_array[:, knot, actuator]
            _append_linear_constraint(rows, bounds, labels, coefficients, plan.max_knot_delta_rad - base_delta, f"adjacent_upper:{name}:{knot}", lower_array, upper_array)
            _append_linear_constraint(rows, bounds, labels, -coefficients, plan.max_knot_delta_rad + base_delta, f"adjacent_lower:{name}:{knot}", lower_array, upper_array)

    limits = resolve_joint_refinement_limits(config)
    for actuator, name in enumerate(ACTIVE_ACTUATORS):
        coefficients = basis_array[:, -1, actuator]
        lo, hi = limits.registered_plan_delta_rad[name]
        _append_linear_constraint(rows, bounds, labels, coefficients, hi - base[-1, actuator], f"terminal_upper:{name}", lower_array, upper_array)
        _append_linear_constraint(rows, bounds, labels, -coefficients, base[-1, actuator] - lo, f"terminal_lower:{name}", lower_array, upper_array)

    base_controls = quintic_bezier_controls(times, base)
    basis_controls = np.stack([quintic_bezier_controls(times, matrix) for matrix in basis_array])
    preload = contact_preload_targets(copy.deepcopy(dict(config)))
    for segment in range(base_controls.shape[0]):
        for control_index in range(base_controls.shape[1]):
            for actuator, name in enumerate(ACTIVE_ACTUATORS):
                command_base = float(preload[name]) + float(base_controls[segment, control_index, actuator])
                coefficients = basis_controls[:, segment, control_index, actuator]
                lo, hi = limits.command_target_rad[name]
                _append_linear_constraint(rows, bounds, labels, coefficients, hi - command_base, f"bezier_upper:{name}:{segment}:{control_index}", lower_array, upper_array)
                _append_linear_constraint(rows, bounds, labels, -coefficients, command_base - lo, f"bezier_lower:{name}:{segment}:{control_index}", lower_array, upper_array)

    identity_payload = {
        "schema_version": ADAPTIVE_EVENT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "descriptor_id": value.descriptor_id,
        "parent_physical_plan_sha256": physical_plan_sha256(config),
        "parameter_names": list(names),
        "lower_bounds": [float(item) for item in lower_array],
        "upper_bounds": [float(item) for item in upper_array],
        "constraint_matrix": [[float(item) for item in row] for row in rows],
        "constraint_upper": list(bounds),
        "constraint_labels": list(labels),
        "base_waypoints_rad": [[float(item) for item in row] for row in base],
        "basis_waypoints_rad": [
            [[float(item) for item in row] for row in matrix] for matrix in basis_array
        ],
        "bump_half_width_progress": float(bump_half_width_progress),
        "projection_tolerance": float(projection_tolerance),
        "projection_max_iterations": int(projection_max_iterations),
    }
    return FeasibleEventPolytope(
        schema_version=ADAPTIVE_EVENT_SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        descriptor_id=value.descriptor_id,
        parent_physical_plan_sha256=physical_plan_sha256(config),
        parameter_names=tuple(names),
        lower_bounds=tuple(float(item) for item in lower_array),
        upper_bounds=tuple(float(item) for item in upper_array),
        constraint_matrix=tuple(tuple(float(item) for item in row) for row in rows),
        constraint_upper=tuple(bounds),
        constraint_labels=tuple(labels),
        base_waypoints_rad=tuple(tuple(float(item) for item in row) for row in base),
        basis_waypoints_rad=tuple(
            tuple(tuple(float(item) for item in row) for row in matrix) for matrix in basis_array
        ),
        bump_half_width_progress=float(bump_half_width_progress),
        projection_tolerance=float(projection_tolerance),
        projection_max_iterations=int(projection_max_iterations),
        polytope_id=canonical_sha256(identity_payload),
    )


def _adaptive_plan_config(
    parent_config: Mapping[str, Any],
    descriptor: AdaptiveEventDescriptor,
    polytope: FeasibleEventPolytope,
    parameters: Sequence[float],
    *,
    stage: str,
    candidate_id: int,
    local_index: int,
) -> dict[str, Any]:
    values = np.asarray(parameters, dtype=float)
    if not polytope.contains(values, tolerance=1e-8):
        raise ValueError("adaptive candidate parameters lie outside the feasible polytope")
    config = copy.deepcopy(dict(parent_config))
    base = np.asarray(polytope.base_waypoints_rad, dtype=float)
    basis = np.asarray(polytope.basis_waypoints_rad, dtype=float)
    waypoints = base + np.tensordot(values, basis, axes=(0, 0))
    waypoints[0] = 0.0
    original = ManipulationPlanParameters.from_config(config["manipulation_plan"])
    plan = ManipulationPlanParameters(
        schema_version=original.schema_version,
        profile=original.profile,
        duration_s=original.duration_s,
        knot_times_s=original.knot_times_s,
        actuator_waypoints_rad={
            name: tuple(float(value) for value in waypoints[:, index])
            for index, name in enumerate(ACTIVE_ACTUATORS)
        },
        desired_cube_position_delta_m=original.desired_cube_position_delta_m,
        desired_cube_rotation_vector_rad=original.desired_cube_rotation_vector_rad,
        max_knot_delta_rad=original.max_knot_delta_rad,
        trust_region_backtracks=original.trust_region_backtracks,
    )
    config["manipulation_plan"] = plan.as_config()
    config["control"]["manipulation_delta_rad"] = {
        name: float(waypoints[-1, index])
        for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    config["object_config_id"] = v14_object_config_id(config)
    config["grasp_pose_id"] = v14_grasp_pose_id(config)
    config["grasp_object_pair_id"] = v14_grasp_object_pair_id(config)
    parameter_mapping = polytope.parameter_mapping(values)
    config["planner_id"] = canonical_sha256(
        {
            "schema_version": ADAPTIVE_EVENT_SCHEMA_VERSION,
            "kind": "v14_adaptive_contact_event_feasible_plan",
            "parent_physical_plan_sha256": polytope.parent_physical_plan_sha256,
            "descriptor_id": descriptor.descriptor_id,
            "polytope_id": polytope.polytope_id,
            "stage": stage,
            "projected_parameters": parameter_mapping,
        }
    )
    config["controller_id"] = _time_warp_controller_id(config)
    metadata = config.setdefault("candidate_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("adaptive parent candidate_metadata is malformed")
    metadata["v14_contact_preserving_adaptive_event_rescue"] = {
        "schema_version": ADAPTIVE_EVENT_SCHEMA_VERSION,
        "candidate_id": int(candidate_id),
        "local_index": int(local_index),
        "stage": stage,
        "source_candidate_id": descriptor.source_candidate_id,
        "center_id": descriptor.center_id,
        "descriptor_id": descriptor.descriptor_id,
        "polytope_id": polytope.polytope_id,
        "projected_parameters": parameter_mapping,
        "full_reset_required": True,
    }
    # The exact linear polytope already proves the continuous Bezier, terminal
    # and adjacent-knot limits.  Schema validation remains independent and
    # avoids recompiling an identical MuJoCo model for every sampled point.
    validate_config(config)
    return config


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    if count <= 0 or dimensions <= 0:
        raise ValueError("adaptive LHS dimensions must be positive")
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=float)
    for column in range(dimensions):
        order = rng.permutation(count)
        result[:, column] = 2.0 * ((order + rng.random(count)) / float(count)) - 1.0
    return result


def _adaptive_sampling_seed(
    center: AdaptiveEventCenter,
    descriptor: AdaptiveEventDescriptor,
    polytope: FeasibleEventPolytope,
    budget: AdaptiveEventBudget,
) -> int:
    digest = canonical_sha256(
        {
            "budget": budget.as_mapping(),
            "center_id": center.center_id,
            "descriptor_id": descriptor.descriptor_id,
            "polytope_id": polytope.polytope_id,
        }
    )
    return int.from_bytes(bytes.fromhex(digest[:16]), "little")


def _normalized_to_physical(unit: np.ndarray, polytope: FeasibleEventPolytope) -> np.ndarray:
    lower = np.asarray(polytope.lower_bounds)
    upper = np.asarray(polytope.upper_bounds)
    return lower + 0.5 * (unit + 1.0) * (upper - lower)


def _candidate_id_from_projection(
    *, center_id: str, stage: str, local_index: int, projected: Mapping[str, float]
) -> int:
    digest = canonical_sha256(
        {
            "schema_version": ADAPTIVE_EVENT_SCHEMA_VERSION,
            "center_id": center_id,
            "stage": stage,
            "local_index": int(local_index),
            "projected_parameters": projected,
        }
    )
    return 14_800_000_000_000_000 + int(digest[:13], 16) % 100_000_000_000_000


def build_adaptive_event_jobs(
    center: AdaptiveEventCenter,
    descriptor: AdaptiveEventDescriptor | Mapping[str, Any],
    polytope: FeasibleEventPolytope | Mapping[str, Any],
    *,
    budget: AdaptiveEventBudget,
    local_index_offset: int = 0,
    refinement_centers: Sequence[Mapping[str, Any]] = (),
    excluded_physical_plan_sha256: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Generate deterministic projected-LHS jobs with physical de-duplication."""

    value = descriptor if isinstance(descriptor, AdaptiveEventDescriptor) else AdaptiveEventDescriptor.from_mapping(descriptor)
    feasible = polytope if isinstance(polytope, FeasibleEventPolytope) else FeasibleEventPolytope.from_mapping(polytope)
    authenticate_adaptive_event_descriptor(value, center)
    if feasible.descriptor_id != value.descriptor_id or feasible.parent_physical_plan_sha256 != center.physical_plan_sha256:
        raise RuntimeError("adaptive job context is inconsistent")
    if local_index_offset < 0:
        raise ValueError("local_index_offset must be non-negative")
    requested_count = int(budget.candidates_per_center)
    pool_count = max(requested_count * budget.resampling_multiplier, requested_count + 32)
    sampling_seed = _adaptive_sampling_seed(center, value, feasible, budget)
    lhs = _latin_hypercube(
        pool_count,
        feasible.dimension,
        sampling_seed,
    )
    lower = np.asarray(feasible.lower_bounds)
    upper = np.asarray(feasible.upper_bounds)
    refinement_vectors: tuple[np.ndarray, ...] = tuple(
        feasible.parameter_vector(raw.get("projected_parameters", raw))
        for raw in refinement_centers
    )
    if budget.stage == "local_refinement" and not refinement_vectors:
        raise ValueError("adaptive local refinement requires projected center parameters")
    excluded = tuple(sorted(set(str(item) for item in excluded_physical_plan_sha256)))
    if any(not _is_sha(item) for item in excluded):
        raise ValueError("adaptive excluded physical-plan set contains an invalid SHA-256")
    exclusion_set_sha = canonical_sha256(list(excluded))
    jobs: list[dict[str, Any]] = []
    seen_physical = {center.physical_plan_sha256, *excluded}
    seen_candidate_ids: set[int] = set()
    for row_index, unit in enumerate(lhs):
        if budget.stage == "local_refinement":
            anchor = refinement_vectors[row_index % len(refinement_vectors)]
            radius = budget.refinement_radius_fraction
            requested = anchor + radius * 0.5 * (upper - lower) * unit
            requested = np.clip(requested, lower, upper)
        else:
            requested = _normalized_to_physical(unit, feasible)
            if budget.stage == "diagnostic":
                midpoint = 0.5 * (lower + upper)
                requested = midpoint + 0.35 * (requested - midpoint)
        projected = feasible.project(requested)
        if np.max(np.abs(projected), initial=0.0) <= 1e-11:
            continue
        projected_mapping = feasible.parameter_mapping(projected)
        local_index = int(local_index_offset) + len(jobs)
        candidate_id = _candidate_id_from_projection(
            center_id=center.center_id,
            stage=budget.stage,
            local_index=local_index,
            projected=projected_mapping,
        )
        if candidate_id in seen_candidate_ids:
            continue
        config = _adaptive_plan_config(
            center.config,
            value,
            feasible,
            projected,
            stage=budget.stage,
            candidate_id=candidate_id,
            local_index=local_index,
        )
        physical = physical_plan_sha256(config)
        if physical in seen_physical:
            continue
        seen_physical.add(physical)
        seen_candidate_ids.add(candidate_id)
        normalized = 2.0 * (requested - lower) / np.maximum(upper - lower, _EPS) - 1.0
        core = {
            "schema_version": ADAPTIVE_EVENT_SCHEMA_VERSION,
            "kind": "v14_adaptive_contact_event_candidate",
            "stage": budget.stage,
            "candidate_id": candidate_id,
            "local_index": local_index,
            "source_candidate_id": center.candidate_id,
            "source_center_id": center.center_id,
            "source_physical_plan_sha256": center.physical_plan_sha256,
            "descriptor_id": value.descriptor_id,
            "polytope_id": feasible.polytope_id,
            "requested_normalized_parameters": feasible.parameter_mapping(normalized),
            "requested_parameters": feasible.parameter_mapping(requested),
            "projected_parameters": projected_mapping,
            "projection_distance": float(np.linalg.norm(projected - requested)),
            "sampling_seed": sampling_seed,
            "lhs_pool_count": pool_count,
            "lhs_pool_row_index": row_index,
            "refinement_centers": [
                feasible.parameter_mapping(item) for item in refinement_vectors
            ],
            "refinement_centers_sha256": canonical_sha256(
                [feasible.parameter_mapping(item) for item in refinement_vectors]
            ),
            "config_semantic_sha256": canonical_sha256(config),
            "physical_plan_sha256": physical,
            "excluded_physical_plan_set_sha256": exclusion_set_sha,
            "excluded_physical_plan_count": len(excluded),
            "full_reset_required": True,
            "budget": budget.as_mapping(),
        }
        payload_sha = canonical_sha256(core)
        jobs.append({**core, "candidate_payload_sha256": payload_sha, "config": config})
        if len(jobs) == requested_count:
            break
    if len(jobs) != requested_count:
        raise RuntimeError(
            f"adaptive projected sampling produced {len(jobs)} unique jobs, expected {requested_count}"
        )
    return tuple(jobs)


def authenticate_adaptive_event_job(
    job: Mapping[str, Any],
    center: AdaptiveEventCenter,
    descriptor: AdaptiveEventDescriptor | Mapping[str, Any],
    polytope: FeasibleEventPolytope | Mapping[str, Any],
    *,
    expected_excluded_physical_plan_sha256: Sequence[str] | None = None,
) -> dict[str, Any]:
    value = descriptor if isinstance(descriptor, AdaptiveEventDescriptor) else AdaptiveEventDescriptor.from_mapping(descriptor)
    feasible = polytope if isinstance(polytope, FeasibleEventPolytope) else FeasibleEventPolytope.from_mapping(polytope)
    authenticate_adaptive_event_descriptor(value, center)
    payload = copy.deepcopy(dict(job))
    config = payload.pop("config", None)
    if not isinstance(config, Mapping):
        raise RuntimeError("adaptive job lost its resolved config")
    recorded_sha = payload.pop("candidate_payload_sha256", None)
    if recorded_sha != canonical_sha256(payload):
        raise RuntimeError("adaptive job payload SHA-256 changed")
    if (
        payload.get("kind") != "v14_adaptive_contact_event_candidate"
        or payload.get("full_reset_required") is not True
        or payload.get("source_center_id") != center.center_id
        or payload.get("descriptor_id") != value.descriptor_id
        or payload.get("polytope_id") != feasible.polytope_id
    ):
        raise RuntimeError("adaptive job provenance changed")
    if payload.get("source_candidate_id") != center.candidate_id or payload.get(
        "source_physical_plan_sha256"
    ) != center.physical_plan_sha256:
        raise RuntimeError("adaptive job source physical provenance changed")
    budget_raw = payload.get("budget")
    if not isinstance(budget_raw, Mapping):
        raise RuntimeError("adaptive job lost its budget evidence")
    budget = AdaptiveEventBudget.from_mapping(budget_raw)
    if budget.stage != payload.get("stage"):
        raise RuntimeError("adaptive job stage and budget disagree")
    expected_sampling_seed = _adaptive_sampling_seed(center, value, feasible, budget)
    if _integer(payload.get("sampling_seed"), -1) != expected_sampling_seed:
        raise RuntimeError("adaptive job sampling seed is not reproducible")
    expected_pool_count = max(
        budget.candidates_per_center * budget.resampling_multiplier,
        budget.candidates_per_center + 32,
    )
    if _integer(payload.get("lhs_pool_count"), -1) != expected_pool_count:
        raise RuntimeError("adaptive job LHS pool evidence changed")
    row_index = _integer(payload.get("lhs_pool_row_index"), -1)
    if not 0 <= row_index < expected_pool_count:
        raise RuntimeError("adaptive job LHS row evidence is invalid")
    centers_raw = payload.get("refinement_centers")
    if not isinstance(centers_raw, list) or payload.get(
        "refinement_centers_sha256"
    ) != canonical_sha256(centers_raw):
        raise RuntimeError("adaptive job refinement-center evidence changed")
    refinement_vectors = tuple(feasible.parameter_vector(item) for item in centers_raw)
    if budget.stage == "local_refinement" and not refinement_vectors:
        raise RuntimeError("adaptive refinement job lost its center")
    if budget.stage != "local_refinement" and refinement_vectors:
        raise RuntimeError("adaptive non-refinement job gained refinement centers")
    lhs_row = _latin_hypercube(
        expected_pool_count, feasible.dimension, expected_sampling_seed
    )[row_index]
    lower = np.asarray(feasible.lower_bounds)
    upper = np.asarray(feasible.upper_bounds)
    if budget.stage == "local_refinement":
        anchor = refinement_vectors[row_index % len(refinement_vectors)]
        expected_requested = anchor + budget.refinement_radius_fraction * 0.5 * (
            upper - lower
        ) * lhs_row
        expected_requested = np.clip(expected_requested, lower, upper)
    else:
        expected_requested = _normalized_to_physical(lhs_row, feasible)
        if budget.stage == "diagnostic":
            midpoint = 0.5 * (lower + upper)
            expected_requested = midpoint + 0.35 * (expected_requested - midpoint)
    projected = feasible.parameter_vector(payload.get("projected_parameters", {}))
    if not feasible.contains(projected, tolerance=1e-8):
        raise RuntimeError("adaptive job projection left its feasible polytope")
    requested = feasible.parameter_vector(payload.get("requested_parameters", {}))
    if not np.allclose(expected_requested, requested, rtol=0.0, atol=1e-14):
        raise RuntimeError("adaptive job request is not reproducible from its budget")
    expected_projection = feasible.project(requested)
    if not np.allclose(expected_projection, projected, rtol=0.0, atol=1e-10):
        raise RuntimeError("adaptive job projected parameters are not reproducible")
    expected_normalized = 2.0 * (requested - lower) / np.maximum(upper - lower, _EPS) - 1.0
    recorded_normalized = feasible.parameter_vector(payload.get("requested_normalized_parameters", {}))
    if not np.allclose(expected_normalized, recorded_normalized, rtol=0.0, atol=1e-12):
        raise RuntimeError("adaptive job normalized request evidence changed")
    expected_distance = float(np.linalg.norm(projected - requested))
    if not math.isclose(
        _finite(payload.get("projection_distance"), math.inf),
        expected_distance,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("adaptive job projection-distance evidence changed")
    projected_mapping = feasible.parameter_mapping(projected)
    expected_candidate_id = _candidate_id_from_projection(
        center_id=center.center_id,
        stage=str(payload["stage"]),
        local_index=int(payload["local_index"]),
        projected=projected_mapping,
    )
    if int(payload.get("candidate_id", -1)) != expected_candidate_id:
        raise RuntimeError("adaptive job candidate ID is not reproducible")
    exclusion_sha = payload.get("excluded_physical_plan_set_sha256")
    exclusion_count = _integer(payload.get("excluded_physical_plan_count"), -1)
    if not _is_sha(exclusion_sha) or exclusion_count < 0:
        raise RuntimeError("adaptive job exclusion-set provenance is invalid")
    if expected_excluded_physical_plan_sha256 is not None:
        excluded = tuple(
            sorted(set(str(item) for item in expected_excluded_physical_plan_sha256))
        )
        if any(not _is_sha(item) for item in excluded):
            raise ValueError("expected adaptive exclusion set is invalid")
        if canonical_sha256(list(excluded)) != exclusion_sha or len(excluded) != exclusion_count:
            raise RuntimeError("adaptive job exclusion-set provenance changed")
    expected = _adaptive_plan_config(
        center.config,
        value,
        feasible,
        projected,
        stage=str(payload["stage"]),
        candidate_id=int(payload["candidate_id"]),
        local_index=int(payload["local_index"]),
    )
    if canonical_sha256(expected) != canonical_sha256(config):
        raise RuntimeError("adaptive job config is not reproducible from its projection")
    if payload.get("config_semantic_sha256") != canonical_sha256(config):
        raise RuntimeError("adaptive job config semantic SHA-256 changed")
    physical = physical_plan_sha256(config)
    if physical == center.physical_plan_sha256 or payload.get("physical_plan_sha256") != physical:
        raise RuntimeError("adaptive job is a parent duplicate or changed physically")
    if expected_excluded_physical_plan_sha256 is not None and physical in set(
        str(item) for item in expected_excluded_physical_plan_sha256
    ):
        raise RuntimeError("adaptive job repeats an excluded physical plan")
    return copy.deepcopy(dict(job))


def _candidate_summary(record: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = record.get("summary")
    return summary if isinstance(summary, Mapping) else record


def _metric(summary: Mapping[str, Any], *path: str, fallback: float) -> float:
    value: Any = summary
    for name in path:
        if not isinstance(value, Mapping):
            return fallback
        value = value.get(name)
    return _finite(value, fallback)


def adaptive_event_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Group-aware v14 ordering; jerk leads only the jerk-only rescue group."""

    summary = _candidate_summary(record)
    failed_raw = summary.get("failed_checks")
    failed = (
        tuple(str(value) for value in failed_raw)
        if isinstance(failed_raw, Sequence) and not isinstance(failed_raw, (str, bytes))
        else ("malformed_failed_checks",)
    )
    nonjerk = tuple(sorted(value for value in failed if value != "smooth_motion_jerk_within_limit"))
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), Mapping) else {}
    smooth = metrics.get("motion_smoothness") if isinstance(metrics, Mapping) and isinstance(metrics.get("motion_smoothness"), Mapping) else {}
    planned = metrics.get("contact_preserving_planned_lift") if isinstance(metrics, Mapping) and isinstance(metrics.get("contact_preserving_planned_lift"), Mapping) else {}
    duties = metrics.get("operation_target_face_contact_duty") if isinstance(metrics, Mapping) and isinstance(metrics.get("operation_target_face_contact_duty"), Mapping) else {}
    min_duty = min((_finite(duties.get(name), -math.inf) for name in _FINGERS), default=-math.inf)
    losses = planned.get("longest_contact_loss_steps") if isinstance(planned.get("longest_contact_loss_steps"), Mapping) else {}
    longest_loss = max((_finite(losses.get(name), math.inf) for name in _FINGERS), default=math.inf)
    full_success = bool(record.get("full_success", summary.get("passed", False))) and not failed
    jerk_only = not nonjerk and failed == ("smooth_motion_jerk_within_limit",)
    group = 0 if full_success else (1 if jerk_only else 2)
    perturbation_passes = _integer(
        record.get(
            "perturbation_pass_count",
            metrics.get("perturbation_pass_count") if isinstance(metrics, Mapping) else None,
        ),
        -1,
    )
    minimum_margin = _finite(
        record.get(
            "minimum_normalized_acceptance_margin",
            metrics.get("minimum_normalized_acceptance_margin")
            if isinstance(metrics, Mapping)
            else None,
        ),
        -math.inf,
    )
    jerk = _finite(smooth.get("operation_peak_abs_filtered_jerk_m_s3"), math.inf)
    if group == 0:
        # Preserve the production v14 success policy: perturbation evidence
        # and hard-margin/contact robustness precede smoothness tie-breakers.
        group_prefix: tuple[Any, ...] = (
            group,
            -perturbation_passes,
            -minimum_margin,
            -min_duty,
            longest_loss,
            jerk,
        )
    elif group == 1:
        # Once every non-jerk check passes, this stage exists specifically to
        # minimize the remaining jerk violation.
        group_prefix = (group, 0, 0.0, 0.0, 0.0, jerk)
    else:
        group_prefix = (group, len(nonjerk), 0.0, -min_duty, longest_loss, jerk)
    return (
        *group_prefix,
        -_finite(metrics.get("operation_minimum_lift_m") if isinstance(metrics, Mapping) else None, -math.inf),
        -_finite(metrics.get("operation_median_lift_m") if isinstance(metrics, Mapping) else None, -math.inf),
        _finite(smooth.get("operation_max_lateral_displacement_m"), math.inf),
        _finite(smooth.get("operation_max_orientation_drift_deg"), math.inf),
        _finite(smooth.get("operation_cumulative_height_backtrack_m"), math.inf),
        _finite(smooth.get("operation_peak_filtered_upward_speed_m_s"), math.inf),
        _finite(smooth.get("operation_peak_abs_filtered_acceleration_m_s2"), math.inf),
        _finite(smooth.get("operation_hold_entry_linear_speed_m_s"), math.inf),
        _finite(metrics.get("actuator_saturation_fraction") if isinstance(metrics, Mapping) else None, math.inf),
        _finite(record.get("projection_distance"), math.inf),
        _integer(record.get("candidate_id"), 1 << 62),
    )
