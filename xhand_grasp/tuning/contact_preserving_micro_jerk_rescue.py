"""Bounded trace-local micro refinement for schema-v14 planned lifts.

The completed force-debias campaign showed three repeatable jerk bands while
all other hard constraints were already satisfied.  This numerical module
keeps the 3 s, 21-knot representation (20 moving intervals) and adds only
compact C2 actuator-space corrections around those observed bands.  It does
not run physics or write artifacts.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..actual_contact_grasp_pose_catalog import validate_stage_ledger
from ..artifacts import file_sha256
from ..config import ACTIVE_ACTUATORS, contact_preload_targets, precontact_targets, validate_config
from ..experiment import ContactFeedbackParameters, ManipulationPlanParameters
from ..grasp_pose import canonical_sha256
from .contact_constrained_planner import (
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
)
from .contact_preserving_candidate_artifacts import authenticate_v14_candidate_artifacts
from .contact_preserving_force_debias_rescue import (
    force_debias_candidate_rank,
    physical_plan_sha256,
)
from .contact_preserving_time_warp import _time_warp_controller_id


MICRO_JERK_SCHEMA_VERSION = 1
EXPERIMENT_ID = "left_opposed_face_palm_down_contact_preserving_planned_lift"
DEFAULT_SEED = 20260821
MICRO_CENTER_COUNT = 8
SENSITIVITY_COUNT_PER_CENTER = 32
TRUST_COUNT_PER_RADIUS = 32
TRUST_RADII = (0.015, 0.03, 0.06)
TRUST_COUNT_PER_CENTER = TRUST_COUNT_PER_RADIUS * len(TRUST_RADII)
MICRO_COUNT_PER_CENTER = SENSITIVITY_COUNT_PER_CENTER + TRUST_COUNT_PER_CENTER
MICRO_CANDIDATE_COUNT = MICRO_CENTER_COUNT * MICRO_COUNT_PER_CENTER
LOCAL_UNLOAD_BOUNDS_RAD = {"early": 0.0002, "main": 0.0002, "terminal": 0.0002}
BAND_CENTER_SHIFT_BOUNDS = {"early": 0.010, "main": 0.015, "terminal": 0.006}
BAND_WIDTH_SCALE_DELTA = 0.15
THUMB_PRELOAD_NEGATIVE_RAD = 0.002
THUMB_PRELOAD_POSITIVE_RAD = 0.004
RATE_SCALE_DELTA = 0.15
ACCELERATION_SCALE_DELTA = 0.25
_FINGERS = ("thumb", "index", "mid")
_FINGER_SLICES = {
    "thumb": slice(0, 3),
    "index": slice(3, 6),
    "mid": slice(6, 8),
}
_BANDS = ("early", "main", "terminal")
_PARAMETER_NAMES = tuple(
    f"band:{band}:{finger}" for band in _BANDS for finger in _FINGERS
) + (
    *(f"band:{band}:center_shift" for band in _BANDS),
    *(f"band:{band}:width_scale_delta" for band in _BANDS),
    "thumb:inward_preload",
    "feedback:rate_scale",
    "feedback:acceleration_scale",
    "timing:retime_scale",
)
_JERK_CHECK = "smooth_motion_jerk_within_limit"
_RETIME_ANCHOR_S = np.asarray(
    [
        0.0, 0.1234398746078118, 0.24687975300121565,
        0.37031979498742784, 0.4937640543463782, 0.617272892170285,
        0.7413656650487401, 0.8685632351820698, 1.0485632351820697,
        1.2012883268133148, 1.3619829985737817, 1.5131997809803683,
        1.6544533377942032, 1.7997752981438913, 2.0397752981438915,
        2.2797752981438917, 2.421827918201138, 2.5516084680559055,
        2.676387196159435, 2.8000000000000007, 3.0,
    ],
    dtype=np.float64,
)


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON evidence is not an object: {path}")
    return payload


def _minimum_jerk(value: np.ndarray) -> np.ndarray:
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def compact_c2_band(
    progress: Sequence[float] | np.ndarray,
    *,
    start: float,
    peak: float,
    end: float,
) -> np.ndarray:
    """Asymmetric compact bump with C2 joins at start, peak and end."""

    values = np.asarray(progress, dtype=np.float64)
    if not np.isfinite(values).all() or not 0.0 <= start < peak < end <= 1.0:
        raise ValueError("micro C2 band bounds are invalid")
    result = np.zeros_like(values)
    rising = (values > start) & (values < peak)
    result[rising] = _minimum_jerk((values[rising] - start) / (peak - start))
    result[values == peak] = 1.0
    falling = (values > peak) & (values < end)
    result[falling] = 1.0 - _minimum_jerk((values[falling] - peak) / (end - peak))
    return result


def _band_envelopes(
    progress: np.ndarray, *, parameters: Mapping[str, float]
) -> dict[str, np.ndarray]:
    early_center = 0.357 + float(parameters["band:early:center_shift"])
    early_scale = float(parameters["band:early:width_scale"])
    main_center = 0.708 + float(parameters["band:main:center_shift"])
    main_scale = float(parameters["band:main:width_scale"])
    # The measured terminal jerk is near progress .981, but the final movable
    # waypoint is knot 19 near .933.  Centering the command-space bump there
    # lets the fixed 21-knot plan influence that filtered terminal event while
    # still returning exactly to zero at progress one.
    terminal_peak = 0.933 + float(parameters["band:terminal:center_shift"])
    terminal_scale = float(parameters["band:terminal:width_scale"])
    return {
        "early": compact_c2_band(
            progress,
            start=early_center - 0.057 * early_scale,
            peak=early_center,
            end=early_center + 0.063 * early_scale,
        ),
        "main": compact_c2_band(
            progress,
            start=main_center - 0.10 * main_scale,
            peak=main_center,
            end=main_center + 0.10 * main_scale,
        ),
        "terminal": compact_c2_band(
            progress,
            start=terminal_peak - 0.073 * terminal_scale,
            peak=terminal_peak,
            end=1.0,
        ),
    }


def _thumb_preload_envelope(progress: np.ndarray) -> np.ndarray:
    result = np.zeros_like(progress)
    rising = (progress > 0.15) & (progress < 0.30)
    result[rising] = _minimum_jerk((progress[rising] - 0.15) / 0.15)
    result[(progress >= 0.30) & (progress <= 0.90)] = 1.0
    falling = (progress > 0.90) & (progress < 1.0)
    result[falling] = 1.0 - _minimum_jerk((progress[falling] - 0.90) / 0.10)
    return result


@dataclass(frozen=True, slots=True)
class MicroJerkCenter:
    candidate_id: int
    center_kind: str
    source_authentication_id: str
    config: dict[str, Any]
    result: dict[str, Any]
    summary: dict[str, Any]
    config_path: Path
    result_path: Path
    physical_plan_sha256: str
    source_center_id: str
    source_candidate_id: int
    event_half_width_progress: float
    source_record_sha256: str
    center_id: str

    def as_mapping(self, *, include_payloads: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": MICRO_JERK_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "center_kind": self.center_kind,
            "source_authentication_id": self.source_authentication_id,
            "config_path": str(self.config_path),
            "result_path": str(self.result_path),
            "physical_plan_sha256": self.physical_plan_sha256,
            "source_center_id": self.source_center_id,
            "source_candidate_id": self.source_candidate_id,
            "event_half_width_progress": self.event_half_width_progress,
            "source_record_sha256": self.source_record_sha256,
            "center_id": self.center_id,
        }
        if include_payloads:
            payload.update(config=copy.deepcopy(self.config), result=copy.deepcopy(self.result), summary=copy.deepcopy(self.summary))
        return payload


@dataclass(frozen=True, slots=True)
class MicroJerkSource:
    root: Path
    source_authentication_id: str
    manifest_path: Path
    ledger_path: Path
    discovery_report_path: Path
    refinement_report_path: Path
    result_path: Path
    catalog_path: Path
    artifact_paths: tuple[Path, ...]
    centers: tuple[MicroJerkCenter, ...]
    prior_physical_plan_sha256: tuple[str, ...]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": MICRO_JERK_SCHEMA_VERSION,
            "root": str(self.root),
            "source_authentication_id": self.source_authentication_id,
            "manifest_path": str(self.manifest_path),
            "ledger_path": str(self.ledger_path),
            "discovery_report_path": str(self.discovery_report_path),
            "refinement_report_path": str(self.refinement_report_path),
            "result_path": str(self.result_path),
            "catalog_path": str(self.catalog_path),
            "artifact_paths": [str(path) for path in self.artifact_paths],
            "centers": [center.as_mapping() for center in self.centers],
            "prior_physical_plan_sha256": list(self.prior_physical_plan_sha256),
        }


def _peak_jerk(record: Mapping[str, Any]) -> float:
    try:
        value = record["summary"]["metrics"]["motion_smoothness"]["operation_peak_abs_filtered_jerk_m_s3"]
        return float(value) if math.isfinite(float(value)) else math.inf
    except (KeyError, TypeError, ValueError):
        return math.inf


def _failed_checks(record: Mapping[str, Any]) -> tuple[str, ...]:
    raw = record.get("summary", {}).get("failed_checks", ())
    return tuple(str(value) for value in raw) if isinstance(raw, list) else ("malformed",)


def _strict_jerk_only(record: Mapping[str, Any]) -> bool:
    return _failed_checks(record) == (_JERK_CHECK,)


def _safe_contact_boundary(record: Mapping[str, Any]) -> bool:
    failed = _failed_checks(record)
    if _JERK_CHECK not in failed or len(failed) <= 1:
        return False
    metrics = record.get("summary", {}).get("metrics", {})
    planned = metrics.get("contact_preserving_planned_lift", {}) if isinstance(metrics, Mapping) else {}
    losses = planned.get("longest_contact_loss_steps", {}) if isinstance(planned, Mapping) else {}
    return (
        planned.get("final_plan_progress") == 1.0
        and max((int(value) for value in losses.values()), default=999) <= 10
        and not bool(metrics.get("controller_aborted", False))
        and not any("forbidden" in value or "penetration" in value for value in failed)
    )


def _select_center_records(records: Sequence[Mapping[str, Any]]) -> tuple[tuple[dict[str, Any], str], ...]:
    strict = sorted((copy.deepcopy(dict(value)) for value in records if _strict_jerk_only(value)), key=force_debias_candidate_rank)
    if len(strict) < 6:
        raise RuntimeError("micro source has fewer than six jerk-only candidates")
    selected: list[tuple[dict[str, Any], str]] = []
    physical: set[str] = set()

    def add(record: Mapping[str, Any], kind: str) -> bool:
        metadata = record.get("rescue_job")
        digest = metadata.get("physical_plan_sha256") if isinstance(metadata, Mapping) else None
        if not _is_sha(digest) or digest in physical:
            return False
        physical.add(str(digest))
        selected.append((copy.deepcopy(dict(record)), kind))
        return True

    # Mandatory diversity: both principal source centers and widths 0.18/0.22.
    best = strict[0]
    add(best, "jerk_only")
    best_source = int(best["rescue_job"]["source_candidate_id"])
    other_sources = sorted({int(value["rescue_job"]["source_candidate_id"]) for value in strict if int(value["rescue_job"]["source_candidate_id"]) != best_source})
    if not other_sources:
        raise RuntimeError("micro source does not cover two source centers")
    for predicate in (
        lambda value: int(value["rescue_job"]["source_candidate_id"]) == other_sources[0] and float(value["rescue_job"]["event_half_width_progress"]) == 0.18,
        lambda value: int(value["rescue_job"]["source_candidate_id"]) == other_sources[0] and float(value["rescue_job"]["event_half_width_progress"]) == 0.22,
        lambda value: int(value["rescue_job"]["source_candidate_id"]) == best_source and float(value["rescue_job"]["event_half_width_progress"]) == 0.18,
    ):
        match = next((value for value in strict if predicate(value)), None)
        if match is None or not add(match, "jerk_only"):
            raise RuntimeError("micro source lost required source/width diversity")
    for value in strict:
        if len(selected) >= 6:
            break
        add(value, "jerk_only")
    if len(selected) != 6:
        raise RuntimeError("micro source could not select six unique jerk-only centers")
    boundaries = sorted((copy.deepcopy(dict(value)) for value in records if _safe_contact_boundary(value)), key=lambda value: (_peak_jerk(value), len(_failed_checks(value)), int(value["candidate_id"])))
    for value in boundaries:
        if len(selected) >= MICRO_CENTER_COUNT:
            break
        add(value, "contact_boundary")
    if len(selected) != MICRO_CENTER_COUNT:
        raise RuntimeError("micro source could not select two safe contact-boundary centers")
    return tuple(selected)


def authenticate_micro_jerk_source(source_campaign: str | Path) -> MicroJerkSource:
    root = Path(source_campaign).expanduser().resolve()
    manifest_path = root / "campaign_manifest.json"
    ledger_path = root / "stage_ledger.json"
    discovery_path = root / "force_debias_discovery/report.json"
    refinement_path = root / "force_debias_refinement/report.json"
    result_path = root / "force_debias_rescue_result_target_1.json"
    catalog_path = root / "catalogs/target_1/manipulation/catalog.json"
    required = (manifest_path, ledger_path, discovery_path, refinement_path, result_path, catalog_path)
    if not root.is_dir() or not all(path.is_file() for path in required):
        raise RuntimeError("micro source lost completed force-debias evidence")
    ledger = validate_stage_ledger(root)
    expected = {"force_debias_source_audit", "force_debias_discovery", "force_debias_refinement", "force_debias_catalog_target_1", "force_debias_result_target_1"}
    if not expected.issubset(ledger.get("stages", {})):
        raise RuntimeError("micro source force-debias campaign is incomplete")
    manifest = _load_json(manifest_path)
    result = _load_json(result_path)
    if manifest.get("campaign_kind") != "contact_preserving_force_debias_rescue" or result.get("complete") is not True or int(result.get("full_success_count", -1)) != 0:
        raise RuntimeError("micro source is not the sealed zero-success force-debias campaign")
    reports = (_load_json(discovery_path), _load_json(refinement_path))
    expected_counts = (160, 512)
    records: list[dict[str, Any]] = []
    for report, count in zip(reports, expected_counts, strict=True):
        raw = report.get("records")
        if report.get("complete") is not True or not isinstance(raw, list) or len(raw) != count or int(report.get("candidate_count", -1)) != count:
            raise RuntimeError("micro source phase report did not exhaust its budget")
        records.extend(copy.deepcopy(dict(value)) for value in raw)
    selected = list(_select_center_records(records))
    prior_physical_set = {
        str(value["rescue_job"]["physical_plan_sha256"])
        for value in records
        if isinstance(value.get("rescue_job"), Mapping)
        and _is_sha(value["rescue_job"].get("physical_plan_sha256"))
    }
    if len(prior_physical_set) != 672:
        raise RuntimeError("micro source lost the 672 unique prior physical plans")

    # This independently rerun trace was committed after force-debias v2 was
    # sealed.  It is deliberately authenticated as a supplemental source,
    # rather than being made to look like a member of the immutable v2 ledger.
    # Requiring the exact relative location also makes a missing or partially
    # copied design-evidence bundle fail closed.
    supplemental_root = (
        root.parent
        / "micro_jerk_design_evidence_v1"
        / "candidate_best_round2"
    ).resolve()
    supplemental_bundle = authenticate_v14_candidate_artifacts(
        supplemental_root,
        expected_candidate_id=15043844903285998,
        require_retained_trace=True,
    )
    supplemental_config = _load_json(supplemental_bundle.config_path)
    supplemental_summary = copy.deepcopy(supplemental_bundle.result.get("summary"))
    if (
        not isinstance(supplemental_summary, Mapping)
        or tuple(supplemental_summary.get("failed_checks", ())) != (_JERK_CHECK,)
        or bool(supplemental_bundle.result.get("full_success", False))
        or abs(_peak_jerk({"summary": supplemental_summary}) - 3.0540186473498876)
        > 1e-12
    ):
        raise RuntimeError("supplemental round2 micro-jerk evidence changed")
    supplemental_physical = physical_plan_sha256(supplemental_config)
    if supplemental_physical in prior_physical_set:
        raise RuntimeError("supplemental round2 evidence duplicates force-debias v2")
    supplemental_center_seed = canonical_sha256(
        {
            "kind": "authenticated_micro_jerk_design_evidence_round2",
            "candidate_id": supplemental_bundle.candidate_id,
            "config_semantic_sha256": canonical_sha256(supplemental_config),
            "result_semantic_sha256": canonical_sha256(supplemental_bundle.result),
            "trace_sha256": file_sha256(supplemental_bundle.trace_path),
        }
    )
    supplemental_record = {
        "candidate_id": supplemental_bundle.candidate_id,
        "summary": supplemental_summary,
        "config_semantic_sha256": canonical_sha256(supplemental_config),
        "rescue_job": {
            "physical_plan_sha256": supplemental_physical,
            "source_center_id": supplemental_center_seed,
            "source_candidate_id": supplemental_bundle.candidate_id,
            "event_half_width_progress": 0.20,
        },
        "supplemental_evidence_root": str(supplemental_root),
    }
    # The strongest authenticated jerk-only trace becomes the primary center;
    # five v2 jerk-only centers retain source/width diversity, and the two
    # contact-boundary centers remain unchanged.
    selected[0] = (supplemental_record, "jerk_only")
    prior_physical_set.add(supplemental_physical)
    prior_physical = tuple(sorted(prior_physical_set))
    artifact_paths: list[Path] = list(required)
    provisional: list[tuple[dict[str, Any], str, Any]] = []
    for record, kind in selected:
        if record.get("supplemental_evidence_root") is not None:
            destination = supplemental_root
            bundle = authenticate_v14_candidate_artifacts(
                destination,
                expected_candidate_id=int(record["candidate_id"]),
                require_retained_trace=True,
            )
        else:
            relative = record.get("artifact_directory")
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise RuntimeError("micro source candidate path is unsafe")
            destination = (root / relative).resolve()
            if not destination.is_relative_to(root):
                raise RuntimeError("micro source candidate escaped campaign")
            bundle = authenticate_v14_candidate_artifacts(destination, expected_candidate_id=int(record["candidate_id"]), expected_retain_grasp_success=False)
        config = _load_json(bundle.config_path)
        if canonical_sha256(config) != str(record.get("config_semantic_sha256")):
            raise RuntimeError("micro source candidate config changed")
        if canonical_sha256(bundle.result.get("summary")) != canonical_sha256(record.get("summary")):
            raise RuntimeError("micro source candidate summary changed")
        artifact_paths.extend(bundle.artifact_paths)
        provisional.append((record, kind, bundle))
    immutable = tuple(dict.fromkeys(path.resolve() for path in artifact_paths))
    auth_id = canonical_sha256({
        "schema_version": MICRO_JERK_SCHEMA_VERSION,
        "root": str(root),
        "artifact_sha256": {str(path): file_sha256(path) for path in immutable},
        "selected": [(int(record["candidate_id"]), kind) for record, kind, _ in provisional],
    })
    centers: list[MicroJerkCenter] = []
    for record, kind, bundle in provisional:
        metadata = record["rescue_job"]
        payload = {
            "candidate_id": int(record["candidate_id"]),
            "center_kind": kind,
            "source_authentication_id": auth_id,
            "physical_plan_sha256": metadata["physical_plan_sha256"],
            "source_record_sha256": canonical_sha256(record),
        }
        centers.append(MicroJerkCenter(
            candidate_id=int(record["candidate_id"]),
            center_kind=kind,
            source_authentication_id=auth_id,
            config=_load_json(bundle.config_path),
            result=copy.deepcopy(bundle.result),
            summary=copy.deepcopy(dict(record["summary"])),
            config_path=bundle.config_path,
            result_path=bundle.result_path,
            physical_plan_sha256=str(metadata["physical_plan_sha256"]),
            source_center_id=str(metadata["source_center_id"]),
            source_candidate_id=int(metadata["source_candidate_id"]),
            event_half_width_progress=float(metadata["event_half_width_progress"]),
            source_record_sha256=canonical_sha256(record),
            center_id=canonical_sha256(payload),
        ))
    return MicroJerkSource(root=root, source_authentication_id=auth_id, manifest_path=manifest_path, ledger_path=ledger_path, discovery_report_path=discovery_path, refinement_report_path=refinement_path, result_path=result_path, catalog_path=catalog_path, artifact_paths=immutable, centers=tuple(centers), prior_physical_plan_sha256=prior_physical)


def _inward_directions(config: Mapping[str, Any]) -> np.ndarray:
    pre = precontact_targets(copy.deepcopy(dict(config)))
    preload = contact_preload_targets(copy.deepcopy(dict(config)))
    vector = np.asarray([float(preload[name]) - float(pre[name]) for name in ACTIVE_ACTUATORS])
    result = np.zeros((3, len(ACTIVE_ACTUATORS)))
    for index, finger in enumerate(_FINGERS):
        owned = _FINGER_SLICES[finger]
        local = vector[owned]
        maximum = float(np.max(np.abs(local), initial=0.0))
        if maximum <= 1e-12:
            raise ValueError(f"micro {finger} inward ray is degenerate")
        result[index, owned] = local / maximum
    return result


def _parameter_mapping(normalized: Sequence[float]) -> dict[str, float]:
    values = np.asarray(normalized, dtype=np.float64)
    if values.shape != (len(_PARAMETER_NAMES),) or not np.isfinite(values).all() or np.max(np.abs(values), initial=0.0) > 1.0 + 1e-12:
        raise ValueError("micro normalized parameters are invalid")
    result: dict[str, float] = {}
    for index, band in enumerate(_BANDS):
        for finger_index, finger in enumerate(_FINGERS):
            bound = LOCAL_UNLOAD_BOUNDS_RAD[band]
            if band == "terminal" and finger == "mid":
                bound *= 0.5
            result[f"band:{band}:{finger}"] = float(values[index * 3 + finger_index] * bound)
    for index, band in enumerate(_BANDS):
        result[f"band:{band}:center_shift"] = float(values[9 + index] * BAND_CENTER_SHIFT_BOUNDS[band])
        result[f"band:{band}:width_scale"] = float(1.0 + values[12 + index] * BAND_WIDTH_SCALE_DELTA)
    result["thumb:inward_preload"] = float(values[15] * (THUMB_PRELOAD_POSITIVE_RAD if values[15] >= 0.0 else THUMB_PRELOAD_NEGATIVE_RAD))
    result["feedback:rate_scale"] = float(1.0 + values[16] * RATE_SCALE_DELTA)
    result["feedback:acceleration_scale"] = float(1.0 + values[17] * ACCELERATION_SCALE_DELTA)
    result["timing:retime_scale"] = float(1.0 + (0.2 * values[18] if values[18] >= 0.0 else values[18]))
    return result


def _feedback_config(parent: Mapping[str, Any], parameters: Mapping[str, float]) -> dict[str, Any]:
    original = ContactFeedbackParameters.from_config(parent)
    value = ContactFeedbackParameters(
        schema_version=original.schema_version,
        strategy=original.strategy,
        filter_time_constant_s=original.filter_time_constant_s,
        kp_rad_per_n=original.kp_rad_per_n,
        ki_rad_per_n_s=original.ki_rad_per_n_s,
        integral_limit_n_s=original.integral_limit_n_s,
        correction_limit_rad=original.correction_limit_rad,
        rate_limit_rad_s=original.rate_limit_rad_s * float(parameters["feedback:rate_scale"]),
        acceleration_limit_rad_s2=original.acceleration_limit_rad_s2 * float(parameters["feedback:acceleration_scale"]),
        force_risk_n=original.force_risk_n,
        freeze_on_risk=original.freeze_on_risk,
        max_loss_s=original.max_loss_s,
        recovery_behavior=original.recovery_behavior,
        operation_contact_duty_min=original.operation_contact_duty_min,
        tangent_slip_freeze_threshold_m=(
            original.tangent_slip_freeze_threshold_m
        ),
        tangent_slip_abort_threshold_m=(
            original.tangent_slip_abort_threshold_m
        ),
    )
    return value.as_config()


def _project_adjacent_waypoints(
    waypoints: np.ndarray, *, maximum_delta_rad: float
) -> np.ndarray:
    """Project deterministically onto adjacent-knot actuator-rate bounds."""

    projected = np.asarray(waypoints, dtype=np.float64).copy()
    projected[0] = 0.0
    limit = float(maximum_delta_rad)
    for knot in range(1, projected.shape[0]):
        delta = np.clip(projected[knot] - projected[knot - 1], -limit, limit)
        projected[knot] = projected[knot - 1] + delta
    return projected


def _candidate_config(center: MicroJerkCenter, normalized: Sequence[float], *, candidate_id: int, local_index: int, sampling_mode: str, trust_radius: float | None) -> dict[str, Any]:
    parameters = _parameter_mapping(normalized)
    config = copy.deepcopy(center.config)
    original = ManipulationPlanParameters.from_config(config["manipulation_plan"])
    base_times = np.asarray(original.knot_times_s, dtype=np.float64)
    if base_times.shape != _RETIME_ANCHOR_S.shape or float(original.duration_s) != 3.0:
        raise ValueError("micro retime requires the registered 3 s/21-knot plan")
    retime_scale = float(parameters["timing:retime_scale"])
    times = base_times + retime_scale * (_RETIME_ANCHOR_S - base_times)
    if not np.all(np.diff(times) > 0.0) or times[0] != 0.0 or abs(times[-1] - 3.0) > 1e-12:
        raise ValueError("micro retime produced invalid knot times")
    progress = times / float(original.duration_s)
    waypoints = np.column_stack([original.actuator_waypoints_rad[name] for name in ACTIVE_ACTUATORS]).astype(np.float64)
    directions = _inward_directions(config)
    envelopes = _band_envelopes(progress, parameters=parameters)
    for band_index, band in enumerate(_BANDS):
        for finger_index, finger in enumerate(_FINGERS):
            amplitude = parameters[f"band:{band}:{finger}"]
            waypoints += amplitude * envelopes[band][:, None] * directions[finger_index][None, :]
    waypoints += parameters["thumb:inward_preload"] * _thumb_preload_envelope(progress)[:, None] * directions[0][None, :]
    waypoints = _project_adjacent_waypoints(
        waypoints, maximum_delta_rad=original.max_knot_delta_rad
    )
    plan = ManipulationPlanParameters(
        schema_version=original.schema_version,
        profile=original.profile,
        duration_s=original.duration_s,
        knot_times_s=tuple(float(value) for value in times),
        actuator_waypoints_rad={name: tuple(float(value) for value in waypoints[:, index]) for index, name in enumerate(ACTIVE_ACTUATORS)},
        desired_cube_position_delta_m=original.desired_cube_position_delta_m,
        desired_cube_rotation_vector_rad=original.desired_cube_rotation_vector_rad,
        max_knot_delta_rad=original.max_knot_delta_rad,
        trust_region_backtracks=original.trust_region_backtracks,
    )
    config["manipulation_plan"] = plan.as_config()
    config["control"]["manipulation_delta_rad"] = {name: float(waypoints[-1, index]) for index, name in enumerate(ACTIVE_ACTUATORS)}
    config["contact_feedback"] = _feedback_config(config["contact_feedback"], parameters)
    config["object_config_id"] = v14_object_config_id(config)
    config["grasp_pose_id"] = v14_grasp_pose_id(config)
    config["grasp_object_pair_id"] = v14_grasp_object_pair_id(config)
    planner_payload = {
        "schema_version": MICRO_JERK_SCHEMA_VERSION,
        "kind": "v14_bounded_micro_jerk_plan",
        "source_center_id": center.center_id,
        "normalized_parameters": {name: float(value) for name, value in zip(_PARAMETER_NAMES, normalized, strict=True)},
        "physical_parameters": parameters,
    }
    config["planner_id"] = canonical_sha256(planner_payload)
    config["controller_id"] = _time_warp_controller_id(config)
    metadata = config.setdefault("candidate_metadata", {})
    metadata["v14_bounded_micro_jerk_rescue"] = {
        "schema_version": MICRO_JERK_SCHEMA_VERSION,
        "candidate_id": int(candidate_id),
        "local_index": int(local_index),
        "source_center_id": center.center_id,
        "sampling_mode": sampling_mode,
        "trust_radius": trust_radius,
        "normalized_parameters": planner_payload["normalized_parameters"],
        "physical_parameters": parameters,
        "branch_stability_soft_objective": True,
        "full_reset_required": True,
    }
    validate_config(config)
    return config


def _candidate_id(source_id: str, center_id: str, local_index: int, normalized: Sequence[float]) -> int:
    digest = canonical_sha256({"schema_version": MICRO_JERK_SCHEMA_VERSION, "source_authentication_id": source_id, "center_id": center_id, "local_index": int(local_index), "normalized_parameters": [float(value) for value in normalized]})
    return 15_100_000_000_000_000 + int(digest[:13], 16) % 100_000_000_000_000


def _project_candidate_config(
    center: MicroJerkCenter,
    requested: Sequence[float],
    *,
    local_index: int,
    sampling_mode: str,
    trust_radius: float | None,
) -> tuple[np.ndarray, int, dict[str, Any], int]:
    """Backtrack only waypoint-shape dimensions into the registered limits."""

    original = np.asarray(requested, dtype=np.float64)
    for backtrack, scale in enumerate(2.0 ** -np.arange(0, 18, dtype=np.float64)):
        projected = original.copy()
        projected[:16] *= scale
        candidate_id = _candidate_id(
            center.source_authentication_id, center.center_id, local_index, projected
        )
        try:
            config = _candidate_config(
                center,
                projected,
                candidate_id=candidate_id,
                local_index=local_index,
                sampling_mode=sampling_mode,
                trust_radius=trust_radius,
            )
        except ValueError as exc:
            if "adjacent manipulation actuator waypoints" not in str(exc):
                raise
            continue
        return projected, candidate_id, config, backtrack
    raise RuntimeError("micro waypoint projection exhausted its fixed backtracks")


def _lhs(count: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimension), dtype=np.float64)
    for column in range(dimension):
        result[:, column] = (rng.permutation(count) + rng.random(count)) / count
    return result


def _requests_for_center(center: MicroJerkCenter, seed: int) -> tuple[tuple[np.ndarray, str, float | None, str], ...]:
    dimension = len(_PARAMETER_NAMES)
    requests: list[tuple[np.ndarray, str, float | None, str]] = []
    # Nine event unload amplitudes receive direct +/- probes.
    for axis in range(9):
        for sign in (-1.0, 1.0):
            value = np.zeros(dimension)
            value[axis] = sign
            requests.append((value, "deterministic_sensitivity", None, f"axis:{_PARAMETER_NAMES[axis]}:{int(sign):+d}"))
    # A center/width change is physically meaningful only with a non-zero
    # event amplitude.  Pair each of the six shape probes with a small thumb
    # unload in the same event cluster; no other cluster is active.
    for axis in range(9, 15):
        cluster = axis - 9 if axis < 12 else axis - 12
        for sign in (-1.0, 1.0):
            value = np.zeros(dimension)
            value[axis] = sign
            value[cluster * 3] = 0.25
            requests.append((value, "deterministic_sensitivity", None, f"shape:{_PARAMETER_NAMES[axis]}:{int(sign):+d}"))
    # The last two deterministic probes bracket the thumb preload ray.
    for sign in (-1.0, 1.0):
        value = np.zeros(dimension)
        value[15] = sign
        requests.append((value, "deterministic_sensitivity", None, f"axis:{_PARAMETER_NAMES[15]}:{int(sign):+d}"))
    if len(requests) != SENSITIVITY_COUNT_PER_CENTER:
        raise RuntimeError("micro sensitivity schedule changed")
    for radius_index, radius in enumerate(TRUST_RADII):
        raw = 2.0 * _lhs(TRUST_COUNT_PER_RADIUS, dimension, seed + 1009 * (radius_index + 1)) - 1.0
        for index, value in enumerate(raw):
            # One trace event cluster per candidate.  Dense simultaneous
            # positive bumps were measured to amplify jerk by an order of
            # magnitude, so all other event amplitudes and shape variables
            # are identically zero.
            cluster = (radius_index * TRUST_COUNT_PER_RADIUS + index) % 3
            for other in range(3):
                if other == cluster:
                    continue
                value[other * 3 : other * 3 + 3] = 0.0
                value[9 + other] = 0.0
                value[12 + other] = 0.0
            norm = float(np.linalg.norm(value))
            if norm <= 0.0:
                raise RuntimeError("micro sparse trust direction is degenerate")
            value = value / norm * float(radius)
            requests.append((value, "trust_shell", float(radius), f"trust:{radius_index}:{index}"))
    if len(requests) != MICRO_COUNT_PER_CENTER:
        raise RuntimeError("micro per-center budget changed")
    return tuple(requests)


def build_micro_jerk_jobs(
    source: MicroJerkSource,
    *,
    total_count: int = MICRO_CANDIDATE_COUNT,
    seed: int = DEFAULT_SEED,
    local_index_offset: int = 0,
    excluded_physical_plan_sha256: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    if int(total_count) != MICRO_CANDIDATE_COUNT or int(seed) != DEFAULT_SEED or int(local_index_offset) < 0:
        raise ValueError("micro campaign requires its fixed 1024-candidate budget and seed")
    if len(source.centers) != MICRO_CENTER_COUNT:
        raise RuntimeError("micro source must contain exactly eight centers")
    excluded = tuple(
        sorted(
            {
                *source.prior_physical_plan_sha256,
                *(str(value) for value in excluded_physical_plan_sha256),
            }
        )
    )
    if any(not _is_sha(value) for value in excluded):
        raise ValueError("micro exclusion set contains an invalid SHA-256")
    seen_physical = {*(center.physical_plan_sha256 for center in source.centers), *excluded}
    seen_ids: set[int] = set()
    jobs: list[dict[str, Any]] = []
    for center_index, center in enumerate(source.centers):
        center_seed = int(canonical_sha256({"source": source.source_authentication_id, "center": center.center_id, "seed": seed})[:16], 16)
        for request, mode, radius, label in _requests_for_center(center, center_seed):
            local_index = int(local_index_offset) + len(jobs)
            projected, candidate_id, config, backtracks = _project_candidate_config(
                center,
                request,
                local_index=local_index,
                sampling_mode=mode,
                trust_radius=radius,
            )
            physical = physical_plan_sha256(config)
            if physical in seen_physical or candidate_id in seen_ids:
                raise RuntimeError("micro deterministic schedule produced a duplicate physical plan or ID")
            seen_physical.add(physical)
            seen_ids.add(candidate_id)
            core = {
                "schema_version": MICRO_JERK_SCHEMA_VERSION,
                "kind": "v14_bounded_micro_jerk_candidate",
                "stage": "micro_refinement",
                "candidate_id": candidate_id,
                "local_index": local_index,
                "center_sequence_index": center_index,
                "center_local_index": len(jobs) % MICRO_COUNT_PER_CENTER,
                "source_authentication_id": source.source_authentication_id,
                "source_candidate_id": center.candidate_id,
                "source_center_id": center.center_id,
                "source_physical_plan_sha256": center.physical_plan_sha256,
                "center_kind": center.center_kind,
                "sampling_mode": mode,
                "schedule_label": label,
                "trust_radius": radius,
                "requested_normalized_parameters": {name: float(value) for name, value in zip(_PARAMETER_NAMES, request, strict=True)},
                "normalized_parameters": {name: float(value) for name, value in zip(_PARAMETER_NAMES, projected, strict=True)},
                "projection_backtracks": backtracks,
                "physical_parameters": _parameter_mapping(projected),
                "excluded_physical_plan_set_sha256": canonical_sha256(list(excluded)),
                "excluded_physical_plan_count": len(excluded),
                "config_semantic_sha256": canonical_sha256(config),
                "physical_plan_sha256": physical,
                "full_reset_required": True,
            }
            jobs.append({**core, "candidate_payload_sha256": canonical_sha256(core), "config": config})
    if len(jobs) != MICRO_CANDIDATE_COUNT:
        raise RuntimeError("micro job builder did not exhaust its fixed budget")
    return tuple(jobs)


def authenticate_micro_jerk_job(
    job: Mapping[str, Any],
    source: MicroJerkSource,
    *,
    expected_excluded_physical_plan_sha256: Sequence[str] | None = None,
) -> dict[str, Any]:
    value = copy.deepcopy(dict(job))
    config = value.pop("config", None)
    recorded = value.pop("candidate_payload_sha256", None)
    if not isinstance(config, Mapping) or recorded != canonical_sha256(value):
        raise RuntimeError("micro job payload SHA-256 changed")
    if value.get("kind") != "v14_bounded_micro_jerk_candidate" or value.get("stage") != "micro_refinement" or value.get("source_authentication_id") != source.source_authentication_id or value.get("full_reset_required") is not True:
        raise RuntimeError("micro job provenance changed")
    centers = {center.center_id: center for center in source.centers}
    center = centers.get(str(value.get("source_center_id", "")))
    if center is None or int(value.get("source_candidate_id", -1)) != center.candidate_id or value.get("source_physical_plan_sha256") != center.physical_plan_sha256:
        raise RuntimeError("micro job source center changed")
    normalized = value.get("normalized_parameters")
    if not isinstance(normalized, Mapping) or tuple(normalized) != _PARAMETER_NAMES:
        raise RuntimeError("micro normalized parameter order changed")
    vector = np.asarray([float(normalized[name]) for name in _PARAMETER_NAMES])
    local_index = int(value.get("local_index", -1))
    expected_id = _candidate_id(source.source_authentication_id, center.center_id, local_index, vector)
    if int(value.get("candidate_id", -1)) != expected_id:
        raise RuntimeError("micro candidate ID is not reproducible")
    expected_config = _candidate_config(center, vector, candidate_id=expected_id, local_index=local_index, sampling_mode=str(value.get("sampling_mode")), trust_radius=value.get("trust_radius"))
    if canonical_sha256(expected_config) != canonical_sha256(config) or value.get("config_semantic_sha256") != canonical_sha256(config):
        raise RuntimeError("micro job config changed")
    physical = physical_plan_sha256(config)
    if physical == center.physical_plan_sha256 or value.get("physical_plan_sha256") != physical:
        raise RuntimeError("micro physical plan changed or duplicates its center")
    if expected_excluded_physical_plan_sha256 is not None:
        excluded = tuple(sorted(set(str(item) for item in expected_excluded_physical_plan_sha256)))
        if value.get("excluded_physical_plan_set_sha256") != canonical_sha256(list(excluded)) or int(value.get("excluded_physical_plan_count", -1)) != len(excluded) or physical in set(excluded):
            raise RuntimeError("micro exclusion-set provenance changed")
    return copy.deepcopy(dict(job))


def micro_jerk_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Hard constraints first; branch evidence is a soft pre-jerk tie-break."""

    failed = _failed_checks(record)
    full = bool(record.get("full_success", False)) and not failed
    if full:
        group = 0
    elif failed == (_JERK_CHECK,):
        group = 1
    elif _safe_contact_boundary(record):
        group = 2
    else:
        group = 3
    branch = record.get("branch_stability")
    if isinstance(branch, Mapping):
        branch_key = (
            int(branch.get("total_taxel_switch_count", 1 << 30)),
            float(branch.get("maximum_witness_jump_m", math.inf)),
        )
    else:
        branch_key = (1 << 30, math.inf)
    return (group, *branch_key, _peak_jerk(record), len(failed), int(record.get("candidate_id", 1 << 62)))


__all__ = [
    "MICRO_CANDIDATE_COUNT",
    "MICRO_CENTER_COUNT",
    "MICRO_COUNT_PER_CENTER",
    "SENSITIVITY_COUNT_PER_CENTER",
    "TRUST_RADII",
    "MicroJerkCenter",
    "MicroJerkSource",
    "authenticate_micro_jerk_job",
    "authenticate_micro_jerk_source",
    "build_micro_jerk_jobs",
    "compact_c2_band",
    "micro_jerk_candidate_rank",
]
