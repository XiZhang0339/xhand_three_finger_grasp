"""Authenticated catalogs for schema-v7 high-thumb size campaigns.

The tuner owns candidate generation and simulation.  This module is a strict
publication boundary: it consumes a *completed* campaign, authenticates every
published result/config/trace (and optional video), independently checks the
pose-preserving acquisition prefix, chooses a deterministic quota-satisfying
set of grasps, and writes a catalog understood by :mod:`xhand_grasp.viewer`.

Acquisition and lift are deliberately separate classifications.  A successful
grasp is never advertised as a lift unless the source result also records a
successful full trajectory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import file_sha256, write_json
from .config import validate_config
from .controller import quaternion_drift_deg
from .pose_preserving_seed_catalog import REQUIRED_ACQUISITION_CHECKS
from .trajectory_catalog import object_physical_parameters


HIGH_THUMB_SIZE_CATALOG_SCHEMA_VERSION = 1
HIGH_THUMB_SIZE_REPORT_SCHEMA_VERSION = 1
GRASP_PREFIX_SCHEMA_VERSION = 1
CAMPAIGN_KIND = "high_thumb_variable_size_pose_preserving_grasp_then_lift"
EXPERIMENT_ID = (
    "left_opposed_face_palm_down_high_thumb_variable_size_"
    "pose_preserving_grasp_then_lift"
)
THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
VALIDATION_SCOPE = "schema_v7_pose_preserving_grasp_acquisition"
DEFAULT_MINIMUM_SELECTED = 12
MINIMUM_EDGE_COUNT = 4
MINIMUM_SEED_FAMILY_COUNT = 3
MAX_PER_EDGE_TARGET_PAIR = 2
BAND_MINIMUM_COUNT = 3
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATIC_TRACE_FIELDS = {
    "actuator_order",
    "face_order",
    "finger_order",
    "grasp_gate_order",
    "initial_cube_pos_m",
    "initial_cube_quat",
}
_EVENT_TRACE_FIELDS = {
    "grasp_acquisition_step",
    "manipulation_start_step",
    "manipulation_end_step",
    "termination_step",
}

VideoExporter = Callable[[Path, Path, Path], None]


@dataclass(frozen=True)
class CampaignCandidate:
    """One authenticated completed candidate and its selection properties."""

    candidate_id: str
    stage: str
    search_stage: str
    source_family_id: str
    source_trajectory_id: str
    edge_m: float
    thumb_target_rad: float
    acquisition_success: bool
    pose_preservation_success: bool
    lift_success: bool
    pose_margin: float
    stable_grasp_margin: float
    all_state_gate_steps: float
    alignment_duty: float
    alignment_height_p95_m: float
    pad_force_fraction: float
    target_face_duty: float
    peak_total_distal_force_n: float
    saturation_duty: float
    config: dict[str, Any]
    result: dict[str, Any]
    config_path: Path
    result_path: Path
    trace_path: Path
    video_path: Path | None
    artifact_sha256: dict[str, str]
    trace_validation: dict[str, Any]

    @property
    def bend_band(self) -> str:
        return thumb_bend_band(self.thumb_target_rad)

    @property
    def edge_mm(self) -> int:
        return int(round(self.edge_m * 1000.0))

    @property
    def edge_target_pair(self) -> tuple[int, int]:
        return self.edge_mm, int(round(self.thumb_target_rad * 1000.0))

    @property
    def grasp_publishable(self) -> bool:
        return bool(
            self.stage == "acquisition"
            and self.search_stage == "exact"
            and self.acquisition_success
            and self.pose_preservation_success
            and self.trace_validation.get("passed", False)
        )

    @property
    def lift_publishable(self) -> bool:
        return bool(
            self.stage == "lift"
            and self.search_stage == "lift"
            and self.acquisition_success
            and self.pose_preservation_success
            and self.lift_success
            and self.trace_validation.get("passed", False)
        )


def thumb_bend_band(value: float) -> str:
    """Return the plan's canonical bend band for one terminal target."""

    target = float(value)
    if not math.isfinite(target):
        raise ValueError("thumb target must be finite")
    if 1.25 - 1e-12 <= target <= 1.31 + 1e-12:
        return "1p25_to_1p31"
    if 1.31 + 1e-12 < target <= 1.38 + 1e-12:
        return "gt_1p31_to_1p38"
    if 1.38 + 1e-12 < target <= 1.45 + 1e-12:
        return "gt_1p38_to_1p45"
    raise ValueError(f"thumb target {target:.9g} is outside [1.25, 1.45] rad")


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _safe_id(value: object, label: str) -> str:
    identifier = str(value)
    if _SAFE_ID.fullmatch(identifier) is None:
        raise ValueError(f"{label} is not a safe identifier: {identifier!r}")
    return identifier


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _digest(value: object, label: str) -> str:
    digest = str(value).lower()
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return digest


def _confined_member(directory: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate artifact {label} must be a relative path")
    candidate = (directory / value).resolve()
    try:
        candidate.relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError(f"candidate artifact {label} escapes {directory}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"candidate artifact {label} is missing: {candidate}")
    return candidate


def _verify_file(path: Path, expected: object, label: str) -> str:
    digest = _digest(expected, f"{label} SHA-256")
    observed = file_sha256(path)
    if observed != digest:
        raise ValueError(
            f"SHA-256 mismatch for {label}: expected {digest}, observed {observed}"
        )
    return observed


def _nested(mapping: Mapping[str, Any], *paths: Sequence[str]) -> object | None:
    for path in paths:
        value: object = mapping
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            return value
    return None


def _summary(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = result.get("summary")
    return value if isinstance(value, Mapping) else {}


def _stage_grasp_success(result: Mapping[str, Any]) -> bool:
    summary = _summary(result)
    stage = summary.get("stage_status")
    status = result.get("experiment_status")
    declared = (
        status.get("grasp_success", False)
        if isinstance(status, Mapping)
        else result.get("grasp_success", False)
        if "grasp_success" in result
        else stage.get("grasp_success", False)
        if isinstance(stage, Mapping)
        else False
    )
    return bool(
        result.get("acquisition_success", False)
        and declared
    )


def _stage_lift_success(result: Mapping[str, Any]) -> bool:
    summary = _summary(result)
    stage = summary.get("stage_status")
    status = result.get("experiment_status")
    declared = (
        status.get("full_success", False)
        if isinstance(status, Mapping)
        else stage.get("full_success", False)
        if isinstance(stage, Mapping)
        else False
    )
    passed = (
        status.get("passed", declared)
        if isinstance(status, Mapping)
        else summary.get("passed", False)
    )
    return bool(
        result.get("lift_success", False)
        and declared
        and passed
    )


def _pose_metrics(result: Mapping[str, Any]) -> tuple[float, float]:
    summary = _summary(result)
    translation = _nested(
        summary,
        ("metrics", "pose_preservation", "max_translation_m"),
        ("metrics", "max_translation_before_acquisition_m"),
    )
    orientation = _nested(
        summary,
        ("metrics", "pose_preservation", "max_orientation_drift_deg"),
        ("metrics", "max_orientation_before_acquisition_deg"),
    )
    if translation is None or orientation is None:
        rank_metrics = result.get("rank_metrics")
        if isinstance(rank_metrics, Mapping):
            translation = rank_metrics.get(
                "max_translation_before_acquisition_m", translation
            )
            orientation = rank_metrics.get(
                "max_orientation_before_acquisition_deg", orientation
            )
    if translation is None or orientation is None:
        raise ValueError("candidate result is missing pose-preservation metrics")
    return _finite(translation, "pose translation"), _finite(
        orientation, "pose orientation"
    )


def _rank_metric(
    result: Mapping[str, Any], names: str | Sequence[str], default: float
) -> float:
    raw = result.get("rank_metrics")
    aliases = (names,) if isinstance(names, str) else tuple(names)
    value: object = default
    if isinstance(raw, Mapping):
        for name in aliases:
            if name in raw:
                value = raw[name]
                break
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return number if math.isfinite(number) else float(default)


def _all_state_gate_steps(result: Mapping[str, Any]) -> float:
    """Resolve the tuner's cross-CLOSE/VERIFY consecutive gate evidence."""

    rank = result.get("rank_metrics")
    if isinstance(rank, Mapping) and "all_state_gate_steps" in rank:
        value = rank.get("all_state_gate_steps")
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        if math.isfinite(number):
            return max(0.0, number)
    trace_metrics = result.get("thumb_bend_trace_metrics")
    if isinstance(trace_metrics, Mapping):
        value = trace_metrics.get("max_consecutive_all_gate_steps_close_verify")
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        if math.isfinite(number):
            return max(0.0, number)
    return max(0.0, _rank_metric(result, "verify_gate_steps", 0.0))


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load authenticated candidate trace: {path}") from exc


def _trace_pose_validation(
    config: Mapping[str, Any], result: Mapping[str, Any], trace: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Audit the immutable-pose prefix through acquisition, inclusive."""

    required = (
        "time",
        "cube_pos",
        "cube_quat",
        "initial_cube_pos_m",
        "initial_cube_quat",
        "grasp_acquisition_step",
        "control_state",
        "support_contact",
        "hand_cube_contact",
    )
    missing = [name for name in required if name not in trace]
    if missing:
        raise ValueError("schema-v7 trace is missing: " + ", ".join(missing))
    time = np.asarray(trace["time"], dtype=np.float64)
    if time.ndim != 1 or time.size == 0 or not np.isfinite(time).all():
        raise ValueError("schema-v7 trace time must be non-empty and finite")
    if time.size > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError("schema-v7 trace time must be strictly increasing")
    total = len(time)
    acquisition = int(np.asarray(trace["grasp_acquisition_step"]).reshape(()))
    event_in_range = 0 <= acquisition < total
    stop = acquisition + 1 if event_in_range else total
    positions = np.asarray(trace["cube_pos"], dtype=np.float64)
    quaternions = np.asarray(trace["cube_quat"], dtype=np.float64)
    initial_position = np.asarray(trace["initial_cube_pos_m"], dtype=np.float64)
    initial_quaternion = np.asarray(trace["initial_cube_quat"], dtype=np.float64)
    states = np.asarray(trace["control_state"]).astype(str)
    support = np.asarray(trace["support_contact"], dtype=bool)
    hand_contact = np.asarray(trace["hand_cube_contact"], dtype=bool)
    if positions.shape != (total, 3) or quaternions.shape != (total, 4):
        raise ValueError("schema-v7 cube pose does not share the time axis")
    if initial_position.shape != (3,) or initial_quaternion.shape != (4,):
        raise ValueError("schema-v7 immutable cube pose has an invalid shape")
    if states.shape != (total,) or support.shape != (total,) or hand_contact.shape != (
        total,
    ):
        raise ValueError("schema-v7 state/contact arrays do not share the time axis")
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError("schema-v7 cube pose contains NaN or Inf")
    if not np.allclose(
        np.linalg.norm(quaternions, axis=1), 1.0, rtol=0.0, atol=1e-5
    ):
        raise ValueError("schema-v7 cube quaternions are not normalized")

    translation = np.linalg.norm(positions[:stop] - initial_position, axis=1)
    orientation = np.asarray(
        [
            quaternion_drift_deg(initial_quaternion, quaternion)
            for quaternion in quaternions[:stop]
        ],
        dtype=np.float64,
    )
    max_translation = float(np.max(translation))
    max_orientation = float(np.max(orientation))
    policy = config.get("pose_preservation")
    if not isinstance(policy, Mapping):
        raise ValueError("schema-v7 config has no pose_preservation block")
    translation_limit = _finite(
        policy.get("max_translation_m"), "pose translation limit"
    )
    orientation_limit = _finite(
        policy.get("max_orientation_drift_deg"), "pose orientation limit"
    )
    summary_checks = _summary(result).get("checks")
    required_integrity = {
        name: bool(summary_checks.get(name, False))
        if isinstance(summary_checks, Mapping)
        else False
        for name in REQUIRED_ACQUISITION_CHECKS
    }
    physical = object_physical_parameters(config)
    configured_pose = physical["initial_cube_pose"]
    configured_position = np.asarray(configured_pose["position_m"], dtype=np.float64)
    configured_quaternion = np.asarray(
        configured_pose["quaternion_wxyz"], dtype=np.float64
    )
    checks = {
        "acquisition_event_in_range": event_in_range,
        "acquisition_event_is_verify": bool(
            event_in_range and states[acquisition] == "VERIFY"
        ),
        "object_pose_preserved_inclusive": bool(
            event_in_range
            and max_translation <= translation_limit + 1e-12
            and max_orientation <= orientation_limit + 1e-12
        ),
        "support_retained_inclusive": bool(
            event_in_range and np.all(support[:stop])
        ),
        "settle_hand_contact_free": bool(
            event_in_range and not np.any(hand_contact[:stop][states[:stop] == "SETTLE"])
        ),
        "reported_grasp_success": _stage_grasp_success(result),
        "reported_pose_preservation_success": bool(
            result.get("pose_preservation_success", False)
        ),
        "required_acquisition_checks_passed": all(required_integrity.values()),
        "immutable_trace_pose_matches_config": bool(
            np.allclose(
                initial_position, configured_position, rtol=0.0, atol=1e-12
            )
            and quaternion_drift_deg(
                initial_quaternion, configured_quaternion
            )
            <= 1e-9
        ),
    }
    return {
        "validation_scope": VALIDATION_SCOPE,
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "required_acquisition_checks": required_integrity,
        "grasp_acquisition_step": acquisition,
        "grasp_acquisition_time_s": (
            float(time[acquisition]) if event_in_range else None
        ),
        "max_translation_before_acquisition_m": max_translation,
        "translation_limit_m": translation_limit,
        "max_orientation_before_acquisition_deg": max_orientation,
        "orientation_limit_deg": orientation_limit,
        "pose_margin": min(
            1.0 - max_translation / translation_limit,
            1.0 - max_orientation / orientation_limit,
        ),
    }


def _result_bindings(
    campaign: Mapping[str, Any], source: Path
) -> dict[Path, dict[str, Any]]:
    """Return externally declared result digests from a completed campaign."""

    raw = campaign.get("candidate_results")
    if not isinstance(raw, list):
        raise ValueError(
            "completed campaign_results.json must contain candidate_results"
        )
    bindings: dict[Path, dict[str, Any]] = {}
    for record in raw:
        if not isinstance(record, Mapping):
            raise ValueError("candidate_results entries must be objects")
        relative = record.get("result")
        if not isinstance(relative, str) or not relative:
            raise ValueError("candidate result binding has no relative result path")
        path = (source / relative).resolve()
        try:
            path.relative_to(source)
        except ValueError as exc:
            raise ValueError("candidate result binding escapes the campaign") from exc
        if path in bindings:
            raise ValueError(f"duplicate result binding: {relative}")
        normalized = copy.deepcopy(dict(record))
        normalized["candidate_id"] = _safe_id(
            record.get("candidate_id"), "campaign candidate_id"
        )
        normalized["candidate_sha256"] = _digest(
            record.get("candidate_sha256"),
            f"campaign candidate {normalized['candidate_id']} semantic hash",
        )
        normalized["result_sha256"] = _digest(
            record.get("result_sha256"), f"candidate result {relative}"
        )
        stage = str(record.get("stage", ""))
        if stage not in {"acquisition", "lift"}:
            raise ValueError("campaign candidate result binding has invalid stage")
        normalized["stage"] = stage
        bindings[path] = normalized
    if not bindings:
        raise ValueError("completed campaign has no candidate result bindings")
    return bindings


def _manifest_candidate_records(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = manifest.get("candidates")
    if not isinstance(raw, list):
        raise ValueError("campaign_manifest.json must contain candidates")
    result: dict[str, dict[str, Any]] = {}
    for record in raw:
        if not isinstance(record, Mapping):
            raise ValueError("manifest candidates must be objects")
        identifier = _safe_id(record.get("candidate_id"), "manifest candidate_id")
        if identifier in result:
            raise ValueError(f"duplicate manifest candidate_id: {identifier}")
        normalized = copy.deepcopy(dict(record))
        normalized["candidate_id"] = identifier
        normalized["candidate_sha256"] = _digest(
            record.get("candidate_sha256"), f"manifest candidate {identifier}"
        )
        result[identifier] = normalized
    return result


def _load_candidate(
    result_path: Path,
    *,
    campaign_binding: Mapping[str, Any],
    manifest_candidates: Mapping[str, Mapping[str, Any]],
) -> CampaignCandidate:
    _verify_file(
        result_path, campaign_binding.get("result_sha256"), "candidate result"
    )
    result = _load_json(result_path)
    if not bool(result.get("complete", False)):
        raise ValueError(f"candidate result is not complete: {result_path}")
    if result.get("candidate_result_schema_version") != 1:
        raise ValueError(f"candidate result has an unsupported schema: {result_path}")
    if result.get("campaign_kind") != CAMPAIGN_KIND:
        raise ValueError(f"candidate result has the wrong campaign kind: {result_path}")
    candidate_id = _safe_id(result.get("candidate_id"), "candidate_id")
    if candidate_id != campaign_binding.get("candidate_id"):
        raise ValueError("candidate result ID disagrees with campaign_results")
    candidate_sha = _digest(
        result.get("candidate_sha256"), f"candidate {candidate_id} semantic hash"
    )
    manifest_binding = manifest_candidates.get(candidate_id)
    if not isinstance(manifest_binding, Mapping) or manifest_binding.get(
        "candidate_sha256"
    ) != candidate_sha:
        raise ValueError(
            f"candidate {candidate_id} semantic hash is not bound by the manifest"
        )
    stage = str(result.get("stage", ""))
    if stage not in {"acquisition", "lift"}:
        raise ValueError(f"candidate {candidate_id} has invalid stage {stage!r}")
    if stage != campaign_binding.get("stage"):
        raise ValueError("candidate result stage disagrees with campaign_results")
    if stage != manifest_binding.get("stage"):
        raise ValueError("candidate result stage disagrees with campaign_manifest")
    search_stage = _safe_id(
        result.get("search_stage", stage), "candidate search_stage"
    )
    for binding_name, binding in (
        ("campaign_results", campaign_binding),
        ("campaign_manifest", manifest_binding),
    ):
        if "search_stage" in binding and str(binding.get("search_stage")) != search_stage:
            raise ValueError(
                f"candidate search_stage disagrees with {binding_name}"
            )
    if candidate_sha != campaign_binding.get("candidate_sha256"):
        raise ValueError(
            "candidate semantic hash disagrees with campaign_results"
        )
    for field in (
        "acquisition_success",
        "pose_preservation_success",
        "lift_success",
    ):
        if bool(result.get(field, False)) != bool(campaign_binding.get(field, False)):
            raise ValueError(f"candidate {field} disagrees with campaign_results")
    if campaign_binding.get("classification") != result.get("classification"):
        raise ValueError("candidate classification disagrees with campaign_results")
    directory = result_path.parent
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"candidate {candidate_id} has no artifact map")
    hashes = artifacts.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"candidate {candidate_id} has no artifact SHA-256 map")
    config_path = _confined_member(
        directory, artifacts.get("resolved_config"), "resolved_config"
    )
    trace_path = _confined_member(directory, artifacts.get("trace"), "trace")
    verified_hashes = {
        "resolved_config": _verify_file(
            config_path, hashes.get("resolved_config"), "resolved_config"
        ),
        "trace": _verify_file(trace_path, hashes.get("trace"), "trace"),
    }
    video_path: Path | None = None
    raw_video = artifacts.get("video")
    if raw_video is not None:
        video_path = _confined_member(directory, raw_video, "video")
        verified_hashes["video"] = _verify_file(
            video_path, hashes.get("video"), "video"
        )

    config = _load_json(config_path)
    canonical_config_sha256 = hashlib.sha256(
        json.dumps(
            config,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if canonical_config_sha256 != candidate_sha:
        raise ValueError(
            f"candidate {candidate_id} semantic hash disagrees with resolved config"
        )
    validate_config(config)
    if int(config.get("schema_version", 0)) != 7:
        raise ValueError(f"candidate {candidate_id} is not schema-v7")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"candidate {candidate_id} has the wrong experiment_id")
    edge_m = _finite(result.get("edge_m"), "candidate edge_m")
    thumb_target = _finite(result.get("thumb_target_rad"), "thumb target")
    source_family_id = _safe_id(
        result.get("source_family_id"), "source_family_id"
    )
    source_trajectory_id = _safe_id(
        result.get("source_trajectory_id", result.get("source_family_id")),
        "source_trajectory_id",
    )
    for binding_name, binding in (
        ("campaign_results", campaign_binding),
        ("campaign_manifest", manifest_binding),
    ):
        if "source_family_id" in binding and str(
            binding.get("source_family_id")
        ) != source_family_id:
            raise ValueError(
                f"candidate source_family_id disagrees with {binding_name}"
            )
        for field, observed in (
            ("edge_m", edge_m),
            ("thumb_target_rad", thumb_target),
        ):
            if field in binding and not math.isclose(
                _finite(binding.get(field), f"{binding_name}.{field}"),
                observed,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"candidate {field} disagrees with {binding_name}")
    metadata = config.get("candidate_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"candidate {candidate_id} config has no candidate_metadata")
    if str(metadata.get("candidate_id")) != candidate_id:
        raise ValueError(f"candidate {candidate_id} config metadata has the wrong ID")
    if str(metadata.get("source_family_id")) != source_family_id:
        raise ValueError(
            f"candidate {candidate_id} config metadata has the wrong seed family"
        )
    if metadata.get("cube_pose_sampled") is not False:
        raise ValueError(f"candidate {candidate_id} sampled the cube world pose")
    if metadata.get("free_cube_pose_reset_during_run") is not False:
        raise ValueError(f"candidate {candidate_id} reset the free cube pose")
    if not math.isclose(
        edge_m, float(config["cube"]["edge_m"]), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"candidate {candidate_id} edge disagrees with config")
    configured_thumb = float(
        config["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR]
    )
    if not math.isclose(
        thumb_target, configured_thumb, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"candidate {candidate_id} thumb target disagrees with config")
    if not math.isclose(float(config["cube"]["mass_kg"]), 0.160, abs_tol=1e-12):
        raise ValueError(f"candidate {candidate_id} is not the fixed 160 g campaign")
    if not math.isclose(float(config["cube"]["friction"]), 0.8, abs_tol=1e-12):
        raise ValueError(f"candidate {candidate_id} is not the friction 0.8 campaign")
    thumb_bend_band(thumb_target)

    trace = _load_trace(trace_path)
    trace_validation = _trace_pose_validation(config, result, trace)
    translation, orientation = _pose_metrics(result)
    if not math.isclose(
        translation,
        float(trace_validation["max_translation_before_acquisition_m"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ) or not math.isclose(
        orientation,
        float(trace_validation["max_orientation_before_acquisition_deg"]),
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            f"candidate {candidate_id} reported pose metrics disagree with trace"
        )
    return CampaignCandidate(
        candidate_id=candidate_id,
        stage=stage,
        search_stage=search_stage,
        source_family_id=source_family_id,
        source_trajectory_id=source_trajectory_id,
        edge_m=edge_m,
        thumb_target_rad=thumb_target,
        acquisition_success=_stage_grasp_success(result),
        pose_preservation_success=bool(
            result.get("pose_preservation_success", False)
        ),
        lift_success=_stage_lift_success(result),
        pose_margin=float(trace_validation["pose_margin"]),
        stable_grasp_margin=_rank_metric(
            result, ("grasp_stability_margin", "stable_grasp_margin"), 0.0
        ),
        all_state_gate_steps=_all_state_gate_steps(result),
        alignment_duty=_rank_metric(result, "alignment_duty", 0.0),
        alignment_height_p95_m=_rank_metric(
            result, "alignment_height_p95_m", 1e300
        ),
        pad_force_fraction=_rank_metric(
            result, ("minimum_pad_force_fraction", "pad_force_fraction"), 0.0
        ),
        target_face_duty=_rank_metric(
            result, "minimum_target_face_duty", 0.0
        ),
        peak_total_distal_force_n=_rank_metric(
            result, "peak_total_distal_contact_force_n", 1e300
        ),
        saturation_duty=_rank_metric(
            result,
            ("actuator_saturation_fraction", "saturation_duty"),
            1.0,
        ),
        config=config,
        result=result,
        config_path=config_path,
        result_path=result_path,
        trace_path=trace_path,
        video_path=video_path,
        artifact_sha256=verified_hashes,
        trace_validation=trace_validation,
    )


def _candidate_rank(candidate: CampaignCandidate) -> tuple[Any, ...]:
    """Smaller is better; stable across discovery order and worker count."""

    return (
        -candidate.thumb_target_rad,
        -candidate.pose_margin,
        -candidate.stable_grasp_margin,
        -candidate.alignment_duty,
        -candidate.pad_force_fraction,
        candidate.saturation_duty,
        candidate.candidate_id,
    )


def _near_miss_rank(candidate: CampaignCandidate) -> tuple[Any, ...]:
    """Rank failed exact dynamics using the tuner's near-miss priorities.

    A larger command is intentionally late in this ordering: a long real gate
    window and usable acceptance margins are better refinement evidence than a
    high thumb target that never approached a valid three-finger grasp.
    """

    return (
        not candidate.acquisition_success,
        not candidate.pose_preservation_success,
        -candidate.all_state_gate_steps,
        -candidate.pose_margin,
        -candidate.alignment_duty,
        candidate.alignment_height_p95_m,
        -candidate.pad_force_fraction,
        -candidate.target_face_duty,
        -candidate.thumb_target_rad,
        candidate.peak_total_distal_force_n,
        candidate.saturation_duty,
        candidate.candidate_id,
    )


def _selection_counts(
    selected: Iterable[CampaignCandidate],
) -> tuple[Counter[str], Counter[int], Counter[str], Counter[tuple[int, int]]]:
    values = tuple(selected)
    return (
        Counter(value.bend_band for value in values),
        Counter(value.edge_mm for value in values),
        Counter(value.source_family_id for value in values),
        Counter(value.edge_target_pair for value in values),
    )


def _selection_is_valid(
    selected: Sequence[CampaignCandidate], minimum_selected: int
) -> bool:
    bands, edges, families, pairs = _selection_counts(selected)
    return bool(
        len(selected) >= minimum_selected
        and all(
            bands[band] >= BAND_MINIMUM_COUNT
            for band in (
                "1p25_to_1p31",
                "gt_1p31_to_1p38",
                "gt_1p38_to_1p45",
            )
        )
        and len(edges) >= MINIMUM_EDGE_COUNT
        and len(families) >= MINIMUM_SEED_FAMILY_COUNT
        and max(pairs.values(), default=0) <= MAX_PER_EDGE_TARGET_PAIR
    )


def _quota_deficiencies(
    selected: Sequence[CampaignCandidate],
    minimum_selected: int,
    *,
    perturbation_budget_ok: bool,
    perturbation_counts_ok: bool,
) -> list[str]:
    bands, edges, families, pairs = _selection_counts(selected)
    deficiencies: list[str] = []
    if len(selected) < minimum_selected:
        deficiencies.append(
            f"selected_grasp_count:{len(selected)}<{minimum_selected}"
        )
    for band in (
        "1p25_to_1p31",
        "gt_1p31_to_1p38",
        "gt_1p38_to_1p45",
    ):
        if bands[band] < BAND_MINIMUM_COUNT:
            deficiencies.append(
                f"bend_band.{band}:{bands[band]}<{BAND_MINIMUM_COUNT}"
            )
    if len(edges) < MINIMUM_EDGE_COUNT:
        deficiencies.append(
            f"distinct_edges:{len(edges)}<{MINIMUM_EDGE_COUNT}"
        )
    if len(families) < MINIMUM_SEED_FAMILY_COUNT:
        deficiencies.append(
            f"distinct_seed_families:{len(families)}<"
            f"{MINIMUM_SEED_FAMILY_COUNT}"
        )
    maximum_pair = max(pairs.values(), default=0)
    if maximum_pair > MAX_PER_EDGE_TARGET_PAIR:
        deficiencies.append(
            f"maximum_edge_target_pair:{maximum_pair}>"
            f"{MAX_PER_EDGE_TARGET_PAIR}"
        )
    if not perturbation_budget_ok:
        deficiencies.append("perturbation_budget_is_not_16_per_grasp")
    if not perturbation_counts_ok:
        deficiencies.append("selected_grasp_perturbations_incomplete")
    return deficiencies


def _marginal_key(
    candidate: CampaignCandidate,
    selected: Sequence[CampaignCandidate],
    *,
    target_band: str | None,
) -> tuple[Any, ...]:
    bands, edges, families, _ = _selection_counts(selected)
    return (
        0 if target_band is not None and candidate.bend_band == target_band else 1,
        0 if len(families) < MINIMUM_SEED_FAMILY_COUNT and not families[candidate.source_family_id] else 1,
        0 if len(edges) < MINIMUM_EDGE_COUNT and not edges[candidate.edge_mm] else 1,
        bands[candidate.bend_band],
        *_candidate_rank(candidate),
    )


def select_diverse_acquisition_candidates(
    candidates: Iterable[CampaignCandidate],
    *,
    minimum_selected: int = DEFAULT_MINIMUM_SELECTED,
) -> list[CampaignCandidate]:
    """Select deterministic high-quality grasps satisfying all declared quotas.

    The constructive pass fills every bend band while prioritising previously
    unseen families and sizes.  A deterministic one-for-one repair pass then
    resolves any remaining diversity deficit without weakening bend quotas.
    Failure is explicit; the exporter never silently publishes a smaller or
    less diverse success set.
    """

    if minimum_selected < 12:
        raise ValueError("minimum_selected must be at least 12")
    passing = sorted(
        (candidate for candidate in candidates if candidate.grasp_publishable),
        key=_candidate_rank,
    )
    if len(passing) < minimum_selected:
        raise ValueError(
            f"need at least {minimum_selected} authenticated acquisition passes; "
            f"found {len(passing)}"
        )
    bands_available, edges_available, families_available, _ = _selection_counts(
        passing
    )
    missing_bands = [
        band
        for band in ("1p25_to_1p31", "gt_1p31_to_1p38", "gt_1p38_to_1p45")
        if bands_available[band] < BAND_MINIMUM_COUNT
    ]
    if missing_bands:
        raise ValueError("bend-band quota is infeasible: " + ", ".join(missing_bands))
    if len(edges_available) < MINIMUM_EDGE_COUNT:
        raise ValueError("four-size acquisition quota is infeasible")
    if len(families_available) < MINIMUM_SEED_FAMILY_COUNT:
        raise ValueError("three-seed-family acquisition quota is infeasible")

    selected: list[CampaignCandidate] = []
    remaining = list(passing)

    def add_best(target_band: str | None) -> None:
        _, _, _, pair_counts = _selection_counts(selected)
        eligible = [
            candidate
            for candidate in remaining
            if (target_band is None or candidate.bend_band == target_band)
            and pair_counts[candidate.edge_target_pair] < MAX_PER_EDGE_TARGET_PAIR
        ]
        if not eligible:
            raise ValueError("edge/target cap makes the acquisition quota infeasible")
        chosen = min(
            eligible,
            key=lambda candidate: _marginal_key(
                candidate, selected, target_band=target_band
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)

    for band in ("gt_1p38_to_1p45", "gt_1p31_to_1p38", "1p25_to_1p31"):
        for _ in range(BAND_MINIMUM_COUNT):
            add_best(band)
    while len(selected) < minimum_selected:
        add_best(None)

    # Repair a rare case where band-local selections could not expose enough
    # global diversity.  Each swap must strictly improve the capped diversity
    # score and preserve every already-hard quota.
    def diversity(values: Sequence[CampaignCandidate]) -> tuple[int, int]:
        _, edges, families, _ = _selection_counts(values)
        return min(len(families), 3), min(len(edges), 4)

    while diversity(selected) != (3, 4):
        current = diversity(selected)
        replacement: tuple[tuple[Any, ...], int, CampaignCandidate] | None = None
        selected_bands, _, _, _ = _selection_counts(selected)
        for index, old in enumerate(selected):
            if selected_bands[old.bend_band] <= BAND_MINIMUM_COUNT:
                allowed_bands = {old.bend_band}
            else:
                allowed_bands = {
                    "1p25_to_1p31",
                    "gt_1p31_to_1p38",
                    "gt_1p38_to_1p45",
                }
            for candidate in remaining:
                if candidate.bend_band not in allowed_bands:
                    continue
                trial = selected.copy()
                trial[index] = candidate
                _, _, _, pairs = _selection_counts(trial)
                if max(pairs.values(), default=0) > MAX_PER_EDGE_TARGET_PAIR:
                    continue
                improved = diversity(trial)
                if improved <= current:
                    continue
                key = (-improved[0], -improved[1], *_candidate_rank(candidate), index)
                if replacement is None or key < replacement[0]:
                    replacement = (key, index, candidate)
        if replacement is None:
            raise ValueError(
                "available passes satisfy individual quotas but no deterministic "
                "12-member diverse selection was found"
            )
        _, index, candidate = replacement
        old = selected[index]
        selected[index] = candidate
        remaining.remove(candidate)
        remaining.append(old)
        remaining.sort(key=_candidate_rank)

    selected.sort(key=_candidate_rank)
    if not _selection_is_valid(selected, minimum_selected):
        raise RuntimeError("internal quota selection error")
    return selected


def _grasp_prefix(
    trace: Mapping[str, np.ndarray], source_trace_sha256: str
) -> dict[str, np.ndarray]:
    acquisition = int(np.asarray(trace["grasp_acquisition_step"]).reshape(()))
    total = len(np.asarray(trace["time"]))
    if not 0 <= acquisition < total:
        raise ValueError("cannot export an out-of-range acquisition prefix")
    stop = acquisition + 1
    payload: dict[str, np.ndarray] = {}
    for name, raw in trace.items():
        value = np.asarray(raw)
        if name in _EVENT_TRACE_FIELDS:
            if name == "grasp_acquisition_step":
                payload[name] = np.asarray(acquisition, dtype=np.int64)
        elif name == "video_frame_steps":
            frames = np.asarray(value, dtype=np.int64)
            payload[name] = frames[frames <= acquisition].copy()
        elif name in _STATIC_TRACE_FIELDS or value.shape == ():
            payload[name] = np.array(value, copy=True)
        elif value.shape[:1] == (total,):
            payload[name] = np.array(value[:stop], copy=True)
        else:
            payload[name] = np.array(value, copy=True)
    payload.update(
        {
            "high_thumb_grasp_prefix_schema_version": np.asarray(
                GRASP_PREFIX_SCHEMA_VERSION, dtype=np.int64
            ),
            "validation_scope": np.asarray(VALIDATION_SCOPE, dtype=np.str_),
            "source_total_steps": np.asarray(total, dtype=np.int64),
            "segment_start_step": np.asarray(0, dtype=np.int64),
            "segment_stop_step_inclusive": np.asarray(acquisition, dtype=np.int64),
            "source_trace_sha256": np.asarray(source_trace_sha256, dtype=np.str_),
        }
    )
    return payload


def _entry_sort_key(candidate: CampaignCandidate) -> tuple[Any, ...]:
    return (0 if candidate.stage == "acquisition" else 1, *_candidate_rank(candidate))


def _render_report(report: Mapping[str, Any]) -> str:
    highest = report.get("highest_thumb_target_rad")
    best_margin = report.get("best_pose_margin")
    near_miss = report.get("best_near_miss")
    lines = [
        "# Schema-v7 high-thumb variable-size catalog report",
        "",
        f"- Authenticated candidates: {report['authenticated_candidate_count']}",
        f"- Selected grasp acquisitions: {report['selected_grasp_count']}",
        f"- Validated lift trajectories: {report['validated_lift_count']}",
        f"- Quota satisfied: {str(bool(report['quota_satisfied'])).lower()}",
        "- Highest thumb target: "
        + (f"{float(highest):.3f} rad" if highest is not None else "none"),
        "- Best pose margin: "
        + (f"{float(best_margin):.6f}" if best_margin is not None else "none"),
        "- Best exact near miss: "
        + (
            str(near_miss.get("candidate_id"))
            if isinstance(near_miss, Mapping)
            else "none"
        ),
        "",
        "| Candidate | Stage | Edge (mm) | Thumb target (rad) | Seed family | Pose margin |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for record in report["published_candidates"]:
        lines.append(
            "| {candidate_id} | {stage} | {edge_mm} | {thumb_target_rad:.3f} | "
            "{source_family_id} | {pose_margin:.6f} |".format(**record)
        )
    lines.append("")
    return "\n".join(lines)


def export_high_thumb_size_catalog(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    minimum_selected: int = DEFAULT_MINIMUM_SELECTED,
    video_exporter: VideoExporter | None = None,
) -> dict[str, Any]:
    """Authenticate a completed v7 campaign and atomically publish its catalog."""

    source = Path(source_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"campaign source does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"catalog output already exists: {destination}")
    if destination == source:
        raise ValueError("catalog output must not replace the campaign source")
    manifest_path = source / "campaign_manifest.json"
    campaign_path = source / "campaign_results.json"
    manifest = _load_json(manifest_path)
    campaign = _load_json(campaign_path)
    if campaign.get("complete") is not True:
        raise ValueError("campaign_results.json does not declare complete=true")
    if campaign.get("campaign_kind") != CAMPAIGN_KIND:
        raise ValueError("campaign_results.json has the wrong campaign kind")
    if manifest.get("campaign_kind") != CAMPAIGN_KIND:
        raise ValueError("campaign_manifest.json has the wrong campaign kind")
    if campaign.get("campaign_schema_version") != 1 or manifest.get(
        "campaign_schema_version"
    ) != 1:
        raise ValueError("campaign artifacts have an unsupported schema version")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("campaign_manifest.json has the wrong experiment_id")
    for field in ("campaign_input_sha256",):
        if manifest.get(field) != campaign.get(field):
            raise ValueError(f"campaign manifest/results disagree on {field}")
    if campaign.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("campaign_results.json has the wrong experiment_id")
    result_bindings = _result_bindings(campaign, source)
    manifest_candidates = _manifest_candidate_records(manifest)
    candidates = [
        _load_candidate(
            path,
            campaign_binding=binding,
            manifest_candidates=manifest_candidates,
        )
        for path, binding in sorted(
            result_bindings.items(), key=lambda item: str(item[0])
        )
    ]
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("completed campaign contains duplicate candidate IDs")
    if set(candidate_ids) != set(manifest_candidates):
        raise ValueError("manifest and completed candidate result sets disagree")

    selected_records = campaign.get("selected_grasps")
    if not isinstance(selected_records, list):
        raise ValueError("completed campaign has no selected_grasps list")
    selected_ids: list[str] = []
    by_candidate_id = {candidate.candidate_id: candidate for candidate in candidates}
    manifest_budget = manifest.get("budget")
    if not isinstance(manifest_budget, Mapping):
        raise ValueError("campaign_manifest.json has no versioned budget")
    expected_perturbations = int(
        manifest_budget.get("perturbations_per_grasp", -1)
    )
    perturbation_counts: dict[str, int] = {}
    for record in selected_records:
        if not isinstance(record, Mapping):
            raise ValueError("selected_grasps entries must be objects")
        candidate_id = _safe_id(
            record.get("candidate_id"), "selected grasp candidate_id"
        )
        if candidate_id in selected_ids:
            raise ValueError("selected_grasps contains duplicate candidate IDs")
        candidate = by_candidate_id.get(candidate_id)
        if candidate is None:
            raise ValueError("selected_grasps references an unknown candidate")
        if str(record.get("source_family_id")) != candidate.source_family_id:
            raise ValueError("selected grasp seed family disagrees with result")
        for field, expected in (
            ("edge_m", candidate.edge_m),
            ("thumb_target_rad", candidate.thumb_target_rad),
        ):
            if not math.isclose(
                _finite(record.get(field), f"selected_grasps.{field}"),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"selected grasp {field} disagrees with result")
        perturbation_counts[candidate_id] = int(
            record.get("perturbation_count", -1)
        )
        selected_ids.append(candidate_id)
    if int(campaign.get("selected_grasp_count", -1)) != len(selected_ids):
        raise ValueError("selected_grasp_count disagrees with selected_grasps")

    selected_pool = sorted(
        (
            by_candidate_id[candidate_id]
            for candidate_id in selected_ids
            if by_candidate_id[candidate_id].grasp_publishable
        ),
        key=_candidate_rank,
    )
    perturbation_budget_ok = expected_perturbations == 16
    perturbation_counts_ok = bool(selected_pool) and all(
        perturbation_counts.get(candidate.candidate_id) == 16
        for candidate in selected_pool
    )
    quota_deficiencies = _quota_deficiencies(
        selected_pool,
        minimum_selected,
        perturbation_budget_ok=perturbation_budget_ok,
        perturbation_counts_ok=perturbation_counts_ok,
    )
    if campaign.get("selection_satisfied") is not True:
        quota_deficiencies.append("campaign_reported_selection_unsatisfied")
    quota_satisfied = not quota_deficiencies
    # The tuner persists its deterministic final selection and perturbations.
    # Publication authenticates and reorders that exact set; it must not pick a
    # different exact pass that was never assigned the final 16 perturbations.
    selected_grasps = selected_pool
    lift_passes = sorted(
        (candidate for candidate in candidates if candidate.lift_publishable),
        key=_candidate_rank,
    )
    published = sorted([*selected_grasps, *lift_passes], key=_entry_sort_key)
    highest = min(selected_grasps, key=_candidate_rank) if selected_grasps else None
    best_pose = (
        min(
            selected_grasps,
            key=lambda candidate: (
                -candidate.pose_margin,
                *_candidate_rank(candidate),
            ),
        )
        if selected_grasps
        else None
    )
    best_lift = lift_passes[0] if lift_passes else None
    near_miss_pool = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.stage == "acquisition"
            and candidate.search_stage == "exact"
            and not candidate.grasp_publishable
        ),
        key=_near_miss_rank,
    )
    best_near_miss = near_miss_pool[0] if near_miss_pool else None

    destination.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}.staging."
    ) as staging_name:
        staging = Path(staging_name)
        for candidate in published:
            prefix = "grasp" if candidate.stage == "acquisition" else "lift"
            trajectory_id = f"high_thumb_{prefix}_{candidate.candidate_id}"
            member = staging / trajectory_id
            member.mkdir()
            config_path = member / "resolved_config.json"
            trace_path = member / "trace.npz"
            result_path = member / "result.json"
            shutil.copy2(candidate.config_path, config_path)
            shutil.copy2(candidate.trace_path, trace_path)
            trace_hash = file_sha256(trace_path)
            trace = _load_trace(trace_path)
            grasp_trace_path = member / "grasp_trace.npz"
            np.savez_compressed(
                grasp_trace_path, **_grasp_prefix(trace, trace_hash)
            )
            video_path: Path | None = None
            if candidate.video_path is not None:
                video_path = member / "trajectory.mp4"
                shutil.copy2(candidate.video_path, video_path)
            elif video_exporter is not None:
                video_path = member / "trajectory.mp4"
                video_exporter(config_path, trace_path, video_path)
                if not video_path.is_file():
                    raise RuntimeError("video_exporter did not create trajectory.mp4")

            classification = (
                "validated_pose_preserving_grasp_acquisition"
                if candidate.stage == "acquisition"
                else "validated_pose_preserving_lift"
            )
            published_result = {
                "high_thumb_size_catalog_schema_version": (
                    HIGH_THUMB_SIZE_CATALOG_SCHEMA_VERSION
                ),
                "validation_scope": VALIDATION_SCOPE,
                "trajectory_id": trajectory_id,
                "candidate_id": candidate.candidate_id,
                "classification": classification,
                "trace_validation": candidate.trace_validation,
                "source_result": candidate.result,
                "source_provenance": {
                    "result": str(candidate.result_path),
                    "result_sha256": file_sha256(candidate.result_path),
                    "resolved_config_sha256": candidate.artifact_sha256[
                        "resolved_config"
                    ],
                    "trace_sha256": candidate.artifact_sha256["trace"],
                },
            }
            write_json(result_path, published_result)
            hashes = {
                "resolved_config": file_sha256(config_path),
                "result": file_sha256(result_path),
                "trace": trace_hash,
                "grasp_trace": file_sha256(grasp_trace_path),
            }
            if video_path is not None:
                hashes["video"] = file_sha256(video_path)
            alias = f"{prefix}_{candidate.candidate_id}"
            entry_aliases = [alias]
            aliases[alias] = trajectory_id
            entry = {
                "trajectory_id": trajectory_id,
                "candidate_id": candidate.candidate_id,
                "label": alias,
                "aliases": entry_aliases,
                "classification": classification,
                "stage": candidate.stage,
                "grasp_success": True,
                "lift_success": candidate.lift_publishable,
                "source_family_id": candidate.source_family_id,
                "source_trajectory_id": candidate.source_trajectory_id,
                "bend_band": candidate.bend_band,
                "physical_parameters": object_physical_parameters(candidate.config),
                "selection_metrics": {
                    "edge_m": candidate.edge_m,
                    "thumb_target_rad": candidate.thumb_target_rad,
                    "pose_margin": candidate.pose_margin,
                    "stable_grasp_margin": candidate.stable_grasp_margin,
                    "alignment_duty": candidate.alignment_duty,
                    "pad_force_fraction": candidate.pad_force_fraction,
                    "saturation_duty": candidate.saturation_duty,
                },
                "artifacts": {
                    "directory": trajectory_id,
                    "resolved_config": f"{trajectory_id}/{config_path.name}",
                    "result": f"{trajectory_id}/{result_path.name}",
                    "trace": f"{trajectory_id}/{trace_path.name}",
                    "grasp_trace": f"{trajectory_id}/{grasp_trace_path.name}",
                    "video": (
                        f"{trajectory_id}/{video_path.name}"
                        if video_path is not None
                        else None
                    ),
                    "sha256": hashes,
                },
            }
            entries.append(entry)

        diagnostic_artifacts: dict[str, Any] | None = None
        if best_near_miss is not None:
            diagnostic_directory = staging / "diagnostics" / "best_near_miss"
            diagnostic_directory.mkdir(parents=True)
            diagnostic_config = diagnostic_directory / "resolved_config.json"
            diagnostic_result = diagnostic_directory / "result.json"
            diagnostic_trace = diagnostic_directory / "trace.npz"
            shutil.copy2(best_near_miss.config_path, diagnostic_config)
            shutil.copy2(best_near_miss.result_path, diagnostic_result)
            shutil.copy2(best_near_miss.trace_path, diagnostic_trace)
            diagnostic_video: Path | None = None
            if best_near_miss.video_path is not None:
                diagnostic_video = diagnostic_directory / "trajectory.mp4"
                shutil.copy2(best_near_miss.video_path, diagnostic_video)
            diagnostic_hashes = {
                "resolved_config": file_sha256(diagnostic_config),
                "result": file_sha256(diagnostic_result),
                "trace": file_sha256(diagnostic_trace),
            }
            if diagnostic_video is not None:
                diagnostic_hashes["video"] = file_sha256(diagnostic_video)
            diagnostic_artifacts = {
                "directory": "diagnostics/best_near_miss",
                "resolved_config": (
                    "diagnostics/best_near_miss/resolved_config.json"
                ),
                "result": "diagnostics/best_near_miss/result.json",
                "trace": "diagnostics/best_near_miss/trace.npz",
                "video": (
                    "diagnostics/best_near_miss/trajectory.mp4"
                    if diagnostic_video is not None
                    else None
                ),
                "sha256": diagnostic_hashes,
                "source_provenance": {
                    "resolved_config": str(best_near_miss.config_path),
                    "result": str(best_near_miss.result_path),
                    "trace": str(best_near_miss.trace_path),
                    "video": (
                        str(best_near_miss.video_path)
                        if best_near_miss.video_path is not None
                        else None
                    ),
                    "sha256": {
                        "resolved_config": best_near_miss.artifact_sha256[
                            "resolved_config"
                        ],
                        "result": file_sha256(best_near_miss.result_path),
                        "trace": best_near_miss.artifact_sha256["trace"],
                        **(
                            {
                                "video": best_near_miss.artifact_sha256[
                                    "video"
                                ]
                            }
                            if best_near_miss.video_path is not None
                            else {}
                        ),
                    },
                },
            }

        by_id = {entry["trajectory_id"]: entry for entry in entries}

        def add_special(alias: str, candidate: CampaignCandidate) -> None:
            prefix = "grasp" if candidate.stage == "acquisition" else "lift"
            trajectory_id = f"high_thumb_{prefix}_{candidate.candidate_id}"
            by_id[trajectory_id]["aliases"].append(alias)
            aliases[alias] = trajectory_id

        if quota_satisfied:
            assert highest is not None and best_pose is not None
            add_special("highest_thumb_target", highest)
            add_special("best_pose_margin", best_pose)
            add_special("best_nominal", highest)
        elif highest is not None:
            add_special("best_attempt", highest)
        if best_lift is not None:
            add_special("best_lift", best_lift)

        grasp_alias_names = {
            name
            for name in aliases
            if name.startswith("grasp_")
            or name
            in {
                "highest_thumb_target",
                "best_pose_margin",
                "best_nominal",
                "best_attempt",
            }
        }
        lift_alias_names = {
            name
            for name in aliases
            if name.startswith("lift_") or name == "best_lift"
        }

        def subcatalog(
            *,
            stage: str,
            alias_names: set[str],
        ) -> dict[str, Any]:
            stage_entries = [
                copy.deepcopy(entry)
                for entry in entries
                if entry["stage"] == stage
            ]
            for entry in stage_entries:
                entry["aliases"] = [
                    alias for alias in entry["aliases"] if alias in alias_names
                ]
            stage_aliases = {
                name: target
                for name, target in aliases.items()
                if name in alias_names
            }
            return {
                "high_thumb_size_catalog_schema_version": (
                    HIGH_THUMB_SIZE_CATALOG_SCHEMA_VERSION
                ),
                "trajectory_catalog_schema_version": 1,
                "catalog_stage": stage,
                "validation_scope": VALIDATION_SCOPE,
                "experiment_id": EXPERIMENT_ID,
                "trajectory_count": len(stage_entries),
                "validated_trajectory_count": len(stage_entries),
                "campaign_has_validated_trajectory": bool(stage_entries),
                "quota_satisfied": (
                    quota_satisfied if stage == "acquisition" else None
                ),
                "all_published_artifacts_authenticated": True,
                "aliases": stage_aliases,
                "trajectories": stage_entries,
            }

        grasp_catalog_path = staging / "grasp_catalog.json"
        lift_catalog_path = staging / "lift_catalog.json"
        write_json(
            grasp_catalog_path,
            subcatalog(stage="acquisition", alias_names=grasp_alias_names),
        )
        write_json(
            lift_catalog_path,
            subcatalog(stage="lift", alias_names=lift_alias_names),
        )
        subcatalog_artifacts = {
            "grasp": {
                "path": grasp_catalog_path.name,
                "sha256": file_sha256(grasp_catalog_path),
            },
            "lift": {
                "path": lift_catalog_path.name,
                "sha256": file_sha256(lift_catalog_path),
            },
        }

        selected_bands, selected_edges, selected_families, selected_pairs = (
            _selection_counts(selected_grasps)
        )
        report = {
            "high_thumb_size_report_schema_version": (
                HIGH_THUMB_SIZE_REPORT_SCHEMA_VERSION
            ),
            "experiment_id": EXPERIMENT_ID,
            "campaign_source": str(source),
            "campaign_manifest_sha256": file_sha256(manifest_path),
            "campaign_results_sha256": file_sha256(campaign_path),
            "catalogs": copy.deepcopy(subcatalog_artifacts),
            "authenticated_candidate_count": len(candidates),
            "selected_grasp_count": len(selected_grasps),
            "validated_lift_count": len(lift_passes),
            "quota_satisfied": quota_satisfied,
            "quota_deficiencies": quota_deficiencies,
            "campaign_reported_selection_satisfied": bool(
                campaign.get("selection_satisfied", False)
            ),
            "highest_thumb_target_rad": (
                highest.thumb_target_rad if highest is not None else None
            ),
            "highest_thumb_candidate_id": (
                highest.candidate_id if highest is not None else None
            ),
            "best_pose_margin": (
                best_pose.pose_margin if best_pose is not None else None
            ),
            "best_pose_margin_candidate_id": (
                best_pose.candidate_id if best_pose is not None else None
            ),
            "best_near_miss": (
                {
                    "candidate_id": best_near_miss.candidate_id,
                    "edge_m": best_near_miss.edge_m,
                    "thumb_target_rad": best_near_miss.thumb_target_rad,
                    "source_family_id": best_near_miss.source_family_id,
                    "search_stage": best_near_miss.search_stage,
                    "rank_evidence": {
                        "all_state_gate_steps": (
                            best_near_miss.all_state_gate_steps
                        ),
                        "pose_margin": best_near_miss.pose_margin,
                        "alignment_duty": best_near_miss.alignment_duty,
                        "alignment_height_p95_m": (
                            best_near_miss.alignment_height_p95_m
                        ),
                        "pad_force_fraction": (
                            best_near_miss.pad_force_fraction
                        ),
                        "target_face_duty": best_near_miss.target_face_duty,
                    },
                    "failed_checks": best_near_miss.trace_validation[
                        "failed_checks"
                    ],
                    "artifacts": copy.deepcopy(diagnostic_artifacts),
                }
                if best_near_miss is not None
                else None
            ),
            "quota": {
                "minimum_selected": minimum_selected,
                "minimum_distinct_edges": MINIMUM_EDGE_COUNT,
                "minimum_distinct_seed_families": MINIMUM_SEED_FAMILY_COUNT,
                "minimum_per_bend_band": BAND_MINIMUM_COUNT,
                "maximum_per_edge_target_pair": MAX_PER_EDGE_TARGET_PAIR,
            },
            "quota_achieved": {
                "bend_band_counts": dict(sorted(selected_bands.items())),
                "distinct_edges_mm": sorted(selected_edges),
                "distinct_seed_families": sorted(selected_families),
                "maximum_edge_target_pair_count": max(
                    selected_pairs.values(), default=0
                ),
            },
            "published_candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "stage": candidate.stage,
                    "edge_mm": candidate.edge_mm,
                    "thumb_target_rad": candidate.thumb_target_rad,
                    "bend_band": candidate.bend_band,
                    "source_family_id": candidate.source_family_id,
                    "pose_margin": candidate.pose_margin,
                    "lift_success": candidate.lift_publishable,
                }
                for candidate in published
            ],
            "unpublished_candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "stage": candidate.stage,
                    "grasp_publishable": candidate.grasp_publishable,
                    "lift_publishable": candidate.lift_publishable,
                    "failed_checks": candidate.trace_validation["failed_checks"],
                }
                for candidate in candidates
                if candidate not in published
            ],
        }
        report_path = staging / "selection_report.json"
        write_json(report_path, report)
        markdown_report_path = staging / "EXPERIMENT_REPORT.md"
        markdown_report_path.write_text(
            _render_report(report), encoding="utf-8"
        )
        catalog = {
            "high_thumb_size_catalog_schema_version": (
                HIGH_THUMB_SIZE_CATALOG_SCHEMA_VERSION
            ),
            "trajectory_catalog_schema_version": 1,
            "validation_scope": VALIDATION_SCOPE,
            "experiment_id": EXPERIMENT_ID,
            "trajectory_count": len(entries),
            "selected_grasp_count": len(selected_grasps),
            "validated_lift_count": len(lift_passes),
            "quota_satisfied": quota_satisfied,
            "quota_deficiencies": quota_deficiencies,
            "all_published_artifacts_authenticated": True,
            "highest_thumb_target_rad": (
                highest.thumb_target_rad if highest is not None else None
            ),
            "highest_thumb_target": aliases.get("highest_thumb_target"),
            "best_pose_margin": {
                "value": best_pose.pose_margin if best_pose is not None else None,
                "trajectory_id": aliases.get("best_pose_margin"),
            },
            "aliases": aliases,
            "catalogs": copy.deepcopy(subcatalog_artifacts),
            "diagnostics": {
                "best_near_miss": copy.deepcopy(diagnostic_artifacts)
            },
            "selection_report": {
                "path": report_path.name,
                "sha256": file_sha256(report_path),
            },
            "experiment_report": {
                "path": markdown_report_path.name,
                "sha256": file_sha256(markdown_report_path),
            },
            "trajectories": entries,
        }
        write_json(staging / "catalog.json", catalog)
        staging.rename(destination)
    return catalog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish an authenticated schema-v7 high-thumb size catalog"
    )
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-selected", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    catalog = export_high_thumb_size_catalog(
        args.source_dir,
        args.output_dir,
        minimum_selected=args.minimum_selected,
    )
    print(json.dumps(catalog, indent=2, sort_keys=True))
    return 0 if catalog["quota_satisfied"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BAND_MINIMUM_COUNT",
    "CAMPAIGN_KIND",
    "CampaignCandidate",
    "DEFAULT_MINIMUM_SELECTED",
    "EXPERIMENT_ID",
    "HIGH_THUMB_SIZE_CATALOG_SCHEMA_VERSION",
    "HIGH_THUMB_SIZE_REPORT_SCHEMA_VERSION",
    "MAX_PER_EDGE_TARGET_PAIR",
    "MINIMUM_EDGE_COUNT",
    "MINIMUM_SEED_FAMILY_COUNT",
    "export_high_thumb_size_catalog",
    "main",
    "select_diverse_acquisition_candidates",
    "thumb_bend_band",
]
