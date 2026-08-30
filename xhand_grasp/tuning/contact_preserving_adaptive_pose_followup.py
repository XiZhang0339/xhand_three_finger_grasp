"""Conditional second-stage pose refinement for schema-v14 contact mode rescue.

This module is intentionally a *pure job generator*.  It does not start a
MuJoCo process, write an artifact, or own a campaign ledger.  A caller first
authenticates the completed 256-candidate contact-mode stage, constructs an
``AdaptiveContactModePoseSource`` with :func:`build_adaptive_pose_source`, and
then hands the generated jobs to the existing atomic candidate runner.

The follow-up is legal only when the first stage contains no hard pass.  Its
centres are the lowest-jerk records whose only failed hard check is jerk.  It
keeps the cube, all acceptance thresholds, the 3 s/20-segment plan, force
feedback and terminal manipulation delta byte-for-byte unchanged.  Only the
hand root pose, measured grasp qpos, precontact command and preload command
receive small deterministic perturbations.
"""

from __future__ import annotations

import copy
import itertools
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import ACTIVE_ACTUATORS, validate_config
from ..experiment import resolve_experiment
from ..grasp_pose import canonical_sha256
from ..relative_wrist_pose import (
    rotation_matrix_to_rpy_degrees,
    rotvec_degrees_to_rotation_matrix,
)
from ..scene import rpy_degrees_to_rotation_matrix
from .contact_constrained_planner import (
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
)
from .contact_preserving_contact_mode_pose_rescue import (
    EXPERIMENT_ID,
    contact_mode_physical_config_sha256,
)
from .contact_preserving_joint_refinement import (
    JointRefinementLimits,
    resolve_joint_refinement_limits,
)
from .contact_preserving_time_warp import _time_warp_controller_id


ADAPTIVE_POSE_FOLLOWUP_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
EXPECTED_FIRST_STAGE_CANDIDATE_COUNT = 256
MAXIMUM_FOLLOWUP_CANDIDATE_COUNT = 128
MAXIMUM_PROBE_CANDIDATE_COUNT = 64
DEFAULT_CENTER_COUNT = 4
_JERK_CHECK = "smooth_motion_jerk_within_limit"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: Any, *, label: str) -> str:
    result = str(value)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _failed_checks(record: Mapping[str, Any]) -> tuple[str, ...]:
    raw = record.get("summary", {}).get("failed_checks", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ("malformed_failed_checks",)
    return tuple(str(value) for value in raw)


def _peak_jerk(record: Mapping[str, Any]) -> float:
    try:
        result = float(
            record["summary"]["metrics"]["motion_smoothness"][
                "operation_peak_abs_filtered_jerk_m_s3"
            ]
        )
    except (KeyError, TypeError, ValueError):
        return math.inf
    return result if math.isfinite(result) else math.inf


def _all_other_hard_checks_pass(record: Mapping[str, Any]) -> bool:
    """Reject summaries that merely *claim* a jerk-only failure inconsistently."""

    if record.get("grasp_success") is not True or record.get("full_success") is True:
        return False
    if _failed_checks(record) != (_JERK_CHECK,):
        return False
    checks = record.get("summary", {}).get("checks")
    if not isinstance(checks, Mapping) or checks.get(_JERK_CHECK) is not False:
        return False
    if any(value is not True for key, value in checks.items() if key != _JERK_CHECK):
        return False
    try:
        contact = record["summary"]["metrics"]["contact_preserving_planned_lift"]
        if float(contact["simultaneous_target_face_effective_duty"]) < 0.99:
            return False
        if float(contact["simultaneous_longest_contact_loss_s"]) > 0.010 + 1e-12:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    diagnostics = record.get("contact_mode_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return False
    if diagnostics.get("measurement_available") is not True:
        return False
    return float(diagnostics.get("simultaneous_effective_contact_duty", -1.0)) >= 0.99


def _fixed_config_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove exactly the four fields that this follow-up may alter."""

    payload = copy.deepcopy(dict(config))
    payload.pop("candidate_metadata", None)
    for key in (
        "object_config_id",
        "grasp_pose_id",
        "grasp_object_pair_id",
        "planner_id",
        "controller_id",
        "experiment_status",
    ):
        payload.pop(key, None)
    payload.pop("hand_pose", None)
    grasp_pose = payload.get("grasp_pose")
    if isinstance(grasp_pose, dict):
        grasp_pose.pop("nominal_joint_qpos_rad", None)
    control = payload.get("control")
    if isinstance(control, dict):
        control.pop("precontact_targets_rad", None)
        control.pop("contact_preload_targets_rad", None)
    return payload


def _center_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    diagnostics = record.get("contact_mode_diagnostics", {})
    transitions = diagnostics.get("taxel_count_transition_count", {})
    centroid_steps = diagnostics.get("contact_centroid_max_step_m", {})
    return (
        _peak_jerk(record),
        int(transitions.get("thumb", 10**9)),
        int(sum(int(transitions.get(name, 10**8)) for name in ("index", "mid"))),
        float(centroid_steps.get("thumb", math.inf)),
        int(record.get("candidate_id", 2**63 - 1)),
    )


@dataclass(frozen=True, slots=True)
class AdaptiveContactModeCenter:
    candidate_id: int
    config: dict[str, Any]
    result: dict[str, Any]
    physical_config_sha256: str
    config_semantic_sha256: str
    trace_sha256: str
    peak_jerk_m_s3: float

    def as_mapping(self, *, include_payloads: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "candidate_id": int(self.candidate_id),
            "physical_config_sha256": self.physical_config_sha256,
            "config_semantic_sha256": self.config_semantic_sha256,
            "trace_sha256": self.trace_sha256,
            "peak_jerk_m_s3": self.peak_jerk_m_s3,
        }
        if include_payloads:
            result["config"] = copy.deepcopy(self.config)
            result["result"] = copy.deepcopy(self.result)
        return result


@dataclass(frozen=True, slots=True)
class AdaptiveContactModePoseSource:
    """Authenticated-data boundary consumed by the pure generator."""

    search_report_sha256: str
    first_stage_candidate_count: int
    centers: tuple[AdaptiveContactModeCenter, ...]
    prior_physical_config_sha256: tuple[str, ...]
    fixed_config_payload_sha256: str
    source_authentication_id: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_POSE_FOLLOWUP_SCHEMA_VERSION,
            "search_report_sha256": self.search_report_sha256,
            "first_stage_candidate_count": self.first_stage_candidate_count,
            "centers": [value.as_mapping() for value in self.centers],
            "prior_physical_config_sha256": list(
                self.prior_physical_config_sha256
            ),
            "fixed_config_payload_sha256": self.fixed_config_payload_sha256,
            "source_authentication_id": self.source_authentication_id,
        }


@dataclass(frozen=True, slots=True)
class AdaptivePoseFollowupBudget:
    candidate_count: int = MAXIMUM_FOLLOWUP_CANDIDATE_COUNT
    seed: int = DEFAULT_SEED
    root_translation_radius_cube_m: tuple[float, float, float] = (
        0.00010,
        0.00010,
        0.00025,
    )
    wrist_local_rotvec_radius_deg: tuple[float, float, float] = (
        0.06,
        0.06,
        0.06,
    )
    wrist_local_rotvec_norm_limit_deg: float = 0.08
    nominal_qpos_radius_rad: float = 0.0010
    precontact_residual_radius_rad: float = 0.0004
    preload_residual_radius_rad: float = 0.0006

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_count, bool)
            or not 1 <= int(self.candidate_count) <= MAXIMUM_FOLLOWUP_CANDIDATE_COUNT
        ):
            raise ValueError(
                "candidate_count must lie within "
                f"[1, {MAXIMUM_FOLLOWUP_CANDIDATE_COUNT}]"
            )
        if isinstance(self.seed, bool) or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in (
            "root_translation_radius_cube_m",
            "wrist_local_rotvec_radius_deg",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"{name} must contain three positive finite values")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        for name in (
            "wrist_local_rotvec_norm_limit_deg",
            "nominal_qpos_radius_rad",
            "precontact_residual_radius_rad",
            "preload_residual_radius_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_POSE_FOLLOWUP_SCHEMA_VERSION,
            "candidate_count": int(self.candidate_count),
            "maximum_candidate_count": MAXIMUM_FOLLOWUP_CANDIDATE_COUNT,
            "seed": int(self.seed),
            "root_translation_radius_cube_m": list(
                self.root_translation_radius_cube_m
            ),
            "wrist_local_rotvec_radius_deg": list(
                self.wrist_local_rotvec_radius_deg
            ),
            "wrist_local_rotvec_norm_limit_deg": (
                self.wrist_local_rotvec_norm_limit_deg
            ),
            "nominal_qpos_radius_rad": self.nominal_qpos_radius_rad,
            "precontact_residual_radius_rad": (
                self.precontact_residual_radius_rad
            ),
            "preload_residual_radius_rad": self.preload_residual_radius_rad,
        }


def build_adaptive_pose_source(
    records: Sequence[Mapping[str, Any]],
    configs_by_candidate_id: Mapping[int, Mapping[str, Any]],
    trace_sha256_by_candidate_id: Mapping[int, str],
    *,
    search_report_sha256: str,
    expected_candidate_count: int = EXPECTED_FIRST_STAGE_CANDIDATE_COUNT,
    center_count: int = DEFAULT_CENTER_COUNT,
) -> AdaptiveContactModePoseSource:
    """Select and bind the top jerk-only centres from a completed stage.

    File authentication is deliberately outside this pure function.  The
    caller should obtain ``records`` and digests through the existing atomic
    artifact authenticator.  This function then binds every prior physical
    configuration into one deterministic source identity, which lets the
    existing campaign manifest reject resume after any source change.
    """

    report_sha = _require_sha256(search_report_sha256, label="search_report_sha256")
    if isinstance(expected_candidate_count, bool) or int(expected_candidate_count) <= 0:
        raise ValueError("expected_candidate_count must be positive")
    if len(records) != int(expected_candidate_count):
        raise RuntimeError(
            "adaptive follow-up requires the complete first-stage candidate set"
        )
    if isinstance(center_count, bool) or not 1 <= int(center_count) <= 8:
        raise ValueError("center_count must lie within [1, 8]")
    identifiers = [int(value.get("candidate_id", -1)) for value in records]
    if len(set(identifiers)) != len(identifiers) or any(value < 0 for value in identifiers):
        raise RuntimeError("first-stage records contain invalid or duplicate candidate IDs")
    if any(bool(value.get("full_success", False)) for value in records):
        raise RuntimeError("adaptive follow-up is forbidden after a first-stage hard pass")
    if set(int(value) for value in configs_by_candidate_id) != set(identifiers):
        raise RuntimeError("adaptive source did not bind every first-stage configuration")
    if set(int(value) for value in trace_sha256_by_candidate_id) != set(identifiers):
        raise RuntimeError("adaptive source did not bind every retained trace digest")

    config_values: dict[int, dict[str, Any]] = {}
    physical_values: dict[int, str] = {}
    for record in records:
        candidate_id = int(record["candidate_id"])
        config = copy.deepcopy(dict(configs_by_candidate_id[candidate_id]))
        if int(config.get("schema_version", 0)) != 14 or config.get("experiment_id") != EXPERIMENT_ID:
            raise RuntimeError("adaptive source contains a non-v14 configuration")
        expected_config_sha = str(record.get("config_semantic_sha256", ""))
        if canonical_sha256(config) != expected_config_sha:
            raise RuntimeError("adaptive source configuration semantic hash changed")
        _require_sha256(
            trace_sha256_by_candidate_id[candidate_id],
            label=f"trace_sha256[{candidate_id}]",
        )
        config_values[candidate_id] = config
        physical_values[candidate_id] = contact_mode_physical_config_sha256(config)
    prior_physical = tuple(sorted(physical_values.values()))
    if len(set(prior_physical)) != len(prior_physical):
        raise RuntimeError("first-stage source lost physical uniqueness")

    eligible = [copy.deepcopy(dict(value)) for value in records if _all_other_hard_checks_pass(value)]
    eligible.sort(key=_center_rank)
    if not eligible:
        raise RuntimeError("first stage contains no strict jerk-only top item")
    selected = eligible[: min(int(center_count), len(eligible))]
    fixed_sha: str | None = None
    centers: list[AdaptiveContactModeCenter] = []
    for record in selected:
        candidate_id = int(record["candidate_id"])
        config = config_values[candidate_id]
        plan = config.get("manipulation_plan", {})
        if (
            abs(float(plan.get("duration_s", math.nan)) - 3.0) > 1e-12
            or len(plan.get("knot_times_s", ())) != 21
        ):
            raise RuntimeError("adaptive source changed the fixed 3 s/20-segment plan")
        current_fixed_sha = canonical_sha256(_fixed_config_payload(config))
        if fixed_sha is None:
            fixed_sha = current_fixed_sha
        elif current_fixed_sha != fixed_sha:
            raise RuntimeError("adaptive top centres disagree outside the four allowed fields")
        centers.append(
            AdaptiveContactModeCenter(
                candidate_id=candidate_id,
                config=copy.deepcopy(config),
                result=copy.deepcopy(record),
                physical_config_sha256=physical_values[candidate_id],
                config_semantic_sha256=canonical_sha256(config),
                trace_sha256=str(trace_sha256_by_candidate_id[candidate_id]),
                peak_jerk_m_s3=_peak_jerk(record),
            )
        )
    assert fixed_sha is not None
    source_payload = {
        "schema_version": ADAPTIVE_POSE_FOLLOWUP_SCHEMA_VERSION,
        "search_report_sha256": report_sha,
        "first_stage_candidate_count": int(expected_candidate_count),
        "centers": [value.as_mapping() for value in centers],
        "prior_physical_config_sha256": list(prior_physical),
        "fixed_config_payload_sha256": fixed_sha,
    }
    return AdaptiveContactModePoseSource(
        search_report_sha256=report_sha,
        first_stage_candidate_count=int(expected_candidate_count),
        centers=tuple(centers),
        prior_physical_config_sha256=prior_physical,
        fixed_config_payload_sha256=fixed_sha,
        source_authentication_id=canonical_sha256(source_payload),
    )


def _probe_samples() -> np.ndarray:
    """Return the fixed 64-candidate one-dimensional finite-difference shell.

    The first contact-mode stage covered root/rotation and three thumb axes
    explicitly, while most index/middle evidence came from destructive 30-D
    random samples.  The follow-up therefore puts all 30 index/middle
    nominal/precontact/preload signs first.  Remaining slots cover fine root,
    wrist and thumb controls without ever mixing dimensions.
    """

    result: list[np.ndarray] = []

    def append(axis: int, value: float) -> None:
        sample = np.zeros(30, dtype=np.float64)
        sample[axis] = value
        result.append(sample)

    # 5 index/middle actuators x 3 state/control blocks x 2 signs = 30.
    for start in (6, 14, 22):
        for actuator_index in range(3, 8):
            for sign in (-1.0, 1.0):
                append(start + actuator_index, sign)
    # Six fine root-Z probes around the measured -0.2 mm first-stage best.
    for value in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
        append(2, value)
    # Root X/Y and local wrist axes: 4 + 6 = 10.
    for axis in (0, 1, 3, 4, 5):
        for sign in (-1.0, 1.0):
            append(axis, sign)
    # 3 thumb actuators x 3 blocks x 2 signs = 18.
    for start in (6, 14, 22):
        for actuator_index in range(3):
            for sign in (-1.0, 1.0):
                append(start + actuator_index, sign)
    values = np.asarray(result, dtype=np.float64)
    if values.shape != (MAXIMUM_PROBE_CANDIDATE_COUNT, 30):
        raise AssertionError("adaptive probe schedule lost its exact 64 candidates")
    if np.any(np.count_nonzero(values, axis=1) != 1):
        raise AssertionError("adaptive finite-difference probes must be one-dimensional")
    return values


def _preserve_closing_rays(
    proposed_preload: dict[str, float],
    proposed_precontact: dict[str, float],
    base_preload: Mapping[str, float],
    base_precontact: Mapping[str, float],
) -> None:
    for names in (
        ACTIVE_ACTUATORS[0:3],
        ACTIVE_ACTUATORS[3:6],
        ACTIVE_ACTUATORS[6:8],
    ):
        base = np.asarray(
            [float(base_preload[name]) - float(base_precontact[name]) for name in names]
        )
        proposed = np.asarray(
            [float(proposed_preload[name]) - float(proposed_precontact[name]) for name in names]
        )
        if float(base @ proposed) <= 0.0 or np.linalg.norm(proposed) <= 1e-6:
            for name in names:
                proposed_preload[name] = float(base_preload[name])
                proposed_precontact[name] = float(base_precontact[name])


def _rotation_matrix_to_rotvec_degrees(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    sine = math.sin(angle)
    if abs(sine) <= 1e-10:
        raise ValueError("adaptive follow-up rotation is numerically singular")
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sine)
    return np.degrees(axis * angle)


def _candidate_identity(
    source: AdaptiveContactModePoseSource,
    center: AdaptiveContactModeCenter,
    sequence: int,
    normalized: np.ndarray,
    *,
    stage: str,
    probe_evidence_sha256: str | None,
) -> tuple[int, str]:
    digest = canonical_sha256(
        {
            "schema_version": ADAPTIVE_POSE_FOLLOWUP_SCHEMA_VERSION,
            "source_authentication_id": source.source_authentication_id,
            "center_candidate_id": center.candidate_id,
            "job_sequence_index": int(sequence),
            "normalized_sample": [float(value) for value in normalized],
            "stage": str(stage),
            "probe_evidence_sha256": probe_evidence_sha256,
        }
    )
    candidate_id = 15_700_000_000_000_000 + int(digest[:13], 16) % 100_000_000_000_000
    return candidate_id, digest


def _materialize_candidate(
    source: AdaptiveContactModePoseSource,
    center: AdaptiveContactModeCenter,
    normalized: np.ndarray,
    sequence: int,
    budget: AdaptivePoseFollowupBudget,
    limits: JointRefinementLimits,
    *,
    validate: bool,
    stage: str,
    probe_evidence_sha256: str | None,
) -> dict[str, Any]:
    config = copy.deepcopy(center.config)
    base_cube = copy.deepcopy(config["cube"])
    base_plan = copy.deepcopy(config["manipulation_plan"])
    base_terminal = copy.deepcopy(config["control"]["manipulation_delta_rad"])
    base_nominal = copy.deepcopy(config["grasp_pose"]["nominal_joint_qpos_rad"])
    base_precontact = copy.deepcopy(config["control"]["precontact_targets_rad"])
    base_preload = copy.deepcopy(config["control"]["contact_preload_targets_rad"])

    translation_delta_cube = normalized[0:3] * np.asarray(
        budget.root_translation_radius_cube_m
    )
    local_rotvec = normalized[3:6] * np.asarray(
        budget.wrist_local_rotvec_radius_deg
    )
    local_norm = float(np.linalg.norm(local_rotvec))
    if local_norm > budget.wrist_local_rotvec_norm_limit_deg:
        local_rotvec *= budget.wrist_local_rotvec_norm_limit_deg / local_norm
    cube_rotation = rpy_degrees_to_rotation_matrix(config["cube"]["rpy_deg"])
    base_translation = np.asarray(config["hand_pose"]["translation_m"], dtype=np.float64)
    base_rpy = np.asarray(config["hand_pose"]["rpy_deg"], dtype=np.float64)
    base_rotation = rpy_degrees_to_rotation_matrix(base_rpy)
    config["hand_pose"]["translation_m"] = (
        base_translation + cube_rotation @ translation_delta_cube
    ).tolist()
    definition = resolve_experiment(config)
    proposed_rotation = base_rotation @ rotvec_degrees_to_rotation_matrix(local_rotvec)
    proposed_rpy = rotation_matrix_to_rpy_degrees(
        proposed_rotation, reference_rpy_deg=base_rpy
    )
    proposed_rpy[0] = np.clip(proposed_rpy[0], *definition.search_bounds.hand_roll_deg)
    proposed_rpy[2] = np.clip(proposed_rpy[2], *definition.search_bounds.hand_yaw_deg)
    effective_rotation = rpy_degrees_to_rotation_matrix(proposed_rpy)
    effective_rotvec = _rotation_matrix_to_rotvec_degrees(
        base_rotation.T @ effective_rotation
    )
    effective_norm = float(np.linalg.norm(effective_rotvec))
    if effective_norm > budget.wrist_local_rotvec_norm_limit_deg:
        effective_rotvec *= budget.wrist_local_rotvec_norm_limit_deg / effective_norm
        effective_rotation = base_rotation @ rotvec_degrees_to_rotation_matrix(
            effective_rotvec
        )
        proposed_rpy = rotation_matrix_to_rpy_degrees(
            effective_rotation, reference_rpy_deg=base_rpy
        )
    config["hand_pose"]["rpy_deg"] = proposed_rpy.tolist()

    nominal_delta = normalized[6:14] * budget.nominal_qpos_radius_rad
    precontact_residual = normalized[14:22] * budget.precontact_residual_radius_rad
    preload_residual = normalized[22:30] * budget.preload_residual_radius_rad
    nominal: dict[str, float] = {}
    precontact: dict[str, float] = {}
    preload: dict[str, float] = {}
    for index, name in enumerate(ACTIVE_ACTUATORS):
        low, high = limits.preload_target_rad[name]
        if name == "left_hand_thumb_bend_joint_actuator":
            low, high = max(low, 1.40), min(high, 1.60)
        nominal[name] = float(
            np.clip(float(base_nominal[name]) + nominal_delta[index], low, high)
        )
        applied_nominal = nominal[name] - float(base_nominal[name])
        command_low, command_high = limits.command_target_rad[name]
        precontact[name] = float(
            np.clip(
                float(base_precontact[name]) + applied_nominal + precontact_residual[index],
                command_low,
                command_high,
            )
        )
        preload[name] = float(
            np.clip(
                float(base_preload[name]) + applied_nominal + preload_residual[index],
                low,
                high,
            )
        )
    _preserve_closing_rays(preload, precontact, base_preload, base_precontact)
    config["grasp_pose"]["nominal_joint_qpos_rad"] = nominal
    config["control"]["precontact_targets_rad"] = precontact
    config["control"]["contact_preload_targets_rad"] = preload

    config["object_config_id"] = v14_object_config_id(config)
    config["grasp_pose_id"] = v14_grasp_pose_id(config)
    config["grasp_object_pair_id"] = v14_grasp_object_pair_id(config)
    candidate_id, candidate_sha = _candidate_identity(
        source,
        center,
        sequence,
        normalized,
        stage=stage,
        probe_evidence_sha256=probe_evidence_sha256,
    )
    config["planner_id"] = canonical_sha256(
        {
            "schema_version": ADAPTIVE_POSE_FOLLOWUP_SCHEMA_VERSION,
            "kind": "v14_adaptive_pose_followup_reused_plan",
            "source_authentication_id": source.source_authentication_id,
            "center_candidate_id": center.candidate_id,
            "source_plan_id": base_plan["plan_id"],
            "grasp_object_pair_id": config["grasp_object_pair_id"],
            "candidate_sha256": candidate_sha,
        }
    )
    config["controller_id"] = _time_warp_controller_id(config)
    metadata = config.setdefault("candidate_metadata", {})
    metadata["v14_adaptive_pose_followup"] = {
        "schema_version": ADAPTIVE_POSE_FOLLOWUP_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha,
        "job_sequence_index": int(sequence),
        "adaptive_stage": str(stage),
        "center_candidate_id": center.candidate_id,
        "center_peak_jerk_m_s3": center.peak_jerk_m_s3,
        "source_authentication_id": source.source_authentication_id,
        "probe_evidence_sha256": probe_evidence_sha256,
        "normalized_sample": [float(value) for value in normalized],
        "root_translation_delta_cube_m": translation_delta_cube.tolist(),
        "wrist_local_rotvec_deg": effective_rotvec.tolist(),
        "nominal_qpos_delta_rad": {
            name: nominal[name] - float(base_nominal[name]) for name in ACTIVE_ACTUATORS
        },
        "precontact_residual_rad": {
            name: precontact[name]
            - float(base_precontact[name])
            - (nominal[name] - float(base_nominal[name]))
            for name in ACTIVE_ACTUATORS
        },
        "preload_residual_rad": {
            name: preload[name]
            - float(base_preload[name])
            - (nominal[name] - float(base_nominal[name]))
            for name in ACTIVE_ACTUATORS
        },
        "cube_pose_sampled": False,
        "manipulation_plan_changed": False,
        "acceptance_thresholds_changed": False,
        "full_reset_required": True,
    }
    if config["cube"] != base_cube:
        raise AssertionError("adaptive pose follow-up changed the cube")
    if (
        config["manipulation_plan"] != base_plan
        or config["control"]["manipulation_delta_rad"] != base_terminal
    ):
        raise AssertionError("adaptive pose follow-up changed the fixed plan")
    if canonical_sha256(_fixed_config_payload(config)) != source.fixed_config_payload_sha256:
        raise AssertionError("adaptive pose follow-up changed a fixed config field")
    if validate:
        validate_config(config)
    physical_sha = contact_mode_physical_config_sha256(config)
    metadata["v14_adaptive_pose_followup"]["physical_config_sha256"] = physical_sha
    return {
        "adaptive_pose_followup_job_schema_version": (
            ADAPTIVE_POSE_FOLLOWUP_SCHEMA_VERSION
        ),
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha,
        "physical_config_sha256": physical_sha,
        "is_exact_parent_reproduction_baseline": False,
        "job_sequence_index": int(sequence),
        "config": config,
        "job_metadata": copy.deepcopy(metadata["v14_adaptive_pose_followup"]),
    }


def build_adaptive_pose_probe_jobs(
    source: AdaptiveContactModePoseSource,
    *,
    budget: AdaptivePoseFollowupBudget = AdaptivePoseFollowupBudget(),
    validate_configs: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Build the first, one-axis-only half of the conditional follow-up."""

    if not source.centers:
        raise ValueError("adaptive pose source has no centres")
    count = min(int(budget.candidate_count), MAXIMUM_PROBE_CANDIDATE_COUNT)
    center = source.centers[0]
    samples = _probe_samples()[:count]
    limits = resolve_joint_refinement_limits(center.config)
    prior = set(source.prior_physical_config_sha256)
    generated: set[str] = set()
    identifiers: set[int] = set()
    jobs: list[dict[str, Any]] = []
    for sequence, base_sample in enumerate(samples):
        accepted: dict[str, Any] | None = None
        # A first-stage finite-difference anchor can coincide after a joint
        # limit clip.  Deterministically shrink it instead of publishing a
        # duplicate physical run.
        for scale in (1.0, 0.75, 0.5, 0.25):
            normalized = base_sample * scale
            job = _materialize_candidate(
                source,
                center,
                normalized,
                sequence,
                budget,
                limits,
                validate=validate_configs,
                stage="finite_difference_probe",
                probe_evidence_sha256=None,
            )
            physical = str(job["physical_config_sha256"])
            identifier = int(job["candidate_id"])
            if physical in prior or physical in generated or identifier in identifiers:
                continue
            accepted = job
            break
        if accepted is None:
            raise RuntimeError(
                "adaptive pose probe exhausted its deterministic unique axis shell"
            )
        generated.add(str(accepted["physical_config_sha256"]))
        identifiers.add(int(accepted["candidate_id"]))
        jobs.append(accepted)
    return tuple(jobs)


def _probe_direction_evidence(
    source: AdaptiveContactModePoseSource,
    probe_jobs: Sequence[Mapping[str, Any]],
    probe_records: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[np.ndarray, ...]]:
    if not probe_jobs or len(probe_jobs) != len(probe_records):
        raise RuntimeError("adaptive combination stage requires every probe result")
    jobs_by_id = {int(value["candidate_id"]): value for value in probe_jobs}
    records_by_id = {int(value.get("candidate_id", -1)): value for value in probe_records}
    if len(jobs_by_id) != len(probe_jobs) or set(jobs_by_id) != set(records_by_id):
        raise RuntimeError("adaptive probe evidence has missing or duplicate candidate IDs")
    if any(bool(value.get("full_success", False)) for value in probe_records):
        raise RuntimeError("adaptive combinations are unnecessary after a probe hard pass")

    eligible: list[tuple[tuple[Any, ...], int, np.ndarray, str]] = []
    evidence_records: list[dict[str, Any]] = []
    for candidate_id in sorted(jobs_by_id):
        job = jobs_by_id[candidate_id]
        record = records_by_id[candidate_id]
        expected_config_sha = canonical_sha256(job["config"])
        if str(record.get("config_semantic_sha256", "")) != expected_config_sha:
            raise RuntimeError("adaptive probe result is bound to a different config")
        result_sha = _require_sha256(
            record.get("result_semantic_sha256"),
            label=f"probe result_semantic_sha256[{candidate_id}]",
        )
        normalized = np.asarray(
            job["job_metadata"]["normalized_sample"], dtype=np.float64
        )
        if normalized.shape != (30,) or np.count_nonzero(normalized) != 1:
            raise RuntimeError("adaptive probe evidence is not one-dimensional")
        evidence_records.append(
            {
                "candidate_id": candidate_id,
                "physical_config_sha256": str(job["physical_config_sha256"]),
                "config_semantic_sha256": expected_config_sha,
                "result_semantic_sha256": result_sha,
                "normalized_sample": normalized.tolist(),
                "strict_jerk_only": _all_other_hard_checks_pass(record),
                "peak_jerk_m_s3": _peak_jerk(record),
            }
        )
        if _all_other_hard_checks_pass(record):
            axis = int(np.flatnonzero(normalized)[0])
            eligible.append((_center_rank(record), axis, normalized, result_sha))
    if len({axis for _, axis, _, _ in eligible}) < 2:
        raise RuntimeError(
            "adaptive combinations need strict jerk-only evidence on two distinct axes"
        )

    # Keep only the best sign/amplitude per dimension.  Thus a 2--4D
    # combination is composed exclusively from directions that independently
    # retained every non-jerk hard constraint.
    directions_by_axis: dict[int, tuple[tuple[Any, ...], np.ndarray]] = {}
    for rank, axis, normalized, _ in eligible:
        previous = directions_by_axis.get(axis)
        if previous is None or rank < previous[0]:
            directions_by_axis[axis] = (rank, normalized.copy())
    ordered = tuple(
        value[1]
        for _, value in sorted(
            directions_by_axis.items(), key=lambda item: (item[1][0], item[0])
        )
    )
    evidence_payload = {
        "schema_version": ADAPTIVE_POSE_FOLLOWUP_SCHEMA_VERSION,
        "source_authentication_id": source.source_authentication_id,
        "probe_records": evidence_records,
        "selected_direction_count": len(ordered),
        "selection": "best_strict_jerk_only_sign_per_axis",
    }
    return canonical_sha256(evidence_payload), ordered


def _combination_samples(
    directions: Sequence[np.ndarray], count: int, seed: int
) -> tuple[np.ndarray, ...]:
    if len(directions) < 2:
        raise ValueError("at least two safe directions are required")
    ranked = tuple(np.asarray(value, dtype=np.float64) for value in directions[:16])
    proposals: list[np.ndarray] = []
    # Deterministic low-order combinations of the best independently safe
    # directions precede random sparse coverage.
    for support in (2, 3, 4):
        if len(ranked) < support:
            continue
        for indices in itertools.combinations(range(min(len(ranked), 10)), support):
            sample = np.zeros(30, dtype=np.float64)
            for index in indices:
                sample += 0.5 * ranked[index]
            if np.count_nonzero(sample) == support and np.max(np.abs(sample)) <= 1.0:
                proposals.append(sample)
    seed_words = np.frombuffer(bytes.fromhex(canonical_sha256([value.tolist() for value in ranked]))[:16], dtype="<u4")
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), *(int(value) for value in seed_words)])
    )
    while len(proposals) < max(count * 12, 96):
        support = int(rng.integers(2, min(4, len(ranked)) + 1))
        indices = rng.choice(len(ranked), size=support, replace=False)
        scales = rng.uniform(0.25, 0.80, size=support)
        sample = np.zeros(30, dtype=np.float64)
        for index, scale in zip(indices, scales):
            sample += float(scale) * ranked[int(index)]
        if 2 <= np.count_nonzero(sample) <= 4 and np.max(np.abs(sample)) <= 1.0:
            proposals.append(sample)
    unique: list[np.ndarray] = []
    digests: set[str] = set()
    for value in proposals:
        digest = canonical_sha256(value.tolist())
        if digest not in digests:
            unique.append(value)
            digests.add(digest)
    return tuple(unique)


def build_adaptive_pose_combination_jobs(
    source: AdaptiveContactModePoseSource,
    probe_jobs: Sequence[Mapping[str, Any]],
    probe_records: Sequence[Mapping[str, Any]],
    *,
    budget: AdaptivePoseFollowupBudget = AdaptivePoseFollowupBudget(),
    validate_configs: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Build 2--4D sparse combinations after the finite-difference results."""

    probe_count = min(int(budget.candidate_count), MAXIMUM_PROBE_CANDIDATE_COUNT)
    combination_count = int(budget.candidate_count) - probe_count
    if combination_count <= 0:
        return ()
    if len(probe_jobs) != probe_count:
        raise RuntimeError("adaptive combination stage received the wrong probe count")
    evidence_sha, directions = _probe_direction_evidence(
        source, probe_jobs, probe_records
    )
    center = source.centers[0]
    limits = resolve_joint_refinement_limits(center.config)
    samples = _combination_samples(directions, combination_count, int(budget.seed))
    excluded = set(source.prior_physical_config_sha256)
    excluded.update(str(value["physical_config_sha256"]) for value in probe_jobs)
    generated: set[str] = set()
    identifiers: set[int] = set()
    jobs: list[dict[str, Any]] = []
    for normalized in samples:
        sequence = probe_count + len(jobs)
        job = _materialize_candidate(
            source,
            center,
            normalized,
            sequence,
            budget,
            limits,
            validate=validate_configs,
            stage="safe_sparse_combination",
            probe_evidence_sha256=evidence_sha,
        )
        physical = str(job["physical_config_sha256"])
        candidate_id = int(job["candidate_id"])
        if physical in excluded or physical in generated or candidate_id in identifiers:
            continue
        generated.add(physical)
        identifiers.add(candidate_id)
        jobs.append(job)
        if len(jobs) == combination_count:
            break
    if len(jobs) != combination_count:
        raise RuntimeError("adaptive sparse combinations exhausted unique proposals")
    return tuple(jobs)


__all__ = [
    "ADAPTIVE_POSE_FOLLOWUP_SCHEMA_VERSION",
    "AdaptiveContactModeCenter",
    "AdaptiveContactModePoseSource",
    "AdaptivePoseFollowupBudget",
    "DEFAULT_CENTER_COUNT",
    "EXPECTED_FIRST_STAGE_CANDIDATE_COUNT",
    "MAXIMUM_PROBE_CANDIDATE_COUNT",
    "MAXIMUM_FOLLOWUP_CANDIDATE_COUNT",
    "build_adaptive_pose_combination_jobs",
    "build_adaptive_pose_probe_jobs",
    "build_adaptive_pose_source",
]
