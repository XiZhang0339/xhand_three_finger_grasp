"""Small, trace-aware grasp-pose rescue for a schema-v14 lift.

The 79 mm micro-jerk evidence has a strict grasp and continuous three-finger
contact, but its thumb pad changes active-taxel branch during MANIPULATE.  A
plan-only search cannot move that boundary.  This module therefore makes a
small, deterministic search around the *contact mode*: fixed cube and fixed
3 s/20-segment manipulation plan, with only the hand root, measured grasp
qpos, precontact qpos and preload command allowed to move.

It deliberately has no CLI and no bespoke simulation loop.  Candidate
execution delegates to the existing atomic schema-v14 artifact runner.  This
keeps legacy paths untouched and makes every result a full reset with a saved
trace, which is required to rank taxel-branch stability rather than infer it
from a semantic summary.
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import file_sha256
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
from .contact_preserving_candidate_artifacts import (
    authenticate_v14_candidate_artifacts,
    run_or_resume_v14_candidate_artifacts,
)
from .contact_preserving_joint_refinement import resolve_joint_refinement_limits
from .contact_preserving_joint_refinement import JointRefinementLimits
from .contact_preserving_time_warp import _time_warp_controller_id


CONTACT_MODE_POSE_RESCUE_SCHEMA_VERSION = 1
EXPERIMENT_ID = "left_opposed_face_palm_down_contact_preserving_planned_lift"
DEFAULT_SEED = 20260821
MAXIMUM_CANDIDATE_COUNT = 256
_JERK_CHECK = "smooth_motion_jerk_within_limit"


@dataclass(frozen=True, slots=True)
class ContactModePoseRescueBudget:
    """Bounded 30-D search around one authenticated contact mode."""

    candidate_count: int = MAXIMUM_CANDIDATE_COUNT
    seed: int = DEFAULT_SEED
    root_translation_radius_cube_m: tuple[float, float, float] = (
        0.00020,
        0.00020,
        0.00100,
    )
    root_z_negative_continuation_m: tuple[float, float, float, float] = (
        -0.00040,
        -0.00060,
        -0.00080,
        -0.00100,
    )
    wrist_local_rotvec_radius_deg: tuple[float, float, float] = (
        0.15,
        0.15,
        0.15,
    )
    wrist_local_rotvec_norm_limit_deg: float = 0.20
    nominal_qpos_radius_rad: float = 0.0020
    precontact_residual_radius_rad: float = 0.0010
    preload_residual_radius_rad: float = 0.0015

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_count, bool)
            or not 1 <= int(self.candidate_count) <= MAXIMUM_CANDIDATE_COUNT
        ):
            raise ValueError(
                f"candidate_count must lie within [1, {MAXIMUM_CANDIDATE_COUNT}]"
            )
        if isinstance(self.seed, bool) or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in (
            "root_translation_radius_cube_m",
            "wrist_local_rotvec_radius_deg",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{name} must contain three finite non-negative values")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        continuation = np.asarray(
            self.root_z_negative_continuation_m, dtype=np.float64
        )
        if (
            continuation.shape != (4,)
            or not np.isfinite(continuation).all()
            or not np.all(np.diff(continuation) < 0.0)
            or continuation[0] >= 0.0
            or continuation[-1] < -self.root_translation_radius_cube_m[2] - 1e-15
        ):
            raise ValueError(
                "root_z_negative_continuation_m must be four descending in-bound negative offsets"
            )
        object.__setattr__(
            self,
            "root_z_negative_continuation_m",
            tuple(float(value) for value in continuation),
        )
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
            "schema_version": CONTACT_MODE_POSE_RESCUE_SCHEMA_VERSION,
            "candidate_count": int(self.candidate_count),
            "maximum_candidate_count": MAXIMUM_CANDIDATE_COUNT,
            "seed": int(self.seed),
            "root_translation_radius_cube_m": list(
                self.root_translation_radius_cube_m
            ),
            "root_z_negative_continuation_m": list(
                self.root_z_negative_continuation_m
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


@dataclass(frozen=True, slots=True)
class ContactModePoseSource:
    """Authenticated immutable source and its retained trace evidence."""

    root: Path
    candidate_id: int
    config: dict[str, Any]
    result: dict[str, Any]
    config_sha256: str
    result_semantic_sha256: str
    trace_sha256: str
    source_authentication_id: str
    trace_diagnostics: dict[str, Any]

    @property
    def trace_path(self) -> Path:
        return self.root / "trace.npz"

    def as_mapping(self, *, include_payloads: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": CONTACT_MODE_POSE_RESCUE_SCHEMA_VERSION,
            "root": str(self.root),
            "candidate_id": self.candidate_id,
            "config_sha256": self.config_sha256,
            "result_semantic_sha256": self.result_semantic_sha256,
            "trace_sha256": self.trace_sha256,
            "source_authentication_id": self.source_authentication_id,
            "trace_diagnostics": copy.deepcopy(self.trace_diagnostics),
        }
        if include_payloads:
            value.update(config=copy.deepcopy(self.config), result=copy.deepcopy(self.result))
        return value


def _failed_checks(result: Mapping[str, Any]) -> tuple[str, ...]:
    raw = result.get("summary", {}).get("failed_checks", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ("malformed_failed_checks",)
    return tuple(str(value) for value in raw)


def _phase_mask(trace: Any, phase: str) -> np.ndarray:
    states = np.asarray(trace["control_state"]).astype(str)
    mask = states == phase
    if mask.ndim != 1 or not np.any(mask):
        raise ValueError(f"trace contains no {phase} samples")
    indices = np.flatnonzero(mask)
    if indices[-1] - indices[0] + 1 != indices.size:
        raise ValueError(f"trace {phase} samples are not contiguous")
    return mask


def contact_mode_trace_diagnostics(trace: Any) -> dict[str, Any]:
    """Measure active-taxel branch stability from one complete raw trace.

    The saved trace currently exposes active taxel *counts*, not individual
    taxel identities.  Count changes are therefore a conservative observable
    branch transition.  A simultaneous contact-centroid jump is recorded as a
    second branch-risk signal, so a nominal 1-to-1 identity swap cannot look
    perfectly stable merely because the count stayed one.
    """

    states = np.asarray(trace["control_state"]).astype(str)
    if states.ndim != 1:
        raise ValueError("control_state must be one-dimensional")
    if not np.any(states == "MANIPULATE"):
        return {
            "schema_version": CONTACT_MODE_POSE_RESCUE_SCHEMA_VERSION,
            "measurement_phase": "MANIPULATE",
            "measurement_available": False,
            "sample_count": 0,
            "taxel_count_transition_count": {
                "thumb": 0,
                "index": 0,
                "mid": 0,
            },
            "taxel_count_transition_steps": {
                "thumb": [],
                "index": [],
                "mid": [],
            },
            "taxel_count_modal_duty": {
                "thumb": 0.0,
                "index": 0.0,
                "mid": 0.0,
            },
            "contact_centroid_max_step_m": {
                "thumb": 0.0,
                "index": 0.0,
                "mid": 0.0,
            },
            "contact_centroid_large_step_count": {
                "thumb": 0,
                "index": 0,
                "mid": 0,
            },
            "per_finger_effective_contact_duty": {
                "thumb": 0.0,
                "index": 0.0,
                "mid": 0.0,
            },
            "simultaneous_effective_contact_duty": 0.0,
            "peak_abs_filtered_jerk_m_s3": None,
            "peak_abs_filtered_jerk_step": -1,
        }
    mask = _phase_mask(trace, "MANIPULATE")
    indices = np.flatnonzero(mask)
    counts = np.asarray(trace["distal_active_taxel_count"], dtype=np.int64)
    centroids = np.asarray(
        trace["target_face_contact_centroid_cube_local_m"], dtype=np.float64
    )
    valid = np.asarray(trace["target_face_contact_centroid_valid"], dtype=bool)
    effective = np.asarray(trace["target_face_effective"], dtype=bool)
    jerk = np.asarray(
        trace["operation_vertical_jerk_filtered_m_s3"], dtype=np.float64
    )
    if (
        counts.shape != (mask.size, 3)
        or centroids.shape != (mask.size, 3, 3)
        or valid.shape != (mask.size, 3)
        or effective.shape != (mask.size, 3)
        or jerk.shape != (mask.size,)
    ):
        raise ValueError("contact-mode trace fields have inconsistent shapes")
    operation_counts = counts[mask]
    transitions = np.count_nonzero(np.diff(operation_counts, axis=0), axis=0)
    modal_duty: list[float] = []
    for finger in range(3):
        values, frequency = np.unique(
            operation_counts[:, finger], return_counts=True
        )
        del values
        modal_duty.append(float(np.max(frequency) / operation_counts.shape[0]))
    centroid_max_step: list[float] = []
    centroid_large_step_count: list[int] = []
    operation_centroids = centroids[mask]
    operation_valid = valid[mask]
    for finger in range(3):
        pair_valid = operation_valid[1:, finger] & operation_valid[:-1, finger]
        steps = np.linalg.norm(
            np.diff(operation_centroids[:, finger, :], axis=0), axis=1
        )
        finite_steps = steps[pair_valid & np.isfinite(steps)]
        centroid_max_step.append(
            float(np.max(finite_steps, initial=0.0))
        )
        # 0.10 mm per 1 ms is far above ordinary contact-point drift and is a
        # useful identity-swap proxy when only one taxel remains active.
        centroid_large_step_count.append(int(np.count_nonzero(finite_steps > 0.00010)))
    operation_jerk = np.abs(jerk[mask])
    peak_local = int(np.argmax(operation_jerk))
    peak_global = int(indices[peak_local])
    transition_steps = {
        finger: [
            int(indices[index + 1])
            for index in np.flatnonzero(
                operation_counts[1:, finger_index]
                != operation_counts[:-1, finger_index]
            )
        ]
        for finger_index, finger in enumerate(("thumb", "index", "mid"))
    }
    return {
        "schema_version": CONTACT_MODE_POSE_RESCUE_SCHEMA_VERSION,
        "measurement_phase": "MANIPULATE",
        "measurement_available": True,
        "sample_count": int(indices.size),
        "taxel_count_transition_count": {
            finger: int(transitions[index])
            for index, finger in enumerate(("thumb", "index", "mid"))
        },
        "taxel_count_transition_steps": transition_steps,
        "taxel_count_modal_duty": {
            finger: modal_duty[index]
            for index, finger in enumerate(("thumb", "index", "mid"))
        },
        "contact_centroid_max_step_m": {
            finger: centroid_max_step[index]
            for index, finger in enumerate(("thumb", "index", "mid"))
        },
        "contact_centroid_large_step_count": {
            finger: centroid_large_step_count[index]
            for index, finger in enumerate(("thumb", "index", "mid"))
        },
        "per_finger_effective_contact_duty": {
            finger: float(np.mean(effective[mask, index]))
            for index, finger in enumerate(("thumb", "index", "mid"))
        },
        "simultaneous_effective_contact_duty": float(
            np.mean(np.all(effective[mask], axis=1))
        ),
        "peak_abs_filtered_jerk_m_s3": float(operation_jerk[peak_local]),
        "peak_abs_filtered_jerk_step": peak_global,
    }


def authenticate_contact_mode_pose_source(
    source_directory: str | Path,
) -> ContactModePoseSource:
    """Authenticate the retained 79 mm jerk-only near miss, fail closed."""

    root = Path(source_directory).expanduser().resolve()
    bundle = authenticate_v14_candidate_artifacts(
        root, require_retained_trace=True
    )
    config = copy.deepcopy(json.loads(bundle.config_path.read_text(encoding="utf-8")))
    result = copy.deepcopy(bundle.result)
    cube = config.get("cube", {})
    plan = config.get("manipulation_plan", {})
    if config.get("experiment_id") != EXPERIMENT_ID or int(config.get("schema_version", 0)) != 14:
        raise RuntimeError("contact-mode rescue source is not the registered schema-v14 experiment")
    if not (
        abs(float(cube.get("edge_m", math.nan)) - 0.079) <= 1e-12
        and abs(float(cube.get("mass_kg", math.nan)) - 0.160) <= 1e-12
        and abs(float(cube.get("friction", math.nan)) - 0.8) <= 1e-12
    ):
        raise RuntimeError("contact-mode rescue source changed its 79 mm/160 g/mu=0.8 object")
    if not (
        abs(float(plan.get("duration_s", math.nan)) - 3.0) <= 1e-12
        and len(plan.get("knot_times_s", ())) == 21
    ):
        raise RuntimeError("contact-mode rescue source changed its 3 s/20-segment plan")
    if result.get("grasp_success") is not True or _failed_checks(result) != (_JERK_CHECK,):
        raise RuntimeError("contact-mode rescue source is not a strict jerk-only grasp success")
    assert bundle.trace_path is not None
    with np.load(bundle.trace_path, allow_pickle=False) as trace:
        diagnostics = contact_mode_trace_diagnostics(trace)
    if diagnostics.get("measurement_available") is not True:
        raise RuntimeError("contact-mode rescue source lost its MANIPULATE trace")
    if diagnostics["simultaneous_effective_contact_duty"] < 0.99:
        raise RuntimeError("contact-mode rescue source lost continuous three-finger contact")
    if diagnostics["taxel_count_transition_count"]["thumb"] <= 0:
        raise RuntimeError("contact-mode rescue source no longer exhibits a thumb branch transition")
    config_sha = str(result["config_semantic_sha256"])
    result_sha = str(result["result_semantic_sha256"])
    trace_sha = file_sha256(bundle.trace_path)
    identity = canonical_sha256(
        {
            "schema_version": CONTACT_MODE_POSE_RESCUE_SCHEMA_VERSION,
            "candidate_id": bundle.candidate_id,
            "config_sha256": config_sha,
            "result_semantic_sha256": result_sha,
            "trace_sha256": trace_sha,
            "trace_diagnostics": diagnostics,
        }
    )
    return ContactModePoseSource(
        root=root,
        candidate_id=bundle.candidate_id,
        config=config,
        result=result,
        config_sha256=config_sha,
        result_semantic_sha256=result_sha,
        trace_sha256=trace_sha,
        source_authentication_id=identity,
        trace_diagnostics=diagnostics,
    )


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=np.float64)
    for column in range(dimensions):
        order = rng.permutation(count)
        result[:, column] = 2.0 * ((order + rng.random(count)) / count) - 1.0
    result[0] = 0.0
    return result


def _normalized_samples(
    source: ContactModePoseSource, budget: ContactModePoseRescueBudget
) -> np.ndarray:
    count = int(budget.candidate_count)
    seed_words = np.frombuffer(
        bytes.fromhex(source.source_authentication_id)[:16], dtype="<u4"
    )
    resolved_seed = int(
        np.random.SeedSequence([budget.seed, *(int(value) for value in seed_words)])
        .generate_state(1)[0]
    )
    result = _latin_hypercube(count, 30, resolved_seed)
    anchors: list[np.ndarray] = [np.zeros(30, dtype=np.float64)]
    # X/Y keep the original +/-0.2 mm diagnostic shell.  Z has a measured
    # monotone improvement toward the negative boundary, so preserve +0.2 mm
    # as a control and explicitly continue through -1.0 mm.
    for axis in range(2):
        for sign in (-1.0, 1.0):
            value = np.zeros(30)
            value[axis] = sign
            anchors.append(value)
    z_radius = budget.root_translation_radius_cube_m[2]
    for offset in (0.00020, -0.00020, *budget.root_z_negative_continuation_m):
        value = np.zeros(30)
        value[2] = float(offset) / z_radius
        anchors.append(value)
    for start, width in ((3, 3),):
        for axis in range(width):
            for sign in (-1.0, 1.0):
                value = np.zeros(30)
                value[start + axis] = sign
                anchors.append(value)
    # Put the three thumb joints on both sides of the contact-mode boundary.
    for start in (6, 14, 22):
        for axis in range(3):
            for sign in (-1.0, 1.0):
                value = np.zeros(30)
                value[start + axis] = sign
                anchors.append(value)
    for index, value in enumerate(anchors[:count]):
        result[index] = value
    # Remaining combinations concentrate on the measured improving side of Z:
    # [-1.0, -0.2] mm with quadratic density near -0.2 mm.  Other 29
    # dimensions retain stratified symmetric coverage.
    for index in range(min(len(anchors), count), count):
        unit = 0.5 * (result[index, 2] + 1.0)
        result[index, 2] = -0.2 - 0.8 * unit**2
    return result


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


def _candidate_id(
    source: ContactModePoseSource, local_index: int, normalized: np.ndarray
) -> tuple[int, str]:
    digest = canonical_sha256(
        {
            "schema_version": CONTACT_MODE_POSE_RESCUE_SCHEMA_VERSION,
            "source_authentication_id": source.source_authentication_id,
            "local_index": int(local_index),
            "normalized_sample": [float(value) for value in normalized],
        }
    )
    return 15_600_000_000_000_000 + int(digest[:13], 16) % 100_000_000_000_000, digest


def contact_mode_physical_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash every runtime-relevant field while excluding provenance IDs."""

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
    return canonical_sha256(payload)


def _rotation_matrix_to_rotvec_degrees(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    sine = math.sin(angle)
    if abs(sine) <= 1e-10:
        # These rescues are capped at 0.2 degrees, so this branch is only a
        # defensive guard against malformed callers rather than a real path.
        raise ValueError("contact-mode rescue rotation is numerically singular")
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sine)
    return np.degrees(axis * angle)


def _materialize_candidate(
    source: ContactModePoseSource,
    normalized: np.ndarray,
    local_index: int,
    budget: ContactModePoseRescueBudget,
    *,
    validate: bool,
    limits: JointRefinementLimits,
) -> dict[str, Any]:
    config = copy.deepcopy(source.config)
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
    norm = float(np.linalg.norm(local_rotvec))
    if norm > budget.wrist_local_rotvec_norm_limit_deg:
        local_rotvec *= budget.wrist_local_rotvec_norm_limit_deg / norm
    cube_rotation = rpy_degrees_to_rotation_matrix(config["cube"]["rpy_deg"])
    base_translation = np.asarray(config["hand_pose"]["translation_m"], dtype=np.float64)
    base_rpy = np.asarray(config["hand_pose"]["rpy_deg"], dtype=np.float64)
    base_rotation = rpy_degrees_to_rotation_matrix(base_rpy)
    config["hand_pose"]["translation_m"] = (
        base_translation + cube_rotation @ translation_delta_cube
    ).tolist()
    definition = resolve_experiment(config)
    if float(np.linalg.norm(local_rotvec)) <= 1e-15:
        proposed_rpy = base_rpy.copy()
        effective_local_rotvec = np.zeros(3, dtype=np.float64)
    else:
        new_rotation = base_rotation @ rotvec_degrees_to_rotation_matrix(local_rotvec)
        proposed_rpy = rotation_matrix_to_rpy_degrees(
            new_rotation, reference_rpy_deg=base_rpy
        )
        proposed_rpy[0] = np.clip(
            proposed_rpy[0], *definition.search_bounds.hand_roll_deg
        )
        proposed_rpy[2] = np.clip(
            proposed_rpy[2], *definition.search_bounds.hand_yaw_deg
        )
        effective_rotation = rpy_degrees_to_rotation_matrix(proposed_rpy)
        effective_local_rotvec = _rotation_matrix_to_rotvec_degrees(
            base_rotation.T @ effective_rotation
        )
        effective_norm = float(np.linalg.norm(effective_local_rotvec))
        if effective_norm > budget.wrist_local_rotvec_norm_limit_deg:
            effective_local_rotvec *= (
                budget.wrist_local_rotvec_norm_limit_deg / effective_norm
            )
            effective_rotation = base_rotation @ rotvec_degrees_to_rotation_matrix(
                effective_local_rotvec
            )
            proposed_rpy = rotation_matrix_to_rpy_degrees(
                effective_rotation, reference_rpy_deg=base_rpy
            )
            proposed_rpy[0] = np.clip(
                proposed_rpy[0], *definition.search_bounds.hand_roll_deg
            )
            proposed_rpy[2] = np.clip(
                proposed_rpy[2], *definition.search_bounds.hand_yaw_deg
            )
            effective_rotation = rpy_degrees_to_rotation_matrix(proposed_rpy)
            effective_local_rotvec = _rotation_matrix_to_rotvec_degrees(
                base_rotation.T @ effective_rotation
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
                float(base_precontact[name])
                + applied_nominal
                + precontact_residual[index],
                command_low,
                command_high,
            )
        )
        preload[name] = float(
            np.clip(
                float(base_preload[name])
                + applied_nominal
                + preload_residual[index],
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
    candidate_id, candidate_sha = _candidate_id(source, local_index, normalized)
    planner_payload = {
        "schema_version": CONTACT_MODE_POSE_RESCUE_SCHEMA_VERSION,
        "kind": "v14_contact_mode_pose_rescue_reused_plan",
        "source_authentication_id": source.source_authentication_id,
        "source_plan_id": base_plan["plan_id"],
        "grasp_object_pair_id": config["grasp_object_pair_id"],
        "candidate_sha256": candidate_sha,
    }
    config["planner_id"] = canonical_sha256(planner_payload)
    config["controller_id"] = _time_warp_controller_id(config)
    metadata = config.setdefault("candidate_metadata", {})
    metadata["v14_contact_mode_pose_rescue"] = {
        "schema_version": CONTACT_MODE_POSE_RESCUE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha,
        "local_index": int(local_index),
        "source_candidate_id": source.candidate_id,
        "source_authentication_id": source.source_authentication_id,
        "root_translation_delta_cube_m": translation_delta_cube.tolist(),
        "requested_wrist_local_rotvec_deg": local_rotvec.tolist(),
        "wrist_local_rotvec_deg": effective_local_rotvec.tolist(),
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
        "full_reset_required": True,
        "taxel_branch_stability_is_trace_ranked": True,
    }
    if config["cube"] != base_cube:
        raise AssertionError("contact-mode rescue changed the cube configuration")
    if config["manipulation_plan"] != base_plan or config["control"]["manipulation_delta_rad"] != base_terminal:
        raise AssertionError("contact-mode rescue changed the fixed manipulation plan")
    if validate:
        validate_config(config)
    physical_sha256 = contact_mode_physical_config_sha256(config)
    source_physical_sha256 = contact_mode_physical_config_sha256(source.config)
    exact_parent_baseline = physical_sha256 == source_physical_sha256
    metadata["v14_contact_mode_pose_rescue"].update(
        {
            "physical_config_sha256": physical_sha256,
            "source_physical_config_sha256": source_physical_sha256,
            "is_exact_parent_reproduction_baseline": exact_parent_baseline,
        }
    )
    return {
        "contact_mode_pose_rescue_job_schema_version": (
            CONTACT_MODE_POSE_RESCUE_SCHEMA_VERSION
        ),
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha,
        "physical_config_sha256": physical_sha256,
        "is_exact_parent_reproduction_baseline": exact_parent_baseline,
        "job_sequence_index": int(local_index),
        "config": config,
        "job_metadata": copy.deepcopy(
            config["candidate_metadata"]["v14_contact_mode_pose_rescue"]
        ),
    }


def build_contact_mode_pose_rescue_jobs(
    source: ContactModePoseSource,
    *,
    budget: ContactModePoseRescueBudget = ContactModePoseRescueBudget(),
    validate_configs: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Build at most 256 worker-order-independent full-reset jobs."""

    samples = _normalized_samples(source, budget)
    limits = resolve_joint_refinement_limits(source.config)
    jobs = tuple(
        _materialize_candidate(
            source,
            samples[index],
            index,
            budget,
            validate=validate_configs,
            limits=limits,
        )
        for index in range(int(budget.candidate_count))
    )
    identifiers = [int(value["candidate_id"]) for value in jobs]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("contact-mode pose-rescue candidate ID collision")
    physical = [str(value["physical_config_sha256"]) for value in jobs]
    if len(set(physical)) != len(physical):
        raise RuntimeError("contact-mode pose-rescue physical candidate collision")
    baseline = [
        value for value in jobs if value["is_exact_parent_reproduction_baseline"]
    ]
    if len(baseline) != 1 or int(baseline[0]["job_sequence_index"]) != 0:
        raise RuntimeError("contact-mode rescue must contain exactly one exact-parent baseline")
    return jobs


def _peak_jerk(record: Mapping[str, Any]) -> float:
    try:
        value = record["summary"]["metrics"]["motion_smoothness"][
            "operation_peak_abs_filtered_jerk_m_s3"
        ]
        return float(value) if math.isfinite(float(value)) else math.inf
    except (KeyError, TypeError, ValueError):
        return math.inf


def contact_mode_pose_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Contact-mode-first order; lift amount never precedes branch stability."""

    diagnostics = record.get("contact_mode_diagnostics", {})
    transitions = diagnostics.get("taxel_count_transition_count", {})
    centroids = diagnostics.get("contact_centroid_max_step_m", {})
    failed = _failed_checks(record)
    non_jerk_failures = tuple(value for value in failed if value != _JERK_CHECK)
    return (
        not bool(record.get("full_success", False)),
        not bool(record.get("grasp_success", False)),
        len(non_jerk_failures),
        not bool(diagnostics.get("measurement_available", False)),
        int(transitions.get("thumb", 10**9)),
        _peak_jerk(record),
        int(sum(int(transitions.get(name, 10**8)) for name in ("index", "mid"))),
        -float(diagnostics.get("simultaneous_effective_contact_duty", -math.inf)),
        float(centroids.get("thumb", math.inf)),
        int(
            diagnostics.get("contact_centroid_large_step_count", {}).get(
                "thumb", 10**9
            )
        ),
        int(record.get("candidate_id", 2**63 - 1)),
    )


def rank_contact_mode_pose_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    values = [copy.deepcopy(dict(value)) for value in records]
    values.sort(key=contact_mode_pose_candidate_rank)
    return tuple(values)


def _execute_job(payload: Mapping[str, Any]) -> dict[str, Any]:
    bundle = run_or_resume_v14_candidate_artifacts(
        payload["config"],
        payload["destination"],
        int(payload["candidate_id"]),
        final_rerun=True,
        retain_grasp_success=False,
    )
    if bundle.trace_path is None:
        raise RuntimeError("contact-mode rescue final rerun did not retain its trace")
    with np.load(bundle.trace_path, allow_pickle=False) as trace:
        diagnostics = contact_mode_trace_diagnostics(trace)
    return {
        **copy.deepcopy(bundle.result),
        "artifact_directory": str(bundle.destination),
        "contact_mode_diagnostics": diagnostics,
        "physical_config_sha256": contact_mode_physical_config_sha256(
            payload["config"]
        ),
        "is_exact_parent_reproduction_baseline": bool(
            payload.get("is_exact_parent_reproduction_baseline", False)
        ),
    }


def run_contact_mode_pose_rescue_jobs(
    jobs: Sequence[Mapping[str, Any]],
    output_directory: str | Path,
    *,
    workers: int = 1,
) -> tuple[dict[str, Any], ...]:
    """Execute immutable jobs with the existing atomic v14 full-reset runner."""

    if isinstance(workers, bool) or int(workers) <= 0:
        raise ValueError("workers must be a positive integer")
    if not 1 <= len(jobs) <= MAXIMUM_CANDIDATE_COUNT:
        raise ValueError("contact-mode rescue execution requires 1..256 jobs")
    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        (copy.deepcopy(dict(value)) for value in jobs),
        key=lambda value: (
            int(value["job_sequence_index"]), int(value["candidate_id"])
        ),
    )
    identifiers = [int(value["candidate_id"]) for value in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("contact-mode rescue jobs contain duplicate candidate IDs")
    payloads = [
        {
            "config": value["config"],
            "candidate_id": int(value["candidate_id"]),
            "is_exact_parent_reproduction_baseline": bool(
                value.get("is_exact_parent_reproduction_baseline", False)
            ),
            "destination": str(root / f"candidate_{int(value['candidate_id'])}"),
        }
        for value in ordered
    ]
    if int(workers) == 1:
        records = tuple(_execute_job(value) for value in payloads)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as pool:
            records = tuple(pool.map(_execute_job, payloads))
    return rank_contact_mode_pose_records(records)


__all__ = [
    "CONTACT_MODE_POSE_RESCUE_SCHEMA_VERSION",
    "ContactModePoseRescueBudget",
    "ContactModePoseSource",
    "MAXIMUM_CANDIDATE_COUNT",
    "authenticate_contact_mode_pose_source",
    "build_contact_mode_pose_rescue_jobs",
    "contact_mode_pose_candidate_rank",
    "contact_mode_physical_config_sha256",
    "contact_mode_trace_diagnostics",
    "rank_contact_mode_pose_records",
    "run_contact_mode_pose_rescue_jobs",
]
