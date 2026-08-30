"""Deterministic dynamic-contact-guided refinement for schema-v11.

The module is intentionally independent of the recovery orchestrator.  It
authenticates existing dynamic evidence, extracts signed contact-centroid
errors, proposes bounded pose/controller changes, and gates every new pose
through the shared static/full-scene evaluator before any dynamic run.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import file_sha256
from ..config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, validate_config
from ..experiment import resolve_experiment
from ..grasp_pose import controller_id, grasp_pose_id
from .actual_contact_grasp_pose import apply_precontact_solution
from .pose_preserving_seed_campaign import canonical_sha256
from .relative_wrist_pose_active_set import (
    ActiveSetEvaluationContext,
    build_active_set_evaluation_context,
    evaluate_active_set_candidate,
    solve_orientation_aware_active_set_dls,
)
from .relative_wrist_pose_search import (
    NON_THUMB_ACTUATORS,
    RelativeWristDLSSettings,
    RelativeWristPoseSearchPolicy,
    RelativeWristVariables,
    materialize_relative_wrist_candidate,
)


POSE_DIMENSION = 13
CONTROL_DIMENSION = 15
VECTOR_DIMENSION = POSE_DIMENSION + CONTROL_DIMENSION
RESPONSE_DIMENSION = 11
GUIDED_ID_BASE = 4_300_000_000_000_000_000
_EPS = 1e-12

VECTOR_NAMES = (
    *NON_THUMB_ACTUATORS,
    "root_delta_cube_m.x",
    "root_delta_cube_m.y",
    "root_delta_cube_m.z",
    "wrist_local_rotvec_rad.x",
    "wrist_local_rotvec_rad.y",
    "wrist_local_rotvec_rad.z",
    *(f"preload.{name}" for name in ACTIVE_ACTUATORS),
    *(f"profile_start.{finger}" for finger in ACTIVE_FINGERS),
    *(f"profile_end.{finger}" for finger in ACTIVE_FINGERS),
    "close_duration_s",
)


@dataclass(frozen=True, slots=True)
class GuidedRefinementPolicy:
    parent_count: int = 3
    guided_per_parent: int = 7
    coordinate_count_per_parent: int = 12
    second_round_per_parent: int = 6
    controller_per_parent: int = 8
    maximum_dynamic_budget: int = 99
    ridge: float = 0.05
    maximum_static_iterations: int = 4
    joint_trust_rad: float = 0.004
    translation_trust_m: float = 0.0003
    rotation_trust_rad: float = math.radians(0.3)

    def __post_init__(self) -> None:
        integer_names = (
            "parent_count",
            "guided_per_parent",
            "coordinate_count_per_parent",
            "second_round_per_parent",
            "controller_per_parent",
            "maximum_dynamic_budget",
            "maximum_static_iterations",
        )
        if any(
            not isinstance(getattr(self, name), int)
            or isinstance(getattr(self, name), bool)
            or getattr(self, name) <= 0
            for name in integer_names
        ):
            raise ValueError("guided refinement integer budgets must be positive")
        declared = self.parent_count * (
            self.guided_per_parent
            + self.coordinate_count_per_parent
            + self.second_round_per_parent
            + self.controller_per_parent
        )
        if declared > self.maximum_dynamic_budget:
            raise ValueError("guided refinement sub-budgets exceed maximum budget")
        if self.ridge <= 0.0 or any(
            value <= 0.0
            for value in (
                self.joint_trust_rad,
                self.translation_trust_m,
                self.rotation_trust_rad,
            )
        ):
            raise ValueError("ridge and trust radii must be positive")


@dataclass(frozen=True, slots=True)
class DynamicContactObservation:
    artifact_key: str
    candidate_id: int
    edge_m: float
    clockwise_orbit_deg: float
    grasp_success: bool
    effective_finger_count: int
    simultaneous_target_face_duty: float
    signed_height_error_m: tuple[float, float]
    height_spread_p95_m: float
    cube_translation_m: tuple[float, float, float]
    cube_orientation_rotvec_rad: tuple[float, float, float]
    thumb_actual_range_duty: float
    force_log_ratio: tuple[float, float]
    first_contact_step: tuple[int, int, int]
    maximum_consecutive_gate_steps: int
    trace_sample_count: int
    maximum_translation_m: float | None = None
    maximum_orientation_drift_rad: float | None = None

    def response_vector(self) -> np.ndarray:
        return np.asarray(
            (
                *self.signed_height_error_m,
                *self.cube_translation_m,
                *self.cube_orientation_rotvec_rad,
                *self.force_log_ratio,
                max(0.0, 1.0 - self.thumb_actual_range_duty),
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class RidgeSVDModel:
    coefficient: np.ndarray
    intercept: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    singular_values: np.ndarray
    ridge: float

    def predict(self, values: Sequence[float]) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != self.center.shape:
            raise ValueError("ridge prediction vector shape changed")
        return self.intercept + ((vector - self.center) / self.scale) @ self.coefficient


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
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


def _quat_to_rotvec(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("cube quaternion must contain four finite values")
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    norm = float(np.linalg.norm(q[1:]))
    if norm <= _EPS:
        return np.zeros(3)
    angle = 2.0 * math.atan2(norm, float(q[0]))
    return q[1:] / norm * angle


def _relative_quat_rotvec(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    conjugate = reference.copy()
    conjugate[1:] *= -1.0
    w1, x1, y1, z1 = current
    w2, x2, y2, z2 = conjugate
    relative = np.asarray(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )
    return _quat_to_rotvec(relative)


def authenticate_dynamic_record(
    record: Mapping[str, Any], source_dynamic_root: str | Path
) -> dict[str, Any]:
    """Authenticate config/result/retained trace and return normalized evidence."""

    root = Path(source_dynamic_root).expanduser().resolve()
    directory = Path(str(record["artifact_directory"]))
    if not directory.is_absolute():
        directory = root / directory
    result_path = directory / "result.json"
    config_path = directory / "resolved_config.json"
    trace_path = directory / "trace.npz"
    if not all(path.is_file() for path in (result_path, config_path)):
        raise RuntimeError(f"guided dynamic evidence is incomplete: {directory}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if result.get("complete") is not True:
        raise RuntimeError(f"guided dynamic result is incomplete: {result_path}")
    candidate_id = int(record["candidate_id"])
    if int(result.get("candidate_id", -1)) != candidate_id:
        raise RuntimeError("guided dynamic candidate id changed")
    candidate_sha = canonical_sha256(config)
    if candidate_sha != str(record["candidate_sha256"]) or candidate_sha != str(
        result.get("candidate_sha256")
    ):
        raise RuntimeError("guided dynamic config semantic hash changed")
    artifacts = result.get("artifacts", {})
    hashes = artifacts.get("sha256", {}) if isinstance(artifacts, Mapping) else {}
    if hashes.get("resolved_config") != file_sha256(config_path):
        raise RuntimeError("guided dynamic persisted config hash changed")
    trace_retained = bool(artifacts.get("trace_retained", True))
    if trace_retained:
        if not trace_path.is_file() or hashes.get("trace") != file_sha256(trace_path):
            raise RuntimeError("guided dynamic trace hash changed")
    elif trace_path.exists():
        raise RuntimeError("compacted guided record unexpectedly retained trace")
    if result.get("grasp_pose_id") != grasp_pose_id(config):
        raise RuntimeError("guided dynamic grasp pose identity changed")
    if result.get("controller_id") != controller_id(config):
        raise RuntimeError("guided dynamic controller identity changed")
    return {
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha,
        "config": config,
        "result": result,
        "result_path": str(result_path),
        "trace_path": str(trace_path),
        "trace_retained": trace_retained,
        "artifact_key": canonical_sha256(
            {
                "candidate_id": candidate_id,
                "candidate_sha256": candidate_sha,
                "artifact_directory": str(directory),
            }
        ),
        "result_file_sha256": file_sha256(result_path),
        "trace_file_sha256": (
            file_sha256(trace_path) if trace_retained else None
        ),
    }


def extract_signed_dynamic_contact_observation(
    config: Mapping[str, Any],
    result: Mapping[str, Any],
    trace: Mapping[str, np.ndarray],
    *,
    artifact_key: str | None = None,
) -> DynamicContactObservation:
    """Extract signed VERIFY centroids and pose response from one trace."""

    required = (
        "control_state",
        "target_face_contact_centroid_world_m",
        "target_face_contact_centroid_valid",
        "cube_pos",
        "cube_quat",
        "initial_cube_pos_m",
        "initial_cube_quat",
    )
    if any(name not in trace for name in required):
        raise ValueError("guided trace is missing dynamic centroid/pose arrays")
    state = np.asarray(trace["control_state"])
    centroid = np.asarray(trace["target_face_contact_centroid_world_m"], dtype=float)
    valid = np.asarray(trace["target_face_contact_centroid_valid"], dtype=bool)
    if centroid.shape != (len(state), 3, 3) or valid.shape != (len(state), 3):
        raise ValueError("guided centroid trace shape changed")
    mask = (state == "VERIFY") & np.all(valid, axis=1)
    if "grasp_gate" in trace and "grasp_gate_order" in trace:
        order = [str(value) for value in np.asarray(trace["grasp_gate_order"])]
        gate = np.asarray(trace["grasp_gate"], dtype=bool)
        for name in (
            "thumb_target_face_effective",
            "index_target_face_effective",
            "mid_target_face_effective",
        ):
            if name in order:
                mask &= gate[:, order.index(name)]
    if not np.any(mask):
        signed = (math.inf, math.inf)
        sample_count = 0
    else:
        gravity = np.asarray(config.get("gravity_m_s2", (0.0, 0.0, -9.81)), dtype=float)
        if gravity.shape != (3,) or np.linalg.norm(gravity) <= _EPS:
            gravity = np.asarray((0.0, 0.0, -1.0))
        up = -gravity / np.linalg.norm(gravity)
        heights = centroid[mask] @ up
        signed = (
            float(np.median(heights[:, 1] - heights[:, 0])),
            float(np.median(heights[:, 2] - heights[:, 0])),
        )
        sample_count = int(np.count_nonzero(mask))
    summary = result["summary"]["metrics"]
    alignment = summary["contact_alignment"]["verify"]
    pose = summary["pose_preservation"]
    onset = pose.get("first_distal_contact_step", {})
    initial_pos = np.asarray(trace["initial_cube_pos_m"], dtype=float)
    cube_pos = np.asarray(trace["cube_pos"], dtype=float)
    verify = state == "VERIFY"
    terminal_index = int(np.flatnonzero(verify)[-1]) if np.any(verify) else len(state) - 1
    translation = cube_pos[terminal_index] - initial_pos
    initial_quat = np.asarray(trace["initial_cube_quat"], dtype=float)
    cube_quat = np.asarray(trace["cube_quat"], dtype=float)
    orientation = _relative_quat_rotvec(initial_quat, cube_quat[terminal_index])
    forces = summary.get("verify_peak_target_face_force_n", {})
    thumb_force = max(float(forces.get("thumb", 0.0)), 1e-9)
    force_ratio = (
        math.log(max(float(forces.get("index", 0.0)), 1e-9) / thumb_force),
        math.log(max(float(forces.get("mid", 0.0)), 1e-9) / thumb_force),
    )
    duty = summary.get("verify_gate_component_duty", {})
    return DynamicContactObservation(
        artifact_key=(
            str(artifact_key)
            if artifact_key is not None
            else canonical_sha256(
                {
                    "candidate_id": int(result["candidate_id"]),
                    "candidate_sha256": str(result.get("candidate_sha256", "")),
                }
            )
        ),
        candidate_id=int(result["candidate_id"]),
        edge_m=float(config["cube"]["edge_m"]),
        clockwise_orbit_deg=float(
            config.get("candidate_metadata", {})
            .get("relative_wrist_pose_search", {})
            .get("clockwise_orbit_deg", 0.0)
        ),
        grasp_success=bool(result.get("grasp_success", False)),
        effective_finger_count=int(summary.get("verify_max_simultaneous_effective_finger_count", 0)),
        simultaneous_target_face_duty=float(summary.get("verify_target_face_simultaneous_duty", 0.0)),
        signed_height_error_m=signed,
        height_spread_p95_m=float(alignment.get("height_spread_p95_m") or math.inf),
        cube_translation_m=tuple(float(value) for value in translation),
        cube_orientation_rotvec_rad=tuple(float(value) for value in orientation),
        thumb_actual_range_duty=float(duty.get("thumb_actual_qpos_within_range", 0.0)),
        force_log_ratio=force_ratio,
        first_contact_step=tuple(
            int(onset.get(finger, -1)) for finger in ACTIVE_FINGERS
        ),
        maximum_consecutive_gate_steps=int(summary.get("verify_max_consecutive_gate_steps", 0)),
        trace_sample_count=sample_count,
        maximum_translation_m=float(pose.get("max_translation_m", np.linalg.norm(translation))),
        maximum_orientation_drift_rad=math.radians(
            float(
                pose.get(
                    "max_orientation_drift_deg",
                    math.degrees(float(np.linalg.norm(orientation))),
                )
            )
        ),
    )


def extract_compacted_dynamic_contact_observation(
    config: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    artifact_key: str,
) -> DynamicContactObservation:
    """Recover selection-only metrics from an authenticated compacted result.

    A compacted result deliberately has no trace, so it cannot contribute a
    signed centroid response to the ridge fit.  Its persisted acceptance
    metrics are nevertheless authenticated evidence for deterministic parent
    coverage.  ``trace_sample_count=0`` keeps that distinction explicit.
    """

    summary = result["summary"]["metrics"]
    alignment = summary["contact_alignment"]["verify"]
    pose = summary["pose_preservation"]
    height_raw = alignment.get("height_spread_p95_m")
    # Finite sentinels keep atomic JSON reports strict while remaining far
    # outside every physical acceptance threshold.  Availability is carried
    # separately by ``trace_sample_count=0``.
    height = 1.0 if height_raw is None else float(height_raw)
    translation_raw = pose.get("position_delta_at_scope_end_m")
    if translation_raw is None:
        translation = np.asarray(
            (float(pose.get("translation_at_scope_end_m", 1.0)), 0.0, 0.0),
            dtype=np.float64,
        )
    else:
        translation = np.asarray(translation_raw, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        translation = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    orientation_at_end_rad = math.radians(
        float(
            pose.get(
                "orientation_at_scope_end_deg",
                pose.get("max_orientation_drift_deg", 180.0),
            )
        )
    )
    forces = summary.get("verify_peak_target_face_force_n", {})
    thumb_force = max(float(forces.get("thumb", 0.0)), 1e-9)
    duty = summary.get("verify_gate_component_duty", {})
    onset = pose.get("first_distal_contact_step", {})
    return DynamicContactObservation(
        artifact_key=str(artifact_key),
        candidate_id=int(result["candidate_id"]),
        edge_m=float(config["cube"]["edge_m"]),
        clockwise_orbit_deg=float(
            config.get("candidate_metadata", {})
            .get("relative_wrist_pose_search", {})
            .get("clockwise_orbit_deg", 0.0)
        ),
        grasp_success=bool(result.get("grasp_success", False)),
        effective_finger_count=int(
            summary.get("verify_max_simultaneous_effective_finger_count", 0)
        ),
        simultaneous_target_face_duty=float(
            summary.get("verify_target_face_simultaneous_duty", 0.0)
        ),
        signed_height_error_m=(0.0, 0.0),
        height_spread_p95_m=height,
        cube_translation_m=tuple(float(value) for value in translation),
        cube_orientation_rotvec_rad=(orientation_at_end_rad, 0.0, 0.0),
        thumb_actual_range_duty=float(
            duty.get("thumb_actual_qpos_within_range", 0.0)
        ),
        force_log_ratio=(
            math.log(max(float(forces.get("index", 0.0)), 1e-9) / thumb_force),
            math.log(max(float(forces.get("mid", 0.0)), 1e-9) / thumb_force),
        ),
        first_contact_step=tuple(
            int(onset.get(finger, -1)) for finger in ACTIVE_FINGERS
        ),
        maximum_consecutive_gate_steps=int(
            summary.get("verify_max_consecutive_gate_steps", 0)
        ),
        trace_sample_count=0,
        maximum_translation_m=float(
            pose.get("max_translation_m", np.linalg.norm(translation))
        ),
        maximum_orientation_drift_rad=math.radians(
            float(pose.get("max_orientation_drift_deg", math.degrees(orientation_at_end_rad)))
        ),
    )


def _maximum_translation(observation: DynamicContactObservation) -> float:
    if observation.maximum_translation_m is not None:
        return float(observation.maximum_translation_m)
    return float(np.linalg.norm(observation.cube_translation_m))


def _maximum_orientation_drift(observation: DynamicContactObservation) -> float:
    if observation.maximum_orientation_drift_rad is not None:
        return float(observation.maximum_orientation_drift_rad)
    return float(np.linalg.norm(observation.cube_orientation_rotvec_rad))


def hard_normalized_dynamic_merit(
    observation: DynamicContactObservation,
) -> tuple[Any, ...]:
    height = max(0.0, observation.height_spread_p95_m / 0.005 - 1.0)
    translation = max(0.0, _maximum_translation(observation) / 0.0005 - 1.0)
    orientation = max(
        0.0,
        _maximum_orientation_drift(observation) / math.radians(1.0) - 1.0,
    )
    return (
        not observation.grasp_success,
        max(0, 3 - observation.effective_finger_count),
        1.0 - observation.simultaneous_target_face_duty,
        max(height, translation, orientation, 1.0 - observation.thumb_actual_range_duty),
        height,
        translation,
        orientation,
        1.0 - observation.thumb_actual_range_duty,
        -observation.maximum_consecutive_gate_steps,
        float(np.linalg.norm(observation.force_log_ratio)),
        observation.artifact_key,
        observation.candidate_id,
    )


def extract_dynamic_centroid_diagnostic(
    authenticated_record: Mapping[str, Any],
    trace_path: str | Path | None = None,
) -> DynamicContactObservation:
    """Compatibility entry for a standalone orchestrator's exact replay."""

    path = Path(
        str(trace_path)
        if trace_path is not None
        else str(authenticated_record["trace_path"])
    )
    if not path.is_file():
        raise RuntimeError(f"dynamic centroid trace is unavailable: {path}")
    with np.load(path) as trace:
        return extract_signed_dynamic_contact_observation(
            authenticated_record["config"],
            authenticated_record["result"],
            trace,
            artifact_key=str(authenticated_record["artifact_key"]),
        )


def select_dynamic_centroid_parents(
    observations: Sequence[DynamicContactObservation], *, top_count: int
) -> tuple[DynamicContactObservation, ...]:
    """Select deterministic artifact-unique parents with metric coverage.

    The first three slots cover complementary evidence before global-rank
    fill: the best hard-ranked pose-safe record, the lowest-drift height-safe
    record, and the best thumb-range/composite record.  This avoids allowing a
    single family of low-height but badly drifting traces (or vice versa) to
    consume the entire guided budget.
    """

    if top_count <= 0:
        raise ValueError("top_count must be positive")
    unique: dict[str, DynamicContactObservation] = {}
    for value in sorted(observations, key=hard_normalized_dynamic_merit):
        unique.setdefault(value.artifact_key, value)
    ranked = sorted(unique.values(), key=hard_normalized_dynamic_merit)
    selected: list[DynamicContactObservation] = []

    def add_best(
        values: Sequence[DynamicContactObservation],
        key: Callable[[DynamicContactObservation], tuple[Any, ...]],
    ) -> None:
        if len(selected) >= top_count:
            return
        chosen_keys = {value.artifact_key for value in selected}
        eligible = [value for value in values if value.artifact_key not in chosen_keys]
        if eligible:
            selected.append(min(eligible, key=key))

    pose_safe = [
        value
        for value in ranked
        if _maximum_translation(value) <= 0.0005
        and _maximum_orientation_drift(value) <= math.radians(1.0)
    ]
    add_best(pose_safe, hard_normalized_dynamic_merit)

    height_safe = [value for value in ranked if value.height_spread_p95_m <= 0.005]
    add_best(
        height_safe,
        lambda value: (
            max(
                _maximum_translation(value) / 0.0005,
                _maximum_orientation_drift(value) / math.radians(1.0),
            ),
            _maximum_translation(value),
            _maximum_orientation_drift(value),
            hard_normalized_dynamic_merit(value),
        ),
    )

    add_best(
        ranked,
        lambda value: (
            1.0 - value.thumb_actual_range_duty,
            hard_normalized_dynamic_merit(value),
        ),
    )
    for value in ranked:
        if len(selected) >= top_count:
            break
        if value.artifact_key not in {item.artifact_key for item in selected}:
            selected.append(value)
    return tuple(selected)


def dynamic_centroid_recovery_rank(
    observation: DynamicContactObservation,
) -> tuple[Any, ...]:
    return hard_normalized_dynamic_merit(observation)


def encode_pose_control_vector(config: Mapping[str, Any]) -> np.ndarray:
    pose = RelativeWristVariables.from_config(config).as_array()
    preload = config["control"]["contact_preload_targets_rad"]
    profile = config["control"]["close_profile"]
    starts = [float(profile[next(iter(_finger_actuators(finger)))]["start_fraction"]) for finger in ACTIVE_FINGERS]
    ends = [float(profile[next(iter(_finger_actuators(finger)))]["end_fraction"]) for finger in ACTIVE_FINGERS]
    vector = np.asarray(
        (
            *pose,
            *(float(preload[name]) for name in ACTIVE_ACTUATORS),
            *starts,
            *ends,
            float(config["control_protocol"]["close_s"]),
        ),
        dtype=float,
    )
    assert vector.shape == (VECTOR_DIMENSION,)
    return vector


def _finger_actuators(finger: str) -> tuple[str, ...]:
    if finger == "thumb":
        return tuple(ACTIVE_ACTUATORS[:3])
    if finger == "index":
        return tuple(ACTIVE_ACTUATORS[3:6])
    return tuple(ACTIVE_ACTUATORS[6:8])


def fit_ridge_svd(
    predictors: Sequence[Sequence[float]],
    responses: Sequence[Sequence[float]],
    *,
    ridge: float = 0.05,
    scale: Sequence[float] | None = None,
) -> RidgeSVDModel:
    x = np.asarray(predictors, dtype=float)
    y = np.asarray(responses, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("ridge fit requires matching 2D arrays with at least two rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or ridge <= 0.0:
        raise ValueError("ridge fit inputs and ridge must be finite")
    center = np.mean(x, axis=0)
    resolved_scale = (
        np.std(x, axis=0)
        if scale is None
        else np.asarray(scale, dtype=float)
    )
    resolved_scale = np.where(resolved_scale > 1e-12, resolved_scale, 1.0)
    normalized = (x - center) / resolved_scale
    intercept = np.mean(y, axis=0)
    u, singular, vt = np.linalg.svd(normalized, full_matrices=False)
    coefficient = (
        vt.T * (singular / (singular**2 + float(ridge)))
    ) @ u.T @ (y - intercept)
    return RidgeSVDModel(coefficient, intercept, center, resolved_scale, singular, float(ridge))


def deterministic_bounded_proposals(
    center: Sequence[float],
    model: RidgeSVDModel,
    *,
    lower: Sequence[float],
    upper: Sequence[float],
    trust_radius: Sequence[float],
    count: int = 7,
) -> tuple[np.ndarray, ...]:
    """Generate deterministic LS/height-only/sensitivity proposals."""

    base = np.asarray(center, dtype=float)
    low, high, trust = (np.asarray(value, dtype=float) for value in (lower, upper, trust_radius))
    if any(value.shape != base.shape for value in (low, high, trust)):
        raise ValueError("proposal bounds/trust shape changed")
    jacobian = model.coefficient.T / model.scale[np.newaxis, :]
    response = model.predict(base)

    def solve(rows: Sequence[int]) -> np.ndarray:
        matrix = jacobian[np.asarray(rows), :POSE_DIMENSION]
        rhs = -response[np.asarray(rows)]
        step = np.linalg.pinv(matrix, rcond=1e-8) @ rhs
        full = np.zeros_like(base)
        full[:POSE_DIMENSION] = np.clip(step, -trust[:POSE_DIMENSION], trust[:POSE_DIMENSION])
        return full

    full_step = solve(tuple(range(RESPONSE_DIMENSION)))
    height_step = solve((0, 1))
    raw: list[np.ndarray] = [
        base + alpha * full_step for alpha in (0.25, 0.5, 1.0)
    ] + [base + alpha * height_step for alpha in (0.5, 1.0)]
    sensitivity = np.linalg.norm(jacobian[:2, :POSE_DIMENSION], axis=0)
    for column in np.argsort(-sensitivity, kind="stable")[:2]:
        direction = -math.copysign(1.0, float(jacobian[np.argmax(np.abs(response[:2])), column] * response[np.argmax(np.abs(response[:2]))]))
        proposal = base.copy()
        proposal[column] += direction * trust[column]
        raw.append(proposal)
    result: list[np.ndarray] = []
    seen: set[str] = set()
    for value in raw:
        clipped = np.minimum(np.maximum(value, low), high)
        key = canonical_sha256(clipped.tolist())
        if key not in seen and not np.array_equal(clipped, base):
            seen.add(key)
            result.append(clipped)
        if len(result) == count:
            break
    return tuple(result)


def deterministic_guided_candidate_id(
    parent_candidate_id: int,
    round_index: int,
    proposal_index: int,
    kind: str,
    *,
    parent_artifact_key: str = "",
) -> int:
    digest = canonical_sha256(
        {
            "parent_candidate_id": int(parent_candidate_id),
            "round_index": int(round_index),
            "proposal_index": int(proposal_index),
            "kind": str(kind),
            "parent_artifact_key": str(parent_artifact_key),
        }
    )
    return GUIDED_ID_BASE + int(digest[:14], 16)


def deduplicate_guided_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    best: dict[str, dict[str, Any]] = {}
    for raw in sorted(candidates, key=lambda value: int(value["candidate_id"])):
        candidate = copy.deepcopy(dict(raw))
        key = str(
            candidate.get("candidate_sha256")
            or canonical_sha256(candidate["config"])
        )
        best.setdefault(key, candidate)
    return tuple(best[key] for key in sorted(best, key=lambda item: int(best[item]["candidate_id"])))


def generate_controller_balance_proposals(
    parent_config: Mapping[str, Any],
    observation: DynamicContactObservation,
    *,
    count: int = 8,
) -> tuple[dict[str, Any], ...]:
    """Freeze grasp pose and deterministically balance preload/contact onset."""

    if count <= 0:
        raise ValueError("controller proposal count must be positive")
    definition = resolve_experiment(dict(parent_config))
    bounds = definition.search_bounds.actuator_targets_rad
    base = copy.deepcopy(dict(parent_config))
    nominal = base["grasp_pose"]["nominal_joint_qpos_rad"]
    precontact = base["control"]["precontact_targets_rad"]
    preload = base["control"]["contact_preload_targets_rad"]
    close_steps = max(float(base["control_protocol"]["close_s"]) / 0.001, 1.0)
    onset = np.asarray(observation.first_contact_step, dtype=float)
    valid = onset >= 0
    onset_center = float(np.mean(onset[valid])) if np.any(valid) else 0.0
    start_shift = np.where(valid, -(onset - onset_center) / close_steps, 0.0)
    start_shift = np.clip(start_shift, -0.04, 0.04)
    # Put the measured controller rescue first.  In the v3 recovery campaign
    # the frozen geometry reached a real 250 ms grasp lock when the index
    # preload was expressed as
    #
    #   nominal + scale * (nominal - precontact)
    #
    # at scale=0.25, with thumb-bend preload 1.405 rad.  Keeping the complete
    # deterministic ladder here makes that evidence reproducible without
    # relying on a one-off hand-edited configuration.  These first proposals
    # deliberately retain the registered parent timing and all other thumb /
    # middle commands.
    index_scale_ladder = (0.0, 0.25, 0.5, 0.75, 1.0)
    from .actual_contact_grasp_pose_dynamic import generate_controller_seeds

    registered_seed = generate_controller_seeds(
        base,
        source_candidate_id=observation.candidate_id,
        count=1,
    )[0].as_dict()
    proposals: list[dict[str, Any]] = []
    for index_scale in index_scale_ladder[:count]:
        proposal = copy.deepcopy(base)
        targets = copy.deepcopy(registered_seed["contact_preload_targets_rad"])
        proposal["control"]["contact_preload_targets_rad"] = targets
        targets[ACTIVE_ACTUATORS[0]] = float(
            np.clip(1.405, *bounds[ACTIVE_ACTUATORS[0]])
        )
        for name in _finger_actuators("index"):
            value = float(nominal[name]) + index_scale * (
                float(nominal[name]) - float(precontact[name])
            )
            targets[name] = float(np.clip(value, *bounds[name]))
        for finger in ACTIVE_FINGERS:
            for name in _finger_actuators(finger):
                proposal["control"]["close_profile"][name] = {
                    "start_fraction": float(
                        registered_seed["group_start_fraction"][finger]
                    ),
                    "end_fraction": float(
                        registered_seed["group_end_fraction"][finger]
                    ),
                }
        proposal["control_protocol"]["close_s"] = float(
            registered_seed["close_s"]
        )
        validate_config(proposal)
        proposals.append(proposal)

    combinations = [
        (thumb, balance, timing)
        for thumb in (0.0, 0.01, 0.02)
        for balance in (0.5, 1.0)
        for timing in (0.5, 1.0)
    ]
    for thumb_offset, balance_scale, timing_scale in combinations:
        if len(proposals) >= count:
            break
        proposal = copy.deepcopy(base)
        targets = copy.deepcopy(preload)
        targets[ACTIVE_ACTUATORS[0]] = float(
            np.clip(float(preload[ACTIVE_ACTUATORS[0]]) + thumb_offset, *bounds[ACTIVE_ACTUATORS[0]])
        )
        # Positive log-ratio means that finger is over-force relative to thumb;
        # move its command a small distance back toward precontact.  Negative
        # means under-force and moves farther along the observed closing ray.
        for finger_index, finger in enumerate(("index", "mid"), start=1):
            log_ratio = observation.force_log_ratio[finger_index - 1]
            correction = float(np.clip(-0.01 * log_ratio, -0.01, 0.01)) * balance_scale
            for name in _finger_actuators(finger):
                closing = float(preload[name]) - float(precontact[name])
                direction = math.copysign(1.0, closing) if abs(closing) > 1e-9 else math.copysign(1.0, float(nominal[name]) - float(precontact[name]) or 1.0)
                targets[name] = float(np.clip(float(preload[name]) + direction * correction, *bounds[name]))
        proposal["control"]["contact_preload_targets_rad"] = targets
        profile = proposal["control"]["close_profile"]
        for finger_index, finger in enumerate(ACTIVE_FINGERS):
            for name in _finger_actuators(finger):
                start = float(np.clip(
                    float(profile[name]["start_fraction"])
                    + timing_scale * start_shift[finger_index],
                    0.0,
                    0.95,
                ))
                profile[name]["start_fraction"] = start
                profile[name]["end_fraction"] = max(
                    start + 0.01, float(profile[name]["end_fraction"])
                )
        # Select a registered close duration without introducing a continuum.
        close_options = tuple(definition.control_protocol.close_duration_options_s)
        current_close = float(base["control_protocol"]["close_s"])
        proposal["control_protocol"]["close_s"] = min(
            close_options, key=lambda value: (abs(float(value) - current_close), float(value))
        )
        if (
            proposal["grasp_pose"] != base["grasp_pose"]
            or proposal["hand_pose"] != base["hand_pose"]
            or proposal["cube"] != base["cube"]
        ):
            raise AssertionError("controller balance changed the grasp pose")
        validate_config(proposal)
        proposals.append(proposal)
    return tuple(proposals)


def _vector_bounds_and_trust(
    config: Mapping[str, Any], context: ActiveSetEvaluationContext, policy: GuidedRefinementPolicy
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    search_policy = RelativeWristPoseSearchPolicy.from_config(config)
    low: list[float] = [context.joint_bounds[name][0] for name in NON_THUMB_ACTUATORS]
    high: list[float] = [context.joint_bounds[name][1] for name in NON_THUMB_ACTUATORS]
    for axis in ("x", "y", "z"):
        bounds = search_policy.root_delta_cube_m[axis]
        low.append(bounds[0]); high.append(bounds[1])
    for axis in ("x", "y", "z"):
        bounds = search_policy.wrist_local_rotvec_deg[axis]
        low.append(math.radians(bounds[0])); high.append(math.radians(bounds[1]))
    definition = resolve_experiment(dict(config))
    preload_bounds = definition.search_bounds.actuator_targets_rad
    low.extend(preload_bounds[name][0] for name in ACTIVE_ACTUATORS)
    high.extend(preload_bounds[name][1] for name in ACTIVE_ACTUATORS)
    low.extend((0.0,) * 3); high.extend((0.95,) * 3)
    low.extend((0.01,) * 3); high.extend((1.0,) * 3)
    close = definition.control_protocol.close_duration_options_s
    low.append(min(close)); high.append(max(close))
    trust = np.asarray(
        [policy.joint_trust_rad] * 7
        + [policy.translation_trust_m] * 3
        + [policy.rotation_trust_rad] * 3
        + [0.02] * 8
        + [0.08] * 3
        + [0.05] * 3
        + [0.5],
        dtype=float,
    )
    return np.asarray(low), np.asarray(high), trust


def _materialize_pose_vector(
    parent: Mapping[str, Any],
    vector: np.ndarray,
    context: ActiveSetEvaluationContext,
    policy: GuidedRefinementPolicy,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    config = copy.deepcopy(dict(parent["config"]))
    relative = config["candidate_metadata"]["relative_wrist_pose_search"]
    orbit = float(relative["clockwise_orbit_deg"])
    variables = RelativeWristVariables.from_array(vector[:POSE_DIMENSION])
    solved = solve_orientation_aware_active_set_dls(
        config,
        clockwise_orbit_deg=orbit,
        initial_variables=variables,
        evaluation_context=context,
        settings=RelativeWristDLSSettings(maximum_iterations=policy.maximum_static_iterations),
    )
    diagnostic = {
        "stop_reason": solved.stop_reason,
        "promotion_config_valid": bool(solved.diagnostics.get("promotion_config_valid")),
        "full_scene_contact_safety": copy.deepcopy(solved.diagnostics.get("full_scene_contact_safety", {})),
        "requested_pose_vector": vector[:POSE_DIMENSION].tolist(),
        "realized_pose_vector": solved.variables.as_array().tolist(),
    }
    if not diagnostic["promotion_config_valid"]:
        return None, diagnostic
    fresh, gate = evaluate_active_set_candidate(context, solved.config)
    if not fresh.safe or not fresh.static_result.static_geometry_pass or not gate.get("passed", False):
        diagnostic["fresh_terminal_safe"] = False
        return None, diagnostic
    promoted = apply_precontact_solution(solved.config, fresh.static_result)
    # Preserve the parent's controller residual relative to its nominal pose.
    parent_nominal = parent["config"]["grasp_pose"]["nominal_joint_qpos_rad"]
    parent_preload = parent["config"]["control"]["contact_preload_targets_rad"]
    bounds = resolve_experiment(promoted).search_bounds.actuator_targets_rad
    promoted["control"]["contact_preload_targets_rad"] = {
        name: float(np.clip(
            promoted["grasp_pose"]["nominal_joint_qpos_rad"][name]
            + float(parent_preload[name]) - float(parent_nominal[name]),
            *bounds[name],
        ))
        for name in ACTIVE_ACTUATORS
    }
    promoted["control"]["close_profile"] = copy.deepcopy(parent["config"]["control"]["close_profile"])
    promoted["control_protocol"]["close_s"] = float(parent["config"]["control_protocol"]["close_s"])
    promoted["control"]["manipulation_delta_rad"] = {name: 0.0 for name in ACTIVE_ACTUATORS}
    validate_config(promoted)
    diagnostic["fresh_terminal_safe"] = True
    return promoted, diagnostic


def _materialized_record(
    config: Mapping[str, Any], *, candidate_id: int, parent_candidate_id: int, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    resolved.setdefault("candidate_metadata", {}).update(
        {
            "campaign_kind": "dynamic_contact_centroid_guided_refinement",
            "candidate_id": int(candidate_id),
            "source_candidate_id": int(parent_candidate_id),
            "dynamic_centroid_guidance": copy.deepcopy(dict(metadata)),
        }
    )
    return {
        "campaign_kind": "dynamic_contact_centroid_guided_refinement",
        "stage": str(metadata["stage"]),
        "candidate_id": int(candidate_id),
        "source_candidate_id": int(parent_candidate_id),
        "controller_seed_index": int(metadata["proposal_index"]),
        "grasp_pose_id": grasp_pose_id(resolved),
        "controller_id": controller_id(resolved),
        "candidate_sha256": canonical_sha256(resolved),
        "config": resolved,
    }


def build_dynamic_centroid_candidate_records(
    configs: Sequence[Mapping[str, Any]],
    *,
    parent_candidate_id: int,
    round_index: int,
    kind: str,
    stage: str,
    parent_artifact_key: str = "",
) -> tuple[dict[str, Any], ...]:
    """Build runner-ready records with deterministic IDs and full identities."""

    records = [
        _materialized_record(
            config,
            candidate_id=deterministic_guided_candidate_id(
                parent_candidate_id, round_index, index, kind
                , parent_artifact_key=parent_artifact_key
            ),
            parent_candidate_id=parent_candidate_id,
            metadata={
                "stage": stage,
                "round_index": round_index,
                "proposal_index": index,
                "kind": kind,
            },
        )
        for index, config in enumerate(configs)
    ]
    return deduplicate_guided_candidates(records)


def _execution_has_success(execution: Any) -> bool:
    return bool(int(execution.summary.get("grasp_success_count", 0)) > 0)


def run_dynamic_contact_centroid_guided_refinement(
    dynamic_records: Sequence[Mapping[str, Any]],
    static_source: Mapping[str, Any],
    source_dynamic_root: str | Path,
    output_dir: str | Path,
    *,
    workers: int,
    seed: int,
    stage: str = "dynamic_centroid_guided",
    policy: GuidedRefinementPolicy | None = None,
    evaluation_context: ActiveSetEvaluationContext | None = None,
    dynamic_executor: Callable[..., Any] | None = None,
) -> Any:
    """Run an independently resumable, hard-gated guided refinement stage."""

    del seed  # Candidate generation is deterministic and does not sample RNG.
    resolved_policy = policy or GuidedRefinementPolicy()
    if workers <= 0:
        raise ValueError("workers must be positive")
    source_config = copy.deepcopy(dict(static_source["config"]))
    context = evaluation_context or build_active_set_evaluation_context(source_config)
    authenticated = tuple(
        authenticate_dynamic_record(record, source_dynamic_root)
        for record in sorted(dynamic_records, key=lambda value: int(value["candidate_id"]))
    )
    observations: list[DynamicContactObservation] = []
    training_observations: list[DynamicContactObservation] = []
    vectors: list[np.ndarray] = []
    for value in authenticated:
        if value["trace_retained"]:
            with np.load(value["trace_path"]) as trace:
                observation = extract_signed_dynamic_contact_observation(
                    value["config"], value["result"], trace,
                    artifact_key=value["artifact_key"],
                )
            if observation.trace_sample_count:
                observations.append(observation)
                training_observations.append(observation)
                vectors.append(encode_pose_control_vector(value["config"]))
            else:
                observations.append(
                    extract_compacted_dynamic_contact_observation(
                        value["config"],
                        value["result"],
                        artifact_key=value["artifact_key"],
                    )
                )
        else:
            observations.append(
                extract_compacted_dynamic_contact_observation(
                    value["config"],
                    value["result"],
                    artifact_key=value["artifact_key"],
                )
            )
    if len(training_observations) < 2:
        raise RuntimeError("guided refinement needs at least two valid dynamic traces")
    by_artifact = {value["artifact_key"]: value for value in authenticated}
    parent_observations = select_dynamic_centroid_parents(
        observations, top_count=resolved_policy.parent_count
    )
    responses = [value.response_vector() for value in training_observations]
    model = fit_ridge_svd(vectors, responses, ridge=resolved_policy.ridge)
    lower, upper, trust = _vector_bounds_and_trust(source_config, context, resolved_policy)
    output = Path(output_dir).expanduser().resolve()
    input_payload = {
        "stage": stage,
        "workers": int(workers),
        "policy": asdict(resolved_policy),
        "static_source_sha256": canonical_sha256(source_config),
        "dynamic_evidence": [
            {
                "candidate_id": value["candidate_id"],
                "candidate_sha256": value["candidate_sha256"],
                "result_file_sha256": value["result_file_sha256"],
                "trace_file_sha256": value["trace_file_sha256"],
            }
            for value in authenticated
        ],
    }
    input_sha = canonical_sha256(input_payload)
    report_path = output / "dynamic" / f"{stage}_guided_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("input_sha256") != input_sha:
            raise RuntimeError("guided refinement resume input changed")
        candidates = tuple(report.get("materialized_candidates", ()))
    else:
        candidates_list: list[dict[str, Any]] = []
        static_diagnostics: list[dict[str, Any]] = []
        for parent_rank, observation in enumerate(parent_observations):
            parent = by_artifact[observation.artifact_key]
            center = encode_pose_control_vector(parent["config"])
            proposals = deterministic_bounded_proposals(
                center, model, lower=lower, upper=upper, trust_radius=trust,
                count=resolved_policy.guided_per_parent,
            )
            for proposal_index, proposal in enumerate(proposals):
                candidate_id = deterministic_guided_candidate_id(
                    observation.candidate_id,
                    0,
                    proposal_index,
                    "guided_pose",
                    parent_artifact_key=observation.artifact_key,
                )
                promoted, diagnostic = _materialize_pose_vector(
                    parent, proposal, context, resolved_policy
                )
                static_diagnostics.append({
                    "candidate_id": candidate_id,
                    "parent_candidate_id": observation.candidate_id,
                    **diagnostic,
                })
                if promoted is None:
                    continue
                candidates_list.append(_materialized_record(
                    promoted,
                    candidate_id=candidate_id,
                    parent_candidate_id=observation.candidate_id,
                    metadata={
                        "stage": stage,
                        "round_index": 0,
                        "proposal_index": proposal_index,
                        "kind": "guided_pose",
                        "parent_dynamic_merit": list(hard_normalized_dynamic_merit(observation)),
                    },
                ))
        candidates = deduplicate_guided_candidates(candidates_list)
        if len(candidates) > resolved_policy.maximum_dynamic_budget:
            candidates = candidates[: resolved_policy.maximum_dynamic_budget]
        report = {
            "dynamic_centroid_guided_report_schema_version": 1,
            "complete": False,
            "input_sha256": input_sha,
            "input": input_payload,
            "parent_candidate_ids": [value.candidate_id for value in parent_observations],
            "observations": [asdict(value) for value in observations],
            "ridge": {
                "singular_values": model.singular_values.tolist(),
                "coefficient_sha256": canonical_sha256(model.coefficient.tolist()),
            },
            "static_diagnostics": static_diagnostics,
            "materialized_candidates": list(candidates),
            "declared_maximum_dynamic_budget": resolved_policy.maximum_dynamic_budget,
        }
        _atomic_json(report_path, report)
    if dynamic_executor is None:
        from .actual_contact_grasp_pose import _run_materialized_local_dynamic_stage

        dynamic_executor = _run_materialized_local_dynamic_stage
    execution = dynamic_executor(
        candidates, output, stage=f"{stage}_round0", workers=workers
    )
    executions = [execution]
    controller_records: tuple[dict[str, Any], ...] = ()
    if not _execution_has_success(execution):
        controller_materialized: list[dict[str, Any]] = []
        for observation in parent_observations:
            parent = by_artifact[observation.artifact_key]
            for proposal_index, controller_config in enumerate(
                generate_controller_balance_proposals(
                    parent["config"],
                    observation,
                    count=resolved_policy.controller_per_parent,
                )
            ):
                # Controller proposals freeze geometry, but still require a
                # fresh static/full-scene binding before dynamic execution.
                fresh, gate = evaluate_active_set_candidate(context, controller_config)
                if (
                    not fresh.safe
                    or not fresh.static_result.static_geometry_pass
                    or not gate.get("passed", False)
                ):
                    continue
                safe_config = apply_precontact_solution(
                    controller_config, fresh.static_result
                )
                validate_config(safe_config)
                controller_materialized.append(
                    _materialized_record(
                        safe_config,
                        candidate_id=deterministic_guided_candidate_id(
                            observation.candidate_id,
                            3,
                            proposal_index,
                            "controller_balance",
                            parent_artifact_key=observation.artifact_key,
                        ),
                        parent_candidate_id=observation.candidate_id,
                        metadata={
                            "stage": stage,
                            "round_index": 3,
                            "proposal_index": proposal_index,
                            "kind": "controller_balance",
                        },
                    )
                )
        controller_records = deduplicate_guided_candidates(controller_materialized)
        remaining_budget = max(
            0, resolved_policy.maximum_dynamic_budget - len(candidates)
        )
        controller_records = controller_records[:remaining_budget]
        if controller_records:
            controller_execution = dynamic_executor(
                controller_records,
                output,
                stage=f"{stage}_controller_balance",
                workers=workers,
            )
            executions.append(controller_execution)
    if len(executions) == 1:
        merged_execution = executions[0]
    else:
        from .actual_contact_grasp_pose import CampaignStageExecution

        merged_execution = CampaignStageExecution(
            records=tuple(
                copy.deepcopy(dict(record))
                for item in executions
                for record in item.records
            ),
            artifacts=tuple(
                artifact for item in executions for artifact in item.artifacts
            ),
            summary={
                "dynamic_candidate_count": sum(len(item.records) for item in executions),
                "grasp_success_count": sum(
                    int(item.summary.get("grasp_success_count", 0))
                    for item in executions
                ),
                "workers": int(workers),
            },
        )
    report = {
        **report,
        "complete": True,
        "executed_dynamic_candidate_count": sum(len(item.records) for item in executions),
        "controller_balance_candidate_count": len(controller_records),
        "grasp_success_count": int(merged_execution.summary.get("grasp_success_count", 0)),
        "stopped_early_after_success": _execution_has_success(execution),
        # The independent first implementation stops after its model-guided
        # round on success.  A no-success orchestrator may call this module
        # again with the newly authenticated traces for the next trust round.
        "next_action": (
            "stop_success"
            if int(merged_execution.summary.get("grasp_success_count", 0)) > 0
            else "refit_or_coordinate_round"
        ),
    }
    _atomic_json(report_path, report)
    return merged_execution


__all__ = [
    "CONTROL_DIMENSION",
    "DynamicContactObservation",
    "GuidedRefinementPolicy",
    "POSE_DIMENSION",
    "RESPONSE_DIMENSION",
    "RidgeSVDModel",
    "VECTOR_DIMENSION",
    "VECTOR_NAMES",
    "authenticate_dynamic_record",
    "deduplicate_guided_candidates",
    "deterministic_bounded_proposals",
    "deterministic_guided_candidate_id",
    "dynamic_centroid_recovery_rank",
    "encode_pose_control_vector",
    "extract_dynamic_centroid_diagnostic",
    "extract_compacted_dynamic_contact_observation",
    "extract_signed_dynamic_contact_observation",
    "fit_ridge_svd",
    "generate_controller_balance_proposals",
    "hard_normalized_dynamic_merit",
    "build_dynamic_centroid_candidate_records",
    "run_dynamic_contact_centroid_guided_refinement",
    "select_dynamic_centroid_parents",
]
