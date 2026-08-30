"""Event-aware rescue candidates for schema-v14 contact-preserving lifts.

The first two v14 rescue stages vary global plan shape and timing.  This
module deliberately addresses a different failure mode: a collision witness
can switch while all three fingers remain in contact and inject a narrow
kinematic impulse into the object trace.  It detects those switches from an
authenticated full-reset trace, derives local point-Jacobian directions, and
builds small C2 actuator-space corrections around the affected event.

The module has no persistence or runner dependency.  Descriptors and jobs are
hash-addressed JSON data so a later runner can schedule them in any order and
fail closed when source evidence or a generated job was modified.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from ..config import ACTIVE_ACTUATORS, DISTAL_BODY_NAMES
from ..experiment import ManipulationPlanParameters
from ..grasp_pose import canonical_sha256
from ..scene import build_model
from .contact_constrained_planner import (
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
)
from .contact_preserving_joint_refinement import (
    JointRefinementLimits,
    resolve_joint_refinement_limits,
)
from .contact_preserving_time_warp import (
    COEFFICIENT_BOUNDS,
    DURATION_S,
    KNOT_COUNT,
    _time_warp_controller_id,
    apply_time_warp_to_config,
    validate_time_warped_config,
)


EVENT_RESCUE_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
EXPERIMENT_ID = "left_opposed_face_palm_down_contact_preserving_planned_lift"
_FINGERS = ("thumb", "index", "mid")
_SELECTED_FINGERS = _FINGERS
_FINGER_ACTUATOR_INDICES = {
    "thumb": (0, 1, 2),
    "index": (3, 4, 5),
    "mid": (6, 7),
}
_FACE_NORMALS = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}
_SHA256_LENGTH = 64
_EPSILON = 1e-12


def _sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_tuple(values: Sequence[float], length: int, label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return result


@dataclass(frozen=True, slots=True)
class EventDetectionSettings:
    """Strict, versioned witness-switch detector settings."""

    tangent_jump_threshold_m: float = 0.00025
    cluster_window_steps: int = 75
    checkpoint_lead_steps: int = 50
    jerk_peak_radius_steps: int = 30
    minimum_peak_abs_jerk_m_s3: float = 2.5
    max_events_per_finger: int = 2
    global_max_events: int = 6
    selected_fingers: tuple[str, ...] = _SELECTED_FINGERS

    def __post_init__(self) -> None:
        threshold = float(self.tangent_jump_threshold_m)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("tangent_jump_threshold_m must be positive and finite")
        object.__setattr__(self, "tangent_jump_threshold_m", threshold)
        for name in (
            "cluster_window_steps",
            "checkpoint_lead_steps",
            "jerk_peak_radius_steps",
            "max_events_per_finger",
            "global_max_events",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        minimum_jerk = float(self.minimum_peak_abs_jerk_m_s3)
        if not math.isfinite(minimum_jerk) or minimum_jerk < 0.0:
            raise ValueError("minimum_peak_abs_jerk_m_s3 must be finite and non-negative")
        object.__setattr__(self, "minimum_peak_abs_jerk_m_s3", minimum_jerk)
        fingers = tuple(str(value) for value in self.selected_fingers)
        if not fingers or len(set(fingers)) != len(fingers) or any(
            value not in _FINGERS for value in fingers
        ):
            raise ValueError("selected_fingers must be unique active finger names")
        object.__setattr__(self, "selected_fingers", fingers)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "tangent_jump_threshold_m": self.tangent_jump_threshold_m,
            "cluster_window_steps": self.cluster_window_steps,
            "checkpoint_lead_steps": self.checkpoint_lead_steps,
            "jerk_peak_radius_steps": self.jerk_peak_radius_steps,
            "minimum_peak_abs_jerk_m_s3": self.minimum_peak_abs_jerk_m_s3,
            "max_events_per_finger": self.max_events_per_finger,
            "global_max_events": self.global_max_events,
            "selected_fingers": list(self.selected_fingers),
        }


@dataclass(frozen=True, slots=True)
class ContactSwitchEvent:
    finger: str
    finger_index: int
    event_step: int
    checkpoint_step: int
    manipulation_progress: float
    tangent_jump_cube_local_m: tuple[float, float, float]
    tangent_jump_m: float
    local_peak_abs_jerk_m_s3: float
    event_id: str

    def as_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EventJacobianDirections:
    event_id: str
    finger: str
    target_face: str
    tangent_direction_rad_unit: tuple[float, ...]
    normal_unload_direction_rad_unit: tuple[float, ...]
    tangent_residual: float
    normal_residual: float
    jacobian_rank: int
    directions_id: str

    def as_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EventRescueDescriptor:
    schema_version: int
    experiment_id: str
    source_candidate_id: int
    source_config_semantic_sha256: str
    source_trace_sha256: str
    source_trace_content_sha256: str
    detection_settings: Mapping[str, Any]
    events: tuple[ContactSwitchEvent, ...]
    directions: tuple[EventJacobianDirections, ...]
    descriptor_id: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EventRescueDescriptor":
        required = {
            "schema_version",
            "experiment_id",
            "source_candidate_id",
            "source_config_semantic_sha256",
            "source_trace_sha256",
            "source_trace_content_sha256",
            "detection_settings",
            "events",
            "directions",
            "descriptor_id",
        }
        if set(raw) != required:
            raise ValueError("event rescue descriptor fields do not match the strict schema")
        events = tuple(ContactSwitchEvent(**dict(value)) for value in raw["events"])
        directions = tuple(
            EventJacobianDirections(**dict(value)) for value in raw["directions"]
        )
        descriptor = cls(
            schema_version=int(raw["schema_version"]),
            experiment_id=str(raw["experiment_id"]),
            source_candidate_id=int(raw["source_candidate_id"]),
            source_config_semantic_sha256=str(raw["source_config_semantic_sha256"]),
            source_trace_sha256=str(raw["source_trace_sha256"]),
            source_trace_content_sha256=str(raw["source_trace_content_sha256"]),
            detection_settings=copy.deepcopy(dict(raw["detection_settings"])),
            events=events,
            directions=directions,
            descriptor_id=str(raw["descriptor_id"]),
        )
        if descriptor.schema_version != EVENT_RESCUE_SCHEMA_VERSION or descriptor.experiment_id != EXPERIMENT_ID:
            raise ValueError("event rescue descriptor version or experiment is invalid")
        if not _sha256(descriptor.source_config_semantic_sha256) or not _sha256(descriptor.source_trace_sha256) or not _sha256(descriptor.source_trace_content_sha256) or not _sha256(descriptor.descriptor_id):
            raise ValueError("event rescue descriptor contains an invalid SHA-256")
        settings = EventDetectionSettings(**dict(descriptor.detection_settings))
        if settings.as_mapping() != dict(descriptor.detection_settings):
            raise ValueError("event detection settings are not canonical")
        if len(events) != len(directions) or tuple(value.event_id for value in events) != tuple(value.event_id for value in directions):
            raise ValueError("event rescue directions do not bind one-to-one to events")
        for event in events:
            expected_event_id = canonical_sha256(
                {
                    "schema_version": EVENT_RESCUE_SCHEMA_VERSION,
                    "finger": event.finger,
                    "event_step": event.event_step,
                    "checkpoint_step": event.checkpoint_step,
                    "tangent_jump_cube_local_m": list(event.tangent_jump_cube_local_m),
                }
            )
            if event.event_id != expected_event_id:
                raise ValueError("contact switch event identity is invalid")
        for direction in directions:
            direction_payload = {
                "schema_version": EVENT_RESCUE_SCHEMA_VERSION,
                "event_id": direction.event_id,
                "finger": direction.finger,
                "target_face": direction.target_face,
                "tangent_direction_rad_unit": list(direction.tangent_direction_rad_unit),
                "normal_unload_direction_rad_unit": list(direction.normal_unload_direction_rad_unit),
                "tangent_residual": direction.tangent_residual,
                "normal_residual": direction.normal_residual,
                "jacobian_rank": direction.jacobian_rank,
            }
            if direction.directions_id != canonical_sha256(direction_payload):
                raise ValueError("event Jacobian direction identity is invalid")
        payload = descriptor.as_mapping()
        recorded = payload.pop("descriptor_id")
        if canonical_sha256(payload) != recorded:
            raise ValueError("event rescue descriptor identity is invalid")
        return descriptor

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "source_candidate_id": self.source_candidate_id,
            "source_config_semantic_sha256": self.source_config_semantic_sha256,
            "source_trace_sha256": self.source_trace_sha256,
            "source_trace_content_sha256": self.source_trace_content_sha256,
            "detection_settings": copy.deepcopy(dict(self.detection_settings)),
            "events": [value.as_mapping() for value in self.events],
            "directions": [value.as_mapping() for value in self.directions],
            "descriptor_id": self.descriptor_id,
        }


@dataclass(frozen=True, slots=True)
class EventRescueBudget:
    """Deterministic exploration or local-refinement job budget."""

    stage: str = "exploration"
    total_candidate_count: int = 512
    seed: int = DEFAULT_SEED
    tangent_radius_rad: float = 0.015
    normal_unload_radius_rad: float = 0.006
    terminal_scale_bounds: tuple[float, float] = (0.97, 1.03)
    time_warp_radius: float = 0.45
    bump_half_width_progress: float = 0.13

    def __post_init__(self) -> None:
        if self.stage not in {"exploration", "local_refinement"}:
            raise ValueError("stage must be exploration or local_refinement")
        if (
            not isinstance(self.total_candidate_count, int)
            or isinstance(self.total_candidate_count, bool)
            or self.total_candidate_count <= 0
        ):
            raise ValueError("total_candidate_count must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in (
            "tangent_radius_rad",
            "normal_unload_radius_rad",
            "time_warp_radius",
            "bump_half_width_progress",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.time_warp_radius > COEFFICIENT_BOUNDS[1]:
            raise ValueError("time_warp_radius exceeds the registered warp bound")
        lower, upper = (float(value) for value in self.terminal_scale_bounds)
        if not (0.0 < lower <= 1.0 <= upper) or not np.isfinite((lower, upper)).all():
            raise ValueError("terminal_scale_bounds must bracket one")
        object.__setattr__(self, "terminal_scale_bounds", (lower, upper))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": EVENT_RESCUE_SCHEMA_VERSION,
            "stage": self.stage,
            "total_candidate_count": self.total_candidate_count,
            "seed": self.seed,
            "tangent_radius_rad": self.tangent_radius_rad,
            "normal_unload_radius_rad": self.normal_unload_radius_rad,
            "terminal_scale_bounds": list(self.terminal_scale_bounds),
            "time_warp_radius": self.time_warp_radius,
            "bump_half_width_progress": self.bump_half_width_progress,
        }


def _trace_array(
    trace: Mapping[str, Any], name: str, *, ndim: int | None = None
) -> np.ndarray:
    if name not in trace:
        raise ValueError(f"full-reset trace is missing {name}")
    values = np.asarray(trace[name])
    if ndim is not None and values.ndim != ndim:
        raise ValueError(f"trace {name} must have {ndim} dimensions")
    if values.dtype.kind in "fc" and not np.isfinite(values).all():
        raise ValueError(f"trace {name} contains non-finite values")
    return values


def trace_content_sha256(trace: Mapping[str, Any]) -> str:
    """Hash every trace key, dtype, shape and byte exactly once.

    The NPZ file digest binds the persisted container; this digest separately
    binds the in-memory arrays from which event evidence was derived.  A
    runner must check the NPZ file digest before loading and this function
    prevents subsequently substituted mappings from authenticating.
    """

    digest = hashlib.sha256()
    for name in sorted(trace):
        if not isinstance(name, str):
            raise ValueError("trace keys must be strings")
        values = np.asarray(trace[name])
        if values.dtype.hasobject:
            raise ValueError("authenticated trace arrays may not contain objects")
        contiguous = np.ascontiguousarray(values)
        header = canonical_sha256(
            {
                "name": name,
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
            }
        )
        digest.update(bytes.fromhex(header))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _scalar_step(trace: Mapping[str, Any], name: str) -> int:
    value = _trace_array(trace, name)
    if value.size != 1:
        raise ValueError(f"trace {name} must be scalar")
    return int(value.reshape(-1)[0])


def _face_tangent(vector: Sequence[float], target_face: str) -> np.ndarray:
    if target_face not in _FACE_NORMALS:
        raise ValueError(f"unknown cube face {target_face}")
    result = np.asarray(_finite_tuple(vector, 3, "contact displacement"))
    normal = np.asarray(_FACE_NORMALS[target_face], dtype=np.float64)
    return result - normal * float(result @ normal)


def detect_contact_switch_events(
    config: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    settings: EventDetectionSettings = EventDetectionSettings(),
) -> tuple[ContactSwitchEvent, ...]:
    """Detect and cluster tangential contact-centroid witness switches."""

    if int(config.get("schema_version", 0)) != 14 or config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("event rescue requires the registered schema-v14 experiment")
    finger_order = tuple(str(value) for value in _trace_array(trace, "finger_order").tolist())
    if finger_order != _FINGERS:
        raise ValueError("trace finger_order does not match the v14 contract")
    centroids = _trace_array(
        trace, "target_face_contact_centroid_cube_local_m", ndim=3
    ).astype(np.float64, copy=False)
    valid = _trace_array(trace, "target_face_contact_centroid_valid", ndim=2).astype(bool)
    effective = _trace_array(trace, "target_face_effective", ndim=2).astype(bool)
    progress = _trace_array(trace, "manipulation_progress", ndim=1).astype(np.float64)
    jerk = _trace_array(
        trace, "operation_vertical_jerk_filtered_m_s3", ndim=1
    ).astype(np.float64)
    length = centroids.shape[0]
    if (
        centroids.shape != (length, len(_FINGERS), 3)
        or valid.shape != (length, len(_FINGERS))
        or effective.shape != valid.shape
        or progress.shape != (length,)
        or jerk.shape != (length,)
        or not np.isfinite(centroids).all()
        or not np.isfinite(progress).all()
        or not np.isfinite(jerk).all()
    ):
        raise ValueError("contact event trace arrays have inconsistent shapes")
    start = _scalar_step(trace, "manipulation_start_step")
    end = _scalar_step(trace, "manipulation_end_step")
    if not 1 <= start < end <= length:
        raise ValueError("trace manipulation bounds are invalid")
    target_faces = config["contact_topology"]["target_faces"]

    detected: list[ContactSwitchEvent] = []
    for finger in settings.selected_fingers:
        finger_index = _FINGERS.index(finger)
        face = str(target_faces[finger])
        raw: list[dict[str, Any]] = []
        for step in range(max(start, 1), end):
            if not (
                valid[step - 1, finger_index]
                and valid[step, finger_index]
                and effective[step - 1, finger_index]
                and effective[step, finger_index]
            ):
                continue
            tangent = _face_tangent(
                centroids[step, finger_index] - centroids[step - 1, finger_index],
                face,
            )
            jump = float(np.linalg.norm(tangent))
            if jump < settings.tangent_jump_threshold_m:
                continue
            low = max(start, step - settings.jerk_peak_radius_steps)
            high = min(end, step + settings.jerk_peak_radius_steps + 1)
            raw.append(
                {
                    "step": step,
                    "tangent": tangent,
                    "jump": jump,
                    "peak": float(np.max(np.abs(jerk[low:high]), initial=0.0)),
                }
            )
        clusters: list[list[dict[str, Any]]] = []
        for candidate in raw:
            if not clusters or candidate["step"] - clusters[-1][-1]["step"] > settings.cluster_window_steps:
                clusters.append([candidate])
            else:
                clusters[-1].append(candidate)
        representatives = [
            max(cluster, key=lambda value: (value["peak"], value["jump"], -value["step"]))
            for cluster in clusters
        ]
        representatives.sort(
            key=lambda value: (-value["peak"], -value["jump"], value["step"])
        )
        representatives = [
            value
            for value in representatives
            if value["peak"] + _EPSILON
            >= settings.minimum_peak_abs_jerk_m_s3
        ]
        for value in representatives:
            checkpoint = max(start, int(value["step"]) - settings.checkpoint_lead_steps)
            payload = {
                "schema_version": EVENT_RESCUE_SCHEMA_VERSION,
                "finger": finger,
                "event_step": int(value["step"]),
                "checkpoint_step": checkpoint,
                "tangent_jump_cube_local_m": [float(item) for item in value["tangent"]],
            }
            detected.append(
                ContactSwitchEvent(
                    finger=finger,
                    finger_index=finger_index,
                    event_step=int(value["step"]),
                    checkpoint_step=checkpoint,
                    manipulation_progress=float(progress[int(value["step"])]),
                    tangent_jump_cube_local_m=tuple(
                        float(item) for item in value["tangent"]
                    ),
                    tangent_jump_m=float(value["jump"]),
                    local_peak_abs_jerk_m_s3=float(value["peak"]),
                    event_id=canonical_sha256(payload),
                )
            )
    # The object-level jerk trace is shared by all fingers.  If two fingers
    # switch within one detector window and resolve to the exact same local
    # jerk peak, retain the larger geometric witness jump.  This prevents a
    # duplicated shared peak from consuming another finger's two-event quota.
    deduplicated: list[ContactSwitchEvent] = []
    for event in sorted(detected, key=lambda value: (value.event_step, value.finger)):
        duplicate_index = next(
            (
                index
                for index, previous in enumerate(deduplicated)
                if abs(event.event_step - previous.event_step)
                <= settings.cluster_window_steps
                and math.isclose(
                    event.local_peak_abs_jerk_m_s3,
                    previous.local_peak_abs_jerk_m_s3,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and min(event.tangent_jump_m, previous.tangent_jump_m)
                / max(event.tangent_jump_m, previous.tangent_jump_m)
                >= 0.95
            ),
            None,
        )
        if duplicate_index is None:
            deduplicated.append(event)
        elif event.tangent_jump_m > deduplicated[duplicate_index].tangent_jump_m:
            deduplicated[duplicate_index] = event

    per_finger: list[ContactSwitchEvent] = []
    for finger in settings.selected_fingers:
        ranked = sorted(
            (value for value in deduplicated if value.finger == finger),
            key=lambda value: (
                -value.local_peak_abs_jerk_m_s3,
                -value.tangent_jump_m,
                value.event_step,
            ),
        )
        per_finger.extend(ranked[: settings.max_events_per_finger])
    detected = per_finger
    detected.sort(
        key=lambda value: (
            -value.local_peak_abs_jerk_m_s3,
            -value.tangent_jump_m,
            value.event_step,
            value.finger,
        )
    )
    detected = detected[: settings.global_max_events]
    detected.sort(key=lambda value: (value.event_step, value.finger))
    if not detected:
        raise ValueError("no authenticated thumb/index witness-switch event was detected")
    return tuple(detected)


def _normalized_direction(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(np.abs(values), initial=0.0))
    if not math.isfinite(maximum) or maximum <= 1e-10:
        raise ValueError("event Jacobian direction is singular")
    return values / maximum


def solve_event_directions(
    point_jacobian_active: Sequence[Sequence[float]] | np.ndarray,
    rotation_world_from_cube: Sequence[Sequence[float]] | np.ndarray,
    target_face: str,
    tangent_jump_cube_local_m: Sequence[float],
    owned_actuator_indices: Sequence[int],
    *,
    damping: float = 1e-6,
) -> dict[str, Any]:
    """Solve pure per-finger tangent stabilization and normal unloading rays."""

    jacobian = np.asarray(point_jacobian_active, dtype=np.float64)
    rotation = np.asarray(rotation_world_from_cube, dtype=np.float64)
    owned = tuple(int(value) for value in owned_actuator_indices)
    if jacobian.shape != (3, len(ACTIVE_ACTUATORS)) or not np.isfinite(jacobian).all():
        raise ValueError("point_jacobian_active must be finite with shape (3, 8)")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_world_from_cube must be a finite 3x3 matrix")
    if not owned or len(set(owned)) != len(owned) or any(
        value < 0 or value >= len(ACTIVE_ACTUATORS) for value in owned
    ):
        raise ValueError("owned_actuator_indices are invalid")
    if target_face not in _FACE_NORMALS:
        raise ValueError(f"unknown cube face {target_face}")
    tangent_local = _face_tangent(tangent_jump_cube_local_m, target_face)
    tangent_norm = float(np.linalg.norm(tangent_local))
    if tangent_norm <= 1e-10:
        raise ValueError("event has no tangential displacement")
    normal_local = np.asarray(_FACE_NORMALS[target_face], dtype=np.float64)
    tangent_target = -(rotation @ (tangent_local / tangent_norm))
    unload_target = rotation @ normal_local
    owned_jacobian = jacobian[:, owned]
    regularized = owned_jacobian @ owned_jacobian.T + float(damping) * np.eye(3)

    def solve(target: np.ndarray) -> tuple[np.ndarray, float]:
        local = owned_jacobian.T @ np.linalg.solve(regularized, target)
        local = _normalized_direction(local)
        full = np.zeros(len(ACTIVE_ACTUATORS), dtype=np.float64)
        full[list(owned)] = local
        predicted = owned_jacobian @ local
        predicted_norm = float(np.linalg.norm(predicted))
        residual = (
            1.0
            if predicted_norm <= 1e-12
            else float(np.linalg.norm(predicted / predicted_norm - target))
        )
        return full, residual

    tangent_direction, tangent_residual = solve(tangent_target)
    unload_direction, normal_residual = solve(unload_target)
    return {
        "tangent_direction_rad_unit": tuple(float(value) for value in tangent_direction),
        "normal_unload_direction_rad_unit": tuple(float(value) for value in unload_direction),
        "tangent_residual": tangent_residual,
        "normal_residual": normal_residual,
        "jacobian_rank": int(np.linalg.matrix_rank(owned_jacobian)),
    }


def _cube_rotation(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(_finite_tuple(quaternion_wxyz, 4, "cube quaternion"))
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("cube quaternion has zero norm")
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion / norm)
    return matrix.reshape(3, 3)


def _checkpoint_jacobian_directions(
    config: Mapping[str, Any], trace: Mapping[str, Any], event: ContactSwitchEvent
) -> EventJacobianDirections:
    model, info = build_model(copy.deepcopy(dict(config)))
    data = mujoco.MjData(model)
    joint_qpos = _trace_array(trace, "joint_qpos", ndim=2).astype(np.float64)
    actuator_order = tuple(
        str(value) for value in _trace_array(trace, "actuator_order").tolist()
    )
    if len(actuator_order) != joint_qpos.shape[1] or len(set(actuator_order)) != len(actuator_order):
        raise ValueError("trace actuator_order does not match joint_qpos")
    if not set(ACTIVE_ACTUATORS).issubset(actuator_order):
        raise ValueError("trace omits an active actuator")
    step = event.checkpoint_step
    if not 0 <= step < joint_qpos.shape[0]:
        raise ValueError("event checkpoint lies outside joint_qpos")
    for column, name in enumerate(actuator_order):
        try:
            actuator_id = int(model.actuator(name).id)
        except KeyError as exc:
            raise ValueError(f"trace names unknown actuator {name}") from exc
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(joint_qpos[step, column])
    cube_pos = _trace_array(trace, "cube_pos", ndim=2).astype(np.float64)
    cube_quat = _trace_array(trace, "cube_quat", ndim=2).astype(np.float64)
    if cube_pos.shape[0] != joint_qpos.shape[0] or cube_pos.shape[1] != 3 or cube_quat.shape != (joint_qpos.shape[0], 4):
        raise ValueError("trace cube pose shape does not match joint_qpos")
    data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 3] = cube_pos[step]
    data.qpos[info.cube_qpos_adr + 3 : info.cube_qpos_adr + 7] = cube_quat[step]
    mujoco.mj_forward(model, data)

    centroids_world = _trace_array(
        trace, "target_face_contact_centroid_world_m", ndim=3
    ).astype(np.float64)
    valid = _trace_array(trace, "target_face_contact_centroid_valid", ndim=2).astype(bool)
    point_step = step if bool(valid[step, event.finger_index]) else event.event_step - 1
    point = centroids_world[point_step, event.finger_index]
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("event checkpoint has no finite contact witness")
    body_id = int(model.body(DISTAL_BODY_NAMES[event.finger]).id)
    jacp = np.empty((3, model.nv), dtype=np.float64)
    jacr = np.empty((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
    active_jacobian = np.stack(
        [
            jacp[:, int(info.actuator_dof_adrs[int(actuator_id)])]
            for actuator_id in info.active_actuator_ids
        ],
        axis=1,
    )
    target_face = str(config["contact_topology"]["target_faces"][event.finger])
    solved = solve_event_directions(
        active_jacobian,
        _cube_rotation(cube_quat[step]),
        target_face,
        event.tangent_jump_cube_local_m,
        _FINGER_ACTUATOR_INDICES[event.finger],
    )
    payload = {
        "schema_version": EVENT_RESCUE_SCHEMA_VERSION,
        "event_id": event.event_id,
        "finger": event.finger,
        "target_face": target_face,
        **solved,
    }
    return EventJacobianDirections(
        event_id=event.event_id,
        finger=event.finger,
        target_face=target_face,
        tangent_direction_rad_unit=solved["tangent_direction_rad_unit"],
        normal_unload_direction_rad_unit=solved[
            "normal_unload_direction_rad_unit"
        ],
        tangent_residual=float(solved["tangent_residual"]),
        normal_residual=float(solved["normal_residual"]),
        jacobian_rank=int(solved["jacobian_rank"]),
        directions_id=canonical_sha256(payload),
    )


def _descriptor_payload(
    *,
    source_candidate_id: int,
    source_config_semantic_sha256: str,
    source_trace_sha256: str,
    source_trace_content_sha256: str,
    settings: EventDetectionSettings,
    events: Sequence[ContactSwitchEvent],
    directions: Sequence[EventJacobianDirections],
) -> dict[str, Any]:
    return {
        "schema_version": EVENT_RESCUE_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "source_candidate_id": int(source_candidate_id),
        "source_config_semantic_sha256": source_config_semantic_sha256,
        "source_trace_sha256": source_trace_sha256,
        "source_trace_content_sha256": source_trace_content_sha256,
        "detection_settings": settings.as_mapping(),
        "events": [value.as_mapping() for value in events],
        "directions": [value.as_mapping() for value in directions],
    }


def build_event_rescue_descriptor(
    config: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    source_trace_sha256: str,
    source_candidate_id: int,
    settings: EventDetectionSettings = EventDetectionSettings(),
) -> EventRescueDescriptor:
    """Build a descriptor only from hash-bound, full-reset source evidence."""

    if not _sha256(source_trace_sha256):
        raise ValueError("source_trace_sha256 must be a lowercase SHA-256")
    if not isinstance(source_candidate_id, int) or isinstance(source_candidate_id, bool) or source_candidate_id <= 0:
        raise ValueError("source_candidate_id must be a positive integer")
    source_config_sha = canonical_sha256(config)
    trace_content_sha = trace_content_sha256(trace)
    events = detect_contact_switch_events(config, trace, settings=settings)
    directions = tuple(
        _checkpoint_jacobian_directions(config, trace, event) for event in events
    )
    payload = _descriptor_payload(
        source_candidate_id=source_candidate_id,
        source_config_semantic_sha256=source_config_sha,
        source_trace_sha256=source_trace_sha256,
        source_trace_content_sha256=trace_content_sha,
        settings=settings,
        events=events,
        directions=directions,
    )
    return EventRescueDescriptor(
        schema_version=EVENT_RESCUE_SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        source_candidate_id=source_candidate_id,
        source_config_semantic_sha256=source_config_sha,
        source_trace_sha256=source_trace_sha256,
        source_trace_content_sha256=trace_content_sha,
        detection_settings=settings.as_mapping(),
        events=events,
        directions=directions,
        descriptor_id=canonical_sha256(payload),
    )


def authenticate_event_rescue_descriptor(
    descriptor: EventRescueDescriptor,
    config: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    source_trace_sha256: str,
    source_candidate_id: int,
) -> None:
    """Recompute the descriptor and reject any source or descriptor mutation."""

    settings = EventDetectionSettings(**dict(descriptor.detection_settings))
    rebuilt = build_event_rescue_descriptor(
        config,
        trace,
        source_trace_sha256=source_trace_sha256,
        source_candidate_id=source_candidate_id,
        settings=settings,
    )
    if descriptor.as_mapping() != rebuilt.as_mapping():
        raise ValueError("event rescue descriptor authentication failed")


def compact_c2_event_bump(
    progress: Sequence[float] | np.ndarray,
    center: float,
    half_width: float,
) -> np.ndarray:
    """Return a compact sin^4 bump with zero value/derivatives at its edges."""

    values = np.asarray(progress, dtype=np.float64)
    center = float(center)
    half_width = float(half_width)
    if not np.isfinite(values).all() or not math.isfinite(center) or not math.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("event bump inputs must be finite and width positive")
    normalized = (values - (center - half_width)) / (2.0 * half_width)
    result = np.zeros_like(values)
    mask = (normalized > 0.0) & (normalized < 1.0)
    result[mask] = np.sin(np.pi * normalized[mask]) ** 4
    boundary = np.isclose(
        np.abs(values - center),
        half_width,
        rtol=0.0,
        atol=16.0 * np.finfo(np.float64).eps * max(1.0, abs(center), half_width),
    )
    result[boundary] = 0.0
    return result


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=np.float64)
    for column in range(dimensions):
        order = rng.permutation(count)
        result[:, column] = (order + rng.random(count)) / float(count)
    result = 2.0 * result - 1.0
    result[0] = 0.0
    return result


def _scale_from_unit(value: float, bounds: tuple[float, float]) -> float:
    return 1.0 + value * (bounds[1] - 1.0) if value >= 0.0 else 1.0 + value * (1.0 - bounds[0])


def _normalized_centers(
    centers: Sequence[Mapping[str, Any]], event_ids: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for raw in centers:
        parameters = raw.get("parameters", raw)
        amplitudes = parameters.get("event_amplitudes", {})
        if set(amplitudes) != set(event_ids):
            raise ValueError("refinement center amplitudes do not match descriptor events")
        result.append(
            {
                "time_warp_a1": float(parameters["time_warp_a1"]),
                "time_warp_a2": float(parameters["time_warp_a2"]),
                "terminal_scale": float(parameters["terminal_scale"]),
                "event_amplitudes": {
                    event_id: {
                        "tangent_rad": float(amplitudes[event_id]["tangent_rad"]),
                        "unload_rad": float(amplitudes[event_id]["unload_rad"]),
                    }
                    for event_id in event_ids
                },
            }
        )
    if not result:
        raise ValueError("local refinement requires at least one center")
    return tuple(result)


def _sample_parameters(
    descriptor: EventRescueDescriptor,
    budget: EventRescueBudget,
    centers: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    event_ids = tuple(value.event_id for value in descriptor.events)
    dimensions = 2 * len(event_ids) + 3
    seed_payload = canonical_sha256(
        {
            "descriptor_id": descriptor.descriptor_id,
            "budget": budget.as_mapping(),
            "centers": list(centers),
        }
    )
    seed_words = np.frombuffer(bytes.fromhex(seed_payload[:32]), dtype="<u4")
    seed = int(np.random.SeedSequence([budget.seed, *(int(value) for value in seed_words)]).generate_state(1)[0])
    lhs = _latin_hypercube(budget.total_candidate_count, dimensions, seed)
    normalized_centers = (
        _normalized_centers(centers, event_ids)
        if budget.stage == "local_refinement"
        else ()
    )
    output: list[dict[str, Any]] = []
    for index, row in enumerate(lhs):
        if budget.stage == "exploration":
            a1 = float(row[-3] * budget.time_warp_radius)
            a2 = float(row[-2] * budget.time_warp_radius)
            terminal_scale = _scale_from_unit(float(row[-1]), budget.terminal_scale_bounds)
            amplitudes = {
                event_id: {
                    "tangent_rad": float(row[2 * event_index] * budget.tangent_radius_rad),
                    "unload_rad": float((row[2 * event_index + 1] + 1.0) * 0.5 * budget.normal_unload_radius_rad),
                }
                for event_index, event_id in enumerate(event_ids)
            }
            if index == 0:
                a1 = a2 = 0.0
                terminal_scale = 1.0
                amplitudes = {
                    event_id: {"tangent_rad": 0.0, "unload_rad": 0.0}
                    for event_id in event_ids
                }
        else:
            center = normalized_centers[index % len(normalized_centers)]
            # Refinement radii are half the exploration radii and are clipped
            # to the original registered domain.
            a1 = float(np.clip(center["time_warp_a1"] + 0.5 * budget.time_warp_radius * row[-3], *COEFFICIENT_BOUNDS))
            a2 = float(np.clip(center["time_warp_a2"] + 0.5 * budget.time_warp_radius * row[-2], *COEFFICIENT_BOUNDS))
            terminal_scale = float(np.clip(center["terminal_scale"] + 0.5 * (budget.terminal_scale_bounds[1] - budget.terminal_scale_bounds[0]) * row[-1], *budget.terminal_scale_bounds))
            amplitudes = {}
            for event_index, event_id in enumerate(event_ids):
                original = center["event_amplitudes"][event_id]
                amplitudes[event_id] = {
                    "tangent_rad": float(np.clip(original["tangent_rad"] + 0.5 * budget.tangent_radius_rad * row[2 * event_index], -budget.tangent_radius_rad, budget.tangent_radius_rad)),
                    "unload_rad": float(np.clip(original["unload_rad"] + 0.5 * budget.normal_unload_radius_rad * row[2 * event_index + 1], 0.0, budget.normal_unload_radius_rad)),
                }
        output.append(
            {
                "time_warp_a1": a1,
                "time_warp_a2": a2,
                "terminal_scale": terminal_scale,
                "event_amplitudes": amplitudes,
            }
        )
    return tuple(output)


def _event_rescue_planner_id(
    parent_config_sha256: str,
    descriptor_id: str,
    stage: str,
    parameters: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "schema_version": EVENT_RESCUE_SCHEMA_VERSION,
            "kind": "v14_contact_witness_event_rescue",
            "parent_config_semantic_sha256": parent_config_sha256,
            "descriptor_id": descriptor_id,
            "stage": stage,
            "parameters": parameters,
        }
    )


def _materialize_candidate(
    parent_config: Mapping[str, Any],
    descriptor: EventRescueDescriptor,
    parameters: Mapping[str, Any],
    *,
    stage: str,
    half_width: float,
    limits: JointRefinementLimits,
    validate_configs: bool,
) -> tuple[dict[str, Any], dict[str, Any], int, str]:
    parent_plan = ManipulationPlanParameters.from_config(parent_config["manipulation_plan"])
    parent_terminal = {
        name: float(parent_config["control"]["manipulation_delta_rad"][name])
        for name in ACTIVE_ACTUATORS
    }
    direction_by_event = {value.event_id: value for value in descriptor.directions}
    event_by_id = {value.event_id: value for value in descriptor.events}

    for backoff_count, factor in enumerate((1.0, 0.5, 0.25, 0.125, 0.0)):
        a1 = float(parameters["time_warp_a1"]) * factor
        a2 = float(parameters["time_warp_a2"]) * factor
        terminal_scale = 1.0 + (float(parameters["terminal_scale"]) - 1.0) * factor
        terminal_offset = {
            name: parent_terminal[name] * (terminal_scale - 1.0)
            for name in ACTIVE_ACTUATORS
        }
        try:
            config = apply_time_warp_to_config(
                parent_config,
                a1=a1,
                a2=a2,
                terminal_offset_rad=terminal_offset,
                category="warp_terminal",
                limits=limits,
                validate=False,
            )
            plan = ManipulationPlanParameters.from_config(config["manipulation_plan"])
            times = np.asarray(plan.knot_times_s, dtype=np.float64)
            progress = times / DURATION_S
            waypoints = {
                name: np.asarray(plan.actuator_waypoints_rad[name], dtype=np.float64).copy()
                for name in ACTIVE_ACTUATORS
            }
            resolved_amplitudes: dict[str, dict[str, float]] = {}
            for event_id in direction_by_event:
                raw = parameters["event_amplitudes"][event_id]
                tangent = float(raw["tangent_rad"]) * factor
                unload = float(raw["unload_rad"]) * factor
                resolved_amplitudes[event_id] = {
                    "tangent_rad": tangent,
                    "unload_rad": unload,
                }
                bump = compact_c2_event_bump(
                    progress,
                    event_by_id[event_id].manipulation_progress,
                    half_width,
                )
                directions = direction_by_event[event_id]
                vector = tangent * np.asarray(directions.tangent_direction_rad_unit) + unload * np.asarray(directions.normal_unload_direction_rad_unit)
                for actuator_index, name in enumerate(ACTIVE_ACTUATORS):
                    waypoints[name] += bump * float(vector[actuator_index])
            for name in ACTIVE_ACTUATORS:
                waypoints[name][0] = 0.0
            new_plan = ManipulationPlanParameters(
                schema_version=plan.schema_version,
                profile=plan.profile,
                duration_s=plan.duration_s,
                knot_times_s=plan.knot_times_s,
                actuator_waypoints_rad={
                    name: tuple(float(value) for value in waypoints[name])
                    for name in ACTIVE_ACTUATORS
                },
                desired_cube_position_delta_m=plan.desired_cube_position_delta_m,
                desired_cube_rotation_vector_rad=plan.desired_cube_rotation_vector_rad,
                max_knot_delta_rad=plan.max_knot_delta_rad,
                trust_region_backtracks=plan.trust_region_backtracks,
            )
            config["manipulation_plan"] = new_plan.as_config()
            config["control"]["manipulation_delta_rad"] = {
                name: float(waypoints[name][-1]) for name in ACTIVE_ACTUATORS
            }
            config["object_config_id"] = v14_object_config_id(config)
            config["grasp_pose_id"] = v14_grasp_pose_id(config)
            config["grasp_object_pair_id"] = v14_grasp_object_pair_id(config)
            resolved = {
                "time_warp_a1": a1,
                "time_warp_a2": a2,
                "terminal_scale": terminal_scale,
                "event_amplitudes": resolved_amplitudes,
            }
            config["planner_id"] = _event_rescue_planner_id(
                descriptor.source_config_semantic_sha256,
                descriptor.descriptor_id,
                stage,
                resolved,
            )
            config["controller_id"] = _time_warp_controller_id(config)
            validate_time_warped_config(
                config, limits=limits, validate_schema=validate_configs
            )
            return config, resolved, backoff_count, "none" if backoff_count == 0 else "constraint_backoff"
        except (ValueError, np.linalg.LinAlgError):
            continue
    raise ValueError("authenticated parent unexpectedly failed exact-parent materialization")


def build_event_rescue_jobs(
    parent_config: Mapping[str, Any],
    descriptor: EventRescueDescriptor,
    *,
    budget: EventRescueBudget = EventRescueBudget(),
    centers: Sequence[Mapping[str, Any]] = (),
    validate_configs: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Build deterministic full-reset jobs for exploration or refinement."""

    if canonical_sha256(parent_config) != descriptor.source_config_semantic_sha256:
        raise ValueError("parent config does not match the event descriptor")
    descriptor = EventRescueDescriptor.from_mapping(descriptor.as_mapping())
    parameters = _sample_parameters(descriptor, budget, centers)
    limits = resolve_joint_refinement_limits(parent_config)
    jobs: list[dict[str, Any]] = []
    for local_index, requested in enumerate(parameters):
        config, resolved, backoff_count, fallback_reason = _materialize_candidate(
            parent_config,
            descriptor,
            requested,
            stage=budget.stage,
            half_width=budget.bump_half_width_progress,
            limits=limits,
            validate_configs=validate_configs,
        )
        identity = canonical_sha256(
            {
                "schema_version": EVENT_RESCUE_SCHEMA_VERSION,
                "kind": "v14_contact_witness_event_rescue_candidate",
                "descriptor_id": descriptor.descriptor_id,
                "budget": budget.as_mapping(),
                "local_index": local_index,
                "requested_parameters": requested,
                "resolved_parameters": resolved,
            }
        )
        candidate_id = 14 * 10**15 + int(identity[:12], 16) % 10**14
        job_payload = {
            "schema_version": EVENT_RESCUE_SCHEMA_VERSION,
            "stage": budget.stage,
            "candidate_id": candidate_id,
            "local_index": local_index,
            "descriptor_id": descriptor.descriptor_id,
            "source_candidate_id": descriptor.source_candidate_id,
            "config_semantic_sha256": canonical_sha256(config),
            "parameters": resolved,
            "requested_parameters": requested,
            "backoff_count": backoff_count,
            "fallback_reason": fallback_reason,
        }
        job = {
            **job_payload,
            "config": config,
            "candidate_payload_sha256": canonical_sha256(job_payload),
        }
        jobs.append(job)
    candidate_ids = [value["candidate_id"] for value in jobs]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("event rescue candidate ID collision")
    return tuple(jobs)


def build_event_rescue_refinement_jobs(
    parent_config: Mapping[str, Any],
    descriptor: EventRescueDescriptor,
    center_parameters: Sequence[Mapping[str, Any]],
    *,
    budget: EventRescueBudget | None = None,
    validate_configs: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Convenience wrapper for the deterministic 512-job refinement stage."""

    resolved_budget = (
        EventRescueBudget(stage="local_refinement", total_candidate_count=512)
        if budget is None
        else budget
    )
    if resolved_budget.stage != "local_refinement":
        raise ValueError("refinement builder requires a local_refinement budget")
    return build_event_rescue_jobs(
        parent_config,
        descriptor,
        budget=resolved_budget,
        centers=center_parameters,
        validate_configs=validate_configs,
    )


def authenticate_event_rescue_job(
    job: Mapping[str, Any], descriptor: EventRescueDescriptor
) -> None:
    """Fail closed if a generated config, parameter, or binding was changed."""

    required = {
        "schema_version",
        "stage",
        "candidate_id",
        "local_index",
        "descriptor_id",
        "source_candidate_id",
        "config_semantic_sha256",
        "parameters",
        "requested_parameters",
        "backoff_count",
        "fallback_reason",
        "config",
        "candidate_payload_sha256",
    }
    if set(job) != required:
        raise ValueError("event rescue job fields do not match the strict schema")
    if job["descriptor_id"] != descriptor.descriptor_id or job["source_candidate_id"] != descriptor.source_candidate_id:
        raise ValueError("event rescue job is bound to a different descriptor")
    if canonical_sha256(job["config"]) != job["config_semantic_sha256"]:
        raise ValueError("event rescue config authentication failed")
    payload = {key: copy.deepcopy(job[key]) for key in required - {"config", "candidate_payload_sha256"}}
    if canonical_sha256(payload) != job["candidate_payload_sha256"]:
        raise ValueError("event rescue job payload authentication failed")
    validate_time_warped_config(job["config"], validate_schema=True)


__all__ = [
    "ContactSwitchEvent",
    "EventDetectionSettings",
    "EventJacobianDirections",
    "EventRescueBudget",
    "EventRescueDescriptor",
    "authenticate_event_rescue_descriptor",
    "authenticate_event_rescue_job",
    "build_event_rescue_descriptor",
    "build_event_rescue_jobs",
    "build_event_rescue_refinement_jobs",
    "compact_c2_event_bump",
    "detect_contact_switch_events",
    "solve_event_directions",
    "trace_content_sha256",
]
