"""Actual-contact grasp-pose identity and stable-window metrics.

Schema-v9 distinguishes a commanded actuator target from the hand shape that
the simulator actually reaches under contact load.  This module is deliberately
independent of MuJoCo's runtime types: the authoritative evaluator operates on
persisted arrays, while :func:`evaluate_actual_grasp_pose_trace` is a thin
adapter for the repository's model/config/trace convention.

An actual grasp pose locks on the final sample of the earliest complete window
that satisfies the caller-provided contact/safety gate *and* all joint-shape
conditions.  The preload command is retained as diagnostic evidence only; it
never contributes to a grasp-pose check.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import ACTIVE_ACTUATORS


THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
_EPSILON = 1e-12


def _readonly(values: ArrayLike, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _finite_active_mapping(values: object, label: str) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(ACTIVE_ACTUATORS):
        raise ValueError(f"{label} must name exactly the eight active actuators")
    result = {name: float(values[name]) for name in ACTIVE_ACTUATORS}
    if not np.isfinite(tuple(result.values())).all():
        raise ValueError(f"{label} must contain only finite values")
    return result


def _finite_vector(values: object, length: int, label: str) -> list[float]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != length
    ):
        raise ValueError(f"{label} must contain {length} values")
    result = [float(value) for value in values]
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    return result


def canonical_sha256(value: Any) -> str:
    """Hash JSON data with deterministic mapping order and no non-finite values."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def grasp_pose_context(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only geometry/topology and the nominal *actual* joint shape.

    Control targets and timing are intentionally absent.  Conversely, the
    complete cube mapping is retained because size and placement are part of
    the contact geometry even when mass and friction happen to be unchanged.
    """

    grasp_pose = config.get("grasp_pose")
    if not isinstance(grasp_pose, Mapping):
        raise ValueError("grasp_pose must be a mapping")
    nominal = _finite_active_mapping(
        grasp_pose.get("nominal_joint_qpos_rad"),
        "grasp_pose.nominal_joint_qpos_rad",
    )
    for key in ("cube", "hand_pose", "contact_topology", "scene"):
        if not isinstance(config.get(key), Mapping):
            raise ValueError(f"{key} must be a mapping")
    cube = config["cube"]
    hand = config["hand_pose"]
    topology = config["contact_topology"]
    scene = config["scene"]
    edge_m = float(cube.get("edge_m"))
    support_top_z_m = float(scene.get("support_top_z_m"))
    z_offset_m = float(cube.get("z_offset_m", 0.0))
    if not math.isfinite(edge_m) or edge_m <= 0.0:
        raise ValueError("cube.edge_m must be positive and finite")
    if not math.isfinite(support_top_z_m):
        raise ValueError("scene.support_top_z_m must be finite")
    if not math.isfinite(z_offset_m):
        raise ValueError("cube.z_offset_m must be finite")
    target_faces = topology.get("target_faces")
    if not isinstance(target_faces, Mapping):
        raise ValueError("contact_topology.target_faces must be a mapping")
    context = {
        "cube_geometry_and_pose": {
            "edge_m": edge_m,
            "center_xy_m": _finite_vector(
                cube.get("center_xy_m"), 2, "cube.center_xy_m"
            ),
            "rpy_deg": _finite_vector(
                cube.get("rpy_deg", (0.0, 0.0, 0.0)), 3, "cube.rpy_deg"
            ),
            "z_offset_m": z_offset_m,
            "support_top_z_m": support_top_z_m,
        },
        "hand_root_pose": {
            "translation_m": _finite_vector(
                hand.get("translation_m"), 3, "hand_pose.translation_m"
            ),
            "rpy_deg": _finite_vector(
                hand.get("rpy_deg"), 3, "hand_pose.rpy_deg"
            ),
        },
        "target_faces": copy.deepcopy(dict(target_faces)),
        "nominal_joint_qpos_rad": nominal,
    }
    # Schema-v12 freezes one cube-local three-point contact plan before the
    # dynamic grasp search begins.  That plan is geometry, not controller
    # state: two otherwise identical hand poses aimed at different surface
    # points must never share evidence or a catalog identity.  Older schemas
    # intentionally omit the key and retain their byte-for-byte identity.
    if "contact_point_plan" in config:
        point_plan = config.get("contact_point_plan")
        if not isinstance(point_plan, Mapping):
            raise ValueError("contact_point_plan must be a mapping")
        context["contact_point_plan"] = copy.deepcopy(dict(point_plan))
    return context


def grasp_pose_id(config: Mapping[str, Any]) -> str:
    """Stable identity for one actual-contact hand/object pose."""

    return canonical_sha256(grasp_pose_context(config))


def controller_context(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the versioned controller state used for identity.

    The nominal actual qpos is not a command and therefore cannot affect this
    identity.  Keeping the entire versioned protocol makes gate-trigger timing
    auditable without coupling this helper to one particular protocol revision.
    Schema v12 is a grasp-only campaign, so its diagnostic manipulation probe
    is intentionally outside the controller identity; older schemas retain
    their historical manipulation-bound identity unchanged.
    """

    control = config.get("control")
    protocol = config.get("control_protocol")
    if not isinstance(control, Mapping):
        raise ValueError("control must be a mapping")
    if not isinstance(protocol, Mapping):
        raise ValueError("control_protocol must be a mapping")
    context = {
        "precontact_targets_rad": _finite_active_mapping(
            control.get("precontact_targets_rad"),
            "control.precontact_targets_rad",
        ),
        "contact_preload_targets_rad": _finite_active_mapping(
            control.get("contact_preload_targets_rad"),
            "control.contact_preload_targets_rad",
        ),
        "close_profile": copy.deepcopy(control.get("close_profile")),
        "control_protocol": copy.deepcopy(protocol),
    }
    if int(config.get("schema_version", 1)) < 12:
        context["manipulation_delta_rad"] = _finite_active_mapping(
            control.get("manipulation_delta_rad"),
            "control.manipulation_delta_rad",
        )
    return context


def controller_id(config: Mapping[str, Any]) -> str:
    """Stable identity for a controller, independent of nominal actual qpos."""

    return canonical_sha256(controller_context(config))


@dataclass(frozen=True, slots=True)
class ActualGraspPoseThresholds:
    """Hard actual-qpos conditions for the schema-v9 verification window."""

    verify_continuous_s: float = 0.25
    thumb_actual_min_rad: float = 1.40
    thumb_actual_max_rad: float = 1.60
    max_nominal_joint_error_rad: float = 0.04
    max_joint_stability_span_rad: float = 0.03

    def __post_init__(self) -> None:
        for name in (
            "verify_continuous_s",
            "max_nominal_joint_error_rad",
            "max_joint_stability_span_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        lower = float(self.thumb_actual_min_rad)
        upper = float(self.thumb_actual_max_rad)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("thumb actual range must be finite and increasing")
        object.__setattr__(self, "thumb_actual_min_rad", lower)
        object.__setattr__(self, "thumb_actual_max_rad", upper)

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any]
    ) -> "ActualGraspPoseThresholds":
        grasp_pose = config.get("grasp_pose")
        protocol = config.get("control_protocol")
        if not isinstance(grasp_pose, Mapping):
            raise ValueError("grasp_pose must be a mapping")
        actual_range = grasp_pose.get("thumb_actual_range_rad")
        if (
            not isinstance(actual_range, Sequence)
            or isinstance(actual_range, (str, bytes))
            or len(actual_range) != 2
        ):
            raise ValueError("grasp_pose.thumb_actual_range_rad must have two values")
        continuous = grasp_pose.get("verify_continuous_s")
        if continuous is None:
            if not isinstance(protocol, Mapping):
                raise ValueError("control_protocol must be a mapping")
            continuous = protocol.get("stable_window_s")
        return cls(
            verify_continuous_s=float(continuous),
            thumb_actual_min_rad=float(actual_range[0]),
            thumb_actual_max_rad=float(actual_range[1]),
            max_nominal_joint_error_rad=float(
                grasp_pose["max_nominal_joint_error_rad"]
            ),
            max_joint_stability_span_rad=float(
                grasp_pose["max_joint_stability_span_rad"]
            ),
        )

    def as_config(self) -> dict[str, Any]:
        return {
            "verify_continuous_s": self.verify_continuous_s,
            "thumb_actual_range_rad": [
                self.thumb_actual_min_rad,
                self.thumb_actual_max_rad,
            ],
            "max_nominal_joint_error_rad": self.max_nominal_joint_error_rad,
            "max_joint_stability_span_rad": self.max_joint_stability_span_rad,
        }


def continuous_window_steps(timestep_s: float, duration_s: float) -> int:
    """Resolve the sample count used by the existing post-step trace convention."""

    timestep = float(timestep_s)
    duration = float(duration_s)
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("timestep_s must be positive and finite")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be positive and finite")
    steps = int(round(duration / timestep))
    if steps <= 0:
        raise ValueError("duration_s must contain at least one simulation step")
    return steps


@dataclass(frozen=True, slots=True)
class GraspPoseEvents:
    """Inclusive event-frame convention for an actual grasp pose."""

    first_contact_step: int
    stable_window_start_step: int
    stable_window_end_step: int
    grasp_lock_step: int
    required_continuous_steps: int

    def __post_init__(self) -> None:
        required = int(self.required_continuous_steps)
        if required <= 0:
            raise ValueError("required_continuous_steps must be positive")
        object.__setattr__(self, "required_continuous_steps", required)
        for name in (
            "first_contact_step",
            "stable_window_start_step",
            "stable_window_end_step",
            "grasp_lock_step",
        ):
            value = int(getattr(self, name))
            if value < -1:
                raise ValueError(f"{name} must be -1 or a non-negative frame")
            object.__setattr__(self, name, value)
        locked = self.grasp_lock_step >= 0
        if locked:
            if self.stable_window_start_step < 0:
                raise ValueError("a locked grasp requires a stable-window start")
            if self.stable_window_end_step != self.grasp_lock_step:
                raise ValueError("grasp_lock_step must equal the inclusive window end")
            count = self.stable_window_end_step - self.stable_window_start_step + 1
            if count != required:
                raise ValueError("locked stable window has the wrong sample count")
        elif self.stable_window_start_step != -1 or self.stable_window_end_step != -1:
            raise ValueError("an unlocked grasp cannot declare a stable window")

    def as_dict(self) -> dict[str, int]:
        return {
            "first_contact_step": self.first_contact_step,
            "stable_window_start_step": self.stable_window_start_step,
            "stable_window_end_step": self.stable_window_end_step,
            "grasp_lock_step": self.grasp_lock_step,
            "required_continuous_steps": self.required_continuous_steps,
        }


@dataclass(frozen=True, slots=True)
class ActualGraspPoseResult:
    """Auditable metrics for the selected authoritative or diagnostic window."""

    actuator_names: tuple[str, ...]
    events: GraspPoseEvents
    checks: Mapping[str, bool]
    metrics: Mapping[str, Any]
    actual_qpos_median_rad: NDArray[np.float64]
    nominal_error_rad: NDArray[np.float64]
    joint_stability_span_rad: NDArray[np.float64]
    contact_preload_command_rad: NDArray[np.float64]

    def __post_init__(self) -> None:
        count = len(self.actuator_names)
        if tuple(self.actuator_names) != tuple(ACTIVE_ACTUATORS):
            raise ValueError("actuator_names must use canonical active-actuator order")
        for name in (
            "actual_qpos_median_rad",
            "nominal_error_rad",
            "joint_stability_span_rad",
            "contact_preload_command_rad",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain {count} finite values")
            object.__setattr__(self, name, _readonly(values))
        object.__setattr__(
            self, "checks", {str(name): bool(value) for name, value in self.checks.items()}
        )
        object.__setattr__(self, "metrics", copy.deepcopy(dict(self.metrics)))

    @property
    def passed(self) -> bool:
        return bool(self.checks.get("actual_grasp_pose_locked", False))

    def as_trace_fields(self) -> dict[str, np.ndarray]:
        """Return schema-v9 NPZ-ready summary/event fields."""

        return {
            "grasp_pose_actual_qpos_rad": self.actual_qpos_median_rad.copy(),
            "grasp_pose_nominal_error_rad": self.nominal_error_rad.copy(),
            "grasp_pose_joint_stability_span_rad": (
                self.joint_stability_span_rad.copy()
            ),
            "contact_preload_command_rad": self.contact_preload_command_rad.copy(),
            "first_contact_step": np.asarray(
                self.events.first_contact_step, dtype=np.int64
            ),
            "grasp_stable_window_start_step": np.asarray(
                self.events.stable_window_start_step, dtype=np.int64
            ),
            "grasp_stable_window_end_step": np.asarray(
                self.events.stable_window_end_step, dtype=np.int64
            ),
            "grasp_lock_step": np.asarray(
                self.events.grasp_lock_step, dtype=np.int64
            ),
        }

    def as_summary(self) -> dict[str, Any]:
        """Return JSON-safe named metrics, checks and event frames."""

        named = lambda values: {
            name: float(values[index]) for index, name in enumerate(self.actuator_names)
        }
        return {
            "passed": self.passed,
            "events": self.events.as_dict(),
            "checks": dict(self.checks),
            "metrics": copy.deepcopy(dict(self.metrics)),
            "actual_qpos_median_rad": named(self.actual_qpos_median_rad),
            "nominal_error_rad": named(self.nominal_error_rad),
            "joint_stability_span_rad": named(self.joint_stability_span_rad),
            "contact_preload_command_rad": named(self.contact_preload_command_rad),
        }


def _joint_matrix(values: ArrayLike, label: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    expected_columns = len(ACTIVE_ACTUATORS)
    if (
        result.ndim != 2
        or result.shape[0] == 0
        or result.shape[1] != expected_columns
        or not np.isfinite(result).all()
    ):
        raise ValueError(
            f"{label} must have shape (T, {expected_columns}) and be finite"
        )
    return result


def _mask(values: ArrayLike, length: int, label: str) -> NDArray[np.bool_]:
    result = np.asarray(values)
    if result.shape != (length,):
        raise ValueError(f"{label} must have shape ({length},)")
    return result.astype(bool, copy=False)


def _command_matrix(values: ArrayLike | None, length: int) -> NDArray[np.float64]:
    columns = len(ACTIVE_ACTUATORS)
    if values is None:
        return np.zeros((length, columns), dtype=np.float64)
    command = np.asarray(values, dtype=np.float64)
    if command.shape == (columns,):
        command = np.broadcast_to(command, (length, columns))
    if command.shape != (length, columns) or not np.isfinite(command).all():
        raise ValueError(
            "contact_preload_command_rad must have shape (8,) or (T, 8) "
            "and be finite"
        )
    return command


def _true_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.concatenate((np.asarray([False]), mask, np.asarray([False])))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end - 1)) for start, end in changes.reshape(-1, 2)]


def evaluate_actual_grasp_pose(
    actual_joint_qpos_rad: ArrayLike,
    nominal_joint_qpos_rad: Mapping[str, float] | ArrayLike,
    base_gate_mask: ArrayLike,
    contact_mask: ArrayLike,
    *,
    timestep_s: float,
    thresholds: ActualGraspPoseThresholds | None = None,
    contact_preload_command_rad: ArrayLike | None = None,
) -> ActualGraspPoseResult:
    """Find and evaluate the earliest stable actual-contact grasp window.

    ``base_gate_mask`` contains all non-qpos contact, topology, safety and pose
    conditions.  Separating it from this evaluator prevents a commanded target
    from being smuggled into the actual-qpos decision.  Window endpoints are
    inclusive; a 250 ms window at 1 ms has 250 samples and locks at
    ``start + 249``.
    """

    settings = thresholds or ActualGraspPoseThresholds()
    actual = _joint_matrix(actual_joint_qpos_rad, "actual_joint_qpos_rad")
    count = actual.shape[0]
    if isinstance(nominal_joint_qpos_rad, Mapping):
        nominal_mapping = _finite_active_mapping(
            nominal_joint_qpos_rad, "nominal_joint_qpos_rad"
        )
        nominal = np.asarray(
            [nominal_mapping[name] for name in ACTIVE_ACTUATORS], dtype=np.float64
        )
    else:
        nominal = np.asarray(nominal_joint_qpos_rad, dtype=np.float64)
        if nominal.shape != (len(ACTIVE_ACTUATORS),) or not np.isfinite(nominal).all():
            raise ValueError("nominal_joint_qpos_rad must contain eight finite values")
    base_gate = _mask(base_gate_mask, count, "base_gate_mask")
    contact = _mask(contact_mask, count, "contact_mask")
    command = _command_matrix(contact_preload_command_rad, count)
    required = continuous_window_steps(timestep_s, settings.verify_continuous_s)
    thumb_index = ACTIVE_ACTUATORS.index(THUMB_BEND_ACTUATOR)
    first_contact_indices = np.flatnonzero(contact)
    first_contact = int(first_contact_indices[0]) if first_contact_indices.size else -1

    runs = _true_runs(base_gate)
    complete_base_window = any(end - start + 1 >= required for start, end in runs)
    selected: tuple[int, int] | None = None
    selected_values: tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool] | None = None
    best_key: tuple[float, ...] | None = None

    def inspect(start: int, end: int):
        window = actual[start : end + 1]
        median = np.percentile(window, 50.0, axis=0)
        p05 = np.percentile(window, 5.0, axis=0)
        p95 = np.percentile(window, 95.0, axis=0)
        span = p95 - p05
        error = np.abs(median - nominal)
        thumb = window[:, thumb_index]
        thumb_ok = bool(
            np.all(thumb >= settings.thumb_actual_min_rad - _EPSILON)
            and np.all(thumb <= settings.thumb_actual_max_rad + _EPSILON)
        )
        nominal_ok = bool(
            np.all(error <= settings.max_nominal_joint_error_rad + _EPSILON)
        )
        span_ok = bool(
            np.all(span <= settings.max_joint_stability_span_rad + _EPSILON)
        )
        return median, error, span, thumb_ok, nominal_ok and span_ok

    # Every complete gate window is authoritative evidence.  Stop at the first
    # hard pass; otherwise retain the least-violating complete window for an
    # auditable near-miss diagnosis.
    for run_start, run_end in runs:
        if run_end - run_start + 1 < required:
            continue
        for start in range(run_start, run_end - required + 2):
            end = start + required - 1
            values = inspect(start, end)
            median, error, span, thumb_ok, _ = values
            nominal_ok = bool(
                np.all(error <= settings.max_nominal_joint_error_rad + _EPSILON)
            )
            span_ok = bool(
                np.all(span <= settings.max_joint_stability_span_rad + _EPSILON)
            )
            pass_window = thumb_ok and nominal_ok and span_ok
            thumb_window = actual[start : end + 1, thumb_index]
            key = (
                0.0 if pass_window else 1.0,
                max(0.0, settings.thumb_actual_min_rad - float(np.min(thumb_window))),
                max(0.0, float(np.max(thumb_window)) - settings.thumb_actual_max_rad),
                float(np.max(error / settings.max_nominal_joint_error_rad)),
                float(np.max(span / settings.max_joint_stability_span_rad)),
                float(start),
            )
            if best_key is None or key < best_key:
                best_key = key
                selected = (start, end)
                selected_values = values
            if pass_window:
                break
        if selected is not None and best_key is not None and best_key[0] == 0.0:
            break

    # When no full gate window exists, report the longest contiguous fragment
    # as diagnostic evidence while keeping all lock event frames at -1.
    if selected is None and runs:
        run_start, run_end = min(
            runs,
            key=lambda run: (-(run[1] - run[0] + 1), run[0]),
        )
        start = max(run_start, run_end - required + 1)
        selected = (start, run_end)
        selected_values = inspect(start, run_end)

    if selected is None or selected_values is None:
        diagnostic_start = diagnostic_end = -1
        diagnostic_samples = 0
        median = np.zeros(len(ACTIVE_ACTUATORS), dtype=np.float64)
        error = np.abs(median - nominal)
        span = np.zeros(len(ACTIVE_ACTUATORS), dtype=np.float64)
        thumb_ok = nominal_and_span_ok = False
        preload = np.median(command, axis=0)
        thumb_min = thumb_max = 0.0
    else:
        diagnostic_start, diagnostic_end = selected
        diagnostic_samples = diagnostic_end - diagnostic_start + 1
        median, error, span, thumb_ok, nominal_and_span_ok = selected_values
        preload = np.median(command[diagnostic_start : diagnostic_end + 1], axis=0)
        thumb_values = actual[diagnostic_start : diagnostic_end + 1, thumb_index]
        thumb_min = float(np.min(thumb_values))
        thumb_max = float(np.max(thumb_values))

    nominal_ok = bool(
        diagnostic_samples > 0
        and np.all(error <= settings.max_nominal_joint_error_rad + _EPSILON)
    )
    span_ok = bool(
        diagnostic_samples > 0
        and np.all(span <= settings.max_joint_stability_span_rad + _EPSILON)
    )
    locked = bool(complete_base_window and thumb_ok and nominal_ok and span_ok)
    if locked:
        stable_start, stable_end = diagnostic_start, diagnostic_end
        lock_step = stable_end
    else:
        stable_start = stable_end = lock_step = -1
    events = GraspPoseEvents(
        first_contact_step=first_contact,
        stable_window_start_step=stable_start,
        stable_window_end_step=stable_end,
        grasp_lock_step=lock_step,
        required_continuous_steps=required,
    )
    checks = {
        "grasp_pose_base_gate_contiguous": complete_base_window,
        "thumb_actual_qpos_within_range": bool(diagnostic_samples > 0 and thumb_ok),
        "actual_joint_median_matches_nominal": nominal_ok,
        "actual_joint_window_stable": span_ok,
        "actual_grasp_pose_locked": locked,
    }
    metrics = {
        "verify_continuous_s": settings.verify_continuous_s,
        "required_continuous_steps": required,
        "resolved_continuous_s": required * float(timestep_s),
        "diagnostic_window_start_step": diagnostic_start,
        "diagnostic_window_end_step": diagnostic_end,
        "diagnostic_window_sample_count": diagnostic_samples,
        "thumb_actual_min_rad": thumb_min,
        "thumb_actual_median_rad": float(median[thumb_index]),
        "thumb_actual_max_rad": thumb_max,
        "maximum_nominal_joint_error_rad": float(np.max(error)),
        "maximum_joint_stability_span_rad": float(np.max(span)),
        "preload_command_used_as_acceptance_evidence": False,
    }
    return ActualGraspPoseResult(
        actuator_names=tuple(ACTIVE_ACTUATORS),
        events=events,
        checks=checks,
        metrics=metrics,
        actual_qpos_median_rad=median,
        nominal_error_rad=error,
        joint_stability_span_rad=span,
        contact_preload_command_rad=preload,
    )


def evaluate_actual_grasp_pose_trace(
    model: Any,
    config: Mapping[str, Any],
    traces: Mapping[str, ArrayLike],
    *,
    base_gate_mask: ArrayLike | None = None,
    contact_mask: ArrayLike | None = None,
) -> ActualGraspPoseResult:
    """Adapt the repository's model/config/trace layout to the pure evaluator."""

    timestep = float(model.opt.timestep)
    ids = np.asarray(
        [int(model.actuator(name).id) for name in ACTIVE_ACTUATORS], dtype=np.int64
    )
    joint_qpos = np.asarray(traces["joint_qpos"], dtype=np.float64)
    if joint_qpos.ndim != 2 or joint_qpos.shape[1] <= int(np.max(ids)):
        raise ValueError("joint_qpos does not contain all active actuator columns")
    total_steps = joint_qpos.shape[0]
    if base_gate_mask is None:
        if "grasp_pose_base_gate" in traces:
            base_gate_mask = traces["grasp_pose_base_gate"]
        else:
            grasp_gate = np.asarray(traces["grasp_gate"])
            if grasp_gate.ndim != 2 or grasp_gate.shape[0] != total_steps:
                raise ValueError("grasp_gate must have shape (T, G)")
            base_gate_mask = np.all(grasp_gate.astype(bool), axis=1)
    if contact_mask is None:
        if "finger_contact_force" in traces:
            force = np.asarray(traces["finger_contact_force"], dtype=np.float64)
            if force.shape != (total_steps, 3) or not np.isfinite(force).all():
                raise ValueError("finger_contact_force must have shape (T, 3)")
            # This event is the first real, force-producing distal contact.  It
            # intentionally precedes face-purity/effective-force qualification;
            # those stricter signals remain part of ``base_gate_mask``.
            contact_mask = np.any(force > 1e-8, axis=1)
        elif "target_face_effective" in traces:
            effective = np.asarray(traces["target_face_effective"])
            if effective.shape != (total_steps, 3):
                raise ValueError("target_face_effective must have shape (T, 3)")
            contact_mask = np.any(effective.astype(bool), axis=1)
        else:
            contact_mask = base_gate_mask
    grasp_pose = config.get("grasp_pose")
    control = config.get("control")
    if not isinstance(grasp_pose, Mapping) or not isinstance(control, Mapping):
        raise ValueError("schema-v9 config requires grasp_pose and control mappings")
    nominal = _finite_active_mapping(
        grasp_pose.get("nominal_joint_qpos_rad"),
        "grasp_pose.nominal_joint_qpos_rad",
    )
    preload = _finite_active_mapping(
        control.get("contact_preload_targets_rad"),
        "control.contact_preload_targets_rad",
    )
    return evaluate_actual_grasp_pose(
        joint_qpos[:, ids],
        nominal,
        base_gate_mask,
        contact_mask,
        timestep_s=timestep,
        thresholds=ActualGraspPoseThresholds.from_config(config),
        contact_preload_command_rad=np.asarray(
            [preload[name] for name in ACTIVE_ACTUATORS], dtype=np.float64
        ),
    )


__all__ = [
    "ActualGraspPoseResult",
    "ActualGraspPoseThresholds",
    "GraspPoseEvents",
    "THUMB_BEND_ACTUATOR",
    "canonical_sha256",
    "continuous_window_steps",
    "controller_context",
    "controller_id",
    "evaluate_actual_grasp_pose",
    "evaluate_actual_grasp_pose_trace",
    "grasp_pose_context",
    "grasp_pose_id",
]
