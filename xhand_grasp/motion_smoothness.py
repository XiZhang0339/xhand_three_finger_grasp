"""Gravity-relative path and smoothness metrics for schema-v8 manipulation.

All functions operate on persisted numeric arrays and have no MuJoCo state
dependency.  The same evaluator can therefore be used while ranking an in-
memory run and while auditing an NPZ trace.  A centred moving average is
returned only where the complete odd-width window exists; no padded endpoint
samples are invented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


_NUMERIC_EPSILON = 1e-12


def _readonly(values: ArrayLike, *, dtype: type = np.float64) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def centered_window_steps(timestep_s: float, window_s: float = 0.051) -> int:
    """Resolve an odd, timestep-aligned centred-window sample count."""

    timestep = float(timestep_s)
    window = float(window_s)
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("timestep_s must be positive and finite")
    if not math.isfinite(window) or window <= 0.0:
        raise ValueError("window_s must be positive and finite")
    steps = int(round(window / timestep))
    if steps <= 0 or abs(steps * timestep - window) > 1e-12:
        raise ValueError("window_s does not align with timestep_s")
    if steps % 2 == 0:
        raise ValueError("a centered valid moving-average window must have odd width")
    return steps


def centered_valid_indices(sample_count: int, window_steps: int) -> NDArray[np.int64]:
    """Return source indices at the centres of all complete windows."""

    count = int(sample_count)
    width = int(window_steps)
    if isinstance(sample_count, bool) or count != sample_count or count < 0:
        raise ValueError("sample_count must be a non-negative integer")
    if isinstance(window_steps, bool) or width != window_steps or width <= 0:
        raise ValueError("window_steps must be a positive integer")
    if width % 2 == 0:
        raise ValueError("window_steps must be odd")
    if count < width:
        return np.empty(0, dtype=np.int64)
    half = width // 2
    return np.arange(half, count - half, dtype=np.int64)


def centered_valid_moving_average(
    values: ArrayLike,
    *,
    timestep_s: float,
    window_s: float = 0.051,
) -> NDArray[np.float64]:
    """Average over complete centred windows along axis zero.

    The returned length is ``T - window_steps + 1``.  One-dimensional and
    fixed-width vector traces are both supported.
    """

    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim < 1 or not np.isfinite(samples).all():
        raise ValueError("values must be a finite array with at least one axis")
    width = centered_window_steps(timestep_s, window_s)
    if samples.shape[0] < width:
        return np.empty((0,) + samples.shape[1:], dtype=np.float64)
    # Prefixing one exact zero avoids special-casing the first complete window
    # and gives identical arithmetic for scalar and vector-valued traces.
    prefix = np.concatenate(
        (
            np.zeros((1,) + samples.shape[1:], dtype=np.float64),
            np.cumsum(samples, axis=0, dtype=np.float64),
        ),
        axis=0,
    )
    return (prefix[width:] - prefix[:-width]) / float(width)


@dataclass(frozen=True, slots=True)
class GravityRelativeMotion:
    """Object path expressed relative to one baseline pose and gravity."""

    up_world: NDArray[np.float64]
    height_m: NDArray[np.float64]
    lateral_displacement_world_m: NDArray[np.float64]
    lateral_distance_m: NDArray[np.float64]
    orientation_drift_deg: NDArray[np.float64]

    def __post_init__(self) -> None:
        up = np.asarray(self.up_world, dtype=np.float64)
        height = np.asarray(self.height_m, dtype=np.float64)
        lateral = np.asarray(self.lateral_displacement_world_m, dtype=np.float64)
        lateral_distance = np.asarray(self.lateral_distance_m, dtype=np.float64)
        orientation = np.asarray(self.orientation_drift_deg, dtype=np.float64)
        if up.shape != (3,) or not np.isfinite(up).all():
            raise ValueError("up_world must contain three finite values")
        if not math.isclose(float(np.linalg.norm(up)), 1.0, abs_tol=1e-12):
            raise ValueError("up_world must be unit length")
        if height.ndim != 1 or not np.isfinite(height).all():
            raise ValueError("height_m must be a finite one-dimensional trace")
        expected = (height.shape[0], 3)
        if lateral.shape != expected or not np.isfinite(lateral).all():
            raise ValueError(
                f"lateral_displacement_world_m must have shape {expected}"
            )
        for values, label in (
            (lateral_distance, "lateral_distance_m"),
            (orientation, "orientation_drift_deg"),
        ):
            if values.shape != height.shape or not np.isfinite(values).all():
                raise ValueError(f"{label} must match height_m and be finite")
            if np.any(values < 0.0):
                raise ValueError(f"{label} must be non-negative")
        object.__setattr__(self, "up_world", _readonly(up))
        object.__setattr__(self, "height_m", _readonly(height))
        object.__setattr__(
            self, "lateral_displacement_world_m", _readonly(lateral)
        )
        object.__setattr__(self, "lateral_distance_m", _readonly(lateral_distance))
        object.__setattr__(self, "orientation_drift_deg", _readonly(orientation))


def _unit_quaternion_trace(values: ArrayLike) -> NDArray[np.float64]:
    quaternion = np.asarray(values, dtype=np.float64)
    if (
        quaternion.ndim != 2
        or quaternion.shape[1] != 4
        or not np.isfinite(quaternion).all()
    ):
        raise ValueError("cube_quaternion must have shape (T, 4) and be finite")
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(norms <= _NUMERIC_EPSILON):
        raise ValueError("cube_quaternion must not contain a zero quaternion")
    return quaternion / norms[:, np.newaxis]


def gravity_relative_motion(
    cube_position_world_m: ArrayLike,
    cube_quaternion: ArrayLike,
    gravity_world_m_s2: ArrayLike,
    *,
    baseline_step: int,
) -> GravityRelativeMotion:
    """Resolve height, lateral displacement and shortest quaternion drift."""

    position = np.asarray(cube_position_world_m, dtype=np.float64)
    if (
        position.ndim != 2
        or position.shape[1] != 3
        or not np.isfinite(position).all()
        or position.shape[0] == 0
    ):
        raise ValueError("cube_position_world_m must have shape (T, 3) and be finite")
    quaternion = _unit_quaternion_trace(cube_quaternion)
    if quaternion.shape[0] != position.shape[0]:
        raise ValueError("cube position and quaternion traces must have equal length")
    gravity = np.asarray(gravity_world_m_s2, dtype=np.float64)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("gravity_world_m_s2 must contain three finite values")
    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm <= _NUMERIC_EPSILON:
        raise ValueError("gravity_world_m_s2 must be non-zero")
    baseline = int(baseline_step)
    if (
        isinstance(baseline_step, bool)
        or baseline != baseline_step
        or not 0 <= baseline < position.shape[0]
    ):
        raise ValueError("baseline_step is outside the trace")

    up = -gravity / gravity_norm
    displacement = position - position[baseline]
    height = displacement @ up
    lateral = displacement - height[:, np.newaxis] * up
    lateral_distance = np.linalg.norm(lateral, axis=1)
    cosine = np.clip(
        np.abs(quaternion @ quaternion[baseline]),
        0.0,
        1.0,
    )
    orientation = np.degrees(2.0 * np.arccos(cosine))
    return GravityRelativeMotion(
        up_world=up,
        height_m=height,
        lateral_displacement_world_m=lateral,
        lateral_distance_m=lateral_distance,
        orientation_drift_deg=orientation,
    )


@dataclass(frozen=True, slots=True)
class MotionSmoothnessThresholds:
    """Versioned schema-v8 path thresholds with plan-default values."""

    filter_window_s: float = 0.051
    downward_speed_threshold_m_s: float = -0.001
    max_downward_speed_duty: float = 0.02
    max_cumulative_height_backtrack_m: float = 0.0002
    max_peak_upward_speed_m_s: float = 0.020
    max_abs_vertical_acceleration_m_s2: float = 0.12
    max_abs_vertical_jerk_m_s3: float = 2.5
    max_hold_entry_linear_speed_m_s: float = 0.005
    max_lateral_displacement_m: float = 0.002
    max_orientation_drift_deg: float = 10.0

    def __post_init__(self) -> None:
        positive = (
            "filter_window_s",
            "max_cumulative_height_backtrack_m",
            "max_peak_upward_speed_m_s",
            "max_abs_vertical_acceleration_m_s2",
            "max_abs_vertical_jerk_m_s3",
            "max_hold_entry_linear_speed_m_s",
            "max_lateral_displacement_m",
            "max_orientation_drift_deg",
        )
        for field in positive:
            value = float(getattr(self, field))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be positive and finite")
            object.__setattr__(self, field, value)
        downward = float(self.downward_speed_threshold_m_s)
        if not math.isfinite(downward) or downward >= 0.0:
            raise ValueError(
                "downward_speed_threshold_m_s must be finite and negative"
            )
        object.__setattr__(self, "downward_speed_threshold_m_s", downward)
        duty = float(self.max_downward_speed_duty)
        if not math.isfinite(duty) or not 0.0 <= duty <= 1.0:
            raise ValueError("max_downward_speed_duty must be within [0, 1]")
        object.__setattr__(self, "max_downward_speed_duty", duty)

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "MotionSmoothnessThresholds":
        expected = set(cls.__dataclass_fields__)
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise ValueError(
                "motion_smoothness fields do not match the schema "
                f"(missing={missing}, extra={extra})"
            )
        return cls(**{name: values[name] for name in expected})

    def as_config(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class SmoothMotionEvaluation:
    """Serializable metrics/checks plus deterministic derived time series."""

    metrics: Mapping[str, float | int]
    checks: Mapping[str, bool]
    relative_motion: GravityRelativeMotion
    filtered_height_m: NDArray[np.float64]
    filtered_center_steps: NDArray[np.int64]
    vertical_speed_m_s: NDArray[np.float64]
    vertical_acceleration_m_s2: NDArray[np.float64]
    vertical_jerk_m_s3: NDArray[np.float64]

    def __post_init__(self) -> None:
        for values, label, dtype in (
            (self.filtered_height_m, "filtered_height_m", np.float64),
            (self.filtered_center_steps, "filtered_center_steps", np.int64),
            (self.vertical_speed_m_s, "vertical_speed_m_s", np.float64),
            (
                self.vertical_acceleration_m_s2,
                "vertical_acceleration_m_s2",
                np.float64,
            ),
            (self.vertical_jerk_m_s3, "vertical_jerk_m_s3", np.float64),
        ):
            array = np.asarray(values, dtype=dtype)
            if array.ndim != 1 or (dtype is np.float64 and not np.isfinite(array).all()):
                raise ValueError(f"{label} must be a finite one-dimensional array")
            object.__setattr__(self, label, _readonly(array, dtype=dtype))
        if self.filtered_height_m.shape != self.filtered_center_steps.shape:
            raise ValueError(
                "filtered_height_m and filtered_center_steps must have equal shape"
            )


def _empty_smooth_motion_evaluation(
    position: NDArray[np.float64],
    quaternion: NDArray[np.float64],
    gravity: NDArray[np.float64],
    thresholds: MotionSmoothnessThresholds,
    timestep_s: float,
) -> SmoothMotionEvaluation:
    """Return finite metrics and failing checks for a missing operation."""

    relative = gravity_relative_motion(
        position,
        quaternion,
        gravity,
        baseline_step=0,
    )
    metrics: dict[str, float | int] = {
        "motion_filter_window_steps": centered_window_steps(
            timestep_s, thresholds.filter_window_s
        ),
        "motion_filtered_sample_count": 0,
        "operation_cumulative_height_backtrack_m": 0.0,
        "operation_downward_speed_duty": 0.0,
        "operation_peak_filtered_upward_speed_m_s": 0.0,
        "operation_peak_abs_filtered_acceleration_m_s2": 0.0,
        "operation_peak_abs_filtered_jerk_m_s3": 0.0,
        "operation_hold_entry_linear_speed_m_s": 0.0,
        "operation_max_lateral_displacement_m": 0.0,
        "operation_max_orientation_drift_deg": 0.0,
    }
    checks = {
        "smooth_motion_event_sequence_valid": False,
        "smooth_motion_filter_window_available": False,
        "smooth_motion_cumulative_backtrack_within_limit": False,
        "smooth_motion_downward_speed_duty_within_limit": False,
        "smooth_motion_peak_upward_speed_within_limit": False,
        "smooth_motion_acceleration_within_limit": False,
        "smooth_motion_jerk_within_limit": False,
        "smooth_motion_hold_entry_speed_within_limit": False,
        "smooth_motion_lateral_displacement_within_limit": False,
        "smooth_motion_orientation_drift_within_limit": False,
    }
    empty_float = np.empty(0, dtype=np.float64)
    return SmoothMotionEvaluation(
        metrics=metrics,
        checks=checks,
        relative_motion=relative,
        filtered_height_m=empty_float,
        filtered_center_steps=np.empty(0, dtype=np.int64),
        vertical_speed_m_s=empty_float,
        vertical_acceleration_m_s2=empty_float,
        vertical_jerk_m_s3=empty_float,
    )


def evaluate_smooth_motion(
    cube_position_world_m: ArrayLike,
    cube_quaternion: ArrayLike,
    cube_linear_velocity_world_m_s: ArrayLike,
    gravity_world_m_s2: ArrayLike,
    *,
    timestep_s: float,
    baseline_step: int,
    manipulation_start_step: int,
    manipulation_end_step: int,
    hold_start_step: int,
    thresholds: MotionSmoothnessThresholds | Mapping[str, Any] = (
        MotionSmoothnessThresholds()
    ),
    evaluation_end_step: int | None = None,
) -> SmoothMotionEvaluation:
    """Evaluate a complete manipulation/hold path against v8 thresholds.

    Vertical derivatives and backtracking are computed over MANIPULATE only.
    Lateral displacement and orientation drift extend through the requested
    evaluation end (the full trace by default), and remain relative to the
    verified-grasp baseline.
    """

    if not isinstance(thresholds, MotionSmoothnessThresholds):
        thresholds = MotionSmoothnessThresholds.from_mapping(thresholds)
    timestep = float(timestep_s)
    centered_window_steps(timestep, thresholds.filter_window_s)
    position = np.asarray(cube_position_world_m, dtype=np.float64)
    quaternion = np.asarray(cube_quaternion, dtype=np.float64)
    velocity = np.asarray(cube_linear_velocity_world_m_s, dtype=np.float64)
    gravity = np.asarray(gravity_world_m_s2, dtype=np.float64)
    if (
        position.ndim != 2
        or position.shape[1] != 3
        or not np.isfinite(position).all()
        or position.shape[0] == 0
    ):
        raise ValueError("cube_position_world_m must have shape (T, 3) and be finite")
    if quaternion.shape != (position.shape[0], 4):
        raise ValueError("cube_quaternion must have shape (T, 4)")
    _unit_quaternion_trace(quaternion)
    if velocity.shape != position.shape or not np.isfinite(velocity).all():
        raise ValueError(
            "cube_linear_velocity_world_m_s must have shape (T, 3) and be finite"
        )
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("gravity_world_m_s2 must contain three finite values")

    total_steps = position.shape[0]
    baseline = int(baseline_step)
    start = int(manipulation_start_step)
    end = int(manipulation_end_step)
    hold = int(hold_start_step)
    evaluation_end = (
        total_steps - 1 if evaluation_end_step is None else int(evaluation_end_step)
    )
    event_sequence_valid = bool(
        not any(
            isinstance(value, bool)
            for value in (
                baseline_step,
                manipulation_start_step,
                manipulation_end_step,
                hold_start_step,
            )
        )
        and 0 <= baseline < start <= end < hold <= evaluation_end < total_steps
        and hold == end + 1
    )
    if not event_sequence_valid:
        return _empty_smooth_motion_evaluation(
            position, quaternion, gravity, thresholds, timestep
        )

    relative = gravity_relative_motion(
        position,
        quaternion,
        gravity,
        baseline_step=baseline,
    )
    operation_height = relative.height_m[start : end + 1]
    filtered_height = centered_valid_moving_average(
        operation_height,
        timestep_s=timestep,
        window_s=thresholds.filter_window_s,
    )
    local_centres = centered_valid_indices(
        operation_height.shape[0],
        centered_window_steps(timestep, thresholds.filter_window_s),
    )
    centre_steps = local_centres + start
    speed = np.diff(filtered_height) / timestep
    acceleration = np.diff(speed) / timestep
    jerk = np.diff(acceleration) / timestep
    filter_available = bool(filtered_height.size >= 4 and jerk.size > 0)

    height_delta = np.diff(filtered_height)
    backtrack = float(np.sum(np.maximum(0.0, -height_delta)))
    downward_duty = (
        float(np.mean(speed < thresholds.downward_speed_threshold_m_s))
        if speed.size
        else 0.0
    )
    peak_upward_speed = float(np.max(speed, initial=0.0))
    peak_acceleration = float(np.max(np.abs(acceleration), initial=0.0))
    peak_jerk = float(np.max(np.abs(jerk), initial=0.0))
    hold_entry_speed = float(np.linalg.norm(velocity[hold]))
    path_slice = slice(start, evaluation_end + 1)
    maximum_lateral = float(np.max(relative.lateral_distance_m[path_slice]))
    maximum_orientation = float(
        np.max(relative.orientation_drift_deg[path_slice])
    )

    metrics: dict[str, float | int] = {
        "motion_filter_window_steps": centered_window_steps(
            timestep, thresholds.filter_window_s
        ),
        "motion_filtered_sample_count": int(filtered_height.size),
        "operation_cumulative_height_backtrack_m": backtrack,
        "operation_downward_speed_duty": downward_duty,
        "operation_peak_filtered_upward_speed_m_s": peak_upward_speed,
        "operation_peak_abs_filtered_acceleration_m_s2": peak_acceleration,
        "operation_peak_abs_filtered_jerk_m_s3": peak_jerk,
        "operation_hold_entry_linear_speed_m_s": hold_entry_speed,
        "operation_max_lateral_displacement_m": maximum_lateral,
        "operation_max_orientation_drift_deg": maximum_orientation,
    }
    tolerance = 1e-12
    checks = {
        "smooth_motion_event_sequence_valid": event_sequence_valid,
        "smooth_motion_filter_window_available": filter_available,
        "smooth_motion_cumulative_backtrack_within_limit": bool(
            filter_available
            and backtrack
            <= thresholds.max_cumulative_height_backtrack_m + tolerance
        ),
        "smooth_motion_downward_speed_duty_within_limit": bool(
            filter_available
            and downward_duty
            <= thresholds.max_downward_speed_duty + tolerance
        ),
        "smooth_motion_peak_upward_speed_within_limit": bool(
            filter_available
            and peak_upward_speed
            <= thresholds.max_peak_upward_speed_m_s + tolerance
        ),
        "smooth_motion_acceleration_within_limit": bool(
            filter_available
            and peak_acceleration
            <= thresholds.max_abs_vertical_acceleration_m_s2 + tolerance
        ),
        "smooth_motion_jerk_within_limit": bool(
            filter_available
            and peak_jerk <= thresholds.max_abs_vertical_jerk_m_s3 + tolerance
        ),
        "smooth_motion_hold_entry_speed_within_limit": bool(
            hold_entry_speed
            <= thresholds.max_hold_entry_linear_speed_m_s + tolerance
        ),
        "smooth_motion_lateral_displacement_within_limit": bool(
            maximum_lateral
            <= thresholds.max_lateral_displacement_m + tolerance
        ),
        "smooth_motion_orientation_drift_within_limit": bool(
            maximum_orientation
            <= thresholds.max_orientation_drift_deg + tolerance
        ),
    }
    return SmoothMotionEvaluation(
        metrics=metrics,
        checks=checks,
        relative_motion=relative,
        filtered_height_m=filtered_height,
        filtered_center_steps=centre_steps,
        vertical_speed_m_s=speed,
        vertical_acceleration_m_s2=acceleration,
        vertical_jerk_m_s3=jerk,
    )


__all__ = [
    "GravityRelativeMotion",
    "MotionSmoothnessThresholds",
    "SmoothMotionEvaluation",
    "centered_valid_indices",
    "centered_valid_moving_average",
    "centered_window_steps",
    "evaluate_smooth_motion",
    "gravity_relative_motion",
]
