from __future__ import annotations

import math

import numpy as np
import pytest

from xhand_grasp.motion_smoothness import (
    MotionSmoothnessThresholds,
    centered_valid_indices,
    centered_valid_moving_average,
    centered_window_steps,
    evaluate_smooth_motion,
    gravity_relative_motion,
)


def test_centered_51_ms_average_uses_only_complete_windows():
    values = np.arange(1.0, 56.0)
    averaged = centered_valid_moving_average(
        values,
        timestep_s=0.001,
        window_s=0.051,
    )

    assert centered_window_steps(0.001, 0.051) == 51
    np.testing.assert_array_equal(centered_valid_indices(55, 51), [25, 26, 27, 28, 29])
    np.testing.assert_allclose(averaged, [26.0, 27.0, 28.0, 29.0, 30.0])


def test_centered_average_supports_vector_traces_and_short_inputs():
    values = np.column_stack((np.arange(7.0), 2.0 * np.arange(7.0)))
    averaged = centered_valid_moving_average(
        values,
        timestep_s=0.001,
        window_s=0.003,
    )
    empty = centered_valid_moving_average(
        values[:2],
        timestep_s=0.001,
        window_s=0.003,
    )

    np.testing.assert_allclose(
        averaged,
        np.column_stack((np.arange(1.0, 6.0), 2.0 * np.arange(1.0, 6.0))),
    )
    assert empty.shape == (0, 2)


def test_centered_average_rejects_even_or_unaligned_windows():
    with pytest.raises(ValueError, match="odd width"):
        centered_window_steps(0.001, 0.050)
    with pytest.raises(ValueError, match="does not align"):
        centered_window_steps(0.003, 0.0515)


def _quaternion_z_deg(angle_deg: float) -> np.ndarray:
    half = math.radians(angle_deg) / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


def test_gravity_relative_motion_supports_nonvertical_gravity_and_rotations():
    position = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ]
    )
    quaternion = np.vstack((_quaternion_z_deg(0.0), _quaternion_z_deg(90.0)))
    relative = gravity_relative_motion(
        position,
        quaternion,
        [0.0, -9.81, 0.0],
        baseline_step=0,
    )

    np.testing.assert_allclose(relative.up_world, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(relative.height_m, [0.0, 2.0])
    np.testing.assert_allclose(relative.lateral_displacement_world_m[1], [1.0, 0.0, 3.0])
    np.testing.assert_allclose(relative.lateral_distance_m, [0.0, math.sqrt(10.0)])
    np.testing.assert_allclose(relative.orientation_drift_deg, [0.0, 90.0])


def _vertical_trace(
    *,
    operation_steps: int = 2001,
    hold_steps: int = 100,
    lift_m: float = 0.010,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    # Sample 0 is the verified-grasp baseline.  MANIPULATE starts at sample 1,
    # ends after exactly two seconds, and HOLD begins on the following sample.
    total = 1 + operation_steps + hold_steps
    position = np.zeros((total, 3), dtype=np.float64)
    position[1 : 1 + operation_steps, 2] = np.linspace(
        0.0, lift_m, operation_steps
    )
    position[1 + operation_steps :, 2] = lift_m
    quaternion = np.tile(_quaternion_z_deg(0.0), (total, 1))
    velocity = np.zeros((total, 3), dtype=np.float64)
    velocity[1 : 1 + operation_steps, 2] = lift_m / (operation_steps - 1) / 0.001
    events = {
        "baseline_step": 0,
        "manipulation_start_step": 1,
        "manipulation_end_step": operation_steps,
        "hold_start_step": operation_steps + 1,
    }
    return position, quaternion, velocity, events


def test_constant_slow_vertical_path_passes_all_smooth_motion_checks():
    position, quaternion, velocity, events = _vertical_trace()
    result = evaluate_smooth_motion(
        position,
        quaternion,
        velocity,
        [0.0, 0.0, -9.81],
        timestep_s=0.001,
        **events,
    )

    assert all(result.checks.values())
    assert result.metrics["motion_filter_window_steps"] == 51
    assert result.metrics["operation_cumulative_height_backtrack_m"] == pytest.approx(0.0)
    assert result.metrics["operation_downward_speed_duty"] == pytest.approx(0.0)
    assert result.metrics["operation_peak_filtered_upward_speed_m_s"] == pytest.approx(
        0.005, abs=1e-10
    )
    assert result.metrics["operation_max_lateral_displacement_m"] == 0.0
    assert result.metrics["operation_max_orientation_drift_deg"] == 0.0
    assert result.filtered_height_m.shape == result.filtered_center_steps.shape
    assert result.vertical_speed_m_s.shape[0] == result.filtered_height_m.shape[0] - 1


def test_backtracking_and_downward_motion_are_detected_after_filtering():
    position, quaternion, velocity, events = _vertical_trace(lift_m=0.012)
    start = events["manipulation_start_step"]
    end = events["manipulation_end_step"]
    midpoint = (start + end) // 2
    # Rise to 8 mm, retreat to 6 mm over 300 ms, then finish at 12 mm.  The
    # long segments survive the centred filter without edge-padding artefacts.
    position[start:midpoint, 2] = np.linspace(0.0, 0.008, midpoint - start)
    retreat_end = midpoint + 300
    position[midpoint:retreat_end, 2] = np.linspace(0.008, 0.006, 300)
    position[retreat_end : end + 1, 2] = np.linspace(
        0.006, 0.012, end - retreat_end + 1
    )
    position[events["hold_start_step"] :, 2] = 0.012

    result = evaluate_smooth_motion(
        position,
        quaternion,
        velocity,
        [0.0, 0.0, -9.81],
        timestep_s=0.001,
        **events,
    )

    assert result.metrics["operation_cumulative_height_backtrack_m"] > 0.0015
    assert result.metrics["operation_downward_speed_duty"] > 0.02
    assert not result.checks["smooth_motion_cumulative_backtrack_within_limit"]
    assert not result.checks["smooth_motion_downward_speed_duty_within_limit"]


def test_lateral_orientation_and_hold_entry_limits_cover_operation_and_hold():
    position, quaternion, velocity, events = _vertical_trace()
    # Apply these violations only in HOLD to prove the path scope extends past
    # MANIPULATE while derivative metrics remain manipulation-only.
    hold = events["hold_start_step"]
    position[hold:, 0] = 0.0021
    quaternion[hold:] = _quaternion_z_deg(10.1)
    velocity[hold] = [0.0, 0.0, 0.0051]

    result = evaluate_smooth_motion(
        position,
        quaternion,
        velocity,
        [0.0, 0.0, -9.81],
        timestep_s=0.001,
        **events,
    )

    assert not result.checks["smooth_motion_lateral_displacement_within_limit"]
    assert not result.checks["smooth_motion_orientation_drift_within_limit"]
    assert not result.checks["smooth_motion_hold_entry_speed_within_limit"]
    assert result.metrics["operation_max_lateral_displacement_m"] == pytest.approx(0.0021)
    assert result.metrics["operation_max_orientation_drift_deg"] == pytest.approx(10.1)
    assert result.metrics["operation_hold_entry_linear_speed_m_s"] == pytest.approx(0.0051)


def test_missing_manipulation_returns_finite_metrics_and_explicit_failures():
    position, quaternion, velocity, _ = _vertical_trace(operation_steps=101)
    result = evaluate_smooth_motion(
        position,
        quaternion,
        velocity,
        [0.0, 0.0, -9.81],
        timestep_s=0.001,
        baseline_step=-1,
        manipulation_start_step=-1,
        manipulation_end_step=-1,
        hold_start_step=-1,
    )

    assert not any(result.checks.values())
    assert all(np.isfinite(float(value)) for value in result.metrics.values())
    assert result.filtered_height_m.size == 0


def test_threshold_mapping_requires_the_exact_versioned_field_set():
    settings = MotionSmoothnessThresholds()
    assert MotionSmoothnessThresholds.from_mapping(settings.as_config()) == settings
    incomplete = settings.as_config()
    incomplete.pop("max_abs_vertical_jerk_m_s3")
    with pytest.raises(ValueError, match="fields do not match"):
        MotionSmoothnessThresholds.from_mapping(incomplete)
