from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS
from xhand_grasp.grasp_pose import (
    ActualGraspPoseThresholds,
    continuous_window_steps,
    controller_id,
    evaluate_actual_grasp_pose,
    evaluate_actual_grasp_pose_trace,
    grasp_pose_id,
)


THUMB = "left_hand_thumb_bend_joint_actuator"


def _mapping(start: float = 0.2) -> dict[str, float]:
    return {
        name: start + 0.1 * index for index, name in enumerate(ACTIVE_ACTUATORS)
    }


def _nominal() -> dict[str, float]:
    values = _mapping()
    values[THUMB] = 1.50
    return values


def _config() -> dict:
    return {
        "schema_version": 9,
        "cube": {
            "edge_m": 0.064,
            "mass_kg": 0.160,
            "friction": [0.8, 0.005, 0.0001],
            "center_xy_m": [0.071, -0.027],
            "rpy_deg": [0.0, 0.0, 27.6099901894],
        },
        "scene": {"support_top_z_m": 0.084},
        "hand_pose": {
            "translation_m": [0.02, -0.01, 0.25],
            "rpy_deg": [0.0, 121.0, 5.0],
        },
        "contact_topology": {
            "target_faces": {"thumb": "-X", "index": "+X", "mid": "+X"}
        },
        "grasp_pose": {
            "nominal_joint_qpos_rad": _nominal(),
            "thumb_actual_range_rad": [1.40, 1.60],
            "max_nominal_joint_error_rad": 0.04,
            "max_joint_stability_span_rad": 0.03,
            "verify_continuous_s": 0.25,
        },
        "control": {
            "precontact_targets_rad": _mapping(0.1),
            # Deliberately not equal to the actual pose: it is a servo command.
            "contact_preload_targets_rad": _mapping(0.3),
            "manipulation_delta_rad": {name: 0.0 for name in ACTIVE_ACTUATORS},
            "close_profile": {
                name: {"start_fraction": 0.0, "end_fraction": 1.0}
                for name in ACTIVE_ACTUATORS
            },
        },
        "control_protocol": {
            "strategy": "grasp_verify_then_manipulate",
            "close_s": 1.25,
            "stable_window_s": 0.25,
            "manipulate_s": 2.0,
            "manipulation_profile": "minimum_jerk_quintic",
        },
    }


def _actual_trace(samples: int = 320) -> np.ndarray:
    nominal = np.asarray([_nominal()[name] for name in ACTIVE_ACTUATORS])
    phase = np.linspace(0.0, 4.0 * np.pi, samples)
    return nominal[np.newaxis, :] + 0.002 * np.sin(phase)[:, np.newaxis]


def test_grasp_pose_and_controller_id_keep_actual_pose_and_commands_separate():
    baseline = _config()
    nominal_change = copy.deepcopy(baseline)
    nominal_change["grasp_pose"]["nominal_joint_qpos_rad"][THUMB] += 0.01
    command_change = copy.deepcopy(baseline)
    command_change["control"]["contact_preload_targets_rad"][THUMB] += 0.01

    assert grasp_pose_id(nominal_change) != grasp_pose_id(baseline)
    assert controller_id(nominal_change) == controller_id(baseline)
    assert grasp_pose_id(command_change) == grasp_pose_id(baseline)
    assert controller_id(command_change) != controller_id(baseline)

    reordered = copy.deepcopy(baseline)
    reordered["grasp_pose"]["nominal_joint_qpos_rad"] = dict(
        reversed(
            tuple(reordered["grasp_pose"]["nominal_joint_qpos_rad"].items())
        )
    )
    assert grasp_pose_id(reordered) == grasp_pose_id(baseline)


def test_grasp_pose_id_uses_geometry_and_pose_but_excludes_cube_physics():
    baseline = _config()
    physics_change = copy.deepcopy(baseline)
    physics_change["cube"]["mass_kg"] = 0.176
    physics_change["cube"]["friction"] = [0.7, 0.004, 0.0002]
    physics_change["cube"]["solref"] = [0.015, 1.2]
    edge_change = copy.deepcopy(baseline)
    edge_change["cube"]["edge_m"] = 0.065
    support_change = copy.deepcopy(baseline)
    support_change["scene"]["support_top_z_m"] += 0.001

    assert grasp_pose_id(physics_change) == grasp_pose_id(baseline)
    assert grasp_pose_id(edge_change) != grasp_pose_id(baseline)
    assert grasp_pose_id(support_change) != grasp_pose_id(baseline)


def test_earliest_250_ms_actual_pose_window_has_inclusive_event_frames():
    actual = _actual_trace()
    gate = np.zeros(actual.shape[0], dtype=bool)
    gate[20:] = True
    contact = np.zeros_like(gate)
    contact[5:] = True
    preload = _mapping(0.4)
    preload[THUMB] = 1.20

    result = evaluate_actual_grasp_pose(
        actual,
        _nominal(),
        gate,
        contact,
        timestep_s=0.001,
        contact_preload_command_rad=np.asarray(
            [preload[name] for name in ACTIVE_ACTUATORS]
        ),
    )

    assert result.passed
    assert result.events.first_contact_step == 5
    assert result.events.stable_window_start_step == 20
    assert result.events.stable_window_end_step == 269
    assert result.events.grasp_lock_step == 269
    assert result.events.required_continuous_steps == 250
    assert result.checks == {
        "grasp_pose_base_gate_contiguous": True,
        "thumb_actual_qpos_within_range": True,
        "actual_joint_median_matches_nominal": True,
        "actual_joint_window_stable": True,
        "actual_grasp_pose_locked": True,
    }
    assert result.metrics["preload_command_used_as_acceptance_evidence"] is False
    assert result.contact_preload_command_rad[0] == pytest.approx(1.20)
    np.testing.assert_allclose(
        result.as_trace_fields()["grasp_pose_actual_qpos_rad"],
        result.actual_qpos_median_rad,
    )


def test_command_at_1p5_cannot_authorize_actual_thumb_at_1p22():
    actual = _actual_trace()
    actual[:, 0] = 1.22
    preload = np.asarray([_nominal()[name] for name in ACTIVE_ACTUATORS])
    result = evaluate_actual_grasp_pose(
        actual,
        _nominal(),
        np.ones(actual.shape[0], dtype=bool),
        np.ones(actual.shape[0], dtype=bool),
        timestep_s=0.001,
        contact_preload_command_rad=preload,
    )

    assert preload[0] == pytest.approx(1.50)
    assert not result.checks["thumb_actual_qpos_within_range"]
    assert not result.passed
    assert result.events.grasp_lock_step == -1
    assert result.metrics["thumb_actual_max_rad"] == pytest.approx(1.22)


def test_actual_pose_can_pass_with_a_different_preload_command():
    actual = _actual_trace()
    preload = np.asarray([_nominal()[name] for name in ACTIVE_ACTUATORS])
    preload[0] = 1.70
    result = evaluate_actual_grasp_pose(
        actual,
        _nominal(),
        np.ones(actual.shape[0], dtype=bool),
        np.ones(actual.shape[0], dtype=bool),
        timestep_s=0.001,
        contact_preload_command_rad=preload,
    )

    assert result.passed
    assert result.actual_qpos_median_rad[0] == pytest.approx(1.50, abs=2e-3)
    assert result.contact_preload_command_rad[0] == pytest.approx(1.70)


@pytest.mark.parametrize("violation", ["median", "span"])
def test_joint_pose_error_and_stability_span_are_independent_hard_checks(violation):
    actual = _actual_trace()
    if violation == "median":
        actual[:, 4] += 0.041
    else:
        actual[::2, 4] += 0.020
        actual[1::2, 4] -= 0.020
    result = evaluate_actual_grasp_pose(
        actual,
        _nominal(),
        np.ones(actual.shape[0], dtype=bool),
        np.ones(actual.shape[0], dtype=bool),
        timestep_s=0.001,
    )

    assert not result.passed
    assert result.events.stable_window_start_step == -1
    if violation == "median":
        assert not result.checks["actual_joint_median_matches_nominal"]
        assert result.checks["actual_joint_window_stable"]
    else:
        assert result.checks["actual_joint_median_matches_nominal"]
        assert not result.checks["actual_joint_window_stable"]


def test_gate_counter_resets_and_lock_is_on_next_complete_run():
    samples = 500
    actual = _actual_trace(samples)
    gate = np.ones(samples, dtype=bool)
    gate[200] = False
    result = evaluate_actual_grasp_pose(
        actual,
        _nominal(),
        gate,
        np.ones(samples, dtype=bool),
        timestep_s=0.001,
    )

    assert result.events.stable_window_start_step == 201
    assert result.events.grasp_lock_step == 450


def test_short_gate_fragment_is_reported_but_never_locks():
    actual = _actual_trace(200)
    result = evaluate_actual_grasp_pose(
        actual,
        _nominal(),
        np.ones(200, dtype=bool),
        np.ones(200, dtype=bool),
        timestep_s=0.001,
    )

    assert not result.passed
    assert result.metrics["diagnostic_window_sample_count"] == 200
    assert not result.checks["grasp_pose_base_gate_contiguous"]
    assert result.events.grasp_lock_step == -1


class _FakeModel:
    def __init__(self):
        self.opt = SimpleNamespace(timestep=0.001)
        self._ids = {name: index + 2 for index, name in enumerate(ACTIVE_ACTUATORS)}

    def actuator(self, name: str):
        return SimpleNamespace(id=self._ids[name])


def test_model_config_trace_adapter_selects_actual_joint_columns():
    model = _FakeModel()
    actual = _actual_trace()
    full_qpos = np.zeros((actual.shape[0], 12), dtype=np.float64)
    full_qpos[:, 2:10] = actual
    physical_force = np.zeros((actual.shape[0], 3), dtype=np.float64)
    physical_force[5:, 1] = 1e-3
    effective = np.zeros((actual.shape[0], 3), dtype=bool)
    effective[20:] = True
    traces = {
        "joint_qpos": full_qpos,
        "grasp_gate": np.ones((actual.shape[0], 4), dtype=bool),
        "finger_contact_force": physical_force,
        "target_face_effective": effective,
    }

    result = evaluate_actual_grasp_pose_trace(model, _config(), traces)

    assert result.passed
    assert result.events.first_contact_step == 5
    np.testing.assert_allclose(
        result.actual_qpos_median_rad,
        np.asarray([_nominal()[name] for name in ACTIVE_ACTUATORS]),
        atol=2e-3,
    )


def test_threshold_config_falls_back_to_protocol_window_and_validates_inputs():
    config = _config()
    del config["grasp_pose"]["verify_continuous_s"]
    settings = ActualGraspPoseThresholds.from_config(config)

    assert settings.verify_continuous_s == pytest.approx(0.25)
    assert continuous_window_steps(0.001, 0.25) == 250
    with pytest.raises(ValueError, match="positive and finite"):
        continuous_window_steps(0.0, 0.25)
    with pytest.raises(ValueError, match=r"shape \(T, 8\)"):
        evaluate_actual_grasp_pose(
            np.zeros((10, 7)),
            _nominal(),
            np.ones(10, dtype=bool),
            np.ones(10, dtype=bool),
            timestep_s=0.001,
        )
