from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, load_config
from xhand_grasp.controller import V6_GRASP_GATE_ORDER
from xhand_grasp.evaluation import _v6_pose_preservation_metrics
from xhand_grasp.scene import build_model
from xhand_grasp.simulation import _allocate_traces


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_pose_preserving_grasp.json"
)


@pytest.fixture(scope="module")
def v6_model_info_config():
    config = load_config(CONFIG)
    model, info = build_model(config)
    # Short protocol durations make the synthetic CLOSE trace easy to audit.
    config = copy.deepcopy(config)
    config["control_protocol"].update(
        {
            "settle_s": 0.002,
            "close_s": 0.004,
            "verify_timeout_s": 0.002,
            "stable_window_s": 0.001,
            "manipulate_s": 0.001,
            "min_hold_s": 0.001,
        }
    )
    return model, info, config


def _z_quaternion(degrees: float) -> np.ndarray:
    half = math.radians(degrees) / 2.0
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)])


def _smoothstep(value: float) -> float:
    clipped = min(1.0, max(0.0, value))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _sync_persisted_v6_fields(model, info, config, traces) -> None:
    """Populate persisted helpers from raw arrays without production helpers."""

    total = len(traces["time"])
    initial_position = traces["initial_cube_pos_m"]
    initial_quaternion = traces["initial_cube_quat"]
    positions = traces["cube_pos"]
    quaternions = traces["cube_quat"]
    translation = np.linalg.norm(positions - initial_position, axis=1)
    dot = np.abs(quaternions @ initial_quaternion)
    orientation = np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
    translation_ok = (
        translation
        <= float(config["pose_preservation"]["max_translation_m"]) + 1e-12
    )
    orientation_ok = (
        orientation
        <= float(config["pose_preservation"]["max_orientation_drift_deg"])
        + 1e-12
    )

    acquisition = int(traces["grasp_acquisition_step"])
    scope_last = acquisition if acquisition >= 0 else total - 1
    translation_latch = np.logical_and.accumulate(translation_ok)
    orientation_latch = np.logical_and.accumulate(orientation_ok)
    if scope_last + 1 < total:
        translation_latch[scope_last + 1 :] = translation_latch[scope_last]
        orientation_latch[scope_last + 1 :] = orientation_latch[scope_last]

    support_latch = np.logical_and.accumulate(traces["support_contact"])
    if scope_last + 1 < total:
        support_latch[scope_last + 1 :] = support_latch[scope_last]

    hand_contact = (
        traces["forbidden_contact"]
        | (np.sum(traces["finger_contact_force"], axis=1) > 0.0)
        | (np.sum(traces["active_nondistal_force_n"], axis=1) > 0.0)
    )
    traces["hand_cube_contact"][:] = hand_contact
    settle_latch = np.ones(total, dtype=bool)
    running = True
    for step, state in enumerate(traces["control_state"]):
        if step <= scope_last and state == "SETTLE":
            running = running and not bool(hand_contact[step])
        settle_latch[step] = running

    traces["cube_translation_from_initial_m"][:] = translation
    traces["cube_orientation_from_initial_deg"][:] = orientation
    traces["initial_pose_translation_history_stable"][:] = translation_latch
    traces["initial_pose_orientation_history_stable"][:] = orientation_latch
    traces["pregrasp_pose_within_limit"][:] = translation_ok & orientation_ok
    traces["pregrasp_support_retained_latched"][:] = support_latch
    traces["settle_hand_contact_free_latched"][:] = settle_latch
    combined = translation_latch & orientation_latch & support_latch & settle_latch
    traces["pregrasp_pose_preserved_latched"][:] = combined
    history_index = tuple(traces["grasp_gate_order"]).index(
        "initial_pose_history_stable"
    )
    traces["grasp_gate"][:, history_index] = combined

    traces["close_progress"][:] = 0.0
    settle_steps = 2
    close_steps = 4
    for step, state in enumerate(traces["control_state"]):
        if state == "CLOSE":
            fraction = (step - settle_steps + 1) / close_steps
            for name, profile in config["control"]["close_profile"].items():
                actuator_id = model.actuator(name).id
                normalized = (
                    fraction - float(profile["start_fraction"])
                ) / (
                    float(profile["end_fraction"])
                    - float(profile["start_fraction"])
                )
                traces["close_progress"][step, actuator_id] = _smoothstep(
                    normalized
                )
        elif state == "ABORT" and step > 0:
            traces["close_progress"][step] = traces["close_progress"][step - 1]
        elif state != "SETTLE":
            traces["close_progress"][step, info.active_actuator_ids] = 1.0

    traces["pregrasp_target_rad"][:] = 0.0
    traces["close_start_fraction"][:] = 0.0
    traces["close_end_fraction"][:] = 1.0
    for name in ACTIVE_ACTUATORS:
        actuator_id = model.actuator(name).id
        traces["pregrasp_target_rad"][actuator_id] = float(
            config["control"]["pregrasp_targets_rad"][name]
        )
        traces["close_start_fraction"][actuator_id] = float(
            config["control"]["close_profile"][name]["start_fraction"]
        )
        traces["close_end_fraction"][actuator_id] = float(
            config["control"]["close_profile"][name]["end_fraction"]
        )

    traces["first_distal_contact_step"][:] = -1
    for finger_index in range(len(ACTIVE_FINGERS)):
        contact_steps = np.flatnonzero(
            traces["finger_contact_force"][:, finger_index] > 0.0
        )
        if contact_steps.size:
            traces["first_distal_contact_step"][finger_index] = int(
                contact_steps[0]
            )


def _valid_synthetic_trace(model, info, config):
    total = 10
    traces = _allocate_traces(model, total, schema_version=6)
    traces["time"][:] = np.arange(1, total + 1) * model.opt.timestep
    initial_position = np.asarray([0.071, -0.027, 0.114])
    initial_quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    traces["initial_cube_pos_m"][:] = initial_position
    traces["initial_cube_quat"][:] = initial_quaternion
    traces["initial_joint_qpos_rad"][:] = 0.0
    for name, value in config["control"]["pregrasp_targets_rad"].items():
        traces["initial_joint_qpos_rad"][model.actuator(name).id] = float(value)
    traces["initialized_at_pregrasp"] = np.asarray(True, dtype=bool)
    traces["cube_pos"][:] = initial_position
    traces["cube_quat"][:] = initial_quaternion
    traces["support_contact"][:] = True
    traces["forbidden_contact"][:] = False
    traces["finger_contact_force"][:] = 0.0
    traces["active_nondistal_force_n"][:] = 0.0
    traces["hand_cube_contact"][:] = False
    traces["control_state"][:] = np.asarray(
        [
            "SETTLE",
            "SETTLE",
            "CLOSE",
            "CLOSE",
            "CLOSE",
            "CLOSE",
            "VERIFY",
            "VERIFY",
            "MANIPULATE",
            "HOLD",
        ]
    )
    traces["grasp_acquisition_step"] = np.asarray(7, dtype=np.int64)
    traces["grasp_gate_order"] = np.asarray(V6_GRASP_GATE_ORDER)
    traces["grasp_gate"][:] = True
    traces["finger_contact_force"][6:, :] = 0.1
    _sync_persisted_v6_fields(model, info, config, traces)
    return traces


def test_history_violation_remains_failed_after_pose_returns(
    v6_model_info_config,
):
    model, info, config = v6_model_info_config
    traces = _valid_synthetic_trace(model, info, config)
    traces["cube_pos"][3, 0] += 0.0006
    _sync_persisted_v6_fields(model, info, config, traces)

    metrics, checks = _v6_pose_preservation_metrics(
        model, info, config, traces
    )

    assert traces["pregrasp_pose_within_limit"][4]
    assert not traces["pregrasp_pose_preserved_latched"][4]
    assert not checks["object_pose_preserved_until_grasp_acquisition"]
    assert checks["v6_pose_preservation_trace_matches_raw_state"]
    assert metrics["pose_preservation"]["max_translation_m"] == pytest.approx(
        0.0006
    )


def test_acquisition_frame_is_included_in_pose_contract(v6_model_info_config):
    model, info, config = v6_model_info_config
    traces = _valid_synthetic_trace(model, info, config)
    acquisition = int(traces["grasp_acquisition_step"])
    traces["cube_quat"][acquisition] = _z_quaternion(1.01)
    # Movement after acquisition is outside this contract; make it much larger
    # to ensure the failing sample above is specifically the inclusive edge.
    traces["cube_quat"][acquisition + 1 :] = _z_quaternion(20.0)
    _sync_persisted_v6_fields(model, info, config, traces)

    metrics, checks = _v6_pose_preservation_metrics(
        model, info, config, traces
    )

    assert metrics["pose_preservation"]["scope_last_step"] == acquisition
    assert metrics["pose_preservation"][
        "max_orientation_drift_deg"
    ] == pytest.approx(1.01)
    assert not checks["object_pose_preserved_until_grasp_acquisition"]


@pytest.mark.parametrize("violation", ["support", "settle_contact"])
def test_support_and_settle_contact_failures_are_sticky(
    v6_model_info_config, violation
):
    model, info, config = v6_model_info_config
    traces = _valid_synthetic_trace(model, info, config)
    if violation == "support":
        traces["support_contact"][3] = False
    else:
        traces["finger_contact_force"][0, 0] = 0.01
    _sync_persisted_v6_fields(model, info, config, traces)

    _, checks = _v6_pose_preservation_metrics(model, info, config, traces)

    if violation == "support":
        assert not checks["support_retained_until_grasp_acquisition"]
        assert not traces["pregrasp_support_retained_latched"][-1]
    else:
        assert not checks["no_hand_cube_contact_during_settle"]
        assert not traces["settle_hand_contact_free_latched"][-1]
    assert checks["v6_pose_preservation_trace_matches_raw_state"]


@pytest.mark.parametrize("corruption", ["close_progress", "first_contact"])
def test_close_progress_and_contact_onset_are_recomputed_from_raw_trace(
    v6_model_info_config, corruption
):
    model, info, config = v6_model_info_config
    traces = _valid_synthetic_trace(model, info, config)
    _, passing = _v6_pose_preservation_metrics(model, info, config, traces)
    assert passing["v6_close_profile_trace_matches_config"]
    assert passing["v6_first_distal_contact_steps_match_raw_trace"]

    if corruption == "close_progress":
        actuator_id = int(info.active_actuator_ids[0])
        traces["close_progress"][3, actuator_id] += 0.125
    else:
        traces["first_distal_contact_step"][1] += 1

    _, checks = _v6_pose_preservation_metrics(model, info, config, traces)
    expected_failed_check = (
        "v6_close_profile_trace_matches_config"
        if corruption == "close_progress"
        else "v6_first_distal_contact_steps_match_raw_trace"
    )
    assert not checks[expected_failed_check]
