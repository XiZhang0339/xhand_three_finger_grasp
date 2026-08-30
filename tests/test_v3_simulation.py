from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.artifacts import json_text, resolved_run_config, write_json
from xhand_grasp.config import load_config
from xhand_grasp.contacts import FACE_ORDER, Face
from xhand_grasp.controller import GRASP_GATE_ORDER
from xhand_grasp.evaluation import evaluate_trace
from xhand_grasp.rendering import expected_video_frame_count
from xhand_grasp.scene import build_model, rpy_degrees_to_quaternion
from xhand_grasp.simulation import _allocate_traces, run_simulation
from xhand_grasp.trajectory import _phase_steps


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_grasp_then_lift.json"
)


@pytest.mark.parametrize(
    ("total_steps", "expected"),
    [(4_750, 142), (5_000, 150), (5_250, 157)],
)
def test_video_frame_count_matches_positive_time_scheduler(total_steps, expected):
    assert expected_video_frame_count(total_steps, 0.001, 30) == expected


def _synthetic_passing_v3_trace(model, info, config):
    phase_steps = _phase_steps(model, config)
    total = sum(phase_steps.values())
    acquisition = phase_steps["settle"] + phase_steps["close"] + 250 - 1
    manipulation_start = acquisition + 1
    manipulation_end = manipulation_start + phase_steps["manipulate"] - 1
    hold_start = manipulation_end + 1
    traces = _allocate_traces(model, total, schema_version=3)

    traces["time"][:] = np.arange(1, total + 1) * model.opt.timestep
    traces["cube_pos"][:] = [0.0, 0.0, 0.1]
    for step in range(manipulation_start, manipulation_end + 1):
        alpha = (step - manipulation_start + 1) / phase_steps["manipulate"]
        progress = alpha * alpha * (3.0 - 2.0 * alpha)
        traces["cube_pos"][step, 2] = 0.1 + 0.012 * progress
    traces["cube_pos"][hold_start:, 2] = 0.112
    traces["cube_quat"][:] = [1.0, 0.0, 0.0, 0.0]
    traces["cube_velocity"][:] = 0.0
    traces["root_pos"][:] = 0.0
    traces["root_quat"][:] = [1.0, 0.0, 0.0, 0.0]
    traces["ctrl"][:] = 0.0
    traces["joint_qpos"][:] = 0.0
    traces["joint_qvel"][:] = 0.0
    traces["actuator_force"][:] = 0.0
    traces["finger_contact_force"][:] = 0.1
    traces["tactile_max"][:] = 0.1
    traces["forbidden_contact"][:] = False
    traces["support_contact"][:] = False
    traces["support_contact"][:manipulation_start] = True
    traces["floor_contact"][:] = False
    traces["max_penetration"][:] = 0.001
    traces["friction_error"][:] = 0.0
    traces["cube_contact_seen"][:] = True
    traces["contact_dim_ok"][:] = True
    traces["finite"][:] = True
    traces["palm_down_angle_deg"][:] = 0.0
    traces["distal_face_force_n"][:] = 0.0
    traces["distal_face_force_n"][:, 0, Face.X_NEG] = 0.1
    traces["distal_face_force_n"][:, 1, Face.X_POS] = 0.1
    traces["distal_face_force_n"][:, 2, Face.X_POS] = 0.1
    traces["active_nondistal_force_n"][:] = 0.0
    traces["target_face_force_purity"][:] = 1.0
    traces["target_face_topology"][:] = True
    traces["target_face_effective"][:] = True

    traces["control_state"][: phase_steps["settle"]] = "SETTLE"
    close_end = phase_steps["settle"] + phase_steps["close"]
    traces["control_state"][phase_steps["settle"] : close_end] = "CLOSE"
    traces["control_state"][close_end : acquisition + 1] = "VERIFY"
    traces["control_state"][manipulation_start : manipulation_end + 1] = (
        "MANIPULATE"
    )
    traces["control_state"][hold_start:] = "HOLD"
    traces["grasp_gate"][:] = True
    support_gate = GRASP_GATE_ORDER.index("support_contact")
    traces["grasp_gate"][manipulation_start:, support_gate] = False
    traces["grasp_gate_consecutive_steps"][:] = 0
    traces["grasp_gate_consecutive_steps"][close_end : acquisition + 1] = np.arange(
        1, 251
    )
    traces["grasp_gate_consecutive_steps"][acquisition + 1 :] = 250
    traces["grasp_acquired"][:] = False
    traces["grasp_acquired"][acquisition:] = True
    traces["manipulation_progress"][:] = 0.0
    for step in range(manipulation_start, manipulation_end + 1):
        alpha = (step - manipulation_start + 1) / phase_steps["manipulate"]
        traces["manipulation_progress"][step] = alpha * alpha * (3.0 - 2.0 * alpha)
    traces["manipulation_progress"][hold_start:] = 1.0
    traces["grasp_acquisition_step"] = np.asarray(acquisition, dtype=np.int64)
    traces["manipulation_start_step"] = np.asarray(
        manipulation_start, dtype=np.int64
    )
    traces["manipulation_end_step"] = np.asarray(manipulation_end, dtype=np.int64)
    traces["termination_step"] = np.asarray(total - 1, dtype=np.int64)
    traces["video_frame_steps"] = np.asarray([], dtype=np.int64)
    return phase_steps, traces, hold_start


def test_v3_real_trace_aborts_before_manipulation_and_recomputes(tmp_path):
    config = load_config(CONFIG)
    trace_path = tmp_path / "trace.npz"
    summary = run_simulation(config, trace_path=trace_path)

    assert summary["stage_status"] == {
        "grasp_success": False,
        "grasp": "failed",
        "manipulation_success": False,
        "manipulation": "not_run",
        "full_success": False,
    }
    assert summary["metrics"]["grasp_acquisition_step"] == -1
    assert summary["metrics"]["manipulation_start_step"] == -1

    with np.load(trace_path, allow_pickle=False) as archive:
        traces = {name: archive[name] for name in archive.files}
    assert traces["time"].shape == (4750,)
    assert traces["control_state"][0] == "SETTLE"
    assert np.all(traces["control_state"][1:] == "ABORT")
    assert not np.any(traces["grasp_acquired"])
    assert np.all(traces["manipulation_progress"] == 0.0)
    assert int(traces["grasp_acquisition_step"]) == -1
    assert int(traces["manipulation_start_step"]) == -1

    model, info = build_model(config)
    recomputed = evaluate_trace(
        model, info, config, _phase_steps(model, config), traces
    )
    assert json_text(recomputed) == json_text(summary)

    resolved = resolved_run_config(config, summary)
    assert resolved["experiment_status"]["classification"] == (
        "failed_grasp_acquisition"
    )
    assert resolved["experiment_status"]["grasp_success"] is False
    assert resolved["experiment_status"]["manipulation_success"] is False


def test_v3_verify_near_miss_metrics_are_recomputed_from_raw_evidence():
    config = load_config(CONFIG)
    model, info = build_model(config)
    phase_steps, traces, _ = _synthetic_passing_v3_trace(model, info, config)
    close_end = phase_steps["settle"] + phase_steps["close"]
    acquisition = int(traces["grasp_acquisition_step"])

    # Persisted convenience arrays are deliberately wrong.  Diagnostics must
    # still be derived from the force/tactile arrays and canonical gate axis.
    traces["target_face_effective"][close_end : acquisition + 1] = False
    traces["target_face_topology"][close_end : acquisition + 1] = False
    traces["target_face_force_purity"][close_end : acquisition + 1] = 0.0

    result = evaluate_trace(model, info, config, phase_steps, traces)
    metrics = result["metrics"]
    assert not result["checks"]["target_face_evidence_matches_raw_trace"]
    assert metrics["verify_sample_count"] == 250
    assert metrics["verify_gate_component_duty"] == {
        name: 1.0 for name in GRASP_GATE_ORDER
    }
    assert metrics["verify_target_face_effective_duty"] == {
        "thumb": 1.0,
        "index": 1.0,
        "mid": 1.0,
    }
    assert metrics["verify_peak_target_face_force_n"] == {
        "thumb": pytest.approx(0.1),
        "index": pytest.approx(0.1),
        "mid": pytest.approx(0.1),
    }
    assert metrics["verify_peak_tactile_n"] == {
        "thumb": pytest.approx(0.1),
        "index": pytest.approx(0.1),
        "mid": pytest.approx(0.1),
    }
    assert metrics["verify_target_face_simultaneous_duty"] == 1.0
    assert metrics["verify_all_gate_duty"] == 1.0
    assert metrics["verify_max_consecutive_gate_steps"] == 250
    assert metrics["verify_max_consecutive_all_gate_steps"] == 250
    assert metrics["verify_effective_finger_count"] == 3
    assert metrics["verify_max_simultaneous_effective_finger_count"] == 3


def test_fixed_non_axis_aligned_root_quaternion_is_not_roundoff_motion():
    config = load_config(CONFIG)
    model, info = build_model(config)
    phase_steps, traces, _ = _synthetic_passing_v3_trace(model, info, config)
    quaternion = rpy_degrees_to_quaternion([2.175827428, 89.659650414, -2.716760923])
    traces["root_quat"][:] = quaternion

    result = evaluate_trace(model, info, config, phase_steps, traces)

    assert result["metrics"]["root_orientation_drift_rad"] == 0.0
    assert result["checks"]["hand_root_pose_did_not_move"]


def test_v3_uses_entire_early_extended_hold_for_all_hold_checks():
    config = load_config(CONFIG)
    model, info = build_model(config)
    phase_steps, passing, hold_start = _synthetic_passing_v3_trace(
        model, info, config
    )
    result = evaluate_trace(model, info, config, phase_steps, passing)
    assert result["passed"], result["failed_checks"]
    assert result["metrics"]["actual_hold_start_step"] == hold_start == 3250
    assert result["metrics"]["actual_hold_steps"] == 1500

    support = copy.deepcopy(passing)
    support["support_contact"][hold_start:3600] = True
    support_gate = GRASP_GATE_ORDER.index("support_contact")
    support["grasp_gate"][hold_start:3600, support_gate] = True
    support_result = evaluate_trace(model, info, config, phase_steps, support)
    assert support_result["checks"]["cube_cleared_support_and_floor"]
    assert not support_result["checks"]["support_cleared_before_hold"]

    contact = copy.deepcopy(passing)
    lost = slice(hold_start, hold_start + 500)
    contact["finger_contact_force"][lost] = 0.0
    contact["tactile_max"][lost, :3] = 0.0
    contact["distal_face_force_n"][lost] = 0.0
    contact["target_face_force_purity"][lost] = 0.0
    contact["target_face_topology"][lost] = False
    contact["target_face_effective"][lost] = False
    for finger_index, finger in enumerate(("thumb", "index", "mid")):
        gate_index = GRASP_GATE_ORDER.index(f"{finger}_target_face_effective")
        contact["grasp_gate"][lost, gate_index] = False
    topology_index = GRASP_GATE_ORDER.index("target_face_topology")
    contact["grasp_gate"][lost, topology_index] = False
    contact_result = evaluate_trace(model, info, config, phase_steps, contact)
    assert contact_result["metrics"]["contact_duty"]["thumb"] == pytest.approx(
        2.0 / 3.0
    )
    assert not contact_result["checks"]["thumb_contact_duty"]
    assert not contact_result["checks"]["thumb_target_face_contact_duty"]

    height = copy.deepcopy(passing)
    height["cube_pos"][hold_start + 10, 2] += 0.003
    height_result = evaluate_trace(model, info, config, phase_steps, height)
    assert height_result["metrics"]["hold_height_span_m"] == pytest.approx(0.003)
    assert not height_result["checks"]["hold_height_is_stable"]

    orientation = copy.deepcopy(passing)
    angle = math.radians(21.0) / 2.0
    orientation["cube_quat"][hold_start + 10] = [
        math.cos(angle),
        0.0,
        0.0,
        math.sin(angle),
    ]
    orientation_result = evaluate_trace(
        model, info, config, phase_steps, orientation
    )
    assert orientation_result["metrics"]["orientation_drift_deg"] == pytest.approx(
        21.0
    )
    assert not orientation_result["checks"]["hold_orientation_is_stable"]


def test_v3_rejects_noncanonical_gate_axis():
    config = load_config(CONFIG)
    model, info = build_model(config)
    phase_steps, traces, _ = _synthetic_passing_v3_trace(model, info, config)
    traces["grasp_gate_order"] = traces["grasp_gate_order"].copy()
    traces["grasp_gate_order"][0] = "renamed_gate"
    with pytest.raises(ValueError, match="canonical v3 gate axis"):
        evaluate_trace(model, info, config, phase_steps, traces)


def test_v3_raw_evidence_tamper_cannot_be_hidden_by_derived_booleans():
    config = load_config(CONFIG)
    model, info = build_model(config)
    phase_steps, traces, _ = _synthetic_passing_v3_trace(model, info, config)
    acquisition = int(traces["grasp_acquisition_step"])
    traces["distal_face_force_n"][acquisition, 0] = 0.0
    result = evaluate_trace(model, info, config, phase_steps, traces)
    assert not result["checks"]["target_face_evidence_matches_raw_trace"]
    assert not result["checks"]["grasp_gate_contiguous_window"]
    assert not result["stage_status"]["grasp_success"]


@pytest.mark.parametrize(
    ("mutate", "failed_check"),
    [
        (
            lambda traces: traces["grasp_acquired"].__setitem__(-1, False),
            "grasp_latch_remains_set",
        ),
        (
            lambda traces: traces["control_state"].__setitem__(-1, "MANIPULATE"),
            "controller_state_sequence_is_consistent",
        ),
        (
            lambda traces: traces.__setitem__(
                "manipulation_end_step",
                np.asarray(int(traces["manipulation_end_step"]) - 1),
            ),
            "controller_operation_events_are_consistent",
        ),
        (
            lambda traces: traces.__setitem__(
                "termination_step", np.asarray(int(traces["termination_step"]) - 1)
            ),
            "controller_termination_event_is_consistent",
        ),
        (
            lambda traces: traces["manipulation_progress"].__setitem__(-1, 0.5),
            "manipulation_progress_is_consistent",
        ),
    ],
)
def test_v3_tampered_controller_trace_fails_independent_audit(
    mutate, failed_check
):
    config = load_config(CONFIG)
    model, info = build_model(config)
    phase_steps, traces, _ = _synthetic_passing_v3_trace(model, info, config)
    mutate(traces)
    result = evaluate_trace(model, info, config, phase_steps, traces)
    assert not result["checks"][failed_check]
    assert not result["stage_status"]["full_success"]


@pytest.mark.slow
def test_v3_real_json_npz_mp4_share_one_trace_and_decode(tmp_path):
    assert os.environ.get("MUJOCO_GL") == "osmesa"
    config = load_config(CONFIG)
    trace_path = tmp_path / "trace.npz"
    video_path = tmp_path / "nominal.mp4"
    summary_path = tmp_path / "summary.json"
    summary = run_simulation(
        config, trace_path=trace_path, video_path=video_path
    )
    write_json(summary_path, summary)
    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted_summary["video"]["decode_verified"] is True
    assert persisted_summary["video"]["codec"] == "h264"
    assert persisted_summary["video"]["frame_count"] == 142
    assert video_path.stat().st_size == persisted_summary["video"]["size_bytes"]

    with np.load(trace_path, allow_pickle=False) as archive:
        traces = {name: archive[name].copy() for name in archive.files}
    for field in (
        "grasp_acquisition_step",
        "manipulation_start_step",
        "manipulation_end_step",
        "termination_step",
    ):
        assert int(traces[field]) == summary["metrics"][field]
    assert traces["video_frame_steps"].tolist() == summary["video"][
        "simulation_step_indices"
    ]

    model, info = build_model(config)
    recomputed = evaluate_trace(
        model, info, config, _phase_steps(model, config), traces
    )
    without_video = copy.deepcopy(summary)
    without_video.pop("video")
    assert json_text(recomputed) == json_text(without_video)
