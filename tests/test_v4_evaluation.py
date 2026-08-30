from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import load_config
from xhand_grasp.contacts import FACE_ORDER, Face
from xhand_grasp.controller import V4_GRASP_GATE_ORDER
from xhand_grasp.evaluation import _v4_alignment_metrics
from xhand_grasp.scene import build_model


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_grasp_then_lift.json"
)


def _aligned_trace():
    source = load_config(SOURCE_CONFIG)
    model, _ = build_model(source)
    config = copy.deepcopy(source)
    config["schema_version"] = 4
    config["contact_alignment"] = {
        "max_height_spread_m": 0.005,
        "verify_continuous_s": 0.002,
        "operation_aligned_duty": 0.70,
    }
    reference_root = np.array([0.0, 0.0, 0.2])
    config["pose_constraints"] = {
        "reference_hand_translation_m": reference_root.tolist(),
        "finger_down_tilt_deg": [10.0, 20.0],
        "palm_plane_ground_angle_deg": [10.0, 20.0],
        "palm_press_depth_m": [0.002, 0.008],
    }

    count = 4
    face_force = np.zeros((count, 3, len(FACE_ORDER)))
    face_moment = np.zeros((count, 3, len(FACE_ORDER), 3))
    target_indices = np.array(
        [
            FACE_ORDER.index(Face.X_NEG),
            FACE_ORDER.index(Face.X_POS),
            FACE_ORDER.index(Face.X_POS),
        ]
    )
    centroids = np.array(
        [[0.0, 0.0, 0.100], [0.0, 0.0, 0.105], [0.0, 0.0, 0.103]]
    )
    for step in range(count):
        for finger, face_index in enumerate(target_indices):
            face_force[step, finger, face_index] = 0.1
            face_moment[step, finger, face_index] = 0.1 * centroids[finger]

    gate = np.ones((count, len(V4_GRASP_GATE_ORDER)), dtype=bool)
    traces = {
        "time": np.arange(count, dtype=float) * model.opt.timestep,
        "distal_face_force_n": face_force,
        "distal_face_position_moment_n_m": face_moment,
        "tactile_max": np.full((count, 5), 0.1),
        "target_face_contact_centroid_world_m": np.broadcast_to(
            centroids, (count, 3, 3)
        ).copy(),
        "target_face_contact_centroid_valid": np.ones((count, 3), dtype=bool),
        "three_contact_height_spread_m": np.full(count, 0.005),
        "three_contact_height_aligned": np.ones(count, dtype=bool),
        "grasp_gate_order": np.asarray(V4_GRASP_GATE_ORDER),
        "grasp_gate": gate,
        "control_state": np.asarray(
            ["VERIFY", "VERIFY", "MANIPULATE", "HOLD"]
        ),
        "grasp_acquisition_step": np.asarray(1),
        "finger_down_tilt_deg": np.full(count, 15.0),
        "palm_down_angle_deg": np.full(count, 15.0),
        "root_pos": np.broadcast_to(
            reference_root - np.array([0.0, 0.0, 0.005]), (count, 3)
        ).copy(),
    }
    return model, config, traces


def test_v4_alignment_recomputes_boundary_and_reports_each_stage():
    model, config, traces = _aligned_trace()
    metrics, checks = _v4_alignment_metrics(model, config, traces)
    assert all(checks.values())
    assert metrics["contact_alignment"]["verify"]["aligned_duty"] == 1.0
    assert metrics["contact_alignment"]["manipulate"]["aligned_duty"] == 1.0
    assert metrics["contact_alignment"]["hold"]["aligned_duty"] == 1.0
    assert metrics["contact_alignment"]["operation"][
        "height_spread_max_m"
    ] == pytest.approx(0.005)


def test_v4_missing_contact_cannot_turn_zero_spread_into_alignment():
    model, config, traces = _aligned_trace()
    traces["distal_face_force_n"][-1, 2] = 0.0
    traces["distal_face_position_moment_n_m"][-1, 2] = 0.0
    traces["target_face_contact_centroid_world_m"][-1, 2] = 0.0
    traces["target_face_contact_centroid_valid"][-1, 2] = False
    traces["three_contact_height_spread_m"][-1] = 0.0
    traces["three_contact_height_aligned"][-1] = False
    alignment_index = V4_GRASP_GATE_ORDER.index("contact_height_aligned")
    traces["grasp_gate"][-1, alignment_index] = False

    metrics, checks = _v4_alignment_metrics(model, config, traces)
    assert checks["v4_alignment_trace_matches_raw_contacts"]
    assert metrics["contact_alignment"]["hold"]["effective_duty"] == 0.0
    assert metrics["contact_alignment"]["hold"]["aligned_duty"] == 0.0
    assert metrics["contact_alignment"]["hold"]["height_spread_max_m"] is None
