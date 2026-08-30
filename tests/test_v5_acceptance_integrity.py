from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from xhand_grasp.config import (
    load_config,
    resolved_pose_constraint_values,
    validate_config,
)
from xhand_grasp.contacts import FACE_ORDER, Face
from xhand_grasp.controller import V4_GRASP_GATE_ORDER
from xhand_grasp.evaluation import (
    _v2_face_metrics,
    _v4_alignment_metrics,
    evaluate_trace,
)
from xhand_grasp.scene import (
    build_model,
    rpy_degrees_to_rotation_matrix,
)
from xhand_grasp.simulation import SimulationSession


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift.json"
)


def test_v5_operation_contact_violations_start_at_manipulation_not_support_clear():
    config = load_config(CONFIG)
    model, _ = build_model(config)
    count = 10
    face_force = np.zeros((count, 3, len(FACE_ORDER)), dtype=np.float64)
    target_indices = (Face.X_NEG, Face.X_POS, Face.X_POS)
    for finger_index, face in enumerate(target_indices):
        face_force[:, finger_index, int(face)] = 0.1
    traces = {
        "time": np.arange(count, dtype=np.float64) * model.opt.timestep,
        "distal_face_force_n": face_force,
        "active_nondistal_force_n": np.zeros((count, 3), dtype=np.float64),
        "tactile_max": np.full((count, 5), 0.1, dtype=np.float64),
        "support_contact": np.asarray(
            [True, True, True, True, True, False, False, False, False, False]
        ),
        "floor_contact": np.zeros(count, dtype=bool),
        "palm_down_angle_deg": np.full(count, 15.0),
    }
    # This proximal-link load occurs after manipulation starts (step 3), but
    # before the final support contact (step 4).  The historical v2-v4 scope
    # intentionally ignores it; schema-v5 must not.
    traces["active_nondistal_force_n"][3, 0] = 0.1
    phase_steps = {"hold": 2}

    legacy_metrics, legacy_checks = _v2_face_metrics(
        model,
        config,
        phase_steps,
        traces,
        hold_start_override=8,
    )
    v5_metrics, v5_checks = _v2_face_metrics(
        model,
        config,
        phase_steps,
        traces,
        hold_start_override=8,
        material_start_override=3,
    )

    assert legacy_metrics["unsupported_interval_start_step"] == 5
    assert legacy_checks["active_nondistal_contacts_within_limit"]
    assert v5_metrics["material_contact_interval_start_step"] == 3
    assert v5_metrics["material_contact_interval_scope"] == "operation_and_hold"
    assert not v5_checks["active_nondistal_contacts_within_limit"]


@pytest.mark.slow
def test_v5_raw_trace_integrity_cannot_be_hidden_by_persisted_helpers():
    config = load_config(CONFIG)
    session = SimulationSession(config)
    try:
        while not session.complete:
            session.advance_one()
        session.finalize()
        model = session.model
        info = session.info
        phase_steps = session.phase_steps
        pristine = {name: value.copy() for name, value in session.traces.items()}
    finally:
        session.close()

    nondistal = copy.deepcopy(pristine)
    nondistal["active_nondistal_force_n"][0, 0] = 10.0
    result = evaluate_trace(model, info, config, phase_steps, nondistal)
    assert not result["checks"]["v5_contact_exclusion_gate_matches_raw_trace"]
    assert not result["stage_status"]["grasp_success"]

    pad_fraction = copy.deepcopy(pristine)
    pad_fraction["distal_pad_force_fraction"][0, 0] = 0.25
    result = evaluate_trace(model, info, config, phase_steps, pad_fraction)
    assert not result["checks"]["v5_pad_fraction_trace_matches_raw_forces"]

    pose = copy.deepcopy(pristine)
    pose["root_cube_center_distance_m"][0] += 1e-6
    result = evaluate_trace(model, info, config, phase_steps, pose)
    assert not result["checks"][
        "v5_pose_and_thumb_bend_trace_matches_raw_state"
    ]

    command = copy.deepcopy(pristine)
    thumb_id = model.actuator("left_hand_thumb_bend_joint_actuator").id
    command["ctrl"][0, thumb_id] += 0.01
    command["thumb_bend_command_rad"][0] += 0.01
    result = evaluate_trace(model, info, config, phase_steps, command)
    assert result["checks"]["v5_pose_and_thumb_bend_trace_matches_raw_state"]
    assert not result["checks"][
        "v5_thumb_bend_command_matches_control_protocol"
    ]


def _distance_trace(model: mujoco.MjModel, config: dict) -> dict[str, np.ndarray]:
    count = 4
    face_force = np.zeros((count, 3, len(FACE_ORDER)), dtype=np.float64)
    face_moment = np.zeros(
        (count, 3, len(FACE_ORDER), 3), dtype=np.float64
    )
    centroids = np.asarray(
        [[0.0, 0.0, 0.100], [0.0, 0.0, 0.103], [0.0, 0.0, 0.102]]
    )
    target_faces = (Face.X_NEG, Face.X_POS, Face.X_POS)
    for step in range(count):
        for finger_index, face in enumerate(target_faces):
            face_force[step, finger_index, int(face)] = 0.1
            face_moment[step, finger_index, int(face)] = (
                0.1 * centroids[finger_index]
            )
    root = np.asarray(config["hand_pose"]["translation_m"], dtype=np.float64)
    cube_in_root = np.asarray(
        resolved_pose_constraint_values(config)["cube_position_in_root_m"],
        dtype=np.float64,
    )
    rotation = rpy_degrees_to_rotation_matrix(config["hand_pose"]["rpy_deg"])
    cube = root + rotation @ cube_in_root
    return {
        "time": np.arange(count, dtype=np.float64) * model.opt.timestep,
        "distal_face_force_n": face_force,
        "distal_face_position_moment_n_m": face_moment,
        "tactile_max": np.full((count, 5), 0.1),
        "target_face_contact_centroid_world_m": np.broadcast_to(
            centroids, (count, 3, 3)
        ).copy(),
        "target_face_contact_centroid_valid": np.ones((count, 3), dtype=bool),
        "three_contact_height_spread_m": np.full(count, 0.003),
        "three_contact_height_aligned": np.ones(count, dtype=bool),
        "grasp_gate_order": np.asarray(V4_GRASP_GATE_ORDER),
        "grasp_gate": np.ones((count, len(V4_GRASP_GATE_ORDER)), dtype=bool),
        "control_state": np.asarray(
            ["VERIFY", "VERIFY", "MANIPULATE", "HOLD"]
        ),
        "grasp_acquisition_step": np.asarray(1),
        "finger_down_tilt_deg": np.full(count, 15.0),
        "palm_down_angle_deg": np.full(count, 15.0),
        "root_pos": np.broadcast_to(root, (count, 3)).copy(),
        "cube_pos": np.broadcast_to(cube, (count, 3)).copy(),
    }


def test_v5_distance_expansion_is_hard_accepted_only_with_valid_metadata():
    base = load_config(CONFIG)
    base_model, base_info = build_model(base)
    cube_world = np.asarray(
        base_model.qpos0[
            base_info.cube_qpos_adr : base_info.cube_qpos_adr + 3
        ],
        dtype=np.float64,
    )
    cube_in_root = np.asarray([0.105, -0.025, 0.120], dtype=np.float64)
    assert 0.160 < np.linalg.norm(cube_in_root) < 0.165
    rotation = rpy_degrees_to_rotation_matrix(base["hand_pose"]["rpy_deg"])

    expanded = copy.deepcopy(base)
    expanded["hand_pose"]["translation_m"] = (
        cube_world - rotation @ cube_in_root
    ).tolist()
    expanded["candidate_metadata"] = {
        "boundary_expansion": {
            "applied": True,
            "count": 1,
            "distance_expanded": True,
            "thumb_bend_expanded": False,
            "reason": "nominal_upper_boundary_hit",
        }
    }
    validate_config(expanded)
    expanded_model, _ = build_model(expanded)
    metrics, checks = _v4_alignment_metrics(
        expanded_model, expanded, _distance_trace(expanded_model, expanded)
    )
    assert checks["initial_root_cube_distance_within_range"]
    assert metrics["root_cube_center_distance_m"]["acceptance_range"] == [
        0.138,
        0.165,
    ]
    assert metrics["root_cube_center_distance_m"]["upper_boundary_expanded"]

    override = copy.deepcopy(expanded)
    override.pop("candidate_metadata")
    override["run_context"] = {"kind": "parameter_override_run"}
    validate_config(override)
    override_model, _ = build_model(override)
    metrics, checks = _v4_alignment_metrics(
        override_model, override, _distance_trace(override_model, override)
    )
    assert not checks["initial_root_cube_distance_within_range"]
    assert metrics["root_cube_center_distance_m"]["acceptance_range"] == [
        0.138,
        0.160,
    ]
    assert not metrics["root_cube_center_distance_m"][
        "upper_boundary_expanded"
    ]
