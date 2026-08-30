from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_FINGERS
from xhand_grasp.contact_point_targeting import (
    contact_point_observation,
    contact_point_plan_from_config,
)
from xhand_grasp.contacts import FACE_ORDER, Face
from xhand_grasp.controller import (
    compute_grasp_gate_evidence,
    grasp_gate_order,
)
from xhand_grasp.evaluation import _v12_contact_point_metrics
from xhand_grasp.scene import build_model
from xhand_grasp.simulation import _allocate_traces
from xhand_grasp.viewer import (
    ViewerSource,
    _append_v12_contact_point_markers,
    _write_live_output,
)


ROOT = Path(__file__).resolve().parents[1]
V11_CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_larger_relative_wrist_pose_"
    "actual_contact_smooth_vertical_lift.json"
)
V12_CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_90mm_contact_point_targeted_actual_grasp_pose.json"
)


def _minimal_v12_config() -> dict:
    return {
        "schema_version": 12,
        "cube": {"edge_m": 0.09},
        "contact_topology": {
            "target_faces": {"thumb": "-X", "index": "+X", "mid": "+X"},
            "target_force_fraction": 0.95,
        },
        "contact_point_plan": {
            "schema_version": 1,
            "point_plan_id": "a" * 64,
            "cube_edge_m": 0.09,
            "coordinate_frame": "cube_local",
            "points": {
                "thumb": {"face": "-X", "yz_m": [0.010, 0.012]},
                "index": {"face": "+X", "yz_m": [0.002, 0.012]},
                "mid": {"face": "+X", "yz_m": [0.021, 0.012]},
            },
            "target_radius_m": 0.002,
            "frozen": True,
        },
        "control_protocol": {
            "stable_window_s": 0.002,
            "grasp_gate": {
                "min_target_face_force_n": 0.05,
                "min_target_force_fraction": 0.95,
                "require_touch": True,
                "require_support_contact": True,
                "max_translation_m": 0.0005,
                "max_orientation_drift_deg": 3.0,
                "max_linear_speed_m_s": 0.01,
                "max_early_lift_m": 0.001,
            },
        },
        "acceptance": {
            "touch_force_min_n": 1e-8,
            "max_palm_down_angle_deg": 40.0,
            "max_penetration_m": 0.002,
        },
    }


def test_plan_expands_face_yz_and_tangent_radius_is_inclusive():
    config = _minimal_v12_config()
    plan = contact_point_plan_from_config(config)
    np.testing.assert_allclose(
        plan.target_points_cube_local_m,
        [[-0.045, 0.010, 0.012], [0.045, 0.002, 0.012], [0.045, 0.021, 0.012]],
    )

    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    position = np.asarray([0.071, -0.027, 0.129])
    local_actual = plan.target_points_cube_local_m.copy()
    # Normal-axis displacement is deliberately ignored; exactly 2 mm in the
    # target face tangent plane remains inside the inclusive target region.
    local_actual[:, 0] += np.asarray([0.001, -0.001, -0.001])
    local_actual[:, 1] += 0.002
    world_actual = (rotation @ local_actual.T).T + position
    observation = contact_point_observation(
        plan,
        world_actual,
        np.ones(3, dtype=bool),
        np.ones(3, dtype=bool),
        position,
        rotation,
    )
    np.testing.assert_allclose(observation.tangent_error_m, 0.002, atol=1e-15)
    assert np.all(observation.within_radius)

    effective = np.ones(3, dtype=bool)
    effective[1] = False
    observation = contact_point_observation(
        plan,
        world_actual,
        np.ones(3, dtype=bool),
        effective,
        position,
        rotation,
    )
    assert observation.within_radius.tolist() == [True, False, True]
    assert not observation.all_effective_within_radius


def test_v12_gate_appends_point_region_without_changing_v11_axis():
    assert grasp_gate_order(11)[-1] == "thumb_actual_qpos_within_range"
    assert grasp_gate_order(12)[:-1] == grasp_gate_order(11)
    assert grasp_gate_order(12)[-1] == "contact_points_within_target_regions"

    config = _minimal_v12_config()
    force = np.zeros((3, len(FACE_ORDER)))
    force[0, FACE_ORDER.index(Face.X_NEG)] = 1.0
    force[1:, FACE_ORDER.index(Face.X_POS)] = 1.0
    kwargs = dict(
        distal_face_force_n=force,
        active_nondistal_force_n=np.zeros(3),
        tactile_force_n=np.ones(3),
        forbidden_contact=False,
        support_contact=True,
        floor_contact=False,
        cube_position=np.zeros(3),
        cube_quaternion=np.asarray([1.0, 0.0, 0.0, 0.0]),
        reference_position=np.zeros(3),
        reference_quaternion=np.asarray([1.0, 0.0, 0.0, 0.0]),
        cube_linear_speed_m_s=0.0,
        early_lift_m=0.0,
        palm_down_angle_deg=30.0,
        max_penetration_m=0.0,
        finite=True,
        joint_limits_respected=True,
        inactive_controls_zero=True,
        contact_height_aligned=True,
        initial_pose_history_stable=True,
        thumb_actual_qpos_within_range=True,
    )
    with pytest.raises(ValueError, match="contact_points_within"):
        compute_grasp_gate_evidence(config, **kwargs)
    evidence = compute_grasp_gate_evidence(
        config, **kwargs, contact_points_within_target_regions=False
    )
    assert not evidence.passed
    assert not evidence.as_mapping()["contact_points_within_target_regions"]


def test_v12_trace_allocation_and_offline_raw_recomputation():
    config = _minimal_v12_config()
    plan = contact_point_plan_from_config(config)
    with V11_CONFIG.open(encoding="utf-8") as handle:
        model_config = json.load(handle)
    model, _ = build_model(model_config)
    traces = _allocate_traces(model, 2, schema_version=12)
    assert traces["target_contact_points_cube_local_m"].shape == (3, 3)
    assert traces["target_face_contact_centroid_cube_local_m"].shape == (2, 3, 3)
    assert traces["target_contact_point_tangent_error_m"].shape == (2, 3)
    assert traces["target_contact_point_within_radius"].shape == (2, 3)

    traces["time"][:] = [0.001, 0.002]
    traces["cube_pos"][:] = 0.0
    traces["cube_quat"][:] = [1.0, 0.0, 0.0, 0.0]
    traces["control_state"][:] = "VERIFY"
    traces["grasp_acquisition_step"] = np.asarray(1, dtype=np.int64)
    traces["contact_point_plan_id"] = np.asarray(plan.point_plan_id)
    traces["target_contact_points_cube_local_m"][:] = plan.target_points_cube_local_m
    traces["target_contact_point_radius_m"] = np.asarray(plan.target_radius_m)
    traces["tactile_max"][:] = 1.0
    traces["grasp_gate_order"] = np.asarray(grasp_gate_order(12))
    traces["grasp_gate"] = np.ones((2, len(grasp_gate_order(12))), dtype=bool)

    target_faces = (Face.X_NEG, Face.X_POS, Face.X_POS)
    for step in range(2):
        for finger, face in enumerate(target_faces):
            face_index = FACE_ORDER.index(face)
            traces["distal_face_force_n"][step, finger, face_index] = 1.0
            traces["distal_face_position_moment_n_m"][step, finger, face_index] = (
                plan.target_points_cube_local_m[finger]
            )
        traces["target_face_contact_centroid_cube_local_m"][step] = (
            plan.target_points_cube_local_m
        )
        traces["target_contact_point_within_radius"][step] = True

    metrics, checks = _v12_contact_point_metrics(model, config, traces)
    assert all(checks.values())
    assert metrics["contact_point_targeting"]["acquisition_window"][
        "all_points_within_radius_duty"
    ] == 1.0
    assert metrics["contact_point_targeting"]["verify"]["per_finger"][
        "thumb"
    ]["tangent_error_max_m"] == pytest.approx(0.0)

    traces["target_contact_point_within_radius"][1, 1] = False
    _, checks = _v12_contact_point_metrics(model, config, traces)
    assert not checks["v12_contact_point_trace_matches_raw_contacts"]


def test_v12_viewer_appends_targets_actual_centroids_and_error_lines():
    with V11_CONFIG.open(encoding="utf-8") as handle:
        config = json.load(handle)
    model, _ = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=32)
    handle = SimpleNamespace(user_scn=scene)
    targets = np.asarray(
        [[-0.0425, 0.010, 0.012], [0.0425, 0.002, 0.012], [0.0425, 0.021, 0.012]]
    )
    actual = targets.copy()
    actual[:, 1] += [0.001, 0.003, 0.0015]
    traces = {
        "target_contact_points_cube_local_m": targets,
        "target_face_contact_centroid_cube_local_m": actual[None, :, :],
        "target_face_contact_centroid_valid": np.ones((1, 3), dtype=bool),
        "target_contact_point_within_radius": np.asarray([[True, False, True]]),
    }

    _append_v12_contact_point_markers(handle, model, data, traces, 0)

    types = [int(scene.geoms[index].type) for index in range(scene.ngeom)]
    assert types.count(int(mujoco.mjtGeom.mjGEOM_SPHERE)) == 6
    assert types.count(int(mujoco.mjtGeom.mjGEOM_LINE)) == 3
    actual_rgba = [scene.geoms[index].rgba.copy() for index in (1, 4, 7)]
    np.testing.assert_allclose(actual_rgba[0], [0.30, 1.0, 0.35, 1.0], atol=1e-7)
    np.testing.assert_allclose(actual_rgba[1], [1.0, 0.20, 0.05, 1.0], atol=1e-7)


def test_v12_viewer_override_persists_recomputed_grasp_scope_status(tmp_path):
    from xhand_grasp.config import load_config

    config = load_config(V12_CONFIG)
    config["run_context"] = {"kind": "parameter_override_run"}
    summary = {
        "passed": False,
        "failed_checks": ["operation_median_lift_reached"],
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": False,
            "full_success": False,
        },
    }
    output = tmp_path / "viewer-output"

    _write_live_output(
        output,
        config,
        summary,
        {"time": np.asarray([0.001])},
        source=ViewerSource(V12_CONFIG, None, "candidate", False),
        overridden=True,
        reference_match=None,
    )

    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    status = result["config"]["experiment_status"]
    assert status["classification"] == "parameter_override_run"
    assert status["passed"] is True
    assert status["hard_constraints_passed"] is True
    assert status["full_hard_constraints_passed"] is False
    assert status["success_scope"] == "grasp_only_contact_point_hard_checks"
