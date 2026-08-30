from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import xhand_grasp.viewer as viewer_module
from xhand_grasp.artifacts import default_artifact_path
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.controller import (
    ControlState,
    GraspGateEvidence,
    GraspVerifyThenManipulateController,
    TargetFaceEvidence,
    grasp_gate_order,
)
from xhand_grasp.experiment import get_experiment, resolve_experiment
from xhand_grasp.experiments.opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift import (
    CAMPAIGN,
    EXPERIMENT_DEFINITION,
    EXPERIMENT_ID,
)
from xhand_grasp.scene import build_model
from xhand_grasp.simulation import SimulationSession, _allocate_traces
from xhand_grasp.trajectory import minimum_jerk, smoothstep
from xhand_grasp.viewer import resolve_joint_monitor


ROOT = Path(__file__).resolve().parents[1]
V8_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_high_thumb_normal_aligned_"
    "smooth_vertical_lift.json"
)
V7_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_high_thumb_variable_size_"
    "pose_preserving_grasp_then_lift.json"
)


def _phase_steps(manipulate: int = 4) -> dict[str, int]:
    return {
        "settle": 1,
        "close": 1,
        "verify": 250,
        "manipulate": manipulate,
        "hold": 1,
    }


def _passing_gate(schema_version: int) -> GraspGateEvidence:
    order = grasp_gate_order(schema_version)
    target = TargetFaceEvidence(
        target_force_n=np.full(3, 0.1),
        total_distal_force_n=np.full(3, 0.1),
        target_force_purity=np.ones(3),
        target_face_effective=np.ones(3, dtype=bool),
        material_off_target=np.zeros(3, dtype=bool),
        material_active_nondistal=np.zeros(3, dtype=bool),
    )
    return GraspGateEvidence(
        components=np.ones(len(order), dtype=bool),
        target_faces=target,
        hard_abort=False,
        gate_order=order,
    )


def _controller_commands(config: dict) -> tuple[
    GraspVerifyThenManipulateController, list
]:
    model, _ = build_model(config)
    controller = GraspVerifyThenManipulateController(
        model, config, _phase_steps()
    )
    if int(config["schema_version"]) >= 6:
        controller.latch_initial_pose(
            np.asarray([0.071, -0.027, 0.1175]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
    gate = _passing_gate(int(config["schema_version"]))
    position = np.asarray([0.071, -0.027, 0.1175])
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    commands = []
    for step in range(controller.total_steps):
        command = controller.command(step)
        commands.append(command)
        controller.observe(step, gate, position, quaternion)
    controller.finish()
    return controller, commands


def test_v8_template_loads_and_resolves_registered_experiment():
    config = load_config(V8_CONFIG)
    definition = resolve_experiment(config)

    assert config["schema_version"] == 8
    assert config["experiment_id"] == EXPERIMENT_ID
    assert definition is EXPERIMENT_DEFINITION
    assert get_experiment(EXPERIMENT_ID) is definition
    assert definition.normal_aligned_smooth_lift_campaign is CAMPAIGN
    assert config["normal_aligned_smooth_lift_campaign"] == CAMPAIGN.as_config()
    assert config["control_protocol"]["manipulation_profile"] == (
        "minimum_jerk_quintic"
    )
    assert default_artifact_path(config, "run") == Path(
        "artifacts/"
        "left_opposed_face_palm_down_high_thumb_normal_aligned_"
        "smooth_vertical_lift/nominal"
    )


def test_v8_controller_uses_minimum_jerk_endpoints_and_v7_stays_cubic():
    v8 = load_config(V8_CONFIG)
    v7 = load_config(V7_CONFIG)
    actuator = "left_hand_thumb_rota_joint2_actuator"
    for config in (v8, v7):
        config["control"]["manipulation_delta_rad"][actuator] = 0.1

    v8_controller, v8_commands = _controller_commands(v8)
    v7_controller, v7_commands = _controller_commands(v7)
    v8_start = v8_controller.manipulation_start_step
    v7_start = v7_controller.manipulation_start_step
    v8_manipulate = v8_commands[v8_start : v8_start + 4]
    v7_manipulate = v7_commands[v7_start : v7_start + 4]

    assert v8_commands[v8_start - 1].state is ControlState.VERIFY
    np.testing.assert_array_equal(
        v8_commands[v8_start - 1].target, v8_controller.grasp_target
    )
    assert [command.manipulation_progress for command in v8_manipulate] == (
        pytest.approx([minimum_jerk(value) for value in (0.25, 0.5, 0.75, 1.0)])
    )
    assert [command.manipulation_progress for command in v7_manipulate] == (
        pytest.approx([smoothstep(value) for value in (0.25, 0.5, 0.75, 1.0)])
    )
    np.testing.assert_array_equal(
        v8_manipulate[-1].target, v8_controller.manipulation_target
    )
    np.testing.assert_array_equal(
        v8_manipulate[-1].target_velocity_rad_s,
        np.zeros(v8_controller.model.nu),
    )
    assert np.any(np.abs(v8_manipulate[0].target_velocity_rad_s) > 0.0)
    for command in v7_commands:
        np.testing.assert_array_equal(
            command.target_velocity_rad_s, np.zeros(v7_controller.model.nu)
        )

    # Finite differences also protect the zero velocity/acceleration contract
    # at both ends of the public profile, not just its position endpoints.
    h = 1e-5
    assert minimum_jerk(0.0) == 0.0
    assert minimum_jerk(1.0) == 1.0
    endpoint_velocity = (
        (minimum_jerk(h) - minimum_jerk(0.0)) / h,
        (minimum_jerk(1.0) - minimum_jerk(1.0 - h)) / h,
    )
    endpoint_acceleration = (
        (
            minimum_jerk(2.0 * h)
            - 2.0 * minimum_jerk(h)
            + minimum_jerk(0.0)
        )
        / h**2,
        (
            minimum_jerk(1.0)
            - 2.0 * minimum_jerk(1.0 - h)
            + minimum_jerk(1.0 - 2.0 * h)
        )
        / h**2,
    )
    assert endpoint_velocity == pytest.approx((0.0, 0.0), abs=1e-8)
    assert endpoint_acceleration == pytest.approx((0.0, 0.0), abs=1e-3)


def test_v8_trace_extension_is_append_only_and_wired_into_session():
    config = load_config(V8_CONFIG)
    model, _ = build_model(config)
    v7_traces = _allocate_traces(model, 3, schema_version=7)
    v8_traces = _allocate_traces(model, 3, schema_version=8)
    expected_v8_fields = {
        "command_target_velocity_rad_s": (3, model.nu),
        "closure_witness_world_m": (3, 3, 3),
        "closure_command_velocity_world_m_s": (3, 3, 3),
        "closure_cube_outward_normal_world": (3, 3, 3),
        "closure_alignment_cosine": (3, 3),
        "closure_alignment_angle_deg": (3, 3),
        "closure_inward_speed_m_s": (3, 3),
        "closure_tangent_speed_m_s": (3, 3),
        "closure_alignment_valid": (3, 3),
        "closure_target_contact_force_n": (3, 3),
        "operation_height_filtered_m": (3,),
        "operation_vertical_velocity_filtered_m_s": (3,),
        "operation_vertical_acceleration_filtered_m_s2": (3,),
        "operation_vertical_jerk_filtered_m_s3": (3,),
        "operation_lateral_displacement_m": (3,),
        "operation_orientation_drift_deg": (3,),
        "motion_filter_valid": (3,),
    }

    assert set(v8_traces) - set(v7_traces) == set(expected_v8_fields)
    for name, shape in expected_v8_fields.items():
        assert v8_traces[name].shape == shape
        assert name not in v7_traces

    session = SimulationSession(config)
    try:
        assert set(expected_v8_fields).issubset(session.traces)
        first = session.advance_one()
        assert first.index == 0
        np.testing.assert_array_equal(
            session.traces["command_target_velocity_rad_s"][0],
            np.zeros(model.nu),
        )
        assert not np.any(session.traces["closure_alignment_valid"][0])
    finally:
        session.close()


def test_v8_viewer_markers_coexist_with_alignment_path_and_joint_axis():
    config = load_config(V8_CONFIG)
    model, _ = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    monitor = resolve_joint_monitor(model, config, None)
    assert monitor is not None
    scene = mujoco.MjvScene(model, maxgeom=64)
    handle = SimpleNamespace(user_scn=scene)
    witnesses = np.asarray(
        [
            [0.040, -0.027, 0.115],
            [0.102, -0.032, 0.115],
            [0.102, -0.022, 0.115],
        ]
    )
    outward = np.asarray(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    traces = {
        "target_face_contact_centroid_world_m": np.tile(
            witnesses[None, :, :], (4, 1, 1)
        ),
        "target_face_contact_centroid_valid": np.ones((4, 3), dtype=bool),
        "closure_witness_world_m": np.tile(witnesses[None, :, :], (4, 1, 1)),
        "closure_command_velocity_world_m_s": np.tile(
            (-0.01 * outward)[None, :, :], (4, 1, 1)
        ),
        "closure_cube_outward_normal_world": np.tile(
            outward[None, :, :], (4, 1, 1)
        ),
        "closure_alignment_valid": np.ones((4, 3), dtype=bool),
        "cube_pos": np.asarray(
            [
                [0.0710, -0.0270, 0.1150],
                [0.0710, -0.0270, 0.1150],
                [0.0713, -0.0270, 0.1200],
                [0.0711, -0.0270, 0.1260],
            ]
        ),
        "control_state": np.asarray(
            ["VERIFY", "MANIPULATE", "MANIPULATE", "HOLD"]
        ),
        "manipulation_start_step": np.asarray(1, dtype=np.int64),
    }

    viewer_module._update_viewer_markers(
        handle,
        model,
        data,
        traces,
        3,
        show_alignment=True,
        joint_monitor=monitor,
    )

    types = [int(scene.geoms[index].type) for index in range(scene.ngeom)]
    assert types.count(int(mujoco.mjtGeom.mjGEOM_SPHERE)) == 3
    assert types.count(int(mujoco.mjtGeom.mjGEOM_BOX)) == 3
    assert types.count(int(mujoco.mjtGeom.mjGEOM_ARROW)) == 7
    assert types.count(int(mujoco.mjtGeom.mjGEOM_CAPSULE)) == 1
    assert types.count(int(mujoco.mjtGeom.mjGEOM_LINE)) == 2
    assert scene.ngeom == 16
