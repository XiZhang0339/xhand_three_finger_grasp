from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from xhand_grasp.config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, load_config
from xhand_grasp.contacts import FACE_ORDER
from xhand_grasp.controller import grasp_gate_order
from xhand_grasp.experiment import ContactForceTargets
from xhand_grasp.evaluation import (
    _protocol_steps,
    _v3_stage_metrics,
    _v14_contact_preservation_metrics,
    face_from_label,
)
from xhand_grasp.scene import build_model
from xhand_grasp.simulation import SimulationSession, _allocate_traces
from xhand_grasp.trajectory import (
    interpolate_quintic_c2,
    quintic_c2_knot_derivatives,
)
from xhand_grasp.v14_identity import (
    V14_TOP_LEVEL_ID_FIELDS,
    v14_base_controller_id,
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_identity_trace_values,
    v14_object_config_id,
    v14_sequential_planner_id,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _model_config():
    config = load_config(CONFIG)
    model, _ = build_model(config)
    return model, config


def _resolved_model_config():
    model, config = _model_config()
    config["object_config_id"] = v14_object_config_id(config)
    config["grasp_pose_id"] = v14_grasp_pose_id(config)
    config["grasp_object_pair_id"] = v14_grasp_object_pair_id(config)
    report_id = "1" * 64
    attempt_id = "2" * 64
    config.setdefault("candidate_metadata", {})[
        "sequential_checkpoint_planning"
    ] = {
        "report_id": report_id,
        "attempt_report_id": attempt_id,
    }
    config["planner_id"] = v14_sequential_planner_id(report_id, attempt_id)
    config["controller_id"] = v14_base_controller_id(
        config, bind_planner=False
    )
    return model, config


def _plan_interpolation_cache(model, config: dict):
    plan = config["manipulation_plan"]
    knots = np.asarray(plan["knot_times_s"], dtype=np.float64)
    waypoint_knots = np.zeros((knots.size, model.nu), dtype=np.float64)
    for name in ACTIVE_ACTUATORS:
        values = np.asarray(plan["actuator_waypoints_rad"][name])
        waypoint_knots[:, model.actuator(name).id] = values
    preload = np.zeros(model.nu, dtype=np.float64)
    for name, value in config["control"]["contact_preload_targets_rad"].items():
        preload[model.actuator(name).id] = float(value)
    position_knots = np.asarray(plan["desired_cube_position_delta_m"])
    rotation_knots = np.asarray(plan["desired_cube_rotation_vector_rad"])
    waypoint_derivatives = quintic_c2_knot_derivatives(knots, waypoint_knots)
    position_derivatives = quintic_c2_knot_derivatives(knots, position_knots)
    rotation_derivatives = quintic_c2_knot_derivatives(knots, rotation_knots)
    return (
        knots,
        waypoint_knots,
        position_knots,
        rotation_knots,
        preload,
        waypoint_derivatives,
        position_derivatives,
        rotation_derivatives,
    )


def _fill_plan_sample(model, config: dict, progress: float, cache=None):
    plan = config["manipulation_plan"]
    duration = float(plan["duration_s"])
    elapsed = float(progress * duration)
    (
        knots,
        waypoint_knots,
        position_knots,
        rotation_knots,
        preload,
        waypoint_derivatives,
        position_derivatives,
        rotation_derivatives,
    ) = _plan_interpolation_cache(model, config) if cache is None else cache
    waypoint, _, _, index = interpolate_quintic_c2(
        knots,
        waypoint_knots,
        elapsed,
        knot_velocities=waypoint_derivatives[0],
        knot_accelerations=waypoint_derivatives[1],
    )
    position, _, _, _ = interpolate_quintic_c2(
        knots,
        position_knots,
        elapsed,
        knot_velocities=position_derivatives[0],
        knot_accelerations=position_derivatives[1],
    )
    rotation, _, _, _ = interpolate_quintic_c2(
        knots,
        rotation_knots,
        elapsed,
        knot_velocities=rotation_derivatives[0],
        knot_accelerations=rotation_derivatives[1],
    )
    return preload + waypoint, position, rotation, index


def _passing_v14_trace(model, config: dict) -> dict[str, np.ndarray]:
    # The 3 s plan runs without a freeze, followed by a short HOLD interval.
    verify_steps = 2
    manipulate_steps = int(round(config["manipulation_plan"]["duration_s"] / model.opt.timestep))
    hold_steps = 100
    total = verify_steps + manipulate_steps + hold_steps
    traces = _allocate_traces(model, total, schema_version=14)
    for name, value in v14_identity_trace_values(config).items():
        traces[name] = np.asarray(value, dtype=np.str_)
    traces["time"][:] = np.arange(1, total + 1) * model.opt.timestep
    traces["control_state"][:verify_steps] = "VERIFY"
    traces["control_state"][verify_steps : verify_steps + manipulate_steps] = (
        "MANIPULATE"
    )
    traces["control_state"][verify_steps + manipulate_steps :] = "HOLD"

    target_indices = [
        FACE_ORDER.index(
            face_from_label(config["contact_topology"]["target_faces"][finger])
        )
        for finger in ACTIVE_FINGERS
    ]
    for finger, face_index in enumerate(target_indices):
        traces["distal_face_force_n"][:, finger, face_index] = 1.0
    traces["tactile_max"][:] = 0.0
    traces["tactile_max"][:, :3] = 1.0
    traces["target_face_effective"][:] = True

    plan = config["manipulation_plan"]
    knots = np.asarray(plan["knot_times_s"], dtype=np.float64)
    traces["manipulation_plan_knot_times_s"][:] = knots
    for name in ACTIVE_ACTUATORS:
        traces["manipulation_plan_waypoints_rad"][:, model.actuator(name).id] = (
            plan["actuator_waypoints_rad"][name]
        )
    traces["manipulation_plan_desired_cube_position_delta_m"][:] = plan[
        "desired_cube_position_delta_m"
    ]
    traces["manipulation_plan_desired_cube_rotation_vector_rad"][:] = plan[
        "desired_cube_rotation_vector_rad"
    ]
    traces["manipulation_plan_id"] = np.asarray(plan["plan_id"])
    traces["contact_feedback_id"] = np.asarray(
        config["contact_feedback"]["feedback_id"]
    )
    traces["operation_feedback_source_step"][:] = np.arange(total) - 1
    interpolation_cache = _plan_interpolation_cache(model, config)

    for offset in range(manipulate_steps):
        step = verify_steps + offset
        progress = (offset + 1) / manipulate_steps
        traces["manipulation_progress"][step] = progress
        feedforward, position, rotation, knot_index = _fill_plan_sample(
            model, config, progress, interpolation_cache
        )
        traces["planned_feedforward_target_rad"][step] = feedforward
        traces["desired_cube_position_delta_m"][step] = position
        traces["desired_cube_rotation_vector_rad"][step] = rotation
        traces["planned_knot_index"][step] = knot_index
        traces["ctrl"][step] = feedforward
    for step in range(verify_steps + manipulate_steps, total):
        traces["manipulation_progress"][step] = 1.0
        feedforward, position, rotation, knot_index = _fill_plan_sample(
            model, config, 1.0, interpolation_cache
        )
        traces["planned_feedforward_target_rad"][step] = feedforward
        traces["desired_cube_position_delta_m"][step] = position
        traces["desired_cube_rotation_vector_rad"][step] = rotation
        traces["planned_knot_index"][step] = knot_index
        traces["ctrl"][step] = feedforward

    configured_targets = np.asarray(
        [
            config["contact_force_targets_n"]["per_finger_n"][finger]
            for finger in ACTIVE_FINGERS
        ],
        dtype=np.float64,
    )
    traces["contact_force_target_n"][:] = configured_targets
    traces["resolved_contact_force_targets_n"] = configured_targets.copy()
    traces["contact_force_filtered_n"][:] = 1.0
    traces["maximum_contact_loss_run_steps"] = np.zeros(3, dtype=np.int64)
    return traces


def test_v14_recomputes_99_percent_duty_loss_and_plan_evidence() -> None:
    model, config = _model_config()
    traces = _passing_v14_trace(model, config)
    metrics, checks = _v14_contact_preservation_metrics(model, config, traces)
    assert all(checks.values())
    contact = metrics["contact_preserving_planned_lift"]
    assert contact["simultaneous_target_face_effective_duty"] == 1.0
    assert contact["maximum_plan_progress"] == 1.0

    corrupted_source = traces.copy()
    corrupted_source["operation_feedback_source_step"] = traces[
        "operation_feedback_source_step"
    ].copy()
    corrupted_source["operation_feedback_source_step"][100] = 100
    _, corrupted_checks = _v14_contact_preservation_metrics(
        model, config, corrupted_source
    )
    assert corrupted_checks["v14_feedback_uses_previous_observation"] is False

    # Eleven consecutive lost samples still exceed 99% duty on this long
    # trace, but violate the independent 10 ms maximum-loss requirement.
    lost = np.arange(100, 111)
    thumb_face = FACE_ORDER.index(
        face_from_label(config["contact_topology"]["target_faces"]["thumb"])
    )
    traces["distal_face_force_n"][lost, 0, thumb_face] = 0.0
    traces["target_face_effective"][lost, 0] = False
    _, failed = _v14_contact_preservation_metrics(model, config, traces)
    assert failed["v14_thumb_contact_duty_at_least_99_percent"] is True
    assert failed["v14_thumb_contact_loss_within_limit"] is False


def test_v14_npz_identity_chain_is_independently_recomputed(
    tmp_path: Path,
) -> None:
    model, config = _resolved_model_config()
    traces = _passing_v14_trace(model, config)
    archive = tmp_path / "trace.npz"
    np.savez_compressed(archive, **traces)
    with np.load(archive, allow_pickle=False) as loaded:
        persisted = {name: loaded[name] for name in loaded.files}
    metrics, checks = _v14_contact_preservation_metrics(
        model, config, persisted
    )
    assert checks[
        "v14_top_level_identity_trace_matches_recomputed_config"
    ] is True
    identity = metrics["contact_preserving_planned_lift"][
        "top_level_identity"
    ]
    assert identity["state"] == "resolved"
    assert identity["expected"] == {
        name: config[name] for name in V14_TOP_LEVEL_ID_FIELDS
    }

    for name in V14_TOP_LEVEL_ID_FIELDS:
        corrupted = dict(persisted)
        corrupted[name] = np.asarray("0" * 64, dtype=np.str_)
        _, failed = _v14_contact_preservation_metrics(
            model, config, corrupted
        )
        assert failed[
            "v14_top_level_identity_trace_matches_recomputed_config"
        ] is False


def test_v14_simulation_session_installs_resolved_identity_scalars() -> None:
    _, config = _resolved_model_config()
    session = SimulationSession(config)
    try:
        for name in V14_TOP_LEVEL_ID_FIELDS:
            assert np.asarray(session.traces[name]).shape == ()
            assert str(np.asarray(session.traces[name]).reshape(())) == config[name]
    finally:
        session.close()


def test_v14_recomputes_scaled_operation_force_targets() -> None:
    model, config = _model_config()
    original = config["contact_force_targets_n"]
    config["contact_force_targets_n"] = ContactForceTargets(
        schema_version=int(original["schema_version"]),
        source=str(original["source"]),
        per_finger_n=copy.deepcopy(original["per_finger_n"]),
        minimum_n=float(original["minimum_n"]),
        maximum_n=float(original["maximum_n"]),
        operation_scale=0.70,
    ).as_config()
    config["control_protocol"] = copy.deepcopy(config["control_protocol"])
    config["control_protocol"]["stable_window_s"] = float(model.opt.timestep)
    traces = _passing_v14_trace(model, config)
    traces["grasp_acquisition_step"] = np.asarray(0, dtype=np.int64)
    configured = np.asarray(
        [
            config["contact_force_targets_n"]["per_finger_n"][finger]
            for finger in ACTIVE_FINGERS
        ],
        dtype=np.float64,
    )
    expected = np.clip(
        0.70 * np.maximum(configured, np.ones(3, dtype=np.float64)),
        float(config["contact_force_targets_n"]["minimum_n"]),
        float(config["contact_force_targets_n"]["maximum_n"]),
    )
    traces["contact_force_target_n"][:] = expected
    traces["resolved_contact_force_targets_n"] = expected.copy()

    _, checks = _v14_contact_preservation_metrics(model, config, traces)
    assert checks["v14_force_feedback_trace_is_bounded_and_recomputable"] is True

    traces["resolved_contact_force_targets_n"] = configured.copy()
    _, corrupted = _v14_contact_preservation_metrics(model, config, traces)
    assert corrupted["v14_force_feedback_trace_is_bounded_and_recomputable"] is False


def test_v14_operation_abort_does_not_revoke_authenticated_grasp() -> None:
    model, config = _model_config()
    phase = _protocol_steps(model, config)
    total = sum(phase.values())
    traces = _allocate_traces(model, total, schema_version=14)
    gate_order = grasp_gate_order(14)
    traces["grasp_gate_order"] = np.asarray(gate_order)
    traces["grasp_gate"] = np.ones((total, len(gate_order)), dtype=bool)
    traces["target_face_effective"][:] = True
    traces["target_face_topology"][:] = True
    traces["target_face_force_purity"][:] = 1.0
    traces["tactile_max"][:] = 0.0
    traces["tactile_max"][:, :3] = 1.0
    for finger, name in enumerate(ACTIVE_FINGERS):
        face = FACE_ORDER.index(
            face_from_label(config["contact_topology"]["target_faces"][name])
        )
        traces["distal_face_force_n"][:, finger, face] = 1.0

    settle_end = phase["settle"]
    close_end = settle_end + phase["close"]
    acquisition = close_end + int(
        round(config["control_protocol"]["stable_window_s"] / model.opt.timestep)
    ) - 1
    manipulation_start = acquisition + 1
    abort_observation = manipulation_start + 100
    traces["control_state"][:settle_end] = "SETTLE"
    traces["control_state"][settle_end:close_end] = "CLOSE"
    traces["control_state"][close_end:manipulation_start] = "VERIFY"
    traces["control_state"][manipulation_start : abort_observation + 1] = (
        "MANIPULATE"
    )
    traces["control_state"][abort_observation + 1 :] = "ABORT"
    traces["grasp_acquired"][acquisition:] = True
    stable_steps = acquisition - close_end + 1
    traces["grasp_gate_consecutive_steps"][close_end:manipulation_start] = (
        np.arange(1, stable_steps + 1)
    )
    traces["grasp_gate_consecutive_steps"][
        manipulation_start:abort_observation
    ] = stable_steps
    traces["manipulation_progress"][
        manipulation_start : abort_observation + 1
    ] = np.arange(1, abort_observation - manipulation_start + 2) / phase[
        "manipulate"
    ]
    traces["grasp_acquisition_step"] = np.asarray(acquisition, dtype=np.int64)
    traces["grasp_lock_step"] = np.asarray(acquisition, dtype=np.int64)
    traces["manipulation_start_step"] = np.asarray(
        manipulation_start, dtype=np.int64
    )
    traces["manipulation_end_step"] = np.asarray(-1, dtype=np.int64)
    traces["termination_step"] = np.asarray(abort_observation, dtype=np.int64)
    traces["cube_pos"][:] = [0.071, -0.027, 0.1]
    traces["ctrl"][:] = 0.0

    _, checks = _v3_stage_metrics(model, config, traces)
    assert checks["stable_grasp_acquired"] is True
    assert checks["grasp_acquisition_event_consistent"] is True
    assert checks["grasp_gate_counter_is_consistent"] is True
    assert checks["controller_state_sequence_is_consistent"] is False
    assert checks["controller_operation_events_are_consistent"] is False


def test_v14_true_freeze_extends_manipulation_into_saved_verify_slack() -> None:
    model, source = _model_config()
    config = copy.deepcopy(source)
    protocol = config["control_protocol"]
    protocol.update(
        {
            "settle_s": 0.001,
            "close_s": 0.001,
            "verify_timeout_s": 0.500,
            "stable_window_s": 0.001,
            "manipulate_s": 3.0,
            "min_hold_s": 0.100,
        }
    )
    phase = _protocol_steps(model, config)
    total = sum(phase.values())
    traces = _allocate_traces(model, total, schema_version=14)
    order = grasp_gate_order(14)
    traces["grasp_gate_order"] = np.asarray(order)
    traces["grasp_gate"] = np.ones((total, len(order)), dtype=bool)
    traces["target_face_effective"][:] = True
    traces["target_face_topology"][:] = True
    traces["target_face_force_purity"][:] = 1.0
    traces["tactile_max"][:] = 0.0
    traces["tactile_max"][:, :3] = 1.0
    for finger, name in enumerate(ACTIVE_FINGERS):
        face = FACE_ORDER.index(
            face_from_label(config["contact_topology"]["target_faces"][name])
        )
        traces["distal_face_force_n"][:, finger, face] = 1.0

    settle_end = phase["settle"]
    close_end = settle_end + phase["close"]
    acquisition = close_end
    manipulation_start = acquisition + 1
    freeze_count = 27
    manipulation_end = manipulation_start + phase["manipulate"] + freeze_count - 1
    traces["control_state"][:settle_end] = "SETTLE"
    traces["control_state"][settle_end:close_end] = "CLOSE"
    traces["control_state"][close_end:manipulation_start] = "VERIFY"
    traces["control_state"][manipulation_start : manipulation_end + 1] = (
        "MANIPULATE"
    )
    traces["control_state"][manipulation_end + 1 :] = "HOLD"
    traces["grasp_acquired"][acquisition:] = True
    traces["grasp_gate_consecutive_steps"][acquisition:] = 1
    traces["contact_progress_frozen"][
        manipulation_start : manipulation_start + freeze_count
    ] = True
    completed = 0
    for step in range(manipulation_start, manipulation_end + 1):
        if not traces["contact_progress_frozen"][step]:
            completed += 1
        traces["manipulation_progress"][step] = completed / phase["manipulate"]
    traces["manipulation_progress"][manipulation_end + 1 :] = 1.0
    traces["grasp_acquisition_step"] = np.asarray(acquisition, dtype=np.int64)
    traces["grasp_lock_step"] = np.asarray(acquisition, dtype=np.int64)
    traces["manipulation_start_step"] = np.asarray(
        manipulation_start, dtype=np.int64
    )
    traces["manipulation_end_step"] = np.asarray(
        manipulation_end, dtype=np.int64
    )
    traces["termination_step"] = np.asarray(total - 1, dtype=np.int64)
    traces["cube_pos"][:] = [0.071, -0.027, 0.1]
    traces["ctrl"][:] = 0.0

    _, checks = _v3_stage_metrics(model, config, traces)
    assert checks["controller_state_sequence_is_consistent"] is True
    assert checks["controller_operation_events_are_consistent"] is True
    assert checks["manipulation_progress_is_consistent"] is True
    assert checks["manipulation_completed"] is True
    assert checks["v14_manipulation_end_is_first_full_progress_sample"] is True
    assert checks["v14_manipulation_completed_within_saved_verify_slack"] is True
    assert checks["v14_minimum_hold_duration_preserved"] is True
