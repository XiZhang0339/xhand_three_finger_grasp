from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.contact_preserving_event_rescue import (
    ContactSwitchEvent,
    EventDetectionSettings,
    EventJacobianDirections,
    EventRescueBudget,
    EventRescueDescriptor,
    authenticate_event_rescue_job,
    build_event_rescue_jobs,
    build_event_rescue_refinement_jobs,
    compact_c2_event_bump,
    detect_contact_switch_events,
    solve_event_directions,
    trace_content_sha256,
)


def _config() -> dict:
    return load_config(
        "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
    )


def _synthetic_trace() -> dict[str, np.ndarray]:
    length = 500
    centroids = np.zeros((length, 3, 3), dtype=np.float64)
    # Valid face-local coordinates with two clusters.  The stronger jerk in
    # each cluster must select the later event despite its slightly smaller
    # geometric jump.
    centroids[150:, 0, 2] += 0.0006
    centroids[180:, 0, 1] += 0.0005
    # A near-identical cross-finger switch resolves to the same object jerk
    # peak and must not consume index's separate later-event quota.
    centroids[170:, 1, 2] += 0.00049
    centroids[310:, 1, 1] += 0.00045
    centroids[330:, 1, 2] += 0.0004
    valid = np.ones((length, 3), dtype=bool)
    effective = np.ones((length, 3), dtype=bool)
    # A large invalid jump must not be detected.
    valid[250, 0] = False
    centroids[250:, 0, 2] += 0.002
    jerk = np.zeros(length, dtype=np.float64)
    jerk[150] = 2.0
    jerk[180] = 7.0
    jerk[310] = 1.5
    jerk[330] = -5.0
    return {
        "finger_order": np.asarray(("thumb", "index", "mid")),
        "target_face_contact_centroid_cube_local_m": centroids,
        "target_face_contact_centroid_valid": valid,
        "target_face_effective": effective,
        "manipulation_progress": np.linspace(0.0, 1.0, length),
        "operation_vertical_jerk_filtered_m_s3": jerk,
        "manipulation_start_step": np.asarray(100),
        "manipulation_end_step": np.asarray(450),
    }


def _descriptor(config: dict) -> EventRescueDescriptor:
    event = ContactSwitchEvent(
        finger="thumb",
        finger_index=0,
        event_step=180,
        checkpoint_step=130,
        manipulation_progress=0.45,
        tangent_jump_cube_local_m=(0.0, 0.0, 0.0005),
        tangent_jump_m=0.0005,
        local_peak_abs_jerk_m_s3=7.0,
        event_id="",
    )
    event = replace(
        event,
        event_id=canonical_sha256(
            {
                "schema_version": 1,
                "finger": event.finger,
                "event_step": event.event_step,
                "checkpoint_step": event.checkpoint_step,
                "tangent_jump_cube_local_m": list(
                    event.tangent_jump_cube_local_m
                ),
            }
        ),
    )
    tangent = (0.25, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    unload = (0.1, -1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0)
    direction_payload = {
        "schema_version": 1,
        "event_id": event.event_id,
        "finger": "thumb",
        "target_face": "-X",
        "tangent_direction_rad_unit": list(tangent),
        "normal_unload_direction_rad_unit": list(unload),
        "tangent_residual": 0.01,
        "normal_residual": 0.02,
        "jacobian_rank": 3,
    }
    direction = EventJacobianDirections(
        event_id=event.event_id,
        finger="thumb",
        target_face="-X",
        tangent_direction_rad_unit=tangent,
        normal_unload_direction_rad_unit=unload,
        tangent_residual=0.01,
        normal_residual=0.02,
        jacobian_rank=3,
        directions_id=canonical_sha256(direction_payload),
    )
    settings = EventDetectionSettings().as_mapping()
    payload = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "source_candidate_id": 14_027_467_936_228_193,
        "source_config_semantic_sha256": canonical_sha256(config),
        "source_trace_sha256": "a" * 64,
        "source_trace_content_sha256": "b" * 64,
        "detection_settings": settings,
        "events": [event.as_mapping()],
        "directions": [direction.as_mapping()],
    }
    return EventRescueDescriptor(
        schema_version=1,
        experiment_id=config["experiment_id"],
        source_candidate_id=payload["source_candidate_id"],
        source_config_semantic_sha256=payload["source_config_semantic_sha256"],
        source_trace_sha256=payload["source_trace_sha256"],
        source_trace_content_sha256=payload["source_trace_content_sha256"],
        detection_settings=settings,
        events=(event,),
        directions=(direction,),
        descriptor_id=canonical_sha256(payload),
    )


def test_detector_clusters_switches_ranks_by_jerk_and_rejects_invalid_masks():
    events = detect_contact_switch_events(
        _config(),
        _synthetic_trace(),
        settings=EventDetectionSettings(
            cluster_window_steps=75, jerk_peak_radius_steps=10
        ),
    )
    assert [(value.finger, value.event_step) for value in events] == [
        ("thumb", 180),
        ("index", 330),
    ]
    assert events[0].checkpoint_step == 130
    assert events[0].local_peak_abs_jerk_m_s3 == 7.0
    assert events[1].local_peak_abs_jerk_m_s3 == 5.0


def test_event_direction_solver_is_finger_local_normalized_and_has_correct_sign():
    jacobian = np.zeros((3, 8), dtype=np.float64)
    jacobian[:, :3] = np.eye(3)
    solved = solve_event_directions(
        jacobian,
        np.eye(3),
        "-X",
        (0.0, 0.0, 0.001),
        (0, 1, 2),
    )
    tangent = np.asarray(solved["tangent_direction_rad_unit"])
    unload = np.asarray(solved["normal_unload_direction_rad_unit"])
    np.testing.assert_allclose(tangent, (0, 0, -1, 0, 0, 0, 0, 0))
    np.testing.assert_allclose(unload, (-1, 0, 0, 0, 0, 0, 0, 0))
    assert solved["jacobian_rank"] == 3


def test_compact_event_bump_has_zero_value_slope_and_curvature_at_support():
    center = 0.5
    width = 0.2
    spacing = 1e-5
    points = np.asarray(
        [center - width - spacing, center - width, center - width + spacing,
         center, center + width - spacing, center + width, center + width + spacing]
    )
    values = compact_c2_event_bump(points, center, width)
    assert values[0] == values[1] == values[5] == values[6] == 0.0
    assert values[3] == pytest.approx(1.0)
    # sin^4 grows as O(distance^4), hence both first and second derivatives
    # converge to zero at either compact-support boundary.
    assert values[2] / spacing**2 < 1e-3
    assert values[4] / spacing**2 < 1e-3


def test_descriptor_round_trip_and_tampering_fail_closed():
    descriptor = _descriptor(_config())
    assert EventRescueDescriptor.from_mapping(descriptor.as_mapping()) == descriptor
    tampered = descriptor.as_mapping()
    tampered["events"][0]["event_step"] += 1
    with pytest.raises(ValueError, match="identity"):
        EventRescueDescriptor.from_mapping(tampered)


def test_trace_content_digest_is_order_independent_and_covers_all_array_bytes():
    first = {
        "b": np.asarray([True, False]),
        "a": np.asarray([[1.0, 2.0]], dtype=np.float64),
    }
    reordered = {"a": first["a"].copy(), "b": first["b"].copy()}
    assert trace_content_sha256(first) == trace_content_sha256(reordered)
    changed = copy.deepcopy(reordered)
    changed["a"][0, 1] += 1e-12
    assert trace_content_sha256(first) != trace_content_sha256(changed)


def test_exploration_jobs_are_deterministic_exact_parent_first_and_safe():
    config = _config()
    descriptor = _descriptor(config)
    budget = EventRescueBudget(total_candidate_count=12)
    first = build_event_rescue_jobs(
        config, descriptor, budget=budget, validate_configs=False
    )
    second = build_event_rescue_jobs(
        copy.deepcopy(config), descriptor, budget=budget, validate_configs=False
    )
    assert [value["candidate_id"] for value in first] == [
        value["candidate_id"] for value in second
    ]
    assert len({value["candidate_id"] for value in first}) == 12
    assert first[0]["parameters"]["time_warp_a1"] == 0.0
    assert first[0]["parameters"]["time_warp_a2"] == 0.0
    assert first[0]["parameters"]["terminal_scale"] == 1.0
    assert first[0]["config"]["manipulation_plan"]["actuator_waypoints_rad"] == config["manipulation_plan"]["actuator_waypoints_rad"]
    for job in first:
        plan = job["config"]["manipulation_plan"]
        assert len(plan["knot_times_s"]) == 21
        assert plan["knot_times_s"][0] == 0.0
        assert plan["knot_times_s"][-1] == 3.0
        for name in ACTIVE_ACTUATORS:
            values = np.asarray(plan["actuator_waypoints_rad"][name])
            assert float(np.max(np.abs(np.diff(values)), initial=0.0)) <= 0.04 + 1e-12


def test_job_authentication_rejects_config_and_parameter_tampering():
    config = _config()
    descriptor = _descriptor(config)
    job = build_event_rescue_jobs(
        config,
        descriptor,
        budget=EventRescueBudget(total_candidate_count=2),
        validate_configs=True,
    )[0]
    authenticate_event_rescue_job(job, descriptor)
    changed_config = copy.deepcopy(job)
    changed_config["config"]["control"]["manipulation_delta_rad"][ACTIVE_ACTUATORS[0]] += 1e-6
    with pytest.raises(ValueError, match="config authentication"):
        authenticate_event_rescue_job(changed_config, descriptor)
    changed_parameter = copy.deepcopy(job)
    changed_parameter["parameters"]["terminal_scale"] = 1.001
    with pytest.raises(ValueError, match="payload authentication"):
        authenticate_event_rescue_job(changed_parameter, descriptor)


def test_refinement_jobs_use_separate_stage_identity_and_are_repeatable():
    config = _config()
    descriptor = _descriptor(config)
    exploration = build_event_rescue_jobs(
        config,
        descriptor,
        budget=EventRescueBudget(total_candidate_count=3),
        validate_configs=False,
    )
    budget = EventRescueBudget(
        stage="local_refinement", total_candidate_count=5
    )
    refined = build_event_rescue_refinement_jobs(
        config,
        descriptor,
        [exploration[1]["parameters"]],
        budget=budget,
        validate_configs=False,
    )
    repeated = build_event_rescue_refinement_jobs(
        config,
        descriptor,
        [exploration[1]["parameters"]],
        budget=budget,
        validate_configs=False,
    )
    assert all(value["stage"] == "local_refinement" for value in refined)
    assert [value["candidate_id"] for value in refined] == [
        value["candidate_id"] for value in repeated
    ]
    assert not set(value["candidate_id"] for value in refined) & set(
        value["candidate_id"] for value in exploration
    )
