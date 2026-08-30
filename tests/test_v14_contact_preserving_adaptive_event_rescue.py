from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS
from xhand_grasp.experiment import ManipulationPlanParameters
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning import contact_preserving_adaptive_event_rescue as adaptive
from xhand_grasp.tuning.contact_preserving_event_rescue import EventJacobianDirections


ROOT = Path(__file__).resolve().parents[1]
PARENT_CONFIG = (
    ROOT
    / "artifacts/left_opposed_face_palm_down_contact_preserving_planned_lift"
    / "tune/formal_campaign_v14_1_event_rescue_v1/catalog_source_reruns"
    / "rescue_target_1/candidate_14047225395912675/resolved_config.json"
)


def _event(*, finger: str = "thumb", step: int = 80, progress: float = 0.5):
    index = ("thumb", "index", "mid").index(finger)
    tangent = (0.0, 0.0003, 0.0)
    payload = {
        "schema_version": adaptive.ADAPTIVE_EVENT_SCHEMA_VERSION,
        "finger": finger,
        "event_step": step,
        "checkpoint_step": step - 50,
        "tangent_jump_cube_local_m": list(tangent),
        "taxel_count_transition": [1, 2],
        "force_jump_n": 0.2,
        "associated_jerk_peak_step": step + 20,
    }
    return adaptive.AdaptiveContactEvent(
        finger=finger,
        finger_index=index,
        event_step=step,
        checkpoint_step=step - 50,
        manipulation_progress=progress,
        tangent_jump_cube_local_m=tangent,
        tangent_jump_m=0.0003,
        taxel_count_before=1,
        taxel_count_after=2,
        target_force_before_n=0.5,
        target_force_after_n=0.7,
        force_jump_n=0.2,
        force_load_transfer_n=(0.2, -0.1, -0.1),
        associated_jerk_peak_step=step + 20,
        associated_peak_abs_jerk_m_s3=4.0,
        association_radius_steps=30,
        event_id=canonical_sha256(payload),
    )


def _direction(event: adaptive.AdaptiveContactEvent):
    tangent = np.zeros(len(ACTIVE_ACTUATORS))
    unload = np.zeros(len(ACTIVE_ACTUATORS))
    tangent[0] = 1.0
    unload[1] = 1.0
    return EventJacobianDirections(
        event_id=event.event_id,
        finger=event.finger,
        target_face="-X",
        tangent_direction_rad_unit=tuple(tangent),
        normal_unload_direction_rad_unit=tuple(unload),
        tangent_residual=0.0,
        normal_residual=0.0,
        jacobian_rank=2,
        directions_id="a" * 64,
    )


def _descriptor(config, center_id="b" * 64):
    event = _event()
    base = adaptive.AdaptiveEventDescriptor(
        schema_version=adaptive.ADAPTIVE_EVENT_SCHEMA_VERSION,
        experiment_id=adaptive.EXPERIMENT_ID,
        center_id=center_id,
        source_candidate_id=123,
        source_config_semantic_sha256=canonical_sha256(config),
        source_trace_sha256="c" * 64,
        source_trace_content_sha256="d" * 64,
        manipulation_start_step=100,
        manipulation_end_step=3100,
        detection_settings=adaptive.AdaptiveEventDetectionSettings().as_mapping(),
        events=(event,),
        directions=(_direction(event),),
        descriptor_id="0" * 64,
    )
    return replace(
        base,
        descriptor_id=canonical_sha256(adaptive._descriptor_identity_payload(base)),
    )


def _center(config, descriptor):
    summary = {
        "passed": False,
        "failed_checks": ["smooth_motion_jerk_within_limit"],
        "stage_status": {"grasp_success": True, "full_success": False},
        "metrics": {
            "motion_smoothness": {
                "operation_peak_abs_filtered_jerk_m_s3": 7.0
            }
        },
    }
    return adaptive.AdaptiveEventCenter(
        candidate_id=123,
        source_stage="test",
        source_authentication_id="e" * 64,
        source_record_sha256="f" * 64,
        config=copy.deepcopy(config),
        result={"summary": summary},
        summary=summary,
        config_path=Path("/tmp/unused-config"),
        result_path=Path("/tmp/unused-result"),
        trace_path=Path("/tmp/unused-trace"),
        config_semantic_sha256=canonical_sha256(config),
        result_semantic_sha256="1" * 64,
        trace_sha256="c" * 64,
        trace_content_sha256="d" * 64,
        physical_plan_sha256=adaptive.physical_plan_sha256(config),
        center_id=descriptor.center_id,
    )


def _load_parent_config():
    return json.loads(PARENT_CONFIG.read_text(encoding="utf-8"))


def test_physical_hash_ignores_ids_but_detects_waypoint_change():
    config = _load_parent_config()
    changed = copy.deepcopy(config)
    changed["planner_id"] = "x"
    changed["controller_id"] = "y"
    changed["candidate_metadata"] = {"arbitrary": 1}
    assert adaptive.physical_plan_sha256(changed) == adaptive.physical_plan_sha256(config)
    changed["manipulation_plan"]["actuator_waypoints_rad"][ACTIVE_ACTUATORS[0]][5] += 1e-6
    assert adaptive.physical_plan_sha256(changed) != adaptive.physical_plan_sha256(config)


def test_detector_uses_witness_taxel_force_and_centered_jerk_association():
    length = 200
    trace = {
        "finger_order": np.asarray(["thumb", "index", "mid"]),
        "target_face_contact_centroid_cube_local_m": np.zeros((length, 3, 3)),
        "target_face_contact_centroid_valid": np.ones((length, 3), dtype=bool),
        "target_face_effective": np.ones((length, 3), dtype=bool),
        "distal_active_taxel_count": np.ones((length, 3), dtype=np.int64),
        "contact_force_filtered_n": np.full((length, 3), 0.5),
        "manipulation_progress": np.linspace(0.0, 1.0, length),
        "operation_vertical_jerk_filtered_m_s3": np.zeros(length),
        "manipulation_start_step": np.asarray(20),
        "manipulation_end_step": np.asarray(180),
    }
    trace["target_face_contact_centroid_cube_local_m"][80:, 0, 1] = 0.0003
    trace["distal_active_taxel_count"][110:, 1] = 2
    trace["contact_force_filtered_n"][110:, 1] = 0.72
    trace["operation_vertical_jerk_filtered_m_s3"][105] = -4.5
    config = {
        "schema_version": 14,
        "experiment_id": adaptive.EXPERIMENT_ID,
        "contact_topology": {
            "target_faces": {"thumb": "-X", "index": "+X", "mid": "+X"}
        },
    }
    events = adaptive.detect_adaptive_contact_events(config, trace)
    assert {(value.finger, value.event_step) for value in events} == {
        ("thumb", 80),
        ("index", 110),
    }
    assert all(value.associated_jerk_peak_step == 105 for value in events)
    assert next(value for value in events if value.finger == "index").taxel_count_after == 2


def test_polytope_projected_jobs_are_feasible_unique_and_authentic():
    config = _load_parent_config()
    descriptor = _descriptor(config)
    center = _center(config, descriptor)
    polytope = adaptive.derive_feasible_event_polytope(config, descriptor)
    assert polytope.contains(np.zeros(polytope.dimension))
    outside = np.asarray(polytope.upper_bounds) + 1.0
    assert polytope.contains(polytope.project(outside), tolerance=1e-8)
    budget = adaptive.AdaptiveEventBudget(
        stage="diagnostic",
        total_candidate_count=4,
        candidates_per_center=4,
        max_center_count=1,
        resampling_multiplier=16,
    )
    first = adaptive.build_adaptive_event_jobs(center, descriptor, polytope, budget=budget)
    second = adaptive.build_adaptive_event_jobs(center, descriptor, polytope, budget=budget)
    assert first == second
    assert len(first) == 4
    assert len({value["physical_plan_sha256"] for value in first}) == 4
    assert center.physical_plan_sha256 not in {
        value["physical_plan_sha256"] for value in first
    }
    for job in first:
        plan = ManipulationPlanParameters.from_config(
            job["config"]["manipulation_plan"]
        )
        assert max(
            abs(following - previous)
            for values in plan.actuator_waypoints_rad.values()
            for previous, following in zip(values, values[1:])
        ) <= plan.max_knot_delta_rad
    adaptive.authenticate_adaptive_event_job(first[0], center, descriptor, polytope)
    tampered = copy.deepcopy(first[0])
    tampered["projected_parameters"][polytope.parameter_names[0]] += 1e-4
    with pytest.raises(RuntimeError, match="payload SHA-256"):
        adaptive.authenticate_adaptive_event_job(tampered, center, descriptor, polytope)

    def rebind(job):
        core = {key: value for key, value in job.items() if key not in {"config", "candidate_payload_sha256"}}
        job["candidate_payload_sha256"] = canonical_sha256(core)

    tampered_id = copy.deepcopy(first[0])
    tampered_id["candidate_id"] += 1
    rebind(tampered_id)
    with pytest.raises(RuntimeError, match="candidate ID"):
        adaptive.authenticate_adaptive_event_job(
            tampered_id, center, descriptor, polytope
        )

    tampered_source = copy.deepcopy(first[0])
    tampered_source["source_physical_plan_sha256"] = "9" * 64
    rebind(tampered_source)
    with pytest.raises(RuntimeError, match="source physical provenance"):
        adaptive.authenticate_adaptive_event_job(
            tampered_source, center, descriptor, polytope
        )

    tampered_budget = copy.deepcopy(first[0])
    tampered_budget["budget"]["seed"] += 1
    rebind(tampered_budget)
    with pytest.raises(RuntimeError, match="sampling seed"):
        adaptive.authenticate_adaptive_event_job(
            tampered_budget, center, descriptor, polytope
        )

    tampered_request = copy.deepcopy(first[0])
    first_name = polytope.parameter_names[0]
    tampered_request["requested_parameters"][first_name] += 1e-5
    rebind(tampered_request)
    with pytest.raises(RuntimeError, match="request is not reproducible"):
        adaptive.authenticate_adaptive_event_job(
            tampered_request, center, descriptor, polytope
        )

    excluded = tuple(value["physical_plan_sha256"] for value in first)
    replacement = adaptive.build_adaptive_event_jobs(
        center,
        descriptor,
        polytope,
        budget=budget,
        excluded_physical_plan_sha256=excluded,
    )
    assert not set(excluded).intersection(
        value["physical_plan_sha256"] for value in replacement
    )
    adaptive.authenticate_adaptive_event_job(
        replacement[0],
        center,
        descriptor,
        polytope,
        expected_excluded_physical_plan_sha256=excluded,
    )
    with pytest.raises(RuntimeError, match="exclusion-set provenance"):
        adaptive.authenticate_adaptive_event_job(
            replacement[0],
            center,
            descriptor,
            polytope,
            expected_excluded_physical_plan_sha256=("8" * 64,),
        )


def test_rank_is_nonjerk_then_jerk_then_contact():
    def record(jerk, *, failures, duty=1.0, candidate=1):
        return {
            "candidate_id": candidate,
            "full_success": False,
            "summary": {
                "failed_checks": failures,
                "metrics": {
                    "operation_target_face_contact_duty": {
                        "thumb": duty,
                        "index": duty,
                        "mid": duty,
                    },
                    "contact_preserving_planned_lift": {
                        "longest_contact_loss_steps": {
                            "thumb": 0,
                            "index": 0,
                            "mid": 0,
                        }
                    },
                    "operation_minimum_lift_m": 0.008,
                    "operation_median_lift_m": 0.010,
                    "actuator_saturation_fraction": 0.0,
                    "motion_smoothness": {
                        "operation_peak_abs_filtered_jerk_m_s3": jerk,
                        "operation_max_lateral_displacement_m": 0.001,
                        "operation_max_orientation_drift_deg": 1.0,
                        "operation_cumulative_height_backtrack_m": 0.0,
                        "operation_peak_filtered_upward_speed_m_s": 0.01,
                        "operation_peak_abs_filtered_acceleration_m_s2": 0.05,
                        "operation_hold_entry_linear_speed_m_s": 0.001,
                    },
                },
            },
        }

    jerk_only_low = record(5.0, failures=["smooth_motion_jerk_within_limit"], candidate=1)
    jerk_only_high = record(8.0, failures=["smooth_motion_jerk_within_limit"], duty=1.0, candidate=2)
    nonjerk = record(1.0, failures=["smooth_motion_jerk_within_limit", "lost_contact"], candidate=3)
    assert adaptive.adaptive_event_candidate_rank(jerk_only_low) < adaptive.adaptive_event_candidate_rank(jerk_only_high)
    assert adaptive.adaptive_event_candidate_rank(jerk_only_high) < adaptive.adaptive_event_candidate_rank(nonjerk)
