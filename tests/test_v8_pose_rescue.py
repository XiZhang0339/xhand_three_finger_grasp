from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

import xhand_grasp.tuning.normal_aligned_smooth_lift as tuning
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.tuning.normal_aligned_smooth_lift import (
    ACTIVE_ACTUATOR_POSITION_LIMITS_RAD,
    DEFAULT_V7_CAMPAIGN_RESULTS,
    DEFAULT_V8_TEMPLATE,
    EXPERIMENT_ID,
    FIXED_RESCUE_BUDGET,
    build_pose_rescue_manifest,
    build_parser,
    candidate_rank_evidence,
    controller_id,
    fit_manipulation_response_model,
    generate_lift_probe_candidates,
    generate_lift_trust_candidates,
    generate_pose_coarse_candidates,
    lift_candidate_evidence,
    manipulation_delta_bounds,
    manipulation_response_from_trace,
    materialize_lift_candidate,
    materialize_v8_rescue_candidate,
    pose_id,
    rank_rescue_results,
    validated_67_control_spec,
)
from xhand_grasp.scene import build_model
from xhand_grasp.trajectory import actuator_target_vector


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_RESULTS = ROOT / DEFAULT_V7_CAMPAIGN_RESULTS


@pytest.fixture(scope="module")
def pose_manifest() -> dict:
    if not CAMPAIGN_RESULTS.is_file():
        pytest.skip("the completed v7 campaign artifacts are not available")
    return build_pose_rescue_manifest(CAMPAIGN_RESULTS)


@pytest.fixture(scope="module")
def source_and_record(pose_manifest):
    record = pose_manifest["poses"][0]
    return load_config(record["source_config"]), record


def synthetic_v8_template(source: dict) -> dict:
    template = copy.deepcopy(source)
    template["schema_version"] = 8
    template["experiment_id"] = EXPERIMENT_ID
    template["description"] = "synthetic v8 unit-test template"
    template["control_protocol"]["manipulation_profile"] = "minimum_jerk_quintic"
    template["closure_alignment"] = {
        "static_max_angle_deg": 45.0,
        "dynamic_p95_max_angle_deg": 30.0,
        "optimization_target_max_angle_deg": 20.0,
        "require_positive_inward_speed": True,
    }
    template["motion_smoothness"] = {
        "filter_window_s": 0.051,
        "max_lateral_displacement_m": 0.002,
    }
    return template


def test_fixed_rescue_budget_is_exactly_1616():
    assert FIXED_RESCUE_BUDGET.tier_a_total == 1_072
    assert FIXED_RESCUE_BUDGET.tier_b_total == 544
    assert FIXED_RESCUE_BUDGET.total == 1_616


def test_manipulation_delta_bounds_respect_compiled_absolute_limits():
    config = load_config(ROOT / DEFAULT_V8_TEMPLATE)
    index_bend = "left_hand_index_bend_joint_actuator"
    config["control"]["grasp_targets_rad"][index_bend] = 0.08

    registered = manipulation_delta_bounds(config)
    bounds = tuning._ctrlrange_safe_manipulation_delta_bounds(config)
    assert registered[index_bend] == pytest.approx((-0.05, 0.10))
    assert bounds[index_bend] == pytest.approx((-0.05, 0.094))

    model, _ = build_model(config)
    for name in ACTIVE_ACTUATORS:
        actuator_id = model.actuator(name).id
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        expected = np.asarray(ACTIVE_ACTUATOR_POSITION_LIMITS_RAD[name])
        assert model.actuator_ctrlrange[actuator_id] == pytest.approx(expected)
        assert model.jnt_range[joint_id] == pytest.approx(expected)
        target = float(config["control"]["grasp_targets_rad"][name])
        assert target + bounds[name][0] >= expected[0] - 1e-12
        assert target + bounds[name][1] <= expected[1] + 1e-12

    config["control"]["grasp_targets_rad"][index_bend] = 0.175
    with pytest.raises(ValueError, match="outside model position limits"):
        tuning._ctrlrange_safe_manipulation_delta_bounds(config)


def test_lift_materialization_clips_absolute_target_to_ctrlrange():
    config = load_config(ROOT / DEFAULT_V8_TEMPLATE)
    index_bend = "left_hand_index_bend_joint_actuator"
    config["control"]["grasp_targets_rad"][index_bend] = 0.08
    delta = {name: 0.0 for name in ACTIVE_ACTUATORS}
    delta[index_bend] = 0.10

    candidate = materialize_lift_candidate(
        config,
        {"pose_id": pose_id(config), "source_candidate_id": 1},
        delta,
        candidate_id_value=2,
        stage="lift_trust",
        validator=None,
    )
    clipped = candidate["control"]["manipulation_delta_rad"]
    assert clipped[index_bend] == pytest.approx(0.094)

    absolute_targets = {
        name: float(candidate["control"]["grasp_targets_rad"][name])
        + float(clipped[name])
        for name in ACTIVE_ACTUATORS
    }
    model, _ = build_model(config)
    actuator_target_vector(model, absolute_targets)
    assert absolute_targets[index_bend] == pytest.approx(0.174)


def test_completed_v7_campaign_resolves_the_21_independent_poses(pose_manifest):
    assert pose_manifest["selection"]["raw_eligible_count"] == 60
    assert pose_manifest["selection"]["deduplicated_count"] == 21
    assert pose_manifest["selection"]["tier_counts"] == {"A": 11, "B": 10}
    assert len({record["pose_id"] for record in pose_manifest["poses"]}) == 21
    assert [record["source_candidate_id"] for record in pose_manifest["poses"]] == [
        73000012000025,
        72000006000050,
        73000014000028,
        72000006000048,
        73000014000029,
        73000009000018,
        73000011000022,
        73000007000014,
        72000006000033,
        73000006000013,
        72000002000023,
        72000006000027,
        72000002000026,
        73000008000017,
        73000010000021,
        72000006000054,
        72000008000052,
        71000026011726,
        72000008000000,
        72000010000025,
        73000013000026,
    ]
    assert max(record["thumb_target_rad"] for record in pose_manifest["poses"]) == 1.30


def test_v8_materialization_preserves_pose_but_separates_controller_identity(
    source_and_record,
):
    source, record = source_and_record
    template = synthetic_v8_template(source)
    original_pose_id = pose_id(source)
    candidate = materialize_v8_rescue_candidate(
        source,
        template,
        record,
        validated_67_control_spec(source),
        candidate_id_value=81,
        stage="test",
        validator=None,
    )
    assert candidate["schema_version"] == 8
    assert candidate["experiment_id"] == EXPERIMENT_ID
    assert pose_id(candidate) == original_pose_id
    assert candidate["candidate_metadata"]["pose_id"] == original_pose_id
    assert candidate["candidate_metadata"]["controller_id"] == controller_id(candidate)
    assert candidate["candidate_metadata"]["controller_id"] != original_pose_id
    for field in ("side", "scene", "hand_pose", "cube", "contact_topology"):
        assert candidate[field] == source[field]
    assert set(candidate["control"]["manipulation_delta_rad"].values()) == {0.0}
    assert candidate["control"]["grasp_targets_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] == source["control"]["grasp_targets_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ]


def test_coarse_controls_are_deterministic_and_include_all_anchor_families(
    source_and_record,
):
    source, record = source_and_record
    template = synthetic_v8_template(source)
    first = generate_pose_coarse_candidates(
        source,
        template,
        record,
        pose_index=0,
        count=16,
        seed=20260821,
        validator=None,
    )
    second = generate_pose_coarse_candidates(
        source,
        template,
        record,
        pose_index=0,
        count=16,
        seed=20260821,
        validator=None,
    )
    assert first == second
    kinds = [
        candidate["config"]["candidate_metadata"]["rescue_control_spec"][
            "anchor_kind"
        ]
        for candidate in first
    ]
    assert kinds[:3] == [
        "original_control",
        "validated_67_control",
        "analytic_contact_synchronization",
    ]
    assert "latin_hypercube" in kinds
    assert len({candidate["candidate_id"] for candidate in first}) == len(first)
    assert all(candidate["pose_id"] == record["pose_id"] for candidate in first)
    assert all(
        set(candidate["config"]["control"]["pregrasp_targets_rad"])
        == set(ACTIVE_ACTUATORS)
        for candidate in first
    )


def _rank_record(
    candidate_id_value: int,
    *,
    evidence: bool,
    angle: float,
    motion_passed: bool = True,
) -> dict:
    metrics = {
        "verify_max_consecutive_all_gate_steps": 250,
        "pose_preservation": {
            "max_translation_m": 0.0001,
            "max_orientation_drift_deg": 0.2,
        },
        "peak_total_distal_contact_force_n": 5.0,
        "actuator_saturation_fraction": 0.1,
    }
    if evidence:
        metrics.update(
            {
                "closure_alignment": {
                    "worst_p95_angle_deg": angle,
                    "per_finger": {
                        finger: {
                            "angle_p95_deg": angle,
                            "angle_max_deg": angle + 1.0,
                            "minimum_inward_speed_m_s": 0.002,
                        }
                        for finger in ("thumb", "index", "mid")
                    },
                },
                "motion_smoothness": {
                    "operation_max_lateral_displacement_m": 0.001,
                    "operation_max_orientation_drift_deg": 2.0,
                    "operation_cumulative_height_backtrack_m": 0.0001,
                    "operation_downward_speed_duty": 0.01,
                    "operation_peak_filtered_upward_speed_m_s": 0.010,
                    "operation_peak_abs_filtered_acceleration_m_s2": 0.06,
                    "operation_peak_abs_filtered_jerk_m_s3": (
                        1.0 if motion_passed else 5.0
                    ),
                    "operation_hold_entry_linear_speed_m_s": 0.002,
                },
            }
        )
    checks = {}
    if evidence:
        checks.update(
            {
                "v8_closure_alignment_trace_matches_vectors": True,
                "closure_alignment_valid_for_all_fingers": True,
                "closure_alignment_p95_within_limit": angle <= 30.0,
                "closure_inward_speed_positive": True,
                "smooth_motion_event_sequence_valid": motion_passed,
                "smooth_motion_filter_window_available": motion_passed,
                "smooth_motion_cumulative_backtrack_within_limit": motion_passed,
                "smooth_motion_downward_speed_duty_within_limit": motion_passed,
                "smooth_motion_peak_upward_speed_within_limit": motion_passed,
                "smooth_motion_acceleration_within_limit": motion_passed,
                "smooth_motion_jerk_within_limit": motion_passed,
                "smooth_motion_hold_entry_speed_within_limit": motion_passed,
                "smooth_motion_lateral_displacement_within_limit": motion_passed,
                "smooth_motion_orientation_drift_within_limit": motion_passed,
            }
        )
    return {
        "candidate_id": candidate_id_value,
        "acquisition_success": True,
        "pose_preservation_success": True,
        "summary": {"metrics": metrics, "checks": checks},
    }


def _lift_result(
    candidate_id_value: int,
    *,
    stage: str,
    pose_id_value: str = "pose-a",
    median_lift_m: float,
    minimum_lift_m: float,
    failed_check_count: int,
    response_available: bool = True,
    full_success: bool = False,
) -> dict:
    result = _rank_record(candidate_id_value, evidence=True, angle=18.0)
    result.update(
        {
            "stage": stage,
            "pose_id": pose_id_value,
            "controller_id": f"controller-{candidate_id_value}",
            "candidate_sha256": f"candidate-{candidate_id_value}",
            "source_candidate_id": -1,
            "edge_m": 0.067,
            "thumb_target_rad": 1.30,
            "artifact_directory": f"candidates/candidate_{candidate_id_value}",
            "rescue_success": False,
        }
    )
    result["summary"]["metrics"].update(
        {
            "operation_median_lift_m": median_lift_m,
            "operation_minimum_lift_m": minimum_lift_m,
        }
    )
    result["summary"].update(
        {
            "passed": full_success,
            "failed_checks": [
                f"synthetic_failed_check_{index}"
                for index in range(failed_check_count)
            ],
            "stage_status": {
                "grasp_success": True,
                "manipulation_success": full_success,
                "full_success": full_success,
            },
        }
    )
    result["manipulation_response"] = (
        {
            "available": True,
            "response_6d": [0.0, 0.0, median_lift_m, 0.0, 0.0, 0.0],
            "lateral_displacement_m": 0.0,
            "orientation_change_deg": 0.0,
        }
        if response_available
        else {"available": False}
    )
    evidence = lift_candidate_evidence(result)
    result["lift_success"] = evidence["lift_success"]
    result["lift_rank_evidence"] = evidence
    result["classification"] = tuning._candidate_classification(
        stage,
        rescue_success=False,
        lift_success=bool(evidence["lift_success"]),
    )
    return result


def test_missing_v8_metric_evidence_is_demoted_to_near_miss():
    missing = _rank_record(1, evidence=False, angle=0.0)
    aligned = _rank_record(2, evidence=True, angle=22.0)
    better = _rank_record(3, evidence=True, angle=18.0)
    assert candidate_rank_evidence(missing)["rescue_success"] is False
    assert candidate_rank_evidence(aligned)["rescue_success"] is True
    assert [record["candidate_id"] for record in rank_rescue_results((missing, aligned, better))] == [
        3,
        2,
        1,
    ]


def test_rescue_does_not_require_zero_operation_motion_smoothness():
    jittery_zero_delta = _rank_record(
        4, evidence=True, angle=18.0, motion_passed=False
    )
    evidence = candidate_rank_evidence(jittery_zero_delta)
    assert evidence["motion_smoothness_passed"] is False
    assert evidence["evidence_complete"] is True
    assert evidence["rescue_success"] is True


def _v8_grasp_candidate(source: dict, record: dict) -> dict:
    return materialize_v8_rescue_candidate(
        source,
        load_config(ROOT / DEFAULT_V8_TEMPLATE),
        record,
        validated_67_control_spec(source),
        candidate_id_value=81,
        stage="test",
        validator=None,
    )


def test_lift_probe_plan_is_exactly_zero_plus_actuator_signs(source_and_record):
    source, record = source_and_record
    base = _v8_grasp_candidate(source, record)
    first = generate_lift_probe_candidates(
        base, pose_index=2, epsilon_rad=0.02, validator=None
    )
    second = generate_lift_probe_candidates(
        base, pose_index=2, epsilon_rad=0.02, validator=None
    )
    assert first == second
    assert len(first) == 17
    metadata = [item["config"]["candidate_metadata"]["lift_search"] for item in first]
    assert metadata[0]["kind"] == "zero"
    assert {(value["actuator"], value["direction"]) for value in metadata[1:]} == {
        (name, direction) for name in ACTIVE_ACTUATORS for direction in (-1, 1)
    }
    assert all(item["pose_id"] == pose_id(base) for item in first)


def test_trace_response_and_response_fit_are_deterministic(source_and_record):
    trace = {
        "cube_pos": np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.001, -0.002, 0.011]]
        ),
        "cube_quat": np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        "manipulation_start_step": np.asarray(1),
        "manipulation_end_step": np.asarray(2),
    }
    response = manipulation_response_from_trace(trace)
    assert response["available"] is True
    assert response["response_6d"] == pytest.approx(
        [0.001, -0.002, 0.011, 0.0, 0.0, 0.0]
    )

    source, record = source_and_record
    base = _v8_grasp_candidate(source, record)
    probes = list(generate_lift_probe_candidates(base, pose_index=0, validator=None))
    jacobian = np.arange(48, dtype=np.float64).reshape(6, 8) * 1e-3
    bias = np.asarray([0.0, 0.0, 0.001, 0.0, 0.0, 0.0])
    for probe in probes:
        delta = np.asarray(
            [probe["config"]["control"]["manipulation_delta_rad"][name] for name in ACTIVE_ACTUATORS]
        )
        probe["manipulation_response"] = {
            "available": True,
            "response_6d": (bias + jacobian @ delta).tolist(),
        }
    bounds = manipulation_delta_bounds(base)
    model = fit_manipulation_response_model(probes, bounds)
    assert model["zero_response_available"] is True
    assert model["available_column_count"] == 8
    assert np.asarray(model["jacobian_6x8"]) == pytest.approx(jacobian)
    for name, value in model["solution_delta_rad"].items():
        assert bounds[name][0] <= value <= bounds[name][1]
    trust_a = generate_lift_trust_candidates(
        base, record, model, pose_index=0, count=16, seed=20260821, validator=None
    )
    trust_b = generate_lift_trust_candidates(
        base, record, model, pose_index=0, count=16, seed=20260821, validator=None
    )
    assert trust_a == trust_b
    assert len(trust_a) == 16


def test_lift_requires_motion_and_full_trace_evidence():
    result = _rank_record(5, evidence=True, angle=18.0)
    result["summary"]["metrics"].update(
        {"operation_median_lift_m": 0.011, "operation_minimum_lift_m": 0.009}
    )
    result["summary"]["stage_status"] = {
        "grasp_success": True,
        "manipulation_success": True,
        "full_success": True,
    }
    result["summary"]["passed"] = True
    result["summary"]["failed_checks"] = []
    result["manipulation_response"] = {
        "available": True,
        "response_6d": [0.0, 0.0, 0.011, 0.0, 0.0, 0.0],
        "lateral_displacement_m": 0.0,
        "orientation_change_deg": 0.0,
    }
    assert lift_candidate_evidence(result)["lift_success"] is True
    result["summary"]["checks"]["smooth_motion_jerk_within_limit"] = False
    assert lift_candidate_evidence(result)["lift_success"] is False


def test_lift_rank_prioritizes_threshold_progress_before_failed_check_count():
    closer = _lift_result(
        87_000_000_000_001,
        stage="lift_trust",
        median_lift_m=0.006533077504739845,
        minimum_lift_m=0.00624240620480189,
        failed_check_count=23,
    )
    almost_zero = _lift_result(
        87_000_000_000_002,
        stage="lift_trust",
        median_lift_m=0.00005638431217497991,
        minimum_lift_m=0.00005541313399178016,
        failed_check_count=7,
    )
    incomplete_higher_lift = _lift_result(
        87_000_000_000_028,
        stage="lift_trust",
        median_lift_m=0.007694138443240489,
        minimum_lift_m=0.0064104479543094645,
        failed_check_count=32,
        response_available=False,
    )

    ranked = tuning.rank_lift_results(
        (almost_zero, incomplete_higher_lift, closer)
    )

    assert [value["candidate_id"] for value in ranked] == [
        closer["candidate_id"],
        almost_zero["candidate_id"],
        incomplete_higher_lift["candidate_id"],
    ]
    assert lift_candidate_evidence(closer)["evidence_complete"] is True
    assert (
        lift_candidate_evidence(incomplete_higher_lift)["evidence_complete"]
        is False
    )


def test_only_exact_lift_success_is_classified_as_validated():
    assert tuning._candidate_classification(
        "lift_probe", rescue_success=False, lift_success=True
    ) == "normal_aligned_smooth_vertical_lift_search_pass"
    assert tuning._candidate_classification(
        "lift_trust", rescue_success=False, lift_success=True
    ) == "normal_aligned_smooth_vertical_lift_search_pass"
    assert tuning._candidate_classification(
        "lift_refine", rescue_success=False, lift_success=True
    ) == "normal_aligned_smooth_vertical_lift_search_pass"
    assert tuning._candidate_classification(
        "lift_exact", rescue_success=False, lift_success=True
    ) == "validated_normal_aligned_smooth_vertical_lift"
    assert tuning._candidate_classification(
        "lift_exact", rescue_success=False, lift_success=False
    ) == "normal_aligned_smooth_vertical_lift_near_miss"


def test_exact_lift_replay_preserves_pose_controller_and_locked_provenance(
    source_and_record,
):
    source, record = source_and_record
    base = _v8_grasp_candidate(source, record)
    response_model = {
        "available": False,
        "solution_delta_rad": {name: 0.0 for name in ACTIVE_ACTUATORS},
    }
    parent = tuning.generate_lift_trust_candidates(
        base,
        record,
        response_model,
        pose_index=0,
        count=1,
        validator=None,
    )[0]

    exact = tuning.generate_lift_exact_candidates(
        (parent,), {str(parent["pose_id"]): record}, count=1, validator=None
    )[0]
    metadata = exact["config"]["candidate_metadata"]

    assert exact["stage"] == "lift_exact"
    assert exact["pose_id"] == parent["pose_id"]
    assert exact["controller_id"] == parent["controller_id"]
    assert metadata["parent_candidate_id"] == parent["candidate_id"]
    assert metadata["stage"] == "lift_exact"
    assert metadata["locked_timestep_s"] == 0.001
    tuning._require_exact_lift_candidate(exact)


def test_lift_catalog_rejects_non_exact_candidates_before_writing(
    tmp_path, source_and_record
):
    source, record = source_and_record
    base = _v8_grasp_candidate(source, record)
    parent = tuning.generate_lift_trust_candidates(
        base,
        record,
        {
            "available": False,
            "solution_delta_rad": {name: 0.0 for name in ACTIVE_ACTUATORS},
        },
        pose_index=0,
        count=1,
        validator=None,
    )[0]
    output = tmp_path / "lift"

    with pytest.raises(ValueError, match="only lift_exact"):
        tuning.publish_lift_trajectory_catalog(output, (parent,), maximum_count=1)

    assert not output.exists()


def test_lift_campaign_certifies_only_exact_results(tmp_path, monkeypatch):
    config = load_config(ROOT / DEFAULT_V8_TEMPLATE)
    identifier = pose_id(config)
    source = {
        "pose_id": identifier,
        "controller_id": controller_id(config),
        "source_candidate_id": -1,
        "source_config": str(ROOT / DEFAULT_V8_TEMPLATE),
        "source_config_sha256": tuning.canonical_sha256(config),
        "edge_m": float(config["cube"]["edge_m"]),
        "thumb_target_rad": float(
            config["control"]["grasp_targets_rad"][
                "left_hand_thumb_bend_joint_actuator"
            ]
        ),
        "config": config,
    }
    probe_candidates = tuple(
        {
            "candidate_id": 86_000_000_000_000 + index,
            "stage": "lift_probe",
            "pose_id": identifier,
        }
        for index in range(17)
    )
    trust_candidate = {
        "candidate_id": 87_000_000_000_000,
        "stage": "lift_trust",
        "pose_id": identifier,
    }
    refine_candidate = {
        "candidate_id": 88_000_000_000_000,
        "stage": "lift_refine",
        "pose_id": identifier,
    }
    exact_candidate = {
        "candidate_id": 89_000_000_000_000,
        "stage": "lift_exact",
        "pose_id": identifier,
    }
    trust_pass = _lift_result(
        trust_candidate["candidate_id"],
        stage="lift_trust",
        pose_id_value=identifier,
        median_lift_m=0.011,
        minimum_lift_m=0.009,
        failed_check_count=0,
        full_success=True,
    )
    refine_miss = _lift_result(
        refine_candidate["candidate_id"],
        stage="lift_refine",
        pose_id_value=identifier,
        median_lift_m=0.006,
        minimum_lift_m=0.005,
        failed_check_count=4,
    )
    exact_miss = _lift_result(
        exact_candidate["candidate_id"],
        stage="lift_exact",
        pose_id_value=identifier,
        median_lift_m=0.0065,
        minimum_lift_m=0.0062,
        failed_check_count=3,
    )
    published: list[dict] = []

    monkeypatch.setattr(tuning, "load_lift_sources", lambda **_kwargs: (source,))
    monkeypatch.setattr(
        tuning,
        "generate_lift_probe_candidates",
        lambda *_args, **_kwargs: probe_candidates,
    )
    monkeypatch.setattr(
        tuning, "_ctrlrange_safe_manipulation_delta_bounds", lambda _config: {}
    )
    monkeypatch.setattr(
        tuning,
        "fit_manipulation_response_model",
        lambda _probes, _bounds: {"available": False},
    )
    monkeypatch.setattr(
        tuning,
        "generate_lift_trust_candidates",
        lambda *_args, **_kwargs: (trust_candidate,),
    )
    monkeypatch.setattr(
        tuning,
        "generate_lift_refine_candidates",
        lambda *_args, **_kwargs: (refine_candidate,),
    )
    monkeypatch.setattr(
        tuning,
        "generate_lift_exact_candidates",
        lambda *_args, **_kwargs: (exact_candidate,),
    )

    def fake_run(candidates, *_args, **_kwargs):
        stage = candidates[0]["stage"]
        if stage == "lift_probe":
            return tuple(
                {
                    **candidate,
                    "manipulation_response": {"available": False},
                }
                for candidate in candidates
            )
        return {
            "lift_trust": (trust_pass,),
            "lift_refine": (refine_miss,),
            "lift_exact": (exact_miss,),
        }[stage]

    def fake_publish(_output, ranked, *, maximum_count):
        assert maximum_count == 1
        published.extend(copy.deepcopy(list(ranked)))
        return {
            "path": "trajectory_catalog/catalog.json",
            "sha256": "synthetic",
            "trajectory_count": len(ranked),
            "aliases": {},
            "best_config": "best_attempt_config.json",
        }

    monkeypatch.setattr(tuning, "_run_or_resume", fake_run)
    monkeypatch.setattr(tuning, "publish_lift_trajectory_catalog", fake_publish)
    budget = tuning.LiftBudget(
        pose_count=1,
        probe_count_per_pose=17,
        trust_candidates_per_pose=1,
        refine_pose_count=1,
        refine_per_pose=1,
        exact_count=1,
    )

    report = tuning.run_lift_campaign(tmp_path / "lift", budget=budget)

    assert report["provisional_lift_pass_count"] == 1
    assert report["lift_pass_count"] == 0
    assert report["validated_pose_count"] == 0
    assert report["best_search_candidate"]["candidate_id"] == trust_pass["candidate_id"]
    assert report["best_candidate"]["candidate_id"] == exact_miss["candidate_id"]
    assert [value["candidate_id"] for value in report["results"]] == [
        exact_miss["candidate_id"]
    ]
    assert {value["stage"] for value in report["search_results"]} == {
        "lift_trust",
        "lift_refine",
    }
    assert [value["stage"] for value in published] == ["lift_exact"]
    source_report = report["source_results"][0]
    assert source_report["provisional_lift_pass_count"] == 1
    assert source_report["lift_pass_count"] == 0
    assert source_report["best_candidate"]["candidate_id"] == exact_miss[
        "candidate_id"
    ]


def test_cli_exposes_rescue_lift_and_all():
    parser = build_parser()
    assert parser.parse_args(["--stage", "rescue"]).stage == "rescue"
    assert parser.parse_args(["--stage", "lift", "--config", "one.json"]).config == [
        "one.json"
    ]
    assert parser.parse_args(["--stage", "all"]).stage == "all"
