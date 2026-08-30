from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import (
    load_config,
    resolved_pose_constraint_values,
    validate_config,
)
from xhand_grasp.experiment import get_experiment, resolve_experiment
from xhand_grasp.experiments.opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift import (
    ACTUATOR_TARGET_BOUNDS_RAD,
    EXPERIMENT_DEFINITION,
    EXPERIMENT_ID,
    FALLBACK_FACE_ASSIGNMENT,
    FAR_HAND_CAMPAIGN,
    FAR_HAND_POSE_CONSTRAINTS,
    FINGERTIP_CONTACT_PREFERENCES,
    PRIMARY_FACE_ASSIGNMENT,
    THUMB_BEND_ACTUATOR,
)
from xhand_grasp.tuning.far_hand_fingertip import (
    FarHandStaticJob,
    FarHandTuningBudget,
    _clamp_relative_to_nominal_domain,
    _evidence_anchor_configs,
    _minimum_margin,
    boundary_expansion_decision,
    deterministic_rank_far_hand_results,
    far_hand_candidate_rank,
    far_hand_static_candidate_advances,
    far_hand_static_jobs,
    generate_far_hand_perturbation_configs,
    generate_far_hand_size_cases,
    materialize_far_hand_candidate,
    run_far_hand_robustness,
    run_parallel_far_hand_static_screen,
    tune_far_hand_fingertip,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift.json"
)


@pytest.fixture
def v5_config() -> dict:
    return load_config(CONFIG_PATH)


def _summary(*, grasp: bool, full: bool, pad_fraction: float = 0.9) -> dict:
    return {
        "passed": full,
        "failed_checks": [] if full else ["mock_incomplete"],
        "checks": {"mock_incomplete": full},
        "stage_status": {
            "grasp_success": grasp,
            "manipulation_success": full,
            "full_success": full,
        },
        "metrics": {
            "minimum_normalized_margin": 0.2 if full else -0.1,
            "grasp_stability_margin": 0.1 if grasp else -0.1,
            "contact_alignment": {
                "operation": {
                    "aligned_duty": 0.8 if full else 0.0,
                    "height_spread_p95_m": 0.003 if full else 0.03,
                }
            },
            "fingertip_contact": {
                "verify": {
                    "sample_count": 10,
                    "force_weighted_pad_fraction": {
                        finger: pad_fraction for finger in ("thumb", "index", "mid")
                    },
                    "max_active_taxel_count": {
                        "thumb": 1,
                        "index": 2,
                        "mid": 1,
                    },
                },
                "operation": {"sample_count": 0},
            },
            "operation_target_face_simultaneous_duty": 0.75,
            "peak_total_distal_contact_force_n": 2.0,
            "orientation_drift_deg": 1.0,
            "actuator_saturation_fraction": 0.05,
        },
    }


def test_v5_template_is_strictly_registered_and_resolved(v5_config):
    definition = resolve_experiment(v5_config)
    resolved = resolved_pose_constraint_values(v5_config)

    assert v5_config["schema_version"] == 5
    assert definition is EXPERIMENT_DEFINITION
    assert get_experiment(EXPERIMENT_ID) is definition
    assert definition.tuning_strategy == "far_hand_fingertip"
    assert definition.candidate_faces == (
        PRIMARY_FACE_ASSIGNMENT,
        FALLBACK_FACE_ASSIGNMENT,
    )
    assert v5_config["far_hand_campaign"] == FAR_HAND_CAMPAIGN.as_config()
    assert v5_config["pose_constraints"] == FAR_HAND_POSE_CONSTRAINTS.as_config()
    assert v5_config["fingertip_contact_preferences"] == (
        FINGERTIP_CONTACT_PREFERENCES.as_config()
    )
    assert "palm_press_depth_m" not in v5_config["pose_constraints"]
    assert resolved["root_cube_distance_m"] == pytest.approx(0.14039311926358694)
    assert resolved["legacy_palm_press_depth_m"] == pytest.approx(
        -0.02550315287223842
    )
    assert resolved["finger_down_tilt_deg"] == pytest.approx(15.0)


def test_v5_domain_material_and_formal_budgets_are_exact():
    bounds = EXPERIMENT_DEFINITION.search_bounds
    campaign = FAR_HAND_CAMPAIGN

    assert FAR_HAND_POSE_CONSTRAINTS.root_cube_distance_m == (0.138, 0.160)
    assert FAR_HAND_POSE_CONSTRAINTS.cube_position_in_root_m == {
        "x": (0.084, 0.105),
        "y": (-0.035, -0.023),
        "z": (0.100, 0.120),
    }
    assert bounds.hand_roll_deg == (-2.0, 3.0)
    assert bounds.hand_yaw_deg == (-5.0, 0.0)
    assert bounds.cube_yaw_deg == (24.0, 38.0)
    assert ACTUATOR_TARGET_BOUNDS_RAD[THUMB_BEND_ACTUATOR] == (0.85, 1.35)
    assert campaign.nominal_edge_m == pytest.approx(0.060)
    assert campaign.nominal_mass_kg == pytest.approx(0.160)
    assert campaign.constant_density_mass_kg(0.060) == pytest.approx(0.160)
    assert campaign.static_sample_count == 120_000
    assert campaign.dynamic_candidate_count == 640
    assert campaign.grasp_refinement_count == 2_560
    assert campaign.manipulation_refinement_count == 1_280
    assert campaign.exact_candidate_count == 80
    assert campaign.closure_alpha_values[0] == pytest.approx(0.8)
    assert campaign.closure_alpha_values[-1] == pytest.approx(1.0)


def test_v5_validation_rejects_small_thumb_targets_and_old_press_semantics(v5_config):
    too_open = copy.deepcopy(v5_config)
    too_open["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR] = 0.84
    with pytest.raises(ValueError, match="registered search bounds|too small"):
        validate_config(too_open)

    low_final = copy.deepcopy(v5_config)
    low_final["control"]["manipulation_delta_rad"][THUMB_BEND_ACTUATOR] = -0.2
    low_final["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR] = 0.9
    with pytest.raises(ValueError, match="manipulated thumb bend"):
        validate_config(low_final)

    press = copy.deepcopy(v5_config)
    press["pose_constraints"]["palm_press_depth_m"] = [0.002, 0.008]
    with pytest.raises(ValueError, match="pose_constraints"):
        validate_config(press)


def test_static_job_plan_has_exact_primary_fallback_allocation_and_ids():
    jobs = far_hand_static_jobs()

    assert len(jobs) == 10
    assert sum(job.samples for job in jobs) == 120_000
    assert len({job.seed for job in jobs}) == len(jobs)
    assert len({job.first_candidate_id for job in jobs}) == len(jobs)
    assert jobs[0].first_candidate_id == 5
    for previous, following in zip(jobs, jobs[1:]):
        assert previous.first_candidate_id + previous.samples == (
            following.first_candidate_id
        )
    for band in FAR_HAND_CAMPAIGN.tilt_band_centers_deg:
        grouped = [job for job in jobs if job.tilt_band_center_deg == band]
        assert [job.topology_kind for job in grouped] == ["primary", "fallback"]
        assert [job.samples for job in grouped] == [20_000, 4_000]
        assert grouped[0].target_assignment == PRIMARY_FACE_ASSIGNMENT
        assert grouped[1].target_assignment == FALLBACK_FACE_ASSIGNMENT


def test_static_gate_allows_distal_preload_but_not_nondistal_penetration():
    diagnostic = {
        "selected_signed_gap_m": [-0.0005, 0.0, 0.003],
        "full_target_distal_preload_m": 0.010,
        "forbidden_penetration": False,
    }
    assert far_hand_static_candidate_advances(diagnostic)

    diagnostic["full_target_distal_preload_m"] = 0.010001
    assert not far_hand_static_candidate_advances(diagnostic)
    diagnostic["full_target_distal_preload_m"] = 0.005
    diagnostic["forbidden_penetration"] = True
    assert not far_hand_static_candidate_advances(diagnostic)


def test_five_evidence_anchor_templates_have_reserved_unique_ids(v5_config):
    anchors = _evidence_anchor_configs(v5_config, resolve_experiment(v5_config))

    assert len(anchors) == 5
    assert [
        item["candidate_metadata"]["static_candidate_id"] for item in anchors
    ] == [0, 1, 2, 3, 4]
    interior = anchors[3:]
    assert all(
        item["candidate_metadata"][
            "independently_reproduced_full_success_seed"
        ]
        is True
        for item in interior
    )
    assert all(
        item["candidate_metadata"]["tilt_band_center_deg"] == 17.5
        for item in interior
    )
    assert resolved_pose_constraint_values(interior[0])[
        "root_cube_distance_m"
    ] == pytest.approx(0.14485026104801968)
    assert resolved_pose_constraint_values(interior[1])[
        "root_cube_distance_m"
    ] == pytest.approx(0.1468099401647703)
def test_real_geom_closure_smoke_records_pad_and_preload(v5_config):
    definition = resolve_experiment(v5_config)
    job = FarHandStaticJob(
        job_id=0,
        first_candidate_id=5,
        tilt_band_center_deg=15.0,
        topology_kind="primary",
        target_assignment=definition.far_hand_campaign.primary_face,
        samples=1,
        seed=17,
    )

    outcome = run_parallel_far_hand_static_screen(
        v5_config, jobs=[job], retain_per_band=2, workers=1
    )

    assert outcome.sample_count == 1
    assert outcome.job_records[0]["sample_count"] == 1
    assert outcome.job_records[0]["topology_kind"] == "primary"
    near = [value for value in outcome.near_miss_by_band.values() if value]
    assert near
    for item in near:
        diagnostic = item["diagnostic"]
        assert len(diagnostic["closure_frames"]) == len(
            FAR_HAND_CAMPAIGN.closure_alpha_values
        )
        assert 0 <= diagnostic["pad_assigned_count"] <= 3
        assert diagnostic["full_target_distal_preload_m"] >= 0.0
        assert "max_active_nondistal_penetration_m" in diagnostic
    retained_15 = outcome.retained_by_band[15.0]
    assert any(
        item["candidate_metadata"].get("evidence_anchor")
        == "transformed_v3_dynamic_contact"
        and item["candidate_metadata"].get("evidence_anchor_force_dynamic")
        is True
        for item in retained_15
    )
    reproduced = next(
        item
        for item in outcome.retained_by_band[20.0]
        if item["candidate_metadata"].get(
            "independently_reproduced_full_success_seed"
        )
        is True
    )
    reproduced_pose = resolved_pose_constraint_values(reproduced)
    assert reproduced["candidate_metadata"]["static_candidate_id"] == 2
    assert reproduced["candidate_metadata"]["evidence_anchor"] == (
        "independently_reproduced_full_success_seed"
    )
    assert reproduced["hand_pose"]["rpy_deg"] == pytest.approx(
        [-2.0, 110.0, -1.7892512430329526]
    )
    assert reproduced_pose["finger_down_tilt_deg"] == pytest.approx(
        19.9872968445099
    )
    assert reproduced_pose["root_cube_distance_m"] == pytest.approx(
        0.14583837792436852
    )
    assert reproduced["control"]["manipulation_delta_rad"][
        THUMB_BEND_ACTUATOR
    ] == pytest.approx(-0.2)
    interior = [
        item
        for item in outcome.retained_by_band[17.5]
        if item["candidate_metadata"].get(
            "independently_reproduced_full_success_seed"
        )
        is True
    ]
    assert {
        item["candidate_metadata"]["static_candidate_id"] for item in interior
    } == {3, 4}
    assert {
        item["candidate_metadata"]["evidence_anchor"] for item in interior
    } == {
        "independently_reproduced_interior_best_r24_full_success_seed",
        "independently_reproduced_interior_grid_full_success_seed",
    }


def test_twenty_degree_static_band_samples_feasible_resolved_tilts(v5_config):
    job = FarHandStaticJob(
        job_id=8,
        first_candidate_id=5,
        tilt_band_center_deg=20.0,
        topology_kind="primary",
        target_assignment=PRIMARY_FACE_ASSIGNMENT,
        samples=16,
        seed=20260821,
    )

    outcome = run_parallel_far_hand_static_screen(
        v5_config, jobs=[job], retain_per_band=2, workers=1
    )

    record = outcome.job_records[0]
    assert record["resolved_tilt_bounds_deg"] == pytest.approx([18.75, 20.0])
    assert record["valid_pose_count"] > 0


def test_boundary_expansion_is_upper_only_and_one_shot(v5_config):
    candidate = copy.deepcopy(v5_config)
    candidate["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR] = 1.34
    validate_config(candidate)
    decision = boundary_expansion_decision(candidate)

    assert decision == {
        "applied": True,
        "count": 1,
        "distance_expanded": False,
        "thumb_bend_expanded": True,
        "reason": "nominal_upper_boundary_hit",
    }
    candidate.setdefault("candidate_metadata", {})["boundary_expansion"] = decision
    exhausted = boundary_expansion_decision(candidate)
    assert exhausted["applied"] is False
    assert exhausted["count"] == 1
    assert exhausted["reason"] == "one_shot_expansion_already_consumed"


def test_dynamic_rank_uses_pad_preference_after_hard_quantitative_metrics(v5_config):
    better = {"candidate_id": 2, "config": v5_config, "summary": _summary(grasp=True, full=True, pad_fraction=0.95)}
    worse = {"candidate_id": 1, "config": v5_config, "summary": _summary(grasp=True, full=True, pad_fraction=0.75)}

    assert far_hand_candidate_rank(better) > far_hand_candidate_rank(worse)
    assert deterministic_rank_far_hand_results([worse, better]) == (better, worse)


def test_dynamic_rank_accepts_null_alignment_metrics_from_empty_windows(v5_config):
    empty_window = {
        "candidate_id": 1,
        "config": v5_config,
        "summary": _summary(grasp=True, full=False),
    }
    empty_window["summary"]["metrics"]["contact_alignment"]["operation"] = {
        "aligned_duty": None,
        "height_spread_p95_m": None,
    }
    measured_window = {
        "candidate_id": 2,
        "config": v5_config,
        "summary": _summary(grasp=True, full=False),
    }

    assert far_hand_candidate_rank(empty_window)
    assert deterministic_rank_far_hand_results(
        [empty_window, measured_window]
    ) == (measured_window, empty_window)


def test_local_pose_projection_handles_box_and_distance_shell_boundaries(v5_config):
    definition = resolve_experiment(v5_config)
    constraints = definition.far_hand_pose_constraints
    assert constraints is not None
    probes = [
        np.array([0.084, -0.023, 0.100]),  # below the 138 mm shell
        np.array([0.105, -0.035, 0.120]),  # above the 160 mm shell
        np.array([0.050, -0.010, 0.070]),  # outside several box faces
        np.array([0.140, -0.060, 0.160]),
    ]
    rng = np.random.default_rng(20260821)
    probes.extend(
        rng.uniform(
            low=[0.060, -0.050, 0.075],
            high=[0.125, -0.010, 0.145],
            size=(5000, 3),
        )
    )

    for probe in probes:
        projected = _clamp_relative_to_nominal_domain(probe, definition)
        assert constraints.contains_cube_position(projected)
        assert _clamp_relative_to_nominal_domain(
            projected, definition
        ) == pytest.approx(projected, abs=1e-14)

    assert np.linalg.norm(
        _clamp_relative_to_nominal_domain(probes[0], definition)
    ) == pytest.approx(0.138, abs=1e-12)
    assert np.linalg.norm(
        _clamp_relative_to_nominal_domain(probes[1], definition)
    ) == pytest.approx(0.160, abs=1e-12)


def test_dynamic_rank_prioritizes_grasp_and_verify_evidence_over_pad_soft_score(
    v5_config,
):
    pad_only = _summary(grasp=False, full=False, pad_fraction=1.0)
    three_finger_near = _summary(grasp=False, full=False, pad_fraction=0.1)
    three_finger_near["metrics"].update(
        {
            "verify_effective_finger_count": 3,
            "verify_max_simultaneous_effective_finger_count": 3,
            "verify_target_face_effective_duty": {
                finger: 0.9 for finger in ("thumb", "index", "mid")
            },
            "verify_target_face_simultaneous_duty": 0.85,
        }
    )
    grasped = _summary(grasp=True, full=False, pad_fraction=0.0)
    cases = [
        {"candidate_id": 1, "config": v5_config, "summary": pad_only},
        {"candidate_id": 2, "config": v5_config, "summary": three_finger_near},
        {"candidate_id": 3, "config": v5_config, "summary": grasped},
    ]

    ranked = deterministic_rank_far_hand_results(cases)
    assert [item["candidate_id"] for item in ranked] == [3, 2, 1]


def test_full_rank_computes_real_minimum_margin_before_alignment_soft_terms(
    v5_config, monkeypatch
):
    def margins(metrics, acceptance, **kwargs):
        del acceptance, kwargs
        return {"synthetic_real_margin": metrics["rank_margin"]}

    monkeypatch.setattr(
        "xhand_grasp.search.normalized_acceptance_margins", margins
    )
    boundary_summary = _summary(grasp=True, full=True, pad_fraction=1.0)
    interior_summary = _summary(grasp=True, full=True, pad_fraction=0.1)
    for summary, margin, aligned in (
        (boundary_summary, 0.01, 0.99),
        (interior_summary, 0.20, 0.76),
    ):
        summary["metrics"].pop("minimum_normalized_margin")
        summary["metrics"]["rank_margin"] = margin
        summary["metrics"]["contact_alignment"]["operation"][
            "aligned_duty"
        ] = aligned
    boundary = {
        "candidate_id": 1,
        "config": v5_config,
        "summary": boundary_summary,
    }
    interior = {
        "candidate_id": 2,
        "config": v5_config,
        "summary": interior_summary,
    }

    assert _minimum_margin(boundary) == pytest.approx(0.01)
    assert _minimum_margin(interior) == pytest.approx(0.20)
    assert far_hand_candidate_rank(interior) > far_hand_candidate_rank(boundary)


def test_size_and_perturbation_cases_are_deterministic_and_schema_valid(v5_config):
    sizes = generate_far_hand_size_cases(v5_config)
    first = generate_far_hand_perturbation_configs(v5_config, count=5, seed=71)
    second = generate_far_hand_perturbation_configs(v5_config, count=5, seed=71)

    assert [round(case["cube"]["edge_m"] * 1000) for case in sizes] == [
        59,
        60,
        61,
        62,
        63,
        64,
    ]
    assert first == second
    for case in sizes + first:
        assert case["run_context"] == {"kind": "robustness_trial"}
        validate_config(case)


def test_tiny_injected_tuner_preserves_grasp_gate_and_stage_counts(v5_config):
    definition = resolve_experiment(v5_config)
    seed_candidate = materialize_far_hand_candidate(
        v5_config,
        tilt_band_center_deg=15.0,
        roll_deg=0.0,
        yaw_deg=-2.6325407810459214,
        cube_in_root_m=(0.084, -0.02886063167725714, 0.10872576454433164),
        cube_yaw_deg=30.0,
        grasp_targets_rad=v5_config["control"]["grasp_targets_rad"],
        target_assignment=PRIMARY_FACE_ASSIGNMENT,
        static_candidate_id=7,
    )

    def static_runner(base, **kwargs):
        del base, kwargs
        return {
            "sample_count": 2,
            "retained_by_band": {15.0: [copy.deepcopy(seed_candidate)]},
        }

    calls = []

    def run_candidates(payloads, workers):
        assert workers == 1
        result = []
        for candidate_id, candidate in reversed(payloads):
            stage = candidate["candidate_metadata"]["search_stage"]
            calls.append(stage)
            grasp = stage != "close_verify" or True
            full = stage in {
                "manipulation_refinement",
                "exact_1ms_confirmation",
                "post_success_size_sweep",
                "robustness_trial",
            }
            result.append(
                {
                    "candidate_id": candidate_id,
                    "config": copy.deepcopy(candidate),
                    "summary": _summary(grasp=grasp, full=full),
                }
            )
        return result

    tiny = FarHandTuningBudget(
        primary_static_samples_per_band=1,
        fallback_static_samples_per_band=1,
        static_retain_per_band=1,
        dynamic_candidates_per_band=1,
        grasp_refine_seed_count_per_band=1,
        grasp_refine_per_seed=1,
        manipulation_seed_count_per_band=1,
        manipulation_refine_per_seed=1,
        exact_candidates_per_band=1,
        perturbation_count=0,
    )
    report = tune_far_hand_fingertip(
        v5_config,
        workers=1,
        seed=9,
        run_candidates=run_candidates,
        static_runner=static_runner,
        budget=tiny,
    )

    assert report["nominal_passed"] is True
    # The four bands omitted by the injected static runner receive a real
    # dynamics fallback so the five-band catalog can never be empty/crash.
    assert report["stage_counts"]["close_verify"] == 5
    assert report["stage_counts"]["pose_grasp_refinement"] == 5
    assert report["stage_counts"]["manipulation_refinement"] == 5
    assert report["stage_counts"]["exact_1ms_confirmation"] == 5
    assert report["stage_counts"]["post_success_size_sweep"] == 6
    assert report["stage_counts"]["robustness_trial"] == 0
    assert "manipulation_refinement" in calls
    assert report["best"]["search_stage"] == "exact_1ms_confirmation"
    assert len(report["selected_band_candidates"]) == 5
    assert report["best"]["candidate_id"] in {
        item["candidate_id"] for item in report["selected_band_candidates"]
    }
    assert report["best"]["local_perturbation_probe"]["trial_count"] == 0


def test_reproduced_full_success_anchor_is_pinned_for_exact_confirmation(
    v5_config,
):
    definition = resolve_experiment(v5_config)
    anchor = _evidence_anchor_configs(v5_config, definition)[2]

    def static_runner(base, **kwargs):
        del base, kwargs
        return {
            "sample_count": 0,
            "retained_by_band": {20.0: [copy.deepcopy(anchor)]},
        }

    def runner(payloads, workers):
        assert workers == 1
        results = []
        for candidate_id, candidate in payloads:
            metadata = candidate["candidate_metadata"]
            original_seed = bool(
                metadata.get("independently_reproduced_full_success_seed")
            )
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": copy.deepcopy(candidate),
                    "summary": _summary(
                        grasp=original_seed,
                        full=original_seed,
                    ),
                }
            )
        return results

    tiny = FarHandTuningBudget(
        primary_static_samples_per_band=1,
        fallback_static_samples_per_band=1,
        static_retain_per_band=1,
        dynamic_candidates_per_band=1,
        grasp_refine_seed_count_per_band=1,
        grasp_refine_per_seed=1,
        manipulation_seed_count_per_band=1,
        manipulation_refine_per_seed=1,
        exact_candidates_per_band=1,
        perturbation_count=0,
    )
    report = tune_far_hand_fingertip(
        v5_config,
        workers=1,
        seed=13,
        run_candidates=runner,
        static_runner=static_runner,
        budget=tiny,
    )

    assert report["nominal_passed"] is True
    assert report["stage_counts"]["exact_1ms_confirmation"] == 1
    assert report["best"]["search_stage"] == "exact_1ms_confirmation"
    assert report["best"]["config"]["candidate_metadata"][
        "independently_reproduced_full_success_seed"
    ] is True
    assert report["best"]["candidate_id"] in {
        item["candidate_id"] for item in report["selected_band_candidates"]
    }


def test_both_interior_success_anchors_survive_tiny_dynamic_and_exact_limits(
    v5_config,
):
    definition = resolve_experiment(v5_config)
    anchors = _evidence_anchor_configs(v5_config, definition)[3:]

    def static_runner(base, **kwargs):
        del base, kwargs
        return {
            "sample_count": 0,
            "retained_by_band": {17.5: copy.deepcopy(anchors)},
        }

    exact_anchor_names = []

    def runner(payloads, workers):
        assert workers == 1
        results = []
        for candidate_id, candidate in payloads:
            metadata = candidate["candidate_metadata"]
            original_seed = bool(
                metadata.get("independently_reproduced_full_success_seed")
            )
            if metadata["search_stage"] == "exact_1ms_confirmation" and original_seed:
                exact_anchor_names.append(metadata["evidence_anchor"])
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": copy.deepcopy(candidate),
                    "summary": _summary(
                        grasp=original_seed,
                        full=original_seed,
                    ),
                }
            )
        return results

    tiny = FarHandTuningBudget(
        primary_static_samples_per_band=1,
        fallback_static_samples_per_band=1,
        static_retain_per_band=1,
        dynamic_candidates_per_band=1,
        grasp_refine_seed_count_per_band=1,
        grasp_refine_per_seed=1,
        manipulation_seed_count_per_band=1,
        manipulation_refine_per_seed=1,
        exact_candidates_per_band=1,
        perturbation_count=0,
    )
    report = tune_far_hand_fingertip(
        v5_config,
        workers=1,
        seed=19,
        run_candidates=runner,
        static_runner=static_runner,
        budget=tiny,
    )

    assert report["nominal_passed"] is True
    assert set(exact_anchor_names) == {
        "independently_reproduced_interior_best_r24_full_success_seed",
        "independently_reproduced_interior_grid_full_success_seed",
    }
    assert report["stage_counts"]["exact_1ms_confirmation"] == 2


def test_robustness_skips_post_success_work_when_nominal_fails(v5_config):
    calls = []

    def runner(payloads, workers):
        calls.extend(payloads)
        return [
            {
                "candidate_id": candidate_id,
                "config": copy.deepcopy(config),
                "summary": _summary(grasp=False, full=False),
            }
            for candidate_id, config in payloads
        ]

    report = run_far_hand_robustness(
        v5_config, workers=1, seed=31, run_candidates=runner
    )

    assert len(calls) == 1
    assert report["nominal_passed"] is False
    assert report["grid_case_count"] == 0
    assert report["perturbation_trial_count"] == 0
    assert report["stop_reason"] == "nominal_failed_post_success_campaign_skipped"
