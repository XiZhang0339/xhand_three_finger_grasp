from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.larger_cube_grasp_search import (
    DEFAULT_PLAN,
    SizeScreenSummary,
    _per_size_outcomes,
    _run_v3_stage,
    adjacent_neighbor_edges_m,
    build_manipulation_refinement_candidates,
    campaign_manifest,
    coarse_screen_jobs,
    constant_density_candidates,
    deterministic_rank_results,
    exact_screen_jobs,
    kinematic_screen_v3,
    larger_cube_candidate_rank,
    larger_cube_robustness_cases,
    manipulation_targets,
    neighbor_screen_jobs,
    robustness_edge_window_m,
    sample_manipulation_delta_candidates,
    select_coarse_edges_m,
    select_exact_edges_m,
    select_grasp_refinement_parents,
    select_manipulation_parents,
    tune_larger_cube_grasp_then_lift,
)
from xhand_grasp.v2_search import KinematicScreenResult


ROOT = Path(__file__).resolve().parents[1]
V3_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_grasp_then_lift.json"
)


def _targets(value: float) -> dict[str, float]:
    return {name: value for name in ACTIVE_ACTUATORS}


def _config(
    *, edge_m: float = 0.058, mass_kg: float = 0.020, friction: float = 0.8
) -> dict:
    return {
        "schema_version": 3,
        "experiment_id": DEFAULT_PLAN.experiment_id,
        "cube": {
            "edge_m": edge_m,
            "mass_kg": mass_kg,
            "friction": friction,
        },
        "control": {
            "grasp_targets_rad": _targets(0.7),
            "manipulation_delta_rad": _targets(0.0),
        },
    }


def _result(
    candidate_id: int,
    *,
    edge_m: float = 0.058,
    grasp: bool = False,
    manipulation: bool = False,
    full: bool = False,
    margin: float = -1.0,
    perturbation_passes: int = 0,
    topology: float = 0.0,
    force: float = 2.0,
    saturation: float = 0.1,
    generic_passed: bool | None = None,
) -> dict:
    summary = {
        "stage_status": {
            "grasp_success": grasp,
            "manipulation_success": manipulation,
            "full_success": full,
        },
        "metrics": {
            "grasp_stability_margin": margin,
            "target_face_simultaneous_duty": topology,
            "peak_total_distal_contact_force_n": force,
            "actuator_saturation_fraction": saturation,
        },
    }
    if generic_passed is not None:
        summary["passed"] = generic_passed
    return {
        "candidate_id": candidate_id,
        "config": _config(edge_m=edge_m),
        "summary": summary,
        "local_perturbation_probe": {"passes": perturbation_passes},
    }


def _screen_summary(
    edge_m: float,
    *,
    eligible: int,
    score: tuple[float, ...] = (0.0,),
) -> SizeScreenSummary:
    return SizeScreenSummary(
        edge_m=edge_m,
        sample_count=35_000,
        eligible_count=eligible,
        clean_three_count=eligible,
        near_three_count=eligible,
        best_score=score,
    )


def test_campaign_manifest_declares_complete_larger_cube_budget():
    plan = DEFAULT_PLAN
    assert tuple(edge * 1000.0 for edge in plan.coarse_edges_m) == pytest.approx(
        (52, 54, 56, 58, 60, 62, 64)
    )
    assert plan.constant_density_mass_kg(0.052) == pytest.approx(
        0.020 * (52 / 30) ** 3
    )
    assert plan.constant_density_mass_kg(0.064) == pytest.approx(
        0.020 * (64 / 30) ** 3
    )
    assert plan.dynamic_candidate_count == 512
    assert plan.grasp_refinement_count == 1_024
    assert plan.manipulation_refinement_count == 1_024
    assert plan.density_refinement_count == 512
    assert plan.robustness_case_count == 100

    manifest = campaign_manifest(plan)
    assert manifest["nominal_counts"]["coarse_job_count"] == 49
    assert manifest["nominal_counts"]["coarse_samples_per_size"] == 35_000
    assert manifest["nominal_counts"]["coarse_sample_count"] == 245_000
    assert manifest["nominal_counts"]["declared_neighbor_sample_count"] == 420_000
    assert manifest["nominal_counts"]["declared_exact_sample_count"] == 700_000
    assert (
        manifest["nominal_counts"]["declared_max_static_sample_count"]
        == 1_365_000
    )
    assert manifest["hard_stage_gates"] == {
        "manipulation_requires_grasp_success": True,
        "constant_density_requires_fixed_mass_full_success": True,
        "campaign_success_requires_constant_density_full_success": True,
    }


def test_real_v3_static_screen_samples_only_a_grasp_pose():
    config = load_config(V3_CONFIG)
    screen = kinematic_screen_v3(
        config, samples_per_pitch=1, retain=4, seed=20260821
    )
    assert screen.sample_count == 7
    assert screen.retained_count == 4
    assert len(screen.candidates) == len(screen.diagnostics) == 4
    for candidate in screen.candidates:
        assert set(candidate["control"]) == {
            "grasp_targets_rad",
            "manipulation_delta_rad",
        }
        assert set(candidate["control"]["grasp_targets_rad"]) == set(
            ACTIVE_ACTUATORS
        )
        assert all(
            value == 0.0
            for value in candidate["control"]["manipulation_delta_rad"].values()
        )


def test_screen_jobs_are_face_balanced_complete_and_seed_stable():
    first = coarse_screen_jobs()
    second = coarse_screen_jobs()
    assert first == second
    assert len(first) == 7 * 7
    assert len({job.seed for job in first}) == len(first)
    assert {tuple(job.face_sample_counts.values()) for job in first} == {
        (1250, 1250, 1250, 1250)
    }
    assert {job.edge_m for job in first} == set(DEFAULT_PLAN.coarse_edges_m)
    assert {job.pitch_deg for job in first} == set(DEFAULT_PLAN.pitch_values_deg)


def test_coarse_neighbor_and_exact_size_selection_is_worker_order_independent():
    summaries = [
        _screen_summary(edge, eligible=index, score=(float(index),))
        for index, edge in enumerate(DEFAULT_PLAN.coarse_edges_m)
    ]
    expected_coarse = (0.064, 0.062, 0.060)
    assert select_coarse_edges_m(summaries) == pytest.approx(expected_coarse)
    random.Random(91).shuffle(summaries)
    assert select_coarse_edges_m(summaries) == pytest.approx(expected_coarse)

    neighbors = adjacent_neighbor_edges_m(expected_coarse)
    assert neighbors == pytest.approx((0.059, 0.061, 0.063))
    neighbor_jobs = neighbor_screen_jobs(expected_coarse)
    assert len(neighbor_jobs) == len(neighbors) * 7
    assert all(job.samples == 10_000 for job in neighbor_jobs)
    assert all(set(job.face_sample_counts.values()) == {2500} for job in neighbor_jobs)

    pool = [
        _screen_summary(edge, eligible=index + 1, score=(float(index),))
        for index, edge in enumerate((*expected_coarse, *neighbors))
    ]
    exact = select_exact_edges_m(pool)
    assert exact == pytest.approx((0.063, 0.061))
    exact_jobs = exact_screen_jobs(exact)
    assert len(exact_jobs) == 14
    assert all(job.samples == 50_000 for job in exact_jobs)
    assert all(set(job.face_sample_counts.values()) == {12_500} for job in exact_jobs)


def test_neighbor_stage_requires_three_declared_coarse_sizes():
    with pytest.raises(ValueError, match="exactly 3"):
        adjacent_neighbor_edges_m((0.052, 0.054))
    with pytest.raises(ValueError, match="declared coarse"):
        adjacent_neighbor_edges_m((0.052, 0.054, 0.055))


def test_v3_stage_reorders_valid_worker_results_and_preserves_id_binding():
    configs = [_config(edge_m=0.052), _config(edge_m=0.054)]

    def reversed_runner(payload, workers):
        assert workers == 3
        return [
            {
                "candidate_id": candidate_id,
                "config": config,
                "summary": {"stage_status": {}, "metrics": {}},
            }
            for candidate_id, config in reversed(payload)
        ]

    results, next_id = _run_v3_stage(
        configs,
        next_id=10,
        workers=3,
        run_candidates=reversed_runner,
        stage="contract_test",
        material_policy="fixed_20g_control",
    )
    assert [result["candidate_id"] for result in results] == [10, 11]
    assert next_id == 12
    assert all(result["search_stage"] == "contract_test" for result in results)


@pytest.mark.parametrize("failure", ("missing", "duplicate", "rebound"))
def test_v3_stage_rejects_invalid_runner_results(failure):
    configs = [_config(edge_m=0.052), _config(edge_m=0.054)]

    def invalid_runner(payload, workers):
        del workers
        results = [
            {
                "candidate_id": candidate_id,
                "config": config,
                "summary": {"stage_status": {}, "metrics": {}},
            }
            for candidate_id, config in payload
        ]
        if failure == "missing":
            return results[:1]
        if failure == "duplicate":
            results[1]["candidate_id"] = results[0]["candidate_id"]
            return results
        results[1]["config"] = copy.deepcopy(results[0]["config"])
        return results

    message = "different configuration" if failure == "rebound" else "exactly match"
    with pytest.raises(RuntimeError, match=message):
        _run_v3_stage(
            configs,
            next_id=0,
            workers=1,
            run_candidates=invalid_runner,
            stage="contract_test",
            material_policy="fixed_20g_control",
        )


def test_tune_blocks_zero_delta_full_results_and_reports_actual_static_budget():
    config = load_config(V3_CONFIG)

    def fake_screen(
        base,
        *,
        samples_per_pitch,
        retain,
        seed,
        definition,
    ):
        sample_count = (
            len(definition.search_bounds.palm_pitch_values_deg)
            * samples_per_pitch
        )
        retained = min(retain, sample_count)
        candidates = tuple(copy.deepcopy(base) for _ in range(retained))
        edge = float(base["cube"]["edge_m"])
        diagnostics = tuple(
            {
                "candidate_id": index,
                "score": (edge, 1.0, 1.0),
                "clean_target_contact_count": 3,
                "near_target_face_count": 3,
                "target_site_signed_distance_m": (0.0, 0.0, 0.0),
                "forbidden_contact": False,
                "max_penetration_m": 0.0,
            }
            for index in range(retained)
        )
        return KinematicScreenResult(
            seed=seed,
            sample_count=sample_count,
            retained_count=retained,
            candidates=candidates,
            diagnostics=diagnostics,
        )

    def fake_runner(payload, workers):
        assert workers == 2
        results = []
        for candidate_id, candidate in payload:
            delta = candidate["control"]["manipulation_delta_rad"]
            zero_delta = all(abs(float(value)) <= 1e-15 for value in delta.values())
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": candidate,
                    "summary": {
                        "passed": zero_delta,
                        "failed_checks": [] if zero_delta else ["mock_lift"],
                        "checks": {},
                        "stage_status": {
                            "grasp_success": True,
                            "manipulation_success": zero_delta,
                            "full_success": zero_delta,
                        },
                        "metrics": {
                            "grasp_stability_margin": 1.0
                            - candidate_id * 1e-6,
                            "operation_target_face_simultaneous_duty": 0.9,
                            "peak_total_distal_contact_force_n": 1.0,
                            "actuator_saturation_fraction": 0.0,
                        },
                    },
                }
            )
        return list(reversed(results))

    def rank(result):
        status = result["summary"]["stage_status"]
        return (
            float(status["full_success"]),
            float(status["grasp_success"]),
            -float(result["candidate_id"]),
        )

    tuned = tune_larger_cube_grasp_then_lift(
        config,
        workers=2,
        seed=20260821,
        run_candidates=fake_runner,
        rank_candidate=rank,
        kinematic_samples_per_pitch=1,
        dynamic_candidate_count=1,
        local_refine_seed_count=1,
        local_refine_per_seed=1,
        final_candidate_count=2,
        perturbations_per_final=1,
        fallback_physics_count=1,
        fallback_kinematic_samples_per_pitch=1,
        perturb_cases=None,
        screen_candidates=fake_screen,
    )

    # The initial and grasp-refinement runs deliberately claim full success,
    # but they have a zero manipulation delta and cannot authorize density.
    assert tuned["fixed_mass_passing_candidates"] == 0
    assert tuned["constant_density_reevaluation_count"] == 0
    assert tuned["fixed_mass_success"] is False
    assert tuned["best"]["search_stage"] == "fixed_mass_manipulation_refinement"
    assert all(
        result["search_stage"] == "fixed_mass_manipulation_refinement"
        for result in tuned["top_candidates"]
    )

    budget = tuned["static_search_budget"]
    assert budget["versioned_declared"]["maximum_total_sample_count"] == 1_365_000
    assert budget["effective_declared"]["maximum_total_sample_count"] == 147
    assert budget["actual"]["total_sample_count"] == 105
    assert budget["actual"]["unique_neighbor_edge_count"] == 3
    assert budget["neighbor_selection"]["duplicate_slot_count"] == 2
    assert budget["neighbor_selection"]["out_of_range_slot_count"] == 1
    assert budget["stop_reason"] == (
        "completed_with_neighbor_edge_deduplication_and_neighbor_range_clamping"
    )

    outcomes = tuned["per_size_outcomes"]
    json.dumps(outcomes, allow_nan=False)
    assert len(outcomes) == 2
    active = max(outcomes, key=lambda item: item["total_dynamic_trial_count"])
    assert active["initial_dynamic_trial_count"] == 1
    assert active["grasp_refinement_trial_count"] == 1
    assert active["stable_grasp_evaluation_count"] == 2
    assert active["stable_grasp_success_count"] == 2
    assert active["stable_grasp_success_rate"] == pytest.approx(1.0)
    assert active["manipulation_refinement_trial_count"] == 256
    assert active["fixed_full_success_count"] == 0
    assert active["density_full_success_count"] == 0


def test_per_size_outcomes_counts_fixed_and_density_successes():
    records = _per_size_outcomes(
        selected_exact_edges_m=(0.058, 0.060),
        initial_runs=(
            _result(0, edge_m=0.058, grasp=True),
            _result(1, edge_m=0.058, grasp=False),
            _result(2, edge_m=0.060, grasp=True),
        ),
        grasp_local_runs=(_result(3, edge_m=0.058, grasp=True),),
        manipulation_runs=(
            _result(4, edge_m=0.058, grasp=True, full=True),
            _result(5, edge_m=0.060, grasp=True, full=False),
        ),
        density_runs=(_result(6, edge_m=0.058, grasp=True, full=True),),
        density_local_runs=(
            _result(7, edge_m=0.058, grasp=True, full=True),
            _result(8, edge_m=0.060, grasp=True, full=False),
        ),
    )
    json.dumps(records, allow_nan=False)
    by_edge = {round(record["edge_mm"]): record for record in records}
    assert by_edge[58]["stable_grasp_evaluation_count"] == 3
    assert by_edge[58]["stable_grasp_success_count"] == 2
    assert by_edge[58]["stable_grasp_success_rate"] == pytest.approx(2 / 3)
    assert by_edge[58]["fixed_full_success_count"] == 1
    assert by_edge[58]["density_full_success_count"] == 2
    assert by_edge[60]["fixed_full_success_count"] == 0
    assert by_edge[60]["density_full_success_count"] == 0


def test_grasp_gate_never_uses_generic_simulation_pass_as_authorization():
    false_grasp = _result(1, generic_passed=True, full=True)
    stable = _result(2, grasp=True, margin=0.2)
    parents = select_manipulation_parents((false_grasp, stable))
    assert [item["candidate_id"] for item in parents] == [2]

    no_verified = build_manipulation_refinement_candidates(
        (false_grasp,),
        delta_bounds_rad={name: (-0.1, 0.1) for name in ACTIVE_ACTUATORS},
    )
    assert no_verified == []


def test_final_rank_follows_full_margin_perturbation_topology_force_saturation():
    full_tight = _result(
        9,
        grasp=True,
        full=True,
        margin=0.01,
        perturbation_passes=16,
        topology=1.0,
        force=1.0,
        saturation=0.0,
    )
    near_miss = _result(
        1,
        grasp=True,
        full=False,
        margin=10.0,
        perturbation_passes=16,
        topology=1.0,
        force=0.1,
        saturation=0.0,
    )
    assert larger_cube_candidate_rank(full_tight) > larger_cube_candidate_rank(
        near_miss
    )

    high_margin = _result(
        4, grasp=True, full=True, margin=0.2, perturbation_passes=1
    )
    many_perturbations = _result(
        3, grasp=True, full=True, margin=0.1, perturbation_passes=16
    )
    assert larger_cube_candidate_rank(high_margin) > larger_cube_candidate_rank(
        many_perturbations
    )

    high_topology = _result(
        8, grasp=True, full=True, margin=0.1, topology=0.9, force=5.0
    )
    low_topology = _result(
        7, grasp=True, full=True, margin=0.1, topology=0.8, force=0.1
    )
    assert larger_cube_candidate_rank(high_topology) > larger_cube_candidate_rank(
        low_topology
    )
    low_force = _result(
        6, grasp=True, full=True, margin=0.1, topology=0.9, force=1.0
    )
    high_force = _result(
        5, grasp=True, full=True, margin=0.1, topology=0.9, force=2.0
    )
    assert larger_cube_candidate_rank(low_force) > larger_cube_candidate_rank(
        high_force
    )
    low_saturation = _result(
        2,
        grasp=True,
        full=True,
        margin=0.1,
        topology=0.9,
        force=1.0,
        saturation=0.01,
    )
    high_saturation = _result(
        1,
        grasp=True,
        full=True,
        margin=0.1,
        topology=0.9,
        force=1.0,
        saturation=0.02,
    )
    assert larger_cube_candidate_rank(
        low_saturation
    ) > larger_cube_candidate_rank(high_saturation)

    lower_id = _result(
        11, grasp=True, full=True, margin=0.1, topology=0.9, force=1.0
    )
    higher_id = _result(
        12, grasp=True, full=True, margin=0.1, topology=0.9, force=1.0
    )
    assert larger_cube_candidate_rank(lower_id) > larger_cube_candidate_rank(higher_id)

    inputs = [full_tight, near_miss, high_margin, many_perturbations]
    first = [item["candidate_id"] for item in deterministic_rank_results(inputs)]
    random.Random(7).shuffle(inputs)
    second = [item["candidate_id"] for item in deterministic_rank_results(inputs)]
    assert first == second
    with pytest.raises(ValueError, match="unique"):
        deterministic_rank_results((full_tight, copy.deepcopy(full_tight)))


def test_ungrasped_rank_prefers_verify_evidence_over_zero_contact_force():
    zero = _result(1, force=0.0)
    near = _result(9, force=10.0)
    near["summary"]["metrics"].update(
        {
            "verify_effective_finger_count": 0,
            "verify_max_simultaneous_effective_finger_count": 0,
            "verify_target_face_effective_duty": {
                "thumb": 0.0,
                "index": 0.0,
                "mid": 0.0,
            },
            "verify_target_face_simultaneous_duty": 0.0,
            "verify_peak_target_face_force_n": {
                "thumb": 0.02,
                "index": 0.02,
                "mid": 0.02,
            },
            "verify_peak_tactile_n": {
                "thumb": 0.02,
                "index": 0.02,
                "mid": 0.02,
            },
            "verify_max_consecutive_all_gate_steps": 0,
            "verify_all_gate_duty": 0.0,
            "verify_gate_component_duty": {},
        }
    )

    # Old ordering preferred the zero-force candidate and its lower ID.
    assert larger_cube_candidate_rank(near) > larger_cube_candidate_rank(zero)
    assert deterministic_rank_results((zero, near))[0]["candidate_id"] == 9


def test_grasp_refinement_selects_four_per_size_before_hard_grasp_gate():
    results = [
        _result(
            edge_index * 10 + index,
            edge_m=edge,
            grasp=index == 0,
            margin=float(5 - index),
        )
        for edge_index, edge in enumerate((0.058, 0.059))
        for index in range(6)
    ]
    selected = select_grasp_refinement_parents(
        reversed(results), exact_edges_m=(0.058, 0.059)
    )
    assert len(selected) == 8
    assert [item["candidate_id"] for item in selected[:4]] == [0, 1, 2, 3]
    assert [item["candidate_id"] for item in selected[4:]] == [10, 11, 12, 13]


def test_manipulation_delta_sampling_is_relative_bounded_and_deterministic():
    parent = _config()
    grasp_before = copy.deepcopy(parent["control"]["grasp_targets_rad"])
    delta_bounds = {name: (-0.2, 0.2) for name in ACTIVE_ACTUATORS}
    absolute_bounds = {name: (0.65, 0.75) for name in ACTIVE_ACTUATORS}
    first = sample_manipulation_delta_candidates(
        parent,
        count=16,
        seed=123,
        delta_bounds_rad=delta_bounds,
        local_radius_rad=0.08,
        absolute_target_bounds_rad=absolute_bounds,
    )
    second = sample_manipulation_delta_candidates(
        parent,
        count=16,
        seed=123,
        delta_bounds_rad=delta_bounds,
        local_radius_rad=0.08,
        absolute_target_bounds_rad=absolute_bounds,
    )
    assert first == second
    assert parent["control"]["grasp_targets_rad"] == grasp_before
    assert parent["control"]["manipulation_delta_rad"] == _targets(0.0)
    for candidate in first:
        assert candidate["control"]["grasp_targets_rad"] == grasp_before
        assert "final_targets_rad" not in candidate["control"]
        assert all(
            -0.05 <= value <= 0.05
            for value in candidate["control"]["manipulation_delta_rad"].values()
        )
        assert all(
            0.65 <= value <= 0.75
            for value in manipulation_targets(candidate).values()
        )


def test_manipulation_refinement_expands_only_verified_grasps():
    verified = [_result(index, grasp=True, margin=float(index)) for index in range(2)]
    rejected = _result(99, grasp=False, margin=100.0, generic_passed=True)
    candidates = build_manipulation_refinement_candidates(
        (*verified, rejected),
        delta_bounds_rad={name: (-0.1, 0.1) for name in ACTIVE_ACTUATORS},
    )
    assert len(candidates) == 2 * DEFAULT_PLAN.manipulation_refine_per_seed
    assert all(candidate["schema_version"] == 3 for candidate in candidates)


def test_density_transition_requires_explicit_full_success_and_uses_cube_edge():
    passed = [
        _result(index, edge_m=0.052 + index * 0.001, grasp=True, full=True, margin=1.0)
        for index in range(18)
    ]
    failed = _result(100, edge_m=0.064, grasp=True, full=False, margin=100.0)
    converted = constant_density_candidates((*passed, failed))
    assert len(converted) == DEFAULT_PLAN.constant_density_max_candidates
    assert all(
        config["cube"]["mass_kg"]
        == pytest.approx(
            DEFAULT_PLAN.constant_density_mass_kg(config["cube"]["edge_m"])
        )
        for config in converted
    )
    assert failed["config"]["cube"]["mass_kg"] == pytest.approx(0.020)

    generic_only = _result(101, full=False, generic_passed=True)
    assert constant_density_candidates((generic_only,)) == []


@pytest.mark.parametrize(
    ("nominal_edge_m", "expected_mm"),
    [
        (0.052, (52, 53, 54, 55, 56)),
        (0.058, (56, 57, 58, 59, 60)),
        (0.064, (60, 61, 62, 63, 64)),
    ],
)
def test_robustness_edge_window_is_five_integer_sizes(nominal_edge_m, expected_mm):
    assert tuple(
        value * 1000.0 for value in robustness_edge_window_m(nominal_edge_m)
    ) == pytest.approx(expected_mm)


def test_robustness_grid_is_ordered_75_density_plus_25_fixed_mass_cases():
    config = _config(edge_m=0.058, mass_kg=DEFAULT_PLAN.constant_density_mass_kg(0.058))
    original = copy.deepcopy(config)
    validated: list[tuple[float, float, float]] = []

    def validator(case):
        validated.append(
            (
                case["cube"]["edge_m"],
                case["cube"]["mass_kg"],
                case["cube"]["friction"],
            )
        )

    cases, metadata = larger_cube_robustness_cases(config, validator=validator)
    assert config == original
    assert len(cases) == len(metadata) == len(validated) == 100
    assert [item["grid_index"] for item in metadata] == list(range(100))
    assert {item["case_family"] for item in metadata[:75]} == {"constant_density"}
    assert {item["case_family"] for item in metadata[75:]} == {"fixed_20g_control"}
    assert sorted({case["cube"]["edge_m"] for case in cases}) == pytest.approx(
        (0.056, 0.057, 0.058, 0.059, 0.060)
    )
    for case, record in zip(cases[:75], metadata[:75]):
        assert case["cube"]["mass_kg"] == pytest.approx(
            DEFAULT_PLAN.constant_density_mass_kg(record["edge_m"])
            * record["density_scale"]
        )
    assert all(
        case["cube"]["mass_kg"] == pytest.approx(0.020)
        for case in cases[75:]
    )
