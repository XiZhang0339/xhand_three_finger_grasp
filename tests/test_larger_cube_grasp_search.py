from __future__ import annotations

import copy
import random

import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS
from xhand_grasp.larger_cube_grasp_search import (
    DEFAULT_PLAN,
    SizeScreenSummary,
    adjacent_neighbor_edges_m,
    build_manipulation_refinement_candidates,
    campaign_manifest,
    coarse_screen_jobs,
    constant_density_candidates,
    deterministic_rank_results,
    exact_screen_jobs,
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
    assert manifest["hard_stage_gates"] == {
        "manipulation_requires_grasp_success": True,
        "constant_density_requires_fixed_mass_full_success": True,
        "campaign_success_requires_constant_density_full_success": True,
    }


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

    inputs = [full_tight, near_miss, high_margin, many_perturbations]
    first = [item["candidate_id"] for item in deterministic_rank_results(inputs)]
    random.Random(7).shuffle(inputs)
    second = [item["candidate_id"] for item in deterministic_rank_results(inputs)]
    assert first == second
    with pytest.raises(ValueError, match="unique"):
        deterministic_rank_results((full_tight, copy.deepcopy(full_tight)))


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
