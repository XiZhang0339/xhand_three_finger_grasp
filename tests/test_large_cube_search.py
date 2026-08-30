from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xhand_grasp import search
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.experiment import resolve_experiment
from xhand_grasp.large_cube_search import (
    _run_stage,
    adjacent_odd_edges_m,
    diagnose_boundary,
    expand_definition_once,
    static_candidate_advances,
    tune_large_cube,
)
from xhand_grasp.v2_search import (
    KinematicScreenResult,
    _cube_position_in_root,
    _cube_world_position,
    _final_targets_around_pregrasp,
    _local_candidates,
    _rpy_matrix,
    kinematic_screen,
)


ROOT = Path(__file__).resolve().parents[1]
LARGE_CONFIG = ROOT / "grasp_configs" / "left_opposed_face_palm_down_large_cube.json"


@pytest.fixture
def large_config() -> dict[str, Any]:
    return load_config(LARGE_CONFIG)


def _summary(passed: bool, candidate_id: int) -> dict[str, Any]:
    return {
        "passed": passed,
        "failed_checks": [] if passed else ["mock_failure"],
        "checks": {"mock_check": passed},
        "metrics": {
            "peak_total_distal_contact_force_n": 1.0 + candidate_id * 1e-6,
            "actuator_saturation_fraction": 0.0,
            "hold_height_span_m": 0.0,
            "median_lift_m": 0.011 if passed else 0.0,
            "contact_duty": {
                "thumb": 1.0 if passed else 0.0,
                "index": 1.0 if passed else 0.0,
                "mid": 1.0 if passed else 0.0,
            },
            "target_face_contact_duty": {
                "thumb": 1.0,
                "index": 1.0,
                "mid": 1.0,
            },
            "target_face_simultaneous_duty": 1.0,
            "peak_target_face_force_n": {
                "thumb": 1.0,
                "index": 1.0,
                "mid": 1.0,
            },
            "material_off_target_duty": 0.0,
            "material_active_nondistal_duty": 0.0,
            "forbidden_contact_steps": 0,
            "max_palm_down_angle_deg": 15.0,
        },
    }


def _screen(*, emitted: int | None = None, calls: list[dict[str, Any]] | None = None):
    def run(
        config: dict[str, Any],
        *,
        samples_per_pitch: int,
        retain: int,
        seed: int,
        definition: Any,
    ) -> KinematicScreenResult:
        if calls is not None:
            calls.append(
                {
                    "edge_m": config["cube"]["edge_m"],
                    "samples_per_pitch": samples_per_pitch,
                    "retain": retain,
                    "seed": seed,
                }
            )
        count = retain if emitted is None else min(retain, emitted)
        candidates = tuple(copy.deepcopy(config) for _ in range(count))
        diagnostics = tuple(
            {
                "candidate_id": index,
                "score": (float(config["cube"]["edge_m"]),),
                "clean_target_contact_count": 3,
                "near_target_face_count": 3,
                "target_site_signed_distance_m": [0.0, 0.0, 0.0],
                "forbidden_contact": False,
                "max_penetration_m": 0.0,
                "target_faces": config["contact_topology"]["target_faces"],
            }
            for index in range(count)
        )
        return KinematicScreenResult(
            seed=seed,
            sample_count=len(definition.search_bounds.palm_pitch_values_deg)
            * samples_per_pitch,
            retained_count=count,
            candidates=candidates,
            diagnostics=diagnostics,
        )

    return run


def _local_clones(
    parent: dict[str, Any], *, count: int, seed: int, definition: Any
) -> list[dict[str, Any]]:
    del seed, definition
    return [copy.deepcopy(parent) for _ in range(count)]


def _runner(
    predicate,
    *,
    calls: list[tuple[int, tuple[int, ...]]] | None = None,
    reverse: bool = False,
):
    def run(payload: list[tuple[int, dict[str, Any]]], workers: int):
        if calls is not None:
            calls.append((workers, tuple(candidate_id for candidate_id, _ in payload)))
        results = [
            {
                "candidate_id": candidate_id,
                "config": config,
                "summary": _summary(bool(predicate(config)), candidate_id),
            }
            for candidate_id, config in payload
        ]
        if reverse and workers > 1:
            results.reverse()
        return results

    return run


def test_static_gate_and_adjacent_odd_size_generation():
    clean_three = {
        "clean_target_contact_count": 3,
        "near_target_face_count": 0,
        "forbidden_contact": False,
        "max_penetration_m": 0.001,
    }
    clean_two_near_three = {
        "clean_target_contact_count": 2,
        "near_target_face_count": 3,
        "target_site_signed_distance_m": [-0.002, 0.0, 0.003],
        "forbidden_contact": False,
        "max_penetration_m": 0.002,
    }
    assert static_candidate_advances(clean_three, max_penetration_m=0.002)
    assert static_candidate_advances(
        clean_two_near_three, max_penetration_m=0.002
    )
    clean_two_near_three["forbidden_contact"] = True
    assert not static_candidate_advances(
        clean_two_near_three, max_penetration_m=0.002
    )
    del clean_two_near_three["forbidden_contact"]
    del clean_two_near_three["target_site_signed_distance_m"]
    assert not static_candidate_advances(
        clean_two_near_three, max_penetration_m=0.002
    )
    assert adjacent_odd_edges_m(
        (0.036, 0.042, 0.050), lower_m=0.036, upper_m=0.050
    ) == (0.037, 0.041, 0.043, 0.049)


def test_boundary_diagnostic_and_one_shot_expansion_are_named_and_capped(
    large_config,
):
    definition = resolve_experiment(large_config)
    candidate = copy.deepcopy(large_config)
    bounds = definition.search_bounds
    cube_in_root = [
        bounds.cube_position_in_root_m["x"][0] + 0.0002,
        -0.02,
        0.10,
    ]
    candidate["hand_pose"]["translation_m"] = (
        _cube_world_position(candidate)
        - _rpy_matrix(candidate["hand_pose"]["rpy_deg"]) @ cube_in_root
    ).tolist()
    actuator = ACTIVE_ACTUATORS[0]
    candidate["control"]["pregrasp_targets_rad"][actuator] = (
        bounds.actuator_targets_rad[actuator][0] + 0.01
    )
    diagnostic = diagnose_boundary(
        candidate,
        bounds,
        position_tolerance_m=0.001,
        actuator_tolerance_rad=0.03,
    )
    assert diagnostic["near_boundary"]
    assert {(hit["axis"], hit["side"]) for hit in diagnostic["position_hits"]} == {
        ("x", "lower")
    }
    assert any(
        hit["actuator"] == actuator and hit["side"] == "lower"
        for hit in diagnostic["actuator_hits"]
    )

    hard_limits = {name: (-1.0, 2.0) for name in ACTIVE_ACTUATORS}
    hard_limits[actuator] = (0.10, 2.0)
    expanded, report = expand_definition_once(
        definition, diagnostic, actuator_hard_limits=hard_limits
    )
    assert report["expansion_count"] == 1
    assert expanded.search_bounds.cube_position_in_root_m["x"][0] == pytest.approx(
        bounds.cube_position_in_root_m["x"][0] - 0.005
    )
    # 0.15 - 0.1 would cross the model lower limit, so it is capped at 0.10.
    assert expanded.search_bounds.actuator_targets_rad[actuator][0] == 0.10
    assert any(
        change.get("capped") is True
        for change in report["changes"]
        if change["kind"] == "actuator" and change["name"] == actuator
    )


def test_large_screen_seed_preserves_the_declared_relative_pose(large_config):
    screen = kinematic_screen(
        copy.deepcopy(large_config), samples_per_pitch=1, retain=1, seed=20260821
    )
    seed_index = next(
        index
        for index, diagnostic in enumerate(screen.diagnostics)
        if diagnostic["candidate_id"] == -1
    )
    seed = screen.candidates[seed_index]
    assert seed["hand_pose"]["translation_m"] == pytest.approx(
        large_config["hand_pose"]["translation_m"]
    )
    assert seed["hand_pose"]["rpy_deg"] == pytest.approx(
        large_config["hand_pose"]["rpy_deg"]
    )


def test_large_campaign_samples_final_targets_over_full_absolute_ranges(
    large_config,
):
    definition = resolve_experiment(large_config)
    assert definition.search_bounds.final_target_delta_rad is None
    pregrasp = {
        name: sum(definition.search_bounds.actuator_targets_rad[name]) / 2.0
        for name in ACTIVE_ACTUATORS
    }
    lower = _final_targets_around_pregrasp(
        pregrasp, np.zeros(len(ACTIVE_ACTUATORS)), definition=definition
    )
    upper = _final_targets_around_pregrasp(
        pregrasp, np.ones(len(ACTIVE_ACTUATORS)), definition=definition
    )
    for name in ACTIVE_ACTUATORS:
        assert (lower[name], upper[name]) == pytest.approx(
            definition.search_bounds.actuator_targets_rad[name]
        )


def test_large_local_sampling_fills_budget_at_search_boundaries(large_config):
    definition = resolve_experiment(large_config)
    bounds = definition.search_bounds
    parent = copy.deepcopy(large_config)
    parent["hand_pose"]["rpy_deg"] = [
        bounds.hand_roll_deg[0],
        bounds.palm_pitch_deg[0],
        bounds.hand_yaw_deg[0],
    ]
    cube_in_root = np.asarray(
        [bounds.cube_position_in_root_m[axis][0] for axis in ("x", "y", "z")]
    )
    parent["cube"]["rpy_deg"][2] = bounds.cube_yaw_deg[0]
    parent["hand_pose"]["translation_m"] = (
        _cube_world_position(parent)
        - _rpy_matrix(parent["hand_pose"]["rpy_deg"]) @ cube_in_root
    ).tolist()
    for phase in ("pregrasp_targets_rad", "final_targets_rad"):
        for name in ACTIVE_ACTUATORS:
            parent["control"][phase][name] = bounds.actuator_targets_rad[name][0]

    candidates = _local_candidates(
        parent, count=128, seed=1, definition=definition
    )
    assert len(candidates) == 128
    assert all(
        bounds.contains_cube_position(_cube_position_in_root(candidate))
        for candidate in candidates
    )


def test_declared_campaign_requests_512_fixed_mass_dynamics(large_config):
    screen_calls: list[dict[str, Any]] = []
    run_calls: list[tuple[int, tuple[int, ...]]] = []
    result = tune_large_cube(
        copy.deepcopy(large_config),
        workers=3,
        seed=20260821,
        run_candidates=_runner(lambda _config: False, calls=run_calls),
        rank_candidate=search.candidate_rank,
        perturb_cases=None,
        kinematic_samples_per_pitch=1,
        local_refine_seed_count=2,
        local_refine_per_seed=1,
        final_candidate_count=1,
        fallback_physics_count=0,
        fallback_kinematic_samples_per_pitch=1,
        screen_candidates=_screen(calls=screen_calls),
        local_candidate_factory=_local_clones,
    )

    assert result["size_campaign_diagnostics"]["effective_budget"][
        "dynamic_candidate_count"
    ] == 512
    assert result["fixed_mass_dynamic_count"] == 512
    assert result["fixed_mass_local_refinement_count"] == 2
    assert result["constant_density_reevaluation_count"] == 0
    assert result["fixed_mass_success"] is False
    assert result["constant_density_success"] is False
    assert result["stop_reason"] == "no_fixed_mass_hard_pass"
    assert len(result["size_stages"]["coarse"]) == 8
    assert len(result["size_stages"]["fine"]) == 2
    assert all(call["retain"] >= 256 for call in screen_calls)
    assert [call["retain"] for call in screen_calls[-2:]] == [1024, 1024]
    assert run_calls[0][0] == 3 and len(run_calls[0][1]) == 512
    assert all(
        record["dynamic_selection"]["shortfall_count"] == 0
        for record in result["size_stages"]["fine"]
    )


def test_stage_runner_must_preserve_exact_id_to_config_binding(large_config):
    first = copy.deepcopy(large_config)
    second = copy.deepcopy(large_config)
    second["cube"]["edge_m"] = 0.041

    def swapped(payload, workers):
        del workers
        return [
            {
                "candidate_id": payload[0][0],
                "config": payload[1][1],
                "summary": _summary(False, payload[0][0]),
            },
            {
                "candidate_id": payload[1][0],
                "config": payload[0][1],
                "summary": _summary(False, payload[1][0]),
            },
        ]

    with pytest.raises(RuntimeError, match="rebound candidate_id"):
        _run_stage(
            [first, second],
            next_id=0,
            workers=2,
            run_candidates=swapped,
            stage="contract_test",
            material_policy="fixed_20g_control",
        )


def test_screen_contract_rejects_mismatched_candidates_and_diagnostics(
    large_config,
):
    def broken_screen(config, **kwargs):
        del kwargs
        return KinematicScreenResult(
            seed=1,
            sample_count=1,
            retained_count=1,
            candidates=(copy.deepcopy(config),),
            diagnostics=(),
        )

    with pytest.raises(RuntimeError, match="inconsistent retention"):
        tune_large_cube(
            copy.deepcopy(large_config),
            workers=1,
            seed=3,
            run_candidates=_runner(lambda _config: False),
            rank_candidate=search.candidate_rank,
            perturb_cases=None,
            kinematic_samples_per_pitch=1,
            dynamic_candidate_count=2,
            local_refine_seed_count=1,
            local_refine_per_seed=1,
            final_candidate_count=1,
            fallback_physics_count=0,
            fallback_kinematic_samples_per_pitch=1,
            screen_candidates=broken_screen,
            local_candidate_factory=_local_clones,
        )


def test_dynamic_gate_shortfall_is_reported_per_exact_size(large_config):
    result = tune_large_cube(
        copy.deepcopy(large_config),
        workers=1,
        seed=5,
        run_candidates=_runner(lambda _config: False),
        rank_candidate=search.candidate_rank,
        perturb_cases=None,
        kinematic_samples_per_pitch=1,
        dynamic_candidate_count=6,
        local_refine_seed_count=1,
        local_refine_per_seed=1,
        final_candidate_count=1,
        fallback_physics_count=0,
        fallback_kinematic_samples_per_pitch=1,
        screen_candidates=_screen(emitted=2),
        local_candidate_factory=_local_clones,
    )

    assert result["fixed_mass_dynamic_count"] == 4
    for record in result["size_stages"]["fine"]:
        selection = record["dynamic_selection"]
        assert selection == {
            "requested_count": 3,
            "eligible_available_count": 2,
            "selected_count": 2,
            "shortfall_count": 1,
            "underfilled_reason": "insufficient_candidates_meeting_static_gate",
        }

def test_fixed_mass_pass_is_not_misreported_as_density_validation(large_config):
    campaign = resolve_experiment(large_config).size_campaign
    assert campaign is not None
    result = tune_large_cube(
        copy.deepcopy(large_config),
        workers=1,
        seed=7,
        run_candidates=_runner(
            lambda candidate: candidate["cube"]["mass_kg"]
            == pytest.approx(campaign.discovery_mass_kg)
        ),
        rank_candidate=search.candidate_rank,
        perturb_cases=None,
        kinematic_samples_per_pitch=1,
        dynamic_candidate_count=2,
        local_refine_seed_count=2,
        local_refine_per_seed=1,
        final_candidate_count=1,
        fallback_physics_count=2,
        fallback_kinematic_samples_per_pitch=1,
        screen_candidates=_screen(emitted=2),
        # Avoid spending the production 4x128 density refinement budget in a
        # policy test; the stage remains present with zero mocked candidates.
        local_candidate_factory=lambda parent, **kwargs: (
            _local_clones(parent, **kwargs)
            if kwargs["count"] == 1
            else []
        ),
    )

    assert result["fixed_mass_success"] is True
    assert result["constant_density_success"] is False
    assert result["campaign_classification"] == (
        "validated_fixed_mass_ablation_only"
    )
    assert result["best_fixed_mass"]["summary"]["passed"] is True
    assert result["best"]["summary"]["passed"] is False
    assert result["best"]["material_policy"] == "constant_density"
    assert result["best"]["config"]["experiment_status"]["classification"] == (
        "best_constant_density_near_miss"
    )
    edge = result["best"]["config"]["cube"]["edge_m"]
    assert result["best"]["config"]["cube"]["mass_kg"] == pytest.approx(
        campaign.constant_density_mass_kg(edge)
    )


def test_disabled_density_stage_keeps_fixed_mass_classification(large_config):
    result = tune_large_cube(
        copy.deepcopy(large_config),
        workers=1,
        seed=11,
        run_candidates=_runner(lambda _candidate: True),
        rank_candidate=search.candidate_rank,
        perturb_cases=None,
        kinematic_samples_per_pitch=1,
        dynamic_candidate_count=2,
        local_refine_seed_count=1,
        local_refine_per_seed=1,
        final_candidate_count=1,
        fallback_physics_count=0,
        fallback_kinematic_samples_per_pitch=1,
        screen_candidates=_screen(emitted=2),
        local_candidate_factory=_local_clones,
    )

    assert result["fixed_mass_success"] is True
    assert result["constant_density_success"] is False
    assert result["best"]["material_policy"] == "fixed_20g_control"
    assert result["best"]["config"]["experiment_status"]["classification"] == (
        "validated_fixed_mass_ablation"
    )
    assert result["stop_reason"] == "constant_density_stage_disabled"


def test_zero_initial_simultaneous_topology_duty_stops_local_refinement(
    large_config,
):
    def no_topology_runner(payload, workers):
        del workers
        results = []
        for candidate_id, config in payload:
            summary = _summary(False, candidate_id)
            summary["metrics"]["target_face_simultaneous_duty"] = 0.0
            summary["metrics"]["target_face_contact_duty"] = {
                "thumb": 0.0,
                "index": 0.0,
                "mid": 0.0,
            }
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": config,
                    "summary": summary,
                }
            )
        return results

    result = tune_large_cube(
        copy.deepcopy(large_config),
        workers=1,
        seed=13,
        run_candidates=no_topology_runner,
        rank_candidate=search.candidate_rank,
        perturb_cases=None,
        kinematic_samples_per_pitch=1,
        dynamic_candidate_count=2,
        local_refine_seed_count=2,
        local_refine_per_seed=1,
        final_candidate_count=1,
        fallback_physics_count=0,
        fallback_kinematic_samples_per_pitch=1,
        screen_candidates=_screen(emitted=2),
        local_candidate_factory=lambda *args, **kwargs: pytest.fail(
            "local refinement must not run without topology signal"
        ),
    )

    assert result["initial_three_finger_topology_signal_count"] == 0
    assert result["fixed_mass_refinement_continued"] is False
    assert result["fixed_mass_local_refinement_count"] == 0
    assert result["stop_reason"] == "no_initial_local_refinement_signal"


def test_simulation_errors_do_not_unlock_local_refinement(large_config):
    def error_runner(payload, workers):
        del workers
        results = []
        for candidate_id, config in payload:
            summary = _summary(False, candidate_id)
            summary["failed_checks"] = ["simulation_error"]
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": config,
                    "summary": summary,
                }
            )
        return results

    result = tune_large_cube(
        copy.deepcopy(large_config),
        workers=1,
        seed=17,
        run_candidates=error_runner,
        rank_candidate=search.candidate_rank,
        perturb_cases=None,
        kinematic_samples_per_pitch=1,
        dynamic_candidate_count=2,
        local_refine_seed_count=2,
        local_refine_per_seed=1,
        final_candidate_count=1,
        fallback_physics_count=0,
        fallback_kinematic_samples_per_pitch=1,
        screen_candidates=_screen(emitted=2),
        local_candidate_factory=lambda *args, **kwargs: pytest.fail(
            "local refinement must not run after simulation errors"
        ),
    )

    assert result["fixed_mass_dynamic_simulation_error_count"] == 2
    assert result["initial_three_finger_topology_signal_count"] == 0
    assert result["fixed_mass_refinement_continued"] is False
    assert result["stop_reason"] == "all_fixed_mass_dynamics_failed_to_simulate"
    assert result["best"]["config"]["experiment_status"]["classification"] == (
        "fixed_mass_simulation_error"
    )


def test_density_simulation_errors_do_not_unlock_density_refinement(
    large_config,
):
    campaign = resolve_experiment(large_config).size_campaign
    assert campaign is not None

    def material_runner(payload, workers):
        del workers
        results = []
        for candidate_id, config in payload:
            is_fixed = config["cube"]["mass_kg"] == pytest.approx(
                campaign.discovery_mass_kg
            )
            summary = _summary(is_fixed, candidate_id)
            if not is_fixed:
                summary["failed_checks"] = ["simulation_error"]
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": config,
                    "summary": summary,
                }
            )
        return results

    result = tune_large_cube(
        copy.deepcopy(large_config),
        workers=1,
        seed=19,
        run_candidates=material_runner,
        rank_candidate=search.candidate_rank,
        perturb_cases=None,
        kinematic_samples_per_pitch=1,
        dynamic_candidate_count=2,
        local_refine_seed_count=1,
        local_refine_per_seed=1,
        final_candidate_count=1,
        fallback_physics_count=2,
        fallback_kinematic_samples_per_pitch=1,
        screen_candidates=_screen(emitted=2),
        local_candidate_factory=_local_clones,
    )

    assert result["fixed_mass_success"] is True
    assert result["constant_density_reevaluation_count"] == 2
    assert result["constant_density_reevaluation_simulation_error_count"] == 2
    assert result["constant_density_local_refinement_count"] == 0
    assert result["stop_reason"] == (
        "all_constant_density_reevaluations_failed_to_simulate"
    )
    assert result["best"]["config"]["experiment_status"]["classification"] == (
        "constant_density_simulation_error"
    )


def test_density_success_and_selection_are_worker_order_independent(large_config):
    campaign = resolve_experiment(large_config).size_campaign
    assert campaign is not None

    def run(workers: int):
        return tune_large_cube(
            copy.deepcopy(large_config),
            workers=workers,
            seed=91,
            run_candidates=_runner(lambda _candidate: True, reverse=True),
            rank_candidate=search.candidate_rank,
            perturb_cases=lambda candidate, *, count, seed: [
                copy.deepcopy(candidate) for _ in range(count)
            ],
            kinematic_samples_per_pitch=1,
            dynamic_candidate_count=2,
            local_refine_seed_count=2,
            local_refine_per_seed=1,
            final_candidate_count=1,
            perturbations_per_final=1,
            fallback_physics_count=2,
            fallback_kinematic_samples_per_pitch=1,
            screen_candidates=_screen(emitted=2),
            local_candidate_factory=_local_clones,
        )

    serial = run(1)
    parallel = run(8)
    assert serial["constant_density_success"] is True
    assert serial["campaign_classification"] == "validated_constant_density"
    assert serial["best"]["config"]["experiment_status"]["passed"] is True
    assert serial["best_fixed_mass"]["config"]["experiment_status"][
        "constant_density_passed"
    ] is False
    assert serial["best_fixed_mass"]["config"]["experiment_status"][
        "campaign_constant_density_passed"
    ] is True
    assert serial["best"]["config"]["cube"]["mass_kg"] == pytest.approx(
        campaign.constant_density_mass_kg(
            serial["best"]["config"]["cube"]["edge_m"]
        )
    )
    assert serial["best"]["candidate_id"] == parallel["best"]["candidate_id"]
    assert [item["candidate_id"] for item in serial["top_candidates"]] == [
        item["candidate_id"] for item in parallel["top_candidates"]
    ]
    assert serial["size_boundary_probe_count"] > 0
    assert serial["size_boundary_probes"] == parallel["size_boundary_probes"]
    assert serial["local_perturbation_probes"][0][
        "parent_material_policy"
    ] == "constant_density"
    assert all(
        trial["material_policy"] == "local_material_and_pose_perturbation"
        and trial["parent_material_policy"] == "constant_density"
        for trial in serial["local_perturbation_probes"][0]["trials"]
    )
    assert all(
        record["step_m"] == pytest.approx(0.0005)
        for record in serial["size_boundary_probes"]
    )


def test_finalist_perturbation_count_is_a_hard_contract(large_config):
    with pytest.raises(RuntimeError, match="must return exactly 2 cases"):
        tune_large_cube(
            copy.deepcopy(large_config),
            workers=1,
            seed=23,
            run_candidates=_runner(lambda _candidate: True),
            rank_candidate=search.candidate_rank,
            perturb_cases=lambda candidate, *, count, seed: [
                copy.deepcopy(candidate)
            ],
            kinematic_samples_per_pitch=1,
            dynamic_candidate_count=2,
            local_refine_seed_count=1,
            local_refine_per_seed=1,
            final_candidate_count=1,
            perturbations_per_final=2,
            fallback_physics_count=1,
            fallback_kinematic_samples_per_pitch=1,
            screen_candidates=_screen(emitted=2),
            local_candidate_factory=_local_clones,
        )


def test_public_tune_dispatches_to_large_campaign(large_config, monkeypatch):
    sentinel = {"best": {"config": large_config}}
    import xhand_grasp.large_cube_search as large_search

    monkeypatch.setattr(large_search, "tune_large_cube", lambda config, **kwargs: sentinel)
    result = search.tune(
        copy.deepcopy(large_config),
        samples=0,
        refine_top=0,
        refine_per=0,
        workers=1,
        seed=3,
    )
    assert result is sentinel
