from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from xhand_grasp import cli, search
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.experiment import opposite_face, resolve_experiment
from xhand_grasp.v2_search import (
    FINAL_TARGET_DELTA_BOUNDS_RAD,
    KinematicScreenResult,
    _candidate_parameters,
    _local_candidates,
    _materialize_candidate,
    kinematic_screen,
    palm_down_angle_for_rpy,
    tune_opposed_face,
)
import xhand_grasp.v2_search as v2_search


ROOT = Path(__file__).resolve().parents[1]
V2_CONFIG = ROOT / "grasp_configs" / "left_opposed_face_palm_down.json"


@pytest.fixture
def v2_config() -> dict[str, Any]:
    return load_config(V2_CONFIG)


def _summary(passed: bool, candidate_id: int) -> dict[str, Any]:
    return {
        "passed": passed,
        "failed_checks": [] if passed else ["mock_failure"],
        "checks": {"mock_check": passed},
        "metrics": {
            "peak_total_distal_contact_force_n": 1.0 + candidate_id * 0.001,
            "actuator_saturation_fraction": 0.0,
            "hold_height_span_m": 0.0,
            "median_lift_m": 0.011 if passed else 0.0,
            "contact_duty": {
                "thumb": 1.0 if passed else 0.0,
                "index": 1.0 if passed else 0.0,
                "mid": 1.0 if passed else 0.0,
            },
            "forbidden_contact_steps": 0,
            "max_palm_down_angle_deg": 15.0,
        },
    }


def _runner(
    passed: bool | Callable[[int, dict[str, Any]], bool],
    *,
    reverse_for_parallel_workers: bool = False,
    calls: list[tuple[int, tuple[int, ...]]] | None = None,
):
    def run(
        payload: list[tuple[int, dict[str, Any]]], workers: int
    ) -> list[dict[str, Any]]:
        if calls is not None:
            calls.append((workers, tuple(candidate_id for candidate_id, _ in payload)))
        results = []
        for candidate_id, config in payload:
            candidate_passed = (
                passed(candidate_id, config) if callable(passed) else passed
            )
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": config,
                    "summary": _summary(bool(candidate_passed), candidate_id),
                }
            )
        if reverse_for_parallel_workers and workers > 1:
            results.reverse()
        return results

    return run


def _screen_with_clones(
    config: dict[str, Any], count: int, *, seed: int = 20260821
) -> KinematicScreenResult:
    return KinematicScreenResult(
        seed=seed,
        sample_count=5,
        retained_count=count,
        candidates=tuple(copy.deepcopy(config) for _ in range(count)),
        diagnostics=tuple(
            {"candidate_id": index, "palm_down_angle_deg": 15.0}
            for index in range(count)
        ),
    )


def _patch_cheap_stages(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any], *, screen_count: int
) -> None:
    monkeypatch.setattr(
        v2_search,
        "kinematic_screen",
        lambda candidate_config, **kwargs: _screen_with_clones(
            candidate_config,
            int(kwargs.get("retain", screen_count)),
            seed=int(kwargs.get("seed", 20260821)),
        ),
    )

    def local_candidates(
        parent: dict[str, Any],
        *,
        count: int,
        seed: int,
        definition: Any,
    ) -> list[dict[str, Any]]:
        del seed, definition
        return [copy.deepcopy(parent) for _ in range(count)]

    monkeypatch.setattr(v2_search, "_local_candidates", local_candidates)


def test_small_kinematic_screen_is_fixed_seed_deterministic(v2_config):
    first = kinematic_screen(
        copy.deepcopy(v2_config), samples_per_pitch=2, retain=5, seed=9173
    )
    second = kinematic_screen(
        copy.deepcopy(v2_config), samples_per_pitch=2, retain=5, seed=9173
    )

    assert first.seed == second.seed == 9173
    assert first.sample_count == second.sample_count == 10
    assert first.retained_count == second.retained_count == 5
    assert first.candidates == second.candidates
    assert first.diagnostics == second.diagnostics


def test_terminal_pose_is_a_bounded_offset_from_sampled_pregrasp(v2_config):
    definition = resolve_experiment(v2_config)
    dimensions = 6 + 2 * len(ACTIVE_ACTUATORS)
    row = np.linspace(0.03, 0.97, dimensions, dtype=np.float64)

    parameters = _candidate_parameters(
        row,
        pitch_deg=75.0,
        candidate_id=4,
        definition=definition,
    )

    for name in ACTIVE_ACTUATORS:
        lower, upper = definition.search_bounds.actuator_targets_rad[name]
        pregrasp = parameters["pregrasp_targets_rad"][name]
        final = parameters["final_targets_rad"][name]
        delta_lower, delta_upper = FINAL_TARGET_DELTA_BOUNDS_RAD[name]
        assert lower <= pregrasp <= upper
        assert lower <= final <= upper
        # Search-bound clipping may shorten, but can never enlarge, the
        # intended terminal displacement.
        assert min(0.0, delta_lower) - 1e-12 <= final - pregrasp
        assert final - pregrasp <= max(0.0, delta_upper) + 1e-12


def test_terminal_sampling_depends_on_pregrasp_not_an_absolute_second_pose(
    v2_config,
):
    definition = resolve_experiment(v2_config)
    dimensions = 6 + 2 * len(ACTIVE_ACTUATORS)
    low = np.full(dimensions, 0.25, dtype=np.float64)
    high = low.copy()
    offset_start = 6 + len(ACTIVE_ACTUATORS)
    high[offset_start:] = 0.75

    low_parameters = _candidate_parameters(
        low, pitch_deg=70.0, candidate_id=0, definition=definition
    )
    high_parameters = _candidate_parameters(
        high, pitch_deg=70.0, candidate_id=0, definition=definition
    )

    assert low_parameters["pregrasp_targets_rad"] == high_parameters[
        "pregrasp_targets_rad"
    ]
    for index, name in enumerate(ACTIVE_ACTUATORS):
        low_delta = (
            low_parameters["final_targets_rad"][name]
            - low_parameters["pregrasp_targets_rad"][name]
        )
        high_delta = (
            high_parameters["final_targets_rad"][name]
            - high_parameters["pregrasp_targets_rad"][name]
        )
        delta_lower, delta_upper = FINAL_TARGET_DELTA_BOUNDS_RAD[name]
        target_lower, target_upper = definition.search_bounds.actuator_targets_rad[
            name
        ]
        pregrasp = low_parameters["pregrasp_targets_rad"][name]
        expected_low = min(
            target_upper,
            max(
                target_lower,
                pregrasp
                + delta_lower
                + low[offset_start + index] * (delta_upper - delta_lower),
            ),
        )
        expected_high = min(
            target_upper,
            max(
                target_lower,
                pregrasp
                + delta_lower
                + high[offset_start + index] * (delta_upper - delta_lower),
            ),
        )
        assert low_parameters["final_targets_rad"][name] == pytest.approx(
            expected_low
        )
        assert high_parameters["final_targets_rad"][name] == pytest.approx(
            expected_high
        )
        assert low_delta == pytest.approx(expected_low - pregrasp)
        assert high_delta == pytest.approx(expected_high - pregrasp)
        assert high_delta >= low_delta - 1e-12


def test_local_refinement_keeps_terminal_pose_near_pregrasp_and_is_deterministic(
    v2_config,
):
    definition = resolve_experiment(v2_config)
    parent = copy.deepcopy(v2_config)
    parent["control"]["pregrasp_targets_rad"] = copy.deepcopy(
        parent["control"]["final_targets_rad"]
    )

    first = _local_candidates(parent, count=5, seed=144, definition=definition)
    second = _local_candidates(parent, count=5, seed=144, definition=definition)

    assert first == second
    for candidate in first:
        bounds = definition.search_bounds
        roll, pitch, yaw = candidate["hand_pose"]["rpy_deg"]
        assert bounds.hand_roll_deg[0] <= roll <= bounds.hand_roll_deg[1]
        assert bounds.contains_pitch(pitch)
        assert bounds.hand_yaw_deg[0] <= yaw <= bounds.hand_yaw_deg[1]
        assert (
            bounds.cube_yaw_deg[0]
            <= candidate["cube"]["rpy_deg"][2]
            <= bounds.cube_yaw_deg[1]
        )
        assert bounds.contains_cube_position(
            v2_search._cube_position_in_root(candidate)
        )
        assert v2_search._pose_is_within_search_bounds(candidate, definition)
        for name in ACTIVE_ACTUATORS:
            lower, upper = bounds.actuator_targets_rad[name]
            pregrasp = candidate["control"]["pregrasp_targets_rad"][name]
            final = candidate["control"]["final_targets_rad"][name]
            delta_lower, delta_upper = FINAL_TARGET_DELTA_BOUNDS_RAD[name]
            assert lower <= pregrasp <= upper
            assert lower <= final <= upper
            assert min(0.0, delta_lower) - 1e-12 <= final - pregrasp
            assert final - pregrasp <= max(0.0, delta_upper) + 1e-12


def test_fallback_material_sampling_preserves_size_specific_geometry(v2_config):
    parents = [copy.deepcopy(v2_config) for _ in range(2)]
    for index, parent in enumerate(parents):
        parent["cube"]["edge_m"] = 0.028
        parent["hand_pose"]["translation_m"][0] += index * 0.001

    first = v2_search._physics_fallback_candidates(parents, count=5, seed=718)
    second = v2_search._physics_fallback_candidates(parents, count=5, seed=718)

    assert first == second
    assert all(candidate["cube"]["edge_m"] == 0.028 for candidate in first)
    assert first[0]["cube"]["mass_kg"] == v2_config["cube"]["mass_kg"]
    assert first[0]["cube"]["friction"] == v2_config["cube"]["friction"]
    assert all(0.010 <= candidate["cube"]["mass_kg"] <= 0.030 for candidate in first)
    assert all(0.4 <= candidate["cube"]["friction"] <= 1.2 for candidate in first)

    nominal_size_singleton = v2_search._physics_fallback_candidates(
        [copy.deepcopy(v2_config)],
        count=1,
        seed=718,
        ensure_non_nominal_material=True,
    )
    assert not v2_search._has_nominal_cube_physics(
        nominal_size_singleton[0], v2_config
    )


def test_kinematic_screen_always_retains_declared_near_miss_seed(v2_config):
    screen = kinematic_screen(
        copy.deepcopy(v2_config), samples_per_pitch=1, retain=1, seed=123
    )

    assert screen.sample_count == 5
    assert screen.retained_count == 1
    assert screen.diagnostics[0]["candidate_id"] == -1
    retained = screen.candidates[0]
    assert retained["experiment_status"]["classification"] == "initial_near_miss"
    assert retained["contact_topology"]["target_faces"] == v2_config[
        "contact_topology"
    ]["target_faces"]
    assert retained["control"]["final_targets_rad"] == v2_config["control"][
        "final_targets_rad"
    ]


@pytest.mark.parametrize(
    ("rpy_deg", "expected_deg"),
    [
        ([0.0, 20.0, 0.0], 70.0),
        ([0.0, 60.0, 0.0], 30.0),
        ([0.0, 75.0, 0.0], 15.0),
        ([12.0, 90.0, -31.0], 0.0),
        ([0.0, -90.0, 0.0], 180.0),
    ],
)
def test_palm_down_angle_for_rpy_uses_local_positive_x(
    rpy_deg, expected_deg
):
    assert palm_down_angle_for_rpy(rpy_deg) == pytest.approx(expected_deg, abs=1e-10)


def test_all_four_horizontal_opposed_pairs_materialize_valid_configs(v2_config):
    definition = resolve_experiment(v2_config)
    shared_faces = tuple(assignment.index for assignment in definition.candidate_faces)
    assert shared_faces == ("+X", "-X", "+Y", "-Y")

    base_parameters = {
        "hand_rpy_deg": list(v2_config["hand_pose"]["rpy_deg"]),
        "cube_in_root_m": list(
            v2_config["initial_near_miss"]["cube_position_in_root_m"]
        ),
        "cube_yaw_deg": float(v2_config["cube"]["rpy_deg"][2]),
        "final_targets_rad": {
            name: float(v2_config["control"]["final_targets_rad"][name])
            for name in ACTIVE_ACTUATORS
        },
    }
    materialized = []
    for candidate_id, assignment in enumerate(definition.candidate_faces):
        parameters = {
            **base_parameters,
            "candidate_id": candidate_id,
            "target_assignment": assignment,
        }
        candidate = _materialize_candidate(copy.deepcopy(v2_config), parameters)
        validate_config(candidate)
        target = candidate["contact_topology"]["target_faces"]
        assert target["index"] == target["mid"] == assignment.index
        assert target["thumb"] == opposite_face(assignment.index)
        materialized.append(target)
    assert len(materialized) == 4


def test_public_tune_dispatches_schema_v2_with_mock_runner(
    v2_config, monkeypatch
):
    calls: list[tuple[int, tuple[int, ...]]] = []
    monkeypatch.setattr(search, "_run_candidates", _runner(True, calls=calls))

    result = search.tune(
        copy.deepcopy(v2_config),
        samples=0,
        refine_top=0,
        refine_per=0,
        workers=3,
        seed=20260821,
        kinematic_samples_per_pitch=1,
        dynamic_candidate_count=1,
        local_refine_seed_count=1,
        local_refine_per_seed=1,
        final_candidate_count=1,
        perturbations_per_final=1,
        fallback_physics_count=0,
        fallback_kinematic_samples_per_pitch=2,
    )

    assert result["experiment_id"] == v2_config["experiment_id"]
    assert result["kinematic_sample_count"] == 5
    assert result["initial_dynamic_count"] == 1
    assert result["local_refinement_count"] == 1
    assert result["fallback_physics_count"] == 0
    assert result["fallback_kinematic_samples_per_pitch"] == 2
    assert result["nominal_success"] is True
    assert calls and all(workers == 3 for workers, _ in calls)


def test_tune_cli_parses_explicit_fallback_budget_overrides():
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--fallback-physics-candidates",
            "37",
            "--fallback-kinematic-samples-per-pitch",
            "19",
        ]
    )

    assert args.fallback_physics_candidates == 37
    assert args.fallback_kinematic_samples_per_pitch == 19


def test_physics_fallback_labels_nominal_and_alternative_success(
    v2_config, monkeypatch
):
    _patch_cheap_stages(monkeypatch, v2_config, screen_count=1)

    nominal = tune_opposed_face(
        copy.deepcopy(v2_config),
        workers=1,
        seed=11,
        run_candidates=_runner(True),
        rank_candidate=search.candidate_rank,
        dynamic_candidate_count=1,
        local_refine_seed_count=1,
        local_refine_per_seed=1,
        final_candidate_count=1,
        perturbations_per_final=1,
        fallback_physics_count=2,
        perturb_cases=None,
    )
    assert nominal["nominal_success"] is True
    assert nominal["alternative_physics_success"] is False
    assert nominal["fallback_physics_count"] == 0
    assert nominal["best"]["config"]["experiment_status"] == {
        "classification": "validated_nominal",
        "passed": True,
        "nominal_physics_passed": True,
        "note": "All declared hard constraints passed.",
    }

    stage = 0

    def alternative_runner(
        payload: list[tuple[int, dict[str, Any]]], workers: int
    ) -> list[dict[str, Any]]:
        nonlocal stage
        del workers
        stage += 1
        # Initial and local stages fail; the third call is physics fallback.
        return [
            {
                "candidate_id": candidate_id,
                "config": config,
                "summary": _summary(
                    stage >= 3
                    and not v2_search._has_nominal_cube_physics(
                        config, v2_config
                    ),
                    candidate_id,
                ),
            }
            for candidate_id, config in payload
        ]

    alternative = tune_opposed_face(
        copy.deepcopy(v2_config),
        workers=4,
        seed=11,
        run_candidates=alternative_runner,
        rank_candidate=search.candidate_rank,
        dynamic_candidate_count=1,
        local_refine_seed_count=1,
        local_refine_per_seed=1,
        final_candidate_count=1,
        perturbations_per_final=1,
        fallback_physics_count=8,
        perturb_cases=None,
    )
    assert alternative["nominal_success"] is False
    assert alternative["alternative_physics_success"] is True
    assert alternative["fallback_physics_count"] == 8
    assert alternative["best"]["search_stage"] == "alternative_physics"
    geometry_screens = alternative["alternative_geometry_screens"]
    assert [record["edge_m"] for record in geometry_screens] == [
        0.026,
        0.028,
        0.03,
        0.032,
        0.034,
    ]
    assert [record["dynamic_candidate_count"] for record in geometry_screens] == [
        2,
        2,
        2,
        1,
        1,
    ]
    assert [record["seed"] for record in geometry_screens] == [
        20_011,
        1_020_014,
        2_020_017,
        3_020_020,
        4_020_023,
    ]
    fallback_edges = [
        item["config"]["cube"]["edge_m"]
        for item in alternative["top_candidates"]
        if item.get("search_stage") == "alternative_physics"
    ]
    assert {edge: fallback_edges.count(edge) for edge in set(fallback_edges)} == {
        0.026: 2,
        0.028: 2,
        0.03: 2,
        0.032: 1,
        0.034: 1,
    }
    thirty_mm = [
        item
        for item in alternative["top_candidates"]
        if item.get("search_stage") == "alternative_physics"
        and item["config"]["cube"]["edge_m"] == 0.03
    ]
    assert any(item["nominal_physics"] for item in thirty_mm)
    assert any(not item["nominal_physics"] for item in thirty_mm)
    assert alternative["best"]["config"]["experiment_status"] == {
        "classification": "validated_alternative_physics",
        "passed": True,
        "nominal_physics_passed": False,
        "note": "All declared hard constraints passed.",
    }


def test_nominal_material_found_during_fallback_is_classified_nominal(
    v2_config, monkeypatch
):
    _patch_cheap_stages(monkeypatch, v2_config, screen_count=1)

    stage = 0

    def nominal_fallback_runner(
        payload: list[tuple[int, dict[str, Any]]], workers: int
    ) -> list[dict[str, Any]]:
        nonlocal stage
        del workers
        stage += 1
        return [
            {
                "candidate_id": candidate_id,
                "config": candidate,
                "summary": _summary(
                    stage >= 3
                    and v2_search._has_nominal_cube_physics(
                        candidate, v2_config
                    ),
                    candidate_id,
                ),
            }
            for candidate_id, candidate in payload
        ]

    result = tune_opposed_face(
        copy.deepcopy(v2_config),
        workers=2,
        seed=31,
        run_candidates=nominal_fallback_runner,
        rank_candidate=search.candidate_rank,
        dynamic_candidate_count=1,
        local_refine_seed_count=1,
        local_refine_per_seed=1,
        final_candidate_count=1,
        perturbations_per_final=1,
        fallback_physics_count=8,
        perturb_cases=None,
    )

    assert result["nominal_success"] is True
    assert result["best"]["search_stage"] == "alternative_physics"
    assert result["best"]["nominal_physics"] is True
    assert result["best"]["config"]["experiment_status"]["classification"] == (
        "validated_nominal"
    )


def test_fallback_defaults_come_from_versioned_experiment_budget(
    v2_config, monkeypatch
):
    _patch_cheap_stages(monkeypatch, v2_config, screen_count=1)
    definition = resolve_experiment(v2_config)
    custom_bounds = replace(
        definition.search_bounds,
        fallback_kinematic_samples_per_pitch=4,
        fallback_candidate_count=11,
    )
    monkeypatch.setattr(
        v2_search,
        "resolve_experiment",
        lambda _config: replace(definition, search_bounds=custom_bounds),
    )

    result = tune_opposed_face(
        copy.deepcopy(v2_config),
        workers=1,
        seed=57,
        run_candidates=_runner(
            lambda _candidate_id, candidate: not np.isclose(
                candidate["cube"]["edge_m"], 0.03
            )
        ),
        rank_candidate=search.candidate_rank,
        dynamic_candidate_count=1,
        local_refine_seed_count=1,
        local_refine_per_seed=1,
        final_candidate_count=1,
        perturbations_per_final=1,
        perturb_cases=None,
    )

    assert result["fallback_physics_count"] == 11
    assert result["fallback_kinematic_samples_per_pitch"] == 4
    assert [
        record["dynamic_candidate_count"]
        for record in result["alternative_geometry_screens"]
    ] == [3, 2, 2, 2, 2]


def test_alternative_size_search_is_worker_independent(v2_config, monkeypatch):
    _patch_cheap_stages(monkeypatch, v2_config, screen_count=2)

    def run(workers: int) -> dict[str, Any]:
        return tune_opposed_face(
            copy.deepcopy(v2_config),
            workers=workers,
            seed=91,
            run_candidates=_runner(
                lambda _candidate_id, candidate: not np.isclose(
                    candidate["cube"]["edge_m"], 0.03
                ),
                reverse_for_parallel_workers=True,
            ),
            rank_candidate=search.candidate_rank,
            kinematic_samples_per_pitch=7,
            dynamic_candidate_count=2,
            local_refine_seed_count=1,
            local_refine_per_seed=1,
            final_candidate_count=1,
            perturbations_per_final=1,
            fallback_physics_count=9,
            fallback_kinematic_samples_per_pitch=3,
            perturb_cases=None,
        )

    serial = run(1)
    parallel = run(8)
    assert serial["fallback_physics_count"] == parallel["fallback_physics_count"] == 9
    assert serial["fallback_kinematic_samples_per_pitch"] == 3
    assert serial["alternative_geometry_screens"] == parallel[
        "alternative_geometry_screens"
    ]
    assert [
        record["dynamic_candidate_count"]
        for record in serial["alternative_geometry_screens"]
    ] == [2, 2, 2, 2, 1]
    assert serial["best"]["candidate_id"] == parallel["best"]["candidate_id"]
    assert serial["best"]["search_stage"] == parallel["best"]["search_stage"]
    assert [item["candidate_id"] for item in serial["top_candidates"]] == [
        item["candidate_id"] for item in parallel["top_candidates"]
    ]


def test_candidate_order_and_selection_are_worker_independent(
    v2_config, monkeypatch
):
    _patch_cheap_stages(monkeypatch, v2_config, screen_count=3)

    def run(workers: int) -> dict[str, Any]:
        return tune_opposed_face(
            copy.deepcopy(v2_config),
            workers=workers,
            seed=44,
            run_candidates=_runner(True, reverse_for_parallel_workers=True),
            rank_candidate=search.candidate_rank,
            dynamic_candidate_count=3,
            local_refine_seed_count=2,
            local_refine_per_seed=1,
            final_candidate_count=1,
            perturbations_per_final=1,
            fallback_physics_count=0,
            perturb_cases=None,
        )

    serial = run(1)
    parallel = run(8)
    assert serial["best"]["candidate_id"] == parallel["best"]["candidate_id"] == 0
    assert [item["candidate_id"] for item in serial["top_candidates"]] == [
        item["candidate_id"] for item in parallel["top_candidates"]
    ] == [0, 1, 2, 3, 4]
    assert serial["best"]["summary"] == parallel["best"]["summary"]
