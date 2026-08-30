from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from xhand_grasp.config import (
    ACTIVE_ACTUATORS,
    load_config,
    resolved_pose_constraint_values,
    validate_config,
)
from xhand_grasp.experiment import resolve_experiment
from xhand_grasp.tuning.high_thumb_grasp import (
    HighThumbCartesianGrid,
    HighThumbLocalSearch,
    REFERENCE_HIGH_THUMB_GRID,
    THUMB_BEND_ACTUATOR,
    cube_in_root_from_distance_yz,
    generate_high_thumb_cartesian_candidates,
    generate_local_high_thumb_candidates,
    high_thumb_grasp_candidate_rank,
    materialize_high_thumb_candidate,
    rank_high_thumb_grasp_results,
    run_high_thumb_grasp_batch,
)


ROOT = Path(__file__).resolve().parents[1]
V5_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift.json"
)
V4_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift.json"
)


@pytest.fixture
def v5_config() -> dict:
    return load_config(V5_CONFIG)


def _targets(config: dict, *, thumb: float = 1.15) -> dict[str, float]:
    result = dict(config["control"]["grasp_targets_rad"])
    result[THUMB_BEND_ACTUATOR] = thumb
    return result


def _seed_vector() -> tuple[float, float, float]:
    return cube_in_root_from_distance_yz(0.158, -0.027, 0.1195)


def _candidate_summary(
    *,
    grasp: bool,
    candidate_force_n: float = 2.0,
    effective_fingers: int = 3,
    full: bool = False,
) -> dict:
    duties = {
        "thumb": 0.9 if effective_fingers >= 1 else 0.0,
        "index": 0.9 if effective_fingers >= 2 else 0.0,
        "mid": 0.9 if effective_fingers >= 3 else 0.0,
    }
    forces = {finger: 1.0 if value else 0.0 for finger, value in duties.items()}
    return {
        "passed": full,
        "failed_checks": [] if full else ["not_a_full_lift"],
        "stage_status": {
            "grasp_success": grasp,
            "manipulation_success": full,
            "full_success": full,
        },
        "metrics": {
            "grasp_stability_margin": 0.2 if grasp else -0.2,
            "verify_effective_finger_count": effective_fingers,
            "verify_max_simultaneous_effective_finger_count": effective_fingers,
            "verify_target_face_effective_duty": duties,
            "verify_target_face_simultaneous_duty": (
                min(duties.values())
            ),
            "verify_peak_target_face_force_n": forces,
            "verify_peak_tactile_n": forces,
            "verify_max_consecutive_all_gate_steps": 250 if grasp else 0,
            "verify_all_gate_duty": 0.5 if grasp else 0.0,
            "verify_gate_component_duty": {
                "topology": min(duties.values()),
                "finite": 1.0,
            },
            "contact_alignment": {
                "verify": {
                    "aligned_duty": 0.8 if grasp else 0.0,
                    "height_spread_p95_m": 0.003 if grasp else 0.02,
                }
            },
            "fingertip_contact": {
                "verify": {
                    "force_weighted_pad_fraction": {
                        finger: 0.95 for finger in duties
                    },
                    "max_active_taxel_count": {
                        finger: 2 for finger in duties
                    },
                }
            },
            "forbidden_contact_steps": 0,
            "material_active_nondistal_duty": 0.0,
            "peak_total_distal_contact_force_n": candidate_force_n,
            "actuator_saturation_fraction": 0.01,
        },
    }


def test_distance_yz_reconstructs_requested_seed_vector():
    vector = cube_in_root_from_distance_yz(0.158, -0.027, 0.1195)

    assert vector == pytest.approx(
        (0.09977349347396833, -0.027, 0.1195), abs=1e-15
    )
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(
        0.158, abs=2e-15
    )
    assert cube_in_root_from_distance_yz(
        0.158, -0.027, 0.1195, positive_x=False
    )[0] == pytest.approx(-vector[0])


@pytest.mark.parametrize(
    "distance,y_value,z_value,match",
    [
        (0.0, 0.0, 0.0, "positive"),
        (0.1, 0.09, 0.09, "shorter"),
        (math.nan, 0.0, 0.0, "finite"),
    ],
)
def test_distance_yz_rejects_invalid_geometry(
    distance, y_value, z_value, match
):
    with pytest.raises(ValueError, match=match):
        cube_in_root_from_distance_yz(distance, y_value, z_value)


def test_materializer_reproduces_declared_60mm_158mm_high_thumb_seed(
    v5_config,
):
    vector = _seed_vector()
    candidate = materialize_high_thumb_candidate(
        v5_config,
        edge_m=0.060,
        cube_in_root_m=vector,
        cube_yaw_deg=26.0,
        grasp_targets_rad=_targets(v5_config),
    )
    resolved = resolved_pose_constraint_values(candidate)

    validate_config(candidate)
    assert "run_context" not in candidate
    assert candidate["cube"]["edge_m"] == pytest.approx(0.060)
    assert candidate["cube"]["mass_kg"] == pytest.approx(0.160)
    assert candidate["cube"]["friction"] == pytest.approx(0.8)
    assert candidate["cube"]["rpy_deg"] == pytest.approx([0.0, 0.0, 26.0])
    assert resolved["cube_position_in_root_m"] == pytest.approx(vector, abs=1e-12)
    assert resolved["root_cube_distance_m"] == pytest.approx(0.158, abs=1e-12)
    assert candidate["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR] == (
        pytest.approx(1.15)
    )
    assert candidate["control"]["manipulation_delta_rad"] == {
        name: 0.0 for name in ACTIVE_ACTUATORS
    }
    assert candidate["candidate_metadata"]["zero_manipulation_delta"] is True


def test_materializer_uses_registered_density_for_nonnominal_edge(v5_config):
    definition = resolve_experiment(v5_config)
    campaign = definition.far_hand_campaign
    assert campaign is not None
    candidate = materialize_high_thumb_candidate(
        v5_config,
        edge_m=0.061,
        cube_in_root_m=_seed_vector(),
        cube_yaw_deg=26.0,
        grasp_targets_rad=_targets(v5_config),
    )

    validate_config(candidate)
    assert candidate["run_context"] == {"kind": "parameter_override_run"}
    assert candidate["cube"]["mass_kg"] == pytest.approx(
        campaign.density_kg_m3 * 0.061**3
    )
    assert candidate["candidate_metadata"]["density_kg_m3"] == pytest.approx(
        campaign.density_kg_m3
    )


def test_materializer_marks_override_without_validator_and_calls_spy_once(
    v5_config,
):
    calls = []
    campaign = resolve_experiment(v5_config).far_hand_campaign
    assert campaign is not None

    def validator(candidate):
        calls.append(copy.deepcopy(candidate))
        validate_config(candidate)

    candidate = materialize_high_thumb_candidate(
        v5_config,
        edge_m=0.061,
        cube_in_root_m=_seed_vector(),
        cube_yaw_deg=26.0,
        grasp_targets_rad=_targets(v5_config),
        validator=validator,
    )
    unchecked = materialize_high_thumb_candidate(
        v5_config,
        edge_m=0.061,
        cube_in_root_m=_seed_vector(),
        cube_yaw_deg=26.0,
        grasp_targets_rad=_targets(v5_config),
        validator=None,
    )

    assert len(calls) == 1
    assert calls[0]["run_context"] == {"kind": "parameter_override_run"}
    assert candidate["run_context"] == {"kind": "parameter_override_run"}
    assert unchecked["run_context"] == {"kind": "parameter_override_run"}
    assert unchecked["cube"]["mass_kg"] == pytest.approx(
        campaign.constant_density_mass_kg(0.061)
    )


def test_materializer_requires_all_eight_targets_and_schema_v5(v5_config):
    missing = _targets(v5_config)
    missing.pop("left_hand_mid_joint2_actuator")
    with pytest.raises(ValueError, match="exactly the eight"):
        materialize_high_thumb_candidate(
            v5_config,
            edge_m=0.060,
            cube_in_root_m=_seed_vector(),
            cube_yaw_deg=26.0,
            grasp_targets_rad=missing,
        )

    with pytest.raises(ValueError, match="schema-v5"):
        materialize_high_thumb_candidate(
            load_config(V4_CONFIG),
            edge_m=0.060,
            cube_in_root_m=_seed_vector(),
            cube_yaw_deg=26.0,
            grasp_targets_rad=_targets(v5_config),
        )


def test_cartesian_grid_reproduces_seed_and_has_stable_order(v5_config):
    grid = HighThumbCartesianGrid(
        edge_m=(0.060, 0.061),
        root_cube_distance_m=(0.158,),
        cube_in_root_y_m=(-0.027,),
        cube_in_root_z_m=(0.1195,),
        cube_yaw_deg=(26.0,),
        thumb_bend_rad=(1.15, 1.20),
    )
    first = generate_high_thumb_cartesian_candidates(v5_config, grid)
    second = generate_high_thumb_cartesian_candidates(v5_config, grid)

    assert first == second
    assert len(first) == grid.candidate_count == 4
    assert [
        (
            candidate["cube"]["edge_m"],
            candidate["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR],
        )
        for candidate in first
    ] == pytest.approx(
        [(0.060, 1.15), (0.060, 1.20), (0.061, 1.15), (0.061, 1.20)]
    )
    resolved = resolved_pose_constraint_values(first[0])
    assert resolved["cube_position_in_root_m"] == pytest.approx(
        _seed_vector(), abs=1e-12
    )
    assert first[0]["cube"]["rpy_deg"][2] == pytest.approx(26.0)
    assert first[0]["candidate_metadata"]["cartesian_candidate_index"] == 0


def test_reference_grid_is_the_declared_single_high_thumb_seed(v5_config):
    candidates = generate_high_thumb_cartesian_candidates(
        v5_config, REFERENCE_HIGH_THUMB_GRID
    )

    assert REFERENCE_HIGH_THUMB_GRID.candidate_count == 1
    assert len(candidates) == 1
    candidate = candidates[0]
    resolved = resolved_pose_constraint_values(candidate)
    assert candidate["cube"]["edge_m"] == pytest.approx(0.060)
    assert resolved["root_cube_distance_m"] == pytest.approx(0.158)
    assert resolved["cube_position_in_root_m"][1:] == pytest.approx(
        (-0.027, 0.1195)
    )
    assert candidate["cube"]["rpy_deg"][2] == pytest.approx(26.0)
    assert candidate["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR] == (
        pytest.approx(1.15)
    )


def test_local_generation_is_deterministic_and_keeps_exact_parent_first(
    v5_config,
):
    parent = materialize_high_thumb_candidate(
        v5_config,
        edge_m=0.060,
        cube_in_root_m=_seed_vector(),
        cube_yaw_deg=26.0,
        grasp_targets_rad=_targets(v5_config),
    )
    search = HighThumbLocalSearch(minimum_thumb_bend_rad=1.10)
    first = generate_local_high_thumb_candidates(
        parent, count=6, seed=20260821, search=search
    )
    second = generate_local_high_thumb_candidates(
        parent, count=6, seed=20260821, search=search
    )
    different = generate_local_high_thumb_candidates(
        parent, count=6, seed=20260822, search=search
    )

    assert first == second
    assert first[1:] != different[1:]
    exact = first[0]
    assert exact["cube"] == parent["cube"]
    assert exact["control"] == parent["control"]
    assert resolved_pose_constraint_values(exact)["cube_position_in_root_m"] == (
        pytest.approx(_seed_vector(), abs=1e-12)
    )
    assert exact["candidate_metadata"]["local_parent_exact"] is True
    for candidate in first:
        validate_config(candidate)
        assert set(candidate["control"]["grasp_targets_rad"]) == set(
            ACTIVE_ACTUATORS
        )
        assert all(
            value == 0.0
            for value in candidate["control"]["manipulation_delta_rad"].values()
        )
        assert candidate["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR] >= 1.10


def test_grasp_rank_ignores_generic_full_and_manipulation_claims():
    acquired = {
        "candidate_id": 9,
        "summary": _candidate_summary(grasp=True, full=False),
    }
    contradictory_full = {
        "candidate_id": 1,
        "summary": _candidate_summary(grasp=False, full=True),
    }

    assert high_thumb_grasp_candidate_rank(acquired) > (
        high_thumb_grasp_candidate_rank(contradictory_full)
    )
    assert rank_high_thumb_grasp_results(
        [contradictory_full, acquired]
    )[0] is acquired


def test_grasp_near_miss_rank_prefers_balanced_three_finger_evidence():
    balanced = {
        "candidate_id": 8,
        "summary": _candidate_summary(
            grasp=False, effective_fingers=3, candidate_force_n=12.0
        ),
    }
    one_finger = {
        "candidate_id": 2,
        "summary": _candidate_summary(
            grasp=False, effective_fingers=1, candidate_force_n=0.1
        ),
    }

    assert high_thumb_grasp_candidate_rank(balanced) > (
        high_thumb_grasp_candidate_rank(one_finger)
    )


def test_rank_is_worker_order_independent_and_rejects_duplicate_ids():
    lower_id = {
        "candidate_id": 2,
        "summary": _candidate_summary(grasp=False),
    }
    higher_id = {
        "candidate_id": 7,
        "summary": _candidate_summary(grasp=False),
    }

    forward = rank_high_thumb_grasp_results([lower_id, higher_id])
    reverse = rank_high_thumb_grasp_results([higher_id, lower_id])
    assert [item["candidate_id"] for item in forward] == [2, 7]
    assert [item["candidate_id"] for item in reverse] == [2, 7]
    with pytest.raises(ValueError, match="unique"):
        rank_high_thumb_grasp_results([lower_id, copy.deepcopy(lower_id)])


def test_injected_batch_runner_preserves_binding_and_sorts_ids(v5_config):
    grid = HighThumbCartesianGrid(
        edge_m=(0.060,),
        root_cube_distance_m=(0.158,),
        cube_in_root_y_m=(-0.027,),
        cube_in_root_z_m=(0.1195,),
        cube_yaw_deg=(26.0,),
        thumb_bend_rad=(1.15, 1.20),
    )
    candidates = generate_high_thumb_cartesian_candidates(v5_config, grid)
    original = copy.deepcopy(candidates)
    calls = []

    def runner(payloads, workers):
        calls.append((copy.deepcopy(payloads), workers))
        return [
            {
                "candidate_id": candidate_id,
                "config": copy.deepcopy(config),
                "summary": _candidate_summary(grasp=candidate_id == 11),
            }
            for candidate_id, config in reversed(payloads)
        ]

    results = run_high_thumb_grasp_batch(
        candidates,
        run_candidates=runner,
        workers=3,
        first_candidate_id=10,
        stage="unit_test_grasp",
    )

    assert candidates == original
    assert len(calls) == 1 and calls[0][1] == 3
    assert [result["candidate_id"] for result in results] == [10, 11]
    assert all(result["search_stage"] == "unit_test_grasp" for result in results)


def test_injected_batch_runner_rejects_rebound_config(v5_config):
    candidate = materialize_high_thumb_candidate(
        v5_config,
        edge_m=0.060,
        cube_in_root_m=_seed_vector(),
        cube_yaw_deg=26.0,
        grasp_targets_rad=_targets(v5_config),
    )

    def runner(payloads, workers):
        del workers
        candidate_id, config = payloads[0]
        rebound = copy.deepcopy(config)
        rebound["cube"]["friction"] = 0.9
        return [
            {
                "candidate_id": candidate_id,
                "config": rebound,
                "summary": _candidate_summary(grasp=False),
            }
        ]

    with pytest.raises(RuntimeError, match="rebound"):
        run_high_thumb_grasp_batch(
            [candidate], run_candidates=runner, workers=1
        )
