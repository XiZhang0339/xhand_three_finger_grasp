from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest

import xhand_grasp.cli as cli
import xhand_grasp.pose_rescue as pose_rescue
import xhand_grasp.search as search
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.pose_rescue import (
    DEFAULT_PLAN,
    MIDDLE_JOINT1,
    MIDDLE_JOINT2,
    PoseRescuePlan,
    continuation_gate_evidence,
    cube_position_in_root,
    generate_local_pose_rescue_candidates,
    generate_pose_rescue_candidates,
    make_anchor_candidate,
    pose_rescue_candidate_rank,
    pose_rescue_continuation_gate,
    pose_rescue_manifest,
    rank_pose_rescue_results,
    run_pose_rescue_batch,
    select_local_refinement_parents,
    tune_relative_pose_rescue,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_grasp_then_lift.json"
)
RESCUE_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_cube_relative_pose_rescue.json"
)


def _base_config() -> dict:
    return load_config(BASE_CONFIG)


def _dynamic_result(
    candidate_id: int,
    *,
    duties: tuple[float, float, float] = (0.0, 0.0, 0.0),
    forces: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gate_steps: int = 0,
    total_force: float = 0.0,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "config": {},
        "summary": {
            "passed": False,
            "stage_status": {
                "grasp_success": False,
                "manipulation_success": False,
                "full_success": False,
            },
            "metrics": {
                "verify_target_face_effective_duty": dict(
                    zip(("thumb", "index", "mid"), duties)
                ),
                "verify_peak_target_face_force_n": dict(
                    zip(("thumb", "index", "mid"), forces)
                ),
                "verify_peak_tactile_n": dict(
                    zip(("thumb", "index", "mid"), forces)
                ),
                "verify_max_consecutive_all_gate_steps": gate_steps,
                "verify_all_gate_duty": 0.0,
                "verify_gate_component_duty": {},
                "peak_total_distal_contact_force_n": total_force,
                "actuator_saturation_fraction": 0.0,
                "forbidden_contact_steps": 0,
            },
        },
    }


def test_plan_fixes_source_anchor_focus_and_correlated_target_ranges():
    plan = DEFAULT_PLAN
    assert plan.seed == 20260821
    assert plan.anchor_edge_m == pytest.approx(0.062)
    assert plan.priority_edges_m == pytest.approx((0.061, 0.062, 0.063))
    assert plan.source_candidate_id == 309288
    assert plan.anchor_cube_in_root_m == pytest.approx(
        (0.082470, -0.029689, 0.108106)
    )
    assert plan.anchor_hand_rpy_deg == pytest.approx((1.0, 90.0, -3.779))
    assert plan.cube_position_in_root_m == {
        "x": (0.08164, 0.08364),
        "y": (-0.03085, -0.02885),
        "z": (0.10736, 0.10936),
    }
    assert plan.hand_pitch_values_deg == pytest.approx((88.5, 89, 89.5, 90))
    assert plan.hand_roll_deg == pytest.approx((-0.5, 2.5))
    assert plan.hand_yaw_deg == pytest.approx((-5.8, -1.8))
    assert plan.cube_yaw_deg == pytest.approx((26.5, 30.5))
    assert plan.thumb_index_target_ranges_rad[
        "left_hand_thumb_rota_joint2_actuator"
    ] == pytest.approx((0.46, 0.58))
    assert plan.thumb_index_target_ranges_rad[
        "left_hand_index_bend_joint_actuator"
    ] == pytest.approx((-0.02, 0.08))
    assert plan.thumb_index_target_ranges_rad[
        "left_hand_index_joint1_actuator"
    ] == pytest.approx((0.30, 0.48))
    assert plan.thumb_index_target_ranges_rad[
        "left_hand_index_joint2_actuator"
    ] == pytest.approx((1.48, 1.70))
    assert plan.actuator_limits_rad[
        "left_hand_index_joint2_actuator"
    ] == pytest.approx((0.30, 1.80))
    assert plan.actuator_limits_rad[MIDDLE_JOINT2] == pytest.approx((0.30, 1.80))
    assert plan.target_faces == {"thumb": "-X", "index": "+X", "mid": "+X"}
    assert (
        plan.anchor_targets_rad[MIDDLE_JOINT1]
        + plan.anchor_targets_rad[MIDDLE_JOINT2]
    ) == pytest.approx(2.0)

    with pytest.raises(ValueError, match="fixed at 20260821"):
        PoseRescuePlan(seed=7)
    with pytest.raises(ValueError, match="priority_edges_m"):
        PoseRescuePlan(priority_edges_m=(0.062, 0.061, 0.063))


def test_anchor_is_materialized_from_relative_pose_without_mutating_base():
    base = _base_config()
    pristine = copy.deepcopy(base)
    anchor = make_anchor_candidate(base, validator=validate_config)

    assert base == pristine
    assert anchor["cube"]["edge_m"] == pytest.approx(0.062)
    assert anchor["hand_pose"]["rpy_deg"] == pytest.approx((1.0, 90.0, -3.779))
    assert anchor["hand_pose"]["translation_m"] == pytest.approx(
        (-0.03425669746601186, 0.0115923774280421, 0.19747), abs=2e-9
    )
    assert cube_position_in_root(anchor) == pytest.approx(
        DEFAULT_PLAN.anchor_cube_in_root_m, abs=2e-9
    )
    assert anchor["cube"]["rpy_deg"] == pytest.approx((0.0, 0.0, 28.549))
    assert anchor["control"]["grasp_targets_rad"] == pytest.approx(
        dict(DEFAULT_PLAN.anchor_targets_rad)
    )
    assert set(anchor["control"]) == {
        "grasp_targets_rad",
        "manipulation_delta_rad",
    }
    assert set(anchor["control"]["manipulation_delta_rad"].values()) == {0.0}
    assert anchor["contact_topology"]["target_faces"] == {
        "thumb": "-X",
        "index": "+X",
        "mid": "+X",
    }


@pytest.mark.parametrize("method", ("lhs", "grid"))
def test_candidate_generation_is_deterministic_bounded_and_correlated(method):
    base = _base_config()
    first = generate_pose_rescue_candidates(
        base,
        method=method,
        samples_per_edge=5,
        validator=validate_config,
    )
    second = generate_pose_rescue_candidates(
        base,
        method=method,
        samples_per_edge=5,
        validator=validate_config,
    )
    assert first == second
    assert len(first) == 15
    assert [candidate["cube"]["edge_m"] for candidate in first] == pytest.approx(
        [0.061] * 5 + [0.062] * 5 + [0.063] * 5
    )
    assert first[5] == make_anchor_candidate(base, validator=validate_config)

    for candidate in first:
        relative = cube_position_in_root(candidate)
        for index, axis in enumerate(("x", "y", "z")):
            lower, upper = DEFAULT_PLAN.cube_position_in_root_m[axis]
            assert lower - 1e-12 <= relative[index] <= upper + 1e-12
        roll, pitch, yaw = candidate["hand_pose"]["rpy_deg"]
        assert DEFAULT_PLAN.hand_roll_deg[0] <= roll <= DEFAULT_PLAN.hand_roll_deg[1]
        assert pitch in DEFAULT_PLAN.hand_pitch_values_deg
        assert DEFAULT_PLAN.hand_yaw_deg[0] <= yaw <= DEFAULT_PLAN.hand_yaw_deg[1]
        cube_yaw = candidate["cube"]["rpy_deg"][2]
        assert DEFAULT_PLAN.cube_yaw_deg[0] <= cube_yaw <= DEFAULT_PLAN.cube_yaw_deg[1]

        targets = candidate["control"]["grasp_targets_rad"]
        assert set(targets) == set(ACTIVE_ACTUATORS)
        is_source_anchor = (
            candidate["cube"]["edge_m"] == pytest.approx(0.062)
            and targets == pytest.approx(dict(DEFAULT_PLAN.anchor_targets_rad))
        )
        if not is_source_anchor:
            for name, bounds in DEFAULT_PLAN.narrow_target_delta_rad.items():
                delta = targets[name] - DEFAULT_PLAN.anchor_targets_rad[name]
                assert bounds[0] - 1e-12 <= delta <= bounds[1] + 1e-12
        assert DEFAULT_PLAN.middle_joint1_rad[0] <= targets[MIDDLE_JOINT1] <= (
            DEFAULT_PLAN.middle_joint1_rad[1]
        )
        assert DEFAULT_PLAN.middle_joint2_rad[0] <= targets[MIDDLE_JOINT2] <= (
            DEFAULT_PLAN.middle_joint2_rad[1]
        )
        middle_sum = targets[MIDDLE_JOINT1] + targets[MIDDLE_JOINT2]
        assert DEFAULT_PLAN.middle_joint_sum_rad[0] <= middle_sum <= (
            DEFAULT_PLAN.middle_joint_sum_rad[1]
        )
        assert set(candidate["control"]["manipulation_delta_rad"].values()) == {0.0}
        assert candidate["contact_topology"]["target_faces"] == dict(
            DEFAULT_PLAN.target_faces
        )


def test_grid_and_lhs_are_distinct_and_invalid_method_is_rejected():
    base = _base_config()
    lhs = generate_pose_rescue_candidates(
        base, method="lhs", samples_per_edge=3, include_anchor=False
    )
    grid = generate_pose_rescue_candidates(
        base, method="grid", samples_per_edge=3, include_anchor=False
    )
    assert lhs != grid
    with pytest.raises(ValueError, match="lhs.*grid"):
        generate_pose_rescue_candidates(base, method="random", samples_per_edge=1)


def test_default_broad_generation_is_256_total_with_priority_remainder():
    candidates = generate_pose_rescue_candidates(_base_config())
    assert len(candidates) == 256
    edges = [candidate["cube"]["edge_m"] for candidate in candidates]
    assert edges[:86] == pytest.approx([0.061] * 86)
    assert edges[86 : 86 + 85] == pytest.approx([0.062] * 85)
    assert edges[86 + 85 :] == pytest.approx([0.063] * 85)


def test_continuation_gate_requires_two_duties_and_remaining_force_or_10ms():
    near = _dynamic_result(
        1,
        duties=(0.10, 0.20, 0.09),
        forces=(0.05, 0.05, 0.02),
    )
    evidence = continuation_gate_evidence(near)
    assert evidence["continue"] is True
    assert evidence["two_duty_plus_third_force"] is True
    assert evidence["duty_fingers"] == ["thumb", "index"]
    assert evidence["third_finger_peak_target_force_n"] == pytest.approx(0.02)

    weak_third = _dynamic_result(
        2,
        duties=(0.10, 0.20, 0.09),
        forces=(0.05, 0.05, 0.019999),
        gate_steps=9,
    )
    assert pose_rescue_continuation_gate(weak_third) is False
    weak_third["summary"]["metrics"]["verify_max_consecutive_all_gate_steps"] = 10
    evidence = continuation_gate_evidence(weak_third)
    assert evidence["continue"] is True
    assert evidence["all_gate_10ms"] is True
    assert evidence["all_gate_duration_s"] == pytest.approx(0.010)

    assert not pose_rescue_continuation_gate(
        _dynamic_result(3, duties=(0.9, 0.0, 0.0), forces=(1.0, 1.0, 1.0))
    )


def test_dynamic_rank_never_prefers_zero_contact_to_two_finger_near():
    zero = _dynamic_result(1, total_force=0.0)
    two_near = _dynamic_result(
        99,
        duties=(0.10, 0.25, 0.0),
        forces=(0.05, 0.05, 0.02),
        total_force=50.0,
    )
    assert pose_rescue_candidate_rank(two_near) > pose_rescue_candidate_rank(zero)
    assert rank_pose_rescue_results((zero, two_near))[0]["candidate_id"] == 99

    shuffled = [zero, two_near]
    random.Random(9).shuffle(shuffled)
    assert [item["candidate_id"] for item in rank_pose_rescue_results(shuffled)] == [
        99,
        1,
    ]
    with pytest.raises(ValueError, match="unique"):
        rank_pose_rescue_results((zero, copy.deepcopy(zero)))


def test_top_stable_parent_gets_deterministic_budgeted_local_neighborhood():
    anchor = make_anchor_candidate(_base_config(), validator=validate_config)
    stable = _dynamic_result(
        7,
        duties=(0.5, 0.5, 0.5),
        forces=(0.1, 0.1, 0.1),
        gate_steps=250,
    )
    stable["config"] = anchor
    stable["summary"]["stage_status"]["grasp_success"] = True
    zero = _dynamic_result(8)
    zero["config"] = anchor

    assert [item["candidate_id"] for item in select_local_refinement_parents((zero, stable))] == [7]
    first = generate_local_pose_rescue_candidates(
        (zero, stable), validator=validate_config
    )
    second = generate_local_pose_rescue_candidates(
        (stable, zero), validator=validate_config
    )
    assert first == second
    assert len(first) == DEFAULT_PLAN.local_samples_per_parent == 32
    for candidate in first:
        assert candidate["cube"]["edge_m"] == pytest.approx(0.062)
        assert set(candidate["control"]["manipulation_delta_rad"].values()) == {0.0}
        targets = candidate["control"]["grasp_targets_rad"]
        assert 0.52 <= targets[MIDDLE_JOINT1] <= 0.78
        assert 1.32 <= targets[MIDDLE_JOINT2] <= 1.58


def test_generic_runner_is_injected_reordered_and_contract_checked():
    configs = generate_pose_rescue_candidates(
        _base_config(), method="grid", samples_per_edge=1
    )[:2]

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

    results = run_pose_rescue_batch(
        configs,
        run_candidates=reversed_runner,
        workers=3,
        first_candidate_id=20,
    )
    assert [item["candidate_id"] for item in results] == [20, 21]

    def missing_runner(payload, workers):
        del workers
        candidate_id, config = payload[0]
        return [{"candidate_id": candidate_id, "config": config}]

    with pytest.raises(RuntimeError, match="every submitted"):
        run_pose_rescue_batch(
            configs, run_candidates=missing_runner, workers=1
        )


def test_two_stage_tuner_uses_injected_runner_and_caps_success_at_stable_grasp():
    config = load_config(RESCUE_CONFIG)
    calls: list[tuple[int, int]] = []

    def fake_runner(payload, workers):
        calls.append((len(payload), workers))
        results = []
        for candidate_id, candidate in payload:
            assert set(candidate["control"]["manipulation_delta_rad"].values()) == {
                0.0
            }
            stable = candidate_id != 2
            duty = 0.5 if stable else 0.0
            force = 0.1 if stable else 0.0
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": candidate,
                    "summary": {
                        # Deliberately overclaim later stages. The rescue
                        # orchestration must retain this only as raw evidence.
                        "passed": stable,
                        "failed_checks": [],
                        "checks": {},
                        "stage_status": {
                            "grasp_success": stable,
                            "manipulation_success": stable,
                            "full_success": stable,
                        },
                        "metrics": {
                            "verify_target_face_effective_duty": {
                                finger: duty for finger in ("thumb", "index", "mid")
                            },
                            "verify_peak_target_face_force_n": {
                                finger: force for finger in ("thumb", "index", "mid")
                            },
                            "verify_peak_tactile_n": {
                                finger: force for finger in ("thumb", "index", "mid")
                            },
                            "verify_max_consecutive_all_gate_steps": (
                                250 if stable else 0
                            ),
                            "verify_all_gate_duty": duty,
                            "verify_gate_component_duty": {},
                            "peak_total_distal_contact_force_n": force,
                            "actuator_saturation_fraction": 0.0,
                            "forbidden_contact_steps": 0,
                        },
                    },
                }
            )
        return list(reversed(results))

    result = tune_relative_pose_rescue(
        config,
        workers=2,
        seed=20260821,
        run_candidates=fake_runner,
        dynamic_candidate_count=3,
        local_refine_seed_count=2,
        local_refine_per_seed=2,
        final_candidate_count=7,
        perturbations_per_final=9,
        fallback_physics_count=4,
        fallback_kinematic_samples_per_pitch=11,
        kinematic_samples_per_pitch=13,
        legacy_parameters={"samples": 99, "refine_top": 4, "refine_per": 5},
    )

    assert calls == [(3, 2), (4, 2)]
    assert result["candidate_count"] == result["simulation_count"] == 7
    assert result["broad_candidate_count"] == 3
    assert result["local_candidate_count"] == 4
    assert result["stable_grasp_candidate_count"] == 6
    assert result["campaign_classification"] == "pose_rescue_stable_grasp"
    assert result["campaign_status"]["passed"] is False
    assert result["campaign_status"]["manipulation_success"] is False
    assert result["campaign_status"]["constant_density_passed"] is False
    assert result["passing_candidates"] == 0
    assert result["perturbation_probe_count"] == 0
    assert result["best_fixed_mass"] is None
    assert result["best"]["summary"]["passed"] is False
    assert result["best"]["summary"]["stage_status"] == {
        "grasp_success": True,
        "manipulation_success": False,
        "full_success": False,
    }
    assert result["best"]["observed_grasp_evidence"]["grasp_success"] is True
    assert result["best"]["local_perturbation_probe"] == {
        "candidate_id": result["best"]["candidate_id"],
        "passes": 0,
        "trial_count": 0,
        "trials": [],
    }
    assert all(
        item["config"]["experiment_status"]["constant_density_passed"] is False
        for item in result["top_candidates"]
    )

    def values_for_key(value, key):
        found = []
        if isinstance(value, dict):
            for name, item in value.items():
                if name == key:
                    found.append(item)
                found.extend(values_for_key(item, key))
        elif isinstance(value, list):
            for item in value:
                found.extend(values_for_key(item, key))
        return found

    for capped_key in (
        "manipulation_success",
        "full_success",
        "constant_density_success",
        "constant_density_passed",
    ):
        assert not any(values_for_key(result, capped_key))
    ignored = result["campaign_manifest"]["not_applicable_overrides"]
    assert ignored["kinematic_samples_per_pitch"] == 13
    assert ignored["final_candidate_count"] == 7
    assert ignored["perturbations_per_final"] == 9
    assert ignored["fallback_physics_count"] == 4
    assert ignored["fallback_kinematic_samples_per_pitch"] == 11


def test_search_tune_dispatches_rescue_before_the_larger_cube_tuner(monkeypatch):
    config = load_config(RESCUE_CONFIG)
    sentinel = {"dispatched": True}
    observed: dict = {}

    def fake_tuner(value, **kwargs):
        observed["config"] = value
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(pose_rescue, "tune_relative_pose_rescue", fake_tuner)
    result = search.tune(
        config,
        samples=12,
        refine_top=3,
        refine_per=4,
        workers=2,
        seed=20260821,
        dynamic_candidate_count=5,
        local_refine_seed_count=2,
        local_refine_per_seed=6,
    )

    assert result is sentinel
    assert observed["config"] is config
    assert observed["dynamic_candidate_count"] == 5
    assert observed["local_refine_seed_count"] == 2
    assert observed["local_refine_per_seed"] == 6
    assert observed["legacy_parameters"] == {
        "samples": 12,
        "refine_top": 3,
        "refine_per": 4,
    }


def test_real_cli_tune_small_budget_uses_explicit_temporary_output(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "explicit-rescue-output"
    args = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(RESCUE_CONFIG),
            "--output-dir",
            str(output),
            "--workers",
            "1",
            "--dynamic-candidates",
            "1",
            "--local-refine-seeds",
            "1",
            "--local-refine-per-seed",
            "1",
        ]
    )

    assert args.func(args) == 2
    capsys.readouterr()
    persisted = json.loads((output / "tune_results.json").read_text())
    assert persisted["campaign_kind"] == "relative_pose_rescue"
    assert persisted["campaign_classification"] in {
        "not_validated",
        "pose_rescue_stable_grasp",
    }
    assert persisted["candidate_count"] in {1, 2}
    assert persisted["simulation_count"] == persisted["candidate_count"]
    assert persisted["perturbation_probe_count"] == 0
    assert persisted["passing_candidates"] == 0
    assert persisted["best_fixed_mass"] is None
    assert persisted["best"]["config"]["experiment_status"]["passed"] is False
    assert not (tmp_path / "artifacts").exists()


def test_manifest_is_json_safe_and_declares_source_budget_and_invariants():
    manifest = pose_rescue_manifest()
    json.dumps(manifest, sort_keys=True, allow_nan=False)
    source = manifest["source_provenance"]
    assert source["experiment_id"] == (
        "left_opposed_face_palm_down_larger_cube_grasp_then_lift"
    )
    assert source["artifact_sha256"] == (
        "5c1cd21ac523904e06c2d2062c34fba0699323619f78b2c7f51108c6cb06a396"
    )
    assert source["source_candidate_id"] == 309288
    assert source["focused_probe"] == {
        "provenance_kind": "follow_up_joint_rescue_measurement",
        "anchor_candidate_id": 309288,
        "candidate_count": 256,
        "stable_grasp_count": 18,
        "stable_grasp_rate": 18 / 256,
    }
    assert manifest["budget"]["lhs_candidate_count"] == 256
    assert manifest["budget"]["grid_candidate_count"] == 256
    assert manifest["budget"]["local_candidate_budget"] == 256
    assert manifest["budget"]["dynamic_candidate_budget"] == 512
    assert manifest["invariants"]["root_pose_runtime_static"] is True
    assert manifest["invariants"]["manipulation_delta_rad_zero"] is True
    assert manifest["continuation_policy"]["operator"] == "branch_a OR branch_b"
