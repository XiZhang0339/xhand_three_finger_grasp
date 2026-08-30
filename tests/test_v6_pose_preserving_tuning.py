from __future__ import annotations

import copy
from pathlib import Path

import pytest

import xhand_grasp.search as search
import xhand_grasp.tuning.pose_preserving_grasp as pose_tuning
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.tuning.pose_preserving_grasp import (
    acquisition_succeeded,
    deterministic_rank_pose_preserving_results,
    generate_pose_preserving_candidates,
    materialize_pose_preserving_candidate,
    pose_preserving_candidate_rank,
    tune_pose_preserving_grasp,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_pose_preserving_grasp.json"
)


@pytest.fixture
def v6_config() -> dict:
    return load_config(CONFIG_PATH)


def _result(
    candidate_id: int,
    config: dict,
    *,
    grasp: bool,
    pose: bool,
    full: bool = False,
) -> dict:
    pose_checks = {
        name: pose for name in pose_tuning._POSE_CHECKS
    }
    return {
        "candidate_id": candidate_id,
        "config": config,
        "summary": {
            "passed": full,
            "failed_checks": [] if full else ["synthetic_incomplete"],
            "checks": {"synthetic_incomplete": full, **pose_checks},
            "stage_status": {
                "grasp_success": grasp,
                "manipulation_success": full,
                "full_success": full,
            },
            "metrics": {
                "pose_preservation": {
                    "max_translation_m": 0.0002 if pose else 0.0008,
                    "max_orientation_drift_deg": 0.4 if pose else 1.4,
                    "translation_limit_m": 0.0005,
                    "orientation_limit_deg": 1.0,
                    "distal_contact_onset_span_s": 0.005,
                },
                "verify_effective_finger_count": 3 if grasp else 2,
                "verify_max_simultaneous_effective_finger_count": (
                    3 if grasp else 2
                ),
                "verify_target_face_effective_duty": {
                    finger: 0.9 if grasp else 0.5
                    for finger in ("thumb", "index", "mid")
                },
                "verify_max_consecutive_all_gate_steps": 250 if grasp else 20,
                "peak_total_distal_contact_force_n": 1.5,
                "actuator_saturation_fraction": 0.02,
            },
        },
    }


def test_materializer_sets_all_v6_acquisition_controls_without_mutating_source(
    v6_config,
):
    source = copy.deepcopy(v6_config)
    hand_pose = copy.deepcopy(v6_config["hand_pose"])
    hand_pose["translation_m"][0] += 0.0001
    pregrasp = copy.deepcopy(v6_config["control"]["pregrasp_targets_rad"])
    pregrasp["left_hand_thumb_bend_joint_actuator"] += 0.01
    grasp = copy.deepcopy(v6_config["control"]["grasp_targets_rad"])
    grasp["left_hand_thumb_bend_joint_actuator"] += 0.01
    profile = copy.deepcopy(v6_config["control"]["close_profile"])
    profile["left_hand_index_joint1_actuator"] = {
        "start_fraction": 0.5,
        "end_fraction": 0.95,
    }

    candidate = materialize_pose_preserving_candidate(
        v6_config,
        hand_pose=hand_pose,
        pregrasp_targets_rad=pregrasp,
        grasp_targets_rad=grasp,
        close_profile=profile,
        metadata={"probe": 7},
    )

    assert v6_config == source
    assert candidate["hand_pose"] == hand_pose
    assert candidate["control"]["pregrasp_targets_rad"] == pregrasp
    assert candidate["control"]["grasp_targets_rad"] == grasp
    assert candidate["control"]["close_profile"] == profile
    assert candidate["candidate_metadata"]["probe"] == 7
    assert candidate["candidate_metadata"]["tilt_band_center_deg"] == 32.5
    validate_config(candidate)


def test_materializer_rejects_incomplete_controls(v6_config):
    pregrasp = copy.deepcopy(v6_config["control"]["pregrasp_targets_rad"])
    pregrasp.pop(ACTIVE_ACTUATORS[-1])
    with pytest.raises(ValueError, match="eight active actuators"):
        materialize_pose_preserving_candidate(
            v6_config, pregrasp_targets_rad=pregrasp
        )

    profile = copy.deepcopy(v6_config["control"]["close_profile"])
    profile[ACTIVE_ACTUATORS[0]] = {
        "start_fraction": 0.8,
        "end_fraction": 0.2,
    }
    with pytest.raises(ValueError, match="start_fraction < end_fraction"):
        materialize_pose_preserving_candidate(v6_config, close_profile=profile)


def test_local_candidate_generation_is_seeded_bounded_and_stable(v6_config):
    first = generate_pose_preserving_candidates(v6_config, count=12, seed=41)
    repeated = generate_pose_preserving_candidates(v6_config, count=12, seed=41)
    different = generate_pose_preserving_candidates(v6_config, count=12, seed=42)

    assert first == repeated
    assert first != different
    assert len(first) == 12
    assert first[0]["control"] == v6_config["control"]
    definition = pose_tuning.resolve_experiment(v6_config)
    for candidate in first:
        validate_config(candidate)
        assert definition.search_bounds.contains_pregrasp_targets(
            candidate["control"]["pregrasp_targets_rad"]
        )
        assert definition.search_bounds.contains_targets(
            candidate["control"]["grasp_targets_rad"]
        )
        for interval in candidate["control"]["close_profile"].values():
            assert 0.0 <= interval["start_fraction"]
            assert interval["start_fraction"] < interval["end_fraction"]
            assert interval["end_fraction"] <= 1.0


def test_rank_prioritizes_combined_grasp_and_pose_and_is_worker_order_free(
    v6_config,
):
    combined = _result(8, v6_config, grasp=True, pose=True)
    grasp_only = _result(1, v6_config, grasp=True, pose=False, full=True)
    pose_only = _result(0, v6_config, grasp=False, pose=True, full=True)

    ranked = deterministic_rank_pose_preserving_results(
        [pose_only, combined, grasp_only]
    )
    reverse_ranked = deterministic_rank_pose_preserving_results(
        [grasp_only, combined, pose_only]
    )

    assert [item["candidate_id"] for item in ranked] == [8, 1, 0]
    assert [item["candidate_id"] for item in reverse_ranked] == [8, 1, 0]
    assert acquisition_succeeded(combined)
    assert not acquisition_succeeded(grasp_only)
    assert pose_preserving_candidate_rank(combined) > (
        pose_preserving_candidate_rank(grasp_only)
    )


def test_tuner_returns_confirmed_best_and_honest_near_miss_with_reordered_runner(
    v6_config,
):
    worker_values = []

    def runner(payloads, workers):
        worker_values.append(workers)
        results = []
        for candidate_id, config in payloads:
            results.append(
                _result(
                    candidate_id,
                    config,
                    grasp=candidate_id != 0,
                    pose=candidate_id != 0,
                )
            )
        return list(reversed(results))

    result = tune_pose_preserving_grasp(
        v6_config,
        workers=3,
        seed=20260821,
        run_candidates=runner,
        samples=99,
        refine_top=9,
        refine_per=9,
        dynamic_candidate_count=3,
        local_refine_seed_count=2,
        local_refine_per_seed=2,
        final_candidate_count=2,
        perturbations_per_final=2,
    )

    assert worker_values == [3, 3, 3, 3]
    assert result["campaign_kind"] == "pose_preserving_grasp"
    assert result["stage_counts"] == {
        "acquisition_discovery": 3,
        "acquisition_local_refinement": 4,
        "acquisition_confirmation": 2,
        "acquisition_local_probe": 2,
    }
    assert result["candidate_count"] == 9
    assert result["simulation_count"] == 11
    assert result["perturbation_probe_count"] == 2
    assert result["best"]["search_stage"] == "acquisition_confirmation"
    assert acquisition_succeeded(result["best"])
    assert result["best"]["config"]["experiment_status"][
        "classification"
    ] == "pose_preserving_stable_grasp_only"
    assert result["best"]["local_perturbation_probe"]["passes"] == 2
    assert result["near_miss"]["candidate_id"] == 0
    assert result["selected_band_candidates"] == [result["best"]]


def test_public_tune_dispatches_v6_with_all_cli_budget_overrides(
    v6_config, monkeypatch
):
    captured = {}

    def fake_tuner(value, **kwargs):
        captured["config"] = value
        captured.update(kwargs)
        return {"campaign_kind": "pose_preserving_grasp"}

    monkeypatch.setattr(
        pose_tuning, "tune_pose_preserving_grasp", fake_tuner
    )
    result = search.tune(
        v6_config,
        samples=7,
        refine_top=3,
        refine_per=2,
        workers=4,
        seed=20260821,
        kinematic_samples_per_pitch=20,
        dynamic_candidate_count=10,
        local_refine_seed_count=5,
        local_refine_per_seed=2,
        final_candidate_count=4,
        perturbations_per_final=3,
        fallback_physics_count=0,
        fallback_kinematic_samples_per_pitch=11,
    )

    assert result == {"campaign_kind": "pose_preserving_grasp"}
    assert captured["config"] is v6_config
    assert captured["run_candidates"] is search._run_candidates
    assert captured["workers"] == 4
    assert captured["seed"] == 20260821
    assert captured["samples"] == 7
    assert captured["refine_top"] == 3
    assert captured["refine_per"] == 2
    assert captured["kinematic_samples_per_pitch"] == 20
    assert captured["dynamic_candidate_count"] == 10
    assert captured["local_refine_seed_count"] == 5
    assert captured["local_refine_per_seed"] == 2
    assert captured["final_candidate_count"] == 4
    assert captured["perturbations_per_final"] == 3
    assert captured["fallback_physics_count"] == 0
    assert captured["fallback_kinematic_samples_per_pitch"] == 11
    assert captured["legacy_parameters"] == {
        "samples": 7,
        "refine_top": 3,
        "refine_per": 2,
    }
