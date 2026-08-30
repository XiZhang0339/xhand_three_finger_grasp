from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import mujoco

import xhand_grasp.tuning.high_thumb_variable_size as tuning
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.scene import (
    build_model,
    cube_inertia,
    rpy_degrees_to_rotation_matrix,
)
from xhand_grasp.tuning.pose_preserving_seed_campaign import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / tuning.DEFAULT_SOURCE_CATALOG
TEMPLATE = ROOT / tuning.DEFAULT_TEMPLATE


@pytest.fixture(scope="module")
def sources() -> tuple[dict, ...]:
    return tuning.load_authenticated_v6_sources(CATALOG)


@pytest.fixture(scope="module")
def template() -> dict:
    return load_config(TEMPLATE)


def test_v7_loader_authenticates_exactly_six_v6_pose_preserving_sources(sources):
    assert len(sources) == 6
    assert [value["source_family_id"] for value in sources] == [
        "117060",
        "116060",
        "115060",
        "117061",
        "116062",
        "110064",
    ]
    for source in sources:
        assert len(source["source_sha256"]) == 64
        assert source["config"]["schema_version"] == 6
        assert source["config"]["control"]["manipulation_delta_rad"] == {
            name: 0.0 for name in ACTIVE_ACTUATORS
        }
        assert all(len(source["provenance"][name]) == 64 for name in (
            "catalog_sha256",
            "resolved_config_sha256",
            "trace_sha256",
            "result_sha256",
        ))


@pytest.mark.parametrize("edge_m", (0.052, 0.060, 0.070))
def test_materializer_keeps_each_size_pose_fixed_and_moves_only_hand(
    sources, template, edge_m
):
    source = sources[0]
    candidate = tuning.materialize_high_thumb_candidate(
        source,
        template,
        edge_m=edge_m,
        thumb_target_rad=1.40,
        candidate_id=7,
    )
    validate_config(candidate)
    expected = tuning.cube_world_pose_for_size(
        candidate, edge_m, (0.071, -0.027)
    )
    metadata = candidate["candidate_metadata"]

    assert candidate["cube"]["mass_kg"] == pytest.approx(0.160)
    assert candidate["cube"]["friction"] == pytest.approx(0.8)
    assert candidate["cube"]["center_xy_m"] == [0.071, -0.027]
    assert expected["position_m"][2] == pytest.approx(0.084 + edge_m / 2.0)
    assert metadata["fixed_cube_initial_pose"] == expected
    assert metadata["cube_pose_sampled"] is False
    assert metadata["free_cube_pose_reset_during_run"] is False
    assert candidate["control"]["manipulation_delta_rad"] == {
        name: 0.0 for name in ACTIVE_ACTUATORS
    }
    rotation = rpy_degrees_to_rotation_matrix(candidate["hand_pose"]["rpy_deg"])
    root = np.asarray(candidate["hand_pose"]["translation_m"])
    local = np.asarray(metadata["candidate_cube_in_root_m"])
    np.testing.assert_allclose(
        root + rotation @ local,
        expected["position_m"],
        rtol=0.0,
        atol=2e-15,
    )


@pytest.mark.parametrize("edge_m", (0.052, 0.070))
def test_v7_scene_keeps_fixed_mass_explicit_inertia_free_cube_and_fixed_root(
    sources, template, edge_m
):
    candidate = tuning.materialize_high_thumb_candidate(
        sources[0],
        template,
        edge_m=edge_m,
        thumb_target_rad=1.40,
        candidate_id=8,
    )

    model, info = build_model(candidate)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert model.geom_size[info.cube_geom_id] == pytest.approx(
        [edge_m / 2.0] * 3
    )
    assert model.body_mass[info.cube_body_id] == pytest.approx(0.160)
    assert model.body_inertia[info.cube_body_id] == pytest.approx(
        cube_inertia(edge_m, 0.160)
    )
    assert model.jnt_type[info.cube_joint_id] == mujoco.mjtJoint.mjJNT_FREE
    cube_position = data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 3]
    assert cube_position[2] - edge_m / 2.0 == pytest.approx(
        candidate["scene"]["support_top_z_m"]
    )
    assert model.body_jntnum[info.root_body_id] == 0
    assert model.body_parentid[info.root_body_id] == 0
    assert model.body_mocapid[info.root_body_id] == -1


def test_coarse_generation_is_prefix_stable_balanced_and_covers_full_bounds(
    sources, template
):
    first = tuning.generate_high_thumb_candidates(
        sources,
        template,
        edge_m=0.060,
        thumb_target_rad=1.45,
        count=60,
        seed=20260821,
        cell_index=49,
        validator=None,
    )
    repeated = tuning.generate_high_thumb_candidates(
        sources,
        template,
        edge_m=0.060,
        thumb_target_rad=1.45,
        count=60,
        seed=20260821,
        cell_index=49,
        validator=None,
    )
    prefix = tuning.generate_high_thumb_candidates(
        sources,
        template,
        edge_m=0.060,
        thumb_target_rad=1.45,
        count=12,
        seed=20260821,
        cell_index=49,
        validator=None,
    )

    assert first == repeated
    assert first[:12] == prefix
    assert Counter(value["source_family_id"] for value in first) == {
        source["source_family_id"]: 10 for source in sources
    }
    assert [
        value["config"]["candidate_metadata"]["pose_sample_mode"]
        for value in first[:6]
    ] == ["source_anchor"] * 6
    modes = Counter(
        value["config"]["candidate_metadata"]["pose_sample_mode"]
        for value in first
    )
    assert modes["full_registered_envelope"] > 0
    assert modes["bend_scaled_source_local"] > 0
    assert min(
        value["config"]["control"]["grasp_targets_rad"][
            "left_hand_thumb_rota_joint1_actuator"
        ]
        for value in first
    ) < 0.21
    assert min(
        value["config"]["control"]["grasp_targets_rad"][
            "left_hand_thumb_rota_joint2_actuator"
        ]
        for value in first
    ) < 0.76
    assert all(
        value["config"]["control"]["grasp_targets_rad"][
            tuning.THUMB_BEND_ACTUATOR
        ]
        == 1.45
        for value in first
    )


def test_chunked_generation_reconstructs_the_same_candidate_sequence(sources, template):
    complete = tuning.generate_high_thumb_candidates(
        sources,
        template,
        edge_m=0.062,
        thumb_target_rad=1.35,
        count=18,
        seed=20260821,
        cell_index=7,
        validator=None,
    )
    chunks = (
        tuning.generate_high_thumb_candidates(
            sources,
            template,
            edge_m=0.062,
            thumb_target_rad=1.35,
            count=6,
            start_index=start,
            seed=20260821,
            cell_index=7,
            validator=None,
        )
        for start in (0, 6, 12)
    )
    reconstructed = tuple(value for chunk in chunks for value in chunk)
    assert reconstructed == complete


def test_61mm_near_miss_family_gets_registered_low_pitch_bias(sources, template):
    generated = tuning.generate_high_thumb_candidates(
        sources,
        template,
        edge_m=0.063,
        thumb_target_rad=1.30,
        count=240,
        seed=20260821,
        cell_index=1,
        validator=None,
    )
    biased = [
        value
        for value in generated
        if value["config"]["candidate_metadata"]["pose_sample_mode"]
        == "family_117061_low_pitch_bias"
    ]
    assert biased
    assert {value["source_family_id"] for value in biased} == {"117061"}
    assert all(
        120.1 <= value["config"]["hand_pose"]["rpy_deg"][1] <= 121.0
        for value in biased
    )
    for value in biased:
        validate_config(value["config"])


def test_dynamic_near_miss_anchor_is_replayed_without_inheriting_success(
    sources, template
):
    generated = tuning.generate_high_thumb_candidates(
        sources,
        template,
        edge_m=0.062,
        thumb_target_rad=1.30,
        count=12,
        seed=20260821,
        cell_index=6,
        validator=None,
    )
    anchors = [
        value
        for value in generated
        if value["config"]["candidate_metadata"]["pose_sample_mode"]
        == "validated_dynamic_near_miss_anchor"
    ]

    assert len(anchors) == 1
    anchor = anchors[0]
    assert anchor["source_family_id"] == "117061"
    assert anchor["config"]["hand_pose"]["rpy_deg"][1] == pytest.approx(
        121.97946403250573 - 1.9
    )
    assert anchor["config"]["control"]["close_profile"][
        "left_hand_index_joint1_actuator"
    ]["start_fraction"] == pytest.approx(0.10)
    assert anchor["config"]["experiment_status"] == template["experiment_status"]
    assert anchor["config"]["experiment_status"]["passed"] is False
    assert anchor["config"]["candidate_metadata"]["cube_pose_sampled"] is False
    validate_config(anchor["config"])

    retained = tuning.screen_static_candidates(generated, retain=6)["retained"]
    retained_modes = {
        value["config"]["candidate_metadata"]["pose_sample_mode"]
        for value in retained
    }
    assert "validated_dynamic_near_miss_anchor" in retained_modes


def test_static_retention_reserves_one_candidate_per_source_family():
    values = []
    for index, family in enumerate(("a", "b", "c", "d", "e", "f")):
        values.append(
            {
                "candidate_id": index,
                "source_family_id": family,
                "static_pass": index < 2,
                "static_score": float(index),
                "static_metrics": {
                    "pregrasp_hand_cube_contact": index >= 2,
                    "clean_closure_alpha_count": 1 if index < 2 else 0,
                    "contact_onset_alpha_span": 0.0,
                },
            }
        )
    # Extra globally good entries from family a must not evict other anchors.
    for index in range(6, 12):
        values.append(
            {
                "candidate_id": index,
                "source_family_id": "a",
                "static_pass": True,
                "static_score": 0.01 * index,
                "static_metrics": {
                    "pregrasp_hand_cube_contact": False,
                    "clean_closure_alpha_count": 1,
                    "contact_onset_alpha_span": 0.0,
                },
            }
        )
    retained = tuning._select_static_family_anchors(values, 8)
    assert {value["source_family_id"] for value in retained} == {
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    }


def test_static_screen_scans_the_grouped_close_profile_without_stepping_cube(
    sources, template
):
    candidate = tuning.generate_high_thumb_candidates(
        sources,
        template,
        edge_m=0.061,
        thumb_target_rad=1.30,
        count=1,
        seed=20260821,
        cell_index=0,
        validator=None,
    )
    screened = tuning.screen_static_candidates(candidate, retain=1)
    metrics = screened["retained"][0]["static_metrics"]

    assert metrics["closure_scan_alpha_values"] == pytest.approx(
        np.linspace(0.0, 1.0, 17)
    )
    assert 0.0 <= metrics["selected_closure_alpha"] <= 1.0
    assert set(metrics["first_target_window_alpha"]) == {
        "thumb",
        "index",
        "mid",
    }
    assert metrics["contact_onset_alpha_span"] >= 0.0
    assert "acquisition_qpos_reference" in metrics


def _successful_result(candidate_id, edge, target, family):
    return {
        "candidate_id": candidate_id,
        "source_family_id": family,
        "source_trajectory_id": f"source_{family}",
        "edge_m": edge,
        "thumb_target_rad": target,
        "stage": "acquisition",
        "acquisition_success": True,
        "pose_preservation_success": True,
        "lift_success": False,
        "summary": {"metrics": {}},
        "config": {},
    }


def test_ranking_maximizes_thumb_only_after_hard_success():
    failed_high = _successful_result(1, 0.060, 1.45, "a")
    failed_high["acquisition_success"] = False
    passed_low = _successful_result(2, 0.060, 1.25, "b")
    passed_high = _successful_result(3, 0.060, 1.40, "c")

    assert [value["candidate_id"] for value in tuning.rank_high_thumb_results(
        (failed_high, passed_low, passed_high)
    )] == [3, 2, 1]


def test_failed_candidate_ranking_prefers_measured_near_miss_over_target():
    failed_high = _successful_result(1, 0.060, 1.45, "a")
    failed_high["acquisition_success"] = False
    failed_high["pose_preservation_success"] = False
    failed_high["thumb_bend_trace_metrics"] = {
        "max_consecutive_all_gate_steps_close_verify": 0
    }
    close_lower = _successful_result(2, 0.062, 1.30, "b")
    close_lower["acquisition_success"] = False
    close_lower["pose_preservation_success"] = False
    close_lower["thumb_bend_trace_metrics"] = {
        "max_consecutive_all_gate_steps_close_verify": 115
    }

    ranked = tuning.rank_high_thumb_results((failed_high, close_lower))

    assert [value["candidate_id"] for value in ranked] == [2, 1]


def test_diversity_selection_enforces_bands_edges_families_and_pair_cap():
    values = []
    candidate_id = 1
    for target in (1.30, 1.35, 1.40):
        for edge_index, edge in enumerate((0.052, 0.058, 0.064, 0.070)):
            values.append(
                _successful_result(
                    candidate_id,
                    edge,
                    target,
                    ("a", "b", "c", "d")[edge_index],
                )
            )
            candidate_id += 1
    selection = tuning.select_diverse_grasps(values)

    assert selection.satisfied
    assert len(selection.selected) == 12
    assert selection.distinct_edges == 4
    assert selection.distinct_seed_families == 4
    assert selection.bend_band_counts == {
        "band_2": 4,
        "band_1": 4,
        "band_0": 4,
    }
    assert max(
        Counter(
            (value["edge_m"], value["thumb_target_rad"])
            for value in selection.selected
        ).values()
    ) == 1


def test_lift_generation_is_deterministic_and_does_not_touch_cube_pose(
    sources, template
):
    acquisition = tuning.generate_high_thumb_candidates(
        sources,
        template,
        edge_m=0.060,
        thumb_target_rad=1.40,
        count=1,
        seed=20260821,
        cell_index=0,
    )[0]
    seed_result = {
        **acquisition,
        "stage": "acquisition",
        "acquisition_success": True,
        "pose_preservation_success": True,
        "lift_success": False,
        "summary": {"metrics": {}},
    }
    first = tuning.generate_lift_candidates(
        (seed_result,), template, count_per_seed=3, seed=20260821
    )
    repeated = tuning.generate_lift_candidates(
        (seed_result,), template, count_per_seed=3, seed=20260821
    )

    assert first == repeated
    assert len(first) == 3
    for value in first:
        assert value["stage"] == "lift"
        assert value["config"]["cube"] == acquisition["config"]["cube"]
        assert value["config"]["hand_pose"] == acquisition["config"]["hand_pose"]
        assert value["config"]["candidate_metadata"]["cube_pose_sampled"] is False
        assert any(
            amount != 0.0
            for amount in value["config"]["control"]["manipulation_delta_rad"].values()
        )


def test_hash_verified_candidate_resume_rejects_tampered_trace(
    tmp_path, sources, template
):
    candidate = tuning.generate_high_thumb_candidates(
        sources,
        template,
        edge_m=0.060,
        thumb_target_rad=1.30,
        count=1,
        seed=20260821,
        cell_index=0,
    )[0]
    directory = tmp_path / "candidate"
    directory.mkdir()
    config_path = directory / "resolved_config.json"
    trace_path = directory / "trace.npz"
    result_path = directory / "result.json"
    write_json(config_path, candidate["config"])
    np.savez(trace_path, marker=np.asarray([1.0]))
    payload = {
        "complete": True,
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": canonical_sha256(candidate["config"]),
        "artifacts": {
            "sha256": {
                "resolved_config": file_sha256(config_path),
                "trace": file_sha256(trace_path),
            }
        },
    }
    write_json(result_path, payload)
    job = {**candidate, "output_directory": str(directory), "artifact_directory": "candidate"}

    reused = tuning._load_reusable_candidate(job)
    assert reused is not None and reused["reused"] is True
    np.savez(trace_path, marker=np.asarray([2.0]))
    with pytest.raises(RuntimeError, match="trace file hash mismatch"):
        tuning._load_reusable_candidate(job)


def test_dry_run_is_stable_and_does_not_create_output(tmp_path):
    output = tmp_path / "must_not_exist"
    policy = tuning.HighThumbCampaignPolicy(
        coarse_edges_m=(0.060,), coarse_thumb_targets_rad=(1.30,)
    )
    budget = tuning.HighThumbCampaignBudget(
        static_samples_per_cell=1,
        static_retain_per_cell=1,
        dynamic_candidate_limit=1,
        local_refine_sizes_per_target=1,
        local_refine_seeds_per_size=1,
        local_refine_per_seed=1,
        fine_dynamic_limit=1,
        exact_reverify_limit=1,
        selected_grasp_count=1,
        selected_lift_seed_count=1,
        lift_candidates_per_seed=1,
        perturbations_per_grasp=1,
    )
    first = tuning.run_high_thumb_variable_size_campaign(
        CATALOG, TEMPLATE, output, dry_run=True, policy=policy, budget=budget
    )
    repeated = tuning.run_high_thumb_variable_size_campaign(
        CATALOG, TEMPLATE, output, dry_run=True, policy=policy, budget=budget
    )

    assert first == repeated
    assert first["coarse_static_sample_count"] == 1
    assert first["output_directory_created"] is False
    assert not output.exists()


def test_cli_exposes_resume_workers_and_range_overrides():
    args = tuning.build_parser().parse_args(
        [
            "--resume",
            "--workers",
            "4",
            "--edges-mm",
            "58",
            "60",
            "--thumb-targets-rad",
            "1.30",
            "1.40",
        ]
    )
    assert args.resume is True
    assert args.workers == 4
    assert args.edges_mm == [58.0, 60.0]
    assert args.thumb_targets_rad == [1.30, 1.40]
