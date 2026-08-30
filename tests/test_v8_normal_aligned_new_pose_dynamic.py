from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config, validate_config
from xhand_grasp.tuning.normal_aligned_new_pose_dynamic import (
    DYNAMIC_CAMPAIGN_KIND,
    DynamicPromotionBudget,
    generate_initial_controller_candidates,
    generate_local_pose_controller_candidates,
    generate_locked_exact_candidates,
    load_static_pose_manifest,
    new_pose_grasp_rank,
    rank_new_pose_grasp_results,
    run_new_pose_dynamic_campaign,
    select_diverse_local_parents,
)
from xhand_grasp.tuning.normal_aligned_pose_search import (
    CAMPAIGN_KIND as STATIC_CAMPAIGN_KIND,
    controller_id_for_config,
    pose_id_for_config,
)
from xhand_grasp.tuning.pose_preserving_seed_campaign import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_high_thumb_normal_aligned_"
    "smooth_vertical_lift.json"
)


def _static_metrics(angle: float) -> dict:
    return {
        "static_geometry_pass": True,
        "missing_target_witness_count": 0,
        "off_target_penetrating_count": 0,
        "minimum_active_nondistal_gap_m": 0.001,
        "target_signed_gap_m": [0.001, 0.001, 0.001],
        "closure_angle_deg": [angle, angle, angle],
        "closure_inward_speed_m_s": [0.01, 0.01, 0.01],
        "contact_height_spread_m": 0.002,
    }


def _write_static_report(
    tmp_path: Path,
    count: int = 2,
    *,
    thumb_targets: tuple[float, ...] | None = None,
) -> Path:
    if thumb_targets is not None and len(thumb_targets) != count:
        raise ValueError("thumb_targets must match count")
    template = load_config(TEMPLATE)
    retained = []
    for index in range(count):
        config = copy.deepcopy(template)
        config["hand_pose"]["translation_m"][0] += index * 0.0001
        if thumb_targets is not None:
            config["control"]["grasp_targets_rad"][
                "left_hand_thumb_bend_joint_actuator"
            ] = thumb_targets[index]
        validate_config(config)
        retained.append(
            {
                "candidate_id": 91_000_000_000_000 + index,
                "candidate_sha256": canonical_sha256(config),
                "cell_id": f"cell-{index}",
                "source_family_id": f"family-{index}",
                "pose_id": pose_id_for_config(config),
                "controller_id": controller_id_for_config(config),
                "static_pass": True,
                "static_metrics": _static_metrics(15.0 + index),
                "config": config,
            }
        )
    cell_dir = tmp_path / "cells" / "cell_0"
    cell_dir.mkdir(parents=True)
    cell_result = cell_dir / "result.json"
    write_json(
        cell_result,
        {
            "cell_index": 0,
            "complete": True,
            "retained_count": len(retained),
            "retained": retained,
        },
    )
    report = tmp_path / "search_report.json"
    write_json(
        report,
        {
            "complete": True,
            "experiment_id": template["experiment_id"],
            "campaign_kind": STATIC_CAMPAIGN_KIND,
            "cell_results": [
                {
                    "cell_index": 0,
                    "result_path": str(cell_result.relative_to(tmp_path)),
                    "result_sha256": file_sha256(cell_result),
                    "evaluated_count": 20,
                    "static_pass_count": len(retained),
                    "retained_count": len(retained),
                }
            ],
        },
    )
    return report


def _passing_executor(jobs, workers):
    results = []
    for job in jobs:
        angle = 12.0 + (int(job["candidate_id"]) % 7) * 0.1
        results.append(
            {
                **copy.deepcopy(dict(job)),
                "acquisition_success": True,
                "pose_preservation_success": True,
                "rescue_success": True,
                "classification": "validated_normal_aligned_control_rescue",
                "rank_evidence": {"rescue_success": True},
                "summary": {
                    "metrics": {
                        "verify_max_consecutive_all_gate_steps": 300,
                        "pose_preservation": {
                            "max_translation_m": 0.0001,
                            "max_orientation_drift_deg": 0.2,
                        },
                        "closure_alignment": {
                            "worst_p95_angle_deg": angle,
                            "worst_minimum_inward_speed_m_s": 0.002,
                            "per_finger": {
                                finger: {
                                    "angle_p95_deg": angle,
                                    "minimum_inward_speed_m_s": 0.002,
                                }
                                for finger in ("thumb", "index", "mid")
                            },
                        },
                        "peak_total_distal_contact_force_n": 3.0,
                        "actuator_saturation_fraction": 0.05,
                    },
                    "checks": {
                        "v8_closure_alignment_trace_matches_vectors": True,
                        "closure_alignment_valid_for_all_fingers": True,
                        "closure_alignment_p95_within_limit": True,
                        "closure_inward_speed_positive": True,
                    },
                },
            }
        )
    # Deliberately model completion order changing with worker count.
    return tuple(reversed(results)) if workers > 1 else tuple(results)


def _ranking_record(
    candidate_id: int,
    *,
    angle_deg: float,
    translation_m: float,
    orientation_deg: float,
    gate_steps: int = 0,
    closure_passed: bool = True,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "acquisition_success": False,
        "pose_preservation_success": False,
        "rescue_success": False,
        "summary": {
            "metrics": {
                "verify_max_consecutive_all_gate_steps": gate_steps,
                "pose_preservation": {
                    "max_translation_m": translation_m,
                    "max_orientation_drift_deg": orientation_deg,
                },
                "closure_alignment": {
                    "worst_p95_angle_deg": angle_deg,
                    "worst_minimum_inward_speed_m_s": 0.001,
                    "per_finger": {
                        finger: {
                            "angle_p95_deg": angle_deg,
                            "minimum_inward_speed_m_s": 0.001,
                        }
                        for finger in ("thumb", "index", "mid")
                    },
                },
                "peak_total_distal_contact_force_n": 3.0,
                "actuator_saturation_fraction": 0.05,
            },
            "checks": {
                "v8_closure_alignment_trace_matches_vectors": True,
                "closure_alignment_valid_for_all_fingers": True,
                "closure_alignment_p95_within_limit": closure_passed,
                "closure_inward_speed_positive": True,
            },
        },
    }


def test_new_pose_rank_prioritizes_grasp_gate_and_pose_before_soft_angle():
    low_angle_large_motion = _ranking_record(
        1,
        angle_deg=19.8,
        translation_m=0.003565,
        orientation_deg=0.2,
    )
    higher_angle_better_pose = _ranking_record(
        2,
        angle_deg=25.01,
        translation_m=0.001346,
        orientation_deg=3.856,
    )
    invalid_closure_positive_pose = _ranking_record(
        3,
        angle_deg=180.0,
        translation_m=0.000325,
        orientation_deg=0.5,
        closure_passed=False,
    )
    longer_gate_run = _ranking_record(
        4,
        angle_deg=29.0,
        translation_m=0.0015,
        orientation_deg=4.0,
        gate_steps=249,
    )

    ranked = rank_new_pose_grasp_results(
        (
            low_angle_large_motion,
            invalid_closure_positive_pose,
            higher_angle_better_pose,
            longer_gate_run,
        )
    )
    assert [value["candidate_id"] for value in ranked] == [4, 2, 1, 3]
    assert new_pose_grasp_rank(higher_angle_better_pose) < new_pose_grasp_rank(
        low_angle_large_motion
    )
    assert new_pose_grasp_rank(low_angle_large_motion) < new_pose_grasp_rank(
        invalid_closure_positive_pose
    )


def _local_parent_record(
    candidate_id: int,
    *,
    thumb_target_rad: float,
    edge_m: float,
    gate_steps: int,
) -> dict:
    value = _ranking_record(
        candidate_id,
        angle_deg=25.0,
        translation_m=0.001,
        orientation_deg=2.0,
        gate_steps=gate_steps,
    )
    value.update(
        {
            "pose_id": f"pose-{candidate_id}",
            "thumb_target_rad": thumb_target_rad,
            "edge_m": edge_m,
        }
    )
    return value


def test_local_parent_selection_preserves_thumb_bands_when_global_top_is_low():
    records = [
        _local_parent_record(
            index,
            thumb_target_rad=1.25,
            edge_m=0.060 + (index % 3) * 0.001,
            gate_steps=1_000 - index,
        )
        for index in range(30)
    ]
    next_id = 100
    for target_index, target in enumerate((1.30, 1.35, 1.40, 1.45)):
        for edge_index, edge in enumerate((0.066, 0.067)):
            records.append(
                _local_parent_record(
                    next_id,
                    thumb_target_rad=target,
                    edge_m=edge,
                    gate_steps=100 - target_index * 10 - edge_index,
                )
            )
            next_id += 1

    selected = select_diverse_local_parents(records, count=20)
    reversed_selected = select_diverse_local_parents(
        tuple(reversed(records)), count=20
    )
    assert len(selected) == 20
    assert len({value["pose_id"] for value in selected}) == 20
    assert [value["candidate_id"] for value in selected] == [
        value["candidate_id"] for value in reversed_selected
    ]
    for target in (1.25, 1.30, 1.35, 1.40, 1.45):
        assert sum(value["thumb_target_rad"] == target for value in selected) >= 2


def test_local_parent_selection_prefers_distinct_edges_within_thumb_band():
    records = (
        _local_parent_record(
            1, thumb_target_rad=1.25, edge_m=0.060, gate_steps=30
        ),
        _local_parent_record(
            2, thumb_target_rad=1.25, edge_m=0.060, gate_steps=29
        ),
        _local_parent_record(
            3, thumb_target_rad=1.25, edge_m=0.061, gate_steps=28
        ),
    )
    selected = select_diverse_local_parents(records, count=2)
    assert [value["candidate_id"] for value in selected] == [1, 3]
    assert {value["edge_m"] for value in selected} == {0.060, 0.061}


def test_static_report_authenticates_per_cell_artifact_and_separates_hashes(tmp_path):
    report = _write_static_report(tmp_path)
    manifest = load_static_pose_manifest(report)
    assert manifest["deduplicated_pose_count"] == 2
    assert manifest["raw_retained_count"] == 2
    assert manifest["source_cell_results"][0]["result_sha256"]

    candidates = generate_initial_controller_candidates(
        manifest["poses"], validator=validate_config
    )
    assert len(candidates) == 16
    for pose in manifest["poses"]:
        family = [value for value in candidates if value["pose_id"] == pose["pose_id"]]
        assert len(family) == 8
        assert len({value["controller_id"] for value in family}) == 8
    kinds = {
        value["config"]["candidate_metadata"]["controller_seed_kind"]
        for value in candidates
    }
    assert kinds == {
        "original_control",
        "validated_67_control",
        "analytic_contact_synchronization",
        "latin_hypercube",
    }


def test_static_result_hash_tampering_is_rejected(tmp_path):
    report = _write_static_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    result_path = report.parent / payload["cell_results"][0]["result_path"]
    result_path.write_text(result_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_static_pose_manifest(report)


def test_local_joint_pose_controller_search_and_locked_replay_are_deterministic(tmp_path):
    manifest = load_static_pose_manifest(_write_static_report(tmp_path))
    initial = list(generate_initial_controller_candidates(manifest["poses"], validator=None))
    parents = [
        {
            **initial[0],
            "summary": _passing_executor([initial[0]], 1)[0]["summary"],
            "acquisition_success": True,
            "pose_preservation_success": True,
        },
        {
            **initial[8],
            "summary": _passing_executor([initial[8]], 1)[0]["summary"],
            "acquisition_success": True,
            "pose_preservation_success": True,
        },
    ]
    first = generate_local_pose_controller_candidates(
        parents, count_per_pose=3, validator=None
    )
    second = generate_local_pose_controller_candidates(
        parents, count_per_pose=3, validator=None
    )
    assert first == second
    assert len(first) == 6
    assert any(value["pose_id"] != parents[0]["pose_id"] for value in first[:3])
    assert any(
        value["controller_id"] != parents[0]["controller_id"] for value in first[:3]
    )

    replay = generate_locked_exact_candidates(parents, count=2, validator=None)
    assert [value["pose_id"] for value in replay] == [value["pose_id"] for value in parents]
    assert [value["controller_id"] for value in replay] == [
        value["controller_id"] for value in parents
    ]
    assert all(
        value["config"]["candidate_metadata"]["locked_timestep_s"] == 0.001
        for value in replay
    )


def test_small_budget_campaign_is_worker_order_independent(tmp_path):
    report = _write_static_report(tmp_path / "static")
    budget = DynamicPromotionBudget(
        controller_seeds_per_pose=8,
        local_pose_count=2,
        local_refine_per_pose=2,
        exact_candidate_count=2,
    )
    one = run_new_pose_dynamic_campaign(
        report,
        TEMPLATE,
        tmp_path / "one",
        workers=1,
        budget=budget,
        executor=_passing_executor,
    )
    four = run_new_pose_dynamic_campaign(
        report,
        TEMPLATE,
        tmp_path / "four",
        workers=4,
        budget=budget,
        executor=_passing_executor,
    )
    assert one["campaign_kind"] == DYNAMIC_CAMPAIGN_KIND
    assert one["initial_candidate_count"] == four["initial_candidate_count"] == 16
    assert one["local_candidate_count"] == four["local_candidate_count"] == 4
    assert one["exact_candidate_count"] == four["exact_candidate_count"] == 2
    assert one["candidate_count"] == four["candidate_count"] == 22
    assert one["certified_grasp_count"] == four["certified_grasp_count"] == 2
    assert one["best_candidate"]["candidate_id"] == four["best_candidate"]["candidate_id"]
    assert [value["candidate_id"] for value in one["results"]] == [
        value["candidate_id"] for value in four["results"]
    ]
    assert one["local_parent_selection"] == four["local_parent_selection"]
    assert one["local_parent_selection"]["selected_count"] == 2
    assert one["local_parent_selection"]["unique_pose_count"] == 2
    assert sum(
        value["count"]
        for value in one["local_parent_selection"]["per_thumb_target"]
    ) == 2
    assert sum(
        value["count"] for value in one["local_parent_selection"]["per_edge_m"]
    ) == 2
    catalog = json.loads((tmp_path / "one" / "grasp_catalog.json").read_text())
    assert catalog["certified_grasp_count"] == 2


def test_exact_replay_keeps_each_physically_passing_thumb_band(tmp_path):
    thumb_targets = (1.25, 1.30, 1.35, 1.40, 1.45)
    report = _write_static_report(
        tmp_path / "static",
        count=len(thumb_targets),
        thumb_targets=thumb_targets,
    )
    result = run_new_pose_dynamic_campaign(
        report,
        TEMPLATE,
        tmp_path / "dynamic",
        workers=4,
        budget=DynamicPromotionBudget(
            controller_seeds_per_pose=8,
            local_pose_count=5,
            local_refine_per_pose=1,
            exact_candidate_count=5,
        ),
        executor=_passing_executor,
    )

    assert result["exact_candidate_count"] == 5
    assert {value["thumb_target_rad"] for value in result["results"]} == set(
        thumb_targets
    )
    assert {
        value["thumb_target_rad"]
        for value in result["exact_parent_selection"]["per_thumb_target"]
        if value["count"]
    } == set(thumb_targets)
    assert all(
        value["count"] == 1
        for value in result["exact_parent_selection"]["per_thumb_target"]
    )
