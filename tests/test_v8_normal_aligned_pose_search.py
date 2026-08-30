from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

import xhand_grasp.tuning.normal_aligned_pose_search as pose_search_module
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.tuning.normal_aligned_pose_search import (
    DEFAULT_EDGES_M,
    DEFAULT_THUMB_TARGETS_RAD,
    PoseSearchCell,
    StaticPoseSearchBudget,
    assert_static_candidate_scope,
    coarse_closure_alphas,
    controller_id_for_config,
    cube_initial_world_pose,
    generate_pose_cell_candidates,
    pose_id_for_config,
    pose_search_cells,
    pose_search_trigger_report,
    retain_top_k_per_cell,
    screen_static_pose_cell,
    screen_static_pose_chunk_two_stage,
    static_pose_search_budget_report,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_high_thumb_normal_aligned_"
    "smooth_vertical_lift.json"
)


@pytest.fixture(scope="module")
def template() -> dict:
    return load_config(TEMPLATE_PATH)


def _cell(template: dict, edge_m: float, thumb_target_rad: float) -> PoseSearchCell:
    return next(
        value
        for value in pose_search_cells(template)
        if math.isclose(value.edge_m, edge_m, abs_tol=1e-12)
        and math.isclose(value.thumb_target_rad, thumb_target_rad, abs_tol=1e-12)
    )


def _screened_record(
    candidate_id: int,
    *,
    cell_index: int,
    passed: bool,
    angle_deg: float,
    gap_m: float = 0.001,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "cell_index": cell_index,
        "pose_id": f"pose-{candidate_id}",
        "controller_id": f"controller-{candidate_id}",
        "static_metrics": {
            "static_geometry_pass": passed,
            "missing_target_witness_count": 0,
            "off_target_penetrating_count": 0,
            "minimum_active_nondistal_gap_m": 0.001,
            "target_signed_gap_m": [gap_m, gap_m, gap_m],
            "closure_angle_deg": [angle_deg, angle_deg, angle_deg],
            "closure_inward_speed_m_s": [0.01, 0.01, 0.01],
            "contact_height_spread_m": 0.001,
            "closure_alpha": 0.5,
        },
    }


def test_registered_grid_and_declared_budget_are_exact(template):
    cells = pose_search_cells(template)
    assert len(cells) == 55
    assert tuple(dict.fromkeys(cell.edge_m for cell in cells)) == DEFAULT_EDGES_M
    assert (
        tuple(dict.fromkeys(cell.thumb_target_rad for cell in cells))
        == DEFAULT_THUMB_TARGETS_RAD
    )
    assert cells[0].cell_id == "edge_60mm_thumb_1.25rad"
    assert cells[-1].cell_id == "edge_70mm_thumb_1.45rad"
    assert [cell.cell_index for cell in cells] == list(range(55))

    budget = StaticPoseSearchBudget().as_dict(cell_count=len(cells))
    assert budget == {
        "cell_count": 55,
        "samples_per_cell": 10_000,
        "declared_sample_count": 550_000,
        "retain_per_cell": 4,
        "maximum_retained_count": 220,
    }


def test_generation_is_prefix_stable_and_never_samples_cube_pose(template):
    cell = _cell(template, 0.060, 1.40)
    short = generate_pose_cell_candidates(
        template, [template], cell, count=6, validator=validate_config
    )
    long = generate_pose_cell_candidates(
        template, [template], cell, count=10, validator=validate_config
    )
    assert short == long[:6]
    assert len({record["candidate_id"] for record in long}) == 10

    fixed_world_pose = cube_initial_world_pose(long[0]["config"])
    for record in long:
        config = record["config"]
        assert config["cube"] == long[0]["config"]["cube"]
        assert cube_initial_world_pose(config) == fixed_world_pose
        assert config["cube"]["edge_m"] == pytest.approx(0.060)
        assert config["cube"]["mass_kg"] == pytest.approx(0.160)
        assert config["cube"]["friction"] == pytest.approx(0.8)
        assert config["control"]["grasp_targets_rad"][
            "left_hand_thumb_bend_joint_actuator"
        ] == pytest.approx(1.40)
        assert set(config["control"]["manipulation_delta_rad"].values()) == {
            0.0
        }
        metadata = config["candidate_metadata"]
        assert metadata["cube_pose_sampled"] is False
        assert metadata["free_cube_pose_reset_during_scan"] is False
        assert set(metadata["sampled_fields"]) == {
            "hand_pose",
            "control.pregrasp_targets_rad",
            "control.grasp_targets_rad",
        }


def test_pose_and_controller_hashes_have_disjoint_ownership(template):
    control_changed = copy.deepcopy(template)
    control_changed["control"]["pregrasp_targets_rad"][
        "left_hand_index_joint1_actuator"
    ] += 0.01
    assert pose_id_for_config(control_changed) == pose_id_for_config(template)
    assert controller_id_for_config(control_changed) != controller_id_for_config(
        template
    )

    pose_changed = copy.deepcopy(template)
    pose_changed["hand_pose"]["translation_m"][0] += 0.001
    assert pose_id_for_config(pose_changed) != pose_id_for_config(template)
    assert controller_id_for_config(pose_changed) == controller_id_for_config(
        template
    )


def test_static_scope_rejects_cube_and_manipulation_mutation(template):
    cell = _cell(template, 0.067, 1.25)
    record = generate_pose_cell_candidates(template, [template], cell, count=1)[0]
    config = record["config"]
    fixed_pose = config["candidate_metadata"]["fixed_cube_initial_world_pose"]
    assert_static_candidate_scope(
        config,
        template=template,
        cell=cell,
        expected_cube_world_pose=fixed_pose,
    )

    cube_changed = copy.deepcopy(config)
    cube_changed["cube"]["center_xy_m"][0] += 0.001
    with pytest.raises(ValueError, match="cube configuration"):
        assert_static_candidate_scope(
            cube_changed,
            template=template,
            cell=cell,
            expected_cube_world_pose=fixed_pose,
        )

    manipulation_changed = copy.deepcopy(config)
    manipulation_changed["control"]["manipulation_delta_rad"][
        ACTIVE_ACTUATORS[0]
    ] = 0.01
    with pytest.raises(ValueError, match="zero manipulation"):
        assert_static_candidate_scope(
            manipulation_changed,
            template=template,
            cell=cell,
            expected_cube_world_pose=fixed_pose,
        )


def test_real_static_scan_reports_distal_witness_and_jacobian_closure(template):
    cell = _cell(template, 0.067, 1.25)
    candidates = generate_pose_cell_candidates(
        template, [template], cell, count=1, validator=validate_config
    )
    result = screen_static_pose_cell(candidates, top_k=1, alpha_count=5)
    assert result["evaluated_count"] == 1
    assert result["retained_count"] == 1
    metrics = result["retained"][0]["static_metrics"]
    assert metrics["cube_freejoint_qpos_unchanged"] is True
    assert metrics["missing_target_witness_count"] == 0
    assert set(metrics["target_witness"]) == {"thumb", "index", "mid"}
    assert [metrics["target_witness"][finger]["face"] for finger in (
        "thumb",
        "index",
        "mid",
    )] == ["X_NEG", "X_POS", "X_POS"]
    assert np.isfinite(metrics["target_signed_gap_m"]).all()
    assert np.isfinite(metrics["closure_angle_deg"]).all()
    assert np.isfinite(metrics["closure_inward_speed_m_s"]).all()
    assert min(metrics["closure_inward_speed_m_s"]) > 0.0
    for witness in metrics["target_witness"].values():
        assert witness["distal_geom_id"] >= 0
        assert len(witness["cube_point_world_m"]) == 3
        assert len(witness["closure_command_velocity_world_m_s"]) == 3
        assert witness["normal_alignment"] >= 0.95


def test_coarse_grid_includes_uniform_samples_and_all_profile_breakpoints(template):
    alphas = coarse_closure_alphas(template, uniform_count=3)
    assert alphas[0] == 0.0
    assert alphas[-1] == 1.0
    assert 0.5 in alphas
    assert {0.04, 0.13, 0.15}.issubset(alphas)
    assert len(alphas) < 33


def test_two_stage_promotes_expanded_near_candidate_and_retains_full_evidence(
    template, monkeypatch
):
    cell = _cell(template, 0.067, 1.25)
    candidates = generate_pose_cell_candidates(template, [template], cell, count=3)
    safe_id = int(candidates[0]["candidate_id"])
    fully_scanned: list[int] = []

    def fake_coarse(model, data, info, record, **kwargs):
        safe = int(record["candidate_id"]) == safe_id
        gap = 0.010 if safe else 0.030
        return {
            "scan_kind": "coarse_promotion_only",
            "alpha_sample_count": 6,
            "declared_alpha_values": [0.0, 0.04, 0.13, 0.15, 0.5, 1.0],
            "closure_alpha": 0.5,
            # Simulate a valid contact between coarse samples: there is no
            # sampled hard pass, but the 8 mm expanded envelope reaches it.
            "coarse_sampled_pass_hint": False,
            "safe_promotion_eligible": safe,
            "safe_early_reject_reason": None,
            "near_margin_m": 0.008,
            "missing_target_witness_count": 0,
            "off_target_penetrating_count": 0,
            "target_signed_gap_m": [gap, gap, gap],
            "promotion_signed_gap_m": [gap, gap, gap],
            "closure_angle_deg": [20.0, 20.0, 20.0],
            "closure_inward_speed_m_s": [0.01, 0.01, 0.01],
            "closure_tangent_speed_m_s": [0.0, 0.0, 0.0],
            "contact_height_spread_m": 0.002,
            "minimum_active_nondistal_gap_m": 0.001,
            "pregrasp_minimum_hand_gap_m": 0.001,
            "target_witness": {},
            "cube_freejoint_qpos_unchanged": True,
        }

    def fake_full(model, data, info, record, **kwargs):
        fully_scanned.append(int(record["candidate_id"]))
        return {
            "scan_kind": "full_static_evidence",
            "alpha_sample_count": int(kwargs["alpha_count"]),
            "declared_alpha_values": list(
                np.linspace(0.0, 1.0, int(kwargs["alpha_count"]))
            ),
            "static_geometry_pass": True,
            "missing_target_witness_count": 0,
            "off_target_penetrating_count": 0,
            "minimum_active_nondistal_gap_m": 0.001,
            "target_signed_gap_m": [0.001, 0.001, 0.001],
            "closure_angle_deg": [20.0, 20.0, 20.0],
            "closure_inward_speed_m_s": [0.01, 0.01, 0.01],
            "contact_height_spread_m": 0.002,
            "closure_alpha": 0.625,
        }

    monkeypatch.setattr(pose_search_module, "_evaluate_candidate_coarse", fake_coarse)
    monkeypatch.setattr(pose_search_module, "_evaluate_candidate_static", fake_full)
    result = screen_static_pose_chunk_two_stage(
        candidates,
        top_k=1,
        alpha_count=33,
        coarse_alpha_count=3,
        promotion_count=1,
    )
    assert result["evaluated_count"] == 3
    assert result["coarse_scan_count"] == 3
    assert result["full_scan_count"] == 1
    assert result["not_promoted_count"] == 2
    assert fully_scanned == [safe_id]
    retained = result["retained"][0]
    assert retained["promotion_reason"] == "expanded_near_envelope"
    assert retained["static_pass"] is True
    assert retained["static_evidence"]["coarse_scan_used_for_pass"] is False
    assert retained["static_metrics"]["alpha_sample_count"] == 33

    fully_scanned.clear()
    reversed_result = screen_static_pose_chunk_two_stage(
        tuple(reversed(candidates)),
        top_k=1,
        alpha_count=33,
        coarse_alpha_count=3,
        promotion_count=1,
    )
    assert fully_scanned == [safe_id]
    assert reversed_result["retained"] == result["retained"]


def test_top_k_is_independent_of_worker_completion_order():
    records = [
        _screened_record(3, cell_index=0, passed=False, angle_deg=15.0),
        _screened_record(2, cell_index=0, passed=True, angle_deg=25.0),
        _screened_record(1, cell_index=0, passed=True, angle_deg=10.0),
        _screened_record(9, cell_index=1, passed=True, angle_deg=20.0),
    ]
    forward = retain_top_k_per_cell(records, top_k=2)
    reverse = retain_top_k_per_cell(list(reversed(records)), top_k=2)
    assert forward == reverse
    assert [value["candidate_id"] for value in forward[0]] == [1, 2]
    assert [value["candidate_id"] for value in forward[1]] == [9]


def test_trigger_and_budget_reports_are_explicit_and_static_only(template):
    insufficient = pose_search_trigger_report(
        template,
        rescued_full_pass_count=4,
        best_worst_closure_angle_deg=19.0,
        force_thumb_band_diversity_search=False,
    )
    assert insufficient["triggered"] is True
    assert insufficient["reasons"] == [
        "rescued_full_pass_count_below_required"
    ]

    angle_limited = pose_search_trigger_report(
        template,
        rescued_full_pass_count=5,
        best_worst_closure_angle_deg=21.0,
        force_thumb_band_diversity_search=False,
    )
    assert angle_limited["reasons"] == [
        "closure_optimization_target_not_met"
    ]
    assert pose_search_trigger_report(
        template,
        rescued_full_pass_count=5,
        best_worst_closure_angle_deg=19.0,
        force_thumb_band_diversity_search=False,
    )["triggered"] is False
    assert pose_search_trigger_report(
        template,
        rescued_full_pass_count=5,
        best_worst_closure_angle_deg=19.0,
    )["reasons"] == ["thumb_band_diversity_search_declared"]

    partial = static_pose_search_budget_report(
        template,
        completed_cell_results=[
            {
                "cell_index": 0,
                "evaluated_count": 10_000,
                "static_pass_count": 7,
                "retained_count": 4,
            }
        ],
    )
    assert partial["declared"]["declared_sample_count"] == 550_000
    assert partial["completed_cell_count"] == 1
    assert partial["remaining_cell_count"] == 54
    assert partial["evaluated_candidate_count"] == 10_000
    assert partial["coarse_scan_count"] == 10_000
    assert partial["full_scan_count"] == 10_000
    assert partial["not_promoted_count"] == 0
    assert partial["dynamics_scheduled"] is False
    assert partial["complete"] is False
