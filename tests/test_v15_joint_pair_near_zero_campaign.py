from __future__ import annotations

from pathlib import Path

import numpy as np

from xhand_grasp.tuning.joint_pair_near_zero_campaign import (
    SOURCE_CANDIDATE_ID,
    JointPairNearZeroBudget,
    authenticate_near_zero_source,
    generate_static_perturbations,
    grasp_control_variants,
    joint_pair_feedback_variants,
    rank_near_zero_candidates,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_v15_declared_budget_has_the_planned_derived_counts() -> None:
    budget = JointPairNearZeroBudget()

    assert budget.static_start_count == 512
    assert budget.dynamic_grasp_count == 384
    assert budget.sequential_plan_count == 64
    assert budget.feedback_grid_count == 1_024
    assert budget.feedback_refine_count == 1_024
    assert budget.exact_rerun_count == 16
    assert budget.final_candidate_count == 5
    assert budget.robustness_trials == 50
    assert budget.robustness_required_passes == 45


def test_v15_source_authenticates_the_sealed_near_zero_member() -> None:
    source = authenticate_near_zero_source(REPOSITORY_ROOT)

    assert source.config_path.is_file()
    assert source.result_path.is_file()
    assert source.trace_path.is_file()
    assert source.catalog_path.is_file()
    assert source.result["candidate_id"] == SOURCE_CANDIDATE_ID
    assert source.result["grasp_success"] is True
    assert len(source.source_id) == 64


def test_v15_static_perturbations_are_bounded_and_deterministic() -> None:
    first = generate_static_perturbations()
    second = generate_static_perturbations()

    assert first == second
    assert len(first) == 512
    assert first[0].joint_qpos_offset_rad == (0.0,) * 8
    assert first[0].root_delta_cube_m == (0.0,) * 3
    assert first[0].wrist_local_rotvec_deg == (0.0,) * 3
    assert len({row.candidate_id for row in first}) == len(first)

    joint = np.asarray([row.joint_qpos_offset_rad for row in first])
    translation = np.asarray([row.root_delta_cube_m for row in first])
    rotvec = np.asarray([row.wrist_local_rotvec_deg for row in first])
    assert np.max(np.abs(joint)) <= 0.03
    assert np.max(np.abs(translation)) <= 0.0015
    assert np.max(np.linalg.norm(rotvec, axis=1)) <= 2.0 + 1e-12


def test_v15_grasp_and_feedback_grids_have_canonical_cartesian_order() -> None:
    grasp = grasp_control_variants()
    assert len(grasp) == 6
    assert [(row.close_s, row.mode) for row in grasp] == [
        (1.25, "original"),
        (1.25, "synchronized_preload"),
        (1.5, "original"),
        (1.5, "synchronized_preload"),
        (1.75, "original"),
        (1.75, "synchronized_preload"),
    ]
    feedback = joint_pair_feedback_variants()
    assert len(feedback) == 16
    assert (feedback[0].alignment_gain, feedback[0].slip_recovery_gain_rad_per_m) == (
        0.25,
        2.0,
    )
    assert (feedback[-1].alignment_gain, feedback[-1].slip_recovery_gain_rad_per_m) == (
        1.0,
        8.0,
    )
    assert len({row.feedback_variant_id for row in feedback}) == 16


def test_v15_rank_is_contact_and_alignment_first_and_order_independent() -> None:
    def candidate(
        candidate_id: int,
        *,
        passed: bool,
        perturbations: int,
        angle_max: float,
        angle_p95: float,
        contact_duty: float,
        lift: float,
    ) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "full_success": passed,
            "perturbation_pass_count": perturbations,
            "summary": {
                "metrics": {
                    "joint_pair_alignment": {
                        "operation_angle_max_deg": angle_max,
                        "operation_angle_p95_deg": angle_p95,
                        "operation_within_limit_duty": 1.0,
                        "operation_longest_violation_s": 0.0,
                    },
                    "contact_preserving_planned_lift": {
                        "target_face_effective_duty": {
                            "thumb": contact_duty,
                            "index": contact_duty,
                            "mid": contact_duty,
                        },
                        "simultaneous_longest_contact_loss_s": 0.0,
                    },
                    "operation_median_lift_m": lift,
                    "motion_smoothness": {
                        "operation_max_lateral_displacement_m": 0.0,
                        "operation_max_orientation_drift_deg": 0.0,
                    },
                    "actuator_saturation_fraction": 0.0,
                }
            },
        }

    passed_robust = candidate(
        3,
        passed=True,
        perturbations=15,
        angle_max=0.8,
        angle_p95=0.4,
        contact_duty=0.995,
        lift=0.010,
    )
    passed_alignment = candidate(
        2,
        passed=True,
        perturbations=14,
        angle_max=0.2,
        angle_p95=0.1,
        contact_duty=1.0,
        lift=0.012,
    )
    failed_high_lift = candidate(
        1,
        passed=False,
        perturbations=16,
        angle_max=0.05,
        angle_p95=0.02,
        contact_duty=1.0,
        lift=0.020,
    )

    expected = [3, 2, 1]
    for records in (
        [failed_high_lift, passed_alignment, passed_robust],
        [passed_robust, failed_high_lift, passed_alignment],
    ):
        ranked = rank_near_zero_candidates(records)
        assert [row["candidate_id"] for row in ranked] == expected


def test_v15_rank_reads_full_reset_evaluator_alignment_metric_names() -> None:
    def candidate(candidate_id: int, maximum: float) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "full_success": True,
            "summary": {
                "passed": True,
                "metrics": {
                    "joint_pair_alignment": {
                        "operation_max_deg": maximum,
                        "operation_p95_deg": maximum * 0.4,
                        "operation_within_limit_duty": 1.0,
                        "operation_longest_violation_s": 0.0,
                    }
                },
            },
        }

    ranked = rank_near_zero_candidates(
        (candidate(2, 0.8), candidate(1, 0.4))
    )
    assert [row["candidate_id"] for row in ranked] == [1, 2]
