from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from xhand_grasp.config import load_config
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.contact_preserving_candidate_artifacts import (
    V14CandidateArtifactBundle,
)
from xhand_grasp.tuning.contact_preserving_refinement_rescue import (
    PARAMETER_BLOCKS,
    RADIUS_FRACTIONS,
    AuthenticatedFormalV2RescueSource,
    AuthenticatedRescueCandidate,
    RefinementRescueBudget,
    RefinementRescueParent,
    allocate_refinement_rescue_budget,
    build_refinement_rescue_job_specs,
    rank_refinement_rescue_results,
    refinement_rescue_parent_eligible,
    refinement_rescue_rank_evidence,
    select_refinement_rescue_parents,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
V14_CONFIG = (
    REPO_ROOT
    / "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _summary(
    candidate_id: int = 1,
    *,
    jerk: float = 6.5,
    median_lift: float = 0.0103,
    minimum_lift: float = 0.0102,
    lateral: float = 0.001,
    failed_checks: list[str] | None = None,
) -> dict:
    checks = {
        "operation_executed": True,
        "manipulation_completed": True,
        "v14_operation_did_not_abort": True,
        "v14_plan_progress_reached_one": True,
        "v14_thumb_contact_duty_at_least_99_percent": True,
        "v14_index_contact_duty_at_least_99_percent": True,
        "v14_middle_contact_duty_at_least_99_percent": True,
        "v14_simultaneous_contact_duty_at_least_99_percent": True,
        "v14_thumb_contact_loss_within_limit": True,
        "v14_index_contact_loss_within_limit": True,
        "v14_middle_contact_loss_within_limit": True,
        "v14_simultaneous_contact_loss_within_limit": True,
        "no_palm_ring_or_pinky_contact": True,
        "active_nondistal_contacts_within_limit": True,
    }
    return {
        "candidate_id": candidate_id,
        "summary": {
            "passed": False,
            "failed_checks": list(
                failed_checks
                if failed_checks is not None
                else ["smooth_motion_jerk_within_limit"]
            ),
            "stage_status": {
                "grasp_success": True,
                "full_success": False,
            },
            "checks": checks,
            "metrics": {
                "controller_aborted": False,
                "forbidden_contact_steps": 0,
                "material_active_nondistal_duty": 0.0,
                "actuator_saturation_fraction": 0.02,
                "operation_median_lift_m": median_lift,
                "operation_minimum_lift_m": minimum_lift,
                "contact_preserving_planned_lift": {
                    "operation_aborted": False,
                    "required_contact_duty": 0.99,
                    "allowed_contact_loss_steps": 10,
                    "final_plan_progress": 1.0,
                    "maximum_plan_progress": 1.0,
                    "target_face_effective_duty": {
                        "thumb": 1.0,
                        "index": 1.0,
                        "mid": 1.0,
                    },
                    "simultaneous_target_face_effective_duty": 1.0,
                    "longest_contact_loss_steps": {
                        "thumb": 0,
                        "index": 0,
                        "mid": 0,
                    },
                    "simultaneous_longest_contact_loss_steps": 0,
                },
                "motion_smoothness": {
                    "operation_cumulative_height_backtrack_m": 0.0,
                    "operation_peak_filtered_upward_speed_m_s": 0.006,
                    "operation_peak_abs_filtered_acceleration_m_s2": 0.02,
                    "operation_peak_abs_filtered_jerk_m_s3": jerk,
                    "operation_hold_entry_linear_speed_m_s": 0.0002,
                    "operation_max_lateral_displacement_m": lateral,
                    "operation_max_orientation_drift_deg": 9.4,
                },
            },
        },
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["summary"]["stage_status"].update(grasp_success=False),
        lambda value: value["summary"]["checks"].update(operation_executed=False),
        lambda value: value["summary"]["checks"].pop("manipulation_completed"),
        lambda value: value["summary"]["metrics"]["contact_preserving_planned_lift"].update(final_plan_progress=0.999),
        lambda value: value["summary"]["metrics"]["contact_preserving_planned_lift"]["target_face_effective_duty"].update(index=0.989),
        lambda value: value["summary"]["metrics"].update(forbidden_contact_steps=1),
        lambda value: value["summary"]["metrics"].update(material_active_nondistal_duty=0.02),
    ),
)
def test_parent_eligibility_is_fail_closed(mutation):
    record = _summary()
    assert refinement_rescue_parent_eligible(record)
    mutation(record)
    assert not refinement_rescue_parent_eligible(record)


def test_rescue_rank_is_contact_first_then_nonjerk_overall_margin():
    requested_parent = _summary(
        14027467936228193,
        jerk=6.580903,
        median_lift=0.010319,
        minimum_lift=0.010266,
        lateral=0.001081,
    )
    lower_jerk_but_other_failures = _summary(
        2,
        jerk=5.01,
        median_lift=0.00894,
        minimum_lift=0.00890,
        lateral=0.00419,
        failed_checks=[
            "median_lift_reached",
            "operation_median_lift_reached",
            "smooth_motion_jerk_within_limit",
            "smooth_motion_lateral_displacement_within_limit",
        ],
    )
    ranked = rank_refinement_rescue_results(
        [lower_jerk_but_other_failures, requested_parent]
    )
    assert ranked[0]["candidate_id"] == 14027467936228193
    evidence = refinement_rescue_rank_evidence(ranked[0])
    assert evidence["parent_eligible"] is True
    assert evidence["non_jerk_failure_count"] == 0
    assert evidence["minimum_contact_duty"] == 1.0


def _authenticated_candidate(
    tmp_path: Path,
    candidate_id: int,
    config: dict,
    record: dict,
) -> AuthenticatedRescueCandidate:
    root = tmp_path / f"candidate_{candidate_id}"
    root.mkdir()
    config_path = root / "resolved_config.json"
    result_path = root / "result.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = {
        "candidate_id": candidate_id,
        "result_semantic_sha256": f"{candidate_id:064x}"[-64:],
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    bundle = V14CandidateArtifactBundle(
        candidate_id=candidate_id,
        destination=root,
        config_path=config_path,
        result_path=result_path,
        trace_path=None,
        trace_retained=False,
        trace_retention_reason="grasp_success_summary_only",
        result=result,
        reused=True,
    )
    return AuthenticatedRescueCandidate(
        stage="joint_refinement",
        candidate_id=candidate_id,
        artifact_directory=root.name,
        config_semantic_sha256=canonical_sha256(config),
        record=record,
        bundle=bundle,
    )


def test_parent_selection_semantically_deduplicates_and_caps_at_eight(tmp_path):
    candidates = []
    for candidate_id in range(1, 11):
        config = {"same": 1} if candidate_id in (1, 2) else {"unique": candidate_id}
        candidates.append(
            _authenticated_candidate(
                tmp_path, candidate_id, config, _summary(candidate_id)
            )
        )
    dummy = tmp_path / "dummy"
    dummy.write_text("x", encoding="utf-8")
    source = AuthenticatedFormalV2RescueSource(
        root=tmp_path,
        manifest_path=dummy,
        ledger_path=dummy,
        candidate_report_path=dummy,
        refinement_report_path=dummy,
        campaign_result_path=dummy,
        manifest_sha256="a" * 64,
        ledger_sha256="b" * 64,
        candidate_report_sha256="c" * 64,
        refinement_report_sha256="d" * 64,
        campaign_result_sha256="e" * 64,
        source_authentication_id="f" * 64,
        records=tuple(candidates),
    )
    parents = select_refinement_rescue_parents(source)
    assert len(parents) == 8
    assert [parent.candidate_id for parent in parents[:2]] == [1, 3]
    assert len({parent.config_semantic_sha256 for parent in parents}) == 8


def test_budget_is_exact_and_balanced():
    assert allocate_refinement_rescue_budget(8) == (128,) * 8
    assert allocate_refinement_rescue_budget(7) == (147, 147, 146, 146, 146, 146, 146)
    with pytest.raises(ValueError, match="exactly 1024"):
        RefinementRescueBudget(total_candidates=40)


def test_job_specs_cover_blocks_and_radii_and_preserve_zero_ki():
    config = load_config(V14_CONFIG)
    parent = RefinementRescueParent(
        rank=0,
        candidate_id=42,
        config_semantic_sha256=canonical_sha256(config),
        source_stage="joint_refinement",
        source_artifact_directory="joint_refinement/candidates/candidate_42",
        source_authentication_id="a" * 64,
        config=config,
        record=_summary(42),
    )
    jobs = build_refinement_rescue_job_specs(
        (parent,), validate_configs=False
    )
    assert len(jobs) == 1024
    assert len({job["candidate_id"] for job in jobs}) == 1024
    exact = [job for job in jobs if job["job_metadata"]["exact_parent"]]
    assert len(exact) == 1
    assert exact[0]["job_metadata"]["local_index"] == 0
    covered = {
        (
            job["job_metadata"]["parameter_block"],
            job["job_metadata"]["radius_fraction"],
        )
        for job in jobs
        if not job["job_metadata"]["exact_parent"]
    }
    assert covered == {
        (block, radius) for block in PARAMETER_BLOCKS for radius in RADIUS_FRACTIONS
    }
    base_ki = config["contact_feedback"]["ki_rad_per_n_s"]
    for job in jobs:
        candidate_ki = job["config"]["contact_feedback"]["ki_rad_per_n_s"]
        for finger, value in base_ki.items():
            if value == 0.0:
                assert candidate_ki[finger] == 0.0
        metadata = job["job_metadata"]
        assert metadata["rescue_id"]
        assert metadata["source_authentication_id"] == "a" * 64
        assert metadata["full_reset_required"] is True


def test_missing_complete_metric_ranks_fail_closed():
    complete = _summary(1)
    missing = copy.deepcopy(complete)
    missing["candidate_id"] = 2
    del missing["summary"]["metrics"]["motion_smoothness"][
        "operation_peak_abs_filtered_jerk_m_s3"
    ]
    ranked = rank_refinement_rescue_results([missing, complete])
    assert ranked[0]["candidate_id"] == 1
    assert refinement_rescue_rank_evidence(missing)["overall_metrics_complete"] is False
