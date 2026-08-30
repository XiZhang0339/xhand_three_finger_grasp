from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import load_config
from xhand_grasp.tuning.joint_pair_near_zero_campaign import (
    JointPairNearZeroBudget,
    authenticate_near_zero_source,
)
from xhand_grasp.tuning.joint_pair_near_zero_campaign_runner import (
    V15CampaignBackend,
    build_exact_rerun_jobs,
    build_feedback_jobs,
    build_grasp_jobs,
    build_plan_jobs,
    build_refinement_jobs,
    build_static_jobs,
    materialize_v15_seed_config,
    run_joint_pair_near_zero_campaign,
    select_exact_rerun_parents,
    select_refinement_feedback_parents,
)
from xhand_grasp.tuning.joint_pair_near_zero_candidate_artifacts import (
    authenticate_v15_candidate_artifacts,
    run_or_resume_v15_candidate_artifacts,
)
from xhand_grasp.v15_identity import validate_v15_top_level_identities


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "grasp_configs/left_opposed_face_palm_down_joint_pair_near_zero_contact_preserving_planned_lift.json"


def _record(candidate_id: int, **values: object) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "static_pass": False,
        "grasp_success": False,
        "full_success": False,
        "summary": {"metrics": {}},
        **values,
    }


def test_v15_job_schedule_has_exact_declared_fanout() -> None:
    budget = JointPairNearZeroBudget()
    static = build_static_jobs(budget)
    static_records = [_record(job.candidate_id, static_pass=True) for job in static]
    grasp = build_grasp_jobs(static_records, budget)
    assert len(grasp) == 384
    grasp_records = [_record(job.candidate_id, grasp_success=True) for job in grasp]
    plans = build_plan_jobs(grasp_records, budget)
    assert len(plans) == 64
    feedback = build_feedback_jobs([_record(job.candidate_id) for job in plans], budget)
    assert len(feedback) == 1_024
    refine = build_refinement_jobs(
        [
            _record(
                job.candidate_id,
                parent_candidate_id=job.parent_candidate_id,
            )
            for job in feedback
        ],
        budget,
    )
    assert len(refine) == 1_024
    exact = build_exact_rerun_jobs([_record(job.candidate_id) for job in (*feedback, *refine)], budget)
    assert len(exact) == 16
    assert len({job.candidate_id for job in (*static, *grasp, *plans, *feedback, *refine, *exact)}) == sum(
        len(value) for value in (static, grasp, plans, feedback, refine, exact)
    )


def test_source_migration_installs_nonempty_content_bound_v15_ids() -> None:
    source = authenticate_near_zero_source(ROOT)
    config = materialize_v15_seed_config(load_config(CONFIG), source)
    validate_v15_top_level_identities(config)
    assert config["schema_version"] == 15
    assert config["cube"] == source.config["cube"]
    assert config["candidate_metadata"]["source_id"] == source.source_id
    assert all(config[name] for name in ("object_config_id", "grasp_pose_id", "grasp_object_pair_id", "planner_id", "controller_id"))


class _FakeSession:
    def __init__(self, config: dict[str, object], *, passed: bool = True):
        self._complete = False
        self._passed = passed

    @property
    def complete(self) -> bool:
        return self._complete

    def advance_one(self) -> None:
        self._complete = True

    def finalize(self, *, trace_path: str | Path | None = None) -> dict[str, object]:
        if trace_path is not None:
            np.savez_compressed(trace_path, time_s=np.asarray([0.0]))
        return {
            "passed": self._passed,
            "stage_status": {"grasp_success": self._passed, "full_success": self._passed},
            "metrics": {},
        }

    def close(self) -> None:
        pass


def test_v15_candidate_artifacts_are_atomic_resumable_and_tamper_evident(tmp_path: Path) -> None:
    source = authenticate_near_zero_source(ROOT)
    config = materialize_v15_seed_config(load_config(CONFIG), source)
    destination = tmp_path / "candidate_42"
    first = run_or_resume_v15_candidate_artifacts(
        config, destination, 42, session_factory=lambda value: _FakeSession(value)
    )
    assert first.reused is False and first.trace_path is not None
    second = run_or_resume_v15_candidate_artifacts(
        config, destination, 42, session_factory=lambda value: pytest.fail("reran")
    )
    assert second.reused is True
    payload = json.loads(second.result_path.read_text("utf-8"))
    payload["candidate_id"] = 43
    second.result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError):
        authenticate_v15_candidate_artifacts(destination)


def test_small_injected_campaign_runs_and_resumes_without_mujoco(tmp_path: Path) -> None:
    budget = JointPairNearZeroBudget(
        static_start_count=2,
        static_retain_count=1,
        measured_grasp_retain_count=1,
        sequential_plans_per_grasp=1,
        feedback_refine_plan_count=1,
        feedback_refine_per_plan=1,
        exact_rerun_count=1,
        final_candidate_count=1,
        perturbations_per_final=1,
        robustness_trials=2,
        robustness_required_passes=1,
    )
    calls: list[str] = []

    def execute(stage, jobs, workspace, context):
        calls.append(stage)
        records = []
        for index, job in enumerate(jobs):
            records.append(
                _record(
                    job.candidate_id,
                    static_pass=True,
                    grasp_success=stage != "static_filter",
                    full_success=stage in {"feedback_grid", "feedback_refinement", "exact_rerun", "local_perturbation", "robustness"} and index == 0,
                    parent_candidate_id=job.parent_candidate_id,
                )
            )
        return records

    output = tmp_path / "campaign"
    result = run_joint_pair_near_zero_campaign(
        CONFIG, output, resume=False, target_success_count=1,
        backend=V15CampaignBackend(execute), budget=budget, publish_catalog=False,
    )
    assert result["stage_counts"] == {
        "static_filter": 2, "dynamic_grasp": 6, "sequential_planning": 1,
        "feedback_grid": 16, "feedback_refinement": 1, "exact_rerun": 1,
        "local_perturbation": 1, "robustness": 2,
    }
    assert result["target_reached"] is True
    calls.clear()
    resumed = run_joint_pair_near_zero_campaign(
        CONFIG, output, resume=True, target_success_count=1,
        backend=V15CampaignBackend(execute), budget=budget, publish_catalog=False,
    )
    assert resumed["stage_counts"] == result["stage_counts"]
    assert calls == []


def _feedback_record(
    candidate_id: int,
    plan_id: int,
    *,
    grasp_success: bool = True,
    operation_executed: bool = True,
    operation_sample_count: int | None = None,
    pair_p95_deg: float = 0.25,
    pair_max_deg: float = 0.5,
    simultaneous_duty: float = 0.995,
    minimum_finger_duty: float = 0.995,
    collision_free: bool = True,
    median_lift_m: float = 0.005,
    full_success: bool = False,
) -> dict[str, object]:
    return _record(
        candidate_id,
        parent_candidate_id=plan_id,
        grasp_success=grasp_success,
        full_success=full_success,
        summary={
            "passed": full_success,
            "stage_status": {
                "grasp_success": grasp_success,
                "full_success": full_success,
            },
            "checks": {
                "operation_executed": operation_executed,
                "manipulation_completed": operation_executed,
                "no_active_finger_self_collision": collision_free,
            },
            "metrics": {
                "manipulation_start_step": 2250 if operation_executed else -1,
                "operation_median_lift_m": median_lift_m,
                "joint_pair_alignment": {
                    "operation_p95_deg": pair_p95_deg,
                    "operation_max_deg": pair_max_deg,
                },
                "active_finger_self_collision": {
                    "collision_free": collision_free,
                },
                "contact_preserving_planned_lift": {
                    "operation_sample_count": (
                        operation_sample_count
                        if operation_sample_count is not None
                        else 1000 if operation_executed else 0
                    ),
                    "simultaneous_target_face_effective_duty": simultaneous_duty,
                    "target_face_effective_duty": {
                        "thumb": minimum_finger_duty,
                        "index": minimum_finger_duty,
                        "mid": minimum_finger_duty,
                    },
                },
            },
        },
    )


def test_refinement_selection_uses_one_best_feedback_per_distinct_plan() -> None:
    records: list[dict[str, object]] = []
    for plan_id in range(100, 110):
        eligible = plan_id < 106
        records.extend(
            (
                _feedback_record(
                    plan_id * 10,
                    plan_id,
                    simultaneous_duty=0.991 if eligible else 0.97,
                    median_lift_m=0.004,
                ),
                _feedback_record(
                    plan_id * 10 + 1,
                    plan_id,
                    simultaneous_duty=0.999 if eligible else 0.98,
                    minimum_finger_duty=0.998,
                    median_lift_m=0.008,
                    full_success=eligible and plan_id == 100,
                ),
            )
        )
    # A high-lift collision case is never eligible, even though its ordinary
    # scalar score would otherwise look attractive.
    records.append(
        _feedback_record(
            1009,
            100,
            simultaneous_duty=1.0,
            median_lift_m=0.02,
            collision_free=False,
        )
    )

    selected = select_refinement_feedback_parents(records, 8)
    reversed_selected = select_refinement_feedback_parents(
        tuple(reversed(records)), 8
    )
    assert [row["candidate_id"] for row in selected] == [
        row["candidate_id"] for row in reversed_selected
    ]
    plan_ids = [
        row["refinement_parent_selection"]["plan_candidate_id"]
        for row in selected
    ]
    assert len(plan_ids) == len(set(plan_ids)) == 8
    assert selected[0]["candidate_id"] == 1001
    modes = [
        row["refinement_parent_selection"]["mode"] for row in selected
    ]
    assert modes.count("eligible") == 6
    assert modes.count("deterministic_fallback") == 2
    for row in selected[6:]:
        assert (
            "simultaneous_target_face_contact_duty_at_least_0p99"
            in row["refinement_parent_selection"]["failed_requirements"]
        )

    budget = JointPairNearZeroBudget(
        feedback_refine_plan_count=8,
        feedback_refine_per_plan=3,
    )
    jobs = build_refinement_jobs(records, budget)
    assert len(jobs) == 8 * 3
    assert len({job.payload["plan_candidate_id"] for job in jobs}) == 8
    for plan_id in {job.payload["plan_candidate_id"] for job in jobs}:
        assert sum(job.payload["plan_candidate_id"] == plan_id for job in jobs) == 3


def test_refinement_selection_fails_closed_and_labels_deterministic_fallback() -> None:
    records = [
        _feedback_record(
            200 + index,
            300 + index,
            operation_executed=False,
            simultaneous_duty=1.0,
        )
        for index in range(3)
    ]
    selected = select_refinement_feedback_parents(records, 2)
    assert [row["candidate_id"] for row in selected] == [200, 201]
    for row in selected:
        evidence = row["refinement_parent_selection"]
        assert evidence["mode"] == "deterministic_fallback"
        assert evidence["eligible"] is False
        assert "operation_executed" in evidence["failed_requirements"]


def test_aborted_real_operation_is_eligible_but_unstarted_or_short_is_not() -> None:
    aborted = _feedback_record(
        501,
        601,
        operation_executed=True,
        operation_sample_count=2667,
        median_lift_m=0.003881,
    )
    # The legacy completion-oriented checks are false, while raw-trace-derived
    # evidence proves that manipulation really ran before the safety abort.
    aborted["summary"]["checks"]["operation_executed"] = False
    aborted["summary"]["checks"]["manipulation_completed"] = False
    unstarted = _feedback_record(
        502,
        602,
        operation_executed=False,
        operation_sample_count=2667,
        median_lift_m=0.02,
    )
    short = _feedback_record(
        503,
        603,
        operation_executed=True,
        operation_sample_count=249,
        median_lift_m=0.02,
    )

    selected = select_refinement_feedback_parents(
        (short, unstarted, aborted), 3
    )
    evidence = {
        row["candidate_id"]: row["refinement_parent_selection"]
        for row in selected
    }
    assert evidence[501]["mode"] == "eligible"
    assert evidence[501]["manipulation_start_step"] == 2250
    assert evidence[501]["operation_sample_count"] == 2667
    assert evidence[501]["reported_operation_executed_check"] is False
    assert evidence[501]["manipulation_completed"] is False
    assert evidence[502]["mode"] == "deterministic_fallback"
    assert evidence[503]["mode"] == "deterministic_fallback"
    assert "operation_executed" in evidence[502]["failed_requirements"]
    assert "operation_executed" in evidence[503]["failed_requirements"]


def test_refinement_selection_prioritizes_real_lift_before_perfect_duty() -> None:
    moving = _feedback_record(
        701,
        801,
        operation_sample_count=2667,
        median_lift_m=0.003881,
        simultaneous_duty=0.9974,
        minimum_finger_duty=0.9985,
        pair_p95_deg=0.28,
        pair_max_deg=0.34,
    )
    nearly_stationary_same_plan = _feedback_record(
        702,
        801,
        operation_sample_count=3000,
        median_lift_m=0.00006,
        simultaneous_duty=1.0,
        minimum_finger_duty=1.0,
        pair_p95_deg=0.05,
        pair_max_deg=0.08,
    )
    nearly_stationary_other_plan = _feedback_record(
        703,
        802,
        operation_sample_count=3000,
        median_lift_m=0.00003,
        simultaneous_duty=1.0,
        minimum_finger_duty=1.0,
        pair_p95_deg=0.03,
        pair_max_deg=0.06,
    )
    second_moving_plan = _feedback_record(
        704,
        803,
        operation_sample_count=2500,
        median_lift_m=0.002,
        simultaneous_duty=0.996,
    )

    selected = select_refinement_feedback_parents(
        (
            nearly_stationary_other_plan,
            nearly_stationary_same_plan,
            second_moving_plan,
            moving,
        ),
        3,
    )
    # Per-plan selection prefers 3.881 mm over the 0.06 mm perfect-duty
    # controller; global selection likewise orders lift before duty/angle.
    assert [row["candidate_id"] for row in selected] == [701, 704, 703]


def test_exact_selection_keeps_high_lift_refinement_and_plan_diversity() -> None:
    full = _feedback_record(
        900,
        1900,
        full_success=True,
        median_lift_m=0.0105,
    )
    full["plan_candidate_id"] = 1900
    high_lift_refinement = _feedback_record(
        901,
        1901,
        median_lift_m=0.003881,
        simultaneous_duty=0.9974,
        minimum_finger_duty=0.9985,
        pair_p95_deg=0.28,
        pair_max_deg=0.34,
    )
    high_lift_refinement.update(
        {"stage": "feedback_refinement", "plan_candidate_id": 1901}
    )
    same_plan_duplicate = _feedback_record(
        902,
        1901,
        median_lift_m=0.0035,
        simultaneous_duty=1.0,
    )
    same_plan_duplicate["plan_candidate_id"] = 1901
    stationary = []
    for index in range(20):
        record = _feedback_record(
            1000 + index,
            2000 + index,
            median_lift_m=0.00006 - index * 1e-6,
            simultaneous_duty=1.0,
            minimum_finger_duty=1.0,
            pair_p95_deg=0.05,
            pair_max_deg=0.08,
        )
        record["plan_candidate_id"] = 2000 + index
        stationary.append(record)

    records = (same_plan_duplicate, *reversed(stationary), high_lift_refinement, full)
    selected = select_exact_rerun_parents(records, 16)
    assert selected[0]["candidate_id"] == 900
    assert selected[1]["candidate_id"] == 901
    assert 902 not in {record["candidate_id"] for record in selected}
    lineages = [
        record["exact_parent_selection"]["plan_lineage_candidate_id"]
        for record in selected
    ]
    assert len(lineages) == len(set(lineages)) == 16

    jobs = build_exact_rerun_jobs(records, JointPairNearZeroBudget())
    assert len(jobs) == 16
    assert jobs[0].parent_candidate_id == 900
    assert jobs[1].parent_candidate_id == 901
    assert jobs[1].payload["exact_parent_selection"]["mode"] == (
        "eligible_operation"
    )
    assert jobs[1].payload["exact_parent_selection"]["median_lift_m"] == pytest.approx(
        0.003881
    )
