from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.contact_preserving_adaptive_pose_followup import (
    AdaptivePoseFollowupBudget,
    build_adaptive_pose_combination_jobs,
    build_adaptive_pose_probe_jobs,
    build_adaptive_pose_source,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)
JERK_CHECK = "smooth_motion_jerk_within_limit"


def _record(candidate_id: int, config: dict, *, jerk: float, eligible: bool) -> dict:
    failed = [JERK_CHECK] if eligible else ["median_lift_reached"]
    checks = {
        "grasp_pose_is_stable": True,
        "simultaneous_target_face_contact_duty": True,
        JERK_CHECK: False if eligible else True,
    }
    if not eligible:
        checks["median_lift_reached"] = False
    return {
        "candidate_id": candidate_id,
        "config_semantic_sha256": canonical_sha256(config),
        "result_semantic_sha256": canonical_sha256(
            {"candidate_id": candidate_id, "jerk": jerk, "eligible": eligible}
        ),
        "grasp_success": True,
        "full_success": False,
        "summary": {
            "failed_checks": failed,
            "checks": checks,
            "metrics": {
                "motion_smoothness": {
                    "operation_peak_abs_filtered_jerk_m_s3": jerk
                },
                "contact_preserving_planned_lift": {
                    "simultaneous_target_face_effective_duty": 1.0,
                    "simultaneous_longest_contact_loss_s": 0.0,
                },
            },
        },
        "contact_mode_diagnostics": {
            "measurement_available": True,
            "simultaneous_effective_contact_duty": 1.0,
            "taxel_count_transition_count": {
                "thumb": candidate_id % 3,
                "index": 0,
                "mid": 0,
            },
            "contact_centroid_max_step_m": {"thumb": 0.0002},
        },
    }


def _source_inputs(count: int = 256):
    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    records: list[dict] = []
    configs: dict[int, dict] = {}
    traces: dict[int, str] = {}
    for index in range(count):
        candidate_id = 15_600_000_000_000_000 + index
        config = copy.deepcopy(base)
        # All source configurations remain physical-unique while differences
        # stay inside an allowed hand-root field.
        config["hand_pose"]["translation_m"][0] += index * 1e-9
        configs[candidate_id] = config
        eligible = index < 8
        # Candidate 3 is the strict minimum and therefore the follow-up centre.
        jerk = 2.995 + abs(index - 3) * 0.01
        records.append(_record(candidate_id, config, jerk=jerk, eligible=eligible))
        traces[candidate_id] = canonical_sha256(
            {"candidate_id": candidate_id, "kind": "trace"}
        )
    return records, configs, traces


def _source():
    records, configs, traces = _source_inputs()
    return build_adaptive_pose_source(
        records,
        configs,
        traces,
        search_report_sha256="a" * 64,
    )


def test_source_requires_complete_zero_pass_stage_and_selects_lowest_jerk() -> None:
    records, configs, traces = _source_inputs()
    source = build_adaptive_pose_source(
        records,
        configs,
        traces,
        search_report_sha256="a" * 64,
    )
    assert source.first_stage_candidate_count == 256
    assert len(source.prior_physical_config_sha256) == 256
    assert len(set(source.prior_physical_config_sha256)) == 256
    assert source.centers[0].candidate_id == 15_600_000_000_000_003
    assert source.centers[0].peak_jerk_m_s3 == pytest.approx(2.995)

    incomplete = records[:-1]
    with pytest.raises(RuntimeError, match="complete first-stage"):
        build_adaptive_pose_source(
            incomplete,
            {key: value for key, value in configs.items() if key != records[-1]["candidate_id"]},
            {key: value for key, value in traces.items() if key != records[-1]["candidate_id"]},
            search_report_sha256="a" * 64,
        )

    passed = copy.deepcopy(records)
    passed[0]["full_success"] = True
    with pytest.raises(RuntimeError, match="forbidden after.*hard pass"):
        build_adaptive_pose_source(
            passed, configs, traces, search_report_sha256="a" * 64
        )


def test_source_identity_binds_report_and_every_prior_physical_config() -> None:
    records, configs, traces = _source_inputs()
    first = build_adaptive_pose_source(
        records, configs, traces, search_report_sha256="a" * 64
    )
    second = build_adaptive_pose_source(
        records, configs, traces, search_report_sha256="b" * 64
    )
    assert first.source_authentication_id != second.source_authentication_id

    tampered = copy.deepcopy(configs)
    candidate_id = records[-1]["candidate_id"]
    tampered[candidate_id]["hand_pose"]["translation_m"][1] += 1e-6
    with pytest.raises(RuntimeError, match="semantic hash changed"):
        build_adaptive_pose_source(
            records, tampered, traces, search_report_sha256="a" * 64
        )


def test_budget_caps_the_entire_conditional_followup_at_128() -> None:
    with pytest.raises(ValueError, match="candidate_count"):
        AdaptivePoseFollowupBudget(candidate_count=129)


def test_probe_schedule_prioritizes_index_middle_axes_and_is_physical_unique() -> None:
    source = _source()
    budget = AdaptivePoseFollowupBudget(candidate_count=128)
    first = build_adaptive_pose_probe_jobs(source, budget=budget)
    second = build_adaptive_pose_probe_jobs(source, budget=budget)
    assert len(first) == 64
    assert [value["candidate_id"] for value in first] == [
        value["candidate_id"] for value in second
    ]
    assert [canonical_sha256(value["config"]) for value in first] == [
        canonical_sha256(value["config"]) for value in second
    ]
    prior = set(source.prior_physical_config_sha256)
    physical = [value["physical_config_sha256"] for value in first]
    assert len(set(physical)) == 64
    assert not prior.intersection(physical)

    expected_axes = {*(range(9, 14)), *(range(17, 22)), *(range(25, 30))}
    observed_axes: list[int] = []
    for job in first[:30]:
        normalized = np.asarray(job["job_metadata"]["normalized_sample"])
        assert np.count_nonzero(normalized) == 1
        observed_axes.append(int(np.flatnonzero(normalized)[0]))
    assert set(observed_axes) == expected_axes
    assert all(observed_axes.count(axis) == 2 for axis in expected_axes)

    base = source.centers[0].config
    for job in first:
        config = job["config"]
        metadata = job["job_metadata"]
        assert metadata["adaptive_stage"] == "finite_difference_probe"
        assert metadata["center_candidate_id"] == source.centers[0].candidate_id
        assert metadata["cube_pose_sampled"] is False
        assert metadata["manipulation_plan_changed"] is False
        assert metadata["acceptance_thresholds_changed"] is False
        assert config["cube"] == base["cube"]
        assert config["manipulation_plan"] == base["manipulation_plan"]
        assert config["control"]["manipulation_delta_rad"] == base["control"][
            "manipulation_delta_rad"
        ]


def _probe_results(jobs: tuple[dict, ...], safe_axes: set[int]) -> list[dict]:
    result: list[dict] = []
    for index, job in enumerate(jobs):
        normalized = np.asarray(job["job_metadata"]["normalized_sample"])
        axis = int(np.flatnonzero(normalized)[0])
        record = _record(
            int(job["candidate_id"]),
            job["config"],
            jerk=2.70 + index * 1e-4,
            eligible=axis in safe_axes,
        )
        result.append(record)
    return result


def test_combination_stage_uses_only_strict_safe_directions_and_2_to_4_axes() -> None:
    source = _source()
    budget = AdaptivePoseFollowupBudget(candidate_count=128)
    probes = build_adaptive_pose_probe_jobs(source, budget=budget)
    safe_axes = {9, 10, 17, 18, 25, 26}
    records = _probe_results(probes, safe_axes)
    first = build_adaptive_pose_combination_jobs(
        source, probes, records, budget=budget
    )
    second = build_adaptive_pose_combination_jobs(
        source, probes, records, budget=budget
    )
    assert len(first) == 64
    assert [value["candidate_id"] for value in first] == [
        value["candidate_id"] for value in second
    ]
    prior = set(source.prior_physical_config_sha256)
    prior.update(value["physical_config_sha256"] for value in probes)
    physical = [value["physical_config_sha256"] for value in first]
    assert len(set(physical)) == 64
    assert not prior.intersection(physical)
    evidence = {value["job_metadata"]["probe_evidence_sha256"] for value in first}
    assert len(evidence) == 1
    assert next(iter(evidence)) is not None
    for job in first:
        normalized = np.asarray(job["job_metadata"]["normalized_sample"])
        support = set(int(value) for value in np.flatnonzero(normalized))
        assert 2 <= len(support) <= 4
        assert support <= safe_axes
        assert job["job_metadata"]["adaptive_stage"] == "safe_sparse_combination"


def test_combination_stage_stops_when_probe_already_hard_passes() -> None:
    source = _source()
    budget = AdaptivePoseFollowupBudget(candidate_count=128)
    probes = build_adaptive_pose_probe_jobs(source, budget=budget)
    records = _probe_results(probes, {9, 10})
    records[0]["full_success"] = True
    with pytest.raises(RuntimeError, match="unnecessary after a probe hard pass"):
        build_adaptive_pose_combination_jobs(
            source, probes, records, budget=budget
        )


def test_inactive_finger_controls_remain_exactly_zero() -> None:
    source = _source()
    probes = build_adaptive_pose_probe_jobs(source)
    for job in probes:
        config = job["config"]
        assert set(config["control"]["precontact_targets_rad"]) == set(
            ACTIVE_ACTUATORS
        )
        assert set(config["control"]["contact_preload_targets_rad"]) == set(
            ACTIVE_ACTUATORS
        )
