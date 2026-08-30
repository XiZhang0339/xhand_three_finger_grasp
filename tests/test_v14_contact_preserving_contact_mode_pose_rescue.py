from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.contact_constrained_planner import (
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
)
from xhand_grasp.tuning.contact_preserving_contact_mode_pose_rescue import (
    ContactModePoseRescueBudget,
    ContactModePoseSource,
    build_contact_mode_pose_rescue_jobs,
    contact_mode_pose_candidate_rank,
    contact_mode_trace_diagnostics,
    rank_contact_mode_pose_records,
)
from xhand_grasp.tuning.contact_preserving_time_warp import (
    _time_warp_controller_id,
)


ROOT = Path(__file__).resolve().parents[1]


def _source() -> ContactModePoseSource:
    config = json.loads(
        (
            ROOT
            / "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
        ).read_text(encoding="utf-8")
    )
    config["object_config_id"] = v14_object_config_id(config)
    config["grasp_pose_id"] = v14_grasp_pose_id(config)
    config["grasp_object_pair_id"] = v14_grasp_object_pair_id(config)
    config["planner_id"] = canonical_sha256({"test_planner": 1})
    config["controller_id"] = _time_warp_controller_id(config)
    return ContactModePoseSource(
        root=Path("/immutable/source"),
        candidate_id=15000000000000001,
        config=config,
        result={"grasp_success": True, "full_success": False},
        config_sha256=canonical_sha256(config),
        result_semantic_sha256="1" * 64,
        trace_sha256="2" * 64,
        source_authentication_id="3" * 64,
        trace_diagnostics={
            "taxel_count_transition_count": {"thumb": 2, "index": 0, "mid": 0}
        },
    )


def _trace(length: int = 20) -> dict[str, np.ndarray]:
    state = np.full(length, "HOLD", dtype="<U10")
    state[2:18] = "MANIPULATE"
    taxels = np.ones((length, 3), dtype=np.int64)
    taxels[7:10, 0] = 2
    taxels[12:, 1] = 2
    centroids = np.zeros((length, 3, 3), dtype=np.float64)
    centroids[:, 0, 1] = np.arange(length) * 1e-6
    centroids[8, 0, 1] += 0.0002
    valid = np.ones((length, 3), dtype=bool)
    effective = np.ones((length, 3), dtype=bool)
    jerk = np.zeros(length)
    jerk[9] = -3.0
    return {
        "control_state": state,
        "distal_active_taxel_count": taxels,
        "target_face_contact_centroid_cube_local_m": centroids,
        "target_face_contact_centroid_valid": valid,
        "target_face_effective": effective,
        "operation_vertical_jerk_filtered_m_s3": jerk,
    }


def test_trace_diagnostics_detects_taxel_branch_and_centroid_jump() -> None:
    result = contact_mode_trace_diagnostics(_trace())
    assert result["sample_count"] == 16
    assert result["taxel_count_transition_count"] == {
        "thumb": 2,
        "index": 1,
        "mid": 0,
    }
    assert result["taxel_count_transition_steps"]["thumb"] == [7, 10]
    assert result["contact_centroid_max_step_m"]["thumb"] > 0.0001
    assert result["contact_centroid_large_step_count"]["thumb"] >= 1
    assert result["simultaneous_effective_contact_duty"] == 1.0
    assert result["peak_abs_filtered_jerk_m_s3"] == 3.0
    assert result["peak_abs_filtered_jerk_step"] == 9


def test_budget_rejects_more_than_256_candidates() -> None:
    with pytest.raises(ValueError, match="candidate_count"):
        ContactModePoseRescueBudget(candidate_count=257)


def test_generator_is_deterministic_bounded_and_keeps_cube_and_plan_exact() -> None:
    source = _source()
    budget = ContactModePoseRescueBudget(candidate_count=40)
    first = build_contact_mode_pose_rescue_jobs(source, budget=budget)
    second = build_contact_mode_pose_rescue_jobs(source, budget=budget)
    assert [value["candidate_id"] for value in first] == [
        value["candidate_id"] for value in second
    ]
    assert [canonical_sha256(value["config"]) for value in first] == [
        canonical_sha256(value["config"]) for value in second
    ]
    assert len({value["candidate_id"] for value in first}) == 40
    base = source.config
    for sequence, job in enumerate(first):
        config = job["config"]
        metadata = job["job_metadata"]
        assert job["job_sequence_index"] == sequence
        assert config["cube"] == base["cube"]
        assert config["manipulation_plan"] == base["manipulation_plan"]
        assert (
            config["control"]["manipulation_delta_rad"]
            == base["control"]["manipulation_delta_rad"]
        )
        assert metadata["cube_pose_sampled"] is False
        assert metadata["manipulation_plan_changed"] is False
        root_delta = np.abs(metadata["root_translation_delta_cube_m"])
        assert np.all(root_delta <= np.asarray([0.0002, 0.0002, 0.001]) + 1e-15)
        assert np.linalg.norm(metadata["wrist_local_rotvec_deg"]) <= 0.2 + 1e-12
        assert config["object_config_id"] == base["object_config_id"]
        assert config["grasp_pose_id"] == v14_grasp_pose_id(config)
        assert config["grasp_object_pair_id"] == v14_grasp_object_pair_id(config)
        assert config["controller_id"] == _time_warp_controller_id(config)

    # Local index zero is the exact physical parent.  Only identities and
    # provenance metadata change, none of which enter MuJoCo dynamics.
    exact = first[0]["config"]
    assert exact["hand_pose"] == base["hand_pose"]
    assert exact["grasp_pose"] == base["grasp_pose"]
    assert exact["control"]["precontact_targets_rad"] == base["control"][
        "precontact_targets_rad"
    ]
    assert exact["control"]["contact_preload_targets_rad"] == base["control"][
        "contact_preload_targets_rad"
    ]
    z_offsets = [
        value["job_metadata"]["root_translation_delta_cube_m"][2]
        for value in first
    ]
    for expected in (-0.0004, -0.0006, -0.0008, -0.0010):
        assert any(abs(value - expected) <= 1e-15 for value in z_offsets)


def _record(
    candidate_id: int,
    *,
    full: bool = False,
    grasp: bool = True,
    transitions: int = 0,
    jerk: float = 3.0,
    failed: list[str] | None = None,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "full_success": full,
        "grasp_success": grasp,
        "summary": {
            "failed_checks": (["smooth_motion_jerk_within_limit"] if failed is None else failed),
            "metrics": {
                "motion_smoothness": {
                    "operation_peak_abs_filtered_jerk_m_s3": jerk
                }
            },
        },
        "contact_mode_diagnostics": {
            "taxel_count_transition_count": {
                "thumb": transitions,
                "index": 0,
                "mid": 0,
            },
            "simultaneous_effective_contact_duty": 1.0,
            "contact_centroid_max_step_m": {
                "thumb": 0.0,
                "index": 0.0,
                "mid": 0.0,
            },
            "contact_centroid_large_step_count": {"thumb": 0},
        },
    }


def test_ranking_prefers_success_then_grasp_and_branch_stability_before_jerk() -> None:
    full = _record(9, full=True, jerk=2.4, failed=[])
    stable = _record(8, transitions=0, jerk=3.1)
    unstable_low_jerk = _record(7, transitions=2, jerk=2.6)
    nongrasp = _record(6, grasp=False, transitions=0, jerk=2.0)
    ordered = rank_contact_mode_pose_records(
        [nongrasp, unstable_low_jerk, stable, full]
    )
    assert [value["candidate_id"] for value in ordered] == [9, 8, 7, 6]
    assert contact_mode_pose_candidate_rank(stable) < contact_mode_pose_candidate_rank(
        unstable_low_jerk
    )
