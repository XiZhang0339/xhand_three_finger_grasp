from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, validate_config
from xhand_grasp.tuning.contact_preserving_candidate_artifacts import (
    run_or_resume_v14_candidate_artifacts,
)
from xhand_grasp.tuning.upright_grasp_self_collision_rescue import (
    ActiveFingerCollisionContact,
    ActiveFingerSelfCollisionAuditedSession,
    ActiveFingerSelfCollisionStep,
    INDEX_BEND_ACTUATOR,
    SELF_COLLISION_CHECK,
    UprightGraspRescueBudget,
    authenticate_upright_grasp_rescue_source,
    build_upright_grasp_rescue_jobs,
    rank_upright_grasp_rescue_records,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _summary(*, full: bool = False) -> dict[str, Any]:
    return {
        "passed": full,
        "failed_checks": [] if full else ["smooth_motion_jerk_within_limit"],
        "checks": {
            "smooth_motion_jerk_within_limit": full,
        },
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": full,
            "full_success": full,
        },
        "metrics": {
            "operation_median_lift_m": 0.011,
            "motion_smoothness": {
                "operation_peak_abs_filtered_jerk_m_s3": 3.0
            },
            "contact_preserving_planned_lift": {
                "simultaneous_target_face_effective_duty": 1.0
            },
            "closure_alignment": {
                "per_finger": {
                    "index": {"angle_p95_deg": 8.0},
                    "mid": {"angle_p95_deg": 9.0},
                }
            },
        },
    }


class _ArtifactSession:
    def __init__(self) -> None:
        self.step = 0

    @property
    def complete(self) -> bool:
        return self.step >= 2

    def advance_one(self) -> None:
        self.step += 1

    def finalize(self, *, trace_path=None) -> Mapping[str, Any]:
        if trace_path is not None:
            np.savez_compressed(
                trace_path,
                time=np.asarray((0.001, 0.002)),
                control_state=np.asarray(("VERIFY", "HOLD")),
            )
        return copy.deepcopy(_summary())

    def close(self) -> None:
        return None


def _source(tmp_path: Path):
    grasp = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan = copy.deepcopy(grasp)
    grasp["hand_pose"]["translation_m"][2] += 0.0005
    grasp["grasp_pose"]["nominal_joint_qpos_rad"][INDEX_BEND_ACTUATOR] = -0.02
    grasp["control"]["contact_preload_targets_rad"][INDEX_BEND_ACTUATOR] = -0.02
    grasp["control_protocol"]["close_s"] = 1.0
    plan["control_protocol"]["close_s"] = 1.5

    grasp_root = tmp_path / "grasp"
    plan_root = tmp_path / "plan"
    run_or_resume_v14_candidate_artifacts(
        grasp,
        grasp_root,
        14001,
        final_rerun=True,
        session_factory=lambda _config: _ArtifactSession(),
        validator=None,
    )
    run_or_resume_v14_candidate_artifacts(
        plan,
        plan_root,
        14002,
        final_rerun=True,
        session_factory=lambda _config: _ArtifactSession(),
        validator=None,
    )
    return authenticate_upright_grasp_rescue_source(grasp_root, plan_root)


def test_authenticated_pairing_and_candidate_generation_are_deterministic(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    budget = UprightGraspRescueBudget(candidate_count=8)
    first = build_upright_grasp_rescue_jobs(source, budget=budget)
    second = build_upright_grasp_rescue_jobs(source, budget=budget)
    assert [value["candidate_id"] for value in first] == [
        value["candidate_id"] for value in second
    ]
    assert [value["candidate_sha256"] for value in first] == [
        value["candidate_sha256"] for value in second
    ]

    exact = first[0]
    config = exact["config"]
    assert config["cube"] == source.grasp_config["cube"]
    assert config["hand_pose"] == source.grasp_config["hand_pose"]
    assert config["grasp_pose"] == source.grasp_config["grasp_pose"]
    assert config["control"]["precontact_targets_rad"] == source.grasp_config[
        "control"
    ]["precontact_targets_rad"]
    assert config["control_protocol"]["close_s"] == 1.0
    assert config["contact_feedback"] == source.plan_config["contact_feedback"]
    assert set(config["control"]["manipulation_delta_rad"]) == set(
        ACTIVE_ACTUATORS
    )
    assert exact["job_metadata"]["index_bend_plan_scale"] == pytest.approx(0.25)
    assert abs(config["control"]["manipulation_delta_rad"][INDEX_BEND_ACTUATOR]) <= (
        budget.index_bend_max_abs_delta_rad + 1e-12
    )
    validate_config(config)


def test_new_planner_lineage_rejects_candidate_metadata_tampering(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    config = build_upright_grasp_rescue_jobs(
        source, budget=UprightGraspRescueBudget(candidate_count=1)
    )[0]["config"]
    changed = copy.deepcopy(config)
    changed["candidate_metadata"][
        "v14_upright_grasp_self_collision_rescue"
    ]["candidate_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="lineage|planner_id"):
        validate_config(changed)

    changed = copy.deepcopy(config)
    changed["control"]["precontact_targets_rad"][
        "left_hand_index_joint1_actuator"
    ] += 1e-4
    with pytest.raises(ValueError, match="resolved payload"):
        validate_config(changed)

    changed = copy.deepcopy(config)
    changed["candidate_metadata"][
        "v14_upright_grasp_self_collision_rescue"
    ]["index_bend_plan_scale"] = 0.9
    with pytest.raises(ValueError, match="bend-scale diagnostic"):
        validate_config(changed)


def test_directed_offsets_and_two_index_plan_scales_are_materialized(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    nominal_offset = [0.0] * len(ACTIVE_ACTUATORS)
    precontact_residual = [0.0] * len(ACTIVE_ACTUATORS)
    nominal_offset[1] = -0.002
    precontact_residual[0] = -0.002
    budget = UprightGraspRescueBudget(
        candidate_count=1,
        grasp_geometry_blend_fraction=0.25,
        use_grasp_close_timing=False,
        index_bend_plan_scale_range=(0.4, 0.4),
        index_joint2_plan_scale=0.9,
        fixed_wrist_local_rotvec_deg=(0.0, 0.0, -0.1),
        fixed_nominal_offset_rad=tuple(nominal_offset),
        fixed_precontact_residual_rad=tuple(precontact_residual),
    )
    job = build_upright_grasp_rescue_jobs(source, budget=budget)[0]
    config = job["config"]
    metadata = job["job_metadata"]
    thumb_bend, thumb_rota1 = ACTIVE_ACTUATORS[:2]
    expected_rota1 = (
        0.75
        * source.plan_config["grasp_pose"]["nominal_joint_qpos_rad"][
            thumb_rota1
        ]
        + 0.25
        * source.grasp_config["grasp_pose"]["nominal_joint_qpos_rad"][
            thumb_rota1
        ]
        - 0.002
    )
    assert config["grasp_pose"]["nominal_joint_qpos_rad"][thumb_rota1] == (
        pytest.approx(expected_rota1)
    )
    assert metadata["precontact_residual_rad"][thumb_bend] == pytest.approx(
        -0.002
    )
    assert metadata["index_bend_plan_scale"] == pytest.approx(0.4)
    assert metadata["budget"]["index_joint2_plan_scale"] == pytest.approx(0.9)
    assert metadata["budget"]["fixed_wrist_local_rotvec_deg"][2] == (
        pytest.approx(-0.1)
    )
    assert -0.1 - 1e-12 <= metadata["wrist_local_rotvec_deg"][2] <= 0.0
    original_joint2 = source.plan_config["manipulation_plan"][
        "actuator_waypoints_rad"
    ]["left_hand_index_joint2_actuator"]
    resolved_joint2 = config["manipulation_plan"]["actuator_waypoints_rad"][
        "left_hand_index_joint2_actuator"
    ]
    assert resolved_joint2 == pytest.approx(
        [0.9 * float(value) for value in original_joint2]
    )
    assert config["control_protocol"]["close_s"] == pytest.approx(
        source.plan_config["control_protocol"]["close_s"]
    )
    validate_config(config)


def test_directed_budget_validation_rejects_bad_vectors_and_scales() -> None:
    with pytest.raises(ValueError, match="fixed_nominal_offset_rad"):
        UprightGraspRescueBudget(fixed_nominal_offset_rad=(0.0,) * 7)
    with pytest.raises(ValueError, match="index_joint2_plan_scale"):
        UprightGraspRescueBudget(index_joint2_plan_scale=0.4)
    with pytest.raises(ValueError, match="wrist rotation norm"):
        UprightGraspRescueBudget(
            fixed_wrist_local_rotvec_deg=(0.0, 0.0, 5.0)
        )


class _AuditedInner:
    def __init__(self, states: tuple[str, ...], *, full: bool = True) -> None:
        self.states = states
        self.step = 0
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.001))
        self.data = object()
        self.traces: dict[str, np.ndarray] = {
            "time": np.arange(len(states), dtype=np.float64) * 0.001
        }
        self.config: dict[str, Any] = {}
        self.summary = _summary(full=full)

    @property
    def complete(self) -> bool:
        return self.step >= len(self.states)

    def advance_one(self):
        state = self.states[self.step]
        result = SimpleNamespace(index=self.step, control_state=state)
        self.step += 1
        return result

    def finalize(self, *, trace_path=None):
        if trace_path is not None:
            np.savez_compressed(trace_path, **self.traces)
        return copy.deepcopy(self.summary)

    def close(self) -> None:
        return None


def _collision() -> ActiveFingerSelfCollisionStep:
    return ActiveFingerSelfCollisionStep(
        (
            ActiveFingerCollisionContact(
                first_finger="index",
                second_finger="mid",
                first_geom="left_hand_index_rota_link2_collision_1",
                second_geom="left_hand_mid_link2_collision_1",
                normal_force_n=2.0,
                penetration_m=0.0001,
            ),
        )
    )


def test_audited_session_makes_operation_self_collision_a_hard_failure(
    tmp_path: Path,
) -> None:
    inner = _AuditedInner(("VERIFY", "MANIPULATE", "MANIPULATE", "HOLD"))
    snapshots = iter(
        (
            ActiveFingerSelfCollisionStep(),
            ActiveFingerSelfCollisionStep(),
            _collision(),
            _collision(),
        )
    )
    session = ActiveFingerSelfCollisionAuditedSession(
        inner, detector=lambda _session: next(snapshots)
    )
    while not session.complete:
        session.advance_one()
    trace_path = tmp_path / "trace.npz"
    summary = session.finalize(trace_path=trace_path)
    assert summary["passed"] is False
    assert summary["checks"][SELF_COLLISION_CHECK] is False
    assert SELF_COLLISION_CHECK in summary["failed_checks"]
    assert summary["stage_status"]["grasp_success"] is True
    assert summary["stage_status"]["manipulation_success"] is False
    metric = summary["metrics"]["active_finger_self_collision"]
    assert metric["collision_frame_count"] == 2
    assert metric["first_collision_step"] == 2
    assert metric["longest_consecutive_collision_s"] == pytest.approx(0.002)
    assert metric["maximum_total_normal_force_n"] == pytest.approx(2.0)
    with np.load(trace_path, allow_pickle=False) as trace:
        assert trace["active_finger_self_collision"].tolist() == [
            False,
            False,
            True,
            True,
        ]
        assert "left_hand_index_rota_link2_collision_1" in str(
            trace["active_finger_self_collision_pairs"][2]
        )


def test_audited_session_preserves_collision_free_success() -> None:
    inner = _AuditedInner(("VERIFY", "MANIPULATE", "HOLD"))
    session = ActiveFingerSelfCollisionAuditedSession(
        inner,
        detector=lambda _session: ActiveFingerSelfCollisionStep(),
    )
    while not session.complete:
        session.advance_one()
    summary = session.finalize()
    assert summary["passed"] is True
    assert summary["checks"][SELF_COLLISION_CHECK] is True
    assert summary["stage_status"]["full_success"] is True


def test_audited_session_persists_and_checks_joint_pair_alignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import xhand_grasp.tuning.upright_grasp_self_collision_rescue as rescue

    inner = _AuditedInner(("VERIFY", "VERIFY", "MANIPULATE", "HOLD"))
    inner.config = {
        "candidate_metadata": {
            "v14_index_middle_joint_pair_alignment_refinement": {
                "joint_names": [
                    "left_hand_index_joint1",
                    "left_hand_mid_joint1",
                ],
                "source_grasp_window_p95_deg": 10.0,
                "minimum_improvement_deg": 2.5,
                "grasp_window_p95_max_deg": 7.5,
                "operation_p95_max_deg": 7.5,
            }
        }
    }
    inner.traces.update(
        {
            "grasp_stable_window_start_step": np.asarray(0),
            "grasp_stable_window_end_step": np.asarray(1),
            "grasp_lock_step": np.asarray(1),
        }
    )
    values = iter((7.2, 7.1, 7.0, 6.9))
    monkeypatch.setattr(rescue, "resolve_joint_pair", lambda *_args: object())
    monkeypatch.setattr(
        rescue,
        "joint_pair_telemetry",
        lambda *_args: {
            "vector_cube_m": np.asarray((0.001, 0.02, 0.001)),
            "angle_to_cube_y_deg": next(values),
        },
    )
    session = ActiveFingerSelfCollisionAuditedSession(
        inner,
        detector=lambda _session: ActiveFingerSelfCollisionStep(),
    )
    while not session.complete:
        session.advance_one()
    trace_path = tmp_path / "alignment.npz"
    summary = session.finalize(trace_path=trace_path)

    metric = summary["metrics"]["index_middle_joint_pair_alignment"]
    assert metric["aligned"] is True
    assert metric["grasp_window_sample_count"] == 2
    assert metric["grasp_window_angle_p95_deg"] == pytest.approx(7.195)
    assert summary["checks"][
        "index_middle_joint1_line_aligned_with_cube_y"
    ] is True
    with np.load(trace_path, allow_pickle=False) as trace:
        assert trace["index_middle_joint1_line_cube_m"].shape == (4, 3)
        assert trace[
            "index_middle_joint1_line_angle_to_cube_y_deg"
        ].tolist() == pytest.approx([7.2, 7.1, 7.0, 6.9])


def test_ranking_prefers_collision_free_contact_over_larger_colliding_lift() -> None:
    clean = {
        "candidate_id": 2,
        "grasp_success": True,
        "full_success": False,
        "summary": _summary(full=False),
    }
    clean["summary"]["checks"][SELF_COLLISION_CHECK] = True
    clean["summary"]["metrics"]["operation_median_lift_m"] = 0.009
    colliding = copy.deepcopy(clean)
    colliding["candidate_id"] = 1
    colliding["summary"]["checks"][SELF_COLLISION_CHECK] = False
    colliding["summary"]["metrics"]["operation_median_lift_m"] = 0.012
    ranked = rank_upright_grasp_rescue_records((colliding, clean))
    assert ranked[0]["candidate_id"] == 2
