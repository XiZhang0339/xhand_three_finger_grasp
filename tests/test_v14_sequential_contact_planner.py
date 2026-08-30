from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from xhand_grasp.checkpoint import PhysicsCheckpoint, capture_physics_checkpoint
from xhand_grasp.config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, load_config, validate_config
from xhand_grasp.tuning.actual_contact_manipulation import GraspPhysicsCheckpoint
from xhand_grasp.tuning.contact_constrained_planner import (
    ContactConstrainedPlannerSettings,
    ExtendedProbeSample,
)
from xhand_grasp.tuning.sequential_contact_planner import (
    SequentialPlanningHooks,
    SequentialSegmentRollout,
    interpolate_segment_command,
    materialize_all_sequential_plan_configs,
    materialize_sequential_plan_config,
    plan_sequential_contact_trajectory,
)


ROOT = Path(__file__).resolve().parents[1]
V14_TEMPLATE = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _fake_grasp() -> GraspPhysicsCheckpoint:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="cube" pos="0 0 .1">
              <freejoint/>
              <geom type="box" size=".03 .03 .03" mass=".16"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    checkpoint = capture_physics_checkpoint(model, data, step_index=250)
    return GraspPhysicsCheckpoint(
        model=model,
        checkpoint=checkpoint,
        cube_body_id=model.body("cube").id,
        grasp_lock_step=249,
        actual_grasp_qpos_rad=np.zeros(8),
        lock_sample_joint_qpos_rad=np.zeros(8),
        cube_position_world_m=data.xpos[model.body("cube").id].copy(),
        cube_quaternion_wxyz=data.xquat[model.body("cube").id].copy(),
        config=load_config(V14_TEMPLATE),
    )


def _samples(step_index: int, segment: int) -> tuple[ExtendedProbeSample, ...]:
    jacobian = np.zeros((12, 8), dtype=np.float64)
    # The changing coefficient proves that every segment is relinearized.
    jacobian[2, 0] = 0.10 + 0.001 * segment
    bias = np.concatenate((np.zeros(6), np.full(3, 0.20), np.full(3, 0.0001)))

    def make(kind: str, actuator: str | None, direction: int, delta: np.ndarray):
        response = bias + jacobian @ delta
        return ExtendedProbeSample(
            kind=kind,
            actuator=actuator,
            direction=direction,
            applied_delta_rad=delta,
            object_response_6d=response[:6],
            target_normal_force_n=response[6:9],
            target_contact_valid=np.ones(3, dtype=bool),
            tangent_slip_m=response[9:12],
            checkpoint_step_index=step_index,
        )

    zero = np.zeros(8)
    result = [make("zero", None, 0, zero)]
    for column, actuator in enumerate(ACTIVE_ACTUATORS):
        for direction in (-1, 1):
            delta = zero.copy()
            delta[column] = 0.02 * direction
            result.append(make("single_actuator", actuator, direction, delta))
    return tuple(result)


class _FakePhysics:
    def __init__(self, *, malformed_segment: int | None = None) -> None:
        self.probe_calls: list[tuple[int, int]] = []
        self.rollout_calls: list[tuple[int, int, float]] = []
        self.malformed_segment = malformed_segment

    def collect(self, grasp: GraspPhysicsCheckpoint, segment: int):
        self.probe_calls.append((grasp.checkpoint.step_index, segment))
        values = _samples(grasp.checkpoint.step_index, segment)
        if segment == self.malformed_segment:
            return values[:-1]
        # Deliberately reverse alternate sets; canonical response/hash ordering
        # must not depend on worker or callback completion order.
        return values if segment % 2 == 0 else tuple(reversed(values))

    def rollout(
        self,
        grasp: GraspPhysicsCheckpoint,
        segment: int,
        start: np.ndarray,
        end: np.ndarray,
        duration_s: float,
    ) -> SequentialSegmentRollout:
        self.rollout_calls.append((grasp.checkpoint.step_index, segment, duration_s))
        source = grasp.checkpoint
        next_checkpoint = PhysicsCheckpoint(
            state=source.state,
            nq=source.nq,
            nv=source.nv,
            na=source.na,
            nu=source.nu,
            state_size=source.state_size,
            step_index=source.step_index + 150,
        )
        next_grasp = GraspPhysicsCheckpoint(
            model=grasp.model,
            checkpoint=next_checkpoint,
            cube_body_id=grasp.cube_body_id,
            grasp_lock_step=grasp.grasp_lock_step,
            actual_grasp_qpos_rad=grasp.actual_grasp_qpos_rad,
            lock_sample_joint_qpos_rad=grasp.lock_sample_joint_qpos_rad,
            cube_position_world_m=grasp.cube_position_world_m,
            cube_quaternion_wxyz=grasp.cube_quaternion_wxyz,
            config=copy.deepcopy(grasp.config),
        )
        cumulative = np.zeros(6)
        cumulative[2] = 0.011 * (segment + 1) / 20.0
        local = np.zeros(6)
        local[2] = 0.011 / 20.0
        return SequentialSegmentRollout(
            next_grasp=next_grasp,
            segment_object_response_6d=local,
            cumulative_object_response_6d=cumulative,
            target_normal_force_n=np.full(3, 0.20),
            target_contact_valid=np.ones(3, dtype=bool),
            tangent_slip_m=np.full(3, 0.0001),
            forbidden_contact=False,
            active_nondistal_contact=False,
            physics_steps=150,
        )

    @property
    def hooks(self) -> SequentialPlanningHooks:
        return SequentialPlanningHooks(self.collect, self.rollout)


def _settings() -> ContactConstrainedPlannerSettings:
    return ContactConstrainedPlannerSettings(
        target_normal_force_n=(0.20, 0.20, 0.20),
        minimum_normal_force_n=(0.05, 0.05, 0.05),
        maximum_tangent_slip_m=(0.005, 0.005, 0.005),
    )


def _bounds() -> dict[str, tuple[float, float]]:
    return {name: (-0.20, 0.20) for name in ACTIVE_ACTUATORS}


def test_sequential_planner_returns_exactly_four_real_rollout_records() -> None:
    grasp = _fake_grasp()
    physics = _FakePhysics()
    report = plan_sequential_contact_trajectory(
        grasp,
        grasp.config["manipulation_plan"],
        _bounds(),
        settings=_settings(),
        hooks=physics.hooks,
    )
    assert len(report.attempts) == 4
    assert len(physics.probe_calls) == 4 * 20
    assert len(physics.rollout_calls) == 4 * 20
    assert report.as_mapping()["total_probe_count"] == 4 * 20 * 17
    for attempt in report.attempts:
        assert len(attempt.segments) == 20
        assert attempt.as_mapping()["probe_count"] == 340
        assert attempt.plan.command_delta_rad.shape == (21, 8)
        assert np.max(np.abs(np.diff(attempt.plan.command_delta_rad, axis=0))) <= 0.04
        assert [segment.start_checkpoint_step_index for segment in attempt.segments] == [
            250 + 150 * index for index in range(20)
        ]
        assert all(len(segment.probe_sha256) == 17 for segment in attempt.segments)
        assert len({segment.response.response_model_id for segment in attempt.segments}) == 20
    payload = report.as_mapping()
    assert payload["search_evidence_only"] is True
    assert payload["final_success_requires_full_reset_rerun"] is True
    assert json.dumps(payload, allow_nan=False, sort_keys=True)


def test_sequential_hashes_and_selection_are_deterministic() -> None:
    grasp = _fake_grasp()
    first = plan_sequential_contact_trajectory(
        grasp,
        grasp.config["manipulation_plan"],
        _bounds(),
        settings=_settings(),
        hooks=_FakePhysics().hooks,
    )
    second = plan_sequential_contact_trajectory(
        grasp,
        grasp.config["manipulation_plan"],
        _bounds(),
        settings=_settings(),
        hooks=_FakePhysics().hooks,
    )
    assert first.as_mapping()["report_id"] == second.as_mapping()["report_id"]
    assert first.selected_attempt_index == second.selected_attempt_index
    assert [value.plan.plan_id for value in first.attempts] == [
        value.plan.plan_id for value in second.attempts
    ]


def test_sequential_planner_rejects_noncanonical_probe_count() -> None:
    grasp = _fake_grasp()
    with pytest.raises(ValueError, match="exactly 17 probes"):
        plan_sequential_contact_trajectory(
            grasp,
            grasp.config["manipulation_plan"],
            _bounds(),
            settings=_settings(),
            hooks=_FakePhysics(malformed_segment=0).hooks,
        )


def test_shared_segment_interpolation_and_materialization() -> None:
    start = np.zeros(8)
    end = np.linspace(-0.02, 0.02, 8)
    np.testing.assert_allclose(interpolate_segment_command(start, end, 0.15, 0.0), start)
    np.testing.assert_allclose(interpolate_segment_command(start, end, 0.15, 0.15), end)
    np.testing.assert_allclose(
        interpolate_segment_command(start, end, 0.15, 0.075),
        0.5 * (start + end),
        atol=1e-12,
    )

    grasp = _fake_grasp()
    report = plan_sequential_contact_trajectory(
        grasp,
        grasp.config["manipulation_plan"],
        _bounds(),
        settings=_settings(),
        hooks=_FakePhysics().hooks,
    )
    resolved = materialize_sequential_plan_config(
        grasp.config, report, attempt_index=0, validate=False
    )
    all_resolved = materialize_all_sequential_plan_configs(
        grasp.config, report, validate=False
    )
    assert len(all_resolved) == 4
    assert [value["manipulation_plan"]["plan_id"] for value in all_resolved] == [
        value.plan.plan_id for value in report.attempts
    ]
    validate_config(resolved)
    metadata = resolved["candidate_metadata"]["sequential_checkpoint_planning"]
    assert metadata["search_evidence_only"] is True
    assert metadata["final_success_requires_full_reset_rerun"] is True
    assert resolved["manipulation_plan"]["plan_id"] == report.attempts[0].plan.plan_id
