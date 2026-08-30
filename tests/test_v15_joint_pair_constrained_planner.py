from __future__ import annotations

import copy
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from xhand_grasp.checkpoint import PhysicsCheckpoint, capture_physics_checkpoint
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.joint_pair_geometry import measure_oriented_joint_pair_geometry
from xhand_grasp.tuning.actual_contact_manipulation import GraspPhysicsCheckpoint
from xhand_grasp.tuning.contact_constrained_planner import (
    ContactConstrainedPlannerSettings,
    ExtendedProbeSample,
)
from xhand_grasp.tuning.joint_pair_constrained_planner import (
    JointPairPlannerSettings,
    JointPairProbeSample,
    JointPairSegmentRollout,
    JointPairSequentialPlanningHooks,
    JointPairStateSample,
    audit_joint_pair_interpolation,
    build_joint_pair_octagonal_band,
    fit_joint_pair_probe_response,
    materialize_joint_pair_constrained_plan_config,
    plan_joint_pair_constrained_sequential_trajectory,
)
from xhand_grasp.tuning.sequential_contact_planner import SequentialSegmentRollout


ROOT = Path(__file__).resolve().parents[1]
V14_TEMPLATE = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _pair_vector(residual: np.ndarray, length_m: float = 0.02) -> np.ndarray:
    direction = np.asarray((residual[0], 1.0, residual[1]), dtype=np.float64)
    return length_m * direction / np.linalg.norm(direction)


def _probe_samples(
    *,
    step_index: int = 250,
    pair_bias: np.ndarray | None = None,
    pair_jacobian: np.ndarray | None = None,
) -> tuple[JointPairProbeSample, ...]:
    bias = np.asarray((0.001, -0.002) if pair_bias is None else pair_bias)
    jacobian = np.zeros((2, 8)) if pair_jacobian is None else pair_jacobian
    object_jacobian = np.zeros((12, 8), dtype=np.float64)
    object_jacobian[2, 0] = 0.1
    response_bias = np.concatenate(
        (np.zeros(6), np.full(3, 0.2), np.full(3, 0.0001))
    )

    def make(
        kind: str, actuator: str | None, direction: int, delta: np.ndarray
    ) -> JointPairProbeSample:
        response = response_bias + object_jacobian @ delta
        residual = bias + jacobian @ delta
        vector = _pair_vector(residual)
        extended = ExtendedProbeSample(
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
        return JointPairProbeSample(
            extended_sample=extended,
            joint_pair_signed_residual=residual,
            joint_pair_vector_cube_m=vector,
            joint_pair_length_m=0.02,
        )

    zero = np.zeros(8)
    samples = [make("zero", None, 0, zero)]
    for column, actuator in enumerate(ACTIVE_ACTUATORS):
        for direction in (-1, 1):
            delta = zero.copy()
            delta[column] = 0.02 * direction
            samples.append(make("single_actuator", actuator, direction, delta))
    return tuple(samples)


def test_oriented_geometry_is_signed_and_rejects_reverse_or_short_pair() -> None:
    first = np.asarray((0.1, 0.2, 0.3))
    vector = np.asarray((0.001, 0.020, -0.002))
    geometry = measure_oriented_joint_pair_geometry(
        first, first + vector, np.eye(3), minimum_separation_m=0.01
    )

    np.testing.assert_allclose(
        geometry["joint_pair_signed_residual"],
        (vector[0] / vector[1], vector[2] / vector[1]),
    )
    assert geometry["angle_to_cube_positive_y_deg"] == pytest.approx(
        math.degrees(math.atan(math.hypot(vector[0], vector[2]) / vector[1]))
    )
    with pytest.raises(ValueError, match=r"\+Y"):
        measure_oriented_joint_pair_geometry(first + vector, first, np.eye(3))
    with pytest.raises(ValueError, match="minimum separation"):
        measure_oriented_joint_pair_geometry(
            first, first + 0.25 * vector, np.eye(3)
        )


def test_pair_response_fits_exact_two_by_eight_central_difference() -> None:
    expected = np.asarray(
        [
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
            [-0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1],
        ]
    )
    response = fit_joint_pair_probe_response(
        tuple(reversed(_probe_samples(pair_jacobian=expected)))
    )

    assert response.joint_pair_jacobian_2x8.shape == (2, 8)
    np.testing.assert_allclose(response.joint_pair_jacobian_2x8, expected)
    np.testing.assert_allclose(
        response.joint_pair_signed_residual, (0.001, -0.002)
    )
    assert all(item["method"] == "central_contact_safe" for item in response.column_evidence)
    assert len(response.probe_evidence) == 17


def test_octagonal_band_is_inscribed_and_audit_samples_every_millisecond() -> None:
    response = fit_joint_pair_probe_response(
        _probe_samples(pair_bias=np.zeros(2), pair_jacobian=np.zeros((2, 8)))
    )
    band = build_joint_pair_octagonal_band(
        response.joint_pair_signed_residual,
        response.joint_pair_jacobian_2x8,
        0.5,
    )
    radius = math.tan(math.radians(0.5))
    assert band.facet_normals_8x2.shape == (8, 2)
    assert band.linear_matrix_8x8.shape == (8, 8)
    assert band.facet_radius == pytest.approx(radius * math.cos(math.pi / 8))
    assert band.contains_residual((0.99 * band.facet_radius, 0.0))
    assert not band.contains_residual((0.99 * radius, 0.0))

    audit = audit_joint_pair_interpolation(
        np.zeros(8), np.full(8, 0.01), 0.15, response
    )
    assert audit.passed
    assert audit.sample_times_s.shape == (151,)
    assert np.diff(audit.sample_times_s) == pytest.approx(0.001)
    nonuniform = audit_joint_pair_interpolation(
        np.zeros(8), np.full(8, 0.01), 0.12225258227556429, response
    )
    assert nonuniform.sample_times_s[-2] == pytest.approx(0.122)
    assert nonuniform.sample_times_s[-1] == pytest.approx(0.12225258227556429)

    parsed = JointPairPlannerSettings.from_alignment_config(
        {
            "joint_names": [
                "left_hand_index_joint1",
                "left_hand_mid_joint1",
            ],
            "frame": "cube_local",
            "axis": "+Y",
            "require_positive_y": True,
            "minimum_length_m": 0.01,
            "residual": "vx_over_vy_vz_over_vy",
            "operation_p95_max_deg": 0.5,
            "constraint_polygon_sides": 8,
            "audit_timestep_s": 0.001,
        }
    )
    assert parsed.joint_names == (
        "left_hand_index_joint1",
        "left_hand_mid_joint1",
    )
    assert parsed.minimum_separation_m == pytest.approx(0.01)
    assert parsed.maximum_angle_deg == pytest.approx(0.5)


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
    return GraspPhysicsCheckpoint(
        model=model,
        checkpoint=capture_physics_checkpoint(model, data, step_index=250),
        cube_body_id=model.body("cube").id,
        grasp_lock_step=249,
        actual_grasp_qpos_rad=np.zeros(8),
        lock_sample_joint_qpos_rad=np.zeros(8),
        cube_position_world_m=data.xpos[model.body("cube").id].copy(),
        cube_quaternion_wxyz=data.xquat[model.body("cube").id].copy(),
        config=load_config(V14_TEMPLATE),
    )


class _FakePairPhysics:
    def __init__(self) -> None:
        self.probe_calls: list[tuple[int, int]] = []
        self.rollout_calls: list[tuple[int, int]] = []

    def collect(self, grasp: GraspPhysicsCheckpoint, segment: int):
        self.probe_calls.append((grasp.checkpoint.step_index, segment))
        samples = _probe_samples(
            step_index=grasp.checkpoint.step_index,
            pair_bias=np.zeros(2),
            pair_jacobian=np.zeros((2, 8)),
        )
        return samples if segment % 2 == 0 else tuple(reversed(samples))

    def rollout(
        self,
        grasp: GraspPhysicsCheckpoint,
        segment: int,
        _start: np.ndarray,
        _end: np.ndarray,
        duration_s: float,
    ) -> JointPairSegmentRollout:
        self.rollout_calls.append((grasp.checkpoint.step_index, segment))
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
        contact = SequentialSegmentRollout(
            next_grasp=next_grasp,
            segment_object_response_6d=local,
            cumulative_object_response_6d=cumulative,
            target_normal_force_n=np.full(3, 0.2),
            target_contact_valid=np.ones(3, dtype=bool),
            tangent_slip_m=np.full(3, 0.0001),
            forbidden_contact=False,
            active_nondistal_contact=False,
            physics_steps=150,
        )
        samples = tuple(
            JointPairStateSample(
                elapsed_s=elapsed,
                vector_cube_m=np.asarray((0.0, 0.02, 0.0)),
                joint_pair_signed_residual=np.zeros(2),
                length_m=0.02,
                angle_to_cube_positive_y_deg=0.0,
            )
            for elapsed in (0.0, duration_s)
        )
        return JointPairSegmentRollout(contact, samples)

    @property
    def hooks(self) -> JointPairSequentialPlanningHooks:
        return JointPairSequentialPlanningHooks(self.collect, self.rollout)


def test_pair_sequential_planner_binds_twenty_by_seventeen_and_exports_v2() -> None:
    grasp = _fake_grasp()
    physics = _FakePairPhysics()
    settings = ContactConstrainedPlannerSettings(
        target_normal_force_n=(0.2, 0.2, 0.2),
        minimum_normal_force_n=(0.05, 0.05, 0.05),
        maximum_tangent_slip_m=(0.005, 0.005, 0.005),
    )
    report = plan_joint_pair_constrained_sequential_trajectory(
        grasp,
        grasp.config["manipulation_plan"],
        {name: (-0.2, 0.2) for name in ACTIVE_ACTUATORS},
        contact_settings=settings,
        joint_pair_settings=JointPairPlannerSettings(),
        hooks=physics.hooks,
    )

    assert len(report.attempts) == 4
    assert len(physics.probe_calls) == 4 * 20
    assert len(physics.rollout_calls) == 4 * 20
    assert report.as_mapping()["total_probe_count"] == 4 * 20 * 17
    for attempt in report.attempts:
        assert attempt.joint_pair_residual_jacobian_2x8.shape == (21, 2, 8)
        assert attempt.object_response_jacobian_6x8.shape == (21, 6, 8)
        assert attempt.target_force_jacobian_3x8.shape == (21, 3, 8)
        assert all(segment.interpolation_audit.passed for segment in attempt.segments)

    resolved = materialize_joint_pair_constrained_plan_config(
        grasp.config, report, attempt_index=0, validate=False
    )
    plan = resolved["manipulation_plan"]
    assert plan["schema_version"] == 2
    assert np.asarray(plan["joint_pair_residual_jacobian_2x8"]).shape == (21, 2, 8)
    assert np.asarray(plan["object_response_jacobian_6x8"]).shape == (21, 6, 8)
    assert np.asarray(plan["target_force_jacobian_3x8"]).shape == (21, 3, 8)
    assert "joint_pair_constrained_sequential_planning" in resolved["candidate_metadata"]
