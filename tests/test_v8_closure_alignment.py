from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from xhand_grasp.closure_alignment import (
    aggregate_force_weighted_closure_contacts,
    closure_alignment_from_velocity,
    closure_direction_within_limits,
    point_jacobian_command_velocity,
    point_jacobian_velocity,
)


def test_single_contact_alignment_uses_negative_cube_outward_normal():
    sample = closure_alignment_from_velocity(
        [-0.03, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        normal_force_n=0.2,
        minimum_normal_force_n=0.05,
    )

    assert sample.valid
    assert sample.cosine == pytest.approx(1.0)
    assert sample.angle_deg == pytest.approx(0.0)
    assert sample.inward_speed_m_s == pytest.approx(0.03)
    assert sample.tangent_speed_m_s == pytest.approx(0.0)
    np.testing.assert_array_equal(
        sample.cube_outward_normal_world,
        np.array([1.0, 0.0, 0.0]),
    )


@pytest.mark.parametrize(
    ("velocity", "expected_cosine", "expected_angle", "expected_inward"),
    [
        ([0.03, 0.0, 0.0], -1.0, 180.0, -0.03),
        ([0.0, 0.03, 0.0], 0.0, 90.0, 0.0),
    ],
)
def test_outward_and_tangential_commands_remain_measurable_for_hard_checks(
    velocity, expected_cosine, expected_angle, expected_inward
):
    sample = closure_alignment_from_velocity(
        velocity,
        [1.0, 0.0, 0.0],
    )

    # ``valid`` is a measurement mask, not an acceptance verdict.  Preserving
    # these samples lets the evaluator fail them via inward_speed <= 0 instead
    # of accidentally dropping them from the p95/minimum calculation.
    assert sample.valid
    assert sample.cosine == pytest.approx(expected_cosine, abs=1e-12)
    assert sample.angle_deg == pytest.approx(expected_angle, abs=1e-12)
    assert sample.inward_speed_m_s == pytest.approx(expected_inward, abs=1e-12)
    assert np.isfinite(
        [
            sample.cosine,
            sample.angle_deg,
            sample.inward_speed_m_s,
            sample.tangent_speed_m_s,
        ]
    ).all()
    assert not closure_direction_within_limits(
        sample,
        maximum_angle_deg=30.0,
        minimum_inward_speed_m_s=0.0,
        require_positive_inward_speed=True,
    )


def test_zero_speed_and_insufficient_force_return_conservative_invalid_samples():
    zero_speed = closure_alignment_from_velocity(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    )
    low_force = closure_alignment_from_velocity(
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        normal_force_n=0.049,
        minimum_normal_force_n=0.05,
    )

    for sample in (zero_speed, low_force):
        assert not sample.valid
        assert sample.cosine == -1.0
        assert sample.angle_deg == 180.0


def test_configured_minimum_inward_speed_is_separate_from_validity():
    sample = closure_alignment_from_velocity(
        [-0.005, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        normal_force_n=0.1,
        minimum_normal_force_n=0.05,
    )

    assert sample.valid
    assert closure_direction_within_limits(
        sample,
        maximum_angle_deg=30.0,
        minimum_inward_speed_m_s=0.005,
    )
    assert not closure_direction_within_limits(
        sample,
        maximum_angle_deg=30.0,
        minimum_inward_speed_m_s=0.006,
    )


def test_multiple_contacts_are_normal_force_weighted_before_alignment():
    # F-weighted velocity = (-1, 0.75, 0), so cosine is 0.8 and the
    # tangential-to-inward speed ratio is 0.75.
    sample = aggregate_force_weighted_closure_contacts(
        [1.0, 3.0, 0.0],
        [
            [-1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [100.0, 100.0, 100.0],
        ],
        [
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        minimum_total_force_n=0.05,
    )

    assert sample.valid
    assert sample.total_normal_force_n == pytest.approx(4.0)
    np.testing.assert_allclose(
        sample.command_velocity_world_m_s,
        [-1.0, 0.75, 0.0],
    )
    np.testing.assert_allclose(sample.cube_outward_normal_world, [1.0, 0.0, 0.0])
    assert sample.cosine == pytest.approx(0.8)
    assert sample.angle_deg == pytest.approx(math.degrees(math.acos(0.8)))
    assert sample.inward_speed_m_s == pytest.approx(1.0)
    assert sample.tangent_speed_m_s == pytest.approx(0.75)


def test_empty_contact_collection_has_fixed_shape_finite_invalid_evidence():
    sample = aggregate_force_weighted_closure_contacts(
        np.empty(0),
        np.empty((0, 3)),
        np.empty((0, 3)),
        minimum_total_force_n=0.05,
    )

    assert not sample.valid
    assert sample.total_normal_force_n == 0.0
    np.testing.assert_array_equal(sample.command_velocity_world_m_s, np.zeros(3))
    np.testing.assert_array_equal(sample.cube_outward_normal_world, np.zeros(3))


def _hinge_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="finger">
              <joint name="finger_joint" type="hinge" axis="0 0 1"/>
              <geom name="finger_geom" type="sphere" size="0.01" pos="1 0 0"/>
            </body>
          </worldbody>
          <actuator>
            <position name="finger_actuator" joint="finger_joint" kp="1"/>
          </actuator>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_point_jacobian_command_velocity_matches_finite_difference():
    model, data = _hinge_model()
    geom_id = model.geom("finger_geom").id
    body_id = model.body("finger").id
    joint_id = model.joint("finger_joint").id
    dof_address = int(model.jnt_dofadr[joint_id])
    point = data.geom_xpos[geom_id].copy()
    target_speed = 2.0

    velocity = point_jacobian_command_velocity(
        model,
        data,
        point,
        body_id,
        [dof_address],
        [target_speed],
    )

    epsilon = 1e-7
    initial = point.copy()
    data.qpos[model.jnt_qposadr[joint_id]] += target_speed * epsilon
    mujoco.mj_forward(model, data)
    finite_difference = (data.geom_xpos[geom_id] - initial) / epsilon
    np.testing.assert_allclose(velocity, finite_difference, atol=3e-7)
    np.testing.assert_allclose(velocity, [0.0, 2.0, 0.0], atol=1e-12)


def test_generalized_and_actuator_address_jacobian_adapters_agree():
    model, data = _hinge_model()
    geom_id = model.geom("finger_geom").id
    body_id = model.body("finger").id
    dof_address = int(model.jnt_dofadr[model.joint("finger_joint").id])
    point = data.geom_xpos[geom_id]
    qvel = np.zeros(model.nv)
    qvel[dof_address] = -0.4

    direct = point_jacobian_velocity(model, data, point, body_id, qvel)
    mapped = point_jacobian_command_velocity(
        model,
        data,
        point,
        body_id,
        [dof_address],
        [-0.4],
    )

    np.testing.assert_array_equal(mapped, direct)


def test_geom_order_canonicalization_is_an_explicit_caller_responsibility():
    # A caller flips MuJoCo's geom1->geom2 normal when the cube is geom2.  Once
    # canonicalized, this module gives identical results for both geom orders.
    contact_normal_geom1_to_geom2 = np.array([1.0, 0.0, 0.0])
    cube_geom1_outward = contact_normal_geom1_to_geom2
    cube_geom2_outward = -(-contact_normal_geom1_to_geom2)

    first = closure_alignment_from_velocity(
        [-0.1, 0.0, 0.0], cube_geom1_outward
    )
    second = closure_alignment_from_velocity(
        [-0.1, 0.0, 0.0], cube_geom2_outward
    )

    np.testing.assert_array_equal(
        first.command_velocity_world_m_s,
        second.command_velocity_world_m_s,
    )
    np.testing.assert_array_equal(
        first.cube_outward_normal_world,
        second.cube_outward_normal_world,
    )
    assert first.cosine == second.cosine
    assert first.angle_deg == second.angle_deg
    assert first.inward_speed_m_s == second.inward_speed_m_s
    assert first.tangent_speed_m_s == second.tangent_speed_m_s
    assert first.valid == second.valid
