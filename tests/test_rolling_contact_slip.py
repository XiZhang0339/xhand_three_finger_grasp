from __future__ import annotations

import mujoco
import numpy as np
import pytest

from xhand_grasp.rolling_contact_slip import (
    ContactPatchKinematics,
    RollingAwareContactSlipEstimator,
    RollingSlipSettings,
    mujoco_contact_patch_kinematics,
    mujoco_contact_tangent_position_jacobian,
)


TARGET_FACES = ("-X", "+X", "+X")


def _sample(
    *,
    finger: int = 0,
    y: float = 0.0,
    z: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    force: float = 1.0,
    identity: str = "cube|thumb_pad",
    source: int = 0,
    normal: tuple[float, float, float] | None = None,
) -> ContactPatchKinematics:
    face_sign = -1.0 if finger == 0 else 1.0
    return ContactPatchKinematics(
        finger_index=finger,
        patch_identity=identity,
        witness_cube_local_m=[face_sign * 0.04, y, z],
        outward_normal_cube_local=(
            (face_sign, 0.0, 0.0) if normal is None else normal
        ),
        relative_velocity_cube_local_m_s=[0.0, vy, vz],
        normal_force_n=force,
        source_contact_index=source,
    )


def test_signed_velocity_integral_and_monotonic_irrecoverable_path_length() -> None:
    estimator = RollingAwareContactSlipEstimator(TARGET_FACES)
    first = estimator.update(0.0, [_sample(vy=0.1, vz=-0.05)])
    assert first.valid.tolist() == [True, False, False]
    assert not first.continuous[0]
    np.testing.assert_array_equal(first.signed_tangent_displacement_m, 0.0)

    second = estimator.update(0.001, [_sample(vy=0.1, vz=-0.05)])
    third = estimator.update(0.002, [_sample(vy=0.1, vz=-0.05)])
    np.testing.assert_allclose(
        second.signed_tangent_displacement_m[0], [0.0001, -0.00005]
    )
    np.testing.assert_allclose(
        third.signed_tangent_displacement_m[0], [0.0002, -0.0001]
    )
    assert third.cumulative_irrecoverable_slip_m[0] == pytest.approx(
        np.hypot(0.1, 0.05) * 0.002
    )

    # Direction reversal cancels signed displacement but cannot undo material
    # slip that already occurred.
    reversed_step = estimator.update(0.003, [_sample(vy=-0.1, vz=0.05)])
    np.testing.assert_allclose(
        reversed_step.signed_tangent_displacement_m[0], [0.0002, -0.0001]
    )
    assert reversed_step.cumulative_irrecoverable_slip_m[0] > (
        third.cumulative_irrecoverable_slip_m[0]
    )


def test_opposing_patch_velocities_do_not_cancel_irrecoverable_slip() -> None:
    estimator = RollingAwareContactSlipEstimator(TARGET_FACES)
    patches = [
        _sample(y=-0.001, vy=0.1, source=0),
        _sample(y=0.001, vy=-0.1, source=1),
    ]
    estimator.update(0.0, patches)
    result = estimator.update(0.001, patches)
    np.testing.assert_allclose(result.relative_tangent_velocity_m_s[0], 0.0)
    np.testing.assert_allclose(result.signed_tangent_displacement_m[0], 0.0)
    assert result.cumulative_irrecoverable_slip_m[0] == pytest.approx(0.0001)


def test_rolling_witness_motion_is_not_misclassified_as_material_slip() -> None:
    settings = RollingSlipSettings(
        maximum_time_gap_s=0.02,
        centroid_jump_threshold_m=0.0005,
        maximum_patch_match_distance_m=0.002,
        rolling_witness_speed_min_m_s=0.01,
        rolling_relative_slip_speed_max_m_s=0.001,
    )
    estimator = RollingAwareContactSlipEstimator(TARGET_FACES, settings=settings)
    estimator.update(0.0, [_sample(y=0.0, vy=0.0)])
    result = estimator.update(0.01, [_sample(y=0.001, vy=0.0)])

    assert result.valid[0] and result.continuous[0]
    assert result.rolling_detected[0]
    assert result.rolling_force_fraction[0] == pytest.approx(1.0)
    assert result.centroid_jump[0]
    assert not result.patch_switch[0]
    np.testing.assert_array_equal(result.signed_tangent_displacement_m[0], 0.0)
    assert result.cumulative_irrecoverable_slip_m[0] == 0.0


def test_manifold_reordering_and_force_centroid_jump_keep_patch_identity() -> None:
    settings = RollingSlipSettings(
        maximum_patch_match_distance_m=0.001,
        centroid_jump_threshold_m=0.001,
    )
    estimator = RollingAwareContactSlipEstimator(TARGET_FACES, settings=settings)
    first = [
        _sample(y=-0.001, vy=0.01, force=9.0, source=10),
        _sample(y=0.001, vy=0.01, force=1.0, source=11),
    ]
    second = [
        # Reverse input/contact indices and invert force dominance.  Nearest
        # one-to-one matching follows witnesses, not list order or centroid.
        _sample(y=0.0011, vy=0.01, force=9.0, source=20),
        _sample(y=-0.0009, vy=0.01, force=1.0, source=21),
    ]
    estimator.update(0.0, first)
    result = estimator.update(0.001, second)

    assert result.matched_patch_count[0] == 2
    assert result.new_patch_count[0] == 0
    assert result.dropped_patch_count[0] == 0
    assert not result.patch_switch[0]
    assert result.centroid_jump[0]
    assert result.signed_tangent_displacement_m[0, 0] == pytest.approx(1e-5)


def test_real_patch_switch_does_not_turn_witness_jump_into_slip() -> None:
    settings = RollingSlipSettings(maximum_patch_match_distance_m=0.001)
    estimator = RollingAwareContactSlipEstimator(TARGET_FACES, settings=settings)
    estimator.update(0.0, [_sample(y=0.0, vy=0.02, identity="old")])
    result = estimator.update(
        0.001, [_sample(y=0.02, vy=0.02, identity="new")]
    )

    assert result.patch_switch[0]
    assert result.patch_switch_count[0] == 1
    assert result.new_patch_count[0] == 1
    assert result.dropped_patch_count[0] == 1
    assert not result.rolling_detected[0]
    # Only integrated material velocity contributes: the 20 mm witness jump
    # is diagnostic and never enters displacement.
    assert result.signed_tangent_displacement_m[0, 0] == pytest.approx(2e-5)
    assert result.cumulative_irrecoverable_slip_m[0] == pytest.approx(2e-5)


def test_contact_loss_reacquisition_and_time_gap_break_continuity() -> None:
    estimator = RollingAwareContactSlipEstimator(TARGET_FACES)
    estimator.update(0.0, [_sample(vy=0.1)])
    lost = estimator.update(0.001, [])
    assert not lost.valid[0]
    assert lost.cumulative_irrecoverable_slip_m[0] == 0.0

    reacquired = estimator.update(0.002, [_sample(vy=0.1)])
    assert reacquired.valid[0]
    assert not reacquired.continuous[0]
    assert reacquired.patch_switch[0]
    assert reacquired.cumulative_irrecoverable_slip_m[0] == 0.0

    resumed = estimator.update(0.003, [_sample(vy=0.1)])
    assert resumed.continuous[0]
    assert resumed.cumulative_irrecoverable_slip_m[0] == pytest.approx(0.0001)

    gap = estimator.update(0.010, [_sample(vy=0.1)])
    assert not gap.continuous[0]
    assert gap.cumulative_irrecoverable_slip_m[0] == pytest.approx(0.0001)


def test_wrong_face_normal_and_insufficient_force_are_invalid() -> None:
    estimator = RollingAwareContactSlipEstimator(TARGET_FACES)
    result = estimator.update(
        0.0,
        [
            _sample(normal=(1.0, 0.0, 0.0)),
            _sample(force=0.001, identity="weak"),
        ],
    )
    # The wrong-normal point is rejected and remaining force is below the
    # default 0.05 N total-contact validity gate.
    assert not result.valid[0]
    assert result.normal_force_n[0] == pytest.approx(0.001)


def test_reset_and_mapping_are_deterministic() -> None:
    estimator = RollingAwareContactSlipEstimator(TARGET_FACES)
    estimator.update(0.0, [_sample(vy=0.1)])
    result = estimator.update(0.001, [_sample(vy=0.1)])
    mapping = result.as_mapping()
    assert mapping["per_finger"]["thumb"]["target_face"] == "-X"
    assert mapping["per_finger"]["thumb"]["signed_tangent_axes_cube"] == [
        "+Y",
        "+Z",
    ]
    assert mapping["per_finger"]["thumb"][
        "signed_tangent_displacement_m"
    ] == pytest.approx([0.0001, 0.0])
    estimator.reset()
    reset = estimator.update(0.0, [_sample(vy=0.1)])
    np.testing.assert_array_equal(reset.signed_tangent_displacement_m, 0.0)


def test_mujoco_adapter_uses_material_point_relative_velocity() -> None:
    xml = """
    <mujoco>
      <option gravity="0 0 0" timestep="0.001"/>
      <worldbody>
        <body name="cube" pos="0 0 0">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size=".05 .05 .05" mass="1"/>
        </body>
        <body name="finger" pos=".058 0 0">
          <freejoint name="finger_joint"/>
          <geom name="finger_geom" type="sphere" size=".01" mass=".1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    finger_joint = model.joint("finger_joint").id
    finger_dof = int(model.jnt_dofadr[finger_joint])
    data.qvel[finger_dof + 1] = 0.2  # world +Y translation
    mujoco.mj_forward(model, data)

    samples = mujoco_contact_patch_kinematics(
        model,
        data,
        cube_geom_id=model.geom("cube_geom").id,
        finger_geom_to_index={model.geom("finger_geom").id: 1},
    )
    assert len(samples) == 1
    sample = samples[0]
    assert sample.finger_index == 1
    assert sample.normal_force_n > 0.0
    np.testing.assert_allclose(sample.witness_cube_local_m, [0.05, 0.0, 0.0])
    np.testing.assert_allclose(sample.outward_normal_cube_local, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        sample.relative_velocity_cube_local_m_s, [0.0, 0.2, 0.0]
    )

    estimator = RollingAwareContactSlipEstimator(TARGET_FACES)
    first = estimator.update(0.0, samples)
    second = estimator.update(0.001, samples)
    assert first.valid[1]
    assert second.signed_tangent_displacement_m[1, 0] == pytest.approx(0.0002)


def test_live_tangent_jacobian_matches_material_relative_velocity() -> None:
    xml = """
    <mujoco>
      <option gravity="0 0 0" timestep="0.001"/>
      <worldbody>
        <body name="cube" pos="0 0 0">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size=".05 .05 .05" mass="1"/>
        </body>
        <body name="finger" pos=".058 0 0">
          <freejoint name="finger_joint"/>
          <geom name="finger_geom" type="sphere" size=".01" mass=".1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    finger_joint = model.joint("finger_joint").id
    finger_y_dof = int(model.jnt_dofadr[finger_joint]) + 1
    mujoco.mj_forward(model, data)
    samples = mujoco_contact_patch_kinematics(
        model,
        data,
        cube_geom_id=model.geom("cube_geom").id,
        finger_geom_to_index={model.geom("finger_geom").id: 1},
    )
    tangent = mujoco_contact_tangent_position_jacobian(
        model,
        data,
        samples,
        cube_geom_id=model.geom("cube_geom").id,
        target_faces=TARGET_FACES,
        active_dof_adrs=[finger_y_dof],
    )
    assert tangent.valid.tolist() == [False, True, False]
    np.testing.assert_allclose(
        tangent.position_jacobian_cube_local_m_per_rad[1, :, 0],
        [1.0, 0.0],
        atol=1e-12,
    )
    data.qvel[finger_y_dof] = 0.2
    predicted = (
        tangent.position_jacobian_cube_local_m_per_rad[1]
        @ data.qvel[[finger_y_dof]]
    )
    np.testing.assert_allclose(predicted, [0.2, 0.0], atol=1e-12)


def test_input_validation_rejects_nonmonotonic_time_and_bad_samples() -> None:
    estimator = RollingAwareContactSlipEstimator(TARGET_FACES)
    estimator.update(0.0, [])
    with pytest.raises(ValueError, match="strictly"):
        estimator.update(0.0, [])
    with pytest.raises(TypeError, match="ContactPatchKinematics"):
        estimator.update(0.001, [object()])  # type: ignore[list-item]
