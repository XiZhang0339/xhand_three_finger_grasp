from __future__ import annotations

import numpy as np
import pytest
import mujoco

from xhand_grasp.contact_geometry import (
    active_nondistal_collision_geom_ids,
    distal_collision_geom_ids,
    geom_distance_witness,
    nearest_distal_target_witness,
    nearest_taxel_assignment,
    scan_distal_closure,
)
from xhand_grasp.contacts import Face
from xhand_grasp.evaluation import _v5_fingertip_contact_metrics
from xhand_grasp.scene import ModelInfo
from xhand_grasp.simulation import _allocate_traces, contact_snapshot


FINGERS = ("thumb", "index", "mid")


def _closure_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    bodies = "".join(
        f"""
        <body name="{finger}" pos="0.05 {y} 0">
          <joint name="{finger}_close" type="slide" axis="-1 0 0"
                 range="0 0.03"/>
          <geom name="{finger}_tip" type="sphere" size="0.005"/>
        </body>
        """
        for finger, y in zip(FINGERS, (-0.015, 0.0, 0.015))
    )
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <option gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="cube" type="box" size="0.03 0.03 0.03"/>
            {bodies}
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_real_geom_distance_witness_uses_cube_and_distal_closest_points():
    model, data = _closure_model()
    cube_id = model.geom("cube").id
    thumb_id = model.geom("thumb_tip").id

    witness = geom_distance_witness(
        model,
        data,
        cube_geom_id=cube_id,
        distal_geom_id=thumb_id,
    )

    assert witness is not None
    assert witness.signed_distance_m == pytest.approx(0.015, abs=1e-12)
    np.testing.assert_allclose(witness.cube_point_world_m, [0.03, -0.015, 0.0])
    np.testing.assert_allclose(witness.distal_point_world_m, [0.045, -0.015, 0.0])
    assert witness.face is Face.X_POS
    assert witness.classification.edge_clearance_m == pytest.approx(0.015)

    target = nearest_distal_target_witness(
        model,
        data,
        cube_geom_id=cube_id,
        distal_geom_ids=[thumb_id],
        target_face=Face.X_POS,
    )
    assert target is not None
    assert target.distal_geom_id == witness.distal_geom_id
    assert target.signed_distance_m == witness.signed_distance_m
    np.testing.assert_array_equal(target.cube_point_world_m, witness.cube_point_world_m)
    assert (
        nearest_distal_target_witness(
            model,
            data,
            cube_geom_id=cube_id,
            distal_geom_ids=[thumb_id],
            target_face=Face.X_NEG,
        )
        is None
    )


def test_collision_geom_mapping_and_closure_sweep_restore_state():
    model, data = _closure_model()
    distal_welds = {
        finger: int(model.body_weldid[model.body(finger).id]) for finger in FINGERS
    }
    distal = distal_collision_geom_ids(model, distal_welds)
    parts = {model.body(finger).id: finger for finger in FINGERS}
    assert active_nondistal_collision_geom_ids(model, parts, distal_welds) == {
        finger: () for finger in FINGERS
    }
    qpos_addresses = [model.jnt_qposadr[model.joint(f"{finger}_close").id] for finger in FINGERS]
    data.qpos[qpos_addresses] = [0.001, 0.002, 0.003]
    saved_qpos = data.qpos.copy()
    saved_ctrl = data.ctrl.copy()
    mujoco.mj_forward(model, data)

    sweep = scan_distal_closure(
        model,
        data,
        cube_geom_id=model.geom("cube").id,
        finger_order=FINGERS,
        distal_geom_ids=distal,
        target_faces=(Face.X_POS, Face.X_POS, Face.X_POS),
        actuator_qpos_addresses=qpos_addresses,
        open_targets_rad=np.zeros(3),
        closed_targets_rad=np.full(3, 0.020),
        alphas=(0.0, 0.60, 0.75, 1.0),
    )

    np.testing.assert_array_equal(data.qpos, saved_qpos)
    np.testing.assert_array_equal(data.ctrl, saved_ctrl)
    assert sweep.first_all_target_alpha == pytest.approx(0.60)
    assert sweep.samples[0].target_signed_gap_m.tolist() == pytest.approx(
        [0.015, 0.015, 0.015]
    )
    assert sweep.samples[2].target_signed_gap_m.tolist() == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-12
    )
    assert sweep.samples[-1].max_distal_preload_m == pytest.approx(0.005)
    assert all(sample.target_height_valid.all() for sample in sweep.samples)
    assert sweep.samples[2].target_height_spread_m == pytest.approx(0.0)
    assert sweep.eligible_samples[0].alpha == pytest.approx(0.60)


def test_nearest_taxel_assignment_has_inclusive_six_mm_boundary():
    sites = np.asarray([[0.006, 0.0, 0.0], [0.020, 0.0, 0.0]])
    exact = nearest_taxel_assignment([0.0, 0.0, 0.0], sites)
    assert exact.assigned
    assert exact.taxel_index == 0
    assert exact.distance_m == pytest.approx(0.006)

    outside = nearest_taxel_assignment(
        [-2e-9, 0.0, 0.0], sites, max_assignment_distance_m=0.006
    )
    assert not outside.assigned
    assert outside.taxel_index is None
    assert outside.distance_m > 0.006


def _dynamic_contact_model() -> tuple[mujoco.MjModel, mujoco.MjData, ModelInfo, np.ndarray]:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0" timestep="0.001"/>
          <worldbody>
            <body name="cube_body">
              <freejoint name="cube_joint"/>
              <geom name="cube" type="box" size="0.03 0.03 0.03"
                    friction="0.8 0.005 0.0001" condim="4"/>
            </body>
            <body name="thumb" pos="0.034 0 0">
              <joint name="thumb_slide" type="slide" axis="1 0 0"/>
              <geom name="thumb_tip" type="sphere" size="0.005"/>
              <site name="thumb_pad" pos="-0.005 0 0" size="0.0001"/>
            </body>
            <body name="index" pos="1 0 0">
              <joint name="index_slide" type="slide" axis="1 0 0"/>
              <geom name="index_tip" type="sphere" size="0.005"/>
              <site name="index_pad" size="0.0001"/>
            </body>
            <body name="mid" pos="0 1 0">
              <joint name="mid_slide" type="slide" axis="1 0 0"/>
              <geom name="mid_tip" type="sphere" size="0.005"/>
              <site name="mid_pad" size="0.0001"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_ids = {finger: model.body(finger).id for finger in FINGERS}
    weld_ids = {
        finger: int(model.body_weldid[body_id]) for finger, body_id in body_ids.items()
    }
    cube_joint_id = model.joint("cube_joint").id
    info = ModelInfo(
        root_body_id=body_ids["thumb"],
        cube_body_id=model.body("cube_body").id,
        cube_geom_id=model.geom("cube").id,
        support_geom_id=-10,
        floor_geom_id=-11,
        cube_joint_id=cube_joint_id,
        cube_qpos_adr=int(model.jnt_qposadr[cube_joint_id]),
        cube_dof_adr=int(model.jnt_dofadr[cube_joint_id]),
        active_actuator_ids=np.zeros(0, dtype=int),
        inactive_actuator_ids=np.zeros(0, dtype=int),
        actuator_joint_ids=np.zeros(0, dtype=int),
        actuator_qpos_adrs=np.zeros(0, dtype=int),
        actuator_dof_adrs=np.zeros(0, dtype=int),
        joint_ranges=np.zeros((0, 2)),
        joint_limited=np.zeros(0, dtype=bool),
        force_limits=np.zeros(0),
        distal_weld_ids=weld_ids,
        hand_body_parts={body_id: finger for finger, body_id in body_ids.items()},
        requested_friction=0.8,
    )
    site_ids = np.asarray(
        [
            [model.site("thumb_pad").id],
            [model.site("index_pad").id],
            [model.site("mid_pad").id],
        ],
        dtype=int,
    )
    return model, data, info, site_ids


def test_dynamic_cube_distal_contact_is_assigned_once_to_nearest_pad_taxel():
    model, data, info, site_ids = _dynamic_contact_model()
    assert data.ncon > 0

    snapshot = contact_snapshot(
        model,
        data,
        info,
        classify_faces=True,
        distal_taxel_site_ids=site_ids,
    )

    assert snapshot.distal_pad_force_n is not None
    assert snapshot.distal_pad_force_n[0] > 0.0
    assert snapshot.distal_nonpad_force_n is not None
    assert snapshot.distal_nonpad_force_n[0] == 0.0
    np.testing.assert_allclose(snapshot.distal_pad_force_fraction, [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(snapshot.distal_active_taxel_count, [1, 0, 0])
    assert snapshot.distal_pad_force_n[0] == pytest.approx(snapshot.finger_forces[0])

    model.site_pos[site_ids[0, 0]] = [0.020, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    outside = contact_snapshot(
        model,
        data,
        info,
        classify_faces=True,
        distal_taxel_site_ids=site_ids,
    )
    assert outside.distal_pad_force_n[0] == 0.0
    assert outside.distal_nonpad_force_n[0] == pytest.approx(
        outside.finger_forces[0]
    )
    assert outside.distal_pad_force_fraction[0] == 0.0
    assert outside.distal_active_taxel_count[0] == 0


def test_v5_json_pad_aggregation_is_soft_and_recomputed_from_raw_force():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body><joint name="thumb"/><geom type="sphere" size="0.01"/></body>
          </worldbody>
          <actuator>
            <position name="left_hand_thumb_bend_joint_actuator" joint="thumb"/>
          </actuator>
        </mujoco>
        """
    )
    pad = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, 0.0, 1.0], [0.0, 2.0, 1.0]]
    )
    nonpad = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    fraction = np.divide(
        pad, pad + nonpad, out=np.zeros_like(pad), where=(pad + nonpad) > 0.0
    )
    root = np.zeros((4, 3))
    cube = np.tile([0.14, 0.0, 0.0], (4, 1))
    ctrl = np.asarray([[0.0], [0.9], [1.0], [1.1]])
    qpos = np.asarray([[0.0], [0.8], [0.95], [1.05]])
    traces = {
        "time": np.arange(4, dtype=float),
        "distal_pad_force_n": pad,
        "distal_nonpad_force_n": nonpad,
        "distal_pad_force_fraction": fraction,
        "distal_active_taxel_count": np.asarray(
            [[0, 0, 0], [1, 2, 2], [1, 0, 1], [0, 1, 1]]
        ),
        "root_cube_center_distance_m": np.full(4, 0.14),
        "thumb_bend_command_rad": ctrl[:, 0],
        "thumb_bend_qpos_rad": qpos[:, 0],
        "root_pos": root,
        "cube_pos": cube,
        "ctrl": ctrl,
        "joint_qpos": qpos,
        "control_state": np.asarray(["CLOSE", "VERIFY", "MANIPULATE", "HOLD"]),
    }
    metrics = _v5_fingertip_contact_metrics(
        model,
        None,  # ModelInfo is not consulted by this metric-only reducer.
        {
            "fingertip_contact_preferences": {
                "taxel_assignment_max_distance_m": 0.006
            }
        },
        traces,
    )

    assert metrics["trace_fraction_matches_raw_forces"]
    assert metrics["trace_pose_and_thumb_bend_match_raw_state"]
    assert metrics["verify"]["force_weighted_pad_fraction"]["index"] == 1.0
    assert metrics["operation"]["force_weighted_pad_fraction"]["thumb"] == 0.5
    assert metrics["operation"]["force_weighted_pad_fraction"]["index"] == 0.5
    assert metrics["operation"]["max_active_taxel_count"] == {
        "thumb": 1,
        "index": 1,
        "mid": 1,
    }
    assert "checks" not in metrics


def test_v1_through_v4_trace_keys_remain_unchanged_by_pad_extension():
    model, _ = _closure_model()
    for schema_version in (1, 2, 3, 4):
        traces = _allocate_traces(model, 2, schema_version=schema_version)
        assert "distal_pad_force_n" not in traces
        assert "root_cube_center_distance_m" not in traces
