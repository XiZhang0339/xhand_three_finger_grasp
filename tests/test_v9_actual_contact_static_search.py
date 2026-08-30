from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.grasp_pose import controller_id, grasp_pose_id
from xhand_grasp.scene import ModelInfo
from xhand_grasp.tuning.actual_contact_grasp_pose import (
    ActualContactStaticThresholds,
    _run_joint_controller_local_refinement_stage,
    actual_contact_search_cells,
    apply_precontact_solution,
    evaluate_direct_actual_contact_pose,
    generate_actual_contact_pose_candidates,
    refine_uniform_gap_equal_height_contact_pose,
    retain_top_actual_contact_candidates,
    screen_actual_contact_pose_cell,
    select_dynamic_local_refinement_parents,
)
from xhand_grasp.tuning.pose_preserving_seed_campaign import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
V9_TEMPLATE = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)


def _toy_scene(
    *,
    yaw_deg: float = 30.0,
    middle_y_m: float = 0.010,
    skew_thumb_deg: float = 0.0,
    forbidden_part: str | None = None,
    extra_index_distal: bool = False,
):
    names = ACTIVE_ACTUATORS
    skew = np.radians(skew_thumb_deg)
    thumb_axis = f"{np.cos(skew):.17g} {np.sin(skew):.17g} 0"
    # Repeating the skew axis makes the deliberately bad test rank-one; the
    # normal case has independent Y/Z helpers and a fully attainable retreat.
    thumb_axes = (
        ("1 0 0", "0 1 0", "0 0 1")
        if skew_thumb_deg == 0.0
        else (thumb_axis, thumb_axis, thumb_axis)
    )
    if forbidden_part not in (None, "palm", "ring", "pinky"):
        raise ValueError("forbidden_part must be palm, ring or pinky")
    forbidden_xml = (
        ""
        if forbidden_part is None
        else f"""
          <body name="{forbidden_part}_probe" pos="0 0 .1">
            <geom name="{forbidden_part}_probe_geom" type="sphere" size=".01"/>
          </body>
        """
    )
    extra_index_xml = (
        '<geom name="index_extra_distal_geom" type="sphere" '
        'pos="-.01 0 0" size=".005"/>'
        if extra_index_distal
        else ""
    )
    xml = f"""
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <body name="cube" pos="0 0 0.1" euler="0 0 {yaw_deg}">
          <freejoint name="cube_free"/>
          <geom name="cube_geom" type="box" size=".03 .03 .03"/>
        </body>
        <body name="hand_root" euler="0 0 {yaw_deg}">
          <body name="thumb" pos="-.0351 0 .1">
            <joint name="{names[0]}_j" type="slide" axis="{thumb_axes[0]}" range="-.02 .02"/>
            <joint name="{names[1]}_j" type="slide" axis="{thumb_axes[1]}" range="-.02 .02"/>
            <joint name="{names[2]}_j" type="slide" axis="{thumb_axes[2]}" range="-.02 .02"/>
            <geom name="thumb_geom" type="sphere" size=".005"/>
          </body>
          <body name="index" pos=".0351 -.01 .1">
            <joint name="{names[3]}_j" type="slide" axis="1 0 0" range="-.02 .02"/>
            <joint name="{names[4]}_j" type="slide" axis="0 1 0" range="-.02 .02"/>
            <joint name="{names[5]}_j" type="slide" axis="0 0 1" range="-.02 .02"/>
            <geom name="index_geom" type="sphere" size=".005"/>
            {extra_index_xml}
          </body>
          <body name="mid" pos=".0351 {middle_y_m} .1">
            <joint name="{names[6]}_j" type="slide" axis="1 0 0" range="-.02 .02"/>
            <joint name="{names[7]}_j" type="slide" axis="0 1 0" range="-.02 .02"/>
            <geom name="mid_geom" type="sphere" size=".005"/>
          </body>
          {forbidden_xml}
        </body>
      </worldbody>
      <actuator>
        {''.join(f'<position name="{name}" joint="{name}_j" kp="10" ctrlrange="-.02 .02"/>' for name in names)}
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    active_ids = np.asarray([model.actuator(name).id for name in names], dtype=int)
    joint_ids = model.actuator_trnid[:, 0].astype(int)
    root_id = model.body("hand_root").id
    cube_id = model.body("cube").id
    cube_joint_id = model.joint("cube_free").id
    finger_body_ids = {finger: model.body(finger).id for finger in ("thumb", "index", "mid")}
    forbidden_body_parts = (
        {}
        if forbidden_part is None
        else {model.body(f"{forbidden_part}_probe").id: forbidden_part}
    )
    info = ModelInfo(
        root_body_id=root_id,
        cube_body_id=cube_id,
        cube_geom_id=model.geom("cube_geom").id,
        support_geom_id=-1,
        floor_geom_id=-1,
        cube_joint_id=cube_joint_id,
        cube_qpos_adr=int(model.jnt_qposadr[cube_joint_id]),
        cube_dof_adr=int(model.jnt_dofadr[cube_joint_id]),
        active_actuator_ids=active_ids,
        inactive_actuator_ids=np.asarray([], dtype=int),
        actuator_joint_ids=joint_ids,
        actuator_qpos_adrs=model.jnt_qposadr[joint_ids].astype(int),
        actuator_dof_adrs=model.jnt_dofadr[joint_ids].astype(int),
        joint_ranges=model.jnt_range[joint_ids].copy(),
        joint_limited=model.jnt_limited[joint_ids].astype(bool),
        force_limits=np.max(np.abs(model.actuator_forcerange), axis=1),
        distal_weld_ids={
            finger: int(model.body_weldid[body_id])
            for finger, body_id in finger_body_ids.items()
        },
        hand_body_parts={
            root_id: "palm",
            **{body_id: finger for finger, body_id in finger_body_ids.items()},
            **forbidden_body_parts,
        },
        requested_friction=0.8,
    )
    config = {
        "schema_version": 9,
        "hand_pose": {
            "translation_m": [0.0, 0.0, 0.0],
            "rpy_deg": [0.0, 0.0, yaw_deg],
        },
        "grasp_pose": {
            "nominal_joint_qpos_rad": {name: 0.0 for name in names}
        },
        # These deliberately absurd precontact commands prove the static
        # contact evidence does not scan or interpolate from them.
        "control": {
            "precontact_targets_rad": {name: 99.0 for name in names},
            "contact_preload_targets_rad": {name: -99.0 for name in names},
        },
        "contact_topology": {
            "target_faces": {"thumb": "-X", "index": "+X", "mid": "+X"}
        },
    }
    return model, data, info, config


def test_registered_grid_and_candidate_generation_are_prefix_deterministic():
    template = load_config(V9_TEMPLATE)
    cells = actual_contact_search_cells(template)
    assert len(cells) == 55
    assert cells[0].cell_id == "edge_60mm_thumb_actual_1.40rad"
    assert cells[-1].cell_id == "edge_70mm_thumb_actual_1.60rad"

    four = generate_actual_contact_pose_candidates(
        template, [template], cells[12], count=4, seed=41
    )
    prefix = generate_actual_contact_pose_candidates(
        template, [template], cells[12], count=2, seed=41
    )
    suffix = generate_actual_contact_pose_candidates(
        template, [template], cells[12], count=2, start_index=2, seed=41
    )
    assert prefix + suffix == four
    for record in four:
        metadata = record["config"]["candidate_metadata"]
        assert metadata["contact_pose_interpolation_used"] is False
        assert metadata["cube_pose_sampled"] is False
        assert record["config"]["cube"]["rpy_deg"][2] == pytest.approx(
            27.609990189403167
        )
        assert record["config"]["grasp_pose"]["nominal_joint_qpos_rad"][
            ACTIVE_ACTUATORS[0]
        ] == pytest.approx(cells[12].thumb_actual_center_rad)


def test_rotated_cube_uses_real_geom_witness_and_direct_contact_qpos():
    model, data, info, config = _toy_scene(yaw_deg=31.0)
    before = data.qpos.copy()
    result = evaluate_direct_actual_contact_pose(
        model,
        data,
        info,
        config,
        thresholds=ActualContactStaticThresholds(),
        requested_retreat_m=0.003,
    )
    assert result.static_geometry_pass
    assert result.evaluation_mode == "direct_actual_contact_qpos"
    assert result.direct_contact_qpos_forward_count == 1
    np.testing.assert_array_equal(data.qpos, before)
    assert result.cube_freejoint_qpos_unchanged
    for witness in result.target_witnesses:
        assert witness is not None and witness.passed
        assert witness.normal_alignment == pytest.approx(1.0)
        assert witness.signed_gap_m == pytest.approx(0.0001, abs=1e-12)
        assert witness.edge_margin_m >= 0.019


def test_legacy_static_result_serialization_does_not_gain_v11_evidence_fields():
    model, data, info, config = _toy_scene()
    result = evaluate_direct_actual_contact_pose(
        model, data, info, config, thresholds=ActualContactStaticThresholds()
    )

    payload = result.as_dict()
    assert "nominal_minimum_forbidden_hand_gap_m" not in payload
    assert "nominal_maximum_all_distal_penetration_m" not in payload
    assert "precontact_geometry_evaluated" not in payload


@pytest.mark.parametrize("forbidden_part", ("palm", "ring", "pinky"))
def test_v11_nominal_forbidden_hand_geometry_is_measured(forbidden_part):
    model, data, info, config = _toy_scene(forbidden_part=forbidden_part)
    config["schema_version"] = 11

    result = evaluate_direct_actual_contact_pose(
        model, data, info, config, thresholds=ActualContactStaticThresholds()
    )

    assert result.nominal_minimum_forbidden_hand_gap_m is not None
    assert result.nominal_minimum_forbidden_hand_gap_m < 0.0
    assert result.precontact_geometry_evaluated is False
    assert not result.static_geometry_pass


def test_v11_all_distal_penetration_includes_nonselected_geom():
    model, data, info, config = _toy_scene(extra_index_distal=True)
    config["schema_version"] = 11

    result = evaluate_direct_actual_contact_pose(
        model, data, info, config, thresholds=ActualContactStaticThresholds()
    )

    # The normal index geom is the selected near-touching witness; the second
    # geom on the same distal weld is much deeper on the same target face.
    assert result.target_witnesses[1] is not None
    assert model.geom(result.target_witnesses[1].distal_geom_id).name == "index_geom"
    assert result.off_target_distal_penetrating_count == 0
    assert result.nominal_maximum_all_distal_penetration_m is not None
    assert result.nominal_maximum_all_distal_penetration_m > 0.002
    assert not result.static_geometry_pass


def test_v11_precontact_evaluation_flag_distinguishes_sentinel_from_query():
    model, data, info, config = _toy_scene()
    config["schema_version"] = 11
    valid = evaluate_direct_actual_contact_pose(
        model, data, info, config, thresholds=ActualContactStaticThresholds()
    )
    assert valid.precontact_geometry_evaluated is True
    assert valid.precontact_minimum_hand_gap_m > 0.0

    far = copy.deepcopy(config["grasp_pose"]["nominal_joint_qpos_rad"])
    # Move the thumb beyond the geometry query radius so target evidence is
    # genuinely unavailable.  A mere target-gap miss remains eligible for the
    # v11 safety query because that gap is a DLS objective.
    far[ACTIVE_ACTUATORS[0]] = -0.02
    invalid = evaluate_direct_actual_contact_pose(
        model,
        data,
        info,
        config,
        nominal_joint_qpos_rad=far,
        thresholds=ActualContactStaticThresholds(),
    )
    assert invalid.precontact_geometry_evaluated is False
    assert invalid.precontact_minimum_hand_gap_m == pytest.approx(-0.02)


def test_jacobian_retreat_is_separated_and_closes_along_face_normals():
    model, data, info, config = _toy_scene(yaw_deg=23.0)
    result = evaluate_direct_actual_contact_pose(
        model,
        data,
        info,
        config,
        thresholds=ActualContactStaticThresholds(),
        requested_retreat_m=0.003,
    )
    assert result.precontact_minimum_hand_gap_m == pytest.approx(0.0031, abs=1e-12)
    for retreat in result.retreat_evidence:
        assert retreat is not None and retreat.passed
        assert retreat.measured_outward_retreat_m == pytest.approx(0.003, abs=1e-12)
        assert retreat.predicted_outward_retreat_m == pytest.approx(0.003, abs=1e-12)
        assert retreat.inward_speed_m_s > 0.0
        assert retreat.closure_angle_deg == pytest.approx(0.0, abs=1e-9)

    promoted = apply_precontact_solution(config, result)
    np.testing.assert_allclose(
        [promoted["control"]["precontact_targets_rad"][name] for name in ACTIVE_ACTUATORS],
        result.precontact_joint_qpos_rad,
    )
    assert promoted["grasp_pose"] == config["grasp_pose"]


def test_bad_gap_and_bad_closure_direction_cannot_be_promoted():
    model, data, info, config = _toy_scene()
    far = copy.deepcopy(config["grasp_pose"]["nominal_joint_qpos_rad"])
    far[ACTIVE_ACTUATORS[0]] = -0.002
    bad_gap = evaluate_direct_actual_contact_pose(
        model,
        data,
        info,
        config,
        nominal_joint_qpos_rad=far,
        thresholds=ActualContactStaticThresholds(),
    )
    assert not bad_gap.static_geometry_pass
    assert bad_gap.target_witnesses[0] is not None
    assert not bad_gap.target_witnesses[0].gap_ok
    with pytest.raises(ValueError, match="passed static result"):
        apply_precontact_solution(config, bad_gap)

    skew_model, skew_data, skew_info, skew_config = _toy_scene(skew_thumb_deg=40.0)
    bad_direction = evaluate_direct_actual_contact_pose(
        skew_model,
        skew_data,
        skew_info,
        skew_config,
        thresholds=ActualContactStaticThresholds(),
        requested_retreat_m=0.004,
    )
    assert not bad_direction.static_geometry_pass
    assert bad_direction.retreat_evidence[0] is not None
    assert bad_direction.retreat_evidence[0].closure_angle_deg == pytest.approx(40.0)
    assert not bad_direction.retreat_evidence[0].direction_ok


def test_edge_margin_is_a_hard_real_witness_check():
    model, data, info, config = _toy_scene(middle_y_m=0.0298)
    result = evaluate_direct_actual_contact_pose(
        model,
        data,
        info,
        config,
        thresholds=ActualContactStaticThresholds(),
    )
    assert not result.static_geometry_pass
    middle = result.target_witnesses[2]
    assert middle is not None
    assert middle.edge_margin_m == pytest.approx(0.0002, abs=1e-12)
    assert not middle.edge_ok


def test_top_k_reduction_is_worker_order_independent():
    def record(candidate_id: int, passed: bool, angle: float):
        return {
            "candidate_id": candidate_id,
            "cell_index": 3,
            "grasp_pose_id": f"pose-{candidate_id}",
            "static_metrics": {
                "static_geometry_pass": passed,
                "missing_target_witness_count": 0,
                "off_target_distal_penetrating_count": 0,
                "minimum_active_nondistal_gap_m": 0.001,
                "precontact_minimum_hand_gap_m": 0.002,
                "contact_height_spread_m": 0.001,
                "target_witness": {
                    finger: {"signed_gap_m": 0.0} for finger in ("thumb", "index", "mid")
                },
                "retreat_evidence": {
                    finger: {"closure_angle_deg": angle}
                    for finger in ("thumb", "index", "mid")
                },
            },
        }

    values = [record(7, False, 1.0), record(8, True, 12.0), record(9, True, 5.0)]
    forward = retain_top_actual_contact_candidates(values, top_k=2)
    reverse = retain_top_actual_contact_candidates(list(reversed(values)), top_k=2)
    assert [item["candidate_id"] for item in forward[3]] == [9, 8]
    assert forward == reverse


def test_uniform_gap_equal_height_refinement_keeps_thumb_center_and_improves(
    tmp_path,
):
    config = load_config(V9_TEMPLATE)
    config["cube"]["edge_m"] = 0.067
    config["hand_pose"] = {
        "rpy_deg": [-0.28889597411236406, 122.05523348398599, 5.820983983477403],
        "translation_m": [
            0.0275032902851953,
            -0.00823848702911208,
            0.2575335035233917,
        ],
    }
    nominal = {
        ACTIVE_ACTUATORS[0]: 1.45,
        ACTIVE_ACTUATORS[1]: 0.167614001939987,
        ACTIVE_ACTUATORS[2]: 0.7910027649352128,
        ACTIVE_ACTUATORS[3]: -0.003339338281401996,
        ACTIVE_ACTUATORS[4]: 0.625539908172425,
        ACTIVE_ACTUATORS[5]: 1.213967553136135,
        ACTIVE_ACTUATORS[6]: 0.736018877411348,
        ACTIVE_ACTUATORS[7]: 1.122022969582119,
    }
    config["grasp_pose"]["nominal_joint_qpos_rad"] = nominal
    config["control"]["contact_preload_targets_rad"] = copy.deepcopy(nominal)
    source = {
        "candidate_id": 101000036008292,
        "cell_index": 36,
        "cell_id": "edge_67mm_thumb_actual_1.45rad",
        "edge_m": 0.067,
        "thumb_actual_center_rad": 1.45,
        "grasp_pose_id": grasp_pose_id(config),
        "controller_id": controller_id(config),
        "candidate_sha256": canonical_sha256(config),
        "config": config,
    }
    screened = screen_actual_contact_pose_cell((source,), top_k=1)["retained"][0]
    assert screened["static_pass"]
    refined = refine_uniform_gap_equal_height_contact_pose(
        screened, maximum_iterations=4
    )
    assert refined["static_pass"]
    assert refined["config"]["grasp_pose"]["nominal_joint_qpos_rad"][
        ACTIVE_ACTUATORS[0]
    ] == pytest.approx(1.45, abs=1e-12)
    metrics = refined["uniform_refinement"]
    assert metrics["improved"]
    assert metrics["final_residual_norm_m"] < 1e-6
    gaps = metrics["final_measurement_m"][:3]
    np.testing.assert_allclose(gaps, 0.00015, atol=3e-8)
    assert abs(metrics["final_measurement_m"][3]) < 1e-6
    assert abs(metrics["final_measurement_m"][4]) < 1e-6

    local = _run_joint_controller_local_refinement_stage(
        (refined,),
        tmp_path,
        stage="small_budget",
        top_count=1,
        candidates_per_pose=2,
        seed=20260821,
    )
    assert local.summary["declared_candidate_budget"] == 2
    assert local.summary["generated_candidate_count"] == 2
    assert 1 <= local.summary["dynamic_promoted_count"] <= 2
    report = json.loads(
        (tmp_path / "static/small_budget/joint_controller_local_refinement.json")
        .read_text(encoding="utf-8")
    )
    assert len(report["diagnostics"]) == 2
    for candidate in local.records:
        metadata = candidate["config"]["candidate_metadata"]
        assert metadata["local_refinement_ik_projection"].startswith("uniform_gap")
        assert "local_refinement_joint_root_lhs" in metadata
        assert set(candidate["config"]["control"]["contact_preload_targets_rad"]) == set(
            ACTIVE_ACTUATORS
        )
        assert candidate["config"]["control_protocol"]["close_s"] in (1.0, 1.25, 1.5)

    resumed = _run_joint_controller_local_refinement_stage(
        (refined,),
        tmp_path,
        stage="small_budget",
        top_count=1,
        candidates_per_pose=2,
        seed=20260821,
    )
    assert [value["candidate_sha256"] for value in resumed.records] == [
        value["candidate_sha256"] for value in local.records
    ]


def test_local_refinement_parent_selection_follows_dynamic_evidence():
    config = load_config(V9_TEMPLATE)

    def dynamic_record(candidate_id: int, grasp_success: bool) -> dict:
        return {
            "candidate_id": candidate_id,
            "candidate_sha256": canonical_sha256(config),
            "grasp_pose_id": grasp_pose_id(config),
            "controller_id": controller_id(config),
            "config": copy.deepcopy(config),
            "summary": {
                "passed": False,
                "stage_status": {"grasp_success": grasp_success},
                "metrics": {},
            },
        }

    first = dynamic_record(100, False)
    second = dynamic_record(200, True)
    selected = select_dynamic_local_refinement_parents(
        (first, second), top_count=1
    )
    assert selected[0]["candidate_id"] == 200

    first["summary"]["stage_status"]["grasp_success"] = True
    second["summary"]["stage_status"]["grasp_success"] = False
    selected_after_evidence_change = select_dynamic_local_refinement_parents(
        (first, second), top_count=1
    )
    assert selected_after_evidence_change[0]["candidate_id"] == 100
