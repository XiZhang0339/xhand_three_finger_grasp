from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.relative_wrist_pose import transform_relative_wrist_pose
from xhand_grasp.scene import rpy_degrees_to_rotation_matrix
from xhand_grasp.tuning.relative_wrist_pose_search import (
    NON_THUMB_ACTUATORS,
    THUMB_BEND_ACTUATOR,
    VARIABLE_COUNT,
    RelativeWristDLSSettings,
    RelativeWristPoseSearchPolicy,
    RelativeWristTrialEvaluation,
    RelativeWristVariables,
    actual_contact_trial_evaluation,
    apply_relative_wrist_transform,
    build_relative_wrist_strata,
    materialize_relative_wrist_candidate,
    relative_wrist_boundary_violations,
    retain_per_edge_orbit_quota,
    solve_orientation_aware_dls,
)


ROOT = Path(__file__).resolve().parents[1]
V10_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)
V11_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift.json"
)


def _policy() -> RelativeWristPoseSearchPolicy:
    return RelativeWristPoseSearchPolicy(
        clockwise_orbit_deg=(0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0),
        root_delta_cube_m={
            "x": (-0.012, 0.012),
            "y": (-0.012, 0.012),
            "z": (-0.015, 0.015),
        },
        wrist_local_rotvec_deg={
            "x": (-6.0, 6.0),
            "y": (-6.0, 6.0),
            "z": (-6.0, 6.0),
        },
        max_wrist_local_rotvec_norm_deg=8.0,
        root_cube_distance_m=(0.135, 0.210),
    )


@pytest.fixture
def base_config() -> dict:
    config = load_config(V10_CONFIG)
    config["relative_wrist_pose_search"] = _policy().as_config()
    return config


def _joint_bounds() -> dict[str, tuple[float, float]]:
    return {name: (-3.0, 3.0) for name in NON_THUMB_ACTUATORS}


def test_schema_v11_strata_are_700_and_actual_contact_cell_compatible():
    strata = build_relative_wrist_strata(
        tuple(value / 1000.0 for value in range(85, 105)),
        (1.40, 1.45, 1.50, 1.55, 1.60),
        _policy().clockwise_orbit_deg,
    )

    assert len(strata) == 700
    assert strata[0].cell_index == strata[0].stratum_index == 0
    assert strata[0].cell_id == strata[0].stratum_id
    assert strata[0].as_dict()["clockwise_orbit_deg"] == 0.0
    assert strata[6].clockwise_orbit_deg == 15.0
    assert strata[7].thumb_actual_center_rad == pytest.approx(1.45)
    assert strata[35].edge_m == pytest.approx(0.086)
    assert strata[-1].edge_m == pytest.approx(0.104)
    assert strata[-1].clockwise_orbit_deg == 15.0


def test_clockwise_orbit_moves_position_and_orientation_together(base_config):
    cube_position = np.asarray([0.1, -0.2, 0.3])
    cube_rotation = rpy_degrees_to_rotation_matrix((11.0, -7.0, 31.0))
    source_local = np.asarray((0.18, 0.0, 0.02))
    source_relative_rotation = rpy_degrees_to_rotation_matrix((3.0, 8.0, 12.0))
    hand_position = cube_position + cube_rotation @ source_local
    hand_rotation = cube_rotation @ source_relative_rotation
    base_config["cube"]["center_xy_m"] = cube_position[:2].tolist()
    base_config["cube"]["rpy_deg"] = [11.0, -7.0, 31.0]
    # The shared support-placement helper determines cube Z, so use that
    # actual center when constructing the equivalent source root.
    from xhand_grasp.tuning.actual_contact_grasp_pose import _cube_world_position

    actual_cube_position = _cube_world_position(base_config)
    base_config["hand_pose"]["translation_m"] = (
        actual_cube_position + cube_rotation @ source_local
    ).tolist()
    base_config["hand_pose"]["rpy_deg"] = [
        3.0,
        8.0,
        43.0,
    ]
    # Recompute the exact root rotation using the same cube-relative source;
    # the Euler branch itself is not part of the orbit assertion.
    from xhand_grasp.relative_wrist_pose import rotation_matrix_to_rpy_degrees

    base_config["hand_pose"]["rpy_deg"] = rotation_matrix_to_rpy_degrees(
        hand_rotation,
        reference_rpy_deg=base_config["hand_pose"]["rpy_deg"],
    ).tolist()

    transformed = apply_relative_wrist_transform(
        base_config,
        clockwise_orbit_deg=15.0,
    )
    shared = transform_relative_wrist_pose(
        source_cube_world_position_m=actual_cube_position,
        source_cube_world_rotation=cube_rotation,
        source_root_world_position_m=base_config["hand_pose"]["translation_m"],
        source_root_world_rotation=hand_rotation,
        clockwise_orbit_deg=15.0,
    )

    np.testing.assert_allclose(
        transformed.hand_translation_world_m, shared.root_world_position_m
    )
    np.testing.assert_allclose(
        transformed.hand_rotation_world_from_root, shared.root_world_rotation
    )
    assert transformed.root_cube_distance_m == pytest.approx(
        np.linalg.norm(source_local)
    )
    # Viewed from cube +Z, +X moves toward -Y for a positive clockwise orbit.
    assert transformed.cube_to_hand_translation_cube_m[1] < 0.0
    assert not np.allclose(
        transformed.cube_to_hand_rotation, source_relative_rotation
    )


def test_materialization_has_exactly_13_variables_and_keeps_cube_and_thumb(base_config):
    initial_cube = copy.deepcopy(base_config["cube"])
    thumb = base_config["grasp_pose"]["nominal_joint_qpos_rad"][
        THUMB_BEND_ACTUATOR
    ]
    variables = RelativeWristVariables.from_config(base_config)
    vector = variables.as_array()
    assert vector.shape == (VARIABLE_COUNT,) == (13,)
    vector[0] += 0.01
    vector[7:10] = (0.001, -0.002, 0.003)
    vector[10:13] = np.radians((1.0, -2.0, 3.0))

    candidate = materialize_relative_wrist_candidate(
        base_config,
        RelativeWristVariables.from_array(vector),
        clockwise_orbit_deg=5.0,
    )

    assert candidate["cube"] == initial_cube
    assert candidate["grasp_pose"]["nominal_joint_qpos_rad"][
        THUMB_BEND_ACTUATOR
    ] == pytest.approx(thumb)
    metadata = candidate["candidate_metadata"]["relative_wrist_pose_search"]
    assert metadata["clockwise_orbit_deg"] == 5.0
    assert metadata["cube_pose_sampled"] is False
    assert metadata["hand_root_fixed_during_simulation"] is True


def test_rematerializing_a_static_record_uses_anchor_and_does_not_double_orbit(
    base_config,
):
    vector = RelativeWristVariables.from_config(base_config).as_array()
    vector[7:10] = (0.001, -0.002, 0.003)
    vector[10:13] = np.radians((1.0, -2.0, 3.0))
    variables = RelativeWristVariables.from_array(vector)
    first = materialize_relative_wrist_candidate(
        base_config,
        variables,
        clockwise_orbit_deg=10.0,
    )
    first["candidate_metadata"]["relative_wrist_pose_search"][
        "source_role"
    ] = "primary_anchor"

    recovered = RelativeWristVariables.from_config(first)
    second = materialize_relative_wrist_candidate(
        first,
        recovered,
        clockwise_orbit_deg=10.0,
    )

    assert recovered.as_array() == pytest.approx(variables.as_array())
    assert second["hand_pose"]["translation_m"] == pytest.approx(
        first["hand_pose"]["translation_m"]
    )
    assert second["hand_pose"]["rpy_deg"] == pytest.approx(
        first["hand_pose"]["rpy_deg"]
    )
    assert second["candidate_metadata"]["relative_wrist_pose_search"][
        "anchor_hand_pose"
    ] == base_config["hand_pose"]
    assert second["candidate_metadata"]["relative_wrist_pose_search"][
        "source_role"
    ] == "primary_anchor"


def test_boundary_interface_rejects_axis_norm_joint_distance_and_pose(base_config):
    policy = _policy()
    values = RelativeWristVariables.from_config(base_config).as_array()
    values[0] = 4.0
    values[7] = 0.200
    values[10:13] = np.radians((6.0, 6.0, 6.0))
    variables = RelativeWristVariables.from_array(values)

    reasons = relative_wrist_boundary_violations(
        base_config,
        variables,
        policy,
        clockwise_orbit_deg=5.0,
        joint_bounds=_joint_bounds(),
        check_pose_constraints=False,
    )

    assert f"joint_out_of_bounds:{NON_THUMB_ACTUATORS[0]}" in reasons
    assert "root_delta_out_of_bounds:x" in reasons
    assert "wrist_rotvec_norm_out_of_bounds" in reasons
    assert "root_cube_distance_out_of_bounds" in reasons


def test_boundary_interface_checks_registered_cube_in_root_axes(base_config):
    changed = copy.deepcopy(base_config)
    changed["pose_constraints"]["cube_position_in_root_m"]["x"] = [0.20, 0.21]

    reasons = relative_wrist_boundary_violations(
        changed,
        RelativeWristVariables.from_config(changed),
        _policy(),
        clockwise_orbit_deg=0.0,
        joint_bounds=_joint_bounds(),
    )

    assert (
        "pose_constraint_out_of_bounds:cube_position_in_root_m.x" in reasons
    )


@dataclass
class _Witness:
    finger: str
    signed_gap_m: float
    normal_alignment: float
    cube_point_world_m: tuple[float, float, float]


@dataclass
class _StaticResult:
    target_witnesses: tuple[_Witness | None, ...]
    off_target_distal_penetrating_count: int = 0
    minimum_active_nondistal_gap_m: float = 0.001
    cube_freejoint_qpos_unchanged: bool = True
    nominal_minimum_forbidden_hand_gap_m: float | None = 0.001
    nominal_maximum_all_distal_penetration_m: float | None = 0.0001
    precontact_geometry_evaluated: bool | None = True
    precontact_minimum_hand_gap_m: float = 0.001


def test_actual_contact_measurement_keeps_forbidden_contact_as_hard_gate():
    witnesses = tuple(
        _Witness(finger, 0.0001, 0.99, (0.0, 0.0, height))
        for finger, height in zip(("thumb", "index", "mid"), (0.10, 0.101, 0.102))
    )
    result = _StaticResult(witnesses, minimum_active_nondistal_gap_m=-1e-5)

    evaluation = actual_contact_trial_evaluation(result, (0.0, 0.0, 1.0))

    assert evaluation.measurement is not None
    assert evaluation.measurement[3:5] == pytest.approx((0.001, 0.002))
    assert evaluation.safe is False
    assert "active_nondistal_penetration" in evaluation.safety_violations


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        (
            "nominal_minimum_forbidden_hand_gap_m",
            -1e-5,
            "nominal_forbidden_hand_penetration",
        ),
        (
            "nominal_maximum_all_distal_penetration_m",
            0.00201,
            "all_distal_penetration_over_limit",
        ),
    ),
)
def test_v11_extended_nominal_geometry_is_a_hard_gate(field, value, reason):
    witnesses = tuple(
        _Witness(finger, 0.0001, 0.99, (0.0, 0.0, height))
        for finger, height in zip(("thumb", "index", "mid"), (0.10, 0.101, 0.102))
    )
    kwargs = {field: value}
    result = _StaticResult(witnesses, **kwargs)

    evaluation = actual_contact_trial_evaluation(
        result, (0.0, 0.0, 1.0), require_v11_safety_evidence=True
    )

    assert not evaluation.safe
    assert reason in evaluation.safety_violations


def test_v11_precontact_sentinel_is_not_treated_as_geometry():
    witnesses = tuple(
        _Witness(finger, 0.0001, 0.99, (0.0, 0.0, height))
        for finger, height in zip(("thumb", "index", "mid"), (0.10, 0.101, 0.102))
    )
    result = _StaticResult(
        witnesses,
        precontact_geometry_evaluated=False,
        precontact_minimum_hand_gap_m=-0.02,
    )

    evaluation = actual_contact_trial_evaluation(
        result, (0.0, 0.0, 1.0), require_v11_safety_evidence=True
    )

    assert "precontact_geometry_not_evaluated" in evaluation.safety_violations
    assert "precontact_hand_penetration" not in evaluation.safety_violations


def test_v11_valid_precontact_evidence_is_safe():
    witnesses = tuple(
        _Witness(finger, 0.0001, 0.99, (0.0, 0.0, height))
        for finger, height in zip(("thumb", "index", "mid"), (0.10, 0.101, 0.102))
    )

    evaluation = actual_contact_trial_evaluation(
        _StaticResult(witnesses),
        (0.0, 0.0, 1.0),
        require_v11_safety_evidence=True,
    )

    assert evaluation.safe


def _linear_fake_evaluator(
    base_q0: float,
    *,
    reject_delta_x_over: float | None = None,
):
    desired_q0 = base_q0 + 0.015

    def evaluate(config):
        metadata = config["candidate_metadata"]["relative_wrist_pose_search"]
        delta = np.asarray(metadata["root_delta_cube_m"])
        rotvec = np.radians(metadata["wrist_local_rotvec_deg"])
        q0 = float(
            config["grasp_pose"]["nominal_joint_qpos_rad"][
                NON_THUMB_ACTUATORS[0]
            ]
        )
        measurement = (
            0.00015 + 0.02 * (q0 - desired_q0),
            0.00015,
            0.00015,
            delta[0] - 0.001,
            0.1 * (rotvec[2] - 0.01),
            1.0,
            1.0,
            1.0,
        )
        unsafe = (
            ("active_nondistal_penetration",)
            if reject_delta_x_over is not None and delta[0] > reject_delta_x_over
            else ()
        )
        return RelativeWristTrialEvaluation(object(), measurement, unsafe)

    return evaluate


def test_orientation_aware_dls_improves_joint_translation_and_rotation(base_config):
    initial = RelativeWristVariables.from_config(base_config)
    base_q0 = initial.non_thumb_joint_qpos_rad[0]
    thumb_before = base_config["grasp_pose"]["nominal_joint_qpos_rad"][
        THUMB_BEND_ACTUATOR
    ]
    settings = RelativeWristDLSSettings(
        maximum_iterations=8,
        contact_tolerance=1e-5,
        regularization_weight=1e-4,
    )

    result = solve_orientation_aware_dls(
        base_config,
        clockwise_orbit_deg=0.0,
        settings=settings,
        evaluator=_linear_fake_evaluator(base_q0),
        joint_bounds=_joint_bounds(),
        check_pose_constraints=False,
    )

    assert result.diagnostics["final_contact_objective"] < 0.03
    assert result.variables.non_thumb_joint_qpos_rad[0] > base_q0
    assert result.variables.root_delta_cube_m[0] > 0.0
    assert result.variables.wrist_local_rotvec_rad[2] > 0.0
    assert result.config["grasp_pose"]["nominal_joint_qpos_rad"][
        THUMB_BEND_ACTUATOR
    ] == pytest.approx(thumb_before)
    assert result.diagnostics["variable_names"][-6:] == [
        "root_delta_cube_m.x",
        "root_delta_cube_m.y",
        "root_delta_cube_m.z",
        "wrist_local_rotvec_rad.x",
        "wrist_local_rotvec_rad.y",
        "wrist_local_rotvec_rad.z",
    ]


def test_dls_line_search_never_accepts_an_unsafe_trial(base_config):
    initial = RelativeWristVariables.from_config(base_config)
    result = solve_orientation_aware_dls(
        base_config,
        clockwise_orbit_deg=0.0,
        settings=RelativeWristDLSSettings(
            maximum_iterations=4,
            regularization_weight=1e-4,
        ),
        evaluator=_linear_fake_evaluator(
            initial.non_thumb_joint_qpos_rad[0], reject_delta_x_over=0.0002
        ),
        joint_bounds=_joint_bounds(),
        check_pose_constraints=False,
    )

    rejected = [
        trial
        for iteration in result.diagnostics["iterations"]
        for trial in iteration["line_search"]
        if "active_nondistal_penetration" in trial["reasons"]
    ]
    assert rejected
    assert result.variables.root_delta_cube_m[0] <= 0.0002 + 1e-12
    assert result.diagnostics["final_safety_violations"] == []


def test_dls_finite_difference_rejects_unsafe_measurement(base_config):
    initial = RelativeWristVariables.from_config(base_config)
    base_q0 = initial.non_thumb_joint_qpos_rad[0]

    def evaluator(config):
        q0 = float(
            config["grasp_pose"]["nominal_joint_qpos_rad"][
                NON_THUMB_ACTUATORS[0]
            ]
        )
        measurement = (
            0.0002 + 0.02 * (q0 - base_q0),
            0.00015,
            0.00015,
            0.001,
            0.001,
            1.0,
            1.0,
            1.0,
        )
        reasons = (
            ("fd_forbidden_penetration",) if q0 > base_q0 + 1e-12 else ()
        )
        return RelativeWristTrialEvaluation(object(), measurement, reasons)

    result = solve_orientation_aware_dls(
        base_config,
        clockwise_orbit_deg=0.0,
        settings=RelativeWristDLSSettings(maximum_iterations=1),
        evaluator=evaluator,
        joint_bounds=_joint_bounds(),
        check_pose_constraints=False,
    )

    rejections = result.diagnostics["iterations"][0][
        "finite_difference_rejections"
    ]
    assert any(
        item["variable"] == NON_THUMB_ACTUATORS[0]
        and item["direction"] == 1.0
        and item["reasons"] == ["fd_forbidden_penetration"]
        for item in rejections
    )


def _record(edge: float, orbit: float, ordinal: int, score: float) -> dict:
    return {
        "candidate_id": ordinal,
        "edge_m": edge,
        "clockwise_orbit_deg": orbit,
        "static_rank": [score, ordinal],
    }


def test_per_edge_quota_is_worker_order_independent_and_covers_orbits():
    records = [
        _record(edge, orbit, ordinal, score=float((ordinal * 17) % 23))
        for ordinal, (edge, orbit) in enumerate(
            (
                (edge, orbit)
                for edge in (0.085, 0.086)
                for orbit in _policy().clockwise_orbit_deg
                for _ in range(2)
            ),
            start=1,
        )
    ]
    shuffled = list(reversed(records[::2])) + list(reversed(records[1::2]))

    first = retain_per_edge_orbit_quota(records, per_edge=12)
    second = retain_per_edge_orbit_quota(shuffled, per_edge=12)

    assert [value["candidate_id"] for value in first.records] == [
        value["candidate_id"] for value in second.records
    ]
    assert first.per_edge_selected_count == {0.085: 12, 0.086: 12}
    assert first.quota_satisfied is True
    for edge in (0.085, 0.086):
        assert first.per_edge_orbit_coverage_deg[edge] == _policy().clockwise_orbit_deg


def test_per_edge_quota_reports_shortfall_without_borrowing_from_other_sizes():
    records = [
        _record(0.085, 0.0, ordinal, float(ordinal)) for ordinal in range(1, 11)
    ] + [
        _record(0.086, 0.0, ordinal, float(ordinal)) for ordinal in range(11, 31)
    ]

    selected = retain_per_edge_orbit_quota(records, per_edge=12)

    assert selected.per_edge_selected_count == {0.085: 10, 0.086: 12}
    assert selected.deficient_edges_m == (0.085,)
    assert selected.quota_satisfied is False


def test_policy_round_trip_accepts_experiment_serialization_shape():
    raw = _policy().as_config()
    config = {"relative_wrist_pose_search": {**raw, "budget": {"ignored": True}}}

    parsed = RelativeWristPoseSearchPolicy.from_config(config)

    assert parsed == _policy()


def test_registered_v11_config_compiles_the_real_contact_evaluator():
    from xhand_grasp.tuning.relative_wrist_pose_search import (
        build_actual_contact_trial_evaluator,
    )

    config = load_config(V11_CONFIG)
    policy = RelativeWristPoseSearchPolicy.from_config(config)
    evaluator, bounds = build_actual_contact_trial_evaluator(config)
    variables = RelativeWristVariables.from_config(config)

    assert not relative_wrist_boundary_violations(
        config,
        variables,
        policy,
        clockwise_orbit_deg=0.0,
        joint_bounds=bounds,
    )
    result = evaluator(config)
    assert result.measurement is not None
    assert len(result.measurement) == 8
    # The template deliberately misses both a target gap and the 5 mm height
    # band, so it is not a static pass.  Those are DLS objective terms, not
    # collision hazards: its precontact safety geometry must still be queried
    # and the real evaluator must return a safe measurement.
    assert not result.static_result.static_geometry_pass
    assert not result.static_result.target_witnesses[1].gap_ok
    assert result.static_result.contact_height_spread_m > 0.005
    assert result.static_result.precontact_geometry_evaluated is True
    assert result.safety_violations == ()


def test_production_dls_can_differentiate_safe_gap_and_height_near_miss():
    config = load_config(V11_CONFIG)

    result = solve_orientation_aware_dls(
        config,
        clockwise_orbit_deg=0.0,
        settings=RelativeWristDLSSettings(maximum_iterations=1),
    )

    assert result.diagnostics["initial_safety_violations"] == []
    iteration = result.diagnostics["iterations"][0]
    assert iteration["jacobian_rank"] > 0
    assert iteration["finite_difference_rejections"] == []
    assert result.static_result.cube_freejoint_qpos_unchanged is True
