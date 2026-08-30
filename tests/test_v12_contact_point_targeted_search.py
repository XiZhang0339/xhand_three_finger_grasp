from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.scene import rpy_degrees_to_rotation_matrix
from xhand_grasp.tuning.contact_point_targeted_search import (
    DEFAULT_SAMPLE_COUNT,
    MEASUREMENT_NAMES,
    SIGNED_ORBIT_DEG,
    VARIABLE_COUNT,
    ContactPointPlan,
    ContactPointSearchPolicy,
    CubeFaceContactPoint,
    PointTargetDLSSettings,
    PointTargetTrialEvaluation,
    PointTargetVariables,
    assert_frozen_contact_point_plan,
    bind_frozen_contact_point_plan,
    build_point_target_trial_evaluator,
    contact_point_world_positions,
    evaluate_contact_point_plan_geometry,
    generate_contact_point_plans,
    materialize_point_target_candidate,
    point_target_boundary_violations,
    point_target_static_acceptance,
    point_target_static_record,
    point_target_trial_evaluation,
    retain_point_target_static_candidates,
    retain_ranked_point_plans,
    solve_point_target_dls,
)
from xhand_grasp.tuning.actual_contact_grasp_pose import _cube_world_position


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_90mm_contact_point_targeted_actual_grasp_pose.json"
)


@pytest.fixture
def config() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture
def policy(config) -> ContactPointSearchPolicy:
    return ContactPointSearchPolicy.from_config(config)


def test_registered_plan_and_policy_use_one_canonical_hash(config):
    plan = ContactPointPlan.from_config(config["contact_point_plan"])
    policy = ContactPointSearchPolicy.from_config(config)

    assert plan.point_plan_id == (
        "fa03e0eff616a29124650a9fc596be18f4534d063eb56c39e4c39417539f2833"
    )
    assert plan.as_config() == config["contact_point_plan"]
    assert policy.as_config() == config["contact_point_search"]
    assert policy.sample_count == DEFAULT_SAMPLE_COUNT
    assert policy.signed_orbit_deg == SIGNED_ORBIT_DEG


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("point_plan_id", "0" * 64),
        lambda value: value["target_points_cube_local_m"]["thumb"].__setitem__(
            1, 0.011
        ),
        lambda value: value.__setitem__("frozen", False),
    ),
)
def test_plan_authentication_rejects_hash_derived_point_and_freeze_tampering(
    config, mutation
):
    raw = copy.deepcopy(config["contact_point_plan"])
    mutation(raw)
    with pytest.raises(ValueError):
        ContactPointPlan.from_config(raw)


def test_full_120k_halton_prefix_is_deterministic_and_hard_filtered(policy):
    first = generate_contact_point_plans(policy)
    second = generate_contact_point_plans(policy)

    assert first.sample_count == second.sample_count == 120_000
    assert first.eligible_count == second.eligible_count
    assert first.eligible_count > 256
    assert len(first.retained) == len(second.retained) == 256
    assert [value.sample_index for value in first.retained] == [
        value.sample_index for value in second.retained
    ]
    assert [value.plan.point_plan_id for value in first.retained] == [
        value.plan.point_plan_id for value in second.retained
    ]
    assert first.retained[0].sample_index == 0
    assert first.retained[0].plan.point_plan_id == (
        "fa03e0eff616a29124650a9fc596be18f4534d063eb56c39e4c39417539f2833"
    )
    for generated in first.retained:
        assert generated.metrics.hard_filter_pass
        assert generated.metrics.minimum_edge_margin_m >= 0.020 - 1e-12
        assert generated.metrics.height_spread_m <= 0.005 + 1e-12
        assert generated.metrics.index_middle_separation_m >= 0.010 - 1e-12
        for finger, reference in policy.reference_points.items():
            actual = generated.plan.points[finger]
            assert abs(actual.y_m - reference.y_m) <= 0.008 + 1e-12
            assert abs(actual.z_m - reference.z_m) <= 0.008 + 1e-12


def test_geometric_filter_boundaries_are_inclusive(policy):
    plan = ContactPointPlan(
        0.090,
        0.002,
        {
            "thumb": CubeFaceContactPoint("-X", 0.025, 0.010),
            "index": CubeFaceContactPoint("+X", 0.005, 0.010),
            "mid": CubeFaceContactPoint("+X", 0.015, 0.015),
        },
    )
    metrics = evaluate_contact_point_plan_geometry(plan, policy)
    assert metrics.hard_filter_pass
    assert metrics.minimum_edge_margin_m == pytest.approx(0.020)
    assert metrics.height_spread_m == pytest.approx(0.005)
    assert metrics.index_middle_separation_m == pytest.approx(0.010)

    too_close = ContactPointPlan(
        0.090,
        0.002,
        {
            **dict(plan.points),
            "mid": CubeFaceContactPoint("+X", 0.014999, 0.015),
        },
    )
    assert not evaluate_contact_point_plan_geometry(too_close, policy).hard_filter_pass


def test_bind_is_copy_only_and_prevents_plan_switch(config, policy):
    seed = ContactPointPlan.from_config(config["contact_point_plan"])
    assert assert_frozen_contact_point_plan(config).point_plan_id == seed.point_plan_id
    before = copy.deepcopy(config)
    rebound = bind_frozen_contact_point_plan(config, seed)
    assert config == before
    assert rebound["contact_point_plan"] == seed.as_config()

    other = generate_contact_point_plans(
        policy, sample_count=500, retain_count=2
    ).retained[1].plan
    with pytest.raises(ValueError, match="different contact_point_plan"):
        bind_frozen_contact_point_plan(config, other)

    unbound = copy.deepcopy(config)
    del unbound["contact_point_plan"]
    selected = bind_frozen_contact_point_plan(unbound, other)
    assert assert_frozen_contact_point_plan(
        selected, other.point_plan_id
    ).point_plan_id == other.point_plan_id


def test_cube_local_target_points_transform_with_rotated_cube(config):
    plan = ContactPointPlan.from_config(config["contact_point_plan"])
    position = np.asarray((0.2, -0.3, 0.4))
    rotation = rpy_degrees_to_rotation_matrix((17.0, -11.0, 53.0))
    world = contact_point_world_positions(
        plan,
        cube_world_position_m=position,
        cube_world_rotation=rotation,
    )
    for finger, point in world.items():
        reconstructed = rotation.T @ (np.asarray(point) - position)
        np.testing.assert_allclose(
            reconstructed, plan.points[finger].local_xyz_m(plan.edge_m), atol=1e-12
        )


def test_fourteen_variables_and_signed_orbit_preserve_cube(config):
    original_cube = copy.deepcopy(config["cube"])
    initial = PointTargetVariables.from_config(config)
    values = initial.as_array()
    assert values.shape == (VARIABLE_COUNT,) == (14,)
    values[:8] += np.linspace(-0.003, 0.003, 8)
    values[8:11] = (0.001, -0.002, 0.003)
    values[11:14] = np.radians((1.0, -1.5, 0.5))
    variables = PointTargetVariables.from_array(values)

    candidate = materialize_point_target_candidate(
        config, variables, signed_orbit_deg=-7.5
    )

    assert candidate["cube"] == original_cube
    assert [
        candidate["grasp_pose"]["nominal_joint_qpos_rad"][name]
        for name in ACTIVE_ACTUATORS
    ] == pytest.approx(values[:8])
    metadata = candidate["candidate_metadata"]["contact_point_target_search"]
    assert metadata["signed_orbit_deg"] == -7.5
    assert metadata["cube_pose_sampled"] is False
    assert metadata["hand_root_fixed_during_simulation"] is True
    assert metadata["point_plan_id"] == config["contact_point_plan"]["point_plan_id"]


@dataclass
class _Witness:
    finger: str
    target_face: str
    signed_gap_m: float
    normal_alignment: float
    cube_point_world_m: tuple[float, float, float]


@dataclass
class _Static:
    target_witnesses: tuple[_Witness, ...]
    static_geometry_pass: bool = True
    off_target_distal_penetrating_count: int = 0
    minimum_active_nondistal_gap_m: float = 0.001
    cube_freejoint_qpos_unchanged: bool = True
    nominal_minimum_forbidden_hand_gap_m: float = 0.001
    nominal_maximum_all_distal_penetration_m: float = 0.0001
    precontact_geometry_evaluated: bool = True
    precontact_minimum_hand_gap_m: float = 0.001

    def as_dict(self):
        return {
            "static_geometry_pass": self.static_geometry_pass,
            "target_witness": {
                witness.finger: {
                    "signed_gap_m": witness.signed_gap_m,
                    "normal_alignment": witness.normal_alignment,
                }
                for witness in self.target_witnesses
            },
            "retreat_evidence": {
                finger: {"closure_angle_deg": 10.0} for finger in ("thumb", "index", "mid")
            },
            "missing_target_witness_count": 0,
            "off_target_distal_penetrating_count": 0,
            "minimum_active_nondistal_gap_m": 0.001,
            "precontact_minimum_hand_gap_m": 0.001,
        }


def _static_at_plan(config, offset_yz=(0.0, 0.0)) -> _Static:
    plan = ContactPointPlan.from_config(config["contact_point_plan"])
    cube_position = _cube_world_position(config)
    cube_rotation = rpy_degrees_to_rotation_matrix(config["cube"]["rpy_deg"])
    points = contact_point_world_positions(
        plan,
        cube_world_position_m=cube_position,
        cube_world_rotation=cube_rotation,
    )
    witnesses = []
    for finger in ("thumb", "index", "mid"):
        local_offset = cube_rotation @ np.asarray((0.0, *offset_yz))
        witnesses.append(
            _Witness(
                finger,
                "X_NEG" if finger == "thumb" else "X_POS",
                0.00015,
                0.99,
                tuple(np.asarray(points[finger]) + local_offset),
            )
        )
    return _Static(tuple(witnesses))


def test_point_measurement_is_gap_plus_two_tangent_errors_plus_normals(config, policy):
    evaluation = point_target_trial_evaluation(
        _static_at_plan(config, (0.003, -0.001)), config
    )

    assert len(evaluation.measurement) == len(MEASUREMENT_NAMES) == 12
    assert evaluation.measurement[:3] == pytest.approx((0.00015,) * 3)
    assert evaluation.measurement[3:9] == pytest.approx((0.003, -0.001) * 3)
    assert evaluation.measurement[9:] == pytest.approx((0.99,) * 3)
    assert evaluation.point_distance_m == pytest.approx((math.sqrt(1e-5),) * 3)
    assert point_target_static_acceptance(evaluation, policy).passed

    outside = point_target_trial_evaluation(
        _static_at_plan(config, (0.004001, 0.0)), config
    )
    acceptance = point_target_static_acceptance(outside, policy)
    assert not acceptance.passed
    assert "point_target_radius_failed:thumb" in acceptance.reasons


def _joint_bounds():
    return {name: (-3.0, 3.0) for name in ACTIVE_ACTUATORS}


def test_boundary_accepts_negative_stratum_and_rejects_thumb_or_unregistered_orbit(
    config, policy
):
    variables = PointTargetVariables.from_config(config)
    assert not point_target_boundary_violations(
        config,
        variables,
        policy,
        signed_orbit_deg=-7.5,
        joint_bounds=_joint_bounds(),
        check_pose_constraints=False,
    )

    values = variables.as_array()
    values[0] = 1.61
    reasons = point_target_boundary_violations(
        config,
        PointTargetVariables.from_array(values),
        policy,
        signed_orbit_deg=-6.0,
        joint_bounds=_joint_bounds(),
        check_pose_constraints=False,
    )
    assert "thumb_actual_out_of_bounds" in reasons
    assert "signed_orbit_out_of_strata" in reasons


def test_injected_point_dls_uses_thumb_and_wrist_variables(config, policy):
    initial = PointTargetVariables.from_config(config).as_array()
    desired = initial.copy()
    desired[0] += 0.012
    desired[1] -= 0.008
    desired[8] += 0.001
    desired[11] += math.radians(0.4)
    target = np.asarray((0.00015,) * 3 + (0.0,) * 6 + (1.0,) * 3)
    coefficients = np.zeros((12, 14))
    coefficients[:, :12] = np.diag(
        (0.02,) * 3 + (0.08,) * 6 + (0.3,) * 3
    )
    # Make translation and wrist variables independently observable too.
    coefficients[3, 8] = 1.0
    coefficients[4, 11] = 0.2
    static = _static_at_plan(config)
    # Geometry pass is irrelevant to this injected numerical test and keeps
    # the solver from asking the synthetic record for a real retreat vector.
    static.static_geometry_pass = False

    def evaluator(candidate):
        variables = PointTargetVariables.from_config(candidate).as_array()
        measurement = target + coefficients @ (variables - desired)
        errors = measurement[3:9].reshape(3, 2)
        return PointTargetTrialEvaluation(
            static,
            tuple(measurement),
            tuple(tuple(row) for row in errors),
            tuple(np.linalg.norm(errors, axis=1)),
            (),
        )

    initial_eval = evaluator(config)
    initial_norm = np.linalg.norm((target - np.asarray(initial_eval.measurement)))
    result = solve_point_target_dls(
        config,
        signed_orbit_deg=0.0,
        policy=policy,
        settings=PointTargetDLSSettings(maximum_iterations=5),
        evaluator=evaluator,
        joint_bounds=_joint_bounds(),
        check_pose_constraints=False,
    )
    final_norm = np.linalg.norm(target - np.asarray(result.evaluation.measurement))

    assert final_norm < initial_norm
    assert result.variables.actual_joint_qpos_rad[0] != pytest.approx(initial[0])
    assert result.diagnostics["variable_names"][:8] == list(ACTIVE_ACTUATORS)
    assert result.diagnostics["signed_orbit_deg"] == 0.0


def test_registered_v12_config_compiles_real_witness_evaluator_and_dls(
    config, policy
):
    evaluator, bounds = build_point_target_trial_evaluator(config)
    variables = PointTargetVariables.from_config(config)
    assert not point_target_boundary_violations(
        config,
        variables,
        policy,
        signed_orbit_deg=0.0,
        joint_bounds=bounds,
    )
    evaluation = evaluator(config)
    assert evaluation.measurement is not None
    assert len(evaluation.measurement) == 12
    assert len(evaluation.point_distance_m) == 3
    # Schema v12 inherits the extended static-safety evidence introduced for
    # the orientation-aware v11 search.  The template itself may be an unsafe
    # near miss, but it must be rejected for measured geometry—not because the
    # evaluator silently omitted the required evidence fields.
    assert evaluation.static_result.nominal_minimum_forbidden_hand_gap_m is not None
    assert evaluation.static_result.nominal_maximum_all_distal_penetration_m is not None
    assert evaluation.static_result.precontact_geometry_evaluated is not None
    assert not any(
        reason.startswith("missing_") for reason in evaluation.safety_violations
    )

    result = solve_point_target_dls(
        config,
        signed_orbit_deg=0.0,
        policy=policy,
        settings=PointTargetDLSSettings(maximum_iterations=1),
    )
    assert result.diagnostics["point_plan_id"] == (
        config["contact_point_plan"]["point_plan_id"]
    )
    assert len(result.diagnostics["iterations"]) == 1


def test_static_record_is_dynamic_runner_compatible(config, policy):
    static = _static_at_plan(config)
    evaluation = point_target_trial_evaluation(static, config)
    acceptance = point_target_static_acceptance(evaluation, policy)
    from xhand_grasp.tuning.contact_point_targeted_search import PointTargetDLSResult

    materialized = materialize_point_target_candidate(
        config, PointTargetVariables.from_config(config), signed_orbit_deg=0.0
    )
    dls = PointTargetDLSResult(
        config=materialized,
        variables=PointTargetVariables.from_config(config),
        static_result=static,
        evaluation=evaluation,
        acceptance=acceptance,
        diagnostics={},
        stop_reason="contact_target_converged",
    )
    record = point_target_static_record(120000001, dls, source_id="source-a")

    assert {
        "candidate_id",
        "config",
        "candidate_sha256",
        "grasp_pose_id",
        "controller_id",
        "static_metrics",
        "static_pass",
    } <= set(record)
    assert record["candidate_sha256"] == canonical_sha256(record["config"])
    assert record["static_pass"]
    assert record["static_metrics"]["point_target"]["point_plan_id"] == (
        config["contact_point_plan"]["point_plan_id"]
    )


def test_plan_and_static_retain_are_order_independent_and_cover_signed_orbits():
    plans = [
        {
            "point_plan_id": f"plan-{index}",
            "reachable": True,
            "reachable_seed_count": index % 2 + 1,
            "minimum_safety_margin_m": 0.001 + index * 1e-5,
            "height_spread_m": 0.001,
            "line_of_action_moment_nm": 0.002,
            "minimum_edge_margin_m": 0.021,
        }
        for index in range(10)
    ]
    assert retain_ranked_point_plans(plans, top_k=4) == retain_ranked_point_plans(
        list(reversed(plans)), top_k=4
    )

    records = []
    for candidate_id, orbit in enumerate(SIGNED_ORBIT_DEG, start=1):
        records.append(
            {
                "candidate_id": candidate_id,
                "signed_orbit_deg": orbit,
                "static_pass": True,
                "static_rank": (False, candidate_id),
            }
        )
    retained = retain_point_target_static_candidates(
        list(reversed(records)), top_k=7
    )
    assert {value["signed_orbit_deg"] for value in retained} == set(
        SIGNED_ORBIT_DEG
    )
