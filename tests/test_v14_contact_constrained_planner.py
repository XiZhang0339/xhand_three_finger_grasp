from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, load_config
from xhand_grasp.experiment import ManipulationPlanParameters
from xhand_grasp.trajectory import (
    interpolate_quintic_c2,
    quintic_c2_knot_derivatives,
)
from xhand_grasp.tuning import contact_constrained_planner as planner


ROOT = Path(__file__).resolve().parents[1]
V13_TEMPLATE = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_scaled_centered_spread_"
    "actual_grasp_then_lift.json"
)
V14_TEMPLATE = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _config() -> dict:
    return load_config(V13_TEMPLATE)


def _synthetic_probe_samples(
    *,
    unsafe_negative_first_axis: bool = False,
) -> tuple[planner.ExtendedProbeSample, ...]:
    object_bias = np.zeros(6, dtype=np.float64)
    force_bias = np.asarray((0.10, 0.11, 0.12), dtype=np.float64)
    slip_bias = np.asarray((0.0002, 0.0003, 0.0004), dtype=np.float64)
    matrix = np.zeros((planner.RESPONSE_DIMENSION, len(ACTIVE_ACTUATORS)))
    matrix[2] = np.linspace(0.10, 0.03, len(ACTIVE_ACTUATORS))
    matrix[0, 1] = 0.01
    matrix[6:9] = np.asarray(
        (
            (0.20, 0.02, 0.01, 0.00, 0.01, 0.00, 0.00, 0.00),
            (0.00, 0.00, 0.00, 0.03, 0.04, 0.02, 0.00, 0.00),
            (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04, 0.05),
        )
    )
    matrix[9:12] = 0.0001 * np.arange(24, dtype=np.float64).reshape(3, 8)
    bias = np.concatenate((object_bias, force_bias, slip_bias))

    def sample(
        kind: str,
        actuator: str | None,
        direction: int,
        delta: np.ndarray,
        *,
        valid: bool = True,
    ) -> planner.ExtendedProbeSample:
        response = bias + matrix @ delta
        return planner.ExtendedProbeSample(
            kind=kind,
            actuator=actuator,
            direction=direction,
            applied_delta_rad=delta,
            object_response_6d=response[:6],
            target_normal_force_n=response[6:9],
            target_contact_valid=np.full(3, valid, dtype=bool),
            tangent_slip_m=response[9:12],
            checkpoint_step_index=250,
        )

    zero = np.zeros(len(ACTIVE_ACTUATORS), dtype=np.float64)
    result = [sample("zero", None, 0, zero)]
    for column, actuator in enumerate(ACTIVE_ACTUATORS):
        for direction in (-1, 1):
            delta = zero.copy()
            delta[column] = 0.02 * direction
            valid = not (
                unsafe_negative_first_axis and column == 0 and direction == -1
            )
            result.append(
                sample(
                    "single_actuator",
                    actuator,
                    direction,
                    delta,
                    valid=valid,
                )
            )
    return tuple(result)


def _simple_response() -> planner.ExtendedProbeResponse:
    samples = list(_synthetic_probe_samples())
    # Make the local object model particularly transparent: the first active
    # axis alone can create the requested 11 mm upward displacement, while all
    # three contact forces and slips remain constant and safely bounded.
    response = planner.fit_extended_probe_response(samples)
    object_jacobian = np.zeros_like(response.object_jacobian_6x8)
    object_jacobian[2, 0] = 0.20
    return planner.ExtendedProbeResponse(
        object_bias_6d=np.zeros(6),
        force_bias_n=np.full(3, 0.10),
        slip_bias_m=np.full(3, 0.0002),
        object_jacobian_6x8=object_jacobian,
        force_jacobian_3x8=np.zeros((3, 8)),
        slip_jacobian_3x8=np.zeros((3, 8)),
        available_actuator_mask=np.ones(8, dtype=bool),
        zero_contact_valid=np.ones(3, dtype=bool),
        zero_forbidden_contact=False,
        zero_active_nondistal_contact=False,
        column_evidence=response.column_evidence,
        probe_evidence=response.probe_evidence,
    )


def _bounds() -> dict[str, tuple[float, float]]:
    return {name: (-0.20, 0.20) for name in ACTIVE_ACTUATORS}


def test_extended_response_uses_safe_one_sided_column_and_is_order_stable() -> None:
    samples = _synthetic_probe_samples(unsafe_negative_first_axis=True)
    forward = planner.fit_extended_probe_response(samples)
    reverse = planner.fit_extended_probe_response(tuple(reversed(samples)))
    assert forward.response_model_id == reverse.response_model_id
    assert forward.column_evidence[0]["method"] == "forward_contact_safe"
    assert forward.column_evidence[0]["negative_contact_safe"] is False
    assert np.all(forward.available_actuator_mask)
    np.testing.assert_allclose(
        forward.object_jacobian_6x8,
        reverse.object_jacobian_6x8,
        atol=0.0,
        rtol=0.0,
    )
    prediction = forward.predict(np.zeros(8))
    np.testing.assert_allclose(prediction["target_normal_force_n"], (0.10, 0.11, 0.12))


def test_extended_response_drops_axis_when_both_probe_sides_break_contact() -> None:
    samples = list(_synthetic_probe_samples())
    for index in (1, 2):
        original = samples[index]
        samples[index] = planner.ExtendedProbeSample(
            kind=original.kind,
            actuator=original.actuator,
            direction=original.direction,
            applied_delta_rad=original.applied_delta_rad,
            object_response_6d=original.object_response_6d,
            target_normal_force_n=original.target_normal_force_n,
            target_contact_valid=np.zeros(3, dtype=bool),
            tangent_slip_m=original.tangent_slip_m,
            checkpoint_step_index=original.checkpoint_step_index,
        )
    response = planner.fit_extended_probe_response(samples)
    assert response.available_actuator_mask[0] is np.False_ or not bool(
        response.available_actuator_mask[0]
    )
    assert response.column_evidence[0]["method"] == "unavailable_contact_unsafe"
    np.testing.assert_array_equal(response.object_jacobian_6x8[:, 0], 0.0)


def test_checkpoint_probe_collection_preserves_requested_specs_and_source_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zeros = np.zeros(8, dtype=np.float64)
    specs = [
        planner.ProbeSpecification("zero", None, 0, tuple(zeros), tuple(zeros))
    ]
    for column, actuator in enumerate(ACTIVE_ACTUATORS):
        for direction in (-1, 1):
            delta = zeros.copy()
            delta[column] = 0.02 * direction
            specs.append(
                planner.ProbeSpecification(
                    "single_actuator",
                    actuator,
                    direction,
                    tuple(delta),
                    tuple(delta),
                )
            )
    monkeypatch.setattr(
        planner, "generate_probe_specifications", lambda *_args, **_kwargs: tuple(specs)
    )
    grasp = SimpleNamespace(
        model=object(),
        config={},
        checkpoint=SimpleNamespace(step_index=250),
    )

    def runner(_grasp: object, specification: planner.ProbeSpecification) -> dict:
        return {
            "probe": specification.as_mapping(),
            "response_6d": [0.0] * 6,
            "checkpoint_step_index": 250,
            "contact_evidence": {
                "target_normal_force_n": [0.1, 0.1, 0.1],
                "target_contact_valid": [True, True, True],
                "tangent_slip_m": [0.0, 0.0, 0.0],
                "forbidden_contact": False,
                "active_nondistal_contact": False,
            },
        }

    observed = planner.collect_extended_probe_samples(grasp, runner)
    assert len(observed) == 17
    assert observed[0].kind == "zero"
    assert all(value.checkpoint_step_index == 250 for value in observed)

    def wrong_step(_grasp: object, specification: planner.ProbeSpecification) -> dict:
        result = runner(_grasp, specification)
        result["checkpoint_step_index"] = 251
        return result

    with pytest.raises(ValueError, match="supplied checkpoint"):
        planner.collect_extended_probe_samples(grasp, wrong_step)


def test_bounded_active_set_and_linear_projection_are_deterministic() -> None:
    first = planner.solve_bounded_projected_least_squares(
        np.eye(2),
        (2.0, -2.0),
        (-1.0, -1.0),
        (1.0, 1.0),
        ridge=0.0,
    )
    np.testing.assert_allclose(first.solution, (1.0, -1.0), atol=1e-12)
    assert first.active_lower == (1,)
    assert first.active_upper == (0,)

    projected = planner.solve_bounded_projected_least_squares(
        np.eye(2),
        (0.0, 0.0),
        (-2.0, -2.0),
        (2.0, 2.0),
        ridge=0.0,
        linear_matrix=((1.0, 1.0),),
        linear_lower=(1.0,),
        linear_upper=(np.inf,),
    )
    replay = planner.solve_bounded_projected_least_squares(
        np.eye(2),
        (0.0, 0.0),
        (-2.0, -2.0),
        (2.0, 2.0),
        ridge=0.0,
        linear_matrix=((1.0, 1.0),),
        linear_lower=(1.0,),
        linear_upper=(np.inf,),
    )
    assert projected.linear_constraints_feasible
    assert projected.max_linear_violation <= 1e-9
    assert float(np.sum(projected.solution)) >= 1.0 - 1e-9
    np.testing.assert_array_equal(projected.solution, replay.solution)

    infeasible = planner.solve_bounded_projected_least_squares(
        ((1.0,),),
        (0.0,),
        (0.0,),
        (0.0,),
        linear_matrix=((1.0,),),
        linear_lower=(1.0,),
        linear_upper=(np.inf,),
    )
    assert not infeasible.linear_constraints_feasible
    assert infeasible.max_linear_violation == pytest.approx(1.0)


def test_planner_generates_four_20_segment_21_knot_attempts_and_interpolates() -> None:
    settings = planner.ContactConstrainedPlannerSettings(
        target_normal_force_n=(0.10, 0.10, 0.10),
        minimum_normal_force_n=(0.05, 0.05, 0.05),
        maximum_tangent_slip_m=(0.005, 0.005, 0.005),
    )
    report = planner.plan_contact_constrained_trajectory(
        _config(), _simple_response(), _bounds(), settings=settings
    )
    assert planner.PLAN_SEGMENT_COUNT == 20
    assert planner.KNOT_COUNT == 21
    assert len(report.attempts) == 4
    assert [value.trust_region_scale for value in report.attempts] == [
        1.0,
        0.5,
        0.25,
        0.125,
    ]
    for attempt in report.attempts:
        assert attempt.command_delta_rad.shape == (21, 8)
        assert attempt.contact_feasible
        assert np.max(np.abs(np.diff(attempt.command_delta_rad, axis=0))) <= (
            0.04 * attempt.trust_region_scale + 1e-12
        )
        np.testing.assert_allclose(attempt.command_delta_rad[0], 0.0)
    plan = report.selected_plan
    midpoint = 0.5 * (plan.knot_fraction[6] + plan.knot_fraction[7])
    observed = planner.interpolate_plan_at_fraction(plan, midpoint)
    knot_times = plan.knot_fraction * plan.duration_s
    velocities, accelerations = quintic_c2_knot_derivatives(
        knot_times, plan.command_delta_rad
    )
    expected, _, _, _ = interpolate_quintic_c2(
        knot_times,
        plan.command_delta_rad,
        midpoint * plan.duration_s,
        knot_velocities=velocities,
        knot_accelerations=accelerations,
    )
    np.testing.assert_allclose(
        [observed["command_delta_rad"][name] for name in ACTIVE_ACTUATORS],
        expected,
    )
    assert observed["bracketing_knot_indices"] == [6, 7]
    assert all(observed["predicted_contact_valid"].values())
    quarter = plan.knot_fraction[6] + 0.25 * (
        plan.knot_fraction[7] - plan.knot_fraction[6]
    )
    quarter_observed = planner.interpolate_plan_at_fraction(plan, float(quarter))
    assert quarter_observed["local_quintic_parameter"] == pytest.approx(0.25)
    assert quarter_observed["interpolation_profile"] == (
        "shared_clamped_c4_quintic"
    )
    quarter_expected, _, _, _ = interpolate_quintic_c2(
        knot_times,
        plan.command_delta_rad,
        float(quarter * plan.duration_s),
        knot_velocities=velocities,
        knot_accelerations=accelerations,
    )
    np.testing.assert_allclose(
        [
            quarter_observed["command_delta_rad"][name]
            for name in ACTIVE_ACTUATORS
        ],
        quarter_expected,
        atol=0.0,
        rtol=0.0,
    )
    runtime_plan = ManipulationPlanParameters.from_config(
        plan.as_manipulation_plan_config()
    )
    assert runtime_plan.plan_id == plan.plan_id
    assert len(runtime_plan.knot_times_s) == 21
    with pytest.raises(ValueError, match="20 segments"):
        planner.ContactConstrainedPlannerSettings(knot_count=20)


def test_continuous_audit_rejects_between_knot_force_overshoot() -> None:
    times = np.linspace(0.0, 1.0, 4)
    commands = np.zeros((4, len(ACTIVE_ACTUATORS)), dtype=np.float64)
    # Every knot clears the 50.5 mN floor.  The exact shared spline dips below
    # it between the first pair, which a knot-only planner would miss.
    force_profile = np.asarray((0.051, 0.200, 0.051, 0.200))
    forces = np.repeat(force_profile[:, None], len(ACTIVE_FINGERS), axis=1)
    slips = np.zeros_like(forces)
    observed = planner._continuous_path_validity(
        times,
        commands,
        forces,
        slips,
        command_lower_rad=np.full(len(ACTIVE_ACTUATORS), -1.0),
        command_upper_rad=np.full(len(ACTIVE_ACTUATORS), 1.0),
        minimum_force_n=np.full(len(ACTIVE_FINGERS), 0.0505),
        maximum_slip_m=np.full(len(ACTIVE_FINGERS), 0.005),
        tolerance=1e-12,
    )
    assert not np.all(observed)
    assert not np.all(observed[0:2])

    safe = planner._continuous_path_validity(
        times,
        commands,
        np.full_like(forces, 0.10),
        slips,
        command_lower_rad=np.full(len(ACTIVE_ACTUATORS), -1.0),
        command_upper_rad=np.full(len(ACTIVE_ACTUATORS), 1.0),
        minimum_force_n=np.full(len(ACTIVE_FINGERS), 0.0505),
        maximum_slip_m=np.full(len(ACTIVE_FINGERS), 0.005),
        tolerance=1e-12,
    )
    assert np.all(safe)


def test_v14_rank_puts_full_hard_pass_before_near_miss_contact_quality() -> None:
    maintained = {
        "candidate_id": 20,
        "summary": {"stage_status": {"full_success": False}},
        "contact_maintenance": {
            "satisfied": True,
            "contact_loss_count": 0,
            "minimum_valid_duty": 1.0,
            "minimum_force_margin_n": 0.02,
            "maximum_tangent_slip_m": 0.001,
        },
        "path_tracking": {"rms_error": 0.010, "terminal_error": 0.010},
        "objective": 10.0,
    }
    lost = {
        "candidate_id": 1,
        "summary": {"stage_status": {"full_success": True}},
        "contact_maintenance": {
            "satisfied": False,
            "contact_loss_count": 1,
            "minimum_valid_duty": 0.99,
            "minimum_force_margin_n": 0.10,
            "maximum_tangent_slip_m": 0.0,
        },
        "path_tracking": {"rms_error": 0.0, "terminal_error": 0.0},
        "objective": 0.0,
    }
    equal_but_lower_id = copy.deepcopy(maintained)
    equal_but_lower_id["candidate_id"] = 10
    serial = planner.rank_contact_constrained_candidates(
        (lost, maintained, equal_but_lower_id)
    )
    reversed_input = planner.rank_contact_constrained_candidates(
        (equal_but_lower_id, maintained, lost)
    )
    assert [value["candidate_id"] for value in serial] == [1, 10, 20]
    assert [value["candidate_id"] for value in reversed_input] == [1, 10, 20]


def test_v14_rank_uses_perturbation_passes_before_contact_margins() -> None:
    lower_robustness = {
        "candidate_id": 1,
        "full_success": True,
        "perturbation_pass_count": 13,
        "contact_maintenance": {
            "satisfied": True,
            "contact_loss_count": 0,
            "minimum_valid_duty": 1.0,
            "minimum_force_margin_n": 1.0,
            "maximum_tangent_slip_m": 0.0,
        },
    }
    higher_robustness = copy.deepcopy(lower_robustness)
    higher_robustness.update(candidate_id=2, perturbation_pass_count=15)
    higher_robustness["contact_maintenance"]["minimum_force_margin_n"] = 0.01
    ranked = planner.rank_contact_constrained_candidates(
        (lower_robustness, higher_robustness)
    )
    assert [value["candidate_id"] for value in ranked] == [2, 1]


def test_contact_first_rank_reads_raw_schema_v14_evaluator_metrics() -> None:
    checks = {
        name: True
        for name in (
            "v14_operation_did_not_abort",
            "v14_thumb_contact_duty_at_least_99_percent",
            "v14_index_contact_duty_at_least_99_percent",
            "v14_middle_contact_duty_at_least_99_percent",
            "v14_simultaneous_contact_duty_at_least_99_percent",
            "v14_thumb_contact_loss_within_limit",
            "v14_index_contact_loss_within_limit",
            "v14_middle_contact_loss_within_limit",
            "v14_simultaneous_contact_loss_within_limit",
            "active_nondistal_contacts_within_limit",
        )
    }
    record = {
        "candidate_id": 7,
        "summary": {
            "passed": True,
            "stage_status": {"full_success": True},
            "checks": checks,
            "metrics": {
                "forbidden_contact_steps": 0,
                "contact_preserving_planned_lift": {
                    "target_face_effective_duty": {
                        "thumb": 1.0,
                        "index": 0.995,
                        "mid": 0.999,
                    },
                    "simultaneous_target_face_effective_duty": 0.994,
                    "longest_contact_loss_steps": {
                        "thumb": 0,
                        "index": 2,
                        "mid": 1,
                    },
                    "simultaneous_longest_contact_loss_steps": 2,
                    "allowed_contact_loss_steps": 10,
                    "required_contact_duty": 0.99,
                    "operation_aborted": False,
                },
            },
        },
    }
    evidence = planner.contact_first_rank_evidence(record)
    assert evidence["contact_constraints_satisfied"] is True
    assert evidence["minimum_valid_duty"] == pytest.approx(0.994)
    assert evidence["contact_loss_count"] == 2
    assert evidence["forbidden_contact"] is False
    assert evidence["active_nondistal_contact"] is False
    assert json.dumps(evidence, allow_nan=False)


def test_identity_domains_separate_object_grasp_pair_planner_and_controller() -> None:
    config = _config()
    response = _simple_response()
    baseline_settings = planner.ContactConstrainedPlannerSettings(
        target_normal_force_n=(0.10, 0.10, 0.10)
    )
    baseline = planner.plan_contact_constrained_trajectory(
        config, response, _bounds(), settings=baseline_settings
    )
    identities = baseline.identities

    changed_mass = copy.deepcopy(config)
    changed_mass["cube"]["mass_kg"] *= 1.1
    assert planner.v14_object_config_id(changed_mass) != identities.object_config_id
    assert planner.v14_grasp_pose_id(changed_mass) == identities.grasp_pose_id
    assert (
        planner.v14_grasp_object_pair_id(changed_mass)
        != identities.grasp_object_pair_id
    )

    changed_grasp = copy.deepcopy(config)
    actuator = ACTIVE_ACTUATORS[-1]
    changed_grasp["grasp_pose"]["nominal_joint_qpos_rad"][actuator] += 0.001
    assert planner.v14_object_config_id(changed_grasp) == identities.object_config_id
    assert planner.v14_grasp_pose_id(changed_grasp) != identities.grasp_pose_id
    assert (
        planner.v14_grasp_object_pair_id(changed_grasp)
        != identities.grasp_object_pair_id
    )

    changed_settings = planner.ContactConstrainedPlannerSettings(
        target_normal_force_n=(0.11, 0.10, 0.10)
    )
    assert planner.v14_planner_id(changed_settings) != identities.planner_id
    changed = planner.plan_contact_constrained_trajectory(
        config, response, _bounds(), settings=changed_settings
    )
    assert changed.identities.controller_id != identities.controller_id

    changed_feedback = copy.deepcopy(config)
    changed_feedback["contact_feedback"] = {
        "feedback_id": "diagnostic-feedback",
        "kp_rad_per_n": {"thumb": 0.01, "index": 0.02, "mid": 0.03},
    }
    assert (
        planner.v14_controller_id(
            changed_feedback,
            baseline_settings,
            response,
            baseline.selected_plan,
        )
        != identities.controller_id
    )
    assert (
        planner.v14_grasp_object_pair_id(changed_feedback)
        == identities.grasp_object_pair_id
    )


def test_report_is_json_serializable_and_binds_selected_plan() -> None:
    report = planner.plan_contact_constrained_trajectory(
        _config(),
        _simple_response(),
        _bounds(),
        settings=planner.ContactConstrainedPlannerSettings(
            target_normal_force_n=(0.10, 0.10, 0.10)
        ),
    )
    payload = report.as_mapping()
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert encoded
    assert payload["search_evidence_only"] is True
    assert payload["final_success_requires_full_reset_rerun"] is True
    assert payload["selected_plan_id"] == report.selected_plan.plan_id
    assert payload["identities"]["plan_id"] == report.selected_plan.plan_id
    assert len(payload["report_id"]) == 64


def test_materialized_plan_updates_terminal_mirror_and_validates_as_schema_v14() -> None:
    config = load_config(V14_TEMPLATE)
    raw_targets = config["contact_force_targets_n"]["per_finger_n"]
    settings = planner.ContactConstrainedPlannerSettings(
        target_normal_force_n=tuple(
            float(raw_targets[finger]) for finger in ACTIVE_FINGERS
        )
    )
    report = planner.plan_contact_constrained_trajectory(
        config, _simple_response(), _bounds(), settings=settings
    )
    resolved = planner.materialize_contact_plan_config(
        config, report.selected_plan, validate=True
    )
    terminal = report.selected_plan.command_delta_rad[-1]
    np.testing.assert_allclose(
        [
            resolved["control"]["manipulation_delta_rad"][name]
            for name in ACTIVE_ACTUATORS
        ],
        terminal,
    )
    assert (
        resolved["manipulation_plan"]["plan_id"]
        == report.selected_plan.plan_id
    )
