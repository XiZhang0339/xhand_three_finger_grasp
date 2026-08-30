from __future__ import annotations

import math
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import xhand_grasp.tuning.relative_wrist_pose_active_set as active_set_module
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.tuning.relative_wrist_pose_active_set import (
    ActiveSetEvaluationContext,
    CachedFullScenePenetrationGate,
    build_active_set_evaluation_context,
    classify_full_scene_penetrations,
    evaluate_active_set_candidate,
    solve_orientation_aware_active_set_dls,
)
from xhand_grasp.tuning.relative_wrist_pose_search import (
    NON_THUMB_ACTUATORS,
    RelativeWristDLSSettings,
    RelativeWristTrialEvaluation,
    RelativeWristVariables,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift.json"
)


@pytest.fixture
def config() -> dict:
    return load_config(CONFIG)


def _bounds(config: dict) -> dict[str, tuple[float, float]]:
    nominal = config["grasp_pose"]["nominal_joint_qpos_rad"]
    return {
        name: (float(nominal[name]) - 0.5, float(nominal[name]) + 0.5)
        for name in NON_THUMB_ACTUATORS
    }


def _static(
    *,
    passed: bool = False,
    edge: float = 0.001,
    precontact: tuple[float, ...] | None = None,
):
    witnesses = tuple(
        SimpleNamespace(edge_margin_m=edge) for _ in range(3)
    )
    retreats = tuple(
        SimpleNamespace(
            measured_outward_retreat_m=0.003,
            closure_angle_deg=10.0,
            inward_speed_m_s=0.001,
        )
        for _ in range(3)
    )
    return SimpleNamespace(
        static_geometry_pass=passed,
        target_witnesses=witnesses,
        retreat_evidence=retreats,
        precontact_joint_qpos_rad=(
            tuple(0.0 for _ in ACTIVE_ACTUATORS)
            if precontact is None
            else precontact
        ),
    )


def _metadata(config: dict) -> tuple[np.ndarray, np.ndarray]:
    raw = config["candidate_metadata"]["relative_wrist_pose_search"]
    return (
        np.asarray(raw["root_delta_cube_m"], dtype=np.float64),
        np.radians(raw["wrist_local_rotvec_deg"]),
    )


def test_active_set_solver_prefers_central_differences(config):
    initial = RelativeWristVariables.from_config(config)
    desired = initial.non_thumb_joint_qpos_rad[0] + 0.02

    def evaluator(candidate):
        q = candidate["grasp_pose"]["nominal_joint_qpos_rad"]
        error = float(q[NON_THUMB_ACTUATORS[0]]) - desired
        measurement = (0.00015 + 0.02 * error, 0.00015, 0.00015, 0.0, 0.0, 1.0, 1.0, 1.0)
        return RelativeWristTrialEvaluation(_static(), measurement, ())

    result = solve_orientation_aware_active_set_dls(
        config,
        clockwise_orbit_deg=0.0,
        evaluator=evaluator,
        joint_bounds=_bounds(config),
        check_pose_constraints=False,
        settings=RelativeWristDLSSettings(maximum_iterations=2, regularization_weight=1e-5),
    )

    assert result.diagnostics["central_difference_count"] >= 13
    assert result.diagnostics["one_sided_difference_count"] == 0
    assert result.variables.non_thumb_joint_qpos_rad[0] > initial.non_thumb_joint_qpos_rad[0]


def test_unsafe_fd_uses_one_sided_fallback_and_is_never_accepted(config):
    initial = RelativeWristVariables.from_config(config)
    q0 = initial.non_thumb_joint_qpos_rad[0]

    def evaluator(candidate):
        q = float(candidate["grasp_pose"]["nominal_joint_qpos_rad"][NON_THUMB_ACTUATORS[0]])
        measurement = (0.00024 + 0.02 * (q - q0), 0.00015, 0.00015, 0.0, 0.0, 1.0, 1.0, 1.0)
        unsafe = ("nominal_forbidden_hand_penetration",) if q > q0 + 1e-12 else ()
        return RelativeWristTrialEvaluation(_static(), measurement, unsafe)

    result = solve_orientation_aware_active_set_dls(
        config,
        clockwise_orbit_deg=0.0,
        evaluator=evaluator,
        joint_bounds=_bounds(config),
        check_pose_constraints=False,
        settings=RelativeWristDLSSettings(maximum_iterations=1),
    )

    assert result.diagnostics["one_sided_difference_count"] >= 1
    assert result.diagnostics["final_safety_violations"] == []
    rejected = result.diagnostics["iterations"][0]["finite_difference_rejections"]
    assert any(item["reasons"] == ["nominal_forbidden_hand_penetration"] for item in rejected)


def test_box_active_set_freezes_an_outward_coordinate_and_resolves(config):
    bounds = _bounds(config)
    initial = RelativeWristVariables.from_config(config).as_array()
    first = NON_THUMB_ACTUATORS[0]
    initial[0] = bounds[first][0]
    variables = RelativeWristVariables.from_array(initial)

    def evaluator(candidate):
        q = float(candidate["grasp_pose"]["nominal_joint_qpos_rad"][first])
        # Increasing q increases an already-too-large gap, so the free solve
        # requests a negative (outward-at-lower-bound) step.
        measurement = (0.001 + 0.02 * (q - initial[0]), 0.00015, 0.00015, 0.0, 0.0, 1.0, 1.0, 1.0)
        return RelativeWristTrialEvaluation(_static(), measurement, ())

    result = solve_orientation_aware_active_set_dls(
        config,
        clockwise_orbit_deg=0.0,
        initial_variables=variables,
        evaluator=evaluator,
        joint_bounds=bounds,
        check_pose_constraints=False,
        settings=RelativeWristDLSSettings(maximum_iterations=1),
    )

    iteration = result.diagnostics["iterations"][0]
    assert first in iteration["active_box_variables"]
    assert result.variables.non_thumb_joint_qpos_rad[0] == pytest.approx(initial[0])


def test_rotvec_sphere_uses_tangent_and_feasible_alpha(config):
    initial = RelativeWristVariables.from_config(config).as_array()
    value = math.radians(8.0 / math.sqrt(2.0))
    initial[10:13] = (value, value, 0.0)
    variables = RelativeWristVariables.from_array(initial)

    def evaluator(candidate):
        _, rotvec = _metadata(candidate)
        # The height residual asks both x/y components to increase radially.
        radial = float(rotvec[0] + rotvec[1] - 2.0 * value)
        measurement = (0.00015, 0.00015, 0.00015, -0.004 + radial, -0.004 + radial, 1.0, 1.0, 1.0)
        return RelativeWristTrialEvaluation(_static(), measurement, ())

    result = solve_orientation_aware_active_set_dls(
        config,
        clockwise_orbit_deg=0.0,
        initial_variables=variables,
        evaluator=evaluator,
        joint_bounds=_bounds(config),
        check_pose_constraints=False,
        settings=RelativeWristDLSSettings(maximum_iterations=1),
    )

    iteration = result.diagnostics["iterations"][0]
    assert iteration["rotvec_tangent_projected"] is True
    assert iteration["maximum_feasible_alpha"] <= 1.0
    assert np.linalg.norm(result.variables.wrist_local_rotvec_rad) <= math.radians(8.0) + 1e-10


def test_hard_static_pass_has_lexicographic_priority(config):
    initial = RelativeWristVariables.from_config(config)
    q0 = initial.non_thumb_joint_qpos_rad[0]

    def evaluator(candidate):
        q = float(candidate["grasp_pose"]["nominal_joint_qpos_rad"][NON_THUMB_ACTUATORS[0]])
        offset = q - q0
        height = 0.0054 - 0.04 * offset
        passed = height <= 0.005
        # Once passing, deliberately make the within-band gap farther from its
        # midpoint.  Hard success must still outrank that smooth residual.
        gap = 0.00015 + (0.00009 if passed else 0.0)
        measurement = (gap, gap, gap, height, 0.0, 1.0, 1.0, 1.0)
        return RelativeWristTrialEvaluation(_static(passed=passed), measurement, ())

    result = solve_orientation_aware_active_set_dls(
        config,
        clockwise_orbit_deg=0.0,
        evaluator=evaluator,
        joint_bounds=_bounds(config),
        check_pose_constraints=False,
        settings=RelativeWristDLSSettings(maximum_iterations=4, regularization_weight=1e-6),
        promotion_validator=lambda candidate: None,
    )

    assert result.static_result.static_geometry_pass is True
    assert result.diagnostics["final_gate"]["hard_pass"] is True
    assert result.stop_reason in {"static_geometry_pass", "maximum_iterations"}


def test_static_pass_with_out_of_bounds_precontact_is_not_promoted_or_stopped(config):
    initial = RelativeWristVariables.from_config(config)
    base_index_bend = initial.non_thumb_joint_qpos_rad[2]

    def evaluator(candidate):
        nominal = candidate["grasp_pose"]["nominal_joint_qpos_rad"]
        precontact = [float(nominal[name]) for name in ACTIVE_ACTUATORS]
        # Reproduce the production failure: the Jacobian retreat crosses the
        # registered -0.15 rad precontact lower bound.
        current_index_bend = float(nominal[NON_THUMB_ACTUATORS[2]])
        precontact[3] = -0.1588 + (current_index_bend - base_index_bend)
        measurement = (0.00015, 0.00015, 0.00015, 0.0, 0.0, 1.0, 1.0, 1.0)
        return RelativeWristTrialEvaluation(
            _static(passed=True, precontact=tuple(precontact)), measurement, ()
        )

    result = solve_orientation_aware_active_set_dls(
        config,
        clockwise_orbit_deg=0.0,
        evaluator=evaluator,
        joint_bounds=_bounds(config),
        check_pose_constraints=False,
        settings=RelativeWristDLSSettings(
            maximum_iterations=1, regularization_weight=1e-6
        ),
    )

    assert result.diagnostics["iterations"], "invalid promotion must not stop early"
    assert result.stop_reason != "static_geometry_pass"
    assert result.diagnostics["promotion_config_valid"] is False
    assert any(
        "precontact" in error.lower()
        for error in result.diagnostics["promotion_validation_errors"]
    )
    assert (
        result.diagnostics["final_gate"][
            "precontact_target_normalized_violation"
        ]["left_hand_index_bend_joint_actuator"]
        > 0.0
    )


def test_synthetic_full_scene_gate_only_exempts_cube_support():
    contacts = (
        {"geom1_id": 70, "geom2_id": 2, "distance_m": -0.010},
        {
            "geom1_id": 20,
            "geom2_id": 31,
            "distance_m": -0.0047,
            "geom1_name": "index",
            "geom2_name": "middle",
        },
        {"geom1_id": 21, "geom2_id": 1, "distance_m": -0.002},
    )

    violations = classify_full_scene_penetrations(
        contacts, cube_geom_id=70, support_geom_id=2
    )

    assert len(violations) == 1
    assert violations[0]["geom1_name"] == "index"
    assert violations[0]["geom2_name"] == "middle"


def test_real_candidate_101000071001713_has_deep_index_middle_collision():
    report_path = (
        ROOT
        / "artifacts"
        / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
        "smooth_vertical_lift"
        / "tune"
        / "campaign_6d_r2"
        / "static"
        / "expanded"
        / "relative_wrist_orientation_refinement.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = next(
        value
        for value in payload["refined_candidates"]
        if int(value["candidate_id"]) == 101000071001713
    )
    metrics = candidate["static_metrics"]
    gate = CachedFullScenePenetrationGate(candidate["config"])

    safety = gate.evaluate(
        candidate["config"],
        nominal_joint_qpos_rad=metrics["nominal_joint_qpos_rad"],
        precontact_joint_qpos_rad=metrics["precontact_joint_qpos_rad"],
    )

    assert safety["passed"] is False
    assert safety["maximum_disallowed_penetration_m"] > 0.004
    assert any(
        "index" in value["geom1_name"] and "mid" in value["geom2_name"]
        for value in safety["violations"]
    )
    assert safety["phases"]["nominal"]["deepest_geom_pair"] is not None


def test_production_solver_reports_cached_full_scene_gate(config):
    result = solve_orientation_aware_active_set_dls(
        config,
        clockwise_orbit_deg=0.0,
        settings=RelativeWristDLSSettings(maximum_iterations=1),
    )

    initial = result.diagnostics["initial_full_scene_contact_safety"]
    final = result.diagnostics["full_scene_contact_safety"]
    assert initial["enabled"] is True
    assert set(initial["phases"]) == {"nominal", "precontact"}
    assert final["enabled"] is True
    assert "maximum_disallowed_penetration_m" in final


def test_shared_evaluation_context_compiles_once_for_multiple_solves(
    config, monkeypatch
):
    counts = {"base_compile": 0, "gate_compile": 0, "evaluate": 0, "gate": 0}
    initial = RelativeWristVariables.from_config(config)

    def evaluator(candidate):
        counts["evaluate"] += 1
        measurement = (0.0003, 0.00015, 0.00015, 0.0, 0.0, 1.0, 1.0, 1.0)
        nominal = candidate["grasp_pose"]["nominal_joint_qpos_rad"]
        precontact = tuple(float(nominal[name]) for name in ACTIVE_ACTUATORS)
        return RelativeWristTrialEvaluation(
            _static(precontact=precontact), measurement, ()
        )

    def fake_build(_config):
        counts["base_compile"] += 1
        return evaluator, _bounds(config)

    class FakeGate:
        def __init__(self, _config):
            counts["gate_compile"] += 1

        def evaluate(self, *_args, **_kwargs):
            counts["gate"] += 1
            return {"enabled": True, "passed": True, "phases": {}, "violations": []}

    monkeypatch.setattr(active_set_module, "build_actual_contact_trial_evaluator", fake_build)
    monkeypatch.setattr(active_set_module, "CachedFullScenePenetrationGate", FakeGate)
    context = build_active_set_evaluation_context(config)

    for _ in range(2):
        solve_orientation_aware_active_set_dls(
            config,
            clockwise_orbit_deg=0.0,
            initial_variables=initial,
            evaluation_context=context,
            check_pose_constraints=False,
            settings=RelativeWristDLSSettings(maximum_iterations=1),
        )

    assert counts["base_compile"] == 1
    assert counts["gate_compile"] == 1
    assert counts["evaluate"] > 2
    assert counts["gate"] == counts["evaluate"]


def test_evaluation_context_is_mutually_exclusive_with_bare_arguments(config):
    context = ActiveSetEvaluationContext(
        lambda candidate: RelativeWristTrialEvaluation(
            _static(), (0.00015,) * 3 + (0.0, 0.0) + (1.0,) * 3, ()
        ),
        _bounds(config),
        None,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        solve_orientation_aware_active_set_dls(
            config,
            clockwise_orbit_deg=0.0,
            evaluation_context=context,
            evaluator=context.evaluator,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        solve_orientation_aware_active_set_dls(
            config,
            clockwise_orbit_deg=0.0,
            evaluation_context=context,
            joint_bounds=_bounds(config),
        )


def test_context_full_scene_collision_rejects_fd_and_line_search(config):
    initial = RelativeWristVariables.from_config(config)
    first = NON_THUMB_ACTUATORS[0]
    q0 = initial.non_thumb_joint_qpos_rad[0]

    def evaluator(candidate):
        q = float(candidate["grasp_pose"]["nominal_joint_qpos_rad"][first])
        measurement = (0.001 - 0.02 * (q - q0), 0.00015, 0.00015, 0.0, 0.0, 1.0, 1.0, 1.0)
        nominal = candidate["grasp_pose"]["nominal_joint_qpos_rad"]
        precontact = tuple(float(nominal[name]) for name in ACTIVE_ACTUATORS)
        return RelativeWristTrialEvaluation(
            _static(precontact=precontact), measurement, ()
        )

    class CollisionGate:
        def evaluate(self, candidate, *, nominal_joint_qpos_rad, precontact_joint_qpos_rad):
            q = float(candidate["grasp_pose"]["nominal_joint_qpos_rad"][first])
            violation = {
                "phase": "nominal",
                "geom1_name": "index",
                "geom2_name": "middle",
                "penetration_m": 0.004,
            }
            return {
                "enabled": True,
                "passed": q <= q0 + 1e-12,
                "phases": {"nominal": {}},
                "violations": [] if q <= q0 + 1e-12 else [violation],
            }

    context = ActiveSetEvaluationContext(evaluator, _bounds(config), CollisionGate())
    result = solve_orientation_aware_active_set_dls(
        config,
        clockwise_orbit_deg=0.0,
        evaluation_context=context,
        check_pose_constraints=False,
        settings=RelativeWristDLSSettings(maximum_iterations=1),
    )

    rejected_reasons = [
        reason
        for trial in result.diagnostics["iterations"][0]["line_search"]
        for reason in trial["reasons"]
    ]
    assert any("full_scene_penetration:nominal:index|middle" in reason for reason in rejected_reasons)
    assert result.diagnostics["promotion_config_valid"] is False


def test_fresh_context_evaluation_reruns_base_and_gate(config):
    calls = {"base": 0, "gate": 0}

    def evaluator(candidate):
        calls["base"] += 1
        nominal = candidate["grasp_pose"]["nominal_joint_qpos_rad"]
        precontact = tuple(float(nominal[name]) for name in ACTIVE_ACTUATORS)
        return RelativeWristTrialEvaluation(
            _static(precontact=precontact),
            (0.00015,) * 3 + (0.0, 0.0) + (1.0,) * 3,
            (),
        )

    class CountingGate:
        def evaluate(self, *_args, **_kwargs):
            calls["gate"] += 1
            return {"enabled": True, "passed": True, "phases": {}, "violations": []}

    context = ActiveSetEvaluationContext(evaluator, _bounds(config), CountingGate())
    for _ in range(2):
        evaluation, report = evaluate_active_set_candidate(context, config)
        assert evaluation.safe
        assert report["passed"]
    assert calls == {"base": 2, "gate": 2}


def test_coordinate_pattern_polish_recovers_when_dls_step_is_zero(config):
    initial = RelativeWristVariables.from_config(config)
    first = NON_THUMB_ACTUATORS[0]
    q0 = initial.non_thumb_joint_qpos_rad[0]

    def evaluator(candidate):
        nominal = candidate["grasp_pose"]["nominal_joint_qpos_rad"]
        q = float(nominal[first])
        passed = q >= q0 + 0.0004 - 1e-12
        gap = 0.00015 if passed else 0.001
        precontact = tuple(float(nominal[name]) for name in ACTIVE_ACTUATORS)
        return RelativeWristTrialEvaluation(
            _static(passed=passed, precontact=precontact),
            (gap, 0.00015, 0.00015, 0.0, 0.0, 1.0, 1.0, 1.0),
            (),
        )

    result = solve_orientation_aware_active_set_dls(
        config,
        clockwise_orbit_deg=0.0,
        evaluator=evaluator,
        joint_bounds=_bounds(config),
        check_pose_constraints=False,
        settings=RelativeWristDLSSettings(maximum_iterations=1),
    )

    iteration = result.diagnostics["iterations"][0]
    polish = iteration["coordinate_pattern_polish"]
    assert iteration["accepted_source"] == "coordinate_pattern_polish"
    assert polish["executed"] is True
    assert polish["accepted"] is True
    assert polish["hard_pass_early_stop"] is True
    assert result.diagnostics["promotion_config_valid"] is True
    assert result.variables.non_thumb_joint_qpos_rad[0] == pytest.approx(q0 + 0.0004)


def test_real_89mm_candidate_coordinate_polish_passes_at_thumb_rota1_plus_1p6mrad(
    monkeypatch,
):
    report_path = (
        ROOT
        / "artifacts"
        / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
        "smooth_vertical_lift"
        / "tune"
        / "active_set_recovery_v2_from_campaign_6d_r2"
        / "static"
        / "active_set_batch_2.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    record = next(
        value
        for value in payload["candidates"]
        if int(value["candidate_id"]) == 8110000000200049
    )
    candidate = record["config"]
    orbit = float(record["clockwise_orbit_deg"])
    thumb_rota1 = "left_hand_thumb_rota_joint1_actuator"
    initial = RelativeWristVariables.from_config(candidate).as_array()
    # The deterministic continuation has already taken the first +4h step;
    # this solve verifies that the fallback takes the second +4h step and the
    # total +1.6 mrad pose is freshly promotable and full-scene safe.
    initial[0] += 0.0008
    monkeypatch.setattr(
        active_set_module,
        "_active_set_step",
        lambda *_args: (np.zeros(13), (), False),
    )
    context = build_active_set_evaluation_context(candidate)

    result = solve_orientation_aware_active_set_dls(
        candidate,
        clockwise_orbit_deg=orbit,
        initial_variables=RelativeWristVariables.from_array(initial),
        evaluation_context=context,
        settings=RelativeWristDLSSettings(maximum_iterations=1),
    )

    original_q = candidate["grasp_pose"]["nominal_joint_qpos_rad"][thumb_rota1]
    final_q = result.config["grasp_pose"]["nominal_joint_qpos_rad"][thumb_rota1]
    iteration = result.diagnostics["iterations"][0]
    polish = iteration["coordinate_pattern_polish"]
    assert final_q - original_q == pytest.approx(0.0016)
    assert result.diagnostics["promotion_config_valid"] is True
    assert result.diagnostics["full_scene_contact_safety"]["passed"] is True
    assert iteration["accepted_source"] == "coordinate_pattern_polish"
    assert polish["hard_pass_early_stop"] is True
    selected = polish["trials"][polish["best_trial_index"]]
    assert selected["variable"] == thumb_rota1
    assert selected["delta"] == pytest.approx(0.0008)
