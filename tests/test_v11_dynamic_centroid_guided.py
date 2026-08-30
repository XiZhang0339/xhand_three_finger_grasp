from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xhand_grasp.config import load_config
from xhand_grasp.artifacts import file_sha256
from xhand_grasp.experiment import resolve_experiment
from xhand_grasp.grasp_pose import grasp_pose_id
from xhand_grasp.tuning import relative_wrist_pose_dynamic_guided as guided_module
from xhand_grasp.tuning.relative_wrist_pose_dynamic_guided import (
    RESPONSE_DIMENSION,
    VECTOR_DIMENSION,
    DynamicContactObservation,
    GuidedRefinementPolicy,
    authenticate_dynamic_record,
    build_dynamic_centroid_candidate_records,
    deduplicate_guided_candidates,
    deterministic_bounded_proposals,
    deterministic_guided_candidate_id,
    dynamic_centroid_recovery_rank,
    encode_pose_control_vector,
    extract_compacted_dynamic_contact_observation,
    extract_signed_dynamic_contact_observation,
    fit_ridge_svd,
    generate_controller_balance_proposals,
    hard_normalized_dynamic_merit,
    run_dynamic_contact_centroid_guided_refinement,
    select_dynamic_centroid_parents,
)
from xhand_grasp.tuning.relative_wrist_pose_search import NON_THUMB_ACTUATORS


ROOT = Path(__file__).resolve().parents[1]
V11_CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift.json"
)
DYNAMIC_ROOT = (
    ROOT
    / "artifacts"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift"
    / "tune"
    / "active_set_coordinate_recovery_v3_from_campaign_6d_r2"
    / "dynamic"
)
AUDITED_SUCCESS_CONFIG = (
    ROOT
    / "artifacts"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift"
    / "tune"
    / "power_recovery_diagnostic_20260826"
    / "reconstructed_first_grasp_config.json"
)


def _real_dynamic_records() -> list[dict]:
    records: list[dict] = []
    for name in (
        "recovery_report.json",
        "recovery_active_set_local_local_refinement_report.json",
    ):
        records.extend(json.loads((DYNAMIC_ROOT / name).read_text())["candidate_records"])
    return records


def _real_observations() -> tuple[DynamicContactObservation, ...]:
    observations: list[DynamicContactObservation] = []
    for record in _real_dynamic_records():
        authenticated = authenticate_dynamic_record(record, DYNAMIC_ROOT)
        if authenticated["trace_retained"]:
            with np.load(authenticated["trace_path"]) as trace:
                observation = extract_signed_dynamic_contact_observation(
                    authenticated["config"],
                    authenticated["result"],
                    trace,
                    artifact_key=authenticated["artifact_key"],
                )
        else:
            observation = extract_compacted_dynamic_contact_observation(
                authenticated["config"],
                authenticated["result"],
                artifact_key=authenticated["artifact_key"],
            )
        observations.append(observation)
    return tuple(observations)


def _observation(
    candidate_id: int,
    *,
    artifact_key: str | None = None,
    edge_m: float = 0.089,
    height: float = 0.007,
    translation: float = 0.0002,
    success: bool = False,
) -> DynamicContactObservation:
    return DynamicContactObservation(
        artifact_key=artifact_key or f"artifact-{candidate_id}",
        candidate_id=candidate_id,
        edge_m=edge_m,
        clockwise_orbit_deg=0.0,
        grasp_success=success,
        effective_finger_count=3,
        simultaneous_target_face_duty=1.0,
        signed_height_error_m=(-height, -height),
        height_spread_p95_m=height,
        cube_translation_m=(translation, 0.0, 0.0),
        cube_orientation_rotvec_rad=(0.0, 0.0, 0.0),
        thumb_actual_range_duty=1.0,
        force_log_ratio=(0.0, 0.0),
        first_contact_step=(1800, 1900, 1850),
        maximum_consecutive_gate_steps=0,
        trace_sample_count=750,
    )


def _real_candidate(candidate_id: int) -> tuple[dict, dict, Path]:
    directory = DYNAMIC_ROOT / "candidates" / f"candidate_{candidate_id}"
    return (
        json.loads((directory / "resolved_config.json").read_text()),
        json.loads((directory / "result.json").read_text()),
        directory / "trace.npz",
    )


def test_real_trace_extracts_signed_thumb_high_centroid_error():
    config, result, trace_path = _real_candidate(3232000000000708)
    with np.load(trace_path) as trace:
        observation = extract_signed_dynamic_contact_observation(
            config, result, trace, artifact_key="real-0708"
        )

    assert observation.trace_sample_count == 750
    assert observation.signed_height_error_m[0] == pytest.approx(-0.00787169, abs=2e-6)
    assert observation.signed_height_error_m[1] == pytest.approx(-0.00831161, abs=2e-6)
    assert observation.height_spread_p95_m == pytest.approx(0.0083146678)
    assert observation.artifact_key == "real-0708"


def test_authentication_binds_config_result_trace_and_artifact_key():
    report = json.loads(
        (DYNAMIC_ROOT / "recovery_active_set_local_local_refinement_report.json").read_text()
    )
    record = next(
        item for item in report["candidate_records"]
        if int(item["candidate_id"]) == 3232000000000708
    )
    authenticated = authenticate_dynamic_record(record, DYNAMIC_ROOT)

    assert authenticated["trace_retained"] is True
    assert Path(authenticated["trace_path"]).is_file()
    assert len(authenticated["artifact_key"]) == 64
    assert authenticated["candidate_id"] == 3232000000000708


def test_pose_control_vector_is_fixed_order_and_complete():
    config, _, _ = _real_candidate(3232000000000708)
    vector = encode_pose_control_vector(config)

    assert vector.shape == (VECTOR_DIMENSION,) == (28,)
    assert np.isfinite(vector).all()
    np.testing.assert_allclose(
        vector[:13],
        __import__(
            "xhand_grasp.tuning.relative_wrist_pose_search",
            fromlist=["RelativeWristVariables"],
        ).RelativeWristVariables.from_config(config).as_array(),
    )


def test_hard_merit_prioritizes_success_and_pose_safety():
    success = _observation(3, success=True, height=0.004)
    safe = _observation(2, height=0.006, translation=0.0002)
    unsafe = _observation(1, height=0.006, translation=0.001)

    assert hard_normalized_dynamic_merit(success) < hard_normalized_dynamic_merit(safe)
    assert dynamic_centroid_recovery_rank(safe) < dynamic_centroid_recovery_rank(unsafe)


def test_ridge_svd_recovers_deterministic_linear_response():
    rng = np.random.default_rng(20260821)
    x = rng.normal(size=(40, VECTOR_DIMENSION))
    truth = rng.normal(size=(VECTOR_DIMENSION, RESPONSE_DIMENSION))
    y = x @ truth
    first = fit_ridge_svd(x, y, ridge=1e-8, scale=np.ones(VECTOR_DIMENSION))
    second = fit_ridge_svd(x, y, ridge=1e-8, scale=np.ones(VECTOR_DIMENSION))

    np.testing.assert_allclose(first.coefficient, second.coefficient)
    np.testing.assert_allclose(first.predict(x[0]), y[0], atol=2e-7)


def test_bounded_proposals_are_deterministic_unique_and_inside_trust():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(50, VECTOR_DIMENSION))
    y = rng.normal(size=(50, RESPONSE_DIMENSION))
    model = fit_ridge_svd(x, y)
    center = np.zeros(VECTOR_DIMENSION)
    lower = np.full(VECTOR_DIMENSION, -0.1)
    upper = np.full(VECTOR_DIMENSION, 0.1)
    trust = np.full(VECTOR_DIMENSION, 0.01)

    first = deterministic_bounded_proposals(
        center, model, lower=lower, upper=upper, trust_radius=trust
    )
    second = deterministic_bounded_proposals(
        center, model, lower=lower, upper=upper, trust_radius=trust
    )

    assert 1 <= len(first) <= 7
    assert len({tuple(value) for value in first}) == len(first)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
        assert np.max(np.abs(a - center)) <= 0.01 + 1e-12


def test_parent_selection_uses_artifact_key_and_round_robin_edges():
    observations = (
        _observation(1, artifact_key="a", edge_m=0.089, height=0.006),
        _observation(1, artifact_key="b", edge_m=0.090, height=0.007),
        _observation(2, artifact_key="c", edge_m=0.089, height=0.008),
        _observation(3, artifact_key="a", edge_m=0.089, height=0.004),
    )
    selected = select_dynamic_centroid_parents(observations, top_count=3)

    assert [item.artifact_key for item in selected] == ["a", "b", "c"]
    assert [item.edge_m for item in selected[:2]] == [0.089, 0.090]


def test_real_v3_parent_selection_covers_pose_height_and_thumb_evidence():
    selected = select_dynamic_centroid_parents(_real_observations(), top_count=3)

    assert [item.candidate_id for item in selected] == [
        3232000000000674,  # best hard-ranked pose-safe trace
        3232000000001331,  # <=5 mm height spread, lowest maximum pose drift
        3232000000001126,  # full thumb-range duty, best composite merit
    ]
    height_parent = selected[1]
    assert height_parent.trace_sample_count == 0  # authenticated compacted evidence
    assert height_parent.height_spread_p95_m == pytest.approx(0.0026822083766192438)
    assert height_parent.maximum_translation_m == pytest.approx(0.0018268920787726554)


def test_candidate_id_binds_artifact_key_and_is_repeatable():
    first = deterministic_guided_candidate_id(7, 1, 2, "pose", parent_artifact_key="a")
    second = deterministic_guided_candidate_id(7, 1, 2, "pose", parent_artifact_key="a")
    other = deterministic_guided_candidate_id(7, 1, 2, "pose", parent_artifact_key="b")
    assert first == second
    assert first != other


def test_controller_balance_freezes_grasp_pose_and_is_deterministic():
    config, result, trace_path = _real_candidate(3232000000000708)
    with np.load(trace_path) as trace:
        observation = extract_signed_dynamic_contact_observation(config, result, trace)
    proposals = generate_controller_balance_proposals(config, observation, count=8)

    assert len(proposals) == 8
    for proposal in proposals:
        assert proposal["grasp_pose"] == config["grasp_pose"]
        assert proposal["hand_pose"] == config["hand_pose"]
        assert proposal["cube"] == config["cube"]
    assert proposals[-1]["control"]["contact_preload_targets_rad"] != config["control"]["contact_preload_targets_rad"]


def test_controller_balance_contains_measured_index_scale_rescue_ladder():
    config, result, _ = _real_candidate(3232000000001331)
    observation = _observation(
        int(result["candidate_id"]), artifact_key="measured-grasp-lock-geometry"
    )
    first = generate_controller_balance_proposals(config, observation, count=8)
    second = generate_controller_balance_proposals(config, observation, count=8)
    nominal = config["grasp_pose"]["nominal_joint_qpos_rad"]
    precontact = config["control"]["precontact_targets_rad"]
    index_names = (
        "left_hand_index_bend_joint_actuator",
        "left_hand_index_joint1_actuator",
        "left_hand_index_joint2_actuator",
    )

    assert len(first) == len(second) == 8
    for proposal, repeated, scale in zip(
        first[:5], second[:5], (0.0, 0.25, 0.5, 0.75, 1.0)
    ):
        assert proposal == repeated
        targets = proposal["control"]["contact_preload_targets_rad"]
        assert targets["left_hand_thumb_bend_joint_actuator"] == pytest.approx(1.405)
        for name in index_names:
            assert targets[name] == pytest.approx(
                nominal[name] + scale * (nominal[name] - precontact[name])
            )
        assert proposal["control_protocol"]["close_s"] == pytest.approx(1.5)
        assert proposal["control"]["close_profile"][index_names[0]][
            "start_fraction"
        ] == pytest.approx(0.6289062500000001)


def test_real_1331_scale_quarter_reconstructs_authenticated_success_controls():
    directory = DYNAMIC_ROOT / "candidates" / "candidate_3232000000001331"
    config = json.loads((directory / "resolved_config.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    observation = extract_compacted_dynamic_contact_observation(
        config, result, artifact_key="real-compacted-1331"
    )
    proposal = generate_controller_balance_proposals(config, observation, count=8)[1]
    audited = json.loads(AUDITED_SUCCESS_CONFIG.read_text())

    assert file_sha256(AUDITED_SUCCESS_CONFIG) == (
        "29c0bdbd24f6ab7489e23a42ff65093e024cb36dbff80fe109a0658a69557219"
    )
    assert proposal["control"] == audited["control"]
    assert proposal["control_protocol"] == audited["control_protocol"]
    for key in set(audited) - {"candidate_metadata", "control", "control_protocol"}:
        assert proposal[key] == audited[key]


def test_guided_runner_executes_real_1331_scale_quarter_controller(
    tmp_path, monkeypatch
):
    wanted_ids = {
        3232000000000674,
        3232000000001126,
        3232000000001331,
    }
    dynamic_records = [
        record
        for record in _real_dynamic_records()
        if int(record["candidate_id"]) in wanted_ids
    ]
    source_config, _, _ = _real_candidate(3232000000001331)
    definition = resolve_experiment(source_config)
    context = SimpleNamespace(
        joint_bounds={
            name: definition.search_bounds.actuator_targets_rad[name]
            for name in NON_THUMB_ACTUATORS
        }
    )
    monkeypatch.setattr(
        guided_module,
        "deterministic_bounded_proposals",
        lambda *args, **kwargs: (),
    )
    static_result = SimpleNamespace(static_geometry_pass=True)
    monkeypatch.setattr(
        guided_module,
        "evaluate_active_set_candidate",
        lambda *args, **kwargs: (
            SimpleNamespace(safe=True, static_result=static_result),
            {"passed": True},
        ),
    )
    monkeypatch.setattr(
        guided_module,
        "apply_precontact_solution",
        lambda config, static: copy.deepcopy(config),
    )
    calls: list[tuple[str, tuple[dict, ...]]] = []

    def executor(records, output, *, stage, workers):
        del output, workers
        materialized = tuple(copy.deepcopy(dict(record)) for record in records)
        calls.append((stage, materialized))
        return SimpleNamespace(
            records=materialized,
            artifacts=(),
            summary={"grasp_success_count": 0},
        )

    run_dynamic_contact_centroid_guided_refinement(
        dynamic_records,
        {"config": source_config},
        DYNAMIC_ROOT,
        tmp_path,
        workers=2,
        seed=20260821,
        evaluation_context=context,
        dynamic_executor=executor,
    )

    assert [stage for stage, _ in calls] == [
        "dynamic_centroid_guided_round0",
        "dynamic_centroid_guided_controller_balance",
    ]
    audited = json.loads(AUDITED_SUCCESS_CONFIG.read_text())
    executed_1331 = [
        record
        for _, records in calls
        for record in records
        if int(record["source_candidate_id"]) == 3232000000001331
    ]
    assert len(executed_1331) == 8
    assert any(
        record["config"]["control"] == audited["control"]
        and record["config"]["control_protocol"] == audited["control_protocol"]
        for record in executed_1331
    )


def test_runner_ready_candidate_records_have_complete_identity():
    config = load_config(V11_CONFIG)
    second = copy.deepcopy(config)
    second["control"]["contact_preload_targets_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] += 0.001
    records = build_dynamic_centroid_candidate_records(
        (config, second),
        parent_candidate_id=17,
        parent_artifact_key="artifact-17",
        round_index=2,
        kind="controller",
        stage="test",
    )

    assert len(records) == 2  # controller variants share pose but remain distinct
    record = records[0]
    assert set(
        (
            "candidate_id",
            "source_candidate_id",
            "controller_seed_index",
            "grasp_pose_id",
            "controller_id",
            "candidate_sha256",
            "config",
        )
    ).issubset(record)
    assert record["grasp_pose_id"] == grasp_pose_id(record["config"])


def test_guided_policy_enforces_total_budget():
    GuidedRefinementPolicy()
    with pytest.raises(ValueError, match="sub-budgets"):
        GuidedRefinementPolicy(maximum_dynamic_budget=98)
