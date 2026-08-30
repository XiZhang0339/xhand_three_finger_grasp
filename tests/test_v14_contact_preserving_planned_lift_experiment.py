from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from xhand_grasp.config import load_config, validate_config
from xhand_grasp.actual_contact_capability import (
    ACTUAL_CONTACT_SCHEMA_VERSIONS,
    resolve_actual_contact_definition,
)
from xhand_grasp.experiment import (
    ACTIVE_ACTUATORS,
    ContactFeedbackParameters,
    ContactForceTargets,
    ManipulationPlanParameters,
    get_experiment,
)
from xhand_grasp.scene import build_model
from xhand_grasp.v14_identity import (
    V14_TOP_LEVEL_ID_FIELDS,
    v14_base_controller_id,
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
    v14_sequential_planner_id,
    validate_v14_top_level_identities,
)
from xhand_grasp.experiments.opposed_face_palm_down_contact_preserving_planned_lift import (
    CAMPAIGN,
    DEFAULT_CONTACT_FEEDBACK,
    DEFAULT_CONTACT_FORCE_TARGETS_N,
    DEFAULT_MANIPULATION_PLAN,
    EDGES_M,
    EXPERIMENT_ID,
    SOURCE_GRASP_CATALOG,
    SOURCE_GRASP_CATALOG_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)
V13_TEMPLATE = REPO_ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift.json"
)
V14_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / (
    "left_opposed_face_palm_down_contact_preserving_planned_lift"
)


def _template() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _resolved_identity_config() -> dict:
    config = _template()
    config["object_config_id"] = v14_object_config_id(config)
    config["grasp_pose_id"] = v14_grasp_pose_id(config)
    config["grasp_object_pair_id"] = v14_grasp_object_pair_id(config)
    report_id = hashlib.sha256(b"v14 identity test report").hexdigest()
    attempt_id = hashlib.sha256(b"v14 identity test attempt").hexdigest()
    config.setdefault("candidate_metadata", {})[
        "sequential_checkpoint_planning"
    ] = {
        "report_id": report_id,
        "attempt_report_id": attempt_id,
    }
    config["planner_id"] = v14_sequential_planner_id(report_id, attempt_id)
    config["controller_id"] = v14_base_controller_id(
        config, bind_planner=False
    )
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v14_template_loads_and_registered_contract_is_exact() -> None:
    config = load_config(TEMPLATE)
    definition = get_experiment(EXPERIMENT_ID)

    assert config["schema_version"] == 14
    assert config["experiment_id"] == EXPERIMENT_ID
    assert definition.contact_preserving_planned_lift_campaign is CAMPAIGN
    assert 14 in ACTUAL_CONTACT_SCHEMA_VERSIONS
    assert resolve_actual_contact_definition(config) is definition
    assert definition.tuning_strategy == "contact_preserving_planned_lift"
    assert config["control_protocol"]["strategy"] == (
        "grasp_verify_then_contact_preserving_planned_lift"
    )
    assert EDGES_M == tuple(value / 1000.0 for value in range(60, 89))
    assert config["cube"]["edge_m"] in EDGES_M
    assert config["cube"]["mass_kg"] == pytest.approx(0.160)
    assert config["cube"]["friction"] == pytest.approx(0.8)
    assert config["search"]["budget"] == CAMPAIGN.budget_config()
    assert config["contact_preserving_planned_lift_campaign"] == (
        CAMPAIGN.as_config()
    )
    assert CAMPAIGN.plan_candidates_per_pair == 4
    assert CAMPAIGN.feedback_plan_candidates_per_pair == 4
    assert CAMPAIGN.maximum_plan_candidate_count == 64
    assert config["actual_contact_grasp_pose_campaign"]["budget"][
        "manipulation_candidates_per_pose"
    ] == 4
    assert config["acceptance"]["finger_contact_duty"] == pytest.approx(0.99)
    assert config["acceptance"]["simultaneous_contact_duty"] == pytest.approx(
        0.99
    )
    assert config["contact_alignment"]["operation_aligned_duty"] == pytest.approx(
        0.70
    )


def test_v14_top_level_identity_states_are_explicit() -> None:
    template = _template()
    assert validate_v14_top_level_identities(template)["state"] == (
        "unresolved_template"
    )

    # This is the sole partial-ID compatibility state: the campaign needs a
    # pair identity before an offline planner/controller has been selected.
    source_pair = copy.deepcopy(template)
    source_pair["object_config_id"] = v14_object_config_id(source_pair)
    source_pair["grasp_pose_id"] = v14_grasp_pose_id(source_pair)
    source_pair["grasp_object_pair_id"] = v14_grasp_object_pair_id(source_pair)
    assert validate_v14_top_level_identities(source_pair)["state"] == (
        "preplanning_source_pair"
    )

    partial = copy.deepcopy(template)
    partial["object_config_id"] = v14_object_config_id(partial)
    with pytest.raises(ValueError, match="search intermediate"):
        validate_v14_top_level_identities(partial)

    resolved = _resolved_identity_config()
    assert validate_v14_top_level_identities(resolved)["state"] == "resolved"
    validate_config(resolved)


@pytest.mark.parametrize("field", V14_TOP_LEVEL_ID_FIELDS)
def test_v14_rejects_each_stale_top_level_identity(field: str) -> None:
    config = _resolved_identity_config()
    config[field] = "0" * 64
    with pytest.raises(ValueError, match=field.split("_id")[0]):
        validate_config(config)


@pytest.mark.parametrize(
    ("relative_path", "planner_verification"),
    (
        (
            "tune/formal_campaign_v2/candidates/"
            "candidate_14036697251896653/resolved_config.json",
            "sequential_checkpoint_planning",
        ),
        (
            "tune/formal_campaign_v14_1_rescue_v2/catalogs/target_1/"
            "grasp_pose/pair_rank_05_14034704765090975/resolved_config.json",
            "legacy_controller_bound_lineage",
        ),
        (
            "tune/formal_campaign_v14_1_adaptive_event_rescue_v2/"
            "catalogs/target_1/grasp_pose/"
            "pair_rank_01_14883962737215073/resolved_config.json",
            "legacy_controller_bound_lineage",
        ),
        (
            "tune/formal_campaign_v14_1_force_debias_rescue_v2/"
            "force_debias_refinement/candidates/"
            "candidate_14983474579738994/resolved_config.json",
            "force_debias_rescue",
        ),
        (
            "tune/formal_campaign_v14_1_micro_jerk_rescue_recovery_v2/"
            "catalogs/target_1/grasp_pose/"
            "pair_rank_01_15191687524351858/resolved_config.json",
            "bounded_micro_jerk_rescue",
        ),
        (
            "tune/formal_campaign_v14_1_contact_mode_pose_rescue_"
            "recovery_v1/top5_current_source_reruns/"
            "candidate_15630100717072254/resolved_config.json",
            "contact_mode_pose_rescue",
        ),
        (
            "tune/formal_campaign_v14_1_adaptive_pose_followup_v1/"
            "catalogs/target_1/grasp_pose/"
            "pair_rank_05_15774899044824772/resolved_config.json",
            "adaptive_pose_followup",
        ),
    ),
)
def test_v14_sealed_resolved_identity_families_remain_valid(
    relative_path: str, planner_verification: str
) -> None:
    path = V14_ARTIFACT_ROOT / relative_path
    if not path.is_file():
        pytest.skip("sealed v14 campaign artifacts are not installed")
    config = json.loads(path.read_text(encoding="utf-8"))
    audit = validate_v14_top_level_identities(config)
    assert audit["state"] == "resolved"
    assert audit["planner_verification"] == planner_verification


def test_v14_legacy_warm_exception_is_exact_and_cannot_spread() -> None:
    path = V14_ARTIFACT_ROOT / (
        "tune/warm_start_sweep_v1/candidates/"
        "candidate_14040766880179603/resolved_config.json"
    )
    if not path.is_file():
        pytest.skip("sealed v14 warm-start artifact is not installed")
    config = json.loads(path.read_text(encoding="utf-8"))
    audit = validate_v14_top_level_identities(config)
    assert audit["planner_verification"] == "legacy_lineage_unavailable"

    # A same-shaped arbitrary config is not allowed to inherit the exception.
    changed = copy.deepcopy(config)
    changed.setdefault("candidate_metadata", {})["unregistered_change"] = True
    with pytest.raises(ValueError, match="explicitly registered legacy warm"):
        validate_v14_top_level_identities(changed)


def test_v14_typed_blocks_round_trip_and_bind_canonical_ids() -> None:
    config = _template()
    plan = ManipulationPlanParameters.from_config(config["manipulation_plan"])
    targets = ContactForceTargets.from_config(config["contact_force_targets_n"])
    feedback = ContactFeedbackParameters.from_config(config["contact_feedback"])

    assert plan == DEFAULT_MANIPULATION_PLAN
    assert targets == DEFAULT_CONTACT_FORCE_TARGETS_N
    assert feedback == DEFAULT_CONTACT_FEEDBACK
    assert len(plan.knot_times_s) == 21
    assert plan.knot_times_s[0] == pytest.approx(0.0)
    assert plan.knot_times_s[-1] == pytest.approx(3.0)
    assert plan.desired_cube_position_delta_m[-1] == pytest.approx(
        (0.0, 0.0, 0.011)
    )
    assert set(plan.actuator_waypoints_rad) == set(ACTIVE_ACTUATORS)
    for actuator in ACTIVE_ACTUATORS:
        assert plan.actuator_waypoints_rad[actuator][-1] == pytest.approx(
            config["control"]["manipulation_delta_rad"][actuator]
        )
    assert targets.source == "verify_window_median_clamped"
    assert feedback.operation_contact_duty_min == pytest.approx(0.99)
    assert feedback.max_loss_s == pytest.approx(0.010)
    CAMPAIGN.validate_candidate(plan, targets, feedback)


def test_v14_optional_operation_force_scale_preserves_legacy_identity() -> None:
    config = _template()
    raw = copy.deepcopy(config["contact_force_targets_n"])
    assert "operation_scale" not in raw
    assert raw["target_id"] == (
        "20d7b483108c7b2e123cc7231847521a10a11e2c5bba913f07d84461f2625ec8"
    )
    legacy = ContactForceTargets.from_config(raw)
    assert legacy.operation_scale == pytest.approx(1.0)
    assert legacy.as_config() == raw

    scaled = ContactForceTargets(
        schema_version=legacy.schema_version,
        source=legacy.source,
        minimum_n=legacy.minimum_n,
        maximum_n=legacy.maximum_n,
        per_finger_n=legacy.per_finger_n,
        operation_scale=0.70,
    )
    scaled_config = scaled.as_config()
    assert scaled_config["operation_scale"] == pytest.approx(0.70)
    assert scaled_config["target_id"] != raw["target_id"]
    assert ContactForceTargets.from_config(scaled_config) == scaled
    config["contact_force_targets_n"] = scaled_config
    validate_config(config)

    explicit_default = copy.deepcopy(raw)
    explicit_default["operation_scale"] = 1.0
    with pytest.raises(ValueError, match="legacy omitted form"):
        ContactForceTargets.from_config(explicit_default)
    for invalid in (0.699, 1.001):
        with pytest.raises(ValueError, match=r"\[0.70, 1.00\]"):
            ContactForceTargets(
                schema_version=legacy.schema_version,
                source=legacy.source,
                minimum_n=legacy.minimum_n,
                maximum_n=legacy.maximum_n,
                per_finger_n=legacy.per_finger_n,
                operation_scale=invalid,
            )


@pytest.mark.parametrize(
    ("block", "id_field", "message"),
    (
        ("manipulation_plan", "plan_id", "plan_id"),
        ("contact_force_targets_n", "target_id", "target_id"),
        ("contact_feedback", "feedback_id", "feedback_id"),
    ),
)
def test_v14_rejects_tampered_controller_identity(
    block: str, id_field: str, message: str
) -> None:
    config = _template()
    config[block][id_field] = "0" * 64
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_v14_rejects_plan_terminal_or_contact_acceptance_drift() -> None:
    terminal = _template()
    terminal["control"]["manipulation_delta_rad"][
        "left_hand_index_joint2_actuator"
    ] += 1e-4
    with pytest.raises(ValueError, match="terminal waypoint"):
        validate_config(terminal)

    feedback = _template()
    feedback["contact_feedback"]["max_loss_s"] = 0.02
    parsed = ContactFeedbackParameters.from_config(
        {
            **feedback["contact_feedback"],
            "feedback_id": ContactFeedbackParameters(
                schema_version=1,
                strategy="per_finger_force_pi",
                filter_time_constant_s=0.005,
                kp_rad_per_n=feedback["contact_feedback"]["kp_rad_per_n"],
                ki_rad_per_n_s=feedback["contact_feedback"]["ki_rad_per_n_s"],
                integral_limit_n_s=0.5,
                correction_limit_rad=0.06,
                rate_limit_rad_s=0.6,
                acceleration_limit_rad_s2=6.0,
                force_risk_n=0.1,
                freeze_on_risk=True,
                max_loss_s=0.02,
                recovery_behavior="freeze_and_inward_preload_then_abort",
                operation_contact_duty_min=0.99,
            ).feedback_id,
        }
    )
    feedback["contact_feedback"] = parsed.as_config()
    with pytest.raises(ValueError, match="loss window"):
        validate_config(feedback)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("edge_m", 0.059, "registered campaign grid"),
        ("mass_kg", 0.161, "mass must match"),
        ("friction", 0.81, "friction must match"),
    ),
)
def test_v14_rejects_cube_values_outside_registered_campaign(
    field: str, value: float, message: str
) -> None:
    config = _template()
    config["cube"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_v14_source_catalog_is_authenticated_and_v13_stays_loadable() -> None:
    assert _sha256(REPO_ROOT / SOURCE_GRASP_CATALOG) == (
        SOURCE_GRASP_CATALOG_SHA256
    )
    source = json.loads(
        (REPO_ROOT / SOURCE_GRASP_CATALOG).read_text(encoding="utf-8")
    )
    assert source["success_count"] == CAMPAIGN.expected_source_grasp_count
    assert all(item["grasp_success"] for item in source["trajectories"])

    legacy = load_config(V13_TEMPLATE)
    assert legacy["schema_version"] == 13
    assert "manipulation_plan" not in legacy
    assert "contact_feedback" not in legacy


def test_v14_blocks_are_required_and_rejected_by_legacy_schema() -> None:
    missing = _template()
    missing.pop("contact_feedback")
    with pytest.raises(ValueError, match="contact_feedback"):
        validate_config(missing)

    legacy = _template()
    legacy["schema_version"] = 13
    legacy["experiment_id"] = json.loads(
        V13_TEMPLATE.read_text(encoding="utf-8")
    )["experiment_id"]
    with pytest.raises(ValueError):
        validate_config(legacy)


def test_v14_registered_actuator_search_bounds_fit_real_ctrl_and_joint_limits() -> None:
    config = load_config(TEMPLATE)
    definition = get_experiment(EXPERIMENT_ID)
    model, _ = build_model(config)
    assert set(definition.search_bounds.actuator_targets_rad) == set(
        ACTIVE_ACTUATORS
    )
    for name, bounds in definition.search_bounds.actuator_targets_rad.items():
        actuator_id = model.actuator(name).id
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        effective_lower = max(
            float(model.actuator_ctrlrange[actuator_id, 0]),
            float(model.jnt_range[joint_id, 0]),
        )
        effective_upper = min(
            float(model.actuator_ctrlrange[actuator_id, 1]),
            float(model.jnt_range[joint_id, 1]),
        )
        assert float(bounds[0]) >= effective_lower
        assert float(bounds[1]) <= effective_upper
