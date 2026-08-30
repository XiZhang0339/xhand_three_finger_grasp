from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from xhand_grasp.actual_contact_capability import (
    is_joint_pair_near_zero_contact_preserving_planned_lift_definition,
    resolve_joint_pair_near_zero_contact_preserving_planned_lift_definition,
)
from xhand_grasp.config import validate_config
from xhand_grasp.experiment import get_experiment, resolve_experiment
from xhand_grasp.experiments.opposed_face_palm_down_joint_pair_near_zero_contact_preserving_planned_lift import (
    EXPERIMENT_ID as V15_EXPERIMENT_ID,
    JOINT_PAIR_FEEDBACK as V15_FEEDBACK,
)
from xhand_grasp.experiments.opposed_face_palm_down_joint_pair_near_zero_rolling_slip_contact_preserving_planned_lift import (
    EXPERIMENT_DEFINITION,
    EXPERIMENT_ID,
    JOINT_PAIR_FEEDBACK,
)
from xhand_grasp.v16_identity import (
    V16_TOP_LEVEL_ID_FIELDS,
    install_v16_top_level_identities,
)


V15_CONFIG = Path(
    "grasp_configs/left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift.json"
)


def _resolved_v16_config() -> dict[str, object]:
    config = json.loads(V15_CONFIG.read_text(encoding="utf-8"))
    config["schema_version"] = 16
    config["experiment_id"] = EXPERIMENT_ID
    config["description"] = EXPERIMENT_DEFINITION.description
    config["control_protocol"] = (
        EXPERIMENT_DEFINITION.control_protocol.as_config()
    )
    config["actual_contact_grasp_pose_campaign"] = (
        EXPERIMENT_DEFINITION.actual_contact_grasp_pose_campaign.as_config()
    )
    config["contact_preserving_planned_lift_campaign"] = (
        EXPERIMENT_DEFINITION.contact_preserving_planned_lift_campaign.as_config()
    )
    config["joint_pair_feedback"] = JOINT_PAIR_FEEDBACK.as_config()
    config["candidate_metadata"]["schema_version"] = 16
    install_v16_top_level_identities(config)
    return config


def test_v16_definition_is_registered_and_schema_is_fail_closed() -> None:
    definition = get_experiment(EXPERIMENT_ID)
    assert definition is EXPERIMENT_DEFINITION
    assert definition.joint_pair_feedback is JOINT_PAIR_FEEDBACK
    assert definition.joint_pair_feedback.schema_version == 2
    assert definition.control_protocol.strategy == (
        "grasp_verify_then_joint_pair_aligned_rolling_slip_"
        "contact_preserving_planned_lift"
    )
    assert resolve_experiment(
        {"schema_version": 16, "experiment_id": EXPERIMENT_ID}
    ) is definition
    assert is_joint_pair_near_zero_contact_preserving_planned_lift_definition(
        definition
    )
    assert resolve_joint_pair_near_zero_contact_preserving_planned_lift_definition(
        {"schema_version": 16, "experiment_id": EXPERIMENT_ID}
    ) is definition

    with pytest.raises(ValueError, match="schema_version 15 requires"):
        resolve_experiment(
            {"schema_version": 15, "experiment_id": EXPERIMENT_ID}
        )
    with pytest.raises(ValueError, match="schema_version 16 requires"):
        resolve_experiment(
            {"schema_version": 16, "experiment_id": V15_EXPERIMENT_ID}
        )


def test_v16_feedback_is_rolling_aware_without_changing_v15_serialization() -> None:
    v15_fields = set(V15_FEEDBACK.as_config())
    v16 = JOINT_PAIR_FEEDBACK.as_config()
    rolling_fields = {
        "slip_resume_threshold_m",
        "slip_recovery_enter_threshold_m",
        "slip_recovery_exit_threshold_m",
        "slip_exit_dwell_s",
        "relative_velocity_filter_time_constant_s",
        "tangent_prediction_horizon_s",
        "suppress_outward_force_pi_during_recovery",
        "maximum_patch_match_distance_m",
        "centroid_jump_diagnostic_threshold_m",
        "maximum_contact_time_gap_s",
    }
    assert v15_fields.isdisjoint(rolling_fields)
    assert rolling_fields <= set(v16)
    assert (
        v16["slip_recovery_exit_threshold_m"]
        < v16["slip_recovery_enter_threshold_m"]
        < v16["slip_resume_threshold_m"]
        < v16["slip_freeze_threshold_m"]
        < v16["slip_abort_threshold_m"]
    )
    assert v16["suppress_outward_force_pi_during_recovery"] is True


def test_v16_config_and_identity_chain_validate_and_bind_rolling_contract() -> None:
    config = _resolved_v16_config()
    validate_config(config)
    assert all(
        isinstance(config[field], str) and len(config[field]) == 64
        for field in V16_TOP_LEVEL_ID_FIELDS
    )

    changed = replace(
        JOINT_PAIR_FEEDBACK,
        maximum_patch_match_distance_m=0.0025,
    )
    tampered = _resolved_v16_config()
    old_controller_id = tampered["controller_id"]
    tampered["joint_pair_feedback"] = changed.as_config()
    install_v16_top_level_identities(tampered)
    assert tampered["controller_id"] != old_controller_id
    with pytest.raises(ValueError, match="fixed contract"):
        validate_config(tampered)

    tampered = _resolved_v16_config()
    tampered["controller_id"] = "0" * 64
    with pytest.raises(ValueError, match="controller_id does not match"):
        validate_config(tampered)
