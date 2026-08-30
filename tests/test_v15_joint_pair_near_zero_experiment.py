from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from xhand_grasp.config import load_config, validate_config
from xhand_grasp.experiment import (
    ACTIVE_ACTUATORS,
    JointPairFeedbackParameters,
    ManipulationPlanParameters,
    get_experiment,
    resolve_experiment,
)
from xhand_grasp.experiments.opposed_face_palm_down_joint_pair_near_zero_contact_preserving_planned_lift import (
    BUDGET,
    CAMPAIGN,
    EXPERIMENT_ID,
    JOINT_PAIR_ALIGNMENT,
    JOINT_PAIR_FEEDBACK,
    JOINT_PAIR_NEAR_ZERO_CONTACT_PRESERVING_PLANNED_LIFT,
)
from xhand_grasp.v15_identity import (
    V15_TOP_LEVEL_ID_FIELDS,
    install_v15_top_level_identities,
)


CONFIG_PATH = Path(
    "grasp_configs/left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift.json"
)
SOURCE_CONFIG_PATH = Path(
    "artifacts/left_opposed_face_palm_down_joint_pair_aligned_"
    "contact_preserving_planned_lift/tune/"
    "index_middle_joint1_near_zero_alignment_refinement_v2/final/"
    "candidate_15941124607131459/resolved_config.json"
)
SEALED_V14_CONFIGS = {
    Path(
        "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
    ): "0c7cf2dd9c26976435dd7a8a6d825a70f2a17abcc91bf3f94a81be095d6b69e2",
    Path(
        "grasp_configs/left_opposed_face_palm_down_joint_pair_aligned_"
        "contact_preserving_planned_lift.json"
    ): "effdfa2fc6a4866391ce43c8d4f92d930fcdd0958e40cbe70af72cf5f14161a0",
}


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_v15_definition_is_registered_with_an_independent_fixed_contract() -> None:
    definition = get_experiment(EXPERIMENT_ID)

    assert definition is JOINT_PAIR_NEAR_ZERO_CONTACT_PRESERVING_PLANNED_LIFT
    assert resolve_experiment({"schema_version": 15, "experiment_id": EXPERIMENT_ID}) is definition
    assert definition.tuning_strategy == (
        "joint_pair_near_zero_contact_preserving_planned_lift"
    )
    assert definition.artifact_root == (
        "artifacts/left_opposed_face_palm_down_joint_pair_near_zero_"
        "contact_preserving_planned_lift"
    )
    assert definition.joint_pair_alignment is JOINT_PAIR_ALIGNMENT
    assert definition.joint_pair_feedback is JOINT_PAIR_FEEDBACK
    assert definition.control_protocol is not None
    assert definition.control_protocol.strategy == (
        "grasp_verify_then_joint_pair_aligned_contact_preserving_planned_lift"
    )
    assert definition.control_protocol.close_duration_options_s == (
        1.25,
        1.5,
        1.75,
    )
    assert CAMPAIGN.edges_m == (0.079,)
    assert CAMPAIGN.fixed_mass_kg == pytest.approx(0.160)
    assert CAMPAIGN.friction == pytest.approx(0.8)
    assert CAMPAIGN.source_candidate_id == 15941124607131459
    assert CAMPAIGN.close_modes == ("original", "synchronized_preload")
    assert CAMPAIGN.plan_candidates_per_pair == 4
    assert CAMPAIGN.feedback_plan_candidates_per_pair == 4
    assert CAMPAIGN.budget_config() == BUDGET.as_mapping()


def test_v15_seed_config_preserves_the_authenticated_physical_grasp() -> None:
    config = load_config(CONFIG_PATH)
    source = json.loads(SOURCE_CONFIG_PATH.read_text(encoding="utf-8"))

    for key in ("cube", "scene", "hand_pose", "contact_point_plan"):
        assert config[key] == source[key]
    for key in (
        "precontact_targets_rad",
        "contact_preload_targets_rad",
        "close_profile",
    ):
        assert config["control"][key] == source["control"][key]
    assert config["grasp_pose"] == source["grasp_pose"]
    assert config["candidate_metadata"]["candidate_id"] == 15107097413537610
    assert config["candidate_metadata"]["source_candidate_id"] == (
        15941124607131459
    )
    assert config["candidate_metadata"]["v15_joint_pair_near_zero_seed"][
        "source_candidate_id"
    ] == 15941124607131459
    campaign = config["contact_preserving_planned_lift_campaign"]
    assert campaign["source_seed"]["candidate_id"] == 15941124607131459
    assert campaign["search_contract"]["close_modes"] == [
        "original",
        "synchronized_preload",
    ]
    assert campaign["budget"]["sequential_plans_per_grasp"] == 4
    assert all(
        isinstance(config[name], str) and len(config[name]) == 64
        for name in V15_TOP_LEVEL_ID_FIELDS
    )


def test_v15_plan_requires_hashed_21_node_response_jacobians() -> None:
    config = load_config(CONFIG_PATH)
    plan = ManipulationPlanParameters.from_config(config["manipulation_plan"])

    assert plan.schema_version == 2
    assert len(plan.joint_pair_residual_jacobian_2x8) == 21
    assert len(plan.joint_pair_residual_jacobian_2x8[0]) == 2
    assert len(plan.object_response_jacobian_6x8[0]) == 6
    assert len(plan.target_force_jacobian_3x8[0]) == 3
    assert all(
        len(row) == len(ACTIVE_ACTUATORS)
        for matrix in plan.object_response_jacobian_6x8
        for row in matrix
    )

    malformed = _config()
    malformed["manipulation_plan"][
        "joint_pair_residual_jacobian_2x8"
    ][0].pop()
    with pytest.raises(ValueError, match="must contain 2 rows"):
        validate_config(malformed)


def test_v15_alignment_feedback_and_top_level_identities_fail_closed() -> None:
    config = _config()
    config["joint_pair_alignment"]["axis"] = "-Y"
    with pytest.raises(ValueError, match=r"axis must be \+Y"):
        validate_config(config)

    config = _config()
    outside = replace(JOINT_PAIR_FEEDBACK, alignment_gain=1.1)
    assert isinstance(outside, JointPairFeedbackParameters)
    config["joint_pair_feedback"] = outside.as_config()
    install_v15_top_level_identities(config)
    with pytest.raises(ValueError, match="alignment_gain is outside"):
        validate_config(config)

    config = _config()
    config["controller_id"] = "0" * 64
    with pytest.raises(ValueError, match="controller_id does not match"):
        validate_config(config)


def test_v14_configs_remain_byte_stable_and_valid() -> None:
    for path, expected_sha256 in SEALED_V14_CONFIGS.items():
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        validate_config(json.loads(payload))
