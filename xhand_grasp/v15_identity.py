"""Canonical identities for schema-v15 joint-pair-aligned lift configs."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from .grasp_pose import canonical_sha256, grasp_pose_context
from .v14_identity import object_configuration_context


V15_TOP_LEVEL_ID_FIELDS = (
    "object_config_id",
    "grasp_pose_id",
    "grasp_object_pair_id",
    "planner_id",
    "controller_id",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _domain_id(domain: str, payload: Any) -> str:
    return canonical_sha256(
        {
            "identity_schema_version": 1,
            "domain": domain,
            "payload": payload,
        }
    )


def v15_object_config_id(config: Mapping[str, Any]) -> str:
    return _domain_id(
        "v15_object_configuration", object_configuration_context(config)
    )


def v15_grasp_pose_id(config: Mapping[str, Any]) -> str:
    return _domain_id("v15_actual_grasp_pose", grasp_pose_context(config))


def _alignment_id(config: Mapping[str, Any]) -> str:
    alignment = config.get("joint_pair_alignment")
    if not isinstance(alignment, Mapping):
        raise ValueError("v15 identity requires joint_pair_alignment")
    value = alignment.get("alignment_id")
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("joint_pair_alignment.alignment_id must be SHA-256")
    return value


def v15_grasp_object_pair_id(config: Mapping[str, Any]) -> str:
    return _domain_id(
        "v15_joint_pair_grasp_object_pair",
        {
            "object_config_id": v15_object_config_id(config),
            "grasp_pose_id": v15_grasp_pose_id(config),
            "alignment_id": _alignment_id(config),
        },
    )


def v15_planner_id(config: Mapping[str, Any]) -> str:
    plan = config.get("manipulation_plan")
    targets = config.get("contact_force_targets_n")
    if not isinstance(plan, Mapping) or not isinstance(targets, Mapping):
        raise ValueError("v15 planner identity requires plan and force targets")
    return _domain_id(
        "v15_joint_pair_constrained_sequential_planner",
        {
            "grasp_object_pair_id": v15_grasp_object_pair_id(config),
            "plan_id": plan.get("plan_id"),
            "target_id": targets.get("target_id"),
            "alignment_id": _alignment_id(config),
            "physical_probe_count": 17,
            "knot_count": 21,
            "continuous_audit_timestep_s": 0.001,
        },
    )


def v15_controller_id(config: Mapping[str, Any]) -> str:
    force_feedback = config.get("contact_feedback")
    pair_feedback = config.get("joint_pair_feedback")
    if not isinstance(force_feedback, Mapping) or not isinstance(
        pair_feedback, Mapping
    ):
        raise ValueError(
            "v15 controller identity requires force and joint-pair feedback"
        )
    return _domain_id(
        "v15_joint_pair_aligned_contact_preserving_controller",
        {
            "planner_id": v15_planner_id(config),
            "force_feedback_id": force_feedback.get("feedback_id"),
            "joint_pair_feedback_id": pair_feedback.get("feedback_id"),
            "control_protocol": copy.deepcopy(config.get("control_protocol")),
            "control": copy.deepcopy(config.get("control")),
        },
    )


def expected_v15_top_level_identities(
    config: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "object_config_id": v15_object_config_id(config),
        "grasp_pose_id": v15_grasp_pose_id(config),
        "grasp_object_pair_id": v15_grasp_object_pair_id(config),
        "planner_id": v15_planner_id(config),
        "controller_id": v15_controller_id(config),
    }


def install_v15_top_level_identities(config: dict[str, Any]) -> dict[str, str]:
    """Install all five identities after every resolved v15 block is present."""

    values = expected_v15_top_level_identities(config)
    config.update(values)
    return values


def validate_v15_top_level_identities(config: Mapping[str, Any]) -> None:
    values = {name: config.get(name) for name in V15_TOP_LEVEL_ID_FIELDS}
    if any(
        not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
        for value in values.values()
    ):
        raise ValueError(
            "schema-v15 requires non-empty object, grasp, pair, planner and "
            "controller SHA-256 identities"
        )
    expected = expected_v15_top_level_identities(config)
    for name in V15_TOP_LEVEL_ID_FIELDS:
        if values[name] != expected[name]:
            raise ValueError(f"schema-v15 {name} does not match its content")


__all__ = [
    "V15_TOP_LEVEL_ID_FIELDS",
    "expected_v15_top_level_identities",
    "install_v15_top_level_identities",
    "v15_controller_id",
    "v15_grasp_object_pair_id",
    "v15_grasp_pose_id",
    "v15_object_config_id",
    "v15_planner_id",
    "validate_v15_top_level_identities",
]
