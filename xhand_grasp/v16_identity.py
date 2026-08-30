"""Canonical identities for schema-v16 rolling-slip lift configurations.

Schema v16 intentionally uses a fresh identity domain.  The physical grasp
may be inherited from a sealed v15 candidate, but its planner and controller
semantics depend on material-point relative tangential velocity and therefore
must never authenticate as a v15 artifact.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from .grasp_pose import canonical_sha256, grasp_pose_context
from .v14_identity import object_configuration_context


V16_TOP_LEVEL_ID_FIELDS = (
    "object_config_id",
    "grasp_pose_id",
    "grasp_object_pair_id",
    "planner_id",
    "controller_id",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ROLLING_FEEDBACK_STRATEGY = (
    "previous_frame_rolling_aware_signed_tangent_nullspace"
)


def _domain_id(domain: str, payload: Any) -> str:
    return canonical_sha256(
        {
            "identity_schema_version": 1,
            "domain": domain,
            "payload": payload,
        }
    )


def _require_v16(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 16:
        raise ValueError("v16 identity requires schema_version 16")


def _sha256_field(block: Mapping[str, Any], name: str, context: str) -> str:
    value = block.get(name)
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context}.{name} must be SHA-256")
    return value


def _alignment_id(config: Mapping[str, Any]) -> str:
    alignment = config.get("joint_pair_alignment")
    if not isinstance(alignment, Mapping):
        raise ValueError("v16 identity requires joint_pair_alignment")
    return _sha256_field(alignment, "alignment_id", "joint_pair_alignment")


def _rolling_feedback_id(config: Mapping[str, Any]) -> str:
    feedback = config.get("joint_pair_feedback")
    if not isinstance(feedback, Mapping):
        raise ValueError("v16 identity requires joint_pair_feedback")
    if feedback.get("schema_version") != 2:
        raise ValueError("v16 identity requires joint-pair feedback schema 2")
    if feedback.get("strategy") != _ROLLING_FEEDBACK_STRATEGY:
        raise ValueError("v16 identity requires rolling-aware feedback")
    return _sha256_field(feedback, "feedback_id", "joint_pair_feedback")


def v16_object_config_id(config: Mapping[str, Any]) -> str:
    _require_v16(config)
    return _domain_id(
        "v16_object_configuration", object_configuration_context(config)
    )


def v16_grasp_pose_id(config: Mapping[str, Any]) -> str:
    _require_v16(config)
    return _domain_id("v16_actual_grasp_pose", grasp_pose_context(config))


def v16_grasp_object_pair_id(config: Mapping[str, Any]) -> str:
    return _domain_id(
        "v16_rolling_slip_joint_pair_grasp_object_pair",
        {
            "object_config_id": v16_object_config_id(config),
            "grasp_pose_id": v16_grasp_pose_id(config),
            "alignment_id": _alignment_id(config),
        },
    )


def v16_planner_id(config: Mapping[str, Any]) -> str:
    plan = config.get("manipulation_plan")
    targets = config.get("contact_force_targets_n")
    if not isinstance(plan, Mapping) or not isinstance(targets, Mapping):
        raise ValueError("v16 planner identity requires plan and force targets")
    return _domain_id(
        "v16_rolling_slip_joint_pair_constrained_sequential_planner",
        {
            "grasp_object_pair_id": v16_grasp_object_pair_id(config),
            "plan_id": plan.get("plan_id"),
            "target_id": targets.get("target_id"),
            "alignment_id": _alignment_id(config),
            "physical_probe_count": 17,
            "knot_count": 21,
            "continuous_audit_timestep_s": 0.001,
            "rolling_contact_slip_schema_version": 1,
            "slip_measurement": (
                "signed_material_point_relative_tangent_velocity"
            ),
        },
    )


def v16_controller_id(config: Mapping[str, Any]) -> str:
    force_feedback = config.get("contact_feedback")
    if not isinstance(force_feedback, Mapping):
        raise ValueError("v16 controller identity requires force feedback")
    rolling_feedback_id = _rolling_feedback_id(config)
    return _domain_id(
        "v16_joint_pair_aligned_rolling_slip_contact_preserving_controller",
        {
            "planner_id": v16_planner_id(config),
            "force_feedback_id": force_feedback.get("feedback_id"),
            "joint_pair_feedback_id": rolling_feedback_id,
            "rolling_contact_slip_schema_version": 1,
            "control_protocol": copy.deepcopy(config.get("control_protocol")),
            "control": copy.deepcopy(config.get("control")),
        },
    )


def expected_v16_top_level_identities(
    config: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "object_config_id": v16_object_config_id(config),
        "grasp_pose_id": v16_grasp_pose_id(config),
        "grasp_object_pair_id": v16_grasp_object_pair_id(config),
        "planner_id": v16_planner_id(config),
        "controller_id": v16_controller_id(config),
    }


def install_v16_top_level_identities(config: dict[str, Any]) -> dict[str, str]:
    """Install all five identities after every resolved v16 block is present."""

    values = expected_v16_top_level_identities(config)
    config.update(values)
    return values


def validate_v16_top_level_identities(config: Mapping[str, Any]) -> None:
    _require_v16(config)
    values = {name: config.get(name) for name in V16_TOP_LEVEL_ID_FIELDS}
    if any(
        not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
        for value in values.values()
    ):
        raise ValueError(
            "schema-v16 requires non-empty object, grasp, pair, planner and "
            "controller SHA-256 identities"
        )
    expected = expected_v16_top_level_identities(config)
    for name in V16_TOP_LEVEL_ID_FIELDS:
        if values[name] != expected[name]:
            raise ValueError(f"schema-v16 {name} does not match its content")


__all__ = [
    "V16_TOP_LEVEL_ID_FIELDS",
    "expected_v16_top_level_identities",
    "install_v16_top_level_identities",
    "v16_controller_id",
    "v16_grasp_object_pair_id",
    "v16_grasp_pose_id",
    "v16_object_config_id",
    "v16_planner_id",
    "validate_v16_top_level_identities",
]
