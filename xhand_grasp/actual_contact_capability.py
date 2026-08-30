"""Shared capability checks for measured actual-contact experiments.

Schema v9 introduced the measured contact-pose protocol.  Later schemas may
reuse that protocol with a different registered campaign, so runtime code must
bind evidence to the resolved experiment rather than to the original v9 ID.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .experiment import ExperimentDefinition, resolve_experiment


ACTUAL_CONTACT_SCHEMA_VERSIONS = (9, 10, 11, 12, 13, 14, 15, 16)
ACTUAL_CONTACT_TUNING_STRATEGY = (
    "actual_contact_grasp_pose_smooth_vertical_lift"
)
CONTACT_POINT_TARGETED_TUNING_STRATEGY = (
    "contact_point_targeted_actual_grasp_pose"
)
SCALED_CONTACT_DOWNSIZE_TUNING_STRATEGY = (
    "scaled_contact_downsize_actual_grasp_then_lift"
)
CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY = (
    "contact_preserving_planned_lift"
)
JOINT_PAIR_NEAR_ZERO_CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY = (
    "joint_pair_near_zero_contact_preserving_planned_lift"
)
JOINT_PAIR_NEAR_ZERO_ROLLING_SLIP_CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY = (
    "joint_pair_near_zero_rolling_slip_contact_preserving_planned_lift"
)
ACTUAL_CONTACT_TUNING_STRATEGIES = frozenset(
    (
        ACTUAL_CONTACT_TUNING_STRATEGY,
        CONTACT_POINT_TARGETED_TUNING_STRATEGY,
        SCALED_CONTACT_DOWNSIZE_TUNING_STRATEGY,
        CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY,
        JOINT_PAIR_NEAR_ZERO_CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY,
        JOINT_PAIR_NEAR_ZERO_ROLLING_SLIP_CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY,
    )
)
LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID = (
    "left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift"
)


def is_actual_contact_definition(definition: ExperimentDefinition) -> bool:
    """Return whether ``definition`` implements the measured-pose protocol."""

    return bool(
        definition.tuning_strategy in ACTUAL_CONTACT_TUNING_STRATEGIES
        and definition.actual_contact_grasp_pose is not None
        and definition.actual_contact_grasp_pose_campaign is not None
    )


def is_contact_preserving_planned_lift_definition(
    definition: ExperimentDefinition,
) -> bool:
    """Return whether a definition owns the schema-v14 planned-lift capability.

    Capability detection intentionally does not key off one historical
    experiment ID.  Independently registered geometry envelopes may reuse the
    same versioned v14 controller and evidence contract without allowing a
    schema-v9--v13 actual-contact experiment into the trace-sparing path.
    """

    protocol = definition.control_protocol
    return bool(
        is_actual_contact_definition(definition)
        and definition.tuning_strategy
        == CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY
        and definition.contact_preserving_planned_lift_campaign is not None
        and protocol is not None
        and protocol.strategy
        == "grasp_verify_then_contact_preserving_planned_lift"
    )


def resolve_contact_preserving_planned_lift_definition(
    config: Mapping[str, Any],
    *,
    context: str = "schema-v14 contact-preserving operation",
) -> ExperimentDefinition:
    """Resolve one registered schema-v14 contact-preserving experiment."""

    try:
        schema_version = int(config.get("schema_version", 0))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} requires schema v14") from error
    if schema_version != 14:
        raise ValueError(f"{context} requires schema v14")
    definition = resolve_experiment(config)
    if not is_contact_preserving_planned_lift_definition(definition):
        raise ValueError(
            f"{context} requires a registered schema-v14 contact-preserving "
            "planned-lift experiment"
        )
    if config.get("experiment_id") != definition.experiment_id:
        raise ValueError(f"{context} experiment_id does not match the registry")
    return definition


def is_joint_pair_near_zero_contact_preserving_planned_lift_definition(
    definition: ExperimentDefinition,
) -> bool:
    """Return whether ``definition`` owns the v15/v16 near-zero protocol.

    This family check is deliberately separate from the schema-v14
    capability.  Schema-specific validation remains fail-closed in
    :func:`resolve_experiment` and in the v15/v16 identity modules.
    """

    protocol = getattr(definition, "control_protocol", None)
    return bool(
        getattr(definition, "tuning_strategy", None)
        in (
            JOINT_PAIR_NEAR_ZERO_CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY,
            JOINT_PAIR_NEAR_ZERO_ROLLING_SLIP_CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY,
        )
        and is_actual_contact_definition(definition)
        and getattr(
            definition, "contact_preserving_planned_lift_campaign", None
        )
        is not None
        and protocol is not None
        and protocol.strategy
        in (
            "grasp_verify_then_joint_pair_aligned_contact_preserving_planned_lift",
            "grasp_verify_then_joint_pair_aligned_rolling_slip_"
            "contact_preserving_planned_lift",
        )
    )


def resolve_joint_pair_near_zero_contact_preserving_planned_lift_definition(
    config: Mapping[str, Any],
    *,
    context: str = "schema-v15/v16 joint-pair near-zero operation",
) -> ExperimentDefinition:
    """Resolve one registered schema-v15/v16 near-zero lift experiment."""

    try:
        schema_version = int(config.get("schema_version", 0))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} requires schema v15 or v16") from error
    if schema_version not in (15, 16):
        raise ValueError(f"{context} requires schema v15 or v16")
    definition = resolve_experiment(config)
    if not is_joint_pair_near_zero_contact_preserving_planned_lift_definition(
        definition
    ):
        raise ValueError(
            f"{context} requires a registered schema-v15/v16 joint-pair "
            "near-zero contact-preserving planned-lift experiment"
        )
    if config.get("experiment_id") != definition.experiment_id:
        raise ValueError(f"{context} experiment_id does not match the registry")
    return definition


def resolve_actual_contact_definition(
    config: Mapping[str, Any],
    *,
    context: str = "actual-contact operation",
) -> ExperimentDefinition:
    """Resolve and authenticate one registered actual-contact configuration."""

    try:
        schema_version = int(config.get("schema_version", 0))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} requires a supported schema version") from error
    if schema_version not in ACTUAL_CONTACT_SCHEMA_VERSIONS:
        supported = ", ".join(str(value) for value in ACTUAL_CONTACT_SCHEMA_VERSIONS)
        raise ValueError(
            f"{context} requires an actual-contact schema ({supported})"
        )
    definition = resolve_experiment(config)
    if not is_actual_contact_definition(definition):
        raise ValueError(
            f"{context} requires a registered actual-contact experiment"
        )
    if config.get("experiment_id") != definition.experiment_id:
        raise ValueError(f"{context} experiment_id does not match the registry")
    return definition


def actual_contact_experiment_id(
    config: Mapping[str, Any],
    *,
    context: str = "actual-contact operation",
) -> str:
    """Return the authenticated experiment ID for ``config``."""

    return resolve_actual_contact_definition(config, context=context).experiment_id


__all__ = [
    "ACTUAL_CONTACT_SCHEMA_VERSIONS",
    "ACTUAL_CONTACT_TUNING_STRATEGY",
    "ACTUAL_CONTACT_TUNING_STRATEGIES",
    "CONTACT_POINT_TARGETED_TUNING_STRATEGY",
    "CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY",
    "JOINT_PAIR_NEAR_ZERO_CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY",
    "JOINT_PAIR_NEAR_ZERO_ROLLING_SLIP_CONTACT_PRESERVING_PLANNED_LIFT_TUNING_STRATEGY",
    "SCALED_CONTACT_DOWNSIZE_TUNING_STRATEGY",
    "LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID",
    "actual_contact_experiment_id",
    "is_contact_preserving_planned_lift_definition",
    "is_joint_pair_near_zero_contact_preserving_planned_lift_definition",
    "is_actual_contact_definition",
    "resolve_actual_contact_definition",
    "resolve_contact_preserving_planned_lift_definition",
    "resolve_joint_pair_near_zero_contact_preserving_planned_lift_definition",
]
