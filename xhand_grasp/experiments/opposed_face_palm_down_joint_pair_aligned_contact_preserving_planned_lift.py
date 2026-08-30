"""Schema-v14 joint-pair-aligned contact-preserving lift experiment.

This experiment deliberately reuses the complete controller, campaign and
acceptance contract of the original schema-v14 planned-lift experiment.  Its
geometry search widens the *static* hand-root yaw interval and its precontact
envelope admits the measured near-zero branch's conservative thumb pose.  The
independent experiment identifier and artifact root keep this search from
mutating or being confused with sealed v14 evidence.
"""

from __future__ import annotations

from dataclasses import replace

from xhand_grasp.experiments.opposed_face_palm_down_contact_preserving_planned_lift import (
    CONTACT_PRESERVING_PLANNED_LIFT as BASE_EXPERIMENT,
)


EXPERIMENT_ID = (
    "left_opposed_face_palm_down_joint_pair_aligned_"
    "contact_preserving_planned_lift"
)
ARTIFACT_ROOT = (
    "artifacts/left_opposed_face_palm_down_joint_pair_aligned_"
    "contact_preserving_planned_lift"
)

_THUMB_BEND = "left_hand_thumb_bend_joint_actuator"
assert BASE_EXPERIMENT.search_bounds.pregrasp_targets_rad is not None
PRECONTACT_TARGET_BOUNDS_RAD = {
    actuator: tuple(bounds)
    for actuator, bounds in BASE_EXPERIMENT.search_bounds.pregrasp_targets_rad.items()
}
# The near-zero joint-pair geometry branch uses a 1.522366-rad collision-free
# thumb precontact.  The compiled XHAND actuator and joint both allow 1.832
# rad, so 1.60 remains a conservative registered bound.  This change is local
# to the new experiment; the sealed base experiment retains its 1.50 ceiling.
PRECONTACT_TARGET_BOUNDS_RAD[_THUMB_BEND] = (
    PRECONTACT_TARGET_BOUNDS_RAD[_THUMB_BEND][0],
    1.60,
)

# Keep the original roll envelope and extend only the root-orientation yaw.
# Palm-plane and finger tilt remain governed by the shared far-hand pose
# constraints, both at 30--40 degrees, so broadening Euler yaw cannot weaken
# those physical checks.
SEARCH_BOUNDS = replace(
    BASE_EXPERIMENT.search_bounds,
    hand_roll_deg=(-10.0, 10.0),
    hand_yaw_deg=(-30.0, 30.0),
    pregrasp_targets_rad=PRECONTACT_TARGET_BOUNDS_RAD,
)

JOINT_PAIR_ALIGNED_CONTACT_PRESERVING_PLANNED_LIFT = replace(
    BASE_EXPERIMENT,
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand 60--88 mm fixed-160-g contact-preserving planned lift "
        "with an expanded static wrist-yaw envelope for index/middle joint-"
        "pair alignment"
    ),
    search_bounds=SEARCH_BOUNDS,
    artifact_root=ARTIFACT_ROOT,
)

EXPERIMENT_DEFINITION = JOINT_PAIR_ALIGNED_CONTACT_PRESERVING_PLANNED_LIFT

# Capability objects are intentionally shared, not copied with silently
# changed thresholds.  Identity remains experiment-specific via EXPERIMENT_ID.
CAMPAIGN = JOINT_PAIR_ALIGNED_CONTACT_PRESERVING_PLANNED_LIFT.contact_preserving_planned_lift_campaign
CONTROL_PROTOCOL = JOINT_PAIR_ALIGNED_CONTACT_PRESERVING_PLANNED_LIFT.control_protocol
POSE_CONSTRAINTS = JOINT_PAIR_ALIGNED_CONTACT_PRESERVING_PLANNED_LIFT.far_hand_pose_constraints

__all__ = [
    "ARTIFACT_ROOT",
    "CAMPAIGN",
    "CONTROL_PROTOCOL",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "JOINT_PAIR_ALIGNED_CONTACT_PRESERVING_PLANNED_LIFT",
    "POSE_CONSTRAINTS",
    "PRECONTACT_TARGET_BOUNDS_RAD",
    "SEARCH_BOUNDS",
]
