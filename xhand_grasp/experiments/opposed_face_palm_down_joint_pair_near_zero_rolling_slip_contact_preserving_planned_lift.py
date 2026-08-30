"""Schema-v16 rolling-aware near-zero joint-pair lift experiment."""

from __future__ import annotations

from dataclasses import replace

from xhand_grasp.experiment import JointPairFeedbackParameters
from xhand_grasp.experiments.opposed_face_palm_down_joint_pair_near_zero_contact_preserving_planned_lift import (
    ACTUAL_CONTACT_CAMPAIGN as V15_ACTUAL_CONTACT_CAMPAIGN,
    BUDGET,
    CAMPAIGN as V15_CAMPAIGN,
    JOINT_PAIR_ALIGNMENT,
    JOINT_PAIR_NEAR_ZERO_CONTACT_PRESERVING_PLANNED_LIFT as V15_EXPERIMENT,
)


EXPERIMENT_ID = (
    "left_opposed_face_palm_down_joint_pair_near_zero_rolling_slip_"
    "contact_preserving_planned_lift"
)
ARTIFACT_ROOT = (
    "artifacts/left_opposed_face_palm_down_joint_pair_near_zero_rolling_slip_"
    "contact_preserving_planned_lift"
)

VALIDATION_LABELS = {
    "grasp": "validated_near_zero_joint_pair_rolling_slip_grasp",
    "manipulation": (
        "validated_near_zero_joint_pair_rolling_slip_planned_lift"
    ),
    "robust": "validated_near_zero_joint_pair_rolling_slip_robust_lift",
}

# Recovery starts before the legacy 1.5 mm freeze limit and only exits after
# a dwell below a separate lower threshold.  Patch matching and centroid jumps
# are diagnostic: the integrated material-point relative tangent velocity is
# the authoritative slip state.
JOINT_PAIR_FEEDBACK = JointPairFeedbackParameters(
    schema_version=2,
    strategy="previous_frame_rolling_aware_signed_tangent_nullspace",
    # Preserve the already-validated near-zero pair controller as the center;
    # the v16 change is the slip observation/recovery law, not a silent
    # weakening of its alignment loop.
    alignment_gain=0.9325871364810551,
    slip_recovery_gain_rad_per_m=4.0,
    correction_limit_rad=0.020,
    rate_limit_rad_s=0.20,
    acceleration_limit_rad_s2=2.0,
    use_previous_observation=True,
    freeze_threshold_deg=0.50,
    abort_threshold_deg=1.0,
    slip_freeze_threshold_m=0.0015,
    slip_abort_threshold_m=0.0020,
    recovery_behavior="signed_tangent_recover_with_hysteresis_then_abort",
    damping=1e-8,
    vertical_response_weight=1.0,
    force_response_weight=1.0,
    slip_resume_threshold_m=0.0012,
    slip_recovery_enter_threshold_m=0.0004,
    slip_recovery_exit_threshold_m=0.00025,
    slip_exit_dwell_s=0.020,
    relative_velocity_filter_time_constant_s=0.010,
    tangent_prediction_horizon_s=0.050,
    suppress_outward_force_pi_during_recovery=True,
    maximum_patch_match_distance_m=0.003,
    # Diagnostic only: deliberately sensitive enough to expose the known
    # ~1 mm collision-manifold jump without feeding it back into control.
    centroid_jump_diagnostic_threshold_m=0.00005,
    maximum_contact_time_gap_s=0.002,
)

CONTROL_PROTOCOL = replace(
    V15_EXPERIMENT.control_protocol,
    strategy=(
        "grasp_verify_then_joint_pair_aligned_rolling_slip_"
        "contact_preserving_planned_lift"
    ),
)

CAMPAIGN = replace(V15_CAMPAIGN, validation_labels=VALIDATION_LABELS)
ACTUAL_CONTACT_CAMPAIGN = replace(
    V15_ACTUAL_CONTACT_CAMPAIGN,
    validation_labels=VALIDATION_LABELS,
)

JOINT_PAIR_NEAR_ZERO_ROLLING_SLIP_CONTACT_PRESERVING_PLANNED_LIFT = replace(
    V15_EXPERIMENT,
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand fixed-79-mm measured grasp with near-zero directed joint "
        "pair alignment and rolling-aware signed tangent-slip recovery"
    ),
    artifact_root=ARTIFACT_ROOT,
    control_protocol=CONTROL_PROTOCOL,
    tuning_strategy=(
        "joint_pair_near_zero_rolling_slip_contact_preserving_planned_lift"
    ),
    actual_contact_grasp_pose_campaign=ACTUAL_CONTACT_CAMPAIGN,
    contact_preserving_planned_lift_campaign=CAMPAIGN,
    joint_pair_feedback=JOINT_PAIR_FEEDBACK,
)

EXPERIMENT_DEFINITION = (
    JOINT_PAIR_NEAR_ZERO_ROLLING_SLIP_CONTACT_PRESERVING_PLANNED_LIFT
)

__all__ = [
    "ACTUAL_CONTACT_CAMPAIGN",
    "ARTIFACT_ROOT",
    "BUDGET",
    "CAMPAIGN",
    "CONTROL_PROTOCOL",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "JOINT_PAIR_ALIGNMENT",
    "JOINT_PAIR_FEEDBACK",
    "JOINT_PAIR_NEAR_ZERO_ROLLING_SLIP_CONTACT_PRESERVING_PLANNED_LIFT",
    "VALIDATION_LABELS",
]
