"""Schema-v6 pose-preserving acquisition around the verified 60 mm grasp.

The v5 campaign established a useful terminal hand/object relationship, but a
terminal grasp is not by itself a safe acquisition trajectory.  This
experiment treats that relationship as a geometric seed, introduces a
collision-free pregrasp command and lets each actuator start closing at an
independently versioned fraction of the CLOSE phase.  The cube remains a free
MuJoCo body throughout; pose preservation is an acceptance contract rather
than a scene constraint.
"""

from __future__ import annotations

from dataclasses import replace

from xhand_grasp.experiment import (
    ExperimentDefinition,
    FarHandPoseConstraints,
    PosePreservationSettings,
    SearchBounds,
)
from xhand_grasp.experiments.opposed_face_palm_down_larger_cube_relative_pose_rescue import (
    RESCUE_MANIPULATION_DELTA_BOUNDS,
)
from xhand_grasp.experiments.opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift import (
    ACTUATOR_TARGET_BOUNDS_RAD,
    FAR_HAND_CAMPAIGN as V5_FAR_HAND_CAMPAIGN,
    FAR_HAND_FINGERTIP_GRASP_THEN_LIFT as V5_EXPERIMENT,
    FINGERTIP_CONTACT_PREFERENCES,
    PRIMARY_FACE_ASSIGNMENT,
    FALLBACK_FACE_ASSIGNMENT,
)


EXPERIMENT_ID = "left_opposed_face_palm_down_pose_preserving_grasp"

# The exact terminal geometry recovered from the 1.17 rad acquisition trace is
# near 35.8 degrees.  Schema v6 therefore owns a separate pose envelope rather
# than weakening or silently reinterpreting schema v5's 10--20 degree range.
POSE_CONSTRAINTS = FarHandPoseConstraints(
    finger_down_tilt_deg=(30.0, 40.0),
    palm_plane_ground_angle_deg=(30.0, 40.0),
    root_cube_distance_m=(0.145, 0.160),
    cube_position_in_root_m={
        "x": (0.095, 0.110),
        "y": (-0.030, -0.018),
        "z": (0.103, 0.118),
    },
    legacy_press_reference_translation_m=(
        -0.036454669709448065,
        0.008499986928773153,
        0.19777491509344874,
    ),
)

POSE_PRESERVATION = PosePreservationSettings(
    max_translation_m=0.0005,
    max_orientation_drift_deg=1.0,
    require_support_contact=True,
    require_no_hand_cube_contact_during_settle=True,
    initialize_active_joints_at_pregrasp=True,
)

# The pregrasp bounds surround the measured joint state at the beginning of
# stable acquisition while retaining enough retraction to remove premature
# side loading.  They are distinct from the terminal grasp bounds by design.
PREGRASP_TARGET_BOUNDS_RAD = {
    "left_hand_thumb_bend_joint_actuator": (0.60, 1.20),
    "left_hand_thumb_rota_joint1_actuator": (0.00, 0.50),
    "left_hand_thumb_rota_joint2_actuator": (0.55, 1.00),
    "left_hand_index_bend_joint_actuator": (-0.08, 0.08),
    "left_hand_index_joint1_actuator": (0.40, 0.70),
    "left_hand_index_joint2_actuator": (1.05, 1.35),
    "left_hand_mid_joint1_actuator": (0.60, 0.90),
    "left_hand_mid_joint2_actuator": (0.85, 1.10),
}

# The terminal *actual* qpos recovered from the validated v5 grasp includes a
# lower thumb rotation-1 value (0.233 rad) than the old command-search floor.
# V6 must be able to reduce preload while keeping thumb bend high; otherwise
# the inherited servo error mechanically pushes the free cube before the grasp
# can be certified.
GRASP_TARGET_BOUNDS_RAD = {
    **ACTUATOR_TARGET_BOUNDS_RAD,
    "left_hand_thumb_bend_joint_actuator": (1.05, 1.35),
    "left_hand_thumb_rota_joint1_actuator": (0.15, 0.80),
    "left_hand_thumb_rota_joint2_actuator": (0.75, 1.10),
    "left_hand_index_joint2_actuator": (1.20, 1.80),
    "left_hand_mid_joint2_actuator": (0.85, 1.70),
}

FAR_HAND_CAMPAIGN = replace(
    V5_FAR_HAND_CAMPAIGN,
    tilt_band_centers_deg=(30.0, 32.5, 35.0, 37.5, 40.0),
)

# The recovered seed requires a palm angle above the old v5 ceiling.  This is
# the only inherited evaluation field changed by the new experiment.
EVALUATION = replace(
    V5_EXPERIMENT.evaluation,
    max_palm_down_angle_deg=40.0,
)

POSE_PRESERVING_GRASP = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand 60 mm opposed-face acquisition with an explicit pregrasp, "
        "delayed per-actuator closure and sub-millimetre initial cube-pose "
        "preservation"
    ),
    evaluation=EVALUATION,
    candidate_faces=(PRIMARY_FACE_ASSIGNMENT, FALLBACK_FACE_ASSIGNMENT),
    search_bounds=SearchBounds(
        palm_pitch_values_deg=FAR_HAND_CAMPAIGN.tilt_band_centers_deg,
        hand_roll_deg=(-7.0, 3.0),
        hand_yaw_deg=(-2.0, 8.0),
        cube_position_in_root_m=POSE_CONSTRAINTS.cube_position_in_root_m,
        cube_yaw_deg=(24.0, 38.0),
        actuator_targets_rad=GRASP_TARGET_BOUNDS_RAD,
        seed=20260821,
        kinematic_samples_per_pitch=FAR_HAND_CAMPAIGN.static_samples_per_band,
        dynamic_candidate_count=FAR_HAND_CAMPAIGN.dynamic_candidate_count,
        local_refine_seed_count=(
            len(FAR_HAND_CAMPAIGN.tilt_band_centers_deg)
            * FAR_HAND_CAMPAIGN.grasp_refine_seed_count_per_band
        ),
        local_refine_per_seed=FAR_HAND_CAMPAIGN.grasp_refine_per_seed,
        final_candidate_count=FAR_HAND_CAMPAIGN.exact_candidate_count,
        perturbations_per_final_candidate=(
            V5_EXPERIMENT.robustness.perturbation_count
        ),
        fallback_kinematic_samples_per_pitch=(
            FAR_HAND_CAMPAIGN.fallback_static_samples_per_band
        ),
        fallback_candidate_count=FAR_HAND_CAMPAIGN.dynamic_candidates_per_band,
        final_target_delta_rad=None,
        manipulation_delta_rad=RESCUE_MANIPULATION_DELTA_BOUNDS,
        pregrasp_targets_rad=PREGRASP_TARGET_BOUNDS_RAD,
    ),
    robustness=V5_EXPERIMENT.robustness,
    artifact_root=(
        "artifacts/left_opposed_face_palm_down_pose_preserving_grasp"
    ),
    control_protocol=V5_EXPERIMENT.control_protocol,
    tuning_strategy="pose_preserving_grasp",
    contact_alignment=V5_EXPERIMENT.contact_alignment,
    far_hand_pose_constraints=POSE_CONSTRAINTS,
    fingertip_contact_preferences=FINGERTIP_CONTACT_PREFERENCES,
    far_hand_campaign=FAR_HAND_CAMPAIGN,
    pose_preservation=POSE_PRESERVATION,
)

EXPERIMENT_DEFINITION = POSE_PRESERVING_GRASP

__all__ = [
    "EVALUATION",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "FAR_HAND_CAMPAIGN",
    "GRASP_TARGET_BOUNDS_RAD",
    "POSE_CONSTRAINTS",
    "POSE_PRESERVATION",
    "POSE_PRESERVING_GRASP",
    "PREGRASP_TARGET_BOUNDS_RAD",
]
