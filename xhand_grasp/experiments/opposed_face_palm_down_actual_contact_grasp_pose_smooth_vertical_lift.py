"""Schema-v9 measured contact-grasp pose and smooth vertical lift experiment."""

from __future__ import annotations

from xhand_grasp.experiment import (
    ActualContactGraspPoseCampaignParameters,
    ActualContactGraspPoseSettings,
    ExperimentDefinition,
    RobustnessParameters,
    SearchBounds,
)
from xhand_grasp.experiments.opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift import (
    CLOSURE_ALIGNMENT,
    CONTROL_PROTOCOL,
    MOTION_SMOOTHNESS,
)
from xhand_grasp.experiments.opposed_face_palm_down_high_thumb_variable_size_pose_preserving import (
    MANIPULATION_DELTA_BOUNDS_RAD,
    POSE_CONSTRAINTS,
    PREGRASP_TARGET_BOUNDS_RAD as V8_PREGRASP_TARGET_BOUNDS_RAD,
)
from xhand_grasp.experiments.opposed_face_palm_down_pose_preserving_grasp import (
    EVALUATION,
    POSE_PRESERVATION,
    POSE_PRESERVING_GRASP as V6_EXPERIMENT,
)
from xhand_grasp.experiments.opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift import (
    FALLBACK_FACE_ASSIGNMENT,
    FINGERTIP_CONTACT_PREFERENCES,
    PRIMARY_FACE_ASSIGNMENT,
)


EXPERIMENT_ID = (
    "left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift"
)
THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
EDGES_M = tuple(value / 1000.0 for value in range(60, 71))
THUMB_ACTUAL_CENTERS_RAD = (1.40, 1.45, 1.50, 1.55, 1.60)
CUBE_YAW_DEG = 27.609990189403167

# These are controller preload bounds, not measured-grasp-pose bounds.  The
# preload may exceed the requested actual qpos to maintain compliant contact.
CONTACT_PRELOAD_TARGET_BOUNDS_RAD = {
    **dict(V6_EXPERIMENT.search_bounds.actuator_targets_rad),
    THUMB_BEND_ACTUATOR: (1.40, 1.75),
}
PRECONTACT_TARGET_BOUNDS_RAD = {
    **dict(V8_PREGRASP_TARGET_BOUNDS_RAD),
    # A Jacobian-derived 2--4 mm outward retreat can retain a high bend value
    # while rotating the pad away from the face.  These are command bounds,
    # not evidence that the actual contact pose has already been reached.
    THUMB_BEND_ACTUATOR: (0.90, 1.50),
    "left_hand_index_bend_joint_actuator": (-0.15, 0.15),
    "left_hand_mid_joint2_actuator": (0.75, 1.30),
}

GRASP_POSE = ActualContactGraspPoseSettings(
    thumb_actual_range_rad=(1.40, 1.60),
    max_nominal_joint_error_rad=0.04,
    max_joint_stability_span_rad=0.03,
    verify_continuous_s=0.25,
)

CAMPAIGN = ActualContactGraspPoseCampaignParameters(
    schema_version=1,
    edges_m=EDGES_M,
    thumb_actual_centers_rad=THUMB_ACTUAL_CENTERS_RAD,
    fixed_mass_kg=0.160,
    friction=0.8,
    cube_center_xy_m=(0.071, -0.027),
    cube_yaw_deg=CUBE_YAW_DEG,
    seed=20260821,
    witness_signed_gap_m=(-0.0005, 0.00025),
    min_witness_normal_alignment=0.95,
    min_witness_edge_margin_m=0.0005,
    max_contact_height_spread_m=0.005,
    precontact_retreat_m=(0.002, 0.004),
)

ROBUSTNESS = RobustnessParameters(
    edge_m=EDGES_M,
    mass_kg=(CAMPAIGN.fixed_mass_kg,),
    friction=(CAMPAIGN.friction,),
    perturbation_count=50,
    required_pass_count=45,
    seed=CAMPAIGN.seed,
    position_xy_delta_m=(-0.0015, 0.0015),
    rpy_delta_deg=(-3.0, 3.0),
    mass_scale=(0.9, 1.1),
    friction_delta=(-0.1, 0.1),
    z_offset_delta_m=(0.0, 0.0005),
)

ACTUAL_CONTACT_GRASP_POSE_SMOOTH_VERTICAL_LIFT = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand 60--70 mm search over measured stable contact qpos, "
        "followed by minimum-jerk near-vertical manipulation"
    ),
    evaluation=EVALUATION,
    candidate_faces=(PRIMARY_FACE_ASSIGNMENT, FALLBACK_FACE_ASSIGNMENT),
    search_bounds=SearchBounds(
        palm_pitch_values_deg=(30.0, 32.5, 35.0, 37.5, 40.0),
        hand_roll_deg=(-7.0, 3.0),
        hand_yaw_deg=(-2.0, 8.0),
        cube_position_in_root_m=POSE_CONSTRAINTS.cube_position_in_root_m,
        cube_yaw_deg=(CUBE_YAW_DEG, CUBE_YAW_DEG),
        actuator_targets_rad=CONTACT_PRELOAD_TARGET_BOUNDS_RAD,
        seed=CAMPAIGN.seed,
        kinematic_samples_per_pitch=CAMPAIGN.quick_static_samples_per_cell,
        dynamic_candidate_count=CAMPAIGN.maximum_dynamic_grasp_candidate_count,
        local_refine_seed_count=CAMPAIGN.local_pose_count,
        local_refine_per_seed=CAMPAIGN.local_refine_per_pose,
        final_candidate_count=CAMPAIGN.exact_grasp_pose_count,
        perturbations_per_final_candidate=CAMPAIGN.perturbations_per_trajectory,
        fallback_kinematic_samples_per_pitch=(
            CAMPAIGN.full_static_samples_per_cell
        ),
        fallback_candidate_count=CAMPAIGN.full_static_retain_per_cell,
        final_target_delta_rad=None,
        manipulation_delta_rad=MANIPULATION_DELTA_BOUNDS_RAD,
        pregrasp_targets_rad=PRECONTACT_TARGET_BOUNDS_RAD,
    ),
    robustness=ROBUSTNESS,
    artifact_root=(
        "artifacts/"
        "left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift"
    ),
    control_protocol=CONTROL_PROTOCOL,
    tuning_strategy="actual_contact_grasp_pose_smooth_vertical_lift",
    contact_alignment=V6_EXPERIMENT.contact_alignment,
    far_hand_pose_constraints=POSE_CONSTRAINTS,
    fingertip_contact_preferences=FINGERTIP_CONTACT_PREFERENCES,
    pose_preservation=POSE_PRESERVATION,
    closure_alignment=CLOSURE_ALIGNMENT,
    motion_smoothness=MOTION_SMOOTHNESS,
    actual_contact_grasp_pose=GRASP_POSE,
    actual_contact_grasp_pose_campaign=CAMPAIGN,
)

EXPERIMENT_DEFINITION = ACTUAL_CONTACT_GRASP_POSE_SMOOTH_VERTICAL_LIFT

__all__ = [
    "ACTUAL_CONTACT_GRASP_POSE_SMOOTH_VERTICAL_LIFT",
    "CAMPAIGN",
    "CONTACT_PRELOAD_TARGET_BOUNDS_RAD",
    "CUBE_YAW_DEG",
    "EDGES_M",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "GRASP_POSE",
    "PRECONTACT_TARGET_BOUNDS_RAD",
    "ROBUSTNESS",
    "THUMB_ACTUAL_CENTERS_RAD",
    "THUMB_BEND_ACTUATOR",
]
