"""Schema-v10 larger fixed-mass actual-contact grasp and smooth-lift experiment."""

from __future__ import annotations

from xhand_grasp.experiment import (
    ActualContactGraspPoseCampaignParameters,
    ControlProtocolSettings,
    ExperimentDefinition,
    FarHandPoseConstraints,
    RobustnessParameters,
    SearchBounds,
)
from xhand_grasp.experiments.opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift import (
    CUBE_YAW_DEG,
    GRASP_POSE,
)
from xhand_grasp.experiments.opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift import (
    CLOSURE_ALIGNMENT,
    MOTION_SMOOTHNESS,
)
from xhand_grasp.experiments.opposed_face_palm_down_high_thumb_variable_size_pose_preserving import (
    MANIPULATION_DELTA_BOUNDS_RAD,
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
    "left_opposed_face_palm_down_larger_actual_contact_grasp_pose_smooth_vertical_lift"
)
THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
EDGES_M = tuple(value / 1000.0 for value in range(72, 91))
BRIDGE_EDGES_M = tuple(value / 1000.0 for value in range(68, 72))
THUMB_ACTUAL_CENTERS_RAD = (1.40, 1.45, 1.50, 1.55, 1.60)
ARTIFACT_ROOT = (
    "artifacts/"
    "left_opposed_face_palm_down_larger_actual_contact_grasp_pose_smooth_vertical_lift"
)
SOURCE_POSE_MANIFEST = (
    "artifacts/"
    "left_opposed_face_palm_down_larger_actual_contact_grasp_pose_smooth_vertical_lift/"
    "source_manifests/v9_measured_grasp_sources.json"
)

# Fixed-mass results are a geometric/control ablation and must never be labelled
# as constant-density or same-material scale validation.
VALIDATION_LABELS = {
    "grasp": "validated_fixed_mass_grasp_ablation",
    "manipulation": "validated_fixed_mass_manipulation_ablation",
    "robust": "validated_fixed_mass_robust_full_success_ablation",
}

# The bridge is an internal continuation aid only.  Published/search cells are
# exactly the 72--90 mm EDGES_M values above.
SIZE_CONTINUATION_POLICY = {
    "method": "damped_least_squares_distal_witness",
    "step_m": 0.001,
    "bridge_edges_m": list(BRIDGE_EDGES_M),
    "published_edges_m": list(EDGES_M),
    "adjust_hand_root_pose": True,
    "adjust_non_thumb_bend_active_joints": True,
    "hold_thumb_bend_at_cell_center": True,
    "publish_bridge_results": False,
}

POSE_CONSTRAINTS = FarHandPoseConstraints(
    finger_down_tilt_deg=(30.0, 40.0),
    palm_plane_ground_angle_deg=(30.0, 40.0),
    root_cube_distance_m=(0.135, 0.190),
    cube_position_in_root_m={
        "x": (0.085, 0.130),
        "y": (-0.045, -0.005),
        "z": (0.090, 0.140),
    },
    legacy_press_reference_translation_m=(
        -0.036454669709448065,
        0.008499986928773153,
        0.19777491509344874,
    ),
)

CONTACT_PRELOAD_TARGET_BOUNDS_RAD = {
    THUMB_BEND_ACTUATOR: (1.40, 1.75),
    "left_hand_thumb_rota_joint1_actuator": (-0.10, 0.80),
    "left_hand_thumb_rota_joint2_actuator": (0.45, 1.10),
    "left_hand_index_bend_joint_actuator": (-0.15, 0.15),
    "left_hand_index_joint1_actuator": (0.20, 0.90),
    "left_hand_index_joint2_actuator": (0.80, 1.80),
    "left_hand_mid_joint1_actuator": (0.35, 1.15),
    "left_hand_mid_joint2_actuator": (0.65, 1.70),
}

# Precontact targets share the expanded seven-joint envelope.  Thumb bend has
# its own retreat range so an actual 1.40--1.60 rad contact pose is not confused
# with the approach command.
PRECONTACT_TARGET_BOUNDS_RAD = {
    **CONTACT_PRELOAD_TARGET_BOUNDS_RAD,
    THUMB_BEND_ACTUATOR: (0.90, 1.50),
}

CONTROL_PROTOCOL = ControlProtocolSettings(
    strategy="grasp_verify_then_manipulate",
    failure_behavior="abort_hold_grasp_pose",
    settle_s=0.5,
    close_s=1.5,
    verify_timeout_s=0.75,
    stable_window_s=0.25,
    manipulate_s=2.0,
    min_hold_s=1.0,
    grasp_gate=V6_EXPERIMENT.control_protocol.grasp_gate,
    manipulation_profile="minimum_jerk_quintic",
    close_duration_options_s=(1.0, 1.25, 1.5, 1.75, 2.0),
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
    source_pose_manifest=SOURCE_POSE_MANIFEST,
    validation_labels=VALIDATION_LABELS,
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
    mass_scale=(1.0, 1.0),
    friction_delta=(-0.1, 0.1),
    z_offset_delta_m=(0.0, 0.0005),
)

LARGER_ACTUAL_CONTACT_GRASP_POSE_SMOOTH_VERTICAL_LIFT = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand 72--90 mm fixed-160-g search over measured stable contact "
        "qpos, followed by minimum-jerk near-vertical manipulation"
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
    artifact_root=ARTIFACT_ROOT,
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

EXPERIMENT_DEFINITION = LARGER_ACTUAL_CONTACT_GRASP_POSE_SMOOTH_VERTICAL_LIFT

__all__ = [
    "ARTIFACT_ROOT",
    "BRIDGE_EDGES_M",
    "CAMPAIGN",
    "CONTACT_PRELOAD_TARGET_BOUNDS_RAD",
    "CONTROL_PROTOCOL",
    "EDGES_M",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "LARGER_ACTUAL_CONTACT_GRASP_POSE_SMOOTH_VERTICAL_LIFT",
    "POSE_CONSTRAINTS",
    "PRECONTACT_TARGET_BOUNDS_RAD",
    "ROBUSTNESS",
    "SIZE_CONTINUATION_POLICY",
    "SOURCE_POSE_MANIFEST",
    "THUMB_ACTUAL_CENTERS_RAD",
    "THUMB_BEND_ACTUATOR",
    "VALIDATION_LABELS",
]
