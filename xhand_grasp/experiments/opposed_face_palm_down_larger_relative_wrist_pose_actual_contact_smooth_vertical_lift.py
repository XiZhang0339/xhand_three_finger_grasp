"""Schema-v11 larger cube-relative 6D wrist-pose grasp experiment."""

from __future__ import annotations

from xhand_grasp.experiment import (
    ActualContactGraspPoseCampaignParameters,
    ControlProtocolSettings,
    ExperimentDefinition,
    FarHandPoseConstraints,
    RelativeWristPoseSearchParameters,
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
    "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift"
)
THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
EDGES_M = tuple(value / 1000.0 for value in range(85, 105))
THUMB_ACTUAL_CENTERS_RAD = (1.40, 1.45, 1.50, 1.55, 1.60)
CLOCKWISE_ORBIT_DEG = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0)
PRIMARY_ANCHOR_CANDIDATE_ID = 4864000014401350
DENSITY_REVALIDATION_KG_M3 = 0.020 / 0.030**3
ARTIFACT_ROOT = (
    "artifacts/"
    "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift"
)
SOURCE_POSE_MANIFEST = (
    ARTIFACT_ROOT + "/source_manifests/v10_measured_grasp_sources.json"
)

VALIDATION_LABELS = {
    "grasp": "validated_fixed_160g_grasp_ablation",
    "manipulation": "validated_fixed_160g_manipulation_ablation",
    "robust": "validated_fixed_160g_robust_full_success_ablation",
}

POSE_CONSTRAINTS = FarHandPoseConstraints(
    finger_down_tilt_deg=(30.0, 40.0),
    palm_plane_ground_angle_deg=(30.0, 40.0),
    root_cube_distance_m=(0.135, 0.210),
    cube_position_in_root_m={
        "x": (0.070, 0.150),
        "y": (-0.065, 0.020),
        "z": (0.075, 0.160),
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
PRECONTACT_TARGET_BOUNDS_RAD = {
    **CONTACT_PRELOAD_TARGET_BOUNDS_RAD,
    THUMB_BEND_ACTUATOR: (0.90, 1.50),
}

CONTROL_PROTOCOL = ControlProtocolSettings(
    strategy="grasp_verify_then_manipulate",
    failure_behavior="abort_hold_grasp_pose",
    settle_s=0.5,
    close_s=2.0,
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
    quick_static_samples_per_cell=1_000,
    quick_static_retain_per_cell=2,
    full_static_samples_per_cell=3_000,
    full_static_retain_per_cell=3,
    controller_seeds_per_pose=6,
    local_pose_count=40,
    local_refine_per_pose=64,
    exact_grasp_pose_count=40,
    manipulation_probe_count=17,
    manipulation_candidates_per_pose=96,
    manipulation_refine_pose_count=8,
    manipulation_refine_per_pose=128,
    perturbations_per_trajectory=16,
    selected_trajectory_count=5,
    minimum_distinct_edges=3,
    minimum_thumb_bands=2,
    initial_target_success_count=1,
    validation_labels=VALIDATION_LABELS,
)

RELATIVE_WRIST_POSE_SEARCH = RelativeWristPoseSearchParameters(
    schema_version=1,
    clockwise_orbit_deg=CLOCKWISE_ORBIT_DEG,
    root_delta_cube_m={
        "x": (-0.012, 0.012),
        "y": (-0.012, 0.012),
        "z": (-0.015, 0.015),
    },
    wrist_local_rotvec_deg={
        "x": (-6.0, 6.0),
        "y": (-6.0, 6.0),
        "z": (-6.0, 6.0),
    },
    max_wrist_local_rotvec_norm_deg=8.0,
    root_cube_distance_m=POSE_CONSTRAINTS.root_cube_distance_m,
    primary_anchor_candidate_id=PRIMARY_ANCHOR_CANDIDATE_ID,
    primary_anchor_fraction=0.70,
    certified_neighbor_fraction=0.30,
    certified_neighbor_edges_m=(0.082, 0.083, 0.084),
    static_samples_per_stratum=1_000,
    expansion_samples_per_stratum=2_000,
    retained_poses_per_edge=12,
    controllers_per_pose=6,
    local_seed_poses_per_edge=2,
    local_refine_per_pose=64,
    exact_candidates_per_edge=2,
    max_manipulation_pose_count=20,
    manipulation_probe_count=17,
    manipulation_candidates_per_pose=96,
    manipulation_refine_pose_count=8,
    manipulation_refine_per_pose=128,
    density_revalidation_kg_m3=DENSITY_REVALIDATION_KG_M3,
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

LARGER_RELATIVE_WRIST_POSE_ACTUAL_CONTACT_SMOOTH_VERTICAL_LIFT = (
    ExperimentDefinition(
        experiment_id=EXPERIMENT_ID,
        description=(
            "Left-hand 85--104 mm fixed-160-g cube-relative clockwise-orbit "
            "and six-DOF wrist-pose actual-contact grasp search"
        ),
        evaluation=EVALUATION,
        candidate_faces=(PRIMARY_FACE_ASSIGNMENT, FALLBACK_FACE_ASSIGNMENT),
        search_bounds=SearchBounds(
            palm_pitch_values_deg=(30.0, 32.5, 35.0, 37.5, 40.0),
            hand_roll_deg=(-10.0, 10.0),
            hand_yaw_deg=(-15.0, 15.0),
            cube_position_in_root_m=POSE_CONSTRAINTS.cube_position_in_root_m,
            cube_yaw_deg=(CUBE_YAW_DEG, CUBE_YAW_DEG),
            actuator_targets_rad=CONTACT_PRELOAD_TARGET_BOUNDS_RAD,
            seed=CAMPAIGN.seed,
            kinematic_samples_per_pitch=CAMPAIGN.quick_static_samples_per_cell,
            dynamic_candidate_count=(
                CAMPAIGN.maximum_dynamic_grasp_candidate_count
            ),
            local_refine_seed_count=CAMPAIGN.local_pose_count,
            local_refine_per_seed=CAMPAIGN.local_refine_per_pose,
            final_candidate_count=CAMPAIGN.exact_grasp_pose_count,
            perturbations_per_final_candidate=(
                CAMPAIGN.perturbations_per_trajectory
            ),
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
        relative_wrist_pose_search=RELATIVE_WRIST_POSE_SEARCH,
    )
)

EXPERIMENT_DEFINITION = (
    LARGER_RELATIVE_WRIST_POSE_ACTUAL_CONTACT_SMOOTH_VERTICAL_LIFT
)

__all__ = [
    "ARTIFACT_ROOT",
    "CAMPAIGN",
    "CLOCKWISE_ORBIT_DEG",
    "CONTACT_PRELOAD_TARGET_BOUNDS_RAD",
    "CONTROL_PROTOCOL",
    "DENSITY_REVALIDATION_KG_M3",
    "EDGES_M",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "LARGER_RELATIVE_WRIST_POSE_ACTUAL_CONTACT_SMOOTH_VERTICAL_LIFT",
    "POSE_CONSTRAINTS",
    "PRECONTACT_TARGET_BOUNDS_RAD",
    "PRIMARY_ANCHOR_CANDIDATE_ID",
    "RELATIVE_WRIST_POSE_SEARCH",
    "ROBUSTNESS",
    "SOURCE_POSE_MANIFEST",
    "THUMB_ACTUAL_CENTERS_RAD",
    "THUMB_BEND_ACTUATOR",
    "VALIDATION_LABELS",
]
