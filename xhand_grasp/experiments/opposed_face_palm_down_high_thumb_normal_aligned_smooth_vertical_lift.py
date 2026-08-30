"""Schema-v8 normal-aligned grasp and smooth near-vertical lift experiment."""

from __future__ import annotations

from xhand_grasp.experiment import (
    ClosureAlignmentSettings,
    ControlProtocolSettings,
    ExperimentDefinition,
    MotionSmoothnessSettings,
    NormalAlignedSmoothLiftCampaignParameters,
    RobustnessParameters,
    SearchBounds,
)
from xhand_grasp.experiments.opposed_face_palm_down_high_thumb_variable_size_pose_preserving import (
    GRASP_TARGET_BOUNDS_RAD,
    MANIPULATION_DELTA_BOUNDS_RAD,
    POSE_CONSTRAINTS,
    PREGRASP_TARGET_BOUNDS_RAD,
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
    "left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift"
)
THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
EDGES_M = tuple(value / 1000.0 for value in range(60, 71))
THUMB_TARGETS_RAD = (1.25, 1.30, 1.35, 1.40, 1.45)

CLOSURE_ALIGNMENT = ClosureAlignmentSettings(
    static_max_angle_deg=45.0,
    dynamic_p95_max_angle_deg=30.0,
    optimization_target_max_angle_deg=20.0,
    min_inward_speed_m_s=0.0,
    min_contact_force_n=0.05,
)

MOTION_SMOOTHNESS = MotionSmoothnessSettings(
    filter_window_s=0.051,
    downward_speed_threshold_m_s=0.001,
    max_downward_speed_duty=0.02,
    max_cumulative_backtrack_m=0.0002,
    max_peak_upward_speed_m_s=0.020,
    max_abs_acceleration_m_s2=0.12,
    max_abs_jerk_m_s3=2.5,
    max_hold_entry_linear_speed_m_s=0.005,
    max_lateral_displacement_m=0.002,
    max_operation_orientation_drift_deg=10.0,
)

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
    close_duration_options_s=(1.0, 1.25, 1.5),
)

CAMPAIGN = NormalAlignedSmoothLiftCampaignParameters(
    schema_version=1,
    edges_m=EDGES_M,
    thumb_targets_rad=THUMB_TARGETS_RAD,
    fixed_mass_kg=0.160,
    friction=0.8,
    cube_center_xy_m=(0.071, -0.027),
    seed=20260821,
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

NORMAL_ALIGNED_SMOOTH_VERTICAL_LIFT = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand 60--70 mm high-thumb grasp with command-normal alignment "
        "and minimum-jerk near-vertical manipulation"
    ),
    evaluation=EVALUATION,
    candidate_faces=(PRIMARY_FACE_ASSIGNMENT, FALLBACK_FACE_ASSIGNMENT),
    search_bounds=SearchBounds(
        palm_pitch_values_deg=(30.0, 32.5, 35.0, 37.5, 40.0),
        hand_roll_deg=(-7.0, 3.0),
        hand_yaw_deg=(-2.0, 8.0),
        cube_position_in_root_m=POSE_CONSTRAINTS.cube_position_in_root_m,
        cube_yaw_deg=(24.0, 38.0),
        actuator_targets_rad=GRASP_TARGET_BOUNDS_RAD,
        seed=CAMPAIGN.seed,
        kinematic_samples_per_pitch=CAMPAIGN.static_samples_per_cell,
        dynamic_candidate_count=(
            CAMPAIGN.cell_count
            * CAMPAIGN.static_retain_per_cell
            * CAMPAIGN.controller_seeds_per_pose
        ),
        local_refine_seed_count=CAMPAIGN.local_pose_count,
        local_refine_per_seed=CAMPAIGN.local_refine_per_pose,
        final_candidate_count=CAMPAIGN.exact_candidate_count,
        perturbations_per_final_candidate=16,
        fallback_kinematic_samples_per_pitch=CAMPAIGN.static_samples_per_cell,
        fallback_candidate_count=CAMPAIGN.static_retain_per_cell,
        final_target_delta_rad=None,
        manipulation_delta_rad=MANIPULATION_DELTA_BOUNDS_RAD,
        pregrasp_targets_rad=PREGRASP_TARGET_BOUNDS_RAD,
    ),
    robustness=ROBUSTNESS,
    artifact_root=(
        "artifacts/"
        "left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift"
    ),
    control_protocol=CONTROL_PROTOCOL,
    tuning_strategy="normal_aligned_smooth_vertical_lift",
    contact_alignment=V6_EXPERIMENT.contact_alignment,
    far_hand_pose_constraints=POSE_CONSTRAINTS,
    fingertip_contact_preferences=FINGERTIP_CONTACT_PREFERENCES,
    pose_preservation=POSE_PRESERVATION,
    closure_alignment=CLOSURE_ALIGNMENT,
    motion_smoothness=MOTION_SMOOTHNESS,
    normal_aligned_smooth_lift_campaign=CAMPAIGN,
)

EXPERIMENT_DEFINITION = NORMAL_ALIGNED_SMOOTH_VERTICAL_LIFT

__all__ = [
    "CAMPAIGN",
    "CLOSURE_ALIGNMENT",
    "CONTROL_PROTOCOL",
    "EDGES_M",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "MOTION_SMOOTHNESS",
    "NORMAL_ALIGNED_SMOOTH_VERTICAL_LIFT",
    "ROBUSTNESS",
    "THUMB_TARGETS_RAD",
]
