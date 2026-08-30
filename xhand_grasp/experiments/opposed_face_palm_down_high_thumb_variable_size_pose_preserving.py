"""Schema-v7 high-thumb, variable-size, pose-preserving grasp campaign."""

from __future__ import annotations

from xhand_grasp.experiment import (
    ExperimentDefinition,
    FarHandPoseConstraints,
    HighThumbSizeCampaignParameters,
    RobustnessParameters,
    SearchBounds,
)
from xhand_grasp.experiments.opposed_face_palm_down_larger_cube_relative_pose_rescue import (
    RESCUE_MANIPULATION_DELTA_BOUNDS,
)
from xhand_grasp.experiments.opposed_face_palm_down_pose_preserving_grasp import (
    EVALUATION,
    POSE_PRESERVATION,
    POSE_PRESERVING_GRASP as V6_EXPERIMENT,
    PREGRASP_TARGET_BOUNDS_RAD as V6_PREGRASP_TARGET_BOUNDS_RAD,
)
from xhand_grasp.experiments.opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift import (
    FALLBACK_FACE_ASSIGNMENT,
    FINGERTIP_CONTACT_PREFERENCES,
    PRIMARY_FACE_ASSIGNMENT,
)


EXPERIMENT_ID = (
    "left_opposed_face_palm_down_high_thumb_variable_size_"
    "pose_preserving_grasp_then_lift"
)
THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"

COARSE_EDGES_M = tuple(value / 1000.0 for value in range(52, 71, 2))
COARSE_THUMB_TARGETS_RAD = (1.25, 1.30, 1.35, 1.40, 1.45)

# Keep the seven established v6 search ranges while making high thumb bend an
# explicit, non-expandable schema-v7 dimension.
GRASP_TARGET_BOUNDS_RAD = {
    **dict(V6_EXPERIMENT.search_bounds.actuator_targets_rad),
    THUMB_BEND_ACTUATOR: (
        COARSE_THUMB_TARGETS_RAD[0],
        COARSE_THUMB_TARGETS_RAD[-1],
    ),
}
PREGRASP_TARGET_BOUNDS_RAD = {
    **dict(V6_PREGRASP_TARGET_BOUNDS_RAD),
    THUMB_BEND_ACTUATOR: (0.60, 1.30),
}
MANIPULATION_DELTA_BOUNDS_RAD = dict(RESCUE_MANIPULATION_DELTA_BOUNDS)

POSE_CONSTRAINTS = FarHandPoseConstraints(
    finger_down_tilt_deg=(30.0, 40.0),
    palm_plane_ground_angle_deg=(30.0, 40.0),
    root_cube_distance_m=(0.138, 0.165),
    cube_position_in_root_m={
        "x": (0.090, 0.115),
        "y": (-0.040, -0.010),
        "z": (0.095, 0.125),
    },
    legacy_press_reference_translation_m=(
        -0.036454669709448065,
        0.008499986928773153,
        0.19777491509344874,
    ),
)

HIGH_THUMB_SIZE_CAMPAIGN = HighThumbSizeCampaignParameters(
    schema_version=1,
    coarse_edges_m=COARSE_EDGES_M,
    coarse_thumb_targets_rad=COARSE_THUMB_TARGETS_RAD,
    edge_fine_step_m=0.001,
    thumb_fine_step_rad=0.01,
    fixed_mass_kg=0.160,
    friction=0.8,
    seed=20260821,
    cube_center_xy_m=(0.071, -0.027),
    grasp_target_bounds_rad=GRASP_TARGET_BOUNDS_RAD,
    pregrasp_target_bounds_rad=PREGRASP_TARGET_BOUNDS_RAD,
    manipulation_delta_bounds_rad=MANIPULATION_DELTA_BOUNDS_RAD,
    source_seed_family_count=6,
    static_samples_per_cell=20_000,
    static_retain_per_cell=8,
    dynamic_candidate_count=400,
    local_sizes_per_thumb_band=4,
    local_seeds_per_size=2,
    local_refine_per_seed=64,
    fine_dynamic_candidate_count=320,
    exact_candidate_count=48,
    selected_grasp_count=12,
    selected_lift_seed_count=6,
    perturbations_per_grasp=16,
    manipulation_candidates_per_lift_seed=256,
    minimum_distinct_edges=4,
    minimum_seed_families=3,
    minimum_per_thumb_band=3,
    maximum_per_edge_target_pair=2,
    thumb_selection_bands_rad=((1.25, 1.31), (1.31, 1.38), (1.38, 1.45)),
)

ROBUSTNESS = RobustnessParameters(
    edge_m=COARSE_EDGES_M,
    mass_kg=(HIGH_THUMB_SIZE_CAMPAIGN.fixed_mass_kg,),
    friction=(HIGH_THUMB_SIZE_CAMPAIGN.friction,),
    perturbation_count=50,
    required_pass_count=45,
    seed=HIGH_THUMB_SIZE_CAMPAIGN.seed,
    position_xy_delta_m=(-0.0015, 0.0015),
    rpy_delta_deg=(-3.0, 3.0),
    mass_scale=(0.9, 1.1),
    friction_delta=(-0.1, 0.1),
    z_offset_delta_m=(0.0, 0.0005),
)

HIGH_THUMB_VARIABLE_SIZE_POSE_PRESERVING = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand 52--70 mm fixed-mass grasp campaign with 1.25--1.45 rad "
        "thumb bend and initial free-cube pose preservation"
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
        seed=HIGH_THUMB_SIZE_CAMPAIGN.seed,
        kinematic_samples_per_pitch=(
            HIGH_THUMB_SIZE_CAMPAIGN.static_samples_per_cell
        ),
        dynamic_candidate_count=(
            HIGH_THUMB_SIZE_CAMPAIGN.dynamic_candidate_count
        ),
        local_refine_seed_count=(
            HIGH_THUMB_SIZE_CAMPAIGN.local_refine_seed_count
        ),
        local_refine_per_seed=HIGH_THUMB_SIZE_CAMPAIGN.local_refine_per_seed,
        final_candidate_count=HIGH_THUMB_SIZE_CAMPAIGN.exact_candidate_count,
        perturbations_per_final_candidate=(
            HIGH_THUMB_SIZE_CAMPAIGN.perturbations_per_grasp
        ),
        fallback_kinematic_samples_per_pitch=(
            HIGH_THUMB_SIZE_CAMPAIGN.static_samples_per_cell
        ),
        fallback_candidate_count=(
            HIGH_THUMB_SIZE_CAMPAIGN.static_retain_per_cell
        ),
        final_target_delta_rad=None,
        manipulation_delta_rad=MANIPULATION_DELTA_BOUNDS_RAD,
        pregrasp_targets_rad=PREGRASP_TARGET_BOUNDS_RAD,
    ),
    robustness=ROBUSTNESS,
    artifact_root=(
        "artifacts/"
        "left_opposed_face_palm_down_high_thumb_variable_size_pose_preserving"
    ),
    control_protocol=V6_EXPERIMENT.control_protocol,
    tuning_strategy="high_thumb_variable_size_pose_preserving",
    contact_alignment=V6_EXPERIMENT.contact_alignment,
    far_hand_pose_constraints=POSE_CONSTRAINTS,
    fingertip_contact_preferences=FINGERTIP_CONTACT_PREFERENCES,
    pose_preservation=POSE_PRESERVATION,
    high_thumb_size_campaign=HIGH_THUMB_SIZE_CAMPAIGN,
)

EXPERIMENT_DEFINITION = HIGH_THUMB_VARIABLE_SIZE_POSE_PRESERVING

__all__ = [
    "COARSE_EDGES_M",
    "COARSE_THUMB_TARGETS_RAD",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "GRASP_TARGET_BOUNDS_RAD",
    "HIGH_THUMB_SIZE_CAMPAIGN",
    "HIGH_THUMB_VARIABLE_SIZE_POSE_PRESERVING",
    "MANIPULATION_DELTA_BOUNDS_RAD",
    "POSE_CONSTRAINTS",
    "PREGRASP_TARGET_BOUNDS_RAD",
    "ROBUSTNESS",
    "THUMB_BEND_ACTUATOR",
]
