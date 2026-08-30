"""Schema-v5 far-root, high-thumb-bend fingertip grasp campaign."""

from __future__ import annotations

from xhand_grasp.experiment import (
    ContactAlignmentSettings,
    ExperimentDefinition,
    FarHandBoundaryExpansion,
    FarHandFingertipCampaignParameters,
    FarHandPerturbationEnvelope,
    FarHandPoseConstraints,
    FingertipContactPreferences,
    OpposedFaceAssignment,
    RobustnessParameters,
    SearchBounds,
)
from xhand_grasp.experiments.opposed_face_palm_down_larger_cube_relative_pose_rescue import (
    LARGER_CUBE_RELATIVE_POSE_RESCUE,
    RESCUE_MANIPULATION_DELTA_BOUNDS,
)
from xhand_grasp.experiments.opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift import (
    REFERENCE_HAND_TRANSLATION_M,
)


EXPERIMENT_ID = (
    "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift"
)
THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"

PRIMARY_FACE_ASSIGNMENT = OpposedFaceAssignment(
    thumb="-X", index="+X", mid="+X"
)
FALLBACK_FACE_ASSIGNMENT = OpposedFaceAssignment(
    thumb="-Y", index="+Y", mid="+Y"
)

FAR_HAND_POSE_CONSTRAINTS = FarHandPoseConstraints(
    finger_down_tilt_deg=(10.0, 20.0),
    palm_plane_ground_angle_deg=(10.0, 20.0),
    root_cube_distance_m=(0.138, 0.160),
    cube_position_in_root_m={
        "x": (0.084, 0.105),
        "y": (-0.035, -0.023),
        "z": (0.100, 0.120),
    },
    legacy_press_reference_translation_m=REFERENCE_HAND_TRANSLATION_M,
)

FINGERTIP_CONTACT_PREFERENCES = FingertipContactPreferences(
    taxel_assignment_max_distance_m=0.006,
)

CONTACT_ALIGNMENT = ContactAlignmentSettings(
    max_height_spread_m=0.005,
    verify_continuous_s=0.25,
    operation_aligned_duty=0.70,
)

PERTURBATION_ENVELOPE = FarHandPerturbationEnvelope(
    cube_center_xy_delta_m=(-0.0015, 0.0015),
    cube_gap_m=(0.0, 0.0005),
    cube_rpy_delta_deg=(-3.0, 3.0),
    hand_roll_yaw_delta_deg=(-1.0, 1.0),
    finger_down_tilt_delta_deg=(-1.0, 1.0),
    root_cube_distance_delta_m=(-0.001, 0.001),
    mass_scale=(0.9, 1.1),
    friction_delta=(-0.1, 0.1),
)

BOUNDARY_EXPANSION = FarHandBoundaryExpansion(
    distance_tolerance_m=0.001,
    expanded_root_cube_distance_max_m=0.165,
    expanded_cube_in_root_x_max_m=0.110,
    thumb_bend_tolerance_rad=0.03,
    thumb_bend_expand_rad=0.10,
    max_expansions=1,
)

FAR_HAND_CAMPAIGN = FarHandFingertipCampaignParameters(
    schema_version=1,
    nominal_edge_m=0.060,
    nominal_mass_kg=0.160,
    density_kg_m3=0.020 / 0.030**3,
    friction=0.8,
    post_success_edges_m=(0.059, 0.060, 0.061, 0.062, 0.063, 0.064),
    tilt_band_centers_deg=(10.0, 12.5, 15.0, 17.5, 20.0),
    primary_face=PRIMARY_FACE_ASSIGNMENT,
    fallback_face=FALLBACK_FACE_ASSIGNMENT,
    primary_static_samples_per_band=20_000,
    fallback_static_samples_per_band=4_000,
    static_retain_per_band=256,
    dynamic_candidates_per_band=128,
    grasp_refine_seed_count_per_band=4,
    grasp_refine_per_seed=128,
    manipulation_seed_count_per_band=2,
    manipulation_refine_per_seed=128,
    exact_candidates_per_band=16,
    closure_alpha_values=(0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 1.0),
    target_signed_gap_m=(-0.0005, 0.003),
    max_static_distal_preload_m=0.010,
    concentrated_thumb_bend_fraction=0.70,
    concentrated_thumb_bend_rad=(1.00, 1.35),
    minimum_manipulated_thumb_bend_rad=0.75,
    perturbation_envelope=PERTURBATION_ENVELOPE,
    boundary_expansion=BOUNDARY_EXPANSION,
)

FAR_HAND_ROBUSTNESS = RobustnessParameters(
    edge_m=FAR_HAND_CAMPAIGN.post_success_edges_m,
    mass_kg=(FAR_HAND_CAMPAIGN.nominal_mass_kg,),
    friction=(FAR_HAND_CAMPAIGN.friction,),
    perturbation_count=50,
    required_pass_count=45,
    seed=20260821,
    position_xy_delta_m=PERTURBATION_ENVELOPE.cube_center_xy_delta_m,
    rpy_delta_deg=PERTURBATION_ENVELOPE.cube_rpy_delta_deg,
    mass_scale=PERTURBATION_ENVELOPE.mass_scale,
    friction_delta=PERTURBATION_ENVELOPE.friction_delta,
    z_offset_delta_m=PERTURBATION_ENVELOPE.cube_gap_m,
)

ACTUATOR_TARGET_BOUNDS_RAD = {
    THUMB_BEND_ACTUATOR: (0.85, 1.35),
    "left_hand_thumb_rota_joint1_actuator": (0.35, 0.80),
    "left_hand_thumb_rota_joint2_actuator": (0.50, 1.05),
    "left_hand_index_bend_joint_actuator": (-0.08, 0.08),
    "left_hand_index_joint1_actuator": (0.30, 0.70),
    "left_hand_index_joint2_actuator": (1.35, 1.80),
    "left_hand_mid_joint1_actuator": (0.50, 1.00),
    "left_hand_mid_joint2_actuator": (1.10, 1.70),
}

_SOURCE = LARGER_CUBE_RELATIVE_POSE_RESCUE

FAR_HAND_FINGERTIP_GRASP_THEN_LIFT = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand 60 mm opposed-face grasp with a 138--160 mm fixed root "
        "distance, high thumb-bend target and soft fingertip-pad preference"
    ),
    evaluation=_SOURCE.evaluation,
    candidate_faces=(PRIMARY_FACE_ASSIGNMENT, FALLBACK_FACE_ASSIGNMENT),
    search_bounds=SearchBounds(
        # These values are geometric tilt-band labels, as in schema v4.
        palm_pitch_values_deg=FAR_HAND_CAMPAIGN.tilt_band_centers_deg,
        hand_roll_deg=(-2.0, 3.0),
        hand_yaw_deg=(-5.0, 0.0),
        cube_position_in_root_m=FAR_HAND_POSE_CONSTRAINTS.cube_position_in_root_m,
        cube_yaw_deg=(24.0, 38.0),
        actuator_targets_rad=ACTUATOR_TARGET_BOUNDS_RAD,
        seed=20260821,
        kinematic_samples_per_pitch=FAR_HAND_CAMPAIGN.static_samples_per_band,
        dynamic_candidate_count=FAR_HAND_CAMPAIGN.dynamic_candidate_count,
        local_refine_seed_count=(
            len(FAR_HAND_CAMPAIGN.tilt_band_centers_deg)
            * FAR_HAND_CAMPAIGN.grasp_refine_seed_count_per_band
        ),
        local_refine_per_seed=FAR_HAND_CAMPAIGN.grasp_refine_per_seed,
        final_candidate_count=FAR_HAND_CAMPAIGN.exact_candidate_count,
        perturbations_per_final_candidate=FAR_HAND_ROBUSTNESS.perturbation_count,
        fallback_kinematic_samples_per_pitch=(
            FAR_HAND_CAMPAIGN.fallback_static_samples_per_band
        ),
        fallback_candidate_count=FAR_HAND_CAMPAIGN.dynamic_candidates_per_band,
        final_target_delta_rad=None,
        manipulation_delta_rad=RESCUE_MANIPULATION_DELTA_BOUNDS,
    ),
    robustness=FAR_HAND_ROBUSTNESS,
    artifact_root=(
        "artifacts/"
        "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift"
    ),
    control_protocol=_SOURCE.control_protocol,
    tuning_strategy="far_hand_fingertip",
    contact_alignment=CONTACT_ALIGNMENT,
    far_hand_pose_constraints=FAR_HAND_POSE_CONSTRAINTS,
    fingertip_contact_preferences=FINGERTIP_CONTACT_PREFERENCES,
    far_hand_campaign=FAR_HAND_CAMPAIGN,
)

EXPERIMENT_DEFINITION = FAR_HAND_FINGERTIP_GRASP_THEN_LIFT

__all__ = [
    "ACTUATOR_TARGET_BOUNDS_RAD",
    "BOUNDARY_EXPANSION",
    "CONTACT_ALIGNMENT",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "FALLBACK_FACE_ASSIGNMENT",
    "FAR_HAND_CAMPAIGN",
    "FAR_HAND_FINGERTIP_GRASP_THEN_LIFT",
    "FAR_HAND_POSE_CONSTRAINTS",
    "FAR_HAND_ROBUSTNESS",
    "FINGERTIP_CONTACT_PREFERENCES",
    "PERTURBATION_ENVELOPE",
    "PRIMARY_FACE_ASSIGNMENT",
    "THUMB_BEND_ACTUATOR",
]
