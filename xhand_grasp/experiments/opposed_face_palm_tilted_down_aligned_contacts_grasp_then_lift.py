"""Tilted-palm opposed-face campaign with height-aligned contacts."""

from __future__ import annotations

from xhand_grasp.experiment import (
    AlignedContactCampaignParameters,
    AlignedContactPerturbationEnvelope,
    ContactAlignmentSettings,
    ExperimentDefinition,
    PoseConstraints,
    RobustnessParameters,
    SearchBounds,
)
from xhand_grasp.experiments.opposed_face_palm_down import SEARCH_FACE_ASSIGNMENTS
from xhand_grasp.experiments.opposed_face_palm_down_larger_cube_relative_pose_rescue import (
    LARGER_CUBE_RELATIVE_POSE_RESCUE,
)


EXPERIMENT_ID = (
    "left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift"
)

# This is the resolved translation of the validated nominal 62 mm source
# configuration.  V4 searches press depth relative to this immutable anchor.
REFERENCE_HAND_TRANSLATION_M = (
    -0.036454669709448065,
    0.008499986928773153,
    0.19777491509344874,
)

POSE_CONSTRAINTS = PoseConstraints(
    finger_down_tilt_deg=(10.0, 20.0),
    palm_plane_ground_angle_deg=(10.0, 20.0),
    palm_press_depth_m=(0.002, 0.008),
    reference_hand_translation_m=REFERENCE_HAND_TRANSLATION_M,
)

CONTACT_ALIGNMENT = ContactAlignmentSettings(
    max_height_spread_m=0.005,
    verify_continuous_s=0.25,
    operation_aligned_duty=0.70,
)

ALIGNED_CONTACT_PERTURBATION_ENVELOPE = AlignedContactPerturbationEnvelope(
    cube_center_xy_delta_m=(-0.0015, 0.0015),
    cube_gap_m=(0.0, 0.0005),
    cube_rpy_delta_deg=(-3.0, 3.0),
    hand_roll_yaw_delta_deg=(-1.0, 1.0),
    finger_down_tilt_delta_deg=(-1.0, 1.0),
    palm_press_depth_delta_m=(-0.001, 0.001),
    mass_scale=(0.9, 1.1),
    friction_delta=(-0.1, 0.1),
)

ALIGNED_CONTACT_CAMPAIGN = AlignedContactCampaignParameters(
    schema_version=1,
    edges_m=(0.059, 0.060, 0.061, 0.062, 0.063, 0.064),
    density_kg_m3=0.020 / 0.030**3,
    friction=0.8,
    tilt_band_centers_deg=(10.0, 12.5, 15.0, 17.5, 20.0),
    static_samples_per_edge_band=20_000,
    dynamic_candidates_per_band=128,
    grasp_refine_seed_count_per_band=4,
    grasp_refine_per_seed=128,
    manipulation_seed_count_per_band=2,
    manipulation_refine_per_seed=128,
    exact_candidates_per_band=16,
    perturbation_envelope=ALIGNED_CONTACT_PERTURBATION_ENVELOPE,
)

ALIGNED_CONTACT_ROBUSTNESS = RobustnessParameters(
    edge_m=ALIGNED_CONTACT_CAMPAIGN.edges_m,
    mass_kg=(ALIGNED_CONTACT_CAMPAIGN.constant_density_mass_kg(0.062),),
    friction=(ALIGNED_CONTACT_CAMPAIGN.friction,),
    perturbation_count=50,
    required_pass_count=45,
    seed=20260821,
    position_xy_delta_m=(-0.0015, 0.0015),
    rpy_delta_deg=(-3.0, 3.0),
    mass_scale=(0.9, 1.1),
    friction_delta=(-0.1, 0.1),
    z_offset_delta_m=(0.0, 0.0005),
)

_SOURCE = LARGER_CUBE_RELATIVE_POSE_RESCUE
_SOURCE_BOUNDS = _SOURCE.search_bounds

OPPOSED_FACE_PALM_TILTED_DOWN_ALIGNED_CONTACTS_GRASP_THEN_LIFT = (
    ExperimentDefinition(
        experiment_id=EXPERIMENT_ID,
        description=(
            "Left-hand 59--64 mm constant-density opposed-face grasp with a "
            "10--20 degree downward palm tilt and continuously height-aligned "
            "thumb, index and middle contacts before and during lift"
        ),
        evaluation=_SOURCE.evaluation,
        candidate_faces=SEARCH_FACE_ASSIGNMENTS,
        search_bounds=SearchBounds(
            # In schema v4 these values identify geometric tilt bands; the
            # actual admissible pose is declared by POSE_CONSTRAINTS.
            palm_pitch_values_deg=ALIGNED_CONTACT_CAMPAIGN.tilt_band_centers_deg,
            hand_roll_deg=(-5.0, 5.0),
            hand_yaw_deg=(-8.0, 2.0),
            cube_position_in_root_m={
                # X is solved from press depth after Y/Z and orientation are
                # sampled; this envelope covers all declared corner cases.
                "x": (0.035, 0.067),
                "y": (-0.036, -0.022),
                "z": (0.101, 0.117),
            },
            cube_yaw_deg=(20.0, 40.0),
            actuator_targets_rad=_SOURCE_BOUNDS.actuator_targets_rad,
            seed=20260821,
            kinematic_samples_per_pitch=(
                ALIGNED_CONTACT_CAMPAIGN.static_samples_per_edge_band
            ),
            dynamic_candidate_count=(
                ALIGNED_CONTACT_CAMPAIGN.dynamic_candidate_count
            ),
            local_refine_seed_count=(
                len(ALIGNED_CONTACT_CAMPAIGN.tilt_band_centers_deg)
                * ALIGNED_CONTACT_CAMPAIGN.grasp_refine_seed_count_per_band
            ),
            local_refine_per_seed=ALIGNED_CONTACT_CAMPAIGN.grasp_refine_per_seed,
            final_candidate_count=ALIGNED_CONTACT_CAMPAIGN.exact_candidate_count,
            perturbations_per_final_candidate=(
                ALIGNED_CONTACT_ROBUSTNESS.perturbation_count
            ),
            fallback_kinematic_samples_per_pitch=(
                ALIGNED_CONTACT_CAMPAIGN.static_samples_per_edge_band
            ),
            fallback_candidate_count=(
                ALIGNED_CONTACT_CAMPAIGN.dynamic_candidate_count
            ),
            final_target_delta_rad=None,
            manipulation_delta_rad=_SOURCE_BOUNDS.manipulation_delta_rad,
        ),
        robustness=ALIGNED_CONTACT_ROBUSTNESS,
        artifact_root=(
            "artifacts/"
            "left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift"
        ),
        control_protocol=_SOURCE.control_protocol,
        tuning_strategy="aligned_contacts",
        pose_constraints=POSE_CONSTRAINTS,
        contact_alignment=CONTACT_ALIGNMENT,
        aligned_contact_campaign=ALIGNED_CONTACT_CAMPAIGN,
    )
)

# Compact aliases for callers and consistency with earlier experiment modules.
TILTED_DOWN_ALIGNED_CONTACTS_GRASP_THEN_LIFT = (
    OPPOSED_FACE_PALM_TILTED_DOWN_ALIGNED_CONTACTS_GRASP_THEN_LIFT
)
EXPERIMENT_DEFINITION = (
    OPPOSED_FACE_PALM_TILTED_DOWN_ALIGNED_CONTACTS_GRASP_THEN_LIFT
)

__all__ = [
    "ALIGNED_CONTACT_CAMPAIGN",
    "ALIGNED_CONTACT_PERTURBATION_ENVELOPE",
    "ALIGNED_CONTACT_ROBUSTNESS",
    "CONTACT_ALIGNMENT",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "OPPOSED_FACE_PALM_TILTED_DOWN_ALIGNED_CONTACTS_GRASP_THEN_LIFT",
    "POSE_CONSTRAINTS",
    "REFERENCE_HAND_TRANSLATION_M",
    "TILTED_DOWN_ALIGNED_CONTACTS_GRASP_THEN_LIFT",
]
