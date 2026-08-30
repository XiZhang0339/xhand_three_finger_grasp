"""Schema-v13 scaled centered/spread contact grasp and lift experiment."""

from __future__ import annotations

from xhand_grasp.experiment import (
    ActualContactGraspPoseCampaignParameters,
    ExperimentDefinition,
    RobustnessParameters,
    SCALED_CONTACT_MAPPING_MODES,
    ScaledContactDownsizeCampaignParameters,
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
from xhand_grasp.experiments.opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_smooth_vertical_lift import (
    CONTACT_PRELOAD_TARGET_BOUNDS_RAD,
    CONTROL_PROTOCOL,
    POSE_CONSTRAINTS,
    PRECONTACT_TARGET_BOUNDS_RAD,
)
from xhand_grasp.experiments.opposed_face_palm_down_pose_preserving_grasp import (
    EVALUATION,
    POSE_PRESERVATION,
    POSE_PRESERVING_GRASP as V6_EXPERIMENT,
)
from xhand_grasp.experiments.opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift import (
    FINGERTIP_CONTACT_PREFERENCES,
    PRIMARY_FACE_ASSIGNMENT,
)


EXPERIMENT_ID = (
    "left_opposed_face_palm_down_scaled_centered_spread_actual_grasp_then_lift"
)
REFERENCE_EDGE_M = 0.089
EDGES_M = tuple(value / 1000.0 for value in range(60, 89))
MASS_KG = 0.160
FRICTION = 0.8
SEED = 20260821
SOURCE_ALIASES = ("best_nominal", "best_thumb_center", "best_pair_center")
SOURCE_CATALOG = (
    "artifacts/left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift/tune/centered_thumb_expanded_opposing_spread_from_"
    "positive_y_best_v1/catalog.json"
)
ARTIFACT_ROOT = (
    "artifacts/left_opposed_face_palm_down_scaled_centered_spread_actual_"
    "grasp_then_lift"
)
SOURCE_MANIFEST = (
    ARTIFACT_ROOT
    + "/source_manifests/centered_spread_authenticated_sources.json"
)

VALIDATION_LABELS = {
    "grasp": "validated_fixed_160g_scaled_grasp_ablation",
    "manipulation": "validated_fixed_160g_scaled_lift_ablation",
    "robust": "validated_fixed_160g_scaled_robust_full_success_ablation",
}

SCALED_CONTACT_DOWNSIZE_CAMPAIGN = ScaledContactDownsizeCampaignParameters(
    schema_version=1,
    reference_edge_m=REFERENCE_EDGE_M,
    edges_m=EDGES_M,
    edge_step_m=0.001,
    fixed_mass_kg=MASS_KG,
    friction=FRICTION,
    seed=SEED,
    source_catalog=SOURCE_CATALOG,
    source_manifest=SOURCE_MANIFEST,
    source_aliases=SOURCE_ALIASES,
    mapping_modes=SCALED_CONTACT_MAPPING_MODES,
    contact_target_radius_m=0.002,
    minimum_edge_guard_m=0.0005,
    dls_starts_per_stratum=4,
    max_dls_iterations=12,
    static_retain_per_stratum=2,
    controllers_per_pose=6,
    local_poses_per_edge_mapping=2,
    local_refine_per_pose=32,
    selected_grasps_per_edge=3,
    manipulation_probe_count=17,
    manipulation_candidates_per_grasp=64,
    manipulation_edge_refine_per_best=32,
    global_refine_pose_count=8,
    global_refine_per_pose=128,
    perturbations_per_published=16,
    robustness_trials=50,
    robustness_required_passes=45,
)

# The generic actual-qpos campaign expresses the same aggregate quotas used by
# the scaled runner: two mapping/source aggregates per edge, six retained
# source/mapping poses per edge and six controllers per pose.
CAMPAIGN = ActualContactGraspPoseCampaignParameters(
    schema_version=1,
    edges_m=EDGES_M,
    thumb_actual_centers_rad=(1.40, 1.60),
    fixed_mass_kg=MASS_KG,
    friction=FRICTION,
    cube_center_xy_m=(0.071, -0.027),
    cube_yaw_deg=CUBE_YAW_DEG,
    seed=SEED,
    witness_signed_gap_m=(-0.0005, 0.00025),
    min_witness_normal_alignment=0.95,
    min_witness_edge_margin_m=0.0005,
    max_contact_height_spread_m=0.005,
    precontact_retreat_m=(0.002, 0.004),
    source_pose_manifest=SOURCE_MANIFEST,
    quick_static_samples_per_cell=4,
    quick_static_retain_per_cell=2,
    full_static_samples_per_cell=12,
    full_static_retain_per_cell=6,
    controller_seeds_per_pose=6,
    local_pose_count=SCALED_CONTACT_DOWNSIZE_CAMPAIGN.local_pose_count,
    local_refine_per_pose=32,
    exact_grasp_pose_count=SCALED_CONTACT_DOWNSIZE_CAMPAIGN.selected_grasp_count,
    manipulation_probe_count=17,
    manipulation_candidates_per_pose=64,
    manipulation_refine_pose_count=8,
    manipulation_refine_per_pose=128,
    perturbations_per_trajectory=16,
    selected_trajectory_count=SCALED_CONTACT_DOWNSIZE_CAMPAIGN.selected_grasp_count,
    minimum_distinct_edges=3,
    minimum_thumb_bands=2,
    initial_target_success_count=1,
    validation_labels=VALIDATION_LABELS,
    grasp_pose_identity=(
        "cube_hand_topology_contact_point_plan_nominal_qpos_sha256"
    ),
)

ROBUSTNESS = RobustnessParameters(
    edge_m=EDGES_M,
    mass_kg=(MASS_KG,),
    friction=(FRICTION,),
    perturbation_count=50,
    required_pass_count=45,
    seed=SEED,
    position_xy_delta_m=(-0.0015, 0.0015),
    rpy_delta_deg=(-3.0, 3.0),
    mass_scale=(1.0, 1.0),
    friction_delta=(-0.1, 0.1),
    z_offset_delta_m=(0.0, 0.0005),
)

SCALED_CENTERED_SPREAD_ACTUAL_GRASP_THEN_LIFT = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand 60--88 mm fixed-160-g continuation of authenticated "
        "centered/spread contact plans into actual grasp and smooth lift"
    ),
    evaluation=EVALUATION,
    candidate_faces=(PRIMARY_FACE_ASSIGNMENT,),
    search_bounds=SearchBounds(
        palm_pitch_values_deg=(30.0, 32.5, 35.0, 37.5, 40.0),
        hand_roll_deg=(-10.0, 10.0),
        hand_yaw_deg=(-15.0, 15.0),
        cube_position_in_root_m=POSE_CONSTRAINTS.cube_position_in_root_m,
        cube_yaw_deg=(CUBE_YAW_DEG, CUBE_YAW_DEG),
        actuator_targets_rad=CONTACT_PRELOAD_TARGET_BOUNDS_RAD,
        seed=SEED,
        kinematic_samples_per_pitch=(
            SCALED_CONTACT_DOWNSIZE_CAMPAIGN.dls_starts_per_stratum
        ),
        dynamic_candidate_count=(
            SCALED_CONTACT_DOWNSIZE_CAMPAIGN.maximum_dynamic_grasp_candidate_count
        ),
        local_refine_seed_count=SCALED_CONTACT_DOWNSIZE_CAMPAIGN.local_pose_count,
        local_refine_per_seed=32,
        final_candidate_count=SCALED_CONTACT_DOWNSIZE_CAMPAIGN.selected_grasp_count,
        perturbations_per_final_candidate=16,
        fallback_kinematic_samples_per_pitch=12,
        fallback_candidate_count=6,
        final_target_delta_rad=None,
        manipulation_delta_rad=MANIPULATION_DELTA_BOUNDS_RAD,
        pregrasp_targets_rad=PRECONTACT_TARGET_BOUNDS_RAD,
    ),
    robustness=ROBUSTNESS,
    artifact_root=ARTIFACT_ROOT,
    control_protocol=CONTROL_PROTOCOL,
    tuning_strategy="scaled_contact_downsize_actual_grasp_then_lift",
    contact_alignment=V6_EXPERIMENT.contact_alignment,
    far_hand_pose_constraints=POSE_CONSTRAINTS,
    fingertip_contact_preferences=FINGERTIP_CONTACT_PREFERENCES,
    pose_preservation=POSE_PRESERVATION,
    closure_alignment=CLOSURE_ALIGNMENT,
    motion_smoothness=MOTION_SMOOTHNESS,
    actual_contact_grasp_pose=GRASP_POSE,
    actual_contact_grasp_pose_campaign=CAMPAIGN,
    scaled_contact_downsize_campaign=SCALED_CONTACT_DOWNSIZE_CAMPAIGN,
)

EXPERIMENT_DEFINITION = SCALED_CENTERED_SPREAD_ACTUAL_GRASP_THEN_LIFT

__all__ = [
    "ARTIFACT_ROOT",
    "CAMPAIGN",
    "EDGES_M",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "FRICTION",
    "MASS_KG",
    "REFERENCE_EDGE_M",
    "ROBUSTNESS",
    "SCALED_CENTERED_SPREAD_ACTUAL_GRASP_THEN_LIFT",
    "SCALED_CONTACT_DOWNSIZE_CAMPAIGN",
    "SEED",
    "SOURCE_ALIASES",
    "SOURCE_CATALOG",
    "SOURCE_MANIFEST",
    "VALIDATION_LABELS",
]
