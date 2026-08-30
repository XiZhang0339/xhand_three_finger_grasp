"""Schema-v12 fixed-90-mm contact-point-targeted grasp experiment."""

from __future__ import annotations

from xhand_grasp.experiment import (
    ActualContactGraspPoseCampaignParameters,
    ContactPointPlanParameters,
    ContactPointSearchParameters,
    ExperimentDefinition,
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
    "left_opposed_face_palm_down_90mm_contact_point_targeted_actual_grasp_pose"
)
EDGE_M = 0.090
MASS_KG = 0.160
FRICTION = 0.8
SEED = 20260821
SIGNED_CLOCKWISE_ORBIT_DEG = (-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5)
SOURCE_POSE_CANDIDATE_IDS = (
    4367930796723148114,
    4370955631788604117,
)
ARTIFACT_ROOT = (
    "artifacts/"
    "left_opposed_face_palm_down_90mm_contact_point_targeted_actual_grasp_pose"
)
SOURCE_POSE_MANIFEST = (
    ARTIFACT_ROOT
    + "/source_manifests/v11_nearest_certified_grasp_sources.json"
)

VALIDATION_LABELS = {
    "grasp": "validated_fixed_160g_grasp_ablation",
    "manipulation": "diagnostic_fixed_160g_manipulability_prefilter",
    "robust": "validated_fixed_160g_robust_grasp_ablation",
}

SEED_CONTACT_POINT_PLAN = ContactPointPlanParameters(
    schema_version=1,
    cube_edge_m=EDGE_M,
    target_faces=PRIMARY_FACE_ASSIGNMENT,
    target_face_yz_m={
        "thumb": (0.010, 0.012),
        "index": (0.002, 0.012),
        "mid": (0.021, 0.012),
    },
    target_radius_m=0.002,
)

CONTACT_POINT_SEARCH = ContactPointSearchParameters(
    schema_version=1,
    seed_plan=SEED_CONTACT_POINT_PLAN,
    seed=SEED,
    face_yz_delta_m=(-0.008, 0.008),
    point_group_sample_count=120_000,
    retained_point_plan_count=256,
    signed_clockwise_orbit_deg=SIGNED_CLOCKWISE_ORBIT_DEG,
    source_pose_candidate_ids=SOURCE_POSE_CANDIDATE_IDS,
    max_reachability_screen_count=3_584,
    min_target_edge_margin_m=0.020,
    max_target_height_spread_m=0.005,
    min_index_middle_separation_m=0.010,
    static_target_error_radius_m=0.004,
    retained_static_pose_count=64,
    controllers_per_pose=6,
    local_pose_count=16,
    local_refine_per_pose=128,
    exact_candidate_count=16,
    selected_trajectory_count=5,
    perturbations_per_trajectory=16,
    manipulation_probe_count=17,
    manipulation_max_delta_rad=0.05,
    virtual_lift_target_m=0.002,
)

CAMPAIGN = ActualContactGraspPoseCampaignParameters(
    schema_version=1,
    edges_m=(EDGE_M,),
    thumb_actual_centers_rad=(1.40, 1.45, 1.50, 1.55, 1.60),
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
    source_pose_manifest=SOURCE_POSE_MANIFEST,
    quick_static_samples_per_cell=2_000,
    quick_static_retain_per_cell=2,
    full_static_samples_per_cell=10_000,
    full_static_retain_per_cell=4,
    controller_seeds_per_pose=6,
    local_pose_count=16,
    local_refine_per_pose=128,
    exact_grasp_pose_count=16,
    manipulation_probe_count=17,
    manipulation_candidates_per_pose=1,
    manipulation_refine_pose_count=1,
    manipulation_refine_per_pose=1,
    perturbations_per_trajectory=16,
    selected_trajectory_count=5,
    minimum_distinct_edges=1,
    minimum_thumb_bands=1,
    initial_target_success_count=1,
    validation_labels=VALIDATION_LABELS,
    allow_single_edge=True,
    grasp_pose_identity=(
        "cube_hand_topology_contact_point_plan_nominal_qpos_sha256"
    ),
)

ROBUSTNESS = RobustnessParameters(
    edge_m=(EDGE_M,),
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

CONTACT_POINT_TARGETED_90MM_ACTUAL_GRASP_POSE = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand fixed-90-mm contact-point-targeted actual grasp-pose search"
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
        kinematic_samples_per_pitch=CONTACT_POINT_SEARCH.point_group_sample_count,
        dynamic_candidate_count=CONTACT_POINT_SEARCH.dynamic_candidate_count,
        local_refine_seed_count=CONTACT_POINT_SEARCH.local_pose_count,
        local_refine_per_seed=CONTACT_POINT_SEARCH.local_refine_per_pose,
        final_candidate_count=CONTACT_POINT_SEARCH.exact_candidate_count,
        perturbations_per_final_candidate=(
            CONTACT_POINT_SEARCH.perturbations_per_trajectory
        ),
        fallback_kinematic_samples_per_pitch=CAMPAIGN.full_static_samples_per_cell,
        fallback_candidate_count=CAMPAIGN.full_static_retain_per_cell,
        final_target_delta_rad=None,
        manipulation_delta_rad=MANIPULATION_DELTA_BOUNDS_RAD,
        pregrasp_targets_rad=PRECONTACT_TARGET_BOUNDS_RAD,
    ),
    robustness=ROBUSTNESS,
    artifact_root=ARTIFACT_ROOT,
    control_protocol=CONTROL_PROTOCOL,
    tuning_strategy="contact_point_targeted_actual_grasp_pose",
    contact_alignment=V6_EXPERIMENT.contact_alignment,
    far_hand_pose_constraints=POSE_CONSTRAINTS,
    fingertip_contact_preferences=FINGERTIP_CONTACT_PREFERENCES,
    pose_preservation=POSE_PRESERVATION,
    closure_alignment=CLOSURE_ALIGNMENT,
    motion_smoothness=MOTION_SMOOTHNESS,
    actual_contact_grasp_pose=GRASP_POSE,
    actual_contact_grasp_pose_campaign=CAMPAIGN,
    contact_point_search=CONTACT_POINT_SEARCH,
)

EXPERIMENT_DEFINITION = CONTACT_POINT_TARGETED_90MM_ACTUAL_GRASP_POSE

__all__ = [
    "ARTIFACT_ROOT",
    "CAMPAIGN",
    "CONTACT_POINT_SEARCH",
    "CONTACT_POINT_TARGETED_90MM_ACTUAL_GRASP_POSE",
    "EDGE_M",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "FRICTION",
    "MASS_KG",
    "ROBUSTNESS",
    "SEED_CONTACT_POINT_PLAN",
    "SIGNED_CLOCKWISE_ORBIT_DEG",
    "SOURCE_POSE_CANDIDATE_IDS",
    "SOURCE_POSE_MANIFEST",
    "VALIDATION_LABELS",
]
