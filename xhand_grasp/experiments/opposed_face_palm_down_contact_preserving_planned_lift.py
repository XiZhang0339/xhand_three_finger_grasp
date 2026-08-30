"""Schema-v14 contact-preserving planned three-finger lift experiment."""

from __future__ import annotations

from dataclasses import replace

from xhand_grasp.experiment import (
    ACTIVE_ACTUATORS,
    ActualContactGraspPoseCampaignParameters,
    ContactFeedbackParameters,
    ContactForceTargets,
    ContactPreservingPlannedLiftCampaignParameters,
    ExperimentDefinition,
    ManipulationPlanParameters,
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
from xhand_grasp.experiments.opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_smooth_vertical_lift import (
    CONTACT_PRELOAD_TARGET_BOUNDS_RAD,
    CONTROL_PROTOCOL as V11_CONTROL_PROTOCOL,
    POSE_CONSTRAINTS,
    PRECONTACT_TARGET_BOUNDS_RAD,
)
from xhand_grasp.experiments.opposed_face_palm_down_pose_preserving_grasp import (
    EVALUATION as V6_EVALUATION,
    POSE_PRESERVATION,
    POSE_PRESERVING_GRASP as V6_EXPERIMENT,
)
from xhand_grasp.experiments.opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift import (
    FINGERTIP_CONTACT_PREFERENCES,
    PRIMARY_FACE_ASSIGNMENT,
)


EXPERIMENT_ID = "left_opposed_face_palm_down_contact_preserving_planned_lift"
EDGES_M = tuple(value / 1000.0 for value in range(60, 89))
MASS_KG = 0.160
FRICTION = 0.8
SEED = 20260821

# Schema v14 owns an independent manipulation envelope.  Earlier campaigns
# intentionally limited endpoint deltas to a small discovery box (for
# example index joint2 stopped at +0.4 rad), but the contact-constrained
# planner is required to use all real actuator headroom.  These bounds are the
# conservative intersection of the XHAND ctrl ranges over every registered
# contact-preload target.  The planner still clips each concrete pair again
# against its exact compiled-model ctrl/joint limits.
MANIPULATION_DELTA_BOUNDS_RAD = {
    "left_hand_thumb_bend_joint_actuator": (-1.750, 0.432),
    "left_hand_thumb_rota_joint1_actuator": (-1.498, 1.670),
    "left_hand_thumb_rota_joint2_actuator": (-1.100, 1.120),
    "left_hand_index_bend_joint_actuator": (-0.324, 0.324),
    "left_hand_index_joint1_actuator": (-0.900, 1.719),
    "left_hand_index_joint2_actuator": (-1.800, 1.119),
    "left_hand_mid_joint1_actuator": (-1.150, 1.569),
    "left_hand_mid_joint2_actuator": (-1.700, 1.269),
}
SOURCE_GRASP_CATALOG = (
    "artifacts/left_opposed_face_palm_down_scaled_centered_spread_actual_"
    "grasp_then_lift/tune/campaign_yaw_bounded_v2/catalogs/target_5/"
    "grasp_pose/catalog.json"
)
SOURCE_GRASP_CATALOG_SHA256 = (
    "715c1c5ddf46586a31be0dd219f7076136a09fac2f546854332d5c1c7ad18fea"
)
ARTIFACT_ROOT = (
    "artifacts/left_opposed_face_palm_down_contact_preserving_planned_lift"
)

VALIDATION_LABELS = {
    "grasp": "validated_fixed_160g_contact_preserving_source_grasp_ablation",
    "manipulation": "validated_fixed_160g_contact_preserving_planned_lift_ablation",
    "robust": "validated_fixed_160g_contact_preserving_robust_lift_ablation",
}

CAMPAIGN = ContactPreservingPlannedLiftCampaignParameters(
    schema_version=1,
    edges_m=EDGES_M,
    fixed_mass_kg=MASS_KG,
    friction=FRICTION,
    seed=SEED,
    source_grasp_catalog=SOURCE_GRASP_CATALOG,
    source_grasp_catalog_sha256=SOURCE_GRASP_CATALOG_SHA256,
    expected_source_grasp_count=18,
    grasp_rescue_candidates_per_edge_mapping=32,
    priority_grasp_rescue_candidates_per_seed=256,
    grasp_rescue_retain_per_edge_mapping=2,
    pair_probe_count=17,
    pair_shortlist_count=16,
    plan_knot_count=21,
    plan_duration_s=3.0,
    # The registered v14 contract generates exactly four physically
    # relinearized plans for each shortlisted grasp/object pair.
    plan_candidates_per_pair=4,
    feedback_plan_candidates_per_pair=4,
    feedback_candidates_per_plan=64,
    feedback_refine_plan_count=8,
    feedback_refine_per_plan=128,
    final_candidate_count=5,
    perturbations_per_final=16,
    robustness_trials=50,
    robustness_required_passes=45,
    force_target_minimum_n=0.2,
    force_target_maximum_n=3.0,
    operation_contact_duty_min=0.99,
    max_contact_loss_s=0.010,
    validation_labels=VALIDATION_LABELS,
)

CONTROL_PROTOCOL = replace(
    V11_CONTROL_PROTOCOL,
    strategy="grasp_verify_then_contact_preserving_planned_lift",
    close_s=1.5,
    manipulate_s=CAMPAIGN.plan_duration_s,
)
CONTACT_ALIGNMENT = replace(
    V6_EXPERIMENT.contact_alignment,
    # Schema v14 raises *target-face contact* duty to 99%, but does not
    # redefine the inherited v13 height-alignment contract. Keep these two
    # acceptance dimensions independent.
    operation_aligned_duty=0.70,
)
EVALUATION = replace(
    V6_EVALUATION,
    finger_contact_duty=CAMPAIGN.operation_contact_duty_min,
    simultaneous_contact_duty=CAMPAIGN.operation_contact_duty_min,
)

ACTUAL_CONTACT_CAMPAIGN = ActualContactGraspPoseCampaignParameters(
    schema_version=1,
    edges_m=EDGES_M,
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
    source_pose_manifest=SOURCE_GRASP_CATALOG,
    quick_static_samples_per_cell=1,
    quick_static_retain_per_cell=1,
    full_static_samples_per_cell=2,
    full_static_retain_per_cell=1,
    controller_seeds_per_pose=1,
    local_pose_count=CAMPAIGN.feedback_refine_plan_count,
    local_refine_per_pose=CAMPAIGN.feedback_refine_per_plan,
    exact_grasp_pose_count=CAMPAIGN.final_candidate_count,
    manipulation_probe_count=CAMPAIGN.pair_probe_count,
    manipulation_candidates_per_pose=CAMPAIGN.plan_candidates_per_pair,
    manipulation_refine_pose_count=CAMPAIGN.feedback_refine_plan_count,
    manipulation_refine_per_pose=CAMPAIGN.feedback_refine_per_plan,
    perturbations_per_trajectory=CAMPAIGN.perturbations_per_final,
    selected_trajectory_count=CAMPAIGN.final_candidate_count,
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
    perturbation_count=CAMPAIGN.robustness_trials,
    required_pass_count=CAMPAIGN.robustness_required_passes,
    seed=SEED,
    position_xy_delta_m=(-0.0015, 0.0015),
    rpy_delta_deg=(-3.0, 3.0),
    mass_scale=(1.0, 1.0),
    friction_delta=(-0.1, 0.1),
    z_offset_delta_m=(0.0, 0.0005),
)


_KNOT_TIMES_S = tuple(3.0 * index / 20.0 for index in range(21))


def _minimum_jerk(value: float) -> float:
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


_PROGRESS = tuple(_minimum_jerk(index / 20.0) for index in range(21))
_TERMINAL_DELTA_RAD = {
    "left_hand_thumb_bend_joint_actuator": 0.10419523579165951,
    "left_hand_thumb_rota_joint1_actuator": -0.05,
    "left_hand_thumb_rota_joint2_actuator": 0.40,
    "left_hand_index_bend_joint_actuator": -0.04796836768068951,
    "left_hand_index_joint1_actuator": -0.21891993821624128,
    "left_hand_index_joint2_actuator": 0.16223998845953624,
    "left_hand_mid_joint1_actuator": -0.1565654688748859,
    "left_hand_mid_joint2_actuator": 0.40,
}

DEFAULT_MANIPULATION_PLAN = ManipulationPlanParameters(
    schema_version=1,
    profile="piecewise_quintic_minimum_jerk",
    duration_s=3.0,
    knot_times_s=_KNOT_TIMES_S,
    actuator_waypoints_rad={
        actuator: tuple(_TERMINAL_DELTA_RAD[actuator] * value for value in _PROGRESS)
        for actuator in ACTIVE_ACTUATORS
    },
    desired_cube_position_delta_m=tuple(
        (0.0, 0.0, 0.011 * value) for value in _PROGRESS
    ),
    desired_cube_rotation_vector_rad=tuple((0.0, 0.0, 0.0) for _ in _PROGRESS),
    max_knot_delta_rad=0.04,
    trust_region_backtracks=4,
)

DEFAULT_CONTACT_FORCE_TARGETS_N = ContactForceTargets(
    schema_version=1,
    source="verify_window_median_clamped",
    minimum_n=CAMPAIGN.force_target_minimum_n,
    maximum_n=CAMPAIGN.force_target_maximum_n,
    per_finger_n={"thumb": 0.923, "index": 1.203, "mid": 2.134},
)

DEFAULT_CONTACT_FEEDBACK = ContactFeedbackParameters(
    schema_version=1,
    strategy="per_finger_force_pi",
    filter_time_constant_s=0.005,
    kp_rad_per_n={"thumb": 0.018, "index": 0.025, "mid": 0.015},
    ki_rad_per_n_s={"thumb": 0.060, "index": 0.080, "mid": 0.050},
    integral_limit_n_s=0.5,
    correction_limit_rad=0.06,
    rate_limit_rad_s=0.6,
    acceleration_limit_rad_s2=6.0,
    force_risk_n=0.1,
    freeze_on_risk=True,
    max_loss_s=CAMPAIGN.max_contact_loss_s,
    recovery_behavior="freeze_and_inward_preload_then_abort",
    operation_contact_duty_min=CAMPAIGN.operation_contact_duty_min,
)
CONTACT_FORCE_TARGETS_N = DEFAULT_CONTACT_FORCE_TARGETS_N
CONTACT_FEEDBACK = DEFAULT_CONTACT_FEEDBACK
CAMPAIGN.validate_candidate(
    DEFAULT_MANIPULATION_PLAN,
    DEFAULT_CONTACT_FORCE_TARGETS_N,
    DEFAULT_CONTACT_FEEDBACK,
)

CONTACT_PRESERVING_PLANNED_LIFT = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand 60--88 mm fixed-160-g authenticated grasp-pair search with "
        "multi-knot planned lift and per-finger contact-force feedback"
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
        kinematic_samples_per_pitch=1,
        dynamic_candidate_count=CAMPAIGN.maximum_plan_candidate_count,
        local_refine_seed_count=CAMPAIGN.feedback_refine_plan_count,
        local_refine_per_seed=CAMPAIGN.feedback_refine_per_plan,
        final_candidate_count=CAMPAIGN.final_candidate_count,
        perturbations_per_final_candidate=CAMPAIGN.perturbations_per_final,
        fallback_kinematic_samples_per_pitch=1,
        fallback_candidate_count=1,
        final_target_delta_rad=None,
        manipulation_delta_rad=MANIPULATION_DELTA_BOUNDS_RAD,
        pregrasp_targets_rad=PRECONTACT_TARGET_BOUNDS_RAD,
    ),
    robustness=ROBUSTNESS,
    artifact_root=ARTIFACT_ROOT,
    control_protocol=CONTROL_PROTOCOL,
    tuning_strategy="contact_preserving_planned_lift",
    contact_alignment=CONTACT_ALIGNMENT,
    far_hand_pose_constraints=POSE_CONSTRAINTS,
    fingertip_contact_preferences=FINGERTIP_CONTACT_PREFERENCES,
    pose_preservation=POSE_PRESERVATION,
    closure_alignment=CLOSURE_ALIGNMENT,
    motion_smoothness=MOTION_SMOOTHNESS,
    actual_contact_grasp_pose=GRASP_POSE,
    actual_contact_grasp_pose_campaign=ACTUAL_CONTACT_CAMPAIGN,
    contact_preserving_planned_lift_campaign=CAMPAIGN,
)

EXPERIMENT_DEFINITION = CONTACT_PRESERVING_PLANNED_LIFT

__all__ = [
    "ACTUAL_CONTACT_CAMPAIGN",
    "ARTIFACT_ROOT",
    "CAMPAIGN",
    "CONTACT_FEEDBACK",
    "CONTACT_FORCE_TARGETS_N",
    "CONTACT_PRESERVING_PLANNED_LIFT",
    "DEFAULT_CONTACT_FEEDBACK",
    "DEFAULT_CONTACT_FORCE_TARGETS_N",
    "DEFAULT_MANIPULATION_PLAN",
    "EDGES_M",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "FRICTION",
    "MANIPULATION_DELTA_BOUNDS_RAD",
    "MASS_KG",
    "ROBUSTNESS",
    "SEED",
    "SOURCE_GRASP_CATALOG",
    "SOURCE_GRASP_CATALOG_SHA256",
    "VALIDATION_LABELS",
]
