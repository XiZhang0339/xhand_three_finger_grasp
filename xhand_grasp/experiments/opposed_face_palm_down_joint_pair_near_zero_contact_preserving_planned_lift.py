"""Schema-v15 near-zero joint-pair contact-preserving lift experiment."""

from __future__ import annotations

from dataclasses import replace

from xhand_grasp.experiment import (
    ContactPreservingPlannedLiftCampaignParameters,
    JointPairAlignmentSettings,
    JointPairFeedbackParameters,
)
from xhand_grasp.experiments.opposed_face_palm_down_joint_pair_aligned_contact_preserving_planned_lift import (
    JOINT_PAIR_ALIGNED_CONTACT_PRESERVING_PLANNED_LIFT as BASE_EXPERIMENT,
)
from xhand_grasp.tuning.joint_pair_near_zero_campaign import (
    EXPERIMENT_ID,
    JointPairNearZeroBudget,
    SOURCE_CANDIDATE_ID,
    SOURCE_CATALOG,
    SOURCE_CATALOG_SHA256,
    SOURCE_CONFIG_SHA256,
    SOURCE_MEMBER,
    SOURCE_RESULT_SHA256,
    SOURCE_TRACE_SHA256,
)


ARTIFACT_ROOT = (
    "artifacts/left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift"
)
EDGE_M = 0.079
MASS_KG = 0.160
FRICTION = 0.8
BUDGET = JointPairNearZeroBudget()

VALIDATION_LABELS = {
    "grasp": "validated_near_zero_joint_pair_grasp",
    "manipulation": "validated_near_zero_joint_pair_planned_lift",
    "robust": "validated_near_zero_joint_pair_robust_lift",
}

JOINT_PAIR_ALIGNMENT = JointPairAlignmentSettings(
    schema_version=1,
    joint_names=("left_hand_index_joint1", "left_hand_mid_joint1"),
    frame="cube_local",
    axis="+Y",
    require_positive_y=True,
    minimum_length_m=0.010,
    residual="vx_over_vy_vz_over_vy",
    grasp_p95_max_deg=0.25,
    grasp_max_deg=0.50,
    operation_p95_max_deg=0.50,
    operation_max_deg=1.0,
    operation_within_p95_limit_duty_min=0.99,
    max_continuous_violation_s=0.010,
    constraint_polygon_sides=8,
    audit_timestep_s=0.001,
)

JOINT_PAIR_FEEDBACK = JointPairFeedbackParameters(
    schema_version=1,
    strategy="previous_frame_weighted_nullspace",
    alignment_gain=0.50,
    slip_recovery_gain_rad_per_m=4.0,
    correction_limit_rad=0.020,
    rate_limit_rad_s=0.20,
    acceleration_limit_rad_s2=2.0,
    use_previous_observation=True,
    freeze_threshold_deg=0.50,
    abort_threshold_deg=1.0,
    slip_freeze_threshold_m=0.0015,
    slip_abort_threshold_m=0.0020,
    recovery_behavior="freeze_inward_recover_then_abort",
    damping=1e-8,
    vertical_response_weight=1.0,
    force_response_weight=1.0,
)

BASE_CAMPAIGN = BASE_EXPERIMENT.contact_preserving_planned_lift_campaign
assert BASE_CAMPAIGN is not None
CAMPAIGN = ContactPreservingPlannedLiftCampaignParameters(
    schema_version=2,
    edges_m=(EDGE_M,),
    fixed_mass_kg=MASS_KG,
    friction=FRICTION,
    seed=BUDGET.seed,
    source_grasp_catalog=SOURCE_CATALOG.as_posix(),
    source_grasp_catalog_sha256=SOURCE_CATALOG_SHA256,
    expected_source_grasp_count=1,
    grasp_rescue_candidates_per_edge_mapping=BUDGET.static_start_count,
    priority_grasp_rescue_candidates_per_seed=BUDGET.static_start_count,
    grasp_rescue_retain_per_edge_mapping=BUDGET.static_retain_count,
    pair_probe_count=17,
    pair_shortlist_count=BUDGET.measured_grasp_retain_count,
    plan_knot_count=21,
    plan_duration_s=3.0,
    plan_candidates_per_pair=BUDGET.sequential_plans_per_grasp,
    feedback_plan_candidates_per_pair=BUDGET.sequential_plans_per_grasp,
    feedback_candidates_per_plan=BUDGET.feedback_candidates_per_plan,
    feedback_refine_plan_count=BUDGET.feedback_refine_plan_count,
    feedback_refine_per_plan=BUDGET.feedback_refine_per_plan,
    final_candidate_count=BUDGET.final_candidate_count,
    perturbations_per_final=BUDGET.perturbations_per_final,
    robustness_trials=BUDGET.robustness_trials,
    robustness_required_passes=BUDGET.robustness_required_passes,
    force_target_minimum_n=BASE_CAMPAIGN.force_target_minimum_n,
    force_target_maximum_n=BASE_CAMPAIGN.force_target_maximum_n,
    operation_contact_duty_min=0.99,
    max_contact_loss_s=0.010,
    validation_labels=VALIDATION_LABELS,
    allow_single_edge=True,
    source_candidate_id=SOURCE_CANDIDATE_ID,
    source_alias="best_near_zero_grasp",
    source_config_path=(SOURCE_MEMBER / "resolved_config.json").as_posix(),
    source_config_sha256=SOURCE_CONFIG_SHA256,
    source_result_path=(SOURCE_MEMBER / "result.json").as_posix(),
    source_result_sha256=SOURCE_RESULT_SHA256,
    source_trace_path=(SOURCE_MEMBER / "trace.npz").as_posix(),
    source_trace_sha256=SOURCE_TRACE_SHA256,
    dls_start_count=BUDGET.static_start_count,
    static_retain_count=BUDGET.static_retain_count,
    grasp_retain_count=BUDGET.measured_grasp_retain_count,
    revalidate_candidate_count=BUDGET.exact_rerun_count,
    root_translation_delta_m=(-0.0015, 0.0015),
    root_rotation_axis_delta_deg=(-1.5, 1.5),
    root_rotation_norm_max_deg=2.0,
    grasp_qpos_delta_rad=(-0.030, 0.030),
    close_duration_options_s=(1.25, 1.50, 1.75),
    close_modes=("original", "synchronized_preload"),
    alignment_gain_options=(0.25, 0.50, 0.75, 1.0),
    slip_recovery_gain_options_rad_per_m=(2.0, 4.0, 6.0, 8.0),
)

CONTROL_PROTOCOL = replace(
    BASE_EXPERIMENT.control_protocol,
    strategy=(
        "grasp_verify_then_joint_pair_aligned_contact_preserving_planned_lift"
    ),
    close_s=1.50,
    close_duration_options_s=(1.25, 1.50, 1.75),
)

BASE_ACTUAL_CONTACT_CAMPAIGN = (
    BASE_EXPERIMENT.actual_contact_grasp_pose_campaign
)
assert BASE_ACTUAL_CONTACT_CAMPAIGN is not None
ACTUAL_CONTACT_CAMPAIGN = replace(
    BASE_ACTUAL_CONTACT_CAMPAIGN,
    edges_m=(EDGE_M,),
    source_pose_manifest=SOURCE_CATALOG.as_posix(),
    local_pose_count=BUDGET.feedback_refine_plan_count,
    local_refine_per_pose=BUDGET.feedback_refine_per_plan,
    exact_grasp_pose_count=BUDGET.final_candidate_count,
    manipulation_probe_count=CAMPAIGN.pair_probe_count,
    manipulation_candidates_per_pose=BUDGET.sequential_plans_per_grasp,
    manipulation_refine_pose_count=BUDGET.feedback_refine_plan_count,
    manipulation_refine_per_pose=BUDGET.feedback_refine_per_plan,
    perturbations_per_trajectory=BUDGET.perturbations_per_final,
    selected_trajectory_count=BUDGET.final_candidate_count,
    minimum_distinct_edges=1,
    minimum_thumb_bands=1,
    validation_labels=VALIDATION_LABELS,
    allow_single_edge=True,
)

SEARCH_BOUNDS = replace(
    BASE_EXPERIMENT.search_bounds,
    seed=BUDGET.seed,
    dynamic_candidate_count=BUDGET.static_start_count,
    local_refine_seed_count=BUDGET.feedback_refine_plan_count,
    local_refine_per_seed=BUDGET.feedback_refine_per_plan,
    final_candidate_count=BUDGET.final_candidate_count,
    perturbations_per_final_candidate=BUDGET.perturbations_per_final,
)

ROBUSTNESS = replace(
    BASE_EXPERIMENT.robustness,
    edge_m=(EDGE_M,),
    mass_kg=(MASS_KG,),
    friction=(FRICTION,),
    perturbation_count=BUDGET.robustness_trials,
    required_pass_count=BUDGET.robustness_required_passes,
    seed=BUDGET.seed,
)

EVALUATION = replace(
    BASE_EXPERIMENT.evaluation,
    max_orientation_drift_deg=10.0,
)

JOINT_PAIR_NEAR_ZERO_CONTACT_PRESERVING_PLANNED_LIFT = replace(
    BASE_EXPERIMENT,
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand fixed-79-mm measured-grasp search with directed index-to-"
        "middle joint1 near-zero +Y alignment and contact-preserving lift"
    ),
    evaluation=EVALUATION,
    search_bounds=SEARCH_BOUNDS,
    robustness=ROBUSTNESS,
    artifact_root=ARTIFACT_ROOT,
    control_protocol=CONTROL_PROTOCOL,
    tuning_strategy="joint_pair_near_zero_contact_preserving_planned_lift",
    actual_contact_grasp_pose_campaign=ACTUAL_CONTACT_CAMPAIGN,
    contact_preserving_planned_lift_campaign=CAMPAIGN,
    joint_pair_alignment=JOINT_PAIR_ALIGNMENT,
    joint_pair_feedback=JOINT_PAIR_FEEDBACK,
)

EXPERIMENT_DEFINITION = JOINT_PAIR_NEAR_ZERO_CONTACT_PRESERVING_PLANNED_LIFT

__all__ = [
    "ACTUAL_CONTACT_CAMPAIGN",
    "ARTIFACT_ROOT",
    "BUDGET",
    "CAMPAIGN",
    "CONTROL_PROTOCOL",
    "EDGE_M",
    "EVALUATION",
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "JOINT_PAIR_ALIGNMENT",
    "JOINT_PAIR_FEEDBACK",
    "JOINT_PAIR_NEAR_ZERO_CONTACT_PRESERVING_PLANNED_LIFT",
    "ROBUSTNESS",
    "SEARCH_BOUNDS",
    "VALIDATION_LABELS",
]
