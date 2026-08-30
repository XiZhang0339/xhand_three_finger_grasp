"""Feedback-gated opposed-face campaign for 52--64 mm cubes."""

from __future__ import annotations

from xhand_grasp.experiment import (
    BoundaryExpansionParameters,
    ControlProtocolSettings,
    DEFAULT_MANIPULATION_DELTA_BOUNDS_RAD,
    ExperimentDefinition,
    GraspGateSettings,
    RobustnessCaseFamilies,
    RobustnessParameters,
    SearchBounds,
    SizeCampaignParameters,
)
from xhand_grasp.experiments.opposed_face_palm_down import (
    OPPOSED_FACE_PALM_DOWN,
    SEARCH_FACE_ASSIGNMENTS,
)


EXPERIMENT_ID = "left_opposed_face_palm_down_larger_cube_grasp_then_lift"

BOUNDARY_EXPANSION = BoundaryExpansionParameters(
    position_tolerance_m=0.001,
    actuator_tolerance_rad=0.03,
    position_expand_m=0.005,
    actuator_expand_rad=0.1,
    max_expansions=1,
)

CONTROL_PROTOCOL = ControlProtocolSettings(
    strategy="grasp_verify_then_manipulate",
    failure_behavior="abort_hold_grasp_pose",
    settle_s=0.5,
    close_s=1.0,
    verify_timeout_s=0.75,
    stable_window_s=0.25,
    manipulate_s=1.5,
    min_hold_s=1.0,
    grasp_gate=GraspGateSettings(
        min_target_face_force_n=0.05,
        min_target_force_fraction=0.95,
        require_touch=True,
        require_support_contact=True,
        max_translation_m=0.0005,
        max_orientation_drift_deg=3.0,
        max_linear_speed_m_s=0.01,
        max_early_lift_m=0.001,
    ),
)

SIZE_CAMPAIGN = SizeCampaignParameters(
    schema_version=1,
    coarse_edges_m=(0.052, 0.054, 0.056, 0.058, 0.060, 0.062, 0.064),
    discovery_mass_kg=0.020,
    discovery_friction=0.8,
    reference_edge_m=0.030,
    reference_mass_kg=0.020,
    coarse_samples_per_pitch=5_000,
    odd_samples_per_pitch=10_000,
    fine_samples_per_pitch=50_000,
    coarse_size_count=3,
    exact_size_count=2,
    fine_dynamic_per_size=256,
    local_seed_count_per_size=4,
    local_refine_per_seed=128,
    constant_density_max_candidates=16,
    density_refine_seed_count=4,
    density_refine_per_seed=128,
    finalist_count=16,
    perturbations_per_final=16,
    boundary=BOUNDARY_EXPANSION,
    target_sampling_policy="grasp_targets_plus_manipulation_delta",
    manipulation_seed_count=8,
    manipulation_refine_per_seed=128,
)

ROBUSTNESS_CASE_FAMILIES = RobustnessCaseFamilies(
    schema_version=1,
    edge_limits_m=(0.052, 0.064),
    edge_window_count=5,
    edge_step_m=0.001,
    density_mass_scales=(0.9, 1.0, 1.1),
    friction=(0.4, 0.6, 0.8, 1.0, 1.2),
    fixed_mass_kg=0.020,
    boundary_probe_step_m=0.0005,
    fixed_mass_case_label="fixed_20g_control",
)

LARGER_CUBE_ROBUSTNESS = RobustnessParameters(
    edge_m=(0.058,),
    mass_kg=(0.020,),
    friction=ROBUSTNESS_CASE_FAMILIES.friction,
    perturbation_count=50,
    required_pass_count=45,
    seed=20260821,
    position_xy_delta_m=(-0.0015, 0.0015),
    rpy_delta_deg=(-3.0, 3.0),
    mass_scale=(0.9, 1.1),
    friction_delta=(-0.1, 0.1),
    z_offset_delta_m=(0.0, 0.0005),
    case_families=ROBUSTNESS_CASE_FAMILIES,
)

LARGER_CUBE_GRASP_THEN_LIFT = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand fixed-palm opposed-face 52--64 mm cube campaign with a "
        "continuously verified grasp before any relative three-finger lift command"
    ),
    evaluation=OPPOSED_FACE_PALM_DOWN.evaluation,
    candidate_faces=SEARCH_FACE_ASSIGNMENTS,
    search_bounds=SearchBounds(
        palm_pitch_values_deg=(60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0),
        hand_roll_deg=(-15.0, 15.0),
        hand_yaw_deg=(-15.0, 15.0),
        cube_position_in_root_m={
            "x": (0.045, 0.095),
            "y": (-0.042, -0.004),
            "z": (0.080, 0.122),
        },
        cube_yaw_deg=(-15.0, 45.0),
        actuator_targets_rad={
            "left_hand_thumb_bend_joint_actuator": (0.15, 1.35),
            "left_hand_thumb_rota_joint1_actuator": (0.15, 1.35),
            "left_hand_thumb_rota_joint2_actuator": (0.40, 1.50),
            "left_hand_index_bend_joint_actuator": (-0.15, 0.15),
            "left_hand_index_joint1_actuator": (0.30, 1.70),
            "left_hand_index_joint2_actuator": (0.30, 1.70),
            "left_hand_mid_joint1_actuator": (0.30, 1.70),
            "left_hand_mid_joint2_actuator": (0.30, 1.70),
        },
        seed=20260821,
        kinematic_samples_per_pitch=SIZE_CAMPAIGN.coarse_samples_per_pitch,
        dynamic_candidate_count=SIZE_CAMPAIGN.dynamic_candidate_count,
        local_refine_seed_count=(
            SIZE_CAMPAIGN.exact_size_count
            * SIZE_CAMPAIGN.local_seed_count_per_size
        ),
        local_refine_per_seed=SIZE_CAMPAIGN.local_refine_per_seed,
        final_candidate_count=SIZE_CAMPAIGN.finalist_count,
        perturbations_per_final_candidate=SIZE_CAMPAIGN.perturbations_per_final,
        fallback_kinematic_samples_per_pitch=SIZE_CAMPAIGN.fine_samples_per_pitch,
        fallback_candidate_count=SIZE_CAMPAIGN.dynamic_candidate_count,
        final_target_delta_rad=None,
        manipulation_delta_rad=DEFAULT_MANIPULATION_DELTA_BOUNDS_RAD,
    ),
    robustness=LARGER_CUBE_ROBUSTNESS,
    artifact_root=(
        "artifacts/left_opposed_face_palm_down_larger_cube_grasp_then_lift"
    ),
    size_campaign=SIZE_CAMPAIGN,
    control_protocol=CONTROL_PROTOCOL,
)

EXPERIMENT_DEFINITION = LARGER_CUBE_GRASP_THEN_LIFT
