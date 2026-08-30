"""Palm-down, exact opposed-face three-finger experiment definition."""

from __future__ import annotations

from xhand_grasp.experiment import (
    COMMON_ROBUSTNESS,
    DEFAULT_FINAL_TARGET_DELTA_BOUNDS_RAD,
    EvaluationSettings,
    ExperimentDefinition,
    OpposedFaceAssignment,
    SearchBounds,
)


EXPERIMENT_ID = "left_opposed_face_palm_down_lift"
TARGET_FACES = OpposedFaceAssignment(thumb="-X", index="+X", mid="+X")
SEARCH_FACE_ASSIGNMENTS = (
    TARGET_FACES,
    OpposedFaceAssignment(thumb="+X", index="-X", mid="-X"),
    OpposedFaceAssignment(thumb="-Y", index="+Y", mid="+Y"),
    OpposedFaceAssignment(thumb="+Y", index="-Y", mid="-Y"),
)

OPPOSED_FACE_PALM_DOWN = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand fixed-palm lift with index and middle on one cube face, "
        "thumb on the exact opposite face, and the palm normal within 30 degrees of down"
    ),
    evaluation=EvaluationSettings(
        target_faces=TARGET_FACES,
        palm_normal_local_axis="+X",
        world_down_axis="-Z",
        max_palm_down_angle_deg=30.0,
        surface_tolerance_m=0.00005,
        edge_margin_m=0.0005,
        min_normal_alignment=0.95,
        contact_force_min_n=0.05,
        target_force_fraction=0.95,
        finger_contact_duty=0.8,
        simultaneous_contact_duty=0.7,
        max_off_target_force_fraction=0.05,
        max_material_off_target_duty=0.01,
        max_material_off_target_run_s=0.010,
        forbid_active_nondistal=True,
    ),
    candidate_faces=SEARCH_FACE_ASSIGNMENTS,
    search_bounds=SearchBounds(
        palm_pitch_values_deg=(60.0, 65.0, 70.0, 75.0, 80.0),
        hand_roll_deg=(-10.0, 10.0),
        hand_yaw_deg=(-10.0, 10.0),
        cube_position_in_root_m={
            "x": (0.055, 0.085),
            "y": (-0.032, -0.014),
            "z": (0.090, 0.112),
        },
        cube_yaw_deg=(-5.0, 35.0),
        actuator_targets_rad={
            "left_hand_thumb_bend_joint_actuator": (0.35, 1.10),
            "left_hand_thumb_rota_joint1_actuator": (0.50, 1.15),
            "left_hand_thumb_rota_joint2_actuator": (0.85, 1.50),
            "left_hand_index_bend_joint_actuator": (-0.08, 0.08),
            "left_hand_index_joint1_actuator": (0.65, 1.20),
            "left_hand_index_joint2_actuator": (0.75, 1.45),
            "left_hand_mid_joint1_actuator": (0.55, 1.45),
            "left_hand_mid_joint2_actuator": (0.65, 1.50),
        },
        seed=20260821,
        kinematic_samples_per_pitch=50_000,
        dynamic_candidate_count=512,
        local_refine_seed_count=8,
        local_refine_per_seed=256,
        final_candidate_count=16,
        perturbations_per_final_candidate=16,
        fallback_kinematic_samples_per_pitch=5_000,
        fallback_candidate_count=2048,
        final_target_delta_rad=DEFAULT_FINAL_TARGET_DELTA_BOUNDS_RAD,
    ),
    robustness=COMMON_ROBUSTNESS,
    artifact_root="artifacts/left_opposed_face_palm_down",
)

# Explicit name for callers that prefer the type in the constant name.
EXPERIMENT_DEFINITION = OPPOSED_FACE_PALM_DOWN
