"""Relative-pose rescue experiment anchored at the 62 mm static solution."""

from __future__ import annotations

from xhand_grasp.experiment import ExperimentDefinition, SearchBounds
from xhand_grasp.experiments.opposed_face_palm_down import SEARCH_FACE_ASSIGNMENTS
from xhand_grasp.experiments.opposed_face_palm_down_larger_cube_grasp_then_lift import (
    LARGER_CUBE_GRASP_THEN_LIFT,
)


EXPERIMENT_ID = "left_opposed_face_palm_down_larger_cube_relative_pose_rescue"

_SOURCE = LARGER_CUBE_GRASP_THEN_LIFT
_SOURCE_BOUNDS = _SOURCE.search_bounds

# The first manipulation sweep placed thumb rota2 within 0.005 rad of its
# declared delta ceiling while retaining all three target-face contacts.  Apply
# the campaign's single 0.1 rad boundary expansion here, and expose the same
# one-shot headroom on the two distal flexion endpoints.  All three limits stay
# below the model's actual ctrl/joint limits (1.57 rad for thumb rota2 and
# 1.919 rad for index/middle joint2).
RESCUE_ACTUATOR_TARGET_BOUNDS = dict(_SOURCE_BOUNDS.actuator_targets_rad)
for _name in (
    "left_hand_index_joint2_actuator",
    "left_hand_mid_joint2_actuator",
):
    _lower, _upper = RESCUE_ACTUATOR_TARGET_BOUNDS[_name]
    RESCUE_ACTUATOR_TARGET_BOUNDS[_name] = (_lower, _upper + 0.1)

RESCUE_MANIPULATION_DELTA_BOUNDS = dict(_SOURCE_BOUNDS.manipulation_delta_rad or {})
_lower, _upper = RESCUE_MANIPULATION_DELTA_BOUNDS[
    "left_hand_thumb_rota_joint2_actuator"
]
RESCUE_MANIPULATION_DELTA_BOUNDS[
    "left_hand_thumb_rota_joint2_actuator"
] = (_lower, _upper + 0.1)

LARGER_CUBE_RELATIVE_POSE_RESCUE = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    description=(
        "Left-hand 61--63 mm relative-pose rescue around the 62 mm static "
        "candidate while retaining the feedback-gated grasp-then-lift contract"
    ),
    evaluation=_SOURCE.evaluation,
    candidate_faces=SEARCH_FACE_ASSIGNMENTS,
    search_bounds=SearchBounds(
        palm_pitch_values_deg=(87.0, 88.0, 89.0, 90.0),
        hand_roll_deg=(-3.0, 5.0),
        hand_yaw_deg=(-8.0, 1.0),
        cube_position_in_root_m={
            "x": (0.080, 0.086),
            "y": (-0.034, -0.026),
            "z": (0.104, 0.114),
        },
        cube_yaw_deg=(24.0, 34.0),
        actuator_targets_rad=RESCUE_ACTUATOR_TARGET_BOUNDS,
        seed=_SOURCE_BOUNDS.seed,
        kinematic_samples_per_pitch=_SOURCE_BOUNDS.kinematic_samples_per_pitch,
        dynamic_candidate_count=_SOURCE_BOUNDS.dynamic_candidate_count,
        local_refine_seed_count=_SOURCE_BOUNDS.local_refine_seed_count,
        local_refine_per_seed=_SOURCE_BOUNDS.local_refine_per_seed,
        final_candidate_count=_SOURCE_BOUNDS.final_candidate_count,
        perturbations_per_final_candidate=(
            _SOURCE_BOUNDS.perturbations_per_final_candidate
        ),
        fallback_kinematic_samples_per_pitch=(
            _SOURCE_BOUNDS.fallback_kinematic_samples_per_pitch
        ),
        fallback_candidate_count=_SOURCE_BOUNDS.fallback_candidate_count,
        final_target_delta_rad=None,
        manipulation_delta_rad=RESCUE_MANIPULATION_DELTA_BOUNDS,
    ),
    robustness=_SOURCE.robustness,
    artifact_root=(
        "artifacts/left_opposed_face_palm_down_larger_cube_relative_pose_rescue"
    ),
    size_campaign=_SOURCE.size_campaign,
    control_protocol=_SOURCE.control_protocol,
    tuning_strategy="relative_pose_rescue",
)

EXPERIMENT_DEFINITION = LARGER_CUBE_RELATIVE_POSE_RESCUE

__all__ = [
    "EXPERIMENT_DEFINITION",
    "EXPERIMENT_ID",
    "LARGER_CUBE_RELATIVE_POSE_RESCUE",
    "RESCUE_ACTUATOR_TARGET_BOUNDS",
    "RESCUE_MANIPULATION_DELTA_BOUNDS",
]
