from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from xhand_grasp.actual_contact_capability import (
    is_contact_preserving_planned_lift_definition,
    resolve_contact_preserving_planned_lift_definition,
)
from xhand_grasp.config import validate_config
from xhand_grasp.experiment import get_experiment
from xhand_grasp.experiments.opposed_face_palm_down_contact_preserving_planned_lift import (
    CONTACT_PRESERVING_PLANNED_LIFT as BASE_EXPERIMENT,
    EXPERIMENT_ID as BASE_EXPERIMENT_ID,
)
from xhand_grasp.experiments.opposed_face_palm_down_joint_pair_aligned_contact_preserving_planned_lift import (
    ARTIFACT_ROOT,
    EXPERIMENT_ID,
    JOINT_PAIR_ALIGNED_CONTACT_PRESERVING_PLANNED_LIFT,
    PRECONTACT_TARGET_BOUNDS_RAD,
)
from xhand_grasp.scene import build_model
from xhand_grasp.tuning.contact_preserving_candidate_artifacts import (
    authenticate_v14_candidate_artifacts,
    run_or_resume_v14_candidate_artifacts,
)


BASE_CONFIG = Path(
    "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)
ALIGNED_CONFIG = Path(
    "grasp_configs/left_opposed_face_palm_down_joint_pair_aligned_"
    "contact_preserving_planned_lift.json"
)
THUMB_BEND = "left_hand_thumb_bend_joint_actuator"


def _config(*, experiment_id: str = EXPERIMENT_ID) -> dict[str, Any]:
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config["experiment_id"] = experiment_id
    return config


class _NearMissSession:
    def __init__(self) -> None:
        self.steps = 0

    @property
    def complete(self) -> bool:
        return self.steps >= 1

    def advance_one(self) -> None:
        self.steps += 1

    def finalize(self, *, trace_path=None) -> dict[str, Any]:
        assert trace_path is None
        return {
            "passed": False,
            "failed_checks": ["operation_median_lift_reached"],
            "stage_status": {
                "grasp_success": False,
                "manipulation_success": False,
                "full_success": False,
            },
        }

    def close(self) -> None:
        return None


def test_joint_pair_aligned_v14_definition_has_independent_identity_and_yaw() -> None:
    definition = get_experiment(EXPERIMENT_ID)
    assert definition is JOINT_PAIR_ALIGNED_CONTACT_PRESERVING_PLANNED_LIFT
    assert definition.search_bounds.hand_roll_deg == (-10.0, 10.0)
    assert definition.search_bounds.hand_yaw_deg == (-30.0, 30.0)
    assert definition.artifact_root == ARTIFACT_ROOT
    assert definition.artifact_root != BASE_EXPERIMENT.artifact_root
    assert definition.contact_preserving_planned_lift_campaign is (
        BASE_EXPERIMENT.contact_preserving_planned_lift_campaign
    )
    assert definition.control_protocol is BASE_EXPERIMENT.control_protocol
    assert definition.far_hand_pose_constraints is (
        BASE_EXPERIMENT.far_hand_pose_constraints
    )
    assert definition.far_hand_pose_constraints is not None
    assert definition.far_hand_pose_constraints.finger_down_tilt_deg == (
        30.0,
        40.0,
    )
    assert definition.far_hand_pose_constraints.palm_plane_ground_angle_deg == (
        30.0,
        40.0,
    )
    assert is_contact_preserving_planned_lift_definition(definition)


def test_aligned_thumb_precontact_bound_is_local_and_inside_real_model_limits() -> None:
    definition = get_experiment(EXPERIMENT_ID)
    assert definition.search_bounds.pregrasp_targets_rad is not None
    assert definition.search_bounds.pregrasp_targets_rad[THUMB_BEND] == (
        0.9,
        1.6,
    )
    assert PRECONTACT_TARGET_BOUNDS_RAD[THUMB_BEND] == (0.9, 1.6)
    assert BASE_EXPERIMENT.search_bounds.pregrasp_targets_rad is not None
    assert BASE_EXPERIMENT.search_bounds.pregrasp_targets_rad[THUMB_BEND] == (
        0.9,
        1.5,
    )
    for actuator in definition.search_bounds.pregrasp_targets_rad:
        if actuator != THUMB_BEND:
            assert definition.search_bounds.pregrasp_targets_rad[actuator] == (
                BASE_EXPERIMENT.search_bounds.pregrasp_targets_rad[actuator]
            )

    config = json.loads(ALIGNED_CONFIG.read_text(encoding="utf-8"))
    config["control"]["precontact_targets_rad"][THUMB_BEND] = 1.522366
    validate_config(config)
    model, _ = build_model(config)
    actuator_id = model.actuator(THUMB_BEND).id
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    effective_upper = min(
        float(model.actuator_ctrlrange[actuator_id, 1]),
        float(model.jnt_range[joint_id, 1]),
    )
    assert effective_upper == pytest.approx(1.832)
    assert PRECONTACT_TARGET_BOUNDS_RAD[THUMB_BEND][1] < effective_upper

    legacy = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    legacy["control"]["precontact_targets_rad"][THUMB_BEND] = 1.522366
    with pytest.raises(ValueError, match="precontact_targets_rad"):
        validate_config(legacy)


def test_joint_pair_aligned_config_accepts_expanded_yaw_without_weakening_base() -> None:
    expanded = _config()
    expanded["hand_pose"]["rpy_deg"][2] = 25.0
    validate_config(expanded)
    assert resolve_contact_preserving_planned_lift_definition(expanded) is (
        JOINT_PAIR_ALIGNED_CONTACT_PRESERVING_PLANNED_LIFT
    )

    legacy = copy.deepcopy(expanded)
    legacy["experiment_id"] = BASE_EXPERIMENT_ID
    with pytest.raises(ValueError, match="hand yaw"):
        validate_config(legacy)


def test_candidate_artifacts_bind_the_registered_aligned_experiment_id(
    tmp_path: Path,
) -> None:
    config = _config()
    destination = tmp_path / "candidate_14021"
    bundle = run_or_resume_v14_candidate_artifacts(
        config,
        destination,
        14021,
        session_factory=lambda _value: _NearMissSession(),
        validator=None,
    )

    assert bundle.result["experiment_id"] == EXPERIMENT_ID
    assert bundle.trace_retained is False
    authenticated = authenticate_v14_candidate_artifacts(
        destination,
        expected_config=config,
        expected_candidate_id=14021,
    )
    assert authenticated.result["experiment_id"] == EXPERIMENT_ID


def test_candidate_artifacts_still_reject_non_capability_experiment(
    tmp_path: Path,
) -> None:
    config = _config(experiment_id=(
        "left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift"
    ))
    with pytest.raises(ValueError, match="registered v14 experiment"):
        run_or_resume_v14_candidate_artifacts(
            config,
            tmp_path / "candidate_14022",
            14022,
            session_factory=lambda _value: pytest.fail("physics must not run"),
            validator=None,
        )
