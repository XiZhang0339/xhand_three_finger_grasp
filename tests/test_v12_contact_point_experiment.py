from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from xhand_grasp.artifacts import resolved_run_config
from xhand_grasp.config import ACTIVE_ACTUATORS, INACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.experiment import ContactPointPlanParameters, get_experiment
from xhand_grasp.experiments.opposed_face_palm_down_90mm_contact_point_targeted_actual_grasp_pose import (
    CAMPAIGN,
    CONTACT_POINT_SEARCH,
    EDGE_M,
    EXPERIMENT_DEFINITION,
    EXPERIMENT_ID,
    FRICTION,
    MASS_KG,
    SEED_CONTACT_POINT_PLAN,
    SIGNED_CLOCKWISE_ORBIT_DEG,
    SOURCE_POSE_CANDIDATE_IDS,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = (
    REPO_ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_90mm_contact_point_targeted_actual_grasp_pose.json"
)


def _raw_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _selected_plan(**points: tuple[float, float]) -> ContactPointPlanParameters:
    merged = dict(SEED_CONTACT_POINT_PLAN.target_face_yz_m)
    merged.update(points)
    return ContactPointPlanParameters(
        schema_version=1,
        cube_edge_m=EDGE_M,
        target_faces=SEED_CONTACT_POINT_PLAN.target_faces,
        target_face_yz_m=merged,
        target_radius_m=0.002,
    )


def test_v12_template_loads_and_resolves_registered_definition() -> None:
    config = load_config(CONFIG_PATH)
    assert config["schema_version"] == 12
    assert config["experiment_id"] == EXPERIMENT_ID
    assert get_experiment(EXPERIMENT_ID) is EXPERIMENT_DEFINITION
    assert config["cube"]["edge_m"] == EDGE_M
    assert config["cube"]["mass_kg"] == MASS_KG
    assert config["cube"]["friction"] == FRICTION
    assert config["cube"]["z_offset_m"] == 0.0
    assert set(config["control"]["contact_preload_targets_rad"]) == set(
        ACTIVE_ACTUATORS
    )
    assert set(config["control"]["contact_preload_targets_rad"]).isdisjoint(
        INACTIVE_ACTUATORS
    )


def test_v12_registered_search_and_plan_are_serialized_exactly() -> None:
    config = _raw_config()
    assert config["contact_point_search"] == CONTACT_POINT_SEARCH.as_config()
    assert config["contact_point_plan"] == SEED_CONTACT_POINT_PLAN.as_config()
    assert config["actual_contact_grasp_pose_campaign"] == CAMPAIGN.as_config()
    assert config["search"]["budget"] == CONTACT_POINT_SEARCH.as_config()["budget"]
    assert config["search"]["shared_faces"] == ["+X"]
    assert config["contact_point_search"]["signed_clockwise_orbit_deg"] == list(
        SIGNED_CLOCKWISE_ORBIT_DEG
    )
    assert tuple(config["contact_point_search"]["source_pose_candidate_ids"]) == (
        SOURCE_POSE_CANDIDATE_IDS
    )


def test_v12_plan_resolves_signed_face_normal_coordinate() -> None:
    assert SEED_CONTACT_POINT_PLAN.resolved_point_cube_local_m("thumb") == (
        -0.045,
        0.010,
        0.012,
    )
    assert SEED_CONTACT_POINT_PLAN.resolved_point_cube_local_m("index") == (
        0.045,
        0.002,
        0.012,
    )
    assert SEED_CONTACT_POINT_PLAN.resolved_point_cube_local_m("mid") == (
        0.045,
        0.021,
        0.012,
    )
    parsed = ContactPointPlanParameters.from_config(
        SEED_CONTACT_POINT_PLAN.as_config()
    )
    assert parsed == SEED_CONTACT_POINT_PLAN
    assert parsed.point_plan_id == (
        "fa03e0eff616a29124650a9fc596be18f4534d063eb56c39e4c39417539f2833"
    )


def test_v12_plan_hash_and_derived_points_fail_closed() -> None:
    payload = SEED_CONTACT_POINT_PLAN.as_config()
    payload["target_points"]["thumb"]["face_yz_m"][0] += 0.001
    with pytest.raises(ValueError, match="point_plan_id"):
        ContactPointPlanParameters.from_config(payload)

    payload = SEED_CONTACT_POINT_PLAN.as_config()
    payload["target_points_cube_local_m"]["index"][0] = -0.045
    with pytest.raises(ValueError, match="derived target_points"):
        ContactPointPlanParameters.from_config(payload)


def test_v12_selected_point_plan_can_move_only_inside_registered_domain() -> None:
    selected = _selected_plan(
        thumb=(0.011, 0.013),
        index=(0.001, 0.011),
        mid=(0.020, 0.012),
    )
    assert CONTACT_POINT_SEARCH.validate_selected_plan(selected) is selected

    with pytest.raises(ValueError, match="outside the registered search box"):
        CONTACT_POINT_SEARCH.validate_selected_plan(
            _selected_plan(thumb=(0.018001, 0.012))
        )
    with pytest.raises(ValueError, match="height-spread"):
        CONTACT_POINT_SEARCH.validate_selected_plan(
            _selected_plan(thumb=(0.010, 0.017001))
        )
    with pytest.raises(ValueError, match="too close"):
        CONTACT_POINT_SEARCH.validate_selected_plan(
            _selected_plan(index=(0.010, 0.012), mid=(0.019, 0.012))
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("cube", "edge_m"), 0.091, "campaign grid"),
        (("cube", "mass_kg"), 0.161, "mass must match"),
        (("cube", "friction"), 0.81, "friction must match"),
    ],
)
def test_v12_fixed_cube_contract_rejects_mutation(
    path: tuple[str, str], value: float, message: str
) -> None:
    config = _raw_config()
    config[path[0]][path[1]] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_v12_config_rejects_contact_plan_switch_or_tampering() -> None:
    config = _raw_config()
    selected = _selected_plan(thumb=(0.011, 0.012))
    config["contact_point_plan"] = selected.as_config()
    validate_config(config)

    config = _raw_config()
    config["contact_point_plan"]["point_plan_id"] = "0" * 64
    with pytest.raises(ValueError, match="point_plan_id"):
        validate_config(config)

    config = _raw_config()
    config["contact_point_search"]["signed_clockwise_orbit_deg"] = [0.0]
    with pytest.raises(ValueError, match="contact_point_search must match"):
        validate_config(config)


def test_v12_single_edge_campaign_requires_explicit_opt_in() -> None:
    assert CAMPAIGN.edges_m == (0.09,)
    assert CAMPAIGN.allow_single_edge is True
    assert CAMPAIGN.grasp_pose_identity == (
        "cube_hand_topology_contact_point_plan_nominal_qpos_sha256"
    )
    with pytest.raises(ValueError, match="edges_m"):
        dataclasses.replace(CAMPAIGN, allow_single_edge=False)


def test_v11_template_still_loads_without_v12_fields() -> None:
    legacy = load_config(
        REPO_ROOT
        / "grasp_configs"
        / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_smooth_vertical_lift.json"
    )
    assert legacy["schema_version"] == 11
    assert "contact_point_search" not in legacy
    assert "contact_point_plan" not in legacy


def test_contact_point_plan_is_deeply_detached_from_config_output() -> None:
    first = SEED_CONTACT_POINT_PLAN.as_config()
    second = copy.deepcopy(first)
    first["target_points"]["thumb"]["face_yz_m"][0] = 0.0
    assert second == SEED_CONTACT_POINT_PLAN.as_config()


def test_v12_resolved_status_treats_grasp_as_the_campaign_success_scope() -> None:
    config = load_config(CONFIG_PATH)
    summary = {
        "passed": False,
        "failed_checks": ["operation_median_lift_reached"],
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": False,
            "full_success": False,
        },
    }

    status = resolved_run_config(config, summary)["experiment_status"]

    assert status["passed"] is True
    assert status["hard_constraints_passed"] is True
    assert status["full_hard_constraints_passed"] is False
    assert status["campaign_validated"] is True
    assert status["classification"] == "validated_fixed_160g_grasp_ablation"
    assert status["success_scope"] == "grasp_only_contact_point_hard_checks"
    assert status["manipulation_success_required"] is False


def test_v11_resolved_status_still_requires_manipulation() -> None:
    config = load_config(
        REPO_ROOT
        / "grasp_configs"
        / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_smooth_vertical_lift.json"
    )
    summary = {
        "passed": False,
        "failed_checks": ["operation_median_lift_reached"],
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": False,
            "full_success": False,
        },
    }

    status = resolved_run_config(config, summary)["experiment_status"]

    assert status["passed"] is False
    assert status["classification"] == "grasp_acquired_manipulation_failed"
