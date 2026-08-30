from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from xhand_grasp.config import load_config, validate_config
from xhand_grasp.experiment import get_experiment, resolve_experiment
from xhand_grasp.experiments.opposed_face_palm_down import (
    OPPOSED_FACE_PALM_DOWN,
)
from xhand_grasp.experiments.opposed_face_palm_down_large_cube import (
    EXPERIMENT_ID,
    LARGE_CUBE_OPPOSED_FACE_PALM_DOWN,
    ROBUSTNESS_CASE_FAMILIES,
    SIZE_CAMPAIGN,
)
from xhand_grasp.scene import build_model, cube_inertia


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "grasp_configs" / "left_opposed_face_palm_down_large_cube.json"


def _read_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_large_cube_experiment_is_registered_and_config_is_version_locked():
    config = load_config(CONFIG_PATH)
    definition = resolve_experiment(config)

    assert definition is LARGE_CUBE_OPPOSED_FACE_PALM_DOWN
    assert get_experiment(EXPERIMENT_ID) is definition
    assert definition.artifact_root == (
        "artifacts/left_opposed_face_palm_down_large_cube"
    )
    assert definition.size_campaign is SIZE_CAMPAIGN
    assert config["size_campaign"] == SIZE_CAMPAIGN.as_config()
    assert config["robustness"] == definition.robustness.as_config()
    assert config["experiment_status"]["passed"] is False
    assert config["experiment_status"]["constant_density_passed"] is False


def test_large_cube_search_domain_and_staged_budgets_match_the_plan():
    bounds = LARGE_CUBE_OPPOSED_FACE_PALM_DOWN.search_bounds

    assert bounds.final_target_delta_rad is None
    assert SIZE_CAMPAIGN.target_sampling_policy == (
        "independent_absolute_pregrasp_and_final"
    )
    assert bounds.palm_pitch_values_deg == (
        60.0,
        65.0,
        70.0,
        75.0,
        80.0,
        85.0,
        90.0,
    )
    assert bounds.hand_roll_deg == (-15.0, 15.0)
    assert bounds.hand_yaw_deg == (-15.0, 15.0)
    assert dict(bounds.cube_position_in_root_m) == {
        "x": (0.045, 0.095),
        "y": (-0.042, -0.004),
        "z": (0.080, 0.122),
    }
    assert bounds.cube_yaw_deg == (-15.0, 45.0)
    assert bounds.actuator_targets_rad[
        "left_hand_thumb_bend_joint_actuator"
    ] == (0.15, 1.35)
    assert bounds.actuator_targets_rad[
        "left_hand_thumb_rota_joint1_actuator"
    ] == (0.15, 1.35)
    assert bounds.actuator_targets_rad[
        "left_hand_thumb_rota_joint2_actuator"
    ] == (0.40, 1.50)
    assert bounds.actuator_targets_rad[
        "left_hand_index_bend_joint_actuator"
    ] == (-0.15, 0.15)
    for name in (
        "left_hand_index_joint1_actuator",
        "left_hand_index_joint2_actuator",
        "left_hand_mid_joint1_actuator",
        "left_hand_mid_joint2_actuator",
    ):
        assert bounds.actuator_targets_rad[name] == (0.30, 1.70)

    assert SIZE_CAMPAIGN.coarse_edges_m == pytest.approx(
        (0.036, 0.038, 0.040, 0.042, 0.044, 0.046, 0.048, 0.050)
    )
    assert SIZE_CAMPAIGN.coarse_samples_per_pitch == 5_000
    assert SIZE_CAMPAIGN.odd_samples_per_pitch == 10_000
    assert SIZE_CAMPAIGN.fine_samples_per_pitch == 50_000
    assert SIZE_CAMPAIGN.dynamic_candidate_count == 512
    assert SIZE_CAMPAIGN.fixed_mass_local_refinement_count == 1_024
    assert SIZE_CAMPAIGN.constant_density_max_candidates == 16
    assert SIZE_CAMPAIGN.density_local_refinement_count == 512
    assert SIZE_CAMPAIGN.finalist_count == 16
    assert SIZE_CAMPAIGN.perturbations_per_final == 16

    boundary = SIZE_CAMPAIGN.boundary
    assert boundary.position_tolerance_m == pytest.approx(0.001)
    assert boundary.actuator_tolerance_rad == pytest.approx(0.03)
    assert boundary.position_expand_m == pytest.approx(0.005)
    assert boundary.actuator_expand_rad == pytest.approx(0.1)
    assert boundary.max_expansions == 1


def test_constant_density_policy_matches_the_30_mm_20_g_reference():
    assert SIZE_CAMPAIGN.density_kg_m3 == pytest.approx(740.7407407407407)
    assert SIZE_CAMPAIGN.constant_density_mass_kg(0.030) == pytest.approx(0.020)
    assert SIZE_CAMPAIGN.constant_density_mass_kg(0.036) == pytest.approx(0.03456)
    assert SIZE_CAMPAIGN.constant_density_mass_kg(0.050) == pytest.approx(
        0.0925925925925926
    )
    assert SIZE_CAMPAIGN.discovery_mass_kg == pytest.approx(0.020)
    assert SIZE_CAMPAIGN.discovery_friction == pytest.approx(0.8)


def test_large_cube_robustness_families_are_75_plus_25_cases():
    families = ROBUSTNESS_CASE_FAMILIES

    assert families.edge_window_m(0.036) == pytest.approx(
        (0.036, 0.037, 0.038, 0.039, 0.040)
    )
    assert families.edge_window_m(0.040) == pytest.approx(
        (0.038, 0.039, 0.040, 0.041, 0.042)
    )
    assert families.edge_window_m(0.050) == pytest.approx(
        (0.046, 0.047, 0.048, 0.049, 0.050)
    )
    assert families.constant_density_case_count == 75
    assert families.fixed_mass_case_count == 25
    assert families.total_case_count == 100
    assert LARGE_CUBE_OPPOSED_FACE_PALM_DOWN.robustness.grid_case_count == 100
    assert families.boundary_probe_step_m == pytest.approx(0.0005)
    assert LARGE_CUBE_OPPOSED_FACE_PALM_DOWN.robustness.perturbation_count == 50
    assert LARGE_CUBE_OPPOSED_FACE_PALM_DOWN.robustness.required_pass_count == 45

    with pytest.raises(ValueError, match="inside edge_limits"):
        families.edge_window_m(0.035)


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        (
            "size_campaign",
            lambda config: config["size_campaign"]["budget"].__setitem__(
                "fine_samples_per_pitch", 49_999
            ),
        ),
        (
            "robustness",
            lambda config: config["robustness"]["case_families"].__setitem__(
                "edge_window_count", 3
            ),
        ),
    ],
)
def test_large_cube_versioned_campaign_blocks_reject_mutation(section, mutate):
    config = copy.deepcopy(_read_config())
    mutate(config)

    with pytest.raises(ValueError, match=section):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        (
            "acceptance",
            lambda config: config["acceptance"].__setitem__(
                "median_lift_m", 1e-9
            ),
        ),
        (
            "acceptance",
            lambda config: config["acceptance"].__setitem__(
                "max_palm_down_angle_deg", 180.0
            ),
        ),
        (
            "contact_topology",
            lambda config: config["contact_topology"].update(
                target_force_fraction=0.01,
                max_off_target_force_fraction=0.99,
            ),
        ),
    ],
)
def test_large_cube_rejects_relaxed_hard_constraints(section, mutate):
    config = copy.deepcopy(_read_config())
    mutate(config)

    with pytest.raises(ValueError, match=section):
        validate_config(config)


@pytest.mark.parametrize("edge_m", (0.0355, 0.0505))
def test_large_cube_rejects_edges_outside_declared_campaign(edge_m):
    config = copy.deepcopy(_read_config())
    config["cube"]["edge_m"] = edge_m

    with pytest.raises(ValueError, match="size_campaign range"):
        validate_config(config)


def test_existing_experiment_rejects_a_foreign_size_campaign():
    config_path = ROOT / "grasp_configs" / "left_opposed_face_palm_down.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert OPPOSED_FACE_PALM_DOWN.size_campaign is None
    config["size_campaign"] = SIZE_CAMPAIGN.as_config()

    with pytest.raises(ValueError, match="size_campaign is not allowed"):
        validate_config(config)


@pytest.mark.parametrize("edge_m", (0.036, 0.050))
def test_large_cube_scene_uses_half_extents_density_mass_and_support_height(edge_m):
    config = load_config(CONFIG_PATH)
    config["cube"]["edge_m"] = edge_m
    config["cube"]["mass_kg"] = SIZE_CAMPAIGN.constant_density_mass_kg(edge_m)

    model, info = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mass_kg = config["cube"]["mass_kg"]
    assert model.geom_size[info.cube_geom_id] == pytest.approx([edge_m / 2.0] * 3)
    assert model.body_mass[info.cube_body_id] == pytest.approx(mass_kg)
    assert model.body_inertia[info.cube_body_id] == pytest.approx(
        cube_inertia(edge_m, mass_kg)
    )
    cube_position = data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 3]
    assert cube_position[2] - edge_m / 2.0 == pytest.approx(
        config["scene"]["support_top_z_m"]
    )
    assert np.isfinite(data.qpos).all()
    assert model.body_jntnum[info.root_body_id] == 0
    assert model.body_parentid[info.root_body_id] == 0
    assert model.body_mocapid[info.root_body_id] == -1
