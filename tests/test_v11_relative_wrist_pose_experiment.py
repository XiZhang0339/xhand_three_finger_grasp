from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import mujoco
import pytest

from xhand_grasp.actual_contact_capability import (
    ACTUAL_CONTACT_SCHEMA_VERSIONS,
    is_actual_contact_definition,
    resolve_actual_contact_definition,
)
from xhand_grasp.config import load_config, resolved_pose_constraint_values, validate_config
from xhand_grasp.experiment import RelativeWristPoseSearchParameters, get_experiment
from xhand_grasp.experiments.opposed_face_palm_down_larger_actual_contact_grasp_pose_smooth_vertical_lift import (
    EXPERIMENT_DEFINITION as V10_EXPERIMENT,
)
from xhand_grasp.experiments.opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_smooth_vertical_lift import (
    ARTIFACT_ROOT,
    CAMPAIGN,
    CLOCKWISE_ORBIT_DEG,
    DENSITY_REVALIDATION_KG_M3,
    EDGES_M,
    EXPERIMENT_DEFINITION,
    EXPERIMENT_ID,
    POSE_CONSTRAINTS,
    PRIMARY_ANCHOR_CANDIDATE_ID,
    RELATIVE_WRIST_POSE_SEARCH,
    SOURCE_POSE_MANIFEST,
    THUMB_ACTUAL_CENTERS_RAD,
    VALIDATION_LABELS,
)
from xhand_grasp.scene import build_model, cube_inertia


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift.json"
)
V10_CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)


@pytest.fixture
def v11_config() -> dict:
    return load_config(CONFIG_PATH)


def test_v11_is_registered_as_an_actual_contact_capability(v11_config):
    definition = get_experiment(EXPERIMENT_ID)

    assert v11_config["schema_version"] == 11
    assert definition is EXPERIMENT_DEFINITION
    assert resolve_actual_contact_definition(v11_config) is definition
    assert is_actual_contact_definition(definition)
    assert 11 in ACTUAL_CONTACT_SCHEMA_VERSIONS
    assert definition.artifact_root == ARTIFACT_ROOT
    assert definition.relative_wrist_pose_search is RELATIVE_WRIST_POSE_SEARCH
    assert V10_EXPERIMENT.relative_wrist_pose_search is None


def test_v11_declares_exact_edges_material_anchor_and_orbit(v11_config):
    assert EDGES_M == pytest.approx(tuple(value / 1000 for value in range(85, 105)))
    assert THUMB_ACTUAL_CENTERS_RAD == pytest.approx(
        (1.40, 1.45, 1.50, 1.55, 1.60)
    )
    assert CLOCKWISE_ORBIT_DEG == pytest.approx(
        (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0)
    )
    assert CAMPAIGN.fixed_mass_kg == pytest.approx(0.160)
    assert CAMPAIGN.friction == pytest.approx(0.8)
    assert CAMPAIGN.validation_labels == VALIDATION_LABELS
    assert CAMPAIGN.source_pose_manifest == SOURCE_POSE_MANIFEST
    assert PRIMARY_ANCHOR_CANDIDATE_ID == 4864000014401350
    assert RELATIVE_WRIST_POSE_SEARCH.primary_anchor_fraction == pytest.approx(0.70)
    assert RELATIVE_WRIST_POSE_SEARCH.certified_neighbor_fraction == pytest.approx(0.30)
    assert RELATIVE_WRIST_POSE_SEARCH.certified_neighbor_edges_m == pytest.approx(
        (0.082, 0.083, 0.084)
    )
    assert DENSITY_REVALIDATION_KG_M3 == pytest.approx(740.7407407407407)
    assert v11_config["actual_contact_grasp_pose_campaign"] == CAMPAIGN.as_config()


def test_v11_relative_pose_contract_uses_cube_frame_and_clockwise_sign(v11_config):
    expected = RELATIVE_WRIST_POSE_SEARCH.as_config(
        edge_count=len(EDGES_M), thumb_band_count=len(THUMB_ACTUAL_CENTERS_RAD)
    )
    declared = v11_config["relative_wrist_pose_search"]

    assert declared == expected
    assert declared["root_delta_cube_m"] == {
        "x": [-0.012, 0.012],
        "y": [-0.012, 0.012],
        "z": [-0.015, 0.015],
    }
    assert declared["wrist_local_rotvec_deg"] == {
        "x": [-6.0, 6.0],
        "y": [-6.0, 6.0],
        "z": [-6.0, 6.0],
    }
    assert declared["max_wrist_local_rotvec_norm_deg"] == pytest.approx(8.0)
    assert declared["root_cube_distance_m"] == pytest.approx([0.135, 0.210])
    convention = declared["orbit_convention"]
    assert convention["axis"] == "cube_local_+Z"
    assert convention["positive_direction"] == (
        "clockwise_viewed_from_cube_local_+Z"
    )
    assert convention["internal_mathematical_angle_sign"] == -1
    assert convention["translation_and_orientation_orbit_together"] is True
    assert convention["world_root_pose_rederived_from_relative_transform"] is True


def test_v11_budget_is_per_edge_thumb_and_orbit_stratum():
    budget = RELATIVE_WRIST_POSE_SEARCH.budget_config(
        edge_count=len(EDGES_M), thumb_band_count=len(THUMB_ACTUAL_CENTERS_RAD)
    )

    assert budget["stratum_count"] == 700
    assert budget["static_sample_count"] == 700_000
    assert budget["expansion_sample_count"] == 1_400_000
    assert budget["maximum_dynamic_grasp_candidate_count"] == 1_440
    assert budget["maximum_local_refinement_count"] == 2_560
    assert budget["maximum_exact_candidate_count"] == 40
    assert budget["max_manipulation_pose_count"] == 20
    assert budget["manipulation_probe_count"] == 17
    assert budget["manipulation_candidates_per_pose"] == 96
    assert budget["manipulation_refine_pose_count"] == 8
    assert budget["manipulation_refine_per_pose"] == 128


def test_v11_template_preserves_the_certified_anchor_relative_pose(v11_config):
    resolved = resolved_pose_constraint_values(v11_config)

    assert v11_config["cube"]["edge_m"] == pytest.approx(0.085)
    assert resolved["cube_position_in_root_m"] == pytest.approx(
        (0.10335902227468312, -0.023962169942032333, 0.10847905351436386)
    )
    assert resolved["root_cube_distance_m"] == pytest.approx(
        0.15173983697526994
    )
    assert POSE_CONSTRAINTS.contains_cube_position(
        resolved["cube_position_in_root_m"]
    )
    assert resolved["finger_down_tilt_deg"] == pytest.approx(30.0)
    assert resolved["palm_plane_ground_angle_deg"] == pytest.approx(
        30.008301409443686
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["relative_wrist_pose_search"][
            "clockwise_orbit_deg"
        ].append(17.5),
        lambda value: value["relative_wrist_pose_search"]["root_delta_cube_m"][
            "x"
        ].__setitem__(1, 0.013),
        lambda value: value["relative_wrist_pose_search"][
            "wrist_local_rotvec_deg"
        ]["z"].__setitem__(0, -7.0),
        lambda value: value["relative_wrist_pose_search"].__setitem__(
            "max_wrist_local_rotvec_norm_deg", 9.0
        ),
        lambda value: value["relative_wrist_pose_search"]["anchor_sampling"].__setitem__(
            "primary_candidate_id", 1
        ),
        lambda value: value["relative_wrist_pose_search"]["budget"].__setitem__(
            "static_sample_count", 1
        ),
    ),
)
def test_v11_rejects_changes_to_the_registered_relative_pose_contract(
    v11_config, mutation
):
    changed = copy.deepcopy(v11_config)
    mutation(changed)

    with pytest.raises(ValueError, match="relative_wrist_pose_search"):
        validate_config(changed)


def test_v10_rejects_v11_relative_pose_settings():
    config = load_config(V10_CONFIG_PATH)
    config["relative_wrist_pose_search"] = RELATIVE_WRIST_POSE_SEARCH.as_config(
        edge_count=len(EDGES_M), thumb_band_count=len(THUMB_ACTUAL_CENTERS_RAD)
    )

    with pytest.raises(ValueError, match="only allowed in schema v11"):
        validate_config(config)


def test_relative_wrist_pose_parameter_type_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="start at zero"):
        replace(RELATIVE_WRIST_POSE_SEARCH, clockwise_orbit_deg=(2.5, 5.0))
    with pytest.raises(ValueError, match="straddle zero"):
        replace(
            RELATIVE_WRIST_POSE_SEARCH,
            root_delta_cube_m={
                "x": (0.0, 0.012),
                "y": (-0.012, 0.012),
                "z": (-0.015, 0.015),
            },
        )
    with pytest.raises(ValueError, match="sum to one"):
        replace(
            RELATIVE_WRIST_POSE_SEARCH,
            primary_anchor_fraction=0.8,
            certified_neighbor_fraction=0.3,
        )
    with pytest.raises(ValueError, match="admit every per-axis"):
        replace(RELATIVE_WRIST_POSE_SEARCH, max_wrist_local_rotvec_norm_deg=5.0)


@pytest.mark.parametrize("edge_m", (0.085, 0.104))
def test_v11_scene_uses_fixed_mass_support_height_inertia_and_freejoint(
    v11_config, edge_m
):
    config = copy.deepcopy(v11_config)
    config["cube"]["edge_m"] = edge_m
    model, info = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert model.geom_size[info.cube_geom_id] == pytest.approx([edge_m / 2.0] * 3)
    assert model.body_mass[info.cube_body_id] == pytest.approx(0.160)
    assert model.body_inertia[info.cube_body_id] == pytest.approx(
        cube_inertia(edge_m, 0.160)
    )
    cube_position = data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 3]
    assert cube_position[2] - edge_m / 2.0 == pytest.approx(
        config["scene"]["support_top_z_m"]
    )
    assert model.jnt_type[info.cube_joint_id] == mujoco.mjtJoint.mjJNT_FREE


def test_relative_wrist_pose_parameter_is_a_public_versioned_value_type():
    assert isinstance(RELATIVE_WRIST_POSE_SEARCH, RelativeWristPoseSearchParameters)
