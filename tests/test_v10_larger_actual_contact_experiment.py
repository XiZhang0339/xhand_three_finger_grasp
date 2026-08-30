from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
import pytest

import xhand_grasp.cli as cli
from xhand_grasp.actual_contact_grasp_pose_catalog import (
    build_campaign_manifest,
    initialize_or_resume_campaign,
)
from xhand_grasp.config import load_config, validate_config
from xhand_grasp.actual_contact_grasp_pose_catalog import build_campaign_manifest
from xhand_grasp.artifacts import file_sha256
from xhand_grasp.experiment import ACTIVE_ACTUATORS, get_experiment, resolve_experiment
from xhand_grasp.experiments.opposed_face_palm_down_larger_actual_contact_grasp_pose_smooth_vertical_lift import (
    ARTIFACT_ROOT,
    BRIDGE_EDGES_M,
    CAMPAIGN,
    CONTACT_PRELOAD_TARGET_BOUNDS_RAD,
    CONTROL_PROTOCOL,
    EDGES_M,
    EXPERIMENT_DEFINITION,
    EXPERIMENT_ID,
    POSE_CONSTRAINTS,
    PRECONTACT_TARGET_BOUNDS_RAD,
    SIZE_CONTINUATION_POLICY,
    SOURCE_POSE_MANIFEST,
    THUMB_ACTUAL_CENTERS_RAD,
    THUMB_BEND_ACTUATOR,
    VALIDATION_LABELS,
)
from xhand_grasp.scene import build_model, cube_inertia
from xhand_grasp.tuning.actual_contact_grasp_pose_dynamic import (
    generate_controller_seeds,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)


@pytest.fixture
def v10_config() -> dict:
    return load_config(CONFIG_PATH)


def test_v10_is_independently_registered_and_version_locked(v10_config):
    definition = get_experiment(EXPERIMENT_ID)

    assert v10_config["schema_version"] == 10
    assert resolve_experiment(v10_config) is definition
    assert definition is EXPERIMENT_DEFINITION
    assert definition.artifact_root == ARTIFACT_ROOT
    assert definition.actual_contact_grasp_pose_campaign is CAMPAIGN
    assert definition.tuning_strategy == (
        "actual_contact_grasp_pose_smooth_vertical_lift"
    )
    assert v10_config["actual_contact_grasp_pose_campaign"] == CAMPAIGN.as_config()
    assert v10_config["robustness"] == definition.robustness.as_config()


def test_v10_declares_exact_size_thumb_material_and_budget_campaign(v10_config):
    assert EDGES_M == pytest.approx(tuple(value / 1000 for value in range(72, 91)))
    assert THUMB_ACTUAL_CENTERS_RAD == pytest.approx(
        (1.40, 1.45, 1.50, 1.55, 1.60)
    )
    assert CAMPAIGN.fixed_mass_kg == pytest.approx(0.160)
    assert CAMPAIGN.friction == pytest.approx(0.8)
    assert CAMPAIGN.cell_count == 95
    assert CAMPAIGN.quick_static_sample_count == 190_000
    assert CAMPAIGN.full_static_sample_count == 950_000
    assert CAMPAIGN.maximum_dynamic_grasp_candidate_count == 3_040
    assert CAMPAIGN.initial_target_success_count == 1
    assert CAMPAIGN.selected_trajectory_count == 5
    assert CAMPAIGN.minimum_distinct_edges == 3
    assert CAMPAIGN.minimum_thumb_bands == 2
    assert v10_config["search"]["budget"] == CAMPAIGN.budget_config()
    assert CONTROL_PROTOCOL.close_duration_options_s == (
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
    )


def test_v10_expands_pose_and_active_actuator_search_envelopes():
    bounds = EXPERIMENT_DEFINITION.search_bounds

    assert POSE_CONSTRAINTS.root_cube_distance_m == pytest.approx((0.135, 0.190))
    assert dict(POSE_CONSTRAINTS.cube_position_in_root_m) == {
        "x": (0.085, 0.130),
        "y": (-0.045, -0.005),
        "z": (0.090, 0.140),
    }
    assert bounds.cube_position_in_root_m == POSE_CONSTRAINTS.cube_position_in_root_m
    assert CONTACT_PRELOAD_TARGET_BOUNDS_RAD == {
        THUMB_BEND_ACTUATOR: (1.40, 1.75),
        "left_hand_thumb_rota_joint1_actuator": (-0.10, 0.80),
        "left_hand_thumb_rota_joint2_actuator": (0.45, 1.10),
        "left_hand_index_bend_joint_actuator": (-0.15, 0.15),
        "left_hand_index_joint1_actuator": (0.20, 0.90),
        "left_hand_index_joint2_actuator": (0.80, 1.80),
        "left_hand_mid_joint1_actuator": (0.35, 1.15),
        "left_hand_mid_joint2_actuator": (0.65, 1.70),
    }
    assert bounds.actuator_targets_rad == CONTACT_PRELOAD_TARGET_BOUNDS_RAD
    assert PRECONTACT_TARGET_BOUNDS_RAD[THUMB_BEND_ACTUATOR] == (0.90, 1.50)
    assert bounds.pregrasp_targets_rad == PRECONTACT_TARGET_BOUNDS_RAD


def test_v10_controller_seed_search_uses_registered_long_close_probes(v10_config):
    seeds = generate_controller_seeds(v10_config, source_candidate_id=10)

    assert len(seeds) == 8
    assert {seed.close_s for seed in seeds}.issubset(
        set(CONTROL_PROTOCOL.close_duration_options_s)
    )
    assert {1.75, 2.0}.issubset({seed.close_s for seed in seeds})


def test_v10_cli_dispatches_registered_actual_contact_runner(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "campaign"
    observed = {}

    def fake_runner(config_path, output_dir, **kwargs):
        observed.update(
            {"config_path": config_path, "output_dir": output_dir, **kwargs}
        )
        initialize_or_resume_campaign(
            output_dir,
            build_campaign_manifest(config_path, seed=kwargs["seed"]),
            resume=False,
        )
        return {"grasp_success_count": 1, "full_success_count": 1}

    monkeypatch.setattr(cli, "_load_v9_tune_runner", lambda: fake_runner)
    arguments = cli.build_parser().parse_args(
        [
            "tune",
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output),
            "--target-success-count",
            "1",
            "--workers",
            "2",
        ]
    )

    assert cli.command_tune(arguments) == 0
    assert observed == {
        "config_path": CONFIG_PATH.resolve(),
        "output_dir": output.resolve(),
        "resume": False,
        "target_success_count": 1,
        "workers": 2,
        "seed": 20260821,
    }
    assert EXPERIMENT_ID in capsys.readouterr().out


def test_v10_actuator_search_envelope_stays_inside_model_ctrl_limits(v10_config):
    model, _ = build_model(v10_config)

    for name in ACTIVE_ACTUATORS:
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        ctrl_low, ctrl_high = model.actuator_ctrlrange[actuator_id]
        for search_range in (
            CONTACT_PRELOAD_TARGET_BOUNDS_RAD[name],
            PRECONTACT_TARGET_BOUNDS_RAD[name],
        ):
            assert ctrl_low <= search_range[0] <= search_range[1] <= ctrl_high


def test_v10_bridge_is_internal_and_fixed_mass_labels_are_explicit(v10_config):
    assert BRIDGE_EDGES_M == pytest.approx((0.068, 0.069, 0.070, 0.071))
    assert not set(BRIDGE_EDGES_M) & set(EDGES_M)
    assert SIZE_CONTINUATION_POLICY["published_edges_m"] == list(EDGES_M)
    assert SIZE_CONTINUATION_POLICY["publish_bridge_results"] is False
    assert v10_config["size_continuation"] == SIZE_CONTINUATION_POLICY
    assert CAMPAIGN.validation_labels == VALIDATION_LABELS
    assert v10_config["actual_contact_grasp_pose_campaign"][
        "validation_labels"
    ] == VALIDATION_LABELS
    assert VALIDATION_LABELS == {
        "grasp": "validated_fixed_mass_grasp_ablation",
        "manipulation": "validated_fixed_mass_manipulation_ablation",
        "robust": "validated_fixed_mass_robust_full_success_ablation",
    }
    assert CAMPAIGN.source_pose_manifest == SOURCE_POSE_MANIFEST
    assert SOURCE_POSE_MANIFEST.startswith(ARTIFACT_ROOT + "/source_manifests/")
    assert EXPERIMENT_DEFINITION.artifact_root != (
        "artifacts/"
        "left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift"
    )


def test_v10_resume_manifest_binds_authenticated_qpos_source_manifest():
    manifest = build_campaign_manifest(CONFIG_PATH, seed=20260821)
    source_path = ROOT / SOURCE_POSE_MANIFEST

    assert manifest["experiment_id"] == EXPERIMENT_ID
    assert manifest["actual_qpos_source_manifest_path"] == SOURCE_POSE_MANIFEST
    assert manifest["actual_qpos_source_manifest_sha256"] == file_sha256(
        source_path
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["cube"].__setitem__("edge_m", 0.071), "edge"),
        (lambda value: value["cube"].__setitem__("edge_m", 0.0725), "edge"),
        (lambda value: value["cube"].__setitem__("mass_kg", 0.161), "mass"),
        (lambda value: value["cube"].__setitem__("friction", 0.81), "friction"),
    ],
)
def test_v10_rejects_non_campaign_size_or_material(v10_config, mutation, message):
    changed = copy.deepcopy(v10_config)
    mutation(changed)

    with pytest.raises(ValueError, match=message):
        validate_config(changed)


@pytest.mark.parametrize("edge_m", (0.072, 0.090))
def test_v10_scene_uses_fixed_mass_inertia_support_height_and_free_cube(
    v10_config, edge_m
):
    config = copy.deepcopy(v10_config)
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
    assert model.body_jntnum[info.cube_body_id] == 1
    assert model.jnt_type[model.body_jntadr[info.cube_body_id]] == mujoco.mjtJoint.mjJNT_FREE
    assert np.isfinite(data.qpos).all()
    assert model.body_jntnum[info.root_body_id] == 0
    assert model.body_mocapid[info.root_body_id] == -1
