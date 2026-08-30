from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.scene import rpy_degrees_to_rotation_matrix
from xhand_grasp.tuning.pose_preserving_seed_campaign import (
    EXPECTED_SOURCE_COUNT,
    build_pose_preserving_seed_campaign,
    extract_pose_preserving_source,
    generate_hand_pose_only_candidates,
    initial_cube_world_pose,
    load_pose_preserving_seed_sources,
    quaternion_to_rotation_matrix,
    retarget_root_pose,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = (
    ROOT
    / "artifacts"
    / "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift"
    / "grasp_acquisition_high_thumb"
    / "trajectory_catalog"
    / "catalog.json"
)
TEMPLATE = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_pose_preserving_grasp.json"
)
SOURCE_IDS_IN_CATALOG_ORDER = (117060, 116060, 115060, 117061, 116062, 110064)
OVERRIDE_SOURCE_IDS = {117061, 116062, 110064}


@pytest.fixture(scope="module")
def sources() -> tuple[dict, ...]:
    return load_pose_preserving_seed_sources(CATALOG)


@pytest.fixture(scope="module")
def template() -> dict:
    return load_config(TEMPLATE)


def _source_artifact(source: dict, field: str) -> Path:
    return CATALOG.parent / source["provenance"][field]


def _without_sample_identity(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("hand_pose")
    result.pop("candidate_metadata")
    result.pop("run_context", None)
    return result


def test_loader_authenticates_six_sources_and_extracts_real_acquisition_state(
    sources,
):
    assert len(sources) == EXPECTED_SOURCE_COUNT
    assert tuple(source["source_order"] for source in sources) == tuple(range(6))
    assert tuple(source["source_candidate_id"] for source in sources) == (
        SOURCE_IDS_IN_CATALOG_ORDER
    )
    assert {
        source["source_candidate_id"]
        for source in sources
        if source["parameter_override_run"]
    } == OVERRIDE_SOURCE_IDS

    for source in sources:
        assert len(source["source_sha256"]) == 64
        assert len(source["provenance"]["catalog_sha256"]) == 64
        assert len(source["provenance"]["resolved_config_sha256"]) == 64
        assert len(source["provenance"]["trace_sha256"]) == 64

        config = load_config(_source_artifact(source, "resolved_config_artifact"))
        assert source["cube"] == config["cube"]
        assert source["scene"] == config["scene"]
        assert source["contact_topology"] == config["contact_topology"]
        assert source["initial_cube_world_pose"] == initial_cube_world_pose(config)
        assert source["grasp_targets_rad"] == config["control"][
            "grasp_targets_rad"
        ]
        assert len(source["fixed_object_sha256"]) == 64
        assert len(source["terminal_relation_sha256"]) == 64
        with np.load(
            _source_artifact(source, "trace_artifact"), allow_pickle=False
        ) as trace:
            step = int(trace["grasp_acquisition_step"])
            assert source["acquisition"]["step"] == step
            assert source["acquisition"]["time_s"] == pytest.approx(
                float(trace["time"][step]), abs=0.0
            )
            actuator_order = tuple(str(value) for value in trace["actuator_order"])
            expected_qpos = {
                name: float(trace["joint_qpos"][step, index])
                for index, name in enumerate(actuator_order)
            }
            assert source["acquisition"]["actuator_qpos_rad"] == expected_qpos
            np.testing.assert_array_equal(
                source["acquisition"]["cube_pose_world"]["position_m"],
                trace["cube_pos"][step],
            )


def test_terminal_cube_in_root_relation_reconstructs_recorded_world_pose(sources):
    for source in sources:
        acquisition = source["acquisition"]
        root = acquisition["root_pose_world"]
        cube = acquisition["cube_pose_world"]
        relative = source["terminal_cube_in_root"]
        root_rotation = quaternion_to_rotation_matrix(root["quaternion_wxyz"])
        relative_rotation = quaternion_to_rotation_matrix(
            relative["quaternion_wxyz"]
        )
        reconstructed_position = np.asarray(root["position_m"]) + (
            root_rotation @ np.asarray(relative["position_m"])
        )
        reconstructed_rotation = root_rotation @ relative_rotation

        np.testing.assert_allclose(
            reconstructed_position,
            cube["position_m"],
            rtol=0.0,
            atol=2e-15,
        )
        np.testing.assert_allclose(
            reconstructed_rotation,
            quaternion_to_rotation_matrix(cube["quaternion_wxyz"]),
            rtol=0.0,
            atol=2e-15,
        )


def test_rigid_retarget_transform_preserves_measured_terminal_relation(
    sources,
    template,
):
    for source in sources:
        acquisition = source["acquisition"]
        root_source = acquisition["root_pose_world"]
        cube_acquired = acquisition["cube_pose_world"]
        cube_configured = source["initial_cube_world_pose"]
        root_seed = retarget_root_pose(
            configured_cube_pose=cube_configured,
            acquired_cube_pose=cube_acquired,
            source_root_pose=root_source,
            reference_rpy_deg=template["hand_pose"]["rpy_deg"],
        )
        root_source_rotation = quaternion_to_rotation_matrix(
            root_source["quaternion_wxyz"]
        )
        root_seed_rotation = quaternion_to_rotation_matrix(
            root_seed["quaternion_wxyz"]
        )
        cube_acquired_rotation = quaternion_to_rotation_matrix(
            cube_acquired["quaternion_wxyz"]
        )
        cube_configured_rotation = quaternion_to_rotation_matrix(
            cube_configured["quaternion_wxyz"]
        )
        np.testing.assert_allclose(
            root_seed_rotation.T @ cube_configured_rotation,
            root_source_rotation.T @ cube_acquired_rotation,
            rtol=0.0,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            root_seed_rotation.T
            @ (
                np.asarray(cube_configured["position_m"])
                - np.asarray(root_seed["translation_m"])
            ),
            root_source_rotation.T
            @ (
                np.asarray(cube_acquired["position_m"])
                - np.asarray(root_source["position_m"])
            ),
            rtol=0.0,
            atol=2e-14,
        )


def test_source_extractor_rejects_a_forged_acquisition_event():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    entry = catalog["trajectories"][0]
    config = load_config(CATALOG.parent / entry["artifacts"]["resolved_config"])
    trace_path = CATALOG.parent / entry["artifacts"]["trace"]
    required = (
        "time",
        "actuator_order",
        "joint_qpos",
        "cube_pos",
        "cube_quat",
        "root_pos",
        "root_quat",
        "control_state",
        "grasp_acquired",
        "grasp_acquisition_step",
    )
    with np.load(trace_path, allow_pickle=False) as archive:
        traces = {name: np.array(archive[name], copy=True) for name in required}
    traces["grasp_acquired"][int(traces["grasp_acquisition_step"])] = False

    with pytest.raises(ValueError, match="latched VERIFY"):
        extract_pose_preserving_source(
            entry,
            config,
            traces,
            source_order=0,
        )


@pytest.mark.parametrize("unsafe_path", ["../escaped.json", "/tmp/absolute.json"])
def test_loader_rejects_non_relative_or_traversing_artifact_paths(
    tmp_path,
    unsafe_path,
):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    catalog["trajectories"][0]["artifacts"]["resolved_config"] = unsafe_path
    candidate = tmp_path / "catalog.json"
    candidate.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ValueError, match="safe and relative"):
        load_pose_preserving_seed_sources(candidate)


def test_loader_recomputes_declared_artifact_hashes(tmp_path):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    entry = copy.deepcopy(catalog["trajectories"][0])
    source_config = CATALOG.parent / entry["artifacts"]["resolved_config"]
    source_trace = CATALOG.parent / entry["artifacts"]["trace"]
    entry["artifacts"]["resolved_config"] = "resolved_config.json"
    entry["artifacts"]["trace"] = "trace.npz"
    entry["artifacts"]["sha256"]["resolved_config"] = "0" * 64
    shutil.copyfile(source_config, tmp_path / "resolved_config.json")
    shutil.copyfile(source_trace, tmp_path / "trace.npz")
    catalog["trajectories"] = [entry]
    catalog["trajectory_count"] = 1
    catalog["validated_grasp_count"] = 1
    candidate = tmp_path / "catalog.json"
    candidate.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_pose_preserving_seed_sources(candidate, expected_count=1)


def test_unperturbed_seed_preserves_source_cube_and_terminal_control(
    sources,
    template,
):
    original_sources = copy.deepcopy(sources)
    original_template = copy.deepcopy(template)

    for source in sources:
        candidate = generate_hand_pose_only_candidates(
            source,
            template,
            count=1,
            seed=20260821,
        )[0]
        config = candidate["config"]
        validate_config(config)
        assert config["cube"] == source["cube"]
        assert config["scene"] == source["scene"]
        assert config["contact_topology"] == source["contact_topology"]
        assert initial_cube_world_pose(config) == source["initial_cube_world_pose"]
        assert config["control"]["grasp_targets_rad"] == source[
            "grasp_targets_rad"
        ]
        assert config["control"]["pregrasp_targets_rad"] == template[
            "control"
        ]["pregrasp_targets_rad"]
        assert config["control"]["close_profile"] == template["control"][
            "close_profile"
        ]
        assert config["control"]["manipulation_delta_rad"] == {
            name: 0.0 for name in ACTIVE_ACTUATORS
        }
        assert set(config["candidate_metadata"]["sampled_fields"]) == {
            "hand_pose.translation_m",
            "hand_pose.rpy_deg",
        }
        assert config["candidate_metadata"]["seed"] is None
        assert config["candidate_metadata"]["fixed_object_sha256"] == source[
            "fixed_object_sha256"
        ]
        assert config["candidate_metadata"]["terminal_relation_sha256"] == source[
            "terminal_relation_sha256"
        ]
        if source["source_candidate_id"] in OVERRIDE_SOURCE_IDS:
            assert config["run_context"] == {
                "kind": "parameter_override_run"
            }
        else:
            assert "run_context" not in config

        cube_world = np.asarray(
            source["initial_cube_world_pose"]["position_m"], dtype=np.float64
        )
        root_world = np.asarray(config["hand_pose"]["translation_m"])
        rotation = rpy_degrees_to_rotation_matrix(config["hand_pose"]["rpy_deg"])
        local = np.asarray(
            config["candidate_metadata"]["candidate_cube_in_root_m"]
        )
        np.testing.assert_allclose(
            root_world + rotation @ local,
            cube_world,
            rtol=0.0,
            atol=2e-15,
        )

    assert sources == original_sources
    assert template == original_template


def test_unperturbed_seed_uses_template_rpy_with_source_cube_yaw_compensation(
    sources,
    template,
):
    template_rotation = rpy_degrees_to_rotation_matrix(
        template["hand_pose"]["rpy_deg"]
    )
    template_cube_yaw = float(template["cube"]["rpy_deg"][2])
    for source in sources:
        config = generate_hand_pose_only_candidates(
            source,
            template,
            count=1,
            seed=20260821,
        )[0]["config"]
        yaw_delta = float(source["cube"]["rpy_deg"][2]) - template_cube_yaw
        expected_rotation = (
            rpy_degrees_to_rotation_matrix([0.0, 0.0, yaw_delta])
            @ template_rotation
        )
        np.testing.assert_allclose(
            rpy_degrees_to_rotation_matrix(config["hand_pose"]["rpy_deg"]),
            expected_rotation,
            rtol=0.0,
            atol=2e-14,
        )
        assert config["candidate_metadata"]["candidate_cube_in_root_m"] == source[
            "terminal_cube_in_root"
        ]["position_m"]


def test_cross_size_sources_are_explicit_overrides_and_pose_envelope_is_audited(
    sources,
    template,
):
    status_by_id = {}
    for source in sources:
        config = generate_hand_pose_only_candidates(
            source,
            template,
            count=1,
            seed=20260821,
        )[0]["config"]
        metadata = config["candidate_metadata"]
        status_by_id[source["source_candidate_id"]] = metadata
        if source["source_candidate_id"] in OVERRIDE_SOURCE_IDS:
            assert config["run_context"] == {"kind": "parameter_override_run"}
            assert metadata["canonical_v6_material"] is False
        else:
            assert "run_context" not in config
            assert metadata["canonical_v6_material"] is True
    assert status_by_id[110064]["canonical_v6_pose_envelope"] is False
    assert status_by_id[117061]["canonical_v6_pose_envelope"] is True
    assert status_by_id[116062]["canonical_v6_pose_envelope"] is True


def test_hand_pose_only_generation_is_seeded_stable_and_id_sorted(
    sources,
    template,
):
    source = sources[0]
    first = generate_hand_pose_only_candidates(
        source,
        template,
        count=8,
        seed=20260821,
    )
    repeated = generate_hand_pose_only_candidates(
        source,
        template,
        count=8,
        seed=20260821,
    )
    different_seed = generate_hand_pose_only_candidates(
        source,
        template,
        count=8,
        seed=20260822,
    )

    assert first == repeated
    assert first[0] == different_seed[0]
    assert first[0]["candidate_sha256"] == different_seed[0]["candidate_sha256"]
    assert first[0]["config"]["candidate_metadata"]["seed"] is None
    assert first[1]["config"]["candidate_metadata"]["seed"] == 20260821
    assert first[1:] != different_seed[1:]
    ids = [candidate["candidate_id"] for candidate in first]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
    reference_physics = _without_sample_identity(first[0]["config"])
    hand_poses = []
    for candidate in first:
        validate_config(candidate["config"])
        assert _without_sample_identity(candidate["config"]) == reference_physics
        hand_poses.append(candidate["config"]["hand_pose"])
    assert len({json.dumps(pose, sort_keys=True) for pose in hand_poses}) == len(first)


def test_six_source_campaign_has_stable_group_order_and_content_hash(
    sources,
    template,
):
    first = build_pose_preserving_seed_campaign(
        sources,
        template,
        count_per_source=3,
        seed=20260821,
    )
    repeated = build_pose_preserving_seed_campaign(
        sources,
        template,
        count_per_source=3,
        seed=20260821,
    )

    assert first == repeated
    assert first["source_count"] == 6
    assert first["candidate_count"] == 18
    assert len(first["campaign_sha256"]) == 64
    assert [source["source_candidate_id"] for source in first["sources"]] == list(
        SOURCE_IDS_IN_CATALOG_ORDER
    )
    assert [candidate["source_order"] for candidate in first["candidates"]] == [
        source_order for source_order in range(6) for _ in range(3)
    ]
    for source_summary in first["sources"]:
        assert source_summary["candidate_ids"] == sorted(
            source_summary["candidate_ids"]
        )

    with pytest.raises(ValueError, match="stable catalog order"):
        build_pose_preserving_seed_campaign(
            reversed(sources),
            template,
            count_per_source=1,
            seed=20260821,
        )
