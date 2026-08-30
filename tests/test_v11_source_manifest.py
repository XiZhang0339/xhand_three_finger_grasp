from __future__ import annotations

import json
from pathlib import Path

import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import build_campaign_manifest
from xhand_grasp.artifacts import file_sha256
from xhand_grasp.config import load_config
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.actual_qpos_sources import (
    ACTUAL_QPOS_SOURCE_MANIFEST_SCHEMA_VERSION,
    AUTHENTICATED_GRASP_SUCCESS,
    MINIMUM_STABLE_WINDOW_S,
    load_actual_qpos_sources,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = (
    "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift"
)
SOURCE_EXPERIMENT_ID = (
    "left_opposed_face_palm_down_larger_actual_contact_grasp_pose_"
    "smooth_vertical_lift"
)
PRIMARY_CANDIDATE_ID = 4864000014401350
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift.json"
)
SOURCE_MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift"
    / "source_manifests"
    / "v10_measured_grasp_sources.json"
)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def source_manifest() -> dict:
    return json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sources(config):
    # This is intentionally the production loader: it authenticates every
    # config/result/trace member and recomputes the stable contact window.
    return load_actual_qpos_sources(config)


def test_v11_source_manifest_declares_eight_authenticated_records(
    source_manifest, sources
):
    declared = source_manifest["sources"]

    assert source_manifest["actual_qpos_source_manifest_schema_version"] == (
        ACTUAL_QPOS_SOURCE_MANIFEST_SCHEMA_VERSION
    )
    assert source_manifest["experiment_id"] == EXPERIMENT_ID
    assert source_manifest["source_experiment_id"] == SOURCE_EXPERIMENT_ID
    assert len(declared) == len(sources) == 8
    assert all(
        record["source_kind"] == AUTHENTICATED_GRASP_SUCCESS
        for record in declared
    )
    assert all(source.source_kind == AUTHENTICATED_GRASP_SUCCESS for source in sources)
    assert len({source.pose_id for source in sources}) == 8


def test_v11_primary_anchor_is_first_and_is_the_certified_84mm_pose(
    source_manifest, sources
):
    declared = source_manifest["sources"]
    primary = declared[0]
    loaded = sources[0]

    assert source_manifest["sampling_policy"] == {
        "primary_candidate_id": PRIMARY_CANDIDATE_ID,
        "primary_fraction": 0.7,
        "secondary_fraction": 0.3,
    }
    assert primary["role"] == "primary_anchor"
    assert primary["candidate_id"] == PRIMARY_CANDIDATE_ID
    assert loaded.source_index == 0
    assert loaded.config["candidate_metadata"]["candidate_id"] == (
        PRIMARY_CANDIDATE_ID
    )
    assert loaded.config["cube"]["edge_m"] == pytest.approx(0.084)
    assert loaded.pose_id == primary["pose_id"]


def test_v11_secondary_sources_are_authenticated_82_to_84mm_neighbors(
    source_manifest, sources
):
    declared_secondary = source_manifest["sources"][1:]
    loaded_secondary = sources[1:]

    assert len(declared_secondary) == len(loaded_secondary) == 7
    assert all(
        record["role"] == "secondary_authenticated_seed"
        for record in declared_secondary
    )
    edges = {float(source.config["cube"]["edge_m"]) for source in loaded_secondary}
    assert sorted(edges) == pytest.approx([0.082, 0.083, 0.084])
    assert all(0.082 <= edge <= 0.084 for edge in edges)
    assert all(
        source.config["experiment_id"] == SOURCE_EXPERIMENT_ID
        for source in loaded_secondary
    )


def test_v11_all_source_hashes_and_250ms_windows_authenticate(
    source_manifest, sources
):
    declared = source_manifest["sources"]

    for record, source in zip(declared, sources, strict=True):
        hashes = record["sha256"]
        config_path = ROOT / record["config_path"]
        result_path = ROOT / record["result_path"]
        trace_path = ROOT / record["trace_path"]

        assert file_sha256(config_path) == hashes["config"] == source.config_sha256
        assert file_sha256(result_path) == hashes["result"] == source.result_sha256
        assert file_sha256(trace_path) == hashes["trace"] == source.trace_sha256
        assert canonical_sha256(source.config) == hashes["config_semantic"]
        assert source.manifest_schema_version == 2
        assert source.gate_evidence_authenticated is True
        assert source.eligible_as_success_evidence is True
        assert source.stable_window_sample_count == 250
        assert source.stable_window_end_step - source.stable_window_start_step + 1 == 250
        assert source.stable_window_duration_s is not None
        assert source.stable_window_duration_s >= MINIMUM_STABLE_WINDOW_S - 1e-9


def test_v11_campaign_manifest_binds_the_v11_config_and_source_manifest(config):
    campaign_manifest = build_campaign_manifest(CONFIG_PATH, seed=20260821)
    relative = config["relative_wrist_pose_search"]

    assert campaign_manifest["experiment_id"] == EXPERIMENT_ID
    assert campaign_manifest["seed"] == 20260821
    assert Path(campaign_manifest["config_path"]) == CONFIG_PATH.resolve()
    assert campaign_manifest["config_sha256"] == file_sha256(CONFIG_PATH)
    assert campaign_manifest["actual_qpos_source_manifest_path"] == str(
        SOURCE_MANIFEST_PATH.relative_to(ROOT)
    )
    assert campaign_manifest["actual_qpos_source_manifest_sha256"] == file_sha256(
        SOURCE_MANIFEST_PATH
    )
    assert len(campaign_manifest["campaign_input_sha256"]) == 64
    assert relative["anchor_sampling"]["primary_candidate_id"] == (
        PRIMARY_CANDIDATE_ID
    )
    assert relative["anchor_sampling"]["certified_neighbor_edges_m"] == pytest.approx(
        [0.082, 0.083, 0.084]
    )
