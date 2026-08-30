from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import search_pose_preserving_static as static_screen
from xhand_grasp.config import ACTIVE_ACTUATORS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_pose_preserving_grasp.json"
)
CATALOG = static_screen.DEFAULT_SOURCE_CATALOG


def _assert_sha256(value: str) -> None:
    assert len(value) == 64
    int(value, 16)


def test_default_catalog_alias_reconstructs_the_trace_acquisition_seed():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    expected_id = catalog["aliases"][static_screen.DEFAULT_SOURCE_TRAJECTORY]
    source = static_screen.load_source_trajectory(CATALOG)

    assert source.requested_trajectory == "best_grasp"
    assert source.trajectory_id == expected_id
    _assert_sha256(source.catalog_sha256)
    _assert_sha256(source.resolved_config_sha256)
    _assert_sha256(source.trace_sha256)
    _assert_sha256(source.source_sha256)

    with np.load(source.trace_path, allow_pickle=False) as archive:
        acquisition = int(archive["grasp_acquisition_step"])
        active_ids = np.asarray(
            [source.model.actuator(name).id for name in ACTIVE_ACTUATORS]
        )
        expected_qpos = archive["joint_qpos"][acquisition, active_ids]
        expected_relative = (
            static_screen._quaternion_rotation(archive["root_quat"][acquisition]).T
            @ (
                archive["cube_pos"][acquisition]
                - archive["root_pos"][acquisition]
            )
        )

    np.testing.assert_array_equal(source.acquisition_qpos_rad, expected_qpos)
    np.testing.assert_allclose(
        source.acquisition_cube_in_root_m,
        expected_relative,
        rtol=0.0,
        atol=1e-15,
    )
    script_source = Path(static_screen.__file__).read_text(encoding="utf-8")
    assert "ACQUISITION_QPOS" not in script_source
    assert expected_id not in script_source


def test_two_catalog_trajectories_produce_distinct_content_addressed_seeds():
    first = static_screen.search(
        CONFIG,
        samples=1,
        seed=20260821,
        retain=1,
        source_catalog=CATALOG,
        trajectory="best_grasp",
    )
    second = static_screen.search(
        CONFIG,
        samples=1,
        seed=20260821,
        retain=1,
        source_catalog=CATALOG,
        trajectory="grasp_116062",
    )

    assert first["schema_version"] == 2
    assert second["schema_version"] == 2
    assert first["source_trajectory_id"] != second["source_trajectory_id"]
    assert first["source_sha256"] != second["source_sha256"]
    assert (
        first["source"]["acquisition_qpos_rad"]
        != second["source"]["acquisition_qpos_rad"]
    )
    assert (
        first["source"]["acquisition_cube_in_root_m"]
        != second["source"]["acquisition_cube_in_root_m"]
    )
    assert first["source"]["fixed_cube"]["edge_m"] != second["source"][
        "fixed_cube"
    ]["edge_m"]
    assert first["source"]["fixed_cube"]["mass_kg"] != second["source"][
        "fixed_cube"
    ]["mass_kg"]
    for result in (first, second):
        assert result["sampled_fields"] == ["hand_pose"]
        assert result["retained_count"] == 1
        candidate = result["candidates"][0]
        assert candidate["qpos_rad"] == result["source"]["acquisition_qpos_rad"]
        assert candidate["acquisition_qpos_rad"] == candidate["qpos_rad"]
        assert candidate["terminal_targets_rad"] == result["source"][
            "terminal_targets_rad"
        ]
        assert candidate["source_trajectory_id"] == result[
            "source_trajectory_id"
        ]
        assert candidate["source_sha256"] == result["source_sha256"]
        assert set(candidate["distal_witness"]) == {"thumb", "index", "mid"}
        assert set(candidate["active_nondistal_gap"]) == {
            "thumb",
            "index",
            "mid",
        }
        for index, finger in enumerate(("thumb", "index", "mid")):
            witness = candidate["distal_witness"][finger]
            assert witness["classified_face"] == witness["target_face"]
            assert witness["signed_distance_m"] == pytest.approx(
                candidate["target_gap_m"][index], abs=1e-15
            )
            assert len(witness["cube_point_world_m"]) == 3
            assert len(witness["distal_point_world_m"]) == 3
            assert witness["distal_geom_name"]
            nondistal = candidate["active_nondistal_gap"][finger]
            assert np.isfinite(nondistal["signed_distance_m"])
            assert nondistal["geom_name"]
        assert candidate["minimum_active_nondistal_gap_m"] == pytest.approx(
            min(
                value["signed_distance_m"]
                for value in candidate["active_nondistal_gap"].values()
            ),
            abs=1e-15,
        )


def test_sampling_changes_only_hand_root_pose_and_keeps_source_values_fixed():
    result = static_screen.search(
        CONFIG,
        samples=32,
        seed=20260821,
        retain=32,
        source_catalog=CATALOG,
        trajectory="grasp_116060",
    )
    source = static_screen.load_source_trajectory(CATALOG, "grasp_116060")

    assert len(result["candidates"]) >= 2
    expected_qpos = source.acquisition_qpos_by_actuator_rad
    for candidate in result["candidates"]:
        assert candidate["qpos_rad"] == expected_qpos
        assert candidate["acquisition_qpos_rad"] == expected_qpos
        assert candidate["terminal_targets_rad"] == source.terminal_targets_rad
        np.testing.assert_array_equal(
            candidate["fixed_cube_world_position_m"],
            source.initial_cube_world_m,
        )
        np.testing.assert_array_equal(
            candidate["fixed_cube_world_quaternion_wxyz"],
            source.initial_cube_quaternion_wxyz,
        )
    root_poses = {
        (
            tuple(candidate["hand_translation_m"]),
            tuple(candidate["hand_rpy_deg"]),
        )
        for candidate in result["candidates"]
    }
    assert len(root_poses) >= 2
    assert result["source"]["fixed_cube"] == source.config["cube"]
    assert result["source"]["fixed_scene"] == source.config["scene"]
    assert result["source"]["terminal_targets_rad"] == (
        source.terminal_targets_rad
    )
    np.testing.assert_array_equal(
        result["source"]["fixed_initial_cube_world_m"],
        source.initial_cube_world_m,
    )


def test_direct_cli_accepts_source_catalog_and_trajectory_options():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "search_pose_preserving_static.py"),
            "--config",
            str(CONFIG),
            "--source-catalog",
            str(CATALOG),
            "--trajectory",
            "grasp_116060",
            "--samples",
            "1",
            "--retain",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["source_trajectory_id"] == "grasp_116060"
    assert result["source"]["requested_trajectory"] == "grasp_116060"
