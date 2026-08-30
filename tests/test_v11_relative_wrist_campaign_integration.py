from __future__ import annotations

import copy
from pathlib import Path

import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.tuning.actual_contact_grasp_pose import (
    actual_contact_search_cells,
    generate_actual_contact_pose_candidates,
    select_dynamic_local_refinement_parents,
)
from xhand_grasp.tuning.actual_qpos_sources import load_actual_qpos_sources
from xhand_grasp.grasp_pose import canonical_sha256


CONFIG_PATH = (
    "grasp_configs/"
    "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift.json"
)
V10_CONFIG_PATH = (
    "grasp_configs/"
    "left_opposed_face_palm_down_larger_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)
THUMB = "left_hand_thumb_bend_joint_actuator"


@pytest.fixture(scope="module")
def v11_config():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def source_records(v11_config):
    return tuple(source.generator_record() for source in load_actual_qpos_sources(v11_config))


def test_v11_builds_all_700_edge_thumb_orbit_strata(v11_config):
    cells = actual_contact_search_cells(v11_config)

    assert len(cells) == 20 * 5 * 7 == 700
    assert cells[0].as_dict() == {
        "stratum_index": 0,
        "stratum_id": "edge_85mm_thumb_actual_1.40rad_clockwise_0deg",
        "cell_index": 0,
        "cell_id": "edge_85mm_thumb_actual_1.40rad_clockwise_0deg",
        "edge_m": 0.085,
        "thumb_actual_center_rad": 1.4,
        "clockwise_orbit_deg": 0.0,
    }
    assert cells[-1].edge_m == pytest.approx(0.104)
    assert cells[-1].thumb_actual_center_rad == pytest.approx(1.60)
    assert cells[-1].clockwise_orbit_deg == pytest.approx(15.0)

    # The legacy partition and therefore every v10 candidate ID stays intact.
    assert len(actual_contact_search_cells(load_config(V10_CONFIG_PATH))) == 19 * 5


def test_v11_generator_realizes_70_30_sources_and_coupled_orbit(
    v11_config, source_records
):
    cell = actual_contact_search_cells(v11_config)[2]  # 85 mm, 1.40 rad, +5 deg.
    records = generate_actual_contact_pose_candidates(
        v11_config, source_records, cell, count=10
    )

    assert [record["config"]["candidate_metadata"]["source_index"] for record in records] == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        2,
        3,
    ]
    cube_poses = []
    for record in records:
        config = record["config"]
        validate_config(config)
        relative = config["candidate_metadata"]["relative_wrist_pose_search"]
        assert relative["clockwise_orbit_deg"] == pytest.approx(5.0)
        assert relative["cube_pose_sampled"] is False
        assert relative["hand_root_fixed_during_simulation"] is True
        assert set(relative["anchor_hand_pose"]) == {"translation_m", "rpy_deg"}
        assert config["grasp_pose"]["nominal_joint_qpos_rad"][THUMB] == pytest.approx(
            1.40
        )
        assert config["control"]["manipulation_delta_rad"] == {
            name: 0.0 for name in ACTIVE_ACTUATORS
        }
        cube_poses.append(
            (
                config["cube"]["edge_m"],
                tuple(config["cube"]["center_xy_m"]),
                tuple(config["cube"]["rpy_deg"]),
                config["cube"]["z_offset_m"],
            )
        )
    assert len(set(cube_poses)) == 1


def test_relative_local_parent_selection_reserves_two_per_edge(monkeypatch, v11_config):
    import xhand_grasp.tuning.actual_contact_grasp_pose_dynamic as dynamic

    records = []
    for edge_rank, edge in enumerate((0.085, 0.086, 0.087)):
        for local_rank in range(5):
            config = copy.deepcopy(v11_config)
            config["cube"]["edge_m"] = edge
            config["candidate_metadata"] = {"candidate_id": edge_rank * 10 + local_rank}
            records.append(
                {
                    "candidate_id": edge_rank * 10 + local_rank,
                    "config": config,
                    "rank": (local_rank, edge_rank),
                }
            )

    monkeypatch.setattr(
        dynamic,
        "rank_dynamic_grasp_results",
        lambda values: tuple(sorted(values, key=lambda value: value["rank"])),
    )
    selected = select_dynamic_local_refinement_parents(records, top_count=40)

    assert len(selected) == 6
    assert {
        edge: [value["candidate_id"] for value in selected if value["config"]["cube"]["edge_m"] == edge]
        for edge in (0.085, 0.086, 0.087)
    } == {0.085: [0, 1], 0.086: [10, 11], 0.087: [20, 21]}


def test_static_stratum_spawn_workers_are_deterministic(
    monkeypatch, tmp_path: Path, v11_config
):
    import xhand_grasp.tuning.actual_contact_grasp_pose as campaign

    sources = load_actual_qpos_sources(v11_config)
    cells = campaign.actual_contact_search_cells(v11_config)[:2]
    monkeypatch.setattr(campaign, "actual_contact_search_cells", lambda config: cells)
    common = {
        "stage": "worker_check",
        "campaign_input_sha256": canonical_sha256(v11_config),
        "seed": 20260821,
        "start_index": 0,
        "sample_count_per_cell": 10,
        "selected_retain_per_cell": 2,
        "pool_retain_per_cell": 3,
    }

    serial = campaign._run_static_stage(
        v11_config, sources, tmp_path / "serial", workers=1, **common
    )
    spawned = campaign._run_static_stage(
        v11_config, sources, tmp_path / "spawned", workers=2, **common
    )

    assert serial.summary == spawned.summary
    assert canonical_sha256(serial.records) == canonical_sha256(spawned.records)
