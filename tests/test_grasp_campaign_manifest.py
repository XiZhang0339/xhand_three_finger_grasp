from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from xhand_grasp.grasp_campaign_manifest import load_grasp_campaign_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "grasp_configs"
    / "left_far_hand_high_thumb_grasp_acquisition_manifest.json"
)


def test_manifest_materializes_canonical_best_and_density_scaled_sizes():
    candidates, best_id, metadata = load_grasp_campaign_manifest(MANIFEST)

    assert best_id == 117060
    assert metadata["candidate_count"] == 6
    assert [item["candidate_id"] for item in candidates] == [
        117060,
        116060,
        115060,
        117061,
        116062,
        110064,
    ]
    best = candidates[0]["config"]
    assert best.get("run_context") is None
    assert best["cube"]["edge_m"] == pytest.approx(0.060)
    assert best["cube"]["mass_kg"] == pytest.approx(0.160)
    assert best["control"]["grasp_targets_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] == pytest.approx(1.17)
    assert set(best["control"]["manipulation_delta_rad"].values()) == {0.0}

    edge_61 = candidates[3]["config"]
    assert edge_61["run_context"] == {"kind": "parameter_override_run"}
    assert edge_61["cube"]["mass_kg"] == pytest.approx(
        0.160 * (0.061 / 0.060) ** 3
    )
    assert edge_61["candidate_metadata"]["manifest_label"] == (
        "thumb_1p17_edge_61mm_distance_158mm"
    )
    assert edge_61["control"]["grasp_targets_rad"][
        "left_hand_thumb_rota_joint1_actuator"
    ] == pytest.approx(0.35)


def test_manifest_rejects_distance_that_disagrees_with_relative_pose(tmp_path):
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["base_config"] = str(
        MANIFEST.parent
        / "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift.json"
    )
    payload["candidates"][0]["root_cube_distance_mm"] = 150.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="disagrees with cube_in_root_mm"):
        load_grasp_campaign_manifest(bad)


def test_manifest_rejects_duplicate_labels(tmp_path):
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["base_config"] = str(
        MANIFEST.parent
        / "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift.json"
    )
    duplicate = copy.deepcopy(payload["candidates"][0])
    duplicate["candidate_id"] = 999999
    payload["candidates"].append(duplicate)
    bad = tmp_path / "duplicate.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="labels must be unique"):
        load_grasp_campaign_manifest(bad)
