from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import xhand_grasp.grasp_trajectory_catalog as catalog_module
from xhand_grasp.config import load_config
from xhand_grasp.contacts import Face
from xhand_grasp.grasp_trajectory_catalog import (
    GLOBAL_INTEGRITY_CHECKS,
    export_grasp_trajectory_catalog,
)
from xhand_grasp.scene import build_model
from xhand_grasp.simulation import _allocate_traces


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift.json"
)


def _passing_summary(acquisition_step: int) -> dict:
    checks = {name: True for name in GLOBAL_INTEGRITY_CHECKS}
    return {
        "passed": False,
        "failed_checks": [
            "operation_median_lift_reached",
            "operation_minimum_lift_reached",
        ],
        "checks": checks,
        "stage_status": {
            "grasp_success": True,
            "grasp": "acquired",
            "manipulation_success": False,
            "manipulation": "failed",
            "full_success": False,
        },
        "metrics": {
            "grasp_acquisition_step": acquisition_step,
            "manipulation_start_step": acquisition_step + 1,
            "manipulation_end_step": acquisition_step + 49,
            "termination_step": acquisition_step + 50,
        },
        "phase_steps": {
            "settle": 500,
            "close": 1000,
            "verify": 750,
            "manipulate": 1500,
            "hold": 1000,
        },
    }


def _passing_trace(config: dict, *, poor_post_hold: bool = False) -> dict[str, np.ndarray]:
    model, _ = build_model(config)
    total = 1300
    acquisition = 249
    traces = _allocate_traces(model, total, schema_version=5)
    traces["time"][:] = np.arange(1, total + 1, dtype=np.float64) * 0.001
    traces["cube_pos"][:] = [0.0, 0.0, 0.1]
    traces["cube_quat"][:] = [1.0, 0.0, 0.0, 0.0]
    traces["cube_velocity"][:] = 0.0
    traces["root_pos"][:] = [0.0, 0.0, 0.0]
    traces["root_quat"][:] = [1.0, 0.0, 0.0, 0.0]
    traces["ctrl"][:] = 0.0
    traces["joint_qpos"][:] = 0.0
    traces["joint_qvel"][:] = 0.0
    traces["actuator_force"][:] = 0.0
    traces["finger_contact_force"][:] = 0.1
    traces["tactile_max"][:] = 0.1
    traces["forbidden_contact"][:] = False
    traces["support_contact"][:] = True
    traces["floor_contact"][:] = False
    traces["max_penetration"][:] = 0.0002
    traces["friction_error"][:] = 0.0
    traces["cube_contact_seen"][:] = True
    traces["contact_dim_ok"][:] = True
    traces["finite"][:] = True
    traces["palm_down_angle_deg"][:] = 15.0
    traces["distal_face_force_n"][:] = 0.0
    traces["distal_face_force_n"][:, 0, Face.X_NEG] = 0.1
    traces["distal_face_force_n"][:, 1, Face.X_POS] = 0.1
    traces["distal_face_force_n"][:, 2, Face.X_POS] = 0.1
    traces["active_nondistal_force_n"][:] = 0.0
    traces["target_face_force_purity"][:] = 1.0
    traces["target_face_topology"][:] = True
    traces["control_state"][: acquisition + 1] = "VERIFY"
    traces["control_state"][acquisition + 1 :] = "MANIPULATE"
    traces["grasp_gate"][:] = True
    traces["grasp_gate_consecutive_steps"][:] = np.minimum(
        np.arange(1, total + 1), 250
    )
    traces["grasp_acquired"][:] = False
    traces["grasp_acquired"][acquisition:] = True
    traces["manipulation_progress"][:] = 0.0
    traces["manipulation_progress"][acquisition + 1 :] = 0.5
    traces["target_face_effective"][:] = True
    traces["finger_down_tilt_deg"][:] = 15.0
    traces["distal_face_position_moment_n_m"][:] = 0.0
    traces["target_face_contact_centroid_world_m"][:] = 0.0
    traces["target_face_contact_centroid_valid"][:] = True
    traces["three_contact_height_spread_m"][:] = 0.003
    traces["three_contact_height_aligned"][:] = True
    traces["root_cube_center_distance_m"][:] = 0.145
    traces["thumb_bend_command_rad"][:] = 1.1
    traces["thumb_bend_qpos_rad"][:] = 1.0
    traces["distal_pad_force_n"][:] = 0.1
    traces["distal_nonpad_force_n"][:] = 0.0
    traces["distal_pad_force_fraction"][:] = 1.0
    traces["distal_active_taxel_count"][:] = 2
    traces["grasp_acquisition_step"] = np.asarray(acquisition, dtype=np.int64)
    traces["manipulation_start_step"] = np.asarray(
        acquisition + 1, dtype=np.int64
    )
    traces["manipulation_end_step"] = np.asarray(
        acquisition + 49, dtype=np.int64
    )
    traces["termination_step"] = np.asarray(acquisition + 50, dtype=np.int64)
    traces["video_frame_steps"] = np.asarray([0, 100, 249, 250], dtype=np.int64)

    if poor_post_hold:
        post = slice(acquisition + 1, acquisition + 1001)
        traces["target_face_effective"][post] = False
        traces["three_contact_height_aligned"][post] = False
        traces["support_contact"][post] = False
        traces["cube_pos"][post, 0] = 0.01
        traces["cube_velocity"][post, 0] = 0.1
    return traces


def _install_runner(
    monkeypatch: pytest.MonkeyPatch,
    traces: dict[str, np.ndarray],
    summary: dict,
) -> list[dict]:
    calls: list[dict] = []

    def run(config, *, trace_path, video_path):
        assert video_path is None
        calls.append(copy.deepcopy(config))
        np.savez_compressed(trace_path, **traces)
        return copy.deepcopy(summary)

    def metadata(path: Path) -> dict:
        return {
            "config_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "python": "synthetic",
            "mujoco": "3.10.0",
            "numpy": "2.2.6",
        }

    monkeypatch.setattr(catalog_module, "run_simulation", run)
    monkeypatch.setattr(catalog_module, "run_metadata", metadata)
    return calls


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_catalog_reruns_and_publishes_acquisition_prefix_with_honest_scope(
    tmp_path, monkeypatch
):
    config = load_config(CONFIG_PATH)
    traces = _passing_trace(config, poor_post_hold=True)
    summary = _passing_summary(249)
    calls = _install_runner(monkeypatch, traces, summary)
    output = tmp_path / "grasp-catalog"

    result = export_grasp_trajectory_catalog(
        [{"candidate_id": 7, "config": config, "summary": summary}],
        output,
        video=False,
        best_candidate_id=7,
    )

    assert len(calls) == 1
    assert result["validation_scope"] == "grasp_acquisition"
    assert result["validated_grasp_count"] == 1
    assert result["failed_grasp_count"] == 0
    assert result["campaign_has_validated_grasp"] is True
    assert result["all_reruns_full_success"] is False
    assert result["aliases"] == {"best_grasp": "grasp_000007"}
    entry = result["trajectories"][0]
    assert entry["classification"] == "validated_grasp_acquisition"
    assert entry["rerun_grasp_success"] is True
    assert entry["rerun_full_success"] is False
    # The deliberately bad future hold is visible, but is not a grasp gate.
    diagnostic = entry["post_acquisition_hold_diagnostic"]
    assert diagnostic["scope"] == "diagnostic_only_not_an_acceptance_gate"
    assert diagnostic["complete_window"] is True
    assert diagnostic["target_face_topology_duty"] == 0.0
    assert diagnostic["contact_height_aligned_duty"] == 0.0
    assert diagnostic["support_contact_duty"] == 0.0
    assert diagnostic["max_translation_from_acquisition_m"] == pytest.approx(0.01)

    artifacts = entry["artifacts"]
    for name in ("resolved_config", "result", "trace", "grasp_trace"):
        path = output / artifacts[name]
        assert path.is_file()
        assert artifacts["sha256"][name] == _sha256(path)
    assert artifacts["video"] is None
    with np.load(output / artifacts["trace"], allow_pickle=False) as archive:
        assert archive["time"].shape == (1300,)
    with np.load(output / artifacts["grasp_trace"], allow_pickle=False) as archive:
        assert int(archive["grasp_trace_schema_version"]) == 1
        assert str(archive["validation_scope"]) == "grasp_acquisition"
        assert archive["time"].shape == (250,)
        assert int(archive["grasp_acquisition_step"]) == 249
        assert int(archive["segment_stop_step_inclusive"]) == 249
        assert int(archive["source_total_steps"]) == 1300
        assert archive["control_state"][-1] == "VERIFY"
        assert not np.any(np.isin(archive["control_state"], ("MANIPULATE", "HOLD")))
        assert np.all(archive["manipulation_progress"] == 0.0)
        assert tuple(archive["finger_order"].tolist()) == ("thumb", "index", "mid")
        assert np.array_equal(archive["video_frame_steps"], [0, 100, 249])
        assert "manipulation_start_step" not in archive.files

    persisted = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    assert persisted == result


def test_global_integrity_failure_is_not_published_as_grasp_success(
    tmp_path, monkeypatch
):
    config = load_config(CONFIG_PATH)
    traces = _passing_trace(config)
    summary = _passing_summary(249)
    summary["checks"]["hand_root_pose_did_not_move"] = False
    _install_runner(monkeypatch, traces, summary)

    result = export_grasp_trajectory_catalog(
        [{"candidate_id": 3, "config": config}],
        tmp_path / "failed-catalog",
        video=False,
    )

    assert result["validated_grasp_count"] == 0
    assert result["failed_grasp_count"] == 1
    assert result["aliases"] == {}
    entry = result["trajectories"][0]
    assert entry["classification"] == "failed_grasp_acquisition_validation"
    assert entry["rerun_grasp_success"] is False
    assert entry["rerun_full_success"] is False
    assert "hand_root_pose_did_not_move" in entry["grasp_validation"][
        "failed_checks"
    ]


def test_reported_grasp_that_does_not_reproduce_aborts_atomically(
    tmp_path, monkeypatch
):
    config = load_config(CONFIG_PATH)
    traces = _passing_trace(config)
    reported = _passing_summary(249)
    rerun = _passing_summary(249)
    rerun["stage_status"]["grasp_success"] = False
    _install_runner(monkeypatch, traces, rerun)
    output = tmp_path / "must-not-exist"

    with pytest.raises(RuntimeError, match="did not reproduce"):
        export_grasp_trajectory_catalog(
            [{"candidate_id": 5, "config": config, "summary": reported}],
            output,
            video=False,
            best_candidate_id=5,
        )

    assert not output.exists()


def test_failed_candidate_cannot_receive_best_grasp_alias(tmp_path, monkeypatch):
    config = load_config(CONFIG_PATH)
    traces = _passing_trace(config)
    summary = _passing_summary(249)
    summary["checks"]["runtime_contact_friction_matches_cube"] = False
    _install_runner(monkeypatch, traces, summary)
    output = tmp_path / "failed-best"

    with pytest.raises(RuntimeError, match="cannot alias a failed grasp"):
        export_grasp_trajectory_catalog(
            [{"candidate_id": 9, "config": config}],
            output,
            video=False,
            best_candidate_id=9,
        )

    assert not output.exists()


def test_parameter_override_is_independently_rerun_and_explicitly_labeled(
    tmp_path, monkeypatch
):
    config = load_config(CONFIG_PATH)
    config["run_context"] = {"kind": "parameter_override_run"}
    traces = _passing_trace(config)
    summary = _passing_summary(249)
    _install_runner(monkeypatch, traces, summary)

    result = export_grasp_trajectory_catalog(
        [
            {
                "candidate_id": 61,
                "label": "edge_61mm",
                "config": config,
            }
        ],
        tmp_path / "override-catalog",
        video=False,
        best_candidate_id=61,
    )

    assert result["parameter_override_trajectory_count"] == 1
    entry = result["trajectories"][0]
    assert entry["label"] == "edge_61mm"
    assert entry["parameter_override_run"] is True
    assert entry["rerun_grasp_success"] is True
