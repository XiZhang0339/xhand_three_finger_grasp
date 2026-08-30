from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
    authenticated_catalog_artifact_paths,
)
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config
from xhand_grasp.scene import build_model
from xhand_grasp.tuning.contact_preserving_planned_lift_campaign import (
    _publish_robust_alias_catalog,
    publish_viewer_catalogs,
)
from xhand_grasp.viewer import (
    _append_v14_plan_force_risk_markers,
    format_v14_contact_overlay_telemetry,
    v14_contact_overlay_state,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _v14_overlay_trace(config: dict) -> dict[str, np.ndarray]:
    target = np.asarray([0.20, 0.30, 0.40])
    actual = np.asarray([0.05, 0.31, 0.42])
    centroids = np.asarray(
        [[0.025, -0.020, 0.140], [0.115, -0.030, 0.140], [0.115, -0.010, 0.140]]
    )
    return {
        "cube_pos": np.asarray([[0.071, -0.027, 0.140]] * 3),
        "control_state": np.asarray(["VERIFY", "MANIPULATE", "MANIPULATE"]),
        "manipulation_start_step": np.asarray(1, dtype=np.int64),
        "manipulation_plan_desired_cube_position_delta_m": np.asarray(
            config["manipulation_plan"]["desired_cube_position_delta_m"],
            dtype=np.float64,
        ),
        "target_face_contact_centroid_world_m": np.tile(
            centroids[None, :, :], (3, 1, 1)
        ),
        "target_face_contact_centroid_valid": np.ones((3, 3), dtype=bool),
        "contact_force_target_n": np.tile(target[None, :], (3, 1)),
        "contact_force_filtered_n": np.tile(actual[None, :], (3, 1)),
        "target_face_effective": np.tile(
            np.asarray([False, True, True])[None, :], (3, 1)
        ),
        "target_face_force_purity": np.ones((3, 3)),
        "contact_loss_run_steps": np.tile(
            np.asarray([2, 0, 0], dtype=np.int64)[None, :], (3, 1)
        ),
        "contact_progress_frozen": np.asarray([False, True, True]),
        "contact_recovery_active": np.asarray([False, True, True]),
        "forbidden_contact": np.zeros(3, dtype=bool),
        "active_nondistal_force_n": np.zeros((3, 3)),
    }


def test_v14_viewer_renders_plan_forces_and_contact_risk() -> None:
    config = load_config(CONFIG)
    model, _ = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    traces = _v14_overlay_trace(config)
    state = v14_contact_overlay_state(traces, 2, config)
    assert state is not None
    assert state["finger_risk"].tolist() == [True, False, False]
    assert state["global_risk"] is True
    assert state["progress_frozen"] is True
    line = format_v14_contact_overlay_telemetry(state)
    assert "thumb=0.050/0.200N!" in line
    assert "frozen=1" in line and "risk=1" in line

    scene = mujoco.MjvScene(model, maxgeom=96)
    handle = SimpleNamespace(user_scn=scene)
    _append_v14_plan_force_risk_markers(
        handle, model, data, traces, 2, config
    )
    labels = [scene.geoms[index].label for index in range(scene.ngeom)]
    assert sum(label.startswith("planned_cube_path_") for label in labels) == 20
    assert sum("_force_" in label for label in labels) == 3
    assert "contact_risk" in labels
    thumb = next(
        scene.geoms[index]
        for index, label in enumerate(labels)
        if label.startswith("thumb_force_")
    )
    np.testing.assert_allclose(thumb.rgba, [1.0, 0.12, 0.05, 1.0])


def _candidate(
    workspace: Path,
    candidate_id: int,
    *,
    force_margin: float = 0.1,
    plan_rank: int = 0,
) -> dict:
    root = workspace / "candidates" / f"candidate_{candidate_id}"
    root.mkdir(parents=True)
    config = load_config(CONFIG)
    config_path = root / "resolved_config.json"
    trace_path = root / "trace.npz"
    result_path = root / "result.json"
    write_json(config_path, config)
    np.savez_compressed(
        trace_path,
        time=np.asarray([0.001, 0.002]),
        cube_pos=np.zeros((2, 3), dtype=np.float64),
        video_frame_steps=np.asarray([], dtype=np.int64),
    )
    summary = {
        "passed": True,
        "failed_checks": [],
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": True,
            "full_success": True,
        },
    }
    result = {
        "contact_preserving_candidate_schema_version": 1,
        "complete": True,
        "candidate_id": candidate_id,
        "experiment_id": config["experiment_id"],
        "classification": "success",
        "grasp_success": True,
        "full_success": True,
        "summary": copy.deepcopy(summary),
        "artifacts": {
            "resolved_config": config_path.name,
            "trace": trace_path.name,
            "sha256": {
                "resolved_config": file_sha256(config_path),
                "trace": file_sha256(trace_path),
            },
        },
    }
    write_json(result_path, result)
    return {
        **result,
        "plan_rank": plan_rank,
        "feedback_index": 0,
        "artifact_directory": str(root.relative_to(workspace)),
        "contact_maintenance": {
            "satisfied": True,
            "forbidden_contact": False,
            "active_nondistal_contact": False,
            "contact_loss_count": 0,
            "minimum_valid_duty": 1.0,
            "minimum_force_margin_n": force_margin,
            "maximum_tangent_slip_m": 0.0,
        },
        "path_tracking": {"rms_error": 0.0, "terminal_error": 0.0},
    }


def _video_evidence() -> dict:
    return {
        "decode_verified": True,
        "codec": "h264",
        "width": 640,
        "height": 480,
        "fps": "30/1",
        "frame_count": 1,
    }


def test_v14_catalog_reruns_mp4_binds_trace_and_copies_to_grasp_catalog(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "campaign"
    record = _candidate(workspace, 14001)
    source_trace = workspace / record["artifact_directory"] / "trace.npz"

    def runner(config, *, trace_path, video_path):
        assert config["schema_version"] == 14
        with np.load(source_trace, allow_pickle=False) as trace:
            np.savez_compressed(
                trace_path,
                time=np.asarray(trace["time"]),
                cube_pos=np.asarray(trace["cube_pos"]),
                video_frame_steps=np.asarray([1], dtype=np.int64),
            )
        video_path.write_bytes(b"synthetic-mp4")
        return {
            **copy.deepcopy(record["summary"]),
            "video": {**_video_evidence(), "simulation_step_indices": [1]},
        }

    catalogs = publish_viewer_catalogs(
        [record],
        workspace,
        workspace / "catalogs" / "target_1",
        experiment_id=load_config(CONFIG)["experiment_id"],
        render_videos=True,
        simulation_runner=runner,
        video_probe_runner=lambda *_args: _video_evidence(),
    )
    manipulation_path = workspace / catalogs["manipulation"]
    manipulation = json.loads(manipulation_path.read_text(encoding="utf-8"))
    entry = manipulation["trajectories"][0]
    assert manipulation["aliases"]["best_first"] == entry["trajectory_id"]
    assert manipulation["aliases"]["best_nominal"] == entry["trajectory_id"]
    assert set(entry["aliases"]) == {
        "best_first",
        "best_nominal",
        "pair_rank_01",
    }
    assert entry["final_video_verified"] is True
    video = manipulation_path.parent / entry["artifacts"]["video"]
    assert video.read_bytes() == b"synthetic-mp4"
    assert entry["artifacts"]["sha256"]["video"] == file_sha256(video)
    with np.load(
        manipulation_path.parent / entry["artifacts"]["trace"],
        allow_pickle=False,
    ) as trace:
        assert trace["video_frame_steps"].tolist() == [1]
    result = json.loads(
        (
            manipulation_path.parent / entry["artifacts"]["result"]
        ).read_text(encoding="utf-8")
    )
    authenticate_candidate_result_semantic_sha256(result)
    assert result["final_video_publication"]["trace_reproduction"][
        "physical_fields_exact"
    ] is True
    authenticated_catalog_artifact_paths(manipulation_path)

    grasp_path = workspace / catalogs["grasp_pose"]
    grasp = json.loads(grasp_path.read_text(encoding="utf-8"))
    assert grasp["aliases"]["best_first"] == grasp["aliases"]["best_nominal"]
    grasp_entry = grasp["trajectories"][0]
    assert set(grasp_entry["aliases"]) == {
        "best_first",
        "best_nominal",
        "pair_rank_01",
    }
    grasp_video = grasp_path.parent / grasp_entry["artifacts"]["video"]
    assert grasp_video.is_file()
    assert grasp_video.stat().st_ino == video.stat().st_ino
    authenticated_catalog_artifact_paths(grasp_path)


def test_v14_catalog_keeps_true_first_and_robust_aliases(tmp_path: Path) -> None:
    workspace = tmp_path / "campaign"
    records = [
        _candidate(
            workspace,
            14100 + index,
            force_margin=1.0 - 0.1 * index,
            plan_rank=index + 1,
        )
        for index in range(5)
    ]
    first = _candidate(
        workspace,
        14999,
        force_margin=0.01,
        plan_rank=0,
    )
    records.append(first)
    catalogs = publish_viewer_catalogs(
        records,
        workspace,
        workspace / "catalogs" / "target_5",
        experiment_id=load_config(CONFIG)["experiment_id"],
    )
    catalog_path = workspace / catalogs["manipulation"]
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert len(catalog["trajectories"]) == 5
    assert set(catalog["aliases"]) >= {
        "best_first",
        "best_nominal",
        "pair_rank_01",
        "pair_rank_02",
        "pair_rank_03",
        "pair_rank_04",
        "pair_rank_05",
    }
    by_id = {
        entry["candidate_id"]: entry["trajectory_id"]
        for entry in catalog["trajectories"]
    }
    assert catalog["aliases"]["best_first"] == by_id["14999"]
    assert catalog["aliases"]["best_nominal"] != catalog["aliases"]["best_first"]
    entries_by_id = {
        entry["trajectory_id"]: entry for entry in catalog["trajectories"]
    }
    for alias, trajectory_id in catalog["aliases"].items():
        assert alias in entries_by_id[trajectory_id]["aliases"]

    robust_path = _publish_robust_alias_catalog(
        catalog_path,
        workspace / "catalogs" / "robust_manipulation",
        robust_candidate_id="14999",
    )
    robust = json.loads(robust_path.read_text(encoding="utf-8"))
    assert robust["aliases"]["best_robust"] == robust["aliases"]["best_first"]
    assert robust["aliases"]["best_robust"] != robust["aliases"]["best_nominal"]
    robust_by_id = {
        entry["trajectory_id"]: entry for entry in robust["trajectories"]
    }
    assert "best_robust" in robust_by_id[
        robust["aliases"]["best_robust"]
    ]["aliases"]
    authenticated_catalog_artifact_paths(robust_path)


def test_v14_catalog_rejects_render_that_changes_physics(tmp_path: Path) -> None:
    workspace = tmp_path / "campaign"
    record = _candidate(workspace, 14002)
    source_trace = workspace / record["artifact_directory"] / "trace.npz"

    def runner(_config, *, trace_path, video_path):
        with np.load(source_trace, allow_pickle=False) as trace:
            changed = np.asarray(trace["cube_pos"]).copy()
            changed[0, 0] += 1e-12
            np.savez_compressed(
                trace_path,
                time=np.asarray(trace["time"]),
                cube_pos=changed,
                video_frame_steps=np.asarray([1], dtype=np.int64),
            )
        video_path.write_bytes(b"synthetic-mp4")
        return {
            **copy.deepcopy(record["summary"]),
            "video": {**_video_evidence(), "simulation_step_indices": [1]},
        }

    with pytest.raises(RuntimeError, match="changed physical trajectory fields"):
        publish_viewer_catalogs(
            [record],
            workspace,
            workspace / "catalogs" / "bad",
            experiment_id=load_config(CONFIG)["experiment_id"],
            render_videos=True,
            simulation_runner=runner,
            video_probe_runner=lambda *_args: _video_evidence(),
        )
