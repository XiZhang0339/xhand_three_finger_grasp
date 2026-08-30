from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
    bind_candidate_result_semantic_sha256,
    export_actual_contact_manipulation_catalog,
)
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift.json"
)


def _summary() -> dict:
    return {
        "passed": True,
        "failed_checks": [],
        "checks": {"synthetic": True},
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": True,
            "full_success": True,
        },
    }


def _production_candidate(root: Path) -> dict:
    directory = root / "candidate"
    directory.mkdir()
    config = load_config(TEMPLATE)
    config_path = directory / "resolved_config.json"
    trace_path = directory / "trace.npz"
    result_path = directory / "result.json"
    write_json(config_path, config)
    np.savez_compressed(
        trace_path,
        time=np.asarray([0.001, 0.002], dtype=np.float64),
        cube_pos=np.asarray([[0.071, -0.027, 0.04]] * 2, dtype=np.float64),
        video_frame_steps=np.asarray([], dtype=np.int64),
    )
    result = {
        "actual_contact_manipulation_candidate_schema_version": 1,
        "complete": True,
        "candidate_id": 19,
        "summary": _summary(),
        "artifacts": {
            "resolved_config": config_path.name,
            "trace": trace_path.name,
            "sha256": {
                "resolved_config": file_sha256(config_path),
                "trace": file_sha256(trace_path),
            },
        },
    }
    result = bind_candidate_result_semantic_sha256(result)
    write_json(result_path, result)
    return {
        "candidate_id": 19,
        "discovery_index": 0,
        "config_path": config_path,
        "result_path": result_path,
        "trace_path": trace_path,
    }


def _video_evidence() -> dict:
    return {
        "decode_verified": True,
        "codec": "h264",
        "width": 640,
        "height": 480,
        "fps": "30/1",
        "frame_count": 1,
        "duration_s": 1.0 / 30.0,
        "size_bytes": 9,
    }


def test_candidate_semantic_digest_normalizes_numpy_scalars_like_artifact_json(
    tmp_path,
):
    payload = bind_candidate_result_semantic_sha256(
        {
            "candidate_result_schema_version": np.int64(1),
            "complete": np.bool_(True),
            "summary": {
                "passed": np.bool_(False),
                "metric": np.float64(0.125),
            },
        }
    )
    path = tmp_path / "result.json"
    write_json(path, payload)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    authenticate_candidate_result_semantic_sha256(persisted, source=path)


def test_v9_final_catalog_reruns_renders_probes_and_binds_same_trace(tmp_path):
    candidate = _production_candidate(tmp_path)
    source_trace = Path(candidate["trace_path"])
    calls: list[str] = []

    def runner(config, *, trace_path, video_path):
        assert config["schema_version"] == 9
        calls.append("simulation")
        with np.load(source_trace, allow_pickle=False) as trace:
            np.savez_compressed(
                trace_path,
                time=np.asarray(trace["time"]),
                cube_pos=np.asarray(trace["cube_pos"]),
                video_frame_steps=np.asarray([1], dtype=np.int64),
            )
        video_path.write_bytes(b"synthetic")
        return {
            **copy.deepcopy(_summary()),
            "video": {
                **_video_evidence(),
                "simulation_step_indices": [1],
            },
        }

    def video_probe(path, expected_frames, settings):
        calls.append("ffprobe_and_decode")
        assert path.read_bytes() == b"synthetic"
        assert expected_frames == 1
        assert (settings.width, settings.height, settings.fps) == (640, 480, 30)
        return _video_evidence()

    output = tmp_path / "catalog"
    catalog = export_actual_contact_manipulation_catalog(
        [candidate],
        output,
        selected_count=1,
        simulation_runner=runner,
        video_probe=video_probe,
    )

    assert calls == ["simulation", "ffprobe_and_decode"]
    entry = catalog["trajectories"][0]
    assert entry["classification"] == "success"
    assert entry["final_video_required"] is True
    assert entry["final_video_verified"] is True
    assert entry["final_video_evidence"]["trace_reproduction"][
        "physical_fields_exact"
    ] is True
    video_path = output / entry["artifacts"]["video"]
    assert video_path.is_file()
    assert entry["artifacts"]["sha256"]["video"] == file_sha256(video_path)
    result = json.loads(
        (output / entry["artifacts"]["result"]).read_text(encoding="utf-8")
    )
    assert result["final_video_publication"]["ffprobe_and_full_decode"][
        "decode_verified"
    ] is True
    assert result["artifacts"]["sha256"]["video"] == file_sha256(video_path)
    with np.load(output / entry["artifacts"]["trace"], allow_pickle=False) as trace:
        assert trace["video_frame_steps"].tolist() == [1]


def test_v9_final_catalog_is_transactional_when_render_changes_physics(tmp_path):
    candidate = _production_candidate(tmp_path)
    source_trace = Path(candidate["trace_path"])

    def bad_runner(_config, *, trace_path, video_path):
        with np.load(source_trace, allow_pickle=False) as trace:
            changed = np.asarray(trace["cube_pos"]).copy()
            changed[0, 0] += 1e-9
            np.savez_compressed(
                trace_path,
                time=np.asarray(trace["time"]),
                cube_pos=changed,
                video_frame_steps=np.asarray([1], dtype=np.int64),
            )
        video_path.write_bytes(b"synthetic")
        return {
            **copy.deepcopy(_summary()),
            "video": {
                **_video_evidence(),
                "simulation_step_indices": [1],
            },
        }

    output = tmp_path / "catalog"
    with pytest.raises(RuntimeError, match="changed physical trajectory"):
        export_actual_contact_manipulation_catalog(
            [candidate],
            output,
            selected_count=1,
            simulation_runner=bad_runner,
            video_probe=lambda *_args: _video_evidence(),
        )
    assert not output.exists()


def test_v9_final_catalog_rejects_tampered_candidate_summary(tmp_path):
    candidate = _production_candidate(tmp_path)
    result_path = Path(candidate["result_path"])
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["summary"]["passed"] = False
    write_json(result_path, payload)

    with pytest.raises(RuntimeError, match="semantic SHA-256 mismatch"):
        export_actual_contact_manipulation_catalog(
            [candidate],
            tmp_path / "catalog",
            selected_count=1,
        )
