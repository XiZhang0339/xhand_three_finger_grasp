from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import xhand_grasp.aligned_trajectory_catalog as catalog_module
from xhand_grasp.aligned_trajectory_catalog import (
    export_aligned_trajectory_catalog,
)
from xhand_grasp.config import load_config
from xhand_grasp.viewer import resolve_viewer_source


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift.json"
)
BANDS = (10.0, 12.5, 15.0, 17.5, 20.0)


def _summary(passed: bool) -> dict:
    return {
        "passed": passed,
        "failed_checks": [] if passed else ["synthetic_near_miss"],
        "stage_status": {
            "grasp_success": passed,
            "manipulation_success": passed,
            "full_success": passed,
        },
        "metrics": {
            "grasp_acquisition_step": 0 if passed else -1,
            "manipulation_start_step": 1 if passed else -1,
            "manipulation_end_step": 2 if passed else -1,
            "termination_step": 3,
            "contact_alignment": {
                "operation": {
                    "aligned_duty": 0.8 if passed else 0.2,
                    "height_spread_p95_m": 0.003 if passed else 0.008,
                }
            },
        },
    }


def _candidate(identifier: int, band: float, passed: bool) -> dict:
    config = load_config(CONFIG_PATH)
    config["search_metadata"] = {"tilt_band_center_deg": band}
    return {
        "candidate_id": identifier,
        "tilt_band_center_deg": band,
        "config": config,
        "summary": _summary(passed),
    }


def test_catalog_rejects_override_candidates_before_publication(tmp_path):
    candidates = [
        _candidate(index, band, False) for index, band in enumerate(BANDS)
    ]
    candidates[0]["config"]["run_context"] = {
        "kind": "parameter_override_run"
    }

    with pytest.raises(ValueError, match="require canonical candidates"):
        export_aligned_trajectory_catalog(
            candidates, tmp_path / "override-catalog", video=False
        )


def _write_trace(path: Path, summary: dict, *, mismatched_event: bool = False) -> None:
    total = 4
    passed = bool(summary["passed"])
    events = {
        "grasp_acquisition_step": 0 if passed else -1,
        "manipulation_start_step": 1 if passed else -1,
        "manipulation_end_step": 2 if passed else -1,
        "termination_step": 3,
    }
    if mismatched_event:
        events["termination_step"] = 2
    np.savez_compressed(
        path,
        time=np.arange(1, total + 1, dtype=np.float64) * 0.001,
        cube_pos=np.zeros((total, 3), dtype=np.float64),
        cube_quat=np.tile([1.0, 0.0, 0.0, 0.0], (total, 1)),
        cube_velocity=np.zeros((total, 6), dtype=np.float64),
        ctrl=np.zeros((total, 12), dtype=np.float64),
        joint_qpos=np.zeros((total, 12), dtype=np.float64),
        joint_qvel=np.zeros((total, 12), dtype=np.float64),
        finger_order=np.asarray(["thumb", "index", "mid"]),
        grasp_gate_order=np.asarray(
            ["finite", "contact_height_aligned"], dtype=np.str_
        ),
        finger_down_tilt_deg=np.full(total, 15.0),
        distal_face_position_moment_n_m=np.zeros((total, 3, 8, 3)),
        target_face_contact_centroid_world_m=np.zeros((total, 3, 3)),
        target_face_contact_centroid_valid=np.zeros((total, 3), dtype=bool),
        three_contact_height_spread_m=np.zeros(total),
        three_contact_height_aligned=np.zeros(total, dtype=bool),
        **{name: np.asarray(value, dtype=np.int64) for name, value in events.items()},
    )


def _fake_run_metadata(path: Path, *, numpy_version: str = "2.2.6") -> dict:
    return {
        "python": "3.10.12",
        "python_executable": "/workspace/.venv/bin/python",
        "mujoco": "3.10.0",
        "numpy": numpy_version,
        "uv": "uv 0.12.5",
        "ffmpeg": "ffmpeg version synthetic",
        "platform": "synthetic-linux",
        "repo_commit": "0" * 40,
        "repo_branch": "feat/test",
        "repo_dirty": True,
        "repo_status_short": ["?? synthetic"],
        "experiment_sha256": "1" * 64,
        "implementation_sha256": "2" * 64,
        "implementation_files": ["xhand_grasp/simulation.py"],
        "model_sha256": "3" * 64,
        "pyproject_sha256": "4" * 64,
        "verify_xhand_sha256": "5" * 64,
        "uv_lock_sha256": "6" * 64,
        "config_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "mujoco_gl": "osmesa",
    }


def _mock_runner(monkeypatch, reported_pass_by_band, *, mismatch_event=False):
    calls = []

    def run(config, *, trace_path, video_path):
        assert video_path is None
        band = float(config["search_metadata"]["tilt_band_center_deg"])
        calls.append((band, copy.deepcopy(config)))
        summary = _summary(bool(reported_pass_by_band[band]))
        _write_trace(Path(trace_path), summary, mismatched_event=mismatch_event)
        return summary

    monkeypatch.setattr(catalog_module, "run_simulation", run)
    monkeypatch.setattr(catalog_module, "run_metadata", _fake_run_metadata)
    return calls


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mixed_band_catalog_is_viewer_compatible_and_hash_complete(
    tmp_path, monkeypatch
):
    pass_by_band = {10.0: True, 12.5: False, 15.0: True, 17.5: False, 20.0: False}
    candidates = [
        _candidate(index, band, pass_by_band[band])
        for index, band in enumerate(BANDS)
    ]
    calls = _mock_runner(monkeypatch, pass_by_band)
    output = tmp_path / "aligned-catalog"

    result = export_aligned_trajectory_catalog(candidates, output, video=False)

    assert [band for band, _ in calls] == list(BANDS)
    assert result["trajectory_count"] == 5
    assert result["passing_trajectory_count"] == 2
    assert result["near_miss_trajectory_count"] == 3
    assert result["passing_tilt_bands_deg"] == [10.0, 15.0]
    assert result["missing_passing_tilt_bands_deg"] == [12.5, 17.5, 20.0]
    assert result["all_selected_passes_reproduced"] is True
    assert result["all_reruns_full_success"] is False
    assert result["all_declared_bands_passed"] is False
    assert [entry["label"] for entry in result["trajectories"]] == [
        "tilt_10",
        "near_miss_tilt_12p5",
        "tilt_15",
        "near_miss_tilt_17p5",
        "near_miss_tilt_20",
    ]
    provenance = result["provenance"]
    assert provenance["all_required_shared_fields_consistent"] is True
    assert provenance["shared_run_metadata"] == {
        "python": "3.10.12",
        "mujoco": "3.10.0",
        "numpy": "2.2.6",
        "uv": "uv 0.12.5",
        "ffmpeg": "ffmpeg version synthetic",
        "model_sha256": "3" * 64,
        "uv_lock_sha256": "6" * 64,
        "pyproject_sha256": "4" * 64,
        "implementation_sha256": "2" * 64,
    }

    persisted = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    assert persisted == result
    for entry in result["trajectories"]:
        artifacts = entry["artifacts"]
        for name in ("resolved_config", "trace", "result"):
            path = output / artifacts[name]
            assert path.is_file()
            assert artifacts["sha256"][name] == _sha256(path)
        assert artifacts["video"] is None
        local_result = json.loads((output / artifacts["result"]).read_text())
        assert local_result["trace_summary_consistent"] is True
        metadata = local_result["metadata"]
        assert {
            "python",
            "mujoco",
            "numpy",
            "uv",
            "ffmpeg",
            "model_sha256",
            "uv_lock_sha256",
            "pyproject_sha256",
            "implementation_sha256",
            "config_sha256",
        } <= metadata.keys()
        assert metadata["config_sha256"] == artifacts["sha256"][
            "resolved_config"
        ]
        assert provenance["resolved_config_sha256_by_trajectory"][
            entry["trajectory_id"]
        ] == metadata["config_sha256"]
        assert "result" not in local_result["artifacts"]["sha256"]
        assert local_result["target_faces"] == {
            "thumb": "-X",
            "index": "+X",
            "mid": "+X",
        }
        assert local_result["physical_parameters"]["edge_m"] == pytest.approx(
            0.062
        )

    resolved = resolve_viewer_source(
        catalog_path=output / "catalog.json",
        trajectory="near_miss_tilt_12p5",
    )
    assert resolved.trajectory == "near_miss_tilt_12p5"
    assert resolved.config_path == (
        output / "near_miss_tilt_12p5" / "resolved_config.json"
    ).resolve()


def test_no_passes_still_publishes_all_declared_near_misses(tmp_path, monkeypatch):
    pass_by_band = {band: False for band in BANDS}
    candidates = [
        _candidate(index, band, False) for index, band in enumerate(BANDS)
    ]
    _mock_runner(monkeypatch, pass_by_band)

    result = export_aligned_trajectory_catalog(
        candidates, tmp_path / "all-near", video=False
    )

    assert result["passing_trajectory_count"] == 0
    assert result["near_miss_trajectory_count"] == 5
    assert result["campaign_has_passing_trajectory"] is False
    assert result["published_tilt_bands_deg"] == list(BANDS)
    assert all(
        entry["label"].startswith("near_miss_tilt_")
        for entry in result["trajectories"]
    )


def test_missing_or_duplicate_band_cannot_be_substituted(tmp_path, monkeypatch):
    candidates = [_candidate(index, band, False) for index, band in enumerate(BANDS)]
    monkeypatch.setattr(
        catalog_module,
        "run_simulation",
        lambda *args, **kwargs: pytest.fail("validation must precede simulation"),
    )

    with pytest.raises(ValueError, match="one honest result for every"):
        export_aligned_trajectory_catalog(
            candidates[:-1], tmp_path / "missing", video=False
        )

    duplicate = candidates + [_candidate(99, 10.0, False)]
    with pytest.raises(ValueError, match="more than one selected"):
        export_aligned_trajectory_catalog(
            duplicate, tmp_path / "duplicate", video=False
        )

    undeclared = copy.deepcopy(candidates)
    undeclared[-1]["tilt_band_center_deg"] = 19.0
    undeclared[-1]["config"]["search_metadata"]["tilt_band_center_deg"] = 19.0
    with pytest.raises(ValueError, match="not one declared"):
        export_aligned_trajectory_catalog(
            undeclared, tmp_path / "undeclared", video=False
        )


def test_reported_hard_pass_must_reproduce_and_failure_is_atomic(
    tmp_path, monkeypatch
):
    candidates = [
        _candidate(index, band, band == 10.0)
        for index, band in enumerate(BANDS)
    ]
    rerun = {band: False for band in BANDS}
    _mock_runner(monkeypatch, rerun)
    destination = tmp_path / "atomic"

    with pytest.raises(RuntimeError, match="did not reproduce full success"):
        export_aligned_trajectory_catalog(candidates, destination, video=False)

    assert not destination.exists()
    assert not list(tmp_path.glob(".atomic.staging.*"))


def test_summary_trace_event_mismatch_refuses_publish(tmp_path, monkeypatch):
    candidates = [_candidate(index, band, False) for index, band in enumerate(BANDS)]
    rerun = {band: False for band in BANDS}
    _mock_runner(monkeypatch, rerun, mismatch_event=True)
    destination = tmp_path / "inconsistent"

    with pytest.raises(ValueError, match="disagrees with trace event"):
        export_aligned_trajectory_catalog(candidates, destination, video=False)

    assert not destination.exists()


def test_near_misses_can_be_excluded_and_existing_output_is_never_touched(
    tmp_path, monkeypatch
):
    candidates = [
        _candidate(index, band, band == 15.0)
        for index, band in enumerate(BANDS)
    ]
    pass_by_band = {band: band == 15.0 for band in BANDS}
    calls = _mock_runner(monkeypatch, pass_by_band)

    result = export_aligned_trajectory_catalog(
        candidates,
        tmp_path / "passes-only",
        include_near_misses=False,
        video=False,
    )
    assert [band for band, _ in calls] == [15.0]
    assert result["trajectory_count"] == 1
    assert result["trajectories"][0]["label"] == "tilt_15"

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        export_aligned_trajectory_catalog(
            candidates,
            existing,
            include_near_misses=False,
            video=False,
        )
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_inconsistent_required_provenance_aborts_atomically(tmp_path, monkeypatch):
    passing = {band: band in (10.0, 12.5) for band in BANDS}
    candidates = [
        _candidate(index, band, passing[band])
        for index, band in enumerate(BANDS)
    ]
    _mock_runner(monkeypatch, passing)
    metadata_calls = 0

    def changing_metadata(path):
        nonlocal metadata_calls
        metadata_calls += 1
        version = "2.2.6" if metadata_calls == 1 else "2.3.0"
        return _fake_run_metadata(path, numpy_version=version)

    monkeypatch.setattr(catalog_module, "run_metadata", changing_metadata)
    destination = tmp_path / "inconsistent-provenance"

    with pytest.raises(RuntimeError, match="run provenance changed.*numpy"):
        export_aligned_trajectory_catalog(
            candidates,
            destination,
            include_near_misses=False,
            video=False,
        )

    assert metadata_calls == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".inconsistent-provenance.staging.*"))
