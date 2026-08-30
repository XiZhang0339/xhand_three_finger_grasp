from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import xhand_grasp.normal_aligned_smooth_lift_catalog as catalog_module
from xhand_grasp.artifacts import file_sha256, resolved_run_config
from xhand_grasp.config import load_config
from xhand_grasp.normal_aligned_smooth_lift_catalog import (
    CatalogCandidate,
    discover_catalog_candidates,
    export_normal_aligned_smooth_lift_catalog,
    load_catalog_candidate,
    select_diverse_successes,
)
from xhand_grasp.tuning.normal_aligned_smooth_lift import (
    CAMPAIGN_KIND,
    CANDIDATE_RESULT_SCHEMA_VERSION,
    canonical_sha256,
    controller_id,
    lift_candidate_rank,
    pose_id,
)
from xhand_grasp.viewer import resolve_replay_source


def _candidate(
    tmp_path: Path,
    index: int,
    *,
    edge_mm: int,
    thumb: float,
    passed: bool = True,
) -> CatalogCandidate:
    config = load_config(
        "grasp_configs/"
        "left_opposed_face_palm_down_high_thumb_normal_aligned_"
        "smooth_vertical_lift.json"
    )
    config = copy.deepcopy(config)
    config["cube"]["edge_m"] = edge_mm / 1000.0
    config["control"]["grasp_targets_rad"][
        "left_hand_thumb_bend_joint_actuator"
    ] = thumb
    config["candidate_metadata"] = {
        "candidate_id": index,
        "stage": "lift_exact",
        "locked_timestep_s": 0.001,
    }
    result_path = tmp_path / f"source_{index}.json"
    result = {
        "candidate_id": index,
        "summary": {
            "passed": passed,
            "failed_checks": [] if passed else ["operation_median_lift_reached"],
            "stage_status": {
                "grasp_success": True,
                "manipulation_success": passed,
                "full_success": passed,
            },
            "metrics": {
                "operation_median_lift_m": 0.011 if passed else 0.007,
                "motion_smoothness": {
                    "operation_max_lateral_displacement_m": 0.001,
                    "operation_max_orientation_drift_deg": 2.0,
                    "operation_peak_abs_filtered_jerk_m_s3": 1.0,
                },
                "closure_alignment": {"worst_p95_angle_deg": 18.0},
            },
        }
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return CatalogCandidate(
        result_path=result_path,
        config_path=result_path,
        result=result,
        config=config,
        candidate_id=str(index),
        full_success=passed,
        edge_mm=edge_mm,
        thumb_target_rad=thumb,
    )


def _patch_publish_evidence(monkeypatch):
    monkeypatch.setattr(
        catalog_module,
        "_verify_same_run_trace",
        lambda _config, _trace, _summary: {
            "video_frame_steps": np.asarray([], dtype=np.int64)
        },
    )
    monkeypatch.setattr(
        catalog_module,
        "_validated_run_metadata",
        lambda path: {"config_sha256": file_sha256(path), "test": True},
    )


def _authenticated_source(
    directory: Path,
    *,
    candidate_id: int = 89_000_000_000_001,
    stage: str = "lift_exact",
    config_stage: str | None = None,
    config_candidate_id: int | None = None,
    locked_timestep_s: float = 0.001,
) -> Path:
    directory.mkdir(parents=True)
    config = load_config(
        "grasp_configs/"
        "left_opposed_face_palm_down_high_thumb_normal_aligned_"
        "smooth_vertical_lift.json"
    )
    config = copy.deepcopy(config)
    config["candidate_metadata"] = {
        "candidate_id": (
            candidate_id if config_candidate_id is None else config_candidate_id
        ),
        "stage": stage if config_stage is None else config_stage,
        "locked_timestep_s": locked_timestep_s,
    }
    config_path = directory / "resolved_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    trace_path = directory / "trace.npz"
    np.savez_compressed(trace_path, time=np.asarray([0.001]))
    summary = {
        "passed": False,
        "failed_checks": ["operation_median_lift_reached"],
        "checks": {},
        "metrics": {},
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": False,
            "full_success": False,
        },
    }
    result = {
        "candidate_result_schema_version": CANDIDATE_RESULT_SCHEMA_VERSION,
        "complete": True,
        "campaign_kind": CAMPAIGN_KIND,
        "stage": stage,
        "candidate_id": candidate_id,
        "candidate_sha256": canonical_sha256(config),
        "pose_id": pose_id(config),
        "controller_id": controller_id(config),
        "lift_success": False,
        "manipulation_response": {"available": False},
        "summary": summary,
        "artifacts": {
            "resolved_config": config_path.name,
            "trace": trace_path.name,
            "sha256": {
                "resolved_config": file_sha256(config_path),
                "trace": file_sha256(trace_path),
            },
        },
    }
    result_path = directory / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path


def test_diverse_selector_covers_edges_and_thumb_bands(tmp_path):
    candidates = [
        _candidate(tmp_path, 0, edge_mm=60, thumb=1.25),
        _candidate(tmp_path, 1, edge_mm=60, thumb=1.26),
        _candidate(tmp_path, 2, edge_mm=61, thumb=1.35),
        _candidate(tmp_path, 3, edge_mm=62, thumb=1.40),
        _candidate(tmp_path, 4, edge_mm=63, thumb=1.30),
        _candidate(tmp_path, 5, edge_mm=64, thumb=1.45),
    ]

    selected = select_diverse_successes(candidates, count=5)

    assert len(selected) == 5
    assert len({value.edge_mm for value in selected}) >= 3
    assert len({value.thumb_band for value in selected}) >= 2


def test_catalog_near_miss_rank_uses_lift_progress_before_failed_check_count(
    tmp_path,
):
    closer = _candidate(tmp_path, 21, edge_mm=67, thumb=1.30, passed=False)
    almost_zero = _candidate(tmp_path, 22, edge_mm=67, thumb=1.30, passed=False)
    for candidate, median, minimum, failed_count in (
        (closer, 0.006533077504739845, 0.00624240620480189, 23),
        (almost_zero, 0.00005638431217497991, 0.00005541313399178016, 7),
    ):
        candidate.result["candidate_id"] = int(candidate.candidate_id)
        candidate.result["summary"]["metrics"].update(
            {
                "operation_median_lift_m": median,
                "operation_minimum_lift_m": minimum,
            }
        )
        candidate.result["summary"]["failed_checks"] = [
            f"synthetic_failed_check_{index}" for index in range(failed_count)
        ]
        candidate.result["manipulation_response"] = {
            "available": True,
            "response_6d": [0.0, 0.0, median, 0.0, 0.0, 0.0],
            "lateral_displacement_m": 0.0,
            "orientation_change_deg": 0.0,
        }

    assert [
        candidate.candidate_id
        for candidate in sorted((almost_zero, closer), key=lambda value: value.rank)
    ] == [closer.candidate_id, almost_zero.candidate_id]
    assert closer.rank == lift_candidate_rank(closer.result)


def test_catalog_only_creates_success_alias_after_same_run_rerun(
    tmp_path, monkeypatch
):
    passing = _candidate(tmp_path, 1, edge_mm=60, thumb=1.25)
    near_miss = _candidate(
        tmp_path, 2, edge_mm=61, thumb=1.35, passed=False
    )

    def fake_run(config, *, trace_path=None, video_path=None):
        assert trace_path is not None
        np.savez_compressed(
            trace_path,
            time=np.asarray([0.001]),
            cube_pos=np.zeros((1, 3)),
            cube_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        )
        passed = int(round(config["cube"]["edge_m"] * 1000.0)) == 60
        return {
            "passed": passed,
            "failed_checks": [] if passed else ["operation_median_lift_reached"],
            "stage_status": {
                "grasp_success": True,
                "manipulation_success": passed,
                "full_success": passed,
            },
            "metrics": {},
        }

    monkeypatch.setattr(
        "xhand_grasp.normal_aligned_smooth_lift_catalog.run_simulation",
        fake_run,
    )
    _patch_publish_evidence(monkeypatch)
    output = tmp_path / "catalog"

    catalog = export_normal_aligned_smooth_lift_catalog(
        [passing, near_miss],
        output,
        selected_count=1,
        video=False,
    )

    assert catalog["aliases"]["best_nominal"].startswith("smooth_lift_")
    assert catalog["aliases"]["best_attempt"].startswith("smooth_lift_")
    assert catalog["aliases"]["best_nominal"] != catalog["aliases"]["best_attempt"]
    replay = resolve_replay_source(
        catalog_path=output / "catalog.json", trajectory="best_nominal"
    )
    assert replay.config_path.is_file()
    assert replay.trace_path.is_file()


def test_catalog_with_only_near_miss_does_not_claim_best_nominal(
    tmp_path, monkeypatch
):
    candidate = _candidate(tmp_path, 9, edge_mm=67, thumb=1.40, passed=False)

    def fake_run(config, *, trace_path=None, video_path=None):
        np.savez_compressed(trace_path, time=np.asarray([0.001]))
        return {
            "passed": False,
            "failed_checks": ["operation_median_lift_reached"],
            "stage_status": {"grasp_success": True, "full_success": False},
            "metrics": {},
        }

    monkeypatch.setattr(
        "xhand_grasp.normal_aligned_smooth_lift_catalog.run_simulation",
        fake_run,
    )
    _patch_publish_evidence(monkeypatch)

    catalog = export_normal_aligned_smooth_lift_catalog(
        [candidate], tmp_path / "catalog", video=False
    )

    assert "best_nominal" not in catalog["aliases"]
    assert "best_attempt" in catalog["aliases"]
    assert catalog["success_count"] == 0


def test_authenticated_loader_binds_config_trace_pose_and_controller(
    tmp_path, monkeypatch
):
    source = _authenticated_source(tmp_path / "candidate")
    monkeypatch.setattr(
        catalog_module,
        "_verify_same_run_trace",
        lambda _config, _trace, _summary: {},
    )

    candidate = load_catalog_candidate(source)

    assert candidate.stage == "lift_exact"
    assert candidate.trace_path == source.parent / "trace.npz"
    assert candidate.pose_id == pose_id(candidate.config)
    assert candidate.controller_id == controller_id(candidate.config)

    with (source.parent / "trace.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="trace SHA-256 mismatch"):
        load_catalog_candidate(source)


def test_authenticated_loader_rejects_non_exact_stage(tmp_path, monkeypatch):
    source = _authenticated_source(
        tmp_path / "candidate", stage="lift_refine"
    )
    monkeypatch.setattr(
        catalog_module,
        "_verify_same_run_trace",
        lambda _config, _trace, _summary: {},
    )

    with pytest.raises(ValueError, match="lift_exact"):
        load_catalog_candidate(source)


@pytest.mark.parametrize(
    "provenance",
    (
        {"config_stage": "lift_refine"},
        {"config_candidate_id": 89_000_000_000_002},
        {"locked_timestep_s": 0.002},
    ),
)
def test_authenticated_loader_binds_locked_exact_config_provenance(
    tmp_path, monkeypatch, provenance
):
    source = _authenticated_source(tmp_path / "candidate", **provenance)
    monkeypatch.setattr(
        catalog_module,
        "_verify_same_run_trace",
        lambda _config, _trace, _summary: {},
    )

    with pytest.raises(ValueError, match="candidate provenance"):
        load_catalog_candidate(source)


def test_catalog_exporter_rejects_direct_non_exact_candidate_before_writing(
    tmp_path, monkeypatch
):
    candidate = replace(
        _candidate(tmp_path, 23, edge_mm=67, thumb=1.30),
        stage="lift_refine",
    )
    output = tmp_path / "catalog"
    monkeypatch.setattr(
        catalog_module,
        "run_simulation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-exact candidate reached simulation")
        ),
    )

    with pytest.raises(ValueError, match="only lift_exact"):
        export_normal_aligned_smooth_lift_catalog(
            [candidate], output, selected_count=1, video=False
        )

    assert not output.exists()


def test_discovery_deduplicates_tuners_intermediate_catalog_copy(
    tmp_path, monkeypatch
):
    original = _authenticated_source(
        tmp_path / "candidates" / "candidate_89000000000001"
    )
    copied_directory = tmp_path / "trajectory_catalog" / "trajectory_01"
    copied_directory.mkdir(parents=True)
    for name in ("resolved_config.json", "trace.npz", "result.json"):
        (copied_directory / name).write_bytes((original.parent / name).read_bytes())
    monkeypatch.setattr(
        catalog_module,
        "_verify_same_run_trace",
        lambda _config, _trace, _summary: {},
    )

    candidates = discover_catalog_candidates([tmp_path])

    assert len(candidates) == 1
    assert "trajectory_catalog" not in candidates[0].result_path.parts


def test_reported_near_miss_that_passes_rerun_gets_success_alias(
    tmp_path, monkeypatch
):
    candidate = _candidate(tmp_path, 12, edge_mm=63, thumb=1.40, passed=False)

    def fake_run(_config, *, trace_path=None, video_path=None):
        assert video_path is None
        np.savez_compressed(trace_path, time=np.asarray([0.001]))
        return {
            "passed": True,
            "failed_checks": [],
            "stage_status": {
                "grasp_success": True,
                "manipulation_success": True,
                "full_success": True,
            },
            "metrics": {},
        }

    monkeypatch.setattr(catalog_module, "run_simulation", fake_run)
    _patch_publish_evidence(monkeypatch)

    catalog = export_normal_aligned_smooth_lift_catalog(
        [candidate], tmp_path / "catalog", selected_count=1, video=False
    )

    assert "best_nominal" in catalog["aliases"]
    assert "best_attempt" not in catalog["aliases"]
    assert catalog["success_count"] == 1
    entry = catalog["trajectories"][0]
    result = json.loads(
        (tmp_path / "catalog" / entry["artifacts"]["result"]).read_text()
    )
    assert result["reported_classification"] == "near_miss"
    assert result["classification"] == "success"
    assert result["trace_summary_consistent"]


def test_reported_success_failure_rolls_back_catalog(tmp_path, monkeypatch):
    candidate = _candidate(tmp_path, 13, edge_mm=64, thumb=1.40, passed=True)

    def fake_run(_config, *, trace_path=None, video_path=None):
        del video_path
        np.savez_compressed(trace_path, time=np.asarray([0.001]))
        return {
            "passed": False,
            "failed_checks": ["operation_median_lift_reached"],
            "stage_status": {"grasp_success": True, "full_success": False},
            "metrics": {},
        }

    monkeypatch.setattr(catalog_module, "run_simulation", fake_run)
    _patch_publish_evidence(monkeypatch)
    output = tmp_path / "catalog"

    with pytest.raises(RuntimeError, match="failed deterministic rerun"):
        export_normal_aligned_smooth_lift_catalog(
            [candidate], output, selected_count=1, video=False
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".catalog.staging.*"))


def test_video_is_bound_to_npz_steps_and_ffprobe_evidence(
    tmp_path, monkeypatch
):
    candidate = _candidate(tmp_path, 14, edge_mm=65, thumb=1.40, passed=False)

    def fake_run(_config, *, trace_path=None, video_path=None):
        np.savez_compressed(trace_path, time=np.asarray([0.001]))
        video_path.write_bytes(b"synthetic-mp4")
        return {
            "passed": False,
            "failed_checks": ["operation_median_lift_reached"],
            "stage_status": {"grasp_success": True, "full_success": False},
            "metrics": {},
            "video": {
                "decode_verified": True,
                "codec": "h264",
                "width": 640,
                "height": 480,
                "fps": "30/1",
                "frame_count": 1,
                "simulation_step_indices": [1],
            },
        }

    monkeypatch.setattr(catalog_module, "run_simulation", fake_run)
    monkeypatch.setattr(
        catalog_module,
        "_verify_same_run_trace",
        lambda _config, _trace, _summary: {
            "video_frame_steps": np.asarray([1], dtype=np.int64)
        },
    )
    monkeypatch.setattr(
        catalog_module,
        "probe_video",
        lambda _path, frames, _settings: {
            "decode_verified": True,
            "codec": "h264",
            "width": 640,
            "height": 480,
            "fps": "30/1",
            "frame_count": frames,
        },
    )
    monkeypatch.setattr(
        catalog_module,
        "_validated_run_metadata",
        lambda path: {"config_sha256": file_sha256(path)},
    )

    catalog = export_normal_aligned_smooth_lift_catalog(
        [candidate], tmp_path / "catalog", selected_count=1, video=True
    )

    entry = catalog["trajectories"][0]
    result = json.loads(
        (tmp_path / "catalog" / entry["artifacts"]["result"]).read_text()
    )
    assert result["video_verified"]
    video_path = tmp_path / "catalog" / entry["artifacts"]["video"]
    assert entry["artifacts"]["sha256"]["video"] == file_sha256(video_path)


def test_v8_resolved_status_distinguishes_nominal_and_parameter_override():
    config = load_config(
        "grasp_configs/"
        "left_opposed_face_palm_down_high_thumb_normal_aligned_"
        "smooth_vertical_lift.json"
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

    nominal = resolved_run_config(config, summary)["experiment_status"]
    assert nominal["classification"] == (
        "validated_normal_aligned_smooth_lift_nominal"
    )
    assert nominal["campaign_validated"]
    overridden_config = copy.deepcopy(config)
    overridden_config["run_context"] = {"kind": "parameter_override_run"}
    overridden = resolved_run_config(overridden_config, summary)[
        "experiment_status"
    ]
    assert overridden["classification"] == "parameter_override_run"
    assert overridden["passed"]
    assert not overridden["campaign_validated"]
