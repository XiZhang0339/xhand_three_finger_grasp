from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import xhand_grasp.pose_preserving_seed_catalog as catalog_module
from xhand_grasp.config import load_config
from xhand_grasp.controller import grasp_gate_order
from xhand_grasp.pose_preserving_seed_catalog import (
    REQUIRED_ACQUISITION_CHECKS,
    export_pose_preserving_seed_catalog,
)
from xhand_grasp.viewer import resolve_viewer_source


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_pose_preserving_grasp.json"
)


def _trace(*, historical_excursion: bool = False) -> dict[str, np.ndarray]:
    total = 300
    acquisition = 249
    time = np.arange(1, total + 1, dtype=np.float64) * 0.001
    states = np.full(total, "VERIFY", dtype="<U10")
    states[:10] = "SETTLE"
    acquired = np.zeros(total, dtype=bool)
    acquired[acquisition:] = True
    cube_pos = np.zeros((total, 3), dtype=np.float64)
    cube_pos[:, 2] = 0.114
    if historical_excursion:
        # Returning before acquisition must not erase this earlier violation.
        cube_pos[100, 0] = 0.0006
    cube_quat = np.zeros((total, 4), dtype=np.float64)
    cube_quat[:, 0] = 1.0
    gate_order = np.asarray(grasp_gate_order(6))
    gate = np.ones((total, len(gate_order)), dtype=bool)
    counter = np.minimum(np.arange(1, total + 1), 250)
    return {
        "time": time,
        "control_state": states,
        "grasp_gate_order": gate_order,
        "grasp_gate": gate,
        "grasp_gate_consecutive_steps": counter,
        "grasp_acquired": acquired,
        "manipulation_progress": np.zeros(total, dtype=np.float64),
        "grasp_acquisition_step": np.asarray(acquisition, dtype=np.int64),
        "manipulation_start_step": np.asarray(250, dtype=np.int64),
        "manipulation_end_step": np.asarray(299, dtype=np.int64),
        "termination_step": np.asarray(299, dtype=np.int64),
        "cube_pos": cube_pos,
        "cube_quat": cube_quat,
        "initial_cube_pos_m": np.asarray([0.0, 0.0, 0.114]),
        "initial_cube_quat": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "support_contact": np.ones(total, dtype=bool),
        "hand_cube_contact": np.zeros(total, dtype=bool),
        "pregrasp_pose_preserved_latched": np.ones(total, dtype=bool),
        "pregrasp_support_retained_latched": np.ones(total, dtype=bool),
        "settle_hand_contact_free_latched": np.ones(total, dtype=bool),
        "finger_order": np.asarray(("thumb", "index", "mid")),
    }


def _summary(*, grasp_success: bool = True) -> dict:
    checks = {name: grasp_success for name in REQUIRED_ACQUISITION_CHECKS}
    return {
        "passed": False,
        "failed_checks": [] if grasp_success else ["stable_grasp_acquired"],
        "checks": checks,
        "stage_status": {
            "grasp_success": grasp_success,
            "manipulation_success": False,
            "full_success": False,
        },
        "metrics": {
            "grasp_acquisition_step": 249,
            "manipulation_start_step": 250,
            "manipulation_end_step": 299,
            "termination_step": 299,
            "pose_preservation": {
                "max_translation_m": 0.0002,
                "max_orientation_drift_deg": 0.4,
            },
        },
        "phase_steps": {
            "settle": 500,
            "close": 1000,
            "verify": 750,
            "manipulate": 1500,
            "hold": 1000,
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(
    root: Path,
    source_id: str,
    *,
    historical_excursion: bool = False,
    reported_grasp_success: bool = True,
) -> Path:
    directory = root / source_id
    directory.mkdir(parents=True)
    config = load_config(CONFIG)
    config_path = directory / "resolved_config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trace_path = directory / "trace.npz"
    np.savez_compressed(
        trace_path, **_trace(historical_excursion=historical_excursion)
    )
    summary = _summary(grasp_success=reported_grasp_success)
    result = {
        "source_id": source_id,
        "summary": summary,
        "artifacts": {
            "resolved_config": config_path.name,
            "trace": trace_path.name,
            "sha256": {
                "resolved_config": _sha256(config_path),
                "trace": _sha256(trace_path),
            },
        },
    }
    (directory / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


def _convert_to_dynamic_best_layout(directory: Path, source_candidate_id: int) -> None:
    config = directory / "resolved_config.json"
    trace = directory / "trace.npz"
    result = directory / "result.json"
    best_config = directory / "best_config.json"
    best_trace = directory / "best_trace.npz"
    best_result = directory / "best_result.json"
    config.rename(best_config)
    trace.rename(best_trace)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["artifacts"] = {
        "resolved_config": best_config.name,
        "trace": best_trace.name,
        "sha256": {
            "resolved_config": _sha256(best_config),
            "trace": _sha256(best_trace),
        },
    }
    best_result.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result.unlink()
    source_payload = {
        "source_candidate_id": source_candidate_id,
        "best_acquisition_success": True,
        "best_artifacts": {
            "resolved_config": best_config.name,
            "result": best_result.name,
            "trace": best_trace.name,
            "sha256": {
                "resolved_config": _sha256(best_config),
                "result": _sha256(best_result),
                "trace": _sha256(best_trace),
            },
        },
    }
    (directory / "source_result.json").write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_catalog_keeps_every_source_but_aliases_only_true_sticky_grasps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = tmp_path / "runs"
    _write_source(sources, "117060")
    _write_source(sources, "116060", historical_excursion=True)
    recomputed = _summary()
    monkeypatch.setattr(
        catalog_module,
        "_recompute_summary",
        lambda config, traces: copy.deepcopy(recomputed),
    )

    # Production keeps the published catalog beside the tuner's source_* run
    # directories.  Discovery is complete before the atomic staging directory
    # is created, so this nested destination remains safe.
    output = sources / "trajectory_catalog"
    result = export_pose_preserving_seed_catalog(
        sources, output, expected_count=2, best_source_id="117060"
    )

    assert result["trajectory_count"] == 2
    assert result["validated_grasp_count"] == 1
    assert result["failed_grasp_count"] == 1
    assert result["aliases"] == {
        "best_nominal": "pose_preserving_117060",
        "grasp_117060": "pose_preserving_117060",
    }
    by_source = {entry["source_id"]: entry for entry in result["trajectories"]}
    assert by_source["117060"]["aliases"] == ["grasp_117060", "best_nominal"]
    assert by_source["116060"]["aliases"] == []
    assert by_source["116060"]["classification"].startswith("failed_")
    assert "raw_pose_preserved_inclusive" in by_source["116060"][
        "failed_checks"
    ]

    # The generic Viewer resolver accepts both the success alias and an honest
    # failed entry selected explicitly by label; aliases never imply success
    # for the latter.
    source = resolve_viewer_source(
        catalog_path=output / "catalog.json", trajectory="best_nominal"
    )
    assert source.trajectory == "117060"
    assert source.config_path.is_file()
    assert source.trace_path is not None and source.trace_path.is_file()
    failed_source = resolve_viewer_source(
        catalog_path=output / "catalog.json", trajectory="116060"
    )
    assert failed_source.trajectory == "116060"

    published = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    success_artifacts = published["trajectories"][0]["artifacts"]
    for field in ("resolved_config", "result", "trace", "grasp_trace"):
        member = output / success_artifacts[field]
        assert _sha256(member) == success_artifacts["sha256"][field]


def test_catalog_refuses_an_authenticated_trace_changed_after_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = tmp_path / "runs"
    source = _write_source(sources, "117060")
    with (source / "trace.npz").open("ab") as stream:
        stream.write(b"tampered")
    monkeypatch.setattr(
        catalog_module, "_recompute_summary", lambda config, traces: _summary()
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch for trace"):
        export_pose_preserving_seed_catalog(
            sources, tmp_path / "published", expected_count=1
        )


def test_reported_success_that_offline_evaluation_rejects_has_no_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = tmp_path / "runs"
    _write_source(sources, "117060", reported_grasp_success=True)
    recomputed = _summary(grasp_success=False)
    monkeypatch.setattr(
        catalog_module,
        "_recompute_summary",
        lambda config, traces: copy.deepcopy(recomputed),
    )

    result = export_pose_preserving_seed_catalog(
        sources, tmp_path / "published", expected_count=1
    )

    assert result["validated_grasp_count"] == 0
    assert result["aliases"] == {}
    entry = result["trajectories"][0]
    assert entry["aliases"] == []
    assert "recomputed_stage_status.grasp_success" in entry["failed_checks"]
    assert "reported_stage_status.grasp_success" in entry["failed_checks"]


def test_catalog_reads_dynamic_tuner_best_artifacts_without_renaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = tmp_path / "runs"
    directory = _write_source(sources, "source_00_117060")
    _convert_to_dynamic_best_layout(directory, 117060)
    monkeypatch.setattr(
        catalog_module,
        "_recompute_summary",
        lambda config, traces: copy.deepcopy(_summary()),
    )

    output = tmp_path / "published"
    result = export_pose_preserving_seed_catalog(
        sources, output, expected_count=1, best_source_id="117060"
    )

    assert result["aliases"]["best_nominal"] == "pose_preserving_117060"
    entry = result["trajectories"][0]
    assert entry["source_id"] == "117060"
    persisted_result = json.loads(
        (output / entry["artifacts"]["result"]).read_text(encoding="utf-8")
    )
    assert persisted_result["source_provenance"]["source_result_sha256"] == _sha256(
        directory / "source_result.json"
    )
