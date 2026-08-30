from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.grasp_pose import canonical_sha256, controller_id, grasp_pose_id
from xhand_grasp.tuning import actual_qpos_sources as sources


ROOT = Path(__file__).resolve().parents[1]
V9_CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)


def _write_v2_source(
    root: Path,
    name: str,
    *,
    source_kind: str,
    grasp_success: bool,
    stable_count: int,
    qpos_offset: float = 0.0,
) -> dict:
    directory = root / "inputs" / name
    directory.mkdir(parents=True)
    config = copy.deepcopy(load_config(V9_CONFIG))
    config["hand_pose"]["translation_m"][0] += qpos_offset * 0.001
    nominal = np.asarray(
        [config["grasp_pose"]["nominal_joint_qpos_rad"][key] for key in ACTIVE_ACTUATORS],
        dtype=np.float64,
    )
    actual = nominal + qpos_offset
    config_path = directory / "resolved_config.json"
    trace_path = directory / "trace.npz"
    result_path = directory / "result.json"
    write_json(config_path, config)

    sample_count = max(300, stable_count + 20)
    gate = np.zeros((sample_count, 3), dtype=bool)
    start = 10
    end = start + stable_count - 1
    if stable_count:
        gate[start : end + 1] = True
    history = np.repeat(actual[None, :], sample_count, axis=0)
    archive = {
        "time": np.arange(1, sample_count + 1, dtype=np.float64) * 0.001,
        "grasp_gate": gate,
        "grasp_pose_actual_joint_qpos_rad": history,
        "grasp_pose_actual_qpos_rad": actual,
    }
    if stable_count:
        archive.update(
            {
                "grasp_stable_window_start_step": np.asarray(start),
                "grasp_stable_window_end_step": np.asarray(end),
                "grasp_lock_step": np.asarray(end),
            }
        )
    np.savez_compressed(trace_path, **archive)
    result = {
        "candidate_id": name,
        "candidate_sha256": canonical_sha256(config),
        "grasp_pose_id": grasp_pose_id(config),
        "controller_id": controller_id(config),
        "summary": {
            "stage_status": {
                "grasp_success": grasp_success,
                "manipulation_success": False,
                "full_success": False,
            }
        },
        "artifacts": {
            "sha256": {
                "resolved_config": file_sha256(config_path),
                "trace": file_sha256(trace_path),
            }
        },
    }
    write_json(result_path, result)
    return {
        "source_kind": source_kind,
        "pose_id": grasp_pose_id(config),
        "config_path": config_path.relative_to(root).as_posix(),
        "result_path": result_path.relative_to(root).as_posix(),
        "trace_path": trace_path.relative_to(root).as_posix(),
        "sha256": {
            "config": file_sha256(config_path),
            "result": file_sha256(result_path),
            "trace": file_sha256(trace_path),
            "config_semantic": canonical_sha256(config),
        },
    }


def _write_v2_manifest(root: Path, records: list[dict]) -> Path:
    path = root / "source_manifest.json"
    write_json(
        path,
        {
            "actual_qpos_source_manifest_schema_version": 2,
            "sources": records,
        },
    )
    return path


def test_v2_accepts_any_nonempty_count_and_certifies_250_ms_gate(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sources, "REPO_ROOT", tmp_path)
    records = [
        _write_v2_source(
            tmp_path,
            f"source_{index}",
            source_kind=sources.AUTHENTICATED_GRASP_SUCCESS,
            grasp_success=True,
            stable_count=250,
            qpos_offset=index * 0.001,
        )
        for index in range(2)
    ]
    manifest = _write_v2_manifest(tmp_path, records)

    loaded = sources.load_actual_qpos_sources({}, source_manifest_path=manifest)

    assert len(loaded) == 2
    assert all(item.manifest_schema_version == 2 for item in loaded)
    assert all(item.stable_window_sample_count == 250 for item in loaded)
    assert all(item.stable_window_duration_s == pytest.approx(0.25) for item in loaded)
    assert all(item.eligible_as_success_evidence for item in loaded)
    assert loaded[0].generator_record()["eligible_as_success_evidence"] is True


def test_v2_diagnostic_seed_is_hash_bound_but_never_success_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sources, "REPO_ROOT", tmp_path)
    record = _write_v2_source(
        tmp_path,
        "diagnostic",
        source_kind=sources.DIAGNOSTIC_GEOMETRY_SEED,
        grasp_success=False,
        stable_count=0,
    )
    manifest = _write_v2_manifest(tmp_path, [record])

    (loaded,) = sources.load_actual_qpos_sources({}, source_manifest_path=manifest)

    assert loaded.source_kind == sources.DIAGNOSTIC_GEOMETRY_SEED
    assert loaded.stable_window_sample_count == 0
    assert loaded.stable_window_start_step == -1
    assert loaded.gate_evidence_authenticated is False
    assert loaded.eligible_as_success_evidence is False
    generated = loaded.generator_record()
    assert generated["source_kind"] == sources.DIAGNOSTIC_GEOMETRY_SEED
    assert generated["eligible_as_success_evidence"] is False


def test_v2_cannot_promote_failed_result_by_labeling_it_authenticated(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sources, "REPO_ROOT", tmp_path)
    record = _write_v2_source(
        tmp_path,
        "failed",
        source_kind=sources.AUTHENTICATED_GRASP_SUCCESS,
        grasp_success=False,
        stable_count=250,
    )
    manifest = _write_v2_manifest(tmp_path, [record])

    with pytest.raises(ValueError, match="not grasp-success"):
        sources.load_actual_qpos_sources({}, source_manifest_path=manifest)


def test_v2_authenticated_source_requires_full_250_ms_window(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sources, "REPO_ROOT", tmp_path)
    record = _write_v2_source(
        tmp_path,
        "short",
        source_kind=sources.AUTHENTICATED_GRASP_SUCCESS,
        grasp_success=True,
        stable_count=249,
    )
    manifest = _write_v2_manifest(tmp_path, [record])

    with pytest.raises(ValueError, match="no 250 ms stable contact window"):
        sources.load_actual_qpos_sources({}, source_manifest_path=manifest)


def test_v2_rejects_file_hash_change_and_repo_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "REPO_ROOT", tmp_path)
    record = _write_v2_source(
        tmp_path,
        "tampered",
        source_kind=sources.AUTHENTICATED_GRASP_SUCCESS,
        grasp_success=True,
        stable_count=250,
    )
    manifest = _write_v2_manifest(tmp_path, [record])
    config_path = tmp_path / record["config_path"]
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="config hash changed"):
        sources.load_actual_qpos_sources({}, source_manifest_path=manifest)

    escaping = copy.deepcopy(record)
    escaping["config_path"] = "../outside.json"
    manifest = _write_v2_manifest(tmp_path, [escaping])
    with pytest.raises(ValueError, match="repository-relative"):
        sources.load_actual_qpos_sources({}, source_manifest_path=manifest)


def test_legacy_v1_wrapper_accepts_non_21_source_manifest(tmp_path):
    directory = tmp_path / "legacy"
    directory.mkdir()
    config = load_config(V9_CONFIG)
    config_path = directory / "resolved_config.json"
    result_path = directory / "result.json"
    trace_path = directory / "trace.npz"
    write_json(config_path, config)
    joint = np.zeros((260, len(ACTIVE_ACTUATORS)), dtype=np.float64)
    gate = np.zeros((260, 2), dtype=bool)
    gate[5:255] = True
    np.savez_compressed(
        trace_path,
        grasp_gate=gate,
        joint_qpos=joint,
        actuator_order=np.asarray(ACTIVE_ACTUATORS),
    )
    write_json(
        result_path,
        {
            "artifacts": {
                "trace": "trace.npz",
                "sha256": {"trace": file_sha256(trace_path)},
            }
        },
    )
    manifest = tmp_path / "legacy_manifest.json"
    write_json(
        manifest,
        {
            "pose_rescue_manifest_schema_version": 1,
            "poses": [
                {
                    "pose_id": "legacy-one",
                    "source_config": str(config_path),
                    "source_result": str(result_path),
                    "source_config_file_sha256": file_sha256(config_path),
                    "source_result_sha256": file_sha256(result_path),
                    "source_config_sha256": canonical_sha256(config),
                }
            ],
        },
    )

    loaded = sources.load_v8_actual_qpos_sources(
        {}, source_manifest_path=manifest
    )

    assert len(loaded) == 1
    assert isinstance(loaded[0], sources.ActualQposSource)
    assert isinstance(loaded[0], sources.V8ActualQposSource)
    assert loaded[0].pose_id == "legacy-one"
    assert loaded[0].eligible_as_success_evidence is True


def test_source_manifest_rejects_empty_and_unknown_versions(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(
        json.dumps(
            {
                "actual_qpos_source_manifest_schema_version": 2,
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty"):
        sources.load_actual_qpos_sources({}, source_manifest_path=empty)

    unknown = tmp_path / "unknown.json"
    unknown.write_text(
        json.dumps(
            {
                "actual_qpos_source_manifest_schema_version": 3,
                "sources": [{}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported"):
        sources.load_actual_qpos_sources({}, source_manifest_path=unknown)
