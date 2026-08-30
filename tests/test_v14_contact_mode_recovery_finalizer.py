from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    authenticated_catalog_artifact_paths,
    initialize_or_resume_campaign,
)
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.contact_preserving_candidate_artifacts import (
    run_or_resume_v14_candidate_artifacts,
)
from xhand_grasp.tuning.contact_preserving_contact_mode_pose_rescue import (
    contact_mode_physical_config_sha256,
    contact_mode_trace_diagnostics,
)
from xhand_grasp.tuning.contact_preserving_contact_mode_recovery_finalizer import (
    _RecoveryBackend,
    _RecoveryPolicy,
    _authenticate_source_workspace,
    _build_recovery_manifest_from_source,
    _run_recovery_finalizer,
    compare_contact_mode_physical_traces,
)
from xhand_grasp.tuning.contact_preserving_lift_rescue_campaign import (
    _commit_report,
)
from xhand_grasp.tuning.contact_preserving_planned_lift_campaign import (
    build_contact_preserving_planned_lift_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)
TEST_POLICY = _RecoveryPolicy(search_candidate_count=5, published_candidate_count=5)


def _summary() -> dict[str, Any]:
    return {
        "passed": False,
        "failed_checks": ["smooth_motion_jerk_within_limit"],
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": False,
            "full_success": False,
        },
        "finite": True,
    }


def _trace_payload(*, changed: bool = False) -> dict[str, np.ndarray]:
    count = 6
    cube = np.zeros((count, 3), dtype=np.float64)
    if changed:
        cube[3, 2] = 1e-9
    centroids = np.zeros((count, 3, 3), dtype=np.float64)
    centroids[:, 0, 1] = -0.01
    centroids[:, 1, 1] = 0.00
    centroids[:, 2, 1] = 0.01
    return {
        "time": np.arange(count, dtype=np.float64) * 0.001,
        "cube_pos": cube,
        "control_state": np.asarray(
            ("SETTLE", "VERIFY", "MANIPULATE", "MANIPULATE", "HOLD", "HOLD")
        ),
        "distal_active_taxel_count": np.ones((count, 3), dtype=np.int64),
        "target_face_contact_centroid_cube_local_m": centroids,
        "target_face_contact_centroid_valid": np.ones((count, 3), dtype=bool),
        "target_face_effective": np.ones((count, 3), dtype=bool),
        "operation_vertical_jerk_filtered_m_s3": np.asarray(
            (0.0, 0.0, 1.0, 2.0, 0.0, 0.0), dtype=np.float64
        ),
        "video_frame_steps": np.asarray((), dtype=np.int64),
    }


class _FakeSession:
    def __init__(self, *, changed: bool = False) -> None:
        self.steps = 0
        self.changed = changed

    @property
    def complete(self) -> bool:
        return self.steps >= 2

    def advance_one(self) -> None:
        self.steps += 1

    def finalize(self, *, trace_path=None) -> Mapping[str, Any]:
        if not self.complete:
            raise RuntimeError("fake session is incomplete")
        if trace_path is not None:
            np.savez_compressed(trace_path, **_trace_payload(changed=self.changed))
        return copy.deepcopy(_summary())

    def close(self) -> None:
        return None


def _run_fake_candidate(
    config: Mapping[str, Any],
    destination: Path,
    candidate_id: int,
    *,
    changed: bool = False,
) -> Mapping[str, Any]:
    bundle = run_or_resume_v14_candidate_artifacts(
        config,
        destination,
        candidate_id,
        final_rerun=True,
        retain_grasp_success=False,
        session_factory=lambda _config: _FakeSession(changed=changed),
        validator=None,
    )
    return bundle.result


def _copy_catalog_member(
    source_root: Path, destination_root: Path, rank: int, candidate_id: int
) -> dict[str, Any]:
    label = f"pair_rank_{rank:02d}_{candidate_id}"
    target = destination_root / label
    target.mkdir(parents=True)
    for name in ("resolved_config.json", "result.json", "trace.npz"):
        shutil.copy2(source_root / name, target / name)
    config = json.loads((target / "resolved_config.json").read_text(encoding="utf-8"))
    return {
        "trajectory_id": label,
        "candidate_id": str(candidate_id),
        "classification": "grasp_success_manipulation_near_miss",
        "grasp_success": True,
        "full_success": False,
        "edge_m": float(config["cube"]["edge_m"]),
        "object_config_id": config.get("object_config_id"),
        "grasp_pose_id": config.get("grasp_pose_id"),
        "grasp_object_pair_id": config.get("grasp_object_pair_id"),
        "planner_id": config.get("planner_id"),
        "controller_id": config.get("controller_id"),
        "artifacts": {
            "resolved_config": f"{label}/resolved_config.json",
            "result": f"{label}/result.json",
            "trace": f"{label}/trace.npz",
            "video": None,
            "sha256": {
                "resolved_config": file_sha256(target / "resolved_config.json"),
                "result": file_sha256(target / "result.json"),
                "trace": file_sha256(target / "trace.npz"),
            },
        },
    }


def _write_catalog(
    path: Path,
    candidate_roots: Sequence[tuple[int, Path]],
    *,
    kind: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        _copy_catalog_member(root, path.parent, rank + 1, candidate_id)
        for rank, (candidate_id, root) in enumerate(candidate_roots)
    ]
    write_json(
        path,
        {
            "trajectory_catalog_schema_version": 1,
            "complete": True,
            "experiment_id": load_config(CONFIG)["experiment_id"],
            "catalog_kind": kind,
            "production_trajectory_video_policy": (
                "disabled_for_injected_test_backend"
            ),
            "aliases": {"best_attempt": entries[0]["trajectory_id"]},
            "trajectories": entries,
        },
    )
    return path


def _source_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "immutable_source"
    manifest = build_contact_preserving_planned_lift_manifest(
        CONFIG, seed=20260821
    )
    manifest.pop("campaign_input_sha256", None)
    manifest.update(
        {
            "campaign_kind": "contact_preserving_contact_mode_pose_rescue",
            "contact_mode_pose_campaign_schema_version": 1,
            "budget": {"candidate_count": 5, "seed": 20260821},
            "source_sha256": "a" * 64,
            "source_files": ["immutable/old_source.py"],
            "source_authentication": {
                "schema_version": 1,
                "source_authentication_id": "b" * 64,
            },
            "source_evidence_sha256": {
                "resolved_config.json": "c" * 64,
                "result.json": "d" * 64,
                "trace.npz": "e" * 64,
            },
        }
    )
    manifest["campaign_input_sha256"] = canonical_sha256(manifest)
    initialize_or_resume_campaign(workspace, manifest, resume=False)

    audit_path = workspace / "contact_mode_pose_source_audit.json"
    audit = {
        "contact_mode_pose_source_audit_schema_version": 1,
        "complete": True,
        "source": copy.deepcopy(manifest["source_authentication"]),
        "source_evidence_sha256": copy.deepcopy(
            manifest["source_evidence_sha256"]
        ),
        "records": [],
    }
    _commit_report(
        workspace,
        "contact_mode_pose_source_audit",
        audit_path,
        audit,
        stage_input={"source": "synthetic"},
    )

    candidate_roots: list[tuple[int, Path]] = []
    records: list[dict[str, Any]] = []
    candidate_artifacts: list[Path] = []
    template = load_config(CONFIG)
    for index in range(5):
        candidate_id = 15_600_000_000_000_000 + index
        config = copy.deepcopy(template)
        config["hand_pose"]["translation_m"][0] += index * 1e-6
        root = (
            workspace
            / "contact_mode_pose_search"
            / "candidates"
            / f"candidate_{candidate_id}"
        )
        result = _run_fake_candidate(config, root, candidate_id)
        with np.load(root / "trace.npz", allow_pickle=False) as trace:
            diagnostics = contact_mode_trace_diagnostics(trace)
        record = {
            **copy.deepcopy(dict(result)),
            "artifact_directory": str(root.relative_to(workspace)),
            "physical_config_sha256": contact_mode_physical_config_sha256(config),
            "is_exact_parent_reproduction_baseline": index == 0,
            "contact_mode_rank": index,
            "contact_mode_diagnostics": diagnostics,
        }
        records.append(record)
        candidate_roots.append((candidate_id, root))
        candidate_artifacts.extend(
            root / name
            for name in ("resolved_config.json", "result.json", "trace.npz")
        )
    search_path = workspace / "contact_mode_pose_search" / "report.json"
    search = {
        "contact_mode_pose_search_report_schema_version": 1,
        "complete": True,
        "source_authentication_id": "b" * 64,
        "candidate_count": 5,
        "grasp_success_count": 5,
        "full_success_count": 0,
        "new_full_success_count": 0,
        "source_reproduction_candidate_count": 1,
        "new_unique_candidate_count": 4,
        "physical_unique_trial_count": 5,
        "physical_duplicate_against_source_count": 1,
        "trace_retained_candidate_count": 5,
        "records": records,
    }
    _commit_report(
        workspace,
        "contact_mode_pose_search",
        search_path,
        search,
        stage_input={"search": "synthetic"},
        artifacts=tuple(candidate_artifacts),
    )

    catalog_paths = {
        kind: _write_catalog(
            workspace / "catalogs" / "target_1" / kind / "catalog.json",
            candidate_roots,
            kind=kind,
        )
        for kind in ("grasp_pose", "manipulation")
    }
    catalog_report_path = workspace / "catalogs" / "target_1" / "report.json"
    catalog_report = {
        "contact_mode_pose_catalog_report_schema_version": 1,
        "complete": True,
        "target_success_count": 1,
        "target_reached": False,
        "full_success_count": 0,
        "published_candidate_count": 5,
        "published_candidate_ids": [value[0] for value in candidate_roots],
        "catalogs": {
            key: str(value.relative_to(workspace))
            for key, value in catalog_paths.items()
        },
        "records": [],
    }
    artifacts: list[Path] = []
    for path in catalog_paths.values():
        artifacts.extend(authenticated_catalog_artifact_paths(path))
    _commit_report(
        workspace,
        "contact_mode_pose_catalog_target_1",
        catalog_report_path,
        catalog_report,
        stage_input={"catalog": "synthetic"},
        artifacts=tuple(dict.fromkeys(artifacts)),
    )
    return workspace


def _fake_publisher(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    destination: Path,
    _experiment_id: str,
) -> Mapping[str, str]:
    roots = [
        (
            int(record["candidate_id"]),
            (workspace / str(record["artifact_directory"])).resolve(),
        )
        for record in records
    ]
    result = {}
    for kind in ("grasp_pose", "manipulation"):
        path = _write_catalog(destination / kind / "catalog.json", roots, kind=kind)
        result[kind] = str(path.relative_to(workspace))
    return result


def _backend(*, changed: bool = False) -> _RecoveryBackend:
    return _RecoveryBackend(
        backend_id=f"injected_exact_backend_changed_{int(changed)}",
        candidate_runner=lambda config, destination, candidate_id: _run_fake_candidate(
            config, destination, candidate_id, changed=changed
        ),
        catalog_publisher=_fake_publisher,
    )


def test_source_authentication_binds_manifest_all_candidates_and_catalogs(
    tmp_path: Path,
) -> None:
    source_root = _source_workspace(tmp_path)
    source = _authenticate_source_workspace(source_root, policy=TEST_POLICY)
    assert len(source.records) == 5
    assert len(source.reference_trace_paths) == 5
    assert source.descriptor()["source_manifest_sha256"] == file_sha256(
        source_root / "campaign_manifest.json"
    )
    assert source.descriptor()["ledger_artifact_count"] == 50
    assert source.evidence_sha256 == canonical_sha256(source.evidence)


def test_source_authentication_fails_on_candidate_or_catalog_tamper(
    tmp_path: Path,
) -> None:
    source_root = _source_workspace(tmp_path)
    trace = next(
        (source_root / "contact_mode_pose_search" / "candidates").glob(
            "candidate_*/trace.npz"
        )
    )
    trace.write_bytes(trace.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _authenticate_source_workspace(source_root, policy=TEST_POLICY)

    source_root = _source_workspace(tmp_path / "catalog_case")
    catalog = source_root / "catalogs/target_1/manipulation/catalog.json"
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["aliases"] = {"changed": payload["trajectories"][0]["trajectory_id"]}
    write_json(catalog, payload)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _authenticate_source_workspace(source_root, policy=TEST_POLICY)


def test_trace_comparison_is_exact_except_renderer_frame_binding(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    left_payload = _trace_payload()
    right_payload = _trace_payload()
    right_payload["video_frame_steps"] = np.asarray((0, 3, 5), dtype=np.int64)
    np.savez_compressed(left, **left_payload)
    np.savez_compressed(right, **right_payload)
    evidence = compare_contact_mode_physical_traces(left, right)
    assert evidence["physical_fields_exact"] is True
    assert evidence["ignored_fields"] == ["video_frame_steps"]

    changed = _trace_payload(changed=True)
    np.savez_compressed(right, **changed)
    with pytest.raises(RuntimeError, match="cube_pos changed values"):
        compare_contact_mode_physical_traces(left, right)


def test_recovery_manifest_binds_current_source_and_old_execution_hash(
    tmp_path: Path,
) -> None:
    source = _authenticate_source_workspace(
        _source_workspace(tmp_path), policy=TEST_POLICY
    )
    manifest = _build_recovery_manifest_from_source(
        CONFIG, source, backend_id="test_backend"
    )
    assert manifest["campaign_kind"] == "contact_mode_pose_recovery_finalizer"
    assert manifest["immutable_old_manifest_sha256"] == file_sha256(
        source.root / "campaign_manifest.json"
    )
    assert manifest["source_workspace_evidence_sha256"] == source.evidence_sha256
    assert manifest["source_sha256"] != source.manifest["source_sha256"]
    assert manifest["campaign_input_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "campaign_input_sha256"}
    )


def test_recovery_runs_top5_publishes_result_and_resumes_without_rerun(
    tmp_path: Path,
) -> None:
    source = _source_workspace(tmp_path)
    output = tmp_path / "recovered"
    backend = _backend()
    result = _run_recovery_finalizer(
        CONFIG,
        output,
        source_workspace=source,
        resume=False,
        workers=1,
        policy=TEST_POLICY,
        backend=backend,
    )
    assert result["complete"] is True
    assert result["top5_current_source_rerun_exact"] is True
    assert result["full_success_count"] == 0
    assert result["exit_code"] == 2
    assert set(json.loads((output / "stage_ledger.json").read_text())["stages"]) == {
        "contact_mode_recovery_source_audit",
        "contact_mode_recovery_top5_rerun",
        "contact_mode_recovery_catalog",
        "contact_mode_recovery_result",
    }
    first_result = next(
        (output / "top5_current_source_reruns").glob("candidate_*/result.json")
    )
    before = file_sha256(first_result)

    resumed = _run_recovery_finalizer(
        CONFIG,
        output,
        source_workspace=source,
        resume=True,
        workers=1,
        policy=TEST_POLICY,
        backend=backend,
    )
    assert resumed == result
    assert file_sha256(first_result) == before


def test_recovery_stops_before_catalog_when_current_rerun_differs(
    tmp_path: Path,
) -> None:
    source = _source_workspace(tmp_path)
    output = tmp_path / "mismatch"
    with pytest.raises(RuntimeError, match="cube_pos changed values"):
        _run_recovery_finalizer(
            CONFIG,
            output,
            source_workspace=source,
            resume=False,
            workers=1,
            policy=TEST_POLICY,
            backend=_backend(changed=True),
        )
    ledger = json.loads((output / "stage_ledger.json").read_text(encoding="utf-8"))
    assert set(ledger["stages"]) == {"contact_mode_recovery_source_audit"}
    assert not (output / "catalogs").exists()


def test_recovery_resume_rejects_old_workspace_tamper(tmp_path: Path) -> None:
    source = _source_workspace(tmp_path)
    output = tmp_path / "recovered"
    backend = _backend()
    _run_recovery_finalizer(
        CONFIG,
        output,
        source_workspace=source,
        resume=False,
        workers=1,
        policy=TEST_POLICY,
        backend=backend,
    )
    report = source / "contact_mode_pose_search" / "report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["records"][0]["contact_mode_rank"] = 99
    write_json(report, payload)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _run_recovery_finalizer(
            CONFIG,
            output,
            source_workspace=source,
            resume=True,
            workers=1,
            policy=TEST_POLICY,
            backend=backend,
        )
