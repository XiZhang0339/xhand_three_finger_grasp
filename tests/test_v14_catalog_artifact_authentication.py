from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    authenticated_catalog_artifact_paths,
    bind_candidate_result_semantic_sha256,
)
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config


CONFIG = Path(
    "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _write_catalog(
    root: Path,
    *,
    result: dict[str, Any],
) -> tuple[Path, Path]:
    member = root / "trajectory"
    member.mkdir(parents=True)
    config_path = member / "resolved_config.json"
    result_path = member / "result.json"
    trace_path = member / "trace.npz"
    config = load_config(CONFIG)
    write_json(config_path, config)
    write_json(result_path, result)
    trace_path.write_bytes(b"catalog-authentication-trace")
    catalog_path = root / "catalog.json"
    write_json(
        catalog_path,
        {
            "trajectory_catalog_schema_version": 1,
            "complete": True,
            "experiment_id": config["experiment_id"],
            "aliases": {"best_nominal": "trajectory"},
            "trajectories": [
                {
                    "trajectory_id": "trajectory",
                    "classification": "success",
                    "artifacts": {
                        "resolved_config": "trajectory/resolved_config.json",
                        "result": "trajectory/result.json",
                        "trace": "trajectory/trace.npz",
                        "video": None,
                        "sha256": {
                            "resolved_config": file_sha256(config_path),
                            "result": file_sha256(result_path),
                            "trace": file_sha256(trace_path),
                        },
                    },
                }
            ],
        },
    )
    return catalog_path, result_path


def test_v14_catalog_authenticates_contact_preserving_result_semantics(
    tmp_path: Path,
) -> None:
    result = bind_candidate_result_semantic_sha256(
        {
            "contact_preserving_candidate_schema_version": 1,
            "complete": True,
            "candidate_id": 14_000_000_000_001,
            "classification": "success",
            "full_success": True,
            "grasp_success": True,
            "summary": {
                "passed": True,
                "stage_status": {"grasp_success": True, "full_success": True},
            },
        }
    )
    catalog_path, result_path = _write_catalog(tmp_path / "catalog", result=result)
    authenticated_catalog_artifact_paths(catalog_path)

    tampered = json.loads(result_path.read_text(encoding="utf-8"))
    tampered["classification"] = "near_miss"
    write_json(result_path, tampered)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["trajectories"][0]["artifacts"]["sha256"]["result"] = (
        file_sha256(result_path)
    )
    write_json(catalog_path, catalog)

    with pytest.raises(RuntimeError, match="semantic SHA-256 mismatch"):
        authenticated_catalog_artifact_paths(catalog_path)


def test_v14_production_catalog_rejects_missing_semantic_digest(
    tmp_path: Path,
) -> None:
    catalog_path, _ = _write_catalog(
        tmp_path / "unsealed_v14_catalog",
        result={
            "contact_preserving_candidate_schema_version": 1,
            "complete": True,
            "classification": "success",
            "summary": {"passed": True},
        },
    )

    with pytest.raises(RuntimeError, match="has no valid semantic SHA-256"):
        authenticated_catalog_artifact_paths(catalog_path)


def test_catalog_keeps_unversioned_legacy_result_compatibility(tmp_path: Path) -> None:
    catalog_path, _ = _write_catalog(
        tmp_path / "legacy_catalog",
        result={
            "complete": True,
            "classification": "diagnostic",
            "summary": {"passed": False},
        },
    )

    authenticated = authenticated_catalog_artifact_paths(catalog_path)
    assert catalog_path.resolve() in authenticated
