"""Authenticate an exhausted schema-v11 campaign for recovery search.

The recovery search is intentionally a new, independent campaign.  Before it
may reuse any near-miss, this module proves that the parent campaign is a
complete, immutable, zero-success run and authenticates every quick and
expanded static-cell payload.  It never writes to the parent directory.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..actual_contact_grasp_pose_catalog import validate_stage_ledger
from ..artifacts import REPO_ROOT, aggregate_source_sha256, file_sha256
from ..experiment import resolve_experiment
from ..grasp_pose import controller_id, grasp_pose_id
from .actual_contact_grasp_pose import (
    _static_cell_input_sha256,
    actual_contact_search_cells,
)
from .pose_preserving_seed_campaign import canonical_sha256


RECOVERY_PARENT_SNAPSHOT_SCHEMA_VERSION = 1
EXPECTED_EXPERIMENT_ID = (
    "left_opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_"
    "smooth_vertical_lift"
)
_STATIC_CELL_SCHEMA_VERSION = 1
_STATIC_STAGE_SCHEMA_VERSION = 1
_REFINEMENT_SCHEMA_VERSION = 1
_REQUIRED_STAGES = (
    "source_bundle",
    "quick_static",
    "quick_relative_wrist_orientation_refinement",
    "expanded_static",
    "expanded_relative_wrist_orientation_refinement",
    "campaign_result_1",
)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _bound_file(path_value: Any, expected_sha256: Any, label: str) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"parent {label} is missing: {path}")
    if file_sha256(path) != str(expected_sha256):
        raise RuntimeError(f"parent {label} SHA-256 mismatch: {path}")
    return path


def _same_number(left: Any, right: float) -> bool:
    try:
        value = float(left)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(
        value, float(right), rel_tol=0.0, abs_tol=1e-12
    )


def _manifest_input_sha256(manifest: Mapping[str, Any]) -> str:
    bound = copy.deepcopy(dict(manifest))
    bound.pop("campaign_input_sha256", None)
    return canonical_sha256(bound)


def _authenticate_manifest(parent: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = parent / "campaign_manifest.json"
    manifest = _load_object(manifest_path, "parent campaign manifest")
    if manifest.get("campaign_manifest_schema_version") != 1:
        raise RuntimeError("unsupported parent campaign manifest schema")
    if manifest.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise RuntimeError("recovery parent is not the schema-v11 experiment")
    expected_input = str(manifest.get("campaign_input_sha256", ""))
    if _manifest_input_sha256(manifest) != expected_input:
        raise RuntimeError("parent campaign manifest semantic SHA-256 mismatch")

    config_path = _bound_file(
        manifest.get("config_path"), manifest.get("config_sha256"), "config"
    )
    _bound_file(manifest.get("model_path"), manifest.get("model_sha256"), "model")
    _bound_file(
        manifest.get("uv_lock_path"), manifest.get("uv_lock_sha256"), "uv.lock"
    )
    _bound_file(
        manifest.get("actual_qpos_source_manifest_path"),
        manifest.get("actual_qpos_source_manifest_sha256"),
        "actual-qpos source manifest",
    )

    source_values = manifest.get("source_files")
    if not isinstance(source_values, list) or not source_values:
        raise RuntimeError("parent manifest has no source file list")
    source_paths: list[Path] = []
    for value in source_values:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise RuntimeError(f"parent source file is missing: {path}")
        source_paths.append(path)
    if aggregate_source_sha256(source_paths) != str(manifest.get("source_sha256")):
        raise RuntimeError("parent campaign source SHA-256 mismatch")
    return manifest, config_path


def _authenticate_source_bundle(
    parent: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    bundle_path = parent / "sources" / "source_bundle.json"
    bundle = _load_object(bundle_path, "parent source bundle")
    if (
        bundle.get("actual_contact_source_bundle_schema_version") != 2
        or bundle.get("complete") is not True
        or bundle.get("experiment_id") != manifest.get("experiment_id")
    ):
        raise RuntimeError("parent source bundle is incomplete or incompatible")
    sources = bundle.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("parent source bundle has no authenticated sources")
    source_bundle_sha256 = canonical_sha256(sources)
    if source_bundle_sha256 != str(bundle.get("source_bundle_sha256")):
        raise RuntimeError("parent source bundle semantic SHA-256 mismatch")
    if int(bundle.get("source_count", -1)) != len(sources):
        raise RuntimeError("parent source bundle count mismatch")
    if bundle.get("source_manifest_sha256") != manifest.get(
        "actual_qpos_source_manifest_sha256"
    ):
        raise RuntimeError("parent source bundle manifest digest mismatch")
    return bundle, source_bundle_sha256


def _require_zero_result(parent: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = _load_object(
        parent / "campaign_result_target_1.json", "parent campaign result"
    )
    if (
        result.get("actual_contact_grasp_pose_campaign_result_schema_version") != 1
        or result.get("complete") is not True
        or result.get("experiment_id") != manifest.get("experiment_id")
        or result.get("campaign_input_sha256")
        != manifest.get("campaign_input_sha256")
    ):
        raise RuntimeError("parent campaign result is incomplete or incompatible")
    for field in ("static_pass_count", "grasp_success_count", "full_success_count"):
        if result.get(field) != 0:
            raise RuntimeError(
                f"recovery requires a zero-success parent; {field} is nonzero"
            )
    if result.get("target_reached") is not False:
        raise RuntimeError("recovery requires an exhausted parent target")
    return result


def _authenticate_stage_report(
    path: Path,
    *,
    stage: str,
    campaign_input_sha256: str,
    source_bundle_sha256: str,
    cell_count: int,
    expected_start_index: int,
    expected_samples_per_cell: int,
) -> dict[str, Any]:
    report = _load_object(path, f"{stage} static report")
    if (
        report.get("actual_contact_static_stage_schema_version")
        != _STATIC_STAGE_SCHEMA_VERSION
        or report.get("complete") is not True
        or report.get("stage") != stage
        or report.get("campaign_input_sha256") != campaign_input_sha256
        or report.get("source_bundle_sha256") != source_bundle_sha256
        or report.get("cell_count") != cell_count
        or report.get("sample_start_index") != expected_start_index
        or report.get("samples_per_cell") != expected_samples_per_cell
    ):
        raise RuntimeError(f"{stage} static report contract mismatch")
    for field in (
        "static_pass_observation_count",
        "prior_pool_static_pass_count",
        "retained_static_pass_count",
    ):
        if report.get(field) != 0:
            raise RuntimeError(f"{stage} static report contains a hard pass")
    return report


def _authenticate_refinement(
    path: Path, *, stage: str, expected_selected_count: int | None = None
) -> dict[str, Any]:
    report = _load_object(path, f"{stage} orientation-aware refinement")
    if (
        report.get("relative_wrist_refinement_schema_version")
        != _REFINEMENT_SCHEMA_VERSION
        or report.get("complete") is not True
        or report.get("stage") != stage
        or report.get("method") != "orientation_aware_actual_contact_dls_13_variables"
        or report.get("variable_count") != 13
        or report.get("selected_static_pass_count") != 0
    ):
        raise RuntimeError(f"{stage} orientation-aware refinement is incompatible")
    refined = report.get("refined_candidates")
    if not isinstance(refined, list) or any(
        bool(value.get("static_pass")) for value in refined if isinstance(value, Mapping)
    ):
        raise RuntimeError(f"{stage} refinement contains a static hard pass")
    if any(not isinstance(value, Mapping) for value in refined):
        raise RuntimeError(f"{stage} refinement candidate is malformed")
    if expected_selected_count is not None and report.get(
        "selected_source_count"
    ) != expected_selected_count:
        raise RuntimeError(f"{stage} refinement selected-count mismatch")
    return report


def _authenticate_candidate(
    raw: Mapping[str, Any], *, cell: Any, experiment_id: str
) -> dict[str, Any]:
    record = copy.deepcopy(dict(raw))
    config = record.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("recovery candidate has no config")
    if config.get("experiment_id") != experiment_id or config.get("schema_version") != 11:
        raise RuntimeError("recovery candidate belongs to another experiment")
    if canonical_sha256(config) != str(record.get("candidate_sha256")):
        raise RuntimeError("recovery candidate config SHA-256 mismatch")
    if grasp_pose_id(config) != str(record.get("grasp_pose_id")):
        raise RuntimeError("recovery candidate grasp_pose_id mismatch")
    if controller_id(config) != str(record.get("controller_id")):
        raise RuntimeError("recovery candidate controller_id mismatch")
    if bool(record.get("static_pass")):
        raise RuntimeError("zero-success parent cell contains a static hard pass")

    for key, expected in cell.as_dict().items():
        actual = record.get(key)
        if isinstance(expected, float):
            matches = _same_number(actual, expected)
        else:
            matches = actual == expected
        if not matches:
            raise RuntimeError(f"recovery candidate {key} differs from its cell")
    if not _same_number(config.get("cube", {}).get("edge_m"), cell.edge_m):
        raise RuntimeError("recovery candidate cube edge differs from its cell")
    thumb = (
        config.get("grasp_pose", {})
        .get("nominal_joint_qpos_rad", {})
        .get("left_hand_thumb_bend_joint_actuator")
    )
    if not _same_number(thumb, cell.thumb_actual_center_rad):
        raise RuntimeError("recovery candidate actual thumb differs from its cell")
    relative = config.get("candidate_metadata", {}).get(
        "relative_wrist_pose_search", {}
    )
    if not _same_number(relative.get("clockwise_orbit_deg"), cell.clockwise_orbit_deg):
        raise RuntimeError("recovery candidate orbit differs from its cell")
    metadata = config.get("candidate_metadata", {})
    if (
        metadata.get("candidate_id") != record.get("candidate_id")
        or metadata.get("cell_index") != cell.cell_index
        or metadata.get("cell_id") != cell.cell_id
    ):
        raise RuntimeError("recovery candidate metadata identity mismatch")
    return record


@dataclass(frozen=True, slots=True)
class RelativeWristRecoveryInput:
    """Authenticated immutable input to a separate recovery campaign."""

    parent_campaign: Path
    experiment_id: str
    campaign_input_sha256: str
    source_bundle_sha256: str
    candidates: tuple[dict[str, Any], ...]
    snapshot: dict[str, Any]

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def candidate_by_id(self) -> Mapping[int, dict[str, Any]]:
        return MappingProxyType(
            {int(value["candidate_id"]): copy.deepcopy(value) for value in self.candidates}
        )


def authenticate_relative_wrist_recovery_parent(
    parent_campaign: str | Path,
) -> RelativeWristRecoveryInput:
    """Read and authenticate one completed zero-success schema-v11 campaign."""

    parent = Path(parent_campaign).expanduser().resolve()
    if not parent.is_dir():
        raise RuntimeError(f"recovery parent is not a directory: {parent}")
    manifest, config_path = _authenticate_manifest(parent)
    ledger = validate_stage_ledger(parent)
    stages = ledger.get("stages", {})
    missing_stages = [name for name in _REQUIRED_STAGES if name not in stages]
    if missing_stages or any(stages[name].get("complete") is not True for name in _REQUIRED_STAGES if name in stages):
        raise RuntimeError(
            "recovery parent is missing committed stages: " + ", ".join(missing_stages)
        )
    bundle, source_bundle_sha256 = _authenticate_source_bundle(parent, manifest)
    result = _require_zero_result(parent, manifest)

    config = _load_object(config_path, "parent config")
    definition = resolve_experiment(config)
    if (
        definition.experiment_id != EXPECTED_EXPERIMENT_ID
        or definition.relative_wrist_pose_search is None
        or definition.actual_contact_grasp_pose_campaign is None
    ):
        raise RuntimeError("parent config has no schema-v11 relative-wrist campaign")
    campaign = definition.actual_contact_grasp_pose_campaign
    relative = definition.relative_wrist_pose_search
    cells = actual_contact_search_cells(config)
    if len(cells) != 700:
        raise RuntimeError("schema-v11 recovery requires all 700 search strata")

    campaign_input_sha256 = str(manifest["campaign_input_sha256"])
    quick_report_path = parent / "static" / "quick" / "report.json"
    expanded_report_path = parent / "static" / "expanded" / "report.json"
    quick_report = _authenticate_stage_report(
        quick_report_path,
        stage="quick",
        campaign_input_sha256=campaign_input_sha256,
        source_bundle_sha256=source_bundle_sha256,
        cell_count=len(cells),
        expected_start_index=0,
        expected_samples_per_cell=relative.static_samples_per_stratum,
    )
    expanded_report = _authenticate_stage_report(
        expanded_report_path,
        stage="expanded",
        campaign_input_sha256=campaign_input_sha256,
        source_bundle_sha256=source_bundle_sha256,
        cell_count=len(cells),
        expected_start_index=relative.static_samples_per_stratum,
        expected_samples_per_cell=relative.expansion_samples_per_stratum,
    )
    refinement_paths = {
        stage: parent
        / "static"
        / stage
        / "relative_wrist_orientation_refinement.json"
        for stage in ("quick", "expanded")
    }
    refinements = {
        stage: _authenticate_refinement(
            path,
            stage=stage,
            expected_selected_count=(
                len(campaign.edges_m) * relative.retained_poses_per_edge
            ),
        )
        for stage, path in refinement_paths.items()
    }

    records_by_id: dict[int, dict[str, Any]] = {}
    origins_by_id: dict[int, list[dict[str, Any]]] = {}
    quick_pool_by_cell: dict[int, tuple[dict[str, Any], ...]] = {}
    cell_snapshots: list[dict[str, Any]] = []
    reports = {"quick": quick_report, "expanded": expanded_report}
    for stage in ("quick", "expanded"):
        report = reports[stage]
        start_index = int(report["sample_start_index"])
        sample_count = int(report["samples_per_cell"])
        retain_count = int(report["pool_retain_per_cell"])
        for cell in cells:
            path = parent / "static" / stage / f"cell_{cell.cell_index:02d}.json"
            payload = _load_object(path, f"{stage} cell {cell.cell_index}")
            if (
                payload.get("actual_contact_static_cell_schema_version")
                != _STATIC_CELL_SCHEMA_VERSION
                or payload.get("complete") is not True
                or payload.get("stage") != stage
                or payload.get("start_index") != start_index
                or payload.get("evaluated_count") != sample_count
                or payload.get("static_pass_count") != 0
            ):
                raise RuntimeError(f"{stage} cell {cell.cell_index} is incomplete")
            for key, expected in cell.as_dict().items():
                actual = payload.get(key)
                matches = (
                    _same_number(actual, expected)
                    if isinstance(expected, float)
                    else actual == expected
                )
                if not matches:
                    raise RuntimeError(f"{stage} cell {cell.cell_index} identity mismatch")
            prior_values = quick_pool_by_cell.get(cell.cell_index, ()) if stage == "expanded" else ()
            prior_sha256 = (
                canonical_sha256(
                    [
                        {
                            "candidate_id": int(value["candidate_id"]),
                            "candidate_sha256": str(value["candidate_sha256"]),
                            "static_pass": bool(value.get("static_pass")),
                        }
                        for value in prior_values
                    ]
                )
                if prior_values
                else None
            )
            expected_input = _static_cell_input_sha256(
                campaign_input_sha256=campaign_input_sha256,
                source_bundle_sha256=source_bundle_sha256,
                stage=stage,
                cell=cell,
                start_index=start_index,
                sample_count=sample_count,
                retain_count=retain_count,
                seed=int(manifest["seed"]),
                prior_pool_sha256=prior_sha256,
            )
            if payload.get("cell_input_sha256") != expected_input:
                raise RuntimeError(f"{stage} cell {cell.cell_index} input SHA-256 mismatch")
            raw_pool = payload.get("new_pool")
            if not isinstance(raw_pool, list) or len(raw_pool) > retain_count:
                raise RuntimeError(f"{stage} cell {cell.cell_index} has an invalid pool")
            pool = tuple(
                _authenticate_candidate(
                    value, cell=cell, experiment_id=definition.experiment_id
                )
                for value in raw_pool
                if isinstance(value, Mapping)
            )
            if len(pool) != len(raw_pool):
                raise RuntimeError(f"{stage} cell {cell.cell_index} has a malformed candidate")
            if stage == "quick":
                quick_pool_by_cell[cell.cell_index] = pool
            candidate_evidence = []
            for candidate in pool:
                identifier = int(candidate["candidate_id"])
                evidence_sha256 = canonical_sha256(candidate)
                previous = records_by_id.get(identifier)
                if previous is not None and canonical_sha256(previous) != evidence_sha256:
                    raise RuntimeError("candidate ID collision across parent static cells")
                records_by_id.setdefault(identifier, candidate)
                origins_by_id.setdefault(identifier, []).append(
                    {"stage": stage, "cell_index": cell.cell_index}
                )
                candidate_evidence.append(
                    {
                        "candidate_id": identifier,
                        "candidate_sha256": candidate["candidate_sha256"],
                        "evidence_sha256": evidence_sha256,
                    }
                )
            cell_snapshots.append(
                {
                    "stage": stage,
                    "cell_index": cell.cell_index,
                    "cell_id": cell.cell_id,
                    "relative_path": str(path.relative_to(parent)),
                    "file_sha256": file_sha256(path),
                    "cell_input_sha256": expected_input,
                    "candidate_count": len(pool),
                    "candidates": candidate_evidence,
                }
            )

    candidates = tuple(copy.deepcopy(records_by_id[key]) for key in sorted(records_by_id))
    if len(candidates) > len(cells) * 2 * campaign.full_static_retain_per_cell:
        raise RuntimeError("parent recovery pool exceeds its declared 4,200 candidate bound")
    candidate_snapshots = [
        {
            "candidate_id": int(value["candidate_id"]),
            "candidate_sha256": str(value["candidate_sha256"]),
            "grasp_pose_id": str(value["grasp_pose_id"]),
            "controller_id": str(value["controller_id"]),
            "evidence_sha256": canonical_sha256(value),
            "edge_m": float(value["edge_m"]),
            "thumb_actual_center_rad": float(value["thumb_actual_center_rad"]),
            "clockwise_orbit_deg": float(value["clockwise_orbit_deg"]),
            "origins": origins_by_id[int(value["candidate_id"])],
        }
        for value in candidates
    ]
    key_paths = {
        "campaign_manifest": parent / "campaign_manifest.json",
        "stage_ledger": parent / "stage_ledger.json",
        "source_bundle": parent / "sources" / "source_bundle.json",
        "campaign_result": parent / "campaign_result_target_1.json",
        "quick_static_report": quick_report_path,
        "expanded_static_report": expanded_report_path,
        "quick_refinement": refinement_paths["quick"],
        "expanded_refinement": refinement_paths["expanded"],
    }
    snapshot_body = {
        "recovery_parent_snapshot_schema_version": (
            RECOVERY_PARENT_SNAPSHOT_SCHEMA_VERSION
        ),
        "complete": True,
        "read_only_parent": True,
        "parent_campaign": str(parent),
        "experiment_id": definition.experiment_id,
        "campaign_input_sha256": campaign_input_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "immutable_inputs": {
            name: {
                "path": str(manifest[f"{name}_path"]),
                "sha256": str(manifest[f"{name}_sha256"]),
            }
            for name in ("config", "model", "uv_lock")
        }
        | {
            "actual_qpos_source_manifest": {
                "path": str(manifest["actual_qpos_source_manifest_path"]),
                "sha256": str(
                    manifest["actual_qpos_source_manifest_sha256"]
                ),
            },
            "source_files": {
                "paths": copy.deepcopy(manifest["source_files"]),
                "aggregate_sha256": str(manifest["source_sha256"]),
            },
        },
        "parent_result_counts": {
            name: int(result[name])
            for name in ("static_pass_count", "grasp_success_count", "full_success_count")
        },
        "parent_files": {
            name: {
                "path": str(path.relative_to(parent)),
                "sha256": file_sha256(path),
            }
            for name, path in key_paths.items()
        },
        "committed_stage_sha256": {
            name: canonical_sha256(stages[name]) for name in _REQUIRED_STAGES
        },
        "cell_count_per_stage": len(cells),
        "authenticated_cell_count": len(cell_snapshots),
        "candidate_count": len(candidates),
        "cells": cell_snapshots,
        "candidates": candidate_snapshots,
        "refinement_sha256": {
            stage: canonical_sha256(refinements[stage])
            for stage in ("quick", "expanded")
        },
        "source_count": int(bundle["source_count"]),
    }
    snapshot = {
        **snapshot_body,
        "snapshot_sha256": canonical_sha256(snapshot_body),
    }
    return RelativeWristRecoveryInput(
        parent_campaign=parent,
        experiment_id=definition.experiment_id,
        campaign_input_sha256=campaign_input_sha256,
        source_bundle_sha256=source_bundle_sha256,
        candidates=candidates,
        snapshot=snapshot,
    )


# Short name for recovery runners; keep the explicit public name for tests and
# diagnostics where the trust boundary should be obvious at the call site.
authenticate_recovery_parent = authenticate_relative_wrist_recovery_parent


__all__ = [
    "EXPECTED_EXPERIMENT_ID",
    "RECOVERY_PARENT_SNAPSHOT_SCHEMA_VERSION",
    "RelativeWristRecoveryInput",
    "authenticate_recovery_parent",
    "authenticate_relative_wrist_recovery_parent",
]
