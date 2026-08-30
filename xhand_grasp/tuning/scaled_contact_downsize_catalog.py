"""Disk-safe Viewer catalog publication for the schema-v13 campaign.

The shared actual-contact catalog module is immutable evidence for earlier
experiments.  This module layers the wider, per-edge schema-v13 publication
semantics on top without changing that legacy implementation: representative
video members are delegated to the proven legacy publisher in groups of at
most three, while every other trace is materialized as a catalog-confined hard
link (with a cross-device copy fallback).
"""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..actual_contact_capability import resolve_actual_contact_definition
from ..actual_contact_grasp_pose_catalog import (
    TRAJECTORY_CATALOG_SCHEMA_VERSION,
    FinalSimulationRunner,
    FinalVideoProbe,
    _CatalogSource,
    _CompactedFailureCandidate,
    _load_source,
    export_actual_contact_grasp_pose_catalog,
    export_actual_contact_manipulation_catalog,
)
from ..actual_contact_selection import (
    ActualContactSelection,
    _coverage,
    _selection_limits,
    final_candidate_rank,
    final_candidate_rank_evidence,
    full_success,
    grasp_success,
)
from ..artifacts import file_sha256, write_json
from .scaled_contact_downsize_ranking import (
    v13_contact_slip_rank_evidence,
    v13_manipulation_candidate_rank,
)


def _publication_rank(
    record: Mapping[str, Any],
    *,
    kind: Literal["grasp_pose", "manipulation"],
) -> tuple[Any, ...]:
    """Use the sealed grasp rank or schema-v13's slip-aware lift rank."""

    if kind == "grasp_pose":
        return final_candidate_rank(record)
    config = record.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("schema-v13 manipulation publication requires config")
    return v13_manipulation_candidate_rank(record, config)


def _source_edge_m(source: _CatalogSource) -> float:
    cube = source.config.get("cube")
    raw = cube.get("edge_m") if isinstance(cube, Mapping) else None
    try:
        edge = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("schema-v13 catalog source has no cube edge") from exc
    if not np.isfinite(edge) or edge <= 0.0:
        raise ValueError("schema-v13 catalog source has an invalid cube edge")
    return edge


def _video_roles(
    successes: Sequence[_CatalogSource],
) -> dict[str, tuple[str, ...]]:
    if not successes:
        return {}
    selected: dict[str, list[str]] = {}

    def add(source: _CatalogSource, role: str) -> None:
        selected.setdefault(source.candidate_id, []).append(role)

    add(successes[0], "best_nominal")
    smallest_edge = min(_source_edge_m(source) for source in successes)
    smallest = next(
        source
        for source in successes
        if np.isclose(_source_edge_m(source), smallest_edge, rtol=0.0, atol=1e-12)
    )
    add(smallest, "smallest_pass")
    add(successes[-1], "hardest_selected_pass")
    return {key: tuple(value) for key, value in selected.items()}


def _materialize_trace(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
        return "copy"
    return "hardlink"


def _load_sources(
    candidates: Iterable[Mapping[str, Any]],
    *,
    kind: Literal["grasp_pose", "manipulation"],
) -> tuple[list[_CatalogSource], dict[str, dict[str, Any]]]:
    sources: list[_CatalogSource] = []
    raw_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(candidates):
        materialized = copy.deepcopy(dict(raw))
        try:
            source = _load_source(materialized, index, kind=kind)
        except _CompactedFailureCandidate:
            continue
        if source.candidate_id in raw_by_id:
            raise ValueError("catalog candidates must have unique candidate IDs")
        sources.append(source)
        raw_by_id[source.candidate_id] = materialized
    if not sources:
        raise ValueError("at least one trajectory candidate is required")
    experiment_ids = {source.experiment_id for source in sources}
    if len(experiment_ids) != 1:
        raise ValueError("catalog candidates must belong to one experiment")
    definition = resolve_actual_contact_definition(
        sources[0].config, context="scaled-downsize catalog publication"
    )
    if definition.scaled_contact_downsize_campaign is None:
        raise ValueError("scaled-downsize publisher requires schema-v13 capability")
    sources.sort(key=lambda value: (value.discovery_index, value.candidate_id))
    return sources, raw_by_id


def _selection_record(source: _CatalogSource) -> dict[str, Any]:
    return {
        "candidate_id": source.candidate_id,
        "discovery_index": source.discovery_index,
        "config": source.config,
        "summary": source.summary,
        "parameter_override": source.parameter_override,
    }


def select_scaled_contact_downsize_candidates(
    records: Iterable[Mapping[str, Any]],
    *,
    kind: Literal["grasp_pose", "manipulation"],
    selected_count: int,
) -> ActualContactSelection:
    """Schema-v13 selection with the registered 29x3 publication capacity."""

    if kind not in ("grasp_pose", "manipulation"):
        raise ValueError("kind must be grasp_pose or manipulation")
    materialized = [copy.deepcopy(dict(value)) for value in records]
    if not materialized:
        raise ValueError("at least one schema-v13 selection record is required")
    capacities: set[int] = set()
    for value in materialized:
        config = value.get("config")
        if not isinstance(config, Mapping) or config.get("schema_version") != 13:
            raise ValueError("scaled-downsize selection accepts only schema-v13")
        registered = config.get("scaled_contact_downsize_campaign")
        budget = registered.get("budget") if isinstance(registered, Mapping) else None
        edges = registered.get("edges_m") if isinstance(registered, Mapping) else None
        per_edge = budget.get("selected_grasps_per_edge") if isinstance(budget, Mapping) else None
        capacity = budget.get("selected_grasp_count") if isinstance(budget, Mapping) else None
        if (
            not isinstance(edges, Sequence)
            or isinstance(edges, (str, bytes))
            or not isinstance(per_edge, int)
            or isinstance(per_edge, bool)
            or not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity != len(edges) * per_edge
        ):
            raise ValueError("schema-v13 selection capacity is malformed")
        capacities.add(capacity)
    if len(capacities) != 1:
        raise ValueError("schema-v13 records disagree on selection capacity")
    capacity = capacities.pop()
    if not 1 <= int(selected_count) <= capacity:
        raise ValueError(f"selected_count must be between 1 and {capacity}")
    predicate = grasp_success if kind == "grasp_pose" else full_success
    eligible = [
        value
        for value in materialized
        if predicate(value) and not bool(value.get("parameter_override", False))
    ]
    eligible.sort(
        key=lambda value: (
            int(value.get("discovery_index", 2**63 - 1)),
            str(value.get("candidate_id")),
        )
    )
    best_first = eligible[0] if eligible else None
    ranked = sorted(
        eligible,
        key=lambda value: _publication_rank(value, kind=kind),
    )
    selected: list[dict[str, Any]] = []
    if best_first is not None:
        selected.append(best_first)
    used = {str(value.get("candidate_id")) for value in selected}
    for value in ranked:
        identity = str(value.get("candidate_id"))
        if identity in used:
            continue
        selected.append(value)
        used.add(identity)
        if len(selected) >= int(selected_count):
            break
    selected = sorted(
        selected[: int(selected_count)],
        key=lambda value: _publication_rank(value, kind=kind),
    )
    minimum_edges, minimum_bands = _selection_limits(
        materialized, int(selected_count)
    )
    edges, bands = _coverage(selected)
    diversity_satisfied = bool(
        int(selected_count) != 5
        or (len(edges) >= minimum_edges and len(bands) >= minimum_bands)
    )
    metadata = {
        "selection_schema_version": 1,
        "kind": kind,
        "requested_success_count": int(selected_count),
        "eligible_success_count": len(eligible),
        "selected_success_count": len(selected),
        "target_reached": bool(
            len(selected) == int(selected_count) and diversity_satisfied
        ),
        "best_first_candidate_id": (
            None if best_first is None else str(best_first.get("candidate_id"))
        ),
        "best_first_discovery_index": (
            None
            if best_first is None
            else int(best_first.get("discovery_index", 2**63 - 1))
        ),
        "selected_candidate_ids_in_final_rank_order": [
            str(value.get("candidate_id")) for value in selected
        ],
        "ranking_policy": (
            "legacy_actual_contact_final_candidate_rank"
            if kind == "grasp_pose"
            else "schema_v13_hard_then_soft_contact_slip"
        ),
        "rank_order": (
            [
                "full_hard_pass",
                "actual_thumb_distance_from_1p50_rad",
                "grasp_stability_margin",
                "pose_preservation_margin",
                "closure_alignment_margin",
                "vertical_motion_margin",
                "smoothness_margin",
                "contact_force_balance",
                "actuator_saturation",
                "candidate_id",
            ]
            if kind == "grasp_pose"
            else [
                "full_hard_pass",
                "grasp_hard_pass",
                "lift_margin",
                "topology_margin",
                "smoothness_margin",
                "grasp_contact_slip_baseline_valid",
                "operation_contact_slip_valid_duty",
                "operation_contact_slip_p95",
                "operation_contact_slip_max",
                "failed_check_count",
                "candidate_id",
            ]
        ),
        "diversity": {
            "required_distinct_edge_count": minimum_edges,
            "required_actual_thumb_band_count": minimum_bands,
            "selected_distinct_edge_count": len(edges),
            "selected_actual_thumb_band_count": len(bands),
            "selected_edges_m": sorted(edges),
            "selected_actual_thumb_bands_rad": sorted(bands),
            "edge_deficit": max(0, minimum_edges - len(edges)),
            "actual_thumb_band_deficit": max(0, minimum_bands - len(bands)),
            "satisfied": diversity_satisfied,
        },
    }
    return ActualContactSelection(
        selected=tuple(copy.deepcopy(value) for value in selected),
        eligible=tuple(copy.deepcopy(value) for value in eligible),
        metadata=metadata,
    )


def _select_sources(
    sources: Sequence[_CatalogSource],
    *,
    kind: Literal["grasp_pose", "manipulation"],
    selected_count: int,
) -> tuple[list[_CatalogSource], dict[str, Any]]:
    by_id = {source.candidate_id: source for source in sources}
    selection = select_scaled_contact_downsize_candidates(
        (_selection_record(source) for source in sources),
        kind=kind,
        selected_count=selected_count,
    )
    return (
        [by_id[str(value["candidate_id"])] for value in selection.selected],
        copy.deepcopy(selection.metadata),
    )


def _publish_catalog(
    candidates: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    kind: Literal["grasp_pose", "manipulation"],
    selected_count: int,
    simulation_runner: FinalSimulationRunner | None = None,
    video_probe: FinalVideoProbe | None = None,
) -> dict[str, Any]:
    sources, raw_by_id = _load_sources(candidates, kind=kind)
    definition = resolve_actual_contact_definition(
        sources[0].config, context="scaled-downsize catalog publication"
    )
    downsize = definition.scaled_contact_downsize_campaign
    assert downsize is not None
    if not 1 <= int(selected_count) <= int(downsize.selected_grasp_count):
        raise ValueError(
            "selected_count must be between 1 and "
            f"{downsize.selected_grasp_count}"
        )
    successes, selection = _select_sources(
        sources,
        kind=kind,
        selected_count=int(selected_count),
    )
    selected_ids = {source.candidate_id for source in successes}
    diagnostic_pool = sorted(
        (source for source in sources if source.candidate_id not in selected_ids),
        key=lambda source: _publication_rank(
            {
                "candidate_id": source.candidate_id,
                "discovery_index": source.discovery_index,
                "config": source.config,
                "summary": source.summary,
                "parameter_override": source.parameter_override,
            },
            kind=kind,
        ),
    )
    selected: list[tuple[str, _CatalogSource]] = [
        *(('success', source) for source in successes)
    ]
    if diagnostic_pool:
        selected.append(("diagnostic", diagnostic_pool[0]))
    roles_by_id = _video_roles(successes)

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    legacy_exporter = (
        export_actual_contact_grasp_pose_catalog
        if kind == "grasp_pose"
        else export_actual_contact_manipulation_catalog
    )
    validation_labels = definition.actual_contact_grasp_pose_campaign.validation_labels
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.staging."
    ) as staging_name:
        staging = Path(staging_name)
        video_root = staging / "_representative_video_catalog"
        video_entries: dict[str, dict[str, Any]] = {}
        video_ids = tuple(roles_by_id)
        if video_ids:
            video_payload = legacy_exporter(
                (raw_by_id[value] for value in video_ids),
                video_root,
                selected_count=len(video_ids),
                simulation_runner=simulation_runner,
                video_probe=video_probe,
            )
            video_entries = {
                str(entry["candidate_id"]): copy.deepcopy(dict(entry))
                for entry in video_payload["trajectories"]
                if entry.get("classification") == "success"
            }
            if set(video_entries) != set(video_ids):
                raise RuntimeError("representative video publisher omitted a source")

        entries: list[dict[str, Any]] = []
        aliases: dict[str, str] = {}
        best_first_id = selection["best_first_candidate_id"]
        success_index = 0
        for index, (classification, source) in enumerate(selected, start=1):
            trajectory_id = f"{kind}_{index:02d}_{source.candidate_id}"
            member = staging / trajectory_id
            roles = roles_by_id.get(source.candidate_id, ())
            if source.candidate_id in video_entries:
                video_entry = video_entries[source.candidate_id]
                legacy_member = (
                    video_root
                    / Path(video_entry["artifacts"]["resolved_config"]).parent
                )
                legacy_member.rename(member)
                entry = copy.deepcopy(video_entry)
                entry["trajectory_id"] = trajectory_id
                entry["label"] = trajectory_id
                artifacts = entry["artifacts"]
                for name, filename in (
                    ("resolved_config", "resolved_config.json"),
                    ("result", "result.json"),
                    ("trace", "trace.npz"),
                    ("video", "trajectory.mp4"),
                ):
                    artifacts[name] = (
                        f"{trajectory_id}/{filename}"
                        if (member / filename).is_file()
                        else None
                    )
                artifacts["trace_materialization"] = "deterministic_video_rerun"
            else:
                member.mkdir()
                config_path = member / "resolved_config.json"
                result_path = member / "result.json"
                trace_path = member / "trace.npz"
                shutil.copy2(source.config_path, config_path)
                shutil.copy2(source.result_path, result_path)
                trace_materialization = _materialize_trace(source.trace_path, trace_path)
                hashes = {
                    "resolved_config": file_sha256(config_path),
                    "result": file_sha256(result_path),
                    "trace": file_sha256(trace_path),
                }
                entry = {
                    "trajectory_id": trajectory_id,
                    "label": trajectory_id,
                    "classification": (
                        "diagnostic_override"
                        if source.parameter_override
                        else classification
                    ),
                    "grasp_success": source.grasp_success,
                    "full_success": source.full_success,
                    "candidate_id": source.candidate_id,
                    "discovery_index": source.discovery_index,
                    "final_rank_evidence": final_candidate_rank_evidence(
                        {
                            "candidate_id": source.candidate_id,
                            "discovery_index": source.discovery_index,
                            "config": source.config,
                            "summary": source.summary,
                            "parameter_override": source.parameter_override,
                        }
                    ),
                    "stage_status": copy.deepcopy(
                        source.summary.get("stage_status", {})
                    ),
                    "failed_checks": copy.deepcopy(
                        source.summary.get("failed_checks", [])
                    ),
                    "final_video_required": False,
                    "final_video_verified": False,
                    "final_video_evidence": None,
                    "artifacts": {
                        "resolved_config": f"{trajectory_id}/resolved_config.json",
                        "result": f"{trajectory_id}/result.json",
                        "trace": f"{trajectory_id}/trace.npz",
                        "video": None,
                        "sha256": hashes,
                        "trace_materialization": trace_materialization,
                    },
                }
            if kind == "manipulation":
                entry["publication_ranking_policy"] = (
                    "schema_v13_hard_then_soft_contact_slip"
                )
                entry["v13_contact_slip_rank_evidence"] = (
                    v13_contact_slip_rank_evidence(
                        {
                            "candidate_id": source.candidate_id,
                            "discovery_index": source.discovery_index,
                            "config": source.config,
                            "summary": source.summary,
                            "parameter_override": source.parameter_override,
                        }
                    )
                )
            entry_aliases: list[str] = []
            if classification == "success":
                success_index += 1
                numbered = f"{kind}_{success_index}"
                entry_aliases.append(numbered)
                aliases[numbered] = trajectory_id
                if success_index == 1:
                    entry_aliases.append("best_nominal")
                    aliases["best_nominal"] = trajectory_id
                if source.candidate_id == best_first_id:
                    entry_aliases.append("best_first")
                    aliases["best_first"] = trajectory_id
                if validation_labels is not None:
                    entry["validation_label"] = validation_labels[
                        "grasp" if kind == "grasp_pose" else "manipulation"
                    ]
            elif not successes:
                entry_aliases.append("best_attempt")
                aliases["best_attempt"] = trajectory_id
            entry["aliases"] = entry_aliases
            entry["final_video_roles"] = list(roles)
            entries.append(entry)

        if video_root.exists():
            shutil.rmtree(video_root)
        catalog = {
            "actual_contact_grasp_pose_catalog_schema_version": (
                TRAJECTORY_CATALOG_SCHEMA_VERSION
            ),
            "trajectory_catalog_schema_version": 1,
            "experiment_id": definition.experiment_id,
            "catalog_kind": kind,
            "complete": True,
            "requested_success_count": int(selected_count),
            "success_count": len(successes),
            "eligible_success_count": selection["eligible_success_count"],
            "target_reached": selection["target_reached"],
            "selection": copy.deepcopy(selection),
            "diversity": copy.deepcopy(selection["diversity"]),
            "production_trajectory_video_policy": (
                "selective_best_smallest_hardest_full_reset_rerun_"
                "ffprobe_and_full_decode"
            ),
            "artifact_materialization_policy": (
                "catalog_confined_hardlink_trace_with_copy_fallback"
            ),
            "aliases": aliases,
            "trajectories": entries,
        }
        if validation_labels is not None:
            catalog["validation_label"] = validation_labels[
                "grasp" if kind == "grasp_pose" else "manipulation"
            ]
        write_json(staging / "catalog.json", catalog)
        staging.rename(output)
    return catalog


def export_scaled_contact_downsize_grasp_catalog(
    candidates: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    selected_count: int,
    simulation_runner: FinalSimulationRunner | None = None,
    video_probe: FinalVideoProbe | None = None,
) -> dict[str, Any]:
    return _publish_catalog(
        candidates,
        output_dir,
        kind="grasp_pose",
        selected_count=selected_count,
        simulation_runner=simulation_runner,
        video_probe=video_probe,
    )


def export_scaled_contact_downsize_manipulation_catalog(
    candidates: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    selected_count: int,
    simulation_runner: FinalSimulationRunner | None = None,
    video_probe: FinalVideoProbe | None = None,
) -> dict[str, Any]:
    return _publish_catalog(
        candidates,
        output_dir,
        kind="manipulation",
        selected_count=selected_count,
        simulation_runner=simulation_runner,
        video_probe=video_probe,
    )


__all__ = [
    "export_scaled_contact_downsize_grasp_catalog",
    "export_scaled_contact_downsize_manipulation_catalog",
    "select_scaled_contact_downsize_candidates",
]
