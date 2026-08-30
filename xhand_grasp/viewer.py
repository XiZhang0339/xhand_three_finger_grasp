"""Interactive MuJoCo Viewer for live simulations and legacy state replay.

The default path reruns the resolved configuration through the same
``SimulationSession`` used by batch evaluation.  A second independently
compiled model/data pair is display-only, so Viewer mouse perturbations and
``sync`` can never change the evaluated physics state.  ``--state-replay``
retains the historical recorded-state animation for diagnosis.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from .actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
)
from .actual_contact_capability import actual_contact_experiment_id
from .artifacts import file_sha256, resolved_run_config, write_json
from .config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, load_config, validate_config
from .contact_environment import (
    ContactEnvironmentSpec,
    apply_to_model as apply_contact_environment_to_model,
)
from .grasp_pose import canonical_sha256
from .joint_pair_geometry import (
    JointPairBinding,
    joint_pair_telemetry,
    resolve_joint_pair,
)
from .relative_wrist_pose import (
    rotation_matrix_to_rpy_degrees,
    transform_relative_wrist_pose,
)
from .scene import (
    ModelInfo,
    build_model,
    cube_vertical_half_extent_m,
    rpy_degrees_to_rotation_matrix,
    solve_press_depth_pose,
)
from .simulation import SimulationSession
from .trajectory import preflight_config


REQUIRED_TRACE_FIELDS = {
    "time": 1,
    "cube_pos": 2,
    "cube_quat": 2,
    "cube_velocity": 2,
    "ctrl": 2,
    "joint_qpos": 2,
    "joint_qvel": 2,
}
DEFAULT_V5_JOINT_MONITOR = "left_hand_thumb_bend_joint_actuator"
SUPPORTED_LIVE_PAUSE_EVENTS = frozenset({"grasp_lock"})
_V4_ALIGNED_EXPERIMENT_ID = (
    "left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift"
)
_V14_CONTACT_PRESERVING_EXPERIMENT_ID = (
    "left_opposed_face_palm_down_contact_preserving_planned_lift"
)
_V15_JOINT_PAIR_NEAR_ZERO_EXPERIMENT_ID = (
    "left_opposed_face_palm_down_joint_pair_near_zero_contact_preserving_"
    "planned_lift"
)
_V15_VIEWER_CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReplaySource:
    """Resolved files and human-readable identity for one replay."""

    config_path: Path
    trace_path: Path
    trajectory: str
    contact_environment_path: Path | None = None


@dataclass(frozen=True)
class ViewerSource:
    """Resolved live source; a reference trace is optional for direct configs."""

    config_path: Path
    trace_path: Path | None
    trajectory: str
    from_catalog: bool
    contact_environment_path: Path | None = None


@dataclass(frozen=True)
class MeasuredViewerData:
    """One authenticated measured grasp that can be selected for viewing.

    ``data_index`` and ``edge_rank`` are deliberately one-based so the value
    printed in the terminal is exactly the value accepted by the CLI.
    Candidate IDs remain integers throughout; production IDs are too large to
    safely round-trip through a floating-point parser.
    """

    data_index: int
    edge_rank: int
    candidate_id: int
    edge_mm: float
    actual_thumb_bend_rad: float
    measured_grasp_pose_success: bool
    pose_translation_mm: float | None
    pose_orientation_deg: float | None
    contact_height_spread_mm: float | None
    config_path: Path
    trace_path: Path
    result_path: Path


@dataclass
class PlaybackState:
    """Small mutable state shared with the Viewer's keyboard callback."""

    paused: bool = False
    looping: bool = False
    reset_requested: bool = False
    show_contact_alignment: bool = True
    show_coordinate_frames: bool = False
    show_joint_pair: bool = True
    event_pause_triggered: bool = False


@dataclass(frozen=True)
class LiveViewerResult:
    """Terminal status returned by the interactive live runner."""

    exit_code: int
    completed: bool
    summary: dict[str, Any] | None
    reference_trace_match: bool | None


@dataclass(frozen=True)
class JointMonitorBinding:
    """Name-resolved actuator/joint addresses used by Viewer diagnostics."""

    actuator_name: str
    actuator_id: int
    joint_id: int
    qpos_adr: int
    finger_index: int | None


def _catalog_member(catalog_path: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"catalog trajectory has no usable {field} path")
    base = catalog_path.parent.resolve()
    member = (base / value).resolve()
    try:
        member.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"catalog {field} escapes its directory: {value}") from exc
    if not member.is_file():
        raise FileNotFoundError(f"catalog {field} does not exist: {member}")
    return member


def _verify_catalog_digest(
    path: Path,
    digests: object,
    field: str,
    *,
    required: bool = False,
    required_catalog: str = "schema-v4 catalog",
) -> None:
    """Verify a catalog member when the versioned catalog supplies a digest."""

    if not isinstance(digests, Mapping) or field not in digests:
        if required:
            raise ValueError(
                f"{required_catalog} is missing required SHA-256 for {field}"
            )
        return
    expected = digests[field]
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"catalog has an invalid SHA-256 for {field}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected.lower():
        raise ValueError(f"catalog SHA-256 mismatch for {field}: {path}")


def _resolve_measured_report(path: str | Path) -> Path:
    """Accept a report, measured directory, dynamic directory, or campaign.

    The permissive path input keeps the command short while the resolved file
    remains unambiguous and authenticated by :func:`discover_measured_viewer_data`.
    """

    source = Path(path).expanduser().resolve()
    if source.is_file():
        return source
    if not source.exists():
        raise FileNotFoundError(f"measured data source does not exist: {source}")
    if not source.is_dir():
        raise ValueError(f"measured data source is not a file or directory: {source}")
    candidates = (
        source / "expanded_report.json",
        source / "measured" / "expanded_report.json",
        source / "dynamic" / "measured" / "expanded_report.json",
    )
    matches = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if len(matches) != 1:
        detail = "none found" if not matches else ", ".join(str(v) for v in matches)
        raise ValueError(
            "measured data source must resolve to exactly one expanded_report.json; "
            f"{detail}"
        )
    return matches[0]


def _measured_member(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"measured result has no usable {field} path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"measured {field} path is not confined: {value}")
    member = (base / relative).resolve()
    try:
        member.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"measured {field} escapes its candidate directory") from exc
    if not member.is_file():
        raise FileNotFoundError(f"measured {field} does not exist: {member}")
    return member


def _mapping_value(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested_mapping(value: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Mapping[str, Any] = value
    for key in keys:
        current = _mapping_value(current.get(key))
    return current


def _finite_optional(value: object, *, scale: float = 1.0) -> float | None:
    try:
        result = float(value) * scale
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def discover_measured_viewer_data(
    report_path: str | Path,
) -> tuple[MeasuredViewerData, ...]:
    """Load and authenticate every successful measured grasp in a report.

    The report is authoritative: the function never discovers candidates by
    an unconstrained filesystem glob.  Every result authenticates its semantic
    payload plus the exact resolved config and NPZ bytes before it can appear
    in the user-visible list.
    """

    report = _resolve_measured_report(report_path)
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid measured data report: {report}") from exc
    if not isinstance(payload, Mapping) or payload.get("complete") is not True:
        raise ValueError(f"measured data report is incomplete: {report}")
    raw_records = payload.get("candidate_records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"measured data report has no candidate_records: {report}")

    # expanded_report.json lives at dynamic/measured/.  Its artifact_directory
    # members (for example measured/candidate_...) are relative to dynamic/.
    dynamic_root = report.parent.parent.resolve()
    authenticated: list[tuple[float, tuple[Any, ...], dict[str, Any]]] = []
    seen_ids: set[int] = set()
    experiment_id: str | None = None
    from .tuning.actual_contact_grasp_pose_dynamic import dynamic_grasp_rank

    for report_record in raw_records:
        if not isinstance(report_record, Mapping):
            raise ValueError("measured candidate_records must contain mappings")
        raw_identifier = report_record.get("candidate_id")
        if isinstance(raw_identifier, bool):
            raise ValueError("measured candidate ID must be an integer")
        try:
            identifier = int(raw_identifier)
        except (TypeError, ValueError) as exc:
            raise ValueError("measured candidate ID must be an integer") from exc
        if identifier < 0 or identifier in seen_ids:
            raise ValueError(f"duplicate or invalid measured candidate ID: {identifier}")
        seen_ids.add(identifier)
        if report_record.get("measured_grasp_pose_success") is not True:
            raise ValueError(
                f"measured report candidate {identifier} is not a successful grasp"
            )

        raw_directory = report_record.get("artifact_directory")
        if not isinstance(raw_directory, str) or not raw_directory:
            raise ValueError(f"candidate {identifier} has no artifact_directory")
        relative_directory = Path(raw_directory)
        if relative_directory.is_absolute() or ".." in relative_directory.parts:
            raise ValueError(
                f"candidate {identifier} artifact_directory is not confined"
            )
        directory = (dynamic_root / relative_directory).resolve()
        try:
            directory.relative_to(dynamic_root)
        except ValueError as exc:
            raise ValueError(
                f"candidate {identifier} artifact_directory escapes dynamic root"
            ) from exc
        result_path = directory / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"measured result does not exist: {result_path}")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid measured result: {result_path}") from exc
        if not isinstance(result, Mapping):
            raise ValueError(f"measured result must be a mapping: {result_path}")
        try:
            authenticate_candidate_result_semantic_sha256(result, source=result_path)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        reported_result_digest = report_record.get("result_semantic_sha256")
        if (
            reported_result_digest is not None
            and reported_result_digest != result.get("result_semantic_sha256")
        ):
            raise ValueError(f"measured result semantic hash changed: {result_path}")
        if int(result.get("candidate_id", -1)) != identifier:
            raise ValueError(f"measured candidate ID changed: {result_path}")
        if (
            result.get("complete") is not True
            or result.get("measured_grasp_pose_success") is not True
            or result.get("grasp_success") is not True
        ):
            raise ValueError(f"candidate {identifier} is not a completed measured grasp")

        artifacts = _mapping_value(result.get("artifacts"))
        if artifacts.get("trace_retained") is not True:
            raise ValueError(f"candidate {identifier} has no retained trace")
        config_path = _measured_member(
            directory, artifacts.get("resolved_config"), "resolved_config"
        )
        trace_path = _measured_member(directory, artifacts.get("trace"), "trace")
        digests = _mapping_value(artifacts.get("sha256"))
        for field, member in (("resolved_config", config_path), ("trace", trace_path)):
            expected = digests.get(field)
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(
                    f"candidate {identifier} has no valid SHA-256 for {field}"
                )
            if file_sha256(member) != expected.lower():
                raise ValueError(f"measured SHA-256 mismatch for {field}: {member}")

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid measured config: {config_path}") from exc
        if not isinstance(config, Mapping):
            raise ValueError(f"measured config must be a mapping: {config_path}")
        config_digest = canonical_sha256(config)
        declared_digest = result.get("candidate_sha256")
        if (
            not isinstance(declared_digest, str)
            or declared_digest != config_digest
            or report_record.get("candidate_sha256") != declared_digest
        ):
            raise ValueError(f"measured candidate semantic hash changed: {result_path}")
        current_experiment = config.get("experiment_id")
        if not isinstance(current_experiment, str) or not current_experiment:
            raise ValueError(f"candidate {identifier} config has no experiment_id")
        try:
            actual_contact_experiment_id(
                config,
                context="Viewer measured-data selection",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"candidate {identifier} is not a registered actual-contact experiment"
            ) from exc
        if experiment_id is None:
            experiment_id = current_experiment
        elif current_experiment != experiment_id:
            raise ValueError("measured report mixes different experiment IDs")

        edge_mm = _finite_optional(
            _mapping_value(config.get("cube")).get("edge_m"), scale=1000.0
        )
        if edge_mm is None or edge_mm <= 0.0:
            raise ValueError(f"candidate {identifier} has an invalid cube edge")
        actual_qpos = _mapping_value(result.get("actual_grasp_pose_qpos_rad"))
        thumb = _finite_optional(actual_qpos.get(DEFAULT_V5_JOINT_MONITOR))
        if thumb is None:
            raise ValueError(
                f"candidate {identifier} has no finite actual thumb bend evidence"
            )
        metrics = _nested_mapping(result, "summary", "metrics")
        pose = _mapping_value(metrics.get("pose_preservation"))
        alignment = _mapping_value(metrics.get("contact_alignment"))
        verify_alignment = _mapping_value(alignment.get("verify"))
        height_spread = _finite_optional(
            verify_alignment.get(
                "height_spread_max_m",
                alignment.get(
                    "verify_max_height_spread_m",
                    metrics.get("verify_contact_height_spread_max_m"),
                ),
            ),
            scale=1000.0,
        )
        ranking_record = {**dict(result), "config": dict(config)}
        authenticated.append(
            (
                edge_mm,
                dynamic_grasp_rank(ranking_record),
                {
                    "candidate_id": identifier,
                    "edge_mm": edge_mm,
                    "actual_thumb_bend_rad": thumb,
                    "pose_translation_mm": _finite_optional(
                        pose.get("max_translation_m"), scale=1000.0
                    ),
                    "pose_orientation_deg": _finite_optional(
                        pose.get("max_orientation_drift_deg")
                    ),
                    "contact_height_spread_mm": height_spread,
                    "config_path": config_path,
                    "trace_path": trace_path,
                    "result_path": result_path.resolve(),
                },
            )
        )

    declared_count = payload.get("measured_grasp_pose_success_count")
    if declared_count is not None and int(declared_count) != len(authenticated):
        raise ValueError("measured report success count does not match candidate_records")
    authenticated.sort(key=lambda value: (value[0], value[1], value[2]["candidate_id"]))
    edge_counts: dict[float, int] = {}
    output: list[MeasuredViewerData] = []
    for data_index, (_, _, values) in enumerate(authenticated, start=1):
        edge_key = round(float(values["edge_mm"]), 9)
        edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
        output.append(
            MeasuredViewerData(
                data_index=data_index,
                edge_rank=edge_counts[edge_key],
                measured_grasp_pose_success=True,
                **values,
            )
        )
    return tuple(output)


def format_measured_viewer_data(entries: Sequence[MeasuredViewerData]) -> str:
    """Return a compact, copy-friendly table for ``view --list-data``."""

    header = (
        "data_index edge_rank edge_mm candidate_id actual_thumb_bend_rad "
        "pose_translation_mm pose_orientation_deg height_spread_mm"
    )

    def optional(value: float | None) -> str:
        return "-" if value is None else f"{value:.6f}"

    rows = [header]
    rows.extend(
        " ".join(
            (
                str(entry.data_index),
                str(entry.edge_rank),
                f"{entry.edge_mm:.3f}",
                str(entry.candidate_id),
                f"{entry.actual_thumb_bend_rad:.9f}",
                optional(entry.pose_translation_mm),
                optional(entry.pose_orientation_deg),
                optional(entry.contact_height_spread_mm),
            )
        )
        for entry in entries
    )
    return "\n".join(rows)


def resolve_measured_viewer_source(
    report_path: str | Path,
    *,
    data_index: int | None = None,
    candidate_id: int | str | None = None,
    edge_mm: float | None = None,
    edge_rank: int | None = None,
) -> ViewerSource:
    """Resolve exactly one authenticated measured grasp for live/replay use."""

    if edge_rank is not None and edge_mm is None:
        raise ValueError("edge rank requires an edge-mm selector")
    selectors = sum(value is not None for value in (data_index, candidate_id, edge_mm))
    if selectors != 1:
        raise ValueError(
            "choose exactly one measured selector: data index, candidate ID, "
            "or cube edge"
        )
    entries = discover_measured_viewer_data(report_path)
    selected: MeasuredViewerData
    if data_index is not None:
        if isinstance(data_index, bool) or data_index < 1 or data_index > len(entries):
            raise ValueError(f"data index must lie within [1, {len(entries)}]")
        selected = entries[data_index - 1]
    elif candidate_id is not None:
        token = str(candidate_id)
        if token.startswith("candidate_"):
            token = token[len("candidate_") :]
        if not token.isascii() or not token.isdigit():
            raise ValueError("candidate ID must contain decimal integer digits only")
        identifier = int(token)
        matches = [entry for entry in entries if entry.candidate_id == identifier]
        if not matches:
            raise ValueError(f"candidate ID {identifier} is not in the measured report")
        selected = matches[0]
    else:
        assert edge_mm is not None
        requested_edge = float(edge_mm)
        if not math.isfinite(requested_edge) or requested_edge <= 0.0:
            raise ValueError("selected cube edge must be positive and finite")
        matches = [
            entry
            for entry in entries
            if math.isclose(entry.edge_mm, requested_edge, rel_tol=0.0, abs_tol=1e-6)
        ]
        if not matches:
            available = sorted({entry.edge_mm for entry in entries})
            raise ValueError(
                f"cube edge {requested_edge:g} mm is unavailable; available: "
                + ", ".join(f"{value:g}" for value in available)
            )
        if edge_rank is None:
            if len(matches) != 1:
                raise ValueError(
                    f"cube edge {requested_edge:g} mm has multiple data; "
                    "select an explicit --edge-rank"
                )
            selected = matches[0]
        else:
            if isinstance(edge_rank, bool) or edge_rank < 1 or edge_rank > len(matches):
                raise ValueError(
                    f"edge rank must lie within [1, {len(matches)}] for "
                    f"{requested_edge:g} mm"
                )
            selected = matches[edge_rank - 1]
    return ViewerSource(
        config_path=selected.config_path,
        trace_path=selected.trace_path,
        trajectory=f"candidate_{selected.candidate_id}",
        from_catalog=True,
    )


def resolve_replay_source(
    *,
    catalog_path: str | Path | None = None,
    trajectory: str = "nominal",
    config_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    catalog_edge_mm: float | None = None,
    catalog_mapping_mode: str | None = None,
    catalog_source_alias: str | None = None,
    catalog_rank: int = 1,
) -> ReplaySource:
    """Resolve either a catalog label/id or an explicit config/trace pair."""

    using_catalog = catalog_path is not None
    using_direct = config_path is not None or trace_path is not None
    if using_catalog == using_direct:
        raise ValueError(
            "choose exactly one replay source: --catalog, or both --config and --trace"
        )

    if using_catalog:
        catalog = Path(catalog_path).expanduser().resolve()
        payload = json.loads(catalog.read_text(encoding="utf-8"))
        entries = payload.get("trajectories") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ValueError(f"invalid trajectory catalog: {catalog}")

        selector = str(trajectory)
        metadata_selection = any(
            value is not None
            for value in (
                catalog_edge_mm,
                catalog_mapping_mode,
                catalog_source_alias,
            )
        )
        if not isinstance(catalog_rank, int) or isinstance(catalog_rank, bool) or catalog_rank <= 0:
            raise ValueError("catalog_rank must be a positive one-based integer")
        if metadata_selection and selector != "nominal":
            raise ValueError(
                "catalog metadata selectors cannot be combined with --trajectory"
            )
        catalog_aliases = payload.get("aliases", {})
        if not isinstance(catalog_aliases, Mapping):
            raise ValueError("catalog aliases must be a mapping")
        # Most catalogs duplicate every top-level alias in the selected
        # trajectory entry.  Sealed schema-v14 contact-preserving catalogs
        # were published with a single authoritative top-level alias map and
        # omitted per-entry ``aliases`` altogether.  Preserve the stronger
        # symmetric check for every other format (and for new v14 catalogs
        # that declare the reverse map), while retaining read compatibility
        # with those immutable v14 artifacts.
        has_entry_alias_contract = any(
            isinstance(candidate, Mapping) and "aliases" in candidate
            for candidate in entries
        )
        permits_v14_top_level_only_aliases = (
            not has_entry_alias_contract
            and payload.get("experiment_id")
            == _V14_CONTACT_PRESERVING_EXPERIMENT_ID
            and payload.get("contact_preserving_viewer_catalog_schema_version") == 1
        )
        alias_target: str | None = None
        if not metadata_selection and selector in catalog_aliases:
            raw_target = catalog_aliases[selector]
            if not isinstance(raw_target, str) or not raw_target:
                raise ValueError(
                    f"catalog alias {selector!r} has no valid trajectory target"
                )
            alias_target = raw_target
        matches: list[Mapping[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            entry_aliases = entry.get("aliases", [])
            if not isinstance(entry_aliases, list) or not all(
                isinstance(alias, str) and alias for alias in entry_aliases
            ):
                raise ValueError("catalog trajectory aliases must be non-empty strings")
            selectors = {
                str(entry.get("label", "")),
                str(entry.get("trajectory_id", "")),
                str(entry.get("grid_index", "")),
            }
            if entry.get("candidate_id") is not None:
                selectors.add(str(entry.get("candidate_id")))
                selectors.add(f"candidate_{entry.get('candidate_id')}")
            selectors.update(entry_aliases)
            if metadata_selection:
                metadata = entry.get("campaign_metadata", {})
                if not isinstance(metadata, Mapping):
                    metadata = {}
                edge_value = entry.get("edge_mm", metadata.get("edge_mm"))
                mapping_value = entry.get(
                    "mapping_mode", metadata.get("mapping_mode")
                )
                source_value = entry.get(
                    "source_alias", metadata.get("source_alias")
                )
                selected = True
                if catalog_edge_mm is not None:
                    try:
                        selected = selected and math.isclose(
                            float(edge_value),
                            float(catalog_edge_mm),
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        )
                    except (TypeError, ValueError):
                        selected = False
                if catalog_mapping_mode is not None:
                    selected = selected and str(mapping_value) == str(
                        catalog_mapping_mode
                    )
                if catalog_source_alias is not None:
                    selected = selected and str(source_value) == str(
                        catalog_source_alias
                    )
            elif alias_target is not None:
                selected = str(entry.get("trajectory_id", "")) == alias_target
            else:
                selected = selector in selectors
            if selected:
                matches.append(entry)
        if not matches:
            available = sorted(
                str(entry.get("label") or entry.get("trajectory_id"))
                for entry in entries
                if isinstance(entry, Mapping)
            )
            if metadata_selection:
                raise ValueError(
                    "no catalog trajectory matches the requested edge/mapping/source "
                    "metadata"
                )
            raise ValueError(
                f"trajectory {selector!r} is not in {catalog}; "
                f"available: {', '.join(available)}"
            )
        if metadata_selection:
            if catalog_rank > len(matches):
                raise ValueError(
                    f"catalog_rank {catalog_rank} exceeds {len(matches)} metadata matches"
                )
            entry = matches[catalog_rank - 1]
            selector = str(entry.get("trajectory_id", "metadata_selection"))
        elif len(matches) != 1:
            raise ValueError(f"trajectory selector is ambiguous: {selector!r}")
        else:
            entry = matches[0]
        if (
            not metadata_selection
            and alias_target is not None
            and not permits_v14_top_level_only_aliases
        ):
            claiming_entries = [
                candidate
                for candidate in entries
                if isinstance(candidate, Mapping)
                and selector in candidate.get("aliases", [])
            ]
            if claiming_entries != [entry]:
                raise ValueError(
                    f"catalog alias {selector!r} disagrees with trajectory aliases"
                )
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"trajectory {selector!r} has no artifact map")

        is_v15_catalog = (
            payload.get("experiment_id")
            == _V15_JOINT_PAIR_NEAR_ZERO_EXPERIMENT_ID
        )
        if is_v15_catalog:
            catalog_version = payload.get(
                "joint_pair_near_zero_viewer_catalog_schema_version"
            )
            if (
                isinstance(catalog_version, bool)
                or catalog_version != _V15_VIEWER_CATALOG_SCHEMA_VERSION
            ):
                raise ValueError(
                    "schema-v15 catalog has an unsupported or missing "
                    "joint-pair Viewer catalog version"
                )

        # The sealed schema-v15 publisher names this artifact ``config`` even
        # though the copied file itself is ``resolved_config.json``.  Accept
        # that versioned contract without weakening generic catalogs.  A
        # future v15 publisher may use the conventional key; if both aliases
        # are present they must identify the same file and both hashes are
        # authenticated below.
        config_fields = ["resolved_config"]
        if is_v15_catalog:
            config_fields = [
                field
                for field in ("resolved_config", "config")
                if field in artifacts
            ]
            if not config_fields:
                raise ValueError(
                    "schema-v15 catalog trajectory has no resolved_config/config "
                    "artifact"
                )
        resolved_configs = [
            _catalog_member(catalog, artifacts.get(field), field)
            for field in config_fields
        ]
        if len(set(resolved_configs)) != 1:
            raise ValueError(
                "schema-v15 catalog config aliases identify different files"
            )
        resolved_config = resolved_configs[0]
        trace = _catalog_member(catalog, artifacts.get("trace"), "trace")
        contact_environment = None
        if "contact_environment" in artifacts:
            contact_environment = _catalog_member(
                catalog,
                artifacts.get("contact_environment"),
                "contact_environment",
            )
        digests = artifacts.get("sha256")
        digests_required = (
            payload.get("experiment_id") == _V4_ALIGNED_EXPERIMENT_ID
            or bool(payload.get("experiment_id"))
            and "declared_tilt_bands_deg" in payload
            or is_v15_catalog
        )
        for config_field in config_fields:
            _verify_catalog_digest(
                resolved_config,
                digests,
                config_field,
                required=digests_required,
                required_catalog=(
                    "schema-v15 catalog"
                    if is_v15_catalog
                    else "schema-v4 catalog"
                ),
            )
        _verify_catalog_digest(
            trace,
            digests,
            "trace",
            required=digests_required,
            required_catalog=(
                "schema-v15 catalog"
                if is_v15_catalog
                else "schema-v4 catalog"
            ),
        )
        if contact_environment is not None:
            _verify_catalog_digest(
                contact_environment,
                digests,
                "contact_environment",
                required=True,
                required_catalog="contact-environment catalog",
            )
        return ReplaySource(
            config_path=resolved_config,
            trace_path=trace,
            trajectory=str(entry.get("label") or entry.get("trajectory_id") or selector),
            contact_environment_path=contact_environment,
        )

    if config_path is None or trace_path is None:
        raise ValueError("direct replay requires both --config and --trace")
    config = Path(config_path).expanduser().resolve()
    trace = Path(trace_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"config does not exist: {config}")
    if not trace.is_file():
        raise FileNotFoundError(f"trace does not exist: {trace}")
    return ReplaySource(config, trace, trace.parent.name)


def resolve_viewer_source(
    *,
    catalog_path: str | Path | None = None,
    trajectory: str = "nominal",
    config_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    catalog_edge_mm: float | None = None,
    catalog_mapping_mode: str | None = None,
    catalog_source_alias: str | None = None,
    catalog_rank: int = 1,
) -> ViewerSource:
    """Resolve a live source without requiring NPZ for a direct config.

    Catalog members still undergo the same confinement and SHA-256 checks as
    state replay.  When a direct trace is supplied it is used only as an exact
    post-run reference unless the caller explicitly selects state replay.
    """

    if (catalog_path is None) == (config_path is None):
        raise ValueError("choose exactly one live source: --catalog or --config")
    if catalog_path is not None:
        if trace_path is not None:
            raise ValueError("--trace cannot be combined with --catalog")
        replay = resolve_replay_source(
            catalog_path=catalog_path,
            trajectory=trajectory,
            catalog_edge_mm=catalog_edge_mm,
            catalog_mapping_mode=catalog_mapping_mode,
            catalog_source_alias=catalog_source_alias,
            catalog_rank=catalog_rank,
        )
        return ViewerSource(
            config_path=replay.config_path,
            trace_path=replay.trace_path,
            trajectory=replay.trajectory,
            from_catalog=True,
            contact_environment_path=replay.contact_environment_path,
        )

    assert config_path is not None
    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"config does not exist: {config}")
    trace: Path | None = None
    if trace_path is not None:
        trace = Path(trace_path).expanduser().resolve()
        if not trace.is_file():
            raise FileNotFoundError(f"trace does not exist: {trace}")
    return ViewerSource(
        config_path=config,
        trace_path=trace,
        trajectory=trace.parent.name if trace is not None else config.stem,
        from_catalog=False,
    )


def load_replay_trace(path: str | Path) -> dict[str, np.ndarray]:
    """Load and structurally validate the state arrays needed by the viewer."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_TRACE_FIELDS) - set(archive.files))
        if missing:
            raise ValueError(f"trace is missing fields: {', '.join(missing)}")
        trace = {name: np.array(archive[name], copy=True) for name in archive.files}

    time_values = trace["time"]
    if time_values.ndim != 1 or len(time_values) == 0:
        raise ValueError("trace time must be a non-empty one-dimensional array")
    frame_count = len(time_values)
    for name, ndim in REQUIRED_TRACE_FIELDS.items():
        values = trace[name]
        if values.ndim != ndim:
            raise ValueError(f"trace field {name} must have {ndim} dimensions")
        if len(values) != frame_count:
            raise ValueError(f"trace field {name} has a different frame count")
        if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
            raise ValueError(f"trace field {name} contains non-finite/non-numeric data")
    if frame_count > 1 and np.any(np.diff(time_values) <= 0.0):
        raise ValueError("trace time must be strictly increasing")
    expected_widths = {
        "cube_pos": 3,
        "cube_quat": 4,
        "cube_velocity": 6,
    }
    for name, width in expected_widths.items():
        if trace[name].shape[1] != width:
            raise ValueError(f"trace field {name} must have width {width}")
    quaternion_norms = np.linalg.norm(trace["cube_quat"], axis=1)
    if not np.allclose(quaternion_norms, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("trace cube quaternions are not normalized")
    return trace


def validate_trace_model_binding(
    model: mujoco.MjModel,
    info: ModelInfo,
    trace: Mapping[str, np.ndarray],
) -> None:
    """Reject traces that cannot be mapped to this compiled model by name."""

    actuator_count = len(info.actuator_qpos_adrs)
    for name in ("ctrl", "joint_qpos", "joint_qvel"):
        if trace[name].shape[1] != actuator_count:
            raise ValueError(
                f"trace {name} width {trace[name].shape[1]} does not match "
                f"model actuator count {actuator_count}"
            )
    if "actuator_order" in trace:
        recorded = tuple(str(value) for value in trace["actuator_order"].tolist())
        compiled = tuple(model.actuator(index).name for index in range(model.nu))
        if recorded != compiled:
            raise ValueError("trace actuator_order does not match the compiled model")


def apply_replay_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: ModelInfo,
    trace: Mapping[str, np.ndarray],
    frame: int,
) -> None:
    """Restore one complete recorded state and refresh derived MuJoCo fields."""

    index = int(frame)
    if not 0 <= index < len(trace["time"]):
        raise IndexError(f"replay frame is out of range: {index}")
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    if model.na:
        data.act[:] = 0.0
    data.qpos[info.actuator_qpos_adrs] = trace["joint_qpos"][index]
    data.qvel[info.actuator_dof_adrs] = trace["joint_qvel"][index]
    cube_qpos = info.cube_qpos_adr
    cube_dof = info.cube_dof_adr
    data.qpos[cube_qpos : cube_qpos + 3] = trace["cube_pos"][index]
    data.qpos[cube_qpos + 3 : cube_qpos + 7] = trace["cube_quat"][index]
    data.qvel[cube_dof : cube_dof + 6] = trace["cube_velocity"][index]
    data.ctrl[:] = trace["ctrl"][index]
    data.time = float(trace["time"][index])
    mujoco.mj_forward(model, data)


def _rpy_matrix(rpy_deg: object) -> np.ndarray:
    return rpy_degrees_to_rotation_matrix(rpy_deg)  # type: ignore[arg-type]


def _configured_cube_center(config: Mapping[str, Any]) -> np.ndarray:
    """Return the compiled initial cube centre for override pose algebra."""

    cube = config["cube"]
    edge = float(cube["edge_m"])
    support_top = float(config["scene"]["support_top_z_m"])
    z_offset = float(cube.get("z_offset_m", 0.0))
    half_height = edge / 2.0
    if int(config.get("schema_version", 1)) >= 4:
        rotation = _rpy_matrix(cube.get("rpy_deg", [0.0, 0.0, 0.0]))
        half_height = cube_vertical_half_extent_m(edge, rotation)
    center_xy = np.asarray(cube["center_xy_m"], dtype=np.float64)
    return np.asarray(
        [center_xy[0], center_xy[1], support_top + half_height + z_offset],
        dtype=np.float64,
    )


def _finite_scalar(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_vector3(value: object, label: str) -> np.ndarray:
    """Return one finite three-vector used by Viewer pose algebra."""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain three finite values")
    return result


def _stored_relative_wrist_pose(
    source_config: Mapping[str, Any],
) -> tuple[dict[str, list[float]], float, np.ndarray, np.ndarray]:
    """Resolve the immutable anchor and stored absolute v11 search values.

    Search candidates already contain the result of the orbit/residual
    transform.  Viewer controls must therefore restart from ``anchor_hand_pose``
    instead of treating the resolved ``hand_pose`` as another anchor.  Older
    direct configs have no search metadata, for which the current hand pose and
    zero residuals are the natural backwards-compatible anchor.
    """

    candidate_metadata = source_config.get("candidate_metadata")
    relative = (
        candidate_metadata.get("relative_wrist_pose_search", {})
        if isinstance(candidate_metadata, Mapping)
        else {}
    )
    if not isinstance(relative, Mapping):
        raise ValueError(
            "candidate_metadata.relative_wrist_pose_search must be an object"
        )
    raw_anchor = relative.get("anchor_hand_pose", source_config["hand_pose"])
    if not isinstance(raw_anchor, Mapping):
        raise ValueError(
            "relative_wrist_pose_search.anchor_hand_pose must be an object"
        )
    anchor = {
        "translation_m": _finite_vector3(
            raw_anchor.get("translation_m"),
            "relative_wrist_pose_search.anchor_hand_pose.translation_m",
        ).tolist(),
        "rpy_deg": _finite_vector3(
            raw_anchor.get("rpy_deg"),
            "relative_wrist_pose_search.anchor_hand_pose.rpy_deg",
        ).tolist(),
    }
    stored_orbit = _finite_scalar(
        relative.get("clockwise_orbit_deg", 0.0),
        "relative_wrist_pose_search.clockwise_orbit_deg",
    )
    assert stored_orbit is not None
    stored_delta = _finite_vector3(
        relative.get("root_delta_cube_m", (0.0, 0.0, 0.0)),
        "relative_wrist_pose_search.root_delta_cube_m",
    )
    stored_rotvec = _finite_vector3(
        relative.get("wrist_local_rotvec_deg", (0.0, 0.0, 0.0)),
        "relative_wrist_pose_search.wrist_local_rotvec_deg",
    )
    return anchor, stored_orbit, stored_delta, stored_rotvec


def parse_actuator_overrides(
    values: Sequence[str] | None,
    *,
    option: str,
) -> dict[str, float]:
    """Parse repeatable ``ACTUATOR=VALUE`` arguments without silent overwrite."""

    parsed: dict[str, float] = {}
    for item in values or ():
        name, separator, raw_value = str(item).partition("=")
        name = name.strip()
        raw_value = raw_value.strip()
        if not separator or not name or not raw_value:
            raise ValueError(f"{option} must use ACTUATOR=VALUE")
        if name not in ACTIVE_ACTUATORS:
            raise ValueError(
                f"{option} has unknown or inactive actuator {name!r}; "
                f"expected one of: {', '.join(ACTIVE_ACTUATORS)}"
            )
        if name in parsed:
            raise ValueError(f"{option} repeats actuator {name!r}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"{option} value for {name!r} must be a finite number"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"{option} value for {name!r} must be a finite number"
            )
        parsed[name] = value
    return parsed


def _finger_down_pitch_deg(roll_deg: float, finger_down_deg: float) -> float:
    """Solve pitch so local +Z has the requested signed downward angle."""

    roll = math.radians(float(roll_deg))
    downward = math.radians(float(finger_down_deg))
    denominator = math.cos(roll)
    if abs(denominator) <= 1e-9:
        raise ValueError("cannot solve finger-down tilt with hand roll near 90 degrees")
    cosine_pitch = -math.sin(downward) / denominator
    if not -1.0 <= cosine_pitch <= 1.0:
        raise ValueError("requested finger-down tilt is incompatible with hand roll")
    return math.degrees(math.acos(float(np.clip(cosine_pitch, -1.0, 1.0))))


def apply_viewer_overrides(
    source_config: Mapping[str, Any],
    *,
    edge_mm: float | None = None,
    mass_g: float | None = None,
    density_scale: float | None = None,
    friction: float | None = None,
    finger_down_deg: float | None = None,
    hand_roll_deg: float | None = None,
    hand_yaw_deg: float | None = None,
    hand_rpy_deg: object | None = None,
    press_mm: float | None = None,
    root_cube_distance_mm: float | None = None,
    cube_in_root_mm: object | None = None,
    cube_rpy_deg: object | None = None,
    clockwise_orbit_deg: float | None = None,
    root_delta_cube_mm: object | None = None,
    wrist_local_rotvec_deg: object | None = None,
    grasp_target_rad: Mapping[str, float] | None = None,
    manipulation_delta_rad: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Apply live-only physical/relative-pose overrides and revalidate.

    A density scale preserves the source material density while accounting for
    an edge override.  Hand-angle overrides preserve the source cube-in-root
    vector unless ``--press-mm``, ``--root-cube-distance-mm`` or
    ``--cube-in-root-mm`` supplies a more specific relative-pose constraint.
    The orbit/translation/rotation-vector family is a coupled rigid transform
    and is therefore deliberately incompatible with those legacy hand-pose
    controls.  Its three arguments are absolute search coordinates: omitted
    coordinates inherit the stored candidate value, supplied coordinates
    replace it, and the complete transform is applied exactly once from the
    candidate's immutable ``anchor_hand_pose``.
    """

    if mass_g is not None and density_scale is not None:
        raise ValueError("--mass-g and --density-scale are mutually exclusive")
    if hand_rpy_deg is not None and any(
        value is not None
        for value in (finger_down_deg, hand_roll_deg, hand_yaw_deg)
    ):
        raise ValueError(
            "--hand-rpy-deg cannot be combined with finger-down/roll/yaw overrides"
        )
    relative_overrides = tuple(
        value is not None
        for value in (press_mm, root_cube_distance_mm, cube_in_root_mm)
    )
    if sum(relative_overrides) > 1:
        raise ValueError(
            "--press-mm, --root-cube-distance-mm and --cube-in-root-mm "
            "are mutually exclusive"
        )
    if int(source_config.get("schema_version", 1)) >= 5 and press_mm is not None:
        raise ValueError(
            "--press-mm is not supported by schema-v5; use "
            "--root-cube-distance-mm or --cube-in-root-mm"
        )
    coupled_relative_override = any(
        value is not None
        for value in (
            clockwise_orbit_deg,
            root_delta_cube_mm,
            wrist_local_rotvec_deg,
        )
    )
    legacy_hand_pose_override = any(
        value is not None
        for value in (
            finger_down_deg,
            hand_roll_deg,
            hand_yaw_deg,
            hand_rpy_deg,
            press_mm,
            root_cube_distance_mm,
            cube_in_root_mm,
        )
    )
    if coupled_relative_override and legacy_hand_pose_override:
        raise ValueError(
            "clockwise orbit/root cube delta/local wrist rotation overrides "
            "cannot be combined with legacy hand-angle, press, distance or "
            "cube-in-root overrides"
        )

    scalar_values = {
        "--edge-mm": edge_mm,
        "--mass-g": mass_g,
        "--density-scale": density_scale,
        "--friction": friction,
        "--finger-down-deg": finger_down_deg,
        "--hand-roll-deg": hand_roll_deg,
        "--hand-yaw-deg": hand_yaw_deg,
        "--press-mm": press_mm,
        "--root-cube-distance-mm": root_cube_distance_mm,
        "--clockwise-orbit-deg": clockwise_orbit_deg,
    }
    checked = {
        key: _finite_scalar(value, key) for key, value in scalar_values.items()
    }
    edge_mm = checked["--edge-mm"]
    mass_g = checked["--mass-g"]
    density_scale = checked["--density-scale"]
    friction = checked["--friction"]
    finger_down_deg = checked["--finger-down-deg"]
    hand_roll_deg = checked["--hand-roll-deg"]
    hand_yaw_deg = checked["--hand-yaw-deg"]
    press_mm = checked["--press-mm"]
    root_cube_distance_mm = checked["--root-cube-distance-mm"]
    clockwise_orbit_deg = checked["--clockwise-orbit-deg"]
    for label, value in (
        ("--edge-mm", edge_mm),
        ("--mass-g", mass_g),
        ("--density-scale", density_scale),
        ("--friction", friction),
        ("--root-cube-distance-mm", root_cube_distance_mm),
    ):
        if value is not None and value <= 0.0:
            raise ValueError(f"{label} must be greater than zero")

    config = copy.deepcopy(dict(source_config))
    source = copy.deepcopy(dict(source_config))
    source_edge = float(source["cube"]["edge_m"])
    source_mass = float(source["cube"]["mass_kg"])
    source_density = source_mass / source_edge**3
    source_cube_world = _configured_cube_center(source)
    source_root = np.asarray(
        source["hand_pose"]["translation_m"], dtype=np.float64
    )
    source_rotation = _rpy_matrix(source["hand_pose"]["rpy_deg"])
    source_cube_rotation = _rpy_matrix(
        source["cube"].get("rpy_deg", [0.0, 0.0, 0.0])
    )
    source_cube_in_root = source_rotation.T @ (source_cube_world - source_root)
    relative_override_metadata: dict[str, Any] | None = None

    control_overridden = bool(grasp_target_rad) or bool(manipulation_delta_rad)
    overridden = control_overridden or any(
        value is not None
        for value in (
            edge_mm,
            mass_g,
            density_scale,
            friction,
            finger_down_deg,
            hand_roll_deg,
            hand_yaw_deg,
            hand_rpy_deg,
            press_mm,
            root_cube_distance_mm,
            cube_in_root_mm,
            cube_rpy_deg,
            clockwise_orbit_deg,
            root_delta_cube_mm,
            wrist_local_rotvec_deg,
        )
    )
    if edge_mm is not None:
        config["cube"]["edge_m"] = edge_mm / 1000.0
    if mass_g is not None:
        config["cube"]["mass_kg"] = mass_g / 1000.0
    elif density_scale is not None:
        config["cube"]["mass_kg"] = (
            source_density
            * float(config["cube"]["edge_m"]) ** 3
            * density_scale
        )
    if friction is not None:
        config["cube"]["friction"] = friction
    if cube_rpy_deg is not None:
        cube_rpy = np.asarray(cube_rpy_deg, dtype=np.float64)
        if cube_rpy.shape != (3,) or not np.isfinite(cube_rpy).all():
            raise ValueError("--cube-rpy-deg must contain three finite values")
        config["cube"]["rpy_deg"] = cube_rpy.tolist()

    pose_override = any(
        value is not None
        for value in (
            finger_down_deg,
            hand_roll_deg,
            hand_yaw_deg,
            hand_rpy_deg,
            press_mm,
            root_cube_distance_mm,
            cube_in_root_mm,
            clockwise_orbit_deg,
            root_delta_cube_mm,
            wrist_local_rotvec_deg,
        )
    )
    if pose_override:
        if coupled_relative_override:
            (
                anchor_hand_pose,
                stored_orbit_deg,
                stored_root_delta_m,
                stored_wrist_rotvec_deg,
            ) = _stored_relative_wrist_pose(source)
            effective_orbit_deg = (
                stored_orbit_deg
                if clockwise_orbit_deg is None
                else clockwise_orbit_deg
            )
            effective_root_delta_m = (
                stored_root_delta_m
                if root_delta_cube_mm is None
                else _finite_vector3(
                    root_delta_cube_mm, "--root-delta-cube-mm"
                )
                / 1000.0
            )
            effective_wrist_rotvec_deg = (
                stored_wrist_rotvec_deg
                if wrist_local_rotvec_deg is None
                else _finite_vector3(
                    wrist_local_rotvec_deg, "--wrist-local-rotvec-deg"
                )
            )
            cube_world = _configured_cube_center(config)
            cube_rotation = _rpy_matrix(
                config["cube"].get("rpy_deg", [0.0, 0.0, 0.0])
            )
            anchor_root = np.asarray(
                anchor_hand_pose["translation_m"], dtype=np.float64
            )
            anchor_rotation = _rpy_matrix(anchor_hand_pose["rpy_deg"])
            # Re-express the immutable zero-residual anchor in the possibly
            # overridden target cube frame.  Persisting this target-frame
            # anchor makes a later Viewer override idempotent even when the
            # first run also changed edge size or cube orientation.
            target_anchor = transform_relative_wrist_pose(
                source_cube_world_position_m=source_cube_world,
                source_cube_world_rotation=source_cube_rotation,
                source_root_world_position_m=anchor_root,
                source_root_world_rotation=anchor_rotation,
                target_cube_world_position_m=cube_world,
                target_cube_world_rotation=cube_rotation,
            )
            target_anchor_pose = {
                "translation_m": list(target_anchor.root_world_position_m),
                "rpy_deg": rotation_matrix_to_rpy_degrees(
                    target_anchor.root_world_rotation,
                    reference_rpy_deg=anchor_hand_pose["rpy_deg"],
                ).tolist(),
            }
            transformed = transform_relative_wrist_pose(
                source_cube_world_position_m=source_cube_world,
                source_cube_world_rotation=source_cube_rotation,
                source_root_world_position_m=anchor_root,
                source_root_world_rotation=anchor_rotation,
                target_cube_world_position_m=cube_world,
                target_cube_world_rotation=cube_rotation,
                clockwise_orbit_deg=effective_orbit_deg,
                root_delta_cube_m=effective_root_delta_m,
                wrist_local_rotvec_deg=effective_wrist_rotvec_deg,
            )
            result_rotation = np.asarray(
                transformed.root_world_rotation, dtype=np.float64
            )
            config["hand_pose"]["translation_m"] = list(
                transformed.root_world_position_m
            )
            config["hand_pose"]["rpy_deg"] = rotation_matrix_to_rpy_degrees(
                result_rotation,
                reference_rpy_deg=source["hand_pose"]["rpy_deg"],
            ).tolist()
            config["relative_wrist_pose_diagnostics"] = (
                transformed.diagnostics.as_dict()
            )
            relative_override_metadata = {
                "anchor_hand_pose": target_anchor_pose,
                "clockwise_orbit_deg": float(effective_orbit_deg),
                "root_delta_cube_m": effective_root_delta_m.tolist(),
                "wrist_local_rotvec_deg": effective_wrist_rotvec_deg.tolist(),
                "cube_to_hand_translation_cube_m": list(
                    transformed.root_in_cube_m
                ),
                "cube_to_hand_rotation": [
                    list(row) for row in transformed.cube_from_root_rotation
                ],
                "root_cube_distance_m": float(
                    transformed.diagnostics.result_root_cube_distance_m
                ),
                "cube_pose_sampled": False,
                "hand_root_fixed_during_simulation": True,
                "diagnostic_only": True,
                "source_success_evidence_inherited": False,
                "viewer_parameter_replacements": {
                    "clockwise_orbit_deg": clockwise_orbit_deg is not None,
                    "root_delta_cube_m": root_delta_cube_mm is not None,
                    "wrist_local_rotvec_deg": wrist_local_rotvec_deg is not None,
                },
            }
        elif hand_rpy_deg is not None:
            hand_rpy = np.asarray(hand_rpy_deg, dtype=np.float64)
            if hand_rpy.shape != (3,) or not np.isfinite(hand_rpy).all():
                raise ValueError("--hand-rpy-deg must contain three finite values")
        else:
            hand_rpy = np.asarray(config["hand_pose"]["rpy_deg"], dtype=np.float64)
            if hand_roll_deg is not None:
                hand_rpy[0] = hand_roll_deg
            if hand_yaw_deg is not None:
                hand_rpy[2] = hand_yaw_deg
            if finger_down_deg is not None:
                hand_rpy[1] = _finger_down_pitch_deg(
                    float(hand_rpy[0]), finger_down_deg
                )
        if not coupled_relative_override:
            config["hand_pose"]["rpy_deg"] = hand_rpy.tolist()
            rotation = _rpy_matrix(hand_rpy)
            cube_world = _configured_cube_center(config)
            if cube_in_root_mm is not None:
                relative = np.asarray(cube_in_root_mm, dtype=np.float64)
                if relative.shape != (3,) or not np.isfinite(relative).all():
                    raise ValueError(
                        "--cube-in-root-mm must contain three finite values"
                    )
                relative = relative / 1000.0
            else:
                relative = source_cube_in_root.copy()
            if root_cube_distance_mm is not None:
                source_distance = float(np.linalg.norm(relative))
                if source_distance <= 1e-12:
                    raise ValueError(
                        "cannot apply --root-cube-distance-mm to a zero relative vector"
                    )
                relative *= (root_cube_distance_mm / 1000.0) / source_distance
            if press_mm is not None:
                pose_constraints = config.get("pose_constraints")
                if int(config.get("schema_version", 1)) >= 4 and isinstance(
                    pose_constraints, Mapping
                ):
                    reference_root = pose_constraints[
                        "reference_hand_translation_m"
                    ]
                else:
                    reference_root = source_root
                solved = solve_press_depth_pose(
                    reference_root_translation_m=reference_root,
                    press_depth_m=press_mm / 1000.0,
                    rotation_world_from_root=rotation,
                    cube_world_position_m=cube_world,
                    cube_in_root_y_m=float(relative[1]),
                    cube_in_root_z_m=float(relative[2]),
                )
                relative = np.asarray(solved.cube_in_root_m, dtype=np.float64)
                root_translation = np.asarray(
                    solved.root_translation_m, dtype=np.float64
                )
            else:
                root_translation = cube_world - rotation @ relative
            config["hand_pose"]["translation_m"] = root_translation.tolist()

    if control_overridden:
        if int(config.get("schema_version", 1)) < 3:
            raise ValueError(
                "grasp/manipulation target overrides require schema-version 3 or newer"
            )
        control = config["control"]
        grasp_field = (
            "contact_preload_targets_rad"
            if int(config.get("schema_version", 1)) >= 9
            else "grasp_targets_rad"
        )
        for option, updates, field in (
            ("--grasp-target-rad", grasp_target_rad, grasp_field),
            (
                "--manipulation-delta-rad",
                manipulation_delta_rad,
                "manipulation_delta_rad",
            ),
        ):
            for name, raw_value in (updates or {}).items():
                if name not in ACTIVE_ACTUATORS:
                    raise ValueError(
                        f"{option} has unknown or inactive actuator {name!r}"
                    )
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError(f"{option} value for {name!r} must be finite")
                control[field][name] = value

    if overridden:
        # Input validation claims are provenance, not evidence for this newly
        # simulated parameter set.  The completed result derives fresh status.
        config.pop("experiment_status", None)
        if relative_override_metadata is None:
            config.pop("candidate_metadata", None)
        else:
            # Preserve only enough metadata to describe/recompute this fresh
            # Viewer run.  Candidate IDs, measured gates and prior success
            # flags are intentionally not inherited as evidence.
            config["candidate_metadata"] = {
                "campaign_kind": "viewer_parameter_override_diagnostic",
                "relative_wrist_pose_search": relative_override_metadata,
                "parameter_override_run": True,
                "source_success_evidence_inherited": False,
            }
        config["run_context"] = {"kind": "parameter_override_run"}
    validate_config(config)
    # Compile the resolved scene once so both grasp and grasp+delta targets are
    # checked against the model's real actuator limits before opening GLFW.
    preflight_config(config)
    return config, overridden


def _require_interactive_gl() -> None:
    backend = os.environ.get("MUJOCO_GL", "").strip().lower()
    if backend in {"osmesa", "egl"}:
        raise RuntimeError(
            f"MuJoCo Viewer cannot use the headless {backend.upper()} backend; "
            "rerun with: MUJOCO_GL=glfw ./scripts/uv.sh run --frozen python "
            "grasp_cube.py view ..."
        )
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError("MuJoCo Viewer requires DISPLAY or WAYLAND_DISPLAY")


def copy_physics_to_display(
    physics_model: mujoco.MjModel,
    physics_data: mujoco.MjData,
    display_model: mujoco.MjModel,
    display_data: mujoco.MjData,
) -> None:
    """Copy integration state to an isolated Viewer model/data pair."""

    dimensions = ("nq", "nv", "na", "nu", "nmocap", "nuserdata")
    mismatched = [
        name
        for name in dimensions
        if int(getattr(physics_model, name)) != int(getattr(display_model, name))
    ]
    if mismatched:
        raise ValueError(
            "physics/display models are incompatible: " + ", ".join(mismatched)
        )
    display_data.time = float(physics_data.time)
    display_data.qpos[:] = physics_data.qpos
    display_data.qvel[:] = physics_data.qvel
    if physics_model.na:
        display_data.act[:] = physics_data.act
    display_data.ctrl[:] = physics_data.ctrl
    if physics_model.nmocap:
        display_data.mocap_pos[:] = physics_data.mocap_pos
        display_data.mocap_quat[:] = physics_data.mocap_quat
    if physics_model.nuserdata:
        display_data.userdata[:] = physics_data.userdata
    mujoco.mj_forward(display_model, display_data)


def resolve_joint_monitor(
    model: mujoco.MjModel,
    config: Mapping[str, Any],
    requested: str | None,
) -> JointMonitorBinding | None:
    """Resolve an optional monitored actuator and its transmitted joint."""

    name = requested
    if name is None and int(config.get("schema_version", 1)) >= 5:
        name = DEFAULT_V5_JOINT_MONITOR
    if name is None:
        return None
    try:
        actuator_id = int(model.actuator(name).id)
    except KeyError as exc:
        raise ValueError(f"--joint-monitor names unknown actuator {name!r}") from exc
    transmission = int(model.actuator_trntype[actuator_id])
    if transmission not in (
        int(mujoco.mjtTrn.mjTRN_JOINT),
        int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
    ):
        raise ValueError(
            f"--joint-monitor actuator {name!r} does not transmit through one joint"
        )
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    if joint_id < 0:
        raise ValueError(f"--joint-monitor actuator {name!r} has no joint")
    finger_index = next(
        (
            index
            for index, finger in enumerate(ACTIVE_FINGERS)
            if name.startswith(f"left_hand_{finger}_")
        ),
        None,
    )
    return JointMonitorBinding(
        actuator_name=name,
        actuator_id=actuator_id,
        joint_id=joint_id,
        qpos_adr=int(model.jnt_qposadr[joint_id]),
        finger_index=finger_index,
    )


def format_joint_pair_telemetry(telemetry: Mapping[str, Any]) -> str:
    """Format the displayed anchor-line geometry for terminal inspection."""

    vector_mm = 1000.0 * np.asarray(
        telemetry["vector_cube_m"], dtype=np.float64
    )
    return (
        f"[joint-pair] {telemetry['first_joint']} -> "
        f"{telemetry['second_joint']}: cube_local_mm="
        f"({vector_mm[0]:+.3f}, {vector_mm[1]:+.3f}, {vector_mm[2]:+.3f}), "
        f"angle_to_cube_y={float(telemetry['angle_to_cube_y_deg']):.3f}deg"
    )


def joint_pair_overlay_status(
    traces: Mapping[str, np.ndarray],
    step: int,
    config: Mapping[str, Any] | None,
) -> str | None:
    """Resolve schema-v15 safe/freeze/risk/abort coloring for one frame."""

    if (
        config is None
        or int(config.get("schema_version", 0)) < 15
        or step < 0
        or "joint_pair_angle_deg" not in traces
    ):
        return None
    states = np.asarray(traces.get("control_state", ())).astype(str)
    abort_risk = np.asarray(
        traces.get("joint_pair_abort_risk", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    frozen = np.asarray(
        traces.get("joint_pair_progress_frozen", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    valid = np.asarray(traces["joint_pair_valid"], dtype=bool)
    positive = np.asarray(traces["joint_pair_positive_y"], dtype=bool)
    angle = np.asarray(traces["joint_pair_angle_deg"], dtype=np.float64)
    if step >= angle.shape[0]:
        return None
    if (
        (step < states.shape[0] and states[step] == "ABORT")
        or (step < abort_risk.shape[0] and abort_risk[step])
    ):
        return "abort"
    if step < frozen.shape[0] and frozen[step]:
        return "frozen"
    if not valid[step] or not positive[step]:
        return "risk"
    thresholds = config["joint_pair_feedback"]
    if angle[step] > float(thresholds["abort_threshold_deg"]) + 1e-12:
        return "risk"
    if angle[step] > float(thresholds["freeze_threshold_deg"]) + 1e-12:
        return "warning"
    return "safe"


def joint_monitor_telemetry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    traces: Mapping[str, np.ndarray],
    step: int,
    binding: JointMonitorBinding,
) -> dict[str, float | int | str | None]:
    """Return one testable telemetry record for the selected actuator."""

    target = float(data.ctrl[binding.actuator_id])
    actual = float(data.qpos[binding.qpos_adr])
    result: dict[str, float | int | str | None] = {
        "actuator": binding.actuator_name,
        "time_s": float(data.time),
        "target_rad": target,
        "actual_rad": actual,
        "error_rad": target - actual,
        "contact_force_n": None,
        "pad_force_n": None,
        "pad_force_fraction": None,
        "active_taxel_count": None,
    }
    finger = binding.finger_index
    if step < 0 or finger is None:
        return result

    def value(name: str) -> float | None:
        values = traces.get(name)
        if values is None or step >= len(values):
            return None
        array = np.asarray(values)
        if array.ndim < 2 or finger >= array.shape[1]:
            return None
        scalar = float(array[step, finger])
        return scalar if math.isfinite(scalar) else None

    result["contact_force_n"] = value("finger_contact_force")
    result["pad_force_n"] = value("distal_pad_force_n")
    result["pad_force_fraction"] = value("distal_pad_force_fraction")
    taxels = value("distal_active_taxel_count")
    result["active_taxel_count"] = int(taxels) if taxels is not None else None
    return result


def format_joint_monitor_telemetry(
    telemetry: Mapping[str, float | int | str | None],
) -> str:
    """Format a stable one-line terminal status for manual Viewer inspection."""

    def scalar(name: str, unit: str, digits: int = 4) -> str:
        value = telemetry.get(name)
        if value is None:
            return f"{name}=n/a"
        return f"{name}={float(value):.{digits}f}{unit}"

    taxels = telemetry.get("active_taxel_count")
    return " ".join(
        (
            f"[joint-monitor] t={float(telemetry['time_s']):.3f}s",
            f"actuator={telemetry['actuator']}",
            scalar("target_rad", "rad"),
            scalar("actual_rad", "rad"),
            scalar("error_rad", "rad"),
            scalar("contact_force_n", "N", 3),
            scalar("pad_force_n", "N", 3),
            scalar("pad_force_fraction", "", 3),
            f"active_taxel_count={taxels if taxels is not None else 'n/a'}",
        )
    )


def actual_grasp_pose_telemetry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: Mapping[str, Any],
    stable_window_median_rad: Sequence[float] | None = None,
) -> dict[str, dict[str, float]] | None:
    """Return lock-sample and stable-window qpos beside nominal/preload values."""

    if int(config.get("schema_version", 1)) < 9:
        return None
    nominal = config["grasp_pose"]["nominal_joint_qpos_rad"]
    preload = config["control"]["contact_preload_targets_rad"]
    stable = None
    if stable_window_median_rad is not None:
        stable = np.asarray(stable_window_median_rad, dtype=np.float64)
        if stable.shape != (len(ACTIVE_ACTUATORS),) or not np.isfinite(stable).all():
            raise ValueError(
                "stable_window_median_rad must contain eight finite values"
            )
    rows: dict[str, dict[str, float]] = {}
    for active_index, name in enumerate(ACTIVE_ACTUATORS):
        actuator_id = int(model.actuator(name).id)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        actual = float(data.qpos[qpos_adr])
        nominal_value = float(nominal[name])
        preload_value = float(preload[name])
        rows[name] = {
            "nominal_actual_qpos_rad": nominal_value,
            # Retain the old key as a compatibility alias for the instantaneous
            # post-step state while naming the two measurements explicitly.
            "measured_qpos_rad": actual,
            "lock_sample_qpos_rad": actual,
            "stable_window_median_qpos_rad": (
                float(stable[active_index]) if stable is not None else actual
            ),
            "nominal_error_rad": actual - nominal_value,
            "contact_preload_command_rad": preload_value,
            "preload_tracking_error_rad": preload_value - actual,
        }
    return rows


def format_actual_grasp_pose_telemetry(
    rows: Mapping[str, Mapping[str, float]],
) -> str:
    """Format an auditable multiline snapshot printed at ``grasp_lock``."""

    lines = [
        "[actual-grasp-pose] 250 ms median and instantaneous grasp_lock qpos "
        "(preload is diagnostic, not pose evidence):"
    ]
    for name in ACTIVE_ACTUATORS:
        row = rows[name]
        lines.append(
            "  "
            f"{name}: stable_median="
            f"{float(row['stable_window_median_qpos_rad']):.6f} rad, "
            f"lock_sample={float(row['lock_sample_qpos_rad']):.6f} rad, "
            f"actual={float(row['measured_qpos_rad']):.6f} rad, "
            f"nominal={float(row['nominal_actual_qpos_rad']):.6f} rad, "
            f"nominal_error={float(row['nominal_error_rad']):+.6f} rad, "
            f"preload={float(row['contact_preload_command_rad']):.6f} rad, "
            f"preload_error={float(row['preload_tracking_error_rad']):+.6f} rad"
        )
    return "\n".join(lines)


def compare_reference_trace(
    actual: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray]
) -> tuple[bool, list[str]]:
    """Compare every persisted field except renderer sampling metadata."""

    ignored = {"video_frame_steps"}
    actual_names = set(actual) - ignored
    reference_names = set(reference) - ignored
    mismatches = sorted(actual_names ^ reference_names)
    for name in sorted(actual_names & reference_names):
        if not np.array_equal(actual[name], reference[name]):
            mismatches.append(name)
    return not mismatches, mismatches


def _set_default_contact_visualization(handle: Any) -> None:
    """Enable native contact point/force flags when exposed by the Viewer."""

    option = getattr(handle, "opt", None)
    flags = getattr(option, "flags", None)
    if flags is None:
        return
    for flag in (
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
    ):
        flags[int(flag)] = 1


def _initialize_draggable_camera(
    handle: Any,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """Initialize one free camera without later overwriting mouse input.

    The named ``three_finger_camera`` is a model-fixed target-body camera.
    Binding the passive Viewer to it makes the normal orbit/pan gestures look
    unresponsive because fixed-camera pose is recomputed from the model.  A
    free camera remains owned by the Viewer thread and can be manipulated by
    the mouse; centering its initial look-at point on the cube retains a useful
    grasp-focused composition.

    Call this exactly once while holding ``handle.lock()``.  In particular it
    must not run in the frame loop, where it would erase user camera changes.
    """

    camera = getattr(handle, "cam", None)
    if camera is None:
        return
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    try:
        cube_body_id = int(model.body("three_finger_cube").id)
    except KeyError:
        return
    lookat = np.asarray(data.xpos[cube_body_id], dtype=np.float64)
    if lookat.shape == (3,) and np.isfinite(lookat).all():
        camera.lookat[:] = lookat


def _alignment_axes(gravity: np.ndarray) -> np.ndarray:
    down = np.asarray(gravity, dtype=np.float64)
    norm = float(np.linalg.norm(down))
    if norm <= 1e-12:
        return np.eye(3)
    vertical = -down / norm
    seed = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, vertical))) > 0.9:
        seed = np.asarray([0.0, 1.0, 0.0])
    axis_x = seed - np.dot(seed, vertical) * vertical
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(vertical, axis_x)
    return np.column_stack((axis_x, axis_y, vertical))


def _update_alignment_markers(
    handle: Any,
    model: mujoco.MjModel,
    traces: Mapping[str, np.ndarray],
    step: int,
    *,
    visible: bool,
) -> None:
    """Append v4 contact centroids and the configured 5 mm height band.

    Clearing ``user_scn`` is deliberately owned by
    :func:`_update_viewer_markers`.  Keeping every specialized renderer
    append-only prevents a later overlay from erasing an earlier one (most
    notably the monitored joint axis).
    """

    scene = getattr(handle, "user_scn", None)
    if (
        scene is None
        or not hasattr(scene, "maxgeom")
        or not hasattr(scene, "geoms")
    ):
        return
    required = {
        "target_face_contact_centroid_world_m",
        "target_face_contact_centroid_valid",
    }
    if not visible or not required.issubset(traces) or step < 0:
        return
    centroids = np.asarray(
        traces["target_face_contact_centroid_world_m"][step], dtype=np.float64
    )
    valid = np.asarray(
        traces["target_face_contact_centroid_valid"][step], dtype=bool
    )
    colors = (
        np.asarray([0.95, 0.15, 0.15, 1.0]),
        np.asarray([0.15, 0.85, 0.25, 1.0]),
        np.asarray([0.15, 0.35, 0.95, 1.0]),
    )
    index = int(scene.ngeom)
    for finger in range(3):
        if not valid[finger] or not np.isfinite(centroids[finger]).all():
            continue
        if index >= int(scene.maxgeom):
            return
        mujoco.mjv_initGeom(
            scene.geoms[index],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.asarray([0.002, 0.0, 0.0]),
            pos=centroids[finger],
            mat=np.eye(3).reshape(-1),
            rgba=colors[finger],
        )
        index += 1
    if not np.all(valid) or not np.isfinite(centroids).all():
        scene.ngeom = index
        return

    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    basis = _alignment_axes(gravity)
    vertical = basis[:, 2]
    mean_height_point = np.mean(centroids, axis=0)
    plane_colors = (
        np.asarray([0.95, 0.95, 0.25, 0.20]),
        np.asarray([0.25, 0.85, 0.95, 0.10]),
        np.asarray([0.25, 0.85, 0.95, 0.10]),
    )
    for offset, rgba in zip((0.0, -0.0025, 0.0025), plane_colors):
        if index >= int(scene.maxgeom):
            break
        mujoco.mjv_initGeom(
            scene.geoms[index],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=np.asarray([0.035, 0.035, 0.00015]),
            pos=mean_height_point + offset * vertical,
            mat=basis.reshape(-1),
            rgba=rgba,
        )
        index += 1
    scene.ngeom = index


def _append_connector_marker(
    scene: Any,
    geom_type: mujoco.mjtGeom,
    width: float,
    start: np.ndarray,
    end: np.ndarray,
    rgba: np.ndarray,
) -> bool:
    """Append one connector geom, returning whether it fit in the scene."""

    index = int(scene.ngeom)
    if index >= int(scene.maxgeom):
        return False
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    if (
        start_array.shape != (3,)
        or end_array.shape != (3,)
        or not np.isfinite(start_array).all()
        or not np.isfinite(end_array).all()
    ):
        return False
    mujoco.mjv_initGeom(
        scene.geoms[index],
        type=geom_type,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float64),
    )
    mujoco.mjv_connector(
        scene.geoms[index],
        geom_type,
        float(width),
        start_array,
        end_array,
    )
    scene.ngeom = index + 1
    return True


def _append_v8_closure_markers(
    handle: Any,
    traces: Mapping[str, np.ndarray],
    step: int,
) -> None:
    """Append v8 inward-normal and commanded-closing direction arrows.

    ``closure_cube_outward_normal_world`` stores the canonical cube outward
    normal.  The cyan arrow intentionally shows its negative: the desired
    fingertip closing direction.  The green arrow shows the normalized
    commanded witness velocity, so well-aligned arrows point the same way.
    """

    scene = getattr(handle, "user_scn", None)
    if (
        scene is None
        or not hasattr(scene, "maxgeom")
        or not hasattr(scene, "geoms")
        or step < 0
    ):
        return
    required = {
        "closure_witness_world_m",
        "closure_command_velocity_world_m_s",
        "closure_cube_outward_normal_world",
        "closure_alignment_valid",
    }
    if not required.issubset(traces):
        return
    try:
        witness = np.asarray(
            traces["closure_witness_world_m"][step], dtype=np.float64
        )
        velocity = np.asarray(
            traces["closure_command_velocity_world_m_s"][step],
            dtype=np.float64,
        )
        outward = np.asarray(
            traces["closure_cube_outward_normal_world"][step],
            dtype=np.float64,
        )
        valid = np.asarray(
            traces["closure_alignment_valid"][step], dtype=bool
        )
    except (IndexError, TypeError, ValueError):
        return
    if (
        witness.shape != (len(ACTIVE_FINGERS), 3)
        or velocity.shape != witness.shape
        or outward.shape != witness.shape
        or valid.shape != (len(ACTIVE_FINGERS),)
    ):
        return

    desired_rgba = np.asarray([0.10, 0.75, 1.0, 0.95])
    command_rgba = np.asarray([0.20, 1.0, 0.25, 0.95])
    for finger in range(len(ACTIVE_FINGERS)):
        if not valid[finger] or not np.isfinite(witness[finger]).all():
            continue
        normal_norm = float(np.linalg.norm(outward[finger]))
        velocity_norm = float(np.linalg.norm(velocity[finger]))
        if (
            not math.isfinite(normal_norm)
            or not math.isfinite(velocity_norm)
            or normal_norm <= 1e-12
            or velocity_norm <= 1e-12
        ):
            continue
        desired = -outward[finger] / normal_norm
        command = velocity[finger] / velocity_norm
        # The desired arrow is slightly longer so it remains visible when the
        # commanded direction is nearly perfectly aligned with it.
        if not _append_connector_marker(
            scene,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.0012,
            witness[finger],
            witness[finger] + 0.014 * desired,
            desired_rgba,
        ):
            return
        if not _append_connector_marker(
            scene,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.0018,
            witness[finger],
            witness[finger] + 0.010 * command,
            command_rgba,
        ):
            return


def _operation_start_step(
    traces: Mapping[str, np.ndarray], step: int
) -> int | None:
    """Resolve the operation start for both completed and live traces."""

    if step < 0:
        return None
    recorded = traces.get("manipulation_start_step")
    if recorded is not None:
        try:
            value = int(np.asarray(recorded).reshape(()))
        except (TypeError, ValueError):
            value = -1
        if 0 <= value <= step:
            return value
    states = traces.get("control_state")
    if states is None:
        return None
    values = np.asarray(states)
    if values.ndim != 1 or step >= len(values):
        return None
    live_prefix = values[: step + 1].astype(str)
    matches = np.flatnonzero(live_prefix == "MANIPULATE")
    return int(matches[0]) if len(matches) else None


def _append_v8_operation_path_markers(
    handle: Any,
    model: mujoco.MjModel,
    traces: Mapping[str, np.ndarray],
    step: int,
) -> None:
    """Append a 2 mm vertical corridor and the sampled operation path."""

    scene = getattr(handle, "user_scn", None)
    if (
        scene is None
        or not hasattr(scene, "maxgeom")
        or not hasattr(scene, "geoms")
    ):
        return
    start_step = _operation_start_step(traces, step)
    cube_positions = traces.get("cube_pos")
    if start_step is None or cube_positions is None:
        return
    positions = np.asarray(cube_positions, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or step >= len(positions)
        or not np.isfinite(positions[start_step : step + 1]).all()
    ):
        return

    origin = positions[start_step]
    vertical = _alignment_axes(np.asarray(model.opt.gravity, dtype=np.float64))[:, 2]
    corridor_end = origin + 0.012 * vertical
    if not _append_connector_marker(
        scene,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        0.002,
        origin,
        corridor_end,
        np.asarray([0.15, 0.70, 1.0, 0.16]),
    ):
        return

    # A decimated polyline shows the actual path without exhausting user_scn
    # during a two-second, 1 kHz manipulation.
    point_count = min(49, step - start_step + 1)
    sample_steps = np.unique(
        np.linspace(start_step, step, num=point_count, dtype=np.int64)
    )
    path_rgba = np.asarray([1.0, 0.45, 0.10, 0.95])
    for first, second in zip(sample_steps[:-1], sample_steps[1:]):
        start = positions[int(first)]
        end = positions[int(second)]
        if float(np.linalg.norm(end - start)) <= 1e-12:
            continue
        if not _append_connector_marker(
            scene,
            mujoco.mjtGeom.mjGEOM_LINE,
            0.0015,
            start,
            end,
            path_rgba,
        ):
            return


def v14_contact_overlay_state(
    traces: Mapping[str, np.ndarray],
    step: int,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the force/risk state rendered by the schema-v14 Viewer.

    This deliberately consumes only persisted, post-step observations.  It is
    therefore identical for live physics and state replay and cannot feed a
    Viewer-side estimate back into the controller.
    """

    required = {
        "contact_force_target_n",
        "contact_force_filtered_n",
        "target_face_effective",
        "target_face_force_purity",
        "contact_loss_run_steps",
        "contact_progress_frozen",
        "contact_recovery_active",
    }
    if step < 0 or not required.issubset(traces):
        return None
    try:
        target = np.asarray(traces["contact_force_target_n"][step], dtype=np.float64)
        actual = np.asarray(
            traces["contact_force_filtered_n"][step], dtype=np.float64
        )
        effective = np.asarray(traces["target_face_effective"][step], dtype=bool)
        purity = np.asarray(
            traces["target_face_force_purity"][step], dtype=np.float64
        )
        loss_steps = np.asarray(
            traces["contact_loss_run_steps"][step], dtype=np.int64
        )
        frozen = bool(np.asarray(traces["contact_progress_frozen"])[step])
        recovering = bool(np.asarray(traces["contact_recovery_active"])[step])
    except (IndexError, TypeError, ValueError):
        return None
    expected = (len(ACTIVE_FINGERS),)
    if any(
        value.shape != expected
        for value in (target, actual, effective, purity, loss_steps)
    ) or not np.isfinite(np.concatenate((target, actual, purity))).all():
        return None

    force_risk_n = 0.1
    minimum_purity = 0.95
    if config is not None and int(config.get("schema_version", 0)) >= 14:
        force_risk_n = float(
            config.get("contact_feedback", {}).get("force_risk_n", force_risk_n)
        )
        minimum_purity = float(
            config.get("control_protocol", {})
            .get("grasp_gate", {})
            .get("min_target_force_fraction", minimum_purity)
        )
    control_state = ""
    if "control_state" in traces:
        try:
            control_state = str(np.asarray(traces["control_state"])[step])
        except (IndexError, TypeError, ValueError):
            control_state = ""
    operation_active = control_state in {"MANIPULATE", "HOLD", "ABORT"}
    raw_finger_risk = (
        (actual < force_risk_n)
        | (purity < minimum_purity)
        | ~effective
        | (loss_steps > 0)
    )
    finger_risk = raw_finger_risk & operation_active
    forbidden = False
    if "forbidden_contact" in traces:
        forbidden = bool(np.asarray(traces["forbidden_contact"])[step])
    nondistal = False
    if "active_nondistal_force_n" in traces:
        values = np.asarray(
            traces["active_nondistal_force_n"][step], dtype=np.float64
        )
        nondistal = bool(np.any(values >= 0.05))
    return {
        "target_force_n": target,
        "filtered_force_n": actual,
        "force_ratio": np.divide(
            actual,
            target,
            out=np.zeros_like(actual),
            where=target > 1e-12,
        ),
        "target_face_effective": effective,
        "target_face_force_purity": purity,
        "contact_loss_run_steps": loss_steps,
        "control_state": control_state,
        "operation_active": operation_active,
        "finger_risk": finger_risk,
        "progress_frozen": frozen,
        "recovery_active": recovering,
        "forbidden_contact": forbidden,
        "active_nondistal_contact": nondistal,
        "global_risk": bool(np.any(finger_risk) or forbidden or nondistal),
    }


def format_v14_contact_overlay_telemetry(state: Mapping[str, Any]) -> str:
    """Format the three force gauges and controller contact-risk state."""

    actual = np.asarray(state["filtered_force_n"], dtype=np.float64)
    target = np.asarray(state["target_force_n"], dtype=np.float64)
    risk = np.asarray(state["finger_risk"], dtype=bool)
    force_text = " ".join(
        f"{finger}={actual[index]:.3f}/{target[index]:.3f}N"
        f"{'!' if risk[index] else ''}"
        for index, finger in enumerate(ACTIVE_FINGERS)
    )
    return (
        "[contact-plan] "
        f"{force_text} frozen={int(bool(state['progress_frozen']))} "
        f"recovery={int(bool(state['recovery_active']))} "
        f"risk={int(bool(state['global_risk']))}"
    )


_FACE_OUTWARD_LOCAL = {
    "+X": np.asarray((1.0, 0.0, 0.0)),
    "-X": np.asarray((-1.0, 0.0, 0.0)),
    "+Y": np.asarray((0.0, 1.0, 0.0)),
    "-Y": np.asarray((0.0, -1.0, 0.0)),
    "+Z": np.asarray((0.0, 0.0, 1.0)),
    "-Z": np.asarray((0.0, 0.0, -1.0)),
}


def _append_v14_plan_force_risk_markers(
    handle: Any,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    traces: Mapping[str, np.ndarray],
    step: int,
    config: Mapping[str, Any] | None,
) -> None:
    """Append the desired path, three force gauges and a contact-risk beacon."""

    scene = getattr(handle, "user_scn", None)
    if (
        scene is None
        or not hasattr(scene, "maxgeom")
        or not hasattr(scene, "geoms")
        or step < 0
    ):
        return
    state = v14_contact_overlay_state(traces, step, config)
    if state is None:
        return

    start_step = _operation_start_step(traces, step)
    cube_positions = traces.get("cube_pos")
    planned_deltas = traces.get(
        "manipulation_plan_desired_cube_position_delta_m"
    )
    if start_step is not None and cube_positions is not None and planned_deltas is not None:
        positions = np.asarray(cube_positions, dtype=np.float64)
        deltas = np.asarray(planned_deltas, dtype=np.float64)
        if (
            positions.ndim == 2
            and positions.shape[1] == 3
            and start_step < len(positions)
            and deltas.ndim == 2
            and deltas.shape[1] == 3
            and len(deltas) >= 2
            and np.isfinite(positions[start_step]).all()
            and np.isfinite(deltas).all()
        ):
            anchor = positions[start_step]
            planned_world = anchor[None, :] + deltas
            for knot, (first, second) in enumerate(
                zip(planned_world[:-1], planned_world[1:])
            ):
                before = int(scene.ngeom)
                if not _append_connector_marker(
                    scene,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    0.0022,
                    first,
                    second,
                    np.asarray([0.15, 0.95, 1.0, 0.90]),
                ):
                    return
                scene.geoms[before].label = f"planned_cube_path_{knot:02d}"

    centroids = traces.get("target_face_contact_centroid_world_m")
    centroid_valid = traces.get("target_face_contact_centroid_valid")
    if centroids is not None and centroid_valid is not None:
        points = np.asarray(centroids[step], dtype=np.float64)
        valid = np.asarray(centroid_valid[step], dtype=bool)
        if points.shape == (len(ACTIVE_FINGERS), 3) and valid.shape == (
            len(ACTIVE_FINGERS),
        ):
            try:
                cube_geom_id = int(model.geom("three_finger_cube_geom").id)
                cube_rotation = np.asarray(
                    data.geom_xmat[cube_geom_id], dtype=np.float64
                ).reshape(3, 3)
            except KeyError:
                cube_rotation = np.eye(3)
            target_faces = (
                config.get("contact_topology", {}).get("target_faces", {})
                if config is not None
                else {}
            )
            for finger_index, finger in enumerate(ACTIVE_FINGERS):
                if not valid[finger_index] or not np.isfinite(points[finger_index]).all():
                    continue
                risky = bool(state["finger_risk"][finger_index])
                recovering = bool(state["recovery_active"] or state["progress_frozen"])
                rgba = np.asarray(
                    [1.0, 0.12, 0.05, 1.0]
                    if risky
                    else (
                        [1.0, 0.72, 0.05, 1.0]
                        if recovering
                        else [0.10, 1.0, 0.30, 1.0]
                    )
                )
                ratio = float(np.clip(state["force_ratio"][finger_index], 0.0, 2.0))
                face = str(target_faces.get(finger, ""))
                outward_local = _FACE_OUTWARD_LOCAL.get(face)
                if outward_local is None:
                    direction = _alignment_axes(
                        np.asarray(model.opt.gravity, dtype=np.float64)
                    )[:, 2]
                else:
                    direction = -(cube_rotation @ outward_local)
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm <= 1e-12:
                    continue
                direction /= direction_norm
                marker_index = int(scene.ngeom)
                if not _append_connector_marker(
                    scene,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.0014 + 0.0005 * min(ratio, 1.0),
                    points[finger_index],
                    points[finger_index]
                    + (0.004 + 0.006 * min(ratio, 1.5)) * direction,
                    rgba,
                ):
                    return
                scene.geoms[marker_index].label = (
                    f"{finger}_force_"
                    f"{float(state['filtered_force_n'][finger_index]):.3f}_of_"
                    f"{float(state['target_force_n'][finger_index]):.3f}_N"
                )

    if bool(state["global_risk"]):
        try:
            cube_body_id = int(model.body("three_finger_cube").id)
        except KeyError:
            return
        index = int(scene.ngeom)
        if index >= int(scene.maxgeom):
            return
        up = _alignment_axes(np.asarray(model.opt.gravity, dtype=np.float64))[:, 2]
        mujoco.mjv_initGeom(
            scene.geoms[index],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.asarray([0.004, 0.0, 0.0]),
            pos=np.asarray(data.xpos[cube_body_id], dtype=np.float64) + 0.065 * up,
            mat=np.eye(3).reshape(-1),
            rgba=np.asarray([1.0, 0.05, 0.02, 0.95]),
        )
        scene.geoms[index].label = "contact_risk"
        scene.ngeom = index + 1


def _append_v12_contact_point_markers(
    handle: Any,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    traces: Mapping[str, np.ndarray],
    step: int,
) -> None:
    """Append frozen targets, measured centroids and tangent-error lines.

    Both point sets are persisted in cube-local coordinates.  Transforming
    them with the display model's current cube geom pose keeps markers attached
    to a rotated/moving free cube without feeding Viewer state back into the
    independently integrated physics session.
    """

    scene = getattr(handle, "user_scn", None)
    if (
        scene is None
        or not hasattr(scene, "maxgeom")
        or not hasattr(scene, "geoms")
        or step < 0
    ):
        return
    required = {
        "target_contact_points_cube_local_m",
        "target_face_contact_centroid_cube_local_m",
        "target_face_contact_centroid_valid",
        "target_contact_point_within_radius",
    }
    if not required.issubset(traces):
        return
    try:
        target_local = np.asarray(
            traces["target_contact_points_cube_local_m"], dtype=np.float64
        )
        actual_local = np.asarray(
            traces["target_face_contact_centroid_cube_local_m"][step],
            dtype=np.float64,
        )
        valid = np.asarray(
            traces["target_face_contact_centroid_valid"][step], dtype=bool
        )
        within = np.asarray(
            traces["target_contact_point_within_radius"][step], dtype=bool
        )
    except (IndexError, TypeError, ValueError):
        return
    if (
        target_local.shape != (len(ACTIVE_FINGERS), 3)
        or actual_local.shape != target_local.shape
        or valid.shape != (len(ACTIVE_FINGERS),)
        or within.shape != valid.shape
        or not np.isfinite(target_local).all()
    ):
        return
    try:
        cube_geom_id = int(model.geom("three_finger_cube_geom").id)
    except KeyError:
        return
    cube_position = np.asarray(data.geom_xpos[cube_geom_id], dtype=np.float64)
    cube_rotation = np.asarray(
        data.geom_xmat[cube_geom_id], dtype=np.float64
    ).reshape(3, 3)
    target_world = (cube_rotation @ target_local.T).T + cube_position
    actual_world = (cube_rotation @ actual_local.T).T + cube_position
    colors = (
        np.asarray([0.95, 0.15, 0.15, 1.0]),
        np.asarray([0.15, 0.85, 0.25, 1.0]),
        np.asarray([0.15, 0.35, 0.95, 1.0]),
    )

    for finger_index, color in enumerate(colors):
        index = int(scene.ngeom)
        if index >= int(scene.maxgeom):
            return
        target_rgba = color.copy()
        target_rgba[3] = 0.42
        mujoco.mjv_initGeom(
            scene.geoms[index],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.asarray([0.002, 0.0, 0.0]),
            pos=target_world[finger_index],
            mat=np.eye(3).reshape(-1),
            rgba=target_rgba,
        )
        scene.ngeom = index + 1
        if (
            not valid[finger_index]
            or not np.isfinite(actual_local[finger_index]).all()
        ):
            continue
        index = int(scene.ngeom)
        if index >= int(scene.maxgeom):
            return
        actual_rgba = np.asarray(
            [0.30, 1.0, 0.35, 1.0]
            if within[finger_index]
            else [1.0, 0.20, 0.05, 1.0]
        )
        mujoco.mjv_initGeom(
            scene.geoms[index],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.asarray([0.0012, 0.0, 0.0]),
            pos=actual_world[finger_index],
            mat=np.eye(3).reshape(-1),
            rgba=actual_rgba,
        )
        scene.ngeom = index + 1
        if float(
            np.linalg.norm(
                actual_world[finger_index] - target_world[finger_index]
            )
        ) <= 1e-12:
            continue
        if not _append_connector_marker(
            scene,
            mujoco.mjtGeom.mjGEOM_LINE,
            0.0015,
            target_world[finger_index],
            actual_world[finger_index],
            actual_rgba,
        ):
            return


def _append_joint_axis_marker(
    handle: Any,
    data: mujoco.MjData,
    binding: JointMonitorBinding | None,
) -> None:
    """Append a yellow arrow along the monitored joint's current world axis."""

    if binding is None:
        return
    scene = getattr(handle, "user_scn", None)
    if scene is None or not hasattr(scene, "maxgeom") or not hasattr(scene, "geoms"):
        return
    index = int(scene.ngeom)
    if index >= int(scene.maxgeom):
        return
    anchor = np.asarray(data.xanchor[binding.joint_id], dtype=np.float64)
    axis = np.asarray(data.xaxis[binding.joint_id], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(anchor).all() or not np.isfinite(axis).all() or norm <= 1e-12:
        return
    axis /= norm
    start = anchor - 0.018 * axis
    end = anchor + 0.018 * axis
    mujoco.mjv_initGeom(
        scene.geoms[index],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).reshape(-1),
        rgba=np.asarray([1.0, 0.72, 0.05, 0.95]),
    )
    mujoco.mjv_connector(
        scene.geoms[index],
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.0015,
        start,
        end,
    )
    scene.ngeom = index + 1


def _append_joint_pair_markers(
    handle: Any,
    data: mujoco.MjData,
    binding: JointPairBinding | None,
    *,
    visible: bool,
    status: str | None = None,
) -> None:
    """Draw two joint anchors, their axes, connector and cube ``+Y`` ray."""

    if binding is None or not visible:
        return
    scene = getattr(handle, "user_scn", None)
    if scene is None or not hasattr(scene, "maxgeom") or not hasattr(scene, "geoms"):
        return
    try:
        telemetry = joint_pair_telemetry(data, binding)
    except ValueError:
        return
    anchors = (
        np.asarray(telemetry["first_anchor_world_m"], dtype=np.float64),
        np.asarray(telemetry["second_anchor_world_m"], dtype=np.float64),
    )
    colors = (
        np.asarray([1.00, 0.12, 0.72, 1.0]),
        np.asarray([0.10, 0.88, 1.00, 1.0]),
    )
    names = (binding.first_joint_name, binding.second_joint_name)
    for joint_id, anchor, color, name in zip(
        (binding.first_joint_id, binding.second_joint_id),
        anchors,
        colors,
        names,
    ):
        index = int(scene.ngeom)
        if index >= int(scene.maxgeom):
            return
        mujoco.mjv_initGeom(
            scene.geoms[index],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.asarray([0.0024, 0.0, 0.0]),
            pos=anchor,
            mat=np.eye(3).reshape(-1),
            rgba=color,
        )
        scene.geoms[index].label = name
        scene.ngeom = index + 1

        axis = np.asarray(data.xaxis[joint_id], dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis).all() or norm <= 1e-12:
            continue
        axis /= norm
        marker_index = int(scene.ngeom)
        if not _append_connector_marker(
            scene,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.00125,
            anchor - 0.014 * axis,
            anchor + 0.014 * axis,
            color,
        ):
            return
        scene.geoms[marker_index].label = f"{name}_axis"

    status_colors = {
        "safe": np.asarray([0.10, 0.95, 0.20, 0.95]),
        "warning": np.asarray([1.00, 0.68, 0.05, 0.98]),
        "frozen": np.asarray([1.00, 0.92, 0.08, 1.0]),
        "risk": np.asarray([1.00, 0.28, 0.03, 1.0]),
        "abort": np.asarray([0.95, 0.03, 0.03, 1.0]),
    }
    line_color = status_colors.get(
        str(status), np.asarray([0.98, 0.98, 0.98, 0.95])
    )
    line_index = int(scene.ngeom)
    if not _append_connector_marker(
        scene,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        0.00075,
        anchors[0],
        anchors[1],
        line_color,
    ):
        return
    scene.geoms[line_index].label = (
        f"joint_pair_{float(telemetry['angle_to_cube_y_deg']):.2f}deg"
        + (f"_{status}" if status is not None else "")
    )

    cube_rotation = np.asarray(
        data.xmat[binding.cube_body_id], dtype=np.float64
    ).reshape(3, 3)
    cube_y_world = cube_rotation[:, 1]
    reference_index = int(scene.ngeom)
    if not _append_connector_marker(
        scene,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.00135,
        anchors[0],
        anchors[0] + float(telemetry["length_m"]) * cube_y_world,
        np.asarray([0.10, 0.95, 0.20, 0.95]),
    ):
        return
    scene.geoms[reference_index].label = "cube_+Y_at_joint_pair"


def _append_coordinate_frame_markers(
    handle: Any,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    visible: bool,
) -> None:
    """Append the hand-root and cube body frames in world coordinates.

    The arrows are display-only ``user_scn`` geoms.  Their origins and axes
    come from the display data's compiled body transforms, so the cube frame
    follows the free body while the fixed hand-root frame remains stationary.
    X, Y and Z use the conventional red, green and blue colors respectively.
    """

    if not visible:
        return
    scene = getattr(handle, "user_scn", None)
    if scene is None or not hasattr(scene, "maxgeom") or not hasattr(scene, "geoms"):
        return

    axis_colors = (
        np.asarray([0.95, 0.10, 0.10, 1.0]),
        np.asarray([0.10, 0.90, 0.20, 1.0]),
        np.asarray([0.10, 0.35, 1.00, 1.0]),
    )
    frame_specs = (
        ("hand_root", "left_hand_link", 0.035, 0.0016),
        ("cube", "three_finger_cube", 0.030, 0.0013),
    )
    for frame_label, body_name, axis_length, axis_width in frame_specs:
        try:
            body_id = int(model.body(body_name).id)
        except KeyError:
            continue
        origin = np.asarray(data.xpos[body_id], dtype=np.float64)
        rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        if not np.isfinite(origin).all() or not np.isfinite(rotation).all():
            continue
        for axis_index, (axis_name, rgba) in enumerate(
            zip(("X", "Y", "Z"), axis_colors)
        ):
            marker_index = int(scene.ngeom)
            if not _append_connector_marker(
                scene,
                mujoco.mjtGeom.mjGEOM_ARROW,
                axis_width,
                origin,
                origin + axis_length * rotation[:, axis_index],
                rgba,
            ):
                return
            # Labels are useful when the Viewer label option is enabled, while
            # the colored arrows remain fully legible without text rendering.
            scene.geoms[marker_index].label = f"{frame_label}_{axis_name}"


def _update_viewer_markers(
    handle: Any,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    traces: Mapping[str, np.ndarray],
    step: int,
    *,
    show_alignment: bool,
    joint_monitor: JointMonitorBinding | None,
    joint_pair: JointPairBinding | None = None,
    show_joint_pair: bool = False,
    show_coordinate_frames: bool = False,
    config: Mapping[str, Any] | None = None,
) -> None:
    scene = getattr(handle, "user_scn", None)
    if scene is not None and hasattr(scene, "ngeom"):
        # This is the sole reset point.  Every marker renderer below is
        # append-only, so alignment, v8 overlays and the joint axis coexist.
        scene.ngeom = 0
    # Coordinate frames are intentionally appended first so they remain
    # available even when dense path/contact overlays approach maxgeom.
    _append_coordinate_frame_markers(
        handle,
        model,
        data,
        visible=show_coordinate_frames,
    )
    _append_joint_pair_markers(
        handle,
        data,
        joint_pair,
        visible=show_joint_pair,
        status=joint_pair_overlay_status(traces, step, config),
    )
    _update_alignment_markers(
        handle,
        model,
        traces,
        step,
        visible=show_alignment,
    )
    if show_alignment:
        _append_v8_closure_markers(handle, traces, step)
        _append_v8_operation_path_markers(handle, model, traces, step)
        _append_v12_contact_point_markers(
            handle, model, data, traces, step
        )
        _append_v14_plan_force_risk_markers(
            handle, model, data, traces, step, config
        )
    _append_joint_axis_marker(handle, data, joint_monitor)


def _write_live_output(
    output_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    traces: Mapping[str, np.ndarray],
    *,
    source: ViewerSource,
    overridden: bool,
    reference_match: bool | None,
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"viewer output directory already exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    resolved = resolved_run_config(config, summary)
    if overridden:
        recomputed_status = resolved["experiment_status"]
        status: dict[str, Any] = {
            "classification": "parameter_override_run",
            "passed": bool(recomputed_status["passed"]),
            "hard_constraints_passed": bool(
                recomputed_status["hard_constraints_passed"]
            ),
            "failed_checks": [
                str(value) for value in summary.get("failed_checks", [])
            ],
            "note": (
                "This status was recomputed from a Viewer parameter "
                "override run; no catalog validation status was inherited."
            ),
        }
        stage_status = summary.get("stage_status")
        if isinstance(stage_status, Mapping):
            for name in ("grasp_success", "manipulation_success", "full_success"):
                status[name] = bool(stage_status.get(name, False))
        for name in (
            "success_scope",
            "manipulation_success_required",
            "full_hard_constraints_passed",
        ):
            if name in recomputed_status:
                status[name] = recomputed_status[name]
        resolved["experiment_status"] = status
    write_json(output_dir / "resolved_config.json", resolved)
    np.savez_compressed(output_dir / "trace.npz", **traces)
    write_json(
        output_dir / "result.json",
        {
            "run_kind": (
                "parameter_override_run" if overridden else "live_physics_rerun"
            ),
            "trajectory": source.trajectory,
            "source_config": str(source.config_path),
            "reference_trace": (
                str(source.trace_path) if source.trace_path is not None else None
            ),
            "reference_trace_match": reference_match,
            "config": resolved,
            "summary": summary,
        },
    )


def simulate_in_viewer(
    source: ViewerSource,
    config: dict[str, Any],
    *,
    speed: float = 1.0,
    loop: bool = False,
    start_paused: bool = False,
    output_dir: str | Path | None = None,
    parameter_overridden: bool = False,
    joint_monitor: str | None = None,
    joint_pair: Sequence[str] | None = None,
    pause_at_event: str | None = None,
    show_coordinate_frames: bool = False,
    contact_environment: ContactEnvironmentSpec | None = None,
) -> LiveViewerResult:
    """Run real physics in a passive Viewer using an isolated display state."""

    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("viewer simulation speed must be finite and greater than zero")
    if pause_at_event not in {None, *SUPPORTED_LIVE_PAUSE_EVENTS}:
        choices = ", ".join(sorted(SUPPORTED_LIVE_PAUSE_EVENTS))
        raise ValueError(f"unsupported live Viewer pause event; choose one of: {choices}")
    _require_interactive_gl()
    requested_output = Path(output_dir).expanduser().resolve() if output_dir else None
    if requested_output is not None and requested_output.exists():
        raise FileExistsError(
            f"viewer output directory already exists: {requested_output}"
        )

    candidate_metadata = config.get("candidate_metadata", {})
    upright_rescue = (
        candidate_metadata.get("v14_upright_grasp_self_collision_rescue")
        if isinstance(candidate_metadata, Mapping)
        else None
    )
    pair_alignment_refinement = (
        candidate_metadata.get(
            "v14_index_middle_joint_pair_alignment_refinement"
        )
        if isinstance(candidate_metadata, Mapping)
        else None
    )
    if isinstance(upright_rescue, Mapping) or isinstance(
        pair_alignment_refinement, Mapping
    ):
        # The published reference contains the strict supplemental
        # self-collision and, for alignment refinements, joint-pair trace.
        # Use the same decorator for live physics so Viewer reruns preserve
        # both the hard evaluation and exact trace schema.
        from .tuning.upright_grasp_self_collision_rescue import (
            ActiveFingerSelfCollisionAuditedSession,
        )

        physics_session = (
            SimulationSession(config)
            if contact_environment is None
            else SimulationSession(
                config, contact_environment=contact_environment
            )
        )
        session = ActiveFingerSelfCollisionAuditedSession(physics_session)
    else:
        session = (
            SimulationSession(config)
            if contact_environment is None
            else SimulationSession(
                config, contact_environment=contact_environment
            )
        )
    if pause_at_event == "grasp_lock" and session.controller is None:
        session.close()
        raise ValueError(
            "--pause-at-event grasp_lock requires a live grasp controller "
            "(schema v3 or newer)"
        )
    monitor = resolve_joint_monitor(session.model, config, joint_monitor)
    pair_binding = resolve_joint_pair(session.model, joint_pair)
    display_model, _ = build_model(config)
    if contact_environment is not None:
        display_cube_geom_id = display_model.geom(
            "three_finger_cube_geom"
        ).id
        apply_contact_environment_to_model(
            display_model,
            display_cube_geom_id,
            contact_environment,
        )
    display_data = mujoco.MjData(display_model)
    copy_physics_to_display(
        session.model, session.data, display_model, display_data
    )
    reference = (
        load_replay_trace(source.trace_path)
        if source.trace_path is not None and not parameter_overridden
        else None
    )
    if reference is not None:
        validate_trace_model_binding(session.model, session.info, reference)

    from mujoco import viewer
    from mujoco.glfw import glfw

    state = PlaybackState(
        paused=bool(start_paused),
        looping=bool(loop),
        show_coordinate_frames=bool(show_coordinate_frames),
        show_joint_pair=pair_binding is not None,
    )

    def key_callback(keycode: int) -> None:
        if keycode == glfw.KEY_SPACE:
            state.paused = not state.paused
        elif keycode == glfw.KEY_R:
            state.reset_requested = True
        elif keycode == glfw.KEY_L:
            state.looping = not state.looping
        elif keycode == glfw.KEY_C:
            state.show_contact_alignment = not state.show_contact_alignment
        elif keycode == glfw.KEY_F:
            state.show_coordinate_frames = not state.show_coordinate_frames
        elif keycode == glfw.KEY_J and pair_binding is not None:
            state.show_joint_pair = not state.show_joint_pair

    summary: dict[str, Any] | None = None
    completed_summary: dict[str, Any] | None = None
    reference_match: bool | None = None
    output_written = False
    completed_once = False
    last_display_step = -1
    last_telemetry_bucket = -1
    last_contact_telemetry_bucket = -1
    last_joint_pair_telemetry_bucket = -1
    accumulator = 0.0
    last_wall = time.monotonic()
    timestep = float(session.model.opt.timestep)
    print(
        f"Simulating {source.trajectory!r}: {session.total_steps} x "
        f"{timestep:g} s physics steps at {speed:g}x. Keys: Space pause/resume, "
        "R restart, L toggle loop, C toggle contact/alignment/plan overlays, "
        "F toggle hand/cube coordinate frames, J toggle joint-pair overlay."
    )
    if state.show_coordinate_frames:
        print(
            "Coordinate frames enabled: hand_root and cube; "
            "X=red, Y=green, Z=blue."
        )
    if monitor is not None:
        print(
            f"Monitoring {monitor.actuator_name!r}; the yellow arrow is its "
            "current world-space joint axis."
        )
    if pair_binding is not None:
        print(
            "Joint pair enabled: magenta=first anchor/axis, "
            "cyan=second anchor/axis, white=anchor line, "
            "green=cube +Y reference."
        )
        if int(config.get("schema_version", 0)) >= 15:
            print(
                "Schema-v15 pair line: green=safe, amber=warning, "
                "yellow=frozen, orange=risk, red=abort."
            )
        print(
            format_joint_pair_telemetry(
                joint_pair_telemetry(display_data, pair_binding)
            )
        )
    if int(config.get("schema_version", 0)) >= 14:
        print(
            "Contact-plan overlay enabled: cyan=planned cube path, "
            "orange=actual path, force arrows green/amber/red="
            "healthy/recovering/at-risk."
        )
    try:
        with viewer.launch_passive(
            display_model, display_data, key_callback=key_callback
        ) as handle:
            with handle.lock():
                _initialize_draggable_camera(
                    handle, display_model, display_data
                )
                _set_default_contact_visualization(handle)
                _update_viewer_markers(
                    handle,
                    display_model,
                    display_data,
                    session.traces,
                    last_display_step,
                    show_alignment=state.show_contact_alignment,
                    joint_monitor=monitor,
                    joint_pair=pair_binding,
                    show_joint_pair=state.show_joint_pair,
                    show_coordinate_frames=state.show_coordinate_frames,
                    config=config,
                )
            handle.sync()

            while handle.is_running():
                now = time.monotonic()
                wall_delta = max(0.0, now - last_wall)
                last_wall = now
                if state.reset_requested:
                    session.reset()
                    accumulator = 0.0
                    summary = None
                    last_display_step = -1
                    last_telemetry_bucket = -1
                    last_contact_telemetry_bucket = -1
                    last_joint_pair_telemetry_bucket = -1
                    state.event_pause_triggered = False
                    state.reset_requested = False
                elif not state.paused and not session.complete:
                    accumulator += wall_delta * speed
                    # Every physical millisecond is integrated; rendering may
                    # coalesce several completed samples at high display speed.
                    while accumulator + 1e-15 >= timestep and not session.complete:
                        completed_step = session.advance_one()
                        last_display_step = completed_step.index
                        accumulator -= timestep
                        if (
                            pause_at_event == "grasp_lock"
                            and not state.event_pause_triggered
                            and session.controller is not None
                            and session.controller.grasp_acquisition_step
                            == completed_step.index
                        ):
                            state.paused = True
                            state.event_pause_triggered = True
                            # Discard accumulated display-wall time so Space
                            # resumes from the exact post-lock sample instead
                            # of immediately integrating a burst of steps.
                            accumulator = 0.0
                            print(
                                "Paused at grasp_lock "
                                f"(step {completed_step.index}, "
                                f"t={completed_step.time_s:.3f} s). "
                                "Press Space to continue."
                            )
                            pose_rows = actual_grasp_pose_telemetry(
                                session.model,
                                session.data,
                                config,
                                session.controller.grasp_pose_actual_qpos_rad,
                            )
                            if pose_rows is not None:
                                print(
                                    format_actual_grasp_pose_telemetry(
                                        pose_rows
                                    )
                                )
                            break

                if monitor is not None and last_display_step >= 0:
                    telemetry_bucket = int(
                        math.floor(float(session.data.time) / 0.1 + 1e-9)
                    )
                    if telemetry_bucket != last_telemetry_bucket:
                        print(
                            format_joint_monitor_telemetry(
                                joint_monitor_telemetry(
                                    session.model,
                                    session.data,
                                    session.traces,
                                    last_display_step,
                                    monitor,
                                )
                            )
                        )
                        last_telemetry_bucket = telemetry_bucket

                if pair_binding is not None and last_display_step >= 0:
                    pair_bucket = int(
                        math.floor(float(session.data.time) / 0.1 + 1e-9)
                    )
                    if pair_bucket != last_joint_pair_telemetry_bucket:
                        print(
                            format_joint_pair_telemetry(
                                joint_pair_telemetry(
                                    session.data, pair_binding
                                )
                            )
                        )
                        last_joint_pair_telemetry_bucket = pair_bucket

                if last_display_step >= 0:
                    contact_bucket = int(
                        math.floor(float(session.data.time) / 0.1 + 1e-9)
                    )
                    if contact_bucket != last_contact_telemetry_bucket:
                        contact_state = v14_contact_overlay_state(
                            session.traces, last_display_step, config
                        )
                        if contact_state is not None:
                            print(
                                format_v14_contact_overlay_telemetry(
                                    contact_state
                                )
                            )
                            last_contact_telemetry_bucket = contact_bucket

                if session.complete and summary is None:
                    summary = session.finalize()
                    completed_summary = summary
                    completed_once = True
                    if reference is not None:
                        reference_match, mismatches = compare_reference_trace(
                            session.traces, reference
                        )
                        if not reference_match:
                            raise RuntimeError(
                                "live physics does not match reference trace: "
                                + ", ".join(mismatches)
                            )
                    if requested_output is not None and not output_written:
                        _write_live_output(
                            requested_output,
                            config,
                            summary,
                            session.traces,
                            source=source,
                            overridden=parameter_overridden,
                            reference_match=reference_match,
                        )
                        output_written = True
                    if state.looping:
                        session.reset()
                        accumulator = 0.0
                        summary = None
                        last_display_step = -1
                        last_telemetry_bucket = -1
                        last_contact_telemetry_bucket = -1
                        last_joint_pair_telemetry_bucket = -1
                        state.event_pause_triggered = False
                    else:
                        state.paused = True

                with handle.lock():
                    copy_physics_to_display(
                        session.model,
                        session.data,
                        display_model,
                        display_data,
                    )
                    _update_viewer_markers(
                        handle,
                        display_model,
                        display_data,
                        session.traces,
                        last_display_step,
                        show_alignment=state.show_contact_alignment,
                        joint_monitor=monitor,
                        joint_pair=pair_binding,
                        show_joint_pair=state.show_joint_pair,
                        show_coordinate_frames=state.show_coordinate_frames,
                        config=config,
                    )
                handle.sync()
                time.sleep(1.0 / 120.0)
    finally:
        session.close()

    if not completed_once:
        return LiveViewerResult(3, False, None, None)
    assert completed_summary is not None
    # A looping run may have reset after its last complete cycle.  The latest
    # completed result was already evaluated and, if requested, persisted.
    completed_config = resolved_run_config(config, completed_summary)
    completed_status = completed_config.get("experiment_status")
    passed = bool(
        completed_status.get("passed", completed_summary["passed"])
        if isinstance(completed_status, Mapping)
        else completed_summary["passed"]
    )
    return LiveViewerResult(
        exit_code=0 if passed else 2,
        completed=True,
        summary=completed_summary,
        reference_trace_match=reference_match,
    )


def replay_in_viewer(
    source: ReplaySource,
    *,
    speed: float = 1.0,
    loop: bool = False,
    start_paused: bool = False,
    joint_monitor: str | None = None,
    joint_pair: Sequence[str] | None = None,
    show_coordinate_frames: bool = False,
) -> None:
    """Open a passive MuJoCo Viewer and replay a recorded trajectory."""

    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("viewer playback speed must be finite and greater than zero")
    _require_interactive_gl()

    config = load_config(source.config_path)
    model, info = build_model(config)
    monitor = resolve_joint_monitor(model, config, joint_monitor)
    pair_binding = resolve_joint_pair(model, joint_pair)
    trace = load_replay_trace(source.trace_path)
    validate_trace_model_binding(model, info, trace)
    data = mujoco.MjData(model)

    # Importing the GUI module only after validation keeps all headless test and
    # batch paths independent of GLFW initialization.
    from mujoco import viewer
    from mujoco.glfw import glfw

    state = PlaybackState(
        paused=bool(start_paused),
        looping=bool(loop),
        show_coordinate_frames=bool(show_coordinate_frames),
        show_joint_pair=pair_binding is not None,
    )

    def key_callback(keycode: int) -> None:
        if keycode == glfw.KEY_SPACE:
            state.paused = not state.paused
        elif keycode == glfw.KEY_R:
            state.reset_requested = True
        elif keycode == glfw.KEY_L:
            state.looping = not state.looping
        elif keycode == glfw.KEY_C:
            state.show_contact_alignment = not state.show_contact_alignment
        elif keycode == glfw.KEY_F:
            state.show_coordinate_frames = not state.show_coordinate_frames
        elif keycode == glfw.KEY_J and pair_binding is not None:
            state.show_joint_pair = not state.show_joint_pair

    times = np.asarray(trace["time"], dtype=np.float64)
    origin = float(times[0])
    duration = float(times[-1] - origin)
    elapsed = 0.0
    last_telemetry_bucket = -1
    last_contact_telemetry_bucket = -1
    last_joint_pair_telemetry_bucket = -1
    last_wall = time.monotonic()
    apply_replay_frame(model, data, info, trace, 0)

    print(
        f"Viewing {source.trajectory!r}: {len(times)} frames, "
        f"{duration:.3f} s at {speed:g}x. "
        "Keys: Space pause/resume, R restart, L toggle loop, "
        "C toggle contact/alignment/plan overlays, "
        "F toggle hand/cube coordinate frames, J toggle joint-pair overlay."
    )
    if state.show_coordinate_frames:
        print(
            "Coordinate frames enabled: hand_root and cube; "
            "X=red, Y=green, Z=blue."
        )
    if monitor is not None:
        print(
            f"Monitoring {monitor.actuator_name!r}; the yellow arrow is its "
            "current world-space joint axis."
        )
    if pair_binding is not None:
        print(
            "Joint pair enabled: magenta=first anchor/axis, "
            "cyan=second anchor/axis, white=anchor line, "
            "green=cube +Y reference."
        )
        if int(config.get("schema_version", 0)) >= 15:
            print(
                "Schema-v15 pair line: green=safe, amber=warning, "
                "yellow=frozen, orange=risk, red=abort."
            )
        print(
            format_joint_pair_telemetry(
                joint_pair_telemetry(data, pair_binding)
            )
        )
    if int(config.get("schema_version", 0)) >= 14:
        print(
            "Contact-plan overlay enabled: cyan=planned cube path, "
            "orange=actual path, force arrows green/amber/red="
            "healthy/recovering/at-risk."
        )
    with viewer.launch_passive(model, data, key_callback=key_callback) as handle:
        with handle.lock():
            _initialize_draggable_camera(handle, model, data)
            _set_default_contact_visualization(handle)
        while handle.is_running():
            now = time.monotonic()
            wall_delta = max(0.0, now - last_wall)
            last_wall = now
            if state.reset_requested:
                elapsed = 0.0
                last_telemetry_bucket = -1
                last_contact_telemetry_bucket = -1
                last_joint_pair_telemetry_bucket = -1
                state.reset_requested = False
            elif not state.paused:
                elapsed += wall_delta * speed

            if duration <= 0.0:
                elapsed = 0.0
                state.paused = True
            elif elapsed >= duration:
                if state.looping:
                    elapsed %= duration
                else:
                    elapsed = duration
                    state.paused = True

            frame = min(
                len(times) - 1,
                max(0, int(np.searchsorted(times, origin + elapsed, side="right") - 1)),
            )
            telemetry_bucket = int(
                math.floor(float(trace["time"][frame]) / 0.1 + 1e-9)
            )
            telemetry_line: str | None = None
            contact_telemetry_line: str | None = None
            with handle.lock():
                apply_replay_frame(model, data, info, trace, frame)
                _update_viewer_markers(
                    handle,
                    model,
                    data,
                    trace,
                    frame,
                    show_alignment=state.show_contact_alignment,
                    joint_monitor=monitor,
                    joint_pair=pair_binding,
                    show_joint_pair=state.show_joint_pair,
                    show_coordinate_frames=state.show_coordinate_frames,
                    config=config,
                )
                if (
                    monitor is not None
                    and telemetry_bucket != last_telemetry_bucket
                ):
                    telemetry_line = format_joint_monitor_telemetry(
                        joint_monitor_telemetry(
                            model,
                            data,
                            trace,
                            frame,
                            monitor,
                        )
                    )
                if telemetry_bucket != last_contact_telemetry_bucket:
                    contact_state = v14_contact_overlay_state(
                        trace, frame, config
                    )
                    if contact_state is not None:
                        contact_telemetry_line = (
                            format_v14_contact_overlay_telemetry(contact_state)
                        )
                joint_pair_telemetry_line = None
                if (
                    pair_binding is not None
                    and telemetry_bucket != last_joint_pair_telemetry_bucket
                ):
                    joint_pair_telemetry_line = format_joint_pair_telemetry(
                        joint_pair_telemetry(data, pair_binding)
                    )
            if telemetry_line is not None:
                print(telemetry_line)
                last_telemetry_bucket = telemetry_bucket
            if contact_telemetry_line is not None:
                print(contact_telemetry_line)
                last_contact_telemetry_bucket = telemetry_bucket
            if joint_pair_telemetry_line is not None:
                print(joint_pair_telemetry_line)
                last_joint_pair_telemetry_bucket = telemetry_bucket
            handle.sync()
            time.sleep(1.0 / 120.0)


__all__ = [
    "actual_grasp_pose_telemetry",
    "DEFAULT_V5_JOINT_MONITOR",
    "SUPPORTED_LIVE_PAUSE_EVENTS",
    "JointPairBinding",
    "JointMonitorBinding",
    "LiveViewerResult",
    "MeasuredViewerData",
    "PlaybackState",
    "ReplaySource",
    "ViewerSource",
    "apply_viewer_overrides",
    "apply_replay_frame",
    "compare_reference_trace",
    "copy_physics_to_display",
    "discover_measured_viewer_data",
    "format_actual_grasp_pose_telemetry",
    "format_joint_monitor_telemetry",
    "format_joint_pair_telemetry",
    "format_v14_contact_overlay_telemetry",
    "format_measured_viewer_data",
    "joint_monitor_telemetry",
    "joint_pair_overlay_status",
    "joint_pair_telemetry",
    "load_replay_trace",
    "parse_actuator_overrides",
    "replay_in_viewer",
    "resolve_replay_source",
    "resolve_joint_monitor",
    "resolve_joint_pair",
    "resolve_measured_viewer_source",
    "resolve_viewer_source",
    "simulate_in_viewer",
    "validate_trace_model_binding",
    "v14_contact_overlay_state",
]
