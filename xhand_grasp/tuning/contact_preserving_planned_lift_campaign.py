"""Authenticated, resumable schema-v14 contact-preserving lift campaign.

The v13 down-size campaign is immutable input.  This module authenticates both
its published grasp catalog and all 22 measured grasp records before creating
any output, promotes those measured hand/object pairs into the registered v14
configuration, and owns the power-loss-safe plan/controller search workspace.

The expensive physics boundaries are injectable.  Production defaults always
reacquire a free-body grasp and run final candidates from reset; tests can
exercise resume, ranking and publication without replacing authentication.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..actual_contact_grasp_pose_catalog import (
    authenticated_catalog_artifact_paths,
    authenticate_candidate_result_semantic_sha256,
    bind_candidate_result_semantic_sha256,
    build_campaign_manifest,
    commit_campaign_stage,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from ..artifacts import (
    REPO_ROOT,
    file_sha256,
    implementation_paths,
    json_text,
    write_json,
)
from ..config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, load_config, validate_config
from ..contacts import target_face_contact_centroids
from ..evaluation import face_from_label
from ..experiment import ContactForceTargets, ExperimentDefinition, resolve_experiment
from ..grasp_pose import canonical_sha256
from ..rendering import VideoSettings, probe_video
from ..scene import build_model
from ..simulation import contact_snapshot, run_simulation
from ..trajectory import actuator_target_vector, minimum_jerk
from .actual_contact_manipulation import (
    GraspPhysicsCheckpoint,
    ProbeSpecification,
    manipulation_delta_bounds,
    prepare_grasp_checkpoint,
)
from .contact_constrained_planner import (
    ContactConstrainedPlannerSettings,
    ExtendedProbeSample,
    fit_extended_probe_response,
    materialize_contact_plan_config,
    plan_contact_constrained_trajectory,
    rank_contact_constrained_candidates,
    v14_controller_id,
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
)
from .contact_preserving_joint_refinement import (
    JointRefinementBudget,
    build_joint_refinement_job_specs,
)
from .contact_preserving_candidate_artifacts import (
    authenticate_v14_candidate_artifacts,
    run_or_resume_v14_candidate_artifacts,
)
from .sequential_contact_planner import (
    materialize_sequential_plan_config,
    plan_sequential_contact_trajectory,
)


CAMPAIGN_RESULT_SCHEMA_VERSION = 1
SOURCE_AUDIT_SCHEMA_VERSION = 1
GRASP_RESCUE_REPORT_SCHEMA_VERSION = 1
PAIR_REPORT_SCHEMA_VERSION = 1
PLAN_REPORT_SCHEMA_VERSION = 1
CANDIDATE_REPORT_SCHEMA_VERSION = 1
CATALOG_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821

PRIMARY_SOURCE_CANDIDATE_ID = 13022705530435533
AUXILIARY_SOURCE_CANDIDATE_ID = 13021892800628222
_SHA256 = frozenset("0123456789abcdef")
_FACE_NORMAL_LOCAL = {
    "+X": np.asarray((1.0, 0.0, 0.0)),
    "-X": np.asarray((-1.0, 0.0, 0.0)),
    "+Y": np.asarray((0.0, 1.0, 0.0)),
    "-Y": np.asarray((0.0, -1.0, 0.0)),
    "+Z": np.asarray((0.0, 0.0, 1.0)),
    "-Z": np.asarray((0.0, 0.0, -1.0)),
}


def _campaign(definition: ExperimentDefinition):
    campaign = getattr(definition, "contact_preserving_planned_lift_campaign", None)
    if campaign is None:
        raise ValueError("experiment has no contact-preserving planned-lift campaign")
    return campaign


def _safe_relative(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or Path(value).is_absolute():
        raise RuntimeError(f"{label} is not a safe relative path")
    result = (root / value).resolve()
    if not result.is_relative_to(root) or not result.is_file():
        raise RuntimeError(f"{label} is missing: {value}")
    return result


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256)
    )


@dataclass(frozen=True, slots=True)
class AuthenticatedV13GraspSource:
    candidate_id: int
    edge_m: float
    mapping_mode: str
    config: dict[str, Any]
    summary: dict[str, Any]
    config_path: Path
    trace_path: Path
    config_sha256: str
    trace_sha256: str
    result_semantic_sha256: str
    published_in_catalog: bool

    def descriptor(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "edge_m": self.edge_m,
            "mapping_mode": self.mapping_mode,
            "config_path": str(self.config_path),
            "trace_path": str(self.trace_path),
            "config_sha256": self.config_sha256,
            "trace_sha256": self.trace_sha256,
            "result_semantic_sha256": self.result_semantic_sha256,
            "published_in_catalog": self.published_in_catalog,
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedV13SourceBundle:
    catalog_path: Path
    catalog_sha256: str
    measured_report_path: Path
    measured_report_sha256: str
    published_candidate_ids: tuple[int, ...]
    sources: tuple[AuthenticatedV13GraspSource, ...]
    warm_start_records: tuple[dict[str, Any], ...]

    def descriptor(self) -> dict[str, Any]:
        return {
            "source_audit_schema_version": SOURCE_AUDIT_SCHEMA_VERSION,
            "complete": True,
            "catalog_path": str(self.catalog_path),
            "catalog_sha256": self.catalog_sha256,
            "measured_report_path": str(self.measured_report_path),
            "measured_report_sha256": self.measured_report_sha256,
            "published_success_count": len(self.published_candidate_ids),
            "published_candidate_ids": list(self.published_candidate_ids),
            "measured_success_count": len(self.sources),
            "records": [value.descriptor() for value in self.sources],
            "warm_start_records": copy.deepcopy(list(self.warm_start_records)),
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedV13StaticAnchor:
    """One immutable v13 static contact-pose anchor for an edge/mapping cell."""

    candidate_id: int
    edge_m: float
    mapping_mode: str
    static_pass: bool
    static_rank: tuple[Any, ...]
    candidate_sha256: str
    config: dict[str, Any]

    def descriptor(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "edge_m": self.edge_m,
            "mapping_mode": self.mapping_mode,
            "static_pass": self.static_pass,
            "static_rank": list(self.static_rank),
            "candidate_sha256": self.candidate_sha256,
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedV13RescueAnchors:
    report_path: Path
    report_sha256: str
    anchors: tuple[AuthenticatedV13StaticAnchor, ...]

    def descriptor(self) -> dict[str, Any]:
        return {
            "report_path": str(self.report_path),
            "report_sha256": self.report_sha256,
            "edge_mapping_count": len(self.anchors),
            "anchors": [value.descriptor() for value in self.anchors],
        }


@dataclass(frozen=True, slots=True)
class GraspRescueJob:
    candidate_id: int
    family: str
    group_id: str
    edge_m: float
    mapping_mode: str
    source_candidate_id: int
    local_index: int
    group_budget: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "group_id": self.group_id,
            "edge_m": self.edge_m,
            "mapping_mode": self.mapping_mode,
            "source_candidate_id": self.source_candidate_id,
            "local_index": self.local_index,
            "group_budget": self.group_budget,
        }


def authenticate_v13_grasp_sources(
    definition: ExperimentDefinition,
) -> AuthenticatedV13SourceBundle:
    """Fail closed on the registered v13 catalog and all measured artifacts."""

    campaign = _campaign(definition)
    catalog_path = (REPO_ROOT / campaign.source_grasp_catalog).resolve()
    if (
        not catalog_path.is_relative_to(REPO_ROOT / "artifacts")
        or not catalog_path.is_file()
    ):
        raise FileNotFoundError(catalog_path)
    catalog_sha = file_sha256(catalog_path)
    if catalog_sha != campaign.source_grasp_catalog_sha256:
        raise RuntimeError("registered v13 grasp catalog SHA-256 mismatch")
    # This validates every config/result/trace/video named by the catalog.
    authenticated_catalog_artifact_paths(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    trajectories = catalog.get("trajectories")
    if not isinstance(trajectories, list):
        raise RuntimeError("v13 source catalog has no trajectory list")
    published_ids = tuple(
        int(value["candidate_id"])
        for value in trajectories
        if isinstance(value, Mapping) and value.get("classification") == "success"
    )
    if len(published_ids) != int(campaign.expected_source_grasp_count):
        raise RuntimeError("v13 published grasp count changed")
    if len(set(published_ids)) != len(published_ids):
        raise RuntimeError("v13 source catalog contains duplicate candidate IDs")

    # .../catalogs/target_5/grasp_pose/catalog.json -> campaign root.
    source_root = catalog_path.parents[3]
    # Authenticate the source campaign's own atomic ledger before trusting the
    # measured report.  The v13 publication pass adds storage/provenance fields
    # to report records after their semantic digest was minted, so recomputing
    # that pre-publication digest on the enriched wrapper would be incorrect;
    # the committed report SHA plus every artifact SHA is the authoritative
    # immutable binding here.
    validate_stage_ledger(source_root)
    measured_path = source_root / "measured_grasp_report.json"
    if not measured_path.is_file():
        raise FileNotFoundError(measured_path)
    measured = json.loads(measured_path.read_text(encoding="utf-8"))
    records = measured.get("records") if isinstance(measured, Mapping) else None
    if (
        measured.get("complete") is not True
        or not isinstance(records, list)
        or int(measured.get("candidate_count", -1)) != len(records)
        or int(measured.get("grasp_success_count", -1)) != len(records)
        or len(records) != 22
    ):
        raise RuntimeError("v13 measured grasp report is incomplete or changed")

    sources: list[AuthenticatedV13GraspSource] = []
    seen: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("v13 measured record is not an object")
        candidate_id = int(record.get("candidate_id", -1))
        if candidate_id < 0 or candidate_id in seen:
            raise RuntimeError("v13 measured candidate IDs are invalid")
        seen.add(candidate_id)
        if (
            record.get("complete") is not True
            or record.get("grasp_success") is not True
            or record.get("measured_grasp_pose_success") is not True
        ):
            raise RuntimeError(f"v13 candidate {candidate_id} is not measured success")
        semantic_sha = record.get("result_semantic_sha256")
        if not _valid_sha256(semantic_sha):
            raise RuntimeError(
                f"v13 measured candidate {candidate_id} lost semantic SHA-256"
            )
        artifact_directory = record.get("artifact_directory")
        artifacts = record.get("artifacts")
        if not isinstance(artifact_directory, str) or not isinstance(artifacts, Mapping):
            raise RuntimeError(f"v13 candidate {candidate_id} lost artifact metadata")
        artifact_root = (source_root / "dynamic" / artifact_directory).resolve()
        if not artifact_root.is_relative_to(source_root / "dynamic"):
            raise RuntimeError("v13 measured artifact path escaped its campaign")
        hashes = artifacts.get("sha256")
        if not isinstance(hashes, Mapping):
            raise RuntimeError(f"v13 candidate {candidate_id} lost artifact hashes")
        paths: dict[str, Path] = {}
        digests: dict[str, str] = {}
        for name in ("resolved_config", "trace"):
            path = _safe_relative(artifact_root, artifacts.get(name), name)
            expected = hashes.get(name)
            actual = file_sha256(path)
            if not _valid_sha256(expected) or expected != actual:
                raise RuntimeError(
                    f"v13 candidate {candidate_id} {name} SHA-256 mismatch"
                )
            paths[name] = path
            digests[name] = actual
        persisted_config = json.loads(paths["resolved_config"].read_text("utf-8"))
        if canonical_sha256(persisted_config) != canonical_sha256(record.get("config")):
            raise RuntimeError(f"v13 candidate {candidate_id} embedded config changed")
        summary = record.get("summary")
        if not isinstance(summary, Mapping):
            raise RuntimeError(f"v13 candidate {candidate_id} has no summary")
        stage = summary.get("stage_status")
        if not isinstance(stage, Mapping) or stage.get("grasp_success") is not True:
            raise RuntimeError(f"v13 candidate {candidate_id} summary lost grasp success")
        config = copy.deepcopy(dict(persisted_config))
        metadata = config.get("candidate_metadata", {})
        mapping_mode = "unknown"
        if isinstance(metadata, Mapping):
            downsize = metadata.get("contact_point_downsize", {})
            if isinstance(downsize, Mapping):
                mapping_mode = str(downsize.get("mapping_mode", "unknown"))
        sources.append(
            AuthenticatedV13GraspSource(
                candidate_id=candidate_id,
                edge_m=float(config["cube"]["edge_m"]),
                mapping_mode=mapping_mode,
                config=config,
                summary=copy.deepcopy(dict(summary)),
                config_path=paths["resolved_config"],
                trace_path=paths["trace"],
                config_sha256=digests["resolved_config"],
                trace_sha256=digests["trace"],
                result_semantic_sha256=semantic_sha,
                published_in_catalog=candidate_id in set(published_ids),
            )
        )
    if not set(published_ids).issubset(seen):
        raise RuntimeError("published v13 grasp is absent from measured report")
    sources.sort(key=lambda value: value.candidate_id)
    warm_start_records: list[dict[str, Any]] = []
    warm_root = (
        source_root
        / "manipulation/local_refinement/set_ee1d71c60ce224cf/candidates"
    )
    for suffix in (8, 9):
        candidate_root = warm_root / f"candidate_1090000000000{suffix:02d}"
        config_path = candidate_root / "resolved_config.json"
        result_path = candidate_root / "result.json"
        if not config_path.is_file() or not result_path.is_file():
            raise RuntimeError("authenticated v13 warm-start candidate is missing")
        warm_config = json.loads(config_path.read_text(encoding="utf-8"))
        warm_result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            warm_result.get("candidate_sha256") != canonical_sha256(warm_config)
            or float(warm_config["cube"]["edge_m"]) != 0.079
        ):
            raise RuntimeError("v13 warm-start candidate config binding changed")
        authenticate_candidate_result_semantic_sha256(
            warm_result, source=f"v13 79mm warm-start {suffix}"
        )
        metrics = warm_result.get("summary", {}).get("metrics", {})
        warm_start_records.append(
            {
                "source_candidate_id": PRIMARY_SOURCE_CANDIDATE_ID,
                "warm_start_alias": f"v13_79mm_local_{suffix:02d}",
                "config_path": str(config_path),
                "config_sha256": file_sha256(config_path),
                "result_path": str(result_path),
                "result_sha256": file_sha256(result_path),
                "result_semantic_sha256": warm_result["result_semantic_sha256"],
                "terminal_delta_rad": copy.deepcopy(
                    warm_config["control"]["manipulation_delta_rad"]
                ),
                "operation_simultaneous_contact_duty": float(
                    metrics.get("operation_target_face_simultaneous_duty", 0.0)
                ),
                "operation_median_lift_m": float(
                    metrics.get("operation_median_lift_m", 0.0)
                ),
                "operation_minimum_lift_m": float(
                    metrics.get("operation_minimum_lift_m", 0.0)
                ),
            }
        )
    return AuthenticatedV13SourceBundle(
        catalog_path=catalog_path,
        catalog_sha256=catalog_sha,
        measured_report_path=measured_path,
        measured_report_sha256=file_sha256(measured_path),
        published_candidate_ids=published_ids,
        sources=tuple(sources),
        warm_start_records=tuple(warm_start_records),
    )


def _static_rank_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    raw = record.get("static_rank", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RuntimeError("v13 static anchor lost its deterministic rank")
    normalized: list[tuple[int, Any]] = []
    for value in raw:
        if isinstance(value, bool):
            normalized.append((0, int(value)))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            normalized.append((1, float(value)))
        else:
            normalized.append((2, str(value)))
    return (
        not bool(record.get("static_pass", False)),
        tuple(normalized),
        int(record.get("candidate_id", 2**63 - 1)),
    )


def authenticate_v13_grasp_rescue_anchors(
    definition: ExperimentDefinition,
    bundle: AuthenticatedV13SourceBundle | None = None,
) -> AuthenticatedV13RescueAnchors:
    """Authenticate one v13 static anchor for every registered edge/mapping.

    The source stage ledger was already authenticated by
    :func:`authenticate_v13_grasp_sources`.  This additional audit prevents a
    rescue implementation from silently considering only the edges which had
    a measured v13 success.
    """

    source_bundle = (
        authenticate_v13_grasp_sources(definition) if bundle is None else bundle
    )
    campaign = _campaign(definition)
    report_path = source_bundle.measured_report_path.parent / "static_downsize_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if (
        payload.get("complete") is not True
        or not isinstance(records, list)
        or int(payload.get("candidate_count", -1)) != len(records)
    ):
        raise RuntimeError("v13 static downsize report is incomplete")
    expected_edges = tuple(round(float(value), 6) for value in campaign.edges_m)
    expected_modes = ("absolute_face_yz", "proportional_face_yz")
    grouped: dict[tuple[float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("v13 static downsize record is not an object")
        edge = round(float(record.get("edge_m", math.nan)), 6)
        mode = str(record.get("mapping_mode", ""))
        if edge not in expected_edges or mode not in expected_modes:
            continue
        config = record.get("config")
        digest = record.get("candidate_sha256")
        if not isinstance(config, Mapping) or not _valid_sha256(digest):
            raise RuntimeError("v13 static anchor lost config identity")
        if canonical_sha256(config) != digest:
            raise RuntimeError("v13 static anchor config SHA-256 mismatch")
        if not math.isclose(float(config["cube"]["edge_m"]), edge, abs_tol=1e-12):
            raise RuntimeError("v13 static anchor edge/config mismatch")
        metadata = config.get("candidate_metadata", {})
        downsize = (
            metadata.get("contact_point_downsize", {})
            if isinstance(metadata, Mapping)
            else {}
        )
        configured_mode = (
            downsize.get("mapping_mode") if isinstance(downsize, Mapping) else None
        )
        if configured_mode != mode:
            raise RuntimeError("v13 static anchor mapping/config mismatch")
        grouped[(edge, mode)].append(record)
    expected_keys = {
        (edge, mode) for edge in expected_edges for mode in expected_modes
    }
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))
        extra = sorted(set(grouped) - expected_keys)
        raise RuntimeError(
            f"v13 static rescue coverage changed; missing={missing}, extra={extra}"
        )
    anchors: list[AuthenticatedV13StaticAnchor] = []
    for key in sorted(expected_keys, key=lambda value: (value[0], value[1])):
        selected = min(grouped[key], key=_static_rank_key)
        config = copy.deepcopy(dict(selected["config"]))
        validate_config(config)
        anchors.append(
            AuthenticatedV13StaticAnchor(
                candidate_id=int(selected["candidate_id"]),
                edge_m=key[0],
                mapping_mode=key[1],
                static_pass=bool(selected.get("static_pass", False)),
                static_rank=tuple(copy.deepcopy(selected.get("static_rank", ()))),
                candidate_sha256=str(selected["candidate_sha256"]),
                config=config,
            )
        )
    return AuthenticatedV13RescueAnchors(
        report_path=report_path,
        report_sha256=file_sha256(report_path),
        anchors=tuple(anchors),
    )


def build_grasp_rescue_jobs(
    definition: ExperimentDefinition,
    bundle: AuthenticatedV13SourceBundle,
    rescue_anchors: AuthenticatedV13RescueAnchors,
) -> tuple[GraspRescueJob, ...]:
    """Return the complete, worker-independent v14 grasp-rescue schedule."""

    campaign = _campaign(definition)
    jobs: list[GraspRescueJob] = []
    for anchor in rescue_anchors.anchors:
        edge_mm = int(round(anchor.edge_m * 1000.0))
        safe_mode = (
            "absolute" if anchor.mapping_mode == "absolute_face_yz" else "proportional"
        )
        group_id = f"edge_{edge_mm:03d}_{safe_mode}"
        count = int(campaign.grasp_rescue_candidates_per_edge_mapping)
        for local_index in range(count):
            identity = {
                "stage": "v14_grasp_rescue",
                "family": "edge_mapping_local",
                "seed": int(campaign.seed),
                "group_id": group_id,
                "source_candidate_id": anchor.candidate_id,
                "local_index": local_index,
            }
            jobs.append(
                GraspRescueJob(
                    candidate_id=_candidate_id(identity),
                    family="edge_mapping_local",
                    group_id=group_id,
                    edge_m=anchor.edge_m,
                    mapping_mode=anchor.mapping_mode,
                    source_candidate_id=anchor.candidate_id,
                    local_index=local_index,
                    group_budget=count,
                )
            )
    source_by_id = {value.candidate_id: value for value in bundle.sources}
    for role, source_id in (
        ("primary_79mm", PRIMARY_SOURCE_CANDIDATE_ID),
        ("auxiliary_force_balanced_76mm", AUXILIARY_SOURCE_CANDIDATE_ID),
    ):
        if source_id not in source_by_id:
            raise RuntimeError(f"v14 priority grasp rescue source {source_id} is missing")
        source = source_by_id[source_id]
        count = int(campaign.priority_grasp_rescue_candidates_per_seed)
        group_id = f"priority_{role}"
        for local_index in range(count):
            identity = {
                "stage": "v14_grasp_rescue",
                "family": role,
                "seed": int(campaign.seed),
                "group_id": group_id,
                "source_candidate_id": source_id,
                "local_index": local_index,
            }
            jobs.append(
                GraspRescueJob(
                    candidate_id=_candidate_id(identity),
                    family=role,
                    group_id=group_id,
                    edge_m=source.edge_m,
                    mapping_mode=source.mapping_mode,
                    source_candidate_id=source_id,
                    local_index=local_index,
                    group_budget=count,
                )
            )
    identifiers = [value.candidate_id for value in jobs]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("v14 grasp rescue candidate ID collision")
    return tuple(jobs)


def build_contact_preserving_planned_lift_manifest(
    config_path: str | Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Bind v14 code plus the complete authenticated v13 source bundle."""

    config = Path(config_path).expanduser().resolve()
    definition = resolve_experiment(load_config(config))
    campaign = _campaign(definition)
    if int(seed) != int(campaign.seed):
        raise ValueError("schema-v14 seed must match its registered campaign seed")
    bundle = authenticate_v13_grasp_sources(definition)
    rescue_anchors = authenticate_v13_grasp_rescue_anchors(definition, bundle)
    base = build_campaign_manifest(
        config,
        seed=int(seed),
        source_paths=tuple(implementation_paths()),
    )
    base.pop("campaign_input_sha256", None)
    base.update(
        {
            "contact_preserving_campaign_manifest_schema_version": 1,
            "contact_preserving_planned_lift_campaign": campaign.as_config(),
            "v13_source_catalog_sha256": bundle.catalog_sha256,
            "v13_measured_grasp_report_path": str(
                bundle.measured_report_path.relative_to(REPO_ROOT)
            ),
            "v13_measured_grasp_report_sha256": bundle.measured_report_sha256,
            "v13_static_downsize_report_path": str(
                rescue_anchors.report_path.relative_to(REPO_ROOT)
            ),
            "v13_static_downsize_report_sha256": rescue_anchors.report_sha256,
            "v13_static_rescue_anchors": [
                value.descriptor() for value in rescue_anchors.anchors
            ],
            "v13_measured_source_records": [
                value.descriptor() for value in bundle.sources
            ],
        }
    )
    return {**base, "campaign_input_sha256": canonical_sha256(base)}


def _minimum_jerk_waypoints(terminal: Mapping[str, Any]) -> dict[str, list[float]]:
    fraction = np.linspace(0.0, 1.0, 21)
    # A sampled global minimum-jerk curve has a 0.0935 maximum progress jump;
    # with the real v14 actuator headroom a useful 0.52-rad endpoint would
    # therefore violate the per-node 0.04-rad trust region.  The C2 runtime
    # interpolator supplies zero endpoint velocity/acceleration, so a sampled
    # cubic smoothstep is the least-distorting monotone knot distribution that
    # keeps these authenticated nonlinear warm starts inside the node bound
    # (maximum progress jump ~= 0.07475).
    profile = 3.0 * fraction**2 - 2.0 * fraction**3
    return {
        name: (profile * float(terminal[name])).tolist() for name in ACTIVE_ACTUATORS
    }


def _plan_config_from_terminal(terminal: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "profile": "piecewise_quintic_minimum_jerk",
        "duration_s": 3.0,
        "knot_times_s": np.linspace(0.0, 3.0, 21).tolist(),
        "actuator_waypoints_rad": _minimum_jerk_waypoints(terminal),
        "desired_cube_position_delta_m": [
            [0.0, 0.0, float(value)]
            for value in 0.011
            * (10.0 * np.linspace(0.0, 1.0, 21) ** 3
               - 15.0 * np.linspace(0.0, 1.0, 21) ** 4
               + 6.0 * np.linspace(0.0, 1.0, 21) ** 5)
        ],
        "desired_cube_rotation_vector_rad": [[0.0, 0.0, 0.0] for _ in range(21)],
        "max_knot_delta_rad": 0.04,
        "trust_region_backtracks": 4,
    }
    return {"plan_id": canonical_sha256(payload), **payload}


def _force_targets_from_source(
    source: AuthenticatedV13GraspSource,
    definition: ExperimentDefinition,
) -> dict[str, Any]:
    campaign = _campaign(definition)
    with np.load(source.trace_path, allow_pickle=False) as trace:
        start = int(np.asarray(trace["grasp_stable_window_start_step"]).reshape(-1)[0])
        end = int(np.asarray(trace["grasp_stable_window_end_step"]).reshape(-1)[0])
        face_force = np.asarray(trace["distal_face_force_n"], dtype=np.float64)
    if start < 0 or end < start or end >= face_force.shape[0]:
        raise RuntimeError(f"v13 candidate {source.candidate_id} stable window is invalid")
    targets: dict[str, float] = {}
    faces = source.config["contact_topology"]["target_faces"]
    for finger_index, finger in enumerate(ACTIVE_FINGERS):
        face_index = int(face_from_label(str(faces[finger])))
        median = float(np.median(face_force[start : end + 1, finger_index, face_index]))
        targets[finger] = float(
            np.clip(
                median,
                campaign.force_target_minimum_n,
                campaign.force_target_maximum_n,
            )
        )
    return ContactForceTargets(
        schema_version=1,
        source="verify_window_median_clamped",
        minimum_n=campaign.force_target_minimum_n,
        maximum_n=campaign.force_target_maximum_n,
        per_finger_n=targets,
    ).as_config()


def materialize_v14_source_pair_config(
    template: Mapping[str, Any],
    definition: ExperimentDefinition,
    source: AuthenticatedV13GraspSource,
) -> dict[str, Any]:
    """Promote measured geometry/qpos while replacing every v13-only block."""

    resolved = copy.deepcopy(dict(template))
    original = source.config
    for key in (
        "cube",
        "hand_pose",
        "pose_constraints",
        "contact_topology",
        "contact_point_plan",
        "closure_alignment",
        "fingertip_contact_preferences",
        "grasp_pose",
    ):
        if key in original:
            resolved[key] = copy.deepcopy(original[key])
    # A measured qpos is the grasp-pose identity, never the source servo command.
    with np.load(source.trace_path, allow_pickle=False) as trace:
        actual = np.asarray(trace["grasp_pose_actual_qpos_rad"], dtype=np.float64)
    resolved["grasp_pose"]["nominal_joint_qpos_rad"] = {
        name: float(actual[index]) for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    resolved["control"] = copy.deepcopy(original["control"])
    terminal = copy.deepcopy(resolved["control"].get("manipulation_delta_rad", {}))
    if set(terminal) != set(ACTIVE_ACTUATORS):
        terminal = {name: 0.0 for name in ACTIVE_ACTUATORS}
    resolved["manipulation_plan"] = _plan_config_from_terminal(terminal)
    resolved["contact_force_targets_n"] = _force_targets_from_source(
        source, definition
    )
    resolved["contact_feedback"] = copy.deepcopy(template["contact_feedback"])
    resolved["control_protocol"] = copy.deepcopy(template["control_protocol"])
    for key in ("settle_s", "close_s", "verify_timeout_s", "stable_window_s"):
        if key in original.get("control_protocol", {}):
            resolved["control_protocol"][key] = float(original["control_protocol"][key])
    resolved["control_protocol"]["strategy"] = (
        "grasp_verify_then_contact_preserving_planned_lift"
    )
    resolved["control_protocol"]["manipulate_s"] = 3.0
    resolved["schema_version"] = 14
    resolved["experiment_id"] = definition.experiment_id
    resolved.pop("scaled_contact_mapping", None)
    resolved["candidate_metadata"] = {
        "schema_version": 1,
        "cube_pose_sampled": False,
        "hand_root_fixed_during_simulation": True,
        "source_experiment_id": original.get("experiment_id"),
        "source_candidate_id": str(source.candidate_id),
        "source_config_sha256": source.config_sha256,
        "source_trace_sha256": source.trace_sha256,
        "source_result_semantic_sha256": source.result_semantic_sha256,
        "source_mapping_mode": source.mapping_mode,
        "source_published_in_catalog": source.published_in_catalog,
    }
    resolved["object_config_id"] = v14_object_config_id(resolved)
    resolved["grasp_pose_id"] = v14_grasp_pose_id(resolved)
    resolved["grasp_object_pair_id"] = v14_grasp_object_pair_id(resolved)
    # Planner/controller IDs are installed after fitting real checkpoint probes.
    resolved.pop("planner_id", None)
    resolved.pop("controller_id", None)
    validate_config(resolved)
    return resolved


def materialize_v14_static_rescue_config(
    template: Mapping[str, Any],
    definition: ExperimentDefinition,
    anchor: AuthenticatedV13StaticAnchor,
) -> dict[str, Any]:
    """Promote an authenticated v13 static pose into a zero-lift v14 grasp."""

    resolved = copy.deepcopy(dict(template))
    original = anchor.config
    for key in (
        "cube",
        "hand_pose",
        "pose_constraints",
        "contact_topology",
        "contact_point_plan",
        "closure_alignment",
        "fingertip_contact_preferences",
        "grasp_pose",
    ):
        if key in original:
            resolved[key] = copy.deepcopy(original[key])
    resolved["control"] = copy.deepcopy(original["control"])
    zero = {name: 0.0 for name in ACTIVE_ACTUATORS}
    resolved["control"]["manipulation_delta_rad"] = copy.deepcopy(zero)
    resolved["manipulation_plan"] = _plan_config_from_terminal(zero)
    resolved["contact_force_targets_n"] = copy.deepcopy(
        template["contact_force_targets_n"]
    )
    resolved["contact_feedback"] = copy.deepcopy(template["contact_feedback"])
    resolved["control_protocol"] = copy.deepcopy(template["control_protocol"])
    for key in ("settle_s", "close_s", "verify_timeout_s", "stable_window_s"):
        if key in original.get("control_protocol", {}):
            resolved["control_protocol"][key] = float(
                original["control_protocol"][key]
            )
    resolved["control_protocol"]["strategy"] = (
        "grasp_verify_then_contact_preserving_planned_lift"
    )
    resolved["control_protocol"]["manipulate_s"] = 3.0
    resolved["schema_version"] = 14
    resolved["experiment_id"] = definition.experiment_id
    resolved.pop("scaled_contact_mapping", None)
    resolved["candidate_metadata"] = {
        "schema_version": 1,
        "cube_pose_sampled": False,
        "hand_root_fixed_during_simulation": True,
        "stage": "v14_edge_mapping_grasp_rescue",
        "source_experiment_id": original.get("experiment_id"),
        "source_static_candidate_id": str(anchor.candidate_id),
        "source_static_candidate_sha256": anchor.candidate_sha256,
        "source_mapping_mode": anchor.mapping_mode,
        "source_static_pass": anchor.static_pass,
    }
    resolved["object_config_id"] = v14_object_config_id(resolved)
    resolved["grasp_pose_id"] = v14_grasp_pose_id(resolved)
    resolved["grasp_object_pair_id"] = v14_grasp_object_pair_id(resolved)
    resolved.pop("planner_id", None)
    resolved.pop("controller_id", None)
    validate_config(resolved)
    return resolved


def _local_rescue_config(
    base: Mapping[str, Any],
    definition: ExperimentDefinition,
    job: GraspRescueJob,
    *,
    seed: int,
) -> dict[str, Any]:
    """Materialize one deterministic 6D-pose/qpos/controller rescue sample."""

    resolved = copy.deepcopy(dict(base))
    seed_words = np.random.SeedSequence(
        [
            int(seed),
            int(job.source_candidate_id) & 0xFFFFFFFF,
            (int(job.source_candidate_id) >> 32) & 0xFFFFFFFF,
            int(job.local_index),
            14_014,
        ]
    )
    unit = np.random.default_rng(seed_words).uniform(-1.0, 1.0, 39)
    bounds = definition.search_bounds.actuator_targets_rad
    nominal = resolved["grasp_pose"]["nominal_joint_qpos_rad"]
    precontact = resolved["control"]["precontact_targets_rad"]
    preload = resolved["control"]["contact_preload_targets_rad"]
    for index, name in enumerate(ACTIVE_ACTUATORS):
        low, high = (float(value) for value in bounds[name])
        nominal_low, nominal_high = low, high
        if name == "left_hand_thumb_bend_joint_actuator":
            nominal_low = max(nominal_low, 1.40)
            nominal_high = min(nominal_high, 1.60)
        nominal[name] = float(
            np.clip(float(nominal[name]) + 0.015 * unit[index], nominal_low, nominal_high)
        )
        precontact[name] = float(
            np.clip(float(precontact[name]) + 0.02 * unit[16 + index], low, high)
        )
        preload[name] = float(
            np.clip(float(preload[name]) + 0.02 * unit[24 + index], low, high)
        )
    translation = np.asarray(resolved["hand_pose"]["translation_m"], dtype=np.float64)
    rpy = np.asarray(resolved["hand_pose"]["rpy_deg"], dtype=np.float64)
    resolved["hand_pose"]["translation_m"] = (
        translation + 0.0003 * unit[8:11]
    ).tolist()
    proposed_rpy = rpy + 0.3 * unit[11:14]
    proposed_rpy[0] = np.clip(proposed_rpy[0], *definition.search_bounds.hand_roll_deg)
    proposed_rpy[2] = np.clip(proposed_rpy[2], *definition.search_bounds.hand_yaw_deg)
    resolved["hand_pose"]["rpy_deg"] = proposed_rpy.tolist()
    profile = resolved["control"]["close_profile"]
    finger_names = {
        "thumb": ACTIVE_ACTUATORS[:3],
        "index": ACTIVE_ACTUATORS[3:6],
        "mid": ACTIVE_ACTUATORS[6:],
    }
    for finger_index, names in enumerate(finger_names.values()):
        base_start = float(profile[names[0]]["start_fraction"])
        base_end = float(profile[names[0]]["end_fraction"])
        start = float(np.clip(base_start + 0.08 * unit[32 + finger_index], 0.0, 0.95))
        end = float(
            np.clip(
                base_end + 0.05 * unit[35 + finger_index],
                max(start + 0.01, 0.01),
                1.0,
            )
        )
        for name in names:
            profile[name] = {"start_fraction": start, "end_fraction": end}
    close_options = tuple(
        float(value)
        for value in resolved["control_protocol"].get(
            "close_duration_options_s", (1.0, 1.25, 1.5, 1.75, 2.0)
        )
    )
    resolved["control_protocol"]["close_s"] = close_options[
        job.local_index % len(close_options)
    ]
    zero = {name: 0.0 for name in ACTIVE_ACTUATORS}
    resolved["control"]["manipulation_delta_rad"] = copy.deepcopy(zero)
    resolved["manipulation_plan"] = _plan_config_from_terminal(zero)
    metadata = resolved.setdefault("candidate_metadata", {})
    metadata.update(
        {
            "stage": "v14_grasp_rescue_full_reset",
            "candidate_id": str(job.candidate_id),
            "grasp_rescue_family": job.family,
            "grasp_rescue_group_id": job.group_id,
            "grasp_rescue_local_index": job.local_index,
            "grasp_rescue_source_candidate_id": str(job.source_candidate_id),
            "grasp_rescue_unit_sample": unit.tolist(),
            "cube_pose_sampled": False,
            "hand_root_fixed_during_simulation": True,
        }
    )
    resolved["object_config_id"] = v14_object_config_id(resolved)
    resolved["grasp_pose_id"] = v14_grasp_pose_id(resolved)
    resolved["grasp_object_pair_id"] = v14_grasp_object_pair_id(resolved)
    resolved.pop("planner_id", None)
    resolved.pop("controller_id", None)
    try:
        validate_config(resolved)
    except ValueError as error:
        # A source pose can lie exactly on a derived palm/tilt boundary.  Keep
        # the translational/joint/controller rescue sample, but do not push a
        # rotational proposal outside the registered experiment envelope.
        resolved["hand_pose"]["rpy_deg"] = rpy.tolist()
        resolved["object_config_id"] = v14_object_config_id(resolved)
        resolved["grasp_pose_id"] = v14_grasp_pose_id(resolved)
        resolved["grasp_object_pair_id"] = v14_grasp_object_pair_id(resolved)
        try:
            validate_config(resolved)
        except ValueError:
            raise error
    return resolved


def _source_rank(source: AuthenticatedV13GraspSource) -> tuple[Any, ...]:
    priority = (
        0
        if source.candidate_id == PRIMARY_SOURCE_CANDIDATE_ID
        else 1
        if source.candidate_id == AUXILIARY_SOURCE_CANDIDATE_ID
        else 2
    )
    metrics = source.summary.get("metrics", {})
    margin = -math.inf
    if isinstance(metrics, Mapping):
        grasp = metrics.get("actual_grasp_pose", {})
        if isinstance(grasp, Mapping):
            margin = float(grasp.get("minimum_normalized_margin", -math.inf))
    return (priority, -margin, -source.edge_m, source.candidate_id)


def build_v14_source_pairs(
    template: Mapping[str, Any],
    definition: ExperimentDefinition,
    bundle: AuthenticatedV13SourceBundle,
    rescue_sources: Sequence[AuthenticatedV13GraspSource] = (),
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    sources = tuple(bundle.sources) + tuple(rescue_sources)
    identifiers = [value.candidate_id for value in sources]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("v14 source/rescue candidate ID collision")
    measured_ids = {value.candidate_id for value in bundle.sources}
    for source in sorted(sources, key=_source_rank):
        config = materialize_v14_source_pair_config(template, definition, source)
        is_rescue = source.candidate_id not in measured_ids
        result.append(
            {
                "source_candidate_id": source.candidate_id,
                "edge_m": source.edge_m,
                "mapping_mode": source.mapping_mode,
                "priority_role": (
                    "primary_79mm"
                    if source.candidate_id == PRIMARY_SOURCE_CANDIDATE_ID
                    else "auxiliary_force_balanced_76mm"
                    if source.candidate_id == AUXILIARY_SOURCE_CANDIDATE_ID
                    else "per_edge_mapping_grasp_rescue"
                    if is_rescue
                    else "authenticated_measured_grasp"
                ),
                "object_config_id": config["object_config_id"],
                "grasp_pose_id": config["grasp_pose_id"],
                "grasp_object_pair_id": config["grasp_object_pair_id"],
                "warm_start_records": (
                    copy.deepcopy(list(bundle.warm_start_records))
                    if source.candidate_id == PRIMARY_SOURCE_CANDIDATE_ID and not is_rescue
                    else []
                ),
                "config": config,
            }
        )
    return tuple(result)


def _commit_json_stage(
    workspace: Path,
    stage: str,
    path: Path,
    payload: Mapping[str, Any],
    *,
    stage_input: Mapping[str, Any],
    extra_artifacts: Sequence[Path] = (),
) -> dict[str, Any]:
    write_json(path, payload)
    commit_campaign_stage(
        workspace,
        stage,
        stage_input=stage_input,
        artifacts=(path, *extra_artifacts),
        summary={
            "complete": bool(payload.get("complete", True)),
            "record_count": len(payload.get("records", ())),
        },
    )
    return copy.deepcopy(dict(payload))


def _load_committed_json(workspace: Path, stage: str, path: Path) -> dict[str, Any] | None:
    ledger = validate_stage_ledger(workspace)
    if stage not in ledger["stages"]:
        return None
    if not path.is_file():
        raise RuntimeError(f"committed stage {stage} lost {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("complete") is not True:
        raise RuntimeError(f"committed stage {stage} report is incomplete")
    return copy.deepcopy(dict(payload))


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(destination) != file_sha256(source):
            raise RuntimeError(f"existing catalog artifact changed: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _candidate_id(payload: Mapping[str, Any]) -> int:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return 14 * 10**15 + int.from_bytes(digest[:6], "big") % 10**14


def _quat_rotvec(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    relative = np.empty(4, dtype=np.float64)
    conjugate = np.asarray(start, dtype=np.float64).copy()
    conjugate[1:] *= -1.0
    mujoco.mju_mulQuat(relative, np.asarray(end, dtype=np.float64), conjugate)
    if relative[0] < 0.0:
        relative *= -1.0
    norm = float(np.linalg.norm(relative[1:]))
    if norm <= 1e-14:
        return np.zeros(3)
    return relative[1:] * (2.0 * math.atan2(norm, float(relative[0])) / norm)


def _run_extended_checkpoint_probe(
    grasp: GraspPhysicsCheckpoint,
    specification: ProbeSpecification,
    *,
    duration_s: float = 0.20,
) -> Mapping[str, Any]:
    """Short causal probe with conservative force/topology/slip evidence."""

    model = grasp.model
    data = mujoco.MjData(model)
    from ..checkpoint import restore_physics_checkpoint

    restore_physics_checkpoint(model, data, grasp.checkpoint)
    _, info = build_model(grasp.config)
    start_position = data.xpos[grasp.cube_body_id].copy()
    start_quat = data.xquat[grasp.cube_body_id].copy()
    preload = actuator_target_vector(model, grasp.config["control"]["contact_preload_targets_rad"])
    delta = np.zeros(model.nu)
    for index, name in enumerate(ACTIVE_ACTUATORS):
        delta[model.actuator(name).id] = specification.applied_delta_rad[index]
    target_labels = grasp.config["contact_topology"]["target_faces"]
    target_faces = tuple(face_from_label(target_labels[name]) for name in ACTIVE_FINGERS)
    initial_contact = contact_snapshot(model, data, info, classify_faces=True)
    assert initial_contact.distal_face_force_n is not None
    assert initial_contact.distal_face_position_moment_n_m is not None
    initial_centroid, initial_valid = target_face_contact_centroids(
        initial_contact.distal_face_force_n,
        initial_contact.distal_face_position_moment_n_m,
        target_faces,
    )
    steps = max(1, int(round(duration_s / model.opt.timestep)))
    force_samples: list[np.ndarray] = []
    valid_samples: list[np.ndarray] = []
    slip_samples: list[np.ndarray] = []
    forbidden = False
    nondistal = False
    for step in range(steps):
        data.ctrl[:] = preload + minimum_jerk((step + 1) / steps) * delta
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        snapshot = contact_snapshot(model, data, info, classify_faces=True)
        assert snapshot.distal_face_force_n is not None
        assert snapshot.distal_face_position_moment_n_m is not None
        assert snapshot.active_nondistal_force_n is not None
        target = np.asarray(
            [snapshot.distal_face_force_n[i, int(target_faces[i])] for i in range(3)]
        )
        total = np.sum(snapshot.distal_face_force_n, axis=1)
        purity = np.divide(target, total, out=np.zeros(3), where=total > 0.0)
        centroids, centroid_valid = target_face_contact_centroids(
            snapshot.distal_face_force_n,
            snapshot.distal_face_position_moment_n_m,
            target_faces,
        )
        cube_rotation = data.xmat[grasp.cube_body_id].reshape(3, 3)
        # Missing a target contact is represented by the discrete validity
        # mask and a large *finite* tangent penalty.  The response fitter
        # intentionally rejects non-finite samples before considering that
        # mask.
        slip = np.full(3, 0.05)
        for i, finger in enumerate(ACTIVE_FINGERS):
            if initial_valid[i] and centroid_valid[i]:
                displacement = centroids[i] - initial_centroid[i]
                normal = cube_rotation @ _FACE_NORMAL_LOCAL[target_labels[finger]]
                tangent = displacement - float(np.dot(displacement, normal)) * normal
                slip[i] = float(np.linalg.norm(tangent))
        if step >= steps // 2:
            force_samples.append(target)
            valid_samples.append((target >= 0.05) & (purity >= 0.95))
            slip_samples.append(slip)
        forbidden = forbidden or snapshot.forbidden
        nondistal = nondistal or bool(np.any(snapshot.active_nondistal_force_n >= 0.05))
    forces = np.min(np.asarray(force_samples), axis=0)
    valid = np.all(np.asarray(valid_samples), axis=0)
    slip = np.max(np.asarray(slip_samples), axis=0)
    response = np.concatenate(
        (
            data.xpos[grasp.cube_body_id] - start_position,
            _quat_rotvec(start_quat, data.xquat[grasp.cube_body_id]),
        )
    )
    return {
        "probe": specification.as_mapping(),
        "checkpoint_step_index": grasp.checkpoint.step_index,
        "response_6d": response.tolist(),
        "contact_evidence": {
            "target_normal_force_n": {
                name: float(forces[index]) for index, name in enumerate(ACTIVE_FINGERS)
            },
            "target_contact_valid": {
                name: bool(valid[index]) for index, name in enumerate(ACTIVE_FINGERS)
            },
            "tangent_slip_m": {
                name: float(slip[index]) for index, name in enumerate(ACTIVE_FINGERS)
            },
            "forbidden_contact": forbidden,
            "active_nondistal_contact": nondistal,
        },
    }


def _planner_setting_variants(force_targets: Mapping[str, Any]) -> tuple[ContactConstrainedPlannerSettings, ...]:
    target = tuple(float(force_targets["per_finger_n"][name]) for name in ACTIVE_FINGERS)
    variants: list[ContactConstrainedPlannerSettings] = []
    # 24 deterministic settings x four trust backoffs = 96 plan candidates.
    for lift in (0.010, 0.0105, 0.011, 0.0115, 0.012, 0.011):
        for multiplier in (0.90, 1.00, 1.10, 1.20):
            variants.append(
                ContactConstrainedPlannerSettings(
                    target_object_response_6d=(0.0, 0.0, lift, 0.0, 0.0, 0.0),
                    target_normal_force_n=tuple(
                        max(0.05, min(3.0, value * multiplier)) for value in target
                    ),
                )
            )
    return tuple(variants)


def _warm_start_plan_records(
    pair: Mapping[str, Any],
    config: Mapping[str, Any],
    bounds: Mapping[str, Sequence[float]],
) -> tuple[dict[str, Any], ...]:
    """Convert authenticated nonlinear v13 lifts into v14 knot plans.

    The source candidate remains diagnostic evidence; every returned config is
    still rerun from reset by the v14 controller.  Four bounded scales around
    each of the two 79-mm terminal commands provide eight deterministic warm
    starts without spending a feedback batch on a near-zero linear solution.
    """

    records: list[dict[str, Any]] = []
    # A deterministic full-reset refinement of the authenticated 79-mm c008
    # source.  These values are only a search seed: they are bound to the
    # immutable c008 source below and must pass the same fresh reset run as
    # every solver-generated plan before they can be ranked or published.
    # Keeping the preset here (rather than importing a /tmp diagnostic) makes
    # the formal manifest and resume ledger cover every command value.
    if (
        int(pair.get("source_candidate_id", -1)) == PRIMARY_SOURCE_CANDIDATE_ID
        and pair.get("warm_start_records")
    ):
        exact_terminal = {
            "left_hand_thumb_bend_joint_actuator": -0.011031995765187281,
            "left_hand_thumb_rota_joint1_actuator": -0.09,
            "left_hand_thumb_rota_joint2_actuator": 0.46,
            "left_hand_index_bend_joint_actuator": -0.05,
            "left_hand_index_joint1_actuator": -0.1095,
            "left_hand_index_joint2_actuator": 0.535,
            "left_hand_mid_joint1_actuator": -0.09743618044100734,
            "left_hand_mid_joint2_actuator": 0.27082973148714146,
        }
        for name, value in exact_terminal.items():
            lower, upper = (float(item) for item in bounds[name])
            if not lower <= value <= upper:
                raise RuntimeError(f"registered refined warm start exceeds {name} bounds")
        source = pair["warm_start_records"][0]
        planned = copy.deepcopy(dict(config))
        planned["manipulation_plan"] = _plan_config_from_terminal(exact_terminal)
        planned["control"]["manipulation_delta_rad"] = copy.deepcopy(exact_terminal)
        planner_id = canonical_sha256(
            {
                "schema_version": 1,
                "kind": "authenticated_79mm_c008_contact_compensated_v1",
                "source_config_sha256": source["config_sha256"],
                "source_result_semantic_sha256": source[
                    "result_semantic_sha256"
                ],
                "terminal_delta_rad": exact_terminal,
                "fresh_full_reset_required": True,
            }
        )
        planned["planner_id"] = planner_id
        planned["controller_id"] = canonical_sha256(
            {
                "schema_version": 1,
                "grasp_object_pair_id": pair["grasp_object_pair_id"],
                "plan_id": planned["manipulation_plan"]["plan_id"],
                "target_id": planned["contact_force_targets_n"]["target_id"],
                "feedback_id": planned["contact_feedback"]["feedback_id"],
                "planner_id": planner_id,
            }
        )
        planned.setdefault("candidate_metadata", {})["formal_warm_start"] = {
            "schema_version": 1,
            "preset": "79mm_c008_contact_compensated_v1",
            "source_config_sha256": source["config_sha256"],
            "source_result_semantic_sha256": source["result_semantic_sha256"],
            "checkpoint_is_success_evidence": False,
            "requires_full_reset_rerun": True,
        }
        validate_config(planned)
        identity = {
            "pair_id": pair["grasp_object_pair_id"],
            "warm_start_alias": "v14_79mm_c3_contact_compensated_v1",
            "plan_id": planned["manipulation_plan"]["plan_id"],
            "planner_id": planner_id,
        }
        records.append(
            {
                "candidate_id": _candidate_id(identity),
                "source_candidate_id": int(pair["source_candidate_id"]),
                "grasp_object_pair_id": pair["grasp_object_pair_id"],
                "warm_start": True,
                "warm_start_priority": -1,
                "warm_start_alias": "v14_79mm_c3_contact_compensated_v1",
                "warm_start_source_config_sha256": source["config_sha256"],
                "warm_start_source_result_sha256": source["result_sha256"],
                "warm_start_source_result_semantic_sha256": source[
                    "result_semantic_sha256"
                ],
                "source_operation_contact_duty": float(
                    source["operation_simultaneous_contact_duty"]
                ),
                "contact_feasible": True,
                "nonlinear_scale_requires_full_reset": True,
                "path_rms_error": 0.0,
                "terminal_path_error": 0.0,
                "terminal_path_rejected": False,
                "plan": copy.deepcopy(planned["manipulation_plan"]),
                "planner_report_id": planner_id,
                "config": planned,
            }
        )
    for source in pair.get("warm_start_records", ()):
        if not isinstance(source, Mapping):
            raise RuntimeError("v14 warm-start descriptor is malformed")
        terminal = source.get("terminal_delta_rad")
        if not isinstance(terminal, Mapping) or set(terminal) != set(ACTIVE_ACTUATORS):
            raise RuntimeError("v14 warm-start terminal command is malformed")
        for scale_index, scale in enumerate((0.85, 1.00, 1.15, 1.30)):
            scaled = {
                name: float(
                    np.clip(
                        float(terminal[name]) * scale,
                        float(bounds[name][0]),
                        float(bounds[name][1]),
                    )
                )
                for name in ACTIVE_ACTUATORS
            }
            planned = copy.deepcopy(dict(config))
            planned["manipulation_plan"] = _plan_config_from_terminal(scaled)
            planned["control"]["manipulation_delta_rad"] = copy.deepcopy(scaled)
            planner_id = canonical_sha256(
                {
                    "schema_version": 1,
                    "kind": "authenticated_v13_nonlinear_warm_start",
                    "source_config_sha256": source["config_sha256"],
                    "source_result_semantic_sha256": source[
                        "result_semantic_sha256"
                    ],
                    "scale": scale,
                }
            )
            controller_id = canonical_sha256(
                {
                    "schema_version": 1,
                    "grasp_object_pair_id": pair["grasp_object_pair_id"],
                    "plan_id": planned["manipulation_plan"]["plan_id"],
                    "target_id": planned["contact_force_targets_n"]["target_id"],
                    "feedback_id": planned["contact_feedback"]["feedback_id"],
                    "planner_id": planner_id,
                }
            )
            planned["planner_id"] = planner_id
            planned["controller_id"] = controller_id
            # The first diagnostic sweep predated self-contained planner
            # lineage and had to be authenticated by a finite legacy registry.
            # Every newly generated nonlinear warm start persists the exact
            # canonical planner inputs so config validation and NPZ evaluation
            # can independently recompute planner_id.
            planned.setdefault("candidate_metadata", {})[
                "v14_authenticated_nonlinear_warm_start"
            ] = {
                "schema_version": 1,
                "source_config_sha256": source["config_sha256"],
                "source_result_semantic_sha256": source[
                    "result_semantic_sha256"
                ],
                "scale": scale,
            }
            validate_config(planned)
            predicted_lift = float(source["operation_median_lift_m"]) * scale
            terminal_error = abs(0.011 - predicted_lift)
            source_duty = float(source["operation_simultaneous_contact_duty"])
            identity = {
                "pair_id": pair["grasp_object_pair_id"],
                "warm_start_alias": source["warm_start_alias"],
                "scale_index": scale_index,
                "plan_id": planned["manipulation_plan"]["plan_id"],
            }
            records.append(
                {
                    "candidate_id": _candidate_id(identity),
                    "source_candidate_id": int(pair["source_candidate_id"]),
                    "grasp_object_pair_id": pair["grasp_object_pair_id"],
                    "warm_start": True,
                    "warm_start_priority": 0,
                    "warm_start_alias": source["warm_start_alias"],
                    "warm_start_scale": scale,
                    "warm_start_source_config_sha256": source["config_sha256"],
                    "warm_start_source_result_sha256": source["result_sha256"],
                    "warm_start_source_result_semantic_sha256": source[
                        "result_semantic_sha256"
                    ],
                    "source_operation_contact_duty": source_duty,
                    # This is a search prior, not success evidence.  The source
                    # trajectory already satisfies the contact contract; an
                    # extrapolated scale is therefore eligible for full-reset
                    # testing and must not be demoted solely by an arbitrary
                    # legacy endpoint box.  Runtime contact checks remain hard.
                    "contact_feasible": bool(source_duty >= 0.99),
                    "nonlinear_scale_requires_full_reset": scale != 1.0,
                    "path_rms_error": terminal_error,
                    "terminal_path_error": terminal_error,
                    "terminal_path_rejected": terminal_error > 0.008,
                    "plan": copy.deepcopy(planned["manipulation_plan"]),
                    "planner_report_id": planner_id,
                    "config": planned,
                }
            )
    return tuple(records)


def _default_plan_runner(
    pair: Mapping[str, Any],
    source: AuthenticatedV13GraspSource,
    output_dir: Path,
) -> Sequence[Mapping[str, Any]]:
    config = copy.deepcopy(dict(pair["config"]))
    grasp = prepare_grasp_checkpoint(source.config, source.trace_path, {"summary": source.summary})
    # The checkpoint model is identical, but v13 source configs intentionally
    # carried a much narrower legacy manipulation search box.  Planning must
    # intersect the real model limits with the registered v14 bounds/preload.
    bounds = manipulation_delta_bounds(grasp.model, config)
    warm_starts = _warm_start_plan_records(pair, config, bounds)
    exact = next(
        (
            value
            for value in warm_starts
            if value.get("warm_start_alias")
            == "v14_79mm_c3_contact_compensated_v1"
        ),
        None,
    )
    initial_plan = (
        exact["config"]["manipulation_plan"]
        if exact is not None
        else config["manipulation_plan"]
    )
    target = tuple(
        float(config["contact_force_targets_n"]["per_finger_n"][name])
        for name in ACTIVE_FINGERS
    )
    settings = ContactConstrainedPlannerSettings(
        target_normal_force_n=target,
        minimum_normal_force_n=(0.05, 0.05, 0.05),
        maximum_tangent_slip_m=(0.002, 0.002, 0.002),
        target_object_response_6d=(0.0, 0.0, 0.011, 0.0, 0.0, 0.0),
        duration_s=3.0,
        max_knot_delta_rad=0.04,
    )
    report = plan_sequential_contact_trajectory(
        grasp,
        initial_plan,
        bounds,
        settings=settings,
    )
    report_payload = report.as_mapping()
    records: list[dict[str, Any]] = []
    for attempt in report.attempts:
        planned = materialize_sequential_plan_config(
            config,
            report,
            attempt_index=attempt.attempt_index,
            validate=False,
        )
        planner_id = canonical_sha256(
            {
                "schema_version": 1,
                "kind": "sequential_checkpoint_relinearized_contact_plan",
                "report_id": report_payload["report_id"],
                "attempt_report_id": attempt.as_mapping()["attempt_report_id"],
                "fresh_full_reset_required": True,
            }
        )
        planned["planner_id"] = planner_id
        planned["controller_id"] = canonical_sha256(
            {
                "schema_version": 1,
                "grasp_object_pair_id": pair["grasp_object_pair_id"],
                "plan_id": planned["manipulation_plan"]["plan_id"],
                "target_id": planned["contact_force_targets_n"]["target_id"],
                "feedback_id": planned["contact_feedback"]["feedback_id"],
                "planner_id": planner_id,
            }
        )
        validate_config(planned)
        identity = {
            "pair_id": pair["grasp_object_pair_id"],
            "sequential_report_id": report_payload["report_id"],
            "attempt_index": attempt.attempt_index,
            "plan_id": attempt.plan.plan_id,
        }
        terminal_delta = (
            attempt.plan.predicted_object_response_6d[-1]
            - attempt.plan.desired_object_response_6d[-1]
        )
        terminal_translation_error = float(np.linalg.norm(terminal_delta[:3]))
        terminal_rotation_error = float(np.linalg.norm(terminal_delta[3:]))
        records.append(
            {
                "candidate_id": _candidate_id(identity),
                "source_candidate_id": int(pair["source_candidate_id"]),
                "grasp_object_pair_id": pair["grasp_object_pair_id"],
                "settings_index": 0,
                "attempt_index": attempt.attempt_index,
                "plan": attempt.plan.as_mapping(),
                "planner_report_id": report_payload["report_id"],
                "attempt_report_id": attempt.as_mapping()["attempt_report_id"],
                "contact_feasible": attempt.search_contact_safe,
                "sequential_checkpoint_relinearized": True,
                "warm_start": exact is not None,
                "warm_start_priority": 0 if exact is not None else 1,
                "path_rms_error": attempt.plan.path_rms_error,
                "terminal_path_error": terminal_translation_error,
                "terminal_translation_error_m": terminal_translation_error,
                "terminal_rotation_error_rad": terminal_rotation_error,
                "terminal_path_rejected": terminal_translation_error > 0.008,
                "config": planned,
            }
        )
    if len(records) != 4:
        raise RuntimeError("v14 sequential planner did not produce exactly four plans")
    write_json(
        output_dir / "sequential_planning_report.json",
        report_payload,
    )
    return records


def _feedback_variants(config: Mapping[str, Any], count: int = 64) -> tuple[dict[str, Any], ...]:
    from ..experiment import ContactFeedbackParameters

    base = config["contact_feedback"]
    variants: list[dict[str, Any]] = []
    for index in range(count):
        # The nonlinear warm starts can already preserve contact open-loop.
        # Cover weak gains as well as the nominal/aggressive regime so the
        # bidirectional PI does not over-release a lightly loaded finger while
        # trying to suppress high-force chatter on another one.
        grid_index = max(0, index - 1)
        kp_scale = (0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.20)[
            grid_index % 8
        ]
        ki_scale = (0.00, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)[
            min(7, grid_index // 8)
        ]
        per_finger_kp_scale = {
            "thumb": 0.01 if index == 0 else kp_scale,
            "index": 0.01 if index == 0 else kp_scale,
            "mid": 0.30 if index == 0 else kp_scale,
        }
        effective_ki_scale = 0.0 if index == 0 else ki_scale
        feedback = ContactFeedbackParameters(
            schema_version=int(base.get("schema_version", 1)),
            strategy=str(base["strategy"]),
            filter_time_constant_s=float(base["filter_time_constant_s"]),
            kp_rad_per_n={
                name: float(base["kp_rad_per_n"][name])
                * per_finger_kp_scale[name]
                for name in ACTIVE_FINGERS
            },
            ki_rad_per_n_s={
                name: float(base["ki_rad_per_n_s"][name])
                * effective_ki_scale
                for name in ACTIVE_FINGERS
            },
            integral_limit_n_s=float(base["integral_limit_n_s"]),
            correction_limit_rad=float(base["correction_limit_rad"]),
            rate_limit_rad_s=float(base["rate_limit_rad_s"]),
            acceleration_limit_rad_s2=float(base["acceleration_limit_rad_s2"]),
            force_risk_n=float(base["force_risk_n"]),
            freeze_on_risk=bool(base["freeze_on_risk"]),
            max_loss_s=float(base["max_loss_s"]),
            recovery_behavior=str(base["recovery_behavior"]),
            operation_contact_duty_min=float(base["operation_contact_duty_min"]),
            tangent_slip_freeze_threshold_m=(
                float(base["tangent_slip_freeze_threshold_m"])
                if int(base.get("schema_version", 1)) >= 2
                else None
            ),
            tangent_slip_abort_threshold_m=(
                float(base["tangent_slip_abort_threshold_m"])
                if int(base.get("schema_version", 1)) >= 2
                else None
            ),
        )
        variants.append(feedback.as_config())
    return tuple(variants)


def _run_full_reset_candidate(
    config: Mapping[str, Any],
    destination: Path,
    candidate_id: int,
    *,
    retain_grasp_success: bool = False,
    final_rerun: bool = False,
) -> dict[str, Any]:
    """Run one immutable v14 candidate without materializing useless traces.

    Search near misses and ordinary grasp-only candidates keep their complete
    semantic summary but no multi-megabyte NPZ.  Rescue evidence can opt into
    grasp trace retention, while every full success and explicit catalog rerun
    always keeps the exact trace produced by its evaluated session.
    """

    legacy_result = destination / "result.json"
    if legacy_result.is_file():
        legacy = json.loads(legacy_result.read_text(encoding="utf-8"))
        if "trace_retention" not in legacy:
            if int(legacy.get("candidate_id", -1)) != int(candidate_id):
                raise RuntimeError(
                    f"resumed v14 candidate {candidate_id} has the wrong candidate ID"
                )
            authenticate_candidate_result_semantic_sha256(
                legacy, source=f"legacy injected v14 candidate {candidate_id}"
            )
            config_path = destination / "resolved_config.json"
            trace_path = destination / "trace.npz"
            hashes = legacy.get("artifacts", {}).get("sha256", {})
            for name, path in (("resolved_config", config_path), ("trace", trace_path)):
                if not path.is_file() or hashes.get(name) != file_sha256(path):
                    raise RuntimeError(
                        f"resumed v14 candidate {candidate_id} {name} SHA-256 mismatch"
                    )
            persisted = json.loads(config_path.read_text(encoding="utf-8"))
            if canonical_sha256(persisted) != canonical_sha256(config):
                raise RuntimeError(
                    f"resumed v14 candidate {candidate_id} requested config changed"
                )
            if final_rerun and not trace_path.is_file():
                raise RuntimeError("legacy v14 final rerun lost trace.npz")
            return copy.deepcopy(dict(legacy))
    bundle = run_or_resume_v14_candidate_artifacts(
        config,
        destination,
        int(candidate_id),
        retain_grasp_success=bool(retain_grasp_success),
        final_rerun=bool(final_rerun),
    )
    return copy.deepcopy(bundle.result)


def _default_candidate_runner(
    plan_records: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    target_success_count: int,
    workers: int,
    feedback_candidates_per_plan: int = 64,
    maximum_candidate_count: int | None = None,
) -> Sequence[Mapping[str, Any]]:
    # A near-zero command can look perfectly contact-feasible while missing
    # the 11-mm object target almost entirely.  Such plans are retained in the
    # planning report as diagnostics, but spending 64 full-reset feedback
    # trials on them cannot establish the requested manipulation.
    eligible_plans = tuple(
        value
        for value in plan_records
        if not bool(value.get("terminal_path_rejected", False))
    )
    ranked_plans = sorted(
        eligible_plans,
        key=lambda value: (
            not bool(value.get("contact_feasible", False)),
            bool(value.get("terminal_path_rejected", False)),
            int(value.get("warm_start_priority", 1)),
            float(value.get("path_rms_error", math.inf)),
            float(value.get("terminal_path_error", math.inf)),
            int(value["candidate_id"]),
        ),
    )
    records: list[dict[str, Any]] = []
    successes = 0
    maximum = (
        len(ranked_plans) * int(feedback_candidates_per_plan)
        if maximum_candidate_count is None
        else int(maximum_candidate_count)
    )
    if maximum <= 0:
        raise ValueError("maximum_candidate_count must be positive")
    attempted = 0
    for plan_rank, plan_record in enumerate(ranked_plans):
        variants = _feedback_variants(
            plan_record["config"], count=int(feedback_candidates_per_plan)
        )
        jobs: list[dict[str, Any]] = []
        metadata: dict[int, tuple[int, str]] = {}
        for feedback_index, feedback in enumerate(variants):
            if attempted >= maximum:
                break
            config = copy.deepcopy(dict(plan_record["config"]))
            config["contact_feedback"] = feedback
            identity = {
                "plan_candidate_id": int(plan_record["candidate_id"]),
                "feedback_id": feedback["feedback_id"],
            }
            candidate_id = _candidate_id(identity)
            config["controller_id"] = canonical_sha256(
                {
                    "schema_version": 1,
                    "grasp_object_pair_id": config["grasp_object_pair_id"],
                    "plan_id": config["manipulation_plan"]["plan_id"],
                    "target_id": config["contact_force_targets_n"]["target_id"],
                    "feedback_id": feedback["feedback_id"],
                }
            )
            jobs.append(
                {
                    "config": config,
                    "destination": str(
                        workspace / "candidates" / f"candidate_{candidate_id}"
                    ),
                    "candidate_id": candidate_id,
                }
            )
            metadata[candidate_id] = (feedback_index, feedback["feedback_id"])
            attempted += 1
        if not jobs:
            break
        if int(workers) == 1:
            batch = tuple(_run_candidate_job(job) for job in jobs)
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=int(workers), mp_context=context
            ) as executor:
                batch = tuple(executor.map(_run_candidate_job, jobs))
        # executor.map preserves input order, but sort again by immutable ID so
        # a future unordered backend cannot leak completion timing into output.
        batch = tuple(sorted(batch, key=lambda value: int(value["candidate_id"])))
        for record in batch:
            candidate_id = int(record["candidate_id"])
            feedback_index, feedback_id = metadata[candidate_id]
            record = {
                **record,
                "plan_candidate_id": int(plan_record["candidate_id"]),
                "plan_rank": plan_rank,
                "feedback_index": feedback_index,
                "feedback_id": feedback_id,
                "artifact_directory": f"candidates/candidate_{candidate_id}",
            }
            records.append(record)
            successes += int(record["full_success"])
        # The terminal condition is evaluated only after a complete plan batch
        # so worker count cannot change which same-plan feedback candidates are
        # persisted or ranked.
        if successes >= target_success_count or attempted >= maximum:
            break
    return tuple(records)


def _select_feedback_plan_quota(
    plan_records: Sequence[Mapping[str, Any]],
    *,
    plans_per_pair: int,
) -> tuple[dict[str, Any], ...]:
    """Retain the declared number of multi-node plans for every pair.

    Offline planning keeps the registered four checkpoint-relinearized plans
    per pair.  The expensive feedback search is balanced per hand/object pair
    so a globally strong 79-mm seed cannot consume every other size's budget.
    """

    if int(plans_per_pair) <= 0:
        raise ValueError("plans_per_pair must be positive")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    pair_order: list[str] = []
    for record in plan_records:
        pair_id = str(record["grasp_object_pair_id"])
        if pair_id not in grouped:
            pair_order.append(pair_id)
        grouped[pair_id].append(record)
    selected: list[dict[str, Any]] = []
    for pair_id in pair_order:
        values = sorted(
            grouped[pair_id],
            key=lambda value: (
                bool(value.get("terminal_path_rejected", False)),
                int(value.get("warm_start_priority", 1)),
                not bool(value.get("contact_feasible", False)),
                float(value.get("terminal_path_error", math.inf)),
                float(value.get("path_rms_error", math.inf)),
                int(value["candidate_id"]),
            ),
        )
        eligible = [
            value
            for value in values
            if not bool(value.get("terminal_path_rejected", False))
        ]
        selected.extend(
            copy.deepcopy(dict(value))
            for value in eligible[: int(plans_per_pair)]
        )
    return tuple(selected)


def _run_candidate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Spawn-safe full-reset candidate boundary."""

    return _run_full_reset_candidate(
        job["config"],
        Path(str(job["destination"])),
        int(job["candidate_id"]),
        retain_grasp_success=bool(job.get("retain_grasp_success", False)),
        final_rerun=bool(job.get("final_rerun", False)),
    )


_CANDIDATE_RESULT_WRAPPER_METADATA_FIELDS = frozenset(
    {
        "plan_candidate_id",
        "plan_rank",
        "feedback_index",
        "feedback_id",
        "artifact_directory",
    }
)


def _authenticate_candidate_result_wrapper(
    persisted: Mapping[str, Any], record: Mapping[str, Any]
) -> None:
    """Authenticate a persisted result embedded in its search-report wrapper.

    The atomic candidate directory owns the immutable simulation result.  The
    production feedback runner adds only scheduling/provenance metadata before
    placing that result in the campaign report.  Compare an exact projection
    over every persisted field and fail closed if the report drops a result
    field or adds anything outside the declared wrapper metadata contract.
    """

    persisted_fields = frozenset(persisted)
    record_fields = frozenset(record)
    missing = sorted(persisted_fields - record_fields)
    if missing:
        raise RuntimeError(
            "v14 candidate wrapper lost persisted result fields: "
            + ", ".join(missing)
        )
    unknown = sorted(
        record_fields
        - persisted_fields
        - _CANDIDATE_RESULT_WRAPPER_METADATA_FIELDS
    )
    if unknown:
        raise RuntimeError(
            "v14 candidate wrapper has unknown metadata fields: "
            + ", ".join(unknown)
        )
    projected = {key: record[key] for key in persisted_fields}
    if canonical_sha256(projected) != canonical_sha256(persisted):
        changed = sorted(
            key for key in persisted_fields if projected[key] != persisted[key]
        )
        detail = ", ".join(changed) if changed else "unknown"
        raise RuntimeError(
            f"v14 candidate wrapper changed persisted result fields: {detail}"
        )


def _candidate_artifact_paths(
    destination: Path, record: Mapping[str, Any]
) -> tuple[Path, ...]:
    """Return exactly the authenticated files retained for one candidate."""

    # Injected unit-test backends predate the production trace-retention
    # contract.  They remain accepted only when every declared artifact and
    # digest is present; production records always carry trace_retention.
    if "trace_retention" not in record:
        artifacts = record.get("artifacts", {})
        hashes = artifacts.get("sha256", {}) if isinstance(artifacts, Mapping) else {}
        paths: list[Path] = []
        for field in ("resolved_config", "result", "trace"):
            name = (
                "result.json"
                if field == "result"
                else artifacts.get(field)
                if isinstance(artifacts, Mapping)
                else None
            )
            if not isinstance(name, str):
                if field == "trace":
                    continue
                raise RuntimeError(f"injected v14 candidate lost {field}")
            path = destination / name
            if not path.is_file():
                raise RuntimeError(f"injected v14 candidate lost {path.name}")
            if field != "result" and hashes.get(field) != file_sha256(path):
                raise RuntimeError(f"injected v14 candidate {field} SHA-256 mismatch")
            paths.append(path)
        return tuple(paths)
    bundle = authenticate_v14_candidate_artifacts(
        destination,
        expected_candidate_id=int(record["candidate_id"]),
    )
    _authenticate_candidate_result_wrapper(bundle.result, record)
    return bundle.artifact_paths


def _default_joint_refinement_runner(
    candidate_records: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    workers: int,
    seed: int,
    parent_count: int,
    candidates_per_parent: int,
) -> Mapping[str, Any]:
    """Execute the registered global top-parent joint/controller refinement."""

    if len(candidate_records) < int(parent_count):
        # This path is useful only for injected unit-test backends.  Production
        # evaluates at least one complete 64-feedback plan batch.
        return {
            "report": {
                "v14_joint_refinement_report_schema_version": 1,
                "complete": True,
                "skipped": True,
                "skip_reason": "fewer_than_registered_parent_count",
                "parent_count": 0,
                "candidates_per_parent": int(candidates_per_parent),
                "candidate_count": 0,
                "full_success_count": 0,
                "records": [],
            },
            "artifacts": (),
        }
    materialized: list[dict[str, Any]] = []
    for record in candidate_records:
        root = workspace / str(record["artifact_directory"])
        config_path = root / "resolved_config.json"
        if not config_path.is_file():
            raise RuntimeError("v14 refinement parent config is missing")
        materialized.append(
            {
                **copy.deepcopy(dict(record)),
                "config": json.loads(config_path.read_text(encoding="utf-8")),
            }
        )
    budget = JointRefinementBudget(
        parent_count=int(parent_count),
        candidates_per_parent=int(candidates_per_parent),
        seed=int(seed),
    )
    specs = build_joint_refinement_job_specs(materialized, budget=budget)
    records: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    batch_size = max(int(workers), int(workers) * 8)
    for start in range(0, len(specs), batch_size):
        batch_specs = specs[start : start + batch_size]
        jobs = []
        metadata: dict[int, Mapping[str, Any]] = {}
        for spec in batch_specs:
            candidate_id = int(spec["candidate_id"])
            destination = (
                workspace
                / "joint_refinement"
                / "candidates"
                / f"candidate_{candidate_id}"
            )
            jobs.append(
                {
                    "config": spec["config"],
                    "destination": str(destination),
                    "candidate_id": candidate_id,
                }
            )
            metadata[candidate_id] = spec
        if int(workers) == 1:
            completed = tuple(_run_candidate_job(value) for value in jobs)
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=int(workers), mp_context=context
            ) as executor:
                completed = tuple(executor.map(_run_candidate_job, jobs))
        for raw in sorted(completed, key=lambda value: int(value["candidate_id"])):
            candidate_id = int(raw["candidate_id"])
            spec = metadata[candidate_id]
            destination = (
                workspace
                / "joint_refinement"
                / "candidates"
                / f"candidate_{candidate_id}"
            )
            record = {
                **copy.deepcopy(dict(raw)),
                "joint_refinement": True,
                "parent_candidate_id": int(spec["parent_candidate_id"]),
                "parent_rank": int(spec["parent_rank"]),
                "local_index": int(spec["local_index"]),
                "job_sequence_index": int(spec["job_sequence_index"]),
                "artifact_directory": str(destination.relative_to(workspace)),
            }
            records.append(record)
            artifacts.extend(_candidate_artifact_paths(destination, raw))
    ranked = rank_contact_constrained_candidates(records)
    return {
        "report": {
            "v14_joint_refinement_report_schema_version": 1,
            "complete": True,
            "skipped": False,
            "budget": budget.as_mapping(),
            "parent_count": int(parent_count),
            "candidates_per_parent": int(candidates_per_parent),
            "candidate_count": len(ranked),
            "full_success_count": sum(
                bool(value.get("full_success", False)) for value in ranked
            ),
            "records": list(ranked),
        },
        "artifacts": tuple(artifacts),
    }


def _authenticate_rescue_record(
    workspace: Path, record: Mapping[str, Any]
) -> tuple[Path, ...]:
    root = (workspace / str(record["artifact_directory"])).resolve()
    if not root.is_relative_to(workspace / "grasp_rescue"):
        raise RuntimeError("v14 grasp rescue artifact escaped its workspace")
    if "trace_retention" not in record:
        config_path = root / "resolved_config.json"
        result_path = root / "result.json"
        trace_path = root / "trace.npz"
        if not all(
            value.is_file() for value in (config_path, result_path, trace_path)
        ):
            raise RuntimeError("injected v14 grasp rescue artifact is missing")
        persisted = json.loads(result_path.read_text(encoding="utf-8"))
        authenticate_candidate_result_semantic_sha256(
            persisted, source=f"injected v14 grasp rescue {record.get('candidate_id')}"
        )
        hashes = persisted.get("artifacts", {}).get("sha256", {})
        if hashes.get("resolved_config") != file_sha256(config_path):
            raise RuntimeError("v14 grasp rescue config SHA-256 mismatch")
        if hashes.get("trace") != file_sha256(trace_path):
            raise RuntimeError("v14 grasp rescue trace SHA-256 mismatch")
        return config_path, result_path, trace_path
    bundle = authenticate_v14_candidate_artifacts(
        root,
        expected_candidate_id=int(record["candidate_id"]),
    )
    persisted = bundle.result
    if int(persisted.get("candidate_id", -1)) != int(record["candidate_id"]):
        raise RuntimeError("v14 grasp rescue result candidate ID changed")
    if bool(record.get("grasp_success", False)) and bundle.trace_path is None:
        raise RuntimeError("successful v14 grasp rescue lost its retained trace")
    if canonical_sha256(persisted) != canonical_sha256(
        {key: value for key, value in record.items() if key not in {
            "artifact_directory", "edge_m", "mapping_mode", "family", "group_id",
            "local_index", "source_candidate_id",
        }}
    ):
        # The group wrapper may contain bookkeeping fields, but every actual
        # result field must remain byte-for-byte semantically identical.
        for key, value in persisted.items():
            if record.get(key) != value:
                raise RuntimeError(f"v14 grasp rescue result field changed: {key}")
    return bundle.artifact_paths


def _rescue_record_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    stage = record.get("summary", {}).get("stage_status", {})
    metrics = record.get("summary", {}).get("metrics", {})
    actual = metrics.get("actual_grasp_pose", {}) if isinstance(metrics, Mapping) else {}
    actual_metrics = actual.get("metrics", {}) if isinstance(actual, Mapping) else {}
    pose = metrics.get("pose_preservation", {}) if isinstance(metrics, Mapping) else {}
    margin = float(actual.get("minimum_normalized_margin", -math.inf)) if isinstance(actual, Mapping) else -math.inf
    translation = float(pose.get("max_translation_m", math.inf)) if isinstance(pose, Mapping) else math.inf
    thumb = float(actual_metrics.get("thumb_actual_median_rad", math.nan)) if isinstance(actual_metrics, Mapping) else math.nan
    return (
        not bool(isinstance(stage, Mapping) and stage.get("grasp_success", False)),
        -margin,
        translation,
        abs(thumb - 1.50) if math.isfinite(thumb) else math.inf,
        int(record["candidate_id"]),
    )


def _rescue_source_from_record(
    workspace: Path, record: Mapping[str, Any]
) -> AuthenticatedV13GraspSource:
    _authenticate_rescue_record(workspace, record)
    root = (workspace / str(record["artifact_directory"])).resolve()
    bundle = authenticate_v14_candidate_artifacts(
        root,
        expected_candidate_id=int(record["candidate_id"]),
        require_retained_trace=True,
    )
    config_path = bundle.config_path
    result_path = bundle.result_path
    trace_path = bundle.trace_path
    assert trace_path is not None
    result = bundle.result
    stage = result.get("summary", {}).get("stage_status", {})
    if not isinstance(stage, Mapping) or stage.get("grasp_success") is not True:
        raise RuntimeError("non-successful grasp rescue cannot seed planning")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return AuthenticatedV13GraspSource(
        candidate_id=int(record["candidate_id"]),
        edge_m=float(record["edge_m"]),
        mapping_mode=str(record["mapping_mode"]),
        config=config,
        summary=copy.deepcopy(dict(result["summary"])),
        config_path=config_path,
        trace_path=trace_path,
        config_sha256=file_sha256(config_path),
        trace_sha256=file_sha256(trace_path),
        result_semantic_sha256=str(result["result_semantic_sha256"]),
        published_in_catalog=False,
    )


def _load_or_run_rescue_group(
    template: Mapping[str, Any],
    definition: ExperimentDefinition,
    workspace: Path,
    jobs: Sequence[GraspRescueJob],
    base_config: Mapping[str, Any],
    *,
    workers: int,
    seed: int,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    if not jobs or len({value.group_id for value in jobs}) != 1:
        raise ValueError("grasp rescue group must be non-empty and homogeneous")
    group_id = jobs[0].group_id
    group_root = workspace / "grasp_rescue" / "groups" / group_id
    report_path = group_root / "report.json"
    input_payload = {
        "schema_version": 1,
        "group_id": group_id,
        "seed": int(seed),
        "base_config_sha256": canonical_sha256(base_config),
        "jobs": [value.descriptor() for value in jobs],
    }
    input_sha = canonical_sha256(input_payload)
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("complete") is not True
            or report.get("group_input_sha256") != input_sha
            or int(report.get("candidate_count", -1)) != len(jobs)
        ):
            raise RuntimeError(f"v14 grasp rescue group resume changed: {group_id}")
        artifacts: list[Path] = [report_path]
        for record in report.get("records", ()):
            artifacts.extend(_authenticate_rescue_record(workspace, record))
        return copy.deepcopy(dict(report)), tuple(artifacts)

    run_jobs: list[dict[str, Any]] = []
    metadata: dict[int, GraspRescueJob] = {}
    for job in jobs:
        config = _local_rescue_config(
            base_config, definition, job, seed=int(seed)
        )
        destination = group_root / "candidates" / f"candidate_{job.candidate_id}"
        run_jobs.append(
            {
                "config": config,
                "destination": str(destination),
                "candidate_id": job.candidate_id,
                # A selected grasp rescue becomes an authenticated planning
                # source and therefore needs its exact lock trace.  Ordinary
                # manipulation search/refinement candidates opt out below.
                "retain_grasp_success": True,
            }
        )
        metadata[job.candidate_id] = job
    if int(workers) == 1:
        raw_records = tuple(_run_candidate_job(value) for value in run_jobs)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as executor:
            raw_records = tuple(executor.map(_run_candidate_job, run_jobs))
    records: list[dict[str, Any]] = []
    artifacts = [report_path]
    for raw in sorted(raw_records, key=lambda value: int(value["candidate_id"])):
        job = metadata[int(raw["candidate_id"])]
        destination = group_root / "candidates" / f"candidate_{job.candidate_id}"
        record = {
            **copy.deepcopy(dict(raw)),
            "artifact_directory": str(destination.relative_to(workspace)),
            "edge_m": job.edge_m,
            "mapping_mode": job.mapping_mode,
            "family": job.family,
            "group_id": job.group_id,
            "local_index": job.local_index,
            "source_candidate_id": job.source_candidate_id,
        }
        records.append(record)
        artifacts.extend(_candidate_artifact_paths(destination, raw))
    ranked = sorted(records, key=_rescue_record_rank)
    selected = [
        value
        for value in ranked
        if bool(value.get("grasp_success", False))
    ][: int(_campaign(definition).grasp_rescue_retain_per_edge_mapping)]
    failures = Counter(
        str(value.get("summary", {}).get("stage_status", {}).get("failure_reason", "unknown"))
        for value in records
        if not bool(value.get("grasp_success", False))
    )
    report = {
        "v14_grasp_rescue_group_report_schema_version": 1,
        "complete": True,
        "group_id": group_id,
        "group_input_sha256": input_sha,
        "family": jobs[0].family,
        "edge_m": jobs[0].edge_m,
        "mapping_mode": jobs[0].mapping_mode,
        "declared_candidate_count": jobs[0].group_budget,
        "candidate_count": len(records),
        "grasp_success_count": sum(bool(value.get("grasp_success", False)) for value in records),
        "selected_candidate_ids": [int(value["candidate_id"]) for value in selected],
        "failure_reason_counts": dict(sorted(failures.items())),
        "records": records,
    }
    write_json(report_path, report)
    return report, tuple(artifacts)


def _default_grasp_rescue_runner(
    template: Mapping[str, Any],
    definition: ExperimentDefinition,
    bundle: AuthenticatedV13SourceBundle,
    rescue_anchors: AuthenticatedV13RescueAnchors,
    workspace: Path,
    *,
    workers: int,
    resume: bool,
) -> Mapping[str, Any]:
    """Execute all registered cells without global early stopping."""

    del resume
    schedule = build_grasp_rescue_jobs(definition, bundle, rescue_anchors)
    grouped: dict[str, list[GraspRescueJob]] = defaultdict(list)
    for job in schedule:
        grouped[job.group_id].append(job)
    anchor_by_id = {value.candidate_id: value for value in rescue_anchors.anchors}
    measured_by_id = {value.candidate_id: value for value in bundle.sources}
    group_reports: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    selected_records: list[dict[str, Any]] = []
    for group_id in dict.fromkeys(value.group_id for value in schedule):
        jobs = tuple(grouped[group_id])
        source_id = jobs[0].source_candidate_id
        if jobs[0].family == "edge_mapping_local":
            base = materialize_v14_static_rescue_config(
                template, definition, anchor_by_id[source_id]
            )
        else:
            base = materialize_v14_source_pair_config(
                template, definition, measured_by_id[source_id]
            )
        report, group_artifacts = _load_or_run_rescue_group(
            template,
            definition,
            workspace,
            jobs,
            base,
            workers=int(workers),
            seed=int(_campaign(definition).seed),
        )
        artifacts.extend(group_artifacts)
        selected_ids = set(int(value) for value in report["selected_candidate_ids"])
        selected_records.extend(
            copy.deepcopy(dict(value))
            for value in report["records"]
            if int(value["candidate_id"]) in selected_ids
        )
        group_reports.append(
            {key: copy.deepcopy(value) for key, value in report.items() if key != "records"}
            | {"report_path": str((workspace / "grasp_rescue" / "groups" / group_id / "report.json").relative_to(workspace))}
        )
    expected_regular_groups = len(_campaign(definition).edges_m) * 2
    if len(group_reports) != expected_regular_groups + 2:
        raise RuntimeError("v14 grasp rescue skipped a declared group")
    payload = {
        "v14_grasp_rescue_report_schema_version": GRASP_RESCUE_REPORT_SCHEMA_VERSION,
        "complete": True,
        "static_anchor_report_sha256": rescue_anchors.report_sha256,
        "declared_edge_mapping_group_count": expected_regular_groups,
        "executed_edge_mapping_group_count": sum(
            value["family"] == "edge_mapping_local" for value in group_reports
        ),
        "priority_group_count": 2,
        "declared_candidate_count": len(schedule),
        "executed_candidate_count": sum(int(value["candidate_count"]) for value in group_reports),
        "grasp_success_count": sum(int(value["grasp_success_count"]) for value in group_reports),
        "selected_count": len(selected_records),
        "groups": group_reports,
        "selected_records": selected_records,
    }
    return {"report": payload, "artifacts": tuple(dict.fromkeys(artifacts))}


@dataclass(frozen=True, slots=True)
class CampaignBackend:
    source_authenticator: Callable[[ExperimentDefinition], AuthenticatedV13SourceBundle] = authenticate_v13_grasp_sources
    grasp_rescue_runner: Callable[..., Mapping[str, Any]] = _default_grasp_rescue_runner
    plan_runner: Callable[[Mapping[str, Any], AuthenticatedV13GraspSource, Path], Sequence[Mapping[str, Any]]] = _default_plan_runner
    candidate_runner: Callable[..., Sequence[Mapping[str, Any]]] = _default_candidate_runner
    joint_refinement_runner: Callable[..., Mapping[str, Any]] = (
        _default_joint_refinement_runner
    )
    robustness_runner: Callable[..., Mapping[str, Any]] | None = None
    catalog_render_videos: bool = False
    catalog_simulation_runner: Callable[..., Mapping[str, Any]] | None = None
    catalog_video_probe: Callable[..., Mapping[str, Any]] | None = None


def _default_robustness_runner(
    catalog_path: Path,
    output_path: Path,
    *,
    workers: int,
    seed: int,
    campaign: Any,
) -> dict[str, Any]:
    """Run 16-per-published plus 50 best-pair full-reset perturbations."""

    from .actual_contact_grasp_pose_robustness import (
        discover_v9_robustness_sources,
        run_v9_robustness_campaign,
    )

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    aliases = catalog.get("aliases", {})
    best_id = aliases.get("best_nominal") if isinstance(aliases, Mapping) else None
    trajectories = {
        str(value.get("trajectory_id")): value
        for value in catalog.get("trajectories", ())
        if isinstance(value, Mapping)
    }
    best = trajectories.get(str(best_id))
    if not isinstance(best, Mapping):
        raise RuntimeError("v14 manipulation catalog has no best_nominal success")
    best_candidate_id = str(best["candidate_id"])
    discovered = discover_v9_robustness_sources((catalog_path,))
    rebound = tuple(
        replace(value, best_first=value.candidate_id == best_candidate_id)
        for value in discovered
    )
    if not any(value.best_first for value in rebound):
        raise RuntimeError("v14 best_nominal is absent from robustness sources")
    result = run_v9_robustness_campaign(
        (catalog_path,),
        output_path,
        workers=int(workers),
        seed=int(seed),
        local_perturbations=int(campaign.perturbations_per_final),
        best_perturbations=int(campaign.robustness_trials),
        max_nominal_trajectories=int(campaign.final_candidate_count),
        candidate_discoverer=lambda _roots: rebound,
        best_selection="local_perturbation_passes",
    )
    best_report = result.get("best_robustness", {})
    passes = (
        int(best_report.get("perturbation_passes", 0))
        if isinstance(best_report, Mapping)
        else 0
    )
    return {
        **result,
        "contact_preserving_robustness_schema_version": 1,
        "best_50_selection_alias": "best_robust_local_16_leader",
        "best_50_candidate_id": str(best_report.get("candidate_id", "")),
        "nominal_best_candidate_id": best_candidate_id,
        "required_passes": int(campaign.robustness_required_passes),
        "robust_success": bool(
            result.get("robust_passed", False)
            and passes >= int(campaign.robustness_required_passes)
        ),
    }


DEFAULT_BACKEND = CampaignBackend(
    robustness_runner=_default_robustness_runner,
    catalog_render_videos=True,
)


def _catalog_entry_from_candidate(
    record: Mapping[str, Any],
    workspace: Path,
    destination: Path,
    rank: int,
    *,
    render_video: bool,
    simulation_runner: Callable[..., Mapping[str, Any]],
    video_probe: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_id = int(record["candidate_id"])
    source_root = (workspace / str(record["artifact_directory"])).resolve()
    label = f"pair_rank_{rank:02d}_{candidate_id}"
    target = destination / label
    artifacts = record["artifacts"]
    copied: dict[str, str] = {}
    hashes: dict[str, str] = {}
    source_config = _safe_relative(
        source_root, artifacts["resolved_config"], "resolved_config"
    )
    source_trace = _safe_relative(source_root, artifacts["trace"], "trace")
    target_config = target / source_config.name
    target_trace = target / source_trace.name
    _copy_or_link(source_config, target_config)
    resolved_config = json.loads(target_config.read_text(encoding="utf-8"))
    copied["resolved_config"] = str(target_config.relative_to(destination))
    copied["trace"] = str(target_trace.relative_to(destination))
    hashes["resolved_config"] = file_sha256(target_config)
    source_result = source_root / "result.json"
    target_result = target / "result.json"
    source_result_payload = json.loads(source_result.read_text(encoding="utf-8"))
    final_video_evidence: dict[str, Any] | None = None
    video_path: Path | None = None
    if render_video:
        video_path = target / "trajectory.mp4"
        config = copy.deepcopy(resolved_config)
        rerun_summary = simulation_runner(
            config,
            trace_path=target_trace,
            video_path=video_path,
        )
        if not isinstance(rerun_summary, Mapping):
            raise RuntimeError("v14 catalog simulation runner returned no summary")
        expected_summary = copy.deepcopy(dict(record.get("summary", {})))
        actual_summary = copy.deepcopy(dict(rerun_summary))
        expected_summary.pop("video", None)
        actual_summary.pop("video", None)
        if json_text(expected_summary) != json_text(actual_summary):
            raise RuntimeError(
                "v14 catalog video rerun changed the deterministic result summary"
            )
        trace_evidence = _verify_catalog_video_trace(source_trace, target_trace)
        declared_video = rerun_summary.get("video")
        if not isinstance(declared_video, Mapping):
            raise RuntimeError("v14 catalog video rerun returned no video evidence")
        frame_steps = trace_evidence["video_frame_steps"]
        if list(declared_video.get("simulation_step_indices", ())) != frame_steps:
            raise RuntimeError("v14 MP4 summary and NPZ frame binding disagree")
        inspected = copy.deepcopy(
            dict(video_probe(video_path, len(frame_steps), VideoSettings()))
        )
        for field in (
            "decode_verified",
            "codec",
            "width",
            "height",
            "fps",
            "frame_count",
        ):
            if declared_video.get(field) != inspected.get(field):
                raise RuntimeError(
                    f"v14 MP4 summary disagrees with decoded field {field}"
                )
        if inspected.get("decode_verified") is not True:
            raise RuntimeError("v14 catalog MP4 did not pass a complete decode")
        final_video_evidence = {
            "rendered_from_initial_no_contact_state": True,
            "checkpoint_used": False,
            "trace_reproduction": trace_evidence,
            "ffprobe_and_full_decode": inspected,
        }
        result_payload = copy.deepcopy(source_result_payload)
        result_payload["summary"] = copy.deepcopy(dict(rerun_summary))
        result_payload["final_video_publication"] = copy.deepcopy(
            final_video_evidence
        )
        result_payload["artifacts"] = {
            "resolved_config": target_config.name,
            "trace": target_trace.name,
            "video": video_path.name,
            "sha256": {
                "resolved_config": file_sha256(target_config),
                "trace": file_sha256(target_trace),
                "video": file_sha256(video_path),
            },
        }
        result_payload = bind_candidate_result_semantic_sha256(result_payload)
        write_json(target_result, result_payload)
    else:
        _copy_or_link(source_trace, target_trace)
        _copy_or_link(source_result, target_result)
    hashes["trace"] = file_sha256(target_trace)
    copied["result"] = str(target_result.relative_to(destination))
    hashes["result"] = file_sha256(target_result)
    if video_path is not None:
        hashes["video"] = file_sha256(video_path)
    return {
        "trajectory_id": label,
        "candidate_id": str(candidate_id),
        "classification": record["classification"],
        "full_success": bool(record.get("full_success", False)),
        "grasp_success": bool(record.get("grasp_success", False)),
        "edge_m": float(resolved_config["cube"]["edge_m"]),
        "object_config_id": resolved_config.get("object_config_id"),
        "grasp_pose_id": resolved_config.get("grasp_pose_id"),
        "grasp_object_pair_id": resolved_config.get("grasp_object_pair_id"),
        "planner_id": resolved_config.get("planner_id"),
        "controller_id": resolved_config.get("controller_id"),
        "validation_label": resolve_experiment(resolved_config)
        .contact_preserving_planned_lift_campaign
        .validation_labels[
            "manipulation" if bool(record.get("full_success", False)) else "grasp"
        ],
        "final_video_required": bool(render_video),
        "final_video_verified": bool(
            final_video_evidence is not None
            and final_video_evidence.get("ffprobe_and_full_decode", {}).get(
                "decode_verified", False
            )
        ),
        "final_video_evidence": final_video_evidence,
        "artifacts": {
            **copied,
            "video": (
                str(video_path.relative_to(destination))
                if video_path is not None
                else None
            ),
            "sha256": hashes,
        },
    }


def _verify_catalog_video_trace(source_path: Path, rendered_path: Path) -> dict[str, Any]:
    """Prove that enabling the renderer changed no persisted physics field."""

    ignored = {"video_frame_steps"}
    with np.load(source_path, allow_pickle=False) as source, np.load(
        rendered_path, allow_pickle=False
    ) as rendered:
        source_names = set(source.files) - ignored
        rendered_names = set(rendered.files) - ignored
        if source_names != rendered_names:
            raise RuntimeError(
                "v14 catalog video rerun trace schema changed: "
                f"missing={sorted(source_names - rendered_names)}, "
                f"added={sorted(rendered_names - source_names)}"
            )
        mismatches: list[str] = []
        for name in sorted(source_names):
            left = np.asarray(source[name])
            right = np.asarray(rendered[name])
            if left.shape != right.shape or left.dtype != right.dtype:
                mismatches.append(name)
                continue
            if left.dtype.kind in "fc":
                equal = np.array_equal(left, right, equal_nan=True)
            else:
                equal = np.array_equal(left, right)
            if not equal:
                mismatches.append(name)
        if mismatches:
            raise RuntimeError(
                "v14 catalog video rerun changed physical trajectory fields: "
                + ", ".join(mismatches)
            )
        frame_steps = np.asarray(
            rendered["video_frame_steps"]
            if "video_frame_steps" in rendered
            else (),
            dtype=np.int64,
        )
        if frame_steps.ndim != 1 or frame_steps.size == 0:
            raise RuntimeError("v14 catalog MP4 has no frame-to-step binding")
    return {
        "source_trace_sha256": file_sha256(source_path),
        "rendered_trace_sha256": file_sha256(rendered_path),
        "compared_field_count": len(source_names),
        "ignored_fields": sorted(ignored),
        "physical_fields_exact": True,
        "video_frame_steps": frame_steps.tolist(),
    }


def _selected_catalog_candidate_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Apply the catalog's deterministic top-five plus first-success rule."""

    ranked = rank_contact_constrained_candidates(records)
    first_success = min(
        (value for value in records if bool(value.get("full_success", False))),
        key=lambda value: (
            int(value.get("plan_rank", 1 << 30)),
            int(value.get("feedback_index", 1 << 30)),
            int(value["candidate_id"]),
        ),
        default=None,
    )
    selected = [copy.deepcopy(dict(value)) for value in ranked[:5]]
    if first_success is not None and all(
        int(value["candidate_id"]) != int(first_success["candidate_id"])
        for value in selected
    ):
        if selected:
            selected[-1] = copy.deepcopy(dict(first_success))
        else:
            selected.append(copy.deepcopy(dict(first_success)))
        selected = list(rank_contact_constrained_candidates(selected))
    return tuple(selected)


def _ensure_catalog_candidate_traces(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    *,
    target_success_count: int,
) -> tuple[dict[str, Any], ...]:
    """Rerun only selected summary-only candidates to obtain publishable NPZs."""

    selected_ids = {
        int(value["candidate_id"])
        for value in _selected_catalog_candidate_records(records)
    }
    materialized: list[dict[str, Any]] = []
    for raw in records:
        record = copy.deepcopy(dict(raw))
        candidate_id = int(record["candidate_id"])
        if candidate_id not in selected_ids:
            materialized.append(record)
            continue
        source_root = workspace / str(record["artifact_directory"])
        trace_name = record.get("artifacts", {}).get("trace")
        if isinstance(trace_name, str) and (source_root / trace_name).is_file():
            materialized.append(record)
            continue
        source_config = json.loads(
            (source_root / "resolved_config.json").read_text(encoding="utf-8")
        )
        final_root = (
            workspace
            / "catalog_source_reruns"
            / f"target_{int(target_success_count)}"
            / f"candidate_{candidate_id}"
        )
        rerun = _run_full_reset_candidate(
            source_config,
            final_root,
            candidate_id,
            final_rerun=True,
        )
        if canonical_sha256(rerun.get("summary")) != canonical_sha256(
            record.get("summary")
        ):
            raise RuntimeError(
                "v14 final catalog rerun changed deterministic candidate summary"
            )
        if (
            bool(rerun.get("full_success", False))
            != bool(record.get("full_success", False))
            or bool(rerun.get("grasp_success", False))
            != bool(record.get("grasp_success", False))
        ):
            raise RuntimeError("v14 final catalog rerun changed candidate status")
        materialized.append(
            {
                **record,
                **rerun,
                "artifact_directory": str(final_root.relative_to(workspace)),
            }
        )
    return tuple(materialized)


def _bind_catalog_entry_aliases(
    entries: Sequence[dict[str, Any]], aliases: Mapping[str, str]
) -> None:
    """Mirror the authoritative catalog alias map into trajectory entries.

    Viewer catalogs intentionally carry both directions so a corrupted alias
    cannot silently select a different trajectory.  Validate the complete map
    before mutating entries, then rebuild every reverse alias list to avoid
    retaining stale names when a rescue publisher restricts the catalog.
    """

    entries_by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("catalog trajectories must be mutable mappings")
        trajectory_id = entry.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise RuntimeError("catalog trajectory_id must be a non-empty string")
        if trajectory_id in entries_by_id:
            raise RuntimeError(f"duplicate catalog trajectory_id: {trajectory_id}")
        entries_by_id[trajectory_id] = entry

    reverse: dict[str, list[str]] = {value: [] for value in entries_by_id}
    for alias, trajectory_id in aliases.items():
        if not isinstance(alias, str) or not alias:
            raise RuntimeError("catalog alias must be a non-empty string")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise RuntimeError(f"catalog alias {alias!r} has an invalid target")
        if trajectory_id not in entries_by_id:
            raise RuntimeError(
                f"catalog alias {alias!r} targets missing trajectory {trajectory_id!r}"
            )
        reverse[trajectory_id].append(alias)

    for trajectory_id, entry in entries_by_id.items():
        entry["aliases"] = sorted(reverse[trajectory_id])


def publish_viewer_catalogs(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    destination: Path,
    *,
    experiment_id: str,
    robust_candidate_id: int | None = None,
    render_videos: bool = False,
    simulation_runner: Callable[..., Mapping[str, Any]] | None = None,
    video_probe_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Publish contact-first manipulation evidence with truthful aliases."""

    first_success = min(
        (value for value in records if bool(value.get("full_success", False))),
        key=lambda value: (
            int(value.get("plan_rank", 1 << 30)),
            int(value.get("feedback_index", 1 << 30)),
            int(value["candidate_id"]),
        ),
        default=None,
    )
    selected = list(_selected_catalog_candidate_records(records))
    execute = run_simulation if simulation_runner is None else simulation_runner
    inspect_video = probe_video if video_probe_runner is None else video_probe_runner
    catalog_root = destination / "manipulation"
    if catalog_root.exists():
        shutil.rmtree(catalog_root)
    catalog_root.mkdir(parents=True)
    entries = [
        _catalog_entry_from_candidate(
            value,
            workspace,
            catalog_root,
            index + 1,
            render_video=bool(render_videos),
            simulation_runner=execute,
            video_probe=inspect_video,
        )
        for index, value in enumerate(selected)
    ]
    aliases = {
        f"pair_rank_{index + 1:02d}": entry["trajectory_id"]
        for index, entry in enumerate(entries)
    }
    successes = [entry for entry in entries if entry["classification"] == "success"]
    if successes:
        first_entry = next(
            (
                entry
                for entry in successes
                if first_success is not None
                and int(entry["candidate_id"])
                == int(first_success["candidate_id"])
            ),
            successes[0],
        )
        aliases["best_first"] = first_entry["trajectory_id"]
        aliases["best_nominal"] = successes[0]["trajectory_id"]
    elif entries:
        aliases["best_attempt"] = entries[0]["trajectory_id"]
    if robust_candidate_id is not None:
        matching = [
            entry for entry in entries if int(entry["candidate_id"]) == robust_candidate_id
        ]
        if matching and matching[0]["classification"] == "success":
            aliases["best_robust"] = matching[0]["trajectory_id"]
    _bind_catalog_entry_aliases(entries, aliases)
    entry_by_trajectory = {value["trajectory_id"]: value for value in entries}
    best_pairs = {
        alias: {
            key: entry_by_trajectory[trajectory].get(key)
            for key in (
                "candidate_id",
                "edge_m",
                "object_config_id",
                "grasp_pose_id",
                "grasp_object_pair_id",
                "planner_id",
                "controller_id",
                "validation_label",
            )
        }
        for alias, trajectory in aliases.items()
        if alias in {"best_first", "best_nominal", "best_robust", "best_attempt"}
    }
    catalog = {
        "contact_preserving_viewer_catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "trajectory_catalog_schema_version": 1,
        "catalog_kind": "manipulation",
        "complete": True,
        "experiment_id": experiment_id,
        "selection_policy": (
            "full_hard_success_then_perturbation_passes_then_contact_force_"
            "and_path_quality"
        ),
        "production_trajectory_video_policy": (
            "deterministic_full_reset_rerun_ffprobe_and_full_decode"
            if render_videos
            else "disabled_for_injected_test_backend"
        ),
        "success_count": len(successes),
        "aliases": aliases,
        "best_grasp_object_pairs": best_pairs,
        "trajectories": entries,
    }
    catalog_path = catalog_root / "catalog.json"
    write_json(catalog_path, catalog)
    # A grasp catalog is deliberately built only from the same full-reset v14
    # evidence.  Successful grasps are preferred; with none, one diagnostic is
    # retained rather than copying and relabeling a v13 source trace.
    grasp_entries = [copy.deepcopy(value) for value in entries if value["grasp_success"]]
    grasp_success_count = len(grasp_entries)
    if not grasp_entries and entries:
        # Preserve one full-reset diagnostic even if every candidate failed to
        # reacquire the imported grasp.  It is intentionally not counted or
        # labeled as grasp success.
        grasp_entries = [copy.deepcopy(entries[0])]
    grasp_root = destination / "grasp_pose"
    if grasp_root.exists():
        shutil.rmtree(grasp_root)
    grasp_root.mkdir(parents=True)
    for entry in grasp_entries:
        # A v14 manipulation candidate can fail after a strictly valid grasp.
        # In the grasp-only catalog that same full-reset trace is success
        # evidence for the grasp stage, while ``full_success`` remains false.
        entry["classification"] = (
            "success" if entry["grasp_success"] else "diagnostic"
        )
        for artifact_name in ("resolved_config", "result", "trace", "video"):
            if entry["artifacts"].get(artifact_name) is None:
                continue
            old = catalog_root / entry["artifacts"][artifact_name]
            new = grasp_root / entry["artifacts"][artifact_name]
            _copy_or_link(old, new)
    grasp_aliases = {
        f"pair_rank_{index + 1:02d}": entry["trajectory_id"]
        for index, entry in enumerate(grasp_entries)
    }
    if grasp_success_count:
        grasp_aliases["best_first"] = grasp_entries[0]["trajectory_id"]
        grasp_aliases["best_nominal"] = grasp_entries[0]["trajectory_id"]
    elif grasp_entries:
        grasp_aliases["best_attempt"] = grasp_entries[0]["trajectory_id"]
    _bind_catalog_entry_aliases(grasp_entries, grasp_aliases)
    grasp_catalog = {
        "contact_preserving_viewer_catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "trajectory_catalog_schema_version": 1,
        "catalog_kind": "grasp_pose",
        "complete": True,
        "experiment_id": experiment_id,
        "production_trajectory_video_policy": catalog[
            "production_trajectory_video_policy"
        ],
        "success_count": grasp_success_count,
        "aliases": grasp_aliases,
        "trajectories": grasp_entries,
    }
    grasp_path = grasp_root / "catalog.json"
    write_json(grasp_path, grasp_catalog)
    return {
        "grasp_pose": str(grasp_path.relative_to(workspace)),
        "manipulation": str(catalog_path.relative_to(workspace)),
    }


def _publish_robust_alias_catalog(
    source_catalog: Path,
    destination: Path,
    *,
    robust_candidate_id: str,
) -> Path:
    """Copy a committed nominal catalog and add a truthful robust alias."""

    if destination.exists():
        catalog_path = destination / "catalog.json"
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        if payload.get("aliases", {}).get("best_robust") != payload.get(
            "aliases", {}
        ).get("best_nominal"):
            raise RuntimeError("existing robust catalog alias changed")
        authenticated_catalog_artifact_paths(catalog_path)
        return catalog_path
    shutil.copytree(source_catalog.parent, destination, copy_function=os.link)
    catalog_path = destination / "catalog.json"
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    aliases = payload.get("aliases")
    if not isinstance(aliases, dict) or "best_nominal" not in aliases:
        raise RuntimeError("nominal catalog cannot seed best_robust")
    matching = [
        value
        for value in payload.get("trajectories", ())
        if str(value.get("candidate_id")) == str(robust_candidate_id)
        and value.get("classification") == "success"
    ]
    if len(matching) != 1:
        raise RuntimeError("robust candidate is absent from the nominal catalog")
    aliases["best_robust"] = matching[0]["trajectory_id"]
    _bind_catalog_entry_aliases(payload.get("trajectories", []), aliases)
    payload.setdefault("best_grasp_object_pairs", {})["best_robust"] = {
        key: matching[0].get(key)
        for key in (
            "candidate_id",
            "edge_m",
            "object_config_id",
            "grasp_pose_id",
            "grasp_object_pair_id",
            "planner_id",
            "controller_id",
            "validation_label",
        )
    }
    payload["catalog_kind"] = "robust_manipulation"
    payload["robustness_selection"] = (
        "best_local_16_perturbation_leader_passed_at_least_45_of_50"
    )
    write_json(catalog_path, payload)
    authenticated_catalog_artifact_paths(catalog_path)
    return catalog_path


def _validate_grasp_rescue_report(
    report: Mapping[str, Any],
    definition: ExperimentDefinition,
    schedule: Sequence[GraspRescueJob],
) -> None:
    campaign = _campaign(definition)
    expected_groups = len(campaign.edges_m) * 2
    if (
        report.get("complete") is not True
        or int(report.get("declared_edge_mapping_group_count", -1)) != expected_groups
        or int(report.get("executed_edge_mapping_group_count", -1)) != expected_groups
        or int(report.get("priority_group_count", -1)) != 2
        or int(report.get("declared_candidate_count", -1)) != len(schedule)
        or int(report.get("executed_candidate_count", -1)) != len(schedule)
    ):
        raise RuntimeError("v14 grasp rescue did not exhaust its registered budget")
    groups = report.get("groups")
    if not isinstance(groups, list) or len(groups) != expected_groups + 2:
        raise RuntimeError("v14 grasp rescue group matrix is incomplete")
    observed = [str(value.get("group_id", "")) for value in groups if isinstance(value, Mapping)]
    expected = list(dict.fromkeys(value.group_id for value in schedule))
    if observed != expected or len(set(observed)) != len(observed):
        raise RuntimeError("v14 grasp rescue group order/coverage changed")
    selected = report.get("selected_records")
    if not isinstance(selected, list) or int(report.get("selected_count", -1)) != len(selected):
        raise RuntimeError("v14 grasp rescue selected-record count changed")


def run_contact_preserving_planned_lift_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool,
    target_success_count: int,
    workers: int,
    seed: int = DEFAULT_SEED,
    evidence_anchor_paths: Sequence[str | Path] = (),
    backend: CampaignBackend = DEFAULT_BACKEND,
) -> dict[str, Any]:
    """Execute/resume authenticated v14 pair planning and contact-first search."""

    workers = int(workers)
    if workers <= 0:
        raise ValueError("workers must be positive")
    if evidence_anchor_paths:
        raise ValueError("schema-v14 sources are fixed by the registered v13 catalog")
    if target_success_count not in (1, 5):
        raise ValueError("target_success_count must be one or five")
    config_file = Path(config_path).expanduser().resolve()
    workspace = Path(output_dir).expanduser().resolve()
    template = load_config(config_file)
    definition = resolve_experiment(template)
    campaign = _campaign(definition)
    if int(seed) != int(campaign.seed):
        raise ValueError("schema-v14 seed differs from the registered campaign")
    manifest = build_contact_preserving_planned_lift_manifest(config_file, seed=seed)
    initialize_or_resume_campaign(workspace, manifest, resume=resume)

    # Re-audit immutable external input on every invocation, even after commit.
    bundle = backend.source_authenticator(definition)
    source_payload = bundle.descriptor()
    source_path = workspace / "source_audit.json"
    existing = _load_committed_json(workspace, "source_audit", source_path)
    if existing is None:
        _commit_json_stage(
            workspace,
            "source_audit",
            source_path,
            source_payload,
            stage_input={
                "catalog_sha256": bundle.catalog_sha256,
                "measured_report_sha256": bundle.measured_report_sha256,
            },
        )
    elif canonical_sha256(existing) != canonical_sha256(source_payload):
        raise RuntimeError("schema-v14 source audit changed on resume")

    rescue_anchors = authenticate_v13_grasp_rescue_anchors(definition, bundle)
    rescue_schedule = build_grasp_rescue_jobs(definition, bundle, rescue_anchors)
    rescue_path = workspace / "grasp_rescue" / "report.json"
    rescue_report = _load_committed_json(
        workspace, "grasp_pose_rescue", rescue_path
    )
    if rescue_report is None:
        rescue_execution = backend.grasp_rescue_runner(
            template,
            definition,
            bundle,
            rescue_anchors,
            workspace,
            workers=workers,
            resume=resume,
        )
        if not isinstance(rescue_execution, Mapping):
            raise RuntimeError("v14 grasp rescue runner returned no execution")
        raw_report = rescue_execution.get("report")
        raw_artifacts = rescue_execution.get("artifacts", ())
        if not isinstance(raw_report, Mapping) or not isinstance(
            raw_artifacts, Sequence
        ):
            raise RuntimeError("v14 grasp rescue runner returned malformed output")
        rescue_report = copy.deepcopy(dict(raw_report))
        _validate_grasp_rescue_report(rescue_report, definition, rescue_schedule)
        rescue_report = _commit_json_stage(
            workspace,
            "grasp_pose_rescue",
            rescue_path,
            rescue_report,
            stage_input={
                "source_audit_sha256": file_sha256(source_path),
                "static_anchor_report_sha256": rescue_anchors.report_sha256,
                "schedule_sha256": canonical_sha256(
                    [value.descriptor() for value in rescue_schedule]
                ),
                "budget": campaign.budget_config(),
            },
            extra_artifacts=tuple(Path(value) for value in raw_artifacts),
        )
    _validate_grasp_rescue_report(rescue_report, definition, rescue_schedule)
    rescue_sources = tuple(
        _rescue_source_from_record(workspace, value)
        for value in rescue_report["selected_records"]
    )

    pair_path = workspace / "grasp_object_pairs.json"
    pair_report = _load_committed_json(workspace, "grasp_object_pairs", pair_path)
    if pair_report is None:
        pairs = build_v14_source_pairs(
            template, definition, bundle, rescue_sources
        )
        pair_config_paths: list[Path] = []
        pair_records: list[dict[str, Any]] = []
        for pair in pairs:
            pair_config = workspace / "pairs" / str(pair["grasp_object_pair_id"]) / "resolved_config.json"
            write_json(pair_config, pair["config"])
            pair_config_paths.append(pair_config)
            pair_records.append({**pair, "config_path": str(pair_config.relative_to(workspace))})
            pair_records[-1].pop("config")
        pair_report = _commit_json_stage(
            workspace,
            "grasp_object_pairs",
            pair_path,
            {
                "contact_preserving_pair_report_schema_version": PAIR_REPORT_SCHEMA_VERSION,
                "complete": True,
                "pair_count": len(pair_records),
                "records": pair_records,
            },
            stage_input={
                "source_audit_sha256": file_sha256(source_path),
                "grasp_rescue_report_sha256": file_sha256(rescue_path),
            },
            extra_artifacts=pair_config_paths,
        )
    pair_records = list(pair_report["records"])
    source_by_id = {
        value.candidate_id: value
        for value in (*bundle.sources, *rescue_sources)
    }

    plan_path = workspace / "planning_report.json"
    plan_report = _load_committed_json(workspace, "offline_planning", plan_path)
    if plan_report is None:
        plans: list[dict[str, Any]] = []
        plan_artifacts: list[Path] = []
        for pair in pair_records[: int(campaign.pair_shortlist_count)]:
            pair_config_path = workspace / pair["config_path"]
            pair_with_config = {**pair, "config": json.loads(pair_config_path.read_text("utf-8"))}
            source = source_by_id[int(pair["source_candidate_id"])]
            pair_root = workspace / "planning" / str(pair["grasp_object_pair_id"])
            generated = backend.plan_runner(pair_with_config, source, pair_root)
            for record in generated:
                config_path_out = pair_root / f"plan_{int(record['candidate_id'])}" / "resolved_config.json"
                write_json(config_path_out, record["config"])
                plan_artifacts.append(config_path_out)
                clean = copy.deepcopy(dict(record))
                clean["config_path"] = str(config_path_out.relative_to(workspace))
                clean.pop("config", None)
                plans.append(clean)
            for evidence_name in (
                "sequential_planning_report.json",
                # Compatibility for injected/legacy planner tests only.
                "extended_probe_response.json",
            ):
                evidence_path = pair_root / evidence_name
                if evidence_path.is_file():
                    plan_artifacts.append(evidence_path)
        if len(plans) > int(campaign.maximum_plan_candidate_count):
            raise RuntimeError("v14 planner exceeded registered plan budget")
        plan_report = _commit_json_stage(
            workspace,
            "offline_planning",
            plan_path,
            {
                "contact_preserving_plan_report_schema_version": PLAN_REPORT_SCHEMA_VERSION,
                "complete": True,
                "shortlisted_pair_count": min(len(pair_records), campaign.pair_shortlist_count),
                "plan_candidate_count": len(plans),
                "authenticated_warm_start_plan_count": sum(
                    bool(value.get("warm_start", False)) for value in plans
                ),
                "terminal_path_rejected_plan_count": sum(
                    bool(value.get("terminal_path_rejected", False))
                    for value in plans
                ),
                "terminal_path_rejection_threshold_m": 0.008,
                "records": plans,
            },
            stage_input={
                "pair_report_sha256": file_sha256(pair_path),
                "budget": campaign.budget_config(),
            },
            extra_artifacts=plan_artifacts,
        )
    plan_records = []
    for record in plan_report["records"]:
        config = json.loads((workspace / record["config_path"]).read_text("utf-8"))
        plan_records.append({**record, "config": config})

    candidate_stage = f"candidate_search_target_{target_success_count}"
    candidate_path = workspace / "candidate_search" / f"target_{target_success_count}.json"
    candidate_report = _load_committed_json(workspace, candidate_stage, candidate_path)
    if candidate_report is None:
        feedback_plan_records = _select_feedback_plan_quota(
            plan_records,
            plans_per_pair=campaign.feedback_plan_candidates_per_pair,
        )
        eligible_plan_count = len(feedback_plan_records)
        candidate_records = backend.candidate_runner(
            feedback_plan_records,
            workspace,
            target_success_count=target_success_count,
            workers=workers,
            feedback_candidates_per_plan=campaign.feedback_candidates_per_plan,
            maximum_candidate_count=(
                campaign.pair_shortlist_count
                * campaign.feedback_plan_candidates_per_pair
                * campaign.feedback_candidates_per_plan
            ),
        )
        candidate_artifacts: list[Path] = []
        for record in candidate_records:
            root = workspace / str(record["artifact_directory"])
            candidate_artifacts.extend(_candidate_artifact_paths(root, record))
        ranked = rank_contact_constrained_candidates(candidate_records)
        full_count = sum(bool(value.get("full_success", False)) for value in ranked)
        candidate_report = _commit_json_stage(
            workspace,
            candidate_stage,
            candidate_path,
            {
                "contact_preserving_candidate_report_schema_version": CANDIDATE_REPORT_SCHEMA_VERSION,
                "complete": True,
                "target_success_count": target_success_count,
                "candidate_count": len(ranked),
                "registered_maximum_candidate_count": (
                    campaign.pair_shortlist_count
                    * campaign.feedback_plan_candidates_per_pair
                    * campaign.feedback_candidates_per_plan
                ),
                "eligible_plan_count": eligible_plan_count,
                "terminal_path_rejected_plan_count": (
                    sum(
                        bool(value.get("terminal_path_rejected", False))
                        for value in plan_records
                    )
                ),
                "feedback_selected_plan_count": eligible_plan_count,
                "feedback_plan_candidates_per_pair": (
                    campaign.feedback_plan_candidates_per_pair
                ),
                "plan_batch_size": campaign.feedback_candidates_per_plan,
                "stop_checked_only_at_plan_batch_boundary": True,
                "full_success_count": full_count,
                "target_reached": full_count >= target_success_count,
                "records": list(ranked),
            },
            stage_input={
                "planning_report_sha256": file_sha256(plan_path),
                "target_success_count": target_success_count,
            },
            extra_artifacts=candidate_artifacts,
        )
    base_candidate_records = tuple(candidate_report["records"])

    refinement_stage = f"joint_refinement_target_{target_success_count}"
    refinement_path = (
        workspace
        / "joint_refinement"
        / f"target_{target_success_count}.json"
    )
    refinement_report = _load_committed_json(
        workspace, refinement_stage, refinement_path
    )
    if refinement_report is None:
        execution = backend.joint_refinement_runner(
            base_candidate_records,
            workspace,
            workers=workers,
            seed=int(campaign.seed),
            parent_count=int(campaign.feedback_refine_plan_count),
            candidates_per_parent=int(campaign.feedback_refine_per_plan),
        )
        if not isinstance(execution, Mapping):
            raise RuntimeError("v14 joint refinement runner returned no execution")
        raw_report = execution.get("report")
        raw_artifacts = execution.get("artifacts", ())
        if not isinstance(raw_report, Mapping) or not isinstance(
            raw_artifacts, Sequence
        ):
            raise RuntimeError("v14 joint refinement runner returned malformed output")
        refinement_report = _commit_json_stage(
            workspace,
            refinement_stage,
            refinement_path,
            copy.deepcopy(dict(raw_report)),
            stage_input={
                "candidate_report_sha256": file_sha256(candidate_path),
                "parent_count": campaign.feedback_refine_plan_count,
                "candidates_per_parent": campaign.feedback_refine_per_plan,
                "seed": campaign.seed,
            },
            extra_artifacts=tuple(Path(value) for value in raw_artifacts),
        )
    refined_records = tuple(refinement_report.get("records", ()))
    candidate_records = rank_contact_constrained_candidates(
        (*base_candidate_records, *refined_records)
    )
    full_count = sum(
        bool(value.get("full_success", False)) for value in candidate_records
    )
    grasp_count = sum(bool(value.get("grasp_success", False)) for value in candidate_records)

    catalog_stage = f"catalog_target_{target_success_count}"
    catalog_report_path = workspace / "catalogs" / f"target_{target_success_count}" / "report.json"
    catalog_report = _load_committed_json(workspace, catalog_stage, catalog_report_path)
    if catalog_report is None:
        catalog_source_records = _ensure_catalog_candidate_traces(
            candidate_records,
            workspace,
            target_success_count=target_success_count,
        )
        catalogs = publish_viewer_catalogs(
            catalog_source_records,
            workspace,
            catalog_report_path.parent,
            experiment_id=definition.experiment_id,
            render_videos=backend.catalog_render_videos,
            simulation_runner=backend.catalog_simulation_runner,
            video_probe_runner=backend.catalog_video_probe,
        )
        catalog_artifacts: list[Path] = []
        for relative in catalogs.values():
            catalog_artifacts.extend(
                authenticated_catalog_artifact_paths(workspace / relative)
            )
        catalog_report = _commit_json_stage(
            workspace,
            catalog_stage,
            catalog_report_path,
            {
                "complete": True,
                "target_success_count": target_success_count,
                "full_success_count": full_count,
                "grasp_success_count": grasp_count,
                "target_reached": full_count >= target_success_count,
                "catalogs": catalogs,
            },
            stage_input={
                "candidate_report_sha256": file_sha256(candidate_path),
                "joint_refinement_report_sha256": file_sha256(refinement_path),
                "target_success_count": target_success_count,
            },
            extra_artifacts=tuple(dict.fromkeys(catalog_artifacts)),
        )
    robustness: dict[str, Any] | None = None
    if full_count and backend.robustness_runner is not None:
        robustness_stage = f"robustness_target_{target_success_count}"
        robustness_path = (
            workspace
            / "robustness"
            / f"target_{target_success_count}"
            / "perturbation_report.json"
        )
        robustness = _load_committed_json(
            workspace, robustness_stage, robustness_path
        )
        if robustness is None:
            manipulation_catalog = workspace / catalog_report["catalogs"]["manipulation"]
            raw_robustness = backend.robustness_runner(
                manipulation_catalog,
                robustness_path,
                workers=workers,
                seed=int(seed),
                campaign=campaign,
            )
            robustness_payload = {
                "complete": True,
                **copy.deepcopy(dict(raw_robustness)),
            }
            # Include every trial result/config/trace written by the robustness
            # runner so resume cannot hide a missing perturbation artifact.
            robustness_artifacts = tuple(
                path
                for path in sorted(robustness_path.parent.rglob("*"))
                if path.is_file() and path != robustness_path
            )
            robustness = _commit_json_stage(
                workspace,
                robustness_stage,
                robustness_path,
                robustness_payload,
                stage_input={
                    "catalog_sha256": file_sha256(manipulation_catalog),
                    "local_trials": campaign.perturbations_per_final,
                    "best_trials": campaign.robustness_trials,
                    "required_passes": campaign.robustness_required_passes,
                },
                extra_artifacts=robustness_artifacts,
            )
    result_catalogs = copy.deepcopy(dict(catalog_report["catalogs"]))
    if isinstance(robustness, Mapping) and robustness.get("robust_success") is True:
        robust_catalog_stage = f"robust_catalog_target_{target_success_count}"
        robust_catalog_report_path = (
            workspace
            / "catalogs"
            / f"target_{target_success_count}"
            / "robust_report.json"
        )
        robust_catalog_report = _load_committed_json(
            workspace, robust_catalog_stage, robust_catalog_report_path
        )
        if robust_catalog_report is None:
            nominal_catalog = workspace / result_catalogs["manipulation"]
            robust_catalog = _publish_robust_alias_catalog(
                nominal_catalog,
                nominal_catalog.parent.parent / "robust_manipulation",
                robust_candidate_id=str(robustness["best_50_candidate_id"]),
            )
            robust_artifacts = authenticated_catalog_artifact_paths(robust_catalog)
            robust_catalog_report = _commit_json_stage(
                workspace,
                robust_catalog_stage,
                robust_catalog_report_path,
                {
                    "complete": True,
                    "catalog": str(robust_catalog.relative_to(workspace)),
                    "best_robust_alias": "best_robust",
                    "robust_candidate_id": str(
                        robustness["best_50_candidate_id"]
                    ),
                },
                stage_input={
                    "nominal_catalog_sha256": file_sha256(nominal_catalog),
                    "robustness_report_sha256": file_sha256(robustness_path),
                },
                extra_artifacts=robust_artifacts,
            )
        result_catalogs["robust"] = robust_catalog_report["catalog"]
    result = {
        "contact_preserving_campaign_result_schema_version": CAMPAIGN_RESULT_SCHEMA_VERSION,
        "complete": True,
        "experiment_id": definition.experiment_id,
        "resume": resume,
        "target_success_count": target_success_count,
        "authenticated_source_grasp_count": len(bundle.sources),
        "grasp_rescue_declared_candidate_count": int(
            rescue_report["declared_candidate_count"]
        ),
        "grasp_rescue_executed_candidate_count": int(
            rescue_report["executed_candidate_count"]
        ),
        "grasp_rescue_success_count": int(rescue_report["grasp_success_count"]),
        "grasp_rescue_promoted_pair_count": len(rescue_sources),
        "base_feedback_candidate_count": len(base_candidate_records),
        "joint_refinement_candidate_count": len(refined_records),
        "joint_refinement_skipped": bool(refinement_report.get("skipped", False)),
        "grasp_success_count": grasp_count,
        "full_success_count": full_count,
        "target_reached": full_count >= target_success_count,
        "fixed_mass_geometry_ablation": True,
        "catalogs": result_catalogs,
        "robustness": copy.deepcopy(robustness),
        "stop_reason": (
            "target_full_success_count_reached"
            if full_count >= target_success_count
            else "declared_contact_first_search_exhausted"
        ),
    }
    final_manifest = build_contact_preserving_planned_lift_manifest(
        config_file, seed=int(seed)
    )
    if canonical_sha256(final_manifest) != canonical_sha256(manifest):
        raise RuntimeError(
            "schema-v14 implementation or authenticated input changed during campaign"
        )
    validate_stage_ledger(workspace)
    result_path = workspace / f"campaign_result_target_{target_success_count}.json"
    write_json(result_path, result)
    return result


__all__ = [
    "AUXILIARY_SOURCE_CANDIDATE_ID",
    "AuthenticatedV13GraspSource",
    "AuthenticatedV13RescueAnchors",
    "AuthenticatedV13SourceBundle",
    "AuthenticatedV13StaticAnchor",
    "CampaignBackend",
    "DEFAULT_BACKEND",
    "GraspRescueJob",
    "PRIMARY_SOURCE_CANDIDATE_ID",
    "authenticate_v13_grasp_rescue_anchors",
    "authenticate_v13_grasp_sources",
    "build_grasp_rescue_jobs",
    "build_contact_preserving_planned_lift_manifest",
    "build_v14_source_pairs",
    "materialize_v14_source_pair_config",
    "publish_viewer_catalogs",
    "run_contact_preserving_planned_lift_campaign",
]
