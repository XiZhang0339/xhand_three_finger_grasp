"""Resumable schema-v12 contact-point-targeted grasp campaign.

The campaign has a deliberately different terminal condition from the v9--v11
smooth-lift campaigns: a measured, full-reset *grasp* is publishable evidence.
The 17 checkpoint branches only rank the grasp's local manipulability and can
never manufacture a success.  A contact-point plan is content-addressed and
committed before any dynamic trial, so a resumed campaign cannot silently aim
at another set of surface points.

Numerically expensive stages are exposed through :class:`CampaignBackend`.
The default backend uses the real MuJoCo/DLS/dynamic implementations; focused
tests can inject deterministic stage doubles without weakening production
authentication or ledger semantics.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
    authenticated_catalog_artifact_paths,
    build_campaign_manifest,
    bind_candidate_result_semantic_sha256,
    commit_campaign_stage,
    export_actual_contact_grasp_pose_catalog,
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from ..artifacts import file_sha256, write_json
from ..config import ACTIVE_ACTUATORS, load_config, validate_config
from ..experiment import resolve_experiment
from ..grasp_pose import canonical_sha256, controller_id, grasp_pose_id
from .actual_contact_grasp_pose_dynamic import (
    dynamic_grasp_rank_evidence,
    grasp_stage_succeeded,
    rank_dynamic_grasp_results,
    run_actual_contact_dynamic_grasp_candidates,
)
from .actual_contact_grasp_pose_measured import (
    measured_grasp_pose_succeeded,
    run_or_resume_measured_grasp_finalization,
)
from .actual_qpos_sources import ActualQposSource, load_actual_qpos_sources
from .contact_point_manipulability import (
    ContactPointManipulabilityBudget,
    run_contact_point_manipulability_prescreen,
)
from .contact_point_grasp_robustness import (
    run_v12_grasp_perturbation_audit,
)
from .contact_point_targeted_search import (
    ContactPointGenerationResult,
    ContactPointPlan,
    ContactPointSearchPolicy,
    PointTargetDLSResult,
    PointTargetVariables,
    assert_frozen_contact_point_plan,
    bind_frozen_contact_point_plan,
    default_point_plan_reachability_rank,
    generate_contact_point_plans,
    point_target_static_record,
    retain_point_target_static_candidates,
    solve_point_target_dls,
)


CAMPAIGN_RESULT_SCHEMA_VERSION = 1
SOURCE_BUNDLE_SCHEMA_VERSION = 1
POINT_CATALOG_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class CampaignBackend:
    """Injectable numerical/artifact boundary for focused orchestration tests."""

    source_loader: Callable[[Mapping[str, Any]], tuple[ActualQposSource, ...]] = (
        load_actual_qpos_sources
    )
    point_generator: Callable[..., ContactPointGenerationResult] = (
        generate_contact_point_plans
    )
    dls_solver: Callable[..., PointTargetDLSResult] = solve_point_target_dls
    dynamic_runner: Callable[..., tuple[dict[str, Any], ...]] = (
        run_actual_contact_dynamic_grasp_candidates
    )
    measured_runner: Callable[..., tuple[dict[str, Any], ...]] = (
        run_or_resume_measured_grasp_finalization
    )
    manipulability_runner: Callable[..., tuple[dict[str, Any], tuple[dict[str, Any], ...]]] = (
        run_contact_point_manipulability_prescreen
    )
    catalog_exporter: Callable[..., dict[str, Any]] = (
        export_actual_contact_grasp_pose_catalog
    )
    robustness_runner: Callable[..., dict[str, Any]] | None = None


DEFAULT_BACKEND = CampaignBackend()


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _safe_stage_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise RuntimeError(f"committed campaign stage is malformed: {path}")
    return payload


def _existing_stage(workspace: Path, stage: str, report: Path) -> dict[str, Any] | None:
    ledger = validate_stage_ledger(workspace)
    if stage not in ledger["stages"]:
        return None
    if not report.is_file():
        raise RuntimeError(f"committed stage {stage} lost {report}")
    return _safe_stage_payload(report)


def _commit_report(
    workspace: Path,
    stage: str,
    report_path: Path,
    payload: Mapping[str, Any],
    *,
    stage_input: Mapping[str, Any],
    extra_artifacts: Sequence[Path] = (),
) -> dict[str, Any]:
    write_json(report_path, payload)
    commit_campaign_stage(
        workspace,
        stage,
        stage_input=stage_input,
        artifacts=(report_path, *extra_artifacts),
        summary={
            key: copy.deepcopy(payload[key])
            for key in (
                "count",
                "source_count",
                "retained_count",
                "reachable_plan_count",
                "static_pass_count",
                "grasp_success_count",
                "target_reached",
            )
            if key in payload
        },
    )
    return copy.deepcopy(dict(payload))


def _candidate_identifier(*parts: Any, namespace: int) -> int:
    digest = hashlib.sha256(
        json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    # Keep IDs comfortably below signed int64 after the dynamic runner's x16
    # controller suffix while retaining deterministic collision resistance.
    return namespace * 10**12 + int.from_bytes(digest[:6], "big") % 10**12


def _cube_center_z(config: Mapping[str, Any]) -> float:
    from ..scene import cube_vertical_half_extent_m, rpy_degrees_to_rotation_matrix

    cube = config["cube"]
    return float(config["scene"]["support_top_z_m"]) + cube_vertical_half_extent_m(
        float(cube["edge_m"]),
        rpy_degrees_to_rotation_matrix(cube.get("rpy_deg", (0.0, 0.0, 0.0))),
    ) + float(cube.get("z_offset_m", 0.0))


def materialize_v12_source_seed(
    template: Mapping[str, Any],
    source: ActualQposSource,
    plan: ContactPointPlan,
) -> dict[str, Any]:
    """Transplant certified hand evidence without importing its old cube/schema.

    The 89 -> 90 mm bridge raises the hand root by the same change in cube
    centre height.  Thus the seed hand/cube relative pose is preserved while
    the v12 cube world pose, experiment identity, search contract and freejoint
    semantics remain those of the template.
    """

    result = copy.deepcopy(dict(template))
    source_config = source.config
    source_hand = copy.deepcopy(dict(source_config["hand_pose"]))
    source_center_z = _cube_center_z(source_config)
    target_center_z = _cube_center_z(result)
    translation = list(map(float, source_hand["translation_m"]))
    translation[2] += target_center_z - source_center_z
    result["hand_pose"] = {
        "translation_m": translation,
        "rpy_deg": list(map(float, source_hand["rpy_deg"])),
    }
    actual = {
        name: float(source.actual_joint_qpos_rad[index])
        for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    result["grasp_pose"]["nominal_joint_qpos_rad"] = copy.deepcopy(actual)
    source_control = source_config.get("control", {})
    for field in ("precontact_targets_rad", "contact_preload_targets_rad", "close_profile"):
        if isinstance(source_control.get(field), Mapping):
            result["control"][field] = copy.deepcopy(dict(source_control[field]))
    result["control"]["contact_preload_targets_rad"] = copy.deepcopy(actual)
    result["control"]["manipulation_delta_rad"] = {
        name: 0.0 for name in ACTIVE_ACTUATORS
    }
    result.pop("contact_point_plan", None)
    result = bind_frozen_contact_point_plan(result, plan)
    metadata = copy.deepcopy(dict(result.get("candidate_metadata", {})))
    metadata["contact_point_target_search"] = {
        "anchor_hand_pose": copy.deepcopy(result["hand_pose"]),
        "root_delta_cube_m": [0.0, 0.0, 0.0],
        "wrist_local_rotvec_deg": [0.0, 0.0, 0.0],
        "signed_orbit_deg": 0.0,
        "point_plan_id": plan.point_plan_id,
        "source_pose_id": source.pose_id,
        "source_candidate_index": source.source_index,
        "source_cube_center_z_m": source_center_z,
        "target_cube_center_z_m": target_center_z,
        "relative_pose_preserved_across_size_bridge": True,
        "cube_pose_sampled": False,
        "hand_root_fixed_during_simulation": True,
    }
    result["candidate_metadata"] = metadata
    validate_config(result)
    return result


def _dls_process_job(job: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = solve_point_target_dls(
            job["config"],
            signed_orbit_deg=float(job["signed_orbit_deg"]),
            initial_variables=PointTargetVariables.from_array(job["initial_variables"]),
        )
        return point_target_static_record(
            int(job["candidate_id"]), result, source_id=job.get("source_id")
        )
    except (ValueError, RuntimeError) as error:
        return _failed_dls_record(job, error)


def _failed_dls_record(job: Mapping[str, Any], error: BaseException) -> dict[str, Any]:
    config = copy.deepcopy(dict(job["config"]))
    return {
        "candidate_id": int(job["candidate_id"]),
        "source_id": str(job.get("source_id", "")),
        "config": config,
        "candidate_sha256": canonical_sha256(config),
        "grasp_pose_id": grasp_pose_id(config),
        "controller_id": controller_id(config),
        "point_plan_id": str(job["point_plan_id"]),
        "signed_orbit_deg": float(job["signed_orbit_deg"]),
        "clockwise_orbit_deg": float(job["signed_orbit_deg"]),
        "static_pass": False,
        "static_metrics": {
            "point_target": {
                "point_plan_id": str(job["point_plan_id"]),
                "point_distance_m": [1.0, 1.0, 1.0],
                "static_acceptance": {
                    "passed": False,
                    "reasons": ["dls_job_rejected"],
                },
            },
            "dls_job_error": f"{type(error).__name__}: {error}",
            "minimum_active_nondistal_gap_m": -1.0,
            "maximum_penetration_m": 1.0,
        },
        "static_rank": (True, 99, 99, 1.0, 3.0, 1.0, 3.0, 1.0, 180.0, int(job["candidate_id"])),
    }


def _run_dls_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    backend: CampaignBackend,
) -> tuple[dict[str, Any], ...]:
    if not jobs:
        return ()
    if backend.dls_solver is not solve_point_target_dls:
        records = []
        for job in jobs:
            try:
                result = backend.dls_solver(
                    job["config"],
                    signed_orbit_deg=float(job["signed_orbit_deg"]),
                    initial_variables=PointTargetVariables.from_array(job["initial_variables"]),
                )
                records.append(
                    point_target_static_record(
                        int(job["candidate_id"]), result, source_id=job.get("source_id")
                    )
                )
            except (ValueError, RuntimeError) as error:
                records.append(_failed_dls_record(job, error))
    elif workers == 1:
        records = [_dls_process_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            records = list(executor.map(_dls_process_job, jobs, chunksize=1))
    identifiers = [int(value["candidate_id"]) for value in records]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("point-target DLS produced duplicate candidate IDs")
    return tuple(sorted((copy.deepcopy(value) for value in records), key=lambda value: int(value["candidate_id"])))


def _line_of_action_moment(plan: ContactPointPlan) -> float:
    torque = np.zeros(3, dtype=np.float64)
    # Unit opposed squeeze: thumb balances index+middle.
    for finger, force_x in (("thumb", 2.0), ("index", -1.0), ("mid", -1.0)):
        point = plan.points[finger].local_xyz_m(plan.edge_m)
        torque += np.cross(point, np.asarray((force_x, 0.0, 0.0)))
    return float(np.linalg.norm(torque))


def _static_safety_margin(record: Mapping[str, Any], policy: ContactPointSearchPolicy) -> float:
    metrics = record.get("static_metrics", {})
    point = metrics.get("point_target", {}) if isinstance(metrics, Mapping) else {}
    distances = tuple(float(value) for value in point.get("point_distance_m", (math.inf,)))
    nondistal = float(metrics.get("minimum_active_nondistal_gap_m", -math.inf))
    forbidden = float(
        metrics.get("nominal_minimum_forbidden_hand_gap_m", -math.inf)
    )
    precontact = float(metrics.get("precontact_minimum_hand_gap_m", -math.inf))
    penetration = float(
        metrics.get("nominal_maximum_all_distal_penetration_m", math.inf)
    )
    return min(
        policy.static_target_radius_m - max(distances, default=math.inf),
        nondistal,
        forbidden,
        policy.maximum_penetration_m - penetration,
        precontact,
    )


def _point_plan_reachability_records(
    generation: ContactPointGenerationResult,
    dls_records: Sequence[Mapping[str, Any]],
    policy: ContactPointSearchPolicy,
) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in dls_records:
        grouped.setdefault(str(record["point_plan_id"]), []).append(record)
    result = []
    for generated in generation.retained:
        records = grouped.get(generated.plan.point_plan_id, [])
        passes = [value for value in records if bool(value.get("static_pass", False))]
        result.append(
            {
                "point_plan_id": generated.plan.point_plan_id,
                "contact_point_plan": generated.plan.as_config(),
                "sample_index": generated.sample_index,
                "reachable": bool(passes),
                "reachable_seed_count": len(passes),
                "evaluated_seed_count": len(records),
                "minimum_safety_margin_m": max(
                    (_static_safety_margin(value, policy) for value in records),
                    default=-math.inf,
                ),
                "height_spread_m": generated.metrics.height_spread_m,
                "line_of_action_moment_nm": _line_of_action_moment(generated.plan),
                "minimum_edge_margin_m": generated.metrics.minimum_edge_margin_m,
                "geometry_metrics": generated.metrics.as_dict(),
                "candidate_ids": [int(value["candidate_id"]) for value in records],
                "passing_candidate_ids": [int(value["candidate_id"]) for value in passes],
            }
        )
    return tuple(sorted(result, key=default_point_plan_reachability_rank))


def _policy_jobs(
    template: Mapping[str, Any],
    sources: Sequence[ActualQposSource],
    generation: ContactPointGenerationResult,
    policy: ContactPointSearchPolicy,
) -> tuple[dict[str, Any], ...]:
    jobs = []
    for generated in generation.retained:
        for orbit_index, orbit in enumerate(policy.signed_orbit_deg):
            for source in sources:
                config = materialize_v12_source_seed(template, source, generated.plan)
                initial = PointTargetVariables.from_config(config)
                candidate_id = _candidate_identifier(
                    generated.sample_index,
                    orbit_index,
                    source.source_index,
                    namespace=12,
                )
                jobs.append(
                    {
                        "candidate_id": candidate_id,
                        "source_id": source.pose_id,
                        "point_plan_id": generated.plan.point_plan_id,
                        "signed_orbit_deg": float(orbit),
                        "initial_variables": initial.as_array().tolist(),
                        "config": config,
                    }
                )
    if len(jobs) > policy.retain_point_plan_count * len(policy.signed_orbit_deg) * len(sources):
        raise AssertionError("reachability job count exceeds the registered product")
    return tuple(jobs)


def _halton_local(count: int, dimensions: int, seed: int) -> np.ndarray:
    # Prefix-stable randomized LHS is sufficient here; the 120k plan stream is
    # the registered Halton stream owned by contact_point_targeted_search.
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        values[:, dimension] = (rng.permutation(count) + rng.random(count)) / count
    return values


def _static_pose_jobs(
    template: Mapping[str, Any],
    sources: Sequence[ActualQposSource],
    plan: ContactPointPlan,
    policy: ContactPointSearchPolicy,
    *,
    seed: int,
    proposal_count: int,
) -> tuple[dict[str, Any], ...]:
    unit = _halton_local(proposal_count, 14, seed + 1200)
    definition = resolve_experiment(dict(template))
    jobs: list[dict[str, Any]] = []
    for index, row in enumerate(unit):
        source = sources[index % len(sources)]
        orbit = policy.signed_orbit_deg[(index // len(sources)) % len(policy.signed_orbit_deg)]
        config = materialize_v12_source_seed(template, source, plan)
        base = PointTargetVariables.from_config(config).as_array()
        values = base.copy()
        for joint, name in enumerate(ACTIVE_ACTUATORS):
            low, high = definition.search_bounds.actuator_targets_rad[name]
            radius = 0.045 if joint else 0.06
            values[joint] = np.clip(base[joint] + (2.0 * row[joint] - 1.0) * radius, low, high)
        values[0] = np.clip(values[0], *policy.thumb_actual_range_rad)
        values[8:11] = np.asarray(
            [
                low + row[8 + axis] * (high - low)
                for axis, (low, high) in enumerate(policy.root_delta_cube_m.values())
            ]
        )
        rot = np.radians((2.0 * row[11:14] - 1.0) * 4.0)
        norm = float(np.linalg.norm(rot))
        limit = math.radians(policy.max_wrist_local_rotvec_norm_deg)
        if norm > limit:
            rot *= limit / norm
        values[11:14] = rot
        jobs.append(
            {
                "candidate_id": _candidate_identifier(index, source.source_index, orbit, namespace=13),
                "source_id": source.pose_id,
                "point_plan_id": plan.point_plan_id,
                "signed_orbit_deg": float(orbit),
                "initial_variables": values.tolist(),
                "config": config,
            }
        )
    return tuple(jobs)


def _record_artifact_paths(record: Mapping[str, Any], dynamic_root: Path) -> tuple[Path, Path, Path]:
    directory = dynamic_root / Path(str(record["artifact_directory"]))
    return directory / "resolved_config.json", directory / "result.json", directory / "trace.npz"


def _candidate_stage_artifacts(
    records: Sequence[Mapping[str, Any]],
    dynamic_root: Path,
    *,
    require_trace: bool,
) -> tuple[Path, ...]:
    """Bind candidate members into the stage ledger, including retained traces."""

    artifacts: list[Path] = []
    for record in records:
        config_path, result_path, trace_path = _record_artifact_paths(
            record, dynamic_root
        )
        if not config_path.is_file() or not result_path.is_file():
            raise RuntimeError(
                "completed dynamic stage lost a candidate config or result"
            )
        artifacts.extend((config_path, result_path))
        if trace_path.is_file():
            artifacts.append(trace_path)
        elif require_trace or grasp_stage_succeeded(record):
            raise RuntimeError("successful candidate lost its retained trace")
    return tuple(dict.fromkeys(artifacts))


def _catalog_candidates(
    records: Sequence[Mapping[str, Any]], dynamic_root: Path
) -> tuple[dict[str, Any], ...]:
    # Callers pass v12-ranked successes.  Preserve that order: the generic
    # publisher's historical v9 full-lift rank is not authoritative here.
    ordered = list(records)
    successes = [value for value in ordered if measured_grasp_pose_succeeded(value) or grasp_stage_succeeded(value)]
    if not successes:
        # The caller already supplied the dedicated v12 order.  Retain its
        # first trace-backed near miss instead of reintroducing the generic
        # v9 actual-contact rank here.
        successes = ordered[:1]
    result = []
    for discovery, record in enumerate(successes):
        config_path, result_path, trace_path = _record_artifact_paths(record, dynamic_root)
        if not all(path.is_file() for path in (config_path, result_path, trace_path)):
            continue
        result.append(
            {
                "candidate_id": int(record["candidate_id"]),
                "discovery_index": discovery,
                "config_path": config_path,
                "result_path": result_path,
                "trace_path": trace_path,
                "summary": copy.deepcopy(dict(record["summary"])),
            }
        )
    return tuple(result)


def _rewrite_v12_catalog_selection(
    catalog_path: Path, ranked_candidate_ids: Sequence[int]
) -> dict[str, Any]:
    """Make aliases follow the v12 grasp/manipulability rank, not v9 lift rank."""

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    trajectories = catalog.get("trajectories", [])
    previous_best_first = catalog.get("aliases", {}).get("best_first")
    by_candidate = {
        int(value["candidate_id"]): value
        for value in trajectories
        if value.get("classification") == "success"
    }
    ranked = [by_candidate[value] for value in ranked_candidate_ids if value in by_candidate]
    aliases = {
        key: value
        for key, value in catalog.get("aliases", {}).items()
        if key == "best_attempt"
    }
    for value in trajectories:
        value["aliases"] = [
            alias for alias in value.get("aliases", []) if alias == "best_attempt"
        ]
    for index, value in enumerate(ranked, start=1):
        alias = f"grasp_pose_{index}"
        value["aliases"].append(alias)
        aliases[alias] = value["trajectory_id"]
        if index == 1:
            value["aliases"].append("best_nominal")
            aliases["best_nominal"] = value["trajectory_id"]
    # ``best_first`` remains the first discovered measured success, while
    # best_nominal is the v12 contact/manipulability rank winner.
    successful = [value for value in trajectories if value.get("classification") == "success"]
    if successful:
        first = next(
            (
                value
                for value in successful
                if value["trajectory_id"] == previous_best_first
            ),
            successful[0],
        )
        first["aliases"].append("best_first")
        aliases["best_first"] = first["trajectory_id"]
    catalog["aliases"] = aliases
    catalog["selection"]["ranking_policy"] = (
        "schema_v12_grasp_point_error_manipulability_perturbation_rank"
    )
    catalog["selection"]["ranked_candidate_ids"] = list(ranked_candidate_ids)
    write_json(catalog_path, catalog)
    return catalog


def _authenticate_resumable_grasp_catalog(
    catalog_path: Path,
    *,
    experiment_id: str,
    point_plan_id: str,
    target_success_count: int,
) -> tuple[Path, ...]:
    if not catalog_path.is_file():
        raise RuntimeError(
            "grasp catalog directory exists without a complete catalog.json"
        )
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != experiment_id:
        raise RuntimeError("resumed grasp catalog changed experiment")
    requested = payload.get("requested_success_count")
    if requested is not None and int(requested) != target_success_count:
        raise RuntimeError("resumed grasp catalog changed target success count")
    root = catalog_path.parent.resolve()
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise RuntimeError("resumed grasp catalog has no trajectory evidence")
    aliases = payload.get("aliases")
    if not isinstance(aliases, Mapping):
        raise RuntimeError("resumed grasp catalog has no alias mapping")
    success_entries = [
        entry
        for entry in trajectories
        if isinstance(entry, Mapping) and entry.get("classification") == "success"
    ]
    diagnostic_only = not success_entries
    if diagnostic_only and ({"best_first", "best_nominal"} & set(aliases)):
        raise RuntimeError(
            "diagnostic v12 grasp catalog may not expose a success alias"
        )

    for entry in trajectories:
        if not isinstance(entry, Mapping):
            raise RuntimeError("resumed grasp catalog trajectory is malformed")
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise RuntimeError("resumed grasp catalog trajectory has no artifacts")
        hashes = artifacts.get("sha256")
        if not isinstance(hashes, Mapping):
            raise RuntimeError("resumed grasp catalog trajectory has no hashes")

        def member_path(name: str) -> Path:
            relative = artifacts.get(name)
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise RuntimeError(
                    f"resumed grasp catalog has no safe {name} artifact"
                )
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise RuntimeError(f"resumed grasp catalog lost its {name} artifact")
            expected = hashes.get(name)
            if not isinstance(expected, str) or file_sha256(path) != expected:
                raise RuntimeError(
                    f"resumed grasp catalog {name} SHA-256 mismatch"
                )
            return path

        config_path = member_path("resolved_config")
        result_path = member_path("result")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert_frozen_contact_point_plan(config, point_plan_id)
        if config.get("experiment_id") != experiment_id:
            raise RuntimeError("resumed grasp catalog member changed experiment")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, Mapping):
            raise RuntimeError("resumed grasp catalog result is malformed")
        if int(result.get("candidate_result_schema_version", 0)) == 1:
            authenticate_candidate_result_semantic_sha256(
                result, source=result_path
            )
        summary = result.get("summary")
        stage_status = (
            summary.get("stage_status", {})
            if isinstance(summary, Mapping)
            else {}
        )
        persisted_grasp = bool(
            isinstance(stage_status, Mapping)
            and stage_status.get("grasp_success", False)
        )
        persisted_measured = bool(
            result.get("measured_grasp_pose_success", False)
        )
        if entry.get("classification") == "success":
            if int(result.get("candidate_result_schema_version", 0)) != 1:
                raise RuntimeError(
                    "v12 success catalog result lacks authenticated semantics"
                )
            if result.get("stage") != "measured_grasp_pose_finalization":
                raise RuntimeError(
                    "v12 success catalog contains an unfinalized grasp result"
                )
            if not (
                persisted_measured
                and bool(result.get("grasp_success", False))
                and persisted_grasp
            ):
                raise RuntimeError(
                    "v12 success catalog result failed measured grasp semantics"
                )
        elif diagnostic_only:
            if persisted_measured or bool(result.get("grasp_success", False)) or persisted_grasp:
                raise RuntimeError(
                    "diagnostic v12 grasp catalog retained success semantics"
                )
    # The shared authenticator expands trace/video members and checks all
    # declared hashes.  The v12 checks above additionally bind publication
    # class to measured-finalization semantics.
    return authenticated_catalog_artifact_paths(catalog_path)


def _publish_v12_diagnostic_catalog(
    candidate: Mapping[str, Any],
    output_dir: Path,
    *,
    experiment_id: str,
    point_plan_id: str,
    target_success_count: int,
) -> dict[str, Any]:
    """Publish a Viewer near miss without inheriting raw dynamic success."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    config_path = Path(candidate["config_path"]).resolve()
    result_path = Path(candidate["result_path"]).resolve()
    trace_path = Path(candidate["trace_path"]).resolve()
    if not all(path.is_file() for path in (config_path, result_path, trace_path)):
        raise RuntimeError("diagnostic catalog source lost a required artifact")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if int(result.get("candidate_result_schema_version", 0)) == 1:
        authenticate_candidate_result_semantic_sha256(result, source=result_path)
    assert_frozen_contact_point_plan(config, point_plan_id)
    if config.get("experiment_id") != experiment_id:
        raise RuntimeError("diagnostic catalog source changed experiment")
    identifier = int(candidate["candidate_id"])
    trajectory_id = f"grasp_pose_diagnostic_{identifier}"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}.staging."
    ) as staging_name:
        staging = Path(staging_name)
        member = staging / trajectory_id
        member.mkdir()
        destinations = {
            "resolved_config": member / "resolved_config.json",
            "result": member / "result.json",
            "trace": member / "trace.npz",
        }
        shutil.copy2(config_path, destinations["resolved_config"])
        shutil.copy2(trace_path, destinations["trace"])
        raw_dynamic_grasp = bool(
            result.get("summary", {})
            .get("stage_status", {})
            .get("grasp_success", False)
        )
        diagnostic_result = copy.deepcopy(result)
        raw_summary = copy.deepcopy(dict(result.get("summary", {})))
        diagnostic_summary = copy.deepcopy(raw_summary)
        stage_status = copy.deepcopy(
            dict(diagnostic_summary.get("stage_status", {}))
        )
        stage_status.update(
            {
                "grasp_success": False,
                "manipulation_success": False,
                "full_success": False,
            }
        )
        diagnostic_summary["stage_status"] = stage_status
        diagnostic_summary["passed"] = False
        failures = list(diagnostic_summary.get("failed_checks", []))
        if "measured_grasp_pose_finalization_required" not in failures:
            failures.append("measured_grasp_pose_finalization_required")
        diagnostic_summary["failed_checks"] = failures
        diagnostic_result.update(
            {
                "summary": diagnostic_summary,
                "grasp_success": False,
                "measured_grasp_pose_success": False,
                "classification": "diagnostic_unfinalized_actual_grasp_pose",
                "diagnostic_source_evidence": {
                    "source_result_sha256": file_sha256(result_path),
                    "source_result_semantic_sha256": result.get(
                        "result_semantic_sha256"
                    ),
                    "source_summary_sha256": canonical_sha256(raw_summary),
                    "raw_dynamic_grasp_success": raw_dynamic_grasp,
                    "measured_finalization_required_for_publication": True,
                },
            }
        )
        if int(diagnostic_result.get("candidate_result_schema_version", 0)) == 1:
            diagnostic_result = bind_candidate_result_semantic_sha256(
                diagnostic_result
            )
        write_json(destinations["result"], diagnostic_result)
        catalog = {
            "actual_contact_grasp_pose_catalog_schema_version": 1,
            "trajectory_catalog_schema_version": 1,
            "experiment_id": experiment_id,
            "catalog_kind": "grasp_pose",
            "complete": True,
            "requested_success_count": target_success_count,
            "success_count": 0,
            "eligible_success_count": 0,
            "target_reached": False,
            "selection": {
                "ranking_policy": "schema_v12_measured_finalization_required",
                "ranked_candidate_ids": [identifier],
                "measured_grasp_success_required": True,
            },
            "aliases": {"best_attempt": trajectory_id},
            "trajectories": [
                {
                    "trajectory_id": trajectory_id,
                    "label": trajectory_id,
                    "aliases": ["best_attempt"],
                    "classification": "diagnostic",
                    "candidate_id": identifier,
                    "grasp_success": False,
                    "measured_grasp_pose_success": False,
                    "raw_dynamic_grasp_success": raw_dynamic_grasp,
                    "full_success": False,
                    "point_plan_id": point_plan_id,
                    "failed_checks": copy.deepcopy(
                        diagnostic_summary.get("failed_checks", [])
                    ),
                    "artifacts": {
                        "resolved_config": f"{trajectory_id}/resolved_config.json",
                        "result": f"{trajectory_id}/result.json",
                        "trace": f"{trajectory_id}/trace.npz",
                        "video": None,
                        "sha256": {
                            name: file_sha256(path)
                            for name, path in destinations.items()
                        },
                    },
                }
            ],
        }
        write_json(staging / "catalog.json", catalog)
        staging.rename(output_dir)
    return catalog


def _rank_finite(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _rank_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _v12_force_imbalance(record: Mapping[str, Any]) -> float:
    """Return opposed-side force imbalance; missing evidence ranks last."""

    metrics = _rank_mapping(
        _rank_mapping(record.get("summary")).get("metrics")
    )
    for field in ("verify_peak_target_face_force_n", "peak_distal_contact_force_n"):
        values = _rank_mapping(metrics.get(field))
        thumb = _rank_finite(values.get("thumb"), math.nan)
        index = _rank_finite(values.get("index"), math.nan)
        middle = _rank_finite(values.get("mid", values.get("middle")), math.nan)
        if all(math.isfinite(value) and value >= 0.0 for value in (thumb, index, middle)):
            opposed = index + middle
            opposed_scale = max(thumb, opposed)
            opposed_imbalance = (
                abs(thumb - opposed) / opposed_scale
                if opposed_scale > 0.0
                else math.inf
            )
            pair_scale = max(index, middle)
            pair_imbalance = (
                abs(index - middle) / pair_scale
                if pair_scale > 0.0
                else math.inf
            )
            # Both conditions matter: the thumb must balance the opposed side,
            # and index/middle must share that side instead of allowing one
            # finger (historically the middle) to dominate the other.
            return max(opposed_imbalance, pair_imbalance)
    return math.inf


def v12_grasp_rank_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the exact fail-closed schema-v12 final-ranking evidence.

    This deliberately does not call ``dynamic_grasp_rank``: its thumb-to-1.50
    preference and peak-force key are v9 policy and are not part of v12.
    """

    dynamic = dynamic_grasp_rank_evidence(record)
    measured_pass = bool(
        measured_grasp_pose_succeeded(record) and grasp_stage_succeeded(record)
    )
    try:
        local_passes = int(record.get("local_perturbation_pass_count", 0))
    except (TypeError, ValueError):
        local_passes = 0
    local_passes = max(local_passes, 0)

    metrics = _rank_mapping(
        _rank_mapping(record.get("summary")).get("metrics")
    )
    point_metrics = _rank_mapping(metrics.get("contact_point_targeting"))
    acquisition = _rank_mapping(point_metrics.get("acquisition_window"))
    per_finger = _rank_mapping(acquisition.get("per_finger"))
    point_errors = [
        _rank_finite(
            _rank_mapping(per_finger.get(finger)).get("tangent_error_max_m"),
            math.inf,
        )
        for finger in ("thumb", "index", "mid")
    ]
    point_error = max(point_errors, default=math.inf)
    config = _rank_mapping(record.get("config"))
    configured_plan = _rank_mapping(config.get("contact_point_plan"))
    target_radius = _rank_finite(
        point_metrics.get(
            "target_radius_m", configured_plan.get("target_radius_m")
        ),
        math.nan,
    )
    point_margin = (
        (target_radius - point_error) / target_radius
        if target_radius > 0.0 and math.isfinite(point_error)
        else -math.inf
    )
    margins = [
        _rank_finite(dynamic.get(name), -math.inf)
        for name in (
            "stability_min_normalized_margin",
            "pose_min_normalized_margin",
            "closure_normalized_margin",
        )
    ]
    margins.append(point_margin)
    minimum_margin = min(margins, default=-math.inf)

    prescreen = _rank_mapping(record.get("manipulability_prescreen"))
    predicted_residual = _rank_finite(
        prescreen.get("weighted_residual_norm"), math.inf
    )

    try:
        line_moment = _line_of_action_moment(
            ContactPointPlan.from_config(config["contact_point_plan"])
        )
    except (KeyError, TypeError, ValueError):
        line_moment = math.inf

    closure_angle = _rank_finite(
        dynamic.get("closure_max_p95_angle_deg"), math.inf
    )
    pose_margin = _rank_finite(
        dynamic.get("pose_min_normalized_margin"), -math.inf
    )
    saturation = _rank_finite(
        dynamic.get("actuator_saturation_fraction"), math.inf
    )
    candidate_id = record.get("candidate_id", 2**63 - 1)
    try:
        candidate_id = int(candidate_id)
    except (TypeError, ValueError):
        candidate_id = 2**63 - 1

    def optional(value: float) -> float | None:
        return value if math.isfinite(value) else None

    return {
        "measured_grasp_hard_pass": measured_pass,
        "local_perturbation_pass_count": local_passes,
        "minimum_acceptance_margin": optional(minimum_margin),
        "maximum_target_point_error_m": optional(point_error),
        "manipulation_predicted_weighted_residual": optional(predicted_residual),
        "line_of_action_moment_nm": optional(line_moment),
        "opposed_force_imbalance_fraction": optional(
            _v12_force_imbalance(record)
        ),
        "closure_max_p95_angle_deg": optional(closure_angle),
        "pose_preservation_min_normalized_margin": optional(pose_margin),
        "actuator_saturation_fraction": optional(saturation),
        "candidate_id": candidate_id,
    }


def _grasp_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    evidence = v12_grasp_rank_evidence(record)
    return (
        not bool(evidence["measured_grasp_hard_pass"]),
        -int(evidence["local_perturbation_pass_count"]),
        -_rank_finite(evidence["minimum_acceptance_margin"], -math.inf),
        _rank_finite(evidence["maximum_target_point_error_m"], math.inf),
        _rank_finite(
            evidence["manipulation_predicted_weighted_residual"], math.inf
        ),
        _rank_finite(evidence["line_of_action_moment_nm"], math.inf),
        _rank_finite(evidence["opposed_force_imbalance_fraction"], math.inf),
        _rank_finite(evidence["closure_max_p95_angle_deg"], math.inf),
        -_rank_finite(
            evidence["pose_preservation_min_normalized_margin"], -math.inf
        ),
        _rank_finite(evidence["actuator_saturation_fraction"], math.inf),
        int(evidence["candidate_id"]),
    )


def _source_report(template: Mapping[str, Any], sources: Sequence[ActualQposSource]) -> dict[str, Any]:
    expected = tuple(resolve_experiment(dict(template)).contact_point_search.source_pose_candidate_ids)
    observed = []
    for source in sources:
        candidate_id = int(source.config.get("candidate_metadata", {}).get("candidate_id", -1))
        observed.append(candidate_id)
    if tuple(observed) != expected:
        raise RuntimeError(
            "v12 source bundle must contain the two registered candidate IDs in order"
        )
    if not all(source.eligible_as_success_evidence for source in sources):
        raise RuntimeError("v12 contact-point source bundle contains diagnostic-only evidence")
    records = [source.report_record() for source in sources]
    return {
        "complete": True,
        "contact_point_source_bundle_schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "source_count": len(sources),
        "source_candidate_ids": observed,
        "all_sources_authenticated_grasp_success": True,
        "sources": records,
        "source_bundle_sha256": canonical_sha256(records),
    }


def _reachability_report_payload(
    records: Sequence[Mapping[str, Any]],
    ranked_plans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = next((value for value in ranked_plans if value["reachable"]), None)
    return {
        "complete": True,
        "point_plan_reachability_schema_version": 1,
        "evaluated_count": len(records),
        "reachable_candidate_count": sum(bool(value.get("static_pass")) for value in records),
        "reachable_plan_count": sum(bool(value["reachable"]) for value in ranked_plans),
        "selected_point_plan_id": None if selected is None else selected["point_plan_id"],
        "selected_contact_point_plan": None if selected is None else selected["contact_point_plan"],
        "plan_ranking": list(ranked_plans),
        "candidates": list(records),
    }


def run_contact_point_targeted_grasp_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool,
    target_success_count: int,
    workers: int,
    seed: int = DEFAULT_SEED,
    backend: CampaignBackend | None = None,
    evidence_anchor_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Execute/resume the registered schema-v12 campaign.

    ``target_success_count`` counts measured grasp successes, not full
    manipulation successes.  It is intentionally excluded from the immutable
    campaign manifest so a target-one workspace may later publish target-five.
    """

    if target_success_count not in (1, 5):
        raise ValueError("target_success_count must be 1 or 5")
    _positive_int(workers, "workers")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if evidence_anchor_paths:
        raise ValueError(
            "schema-v12 uses its registered two-source manifest; ad-hoc evidence "
            "anchors are not allowed"
        )
    selected_backend = backend or DEFAULT_BACKEND
    resolved_config = Path(config_path).expanduser().resolve()
    template = load_config(resolved_config)
    definition = resolve_experiment(template)
    search = definition.contact_point_search
    if search is None or int(template.get("schema_version", 0)) != 12:
        raise ValueError("contact-point-targeted campaign requires schema-v12 config")
    if seed != search.seed:
        raise ValueError("seed must match registered contact_point_search.seed")
    policy = ContactPointSearchPolicy.from_config(template)
    # Authenticate every source config/result/trace and its stable-window
    # evidence before creating a new artifact tree.  A corrupt v11 anchor may
    # therefore never leave a half-created v12 campaign behind.
    sources = selected_backend.source_loader(template)
    authenticated_source_payload = _source_report(template, sources)
    manifest = build_campaign_manifest(resolved_config, seed=seed)
    workspace = initialize_or_resume_campaign(output_dir, manifest, resume=resume)
    input_sha = str(manifest["campaign_input_sha256"])

    source_path = workspace / "sources" / "source_bundle.json"
    source_payload = _existing_stage(workspace, "source_bundle", source_path)
    if source_payload is None:
        source_payload = authenticated_source_payload
        _commit_report(
            workspace, "source_bundle", source_path, source_payload,
            stage_input={"campaign_input_sha256": input_sha},
        )
    elif source_payload.get("source_bundle_sha256") != authenticated_source_payload.get(
        "source_bundle_sha256"
    ):
        raise RuntimeError("authenticated source bundle differs from committed evidence")

    point_generation_path = workspace / "contact_points" / "generation_report.json"
    point_generation_payload = _existing_stage(workspace, "point_generation", point_generation_path)
    generation = selected_backend.point_generator(policy)
    if point_generation_payload is None:
        point_generation_payload = {
            "complete": True,
            "contact_point_generation_schema_version": 1,
            **generation.as_report(),
        }
        _commit_report(
            workspace, "point_generation", point_generation_path, point_generation_payload,
            stage_input={"campaign_input_sha256": input_sha, "source_bundle_sha256": source_payload["source_bundle_sha256"]},
        )
    elif canonical_sha256(point_generation_payload["retained"]) != canonical_sha256(generation.as_report()["retained"]):
        raise RuntimeError("resumed point-plan prefix differs from committed generation")

    reach_path = workspace / "contact_points" / "reachability_report.json"
    reach_payload = _existing_stage(workspace, "point_reachability", reach_path)
    if reach_payload is None:
        reach_jobs = _policy_jobs(template, sources, generation, policy)
        if len(reach_jobs) != search.max_reachability_screen_count:
            raise RuntimeError("registered reachability budget was not exhausted exactly")
        reach_records = _run_dls_jobs(reach_jobs, workers=workers, backend=selected_backend)
        ranked_plans = _point_plan_reachability_records(generation, reach_records, policy)
        reach_payload = _reachability_report_payload(reach_records, ranked_plans)
        _commit_report(
            workspace, "point_reachability", reach_path, reach_payload,
            stage_input={"campaign_input_sha256": input_sha, "retained_plan_sha256": canonical_sha256(point_generation_payload["retained"]), "reachability_budget": search.max_reachability_screen_count},
        )

    selected_plan_raw = reach_payload.get("selected_contact_point_plan")
    if selected_plan_raw is None:
        diagnostic_catalog_path = workspace / "contact_points" / "catalog.json"
        diagnostic_catalog = _existing_stage(
            workspace, "contact_point_catalog", diagnostic_catalog_path
        )
        if diagnostic_catalog is None:
            diagnostic_catalog = {
                "complete": True,
                "contact_point_catalog_schema_version": POINT_CATALOG_SCHEMA_VERSION,
                "experiment_id": definition.experiment_id,
                "cube_edge_m": float(template["cube"]["edge_m"]),
                "aliases": {},
                "selected_point_plan_id": None,
                "selected_contact_point_plan": None,
                "selection_policy": (
                    "reachable_seed_count_safety_equal_height_line_moment_edge_margin_id"
                ),
                "stop_reason": "no_reachable_contact_point_plan",
                "plans": copy.deepcopy(reach_payload["plan_ranking"]),
                "reachability_candidates": copy.deepcopy(
                    reach_payload.get("candidates", ())
                ),
            }
            _commit_report(
                workspace,
                "contact_point_catalog",
                diagnostic_catalog_path,
                diagnostic_catalog,
                stage_input={
                    "campaign_input_sha256": input_sha,
                    "reachability_report_sha256": file_sha256(reach_path),
                    "selected_point_plan_id": None,
                },
            )
        result = {
            "contact_point_targeted_campaign_result_schema_version": CAMPAIGN_RESULT_SCHEMA_VERSION,
            "complete": True,
            "experiment_id": definition.experiment_id,
            "target_success_count": target_success_count,
            "target_reached": False,
            "grasp_success_count": 0,
            "full_success_count": 0,
            "stop_reason": "no_reachable_contact_point_plan",
            "contact_point_plan_frozen": False,
            "manipulability_prescreen_is_success_evidence": False,
            "catalogs": {
                "contact_point": str(
                    diagnostic_catalog_path.relative_to(workspace)
                )
            },
        }
        result_path = workspace / f"campaign_result_target_{target_success_count}.json"
        _commit_report(
            workspace, f"campaign_result_{target_success_count}", result_path, result,
            stage_input={"campaign_input_sha256": input_sha, "target_success_count": target_success_count},
        )
        return result

    plan = ContactPointPlan.from_config(selected_plan_raw)
    frozen_path = workspace / "contact_points" / "frozen_plan.json"
    frozen_payload = _existing_stage(workspace, "frozen_contact_point_plan", frozen_path)
    if frozen_payload is None:
        frozen_payload = {
            "complete": True,
            "point_plan_id": plan.point_plan_id,
            "contact_point_plan": plan.as_config(),
            "selected_before_dynamic_search": True,
            "immutable_for_remaining_campaign": True,
            "selection_rank": next(
                index for index, value in enumerate(reach_payload["plan_ranking"]) if value["point_plan_id"] == plan.point_plan_id
            ),
        }
        _commit_report(
            workspace, "frozen_contact_point_plan", frozen_path, frozen_payload,
            stage_input={"campaign_input_sha256": input_sha, "reachability_report_sha256": file_sha256(reach_path)},
        )
    elif frozen_payload.get("point_plan_id") != plan.point_plan_id:
        raise RuntimeError("resumed campaign attempted to switch the frozen point plan")

    point_catalog_path = workspace / "contact_points" / "catalog.json"
    point_catalog_payload = _existing_stage(
        workspace, "contact_point_catalog", point_catalog_path
    )
    if point_catalog_payload is None:
        point_catalog_payload = {
            "complete": True,
            "contact_point_catalog_schema_version": POINT_CATALOG_SCHEMA_VERSION,
            "experiment_id": definition.experiment_id,
            "cube_edge_m": plan.edge_m,
            "aliases": {"selected_frozen_plan": plan.point_plan_id},
            "selected_point_plan_id": plan.point_plan_id,
            "selected_contact_point_plan": plan.as_config(),
            "selection_policy": (
                "reachable_seed_count_safety_equal_height_line_moment_edge_margin_id"
            ),
            "plans": copy.deepcopy(reach_payload["plan_ranking"]),
        }
        _commit_report(
            workspace,
            "contact_point_catalog",
            point_catalog_path,
            point_catalog_payload,
            stage_input={
                "campaign_input_sha256": input_sha,
                "point_plan_id": plan.point_plan_id,
                "frozen_plan_sha256": file_sha256(frozen_path),
            },
        )

    static_path = workspace / "static" / "point_targeted_report.json"
    static_payload = _existing_stage(workspace, "point_targeted_static", static_path)
    if static_payload is None:
        proposal_count = max(search.retained_static_pose_count * 4, len(policy.signed_orbit_deg) * len(sources) * 16)
        static_jobs = _static_pose_jobs(
            template, sources, plan, policy, seed=seed, proposal_count=proposal_count
        )
        all_static = _run_dls_jobs(static_jobs, workers=workers, backend=selected_backend)
        # The plan was frozen only because at least one authenticated
        # plan/orbit/source DLS seed was reachable.  Carry those exact seeds
        # forward so the subsequent local proposal stream cannot accidentally
        # discard all proven static geometry.
        reachability_seeds = [
            copy.deepcopy(value)
            for value in reach_payload.get("candidates", ())
            if value.get("point_plan_id") == plan.point_plan_id
            and bool(value.get("static_pass", False))
        ]
        accepted_by_id = {
            int(value["candidate_id"]): copy.deepcopy(value)
            for value in (*reachability_seeds, *all_static)
            if bool(value.get("static_pass", False))
        }
        accepted = list(accepted_by_id.values())
        retained = retain_point_target_static_candidates(
            accepted, top_k=search.retained_static_pose_count
        ) if accepted else ()
        static_payload = {
            "complete": True,
            "point_targeted_static_stage_schema_version": 1,
            "point_plan_id": plan.point_plan_id,
            "proposal_count": proposal_count,
            "evaluated_count": len(all_static),
            "reachable_seed_count": len(reachability_seeds),
            "static_pass_count": len(accepted),
            "retained_count": len(retained),
            "retained_candidates": list(retained),
            "best_near_miss": None if not all_static else min(all_static, key=lambda value: tuple(value["static_rank"])),
        }
        _commit_report(
            workspace, "point_targeted_static", static_path, static_payload,
            stage_input={"campaign_input_sha256": input_sha, "point_plan_id": plan.point_plan_id, "proposal_count": proposal_count},
        )

    static_records = tuple(copy.deepcopy(value) for value in static_payload["retained_candidates"])
    dynamic_root = workspace / "dynamic"
    dynamic_path = workspace / "dynamic" / "base_report.json"
    dynamic_payload = _existing_stage(workspace, "point_targeted_dynamic", dynamic_path)
    if dynamic_payload is None:
        dynamic_records = selected_backend.dynamic_runner(
            static_records,
            dynamic_root,
            workers=workers,
            resume=resume,
            seed=seed,
            controller_seed_count=search.controllers_per_pose,
            retain_failure_trace_count=24,
        ) if static_records else ()
        dynamic_payload = {
            "complete": True,
            "point_targeted_dynamic_stage_schema_version": 1,
            "point_plan_id": plan.point_plan_id,
            "static_pose_count": len(static_records),
            "dynamic_candidate_count": len(dynamic_records),
            "grasp_success_count": sum(grasp_stage_succeeded(value) for value in dynamic_records),
            "records": list(dynamic_records),
        }
        _commit_report(
            workspace, "point_targeted_dynamic", dynamic_path, dynamic_payload,
            stage_input={"campaign_input_sha256": input_sha, "static_candidate_sha256": canonical_sha256([value["candidate_sha256"] for value in static_records]), "controllers_per_pose": search.controllers_per_pose},
            extra_artifacts=_candidate_stage_artifacts(
                dynamic_records, dynamic_root, require_trace=False
            ),
        )
    dynamic_records = tuple(copy.deepcopy(value) for value in dynamic_payload["records"])

    # The expensive 16x128 refinement is deliberately a separate resumable
    # stage.  It varies the actual contact pose and wrist through the same DLS
    # projection, then executes one deterministic controller per proposal.
    local_path = workspace / "dynamic" / "local_refinement_report.json"
    local_payload = _existing_stage(workspace, "point_targeted_local_refinement", local_path)
    if local_payload is None:
        parents = tuple(rank_dynamic_grasp_results(dynamic_records)[: search.local_pose_count])
        local_jobs: list[dict[str, Any]] = []
        actual_local_count = len(parents) * search.local_refine_per_pose
        unit = _halton_local(actual_local_count, 14, seed + 12_800) if actual_local_count else np.empty((0, 14))
        for flat_index, row in enumerate(unit):
            if not parents:
                break
            parent = parents[flat_index // search.local_refine_per_pose]
            config = copy.deepcopy(dict(parent["config"]))
            orbit = float(config.get("candidate_metadata", {}).get("contact_point_target_search", {}).get("signed_orbit_deg", 0.0))
            base = PointTargetVariables.from_config(config).as_array()
            base[:8] += (2.0 * row[:8] - 1.0) * 0.02
            base[0] = np.clip(base[0], *policy.thumb_actual_range_rad)
            base[8:11] += (2.0 * row[8:11] - 1.0) * 0.0015
            base[11:14] += np.radians((2.0 * row[11:14] - 1.0) * 0.75)
            local_jobs.append({
                "candidate_id": _candidate_identifier(int(parent["candidate_id"]), flat_index, namespace=14),
                "source_id": int(parent["candidate_id"]),
                "point_plan_id": plan.point_plan_id,
                "signed_orbit_deg": orbit,
                "initial_variables": base.tolist(),
                "config": config,
            })
        refined_static_all = _run_dls_jobs(local_jobs, workers=workers, backend=selected_backend)
        refined_static = tuple(value for value in refined_static_all if bool(value.get("static_pass", False)))
        refined_dynamic = selected_backend.dynamic_runner(
            refined_static,
            dynamic_root,
            workers=workers,
            resume=True,
            seed=seed + 1,
            controller_seed_count=1,
            retain_failure_trace_count=24,
        ) if refined_static else ()
        local_payload = {
            "complete": True,
            "point_targeted_local_refinement_schema_version": 1,
            "point_plan_id": plan.point_plan_id,
            "parent_count": len(parents),
            "declared_proposal_count": search.local_refinement_count,
            "generated_proposal_count": len(local_jobs),
            "static_pass_count": len(refined_static),
            "dynamic_candidate_count": len(refined_dynamic),
            "grasp_success_count": sum(grasp_stage_succeeded(value) for value in refined_dynamic),
            "records": list(refined_dynamic),
        }
        _commit_report(
            workspace, "point_targeted_local_refinement", local_path, local_payload,
            stage_input={"campaign_input_sha256": input_sha, "point_plan_id": plan.point_plan_id, "parent_ids": [int(value["candidate_id"]) for value in parents], "proposal_count": search.local_refinement_count},
            extra_artifacts=_candidate_stage_artifacts(
                refined_dynamic, dynamic_root, require_trace=False
            ),
        )
    all_dynamic = tuple(dynamic_records) + tuple(copy.deepcopy(value) for value in local_payload["records"])

    measured_path = workspace / "dynamic" / "measured_report.json"
    measured_payload = _existing_stage(workspace, "point_targeted_measured_finalization", measured_path)
    if measured_payload is None:
        exact_sources = tuple(rank_dynamic_grasp_results(all_dynamic)[: search.exact_candidate_count])
        measured_records = selected_backend.measured_runner(
            exact_sources,
            dynamic_root,
            workers=workers,
            resume=resume,
            max_iterations=4,
        ) if exact_sources else ()
        measured_payload = {
            "complete": True,
            "point_targeted_measured_finalization_schema_version": 1,
            "point_plan_id": plan.point_plan_id,
            "exact_source_count": len(exact_sources),
            "measured_candidate_count": len(measured_records),
            "grasp_success_count": sum(measured_grasp_pose_succeeded(value) for value in measured_records),
            "full_success_count": 0,
            "records": list(measured_records),
        }
        _commit_report(
            workspace, "point_targeted_measured_finalization", measured_path, measured_payload,
            stage_input={"campaign_input_sha256": input_sha, "point_plan_id": plan.point_plan_id, "source_candidate_sha256": canonical_sha256([value["candidate_sha256"] for value in exact_sources])},
            extra_artifacts=_candidate_stage_artifacts(
                measured_records, dynamic_root, require_trace=True
            ),
        )
    measured_records = [copy.deepcopy(value) for value in measured_payload["records"]]

    manipulability_path = workspace / "manipulability" / "prescreen_report.json"
    manipulability_payload = _existing_stage(workspace, "point_targeted_manipulability", manipulability_path)
    if manipulability_payload is None:
        prescreens = []
        probe_artifacts: list[Path] = []
        budget = ContactPointManipulabilityBudget(
            maximum_delta_rad=search.manipulation_max_delta_rad,
            target_upward_m=search.virtual_lift_target_m,
        )
        for record in measured_records:
            if not measured_grasp_pose_succeeded(record):
                continue
            config_path, result_path, trace_path = _record_artifact_paths(record, dynamic_root)
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            score, probes = selected_backend.manipulability_runner(
                record["config"], trace_path, result_payload, budget=budget
            )
            if int(score.get("probe_count", -1)) != search.manipulation_probe_count:
                raise RuntimeError("manipulability runner did not execute the registered 17 probes")
            record["manipulability_prescreen"] = copy.deepcopy(score)
            probe_path = (
                workspace
                / "manipulability"
                / f"candidate_{int(record['candidate_id'])}"
                / "probes.json"
            )
            probe_payload = {
                "complete": True,
                "contact_point_manipulability_probe_set_schema_version": 1,
                "candidate_id": int(record["candidate_id"]),
                "point_plan_id": plan.point_plan_id,
                "probe_count": len(probes),
                "success_evidence": False,
                "probes": list(probes),
            }
            write_json(probe_path, probe_payload)
            probe_artifacts.append(probe_path)
            prescreens.append({
                "candidate_id": int(record["candidate_id"]),
                "score": score,
                "probe_sha256": canonical_sha256(probes),
                "probe_count": len(probes),
                "probe_artifact": str(probe_path.relative_to(workspace)),
                "probe_artifact_sha256": file_sha256(probe_path),
            })
        manipulability_payload = {
            "complete": True,
            "contact_point_manipulability_campaign_schema_version": 1,
            "count": len(prescreens),
            "success_evidence": False,
            "virtual_lift_target_m": search.virtual_lift_target_m,
            "maximum_delta_rad": search.manipulation_max_delta_rad,
            "prescreens": prescreens,
        }
        _commit_report(
            workspace, "point_targeted_manipulability", manipulability_path, manipulability_payload,
            stage_input={"campaign_input_sha256": input_sha, "measured_result_sha256": canonical_sha256([value.get("result_semantic_sha256") for value in measured_records]), "success_evidence": False},
            extra_artifacts=probe_artifacts,
        )
    else:
        by_id = {int(value["candidate_id"]): value["score"] for value in manipulability_payload["prescreens"]}
        for record in measured_records:
            if int(record["candidate_id"]) in by_id:
                record["manipulability_prescreen"] = copy.deepcopy(by_id[int(record["candidate_id"])])

    ranked_successes = sorted(
        (value for value in measured_records if measured_grasp_pose_succeeded(value)),
        key=_grasp_rank,
    )
    selected = ranked_successes[: min(search.selected_trajectory_count, target_success_count)]

    robustness_path = workspace / "robustness" / f"grasp_target_{target_success_count}.json"
    robustness_stage = f"point_targeted_grasp_robustness_{target_success_count}"
    robustness_payload = _existing_stage(workspace, robustness_stage, robustness_path)
    if robustness_payload is None:
        if selected and selected_backend.robustness_runner is not None:
            robustness_payload = selected_backend.robustness_runner(
                selected,
                workers=workers,
                seed=seed,
                local_count=search.perturbations_per_trajectory,
                best_count=definition.robustness.perturbation_count,
            )
        elif selected:
            audit_sources = []
            for discovery_index, record in enumerate(selected):
                source = copy.deepcopy(dict(record))
                source["discovery_index"] = discovery_index
                source["best"] = discovery_index == 0
                config_path, result_path, trace_path = _record_artifact_paths(
                    record, dynamic_root
                )
                source.update(
                    {
                        "config_path": config_path,
                        "result_path": result_path,
                        "trace_path": trace_path,
                    }
                )
                audit_sources.append(source)
            robustness_payload = run_v12_grasp_perturbation_audit(
                audit_sources,
                robustness_path,
                workers=workers,
                seed=seed,
                local_perturbations=search.perturbations_per_trajectory,
                best_perturbations=definition.robustness.perturbation_count,
            )
            robustness_payload["robust_grasp"] = bool(
                robustness_payload.get("robust_passed", False)
            )
        else:
            robustness_payload = {
            "complete": True,
            "grasp_only_robustness_schema_version": 1,
            "robust_grasp": False,
            "stop_reason": "no_measured_grasp_success",
            "trials": [],
            }
        _commit_report(
            workspace, robustness_stage, robustness_path, robustness_payload,
            stage_input={"campaign_input_sha256": input_sha, "target_success_count": target_success_count, "selected_candidate_ids": [int(value["candidate_id"]) for value in selected]},
        )

    # The registered order gives the 16-run local audit precedence over the
    # manipulability score.  Feed the authenticated per-grasp counts back into
    # the candidate evidence, then perform the final v12 rank used by Viewer
    # aliases.  The audit module separately chooses this winner for its 50-run
    # robustness branch.
    local_passes = {
        int(value["candidate_id"]): int(value.get("grasp_passes", 0))
        for value in robustness_payload.get("per_grasp", ())
        if isinstance(value, Mapping) and "candidate_id" in value
    }
    for record in selected:
        record["local_perturbation_pass_count"] = local_passes.get(
            int(record["candidate_id"]), 0
        )
    selected = sorted(selected, key=_grasp_rank)

    selection_path = workspace / "selection" / f"target_{target_success_count}.json"
    selection_stage = f"point_targeted_final_selection_{target_success_count}"
    selection_payload = _existing_stage(workspace, selection_stage, selection_path)
    if selection_payload is None:
        selection_payload = {
            "complete": True,
            "point_targeted_final_selection_schema_version": 1,
            "target_success_count": target_success_count,
            "point_plan_id": plan.point_plan_id,
            "ranking_policy": (
                "schema_v12_measured_local_margin_point_residual_line_"
                "force_closure_pose_saturation_candidate_id"
            ),
            "selected_candidate_ids": [
                int(value["candidate_id"]) for value in selected
            ],
            "selected": [
                {
                    "rank": index,
                    "candidate_id": int(value["candidate_id"]),
                    "local_perturbation_pass_count": int(
                        value.get("local_perturbation_pass_count", 0)
                    ),
                    "manipulability_prescreen": copy.deepcopy(
                        value.get("manipulability_prescreen", {})
                    ),
                    "contact_point_acquisition_window": copy.deepcopy(
                        value.get("summary", {})
                        .get("metrics", {})
                        .get("contact_point_targeting", {})
                        .get("acquisition_window", {})
                    ),
                    "v12_rank_evidence": v12_grasp_rank_evidence(value),
                }
                for index, value in enumerate(selected, start=1)
            ],
        }
        _commit_report(
            workspace,
            selection_stage,
            selection_path,
            selection_payload,
            stage_input={
                "campaign_input_sha256": input_sha,
                "target_success_count": target_success_count,
                "robustness_report_sha256": file_sha256(robustness_path),
            },
        )

    catalog_root = workspace / "catalogs" / f"target_{target_success_count}" / "grasp_pose"
    catalog_path = catalog_root / "catalog.json"
    catalog_stage = f"point_targeted_grasp_catalog_{target_success_count}"
    catalog_ledger = validate_stage_ledger(workspace)
    if catalog_stage not in catalog_ledger["stages"]:
        diagnostic_records = (
            sorted(measured_records, key=_grasp_rank)
            if measured_records
            else sorted(all_dynamic, key=_grasp_rank)
        )
        catalog_records = selected if selected else diagnostic_records[:1]
        candidates = _catalog_candidates(catalog_records, dynamic_root)
        if candidates:
            if catalog_root.exists():
                if not catalog_path.is_file():
                    raise RuntimeError(
                        "grasp catalog publication was interrupted before atomic completion"
                    )
            else:
                if selected:
                    selected_backend.catalog_exporter(
                        candidates,
                        catalog_root,
                        selected_count=target_success_count,
                    )
                else:
                    _publish_v12_diagnostic_catalog(
                        candidates[0],
                        catalog_root,
                        experiment_id=definition.experiment_id,
                        point_plan_id=plan.point_plan_id,
                        target_success_count=target_success_count,
                    )
            if selected:
                _rewrite_v12_catalog_selection(
                    catalog_path,
                    [int(value["candidate_id"]) for value in selected],
                )
            catalog_artifacts = _authenticate_resumable_grasp_catalog(
                catalog_path,
                experiment_id=definition.experiment_id,
                point_plan_id=plan.point_plan_id,
                target_success_count=target_success_count,
            )
            commit_campaign_stage(
                workspace,
                catalog_stage,
                stage_input={"campaign_input_sha256": input_sha, "target_success_count": target_success_count, "point_plan_id": plan.point_plan_id, "selection_report_sha256": file_sha256(selection_path)},
                artifacts=catalog_artifacts,
                summary={"catalog": str(catalog_path.relative_to(workspace))},
            )
    catalogs = {"contact_point": str(point_catalog_path.relative_to(workspace))}
    if catalog_path.is_file():
        catalogs["grasp_pose"] = str(catalog_path.relative_to(workspace))

    grasp_success_count = len(ranked_successes)
    result = {
        "contact_point_targeted_campaign_result_schema_version": CAMPAIGN_RESULT_SCHEMA_VERSION,
        "complete": True,
        "experiment_id": definition.experiment_id,
        "campaign_input_sha256": input_sha,
        "target_success_count": target_success_count,
        "target_reached": grasp_success_count >= target_success_count,
        "grasp_success_count": grasp_success_count,
        "full_success_count": 0,
        "success_semantics": "measured_full_reset_grasp_success",
        "manipulation_required_for_success": False,
        "manipulability_prescreen_is_success_evidence": False,
        "contact_point_plan_frozen": True,
        "point_plan_id": plan.point_plan_id,
        "contact_point_plan": plan.as_config(),
        "budgets": copy.deepcopy(template["contact_point_search"]["budget"]),
        "source_count": len(sources),
        "static_retained_count": int(static_payload["retained_count"]),
        "dynamic_candidate_count": len(all_dynamic),
        "measured_candidate_count": len(measured_records),
        "manipulability_prescreen_count": int(manipulability_payload["count"]),
        "selected_candidate_ids": [int(value["candidate_id"]) for value in selected],
        "robust_grasp": bool(robustness_payload.get("robust_grasp", False)),
        "validation_label": definition.actual_contact_grasp_pose_campaign.validation_labels["grasp"],
        "robust_validation_label": (
            definition.actual_contact_grasp_pose_campaign.validation_labels["robust"]
            if robustness_payload.get("robust_grasp", False)
            else None
        ),
        "catalogs": catalogs,
        "stop_reason": None if grasp_success_count >= target_success_count else "declared_budget_exhausted_before_target_grasp_count",
    }
    result_path = workspace / f"campaign_result_target_{target_success_count}.json"
    _commit_report(
        workspace, f"campaign_result_{target_success_count}", result_path, result,
        stage_input={"campaign_input_sha256": input_sha, "target_success_count": target_success_count, "point_plan_id": plan.point_plan_id, "catalogs": catalogs},
    )
    return result


# CLI-facing name.  Keep the more explicit ``*_grasp_campaign`` spelling as
# a public compatibility alias for direct Python callers and focused tests.
run_contact_point_targeted_campaign = run_contact_point_targeted_grasp_campaign


__all__ = [
    "CAMPAIGN_RESULT_SCHEMA_VERSION",
    "CampaignBackend",
    "DEFAULT_BACKEND",
    "DEFAULT_SEED",
    "materialize_v12_source_seed",
    "v12_grasp_rank_evidence",
    "run_contact_point_targeted_campaign",
    "run_contact_point_targeted_grasp_campaign",
]
