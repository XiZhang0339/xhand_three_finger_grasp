"""Authenticated MuJoCo contact-environment ablation campaigns.

This module deliberately sits beside, rather than inside, the schema-v15
search machinery.  A campaign replays one already-resolved configuration and
changes only a versioned :class:`~xhand_grasp.contact_environment.ContactEnvironmentSpec`.
The source hand/object pose, controller, plan and persisted identity fields are
therefore immutable inputs rather than new search variables.

The runner is intentionally not registered with the public CLI.  Long-running
experiments can call :func:`run_contact_environment_campaign` directly, while
unit tests can inject a small simulation runner without weakening the on-disk
resume checks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import file_sha256, json_compatible, write_json
from ..config import load_config
from ..contact_environment import (
    ContactEnvironmentSpec,
    compiled_environment_snapshot,
)
from ..grasp_pose import canonical_sha256
from ..scene import build_model
from ..simulation import run_simulation


CONTACT_ENVIRONMENT_CAMPAIGN_SCHEMA_VERSION = 1
CONTACT_ENVIRONMENT_CASE_SCHEMA_VERSION = 1
CONTACT_ENVIRONMENT_RESULT_SCHEMA_VERSION = 1
CONTACT_ENVIRONMENT_CATALOG_SCHEMA_VERSION = 1

_IDENTITY_FIELDS = (
    "object_config_id",
    "grasp_pose_id",
    "grasp_object_pair_id",
    "planner_id",
    "controller_id",
)
_FINGER_ORDER = ("thumb", "index", "mid")
_MATERIAL_FIELDS = (
    "sliding_friction",
    "torsional_friction",
    "rolling_friction",
)


SimulationRunner = Callable[..., Mapping[str, Any]]


def _strict_sha256(value: Any) -> str:
    encoded = json.dumps(
        json_compatible(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _result_semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("result_semantic_sha256", None)
    return _strict_sha256(payload)


def _case_id(source_config_semantic_sha256: str, environment_id: str) -> str:
    return canonical_sha256(
        {
            "identity_schema_version": 1,
            "domain": "xhand_contact_environment_campaign_case",
            "source_config_semantic_sha256": source_config_semantic_sha256,
            "environment_id": environment_id,
        }
    )


def _fixed_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": config.get("experiment_id"),
        "schema_version": int(config.get("schema_version", 0)),
        "identity_fields": {name: config.get(name) for name in _IDENTITY_FIELDS},
        "hand_pose_sha256": canonical_sha256(config.get("hand_pose", {})),
        "cube_sha256": canonical_sha256(config.get("cube", {})),
        "grasp_pose_sha256": canonical_sha256(config.get("grasp_pose", {})),
        "control_sha256": canonical_sha256(config.get("control", {})),
        "manipulation_plan_sha256": canonical_sha256(
            config.get("manipulation_plan", {})
        ),
        "contact_feedback_sha256": canonical_sha256(
            config.get("contact_feedback", {})
        ),
        "joint_pair_feedback_sha256": canonical_sha256(
            config.get("joint_pair_feedback", {})
        ),
    }


def _baseline_environment(config: Mapping[str, Any]) -> ContactEnvironmentSpec:
    model, info = build_model(copy.deepcopy(dict(config)))
    snapshot = compiled_environment_snapshot(model, info.cube_geom_id)
    return ContactEnvironmentSpec.from_config(
        {
            key: copy.deepcopy(snapshot[key])
            for key in ("schema_version", "contact", "solver", "required_model")
        }
    )


def default_contact_environment_cases(
    source_config: Mapping[str, Any] | str | Path,
) -> tuple[ContactEnvironmentSpec, ...]:
    """Return the focused default solver/no-slip/torsional-friction sweep.

    Sliding friction, ``condim``, normal ``solref``/``solimp``, timestep,
    integrator and gravity are copied from the source model in every case.
    Torsional-friction cases are explicitly classified as material ablations;
    the remaining cases are solver/contact-numerics ablations.
    """

    config = (
        load_config(source_config)
        if isinstance(source_config, (str, Path))
        else copy.deepcopy(dict(source_config))
    )
    baseline = _baseline_environment(config)

    def replaced(**updates: Any) -> ContactEnvironmentSpec:
        values = {
            name: getattr(baseline, name)
            for name in ContactEnvironmentSpec.__dataclass_fields__
        }
        values.update(updates)
        return ContactEnvironmentSpec(**values)

    return (
        baseline,
        replaced(iterations=200),
        replaced(tolerance=1e-10),
        replaced(impratio=30.0),
        replaced(impratio=100.0),
        replaced(noslip_iterations=5),
        replaced(noslip_iterations=10),
        replaced(noslip_iterations=20),
        replaced(torsional_friction=0.0025),
        replaced(torsional_friction=0.0075),
        replaced(torsional_friction=0.01),
        replaced(torsional_friction=0.02),
    )


def _classification(
    baseline: ContactEnvironmentSpec, candidate: ContactEnvironmentSpec
) -> str:
    if any(
        getattr(baseline, name) != getattr(candidate, name)
        for name in _MATERIAL_FIELDS
    ):
        return "material_ablation"
    return "same_object_solver_contact_ablation"


def _normalize_cases(
    cases: Sequence[ContactEnvironmentSpec | Mapping[str, Any]],
) -> tuple[ContactEnvironmentSpec, ...]:
    result: list[ContactEnvironmentSpec] = []
    seen: set[str] = set()
    for value in cases:
        spec = (
            value
            if isinstance(value, ContactEnvironmentSpec)
            else ContactEnvironmentSpec.from_config(value)
        )
        if spec.environment_id in seen:
            raise ValueError(
                "contact-environment case list contains duplicate environment_id "
                f"{spec.environment_id}"
            )
        seen.add(spec.environment_id)
        result.append(spec)
    if not result:
        raise ValueError("contact-environment campaign requires at least one case")
    # Environment identity, not caller order or worker count, defines execution
    # and ranking tie order.
    return tuple(sorted(result, key=lambda item: item.environment_id))


def _campaign_manifest(
    source_path: Path,
    source_config: Mapping[str, Any],
    cases: Sequence[ContactEnvironmentSpec],
    baseline: ContactEnvironmentSpec,
) -> dict[str, Any]:
    body = {
        "contact_environment_campaign_schema_version": (
            CONTACT_ENVIRONMENT_CAMPAIGN_SCHEMA_VERSION
        ),
        "source_config_path": str(source_path),
        "source_config_sha256": file_sha256(source_path),
        "source_config_semantic_sha256": canonical_sha256(source_config),
        "fixed_identity": _fixed_identity(source_config),
        "baseline_environment_id": baseline.environment_id,
        "environment_ids": [spec.environment_id for spec in cases],
        "environments_sha256": canonical_sha256(
            [spec.as_config() for spec in cases]
        ),
    }
    return {**body, "campaign_input_sha256": canonical_sha256(body)}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _initialize_workspace(
    output_dir: Path, manifest: Mapping[str, Any], *, resume: bool
) -> None:
    manifest_path = output_dir / "campaign_manifest.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(
                f"output directory already exists: {output_dir}; use resume=True"
            )
        if not manifest_path.is_file():
            raise RuntimeError("resume workspace has no campaign_manifest.json")
        existing = _read_json(manifest_path)
        if existing != json_compatible(manifest):
            raise RuntimeError("contact-environment resume inputs changed")
        return
    output_dir.mkdir(parents=True)
    write_json(manifest_path, manifest)


def _atomic_trace_path(destination: Path) -> tuple[Path, int]:
    descriptor, name = tempfile.mkstemp(
        dir=destination, prefix=".trace.", suffix=".npz"
    )
    os.close(descriptor)
    temporary = Path(name)
    # ``np.savez_compressed`` and the simulation runner expect to create the
    # archive themselves.  Remove only this freshly-created empty inode.
    temporary.unlink()
    return temporary, descriptor


def _trace_contact_count_metrics(trace_path: Path) -> dict[str, Any]:
    with np.load(trace_path, allow_pickle=False) as trace:
        metrics: dict[str, Any] = {}
        for label, field in (
            ("cube", "contact_environment_cube_contact_count"),
            ("active_distal", "contact_environment_active_distal_contact_count"),
        ):
            if field not in trace.files:
                raise RuntimeError(f"environment trace is missing {field}")
            values = np.asarray(trace[field], dtype=np.int64)
            if values.ndim != 1 or values.size == 0:
                raise RuntimeError(f"environment trace field {field} is malformed")
            jumps = np.abs(np.diff(values))
            metrics[label] = {
                "minimum": int(np.min(values)),
                "maximum": int(np.max(values)),
                "maximum_step_jump": int(np.max(jumps)) if jumps.size else 0,
                "nonzero_duty": float(np.mean(values > 0)),
            }
        environment_id = np.asarray(trace["contact_environment_id"]).item()
        metrics["environment_id"] = str(environment_id)
    return metrics


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    return resolved if np.isfinite(resolved) else None


def _extract_metrics(
    summary: Mapping[str, Any], trace_path: Path
) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    stage = summary.get("stage_status", {})
    if not isinstance(metrics, Mapping) or not isinstance(stage, Mapping):
        raise RuntimeError("simulation summary is missing metrics/stage_status")
    targeting = metrics.get("contact_point_targeting", {})
    if not isinstance(targeting, Mapping):
        targeting = {}
    slip_root = targeting.get("contact_slip_from_grasp", {})
    operation = slip_root.get("operation", {}) if isinstance(slip_root, Mapping) else {}
    per_finger = operation.get("per_finger", {}) if isinstance(operation, Mapping) else {}
    slips = {
        finger: _optional_float(
            per_finger.get(finger, {}).get("tangent_slip_max_m")
            if isinstance(per_finger.get(finger, {}), Mapping)
            else None
        )
        for finger in _FINGER_ORDER
    }
    planned = metrics.get("contact_preserving_planned_lift", {})
    pair = metrics.get("joint_pair_alignment", {})
    if not isinstance(planned, Mapping):
        planned = {}
    if not isinstance(pair, Mapping):
        pair = {}
    return {
        "grasp_status": str(stage.get("grasp", "unknown")),
        "manipulation_status": str(stage.get("manipulation", "unknown")),
        "full_success": bool(stage.get("full_success", summary.get("passed", False))),
        "passed": bool(summary.get("passed", False)),
        "maximum_plan_progress": _optional_float(
            planned.get("maximum_plan_progress")
        ),
        "final_plan_progress": _optional_float(planned.get("final_plan_progress")),
        "median_lift_m": _optional_float(metrics.get("median_lift_m")),
        "minimum_lift_m": _optional_float(metrics.get("minimum_lift_m")),
        "tangential_slip_max_m": slips,
        "joint_pair_operation_p95_deg": _optional_float(
            pair.get("operation_p95_deg")
        ),
        "joint_pair_operation_max_deg": _optional_float(
            pair.get("operation_max_deg")
        ),
        "contact_count": _trace_contact_count_metrics(trace_path),
    }


def _rank_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = result["metrics"]
    slips = metrics["tangential_slip_max_m"]
    thumb = slips.get("thumb")
    thumb_rank = float("inf") if thumb is None else float(thumb)
    progress = metrics.get("maximum_plan_progress")
    minimum_lift = metrics.get("minimum_lift_m")
    pair = metrics.get("joint_pair_operation_p95_deg")
    contact_jump = metrics["contact_count"]["active_distal"]["maximum_step_jump"]
    thumb_slip_safe = thumb is not None and float(thumb) < 0.0015
    full_success = bool(metrics.get("full_success", False))
    return (
        not (full_success and thumb_slip_safe),
        not full_success,
        not thumb_slip_safe,
        str(metrics.get("manipulation_status")) not in {"passed", "succeeded"},
        str(metrics.get("grasp_status")) != "acquired",
        thumb_rank,
        -(float(progress) if progress is not None else -1.0),
        -(float(minimum_lift) if minimum_lift is not None else -1.0),
        float(pair) if pair is not None else float("inf"),
        int(contact_jump),
        str(result["environment_id"]),
    )


def _case_environment_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    spec = ContactEnvironmentSpec.from_config(payload["environment"])
    body = {
        "contact_environment_case_schema_version": (
            CONTACT_ENVIRONMENT_CASE_SCHEMA_VERSION
        ),
        "case_id": payload["case_id"],
        "environment_id": spec.environment_id,
        "classification": payload["classification"],
        "source_config_sha256": payload["source_config_sha256"],
        "source_config_semantic_sha256": payload[
            "source_config_semantic_sha256"
        ],
        "fixed_identity": copy.deepcopy(payload["fixed_identity"]),
        "environment": spec.as_config(),
    }
    body["case_document_sha256"] = canonical_sha256(body)
    return body


def _authenticate_case_result(
    case_dir: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    environment_path = case_dir / "environment.json"
    result_path = case_dir / "result.json"
    trace_path = case_dir / "trace.npz"
    if not all(path.is_file() for path in (environment_path, result_path, trace_path)):
        raise RuntimeError(f"committed environment case is incomplete: {case_dir}")
    expected_environment = _case_environment_document(payload)
    if _read_json(environment_path) != json_compatible(expected_environment):
        raise RuntimeError(f"environment case changed on resume: {case_dir.name}")
    result = _read_json(result_path)
    expected_semantic = result.get("result_semantic_sha256")
    if expected_semantic != _result_semantic_sha256(result):
        raise RuntimeError(f"environment result semantic SHA mismatch: {case_dir.name}")
    if result.get("summary_sha256") != _strict_sha256(result.get("summary")):
        raise RuntimeError(f"environment summary SHA mismatch: {case_dir.name}")
    checks = {
        "case_id": payload["case_id"],
        "environment_id": payload["environment_id"],
        "source_config_sha256": payload["source_config_sha256"],
        "source_config_semantic_sha256": payload[
            "source_config_semantic_sha256"
        ],
        "environment_sha256": file_sha256(environment_path),
        "trace_sha256": file_sha256(trace_path),
    }
    for name, expected in checks.items():
        if result.get(name) != expected:
            raise RuntimeError(
                f"environment result {name} mismatch: {case_dir.name}"
            )
    with np.load(trace_path, allow_pickle=False) as trace:
        actual_environment_id = str(
            np.asarray(trace["contact_environment_id"]).item()
        )
    if actual_environment_id != payload["environment_id"]:
        raise RuntimeError(f"environment trace ID mismatch: {case_dir.name}")
    if result.get("fixed_identity") != json_compatible(payload["fixed_identity"]):
        raise RuntimeError(f"fixed source identity changed: {case_dir.name}")
    return result


def _authenticate_existing_catalog(workspace: Path) -> None:
    path = workspace / "catalog.json"
    if not path.is_file():
        return
    catalog = _read_json(path)
    declared = catalog.pop("catalog_sha256", None)
    if declared != canonical_sha256(catalog):
        raise RuntimeError("contact-environment catalog SHA mismatch")
    for entry in catalog.get("trajectories", []):
        if not isinstance(entry, Mapping):
            raise RuntimeError("contact-environment catalog entry is malformed")
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise RuntimeError("contact-environment catalog artifacts are malformed")
        hashes = artifacts.get("sha256")
        if not isinstance(hashes, Mapping):
            raise RuntimeError("contact-environment catalog hashes are malformed")
        for name in (
            "resolved_config",
            "contact_environment",
            "result",
            "trace",
        ):
            relative = artifacts.get(name)
            expected = hashes.get(name)
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise RuntimeError(
                    f"contact-environment catalog lost {name} evidence"
                )
            candidate = (workspace / relative).resolve()
            if not candidate.is_relative_to(workspace) or not candidate.is_file():
                raise RuntimeError(
                    f"contact-environment catalog has unsafe {name} evidence"
                )
            if file_sha256(candidate) != expected:
                raise RuntimeError(
                    f"contact-environment catalog {name} SHA mismatch"
                )


def _execute_case(
    payload: Mapping[str, Any],
    *,
    simulation_runner: SimulationRunner = run_simulation,
) -> dict[str, Any]:
    case_dir = Path(payload["case_dir"])
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    if result_path.is_file():
        return _authenticate_case_result(case_dir, payload)

    environment_document = _case_environment_document(payload)
    environment_path = case_dir / "environment.json"
    write_json(environment_path, environment_document)
    spec = ContactEnvironmentSpec.from_config(payload["environment"])
    config = load_config(payload["source_config_path"])
    if canonical_sha256(config) != payload["source_config_semantic_sha256"]:
        raise RuntimeError("source config changed before environment execution")
    if _fixed_identity(config) != payload["fixed_identity"]:
        raise RuntimeError("source pose/control/identity changed before execution")

    temporary_trace, _ = _atomic_trace_path(case_dir)
    try:
        summary = dict(
            simulation_runner(
                copy.deepcopy(config),
                trace_path=temporary_trace,
                contact_environment=spec,
            )
        )
        if not temporary_trace.is_file():
            raise RuntimeError("simulation runner did not produce trace.npz")
        with np.load(temporary_trace, allow_pickle=False) as trace:
            environment_id = str(
                np.asarray(trace["contact_environment_id"]).item()
            )
        if environment_id != spec.environment_id:
            raise RuntimeError("simulation trace carries the wrong environment_id")
        evidence = summary.get("contact_environment")
        if not isinstance(evidence, Mapping):
            raise RuntimeError("simulation summary has no environment evidence")
        requested = evidence.get("requested")
        compiled = evidence.get("compiled")
        if not isinstance(requested, Mapping) or not isinstance(compiled, Mapping):
            raise RuntimeError("simulation environment evidence is incomplete")
        if requested.get("environment_id") != spec.environment_id or compiled.get(
            "environment_id"
        ) != spec.environment_id:
            raise RuntimeError("requested/compiled environment evidence disagrees")

        metrics = _extract_metrics(summary, temporary_trace)
        trace_path = case_dir / "trace.npz"
        os.replace(temporary_trace, trace_path)
        result: dict[str, Any] = {
            "contact_environment_result_schema_version": (
                CONTACT_ENVIRONMENT_RESULT_SCHEMA_VERSION
            ),
            "case_id": payload["case_id"],
            "environment_id": spec.environment_id,
            "classification": payload["classification"],
            "source_config_path": payload["source_config_path"],
            "source_config_sha256": payload["source_config_sha256"],
            "source_config_semantic_sha256": payload[
                "source_config_semantic_sha256"
            ],
            "fixed_identity": copy.deepcopy(payload["fixed_identity"]),
            "environment_sha256": file_sha256(environment_path),
            "trace_sha256": file_sha256(trace_path),
            "summary_sha256": _strict_sha256(summary),
            "metrics": metrics,
            "summary": summary,
        }
        result["result_semantic_sha256"] = _result_semantic_sha256(result)
        write_json(result_path, result)
        return _authenticate_case_result(case_dir, payload)
    finally:
        temporary_trace.unlink(missing_ok=True)


def _spawn_case(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _execute_case(payload)


def _catalog_entry(
    result: Mapping[str, Any], workspace: Path, rank: int
) -> dict[str, Any]:
    case_dir = workspace / "cases" / str(result["case_id"])
    environment_path = case_dir / "environment.json"
    result_path = case_dir / "result.json"
    trace_path = case_dir / "trace.npz"
    return {
        "trajectory_id": str(result["case_id"]),
        "label": f"environment_rank_{int(rank):02d}",
        "aliases": [],
        "environment_id": str(result["environment_id"]),
        "rank": int(rank),
        "classification": str(result["classification"]),
        "metrics": copy.deepcopy(result["metrics"]),
        "fixed_identity": copy.deepcopy(result["fixed_identity"]),
        "artifacts": {
            "resolved_config": "source_config.json",
            "contact_environment": str(environment_path.relative_to(workspace)),
            "result": str(result_path.relative_to(workspace)),
            "trace": str(trace_path.relative_to(workspace)),
            "sha256": {
                "resolved_config": file_sha256(workspace / "source_config.json"),
                "contact_environment": file_sha256(environment_path),
                "result": file_sha256(result_path),
                "trace": file_sha256(trace_path),
            },
        },
    }


def run_contact_environment_campaign(
    source_config_path: str | Path,
    output_dir: str | Path,
    cases: Sequence[ContactEnvironmentSpec | Mapping[str, Any]] | None = None,
    *,
    resume: bool = False,
    workers: int = 1,
    expected_source_sha256: str | None = None,
    simulation_runner: SimulationRunner | None = None,
) -> dict[str, Any]:
    """Run or authenticate a fixed-config contact-environment campaign.

    ``workers>1`` always uses Python's ``spawn`` context.  A custom runner is
    intentionally restricted to one worker so a unit-test closure cannot
    accidentally become part of a non-reproducible multiprocessing protocol.
    """

    if isinstance(workers, bool) or int(workers) != workers or int(workers) < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    if simulation_runner is not None and workers != 1:
        raise ValueError("a custom simulation_runner requires workers=1")
    source_path = Path(source_config_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    actual_source_sha256 = file_sha256(source_path)
    if (
        expected_source_sha256 is not None
        and actual_source_sha256 != expected_source_sha256
    ):
        raise RuntimeError("source config SHA-256 does not match expectation")
    source_config = load_config(source_path)
    baseline = _baseline_environment(source_config)
    requested_cases = list(
        default_contact_environment_cases(source_config) if cases is None else cases
    )
    requested_ids = {
        (
            value.environment_id
            if isinstance(value, ContactEnvironmentSpec)
            else ContactEnvironmentSpec.from_config(value).environment_id
        )
        for value in requested_cases
    }
    if baseline.environment_id not in requested_ids:
        requested_cases.append(baseline)
    normalized = _normalize_cases(requested_cases)
    manifest = _campaign_manifest(source_path, source_config, normalized, baseline)
    workspace = Path(output_dir).expanduser().resolve()
    _initialize_workspace(workspace, manifest, resume=bool(resume))
    published_source = workspace / "source_config.json"
    if published_source.is_file():
        if load_config(published_source) != source_config:
            raise RuntimeError("published source_config.json changed on resume")
    else:
        write_json(published_source, source_config)
    if resume:
        _authenticate_existing_catalog(workspace)

    payloads: list[dict[str, Any]] = []
    for spec in normalized:
        case_id = _case_id(manifest["source_config_semantic_sha256"], spec.environment_id)
        payloads.append(
            {
                "case_id": case_id,
                "case_dir": str(workspace / "cases" / case_id),
                "source_config_path": str(source_path),
                "source_config_sha256": actual_source_sha256,
                "source_config_semantic_sha256": manifest[
                    "source_config_semantic_sha256"
                ],
                "fixed_identity": copy.deepcopy(manifest["fixed_identity"]),
                "environment_id": spec.environment_id,
                "environment": spec.as_config(),
                "classification": _classification(baseline, spec),
            }
        )

    if workers == 1:
        runner = run_simulation if simulation_runner is None else simulation_runner
        results = [_execute_case(value, simulation_runner=runner) for value in payloads]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            results = list(executor.map(_spawn_case, payloads))

    ranked = sorted(results, key=_rank_key)
    baseline_result = next(
        (value for value in results if value["environment_id"] == baseline.environment_id),
        None,
    )
    best = ranked[0]
    aliases = {
        "best_environment": str(best["case_id"]),
        "best_attempt": str(best["case_id"]),
    }
    same_object = [
        value
        for value in ranked
        if value["classification"] == "same_object_solver_contact_ablation"
    ]
    if same_object:
        aliases["best_same_object"] = str(same_object[0]["case_id"])
    highest_progress = max(
        results,
        key=lambda value: (
            float(value["metrics"].get("maximum_plan_progress") or -1.0),
            float(value["metrics"].get("median_lift_m") or -1.0),
            -float(
                value["metrics"]["tangential_slip_max_m"].get("thumb")
                or float("inf")
            ),
            str(value["case_id"]),
        ),
    )
    aliases["highest_progress"] = str(highest_progress["case_id"])
    if baseline_result is not None:
        aliases["baseline"] = str(baseline_result["case_id"])
    trajectories = [
        _catalog_entry(value, workspace, index)
        for index, value in enumerate(ranked, start=1)
    ]
    by_id = {entry["trajectory_id"]: entry for entry in trajectories}
    for alias, trajectory_id in aliases.items():
        entry = by_id[trajectory_id]
        entry["aliases"].append(alias)
    for entry in trajectories:
        entry["aliases"].sort()
    catalog_body = {
        "contact_environment_catalog_schema_version": (
            CONTACT_ENVIRONMENT_CATALOG_SCHEMA_VERSION
        ),
        "campaign_input_sha256": manifest["campaign_input_sha256"],
        "source_config_sha256": actual_source_sha256,
        "source_config_semantic_sha256": manifest[
            "source_config_semantic_sha256"
        ],
        "fixed_identity": copy.deepcopy(manifest["fixed_identity"]),
        "aliases": aliases,
        "trajectories": trajectories,
    }
    catalog = {**catalog_body, "catalog_sha256": canonical_sha256(catalog_body)}
    write_json(workspace / "catalog.json", catalog)
    report_body = {
        "contact_environment_campaign_report_schema_version": 1,
        "campaign_input_sha256": manifest["campaign_input_sha256"],
        "case_count": len(results),
        "worker_count": workers,
        "resume": bool(resume),
        "aliases": aliases,
        "ranked_case_ids": [str(value["case_id"]) for value in ranked],
        "classification_counts": {
            label: sum(value["classification"] == label for value in results)
            for label in (
                "same_object_solver_contact_ablation",
                "material_ablation",
            )
        },
        "results": [
            {
                "case_id": value["case_id"],
                "environment_id": value["environment_id"],
                "classification": value["classification"],
                "metrics": value["metrics"],
            }
            for value in ranked
        ],
    }
    report = {**report_body, "report_sha256": canonical_sha256(report_body)}
    write_json(workspace / "campaign_report.json", report)
    return {
        "workspace": str(workspace),
        "manifest": manifest,
        "catalog": catalog,
        "report": report,
    }


__all__ = [
    "CONTACT_ENVIRONMENT_CAMPAIGN_SCHEMA_VERSION",
    "CONTACT_ENVIRONMENT_CASE_SCHEMA_VERSION",
    "CONTACT_ENVIRONMENT_CATALOG_SCHEMA_VERSION",
    "CONTACT_ENVIRONMENT_RESULT_SCHEMA_VERSION",
    "default_contact_environment_cases",
    "run_contact_environment_campaign",
]
