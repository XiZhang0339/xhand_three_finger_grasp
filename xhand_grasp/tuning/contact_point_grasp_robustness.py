"""Schema-v12 grasp-only full-reset perturbation audit.

Unlike the schema-v9 manipulation robustness campaign, this audit deliberately
does not require ``summary.passed``, manipulation success, or full success.
Every published source must already be a verified schema-v12 grasp and every
perturbation passes only when the grasp stage and all contact-point-specific
hard checks pass after a complete no-contact reset rerun.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import REPO_ROOT, file_sha256, write_json
from ..actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
)
from ..config import validate_config
from ..experiment import resolve_experiment
from ..grasp_pose import canonical_sha256, controller_id, grasp_pose_id
from ..search import _run_candidates
from .actual_contact_grasp_pose_robustness import (
    _derived_seed,
    _latin_hypercube,
    _scale,
)


ROBUSTNESS_REPORT_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
LOCAL_PERTURBATION_COUNT = 16
BEST_PERTURBATION_COUNT = 50
MAX_NOMINAL_GRASPS = 5
REQUIRED_BEST_PASSES = 45
LOCAL_FAMILY = "v12_grasp_per_nominal_local_16"
BEST_FAMILY = "v12_grasp_best_pose_material_50"
V12_CONTACT_POINT_HARD_CHECKS = (
    "v12_contact_point_trace_matches_raw_contacts",
    "v12_contact_point_gate_matches_raw_contacts",
    "grasp_contact_points_contiguous",
)


CandidateRunner = Callable[
    [list[tuple[int, dict[str, Any]]], int], list[dict[str, Any]]
]


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _v12_definition(config: Mapping[str, Any]):
    schema_version = int(config.get("schema_version", 0))
    if schema_version not in (12, 13):
        raise ValueError("grasp robustness requires schema_version 12 or 13")
    definition = resolve_experiment(config)
    if (
        definition.contact_point_search is None
        and definition.scaled_contact_downsize_campaign is None
    ):
        raise ValueError(
            "grasp robustness requires a registered contact-point experiment"
        )
    return definition


def v12_grasp_hard_success(summary: Mapping[str, Any]) -> bool:
    """Return the grasp-only verdict, explicitly ignoring manipulation."""

    stage = summary.get("stage_status")
    checks = summary.get("checks")
    return bool(
        isinstance(stage, Mapping)
        and stage.get("grasp_success") is True
        and isinstance(checks, Mapping)
        and all(checks.get(name) is True for name in V12_CONTACT_POINT_HARD_CHECKS)
    )


def v12_grasp_failure_reasons(summary: Mapping[str, Any]) -> list[str]:
    """Return stable, grasp-specific reasons while retaining raw failures too."""

    reasons: list[str] = []
    stage = summary.get("stage_status")
    if not isinstance(stage, Mapping) or stage.get("grasp_success") is not True:
        reasons.append("stage_status.grasp_success")
    checks = summary.get("checks")
    for name in V12_CONTACT_POINT_HARD_CHECKS:
        if not isinstance(checks, Mapping) or checks.get(name) is not True:
            reasons.append(name)
    if summary.get("error") is not None:
        reasons.append("simulation_error")
    return reasons


@dataclass(frozen=True, slots=True)
class V12GraspRobustnessSource:
    """One verified nominal schema-v12 grasp supplied by the campaign."""

    candidate_id: str
    config: dict[str, Any]
    summary: dict[str, Any]
    discovery_index: int = 0
    best: bool = False
    config_path: Path | None = None
    result_path: Path | None = None
    trace_path: Path | None = None

    def __post_init__(self) -> None:
        config = copy.deepcopy(dict(self.config))
        summary = copy.deepcopy(dict(self.summary))
        _v12_definition(config)
        validate_config(config)
        if config.get("run_context") is not None:
            raise ValueError(
                "robustness sources must be canonical full-reset nominal configs"
            )
        if not v12_grasp_hard_success(summary):
            raise ValueError(
                "robustness source must be a verified schema-v12 grasp"
            )
        if (
            not isinstance(self.discovery_index, int)
            or isinstance(self.discovery_index, bool)
            or self.discovery_index < 0
        ):
            raise ValueError("discovery_index must be a non-negative integer")
        if not isinstance(self.best, bool):
            raise ValueError("best must be boolean")
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "summary", summary)
        for name in ("config_path", "result_path", "trace_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())

    @property
    def point_plan_id(self) -> str:
        return str(self.config["contact_point_plan"]["point_plan_id"])

    @property
    def rank(self) -> tuple[Any, ...]:
        return (not self.best, self.discovery_index, self.candidate_id)

    @classmethod
    def from_record(
        cls, value: "V12GraspRobustnessSource | Mapping[str, Any]", index: int
    ) -> "V12GraspRobustnessSource":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("robustness source must be a source object or mapping")
        result = value.get("result")
        result_mapping = result if isinstance(result, Mapping) else {}
        summary = value.get("summary", result_mapping.get("summary"))
        config = value.get("config", result_mapping.get("config"))
        if not isinstance(config, Mapping) or not isinstance(summary, Mapping):
            raise ValueError("source record must contain config and summary mappings")
        aliases = value.get("aliases", ())
        best = bool(
            value.get("best", False)
            or value.get("best_first", False)
            or (isinstance(aliases, Sequence) and "best_first" in aliases)
            or (isinstance(aliases, Sequence) and "best_nominal" in aliases)
        )
        return cls(
            candidate_id=str(
                value.get(
                    "candidate_id",
                    result_mapping.get("candidate_id", index),
                )
            ),
            config=copy.deepcopy(dict(config)),
            summary=copy.deepcopy(dict(summary)),
            discovery_index=int(value.get("discovery_index", index)),
            best=best,
            config_path=value.get("config_path"),
            result_path=value.get("result_path"),
            trace_path=value.get("trace_path"),
        )


def discover_v12_grasp_robustness_sources(
    search_roots: Sequence[str | Path],
    *,
    maximum_nominal_grasps: int = MAX_NOMINAL_GRASPS,
) -> tuple[V12GraspRobustnessSource, ...]:
    """Discover authenticated published v12 *grasp* trajectories.

    The filesystem/authentication layer is shared with the actual-contact
    catalog reader, but eligibility is deliberately recomputed with the v12
    grasp-only hard contract.  In particular, a missing/failed manipulation
    stage does not disqualify a nominal grasp and a diagnostic near miss can
    never become a robustness source.
    """

    from .actual_contact_grasp_pose_robustness import (
        discover_v9_robustness_sources,
    )

    allowed_results: set[Path] = set()
    for raw_root in search_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file() and root.name == "result.json":
            allowed_results.add(root)
            continue
        catalog_paths = (
            (root,)
            if root.is_file()
            else tuple(sorted(root.rglob("catalog.json")))
        )
        for catalog_path in catalog_paths:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            if not isinstance(catalog, Mapping) or not isinstance(
                catalog.get("trajectories"), list
            ):
                continue
            catalog_root = catalog_path.parent.resolve()
            for entry in catalog["trajectories"]:
                if (
                    not isinstance(entry, Mapping)
                    or entry.get("classification") != "success"
                ):
                    continue
                artifacts = entry.get("artifacts")
                relative = (
                    artifacts.get("result")
                    if isinstance(artifacts, Mapping)
                    else None
                )
                if (
                    not isinstance(relative, str)
                    or Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                ):
                    raise ValueError(
                        "schema-v12 success catalog has an unsafe result path"
                    )
                result_path = (catalog_root / relative).resolve()
                if (
                    not result_path.is_relative_to(catalog_root)
                    or not result_path.is_file()
                ):
                    raise ValueError(
                        "schema-v12 success catalog lost its result artifact"
                    )
                allowed_results.add(result_path)

    source_limit = _positive_int(
        maximum_nominal_grasps, "maximum_nominal_grasps"
    )
    discovered = discover_v9_robustness_sources(search_roots)
    eligible = []
    for source in discovered:
        if source.result_path is None or source.result_path not in allowed_results:
            continue
        result = json.loads(source.result_path.read_text(encoding="utf-8"))
        if int(result.get("candidate_result_schema_version", 0)) != 1:
            raise ValueError(
                "schema-v12 robustness source lacks authenticated candidate evidence"
            )
        authenticate_candidate_result_semantic_sha256(
            result, source=source.result_path
        )
        if (
            result.get("stage") != "measured_grasp_pose_finalization"
            or result.get("measured_grasp_pose_success") is not True
            or result.get("grasp_success") is not True
        ):
            raise ValueError(
                "schema-v12 robustness source is not a successful measured "
                "full-reset grasp"
            )
        try:
            definition = _v12_definition(source.config)
        except ValueError:
            continue
        if (
            definition.contact_point_search is None
            and definition.scaled_contact_downsize_campaign is None
        ) or not v12_grasp_hard_success(source.summary):
            continue
        eligible.append(source)
    if not eligible:
        raise ValueError(
            "no authenticated schema-v12 grasp-success trajectories were found"
        )

    # The catalog reader has already de-duplicated exact
    # candidate/grasp/controller identities.  Preserve deterministic discovery
    # order and the canonical best-first alias used by published catalogs.
    ordered = sorted(
        eligible,
        key=lambda source: (source.discovery_index, source.candidate_id),
    )
    registered = _v12_definition(ordered[0].config).scaled_contact_downsize_campaign
    registered_limit = (
        MAX_NOMINAL_GRASPS
        if registered is None
        else registered.selected_grasp_count
    )
    if source_limit > registered_limit:
        raise ValueError(
            "maximum_nominal_grasps exceeds the registered publication capacity"
        )
    explicit_best = [source for source in ordered if source.best_first]
    best = explicit_best[0] if explicit_best else ordered[0]
    selected = ordered[:source_limit]
    if best not in selected:
        selected[-1] = best
        selected.sort(key=lambda source: (source.discovery_index, source.candidate_id))
    return tuple(
        V12GraspRobustnessSource(
            candidate_id=source.candidate_id,
            config=source.config,
            summary=source.summary,
            discovery_index=source.discovery_index,
            best=source is best,
            config_path=source.config_path,
            result_path=source.result_path,
            trace_path=source.trace_path,
        )
        for source in selected
    )


def generate_v12_grasp_perturbation_configs(
    config: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    source_candidate_id: str | int,
    family: str,
) -> list[dict[str, Any]]:
    """Generate deterministic v12 pose/material cases from the registered envelope."""

    sample_count = _positive_int(count, "count")
    if family not in {LOCAL_FAMILY, BEST_FAMILY}:
        raise ValueError("unknown schema-v12 grasp perturbation family")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    base = copy.deepcopy(dict(config))
    if base.get("run_context") is not None:
        raise ValueError("v12 robustness requires a canonical full-reset config")
    validate_config(base)
    definition = _v12_definition(base)
    parameters = definition.robustness
    matrix = _latin_hypercube(sample_count, 8, seed)
    base_xy = np.asarray(base["cube"]["center_xy_m"], dtype=np.float64)
    base_rpy = np.asarray(
        base["cube"].get("rpy_deg", (0.0, 0.0, 0.0)), dtype=np.float64
    )
    base_gap = float(base["cube"].get("z_offset_m", 0.0))
    base_mass = float(base["cube"]["mass_kg"])
    base_friction = float(base["cube"]["friction"])
    source_binding = {}
    if int(base.get("schema_version", 0)) == 13:
        source_binding = {
            "source_config_sha256": canonical_sha256(base),
            "source_grasp_pose_id": grasp_pose_id(base),
            "source_controller_id": controller_id(base),
            "source_had_candidate_metadata": "candidate_metadata" in base,
            "source_cube": copy.deepcopy(base["cube"]),
        }
        source_binding["source_candidate_binding_sha256"] = canonical_sha256(
            {
                "source_candidate_id": str(source_candidate_id),
                "source_config_sha256": source_binding["source_config_sha256"],
                "source_grasp_pose_id": source_binding["source_grasp_pose_id"],
                "source_controller_id": source_binding["source_controller_id"],
            }
        )
    cases: list[dict[str, Any]] = []
    for trial, row in enumerate(matrix):
        xy_delta = np.asarray(
            (
                _scale(row[0], parameters.position_xy_delta_m),
                _scale(row[1], parameters.position_xy_delta_m),
            ),
            dtype=np.float64,
        )
        gap_delta = _scale(row[2], parameters.z_offset_delta_m)
        rpy_delta = np.asarray(
            [
                _scale(row[3 + axis], parameters.rpy_delta_deg)
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        mass_scale = _scale(row[6], parameters.mass_scale)
        friction_delta = _scale(row[7], parameters.friction_delta)
        case = copy.deepcopy(base)
        case.pop("experiment_status", None)
        case["run_context"] = {"kind": "robustness_trial"}
        case["cube"]["center_xy_m"] = (base_xy + xy_delta).tolist()
        case["cube"]["z_offset_m"] = base_gap + gap_delta
        case["cube"]["rpy_deg"] = (base_rpy + rpy_delta).tolist()
        case["cube"]["mass_kg"] = base_mass * mass_scale
        case["cube"]["friction"] = base_friction + friction_delta
        resolved = {
            "cube_center_xy_delta_m": xy_delta.tolist(),
            "cube_gap_delta_m": float(gap_delta),
            "cube_rpy_delta_deg": rpy_delta.tolist(),
            "mass_scale": float(mass_scale),
            "friction_delta": float(friction_delta),
        }
        metadata = copy.deepcopy(dict(case.get("candidate_metadata", {})))
        metadata["v12_grasp_robustness_trial"] = {
            "family": family,
            "seed": int(seed),
            "trial": int(trial),
            "source_candidate_id": str(source_candidate_id),
            **source_binding,
            "point_plan_id": str(
                case["contact_point_plan"]["point_plan_id"]
            ),
            "resolved_perturbations": resolved,
            "success_scope": "grasp_only_contact_point_hard_checks",
            "manipulation_success_required": False,
            "full_success_required": False,
            "full_reset_rerun": True,
            "initial_state_source": "configured_no_contact_reset",
            "checkpoint_used": False,
        }
        case["candidate_metadata"] = metadata
        validate_config(case)
        cases.append(case)
    return cases


def _source_record(source: V12GraspRobustnessSource) -> dict[str, Any]:
    def artifact(path: Path | None) -> tuple[str | None, str | None]:
        return (
            (str(path), file_sha256(path))
            if path is not None and path.is_file()
            else (None, None)
        )

    config_path, config_sha = artifact(source.config_path)
    result_path, result_sha = artifact(source.result_path)
    trace_path, trace_sha = artifact(source.trace_path)
    return {
        "candidate_id": source.candidate_id,
        "candidate_sha256": canonical_sha256(source.config),
        "grasp_pose_id": grasp_pose_id(source.config),
        "controller_id": controller_id(source.config),
        "point_plan_id": source.point_plan_id,
        "best": source.best,
        "nominal_grasp_success": True,
        "nominal_manipulation_success_ignored": True,
        "artifacts": {
            "config": config_path,
            "result": result_path,
            "trace": trace_path,
            "sha256": {
                "config": config_sha,
                "result": result_sha,
                "trace": trace_sha,
            },
        },
    }


def _trial_record(
    result: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    summary = _json_safe(result.get("summary", {}))
    config = copy.deepcopy(dict(result.get("config", {})))
    trial_metadata = config.get("candidate_metadata", {}).get(
        "v12_grasp_robustness_trial", {}
    )
    return {
        "trial": int(metadata["trial"]),
        "run_candidate_id": int(result["candidate_id"]),
        "source_candidate_id": str(metadata["source_candidate_id"]),
        "point_plan_id": str(metadata["point_plan_id"]),
        "family": str(metadata["family"]),
        "seed": int(metadata["seed"]),
        "grasp_passed": v12_grasp_hard_success(summary),
        "manipulation_and_full_success_ignored": True,
        "config_sha256": canonical_sha256(config),
        "grasp_failure_reasons": v12_grasp_failure_reasons(summary),
        "failed_checks": copy.deepcopy(summary.get("failed_checks", [])),
        "checks": copy.deepcopy(summary.get("checks", {})),
        "stage_status": copy.deepcopy(summary.get("stage_status", {})),
        "metrics": copy.deepcopy(summary.get("metrics", {})),
        "error": summary.get("error"),
        "cube": copy.deepcopy(config.get("cube", {})),
        "resolved_perturbations": copy.deepcopy(
            trial_metadata.get("resolved_perturbations", {})
        ),
        "full_reset_rerun": bool(
            trial_metadata.get("full_reset_rerun", False)
        ),
        "initial_state_source": trial_metadata.get("initial_state_source"),
        "checkpoint_used": bool(trial_metadata.get("checkpoint_used", True)),
    }


def run_v12_grasp_perturbation_audit(
    sources: Sequence[V12GraspRobustnessSource | Mapping[str, Any]],
    output_path: str | Path,
    *,
    workers: int,
    seed: int = DEFAULT_SEED,
    local_perturbations: int = LOCAL_PERTURBATION_COUNT,
    best_perturbations: int = BEST_PERTURBATION_COUNT,
    maximum_nominal_grasps: int = MAX_NOMINAL_GRASPS,
    runner: CandidateRunner | None = None,
) -> dict[str, Any]:
    """Audit verified grasps with deterministic full-reset trials.

    Legacy contact-point experiments retain their five-grasp ceiling.  The
    registered schema-v13 downsize campaign may explicitly raise the ceiling
    so every published per-edge grasp receives its required local-16 audit.
    """

    worker_count = _positive_int(workers, "workers")
    local_count = _positive_int(local_perturbations, "local_perturbations")
    best_count = _positive_int(best_perturbations, "best_perturbations")
    source_limit = _positive_int(
        maximum_nominal_grasps, "maximum_nominal_grasps"
    )
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    materialized = tuple(
        V12GraspRobustnessSource.from_record(value, index)
        for index, value in enumerate(sources)
    )
    if not materialized:
        raise ValueError("at least one verified schema-v12 grasp is required")
    if len({source.candidate_id for source in materialized}) != len(materialized):
        raise ValueError("robustness source candidate IDs must be unique")
    if len({source.config["experiment_id"] for source in materialized}) != 1:
        raise ValueError("robustness sources must belong to one experiment")
    explicit_best = [source for source in materialized if source.best]
    if len(explicit_best) > 1:
        raise ValueError("exactly zero or one robustness source may be marked best")
    ordered = tuple(
        sorted(materialized, key=lambda source: (source.discovery_index, source.candidate_id))
    )
    definition = _v12_definition(ordered[0].config)
    downsize = definition.scaled_contact_downsize_campaign
    registered_limit = (
        MAX_NOMINAL_GRASPS
        if downsize is None
        else downsize.selected_grasp_count
    )
    if source_limit > registered_limit:
        raise ValueError(
            "maximum_nominal_grasps exceeds the registered publication capacity"
        )
    if len(materialized) > source_limit:
        if source_limit == MAX_NOMINAL_GRASPS:
            raise ValueError(
                "at most five verified schema-v12 grasps may be audited"
            )
        raise ValueError(
            "verified grasp count exceeds maximum_nominal_grasps"
        )
    validation_labels = definition.actual_contact_grasp_pose_campaign.validation_labels

    local_jobs: list[tuple[int, dict[str, Any]]] = []
    local_job_metadata: dict[int, dict[str, Any]] = {}
    local_seeds: dict[str, int] = {}
    for source_index, source in enumerate(ordered):
        local_seed = _derived_seed(seed, source.candidate_id, 12_016)
        local_seeds[source.candidate_id] = local_seed
        configs = generate_v12_grasp_perturbation_configs(
            source.config,
            count=local_count,
            seed=local_seed,
            source_candidate_id=source.candidate_id,
            family=LOCAL_FAMILY,
        )
        for trial, config in enumerate(configs):
            run_id = (source_index + 1) * 1_000_000 + trial
            local_jobs.append((run_id, config))
            local_job_metadata[run_id] = {
                "trial": trial,
                "source_candidate_id": source.candidate_id,
                "point_plan_id": source.point_plan_id,
                "family": LOCAL_FAMILY,
                "seed": local_seed,
            }

    execute = _run_candidates if runner is None else runner

    def execute_complete_job_set(
        jobs: list[tuple[int, dict[str, Any]]],
        metadata: Mapping[int, Mapping[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        raw_results = list(execute(jobs, worker_count))
        expected_ids = {identifier for identifier, _ in jobs}
        observed_ids = [
            int(result.get("candidate_id", -1)) for result in raw_results
        ]
        if (
            len(raw_results) != len(jobs)
            or len(set(observed_ids)) != len(observed_ids)
            or set(observed_ids) != expected_ids
        ):
            raise RuntimeError(
                "robustness runner did not preserve the complete job set"
            )
        return {
            int(result["candidate_id"]): _trial_record(
                result, metadata[int(result["candidate_id"])]
            )
            for result in raw_results
        }

    # Phase one is complete before best selection.  This makes the extra
    # 50-run budget follow measured local robustness instead of a catalog alias.
    local_records = execute_complete_job_set(local_jobs, local_job_metadata)
    local_trials_by_source: dict[str, list[dict[str, Any]]] = {}
    local_passes_by_source: dict[str, int] = {}
    for source in ordered:
        trials = sorted(
            (
                record
                for record in local_records.values()
                if record["source_candidate_id"] == source.candidate_id
            ),
            key=lambda record: record["trial"],
        )
        local_trials_by_source[source.candidate_id] = trials
        local_passes_by_source[source.candidate_id] = sum(
            record["grasp_passed"] for record in trials
        )

    best = min(
        ordered,
        key=lambda source: (
            -local_passes_by_source[source.candidate_id],
            not source.best,
            source.discovery_index,
            source.candidate_id,
        ),
    )
    best_seed = _derived_seed(seed, best.candidate_id, 12_050)
    best_jobs: list[tuple[int, dict[str, Any]]] = []
    best_job_metadata: dict[int, dict[str, Any]] = {}
    for trial, config in enumerate(
        generate_v12_grasp_perturbation_configs(
            best.config,
            count=best_count,
            seed=best_seed,
            source_candidate_id=best.candidate_id,
            family=BEST_FAMILY,
        )
    ):
        run_id = 900_000_000 + trial
        best_jobs.append((run_id, config))
        best_job_metadata[run_id] = {
            "trial": trial,
            "source_candidate_id": best.candidate_id,
            "point_plan_id": best.point_plan_id,
            "family": BEST_FAMILY,
            "seed": best_seed,
        }

    best_records = execute_complete_job_set(best_jobs, best_job_metadata)
    records = {**local_records, **best_records}

    per_source: list[dict[str, Any]] = []
    for source in ordered:
        trials = local_trials_by_source[source.candidate_id]
        per_source.append(
            {
                **_source_record(source),
                "perturbation_seed": local_seeds[source.candidate_id],
                "perturbation_count": len(trials),
                "grasp_passes": local_passes_by_source[source.candidate_id],
                "trials": trials,
            }
        )
    best_trials = sorted(
        (
            record
            for record in records.values()
            if record["family"] == BEST_FAMILY
        ),
        key=lambda record: record["trial"],
    )
    best_passes = sum(record["grasp_passed"] for record in best_trials)
    registered_budget_complete = bool(
        local_count == LOCAL_PERTURBATION_COUNT
        and best_count == BEST_PERTURBATION_COUNT
        and all(
            record["perturbation_count"] == LOCAL_PERTURBATION_COUNT
            for record in per_source
        )
        and len(best_trials) == BEST_PERTURBATION_COUNT
    )
    robust_passed = bool(
        registered_budget_complete and best_passes >= REQUIRED_BEST_PASSES
    )
    report = {
        "v12_grasp_robustness_report_schema_version": (
            ROBUSTNESS_REPORT_SCHEMA_VERSION
        ),
        "complete": True,
        "experiment_id": best.config["experiment_id"],
        "audit_scope": "grasp_only_contact_point_hard_checks",
        "seed": int(seed),
        "workers": worker_count,
        "selected_grasp_count": len(ordered),
        "maximum_nominal_grasps": source_limit,
        "selected_candidate_ids": [source.candidate_id for source in ordered],
        "local_perturbations_per_grasp": local_count,
        "best_perturbation_count": best_count,
        "registered_budget_complete": registered_budget_complete,
        "total_perturbation_count": len(records),
        "total_grasp_passes": sum(
            record["grasp_passed"] for record in records.values()
        ),
        "per_grasp": per_source,
        "best_selection": {
            "candidate_id": best.candidate_id,
            "local_perturbation_count": local_count,
            "local_grasp_passes": local_passes_by_source[best.candidate_id],
            "basis": (
                "maximum_local_grasp_passes_then_original_best_then_"
                "discovery_index_then_candidate_id"
            ),
            "tie_break_order": [
                "original_best_descending",
                "discovery_index_ascending",
                "candidate_id_ascending",
            ],
            "selection_completed_before_best_50": True,
        },
        "best_robustness": {
            **_source_record(best),
            "perturbation_seed": best_seed,
            "perturbation_count": len(best_trials),
            "grasp_passes": best_passes,
            "required_grasp_passes": REQUIRED_BEST_PASSES,
            "robust_passed": robust_passed,
            "trials": best_trials,
        },
        "robust_passed": robust_passed,
        "validation_label": (
            validation_labels["robust"]
            if robust_passed and validation_labels is not None
            else None
        ),
        "stop_reason": (
            "robust_grasp_passed"
            if robust_passed
            else (
                "registered_perturbation_budget_incomplete"
                if not registered_budget_complete
                else "best_grasp_pass_count_below_45_of_50"
            )
        ),
        "acceptance_contract": {
            "required_stage_status": "grasp_success",
            "required_v12_contact_point_checks": list(
                V12_CONTACT_POINT_HARD_CHECKS
            ),
            "summary_passed_required": False,
            "manipulation_success_required": False,
            "full_success_required": False,
        },
        "execution_contract": {
            "full_reset_rerun": True,
            "initial_state_source": "configured_no_contact_reset",
            "checkpoint_used": False,
            "worker_order_affects_result": False,
        },
        "provenance": {
            "implementation_sha256": file_sha256(Path(__file__)),
            "model_sha256": file_sha256(REPO_ROOT / "xhand_left.xml"),
            "uv_lock_sha256": file_sha256(REPO_ROOT / "uv.lock"),
        },
    }
    safe_report = _json_safe(report)
    write_json(Path(output_path).expanduser().resolve(), safe_report)
    return safe_report


__all__ = [
    "BEST_FAMILY",
    "BEST_PERTURBATION_COUNT",
    "LOCAL_FAMILY",
    "LOCAL_PERTURBATION_COUNT",
    "MAX_NOMINAL_GRASPS",
    "REQUIRED_BEST_PASSES",
    "ROBUSTNESS_REPORT_SCHEMA_VERSION",
    "V12_CONTACT_POINT_HARD_CHECKS",
    "V12GraspRobustnessSource",
    "discover_v12_grasp_robustness_sources",
    "generate_v12_grasp_perturbation_configs",
    "run_v12_grasp_perturbation_audit",
    "v12_grasp_failure_reasons",
    "v12_grasp_hard_success",
]
