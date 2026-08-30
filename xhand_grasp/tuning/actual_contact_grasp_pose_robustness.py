"""Schema-v9 full-reset pose/material robustness campaign.

Only nominal trajectories that passed the complete manipulation contract are
eligible for the per-trajectory 16-run audit.  The catalog's ``best_first``
trajectory receives an additional 50 deterministic perturbations and is
called robust only when at least 45 pass.  A failed nominal may be evaluated
as a diagnostic source, but can never acquire a robustness claim from passing
perturbations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..actual_contact_capability import (
    LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID,
    resolve_actual_contact_definition,
)
from ..artifacts import REPO_ROOT, file_sha256, json_text, write_json
from ..actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
)
from ..config import validate_config
from ..experiment import resolve_experiment
from ..grasp_pose import canonical_sha256, controller_id, grasp_pose_id
from ..search import _run_candidates
from .contact_constrained_planner import (
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
)


EXPERIMENT_ID = LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID
ROBUSTNESS_REPORT_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
LOCAL_PERTURBATION_COUNT = 16
BEST_PERTURBATION_COUNT = 50
MAX_NOMINAL_TRAJECTORIES = 5
REQUIRED_BEST_PASSES = 45
DEFAULT_SEARCH_ROOT = Path(
    "artifacts/"
    "left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift/"
    "tune/campaign"
)
DEFAULT_OUTPUT = Path(
    "artifacts/"
    "left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift/"
    "robustness/perturbation_report.json"
)


CandidateRunner = Callable[
    [list[tuple[int, dict[str, Any]]], int], list[dict[str, Any]]
]
CandidateDiscoverer = Callable[
    [Sequence[str | Path]], tuple["V9RobustnessSource", ...]
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


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    _positive_int(count, "count")
    _positive_int(dimensions, "dimensions")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    values = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        values[:, dimension] = (
            rng.permutation(count) + rng.random(count)
        ) / count
    return values


def _scale(unit: float, bounds: Sequence[float]) -> float:
    return float(float(bounds[0]) + float(unit) * (float(bounds[1]) - float(bounds[0])))


def _derived_seed(seed: int, candidate_id: str, family: int) -> int:
    digest = hashlib.sha256(str(candidate_id).encode("utf-8")).digest()
    identifier = int.from_bytes(digest[:4], "little")
    state = np.random.SeedSequence(
        [int(seed), identifier, int(family)]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


def _hard_full_success(summary: Mapping[str, Any]) -> bool:
    stage = summary.get("stage_status")
    return bool(
        summary.get("passed", False)
        and isinstance(stage, Mapping)
        and stage.get("grasp_success", False)
        and stage.get("manipulation_success", False)
        and stage.get("full_success", False)
    )


def _source_grasp_pose_id(config: Mapping[str, Any]) -> str:
    if int(config.get("schema_version", 0)) == 14:
        return str(config["grasp_pose_id"])
    return grasp_pose_id(config)


def _source_controller_id(config: Mapping[str, Any]) -> str:
    if int(config.get("schema_version", 0)) == 14:
        return str(config["controller_id"])
    return controller_id(config)


@dataclass(frozen=True, slots=True)
class V9RobustnessSource:
    """One authenticated nominal or diagnostic schema-v9 trajectory."""

    candidate_id: str
    config: dict[str, Any]
    summary: dict[str, Any]
    config_path: Path | None = None
    result_path: Path | None = None
    trace_path: Path | None = None
    discovery_index: int = 0
    best_first: bool = False

    def __post_init__(self) -> None:
        config = copy.deepcopy(dict(self.config))
        summary = copy.deepcopy(dict(self.summary))
        resolve_actual_contact_definition(
            config, context="actual-contact robustness source"
        )
        if config.get("run_context") is not None:
            raise ValueError(
                "robustness sources must be canonical nominal configs, not prior trials"
            )
        validate_config(config)
        if not isinstance(self.discovery_index, int) or self.discovery_index < 0:
            raise ValueError("discovery_index must be a non-negative integer")
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "summary", summary)
        for name in ("config_path", "result_path", "trace_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())

    @property
    def nominal_full_success(self) -> bool:
        return _hard_full_success(self.summary)

    @property
    def experiment_id(self) -> str:
        return str(self.config["experiment_id"])

    @property
    def identity(self) -> tuple[str, str, str]:
        return (
            self.candidate_id,
            _source_grasp_pose_id(self.config),
            _source_controller_id(self.config),
        )

    @property
    def rank(self) -> tuple[Any, ...]:
        metrics = self.summary.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        motion = metrics.get("motion_smoothness", {})
        if not isinstance(motion, Mapping):
            motion = {}
        actual = metrics.get("actual_grasp_pose", {})
        if not isinstance(actual, Mapping):
            actual = {}
        actual_metrics = actual.get("metrics", {})
        if not isinstance(actual_metrics, Mapping):
            actual_metrics = {}
        thumb = _finite(actual_metrics.get("thumb_actual_median_rad"), math.inf)
        failed = self.summary.get("failed_checks", ())
        return (
            not self.nominal_full_success,
            abs(thumb - 1.50) if math.isfinite(thumb) else math.inf,
            len(failed) if isinstance(failed, Sequence) else 10**6,
            self.discovery_index,
            self.candidate_id,
        )


def _finite(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def generate_v9_perturbation_configs(
    config: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    source_candidate_id: str | int,
    family: str,
) -> list[dict[str, Any]]:
    """Generate the fixed eight-dimensional v9 perturbation envelope."""

    sample_count = _positive_int(count, "count")
    if family not in {"per_full_success_local_16", "best_first_pose_material_50"}:
        raise ValueError("unknown schema-v9 perturbation family")
    base = copy.deepcopy(dict(config))
    if base.get("run_context") is not None:
        raise ValueError("v9 robustness requires a canonical full-reset config")
    validate_config(base)
    definition = resolve_actual_contact_definition(
        base, context="actual-contact robustness"
    )
    parameters = definition.robustness
    matrix = _latin_hypercube(sample_count, 8, seed)
    base_xy = np.asarray(base["cube"]["center_xy_m"], dtype=np.float64)
    base_rpy = np.asarray(base["cube"].get("rpy_deg", (0.0, 0.0, 0.0)), dtype=np.float64)
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
    elif int(base.get("schema_version", 0)) == 14:
        source_binding = {
            "source_config_sha256": canonical_sha256(base),
            "source_object_config_id": base.get("object_config_id"),
            "source_grasp_pose_id": base.get("grasp_pose_id"),
            "source_grasp_object_pair_id": base.get("grasp_object_pair_id"),
            "source_controller_id": base.get("controller_id"),
            "source_cube": copy.deepcopy(base["cube"]),
        }
        source_binding["source_candidate_binding_sha256"] = canonical_sha256(
            {
                "source_candidate_id": str(source_candidate_id),
                **source_binding,
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
            [_scale(row[3 + axis], parameters.rpy_delta_deg) for axis in range(3)],
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
        metadata["robustness_trial"] = {
            "family": family,
            "seed": int(seed),
            "trial": int(trial),
            "source_candidate_id": str(source_candidate_id),
            **source_binding,
            "resolved_perturbations": resolved,
            "full_reset_rerun": True,
            "initial_state_source": "configured_no_contact_reset",
            "checkpoint_used": False,
        }
        case["candidate_metadata"] = metadata
        if int(case.get("schema_version", 0)) == 14:
            case["object_config_id"] = v14_object_config_id(case)
            case["grasp_pose_id"] = v14_grasp_pose_id(case)
            case["grasp_object_pair_id"] = v14_grasp_object_pair_id(case)
            case["controller_id"] = canonical_sha256(
                {
                    "schema_version": 1,
                    "grasp_object_pair_id": case["grasp_object_pair_id"],
                    "plan_id": case["manipulation_plan"]["plan_id"],
                    "target_id": case["contact_force_targets_n"]["target_id"],
                    "feedback_id": case["contact_feedback"]["feedback_id"],
                }
            )
        validate_config(case)
        cases.append(case)
    return cases


def _safe_catalog_member(root: Path, value: Any, label: str) -> Path:
    raw = Path(str(value))
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"catalog {label} is missing or escapes its directory")
    return path


def _source_from_result(
    result_path: Path,
    *,
    discovery_index: int,
    best_first: bool,
    config_path: Path | None = None,
    trace_path: Path | None = None,
) -> V9RobustnessSource | None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    production_result = bool(
        int(result.get("candidate_result_schema_version", 0)) == 1
        or int(result.get("contact_preserving_candidate_schema_version", 0)) == 1
        or int(
            result.get(
                "actual_contact_manipulation_candidate_schema_version", 0
            )
        )
        == 1
    )
    if production_result:
        authenticate_candidate_result_semantic_sha256(
            result, source=result_path
        )
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        return None
    resolved_config = config_path or result_path.parent / "resolved_config.json"
    if not resolved_config.is_file():
        return None
    config = json.loads(resolved_config.read_text(encoding="utf-8"))
    try:
        resolve_actual_contact_definition(
            config, context="discovered actual-contact robustness source"
        )
    except ValueError:
        return None
    # Directory discovery can encounter earlier perturbation runs or Viewer
    # overrides.  They are useful diagnostics, but are not nominal evidence
    # and must never recursively seed another robustness campaign.
    if config.get("run_context") is not None:
        return None
    artifacts = result.get("artifacts", {})
    hashes = artifacts.get("sha256", {}) if isinstance(artifacts, Mapping) else {}
    if isinstance(hashes, Mapping) and hashes.get("resolved_config") is not None:
        if hashes["resolved_config"] != file_sha256(resolved_config):
            raise ValueError(f"source config hash mismatch: {resolved_config}")
    candidate_sha = result.get("candidate_sha256")
    if candidate_sha is not None and candidate_sha != canonical_sha256(config):
        raise ValueError(f"source semantic config hash mismatch: {resolved_config}")
    for name, expected in (
        ("grasp_pose_id", grasp_pose_id(config)),
        ("controller_id", controller_id(config)),
    ):
        if result.get(name) is not None and result[name] != expected:
            raise ValueError(f"source {name} mismatch: {result_path}")
    resolved_trace = trace_path
    if resolved_trace is None:
        adjacent = result_path.parent / "trace.npz"
        resolved_trace = adjacent if adjacent.is_file() else None
    if resolved_trace is not None and isinstance(hashes, Mapping) and hashes.get("trace") is not None:
        if hashes["trace"] != file_sha256(resolved_trace):
            raise ValueError(f"source trace hash mismatch: {resolved_trace}")
    return V9RobustnessSource(
        candidate_id=str(result.get("candidate_id", discovery_index)),
        config=config,
        summary=copy.deepcopy(dict(summary)),
        config_path=resolved_config,
        result_path=result_path,
        trace_path=resolved_trace,
        discovery_index=discovery_index,
        best_first=best_first,
    )


def discover_v9_robustness_sources(
    search_roots: Sequence[str | Path],
) -> tuple[V9RobustnessSource, ...]:
    """Discover authenticated sources from v9 catalogs or result directories."""

    discovered: list[V9RobustnessSource] = []
    index = 0
    for value in search_roots:
        root = Path(value).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        catalog_paths: list[Path] = []
        result_paths: list[Path] = []
        if root.is_file():
            payload = json.loads(root.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping) and isinstance(payload.get("trajectories"), list):
                catalog_paths.append(root)
            elif root.name == "result.json":
                result_paths.append(root)
            else:
                raise ValueError(f"unsupported robustness source file: {root}")
        else:
            catalog_paths.extend(sorted(root.rglob("catalog.json")))
            result_paths.extend(sorted(root.rglob("result.json")))

        catalog_result_paths: set[Path] = set()
        for catalog_path in catalog_paths:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            if not isinstance(catalog.get("experiment_id"), str):
                continue
            catalog_root = catalog_path.parent.resolve()
            best_target = (
                catalog.get("aliases", {}).get("best_first")
                if isinstance(catalog.get("aliases"), Mapping)
                else None
            )
            for entry in catalog.get("trajectories", ()):
                if not isinstance(entry, Mapping):
                    continue
                artifacts = entry.get("artifacts", {})
                if not isinstance(artifacts, Mapping):
                    continue
                result_path = _safe_catalog_member(catalog_root, artifacts.get("result"), "result")
                config_path = _safe_catalog_member(
                    catalog_root, artifacts.get("resolved_config"), "resolved_config"
                )
                trace_path = _safe_catalog_member(catalog_root, artifacts.get("trace"), "trace")
                member_hashes = artifacts.get("sha256")
                if not isinstance(member_hashes, Mapping):
                    raise ValueError("schema-v9 catalog entry has no artifact hashes")
                for label, path in (
                    ("resolved_config", config_path),
                    ("result", result_path),
                    ("trace", trace_path),
                ):
                    expected = member_hashes.get(label)
                    if not isinstance(expected, str) or expected != file_sha256(path):
                        raise ValueError(
                            f"catalog {label} SHA-256 mismatch: {path}"
                        )
                aliases = entry.get("aliases", ())
                is_best = bool(
                    best_target == entry.get("trajectory_id")
                    or (isinstance(aliases, Sequence) and "best_first" in aliases)
                )
                source = _source_from_result(
                    result_path,
                    discovery_index=index,
                    best_first=is_best,
                    config_path=config_path,
                    trace_path=trace_path,
                )
                if source is not None:
                    discovered.append(source)
                    index += 1
                    catalog_result_paths.add(result_path.resolve())
        for result_path in result_paths:
            if result_path.resolve() in catalog_result_paths:
                continue
            source = _source_from_result(
                result_path,
                discovery_index=index,
                best_first=False,
            )
            if source is not None:
                discovered.append(source)
                index += 1

    best_by_identity: dict[tuple[str, str, str], V9RobustnessSource] = {}
    for source in discovered:
        previous = best_by_identity.get(source.identity)
        if previous is None or (not previous.best_first and source.best_first) or (
            previous.best_first == source.best_first and source.rank < previous.rank
        ):
            best_by_identity[source.identity] = source
    return tuple(sorted(best_by_identity.values(), key=lambda source: source.rank))


def _source_record(source: V9RobustnessSource) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "config": str(source.config_path) if source.config_path else None,
        "result": str(source.result_path) if source.result_path else None,
        "trace": str(source.trace_path) if source.trace_path else None,
        "sha256": {
            "config": file_sha256(source.config_path) if source.config_path else None,
            "result": file_sha256(source.result_path) if source.result_path else None,
            "trace": file_sha256(source.trace_path) if source.trace_path else None,
        },
    }
    return {
        "candidate_id": source.candidate_id,
        "candidate_sha256": canonical_sha256(source.config),
        "grasp_pose_id": _source_grasp_pose_id(source.config),
        "controller_id": _source_controller_id(source.config),
        "best_first": source.best_first,
        "nominal_full_success": source.nominal_full_success,
        "nominal_failed_checks": copy.deepcopy(source.summary.get("failed_checks", [])),
        "artifacts": artifacts,
    }


def _trial_record(result: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    summary = _json_safe(result.get("summary", {}))
    config = copy.deepcopy(dict(result.get("config", {})))
    robustness = config.get("candidate_metadata", {}).get("robustness_trial", {})
    return {
        "trial": int(metadata["trial"]),
        "run_candidate_id": int(result["candidate_id"]),
        "source_candidate_id": str(metadata["source_candidate_id"]),
        "family": str(metadata["family"]),
        "seed": int(metadata["seed"]),
        "passed": _hard_full_success(summary),
        "config_sha256": canonical_sha256(config),
        "failed_checks": copy.deepcopy(summary.get("failed_checks", [])),
        "checks": copy.deepcopy(summary.get("checks", {})),
        "stage_status": copy.deepcopy(summary.get("stage_status", {})),
        "metrics": copy.deepcopy(summary.get("metrics", {})),
        "error": summary.get("error"),
        "cube": copy.deepcopy(config.get("cube", {})),
        "resolved_perturbations": copy.deepcopy(
            robustness.get("resolved_perturbations", {})
        ),
        "full_reset_rerun": bool(robustness.get("full_reset_rerun", False)),
        "initial_state_source": robustness.get("initial_state_source"),
        "checkpoint_used": bool(robustness.get("checkpoint_used", True)),
    }


def run_v9_robustness_campaign(
    search_roots: Sequence[str | Path],
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    workers: int,
    seed: int = DEFAULT_SEED,
    local_perturbations: int = LOCAL_PERTURBATION_COUNT,
    best_perturbations: int = BEST_PERTURBATION_COUNT,
    max_nominal_trajectories: int = MAX_NOMINAL_TRAJECTORIES,
    candidate_discoverer: CandidateDiscoverer | None = None,
    runner: CandidateRunner | None = None,
    best_selection: str = "best_first",
) -> dict[str, Any]:
    """Run per-full-success and best-first robustness from no-contact reset."""

    worker_count = _positive_int(workers, "workers")
    local_count = _positive_int(local_perturbations, "local_perturbations")
    best_count = _positive_int(best_perturbations, "best_perturbations")
    source_limit = _positive_int(max_nominal_trajectories, "max_nominal_trajectories")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if best_selection not in {"best_first", "local_perturbation_passes"}:
        raise ValueError("unknown robustness best-selection policy")
    discoverer = discover_v9_robustness_sources if candidate_discoverer is None else candidate_discoverer
    execute = _run_candidates if runner is None else runner
    roots = tuple(Path(value).expanduser().resolve() for value in search_roots)
    discovered = tuple(sorted(discoverer(roots), key=lambda source: source.rank))
    if not discovered:
        raise ValueError("no authenticated schema-v9 trajectory candidates were found")
    experiment_ids = {source.experiment_id for source in discovered}
    if len(experiment_ids) != 1:
        raise ValueError("robustness sources must belong to one experiment")
    experiment_id = experiment_ids.pop()
    definition = resolve_actual_contact_definition(
        discovered[0].config, context="actual-contact robustness campaign"
    )
    downsize = definition.scaled_contact_downsize_campaign
    registered_source_limit = (
        MAX_NOMINAL_TRAJECTORIES
        if downsize is None
        else downsize.selected_grasp_count
    )
    if source_limit > registered_source_limit:
        if registered_source_limit == MAX_NOMINAL_TRAJECTORIES:
            raise ValueError("max_nominal_trajectories cannot exceed five")
        raise ValueError(
            "max_nominal_trajectories exceeds the registered publication capacity"
        )
    campaign = definition.actual_contact_grasp_pose_campaign
    validation_labels = None if campaign is None else campaign.validation_labels
    successful = tuple(
        source for source in discovered if source.nominal_full_success
    )[:source_limit]
    explicit_best = [source for source in discovered if source.best_first]
    best = explicit_best[0] if explicit_best else (successful[0] if successful else discovered[0])

    jobs: list[tuple[int, dict[str, Any]]] = []
    job_metadata: dict[int, dict[str, Any]] = {}
    local_seeds: dict[str, int] = {}
    for source_index, source in enumerate(successful):
        local_seed = _derived_seed(seed, source.candidate_id, 16)
        local_seeds[source.candidate_id] = local_seed
        configs = generate_v9_perturbation_configs(
            source.config,
            count=local_count,
            seed=local_seed,
            source_candidate_id=source.candidate_id,
            family="per_full_success_local_16",
        )
        for trial, config in enumerate(configs):
            run_id = (source_index + 1) * 1_000_000 + trial
            jobs.append((run_id, config))
            job_metadata[run_id] = {
                "trial": trial,
                "source_candidate_id": source.candidate_id,
                "family": "per_full_success_local_16",
                "seed": local_seed,
            }

    best_seed: int | None = None
    if best_selection == "best_first":
        best_seed = _derived_seed(seed, best.candidate_id, 50)
        best_configs = generate_v9_perturbation_configs(
            best.config,
            count=best_count,
            seed=best_seed,
            source_candidate_id=best.candidate_id,
            family="best_first_pose_material_50",
        )
        for trial, config in enumerate(best_configs):
            run_id = 900_000_000 + trial
            jobs.append((run_id, config))
            job_metadata[run_id] = {
                "trial": trial,
                "source_candidate_id": best.candidate_id,
                "family": "best_first_pose_material_50",
                "seed": best_seed,
            }

    raw_results = list(execute(jobs, worker_count))
    expected_ids = {identifier for identifier, _ in jobs}
    observed_ids = {int(result.get("candidate_id", -1)) for result in raw_results}
    if observed_ids != expected_ids or len(raw_results) != len(jobs):
        raise RuntimeError("robustness runner did not preserve the complete job set")
    records = {
        int(result["candidate_id"]): _trial_record(
            result, job_metadata[int(result["candidate_id"])]
        )
        for result in raw_results
    }

    if best_selection == "local_perturbation_passes":
        if not successful:
            raise RuntimeError("local robustness selection requires a nominal success")
        local_passes = {
            source.candidate_id: sum(
                record["passed"]
                for record in records.values()
                if record["family"] == "per_full_success_local_16"
                and record["source_candidate_id"] == source.candidate_id
            )
            for source in successful
        }
        best = min(
            successful,
            key=lambda source: (
                -int(local_passes[source.candidate_id]),
                source.rank,
            ),
        )
        best_seed = _derived_seed(seed, best.candidate_id, 50)
        best_jobs: list[tuple[int, dict[str, Any]]] = []
        best_metadata: dict[int, dict[str, Any]] = {}
        best_configs = generate_v9_perturbation_configs(
            best.config,
            count=best_count,
            seed=best_seed,
            source_candidate_id=best.candidate_id,
            family="best_first_pose_material_50",
        )
        for trial, config in enumerate(best_configs):
            run_id = 900_000_000 + trial
            best_jobs.append((run_id, config))
            best_metadata[run_id] = {
                "trial": trial,
                "source_candidate_id": best.candidate_id,
                "family": "best_first_pose_material_50",
                "seed": best_seed,
            }
        best_raw_results = list(execute(best_jobs, worker_count))
        expected_best_ids = {identifier for identifier, _ in best_jobs}
        observed_best_ids = {
            int(result.get("candidate_id", -1)) for result in best_raw_results
        }
        if observed_best_ids != expected_best_ids or len(best_raw_results) != len(
            best_jobs
        ):
            raise RuntimeError("robustness runner did not preserve best job set")
        records.update(
            {
                int(result["candidate_id"]): _trial_record(
                    result, best_metadata[int(result["candidate_id"])]
                )
                for result in best_raw_results
            }
        )
    assert best_seed is not None

    per_nominal: list[dict[str, Any]] = []
    for source in successful:
        trials = sorted(
            (
                record
                for record in records.values()
                if record["family"] == "per_full_success_local_16"
                and record["source_candidate_id"] == source.candidate_id
            ),
            key=lambda record: record["trial"],
        )
        per_nominal.append(
            {
                **_source_record(source),
                "perturbation_seed": local_seeds[source.candidate_id],
                "perturbation_count": len(trials),
                "perturbation_passes": sum(record["passed"] for record in trials),
                "trials": trials,
            }
        )
    best_trials = sorted(
        (
            record
            for record in records.values()
            if record["family"] == "best_first_pose_material_50"
        ),
        key=lambda record: record["trial"],
    )
    best_source = _source_record(best)
    best_passes = sum(record["passed"] for record in best_trials)
    registered_budget_complete = bool(
        local_count == LOCAL_PERTURBATION_COUNT
        and best_count == BEST_PERTURBATION_COUNT
        and (
            source_limit == MAX_NOMINAL_TRAJECTORIES
            if downsize is None
            else len(successful)
            == sum(source.nominal_full_success for source in discovered)
        )
        and len(successful) >= 1
        and all(
            record["perturbation_count"] == LOCAL_PERTURBATION_COUNT
            for record in per_nominal
        )
        and len(best_trials) == BEST_PERTURBATION_COUNT
    )
    robust_passed = bool(
        best.nominal_full_success
        and registered_budget_complete
        and best_passes >= REQUIRED_BEST_PASSES
    )
    report = {
        "v9_robustness_report_schema_version": ROBUSTNESS_REPORT_SCHEMA_VERSION,
        "complete": True,
        "experiment_id": experiment_id,
        "seed": int(seed),
        "workers": worker_count,
        "search_roots": [str(value) for value in roots],
        "discovered_candidate_count": len(discovered),
        "nominal_full_success_count": sum(source.nominal_full_success for source in discovered),
        "selected_nominal_count": len(successful),
        "selected_nominal_candidate_ids": [source.candidate_id for source in successful],
        "local_perturbations_per_nominal": local_count,
        "best_perturbation_count": best_count,
        "registered_budget_complete": registered_budget_complete,
        "total_perturbation_count": len(records),
        "total_perturbation_passes": sum(record["passed"] for record in records.values()),
        "per_nominal": per_nominal,
        "best_robustness": {
            **best_source,
            "perturbation_seed": best_seed,
            "perturbation_count": len(best_trials),
            "perturbation_passes": best_passes,
            "required_perturbation_passes": REQUIRED_BEST_PASSES,
            "robust_passed": robust_passed,
            "diagnostic_only_due_to_nominal_failure": not best.nominal_full_success,
            "trials": best_trials,
        },
        "robust_passed": robust_passed,
        "stop_reason": (
            "robust_passed"
            if robust_passed
            else (
                "best_first_nominal_failed_hard_constraints"
                if not best.nominal_full_success
                else (
                    "registered_perturbation_budget_incomplete"
                    if not registered_budget_complete
                    else "perturbation_pass_count_below_45_of_50"
                )
            )
        ),
        "execution_contract": {
            "full_reset_rerun": True,
            "initial_state_source": "configured_no_contact_reset",
            "checkpoint_used": False,
            "worker_order_affects_result": False,
        },
        "best_selection_policy": best_selection,
        "provenance": {
            "implementation_sha256": file_sha256(Path(__file__)),
            "model_sha256": file_sha256(REPO_ROOT / "xhand_left.xml"),
            "uv_lock_sha256": file_sha256(REPO_ROOT / "uv.lock"),
        },
    }
    if robust_passed and validation_labels is not None:
        report["validation_label"] = validation_labels["robust"]
    write_json(Path(output_path).expanduser().resolve(), _json_safe(report))
    return _json_safe(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run schema-v9 actual-contact full-reset robustness"
    )
    parser.add_argument(
        "search_roots",
        nargs="*",
        default=[str(DEFAULT_SEARCH_ROOT)],
        help="v9 manipulation catalog, result.json, or search directory",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--local-perturbations", type=int, default=LOCAL_PERTURBATION_COUNT)
    parser.add_argument("--best-perturbations", type=int, default=BEST_PERTURBATION_COUNT)
    parser.add_argument(
        "--max-nominal-trajectories", type=int, default=MAX_NOMINAL_TRAJECTORIES
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_v9_robustness_campaign(
        args.search_roots,
        args.output,
        workers=args.workers,
        seed=args.seed,
        local_perturbations=args.local_perturbations,
        best_perturbations=args.best_perturbations,
        max_nominal_trajectories=args.max_nominal_trajectories,
    )
    print(
        json_text(
            {
                key: value
                for key, value in report.items()
                if key not in {"per_nominal", "best_robustness"}
            }
        )
    )
    return 0 if report["robust_passed"] else 2


__all__ = [
    "BEST_PERTURBATION_COUNT",
    "DEFAULT_OUTPUT",
    "DEFAULT_SEARCH_ROOT",
    "LOCAL_PERTURBATION_COUNT",
    "MAX_NOMINAL_TRAJECTORIES",
    "REQUIRED_BEST_PASSES",
    "ROBUSTNESS_REPORT_SCHEMA_VERSION",
    "V9RobustnessSource",
    "build_parser",
    "discover_v9_robustness_sources",
    "generate_v9_perturbation_configs",
    "main",
    "run_v9_robustness_campaign",
]
