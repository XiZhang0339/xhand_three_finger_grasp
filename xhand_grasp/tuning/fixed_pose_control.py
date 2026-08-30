"""Control-only refinement for a fixed schema-v7 hand/object scene.

The tuner is deliberately narrower than the schema-v7 campaign tuner.  It
copies a resolved candidate and permits changes only inside ``control`` (plus
provenance metadata).  In particular, the free cube initial pose, material,
hand root pose, contact topology, protocol and acceptance thresholds are
byte-for-byte equivalent to the source candidate.

The default seed is the strongest pose-preserving control found while
diagnosing the 67 mm schema-v7 near miss.  The implementation is still useful
for later experiments: callers can supply another centre vector or construct
candidates directly with :func:`materialize_control_candidate`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import file_sha256, json_text, write_json
from ..config import ACTIVE_ACTUATORS, load_config, validate_config
from ..experiment import resolve_experiment
from .high_thumb_variable_size import (
    THUMB_BEND_ACTUATOR,
    execute_high_thumb_candidate_job,
)
from .pose_preserving_grasp import (
    acquisition_succeeded,
    pose_preservation_succeeded,
)
from .pose_preserving_seed_campaign import canonical_sha256
from .pose_preserving_seed_dynamic import CLOSE_GROUP_ACTUATORS, CLOSE_GROUP_ORDER


CAMPAIGN_KIND = "fixed_pose_control_stability_refinement"
CAMPAIGN_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
DEFAULT_BASE_CONFIG = Path(
    "artifacts/left_opposed_face_palm_down_high_thumb_variable_size_pose_preserving/"
    "trajectory_catalog/diagnostics/best_near_miss/resolved_config.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/left_opposed_face_palm_down_high_thumb_variable_size_pose_preserving/"
    "fixed_pose_control_refine"
)

THUMB_ROTA1 = "left_hand_thumb_rota_joint1_actuator"
THUMB_ROTA2 = "left_hand_thumb_rota_joint2_actuator"
INDEX_JOINT1 = "left_hand_index_joint1_actuator"
INDEX_JOINT2 = "left_hand_index_joint2_actuator"
MID_JOINT1 = "left_hand_mid_joint1_actuator"
MID_JOINT2 = "left_hand_mid_joint2_actuator"

VARIABLE_ORDER = (
    "start.thumb",
    "start.index",
    "start.mid",
    f"target.{THUMB_ROTA1}",
    f"target.{THUMB_ROTA2}",
    f"target.{INDEX_JOINT1}",
    f"target.{INDEX_JOINT2}",
    f"target.{MID_JOINT1}",
    f"target.{MID_JOINT2}",
)

# Evidence-backed control centre from a 229-run control-only refinement.  The
# geometry and all other commands are inherited from the resolved candidate.
DEFAULT_CONTROL_CENTER = np.asarray(
    (
        0.150,
        0.130,
        0.040,
        0.3139796524041977,
        0.8842179936059648,
        0.6810364882476693,
        1.267336875450936,
        0.8554395853489825,
        1.0934414295482138,
    ),
    dtype=np.float64,
)
DEFAULT_STAGE1_RADIUS = np.asarray(
    (0.035, 0.035, 0.025, 0.012, 0.015, 0.010, 0.010, 0.010, 0.010),
    dtype=np.float64,
)
DEFAULT_STAGE2_RADIUS = np.asarray(
    (0.008, 0.008, 0.006, 0.0020, 0.0025, 0.0020, 0.0020, 0.0020, 0.0020),
    dtype=np.float64,
)

_MUTABLE_TOP_LEVEL = {"control", "candidate_metadata", "run_context"}
_STAGE1_ID_BASE = 77_000_000_000_000
_STAGE2_ID_BASE = 78_000_000_000_000
_FINAL_ID_BASE = 79_000_000_000_000


@dataclass(frozen=True, slots=True)
class ControlSearchBudget:
    stage1_count: int = 48
    stage2_count: int = 64
    stage2_seed_count: int = 4

    def __post_init__(self) -> None:
        for name in ("stage1_count", "stage2_count", "stage2_seed_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def _finite_vector(values: Sequence[Any], length: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain {length} finite values") from error
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return vector.copy()


def _immutable_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    projection = copy.deepcopy(dict(config))
    for field in _MUTABLE_TOP_LEVEL:
        projection.pop(field, None)
    return projection


def frozen_context_sha256(config: Mapping[str, Any]) -> str:
    """Hash every field that the control-only campaign must preserve."""

    return canonical_sha256(_immutable_projection(config))


def assert_control_only_candidate(
    base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    """Reject scene, pose, material, protocol or threshold changes."""

    if _immutable_projection(base) != _immutable_projection(candidate):
        raise ValueError("control-only candidate changed the frozen experiment context")
    base_delta = base["control"]["manipulation_delta_rad"]
    candidate_delta = candidate["control"]["manipulation_delta_rad"]
    if candidate_delta != base_delta or any(float(value) != 0.0 for value in candidate_delta.values()):
        raise ValueError("control-only acquisition must keep manipulation delta at zero")
    if (
        float(candidate["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR])
        != float(base["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR])
    ):
        raise ValueError("control-only refinement must keep the thumb bend target fixed")


def control_vector(config: Mapping[str, Any]) -> np.ndarray:
    profile = config["control"]["close_profile"]
    starts: list[float] = []
    for group in CLOSE_GROUP_ORDER:
        values = {
            float(profile[name]["start_fraction"])
            for name in CLOSE_GROUP_ACTUATORS[group]
        }
        if len(values) != 1:
            raise ValueError(f"close profile group {group!r} is not synchronized")
        starts.append(values.pop())
    targets = config["control"]["grasp_targets_rad"]
    return np.asarray(
        starts
        + [
            float(targets[name])
            for name in (
                THUMB_ROTA1,
                THUMB_ROTA2,
                INDEX_JOINT1,
                INDEX_JOINT2,
                MID_JOINT1,
                MID_JOINT2,
            )
        ],
        dtype=np.float64,
    )


def _clip_control_vector(base: Mapping[str, Any], vector: np.ndarray) -> np.ndarray:
    result = _finite_vector(vector, len(VARIABLE_ORDER), "control vector")
    definition = resolve_experiment(dict(base))
    bounds = definition.search_bounds.actuator_targets_rad
    target_names = (
        THUMB_ROTA1,
        THUMB_ROTA2,
        INDEX_JOINT1,
        INDEX_JOINT2,
        MID_JOINT1,
        MID_JOINT2,
    )
    for index, name in enumerate(target_names, start=3):
        result[index] = float(np.clip(result[index], *bounds[name]))
    profile = base["control"]["close_profile"]
    for index, group in enumerate(CLOSE_GROUP_ORDER):
        end = min(
            float(profile[name]["end_fraction"])
            for name in CLOSE_GROUP_ACTUATORS[group]
        )
        result[index] = float(np.clip(result[index], 0.0, end - 0.02))
    return result


def materialize_control_candidate(
    base: Mapping[str, Any],
    vector: Sequence[Any],
    *,
    candidate_id: int,
    stage: str,
    parent_candidate_id: int | None = None,
    seed: int | None = None,
    validator=validate_config,
) -> dict[str, Any]:
    """Apply a nine-dimensional control vector without touching geometry."""

    candidate = copy.deepcopy(dict(base))
    candidate.pop("run_context", None)
    values = _clip_control_vector(base, np.asarray(vector, dtype=np.float64))
    profile = candidate["control"]["close_profile"]
    for index, group in enumerate(CLOSE_GROUP_ORDER):
        for name in CLOSE_GROUP_ACTUATORS[group]:
            profile[name]["start_fraction"] = float(values[index])
    targets = candidate["control"]["grasp_targets_rad"]
    for index, name in enumerate(
        (THUMB_ROTA1, THUMB_ROTA2, INDEX_JOINT1, INDEX_JOINT2, MID_JOINT1, MID_JOINT2),
        start=3,
    ):
        targets[name] = float(values[index])
    metadata = copy.deepcopy(dict(candidate.get("candidate_metadata", {})))
    metadata.update(
        {
            "campaign_kind": CAMPAIGN_KIND,
            "candidate_id": int(candidate_id),
            "stage": str(stage),
            "control_only": True,
            "frozen_context_sha256": frozen_context_sha256(base),
            "control_vector_order": list(VARIABLE_ORDER),
            "control_vector": values.tolist(),
        }
    )
    if parent_candidate_id is not None:
        metadata["parent_candidate_id"] = int(parent_candidate_id)
    if seed is not None:
        metadata["seed"] = int(seed)
    candidate["candidate_metadata"] = metadata
    assert_control_only_candidate(base, candidate)
    if validator is not None:
        validator(candidate)
    return candidate


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        values = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        rng.shuffle(values)
        result[:, dimension] = 2.0 * values - 1.0
    return result


def generate_stage1_candidates(
    base: Mapping[str, Any],
    *,
    count: int,
    seed: int = DEFAULT_SEED,
    center: Sequence[Any] = DEFAULT_CONTROL_CENTER,
    radius: Sequence[Any] = DEFAULT_STAGE1_RADIUS,
) -> tuple[dict[str, Any], ...]:
    """Generate an anchor, coordinate probes and deterministic LHS samples."""

    if count <= 0:
        raise ValueError("count must be positive")
    centre = _clip_control_vector(base, _finite_vector(center, len(VARIABLE_ORDER), "center"))
    radii = _finite_vector(radius, len(VARIABLE_ORDER), "radius")
    if np.any(radii < 0.0):
        raise ValueError("radius values must be non-negative")
    vectors: list[np.ndarray] = [centre]
    # Tiny coordinate probes are important near a strict history threshold:
    # target steps are 0.0005 rad and timing steps are 0.002.
    coordinate_step = np.asarray((0.002, 0.002, 0.002, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005))
    for dimension in range(len(VARIABLE_ORDER)):
        for sign in (-1.0, 1.0):
            probe = centre.copy()
            probe[dimension] += sign * coordinate_step[dimension]
            vectors.append(probe)
            if len(vectors) >= count:
                break
        if len(vectors) >= count:
            break
    remaining = count - len(vectors)
    if remaining > 0:
        units = _latin_hypercube(remaining, len(VARIABLE_ORDER), seed)
        vectors.extend(centre + unit * radii for unit in units)
    return tuple(
        materialize_control_candidate(
            base,
            vector,
            candidate_id=_STAGE1_ID_BASE + index,
            stage="control_stage1",
            parent_candidate_id=int(base.get("candidate_metadata", {}).get("candidate_id", -1)),
            seed=None if index == 0 else seed,
        )
        for index, vector in enumerate(vectors)
    )


def generate_stage2_candidates(
    base: Mapping[str, Any],
    ranked_stage1: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed_count: int,
    seed: int = DEFAULT_SEED,
    radius: Sequence[Any] = DEFAULT_STAGE2_RADIUS,
) -> tuple[dict[str, Any], ...]:
    if count <= 0 or seed_count <= 0:
        raise ValueError("count and seed_count must be positive")
    selected = list(ranked_stage1[: min(seed_count, len(ranked_stage1))])
    if not selected:
        return ()
    radii = _finite_vector(radius, len(VARIABLE_ORDER), "stage2 radius")
    units = _latin_hypercube(count, len(VARIABLE_ORDER), seed + 1)
    generated = []
    for index in range(count):
        parent = selected[index % len(selected)]
        centre = control_vector(parent["config"])
        # Preserve each selected parent exactly once before perturbing it.
        vector = centre if index < len(selected) else centre + units[index] * radii
        generated.append(
            materialize_control_candidate(
                base,
                vector,
                candidate_id=_STAGE2_ID_BASE + index,
                stage="control_stage2",
                parent_candidate_id=int(parent["candidate_id"]),
                seed=None if index < len(selected) else seed,
            )
        )
    return tuple(generated)


def _summary_record(candidate: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    pose = metrics["pose_preservation"]
    record_for_checks = {"summary": summary}
    acquisition = acquisition_succeeded(record_for_checks)
    preserved = pose_preservation_succeeded(record_for_checks)
    forces = metrics["verify_peak_target_face_force_n"]
    opposed = float(forces["index"]) + float(forces["mid"])
    force_imbalance = abs(float(forces["thumb"]) - opposed)
    return {
        "candidate_id": int(candidate["candidate_metadata"]["candidate_id"]),
        "stage": str(candidate["candidate_metadata"]["stage"]),
        "candidate_sha256": canonical_sha256(candidate),
        "config": copy.deepcopy(dict(candidate)),
        "acquisition_success": bool(acquisition),
        "pose_preservation_success": bool(preserved),
        "grasp_success": bool(summary["stage_status"]["grasp_success"]),
        "verify_gate_steps": int(metrics["verify_max_consecutive_all_gate_steps"]),
        "translation_max_m": float(pose["max_translation_m"]),
        "orientation_max_deg": float(pose["max_orientation_drift_deg"]),
        "contact_onset_span_steps": int(pose["distal_contact_onset_span_steps"]),
        "first_distal_contact_step": copy.deepcopy(pose["first_distal_contact_step"]),
        "verify_peak_target_face_force_n": copy.deepcopy(forces),
        "opposed_normal_force_imbalance_n": force_imbalance,
        "actuator_saturation_fraction": float(metrics["actuator_saturation_fraction"]),
        "failed_checks": list(summary["failed_checks"]),
    }


def _evaluate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    from ..simulation import run_simulation

    summary = run_simulation(copy.deepcopy(dict(candidate)))
    return _summary_record(candidate, summary)


def run_candidate_batch(
    candidates: Sequence[Mapping[str, Any]], workers: int
) -> tuple[dict[str, Any], ...]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1:
        results = [_evaluate_candidate(candidate) for candidate in candidates]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            results = list(executor.map(_evaluate_candidate, candidates, chunksize=1))
    return tuple(sorted(results, key=lambda value: int(value["candidate_id"])))


def candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    hard_pass = bool(record["acquisition_success"] and record["pose_preservation_success"])
    translation_ratio = float(record["translation_max_m"]) / 0.0005
    orientation_ratio = float(record["orientation_max_deg"]) / 1.0
    return (
        not hard_pass,
        -int(record["verify_gate_steps"]),
        max(translation_ratio, orientation_ratio),
        float(record["opposed_normal_force_imbalance_n"]),
        int(record["contact_onset_span_steps"]),
        float(record["actuator_saturation_fraction"]),
        int(record["candidate_id"]),
    )


def rank_results(results: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    materialized = [copy.deepcopy(dict(value)) for value in results]
    materialized.sort(key=candidate_rank)
    return tuple(materialized)


def _report_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    result.pop("config", None)
    result["control_vector"] = control_vector(record["config"]).tolist()
    return result


def run_fixed_pose_control_search(
    base_config_path: str | Path = DEFAULT_BASE_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    workers: int = 1,
    seed: int = DEFAULT_SEED,
    budget: ControlSearchBudget | None = None,
) -> dict[str, Any]:
    """Run two deterministic control-only stages and persist the best trace."""

    resolved_budget = budget or ControlSearchBudget()
    base_path = Path(base_config_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    base = load_config(base_path)
    validate_config(base)
    if int(base.get("schema_version", 0)) != 7:
        raise ValueError("fixed-pose control refinement currently requires schema v7")
    output.mkdir(parents=True)

    stage1_candidates = generate_stage1_candidates(
        base, count=resolved_budget.stage1_count, seed=seed
    )
    stage1_results = run_candidate_batch(stage1_candidates, workers)
    ranked_stage1 = rank_results(stage1_results)
    write_json(
        output / "stage1_results.json",
        {
            "complete": True,
            "candidate_count": len(stage1_results),
            "results": [_report_record(value) for value in ranked_stage1],
        },
    )

    stage2_candidates = generate_stage2_candidates(
        base,
        ranked_stage1,
        count=resolved_budget.stage2_count,
        seed_count=resolved_budget.stage2_seed_count,
        seed=seed,
    )
    stage2_results = run_candidate_batch(stage2_candidates, workers)
    ranked = rank_results((*stage1_results, *stage2_results))
    best = ranked[0]
    final_config = copy.deepcopy(best["config"])
    final_id = _FINAL_ID_BASE
    final_config["candidate_metadata"].update(
        {
            "candidate_id": final_id,
            "stage": "control_exact",
            "parent_candidate_id": int(best["candidate_id"]),
        }
    )
    assert_control_only_candidate(base, final_config)
    validate_config(final_config)
    final_directory = output / "best_stable" if (
        best["acquisition_success"] and best["pose_preservation_success"]
    ) else output / "best_near_miss"
    final_job = {
        "campaign_kind": CAMPAIGN_KIND,
        "candidate_id": final_id,
        "candidate_sha256": canonical_sha256(final_config),
        "stage": "control_exact",
        "source_family_id": str(final_config["candidate_metadata"].get("source_family_id", "unknown")),
        "source_trajectory_id": str(final_config["candidate_metadata"].get("source_trajectory_id", "unknown")),
        "edge_m": float(final_config["cube"]["edge_m"]),
        "thumb_target_rad": float(final_config["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR]),
        "config": final_config,
        "output_directory": str(final_directory),
        "artifact_directory": str(final_directory.relative_to(output)),
    }
    exact = execute_high_thumb_candidate_job(final_job)
    exact.pop("config", None)

    pass_count = sum(
        bool(value["acquisition_success"] and value["pose_preservation_success"])
        for value in ranked
    )
    alias = "best_stable" if pass_count else "best_near_miss"
    report = {
        "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_kind": CAMPAIGN_KIND,
        "complete": True,
        "base_config": str(base_path),
        "base_config_sha256": file_sha256(base_path),
        "frozen_context_sha256": frozen_context_sha256(base),
        "seed": seed,
        "workers": workers,
        "budget": {
            "stage1_count": resolved_budget.stage1_count,
            "stage2_count": resolved_budget.stage2_count,
            "stage2_seed_count": resolved_budget.stage2_seed_count,
        },
        "candidate_count": len(ranked),
        "hard_pass_count": pass_count,
        "classification": (
            "validated_fixed_pose_control_stable_grasp"
            if pass_count
            else "fixed_pose_control_near_miss"
        ),
        "best_candidate": _report_record(best),
        "exact_result": exact,
        "aliases": {
            alias: {
                "resolved_config": f"{alias}/resolved_config.json",
                "result": f"{alias}/result.json",
                "trace": f"{alias}/trace.npz",
            }
        },
        "results": [_report_record(value) for value in ranked],
    }
    write_json(output / "search_report.json", report)
    trajectory_id = "fixed_pose_control_best"
    write_json(
        output / "catalog.json",
        {
            "trajectory_catalog_schema_version": 1,
            "campaign_kind": CAMPAIGN_KIND,
            "experiment_id": str(base["experiment_id"]),
            "frozen_context_sha256": frozen_context_sha256(base),
            "aliases": {alias: trajectory_id},
            "trajectories": [
                {
                    "trajectory_id": trajectory_id,
                    "label": alias,
                    "aliases": [alias],
                    "classification": report["classification"],
                    "artifacts": {
                        "directory": alias,
                        "resolved_config": f"{alias}/resolved_config.json",
                        "result": f"{alias}/result.json",
                        "trace": f"{alias}/trace.npz",
                        "sha256": {
                            "resolved_config": file_sha256(
                                final_directory / "resolved_config.json"
                            ),
                            "result": file_sha256(final_directory / "result.json"),
                            "trace": file_sha256(final_directory / "trace.npz"),
                            "video": None,
                        },
                    },
                }
            ],
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine only closure controls while freezing hand/object geometry."
    )
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--stage1-count", type=int, default=48)
    parser.add_argument("--stage2-count", type=int, default=64)
    parser.add_argument("--stage2-seed-count", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_fixed_pose_control_search(
        args.base_config,
        args.output_dir,
        workers=args.workers,
        seed=args.seed,
        budget=ControlSearchBudget(
            stage1_count=args.stage1_count,
            stage2_count=args.stage2_count,
            stage2_seed_count=args.stage2_seed_count,
        ),
    )
    print(json_text({key: value for key, value in report.items() if key != "results"}))
    return 0 if int(report["hard_pass_count"]) > 0 else 2


__all__ = [
    "CAMPAIGN_KIND",
    "ControlSearchBudget",
    "DEFAULT_BASE_CONFIG",
    "DEFAULT_CONTROL_CENTER",
    "DEFAULT_OUTPUT",
    "VARIABLE_ORDER",
    "assert_control_only_candidate",
    "build_parser",
    "candidate_rank",
    "control_vector",
    "frozen_context_sha256",
    "generate_stage1_candidates",
    "generate_stage2_candidates",
    "main",
    "materialize_control_candidate",
    "rank_results",
    "run_fixed_pose_control_search",
]
