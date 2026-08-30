"""Authenticated, out-of-tree active-set recovery for schema-v11 campaigns.

This command deliberately does not resume or mutate its parent campaign.  It
authenticates the committed quick/expanded pools, ranks only candidates with
complete v11 collision-safety evidence, and runs projected active-set DLS in a
separate workspace.  A child is promoted only after a fresh real MuJoCo
collision-witness evaluation satisfies every static hard gate.

The public selection helpers are intentionally pure.  Besides making the
per-edge budget auditable, that keeps the result independent of worker count.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import file_sha256
from ..config import ACTIVE_FINGERS, validate_config
from ..grasp_pose import controller_id, grasp_pose_id
from .pose_preserving_seed_campaign import canonical_sha256


RECOVERY_SCHEMA_VERSION = 2
RECOVERY_ID_BASE = 8_110_000_000_000_000
DEFAULT_PER_EDGE_BATCH = 12
_TOLERANCE = 1e-12


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write strict JSON by atomic same-filesystem replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _mapping_value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _finite(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def candidate_has_complete_v11_safety(record: Mapping[str, Any]) -> bool:
    """Fail closed unless every independent schema-v11 safety proof is clear."""

    metrics = record.get("static_metrics")
    if not isinstance(metrics, Mapping):
        return False
    witnesses = metrics.get("target_witness")
    if not isinstance(witnesses, Mapping) or any(
        not isinstance(witnesses.get(finger), Mapping) for finger in ACTIVE_FINGERS
    ):
        return False
    return bool(
        metrics.get("precontact_geometry_evaluated") is True
        and metrics.get("cube_freejoint_qpos_unchanged") is True
        and int(metrics.get("missing_target_witness_count", 1)) == 0
        and int(metrics.get("off_target_distal_penetrating_count", 1)) == 0
        and _finite(metrics.get("minimum_active_nondistal_gap_m"), -math.inf)
        >= -_TOLERANCE
        and _finite(metrics.get("nominal_minimum_forbidden_hand_gap_m"), -math.inf)
        >= -_TOLERANCE
        and _finite(metrics.get("nominal_maximum_all_distal_penetration_m"), math.inf)
        <= 0.002 + _TOLERANCE
        and _finite(metrics.get("precontact_minimum_hand_gap_m"), -math.inf)
        >= -_TOLERANCE
    )


def normalized_gate_violations(record: Mapping[str, Any]) -> tuple[float, ...]:
    """Return dimensionless hard-gate excesses used by the recovery rank.

    Safety is intentionally absent: unsafe candidates are discarded rather
    than assigned a merely large score.  A zero vector is equivalent to the
    static geometric gates, though it is still not dynamic success evidence.
    """

    metrics = record["static_metrics"]
    witnesses = metrics["target_witness"]
    values: list[float] = []
    for finger in ACTIVE_FINGERS:
        witness = witnesses[finger]
        gap = _finite(witness.get("signed_gap_m"), math.inf)
        values.append(max(0.0, -0.0005 - gap, gap - 0.00025) / 0.0005)
        normal = _finite(witness.get("normal_alignment"), -math.inf)
        values.append(max(0.0, 0.95 - normal) / 0.05)
        edge = _finite(witness.get("edge_margin_m"), -math.inf)
        values.append(max(0.0, 0.0005 - edge) / 0.0005)
    height = _finite(metrics.get("contact_height_spread_m"), math.inf)
    values.append(max(0.0, height - 0.005) / 0.005)
    retreats = metrics.get("retreat_evidence", {})
    for finger in ACTIVE_FINGERS:
        retreat = retreats.get(finger) if isinstance(retreats, Mapping) else None
        if not isinstance(retreat, Mapping):
            values.extend((math.inf, math.inf))
            continue
        distance = _finite(retreat.get("measured_outward_retreat_m"), -math.inf)
        values.append(max(0.0, 0.002 - distance, distance - 0.004) / 0.002)
        angle = _finite(retreat.get("closure_angle_deg"), math.inf)
        values.append(max(0.0, angle - 30.0) / 30.0)
    return tuple(float(value) for value in values)


def recovery_gate_merit(record: Mapping[str, Any]) -> tuple[float, float, int]:
    """Minimize the worst normalized gate first, then aggregate L2 error."""

    violations = normalized_gate_violations(record)
    maximum = max(violations, default=math.inf)
    l2 = math.sqrt(sum(value * value for value in violations))
    return maximum, l2, int(record["candidate_id"])


@dataclass(frozen=True, slots=True)
class RecoveryBatchSelection:
    first: tuple[dict[str, Any], ...]
    second: tuple[dict[str, Any], ...]
    unsafe_candidate_count: int
    safe_candidate_count: int
    per_edge_available: Mapping[float, int]


@dataclass(frozen=True, slots=True)
class FreshFullSceneFilterResult:
    accepted: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]
    input_candidate_count: int
    stored_v11_safe_count: int
    fresh_full_scene_safe_count: int
    model_compile_count: int


def full_scene_model_signature(config: Mapping[str, Any]) -> str:
    """Hash every config field that can change the compiled collision scene."""

    return canonical_sha256(
        {
            "schema_version": int(config["schema_version"]),
            "experiment_id": str(config["experiment_id"]),
            "side": str(config["side"]),
            # Keep the entire cube and scene blocks.  Some fields are only
            # dynamic today, but binding them fails safely if MjSpec evolves.
            "cube": copy.deepcopy(dict(config["cube"])),
            "scene": copy.deepcopy(dict(config["scene"])),
        }
    )


def fresh_full_scene_source_filter(
    records: Sequence[Mapping[str, Any]],
    *,
    gate_factory: Callable[[Mapping[str, Any]], Any] | None = None,
) -> FreshFullSceneFilterResult:
    """Freshly reject self/environment penetration before merit selection.

    The recorded direct-pose nominal and precontact qpos are authoritative;
    controller fields are intentionally ignored.  One compiled gate is reused
    for every candidate with an identical collision-model signature.
    """

    if gate_factory is None:
        from .relative_wrist_pose_active_set import CachedFullScenePenetrationGate

        gate_factory = CachedFullScenePenetrationGate
    safe_sources = sorted(
        (
            copy.deepcopy(dict(value))
            for value in records
            if candidate_has_complete_v11_safety(value)
        ),
        key=lambda value: int(value["candidate_id"]),
    )
    gates: dict[str, Any] = {}
    accepted: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for value in safe_sources:
        config = value.get("config")
        metrics = value.get("static_metrics")
        if not isinstance(config, Mapping) or not isinstance(metrics, Mapping):
            raise RuntimeError("stored-v11-safe candidate lost config/static metrics")
        signature = full_scene_model_signature(config)
        gate = gates.get(signature)
        if gate is None:
            gate = gate_factory(config)
            gates[signature] = gate
        nominal = metrics.get("nominal_joint_qpos_rad")
        precontact = metrics.get("precontact_joint_qpos_rad")
        report = gate.evaluate(
            config,
            nominal_joint_qpos_rad=nominal,
            precontact_joint_qpos_rad=precontact,
        )
        passed = bool(report.get("passed", False))
        diagnostic = {
            "candidate_id": int(value["candidate_id"]),
            "candidate_sha256": str(value["candidate_sha256"]),
            "edge_m": float(value["edge_m"]),
            "clockwise_orbit_deg": float(value.get("clockwise_orbit_deg", 0.0)),
            "model_signature": signature,
            "fresh_full_scene_safe": passed,
            "full_scene_contact_safety": copy.deepcopy(dict(report)),
            "qpos_source": "record.static_metrics.nominal_and_precontact",
        }
        diagnostics.append(diagnostic)
        if passed:
            promoted = copy.deepcopy(value)
            promoted["fresh_full_scene_source_safety"] = copy.deepcopy(diagnostic)
            accepted.append(promoted)
    return FreshFullSceneFilterResult(
        accepted=tuple(accepted),
        diagnostics=tuple(diagnostics),
        input_candidate_count=len(records),
        stored_v11_safe_count=len(safe_sources),
        fresh_full_scene_safe_count=len(accepted),
        model_compile_count=len(gates),
    )


def _orbit_covering_batch(
    records: Sequence[Mapping[str, Any]], limit: int
) -> tuple[dict[str, Any], ...]:
    ranked = sorted(records, key=recovery_gate_merit)
    by_orbit: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for value in ranked:
        by_orbit[float(value.get("clockwise_orbit_deg", 0.0))].append(value)
    selected: list[Mapping[str, Any]] = []
    # Take the best member of each feasible orbit before filling from global
    # merit.  Orbits themselves are ordered by their best merit, not angle.
    orbit_heads = sorted(
        (values[0] for values in by_orbit.values()), key=recovery_gate_merit
    )
    selected.extend(orbit_heads[:limit])
    selected_ids = {int(value["candidate_id"]) for value in selected}
    selected.extend(
        value
        for value in ranked
        if int(value["candidate_id"]) not in selected_ids
    )
    return tuple(copy.deepcopy(dict(value)) for value in selected[:limit])


def select_recovery_batches(
    records: Sequence[Mapping[str, Any]], *, per_edge: int = DEFAULT_PER_EDGE_BATCH
) -> RecoveryBatchSelection:
    """Build two deterministic, non-overlapping, orbit-covering edge batches."""

    if not isinstance(per_edge, int) or isinstance(per_edge, bool) or per_edge <= 0:
        raise ValueError("per_edge must be a positive integer")
    safe = [value for value in records if candidate_has_complete_v11_safety(value)]
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for value in safe:
        grouped[float(value["edge_m"])].append(value)
    first: list[dict[str, Any]] = []
    second: list[dict[str, Any]] = []
    available: dict[float, int] = {}
    for edge in sorted(grouped):
        candidates = grouped[edge]
        available[edge] = len(candidates)
        batch_one = _orbit_covering_batch(candidates, per_edge)
        used = {int(value["candidate_id"]) for value in batch_one}
        remainder = [
            value for value in candidates if int(value["candidate_id"]) not in used
        ]
        batch_two = _orbit_covering_batch(remainder, per_edge)
        first.extend(batch_one)
        second.extend(batch_two)
    return RecoveryBatchSelection(
        first=tuple(first),
        second=tuple(second),
        unsafe_candidate_count=len(records) - len(safe),
        safe_candidate_count=len(safe),
        per_edge_available=available,
    )


def select_second_batch_for_failed_edges(
    second_batch: Sequence[Mapping[str, Any]],
    first_results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return the disjoint second batch only for edges with zero first passes."""

    passed_edges = {
        float(value["edge_m"])
        for value in first_results
        if bool(value.get("static_pass", False))
    }
    return tuple(
        copy.deepcopy(dict(value))
        for value in second_batch
        if float(value["edge_m"]) not in passed_edges
    )


def validate_recovery_paths(parent_campaign: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    parent = Path(parent_campaign).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError(
            "recovery output must be disjoint from the authenticated parent campaign"
        )
    return parent, output


def _candidate_child_id(batch_index: int, order: int) -> int:
    return RECOVERY_ID_BASE + batch_index * 100_000 + order


def classify_deep_scene_contacts(
    contacts: Sequence[Mapping[str, Any]],
    *,
    cube_geom_id: int,
    support_geom_id: int,
    maximum_penetration_m: float = 0.002,
) -> tuple[dict[str, Any], ...]:
    """Reject every deep contact except the intended cube/support pair."""

    allowed = frozenset((int(cube_geom_id), int(support_geom_id)))
    violations: list[dict[str, Any]] = []
    for raw in contacts:
        pair = frozenset((int(raw["geom1_id"]), int(raw["geom2_id"])))
        distance = float(raw["distance_m"])
        if pair == allowed or distance >= -maximum_penetration_m - _TOLERANCE:
            continue
        violations.append(copy.deepcopy(dict(raw)))
    return tuple(violations)


def boundary_release_initial_variables(
    config: Mapping[str, Any],
    joint_bounds: Mapping[str, tuple[float, float]],
    *,
    boundary_tolerance_rad: float = 1e-6,
    maximum_seed_count: int = 12,
) -> tuple[tuple[str, Any], ...]:
    """Generate small deterministic inward seeds for active joint bounds.

    Same-finger coupling is tried first because it preserves the other two
    contact geometries.  Opposite coupling signs are retained: the useful
    direction depends on the local link Jacobian, not the joint's name.
    """

    from .relative_wrist_pose_search import NON_THUMB_ACTUATORS, RelativeWristVariables

    base = RelativeWristVariables.from_config(config)
    groups = (
        frozenset(name for name in NON_THUMB_ACTUATORS if "thumb" in name),
        frozenset(name for name in NON_THUMB_ACTUATORS if "index" in name),
        frozenset(name for name in NON_THUMB_ACTUATORS if "mid" in name),
    )
    seeds: list[tuple[str, RelativeWristVariables]] = []
    qpos = np.asarray(base.non_thumb_joint_qpos_rad, dtype=np.float64)
    for index, name in enumerate(NON_THUMB_ACTUATORS):
        low, high = joint_bounds[name]
        direction = 0.0
        if qpos[index] <= low + boundary_tolerance_rad:
            direction = 1.0
        elif qpos[index] >= high - boundary_tolerance_rad:
            direction = -1.0
        if direction == 0.0:
            continue
        same_group = next(group for group in groups if name in group)
        partner_names = [
            candidate
            for candidate in NON_THUMB_ACTUATORS
            if candidate != name and candidate in same_group
        ] + [
            candidate
            for candidate in NON_THUMB_ACTUATORS
            if candidate != name and candidate not in same_group
        ]
        for inward, coupling in ((0.01, 0.003), (0.02, 0.006)):
            shifted = qpos.copy()
            shifted[index] = np.clip(
                shifted[index] + direction * inward, low, high
            )
            seeds.append(
                (
                    f"{name}:inward_{inward:.3f}",
                    RelativeWristVariables(
                        tuple(shifted),
                        base.root_delta_cube_m,
                        base.wrist_local_rotvec_rad,
                    ),
                )
            )
            for partner in partner_names:
                partner_index = NON_THUMB_ACTUATORS.index(partner)
                partner_low, partner_high = joint_bounds[partner]
                for sign in (-1.0, 1.0):
                    coupled = shifted.copy()
                    coupled[partner_index] = np.clip(
                        coupled[partner_index] + sign * coupling,
                        partner_low,
                        partner_high,
                    )
                    seeds.append(
                        (
                            f"{name}:inward_{inward:.3f};{partner}:{sign * coupling:+.3f}",
                            RelativeWristVariables(
                                tuple(coupled),
                                base.root_delta_cube_m,
                                base.wrist_local_rotvec_rad,
                            ),
                        )
                    )
                    if len(seeds) >= maximum_seed_count:
                        return tuple(seeds)
    return tuple(seeds)


def _solver_result_promotable(value: Any) -> bool:
    return bool(
        getattr(value.static_result, "static_geometry_pass", False)
        and value.diagnostics.get("promotion_config_valid") is True
    )


def _solve_one(job: tuple[dict[str, Any], int, int]) -> dict[str, Any]:
    """Spawn-safe active-set solve followed by an independent real recheck."""

    source, batch_index, order = job
    from .actual_contact_grasp_pose import apply_precontact_solution
    from .relative_wrist_pose_active_set import (
        build_active_set_evaluation_context,
        evaluate_active_set_candidate,
        solve_orientation_aware_active_set_dls,
    )

    candidate_id = _candidate_child_id(batch_index, order)
    orbit = float(source.get("clockwise_orbit_deg", 0.0))
    try:
        evaluation_context = build_active_set_evaluation_context(source["config"])
        joint_bounds = evaluation_context.joint_bounds
        solved = solve_orientation_aware_active_set_dls(
            source["config"],
            clockwise_orbit_deg=orbit,
            evaluation_context=evaluation_context,
        )
        seed_attempts: list[dict[str, Any]] = [
            {
                "seed": "unmodified",
                "stop_reason": solved.stop_reason,
                "static_geometry_pass": bool(
                    getattr(solved.static_result, "static_geometry_pass", False)
                ),
                "promotion_config_valid": solved.diagnostics.get(
                    "promotion_config_valid"
                ),
            }
        ]
        if not _solver_result_promotable(solved):
            for label, variables in boundary_release_initial_variables(
                source["config"], joint_bounds
            ):
                try:
                    released = solve_orientation_aware_active_set_dls(
                        source["config"],
                        clockwise_orbit_deg=orbit,
                        initial_variables=variables,
                        evaluation_context=evaluation_context,
                    )
                except (RuntimeError, ValueError, ArithmeticError) as error:
                    seed_attempts.append(
                        {"seed": label, "error": f"{type(error).__name__}: {error}"}
                    )
                    continue
                seed_attempts.append(
                    {
                        "seed": label,
                        "stop_reason": released.stop_reason,
                        "static_geometry_pass": bool(
                            getattr(released.static_result, "static_geometry_pass", False)
                        ),
                        "promotion_config_valid": released.diagnostics.get(
                            "promotion_config_valid"
                        ),
                    }
                )
                if _solver_result_promotable(released):
                    solved = released
                    break
        # Never trust the optimizer's terminal observation as promotion
        # evidence.  Re-run fresh forwards through the same compiled context.
        evaluation, full_scene_safety = evaluate_active_set_candidate(
            evaluation_context, solved.config
        )
        static_result = evaluation.static_result
        static_pass = bool(
            evaluation.safe
            and getattr(static_result, "static_geometry_pass", False)
            and solved.diagnostics.get("promotion_config_valid") is True
            and full_scene_safety.get("passed") is True
        )
        child_config = copy.deepcopy(dict(solved.config))
        promotion_error: str | None = None
        if not bool(full_scene_safety.get("passed", False)):
            promotion_error = "full_scene_contact_penetration_over_2mm"
        if static_pass:
            child_config = apply_precontact_solution(child_config, static_result)
            try:
                validate_config(child_config)
            except ValueError as error:
                static_pass = False
                promotion_error = str(error)
        metadata = child_config.setdefault("candidate_metadata", {})
        metadata.update(
            {
                "stage": "relative_wrist_active_set_recovery",
                "candidate_id": candidate_id,
                "parent_candidate_id": int(source["candidate_id"]),
                "recovery_batch_index": batch_index,
                "active_set_dls": True,
            }
        )
        result = {
            **{key: copy.deepcopy(source[key]) for key in (
                "edge_m", "thumb_actual_center_rad", "clockwise_orbit_deg"
            ) if key in source},
            "candidate_id": candidate_id,
            "parent_candidate_id": int(source["candidate_id"]),
            "config": child_config,
            "static_pass": static_pass,
            "static_metrics": static_result.as_dict(),
            "active_set_dls": {
                "executed": True,
                "stop_reason": solved.stop_reason,
                "diagnostics": copy.deepcopy(solved.diagnostics),
                "boundary_release_seed_attempts": seed_attempts,
                "fresh_terminal_evaluation": True,
                "promotion_config_error": promotion_error,
                "full_scene_contact_safety": full_scene_safety,
            },
        }
        result["promotion_config_valid"] = bool(
            solved.diagnostics.get("promotion_config_valid") is True
        )
        result["initial_contact_safety_pass"] = bool(
            full_scene_safety.get("passed") is True
        )
        result["recovery_static_promotable"] = static_pass
        result["grasp_pose_id"] = grasp_pose_id(child_config)
        result["controller_id"] = controller_id(child_config)
        result["candidate_sha256"] = canonical_sha256(child_config)
        result["gate_merit"] = list(recovery_gate_merit(result))
        return result
    except (RuntimeError, ValueError, ArithmeticError) as error:
        return {
            "candidate_id": candidate_id,
            "parent_candidate_id": int(source["candidate_id"]),
            "edge_m": float(source["edge_m"]),
            "clockwise_orbit_deg": float(source.get("clockwise_orbit_deg", 0.0)),
            "static_pass": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _execute_jobs(
    records: Sequence[Mapping[str, Any]], *, batch_index: int, workers: int
) -> tuple[dict[str, Any], ...]:
    jobs = tuple(
        (copy.deepcopy(dict(value)), batch_index, order)
        for order, value in enumerate(records)
    )
    if workers == 1:
        values = [_solve_one(job) for job in jobs]
    else:
        import multiprocessing

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            values = list(executor.map(_solve_one, jobs))
    return tuple(sorted(values, key=lambda value: int(value["candidate_id"])))


def _artifact_descriptor(output: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(output)),
        "sha256": file_sha256(path),
    }


def recovery_source_hashes() -> dict[str, dict[str, str]]:
    """Hash the optimizer, trust boundary and every private downstream owner."""

    module_names = (
        "xhand_grasp.tuning.relative_wrist_pose_active_set_recovery",
        "xhand_grasp.tuning.relative_wrist_pose_active_set",
        "xhand_grasp.tuning.relative_wrist_pose_recovery_auth",
        "xhand_grasp.tuning.actual_contact_grasp_pose",
        "xhand_grasp.tuning.actual_contact_grasp_pose_dynamic",
        "xhand_grasp.tuning.actual_contact_grasp_pose_measured",
        "xhand_grasp.tuning.actual_contact_manipulation",
        "xhand_grasp.actual_contact_grasp_pose_catalog",
        "xhand_grasp.actual_contact_selection",
    )
    result: dict[str, dict[str, str]] = {}
    for name in module_names:
        module = importlib.import_module(name)
        source = inspect.getsourcefile(module)
        if source is None:
            raise RuntimeError(f"recovery dependency has no source file: {name}")
        path = Path(source).resolve()
        result[name] = {"path": str(path), "sha256": file_sha256(path)}
    return result


def build_recovery_base_manifest(
    *,
    parent: Path,
    experiment_id: str,
    campaign_input_sha256: str,
    source_bundle_sha256: str,
    parent_snapshot_sha256: str,
    seed: int,
) -> dict[str, Any]:
    """Build the immutable target/worker-independent recovery identity."""

    manifest = {
        "relative_wrist_active_set_recovery_manifest_schema_version": (
            RECOVERY_SCHEMA_VERSION
        ),
        "complete": True,
        "parent_campaign": str(parent),
        "parent_snapshot_sha256": str(parent_snapshot_sha256),
        "experiment_id": str(experiment_id),
        "parent_campaign_input_sha256": str(campaign_input_sha256),
        "source_bundle_sha256": str(source_bundle_sha256),
        "seed": int(seed),
        "source_code": recovery_source_hashes(),
        "selection_policy": (
            "v11_safe_then_normalized_max_l2_per_edge_orbit_coverage"
        ),
        "first_batch_per_edge": DEFAULT_PER_EDGE_BATCH,
        "second_batch_per_edge_if_first_has_zero_static_pass": (
            DEFAULT_PER_EDGE_BATCH
        ),
    }
    manifest["recovery_input_sha256"] = canonical_sha256(manifest)
    return manifest


def build_recovery_run_manifest(
    *,
    recovery_input_sha256: str,
    target_success_count: int,
    workers: int,
) -> dict[str, Any]:
    result = {
        "relative_wrist_active_set_recovery_run_manifest_schema_version": 1,
        "complete": True,
        "recovery_input_sha256": str(recovery_input_sha256),
        "target_success_count": int(target_success_count),
        "workers": int(workers),
        "worker_count_changes_numerical_result": False,
    }
    result["run_input_sha256"] = canonical_sha256(result)
    return result


def _load_ledger(output: Path) -> dict[str, Any]:
    path = output / "recovery_stage_ledger.json"
    if not path.is_file():
        return {"recovery_stage_ledger_schema_version": 1, "stages": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("recovery_stage_ledger_schema_version", 0)) != 1:
        raise RuntimeError("unsupported recovery stage ledger")
    previous = None
    for stage in payload.get("stages", ()):
        expected = canonical_sha256({key: value for key, value in stage.items() if key != "stage_sha256"})
        if stage.get("stage_sha256") != expected or stage.get("previous_stage_sha256") != previous:
            raise RuntimeError("recovery stage ledger hash chain changed")
        for artifact in stage.get("artifacts", ()):
            path = output / artifact["path"]
            if not path.is_file() or file_sha256(path) != artifact["sha256"]:
                raise RuntimeError(f"recovery stage artifact changed: {path}")
        previous = stage["stage_sha256"]
    return payload


def _stage_input_without_runtime_workers(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("workers", None)
    return result


def _commit_stage(
    output: Path,
    name: str,
    *,
    stage_input: Mapping[str, Any],
    artifacts: Sequence[Path],
    summary: Mapping[str, Any],
) -> None:
    ledger = _load_ledger(output)
    stages = list(ledger["stages"])
    input_sha = canonical_sha256(stage_input)
    existing = next((value for value in stages if value["name"] == name), None)
    descriptors = [_artifact_descriptor(output, value) for value in artifacts]
    if existing is not None:
        same_input = bool(
            existing["stage_input_sha256"] == input_sha
            or _stage_input_without_runtime_workers(existing.get("stage_input", {}))
            == _stage_input_without_runtime_workers(stage_input)
        )
        if not same_input or existing["artifacts"] != descriptors:
            raise RuntimeError(f"recovery stage resume input changed: {name}")
        return
    previous = None if not stages else stages[-1]["stage_sha256"]
    record = {
        "name": name,
        "stage_input_sha256": input_sha,
        "stage_input": copy.deepcopy(dict(stage_input)),
        "artifacts": descriptors,
        "summary": copy.deepcopy(dict(summary)),
        "previous_stage_sha256": previous,
    }
    record["stage_sha256"] = canonical_sha256(record)
    stages.append(record)
    _atomic_json(
        output / "recovery_stage_ledger.json",
        {"recovery_stage_ledger_schema_version": 1, "stages": stages},
    )


def _write_or_authenticate(path: Path, payload: Mapping[str, Any], *, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(f"recovery artifact already exists: {path}")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != canonical_sha256(payload):
            raise RuntimeError(f"recovery resume input changed: {path}")
        return
    _atomic_json(path, payload)


def _load_committed_batch(
    output: Path,
    *,
    stage_name: str,
    path: Path,
    batch_index: int,
    stage_input: Mapping[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    """Load a hash-committed batch without invoking DLS a second time."""

    ledger = _load_ledger(output)
    stage = next(
        (value for value in ledger.get("stages", ()) if value["name"] == stage_name),
        None,
    )
    if stage is None:
        if path.exists():
            raise RuntimeError(
                f"uncommitted recovery batch artifact requires audit: {path}"
            )
        return None
    if _stage_input_without_runtime_workers(stage.get("stage_input", {})) != (
        _stage_input_without_runtime_workers(stage_input)
    ):
        raise RuntimeError(f"recovery stage resume input changed: {stage_name}")
    expected_artifact = _artifact_descriptor(output, path)
    if expected_artifact not in stage.get("artifacts", ()):
        raise RuntimeError(
            f"committed recovery stage does not bind expected batch: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True or int(payload.get("batch", -1)) != batch_index:
        raise RuntimeError(f"invalid committed recovery batch: {path}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError(f"committed recovery batch has no candidates: {path}")
    result = tuple(copy.deepcopy(dict(value)) for value in candidates)
    identifiers = [int(value["candidate_id"]) for value in result]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"committed recovery batch has duplicate IDs: {path}")
    for value in result:
        config = value.get("config")
        if config is None:
            if bool(value.get("static_pass", False)):
                raise RuntimeError(
                    f"promotable committed recovery candidate has no config: {path}"
                )
            continue
        if not isinstance(config, Mapping):
            raise RuntimeError(f"committed recovery candidate config is invalid: {path}")
        if canonical_sha256(config) != str(value.get("candidate_sha256")):
            raise RuntimeError(f"committed recovery candidate hash changed: {path}")
        if grasp_pose_id(config) != str(value.get("grasp_pose_id")):
            raise RuntimeError(f"committed recovery grasp pose changed: {path}")
    return result


def _fresh_filter_source_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        [
            {
                "candidate_id": int(value["candidate_id"]),
                "candidate_sha256": str(value["candidate_sha256"]),
                "static_metrics_sha256": canonical_sha256(value["static_metrics"]),
            }
            for value in sorted(records, key=lambda item: int(item["candidate_id"]))
        ]
    )


def _load_committed_fresh_filter(
    output: Path,
    *,
    path: Path,
    stage_input: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
) -> FreshFullSceneFilterResult | None:
    """Strictly restore the source filter without recompiling a MuJoCo model."""

    ledger = _load_ledger(output)
    stage_name = "fresh_full_scene_source_filter"
    stage = next(
        (value for value in ledger.get("stages", ()) if value["name"] == stage_name),
        None,
    )
    if stage is None:
        if path.exists():
            raise RuntimeError(
                f"uncommitted fresh full-scene filter requires audit: {path}"
            )
        return None
    if stage.get("stage_input") != dict(stage_input):
        raise RuntimeError("fresh full-scene filter resume input changed")
    expected_artifact = _artifact_descriptor(output, path)
    if expected_artifact not in stage.get("artifacts", ()):
        raise RuntimeError("fresh full-scene filter stage lost its report binding")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(payload.get("fresh_full_scene_source_filter_schema_version", 0)) != 1
        or payload.get("complete") is not True
        or payload.get("source_candidate_sha256")
        != stage_input["source_candidate_sha256"]
    ):
        raise RuntimeError("invalid committed fresh full-scene filter report")
    by_id = {int(value["candidate_id"]): value for value in source_records}
    diagnostics_raw = payload.get("candidate_diagnostics")
    if not isinstance(diagnostics_raw, list):
        raise RuntimeError("fresh full-scene filter report has no diagnostics")
    diagnostics: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in diagnostics_raw:
        diagnostic = copy.deepcopy(dict(raw))
        identifier = int(diagnostic["candidate_id"])
        if identifier in seen or identifier not in by_id:
            raise RuntimeError("fresh full-scene filter candidate IDs changed")
        seen.add(identifier)
        source = by_id[identifier]
        if str(diagnostic["candidate_sha256"]) != str(source["candidate_sha256"]):
            raise RuntimeError("fresh full-scene filter candidate hash changed")
        diagnostics.append(diagnostic)
        if bool(diagnostic.get("fresh_full_scene_safe", False)):
            value = copy.deepcopy(dict(source))
            value["fresh_full_scene_source_safety"] = copy.deepcopy(diagnostic)
            accepted.append(value)
    stored_safe_count = sum(
        candidate_has_complete_v11_safety(value) for value in source_records
    )
    if len(diagnostics) != stored_safe_count:
        raise RuntimeError("fresh full-scene filter diagnostic count changed")
    if len(accepted) != int(payload.get("fresh_full_scene_safe_count", -1)):
        raise RuntimeError("fresh full-scene filter accepted count changed")
    return FreshFullSceneFilterResult(
        accepted=tuple(accepted),
        diagnostics=tuple(diagnostics),
        input_candidate_count=len(source_records),
        stored_v11_safe_count=stored_safe_count,
        fresh_full_scene_safe_count=len(accepted),
        model_compile_count=int(payload["model_compile_count"]),
    )


def _run_or_resume_fresh_full_scene_filter(
    records: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    recovery_input_sha256: str,
    resume: bool,
    gate_factory: Callable[[Mapping[str, Any]], Any] | None = None,
) -> FreshFullSceneFilterResult:
    path = output / "static" / "fresh_full_scene_source_filter.json"
    stage_input = {
        "recovery_input_sha256": str(recovery_input_sha256),
        "source_candidate_sha256": _fresh_filter_source_sha256(records),
        "filter": {
            "qpos_source": "record.static_metrics.nominal_and_precontact",
            "maximum_penetration_m": 0.002,
            "only_exempt_contact": "cube_support",
            "model_reuse": "collision_model_signature",
        },
    }
    loaded = _load_committed_fresh_filter(
        output,
        path=path,
        stage_input=stage_input,
        source_records=records,
    )
    if loaded is not None:
        return loaded
    result = fresh_full_scene_source_filter(records, gate_factory=gate_factory)
    report = {
        "fresh_full_scene_source_filter_schema_version": 1,
        "complete": True,
        "source_candidate_sha256": stage_input["source_candidate_sha256"],
        "input_candidate_count": result.input_candidate_count,
        "stored_v11_safe_count": result.stored_v11_safe_count,
        "fresh_full_scene_safe_count": result.fresh_full_scene_safe_count,
        "rejected_count": (
            result.stored_v11_safe_count - result.fresh_full_scene_safe_count
        ),
        "model_compile_count": result.model_compile_count,
        "candidate_diagnostics": list(result.diagnostics),
    }
    _write_or_authenticate(path, report, resume=resume)
    _commit_stage(
        output,
        "fresh_full_scene_source_filter",
        stage_input=stage_input,
        artifacts=(path,),
        summary={
            "stored_v11_safe_count": result.stored_v11_safe_count,
            "fresh_full_scene_safe_count": result.fresh_full_scene_safe_count,
            "model_compile_count": result.model_compile_count,
        },
    )
    return result


def _run_dynamic_with_optional_local_refinement(
    static_records: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    workers: int,
    seed: int,
    target_success_count: int,
    recovery_input_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    """Run base controllers, then the registered per-edge local budget if needed."""

    from ..experiment import resolve_experiment
    from .actual_contact_grasp_pose import (
        _merge_dynamic_executions,
        _run_dynamic_stage,
        _run_joint_controller_local_refinement_stage,
        _run_materialized_local_dynamic_stage,
    )

    for value in static_records:
        if not bool(value.get("static_pass", False)) or not bool(
            value.get("recovery_static_promotable", False)
        ) or not bool(value.get("initial_contact_safety_pass", False)):
            raise RuntimeError(
                "local-controller recovery requires a freshly safe static source"
            )
    base = _run_dynamic_stage(
        static_records, output, stage="recovery", workers=workers, seed=seed
    )
    _commit_stage(
        output,
        "recovery_dynamic",
        stage_input={
            "recovery_input_sha256": recovery_input_sha256,
            "workers": workers,
        },
        artifacts=base.artifacts,
        summary=base.summary,
    )
    base_grasp_count = int(base.summary.get("grasp_success_count", 0))
    execute_local = base_grasp_count < target_success_count
    decision = {
        "recovery_local_refinement_decision_schema_version": 1,
        "complete": True,
        "target_success_count": target_success_count,
        "base_dynamic_candidate_count": len(base.records),
        "base_grasp_success_count": base_grasp_count,
        "local_refinement_executed": execute_local,
        "decision_reason": (
            "base_grasp_count_below_target"
            if execute_local
            else "base_grasp_target_reached"
        ),
    }
    decision_path = (
        output
        / "dynamic"
        / "local_refinement_decisions"
        / f"target_{target_success_count}.json"
    )
    _write_or_authenticate(decision_path, decision, resume=decision_path.exists())
    _commit_stage(
        output,
        f"recovery_local_refinement_decision_{target_success_count}",
        stage_input={
            "recovery_input_sha256": recovery_input_sha256,
            "target_success_count": target_success_count,
            "base_dynamic_evidence_sha256": canonical_sha256(
                [
                    {
                        "candidate_id": int(value["candidate_id"]),
                        "candidate_sha256": str(value["candidate_sha256"]),
                        "summary_sha256": canonical_sha256(value.get("summary", {})),
                    }
                    for value in base.records
                ]
            ),
        },
        artifacts=(decision_path,),
        summary=decision,
    )

    local_refinement = None
    local_dynamic = None
    merged = base
    if execute_local:
        definition = resolve_experiment(dict(static_records[0]["config"]))
        campaign = definition.actual_contact_grasp_pose_campaign
        if campaign is None or definition.relative_wrist_pose_search is None:
            raise RuntimeError("v11 local refinement requires registered budgets")
        local_refinement = _run_joint_controller_local_refinement_stage(
            base.records,
            output,
            stage="recovery_active_set_local",
            top_count=int(campaign.local_pose_count),
            candidates_per_pose=int(campaign.local_refine_per_pose),
            seed=seed,
        )
        _commit_stage(
            output,
            "recovery_joint_controller_local_refinement",
            stage_input={
                "recovery_input_sha256": recovery_input_sha256,
                "base_dynamic_candidate_sha256": canonical_sha256(
                    [value["candidate_sha256"] for value in base.records]
                ),
                "top_count": int(campaign.local_pose_count),
                "candidates_per_pose": int(campaign.local_refine_per_pose),
                "seed": seed,
            },
            artifacts=local_refinement.artifacts,
            summary=local_refinement.summary,
        )
        local_dynamic_report = (
            output
            / "dynamic"
            / "recovery_active_set_local_local_refinement_report.json"
        )
        local_workers = workers
        if local_dynamic_report.is_file():
            prior_local = json.loads(
                local_dynamic_report.read_text(encoding="utf-8")
            )
            local_workers = int(prior_local.get("workers", workers))
        local_dynamic = _run_materialized_local_dynamic_stage(
            local_refinement.records,
            output,
            stage="recovery_active_set_local",
            workers=local_workers,
        )
        _commit_stage(
            output,
            "recovery_local_refinement_dynamic",
            stage_input={
                "recovery_input_sha256": recovery_input_sha256,
                "local_candidate_sha256": canonical_sha256(
                    [value["candidate_sha256"] for value in local_refinement.records]
                ),
                "workers": local_workers,
            },
            artifacts=local_dynamic.artifacts,
            summary=local_dynamic.summary,
        )
        merged = _merge_dynamic_executions(base, local_dynamic)

    merge_report = {
        "recovery_dynamic_merge_schema_version": 1,
        "complete": True,
        "target_success_count": target_success_count,
        "local_refinement_executed": execute_local,
        "decision_reason": decision["decision_reason"],
        "base_dynamic_candidate_count": len(base.records),
        "base_grasp_success_count": base_grasp_count,
        "local_refinement_selected_pose_count": (
            0
            if local_refinement is None
            else int(local_refinement.summary.get("selected_pose_count", 0))
        ),
        "local_refinement_generated_candidate_count": (
            0
            if local_refinement is None
            else int(local_refinement.summary.get("generated_candidate_count", 0))
        ),
        "local_refinement_dynamic_promoted_count": (
            0
            if local_refinement is None
            else int(local_refinement.summary.get("dynamic_promoted_count", 0))
        ),
        "local_dynamic_candidate_count": (
            0 if local_dynamic is None else len(local_dynamic.records)
        ),
        "local_grasp_success_count": (
            0
            if local_dynamic is None
            else int(local_dynamic.summary.get("grasp_success_count", 0))
        ),
        "merged_dynamic_candidate_count": len(merged.records),
        "merged_grasp_success_count": int(
            merged.summary.get("grasp_success_count", 0)
        ),
        "merged_candidate_ids": [int(value["candidate_id"]) for value in merged.records],
        "merged_candidate_sha256": canonical_sha256(
            [value["candidate_sha256"] for value in merged.records]
        ),
    }
    merge_path = (
        output / "dynamic" / "merged" / f"target_{target_success_count}.json"
    )
    _write_or_authenticate(merge_path, merge_report, resume=merge_path.exists())
    _commit_stage(
        output,
        f"recovery_dynamic_merge_{target_success_count}",
        stage_input={
            "recovery_input_sha256": recovery_input_sha256,
            "target_success_count": target_success_count,
            "base_candidate_sha256": canonical_sha256(
                [value["candidate_sha256"] for value in base.records]
            ),
            "local_candidate_sha256": canonical_sha256(
                []
                if local_dynamic is None
                else [value["candidate_sha256"] for value in local_dynamic.records]
            ),
        },
        artifacts=(merge_path,),
        summary={
            key: value
            for key, value in merge_report.items()
            if key not in {"merged_candidate_ids"}
        },
    )
    return merged, copy.deepcopy(merge_report)


def _run_downstream(
    static_records: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    workers: int,
    seed: int,
    target_success_count: int,
    experiment_id: str,
    recovery_input_sha256: str,
) -> dict[str, Any]:
    """Reuse the production dynamic/measured/manipulation evidence stages."""

    from .actual_contact_grasp_pose import (
        _publish_campaign_catalogs,
        _run_manipulation_stage,
        _run_measured_grasp_pose_finalization_stage,
    )

    dynamic, dynamic_breakdown = _run_dynamic_with_optional_local_refinement(
        static_records,
        output,
        workers=workers,
        seed=seed,
        target_success_count=target_success_count,
        recovery_input_sha256=recovery_input_sha256,
    )
    measured_stage = f"recovery_target_{target_success_count}"
    measured_report = (
        output / "dynamic" / "measured" / f"{measured_stage}_report.json"
    )
    measured_workers = workers
    if measured_report.is_file():
        previous = json.loads(measured_report.read_text(encoding="utf-8"))
        measured_workers = int(previous.get("workers", workers))
    measured = _run_measured_grasp_pose_finalization_stage(
        dynamic.records, output, stage=measured_stage, workers=measured_workers
    )
    _commit_stage(
        output,
        f"recovery_measured_grasp_pose_finalization_{target_success_count}",
        stage_input={
            "recovery_input_sha256": recovery_input_sha256,
            "target_success_count": target_success_count,
            "merged_dynamic_candidate_sha256": canonical_sha256(
                [value["candidate_sha256"] for value in dynamic.records]
            ),
            "workers": measured_workers,
        },
        artifacts=measured.artifacts,
        summary=measured.summary,
    )
    manipulation = _run_manipulation_stage(
        measured.records,
        output,
        target_success_count=target_success_count,
        seed=seed,
        workers=workers,
        stage="recovery",
        enable_local_refinement=True,
    )
    _commit_stage(
        output,
        f"recovery_manipulation_{target_success_count}",
        stage_input={
            "recovery_input_sha256": recovery_input_sha256,
            "target_success_count": target_success_count,
            "workers": workers,
        },
        artifacts=manipulation.artifacts,
        summary=manipulation.summary,
    )
    catalogs, artifacts = _publish_campaign_catalogs(
        output,
        measured.records,
        manipulation.records,
        target_success_count=target_success_count,
        experiment_id=experiment_id,
    )
    _commit_stage(
        output,
        f"recovery_catalogs_{target_success_count}",
        stage_input={
            "recovery_input_sha256": recovery_input_sha256,
            "target_success_count": target_success_count,
            "workers": workers,
        },
        artifacts=artifacts,
        summary={"catalogs": catalogs},
    )
    return {
        "dynamic_candidate_count": len(dynamic.records),
        "grasp_success_count": int(dynamic.summary.get("grasp_success_count", 0)),
        "dynamic_refinement": dynamic_breakdown,
        "measured_grasp_pose_count": len(measured.records),
        "manipulation_candidate_count": len(manipulation.records),
        "full_success_count": int(manipulation.summary.get("full_success_count", 0)),
        "catalogs": catalogs,
        "catalog_artifact_count": len(artifacts),
    }


def recovery_stop_reason(
    *,
    static_pass_count: int,
    grasp_success_count: int,
    measured_grasp_pose_count: int,
    full_success_count: int,
) -> str:
    if static_pass_count == 0:
        return "active_set_static_budget_exhausted"
    if grasp_success_count == 0:
        return "dynamic_grasp_not_verified"
    if measured_grasp_pose_count == 0:
        return "measured_grasp_pose_not_verified"
    if full_success_count == 0:
        return "manipulation_full_success_not_verified"
    return "manipulation_full_success_verified"


def run_recovery(
    parent_campaign: str | Path,
    output_dir: str | Path,
    *,
    workers: int,
    resume: bool,
    seed: int,
    target_success_count: int,
) -> dict[str, Any]:
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if target_success_count not in (1, 5):
        raise ValueError("target_success_count must be 1 or 5")
    parent, output = validate_recovery_paths(parent_campaign, output_dir)
    from .relative_wrist_pose_recovery_auth import authenticate_recovery_parent

    authenticated = authenticate_recovery_parent(parent)
    candidates = tuple(copy.deepcopy(dict(value)) for value in _mapping_value(authenticated, "candidates"))
    snapshot = copy.deepcopy(dict(_mapping_value(authenticated, "snapshot")))
    experiment_id = str(_mapping_value(authenticated, "experiment_id"))
    campaign_input_sha256 = str(_mapping_value(authenticated, "campaign_input_sha256"))
    manifest = build_recovery_base_manifest(
        parent=parent,
        experiment_id=experiment_id,
        campaign_input_sha256=campaign_input_sha256,
        source_bundle_sha256=str(
            _mapping_value(authenticated, "source_bundle_sha256")
        ),
        parent_snapshot_sha256=str(snapshot["snapshot_sha256"]),
        seed=seed,
    )
    run_manifest = build_recovery_run_manifest(
        recovery_input_sha256=manifest["recovery_input_sha256"],
        target_success_count=target_success_count,
        workers=workers,
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_or_authenticate(output / "recovery_manifest.json", manifest, resume=resume)
    _write_or_authenticate(output / "parent_snapshot.json", snapshot, resume=resume)
    run_manifest_path = (
        output
        / "run_manifests"
        / f"target_{target_success_count}"
        / f"workers_{workers}.json"
    )
    _write_or_authenticate(run_manifest_path, run_manifest, resume=resume)
    _commit_stage(
        output,
        "authenticated_parent",
        stage_input={"recovery_input_sha256": manifest["recovery_input_sha256"]},
        artifacts=(output / "parent_snapshot.json", output / "recovery_manifest.json"),
        summary={"authenticated_candidate_count": len(candidates)},
    )
    _commit_stage(
        output,
        f"run_target_{target_success_count}_workers_{workers}",
        stage_input={
            "recovery_input_sha256": manifest["recovery_input_sha256"],
            "target_success_count": target_success_count,
            "workers": workers,
        },
        artifacts=(run_manifest_path,),
        summary={
            "target_success_count": target_success_count,
            "workers": workers,
            "worker_count_changes_numerical_result": False,
        },
    )

    fresh_filter = _run_or_resume_fresh_full_scene_filter(
        candidates,
        output,
        recovery_input_sha256=manifest["recovery_input_sha256"],
        resume=resume,
    )
    selection = select_recovery_batches(fresh_filter.accepted)
    selection_report = {
        "relative_wrist_active_set_selection_schema_version": 1,
        "complete": True,
        "safe_candidate_count": selection.safe_candidate_count,
        "unsafe_candidate_count": selection.unsafe_candidate_count,
        "stored_v11_safe_count_before_fresh_filter": (
            fresh_filter.stored_v11_safe_count
        ),
        "fresh_full_scene_safe_count": (
            fresh_filter.fresh_full_scene_safe_count
        ),
        "fresh_full_scene_model_compile_count": fresh_filter.model_compile_count,
        "per_edge_available": {str(key): value for key, value in selection.per_edge_available.items()},
        "first_candidate_ids": [int(value["candidate_id"]) for value in selection.first],
        "second_candidate_ids": [int(value["candidate_id"]) for value in selection.second],
        "first_orbit_coverage": sorted({float(value.get("clockwise_orbit_deg", 0.0)) for value in selection.first}),
        "second_overlaps_first": bool(
            {int(value["candidate_id"]) for value in selection.first}
            & {int(value["candidate_id"]) for value in selection.second}
        ),
    }
    selection_path = output / "static" / "selection_report.json"
    _write_or_authenticate(selection_path, selection_report, resume=resume)
    _commit_stage(
        output,
        "safe_merit_selection",
        stage_input={"recovery_input_sha256": manifest["recovery_input_sha256"]},
        artifacts=(selection_path,),
        summary={"first_count": len(selection.first), "second_count": len(selection.second)},
    )

    first_path = output / "static" / "active_set_batch_1.json"
    first_stage_input = {
        "recovery_input_sha256": manifest["recovery_input_sha256"],
        "workers": workers,
    }
    first = _load_committed_batch(
        output,
        stage_name="active_set_batch_1",
        path=first_path,
        batch_index=1,
        stage_input=first_stage_input,
    )
    if first is None:
        first = _execute_jobs(selection.first, batch_index=1, workers=workers)
        _write_or_authenticate(
            first_path,
            {"complete": True, "batch": 1, "candidates": list(first)},
            resume=resume,
        )
    first_passes = tuple(value for value in first if bool(value.get("static_pass")))
    _commit_stage(
        output,
        "active_set_batch_1",
        stage_input=first_stage_input,
        artifacts=(first_path,),
        summary={"candidate_count": len(first), "static_pass_count": len(first_passes)},
    )

    # Recovery is budgeted per size: an easy small edge must not prevent a
    # second, disjoint attempt for a larger edge whose first twelve all miss.
    second_sources = select_second_batch_for_failed_edges(selection.second, first)
    second: tuple[dict[str, Any], ...] = ()
    if second_sources:
        second_path = output / "static" / "active_set_batch_2.json"
        second_stage_input = {
            "recovery_input_sha256": manifest["recovery_input_sha256"],
            "workers": workers,
        }
        loaded_second = _load_committed_batch(
            output,
            stage_name="active_set_batch_2",
            path=second_path,
            batch_index=2,
            stage_input=second_stage_input,
        )
        if loaded_second is None:
            second = _execute_jobs(second_sources, batch_index=2, workers=workers)
            _write_or_authenticate(
                second_path,
                {"complete": True, "batch": 2, "candidates": list(second)},
                resume=resume,
            )
        else:
            second = loaded_second
        _commit_stage(
            output,
            "active_set_batch_2",
            stage_input=second_stage_input,
            artifacts=(second_path,),
            summary={
                "candidate_count": len(second),
                "static_pass_count": sum(bool(value.get("static_pass")) for value in second),
            },
        )
    passes = tuple(value for value in (*first, *second) if bool(value.get("static_pass")))
    downstream = (
        _run_downstream(
            passes,
            output,
            workers=workers,
            seed=seed,
            target_success_count=target_success_count,
            experiment_id=experiment_id,
            recovery_input_sha256=manifest["recovery_input_sha256"],
        )
        if passes
        else {
            "dynamic_candidate_count": 0,
            "grasp_success_count": 0,
            "measured_grasp_pose_count": 0,
            "manipulation_candidate_count": 0,
            "full_success_count": 0,
            "catalogs": {},
        }
    )
    full_success_count = int(downstream.get("full_success_count", 0))
    grasp_success_count = int(downstream.get("grasp_success_count", 0))
    measured_count = int(downstream.get("measured_grasp_pose_count", 0))
    stop_reason = recovery_stop_reason(
        static_pass_count=len(passes),
        grasp_success_count=grasp_success_count,
        measured_grasp_pose_count=measured_count,
        full_success_count=full_success_count,
    )
    report = {
        "relative_wrist_active_set_recovery_report_schema_version": 1,
        "complete": True,
        "recovery_input_sha256": manifest["recovery_input_sha256"],
        "first_batch_candidate_count": len(first),
        "first_batch_static_pass_count": len(first_passes),
        "second_batch_executed": bool(second_sources),
        "second_batch_candidate_count": len(second),
        "static_pass_count": len(passes),
        "grasp_success_count": grasp_success_count,
        "full_success_count": full_success_count,
        "downstream": downstream,
        "stop_reason": stop_reason,
    }
    report_path = output / "recovery_reports" / f"target_{target_success_count}.json"
    _write_or_authenticate(report_path, report, resume=resume)
    _commit_stage(
        output,
        f"final_report_{target_success_count}",
        stage_input={
            "recovery_input_sha256": manifest["recovery_input_sha256"],
            "target_success_count": target_success_count,
            "workers": workers,
        },
        artifacts=(report_path,),
        summary={"static_pass_count": len(passes), **downstream},
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticated active-set recovery for a completed v11 campaign"
    )
    parser.add_argument("--parent-campaign", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--target-success-count", type=int, choices=(1, 5), default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_recovery(
        arguments.parent_campaign,
        arguments.output_dir,
        workers=arguments.workers,
        resume=arguments.resume,
        seed=arguments.seed,
        target_success_count=arguments.target_success_count,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if int(report["downstream"].get("full_success_count", 0)) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
