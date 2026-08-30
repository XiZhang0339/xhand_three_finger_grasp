"""Deterministic dynamic acquisition for schema-v9 actual grasp poses.

The static v9 screen proposes a *measured* contact shape and derives a
separated pre-contact shape.  This module is the dynamics boundary: it keeps
the proposed geometry and nominal joint qpos immutable, tries eight
controller-only preload/timing seeds, and promotes a candidate only when the
full free-body simulation reports ``stage_status.grasp_success``.

Every production trial is committed as one directory rename containing the
resolved config, complete summary and NPZ trace.  Candidate identities are
independent of worker completion order and resumed results are authenticated
before reuse.
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..actual_contact_capability import (
    LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID,
    resolve_actual_contact_definition,
)
from ..artifacts import file_sha256, write_json
from ..actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
    bind_candidate_result_semantic_sha256,
)
from ..config import ACTIVE_ACTUATORS, validate_config
from ..experiment import resolve_experiment
from ..grasp_pose import canonical_sha256, controller_id, grasp_pose_id


EXPERIMENT_ID = LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID
CAMPAIGN_KIND = "actual_contact_grasp_pose_dynamic_acquisition"
CANDIDATE_RESULT_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
CONTROLLER_SEED_COUNT = 8
THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"

FINGER_GROUP_ACTUATORS = {
    "thumb": ACTIVE_ACTUATORS[0:3],
    "index": ACTIVE_ACTUATORS[3:6],
    "mid": ACTIVE_ACTUATORS[6:8],
}
FINGER_GROUP_ORDER = tuple(FINGER_GROUP_ACTUATORS)
_CANDIDATE_ID_STRIDE = 16


CandidateExecutor = Callable[
    [Sequence[Mapping[str, Any]], int], Sequence[Mapping[str, Any]]
]


def _registered_close_duration_options(
    config: Mapping[str, Any],
) -> tuple[float, ...]:
    """Return the exact duration grid owned by the resolved experiment.

    Schema v9 used three durations, while the larger-object actual-contact
    campaign deliberately adds 1.75 s and 2.0 s probes.  Keeping this lookup
    registry-driven prevents a valid later campaign from being rejected by a
    controller-only helper.
    """

    definition = resolve_actual_contact_definition(
        config, context="actual-contact controller duration lookup"
    )
    protocol = definition.control_protocol
    if protocol is None or protocol.close_duration_options_s is None:
        raise ValueError(
            "actual-contact experiment must register close duration options"
        )
    return tuple(float(value) for value in protocol.close_duration_options_s)


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _mapping_at(value: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _json_safe(value: Any) -> Any:
    """Recursively normalize NumPy scalar evidence before strict JSON output."""

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _validate_v9_config(config: Mapping[str, Any]) -> None:
    resolve_actual_contact_definition(
        config, context="dynamic actual-contact acquisition"
    )
    nominal = _mapping_at(config, "grasp_pose", "nominal_joint_qpos_rad")
    control = _mapping_at(config, "control")
    if set(nominal) != set(ACTIVE_ACTUATORS):
        raise ValueError("grasp_pose.nominal_joint_qpos_rad must name eight actuators")
    for field in (
        "precontact_targets_rad",
        "contact_preload_targets_rad",
        "manipulation_delta_rad",
        "close_profile",
    ):
        values = control.get(field)
        if not isinstance(values, Mapping) or set(values) != set(ACTIVE_ACTUATORS):
            raise ValueError(f"control.{field} must name eight actuators")


def _group_profile(config: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    profile = _mapping_at(config, "control", "close_profile")
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    for group, actuators in FINGER_GROUP_ACTUATORS.items():
        group_starts = {
            float(_mapping_at(profile, name).get("start_fraction"))
            for name in actuators
        }
        group_ends = {
            float(_mapping_at(profile, name).get("end_fraction"))
            for name in actuators
        }
        if len(group_starts) != 1 or len(group_ends) != 1:
            raise ValueError(f"close_profile group {group!r} must be synchronized")
        starts[group] = group_starts.pop()
        ends[group] = group_ends.pop()
    return starts, ends


@dataclass(frozen=True, slots=True)
class ActualContactControllerSeed:
    """One controller-only variant around a frozen actual grasp pose."""

    seed_index: int
    kind: str
    contact_preload_targets_rad: tuple[float, ...]
    group_start_fraction: tuple[float, float, float]
    group_end_fraction: tuple[float, float, float]
    close_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.seed_index, int) or not 0 <= self.seed_index < 8:
            raise ValueError("seed_index must lie within [0, 7]")
        if len(self.contact_preload_targets_rad) != len(ACTIVE_ACTUATORS):
            raise ValueError("contact preload seed must contain eight values")
        if len(self.group_start_fraction) != 3 or len(self.group_end_fraction) != 3:
            raise ValueError("controller seed must contain three group intervals")
        values = (
            *self.contact_preload_targets_rad,
            *self.group_start_fraction,
            *self.group_end_fraction,
            self.close_s,
        )
        if not np.isfinite(values).all():
            raise ValueError("controller seed must contain only finite values")
        if self.close_s <= 0.0:
            raise ValueError("close_s must be positive")
        for start, end in zip(self.group_start_fraction, self.group_end_fraction):
            if not 0.0 <= start < end <= 1.0 or end - start < 0.05 - 1e-12:
                raise ValueError("controller close intervals must be valid")

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["contact_preload_targets_rad"] = {
            name: float(self.contact_preload_targets_rad[index])
            for index, name in enumerate(ACTIVE_ACTUATORS)
        }
        result["group_start_fraction"] = {
            group: float(self.group_start_fraction[index])
            for index, group in enumerate(FINGER_GROUP_ORDER)
        }
        result["group_end_fraction"] = {
            group: float(self.group_end_fraction[index])
            for index, group in enumerate(FINGER_GROUP_ORDER)
        }
        return result


def generate_controller_seeds(
    config: Mapping[str, Any],
    *,
    source_candidate_id: int = 0,
    seed: int = DEFAULT_SEED,
    count: int = CONTROLLER_SEED_COUNT,
) -> tuple[ActualContactControllerSeed, ...]:
    """Return eight evidence-backed, prefix-stable controller variants.

    A full free-body diagnostic of the aligned 67 mm pose showed the original
    controller contacting thumb/middle/index at steps 1110/1387/1797 and
    moving the cube 4.03 mm.  The direct pose itself has only 0.10--0.39 mm of
    distal penetration.  The table below therefore uses low preload and late,
    group-synchronized closure instead of adding an arbitrary high command.

    Preload is expressed as a group scale of ``nominal - precontact``.  A
    negative index scale slightly retreats the over-powered index while a
    modest middle preload stabilizes the second pad on the +X face.  Measured
    local contact fractions 0.496/0.488/0.755 were used to resolve the stored
    close-profile starts.  The first six designs are measured 1.5 s schedules;
    the last two retain the registered duration probes.  Nominal actual qpos
    and precontact qpos are never changed.
    """

    _validate_v9_config(config)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 8:
        raise ValueError("count must lie within [1, 8]")
    if not isinstance(source_candidate_id, int) or isinstance(source_candidate_id, bool):
        raise ValueError("source_candidate_id must be an integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    definition = resolve_experiment(dict(config))
    preload_bounds = definition.search_bounds.actuator_targets_rad
    nominal = _mapping_at(config, "grasp_pose", "nominal_joint_qpos_rad")
    precontact = _mapping_at(config, "control", "precontact_targets_rad")
    # Retain validation of the static candidate's source controller structure.
    _group_profile(config)
    close_options = _registered_close_duration_options(config)
    configured_options = tuple(
        float(value)
        for value in _mapping_at(config, "control_protocol").get(
            "close_duration_options_s", ()
        )
    )
    if configured_options != close_options:
        raise ValueError(
            "control_protocol close durations must match the registered experiment"
        )
    primary_close_s = min(close_options, key=lambda value: (abs(value - 1.5), value))
    remaining_close_options = tuple(
        value for value in close_options if not math.isclose(value, primary_close_s)
    )
    if len(remaining_close_options) < 2:
        raise ValueError(
            "actual-contact controller search requires at least three close durations"
        )
    duration_probes = tuple(sorted(remaining_close_options, reverse=True)[:2])
    if close_options == (1.0, 1.25, 1.5):
        duration_probe_labels = (
            "registered_duration_1p25_sync_probe",
            "registered_duration_1p00_sync_probe",
        )
    else:
        duration_probe_labels = (
            "registered_duration_probe_high",
            "registered_duration_probe_second",
        )

    # group preload scales, resolved group start/end fractions, registered
    # close duration, and explicit evidence label.  One deliberately less
    # synchronous force-balanced seed is retained because an independent full
    # dynamics run passed at 0.362 mm / 0.800 degrees pose drift.
    designs = (
        (
            (0.05, -0.22, 0.25),
            (0.7222222222222222, 0.6289062500000001, 0.6734693877551022),
            (1.0, 1.0, 1.0),
            primary_close_s,
            "measured_sync_best_margin",
        ),
        (
            (0.00, -0.22, 0.35),
            (0.7023809523809523, 0.6484375, 0.7142857142857143),
            (1.0, 1.0, 1.0),
            primary_close_s,
            "measured_sync_low_thumb",
        ),
        (
            (0.10, -0.22, 0.30),
            (0.7023809523809523, 0.6484375, 0.7142857142857143),
            (1.0, 1.0, 1.0),
            primary_close_s,
            "measured_sync_thumb_balance",
        ),
        (
            (0.00, -0.15, 0.40),
            (0.10, 0.16, 0.00),
            (1.0, 1.0, 0.72),
            primary_close_s,
            "independent_force_balance_pass",
        ),
        (
            (0.05, -0.22, 0.35),
            (0.7023809523809523, 0.70703125, 0.38775510204081626),
            (1.0, 1.0, 1.0),
            primary_close_s,
            "measured_common_late_mid_preload",
        ),
        (
            (0.05, -0.20, 0.30),
            (0.6428571428571428, 0.6484375, 0.2653061224489794),
            (1.0, 1.0, 1.0),
            primary_close_s,
            "measured_common_early_balance",
        ),
        (
            (0.05, -0.22, 0.25),
            (0.9007936507936507, 0.84375, 0.9183673469387754),
            (1.0, 1.0, 1.0),
            duration_probes[0],
            duration_probe_labels[0],
        ),
        (
            (0.05, -0.22, 0.25),
            (0.9206349206349206, 0.86328125, 0.9387755102040815),
            (1.0, 1.0, 1.0),
            duration_probes[1],
            duration_probe_labels[1],
        ),
    )
    generated: list[ActualContactControllerSeed] = []
    for row_index, (scales, starts, ends, close_s, kind) in enumerate(designs):
        scale_by_group = dict(zip(FINGER_GROUP_ORDER, scales))
        preload: list[float] = []
        for group in FINGER_GROUP_ORDER:
            scale = float(scale_by_group[group])
            for name in FINGER_GROUP_ACTUATORS[group]:
                lower, upper = preload_bounds[name]
                value = float(nominal[name]) + scale * (
                    float(nominal[name]) - float(precontact[name])
                )
                preload.append(float(np.clip(value, lower, upper)))
        generated.append(
            ActualContactControllerSeed(
                seed_index=row_index,
                kind=kind,
                contact_preload_targets_rad=tuple(preload),
                group_start_fraction=tuple(float(value) for value in starts),
                group_end_fraction=tuple(float(value) for value in ends),
                close_s=close_s,
            )
        )
    return tuple(generated[:count])


def materialize_controller_candidate(
    source: Mapping[str, Any],
    spec: ActualContactControllerSeed,
    *,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> dict[str, Any]:
    """Apply a controller seed while preserving the actual grasp-pose ID."""

    if "config" not in source or "candidate_id" not in source:
        raise ValueError("source must contain candidate_id and config")
    source_id = int(source["candidate_id"])
    if source_id < 0:
        raise ValueError("source candidate_id must be non-negative")
    source_config = copy.deepcopy(dict(source["config"]))
    _validate_v9_config(source_config)
    frozen_nominal = copy.deepcopy(
        source_config["grasp_pose"]["nominal_joint_qpos_rad"]
    )
    frozen_pose_id = grasp_pose_id(source_config)
    config = copy.deepcopy(source_config)
    config["control"]["contact_preload_targets_rad"] = {
        name: float(spec.contact_preload_targets_rad[index])
        for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    config["control"]["manipulation_delta_rad"] = {
        name: 0.0 for name in ACTIVE_ACTUATORS
    }
    starts = dict(zip(FINGER_GROUP_ORDER, spec.group_start_fraction))
    ends = dict(zip(FINGER_GROUP_ORDER, spec.group_end_fraction))
    config["control"]["close_profile"] = {
        name: {
            "start_fraction": float(starts[group]),
            "end_fraction": float(ends[group]),
        }
        for group in FINGER_GROUP_ORDER
        for name in FINGER_GROUP_ACTUATORS[group]
    }
    config["control_protocol"]["close_s"] = float(spec.close_s)
    dynamic_id = source_id * _CANDIDATE_ID_STRIDE + spec.seed_index
    metadata = config.setdefault("candidate_metadata", {})
    metadata.update(
        {
            "campaign_kind": CAMPAIGN_KIND,
            "stage": "dynamic_grasp_acquisition",
            "candidate_id": dynamic_id,
            "source_candidate_id": source_id,
            "controller_seed": spec.as_dict(),
            "static_filter_is_success_evidence": False,
            "manipulation_delta_is_zero": True,
        }
    )
    if config["grasp_pose"]["nominal_joint_qpos_rad"] != frozen_nominal:
        raise RuntimeError("controller materialization changed nominal actual qpos")
    if grasp_pose_id(config) != frozen_pose_id:
        raise RuntimeError("controller materialization changed grasp_pose_id")
    metadata["grasp_pose_id"] = frozen_pose_id
    metadata["controller_id"] = controller_id(config)
    if validator is not None:
        validator(config)
    return {
        "campaign_kind": CAMPAIGN_KIND,
        "stage": "dynamic_grasp_acquisition",
        "candidate_id": dynamic_id,
        "source_candidate_id": source_id,
        "controller_seed_index": spec.seed_index,
        "grasp_pose_id": frozen_pose_id,
        "controller_id": metadata["controller_id"],
        "candidate_sha256": canonical_sha256(config),
        "config": config,
    }


def expand_controller_candidates(
    sources: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    controller_seed_count: int = CONTROLLER_SEED_COUNT,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    """Expand retained static poses into deterministic controller candidates."""

    source_ids = [int(value["candidate_id"]) for value in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source candidate IDs must be unique")
    records: list[dict[str, Any]] = []
    for source in sorted(sources, key=lambda value: int(value["candidate_id"])):
        specs = generate_controller_seeds(
            source["config"],
            source_candidate_id=int(source["candidate_id"]),
            seed=seed,
            count=controller_seed_count,
        )
        records.extend(
            materialize_controller_candidate(source, spec, validator=validator)
            for spec in specs
        )
    identifiers = [int(value["candidate_id"]) for value in records]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("dynamic candidate ID collision")
    return tuple(sorted(records, key=lambda value: int(value["candidate_id"])))


def grasp_stage_succeeded(record: Mapping[str, Any]) -> bool:
    """Return only the authoritative stage-status grasp decision."""

    stage = _mapping_at(record, "summary", "stage_status")
    return bool(stage.get("grasp_success", False))


def _maximum_closure_p95(metrics: Mapping[str, Any]) -> float:
    closure = _mapping_at(metrics, "closure_alignment")
    close = _mapping_at(closure, "close") or closure
    direct = _finite(
        close.get("max_p95_angle_deg", close.get("worst_p95_angle_deg")),
        math.inf,
    )
    per_finger = _mapping_at(close, "per_finger")
    values = []
    for finger in FINGER_GROUP_ORDER:
        finger_metrics = _mapping_at(per_finger, finger)
        values.append(
            _finite(
                finger_metrics.get(
                    "angle_p95_deg", finger_metrics.get("p95_angle_deg")
                ),
                math.inf,
            )
        )
    finite_values = [value for value in (direct, *values) if math.isfinite(value)]
    return max(finite_values, default=math.inf)


def dynamic_grasp_rank_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract JSON-safe hard decisions and soft margins for ranking."""

    summary = _mapping_at(record, "summary")
    metrics = _mapping_at(summary, "metrics")
    actual = _mapping_at(metrics, "actual_grasp_pose")
    actual_metrics = _mapping_at(actual, "metrics")
    thumb = _finite(actual_metrics.get("thumb_actual_median_rad"), math.nan)
    maximum_error = _finite(
        actual_metrics.get("maximum_nominal_joint_error_rad"), math.nan
    )
    maximum_span = _finite(
        actual_metrics.get("maximum_joint_stability_span_rad"), math.nan
    )
    config = _mapping_at(record, "config")
    grasp_settings = _mapping_at(config, "grasp_pose")
    error_limit = _finite(grasp_settings.get("max_nominal_joint_error_rad"), 0.04)
    span_limit = _finite(grasp_settings.get("max_joint_stability_span_rad"), 0.03)
    if math.isfinite(maximum_error) and math.isfinite(maximum_span):
        stability_margin = min(
            (error_limit - maximum_error) / error_limit,
            (span_limit - maximum_span) / span_limit,
        )
    else:
        stability_margin = None

    pose = _mapping_at(metrics, "pose_preservation")
    pose_settings = _mapping_at(config, "pose_preservation")
    translation = _finite(pose.get("max_translation_m"), math.nan)
    orientation = _finite(pose.get("max_orientation_drift_deg"), math.nan)
    translation_limit = _finite(pose_settings.get("max_translation_m"), 0.0005)
    orientation_limit = _finite(
        pose_settings.get("max_orientation_drift_deg"), 1.0
    )
    if math.isfinite(translation) and math.isfinite(orientation):
        pose_margin = min(
            (translation_limit - translation) / translation_limit,
            (orientation_limit - orientation) / orientation_limit,
        )
    else:
        pose_margin = None

    closure_p95 = _maximum_closure_p95(metrics)
    closure_limit = _finite(
        _mapping_at(config, "closure_alignment").get(
            "dynamic_p95_max_angle_deg"
        ),
        30.0,
    )
    closure_margin = (
        None
        if not math.isfinite(closure_p95)
        else (closure_limit - closure_p95) / closure_limit
    )
    force = _finite(metrics.get("peak_total_distal_contact_force_n"), math.nan)
    saturation = _finite(metrics.get("actuator_saturation_fraction"), math.nan)
    return {
        "grasp_success": grasp_stage_succeeded(record),
        "thumb_actual_median_rad": thumb if math.isfinite(thumb) else None,
        "thumb_distance_from_1p50_rad": (
            abs(thumb - 1.50) if math.isfinite(thumb) else None
        ),
        "stability_min_normalized_margin": stability_margin,
        "pose_min_normalized_margin": pose_margin,
        "closure_max_p95_angle_deg": (
            closure_p95 if math.isfinite(closure_p95) else None
        ),
        "closure_normalized_margin": closure_margin,
        "peak_total_distal_contact_force_n": (
            force if math.isfinite(force) else None
        ),
        "actuator_saturation_fraction": (
            saturation if math.isfinite(saturation) else None
        ),
    }


def dynamic_grasp_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    evidence = dynamic_grasp_rank_evidence(record)
    return (
        not bool(evidence["grasp_success"]),
        _finite(evidence["thumb_distance_from_1p50_rad"], math.inf),
        -_finite(evidence["stability_min_normalized_margin"], -math.inf),
        -_finite(evidence["pose_min_normalized_margin"], -math.inf),
        -_finite(evidence["closure_normalized_margin"], -math.inf),
        _finite(evidence["peak_total_distal_contact_force_n"], math.inf),
        _finite(evidence["actuator_saturation_fraction"], math.inf),
        int(record.get("candidate_id", 2**63 - 1)),
    )


def rank_dynamic_grasp_results(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    materialized = [copy.deepcopy(dict(record)) for record in records]
    materialized.sort(key=dynamic_grasp_rank)
    return tuple(materialized)


def promotable_grasp_results(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return ranked candidates whose authoritative grasp stage passed."""

    return rank_dynamic_grasp_results(
        record for record in records if grasp_stage_succeeded(record)
    )


def _evaluate_dynamic_candidate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    from ..simulation import run_simulation

    output = Path(str(job["output_directory"]))
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}."
    ) as staging:
        staging_path = Path(staging)
        config_path = staging_path / "resolved_config.json"
        trace_path = staging_path / "trace.npz"
        write_json(config_path, job["config"])
        summary = _json_safe(
            run_simulation(copy.deepcopy(dict(job["config"])), trace_path=trace_path)
        )
        if not trace_path.is_file():
            raise RuntimeError("run_simulation did not create trace.npz")
        actual_metrics = _mapping_at(summary, "metrics", "actual_grasp_pose")
        if not actual_metrics:
            raise RuntimeError("schema-v9 simulation summary has no actual grasp metrics")
        expected_pose_id = grasp_pose_id(job["config"])
        expected_controller_id = controller_id(job["config"])
        if str(job["grasp_pose_id"]) != expected_pose_id:
            raise RuntimeError("dynamic job grasp_pose_id does not match its config")
        if str(job["controller_id"]) != expected_controller_id:
            raise RuntimeError("dynamic job controller_id does not match its config")
        provisional = {
            **copy.deepcopy(dict(job)),
            "summary": summary,
        }
        provisional.pop("output_directory", None)
        evidence = dynamic_grasp_rank_evidence(provisional)
        payload = {
            "candidate_result_schema_version": CANDIDATE_RESULT_SCHEMA_VERSION,
            "complete": True,
            "campaign_kind": CAMPAIGN_KIND,
            "stage": "dynamic_grasp_acquisition",
            "candidate_id": int(job["candidate_id"]),
            "source_candidate_id": int(job["source_candidate_id"]),
            "controller_seed_index": int(job["controller_seed_index"]),
            "candidate_sha256": str(job["candidate_sha256"]),
            "grasp_pose_id": expected_pose_id,
            "controller_id": expected_controller_id,
            "grasp_success": bool(evidence["grasp_success"]),
            "classification": (
                "actual_contact_grasp_pose_acquired"
                if evidence["grasp_success"]
                else "actual_contact_grasp_pose_near_miss"
            ),
            "rank_evidence": evidence,
            "actual_grasp_pose": copy.deepcopy(dict(actual_metrics)),
            "summary": summary,
            "artifacts": {
                "resolved_config": "resolved_config.json",
                "trace": "trace.npz",
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "trace": file_sha256(trace_path),
                },
            },
        }
        payload = bind_candidate_result_semantic_sha256(payload)
        write_json(staging_path / "result.json", payload)
        staging_path.rename(output)
    return {
        **payload,
        "config": copy.deepcopy(dict(job["config"])),
        "artifact_directory": str(job["artifact_directory"]),
        "reused": False,
    }


def run_dynamic_candidate_jobs(
    jobs: Sequence[Mapping[str, Any]], workers: int
) -> tuple[dict[str, Any], ...]:
    """Execute real v9 simulations using deterministic spawn map ordering."""

    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not jobs:
        return ()
    if workers == 1:
        results = [_evaluate_dynamic_candidate_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            results = list(
                executor.map(_evaluate_dynamic_candidate_job, jobs, chunksize=1)
            )
    results.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(results)


def _load_reusable_candidate(job: Mapping[str, Any]) -> dict[str, Any] | None:
    directory = Path(str(job["output_directory"]))
    if not directory.exists():
        return None
    result_path = directory / "result.json"
    config_path = directory / "resolved_config.json"
    trace_path = directory / "trace.npz"
    if not all(path.is_file() for path in (result_path, config_path)):
        raise RuntimeError(f"incomplete dynamic candidate directory: {directory}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    authenticate_candidate_result_semantic_sha256(payload, source=result_path)
    if payload.get("complete") is not True:
        raise RuntimeError(f"dynamic candidate is not complete: {result_path}")
    if int(payload.get("candidate_id", -1)) != int(job["candidate_id"]):
        raise RuntimeError(f"dynamic candidate ID mismatch: {result_path}")
    if payload.get("candidate_sha256") != job["candidate_sha256"]:
        raise RuntimeError(f"dynamic candidate semantic hash mismatch: {result_path}")
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    if canonical_sha256(persisted) != job["candidate_sha256"]:
        raise RuntimeError(f"persisted dynamic config changed: {config_path}")
    if payload.get("grasp_pose_id") != grasp_pose_id(persisted):
        raise RuntimeError(f"persisted grasp_pose_id changed: {result_path}")
    if payload.get("controller_id") != controller_id(persisted):
        raise RuntimeError(f"persisted controller_id changed: {result_path}")
    artifacts = _mapping_at(payload, "artifacts")
    hashes = _mapping_at(artifacts, "sha256")
    if hashes.get("resolved_config") != file_sha256(config_path):
        raise RuntimeError(f"dynamic config file hash mismatch: {config_path}")
    trace_retained = bool(artifacts.get("trace_retained", True))
    if trace_retained:
        if not trace_path.is_file() or hashes.get("trace") != file_sha256(trace_path):
            raise RuntimeError(f"dynamic trace file hash mismatch: {trace_path}")
    elif trace_path.exists():
        raise RuntimeError(f"compacted candidate unexpectedly retained trace: {trace_path}")
    return {
        **payload,
        "config": copy.deepcopy(dict(job["config"])),
        "artifact_directory": str(job["artifact_directory"]),
        "reused": True,
    }


def run_or_resume_dynamic_candidates(
    candidates: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    workers: int = 1,
    resume: bool = False,
    executor: CandidateExecutor = run_dynamic_candidate_jobs,
) -> tuple[dict[str, Any], ...]:
    """Run or authenticate an expanded candidate set in ID-sorted order."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    output = Path(output_dir).expanduser().resolve()
    identifiers = [int(value["candidate_id"]) for value in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate IDs must be unique")
    jobs: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda value: int(value["candidate_id"])):
        relative = Path("candidates") / f"candidate_{int(candidate['candidate_id'])}"
        jobs.append(
            {
                **copy.deepcopy(dict(candidate)),
                "artifact_directory": str(relative),
                "output_directory": str(output / relative),
            }
        )
    complete: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for job in jobs:
        reusable = _load_reusable_candidate(job) if resume else None
        if reusable is None:
            pending.append(job)
        else:
            complete.append(reusable)
    executed = tuple(executor(tuple(pending), workers)) if pending else ()
    expected = {int(job["candidate_id"]): job for job in pending}
    observed = {int(value.get("candidate_id", -1)) for value in executed}
    if observed != set(expected) or len(executed) != len(expected):
        raise RuntimeError("dynamic executor did not preserve candidate IDs")
    for raw in executed:
        result = copy.deepcopy(dict(raw))
        job = expected[int(result["candidate_id"])]
        if result.get("candidate_sha256") != job["candidate_sha256"]:
            raise RuntimeError("dynamic executor rebound a different candidate config")
        if result.get("grasp_pose_id") != job["grasp_pose_id"]:
            raise RuntimeError("dynamic executor rebound a different grasp pose")
        if result.get("controller_id") != job["controller_id"]:
            raise RuntimeError("dynamic executor rebound a different controller")
        result.setdefault("config", copy.deepcopy(job["config"]))
        result.setdefault("artifact_directory", str(job["artifact_directory"]))
        complete.append(result)
    complete.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(complete)


def compact_dynamic_candidate_artifacts(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    retain_failure_trace_count: int = 24,
) -> tuple[dict[str, Any], ...]:
    """Keep traces for every success and only the best N failed candidates.

    Dynamic trials initially commit a complete trace atomically.  Once the
    globally deterministic rank is known, low-ranked failures retain their
    config, summary, result and the trace hash observed at evaluation, but the
    bulky NPZ is removed.  Such compact records remain authenticated and
    reusable on ``--resume``.
    """

    if (
        not isinstance(retain_failure_trace_count, int)
        or isinstance(retain_failure_trace_count, bool)
        or retain_failure_trace_count < 0
    ):
        raise ValueError("retain_failure_trace_count must be a non-negative integer")
    output = Path(output_dir).expanduser().resolve()
    ranked_failures = [
        value
        for value in rank_dynamic_grasp_results(records)
        if not grasp_stage_succeeded(value)
    ]
    retained_ids = {
        int(value["candidate_id"])
        for value in ranked_failures[:retain_failure_trace_count]
    }
    retained_ids.update(
        int(value["candidate_id"])
        for value in records
        if grasp_stage_succeeded(value)
    )
    compacted: list[dict[str, Any]] = []
    for raw in records:
        record = copy.deepcopy(dict(raw))
        identifier = int(record["candidate_id"])
        relative = Path(
            str(
                record.get(
                    "artifact_directory",
                    Path("candidates") / f"candidate_{identifier}",
                )
            )
        )
        directory = output / relative
        result_path = directory / "result.json"
        trace_path = directory / "trace.npz"
        if identifier not in retained_ids and result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            artifacts = payload.setdefault("artifacts", {})
            hashes = artifacts.setdefault("sha256", {})
            observed_hash = hashes.pop("trace", None)
            if trace_path.is_file():
                actual_hash = file_sha256(trace_path)
                if observed_hash is not None and observed_hash != actual_hash:
                    raise RuntimeError(
                        f"refusing to compact changed dynamic trace: {trace_path}"
                    )
                trace_path.unlink()
                observed_hash = actual_hash
            elif observed_hash is None:
                # Trace compaction is intentionally irreversible.  Re-running
                # an earlier subset after a later global compaction must keep
                # the evaluation digest already persisted by that later pass.
                observed_hash = artifacts.get("trace_sha256_at_evaluation")
            if not isinstance(observed_hash, str):
                raise RuntimeError(
                    f"compacted dynamic candidate lost its trace digest: {result_path}"
                )
            artifacts["trace"] = None
            artifacts["trace_retained"] = False
            artifacts["trace_sha256_at_evaluation"] = observed_hash
            write_json(result_path, payload)
            record.update(payload)
        elif result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            artifacts = payload.setdefault("artifacts", {})
            hashes = artifacts.setdefault("sha256", {})
            if trace_path.is_file():
                observed_hash = file_sha256(trace_path)
                if hashes.get("trace") != observed_hash:
                    raise RuntimeError(
                        f"retained dynamic trace hash mismatch: {trace_path}"
                    )
                artifacts["trace"] = trace_path.name
                artifacts["trace_retained"] = True
            elif artifacts.get("trace_retained") is False:
                if grasp_stage_succeeded(payload):
                    raise RuntimeError(
                        f"successful dynamic candidate lost its trace: {trace_path}"
                    )
                if not isinstance(
                    artifacts.get("trace_sha256_at_evaluation"), str
                ):
                    raise RuntimeError(
                        f"compacted dynamic candidate lost its trace digest: {result_path}"
                    )
                # A failed trace compacted by a later superset cannot be
                # resurrected merely because this subset would rank it in its
                # local top-N.  Preserve the authenticated compacted state.
            else:
                raise RuntimeError(
                    f"dynamic candidate declares a missing retained trace: {trace_path}"
                )
            write_json(result_path, payload)
            record.update(payload)
        compacted.append(record)
    compacted.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(compacted)


def run_actual_contact_dynamic_grasp_candidates(
    static_candidates: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    workers: int = 1,
    resume: bool = False,
    seed: int = DEFAULT_SEED,
    controller_seed_count: int = CONTROLLER_SEED_COUNT,
    retain_failure_trace_count: int = 24,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
    executor: CandidateExecutor = run_dynamic_candidate_jobs,
) -> tuple[dict[str, Any], ...]:
    """Expand retained static poses, execute dynamics, and return all results."""

    candidates = expand_controller_candidates(
        static_candidates,
        seed=seed,
        controller_seed_count=controller_seed_count,
        validator=validator,
    )
    results = run_or_resume_dynamic_candidates(
        candidates,
        output_dir,
        workers=workers,
        resume=resume,
        executor=executor,
    )
    return compact_dynamic_candidate_artifacts(
        results,
        output_dir,
        retain_failure_trace_count=retain_failure_trace_count,
    )


__all__ = [
    "CAMPAIGN_KIND",
    "CANDIDATE_RESULT_SCHEMA_VERSION",
    "CONTROLLER_SEED_COUNT",
    "DEFAULT_SEED",
    "EXPERIMENT_ID",
    "ActualContactControllerSeed",
    "CandidateExecutor",
    "dynamic_grasp_rank",
    "dynamic_grasp_rank_evidence",
    "compact_dynamic_candidate_artifacts",
    "expand_controller_candidates",
    "generate_controller_seeds",
    "grasp_stage_succeeded",
    "materialize_controller_candidate",
    "promotable_grasp_results",
    "rank_dynamic_grasp_results",
    "run_actual_contact_dynamic_grasp_candidates",
    "run_dynamic_candidate_jobs",
    "run_or_resume_dynamic_candidates",
]
