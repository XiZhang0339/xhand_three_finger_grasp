"""Dynamic promotion of schema-v8 normal-aligned static pose candidates.

The preceding static search deliberately never steps MuJoCo.  This module is
the auditable boundary at which its retained configurations become real
``SETTLE+CLOSE+VERIFY`` trials.  Geometry and controller identities use
separate SHA-256 projections, candidate IDs are allocation-order independent,
and every real trial is persisted by the same spawn/resume runner used by the
old-pose rescue campaign.

The default registered budget is:

* eight controller seeds for every retained static pose;
* 64 joint pose/controller perturbations around each of the best 20 poses;
* locked-1-ms replay of the best 24 dynamic candidates.

The budget is injectable so the orchestration can be tested without replacing
the real ``run_simulation`` production path.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import file_sha256, json_text, write_json
from ..config import ACTIVE_ACTUATORS, load_config, validate_config
from ..experiment import resolve_experiment
from .normal_aligned_pose_search import (
    CAMPAIGN_KIND as STATIC_CAMPAIGN_KIND,
    controller_id_for_config,
    pose_id_for_config,
)
from .normal_aligned_smooth_lift import (
    CAMPAIGN_KIND,
    CLOSE_DURATION_OPTIONS_S,
    CLOSE_GROUP_ACTUATORS,
    CLOSE_GROUP_ORDER,
    DEFAULT_SEED,
    EXPERIMENT_ID,
    NON_THUMB_TARGET_ACTUATORS,
    THUMB_BEND_ACTUATOR,
    CandidateExecutor,
    RescueControlSpec,
    _latin_hypercube,
    _run_or_resume,
    candidate_rank_evidence,
    generate_control_specs,
    run_candidate_jobs,
)
from .pose_preserving_seed_campaign import canonical_sha256


DYNAMIC_CAMPAIGN_KIND = "normal_aligned_new_pose_dynamic_promotion"
DYNAMIC_REPORT_SCHEMA_VERSION = 1
STATIC_REPORT_SCHEMA_VERSION = 1
DEFAULT_TEMPLATE = Path(
    "grasp_configs/"
    "left_opposed_face_palm_down_high_thumb_normal_aligned_"
    "smooth_vertical_lift.json"
)
DEFAULT_STATIC_REPORT = Path(
    "artifacts/"
    "left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/"
    "tune/new_pose_static/search_report.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/"
    "left_opposed_face_palm_down_high_thumb_normal_aligned_smooth_vertical_lift/"
    "tune/new_pose_dynamic"
)

_INITIAL_BASE = 92_000_000_000_000
_LOCAL_BASE = 93_000_000_000_000
_EXACT_BASE = 94_000_000_000_000
_POSE_STRIDE = 10_000
LOCAL_PARENT_THUMB_TARGETS_RAD = (1.25, 1.30, 1.35, 1.40, 1.45)
LOCAL_PARENT_PER_THUMB_QUOTA = 2


@dataclass(frozen=True, slots=True)
class DynamicPromotionBudget:
    """Registered new-pose dynamics budget, with small-test injection support."""

    controller_seeds_per_pose: int = 8
    local_pose_count: int = 20
    local_refine_per_pose: int = 64
    exact_candidate_count: int = 24

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"budget.{name} must be a positive integer")
        if self.controller_seeds_per_pose != 8:
            raise ValueError(
                "controller_seeds_per_pose must remain original + validated67 + "
                "analytic + five deterministic LHS seeds"
            )
        if self.local_refine_per_pose >= _POSE_STRIDE:
            raise ValueError("local_refine_per_pose must be smaller than the pose stride")

    @property
    def registered(self) -> bool:
        return self == REGISTERED_DYNAMIC_BUDGET


REGISTERED_DYNAMIC_BUDGET = DynamicPromotionBudget()


def _safe_path(root: Path, value: Any, label: str) -> Path:
    raw = Path(str(value))
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    # Absolute result paths are legitimate, but relative paths must never
    # escape the static report directory.
    if not raw.is_absolute() and path != root and root not in path.parents:
        raise ValueError(f"{label} escapes the static report directory")
    return path


def _static_record_rank(value: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = value.get("static_metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    angles = tuple(float(number) for number in metrics.get("closure_angle_deg", ()))
    gaps = tuple(float(number) for number in metrics.get("target_signed_gap_m", ()))
    gap_violation = sum(
        max(0.0, -0.0005 - number) + max(0.0, number - 0.003)
        for number in gaps
    )
    minimum_gap_value = metrics.get("minimum_active_nondistal_gap_m")
    try:
        minimum_gap = float(minimum_gap_value)
    except (TypeError, ValueError):
        minimum_gap = -math.inf
    if not math.isfinite(minimum_gap):
        minimum_gap = -math.inf
    try:
        height_spread = float(metrics.get("contact_height_spread_m"))
    except (TypeError, ValueError):
        height_spread = math.inf
    if not math.isfinite(height_spread):
        height_spread = math.inf
    return (
        not bool(metrics.get("static_geometry_pass", value.get("static_pass", False))),
        int(metrics.get("missing_target_witness_count", 99)),
        int(metrics.get("off_target_penetrating_count", 99)),
        max(0.0, -minimum_gap),
        gap_violation,
        max(angles, default=math.inf),
        height_spread,
        int(value.get("candidate_id", 2**63 - 1)),
    )


def _finite_rank_metric(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def new_pose_grasp_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Rank new-pose grasp evidence before any soft angle optimization.

    The canonical rescue rank intentionally remains available to the old-pose
    campaign.  New geometry promotion has a different failure mode: a very
    small closure angle can accompany centimetres of object motion.  Treat the
    closure hard gate, longest VERIFY run, and pose margin as prerequisites so
    such a controller cannot displace a pose-preserving near miss during local
    parent selection or exact replay.
    """

    evidence = candidate_rank_evidence(record)
    summary = record.get("summary", {})
    metrics = summary.get("metrics", {}) if isinstance(summary, Mapping) else {}
    if not isinstance(metrics, Mapping):
        metrics = {}
    force = _finite_rank_metric(
        metrics.get("peak_total_distal_contact_force_n"), math.inf
    )
    saturation = _finite_rank_metric(
        metrics.get("actuator_saturation_fraction"), math.inf
    )
    return (
        not bool(evidence["rescue_success"]),
        not bool(evidence["evidence_complete"]),
        not bool(record.get("acquisition_success", False)),
        not bool(record.get("pose_preservation_success", False)),
        not bool(evidence["closure_alignment_passed"]),
        -int(evidence["verify_gate_steps"]),
        -_finite_rank_metric(evidence["pose_min_normalized_margin"], -math.inf),
        _finite_rank_metric(evidence["max_closure_p95_angle_deg"], math.inf),
        force,
        saturation,
        int(record.get("candidate_id", 2**63 - 1)),
    )


def rank_new_pose_grasp_results(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return a worker-order-independent ranking for new-pose promotion."""

    materialized = [copy.deepcopy(dict(record)) for record in records]
    materialized.sort(key=new_pose_grasp_rank)
    return tuple(materialized)


def _retained_from_static_report(
    payload: Mapping[str, Any], report_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the canonical per-cell result files and a few legacy inline forms."""

    root = report_path.parent.resolve()
    cell_bindings = payload.get("cell_results")
    records: list[dict[str, Any]] = []
    authenticated_cells: list[dict[str, Any]] = []
    if isinstance(cell_bindings, list):
        for raw_binding in cell_bindings:
            if not isinstance(raw_binding, Mapping):
                raise ValueError("static cell_results entries must be objects")
            if not raw_binding.get("result_path"):
                raise ValueError("static cell result binding is missing result_path")
            result_path = _safe_path(root, raw_binding["result_path"], "result_path")
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            expected_hash = raw_binding.get("result_sha256")
            actual_hash = file_sha256(result_path)
            if expected_hash is not None and str(expected_hash) != actual_hash:
                raise ValueError(f"static cell result hash mismatch: {result_path}")
            cell = json.loads(result_path.read_text(encoding="utf-8"))
            retained = cell.get("retained")
            if not isinstance(retained, list):
                raise ValueError(f"static cell result has no retained list: {result_path}")
            if int(raw_binding.get("retained_count", len(retained))) != len(retained):
                raise ValueError(f"static retained count mismatch: {result_path}")
            records.extend(copy.deepcopy(dict(value)) for value in retained)
            authenticated_cells.append(
                {
                    "cell_index": int(raw_binding["cell_index"]),
                    "result_path": str(result_path),
                    "result_sha256": actual_hash,
                    "retained_count": len(retained),
                }
            )
        return records, authenticated_cells

    # Compatibility for unit-size and early static runners that embedded the
    # per-cell objects directly.  Production output uses ``cell_results``.
    containers = payload.get("results", payload.get("cells", ()))
    if isinstance(containers, list):
        for index, cell in enumerate(containers):
            if not isinstance(cell, Mapping):
                raise ValueError("inline static cell must be an object")
            retained = cell.get("retained", ())
            if not isinstance(retained, (list, tuple)):
                raise ValueError("inline static retained must be a list")
            records.extend(copy.deepcopy(dict(value)) for value in retained)
            authenticated_cells.append(
                {
                    "cell_index": int(cell.get("cell_index", index)),
                    "result_path": None,
                    "result_sha256": canonical_sha256(cell),
                    "retained_count": len(retained),
                }
            )
    elif isinstance(payload.get("retained"), list):
        records.extend(copy.deepcopy(dict(value)) for value in payload["retained"])
        authenticated_cells.append(
            {
                "cell_index": 0,
                "result_path": None,
                "result_sha256": canonical_sha256(payload["retained"]),
                "retained_count": len(records),
            }
        )
    else:
        raise ValueError("static search report contains no retained candidates")
    return records, authenticated_cells


def load_static_pose_manifest(
    static_report_path: str | Path,
) -> dict[str, Any]:
    """Authenticate and pose-deduplicate static candidates for dynamics."""

    path = Path(static_report_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise ValueError("static search report must declare complete=true")
    experiment = payload.get("experiment_id")
    if experiment is not None and experiment != EXPERIMENT_ID:
        raise ValueError("static search report belongs to the wrong experiment")
    campaign = payload.get("campaign_kind")
    if campaign is not None and campaign != STATIC_CAMPAIGN_KIND:
        raise ValueError("static search report has the wrong campaign kind")
    raw, cells = _retained_from_static_report(payload, path)
    if not raw:
        raise ValueError("static search retained no candidates for dynamics")

    authenticated: list[dict[str, Any]] = []
    for value in raw:
        config = value.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("retained static candidate must embed its resolved config")
        candidate = copy.deepcopy(dict(config))
        validate_config(candidate)
        if int(candidate.get("schema_version", 0)) != 8:
            raise ValueError("retained static candidate must use schema version 8")
        if candidate.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError("retained static candidate has the wrong experiment")
        if any(
            float(number) != 0.0
            for number in candidate["control"]["manipulation_delta_rad"].values()
        ):
            raise ValueError("new-pose acquisition requires zero manipulation delta")
        pose_identifier = pose_id_for_config(candidate)
        control_identifier = controller_id_for_config(candidate)
        if value.get("pose_id") not in (None, pose_identifier):
            raise ValueError("static pose hash no longer matches its config")
        if value.get("controller_id") not in (None, control_identifier):
            raise ValueError("static controller hash no longer matches its config")
        semantic_hash = canonical_sha256(candidate)
        if value.get("candidate_sha256") not in (None, semantic_hash):
            raise ValueError("static candidate semantic hash mismatch")
        authenticated.append(
            {
                **copy.deepcopy(dict(value)),
                "candidate_id": int(value["candidate_id"]),
                "candidate_sha256": semantic_hash,
                "pose_id": pose_identifier,
                "controller_id": control_identifier,
                "config": candidate,
            }
        )

    # A static scan may retain the same geometry from neighboring controller
    # samples.  Run dynamics once per pose, starting from the best static one.
    best_by_pose: dict[str, dict[str, Any]] = {}
    for value in authenticated:
        identifier = str(value["pose_id"])
        previous = best_by_pose.get(identifier)
        if previous is None or _static_record_rank(value) < _static_record_rank(previous):
            best_by_pose[identifier] = value
    poses = sorted(best_by_pose.values(), key=_static_record_rank)
    return {
        "static_manifest_schema_version": STATIC_REPORT_SCHEMA_VERSION,
        "campaign_kind": DYNAMIC_CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        "source_static_report": str(path),
        "source_static_report_sha256": file_sha256(path),
        "source_cell_results": cells,
        "raw_retained_count": len(authenticated),
        "deduplicated_pose_count": len(poses),
        "poses": poses,
    }


def _bounds(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    search = resolve_experiment(dict(config)).search_bounds
    pregrasp = search.pregrasp_targets_rad
    if pregrasp is None:
        raise ValueError("schema-v8 search has no pregrasp bounds")
    return pregrasp, search.actuator_targets_rad


def _clip(value: float, bounds: Mapping[str, Any], name: str) -> float:
    lower, upper = bounds[name]
    return float(np.clip(float(value), float(lower), float(upper)))


def _apply_control_spec(
    source: Mapping[str, Any], spec: RescueControlSpec
) -> dict[str, Any]:
    """Apply a rescue control spec while preserving a schema-v8 static pose."""

    candidate = copy.deepcopy(dict(source))
    pregrasp_bounds, target_bounds = _bounds(candidate)
    original = source["control"]
    target_residuals = dict(
        zip(NON_THUMB_TARGET_ACTUATORS, spec.non_thumb_target_residual_rad)
    )
    targets = {
        name: _clip(
            float(original["grasp_targets_rad"][name])
            + float(target_residuals.get(name, 0.0)),
            target_bounds,
            name,
        )
        for name in ACTIVE_ACTUATORS
    }
    targets[THUMB_BEND_ACTUATOR] = float(
        original["grasp_targets_rad"][THUMB_BEND_ACTUATOR]
    )
    proposed_pregrasp = {
        name: float(original["pregrasp_targets_rad"][name])
        + float(spec.pregrasp_residual_rad[index])
        for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    amplitudes = dict(zip(CLOSE_GROUP_ORDER, spec.group_amplitude_scale))
    pregrasp: dict[str, float] = {}
    for group in CLOSE_GROUP_ORDER:
        for name in CLOSE_GROUP_ACTUATORS[group]:
            value = targets[name] - float(amplitudes[group]) * (
                targets[name] - proposed_pregrasp[name]
            )
            pregrasp[name] = _clip(value, pregrasp_bounds, name)
    starts = dict(zip(CLOSE_GROUP_ORDER, spec.group_start_fraction))
    ends = dict(zip(CLOSE_GROUP_ORDER, spec.group_end_fraction))
    candidate["control"] = {
        "pregrasp_targets_rad": pregrasp,
        "grasp_targets_rad": targets,
        "manipulation_delta_rad": {name: 0.0 for name in ACTIVE_ACTUATORS},
        "close_profile": {
            name: {
                "start_fraction": float(starts[group]),
                "end_fraction": float(ends[group]),
            }
            for group in CLOSE_GROUP_ORDER
            for name in CLOSE_GROUP_ACTUATORS[group]
        },
    }
    candidate["control_protocol"]["close_s"] = float(spec.close_s)
    candidate.pop("run_context", None)
    return candidate


def _dynamic_record(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    candidate_id: int,
    stage: str,
    parent_candidate_id: int | None = None,
) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(config))
    source_static_id = int(source.get("source_static_candidate_id", source["candidate_id"]))
    metadata = copy.deepcopy(dict(candidate.get("candidate_metadata", {})))
    metadata.update(
        {
            "campaign_kind": DYNAMIC_CAMPAIGN_KIND,
            "stage": stage,
            "candidate_id": int(candidate_id),
            "source_static_candidate_id": source_static_id,
            "cube_pose_sampled": False,
            "free_cube_pose_reset_during_run": False,
        }
    )
    if parent_candidate_id is not None:
        metadata["parent_candidate_id"] = int(parent_candidate_id)
    candidate["candidate_metadata"] = metadata
    pose_identifier = pose_id_for_config(candidate)
    controller_identifier = controller_id_for_config(candidate)
    metadata["pose_id"] = pose_identifier
    metadata["controller_id"] = controller_identifier
    return {
        "campaign_kind": CAMPAIGN_KIND,
        "candidate_id": int(candidate_id),
        "candidate_sha256": canonical_sha256(candidate),
        "stage": stage,
        "pose_id": pose_identifier,
        "controller_id": controller_identifier,
        "tier": "new_pose",
        "source_candidate_id": source_static_id,
        "source_family_id": str(source.get("source_family_id", "new_pose_static")),
        "source_trajectory_id": str(source.get("cell_id", "new_pose_static")),
        "edge_m": float(candidate["cube"]["edge_m"]),
        "thumb_target_rad": float(
            candidate["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR]
        ),
        "config": candidate,
    }


def generate_initial_controller_candidates(
    poses: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    """Generate exactly eight evidence/LHS control seeds per static pose."""

    result: list[dict[str, Any]] = []
    for pose_index, raw in enumerate(poses):
        source = copy.deepcopy(dict(raw))
        source_config = source["config"]
        pose_seed = int(
            np.random.SeedSequence(
                [seed, int(source["candidate_id"]), 8_600_001]
            ).generate_state(1)[0]
        )
        specs = generate_control_specs(
            source_config,
            {"first_distal_contact_step": {}},
            count=8,
            seed=pose_seed,
        )
        for controller_index, spec in enumerate(specs):
            config = _apply_control_spec(source_config, spec)
            identifier = _INITIAL_BASE + pose_index * _POSE_STRIDE + controller_index
            config["candidate_metadata"] = {
                **copy.deepcopy(dict(config.get("candidate_metadata", {}))),
                "controller_seed_index": controller_index,
                "controller_seed_kind": spec.anchor_kind,
                "controller_seed_spec": spec.as_dict(),
            }
            record = _dynamic_record(
                config,
                source,
                candidate_id=identifier,
                stage="new_pose_dynamic_controller_seed",
            )
            if record["pose_id"] != source["pose_id"]:
                raise RuntimeError("controller seed changed the static pose identity")
            if validator is not None:
                validator(record["config"])
            result.append(record)
    return tuple(result)


def _materialize_local(
    parent: Mapping[str, Any],
    unit: np.ndarray,
    *,
    candidate_id: int,
    scale: float,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(parent["config"]))
    pregrasp_bounds, target_bounds = _bounds(config)
    config["hand_pose"]["translation_m"] = [
        float(value) + float(unit[index]) * 0.0004 * scale
        for index, value in enumerate(config["hand_pose"]["translation_m"])
    ]
    config["hand_pose"]["rpy_deg"] = [
        float(value) + float(unit[3 + index]) * 0.35 * scale
        for index, value in enumerate(config["hand_pose"]["rpy_deg"])
    ]
    pregrasp = config["control"]["pregrasp_targets_rad"]
    targets = config["control"]["grasp_targets_rad"]
    for index, name in enumerate(ACTIVE_ACTUATORS):
        pregrasp[name] = _clip(
            float(pregrasp[name]) + float(unit[6 + index]) * 0.02 * scale,
            pregrasp_bounds,
            name,
        )
    for index, name in enumerate(NON_THUMB_TARGET_ACTUATORS):
        targets[name] = _clip(
            float(targets[name]) + float(unit[14 + index]) * 0.025 * scale,
            target_bounds,
            name,
        )
    # The cell's commanded thumb band is immutable during local refinement.
    targets[THUMB_BEND_ACTUATOR] = float(
        parent["config"]["control"]["grasp_targets_rad"][THUMB_BEND_ACTUATOR]
    )
    profile = config["control"]["close_profile"]
    for group_index, group in enumerate(CLOSE_GROUP_ORDER):
        names = CLOSE_GROUP_ACTUATORS[group]
        centre_start = float(profile[names[0]]["start_fraction"])
        centre_end = float(profile[names[0]]["end_fraction"])
        start = centre_start + float(unit[21 + group_index]) * 0.025 * scale
        end = centre_end + float(unit[24 + group_index]) * 0.025 * scale
        end = float(np.clip(end, 0.10, 1.0))
        start = float(np.clip(start, 0.0, end - 0.05))
        for name in names:
            profile[name] = {"start_fraction": start, "end_fraction": end}
    current_close = float(config["control_protocol"]["close_s"])
    close_index = min(
        range(len(CLOSE_DURATION_OPTIONS_S)),
        key=lambda index: abs(CLOSE_DURATION_OPTIONS_S[index] - current_close),
    )
    if scale > 0.0 and unit[27] < -0.33:
        close_index = max(0, close_index - 1)
    elif scale > 0.0 and unit[27] > 0.33:
        close_index = min(len(CLOSE_DURATION_OPTIONS_S) - 1, close_index + 1)
    config["control_protocol"]["close_s"] = CLOSE_DURATION_OPTIONS_S[close_index]
    config["control"]["manipulation_delta_rad"] = {
        name: 0.0 for name in ACTIVE_ACTUATORS
    }
    config["candidate_metadata"] = {
        **copy.deepcopy(dict(config.get("candidate_metadata", {}))),
        "local_refinement_scale": float(scale),
        "local_refinement_unit": unit.tolist(),
    }
    source = {
        "candidate_id": int(parent["source_candidate_id"]),
        "source_static_candidate_id": int(parent["source_candidate_id"]),
        "source_family_id": parent.get("source_family_id", "new_pose_static"),
        "cell_id": parent.get("source_trajectory_id", "new_pose_static"),
    }
    return _dynamic_record(
        config,
        source,
        candidate_id=candidate_id,
        stage="new_pose_dynamic_local_refine",
        parent_candidate_id=int(parent["candidate_id"]),
    )


def generate_local_pose_controller_candidates(
    parents: Sequence[Mapping[str, Any]],
    *,
    count_per_pose: int,
    seed: int = DEFAULT_SEED,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    """Jointly perturb the best pose and its controller, deterministically."""

    if count_per_pose <= 0 or count_per_pose >= _POSE_STRIDE:
        raise ValueError("count_per_pose must be positive and smaller than pose stride")
    records: list[dict[str, Any]] = []
    for pose_index, parent in enumerate(parents):
        units = _latin_hypercube(
            count_per_pose,
            28,
            int(
                np.random.SeedSequence(
                    [seed, int(parent["candidate_id"]), 8_700_001]
                ).generate_state(1)[0]
            ),
        )
        for local_index, unit in enumerate(units):
            identifier = _LOCAL_BASE + pose_index * _POSE_STRIDE + local_index
            scales = (1.0, 0.5, 0.25, 0.125, 0.0) if validator else (1.0,)
            last_error: ValueError | None = None
            for scale in scales:
                record = _materialize_local(
                    parent, unit, candidate_id=identifier, scale=scale
                )
                if validator is None:
                    break
                try:
                    validator(record["config"])
                    break
                except ValueError as error:
                    last_error = error
            else:  # pragma: no cover - scale zero is the already-valid parent.
                raise RuntimeError("could not materialize a valid local candidate") from last_error
            records.append(record)
    return tuple(records)


def select_diverse_local_parents(
    results: Sequence[Mapping[str, Any]], *, count: int
) -> tuple[dict[str, Any], ...]:
    """Select unique poses with deterministic thumb-band and edge diversity.

    Each registered thumb target first receives one slot, then a second slot
    whose edge differs from its first selection whenever such a pose exists.
    Remaining capacity is filled from the global new-pose grasp rank.  This
    prevents an abundant low-thumb cell from consuming all 20 local-search
    parents while retaining deterministic worker-order independence.
    """

    if count <= 0:
        return ()
    best: dict[str, dict[str, Any]] = {}
    for result in results:
        identifier = str(result["pose_id"])
        previous = best.get(identifier)
        if previous is None or new_pose_grasp_rank(result) < new_pose_grasp_rank(previous):
            best[identifier] = copy.deepcopy(dict(result))
    ranked = rank_new_pose_grasp_results(tuple(best.values()))
    selected: list[dict[str, Any]] = []
    selected_pose_ids: set[str] = set()

    def add(value: Mapping[str, Any]) -> bool:
        pose_id = str(value["pose_id"])
        if pose_id in selected_pose_ids or len(selected) >= count:
            return False
        selected.append(copy.deepcopy(dict(value)))
        selected_pose_ids.add(pose_id)
        return True

    groups = {
        target: [
            value
            for value in ranked
            if math.isclose(
                float(value["thumb_target_rad"]), target, rel_tol=0.0, abs_tol=1e-9
            )
        ]
        for target in LOCAL_PARENT_THUMB_TARGETS_RAD
    }
    first_by_target: dict[float, dict[str, Any]] = {}
    for target in LOCAL_PARENT_THUMB_TARGETS_RAD:
        if groups[target] and add(groups[target][0]):
            first_by_target[target] = groups[target][0]

    for target in LOCAL_PARENT_THUMB_TARGETS_RAD:
        first = first_by_target.get(target)
        if first is None or len(selected) >= count:
            continue
        first_edge = float(first["edge_m"])
        available = [
            value
            for value in groups[target]
            if str(value["pose_id"]) not in selected_pose_ids
        ]
        different_edge = next(
            (
                value
                for value in available
                if not math.isclose(
                    float(value["edge_m"]), first_edge, rel_tol=0.0, abs_tol=1e-12
                )
            ),
            None,
        )
        if different_edge is not None:
            add(different_edge)
        elif available:
            add(available[0])

    for value in ranked:
        if len(selected) >= count:
            break
        add(value)
    return tuple(selected)


def _local_parent_selection_report(
    parents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    per_target = []
    for target in LOCAL_PARENT_THUMB_TARGETS_RAD:
        selected = [
            value
            for value in parents
            if math.isclose(
                float(value["thumb_target_rad"]), target, rel_tol=0.0, abs_tol=1e-9
            )
        ]
        per_target.append(
            {
                "thumb_target_rad": target,
                "count": len(selected),
                "edge_m": sorted({float(value["edge_m"]) for value in selected}),
            }
        )
    edges = sorted({float(value["edge_m"]) for value in parents})
    return {
        "strategy": (
            "one_per_thumb_then_second_distinct_edge_then_global_new_pose_rank"
        ),
        "per_thumb_target_quota": LOCAL_PARENT_PER_THUMB_QUOTA,
        "thumb_targets_rad": list(LOCAL_PARENT_THUMB_TARGETS_RAD),
        "selected_count": len(parents),
        "unique_pose_count": len({str(value["pose_id"]) for value in parents}),
        "per_thumb_target": per_target,
        "per_edge_m": [
            {
                "edge_m": edge,
                "count": sum(
                    math.isclose(
                        float(value["edge_m"]), edge, rel_tol=0.0, abs_tol=1e-12
                    )
                    for value in parents
                ),
            }
            for edge in edges
        ],
    }


def generate_locked_exact_candidates(
    ranked: Sequence[Mapping[str, Any]],
    *,
    count: int,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    """Replay the top candidates with explicit locked 1-ms provenance."""

    generated: list[dict[str, Any]] = []
    for index, parent in enumerate(ranked[:count]):
        config = copy.deepcopy(dict(parent["config"]))
        metadata = copy.deepcopy(dict(config.get("candidate_metadata", {})))
        metadata.update(
            {
                "campaign_kind": DYNAMIC_CAMPAIGN_KIND,
                "stage": "new_pose_dynamic_exact_1ms",
                "candidate_id": _EXACT_BASE + index,
                "parent_candidate_id": int(parent["candidate_id"]),
                "locked_timestep_s": 0.001,
                "pose_id": str(parent["pose_id"]),
                "controller_id": str(parent["controller_id"]),
            }
        )
        config["candidate_metadata"] = metadata
        if pose_id_for_config(config) != parent["pose_id"]:
            raise RuntimeError("exact replay changed its pose")
        if controller_id_for_config(config) != parent["controller_id"]:
            raise RuntimeError("exact replay changed its controller")
        if validator is not None:
            validator(config)
        record = {
            **{
                key: copy.deepcopy(parent[key])
                for key in (
                    "pose_id",
                    "controller_id",
                    "tier",
                    "source_candidate_id",
                    "source_family_id",
                    "source_trajectory_id",
                    "edge_m",
                    "thumb_target_rad",
                )
            },
            "campaign_kind": CAMPAIGN_KIND,
            "candidate_id": _EXACT_BASE + index,
            "candidate_sha256": canonical_sha256(config),
            "stage": "new_pose_dynamic_exact_1ms",
            "config": config,
        }
        generated.append(record)
    return tuple(generated)


def _compact(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value.get(key))
        for key in (
            "candidate_id",
            "candidate_sha256",
            "stage",
            "pose_id",
            "controller_id",
            "source_candidate_id",
            "edge_m",
            "thumb_target_rad",
            "acquisition_success",
            "pose_preservation_success",
            "rescue_success",
            "classification",
            "rank_evidence",
            "artifact_directory",
        )
    }


def _certify_exact_grasps(
    output: Path, exact_results: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    passes = [
        value
        for value in rank_new_pose_grasp_results(exact_results)
        if candidate_rank_evidence(value)["rescue_success"]
    ]
    certified_root = output / "certified_grasps"
    catalog: list[dict[str, Any]] = []
    for index, result in enumerate(passes):
        alias = f"new_pose_grasp_{index + 1:02d}"
        destination = certified_root / alias
        destination.mkdir(parents=True, exist_ok=True)
        config_path = destination / "resolved_config.json"
        if config_path.is_file():
            persisted = json.loads(config_path.read_text(encoding="utf-8"))
            if canonical_sha256(persisted) != canonical_sha256(result["config"]):
                raise RuntimeError(f"resume certified config mismatch: {config_path}")
        else:
            write_json(config_path, result["config"])
        binding = {
            "alias": alias,
            "candidate_id": int(result["candidate_id"]),
            "pose_id": str(result["pose_id"]),
            "controller_id": str(result["controller_id"]),
            "edge_m": float(result["edge_m"]),
            "thumb_target_rad": float(result["thumb_target_rad"]),
            "source_artifact_directory": str(result.get("artifact_directory", "")),
            "resolved_config": str(config_path.relative_to(output)),
            "resolved_config_sha256": file_sha256(config_path),
            "resolved_config_semantic_sha256": canonical_sha256(result["config"]),
        }
        write_json(destination / "binding.json", binding)
        catalog.append(binding)
    write_json(
        output / "grasp_catalog.json",
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "campaign_kind": DYNAMIC_CAMPAIGN_KIND,
            "certified_grasp_count": len(catalog),
            "trajectories": catalog,
        },
    )
    return tuple(catalog)


def run_new_pose_dynamic_campaign(
    static_report_path: str | Path = DEFAULT_STATIC_REPORT,
    template_path: str | Path = DEFAULT_TEMPLATE,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    workers: int = 1,
    resume: bool = False,
    seed: int = DEFAULT_SEED,
    budget: DynamicPromotionBudget = REGISTERED_DYNAMIC_BUDGET,
    executor: CandidateExecutor = run_candidate_jobs,
) -> dict[str, Any]:
    """Run controller seeds, local refinement and locked exact promotion."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    output = Path(output_dir).expanduser().resolve()
    template_file = Path(template_path).expanduser().resolve()
    template = load_config(template_file)
    if (
        int(template.get("schema_version", 0)) != 8
        or template.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("dynamic promotion template must use schema version 8")
    manifest = load_static_pose_manifest(static_report_path)
    input_payload = {
        "dynamic_report_schema_version": DYNAMIC_REPORT_SCHEMA_VERSION,
        "campaign_kind": DYNAMIC_CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        "seed": int(seed),
        "budget": asdict(budget),
        "registered_budget": budget.registered,
        "static_manifest_sha256": canonical_sha256(manifest),
        "template_file_sha256": file_sha256(template_file),
        "template_semantic_sha256": canonical_sha256(template),
        "tuner_source_sha256": file_sha256(Path(__file__)),
    }
    input_sha = canonical_sha256(input_payload)
    campaign_manifest_path = output / "campaign_manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output}; pass --resume")
        existing = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
        if existing.get("campaign_input_sha256") != input_sha:
            raise RuntimeError("resume inputs do not match the existing dynamic manifest")
    else:
        output.mkdir(parents=True)
        write_json(output / "static_pose_manifest.json", manifest)
        write_json(
            campaign_manifest_path,
            {
                **input_payload,
                "campaign_input_sha256": input_sha,
                "output_directory": str(output),
                "complete": False,
            },
        )

    initial_candidates = generate_initial_controller_candidates(
        manifest["poses"], seed=seed
    )
    initial_results = _run_or_resume(
        initial_candidates,
        output,
        workers=workers,
        resume=resume,
        executor=executor,
    )
    local_parents = select_diverse_local_parents(
        initial_results, count=budget.local_pose_count
    )
    local_candidates = generate_local_pose_controller_candidates(
        local_parents,
        count_per_pose=budget.local_refine_per_pose,
        seed=seed,
    )
    local_results = _run_or_resume(
        local_candidates,
        output,
        workers=workers,
        resume=resume,
        executor=executor,
    )
    promoted = rank_new_pose_grasp_results((*initial_results, *local_results))
    exact_parents = select_diverse_local_parents(
        promoted, count=budget.exact_candidate_count
    )
    exact_candidates = generate_locked_exact_candidates(
        exact_parents, count=budget.exact_candidate_count
    )
    exact_results = _run_or_resume(
        exact_candidates,
        output,
        workers=workers,
        resume=resume,
        executor=executor,
    )
    ranked_exact = rank_new_pose_grasp_results(exact_results)
    certified = _certify_exact_grasps(output, ranked_exact)
    all_results = (*initial_results, *local_results, *exact_results)
    report = {
        "dynamic_report_schema_version": DYNAMIC_REPORT_SCHEMA_VERSION,
        "campaign_kind": DYNAMIC_CAMPAIGN_KIND,
        "experiment_id": EXPERIMENT_ID,
        "stage": "new_pose_dynamic",
        "complete": True,
        "campaign_input_sha256": input_sha,
        "workers": int(workers),
        "seed": int(seed),
        "budget": {
            **asdict(budget),
            "registered": budget.registered,
            "maximum_initial_count": (
                manifest["deduplicated_pose_count"] * budget.controller_seeds_per_pose
            ),
            "maximum_local_count": (
                min(manifest["deduplicated_pose_count"], budget.local_pose_count)
                * budget.local_refine_per_pose
            ),
        },
        "static_pose_count": manifest["deduplicated_pose_count"],
        "initial_candidate_count": len(initial_results),
        "local_parent_count": len(local_parents),
        "local_parent_selection": _local_parent_selection_report(local_parents),
        "local_candidate_count": len(local_results),
        "exact_parent_selection": _local_parent_selection_report(exact_parents),
        "exact_candidate_count": len(exact_results),
        "candidate_count": len(all_results),
        "certified_grasp_count": len(certified),
        "certified_grasps": list(certified),
        "best_candidate": _compact(ranked_exact[0]),
        "results": [_compact(value) for value in ranked_exact],
        "ranking_policy": (
            "hard_rescue_evidence_acquisition_pose_preservation_closure_gate_"
            "pose_margin_angle_force_saturation_candidate_id"
        ),
        "pose_controller_hashes_separate": True,
        "next_stage": "response_guided_smooth_vertical_lift",
    }
    write_json(output / "search_report.json", report)
    write_json(
        campaign_manifest_path,
        {
            **input_payload,
            "campaign_input_sha256": input_sha,
            "output_directory": str(output),
            "complete": True,
            "candidate_count": len(all_results),
            "certified_grasp_count": len(certified),
            "search_report": "search_report.json",
            "search_report_sha256": file_sha256(output / "search_report.json"),
            "grasp_catalog": "grasp_catalog.json",
            "grasp_catalog_sha256": file_sha256(output / "grasp_catalog.json"),
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote retained schema-v8 static poses through real MuJoCo "
            "grasp dynamics, local refinement and locked 1-ms replay."
        )
    )
    parser.add_argument("--static-report", default=str(DEFAULT_STATIC_REPORT))
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_new_pose_dynamic_campaign(
        args.static_report,
        args.template,
        args.output_dir,
        workers=args.workers,
        resume=args.resume,
        seed=args.seed,
    )
    print(json_text({key: value for key, value in report.items() if key != "results"}))
    return 0 if int(report["certified_grasp_count"]) > 0 else 2


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_STATIC_REPORT",
    "DEFAULT_TEMPLATE",
    "DYNAMIC_CAMPAIGN_KIND",
    "DynamicPromotionBudget",
    "REGISTERED_DYNAMIC_BUDGET",
    "generate_initial_controller_candidates",
    "generate_local_pose_controller_candidates",
    "generate_locked_exact_candidates",
    "load_static_pose_manifest",
    "run_new_pose_dynamic_campaign",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
