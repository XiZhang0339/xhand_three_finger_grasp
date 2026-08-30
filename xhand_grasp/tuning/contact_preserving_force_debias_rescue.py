"""Force-debias rescue numerics for the completed schema-v14 adaptive campaign.

The adaptive event campaign is immutable input.  This module authenticates its
five published full-reset traces and creates deterministic, constraint-
projected jobs which combine two complementary changes:

* a broad C2 outward feed-forward basis for each active finger; and
* a wider compact event basis (0.18/0.22/0.26 progress half-widths).

The force controller is searched independently through gain multipliers,
integral/correction limits and one of three filter constants.  No simulation,
CLI dispatch or artifact writing lives here; campaign runners own those jobs.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..actual_contact_grasp_pose_catalog import validate_stage_ledger
from ..artifacts import file_sha256
from ..config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    contact_preload_targets,
    precontact_targets,
    validate_config,
)
from ..experiment import (
    ContactFeedbackParameters,
    ContactForceTargets,
    ManipulationPlanParameters,
)
from ..grasp_pose import canonical_sha256
from .contact_constrained_planner import (
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
)
from .contact_preserving_adaptive_event_rescue import (
    AdaptiveEventDetectionSettings,
    FeasibleEventPolytope,
    _append_linear_constraint,
    _checkpoint_jacobian_directions,
    adaptive_event_candidate_rank,
    detect_adaptive_contact_events,
    physical_plan_sha256,
)
from .contact_preserving_candidate_artifacts import (
    authenticate_v14_candidate_artifacts,
)
from .contact_preserving_event_rescue import (
    EventJacobianDirections,
    compact_c2_event_bump,
    trace_content_sha256,
)
from .contact_preserving_joint_refinement import resolve_joint_refinement_limits
from .contact_preserving_time_warp import (
    _time_warp_controller_id,
    quintic_bezier_controls,
)


FORCE_DEBIAS_SCHEMA_VERSION = 1
EXPERIMENT_ID = "left_opposed_face_palm_down_contact_preserving_planned_lift"
DEFAULT_SEED = 20260821
EVENT_HALF_WIDTHS_PROGRESS = (0.18, 0.22, 0.26)
BROAD_UNLOAD_MAX_RAD = 0.035
EVENT_TANGENT_MAX_RAD = 0.015
EVENT_UNLOAD_MAX_RAD = 0.006
KP_MULTIPLIER_RANGE = (1.0, 3.0)
KI_MULTIPLIER_RANGE = (1.0, 2.0)
INTEGRAL_LIMIT_RANGE_N_S = (0.5, 1.25)
CORRECTION_LIMIT_RANGE_RAD = (0.06, 0.09)
FILTER_TIME_CONSTANTS_S = (0.005, 0.008, 0.012)
OPERATION_FORCE_SCALE_RANGE = (0.70, 1.00)
DISCOVERY_CANDIDATES_PER_CENTER = 32
STRUCTURED_DISCOVERY_ANCHOR_COUNT = 12
SMALL_RADIUS_DISCOVERY_COUNTS = ((0.05, 7), (0.10, 7), (0.20, 6))
_FINGER_SLICES = {
    "thumb": slice(0, 3),
    "index": slice(3, 6),
    "mid": slice(6, 8),
}
_EPS = 1e-12
_DESCRIPTOR_CACHE: dict[tuple[str, str, str], "ForceDebiasDescriptor"] = {}
_POLYTOPE_CACHE: dict[tuple[str, str, float], FeasibleEventPolytope] = {}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"force-debias evidence is not a JSON object: {path}")
    return value


def _minimum_jerk(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped**3 * (10.0 + clipped * (-15.0 + 6.0 * clipped))


def broad_c2_force_debias_envelope(
    progress: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """C2 rise/plateau/terminal compensation envelope for broad unloading.

    It is identically zero through progress 0.20, rises with a quintic
    minimum-jerk profile through 0.40, stays at one through 0.85, then returns
    smoothly to zero at the terminal knot.  Value, velocity and acceleration
    therefore agree on every piece boundary.
    """

    values = np.asarray(progress, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("force-debias progress must be finite")
    result = np.zeros_like(values)
    rising = (values > 0.20) & (values < 0.40)
    result[rising] = _minimum_jerk((values[rising] - 0.20) / 0.20)
    result[(values >= 0.40) & (values <= 0.85)] = 1.0
    falling = (values > 0.85) & (values < 1.0)
    result[falling] = 1.0 - _minimum_jerk((values[falling] - 0.85) / 0.15)
    return result


@dataclass(frozen=True, slots=True)
class ForceDebiasCenter:
    candidate_id: int
    source_authentication_id: str
    config: dict[str, Any]
    result: dict[str, Any]
    summary: dict[str, Any]
    config_path: Path
    result_path: Path
    trace_path: Path
    config_semantic_sha256: str
    result_semantic_sha256: str
    trace_sha256: str
    trace_content_sha256: str
    physical_plan_sha256: str
    center_id: str

    def as_mapping(self, *, include_payloads: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": FORCE_DEBIAS_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "source_authentication_id": self.source_authentication_id,
            "config_path": str(self.config_path),
            "result_path": str(self.result_path),
            "trace_path": str(self.trace_path),
            "config_semantic_sha256": self.config_semantic_sha256,
            "result_semantic_sha256": self.result_semantic_sha256,
            "trace_sha256": self.trace_sha256,
            "trace_content_sha256": self.trace_content_sha256,
            "physical_plan_sha256": self.physical_plan_sha256,
            "center_id": self.center_id,
        }
        if include_payloads:
            value.update(
                config=copy.deepcopy(self.config),
                result=copy.deepcopy(self.result),
                summary=copy.deepcopy(self.summary),
            )
        return value

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ForceDebiasCenter":
        value = copy.deepcopy(dict(raw))
        if int(value.pop("schema_version", -1)) != FORCE_DEBIAS_SCHEMA_VERSION:
            raise ValueError("force-debias center schema is invalid")
        for name in ("config_path", "result_path", "trace_path"):
            value[name] = Path(value[name]).expanduser().resolve()
        if "config" not in value:
            value["config"] = _load_json(value["config_path"])
        if "result" not in value:
            value["result"] = _load_json(value["result_path"])
        if "summary" not in value:
            value["summary"] = copy.deepcopy(value["result"]["summary"])
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ForceDebiasSource:
    root: Path
    source_authentication_id: str
    manifest_path: Path
    ledger_path: Path
    result_path: Path
    catalog_path: Path
    artifact_paths: tuple[Path, ...]
    centers: tuple[ForceDebiasCenter, ...]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": FORCE_DEBIAS_SCHEMA_VERSION,
            "root": str(self.root),
            "source_authentication_id": self.source_authentication_id,
            "manifest_path": str(self.manifest_path),
            "ledger_path": str(self.ledger_path),
            "result_path": str(self.result_path),
            "catalog_path": str(self.catalog_path),
            "artifact_paths": [str(path) for path in self.artifact_paths],
            "centers": [center.as_mapping() for center in self.centers],
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ForceDebiasSource":
        value = copy.deepcopy(dict(raw))
        if int(value.pop("schema_version", -1)) != FORCE_DEBIAS_SCHEMA_VERSION:
            raise ValueError("force-debias source schema is invalid")
        for name in ("root", "manifest_path", "ledger_path", "result_path", "catalog_path"):
            value[name] = Path(value[name]).expanduser().resolve()
        value["artifact_paths"] = tuple(
            Path(path).expanduser().resolve() for path in value["artifact_paths"]
        )
        value["centers"] = tuple(
            ForceDebiasCenter.from_mapping(center) for center in value["centers"]
        )
        return cls(**value)


def _catalog_trajectory_by_candidate(
    catalog: Mapping[str, Any], candidate_id: int
) -> Mapping[str, Any]:
    trajectories = catalog.get("trajectories")
    if not isinstance(trajectories, list):
        raise RuntimeError("force-debias source catalog lost trajectories")
    matches = [
        value
        for value in trajectories
        if isinstance(value, Mapping) and int(value.get("candidate_id", -1)) == candidate_id
    ]
    if len(matches) != 1:
        raise RuntimeError("force-debias source catalog candidate set changed")
    return matches[0]


def authenticate_force_debias_source(source_campaign: str | Path) -> ForceDebiasSource:
    """Authenticate the completed adaptive-v2 top-five full-reset evidence."""

    root = Path(source_campaign).expanduser().resolve()
    manifest_path = root / "campaign_manifest.json"
    ledger_path = root / "stage_ledger.json"
    result_path = root / "adaptive_event_rescue_result_target_1.json"
    catalog_path = root / "catalogs/target_1/manipulation/catalog.json"
    reports = (
        root / "adaptive_diagnostics/report.json",
        root / "adaptive_projected_exploration/report.json",
        root / "adaptive_local_refinement/report.json",
        root / "catalogs/target_1/report.json",
    )
    required = (manifest_path, ledger_path, result_path, catalog_path, *reports)
    if not root.is_dir() or not all(path.is_file() for path in required):
        raise RuntimeError("force-debias source lost completed adaptive evidence")
    ledger = validate_stage_ledger(root)
    expected_stages = {
        "adaptive_source_audit",
        "adaptive_source_materialization",
        "adaptive_diagnostics",
        "adaptive_projected_exploration",
        "adaptive_local_refinement",
        "adaptive_catalog_target_1",
        "adaptive_result_target_1",
    }
    if not expected_stages.issubset(ledger.get("stages", {})):
        raise RuntimeError("force-debias source adaptive campaign is incomplete")
    manifest = _load_json(manifest_path)
    result = _load_json(result_path)
    catalog = _load_json(catalog_path)
    if (
        manifest.get("campaign_kind") != "contact_preserving_adaptive_event_rescue"
        or result.get("complete") is not True
        or int(result.get("full_success_count", -1)) != 0
        or catalog.get("complete") is not True
    ):
        raise RuntimeError("force-debias source is not the sealed zero-success adaptive campaign")
    raw_ids = result.get("published_candidate_ids")
    if not isinstance(raw_ids, list) or len(raw_ids) != 5:
        raise RuntimeError("force-debias source must expose exactly five adaptive centers")
    candidate_ids = tuple(int(value) for value in raw_ids)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("force-debias source published duplicate candidates")

    immutable_paths: list[Path] = list(required)
    provisional: list[tuple[int, dict[str, Any], dict[str, Any], Path, Path, Path]] = []
    for candidate_id in candidate_ids:
        entry = _catalog_trajectory_by_candidate(catalog, candidate_id)
        rerun = root / "catalog_source_reruns/adaptive_target_1" / f"candidate_{candidate_id}"
        bundle = authenticate_v14_candidate_artifacts(
            rerun,
            expected_candidate_id=candidate_id,
            require_retained_trace=True,
        )
        if bundle.trace_path is None:
            raise RuntimeError("force-debias source center lost its trace")
        config = _load_json(bundle.config_path)
        catalog_relative = entry.get("artifacts", {}).get("resolved_config")
        if not isinstance(catalog_relative, str):
            raise RuntimeError("force-debias source catalog lost its config reference")
        catalog_dir = catalog_path.parent / Path(catalog_relative).parent
        catalog_config = catalog_dir / "resolved_config.json"
        catalog_result = catalog_dir / "result.json"
        catalog_trace = catalog_dir / "trace.npz"
        catalog_video = catalog_dir / "trajectory.mp4"
        for path in (catalog_config, catalog_result, catalog_trace, catalog_video):
            if not path.is_file():
                raise RuntimeError("force-debias source catalog artifact is missing")
        catalog_hashes = entry.get("artifacts", {}).get("sha256")
        if not isinstance(catalog_hashes, Mapping) or any(
            catalog_hashes.get(name) != file_sha256(path)
            for name, path in (
                ("resolved_config", catalog_config),
                ("result", catalog_result),
                ("trace", catalog_trace),
                ("video", catalog_video),
            )
        ):
            raise RuntimeError("force-debias source catalog artifact SHA-256 changed")
        if canonical_sha256(_load_json(catalog_config)) != canonical_sha256(config):
            raise RuntimeError("force-debias source catalog and rerun configs disagree")
        immutable_paths.extend((*bundle.artifact_paths, catalog_config, catalog_result, catalog_trace, catalog_video))
        provisional.append(
            (
                candidate_id,
                config,
                copy.deepcopy(bundle.result),
                bundle.config_path,
                bundle.result_path,
                bundle.trace_path,
            )
        )

    immutable = tuple(dict.fromkeys(path.resolve() for path in immutable_paths))
    evidence_sha = {str(path): file_sha256(path) for path in immutable}
    authentication_id = canonical_sha256(
        {
            "schema_version": FORCE_DEBIAS_SCHEMA_VERSION,
            "root": str(root),
            "published_candidate_ids": list(candidate_ids),
            "artifact_sha256": dict(sorted(evidence_sha.items())),
        }
    )
    centers: list[ForceDebiasCenter] = []
    for candidate_id, config, candidate_result, config_path, candidate_result_path, trace_path in provisional:
        summary = candidate_result.get("summary")
        if not isinstance(summary, Mapping):
            raise RuntimeError("force-debias source result lost its summary")
        trace_sha = file_sha256(trace_path)
        with np.load(trace_path, allow_pickle=False) as loaded:
            trace = {name: loaded[name] for name in loaded.files}
        content_sha = trace_content_sha256(trace)
        physical = physical_plan_sha256(config)
        center_payload = {
            "candidate_id": candidate_id,
            "source_authentication_id": authentication_id,
            "config_semantic_sha256": canonical_sha256(config),
            "result_semantic_sha256": candidate_result.get("result_semantic_sha256"),
            "trace_sha256": trace_sha,
            "trace_content_sha256": content_sha,
            "physical_plan_sha256": physical,
        }
        centers.append(
            ForceDebiasCenter(
                candidate_id=candidate_id,
                source_authentication_id=authentication_id,
                config=config,
                result=candidate_result,
                summary=copy.deepcopy(dict(summary)),
                config_path=config_path,
                result_path=candidate_result_path,
                trace_path=trace_path,
                config_semantic_sha256=canonical_sha256(config),
                result_semantic_sha256=str(candidate_result.get("result_semantic_sha256", "")),
                trace_sha256=trace_sha,
                trace_content_sha256=content_sha,
                physical_plan_sha256=physical,
                center_id=canonical_sha256(center_payload),
            )
        )
    return ForceDebiasSource(
        root=root,
        source_authentication_id=authentication_id,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        result_path=result_path,
        catalog_path=catalog_path,
        artifact_paths=immutable,
        centers=tuple(centers),
    )


@dataclass(frozen=True, slots=True)
class ForceDebiasDescriptor:
    source_center_id: str
    source_candidate_id: int
    source_config_semantic_sha256: str
    source_trace_sha256: str
    source_trace_content_sha256: str
    events: tuple[Any, ...]
    directions: tuple[EventJacobianDirections, ...]
    inward_directions_rad_unit: tuple[tuple[float, ...], ...]
    descriptor_id: str


def _controller_inward_directions(config: Mapping[str, Any]) -> np.ndarray:
    pre = precontact_targets(copy.deepcopy(dict(config)))
    preload = contact_preload_targets(copy.deepcopy(dict(config)))
    vector = np.asarray(
        [float(preload[name]) - float(pre[name]) for name in ACTIVE_ACTUATORS]
    )
    directions = np.zeros((len(ACTIVE_FINGERS), len(ACTIVE_ACTUATORS)))
    for finger_index, finger in enumerate(ACTIVE_FINGERS):
        owned = _FINGER_SLICES[finger]
        local = vector[owned]
        maximum = float(np.max(np.abs(local), initial=0.0))
        if maximum <= _EPS:
            raise ValueError(f"force-debias {finger} inward ray is degenerate")
        directions[finger_index, owned] = local / maximum
    return directions


def _primary_events(events: Sequence[Any]) -> tuple[Any, ...]:
    selected: list[Any] = []
    for finger in ACTIVE_FINGERS:
        matches = [value for value in events if value.finger == finger]
        if not matches:
            continue
        selected.append(
            min(
                matches,
                key=lambda value: (
                    -float(value.associated_peak_abs_jerk_m_s3),
                    -float(value.tangent_jump_m),
                    int(value.event_step),
                    str(value.event_id),
                ),
            )
        )
    return tuple(selected)


def build_force_debias_descriptor(center: ForceDebiasCenter) -> ForceDebiasDescriptor:
    if center.source_authentication_id == "" or not _is_sha256(center.center_id):
        raise ValueError("force-debias center identity is invalid")
    if canonical_sha256(center.config) != center.config_semantic_sha256:
        raise RuntimeError("force-debias center config content changed")
    if file_sha256(center.trace_path) != center.trace_sha256:
        raise RuntimeError("force-debias center trace SHA-256 changed")
    cache_key = (
        center.center_id,
        center.config_semantic_sha256,
        center.trace_sha256,
    )
    cached = _DESCRIPTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with np.load(center.trace_path, allow_pickle=False) as loaded:
        trace = {name: loaded[name] for name in loaded.files}
    if trace_content_sha256(trace) != center.trace_content_sha256:
        raise RuntimeError("force-debias center trace content changed")
    settings = AdaptiveEventDetectionSettings(global_max_events=24)
    events = _primary_events(
        detect_adaptive_contact_events(center.config, trace, settings=settings)
    )
    directions = tuple(
        _checkpoint_jacobian_directions(center.config, trace, event)
        for event in events
    )
    inward = _controller_inward_directions(center.config)
    payload = {
        "schema_version": FORCE_DEBIAS_SCHEMA_VERSION,
        "source_center_id": center.center_id,
        "source_candidate_id": center.candidate_id,
        "source_config_semantic_sha256": center.config_semantic_sha256,
        "source_trace_sha256": center.trace_sha256,
        "source_trace_content_sha256": center.trace_content_sha256,
        "event_ids": [value.event_id for value in events],
        "direction_ids": [value.directions_id for value in directions],
        "inward_directions_rad_unit": inward.tolist(),
    }
    descriptor = ForceDebiasDescriptor(
        source_center_id=center.center_id,
        source_candidate_id=center.candidate_id,
        source_config_semantic_sha256=center.config_semantic_sha256,
        source_trace_sha256=center.trace_sha256,
        source_trace_content_sha256=center.trace_content_sha256,
        events=events,
        directions=directions,
        inward_directions_rad_unit=tuple(tuple(float(item) for item in row) for row in inward),
        descriptor_id=canonical_sha256(payload),
    )
    _DESCRIPTOR_CACHE[cache_key] = descriptor
    return descriptor


def derive_force_debias_polytope(
    config: Mapping[str, Any],
    descriptor: ForceDebiasDescriptor,
    *,
    event_half_width_progress: float,
    projection_tolerance: float = 1e-12,
    projection_max_iterations: int = 512,
) -> FeasibleEventPolytope:
    """Build the exact waypoint/Bezier/joint feasible set for one width."""

    width = float(event_half_width_progress)
    if width not in EVENT_HALF_WIDTHS_PROGRESS:
        raise ValueError("force-debias event half-width is not registered")
    if canonical_sha256(config) != descriptor.source_config_semantic_sha256:
        raise RuntimeError("force-debias descriptor belongs to another config")
    cache_key = (
        descriptor.descriptor_id,
        physical_plan_sha256(config),
        width,
    )
    cached = _POLYTOPE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    plan = ManipulationPlanParameters.from_config(config["manipulation_plan"])
    times = np.asarray(plan.knot_times_s, dtype=np.float64)
    progress = times / plan.duration_s
    base = np.stack(
        [np.asarray(plan.actuator_waypoints_rad[name]) for name in ACTIVE_ACTUATORS],
        axis=1,
    )
    envelope = broad_c2_force_debias_envelope(progress)
    inward = np.asarray(descriptor.inward_directions_rad_unit)
    names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    basis: list[np.ndarray] = []
    for finger_index, finger in enumerate(ACTIVE_FINGERS):
        names.append(f"broad_unload:{finger}")
        lower.append(0.0)
        upper.append(BROAD_UNLOAD_MAX_RAD)
        basis.append(-envelope[:, None] * inward[finger_index][None, :])
    directions = {value.event_id: value for value in descriptor.directions}
    for event in descriptor.events:
        bump = compact_c2_event_bump(progress, event.manipulation_progress, width)
        direction = directions[event.event_id]
        names.extend((f"event:{event.event_id}:tangent", f"event:{event.event_id}:unload"))
        lower.extend((-EVENT_TANGENT_MAX_RAD, 0.0))
        upper.extend((EVENT_TANGENT_MAX_RAD, EVENT_UNLOAD_MAX_RAD))
        basis.extend(
            (
                bump[:, None]
                * np.asarray(direction.tangent_direction_rad_unit)[None, :],
                bump[:, None]
                * np.asarray(direction.normal_unload_direction_rad_unit)[None, :],
            )
        )
    basis_array = np.asarray(basis, dtype=np.float64)
    lower_array = np.asarray(lower)
    upper_array = np.asarray(upper)
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    labels: list[str] = []
    for knot in range(base.shape[0] - 1):
        for actuator, name in enumerate(ACTIVE_ACTUATORS):
            delta = float(base[knot + 1, actuator] - base[knot, actuator])
            coefficients = basis_array[:, knot + 1, actuator] - basis_array[:, knot, actuator]
            _append_linear_constraint(rows, bounds, labels, coefficients, plan.max_knot_delta_rad - delta, f"adjacent_upper:{name}:{knot}", lower_array, upper_array)
            _append_linear_constraint(rows, bounds, labels, -coefficients, plan.max_knot_delta_rad + delta, f"adjacent_lower:{name}:{knot}", lower_array, upper_array)
    limits = resolve_joint_refinement_limits(config)
    for actuator, name in enumerate(ACTIVE_ACTUATORS):
        coefficients = basis_array[:, -1, actuator]
        lo, hi = limits.registered_plan_delta_rad[name]
        _append_linear_constraint(rows, bounds, labels, coefficients, hi - base[-1, actuator], f"terminal_upper:{name}", lower_array, upper_array)
        _append_linear_constraint(rows, bounds, labels, -coefficients, base[-1, actuator] - lo, f"terminal_lower:{name}", lower_array, upper_array)
    base_controls = quintic_bezier_controls(times, base)
    basis_controls = np.stack(
        [quintic_bezier_controls(times, matrix) for matrix in basis_array]
    )
    preload = contact_preload_targets(copy.deepcopy(dict(config)))
    for segment in range(base_controls.shape[0]):
        for control_index in range(base_controls.shape[1]):
            for actuator, name in enumerate(ACTIVE_ACTUATORS):
                command_base = float(preload[name]) + float(base_controls[segment, control_index, actuator])
                coefficients = basis_controls[:, segment, control_index, actuator]
                lo, hi = limits.command_target_rad[name]
                _append_linear_constraint(rows, bounds, labels, coefficients, hi - command_base, f"bezier_upper:{name}:{segment}:{control_index}", lower_array, upper_array)
                _append_linear_constraint(rows, bounds, labels, -coefficients, command_base - lo, f"bezier_lower:{name}:{segment}:{control_index}", lower_array, upper_array)
    identity = {
        "schema_version": FORCE_DEBIAS_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "descriptor_id": descriptor.descriptor_id,
        "parent_physical_plan_sha256": physical_plan_sha256(config),
        "parameter_names": names,
        "lower_bounds": lower,
        "upper_bounds": upper,
        "constraint_matrix": [row.tolist() for row in rows],
        "constraint_upper": bounds,
        "constraint_labels": labels,
        "base_waypoints_rad": base.tolist(),
        "basis_waypoints_rad": basis_array.tolist(),
        "bump_half_width_progress": width,
        "projection_tolerance": projection_tolerance,
        "projection_max_iterations": projection_max_iterations,
    }
    polytope = FeasibleEventPolytope(
        schema_version=FORCE_DEBIAS_SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        descriptor_id=descriptor.descriptor_id,
        parent_physical_plan_sha256=physical_plan_sha256(config),
        parameter_names=tuple(names),
        lower_bounds=tuple(lower),
        upper_bounds=tuple(upper),
        constraint_matrix=tuple(tuple(float(item) for item in row) for row in rows),
        constraint_upper=tuple(bounds),
        constraint_labels=tuple(labels),
        base_waypoints_rad=tuple(tuple(float(item) for item in row) for row in base),
        basis_waypoints_rad=tuple(
            tuple(tuple(float(item) for item in row) for row in matrix)
            for matrix in basis_array
        ),
        bump_half_width_progress=width,
        projection_tolerance=projection_tolerance,
        projection_max_iterations=projection_max_iterations,
        polytope_id=canonical_sha256(identity),
    )
    _POLYTOPE_CACHE[cache_key] = polytope
    return polytope


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    if count <= 0 or dimensions <= 0:
        raise ValueError("force-debias LHS dimensions must be positive")
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=np.float64)
    for column in range(dimensions):
        order = rng.permutation(count)
        result[:, column] = (order + rng.random(count)) / float(count)
    return result


def _feedback_mapping(
    parent: Mapping[str, Any], parameters: Mapping[str, float]
) -> dict[str, Any]:
    original = ContactFeedbackParameters.from_config(parent)
    feedback = ContactFeedbackParameters(
        schema_version=original.schema_version,
        strategy=original.strategy,
        filter_time_constant_s=float(parameters["filter_time_constant_s"]),
        kp_rad_per_n={
            finger: float(original.kp_rad_per_n[finger]) * float(parameters["kp_multiplier"])
            for finger in ACTIVE_FINGERS
        },
        ki_rad_per_n_s={
            finger: float(original.ki_rad_per_n_s[finger]) * float(parameters["ki_multiplier"])
            for finger in ACTIVE_FINGERS
        },
        integral_limit_n_s=float(parameters["integral_limit_n_s"]),
        correction_limit_rad=float(parameters["correction_limit_rad"]),
        rate_limit_rad_s=original.rate_limit_rad_s,
        acceleration_limit_rad_s2=original.acceleration_limit_rad_s2,
        force_risk_n=original.force_risk_n,
        freeze_on_risk=original.freeze_on_risk,
        max_loss_s=original.max_loss_s,
        recovery_behavior=original.recovery_behavior,
        operation_contact_duty_min=original.operation_contact_duty_min,
        tangent_slip_freeze_threshold_m=(
            original.tangent_slip_freeze_threshold_m
        ),
        tangent_slip_abort_threshold_m=(
            original.tangent_slip_abort_threshold_m
        ),
    )
    return feedback.as_config()


def _force_target_mapping(
    parent: Mapping[str, Any], operation_scale: float
) -> dict[str, Any]:
    original = ContactForceTargets.from_config(parent)
    return ContactForceTargets(
        schema_version=original.schema_version,
        source=original.source,
        minimum_n=original.minimum_n,
        maximum_n=original.maximum_n,
        per_finger_n=original.per_finger_n,
        operation_scale=float(operation_scale),
    ).as_config()


def _candidate_config(
    center: ForceDebiasCenter,
    descriptor: ForceDebiasDescriptor,
    polytope: FeasibleEventPolytope,
    plan_parameters: Sequence[float],
    feedback_parameters: Mapping[str, float],
    *,
    stage: str,
    candidate_id: int,
    local_index: int,
) -> dict[str, Any]:
    values = np.asarray(plan_parameters, dtype=np.float64)
    if not polytope.contains(values, tolerance=1e-8):
        raise ValueError("force-debias plan parameters left the feasible polytope")
    config = copy.deepcopy(center.config)
    base = np.asarray(polytope.base_waypoints_rad)
    basis = np.asarray(polytope.basis_waypoints_rad)
    waypoints = base + np.tensordot(values, basis, axes=(0, 0))
    waypoints[0] = 0.0
    original = ManipulationPlanParameters.from_config(config["manipulation_plan"])
    plan = ManipulationPlanParameters(
        schema_version=original.schema_version,
        profile=original.profile,
        duration_s=original.duration_s,
        knot_times_s=original.knot_times_s,
        actuator_waypoints_rad={
            name: tuple(float(item) for item in waypoints[:, index])
            for index, name in enumerate(ACTIVE_ACTUATORS)
        },
        desired_cube_position_delta_m=original.desired_cube_position_delta_m,
        desired_cube_rotation_vector_rad=original.desired_cube_rotation_vector_rad,
        max_knot_delta_rad=original.max_knot_delta_rad,
        trust_region_backtracks=original.trust_region_backtracks,
    )
    config["manipulation_plan"] = plan.as_config()
    config["control"]["manipulation_delta_rad"] = {
        name: float(waypoints[-1, index])
        for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    config["contact_feedback"] = _feedback_mapping(
        config["contact_feedback"], feedback_parameters
    )
    config["contact_force_targets_n"] = _force_target_mapping(
        config["contact_force_targets_n"], feedback_parameters["operation_scale"]
    )
    config["object_config_id"] = v14_object_config_id(config)
    config["grasp_pose_id"] = v14_grasp_pose_id(config)
    config["grasp_object_pair_id"] = v14_grasp_object_pair_id(config)
    projected = {
        **polytope.parameter_mapping(values),
        **{f"feedback:{key}": float(value) for key, value in feedback_parameters.items()},
    }
    config["planner_id"] = canonical_sha256(
        {
            "schema_version": FORCE_DEBIAS_SCHEMA_VERSION,
            "kind": "v14_contact_preserving_force_debias_plan",
            "source_center_id": center.center_id,
            "descriptor_id": descriptor.descriptor_id,
            "polytope_id": polytope.polytope_id,
            "stage": stage,
            "projected_parameters": projected,
        }
    )
    config["controller_id"] = _time_warp_controller_id(config)
    metadata = config.setdefault("candidate_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("force-debias parent candidate_metadata is malformed")
    metadata["v14_contact_preserving_force_debias_rescue"] = {
        "schema_version": FORCE_DEBIAS_SCHEMA_VERSION,
        "candidate_id": int(candidate_id),
        "local_index": int(local_index),
        "stage": stage,
        "source_candidate_id": center.candidate_id,
        "source_center_id": center.center_id,
        "descriptor_id": descriptor.descriptor_id,
        "polytope_id": polytope.polytope_id,
        "event_half_width_progress": polytope.bump_half_width_progress,
        "projected_parameters": projected,
        "full_reset_required": True,
    }
    validate_config(config)
    return config


def _feedback_from_unit(row: np.ndarray, tau_index: int) -> dict[str, float]:
    return {
        "kp_multiplier": float(KP_MULTIPLIER_RANGE[0] + row[0] * (KP_MULTIPLIER_RANGE[1] - KP_MULTIPLIER_RANGE[0])),
        "ki_multiplier": float(KI_MULTIPLIER_RANGE[0] + row[1] * (KI_MULTIPLIER_RANGE[1] - KI_MULTIPLIER_RANGE[0])),
        "integral_limit_n_s": float(INTEGRAL_LIMIT_RANGE_N_S[0] + row[2] * (INTEGRAL_LIMIT_RANGE_N_S[1] - INTEGRAL_LIMIT_RANGE_N_S[0])),
        "correction_limit_rad": float(CORRECTION_LIMIT_RANGE_RAD[0] + row[3] * (CORRECTION_LIMIT_RANGE_RAD[1] - CORRECTION_LIMIT_RANGE_RAD[0])),
        "filter_time_constant_s": FILTER_TIME_CONSTANTS_S[tau_index % len(FILTER_TIME_CONSTANTS_S)],
        "operation_scale": float(OPERATION_FORCE_SCALE_RANGE[0] + row[4] * (OPERATION_FORCE_SCALE_RANGE[1] - OPERATION_FORCE_SCALE_RANGE[0])),
    }


def _baseline_feedback_parameters() -> dict[str, float]:
    return {
        "kp_multiplier": 1.0,
        "ki_multiplier": 1.0,
        "integral_limit_n_s": 0.5,
        "correction_limit_rad": 0.06,
        "filter_time_constant_s": 0.005,
        "operation_scale": 1.0,
    }


def _structured_discovery_parameters(
    polytope: FeasibleEventPolytope, sample_index: int
) -> tuple[np.ndarray, dict[str, float], str, float]:
    if not 0 <= sample_index < STRUCTURED_DISCOVERY_ANCHOR_COUNT:
        raise ValueError("force-debias structured anchor index is invalid")
    requested = np.zeros(polytope.dimension, dtype=np.float64)
    feedback = _baseline_feedback_parameters()
    labels = (
        "near_identity_thumb",
        "near_identity_index",
        "near_identity_mid",
        "operation_scale_0p70",
        "operation_scale_0p85",
        "kp_multiplier_2p0",
        "ki_multiplier_1p5",
        "integral_limit_1p0",
        "correction_limit_0p075",
        "filter_tau_8ms",
        "all_finger_broad_0p001",
        "all_finger_broad_0p002",
    )
    if sample_index < 3:
        requested[sample_index] = 1e-6
    elif sample_index <= 9:
        # A distinct physically negligible plan epsilon keeps every
        # controller-only ablation unique without duplicating its sealed
        # source center.
        requested[0] = float(sample_index - 1) * 1e-6
        if sample_index == 3:
            feedback["operation_scale"] = 0.70
        elif sample_index == 4:
            feedback["operation_scale"] = 0.85
        elif sample_index == 5:
            feedback["kp_multiplier"] = 2.0
        elif sample_index == 6:
            feedback["ki_multiplier"] = 1.5
        elif sample_index == 7:
            feedback["integral_limit_n_s"] = 1.0
        elif sample_index == 8:
            feedback["correction_limit_rad"] = 0.075
        else:
            feedback["filter_time_constant_s"] = 0.008
    else:
        requested[:3] = 0.001 if sample_index == 10 else 0.002
    return requested, feedback, labels[sample_index], 0.0


def _small_radius_discovery_parameters(
    polytope: FeasibleEventPolytope,
    unit_row: np.ndarray,
    *,
    radius: float,
    filter_time_constant_s: float = 0.005,
) -> tuple[np.ndarray, dict[str, float]]:
    if unit_row.shape != (polytope.dimension + 5,):
        raise ValueError("force-debias small-radius LHS row has the wrong shape")
    if radius not in {value for value, _ in SMALL_RADIUS_DISCOVERY_COUNTS}:
        raise ValueError("force-debias small-radius shell is not registered")
    if filter_time_constant_s not in FILTER_TIME_CONSTANTS_S:
        raise ValueError("force-debias small-radius filter constant is not registered")
    lower = np.asarray(polytope.lower_bounds, dtype=np.float64)
    upper = np.asarray(polytope.upper_bounds, dtype=np.float64)
    requested = np.empty(polytope.dimension, dtype=np.float64)
    for index, unit in enumerate(unit_row[: polytope.dimension]):
        signed = 2.0 * float(unit) - 1.0
        span = upper[index] if signed >= 0.0 else abs(lower[index])
        requested[index] = radius * signed * span
        if lower[index] >= 0.0:
            requested[index] = radius * float(unit) * upper[index]
    feedback_unit = unit_row[polytope.dimension :]
    feedback = {
        "kp_multiplier": float(1.0 + radius * feedback_unit[0] * 2.0),
        "ki_multiplier": float(1.0 + radius * feedback_unit[1]),
        "integral_limit_n_s": float(0.5 + radius * feedback_unit[2] * 0.75),
        "correction_limit_rad": float(0.06 + radius * feedback_unit[3] * 0.03),
        "filter_time_constant_s": float(filter_time_constant_s),
        "operation_scale": float(1.0 - radius * feedback_unit[4] * 0.30),
    }
    return requested, feedback


def _small_radius_for_index(small_index: int) -> float:
    if not 0 <= small_index < sum(count for _, count in SMALL_RADIUS_DISCOVERY_COUNTS):
        raise ValueError("force-debias small-radius sample index is invalid")
    cursor = 0
    for radius, count in SMALL_RADIUS_DISCOVERY_COUNTS:
        if cursor <= small_index < cursor + count:
            return radius
        cursor += count
    raise RuntimeError("force-debias small-radius schedule is incomplete")


def _extract_job_metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value: Mapping[str, Any] = record
    for name in ("rescue_job", "job"):
        nested = value.get(name)
        if isinstance(nested, Mapping):
            value = nested
            break
    return value


def _candidate_id(
    *, source_authentication_id: str, center_id: str, stage: str,
    local_index: int, projected_parameters: Mapping[str, float]
) -> int:
    digest = canonical_sha256(
        {
            "schema_version": FORCE_DEBIAS_SCHEMA_VERSION,
            "source_authentication_id": source_authentication_id,
            "center_id": center_id,
            "stage": stage,
            "local_index": int(local_index),
            "projected_parameters": projected_parameters,
        }
    )
    return 14_900_000_000_000_000 + int(digest[:13], 16) % 100_000_000_000_000


def build_force_debias_jobs(
    source: ForceDebiasSource,
    *,
    stage: str,
    total_count: int,
    seed: int = DEFAULT_SEED,
    local_index_offset: int = 0,
    refinement_records: Sequence[Mapping[str, Any]] = (),
    excluded_physical_plan_sha256: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Build deterministic discovery or local-refinement full-reset jobs."""

    if stage not in {"discovery", "refinement"}:
        raise ValueError("unknown force-debias stage")
    if not isinstance(total_count, int) or isinstance(total_count, bool) or total_count <= 0:
        raise ValueError("force-debias total_count must be a positive integer")
    if local_index_offset < 0 or not isinstance(seed, int) or seed < 0:
        raise ValueError("force-debias index offset or seed is invalid")
    if len(source.centers) != 5 or any(
        center.source_authentication_id != source.source_authentication_id
        for center in source.centers
    ):
        raise RuntimeError("force-debias source center set is invalid")
    excluded = tuple(sorted(set(str(value) for value in excluded_physical_plan_sha256)))
    if any(not _is_sha256(value) for value in excluded):
        raise ValueError("force-debias exclusion set contains an invalid SHA-256")
    descriptors = {center.center_id: build_force_debias_descriptor(center) for center in source.centers}
    contexts: list[tuple[ForceDebiasCenter, ForceDebiasDescriptor, float, Mapping[str, Any] | None]] = []
    if stage == "discovery":
        expected_count = len(source.centers) * DISCOVERY_CANDIDATES_PER_CENTER
        if total_count != expected_count:
            raise ValueError(
                f"force-debias discovery budget must be exactly {expected_count}"
            )
        for center in source.centers:
            descriptor = descriptors[center.center_id]
            for width in EVENT_HALF_WIDTHS_PROGRESS:
                contexts.append((center, descriptor, width, None))
    else:
        if not refinement_records:
            raise ValueError("force-debias refinement requires source records")
        by_center = {center.center_id: center for center in source.centers}
        for raw in refinement_records:
            metadata = _extract_job_metadata(raw)
            center_id = str(metadata.get("source_center_id", ""))
            center = by_center.get(center_id)
            if center is None:
                raise ValueError("force-debias refinement record references an unknown center")
            width = float(metadata.get("event_half_width_progress", math.nan))
            if width not in EVENT_HALF_WIDTHS_PROGRESS:
                raise ValueError("force-debias refinement record has an invalid width")
            contexts.append((center, descriptors[center_id], width, metadata))
    if stage == "discovery":
        quotas = [
            len(range(width_index, DISCOVERY_CANDIDATES_PER_CENTER, 3))
            for _center in source.centers
            for width_index in range(len(EVENT_HALF_WIDTHS_PROGRESS))
        ]
    else:
        quotient, remainder = divmod(total_count, len(contexts))
        quotas = [quotient + int(index < remainder) for index in range(len(contexts))]
    if min(quotas) <= 0:
        raise ValueError("force-debias budget cannot cover every search context")
    excluded_sha = canonical_sha256(list(excluded))
    seen_physical = {*(center.physical_plan_sha256 for center in source.centers), *excluded}
    jobs: list[dict[str, Any]] = []
    for context_index, ((center, descriptor, width, anchor), quota) in enumerate(zip(contexts, quotas, strict=True)):
        polytope = derive_force_debias_polytope(
            center.config, descriptor, event_half_width_progress=width
        )
        dimensions = polytope.dimension + 5
        sampling_seed = int.from_bytes(
            bytes.fromhex(
                canonical_sha256(
                    {
                        "source_authentication_id": source.source_authentication_id,
                        "stage": stage,
                        "seed": seed,
                        "center_id": center.center_id,
                        "polytope_id": polytope.polytope_id,
                        "context_index": context_index,
                    }
                )[:16]
            ),
            "little",
        )
        pool_count = max(quota * 64, quota + 32)
        lhs = _latin_hypercube(pool_count, dimensions, sampling_seed)
        discovery_indices: tuple[int, ...] = ()
        discovery_lhs: np.ndarray | None = None
        if stage == "discovery":
            width_index = EVENT_HALF_WIDTHS_PROGRESS.index(width)
            discovery_indices = tuple(
                range(width_index, DISCOVERY_CANDIDATES_PER_CENTER, 3)
            )
            discovery_seed = int.from_bytes(
                bytes.fromhex(
                    canonical_sha256(
                        {
                            "source_authentication_id": source.source_authentication_id,
                            "stage": "discovery_small_radius",
                            "seed": seed,
                            "center_id": center.center_id,
                            "descriptor_id": descriptor.descriptor_id,
                        }
                    )[:16]
                ),
                "little",
            )
            discovery_lhs = _latin_hypercube(20, dimensions, discovery_seed)
            sampling_seed = discovery_seed
            pool_count = 20
        lower = np.asarray(polytope.lower_bounds)
        upper = np.asarray(polytope.upper_bounds)
        anchor_plan: np.ndarray | None = None
        anchor_feedback: dict[str, float] | None = None
        if anchor is not None:
            projected = anchor.get("projected_parameters")
            if not isinstance(projected, Mapping):
                raise ValueError("force-debias refinement record lost projected parameters")
            anchor_plan = polytope.parameter_vector(
                {name: projected[name] for name in polytope.parameter_names}
            )
            anchor_feedback = {
                name: float(projected[f"feedback:{name}"])
                for name in (
                    "kp_multiplier", "ki_multiplier", "integral_limit_n_s",
                    "correction_limit_rad", "filter_time_constant_s",
                    "operation_scale",
                )
            }
        accepted = 0
        rows = (
            ((sample_index, None) for sample_index in discovery_indices)
            if stage == "discovery"
            else enumerate(lhs)
        )
        for row_index, row in rows:
            if accepted >= quota:
                break
            if stage == "refinement":
                assert row is not None
                assert anchor_plan is not None and anchor_feedback is not None
                request = anchor_plan + 0.25 * (row[: polytope.dimension] - 0.5) * (upper - lower)
                request = np.clip(request, lower, upper)
                feedback = _feedback_from_unit(row[polytope.dimension :], row_index)
                sampling_mode = "refinement_lhs"
                structured_anchor: str | None = None
                normalized_radius = 0.25
            else:
                assert discovery_lhs is not None
                if row_index < STRUCTURED_DISCOVERY_ANCHOR_COUNT:
                    request, feedback, structured_anchor, normalized_radius = (
                        _structured_discovery_parameters(polytope, row_index)
                    )
                    sampling_mode = "structured_anchor"
                else:
                    small_index = row_index - STRUCTURED_DISCOVERY_ANCHOR_COUNT
                    normalized_radius = _small_radius_for_index(small_index)
                    request, feedback = _small_radius_discovery_parameters(
                        polytope,
                        discovery_lhs[small_index],
                        radius=normalized_radius,
                        filter_time_constant_s=(
                            0.012 if small_index == 19 else 0.005
                        ),
                    )
                    structured_anchor = None
                    sampling_mode = "small_radius_lhs"
            plan_parameters = polytope.project(request)
            if anchor_feedback is not None:
                feedback["kp_multiplier"] = float(np.clip(anchor_feedback["kp_multiplier"] + 0.25 * (feedback["kp_multiplier"] - 2.0), *KP_MULTIPLIER_RANGE))
                feedback["ki_multiplier"] = float(np.clip(anchor_feedback["ki_multiplier"] + 0.25 * (feedback["ki_multiplier"] - 1.5), *KI_MULTIPLIER_RANGE))
                feedback["integral_limit_n_s"] = float(np.clip(anchor_feedback["integral_limit_n_s"] + 0.25 * (feedback["integral_limit_n_s"] - 0.875), *INTEGRAL_LIMIT_RANGE_N_S))
                feedback["correction_limit_rad"] = float(np.clip(anchor_feedback["correction_limit_rad"] + 0.25 * (feedback["correction_limit_rad"] - 0.075), *CORRECTION_LIMIT_RANGE_RAD))
                feedback["operation_scale"] = float(np.clip(anchor_feedback["operation_scale"] + 0.25 * (feedback["operation_scale"] - 0.85), *OPERATION_FORCE_SCALE_RANGE))
                # Discrete tau is deliberately still explored in refinement.
            projected = {
                **polytope.parameter_mapping(plan_parameters),
                **{f"feedback:{name}": float(value) for name, value in feedback.items()},
            }
            local_index = local_index_offset + len(jobs)
            candidate_id = _candidate_id(
                source_authentication_id=source.source_authentication_id,
                center_id=center.center_id,
                stage=stage,
                local_index=local_index,
                projected_parameters=projected,
            )
            config = _candidate_config(
                center, descriptor, polytope, plan_parameters, feedback,
                stage=stage, candidate_id=candidate_id, local_index=local_index,
            )
            physical = physical_plan_sha256(config)
            if physical in seen_physical:
                continue
            seen_physical.add(physical)
            core = {
                "schema_version": FORCE_DEBIAS_SCHEMA_VERSION,
                "kind": "v14_contact_preserving_force_debias_candidate",
                "stage": stage,
                "candidate_id": candidate_id,
                "local_index": local_index,
                "source_authentication_id": source.source_authentication_id,
                "source_candidate_id": center.candidate_id,
                "source_center_id": center.center_id,
                "source_physical_plan_sha256": center.physical_plan_sha256,
                "descriptor_id": descriptor.descriptor_id,
                "polytope_id": polytope.polytope_id,
                "event_half_width_progress": width,
                "projected_parameters": projected,
                "feedback_parameters": feedback,
                "sampling_seed": sampling_seed,
                "lhs_pool_count": pool_count,
                "lhs_pool_row_index": row_index,
                "context_index": context_index,
                "sampling_mode": sampling_mode,
                "structured_anchor": structured_anchor,
                "normalized_radius": float(normalized_radius),
                "total_candidate_count": total_count,
                "requested_seed": seed,
                "local_index_offset": local_index_offset,
                "refinement_anchor_candidate_id": (
                    None if anchor is None else int(anchor.get("candidate_id", -1))
                ),
                "excluded_physical_plan_set_sha256": excluded_sha,
                "excluded_physical_plan_count": len(excluded),
                "config_semantic_sha256": canonical_sha256(config),
                "physical_plan_sha256": physical,
                "full_reset_required": True,
            }
            jobs.append(
                {
                    **core,
                    "candidate_payload_sha256": canonical_sha256(core),
                    "config": config,
                }
            )
            accepted += 1
        if accepted != quota:
            raise RuntimeError("force-debias projected sampling exhausted its unique pool")
    if len(jobs) != total_count:
        raise RuntimeError("force-debias job budget was not filled")
    return tuple(jobs)


def authenticate_force_debias_job(
    job: Mapping[str, Any],
    source: ForceDebiasSource,
    *,
    expected_stage: str | None = None,
    expected_excluded_physical_plan_sha256: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fail closed on job/config/provenance or physical-plan mutation."""

    value = copy.deepcopy(dict(job))
    config = value.pop("config", None)
    recorded_payload = value.pop("candidate_payload_sha256", None)
    if not isinstance(config, Mapping) or recorded_payload != canonical_sha256(value):
        raise RuntimeError("force-debias job payload SHA-256 changed")
    if (
        value.get("kind") != "v14_contact_preserving_force_debias_candidate"
        or value.get("full_reset_required") is not True
        or value.get("source_authentication_id") != source.source_authentication_id
    ):
        raise RuntimeError("force-debias job provenance changed")
    stage = str(value.get("stage", ""))
    if stage not in {"discovery", "refinement"} or (
        expected_stage is not None and stage != expected_stage
    ):
        raise RuntimeError("force-debias job stage changed")
    centers = {center.center_id: center for center in source.centers}
    center = centers.get(str(value.get("source_center_id", "")))
    if center is None or int(value.get("source_candidate_id", -1)) != center.candidate_id:
        raise RuntimeError("force-debias job source center changed")
    if value.get("source_physical_plan_sha256") != center.physical_plan_sha256:
        raise RuntimeError("force-debias job source physical provenance changed")
    descriptor = build_force_debias_descriptor(center)
    if value.get("descriptor_id") != descriptor.descriptor_id:
        raise RuntimeError("force-debias job descriptor changed")
    width = float(value.get("event_half_width_progress", math.nan))
    polytope = derive_force_debias_polytope(
        center.config, descriptor, event_half_width_progress=width
    )
    if value.get("polytope_id") != polytope.polytope_id:
        raise RuntimeError("force-debias job polytope changed")
    projected = value.get("projected_parameters")
    feedback = value.get("feedback_parameters")
    if not isinstance(projected, Mapping) or not isinstance(feedback, Mapping):
        raise RuntimeError("force-debias job parameters changed")
    plan_values = polytope.parameter_vector(
        {name: projected[name] for name in polytope.parameter_names}
    )
    if not polytope.contains(plan_values, tolerance=1e-8):
        raise RuntimeError("force-debias job left the feasible polytope")
    expected_feedback = {
        name: float(projected[f"feedback:{name}"])
        for name in (
            "kp_multiplier", "ki_multiplier", "integral_limit_n_s",
            "correction_limit_rad", "filter_time_constant_s",
            "operation_scale",
        )
    }
    if canonical_sha256(expected_feedback) != canonical_sha256(feedback):
        raise RuntimeError("force-debias job feedback parameters changed")
    bounded_feedback = (
        ("kp_multiplier", KP_MULTIPLIER_RANGE),
        ("ki_multiplier", KI_MULTIPLIER_RANGE),
        ("integral_limit_n_s", INTEGRAL_LIMIT_RANGE_N_S),
        ("correction_limit_rad", CORRECTION_LIMIT_RANGE_RAD),
        ("operation_scale", OPERATION_FORCE_SCALE_RANGE),
    )
    if any(
        not lower <= expected_feedback[name] <= upper
        for name, (lower, upper) in bounded_feedback
    ) or expected_feedback["filter_time_constant_s"] not in FILTER_TIME_CONSTANTS_S:
        raise RuntimeError("force-debias job feedback parameters left registered bounds")
    if stage == "discovery":
        sample_index = int(value.get("lhs_pool_row_index", -1))
        if not 0 <= sample_index < DISCOVERY_CANDIDATES_PER_CENTER:
            raise RuntimeError("force-debias discovery sample index changed")
        expected_width = EVENT_HALF_WIDTHS_PROGRESS[
            sample_index % len(EVENT_HALF_WIDTHS_PROGRESS)
        ]
        if width != expected_width:
            raise RuntimeError("force-debias discovery width schedule changed")
        dimensions = polytope.dimension + 5
        expected_seed = int.from_bytes(
            bytes.fromhex(
                canonical_sha256(
                    {
                        "source_authentication_id": source.source_authentication_id,
                        "stage": "discovery_small_radius",
                        "seed": int(value.get("requested_seed", -1)),
                        "center_id": center.center_id,
                        "descriptor_id": descriptor.descriptor_id,
                    }
                )[:16]
            ),
            "little",
        )
        if (
            int(value.get("sampling_seed", -1)) != expected_seed
            or int(value.get("lhs_pool_count", -1)) != 20
        ):
            raise RuntimeError("force-debias discovery sampling evidence changed")
        if sample_index < STRUCTURED_DISCOVERY_ANCHOR_COUNT:
            expected_request, expected_parameters, expected_anchor, expected_radius = (
                _structured_discovery_parameters(polytope, sample_index)
            )
            expected_mode = "structured_anchor"
        else:
            small_index = sample_index - STRUCTURED_DISCOVERY_ANCHOR_COUNT
            expected_radius = _small_radius_for_index(small_index)
            lhs = _latin_hypercube(20, dimensions, expected_seed)
            expected_request, expected_parameters = (
                _small_radius_discovery_parameters(
                    polytope,
                    lhs[small_index],
                    radius=expected_radius,
                    filter_time_constant_s=(
                        0.012 if small_index == 19 else 0.005
                    ),
                )
            )
            expected_anchor = None
            expected_mode = "small_radius_lhs"
        expected_plan = polytope.project(expected_request)
        if not np.allclose(expected_plan, plan_values, rtol=0.0, atol=1e-12):
            raise RuntimeError("force-debias discovery plan is not reproducible")
        if canonical_sha256(expected_parameters) != canonical_sha256(expected_feedback):
            raise RuntimeError("force-debias discovery feedback is not reproducible")
        if (
            value.get("sampling_mode") != expected_mode
            or value.get("structured_anchor") != expected_anchor
            or not math.isclose(
                float(value.get("normalized_radius", math.nan)),
                expected_radius,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise RuntimeError("force-debias discovery anchor/radius identity changed")
    local_index = int(value.get("local_index", -1))
    expected_id = _candidate_id(
        source_authentication_id=source.source_authentication_id,
        center_id=center.center_id,
        stage=stage,
        local_index=local_index,
        projected_parameters={str(key): float(item) for key, item in projected.items()},
    )
    if int(value.get("candidate_id", -1)) != expected_id:
        raise RuntimeError("force-debias job candidate ID is not reproducible")
    expected_config = _candidate_config(
        center, descriptor, polytope, plan_values, expected_feedback,
        stage=stage, candidate_id=expected_id, local_index=local_index,
    )
    if canonical_sha256(expected_config) != canonical_sha256(config):
        raise RuntimeError("force-debias job config is not reproducible")
    if value.get("config_semantic_sha256") != canonical_sha256(config):
        raise RuntimeError("force-debias job config SHA-256 changed")
    physical = physical_plan_sha256(config)
    if physical == center.physical_plan_sha256 or value.get("physical_plan_sha256") != physical:
        raise RuntimeError("force-debias job physical plan changed or duplicates its source")
    if expected_excluded_physical_plan_sha256 is not None:
        excluded = tuple(sorted(set(str(item) for item in expected_excluded_physical_plan_sha256)))
        if any(not _is_sha256(item) for item in excluded):
            raise ValueError("expected force-debias exclusion set is invalid")
        if (
            value.get("excluded_physical_plan_set_sha256") != canonical_sha256(list(excluded))
            or int(value.get("excluded_physical_plan_count", -1)) != len(excluded)
            or physical in set(excluded)
        ):
            raise RuntimeError("force-debias job exclusion-set provenance changed")
    return copy.deepcopy(dict(job))


def force_debias_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Keep hard/contact policy first, then delegate jerk-aware tie breaking."""

    return adaptive_event_candidate_rank(record)


__all__ = [
    "BROAD_UNLOAD_MAX_RAD",
    "CORRECTION_LIMIT_RANGE_RAD",
    "DEFAULT_SEED",
    "EVENT_HALF_WIDTHS_PROGRESS",
    "FILTER_TIME_CONSTANTS_S",
    "FORCE_DEBIAS_SCHEMA_VERSION",
    "ForceDebiasCenter",
    "ForceDebiasDescriptor",
    "ForceDebiasSource",
    "INTEGRAL_LIMIT_RANGE_N_S",
    "KI_MULTIPLIER_RANGE",
    "KP_MULTIPLIER_RANGE",
    "OPERATION_FORCE_SCALE_RANGE",
    "authenticate_force_debias_job",
    "authenticate_force_debias_source",
    "broad_c2_force_debias_envelope",
    "build_force_debias_descriptor",
    "build_force_debias_jobs",
    "derive_force_debias_polytope",
    "force_debias_candidate_rank",
    "physical_plan_sha256",
]
