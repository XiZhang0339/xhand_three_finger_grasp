"""Pure helpers for high-thumb, grasp-only schema-v5 searches.

This module deliberately does not register an experiment or dispatch from the
public CLI.  It turns an already validated schema-v5 configuration into
constant-density, zero-manipulation candidates and provides deterministic
candidate generation, ranking and injected-runner batching.  A later
experiment can therefore compose these pieces without changing the existing
far-hand lift tuner.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from ..config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    resolved_pose_constraint_values,
    validate_config,
)
from ..experiment import ExperimentDefinition, resolve_experiment
from ..scene import (
    cube_vertical_half_extent_m,
    rpy_degrees_to_rotation_matrix,
)
from .far_hand_fingertip import root_pitch_for_finger_down_tilt_deg


THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"

CandidateRunner = Callable[
    [list[tuple[int, dict[str, Any]]], int], list[dict[str, Any]]
]


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite_axis(
    values: Iterable[float], label: str, *, positive: bool = False
) -> tuple[float, ...]:
    result = tuple(_finite(value, label) for value in values)
    if not result:
        raise ValueError(f"{label} must not be empty")
    if positive and any(value <= 0.0 for value in result):
        raise ValueError(f"{label} values must be positive")
    return result


def _schema_v5_definition(base: Mapping[str, Any]) -> ExperimentDefinition:
    candidate = copy.deepcopy(dict(base))
    validate_config(candidate)
    if int(candidate.get("schema_version", 0)) != 5:
        raise ValueError("high-thumb candidates require a schema-v5 base config")
    definition = resolve_experiment(candidate)
    if definition.far_hand_campaign is None:
        raise ValueError("schema-v5 base is missing a far-hand campaign")
    return definition


def _target_mapping(
    values: Mapping[str, float], *, label: str
) -> dict[str, float]:
    if set(values) != set(ACTIVE_ACTUATORS):
        raise ValueError(f"{label} must contain exactly the eight active actuators")
    return {
        name: _finite(values[name], f"{label}.{name}")
        for name in ACTIVE_ACTUATORS
    }


def cube_in_root_from_distance_yz(
    root_cube_distance_m: float,
    cube_in_root_y_m: float,
    cube_in_root_z_m: float,
    *,
    positive_x: bool = True,
) -> tuple[float, float, float]:
    """Resolve a complete root-frame vector from distance and its Y/Z values."""

    distance = _positive(root_cube_distance_m, "root_cube_distance_m")
    y_value = _finite(cube_in_root_y_m, "cube_in_root_y_m")
    z_value = _finite(cube_in_root_z_m, "cube_in_root_z_m")
    x_squared = distance * distance - y_value * y_value - z_value * z_value
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, distance * distance)
    if x_squared < -tolerance:
        raise ValueError(
            "root_cube_distance_m is shorter than the requested Y/Z projection"
        )
    x_value = math.sqrt(max(0.0, x_squared))
    if not positive_x:
        x_value = -x_value
    result = (float(x_value), y_value, z_value)
    if not math.isclose(
        math.sqrt(sum(value * value for value in result)),
        distance,
        rel_tol=0.0,
        abs_tol=2e-15,
    ):
        raise RuntimeError("distance/Y/Z reconstruction lost numerical precision")
    return result


def _resolved_defaults(
    base: Mapping[str, Any], definition: ExperimentDefinition
) -> dict[str, Any]:
    resolved = resolved_pose_constraint_values(copy.deepcopy(dict(base)))
    hand_rpy = tuple(float(value) for value in base["hand_pose"]["rpy_deg"])
    campaign = definition.far_hand_campaign
    assert campaign is not None
    tilt = float(resolved["finger_down_tilt_deg"])
    band = min(
        (float(value) for value in campaign.tilt_band_centers_deg),
        key=lambda value: (abs(value - tilt), value),
    )
    return {
        "finger_down_tilt_deg": tilt,
        "tilt_band_center_deg": band,
        "hand_roll_deg": hand_rpy[0],
        "hand_yaw_deg": hand_rpy[2],
        "target_faces": copy.deepcopy(
            base["contact_topology"]["target_faces"]
        ),
    }


def _cube_world_position_m(config: Mapping[str, Any]) -> np.ndarray:
    cube = config["cube"]
    rotation = rpy_degrees_to_rotation_matrix(cube["rpy_deg"])
    return np.asarray(
        [
            float(cube["center_xy_m"][0]),
            float(cube["center_xy_m"][1]),
            float(config["scene"]["support_top_z_m"])
            + cube_vertical_half_extent_m(float(cube["edge_m"]), rotation)
            + float(cube.get("z_offset_m", 0.0)),
        ],
        dtype=np.float64,
    )


def _is_canonical_far_hand_candidate(
    candidate: Mapping[str, Any], definition: ExperimentDefinition
) -> bool:
    """Return whether schema-v5 should apply its nominal campaign checks."""

    campaign = definition.far_hand_campaign
    constraints = definition.far_hand_pose_constraints
    if campaign is None or constraints is None:
        return False
    cube = candidate["cube"]
    nominal_material = all(
        math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15)
        for actual, expected in (
            (cube["edge_m"], campaign.nominal_edge_m),
            (cube["mass_kg"], campaign.nominal_mass_kg),
            (cube["friction"], campaign.friction),
        )
    )
    resolved = resolved_pose_constraint_values(copy.deepcopy(dict(candidate)))
    tilt = float(resolved["finger_down_tilt_deg"])
    palm = float(resolved["palm_plane_ground_angle_deg"])
    hand_rpy = tuple(float(value) for value in candidate["hand_pose"]["rpy_deg"])
    cube_rpy = tuple(float(value) for value in cube["rpy_deg"])
    bounds = definition.search_bounds
    nominal_pose = bool(
        constraints.finger_down_tilt_deg[0] - 1e-12
        <= tilt
        <= constraints.finger_down_tilt_deg[1] + 1e-12
        and constraints.palm_plane_ground_angle_deg[0] - 1e-12
        <= palm
        <= constraints.palm_plane_ground_angle_deg[1] + 1e-12
        and constraints.contains_cube_position(
            resolved["cube_position_in_root_m"]
        )
        and bounds.hand_roll_deg[0] - 1e-12
        <= hand_rpy[0]
        <= bounds.hand_roll_deg[1] + 1e-12
        and bounds.hand_yaw_deg[0] - 1e-12
        <= hand_rpy[2]
        <= bounds.hand_yaw_deg[1] + 1e-12
        and bounds.cube_yaw_deg[0] - 1e-12
        <= cube_rpy[2]
        <= bounds.cube_yaw_deg[1] + 1e-12
    )
    return nominal_material and nominal_pose


def materialize_high_thumb_candidate(
    base: Mapping[str, Any],
    *,
    edge_m: float,
    cube_in_root_m: Sequence[float],
    cube_yaw_deg: float,
    grasp_targets_rad: Mapping[str, float],
    finger_down_tilt_deg: float | None = None,
    tilt_band_center_deg: float | None = None,
    hand_roll_deg: float | None = None,
    hand_yaw_deg: float | None = None,
    target_faces: Mapping[str, str] | None = None,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> dict[str, Any]:
    """Materialize one constant-density, zero-manipulation schema-v5 candidate.

    The complete ``cube_in_root_m`` vector is authoritative.  Nonnominal
    material or poses are explicitly marked as ``parameter_override_run`` so
    the unchanged schema-v5 validator evaluates them as newly simulated
    parameter probes rather than inheriting the source campaign's success.
    """

    definition = _schema_v5_definition(base)
    campaign = definition.far_hand_campaign
    assert campaign is not None
    edge = _positive(edge_m, "edge_m")
    cube_yaw = _finite(cube_yaw_deg, "cube_yaw_deg")
    relative = np.asarray(tuple(cube_in_root_m), dtype=np.float64)
    if relative.shape != (3,) or not np.isfinite(relative).all():
        raise ValueError("cube_in_root_m must contain three finite values")
    if float(np.linalg.norm(relative)) <= 0.0:
        raise ValueError("cube_in_root_m must have non-zero length")
    targets = _target_mapping(grasp_targets_rad, label="grasp_targets_rad")
    defaults = _resolved_defaults(base, definition)
    tilt = _finite(
        defaults["finger_down_tilt_deg"]
        if finger_down_tilt_deg is None
        else finger_down_tilt_deg,
        "finger_down_tilt_deg",
    )
    band = _finite(
        defaults["tilt_band_center_deg"]
        if tilt_band_center_deg is None
        else tilt_band_center_deg,
        "tilt_band_center_deg",
    )
    roll = _finite(
        defaults["hand_roll_deg"] if hand_roll_deg is None else hand_roll_deg,
        "hand_roll_deg",
    )
    hand_yaw = _finite(
        defaults["hand_yaw_deg"] if hand_yaw_deg is None else hand_yaw_deg,
        "hand_yaw_deg",
    )

    candidate = copy.deepcopy(dict(base))
    candidate.pop("experiment_status", None)
    candidate.pop("candidate_metadata", None)
    candidate.pop("run_context", None)
    candidate["cube"]["edge_m"] = edge
    candidate["cube"]["mass_kg"] = campaign.constant_density_mass_kg(edge)
    candidate["cube"]["friction"] = campaign.friction
    candidate["cube"]["rpy_deg"] = [0.0, 0.0, cube_yaw]
    pitch = root_pitch_for_finger_down_tilt_deg(roll, tilt)
    root_rpy = [roll, pitch, hand_yaw]
    root_rotation = rpy_degrees_to_rotation_matrix(root_rpy)
    root_translation = _cube_world_position_m(candidate) - root_rotation @ relative
    candidate["hand_pose"] = {
        "translation_m": root_translation.tolist(),
        "rpy_deg": root_rpy,
    }
    candidate["control"] = {
        "grasp_targets_rad": targets,
        "manipulation_delta_rad": {
            name: 0.0 for name in ACTIVE_ACTUATORS
        },
    }
    candidate["contact_topology"]["target_faces"] = copy.deepcopy(
        defaults["target_faces"] if target_faces is None else dict(target_faces)
    )
    candidate["candidate_metadata"] = {
        "search_stage": "high_thumb_grasp_materialization",
        "search_scope": "stable_grasp_only",
        "tilt_band_center_deg": band,
        "resolved_finger_down_tilt_deg": tilt,
        "cube_in_root_m": relative.tolist(),
        "root_cube_distance_m": float(np.linalg.norm(relative)),
        "edge_m": edge,
        "density_kg_m3": float(campaign.density_kg_m3),
        "thumb_bend_target_rad": targets[THUMB_BEND_ACTUATOR],
        "zero_manipulation_delta": True,
    }

    # Existing schema-v5 semantics deliberately bind canonical configs to one
    # nominal material and pose envelope.  Search probes outside that envelope
    # must be provenance-marked before validation and independent simulation.
    if not _is_canonical_far_hand_candidate(candidate, definition):
        candidate["run_context"] = {"kind": "parameter_override_run"}
    if validator is not None:
        validator(candidate)
    return candidate


@dataclass(frozen=True)
class HighThumbCartesianGrid:
    """Ordered axes for a deterministic high-thumb Cartesian seed grid."""

    edge_m: tuple[float, ...]
    root_cube_distance_m: tuple[float, ...]
    cube_in_root_y_m: tuple[float, ...]
    cube_in_root_z_m: tuple[float, ...]
    cube_yaw_deg: tuple[float, ...]
    thumb_bend_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "edge_m", _finite_axis(self.edge_m, "edge_m", positive=True)
        )
        object.__setattr__(
            self,
            "root_cube_distance_m",
            _finite_axis(
                self.root_cube_distance_m,
                "root_cube_distance_m",
                positive=True,
            ),
        )
        for name in (
            "cube_in_root_y_m",
            "cube_in_root_z_m",
            "cube_yaw_deg",
            "thumb_bend_rad",
        ):
            object.__setattr__(self, name, _finite_axis(getattr(self, name), name))

    @property
    def candidate_count(self) -> int:
        return math.prod(
            len(getattr(self, name))
            for name in (
                "edge_m",
                "root_cube_distance_m",
                "cube_in_root_y_m",
                "cube_in_root_z_m",
                "cube_yaw_deg",
                "thumb_bend_rad",
            )
        )


REFERENCE_HIGH_THUMB_GRID = HighThumbCartesianGrid(
    edge_m=(0.060,),
    root_cube_distance_m=(0.158,),
    cube_in_root_y_m=(-0.027,),
    cube_in_root_z_m=(0.1195,),
    cube_yaw_deg=(26.0,),
    thumb_bend_rad=(1.15,),
)


def generate_high_thumb_cartesian_candidates(
    base: Mapping[str, Any],
    grid: HighThumbCartesianGrid,
    *,
    grasp_targets_rad: Mapping[str, float] | None = None,
    finger_down_tilt_deg: float | None = None,
    tilt_band_center_deg: float | None = None,
    hand_roll_deg: float | None = None,
    hand_yaw_deg: float | None = None,
    target_faces: Mapping[str, str] | None = None,
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    """Generate candidates in stable, documented Cartesian-product order."""

    _schema_v5_definition(base)
    template = _target_mapping(
        base["control"]["grasp_targets_rad"]
        if grasp_targets_rad is None
        else grasp_targets_rad,
        label="grasp_targets_rad",
    )
    candidates: list[dict[str, Any]] = []
    axes = product(
        grid.edge_m,
        grid.root_cube_distance_m,
        grid.cube_in_root_y_m,
        grid.cube_in_root_z_m,
        grid.cube_yaw_deg,
        grid.thumb_bend_rad,
    )
    for index, (edge, distance, y_value, z_value, cube_yaw, thumb) in enumerate(
        axes
    ):
        targets = dict(template)
        targets[THUMB_BEND_ACTUATOR] = float(thumb)
        relative = cube_in_root_from_distance_yz(distance, y_value, z_value)
        candidate = materialize_high_thumb_candidate(
            base,
            edge_m=edge,
            cube_in_root_m=relative,
            cube_yaw_deg=cube_yaw,
            grasp_targets_rad=targets,
            finger_down_tilt_deg=finger_down_tilt_deg,
            tilt_band_center_deg=tilt_band_center_deg,
            hand_roll_deg=hand_roll_deg,
            hand_yaw_deg=hand_yaw_deg,
            target_faces=target_faces,
            validator=validator,
        )
        candidate["candidate_metadata"].update(
            {
                "search_stage": "high_thumb_grasp_cartesian",
                "cartesian_candidate_index": index,
                "requested_root_cube_distance_m": float(distance),
                "requested_cube_in_root_y_m": float(y_value),
                "requested_cube_in_root_z_m": float(z_value),
            }
        )
        candidates.append(candidate)
    return tuple(candidates)


@dataclass(frozen=True)
class HighThumbLocalSearch:
    """Gaussian local-search radii around one materialized candidate."""

    edge_radius_m: float = 0.0015
    root_cube_distance_radius_m: float = 0.006
    cube_in_root_y_radius_m: float = 0.003
    cube_in_root_z_radius_m: float = 0.003
    cube_yaw_radius_deg: float = 3.0
    target_radius_rad: float = 0.05
    thumb_bend_radius_rad: float = 0.12
    minimum_thumb_bend_rad: float = 1.10

    def __post_init__(self) -> None:
        for name in (
            "edge_radius_m",
            "root_cube_distance_radius_m",
            "cube_in_root_y_radius_m",
            "cube_in_root_z_radius_m",
            "cube_yaw_radius_deg",
            "target_radius_rad",
            "thumb_bend_radius_rad",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        minimum = _finite(self.minimum_thumb_bend_rad, "minimum_thumb_bend_rad")
        object.__setattr__(self, "minimum_thumb_bend_rad", minimum)


def _project_yz_inside_distance(
    y_value: float, z_value: float, distance: float
) -> tuple[float, float]:
    yz_norm = math.hypot(y_value, z_value)
    maximum = max(0.0, distance * (1.0 - 1e-9))
    if yz_norm <= maximum or yz_norm == 0.0:
        return y_value, z_value
    scale = maximum / yz_norm
    return y_value * scale, z_value * scale


def generate_local_high_thumb_candidates(
    parent: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    search: HighThumbLocalSearch = HighThumbLocalSearch(),
    validator: Callable[[dict[str, Any]], None] | None = validate_config,
) -> tuple[dict[str, Any], ...]:
    """Generate an exact parent followed by deterministic local perturbations."""

    sample_count = _positive_integer(count, "count")
    random_seed = _nonnegative_integer(seed, "seed")
    base = copy.deepcopy(dict(parent.get("config", parent)))
    definition = _schema_v5_definition(base)
    resolved = resolved_pose_constraint_values(base)
    parent_relative = np.asarray(
        resolved["cube_position_in_root_m"], dtype=np.float64
    )
    parent_distance = float(np.linalg.norm(parent_relative))
    parent_edge = float(base["cube"]["edge_m"])
    parent_yaw = float(base["cube"]["rpy_deg"][2])
    parent_targets = _target_mapping(
        base["control"]["grasp_targets_rad"], label="grasp_targets_rad"
    )
    defaults = _resolved_defaults(base, definition)
    metadata = base.get("candidate_metadata", {})
    band = float(
        metadata.get(
            "tilt_band_center_deg", defaults["tilt_band_center_deg"]
        )
    )
    bounds = definition.search_bounds.actuator_targets_rad
    rng = np.random.default_rng(random_seed)
    candidates: list[dict[str, Any]] = []
    for index in range(sample_count):
        if index == 0:
            edge = parent_edge
            distance = parent_distance
            y_value = float(parent_relative[1])
            z_value = float(parent_relative[2])
            cube_yaw = parent_yaw
            targets = dict(parent_targets)
        else:
            edge = max(
                np.finfo(np.float64).eps,
                parent_edge + rng.normal(0.0, search.edge_radius_m),
            )
            distance = max(
                np.finfo(np.float64).eps,
                parent_distance
                + rng.normal(0.0, search.root_cube_distance_radius_m),
            )
            y_value = float(
                parent_relative[1]
                + rng.normal(0.0, search.cube_in_root_y_radius_m)
            )
            z_value = float(
                parent_relative[2]
                + rng.normal(0.0, search.cube_in_root_z_radius_m)
            )
            y_value, z_value = _project_yz_inside_distance(
                y_value, z_value, distance
            )
            cube_yaw = parent_yaw + rng.normal(0.0, search.cube_yaw_radius_deg)
            targets = {}
            for name in ACTIVE_ACTUATORS:
                lower, upper = (float(value) for value in bounds[name])
                if name == THUMB_BEND_ACTUATOR:
                    lower = max(lower, search.minimum_thumb_bend_rad)
                    radius = search.thumb_bend_radius_rad
                else:
                    radius = search.target_radius_rad
                if lower > upper:
                    raise ValueError(
                        "minimum_thumb_bend_rad exceeds registered target bounds"
                    )
                targets[name] = float(
                    np.clip(
                        parent_targets[name] + rng.normal(0.0, radius),
                        lower,
                        upper,
                    )
                )
        relative = cube_in_root_from_distance_yz(
            distance, y_value, z_value
        )
        candidate = materialize_high_thumb_candidate(
            base,
            edge_m=float(edge),
            cube_in_root_m=relative,
            cube_yaw_deg=float(cube_yaw),
            grasp_targets_rad=targets,
            finger_down_tilt_deg=float(resolved["finger_down_tilt_deg"]),
            tilt_band_center_deg=band,
            hand_roll_deg=float(base["hand_pose"]["rpy_deg"][0]),
            hand_yaw_deg=float(base["hand_pose"]["rpy_deg"][2]),
            target_faces=base["contact_topology"]["target_faces"],
            validator=validator,
        )
        candidate["candidate_metadata"].update(
            {
                "search_stage": "high_thumb_grasp_local",
                "local_search_seed": random_seed,
                "local_candidate_index": index,
                "local_parent_exact": index == 0,
            }
        )
        candidates.append(candidate)
    return tuple(candidates)


def _summary(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = result.get("summary", {})
    return value if isinstance(value, Mapping) else {}


def _metrics(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _summary(result).get("metrics", {})
    return value if isinstance(value, Mapping) else {}


def _finite_metric(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _finger_metric(metrics: Mapping[str, Any], name: str) -> tuple[float, ...]:
    value = metrics.get(name, {})
    mapping = value if isinstance(value, Mapping) else {}
    return tuple(
        sorted(
            _finite_metric(mapping.get(finger), 0.0)
            for finger in ACTIVE_FINGERS
        )
    )


def _grasp_success(result: Mapping[str, Any]) -> bool:
    status = _summary(result).get("stage_status", {})
    return bool(
        isinstance(status, Mapping) and status.get("grasp_success") is True
    )


def _grasp_margin(metrics: Mapping[str, Any]) -> float:
    direct = _finite_metric(
        metrics.get(
            "grasp_stability_margin",
            metrics.get("grasp_minimum_normalized_margin"),
        ),
        -math.inf,
    )
    if math.isfinite(direct):
        return direct
    reached = _finite_metric(
        metrics.get(
            "verify_max_consecutive_all_gate_steps",
            metrics.get(
                "verify_max_consecutive_gate_steps",
                metrics.get("grasp_gate_final_consecutive_steps"),
            ),
        ),
        -math.inf,
    )
    required = _finite_metric(metrics.get("grasp_stable_window_steps"), -math.inf)
    if math.isfinite(reached) and math.isfinite(required) and required > 0.0:
        return (reached - required) / required
    return -math.inf


def _verify_rank(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    components = metrics.get("verify_gate_component_duty", {})
    if isinstance(components, Mapping) and components:
        component_values = tuple(
            _finite_metric(value, 0.0) for value in components.values()
        )
        component_minimum = min(component_values)
        component_mean = sum(component_values) / len(component_values)
    else:
        component_minimum = 0.0
        component_mean = 0.0
    return (
        _finite_metric(metrics.get("verify_effective_finger_count"), 0.0),
        _finite_metric(
            metrics.get("verify_max_simultaneous_effective_finger_count"), 0.0
        ),
        *_finger_metric(metrics, "verify_target_face_effective_duty"),
        _finite_metric(metrics.get("verify_target_face_simultaneous_duty"), 0.0),
        *_finger_metric(metrics, "verify_peak_target_face_force_n"),
        *_finger_metric(metrics, "verify_peak_tactile_n"),
        _finite_metric(
            metrics.get(
                "verify_max_consecutive_all_gate_steps",
                metrics.get("verify_max_consecutive_gate_steps"),
            ),
            0.0,
        ),
        _finite_metric(metrics.get("verify_all_gate_duty"), 0.0),
        component_minimum,
        component_mean,
    )


def _alignment_rank(metrics: Mapping[str, Any]) -> tuple[float, float]:
    alignment = metrics.get("contact_alignment", {})
    verify = alignment.get("verify", {}) if isinstance(alignment, Mapping) else {}
    if not isinstance(verify, Mapping):
        verify = {}
    return (
        _finite_metric(verify.get("aligned_duty"), 0.0),
        -_finite_metric(verify.get("height_spread_p95_m"), math.inf),
    )


def _pad_rank(metrics: Mapping[str, Any]) -> tuple[float, float]:
    fingertip = metrics.get("fingertip_contact", {})
    verify = fingertip.get("verify", {}) if isinstance(fingertip, Mapping) else {}
    if not isinstance(verify, Mapping):
        return -math.inf, -math.inf
    fractions = verify.get("force_weighted_pad_fraction", {})
    taxels = verify.get("max_active_taxel_count", {})
    if not isinstance(fractions, Mapping) or not isinstance(taxels, Mapping):
        return -math.inf, -math.inf
    return (
        min(
            _finite_metric(fractions.get(finger), 0.0)
            for finger in ACTIVE_FINGERS
        ),
        sum(
            _finite_metric(taxels.get(finger), 0.0)
            for finger in ACTIVE_FINGERS
        ),
    )


def _candidate_id(result: Mapping[str, Any]) -> int:
    value = result.get("candidate_id")
    if isinstance(value, bool):
        raise ValueError("candidate_id must be a non-negative integer")
    try:
        identifier = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("candidate_id must be a non-negative integer") from error
    if identifier < 0 or identifier != value:
        raise ValueError("candidate_id must be a non-negative integer")
    return identifier


def high_thumb_grasp_candidate_rank(
    result: Mapping[str, Any],
) -> tuple[float, ...]:
    """Return a grasp-only rank that ignores manipulation/full-pass claims."""

    identifier = _candidate_id(result)
    metrics = _metrics(result)
    pad_fraction, pad_taxels = _pad_rank(metrics)
    forbidden = _finite_metric(metrics.get("forbidden_contact_steps"), math.inf)
    nondistal = _finite_metric(
        metrics.get("material_active_nondistal_duty"), math.inf
    )
    force = _finite_metric(
        metrics.get("peak_total_distal_contact_force_n"), math.inf
    )
    saturation = _finite_metric(
        metrics.get("actuator_saturation_fraction"), math.inf
    )
    return (
        float(_grasp_success(result)),
        _grasp_margin(metrics),
        *_verify_rank(metrics),
        *_alignment_rank(metrics),
        pad_fraction,
        pad_taxels,
        -forbidden,
        -nondistal,
        -force,
        -saturation,
        -float(identifier),
    )


def rank_high_thumb_grasp_results(
    results: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Return a worker-order-independent total order."""

    materialized = tuple(results)
    identifiers = tuple(_candidate_id(result) for result in materialized)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate_id values must be unique")
    return tuple(
        sorted(
            materialized,
            key=high_thumb_grasp_candidate_rank,
            reverse=True,
        )
    )


def run_high_thumb_grasp_batch(
    candidates: Iterable[Mapping[str, Any]],
    *,
    run_candidates: CandidateRunner,
    workers: int,
    first_candidate_id: int = 0,
    stage: str = "high_thumb_grasp_dynamics",
) -> tuple[dict[str, Any], ...]:
    """Run candidates through an injected runner and verify ID/config binding."""

    worker_count = _positive_integer(workers, "workers")
    first_id = _nonnegative_integer(first_candidate_id, "first_candidate_id")
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")
    materialized = [copy.deepcopy(dict(candidate)) for candidate in candidates]
    for candidate in materialized:
        validate_config(candidate)
    payloads = [
        (first_id + index, copy.deepcopy(candidate))
        for index, candidate in enumerate(materialized)
    ]
    if not payloads:
        return ()
    submitted = {identifier: config for identifier, config in payloads}
    raw_results = run_candidates(payloads, worker_count)
    results = [copy.deepcopy(dict(result)) for result in raw_results]
    identifiers = tuple(_candidate_id(result) for result in results)
    if len(results) != len(payloads) or set(identifiers) != set(submitted):
        raise RuntimeError("runner did not preserve the submitted candidate IDs")
    for result in results:
        identifier = _candidate_id(result)
        if result.get("config") != submitted[identifier]:
            raise RuntimeError("runner rebound a candidate configuration")
        result["search_stage"] = stage
    results.sort(key=lambda result: _candidate_id(result))
    return tuple(results)


__all__ = [
    "CandidateRunner",
    "HighThumbCartesianGrid",
    "HighThumbLocalSearch",
    "REFERENCE_HIGH_THUMB_GRID",
    "THUMB_BEND_ACTUATOR",
    "cube_in_root_from_distance_yz",
    "generate_high_thumb_cartesian_candidates",
    "generate_local_high_thumb_candidates",
    "high_thumb_grasp_candidate_rank",
    "materialize_high_thumb_candidate",
    "rank_high_thumb_grasp_results",
    "run_high_thumb_grasp_batch",
]
