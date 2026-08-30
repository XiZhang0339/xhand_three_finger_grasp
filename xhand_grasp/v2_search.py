"""Deterministic staged search for the palm-down opposed-face experiment."""

from __future__ import annotations

import copy
import heapq
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import mujoco
import numpy as np

from .config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, DISTAL_BODY_NAMES, validate_config
from .contacts import BoxContactThresholds, FACE_ORDER
from .evaluation import face_from_label
from .experiment import (
    DEFAULT_FINAL_TARGET_DELTA_BOUNDS_RAD,
    ExperimentDefinition,
    OpposedFaceAssignment,
    resolve_experiment,
)
from .scene import build_model, rpy_degrees_to_quaternion
from .simulation import contact_snapshot


CandidateRunner = Callable[
    [list[tuple[int, dict[str, Any]]], int], list[dict[str, Any]]
]
CandidateRank = Callable[[dict[str, Any]], tuple[float, ...]]


# The kinematic screen scores the pregrasp pose.  Sampling the terminal pose
# independently can therefore retain an excellent static topology and then
# replace it with an unrelated pose during the lift phase.  These small,
# actuator-specific offsets keep the terminal pose in the same neighbourhood.
# Positive thumb rotation and distal-finger rotation move the corresponding
# pads both inward and upward around the audited palm-down pose; the small
# signed proximal ranges retain enough freedom to trade closure for lift.
FINAL_TARGET_DELTA_BOUNDS_RAD = DEFAULT_FINAL_TARGET_DELTA_BOUNDS_RAD

@dataclass(frozen=True)
class KinematicScreenResult:
    seed: int
    sample_count: int
    retained_count: int
    candidates: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]


def _latin_hypercube(
    samples: int, dimensions: int, rng: np.random.Generator
) -> np.ndarray:
    if samples <= 0 or dimensions <= 0:
        raise ValueError("samples and dimensions must be positive")
    result = np.empty((samples, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        result[:, dimension] = (
            rng.permutation(samples) + rng.random(samples)
        ) / samples
    return result


def _rpy_matrix(rpy_deg: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = np.radians(np.asarray(tuple(rpy_deg), dtype=np.float64))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def palm_down_angle_for_rpy(rpy_deg: Iterable[float]) -> float:
    rotation = _rpy_matrix(rpy_deg)
    cosine = float(rotation[:, 0] @ np.asarray([0.0, 0.0, -1.0]))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _scale(unit: float, bounds: tuple[float, float]) -> float:
    return float(bounds[0] + unit * (bounds[1] - bounds[0]))


def _cube_world_position(config: dict[str, Any]) -> np.ndarray:
    cube = config["cube"]
    return np.asarray(
        [
            float(cube["center_xy_m"][0]),
            float(cube["center_xy_m"][1]),
            float(config["scene"]["support_top_z_m"])
            + float(cube["edge_m"]) / 2.0
            + float(cube.get("z_offset_m", 0.0)),
        ],
        dtype=np.float64,
    )


def _cube_position_in_root(config: dict[str, Any]) -> np.ndarray:
    rotation = _rpy_matrix(config["hand_pose"]["rpy_deg"])
    translation = np.asarray(
        config["hand_pose"]["translation_m"], dtype=np.float64
    )
    return rotation.T @ (_cube_world_position(config) - translation)


def _pose_is_within_search_bounds(
    config: dict[str, Any], definition: ExperimentDefinition
) -> bool:
    bounds = definition.search_bounds
    roll, pitch, yaw = (
        float(value) for value in config["hand_pose"]["rpy_deg"]
    )
    cube_yaw = float(config["cube"]["rpy_deg"][2])
    return (
        bounds.hand_roll_deg[0] <= roll <= bounds.hand_roll_deg[1]
        and bounds.contains_pitch(pitch)
        and bounds.hand_yaw_deg[0] <= yaw <= bounds.hand_yaw_deg[1]
        and bounds.cube_yaw_deg[0] <= cube_yaw <= bounds.cube_yaw_deg[1]
        and bounds.contains_cube_position(_cube_position_in_root(config))
    )


def _final_targets_around_pregrasp(
    pregrasp_targets: dict[str, float],
    unit: np.ndarray,
    *,
    definition: ExperimentDefinition,
) -> dict[str, float]:
    """Generate a terminal pose according to the experiment's declared policy.

    A delta map retains the original v2 neighbourhood policy.  ``None`` means
    the endpoint is sampled independently over the named absolute actuator
    range, as required by the large-cube size campaign.
    """

    values = np.asarray(unit, dtype=np.float64)
    if values.shape != (len(ACTIVE_ACTUATORS),):
        raise ValueError(
            f"terminal offset sample must contain {len(ACTIVE_ACTUATORS)} values"
        )
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(
            "terminal offset sample values must be finite and within [0, 1]"
        )
    if set(pregrasp_targets) != set(ACTIVE_ACTUATORS):
        raise ValueError("pregrasp targets must contain exactly the active actuators")

    final: dict[str, float] = {}
    delta_bounds = definition.search_bounds.final_target_delta_rad
    for index, name in enumerate(ACTIVE_ACTUATORS):
        pregrasp = float(pregrasp_targets[name])
        if not math.isfinite(pregrasp):
            raise ValueError(f"pregrasp target {name!r} must be finite")
        lower, upper = definition.search_bounds.actuator_targets_rad[name]
        if delta_bounds is None:
            final[name] = _scale(values[index], (lower, upper))
        else:
            delta = _scale(values[index], delta_bounds[name])
            final[name] = float(np.clip(pregrasp + delta, lower, upper))
    return final


def _candidate_parameters(
    row: np.ndarray,
    *,
    pitch_deg: float,
    candidate_id: int,
    definition: ExperimentDefinition,
) -> dict[str, Any]:
    bounds = definition.search_bounds
    cursor = 0
    roll = _scale(row[cursor], bounds.hand_roll_deg)
    cursor += 1
    yaw = _scale(row[cursor], bounds.hand_yaw_deg)
    cursor += 1
    cube_in_root = np.asarray(
        [
            _scale(row[cursor + index], bounds.cube_position_in_root_m[axis])
            for index, axis in enumerate(("x", "y", "z"))
        ],
        dtype=np.float64,
    )
    cursor += 3
    cube_yaw = _scale(row[cursor], bounds.cube_yaw_deg)
    cursor += 1
    pregrasp_targets = {
        name: _scale(row[cursor + index], bounds.actuator_targets_rad[name])
        for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    cursor += len(ACTIVE_ACTUATORS)
    final_targets = _final_targets_around_pregrasp(
        pregrasp_targets,
        row[cursor : cursor + len(ACTIVE_ACTUATORS)],
        definition=definition,
    )
    assignment = definition.candidate_faces[candidate_id % len(definition.candidate_faces)]
    return {
        "candidate_id": candidate_id,
        "hand_rpy_deg": [roll, float(pitch_deg), yaw],
        "cube_in_root_m": cube_in_root,
        "cube_yaw_deg": cube_yaw,
        "final_targets_rad": final_targets,
        "pregrasp_targets_rad": pregrasp_targets,
        "target_assignment": assignment,
    }


def _materialize_candidate(
    base: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    candidate = copy.deepcopy(base)
    rpy = [float(value) for value in parameters["hand_rpy_deg"]]
    cube = candidate["cube"]
    cube_world = _cube_world_position(candidate)
    root_translation = cube_world - _rpy_matrix(rpy) @ np.asarray(
        parameters["cube_in_root_m"], dtype=np.float64
    )
    candidate["hand_pose"]["translation_m"] = root_translation.tolist()
    candidate["hand_pose"]["rpy_deg"] = rpy
    candidate["cube"]["rpy_deg"] = [0.0, 0.0, float(parameters["cube_yaw_deg"])]
    final = {
        name: float(parameters["final_targets_rad"][name])
        for name in ACTIVE_ACTUATORS
    }
    candidate["control"]["final_targets_rad"] = final
    candidate["control"]["pregrasp_targets_rad"] = {
        name: float(parameters.get("pregrasp_targets_rad", final)[name])
        for name in ACTIVE_ACTUATORS
    }
    assignment: OpposedFaceAssignment = parameters["target_assignment"]
    candidate["contact_topology"]["target_faces"] = assignment.as_dict()
    validate_config(candidate)
    return candidate


def _static_score(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: Any,
    base: dict[str, Any],
    parameters: dict[str, Any],
    box_thresholds: BoxContactThresholds,
    distal_geom_ids: dict[str, tuple[int, ...]],
    distal_site_ids: dict[str, np.ndarray],
) -> tuple[tuple[float, ...], dict[str, Any]]:
    candidate_id = int(parameters["candidate_id"])
    rpy = parameters["hand_rpy_deg"]
    palm_angle = palm_down_angle_for_rpy(rpy)
    maximum_palm_angle = float(base["acceptance"]["max_palm_down_angle_deg"])
    if palm_angle > maximum_palm_angle + 1e-12:
        return (
            (-1.0, -palm_angle, -float(candidate_id)),
            {"candidate_id": candidate_id, "palm_down_angle_deg": palm_angle},
        )

    cube_config = base["cube"]
    cube_world = np.asarray(
        [
            float(cube_config["center_xy_m"][0]),
            float(cube_config["center_xy_m"][1]),
            float(base["scene"]["support_top_z_m"])
            + float(cube_config["edge_m"]) / 2.0
            + float(cube_config.get("z_offset_m", 0.0)),
        ],
        dtype=np.float64,
    )
    root_rotation = _rpy_matrix(rpy)
    root_translation = cube_world - root_rotation @ np.asarray(
        parameters["cube_in_root_m"], dtype=np.float64
    )

    mujoco.mj_resetData(model, data)
    model.body_pos[info.root_body_id] = root_translation
    model.body_quat[info.root_body_id] = rpy_degrees_to_quaternion(rpy)
    data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 3] = cube_world
    data.qpos[info.cube_qpos_adr + 3 : info.cube_qpos_adr + 7] = (
        rpy_degrees_to_quaternion([0.0, 0.0, parameters["cube_yaw_deg"]])
    )
    for name, value in parameters["pregrasp_targets_rad"].items():
        actuator_id = model.actuator(name).id
        qpos_adr = info.actuator_qpos_adrs[actuator_id]
        data.qpos[qpos_adr] = float(value)
        data.ctrl[actuator_id] = float(value)
    mujoco.mj_forward(model, data)

    snapshot = contact_snapshot(
        model,
        data,
        info,
        classify_faces=True,
        box_thresholds=box_thresholds,
    )
    assert snapshot.distal_face_force_n is not None
    assignment: OpposedFaceAssignment = parameters["target_assignment"]
    labels = assignment.as_dict()
    target_indices = np.asarray(
        [FACE_ORDER.index(face_from_label(labels[finger])) for finger in ACTIVE_FINGERS]
    )
    target_force = snapshot.distal_face_force_n[np.arange(3), target_indices]
    all_force = np.sum(snapshot.distal_face_force_n, axis=1)
    purity = np.divide(
        target_force,
        all_force,
        out=np.zeros_like(target_force),
        where=all_force > 0.0,
    )
    force_threshold = float(base["acceptance"]["contact_force_min_n"])
    purity_threshold = float(base["contact_topology"]["target_force_fraction"])
    clean_contact_count = int(
        np.count_nonzero(
            (target_force >= force_threshold) & (purity >= purity_threshold)
        )
    )
    any_contact_count = int(np.count_nonzero(all_force > 1e-6))

    cube_rotation = data.geom_xmat[info.cube_geom_id].reshape(3, 3)
    cube_position = data.geom_xpos[info.cube_geom_id]
    half_extent = model.geom_size[info.cube_geom_id]
    geometric_distance = 0.0
    for finger, target_label in labels.items():
        body_id = model.body(DISTAL_BODY_NAMES[finger]).id
        local = cube_rotation.T @ (data.xpos[body_id] - cube_position)
        face = face_from_label(target_label)
        plane_distance = abs(local[face.axis] - face.sign * half_extent[face.axis])
        tangential_axes = [axis for axis in range(3) if axis != face.axis]
        outside = np.maximum(
            np.abs(local[tangential_axes]) - half_extent[tangential_axes], 0.0
        )
        geometric_distance += float(plane_distance + np.linalg.norm(outside))

    target_geom_distance: list[float] = []
    for finger, target_label in labels.items():
        face = face_from_label(target_label)
        tangential_axes = [axis for axis in range(3) if axis != face.axis]
        best_distance = math.inf
        for geom_id in distal_geom_ids[finger]:
            from_to = np.zeros(6, dtype=np.float64)
            distance = float(
                mujoco.mj_geomDistance(
                    model,
                    data,
                    info.cube_geom_id,
                    geom_id,
                    1.0,
                    from_to,
                )
            )
            cube_point_local = cube_rotation.T @ (from_to[:3] - cube_position)
            surface_error = abs(
                cube_point_local[face.axis] - face.sign * half_extent[face.axis]
            )
            edge_clearance = float(
                np.min(
                    half_extent[tangential_axes]
                    - np.abs(cube_point_local[tangential_axes])
                )
            )
            if (
                surface_error <= box_thresholds.surface_tolerance_m + 1e-9
                and edge_clearance + 1e-12 >= box_thresholds.edge_margin_m
            ):
                best_distance = min(best_distance, distance)
        target_geom_distance.append(best_distance)
    target_site_signed_distance: list[float] = []
    for finger, target_label in labels.items():
        face = face_from_label(target_label)
        tangential_axes = [axis for axis in range(3) if axis != face.axis]
        site_local = (
            data.site_xpos[distal_site_ids[finger]] - cube_position
        ) @ cube_rotation
        inside = np.all(
            np.abs(site_local[:, tangential_axes])
            <= half_extent[tangential_axes] - box_thresholds.edge_margin_m,
            axis=1,
        )
        if np.any(inside):
            signed = face.sign * (
                site_local[inside, face.axis] - face.sign * half_extent[face.axis]
            )
            best_site_distance = float(signed[np.argmin(np.abs(signed))])
        else:
            best_site_distance = math.inf
        target_site_signed_distance.append(best_site_distance)
    near_target_count = sum(
        -float(base["acceptance"]["max_penetration_m"]) <= distance <= 0.003
        for distance in target_site_signed_distance
    )
    finite_target_distance = sum(
        abs(distance) if math.isfinite(distance) else 1.0
        for distance in target_site_signed_distance
    )

    score = (
        1.0,
        float(not snapshot.forbidden),
        float(
            snapshot.max_penetration
            <= float(base["acceptance"]["max_penetration_m"]) + 1e-12
        ),
        float(clean_contact_count),
        float(near_target_count),
        float(any_contact_count),
        -finite_target_distance,
        -float(np.sum(target_force)),
        -geometric_distance,
        -snapshot.max_penetration,
        -float(candidate_id),
    )
    diagnostic = {
        "candidate_id": candidate_id,
        "score": score[:-1],
        "palm_down_angle_deg": palm_angle,
        "target_faces": labels,
        "clean_target_contact_count": clean_contact_count,
        "near_target_face_count": near_target_count,
        "target_geom_signed_distance_m": target_geom_distance,
        "target_site_signed_distance_m": target_site_signed_distance,
        "any_distal_contact_count": any_contact_count,
        "target_force_n": target_force.tolist(),
        "target_force_purity": purity.tolist(),
        "distal_origin_face_distance_m": geometric_distance,
        "forbidden_contact": snapshot.forbidden,
        "max_penetration_m": snapshot.max_penetration,
    }
    return score, diagnostic


def kinematic_screen(
    base: dict[str, Any],
    *,
    samples_per_pitch: int | None = None,
    retain: int | None = None,
    seed: int | None = None,
    definition: ExperimentDefinition | None = None,
) -> KinematicScreenResult:
    """Run a cheap static screen and retain only deterministic top candidates."""

    definition = resolve_experiment(base) if definition is None else definition
    if not definition.candidate_faces:
        raise ValueError("the selected experiment does not define candidate faces")
    bounds = definition.search_bounds
    samples_per_pitch = (
        bounds.kinematic_samples_per_pitch
        if samples_per_pitch is None
        else int(samples_per_pitch)
    )
    retain = bounds.dynamic_candidate_count if retain is None else int(retain)
    seed = bounds.seed if seed is None else int(seed)
    if samples_per_pitch <= 0 or retain <= 0:
        raise ValueError("kinematic sample and retention counts must be positive")

    model, info = build_model(base)
    data = mujoco.MjData(model)
    topology = base["contact_topology"]
    box_thresholds = BoxContactThresholds(
        surface_tolerance_m=float(topology["surface_tolerance_m"]),
        edge_margin_m=float(topology["edge_margin_m"]),
        normal_alignment_min=float(topology["min_normal_alignment"]),
    )
    distal_geom_ids = {
        finger: tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.body_weldid[int(model.geom_bodyid[geom_id])])
            == info.distal_weld_ids[finger]
            and (
                int(model.geom_contype[geom_id]) != 0
                or int(model.geom_conaffinity[geom_id]) != 0
            )
        )
        for finger in ACTIVE_FINGERS
    }
    distal_site_ids = {
        finger: np.asarray(
            [
                site_id
                for site_id in range(model.nsite)
                if int(model.body_weldid[int(model.site_bodyid[site_id])])
                == info.distal_weld_ids[finger]
            ],
            dtype=int,
        )
        for finger in ACTIVE_FINGERS
    }
    if any(site_ids.size == 0 for site_ids in distal_site_ids.values()):
        raise ValueError("active distal tactile site mapping is incomplete")
    entry_type = tuple[tuple[float, ...], int, dict[str, Any], dict[str, Any]]
    assignment_capacities = {
        assignment: retain // len(definition.candidate_faces)
        + (index < retain % len(definition.candidate_faces))
        for index, assignment in enumerate(definition.candidate_faces)
    }
    heaps: dict[OpposedFaceAssignment, list[entry_type]] = {
        assignment: [] for assignment in definition.candidate_faces
    }
    seed_parameters = {
        "candidate_id": -1,
        "hand_rpy_deg": [float(value) for value in base["hand_pose"]["rpy_deg"]],
        "cube_in_root_m": np.asarray(
            base.get("initial_near_miss", {}).get(
                "cube_position_in_root_m", _cube_position_in_root(base)
            ),
            dtype=np.float64,
        ),
        "cube_yaw_deg": float(base["cube"].get("rpy_deg", [0.0, 0.0, 0.0])[2]),
        "final_targets_rad": {
            name: float(base["control"]["final_targets_rad"][name])
            for name in ACTIVE_ACTUATORS
        },
        # The audited static seed is the declared final pose.  Use it as the
        # approach endpoint too, so the seed is tested as a reachable grasp
        # instead of only as a teleport-only geometry.
        "pregrasp_targets_rad": {
            name: float(base["control"]["final_targets_rad"][name])
            for name in ACTIVE_ACTUATORS
        },
        "target_assignment": OpposedFaceAssignment.from_mapping(
            base["contact_topology"]["target_faces"]
        ),
    }
    seed_score, seed_diagnostic = _static_score(
        model,
        data,
        info,
        base,
        seed_parameters,
        box_thresholds,
        distal_geom_ids,
        distal_site_ids,
    )
    seed_entry = (seed_score, 1, seed_parameters, seed_diagnostic)
    seed_assignment: OpposedFaceAssignment = seed_parameters["target_assignment"]
    sampled_capacities = dict(assignment_capacities)
    sampled_capacities[seed_assignment] = max(
        0, sampled_capacities[seed_assignment] - 1
    )
    dimensions = 6 + 2 * len(ACTIVE_ACTUATORS)
    total_samples = 0
    for pitch_index, pitch in enumerate(bounds.palm_pitch_values_deg):
        rng = np.random.default_rng(seed + pitch_index * 1_000_003)
        matrix = _latin_hypercube(samples_per_pitch, dimensions, rng)
        for row_index, row in enumerate(matrix):
            candidate_id = pitch_index * samples_per_pitch + row_index
            parameters = _candidate_parameters(
                row,
                pitch_deg=pitch,
                candidate_id=candidate_id,
                definition=definition,
            )
            score, diagnostic = _static_score(
                model,
                data,
                info,
                base,
                parameters,
                box_thresholds,
                distal_geom_ids,
                distal_site_ids,
            )
            entry = (score, -candidate_id, parameters, diagnostic)
            assignment: OpposedFaceAssignment = parameters["target_assignment"]
            heap = heaps[assignment]
            capacity = sampled_capacities[assignment]
            if len(heap) < capacity:
                heapq.heappush(heap, entry)
            elif capacity and entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
            total_samples += 1

    ranked = sorted(
        [entry for heap in heaps.values() for entry in heap] + [seed_entry],
        key=lambda entry: (entry[0], entry[1]),
        reverse=True,
    )
    candidates = tuple(_materialize_candidate(base, entry[2]) for entry in ranked)
    diagnostics = tuple(entry[3] for entry in ranked)
    return KinematicScreenResult(
        seed=seed,
        sample_count=total_samples,
        retained_count=len(candidates),
        candidates=candidates,
        diagnostics=diagnostics,
    )


def _local_candidates(
    parent: dict[str, Any],
    *,
    count: int,
    seed: int,
    definition: ExperimentDefinition,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    result: list[dict[str, Any]] = []
    bounds = definition.search_bounds
    parent_rpy = np.asarray(parent["hand_pose"]["rpy_deg"], dtype=np.float64)
    parent_cube_in_root = _cube_position_in_root(parent)
    for _ in range(count):
        candidate = copy.deepcopy(parent)
        candidate_rpy = parent_rpy + rng.uniform(-2.0, 2.0, 3)
        candidate_rpy = np.clip(
            candidate_rpy,
            np.asarray(
                [
                    bounds.hand_roll_deg[0],
                    bounds.palm_pitch_deg[0],
                    bounds.hand_yaw_deg[0],
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    bounds.hand_roll_deg[1],
                    bounds.palm_pitch_deg[1],
                    bounds.hand_yaw_deg[1],
                ],
                dtype=np.float64,
            ),
        )
        candidate["hand_pose"]["rpy_deg"] = candidate_rpy.tolist()
        if palm_down_angle_for_rpy(candidate["hand_pose"]["rpy_deg"]) > float(
            candidate["acceptance"]["max_palm_down_angle_deg"]
        ) + 1e-12:
            raise RuntimeError("declared palm search bounds violate palm-down acceptance")

        cube_in_root = parent_cube_in_root + rng.uniform(-0.0025, 0.0025, 3)
        cube_in_root = np.clip(
            cube_in_root,
            np.asarray(
                [bounds.cube_position_in_root_m[axis][0] for axis in ("x", "y", "z")],
                dtype=np.float64,
            ),
            np.asarray(
                [bounds.cube_position_in_root_m[axis][1] for axis in ("x", "y", "z")],
                dtype=np.float64,
            ),
        )
        candidate["cube"]["center_xy_m"] = (
            np.asarray(parent["cube"]["center_xy_m"], dtype=np.float64)
            + rng.uniform(-0.001, 0.001, 2)
        ).tolist()
        cube_yaw = float(
            np.clip(
                parent["cube"]["rpy_deg"][2] + rng.uniform(-2.0, 2.0),
                bounds.cube_yaw_deg[0],
                bounds.cube_yaw_deg[1],
            )
        )
        candidate["cube"]["rpy_deg"][2] = cube_yaw
        candidate["hand_pose"]["translation_m"] = (
            _cube_world_position(candidate)
            - _rpy_matrix(candidate_rpy) @ cube_in_root
        ).tolist()
        # Refine the contact-producing pregrasp first, then refine only the
        # small terminal offset.  Perturbing both absolute poses independently
        # recreated the same topology-destroying jump as the global sampler.
        for name in ACTIVE_ACTUATORS:
            lower, upper = bounds.actuator_targets_rad[name]
            target_span = upper - lower
            parent_pregrasp = float(
                parent["control"]["pregrasp_targets_rad"][name]
            )
            pregrasp = float(
                np.clip(
                    parent_pregrasp + rng.normal(0.0, 0.02 * target_span),
                    lower,
                    upper,
                )
            )

            delta_bounds = bounds.final_target_delta_rad
            if delta_bounds is None:
                parent_final = float(
                    parent["control"]["final_targets_rad"][name]
                )
                final = float(
                    np.clip(
                        parent_final + rng.normal(0.0, 0.02 * target_span),
                        lower,
                        upper,
                    )
                )
            else:
                delta_lower, delta_upper = delta_bounds[name]
                parent_delta = float(
                    parent["control"]["final_targets_rad"][name]
                ) - parent_pregrasp
                parent_delta = float(
                    np.clip(parent_delta, delta_lower, delta_upper)
                )
                delta_span = delta_upper - delta_lower
                delta = float(
                    np.clip(
                        parent_delta + rng.normal(0.0, 0.08 * delta_span),
                        delta_lower,
                        delta_upper,
                    )
                )
                final = float(np.clip(pregrasp + delta, lower, upper))
            candidate["control"]["pregrasp_targets_rad"][name] = pregrasp
            candidate["control"]["final_targets_rad"][name] = final
        if not _pose_is_within_search_bounds(candidate, definition):
            raise RuntimeError("clipped local candidate escaped declared search bounds")
        validate_config(candidate)
        result.append(candidate)
    return result


def _physics_fallback_candidates(
    parents: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    ensure_non_nominal_material: bool = False,
) -> list[dict[str, Any]]:
    """Sample mass/friction without changing size-specific screened geometry."""

    if count < 0:
        raise ValueError("fallback candidate count must be non-negative")
    if count and not parents:
        raise ValueError("fallback material sampling requires a geometry parent")
    if not count:
        return []
    rng = np.random.default_rng(seed)
    # Preserve one exact nominal-material run for the best geometry, then cover
    # the declared fallback material box.  The nominal-size bucket separately
    # guarantees at least one non-nominal material case.
    matrix = (
        _latin_hypercube(count - 1, 2, rng)
        if count > 1
        else np.empty((0, 2), dtype=np.float64)
    )
    cases: list[dict[str, Any]] = []
    for index in range(count):
        candidate = copy.deepcopy(parents[index % len(parents)])
        if index:
            row = matrix[index - 1]
            candidate["cube"]["mass_kg"] = 0.010 + float(row[0]) * 0.020
            candidate["cube"]["friction"] = 0.4 + float(row[1]) * 0.8
        validate_config(candidate)
        cases.append(candidate)
    if ensure_non_nominal_material:
        nominal_mass = float(parents[0]["cube"]["mass_kg"])
        nominal_friction = float(parents[0]["cube"]["friction"])
        cases[-1]["cube"]["mass_kg"] = (
            0.010
            if not math.isclose(nominal_mass, 0.010, rel_tol=0.0, abs_tol=1e-12)
            else 0.030
        )
        cases[-1]["cube"]["friction"] = (
            0.4
            if not math.isclose(
                nominal_friction, 0.4, rel_tol=0.0, abs_tol=1e-12
            )
            else 1.2
        )
        validate_config(cases[-1])
    return cases


def _has_nominal_cube_physics(
    candidate: dict[str, Any], nominal: dict[str, Any]
) -> bool:
    return all(
        math.isclose(
            float(candidate["cube"][field]),
            float(nominal["cube"][field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for field in ("edge_m", "mass_kg", "friction")
    )


def _even_capacities(total: int, bucket_count: int) -> tuple[int, ...]:
    if total < 0:
        raise ValueError("total candidate count must be non-negative")
    if bucket_count <= 0:
        raise ValueError("bucket count must be positive")
    quotient, remainder = divmod(total, bucket_count)
    return tuple(
        quotient + int(index < remainder) for index in range(bucket_count)
    )


def _diverse_top_candidates(
    ranked: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Round-robin face assignments while preserving rank inside each bucket."""

    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in ranked:
        target = item["config"]["contact_topology"]["target_faces"]
        key = (str(target["thumb"]), str(target["index"]), str(target["mid"]))
        buckets.setdefault(key, []).append(item)
    selected: list[dict[str, Any]] = []
    depth = 0
    ordered_keys = sorted(buckets)
    while len(selected) < count:
        added = False
        for key in ordered_keys:
            if depth < len(buckets[key]):
                selected.append(buckets[key][depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    return selected


def tune_opposed_face(
    config: dict[str, Any],
    *,
    workers: int,
    seed: int,
    run_candidates: CandidateRunner,
    rank_candidate: CandidateRank,
    kinematic_samples_per_pitch: int | None = None,
    dynamic_candidate_count: int | None = None,
    local_refine_seed_count: int | None = None,
    local_refine_per_seed: int | None = None,
    final_candidate_count: int | None = None,
    perturbations_per_final: int | None = None,
    fallback_physics_count: int | None = None,
    fallback_kinematic_samples_per_pitch: int | None = None,
    perturb_cases: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Execute the declared nominal, refinement, and fallback stages."""

    definition = resolve_experiment(config)
    bounds = definition.search_bounds
    dynamic_count = dynamic_candidate_count or bounds.dynamic_candidate_count
    refine_seed_count = local_refine_seed_count or bounds.local_refine_seed_count
    refine_per_seed = local_refine_per_seed or bounds.local_refine_per_seed
    final_count = final_candidate_count or bounds.final_candidate_count
    perturbation_count = (
        perturbations_per_final
        or bounds.perturbations_per_final_candidate
    )
    fallback_count = (
        bounds.fallback_candidate_count
        if fallback_physics_count is None
        else int(fallback_physics_count)
    )
    if fallback_count < 0:
        raise ValueError("fallback_physics_count must be non-negative")
    fallback_screen_samples = int(
        bounds.fallback_kinematic_samples_per_pitch
        if fallback_kinematic_samples_per_pitch is None
        else fallback_kinematic_samples_per_pitch
    )
    if fallback_screen_samples <= 0:
        raise ValueError(
            "fallback_kinematic_samples_per_pitch must be positive"
        )
    screen = kinematic_screen(
        config,
        samples_per_pitch=kinematic_samples_per_pitch,
        retain=dynamic_count,
        seed=seed,
    )
    initial_payload = list(enumerate(screen.candidates))
    initial_results = run_candidates(initial_payload, workers)
    for result in initial_results:
        result["search_stage"] = "nominal_geometry"
        result["nominal_physics"] = _has_nominal_cube_physics(
            result["config"], config
        )
    all_nominal = list(initial_results)
    next_id = len(initial_payload)

    ranked_initial = sorted(initial_results, key=rank_candidate, reverse=True)
    local_payload: list[tuple[int, dict[str, Any]]] = []
    refinement_parents = _diverse_top_candidates(
        ranked_initial, refine_seed_count
    )
    for parent_rank, parent in enumerate(refinement_parents):
        local = _local_candidates(
            parent["config"],
            count=refine_per_seed,
            seed=seed + 10_000 + parent_rank,
            definition=definition,
        )
        for candidate in local:
            local_payload.append((next_id, candidate))
            next_id += 1
    local_results = run_candidates(local_payload, workers)
    for result in local_results:
        result["search_stage"] = "nominal_local_refinement"
        result["nominal_physics"] = _has_nominal_cube_physics(
            result["config"], config
        )
    all_nominal.extend(local_results)

    nominal_passes = [
        result
        for result in all_nominal
        if result["summary"]["passed"] and result["nominal_physics"]
    ]
    fallback_results: list[dict[str, Any]] = []
    fallback_geometry_screens: list[dict[str, Any]] = []
    if not nominal_passes and fallback_count:
        fallback_edges = tuple(float(edge) for edge in definition.robustness.edge_m)
        capacities = _even_capacities(fallback_count, len(fallback_edges))
        fallback_cases: list[dict[str, Any]] = []
        for edge_index, (edge_m, capacity) in enumerate(
            zip(fallback_edges, capacities)
        ):
            alternative_base = copy.deepcopy(config)
            alternative_base["cube"]["edge_m"] = edge_m
            validate_config(alternative_base)
            screen_seed = seed + 20_000 + edge_index * 1_000_003
            alternative_screen = kinematic_screen(
                alternative_base,
                samples_per_pitch=fallback_screen_samples,
                retain=max(1, capacity),
                seed=screen_seed,
            )
            geometry_parents = list(alternative_screen.candidates)
            size_cases = _physics_fallback_candidates(
                geometry_parents,
                count=capacity,
                seed=seed + 40_000 + edge_index * 1_000_003,
                ensure_non_nominal_material=math.isclose(
                    edge_m,
                    float(config["cube"]["edge_m"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
            )
            fallback_cases.extend(size_cases)
            fallback_geometry_screens.append(
                {
                    "edge_m": edge_m,
                    "seed": alternative_screen.seed,
                    "samples_per_pitch": fallback_screen_samples,
                    "sample_count": alternative_screen.sample_count,
                    "retained_count": alternative_screen.retained_count,
                    "dynamic_candidate_count": len(size_cases),
                    "top_diagnostics": list(
                        alternative_screen.diagnostics[:5]
                    ),
                }
            )
        fallback_payload = []
        for candidate in fallback_cases:
            fallback_payload.append((next_id, candidate))
            next_id += 1
        fallback_results = run_candidates(fallback_payload, workers)
        for result in fallback_results:
            result["search_stage"] = "alternative_physics"
            result["nominal_physics"] = _has_nominal_cube_physics(
                result["config"], config
            )

    all_results = all_nominal + fallback_results
    ranked = sorted(all_results, key=rank_candidate, reverse=True)
    hard_candidates = [item for item in ranked if item["summary"]["passed"]]
    probe_parents = (
        hard_candidates[:final_count]
        if hard_candidates
        else _diverse_top_candidates(ranked, final_count)
    )
    probe_results: list[dict[str, Any]] = []
    if perturb_cases is not None:
        for parent_rank, parent in enumerate(probe_parents):
            cases = perturb_cases(
                parent["config"],
                count=perturbation_count,
                seed=seed + 30_000 + parent_rank,
            )
            payload = []
            for case in cases:
                payload.append((next_id, case))
                next_id += 1
            trials = run_candidates(payload, workers)
            probe_results.append(
                {
                    "candidate_id": int(parent["candidate_id"]),
                    "passes": sum(trial["summary"]["passed"] for trial in trials),
                    "trial_count": len(trials),
                    "trials": trials,
                }
            )
    probe_lookup = {record["candidate_id"]: record for record in probe_results}

    def selection_key(item: dict[str, Any]) -> tuple[float, ...]:
        probe = probe_lookup.get(int(item["candidate_id"]), {"passes": -1})
        metrics = item["summary"]["metrics"]
        return (
            float(item["summary"]["passed"]),
            float(
                item["summary"]["passed"]
                and item.get("nominal_physics", False)
            ),
            float(probe["passes"]),
            *rank_candidate(item),
            -float(metrics.get("max_palm_down_angle_deg", math.inf)),
            -int(item["candidate_id"]),
        )

    best = copy.deepcopy(max(ranked, key=selection_key))
    best["local_perturbation_probe"] = probe_lookup.get(
        int(best["candidate_id"]),
        {
            "candidate_id": int(best["candidate_id"]),
            "passes": 0,
            "trial_count": 0,
            "trials": [],
        },
    )
    nominal_success = any(
        item["summary"]["passed"] and item.get("nominal_physics", False)
        for item in all_results
    )
    alternative_success = any(
        item["summary"]["passed"] and not item.get("nominal_physics", False)
        for item in fallback_results
    )
    best_is_nominal = bool(best.get("nominal_physics", False))
    best["config"]["experiment_status"] = {
        "classification": (
            "validated_nominal"
            if best["summary"]["passed"] and best_is_nominal
            else "validated_alternative_physics"
            if best["summary"]["passed"]
            else "best_near_miss"
        ),
        "passed": bool(best["summary"]["passed"]),
        "nominal_physics_passed": nominal_success,
        "note": (
            "All declared hard constraints passed."
            if best["summary"]["passed"]
            else "No candidate passed all declared hard constraints within the search budget."
        ),
    }
    return {
        "experiment_id": definition.experiment_id,
        "seed": seed,
        "workers": workers,
        "kinematic_sample_count": screen.sample_count,
        "kinematic_retained_count": screen.retained_count,
        "kinematic_top_diagnostics": list(screen.diagnostics[:20]),
        "initial_dynamic_count": len(initial_results),
        "local_refinement_count": len(local_results),
        "fallback_physics_count": len(fallback_results),
        "fallback_kinematic_samples_per_pitch": fallback_screen_samples,
        "alternative_geometry_screens": fallback_geometry_screens,
        "candidate_count": len(all_results),
        "perturbation_probe_count": sum(
            record["trial_count"] for record in probe_results
        ),
        "simulation_count": len(all_results)
        + sum(record["trial_count"] for record in probe_results),
        "passing_candidates": len(hard_candidates),
        "nominal_success": nominal_success,
        "alternative_physics_success": alternative_success,
        "best": best,
        "top_candidates": ranked[:20],
        "local_perturbation_probes": probe_results,
    }
