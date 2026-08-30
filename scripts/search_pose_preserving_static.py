#!/usr/bin/env python3
"""Deterministic static screen for schema-v6 pose-preserving grasps.

This utility searches only fixed-root kinematics.  It never writes qpos into a
running acceptance simulation and never classifies a static candidate as a
successful grasp; its JSON output is a compact seed list for the shared
dynamic ``SimulationSession``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


# Running ``python scripts/search_pose_preserving_static.py`` sets sys.path[0]
# to ``scripts/`` rather than the repository root.  Keep the documented direct
# entry point usable without relying on an ambient PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np

from xhand_tactile import TactileReader

from xhand_grasp.config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, load_config
from xhand_grasp.contact_geometry import (
    active_nondistal_collision_geom_ids,
    distal_collision_geom_ids,
    nearest_distal_target_witness,
    nearest_taxel_assignment,
)
from xhand_grasp.contacts import Face
from xhand_grasp.scene import (
    ModelInfo,
    build_model,
    rpy_degrees_to_quaternion,
    rpy_degrees_to_rotation_matrix,
)


FACE_BY_LABEL = {
    "+X": Face.X_POS,
    "-X": Face.X_NEG,
    "+Y": Face.Y_POS,
    "-Y": Face.Y_NEG,
}

LABEL_BY_FACE = {
    Face.X_POS: "+X",
    Face.X_NEG: "-X",
    Face.Y_POS: "+Y",
    Face.Y_NEG: "-Y",
    Face.Z_POS: "+Z",
    Face.Z_NEG: "-Z",
    Face.EDGE_CORNER: "EDGE_CORNER",
    Face.UNKNOWN: "UNKNOWN",
}

DEFAULT_CONFIG = REPO_ROOT / (
    "grasp_configs/left_opposed_face_palm_down_pose_preserving_grasp.json"
)
DEFAULT_SOURCE_CATALOG = REPO_ROOT / (
    "artifacts/left_opposed_face_palm_tilted_down_far_hand_"
    "fingertip_grasp_then_lift/grasp_acquisition_high_thumb/"
    "trajectory_catalog/catalog.json"
)
DEFAULT_SOURCE_TRAJECTORY = "best_grasp"


@dataclass(frozen=True, slots=True)
class SourceTrajectory:
    """Content-addressed acquisition seed resolved from one catalog entry."""

    trajectory_id: str
    requested_trajectory: str
    catalog_path: Path
    catalog_sha256: str
    resolved_config_path: Path
    resolved_config_sha256: str
    trace_path: Path
    trace_sha256: str
    source_sha256: str
    config: dict[str, Any]
    model: mujoco.MjModel
    info: ModelInfo
    acquisition_step: int
    acquisition_qpos_rad: np.ndarray
    acquisition_qpos_by_actuator_rad: dict[str, float]
    acquisition_cube_in_root_m: np.ndarray
    acquisition_cube_world_m: np.ndarray
    acquisition_root_world_m: np.ndarray
    initial_cube_world_m: np.ndarray
    initial_cube_quaternion_wxyz: np.ndarray
    terminal_targets_rad: dict[str, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_artifact_path(catalog_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"catalog {label} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = catalog_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"catalog {label} does not exist: {path}")
    return path


def _catalog_entry(
    catalog: Mapping[str, Any], requested: str
) -> tuple[str, Mapping[str, Any]]:
    aliases = catalog.get("aliases", {})
    if not isinstance(aliases, Mapping):
        raise ValueError("catalog aliases must be a mapping")
    resolved = str(requested)
    seen: set[str] = set()
    while resolved in aliases:
        if resolved in seen:
            raise ValueError("catalog trajectory aliases contain a cycle")
        seen.add(resolved)
        target = aliases[resolved]
        if not isinstance(target, str) or not target:
            raise ValueError(f"catalog alias {resolved!r} has an invalid target")
        resolved = target

    trajectories = catalog.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError("catalog trajectories must be a list")
    matches = []
    for entry in trajectories:
        if not isinstance(entry, Mapping):
            raise ValueError("catalog trajectory entries must be mappings")
        identifiers = {str(entry.get("trajectory_id", ""))}
        entry_aliases = entry.get("aliases", [])
        if isinstance(entry_aliases, list):
            identifiers.update(str(value) for value in entry_aliases)
        if resolved in identifiers or requested in identifiers:
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError(
            f"catalog trajectory {requested!r} resolved to {len(matches)} entries"
        )
    trajectory_id = str(matches[0].get("trajectory_id", ""))
    if not trajectory_id:
        raise ValueError("catalog trajectory is missing trajectory_id")
    return trajectory_id, matches[0]


def _checked_artifact_hash(
    path: Path,
    *,
    entry: Mapping[str, Any],
    artifact_key: str,
) -> str:
    actual = _sha256(path)
    artifacts = entry.get("artifacts", {})
    expected = None
    if isinstance(artifacts, Mapping):
        hashes = artifacts.get("sha256", {})
        if isinstance(hashes, Mapping):
            expected = hashes.get(artifact_key)
    if expected is not None and str(expected) != actual:
        raise ValueError(
            f"catalog {artifact_key} SHA-256 does not match {path}"
        )
    return actual


def _quaternion_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("trace root quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("trace root quaternion must have non-zero length")
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quaternion / norm)
    return rotation.reshape(3, 3)


def load_source_trajectory(
    source_catalog: str | Path,
    trajectory: str = DEFAULT_SOURCE_TRAJECTORY,
) -> SourceTrajectory:
    """Resolve and validate a catalog-backed static-screen source.

    The resolved configuration supplies immutable object physics and terminal
    targets.  The trace supplies the *actual* active-joint qpos and cube pose
    relative to the fixed hand root at the acquisition event.
    """

    catalog_path = Path(source_catalog).resolve()
    if not catalog_path.is_file():
        raise ValueError(f"source catalog does not exist: {catalog_path}")
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read source catalog: {catalog_path}") from error
    if not isinstance(catalog, Mapping):
        raise ValueError("source catalog root must be a mapping")
    trajectory_id, entry = _catalog_entry(catalog, str(trajectory))
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("catalog trajectory is missing artifacts")
    config_path = _resolved_artifact_path(
        catalog_path, artifacts.get("resolved_config"), "resolved_config"
    )
    if isinstance(artifacts.get("grasp_trace"), str) and artifacts.get(
        "grasp_trace"
    ):
        trace_key = "grasp_trace"
    elif isinstance(artifacts.get("trace"), str) and artifacts.get("trace"):
        trace_key = "trace"
    else:
        raise ValueError("catalog trajectory has no grasp_trace or trace artifact")
    trace_path = _resolved_artifact_path(
        catalog_path, artifacts.get(trace_key), trace_key
    )
    catalog_digest = _sha256(catalog_path)
    config_digest = _checked_artifact_hash(
        config_path, entry=entry, artifact_key="resolved_config"
    )
    trace_digest = _checked_artifact_hash(
        trace_path, entry=entry, artifact_key=trace_key
    )
    source_digest = hashlib.sha256(
        (
            trajectory_id
            + "\0"
            + config_digest
            + "\0"
            + trace_digest
        ).encode("utf-8")
    ).hexdigest()

    source_config = load_config(config_path)
    model, info = build_model(source_config)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    initial_cube_world = np.asarray(
        data.xpos[info.cube_body_id], dtype=np.float64
    ).copy()
    initial_cube_quaternion = np.asarray(
        data.xquat[info.cube_body_id], dtype=np.float64
    ).copy()

    try:
        with np.load(trace_path, allow_pickle=False) as archive:
            required = {
                "joint_qpos",
                "cube_pos",
                "root_pos",
                "root_quat",
                "grasp_acquisition_step",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    "source trace is missing fields: " + ", ".join(sorted(missing))
                )
            joint_qpos = np.asarray(archive["joint_qpos"], dtype=np.float64)
            cube_pos = np.asarray(archive["cube_pos"], dtype=np.float64)
            root_pos = np.asarray(archive["root_pos"], dtype=np.float64)
            root_quat = np.asarray(archive["root_quat"], dtype=np.float64)
            acquisition_step = int(
                np.asarray(archive["grasp_acquisition_step"]).reshape(())
            )
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("source trace"):
            raise
        raise ValueError(f"unable to read source trace: {trace_path}") from error

    total_steps = joint_qpos.shape[0] if joint_qpos.ndim == 2 else -1
    if joint_qpos.shape != (total_steps, model.nu):
        raise ValueError(
            f"source joint_qpos must have shape (T, {model.nu})"
        )
    for label, values, width in (
        ("cube_pos", cube_pos, 3),
        ("root_pos", root_pos, 3),
        ("root_quat", root_quat, 4),
    ):
        if values.shape != (total_steps, width):
            raise ValueError(f"source {label} must have shape (T, {width})")
        if not np.isfinite(values).all():
            raise ValueError(f"source {label} must be finite")
    if not np.isfinite(joint_qpos).all():
        raise ValueError("source joint_qpos must be finite")
    if not 0 <= acquisition_step < total_steps:
        raise ValueError("source grasp_acquisition_step is outside the trace")

    active_ids = np.asarray(
        [model.actuator(name).id for name in ACTIVE_ACTUATORS], dtype=np.int64
    )
    acquisition_qpos = joint_qpos[acquisition_step, active_ids].copy()
    acquisition_qpos_by_name = {
        name: float(acquisition_qpos[index])
        for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    rotation = _quaternion_rotation(root_quat[acquisition_step])
    cube_in_root = rotation.T @ (
        cube_pos[acquisition_step] - root_pos[acquisition_step]
    )
    control = source_config.get("control", {})
    terminal_mapping = control.get("grasp_targets_rad", {})
    if not isinstance(terminal_mapping, Mapping) or set(terminal_mapping) != set(
        ACTIVE_ACTUATORS
    ):
        raise ValueError(
            "source control.grasp_targets_rad must contain the active actuators"
        )
    terminal_targets = {
        name: float(terminal_mapping[name]) for name in ACTIVE_ACTUATORS
    }
    if not np.isfinite(list(terminal_targets.values())).all():
        raise ValueError("source terminal grasp targets must be finite")

    return SourceTrajectory(
        trajectory_id=trajectory_id,
        requested_trajectory=str(trajectory),
        catalog_path=catalog_path,
        catalog_sha256=catalog_digest,
        resolved_config_path=config_path,
        resolved_config_sha256=config_digest,
        trace_path=trace_path,
        trace_sha256=trace_digest,
        source_sha256=source_digest,
        config=source_config,
        model=model,
        info=info,
        acquisition_step=acquisition_step,
        acquisition_qpos_rad=acquisition_qpos,
        acquisition_qpos_by_actuator_rad=acquisition_qpos_by_name,
        acquisition_cube_in_root_m=cube_in_root,
        acquisition_cube_world_m=cube_pos[acquisition_step].copy(),
        acquisition_root_world_m=root_pos[acquisition_step].copy(),
        initial_cube_world_m=initial_cube_world,
        initial_cube_quaternion_wxyz=initial_cube_quaternion,
        terminal_targets_rad=terminal_targets,
    )


def _nondistal_gap_record(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
    geom_ids: tuple[int, ...],
) -> dict[str, Any]:
    if not geom_ids:
        raise ValueError("active finger has no non-distal collision geometry")
    best: tuple[float, int, np.ndarray] | None = None
    segment = np.empty(6, dtype=np.float64)
    for geom_id in geom_ids:
        distance = float(
            mujoco.mj_geomDistance(
                model, data, cube_geom_id, int(geom_id), 1.0, segment
            )
        )
        key = (distance, int(geom_id))
        if best is None or key < (best[0], best[1]):
            best = (distance, int(geom_id), segment.copy())
    assert best is not None
    distance, geom_id, closest_segment = best
    return {
        "geom_id": geom_id,
        "geom_name": model.geom(geom_id).name,
        "signed_distance_m": distance,
        "cube_point_world_m": closest_segment[:3].tolist(),
        "nondistal_point_world_m": closest_segment[3:].tolist(),
    }


def _distal_witness_record(
    model: mujoco.MjModel,
    witness: Any,
    *,
    target_face: Face,
    taxel_index: int | None,
    taxel_distance_m: float,
) -> dict[str, Any]:
    classification = witness.classification
    return {
        "distal_geom_id": int(witness.distal_geom_id),
        "distal_geom_name": model.geom(int(witness.distal_geom_id)).name,
        "target_face": LABEL_BY_FACE[target_face],
        "classified_face": LABEL_BY_FACE[witness.face],
        "signed_distance_m": float(witness.signed_distance_m),
        "penetration_m": float(witness.penetration_m),
        "cube_point_world_m": witness.cube_point_world_m.tolist(),
        "distal_point_world_m": witness.distal_point_world_m.tolist(),
        "cube_point_local_m": witness.cube_point_local_m.tolist(),
        "candidate_faces": [
            LABEL_BY_FACE[face] for face in classification.candidate_faces
        ],
        "surface_error_m": float(classification.surface_error_m),
        "edge_clearance_m": float(classification.edge_clearance_m),
        "normal_alignment": float(classification.normal_alignment),
        "nearest_taxel_index": (
            None if taxel_index is None else int(taxel_index)
        ),
        "nearest_taxel_distance_m": float(taxel_distance_m),
    }


def search(
    config_path: str | Path,
    *,
    samples: int,
    seed: int,
    retain: int,
    source_catalog: str | Path = DEFAULT_SOURCE_CATALOG,
    trajectory: str = DEFAULT_SOURCE_TRAJECTORY,
) -> dict[str, Any]:
    """Rank fixed-object acquisition geometry around one catalog source.

    ``config_path`` remains the backward-compatible hand-root pose seed.  The
    selected source resolved config owns the compiled cube pose, physical
    parameters, contact topology and terminal targets.  Every sample uses the
    same acquisition qpos from the source trace; only the fixed hand root pose
    is changed.
    """

    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if not isinstance(retain, int) or isinstance(retain, bool) or retain <= 0:
        raise ValueError("retain must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    source = load_source_trajectory(source_catalog, trajectory)
    model, info = source.model, source.info
    data = mujoco.MjData(model)
    reader = TactileReader(model, data, "left")
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    cube_world = np.asarray(
        data.xpos[info.cube_body_id], dtype=np.float64
    ).copy()
    cube_quaternion = np.asarray(
        data.xquat[info.cube_body_id], dtype=np.float64
    ).copy()
    cube_qpos = np.asarray(
        data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 7],
        dtype=np.float64,
    ).copy()
    if not np.array_equal(cube_world, source.initial_cube_world_m):
        raise RuntimeError("source cube world pose changed while creating search data")
    root_id = info.root_body_id
    root_rpy_seed = np.asarray(config["hand_pose"]["rpy_deg"], dtype=np.float64)
    relative_seed = source.acquisition_cube_in_root_m.copy()
    acquisition_qpos = source.acquisition_qpos_rad.copy()
    active_ids = np.asarray(
        [model.actuator(name).id for name in ACTIVE_ACTUATORS], dtype=np.int64
    )
    active_qpos_addresses = info.actuator_qpos_adrs[active_ids]

    distal_geoms = distal_collision_geom_ids(model, info.distal_weld_ids)
    nondistal_by_finger = active_nondistal_collision_geom_ids(
        model, info.hand_body_parts, info.distal_weld_ids
    )
    target_faces = {
        finger: FACE_BY_LABEL[
            source.config["contact_topology"]["target_faces"][finger]
        ]
        for finger in ACTIVE_FINGERS
    }
    rng = np.random.default_rng(seed)
    ranked: list[tuple[tuple[float, ...], dict]] = []

    for candidate_id in range(samples):
        if candidate_id == 0:
            root_rpy = root_rpy_seed.copy()
            relative = relative_seed.copy()
        else:
            root_rpy = root_rpy_seed + rng.normal(
                0.0, [2.0, 2.5, 2.0], size=3
            )
            relative = relative_seed + rng.normal(
                0.0, [0.003, 0.003, 0.003], size=3
            )

        rotation = rpy_degrees_to_rotation_matrix(root_rpy)
        root_position = cube_world - rotation @ relative
        model.body_pos[root_id] = root_position
        model.body_quat[root_id] = rpy_degrees_to_quaternion(root_rpy)
        data.qpos[active_qpos_addresses] = acquisition_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if not np.array_equal(
            data.qpos[info.cube_qpos_adr : info.cube_qpos_adr + 7], cube_qpos
        ):
            raise RuntimeError("static screen modified the source cube qpos")

        witnesses: list[Any] = []
        missing = 0
        for finger in ACTIVE_FINGERS:
            witness = nearest_distal_target_witness(
                model,
                data,
                cube_geom_id=info.cube_geom_id,
                distal_geom_ids=distal_geoms[finger],
                target_face=target_faces[finger],
                distance_max_m=0.020,
            )
            if witness is None:
                missing += 1
            witnesses.append(witness)
        if missing:
            continue
        assert all(witness is not None for witness in witnesses)
        gaps = np.asarray(
            [witness.signed_distance_m for witness in witnesses],
            dtype=np.float64,
        )
        heights = np.asarray(
            [witness.cube_point_world_m[2] for witness in witnesses],
            dtype=np.float64,
        )
        taxel_assignments = [
            nearest_taxel_assignment(
                witness.distal_point_world_m,
                data.site_xpos[reader.site_ids[index]],
                max_assignment_distance_m=0.006,
            )
            for index, witness in enumerate(witnesses)
        ]
        taxel_distance = np.asarray(
            [assignment.distance_m for assignment in taxel_assignments],
            dtype=np.float64,
        )
        nondistal_records = {
            finger: _nondistal_gap_record(
                model,
                data,
                info.cube_geom_id,
                nondistal_by_finger[finger],
            )
            for finger in ACTIVE_FINGERS
        }
        nondistal_gap = min(
            float(record["signed_distance_m"])
            for record in nondistal_records.values()
        )
        height_spread = float(np.ptp(heights))
        gap_violation = float(
            np.sum(np.maximum(0.0, -0.0005 - gaps))
            + np.sum(np.maximum(0.0, gaps - 0.003))
        )
        taxel_violation = float(np.sum(np.maximum(0.0, taxel_distance - 0.006)))
        nondistal_violation = max(0.0, 0.0005 - nondistal_gap)
        score = (
            1000.0 * gap_violation
            + 1000.0 * max(0.0, height_spread - 0.005)
            + 1000.0 * taxel_violation
            + 1000.0 * nondistal_violation
            + 10.0 * float(np.sum(np.abs(gaps - 0.0015)))
            + height_spread
            + float(np.sum(taxel_distance))
        )
        record = {
            "candidate_id": candidate_id,
            "score": score,
            "source_trajectory_id": source.trajectory_id,
            "source_sha256": source.source_sha256,
            "hand_rpy_deg": root_rpy.tolist(),
            "hand_translation_m": root_position.tolist(),
            "cube_in_root_m": relative.tolist(),
            "fixed_cube_world_position_m": cube_world.tolist(),
            "fixed_cube_world_quaternion_wxyz": cube_quaternion.tolist(),
            "qpos_rad": {
                name: float(acquisition_qpos[index])
                for index, name in enumerate(ACTIVE_ACTUATORS)
            },
            "acquisition_qpos_rad": copy.deepcopy(
                source.acquisition_qpos_by_actuator_rad
            ),
            "terminal_targets_rad": copy.deepcopy(
                source.terminal_targets_rad
            ),
            "target_gap_m": gaps.tolist(),
            "contact_height_m": heights.tolist(),
            "height_spread_m": height_spread,
            "nearest_taxel_distance_m": taxel_distance.tolist(),
            "minimum_active_nondistal_gap_m": nondistal_gap,
            "distal_witness": {
                finger: _distal_witness_record(
                    model,
                    witnesses[index],
                    target_face=target_faces[finger],
                    taxel_index=taxel_assignments[index].taxel_index,
                    taxel_distance_m=taxel_assignments[index].distance_m,
                )
                for index, finger in enumerate(ACTIVE_FINGERS)
            },
            "active_nondistal_gap": nondistal_records,
        }
        key = (
            float(gap_violation > 0.0),
            float(height_spread > 0.005),
            float(taxel_violation > 0.0),
            float(nondistal_violation > 0.0),
            score,
            candidate_id,
        )
        ranked.append((key, record))

    ranked.sort(key=lambda item: item[0])
    selected = [record for _, record in ranked[:retain]]
    return {
        "schema_version": 2,
        "config": str(config_path),
        "seed": seed,
        "sample_count": samples,
        "evaluated_count": len(ranked),
        "retained_count": len(selected),
        "sampled_fields": ["hand_pose"],
        "source_trajectory_id": source.trajectory_id,
        "source_sha256": source.source_sha256,
        "source": {
            "requested_trajectory": source.requested_trajectory,
            "trajectory_id": source.trajectory_id,
            "catalog": str(source.catalog_path),
            "catalog_sha256": source.catalog_sha256,
            "resolved_config": str(source.resolved_config_path),
            "resolved_config_sha256": source.resolved_config_sha256,
            "trace": str(source.trace_path),
            "trace_sha256": source.trace_sha256,
            "source_sha256": source.source_sha256,
            "acquisition_step": source.acquisition_step,
            "acquisition_qpos_rad": copy.deepcopy(
                source.acquisition_qpos_by_actuator_rad
            ),
            "acquisition_cube_in_root_m": (
                source.acquisition_cube_in_root_m.tolist()
            ),
            "acquisition_cube_world_m": (
                source.acquisition_cube_world_m.tolist()
            ),
            "acquisition_root_world_m": (
                source.acquisition_root_world_m.tolist()
            ),
            "fixed_initial_cube_world_m": cube_world.tolist(),
            "fixed_initial_cube_quaternion_wxyz": cube_quaternion.tolist(),
            "fixed_cube": copy.deepcopy(source.config["cube"]),
            "fixed_scene": copy.deepcopy(source.config["scene"]),
            "terminal_targets_rad": copy.deepcopy(
                source.terminal_targets_rad
            ),
        },
        "candidates": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        type=Path,
        help="hand-root pose seed config (object physics come from the source)",
    )
    parser.add_argument(
        "--source-catalog",
        default=DEFAULT_SOURCE_CATALOG,
        type=Path,
        help="catalog containing a resolved acquisition config and trace",
    )
    parser.add_argument(
        "--trajectory",
        default=DEFAULT_SOURCE_TRAJECTORY,
        help="trajectory id or catalog alias (default: best_grasp)",
    )
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--retain", type=int, default=64)
    args = parser.parse_args()
    print(
        json.dumps(
            search(
                args.config,
                samples=args.samples,
                seed=args.seed,
                retain=args.retain,
                source_catalog=args.source_catalog,
                trajectory=args.trajectory,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
