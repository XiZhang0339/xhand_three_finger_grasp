"""Deterministic hand-pose seeds derived from validated schema-v5 grasps.

The high-thumb grasp catalog contains six independently rerun acquisitions.
Those runs are useful geometric evidence, but their cubes moved before the
grasp latch.  This module extracts the *measured* terminal hand/cube relation
and transfers it back to the source cube's unchanged initial world pose.

Only the hand root pose is sampled.  The source cube block and terminal grasp
targets are copied byte-for-byte (at the Python value level), manipulation is
kept at zero, and the pregrasp command/close profile come from the registered
schema-v6 template.  The functions that transform already-loaded mappings and
arrays are pure; filesystem loading and SHA-256 verification are confined to
``load_pose_preserving_seed_sources``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..config import (
    ACTIVE_ACTUATORS,
    load_config,
    resolved_pose_constraint_values,
    validate_config,
)
from ..experiment import resolve_experiment
from ..scene import (
    cube_vertical_half_extent_m,
    rpy_degrees_to_quaternion,
    rpy_degrees_to_rotation_matrix,
)


CAMPAIGN_SCHEMA_VERSION = 1
EXPECTED_SOURCE_COUNT = 6
RIGID_RETARGET_METHOD = "align_acquired_cube_to_configured_initial_pose_v1"
RETARGET_METHOD = "template_rpy_cube_yaw_compensated_terminal_position_v1"
SOURCE_EXPERIMENT_ID = (
    "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift"
)
TARGET_EXPERIMENT_ID = "left_opposed_face_palm_down_pose_preserving_grasp"
DEFAULT_HAND_RPY_RADIUS_DEG = (0.75, 1.50, 0.75)
DEFAULT_CUBE_IN_ROOT_RADIUS_M = (0.0015, 0.0015, 0.0015)
_CANDIDATE_ID_STRIDE = 1_000_000

_TRACE_FIELDS = (
    "time",
    "actuator_order",
    "joint_qpos",
    "cube_pos",
    "cube_quat",
    "root_pos",
    "root_quat",
    "control_state",
    "grasp_acquired",
    "grasp_acquisition_step",
)


def file_sha256(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest without interpreting the file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using one canonical serialization."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_vector(
    values: Any,
    length: int,
    label: str,
) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain {length} finite values") from error
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return vector.copy()


def _unit_quaternion(values: Any, label: str) -> np.ndarray:
    quaternion = _finite_vector(values, 4, label)
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{label} must be non-zero")
    quaternion /= norm
    # q and -q encode the same rotation.  A canonical sign makes provenance
    # independent of an otherwise harmless sign flip in a recorded trace.
    nonzero = np.flatnonzero(np.abs(quaternion) > 1e-15)
    if nonzero.size and quaternion[int(nonzero[0])] < 0.0:
        quaternion *= -1.0
    return quaternion


def quaternion_to_rotation_matrix(values: Any) -> np.ndarray:
    """Return a world-from-local rotation matrix for a WXYZ quaternion."""

    w, x, y, z = _unit_quaternion(values, "quaternion")
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(values: Any) -> np.ndarray:
    """Return a canonical WXYZ quaternion for a proper rotation matrix."""

    rotation = np.asarray(values, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation matrix must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0):
        raise ValueError("rotation matrix must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-10):
        raise ValueError("rotation matrix must have determinant one")

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return _unit_quaternion(quaternion, "derived quaternion")


def _wrap_near(value_deg: float, reference_deg: float) -> float:
    return float(reference_deg + (value_deg - reference_deg + 180.0) % 360.0 - 180.0)


def rotation_matrix_to_rpy_degrees_near(
    values: Any,
    reference_rpy_deg: Sequence[float],
) -> np.ndarray:
    """Resolve the XYZ Euler branch nearest a supplied RPY reference.

    The hand templates deliberately use pitch angles above 90 degrees.  A
    conventional inverse returns the equivalent ``roll~=180, pitch<90,
    yaw~=180`` branch, which falls outside the registered search envelope.
    Choosing between the two exact branches near the template avoids that
    representation-only discontinuity.
    """

    rotation = np.asarray(values, dtype=np.float64)
    reference = _finite_vector(reference_rpy_deg, 3, "reference_rpy_deg")
    # Validate the matrix and reuse the quaternion conversion's strict checks.
    rotation_matrix_to_quaternion(rotation)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-10:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        yaw = 0.0
    principal = np.degrees([roll, pitch, yaw])
    alternate = np.asarray(
        [principal[0] + 180.0, 180.0 - principal[1], principal[2] + 180.0],
        dtype=np.float64,
    )
    branches = tuple(
        np.asarray(
            [_wrap_near(branch[index], reference[index]) for index in range(3)],
            dtype=np.float64,
        )
        for branch in (principal, alternate)
    )
    selected = min(
        branches,
        key=lambda branch: float(np.sum(np.square(branch - reference))),
    )
    if not np.allclose(
        rpy_degrees_to_rotation_matrix(selected),
        rotation,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("Euler branch selection changed the rotation")
    return selected


def retarget_root_pose(
    *,
    configured_cube_pose: Mapping[str, Any],
    acquired_cube_pose: Mapping[str, Any],
    source_root_pose: Mapping[str, Any],
    reference_rpy_deg: Sequence[float],
) -> dict[str, list[float]]:
    """Move the hand with the acquired cube back to its configured cube pose.

    This applies ``T_WR_seed = T_WC0 @ inv(T_WC_acq) @ T_WR_src``.  The
    object's configured world pose is therefore unchanged and only the hand
    root is transformed.  The reference selects the deterministic Euler
    branch used by schema-v6, whose pitch is intentionally above 90 degrees.
    """

    cube_zero_position = _finite_vector(
        configured_cube_pose["position_m"], 3, "configured cube position"
    )
    cube_acquired_position = _finite_vector(
        acquired_cube_pose["position_m"], 3, "acquired cube position"
    )
    root_source_position = _finite_vector(
        source_root_pose["position_m"], 3, "source root position"
    )
    cube_zero_rotation = quaternion_to_rotation_matrix(
        configured_cube_pose["quaternion_wxyz"]
    )
    cube_acquired_rotation = quaternion_to_rotation_matrix(
        acquired_cube_pose["quaternion_wxyz"]
    )
    root_source_rotation = quaternion_to_rotation_matrix(
        source_root_pose["quaternion_wxyz"]
    )
    alignment = cube_zero_rotation @ cube_acquired_rotation.T
    root_seed_rotation = alignment @ root_source_rotation
    root_seed_position = cube_zero_position + alignment @ (
        root_source_position - cube_acquired_position
    )
    return {
        "translation_m": root_seed_position.tolist(),
        "rpy_deg": rotation_matrix_to_rpy_degrees_near(
            root_seed_rotation, reference_rpy_deg
        ).tolist(),
        "quaternion_wxyz": rotation_matrix_to_quaternion(
            root_seed_rotation
        ).tolist(),
    }


def initial_cube_world_pose(config: Mapping[str, Any]) -> dict[str, list[float]]:
    """Resolve the source cube pose before its first integration step."""

    cube = config["cube"]
    scene = config["scene"]
    rpy = _finite_vector(cube.get("rpy_deg", (0.0, 0.0, 0.0)), 3, "cube.rpy_deg")
    rotation = rpy_degrees_to_rotation_matrix(rpy)
    vertical_extent = cube_vertical_half_extent_m(float(cube["edge_m"]), rotation)
    center_xy = _finite_vector(cube["center_xy_m"], 2, "cube.center_xy_m")
    position = np.asarray(
        [
            center_xy[0],
            center_xy[1],
            float(scene["support_top_z_m"])
            + vertical_extent
            + float(cube.get("z_offset_m", 0.0)),
        ],
        dtype=np.float64,
    )
    return {
        "position_m": position.tolist(),
        "quaternion_wxyz": rpy_degrees_to_quaternion(rpy).tolist(),
    }


def _target_mapping(values: Mapping[str, Any], label: str) -> dict[str, float]:
    if set(values) != set(ACTIVE_ACTUATORS):
        raise ValueError(f"{label} must contain exactly the active actuators")
    result = {name: float(values[name]) for name in ACTIVE_ACTUATORS}
    if not np.isfinite(tuple(result.values())).all():
        raise ValueError(f"{label} must contain only finite values")
    return result


def extract_pose_preserving_source(
    entry: Mapping[str, Any],
    source_config: Mapping[str, Any],
    traces: Mapping[str, np.ndarray],
    *,
    source_order: int,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract one immutable geometric seed from loaded catalog evidence."""

    if int(source_config.get("schema_version", 0)) != 5:
        raise ValueError("source grasp config must use schema version 5")
    if source_config.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("source grasp config belongs to the wrong experiment")
    if not isinstance(source_order, int) or isinstance(source_order, bool) or source_order < 0:
        raise ValueError("source_order must be a non-negative integer")
    candidate_id = entry.get("candidate_id")
    if (
        not isinstance(candidate_id, int)
        or isinstance(candidate_id, bool)
        or candidate_id < 0
    ):
        raise ValueError("source candidate_id must be a non-negative integer")
    trajectory_id = entry.get("trajectory_id")
    label = entry.get("label")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("source trajectory_id must be a non-empty string")
    if not isinstance(label, str) or not label:
        raise ValueError("source label must be a non-empty string")
    parameter_override = entry.get("parameter_override_run")
    if not isinstance(parameter_override, bool):
        raise ValueError("parameter_override_run must be boolean")
    expected_context = {"kind": "parameter_override_run"} if parameter_override else None
    if source_config.get("run_context") != expected_context:
        raise ValueError("catalog override flag disagrees with source config")
    if entry.get("classification") != "validated_grasp_acquisition":
        raise ValueError("source trajectory is not classified as a validated grasp")
    if entry.get("rerun_grasp_success") is not True:
        raise ValueError("source trajectory did not reproduce grasp acquisition")
    grasp_validation = entry.get("grasp_validation")
    if not isinstance(grasp_validation, Mapping):
        raise ValueError("source trajectory has no grasp validation record")
    if grasp_validation.get("validation_scope") != "grasp_acquisition":
        raise ValueError("source grasp validation has the wrong scope")

    missing = [name for name in _TRACE_FIELDS if name not in traces]
    if missing:
        raise ValueError("source trace is missing: " + ", ".join(missing))
    time = np.asarray(traces["time"], dtype=np.float64)
    if time.ndim != 1 or not time.size or not np.isfinite(time).all():
        raise ValueError("source trace time must be a non-empty finite vector")
    step_array = np.asarray(traces["grasp_acquisition_step"])
    if step_array.shape != ():
        raise ValueError("grasp_acquisition_step must be scalar")
    step = int(step_array)
    if not 0 <= step < len(time):
        raise ValueError("grasp acquisition step is outside the source trace")
    states = np.asarray(traces["control_state"]).astype(str)
    acquired = np.asarray(traces["grasp_acquired"], dtype=bool)
    if states.shape != time.shape or acquired.shape != time.shape:
        raise ValueError("source control-state arrays must match the time axis")
    if states[step] != "VERIFY" or not acquired[step]:
        raise ValueError("source acquisition event is not a latched VERIFY sample")
    if np.any(acquired[:step]):
        raise ValueError("source grasp latch became true before its event step")
    if int(grasp_validation.get("grasp_acquisition_step", -1)) != step:
        raise ValueError("catalog and trace grasp acquisition steps disagree")
    if not math.isclose(
        float(grasp_validation.get("grasp_acquisition_time_s", math.nan)),
        float(time[step]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("catalog and trace grasp acquisition times disagree")
    if grasp_validation.get("no_manipulation_at_acquisition") is not True:
        raise ValueError("source grasp was not acquired before manipulation")

    actuator_order = tuple(str(value) for value in np.asarray(traces["actuator_order"]))
    if len(actuator_order) != len(set(actuator_order)):
        raise ValueError("source actuator_order contains duplicates")
    if not set(ACTIVE_ACTUATORS).issubset(actuator_order):
        raise ValueError("source actuator_order is missing active actuators")
    joint_qpos = np.asarray(traces["joint_qpos"], dtype=np.float64)
    if joint_qpos.shape != (len(time), len(actuator_order)) or not np.isfinite(
        joint_qpos
    ).all():
        raise ValueError("source joint_qpos shape or values are invalid")
    acquisition_qpos = {
        name: float(joint_qpos[step, index])
        for index, name in enumerate(actuator_order)
    }

    def pose_at(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(traces[f"{prefix}_pos"], dtype=np.float64)
        quaternions = np.asarray(traces[f"{prefix}_quat"], dtype=np.float64)
        if positions.shape != (len(time), 3) or quaternions.shape != (len(time), 4):
            raise ValueError(f"source {prefix} pose arrays have invalid shapes")
        if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
            raise ValueError(f"source {prefix} pose arrays contain NaN or Inf")
        return positions[step].copy(), _unit_quaternion(
            quaternions[step], f"source {prefix} quaternion"
        )

    cube_position, cube_quaternion = pose_at("cube")
    root_position, root_quaternion = pose_at("root")
    root_rotation = quaternion_to_rotation_matrix(root_quaternion)
    cube_rotation = quaternion_to_rotation_matrix(cube_quaternion)
    relative_position = root_rotation.T @ (cube_position - root_position)
    relative_rotation = root_rotation.T @ cube_rotation
    relative_quaternion = rotation_matrix_to_quaternion(relative_rotation)

    manipulation_delta = _target_mapping(
        source_config["control"]["manipulation_delta_rad"],
        "source manipulation_delta_rad",
    )
    if any(value != 0.0 for value in manipulation_delta.values()):
        raise ValueError("source manipulation deltas must be exactly zero")
    grasp_targets = _target_mapping(
        source_config["control"]["grasp_targets_rad"],
        "source grasp_targets_rad",
    )
    fixed_pose = initial_cube_world_pose(source_config)
    source_record: dict[str, Any] = {
        "source_order": source_order,
        "source_candidate_id": candidate_id,
        "trajectory_id": trajectory_id,
        "label": label,
        "parameter_override_run": parameter_override,
        "cube": copy.deepcopy(source_config["cube"]),
        "scene": copy.deepcopy(source_config["scene"]),
        "contact_topology": copy.deepcopy(source_config["contact_topology"]),
        "grasp_targets_rad": grasp_targets,
        "initial_cube_world_pose": fixed_pose,
        "acquisition": {
            "step": step,
            "time_s": float(time[step]),
            "actuator_qpos_rad": acquisition_qpos,
            "active_actuator_qpos_rad": {
                name: acquisition_qpos[name] for name in ACTIVE_ACTUATORS
            },
            "cube_pose_world": {
                "position_m": cube_position.tolist(),
                "quaternion_wxyz": cube_quaternion.tolist(),
            },
            "root_pose_world": {
                "position_m": root_position.tolist(),
                "quaternion_wxyz": root_quaternion.tolist(),
            },
        },
        "terminal_cube_in_root": {
            "position_m": relative_position.tolist(),
            "quaternion_wxyz": relative_quaternion.tolist(),
        },
        "provenance": copy.deepcopy(dict(provenance or {})),
    }
    source_record["fixed_object_sha256"] = canonical_sha256(
        {
            "cube": source_record["cube"],
            "scene": source_record["scene"],
            "initial_cube_world_pose": source_record["initial_cube_world_pose"],
        }
    )
    source_record["terminal_relation_sha256"] = canonical_sha256(
        {
            "method": RETARGET_METHOD,
            "acquisition_cube_pose_world": source_record["acquisition"][
                "cube_pose_world"
            ],
            "acquisition_root_pose_world": source_record["acquisition"][
                "root_pose_world"
            ],
            "terminal_cube_in_root": source_record["terminal_cube_in_root"],
            "terminal_grasp_targets_rad": source_record["grasp_targets_rad"],
        }
    )
    fingerprint_payload = {
        key: copy.deepcopy(source_record[key])
        for key in (
            "source_candidate_id",
            "trajectory_id",
            "parameter_override_run",
            "cube",
            "scene",
            "contact_topology",
            "grasp_targets_rad",
            "initial_cube_world_pose",
            "acquisition",
            "terminal_cube_in_root",
            "fixed_object_sha256",
            "terminal_relation_sha256",
            "provenance",
        )
    }
    source_record["source_sha256"] = canonical_sha256(fingerprint_payload)
    return source_record


def _catalog_member(catalog_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"catalog {label} path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"catalog {label} path must be safe and relative")
    root = catalog_path.parent.resolve()
    member = (root / relative).resolve()
    try:
        member.relative_to(root)
    except ValueError as error:
        raise ValueError(f"catalog {label} path escapes its directory") from error
    if not member.is_file():
        raise FileNotFoundError(f"catalog {label} does not exist: {member}")
    return member


def _verified_digest(
    path: Path,
    declared: Mapping[str, Any],
    field: str,
) -> str:
    expected = declared.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"catalog has no valid SHA-256 for {field}")
    actual = file_sha256(path)
    if actual != expected.lower():
        raise ValueError(f"catalog SHA-256 mismatch for {field}: {path}")
    return actual


def load_pose_preserving_seed_sources(
    catalog_path: str | Path,
    *,
    expected_count: int = EXPECTED_SOURCE_COUNT,
) -> tuple[dict[str, Any], ...]:
    """Load, authenticate and extract the six validated source trajectories."""

    catalog = Path(catalog_path).expanduser().resolve()
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("source grasp catalog must be a mapping")
    if payload.get("grasp_trajectory_catalog_schema_version") != 1:
        raise ValueError("unsupported source grasp catalog schema")
    if payload.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("source grasp catalog belongs to the wrong experiment")
    if payload.get("validation_scope") != "grasp_acquisition":
        raise ValueError("source catalog must validate grasp acquisition")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count <= 0
    ):
        raise ValueError("expected_count must be a positive integer")
    entries = payload.get("trajectories")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError(f"source catalog must contain exactly {expected_count} trajectories")
    if payload.get("validated_grasp_count") != expected_count:
        raise ValueError("every source trajectory must be a validated grasp")

    catalog_digest = file_sha256(catalog)
    sources: list[dict[str, Any]] = []
    seen_candidates: set[int] = set()
    seen_trajectories: set[str] = set()
    for source_order, entry_value in enumerate(entries):
        if not isinstance(entry_value, Mapping):
            raise ValueError("every source catalog trajectory must be a mapping")
        entry = entry_value
        validation = entry.get("grasp_validation")
        if not isinstance(validation, Mapping) or validation.get("passed") is not True:
            raise ValueError("every source trajectory must pass grasp validation")
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("source trajectory has no artifact mapping")
        digests = artifacts.get("sha256")
        if not isinstance(digests, Mapping):
            raise ValueError("source trajectory has no artifact SHA-256 mapping")
        config_path = _catalog_member(
            catalog, artifacts.get("resolved_config"), "resolved_config"
        )
        trace_path = _catalog_member(catalog, artifacts.get("trace"), "trace")
        config_digest = _verified_digest(config_path, digests, "resolved_config")
        trace_digest = _verified_digest(trace_path, digests, "trace")
        source_config = load_config(config_path)
        with np.load(trace_path, allow_pickle=False) as archive:
            missing = [name for name in _TRACE_FIELDS if name not in archive.files]
            if missing:
                raise ValueError("source trace is missing: " + ", ".join(missing))
            traces = {name: np.array(archive[name], copy=True) for name in _TRACE_FIELDS}
        provenance = {
            "catalog_sha256": catalog_digest,
            "resolved_config_artifact": str(artifacts["resolved_config"]),
            "resolved_config_sha256": config_digest,
            "trace_artifact": str(artifacts["trace"]),
            "trace_sha256": trace_digest,
        }
        source = extract_pose_preserving_source(
            entry,
            source_config,
            traces,
            source_order=source_order,
            provenance=provenance,
        )
        candidate_id = int(source["source_candidate_id"])
        trajectory_id = str(source["trajectory_id"])
        if candidate_id in seen_candidates:
            raise ValueError("source candidate_id values must be unique")
        if trajectory_id in seen_trajectories:
            raise ValueError("source trajectory_id values must be unique")
        seen_candidates.add(candidate_id)
        seen_trajectories.add(trajectory_id)
        sources.append(source)
    return tuple(sources)


def _v6_template(template: Mapping[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(template))
    validate_config(candidate)
    if int(candidate.get("schema_version", 0)) != 6:
        raise ValueError("pose-preserving seed template must use schema version 6")
    if candidate.get("experiment_id") != TARGET_EXPERIMENT_ID:
        raise ValueError("pose-preserving seed template belongs to the wrong experiment")
    return candidate


def _seed_hand_relation(
    source: Mapping[str, Any],
    template: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    # The acquired cube's small dynamic orientation drift must not be copied
    # into the new hand pitch: doing so produces a ~126 degree root pitch and
    # destroys the registered equal-height contact geometry.  Use the proven
    # v6 hand orientation, rotating it in world yaw only when a source cube's
    # configured yaw differs from the template cube yaw.
    source_cube_rpy = _finite_vector(
        source["cube"].get("rpy_deg", (0.0, 0.0, 0.0)),
        3,
        "source cube.rpy_deg",
    )
    template_cube_rpy = _finite_vector(
        template["cube"].get("rpy_deg", (0.0, 0.0, 0.0)),
        3,
        "template cube.rpy_deg",
    )
    reference_rpy = _finite_vector(
        template["hand_pose"]["rpy_deg"], 3, "template hand rpy_deg"
    )
    yaw_delta = float(source_cube_rpy[2] - template_cube_rpy[2])
    root_rotation = (
        rpy_degrees_to_rotation_matrix((0.0, 0.0, yaw_delta))
        @ rpy_degrees_to_rotation_matrix(reference_rpy)
    )
    root_rpy = rotation_matrix_to_rpy_degrees_near(
        root_rotation,
        reference_rpy + np.asarray((0.0, 0.0, yaw_delta)),
    )
    local_position = _finite_vector(
        source["terminal_cube_in_root"]["position_m"],
        3,
        "terminal_cube_in_root.position_m",
    )
    return root_rpy, local_position


def _registered_v6_canonical_status(config: Mapping[str, Any]) -> dict[str, bool]:
    """Check the constraints that ``run_context`` intentionally bypasses."""

    definition = resolve_experiment(dict(config))
    constraints = definition.far_hand_pose_constraints
    campaign = definition.far_hand_campaign
    if constraints is None or campaign is None:
        raise ValueError("schema-v6 experiment has no far-hand campaign constraints")
    cube = config["cube"]
    material = all(
        math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15)
        for actual, expected in (
            (cube["edge_m"], campaign.nominal_edge_m),
            (cube["mass_kg"], campaign.nominal_mass_kg),
            (cube["friction"], campaign.friction),
        )
    )
    resolved = resolved_pose_constraint_values(dict(config))
    pose = all(
        lower - 1e-12 <= float(resolved[label]) <= upper + 1e-12
        for label, (lower, upper) in (
            ("finger_down_tilt_deg", constraints.finger_down_tilt_deg),
            (
                "palm_plane_ground_angle_deg",
                constraints.palm_plane_ground_angle_deg,
            ),
            ("root_cube_distance_m", constraints.root_cube_distance_m),
        )
    )
    bounds = definition.search_bounds
    hand_rpy = config["hand_pose"]["rpy_deg"]
    cube_rpy = cube.get("rpy_deg", [0.0, 0.0, 0.0])
    pose = bool(
        pose
        and bounds.hand_roll_deg[0] - 1e-12
        <= float(hand_rpy[0])
        <= bounds.hand_roll_deg[1] + 1e-12
        and bounds.hand_yaw_deg[0] - 1e-12
        <= float(hand_rpy[2])
        <= bounds.hand_yaw_deg[1] + 1e-12
        and bounds.cube_yaw_deg[0] - 1e-12
        <= float(cube_rpy[2])
        <= bounds.cube_yaw_deg[1] + 1e-12
        and constraints.contains_cube_position(
            resolved["cube_position_in_root_m"]
        )
    )
    return {
        "material": bool(material),
        "pose_envelope": pose,
        "all": bool(material and pose),
    }


def assert_hand_pose_only_candidate_invariants(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    template: Mapping[str, Any],
) -> None:
    """Assert the fixed-object/fixed-terminal contracts of this campaign."""

    if config["cube"] != source["cube"]:
        raise ValueError("candidate cube block changed from its source")
    if config["scene"] != source["scene"]:
        raise ValueError("candidate scene block changed from its source")
    if config["contact_topology"] != source["contact_topology"]:
        raise ValueError("candidate contact topology changed from its source")
    control = config["control"]
    if control["grasp_targets_rad"] != source["grasp_targets_rad"]:
        raise ValueError("candidate terminal grasp targets changed from its source")
    if control["pregrasp_targets_rad"] != template["control"][
        "pregrasp_targets_rad"
    ]:
        raise ValueError("candidate pregrasp targets changed from the v6 template")
    if control["close_profile"] != template["control"]["close_profile"]:
        raise ValueError("candidate close profile changed from the v6 template")
    if control["manipulation_delta_rad"] != {
        name: 0.0 for name in ACTIVE_ACTUATORS
    }:
        raise ValueError("candidate manipulation delta must remain exactly zero")
    if initial_cube_world_pose(config) != source["initial_cube_world_pose"]:
        raise ValueError("candidate changed the configured initial cube world pose")
    resolved_local = np.asarray(
        resolved_pose_constraint_values(dict(config))["cube_position_in_root_m"],
        dtype=np.float64,
    )
    recorded_local = _finite_vector(
        config["candidate_metadata"]["candidate_cube_in_root_m"],
        3,
        "candidate_metadata.candidate_cube_in_root_m",
    )
    if not np.allclose(resolved_local, recorded_local, rtol=0.0, atol=2e-12):
        raise ValueError("candidate hand pose disagrees with its local cube relation")


def materialize_hand_pose_only_candidate(
    source: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    hand_rpy_deg: Sequence[float],
    cube_in_root_m: Sequence[float],
    candidate_id: int,
    local_index: int,
    seed: int,
    rpy_delta_deg: Sequence[float],
    cube_in_root_delta_m: Sequence[float],
) -> dict[str, Any]:
    """Create one valid schema-v6 config whose only sampled physics is hand pose."""

    resolved_template = _v6_template(template)
    if (
        not isinstance(candidate_id, int)
        or isinstance(candidate_id, bool)
        or candidate_id < 0
    ):
        raise ValueError("candidate_id must be a non-negative integer")
    if not isinstance(local_index, int) or isinstance(local_index, bool) or local_index < 0:
        raise ValueError("local_index must be a non-negative integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rpy = _finite_vector(hand_rpy_deg, 3, "hand_rpy_deg")
    local_position = _finite_vector(cube_in_root_m, 3, "cube_in_root_m")
    rpy_delta = _finite_vector(rpy_delta_deg, 3, "rpy_delta_deg")
    local_delta = _finite_vector(
        cube_in_root_delta_m, 3, "cube_in_root_delta_m"
    )
    cube_world_position = _finite_vector(
        source["initial_cube_world_pose"]["position_m"],
        3,
        "initial cube world position",
    )
    root_rotation = rpy_degrees_to_rotation_matrix(rpy)
    root_translation = cube_world_position - root_rotation @ local_position

    config = resolved_template
    config.pop("experiment_status", None)
    config.pop("candidate_metadata", None)
    config.pop("run_context", None)
    config["cube"] = copy.deepcopy(source["cube"])
    config["scene"] = copy.deepcopy(source["scene"])
    config["contact_topology"] = copy.deepcopy(source["contact_topology"])
    config["hand_pose"] = {
        "translation_m": root_translation.tolist(),
        "rpy_deg": rpy.tolist(),
    }
    template_control = template["control"]
    config["control"] = {
        "pregrasp_targets_rad": copy.deepcopy(
            template_control["pregrasp_targets_rad"]
        ),
        "grasp_targets_rad": copy.deepcopy(source["grasp_targets_rad"]),
        "manipulation_delta_rad": {
            name: 0.0 for name in ACTIVE_ACTUATORS
        },
        "close_profile": copy.deepcopy(template_control["close_profile"]),
    }
    metadata = {
        "campaign_kind": "pose_preserving_six_seed_hand_pose_only",
        "candidate_id": candidate_id,
        "local_index": local_index,
        # The exact retarget (local index zero) is non-random and must remain
        # byte-stable when a caller changes the perturbation seed.
        "seed": None if local_index == 0 else seed,
        "source_order": int(source["source_order"]),
        "source_candidate_id": int(source["source_candidate_id"]),
        "source_trajectory_id": str(source["trajectory_id"]),
        "source_sha256": str(source["source_sha256"]),
        "source_provenance": copy.deepcopy(source["provenance"]),
        "fixed_object_sha256": str(source["fixed_object_sha256"]),
        "terminal_relation_sha256": str(source["terminal_relation_sha256"]),
        "retarget_method": RETARGET_METHOD,
        "terminal_acquisition_step": int(source["acquisition"]["step"]),
        "terminal_active_qpos_rad": copy.deepcopy(
            source["acquisition"]["active_actuator_qpos_rad"]
        ),
        "terminal_cube_in_root": copy.deepcopy(
            source["terminal_cube_in_root"]
        ),
        "candidate_cube_in_root_m": local_position.tolist(),
        "hand_rpy_delta_deg": rpy_delta.tolist(),
        "cube_in_root_delta_m": local_delta.tolist(),
        "sampled_fields": ["hand_pose.translation_m", "hand_pose.rpy_deg"],
    }
    config["candidate_metadata"] = metadata

    canonical_status = _registered_v6_canonical_status(config)
    source_override = bool(source["parameter_override_run"])
    if source_override or not canonical_status["all"]:
        config["run_context"] = {"kind": "parameter_override_run"}
    metadata["canonical_v6_material"] = canonical_status["material"]
    metadata["canonical_v6_pose_envelope"] = canonical_status["pose_envelope"]
    metadata["parameter_override_run"] = "run_context" in config
    materialization_payload = {
        "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "retarget_method": RETARGET_METHOD,
        "source_sha256": source["source_sha256"],
        "fixed_object_sha256": source["fixed_object_sha256"],
        "terminal_relation_sha256": source["terminal_relation_sha256"],
        "hand_pose": config["hand_pose"],
        "cube": config["cube"],
        "scene": config["scene"],
        "contact_topology": config["contact_topology"],
        "control": config["control"],
        "run_context": config.get("run_context"),
    }
    metadata["materialization_sha256"] = canonical_sha256(
        materialization_payload
    )
    config["candidate_metadata"] = metadata
    assert_hand_pose_only_candidate_invariants(config, source, resolved_template)
    validate_config(config)
    return config


def _latin_hypercube(
    samples: int,
    dimensions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if samples <= 0 or dimensions <= 0:
        raise ValueError("Latin-hypercube dimensions must be positive")
    values = np.empty((samples, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        values[:, dimension] = (
            rng.permutation(samples) + rng.random(samples)
        ) / samples
    return values


def generate_hand_pose_only_candidates(
    source: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    hand_rpy_radius_deg: Sequence[float] = DEFAULT_HAND_RPY_RADIUS_DEG,
    cube_in_root_radius_m: Sequence[float] = DEFAULT_CUBE_IN_ROOT_RADIUS_M,
) -> tuple[dict[str, Any], ...]:
    """Generate stable-ID hand-root candidates for one source trajectory."""

    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count < _CANDIDATE_ID_STRIDE:
        raise ValueError(
            f"count must be within [1, {_CANDIDATE_ID_STRIDE - 1}]"
        )
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rpy_radius = _finite_vector(
        hand_rpy_radius_deg, 3, "hand_rpy_radius_deg"
    )
    local_radius = _finite_vector(
        cube_in_root_radius_m, 3, "cube_in_root_radius_m"
    )
    if np.any(rpy_radius < 0.0) or np.any(local_radius < 0.0):
        raise ValueError("candidate radii must be non-negative")
    source_candidate_id = int(source["source_candidate_id"])
    if source_candidate_id < 0:
        raise ValueError("source_candidate_id must be non-negative")
    base_rpy, base_local = _seed_hand_relation(source, template)
    offsets = np.zeros((count, 6), dtype=np.float64)
    if count > 1:
        seed_sequence = np.random.SeedSequence([seed, source_candidate_id])
        offsets[1:] = 2.0 * _latin_hypercube(
            count - 1, 6, np.random.default_rng(seed_sequence)
        ) - 1.0
    candidates: list[dict[str, Any]] = []
    for local_index, unit_offset in enumerate(offsets):
        rpy_delta = unit_offset[:3] * rpy_radius
        local_delta = unit_offset[3:] * local_radius
        candidate_id = source_candidate_id * _CANDIDATE_ID_STRIDE + local_index
        config = materialize_hand_pose_only_candidate(
            source,
            template,
            hand_rpy_deg=base_rpy + rpy_delta,
            cube_in_root_m=base_local + local_delta,
            candidate_id=candidate_id,
            local_index=local_index,
            seed=seed,
            rpy_delta_deg=rpy_delta,
            cube_in_root_delta_m=local_delta,
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source_order": int(source["source_order"]),
                "source_candidate_id": source_candidate_id,
                "source_trajectory_id": str(source["trajectory_id"]),
                "local_index": local_index,
                "config": config,
                "candidate_sha256": canonical_sha256(config),
            }
        )
    candidates.sort(key=lambda item: int(item["candidate_id"]))
    return tuple(candidates)


def build_pose_preserving_seed_campaign(
    sources: Iterable[Mapping[str, Any]],
    template: Mapping[str, Any],
    *,
    count_per_source: int,
    seed: int,
    hand_rpy_radius_deg: Sequence[float] = DEFAULT_HAND_RPY_RADIUS_DEG,
    cube_in_root_radius_m: Sequence[float] = DEFAULT_CUBE_IN_ROOT_RADIUS_M,
) -> dict[str, Any]:
    """Materialize one deterministic, source-stable six-seed campaign."""

    source_values = tuple(copy.deepcopy(dict(source)) for source in sources)
    if len(source_values) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            f"pose-preserving seed campaign requires {EXPECTED_SOURCE_COUNT} sources"
        )
    orders = tuple(int(source["source_order"]) for source in source_values)
    if orders != tuple(range(EXPECTED_SOURCE_COUNT)):
        raise ValueError("sources must retain their stable catalog order")
    source_ids = tuple(int(source["source_candidate_id"]) for source in source_values)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source candidate IDs must be unique")
    resolved_template = _v6_template(template)
    template_sha256 = canonical_sha256(resolved_template)
    candidates: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source in source_values:
        generated = generate_hand_pose_only_candidates(
            source,
            resolved_template,
            count=count_per_source,
            seed=seed,
            hand_rpy_radius_deg=hand_rpy_radius_deg,
            cube_in_root_radius_m=cube_in_root_radius_m,
        )
        candidates.extend(generated)
        source_summaries.append(
            {
                "source_order": int(source["source_order"]),
                "source_candidate_id": int(source["source_candidate_id"]),
                "trajectory_id": str(source["trajectory_id"]),
                "source_sha256": str(source["source_sha256"]),
                "candidate_ids": [
                    int(candidate["candidate_id"]) for candidate in generated
                ],
            }
        )
    candidates.sort(
        key=lambda item: (
            int(item["source_order"]),
            int(item["candidate_id"]),
        )
    )
    campaign = {
        "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_kind": "pose_preserving_six_seed_hand_pose_only",
        "target_experiment_id": TARGET_EXPERIMENT_ID,
        "seed": seed,
        "count_per_source": count_per_source,
        "source_count": len(source_values),
        "candidate_count": len(candidates),
        "hand_rpy_radius_deg": list(hand_rpy_radius_deg),
        "cube_in_root_radius_m": list(cube_in_root_radius_m),
        "template_sha256": template_sha256,
        "sources": source_summaries,
        "candidates": candidates,
    }
    campaign["campaign_sha256"] = canonical_sha256(campaign)
    return campaign


def load_and_build_pose_preserving_seed_campaign(
    catalog_path: str | Path,
    template_path: str | Path,
    *,
    count_per_source: int,
    seed: int,
    hand_rpy_radius_deg: Sequence[float] = DEFAULT_HAND_RPY_RADIUS_DEG,
    cube_in_root_radius_m: Sequence[float] = DEFAULT_CUBE_IN_ROOT_RADIUS_M,
) -> dict[str, Any]:
    """Filesystem convenience wrapper around the pure campaign builder."""

    sources = load_pose_preserving_seed_sources(catalog_path)
    template = load_config(template_path)
    campaign = build_pose_preserving_seed_campaign(
        sources,
        template,
        count_per_source=count_per_source,
        seed=seed,
        hand_rpy_radius_deg=hand_rpy_radius_deg,
        cube_in_root_radius_m=cube_in_root_radius_m,
    )
    campaign["provenance"] = {
        "catalog_path": str(Path(catalog_path).expanduser().resolve()),
        "catalog_sha256": file_sha256(catalog_path),
        "template_path": str(Path(template_path).expanduser().resolve()),
        "template_file_sha256": file_sha256(template_path),
    }
    # The content hash intentionally excludes checkout-specific absolute paths.
    return campaign


__all__ = [
    "CAMPAIGN_SCHEMA_VERSION",
    "DEFAULT_CUBE_IN_ROOT_RADIUS_M",
    "DEFAULT_HAND_RPY_RADIUS_DEG",
    "EXPECTED_SOURCE_COUNT",
    "RETARGET_METHOD",
    "RIGID_RETARGET_METHOD",
    "SOURCE_EXPERIMENT_ID",
    "TARGET_EXPERIMENT_ID",
    "assert_hand_pose_only_candidate_invariants",
    "build_pose_preserving_seed_campaign",
    "canonical_sha256",
    "extract_pose_preserving_source",
    "file_sha256",
    "generate_hand_pose_only_candidates",
    "initial_cube_world_pose",
    "load_and_build_pose_preserving_seed_campaign",
    "load_pose_preserving_seed_sources",
    "materialize_hand_pose_only_candidate",
    "quaternion_to_rotation_matrix",
    "retarget_root_pose",
    "rotation_matrix_to_quaternion",
    "rotation_matrix_to_rpy_degrees_near",
]
