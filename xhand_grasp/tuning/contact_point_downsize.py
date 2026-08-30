"""Deterministic contact-point continuation from a measured larger grasp.

This module contains the side-effect-free core used by a downsize campaign.
It deliberately owns neither CLI nor artifact-writing policy.  Its inputs are
an authenticated trajectory catalog and a registered target template; its
outputs are immutable evidence records, contact-point plans, mapped seeds and
static records directly consumable by the existing actual-contact dynamic
runner.

Two face-local mappings are supported and persisted under exact names:

``absolute_face_yz``
    Keep the measured cube-local ``(y, z)`` coordinates while deriving the
    face-normal coordinate from the target half edge.

``proportional_face_yz``
    Scale the measured ``(y, z)`` coordinates by
    ``target_edge / source_edge``.

Static DLS evidence is never a grasp-success claim.  Every promoted record
must still acquire the free cube from the configured no-contact initial state
through :func:`run_actual_contact_dynamic_grasp_candidates`.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..artifacts import file_sha256
from ..config import ACTIVE_ACTUATORS, load_config
from ..contacts import FACE_COUNT, FINGER_ORDER
from ..evaluation import face_from_label
from ..grasp_pose import canonical_sha256
from ..scene import rpy_degrees_to_rotation_matrix
from .actual_contact_grasp_pose import _cube_world_position
from .contact_point_targeted_search import (
    ContactPointPlan,
    ContactPointSearchPolicy,
    CubeFaceContactPoint,
    PointTargetDLSResult,
    PointTargetDLSSettings,
    PointTargetEvaluator,
    PointTargetVariables,
    bind_frozen_contact_point_plan,
    build_point_target_trial_evaluator,
    point_target_static_record,
    project_point_target_variables,
    solve_point_target_dls,
)


ABSOLUTE_FACE_YZ = "absolute_face_yz"
PROPORTIONAL_FACE_YZ = "proportional_face_yz"
MAPPING_MODES = (PROPORTIONAL_FACE_YZ, ABSOLUTE_FACE_YZ)
DEFAULT_SOURCE_EDGE_M = 0.089
DEFAULT_MAXIMUM_EDGE_M = 0.088
DEFAULT_MINIMUM_EDGE_M = 0.060
DEFAULT_EDGE_STEP_M = 0.001
DEFAULT_TARGET_RADIUS_M = 0.002
DEFAULT_EDGE_GUARD_M = 0.0005
DEFAULT_STABLE_WINDOW_STEPS = 250

_MAPPING_ALIASES = {
    ABSOLUTE_FACE_YZ: ABSOLUTE_FACE_YZ,
    "absolute": ABSOLUTE_FACE_YZ,
    PROPORTIONAL_FACE_YZ: PROPORTIONAL_FACE_YZ,
    "proportional": PROPORTIONAL_FACE_YZ,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOLERANCE = 1e-12


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(value: Any, length: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain {length} finite values") from error
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return result.copy()


def canonical_mapping_mode(value: str) -> str:
    """Return the persisted mapping name, accepting two short CLI aliases."""

    try:
        return _MAPPING_ALIASES[str(value).strip().lower()]
    except KeyError as error:
        raise ValueError(
            "mapping_mode must be absolute_face_yz or proportional_face_yz"
        ) from error


def _reference_target_face_yz_m(
    centroids_cube_local_m: Sequence[Sequence[float]],
) -> dict[str, list[float]]:
    centroids = np.asarray(centroids_cube_local_m, dtype=np.float64)
    if centroids.shape != (len(FINGER_ORDER), 3) or not np.isfinite(
        centroids
    ).all():
        raise ValueError("reference centroids must have shape (3, 3)")
    return {
        finger: [float(centroids[index, 1]), float(centroids[index, 2])]
        for index, finger in enumerate(FINGER_ORDER)
    }


@dataclass(frozen=True, slots=True)
class StableWindowContactEvidence:
    """Force-weighted target-face centroids over one authenticated window."""

    start_step: int
    end_step: int
    sample_count: int
    centroid_cube_local_m: tuple[tuple[float, float, float], ...]
    valid_counts: tuple[int, int, int]
    force_weight_sum_n: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.start_step < 0 or self.end_step < self.start_step:
            raise ValueError("stable-window step bounds are invalid")
        if self.sample_count != self.end_step - self.start_step + 1:
            raise ValueError("stable-window sample_count does not match its bounds")
        centroids = np.asarray(self.centroid_cube_local_m, dtype=np.float64)
        if centroids.shape != (len(FINGER_ORDER), 3) or not np.isfinite(
            centroids
        ).all():
            raise ValueError("centroid_cube_local_m must have shape (3, 3)")
        counts = tuple(int(value) for value in self.valid_counts)
        if len(counts) != len(FINGER_ORDER) or any(
            value < 0 or value > self.sample_count for value in counts
        ):
            raise ValueError("valid_counts must contain three in-window counts")
        force = np.asarray(self.force_weight_sum_n, dtype=np.float64)
        if force.shape != (len(FINGER_ORDER),) or not np.isfinite(force).all():
            raise ValueError("force_weight_sum_n must contain three finite values")
        if np.any(force <= 0.0):
            raise ValueError("force_weight_sum_n must be positive")

    def centroid_array(self) -> np.ndarray:
        """Return a mutable ``(thumb, index, mid) x xyz`` copy."""

        return np.asarray(self.centroid_cube_local_m, dtype=np.float64).copy()

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "sample_count": self.sample_count,
            "centroid_cube_local_m": [list(value) for value in self.centroid_cube_local_m],
            "valid_counts": list(self.valid_counts),
            "force_weight_sum_n": list(self.force_weight_sum_n),
            "finger_order": list(FINGER_ORDER),
            "aggregation": "sum_force_times_cube_local_position_over_sum_force",
        }


@dataclass(frozen=True, slots=True)
class DownsizeSourceEvidence:
    """Hash-authenticated catalog selection and its measured contact window."""

    catalog_path: Path
    catalog_sha256: str
    trajectory_id: str
    config_path: Path
    result_path: Path
    trace_path: Path
    config_sha256: str
    result_sha256: str
    trace_sha256: str
    config: dict[str, Any]
    result: dict[str, Any]
    stable_window: StableWindowContactEvidence

    @property
    def source_edge_m(self) -> float:
        return float(self.config["cube"]["edge_m"])

    def reference_contact_payload(
        self, source_alias: str | None = None
    ) -> dict[str, Any]:
        """Return the canonical payload that authenticates measured contacts.

        The mapping mode is intentionally absent: absolute and proportional
        continuation are two uses of the same measured source evidence.  The
        three artifact digests and exact stable-window bounds prevent a trace
        replacement from retaining the same semantic identity merely because
        it happens to report equal rounded centroids.
        """

        alias = self.trajectory_id if source_alias is None else str(source_alias)
        if not alias:
            raise ValueError("source_alias must be non-empty")
        return {
            "source_alias": alias,
            "config_sha256": self.config_sha256,
            "result_sha256": self.result_sha256,
            "trace_sha256": self.trace_sha256,
            "stable_window_start_step": self.stable_window.start_step,
            "stable_window_end_step": self.stable_window.end_step,
            "reference_edge_m": self.source_edge_m,
            "reference_target_face_yz_m": _reference_target_face_yz_m(
                self.stable_window.centroid_cube_local_m
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "catalog_path": str(self.catalog_path),
            "catalog_sha256": self.catalog_sha256,
            "trajectory_id": self.trajectory_id,
            "config_path": str(self.config_path),
            "result_path": str(self.result_path),
            "trace_path": str(self.trace_path),
            "config_sha256": self.config_sha256,
            "result_sha256": self.result_sha256,
            "trace_sha256": self.trace_sha256,
            "source_edge_m": self.source_edge_m,
            "stable_window": self.stable_window.as_dict(),
            "reference_contact_payload": self.reference_contact_payload(),
            "reference_contact_evidence_sha256": (
                self.reference_contact_evidence_sha256()
            ),
        }

    def reference_contact_evidence_sha256(
        self, source_alias: str | None = None
    ) -> str:
        return canonical_sha256(self.reference_contact_payload(source_alias))

    def scaled_contact_mapping(
        self,
        mapping_mode: str,
        point_plan: ContactPointPlan,
        source_alias: str | None = None,
    ) -> dict[str, Any]:
        """Return the complete schema-v13 mapping bound to one frozen plan."""

        alias = self.trajectory_id if source_alias is None else str(source_alias)
        payload = self.reference_contact_payload(alias)
        if not isinstance(point_plan, ContactPointPlan):
            raise TypeError("point_plan must be ContactPointPlan")
        return {
            "schema_version": 1,
            "source_alias": alias,
            "mapping_mode": canonical_mapping_mode(mapping_mode),
            "config_sha256": self.config_sha256,
            "result_sha256": self.result_sha256,
            "trace_sha256": self.trace_sha256,
            "stable_window_start_step": self.stable_window.start_step,
            "stable_window_end_step": self.stable_window.end_step,
            "reference_edge_m": self.source_edge_m,
            "reference_contact_evidence_sha256": canonical_sha256(payload),
            "reference_target_face_yz_m": copy.deepcopy(
                payload["reference_target_face_yz_m"]
            ),
            "target_edge_m": point_plan.edge_m,
            "derived_target_face_yz_m": {
                finger: [
                    float(point_plan.points[finger].y_m),
                    float(point_plan.points[finger].z_m),
                ]
                for finger in FINGER_ORDER
            },
            "contact_point_plan_id": point_plan.point_plan_id,
        }


def reference_contact_evidence_payload(
    evidence: DownsizeSourceEvidence, source_alias: str | None = None
) -> dict[str, Any]:
    """Public canonical projection used by schema and campaign validators."""

    if not isinstance(evidence, DownsizeSourceEvidence):
        raise TypeError("evidence must be DownsizeSourceEvidence")
    return evidence.reference_contact_payload(source_alias)


def reference_contact_evidence_sha256(
    evidence: DownsizeSourceEvidence, source_alias: str | None = None
) -> str:
    """Hash authenticated source artifacts, window bounds and measured Y/Z."""

    return canonical_sha256(
        reference_contact_evidence_payload(evidence, source_alias)
    )


def build_scaled_contact_mapping(
    evidence: DownsizeSourceEvidence,
    mapping_mode: str,
    point_plan: ContactPointPlan,
    source_alias: str | None = None,
) -> dict[str, Any]:
    """Build the exact fourteen-field schema-v13 mapping descriptor."""

    return evidence.scaled_contact_mapping(mapping_mode, point_plan, source_alias)


@dataclass(frozen=True, slots=True)
class DownsizeSeedMapping:
    """A target-size config and diagnostics for the preserved hand relation."""

    config: dict[str, Any]
    source_edge_m: float
    target_edge_m: float
    source_cube_world_position_m: tuple[float, float, float]
    target_cube_world_position_m: tuple[float, float, float]
    cube_in_root_m: tuple[float, float, float]
    hand_translation_delta_world_m: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_edge_m": self.source_edge_m,
            "target_edge_m": self.target_edge_m,
            "source_cube_world_position_m": list(self.source_cube_world_position_m),
            "target_cube_world_position_m": list(self.target_cube_world_position_m),
            "cube_in_root_m": list(self.cube_in_root_m),
            "hand_translation_delta_world_m": list(
                self.hand_translation_delta_world_m
            ),
            "cube_pose_sampled": False,
            "hand_root_fixed_during_simulation": True,
            "mapping": "preserve_source_cube_in_root",
        }


@dataclass(frozen=True, slots=True)
class DownsizeStaticCandidate:
    """One mapped point plan and its real-witness 14-variable DLS result."""

    source_id: str
    source_edge_m: float
    target_edge_m: float
    mapping_mode: str
    seed_mapping: DownsizeSeedMapping
    point_plan: ContactPointPlan
    dls_result: PointTargetDLSResult

    @property
    def static_pass(self) -> bool:
        return bool(self.dls_result.acceptance.passed)

    def as_dynamic_record(
        self, candidate_id: int, source_id: str | int | None = None
    ) -> dict[str, Any]:
        """Adapt this proposal to the existing actual-contact dynamic runner."""

        resolved_source = self.source_id if source_id is None else source_id
        record = point_target_static_record(
            candidate_id, self.dls_result, source_id=resolved_source
        )
        record["edge_m"] = self.target_edge_m
        record["mapping_mode"] = self.mapping_mode
        record["downsize"] = {
            "source_id": str(resolved_source),
            "source_alias": str(resolved_source),
            "source_edge_m": self.source_edge_m,
            "target_edge_m": self.target_edge_m,
            "edge_m": self.target_edge_m,
            "mapping_mode": self.mapping_mode,
            "point_plan_id": self.point_plan.point_plan_id,
            "seed_mapping": self.seed_mapping.as_dict(),
            "static_filter_is_success_evidence": False,
        }
        return record


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return copy.deepcopy(dict(value))


def _resolve_catalog_artifact(catalog_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"catalog artifact {label} must be a non-empty path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"catalog artifact {label} must be relative")
    root = catalog_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"catalog artifact {label} escapes the catalog root") from error
    if not path.is_file():
        raise FileNotFoundError(f"catalog artifact {label} is missing: {path}")
    return path


def _verify_artifact_sha(path: Path, expected: Any, label: str) -> str:
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise ValueError(f"catalog {label} SHA-256 is missing or malformed")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"catalog {label} SHA-256 mismatch: expected {expected}, got {observed}"
        )
    return observed


def _select_catalog_entry(catalog: Mapping[str, Any], selection: str) -> dict[str, Any]:
    trajectories = catalog.get("trajectories")
    if not isinstance(trajectories, Sequence) or isinstance(
        trajectories, (str, bytes)
    ):
        raise ValueError("catalog.trajectories must be a sequence")
    aliases = catalog.get("aliases", {})
    if not isinstance(aliases, Mapping):
        raise ValueError("catalog.aliases must be a mapping")
    selected = str(aliases.get(selection, selection))
    matches: list[Mapping[str, Any]] = []
    for raw in trajectories:
        if not isinstance(raw, Mapping):
            raise ValueError("catalog trajectory entries must be mappings")
        entry_aliases = raw.get("aliases", ())
        if not isinstance(entry_aliases, Sequence) or isinstance(
            entry_aliases, (str, bytes)
        ):
            raise ValueError("catalog trajectory aliases must be a sequence")
        identities = {
            str(raw.get("trajectory_id", "")),
            str(raw.get("label", "")),
            *(str(value) for value in entry_aliases),
        }
        if selected in identities:
            matches.append(raw)
    if len(matches) != 1:
        raise ValueError(
            f"catalog selection {selection!r} resolved to {len(matches)} entries"
        )
    return copy.deepcopy(dict(matches[0]))


def extract_stable_window_force_weighted_centroids(
    trace: Any,
    config: Mapping[str, Any],
    *,
    required_sample_count: int = DEFAULT_STABLE_WINDOW_STEPS,
    verify_persisted_cache: bool = True,
) -> StableWindowContactEvidence:
    """Recompute force-weighted cube-local centroids from raw trace evidence.

    The stable-window endpoints are inclusive.  Every target finger must have
    positive target-face normal force in all required samples; a missing frame
    is rejected rather than silently reducing the averaging denominator.
    """

    if (
        not isinstance(required_sample_count, int)
        or isinstance(required_sample_count, bool)
        or required_sample_count <= 0
    ):
        raise ValueError("required_sample_count must be a positive integer")
    required = (
        "distal_face_force_n",
        "distal_face_position_moment_n_m",
        "cube_pos",
        "cube_quat",
        "grasp_stable_window_start_step",
        "grasp_stable_window_end_step",
    )
    missing = [key for key in required if key not in trace]
    if missing:
        raise ValueError("trace is missing required fields: " + ", ".join(missing))
    force = np.asarray(trace["distal_face_force_n"], dtype=np.float64)
    moment = np.asarray(
        trace["distal_face_position_moment_n_m"], dtype=np.float64
    )
    cube_pos = np.asarray(trace["cube_pos"], dtype=np.float64)
    cube_quat = np.asarray(trace["cube_quat"], dtype=np.float64)
    if force.ndim != 3 or force.shape[1:] != (len(FINGER_ORDER), FACE_COUNT):
        raise ValueError("distal_face_force_n must have shape (T, 3, 8)")
    total_steps = force.shape[0]
    if moment.shape != force.shape + (3,):
        raise ValueError(
            "distal_face_position_moment_n_m must have shape (T, 3, 8, 3)"
        )
    if cube_pos.shape != (total_steps, 3) or cube_quat.shape != (total_steps, 4):
        raise ValueError("cube_pos/cube_quat have incompatible trace shapes")
    if (
        not np.isfinite(force).all()
        or np.any(force < 0.0)
        or not np.isfinite(moment).all()
        or not np.isfinite(cube_pos).all()
        or not np.isfinite(cube_quat).all()
    ):
        raise ValueError("stable-window raw contact trace must be finite")
    start = int(np.asarray(trace["grasp_stable_window_start_step"]))
    end = int(np.asarray(trace["grasp_stable_window_end_step"]))
    if start < 0 or end < start or end >= total_steps:
        raise ValueError("trace stable-window bounds are invalid")
    sample_count = end - start + 1
    if sample_count != required_sample_count:
        raise ValueError(
            f"stable window must contain exactly {required_sample_count} samples"
        )
    topology = config.get("contact_topology", {})
    raw_faces = topology.get("target_faces", {}) if isinstance(topology, Mapping) else {}
    if not isinstance(raw_faces, Mapping) or set(raw_faces) != set(FINGER_ORDER):
        raise ValueError("config contact_topology must contain three target faces")
    face_indices = np.asarray(
        [int(face_from_label(raw_faces[finger])) for finger in FINGER_ORDER],
        dtype=np.intp,
    )
    finger_indices = np.arange(len(FINGER_ORDER), dtype=np.intp)
    target_force = force[start : end + 1, finger_indices, face_indices]
    target_moment = moment[start : end + 1, finger_indices, face_indices]
    valid = target_force > 0.0
    valid_counts = np.count_nonzero(valid, axis=0).astype(np.int64)
    if np.any(valid_counts != sample_count):
        detail = ", ".join(
            f"{finger}={int(valid_counts[index])}"
            for index, finger in enumerate(FINGER_ORDER)
        )
        raise ValueError(
            "every finger must have target-face force in all stable-window "
            f"samples ({detail})"
        )
    world = target_moment / target_force[:, :, np.newaxis]
    local = np.empty_like(world)
    for local_step, absolute_step in enumerate(range(start, end + 1)):
        quaternion = cube_quat[absolute_step]
        norm = float(np.linalg.norm(quaternion))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("cube_quat contains a zero quaternion")
        rotation_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation_flat, quaternion / norm)
        rotation = rotation_flat.reshape(3, 3)
        local[local_step] = (
            rotation.T @ (world[local_step] - cube_pos[absolute_step]).T
        ).T
    if verify_persisted_cache and "target_face_contact_centroid_world_m" in trace:
        persisted = np.asarray(
            trace["target_face_contact_centroid_world_m"], dtype=np.float64
        )[start : end + 1]
        if persisted.shape != world.shape or not np.allclose(
            persisted, world, atol=1e-12, rtol=0.0
        ):
            raise ValueError("persisted target-face centroid cache is inconsistent")
    if verify_persisted_cache and "target_face_contact_centroid_valid" in trace:
        persisted_valid = np.asarray(
            trace["target_face_contact_centroid_valid"], dtype=bool
        )[start : end + 1]
        if persisted_valid.shape != valid.shape or not np.array_equal(
            persisted_valid, valid
        ):
            raise ValueError("persisted target-face centroid mask is inconsistent")
    force_sum = np.sum(target_force, axis=0)
    centroids = np.sum(target_force[:, :, np.newaxis] * local, axis=0) / force_sum[
        :, np.newaxis
    ]
    return StableWindowContactEvidence(
        start_step=start,
        end_step=end,
        sample_count=sample_count,
        centroid_cube_local_m=tuple(
            tuple(float(value) for value in row) for row in centroids
        ),
        valid_counts=tuple(int(value) for value in valid_counts),
        force_weight_sum_n=tuple(float(value) for value in force_sum),
    )


def audit_downsize_source(
    catalog_path: str | Path,
    trajectory: str = "best_nominal",
    *,
    require_hard_pass: bool = True,
    required_stable_window_steps: int = DEFAULT_STABLE_WINDOW_STEPS,
) -> DownsizeSourceEvidence:
    """Resolve one catalog entry and fail closed on any artifact corruption."""

    catalog_file = Path(catalog_path).expanduser().resolve()
    if not catalog_file.is_file():
        raise FileNotFoundError(f"source catalog is missing: {catalog_file}")
    catalog = _load_json_mapping(catalog_file, "source catalog")
    entry = _select_catalog_entry(catalog, trajectory)
    if require_hard_pass and entry.get("hard_pass") is not True:
        raise ValueError("selected source trajectory is not a hard-pass grasp")
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("selected catalog entry has no artifacts mapping")
    hashes = artifacts.get("sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("selected catalog entry has no artifact SHA-256 mapping")
    root = catalog_file.parent
    config_path = _resolve_catalog_artifact(
        root, artifacts.get("resolved_config"), "resolved_config"
    )
    result_path = _resolve_catalog_artifact(root, artifacts.get("result"), "result")
    trace_path = _resolve_catalog_artifact(root, artifacts.get("trace"), "trace")
    config_sha = _verify_artifact_sha(
        config_path, hashes.get("resolved_config"), "resolved_config"
    )
    result_sha = _verify_artifact_sha(result_path, hashes.get("result"), "result")
    trace_sha = _verify_artifact_sha(trace_path, hashes.get("trace"), "trace")
    config = load_config(config_path)
    result = _load_json_mapping(result_path, "source result")
    record = result.get("record", {})
    if require_hard_pass and (
        not isinstance(record, Mapping)
        or record.get("hard_pass") is not True
        or record.get("strict_grasp_success") is not True
    ):
        raise ValueError("source result does not contain strict hard-pass evidence")
    with np.load(trace_path, allow_pickle=False) as trace:
        stable = extract_stable_window_force_weighted_centroids(
            trace,
            config,
            required_sample_count=required_stable_window_steps,
        )
    if isinstance(record, Mapping):
        for field, expected in (
            ("stable_window_start_step", stable.start_step),
            ("stable_window_end_step", stable.end_step),
        ):
            if field in record and int(record[field]) != expected:
                raise ValueError(f"source result {field} disagrees with trace")
    return DownsizeSourceEvidence(
        catalog_path=catalog_file,
        catalog_sha256=file_sha256(catalog_file),
        trajectory_id=str(entry.get("trajectory_id", entry.get("label"))),
        config_path=config_path,
        result_path=result_path,
        trace_path=trace_path,
        config_sha256=config_sha,
        result_sha256=result_sha,
        trace_sha256=trace_sha,
        config=copy.deepcopy(config),
        result=result,
        stable_window=stable,
    )


def descending_edge_schedule(
    source_edge_m: float = DEFAULT_SOURCE_EDGE_M,
    minimum_edge_m: float = DEFAULT_MINIMUM_EDGE_M,
    maximum_edge_m: float = DEFAULT_MAXIMUM_EDGE_M,
    step_m: float = DEFAULT_EDGE_STEP_M,
) -> tuple[float, ...]:
    """Return an exact descending, prefix-stable target-size schedule."""

    source = _finite(source_edge_m, "source_edge_m")
    minimum = _finite(minimum_edge_m, "minimum_edge_m")
    maximum = _finite(maximum_edge_m, "maximum_edge_m")
    step = _finite(step_m, "step_m")
    if step <= 0.0 or minimum <= 0.0 or maximum < minimum:
        raise ValueError("edge schedule bounds and step are invalid")

    def units(value: float, label: str) -> int:
        result = int(round(value / step))
        if not math.isclose(result * step, value, abs_tol=1e-10, rel_tol=0.0):
            raise ValueError(f"{label} must align to step_m")
        return result

    source_unit = units(source, "source_edge_m")
    minimum_unit = units(minimum, "minimum_edge_m")
    maximum_unit = min(units(maximum, "maximum_edge_m"), source_unit - 1)
    if maximum_unit < minimum_unit:
        return ()
    return tuple(unit * step for unit in range(maximum_unit, minimum_unit - 1, -1))


def build_downsize_contact_point_plan(
    source_edge_m: float,
    target_edge_m: float,
    source_centroids_cube_local_m: Sequence[Sequence[float]],
    *,
    mapping_mode: str = ABSOLUTE_FACE_YZ,
    target_radius_m: float = DEFAULT_TARGET_RADIUS_M,
    edge_guard_m: float = DEFAULT_EDGE_GUARD_M,
) -> ContactPointPlan:
    """Map measured centroids to one immutable target-edge point plan."""

    source_edge = _finite(source_edge_m, "source_edge_m")
    target_edge = _finite(target_edge_m, "target_edge_m")
    radius = _finite(target_radius_m, "target_radius_m")
    guard = _finite(edge_guard_m, "edge_guard_m")
    if source_edge <= 0.0 or target_edge <= 0.0 or target_edge >= source_edge:
        raise ValueError("target_edge_m must be positive and smaller than source_edge_m")
    if radius <= 0.0 or guard < 0.0:
        raise ValueError("target radius must be positive and edge guard non-negative")
    centroids = np.asarray(source_centroids_cube_local_m, dtype=np.float64)
    if centroids.shape != (len(FINGER_ORDER), 3) or not np.isfinite(
        centroids
    ).all():
        raise ValueError("source_centroids_cube_local_m must have shape (3, 3)")
    mode = canonical_mapping_mode(mapping_mode)
    scale = 1.0 if mode == ABSOLUTE_FACE_YZ else target_edge / source_edge
    yz = centroids[:, 1:3] * scale
    half = 0.5 * target_edge
    required_margin = radius + guard
    margins = half - np.max(np.abs(yz), axis=1)
    if np.any(margins < required_margin - _TOLERANCE):
        failing = ", ".join(
            f"{FINGER_ORDER[index]}={margins[index]:.9g}m"
            for index in np.flatnonzero(margins < required_margin - _TOLERANCE)
        )
        raise ValueError(
            "mapped target circle violates edge guard "
            f"(requires radius+guard={required_margin:.9g}m; {failing})"
        )
    faces = {"thumb": "-X", "index": "+X", "mid": "+X"}
    return ContactPointPlan(
        edge_m=target_edge,
        target_radius_m=radius,
        points={
            finger: CubeFaceContactPoint(
                faces[finger], float(yz[index, 0]), float(yz[index, 1])
            )
            for index, finger in enumerate(FINGER_ORDER)
        },
    )


def map_downsize_seed_config(
    source_config: Mapping[str, Any],
    target_edge_m: float,
    *,
    target_template: Mapping[str, Any] | None = None,
    scaled_contact_mapping: Mapping[str, Any] | None = None,
) -> DownsizeSeedMapping:
    """Map the source cube-in-root relation to a smaller supported cube."""

    source = copy.deepcopy(dict(source_config))
    source_edge = _finite(source["cube"]["edge_m"], "source cube.edge_m")
    target_edge = _finite(target_edge_m, "target_edge_m")
    if not 0.0 < target_edge < source_edge:
        raise ValueError("target_edge_m must be positive and smaller than source edge")
    result = copy.deepcopy(dict(target_template)) if target_template is not None else copy.deepcopy(source)
    if not isinstance(result.get("cube"), Mapping):
        raise ValueError("target template has no cube mapping")
    if target_template is not None:
        # The source defines the fixed world pose; a target template defines
        # experiment/schema policy and physical properties such as mass.
        result["cube"]["center_xy_m"] = copy.deepcopy(source["cube"]["center_xy_m"])
        result["cube"]["rpy_deg"] = copy.deepcopy(
            source["cube"].get("rpy_deg", (0.0, 0.0, 0.0))
        )
        result["cube"]["z_offset_m"] = float(
            source["cube"].get("z_offset_m", 0.0)
        )
    result["cube"]["edge_m"] = target_edge

    source_cube = _cube_world_position(source)
    target_cube = _cube_world_position(result)
    source_rotation = rpy_degrees_to_rotation_matrix(source["hand_pose"]["rpy_deg"])
    source_root = _vector(
        source["hand_pose"]["translation_m"], 3, "source hand translation"
    )
    cube_in_root = source_rotation.T @ (source_cube - source_root)
    target_root = target_cube - source_rotation @ cube_in_root
    result["hand_pose"] = {
        "rpy_deg": copy.deepcopy(source["hand_pose"]["rpy_deg"]),
        "translation_m": target_root.tolist(),
    }
    if scaled_contact_mapping is not None:
        result["scaled_contact_mapping"] = copy.deepcopy(
            dict(scaled_contact_mapping)
        )
    if target_template is not None:
        result.setdefault("grasp_pose", {})["nominal_joint_qpos_rad"] = copy.deepcopy(
            source["grasp_pose"]["nominal_joint_qpos_rad"]
        )
        result.setdefault("control", {})
        for field in (
            "precontact_targets_rad",
            "contact_preload_targets_rad",
        ):
            if field in source.get("control", {}):
                result["control"][field] = copy.deepcopy(source["control"][field])
        if "manipulation_delta_rad" in result["control"]:
            result["control"]["manipulation_delta_rad"] = {
                name: 0.0 for name in ACTIVE_ACTUATORS
            }
    return DownsizeSeedMapping(
        config=result,
        source_edge_m=source_edge,
        target_edge_m=target_edge,
        source_cube_world_position_m=tuple(float(value) for value in source_cube),
        target_cube_world_position_m=tuple(float(value) for value in target_cube),
        cube_in_root_m=tuple(float(value) for value in cube_in_root),
        hand_translation_delta_world_m=tuple(
            float(value) for value in target_root - source_root
        ),
    )


def _downsize_point_policy(
    config: Mapping[str, Any],
    plan: ContactPointPlan,
    *,
    edge_guard_m: float,
) -> ContactPointSearchPolicy:
    relative = config.get("relative_wrist_pose_search", {})
    if isinstance(relative, Mapping):
        root_bounds = relative.get("root_delta_cube_m")
        wrist_bounds = relative.get("wrist_local_rotvec_deg")
        wrist_norm = relative.get("max_wrist_local_rotvec_norm_deg", 8.0)
        distance = relative.get("root_cube_distance_m")
    else:
        root_bounds = wrist_bounds = distance = None
        wrist_norm = 8.0
    constraints = config.get("pose_constraints", {})
    if distance is None and isinstance(constraints, Mapping):
        distance = constraints.get("root_cube_distance_m")
    return ContactPointSearchPolicy(
        sample_count=1,
        retain_point_plan_count=1,
        reference_points=plan.points,
        minimum_edge_margin_m=plan.target_radius_m + edge_guard_m,
        minimum_index_middle_separation_m=0.001,
        static_target_radius_m=plan.target_radius_m,
        target_radius_m=plan.target_radius_m,
        signed_orbit_deg=(0.0,),
        root_delta_cube_m=root_bounds,
        wrist_local_rotvec_deg=wrist_bounds,
        max_wrist_local_rotvec_norm_deg=float(wrist_norm),
        root_cube_distance_m=(
            (0.135, 0.210) if distance is None else tuple(distance)
        ),
        thumb_actual_range_rad=tuple(
            config.get("grasp_pose", {}).get(
                "thumb_actual_range_rad", (1.40, 1.60)
            )
        ),
    )


def point_target_variables_from_previous_edge(
    base_config: Mapping[str, Any],
    previous_edge_config: Mapping[str, Any],
) -> PointTargetVariables:
    """Express a previous-edge hand relation in the new DLS anchor frame.

    The free cube's world height changes with edge length, so copying a world
    hand translation directly is incorrect.  This maps the prior root through
    cube coordinates, while preserving its actual nominal joint shape and its
    cube-relative wrist orientation.
    """

    target_edge = _finite(base_config["cube"]["edge_m"], "target cube.edge_m")
    previous_edge = _finite(
        previous_edge_config["cube"]["edge_m"], "previous cube.edge_m"
    )
    if previous_edge <= target_edge + _TOLERANCE:
        raise ValueError("previous-edge continuation requires a larger prior edge")
    for field in ("center_xy_m", "rpy_deg"):
        if not np.allclose(
            np.asarray(base_config["cube"][field], dtype=np.float64),
            np.asarray(previous_edge_config["cube"][field], dtype=np.float64),
            atol=_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError(f"previous-edge cube {field} changed")
    current_mapping = base_config.get("scaled_contact_mapping", {})
    previous_mapping = previous_edge_config.get("scaled_contact_mapping", {})
    if isinstance(current_mapping, Mapping) and isinstance(
        previous_mapping, Mapping
    ):
        for field in ("source_alias", "mapping_mode"):
            if field in previous_mapping and previous_mapping[field] != current_mapping.get(
                field
            ):
                raise ValueError(f"previous-edge {field} changed")

    target_cube_position = _cube_world_position(base_config)
    previous_cube_position = _cube_world_position(previous_edge_config)
    target_cube_rotation = rpy_degrees_to_rotation_matrix(
        base_config["cube"]["rpy_deg"]
    )
    previous_cube_rotation = rpy_degrees_to_rotation_matrix(
        previous_edge_config["cube"]["rpy_deg"]
    )
    metadata = base_config.get("candidate_metadata", {})
    point_metadata = (
        metadata.get("contact_point_target_search", {})
        if isinstance(metadata, Mapping)
        else {}
    )
    anchor = point_metadata.get("anchor_hand_pose", base_config["hand_pose"])
    anchor_position = _vector(anchor["translation_m"], 3, "anchor translation")
    anchor_rotation = rpy_degrees_to_rotation_matrix(anchor["rpy_deg"])
    previous_position = _vector(
        previous_edge_config["hand_pose"]["translation_m"],
        3,
        "previous hand translation",
    )
    previous_rotation = rpy_degrees_to_rotation_matrix(
        previous_edge_config["hand_pose"]["rpy_deg"]
    )
    anchor_in_cube = target_cube_rotation.T @ (
        anchor_position - target_cube_position
    )
    previous_in_cube = previous_cube_rotation.T @ (
        previous_position - previous_cube_position
    )
    root_delta = previous_in_cube - anchor_in_cube
    anchor_cube_from_root = target_cube_rotation.T @ anchor_rotation
    previous_cube_from_root = previous_cube_rotation.T @ previous_rotation
    local_rotation = anchor_cube_from_root.T @ previous_cube_from_root
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, local_rotation.reshape(-1))
    rotvec = np.empty(3, dtype=np.float64)
    mujoco.mju_quat2Vel(rotvec, quaternion, 1.0)
    nominal = previous_edge_config["grasp_pose"]["nominal_joint_qpos_rad"]
    return PointTargetVariables(
        tuple(float(nominal[name]) for name in ACTIVE_ACTUATORS),
        tuple(float(value) for value in root_delta),
        tuple(float(value) for value in rotvec),
    )


def solve_downsize_static_candidate(
    source_config: Mapping[str, Any],
    target_edge_m: float,
    source_centroids_cube_local_m: Sequence[Sequence[float]],
    *,
    mapping_mode: str = ABSOLUTE_FACE_YZ,
    source_id: str | int = "source",
    source_evidence: DownsizeSourceEvidence | None = None,
    previous_edge_config: Mapping[str, Any] | None = None,
    target_template: Mapping[str, Any] | None = None,
    target_radius_m: float = DEFAULT_TARGET_RADIUS_M,
    edge_guard_m: float = DEFAULT_EDGE_GUARD_M,
    settings: PointTargetDLSSettings | None = None,
    seed: int = 20260821,
    start_index: int = 0,
    max_iterations: int | None = None,
    evaluator: PointTargetEvaluator | None = None,
    joint_bounds: Mapping[str, tuple[float, float]] | None = None,
    check_pose_constraints: bool = True,
) -> DownsizeStaticCandidate:
    """Map and solve one target edge using the existing 14-variable DLS.

    Production callers should pass a registered target template.  Tests may
    inject an evaluator and joint bounds to exercise orchestration without
    compiling MuJoCo.  The exact source hand pose is rebased as the zero-orbit,
    zero-residual anchor, avoiding an implicit Euler-only wrist edit.
    """

    mode = canonical_mapping_mode(mapping_mode)
    if source_evidence is None:
        raise ValueError(
            "source_evidence is required to bind scaled_contact_mapping to "
            "authenticated config/result/trace evidence"
        )
    if canonical_sha256(dict(source_config)) != canonical_sha256(
        source_evidence.config
    ):
        raise ValueError("source_config does not match source_evidence")
    if not np.array_equal(
        np.asarray(source_centroids_cube_local_m, dtype=np.float64),
        source_evidence.stable_window.centroid_array(),
    ):
        raise ValueError("source centroids do not match source_evidence")
    source_alias = str(source_id)
    mapped_seed = map_downsize_seed_config(
        source_config,
        target_edge_m,
        target_template=target_template,
    )
    plan = build_downsize_contact_point_plan(
        mapped_seed.source_edge_m,
        mapped_seed.target_edge_m,
        source_centroids_cube_local_m,
        mapping_mode=mode,
        target_radius_m=target_radius_m,
        edge_guard_m=edge_guard_m,
    )
    scaled_mapping = source_evidence.scaled_contact_mapping(
        mode, plan, source_alias
    )
    base = copy.deepcopy(mapped_seed.config)
    base.pop("contact_point_plan", None)
    base["scaled_contact_mapping"] = scaled_mapping
    base = bind_frozen_contact_point_plan(base, plan)
    metadata = base.setdefault("candidate_metadata", {})
    metadata["contact_point_target_search"] = {
        "anchor_hand_pose": copy.deepcopy(mapped_seed.config["hand_pose"]),
        "signed_orbit_deg": 0.0,
        "root_delta_cube_m": [0.0, 0.0, 0.0],
        "wrist_local_rotvec_deg": [0.0, 0.0, 0.0],
        "point_plan_id": plan.point_plan_id,
        "cube_pose_sampled": False,
        "hand_root_fixed_during_simulation": True,
    }
    metadata["contact_point_downsize"] = {
        "source_id": str(source_id),
        "source_alias": str(source_id),
        "source_edge_m": mapped_seed.source_edge_m,
        "target_edge_m": mapped_seed.target_edge_m,
        "edge_m": mapped_seed.target_edge_m,
        "mapping_mode": mode,
        "target_radius_m": plan.target_radius_m,
        "edge_guard_m": float(edge_guard_m),
        "seed_mapping": mapped_seed.as_dict(),
        "reference_contact_evidence_sha256": scaled_mapping[
            "reference_contact_evidence_sha256"
        ],
        "static_filter_is_success_evidence": False,
        "initialization_branch": (
            "previous_edge_continuation"
            if previous_edge_config is not None
            else "reference_edge_direct"
        ),
    }
    policy = _downsize_point_policy(base, plan, edge_guard_m=edge_guard_m)
    initial = PointTargetVariables(
        tuple(
            float(base["grasp_pose"]["nominal_joint_qpos_rad"][name])
            for name in ACTIVE_ACTUATORS
        ),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    if previous_edge_config is not None:
        initial = point_target_variables_from_previous_edge(
            base, previous_edge_config
        )
    if isinstance(start_index, bool) or int(start_index) != start_index or start_index < 0:
        raise ValueError("start_index must be a non-negative integer")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    if evaluator is None:
        evaluator, model_bounds = build_point_target_trial_evaluator(base, plan)
        if joint_bounds is None:
            joint_bounds = model_bounds
    if joint_bounds is None:
        raise ValueError("joint_bounds are required with an injected evaluator")
    if start_index:
        # Production bounds become available only after the evaluator/model is
        # built.  Recreate the same deterministic start and project it now.
        generator = np.random.default_rng(
            np.random.SeedSequence(
                [int(seed), int(start_index), int(round(target_edge_m * 1e6))]
            )
        )
        values = initial.as_array()
        values[:8] += generator.uniform(-0.02, 0.02, 8)
        values[8:11] += generator.uniform(-0.0005, 0.0005, 3)
        values[11:14] += generator.uniform(
            -math.radians(0.25), math.radians(0.25), 3
        )
        initial = PointTargetVariables.from_array(
            project_point_target_variables(values, policy, joint_bounds)
        )
    if max_iterations is not None:
        if (
            isinstance(max_iterations, bool)
            or int(max_iterations) != max_iterations
            or max_iterations <= 0
        ):
            raise ValueError("max_iterations must be a positive integer")
        settings = replace(
            settings or PointTargetDLSSettings(),
            maximum_iterations=int(max_iterations),
        )
    result = solve_point_target_dls(
        base,
        signed_orbit_deg=0.0,
        policy=policy,
        initial_variables=initial,
        settings=settings,
        evaluator=evaluator,
        joint_bounds=joint_bounds,
        check_pose_constraints=check_pose_constraints,
    )
    return DownsizeStaticCandidate(
        source_id=str(source_id),
        source_edge_m=mapped_seed.source_edge_m,
        target_edge_m=mapped_seed.target_edge_m,
        mapping_mode=mode,
        seed_mapping=mapped_seed,
        point_plan=plan,
        dls_result=result,
    )


__all__ = [
    "ABSOLUTE_FACE_YZ",
    "DEFAULT_EDGE_GUARD_M",
    "DEFAULT_EDGE_STEP_M",
    "DEFAULT_MAXIMUM_EDGE_M",
    "DEFAULT_MINIMUM_EDGE_M",
    "DEFAULT_SOURCE_EDGE_M",
    "DEFAULT_STABLE_WINDOW_STEPS",
    "DEFAULT_TARGET_RADIUS_M",
    "MAPPING_MODES",
    "PROPORTIONAL_FACE_YZ",
    "DownsizeSeedMapping",
    "DownsizeSourceEvidence",
    "DownsizeStaticCandidate",
    "StableWindowContactEvidence",
    "audit_downsize_source",
    "build_scaled_contact_mapping",
    "build_downsize_contact_point_plan",
    "canonical_mapping_mode",
    "descending_edge_schedule",
    "extract_stable_window_force_weighted_centroids",
    "map_downsize_seed_config",
    "point_target_variables_from_previous_edge",
    "reference_contact_evidence_payload",
    "reference_contact_evidence_sha256",
    "solve_downsize_static_candidate",
]
