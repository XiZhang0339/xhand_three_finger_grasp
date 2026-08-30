"""Deterministic refinement of the index/middle joint-base alignment.

This supplemental v14 path keeps the cube and the eight-finger controller
semantics fixed while applying a coupled wrist rotation/translation around a
measured contact pivot.  It deliberately publishes into a new directory; the
authenticated source artifact remains immutable.
"""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..artifacts import file_sha256, write_json
from ..config import ACTIVE_ACTUATORS, validate_config
from ..experiment import ManipulationPlanParameters
from ..grasp_pose import canonical_sha256
from ..relative_wrist_pose import (
    rotation_matrix_to_rpy_degrees,
    rotvec_degrees_to_rotation_matrix,
)
from ..scene import rpy_degrees_to_rotation_matrix
from .contact_preserving_candidate_artifacts import V14CandidateArtifactBundle


ALIGNMENT_REFINEMENT_SCHEMA_VERSION = 1
JOINT_PAIR = ("left_hand_index_joint1", "left_hand_mid_joint1")
_CLOSE_PROFILE_FINGERS = ("thumb", "index", "mid")
_CLOSE_PROFILE_ACTUATORS = {
    "thumb": ACTIVE_ACTUATORS[:3],
    "index": ACTIVE_ACTUATORS[3:6],
    "mid": ACTIVE_ACTUATORS[6:],
}
_TOP_LEVEL_IDENTITIES = (
    "object_config_id",
    "grasp_pose_id",
    "grasp_object_pair_id",
    "planner_id",
    "controller_id",
    "run_context",
)


def _finite_vector(
    values: Sequence[float], length: int, label: str
) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return tuple(float(value) for value in array)


@dataclass(frozen=True, slots=True)
class JointPairAlignmentAdjustment:
    """One reproducible coupled hand-pose and plan adjustment."""

    wrist_local_rotvec_deg: tuple[float, float, float]
    contact_pivot_world_m: tuple[float, float, float]
    root_delta_cube_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixed_grasp_qpos_offset_rad: tuple[float, ...] = (0.0,) * len(ACTIVE_ACTUATORS)
    precontact_qpos_residual_rad: tuple[float, ...] = (0.0,) * len(ACTIVE_ACTUATORS)
    preload_target_residual_rad: tuple[float, ...] = (0.0,) * len(ACTIVE_ACTUATORS)
    close_profile_start_residual_fraction: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    close_profile_end_residual_fraction: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    close_duration_s: float | None = None
    plan_waypoint_scale: float = 1.0
    source_grasp_window_p95_deg: float = 9.980427
    grasp_window_p95_max_deg: float = 8.10
    operation_p95_max_deg: float = 8.10
    minimum_improvement_deg: float = 1.5

    def __post_init__(self) -> None:
        for name in (
            "wrist_local_rotvec_deg",
            "contact_pivot_world_m",
            "root_delta_cube_m",
        ):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), 3, name))
        object.__setattr__(
            self,
            "fixed_grasp_qpos_offset_rad",
            _finite_vector(
                self.fixed_grasp_qpos_offset_rad,
                len(ACTIVE_ACTUATORS),
                "fixed_grasp_qpos_offset_rad",
            ),
        )
        for name in (
            "precontact_qpos_residual_rad",
            "preload_target_residual_rad",
        ):
            object.__setattr__(
                self,
                name,
                _finite_vector(
                    getattr(self, name),
                    len(ACTIVE_ACTUATORS),
                    name,
                ),
            )
        for name in (
            "close_profile_start_residual_fraction",
            "close_profile_end_residual_fraction",
        ):
            object.__setattr__(
                self,
                name,
                _finite_vector(getattr(self, name), len(_CLOSE_PROFILE_FINGERS), name),
            )
        if self.close_duration_s is not None:
            close_duration = float(self.close_duration_s)
            if not np.isfinite(close_duration) or close_duration <= 0.0:
                raise ValueError("close_duration_s must be positive and finite")
            object.__setattr__(self, "close_duration_s", close_duration)
        scale = float(self.plan_waypoint_scale)
        if not np.isfinite(scale) or not 0.5 <= scale <= 1.10:
            raise ValueError("plan_waypoint_scale must lie within [0.5, 1.10]")
        object.__setattr__(self, "plan_waypoint_scale", scale)
        for name in (
            "source_grasp_window_p95_deg",
            "grasp_window_p95_max_deg",
            "operation_p95_max_deg",
            "minimum_improvement_deg",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
            object.__setattr__(self, name, value)

    def as_mapping(self) -> dict[str, Any]:
        extended_control = bool(
            any(self.precontact_qpos_residual_rad)
            or any(self.preload_target_residual_rad)
            or any(self.close_profile_start_residual_fraction)
            or any(self.close_profile_end_residual_fraction)
            or self.close_duration_s is not None
        )
        result = {
            # Preserve the exact legacy mapping (and therefore its candidate
            # identity) when none of the new controller residuals are used.
            "schema_version": (
                ALIGNMENT_REFINEMENT_SCHEMA_VERSION + 1
                if extended_control
                else ALIGNMENT_REFINEMENT_SCHEMA_VERSION
            ),
            "joint_names": list(JOINT_PAIR),
            "wrist_local_rotvec_deg": list(self.wrist_local_rotvec_deg),
            "contact_pivot_world_m": list(self.contact_pivot_world_m),
            "root_delta_cube_m": list(self.root_delta_cube_m),
            "fixed_grasp_qpos_offset_rad": list(
                self.fixed_grasp_qpos_offset_rad
            ),
            "plan_waypoint_scale": self.plan_waypoint_scale,
            "source_grasp_window_p95_deg": self.source_grasp_window_p95_deg,
            "grasp_window_p95_max_deg": self.grasp_window_p95_max_deg,
            "operation_p95_max_deg": self.operation_p95_max_deg,
            "minimum_improvement_deg": self.minimum_improvement_deg,
        }
        if extended_control:
            result.update(
                {
                    "precontact_qpos_residual_rad": list(
                        self.precontact_qpos_residual_rad
                    ),
                    "preload_target_residual_rad": list(
                        self.preload_target_residual_rad
                    ),
                    "close_profile_start_residual_fraction": list(
                        self.close_profile_start_residual_fraction
                    ),
                    "close_profile_end_residual_fraction": list(
                        self.close_profile_end_residual_fraction
                    ),
                    "close_duration_s": self.close_duration_s,
                }
            )
        return result


def _scaled_plan(config: Mapping[str, Any], scale: float) -> ManipulationPlanParameters:
    source = ManipulationPlanParameters.from_config(config)
    return ManipulationPlanParameters(
        schema_version=source.schema_version,
        profile=source.profile,
        duration_s=source.duration_s,
        knot_times_s=source.knot_times_s,
        actuator_waypoints_rad={
            name: tuple(scale * value for value in source.actuator_waypoints_rad[name])
            for name in ACTIVE_ACTUATORS
        },
        desired_cube_position_delta_m=source.desired_cube_position_delta_m,
        desired_cube_rotation_vector_rad=source.desired_cube_rotation_vector_rad,
        max_knot_delta_rad=source.max_knot_delta_rad,
        trust_region_backtracks=source.trust_region_backtracks,
    )


def build_joint_pair_alignment_config(
    source_config: Mapping[str, Any],
    adjustment: JointPairAlignmentAdjustment,
    *,
    source_config_sha256: str | None = None,
    source_trace_sha256: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Resolve an immutable cube-preserving v14 alignment candidate."""

    config = copy.deepcopy(dict(source_config))
    fixed_cube_sha = canonical_sha256(config["cube"])
    source_semantic_sha = canonical_sha256(config)
    for key in _TOP_LEVEL_IDENTITIES:
        config.pop(key, None)

    base_translation = np.asarray(config["hand_pose"]["translation_m"], dtype=float)
    base_rotation = rpy_degrees_to_rotation_matrix(config["hand_pose"]["rpy_deg"])
    proposed_rotation = base_rotation @ rotvec_degrees_to_rotation_matrix(
        adjustment.wrist_local_rotvec_deg
    )
    pivot = np.asarray(adjustment.contact_pivot_world_m, dtype=float)
    cube_rotation = rpy_degrees_to_rotation_matrix(config["cube"]["rpy_deg"])
    proposed_translation = (
        pivot
        - proposed_rotation @ base_rotation.T @ (pivot - base_translation)
        + cube_rotation @ np.asarray(adjustment.root_delta_cube_m, dtype=float)
    )
    config["hand_pose"] = {
        "translation_m": proposed_translation.tolist(),
        "rpy_deg": rotation_matrix_to_rpy_degrees(
            proposed_rotation,
            reference_rpy_deg=config["hand_pose"]["rpy_deg"],
        ).tolist(),
    }

    for index, name in enumerate(ACTIVE_ACTUATORS):
        offset = float(adjustment.fixed_grasp_qpos_offset_rad[index])
        config["grasp_pose"]["nominal_joint_qpos_rad"][name] += offset
        config["control"]["precontact_targets_rad"][name] += (
            offset + float(adjustment.precontact_qpos_residual_rad[index])
        )
        config["control"]["contact_preload_targets_rad"][name] += (
            offset + float(adjustment.preload_target_residual_rad[index])
        )

    close_profile = config["control"]["close_profile"]
    for finger_index, finger in enumerate(_CLOSE_PROFILE_FINGERS):
        start_residual = float(
            adjustment.close_profile_start_residual_fraction[finger_index]
        )
        end_residual = float(
            adjustment.close_profile_end_residual_fraction[finger_index]
        )
        for name in _CLOSE_PROFILE_ACTUATORS[finger]:
            interval = close_profile[name]
            interval["start_fraction"] = (
                float(interval["start_fraction"]) + start_residual
            )
            interval["end_fraction"] = (
                float(interval["end_fraction"]) + end_residual
            )
    if adjustment.close_duration_s is not None:
        config["control_protocol"]["close_s"] = adjustment.close_duration_s

    plan = _scaled_plan(
        config["manipulation_plan"], adjustment.plan_waypoint_scale
    )
    config["manipulation_plan"] = plan.as_config()
    config["control"]["manipulation_delta_rad"] = {
        name: float(plan.actuator_waypoints_rad[name][-1])
        for name in ACTIVE_ACTUATORS
    }

    identity_payload = {
        "schema_version": ALIGNMENT_REFINEMENT_SCHEMA_VERSION,
        "kind": "v14_index_middle_joint_pair_alignment_refinement",
        "source_config_semantic_sha256": source_semantic_sha,
        "source_config_file_sha256": source_config_sha256,
        "source_trace_sha256": source_trace_sha256,
        "adjustment": adjustment.as_mapping(),
    }
    digest = canonical_sha256(identity_payload)
    candidate_id = 15_900_000_000_000_000 + int(digest[:13], 16) % 100_000_000_000_000
    config["candidate_metadata"] = {
        "schema_version": 14,
        "candidate_id": candidate_id,
        "cube_pose_sampled": False,
        "hand_root_fixed_during_simulation": True,
        "v14_index_middle_joint_pair_alignment_refinement": {
            **identity_payload,
            "candidate_id": candidate_id,
            "candidate_sha256": digest,
            "joint_names": list(JOINT_PAIR),
            "source_grasp_window_p95_deg": adjustment.source_grasp_window_p95_deg,
            "grasp_window_p95_max_deg": adjustment.grasp_window_p95_max_deg,
            "operation_p95_max_deg": adjustment.operation_p95_max_deg,
            "minimum_improvement_deg": adjustment.minimum_improvement_deg,
        },
    }
    if canonical_sha256(config["cube"]) != fixed_cube_sha:
        raise AssertionError("joint-pair refinement changed the cube")
    validate_config(config)
    return candidate_id, config


def publish_joint_pair_alignment_catalog(
    bundle: V14CandidateArtifactBundle,
    output_root: str | Path,
) -> Path:
    """Publish one authenticated final rerun as a Viewer-selectable catalog."""

    if bundle.trace_path is None:
        raise RuntimeError("joint-pair alignment catalog requires a retained trace")
    root = Path(output_root).expanduser().resolve()
    catalog_root = root / "catalogs" / "target_1" / "manipulation"
    trajectory_id = f"pair_rank_01_{bundle.candidate_id}"
    member = catalog_root / trajectory_id
    member.mkdir(parents=True, exist_ok=True)
    destinations = {
        "resolved_config": member / "resolved_config.json",
        "result": member / "result.json",
        "trace": member / "trace.npz",
    }
    for source, destination in (
        (bundle.config_path, destinations["resolved_config"]),
        (bundle.result_path, destinations["result"]),
        (bundle.trace_path, destinations["trace"]),
    ):
        if destination.exists() and file_sha256(destination) != file_sha256(source):
            raise RuntimeError(f"refusing to overwrite changed catalog member {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
    summary = bundle.result["summary"]
    near_zero_grasp = bool(
        bundle.result.get("experiment_id")
        == (
            "left_opposed_face_palm_down_joint_pair_aligned_"
            "contact_preserving_planned_lift"
        )
    )
    aliases = ["best_attempt", "best_aligned_grasp"]
    if near_zero_grasp:
        aliases.append("best_near_zero_grasp")
    resolved_config = json.loads(bundle.config_path.read_text(encoding="utf-8"))
    entry = {
        "trajectory_id": trajectory_id,
        "candidate_id": str(bundle.candidate_id),
        "aliases": aliases,
        "classification": bundle.result["classification"],
        "grasp_success": bool(bundle.result["grasp_success"]),
        "full_success": bool(bundle.result["full_success"]),
        "edge_m": float(resolved_config["cube"]["edge_m"]),
        "alignment_metrics": summary.get("metrics", {}).get(
            "index_middle_joint_pair_alignment", {}
        ),
        "artifacts": {
            "resolved_config": f"{trajectory_id}/resolved_config.json",
            "result": f"{trajectory_id}/result.json",
            "trace": f"{trajectory_id}/trace.npz",
            "video": None,
            "sha256": {
                name: file_sha256(path) for name, path in destinations.items()
            },
        },
    }
    payload = {
        "trajectory_catalog_schema_version": 1,
        "contact_preserving_viewer_catalog_schema_version": 1,
        "complete": True,
        "experiment_id": bundle.result["experiment_id"],
        "catalog_kind": "manipulation",
        "selection_policy": (
            "grasp_then_near_zero_joint_pair_alignment_then_manipulation_diagnostic"
            if near_zero_grasp
            else "grasp_then_joint_pair_alignment_then_contact_and_lift"
        ),
        "success_count": int(bundle.result["full_success"]),
        "aliases": {
            "best_attempt": trajectory_id,
            "best_aligned_grasp": trajectory_id,
            **(
                {"best_near_zero_grasp": trajectory_id}
                if near_zero_grasp
                else {}
            ),
        },
        "trajectories": [entry],
    }
    catalog_path = catalog_root / "catalog.json"
    if catalog_path.exists():
        existing = json.loads(catalog_path.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != canonical_sha256(payload):
            raise RuntimeError("refusing to overwrite a changed alignment catalog")
    else:
        write_json(catalog_path, payload)
    return catalog_path


__all__ = [
    "ALIGNMENT_REFINEMENT_SCHEMA_VERSION",
    "JOINT_PAIR",
    "JointPairAlignmentAdjustment",
    "build_joint_pair_alignment_config",
    "publish_joint_pair_alignment_catalog",
]
