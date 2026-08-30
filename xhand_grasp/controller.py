"""Closed-loop control policy for feedback-gated grasp/manipulation tasks.

The policy is intentionally independent of MuJoCo state objects.  The generic
simulation loop supplies one post-step observation at a time, which makes the
state machine and its persisted evidence straightforward to unit test.  A
command state names the target that produced the sample at the same trace
index.  Consequently, a grasp acquired on sample ``t`` can first issue a
manipulation command on sample ``t + 1``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

import mujoco
import numpy as np

from .config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    contact_preload_targets,
    precontact_targets,
)
from .contacts import FACE_ORDER, Face
from .trajectory import (
    actuator_target_vector,
    interpolate_quintic_c2,
    minimum_jerk,
    quintic_c2_knot_derivatives,
    smoothstep,
)


class ControlState(str, Enum):
    """Persisted schema-v3 command states."""

    SETTLE = "SETTLE"
    CLOSE = "CLOSE"
    VERIFY = "VERIFY"
    MANIPULATE = "MANIPULATE"
    HOLD = "HOLD"
    ABORT = "ABORT"


# The order is a persisted axis.  Append-only changes require a new trace
# schema; reordering would silently change the meaning of existing NPZ files.
GRASP_GATE_ORDER = (
    "thumb_target_face_effective",
    "index_target_face_effective",
    "mid_target_face_effective",
    "target_face_topology",
    "no_off_target_contact",
    "no_active_nondistal_contact",
    "no_forbidden_contact",
    "support_contact",
    "no_floor_contact",
    "cube_translation_stable",
    "cube_orientation_stable",
    "cube_linear_speed_low",
    "no_early_lift",
    "palm_down",
    "penetration_within_limit",
    "finite",
    "joint_limits_respected",
    "inactive_controls_zero",
)

# Schema-v4 makes contact-height alignment part of the same continuous grasp
# verification window.  Keep the v3 axis byte-for-byte stable: persisted v3
# traces remain independently decodable and no missing v4 evidence can be
# mistaken for a passing sample.
V4_GRASP_GATE_ORDER = GRASP_GATE_ORDER + ("contact_height_aligned",)

# Schema-v6 makes preservation of the *original*, pre-integration cube pose a
# sticky grasp prerequisite.  The history bit is deliberately appended after
# the v4/v5 axis so every older persisted trace keeps exactly the same meaning.
V6_GRASP_GATE_ORDER = V4_GRASP_GATE_ORDER + (
    "initial_pose_history_stable",
)

# Schema-v9 makes the measured thumb angle (not its servo command) an online
# prerequisite.  Median nominal-pose error and p95-p5 stability are window
# properties and are checked by the controller over the same 250 ms samples;
# they cannot truthfully be represented by a per-frame gate component.
V9_GRASP_GATE_ORDER = V6_GRASP_GATE_ORDER + (
    "thumb_actual_qpos_within_range",
)

# Schema-v12 freezes one cube-local target region for each active fingertip.
# Appending the aggregate point-region predicate keeps every v3-v11 gate axis
# byte-for-byte stable while making all three effective contacts part of the
# same continuous 250 ms verification window.
V12_GRASP_GATE_ORDER = V9_GRASP_GATE_ORDER + (
    "contact_points_within_target_regions",
)

# Schema-v15 makes the directed index-joint1 -> middle-joint1 line part of
# the same continuous grasp-verification evidence.  The aggregate percentile
# requirement is checked by the controller over the completed window; this
# per-frame bit enforces direction, minimum length and the hard 0.5 degree
# grasp-window maximum without changing any persisted v3-v14 axis.
V15_GRASP_GATE_ORDER = V12_GRASP_GATE_ORDER + (
    "joint_pair_alignment_safe",
    "no_active_finger_self_collision",
)


def grasp_gate_order(schema_version: int) -> tuple[str, ...]:
    """Return the canonical persisted gate axis for a trace schema."""

    version = int(schema_version)
    if version == 3:
        return GRASP_GATE_ORDER
    if 4 <= version <= 5:
        return V4_GRASP_GATE_ORDER
    if 6 <= version <= 8:
        return V6_GRASP_GATE_ORDER
    if 9 <= version <= 11:
        return V9_GRASP_GATE_ORDER
    if 12 <= version <= 14:
        return V12_GRASP_GATE_ORDER
    if version >= 15:
        return V15_GRASP_GATE_ORDER
    raise ValueError("feedback grasp gates are defined only for schema v3+")

_FACE_BY_LABEL = {
    "+X": Face.X_POS,
    "-X": Face.X_NEG,
    "+Y": Face.Y_POS,
    "-Y": Face.Y_NEG,
    "+Z": Face.Z_POS,
    "-Z": Face.Z_NEG,
}


@dataclass(frozen=True)
class TargetFaceEvidence:
    """Per-finger force evidence used both online and in trace evaluation."""

    target_force_n: np.ndarray
    total_distal_force_n: np.ndarray
    target_force_purity: np.ndarray
    target_face_effective: np.ndarray
    material_off_target: np.ndarray
    material_active_nondistal: np.ndarray


@dataclass(frozen=True)
class GraspGateEvidence:
    """All evidence evaluated for one post-step sample."""

    components: np.ndarray
    target_faces: TargetFaceEvidence
    hard_abort: bool
    gate_order: tuple[str, ...] = GRASP_GATE_ORDER

    def __post_init__(self) -> None:
        components = np.asarray(self.components, dtype=bool)
        if components.shape != (len(self.gate_order),):
            raise ValueError(
                "grasp gate components must match the persisted gate order"
            )
        object.__setattr__(self, "components", components.copy())

    @property
    def passed(self) -> bool:
        return bool(np.all(self.components))

    def as_mapping(self) -> dict[str, bool]:
        return {
            name: bool(self.components[index])
            for index, name in enumerate(self.gate_order)
        }


@dataclass(frozen=True)
class InitialPoseHistoryEvidence:
    """Sticky displacement evidence relative to a pre-step cube pose.

    ``current_*`` describes the supplied sample, while ``maximum_*`` and the
    two history predicates include every sample observed since the initial pose
    was explicitly latched.  Returning to the initial pose therefore cannot
    erase an earlier displacement violation.
    """

    current_translation_m: float
    current_orientation_drift_deg: float
    maximum_translation_m: float
    maximum_orientation_drift_deg: float
    translation_history_stable: bool
    orientation_history_stable: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.translation_history_stable
            and self.orientation_history_stable
        )


@dataclass(frozen=True)
class ControlCommand:
    """One command emitted before a MuJoCo integration step."""

    state: ControlState
    target: np.ndarray
    manipulation_progress: float
    target_velocity_rad_s: np.ndarray


@dataclass(frozen=True)
class OperationFeedback:
    """Post-step operation evidence consumed on the *next* control sample.

    The simulation loop deliberately constructs this value only after
    ``mj_step``/``mj_forward``.  Keeping it independent of ``MjData`` makes
    the one-sample delay explicit and lets controller tests prove that no
    same-step contact information leaks into a command.
    """

    target_force_n: np.ndarray
    target_force_purity: np.ndarray
    target_face_effective: np.ndarray
    material_off_target: np.ndarray
    material_active_nondistal: np.ndarray
    tactile_force_n: np.ndarray
    contact_centroid_world_m: np.ndarray
    contact_centroid_valid: np.ndarray
    cube_position_m: np.ndarray
    cube_quaternion_wxyz: np.ndarray
    cube_velocity: np.ndarray
    joint_qpos_rad: np.ndarray
    joint_qvel_rad_s: np.ndarray
    forbidden_contact: bool
    max_penetration_m: float
    finite: bool
    joint_limits_respected: bool
    inactive_controls_zero: bool
    # Schema-v14 feedback schema 2 adds a cube-local, grasp-relative
    # tangential slip observation.  Defaults keep hand-written schema-1
    # controller fixtures and archived controller inputs source compatible.
    contact_centroid_cube_local_m: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )
    tangent_slip_from_grasp_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    tangent_slip_valid: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=bool)
    )
    # Schema-v15 adds a directed joint-pair observation in the cube frame.
    # Invalid defaults keep every schema-v14 fixture and call site source- and
    # numerically-compatible while making accidental use fail safe.
    joint_pair_vector_cube_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    joint_pair_residual: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    joint_pair_angle_deg: float = 180.0
    joint_pair_length_m: float = 0.0
    joint_pair_positive_y: bool = False
    joint_pair_valid: bool = False
    active_finger_self_collision: bool = False
    # Schema-v16 replaces the force-centroid displacement proxy with a causal
    # material-point velocity integral.  These defaults keep every v14/v15
    # caller and archived controller fixture source- and numerically-compatible.
    rolling_signed_tangent_displacement_m: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 2), dtype=np.float64)
    )
    rolling_cumulative_irrecoverable_slip_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    rolling_relative_tangent_velocity_m_s: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 2), dtype=np.float64)
    )
    rolling_contact_valid: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=bool)
    )
    rolling_contact_continuous: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=bool)
    )
    rolling_detected: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=bool)
    )
    rolling_patch_switch: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=bool)
    )
    rolling_tangent_jacobian_m_per_rad: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (3, 2, len(ACTIVE_ACTUATORS)), dtype=np.float64
        )
    )
    rolling_tangent_jacobian_valid: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=bool)
    )

    def __post_init__(self) -> None:
        shapes = {
            "target_force_n": (3,),
            "target_force_purity": (3,),
            "target_face_effective": (3,),
            "material_off_target": (3,),
            "material_active_nondistal": (3,),
            "tactile_force_n": (3,),
            "contact_centroid_world_m": (3, 3),
            "contact_centroid_valid": (3,),
            "contact_centroid_cube_local_m": (3, 3),
            "tangent_slip_from_grasp_m": (3,),
            "tangent_slip_valid": (3,),
            "joint_pair_vector_cube_m": (3,),
            "joint_pair_residual": (2,),
            "rolling_signed_tangent_displacement_m": (3, 2),
            "rolling_cumulative_irrecoverable_slip_m": (3,),
            "rolling_relative_tangent_velocity_m_s": (3, 2),
            "rolling_contact_valid": (3,),
            "rolling_contact_continuous": (3,),
            "rolling_detected": (3,),
            "rolling_patch_switch": (3,),
            "rolling_tangent_jacobian_m_per_rad": (
                3,
                2,
                len(ACTIVE_ACTUATORS),
            ),
            "rolling_tangent_jacobian_valid": (3,),
            "cube_position_m": (3,),
            "cube_quaternion_wxyz": (4,),
            "cube_velocity": (6,),
        }
        boolean_names = {
            "target_face_effective",
            "material_off_target",
            "material_active_nondistal",
            "contact_centroid_valid",
            "tangent_slip_valid",
            "rolling_contact_valid",
            "rolling_contact_continuous",
            "rolling_detected",
            "rolling_patch_switch",
            "rolling_tangent_jacobian_valid",
        }
        for name, shape in shapes.items():
            dtype = bool if name in boolean_names else np.float64
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if dtype is not bool and not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value.copy())
        for name in ("joint_qpos_rad", "joint_qvel_rad_s"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite vector")
            object.__setattr__(self, name, value.copy())
        if np.any(self.target_force_n < 0.0) or np.any(
            self.target_force_purity < 0.0
        ):
            raise ValueError("operation contact force evidence must be non-negative")
        if np.any(self.tangent_slip_from_grasp_m < 0.0):
            raise ValueError("operation tangent slip evidence must be non-negative")
        if np.any(self.rolling_cumulative_irrecoverable_slip_m < 0.0):
            raise ValueError(
                "rolling cumulative slip evidence must be non-negative"
            )
        if not math.isfinite(float(self.max_penetration_m)):
            raise ValueError("max_penetration_m must be finite")
        if not math.isfinite(float(self.joint_pair_angle_deg)):
            raise ValueError("joint_pair_angle_deg must be finite")
        if (
            not math.isfinite(float(self.joint_pair_length_m))
            or float(self.joint_pair_length_m) < 0.0
        ):
            raise ValueError("joint_pair_length_m must be finite and non-negative")


def merge_rolling_operation_target_face_effective(
    value: OperationFeedback,
    *,
    minimum_force_n: float,
    minimum_purity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return strict rolling evidence and its OR with legacy effectiveness.

    Native tactile evidence remains authoritative during grasp verification.
    This helper is only used by the schema-v16 controller after grasp lock,
    where a distal fingertip is allowed to roll between tactile taxels.  Such
    a sample can replace a missing legacy/native-touch bit only when the real
    rolling contact is on the target face, forceful and pure, and no forbidden
    or non-distal material contact is present.
    """

    force_threshold = float(minimum_force_n)
    purity_threshold = float(minimum_purity)
    if not math.isfinite(force_threshold) or force_threshold <= 0.0:
        raise ValueError("minimum_force_n must be finite and positive")
    if (
        not math.isfinite(purity_threshold)
        or purity_threshold <= 0.0
        or purity_threshold > 1.0
    ):
        raise ValueError("minimum_purity must lie in (0, 1]")
    safe_material = (
        ~np.asarray(value.material_off_target, dtype=bool)
        & ~np.asarray(value.material_active_nondistal, dtype=bool)
    )
    rolling_effective = (
        np.asarray(value.rolling_contact_valid, dtype=bool)
        & (
            np.asarray(value.target_force_n, dtype=np.float64)
            >= force_threshold - 1e-12
        )
        & (
            np.asarray(value.target_force_purity, dtype=np.float64)
            >= purity_threshold - 1e-12
        )
        & safe_material
        & (not bool(value.forbidden_contact))
    )
    legacy = np.asarray(value.target_face_effective, dtype=bool)
    return rolling_effective.copy(), (legacy | rolling_effective).copy()


def _unit_quaternion(value: np.ndarray, label: str) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{label} must be a finite non-zero quaternion")
    return quaternion / norm


def quaternion_drift_deg(reference: np.ndarray, value: np.ndarray) -> float:
    """Return the shortest orientation difference in degrees."""

    reference_unit = _unit_quaternion(reference, "reference quaternion")
    value_unit = _unit_quaternion(value, "cube quaternion")
    cosine = float(np.clip(abs(np.dot(reference_unit, value_unit)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(cosine)))


def compute_target_face_evidence(
    config: Mapping[str, Any],
    distal_face_force_n: np.ndarray,
    active_nondistal_force_n: np.ndarray,
    tactile_force_n: np.ndarray,
) -> TargetFaceEvidence:
    """Compute the exact online target-face force predicates.

    This helper is shared with evaluators so the gate and persisted NPZ
    evidence cannot disagree about force thresholds, tactile requirements, or
    the treatment of edge/corner force.
    """

    face_force = np.asarray(distal_face_force_n, dtype=np.float64)
    nondistal_force = np.asarray(active_nondistal_force_n, dtype=np.float64)
    tactile = np.asarray(tactile_force_n, dtype=np.float64)
    if face_force.shape != (3, len(FACE_ORDER)):
        raise ValueError(
            "distal_face_force_n must have shape "
            f"(3, {len(FACE_ORDER)})"
        )
    if nondistal_force.shape != (3,):
        raise ValueError("active_nondistal_force_n must have shape (3,)")
    if tactile.shape != (3,):
        raise ValueError("tactile_force_n must have shape (3,)")
    if (
        not np.isfinite(face_force).all()
        or not np.isfinite(nondistal_force).all()
        or not np.isfinite(tactile).all()
        or np.any(face_force < 0.0)
        or np.any(nondistal_force < 0.0)
    ):
        raise ValueError("contact and tactile evidence must be finite and non-negative")

    topology = config["contact_topology"]
    gate = config["control_protocol"]["grasp_gate"]
    target_indices = np.asarray(
        [
            FACE_ORDER.index(_FACE_BY_LABEL[str(topology["target_faces"][finger])])
            for finger in ACTIVE_FINGERS
        ],
        dtype=int,
    )
    target_force = face_force[np.arange(3), target_indices]
    total_distal = np.sum(face_force, axis=1)
    purity = np.divide(
        target_force,
        total_distal,
        out=np.zeros_like(target_force),
        where=total_distal > 0.0,
    )

    force_min = float(gate["min_target_face_force_n"])
    purity_min = float(gate["min_target_force_fraction"])
    touch_min = float(config["acceptance"]["touch_force_min_n"])
    require_touch = bool(gate["require_touch"])
    effective = (target_force >= force_min) & (purity >= purity_min)
    if require_touch:
        effective &= tactile >= touch_min

    off_target = np.maximum(0.0, total_distal - target_force)
    off_fraction = np.divide(
        off_target,
        total_distal,
        out=np.zeros_like(off_target),
        where=total_distal > 0.0,
    )
    # The versioned gate owns the 95% threshold.  Off-target force becomes
    # material at the complementary fraction, and edge/corner bins are never
    # split or credited to a physical target face.
    max_off_fraction = 1.0 - purity_min
    material_off = (off_target >= force_min) & (
        off_fraction > max_off_fraction + 1e-12
    )
    combined = total_distal + nondistal_force
    nondistal_fraction = np.divide(
        nondistal_force,
        combined,
        out=np.zeros_like(nondistal_force),
        where=combined > 0.0,
    )
    material_nondistal = (nondistal_force >= force_min) & (
        nondistal_fraction > max_off_fraction + 1e-12
    )

    return TargetFaceEvidence(
        target_force_n=target_force,
        total_distal_force_n=total_distal,
        target_force_purity=purity,
        target_face_effective=effective,
        material_off_target=material_off,
        material_active_nondistal=material_nondistal,
    )


def compute_grasp_gate_evidence(
    config: Mapping[str, Any],
    *,
    distal_face_force_n: np.ndarray,
    active_nondistal_force_n: np.ndarray,
    tactile_force_n: np.ndarray,
    forbidden_contact: bool,
    support_contact: bool,
    floor_contact: bool,
    cube_position: np.ndarray,
    cube_quaternion: np.ndarray,
    reference_position: np.ndarray,
    reference_quaternion: np.ndarray,
    cube_linear_speed_m_s: float,
    early_lift_m: float,
    palm_down_angle_deg: float,
    max_penetration_m: float,
    finite: bool,
    joint_limits_respected: bool,
    inactive_controls_zero: bool,
    contact_height_aligned: bool | None = None,
    initial_pose_history_stable: bool | None = None,
    thumb_actual_qpos_within_range: bool | None = None,
    contact_points_within_target_regions: bool | None = None,
    joint_pair_alignment_safe: bool | None = None,
    active_finger_self_collision: bool | None = None,
) -> GraspGateEvidence:
    """Evaluate one complete schema-v3+ grasp-verification gate."""

    target = compute_target_face_evidence(
        config,
        distal_face_force_n,
        active_nondistal_force_n,
        tactile_force_n,
    )
    gate = config["control_protocol"]["grasp_gate"]
    acceptance = config["acceptance"]
    cube_position = np.asarray(cube_position, dtype=np.float64).reshape(3)
    reference_position = np.asarray(reference_position, dtype=np.float64).reshape(3)
    translation = float(np.linalg.norm(cube_position - reference_position))
    orientation = quaternion_drift_deg(reference_quaternion, cube_quaternion)

    components_by_name = {
        "thumb_target_face_effective": target.target_face_effective[0],
        "index_target_face_effective": target.target_face_effective[1],
        "mid_target_face_effective": target.target_face_effective[2],
        "target_face_topology": np.all(target.target_face_effective),
        "no_off_target_contact": not np.any(target.material_off_target),
        "no_active_nondistal_contact": not np.any(
            target.material_active_nondistal
        ),
        "no_forbidden_contact": not bool(forbidden_contact),
        "support_contact": (
            bool(support_contact) if bool(gate["require_support_contact"]) else True
        ),
        "no_floor_contact": not bool(floor_contact),
        "cube_translation_stable": translation
        <= float(gate["max_translation_m"]) + 1e-12,
        "cube_orientation_stable": orientation
        <= float(gate["max_orientation_drift_deg"]) + 1e-12,
        "cube_linear_speed_low": float(cube_linear_speed_m_s)
        <= float(gate["max_linear_speed_m_s"]) + 1e-12,
        "no_early_lift": float(early_lift_m)
        <= float(gate["max_early_lift_m"]) + 1e-12,
        "palm_down": float(palm_down_angle_deg)
        <= float(acceptance["max_palm_down_angle_deg"]) + 1e-12,
        "penetration_within_limit": float(max_penetration_m)
        <= float(acceptance["max_penetration_m"]) + 1e-12,
        "finite": bool(finite),
        "joint_limits_respected": bool(joint_limits_respected),
        "inactive_controls_zero": bool(inactive_controls_zero),
    }
    schema_version = int(config.get("schema_version", 3))
    order = grasp_gate_order(schema_version)
    if schema_version >= 4:
        if contact_height_aligned is None:
            raise ValueError(
                f"schema v{schema_version} grasp evidence requires "
                "contact_height_aligned"
            )
        components_by_name["contact_height_aligned"] = bool(
            contact_height_aligned
        )
    if schema_version >= 6:
        if initial_pose_history_stable is None:
            raise ValueError(
                f"schema v{schema_version} grasp evidence requires "
                "initial_pose_history_stable"
            )
        components_by_name["initial_pose_history_stable"] = bool(
            initial_pose_history_stable
        )
    if schema_version >= 9:
        if thumb_actual_qpos_within_range is None:
            raise ValueError(
                "schema v9 grasp evidence requires "
                "thumb_actual_qpos_within_range"
            )
        components_by_name["thumb_actual_qpos_within_range"] = bool(
            thumb_actual_qpos_within_range
        )
    if schema_version >= 12:
        if contact_points_within_target_regions is None:
            raise ValueError(
                "schema v12 grasp evidence requires "
                "contact_points_within_target_regions"
            )
        components_by_name["contact_points_within_target_regions"] = bool(
            contact_points_within_target_regions
        )
    if schema_version >= 15:
        if joint_pair_alignment_safe is None:
            raise ValueError(
                "schema v15 grasp evidence requires joint_pair_alignment_safe"
            )
        components_by_name["joint_pair_alignment_safe"] = bool(
            joint_pair_alignment_safe
        )
        if active_finger_self_collision is None:
            raise ValueError(
                "schema v15 grasp evidence requires active-finger self-collision evidence"
            )
        components_by_name["no_active_finger_self_collision"] = not bool(
            active_finger_self_collision
        )
    components = np.asarray(
        [bool(components_by_name[name]) for name in order], dtype=bool
    )
    hard_abort_names = (
        "no_forbidden_contact",
        "palm_down",
        "penetration_within_limit",
        "finite",
        "joint_limits_respected",
        "inactive_controls_zero",
    )
    hard_abort = not all(bool(components_by_name[name]) for name in hard_abort_names)
    return GraspGateEvidence(
        components=components,
        target_faces=target,
        hard_abort=hard_abort,
        gate_order=order,
    )


class GraspVerifyThenManipulateController:
    """Deterministic feedback-gated schema-v3+ command policy.

    Schema v6 retains the existing five-phase state machine.  It can declare a
    collision-free pregrasp target as the active-hand initial condition and
    hold it throughout ``SETTLE``; legacy configurations can still ramp from
    zero.  During ``CLOSE``, each active actuator follows its own delayed
    smoothstep interval from the pregrasp target to the grasp target.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        config: Mapping[str, Any],
        phase_steps: Mapping[str, int],
    ) -> None:
        self.schema_version = int(config.get("schema_version", 3))
        protocol = config["control_protocol"]
        if protocol["strategy"] != "grasp_verify_then_manipulate":
            raise ValueError("unsupported schema-v3 control protocol strategy")
        if protocol["failure_behavior"] != "abort_hold_grasp_pose":
            raise ValueError("unsupported schema-v3 failure behavior")

        expected_phases = {"settle", "close", "verify", "manipulate", "hold"}
        if set(phase_steps) != expected_phases:
            raise ValueError(
                "schema-v3 phase steps must contain settle, close, verify, "
                "manipulate and hold"
            )
        self.phase_steps = {name: int(phase_steps[name]) for name in expected_phases}
        if any(value <= 0 for value in self.phase_steps.values()):
            raise ValueError("all schema-v3 phases must contain at least one step")

        self.model = model
        self.manipulation_profile = protocol.get(
            "manipulation_profile", "cubic_smoothstep"
        )
        self.total_steps = sum(self.phase_steps.values())
        self.settle_end = self.phase_steps["settle"]
        self.close_end = self.settle_end + self.phase_steps["close"]
        self.verify_end = self.close_end + self.phase_steps["verify"]
        stable_seconds = float(protocol["stable_window_s"])
        self.required_stable_steps = int(round(stable_seconds / model.opt.timestep))
        if (
            self.required_stable_steps <= 0
            or abs(self.required_stable_steps * float(model.opt.timestep) - stable_seconds)
            > 0.5 * float(model.opt.timestep) + 1e-12
        ):
            raise ValueError("control_protocol.stable_window_s does not align with timestep")
        if self.required_stable_steps > self.phase_steps["verify"]:
            raise ValueError("stable_window_s must not exceed verify_timeout_s")

        control = config["control"]
        preload_mapping = contact_preload_targets(dict(config))
        self.grasp_target = actuator_target_vector(
            model, preload_mapping
        )
        delta_mapping = control["manipulation_delta_rad"]
        if set(delta_mapping) != set(ACTIVE_ACTUATORS):
            raise ValueError(
                "manipulation_delta_rad must contain exactly the active actuators"
            )
        manipulation_mapping = {
            name: float(preload_mapping[name])
            + float(delta_mapping[name])
            for name in ACTIVE_ACTUATORS
        }
        self.manipulation_target = actuator_target_vector(model, manipulation_mapping)
        self.zero_target = np.zeros(model.nu, dtype=np.float64)

        self.pregrasp_target = self.zero_target.copy()
        self.close_start_fraction = np.zeros(model.nu, dtype=np.float64)
        self.close_end_fraction = np.ones(model.nu, dtype=np.float64)
        self.initialize_active_joints_at_pregrasp = False
        if self.schema_version >= 6:
            self.pregrasp_target = actuator_target_vector(
                model, precontact_targets(dict(config))
            )
            self.initialize_active_joints_at_pregrasp = bool(
                config["pose_preservation"][
                    "initialize_active_joints_at_pregrasp"
                ]
            )
            close_profile = control["close_profile"]
            if set(close_profile) != set(ACTIVE_ACTUATORS):
                raise ValueError(
                    "close_profile must contain exactly the active actuators"
                )
            for name in ACTIVE_ACTUATORS:
                profile = close_profile[name]
                if not isinstance(profile, Mapping) or set(profile) != {
                    "start_fraction",
                    "end_fraction",
                }:
                    raise ValueError(
                        f"close_profile[{name!r}] must contain exactly "
                        "start_fraction and end_fraction"
                    )
                start = float(profile["start_fraction"])
                end = float(profile["end_fraction"])
                if (
                    not math.isfinite(start)
                    or not math.isfinite(end)
                    or not 0.0 <= start < end <= 1.0
                ):
                    raise ValueError(
                        f"close_profile[{name!r}] fractions must satisfy "
                        "0 <= start_fraction < end_fraction <= 1"
                    )
                actuator_id = model.actuator(name).id
                self.close_start_fraction[actuator_id] = start
                self.close_end_fraction[actuator_id] = end

        self.max_initial_translation_m = math.inf
        self.max_initial_orientation_drift_deg = math.inf
        if self.schema_version >= 6:
            pose_preservation = config["pose_preservation"]
            self.max_initial_translation_m = float(
                pose_preservation["max_translation_m"]
            )
            self.max_initial_orientation_drift_deg = float(
                pose_preservation["max_orientation_drift_deg"]
            )
            if (
                not math.isfinite(self.max_initial_translation_m)
                or self.max_initial_translation_m < 0.0
                or not math.isfinite(self.max_initial_orientation_drift_deg)
                or self.max_initial_orientation_drift_deg < 0.0
            ):
                raise ValueError(
                    "initial pose history limits must be finite and non-negative"
                )
        self._initial_pose_position: np.ndarray | None = None
        self._initial_pose_quaternion: np.ndarray | None = None
        self._initial_pose_max_translation_m = 0.0
        self._initial_pose_max_orientation_drift_deg = 0.0
        self._last_sent_target = self.zero_target.copy()
        self._abort_hold_target: np.ndarray | None = None

        self._next_step = 0
        self._last_command_state: ControlState | None = None
        self._consecutive_steps = 0
        self._acquired = False
        self._aborted = False
        self._stability_reference_position: np.ndarray | None = None
        self._stability_reference_quaternion: np.ndarray | None = None
        self.grasp_acquisition_step = -1
        self.grasp_stable_window_start_step = -1
        self.grasp_stable_window_end_step = -1
        self.manipulation_start_step = -1
        self.manipulation_end_step = -1
        self.termination_step = -1
        self._actual_qpos_window: list[np.ndarray] = []
        self.grasp_pose_actual_qpos_rad = np.zeros(
            len(ACTIVE_ACTUATORS), dtype=np.float64
        )
        self.grasp_pose_nominal_error_rad = np.zeros(
            len(ACTIVE_ACTUATORS), dtype=np.float64
        )
        self.grasp_pose_joint_stability_span_rad = np.zeros(
            len(ACTIVE_ACTUATORS), dtype=np.float64
        )
        self._active_actuator_ids = np.asarray(
            [model.actuator(name).id for name in ACTIVE_ACTUATORS],
            dtype=np.int64,
        )
        self._nominal_actual_qpos = np.zeros(len(ACTIVE_ACTUATORS))
        self._maximum_nominal_error_rad = math.inf
        self._maximum_stability_span_rad = math.inf
        if self.schema_version >= 9:
            grasp_pose = config["grasp_pose"]
            self._nominal_actual_qpos = np.asarray(
                [
                    float(grasp_pose["nominal_joint_qpos_rad"][name])
                    for name in ACTIVE_ACTUATORS
                ],
                dtype=np.float64,
            )
            self._maximum_nominal_error_rad = float(
                grasp_pose["max_nominal_joint_error_rad"]
            )
            self._maximum_stability_span_rad = float(
                grasp_pose["max_joint_stability_span_rad"]
            )

    @property
    def acquired(self) -> bool:
        return self._acquired

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def consecutive_steps(self) -> int:
        return self._consecutive_steps

    @property
    def initial_pose_latched(self) -> bool:
        """Whether an exact pre-integration cube pose has been supplied."""

        return self._initial_pose_position is not None

    def latch_initial_pose(
        self,
        cube_position: np.ndarray,
        cube_quaternion: np.ndarray,
    ) -> None:
        """Latch the immutable schema-v6 cube pose before the first step.

        The caller must invoke this while ``MjData.time == 0``.  The controller
        deliberately does not auto-latch from a later observation because the
        first persisted sample is already post-integration in the shared
        simulation loop.
        """

        if self.schema_version < 6:
            raise RuntimeError("fixed initial pose history is defined for schema v6+")
        if self.initial_pose_latched:
            raise RuntimeError("initial cube pose has already been latched")
        position = np.asarray(cube_position, dtype=np.float64).reshape(3)
        if not np.isfinite(position).all():
            raise ValueError("initial cube position must be finite")
        self._initial_pose_position = position.copy()
        self._initial_pose_quaternion = _unit_quaternion(
            cube_quaternion, "initial cube quaternion"
        ).copy()

    def initial_pose_reference(self) -> tuple[np.ndarray, np.ndarray]:
        """Return defensive copies of the fixed schema-v6 pose reference."""

        if not self.initial_pose_latched:
            raise RuntimeError("initial cube pose must be latched before use")
        assert self._initial_pose_position is not None
        assert self._initial_pose_quaternion is not None
        return (
            self._initial_pose_position.copy(),
            self._initial_pose_quaternion.copy(),
        )

    def observe_initial_pose_history(
        self,
        cube_position: np.ndarray,
        cube_quaternion: np.ndarray,
    ) -> InitialPoseHistoryEvidence:
        """Update sticky schema-v6 displacement evidence for one sample."""

        reference_position, reference_quaternion = self.initial_pose_reference()
        position = np.asarray(cube_position, dtype=np.float64).reshape(3)
        if not np.isfinite(position).all():
            raise ValueError("cube position must be finite")
        translation = float(np.linalg.norm(position - reference_position))
        orientation = quaternion_drift_deg(
            reference_quaternion, cube_quaternion
        )
        self._initial_pose_max_translation_m = max(
            self._initial_pose_max_translation_m, translation
        )
        self._initial_pose_max_orientation_drift_deg = max(
            self._initial_pose_max_orientation_drift_deg, orientation
        )
        return InitialPoseHistoryEvidence(
            current_translation_m=translation,
            current_orientation_drift_deg=orientation,
            maximum_translation_m=self._initial_pose_max_translation_m,
            maximum_orientation_drift_deg=(
                self._initial_pose_max_orientation_drift_deg
            ),
            translation_history_stable=(
                self._initial_pose_max_translation_m
                <= self.max_initial_translation_m + 1e-12
            ),
            orientation_history_stable=(
                self._initial_pose_max_orientation_drift_deg
                <= self.max_initial_orientation_drift_deg + 1e-12
            ),
        )

    def _schema_v6_close_target(self, fraction: float) -> np.ndarray:
        target = self.pregrasp_target.copy()
        for name in ACTIVE_ACTUATORS:
            actuator_id = self.model.actuator(name).id
            start = self.close_start_fraction[actuator_id]
            end = self.close_end_fraction[actuator_id]
            alpha = smoothstep((fraction - start) / (end - start))
            target[actuator_id] += alpha * (
                self.grasp_target[actuator_id]
                - self.pregrasp_target[actuator_id]
            )
        return target

    def _schema_v8_close_velocity(self, fraction: float) -> np.ndarray:
        """Analytic command-target derivative for profiled CLOSE motion."""

        velocity = np.zeros(self.model.nu, dtype=np.float64)
        close_seconds = self.phase_steps["close"] * float(
            self.model.opt.timestep
        )
        for name in ACTIVE_ACTUATORS:
            actuator_id = self.model.actuator(name).id
            start = self.close_start_fraction[actuator_id]
            end = self.close_end_fraction[actuator_id]
            u = (fraction - start) / (end - start)
            if not 0.0 < u < 1.0:
                continue
            derivative = 6.0 * u * (1.0 - u) / (
                close_seconds * (end - start)
            )
            velocity[actuator_id] = derivative * (
                self.grasp_target[actuator_id]
                - self.pregrasp_target[actuator_id]
            )
        return velocity

    def _latch_abort_hold_target(self) -> None:
        """Freeze the last command so schema-v6 ABORT cannot introduce a jump."""

        if self.schema_version >= 6 and self._abort_hold_target is None:
            self._abort_hold_target = self._last_sent_target.copy()

    def command(self, step: int) -> ControlCommand:
        """Return the command for ``step``; calls must be strictly sequential."""

        if step != self._next_step:
            raise ValueError(
                f"controller expected step {self._next_step}, received {step}"
            )
        if not 0 <= step < self.total_steps:
            raise ValueError(f"step {step} is outside the configured simulation")

        target_velocity = np.zeros(self.model.nu, dtype=np.float64)
        if self._aborted:
            state = ControlState.ABORT
            if self.schema_version >= 6:
                if self._abort_hold_target is None:
                    raise RuntimeError("schema-v6 abort target was not latched")
                target = self._abort_hold_target
            else:
                target = self.grasp_target
            progress = 0.0
        elif step < self.settle_end:
            state = ControlState.SETTLE
            if self.schema_version >= 6:
                if self.initialize_active_joints_at_pregrasp:
                    target = self.pregrasp_target
                else:
                    local_step = step + 1
                    alpha = smoothstep(local_step / self.phase_steps["settle"])
                    target = alpha * self.pregrasp_target
            else:
                target = self.zero_target
            progress = 0.0
        elif step < self.close_end:
            state = ControlState.CLOSE
            local_step = step - self.settle_end + 1
            fraction = local_step / self.phase_steps["close"]
            if self.schema_version >= 6:
                target = self._schema_v6_close_target(fraction)
                if self.schema_version >= 8:
                    target_velocity = self._schema_v8_close_velocity(fraction)
            else:
                alpha = smoothstep(fraction)
                target = alpha * self.grasp_target
            progress = 0.0
        elif not self._acquired:
            # observe() marks timeout on the last permitted VERIFY sample, so
            # this branch never extends VERIFY beyond its declared budget.
            state = ControlState.VERIFY
            target = self.grasp_target
            progress = 0.0
        else:
            offset = step - self.manipulation_start_step
            if offset < self.phase_steps["manipulate"]:
                state = ControlState.MANIPULATE
                raw_progress = (offset + 1) / self.phase_steps["manipulate"]
                if (
                    self.schema_version >= 8
                    and self.manipulation_profile == "minimum_jerk_quintic"
                ):
                    progress = minimum_jerk(raw_progress)
                    manipulate_seconds = self.phase_steps[
                        "manipulate"
                    ] * float(self.model.opt.timestep)
                    derivative = (
                        30.0
                        * raw_progress**2
                        * (1.0 - raw_progress) ** 2
                        / manipulate_seconds
                    )
                    target_velocity = derivative * (
                        self.manipulation_target - self.grasp_target
                    )
                else:
                    progress = smoothstep(raw_progress)
                target = self.grasp_target + progress * (
                    self.manipulation_target - self.grasp_target
                )
            else:
                state = ControlState.HOLD
                target = self.manipulation_target
                progress = 1.0

        self._last_command_state = state
        self._next_step += 1
        self._last_sent_target = np.asarray(target, dtype=np.float64).copy()
        return ControlCommand(
            state=state,
            target=np.asarray(target, dtype=np.float64).copy(),
            manipulation_progress=float(progress),
            target_velocity_rad_s=target_velocity.copy(),
        )

    def stability_reference(
        self,
        cube_position: np.ndarray,
        cube_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the current continuous-window pose reference.

        Before VERIFY, returning the current pose makes the recorded raw gate
        finite without starting or advancing the verification window.
        """

        position = np.asarray(cube_position, dtype=np.float64).reshape(3)
        quaternion = _unit_quaternion(cube_quaternion, "cube quaternion")
        if self.schema_version >= 6:
            return self.initial_pose_reference()
        if self._last_command_state is not ControlState.VERIFY:
            return position.copy(), quaternion.copy()
        if self._stability_reference_position is None:
            self._stability_reference_position = position.copy()
            self._stability_reference_quaternion = quaternion.copy()
        assert self._stability_reference_quaternion is not None
        return (
            self._stability_reference_position.copy(),
            self._stability_reference_quaternion.copy(),
        )

    def _additional_verification_window_passed(self) -> bool:
        """Return schema-specific aggregate VERIFY-window acceptance.

        Older schemas have no additional aggregate measurement.  Keeping the
        default as a constant preserves their state transitions exactly while
        giving schema-v15 a hook for its joint-pair p95 requirement.
        """

        return True

    def observe(
        self,
        step: int,
        evidence: GraspGateEvidence,
        cube_position: np.ndarray,
        cube_quaternion: np.ndarray,
        actual_joint_qpos_rad: np.ndarray | None = None,
        operation_feedback: OperationFeedback | None = None,
    ) -> None:
        """Consume the post-step gate and update next-step state."""

        # Legacy controllers intentionally ignore operation feedback.  The
        # optional argument keeps their command path byte-for-byte unchanged
        # while allowing the schema-v14 subclass to consume the same post-step
        # observation after this historical state-machine update.
        del operation_feedback

        if step != self._next_step - 1 or self._last_command_state is None:
            raise ValueError("observe must immediately follow command for the same step")
        if evidence.hard_abort and not self._aborted:
            self._aborted = True
            self.termination_step = step
            self._consecutive_steps = 0
            self._latch_abort_hold_target()
            return

        state = self._last_command_state
        if state is ControlState.VERIFY:
            if evidence.passed:
                self._consecutive_steps += 1
                if self.schema_version >= 9:
                    if actual_joint_qpos_rad is None:
                        raise ValueError(
                            "schema v9 observe requires actual_joint_qpos_rad"
                        )
                    actual = np.asarray(
                        actual_joint_qpos_rad, dtype=np.float64
                    ).reshape(self.model.nu)
                    active = actual[self._active_actuator_ids]
                    if not np.isfinite(active).all():
                        raise ValueError("actual_joint_qpos_rad must be finite")
                    self._actual_qpos_window.append(active.copy())
                    if len(self._actual_qpos_window) > self.required_stable_steps:
                        self._actual_qpos_window.pop(0)
            else:
                self._consecutive_steps = 0
                self._actual_qpos_window.clear()
                if self.schema_version < 6:
                    self._stability_reference_position = np.asarray(
                        cube_position, dtype=np.float64
                    ).reshape(3).copy()
                    self._stability_reference_quaternion = _unit_quaternion(
                        cube_quaternion, "cube quaternion"
                    ).copy()

            window_passed = self._consecutive_steps >= self.required_stable_steps
            if window_passed and self.schema_version >= 9:
                if len(self._actual_qpos_window) != self.required_stable_steps:
                    raise RuntimeError(
                        "schema v9 actual-qpos window and gate counter diverged"
                    )
                window = np.asarray(self._actual_qpos_window, dtype=np.float64)
                median = np.percentile(window, 50.0, axis=0)
                span = (
                    np.percentile(window, 95.0, axis=0)
                    - np.percentile(window, 5.0, axis=0)
                )
                error = np.abs(median - self._nominal_actual_qpos)
                window_passed = bool(
                    np.all(
                        error
                        <= self._maximum_nominal_error_rad + 1e-12
                    )
                    and np.all(
                        span
                        <= self._maximum_stability_span_rad + 1e-12
                    )
                )
                if window_passed:
                    self.grasp_pose_actual_qpos_rad[:] = median
                    self.grasp_pose_nominal_error_rad[:] = error
                    self.grasp_pose_joint_stability_span_rad[:] = span

            if window_passed:
                window_passed = self._additional_verification_window_passed()

            if window_passed:
                self._acquired = True
                self.grasp_acquisition_step = step
                self.grasp_stable_window_start_step = (
                    step - self.required_stable_steps + 1
                )
                self.grasp_stable_window_end_step = step
                self.manipulation_start_step = step + 1
            elif step >= self.verify_end - 1:
                self._aborted = True
                self.termination_step = step
                self._latch_abort_hold_target()

        if (
            state is ControlState.MANIPULATE
            and self.manipulation_start_step >= 0
            and step
            == self.manipulation_start_step + self.phase_steps["manipulate"] - 1
        ):
            self.manipulation_end_step = step

    def finish(self) -> None:
        """Seal the terminal event after the final sample."""

        if self._next_step != self.total_steps:
            raise ValueError("cannot finish controller before all configured steps")
        if self.termination_step < 0:
            self.termination_step = self.total_steps - 1

    def event_traces(self) -> dict[str, np.ndarray]:
        """Return scalar, pickle-free arrays for NPZ persistence."""

        result = {
            "grasp_acquisition_step": np.asarray(
                self.grasp_acquisition_step, dtype=np.int64
            ),
            "manipulation_start_step": np.asarray(
                self.manipulation_start_step, dtype=np.int64
            ),
            "manipulation_end_step": np.asarray(
                self.manipulation_end_step, dtype=np.int64
            ),
            "termination_step": np.asarray(self.termination_step, dtype=np.int64),
        }
        if self.schema_version >= 9:
            result.update(
                {
                    "grasp_stable_window_start_step": np.asarray(
                        self.grasp_stable_window_start_step, dtype=np.int64
                    ),
                    "grasp_stable_window_end_step": np.asarray(
                        self.grasp_stable_window_end_step, dtype=np.int64
                    ),
                    "grasp_lock_step": np.asarray(
                        self.grasp_acquisition_step, dtype=np.int64
                    ),
                    "grasp_pose_actual_qpos_rad": (
                        self.grasp_pose_actual_qpos_rad.copy()
                    ),
                    "grasp_pose_nominal_error_rad": (
                        self.grasp_pose_nominal_error_rad.copy()
                    ),
                    "grasp_pose_joint_stability_span_rad": (
                        self.grasp_pose_joint_stability_span_rad.copy()
                    ),
                    "contact_preload_command_rad": self.grasp_target.copy(),
                }
            )
        return result


class ContactPreservingPlannedLiftController(
    GraspVerifyThenManipulateController
):
    """Schema-v14 multi-knot lift with causal per-finger force feedback.

    Geometry and trajectory planning happen offline.  At runtime this policy
    tracks the resolved plan and applies a small correction only along each
    finger's verified precontact-to-preload closing ray.  A weak contact
    freezes path time; losing a target-face contact for longer than the
    declared solver-noise allowance aborts instead of silently completing a
    two-finger manipulation.
    """

    SCHEMA_VERSION = 14
    STRATEGY = "grasp_verify_then_contact_preserving_planned_lift"

    def __init__(
        self,
        model: mujoco.MjModel,
        config: Mapping[str, Any],
        phase_steps: Mapping[str, int],
    ) -> None:
        if int(config.get("schema_version", 0)) != self.SCHEMA_VERSION:
            raise ValueError(
                "contact-preserving planned lift requires schema "
                f"v{self.SCHEMA_VERSION}"
            )
        protocol = config["control_protocol"]
        if protocol.get("strategy") != self.STRATEGY:
            raise ValueError(
                f"schema-v{self.SCHEMA_VERSION} control protocol strategy is invalid"
            )

        # Reuse the mature SETTLE/CLOSE/VERIFY event logic without permitting
        # the legacy open-loop MANIPULATE branch to observe the new strategy.
        legacy = copy.deepcopy(dict(config))
        legacy["control_protocol"]["strategy"] = "grasp_verify_then_manipulate"
        super().__init__(model, legacy, phase_steps)
        self.config = copy.deepcopy(dict(config))
        self.timestep = float(model.opt.timestep)

        plan = config["manipulation_plan"]
        self.plan_id = str(plan["plan_id"])
        self.knot_times_s = np.asarray(plan["knot_times_s"], dtype=np.float64)
        if (
            self.knot_times_s.ndim != 1
            or self.knot_times_s.size < 2
            or self.knot_times_s[0] != 0.0
            or np.any(np.diff(self.knot_times_s) <= 0.0)
            or not np.isfinite(self.knot_times_s).all()
        ):
            raise ValueError("manipulation_plan knot times must start at zero and increase")
        self.plan_duration_s = float(
            plan.get("duration_s", self.knot_times_s[-1])
        )
        if not math.isclose(
            self.plan_duration_s,
            float(self.knot_times_s[-1]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("manipulation plan duration must equal its final knot")
        waypoint_mapping = plan["actuator_waypoints_rad"]
        if set(waypoint_mapping) != set(ACTIVE_ACTUATORS):
            raise ValueError("manipulation plan must contain every active actuator")
        self.plan_waypoints = np.zeros(
            (self.knot_times_s.size, model.nu), dtype=np.float64
        )
        for name in ACTIVE_ACTUATORS:
            values = np.asarray(waypoint_mapping[name], dtype=np.float64)
            if values.shape != self.knot_times_s.shape or not np.isfinite(values).all():
                raise ValueError(f"invalid manipulation waypoints for {name}")
            self.plan_waypoints[:, model.actuator(name).id] = values
        if not np.allclose(self.plan_waypoints[0], 0.0, rtol=0.0, atol=1e-14):
            raise ValueError("the first manipulation waypoint must equal preload")
        (
            self.plan_waypoint_velocities_rad_s,
            self.plan_waypoint_accelerations_rad_s2,
        ) = quintic_c2_knot_derivatives(
            self.knot_times_s, self.plan_waypoints
        )

        self.desired_cube_position_delta_m = np.asarray(
            plan["desired_cube_position_delta_m"], dtype=np.float64
        )
        self.desired_cube_rotation_vector_rad = np.asarray(
            plan["desired_cube_rotation_vector_rad"], dtype=np.float64
        )
        expected_path_shape = (self.knot_times_s.size, 3)
        if (
            self.desired_cube_position_delta_m.shape != expected_path_shape
            or self.desired_cube_rotation_vector_rad.shape != expected_path_shape
            or not np.isfinite(self.desired_cube_position_delta_m).all()
            or not np.isfinite(self.desired_cube_rotation_vector_rad).all()
        ):
            raise ValueError("desired object paths must have shape (knot_count, 3)")
        (
            self.desired_cube_position_velocities_m_s,
            self.desired_cube_position_accelerations_m_s2,
        ) = quintic_c2_knot_derivatives(
            self.knot_times_s, self.desired_cube_position_delta_m
        )
        (
            self.desired_cube_rotation_velocities_rad_s,
            self.desired_cube_rotation_accelerations_rad_s2,
        ) = quintic_c2_knot_derivatives(
            self.knot_times_s, self.desired_cube_rotation_vector_rad
        )

        targets = config["contact_force_targets_n"]
        self.minimum_force_target_n = float(targets["minimum_n"])
        self.maximum_force_target_n = float(targets["maximum_n"])
        self.operation_force_target_scale = float(
            targets.get("operation_scale", 1.0)
        )
        if (
            not math.isfinite(self.operation_force_target_scale)
            or not 0.70 <= self.operation_force_target_scale <= 1.0
        ):
            raise ValueError("operation force target scale lies outside [0.70, 1.00]")
        self.force_targets_n = np.asarray(
            [float(targets["per_finger_n"][name]) for name in ACTIVE_FINGERS],
            dtype=np.float64,
        )
        if (
            not np.isfinite(self.force_targets_n).all()
            or np.any(self.force_targets_n < self.minimum_force_target_n)
            or np.any(self.force_targets_n > self.maximum_force_target_n)
        ):
            raise ValueError("contact force targets lie outside their limits")

        feedback = config["contact_feedback"]
        if feedback["strategy"] != "per_finger_force_pi":
            raise ValueError("unsupported schema-v14 contact feedback strategy")
        self.feedback_id = str(feedback["feedback_id"])
        self.filter_time_constant_s = float(feedback["filter_time_constant_s"])
        self.kp = np.asarray(
            [float(feedback["kp_rad_per_n"][name]) for name in ACTIVE_FINGERS]
        )
        self.ki = np.asarray(
            [float(feedback["ki_rad_per_n_s"][name]) for name in ACTIVE_FINGERS]
        )
        self.integral_limit_n_s = float(feedback["integral_limit_n_s"])
        self.correction_limit_rad = float(feedback["correction_limit_rad"])
        self.correction_rate_limit_rad_s = float(feedback["rate_limit_rad_s"])
        self.correction_acceleration_limit_rad_s2 = float(
            feedback["acceleration_limit_rad_s2"]
        )
        self.force_risk_n = float(feedback["force_risk_n"])
        self.freeze_on_risk = bool(feedback["freeze_on_risk"])
        self.max_loss_s = float(feedback["max_loss_s"])
        self.maximum_loss_steps = int(round(self.max_loss_s / self.timestep))
        self.minimum_purity = float(
            config["control_protocol"]["grasp_gate"][
                "min_target_force_fraction"
            ]
        )
        if self.maximum_loss_steps <= 0:
            raise ValueError("contact loss allowance must contain at least one step")
        self.slip_feedback_schema_version = int(feedback.get("schema_version", 1))
        self.slip_feedback_enabled = self.slip_feedback_schema_version >= 2
        self.tangent_slip_freeze_threshold_m = float(
            feedback.get("tangent_slip_freeze_threshold_m", math.inf)
        )
        self.tangent_slip_abort_threshold_m = float(
            feedback.get("tangent_slip_abort_threshold_m", math.inf)
        )
        if self.slip_feedback_enabled and not (
            0.0 < self.tangent_slip_freeze_threshold_m
            <= self.tangent_slip_abort_threshold_m
            and math.isfinite(self.tangent_slip_abort_threshold_m)
        ):
            raise ValueError("schema-v14 tangent-slip feedback thresholds are invalid")
        face_axes = {"+X": 0, "-X": 0, "+Y": 1, "-Y": 1, "+Z": 2, "-Z": 2}
        self._target_face_normal_axes = np.asarray(
            [
                face_axes[str(config["contact_topology"]["target_faces"][finger])]
                for finger in ACTIVE_FINGERS
            ],
            dtype=np.int64,
        )

        preload = actuator_target_vector(
            model, contact_preload_targets(copy.deepcopy(dict(config)))
        )
        precontact = actuator_target_vector(
            model, precontact_targets(copy.deepcopy(dict(config)))
        )
        finger_actuators = (
            ACTIVE_ACTUATORS[0:3],
            ACTIVE_ACTUATORS[3:6],
            ACTIVE_ACTUATORS[6:8],
        )
        self.inward_direction = np.zeros((3, model.nu), dtype=np.float64)
        for finger_index, names in enumerate(finger_actuators):
            ids = np.asarray([model.actuator(name).id for name in names], dtype=int)
            direction = preload[ids] - precontact[ids]
            maximum = float(np.max(np.abs(direction), initial=0.0))
            if maximum <= 1e-12:
                raise ValueError(
                    f"{ACTIVE_FINGERS[finger_index]} closing ray is degenerate"
                )
            self.inward_direction[finger_index, ids] = direction / maximum

        self.ctrl_lower = np.where(
            np.asarray(model.actuator_ctrllimited, dtype=bool),
            np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float64),
            -np.inf,
        )
        self.ctrl_upper = np.where(
            np.asarray(model.actuator_ctrllimited, dtype=bool),
            np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64),
            np.inf,
        )
        self._validate_model_command_limits(config, precontact, preload)
        active_ids = {model.actuator(name).id for name in ACTIVE_ACTUATORS}
        self._inactive_actuator_ids = np.asarray(
            [index for index in range(model.nu) if index not in active_ids],
            dtype=np.int64,
        )
        self._last_feedback: OperationFeedback | None = None
        self._verify_force_history: list[np.ndarray] = []
        self._verify_centroid_cube_local_history: list[np.ndarray] = []
        self._verify_centroid_valid_history: list[np.ndarray] = []
        self._force_target_latched = False
        self.filtered_force_n = np.zeros(3, dtype=np.float64)
        self.force_error_n = np.zeros(3, dtype=np.float64)
        self.force_integral_n_s = np.zeros(3, dtype=np.float64)
        self.feedback_scalar_rad = np.zeros(3, dtype=np.float64)
        self.feedback_scalar_velocity_rad_s = np.zeros(3, dtype=np.float64)
        self.feedback_correction_rad = np.zeros(model.nu, dtype=np.float64)
        self.nominal_plan_target_rad = self.grasp_target.copy()
        self._planned_elapsed_s = 0.0
        self._planned_completed_steps = 0
        self.plan_completion_deadline_step = (
            self.total_steps - self.phase_steps["hold"]
        )
        self.current_plan_progress = 0.0
        self.progress_frozen = False
        self.recovery_active = False
        self.tangent_slip_risk_active = np.zeros(3, dtype=bool)
        self.tangent_slip_abort_risk = np.zeros(3, dtype=bool)
        self.last_observed_tangent_slip_m = np.zeros(3, dtype=np.float64)
        self.last_observed_tangent_slip_valid = np.zeros(3, dtype=bool)
        self.grasp_contact_centroid_baseline_cube_local_m = np.zeros(
            (3, 3), dtype=np.float64
        )
        self.grasp_contact_centroid_baseline_valid = np.zeros(3, dtype=bool)
        self.grasp_contact_centroid_baseline_force_sum_n = np.zeros(
            3, dtype=np.float64
        )
        self._slip_baseline_latched = False
        self.contact_loss_run_steps = np.zeros(3, dtype=np.int64)
        self.maximum_contact_loss_run_steps = np.zeros(3, dtype=np.int64)
        self.planned_knot_index = 0
        self.desired_position_delta_now_m = np.zeros(3, dtype=np.float64)
        self.desired_rotation_vector_now_rad = np.zeros(3, dtype=np.float64)
        self.grasp_lock_cube_position_m = np.zeros(3, dtype=np.float64)
        self.grasp_lock_cube_quaternion_wxyz = np.asarray(
            (1.0, 0.0, 0.0, 0.0), dtype=np.float64
        )
        self.feedback_headroom_abort_step = -1

    def _interpolate_plan(
        self, elapsed_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        waypoint, _, _, index = interpolate_quintic_c2(
            self.knot_times_s,
            self.plan_waypoints,
            elapsed_s,
            knot_velocities=self.plan_waypoint_velocities_rad_s,
            knot_accelerations=self.plan_waypoint_accelerations_rad_s2,
        )
        position, _, _, position_index = interpolate_quintic_c2(
            self.knot_times_s,
            self.desired_cube_position_delta_m,
            elapsed_s,
            knot_velocities=self.desired_cube_position_velocities_m_s,
            knot_accelerations=self.desired_cube_position_accelerations_m_s2,
        )
        rotation, _, _, rotation_index = interpolate_quintic_c2(
            self.knot_times_s,
            self.desired_cube_rotation_vector_rad,
            elapsed_s,
            knot_velocities=self.desired_cube_rotation_velocities_rad_s,
            knot_accelerations=self.desired_cube_rotation_accelerations_rad_s2,
        )
        if position_index != index or rotation_index != index:
            raise RuntimeError("schema-v14 plan interpolation axes diverged")
        return waypoint, position, rotation, index

    def _validate_model_command_limits(
        self,
        config: Mapping[str, Any],
        precontact: np.ndarray,
        preload: np.ndarray,
    ) -> None:
        """Reject v14 commands that rely on MuJoCo's silent ctrl clipping.

        XHAND position actuators use unit joint transmissions, so their ctrl
        targets and the corresponding joint coordinates share radians.  The
        check covers precontact, contact preload, the measured nominal grasp
        qpos and the entire C2 plan.  For the latter, quintic-Hermite segments
        are converted to their six Bezier control points: the curve lies in
        their component-wise convex hull, giving a conservative continuous-
        time bound without sampling between physics steps.

        Feedback is intentionally excluded here.  Its signed scalar is
        bounded online by :meth:`_feedback_scalar_bounds`, which uses the
        remaining two-sided headroom of every actuator owned by that finger.
        """

        tolerance = 1e-12
        active_ids = np.asarray(
            [self.model.actuator(name).id for name in ACTIVE_ACTUATORS],
            dtype=np.int64,
        )
        joint_ids = np.empty(active_ids.size, dtype=np.int64)
        for index, (name, actuator_id) in enumerate(
            zip(ACTIVE_ACTUATORS, active_ids)
        ):
            if int(self.model.actuator_trntype[actuator_id]) != int(
                mujoco.mjtTrn.mjTRN_JOINT
            ):
                raise ValueError(
                    f"schema-v14 actuator {name} must use a joint transmission"
                )
            gear = np.asarray(
                self.model.actuator_gear[actuator_id], dtype=np.float64
            )
            if not np.array_equal(
                gear, np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ):
                raise ValueError(
                    f"schema-v14 actuator {name} must use unit joint gear"
                )
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0 or not bool(self.model.jnt_limited[joint_id]):
                raise ValueError(
                    f"schema-v14 actuator {name} must target a limited joint"
                )
            if not bool(self.model.actuator_ctrllimited[actuator_id]):
                raise ValueError(
                    f"schema-v14 actuator {name} must have a finite ctrl range"
                )
            joint_ids[index] = joint_id

        ctrl_lower = self.ctrl_lower[active_ids]
        ctrl_upper = self.ctrl_upper[active_ids]
        joint_lower = np.asarray(
            self.model.jnt_range[joint_ids, 0], dtype=np.float64
        )
        joint_upper = np.asarray(
            self.model.jnt_range[joint_ids, 1], dtype=np.float64
        )
        self.feedback_command_lower_rad = self.ctrl_lower.copy()
        self.feedback_command_upper_rad = self.ctrl_upper.copy()
        self.feedback_command_lower_rad[active_ids] = np.maximum(
            ctrl_lower, joint_lower
        )
        self.feedback_command_upper_rad[active_ids] = np.minimum(
            ctrl_upper, joint_upper
        )

        def check_values(label: str, values: np.ndarray) -> None:
            active = np.asarray(values, dtype=np.float64)[active_ids]
            if (
                not np.isfinite(active).all()
                or np.any(active < ctrl_lower - tolerance)
                or np.any(active > ctrl_upper + tolerance)
            ):
                raise ValueError(
                    f"schema-v14 {label} exceeds a compiled actuator ctrl range"
                )
            if (
                np.any(active < joint_lower - tolerance)
                or np.any(active > joint_upper + tolerance)
            ):
                raise ValueError(
                    f"schema-v14 {label} exceeds a compiled joint range"
                )

        check_values("precontact target", precontact)
        check_values("contact preload target", preload)

        nominal_qpos = np.zeros(self.model.nu, dtype=np.float64)
        for name in ACTIVE_ACTUATORS:
            nominal_qpos[self.model.actuator(name).id] = float(
                config["grasp_pose"]["nominal_joint_qpos_rad"][name]
            )
        nominal_active = nominal_qpos[active_ids]
        if (
            not np.isfinite(nominal_active).all()
            or np.any(nominal_active < joint_lower - tolerance)
            or np.any(nominal_active > joint_upper + tolerance)
        ):
            raise ValueError(
                "schema-v14 nominal grasp qpos exceeds a compiled joint range"
            )

        relative_lower = np.full(self.model.nu, np.inf, dtype=np.float64)
        relative_upper = np.full(self.model.nu, -np.inf, dtype=np.float64)
        for index, duration in enumerate(np.diff(self.knot_times_s)):
            value0 = self.plan_waypoints[index]
            value1 = self.plan_waypoints[index + 1]
            velocity0 = self.plan_waypoint_velocities_rad_s[index]
            velocity1 = self.plan_waypoint_velocities_rad_s[index + 1]
            acceleration0 = self.plan_waypoint_accelerations_rad_s2[index]
            acceleration1 = self.plan_waypoint_accelerations_rad_s2[index + 1]
            controls = np.stack(
                (
                    value0,
                    value0 + duration * velocity0 / 5.0,
                    value0
                    + 2.0 * duration * velocity0 / 5.0
                    + duration**2 * acceleration0 / 20.0,
                    value1
                    - 2.0 * duration * velocity1 / 5.0
                    + duration**2 * acceleration1 / 20.0,
                    value1 - duration * velocity1 / 5.0,
                    value1,
                )
            )
            relative_lower = np.minimum(relative_lower, np.min(controls, axis=0))
            relative_upper = np.maximum(relative_upper, np.max(controls, axis=0))

        nominal_lower = preload + relative_lower
        nominal_upper = preload + relative_upper
        if (
            np.any(nominal_lower[active_ids] < ctrl_lower - tolerance)
            or np.any(nominal_upper[active_ids] > ctrl_upper + tolerance)
        ):
            raise ValueError(
                "schema-v14 C2 manipulation plan exceeds a compiled actuator ctrl range"
            )
        if (
            np.any(nominal_lower[active_ids] < joint_lower - tolerance)
            or np.any(nominal_upper[active_ids] > joint_upper + tolerance)
        ):
            raise ValueError(
                "schema-v14 C2 manipulation plan exceeds a compiled joint range"
            )
        self.nominal_plan_command_lower_rad = nominal_lower
        self.nominal_plan_command_upper_rad = nominal_upper

    def _feedback_is_at_risk(self) -> bool:
        value = self._last_feedback
        if value is None:
            self.tangent_slip_risk_active[:] = False
            return True
        self.tangent_slip_risk_active[:] = bool(
            self.slip_feedback_enabled
        ) & np.asarray(value.tangent_slip_valid, dtype=bool) & (
            np.asarray(value.tangent_slip_from_grasp_m, dtype=np.float64)
            > self.tangent_slip_freeze_threshold_m
        )
        return bool(
            np.any(value.target_force_n < self.force_risk_n)
            or np.any(value.target_force_purity < self.minimum_purity)
            or not np.all(value.target_face_effective)
            or np.any(value.material_off_target)
            or np.any(value.material_active_nondistal)
            or np.any(self.tangent_slip_risk_active)
        )

    def _latch_slip_baseline(self) -> None:
        """Latch the force-weighted cube-local VERIFY contact centroids."""

        if not self.slip_feedback_enabled or self._slip_baseline_latched:
            return
        if (
            len(self._verify_force_history) != self.required_stable_steps
            or len(self._verify_centroid_cube_local_history)
            != self.required_stable_steps
            or len(self._verify_centroid_valid_history)
            != self.required_stable_steps
        ):
            raise RuntimeError("slip baseline window diverged from grasp lock")
        forces = np.asarray(self._verify_force_history, dtype=np.float64)
        centroids = np.asarray(
            self._verify_centroid_cube_local_history, dtype=np.float64
        )
        valid = np.asarray(self._verify_centroid_valid_history, dtype=bool)
        for finger_index in range(3):
            usable = valid[:, finger_index] & (forces[:, finger_index] > 0.0)
            weights = forces[usable, finger_index]
            if weights.size and float(np.sum(weights)) > 0.0:
                self.grasp_contact_centroid_baseline_force_sum_n[
                    finger_index
                ] = float(np.sum(weights))
                self.grasp_contact_centroid_baseline_cube_local_m[
                    finger_index
                ] = np.average(
                    centroids[usable, finger_index], axis=0, weights=weights
                )
                self.grasp_contact_centroid_baseline_valid[finger_index] = True
        self._slip_baseline_latched = True

    def _with_online_tangent_slip(
        self, value: OperationFeedback
    ) -> OperationFeedback:
        """Derive finite masked slip for a post-step observation."""

        slip = np.zeros(3, dtype=np.float64)
        valid = (
            np.asarray(value.contact_centroid_valid, dtype=bool)
            & np.asarray(value.target_face_effective, dtype=bool)
            & self.grasp_contact_centroid_baseline_valid
        )
        for finger_index, normal_axis in enumerate(
            self._target_face_normal_axes
        ):
            tangent_axes = tuple(axis for axis in range(3) if axis != normal_axis)
            delta = (
                value.contact_centroid_cube_local_m[finger_index]
                - self.grasp_contact_centroid_baseline_cube_local_m[finger_index]
            )
            slip[finger_index] = float(np.linalg.norm(delta[list(tangent_axes)]))
        slip[~valid] = 0.0
        return replace(
            value,
            tangent_slip_from_grasp_m=slip,
            tangent_slip_valid=valid,
        )

    def _feedback_scalar_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return safe outward/inward scalar correction bounds per finger.

        Each finger owns a disjoint actuator subset.  Computing the scalar
        headroom on that subset keeps the PI integrator from winding up behind
        ``np.clip`` when a planned waypoint approaches a real control limit.
        Positive scalar motion follows the verified inward closing ray;
        negative scalar motion is a bounded release along that same ray.
        """

        lower = np.full(3, -self.correction_limit_rad, dtype=np.float64)
        upper = np.full(3, self.correction_limit_rad, dtype=np.float64)
        for finger_index in range(3):
            direction = self.inward_direction[finger_index]
            for actuator_id in np.flatnonzero(np.abs(direction) > 1e-14):
                coefficient = float(direction[actuator_id])
                nominal = float(self.nominal_plan_target_rad[actuator_id])
                if coefficient > 0.0:
                    actuator_lower = (
                        float(self.feedback_command_lower_rad[actuator_id])
                        - nominal
                    ) / coefficient
                    actuator_upper = (
                        float(self.feedback_command_upper_rad[actuator_id])
                        - nominal
                    ) / coefficient
                else:
                    actuator_lower = (
                        float(self.feedback_command_upper_rad[actuator_id])
                        - nominal
                    ) / coefficient
                    actuator_upper = (
                        float(self.feedback_command_lower_rad[actuator_id])
                        - nominal
                    ) / coefficient
                lower[finger_index] = max(
                    lower[finger_index], float(actuator_lower)
                )
                upper[finger_index] = min(
                    upper[finger_index], float(actuator_upper)
                )
        lower = np.clip(
            lower, -self.correction_limit_rad, self.correction_limit_rad
        )
        upper = np.clip(
            upper, -self.correction_limit_rad, self.correction_limit_rad
        )
        if np.any(lower > upper + 1e-12):
            raise RuntimeError("schema-v14 feedback ctrl headroom is empty")
        return lower, upper

    def _feedback_scalar_upper_bounds(self) -> np.ndarray:
        """Compatibility accessor for tests and inward-recovery diagnostics."""

        return self._feedback_scalar_bounds()[1]

    def _update_feedback_correction(self) -> np.ndarray:
        value = self._last_feedback
        if value is None:
            self.force_error_n[:] = self.force_targets_n
            needs_recovery = np.ones(3, dtype=bool)
        else:
            self.force_error_n[:] = self.force_targets_n - self.filtered_force_n

            # A force-pure contact can briefly become ineffective before the
            # filtered force has decayed.  In that case a force-only PI error
            # would incorrectly request zero (or outward) correction.  The
            # recovery contract is explicitly inward-only, so seed a small
            # positive error for precisely the finger that is at risk.
            needs_recovery = (
                (value.target_force_n < self.force_risk_n)
                | (value.target_force_purity < self.minimum_purity)
                | ~np.asarray(value.target_face_effective, dtype=bool)
            )
            self.force_error_n[needs_recovery] = np.maximum(
                self.force_error_n[needs_recovery], self.force_risk_n
            )

        scalar_lower, scalar_upper = self._feedback_scalar_bounds()

        proposed_integral = np.clip(
            self.force_integral_n_s + self.force_error_n * self.timestep,
            -self.integral_limit_n_s,
            self.integral_limit_n_s,
        )
        # Contact recovery always requests inward motion.  A stale negative
        # integral accumulated while unloading a high-force contact must not
        # reverse the recovery objective when force/purity becomes unsafe.
        proposed_integral[needs_recovery] = np.maximum(
            proposed_integral[needs_recovery], 0.0
        )
        raw_with_integral = self.kp * self.force_error_n + self.ki * proposed_integral
        # Conditional integration is symmetric: freeze only the component
        # that would drive farther into the currently active saturation.
        drives_farther_into_saturation = (
            (raw_with_integral > scalar_upper)
            & (self.force_error_n > 0.0)
            & (proposed_integral >= 0.0)
        ) | (
            (raw_with_integral < scalar_lower)
            & (self.force_error_n < 0.0)
            & (proposed_integral <= 0.0)
        )
        self.force_integral_n_s[:] = np.where(
            drives_farther_into_saturation,
            self.force_integral_n_s,
            proposed_integral,
        )
        desired = np.clip(
            self.kp * self.force_error_n + self.ki * self.force_integral_n_s,
            scalar_lower,
            scalar_upper,
        )
        inward_recovery_request = np.minimum(
            scalar_upper,
            self.kp * np.maximum(self.force_error_n, self.force_risk_n),
        )
        desired[needs_recovery] = np.maximum(
            desired[needs_recovery], inward_recovery_request[needs_recovery]
        )

        # A bang-bang ``(desired-current)/dt`` velocity can hit either scalar
        # bound with non-zero speed, after which clipping creates an
        # arbitrarily large apparent acceleration.  A discrete stopping-speed
        # envelope brakes before both the desired correction and hard bounds.
        acceleration = self.correction_acceleration_limit_rad_s2

        def stopping_speed(distance: np.ndarray) -> np.ndarray:
            nonnegative = np.maximum(np.asarray(distance), 0.0)
            return np.maximum(
                0.0,
                -acceleration * self.timestep
                + np.sqrt(
                    (acceleration * self.timestep) ** 2
                    + 2.0 * acceleration * nonnegative
                ),
            )

        error_to_target = desired - self.feedback_scalar_rad
        desired_velocity = np.sign(error_to_target) * np.minimum(
            self.correction_rate_limit_rad_s,
            stopping_speed(np.abs(error_to_target)),
        )
        velocity_change = np.clip(
            desired_velocity - self.feedback_scalar_velocity_rad_s,
            -acceleration * self.timestep,
            acceleration * self.timestep,
        )
        next_velocity = np.clip(
            self.feedback_scalar_velocity_rad_s + velocity_change,
            -self.correction_rate_limit_rad_s,
            self.correction_rate_limit_rad_s,
        )
        next_velocity = np.clip(
            next_velocity,
            -stopping_speed(self.feedback_scalar_rad - scalar_lower),
            stopping_speed(scalar_upper - self.feedback_scalar_rad),
        )
        updated = self.feedback_scalar_rad + next_velocity * self.timestep
        bounded = np.minimum(np.maximum(updated, scalar_lower), scalar_upper)
        self.feedback_scalar_velocity_rad_s[:] = (
            bounded - self.feedback_scalar_rad
        ) / self.timestep
        self.feedback_scalar_rad[:] = bounded
        # The dynamic bound, not merely the configured global limit, is the
        # final authority.  Thus ctrl clipping cannot conceal PI windup.
        self.feedback_scalar_rad[:] = np.minimum(
            np.maximum(self.feedback_scalar_rad, scalar_lower), scalar_upper
        )
        self.feedback_correction_rad[:] = np.sum(
            self.feedback_scalar_rad[:, None] * self.inward_direction, axis=0
        )
        return self.feedback_correction_rad.copy()

    def command(self, step: int) -> ControlCommand:
        # ``super().command`` advances the legacy state machine and records
        # its open-loop target in ``_last_sent_target``.  Preserve the command
        # that was actually sent on the previous physics sample so the v14
        # rate trace and rate limiter are not accidentally measured against
        # the unused legacy interpolation.
        previous_sent_target = self._last_sent_target.copy()
        if (
            self.acquired
            and not self.aborted
            and self._planned_elapsed_s
            < self.plan_duration_s - 0.5 * self.timestep
            and step >= self.plan_completion_deadline_step
        ):
            # Do not consume the declared one-second HOLD in order to disguise
            # excessive freezes.  The candidate is infeasible within the
            # fixed experiment duration and must remain a failed attempt.
            self._aborted = True
            self.termination_step = int(step)
            self._consecutive_steps = 0
            self._abort_hold_target = previous_sent_target.copy()
        legacy = super().command(step)
        if legacy.state not in (ControlState.MANIPULATE, ControlState.HOLD):
            self.nominal_plan_target_rad[:] = legacy.target
            self.feedback_correction_rad[:] = 0.0
            self.progress_frozen = False
            self.recovery_active = False
            self.tangent_slip_risk_active[:] = False
            return legacy

        at_risk = self._feedback_is_at_risk()
        self.recovery_active = at_risk
        plan_was_active = self._planned_elapsed_s < (
            self.plan_duration_s - 0.5 * self.timestep
        )
        self.progress_frozen = bool(
            plan_was_active
            and self.freeze_on_risk
            and at_risk
        )
        if plan_was_active and not self.progress_frozen:
            self._planned_completed_steps += 1
            # True freeze/resume semantics: risk samples do not advance the
            # plan clock and recovered samples advance by exactly one physics
            # interval.  In particular, recovery never jumps across knots to
            # catch up with wall-clock time.
            elapsed = self._planned_completed_steps * self.timestep
            self._planned_elapsed_s = min(
                self.plan_duration_s, elapsed
            )
        waypoint, desired_position, desired_rotation, knot = self._interpolate_plan(
            self._planned_elapsed_s
        )
        self.planned_knot_index = knot
        self.desired_position_delta_now_m[:] = desired_position
        self.desired_rotation_vector_now_rad[:] = desired_rotation
        self.nominal_plan_target_rad[:] = self.grasp_target + waypoint
        scalar_lower, scalar_upper = self._feedback_scalar_bounds()
        if np.any(
            (self.feedback_scalar_rad < scalar_lower - 1e-12)
            | (self.feedback_scalar_rad > scalar_upper + 1e-12)
        ):
            # A planned feed-forward change removed correction headroom faster
            # than the bounded feedback state can retract.  Holding the last
            # valid command is the only action that simultaneously preserves
            # ctrl, correction-rate and acceleration invariants.
            self._aborted = True
            self.termination_step = int(step)
            self._consecutive_steps = 0
            self._abort_hold_target = previous_sent_target.copy()
            self.feedback_headroom_abort_step = int(step)
            self._last_command_state = ControlState.ABORT
            self._last_sent_target = previous_sent_target.copy()
            return ControlCommand(
                state=ControlState.ABORT,
                target=previous_sent_target.copy(),
                manipulation_progress=float(self.current_plan_progress),
                target_velocity_rad_s=np.zeros(self.model.nu, dtype=np.float64),
            )
        correction = self._update_feedback_correction()
        target = np.clip(
            self.nominal_plan_target_rad + correction,
            self.ctrl_lower,
            self.ctrl_upper,
        )
        # The inactive controls are a safety invariant, independent of model
        # ctrl ranges or future plan-format changes.
        target[self._inactive_actuator_ids] = 0.0
        velocity = (target - previous_sent_target) / self.timestep
        progress = min(1.0, self._planned_elapsed_s / self.plan_duration_s)
        self.current_plan_progress = float(progress)
        # Early acquisition leaves up to ``verify_timeout-stable_window`` of
        # slack before the mandatory final HOLD.  Use that slack to finish a
        # genuinely frozen plan instead of either jumping ahead on recovery or
        # silently stopping at the legacy three-second boundary.
        effective_state = (
            ControlState.MANIPULATE
            if plan_was_active
            else ControlState.HOLD
        )
        self._last_command_state = effective_state
        self._last_sent_target = target.copy()
        return ControlCommand(
            state=effective_state,
            target=target.copy(),
            manipulation_progress=float(progress),
            target_velocity_rad_s=velocity.copy(),
        )

    def _abort_for_operation(self, step: int) -> None:
        if self._aborted:
            return
        self._aborted = True
        self.termination_step = int(step)
        self._consecutive_steps = 0
        self._latch_abort_hold_target()

    def observe(
        self,
        step: int,
        evidence: GraspGateEvidence,
        cube_position: np.ndarray,
        cube_quaternion: np.ndarray,
        actual_joint_qpos_rad: np.ndarray | None = None,
        operation_feedback: OperationFeedback | None = None,
    ) -> None:
        state = self._last_command_state
        if state is ControlState.VERIFY:
            if evidence.passed:
                self._verify_force_history.append(
                    np.asarray(evidence.target_faces.target_force_n).copy()
                )
                if self.slip_feedback_enabled:
                    if operation_feedback is None:
                        raise ValueError(
                            "schema-v14 slip feedback requires operation feedback"
                        )
                    self._verify_centroid_cube_local_history.append(
                        operation_feedback.contact_centroid_cube_local_m.copy()
                    )
                    self._verify_centroid_valid_history.append(
                        operation_feedback.contact_centroid_valid.copy()
                    )
                if len(self._verify_force_history) > self.required_stable_steps:
                    self._verify_force_history.pop(0)
                    if self.slip_feedback_enabled:
                        self._verify_centroid_cube_local_history.pop(0)
                        self._verify_centroid_valid_history.pop(0)
            else:
                self._verify_force_history.clear()
                self._verify_centroid_cube_local_history.clear()
                self._verify_centroid_valid_history.clear()
        acquired_before_observation = self.acquired
        super().observe(
            step,
            evidence,
            cube_position,
            cube_quaternion,
            actual_joint_qpos_rad=actual_joint_qpos_rad,
            operation_feedback=operation_feedback,
        )
        if (
            state is ControlState.MANIPULATE
            and self.current_plan_progress < 1.0 - 1e-12
            and self.manipulation_end_step == step
        ):
            # The fixed legacy window ended while contact-risk freezes were
            # active.  Continue the unresolved tail into verification time
            # that was saved by early grasp acquisition; do not publish a
            # false manipulation-complete event.
            self.manipulation_end_step = -1
        elif (
            state is ControlState.MANIPULATE
            and self.current_plan_progress >= 1.0 - 1e-12
        ):
            self.manipulation_end_step = step
        if self.acquired and not acquired_before_observation:
            self.grasp_lock_cube_position_m[:] = np.asarray(
                cube_position, dtype=np.float64
            ).reshape(3)
            self.grasp_lock_cube_quaternion_wxyz[:] = _unit_quaternion(
                cube_quaternion, "grasp-lock cube quaternion"
            )
            self._latch_slip_baseline()
        if self.acquired and not self._force_target_latched:
            if len(self._verify_force_history) != self.required_stable_steps:
                raise RuntimeError("force target window diverged from grasp lock")
            measured = np.median(np.asarray(self._verify_force_history), axis=0)
            if self.operation_force_target_scale == 1.0:
                # Preserve the exact legacy arithmetic path and therefore all
                # existing schema-v14 command/trace samples bit-for-bit.
                self.force_targets_n[:] = np.clip(
                    np.maximum(self.force_targets_n, measured),
                    self.minimum_force_target_n,
                    self.maximum_force_target_n,
                )
            else:
                self.force_targets_n[:] = np.clip(
                    self.operation_force_target_scale
                    * np.maximum(self.force_targets_n, measured),
                    self.minimum_force_target_n,
                    self.maximum_force_target_n,
                )
            self._force_target_latched = True

        if operation_feedback is None:
            return
        if self.slip_feedback_enabled:
            operation_feedback = self._with_online_tangent_slip(
                operation_feedback
            )
        self.last_observed_tangent_slip_m[:] = (
            operation_feedback.tangent_slip_from_grasp_m
        )
        self.last_observed_tangent_slip_valid[:] = (
            operation_feedback.tangent_slip_valid
        )
        alpha = self.timestep / (self.filter_time_constant_s + self.timestep)
        if self._last_feedback is None:
            self.filtered_force_n[:] = operation_feedback.target_force_n
        else:
            self.filtered_force_n[:] += alpha * (
                operation_feedback.target_force_n - self.filtered_force_n
            )

        self.tangent_slip_abort_risk[:] = False
        if state in (ControlState.MANIPULATE, ControlState.HOLD):
            effective = np.asarray(
                operation_feedback.target_face_effective, dtype=bool
            )
            self.contact_loss_run_steps[:] = np.where(
                effective, 0, self.contact_loss_run_steps + 1
            )
            self.maximum_contact_loss_run_steps[:] = np.maximum(
                self.maximum_contact_loss_run_steps,
                self.contact_loss_run_steps,
            )
            self.tangent_slip_abort_risk[:] = bool(
                self.slip_feedback_enabled
            ) & operation_feedback.tangent_slip_valid & (
                operation_feedback.tangent_slip_from_grasp_m
                > self.tangent_slip_abort_threshold_m
            )
            unsafe = bool(
                operation_feedback.forbidden_contact
                or np.any(operation_feedback.material_off_target)
                or np.any(operation_feedback.material_active_nondistal)
                or operation_feedback.max_penetration_m
                > float(self.config["acceptance"]["max_penetration_m"]) + 1e-12
                or not operation_feedback.finite
                or not operation_feedback.joint_limits_respected
                or not operation_feedback.inactive_controls_zero
                or np.any(self.contact_loss_run_steps > self.maximum_loss_steps)
                or np.any(self.tangent_slip_abort_risk)
            )
            if unsafe:
                self._abort_for_operation(step)
        self._last_feedback = operation_feedback

    def event_traces(self) -> dict[str, np.ndarray]:
        result = super().event_traces()
        result.update(
            {
                "manipulation_plan_id": np.asarray(self.plan_id, dtype=np.str_),
                "contact_feedback_id": np.asarray(self.feedback_id, dtype=np.str_),
                "maximum_contact_loss_run_steps": (
                    self.maximum_contact_loss_run_steps.copy()
                ),
                "resolved_contact_force_targets_n": self.force_targets_n.copy(),
                "grasp_lock_cube_position_m": (
                    self.grasp_lock_cube_position_m.copy()
                ),
                "grasp_lock_cube_quaternion_wxyz": (
                    self.grasp_lock_cube_quaternion_wxyz.copy()
                ),
                "feedback_headroom_abort_step": np.asarray(
                    self.feedback_headroom_abort_step, dtype=np.int64
                ),
            }
        )
        if self.slip_feedback_enabled:
            result.update(
                {
                    "online_grasp_contact_centroid_baseline_cube_local_m": (
                        self.grasp_contact_centroid_baseline_cube_local_m.copy()
                    ),
                    "online_grasp_contact_centroid_baseline_valid": (
                        self.grasp_contact_centroid_baseline_valid.copy()
                    ),
                    "online_grasp_contact_centroid_baseline_force_sum_n": (
                        self.grasp_contact_centroid_baseline_force_sum_n.copy()
                    ),
                }
            )
        return result


class JointPairAlignedContactPreservingPlannedLiftController(
    ContactPreservingPlannedLiftController
):
    """Schema-v15 causal joint-pair alignment and slip-recovery policy.

    The inherited controller remains the sole owner of plan time and
    per-finger force PI state.  This layer consumes only the observation
    latched after the previous physics step, computes an alignment request in
    the weighted object-z/force nullspace, adds finger-specific inward slip
    recovery, and applies independent position/rate/acceleration bounds.
    """

    SCHEMA_VERSION = 15
    STRATEGY = (
        "grasp_verify_then_joint_pair_aligned_contact_preserving_planned_lift"
    )

    def __init__(
        self,
        model: mujoco.MjModel,
        config: Mapping[str, Any],
        phase_steps: Mapping[str, int],
    ) -> None:
        super().__init__(model, config, phase_steps)
        alignment = config["joint_pair_alignment"]
        feedback = config["joint_pair_feedback"]

        self.joint_pair_alignment_id = str(alignment["alignment_id"])
        self.joint_pair_feedback_id = str(feedback["feedback_id"])
        self.joint_pair_joint_names = tuple(
            str(value) for value in alignment["joint_names"]
        )
        if len(self.joint_pair_joint_names) != 2:
            raise ValueError("joint_pair_alignment.joint_names must contain two names")
        self.joint_pair_minimum_length_m = float(alignment["minimum_length_m"])
        self.joint_pair_grasp_p95_max_deg = float(
            alignment["grasp_p95_max_deg"]
        )
        self.joint_pair_grasp_max_deg = float(alignment["grasp_max_deg"])
        self.joint_pair_operation_p95_max_deg = float(
            alignment["operation_p95_max_deg"]
        )
        self.joint_pair_operation_max_deg = float(
            alignment["operation_max_deg"]
        )
        self.joint_pair_operation_duty_min = float(
            alignment["operation_within_p95_limit_duty_min"]
        )
        self.joint_pair_max_violation_s = float(
            alignment["max_continuous_violation_s"]
        )
        self.joint_pair_max_violation_steps = int(
            round(self.joint_pair_max_violation_s / self.timestep)
        )
        if self.joint_pair_max_violation_steps <= 0:
            raise ValueError(
                "joint-pair continuous violation allowance must contain a step"
            )

        feedback_schema_version = int(feedback.get("schema_version", 1))
        expected_feedback_strategy = (
            "previous_frame_weighted_nullspace"
            if feedback_schema_version == 1
            else "previous_frame_rolling_aware_signed_tangent_nullspace"
        )
        if feedback["strategy"] != expected_feedback_strategy:
            raise ValueError(
                "unsupported joint-pair feedback strategy for schema "
                f"{feedback_schema_version}"
            )
        if not bool(feedback["use_previous_observation"]):
            raise ValueError("schema-v15 joint-pair feedback must use the previous frame")
        self.joint_pair_alignment_gain = float(feedback["alignment_gain"])
        self.joint_pair_slip_recovery_gain_rad_per_m = float(
            feedback["slip_recovery_gain_rad_per_m"]
        )
        self.joint_pair_correction_limit_rad = float(
            feedback["correction_limit_rad"]
        )
        self.joint_pair_rate_limit_rad_s = float(feedback["rate_limit_rad_s"])
        self.joint_pair_acceleration_limit_rad_s2 = float(
            feedback["acceleration_limit_rad_s2"]
        )
        self.joint_pair_freeze_threshold_deg = float(
            feedback["freeze_threshold_deg"]
        )
        self.joint_pair_abort_threshold_deg = float(
            feedback["abort_threshold_deg"]
        )
        self.joint_pair_slip_freeze_threshold_m = float(
            feedback["slip_freeze_threshold_m"]
        )
        self.joint_pair_slip_abort_threshold_m = float(
            feedback["slip_abort_threshold_m"]
        )
        self.joint_pair_damping = float(feedback.get("damping", 1e-8))
        self.joint_pair_vertical_response_weight = float(
            feedback.get("vertical_response_weight", 1.0)
        )
        self.joint_pair_force_response_weight = float(
            feedback.get("force_response_weight", 1.0)
        )

        # Schema-v15 owns the stricter 1.5/2.0 mm slip contract even when the
        # inherited v14 force-feedback block uses schema 1.
        self.slip_feedback_enabled = True
        self.tangent_slip_freeze_threshold_m = (
            self.joint_pair_slip_freeze_threshold_m
        )
        self.tangent_slip_abort_threshold_m = (
            self.joint_pair_slip_abort_threshold_m
        )

        active_count = len(ACTIVE_ACTUATORS)
        knot_count = int(self.knot_times_s.size)

        def plan_matrix(name: str, rows: int) -> tuple[np.ndarray, np.ndarray]:
            active = np.asarray(
                config["manipulation_plan"][name], dtype=np.float64
            )
            expected = (knot_count, rows, active_count)
            if active.shape != expected or not np.isfinite(active).all():
                raise ValueError(f"manipulation_plan.{name} must have shape {expected}")
            full = np.zeros((knot_count, rows, model.nu), dtype=np.float64)
            full[:, :, self._active_actuator_ids] = active
            return active.copy(), full

        (
            self.joint_pair_plan_residual_jacobian_active,
            self.joint_pair_plan_residual_jacobian,
        ) = plan_matrix("joint_pair_residual_jacobian_2x8", 2)
        (
            self.joint_pair_plan_object_jacobian_active,
            self.joint_pair_plan_object_jacobian,
        ) = plan_matrix("object_response_jacobian_6x8", 6)
        (
            self.joint_pair_plan_force_jacobian_active,
            self.joint_pair_plan_force_jacobian,
        ) = plan_matrix("target_force_jacobian_3x8", 3)

        self._joint_pair_verify_angles_deg: list[float] = []
        self.joint_pair_grasp_p95_deg = 180.0
        self.joint_pair_grasp_max_observed_deg = 180.0
        self.joint_pair_vector_cube_m = np.zeros(3, dtype=np.float64)
        self.joint_pair_residual = np.zeros(2, dtype=np.float64)
        self.joint_pair_angle_deg = 180.0
        self.joint_pair_length_m = 0.0
        self.joint_pair_positive_y = False
        self.joint_pair_valid = False
        self.joint_pair_freeze_risk_active = False
        self.joint_pair_abort_risk = False
        self.joint_pair_slip_recovery_active = np.zeros(3, dtype=bool)
        self.joint_pair_violation_run_steps = 0
        self.joint_pair_maximum_violation_run_steps = 0
        self.joint_pair_alignment_request_rad = np.zeros(
            model.nu, dtype=np.float64
        )
        self.joint_pair_slip_recovery_correction_rad = np.zeros(
            model.nu, dtype=np.float64
        )
        self.joint_pair_feedback_correction_rad = np.zeros(
            model.nu, dtype=np.float64
        )
        self.joint_pair_feedback_velocity_rad_s = np.zeros(
            model.nu, dtype=np.float64
        )
        self.joint_pair_active_residual_jacobian_2x8 = np.zeros(
            (2, active_count), dtype=np.float64
        )
        self.joint_pair_feedback_saturated = False
        self.joint_pair_feedback_headroom_abort_step = -1
        self.joint_pair_abort_reason = ""

    def _additional_verification_window_passed(self) -> bool:
        if len(self._joint_pair_verify_angles_deg) != self.required_stable_steps:
            return False
        values = np.asarray(self._joint_pair_verify_angles_deg, dtype=np.float64)
        self.joint_pair_grasp_p95_deg = float(np.percentile(values, 95.0))
        self.joint_pair_grasp_max_observed_deg = float(np.max(values))
        return bool(
            self.joint_pair_grasp_p95_deg
            <= self.joint_pair_grasp_p95_max_deg + 1e-12
            and self.joint_pair_grasp_max_observed_deg
            <= self.joint_pair_grasp_max_deg + 1e-12
        )

    def _feedback_is_at_risk(self) -> bool:
        inherited = super()._feedback_is_at_risk()
        value = self._last_feedback
        self.joint_pair_freeze_risk_active = bool(
            value is None
            or not value.joint_pair_valid
            or not value.joint_pair_positive_y
            or value.joint_pair_length_m
            < self.joint_pair_minimum_length_m - 1e-12
            or value.joint_pair_angle_deg
            > self.joint_pair_freeze_threshold_deg + 1e-12
        )
        self.joint_pair_slip_recovery_active[:] = False
        if value is not None:
            self.joint_pair_slip_recovery_active[:] = (
                value.tangent_slip_valid
                & (
                    value.tangent_slip_from_grasp_m
                    >= self.joint_pair_slip_freeze_threshold_m - 1e-12
                )
            )
        return bool(
            inherited
            or self.joint_pair_freeze_risk_active
            or np.any(self.joint_pair_slip_recovery_active)
        )

    def _weighted_nullspace_alignment_request(self) -> np.ndarray:
        result = np.zeros(self.model.nu, dtype=np.float64)
        value = self._last_feedback
        if (
            value is None
            or not value.joint_pair_valid
            or not value.joint_pair_positive_y
            or value.joint_pair_length_m < self.joint_pair_minimum_length_m
        ):
            self.joint_pair_active_residual_jacobian_2x8[:] = 0.0
            return result
        knot = int(np.clip(self.planned_knot_index, 0, self.knot_times_s.size - 1))
        pair_jacobian = self.joint_pair_plan_residual_jacobian_active[knot]
        self.joint_pair_active_residual_jacobian_2x8[:] = pair_jacobian
        protected = np.vstack(
            (
                self.joint_pair_vertical_response_weight
                * self.joint_pair_plan_object_jacobian_active[knot, 2:3],
                self.joint_pair_force_response_weight
                * self.joint_pair_plan_force_jacobian_active[knot],
            )
        )
        identity = np.eye(len(ACTIVE_ACTUATORS), dtype=np.float64)
        protected_gram = protected @ protected.T
        projector = identity - protected.T @ np.linalg.solve(
            protected_gram
            + self.joint_pair_damping * np.eye(protected.shape[0]),
            protected,
        )
        effective = pair_jacobian @ projector
        gram = effective @ effective.T
        active_request = -self.joint_pair_alignment_gain * (
            projector
            @ effective.T
            @ np.linalg.solve(
                gram + self.joint_pair_damping * np.eye(2),
                np.asarray(value.joint_pair_residual, dtype=np.float64),
            )
        )
        result[self._active_actuator_ids] = active_request
        return result

    def _slip_recovery_request(self) -> np.ndarray:
        result = np.zeros(self.model.nu, dtype=np.float64)
        value = self._last_feedback
        if value is None:
            return result
        active = value.tangent_slip_valid & (
            value.tangent_slip_from_grasp_m
            >= self.joint_pair_slip_freeze_threshold_m - 1e-12
        )
        scalars = np.where(
            active,
            self.joint_pair_slip_recovery_gain_rad_per_m
            * value.tangent_slip_from_grasp_m,
            0.0,
        )
        result[:] = np.sum(scalars[:, None] * self.inward_direction, axis=0)
        return result

    def _bounded_joint_pair_correction(
        self, desired: np.ndarray, base_target: np.ndarray
    ) -> np.ndarray | None:
        lower = np.maximum(
            -self.joint_pair_correction_limit_rad,
            self.feedback_command_lower_rad - base_target,
        )
        upper = np.minimum(
            self.joint_pair_correction_limit_rad,
            self.feedback_command_upper_rad - base_target,
        )
        lower[self._inactive_actuator_ids] = 0.0
        upper[self._inactive_actuator_ids] = 0.0
        if np.any(
            (self.joint_pair_feedback_correction_rad < lower - 1e-12)
            | (self.joint_pair_feedback_correction_rad > upper + 1e-12)
        ):
            return None
        clipped_desired = np.minimum(np.maximum(desired, lower), upper)
        self.joint_pair_feedback_saturated = bool(
            not np.allclose(clipped_desired, desired, rtol=0.0, atol=1e-14)
        )
        acceleration = self.joint_pair_acceleration_limit_rad_s2

        def stopping_speed(distance: np.ndarray) -> np.ndarray:
            nonnegative = np.maximum(np.asarray(distance), 0.0)
            return np.maximum(
                0.0,
                -acceleration * self.timestep
                + np.sqrt(
                    (acceleration * self.timestep) ** 2
                    + 2.0 * acceleration * nonnegative
                ),
            )

        error = clipped_desired - self.joint_pair_feedback_correction_rad
        desired_velocity = np.sign(error) * np.minimum(
            self.joint_pair_rate_limit_rad_s,
            stopping_speed(np.abs(error)),
        )
        velocity_delta = np.clip(
            desired_velocity - self.joint_pair_feedback_velocity_rad_s,
            -acceleration * self.timestep,
            acceleration * self.timestep,
        )
        next_velocity = np.clip(
            self.joint_pair_feedback_velocity_rad_s + velocity_delta,
            -self.joint_pair_rate_limit_rad_s,
            self.joint_pair_rate_limit_rad_s,
        )
        next_velocity = np.clip(
            next_velocity,
            -stopping_speed(self.joint_pair_feedback_correction_rad - lower),
            stopping_speed(upper - self.joint_pair_feedback_correction_rad),
        )
        updated = self.joint_pair_feedback_correction_rad + (
            next_velocity * self.timestep
        )
        bounded = np.minimum(np.maximum(updated, lower), upper)
        self.joint_pair_feedback_velocity_rad_s[:] = (
            bounded - self.joint_pair_feedback_correction_rad
        ) / self.timestep
        self.joint_pair_feedback_correction_rad[:] = bounded
        return bounded.copy()

    def command(self, step: int) -> ControlCommand:
        previous_sent_target = self._last_sent_target.copy()
        base = super().command(step)
        if base.state not in (ControlState.MANIPULATE, ControlState.HOLD):
            self.joint_pair_alignment_request_rad[:] = 0.0
            self.joint_pair_slip_recovery_correction_rad[:] = 0.0
            self.joint_pair_active_residual_jacobian_2x8[:] = 0.0
            self.joint_pair_freeze_risk_active = False
            self.joint_pair_slip_recovery_active[:] = False
            if base.state is not ControlState.ABORT:
                self.joint_pair_feedback_correction_rad[:] = 0.0
                self.joint_pair_feedback_velocity_rad_s[:] = 0.0
            return base

        self.joint_pair_alignment_request_rad[:] = (
            self._weighted_nullspace_alignment_request()
        )
        self.joint_pair_slip_recovery_correction_rad[:] = (
            self._slip_recovery_request()
        )
        desired = np.clip(
            self.joint_pair_alignment_request_rad
            + self.joint_pair_slip_recovery_correction_rad,
            -self.joint_pair_correction_limit_rad,
            self.joint_pair_correction_limit_rad,
        )
        desired[self._inactive_actuator_ids] = 0.0
        correction = self._bounded_joint_pair_correction(desired, base.target)
        if correction is None:
            self._abort_for_operation(step)
            self.joint_pair_feedback_headroom_abort_step = int(step)
            self.joint_pair_abort_reason = "feedback_headroom"
            self._last_command_state = ControlState.ABORT
            self._last_sent_target = previous_sent_target.copy()
            return ControlCommand(
                state=ControlState.ABORT,
                target=previous_sent_target.copy(),
                manipulation_progress=float(self.current_plan_progress),
                target_velocity_rad_s=np.zeros(self.model.nu, dtype=np.float64),
            )
        target = base.target + correction
        target[self._inactive_actuator_ids] = 0.0
        velocity = (target - previous_sent_target) / self.timestep
        self._last_sent_target = target.copy()
        return ControlCommand(
            state=base.state,
            target=target.copy(),
            manipulation_progress=base.manipulation_progress,
            target_velocity_rad_s=velocity,
        )

    def observe(
        self,
        step: int,
        evidence: GraspGateEvidence,
        cube_position: np.ndarray,
        cube_quaternion: np.ndarray,
        actual_joint_qpos_rad: np.ndarray | None = None,
        operation_feedback: OperationFeedback | None = None,
    ) -> None:
        if operation_feedback is None:
            raise ValueError("schema-v15 requires joint-pair operation feedback")
        state = self._last_command_state
        if state is ControlState.VERIFY:
            if evidence.passed:
                self._joint_pair_verify_angles_deg.append(
                    float(operation_feedback.joint_pair_angle_deg)
                )
                if (
                    len(self._joint_pair_verify_angles_deg)
                    > self.required_stable_steps
                ):
                    self._joint_pair_verify_angles_deg.pop(0)
            else:
                self._joint_pair_verify_angles_deg.clear()

        acquired_before = self.acquired
        aborted_before = self.aborted
        super().observe(
            step,
            evidence,
            cube_position,
            cube_quaternion,
            actual_joint_qpos_rad=actual_joint_qpos_rad,
            operation_feedback=operation_feedback,
        )
        value = self._last_feedback
        if value is None:
            return
        self.joint_pair_vector_cube_m[:] = value.joint_pair_vector_cube_m
        self.joint_pair_residual[:] = value.joint_pair_residual
        self.joint_pair_angle_deg = float(value.joint_pair_angle_deg)
        self.joint_pair_length_m = float(value.joint_pair_length_m)
        self.joint_pair_positive_y = bool(value.joint_pair_positive_y)
        self.joint_pair_valid = bool(value.joint_pair_valid)

        self.joint_pair_abort_risk = False
        if state in (ControlState.MANIPULATE, ControlState.HOLD):
            violation = bool(
                not value.joint_pair_valid
                or not value.joint_pair_positive_y
                or value.joint_pair_length_m
                < self.joint_pair_minimum_length_m - 1e-12
                or value.joint_pair_angle_deg
                > self.joint_pair_freeze_threshold_deg + 1e-12
            )
            self.joint_pair_violation_run_steps = (
                self.joint_pair_violation_run_steps + 1 if violation else 0
            )
            self.joint_pair_maximum_violation_run_steps = max(
                self.joint_pair_maximum_violation_run_steps,
                self.joint_pair_violation_run_steps,
            )
            slip_abort = bool(
                np.any(
                    value.tangent_slip_valid
                    & (
                        value.tangent_slip_from_grasp_m
                        >= self.joint_pair_slip_abort_threshold_m - 1e-12
                    )
                )
            )
            reason = ""
            if value.active_finger_self_collision:
                reason = "active_finger_self_collision"
            elif not value.joint_pair_valid:
                reason = "invalid_geometry"
            elif not value.joint_pair_positive_y:
                reason = "reversed_direction"
            elif (
                value.joint_pair_length_m
                < self.joint_pair_minimum_length_m - 1e-12
            ):
                reason = "minimum_length"
            elif (
                value.joint_pair_angle_deg
                > self.joint_pair_abort_threshold_deg + 1e-12
            ):
                reason = "alignment_angle"
            elif (
                self.joint_pair_violation_run_steps
                > self.joint_pair_max_violation_steps
            ):
                reason = "continuous_alignment_violation"
            elif slip_abort:
                reason = "tangent_slip"
            self.joint_pair_abort_risk = bool(reason)
            if reason:
                if not self.joint_pair_abort_reason:
                    self.joint_pair_abort_reason = reason
                self._abort_for_operation(step)

        if (
            self.aborted
            and not aborted_before
            and not self.joint_pair_abort_reason
        ):
            self.joint_pair_abort_reason = "contact_or_controller_safety"
        if self.acquired and not acquired_before:
            # Preserve the exact aggregate window values that authorized the
            # latch even after the rolling list continues to exist.
            self._additional_verification_window_passed()

    def event_traces(self) -> dict[str, np.ndarray]:
        result = super().event_traces()
        result.update(
            {
                "joint_pair_alignment_id": np.asarray(
                    self.joint_pair_alignment_id, dtype=np.str_
                ),
                "joint_pair_feedback_id": np.asarray(
                    self.joint_pair_feedback_id, dtype=np.str_
                ),
                "joint_pair_joint_names": np.asarray(
                    self.joint_pair_joint_names, dtype=np.str_
                ),
                "joint_pair_grasp_p95_deg": np.asarray(
                    self.joint_pair_grasp_p95_deg, dtype=np.float64
                ),
                "joint_pair_grasp_max_deg": np.asarray(
                    self.joint_pair_grasp_max_observed_deg, dtype=np.float64
                ),
                "joint_pair_maximum_violation_run_steps": np.asarray(
                    self.joint_pair_maximum_violation_run_steps,
                    dtype=np.int64,
                ),
                "joint_pair_abort_reason": np.asarray(
                    self.joint_pair_abort_reason, dtype=np.str_
                ),
                "joint_pair_feedback_headroom_abort_step": np.asarray(
                    self.joint_pair_feedback_headroom_abort_step,
                    dtype=np.int64,
                ),
            }
        )
        return result


class RollingSlipAwareJointPairAlignedContactPreservingPlannedLiftController(
    JointPairAlignedContactPreservingPlannedLiftController
):
    """Schema-v16 signed material-slip recovery with force-PI priority.

    Schema v15 inferred slip from movement of a force centroid.  A rolling
    contact or a collision-manifold point switch can move that centroid even
    when the same material points are not sliding.  V16 therefore consumes
    the rolling-aware estimator's *signed two-dimensional material
    displacement* and its measured tangent Jacobian.  Recovery has separate
    enter/exit and freeze/resume hysteresis, while the cumulative material
    slip remains the hard-abort authority.

    The inherited force PI and joint-pair controller run first.  The tangent
    correction is solved in the measured force-response nullspace, cannot
    command an outward component along a recovering finger's closing ray,
    and is bounded only inside the actuator headroom left by those higher
    priority corrections.  Thus tangent recovery cannot silently clip or
    cancel a normal-force recovery command.
    """

    SCHEMA_VERSION = 16
    STRATEGY = (
        "grasp_verify_then_joint_pair_aligned_rolling_slip_"
        "contact_preserving_planned_lift"
    )

    def __init__(
        self,
        model: mujoco.MjModel,
        config: Mapping[str, Any],
        phase_steps: Mapping[str, int],
    ) -> None:
        super().__init__(model, config, phase_steps)
        feedback = config["joint_pair_feedback"]
        if int(feedback.get("schema_version", 0)) != 2:
            raise ValueError("schema-v16 requires joint-pair feedback schema 2")
        if feedback.get("strategy") != (
            "previous_frame_rolling_aware_signed_tangent_nullspace"
        ):
            raise ValueError("schema-v16 rolling-slip feedback strategy is invalid")

        self.rolling_slip_recovery_enter_threshold_m = float(
            feedback["slip_recovery_enter_threshold_m"]
        )
        self.rolling_slip_recovery_exit_threshold_m = float(
            feedback["slip_recovery_exit_threshold_m"]
        )
        self.rolling_slip_resume_threshold_m = float(
            feedback["slip_resume_threshold_m"]
        )
        self.rolling_slip_freeze_threshold_m = float(
            feedback["slip_freeze_threshold_m"]
        )
        self.rolling_slip_abort_threshold_m = float(
            feedback["slip_abort_threshold_m"]
        )
        self.rolling_slip_exit_dwell_s = float(feedback["slip_exit_dwell_s"])
        self.rolling_slip_exit_dwell_steps = max(
            1, int(round(self.rolling_slip_exit_dwell_s / self.timestep))
        )
        self.rolling_velocity_filter_time_constant_s = float(
            feedback["relative_velocity_filter_time_constant_s"]
        )
        self.rolling_tangent_prediction_horizon_s = float(
            feedback["tangent_prediction_horizon_s"]
        )
        self.suppress_outward_force_pi_during_recovery = bool(
            feedback["suppress_outward_force_pi_during_recovery"]
        )
        if not self.suppress_outward_force_pi_during_recovery:
            raise ValueError(
                "schema-v16 must suppress outward force PI during slip recovery"
            )
        if not (
            0.0 < self.rolling_slip_recovery_exit_threshold_m
            < self.rolling_slip_recovery_enter_threshold_m
            < self.rolling_slip_resume_threshold_m
            < self.rolling_slip_freeze_threshold_m
            < self.rolling_slip_abort_threshold_m
        ):
            raise ValueError("schema-v16 rolling-slip thresholds are invalid")

        # Disable every legacy centroid-slip branch.  This assignment occurs
        # only for the schema-v16 subclass; v14/v15 retain their original
        # conditionals and arithmetic paths unchanged.
        self.slip_feedback_enabled = False
        self.rolling_slip_filtered_velocity_m_s = np.zeros(
            (3, 2), dtype=np.float64
        )
        self.rolling_slip_predicted_displacement_m = np.zeros(
            (3, 2), dtype=np.float64
        )
        self.rolling_slip_predicted_magnitude_m = np.zeros(
            3, dtype=np.float64
        )
        self.rolling_slip_cumulative_m = np.zeros(3, dtype=np.float64)
        self.rolling_slip_observation_valid = np.zeros(3, dtype=bool)
        self.rolling_slip_velocity_filter_initialized = np.zeros(3, dtype=bool)
        self.rolling_slip_recovery_active = np.zeros(3, dtype=bool)
        self.rolling_slip_freeze_active = np.zeros(3, dtype=bool)
        self.rolling_slip_abort_risk = np.zeros(3, dtype=bool)
        self.rolling_slip_exit_run_steps = np.zeros(3, dtype=np.int64)
        self.rolling_slip_maximum_exit_run_steps = np.zeros(3, dtype=np.int64)
        self.rolling_slip_request_rad = np.zeros(model.nu, dtype=np.float64)
        self.rolling_slip_correction_rad = np.zeros(model.nu, dtype=np.float64)
        self.rolling_slip_correction_velocity_rad_s = np.zeros(
            model.nu, dtype=np.float64
        )
        self.rolling_slip_active_jacobian_m_per_rad = np.zeros(
            (3, 2, len(ACTIVE_ACTUATORS)), dtype=np.float64
        )
        self.rolling_slip_feedback_saturated = False
        self.rolling_slip_feedback_headroom_abort_step = -1
        self.rolling_slip_abort_reason = ""
        self.rolling_slip_maximum_cumulative_m = np.zeros(3, dtype=np.float64)
        self.rolling_operation_minimum_force_n = float(
            config["control_protocol"]["grasp_gate"][
                "min_target_face_force_n"
            ]
        )
        self.rolling_native_target_face_effective = np.zeros(3, dtype=bool)
        self.rolling_physical_target_face_effective = np.zeros(3, dtype=bool)
        self.rolling_merged_target_face_effective = np.zeros(3, dtype=bool)
        self.rolling_contact_substitution_active = np.zeros(3, dtype=bool)
        self.rolling_contact_substitution_step_count = np.zeros(
            3, dtype=np.int64
        )

    @staticmethod
    def _without_legacy_centroid_slip(
        value: OperationFeedback,
    ) -> OperationFeedback:
        """Mask the v15 proxy while preserving every rolling-aware field."""

        return replace(
            value,
            tangent_slip_from_grasp_m=np.zeros(3, dtype=np.float64),
            tangent_slip_valid=np.zeros(3, dtype=bool),
        )

    def _with_operation_rolling_contact_evidence(
        self,
        value: OperationFeedback,
        state: ControlState,
    ) -> OperationFeedback:
        """Merge rolling contact only after the strict grasp has latched."""

        rolling, merged = merge_rolling_operation_target_face_effective(
            value,
            minimum_force_n=self.rolling_operation_minimum_force_n,
            minimum_purity=self.minimum_purity,
        )
        legacy = np.asarray(value.target_face_effective, dtype=bool)
        self.rolling_native_target_face_effective[:] = legacy
        self.rolling_physical_target_face_effective[:] = rolling
        operation = state in (ControlState.MANIPULATE, ControlState.HOLD)
        if not operation:
            # VERIFY retains the original native-touch-aware bit.  Recording
            # the physical mask is diagnostic only and cannot advance the
            # stable grasp window.
            self.rolling_merged_target_face_effective[:] = legacy
            self.rolling_contact_substitution_active[:] = False
            return value
        substitution = ~legacy & rolling
        self.rolling_merged_target_face_effective[:] = merged
        self.rolling_contact_substitution_active[:] = substitution
        self.rolling_contact_substitution_step_count[:] += substitution.astype(
            np.int64
        )
        return replace(value, target_face_effective=merged)

    def _update_rolling_slip_observation(
        self, value: OperationFeedback
    ) -> None:
        valid = (
            np.asarray(value.rolling_contact_valid, dtype=bool)
            & np.asarray(value.rolling_contact_continuous, dtype=bool)
            & np.asarray(value.rolling_tangent_jacobian_valid, dtype=bool)
            & np.asarray(value.target_face_effective, dtype=bool)
        )
        raw_velocity = np.asarray(
            value.rolling_relative_tangent_velocity_m_s, dtype=np.float64
        )
        alpha = self.timestep / (
            self.rolling_velocity_filter_time_constant_s + self.timestep
        )
        first = valid & ~self.rolling_slip_velocity_filter_initialized
        continuing = valid & self.rolling_slip_velocity_filter_initialized
        self.rolling_slip_filtered_velocity_m_s[first] = raw_velocity[first]
        self.rolling_slip_filtered_velocity_m_s[continuing] += alpha * (
            raw_velocity[continuing]
            - self.rolling_slip_filtered_velocity_m_s[continuing]
        )
        invalid = ~valid
        self.rolling_slip_filtered_velocity_m_s[invalid] = 0.0
        self.rolling_slip_velocity_filter_initialized[:] = valid

        signed = np.asarray(
            value.rolling_signed_tangent_displacement_m, dtype=np.float64
        )
        predicted = signed + (
            self.rolling_tangent_prediction_horizon_s
            * self.rolling_slip_filtered_velocity_m_s
        )
        predicted[~valid] = 0.0
        self.rolling_slip_predicted_displacement_m[:] = predicted
        self.rolling_slip_predicted_magnitude_m[:] = np.linalg.norm(
            predicted, axis=1
        )
        self.rolling_slip_cumulative_m[:] = np.asarray(
            value.rolling_cumulative_irrecoverable_slip_m,
            dtype=np.float64,
        )
        self.rolling_slip_maximum_cumulative_m[:] = np.maximum(
            self.rolling_slip_maximum_cumulative_m,
            self.rolling_slip_cumulative_m,
        )
        self.rolling_slip_observation_valid[:] = valid
        self.rolling_slip_active_jacobian_m_per_rad[:] = np.asarray(
            value.rolling_tangent_jacobian_m_per_rad, dtype=np.float64
        )

        for finger_index in range(3):
            if not valid[finger_index]:
                self.rolling_slip_exit_run_steps[finger_index] = 0
                continue
            magnitude = self.rolling_slip_predicted_magnitude_m[finger_index]
            if not self.rolling_slip_recovery_active[finger_index]:
                if (
                    magnitude
                    >= self.rolling_slip_recovery_enter_threshold_m - 1e-12
                ):
                    self.rolling_slip_recovery_active[finger_index] = True
            if self.rolling_slip_recovery_active[finger_index]:
                if (
                    magnitude
                    <= self.rolling_slip_recovery_exit_threshold_m + 1e-12
                ):
                    self.rolling_slip_exit_run_steps[finger_index] += 1
                    self.rolling_slip_maximum_exit_run_steps[finger_index] = max(
                        self.rolling_slip_maximum_exit_run_steps[finger_index],
                        self.rolling_slip_exit_run_steps[finger_index],
                    )
                    if (
                        self.rolling_slip_exit_run_steps[finger_index]
                        >= self.rolling_slip_exit_dwell_steps
                    ):
                        self.rolling_slip_recovery_active[finger_index] = False
                        self.rolling_slip_exit_run_steps[finger_index] = 0
                else:
                    self.rolling_slip_exit_run_steps[finger_index] = 0

            if self.rolling_slip_freeze_active[finger_index]:
                if magnitude <= self.rolling_slip_resume_threshold_m + 1e-12:
                    self.rolling_slip_freeze_active[finger_index] = False
            elif magnitude >= self.rolling_slip_freeze_threshold_m - 1e-12:
                self.rolling_slip_freeze_active[finger_index] = True

        self.rolling_slip_abort_risk[:] = (
            self.rolling_slip_cumulative_m
            >= self.rolling_slip_abort_threshold_m - 1e-12
        ) | (
            valid
            & (
                self.rolling_slip_predicted_magnitude_m
                >= self.rolling_slip_abort_threshold_m - 1e-12
            )
        )

    def _feedback_is_at_risk(self) -> bool:
        inherited = super()._feedback_is_at_risk()
        return bool(inherited or np.any(self.rolling_slip_freeze_active))

    def _slip_recovery_request(self) -> np.ndarray:
        # The v15 unsigned inward-only proxy is intentionally unreachable in
        # v16.  Signed material recovery is added after the force PI and pair
        # correction have consumed their command headroom.
        return np.zeros(self.model.nu, dtype=np.float64)

    def _update_feedback_correction(self) -> np.ndarray:
        """Keep normal-force PI inward/non-negative during slip recovery."""

        active = np.asarray(self.rolling_slip_recovery_active, dtype=bool)
        if not self.suppress_outward_force_pi_during_recovery or not np.any(active):
            return super()._update_feedback_correction()
        filtered_force = self.filtered_force_n.copy()
        self.filtered_force_n[active] = np.minimum(
            self.filtered_force_n[active], self.force_targets_n[active]
        )
        # A negative integral is historical unloading.  Retaining it while
        # material slip is active would keep requesting outward motion even
        # after the proportional term is clamped to zero.
        self.force_integral_n_s[active] = np.maximum(
            self.force_integral_n_s[active], 0.0
        )
        try:
            return super()._update_feedback_correction()
        finally:
            self.filtered_force_n[:] = filtered_force

    def _rolling_force_nullspace_projector(self) -> np.ndarray:
        knot = int(np.clip(self.planned_knot_index, 0, self.knot_times_s.size - 1))
        force = (
            self.joint_pair_force_response_weight
            * self.joint_pair_plan_force_jacobian_active[knot]
        )
        identity = np.eye(len(ACTIVE_ACTUATORS), dtype=np.float64)
        gram = force @ force.T
        return identity - force.T @ np.linalg.solve(
            gram + self.joint_pair_damping * np.eye(force.shape[0]),
            force,
        )

    def _signed_rolling_slip_request(self) -> np.ndarray:
        result = np.zeros(self.model.nu, dtype=np.float64)
        value = self._last_feedback
        usable = (
            self.rolling_slip_recovery_active
            & self.rolling_slip_observation_valid
        )
        if value is None or not np.any(usable):
            return result

        projector = self._rolling_force_nullspace_projector()
        active_request = np.zeros(len(ACTIVE_ACTUATORS), dtype=np.float64)
        for finger_index in np.flatnonzero(usable):
            jacobian = self.rolling_slip_active_jacobian_m_per_rad[
                finger_index
            ]
            effective = jacobian @ projector
            gram = effective @ effective.T
            direction = -(
                projector
                @ effective.T
                @ np.linalg.solve(
                    gram + self.joint_pair_damping * np.eye(2),
                    self.rolling_slip_predicted_displacement_m[finger_index],
                )
            )
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1e-14:
                continue
            magnitude = (
                self.joint_pair_slip_recovery_gain_rad_per_m
                * self.rolling_slip_predicted_magnitude_m[finger_index]
            )
            active_request += magnitude * direction / direction_norm

        # The tangent solver is lower priority than normal-force PI.  It may
        # use a direction orthogonal to, or further inward along, each
        # recovering finger's closing ray, but may never unload that finger.
        inward_active = self.inward_direction[:, self._active_actuator_ids]
        for finger_index in np.flatnonzero(usable):
            direction = inward_active[finger_index]
            denominator = float(direction @ direction)
            if denominator <= 1e-14:
                continue
            outward_component = float(active_request @ direction)
            if outward_component < 0.0:
                active_request -= (
                    outward_component / denominator
                ) * direction

        result[self._active_actuator_ids] = active_request
        return result

    def _bounded_rolling_slip_correction(
        self, desired: np.ndarray, base_target: np.ndarray
    ) -> np.ndarray | None:
        lower = np.maximum(
            -self.joint_pair_correction_limit_rad,
            self.feedback_command_lower_rad - base_target,
        )
        upper = np.minimum(
            self.joint_pair_correction_limit_rad,
            self.feedback_command_upper_rad - base_target,
        )
        lower[self._inactive_actuator_ids] = 0.0
        upper[self._inactive_actuator_ids] = 0.0
        if np.any(
            (self.rolling_slip_correction_rad < lower - 1e-12)
            | (self.rolling_slip_correction_rad > upper + 1e-12)
        ):
            return None
        clipped_desired = np.minimum(np.maximum(desired, lower), upper)
        self.rolling_slip_feedback_saturated = bool(
            not np.allclose(clipped_desired, desired, rtol=0.0, atol=1e-14)
        )
        acceleration = self.joint_pair_acceleration_limit_rad_s2

        def stopping_speed(distance: np.ndarray) -> np.ndarray:
            nonnegative = np.maximum(np.asarray(distance), 0.0)
            return np.maximum(
                0.0,
                -acceleration * self.timestep
                + np.sqrt(
                    (acceleration * self.timestep) ** 2
                    + 2.0 * acceleration * nonnegative
                ),
            )

        error = clipped_desired - self.rolling_slip_correction_rad
        desired_velocity = np.sign(error) * np.minimum(
            self.joint_pair_rate_limit_rad_s,
            stopping_speed(np.abs(error)),
        )
        velocity_delta = np.clip(
            desired_velocity - self.rolling_slip_correction_velocity_rad_s,
            -acceleration * self.timestep,
            acceleration * self.timestep,
        )
        next_velocity = np.clip(
            self.rolling_slip_correction_velocity_rad_s + velocity_delta,
            -self.joint_pair_rate_limit_rad_s,
            self.joint_pair_rate_limit_rad_s,
        )
        next_velocity = np.clip(
            next_velocity,
            -stopping_speed(self.rolling_slip_correction_rad - lower),
            stopping_speed(upper - self.rolling_slip_correction_rad),
        )
        updated = self.rolling_slip_correction_rad + (
            next_velocity * self.timestep
        )
        bounded = np.minimum(np.maximum(updated, lower), upper)
        self.rolling_slip_correction_velocity_rad_s[:] = (
            bounded - self.rolling_slip_correction_rad
        ) / self.timestep
        self.rolling_slip_correction_rad[:] = bounded
        return bounded.copy()

    def command(self, step: int) -> ControlCommand:
        previous_sent_target = self._last_sent_target.copy()
        base = super().command(step)
        if base.state not in (ControlState.MANIPULATE, ControlState.HOLD):
            self.rolling_slip_request_rad[:] = 0.0
            if base.state is not ControlState.ABORT:
                self.rolling_slip_correction_rad[:] = 0.0
                self.rolling_slip_correction_velocity_rad_s[:] = 0.0
            return base

        self.rolling_slip_request_rad[:] = self._signed_rolling_slip_request()
        desired = np.clip(
            self.rolling_slip_request_rad,
            -self.joint_pair_correction_limit_rad,
            self.joint_pair_correction_limit_rad,
        )
        desired[self._inactive_actuator_ids] = 0.0
        correction = self._bounded_rolling_slip_correction(desired, base.target)
        if correction is None:
            self._abort_for_operation(step)
            self.rolling_slip_feedback_headroom_abort_step = int(step)
            self.rolling_slip_abort_reason = "feedback_headroom"
            self._last_command_state = ControlState.ABORT
            self._last_sent_target = previous_sent_target.copy()
            return ControlCommand(
                state=ControlState.ABORT,
                target=previous_sent_target.copy(),
                manipulation_progress=float(self.current_plan_progress),
                target_velocity_rad_s=np.zeros(self.model.nu, dtype=np.float64),
            )
        target = base.target + correction
        target[self._inactive_actuator_ids] = 0.0
        velocity = (target - previous_sent_target) / self.timestep
        self._last_sent_target = target.copy()
        return ControlCommand(
            state=base.state,
            target=target.copy(),
            manipulation_progress=base.manipulation_progress,
            target_velocity_rad_s=velocity,
        )

    def observe(
        self,
        step: int,
        evidence: GraspGateEvidence,
        cube_position: np.ndarray,
        cube_quaternion: np.ndarray,
        actual_joint_qpos_rad: np.ndarray | None = None,
        operation_feedback: OperationFeedback | None = None,
    ) -> None:
        if operation_feedback is None:
            raise ValueError("schema-v16 requires rolling-aware operation feedback")
        state = self._last_command_state
        effective_feedback = self._with_operation_rolling_contact_evidence(
            operation_feedback, state
        )
        sanitized = self._without_legacy_centroid_slip(effective_feedback)
        super().observe(
            step,
            evidence,
            cube_position,
            cube_quaternion,
            actual_joint_qpos_rad=actual_joint_qpos_rad,
            operation_feedback=sanitized,
        )
        self._update_rolling_slip_observation(effective_feedback)
        if (
            state in (ControlState.MANIPULATE, ControlState.HOLD)
            and np.any(self.rolling_slip_abort_risk)
        ):
            if not self.rolling_slip_abort_reason:
                self.rolling_slip_abort_reason = "rolling_material_slip"
            self._abort_for_operation(step)

    def event_traces(self) -> dict[str, np.ndarray]:
        result = super().event_traces()
        result.update(
            {
                "rolling_slip_feedback_id": np.asarray(
                    self.joint_pair_feedback_id, dtype=np.str_
                ),
                "rolling_slip_maximum_cumulative_m": (
                    self.rolling_slip_maximum_cumulative_m.copy()
                ),
                "rolling_slip_maximum_exit_run_steps": (
                    self.rolling_slip_maximum_exit_run_steps.copy()
                ),
                "rolling_slip_abort_reason": np.asarray(
                    self.rolling_slip_abort_reason, dtype=np.str_
                ),
                "rolling_slip_feedback_headroom_abort_step": np.asarray(
                    self.rolling_slip_feedback_headroom_abort_step,
                    dtype=np.int64,
                ),
                "rolling_native_target_face_effective": (
                    self.rolling_native_target_face_effective.copy()
                ),
                "rolling_physical_target_face_effective": (
                    self.rolling_physical_target_face_effective.copy()
                ),
                "rolling_merged_target_face_effective": (
                    self.rolling_merged_target_face_effective.copy()
                ),
                "rolling_contact_substitution_active": (
                    self.rolling_contact_substitution_active.copy()
                ),
                "rolling_contact_substitution_step_count": (
                    self.rolling_contact_substitution_step_count.copy()
                ),
            }
        )
        return result


def build_grasp_controller(
    model: mujoco.MjModel,
    config: Mapping[str, Any],
    phase_steps: Mapping[str, int],
) -> GraspVerifyThenManipulateController:
    """Select a schema-gated controller without altering legacy behavior."""

    strategy = str(config["control_protocol"]["strategy"])
    if int(config.get("schema_version", 1)) == 16:
        if strategy != (
            RollingSlipAwareJointPairAlignedContactPreservingPlannedLiftController.STRATEGY
        ):
            raise ValueError("schema-v16 requires the rolling-slip-aware strategy")
        return RollingSlipAwareJointPairAlignedContactPreservingPlannedLiftController(
            model, config, phase_steps
        )
    if int(config.get("schema_version", 1)) == 15:
        if strategy != JointPairAlignedContactPreservingPlannedLiftController.STRATEGY:
            raise ValueError("schema-v15 requires the joint-pair-aligned strategy")
        return JointPairAlignedContactPreservingPlannedLiftController(
            model, config, phase_steps
        )
    if int(config.get("schema_version", 1)) == 14:
        if strategy != ContactPreservingPlannedLiftController.STRATEGY:
            raise ValueError("schema-v14 requires the contact-preserving strategy")
        return ContactPreservingPlannedLiftController(model, config, phase_steps)
    return GraspVerifyThenManipulateController(model, config, phase_steps)
