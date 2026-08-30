"""Versioned experiment definitions and deterministic experiment discovery.

The simulation implementation is intentionally kept separate from this module.
An :class:`ExperimentDefinition` describes what should be evaluated and searched;
the JSON configuration contains one concrete candidate for that experiment.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any


FINGERS = ("thumb", "index", "mid")
FACES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
OPPOSITE_FACES = MappingProxyType(
    {"+X": "-X", "-X": "+X", "+Y": "-Y", "-Y": "+Y", "+Z": "-Z", "-Z": "+Z"}
)

ACTIVE_ACTUATORS = (
    "left_hand_thumb_bend_joint_actuator",
    "left_hand_thumb_rota_joint1_actuator",
    "left_hand_thumb_rota_joint2_actuator",
    "left_hand_index_bend_joint_actuator",
    "left_hand_index_joint1_actuator",
    "left_hand_index_joint2_actuator",
    "left_hand_mid_joint1_actuator",
    "left_hand_mid_joint2_actuator",
)

# The original palm-down v2 experiment keeps its terminal pose near the
# statically screened pregrasp.  Experiments that need independent absolute
# endpoint sampling (the large-cube campaign does) set the corresponding
# SearchBounds field to ``None`` explicitly.
DEFAULT_FINAL_TARGET_DELTA_BOUNDS_RAD = MappingProxyType(
    {
        "left_hand_thumb_bend_joint_actuator": (-0.35, 0.12),
        "left_hand_thumb_rota_joint1_actuator": (-0.05, 0.50),
        "left_hand_thumb_rota_joint2_actuator": (-0.05, 0.40),
        "left_hand_index_bend_joint_actuator": (-0.05, 0.15),
        "left_hand_index_joint1_actuator": (-0.30, 0.12),
        "left_hand_index_joint2_actuator": (0.00, 0.50),
        "left_hand_mid_joint1_actuator": (-0.30, 0.12),
        "left_hand_mid_joint2_actuator": (0.00, 0.60),
    }
)

# Schema-v3 experiments search a stable grasp pose first and only then sample a
# bounded *relative* manipulation command.  This is deliberately a separate
# contract from ``final_target_delta_rad``: the latter belongs to the legacy
# open-loop pregrasp/final trajectory and must never be mistaken for a command
# that is gated on a verified grasp.
DEFAULT_MANIPULATION_DELTA_BOUNDS_RAD = MappingProxyType(
    {
        "left_hand_thumb_bend_joint_actuator": (-0.20, 0.12),
        "left_hand_thumb_rota_joint1_actuator": (-0.05, 0.30),
        "left_hand_thumb_rota_joint2_actuator": (-0.05, 0.30),
        "left_hand_index_bend_joint_actuator": (-0.05, 0.10),
        "left_hand_index_joint1_actuator": (-0.25, 0.10),
        "left_hand_index_joint2_actuator": (0.00, 0.40),
        "left_hand_mid_joint1_actuator": (-0.25, 0.10),
        "left_hand_mid_joint2_actuator": (0.00, 0.45),
    }
)

DEFAULT_V1_EXPERIMENT_ID = "left_three_finger_cube"
_EXPERIMENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
TUNING_STRATEGIES = (
    "default",
    "relative_pose_rescue",
    "aligned_contacts",
    "far_hand_fingertip",
    "pose_preserving_grasp",
    "high_thumb_variable_size_pose_preserving",
    "normal_aligned_smooth_vertical_lift",
    "actual_contact_grasp_pose_smooth_vertical_lift",
    "contact_point_targeted_actual_grasp_pose",
    "scaled_contact_downsize_actual_grasp_then_lift",
    "contact_preserving_planned_lift",
    "joint_pair_near_zero_contact_preserving_planned_lift",
    "joint_pair_near_zero_rolling_slip_contact_preserving_planned_lift",
)


def opposite_face(face: str) -> str:
    """Return the exact opposite of a signed local cube face."""

    try:
        return OPPOSITE_FACES[face]
    except KeyError as error:
        raise ValueError(f"unknown cube face {face!r}; expected one of {FACES}") from error


@dataclass(frozen=True)
class OpposedFaceAssignment:
    """A thumb-versus-two-fingers cube-face assignment.

    The index and middle finger must share one face and the thumb must use its
    exact opposite.  Keeping this invariant in the value type prevents an
    invalid topology from entering either evaluation or search configuration.
    """

    thumb: str
    index: str
    mid: str

    def __post_init__(self) -> None:
        for finger in FINGERS:
            face = getattr(self, finger)
            if face not in FACES:
                raise ValueError(f"{finger} face must be one of {FACES}, got {face!r}")
        if self.index != self.mid:
            raise ValueError("index and mid must target the same cube face")
        if self.thumb != opposite_face(self.index):
            raise ValueError("thumb must target the exact opposite of index and mid")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OpposedFaceAssignment":
        if set(value) != set(FINGERS):
            raise ValueError(f"target faces must contain exactly {FINGERS}")
        return cls(**{finger: str(value[finger]) for finger in FINGERS})

    def as_dict(self) -> dict[str, str]:
        return {finger: getattr(self, finger) for finger in FINGERS}


# A short public alias is convenient for callers that do not need to mention
# the topology in the type name.
FaceAssignment = OpposedFaceAssignment


def _finite_number(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _closed_range(value: Sequence[float], label: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{label} must contain [minimum, maximum]")
    lower = _finite_number(value[0], f"{label}.minimum")
    upper = _finite_number(value[1], f"{label}.maximum")
    if lower > upper:
        raise ValueError(f"{label} minimum must not exceed its maximum")
    return (lower, upper)


def _frozen_ranges(
    values: Mapping[str, Sequence[float]], label: str
) -> Mapping[str, tuple[float, float]]:
    return MappingProxyType(
        {str(name): _closed_range(bounds, f"{label}.{name}") for name, bounds in values.items()}
    )


@dataclass(frozen=True)
class GraspGateSettings:
    """Versioned conditions that must remain true before manipulation starts."""

    min_target_face_force_n: float
    min_target_force_fraction: float
    require_touch: bool
    require_support_contact: bool
    max_translation_m: float
    max_orientation_drift_deg: float
    max_linear_speed_m_s: float
    max_early_lift_m: float

    def __post_init__(self) -> None:
        for label in (
            "min_target_face_force_n",
            "max_translation_m",
            "max_orientation_drift_deg",
            "max_linear_speed_m_s",
            "max_early_lift_m",
        ):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        fraction = _finite_number(
            self.min_target_force_fraction, "min_target_force_fraction"
        )
        if not 0.0 < fraction <= 1.0:
            raise ValueError("min_target_force_fraction must be within (0, 1]")
        object.__setattr__(self, "min_target_force_fraction", fraction)
        for label in ("require_touch", "require_support_contact"):
            if not isinstance(getattr(self, label), bool):
                raise ValueError(f"{label} must be boolean")

    def as_config(self) -> dict[str, Any]:
        return {
            "min_target_face_force_n": self.min_target_face_force_n,
            "min_target_force_fraction": self.min_target_force_fraction,
            "require_touch": self.require_touch,
            "require_support_contact": self.require_support_contact,
            "max_translation_m": self.max_translation_m,
            "max_orientation_drift_deg": self.max_orientation_drift_deg,
            "max_linear_speed_m_s": self.max_linear_speed_m_s,
            "max_early_lift_m": self.max_early_lift_m,
        }


@dataclass(frozen=True)
class ControlProtocolSettings:
    """Feedback-gated schema-v3 controller and its fixed phase schedule."""

    strategy: str
    failure_behavior: str
    settle_s: float
    close_s: float
    verify_timeout_s: float
    stable_window_s: float
    manipulate_s: float
    min_hold_s: float
    grasp_gate: GraspGateSettings
    manipulation_profile: str | None = None
    close_duration_options_s: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.strategy not in (
            "grasp_verify_then_manipulate",
            "grasp_verify_then_contact_preserving_planned_lift",
            "grasp_verify_then_joint_pair_aligned_contact_preserving_planned_lift",
            "grasp_verify_then_joint_pair_aligned_rolling_slip_contact_preserving_planned_lift",
        ):
            raise ValueError(
                "unsupported feedback-gated control strategy"
            )
        if self.failure_behavior != "abort_hold_grasp_pose":
            raise ValueError(
                "schema-v3 failure behavior must be abort_hold_grasp_pose"
            )
        for label in (
            "settle_s",
            "close_s",
            "verify_timeout_s",
            "stable_window_s",
            "manipulate_s",
            "min_hold_s",
        ):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        if self.stable_window_s > self.verify_timeout_s:
            raise ValueError("stable_window_s must not exceed verify_timeout_s")
        if not isinstance(self.grasp_gate, GraspGateSettings):
            raise TypeError("grasp_gate must be GraspGateSettings")
        if self.manipulation_profile not in (None, "minimum_jerk_quintic"):
            raise ValueError("unsupported manipulation_profile")
        if self.close_duration_options_s is not None:
            options = tuple(
                _finite_number(value, "close_duration_options_s")
                for value in self.close_duration_options_s
            )
            if (
                not options
                or any(value <= 0.0 for value in options)
                or tuple(sorted(set(options))) != options
                or not any(
                    math.isclose(self.close_s, value, abs_tol=1e-12)
                    for value in options
                )
            ):
                raise ValueError(
                    "close_duration_options_s must be positive, unique, increasing "
                    "and include close_s"
                )
            object.__setattr__(self, "close_duration_options_s", options)

    @property
    def total_duration_s(self) -> float:
        """Maximum run duration; early verification time is reassigned to hold."""

        return (
            self.settle_s
            + self.close_s
            + self.verify_timeout_s
            + self.manipulate_s
            + self.min_hold_s
        )

    def as_config(self) -> dict[str, Any]:
        result = {
            "strategy": self.strategy,
            "failure_behavior": self.failure_behavior,
            "settle_s": self.settle_s,
            "close_s": self.close_s,
            "verify_timeout_s": self.verify_timeout_s,
            "stable_window_s": self.stable_window_s,
            "manipulate_s": self.manipulate_s,
            "min_hold_s": self.min_hold_s,
            "grasp_gate": self.grasp_gate.as_config(),
        }
        if self.manipulation_profile is not None:
            result["manipulation_profile"] = self.manipulation_profile
        if self.close_duration_options_s is not None:
            result["close_duration_options_s"] = list(
                self.close_duration_options_s
            )
        return result


@dataclass(frozen=True)
class PoseConstraints:
    """Schema-v4 palm pose envelope anchored in world coordinates.

    The two angular ranges are deliberately named for the geometric quantities
    that the evaluator measures.  They are not aliases for Euler angles.  The
    reference translation is the measured 62 mm solution from which press
    depth is applied along the resolved palm normal.
    """

    finger_down_tilt_deg: tuple[float, float]
    palm_plane_ground_angle_deg: tuple[float, float]
    palm_press_depth_m: tuple[float, float]
    reference_hand_translation_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        for label in (
            "finger_down_tilt_deg",
            "palm_plane_ground_angle_deg",
        ):
            bounds = _closed_range(getattr(self, label), label)
            if bounds[0] < 0.0 or bounds[1] > 90.0:
                raise ValueError(f"{label} must stay within [0, 90]")
            object.__setattr__(self, label, bounds)
        press = _closed_range(self.palm_press_depth_m, "palm_press_depth_m")
        if press[0] < 0.0 or math.isclose(press[0], press[1]):
            raise ValueError("palm_press_depth_m must be a non-empty non-negative range")
        object.__setattr__(self, "palm_press_depth_m", press)
        translation = tuple(
            _finite_number(value, "reference_hand_translation_m")
            for value in self.reference_hand_translation_m
        )
        if len(translation) != 3:
            raise ValueError("reference_hand_translation_m must contain three values")
        object.__setattr__(self, "reference_hand_translation_m", translation)

    def as_config(self) -> dict[str, Any]:
        return {
            "finger_down_tilt_deg": list(self.finger_down_tilt_deg),
            "palm_plane_ground_angle_deg": list(
                self.palm_plane_ground_angle_deg
            ),
            "palm_press_depth_m": list(self.palm_press_depth_m),
            "reference_hand_translation_m": list(
                self.reference_hand_translation_m
            ),
        }


# Both names read naturally at call sites; retain the longer form as a public
# compatibility alias for code that names all experiment settings uniformly.
PoseConstraintSettings = PoseConstraints


@dataclass(frozen=True)
class FarHandPoseConstraints:
    """Schema-v5 pose envelope expressed directly in the hand-root frame.

    Unlike :class:`PoseConstraints`, this contract does not solve or constrain
    a world-Z press depth.  The hand root is derived from the complete
    cube-in-root vector and is fixed after model compilation.  The legacy
    reference is retained only so reports can explain how far the new root is
    from the former press-based campaign.
    """

    finger_down_tilt_deg: tuple[float, float]
    palm_plane_ground_angle_deg: tuple[float, float]
    root_cube_distance_m: tuple[float, float]
    cube_position_in_root_m: Mapping[str, tuple[float, float]]
    legacy_press_reference_translation_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        for label in (
            "finger_down_tilt_deg",
            "palm_plane_ground_angle_deg",
        ):
            bounds = _closed_range(getattr(self, label), label)
            if bounds[0] < 0.0 or bounds[1] > 90.0:
                raise ValueError(f"{label} must stay within [0, 90]")
            object.__setattr__(self, label, bounds)
        distance = _closed_range(
            self.root_cube_distance_m, "root_cube_distance_m"
        )
        if distance[0] <= 0.0 or math.isclose(distance[0], distance[1]):
            raise ValueError("root_cube_distance_m must be a non-empty positive range")
        object.__setattr__(self, "root_cube_distance_m", distance)
        cube_ranges = _frozen_ranges(
            self.cube_position_in_root_m, "cube_position_in_root_m"
        )
        if set(cube_ranges) != {"x", "y", "z"}:
            raise ValueError(
                "cube_position_in_root_m must contain exactly x, y and z"
            )
        object.__setattr__(self, "cube_position_in_root_m", cube_ranges)
        reference = tuple(
            _finite_number(value, "legacy_press_reference_translation_m")
            for value in self.legacy_press_reference_translation_m
        )
        if len(reference) != 3:
            raise ValueError(
                "legacy_press_reference_translation_m must contain three values"
            )
        object.__setattr__(
            self, "legacy_press_reference_translation_m", reference
        )

    def contains_cube_position(self, xyz: Sequence[float]) -> bool:
        if len(xyz) != 3:
            return False
        vector = tuple(float(value) for value in xyz)
        if not all(math.isfinite(value) for value in vector):
            return False
        inside_axes = all(
            self.cube_position_in_root_m[axis][0] - 1e-12
            <= value
            <= self.cube_position_in_root_m[axis][1] + 1e-12
            for axis, value in zip(("x", "y", "z"), vector)
        )
        distance = math.sqrt(sum(value * value for value in vector))
        lower, upper = self.root_cube_distance_m
        return inside_axes and lower - 1e-12 <= distance <= upper + 1e-12

    def as_config(self) -> dict[str, Any]:
        return {
            "finger_down_tilt_deg": list(self.finger_down_tilt_deg),
            "palm_plane_ground_angle_deg": list(
                self.palm_plane_ground_angle_deg
            ),
            "root_cube_distance_m": list(self.root_cube_distance_m),
            "cube_position_in_root_m": {
                axis: list(self.cube_position_in_root_m[axis])
                for axis in ("x", "y", "z")
            },
            "legacy_press_reference_translation_m": list(
                self.legacy_press_reference_translation_m
            ),
        }


@dataclass(frozen=True)
class FingertipContactPreferences:
    """Soft schema-v5 fingertip objective; it never weakens hard topology."""

    taxel_assignment_max_distance_m: float
    ranking_policy: str = "soft_maximize_force_fraction_then_taxel_coverage"
    hard_acceptance: bool = False

    def __post_init__(self) -> None:
        distance = _finite_number(
            self.taxel_assignment_max_distance_m,
            "taxel_assignment_max_distance_m",
        )
        if distance <= 0.0:
            raise ValueError("taxel_assignment_max_distance_m must be positive")
        object.__setattr__(self, "taxel_assignment_max_distance_m", distance)
        if self.ranking_policy != (
            "soft_maximize_force_fraction_then_taxel_coverage"
        ):
            raise ValueError("unsupported fingertip ranking policy")
        if self.hard_acceptance is not False:
            raise ValueError("fingertip preference must remain a soft objective")

    def as_config(self) -> dict[str, Any]:
        return {
            "taxel_assignment_max_distance_m": (
                self.taxel_assignment_max_distance_m
            ),
            "ranking_policy": self.ranking_policy,
            "hard_acceptance": self.hard_acceptance,
        }


@dataclass(frozen=True)
class ContactAlignmentSettings:
    """Schema-v4 three-finger contact-height alignment contract."""

    max_height_spread_m: float
    verify_continuous_s: float
    operation_aligned_duty: float

    def __post_init__(self) -> None:
        for label in ("max_height_spread_m", "verify_continuous_s"):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        duty = _finite_number(
            self.operation_aligned_duty, "operation_aligned_duty"
        )
        if not 0.0 <= duty <= 1.0:
            raise ValueError("operation_aligned_duty must be within [0, 1]")
        object.__setattr__(self, "operation_aligned_duty", duty)

    def as_config(self) -> dict[str, float]:
        return {
            "max_height_spread_m": self.max_height_spread_m,
            "verify_continuous_s": self.verify_continuous_s,
            "operation_aligned_duty": self.operation_aligned_duty,
        }


ContactAlignment = ContactAlignmentSettings


@dataclass(frozen=True)
class ClosureAlignmentSettings:
    """Schema-v8 command-velocity alignment with the cube contact normal."""

    static_max_angle_deg: float
    dynamic_p95_max_angle_deg: float
    optimization_target_max_angle_deg: float
    min_inward_speed_m_s: float
    min_contact_force_n: float

    def __post_init__(self) -> None:
        angles = []
        for label in (
            "static_max_angle_deg",
            "dynamic_p95_max_angle_deg",
            "optimization_target_max_angle_deg",
        ):
            value = _finite_number(getattr(self, label), label)
            if not 0.0 < value < 180.0:
                raise ValueError(f"{label} must be within (0, 180)")
            object.__setattr__(self, label, value)
            angles.append(value)
        if not angles[2] <= angles[1] <= angles[0]:
            raise ValueError(
                "closure angles must satisfy optimization <= dynamic <= static"
            )
        for label in ("min_inward_speed_m_s", "min_contact_force_n"):
            value = _finite_number(getattr(self, label), label)
            if value < 0.0 or (label == "min_contact_force_n" and value <= 0.0):
                raise ValueError(f"{label} must be non-negative")
            object.__setattr__(self, label, value)

    def as_config(self) -> dict[str, Any]:
        return {
            "static_max_angle_deg": self.static_max_angle_deg,
            "dynamic_p95_max_angle_deg": self.dynamic_p95_max_angle_deg,
            "optimization_target_max_angle_deg": (
                self.optimization_target_max_angle_deg
            ),
            "min_inward_speed_m_s": self.min_inward_speed_m_s,
            "min_contact_force_n": self.min_contact_force_n,
            "require_positive_inward_speed": True,
            "measurement_phase": "close_target_face_contacts",
        }


@dataclass(frozen=True)
class MotionSmoothnessSettings:
    """Schema-v8 near-vertical, minimum-jerk motion acceptance."""

    filter_window_s: float
    downward_speed_threshold_m_s: float
    max_downward_speed_duty: float
    max_cumulative_backtrack_m: float
    max_peak_upward_speed_m_s: float
    max_abs_acceleration_m_s2: float
    max_abs_jerk_m_s3: float
    max_hold_entry_linear_speed_m_s: float
    max_lateral_displacement_m: float
    max_operation_orientation_drift_deg: float

    def __post_init__(self) -> None:
        positive = (
            "filter_window_s",
            "downward_speed_threshold_m_s",
            "max_cumulative_backtrack_m",
            "max_peak_upward_speed_m_s",
            "max_abs_acceleration_m_s2",
            "max_abs_jerk_m_s3",
            "max_hold_entry_linear_speed_m_s",
            "max_lateral_displacement_m",
            "max_operation_orientation_drift_deg",
        )
        for label in positive:
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        duty = _finite_number(
            self.max_downward_speed_duty, "max_downward_speed_duty"
        )
        if not 0.0 <= duty <= 1.0:
            raise ValueError("max_downward_speed_duty must be within [0, 1]")
        object.__setattr__(self, "max_downward_speed_duty", duty)

    def as_config(self) -> dict[str, Any]:
        return {
            "filter_window_s": self.filter_window_s,
            "downward_speed_threshold_m_s": -self.downward_speed_threshold_m_s,
            "max_downward_speed_duty": self.max_downward_speed_duty,
            "max_cumulative_height_backtrack_m": (
                self.max_cumulative_backtrack_m
            ),
            "max_peak_upward_speed_m_s": self.max_peak_upward_speed_m_s,
            "max_abs_vertical_acceleration_m_s2": (
                self.max_abs_acceleration_m_s2
            ),
            "max_abs_vertical_jerk_m_s3": self.max_abs_jerk_m_s3,
            "max_hold_entry_linear_speed_m_s": (
                self.max_hold_entry_linear_speed_m_s
            ),
            "max_lateral_displacement_m": self.max_lateral_displacement_m,
            "max_orientation_drift_deg": (
                self.max_operation_orientation_drift_deg
            ),
        }


@dataclass(frozen=True)
class ActualContactGraspPoseSettings:
    """Schema-v9 acceptance for a measured, stable contact configuration.

    The actuator preload is deliberately not represented here.  A grasp pose
    is the measured eight-joint configuration during the latched contact
    window, not the position-controller command that happened to produce it.
    """

    thumb_actual_range_rad: tuple[float, float]
    max_nominal_joint_error_rad: float
    max_joint_stability_span_rad: float
    verify_continuous_s: float
    thumb_bend_actuator: str = "left_hand_thumb_bend_joint_actuator"
    qpos_reference: str = "stable_contact_window_actual_qpos"
    require_all_thumb_samples_in_range: bool = True

    def __post_init__(self) -> None:
        bounds = _closed_range(
            self.thumb_actual_range_rad, "thumb_actual_range_rad"
        )
        if bounds[0] < 0.0 or math.isclose(bounds[0], bounds[1]):
            raise ValueError(
                "thumb_actual_range_rad must be a non-empty non-negative range"
            )
        object.__setattr__(self, "thumb_actual_range_rad", bounds)
        for label in (
            "max_nominal_joint_error_rad",
            "max_joint_stability_span_rad",
            "verify_continuous_s",
        ):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        if self.thumb_bend_actuator not in ACTIVE_ACTUATORS:
            raise ValueError("thumb_bend_actuator must name an active actuator")
        if self.thumb_bend_actuator != "left_hand_thumb_bend_joint_actuator":
            raise ValueError("thumb_bend_actuator must name the thumb bend actuator")
        if self.qpos_reference != "stable_contact_window_actual_qpos":
            raise ValueError(
                "qpos_reference must be stable_contact_window_actual_qpos"
            )
        if self.require_all_thumb_samples_in_range is not True:
            raise ValueError(
                "schema-v9 requires every stable-window thumb sample in range"
            )

    def validate_nominal_joint_qpos_rad(
        self, values: Mapping[str, Any]
    ) -> Mapping[str, float]:
        if set(values) != set(ACTIVE_ACTUATORS):
            raise ValueError(
                "nominal_joint_qpos_rad must contain exactly the eight active actuators"
            )
        nominal = {
            name: _finite_number(values[name], f"nominal_joint_qpos_rad.{name}")
            for name in ACTIVE_ACTUATORS
        }
        thumb = nominal[self.thumb_bend_actuator]
        lower, upper = self.thumb_actual_range_rad
        if not lower <= thumb <= upper:
            raise ValueError(
                "nominal thumb bend qpos must lie inside thumb_actual_range_rad"
            )
        return MappingProxyType(nominal)

    def as_config(self) -> dict[str, Any]:
        return {
            "thumb_bend_actuator": self.thumb_bend_actuator,
            "thumb_actual_range_rad": list(self.thumb_actual_range_rad),
            "max_nominal_joint_error_rad": self.max_nominal_joint_error_rad,
            "max_joint_stability_span_rad": self.max_joint_stability_span_rad,
            "verify_continuous_s": self.verify_continuous_s,
            "qpos_reference": self.qpos_reference,
            "require_all_thumb_samples_in_range": (
                self.require_all_thumb_samples_in_range
            ),
        }

    def resolved_config(
        self, nominal_joint_qpos_rad: Mapping[str, Any]
    ) -> dict[str, Any]:
        nominal = self.validate_nominal_joint_qpos_rad(nominal_joint_qpos_rad)
        return {
            "nominal_joint_qpos_rad": dict(nominal),
            **self.as_config(),
        }


@dataclass(frozen=True)
class PosePreservationSettings:
    """Schema-v6 initial-object-pose contract.

    The reference and scope strings are intentionally closed enums.  A
    configuration therefore cannot weaken the requirement by silently moving
    the reference to the first contact or to the final verification window.
    MuJoCo's cube remains a free body; these settings describe evidence and
    acceptance only, never a weld, mocap body or runtime cube-qpos rewrite.
    Schema v6 may initialize the active hand joints at a collision-free
    pregrasp pose, which is part of the declared initial condition rather than
    an intervention on the object.
    """

    max_translation_m: float
    max_orientation_drift_deg: float
    require_support_contact: bool = True
    require_no_hand_cube_contact_during_settle: bool = True
    initialize_active_joints_at_pregrasp: bool = True
    reference: str = "reset_pre_step_cube_pose"
    scope: str = "through_grasp_acquisition_inclusive"

    def __post_init__(self) -> None:
        translation = _finite_number(self.max_translation_m, "max_translation_m")
        orientation = _finite_number(
            self.max_orientation_drift_deg,
            "max_orientation_drift_deg",
        )
        if translation <= 0.0:
            raise ValueError("max_translation_m must be positive")
        if not 0.0 < orientation <= 180.0:
            raise ValueError(
                "max_orientation_drift_deg must be within (0, 180]"
            )
        object.__setattr__(self, "max_translation_m", translation)
        object.__setattr__(self, "max_orientation_drift_deg", orientation)
        if self.reference != "reset_pre_step_cube_pose":
            raise ValueError(
                "pose preservation reference must be reset_pre_step_cube_pose"
            )
        if self.scope != "through_grasp_acquisition_inclusive":
            raise ValueError(
                "pose preservation scope must be "
                "through_grasp_acquisition_inclusive"
            )
        for label in (
            "require_support_contact",
            "require_no_hand_cube_contact_during_settle",
            "initialize_active_joints_at_pregrasp",
        ):
            if not isinstance(getattr(self, label), bool):
                raise ValueError(f"{label} must be boolean")

    def as_config(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "scope": self.scope,
            "max_translation_m": self.max_translation_m,
            "max_orientation_drift_deg": self.max_orientation_drift_deg,
            "require_support_contact": self.require_support_contact,
            "require_no_hand_cube_contact_during_settle": (
                self.require_no_hand_cube_contact_during_settle
            ),
            "initialize_active_joints_at_pregrasp": (
                self.initialize_active_joints_at_pregrasp
            ),
        }


@dataclass(frozen=True)
class EvaluationSettings:
    """Topology-specific acceptance settings layered on common lift checks.

    Field names intentionally mirror the persisted ``contact_topology`` block
    so a runner can serialize these settings without an implicit rename table.
    """

    target_faces: OpposedFaceAssignment | None = None
    palm_normal_local_axis: str | None = None
    world_down_axis: str = "-Z"
    max_palm_down_angle_deg: float | None = None
    median_lift_m: float = 0.010
    minimum_lift_m: float = 0.008
    height_window_s: float = 0.5
    max_height_span_m: float = 0.002
    max_orientation_drift_deg: float = 20.0
    max_end_linear_speed_m_s: float = 0.05
    max_penetration_m: float = 0.002
    touch_force_min_n: float = 1e-8
    inactive_joint_abs_max_rad: float = 0.02
    surface_tolerance_m: float = 0.0
    edge_margin_m: float = 0.0
    min_normal_alignment: float = 0.0
    contact_force_min_n: float = 0.0
    target_force_fraction: float = 0.0
    finger_contact_duty: float = 0.0
    simultaneous_contact_duty: float = 0.0
    max_off_target_force_fraction: float = 1.0
    max_material_off_target_duty: float = 1.0
    max_material_off_target_run_s: float = math.inf
    forbid_active_nondistal: bool = False

    def __post_init__(self) -> None:
        if self.world_down_axis not in FACES:
            raise ValueError(f"world_down_axis must be one of {FACES}")
        if self.target_faces is None:
            return
        if self.palm_normal_local_axis not in FACES:
            raise ValueError(f"palm_normal_local_axis must be one of {FACES}")
        if self.max_palm_down_angle_deg is None or not 0.0 <= float(
            self.max_palm_down_angle_deg
        ) <= 180.0:
            raise ValueError("max_palm_down_angle_deg must be within [0, 180]")
        for label in (
            "min_normal_alignment",
            "target_force_fraction",
            "finger_contact_duty",
            "simultaneous_contact_duty",
            "max_off_target_force_fraction",
            "max_material_off_target_duty",
        ):
            value = float(getattr(self, label))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be within [0, 1]")
        for label in (
            "median_lift_m",
            "minimum_lift_m",
            "height_window_s",
            "max_height_span_m",
            "max_orientation_drift_deg",
            "max_end_linear_speed_m_s",
            "max_penetration_m",
            "touch_force_min_n",
            "inactive_joint_abs_max_rad",
            "surface_tolerance_m",
            "edge_margin_m",
            "contact_force_min_n",
        ):
            value = float(getattr(self, label))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be positive and finite")
        if self.surface_tolerance_m >= self.edge_margin_m:
            raise ValueError("surface_tolerance_m must be smaller than edge_margin_m")
        if not math.isclose(
            self.target_force_fraction + self.max_off_target_force_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "target_force_fraction and max_off_target_force_fraction must sum to 1"
            )
        if (
            not math.isfinite(float(self.max_material_off_target_run_s))
            or self.max_material_off_target_run_s < 0
        ):
            raise ValueError(
                "max_material_off_target_run_s must be finite and non-negative"
            )
        if not isinstance(self.forbid_active_nondistal, bool):
            raise ValueError("forbid_active_nondistal must be boolean")

    def contact_topology_config(self) -> dict[str, Any]:
        """Return exactly the v2 JSON ``contact_topology`` contract."""

        if self.target_faces is None:
            return {}
        return {
            "target_faces": self.target_faces.as_dict(),
            "surface_tolerance_m": self.surface_tolerance_m,
            "edge_margin_m": self.edge_margin_m,
            "min_normal_alignment": self.min_normal_alignment,
            "target_force_fraction": self.target_force_fraction,
            "max_off_target_force_fraction": self.max_off_target_force_fraction,
            "max_material_off_target_duty": self.max_material_off_target_duty,
            "max_material_off_target_run_s": self.max_material_off_target_run_s,
            "forbid_active_nondistal": self.forbid_active_nondistal,
        }

    def acceptance_config(self) -> dict[str, Any]:
        """Return the immutable schema-v2 hard-acceptance contract."""

        if self.target_faces is None:
            return {}
        return {
            "median_lift_m": self.median_lift_m,
            "minimum_lift_m": self.minimum_lift_m,
            "height_window_s": self.height_window_s,
            "finger_contact_duty": self.finger_contact_duty,
            "simultaneous_contact_duty": self.simultaneous_contact_duty,
            "contact_force_min_n": self.contact_force_min_n,
            "touch_force_min_n": self.touch_force_min_n,
            "max_height_span_m": self.max_height_span_m,
            "max_orientation_drift_deg": self.max_orientation_drift_deg,
            "max_end_linear_speed_m_s": self.max_end_linear_speed_m_s,
            "max_penetration_m": self.max_penetration_m,
            "inactive_joint_abs_max_rad": self.inactive_joint_abs_max_rad,
            "max_palm_down_angle_deg": self.max_palm_down_angle_deg,
        }

    @property
    def max_face_normal_angle_deg(self) -> float:
        return math.degrees(math.acos(self.min_normal_alignment))

    @property
    def min_target_face_contact_duty(self) -> Mapping[str, float]:
        return MappingProxyType({finger: self.finger_contact_duty for finger in FINGERS})

    # Compatibility/readability aliases for evaluators using metric-oriented names.
    min_face_normal_alignment = property(lambda self: self.min_normal_alignment)
    min_face_edge_margin_m = property(lambda self: self.edge_margin_m)
    min_contact_force_n = property(lambda self: self.contact_force_min_n)
    min_target_force_purity = property(lambda self: self.target_force_fraction)
    min_simultaneous_topology_duty = property(
        lambda self: self.simultaneous_contact_duty
    )
    max_off_target_duty = property(lambda self: self.max_material_off_target_duty)
    max_off_target_run_s = property(lambda self: self.max_material_off_target_run_s)


@dataclass(frozen=True)
class SearchBounds:
    """Deterministic pose, object and actuator search domain."""

    palm_pitch_values_deg: tuple[float, ...]
    hand_roll_deg: tuple[float, float]
    hand_yaw_deg: tuple[float, float]
    cube_position_in_root_m: Mapping[str, tuple[float, float]]
    cube_yaw_deg: tuple[float, float]
    actuator_targets_rad: Mapping[str, tuple[float, float]]
    seed: int
    kinematic_samples_per_pitch: int
    dynamic_candidate_count: int
    local_refine_seed_count: int
    local_refine_per_seed: int
    final_candidate_count: int
    perturbations_per_final_candidate: int
    fallback_kinematic_samples_per_pitch: int
    fallback_candidate_count: int
    final_target_delta_rad: Mapping[str, tuple[float, float]] | None = None
    manipulation_delta_rad: Mapping[str, tuple[float, float]] | None = None
    pregrasp_targets_rad: Mapping[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        pitches = tuple(_finite_number(value, "palm_pitch_values_deg") for value in self.palm_pitch_values_deg)
        if not pitches or tuple(sorted(set(pitches))) != pitches:
            raise ValueError("palm_pitch_values_deg must be non-empty, unique and increasing")
        object.__setattr__(self, "palm_pitch_values_deg", pitches)
        object.__setattr__(self, "hand_roll_deg", _closed_range(self.hand_roll_deg, "hand_roll_deg"))
        object.__setattr__(self, "hand_yaw_deg", _closed_range(self.hand_yaw_deg, "hand_yaw_deg"))
        object.__setattr__(self, "cube_yaw_deg", _closed_range(self.cube_yaw_deg, "cube_yaw_deg"))

        cube_ranges = _frozen_ranges(self.cube_position_in_root_m, "cube_position_in_root_m")
        if set(cube_ranges) != {"x", "y", "z"}:
            raise ValueError("cube_position_in_root_m must contain exactly x, y and z")
        object.__setattr__(self, "cube_position_in_root_m", cube_ranges)

        actuator_ranges = _frozen_ranges(self.actuator_targets_rad, "actuator_targets_rad")
        if set(actuator_ranges) != set(ACTIVE_ACTUATORS):
            raise ValueError("actuator_targets_rad must contain exactly the eight active actuators")
        object.__setattr__(self, "actuator_targets_rad", actuator_ranges)
        if self.final_target_delta_rad is not None:
            delta_ranges = _frozen_ranges(
                self.final_target_delta_rad, "final_target_delta_rad"
            )
            if set(delta_ranges) != set(ACTIVE_ACTUATORS):
                raise ValueError(
                    "final_target_delta_rad must contain exactly the eight active actuators"
                )
            object.__setattr__(self, "final_target_delta_rad", delta_ranges)
        if self.manipulation_delta_rad is not None:
            manipulation_ranges = _frozen_ranges(
                self.manipulation_delta_rad, "manipulation_delta_rad"
            )
            if set(manipulation_ranges) != set(ACTIVE_ACTUATORS):
                raise ValueError(
                    "manipulation_delta_rad must contain exactly the eight active actuators"
                )
            object.__setattr__(
                self, "manipulation_delta_rad", manipulation_ranges
            )
        if self.pregrasp_targets_rad is not None:
            pregrasp_ranges = _frozen_ranges(
                self.pregrasp_targets_rad, "pregrasp_targets_rad"
            )
            if set(pregrasp_ranges) != set(ACTIVE_ACTUATORS):
                raise ValueError(
                    "pregrasp_targets_rad must contain exactly the eight active "
                    "actuators"
                )
            object.__setattr__(
                self, "pregrasp_targets_rad", pregrasp_ranges
            )

        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for label in (
            "kinematic_samples_per_pitch",
            "dynamic_candidate_count",
            "local_refine_seed_count",
            "local_refine_per_seed",
            "final_candidate_count",
            "perturbations_per_final_candidate",
            "fallback_kinematic_samples_per_pitch",
            "fallback_candidate_count",
        ):
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")

    @property
    def palm_pitch_deg(self) -> tuple[float, float]:
        return (self.palm_pitch_values_deg[0], self.palm_pitch_values_deg[-1])

    def contains_pitch(self, value: float) -> bool:
        number = float(value)
        return self.palm_pitch_deg[0] <= number <= self.palm_pitch_deg[1]

    def contains_cube_position(self, xyz: Sequence[float]) -> bool:
        if len(xyz) != 3:
            return False
        return all(
            self.cube_position_in_root_m[axis][0] - 1e-12
            <= float(value)
            <= self.cube_position_in_root_m[axis][1] + 1e-12
            for axis, value in zip(("x", "y", "z"), xyz)
        )

    def contains_targets(self, targets: Mapping[str, Any]) -> bool:
        if set(targets) != set(self.actuator_targets_rad):
            return False
        return all(
            lower <= float(targets[name]) <= upper
            for name, (lower, upper) in self.actuator_targets_rad.items()
        )

    def contains_manipulation_delta(self, delta: Mapping[str, Any]) -> bool:
        if self.manipulation_delta_rad is None:
            return False
        if set(delta) != set(self.manipulation_delta_rad):
            return False
        return all(
            lower <= float(delta[name]) <= upper
            for name, (lower, upper) in self.manipulation_delta_rad.items()
        )

    def contains_pregrasp_targets(self, targets: Mapping[str, Any]) -> bool:
        if self.pregrasp_targets_rad is None:
            return False
        if set(targets) != set(self.pregrasp_targets_rad):
            return False
        return all(
            lower <= float(targets[name]) <= upper
            for name, (lower, upper) in self.pregrasp_targets_rad.items()
        )


@dataclass(frozen=True)
class BoundaryExpansionParameters:
    """One-shot search-bound expansion policy for a size campaign."""

    position_tolerance_m: float
    actuator_tolerance_rad: float
    position_expand_m: float
    actuator_expand_rad: float
    max_expansions: int = 1

    def __post_init__(self) -> None:
        for label in (
            "position_tolerance_m",
            "actuator_tolerance_rad",
            "position_expand_m",
            "actuator_expand_rad",
        ):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        if (
            not isinstance(self.max_expansions, int)
            or isinstance(self.max_expansions, bool)
            or self.max_expansions != 1
        ):
            raise ValueError("size campaigns allow exactly one boundary expansion")

    def as_config(self) -> dict[str, Any]:
        return {
            "position_tolerance_m": self.position_tolerance_m,
            "actuator_tolerance_rad": self.actuator_tolerance_rad,
            "position_expand_m": self.position_expand_m,
            "actuator_expand_rad": self.actuator_expand_rad,
            "max_expansions": self.max_expansions,
        }


@dataclass(frozen=True)
class RobustnessCaseFamilies:
    """Two material families sharing a nominal-centred edge window."""

    schema_version: int
    edge_limits_m: tuple[float, float]
    edge_window_count: int
    edge_step_m: float
    density_mass_scales: tuple[float, ...]
    friction: tuple[float, ...]
    fixed_mass_kg: float
    boundary_probe_step_m: float
    fixed_mass_case_label: str = "fixed_mass_control"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("robustness case-family schema_version must be 1")
        limits = _closed_range(self.edge_limits_m, "edge_limits_m")
        if limits[0] <= 0.0 or math.isclose(limits[0], limits[1]):
            raise ValueError("edge_limits_m must contain a positive non-empty range")
        object.__setattr__(self, "edge_limits_m", limits)
        if (
            not isinstance(self.edge_window_count, int)
            or isinstance(self.edge_window_count, bool)
            or self.edge_window_count <= 0
            or self.edge_window_count % 2 == 0
        ):
            raise ValueError("edge_window_count must be a positive odd integer")
        edge_step = _finite_number(self.edge_step_m, "edge_step_m")
        boundary_step = _finite_number(
            self.boundary_probe_step_m, "boundary_probe_step_m"
        )
        fixed_mass = _finite_number(self.fixed_mass_kg, "fixed_mass_kg")
        if edge_step <= 0.0 or boundary_step <= 0.0 or fixed_mass <= 0.0:
            raise ValueError("robustness edge steps and fixed mass must be positive")
        object.__setattr__(self, "edge_step_m", edge_step)
        object.__setattr__(self, "boundary_probe_step_m", boundary_step)
        object.__setattr__(self, "fixed_mass_kg", fixed_mass)
        if (
            not _EXPERIMENT_ID_PATTERN.fullmatch(self.fixed_mass_case_label)
            or self.fixed_mass_case_label == "constant_density"
        ):
            raise ValueError(
                "fixed_mass_case_label must be lower-case snake_case and distinct "
                "from constant_density"
            )

        span_steps = (limits[1] - limits[0]) / edge_step
        if not math.isclose(span_steps, round(span_steps), abs_tol=1e-9):
            raise ValueError("edge_limits_m span must be divisible by edge_step_m")
        if self.edge_window_count > round(span_steps) + 1:
            raise ValueError("edge_window_count does not fit inside edge_limits_m")

        scales = tuple(
            _finite_number(value, "density_mass_scales")
            for value in self.density_mass_scales
        )
        frictions = tuple(_finite_number(value, "friction") for value in self.friction)
        if (
            not scales
            or any(value <= 0.0 for value in scales)
            or tuple(sorted(set(scales))) != scales
        ):
            raise ValueError(
                "density_mass_scales must be positive, unique and increasing"
            )
        if (
            not frictions
            or any(value <= 0.0 for value in frictions)
            or tuple(sorted(set(frictions))) != frictions
        ):
            raise ValueError("friction must be positive, unique and increasing")
        object.__setattr__(self, "density_mass_scales", scales)
        object.__setattr__(self, "friction", frictions)

    @property
    def constant_density_case_count(self) -> int:
        return (
            self.edge_window_count
            * len(self.density_mass_scales)
            * len(self.friction)
        )

    @property
    def fixed_mass_case_count(self) -> int:
        return self.edge_window_count * len(self.friction)

    @property
    def total_case_count(self) -> int:
        return self.constant_density_case_count + self.fixed_mass_case_count

    def edge_window_m(self, nominal_edge_m: float) -> tuple[float, ...]:
        """Return a centred, clamped sequence of exact edge-grid values."""

        nominal = _finite_number(nominal_edge_m, "nominal_edge_m")
        lower, upper = self.edge_limits_m
        if not lower <= nominal <= upper:
            raise ValueError("nominal_edge_m must lie inside edge_limits_m")
        grid_size = round((upper - lower) / self.edge_step_m) + 1
        centre_index = round((nominal - lower) / self.edge_step_m)
        half_window = self.edge_window_count // 2
        start = min(
            max(0, centre_index - half_window), grid_size - self.edge_window_count
        )
        return tuple(
            round(lower + (start + offset) * self.edge_step_m, 12)
            for offset in range(self.edge_window_count)
        )

    def as_config(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "edge_limits_m": list(self.edge_limits_m),
            "edge_window_count": self.edge_window_count,
            "edge_step_m": self.edge_step_m,
            "density_mass_scales": list(self.density_mass_scales),
            "friction": list(self.friction),
            "fixed_mass_kg": self.fixed_mass_kg,
            "fixed_mass_case_label": self.fixed_mass_case_label,
            "boundary_probe_step_m": self.boundary_probe_step_m,
            "case_counts": {
                "constant_density": self.constant_density_case_count,
                self.fixed_mass_case_label: self.fixed_mass_case_count,
                "total": self.total_case_count,
            },
        }


@dataclass(frozen=True)
class SizeCampaignParameters:
    """Versioned coarse-to-fine large-cube search and material policy."""

    schema_version: int
    coarse_edges_m: tuple[float, ...]
    discovery_mass_kg: float
    discovery_friction: float
    reference_edge_m: float
    reference_mass_kg: float
    coarse_samples_per_pitch: int
    odd_samples_per_pitch: int
    fine_samples_per_pitch: int
    coarse_size_count: int
    exact_size_count: int
    fine_dynamic_per_size: int
    local_seed_count_per_size: int
    local_refine_per_seed: int
    constant_density_max_candidates: int
    density_refine_seed_count: int
    density_refine_per_seed: int
    finalist_count: int
    perturbations_per_final: int
    boundary: BoundaryExpansionParameters
    target_sampling_policy: str = "independent_absolute_pregrasp_and_final"
    local_refinement_min_target_fingers: int = 2
    manipulation_seed_count: int = 0
    manipulation_refine_per_seed: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("size_campaign.schema_version must be 1")
        edges = tuple(_finite_number(value, "coarse_edges_m") for value in self.coarse_edges_m)
        if (
            not edges
            or any(value <= 0.0 for value in edges)
            or tuple(sorted(set(edges))) != edges
        ):
            raise ValueError("coarse_edges_m must be positive, unique and increasing")
        object.__setattr__(self, "coarse_edges_m", edges)
        for label in (
            "discovery_mass_kg",
            "discovery_friction",
            "reference_edge_m",
            "reference_mass_kg",
        ):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        integer_fields = (
            "coarse_samples_per_pitch",
            "odd_samples_per_pitch",
            "fine_samples_per_pitch",
            "coarse_size_count",
            "exact_size_count",
            "fine_dynamic_per_size",
            "local_seed_count_per_size",
            "local_refine_per_seed",
            "constant_density_max_candidates",
            "density_refine_seed_count",
            "density_refine_per_seed",
            "finalist_count",
            "perturbations_per_final",
        )
        for label in integer_fields:
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.coarse_size_count > len(edges):
            raise ValueError("coarse_size_count cannot exceed coarse_edges_m")
        if self.exact_size_count > self.coarse_size_count:
            raise ValueError("exact_size_count cannot exceed coarse_size_count")
        if not isinstance(self.boundary, BoundaryExpansionParameters):
            raise TypeError("boundary must be BoundaryExpansionParameters")
        target_policies = {
            "independent_absolute_pregrasp_and_final",
            "grasp_targets_plus_manipulation_delta",
        }
        if self.target_sampling_policy not in target_policies:
            raise ValueError(
                "size campaign target_sampling_policy is not supported"
            )
        if self.target_sampling_policy == "independent_absolute_pregrasp_and_final":
            if self.manipulation_seed_count != 0 or self.manipulation_refine_per_seed != 0:
                raise ValueError(
                    "legacy size campaigns cannot declare manipulation refinement"
                )
        else:
            for label in (
                "manipulation_seed_count",
                "manipulation_refine_per_seed",
            ):
                value = getattr(self, label)
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"{label} must be a positive integer")
        if (
            not isinstance(self.local_refinement_min_target_fingers, int)
            or isinstance(self.local_refinement_min_target_fingers, bool)
            or not 1 <= self.local_refinement_min_target_fingers <= len(FINGERS)
        ):
            raise ValueError(
                "local_refinement_min_target_fingers must be within [1, 3]"
            )

    @property
    def density_kg_m3(self) -> float:
        return self.reference_mass_kg / self.reference_edge_m**3

    def constant_density_mass_kg(self, edge_m: float) -> float:
        edge = _finite_number(edge_m, "edge_m")
        if edge <= 0.0:
            raise ValueError("edge_m must be positive")
        return self.reference_mass_kg * (edge / self.reference_edge_m) ** 3

    @property
    def dynamic_candidate_count(self) -> int:
        return self.exact_size_count * self.fine_dynamic_per_size

    @property
    def fixed_mass_local_refinement_count(self) -> int:
        return (
            self.exact_size_count
            * self.local_seed_count_per_size
            * self.local_refine_per_seed
        )

    @property
    def density_local_refinement_count(self) -> int:
        return self.density_refine_seed_count * self.density_refine_per_seed

    @property
    def manipulation_local_refinement_count(self) -> int:
        return self.manipulation_seed_count * self.manipulation_refine_per_seed

    def as_config(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "coarse_edges_m": list(self.coarse_edges_m),
            "material_policy": {
                "discovery_mass_kg": self.discovery_mass_kg,
                "discovery_friction": self.discovery_friction,
                "reference_edge_m": self.reference_edge_m,
                "reference_mass_kg": self.reference_mass_kg,
                "density_kg_m3": self.density_kg_m3,
                "success_requires_constant_density": True,
            },
            "budget": {
                "coarse_samples_per_pitch": self.coarse_samples_per_pitch,
                "odd_samples_per_pitch": self.odd_samples_per_pitch,
                "fine_samples_per_pitch": self.fine_samples_per_pitch,
                "coarse_size_count": self.coarse_size_count,
                "exact_size_count": self.exact_size_count,
                "fine_dynamic_per_size": self.fine_dynamic_per_size,
                "local_seed_count_per_size": self.local_seed_count_per_size,
                "local_refine_per_seed": self.local_refine_per_seed,
                "constant_density_max_candidates": (
                    self.constant_density_max_candidates
                ),
                "density_refine_seed_count": self.density_refine_seed_count,
                "density_refine_per_seed": self.density_refine_per_seed,
                "finalist_count": self.finalist_count,
                "perturbations_per_final": self.perturbations_per_final,
            },
            "trajectory_policy": {
                "target_sampling": self.target_sampling_policy,
            },
            "continuation_policy": {
                "local_refinement_min_target_fingers": (
                    self.local_refinement_min_target_fingers
                ),
                "boundary_hit_also_continues": True,
            },
            "boundary": self.boundary.as_config(),
        }
        if self.target_sampling_policy == "grasp_targets_plus_manipulation_delta":
            result["budget"].update(
                {
                    "manipulation_seed_count": self.manipulation_seed_count,
                    "manipulation_refine_per_seed": (
                        self.manipulation_refine_per_seed
                    ),
                }
            )
        return result


@dataclass(frozen=True)
class AlignedContactPerturbationEnvelope:
    """Versioned 12-D local robustness envelope for schema-v4 campaigns."""

    cube_center_xy_delta_m: tuple[float, float]
    cube_gap_m: tuple[float, float]
    cube_rpy_delta_deg: tuple[float, float]
    hand_roll_yaw_delta_deg: tuple[float, float]
    finger_down_tilt_delta_deg: tuple[float, float]
    palm_press_depth_delta_m: tuple[float, float]
    mass_scale: tuple[float, float]
    friction_delta: tuple[float, float]
    sampling: str = "latin_hypercube"
    edge_policy: str = "fixed_nominal"

    def __post_init__(self) -> None:
        for label in (
            "cube_center_xy_delta_m",
            "cube_gap_m",
            "cube_rpy_delta_deg",
            "hand_roll_yaw_delta_deg",
            "finger_down_tilt_delta_deg",
            "palm_press_depth_delta_m",
            "mass_scale",
            "friction_delta",
        ):
            object.__setattr__(
                self, label, _closed_range(getattr(self, label), label)
            )
        if self.sampling != "latin_hypercube":
            raise ValueError("aligned perturbations require latin_hypercube sampling")
        if self.edge_policy != "fixed_nominal":
            raise ValueError("aligned perturbations require fixed_nominal edge policy")

    @property
    def dimensions(self) -> int:
        # hand roll/yaw (2), tilt (1), press (1), cube RPY (3),
        # cube XY (2), gap (1), mass (1), friction (1)
        return 12

    def as_dict(self) -> dict[str, list[float]]:
        return {
            name: [float(value) for value in getattr(self, name)]
            for name in (
                "cube_center_xy_delta_m",
                "cube_gap_m",
                "cube_rpy_delta_deg",
                "hand_roll_yaw_delta_deg",
                "finger_down_tilt_delta_deg",
                "palm_press_depth_delta_m",
                "mass_scale",
                "friction_delta",
            )
        }

    def as_config(self) -> dict[str, Any]:
        return {
            "sampling": self.sampling,
            "dimensions": self.dimensions,
            "edge_policy": self.edge_policy,
            "ranges": self.as_dict(),
        }


@dataclass(frozen=True)
class AlignedContactCampaignParameters:
    """Decision-complete schema-v4 edge-by-tilt search declaration.

    A *band* is one finger-down tilt centre.  Keeping every downstream budget
    per band makes it impossible for an implementation to compensate for an
    empty band by taking additional candidates from a different orientation.
    """

    schema_version: int
    edges_m: tuple[float, ...]
    density_kg_m3: float
    friction: float
    tilt_band_centers_deg: tuple[float, ...]
    static_samples_per_edge_band: int
    dynamic_candidates_per_band: int
    grasp_refine_seed_count_per_band: int
    grasp_refine_per_seed: int
    manipulation_seed_count_per_band: int
    manipulation_refine_per_seed: int
    exact_candidates_per_band: int
    perturbation_envelope: AlignedContactPerturbationEnvelope

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("aligned_contact_campaign.schema_version must be 1")
        edges = tuple(_finite_number(value, "edges_m") for value in self.edges_m)
        if (
            not edges
            or any(value <= 0.0 for value in edges)
            or tuple(sorted(set(edges))) != edges
        ):
            raise ValueError("edges_m must be positive, unique and increasing")
        object.__setattr__(self, "edges_m", edges)
        bands = tuple(
            _finite_number(value, "tilt_band_centers_deg")
            for value in self.tilt_band_centers_deg
        )
        if (
            not bands
            or tuple(sorted(set(bands))) != bands
            or bands[0] < 0.0
            or bands[-1] > 90.0
        ):
            raise ValueError(
                "tilt_band_centers_deg must be unique and increasing within [0, 90]"
            )
        object.__setattr__(self, "tilt_band_centers_deg", bands)
        for label in ("density_kg_m3", "friction"):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        for label in (
            "static_samples_per_edge_band",
            "dynamic_candidates_per_band",
            "grasp_refine_seed_count_per_band",
            "grasp_refine_per_seed",
            "manipulation_seed_count_per_band",
            "manipulation_refine_per_seed",
            "exact_candidates_per_band",
        ):
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if not isinstance(
            self.perturbation_envelope, AlignedContactPerturbationEnvelope
        ):
            raise TypeError(
                "perturbation_envelope must be AlignedContactPerturbationEnvelope"
            )

    @property
    def static_sample_count(self) -> int:
        return (
            len(self.edges_m)
            * len(self.tilt_band_centers_deg)
            * self.static_samples_per_edge_band
        )

    @property
    def dynamic_candidate_count(self) -> int:
        return len(self.tilt_band_centers_deg) * self.dynamic_candidates_per_band

    @property
    def grasp_refinement_count(self) -> int:
        return (
            len(self.tilt_band_centers_deg)
            * self.grasp_refine_seed_count_per_band
            * self.grasp_refine_per_seed
        )

    @property
    def manipulation_refinement_count(self) -> int:
        return (
            len(self.tilt_band_centers_deg)
            * self.manipulation_seed_count_per_band
            * self.manipulation_refine_per_seed
        )

    @property
    def exact_candidate_count(self) -> int:
        return len(self.tilt_band_centers_deg) * self.exact_candidates_per_band

    def constant_density_mass_kg(self, edge_m: float) -> float:
        edge = _finite_number(edge_m, "edge_m")
        if edge <= 0.0:
            raise ValueError("edge_m must be positive")
        return self.density_kg_m3 * edge**3

    # Metric-oriented aliases used by orchestration/reporting code.
    local_grasp_candidate_count = property(lambda self: self.grasp_refinement_count)
    local_manipulation_candidate_count = property(
        lambda self: self.manipulation_refinement_count
    )

    def budget_config(self) -> dict[str, Any]:
        return {
            "tilt_band_centers_deg": list(self.tilt_band_centers_deg),
            "static_samples_per_edge_band": self.static_samples_per_edge_band,
            "dynamic_candidates_per_band": self.dynamic_candidates_per_band,
            "grasp_refine_seed_count_per_band": (
                self.grasp_refine_seed_count_per_band
            ),
            "grasp_refine_per_seed": self.grasp_refine_per_seed,
            "manipulation_seed_count_per_band": (
                self.manipulation_seed_count_per_band
            ),
            "manipulation_refine_per_seed": self.manipulation_refine_per_seed,
            "exact_candidates_per_band": self.exact_candidates_per_band,
            "static_sample_count": self.static_sample_count,
            "dynamic_candidate_count": self.dynamic_candidate_count,
            "grasp_refinement_count": self.grasp_refinement_count,
            "manipulation_refinement_count": self.manipulation_refinement_count,
            "exact_candidate_count": self.exact_candidate_count,
        }

    def as_config(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "edges_m": list(self.edges_m),
            "material_policy": {
                "density_kg_m3": self.density_kg_m3,
                "friction": self.friction,
                "mass_policy": "constant_density",
            },
            "budget": self.budget_config(),
            "perturbation_envelope": self.perturbation_envelope.as_config(),
        }


@dataclass(frozen=True)
class FarHandPerturbationEnvelope:
    """Independent pose/material perturbations for the schema-v5 campaign."""

    cube_center_xy_delta_m: tuple[float, float]
    cube_gap_m: tuple[float, float]
    cube_rpy_delta_deg: tuple[float, float]
    hand_roll_yaw_delta_deg: tuple[float, float]
    finger_down_tilt_delta_deg: tuple[float, float]
    root_cube_distance_delta_m: tuple[float, float]
    mass_scale: tuple[float, float]
    friction_delta: tuple[float, float]
    sampling: str = "latin_hypercube"

    def __post_init__(self) -> None:
        for label in (
            "cube_center_xy_delta_m",
            "cube_gap_m",
            "cube_rpy_delta_deg",
            "hand_roll_yaw_delta_deg",
            "finger_down_tilt_delta_deg",
            "root_cube_distance_delta_m",
            "mass_scale",
            "friction_delta",
        ):
            object.__setattr__(
                self, label, _closed_range(getattr(self, label), label)
            )
        if self.sampling != "latin_hypercube":
            raise ValueError("far-hand perturbations require latin_hypercube")
        if self.mass_scale[0] <= 0.0:
            raise ValueError("mass_scale must stay positive")

    @property
    def dimensions(self) -> int:
        # cube XY (2), gap (1), cube RPY (3), hand roll/yaw (2), tilt
        # (1), radial distance (1), mass (1), friction (1).
        return 12

    def as_config(self) -> dict[str, Any]:
        names = (
            "cube_center_xy_delta_m",
            "cube_gap_m",
            "cube_rpy_delta_deg",
            "hand_roll_yaw_delta_deg",
            "finger_down_tilt_delta_deg",
            "root_cube_distance_delta_m",
            "mass_scale",
            "friction_delta",
        )
        return {
            "sampling": self.sampling,
            "dimensions": self.dimensions,
            "ranges": {
                name: [float(value) for value in getattr(self, name)]
                for name in names
            },
        }


@dataclass(frozen=True)
class FarHandBoundaryExpansion:
    """The only legal one-shot extension of the v5 nominal search domain."""

    distance_tolerance_m: float
    expanded_root_cube_distance_max_m: float
    expanded_cube_in_root_x_max_m: float
    thumb_bend_tolerance_rad: float
    thumb_bend_expand_rad: float
    max_expansions: int = 1

    def __post_init__(self) -> None:
        for label in (
            "distance_tolerance_m",
            "expanded_root_cube_distance_max_m",
            "expanded_cube_in_root_x_max_m",
            "thumb_bend_tolerance_rad",
            "thumb_bend_expand_rad",
        ):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        if self.max_expansions != 1:
            raise ValueError("far-hand search allows exactly one expansion")

    def as_config(self) -> dict[str, Any]:
        return {
            "distance_tolerance_m": self.distance_tolerance_m,
            "expanded_root_cube_distance_max_m": (
                self.expanded_root_cube_distance_max_m
            ),
            "expanded_cube_in_root_x_max_m": (
                self.expanded_cube_in_root_x_max_m
            ),
            "thumb_bend_tolerance_rad": self.thumb_bend_tolerance_rad,
            "thumb_bend_expand_rad": self.thumb_bend_expand_rad,
            "max_expansions": self.max_expansions,
        }


@dataclass(frozen=True)
class FarHandFingertipCampaignParameters:
    """Decision-complete staged budget and geometry policy for schema v5."""

    schema_version: int
    nominal_edge_m: float
    nominal_mass_kg: float
    density_kg_m3: float
    friction: float
    post_success_edges_m: tuple[float, ...]
    tilt_band_centers_deg: tuple[float, ...]
    primary_face: OpposedFaceAssignment
    fallback_face: OpposedFaceAssignment
    primary_static_samples_per_band: int
    fallback_static_samples_per_band: int
    static_retain_per_band: int
    dynamic_candidates_per_band: int
    grasp_refine_seed_count_per_band: int
    grasp_refine_per_seed: int
    manipulation_seed_count_per_band: int
    manipulation_refine_per_seed: int
    exact_candidates_per_band: int
    closure_alpha_values: tuple[float, ...]
    target_signed_gap_m: tuple[float, float]
    max_static_distal_preload_m: float
    concentrated_thumb_bend_fraction: float
    concentrated_thumb_bend_rad: tuple[float, float]
    minimum_manipulated_thumb_bend_rad: float
    perturbation_envelope: FarHandPerturbationEnvelope
    boundary_expansion: FarHandBoundaryExpansion

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("far_hand_campaign.schema_version must be 1")
        for label in (
            "nominal_edge_m",
            "nominal_mass_kg",
            "density_kg_m3",
            "friction",
            "max_static_distal_preload_m",
            "minimum_manipulated_thumb_bend_rad",
        ):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        edges = tuple(
            _finite_number(value, "post_success_edges_m")
            for value in self.post_success_edges_m
        )
        if not edges or tuple(sorted(set(edges))) != edges:
            raise ValueError(
                "post_success_edges_m must be non-empty, unique and increasing"
            )
        object.__setattr__(self, "post_success_edges_m", edges)
        bands = tuple(
            _finite_number(value, "tilt_band_centers_deg")
            for value in self.tilt_band_centers_deg
        )
        if not bands or tuple(sorted(set(bands))) != bands:
            raise ValueError(
                "tilt_band_centers_deg must be non-empty, unique and increasing"
            )
        object.__setattr__(self, "tilt_band_centers_deg", bands)
        if not isinstance(self.primary_face, OpposedFaceAssignment) or not isinstance(
            self.fallback_face, OpposedFaceAssignment
        ):
            raise TypeError("primary_face and fallback_face must be opposed faces")
        if self.primary_face == self.fallback_face:
            raise ValueError("primary and fallback face assignments must differ")
        for label in (
            "primary_static_samples_per_band",
            "fallback_static_samples_per_band",
            "static_retain_per_band",
            "dynamic_candidates_per_band",
            "grasp_refine_seed_count_per_band",
            "grasp_refine_per_seed",
            "manipulation_seed_count_per_band",
            "manipulation_refine_per_seed",
            "exact_candidates_per_band",
        ):
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.dynamic_candidates_per_band > self.static_retain_per_band:
            raise ValueError("dynamic candidates cannot exceed static retention")
        alphas = tuple(
            _finite_number(value, "closure_alpha_values")
            for value in self.closure_alpha_values
        )
        if (
            not alphas
            or tuple(sorted(set(alphas))) != alphas
            or alphas[0] < 0.0
            or alphas[-1] > 1.0
            or not math.isclose(alphas[-1], 1.0, abs_tol=1e-12)
        ):
            raise ValueError(
                "closure_alpha_values must increase within [0, 1] and end at 1"
            )
        object.__setattr__(self, "closure_alpha_values", alphas)
        gap = _closed_range(self.target_signed_gap_m, "target_signed_gap_m")
        if gap[0] >= 0.0 or gap[1] <= 0.0:
            raise ValueError("target_signed_gap_m must straddle zero")
        object.__setattr__(self, "target_signed_gap_m", gap)
        fraction = _finite_number(
            self.concentrated_thumb_bend_fraction,
            "concentrated_thumb_bend_fraction",
        )
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                "concentrated_thumb_bend_fraction must be within (0, 1]"
            )
        object.__setattr__(self, "concentrated_thumb_bend_fraction", fraction)
        object.__setattr__(
            self,
            "concentrated_thumb_bend_rad",
            _closed_range(
                self.concentrated_thumb_bend_rad,
                "concentrated_thumb_bend_rad",
            ),
        )
        if not isinstance(self.perturbation_envelope, FarHandPerturbationEnvelope):
            raise TypeError(
                "perturbation_envelope must be FarHandPerturbationEnvelope"
            )
        if not isinstance(self.boundary_expansion, FarHandBoundaryExpansion):
            raise TypeError("boundary_expansion must be FarHandBoundaryExpansion")
        expected_mass = self.density_kg_m3 * self.nominal_edge_m**3
        if not math.isclose(
            self.nominal_mass_kg, expected_mass, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError("nominal mass must match the declared density")

    @property
    def static_samples_per_band(self) -> int:
        return (
            self.primary_static_samples_per_band
            + self.fallback_static_samples_per_band
        )

    @property
    def static_sample_count(self) -> int:
        return len(self.tilt_band_centers_deg) * self.static_samples_per_band

    @property
    def dynamic_candidate_count(self) -> int:
        return len(self.tilt_band_centers_deg) * self.dynamic_candidates_per_band

    @property
    def grasp_refinement_count(self) -> int:
        return (
            len(self.tilt_band_centers_deg)
            * self.grasp_refine_seed_count_per_band
            * self.grasp_refine_per_seed
        )

    @property
    def manipulation_refinement_count(self) -> int:
        return (
            len(self.tilt_band_centers_deg)
            * self.manipulation_seed_count_per_band
            * self.manipulation_refine_per_seed
        )

    @property
    def exact_candidate_count(self) -> int:
        return len(self.tilt_band_centers_deg) * self.exact_candidates_per_band

    def constant_density_mass_kg(self, edge_m: float) -> float:
        edge = _finite_number(edge_m, "edge_m")
        if edge <= 0.0:
            raise ValueError("edge_m must be positive")
        return self.density_kg_m3 * edge**3

    def budget_config(self) -> dict[str, Any]:
        return {
            "tilt_band_centers_deg": list(self.tilt_band_centers_deg),
            "primary_static_samples_per_band": (
                self.primary_static_samples_per_band
            ),
            "fallback_static_samples_per_band": (
                self.fallback_static_samples_per_band
            ),
            "static_retain_per_band": self.static_retain_per_band,
            "dynamic_candidates_per_band": self.dynamic_candidates_per_band,
            "grasp_refine_seed_count_per_band": (
                self.grasp_refine_seed_count_per_band
            ),
            "grasp_refine_per_seed": self.grasp_refine_per_seed,
            "manipulation_seed_count_per_band": (
                self.manipulation_seed_count_per_band
            ),
            "manipulation_refine_per_seed": self.manipulation_refine_per_seed,
            "exact_candidates_per_band": self.exact_candidates_per_band,
            "static_sample_count": self.static_sample_count,
            "dynamic_candidate_count": self.dynamic_candidate_count,
            "grasp_refinement_count": self.grasp_refinement_count,
            "manipulation_refinement_count": self.manipulation_refinement_count,
            "exact_candidate_count": self.exact_candidate_count,
        }

    def as_config(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "nominal_material": {
                "edge_m": self.nominal_edge_m,
                "mass_kg": self.nominal_mass_kg,
                "density_kg_m3": self.density_kg_m3,
                "friction": self.friction,
            },
            "post_success_edges_m": list(self.post_success_edges_m),
            "face_policy": {
                "primary": self.primary_face.as_dict(),
                "fallback": self.fallback_face.as_dict(),
                "reverse_topologies_enabled": False,
            },
            "closure_screen": {
                "alpha_values": list(self.closure_alpha_values),
                "target_signed_gap_m": list(self.target_signed_gap_m),
                "max_static_distal_preload_m": (
                    self.max_static_distal_preload_m
                ),
                "active_nondistal_penetration_allowed": False,
                "dynamic_max_penetration_unchanged": True,
            },
            "thumb_bend_sampling": {
                "concentrated_fraction": self.concentrated_thumb_bend_fraction,
                "concentrated_range_rad": list(
                    self.concentrated_thumb_bend_rad
                ),
                "minimum_manipulated_target_rad": (
                    self.minimum_manipulated_thumb_bend_rad
                ),
            },
            "budget": self.budget_config(),
            "boundary_expansion": self.boundary_expansion.as_config(),
            "perturbation_envelope": self.perturbation_envelope.as_config(),
        }


@dataclass(frozen=True)
class HighThumbSizeCampaignParameters:
    """Versioned schema-v7 high-thumb, variable-size search contract.

    Unlike the schema-v5 far-hand campaign, this campaign intentionally keeps
    mass fixed while the edge length changes.  The cube world pose is an input
    invariant for each candidate: tuning may move the fixed hand root and its
    active joints, but may not sample, weld or rewrite the free cube pose.
    """

    schema_version: int
    coarse_edges_m: tuple[float, ...]
    coarse_thumb_targets_rad: tuple[float, ...]
    edge_fine_step_m: float
    thumb_fine_step_rad: float
    fixed_mass_kg: float
    friction: float
    seed: int
    cube_center_xy_m: tuple[float, float]
    grasp_target_bounds_rad: Mapping[str, tuple[float, float]]
    pregrasp_target_bounds_rad: Mapping[str, tuple[float, float]]
    manipulation_delta_bounds_rad: Mapping[str, tuple[float, float]]
    source_seed_family_count: int
    static_samples_per_cell: int
    static_retain_per_cell: int
    dynamic_candidate_count: int
    local_sizes_per_thumb_band: int
    local_seeds_per_size: int
    local_refine_per_seed: int
    fine_dynamic_candidate_count: int
    exact_candidate_count: int
    selected_grasp_count: int
    selected_lift_seed_count: int
    perturbations_per_grasp: int
    manipulation_candidates_per_lift_seed: int
    minimum_distinct_edges: int
    minimum_seed_families: int
    minimum_per_thumb_band: int
    maximum_per_edge_target_pair: int
    thumb_selection_bands_rad: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("high_thumb_size_campaign.schema_version must be 1")
        edges = tuple(
            _finite_number(value, "coarse_edges_m")
            for value in self.coarse_edges_m
        )
        targets = tuple(
            _finite_number(value, "coarse_thumb_targets_rad")
            for value in self.coarse_thumb_targets_rad
        )
        for values, label in (
            (edges, "coarse_edges_m"),
            (targets, "coarse_thumb_targets_rad"),
        ):
            if (
                len(values) < 2
                or any(value <= 0.0 for value in values)
                or tuple(sorted(set(values))) != values
            ):
                raise ValueError(
                    f"{label} must contain at least two positive, unique and "
                    "increasing values"
                )
        object.__setattr__(self, "coarse_edges_m", edges)
        object.__setattr__(self, "coarse_thumb_targets_rad", targets)

        for label in (
            "edge_fine_step_m",
            "thumb_fine_step_rad",
            "fixed_mass_kg",
            "friction",
        ):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        if self.edge_fine_step_m >= edges[1] - edges[0]:
            raise ValueError("edge_fine_step_m must be smaller than the coarse step")
        if self.thumb_fine_step_rad >= targets[1] - targets[0]:
            raise ValueError(
                "thumb_fine_step_rad must be smaller than the coarse target step"
            )

        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        centre = tuple(
            _finite_number(value, "cube_center_xy_m")
            for value in self.cube_center_xy_m
        )
        if len(centre) != 2:
            raise ValueError("cube_center_xy_m must contain exactly two values")
        object.__setattr__(self, "cube_center_xy_m", centre)

        for field_name in (
            "grasp_target_bounds_rad",
            "pregrasp_target_bounds_rad",
            "manipulation_delta_bounds_rad",
        ):
            ranges = _frozen_ranges(getattr(self, field_name), field_name)
            if set(ranges) != set(ACTIVE_ACTUATORS):
                raise ValueError(
                    f"{field_name} must contain exactly the eight active actuators"
                )
            object.__setattr__(self, field_name, ranges)
        thumb = "left_hand_thumb_bend_joint_actuator"
        if self.grasp_target_bounds_rad[thumb] != (targets[0], targets[-1]):
            raise ValueError(
                "thumb grasp bounds must match the coarse target endpoints"
            )

        integer_fields = (
            "source_seed_family_count",
            "static_samples_per_cell",
            "static_retain_per_cell",
            "dynamic_candidate_count",
            "local_sizes_per_thumb_band",
            "local_seeds_per_size",
            "local_refine_per_seed",
            "fine_dynamic_candidate_count",
            "exact_candidate_count",
            "selected_grasp_count",
            "selected_lift_seed_count",
            "perturbations_per_grasp",
            "manipulation_candidates_per_lift_seed",
            "minimum_distinct_edges",
            "minimum_seed_families",
            "minimum_per_thumb_band",
            "maximum_per_edge_target_pair",
        )
        for label in integer_fields:
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.static_retain_per_cell * self.cell_count < self.dynamic_candidate_count:
            raise ValueError("dynamic candidates exceed total static retention")
        if self.selected_lift_seed_count > self.selected_grasp_count:
            raise ValueError("lift seed count cannot exceed selected grasp count")
        if self.minimum_distinct_edges > len(edges):
            raise ValueError("minimum_distinct_edges exceeds the edge campaign")
        if self.minimum_seed_families > self.source_seed_family_count:
            raise ValueError("minimum_seed_families exceeds available seed families")

        bands = tuple(
            _closed_range(value, "thumb_selection_bands_rad")
            for value in self.thumb_selection_bands_rad
        )
        if not bands:
            raise ValueError("thumb_selection_bands_rad must not be empty")
        if not math.isclose(bands[0][0], targets[0], abs_tol=1e-12) or not math.isclose(
            bands[-1][1], targets[-1], abs_tol=1e-12
        ):
            raise ValueError("thumb selection bands must span the target range")
        for previous, following in zip(bands, bands[1:]):
            if not math.isclose(previous[1], following[0], abs_tol=1e-12):
                raise ValueError("thumb selection bands must be contiguous")
        if self.minimum_per_thumb_band * len(bands) > self.selected_grasp_count:
            raise ValueError("thumb-band quotas exceed selected_grasp_count")
        object.__setattr__(self, "thumb_selection_bands_rad", bands)

    @property
    def cell_count(self) -> int:
        return len(self.coarse_edges_m) * len(self.coarse_thumb_targets_rad)

    @property
    def static_sample_count(self) -> int:
        return self.cell_count * self.static_samples_per_cell

    @property
    def local_refine_seed_count(self) -> int:
        return (
            len(self.coarse_thumb_targets_rad)
            * self.local_sizes_per_thumb_band
            * self.local_seeds_per_size
        )

    @property
    def local_refinement_count(self) -> int:
        return self.local_refine_seed_count * self.local_refine_per_seed

    @property
    def manipulation_refinement_count(self) -> int:
        return (
            self.selected_lift_seed_count
            * self.manipulation_candidates_per_lift_seed
        )

    def contains_edge(self, edge_m: float) -> bool:
        edge = _finite_number(edge_m, "edge_m")
        return self.coarse_edges_m[0] - 1e-12 <= edge <= self.coarse_edges_m[-1] + 1e-12

    @staticmethod
    def _serialize_ranges(
        ranges: Mapping[str, tuple[float, float]],
    ) -> dict[str, list[float]]:
        return {name: list(ranges[name]) for name in ACTIVE_ACTUATORS}

    def budget_config(self) -> dict[str, int]:
        return {
            "static_samples_per_cell": self.static_samples_per_cell,
            "static_retain_per_cell": self.static_retain_per_cell,
            "cell_count": self.cell_count,
            "static_sample_count": self.static_sample_count,
            "dynamic_candidate_count": self.dynamic_candidate_count,
            "local_sizes_per_thumb_band": self.local_sizes_per_thumb_band,
            "local_seeds_per_size": self.local_seeds_per_size,
            "local_refine_seed_count": self.local_refine_seed_count,
            "local_refine_per_seed": self.local_refine_per_seed,
            "local_refinement_count": self.local_refinement_count,
            "fine_dynamic_candidate_count": self.fine_dynamic_candidate_count,
            "exact_candidate_count": self.exact_candidate_count,
            "perturbations_per_grasp": self.perturbations_per_grasp,
            "manipulation_candidates_per_lift_seed": (
                self.manipulation_candidates_per_lift_seed
            ),
            "manipulation_refinement_count": self.manipulation_refinement_count,
        }

    def as_config(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "coarse_edges_m": list(self.coarse_edges_m),
            "coarse_thumb_targets_rad": list(self.coarse_thumb_targets_rad),
            "edge_fine_step_m": self.edge_fine_step_m,
            "thumb_fine_step_rad": self.thumb_fine_step_rad,
            "fixed_mass_kg": self.fixed_mass_kg,
            "friction": self.friction,
            "seed": self.seed,
            "cube_pose_policy": {
                "center_xy_m": list(self.cube_center_xy_m),
                "support_placement": "support_top_z_plus_half_edge",
                "retain_source_seed_yaw": True,
                "freejoint_required": True,
                "runtime_pose_rewrite_forbidden": True,
                "candidate_world_pose_sampling_forbidden": True,
            },
            "control_search": {
                "grasp_target_bounds_rad": self._serialize_ranges(
                    self.grasp_target_bounds_rad
                ),
                "pregrasp_target_bounds_rad": self._serialize_ranges(
                    self.pregrasp_target_bounds_rad
                ),
                "manipulation_delta_bounds_rad": self._serialize_ranges(
                    self.manipulation_delta_bounds_rad
                ),
                "acquisition_manipulation_delta_zero": True,
            },
            "source_seed_family_count": self.source_seed_family_count,
            "budget": self.budget_config(),
            "selection": {
                "selected_grasp_count": self.selected_grasp_count,
                "selected_lift_seed_count": self.selected_lift_seed_count,
                "minimum_distinct_edges": self.minimum_distinct_edges,
                "minimum_seed_families": self.minimum_seed_families,
                "minimum_per_thumb_band": self.minimum_per_thumb_band,
                "maximum_per_edge_target_pair": (
                    self.maximum_per_edge_target_pair
                ),
                "thumb_bend_bands_rad": [
                    list(bounds) for bounds in self.thumb_selection_bands_rad
                ],
                "band_boundary_policy": "lower_open_except_first_upper_closed",
            },
        }


@dataclass(frozen=True)
class NormalAlignedSmoothLiftCampaignParameters:
    """Versioned schema-v8 rescue, pose search and smooth-lift budget."""

    schema_version: int
    edges_m: tuple[float, ...]
    thumb_targets_rad: tuple[float, ...]
    fixed_mass_kg: float
    friction: float
    cube_center_xy_m: tuple[float, float]
    seed: int
    old_pose_count: int = 21
    tier_a_pose_count: int = 11
    tier_b_pose_count: int = 10
    tier_a_budget: int = 1072
    tier_b_budget: int = 544
    static_samples_per_cell: int = 10_000
    static_retain_per_cell: int = 4
    controller_seeds_per_pose: int = 8
    local_pose_count: int = 20
    local_refine_per_pose: int = 64
    exact_candidate_count: int = 24
    manipulation_probe_count: int = 17
    manipulation_candidates_per_pose: int = 64
    manipulation_refine_pose_count: int = 8
    manipulation_refine_per_pose: int = 128
    selected_trajectory_count: int = 5
    minimum_distinct_edges: int = 3
    minimum_thumb_bands: int = 2
    maximum_per_edge_target_pair: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("normal-aligned campaign schema_version must be 1")
        edges = tuple(_finite_number(value, "edges_m") for value in self.edges_m)
        targets = tuple(
            _finite_number(value, "thumb_targets_rad")
            for value in self.thumb_targets_rad
        )
        for values, label in ((edges, "edges_m"), (targets, "thumb_targets_rad")):
            if (
                len(values) < 2
                or any(value <= 0.0 for value in values)
                or tuple(sorted(set(values))) != values
            ):
                raise ValueError(f"{label} must be positive, unique and increasing")
        object.__setattr__(self, "edges_m", edges)
        object.__setattr__(self, "thumb_targets_rad", targets)
        for label in ("fixed_mass_kg", "friction"):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        centre = tuple(
            _finite_number(value, "cube_center_xy_m")
            for value in self.cube_center_xy_m
        )
        if len(centre) != 2:
            raise ValueError("cube_center_xy_m must contain two values")
        object.__setattr__(self, "cube_center_xy_m", centre)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        integer_fields = (
            "old_pose_count", "tier_a_pose_count", "tier_b_pose_count",
            "tier_a_budget", "tier_b_budget", "static_samples_per_cell",
            "static_retain_per_cell", "controller_seeds_per_pose",
            "local_pose_count", "local_refine_per_pose", "exact_candidate_count",
            "manipulation_probe_count", "manipulation_candidates_per_pose",
            "manipulation_refine_pose_count", "manipulation_refine_per_pose",
            "selected_trajectory_count", "minimum_distinct_edges",
            "minimum_thumb_bands", "maximum_per_edge_target_pair",
        )
        for label in integer_fields:
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.tier_a_pose_count + self.tier_b_pose_count != self.old_pose_count:
            raise ValueError("rescue tier counts must sum to old_pose_count")

    @property
    def cell_count(self) -> int:
        return len(self.edges_m) * len(self.thumb_targets_rad)

    @property
    def static_sample_count(self) -> int:
        return self.cell_count * self.static_samples_per_cell

    @property
    def rescue_budget(self) -> int:
        return self.tier_a_budget + self.tier_b_budget

    def budget_config(self) -> dict[str, int]:
        return {
            "old_pose_count": self.old_pose_count,
            "tier_a_pose_count": self.tier_a_pose_count,
            "tier_b_pose_count": self.tier_b_pose_count,
            "tier_a_budget": self.tier_a_budget,
            "tier_b_budget": self.tier_b_budget,
            "rescue_budget": self.rescue_budget,
            "cell_count": self.cell_count,
            "static_samples_per_cell": self.static_samples_per_cell,
            "static_sample_count": self.static_sample_count,
            "static_retain_per_cell": self.static_retain_per_cell,
            "controller_seeds_per_pose": self.controller_seeds_per_pose,
            "local_pose_count": self.local_pose_count,
            "local_refine_per_pose": self.local_refine_per_pose,
            "exact_candidate_count": self.exact_candidate_count,
            "manipulation_probe_count": self.manipulation_probe_count,
            "manipulation_candidates_per_pose": self.manipulation_candidates_per_pose,
            "manipulation_refine_pose_count": self.manipulation_refine_pose_count,
            "manipulation_refine_per_pose": self.manipulation_refine_per_pose,
        }

    def as_config(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "edges_m": list(self.edges_m),
            "thumb_targets_rad": list(self.thumb_targets_rad),
            "fixed_mass_kg": self.fixed_mass_kg,
            "friction": self.friction,
            "cube_center_xy_m": list(self.cube_center_xy_m),
            "seed": self.seed,
            "budget": self.budget_config(),
            "selection": {
                "selected_trajectory_count": self.selected_trajectory_count,
                "minimum_distinct_edges": self.minimum_distinct_edges,
                "minimum_thumb_bands": self.minimum_thumb_bands,
                "maximum_per_edge_target_pair": self.maximum_per_edge_target_pair,
            },
            "pose_controller_identity": "separate_sha256",
            "new_pose_search_trigger": (
                "always_for_thumb_band_diversity_or_when_rescue_has_fewer_than_five"
            ),
        }


@dataclass(frozen=True)
class ActualContactGraspPoseCampaignParameters:
    """Versioned direct contact-pose and manipulation search plan."""

    schema_version: int
    edges_m: tuple[float, ...]
    thumb_actual_centers_rad: tuple[float, ...]
    fixed_mass_kg: float
    friction: float
    cube_center_xy_m: tuple[float, float]
    cube_yaw_deg: float
    seed: int
    witness_signed_gap_m: tuple[float, float]
    min_witness_normal_alignment: float
    min_witness_edge_margin_m: float
    max_contact_height_spread_m: float
    precontact_retreat_m: tuple[float, float]
    source_pose_manifest: str = (
        "artifacts/left_opposed_face_palm_down_high_thumb_normal_aligned_"
        "smooth_vertical_lift/tune/new_pose_static/source_pose_manifest.json"
    )
    quick_static_samples_per_cell: int = 2_000
    quick_static_retain_per_cell: int = 2
    full_static_samples_per_cell: int = 10_000
    full_static_retain_per_cell: int = 4
    controller_seeds_per_pose: int = 8
    local_pose_count: int = 20
    local_refine_per_pose: int = 64
    exact_grasp_pose_count: int = 24
    manipulation_probe_count: int = 17
    manipulation_candidates_per_pose: int = 64
    manipulation_refine_pose_count: int = 8
    manipulation_refine_per_pose: int = 128
    perturbations_per_trajectory: int = 16
    selected_trajectory_count: int = 5
    minimum_distinct_edges: int = 3
    minimum_thumb_bands: int = 2
    initial_target_success_count: int = 1
    validation_labels: Mapping[str, str] | None = None
    allow_single_edge: bool = False
    grasp_pose_identity: str = "cube_hand_topology_nominal_qpos_sha256"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("actual-contact campaign schema_version must be 1")
        edges = tuple(_finite_number(value, "edges_m") for value in self.edges_m)
        centers = tuple(
            _finite_number(value, "thumb_actual_centers_rad")
            for value in self.thumb_actual_centers_rad
        )
        for values, label in (
            (edges, "edges_m"),
            (centers, "thumb_actual_centers_rad"),
        ):
            if (
                (len(values) < 2 and not (label == "edges_m" and self.allow_single_edge))
                or any(value <= 0.0 for value in values)
                or tuple(sorted(set(values))) != values
            ):
                raise ValueError(f"{label} must be positive, unique and increasing")
        object.__setattr__(self, "edges_m", edges)
        object.__setattr__(self, "thumb_actual_centers_rad", centers)
        if not isinstance(self.allow_single_edge, bool):
            raise ValueError("allow_single_edge must be boolean")
        if self.grasp_pose_identity not in (
            "cube_hand_topology_nominal_qpos_sha256",
            "cube_hand_topology_contact_point_plan_nominal_qpos_sha256",
        ):
            raise ValueError("unsupported grasp_pose_identity")
        for label in (
            "fixed_mass_kg",
            "friction",
            "min_witness_edge_margin_m",
            "max_contact_height_spread_m",
        ):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        centre = tuple(
            _finite_number(value, "cube_center_xy_m")
            for value in self.cube_center_xy_m
        )
        if len(centre) != 2:
            raise ValueError("cube_center_xy_m must contain two values")
        object.__setattr__(self, "cube_center_xy_m", centre)
        object.__setattr__(
            self, "cube_yaw_deg", _finite_number(self.cube_yaw_deg, "cube_yaw_deg")
        )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        witness_gap = _closed_range(
            self.witness_signed_gap_m, "witness_signed_gap_m"
        )
        if witness_gap[0] >= 0.0 or witness_gap[1] <= 0.0:
            raise ValueError("witness_signed_gap_m must straddle zero")
        object.__setattr__(self, "witness_signed_gap_m", witness_gap)
        normal = _finite_number(
            self.min_witness_normal_alignment, "min_witness_normal_alignment"
        )
        if not 0.0 < normal <= 1.0:
            raise ValueError("min_witness_normal_alignment must be within (0, 1]")
        object.__setattr__(self, "min_witness_normal_alignment", normal)
        retreat = _closed_range(self.precontact_retreat_m, "precontact_retreat_m")
        if retreat[0] <= 0.0 or math.isclose(retreat[0], retreat[1]):
            raise ValueError("precontact_retreat_m must be a non-empty positive range")
        object.__setattr__(self, "precontact_retreat_m", retreat)
        source_path = PurePosixPath(str(self.source_pose_manifest))
        if (
            source_path.is_absolute()
            or ".." in source_path.parts
            or source_path.suffix != ".json"
        ):
            raise ValueError(
                "source_pose_manifest must be a repository-relative JSON path"
            )
        object.__setattr__(self, "source_pose_manifest", str(source_path))
        integer_fields = (
            "quick_static_samples_per_cell",
            "quick_static_retain_per_cell",
            "full_static_samples_per_cell",
            "full_static_retain_per_cell",
            "controller_seeds_per_pose",
            "local_pose_count",
            "local_refine_per_pose",
            "exact_grasp_pose_count",
            "manipulation_probe_count",
            "manipulation_candidates_per_pose",
            "manipulation_refine_pose_count",
            "manipulation_refine_per_pose",
            "perturbations_per_trajectory",
            "selected_trajectory_count",
            "minimum_distinct_edges",
            "minimum_thumb_bands",
            "initial_target_success_count",
        )
        for label in integer_fields:
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.quick_static_samples_per_cell >= self.full_static_samples_per_cell:
            raise ValueError("quick static budget must be smaller than full budget")
        if self.quick_static_retain_per_cell > self.full_static_retain_per_cell:
            raise ValueError("quick retain count must not exceed full retain count")
        if self.initial_target_success_count > self.selected_trajectory_count:
            raise ValueError(
                "initial_target_success_count must not exceed selected_trajectory_count"
            )
        if self.minimum_distinct_edges > len(edges):
            raise ValueError("minimum_distinct_edges exceeds the edge campaign")
        if self.minimum_thumb_bands > len(centers):
            raise ValueError("minimum_thumb_bands exceeds the thumb campaign")
        if self.validation_labels is not None:
            labels = dict(self.validation_labels)
            expected = {"grasp", "manipulation", "robust"}
            if set(labels) != expected or any(
                not isinstance(value, str)
                or _EXPERIMENT_ID_PATTERN.fullmatch(value) is None
                for value in labels.values()
            ):
                raise ValueError(
                    "validation_labels must contain safe grasp, manipulation "
                    "and robust labels"
                )
            object.__setattr__(self, "validation_labels", MappingProxyType(labels))

    @property
    def cell_count(self) -> int:
        return len(self.edges_m) * len(self.thumb_actual_centers_rad)

    @property
    def quick_static_sample_count(self) -> int:
        return self.cell_count * self.quick_static_samples_per_cell

    @property
    def full_static_sample_count(self) -> int:
        return self.cell_count * self.full_static_samples_per_cell

    @property
    def maximum_dynamic_grasp_candidate_count(self) -> int:
        return (
            self.cell_count
            * self.full_static_retain_per_cell
            * self.controller_seeds_per_pose
        )

    def budget_config(self) -> dict[str, int]:
        return {
            "cell_count": self.cell_count,
            "quick_static_samples_per_cell": self.quick_static_samples_per_cell,
            "quick_static_sample_count": self.quick_static_sample_count,
            "quick_static_retain_per_cell": self.quick_static_retain_per_cell,
            "full_static_samples_per_cell": self.full_static_samples_per_cell,
            "full_static_sample_count": self.full_static_sample_count,
            "full_static_retain_per_cell": self.full_static_retain_per_cell,
            "controller_seeds_per_pose": self.controller_seeds_per_pose,
            "maximum_dynamic_grasp_candidate_count": (
                self.maximum_dynamic_grasp_candidate_count
            ),
            "local_pose_count": self.local_pose_count,
            "local_refine_per_pose": self.local_refine_per_pose,
            "exact_grasp_pose_count": self.exact_grasp_pose_count,
            "manipulation_probe_count": self.manipulation_probe_count,
            "manipulation_candidates_per_pose": self.manipulation_candidates_per_pose,
            "manipulation_refine_pose_count": self.manipulation_refine_pose_count,
            "manipulation_refine_per_pose": self.manipulation_refine_per_pose,
            "perturbations_per_trajectory": self.perturbations_per_trajectory,
        }

    def as_config(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "edges_m": list(self.edges_m),
            "thumb_actual_centers_rad": list(self.thumb_actual_centers_rad),
            "fixed_mass_kg": self.fixed_mass_kg,
            "friction": self.friction,
            "cube_center_xy_m": list(self.cube_center_xy_m),
            "cube_yaw_deg": self.cube_yaw_deg,
            "seed": self.seed,
            "source_pose_manifest": self.source_pose_manifest,
            "static_contact_geometry": {
                "witness_signed_gap_m": list(self.witness_signed_gap_m),
                "min_witness_normal_alignment": self.min_witness_normal_alignment,
                "min_witness_edge_margin_m": self.min_witness_edge_margin_m,
                "max_contact_height_spread_m": self.max_contact_height_spread_m,
                "precontact_retreat_m": list(self.precontact_retreat_m),
                "cube_body_type": "freejoint",
                "static_filter_is_success_evidence": False,
            },
            "budget": self.budget_config(),
            "selection": {
                "initial_target_success_count": self.initial_target_success_count,
                "selected_trajectory_count": self.selected_trajectory_count,
                "minimum_distinct_edges": self.minimum_distinct_edges,
                "minimum_thumb_bands": self.minimum_thumb_bands,
            },
            "identity": {
                "grasp_pose": self.grasp_pose_identity,
                "controller": "precontact_preload_timing_manipulation_sha256",
            },
            "checkpoint_policy": {
                "search_probes_may_restore_latched_free_dynamics": True,
                "final_trajectories_require_full_reset_rerun": True,
            },
        }
        if self.validation_labels is not None:
            result["validation_labels"] = dict(self.validation_labels)
        if self.allow_single_edge:
            result["single_edge_campaign"] = True
        return result


SCALED_CONTACT_MAPPING_MODES = (
    "proportional_face_yz",
    "absolute_face_yz",
)


@dataclass(frozen=True)
class ScaledContactDownsizeCampaignParameters:
    """Versioned fixed-hand-scale continuation from authenticated grasp data.

    The cube becomes smaller while the XHAND geometry remains unchanged.  A
    mapping therefore defines only the desired point on the resized cube face;
    it does not scale the hand or declare a successful grasp.  Every published
    result still requires a fresh free-body dynamics run.
    """

    schema_version: int
    reference_edge_m: float
    edges_m: tuple[float, ...]
    edge_step_m: float
    fixed_mass_kg: float
    friction: float
    seed: int
    source_catalog: str
    source_manifest: str
    source_aliases: tuple[str, ...]
    mapping_modes: tuple[str, ...]
    contact_target_radius_m: float
    minimum_edge_guard_m: float
    dls_starts_per_stratum: int
    max_dls_iterations: int
    static_retain_per_stratum: int
    controllers_per_pose: int
    local_poses_per_edge_mapping: int
    local_refine_per_pose: int
    selected_grasps_per_edge: int
    manipulation_probe_count: int
    manipulation_candidates_per_grasp: int
    manipulation_edge_refine_per_best: int
    global_refine_pose_count: int
    global_refine_per_pose: int
    perturbations_per_published: int
    robustness_trials: int
    robustness_required_passes: int

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                "scaled_contact_downsize_campaign.schema_version must be 1"
            )
        reference = _finite_number(self.reference_edge_m, "reference_edge_m")
        step = _finite_number(self.edge_step_m, "edge_step_m")
        edges = tuple(_finite_number(value, "edges_m") for value in self.edges_m)
        if reference <= 0.0 or step <= 0.0:
            raise ValueError("reference_edge_m and edge_step_m must be positive")
        if (
            not edges
            or any(value <= 0.0 or value >= reference for value in edges)
            or tuple(sorted(set(edges))) != edges
        ):
            raise ValueError(
                "edges_m must be positive, unique, increasing and below reference_edge_m"
            )
        if any(
            not math.isclose(
                right - left,
                step,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for left, right in zip(edges, edges[1:])
        ):
            raise ValueError("edges_m must form a contiguous edge_step_m grid")
        object.__setattr__(self, "reference_edge_m", reference)
        object.__setattr__(self, "edge_step_m", step)
        object.__setattr__(self, "edges_m", edges)

        for label in (
            "fixed_mass_kg",
            "friction",
            "contact_target_radius_m",
            "minimum_edge_guard_m",
        ):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        if (
            self.contact_target_radius_m + self.minimum_edge_guard_m
            >= 0.5 * edges[0]
        ):
            raise ValueError(
                "contact target circle plus edge guard must fit the smallest cube face"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

        for field_name in ("source_catalog", "source_manifest"):
            path = PurePosixPath(str(getattr(self, field_name)))
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.parts[:1] != ("artifacts",)
                or path.suffix != ".json"
            ):
                raise ValueError(
                    f"{field_name} must be a repository-relative JSON path below artifacts/"
                )
            object.__setattr__(self, field_name, str(path))

        aliases = tuple(str(value) for value in self.source_aliases)
        if (
            not aliases
            or len(set(aliases)) != len(aliases)
            or any(_EXPERIMENT_ID_PATTERN.fullmatch(value) is None for value in aliases)
        ):
            raise ValueError(
                "source_aliases must be non-empty, unique lower-case snake_case names"
            )
        object.__setattr__(self, "source_aliases", aliases)
        modes = tuple(str(value) for value in self.mapping_modes)
        if modes != SCALED_CONTACT_MAPPING_MODES:
            raise ValueError(
                "mapping_modes must be proportional_face_yz then absolute_face_yz"
            )
        object.__setattr__(self, "mapping_modes", modes)

        integer_fields = (
            "dls_starts_per_stratum",
            "max_dls_iterations",
            "static_retain_per_stratum",
            "controllers_per_pose",
            "local_poses_per_edge_mapping",
            "local_refine_per_pose",
            "selected_grasps_per_edge",
            "manipulation_probe_count",
            "manipulation_candidates_per_grasp",
            "manipulation_edge_refine_per_best",
            "global_refine_pose_count",
            "global_refine_per_pose",
            "perturbations_per_published",
            "robustness_trials",
            "robustness_required_passes",
        )
        for label in integer_fields:
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.static_retain_per_stratum > self.dls_starts_per_stratum:
            raise ValueError(
                "static_retain_per_stratum cannot exceed dls_starts_per_stratum"
            )
        if self.robustness_required_passes > self.robustness_trials:
            raise ValueError(
                "robustness_required_passes cannot exceed robustness_trials"
            )

    @property
    def stratum_count(self) -> int:
        return len(self.edges_m) * len(self.source_aliases) * len(self.mapping_modes)

    @property
    def maximum_dynamic_grasp_candidate_count(self) -> int:
        return (
            self.stratum_count
            * self.static_retain_per_stratum
            * self.controllers_per_pose
        )

    @property
    def local_pose_count(self) -> int:
        return (
            len(self.edges_m)
            * len(self.mapping_modes)
            * self.local_poses_per_edge_mapping
        )

    @property
    def selected_grasp_count(self) -> int:
        return len(self.edges_m) * self.selected_grasps_per_edge

    def mapped_face_point_m(
        self,
        *,
        face: str,
        reference_face_yz_m: Sequence[float],
        edge_m: float,
        mapping_mode: str,
    ) -> tuple[float, float, float]:
        """Resolve a reference face point on a registered smaller edge."""

        if face not in ("-X", "+X"):
            raise ValueError("scaled contact points must lie on -X or +X")
        edge = _finite_number(edge_m, "edge_m")
        if not any(
            math.isclose(edge, value, rel_tol=0.0, abs_tol=1e-12)
            for value in self.edges_m
        ):
            raise ValueError("edge_m is outside the registered downsize grid")
        yz = tuple(
            _finite_number(value, "reference_face_yz_m")
            for value in reference_face_yz_m
        )
        if len(yz) != 2:
            raise ValueError("reference_face_yz_m must contain y and z")
        if mapping_mode == "proportional_face_yz":
            scale = edge / self.reference_edge_m
            y, z = yz[0] * scale, yz[1] * scale
        elif mapping_mode == "absolute_face_yz":
            y, z = yz
        else:
            raise ValueError("mapping_mode is not registered")
        required_margin = self.contact_target_radius_m + self.minimum_edge_guard_m
        if max(abs(y), abs(z)) > 0.5 * edge - required_margin + 1e-12:
            raise ValueError(
                "mapped target circle violates the minimum edge guard"
            )
        return ((-0.5 if face == "-X" else 0.5) * edge, y, z)

    def budget_config(self) -> dict[str, int]:
        return {
            "stratum_count": self.stratum_count,
            "dls_starts_per_stratum": self.dls_starts_per_stratum,
            "max_dls_iterations": self.max_dls_iterations,
            "static_retain_per_stratum": self.static_retain_per_stratum,
            "controllers_per_pose": self.controllers_per_pose,
            "maximum_dynamic_grasp_candidate_count": (
                self.maximum_dynamic_grasp_candidate_count
            ),
            "local_poses_per_edge_mapping": self.local_poses_per_edge_mapping,
            "local_pose_count": self.local_pose_count,
            "local_refine_per_pose": self.local_refine_per_pose,
            "selected_grasps_per_edge": self.selected_grasps_per_edge,
            "selected_grasp_count": self.selected_grasp_count,
            "manipulation_probe_count": self.manipulation_probe_count,
            "manipulation_candidates_per_grasp": (
                self.manipulation_candidates_per_grasp
            ),
            "manipulation_edge_refine_per_best": (
                self.manipulation_edge_refine_per_best
            ),
            "global_refine_pose_count": self.global_refine_pose_count,
            "global_refine_per_pose": self.global_refine_per_pose,
            "perturbations_per_published": self.perturbations_per_published,
            "robustness_trials": self.robustness_trials,
            "robustness_required_passes": self.robustness_required_passes,
        }

    def as_config(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_edge_m": self.reference_edge_m,
            "edges_m": list(self.edges_m),
            "edge_step_m": self.edge_step_m,
            "fixed_mass_kg": self.fixed_mass_kg,
            "friction": self.friction,
            "seed": self.seed,
            "source_catalog": self.source_catalog,
            "source_manifest": self.source_manifest,
            "source_aliases": list(self.source_aliases),
            "mapping_modes": list(self.mapping_modes),
            "mapping_definitions": {
                "proportional_face_yz": {
                    "point": "[face_sign * edge_m / 2, (edge_m / reference_edge_m) * y_ref, (edge_m / reference_edge_m) * z_ref]",
                },
                "absolute_face_yz": {
                    "point": "[face_sign * edge_m / 2, y_ref, z_ref]",
                },
            },
            "contact_target_radius_m": self.contact_target_radius_m,
            "minimum_edge_guard_m": self.minimum_edge_guard_m,
            "cube_pose_policy": {
                "center_xy_and_yaw": "fixed_from_authenticated_source",
                "center_z": "support_top_z_m + edge_m / 2",
                "body_type": "freejoint",
                "runtime_pose_reset": False,
            },
            "mass_policy": "fixed_160g_geometry_ablation",
            "budget": self.budget_config(),
            "continuation_order": "descending_edge_m",
            "final_evidence": "fresh_full_reset_free_dynamics_only",
        }


@dataclass(frozen=True)
class ContactPointPlanParameters:
    """A frozen, hash-addressed three-finger target-point plan.

    The two stored coordinates are cube-local ``(y, z)`` values.  The normal
    coordinate is intentionally omitted from persisted input and is resolved
    from the signed ``+/-X`` face and half of ``cube_edge_m``.  This prevents a
    stale normal coordinate from disagreeing with the fixed cube size.
    """

    schema_version: int
    cube_edge_m: float
    target_faces: OpposedFaceAssignment
    target_face_yz_m: Mapping[str, tuple[float, float]]
    target_radius_m: float
    frozen: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("contact_point_plan.schema_version must be 1")
        edge = _finite_number(self.cube_edge_m, "contact_point_plan.cube_edge_m")
        if edge <= 0.0:
            raise ValueError("contact_point_plan.cube_edge_m must be positive")
        object.__setattr__(self, "cube_edge_m", edge)
        if not isinstance(self.target_faces, OpposedFaceAssignment):
            raise TypeError("contact_point_plan.target_faces must be OpposedFaceAssignment")
        if set(self.target_faces.as_dict().values()) != {"-X", "+X"}:
            raise ValueError("contact-point targets must use opposed cube-local +/-X faces")
        if set(self.target_face_yz_m) != set(FINGERS):
            raise ValueError("target_face_yz_m must contain exactly thumb, index and mid")
        points: dict[str, tuple[float, float]] = {}
        for finger in FINGERS:
            values = tuple(
                _finite_number(value, f"target_face_yz_m.{finger}")
                for value in self.target_face_yz_m[finger]
            )
            if len(values) != 2:
                raise ValueError(
                    f"target_face_yz_m.{finger} must contain cube-local y and z"
                )
            if any(abs(value) >= edge / 2.0 for value in values):
                raise ValueError(
                    f"target_face_yz_m.{finger} must lie strictly inside the face"
                )
            points[finger] = values
        object.__setattr__(self, "target_face_yz_m", MappingProxyType(points))
        radius = _finite_number(
            self.target_radius_m, "contact_point_plan.target_radius_m"
        )
        if radius <= 0.0 or radius >= edge / 2.0:
            raise ValueError("contact_point_plan.target_radius_m is outside the cube face")
        object.__setattr__(self, "target_radius_m", radius)
        if self.frozen is not True:
            raise ValueError("contact_point_plan must be frozen")

    def resolved_point_cube_local_m(self, finger: str) -> tuple[float, float, float]:
        if finger not in FINGERS:
            raise ValueError(f"unknown finger {finger!r}")
        face = getattr(self.target_faces, finger)
        x = self.cube_edge_m / 2.0 if face == "+X" else -self.cube_edge_m / 2.0
        y, z = self.target_face_yz_m[finger]
        return (x, y, z)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cube_edge_m": self.cube_edge_m,
            "target_points": {
                finger: {
                    "face": getattr(self.target_faces, finger),
                    "face_yz_m": list(self.target_face_yz_m[finger]),
                }
                for finger in FINGERS
            },
            "target_radius_m": self.target_radius_m,
            "frozen": True,
        }

    @property
    def point_plan_id(self) -> str:
        canonical = json.dumps(
            self._identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def as_config(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "point_plan_id": self.point_plan_id,
            "cube_edge_m": self.cube_edge_m,
            "coordinate_frame": "cube_local",
            "target_points": self._identity_payload()["target_points"],
            "target_points_cube_local_m": {
                finger: list(self.resolved_point_cube_local_m(finger))
                for finger in FINGERS
            },
            "target_radius_m": self.target_radius_m,
            "frozen": True,
            "identity_binding": "grasp_pose_id_includes_point_plan_id",
        }

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "ContactPointPlanParameters":
        expected_fields = {
            "schema_version",
            "point_plan_id",
            "cube_edge_m",
            "coordinate_frame",
            "target_points",
            "target_points_cube_local_m",
            "target_radius_m",
            "frozen",
            "identity_binding",
        }
        if set(value) != expected_fields:
            raise ValueError("contact_point_plan contains unexpected or missing fields")
        if value["coordinate_frame"] != "cube_local":
            raise ValueError("contact_point_plan coordinate_frame must be cube_local")
        if value["identity_binding"] != "grasp_pose_id_includes_point_plan_id":
            raise ValueError("contact_point_plan has the wrong identity binding")
        target_points = value["target_points"]
        if not isinstance(target_points, Mapping) or set(target_points) != set(FINGERS):
            raise ValueError("contact_point_plan.target_points must contain three fingers")
        faces: dict[str, str] = {}
        yz: dict[str, tuple[float, float]] = {}
        for finger in FINGERS:
            entry = target_points[finger]
            if not isinstance(entry, Mapping) or set(entry) != {"face", "face_yz_m"}:
                raise ValueError(
                    f"contact_point_plan.target_points.{finger} must contain face and face_yz_m"
                )
            faces[finger] = str(entry["face"])
            yz[finger] = tuple(entry["face_yz_m"])
        plan = cls(
            schema_version=int(value["schema_version"]),
            cube_edge_m=float(value["cube_edge_m"]),
            target_faces=OpposedFaceAssignment.from_mapping(faces),
            target_face_yz_m=yz,
            target_radius_m=float(value["target_radius_m"]),
            frozen=value["frozen"],
        )
        if value["point_plan_id"] != plan.point_plan_id:
            raise ValueError("contact_point_plan point_plan_id does not match its content")
        resolved_points = value["target_points_cube_local_m"]
        if not isinstance(resolved_points, Mapping) or set(resolved_points) != set(FINGERS):
            raise ValueError(
                "contact_point_plan.target_points_cube_local_m must contain three fingers"
            )
        for finger in FINGERS:
            declared = tuple(
                _finite_number(value, f"target_points_cube_local_m.{finger}")
                for value in resolved_points[finger]
            )
            if len(declared) != 3 or any(
                not math.isclose(actual, expected, abs_tol=1e-12)
                for actual, expected in zip(
                    declared, plan.resolved_point_cube_local_m(finger)
                )
            ):
                raise ValueError(
                    "derived target_points_cube_local_m does not match face/face_yz_m"
                )
        return plan


@dataclass(frozen=True)
class ScaledContactMappingParameters:
    """Authenticated derivation of a resized contact-point plan.

    The evidence digest binds the measured stable-window centroids to one of
    the three registered source aliases.  The target values are then derived
    mechanically by a registered mapping mode; they are not independently
    searchable configuration.
    """

    schema_version: int
    source_alias: str
    mapping_mode: str
    config_sha256: str
    result_sha256: str
    trace_sha256: str
    stable_window_start_step: int
    stable_window_end_step: int
    reference_edge_m: float
    reference_target_face_yz_m: Mapping[str, tuple[float, float]]
    reference_contact_evidence_sha256: str
    target_edge_m: float
    derived_target_face_yz_m: Mapping[str, tuple[float, float]]
    contact_point_plan_id: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("scaled_contact_mapping.schema_version must be 1")
        if _EXPERIMENT_ID_PATTERN.fullmatch(str(self.source_alias)) is None:
            raise ValueError("scaled_contact_mapping.source_alias must be snake_case")
        if self.mapping_mode not in SCALED_CONTACT_MAPPING_MODES:
            raise ValueError("scaled_contact_mapping.mapping_mode is not registered")
        digest_fields = (
            "config_sha256",
            "result_sha256",
            "trace_sha256",
            "reference_contact_evidence_sha256",
            "contact_point_plan_id",
        )
        for label in digest_fields:
            digest = str(getattr(self, label))
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"scaled_contact_mapping.{label} must be lower-case SHA-256")
            object.__setattr__(self, label, digest)
        for label in ("stable_window_start_step", "stable_window_end_step"):
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"scaled_contact_mapping.{label} must be non-negative")
        if self.stable_window_end_step < self.stable_window_start_step:
            raise ValueError("scaled_contact_mapping stable window is reversed")
        for label in ("reference_edge_m", "target_edge_m"):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"scaled_contact_mapping.{label} must be positive")
            object.__setattr__(self, label, value)
        for label in (
            "reference_target_face_yz_m",
            "derived_target_face_yz_m",
        ):
            raw = getattr(self, label)
            if set(raw) != set(FINGERS):
                raise ValueError(
                    f"scaled_contact_mapping.{label} must contain three fingers"
                )
            points: dict[str, tuple[float, float]] = {}
            for finger in FINGERS:
                point = tuple(
                    _finite_number(value, f"{label}.{finger}")
                    for value in raw[finger]
                )
                if len(point) != 2:
                    raise ValueError(f"{label}.{finger} must contain y and z")
                points[finger] = point
            object.__setattr__(self, label, MappingProxyType(points))
        if self.reference_contact_evidence_sha256 != self.computed_evidence_sha256:
            raise ValueError(
                "reference_contact_evidence_sha256 does not match its canonical payload"
            )

    def evidence_payload(self) -> dict[str, Any]:
        return {
            "source_alias": self.source_alias,
            "config_sha256": self.config_sha256,
            "result_sha256": self.result_sha256,
            "trace_sha256": self.trace_sha256,
            "stable_window_start_step": self.stable_window_start_step,
            "stable_window_end_step": self.stable_window_end_step,
            "reference_edge_m": self.reference_edge_m,
            "reference_target_face_yz_m": {
                finger: list(self.reference_target_face_yz_m[finger])
                for finger in FINGERS
            },
        }

    @property
    def computed_evidence_sha256(self) -> str:
        canonical = json.dumps(
            self.evidence_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def validate_for_campaign(
        self,
        campaign: ScaledContactDownsizeCampaignParameters,
        point_plan: ContactPointPlanParameters,
    ) -> ContactPointPlanParameters:
        if self.source_alias not in campaign.source_aliases:
            raise ValueError("scaled contact source alias is not registered")
        if self.mapping_mode not in campaign.mapping_modes:
            raise ValueError("scaled contact mapping mode is not registered")
        if not math.isclose(
            self.reference_edge_m,
            campaign.reference_edge_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("scaled contact reference edge does not match campaign")
        if not math.isclose(
            self.target_edge_m,
            point_plan.cube_edge_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("scaled contact target edge does not match point plan")
        if not math.isclose(
            point_plan.target_radius_m,
            campaign.contact_target_radius_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("scaled contact target radius does not match campaign")
        if self.contact_point_plan_id != point_plan.point_plan_id:
            raise ValueError("scaled contact mapping does not bind the point plan")
        for finger in FINGERS:
            face = getattr(point_plan.target_faces, finger)
            mapped = campaign.mapped_face_point_m(
                face=face,
                reference_face_yz_m=self.reference_target_face_yz_m[finger],
                edge_m=self.target_edge_m,
                mapping_mode=self.mapping_mode,
            )
            expected_yz = mapped[1:]
            for declared, expected in zip(
                self.derived_target_face_yz_m[finger], expected_yz
            ):
                if not math.isclose(
                    declared, expected, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        "derived target face coordinates do not match mapping provenance"
                    )
            for actual, expected in zip(
                point_plan.target_face_yz_m[finger], expected_yz
            ):
                if not math.isclose(
                    actual, expected, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        "contact point plan does not match scaled mapping provenance"
                    )
        return point_plan

    def as_config(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            **self.evidence_payload(),
            "reference_contact_evidence_sha256": (
                self.reference_contact_evidence_sha256
            ),
            "mapping_mode": self.mapping_mode,
            "target_edge_m": self.target_edge_m,
            "derived_target_face_yz_m": {
                finger: list(self.derived_target_face_yz_m[finger])
                for finger in FINGERS
            },
            "contact_point_plan_id": self.contact_point_plan_id,
        }

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "ScaledContactMappingParameters":
        expected = {
            "schema_version",
            "source_alias",
            "mapping_mode",
            "config_sha256",
            "result_sha256",
            "trace_sha256",
            "stable_window_start_step",
            "stable_window_end_step",
            "reference_edge_m",
            "reference_target_face_yz_m",
            "reference_contact_evidence_sha256",
            "target_edge_m",
            "derived_target_face_yz_m",
            "contact_point_plan_id",
        }
        if set(value) != expected:
            raise ValueError(
                "scaled_contact_mapping contains unexpected or missing fields"
            )
        return cls(
            schema_version=int(value["schema_version"]),
            source_alias=str(value["source_alias"]),
            mapping_mode=str(value["mapping_mode"]),
            config_sha256=str(value["config_sha256"]),
            result_sha256=str(value["result_sha256"]),
            trace_sha256=str(value["trace_sha256"]),
            stable_window_start_step=int(value["stable_window_start_step"]),
            stable_window_end_step=int(value["stable_window_end_step"]),
            reference_edge_m=float(value["reference_edge_m"]),
            reference_target_face_yz_m={
                finger: tuple(value["reference_target_face_yz_m"][finger])
                for finger in FINGERS
            },
            reference_contact_evidence_sha256=str(
                value["reference_contact_evidence_sha256"]
            ),
            target_edge_m=float(value["target_edge_m"]),
            derived_target_face_yz_m={
                finger: tuple(value["derived_target_face_yz_m"][finger])
                for finger in FINGERS
            },
            contact_point_plan_id=str(value["contact_point_plan_id"]),
        )


@dataclass(frozen=True)
class ContactPointSearchParameters:
    """Versioned schema-v12 contact-point selection and grasp-search policy."""

    schema_version: int
    seed_plan: ContactPointPlanParameters
    seed: int
    face_yz_delta_m: tuple[float, float]
    point_group_sample_count: int
    retained_point_plan_count: int
    signed_clockwise_orbit_deg: tuple[float, ...]
    source_pose_candidate_ids: tuple[int, ...]
    max_reachability_screen_count: int
    min_target_edge_margin_m: float
    max_target_height_spread_m: float
    min_index_middle_separation_m: float
    static_target_error_radius_m: float
    retained_static_pose_count: int
    controllers_per_pose: int
    local_pose_count: int
    local_refine_per_pose: int
    exact_candidate_count: int
    selected_trajectory_count: int
    perturbations_per_trajectory: int
    manipulation_probe_count: int
    manipulation_max_delta_rad: float
    virtual_lift_target_m: float

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("contact_point_search.schema_version must be 1")
        if not isinstance(self.seed_plan, ContactPointPlanParameters):
            raise TypeError("seed_plan must be ContactPointPlanParameters")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("contact_point_search.seed must be a non-negative integer")
        delta = _closed_range(self.face_yz_delta_m, "face_yz_delta_m")
        if not delta[0] < 0.0 < delta[1]:
            raise ValueError("face_yz_delta_m must straddle zero")
        object.__setattr__(self, "face_yz_delta_m", delta)
        orbit = tuple(
            _finite_number(value, "signed_clockwise_orbit_deg")
            for value in self.signed_clockwise_orbit_deg
        )
        if not orbit or tuple(sorted(set(orbit))) != orbit or 0.0 not in orbit:
            raise ValueError(
                "signed_clockwise_orbit_deg must be unique, increasing and include zero"
            )
        object.__setattr__(self, "signed_clockwise_orbit_deg", orbit)
        sources = tuple(self.source_pose_candidate_ids)
        if (
            not sources
            or len(set(sources)) != len(sources)
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in sources)
        ):
            raise ValueError("source_pose_candidate_ids must be unique positive integers")
        object.__setattr__(self, "source_pose_candidate_ids", sources)
        integer_fields = (
            "point_group_sample_count",
            "retained_point_plan_count",
            "max_reachability_screen_count",
            "retained_static_pose_count",
            "controllers_per_pose",
            "local_pose_count",
            "local_refine_per_pose",
            "exact_candidate_count",
            "selected_trajectory_count",
            "perturbations_per_trajectory",
            "manipulation_probe_count",
        )
        for label in integer_fields:
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.retained_point_plan_count > self.point_group_sample_count:
            raise ValueError("retained point plans cannot exceed sampled point groups")
        if self.local_pose_count > self.retained_static_pose_count:
            raise ValueError("local pose count cannot exceed retained static poses")
        if self.selected_trajectory_count > self.exact_candidate_count:
            raise ValueError("selected trajectories cannot exceed exact candidates")
        for label in (
            "min_target_edge_margin_m",
            "max_target_height_spread_m",
            "min_index_middle_separation_m",
            "static_target_error_radius_m",
            "manipulation_max_delta_rad",
            "virtual_lift_target_m",
        ):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        if self.seed_plan.target_radius_m >= self.static_target_error_radius_m:
            raise ValueError("dynamic target radius must be smaller than static radius")
        if self.max_reachability_screen_count > (
            self.retained_point_plan_count
            * len(self.signed_clockwise_orbit_deg)
            * len(self.source_pose_candidate_ids)
        ):
            raise ValueError("reachability-screen budget exceeds point/orbit/source product")
        self.validate_selected_plan(self.seed_plan)

    @property
    def dynamic_candidate_count(self) -> int:
        return self.retained_static_pose_count * self.controllers_per_pose

    @property
    def local_refinement_count(self) -> int:
        return self.local_pose_count * self.local_refine_per_pose

    def validate_selected_plan(
        self, plan: ContactPointPlanParameters
    ) -> ContactPointPlanParameters:
        if not math.isclose(
            plan.cube_edge_m, self.seed_plan.cube_edge_m, abs_tol=1e-12
        ):
            raise ValueError("selected contact-point plan must preserve cube edge")
        if plan.target_faces != self.seed_plan.target_faces:
            raise ValueError("selected contact-point plan cannot switch target faces")
        if not math.isclose(
            plan.target_radius_m, self.seed_plan.target_radius_m, abs_tol=1e-12
        ):
            raise ValueError("selected contact-point plan cannot change target radius")
        lower_delta, upper_delta = self.face_yz_delta_m
        half = plan.cube_edge_m / 2.0
        for finger in FINGERS:
            for actual, origin in zip(
                plan.target_face_yz_m[finger],
                self.seed_plan.target_face_yz_m[finger],
            ):
                delta = actual - origin
                if not lower_delta - 1e-12 <= delta <= upper_delta + 1e-12:
                    raise ValueError("selected contact point lies outside the registered search box")
                if half - abs(actual) < self.min_target_edge_margin_m - 1e-12:
                    raise ValueError("selected contact point is too close to a cube edge")
        heights = [plan.target_face_yz_m[finger][1] for finger in FINGERS]
        if max(heights) - min(heights) > self.max_target_height_spread_m + 1e-12:
            raise ValueError("selected contact points exceed the height-spread limit")
        index_y = plan.target_face_yz_m["index"][0]
        middle_y = plan.target_face_yz_m["mid"][0]
        if abs(index_y - middle_y) < self.min_index_middle_separation_m - 1e-12:
            raise ValueError("index and middle contact points are too close")
        return plan

    def as_config(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "seed_point_plan_id": self.seed_plan.point_plan_id,
            "seed_target_points": self.seed_plan.as_config()["target_points"],
            "face_yz_delta_m": list(self.face_yz_delta_m),
            "signed_clockwise_orbit_deg": list(self.signed_clockwise_orbit_deg),
            "orbit_convention": {
                "axis": "cube_local_+Z",
                "positive_direction": "clockwise_viewed_from_cube_local_+Z",
                "internal_mathematical_angle_sign": -1,
                "translation_and_orientation_orbit_together": True,
            },
            "source_pose_candidate_ids": list(self.source_pose_candidate_ids),
            "constraints": {
                "min_target_edge_margin_m": self.min_target_edge_margin_m,
                "max_target_height_spread_m": self.max_target_height_spread_m,
                "min_index_middle_separation_m": self.min_index_middle_separation_m,
                "static_target_error_radius_m": self.static_target_error_radius_m,
                "dynamic_target_error_radius_m": self.seed_plan.target_radius_m,
                "min_witness_normal_alignment": 0.95,
                "max_penetration_m": 0.002,
            },
            "optimization": {
                "method": "contact_point_orientation_aware_damped_least_squares",
                "variables": [
                    "eight_active_actual_qpos",
                    "root_translation_cube_frame_xyz",
                    "wrist_local_rotvec_xyz",
                ],
                "precontact_retreat_source": "contact_pose_point_jacobian",
                "cube_world_pose_is_not_sampled": True,
            },
            "budget": {
                "point_group_sample_count": self.point_group_sample_count,
                "retained_point_plan_count": self.retained_point_plan_count,
                "max_reachability_screen_count": self.max_reachability_screen_count,
                "retained_static_pose_count": self.retained_static_pose_count,
                "controllers_per_pose": self.controllers_per_pose,
                "maximum_dynamic_grasp_candidate_count": self.dynamic_candidate_count,
                "local_pose_count": self.local_pose_count,
                "local_refine_per_pose": self.local_refine_per_pose,
                "maximum_local_refinement_count": self.local_refinement_count,
                "exact_candidate_count": self.exact_candidate_count,
                "selected_trajectory_count": self.selected_trajectory_count,
                "perturbations_per_trajectory": self.perturbations_per_trajectory,
            },
            "manipulability_prefilter": {
                "probe_count": self.manipulation_probe_count,
                "max_active_delta_rad": self.manipulation_max_delta_rad,
                "virtual_translation_cube_world_m": [0.0, 0.0, self.virtual_lift_target_m],
                "success_evidence": False,
                "final_trajectories_require_full_reset_rerun": True,
            },
        }


@dataclass(frozen=True)
class RelativeWristPoseSearchParameters:
    """Versioned schema-v11 cube-relative six-DOF wrist search contract.

    The orbit is expressed in the cube frame and deliberately carries an
    explicit sign convention: positive values are clockwise when viewed from
    cube-local ``+Z`` and are therefore applied internally as ``Rz(-angle)``.
    Translation and local rotation-vector residuals are kept separate so a
    caller cannot rotate the wrist orientation without moving the root around
    the cube by the same orbit transform.
    """

    schema_version: int
    clockwise_orbit_deg: tuple[float, ...]
    root_delta_cube_m: Mapping[str, tuple[float, float]]
    wrist_local_rotvec_deg: Mapping[str, tuple[float, float]]
    max_wrist_local_rotvec_norm_deg: float
    root_cube_distance_m: tuple[float, float]
    primary_anchor_candidate_id: int
    primary_anchor_fraction: float
    certified_neighbor_fraction: float
    certified_neighbor_edges_m: tuple[float, ...]
    static_samples_per_stratum: int
    expansion_samples_per_stratum: int
    retained_poses_per_edge: int
    controllers_per_pose: int
    local_seed_poses_per_edge: int
    local_refine_per_pose: int
    exact_candidates_per_edge: int
    max_manipulation_pose_count: int
    manipulation_probe_count: int
    manipulation_candidates_per_pose: int
    manipulation_refine_pose_count: int
    manipulation_refine_per_pose: int
    density_revalidation_kg_m3: float
    dls_method: str = "orientation_aware_damped_least_squares"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("relative wrist-pose search schema_version must be 1")
        orbit = tuple(
            _finite_number(value, "clockwise_orbit_deg")
            for value in self.clockwise_orbit_deg
        )
        if (
            not orbit
            or orbit[0] != 0.0
            or any(value < 0.0 for value in orbit)
            or tuple(sorted(set(orbit))) != orbit
        ):
            raise ValueError(
                "clockwise_orbit_deg must start at zero and be non-negative, "
                "unique and increasing"
            )
        object.__setattr__(self, "clockwise_orbit_deg", orbit)

        for label in ("root_delta_cube_m", "wrist_local_rotvec_deg"):
            ranges = _frozen_ranges(getattr(self, label), label)
            if set(ranges) != {"x", "y", "z"}:
                raise ValueError(f"{label} must contain exactly x, y and z")
            if any(not lower < 0.0 < upper for lower, upper in ranges.values()):
                raise ValueError(f"{label} ranges must straddle zero")
            object.__setattr__(self, label, ranges)

        norm_limit = _finite_number(
            self.max_wrist_local_rotvec_norm_deg,
            "max_wrist_local_rotvec_norm_deg",
        )
        largest_axis = max(
            max(abs(lower), abs(upper))
            for lower, upper in self.wrist_local_rotvec_deg.values()
        )
        if norm_limit < largest_axis:
            raise ValueError(
                "max_wrist_local_rotvec_norm_deg must admit every per-axis bound"
            )
        object.__setattr__(
            self, "max_wrist_local_rotvec_norm_deg", norm_limit
        )

        distance = _closed_range(
            self.root_cube_distance_m, "root_cube_distance_m"
        )
        if distance[0] <= 0.0 or math.isclose(distance[0], distance[1]):
            raise ValueError("root_cube_distance_m must be a non-empty positive range")
        object.__setattr__(self, "root_cube_distance_m", distance)

        candidate_id = self.primary_anchor_candidate_id
        if (
            not isinstance(candidate_id, int)
            or isinstance(candidate_id, bool)
            or candidate_id <= 0
        ):
            raise ValueError("primary_anchor_candidate_id must be a positive integer")
        primary = _finite_number(
            self.primary_anchor_fraction, "primary_anchor_fraction"
        )
        neighbors = _finite_number(
            self.certified_neighbor_fraction, "certified_neighbor_fraction"
        )
        if (
            not 0.0 < primary < 1.0
            or not 0.0 < neighbors < 1.0
            or not math.isclose(primary + neighbors, 1.0, abs_tol=1e-12)
        ):
            raise ValueError("anchor sampling fractions must be positive and sum to one")
        object.__setattr__(self, "primary_anchor_fraction", primary)
        object.__setattr__(self, "certified_neighbor_fraction", neighbors)
        neighbor_edges = tuple(
            _finite_number(value, "certified_neighbor_edges_m")
            for value in self.certified_neighbor_edges_m
        )
        if (
            not neighbor_edges
            or any(value <= 0.0 for value in neighbor_edges)
            or tuple(sorted(set(neighbor_edges))) != neighbor_edges
        ):
            raise ValueError(
                "certified_neighbor_edges_m must be positive, unique and increasing"
            )
        object.__setattr__(self, "certified_neighbor_edges_m", neighbor_edges)

        integer_fields = (
            "static_samples_per_stratum",
            "expansion_samples_per_stratum",
            "retained_poses_per_edge",
            "controllers_per_pose",
            "local_seed_poses_per_edge",
            "local_refine_per_pose",
            "exact_candidates_per_edge",
            "max_manipulation_pose_count",
            "manipulation_probe_count",
            "manipulation_candidates_per_pose",
            "manipulation_refine_pose_count",
            "manipulation_refine_per_pose",
        )
        for label in integer_fields:
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        density = _finite_number(
            self.density_revalidation_kg_m3, "density_revalidation_kg_m3"
        )
        if density <= 0.0:
            raise ValueError("density_revalidation_kg_m3 must be positive")
        object.__setattr__(self, "density_revalidation_kg_m3", density)
        if self.dls_method != "orientation_aware_damped_least_squares":
            raise ValueError("unsupported relative wrist-pose DLS method")

    def stratum_count(self, *, edge_count: int, thumb_band_count: int) -> int:
        for value, label in (
            (edge_count, "edge_count"),
            (thumb_band_count, "thumb_band_count"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        return edge_count * thumb_band_count * len(self.clockwise_orbit_deg)

    def budget_config(
        self, *, edge_count: int, thumb_band_count: int
    ) -> dict[str, int]:
        strata = self.stratum_count(
            edge_count=edge_count, thumb_band_count=thumb_band_count
        )
        return {
            "edge_count": edge_count,
            "thumb_band_count": thumb_band_count,
            "orbit_band_count": len(self.clockwise_orbit_deg),
            "stratum_count": strata,
            "static_samples_per_stratum": self.static_samples_per_stratum,
            "static_sample_count": strata * self.static_samples_per_stratum,
            "expansion_samples_per_stratum": self.expansion_samples_per_stratum,
            "expansion_sample_count": (
                strata * self.expansion_samples_per_stratum
            ),
            "retained_poses_per_edge": self.retained_poses_per_edge,
            "controllers_per_pose": self.controllers_per_pose,
            "maximum_dynamic_grasp_candidate_count": (
                edge_count
                * self.retained_poses_per_edge
                * self.controllers_per_pose
            ),
            "local_seed_poses_per_edge": self.local_seed_poses_per_edge,
            "local_refine_per_pose": self.local_refine_per_pose,
            "maximum_local_refinement_count": (
                edge_count
                * self.local_seed_poses_per_edge
                * self.local_refine_per_pose
            ),
            "exact_candidates_per_edge": self.exact_candidates_per_edge,
            "maximum_exact_candidate_count": (
                edge_count * self.exact_candidates_per_edge
            ),
            "max_manipulation_pose_count": self.max_manipulation_pose_count,
            "manipulation_probe_count": self.manipulation_probe_count,
            "manipulation_candidates_per_pose": (
                self.manipulation_candidates_per_pose
            ),
            "manipulation_refine_pose_count": self.manipulation_refine_pose_count,
            "manipulation_refine_per_pose": self.manipulation_refine_per_pose,
        }

    def as_config(
        self, *, edge_count: int, thumb_band_count: int
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "clockwise_orbit_deg": list(self.clockwise_orbit_deg),
            "root_delta_cube_m": {
                axis: list(self.root_delta_cube_m[axis])
                for axis in ("x", "y", "z")
            },
            "wrist_local_rotvec_deg": {
                axis: list(self.wrist_local_rotvec_deg[axis])
                for axis in ("x", "y", "z")
            },
            "max_wrist_local_rotvec_norm_deg": (
                self.max_wrist_local_rotvec_norm_deg
            ),
            "root_cube_distance_m": list(self.root_cube_distance_m),
            "orbit_convention": {
                "axis": "cube_local_+Z",
                "positive_direction": "clockwise_viewed_from_cube_local_+Z",
                "internal_mathematical_angle_sign": -1,
                "translation_and_orientation_orbit_together": True,
                "world_root_pose_rederived_from_relative_transform": True,
            },
            "anchor_sampling": {
                "primary_candidate_id": self.primary_anchor_candidate_id,
                "primary_fraction": self.primary_anchor_fraction,
                "certified_neighbor_fraction": self.certified_neighbor_fraction,
                "certified_neighbor_edges_m": list(
                    self.certified_neighbor_edges_m
                ),
            },
            "optimization": {
                "method": self.dls_method,
                "adjust_root_translation_cube_frame": True,
                "adjust_wrist_local_rotvec": True,
                "adjust_non_thumb_bend_active_joints": True,
                "hold_thumb_bend_at_cell_center": True,
                "reject_forbidden_or_nondistal_contact": True,
                "max_penetration_m": 0.002,
            },
            "budget": self.budget_config(
                edge_count=edge_count, thumb_band_count=thumb_band_count
            ),
            "material_revalidation": {
                "density_kg_m3": self.density_revalidation_kg_m3,
                "run_only_after_fixed_mass_full_success": True,
                "report_separately_from_fixed_mass": True,
            },
        }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Return the lower-case SHA-256 of a JSON identity payload."""

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class ManipulationPlanParameters:
    """Hash-addressed multi-knot manipulation plan.

    Waypoints are actuator-command deltas relative to the verified contact
    preload.  This keeps measured grasp qpos, preload commands and planned
    motion separate.  The class validates only the immutable plan contract;
    the controller and planner remain independent consumers.  Schema 2 adds
    the per-knot physical response matrices required by the schema-v15
    joint-pair alignment controller.  Schema 1 remains byte-for-byte
    compatible with every sealed v14 plan.
    """

    schema_version: int
    profile: str
    duration_s: float
    knot_times_s: tuple[float, ...]
    actuator_waypoints_rad: Mapping[str, tuple[float, ...]]
    desired_cube_position_delta_m: tuple[tuple[float, float, float], ...]
    desired_cube_rotation_vector_rad: tuple[tuple[float, float, float], ...]
    max_knot_delta_rad: float
    trust_region_backtracks: int
    joint_pair_residual_jacobian_2x8: (
        tuple[tuple[tuple[float, ...], ...], ...] | None
    ) = None
    object_response_jacobian_6x8: (
        tuple[tuple[tuple[float, ...], ...], ...] | None
    ) = None
    target_force_jacobian_3x8: (
        tuple[tuple[tuple[float, ...], ...], ...] | None
    ) = None

    def __post_init__(self) -> None:
        if self.schema_version not in (1, 2):
            raise ValueError("manipulation_plan.schema_version must be 1 or 2")
        if self.profile != "piecewise_quintic_minimum_jerk":
            raise ValueError("unsupported manipulation_plan.profile")
        duration = _finite_number(self.duration_s, "manipulation_plan.duration_s")
        if duration <= 0.0:
            raise ValueError("manipulation_plan.duration_s must be positive")
        object.__setattr__(self, "duration_s", duration)

        times = tuple(
            _finite_number(value, "manipulation_plan.knot_times_s")
            for value in self.knot_times_s
        )
        if (
            len(times) < 2
            or not math.isclose(times[0], 0.0, abs_tol=1e-12)
            or any(following <= previous for previous, following in zip(times, times[1:]))
            or not math.isclose(times[-1], duration, abs_tol=1e-12)
        ):
            raise ValueError(
                "manipulation_plan.knot_times_s must start at zero, increase "
                "strictly and end at duration_s"
            )
        object.__setattr__(self, "knot_times_s", times)

        knot_limit = _finite_number(
            self.max_knot_delta_rad,
            "manipulation_plan.max_knot_delta_rad",
        )
        if knot_limit <= 0.0:
            raise ValueError("manipulation_plan.max_knot_delta_rad must be positive")
        object.__setattr__(self, "max_knot_delta_rad", knot_limit)

        if set(self.actuator_waypoints_rad) != set(ACTIVE_ACTUATORS):
            raise ValueError(
                "manipulation_plan.actuator_waypoints_rad must contain exactly "
                "the eight active actuators"
            )
        waypoints: dict[str, tuple[float, ...]] = {}
        for actuator in ACTIVE_ACTUATORS:
            values = tuple(
                _finite_number(
                    value,
                    f"manipulation_plan.actuator_waypoints_rad.{actuator}",
                )
                for value in self.actuator_waypoints_rad[actuator]
            )
            if len(values) != len(times):
                raise ValueError(
                    "each actuator waypoint sequence must match knot_times_s"
                )
            if not math.isclose(values[0], 0.0, abs_tol=1e-12):
                raise ValueError(
                    "manipulation actuator waypoints must start at zero relative "
                    "to contact preload"
                )
            if any(
                abs(following - previous) > knot_limit + 1e-12
                for previous, following in zip(values, values[1:])
            ):
                raise ValueError(
                    "adjacent manipulation actuator waypoints exceed max_knot_delta_rad"
                )
            waypoints[actuator] = values
        object.__setattr__(
            self, "actuator_waypoints_rad", MappingProxyType(waypoints)
        )

        def validate_vectors(
            raw: Sequence[Sequence[float]], label: str
        ) -> tuple[tuple[float, float, float], ...]:
            vectors: list[tuple[float, float, float]] = []
            for index, vector in enumerate(raw):
                values = tuple(
                    _finite_number(value, f"{label}[{index}]") for value in vector
                )
                if len(values) != 3:
                    raise ValueError(f"{label} entries must contain three values")
                vectors.append(values)
            if len(vectors) != len(times):
                raise ValueError(f"{label} must match knot_times_s")
            if any(not math.isclose(value, 0.0, abs_tol=1e-12) for value in vectors[0]):
                raise ValueError(f"{label} must start at zero")
            return tuple(vectors)

        positions = validate_vectors(
            self.desired_cube_position_delta_m,
            "manipulation_plan.desired_cube_position_delta_m",
        )
        rotations = validate_vectors(
            self.desired_cube_rotation_vector_rad,
            "manipulation_plan.desired_cube_rotation_vector_rad",
        )
        if positions[-1][2] <= 0.0:
            raise ValueError("manipulation plan must request a positive final lift")
        object.__setattr__(self, "desired_cube_position_delta_m", positions)
        object.__setattr__(self, "desired_cube_rotation_vector_rad", rotations)

        backtracks = self.trust_region_backtracks
        if (
            not isinstance(backtracks, int)
            or isinstance(backtracks, bool)
            or backtracks < 0
        ):
            raise ValueError(
                "manipulation_plan.trust_region_backtracks must be a non-negative integer"
            )

        response_fields = (
            ("joint_pair_residual_jacobian_2x8", 2),
            ("object_response_jacobian_6x8", 6),
            ("target_force_jacobian_3x8", 3),
        )
        if self.schema_version == 1:
            if any(getattr(self, name) is not None for name, _ in response_fields):
                raise ValueError(
                    "manipulation_plan schema 1 cannot declare response Jacobians"
                )
        else:
            for name, row_count in response_fields:
                raw = getattr(self, name)
                if raw is None:
                    raise ValueError(
                        f"manipulation_plan schema 2 requires {name}"
                    )
                matrices: list[tuple[tuple[float, ...], ...]] = []
                for knot_index, matrix in enumerate(raw):
                    rows: list[tuple[float, ...]] = []
                    for row_index, row in enumerate(matrix):
                        values = tuple(
                            _finite_number(
                                value,
                                f"manipulation_plan.{name}"
                                f"[{knot_index}][{row_index}]",
                            )
                            for value in row
                        )
                        if len(values) != len(ACTIVE_ACTUATORS):
                            raise ValueError(
                                f"manipulation_plan.{name} rows must contain "
                                "eight actuator derivatives"
                            )
                        rows.append(values)
                    if len(rows) != row_count:
                        raise ValueError(
                            f"manipulation_plan.{name} matrices must contain "
                            f"{row_count} rows"
                        )
                    matrices.append(tuple(rows))
                if len(matrices) != len(times):
                    raise ValueError(
                        f"manipulation_plan.{name} must match knot_times_s"
                    )
                object.__setattr__(self, name, tuple(matrices))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "duration_s": self.duration_s,
            "knot_times_s": list(self.knot_times_s),
            "actuator_waypoints_rad": {
                actuator: list(self.actuator_waypoints_rad[actuator])
                for actuator in ACTIVE_ACTUATORS
            },
            "desired_cube_position_delta_m": [
                list(vector) for vector in self.desired_cube_position_delta_m
            ],
            "desired_cube_rotation_vector_rad": [
                list(vector) for vector in self.desired_cube_rotation_vector_rad
            ],
            "max_knot_delta_rad": self.max_knot_delta_rad,
            "trust_region_backtracks": self.trust_region_backtracks,
        }
        if self.schema_version >= 2:
            for name in (
                "joint_pair_residual_jacobian_2x8",
                "object_response_jacobian_6x8",
                "target_force_jacobian_3x8",
            ):
                matrices = getattr(self, name)
                assert matrices is not None
                payload[name] = [
                    [list(row) for row in matrix] for matrix in matrices
                ]
        return payload

    @property
    def plan_id(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def as_config(self) -> dict[str, Any]:
        return {"plan_id": self.plan_id, **self.identity_payload()}

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "ManipulationPlanParameters":
        schema_version = int(value.get("schema_version", -1))
        expected = {
            "schema_version",
            "plan_id",
            "profile",
            "duration_s",
            "knot_times_s",
            "actuator_waypoints_rad",
            "desired_cube_position_delta_m",
            "desired_cube_rotation_vector_rad",
            "max_knot_delta_rad",
            "trust_region_backtracks",
        }
        response_fields = {
            "joint_pair_residual_jacobian_2x8",
            "object_response_jacobian_6x8",
            "target_force_jacobian_3x8",
        }
        if schema_version >= 2:
            expected.update(response_fields)
        if set(value) != expected:
            raise ValueError("manipulation_plan contains unexpected or missing fields")
        raw_waypoints = value["actuator_waypoints_rad"]
        if not isinstance(raw_waypoints, Mapping) or set(raw_waypoints) != set(
            ACTIVE_ACTUATORS
        ):
            raise ValueError(
                "manipulation_plan.actuator_waypoints_rad must contain exactly "
                "the eight active actuators"
            )
        plan = cls(
            schema_version=schema_version,
            profile=str(value["profile"]),
            duration_s=float(value["duration_s"]),
            knot_times_s=tuple(value["knot_times_s"]),
            actuator_waypoints_rad={
                actuator: tuple(raw_waypoints[actuator])
                for actuator in ACTIVE_ACTUATORS
            },
            desired_cube_position_delta_m=tuple(
                tuple(vector) for vector in value["desired_cube_position_delta_m"]
            ),
            desired_cube_rotation_vector_rad=tuple(
                tuple(vector)
                for vector in value["desired_cube_rotation_vector_rad"]
            ),
            max_knot_delta_rad=float(value["max_knot_delta_rad"]),
            trust_region_backtracks=int(value["trust_region_backtracks"]),
            joint_pair_residual_jacobian_2x8=(
                tuple(
                    tuple(tuple(row) for row in matrix)
                    for matrix in value["joint_pair_residual_jacobian_2x8"]
                )
                if schema_version >= 2
                else None
            ),
            object_response_jacobian_6x8=(
                tuple(
                    tuple(tuple(row) for row in matrix)
                    for matrix in value["object_response_jacobian_6x8"]
                )
                if schema_version >= 2
                else None
            ),
            target_force_jacobian_3x8=(
                tuple(
                    tuple(tuple(row) for row in matrix)
                    for matrix in value["target_force_jacobian_3x8"]
                )
                if schema_version >= 2
                else None
            ),
        )
        if value["plan_id"] != plan.plan_id:
            raise ValueError("manipulation_plan.plan_id does not match its content")
        return plan


@dataclass(frozen=True)
class ContactForceTargets:
    """Schema-v14 per-finger force references measured at grasp verification."""

    schema_version: int
    source: str
    minimum_n: float
    maximum_n: float
    per_finger_n: Mapping[str, float]
    operation_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("contact_force_targets_n.schema_version must be 1")
        if self.source != "verify_window_median_clamped":
            raise ValueError("unsupported contact force target source")
        minimum = _finite_number(self.minimum_n, "contact_force_targets_n.minimum_n")
        maximum = _finite_number(self.maximum_n, "contact_force_targets_n.maximum_n")
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError(
                "contact_force_targets_n minimum/maximum must form a positive range"
            )
        object.__setattr__(self, "minimum_n", minimum)
        object.__setattr__(self, "maximum_n", maximum)
        if set(self.per_finger_n) != set(FINGERS):
            raise ValueError(
                "contact_force_targets_n.per_finger_n must contain thumb, index and mid"
            )
        targets = {
            finger: _finite_number(
                self.per_finger_n[finger],
                f"contact_force_targets_n.per_finger_n.{finger}",
            )
            for finger in FINGERS
        }
        if any(not minimum <= value <= maximum for value in targets.values()):
            raise ValueError("per-finger contact force targets must lie inside the clamp")
        object.__setattr__(self, "per_finger_n", MappingProxyType(targets))
        operation_scale = _finite_number(
            self.operation_scale, "contact_force_targets_n.operation_scale"
        )
        if not 0.70 <= operation_scale <= 1.0:
            raise ValueError(
                "contact_force_targets_n.operation_scale must lie within [0.70, 1.00]"
            )
        object.__setattr__(self, "operation_scale", operation_scale)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "source": self.source,
            "minimum_n": self.minimum_n,
            "maximum_n": self.maximum_n,
            "per_finger_n": {
                finger: self.per_finger_n[finger] for finger in FINGERS
            },
        }
        # Scale 1.0 is the legacy physical/controller contract.  Omitting it
        # preserves every pre-existing target/config/controller identifier.
        if self.operation_scale != 1.0:
            payload["operation_scale"] = self.operation_scale
        return payload

    @property
    def target_id(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def as_config(self) -> dict[str, Any]:
        return {"target_id": self.target_id, **self.identity_payload()}

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "ContactForceTargets":
        expected = {
            "schema_version",
            "target_id",
            "source",
            "minimum_n",
            "maximum_n",
            "per_finger_n",
        }
        observed = set(value)
        if observed not in (expected, expected | {"operation_scale"}):
            raise ValueError(
                "contact_force_targets_n contains unexpected or missing fields"
            )
        if "operation_scale" in value and float(value["operation_scale"]) == 1.0:
            raise ValueError(
                "contact_force_targets_n.operation_scale=1.0 must use the legacy omitted form"
            )
        raw_targets = value["per_finger_n"]
        if not isinstance(raw_targets, Mapping) or set(raw_targets) != set(FINGERS):
            raise ValueError(
                "contact_force_targets_n.per_finger_n must contain thumb, index and mid"
            )
        targets = cls(
            schema_version=int(value["schema_version"]),
            source=str(value["source"]),
            minimum_n=float(value["minimum_n"]),
            maximum_n=float(value["maximum_n"]),
            per_finger_n={
                finger: float(raw_targets[finger]) for finger in FINGERS
            },
            operation_scale=float(value.get("operation_scale", 1.0)),
        )
        if value["target_id"] != targets.target_id:
            raise ValueError(
                "contact_force_targets_n.target_id does not match its content"
            )
        return targets


@dataclass(frozen=True)
class ContactFeedbackParameters:
    """Schema-v14 deterministic per-finger force-PI controller contract."""

    schema_version: int
    strategy: str
    filter_time_constant_s: float
    kp_rad_per_n: Mapping[str, float]
    ki_rad_per_n_s: Mapping[str, float]
    integral_limit_n_s: float
    correction_limit_rad: float
    rate_limit_rad_s: float
    acceleration_limit_rad_s2: float
    force_risk_n: float
    freeze_on_risk: bool
    max_loss_s: float
    recovery_behavior: str
    operation_contact_duty_min: float
    tangent_slip_freeze_threshold_m: float | None = None
    tangent_slip_abort_threshold_m: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in (1, 2):
            raise ValueError("contact_feedback.schema_version must be 1 or 2")
        if self.strategy != "per_finger_force_pi":
            raise ValueError("unsupported contact_feedback.strategy")
        if self.recovery_behavior != "freeze_and_inward_preload_then_abort":
            raise ValueError("unsupported contact_feedback.recovery_behavior")
        if self.freeze_on_risk is not True:
            raise ValueError("contact_feedback.freeze_on_risk must be true")
        for label in (
            "filter_time_constant_s",
            "integral_limit_n_s",
            "correction_limit_rad",
            "rate_limit_rad_s",
            "acceleration_limit_rad_s2",
            "force_risk_n",
            "max_loss_s",
        ):
            value = _finite_number(getattr(self, label), f"contact_feedback.{label}")
            if value <= 0.0:
                raise ValueError(f"contact_feedback.{label} must be positive")
            object.__setattr__(self, label, value)
        for field_name, allow_zero in (
            ("kp_rad_per_n", False),
            ("ki_rad_per_n_s", True),
        ):
            raw = getattr(self, field_name)
            if set(raw) != set(FINGERS):
                raise ValueError(
                    f"contact_feedback.{field_name} must contain thumb, index and mid"
                )
            gains = {
                finger: _finite_number(
                    raw[finger], f"contact_feedback.{field_name}.{finger}"
                )
                for finger in FINGERS
            }
            if any(value < 0.0 or (not allow_zero and value == 0.0) for value in gains.values()):
                raise ValueError(f"contact_feedback.{field_name} gains are invalid")
            object.__setattr__(self, field_name, MappingProxyType(gains))
        duty = _finite_number(
            self.operation_contact_duty_min,
            "contact_feedback.operation_contact_duty_min",
        )
        if not 0.0 < duty <= 1.0:
            raise ValueError(
                "contact_feedback.operation_contact_duty_min must be within (0, 1]"
            )
        object.__setattr__(self, "operation_contact_duty_min", duty)
        slip_values = (
            self.tangent_slip_freeze_threshold_m,
            self.tangent_slip_abort_threshold_m,
        )
        if self.schema_version == 1:
            if any(value is not None for value in slip_values):
                raise ValueError(
                    "contact_feedback schema 1 cannot declare tangent-slip thresholds"
                )
        else:
            if any(value is None for value in slip_values):
                raise ValueError(
                    "contact_feedback schema 2 requires tangent-slip thresholds"
                )
            freeze = _finite_number(
                self.tangent_slip_freeze_threshold_m,
                "contact_feedback.tangent_slip_freeze_threshold_m",
            )
            abort = _finite_number(
                self.tangent_slip_abort_threshold_m,
                "contact_feedback.tangent_slip_abort_threshold_m",
            )
            if freeze <= 0.0 or abort < freeze:
                raise ValueError(
                    "contact_feedback tangent-slip thresholds must satisfy "
                    "0 < freeze <= abort"
                )
            object.__setattr__(self, "tangent_slip_freeze_threshold_m", freeze)
            object.__setattr__(self, "tangent_slip_abort_threshold_m", abort)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "filter_time_constant_s": self.filter_time_constant_s,
            "kp_rad_per_n": {
                finger: self.kp_rad_per_n[finger] for finger in FINGERS
            },
            "ki_rad_per_n_s": {
                finger: self.ki_rad_per_n_s[finger] for finger in FINGERS
            },
            "integral_limit_n_s": self.integral_limit_n_s,
            "correction_limit_rad": self.correction_limit_rad,
            "rate_limit_rad_s": self.rate_limit_rad_s,
            "acceleration_limit_rad_s2": self.acceleration_limit_rad_s2,
            "force_risk_n": self.force_risk_n,
            "freeze_on_risk": self.freeze_on_risk,
            "max_loss_s": self.max_loss_s,
            "recovery_behavior": self.recovery_behavior,
            "operation_contact_duty_min": self.operation_contact_duty_min,
        }
        if self.schema_version >= 2:
            payload.update(
                {
                    "tangent_slip_freeze_threshold_m": (
                        self.tangent_slip_freeze_threshold_m
                    ),
                    "tangent_slip_abort_threshold_m": (
                        self.tangent_slip_abort_threshold_m
                    ),
                }
            )
        return payload

    @property
    def feedback_id(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def as_config(self) -> dict[str, Any]:
        return {"feedback_id": self.feedback_id, **self.identity_payload()}

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "ContactFeedbackParameters":
        schema_version = int(value.get("schema_version", -1))
        expected = {
            "schema_version",
            "feedback_id",
            "strategy",
            "filter_time_constant_s",
            "kp_rad_per_n",
            "ki_rad_per_n_s",
            "integral_limit_n_s",
            "correction_limit_rad",
            "rate_limit_rad_s",
            "acceleration_limit_rad_s2",
            "force_risk_n",
            "freeze_on_risk",
            "max_loss_s",
            "recovery_behavior",
            "operation_contact_duty_min",
        }
        if schema_version >= 2:
            expected.update(
                {
                    "tangent_slip_freeze_threshold_m",
                    "tangent_slip_abort_threshold_m",
                }
            )
        if set(value) != expected:
            raise ValueError("contact_feedback contains unexpected or missing fields")
        raw_kp = value["kp_rad_per_n"]
        raw_ki = value["ki_rad_per_n_s"]
        for raw, label in (
            (raw_kp, "kp_rad_per_n"),
            (raw_ki, "ki_rad_per_n_s"),
        ):
            if not isinstance(raw, Mapping) or set(raw) != set(FINGERS):
                raise ValueError(
                    f"contact_feedback.{label} must contain thumb, index and mid"
                )
        feedback = cls(
            schema_version=schema_version,
            strategy=str(value["strategy"]),
            filter_time_constant_s=float(value["filter_time_constant_s"]),
            kp_rad_per_n={
                finger: float(raw_kp[finger]) for finger in FINGERS
            },
            ki_rad_per_n_s={
                finger: float(raw_ki[finger]) for finger in FINGERS
            },
            integral_limit_n_s=float(value["integral_limit_n_s"]),
            correction_limit_rad=float(value["correction_limit_rad"]),
            rate_limit_rad_s=float(value["rate_limit_rad_s"]),
            acceleration_limit_rad_s2=float(value["acceleration_limit_rad_s2"]),
            force_risk_n=float(value["force_risk_n"]),
            freeze_on_risk=value["freeze_on_risk"],
            max_loss_s=float(value["max_loss_s"]),
            recovery_behavior=str(value["recovery_behavior"]),
            operation_contact_duty_min=float(value["operation_contact_duty_min"]),
            tangent_slip_freeze_threshold_m=(
                float(value["tangent_slip_freeze_threshold_m"])
                if schema_version >= 2
                else None
            ),
            tangent_slip_abort_threshold_m=(
                float(value["tangent_slip_abort_threshold_m"])
                if schema_version >= 2
                else None
            ),
        )
        if value["feedback_id"] != feedback.feedback_id:
            raise ValueError("contact_feedback.feedback_id does not match its content")
        return feedback


@dataclass(frozen=True)
class JointPairAlignmentSettings:
    """Immutable schema-v15 index-to-middle alignment acceptance contract."""

    schema_version: int
    joint_names: tuple[str, str]
    frame: str
    axis: str
    require_positive_y: bool
    minimum_length_m: float
    residual: str
    grasp_p95_max_deg: float
    grasp_max_deg: float
    operation_p95_max_deg: float
    operation_max_deg: float
    operation_within_p95_limit_duty_min: float
    max_continuous_violation_s: float
    constraint_polygon_sides: int
    audit_timestep_s: float

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("joint_pair_alignment.schema_version must be 1")
        names = tuple(str(value) for value in self.joint_names)
        if names != (
            "left_hand_index_joint1",
            "left_hand_mid_joint1",
        ):
            raise ValueError(
                "joint_pair_alignment.joint_names must define the directed "
                "index-joint1 to middle-joint1 pair"
            )
        object.__setattr__(self, "joint_names", names)
        if self.frame != "cube_local":
            raise ValueError("joint_pair_alignment.frame must be cube_local")
        if self.axis != "+Y":
            raise ValueError("joint_pair_alignment.axis must be +Y")
        if self.require_positive_y is not True:
            raise ValueError(
                "joint_pair_alignment.require_positive_y must be true"
            )
        if self.residual != "vx_over_vy_vz_over_vy":
            raise ValueError("unsupported joint_pair_alignment.residual")
        for label in (
            "minimum_length_m",
            "grasp_p95_max_deg",
            "grasp_max_deg",
            "operation_p95_max_deg",
            "operation_max_deg",
            "max_continuous_violation_s",
            "audit_timestep_s",
        ):
            value = _finite_number(
                getattr(self, label), f"joint_pair_alignment.{label}"
            )
            if value <= 0.0:
                raise ValueError(
                    f"joint_pair_alignment.{label} must be positive"
                )
            object.__setattr__(self, label, value)
        if self.grasp_p95_max_deg > self.grasp_max_deg:
            raise ValueError(
                "joint_pair_alignment grasp p95 cannot exceed its maximum"
            )
        if self.operation_p95_max_deg > self.operation_max_deg:
            raise ValueError(
                "joint_pair_alignment operation p95 cannot exceed its maximum"
            )
        duty = _finite_number(
            self.operation_within_p95_limit_duty_min,
            "joint_pair_alignment.operation_within_p95_limit_duty_min",
        )
        if not 0.0 < duty <= 1.0:
            raise ValueError(
                "joint_pair_alignment operation duty must be within (0, 1]"
            )
        object.__setattr__(
            self, "operation_within_p95_limit_duty_min", duty
        )
        if (
            not isinstance(self.constraint_polygon_sides, int)
            or isinstance(self.constraint_polygon_sides, bool)
            or self.constraint_polygon_sides != 8
        ):
            raise ValueError(
                "joint_pair_alignment.constraint_polygon_sides must be 8"
            )
        if self.audit_timestep_s > self.max_continuous_violation_s:
            raise ValueError(
                "joint-pair audit timestep cannot exceed the violation window"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "joint_names": list(self.joint_names),
            "frame": self.frame,
            "axis": self.axis,
            "require_positive_y": self.require_positive_y,
            "minimum_length_m": self.minimum_length_m,
            "residual": self.residual,
            "grasp_p95_max_deg": self.grasp_p95_max_deg,
            "grasp_max_deg": self.grasp_max_deg,
            "operation_p95_max_deg": self.operation_p95_max_deg,
            "operation_max_deg": self.operation_max_deg,
            "operation_within_p95_limit_duty_min": (
                self.operation_within_p95_limit_duty_min
            ),
            "max_continuous_violation_s": self.max_continuous_violation_s,
            "constraint_polygon_sides": self.constraint_polygon_sides,
            "audit_timestep_s": self.audit_timestep_s,
        }

    @property
    def alignment_id(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def as_config(self) -> dict[str, Any]:
        return {"alignment_id": self.alignment_id, **self.identity_payload()}

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "JointPairAlignmentSettings":
        expected = {
            "schema_version",
            "alignment_id",
            "joint_names",
            "frame",
            "axis",
            "require_positive_y",
            "minimum_length_m",
            "residual",
            "grasp_p95_max_deg",
            "grasp_max_deg",
            "operation_p95_max_deg",
            "operation_max_deg",
            "operation_within_p95_limit_duty_min",
            "max_continuous_violation_s",
            "constraint_polygon_sides",
            "audit_timestep_s",
        }
        if set(value) != expected:
            raise ValueError(
                "joint_pair_alignment contains unexpected or missing fields"
            )
        settings = cls(
            schema_version=int(value["schema_version"]),
            joint_names=tuple(value["joint_names"]),
            frame=str(value["frame"]),
            axis=str(value["axis"]),
            require_positive_y=value["require_positive_y"],
            minimum_length_m=float(value["minimum_length_m"]),
            residual=str(value["residual"]),
            grasp_p95_max_deg=float(value["grasp_p95_max_deg"]),
            grasp_max_deg=float(value["grasp_max_deg"]),
            operation_p95_max_deg=float(value["operation_p95_max_deg"]),
            operation_max_deg=float(value["operation_max_deg"]),
            operation_within_p95_limit_duty_min=float(
                value["operation_within_p95_limit_duty_min"]
            ),
            max_continuous_violation_s=float(
                value["max_continuous_violation_s"]
            ),
            constraint_polygon_sides=int(value["constraint_polygon_sides"]),
            audit_timestep_s=float(value["audit_timestep_s"]),
        )
        if value["alignment_id"] != settings.alignment_id:
            raise ValueError(
                "joint_pair_alignment.alignment_id does not match its content"
            )
        return settings


@dataclass(frozen=True)
class JointPairFeedbackParameters:
    """Schema-v15 causal weighted-nullspace alignment feedback contract."""

    schema_version: int
    strategy: str
    alignment_gain: float
    slip_recovery_gain_rad_per_m: float
    correction_limit_rad: float
    rate_limit_rad_s: float
    acceleration_limit_rad_s2: float
    use_previous_observation: bool
    freeze_threshold_deg: float
    abort_threshold_deg: float
    slip_freeze_threshold_m: float
    slip_abort_threshold_m: float
    recovery_behavior: str
    damping: float
    vertical_response_weight: float
    force_response_weight: float
    # Schema 2 replaces the force-centroid displacement proxy with true
    # material-point relative tangent velocity.  Optional defaults are
    # intentionally omitted from schema-1 serialization so sealed v15 IDs do
    # not change.
    slip_resume_threshold_m: float | None = None
    slip_recovery_enter_threshold_m: float | None = None
    slip_recovery_exit_threshold_m: float | None = None
    slip_exit_dwell_s: float | None = None
    relative_velocity_filter_time_constant_s: float | None = None
    tangent_prediction_horizon_s: float | None = None
    suppress_outward_force_pi_during_recovery: bool | None = None
    maximum_patch_match_distance_m: float | None = None
    centroid_jump_diagnostic_threshold_m: float | None = None
    maximum_contact_time_gap_s: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in (1, 2):
            raise ValueError("joint_pair_feedback.schema_version must be 1 or 2")
        expected_strategy = (
            "previous_frame_weighted_nullspace"
            if self.schema_version == 1
            else "previous_frame_rolling_aware_signed_tangent_nullspace"
        )
        if self.strategy != expected_strategy:
            raise ValueError("unsupported joint_pair_feedback.strategy")
        if self.use_previous_observation is not True:
            raise ValueError(
                "joint_pair_feedback.use_previous_observation must be true"
            )
        expected_recovery = (
            "freeze_inward_recover_then_abort"
            if self.schema_version == 1
            else "signed_tangent_recover_with_hysteresis_then_abort"
        )
        if self.recovery_behavior != expected_recovery:
            raise ValueError("unsupported joint_pair_feedback.recovery_behavior")
        for label in (
            "alignment_gain",
            "slip_recovery_gain_rad_per_m",
            "correction_limit_rad",
            "rate_limit_rad_s",
            "acceleration_limit_rad_s2",
            "freeze_threshold_deg",
            "abort_threshold_deg",
            "slip_freeze_threshold_m",
            "slip_abort_threshold_m",
            "damping",
            "vertical_response_weight",
            "force_response_weight",
        ):
            value = _finite_number(
                getattr(self, label), f"joint_pair_feedback.{label}"
            )
            if value <= 0.0:
                raise ValueError(f"joint_pair_feedback.{label} must be positive")
            object.__setattr__(self, label, value)
        if self.freeze_threshold_deg > self.abort_threshold_deg:
            raise ValueError(
                "joint_pair_feedback freeze angle cannot exceed abort angle"
            )
        if self.slip_freeze_threshold_m > self.slip_abort_threshold_m:
            raise ValueError(
                "joint_pair_feedback slip freeze threshold cannot exceed abort threshold"
            )
        rolling_fields = (
            "slip_resume_threshold_m",
            "slip_recovery_enter_threshold_m",
            "slip_recovery_exit_threshold_m",
            "slip_exit_dwell_s",
            "relative_velocity_filter_time_constant_s",
            "tangent_prediction_horizon_s",
            "maximum_patch_match_distance_m",
            "centroid_jump_diagnostic_threshold_m",
            "maximum_contact_time_gap_s",
        )
        if self.schema_version == 1:
            if any(getattr(self, name) is not None for name in rolling_fields) or (
                self.suppress_outward_force_pi_during_recovery is not None
            ):
                raise ValueError(
                    "rolling-aware fields are only allowed in joint-pair feedback schema 2"
                )
        else:
            for name in rolling_fields:
                raw = getattr(self, name)
                if raw is None:
                    raise ValueError(f"joint_pair_feedback.{name} is required")
                value = _finite_number(raw, f"joint_pair_feedback.{name}")
                if value <= 0.0:
                    raise ValueError(f"joint_pair_feedback.{name} must be positive")
                object.__setattr__(self, name, value)
            if self.suppress_outward_force_pi_during_recovery is not True:
                raise ValueError(
                    "schema-2 feedback must suppress outward force PI during recovery"
                )
            assert self.slip_resume_threshold_m is not None
            assert self.slip_recovery_enter_threshold_m is not None
            assert self.slip_recovery_exit_threshold_m is not None
            if not (
                self.slip_recovery_exit_threshold_m
                < self.slip_recovery_enter_threshold_m
                < self.slip_resume_threshold_m
                < self.slip_freeze_threshold_m
                < self.slip_abort_threshold_m
            ):
                raise ValueError(
                    "rolling-aware slip thresholds must increase from recovery "
                    "exit through hard abort"
                )

    def identity_payload(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "alignment_gain": self.alignment_gain,
            "slip_recovery_gain_rad_per_m": (
                self.slip_recovery_gain_rad_per_m
            ),
            "correction_limit_rad": self.correction_limit_rad,
            "rate_limit_rad_s": self.rate_limit_rad_s,
            "acceleration_limit_rad_s2": self.acceleration_limit_rad_s2,
            "use_previous_observation": self.use_previous_observation,
            "freeze_threshold_deg": self.freeze_threshold_deg,
            "abort_threshold_deg": self.abort_threshold_deg,
            "slip_freeze_threshold_m": self.slip_freeze_threshold_m,
            "slip_abort_threshold_m": self.slip_abort_threshold_m,
            "recovery_behavior": self.recovery_behavior,
            "damping": self.damping,
            "vertical_response_weight": self.vertical_response_weight,
            "force_response_weight": self.force_response_weight,
        }
        if self.schema_version >= 2:
            result.update(
                {
                    "slip_resume_threshold_m": self.slip_resume_threshold_m,
                    "slip_recovery_enter_threshold_m": (
                        self.slip_recovery_enter_threshold_m
                    ),
                    "slip_recovery_exit_threshold_m": (
                        self.slip_recovery_exit_threshold_m
                    ),
                    "slip_exit_dwell_s": self.slip_exit_dwell_s,
                    "relative_velocity_filter_time_constant_s": (
                        self.relative_velocity_filter_time_constant_s
                    ),
                    "tangent_prediction_horizon_s": (
                        self.tangent_prediction_horizon_s
                    ),
                    "suppress_outward_force_pi_during_recovery": (
                        self.suppress_outward_force_pi_during_recovery
                    ),
                    "maximum_patch_match_distance_m": (
                        self.maximum_patch_match_distance_m
                    ),
                    "centroid_jump_diagnostic_threshold_m": (
                        self.centroid_jump_diagnostic_threshold_m
                    ),
                    "maximum_contact_time_gap_s": (
                        self.maximum_contact_time_gap_s
                    ),
                }
            )
        return result

    @property
    def feedback_id(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def as_config(self) -> dict[str, Any]:
        return {"feedback_id": self.feedback_id, **self.identity_payload()}

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "JointPairFeedbackParameters":
        schema_version = int(value.get("schema_version", -1))
        expected = {
            "schema_version",
            "feedback_id",
            "strategy",
            "alignment_gain",
            "slip_recovery_gain_rad_per_m",
            "correction_limit_rad",
            "rate_limit_rad_s",
            "acceleration_limit_rad_s2",
            "use_previous_observation",
            "freeze_threshold_deg",
            "abort_threshold_deg",
            "slip_freeze_threshold_m",
            "slip_abort_threshold_m",
            "recovery_behavior",
            "damping",
            "vertical_response_weight",
            "force_response_weight",
        }
        if schema_version >= 2:
            expected.update(
                {
                    "slip_resume_threshold_m",
                    "slip_recovery_enter_threshold_m",
                    "slip_recovery_exit_threshold_m",
                    "slip_exit_dwell_s",
                    "relative_velocity_filter_time_constant_s",
                    "tangent_prediction_horizon_s",
                    "suppress_outward_force_pi_during_recovery",
                    "maximum_patch_match_distance_m",
                    "centroid_jump_diagnostic_threshold_m",
                    "maximum_contact_time_gap_s",
                }
            )
        if set(value) != expected:
            raise ValueError(
                "joint_pair_feedback contains unexpected or missing fields"
            )
        feedback = cls(
            schema_version=schema_version,
            strategy=str(value["strategy"]),
            alignment_gain=float(value["alignment_gain"]),
            slip_recovery_gain_rad_per_m=float(
                value["slip_recovery_gain_rad_per_m"]
            ),
            correction_limit_rad=float(value["correction_limit_rad"]),
            rate_limit_rad_s=float(value["rate_limit_rad_s"]),
            acceleration_limit_rad_s2=float(
                value["acceleration_limit_rad_s2"]
            ),
            use_previous_observation=value["use_previous_observation"],
            freeze_threshold_deg=float(value["freeze_threshold_deg"]),
            abort_threshold_deg=float(value["abort_threshold_deg"]),
            slip_freeze_threshold_m=float(value["slip_freeze_threshold_m"]),
            slip_abort_threshold_m=float(value["slip_abort_threshold_m"]),
            recovery_behavior=str(value["recovery_behavior"]),
            damping=float(value["damping"]),
            vertical_response_weight=float(value["vertical_response_weight"]),
            force_response_weight=float(value["force_response_weight"]),
            slip_resume_threshold_m=(
                float(value["slip_resume_threshold_m"])
                if schema_version >= 2
                else None
            ),
            slip_recovery_enter_threshold_m=(
                float(value["slip_recovery_enter_threshold_m"])
                if schema_version >= 2
                else None
            ),
            slip_recovery_exit_threshold_m=(
                float(value["slip_recovery_exit_threshold_m"])
                if schema_version >= 2
                else None
            ),
            slip_exit_dwell_s=(
                float(value["slip_exit_dwell_s"])
                if schema_version >= 2
                else None
            ),
            relative_velocity_filter_time_constant_s=(
                float(value["relative_velocity_filter_time_constant_s"])
                if schema_version >= 2
                else None
            ),
            tangent_prediction_horizon_s=(
                float(value["tangent_prediction_horizon_s"])
                if schema_version >= 2
                else None
            ),
            suppress_outward_force_pi_during_recovery=(
                value["suppress_outward_force_pi_during_recovery"]
                if schema_version >= 2
                else None
            ),
            maximum_patch_match_distance_m=(
                float(value["maximum_patch_match_distance_m"])
                if schema_version >= 2
                else None
            ),
            centroid_jump_diagnostic_threshold_m=(
                float(value["centroid_jump_diagnostic_threshold_m"])
                if schema_version >= 2
                else None
            ),
            maximum_contact_time_gap_s=(
                float(value["maximum_contact_time_gap_s"])
                if schema_version >= 2
                else None
            ),
        )
        if value["feedback_id"] != feedback.feedback_id:
            raise ValueError(
                "joint_pair_feedback.feedback_id does not match its content"
            )
        return feedback


@dataclass(frozen=True)
class ContactPreservingPlannedLiftCampaignParameters:
    """Registered source-pair, planner and feedback search policy.

    Schema 1 is the sealed v14 multi-edge campaign.  Schema 2 is the fixed
    79-mm v15 near-zero joint-pair campaign and adds its authenticated source,
    bounded 14-variable search, closure grid and feedback-gain grid without
    changing schema-1 serialization.
    """

    schema_version: int
    edges_m: tuple[float, ...]
    fixed_mass_kg: float
    friction: float
    seed: int
    source_grasp_catalog: str
    source_grasp_catalog_sha256: str
    expected_source_grasp_count: int
    grasp_rescue_candidates_per_edge_mapping: int
    priority_grasp_rescue_candidates_per_seed: int
    grasp_rescue_retain_per_edge_mapping: int
    pair_probe_count: int
    pair_shortlist_count: int
    plan_knot_count: int
    plan_duration_s: float
    plan_candidates_per_pair: int
    feedback_plan_candidates_per_pair: int
    feedback_candidates_per_plan: int
    feedback_refine_plan_count: int
    feedback_refine_per_plan: int
    final_candidate_count: int
    perturbations_per_final: int
    robustness_trials: int
    robustness_required_passes: int
    force_target_minimum_n: float
    force_target_maximum_n: float
    operation_contact_duty_min: float
    max_contact_loss_s: float
    validation_labels: Mapping[str, str]
    allow_single_edge: bool = False
    source_candidate_id: int | None = None
    source_alias: str | None = None
    source_config_path: str | None = None
    source_config_sha256: str | None = None
    source_result_path: str | None = None
    source_result_sha256: str | None = None
    source_trace_path: str | None = None
    source_trace_sha256: str | None = None
    dls_start_count: int | None = None
    static_retain_count: int | None = None
    grasp_retain_count: int | None = None
    revalidate_candidate_count: int | None = None
    root_translation_delta_m: tuple[float, float] | None = None
    root_rotation_axis_delta_deg: tuple[float, float] | None = None
    root_rotation_norm_max_deg: float | None = None
    grasp_qpos_delta_rad: tuple[float, float] | None = None
    close_duration_options_s: tuple[float, ...] | None = None
    close_modes: tuple[str, ...] | None = None
    alignment_gain_options: tuple[float, ...] | None = None
    slip_recovery_gain_options_rad_per_m: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in (1, 2):
            raise ValueError(
                "contact_preserving_planned_lift_campaign.schema_version must be 1 or 2"
            )
        edges = tuple(
            _finite_number(value, "contact preserving edges_m")
            for value in self.edges_m
        )
        if (
            (len(edges) < 2 and not self.allow_single_edge)
            or tuple(sorted(set(edges))) != edges
            or any(value <= 0.0 for value in edges)
        ):
            raise ValueError("contact preserving edges_m must be positive and increasing")
        object.__setattr__(self, "edges_m", edges)
        if not isinstance(self.allow_single_edge, bool):
            raise ValueError("allow_single_edge must be boolean")
        if self.schema_version == 1 and self.allow_single_edge:
            raise ValueError("campaign schema 1 cannot enable a single edge")
        for label in (
            "fixed_mass_kg",
            "friction",
            "plan_duration_s",
            "force_target_minimum_n",
            "force_target_maximum_n",
            "max_contact_loss_s",
        ):
            value = _finite_number(getattr(self, label), label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
            object.__setattr__(self, label, value)
        if self.force_target_maximum_n < self.force_target_minimum_n:
            raise ValueError("force target range is reversed")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        source_path = PurePosixPath(str(self.source_grasp_catalog))
        if (
            source_path.is_absolute()
            or ".." in source_path.parts
            or source_path.parts[:1] != ("artifacts",)
        ):
            raise ValueError("source_grasp_catalog must be below artifacts/")
        digest = str(self.source_grasp_catalog_sha256)
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("source_grasp_catalog_sha256 must be lower-case SHA-256")
        object.__setattr__(self, "source_grasp_catalog_sha256", digest)
        integer_fields = (
            "expected_source_grasp_count",
            "grasp_rescue_candidates_per_edge_mapping",
            "priority_grasp_rescue_candidates_per_seed",
            "grasp_rescue_retain_per_edge_mapping",
            "pair_probe_count",
            "pair_shortlist_count",
            "plan_knot_count",
            "plan_candidates_per_pair",
            "feedback_plan_candidates_per_pair",
            "feedback_candidates_per_plan",
            "feedback_refine_plan_count",
            "feedback_refine_per_plan",
            "final_candidate_count",
            "perturbations_per_final",
            "robustness_trials",
            "robustness_required_passes",
        )
        for label in integer_fields:
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.plan_knot_count < 2:
            raise ValueError("plan_knot_count must be at least two")
        if (
            self.schema_version == 1
            and self.pair_shortlist_count > self.expected_source_grasp_count
        ):
            raise ValueError("pair shortlist exceeds authenticated source grasps")
        if self.feedback_refine_plan_count > (
            self.pair_shortlist_count * self.feedback_plan_candidates_per_pair
        ):
            raise ValueError("feedback refine count exceeds planned candidates")
        if self.feedback_plan_candidates_per_pair > self.plan_candidates_per_pair:
            raise ValueError("feedback plan quota exceeds generated plans per pair")
        if self.final_candidate_count > self.feedback_refine_plan_count:
            raise ValueError("final candidate count exceeds feedback refinement set")
        if self.robustness_required_passes > self.robustness_trials:
            raise ValueError("robustness pass count exceeds trial count")
        duty = _finite_number(
            self.operation_contact_duty_min, "operation_contact_duty_min"
        )
        if not 0.0 < duty <= 1.0:
            raise ValueError("operation_contact_duty_min must be within (0, 1]")
        object.__setattr__(self, "operation_contact_duty_min", duty)
        labels = {str(key): str(value) for key, value in self.validation_labels.items()}
        if set(labels) != {"grasp", "manipulation", "robust"} or any(
            not value for value in labels.values()
        ):
            raise ValueError("validation_labels must define grasp/manipulation/robust")
        object.__setattr__(self, "validation_labels", MappingProxyType(labels))

        v15_fields = (
            self.source_candidate_id,
            self.source_alias,
            self.source_config_path,
            self.source_config_sha256,
            self.source_result_path,
            self.source_result_sha256,
            self.source_trace_path,
            self.source_trace_sha256,
            self.dls_start_count,
            self.static_retain_count,
            self.grasp_retain_count,
            self.revalidate_candidate_count,
            self.root_translation_delta_m,
            self.root_rotation_axis_delta_deg,
            self.root_rotation_norm_max_deg,
            self.grasp_qpos_delta_rad,
            self.close_duration_options_s,
            self.close_modes,
            self.alignment_gain_options,
            self.slip_recovery_gain_options_rad_per_m,
        )
        if self.schema_version == 1:
            if any(value is not None for value in v15_fields):
                raise ValueError(
                    "campaign schema 1 cannot declare schema-v15 search fields"
                )
            return
        if not self.allow_single_edge or len(edges) != 1:
            raise ValueError("campaign schema 2 requires exactly one fixed edge")
        if (
            self.source_candidate_id is None
            or not isinstance(self.source_candidate_id, int)
            or isinstance(self.source_candidate_id, bool)
            or self.source_candidate_id <= 0
        ):
            raise ValueError("schema-v15 source_candidate_id must be positive")
        if self.source_alias != "best_near_zero_grasp":
            raise ValueError(
                "schema-v15 source_alias must be best_near_zero_grasp"
            )
        for path_field, hash_field in (
            ("source_config_path", "source_config_sha256"),
            ("source_result_path", "source_result_sha256"),
            ("source_trace_path", "source_trace_sha256"),
        ):
            path_value = getattr(self, path_field)
            digest_value = getattr(self, hash_field)
            if path_value is None:
                raise ValueError(f"schema-v15 {path_field} is required")
            path = PurePosixPath(str(path_value))
            if path.is_absolute() or ".." in path.parts or path.parts[:1] != (
                "artifacts",
            ):
                raise ValueError(f"schema-v15 {path_field} must be below artifacts/")
            if re.fullmatch(r"[0-9a-f]{64}", str(digest_value)) is None:
                raise ValueError(f"schema-v15 {hash_field} must be SHA-256")
            object.__setattr__(self, path_field, str(path))
            object.__setattr__(self, hash_field, str(digest_value))
        for label in (
            "dls_start_count",
            "static_retain_count",
            "grasp_retain_count",
            "revalidate_candidate_count",
        ):
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"schema-v15 {label} must be positive")
        assert self.dls_start_count is not None
        assert self.static_retain_count is not None
        assert self.grasp_retain_count is not None
        assert self.revalidate_candidate_count is not None
        if not (
            self.grasp_retain_count
            <= self.static_retain_count
            <= self.dls_start_count
        ):
            raise ValueError("schema-v15 DLS retention counts are inconsistent")
        if self.pair_shortlist_count != self.grasp_retain_count:
            raise ValueError(
                "schema-v15 pair shortlist must equal retained measured grasps"
            )
        if self.revalidate_candidate_count < self.final_candidate_count:
            raise ValueError(
                "schema-v15 revalidation count cannot be below final count"
            )
        for label in (
            "root_translation_delta_m",
            "root_rotation_axis_delta_deg",
            "grasp_qpos_delta_rad",
        ):
            raw = getattr(self, label)
            if raw is None:
                raise ValueError(f"schema-v15 {label} is required")
            bounds = _closed_range(raw, f"schema-v15 {label}")
            if not bounds[0] < 0.0 < bounds[1]:
                raise ValueError(f"schema-v15 {label} must straddle zero")
            object.__setattr__(self, label, bounds)
        rotation_norm = _finite_number(
            self.root_rotation_norm_max_deg,
            "schema-v15 root_rotation_norm_max_deg",
        )
        if rotation_norm <= 0.0:
            raise ValueError(
                "schema-v15 root_rotation_norm_max_deg must be positive"
            )
        object.__setattr__(self, "root_rotation_norm_max_deg", rotation_norm)
        for label in (
            "close_duration_options_s",
            "alignment_gain_options",
            "slip_recovery_gain_options_rad_per_m",
        ):
            raw = getattr(self, label)
            if raw is None:
                raise ValueError(f"schema-v15 {label} is required")
            values = tuple(
                _finite_number(value, f"schema-v15 {label}") for value in raw
            )
            if (
                not values
                or any(value <= 0.0 for value in values)
                or tuple(sorted(set(values))) != values
            ):
                raise ValueError(
                    f"schema-v15 {label} must be positive, unique and increasing"
                )
            object.__setattr__(self, label, values)
        modes = tuple(str(value) for value in (self.close_modes or ()))
        if modes != ("original", "synchronized_preload"):
            raise ValueError(
                "schema-v15 close_modes must be original and "
                "synchronized_preload"
            )
        object.__setattr__(self, "close_modes", modes)

    @property
    def maximum_plan_candidate_count(self) -> int:
        return self.pair_shortlist_count * self.plan_candidates_per_pair

    def budget_config(self) -> dict[str, int]:
        if self.schema_version >= 2:
            assert self.dls_start_count is not None
            assert self.static_retain_count is not None
            assert self.grasp_retain_count is not None
            assert self.revalidate_candidate_count is not None
            assert self.close_duration_options_s is not None
            assert self.close_modes is not None
            assert self.alignment_gain_options is not None
            assert self.slip_recovery_gain_options_rad_per_m is not None
            close_control_count = (
                len(self.close_duration_options_s) * len(self.close_modes)
            )
            plan_candidate_count = (
                self.grasp_retain_count * self.plan_candidates_per_pair
            )
            feedback_grid_count = (
                len(self.alignment_gain_options)
                * len(self.slip_recovery_gain_options_rad_per_m)
            )
            return {
                "seed": self.seed,
                "static_start_count": self.dls_start_count,
                "static_retain_count": self.static_retain_count,
                "close_controls_per_pose": close_control_count,
                "measured_grasp_retain_count": self.grasp_retain_count,
                "sequential_plans_per_grasp": self.plan_candidates_per_pair,
                "feedback_candidates_per_plan": feedback_grid_count,
                "feedback_refine_plan_count": self.feedback_refine_plan_count,
                "feedback_refine_per_plan": self.feedback_refine_per_plan,
                "exact_rerun_count": self.revalidate_candidate_count,
                "final_candidate_count": self.final_candidate_count,
                "perturbations_per_final": self.perturbations_per_final,
                "robustness_trials": self.robustness_trials,
                "robustness_required_passes": self.robustness_required_passes,
                "dynamic_grasp_count": (
                    self.static_retain_count * close_control_count
                ),
                "sequential_plan_count": plan_candidate_count,
                "feedback_grid_count": (
                    plan_candidate_count * feedback_grid_count
                ),
                "feedback_refine_count": (
                    self.feedback_refine_plan_count
                    * self.feedback_refine_per_plan
                ),
            }
        return {
            "expected_source_grasp_count": self.expected_source_grasp_count,
            "grasp_rescue_candidates_per_edge_mapping": (
                self.grasp_rescue_candidates_per_edge_mapping
            ),
            "priority_grasp_rescue_candidates_per_seed": (
                self.priority_grasp_rescue_candidates_per_seed
            ),
            "grasp_rescue_retain_per_edge_mapping": (
                self.grasp_rescue_retain_per_edge_mapping
            ),
            "pair_probe_count": self.pair_probe_count,
            "pair_shortlist_count": self.pair_shortlist_count,
            "plan_knot_count": self.plan_knot_count,
            "plan_candidates_per_pair": self.plan_candidates_per_pair,
            "feedback_plan_candidates_per_pair": (
                self.feedback_plan_candidates_per_pair
            ),
            "maximum_plan_candidate_count": self.maximum_plan_candidate_count,
            "feedback_candidates_per_plan": self.feedback_candidates_per_plan,
            "feedback_refine_plan_count": self.feedback_refine_plan_count,
            "feedback_refine_per_plan": self.feedback_refine_per_plan,
            "final_candidate_count": self.final_candidate_count,
            "perturbations_per_final": self.perturbations_per_final,
            "robustness_trials": self.robustness_trials,
            "robustness_required_passes": self.robustness_required_passes,
        }

    def as_config(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "edges_m": list(self.edges_m),
            "fixed_mass_kg": self.fixed_mass_kg,
            "friction": self.friction,
            "seed": self.seed,
            "source_grasp_catalog": {
                "path": self.source_grasp_catalog,
                "sha256": self.source_grasp_catalog_sha256,
                "expected_success_count": self.expected_source_grasp_count,
                "success_evidence_only": True,
            },
            "identity": {
                "manipulation_plan": "plan_id_sha256",
                "contact_force_targets_n": "target_id_sha256",
                "contact_feedback": "feedback_id_sha256",
                "final_controller": "plan_id_target_id_feedback_id_sha256",
            },
            "plan_contract": {
                "profile": "piecewise_quintic_minimum_jerk",
                "duration_s": self.plan_duration_s,
                "knot_count": self.plan_knot_count,
                "relative_to": "contact_preload_targets_rad",
            },
            "force_target_contract": {
                "source": "verify_window_median_clamped",
                "minimum_n": self.force_target_minimum_n,
                "maximum_n": self.force_target_maximum_n,
            },
            "contact_acceptance": {
                "operation_contact_duty_min": self.operation_contact_duty_min,
                "max_contact_loss_s": self.max_contact_loss_s,
                "applies_to_each_finger_and_simultaneous_topology": True,
            },
            "budget": self.budget_config(),
            "validation_labels": dict(self.validation_labels),
            "final_evidence": "fresh_full_reset_free_dynamics_only",
        }
        if self.schema_version >= 2:
            assert self.source_candidate_id is not None
            assert self.source_alias is not None
            assert self.source_config_path is not None
            assert self.source_config_sha256 is not None
            assert self.source_result_path is not None
            assert self.source_result_sha256 is not None
            assert self.source_trace_path is not None
            assert self.source_trace_sha256 is not None
            assert self.root_translation_delta_m is not None
            assert self.root_rotation_axis_delta_deg is not None
            assert self.root_rotation_norm_max_deg is not None
            assert self.grasp_qpos_delta_rad is not None
            assert self.close_duration_options_s is not None
            assert self.close_modes is not None
            assert self.alignment_gain_options is not None
            assert self.slip_recovery_gain_options_rad_per_m is not None
            result["source_seed"] = {
                "candidate_id": self.source_candidate_id,
                "alias": self.source_alias,
                "resolved_config": {
                    "path": self.source_config_path,
                    "sha256": self.source_config_sha256,
                },
                "result": {
                    "path": self.source_result_path,
                    "sha256": self.source_result_sha256,
                },
                "trace": {
                    "path": self.source_trace_path,
                    "sha256": self.source_trace_sha256,
                },
            }
            result["search_contract"] = {
                "root_translation_delta_m": list(
                    self.root_translation_delta_m
                ),
                "root_rotation_axis_delta_deg": list(
                    self.root_rotation_axis_delta_deg
                ),
                "root_rotation_norm_max_deg": self.root_rotation_norm_max_deg,
                "grasp_qpos_delta_rad": list(self.grasp_qpos_delta_rad),
                "close_duration_options_s": list(
                    self.close_duration_options_s
                ),
                "close_modes": list(self.close_modes),
                "alignment_gain_options": list(self.alignment_gain_options),
                "slip_recovery_gain_options_rad_per_m": list(
                    self.slip_recovery_gain_options_rad_per_m
                ),
            }
            result["identity"] = {
                "object": "object_config_id_sha256",
                "grasp": "grasp_pose_id_sha256",
                "pair": "grasp_object_pair_id_sha256",
                "planner": "pair_plan_alignment_sha256",
                "controller": "planner_force_and_feedback_sha256",
            }
            result["source_grasp_catalog"]["success_evidence_only"] = False
            result["source_grasp_catalog"][
                "required_classification"
            ] = "grasp_success_manipulation_near_miss"
            result["single_edge_campaign"] = True
        return result

    def validate_candidate(
        self,
        plan: ManipulationPlanParameters,
        force_targets: ContactForceTargets,
        feedback: ContactFeedbackParameters,
    ) -> None:
        expected_plan_schema = self.schema_version
        if plan.schema_version != expected_plan_schema:
            raise ValueError(
                "manipulation_plan schema must match the planned-lift campaign"
            )
        if len(plan.knot_times_s) != self.plan_knot_count:
            raise ValueError("manipulation_plan knot count must match the campaign")
        if not math.isclose(plan.duration_s, self.plan_duration_s, abs_tol=1e-12):
            raise ValueError("manipulation_plan duration must match the campaign")
        if not math.isclose(
            force_targets.minimum_n, self.force_target_minimum_n, abs_tol=1e-12
        ) or not math.isclose(
            force_targets.maximum_n, self.force_target_maximum_n, abs_tol=1e-12
        ):
            raise ValueError("contact force clamp must match the campaign")
        if feedback.force_risk_n > force_targets.minimum_n + 1e-12:
            raise ValueError("contact feedback force risk exceeds target minimum")
        if not math.isclose(
            feedback.operation_contact_duty_min,
            self.operation_contact_duty_min,
            abs_tol=1e-12,
        ):
            raise ValueError("contact feedback duty must match the campaign")
        if not math.isclose(
            feedback.max_loss_s, self.max_contact_loss_s, abs_tol=1e-12
        ):
            raise ValueError("contact feedback loss window must match the campaign")


@dataclass(frozen=True)
class RobustnessParameters:
    edge_m: tuple[float, ...]
    mass_kg: tuple[float, ...]
    friction: tuple[float, ...]
    perturbation_count: int
    required_pass_count: int
    seed: int
    position_xy_delta_m: tuple[float, float]
    rpy_delta_deg: tuple[float, float]
    mass_scale: tuple[float, float]
    friction_delta: tuple[float, float]
    z_offset_delta_m: tuple[float, float]
    case_families: RobustnessCaseFamilies | None = None

    def __post_init__(self) -> None:
        for label in ("edge_m", "mass_kg", "friction"):
            values = tuple(_finite_number(value, label) for value in getattr(self, label))
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"{label} must contain positive finite values")
            object.__setattr__(self, label, values)
        for label in (
            "position_xy_delta_m",
            "rpy_delta_deg",
            "mass_scale",
            "friction_delta",
            "z_offset_delta_m",
        ):
            object.__setattr__(self, label, _closed_range(getattr(self, label), label))
        if self.perturbation_count <= 0:
            raise ValueError("perturbation_count must be positive")
        if not 0 <= self.required_pass_count <= self.perturbation_count:
            raise ValueError("required_pass_count must be within the perturbation count")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.case_families is not None and not isinstance(
            self.case_families, RobustnessCaseFamilies
        ):
            raise TypeError("case_families must be RobustnessCaseFamilies or None")

    @property
    def grid_case_count(self) -> int:
        if self.case_families is not None:
            return self.case_families.total_case_count
        return len(self.edge_m) * len(self.mass_kg) * len(self.friction)

    def as_config(self) -> dict[str, Any]:
        result = {
            "edge_m": list(self.edge_m),
            "mass_kg": list(self.mass_kg),
            "friction": list(self.friction),
            "perturbation_count": self.perturbation_count,
            "required_pass_count": self.required_pass_count,
            "seed": self.seed,
            "position_xy_delta_m": list(self.position_xy_delta_m),
            "rpy_delta_deg": list(self.rpy_delta_deg),
            "mass_scale": list(self.mass_scale),
            "friction_delta": list(self.friction_delta),
            "z_offset_delta_m": list(self.z_offset_delta_m),
        }
        if self.case_families is not None:
            # Family-driven grids are nominal-centred, so the legacy Cartesian
            # edge/mass lists are implementation fallbacks rather than part of
            # their serialized contract.
            result.pop("edge_m")
            result.pop("mass_kg")
            result["case_families"] = self.case_families.as_config()
        return result


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    description: str
    evaluation: EvaluationSettings
    candidate_faces: tuple[OpposedFaceAssignment, ...]
    search_bounds: SearchBounds
    robustness: RobustnessParameters
    artifact_root: str | None = None
    size_campaign: SizeCampaignParameters | None = None
    control_protocol: ControlProtocolSettings | None = None
    tuning_strategy: str = "default"
    pose_constraints: PoseConstraints | None = None
    contact_alignment: ContactAlignmentSettings | None = None
    aligned_contact_campaign: AlignedContactCampaignParameters | None = None
    far_hand_pose_constraints: FarHandPoseConstraints | None = None
    fingertip_contact_preferences: FingertipContactPreferences | None = None
    far_hand_campaign: FarHandFingertipCampaignParameters | None = None
    pose_preservation: PosePreservationSettings | None = None
    high_thumb_size_campaign: HighThumbSizeCampaignParameters | None = None
    closure_alignment: ClosureAlignmentSettings | None = None
    motion_smoothness: MotionSmoothnessSettings | None = None
    normal_aligned_smooth_lift_campaign: (
        NormalAlignedSmoothLiftCampaignParameters | None
    ) = None
    actual_contact_grasp_pose: ActualContactGraspPoseSettings | None = None
    actual_contact_grasp_pose_campaign: (
        ActualContactGraspPoseCampaignParameters | None
    ) = None
    relative_wrist_pose_search: RelativeWristPoseSearchParameters | None = None
    contact_point_search: ContactPointSearchParameters | None = None
    scaled_contact_downsize_campaign: (
        ScaledContactDownsizeCampaignParameters | None
    ) = None
    contact_preserving_planned_lift_campaign: (
        ContactPreservingPlannedLiftCampaignParameters | None
    ) = None
    joint_pair_alignment: JointPairAlignmentSettings | None = None
    joint_pair_feedback: JointPairFeedbackParameters | None = None

    def __post_init__(self) -> None:
        if not _EXPERIMENT_ID_PATTERN.fullmatch(self.experiment_id):
            raise ValueError("experiment_id must be lower-case snake_case")
        if not self.description.strip():
            raise ValueError("description must not be empty")
        if self.tuning_strategy not in TUNING_STRATEGIES:
            raise ValueError(
                "tuning_strategy must be one of " + ", ".join(TUNING_STRATEGIES)
            )
        candidates = tuple(self.candidate_faces)
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidate_faces must not contain duplicates")
        object.__setattr__(self, "candidate_faces", candidates)
        if self.evaluation.target_faces is not None and self.evaluation.target_faces not in candidates:
            raise ValueError("evaluation target_faces must be present in candidate_faces")
        if self.artifact_root is not None:
            artifact_root = str(self.artifact_root)
            path = PurePosixPath(artifact_root)
            if (
                not artifact_root
                or path.is_absolute()
                or ".." in path.parts
                or path.parts[:1] != ("artifacts",)
            ):
                raise ValueError(
                    "artifact_root must be a repository-relative path below artifacts/"
                )
            object.__setattr__(self, "artifact_root", artifact_root.rstrip("/"))
        if self.size_campaign is not None and not isinstance(
            self.size_campaign, SizeCampaignParameters
        ):
            raise TypeError("size_campaign must be SizeCampaignParameters or None")
        if self.control_protocol is not None and not isinstance(
            self.control_protocol, ControlProtocolSettings
        ):
            raise TypeError("control_protocol must be ControlProtocolSettings or None")
        if self.pose_constraints is not None and not isinstance(
            self.pose_constraints, PoseConstraints
        ):
            raise TypeError("pose_constraints must be PoseConstraints or None")
        if self.contact_alignment is not None and not isinstance(
            self.contact_alignment, ContactAlignmentSettings
        ):
            raise TypeError(
                "contact_alignment must be ContactAlignmentSettings or None"
            )
        if self.aligned_contact_campaign is not None and not isinstance(
            self.aligned_contact_campaign, AlignedContactCampaignParameters
        ):
            raise TypeError(
                "aligned_contact_campaign must be "
                "AlignedContactCampaignParameters or None"
            )
        if self.far_hand_pose_constraints is not None and not isinstance(
            self.far_hand_pose_constraints, FarHandPoseConstraints
        ):
            raise TypeError(
                "far_hand_pose_constraints must be FarHandPoseConstraints or None"
            )
        if self.fingertip_contact_preferences is not None and not isinstance(
            self.fingertip_contact_preferences, FingertipContactPreferences
        ):
            raise TypeError(
                "fingertip_contact_preferences must be "
                "FingertipContactPreferences or None"
            )
        if self.far_hand_campaign is not None and not isinstance(
            self.far_hand_campaign, FarHandFingertipCampaignParameters
        ):
            raise TypeError(
                "far_hand_campaign must be "
                "FarHandFingertipCampaignParameters or None"
            )
        if self.pose_preservation is not None and not isinstance(
            self.pose_preservation, PosePreservationSettings
        ):
            raise TypeError(
                "pose_preservation must be PosePreservationSettings or None"
            )
        if self.high_thumb_size_campaign is not None and not isinstance(
            self.high_thumb_size_campaign, HighThumbSizeCampaignParameters
        ):
            raise TypeError(
                "high_thumb_size_campaign must be "
                "HighThumbSizeCampaignParameters or None"
            )
        if self.closure_alignment is not None and not isinstance(
            self.closure_alignment, ClosureAlignmentSettings
        ):
            raise TypeError(
                "closure_alignment must be ClosureAlignmentSettings or None"
            )
        if self.motion_smoothness is not None and not isinstance(
            self.motion_smoothness, MotionSmoothnessSettings
        ):
            raise TypeError(
                "motion_smoothness must be MotionSmoothnessSettings or None"
            )
        if (
            self.normal_aligned_smooth_lift_campaign is not None
            and not isinstance(
                self.normal_aligned_smooth_lift_campaign,
                NormalAlignedSmoothLiftCampaignParameters,
            )
        ):
            raise TypeError(
                "normal_aligned_smooth_lift_campaign must be "
                "NormalAlignedSmoothLiftCampaignParameters or None"
            )
        if self.actual_contact_grasp_pose is not None and not isinstance(
            self.actual_contact_grasp_pose, ActualContactGraspPoseSettings
        ):
            raise TypeError(
                "actual_contact_grasp_pose must be "
                "ActualContactGraspPoseSettings or None"
            )
        if (
            self.actual_contact_grasp_pose_campaign is not None
            and not isinstance(
                self.actual_contact_grasp_pose_campaign,
                ActualContactGraspPoseCampaignParameters,
            )
        ):
            raise TypeError(
                "actual_contact_grasp_pose_campaign must be "
                "ActualContactGraspPoseCampaignParameters or None"
            )
        if self.relative_wrist_pose_search is not None and not isinstance(
            self.relative_wrist_pose_search, RelativeWristPoseSearchParameters
        ):
            raise TypeError(
                "relative_wrist_pose_search must be "
                "RelativeWristPoseSearchParameters or None"
            )
        if (
            self.relative_wrist_pose_search is not None
            and self.actual_contact_grasp_pose_campaign is None
        ):
            raise ValueError(
                "relative_wrist_pose_search requires an actual-contact campaign"
            )
        if self.contact_point_search is not None and not isinstance(
            self.contact_point_search, ContactPointSearchParameters
        ):
            raise TypeError(
                "contact_point_search must be ContactPointSearchParameters or None"
            )
        if (
            self.contact_point_search is not None
            and self.actual_contact_grasp_pose_campaign is None
        ):
            raise ValueError(
                "contact_point_search requires an actual-contact campaign"
            )
        if (
            self.contact_point_search is not None
            and self.relative_wrist_pose_search is not None
        ):
            raise ValueError(
                "contact_point_search and relative_wrist_pose_search are mutually exclusive"
            )
        if (
            self.scaled_contact_downsize_campaign is not None
            and not isinstance(
                self.scaled_contact_downsize_campaign,
                ScaledContactDownsizeCampaignParameters,
            )
        ):
            raise TypeError(
                "scaled_contact_downsize_campaign must be "
                "ScaledContactDownsizeCampaignParameters or None"
            )
        if (
            self.scaled_contact_downsize_campaign is not None
            and self.actual_contact_grasp_pose_campaign is None
        ):
            raise ValueError(
                "scaled_contact_downsize_campaign requires an actual-contact campaign"
            )
        if self.scaled_contact_downsize_campaign is not None and (
            self.relative_wrist_pose_search is not None
            or self.contact_point_search is not None
        ):
            raise ValueError(
                "scaled_contact_downsize_campaign is mutually exclusive with "
                "relative_wrist_pose_search and contact_point_search"
            )
        if (
            self.contact_preserving_planned_lift_campaign is not None
            and not isinstance(
                self.contact_preserving_planned_lift_campaign,
                ContactPreservingPlannedLiftCampaignParameters,
            )
        ):
            raise TypeError(
                "contact_preserving_planned_lift_campaign must be "
                "ContactPreservingPlannedLiftCampaignParameters or None"
            )
        if self.contact_preserving_planned_lift_campaign is not None:
            if self.actual_contact_grasp_pose_campaign is None:
                raise ValueError(
                    "contact-preserving planned lift requires an actual-contact campaign"
                )
            if any(
                value is not None
                for value in (
                    self.scaled_contact_downsize_campaign,
                    self.contact_point_search,
                    self.relative_wrist_pose_search,
                )
            ):
                raise ValueError(
                    "contact-preserving planned lift is mutually exclusive with "
                    "legacy auxiliary search campaigns"
                )
        joint_pair_selected = (
            self.joint_pair_alignment is not None
            or self.joint_pair_feedback is not None
        )
        if joint_pair_selected:
            if not isinstance(
                self.joint_pair_alignment, JointPairAlignmentSettings
            ):
                raise TypeError(
                    "joint_pair_alignment must be JointPairAlignmentSettings"
                )
            if not isinstance(
                self.joint_pair_feedback, JointPairFeedbackParameters
            ):
                raise TypeError(
                    "joint_pair_feedback must be JointPairFeedbackParameters"
                )
            planned_lift = self.contact_preserving_planned_lift_campaign
            if planned_lift is None or planned_lift.schema_version != 2:
                raise ValueError(
                    "joint-pair alignment requires a schema-2 planned-lift campaign"
                )
            feedback = self.joint_pair_feedback
            assert feedback is not None
            rolling_aware = feedback.schema_version >= 2
            expected_tuning_strategy = (
                "joint_pair_near_zero_rolling_slip_contact_preserving_planned_lift"
                if rolling_aware
                else "joint_pair_near_zero_contact_preserving_planned_lift"
            )
            if self.tuning_strategy != expected_tuning_strategy:
                raise ValueError(
                    "joint-pair alignment requires its dedicated tuning strategy"
                )
            protocol = self.control_protocol
            expected_control_strategy = (
                "grasp_verify_then_joint_pair_aligned_rolling_slip_"
                "contact_preserving_planned_lift"
                if rolling_aware
                else "grasp_verify_then_joint_pair_aligned_"
                "contact_preserving_planned_lift"
            )
            if protocol is None or protocol.strategy != expected_control_strategy:
                raise ValueError(
                    "joint-pair alignment requires its dedicated control strategy"
                )
            alignment = self.joint_pair_alignment
            assert alignment is not None and feedback is not None
            if not math.isclose(
                feedback.freeze_threshold_deg,
                alignment.operation_p95_max_deg,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "joint-pair freeze angle must match operation p95 threshold"
                )
            if not math.isclose(
                feedback.abort_threshold_deg,
                alignment.operation_max_deg,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "joint-pair abort angle must match operation maximum"
                )
            if not math.isclose(
                feedback.slip_freeze_threshold_m, 0.0015, abs_tol=1e-12
            ) or not math.isclose(
                feedback.slip_abort_threshold_m, 0.002, abs_tol=1e-12
            ):
                raise ValueError(
                    "joint-pair slip thresholds must be 1.5 mm and 2 mm"
                )
        elif self.tuning_strategy in (
            "joint_pair_near_zero_contact_preserving_planned_lift",
            "joint_pair_near_zero_rolling_slip_contact_preserving_planned_lift",
        ):
            raise ValueError(
                "joint-pair tuning strategy requires alignment and feedback"
            )
        v4_selected = (
            self.pose_constraints is not None
            or self.aligned_contact_campaign is not None
        )
        if v4_selected and not all(
            value is not None
            for value in (
                self.pose_constraints,
                self.contact_alignment,
                self.aligned_contact_campaign,
            )
        ):
            raise ValueError(
                "schema-v4 definitions require pose_constraints, "
                "contact_alignment and aligned_contact_campaign together"
            )
        far_hand_shared_fields = (
            self.far_hand_pose_constraints,
            self.fingertip_contact_preferences,
        )
        far_hand_campaign_fields = (
            self.far_hand_campaign,
            self.high_thumb_size_campaign,
            self.normal_aligned_smooth_lift_campaign,
            self.actual_contact_grasp_pose_campaign,
        )
        far_hand_selected = any(
            value is not None
            for value in far_hand_shared_fields + far_hand_campaign_fields
        )
        if far_hand_selected and not all(
            value is not None for value in far_hand_shared_fields
        ):
            raise ValueError(
                "far-hand definitions require far_hand_pose_constraints and "
                "fingertip_contact_preferences together"
            )
        if far_hand_selected and sum(
            value is not None for value in far_hand_campaign_fields
        ) != 1:
            raise ValueError(
                "far-hand definitions require exactly one registered campaign"
            )
        if self.pose_preservation is not None:
            if not far_hand_selected:
                raise ValueError(
                    "pose preservation requires the complete far-hand "
                    "fingertip experiment definition"
                )
            if self.actual_contact_grasp_pose_campaign is not None:
                expected_pose_strategy = (
                    (
                        (
                            "joint_pair_near_zero_rolling_slip_contact_preserving_planned_lift"
                            if self.joint_pair_feedback is not None
                            and self.joint_pair_feedback.schema_version >= 2
                            else "joint_pair_near_zero_contact_preserving_planned_lift"
                        )
                        if joint_pair_selected
                        else "contact_preserving_planned_lift"
                    )
                    if self.contact_preserving_planned_lift_campaign is not None
                    else (
                        "scaled_contact_downsize_actual_grasp_then_lift"
                        if self.scaled_contact_downsize_campaign is not None
                        else (
                            "contact_point_targeted_actual_grasp_pose"
                            if self.contact_point_search is not None
                            else "actual_contact_grasp_pose_smooth_vertical_lift"
                        )
                    )
                )
            elif self.normal_aligned_smooth_lift_campaign is not None:
                expected_pose_strategy = "normal_aligned_smooth_vertical_lift"
            elif self.high_thumb_size_campaign is not None:
                expected_pose_strategy = "high_thumb_variable_size_pose_preserving"
            else:
                expected_pose_strategy = "pose_preserving_grasp"
            if self.tuning_strategy != expected_pose_strategy:
                raise ValueError(
                    "pose-preserving definition has the wrong tuning_strategy"
                )
            if self.search_bounds.pregrasp_targets_rad is None:
                raise ValueError(
                    "pose-preserving definitions require pregrasp target search bounds"
                )
        elif self.search_bounds.pregrasp_targets_rad is not None:
            raise ValueError(
                "pregrasp target search bounds require pose_preservation"
            )
        if (
            self.contact_alignment is not None
            and self.aligned_contact_campaign is None
            and self.far_hand_campaign is None
            and self.high_thumb_size_campaign is None
            and self.normal_aligned_smooth_lift_campaign is None
            and self.actual_contact_grasp_pose_campaign is None
        ):
            raise ValueError(
                "contact_alignment requires a registered feedback campaign"
            )
        if self.aligned_contact_campaign is not None:
            if self.size_campaign is not None:
                raise ValueError(
                    "aligned_contact_campaign cannot be combined with size_campaign"
                )
            if self.control_protocol is None:
                raise ValueError(
                    "aligned-contact campaigns require control_protocol"
                )
            if self.tuning_strategy != "aligned_contacts":
                raise ValueError(
                    "aligned-contact campaigns require aligned_contacts tuning_strategy"
                )
            campaign = self.aligned_contact_campaign
            constraints = self.pose_constraints
            alignment = self.contact_alignment
            assert constraints is not None and alignment is not None
            if any(
                not constraints.finger_down_tilt_deg[0]
                <= centre
                <= constraints.finger_down_tilt_deg[1]
                for centre in campaign.tilt_band_centers_deg
            ):
                raise ValueError(
                    "tilt band centres must lie inside finger_down_tilt_deg"
                )
            if tuple(self.search_bounds.palm_pitch_values_deg) != tuple(
                campaign.tilt_band_centers_deg
            ):
                raise ValueError(
                    "search bounds pitch values must declare the v4 tilt bands"
                )
            if self.search_bounds.dynamic_candidate_count != (
                campaign.dynamic_candidate_count
            ):
                raise ValueError(
                    "search dynamic candidate count must match aligned campaign"
                )
            if self.search_bounds.local_refine_seed_count != (
                len(campaign.tilt_band_centers_deg)
                * campaign.grasp_refine_seed_count_per_band
            ):
                raise ValueError(
                    "search local seed count must match aligned campaign"
                )
            if self.search_bounds.local_refine_per_seed != (
                campaign.grasp_refine_per_seed
            ):
                raise ValueError(
                    "search local samples must match aligned campaign"
                )
            if self.search_bounds.final_candidate_count != (
                campaign.exact_candidate_count
            ):
                raise ValueError(
                    "search final candidate count must match aligned campaign"
                )
            if not math.isclose(
                alignment.verify_continuous_s,
                self.control_protocol.stable_window_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "contact alignment verification must match the control stable window"
                )
            if not math.isclose(
                alignment.operation_aligned_duty,
                self.evaluation.simultaneous_contact_duty,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "operation aligned duty must match simultaneous contact duty"
                )
            envelope = campaign.perturbation_envelope
            for envelope_value, robustness_value, label in (
                (
                    envelope.cube_center_xy_delta_m,
                    self.robustness.position_xy_delta_m,
                    "cube XY",
                ),
                (
                    envelope.cube_gap_m,
                    self.robustness.z_offset_delta_m,
                    "cube gap",
                ),
                (
                    envelope.cube_rpy_delta_deg,
                    self.robustness.rpy_delta_deg,
                    "cube RPY",
                ),
                (
                    envelope.mass_scale,
                    self.robustness.mass_scale,
                    "mass scale",
                ),
                (
                    envelope.friction_delta,
                    self.robustness.friction_delta,
                    "friction",
                ),
            ):
                if envelope_value != robustness_value:
                    raise ValueError(
                        f"aligned perturbation {label} range must match robustness"
                    )
        if self.far_hand_campaign is not None:
            if self.size_campaign is not None or self.aligned_contact_campaign is not None:
                raise ValueError(
                    "far_hand_campaign cannot be combined with legacy campaigns"
                )
            if self.control_protocol is None or self.contact_alignment is None:
                raise ValueError(
                    "far-hand fingertip campaigns require feedback control and "
                    "contact alignment"
                )
            expected_strategy = (
                "pose_preserving_grasp"
                if self.pose_preservation is not None
                else "far_hand_fingertip"
            )
            if self.tuning_strategy != expected_strategy:
                raise ValueError(
                    "far-hand campaign tuning_strategy disagrees with its schema"
                )
            campaign = self.far_hand_campaign
            constraints = self.far_hand_pose_constraints
            assert constraints is not None
            if tuple(self.search_bounds.palm_pitch_values_deg) != tuple(
                campaign.tilt_band_centers_deg
            ):
                raise ValueError(
                    "search pitch values must declare the v5 tilt bands"
                )
            if any(
                not constraints.finger_down_tilt_deg[0]
                <= centre
                <= constraints.finger_down_tilt_deg[1]
                for centre in campaign.tilt_band_centers_deg
            ):
                raise ValueError(
                    "v5 tilt band centres must lie inside pose constraints"
                )
            if tuple(self.candidate_faces) != (
                campaign.primary_face,
                campaign.fallback_face,
            ):
                raise ValueError(
                    "v5 candidate faces must list primary then fallback topology"
                )
            if self.search_bounds.dynamic_candidate_count != (
                campaign.dynamic_candidate_count
            ):
                raise ValueError(
                    "search dynamic candidate count must match v5 campaign"
                )
            if self.search_bounds.local_refine_seed_count != (
                len(campaign.tilt_band_centers_deg)
                * campaign.grasp_refine_seed_count_per_band
            ):
                raise ValueError("search local seed count must match v5 campaign")
            if self.search_bounds.local_refine_per_seed != (
                campaign.grasp_refine_per_seed
            ):
                raise ValueError("search local sample count must match v5 campaign")
            if self.search_bounds.final_candidate_count != (
                campaign.exact_candidate_count
            ):
                raise ValueError("search finalist count must match v5 campaign")
            if not math.isclose(
                self.contact_alignment.verify_continuous_s,
                self.control_protocol.stable_window_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "v5 alignment verification must match stable grasp window"
                )
            envelope = campaign.perturbation_envelope
            for envelope_value, robustness_value, label in (
                (
                    envelope.cube_center_xy_delta_m,
                    self.robustness.position_xy_delta_m,
                    "cube XY",
                ),
                (
                    envelope.cube_gap_m,
                    self.robustness.z_offset_delta_m,
                    "cube gap",
                ),
                (
                    envelope.cube_rpy_delta_deg,
                    self.robustness.rpy_delta_deg,
                    "cube RPY",
                ),
                (envelope.mass_scale, self.robustness.mass_scale, "mass scale"),
                (
                    envelope.friction_delta,
                    self.robustness.friction_delta,
                    "friction",
                ),
            ):
                if envelope_value != robustness_value:
                    raise ValueError(
                        f"far-hand perturbation {label} must match robustness"
                    )
        if self.high_thumb_size_campaign is not None:
            if any(
                campaign is not None
                for campaign in (
                    self.size_campaign,
                    self.aligned_contact_campaign,
                    self.far_hand_campaign,
                    self.normal_aligned_smooth_lift_campaign,
                    self.actual_contact_grasp_pose_campaign,
                )
            ):
                raise ValueError(
                    "high_thumb_size_campaign cannot be combined with another "
                    "campaign"
                )
            if (
                self.control_protocol is None
                or self.contact_alignment is None
                or self.pose_preservation is None
            ):
                raise ValueError(
                    "high-thumb campaigns require feedback control, contact "
                    "alignment and pose preservation"
                )
            campaign = self.high_thumb_size_campaign
            bounds = self.search_bounds
            if self.tuning_strategy != "high_thumb_variable_size_pose_preserving":
                raise ValueError(
                    "high-thumb campaigns require their dedicated tuning strategy"
                )
            if bounds.seed != campaign.seed:
                raise ValueError("search seed must match high-thumb campaign seed")
            if bounds.actuator_targets_rad != campaign.grasp_target_bounds_rad:
                raise ValueError(
                    "search grasp bounds must match high-thumb campaign bounds"
                )
            if bounds.pregrasp_targets_rad != campaign.pregrasp_target_bounds_rad:
                raise ValueError(
                    "search pregrasp bounds must match high-thumb campaign bounds"
                )
            if (
                bounds.manipulation_delta_rad
                != campaign.manipulation_delta_bounds_rad
            ):
                raise ValueError(
                    "search manipulation bounds must match high-thumb campaign bounds"
                )
            for actual, expected, label in (
                (
                    bounds.dynamic_candidate_count,
                    campaign.dynamic_candidate_count,
                    "dynamic candidate count",
                ),
                (
                    bounds.local_refine_seed_count,
                    campaign.local_refine_seed_count,
                    "local refine seed count",
                ),
                (
                    bounds.local_refine_per_seed,
                    campaign.local_refine_per_seed,
                    "local refinement count",
                ),
                (
                    bounds.final_candidate_count,
                    campaign.exact_candidate_count,
                    "exact candidate count",
                ),
                (
                    bounds.perturbations_per_final_candidate,
                    campaign.perturbations_per_grasp,
                    "grasp perturbation count",
                ),
            ):
                if actual != expected:
                    raise ValueError(
                        f"search {label} must match high-thumb campaign"
                    )
            if not math.isclose(
                self.contact_alignment.verify_continuous_s,
                self.control_protocol.stable_window_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "high-thumb alignment verification must match stable window"
                )
            if tuple(self.robustness.edge_m) != campaign.coarse_edges_m:
                raise ValueError(
                    "robustness edges must match high-thumb coarse edges"
                )
            if tuple(self.robustness.mass_kg) != (campaign.fixed_mass_kg,):
                raise ValueError("robustness mass must match fixed campaign mass")
            if tuple(self.robustness.friction) != (campaign.friction,):
                raise ValueError(
                    "robustness friction must match fixed campaign friction"
                )
        if self.normal_aligned_smooth_lift_campaign is not None:
            if any(
                campaign is not None
                for campaign in (
                    self.size_campaign,
                    self.aligned_contact_campaign,
                    self.far_hand_campaign,
                    self.high_thumb_size_campaign,
                    self.actual_contact_grasp_pose_campaign,
                )
            ):
                raise ValueError(
                    "normal-aligned campaign cannot be combined with another campaign"
                )
            if any(
                value is None
                for value in (
                    self.control_protocol,
                    self.contact_alignment,
                    self.pose_preservation,
                    self.closure_alignment,
                    self.motion_smoothness,
                )
            ):
                raise ValueError(
                    "normal-aligned campaigns require feedback, contact/closure "
                    "alignment, pose preservation and motion smoothness"
                )
            if self.tuning_strategy != "normal_aligned_smooth_vertical_lift":
                raise ValueError(
                    "normal-aligned campaigns require their dedicated tuning strategy"
                )
            protocol = self.control_protocol
            assert protocol is not None
            if protocol.manipulation_profile != "minimum_jerk_quintic":
                raise ValueError("schema-v8 manipulation must use minimum jerk")
            if protocol.close_duration_options_s != (1.0, 1.25, 1.5):
                raise ValueError(
                    "schema-v8 close duration options must be 1.0, 1.25 and 1.5 s"
                )
            campaign = self.normal_aligned_smooth_lift_campaign
            bounds = self.search_bounds
            thumb = "left_hand_thumb_bend_joint_actuator"
            if bounds.seed != campaign.seed:
                raise ValueError("schema-v8 search seed must match campaign")
            if bounds.actuator_targets_rad[thumb] != (
                campaign.thumb_targets_rad[0], campaign.thumb_targets_rad[-1]
            ):
                raise ValueError("schema-v8 thumb bounds must match campaign")
            if tuple(self.robustness.edge_m) != campaign.edges_m:
                raise ValueError("schema-v8 robustness edges must match campaign")
            if tuple(self.robustness.mass_kg) != (campaign.fixed_mass_kg,):
                raise ValueError("schema-v8 robustness mass must match campaign")
            if tuple(self.robustness.friction) != (campaign.friction,):
                raise ValueError("schema-v8 robustness friction must match campaign")
        if self.actual_contact_grasp_pose_campaign is not None:
            if any(
                campaign is not None
                for campaign in (
                    self.size_campaign,
                    self.aligned_contact_campaign,
                    self.far_hand_campaign,
                    self.high_thumb_size_campaign,
                    self.normal_aligned_smooth_lift_campaign,
                )
            ):
                raise ValueError(
                    "actual-contact grasp-pose campaign cannot be combined with "
                    "another campaign"
                )
            if any(
                value is None
                for value in (
                    self.control_protocol,
                    self.contact_alignment,
                    self.pose_preservation,
                    self.closure_alignment,
                    self.motion_smoothness,
                    self.actual_contact_grasp_pose,
                )
            ):
                raise ValueError(
                    "actual-contact campaigns require feedback, contact/closure "
                    "alignment, pose preservation, smoothness and grasp-pose settings"
                )
            expected_actual_contact_strategy = (
                (
                    (
                        "joint_pair_near_zero_rolling_slip_contact_preserving_planned_lift"
                        if self.joint_pair_feedback is not None
                        and self.joint_pair_feedback.schema_version >= 2
                        else "joint_pair_near_zero_contact_preserving_planned_lift"
                    )
                    if joint_pair_selected
                    else "contact_preserving_planned_lift"
                )
                if self.contact_preserving_planned_lift_campaign is not None
                else (
                    "scaled_contact_downsize_actual_grasp_then_lift"
                    if self.scaled_contact_downsize_campaign is not None
                    else (
                        "contact_point_targeted_actual_grasp_pose"
                        if self.contact_point_search is not None
                        else "actual_contact_grasp_pose_smooth_vertical_lift"
                    )
                )
            )
            if self.tuning_strategy != expected_actual_contact_strategy:
                raise ValueError(
                    "actual-contact campaigns require their dedicated tuning strategy"
                )
            protocol = self.control_protocol
            grasp_pose = self.actual_contact_grasp_pose
            campaign = self.actual_contact_grasp_pose_campaign
            assert protocol is not None and grasp_pose is not None
            if protocol.manipulation_profile != "minimum_jerk_quintic":
                raise ValueError("schema-v9 manipulation must use minimum jerk")
            if not math.isclose(
                grasp_pose.verify_continuous_s,
                protocol.stable_window_s,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "schema-v9 grasp-pose verify window must match control protocol"
                )
            bounds = self.search_bounds
            if bounds.seed != campaign.seed:
                raise ValueError("schema-v9 search seed must match campaign")
            if (
                self.contact_point_search is None
                and self.contact_preserving_planned_lift_campaign is None
            ):
                if bounds.dynamic_candidate_count != (
                    campaign.maximum_dynamic_grasp_candidate_count
                ):
                    raise ValueError(
                        "schema-v9 dynamic candidate count must match campaign"
                    )
                for actual, expected, label in (
                    (
                        bounds.local_refine_seed_count,
                        campaign.local_pose_count,
                        "local pose count",
                    ),
                    (
                        bounds.local_refine_per_seed,
                        campaign.local_refine_per_pose,
                        "local refinement count",
                    ),
                    (
                        bounds.final_candidate_count,
                        campaign.exact_grasp_pose_count,
                        "exact grasp-pose count",
                    ),
                    (
                        bounds.perturbations_per_final_candidate,
                        campaign.perturbations_per_trajectory,
                        "trajectory perturbation count",
                    ),
                ):
                    if actual != expected:
                        raise ValueError(f"schema-v9 search {label} must match campaign")
            if tuple(self.robustness.edge_m) != campaign.edges_m:
                raise ValueError("schema-v9 robustness edges must match campaign")
            if tuple(self.robustness.mass_kg) != (campaign.fixed_mass_kg,):
                raise ValueError("schema-v9 robustness mass must match campaign")
            if tuple(self.robustness.friction) != (campaign.friction,):
                raise ValueError("schema-v9 robustness friction must match campaign")
            contact_point_search = self.contact_point_search
            if contact_point_search is not None:
                if campaign.edges_m != (contact_point_search.seed_plan.cube_edge_m,):
                    raise ValueError(
                        "contact-point campaign edge must match its seed point plan"
                    )
                if not campaign.allow_single_edge:
                    raise ValueError(
                        "contact-point campaign must explicitly allow one edge"
                    )
                if campaign.grasp_pose_identity != (
                    "cube_hand_topology_contact_point_plan_nominal_qpos_sha256"
                ):
                    raise ValueError(
                        "contact-point grasp identity must bind point_plan_id"
                    )
                for actual, expected, label in (
                    (
                        bounds.dynamic_candidate_count,
                        contact_point_search.dynamic_candidate_count,
                        "dynamic candidate count",
                    ),
                    (
                        bounds.local_refine_seed_count,
                        contact_point_search.local_pose_count,
                        "local pose count",
                    ),
                    (
                        bounds.local_refine_per_seed,
                        contact_point_search.local_refine_per_pose,
                        "local refinement count",
                    ),
                    (
                        bounds.final_candidate_count,
                        contact_point_search.exact_candidate_count,
                        "exact candidate count",
                    ),
                    (
                        bounds.perturbations_per_final_candidate,
                        contact_point_search.perturbations_per_trajectory,
                        "perturbation count",
                    ),
                ):
                    if actual != expected:
                        raise ValueError(
                            f"contact-point search {label} must match campaign bounds"
                        )
            relative_search = self.relative_wrist_pose_search
            if relative_search is not None:
                if relative_search.root_cube_distance_m != (
                    self.far_hand_pose_constraints.root_cube_distance_m
                ):
                    raise ValueError(
                        "relative wrist-pose root distance must match pose constraints"
                    )
                if relative_search.primary_anchor_candidate_id <= 0:
                    raise ValueError("relative wrist-pose search requires an anchor")
            downsize = self.scaled_contact_downsize_campaign
            if downsize is not None:
                if campaign.edges_m != downsize.edges_m:
                    raise ValueError(
                        "actual-contact edges must match scaled downsize campaign"
                    )
                if not math.isclose(
                    campaign.fixed_mass_kg,
                    downsize.fixed_mass_kg,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "actual-contact mass must match scaled downsize campaign"
                    )
                if not math.isclose(
                    campaign.friction,
                    downsize.friction,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "actual-contact friction must match scaled downsize campaign"
                    )
                if campaign.source_pose_manifest != downsize.source_manifest:
                    raise ValueError(
                        "actual-contact source manifest must match scaled downsize campaign"
                    )
                for actual, expected, label in (
                    (
                        bounds.dynamic_candidate_count,
                        downsize.maximum_dynamic_grasp_candidate_count,
                        "dynamic candidate count",
                    ),
                    (
                        bounds.local_refine_seed_count,
                        downsize.local_pose_count,
                        "local pose count",
                    ),
                    (
                        bounds.local_refine_per_seed,
                        downsize.local_refine_per_pose,
                        "local refinement count",
                    ),
                    (
                        bounds.final_candidate_count,
                        downsize.selected_grasp_count,
                        "selected grasp count",
                    ),
                    (
                        bounds.perturbations_per_final_candidate,
                        downsize.perturbations_per_published,
                        "published perturbation count",
                    ),
                ):
                    if actual != expected:
                        raise ValueError(
                            f"scaled downsize search {label} must match campaign"
                        )
                if self.robustness.perturbation_count != downsize.robustness_trials:
                    raise ValueError(
                        "robustness trials must match scaled downsize campaign"
                    )
                if (
                    self.robustness.required_pass_count
                    != downsize.robustness_required_passes
                ):
                    raise ValueError(
                        "robustness pass count must match scaled downsize campaign"
                    )
            planned_lift = self.contact_preserving_planned_lift_campaign
            if planned_lift is not None:
                if campaign.edges_m != planned_lift.edges_m:
                    raise ValueError(
                        "actual-contact edges must match planned-lift campaign"
                    )
                if not math.isclose(
                    campaign.fixed_mass_kg,
                    planned_lift.fixed_mass_kg,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "actual-contact mass must match planned-lift campaign"
                    )
                if not math.isclose(
                    campaign.friction,
                    planned_lift.friction,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "actual-contact friction must match planned-lift campaign"
                    )
                if campaign.source_pose_manifest != planned_lift.source_grasp_catalog:
                    raise ValueError(
                        "actual-contact source must match planned-lift campaign"
                    )
                if planned_lift.schema_version == 1:
                    expected_dynamic_count = (
                        planned_lift.maximum_plan_candidate_count
                    )
                else:
                    assert planned_lift.dls_start_count is not None
                    expected_dynamic_count = planned_lift.dls_start_count
                for actual, expected, label in (
                    (
                        bounds.dynamic_candidate_count,
                        expected_dynamic_count,
                        (
                            "plan candidate count"
                            if planned_lift.schema_version == 1
                            else "DLS start count"
                        ),
                    ),
                    (
                        bounds.local_refine_seed_count,
                        planned_lift.feedback_refine_plan_count,
                        "feedback refine plan count",
                    ),
                    (
                        bounds.local_refine_per_seed,
                        planned_lift.feedback_refine_per_plan,
                        "feedback refinement count",
                    ),
                    (
                        bounds.final_candidate_count,
                        planned_lift.final_candidate_count,
                        "final candidate count",
                    ),
                    (
                        bounds.perturbations_per_final_candidate,
                        planned_lift.perturbations_per_final,
                        "final perturbation count",
                    ),
                ):
                    if actual != expected:
                        raise ValueError(
                            f"planned-lift search {label} must match campaign"
                        )
                if self.robustness.perturbation_count != planned_lift.robustness_trials:
                    raise ValueError(
                        "robustness trials must match planned-lift campaign"
                    )
                if (
                    self.robustness.required_pass_count
                    != planned_lift.robustness_required_passes
                ):
                    raise ValueError(
                        "robustness pass count must match planned-lift campaign"
                    )
        families = self.robustness.case_families
        if self.size_campaign is None and families is not None:
            raise ValueError("robustness case_families require a size_campaign")
        if self.size_campaign is not None:
            if families is None:
                raise ValueError("size_campaign requires robustness case_families")
            campaign = self.size_campaign
            if campaign.target_sampling_policy == (
                "independent_absolute_pregrasp_and_final"
            ):
                if self.search_bounds.final_target_delta_rad is not None:
                    raise ValueError(
                        "size_campaign search must sample pregrasp and final targets independently"
                    )
                if self.search_bounds.manipulation_delta_rad is not None:
                    raise ValueError(
                        "legacy size_campaign cannot define manipulation delta bounds"
                    )
                if self.control_protocol is not None:
                    raise ValueError(
                        "legacy size_campaign cannot define a feedback control protocol"
                    )
            else:
                if self.control_protocol is None:
                    raise ValueError(
                        "grasp-then-manipulate campaign requires control_protocol"
                    )
                if self.search_bounds.final_target_delta_rad is not None:
                    raise ValueError(
                        "grasp-then-manipulate search cannot use legacy final target deltas"
                    )
                if self.search_bounds.manipulation_delta_rad is None:
                    raise ValueError(
                        "grasp-then-manipulate search requires manipulation delta bounds"
                    )
            if not math.isclose(
                campaign.discovery_mass_kg,
                families.fixed_mass_kg,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "size_campaign discovery mass must match robustness fixed mass"
                )
            if tuple(self.robustness.friction) != tuple(families.friction):
                raise ValueError(
                    "robustness friction must match case_families friction"
                )
            if (
                campaign.coarse_edges_m[0] < families.edge_limits_m[0]
                or campaign.coarse_edges_m[-1] > families.edge_limits_m[1]
            ):
                raise ValueError(
                    "size_campaign coarse edges must lie inside robustness edge limits"
                )
        if self.control_protocol is not None:
            gate = self.control_protocol.grasp_gate
            if not math.isclose(
                gate.min_target_face_force_n,
                self.evaluation.contact_force_min_n,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "grasp gate force must match evaluation contact force threshold"
                )
            if not math.isclose(
                gate.min_target_force_fraction,
                self.evaluation.target_force_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "grasp gate force purity must match evaluation target force fraction"
                )


COMMON_ROBUSTNESS = RobustnessParameters(
    edge_m=(0.026, 0.028, 0.030, 0.032, 0.034),
    mass_kg=(0.010, 0.020, 0.030, 0.050),
    friction=(0.4, 0.6, 0.8, 1.0, 1.2),
    perturbation_count=50,
    required_pass_count=45,
    seed=20260821,
    position_xy_delta_m=(-0.0015, 0.0015),
    rpy_delta_deg=(-3.0, 3.0),
    mass_scale=(0.9, 1.1),
    friction_delta=(-0.1, 0.1),
    z_offset_delta_m=(0.0, 0.0005),
)


LEGACY_V1_EXPERIMENT = ExperimentDefinition(
    experiment_id=DEFAULT_V1_EXPERIMENT_ID,
    description="Legacy schema-v1 fixed-palm three-finger cube lift",
    evaluation=EvaluationSettings(),
    candidate_faces=(),
    search_bounds=SearchBounds(
        palm_pitch_values_deg=(15.0, 20.0, 25.0),
        hand_roll_deg=(-5.0, 5.0),
        hand_yaw_deg=(-5.0, 5.0),
        cube_position_in_root_m={
            "x": (0.05, 0.10),
            "y": (-0.04, -0.005),
            "z": (0.08, 0.13),
        },
        cube_yaw_deg=(-5.0, 5.0),
        actuator_targets_rad={
            "left_hand_thumb_bend_joint_actuator": (0.35, 1.05),
            "left_hand_thumb_rota_joint1_actuator": (0.55, 1.10),
            "left_hand_thumb_rota_joint2_actuator": (0.75, 1.40),
            "left_hand_index_bend_joint_actuator": (-0.12, 0.12),
            "left_hand_index_joint1_actuator": (0.75, 1.50),
            "left_hand_index_joint2_actuator": (0.65, 1.25),
            "left_hand_mid_joint1_actuator": (0.75, 1.50),
            "left_hand_mid_joint2_actuator": (0.65, 1.25),
        },
        seed=20260821,
        kinematic_samples_per_pitch=1,
        dynamic_candidate_count=64,
        local_refine_seed_count=16,
        local_refine_per_seed=16,
        final_candidate_count=16,
        perturbations_per_final_candidate=16,
        fallback_kinematic_samples_per_pitch=1,
        fallback_candidate_count=1,
    ),
    robustness=COMMON_ROBUSTNESS,
    artifact_root="artifacts/left_three_finger_cube",
)


_REGISTRY: dict[str, ExperimentDefinition] = {
    LEGACY_V1_EXPERIMENT.experiment_id: LEGACY_V1_EXPERIMENT
}
_BUILTINS_LOADED = False


def register_experiment(
    definition: ExperimentDefinition, *, replace: bool = False
) -> ExperimentDefinition:
    """Register a definition, rejecting accidental ID replacement by default."""

    existing = _REGISTRY.get(definition.experiment_id)
    if existing is not None and existing != definition and not replace:
        raise ValueError(f"experiment_id {definition.experiment_id!r} is already registered")
    _REGISTRY[definition.experiment_id] = definition
    return definition


def _load_builtin_experiments() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    importlib.import_module("xhand_grasp.experiments")


def registered_experiments() -> Mapping[str, ExperimentDefinition]:
    _load_builtin_experiments()
    return MappingProxyType(dict(sorted(_REGISTRY.items())))


def get_experiment(experiment_id: str) -> ExperimentDefinition:
    _load_builtin_experiments()
    try:
        return _REGISTRY[experiment_id]
    except KeyError as error:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown experiment_id {experiment_id!r}; registered: {known}") from error


def resolve_experiment(config_or_id: Mapping[str, Any] | str) -> ExperimentDefinition:
    """Resolve an ID or a versioned configuration to a registered definition.

    Schema-v1 files predate ``experiment_id`` and intentionally resolve to the
    historical three-finger experiment.  Schema-v2 and later files must name
    their experiment explicitly so evaluation semantics cannot change silently.
    """

    if isinstance(config_or_id, str):
        return get_experiment(config_or_id)
    if not isinstance(config_or_id, Mapping):
        raise TypeError("experiment selection must be an experiment_id or configuration mapping")
    schema_version = config_or_id.get("schema_version")
    if schema_version == 1:
        experiment_id = config_or_id.get("experiment_id", DEFAULT_V1_EXPERIMENT_ID)
    elif schema_version in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        experiment_id = config_or_id.get("experiment_id")
        if not experiment_id:
            raise ValueError(f"schema_version {schema_version} requires experiment_id")
    else:
        raise ValueError(f"unsupported schema_version {schema_version!r}")
    if not isinstance(experiment_id, str):
        raise ValueError("experiment_id must be a string")
    definition = get_experiment(experiment_id)
    is_v4_definition = definition.aligned_contact_campaign is not None
    is_v5_definition = definition.far_hand_campaign is not None
    is_v6_definition = definition.pose_preservation is not None
    is_v7_definition = definition.high_thumb_size_campaign is not None
    is_v8_definition = (
        definition.normal_aligned_smooth_lift_campaign is not None
    )
    is_v9_definition = (
        definition.actual_contact_grasp_pose_campaign is not None
    )
    is_v11_definition = definition.relative_wrist_pose_search is not None
    is_v12_definition = definition.contact_point_search is not None
    is_v13_definition = definition.scaled_contact_downsize_campaign is not None
    is_v14_definition = (
        definition.contact_preserving_planned_lift_campaign is not None
    )
    is_joint_pair_definition = (
        definition.joint_pair_alignment is not None
        and definition.joint_pair_feedback is not None
    )
    is_v15_definition = bool(
        is_joint_pair_definition
        and definition.joint_pair_feedback is not None
        and definition.joint_pair_feedback.schema_version == 1
    )
    is_v16_definition = bool(
        is_joint_pair_definition
        and definition.joint_pair_feedback is not None
        and definition.joint_pair_feedback.schema_version == 2
    )
    if schema_version in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) and definition.control_protocol is None:
        raise ValueError(
            f"schema_version {schema_version} requires a registered control_protocol"
        )
    if schema_version == 4 and (not is_v4_definition or is_v5_definition):
        raise ValueError(
            "schema_version 4 requires a registered aligned-contact experiment"
        )
    if schema_version == 5 and (
        not is_v5_definition or is_v6_definition or is_v7_definition
    ):
        raise ValueError(
            "schema_version 5 requires a registered far-hand fingertip experiment"
        )
    if schema_version == 6 and (
        not is_v5_definition
        or not is_v6_definition
        or is_v7_definition
        or is_v8_definition
        or is_v9_definition
        or is_v14_definition
    ):
        raise ValueError(
            "schema_version 6 requires a registered pose-preserving far-hand "
            "experiment"
        )
    if schema_version == 7 and (
        not is_v7_definition
        or not is_v6_definition
        or is_v5_definition
        or is_v8_definition
        or is_v9_definition
        or is_v14_definition
    ):
        raise ValueError(
            "schema_version 7 requires a registered high-thumb variable-size "
            "pose-preserving experiment"
        )
    if schema_version == 8 and (
        not is_v8_definition
        or not is_v6_definition
        or is_v5_definition
        or is_v7_definition
        or is_v9_definition
        or is_v14_definition
    ):
        raise ValueError(
            "schema_version 8 requires a registered normal-aligned smooth-lift "
            "pose-preserving experiment"
        )
    if schema_version in (9, 10) and (
        not is_v9_definition
        or not is_v6_definition
        or is_v5_definition
        or is_v7_definition
        or is_v8_definition
        or is_v11_definition
        or is_v12_definition
        or is_v13_definition
        or is_v14_definition
    ):
        raise ValueError(
            f"schema_version {schema_version} requires a registered actual-contact grasp-pose "
            "smooth-lift experiment"
        )
    if schema_version == 11 and (
        not is_v9_definition
        or not is_v6_definition
        or is_v5_definition
        or is_v7_definition
        or is_v8_definition
        or not is_v11_definition
        or is_v12_definition
        or is_v13_definition
        or is_v14_definition
    ):
        raise ValueError(
            "schema_version 11 requires a registered cube-relative wrist-pose "
            "actual-contact smooth-lift experiment"
        )
    if schema_version == 12 and (
        not is_v9_definition
        or not is_v6_definition
        or is_v5_definition
        or is_v7_definition
        or is_v8_definition
        or is_v11_definition
        or not is_v12_definition
        or is_v13_definition
        or is_v14_definition
    ):
        raise ValueError(
            "schema_version 12 requires a registered contact-point-targeted "
            "actual-contact grasp-pose experiment"
        )
    if schema_version == 13 and (
        not is_v9_definition
        or not is_v6_definition
        or is_v5_definition
        or is_v7_definition
        or is_v8_definition
        or is_v11_definition
        or is_v12_definition
        or not is_v13_definition
        or is_v14_definition
    ):
        raise ValueError(
            "schema_version 13 requires a registered scaled-contact downsize "
            "actual-contact grasp-pose experiment"
        )
    if schema_version == 14 and (
        not is_v9_definition
        or not is_v6_definition
        or is_v5_definition
        or is_v7_definition
        or is_v8_definition
        or is_v11_definition
        or is_v12_definition
        or is_v13_definition
        or not is_v14_definition
        or is_joint_pair_definition
    ):
        raise ValueError(
            "schema_version 14 requires a registered contact-preserving "
            "planned-lift actual-contact experiment"
        )
    if schema_version == 15 and (
        not is_v9_definition
        or not is_v6_definition
        or is_v5_definition
        or is_v7_definition
        or is_v8_definition
        or is_v11_definition
        or is_v12_definition
        or is_v13_definition
        or not is_v14_definition
        or not is_v15_definition
    ):
        raise ValueError(
            "schema_version 15 requires a registered joint-pair-aligned "
            "contact-preserving planned-lift experiment"
        )
    if schema_version == 16 and (
        not is_v9_definition
        or not is_v6_definition
        or is_v5_definition
        or is_v7_definition
        or is_v8_definition
        or is_v11_definition
        or is_v12_definition
        or is_v13_definition
        or not is_v14_definition
        or not is_v16_definition
    ):
        raise ValueError(
            "schema_version 16 requires a registered joint-pair-aligned "
            "rolling-slip contact-preserving planned-lift experiment"
        )
    if schema_version == 3 and (
        is_v4_definition
        or is_v5_definition
        or is_v7_definition
        or is_v8_definition
        or is_v9_definition
    ):
        raise ValueError(
            "aligned-contact experiments require schema_version 4 or 5"
        )
    if schema_version == 4 and is_v5_definition:
        raise ValueError("far-hand fingertip experiments require schema_version 5")
    if schema_version == 5 and is_v4_definition:
        raise ValueError("schema-v4 aligned experiments require schema_version 4")
    if schema_version not in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) and definition.control_protocol is not None:
        raise ValueError(
            "feedback-gated experiments require schema_version 3"
        )
    topology = config_or_id.get("contact_topology")
    if topology is not None:
        if not isinstance(topology, Mapping):
            raise ValueError("contact_topology must be a mapping")
        declared_faces = topology.get("target_faces")
        if declared_faces is None:
            raise ValueError("contact_topology requires target_faces")
        if not isinstance(declared_faces, Mapping):
            raise ValueError("contact_topology.target_faces must be a mapping")
        assignment = OpposedFaceAssignment.from_mapping(declared_faces)
        if not definition.candidate_faces:
            raise ValueError("the selected experiment does not define target faces")
        if assignment not in definition.candidate_faces:
            raise ValueError(
                "configuration target_faces are outside the experiment candidate faces"
            )
    return definition


# Semantic aliases kept deliberately small: callers may say parse when handling
# configuration and resolve when they already have an ID.
parse_experiment = resolve_experiment
