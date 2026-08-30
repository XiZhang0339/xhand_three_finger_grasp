"""Schema-v15 joint-pair constrained sequential contact planner.

This module is an additive extension of the frozen schema-v14 planners.  It
keeps their 20-segment/21-knot, four-backoff and canonical 17-probe contracts,
but augments every checkpoint response with the directed index-to-middle
residual ``(v_x / v_y, v_z / v_y)``.  The local ``2 x 8`` response is fitted
from the same real probe branches, constrained by a conservative regular
octagon, and audited along the shared quintic command interpolation at 1 ms.

Checkpoint branches remain search evidence only.  As in v14, a materialized
plan is not success evidence until the normal runner executes it again from
the initial no-contact state.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from ..checkpoint import restore_physics_checkpoint
from ..config import ACTIVE_ACTUATORS, ACTIVE_FINGERS
from ..contacts import target_face_contact_centroids
from ..experiment import ManipulationPlanParameters
from ..grasp_pose import canonical_sha256
from ..joint_pair_geometry import (
    DEFAULT_MINIMUM_JOINT_PAIR_SEPARATION_M,
    JointPairBinding,
    joint_pair_angle_from_signed_residual_deg,
    measure_oriented_joint_pair_geometry,
    resolve_joint_pair,
)
from ..simulation import contact_snapshot
from ..trajectory import interpolate_quintic_c2, quintic_c2_knot_derivatives
from .actual_contact_manipulation import (
    GraspPhysicsCheckpoint,
    ProbeSpecification,
    generate_probe_specifications,
)
from .contact_constrained_planner import (
    KNOT_COUNT,
    PLAN_SEGMENT_COUNT,
    ContactConstrainedPlannerSettings,
    ExtendedProbeResponse,
    ExtendedProbeSample,
    TwentyOneKnotContactPlan,
    fit_extended_probe_response,
    materialize_contact_plan_config,
    solve_bounded_projected_least_squares,
)
from .sequential_contact_planner import (
    MuJoCoSequentialPlanningHooks,
    SequentialSegmentRollout,
    _checkpoint_sha256,
    _path_progress,
    _quat_rotation_vector_wxyz,
)


JOINT_PAIR_CONSTRAINED_PLANNER_SCHEMA_VERSION = 1
JOINT_PAIR_PROBE_RESPONSE_SCHEMA_VERSION = 1
JOINT_PAIR_CONSTRAINED_PLAN_SCHEMA_VERSION = 1
DEFAULT_JOINT_PAIR = (
    "left_hand_index_joint1",
    "left_hand_mid_joint1",
)
DEFAULT_CONSTRAINT_ANGLE_DEG = 0.5
DEFAULT_AUDIT_TIMESTEP_S = 0.001
OCTAGON_SIDE_COUNT = 8
_EPSILON = 1e-12


def _readonly_array(
    values: Any,
    shape: tuple[int, ...],
    label: str,
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError(f"{label} must contain finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


def _finite_positive(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _probe_key(sample: ExtendedProbeSample) -> tuple[int, int]:
    return (
        -1
        if sample.kind == "zero"
        else ACTIVE_ACTUATORS.index(str(sample.actuator)),
        int(sample.direction),
    )


@dataclass(frozen=True, slots=True)
class JointPairPlannerSettings:
    """Versioned settings for the additive schema-v15 pair constraint."""

    schema_version: int = JOINT_PAIR_CONSTRAINED_PLANNER_SCHEMA_VERSION
    joint_names: tuple[str, str] = DEFAULT_JOINT_PAIR
    minimum_separation_m: float = DEFAULT_MINIMUM_JOINT_PAIR_SEPARATION_M
    maximum_angle_deg: float = DEFAULT_CONSTRAINT_ANGLE_DEG
    constraint_polygon_sides: int = OCTAGON_SIDE_COUNT
    audit_timestep_s: float = DEFAULT_AUDIT_TIMESTEP_S
    signed_residual_scale: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if int(self.schema_version) != JOINT_PAIR_CONSTRAINED_PLANNER_SCHEMA_VERSION:
            raise ValueError("joint-pair planner schema_version must be 1")
        names = tuple(str(value) for value in self.joint_names)
        if len(names) != 2 or not all(names) or names[0] == names[1]:
            raise ValueError("joint_names must contain two distinct names")
        object.__setattr__(self, "joint_names", names)
        minimum = _finite_positive(self.minimum_separation_m, "minimum_separation_m")
        maximum = _finite_positive(self.maximum_angle_deg, "maximum_angle_deg")
        if maximum >= 90.0:
            raise ValueError("maximum_angle_deg must be below 90 degrees")
        if int(self.constraint_polygon_sides) != OCTAGON_SIDE_COUNT:
            raise ValueError("schema-v15 pair constraints require an octagon")
        audit = _finite_positive(self.audit_timestep_s, "audit_timestep_s")
        scale = self.signed_residual_scale
        if scale is None:
            radius = math.tan(math.radians(maximum))
            scale = (radius, radius)
        normalized_scale = tuple(float(value) for value in scale)
        if (
            len(normalized_scale) != 2
            or not np.isfinite(normalized_scale).all()
            or any(value <= 0.0 for value in normalized_scale)
        ):
            raise ValueError("signed_residual_scale must contain two positive values")
        object.__setattr__(self, "minimum_separation_m", minimum)
        object.__setattr__(self, "maximum_angle_deg", maximum)
        object.__setattr__(self, "constraint_polygon_sides", OCTAGON_SIDE_COUNT)
        object.__setattr__(self, "audit_timestep_s", audit)
        object.__setattr__(self, "signed_residual_scale", normalized_scale)

    @classmethod
    def from_alignment_config(
        cls, value: Mapping[str, Any]
    ) -> "JointPairPlannerSettings":
        """Read the canonical flat ``joint_pair_alignment`` v15 block."""

        if str(value.get("frame")) != "cube_local":
            raise ValueError("joint_pair_alignment.frame must be cube_local")
        if str(value.get("axis")) != "+Y" or value.get("require_positive_y") is not True:
            raise ValueError("joint_pair_alignment must require directed cube-local +Y")
        if str(value.get("residual")) != "vx_over_vy_vz_over_vy":
            raise ValueError("unsupported joint_pair_alignment.residual")
        return cls(
            joint_names=tuple(value.get("joint_names", ())),
            minimum_separation_m=value.get(
                "minimum_length_m", DEFAULT_MINIMUM_JOINT_PAIR_SEPARATION_M
            ),
            maximum_angle_deg=value.get(
                "operation_p95_max_deg", DEFAULT_CONSTRAINT_ANGLE_DEG
            ),
            constraint_polygon_sides=value.get(
                "constraint_polygon_sides", OCTAGON_SIDE_COUNT
            ),
            audit_timestep_s=value.get(
                "audit_timestep_s", DEFAULT_AUDIT_TIMESTEP_S
            ),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "joint_pair_planner_settings_schema_version": self.schema_version,
            "joint_names": list(self.joint_names),
            "frame": "cube_local",
            "axis": "+Y",
            "require_positive_y": True,
            "minimum_separation_m": self.minimum_separation_m,
            "residual": "vx_over_vy_vz_over_vy",
            "maximum_angle_deg": self.maximum_angle_deg,
            "constraint_polygon_sides": self.constraint_polygon_sides,
            "audit_timestep_s": self.audit_timestep_s,
            "signed_residual_scale": list(self.signed_residual_scale),
        }


@dataclass(frozen=True, slots=True)
class JointPairProbeSample:
    """One canonical contact probe augmented with directed pair geometry."""

    extended_sample: ExtendedProbeSample
    joint_pair_signed_residual: np.ndarray
    joint_pair_vector_cube_m: np.ndarray | None = None
    joint_pair_length_m: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.extended_sample, ExtendedProbeSample):
            raise TypeError("extended_sample must be an ExtendedProbeSample")
        residual = _readonly_array(
            self.joint_pair_signed_residual,
            (2,),
            "joint_pair_signed_residual",
        )
        object.__setattr__(self, "joint_pair_signed_residual", residual)
        vector = self.joint_pair_vector_cube_m
        length = self.joint_pair_length_m
        if vector is not None:
            normalized = _readonly_array(
                vector, (3,), "joint_pair_vector_cube_m"
            )
            if float(normalized[1]) <= 0.0:
                raise ValueError("joint-pair probe vector must point toward +Y")
            derived = np.asarray(
                (normalized[0] / normalized[1], normalized[2] / normalized[1])
            )
            if not np.allclose(residual, derived, rtol=1e-10, atol=1e-12):
                raise ValueError("joint-pair vector and signed residual disagree")
            derived_length = float(np.linalg.norm(normalized))
            if length is not None and not math.isclose(
                float(length), derived_length, rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError("joint-pair vector and length disagree")
            length = derived_length
            object.__setattr__(self, "joint_pair_vector_cube_m", normalized)
        if length is not None:
            object.__setattr__(
                self,
                "joint_pair_length_m",
                _finite_positive(length, "joint_pair_length_m"),
            )

    @property
    def angle_to_cube_positive_y_deg(self) -> float:
        return joint_pair_angle_from_signed_residual_deg(
            self.joint_pair_signed_residual
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "JointPairProbeSample":
        pair = value.get("joint_pair_geometry", value)
        if not isinstance(pair, Mapping):
            raise ValueError("joint_pair_geometry must be a mapping")
        residual = pair.get(
            "joint_pair_signed_residual", value.get("joint_pair_signed_residual")
        )
        if residual is None:
            raise ValueError("probe is missing joint_pair_signed_residual")
        vector = pair.get("vector_cube_m", pair.get("joint_pair_vector_cube_m"))
        length = pair.get("length_m", pair.get("joint_pair_length_m"))
        return cls(
            extended_sample=ExtendedProbeSample.from_mapping(value),
            joint_pair_signed_residual=residual,
            joint_pair_vector_cube_m=vector,
            joint_pair_length_m=length,
        )

    def as_mapping(self) -> dict[str, Any]:
        payload = self.extended_sample.as_mapping()
        geometry: dict[str, Any] = {
            "joint_pair_signed_residual": self.joint_pair_signed_residual.tolist(),
            "angle_to_cube_positive_y_deg": self.angle_to_cube_positive_y_deg,
            "points_toward_positive_cube_y": True,
        }
        if self.joint_pair_vector_cube_m is not None:
            geometry["vector_cube_m"] = self.joint_pair_vector_cube_m.tolist()
        if self.joint_pair_length_m is not None:
            geometry["length_m"] = self.joint_pair_length_m
        payload["joint_pair_geometry"] = geometry
        return payload


@dataclass(frozen=True, slots=True)
class JointPairProbeResponse:
    """Contact response plus the fitted signed ``2 x 8`` pair response."""

    extended_response: ExtendedProbeResponse
    joint_pair_signed_residual: np.ndarray
    joint_pair_jacobian_2x8: np.ndarray
    column_evidence: tuple[Mapping[str, Any], ...]
    probe_evidence: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.extended_response, ExtendedProbeResponse):
            raise TypeError("extended_response must be an ExtendedProbeResponse")
        object.__setattr__(
            self,
            "joint_pair_signed_residual",
            _readonly_array(
                self.joint_pair_signed_residual,
                (2,),
                "joint_pair_signed_residual",
            ),
        )
        object.__setattr__(
            self,
            "joint_pair_jacobian_2x8",
            _readonly_array(
                self.joint_pair_jacobian_2x8,
                (2, len(ACTIVE_ACTUATORS)),
                "joint_pair_jacobian_2x8",
            ),
        )
        if len(self.column_evidence) != len(ACTIVE_ACTUATORS):
            raise ValueError("column_evidence must contain eight entries")
        if len(self.probe_evidence) != 1 + 2 * len(ACTIVE_ACTUATORS):
            raise ValueError("probe_evidence must contain exactly 17 probes")
        object.__setattr__(
            self,
            "column_evidence",
            tuple(copy.deepcopy(dict(value)) for value in self.column_evidence),
        )
        object.__setattr__(
            self,
            "probe_evidence",
            tuple(copy.deepcopy(dict(value)) for value in self.probe_evidence),
        )

    @property
    def available_actuator_mask(self) -> np.ndarray:
        return self.extended_response.available_actuator_mask

    @property
    def response_model_id(self) -> str:
        return canonical_sha256(self.as_mapping(include_id=False))

    def as_mapping(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "joint_pair_probe_response_schema_version": (
                JOINT_PAIR_PROBE_RESPONSE_SCHEMA_VERSION
            ),
            "residual_definition": "cube_local_vx_over_vy_vz_over_vy",
            "joint_pair_signed_residual": self.joint_pair_signed_residual.tolist(),
            "joint_pair_jacobian_2x8": self.joint_pair_jacobian_2x8.tolist(),
            "actuator_order": list(ACTIVE_ACTUATORS),
            "available_actuator_mask": self.available_actuator_mask.tolist(),
            "extended_response_model_id": self.extended_response.response_model_id,
            "extended_response": self.extended_response.as_mapping(),
            "column_evidence": [copy.deepcopy(dict(v)) for v in self.column_evidence],
            "probe_evidence": [copy.deepcopy(dict(v)) for v in self.probe_evidence],
        }
        if include_id:
            payload["response_model_id"] = canonical_sha256(payload)
        return payload


def fit_joint_pair_probe_response(
    probe_samples: Sequence[JointPairProbeSample | Mapping[str, Any]],
) -> JointPairProbeResponse:
    """Fit the pair ``2 x 8`` Jacobian from one canonical 17-probe set.

    Contact-unsafe sides use exactly the same central/one-sided availability
    decision as :func:`fit_extended_probe_response`; pair and contact models
    therefore cannot silently rely on different physical branches.
    """

    unsorted = tuple(
        value
        if isinstance(value, JointPairProbeSample)
        else JointPairProbeSample.from_mapping(value)
        for value in probe_samples
    )
    if len(unsorted) != 1 + 2 * len(ACTIVE_ACTUATORS):
        raise ValueError("joint-pair response fitting requires exactly 17 probes")
    keys = [_probe_key(value.extended_sample) for value in unsorted]
    if len(set(keys)) != len(keys):
        raise ValueError("joint-pair response contains duplicate probe directions")
    samples = tuple(sorted(unsorted, key=lambda value: _probe_key(value.extended_sample)))
    extended = fit_extended_probe_response(
        tuple(value.extended_sample for value in samples)
    )
    by_key = {_probe_key(value.extended_sample): value for value in samples}
    zero = by_key.get((-1, 0))
    if zero is None:
        raise ValueError("joint-pair response fitting requires one zero probe")
    bias = zero.joint_pair_signed_residual
    jacobian = np.zeros((2, len(ACTIVE_ACTUATORS)), dtype=np.float64)
    evidence: list[dict[str, Any]] = []
    for column, actuator in enumerate(ACTIVE_ACTUATORS):
        base_evidence = dict(extended.column_evidence[column])
        used = tuple(int(value) for value in base_evidence["used_directions"])
        if used == (-1, 1):
            negative = by_key[(column, -1)]
            positive = by_key[(column, 1)]
            denominator = float(
                positive.extended_sample.applied_delta_rad[column]
                - negative.extended_sample.applied_delta_rad[column]
            )
            if abs(denominator) <= _EPSILON:
                raise ValueError(f"degenerate pair probe spacing for {actuator}")
            jacobian[:, column] = (
                positive.joint_pair_signed_residual
                - negative.joint_pair_signed_residual
            ) / denominator
        elif len(used) == 1:
            direction = used[0]
            sample = by_key[(column, direction)]
            step = float(sample.extended_sample.applied_delta_rad[column])
            if abs(step) <= _EPSILON:
                raise ValueError(f"degenerate pair probe spacing for {actuator}")
            jacobian[:, column] = (
                sample.joint_pair_signed_residual - bias
            ) / step
        elif used:
            raise ValueError("extended response reported unsupported probe evidence")
        evidence.append(
            {
                **base_evidence,
                "joint_pair_derivative_2": jacobian[:, column].tolist(),
            }
        )
    return JointPairProbeResponse(
        extended_response=extended,
        joint_pair_signed_residual=bias,
        joint_pair_jacobian_2x8=jacobian,
        column_evidence=tuple(evidence),
        probe_evidence=tuple(value.as_mapping() for value in samples),
    )


# Compact alias matching the response name used by planner reports.
fit_joint_pair_response = fit_joint_pair_probe_response


@dataclass(frozen=True, slots=True)
class JointPairOctagonalBand:
    """Conservative regular-octagon approximation to the angular disk."""

    maximum_angle_deg: float
    residual_radius: float
    facet_radius: float
    facet_normals_8x2: np.ndarray
    linear_matrix_8x8: np.ndarray
    linear_lower_8: np.ndarray
    linear_upper_8: np.ndarray

    def __post_init__(self) -> None:
        maximum = _finite_positive(self.maximum_angle_deg, "maximum_angle_deg")
        if maximum >= 90.0:
            raise ValueError("maximum_angle_deg must be below 90 degrees")
        radius = _finite_positive(self.residual_radius, "residual_radius")
        facet = _finite_positive(self.facet_radius, "facet_radius")
        expected_radius = math.tan(math.radians(maximum))
        if not math.isclose(radius, expected_radius, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("residual_radius disagrees with maximum_angle_deg")
        if facet > radius + _EPSILON:
            raise ValueError("octagonal facet radius cannot exceed the angular disk")
        for name, values, shape in (
            ("facet_normals_8x2", self.facet_normals_8x2, (8, 2)),
            ("linear_matrix_8x8", self.linear_matrix_8x8, (8, 8)),
            ("linear_lower_8", self.linear_lower_8, (8,)),
            ("linear_upper_8", self.linear_upper_8, (8,)),
        ):
            array = np.asarray(values, dtype=np.float64)
            if array.shape != shape or np.isnan(array).any():
                raise ValueError(f"{name} must have shape {shape} and contain no NaN")
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "maximum_angle_deg", maximum)
        object.__setattr__(self, "residual_radius", radius)
        object.__setattr__(self, "facet_radius", facet)

    def contains_residual(
        self, residual: Sequence[float] | np.ndarray, *, tolerance: float = 1e-12
    ) -> bool:
        value = _readonly_array(residual, (2,), "joint_pair_signed_residual")
        return bool(
            np.all(
                self.facet_normals_8x2 @ value
                <= self.facet_radius + float(tolerance)
            )
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "kind": "inscribed_regular_octagon",
            "side_count": OCTAGON_SIDE_COUNT,
            "guarantees_circular_angle_bound": True,
            "maximum_angle_deg": self.maximum_angle_deg,
            "residual_radius": self.residual_radius,
            "facet_radius": self.facet_radius,
            "facet_normals_8x2": self.facet_normals_8x2.tolist(),
            "linear_matrix_8x8": self.linear_matrix_8x8.tolist(),
            "linear_lower_8": [
                None if not math.isfinite(float(value)) else float(value)
                for value in self.linear_lower_8
            ],
            "linear_upper_8": self.linear_upper_8.tolist(),
        }


def build_joint_pair_octagonal_band(
    joint_pair_signed_residual: Sequence[float] | np.ndarray,
    joint_pair_jacobian_2x8: Sequence[Sequence[float]] | np.ndarray,
    maximum_angle_deg: float = DEFAULT_CONSTRAINT_ANGLE_DEG,
) -> JointPairOctagonalBand:
    """Build eight affine half-spaces for a safe near-zero angle band.

    The octagon is *inscribed* in the exact residual disk, so satisfying the
    linear constraints guarantees ``atan(norm(residual)) <= maximum_angle``.
    This is slightly conservative at facet vertices and avoids accepting the
    corner leakage of a circumscribed polygon.
    """

    bias = _readonly_array(
        joint_pair_signed_residual, (2,), "joint_pair_signed_residual"
    )
    jacobian = _readonly_array(
        joint_pair_jacobian_2x8,
        (2, len(ACTIVE_ACTUATORS)),
        "joint_pair_jacobian_2x8",
    )
    maximum = _finite_positive(maximum_angle_deg, "maximum_angle_deg")
    if maximum >= 90.0:
        raise ValueError("maximum_angle_deg must be below 90 degrees")
    radius = math.tan(math.radians(maximum))
    facet_radius = radius * math.cos(math.pi / OCTAGON_SIDE_COUNT)
    angles = np.arange(OCTAGON_SIDE_COUNT, dtype=np.float64) * (
        2.0 * math.pi / OCTAGON_SIDE_COUNT
    )
    normals = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    linear_matrix = normals @ jacobian
    linear_upper = np.full(OCTAGON_SIDE_COUNT, facet_radius) - normals @ bias
    return JointPairOctagonalBand(
        maximum_angle_deg=maximum,
        residual_radius=radius,
        facet_radius=facet_radius,
        facet_normals_8x2=normals,
        linear_matrix_8x8=linear_matrix,
        linear_lower_8=np.full(OCTAGON_SIDE_COUNT, -math.inf),
        linear_upper_8=linear_upper,
    )


joint_pair_octagonal_linear_band = build_joint_pair_octagonal_band


@dataclass(frozen=True, slots=True)
class ContinuousJointPairAudit:
    """Predicted pair residual at every 1 ms interpolation sample."""

    sample_times_s: np.ndarray
    command_delta_rad: np.ndarray
    joint_pair_signed_residual: np.ndarray
    angle_to_cube_positive_y_deg: np.ndarray
    within_octagonal_band: np.ndarray
    within_exact_angle_bound: np.ndarray
    maximum_angle_deg: float
    audit_timestep_s: float

    def __post_init__(self) -> None:
        times = np.asarray(self.sample_times_s, dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or not np.isfinite(times).all():
            raise ValueError("sample_times_s must be one finite vector")
        if abs(float(times[0])) > _EPSILON or np.any(np.diff(times) <= 0.0):
            raise ValueError("audit times must start at zero and increase")
        count = int(times.size)
        shapes = {
            "command_delta_rad": (self.command_delta_rad, (count, 8), np.float64),
            "joint_pair_signed_residual": (
                self.joint_pair_signed_residual,
                (count, 2),
                np.float64,
            ),
            "angle_to_cube_positive_y_deg": (
                self.angle_to_cube_positive_y_deg,
                (count,),
                np.float64,
            ),
            "within_octagonal_band": (
                self.within_octagonal_band,
                (count,),
                bool,
            ),
            "within_exact_angle_bound": (
                self.within_exact_angle_bound,
                (count,),
                bool,
            ),
        }
        times = times.copy()
        times.setflags(write=False)
        object.__setattr__(self, "sample_times_s", times)
        for name, (values, shape, dtype) in shapes.items():
            object.__setattr__(self, name, _readonly_array(values, shape, name, dtype=dtype))
        maximum = _finite_positive(self.maximum_angle_deg, "maximum_angle_deg")
        audit = _finite_positive(self.audit_timestep_s, "audit_timestep_s")
        deltas = np.diff(times)
        if deltas.size > 1 and np.any(
            np.abs(deltas[:-1] - audit) > max(1e-12, 1e-9 * audit)
        ):
            raise ValueError("audit samples must use the declared timestep")
        object.__setattr__(self, "maximum_angle_deg", maximum)
        object.__setattr__(self, "audit_timestep_s", audit)

    @property
    def passed(self) -> bool:
        return bool(
            np.all(self.within_octagonal_band)
            and np.all(self.within_exact_angle_bound)
        )

    @property
    def maximum_observed_angle_deg(self) -> float:
        return float(np.max(self.angle_to_cube_positive_y_deg))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "audit_schema_version": 1,
            "profile": "shared_clamped_c4_quintic",
            "audit_timestep_s": self.audit_timestep_s,
            "sample_count": int(self.sample_times_s.size),
            "sample_times_s": self.sample_times_s.tolist(),
            "command_delta_rad": self.command_delta_rad.tolist(),
            "joint_pair_signed_residual": self.joint_pair_signed_residual.tolist(),
            "angle_to_cube_positive_y_deg": (
                self.angle_to_cube_positive_y_deg.tolist()
            ),
            "within_octagonal_band": self.within_octagonal_band.tolist(),
            "within_exact_angle_bound": self.within_exact_angle_bound.tolist(),
            "maximum_angle_deg": self.maximum_angle_deg,
            "maximum_observed_angle_deg": self.maximum_observed_angle_deg,
            "passed": self.passed,
        }


def audit_joint_pair_interpolation(
    start_command_delta_rad: Sequence[float] | np.ndarray,
    end_command_delta_rad: Sequence[float] | np.ndarray,
    duration_s: float,
    response: JointPairProbeResponse,
    *,
    maximum_angle_deg: float = DEFAULT_CONSTRAINT_ANGLE_DEG,
    audit_timestep_s: float = DEFAULT_AUDIT_TIMESTEP_S,
) -> ContinuousJointPairAudit:
    """Audit the affine pair response along every shared-quintic millisecond."""

    start = _readonly_array(start_command_delta_rad, (8,), "start_command_delta_rad")
    end = _readonly_array(end_command_delta_rad, (8,), "end_command_delta_rad")
    duration = _finite_positive(duration_s, "duration_s")
    timestep = _finite_positive(audit_timestep_s, "audit_timestep_s")
    full_step_count = int(math.floor(duration / timestep + 1e-12))
    times = np.arange(full_step_count + 1, dtype=np.float64) * timestep
    if times.size == 0 or abs(float(times[-1]) - duration) > 1e-12:
        times = np.concatenate((times, np.asarray((duration,), dtype=np.float64)))
    else:
        times[-1] = duration
    knot_times = np.asarray((0.0, duration), dtype=np.float64)
    values = np.stack((start, end))
    velocities, accelerations = quintic_c2_knot_derivatives(knot_times, values)
    commands = np.empty((times.size, len(ACTIVE_ACTUATORS)), dtype=np.float64)
    for index, elapsed in enumerate(times):
        commands[index], _, _, _ = interpolate_quintic_c2(
            knot_times,
            values,
            float(elapsed),
            knot_velocities=velocities,
            knot_accelerations=accelerations,
        )
    applied = commands - start
    residuals = response.joint_pair_signed_residual + (
        response.joint_pair_jacobian_2x8 @ applied.T
    ).T
    angles = np.degrees(np.arctan(np.linalg.norm(residuals, axis=1)))
    band = build_joint_pair_octagonal_band(
        response.joint_pair_signed_residual,
        response.joint_pair_jacobian_2x8,
        maximum_angle_deg,
    )
    within_octagon = np.all(
        band.facet_normals_8x2 @ residuals.T
        <= band.facet_radius + 1e-12,
        axis=0,
    )
    within_exact = angles <= float(maximum_angle_deg) + 1e-12
    return ContinuousJointPairAudit(
        sample_times_s=times,
        command_delta_rad=commands,
        joint_pair_signed_residual=residuals,
        angle_to_cube_positive_y_deg=angles,
        within_octagonal_band=within_octagon,
        within_exact_angle_bound=within_exact,
        maximum_angle_deg=maximum_angle_deg,
        audit_timestep_s=timestep,
    )


audit_joint_pair_segment_interpolation = audit_joint_pair_interpolation


@dataclass(frozen=True, slots=True)
class JointPairStateSample:
    """One true-physics pair measurement during a segment rollout."""

    elapsed_s: float
    vector_cube_m: np.ndarray
    joint_pair_signed_residual: np.ndarray
    length_m: float
    angle_to_cube_positive_y_deg: float

    def __post_init__(self) -> None:
        elapsed = float(self.elapsed_s)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_s must be non-negative and finite")
        vector = _readonly_array(self.vector_cube_m, (3,), "vector_cube_m")
        residual = _readonly_array(
            self.joint_pair_signed_residual,
            (2,),
            "joint_pair_signed_residual",
        )
        if float(vector[1]) <= 0.0:
            raise ValueError("joint-pair rollout vector must point toward +Y")
        derived = np.asarray((vector[0] / vector[1], vector[2] / vector[1]))
        if not np.allclose(residual, derived, rtol=1e-10, atol=1e-12):
            raise ValueError("joint-pair rollout vector and residual disagree")
        length = _finite_positive(self.length_m, "length_m")
        if not math.isclose(
            length, float(np.linalg.norm(vector)), rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError("joint-pair rollout vector and length disagree")
        angle = float(self.angle_to_cube_positive_y_deg)
        expected_angle = joint_pair_angle_from_signed_residual_deg(residual)
        if not math.isfinite(angle) or not math.isclose(
            angle, expected_angle, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError("joint-pair rollout residual and angle disagree")
        object.__setattr__(self, "elapsed_s", elapsed)
        object.__setattr__(self, "vector_cube_m", vector)
        object.__setattr__(self, "joint_pair_signed_residual", residual)
        object.__setattr__(self, "length_m", length)
        object.__setattr__(self, "angle_to_cube_positive_y_deg", angle)

    @classmethod
    def from_geometry(
        cls, elapsed_s: float, geometry: Mapping[str, Any]
    ) -> "JointPairStateSample":
        return cls(
            elapsed_s=elapsed_s,
            vector_cube_m=geometry["vector_cube_m"],
            joint_pair_signed_residual=geometry["joint_pair_signed_residual"],
            length_m=geometry["length_m"],
            angle_to_cube_positive_y_deg=geometry[
                "angle_to_cube_positive_y_deg"
            ],
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "elapsed_s": self.elapsed_s,
            "vector_cube_m": self.vector_cube_m.tolist(),
            "joint_pair_signed_residual": self.joint_pair_signed_residual.tolist(),
            "length_m": self.length_m,
            "angle_to_cube_positive_y_deg": self.angle_to_cube_positive_y_deg,
            "points_toward_positive_cube_y": True,
        }


@dataclass(frozen=True, slots=True)
class JointPairSegmentRollout:
    """Real contact rollout plus millisecond directed-pair observations."""

    contact_rollout: SequentialSegmentRollout
    joint_pair_samples: tuple[JointPairStateSample, ...]
    minimum_separation_m: float = DEFAULT_MINIMUM_JOINT_PAIR_SEPARATION_M
    maximum_angle_deg: float = DEFAULT_CONSTRAINT_ANGLE_DEG

    def __post_init__(self) -> None:
        if not isinstance(self.contact_rollout, SequentialSegmentRollout):
            raise TypeError("contact_rollout must be a SequentialSegmentRollout")
        samples = tuple(self.joint_pair_samples)
        if len(samples) < 2 or not all(
            isinstance(value, JointPairStateSample) for value in samples
        ):
            raise ValueError("joint_pair_samples must contain a sampled segment")
        times = np.asarray([value.elapsed_s for value in samples])
        if abs(float(times[0])) > _EPSILON or np.any(np.diff(times) <= 0.0):
            raise ValueError("joint_pair_samples must start at zero and increase")
        minimum = _finite_positive(self.minimum_separation_m, "minimum_separation_m")
        maximum = _finite_positive(self.maximum_angle_deg, "maximum_angle_deg")
        object.__setattr__(self, "joint_pair_samples", samples)
        object.__setattr__(self, "minimum_separation_m", minimum)
        object.__setattr__(self, "maximum_angle_deg", maximum)

    @property
    def pair_safe(self) -> bool:
        return bool(
            all(value.length_m + _EPSILON >= self.minimum_separation_m for value in self.joint_pair_samples)
            and all(
                value.angle_to_cube_positive_y_deg <= self.maximum_angle_deg + _EPSILON
                for value in self.joint_pair_samples
            )
        )

    @property
    def contact_and_pair_safe(self) -> bool:
        return bool(self.contact_rollout.contact_safe and self.pair_safe)

    @property
    def maximum_observed_angle_deg(self) -> float:
        return max(
            value.angle_to_cube_positive_y_deg for value in self.joint_pair_samples
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            **self.contact_rollout.as_mapping(),
            "joint_pair_audit": {
                "true_physics": True,
                "sample_count": len(self.joint_pair_samples),
                "minimum_separation_m": self.minimum_separation_m,
                "maximum_angle_deg": self.maximum_angle_deg,
                "maximum_observed_angle_deg": self.maximum_observed_angle_deg,
                "pair_safe": self.pair_safe,
                "samples": [value.as_mapping() for value in self.joint_pair_samples],
            },
            "contact_and_pair_safe": self.contact_and_pair_safe,
        }


@dataclass(frozen=True, slots=True)
class JointPairSequentialPlanningHooks:
    """Injectable real-physics boundaries for schema-v15 planning."""

    collect_probes: Any
    rollout_segment: Any

    def __post_init__(self) -> None:
        if not callable(self.collect_probes) or not callable(self.rollout_segment):
            raise TypeError("joint-pair planning hooks must be callable")


@dataclass(frozen=True, slots=True)
class JointPairSequentialSegmentEvidence:
    segment_index: int
    start_checkpoint_step_index: int
    start_checkpoint_sha256: str
    probe_sha256: tuple[str, ...]
    probe_set_sha256: str
    response: JointPairProbeResponse
    octagonal_band: JointPairOctagonalBand
    interpolation_audit: ContinuousJointPairAudit
    solver: Mapping[str, Any]
    start_command_delta_rad: np.ndarray
    end_command_delta_rad: np.ndarray
    rollout: JointPairSegmentRollout

    def __post_init__(self) -> None:
        if not 0 <= int(self.segment_index) < PLAN_SEGMENT_COUNT:
            raise ValueError("segment_index is out of range")
        if len(self.probe_sha256) != 17:
            raise ValueError("each segment must bind exactly 17 probe hashes")
        for value in (
            self.start_checkpoint_sha256,
            self.probe_set_sha256,
            *self.probe_sha256,
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("segment evidence contains an invalid SHA-256")
        for name in ("start_command_delta_rad", "end_command_delta_rad"):
            object.__setattr__(
                self,
                name,
                _readonly_array(getattr(self, name), (8,), name),
            )
        object.__setattr__(self, "segment_index", int(self.segment_index))
        object.__setattr__(self, "start_checkpoint_step_index", int(self.start_checkpoint_step_index))
        object.__setattr__(self, "solver", copy.deepcopy(dict(self.solver)))

    @property
    def search_safe(self) -> bool:
        return bool(
            self.response.extended_response.zero_contact_safe
            and self.interpolation_audit.passed
            and self.rollout.contact_and_pair_safe
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "start_checkpoint_step_index": self.start_checkpoint_step_index,
            "start_checkpoint_sha256": self.start_checkpoint_sha256,
            "probe_count": len(self.probe_sha256),
            "probe_sha256": list(self.probe_sha256),
            "probe_set_sha256": self.probe_set_sha256,
            "response_model_id": self.response.response_model_id,
            "response": self.response.as_mapping(),
            "octagonal_band": self.octagonal_band.as_mapping(),
            "interpolation_audit": self.interpolation_audit.as_mapping(),
            "solver": copy.deepcopy(dict(self.solver)),
            "start_command_delta_rad": {
                name: float(self.start_command_delta_rad[index])
                for index, name in enumerate(ACTIVE_ACTUATORS)
            },
            "end_command_delta_rad": {
                name: float(self.end_command_delta_rad[index])
                for index, name in enumerate(ACTIVE_ACTUATORS)
            },
            "rollout": self.rollout.as_mapping(),
            "search_safe": self.search_safe,
        }


@dataclass(frozen=True, slots=True)
class JointPairSequentialPlanAttempt:
    attempt_index: int
    trust_region_scale: float
    plan: TwentyOneKnotContactPlan
    segments: tuple[JointPairSequentialSegmentEvidence, ...]
    predicted_joint_pair_signed_residual: np.ndarray
    joint_pair_residual_jacobian_2x8: np.ndarray = field(init=False)
    object_response_jacobian_6x8: np.ndarray = field(init=False)
    target_force_jacobian_3x8: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if len(segments) != PLAN_SEGMENT_COUNT:
            raise ValueError("a pair-constrained attempt must contain twenty segments")
        if tuple(value.segment_index for value in segments) != tuple(range(20)):
            raise ValueError("pair-constrained segment indices must be canonical")
        residual = _readonly_array(
            self.predicted_joint_pair_signed_residual,
            (KNOT_COUNT, 2),
            "predicted_joint_pair_signed_residual",
        )
        pair_jacobian = np.stack(
            [value.response.joint_pair_jacobian_2x8 for value in segments]
            + [segments[-1].response.joint_pair_jacobian_2x8]
        )
        object_jacobian = np.stack(
            [value.response.extended_response.object_jacobian_6x8 for value in segments]
            + [segments[-1].response.extended_response.object_jacobian_6x8]
        )
        force_jacobian = np.stack(
            [value.response.extended_response.force_jacobian_3x8 for value in segments]
            + [segments[-1].response.extended_response.force_jacobian_3x8]
        )
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "predicted_joint_pair_signed_residual", residual)
        object.__setattr__(
            self,
            "joint_pair_residual_jacobian_2x8",
            _readonly_array(pair_jacobian, (21, 2, 8), "joint_pair_residual_jacobian_2x8"),
        )
        object.__setattr__(
            self,
            "object_response_jacobian_6x8",
            _readonly_array(object_jacobian, (21, 6, 8), "object_response_jacobian_6x8"),
        )
        object.__setattr__(
            self,
            "target_force_jacobian_3x8",
            _readonly_array(force_jacobian, (21, 3, 8), "target_force_jacobian_3x8"),
        )

    @property
    def search_contact_and_pair_safe(self) -> bool:
        return bool(
            self.plan.contact_feasible and all(value.search_safe for value in self.segments)
        )

    @property
    def maximum_pair_angle_deg(self) -> float:
        return max(
            max(
                value.interpolation_audit.maximum_observed_angle_deg,
                value.rollout.maximum_observed_angle_deg,
            )
            for value in self.segments
        )

    def as_mapping(self) -> dict[str, Any]:
        payload = {
            "joint_pair_constrained_plan_attempt_schema_version": (
                JOINT_PAIR_CONSTRAINED_PLAN_SCHEMA_VERSION
            ),
            "attempt_index": int(self.attempt_index),
            "trust_region_scale": float(self.trust_region_scale),
            "segment_count": len(self.segments),
            "probe_count": 17 * len(self.segments),
            "search_contact_and_pair_safe": self.search_contact_and_pair_safe,
            "maximum_pair_angle_deg": self.maximum_pair_angle_deg,
            "plan": self.plan.as_mapping(),
            "predicted_joint_pair_signed_residual": (
                self.predicted_joint_pair_signed_residual.tolist()
            ),
            "joint_pair_residual_jacobian_2x8": (
                self.joint_pair_residual_jacobian_2x8.tolist()
            ),
            "object_response_jacobian_6x8": (
                self.object_response_jacobian_6x8.tolist()
            ),
            "target_force_jacobian_3x8": (
                self.target_force_jacobian_3x8.tolist()
            ),
            "segments": [value.as_mapping() for value in self.segments],
        }
        payload["attempt_report_id"] = canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class JointPairSequentialPlanningReport:
    initial_plan_id: str
    initial_checkpoint_sha256: str
    endpoint_bounds: Mapping[str, tuple[float, float]]
    contact_settings: ContactConstrainedPlannerSettings
    joint_pair_settings: JointPairPlannerSettings
    attempts: tuple[JointPairSequentialPlanAttempt, ...]
    selected_attempt_index: int

    def __post_init__(self) -> None:
        attempts = tuple(self.attempts)
        if len(attempts) != 4:
            raise ValueError("pair-constrained planning must return exactly four plans")
        if not 0 <= int(self.selected_attempt_index) < 4:
            raise ValueError("selected_attempt_index is out of range")
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "selected_attempt_index", int(self.selected_attempt_index))
        object.__setattr__(self, "endpoint_bounds", copy.deepcopy(dict(self.endpoint_bounds)))

    @property
    def selected_attempt(self) -> JointPairSequentialPlanAttempt:
        return self.attempts[self.selected_attempt_index]

    @property
    def selected_plan(self) -> TwentyOneKnotContactPlan:
        return self.selected_attempt.plan

    def as_mapping(self) -> dict[str, Any]:
        payload = {
            "joint_pair_constrained_sequential_planning_report_schema_version": (
                JOINT_PAIR_CONSTRAINED_PLANNER_SCHEMA_VERSION
            ),
            "search_evidence_only": True,
            "checkpoint_branches_are_not_success_evidence": True,
            "final_success_requires_full_reset_rerun": True,
            "initial_plan_id": self.initial_plan_id,
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "endpoint_bounds": {
                name: [float(self.endpoint_bounds[name][0]), float(self.endpoint_bounds[name][1])]
                for name in ACTIVE_ACTUATORS
            },
            "contact_settings": self.contact_settings.as_mapping(),
            "joint_pair_settings": self.joint_pair_settings.as_mapping(),
            "plan_count": len(self.attempts),
            "segment_count_per_plan": PLAN_SEGMENT_COUNT,
            "probe_count_per_segment": 17,
            "total_probe_count": 4 * PLAN_SEGMENT_COUNT * 17,
            "continuous_interpolation_audit": True,
            "continuous_interpolation_audit_timestep_s": (
                self.joint_pair_settings.audit_timestep_s
            ),
            "attempts": [value.as_mapping() for value in self.attempts],
            "selected_attempt_index": self.selected_attempt_index,
            "selected_plan_id": self.selected_plan.plan_id,
        }
        payload["report_id"] = canonical_sha256(payload)
        return payload


def _normalized_endpoint_bounds(
    bounds: Mapping[str, Sequence[float]],
) -> dict[str, tuple[float, float]]:
    if set(bounds) != set(ACTIVE_ACTUATORS):
        raise ValueError("endpoint_bounds must name exactly eight active actuators")
    normalized: dict[str, tuple[float, float]] = {}
    for name in ACTIVE_ACTUATORS:
        pair = tuple(float(value) for value in bounds[name])
        if len(pair) != 2 or not np.isfinite(pair).all() or pair[0] > pair[1]:
            raise ValueError(f"invalid endpoint bound for {name}")
        if not pair[0] - _EPSILON <= 0.0 <= pair[1] + _EPSILON:
            raise ValueError("endpoint bounds must contain the zero first waypoint")
        normalized[name] = pair
    return normalized


def _pair_plan_arrays(
    plan_config: Mapping[str, Any],
) -> tuple[str, float, str, float, np.ndarray, np.ndarray, np.ndarray]:
    """Read common v1/v2 plan fields without routing v15 through v14 parsing."""

    required = {
        "plan_id",
        "profile",
        "duration_s",
        "knot_times_s",
        "actuator_waypoints_rad",
        "desired_cube_position_delta_m",
        "desired_cube_rotation_vector_rad",
        "max_knot_delta_rad",
    }
    if not required.issubset(plan_config):
        raise ValueError("initial manipulation plan is missing required fields")
    profile = str(plan_config["profile"])
    if profile != "piecewise_quintic_minimum_jerk":
        raise ValueError("unsupported manipulation plan profile")
    duration = _finite_positive(plan_config["duration_s"], "duration_s")
    max_delta = _finite_positive(
        plan_config["max_knot_delta_rad"], "max_knot_delta_rad"
    )
    times = _readonly_array(plan_config["knot_times_s"], (21,), "knot_times_s")
    if (
        abs(float(times[0])) > _EPSILON
        or abs(float(times[-1]) - duration) > _EPSILON
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("initial plan must contain 21 increasing knots over duration_s")
    raw_commands = plan_config["actuator_waypoints_rad"]
    if not isinstance(raw_commands, Mapping) or set(raw_commands) != set(ACTIVE_ACTUATORS):
        raise ValueError("initial plan must name exactly eight active actuators")
    commands = _readonly_array(
        [raw_commands[name] for name in ACTIVE_ACTUATORS],
        (8, 21),
        "actuator_waypoints_rad",
    ).T.copy()
    desired_position = _readonly_array(
        plan_config["desired_cube_position_delta_m"],
        (21, 3),
        "desired_cube_position_delta_m",
    )
    desired_rotation = _readonly_array(
        plan_config["desired_cube_rotation_vector_rad"],
        (21, 3),
        "desired_cube_rotation_vector_rad",
    )
    desired = np.concatenate((desired_position, desired_rotation), axis=1)
    if not 0.010 - _EPSILON <= float(desired[-1, 2]) <= 0.012 + _EPSILON:
        raise ValueError("initial plan must target a 10--12 mm total lift")
    if np.any(np.abs(np.diff(commands, axis=0)) > max_delta + _EPSILON):
        raise ValueError("initial plan adjacent commands exceed max_knot_delta_rad")
    return (
        str(plan_config["plan_id"]),
        duration,
        profile,
        max_delta,
        times,
        commands,
        desired,
    )


def _canonical_joint_pair_probes(
    samples: Sequence[JointPairProbeSample | Mapping[str, Any]],
) -> tuple[tuple[JointPairProbeSample, ...], tuple[str, ...], str, JointPairProbeResponse]:
    response = fit_joint_pair_probe_response(samples)
    canonical = tuple(
        JointPairProbeSample.from_mapping(value)
        for value in response.probe_evidence
    )
    payloads = tuple(value.as_mapping() for value in canonical)
    hashes = tuple(canonical_sha256(value) for value in payloads)
    return canonical, hashes, canonical_sha256(payloads), response


def _solve_joint_pair_segment(
    response: JointPairProbeResponse,
    current_command: np.ndarray,
    seed_next_command: np.ndarray,
    desired_segment_response: np.ndarray,
    bounds: Mapping[str, tuple[float, float]],
    contact_settings: ContactConstrainedPlannerSettings,
    pair_settings: JointPairPlannerSettings,
    scale: float,
):
    extended = response.extended_response
    absolute_lower = np.asarray([bounds[name][0] for name in ACTIVE_ACTUATORS])
    absolute_upper = np.asarray([bounds[name][1] for name in ACTIVE_ACTUATORS])
    radius = np.minimum(
        np.asarray(contact_settings.trust_radius_rad) * float(scale),
        contact_settings.max_knot_delta_rad,
    )
    lower = np.maximum(absolute_lower - current_command, -radius)
    upper = np.minimum(absolute_upper - current_command, radius)
    unavailable = ~extended.available_actuator_mask
    lower[unavailable] = 0.0
    upper[unavailable] = 0.0

    target_force = np.asarray(contact_settings.target_normal_force_n)
    minimum_force = np.asarray(contact_settings.minimum_normal_force_n)
    maximum_slip = np.asarray(contact_settings.maximum_tangent_slip_m)
    matrix = np.vstack(
        (
            extended.object_jacobian_6x8,
            extended.force_jacobian_3x8,
            extended.slip_jacobian_3x8,
            response.joint_pair_jacobian_2x8,
        )
    )
    target = np.concatenate(
        (
            desired_segment_response,
            target_force,
            np.zeros(3),
            np.zeros(2),
        )
    ) - np.concatenate(
        (
            extended.object_bias_6d,
            extended.force_bias_n,
            extended.slip_bias_m,
            response.joint_pair_signed_residual,
        )
    )
    weights = 1.0 / np.concatenate(
        (
            np.asarray(contact_settings.object_response_scale),
            np.asarray(contact_settings.force_response_scale_n),
            np.asarray(contact_settings.slip_response_scale_m),
            np.asarray(pair_settings.signed_residual_scale),
        )
    )

    pair_band = build_joint_pair_octagonal_band(
        response.joint_pair_signed_residual,
        response.joint_pair_jacobian_2x8,
        pair_settings.maximum_angle_deg,
    )
    band_matrix = np.vstack(
        (
            extended.force_jacobian_3x8,
            extended.slip_jacobian_3x8,
            pair_band.linear_matrix_8x8,
        )
    )
    band_lower = np.concatenate(
        (
            minimum_force - extended.force_bias_n,
            -extended.slip_bias_m,
            pair_band.linear_lower_8,
        )
    )
    band_upper = np.concatenate(
        (
            np.full(3, math.inf),
            maximum_slip - extended.slip_bias_m,
            pair_band.linear_upper_8,
        )
    )
    center = np.clip(seed_next_command - current_command, lower, upper)
    solved = solve_bounded_projected_least_squares(
        matrix,
        target,
        lower,
        upper,
        weights=weights,
        ridge=contact_settings.ridge,
        center=center,
        linear_matrix=band_matrix,
        linear_lower=band_lower,
        linear_upper=band_upper,
        max_iterations=contact_settings.solver_max_iterations,
        tolerance=contact_settings.solver_tolerance,
    )
    return solved, pair_band


def _pair_attempt_rank(attempt: JointPairSequentialPlanAttempt) -> tuple[Any, ...]:
    plan = attempt.plan
    return (
        not attempt.search_contact_and_pair_safe,
        attempt.maximum_pair_angle_deg,
        plan.contact_loss_count,
        -float(np.min(plan.predicted_target_normal_force_n)),
        float(np.max(plan.predicted_tangent_slip_m, initial=0.0)),
        plan.path_rms_error,
        plan.terminal_path_error,
        int(attempt.attempt_index),
    )


def plan_joint_pair_constrained_sequential_trajectory(
    grasp: GraspPhysicsCheckpoint,
    initial_plan_config: Mapping[str, Any],
    endpoint_bounds: Mapping[str, Sequence[float]],
    *,
    contact_settings: ContactConstrainedPlannerSettings = ContactConstrainedPlannerSettings(),
    joint_pair_settings: JointPairPlannerSettings = JointPairPlannerSettings(),
    hooks: JointPairSequentialPlanningHooks | None = None,
) -> JointPairSequentialPlanningReport:
    """Generate four pair-constrained plans with 20 x 17 real probes each."""

    (
        initial_plan_id,
        duration,
        profile,
        max_knot_delta,
        times,
        seed_commands,
        desired,
    ) = _pair_plan_arrays(initial_plan_config)
    bounds = _normalized_endpoint_bounds(endpoint_bounds)
    if abs(float(contact_settings.duration_s) - duration) > _EPSILON:
        raise ValueError("planner settings duration must match the initial plan")
    if abs(float(contact_settings.max_knot_delta_rad) - max_knot_delta) > _EPSILON:
        raise ValueError("planner max_knot_delta_rad must match the initial plan")
    active_hooks = hooks
    if active_hooks is None:
        active_hooks = MuJoCoJointPairSequentialPlanningHooks(
            grasp,
            contact_settings,
            joint_pair_settings,
        ).as_hooks()

    attempts: list[JointPairSequentialPlanAttempt] = []
    for attempt_index, scale in enumerate(contact_settings.backoff_scales):
        current_grasp = grasp
        commands = np.zeros((KNOT_COUNT, len(ACTIVE_ACTUATORS)), dtype=np.float64)
        commands[0] = seed_commands[0]
        actual_object = np.zeros((KNOT_COUNT, 6), dtype=np.float64)
        actual_force = np.zeros((KNOT_COUNT, len(ACTIVE_FINGERS)), dtype=np.float64)
        actual_slip = np.zeros_like(actual_force)
        actual_valid = np.zeros_like(actual_force, dtype=bool)
        predicted_pair = np.zeros((KNOT_COUNT, 2), dtype=np.float64)
        objectives = np.zeros(KNOT_COUNT)
        violations = np.zeros(KNOT_COUNT)
        converged = np.ones(KNOT_COUNT, dtype=bool)
        segments: list[JointPairSequentialSegmentEvidence] = []
        for segment in range(PLAN_SEGMENT_COUNT):
            raw_samples = tuple(active_hooks.collect_probes(current_grasp, segment))
            (
                _canonical_samples,
                probe_hashes,
                probe_set_hash,
                response,
            ) = _canonical_joint_pair_probes(raw_samples)
            extended = response.extended_response
            if segment == 0:
                actual_force[0] = extended.force_bias_n
                actual_slip[0] = extended.slip_bias_m
                actual_valid[0] = (
                    extended.zero_contact_valid & extended.zero_contact_safe
                )
                predicted_pair[0] = response.joint_pair_signed_residual
            solved, pair_band = _solve_joint_pair_segment(
                response,
                commands[segment],
                seed_commands[segment + 1],
                desired[segment + 1] - desired[segment],
                bounds,
                contact_settings,
                joint_pair_settings,
                float(scale),
            )
            commands[segment + 1] = commands[segment] + solved.solution
            duration_s = float(times[segment + 1] - times[segment])
            audit = audit_joint_pair_interpolation(
                commands[segment],
                commands[segment + 1],
                duration_s,
                response,
                maximum_angle_deg=joint_pair_settings.maximum_angle_deg,
                audit_timestep_s=joint_pair_settings.audit_timestep_s,
            )
            rollout = active_hooks.rollout_segment(
                current_grasp,
                segment,
                commands[segment].copy(),
                commands[segment + 1].copy(),
                duration_s,
            )
            if not isinstance(rollout, JointPairSegmentRollout):
                raise TypeError(
                    "joint-pair rollout hook must return JointPairSegmentRollout"
                )
            actual_object[segment + 1] = (
                rollout.contact_rollout.cumulative_object_response_6d
            )
            actual_force[segment + 1] = rollout.contact_rollout.target_normal_force_n
            actual_slip[segment + 1] = rollout.contact_rollout.tangent_slip_m
            actual_valid[segment + 1] = (
                rollout.contact_rollout.target_contact_valid
                & rollout.contact_and_pair_safe
            )
            predicted_pair[segment + 1] = (
                response.joint_pair_signed_residual
                + response.joint_pair_jacobian_2x8 @ solved.solution
            )
            objectives[segment + 1] = solved.objective
            violations[segment + 1] = solved.max_linear_violation
            converged[segment + 1] = bool(
                extended.zero_contact_safe
                and solved.converged
                and audit.passed
                and rollout.contact_and_pair_safe
            )
            segments.append(
                JointPairSequentialSegmentEvidence(
                    segment_index=segment,
                    start_checkpoint_step_index=current_grasp.checkpoint.step_index,
                    start_checkpoint_sha256=_checkpoint_sha256(current_grasp),
                    probe_sha256=probe_hashes,
                    probe_set_sha256=probe_set_hash,
                    response=response,
                    octagonal_band=pair_band,
                    interpolation_audit=audit,
                    solver=solved.as_mapping(ACTIVE_ACTUATORS),
                    start_command_delta_rad=commands[segment],
                    end_command_delta_rad=commands[segment + 1],
                    rollout=rollout,
                )
            )
            current_grasp = rollout.contact_rollout.next_grasp

        contact_plan = TwentyOneKnotContactPlan(
            trust_region_scale=float(scale),
            duration_s=duration,
            profile=profile,
            max_knot_delta_rad=max_knot_delta,
            trust_region_backtracks=4,
            knot_fraction=times / float(times[-1]),
            path_progress=_path_progress(desired),
            desired_object_response_6d=desired,
            command_delta_rad=commands,
            predicted_object_response_6d=actual_object,
            predicted_target_normal_force_n=np.maximum(0.0, actual_force),
            predicted_tangent_slip_m=np.maximum(0.0, actual_slip),
            predicted_contact_valid=actual_valid,
            solver_objective=objectives,
            solver_max_linear_violation=violations,
            solver_converged=converged,
        )
        attempts.append(
            JointPairSequentialPlanAttempt(
                attempt_index=attempt_index,
                trust_region_scale=float(scale),
                plan=contact_plan,
                segments=tuple(segments),
                predicted_joint_pair_signed_residual=predicted_pair,
            )
        )

    selected = min(range(4), key=lambda index: _pair_attempt_rank(attempts[index]))
    return JointPairSequentialPlanningReport(
        initial_plan_id=initial_plan_id,
        initial_checkpoint_sha256=_checkpoint_sha256(grasp),
        endpoint_bounds=bounds,
        contact_settings=contact_settings,
        joint_pair_settings=joint_pair_settings,
        attempts=tuple(attempts),
        selected_attempt_index=selected,
    )


# Alternate word order kept as a discoverable public alias.
plan_sequential_joint_pair_constrained_trajectory = (
    plan_joint_pair_constrained_sequential_trajectory
)


def _v2_plan_payload(attempt: JointPairSequentialPlanAttempt) -> dict[str, Any]:
    payload = attempt.plan.as_manipulation_plan_config()
    payload.pop("plan_id", None)
    payload["schema_version"] = 2
    payload["joint_pair_residual_jacobian_2x8"] = (
        attempt.joint_pair_residual_jacobian_2x8.tolist()
    )
    payload["object_response_jacobian_6x8"] = (
        attempt.object_response_jacobian_6x8.tolist()
    )
    payload["target_force_jacobian_3x8"] = (
        attempt.target_force_jacobian_3x8.tolist()
    )
    # ManipulationPlanParameters v2 uses the same canonical identity rule as
    # v1: every immutable plan field except plan_id participates in the hash.
    resolved = {"plan_id": canonical_sha256(payload), **payload}
    # Bind the materializer to the public schema-v2 parser as an integration
    # assertion; ``as_config`` also normalizes every scalar to JSON-safe types.
    return ManipulationPlanParameters.from_config(resolved).as_config()


def materialize_joint_pair_constrained_plan_config(
    base_config: Mapping[str, Any],
    report: JointPairSequentialPlanningReport,
    *,
    attempt_index: int | None = None,
    validate: bool = False,
) -> dict[str, Any]:
    """Materialize a v2 plan and its three 21-node runtime Jacobians."""

    index = report.selected_attempt_index if attempt_index is None else int(attempt_index)
    if not 0 <= index < 4:
        raise ValueError("attempt_index must select one of the four plans")
    attempt = report.attempts[index]
    # Reuse the frozen v14 materializer only for the unchanged command/desired
    # path fields, then replace that v1 mapping with the additive v2 identity.
    resolved = materialize_contact_plan_config(
        base_config,
        attempt.plan,
        validate=False,
    )
    resolved["manipulation_plan"] = _v2_plan_payload(attempt)
    resolved["control"]["manipulation_delta_rad"] = {
        name: float(attempt.plan.command_delta_rad[-1, actuator_index])
        for actuator_index, name in enumerate(ACTIVE_ACTUATORS)
    }
    metadata = resolved.setdefault("candidate_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("candidate_metadata must be a mapping")
    metadata["joint_pair_constrained_sequential_planning"] = {
        "schema_version": JOINT_PAIR_CONSTRAINED_PLANNER_SCHEMA_VERSION,
        "report_id": report.as_mapping()["report_id"],
        "attempt_index": index,
        "attempt_report_id": attempt.as_mapping()["attempt_report_id"],
        "response_node_count": KNOT_COUNT,
        "actuator_order": list(ACTIVE_ACTUATORS),
        "search_evidence_only": True,
        "final_success_requires_full_reset_rerun": True,
    }
    if int(resolved.get("schema_version", -1)) == 15:
        from ..v15_identity import install_v15_top_level_identities

        install_v15_top_level_identities(resolved)
    if validate:
        # Import lazily so this module remains usable while the v15 schema is
        # assembled independently of the frozen v14 code.
        from ..config import validate_config

        validate_config(resolved)
    return resolved


def materialize_all_joint_pair_constrained_plan_configs(
    base_config: Mapping[str, Any],
    report: JointPairSequentialPlanningReport,
    *,
    validate: bool = False,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        materialize_joint_pair_constrained_plan_config(
            base_config,
            report,
            attempt_index=index,
            validate=validate,
        )
        for index in range(4)
    )


class MuJoCoJointPairSequentialPlanningHooks(MuJoCoSequentialPlanningHooks):
    """Production 17-probe/rollout hooks with true 1 ms pair observations."""

    def __init__(
        self,
        initial_grasp: GraspPhysicsCheckpoint,
        contact_settings: ContactConstrainedPlannerSettings,
        joint_pair_settings: JointPairPlannerSettings = JointPairPlannerSettings(),
        *,
        probe_duration_s: float | None = None,
        probe_epsilon_rad: float = 0.02,
    ) -> None:
        super().__init__(
            initial_grasp,
            contact_settings,
            probe_duration_s=probe_duration_s,
            probe_epsilon_rad=probe_epsilon_rad,
        )
        self.joint_pair_settings = joint_pair_settings
        binding = resolve_joint_pair(self.model, joint_pair_settings.joint_names)
        if binding is None:  # pragma: no cover - names above are never None
            raise ValueError("joint-pair planner requires two joint names")
        self.joint_pair_binding: JointPairBinding = binding

    def as_hooks(self) -> JointPairSequentialPlanningHooks:
        return JointPairSequentialPlanningHooks(
            self.collect_joint_pair_probes,
            self.rollout_joint_pair_segment,
        )

    def _measure_pair(
        self, data: mujoco.MjData, elapsed_s: float
    ) -> JointPairStateSample:
        binding = self.joint_pair_binding
        geometry = measure_oriented_joint_pair_geometry(
            data.xanchor[binding.first_joint_id],
            data.xanchor[binding.second_joint_id],
            np.asarray(data.xmat[binding.cube_body_id], dtype=np.float64).reshape(3, 3),
            minimum_separation_m=self.joint_pair_settings.minimum_separation_m,
        )
        return JointPairStateSample.from_geometry(elapsed_s, geometry)

    def _run_joint_pair_branch(
        self,
        grasp: GraspPhysicsCheckpoint,
        start_delta: np.ndarray,
        end_delta: np.ndarray,
        duration_s: float,
    ) -> tuple[mujoco.MjData, dict[str, Any]]:
        data = mujoco.MjData(self.model)
        restore_physics_checkpoint(self.model, data, grasp.checkpoint)
        mujoco.mj_forward(self.model, data)
        start_position = data.xpos[grasp.cube_body_id].copy()
        start_quaternion = data.xquat[grasp.cube_body_id].copy()
        snapshot = contact_snapshot(self.model, data, self.info, classify_faces=True)
        assert snapshot.distal_face_force_n is not None
        assert snapshot.distal_face_position_moment_n_m is not None
        initial_centroid, initial_valid = target_face_contact_centroids(
            snapshot.distal_face_force_n,
            snapshot.distal_face_position_moment_n_m,
            self.target_faces,
        )

        timestep = float(self.model.opt.timestep)
        steps = int(round(float(duration_s) / timestep))
        if steps <= 0 or abs(steps * timestep - duration_s) > 0.5 * timestep + _EPSILON:
            raise ValueError("segment/probe duration does not align with model timestep")
        audit_stride = int(
            round(self.joint_pair_settings.audit_timestep_s / timestep)
        )
        if (
            audit_stride <= 0
            or abs(
                audit_stride * timestep
                - self.joint_pair_settings.audit_timestep_s
            )
            > 1e-12
        ):
            raise ValueError(
                "model timestep must divide the joint-pair audit timestep"
            )

        interpolation_times = np.asarray((0.0, duration_s), dtype=np.float64)
        interpolation_values = np.stack((start_delta, end_delta))
        interpolation_velocities, interpolation_accelerations = (
            quintic_c2_knot_derivatives(
                interpolation_times, interpolation_values
            )
        )
        forces: list[np.ndarray] = []
        validity: list[np.ndarray] = []
        slips: list[np.ndarray] = []
        pair_samples = [self._measure_pair(data, 0.0)]
        forbidden = False
        nondistal = False
        for step in range(steps):
            elapsed = (step + 1) * timestep
            delta, _, _, _ = interpolate_quintic_c2(
                interpolation_times,
                interpolation_values,
                elapsed,
                knot_velocities=interpolation_velocities,
                knot_accelerations=interpolation_accelerations,
            )
            data.ctrl[:] = self.base_preload
            for index, name in enumerate(ACTIVE_ACTUATORS):
                data.ctrl[self.model.actuator(name).id] += delta[index]
            mujoco.mj_step(self.model, data)
            mujoco.mj_forward(self.model, data)
            force, valid, slip, bad, nonterminal = self._observe_window(
                data, initial_centroid, initial_valid
            )
            if step >= steps // 2:
                forces.append(force)
                validity.append(valid)
                slips.append(slip)
            forbidden = forbidden or bad
            nondistal = nondistal or nonterminal
            if (step + 1) % audit_stride == 0 or step + 1 == steps:
                pair_samples.append(self._measure_pair(data, elapsed))
        evidence = {
            "steps": steps,
            "segment_response_6d": np.concatenate(
                (
                    data.xpos[grasp.cube_body_id] - start_position,
                    _quat_rotation_vector_wxyz(
                        start_quaternion, data.xquat[grasp.cube_body_id]
                    ),
                )
            ),
            "cumulative_response_6d": np.concatenate(
                (
                    data.xpos[grasp.cube_body_id] - self.initial_position,
                    _quat_rotation_vector_wxyz(
                        self.initial_quaternion,
                        data.xquat[grasp.cube_body_id],
                    ),
                )
            ),
            "force": np.min(np.asarray(forces), axis=0),
            "valid": np.all(np.asarray(validity), axis=0),
            "slip": np.max(np.asarray(slips), axis=0),
            "forbidden": forbidden,
            "nondistal": nondistal,
            "joint_pair_samples": tuple(pair_samples),
        }
        return data, evidence

    def _joint_pair_probe(
        self,
        grasp: GraspPhysicsCheckpoint,
        specification: ProbeSpecification,
    ) -> JointPairProbeSample:
        current = np.asarray(
            [
                float(
                    grasp.config["control"]["contact_preload_targets_rad"][name]
                )
                - float(self.base_preload[self.model.actuator(name).id])
                for name in ACTIVE_ACTUATORS
            ],
            dtype=np.float64,
        )
        applied = np.asarray(specification.applied_delta_rad, dtype=np.float64)
        _, evidence = self._run_joint_pair_branch(
            grasp,
            current,
            current + applied,
            self.probe_duration_s,
        )
        terminal = evidence["joint_pair_samples"][-1]
        extended = ExtendedProbeSample.from_mapping(
            {
                "probe": specification.as_mapping(),
                "checkpoint_step_index": grasp.checkpoint.step_index,
                "response_6d": evidence["segment_response_6d"].tolist(),
                "contact_evidence": {
                    "target_normal_force_n": evidence["force"].tolist(),
                    "target_contact_valid": evidence["valid"].tolist(),
                    "tangent_slip_m": evidence["slip"].tolist(),
                    "forbidden_contact": evidence["forbidden"],
                    "active_nondistal_contact": evidence["nondistal"],
                },
            }
        )
        return JointPairProbeSample(
            extended_sample=extended,
            joint_pair_signed_residual=terminal.joint_pair_signed_residual,
            joint_pair_vector_cube_m=terminal.vector_cube_m,
            joint_pair_length_m=terminal.length_m,
        )

    def collect_joint_pair_probes(
        self,
        grasp: GraspPhysicsCheckpoint,
        _segment_index: int,
    ) -> tuple[JointPairProbeSample, ...]:
        specifications = generate_probe_specifications(
            grasp.model,
            grasp.config,
            epsilon_rad=self.probe_epsilon_rad,
        )
        samples = tuple(
            self._joint_pair_probe(grasp, specification)
            for specification in specifications
        )
        if len(samples) != 17:
            raise AssertionError("canonical joint-pair probe schedule changed")
        return samples

    def rollout_joint_pair_segment(
        self,
        grasp: GraspPhysicsCheckpoint,
        _segment_index: int,
        start_delta: np.ndarray,
        end_delta: np.ndarray,
        duration_s: float,
    ) -> JointPairSegmentRollout:
        data, evidence = self._run_joint_pair_branch(
            grasp,
            start_delta,
            end_delta,
            duration_s,
        )
        next_grasp = self._wrap_checkpoint(
            data,
            step_index=grasp.checkpoint.step_index + int(evidence["steps"]),
            command_delta=end_delta,
        )
        contact_rollout = SequentialSegmentRollout(
            next_grasp=next_grasp,
            segment_object_response_6d=evidence["segment_response_6d"],
            cumulative_object_response_6d=evidence["cumulative_response_6d"],
            target_normal_force_n=evidence["force"],
            target_contact_valid=evidence["valid"],
            tangent_slip_m=evidence["slip"],
            forbidden_contact=bool(evidence["forbidden"]),
            active_nondistal_contact=bool(evidence["nondistal"]),
            physics_steps=int(evidence["steps"]),
        )
        return JointPairSegmentRollout(
            contact_rollout=contact_rollout,
            joint_pair_samples=evidence["joint_pair_samples"],
            minimum_separation_m=self.joint_pair_settings.minimum_separation_m,
            maximum_angle_deg=self.joint_pair_settings.maximum_angle_deg,
        )


__all__ = [
    "ContinuousJointPairAudit",
    "DEFAULT_AUDIT_TIMESTEP_S",
    "DEFAULT_CONSTRAINT_ANGLE_DEG",
    "DEFAULT_JOINT_PAIR",
    "JOINT_PAIR_CONSTRAINED_PLAN_SCHEMA_VERSION",
    "JOINT_PAIR_CONSTRAINED_PLANNER_SCHEMA_VERSION",
    "JOINT_PAIR_PROBE_RESPONSE_SCHEMA_VERSION",
    "JointPairOctagonalBand",
    "JointPairPlannerSettings",
    "JointPairProbeResponse",
    "JointPairProbeSample",
    "JointPairSegmentRollout",
    "JointPairSequentialPlanAttempt",
    "JointPairSequentialPlanningHooks",
    "JointPairSequentialPlanningReport",
    "JointPairSequentialSegmentEvidence",
    "JointPairStateSample",
    "MuJoCoJointPairSequentialPlanningHooks",
    "OCTAGON_SIDE_COUNT",
    "audit_joint_pair_interpolation",
    "audit_joint_pair_segment_interpolation",
    "build_joint_pair_octagonal_band",
    "fit_joint_pair_probe_response",
    "fit_joint_pair_response",
    "joint_pair_octagonal_linear_band",
    "materialize_all_joint_pair_constrained_plan_configs",
    "materialize_joint_pair_constrained_plan_config",
    "plan_joint_pair_constrained_sequential_trajectory",
    "plan_sequential_joint_pair_constrained_trajectory",
]
