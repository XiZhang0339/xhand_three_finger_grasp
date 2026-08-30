"""Offline contact-constrained manipulation planning helpers for schema v14.

The module is deliberately independent of the experiment registry, CLI and
runtime controller.  A future schema-v14 runner can use a verified
``GraspPhysicsCheckpoint`` to collect the canonical 17 local probes, fit a
small deterministic response model, and turn that model into four bounded
20-segment/21-knot command plans.  Published success must still come from a full reset;
checkpoint branches are search evidence only.

The continuous response has twelve rows in this fixed order::

    object translation xyz, object rotation-vector xyz,
    thumb/index/mid target-face normal force,
    thumb/index/mid tangent-slip magnitude

Contact validity, forbidden contact and active non-distal contact remain
discrete evidence.  Unsafe probe sides are never used for finite differences.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import ACTIVE_ACTUATORS, ACTIVE_FINGERS
from ..grasp_pose import (
    canonical_sha256,
    controller_context,
)
from ..v14_identity import (
    object_configuration_context,
    v14_grasp_object_pair_id,
    v14_grasp_pose_id,
    v14_object_config_id,
)
from ..trajectory import (
    interpolate_quintic_c2,
    quintic_c2_knot_derivatives,
)
from .actual_contact_manipulation import (
    GraspPhysicsCheckpoint,
    ProbeSpecification,
    generate_probe_specifications,
)


V14_PLANNER_SCHEMA_VERSION = 1
PLAN_SEGMENT_COUNT = 20
KNOT_COUNT = PLAN_SEGMENT_COUNT + 1
RESPONSE_DIMENSION = 12
TRUST_REGION_BACKOFF_SCALES = (1.0, 0.5, 0.25, 0.125)
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
        raise ValueError(f"{label} must contain only finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _domain_id(domain: str, payload: Any) -> str:
    return canonical_sha256(
        {
            "identity_schema_version": 1,
            "domain": str(domain),
            "payload": payload,
        }
    )


def _named_vector(values: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    return {str(name): float(values[index]) for index, name in enumerate(names)}


def _vector_from_named_or_sequence(
    values: object,
    names: Sequence[str],
    label: str,
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    if isinstance(values, Mapping):
        if set(values) != set(names):
            raise ValueError(f"{label} must name exactly {tuple(names)}")
        ordered = [values[name] for name in names]
    elif not isinstance(values, (str, bytes)):
        try:
            ordered = list(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(
                f"{label} must be a named mapping or sequence"
            ) from exc
    else:
        raise ValueError(f"{label} must be a named mapping or sequence")
    return _readonly_array(ordered, (len(names),), label, dtype=dtype)


@dataclass(frozen=True, slots=True)
class ExtendedProbeSample:
    """One checkpoint branch with object, force, topology and slip evidence.

    Runners should report the minimum operation-window target-face normal force
    and maximum tangent slip for each finger.  That conservative convention is
    persisted in the report; this helper never substitutes a terminal-only
    force sample for the supplied evidence.
    """

    kind: str
    actuator: str | None
    direction: int
    applied_delta_rad: np.ndarray
    object_response_6d: np.ndarray
    target_normal_force_n: np.ndarray
    target_contact_valid: np.ndarray
    tangent_slip_m: np.ndarray
    forbidden_contact: bool = False
    active_nondistal_contact: bool = False
    checkpoint_step_index: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("zero", "single_actuator"):
            raise ValueError("probe kind must be zero or single_actuator")
        if self.kind == "zero":
            if self.actuator is not None or int(self.direction) != 0:
                raise ValueError("zero probe cannot name an actuator or direction")
        else:
            if self.actuator not in ACTIVE_ACTUATORS:
                raise ValueError("single-actuator probe names an unknown actuator")
            if int(self.direction) not in (-1, 1):
                raise ValueError("single-actuator probe direction must be -1 or +1")
        delta = _readonly_array(
            self.applied_delta_rad,
            (len(ACTIVE_ACTUATORS),),
            "applied_delta_rad",
        )
        response = _readonly_array(
            self.object_response_6d, (6,), "object_response_6d"
        )
        force = _readonly_array(
            self.target_normal_force_n,
            (len(ACTIVE_FINGERS),),
            "target_normal_force_n",
        )
        valid = _readonly_array(
            self.target_contact_valid,
            (len(ACTIVE_FINGERS),),
            "target_contact_valid",
            dtype=bool,
        )
        slip = _readonly_array(
            self.tangent_slip_m,
            (len(ACTIVE_FINGERS),),
            "tangent_slip_m",
        )
        if np.any(force < 0.0):
            raise ValueError("target_normal_force_n must be non-negative")
        if np.any(slip < 0.0):
            raise ValueError("tangent_slip_m must be non-negative")
        if self.kind == "zero" and np.any(np.abs(delta) > _EPSILON):
            raise ValueError("zero probe must apply zero actuator delta")
        if self.kind == "single_actuator":
            column = ACTIVE_ACTUATORS.index(str(self.actuator))
            other = np.delete(delta, column)
            if np.any(np.abs(other) > _EPSILON):
                raise ValueError("single-actuator probe changed another actuator")
        checkpoint_step = self.checkpoint_step_index
        if checkpoint_step is not None and int(checkpoint_step) < 0:
            raise ValueError("checkpoint_step_index must be non-negative")
        object.__setattr__(self, "direction", int(self.direction))
        object.__setattr__(self, "applied_delta_rad", delta)
        object.__setattr__(self, "object_response_6d", response)
        object.__setattr__(self, "target_normal_force_n", force)
        object.__setattr__(self, "target_contact_valid", valid)
        object.__setattr__(self, "tangent_slip_m", slip)
        object.__setattr__(self, "forbidden_contact", bool(self.forbidden_contact))
        object.__setattr__(
            self, "active_nondistal_contact", bool(self.active_nondistal_contact)
        )
        if checkpoint_step is not None:
            object.__setattr__(self, "checkpoint_step_index", int(checkpoint_step))

    @property
    def contact_safe(self) -> bool:
        return bool(
            np.all(self.target_contact_valid)
            and not self.forbidden_contact
            and not self.active_nondistal_contact
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExtendedProbeSample":
        probe = value.get("probe")
        if not isinstance(probe, Mapping):
            raise ValueError("extended probe record must contain probe metadata")
        evidence = value.get("contact_evidence", value)
        if not isinstance(evidence, Mapping):
            raise ValueError("contact_evidence must be a mapping")
        return cls(
            kind=str(probe.get("kind")),
            actuator=(
                None if probe.get("actuator") is None else str(probe.get("actuator"))
            ),
            direction=int(probe.get("direction", 0)),
            applied_delta_rad=_vector_from_named_or_sequence(
                probe.get("applied_delta_rad"),
                ACTIVE_ACTUATORS,
                "probe.applied_delta_rad",
            ),
            object_response_6d=value.get("response_6d"),
            target_normal_force_n=_vector_from_named_or_sequence(
                evidence.get("target_normal_force_n"),
                ACTIVE_FINGERS,
                "contact_evidence.target_normal_force_n",
            ),
            target_contact_valid=_vector_from_named_or_sequence(
                evidence.get("target_contact_valid"),
                ACTIVE_FINGERS,
                "contact_evidence.target_contact_valid",
                dtype=bool,
            ),
            tangent_slip_m=_vector_from_named_or_sequence(
                evidence.get("tangent_slip_m"),
                ACTIVE_FINGERS,
                "contact_evidence.tangent_slip_m",
            ),
            forbidden_contact=bool(evidence.get("forbidden_contact", False)),
            active_nondistal_contact=bool(
                evidence.get("active_nondistal_contact", False)
            ),
            checkpoint_step_index=(
                None
                if value.get("checkpoint_step_index") is None
                else int(value["checkpoint_step_index"])
            ),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "probe": {
                "kind": self.kind,
                "actuator": self.actuator,
                "direction": self.direction,
                "applied_delta_rad": _named_vector(
                    self.applied_delta_rad, ACTIVE_ACTUATORS
                ),
            },
            "response_6d": self.object_response_6d.tolist(),
            "contact_evidence": {
                "target_normal_force_n": _named_vector(
                    self.target_normal_force_n, ACTIVE_FINGERS
                ),
                "target_contact_valid": {
                    finger: bool(self.target_contact_valid[index])
                    for index, finger in enumerate(ACTIVE_FINGERS)
                },
                "tangent_slip_m": _named_vector(
                    self.tangent_slip_m, ACTIVE_FINGERS
                ),
                "forbidden_contact": self.forbidden_contact,
                "active_nondistal_contact": self.active_nondistal_contact,
                "contact_safe": self.contact_safe,
            },
            "checkpoint_step_index": self.checkpoint_step_index,
        }


@dataclass(frozen=True, slots=True)
class ExtendedProbeResponse:
    """Central/one-sided local response model fitted from safe probes only."""

    object_bias_6d: np.ndarray
    force_bias_n: np.ndarray
    slip_bias_m: np.ndarray
    object_jacobian_6x8: np.ndarray
    force_jacobian_3x8: np.ndarray
    slip_jacobian_3x8: np.ndarray
    available_actuator_mask: np.ndarray
    zero_contact_valid: np.ndarray
    zero_forbidden_contact: bool
    zero_active_nondistal_contact: bool
    column_evidence: tuple[dict[str, Any], ...]
    probe_evidence: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        arrays = {
            "object_bias_6d": (self.object_bias_6d, (6,), np.float64),
            "force_bias_n": (
                self.force_bias_n,
                (len(ACTIVE_FINGERS),),
                np.float64,
            ),
            "slip_bias_m": (
                self.slip_bias_m,
                (len(ACTIVE_FINGERS),),
                np.float64,
            ),
            "object_jacobian_6x8": (
                self.object_jacobian_6x8,
                (6, len(ACTIVE_ACTUATORS)),
                np.float64,
            ),
            "force_jacobian_3x8": (
                self.force_jacobian_3x8,
                (len(ACTIVE_FINGERS), len(ACTIVE_ACTUATORS)),
                np.float64,
            ),
            "slip_jacobian_3x8": (
                self.slip_jacobian_3x8,
                (len(ACTIVE_FINGERS), len(ACTIVE_ACTUATORS)),
                np.float64,
            ),
            "available_actuator_mask": (
                self.available_actuator_mask,
                (len(ACTIVE_ACTUATORS),),
                bool,
            ),
            "zero_contact_valid": (
                self.zero_contact_valid,
                (len(ACTIVE_FINGERS),),
                bool,
            ),
        }
        for name, (value, shape, dtype) in arrays.items():
            object.__setattr__(
                self,
                name,
                _readonly_array(value, shape, name, dtype=dtype),
            )
        if np.any(self.force_bias_n < 0.0) or np.any(self.slip_bias_m < 0.0):
            raise ValueError("force/slip biases must be non-negative")
        if len(self.column_evidence) != len(ACTIVE_ACTUATORS):
            raise ValueError("column_evidence must contain eight entries")
        if len(self.probe_evidence) != 1 + 2 * len(ACTIVE_ACTUATORS):
            raise ValueError("probe_evidence must contain the canonical 17 probes")
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
        object.__setattr__(
            self, "zero_forbidden_contact", bool(self.zero_forbidden_contact)
        )
        object.__setattr__(
            self,
            "zero_active_nondistal_contact",
            bool(self.zero_active_nondistal_contact),
        )

    @property
    def zero_contact_safe(self) -> bool:
        return bool(
            np.all(self.zero_contact_valid)
            and not self.zero_forbidden_contact
            and not self.zero_active_nondistal_contact
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "extended_probe_response_schema_version": 1,
            "response_order": {
                "object_6d": ["x", "y", "z", "rx", "ry", "rz"],
                "target_normal_force_n": list(ACTIVE_FINGERS),
                "tangent_slip_m": list(ACTIVE_FINGERS),
            },
            "object_bias_6d": self.object_bias_6d.tolist(),
            "force_bias_n": _named_vector(self.force_bias_n, ACTIVE_FINGERS),
            "slip_bias_m": _named_vector(self.slip_bias_m, ACTIVE_FINGERS),
            "object_jacobian_6x8": self.object_jacobian_6x8.tolist(),
            "force_jacobian_3x8": self.force_jacobian_3x8.tolist(),
            "slip_jacobian_3x8": self.slip_jacobian_3x8.tolist(),
            "available_actuator_mask": {
                name: bool(self.available_actuator_mask[index])
                for index, name in enumerate(ACTIVE_ACTUATORS)
            },
            "zero_contact_valid": {
                finger: bool(self.zero_contact_valid[index])
                for index, finger in enumerate(ACTIVE_FINGERS)
            },
            "zero_forbidden_contact": self.zero_forbidden_contact,
            "zero_active_nondistal_contact": self.zero_active_nondistal_contact,
            "zero_contact_safe": self.zero_contact_safe,
            "column_evidence": [copy.deepcopy(value) for value in self.column_evidence],
            "probe_evidence": [copy.deepcopy(value) for value in self.probe_evidence],
        }

    @property
    def response_model_id(self) -> str:
        return _domain_id("v14_extended_probe_response", self._payload())

    def as_mapping(self) -> dict[str, Any]:
        result = self._payload()
        result["response_model_id"] = self.response_model_id
        return result

    def predict(self, actuator_delta_rad: Sequence[float]) -> dict[str, np.ndarray]:
        delta = np.asarray(actuator_delta_rad, dtype=np.float64)
        if delta.shape != (len(ACTIVE_ACTUATORS),) or not np.isfinite(delta).all():
            raise ValueError("actuator_delta_rad must contain eight finite values")
        return {
            "object_response_6d": self.object_bias_6d
            + self.object_jacobian_6x8 @ delta,
            "target_normal_force_n": self.force_bias_n
            + self.force_jacobian_3x8 @ delta,
            "tangent_slip_m": np.maximum(
                0.0, self.slip_bias_m + self.slip_jacobian_3x8 @ delta
            ),
        }


def collect_extended_probe_samples(
    grasp: GraspPhysicsCheckpoint,
    probe_runner: Callable[
        [GraspPhysicsCheckpoint, ProbeSpecification], Mapping[str, Any]
    ],
    *,
    epsilon_rad: float = 0.02,
) -> tuple[ExtendedProbeSample, ...]:
    """Collect the canonical 17 branches from one authenticated checkpoint.

    ``probe_runner`` owns MuJoCo integration and must independently restore the
    supplied checkpoint for every call.  If it reports a checkpoint index, it
    is checked against the source checkpoint before the sample is accepted.
    """

    specifications = generate_probe_specifications(
        grasp.model, grasp.config, epsilon_rad=epsilon_rad
    )
    samples: list[ExtendedProbeSample] = []
    for specification in specifications:
        raw = copy.deepcopy(dict(probe_runner(grasp, specification)))
        expected_probe = specification.as_mapping()
        if "probe" in raw and raw["probe"] != expected_probe:
            raise ValueError("probe_runner changed the requested probe metadata")
        raw["probe"] = expected_probe
        sample = ExtendedProbeSample.from_mapping(raw)
        if (
            sample.checkpoint_step_index is not None
            and sample.checkpoint_step_index != grasp.checkpoint.step_index
        ):
            raise ValueError("probe branch did not start from the supplied checkpoint")
        samples.append(sample)
    return tuple(samples)


def fit_extended_probe_response(
    probe_samples: Sequence[ExtendedProbeSample | Mapping[str, Any]],
) -> ExtendedProbeResponse:
    """Fit a 12x8 response, excluding every unsafe finite-difference side."""

    unsorted_samples = tuple(
        value
        if isinstance(value, ExtendedProbeSample)
        else ExtendedProbeSample.from_mapping(value)
        for value in probe_samples
    )
    if len(unsorted_samples) != 1 + 2 * len(ACTIVE_ACTUATORS):
        raise ValueError("extended response fitting requires exactly 17 probes")
    samples = tuple(
        sorted(
            unsorted_samples,
            key=lambda value: (
                -1
                if value.kind == "zero"
                else ACTIVE_ACTUATORS.index(str(value.actuator)),
                int(value.direction),
            ),
        )
    )
    zero_samples = [value for value in samples if value.kind == "zero"]
    if len(zero_samples) != 1:
        raise ValueError("extended response fitting requires one zero probe")
    zero = zero_samples[0]
    bias = np.concatenate(
        (zero.object_response_6d, zero.target_normal_force_n, zero.tangent_slip_m)
    )
    matrix = np.zeros(
        (RESPONSE_DIMENSION, len(ACTIVE_ACTUATORS)), dtype=np.float64
    )
    available = np.zeros(len(ACTIVE_ACTUATORS), dtype=bool)
    column_evidence: list[dict[str, Any]] = []
    for column, actuator in enumerate(ACTIVE_ACTUATORS):
        directional: dict[int, ExtendedProbeSample] = {}
        for sample in samples:
            if sample.kind == "single_actuator" and sample.actuator == actuator:
                if sample.direction in directional:
                    raise ValueError(f"duplicate probe direction for {actuator}")
                directional[sample.direction] = sample
        if set(directional) != {-1, 1}:
            raise ValueError(f"missing probe pair for {actuator}")
        safe = {
            direction: sample
            for direction, sample in directional.items()
            if sample.contact_safe
            and abs(float(sample.applied_delta_rad[column])) > _EPSILON
        }
        method = "unavailable_contact_unsafe"
        used_directions: tuple[int, ...] = ()
        if set(safe) == {-1, 1}:
            negative = safe[-1]
            positive = safe[1]
            denominator = float(
                positive.applied_delta_rad[column]
                - negative.applied_delta_rad[column]
            )
            if abs(denominator) > _EPSILON:
                negative_response = np.concatenate(
                    (
                        negative.object_response_6d,
                        negative.target_normal_force_n,
                        negative.tangent_slip_m,
                    )
                )
                positive_response = np.concatenate(
                    (
                        positive.object_response_6d,
                        positive.target_normal_force_n,
                        positive.tangent_slip_m,
                    )
                )
                matrix[:, column] = (
                    positive_response - negative_response
                ) / denominator
                method = "central_contact_safe"
                used_directions = (-1, 1)
                available[column] = True
        if not available[column] and safe:
            direction = 1 if 1 in safe else -1
            sample = safe[direction]
            step = float(sample.applied_delta_rad[column])
            response = np.concatenate(
                (
                    sample.object_response_6d,
                    sample.target_normal_force_n,
                    sample.tangent_slip_m,
                )
            )
            matrix[:, column] = (response - bias) / step
            method = (
                "forward_contact_safe" if direction == 1 else "backward_contact_safe"
            )
            used_directions = (direction,)
            available[column] = True
        column_evidence.append(
            {
                "actuator": actuator,
                "method": method,
                "available": bool(available[column]),
                "used_directions": list(used_directions),
                "negative_contact_safe": directional[-1].contact_safe,
                "positive_contact_safe": directional[1].contact_safe,
                "negative_step_rad": float(
                    directional[-1].applied_delta_rad[column]
                ),
                "positive_step_rad": float(
                    directional[1].applied_delta_rad[column]
                ),
            }
        )
    return ExtendedProbeResponse(
        object_bias_6d=bias[:6],
        force_bias_n=bias[6:9],
        slip_bias_m=bias[9:12],
        object_jacobian_6x8=matrix[:6],
        force_jacobian_3x8=matrix[6:9],
        slip_jacobian_3x8=matrix[9:12],
        available_actuator_mask=available,
        zero_contact_valid=zero.target_contact_valid,
        zero_forbidden_contact=zero.forbidden_contact,
        zero_active_nondistal_contact=zero.active_nondistal_contact,
        column_evidence=tuple(column_evidence),
        probe_evidence=tuple(value.as_mapping() for value in samples),
    )


@dataclass(frozen=True, slots=True)
class BoundedProjectedLeastSquaresResult:
    """Deterministic box-active-set plus linear-band projection result."""

    solution: np.ndarray
    active_lower: tuple[int, ...]
    active_upper: tuple[int, ...]
    iterations: int
    converged: bool
    linear_constraints_feasible: bool
    max_linear_violation: float
    objective: float
    weighted_residual_norm: float
    projected_gradient_inf_norm: float

    def __post_init__(self) -> None:
        solution = np.asarray(self.solution, dtype=np.float64)
        if solution.ndim != 1 or not np.isfinite(solution).all():
            raise ValueError("solution must be one finite vector")
        solution = solution.copy()
        solution.setflags(write=False)
        object.__setattr__(self, "solution", solution)
        object.__setattr__(self, "active_lower", tuple(int(v) for v in self.active_lower))
        object.__setattr__(self, "active_upper", tuple(int(v) for v in self.active_upper))
        object.__setattr__(self, "iterations", int(self.iterations))
        for name in (
            "max_linear_violation",
            "objective",
            "weighted_residual_norm",
            "projected_gradient_inf_norm",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))

    def as_mapping(self, names: Sequence[str] | None = None) -> dict[str, Any]:
        if names is not None and len(names) != self.solution.size:
            raise ValueError("names must match solution length")
        solution: Any = self.solution.tolist()
        if names is not None:
            solution = _named_vector(self.solution, names)
        return {
            "solver_schema_version": 1,
            "method": "cyclic_active_set_projected_weighted_least_squares",
            "solution": solution,
            "active_lower": list(self.active_lower),
            "active_upper": list(self.active_upper),
            "iterations": self.iterations,
            "converged": bool(self.converged),
            "linear_constraints_feasible": bool(
                self.linear_constraints_feasible
            ),
            "max_linear_violation": self.max_linear_violation,
            "objective": self.objective,
            "weighted_residual_norm": self.weighted_residual_norm,
            "projected_gradient_inf_norm": self.projected_gradient_inf_norm,
        }


def _linear_band_violation(
    matrix: np.ndarray,
    value: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    if matrix.shape[0] == 0:
        return 0.0
    projected = matrix @ value
    low_error = np.where(np.isfinite(lower), np.maximum(0.0, lower - projected), 0.0)
    high_error = np.where(np.isfinite(upper), np.maximum(0.0, projected - upper), 0.0)
    return float(max(np.max(low_error, initial=0.0), np.max(high_error, initial=0.0)))


def _project_linear_bands(
    value: np.ndarray,
    *,
    box_lower: np.ndarray,
    box_upper: np.ndarray,
    matrix: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float,
    max_cycles: int,
) -> tuple[np.ndarray, int, float]:
    result = np.clip(np.asarray(value, dtype=np.float64), box_lower, box_upper)
    if matrix.shape[0] == 0:
        return result, 0, 0.0
    cycles = 0
    for cycles in range(1, max_cycles + 1):
        for row_index, row in enumerate(matrix):
            observed = float(row @ result)
            target = observed
            if math.isfinite(float(lower[row_index])) and observed < lower[row_index]:
                target = float(lower[row_index])
            if math.isfinite(float(upper[row_index])) and observed > upper[row_index]:
                target = float(upper[row_index])
            residual = target - observed
            if abs(residual) <= tolerance:
                continue
            movable = np.ones(result.size, dtype=bool)
            if residual > 0.0:
                movable &= ~(
                    (result >= box_upper - tolerance) & (row > 0.0)
                )
                movable &= ~(
                    (result <= box_lower + tolerance) & (row < 0.0)
                )
            else:
                movable &= ~(
                    (result <= box_lower + tolerance) & (row > 0.0)
                )
                movable &= ~(
                    (result >= box_upper - tolerance) & (row < 0.0)
                )
            direction = np.where(movable, row, 0.0)
            denominator = float(direction @ direction)
            if denominator <= _EPSILON:
                continue
            result = np.clip(
                result + residual * direction / denominator,
                box_lower,
                box_upper,
            )
        violation = _linear_band_violation(matrix, result, lower, upper)
        if violation <= tolerance:
            return result, cycles, violation
    return result, cycles, _linear_band_violation(matrix, result, lower, upper)


def solve_bounded_projected_least_squares(
    matrix: Sequence[Sequence[float]],
    target: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    ridge: float = 1e-6,
    center: Sequence[float] | None = None,
    linear_matrix: Sequence[Sequence[float]] | None = None,
    linear_lower: Sequence[float] | None = None,
    linear_upper: Sequence[float] | None = None,
    max_iterations: int = 512,
    tolerance: float = 1e-10,
) -> BoundedProjectedLeastSquaresResult:
    """Solve weighted least squares with box and optional linear-band bounds.

    The quadratic box problem uses deterministic cyclic exact-coordinate
    updates.  Bound coordinates form the reported active set.  A cyclic
    half-space projection then enforces the contact force/slip bands, followed
    by projected-gradient refinements that never intentionally leave them.
    """

    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if a.ndim != 2 or b.shape != (a.shape[0],):
        raise ValueError("matrix and target shapes are incompatible")
    if low.shape != (a.shape[1],) or high.shape != low.shape:
        raise ValueError("box bounds must match the matrix columns")
    if (
        not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or np.any(low > high)
    ):
        raise ValueError("least-squares inputs and box bounds must be finite")
    if weights is None:
        weight = np.ones(a.shape[0], dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64)
        if weight.shape != (a.shape[0],):
            raise ValueError("weights must match the matrix rows")
        if not np.isfinite(weight).all() or np.any(weight < 0.0):
            raise ValueError("weights must be finite and non-negative")
    if not np.any(weight > 0.0):
        raise ValueError("at least one least-squares weight must be positive")
    regularization = _finite_float(ridge, "ridge")
    if regularization < 0.0:
        raise ValueError("ridge must be non-negative")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    tolerance = _finite_float(tolerance, "tolerance")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    anchor = (
        np.zeros(a.shape[1], dtype=np.float64)
        if center is None
        else np.asarray(center, dtype=np.float64)
    )
    if anchor.shape != low.shape or not np.isfinite(anchor).all():
        raise ValueError("center must match the matrix columns and be finite")
    weighted_a = weight[:, None] * a
    weighted_b = weight * b
    hessian = weighted_a.T @ weighted_a + regularization * np.eye(a.shape[1])
    offset = weighted_a.T @ weighted_b + regularization * anchor
    value = np.clip(anchor, low, high)
    iterations = 0
    coordinate_converged = False
    diagonal = np.diag(hessian)
    for iterations in range(1, max_iterations + 1):
        previous = value.copy()
        for column in range(value.size):
            if diagonal[column] <= _EPSILON:
                continue
            gradient = float(hessian[column] @ value - offset[column])
            value[column] = float(
                np.clip(
                    value[column] - gradient / diagonal[column],
                    low[column],
                    high[column],
                )
            )
        if float(np.max(np.abs(value - previous), initial=0.0)) <= tolerance:
            coordinate_converged = True
            break

    if linear_matrix is None:
        band_matrix = np.empty((0, value.size), dtype=np.float64)
        band_lower = np.empty(0, dtype=np.float64)
        band_upper = np.empty(0, dtype=np.float64)
    else:
        band_matrix = np.asarray(linear_matrix, dtype=np.float64)
        if band_matrix.ndim != 2 or band_matrix.shape[1] != value.size:
            raise ValueError("linear_matrix must match the solution width")
        if linear_lower is None or linear_upper is None:
            raise ValueError("linear band bounds are required with linear_matrix")
        band_lower = np.asarray(linear_lower, dtype=np.float64)
        band_upper = np.asarray(linear_upper, dtype=np.float64)
        if band_lower.shape != (band_matrix.shape[0],) or band_upper.shape != band_lower.shape:
            raise ValueError("linear band bounds must match linear_matrix rows")
        if (
            not np.isfinite(band_matrix).all()
            or np.isnan(band_lower).any()
            or np.isnan(band_upper).any()
            or np.any(band_lower > band_upper)
        ):
            raise ValueError("linear constraints are invalid")
    value, projection_cycles, violation = _project_linear_bands(
        value,
        box_lower=low,
        box_upper=high,
        matrix=band_matrix,
        lower=band_lower,
        upper=band_upper,
        tolerance=tolerance,
        max_cycles=max_iterations,
    )
    iterations += projection_cycles

    # Improve the objective while repeatedly projecting back into the linear
    # bands.  Monotone acceptance keeps the result deterministic and stable.
    lipschitz = max(float(np.linalg.eigvalsh(hessian)[-1]), _EPSILON)
    for projected_iteration in range(max_iterations):
        gradient = hessian @ value - offset
        candidate = np.clip(value - gradient / lipschitz, low, high)
        candidate, _, candidate_violation = _project_linear_bands(
            candidate,
            box_lower=low,
            box_upper=high,
            matrix=band_matrix,
            lower=band_lower,
            upper=band_upper,
            tolerance=tolerance,
            max_cycles=max_iterations,
        )
        old_objective = 0.5 * float(np.linalg.norm(weighted_a @ value - weighted_b) ** 2) + 0.5 * regularization * float(np.linalg.norm(value - anchor) ** 2)
        new_objective = 0.5 * float(np.linalg.norm(weighted_a @ candidate - weighted_b) ** 2) + 0.5 * regularization * float(np.linalg.norm(candidate - anchor) ** 2)
        if candidate_violation <= max(violation, tolerance) and new_objective <= old_objective + 1e-14:
            change = float(np.max(np.abs(candidate - value), initial=0.0))
            value = candidate
            violation = candidate_violation
            iterations += 1
            if change <= tolerance:
                break
        else:
            break

    residual = weighted_a @ value - weighted_b
    objective = 0.5 * float(residual @ residual) + 0.5 * regularization * float(
        np.linalg.norm(value - anchor) ** 2
    )
    gradient = hessian @ value - offset
    projected_gradient = value - np.clip(value - gradient, low, high)
    active_lower = tuple(
        int(index) for index in np.flatnonzero(value <= low + tolerance)
    )
    active_upper = tuple(
        int(index) for index in np.flatnonzero(value >= high - tolerance)
    )
    return BoundedProjectedLeastSquaresResult(
        solution=value,
        active_lower=active_lower,
        active_upper=active_upper,
        iterations=iterations,
        converged=bool(coordinate_converged and violation <= tolerance),
        linear_constraints_feasible=bool(violation <= tolerance),
        max_linear_violation=violation,
        objective=objective,
        weighted_residual_norm=float(np.linalg.norm(residual)),
        projected_gradient_inf_norm=float(
            np.max(np.abs(projected_gradient), initial=0.0)
        ),
    )


@dataclass(frozen=True, slots=True)
class ContactConstrainedPlannerSettings:
    """Versioned planner settings for exactly 20 segments and 21 knots."""

    knot_count: int = KNOT_COUNT
    duration_s: float = 3.0
    profile: str = "piecewise_quintic_minimum_jerk"
    max_knot_delta_rad: float = 0.04
    target_object_response_6d: tuple[float, ...] = (
        0.0,
        0.0,
        0.011,
        0.0,
        0.0,
        0.0,
    )
    target_normal_force_n: tuple[float, ...] = (0.20, 0.20, 0.20)
    minimum_normal_force_n: tuple[float, ...] = (0.05, 0.05, 0.05)
    maximum_tangent_slip_m: tuple[float, ...] = (0.005, 0.005, 0.005)
    object_response_scale: tuple[float, ...] = (
        0.002,
        0.002,
        0.011,
        math.radians(10.0),
        math.radians(10.0),
        math.radians(10.0),
    )
    force_response_scale_n: tuple[float, ...] = (0.10, 0.10, 0.10)
    slip_response_scale_m: tuple[float, ...] = (0.002, 0.002, 0.002)
    trust_radius_rad: tuple[float, ...] = (0.04,) * len(ACTIVE_ACTUATORS)
    backoff_scales: tuple[float, ...] = TRUST_REGION_BACKOFF_SCALES
    ridge: float = 1e-5
    solver_max_iterations: int = 512
    solver_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.knot_count != KNOT_COUNT:
            raise ValueError(
                "schema-v14 offline plans require exactly 20 segments / "
                f"{KNOT_COUNT} knots"
            )
        duration = _finite_float(self.duration_s, "duration_s")
        max_knot_delta = _finite_float(
            self.max_knot_delta_rad, "max_knot_delta_rad"
        )
        if duration <= 0.0 or max_knot_delta <= 0.0:
            raise ValueError("duration_s and max_knot_delta_rad must be positive")
        if self.profile != "piecewise_quintic_minimum_jerk":
            raise ValueError("unsupported offline manipulation plan profile")
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "max_knot_delta_rad", max_knot_delta)
        vectors = {
            "target_object_response_6d": (self.target_object_response_6d, 6, False),
            "target_normal_force_n": (
                self.target_normal_force_n,
                len(ACTIVE_FINGERS),
                True,
            ),
            "minimum_normal_force_n": (
                self.minimum_normal_force_n,
                len(ACTIVE_FINGERS),
                True,
            ),
            "maximum_tangent_slip_m": (
                self.maximum_tangent_slip_m,
                len(ACTIVE_FINGERS),
                True,
            ),
            "object_response_scale": (self.object_response_scale, 6, True),
            "force_response_scale_n": (
                self.force_response_scale_n,
                len(ACTIVE_FINGERS),
                True,
            ),
            "slip_response_scale_m": (
                self.slip_response_scale_m,
                len(ACTIVE_FINGERS),
                True,
            ),
            "trust_radius_rad": (
                self.trust_radius_rad,
                len(ACTIVE_ACTUATORS),
                True,
            ),
        }
        for name, (raw, length, positive) in vectors.items():
            values = tuple(float(value) for value in raw)
            if len(values) != length or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain {length} finite values")
            if positive and any(value <= 0.0 for value in values):
                raise ValueError(f"{name} must contain positive values")
            object.__setattr__(self, name, values)
        if any(
            target + _EPSILON < minimum
            for target, minimum in zip(
                self.target_normal_force_n, self.minimum_normal_force_n
            )
        ):
            raise ValueError("target normal force must not be below its hard floor")
        if any(
            value > self.max_knot_delta_rad + _EPSILON
            for value in self.trust_radius_rad
        ):
            raise ValueError(
                "trust_radius_rad cannot exceed max_knot_delta_rad"
            )
        backoffs = tuple(float(value) for value in self.backoff_scales)
        if (
            len(backoffs) != 4
            or not np.isfinite(backoffs).all()
            or any(value <= 0.0 for value in backoffs)
            or any(backoffs[index] <= backoffs[index + 1] for index in range(3))
        ):
            raise ValueError("backoff_scales must contain four strictly decreasing values")
        object.__setattr__(self, "backoff_scales", backoffs)
        ridge = _finite_float(self.ridge, "ridge")
        tolerance = _finite_float(self.solver_tolerance, "solver_tolerance")
        if ridge < 0.0 or tolerance <= 0.0:
            raise ValueError("ridge must be non-negative and tolerance positive")
        if not isinstance(self.solver_max_iterations, int) or self.solver_max_iterations <= 0:
            raise ValueError("solver_max_iterations must be a positive integer")
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "solver_tolerance", tolerance)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "contact_constrained_planner_settings_schema_version": 1,
            "knot_count": self.knot_count,
            "duration_s": self.duration_s,
            "profile": self.profile,
            "max_knot_delta_rad": self.max_knot_delta_rad,
            "target_object_response_6d": list(self.target_object_response_6d),
            "target_normal_force_n": _named_vector(
                np.asarray(self.target_normal_force_n), ACTIVE_FINGERS
            ),
            "minimum_normal_force_n": _named_vector(
                np.asarray(self.minimum_normal_force_n), ACTIVE_FINGERS
            ),
            "maximum_tangent_slip_m": _named_vector(
                np.asarray(self.maximum_tangent_slip_m), ACTIVE_FINGERS
            ),
            "object_response_scale": list(self.object_response_scale),
            "force_response_scale_n": list(self.force_response_scale_n),
            "slip_response_scale_m": list(self.slip_response_scale_m),
            "trust_radius_rad": _named_vector(
                np.asarray(self.trust_radius_rad), ACTIVE_ACTUATORS
            ),
            "backoff_scales": list(self.backoff_scales),
            "ridge": self.ridge,
            "solver_max_iterations": self.solver_max_iterations,
            "solver_tolerance": self.solver_tolerance,
        }


def _minimum_jerk(values: np.ndarray) -> np.ndarray:
    return 10.0 * values**3 - 15.0 * values**4 + 6.0 * values**5


@dataclass(frozen=True, slots=True)
class TwentyOneKnotContactPlan:
    """Immutable, JSON-serializable 20-segment/21-knot command plan."""

    trust_region_scale: float
    duration_s: float
    profile: str
    max_knot_delta_rad: float
    trust_region_backtracks: int
    knot_fraction: np.ndarray
    path_progress: np.ndarray
    desired_object_response_6d: np.ndarray
    command_delta_rad: np.ndarray
    predicted_object_response_6d: np.ndarray
    predicted_target_normal_force_n: np.ndarray
    predicted_tangent_slip_m: np.ndarray
    predicted_contact_valid: np.ndarray
    solver_objective: np.ndarray
    solver_max_linear_violation: np.ndarray
    solver_converged: np.ndarray

    def __post_init__(self) -> None:
        scale = _finite_float(self.trust_region_scale, "trust_region_scale")
        if scale <= 0.0:
            raise ValueError("trust_region_scale must be positive")
        object.__setattr__(self, "trust_region_scale", scale)
        duration = _finite_float(self.duration_s, "duration_s")
        max_knot_delta = _finite_float(
            self.max_knot_delta_rad, "max_knot_delta_rad"
        )
        if duration <= 0.0 or max_knot_delta <= 0.0:
            raise ValueError("duration_s and max_knot_delta_rad must be positive")
        if self.profile != "piecewise_quintic_minimum_jerk":
            raise ValueError("unsupported plan interpolation profile")
        if int(self.trust_region_backtracks) != 4:
            raise ValueError("schema-v14 plans require exactly four trust backtracks")
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "max_knot_delta_rad", max_knot_delta)
        object.__setattr__(self, "trust_region_backtracks", 4)
        arrays = {
            "knot_fraction": (self.knot_fraction, (KNOT_COUNT,), np.float64),
            "path_progress": (self.path_progress, (KNOT_COUNT,), np.float64),
            "desired_object_response_6d": (
                self.desired_object_response_6d,
                (KNOT_COUNT, 6),
                np.float64,
            ),
            "command_delta_rad": (
                self.command_delta_rad,
                (KNOT_COUNT, len(ACTIVE_ACTUATORS)),
                np.float64,
            ),
            "predicted_object_response_6d": (
                self.predicted_object_response_6d,
                (KNOT_COUNT, 6),
                np.float64,
            ),
            "predicted_target_normal_force_n": (
                self.predicted_target_normal_force_n,
                (KNOT_COUNT, len(ACTIVE_FINGERS)),
                np.float64,
            ),
            "predicted_tangent_slip_m": (
                self.predicted_tangent_slip_m,
                (KNOT_COUNT, len(ACTIVE_FINGERS)),
                np.float64,
            ),
            "predicted_contact_valid": (
                self.predicted_contact_valid,
                (KNOT_COUNT, len(ACTIVE_FINGERS)),
                bool,
            ),
            "solver_objective": (
                self.solver_objective,
                (KNOT_COUNT,),
                np.float64,
            ),
            "solver_max_linear_violation": (
                self.solver_max_linear_violation,
                (KNOT_COUNT,),
                np.float64,
            ),
            "solver_converged": (
                self.solver_converged,
                (KNOT_COUNT,),
                bool,
            ),
        }
        for name, (values, shape, dtype) in arrays.items():
            object.__setattr__(
                self,
                name,
                _readonly_array(values, shape, name, dtype=dtype),
            )
        if (
            abs(float(self.knot_fraction[0])) > _EPSILON
            or abs(float(self.knot_fraction[-1]) - 1.0) > _EPSILON
            or np.any(np.diff(self.knot_fraction) <= 0.0)
        ):
            raise ValueError("knot_fraction must increase strictly from zero to one")
        if (
            abs(float(self.path_progress[0])) > _EPSILON
            or abs(float(self.path_progress[-1]) - 1.0) > _EPSILON
            or np.any(np.diff(self.path_progress) < -_EPSILON)
        ):
            raise ValueError("path_progress must increase from zero to one")
        if np.any(self.predicted_target_normal_force_n < 0.0):
            raise ValueError("predicted target normal force must be non-negative")
        if np.any(self.predicted_tangent_slip_m < 0.0):
            raise ValueError("predicted tangent slip must be non-negative")
        if (
            np.max(np.abs(np.diff(self.command_delta_rad, axis=0)), initial=0.0)
            > self.max_knot_delta_rad + _EPSILON
        ):
            raise ValueError("adjacent actuator waypoints exceed max_knot_delta_rad")

    @property
    def contact_feasible(self) -> bool:
        return bool(
            np.all(self.predicted_contact_valid)
            and np.all(self.solver_converged)
            and np.max(self.solver_max_linear_violation, initial=0.0) <= 1e-8
        )

    @property
    def contact_loss_count(self) -> int:
        return int(np.count_nonzero(~self.predicted_contact_valid))

    @property
    def path_rms_error(self) -> float:
        return float(
            np.sqrt(
                np.mean(
                    (self.predicted_object_response_6d - self.desired_object_response_6d)
                    ** 2
                )
            )
        )

    @property
    def terminal_path_error(self) -> float:
        return float(
            np.linalg.norm(
                self.predicted_object_response_6d[-1]
                - self.desired_object_response_6d[-1]
            )
        )

    @property
    def plan_id(self) -> str:
        return canonical_sha256(self.manipulation_plan_identity_payload())

    def manipulation_plan_identity_payload(self) -> dict[str, Any]:
        """Return the exact payload consumed by ManipulationPlanParameters."""

        return {
            "schema_version": 1,
            "profile": self.profile,
            "duration_s": self.duration_s,
            "knot_times_s": (self.knot_fraction * self.duration_s).tolist(),
            "actuator_waypoints_rad": {
                actuator: self.command_delta_rad[:, index].tolist()
                for index, actuator in enumerate(ACTIVE_ACTUATORS)
            },
            "desired_cube_position_delta_m": self.desired_object_response_6d[
                :, :3
            ].tolist(),
            "desired_cube_rotation_vector_rad": self.desired_object_response_6d[
                :, 3:
            ].tolist(),
            "max_knot_delta_rad": self.max_knot_delta_rad,
            "trust_region_backtracks": self.trust_region_backtracks,
        }

    def as_manipulation_plan_config(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            **self.manipulation_plan_identity_payload(),
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "twenty_one_knot_contact_plan_schema_version": 1,
            "segment_count": PLAN_SEGMENT_COUNT,
            "knot_count": KNOT_COUNT,
            "active_actuators": list(ACTIVE_ACTUATORS),
            "active_fingers": list(ACTIVE_FINGERS),
            "trust_region_scale": self.trust_region_scale,
            "runtime_manipulation_plan": self.as_manipulation_plan_config(),
            "knot_fraction": self.knot_fraction.tolist(),
            "path_progress": self.path_progress.tolist(),
            "desired_object_response_6d": self.desired_object_response_6d.tolist(),
            "command_delta_rad": self.command_delta_rad.tolist(),
            "predicted_object_response_6d": self.predicted_object_response_6d.tolist(),
            "predicted_target_normal_force_n": self.predicted_target_normal_force_n.tolist(),
            "predicted_tangent_slip_m": self.predicted_tangent_slip_m.tolist(),
            "predicted_contact_valid": self.predicted_contact_valid.tolist(),
            "solver_objective": self.solver_objective.tolist(),
            "solver_max_linear_violation": self.solver_max_linear_violation.tolist(),
            "solver_converged": self.solver_converged.tolist(),
            "summary": {
                "contact_feasible": self.contact_feasible,
                "contact_loss_count": self.contact_loss_count,
                "path_rms_error": self.path_rms_error,
                "terminal_path_error": self.terminal_path_error,
                "maximum_predicted_tangent_slip_m": float(
                    np.max(self.predicted_tangent_slip_m, initial=0.0)
                ),
            },
        }

    def as_mapping(self) -> dict[str, Any]:
        result = self._payload()
        result["plan_id"] = self.plan_id
        return result


def interpolate_plan_at_fraction(
    plan: TwentyOneKnotContactPlan,
    fraction: float,
) -> dict[str, Any]:
    """Interpolate one runtime setpoint from the sealed 21-knot plan.

    Continuous values use the exact shared clamped quintic spline used by the
    runtime controller and raw-trace evaluator.  Contact validity is
    conservative: an in-between sample is valid only when both bracketing
    knots are valid.  The function is pure and therefore suitable for planner,
    controller and trace recomputation equivalence tests.
    """

    value = _finite_float(fraction, "fraction")
    if not 0.0 <= value <= 1.0:
        raise ValueError("fraction must lie within [0, 1]")

    upper = int(np.searchsorted(plan.knot_fraction, value, side="left"))
    upper = min(max(upper, 0), KNOT_COUNT - 1)
    if abs(float(plan.knot_fraction[upper]) - value) <= _EPSILON:
        lower = upper
    else:
        lower = max(0, upper - 1)
    if lower == upper:
        local_parameter = 0.0
    else:
        local_parameter = (value - float(plan.knot_fraction[lower])) / (
            float(plan.knot_fraction[upper])
            - float(plan.knot_fraction[lower])
        )

    knot_times_s = plan.knot_fraction * plan.duration_s
    elapsed_s = value * plan.duration_s

    # The derivative solve is linear, so solve all continuous planner fields in
    # one system.  Apart from being substantially cheaper for diagnostic sweep
    # callers, this prevents apparently identical fields from accidentally
    # taking different interpolation paths in future changes.
    field_widths = (len(ACTIVE_ACTUATORS), 6, 6, len(ACTIVE_FINGERS), len(ACTIVE_FINGERS))
    continuous_knots = np.concatenate(
        (
            plan.command_delta_rad,
            plan.desired_object_response_6d,
            plan.predicted_object_response_6d,
            plan.predicted_target_normal_force_n,
            plan.predicted_tangent_slip_m,
        ),
        axis=1,
    )
    velocities, accelerations = quintic_c2_knot_derivatives(
        knot_times_s, continuous_knots
    )
    continuous_value, _, _, _ = interpolate_quintic_c2(
        knot_times_s,
        continuous_knots,
        elapsed_s,
        knot_velocities=velocities,
        knot_accelerations=accelerations,
    )
    split = np.cumsum(field_widths[:-1])
    command, desired, predicted, force, slip = np.split(
        np.asarray(continuous_value, dtype=np.float64), split
    )

    valid = plan.predicted_contact_valid[lower] & plan.predicted_contact_valid[upper]
    return {
        "plan_id": plan.plan_id,
        "fraction": value,
        "bracketing_knot_indices": [lower, upper],
        "local_quintic_parameter": float(local_parameter),
        "interpolation_profile": "shared_clamped_c4_quintic",
        "command_delta_rad": _named_vector(
            command, ACTIVE_ACTUATORS
        ),
        "desired_object_response_6d": desired.tolist(),
        "predicted_object_response_6d": predicted.tolist(),
        "predicted_target_normal_force_n": _named_vector(
            force, ACTIVE_FINGERS
        ),
        "predicted_tangent_slip_m": _named_vector(
            slip, ACTIVE_FINGERS
        ),
        "predicted_contact_valid": {
            finger: bool(valid[index])
            for index, finger in enumerate(ACTIVE_FINGERS)
        },
    }


def v14_planner_id(settings: ContactConstrainedPlannerSettings) -> str:
    return _domain_id(
        "v14_contact_constrained_offline_planner",
        {
            "algorithm": "extended_probe_active_set_projected_ls_four_backoffs",
            "settings": settings.as_mapping(),
        },
    )


def v14_controller_id(
    config: Mapping[str, Any],
    settings: ContactConstrainedPlannerSettings,
    response: ExtendedProbeResponse,
    plan: TwentyOneKnotContactPlan,
) -> str:
    """Bind acquisition law, pair, fitted evidence, planner and exact knots."""

    return _domain_id(
        "v14_contact_constrained_controller",
        {
            "grasp_object_pair_id": v14_grasp_object_pair_id(config),
            "acquisition_controller": controller_context(config),
            "contact_force_targets_n": copy.deepcopy(
                config.get("contact_force_targets_n")
            ),
            "contact_feedback": copy.deepcopy(config.get("contact_feedback")),
            "planner_id": v14_planner_id(settings),
            "response_model_id": response.response_model_id,
            "plan_id": plan.plan_id,
        },
    )


@dataclass(frozen=True, slots=True)
class V14PlanningIdentities:
    object_config_id: str
    grasp_pose_id: str
    grasp_object_pair_id: str
    planner_id: str
    response_model_id: str
    plan_id: str
    controller_id: str

    def as_mapping(self) -> dict[str, str]:
        return {
            "object_config_id": self.object_config_id,
            "grasp_pose_id": self.grasp_pose_id,
            "grasp_object_pair_id": self.grasp_object_pair_id,
            "planner_id": self.planner_id,
            "response_model_id": self.response_model_id,
            "plan_id": self.plan_id,
            "controller_id": self.controller_id,
        }


def build_v14_planning_identities(
    config: Mapping[str, Any],
    settings: ContactConstrainedPlannerSettings,
    response: ExtendedProbeResponse,
    plan: TwentyOneKnotContactPlan,
) -> V14PlanningIdentities:
    return V14PlanningIdentities(
        object_config_id=v14_object_config_id(config),
        grasp_pose_id=v14_grasp_pose_id(config),
        grasp_object_pair_id=v14_grasp_object_pair_id(config),
        planner_id=v14_planner_id(settings),
        response_model_id=response.response_model_id,
        plan_id=plan.plan_id,
        controller_id=v14_controller_id(config, settings, response, plan),
    )


def _validate_delta_bounds(
    bounds: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    if set(bounds) != set(ACTIVE_ACTUATORS):
        raise ValueError("delta_bounds must name exactly the eight active actuators")
    lower = np.asarray([float(bounds[name][0]) for name in ACTIVE_ACTUATORS])
    upper = np.asarray([float(bounds[name][1]) for name in ACTIVE_ACTUATORS])
    if (
        not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
    ):
        raise ValueError("delta_bounds must be finite and increasing")
    if np.any(lower > 0.0) or np.any(upper < 0.0):
        raise ValueError("delta_bounds must contain the zero grasp command")
    return lower, upper


def _continuous_path_validity(
    knot_times_s: np.ndarray,
    command_delta_rad: np.ndarray,
    predicted_force_n: np.ndarray,
    predicted_slip_m: np.ndarray,
    *,
    command_lower_rad: np.ndarray,
    command_upper_rad: np.ndarray,
    minimum_force_n: np.ndarray,
    maximum_slip_m: np.ndarray,
    tolerance: float,
    sample_period_s: float = 0.001,
) -> np.ndarray:
    """Audit planner constraints on the exact runtime spline.

    Knot-only feasibility is insufficient for a shared high-order spline: an
    interior segment can overshoot a command bound, the minimum contact force,
    or the slip limit while both endpoints remain feasible.  This audit uses
    the same interpolation helper as the runtime controller at every nominal
    1 ms physics instant.  A violation invalidates both endpoints of that
    segment, making the existing contact-first plan rank conservative without
    adding a second, contradictory feasibility representation.
    """

    times = np.asarray(knot_times_s, dtype=np.float64)
    command = np.asarray(command_delta_rad, dtype=np.float64)
    force = np.asarray(predicted_force_n, dtype=np.float64)
    slip = np.asarray(predicted_slip_m, dtype=np.float64)
    lower = np.asarray(command_lower_rad, dtype=np.float64)
    upper = np.asarray(command_upper_rad, dtype=np.float64)
    minimum_force = np.asarray(minimum_force_n, dtype=np.float64)
    maximum_slip = np.asarray(maximum_slip_m, dtype=np.float64)
    period = _finite_float(sample_period_s, "sample_period_s")
    epsilon = _finite_float(tolerance, "tolerance")
    if period <= 0.0 or epsilon < 0.0:
        raise ValueError("continuous audit period must be positive and tolerance non-negative")

    continuous = np.concatenate((command, force, slip), axis=1)
    velocities, accelerations = quintic_c2_knot_derivatives(times, continuous)
    validity = np.ones((times.size, len(ACTIVE_FINGERS)), dtype=bool)
    command_width = len(ACTIVE_ACTUATORS)
    force_end = command_width + len(ACTIVE_FINGERS)
    for segment in range(times.size - 1):
        duration = float(times[segment + 1] - times[segment])
        sample_count = max(1, int(math.ceil(duration / period)))
        segment_valid = np.ones(len(ACTIVE_FINGERS), dtype=bool)
        for offset in range(sample_count + 1):
            elapsed = float(times[segment] + duration * offset / sample_count)
            value, _, _, _ = interpolate_quintic_c2(
                times,
                continuous,
                elapsed,
                knot_velocities=velocities,
                knot_accelerations=accelerations,
            )
            command_now = value[:command_width]
            force_now = value[command_width:force_end]
            slip_now = value[force_end:]
            command_safe = bool(
                np.isfinite(command_now).all()
                and np.all(command_now >= lower - epsilon)
                and np.all(command_now <= upper + epsilon)
            )
            finger_safe = (
                np.isfinite(force_now)
                & np.isfinite(slip_now)
                & (force_now >= minimum_force - epsilon)
                & (slip_now >= -epsilon)
                & (slip_now <= maximum_slip + epsilon)
            )
            if not command_safe:
                finger_safe[:] = False
            segment_valid &= finger_safe
        validity[segment] &= segment_valid
        validity[segment + 1] &= segment_valid
    return validity


def _build_plan_attempt(
    response: ExtendedProbeResponse,
    delta_bounds: Mapping[str, Sequence[float]],
    settings: ContactConstrainedPlannerSettings,
    *,
    trust_region_scale: float,
) -> TwentyOneKnotContactPlan:
    lower, upper = _validate_delta_bounds(delta_bounds)
    knot_fraction = np.linspace(0.0, 1.0, KNOT_COUNT, dtype=np.float64)
    progress = _minimum_jerk(knot_fraction)
    target_object = np.asarray(settings.target_object_response_6d)
    desired_object = progress[:, None] * target_object[None, :]
    target_force = np.asarray(settings.target_normal_force_n)
    minimum_force = np.asarray(settings.minimum_normal_force_n)
    maximum_slip = np.asarray(settings.maximum_tangent_slip_m)
    trust_radius = np.asarray(settings.trust_radius_rad) * float(trust_region_scale)
    matrix = np.vstack(
        (
            response.object_jacobian_6x8,
            response.force_jacobian_3x8,
            response.slip_jacobian_3x8,
        )
    )
    bias = np.concatenate(
        (response.object_bias_6d, response.force_bias_n, response.slip_bias_m)
    )
    weights = 1.0 / np.concatenate(
        (
            np.asarray(settings.object_response_scale),
            np.asarray(settings.force_response_scale_n),
            np.asarray(settings.slip_response_scale_m),
        )
    )
    band_matrix = np.vstack(
        (response.force_jacobian_3x8, response.slip_jacobian_3x8)
    )
    band_lower = np.concatenate(
        (minimum_force - response.force_bias_n, -response.slip_bias_m)
    )
    band_upper = np.concatenate(
        (
            np.full(len(ACTIVE_FINGERS), math.inf),
            maximum_slip - response.slip_bias_m,
        )
    )

    commands = np.zeros((KNOT_COUNT, len(ACTIVE_ACTUATORS)), dtype=np.float64)
    predicted_object = np.empty((KNOT_COUNT, 6), dtype=np.float64)
    predicted_force = np.empty((KNOT_COUNT, len(ACTIVE_FINGERS)), dtype=np.float64)
    predicted_slip = np.empty((KNOT_COUNT, len(ACTIVE_FINGERS)), dtype=np.float64)
    predicted_valid = np.empty((KNOT_COUNT, len(ACTIVE_FINGERS)), dtype=bool)
    objectives = np.zeros(KNOT_COUNT, dtype=np.float64)
    violations = np.zeros(KNOT_COUNT, dtype=np.float64)
    converged = np.ones(KNOT_COUNT, dtype=bool)
    model_contact_safe = response.zero_contact_safe

    for knot in range(KNOT_COUNT):
        previous = commands[knot - 1] if knot else np.zeros(len(ACTIVE_ACTUATORS))
        if knot:
            local_lower = np.maximum(lower, previous - trust_radius)
            local_upper = np.minimum(upper, previous + trust_radius)
            target = np.concatenate(
                (desired_object[knot], target_force, np.zeros(len(ACTIVE_FINGERS)))
            ) - bias
            solved = solve_bounded_projected_least_squares(
                matrix,
                target,
                local_lower,
                local_upper,
                weights=weights,
                ridge=settings.ridge,
                center=previous,
                linear_matrix=band_matrix,
                linear_lower=band_lower,
                linear_upper=band_upper,
                max_iterations=settings.solver_max_iterations,
                tolerance=settings.solver_tolerance,
            )
            commands[knot] = solved.solution
            objectives[knot] = solved.objective
            violations[knot] = solved.max_linear_violation
            converged[knot] = solved.converged
        prediction = response.predict(commands[knot])
        predicted_object[knot] = prediction["object_response_6d"]
        predicted_force[knot] = np.maximum(
            0.0, prediction["target_normal_force_n"]
        )
        predicted_slip[knot] = prediction["tangent_slip_m"]
        predicted_valid[knot] = bool(model_contact_safe) & (
            predicted_force[knot] >= minimum_force - settings.solver_tolerance
        ) & (predicted_slip[knot] <= maximum_slip + settings.solver_tolerance)
    predicted_valid &= _continuous_path_validity(
        knot_fraction * settings.duration_s,
        commands,
        predicted_force,
        predicted_slip,
        command_lower_rad=lower,
        command_upper_rad=upper,
        minimum_force_n=minimum_force,
        maximum_slip_m=maximum_slip,
        tolerance=settings.solver_tolerance,
    )
    return TwentyOneKnotContactPlan(
        trust_region_scale=trust_region_scale,
        duration_s=settings.duration_s,
        profile=settings.profile,
        max_knot_delta_rad=settings.max_knot_delta_rad,
        trust_region_backtracks=len(settings.backoff_scales),
        knot_fraction=knot_fraction,
        path_progress=progress,
        desired_object_response_6d=desired_object,
        command_delta_rad=commands,
        predicted_object_response_6d=predicted_object,
        predicted_target_normal_force_n=predicted_force,
        predicted_tangent_slip_m=predicted_slip,
        predicted_contact_valid=predicted_valid,
        solver_objective=objectives,
        solver_max_linear_violation=violations,
        solver_converged=converged,
    )


def _attempt_rank(plan: TwentyOneKnotContactPlan, index: int) -> tuple[Any, ...]:
    minimum_force = float(np.min(plan.predicted_target_normal_force_n))
    return (
        not plan.contact_feasible,
        plan.contact_loss_count,
        -minimum_force,
        float(np.max(plan.predicted_tangent_slip_m, initial=0.0)),
        plan.path_rms_error,
        plan.terminal_path_error,
        float(np.max(plan.solver_max_linear_violation, initial=0.0)),
        float(np.sum(plan.solver_objective)),
        int(index),
    )


@dataclass(frozen=True, slots=True)
class ContactConstrainedPlanningReport:
    """Four attempts, one deterministic selection and all identity bindings."""

    settings: ContactConstrainedPlannerSettings
    response: ExtendedProbeResponse
    delta_bounds: dict[str, tuple[float, float]]
    attempts: tuple[TwentyOneKnotContactPlan, ...]
    selected_attempt_index: int
    identities: V14PlanningIdentities

    def __post_init__(self) -> None:
        if len(self.attempts) != 4:
            raise ValueError("planning report must contain exactly four backoff attempts")
        if not 0 <= int(self.selected_attempt_index) < len(self.attempts):
            raise ValueError("selected_attempt_index is out of range")
        normalized = {
            str(name): (float(value[0]), float(value[1]))
            for name, value in self.delta_bounds.items()
        }
        _validate_delta_bounds(normalized)
        object.__setattr__(self, "delta_bounds", normalized)
        object.__setattr__(self, "attempts", tuple(self.attempts))
        object.__setattr__(self, "selected_attempt_index", int(self.selected_attempt_index))

    @property
    def selected_plan(self) -> TwentyOneKnotContactPlan:
        return self.attempts[self.selected_attempt_index]

    def as_mapping(self) -> dict[str, Any]:
        payload = {
            "contact_constrained_planning_report_schema_version": 1,
            "search_evidence_only": True,
            "final_success_requires_full_reset_rerun": True,
            "settings": self.settings.as_mapping(),
            "response": self.response.as_mapping(),
            "delta_bounds": {
                name: [float(self.delta_bounds[name][0]), float(self.delta_bounds[name][1])]
                for name in ACTIVE_ACTUATORS
            },
            "attempts": [value.as_mapping() for value in self.attempts],
            "selected_attempt_index": self.selected_attempt_index,
            "selected_plan_id": self.selected_plan.plan_id,
            "selected_contact_feasible": self.selected_plan.contact_feasible,
            "identities": self.identities.as_mapping(),
        }
        payload["report_id"] = _domain_id("v14_contact_planning_report", payload)
        return payload


def plan_contact_constrained_trajectory(
    config: Mapping[str, Any],
    response: ExtendedProbeResponse,
    delta_bounds: Mapping[str, Sequence[float]],
    *,
    settings: ContactConstrainedPlannerSettings = ContactConstrainedPlannerSettings(),
) -> ContactConstrainedPlanningReport:
    """Generate all four trust-region attempts and select contact-first."""

    normalized_bounds = {
        name: (float(delta_bounds[name][0]), float(delta_bounds[name][1]))
        for name in ACTIVE_ACTUATORS
    }
    _validate_delta_bounds(normalized_bounds)
    attempts = tuple(
        _build_plan_attempt(
            response,
            normalized_bounds,
            settings,
            trust_region_scale=scale,
        )
        for scale in settings.backoff_scales
    )
    selected_index = min(
        range(len(attempts)),
        key=lambda index: _attempt_rank(attempts[index], index),
    )
    selected = attempts[selected_index]
    identities = build_v14_planning_identities(
        config, settings, response, selected
    )
    return ContactConstrainedPlanningReport(
        settings=settings,
        response=response,
        delta_bounds=normalized_bounds,
        attempts=attempts,
        selected_attempt_index=selected_index,
        identities=identities,
    )


def materialize_contact_plan_config(
    base_config: Mapping[str, Any],
    plan: TwentyOneKnotContactPlan,
    *,
    validate: bool = False,
) -> dict[str, Any]:
    """Install a planned path and its matching legacy terminal delta.

    Schema-v14 keeps ``control.manipulation_delta_rad`` as a compatibility
    mirror of the last waypoint.  Updating only ``manipulation_plan`` is
    intentionally rejected by configuration validation, so runners should use
    this helper rather than editing either field independently.
    """

    resolved = copy.deepcopy(dict(base_config))
    control = resolved.get("control")
    if not isinstance(control, dict):
        raise ValueError("base_config.control must be a mutable mapping")
    resolved["manipulation_plan"] = plan.as_manipulation_plan_config()
    control["manipulation_delta_rad"] = _named_vector(
        plan.command_delta_rad[-1], ACTIVE_ACTUATORS
    )
    if validate:
        from ..config import validate_config

        validate_config(resolved)
    return resolved


def _finite_rank_value(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _nonnegative_rank_int(value: Any, default: int = 10**9) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result >= 0 else default


def contact_first_rank_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize runner output into the schema-v14 contact-first rank order."""

    summary = record.get("summary", record)
    if not isinstance(summary, Mapping):
        summary = {}
    stage = summary.get("stage_status", record.get("stage_status", {}))
    if not isinstance(stage, Mapping):
        stage = {}
    metrics = summary.get("metrics", record.get("metrics", {}))
    if not isinstance(metrics, Mapping):
        metrics = {}
    contact = record.get("contact_maintenance", metrics.get("contact_maintenance", {}))
    if not isinstance(contact, Mapping):
        contact = {}
    checks = summary.get("checks", record.get("checks", {}))
    if not isinstance(checks, Mapping):
        checks = {}
    if not contact:
        # Direct adapter for the evaluator's persisted schema-v14 metrics.  A
        # runner may alternatively provide the compact contact_maintenance
        # block above; both routes yield the same rank axes.
        planned = metrics.get("contact_preserving_planned_lift", {})
        if isinstance(planned, Mapping):
            duty_mapping = planned.get("target_face_effective_duty", {})
            duties = []
            if isinstance(duty_mapping, Mapping):
                duties.extend(
                    _finite_rank_value(duty_mapping.get(finger), -math.inf)
                    for finger in ACTIVE_FINGERS
                )
            duties.append(
                _finite_rank_value(
                    planned.get("simultaneous_target_face_effective_duty"),
                    -math.inf,
                )
            )
            longest_mapping = planned.get("longest_contact_loss_steps", {})
            longest = []
            if isinstance(longest_mapping, Mapping):
                longest.extend(
                    _nonnegative_rank_int(longest_mapping.get(finger))
                    for finger in ACTIVE_FINGERS
                )
            longest.append(
                _nonnegative_rank_int(
                    planned.get("simultaneous_longest_contact_loss_steps")
                )
            )
            required_duty = _finite_rank_value(
                planned.get("required_contact_duty"), math.inf
            )
            allowed_loss = _nonnegative_rank_int(
                planned.get("allowed_contact_loss_steps"), -1
            )
            minimum_duty = min(duties, default=-math.inf)
            longest_loss = max(longest, default=10**9)
            contact_check_names = (
                "v14_operation_did_not_abort",
                "v14_thumb_contact_duty_at_least_99_percent",
                "v14_index_contact_duty_at_least_99_percent",
                "v14_middle_contact_duty_at_least_99_percent",
                "v14_simultaneous_contact_duty_at_least_99_percent",
                "v14_thumb_contact_loss_within_limit",
                "v14_index_contact_loss_within_limit",
                "v14_middle_contact_loss_within_limit",
                "v14_simultaneous_contact_loss_within_limit",
            )
            checks_present = all(name in checks for name in contact_check_names)
            checks_passed = checks_present and all(
                bool(checks[name]) for name in contact_check_names
            )
            satisfied = bool(
                checks_passed
                and minimum_duty >= required_duty - _EPSILON
                and allowed_loss >= 0
                and longest_loss <= allowed_loss
                and not bool(planned.get("operation_aborted", True))
            )
            slip_max = math.inf
            targeting = metrics.get("contact_point_targeting", {})
            if isinstance(targeting, Mapping):
                slip = targeting.get("contact_slip_from_grasp", {})
                operation = slip.get("operation", {}) if isinstance(slip, Mapping) else {}
                per_finger = (
                    operation.get("per_finger", {})
                    if isinstance(operation, Mapping)
                    else {}
                )
                if isinstance(per_finger, Mapping):
                    values = []
                    for finger in ACTIVE_FINGERS:
                        item = per_finger.get(finger, {})
                        if isinstance(item, Mapping):
                            values.append(
                                _finite_rank_value(
                                    item.get("tangent_slip_max_m"), math.inf
                                )
                            )
                    slip_max = max(values, default=math.inf)
            contact = {
                "satisfied": satisfied,
                "forbidden_contact": _nonnegative_rank_int(
                    metrics.get("forbidden_contact_steps"), 10**9
                )
                > 0,
                "active_nondistal_contact": not bool(
                    checks.get("active_nondistal_contacts_within_limit", False)
                ),
                "contact_loss_count": longest_loss,
                "minimum_valid_duty": minimum_duty,
                "minimum_force_margin_n": -math.inf,
                "maximum_tangent_slip_m": slip_max,
            }
    path = record.get("path_tracking", metrics.get("path_tracking", {}))
    if not isinstance(path, Mapping):
        path = {}
    identifier = record.get("candidate_id", record.get("candidate_index", ""))
    if isinstance(identifier, bool):
        identifier = str(identifier)
    stable_identifier: tuple[int, int | str]
    if isinstance(identifier, int):
        stable_identifier = (0, int(identifier))
    else:
        stable_identifier = (1, str(identifier))

    def optional(value: float) -> float | None:
        return value if math.isfinite(value) else None

    minimum_valid_duty = _finite_rank_value(
        contact.get("minimum_valid_duty"), -math.inf
    )
    minimum_force_margin = _finite_rank_value(
        contact.get("minimum_force_margin_n"), -math.inf
    )
    maximum_slip = _finite_rank_value(
        contact.get("maximum_tangent_slip_m"), math.inf
    )
    path_rms = _finite_rank_value(path.get("rms_error"), math.inf)
    terminal_path = _finite_rank_value(path.get("terminal_error"), math.inf)
    objective = _finite_rank_value(record.get("objective"), math.inf)
    perturbation_pass_count = _nonnegative_rank_int(
        record.get(
            "perturbation_pass_count",
            record.get("perturbation_passes", 0),
        )
    )
    return {
        "contact_constraints_satisfied": bool(contact.get("satisfied", False)),
        "forbidden_contact": bool(contact.get("forbidden_contact", False)),
        "active_nondistal_contact": bool(
            contact.get("active_nondistal_contact", False)
        ),
        "contact_loss_count": _nonnegative_rank_int(
            contact.get("contact_loss_count", 10**9)
        ),
        "minimum_valid_duty": optional(minimum_valid_duty),
        "minimum_force_margin_n": optional(minimum_force_margin),
        "maximum_tangent_slip_m": optional(maximum_slip),
        "full_success": bool(stage.get("full_success", summary.get("passed", False))),
        "perturbation_pass_count": perturbation_pass_count,
        "path_rms_error": optional(path_rms),
        "terminal_path_error": optional(terminal_path),
        "objective": optional(objective),
        "stable_candidate_id": stable_identifier,
    }


def contact_first_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Apply the declared v14 hard-pass, robustness, then contact order."""

    evidence = contact_first_rank_evidence(record)
    return (
        not evidence["full_success"],
        -int(evidence["perturbation_pass_count"]),
        not evidence["contact_constraints_satisfied"],
        evidence["forbidden_contact"],
        evidence["active_nondistal_contact"],
        int(evidence["contact_loss_count"]),
        -_finite_rank_value(evidence["minimum_valid_duty"], -math.inf),
        -_finite_rank_value(evidence["minimum_force_margin_n"], -math.inf),
        _finite_rank_value(evidence["maximum_tangent_slip_m"], math.inf),
        _finite_rank_value(evidence["path_rms_error"], math.inf),
        _finite_rank_value(evidence["terminal_path_error"], math.inf),
        _finite_rank_value(evidence["objective"], math.inf),
        evidence["stable_candidate_id"],
    )


def rank_contact_constrained_candidates(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Deep-copy and sort independently of input/worker completion order."""

    materialized = [copy.deepcopy(dict(record)) for record in records]
    materialized.sort(key=contact_first_candidate_rank)
    return tuple(materialized)


__all__ = [
    "BoundedProjectedLeastSquaresResult",
    "ContactConstrainedPlannerSettings",
    "ContactConstrainedPlanningReport",
    "ExtendedProbeResponse",
    "ExtendedProbeSample",
    "KNOT_COUNT",
    "RESPONSE_DIMENSION",
    "TRUST_REGION_BACKOFF_SCALES",
    "PLAN_SEGMENT_COUNT",
    "TwentyOneKnotContactPlan",
    "V14PlanningIdentities",
    "build_v14_planning_identities",
    "collect_extended_probe_samples",
    "contact_first_candidate_rank",
    "contact_first_rank_evidence",
    "fit_extended_probe_response",
    "interpolate_plan_at_fraction",
    "materialize_contact_plan_config",
    "object_configuration_context",
    "plan_contact_constrained_trajectory",
    "rank_contact_constrained_candidates",
    "solve_bounded_projected_least_squares",
    "v14_controller_id",
    "v14_grasp_object_pair_id",
    "v14_grasp_pose_id",
    "v14_object_config_id",
    "v14_planner_id",
]
