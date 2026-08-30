"""Rolling-aware contact slip from material-point relative velocity.

Centroid displacement is not a reliable slip measurement for a collision
manifold: contact points can be reordered, created or deleted by the solver,
and a fingertip can roll across an object without material slip.  This module
therefore integrates the *relative tangential velocity* of the two contacting
material points.  Contact witnesses are used only to maintain patch identity
and to diagnose rolling/manifold switches; witness jumps never enter the slip
integral.

The implementation is an opt-in incremental primitive.  It does not alter the
legacy centroid-based diagnostics, controller, or simulation loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import ACTIVE_FINGERS
from .contact_point_targeting import FACE_AXIS_AND_SIGN
from .contacts import surface_witness


ROLLING_CONTACT_SLIP_SCHEMA_VERSION = 1
_EPSILON = 1e-12
_POSITIVE_AXIS_LABELS = ("+X", "+Y", "+Z")


def _readonly(value: ArrayLike, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must have shape {shape} and contain finite values")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _unit_vector(value: ArrayLike, label: str) -> np.ndarray:
    vector = _readonly(value, (3,), label)
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{label} must have non-zero length")
    result = np.array(vector / norm, copy=True)
    result.setflags(write=False)
    return result


def _nonnegative(value: Any, label: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return resolved


def _positive(value: Any, label: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return resolved


def _finger_index(value: Any, finger_count: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("finger_index must be an integer")
    resolved = int(value)
    if resolved != value or not 0 <= resolved < finger_count:
        raise ValueError("finger_index is outside the estimator finger range")
    return resolved


def _target_faces(value: Sequence[str], finger_count: int) -> tuple[str, ...]:
    result = tuple(str(item) for item in value)
    if len(result) != finger_count or any(
        face not in FACE_AXIS_AND_SIGN for face in result
    ):
        raise ValueError(
            "target_faces must contain one canonical cube face per finger"
        )
    return result


def _tangent_axes(face: str) -> tuple[int, int]:
    normal_axis, _ = FACE_AXIS_AND_SIGN[face]
    axes = tuple(axis for axis in range(3) if axis != normal_axis)
    assert len(axes) == 2
    return axes


@dataclass(frozen=True, slots=True)
class RollingSlipSettings:
    """Numerical gates for patch matching and rolling diagnostics."""

    schema_version: int = ROLLING_CONTACT_SLIP_SCHEMA_VERSION
    minimum_sample_normal_force_n: float = 1e-8
    minimum_total_normal_force_n: float = 0.05
    minimum_normal_alignment: float = 0.95
    maximum_patch_match_distance_m: float = 0.003
    centroid_jump_threshold_m: float = 0.0015
    rolling_witness_speed_min_m_s: float = 0.0001
    rolling_relative_slip_speed_max_m_s: float = 0.0001
    rolling_force_fraction_min: float = 0.5
    maximum_time_gap_s: float = 0.002

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, (bool, np.bool_))
            or int(self.schema_version) != self.schema_version
            or int(self.schema_version) != ROLLING_CONTACT_SLIP_SCHEMA_VERSION
        ):
            raise ValueError("unsupported rolling-slip schema_version")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        for name in (
            "minimum_sample_normal_force_n",
            "rolling_witness_speed_min_m_s",
            "rolling_relative_slip_speed_max_m_s",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        for name in (
            "minimum_total_normal_force_n",
            "maximum_patch_match_distance_m",
            "centroid_jump_threshold_m",
            "maximum_time_gap_s",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("minimum_normal_alignment", "rolling_force_fraction_min"):
            resolved = float(getattr(self, name))
            if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, resolved)
        if self.minimum_normal_alignment <= 0.0:
            raise ValueError("minimum_normal_alignment must be positive")


@dataclass(frozen=True, slots=True)
class ContactPatchKinematics:
    """One cube/fingertip contact sample after ``mj_forward``.

    ``relative_velocity_cube_local_m_s`` is finger material velocity minus
    cube material velocity, evaluated at the same world contact point and then
    rotated into the cube frame.  ``patch_identity`` should be stable for a
    geom pair; multiple manifold points sharing it are disambiguated by nearest
    one-to-one witness matching.
    """

    finger_index: int
    patch_identity: str
    witness_cube_local_m: NDArray[np.float64]
    outward_normal_cube_local: NDArray[np.float64]
    relative_velocity_cube_local_m_s: NDArray[np.float64]
    normal_force_n: float
    source_contact_index: int = -1

    def __post_init__(self) -> None:
        if isinstance(self.finger_index, (bool, np.bool_)):
            raise ValueError("finger_index must be a nonnegative integer")
        finger = int(self.finger_index)
        if finger != self.finger_index or finger < 0:
            raise ValueError("finger_index must be a nonnegative integer")
        if not isinstance(self.patch_identity, str) or not self.patch_identity:
            raise ValueError("patch_identity must be a non-empty string")
        source = int(self.source_contact_index)
        if (
            isinstance(self.source_contact_index, (bool, np.bool_))
            or source != self.source_contact_index
            or source < -1
        ):
            raise ValueError("source_contact_index must be an integer >= -1")
        object.__setattr__(self, "finger_index", finger)
        object.__setattr__(self, "source_contact_index", source)
        object.__setattr__(
            self,
            "witness_cube_local_m",
            _readonly(self.witness_cube_local_m, (3,), "witness_cube_local_m"),
        )
        object.__setattr__(
            self,
            "outward_normal_cube_local",
            _unit_vector(
                self.outward_normal_cube_local,
                "outward_normal_cube_local",
            ),
        )
        object.__setattr__(
            self,
            "relative_velocity_cube_local_m_s",
            _readonly(
                self.relative_velocity_cube_local_m_s,
                (3,),
                "relative_velocity_cube_local_m_s",
            ),
        )
        object.__setattr__(
            self,
            "normal_force_n",
            _nonnegative(self.normal_force_n, "normal_force_n"),
        )


@dataclass(frozen=True, slots=True)
class RollingAwareSlipEstimate:
    """One incremental three-finger rolling/slip observation."""

    time_s: float
    target_faces: tuple[str, ...]
    signed_tangent_displacement_m: NDArray[np.float64]
    cumulative_irrecoverable_slip_m: NDArray[np.float64]
    relative_tangent_velocity_m_s: NDArray[np.float64]
    normal_force_n: NDArray[np.float64]
    valid: NDArray[np.bool_]
    continuous: NDArray[np.bool_]
    rolling_detected: NDArray[np.bool_]
    rolling_force_fraction: NDArray[np.float64]
    patch_switch: NDArray[np.bool_]
    centroid_jump: NDArray[np.bool_]
    centroid_tangent_step_m: NDArray[np.float64]
    matched_patch_count: NDArray[np.int64]
    new_patch_count: NDArray[np.int64]
    dropped_patch_count: NDArray[np.int64]
    patch_switch_count: NDArray[np.int64]

    def __post_init__(self) -> None:
        time_s = float(self.time_s)
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and nonnegative")
        object.__setattr__(self, "time_s", time_s)
        finger_count = np.asarray(self.valid).size
        object.__setattr__(
            self,
            "target_faces",
            _target_faces(self.target_faces, finger_count),
        )
        shapes = {
            "signed_tangent_displacement_m": (finger_count, 2),
            "cumulative_irrecoverable_slip_m": (finger_count,),
            "relative_tangent_velocity_m_s": (finger_count, 2),
            "normal_force_n": (finger_count,),
            "rolling_force_fraction": (finger_count,),
            "centroid_tangent_step_m": (finger_count,),
        }
        boolean_names = {
            "valid",
            "continuous",
            "rolling_detected",
            "patch_switch",
            "centroid_jump",
        }
        integer_names = {
            "matched_patch_count",
            "new_patch_count",
            "dropped_patch_count",
            "patch_switch_count",
        }
        for name, shape in shapes.items():
            value = _readonly(getattr(self, name), shape, name)
            if name in {
                "cumulative_irrecoverable_slip_m",
                "normal_force_n",
                "rolling_force_fraction",
                "centroid_tangent_step_m",
            } and np.any(value < 0.0):
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        for name in boolean_names:
            value = np.asarray(getattr(self, name), dtype=bool)
            if value.shape != (finger_count,):
                raise ValueError(f"{name} must have shape ({finger_count},)")
            result = np.array(value, copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)
        for name in integer_names:
            value = np.asarray(getattr(self, name), dtype=np.int64)
            if value.shape != (finger_count,) or np.any(value < 0):
                raise ValueError(
                    f"{name} must have shape ({finger_count},) and be nonnegative"
                )
            result = np.array(value, copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)

    def as_mapping(self, finger_names: Sequence[str] = ACTIVE_FINGERS) -> dict[str, Any]:
        names = tuple(str(value) for value in finger_names)
        if len(names) != self.valid.size:
            raise ValueError("finger_names does not match estimate finger count")
        return {
            "rolling_contact_slip_schema_version": ROLLING_CONTACT_SLIP_SCHEMA_VERSION,
            "time_s": self.time_s,
            "per_finger": {
                finger: {
                    "target_face": self.target_faces[index],
                    "signed_tangent_axes_cube": [
                        _POSITIVE_AXIS_LABELS[axis]
                        for axis in _tangent_axes(self.target_faces[index])
                    ],
                    "signed_tangent_displacement_m": (
                        self.signed_tangent_displacement_m[index].tolist()
                    ),
                    "cumulative_irrecoverable_slip_m": float(
                        self.cumulative_irrecoverable_slip_m[index]
                    ),
                    "relative_tangent_velocity_m_s": (
                        self.relative_tangent_velocity_m_s[index].tolist()
                    ),
                    "normal_force_n": float(self.normal_force_n[index]),
                    "valid": bool(self.valid[index]),
                    "continuous": bool(self.continuous[index]),
                    "rolling_detected": bool(self.rolling_detected[index]),
                    "rolling_force_fraction": float(
                        self.rolling_force_fraction[index]
                    ),
                    "patch_switch": bool(self.patch_switch[index]),
                    "centroid_jump": bool(self.centroid_jump[index]),
                    "centroid_tangent_step_m": float(
                        self.centroid_tangent_step_m[index]
                    ),
                    "matched_patch_count": int(self.matched_patch_count[index]),
                    "new_patch_count": int(self.new_patch_count[index]),
                    "dropped_patch_count": int(
                        self.dropped_patch_count[index]
                    ),
                    "patch_switch_count": int(self.patch_switch_count[index]),
                }
                for index, finger in enumerate(names)
            },
        }


@dataclass(frozen=True, slots=True)
class RollingTangentJacobian:
    """Force-weighted contact-position Jacobian in each target-face plane.

    The last axis follows the caller-provided active-DoF order.  Keeping this
    kinematic map beside the material-slip estimate lets a controller recover
    in the *signed* two-dimensional tangent plane instead of adding an
    unrelated scalar preload along the finger closing ray.
    """

    position_jacobian_cube_local_m_per_rad: NDArray[np.float64]
    normal_force_n: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        jacobian = np.asarray(
            self.position_jacobian_cube_local_m_per_rad, dtype=np.float64
        )
        if (
            jacobian.ndim != 3
            or jacobian.shape[:2] != (len(ACTIVE_FINGERS), 2)
            or not np.isfinite(jacobian).all()
        ):
            raise ValueError(
                "position_jacobian_cube_local_m_per_rad must have shape "
                "(3, 2, active_dofs) and be finite"
            )
        force = np.asarray(self.normal_force_n, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        if (
            force.shape != (len(ACTIVE_FINGERS),)
            or valid.shape != (len(ACTIVE_FINGERS),)
            or not np.isfinite(force).all()
            or np.any(force < 0.0)
        ):
            raise ValueError("rolling tangent Jacobian force/valid arrays are invalid")
        jacobian = np.array(jacobian, copy=True)
        force = np.array(force, copy=True)
        valid = np.array(valid, copy=True)
        jacobian.setflags(write=False)
        force.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(
            self, "position_jacobian_cube_local_m_per_rad", jacobian
        )
        object.__setattr__(self, "normal_force_n", force)
        object.__setattr__(self, "valid", valid)


@dataclass(slots=True)
class _TrackedPatch:
    track_id: int
    patch_identity: str
    witness_cube_local_m: np.ndarray
    tangent_velocity_m_s: np.ndarray
    normal_force_n: float


@dataclass(slots=True)
class _FingerState:
    tracks: list[_TrackedPatch]
    valid: bool
    ever_valid: bool
    tangent_velocity_m_s: np.ndarray
    scalar_slip_speed_m_s: float
    centroid_cube_local_m: np.ndarray
    signed_displacement_m: np.ndarray
    cumulative_slip_m: float
    patch_switch_count: int


def _initial_finger_state() -> _FingerState:
    return _FingerState(
        tracks=[],
        valid=False,
        ever_valid=False,
        tangent_velocity_m_s=np.zeros(2, dtype=np.float64),
        scalar_slip_speed_m_s=0.0,
        centroid_cube_local_m=np.zeros(3, dtype=np.float64),
        signed_displacement_m=np.zeros(2, dtype=np.float64),
        cumulative_slip_m=0.0,
        patch_switch_count=0,
    )


class RollingAwareContactSlipEstimator:
    """Causal force-weighted estimator for three-finger material slip."""

    def __init__(
        self,
        target_faces: Sequence[str],
        *,
        settings: RollingSlipSettings | None = None,
        finger_count: int = len(ACTIVE_FINGERS),
    ) -> None:
        if isinstance(finger_count, bool) or int(finger_count) != finger_count:
            raise ValueError("finger_count must be a positive integer")
        self.finger_count = int(finger_count)
        if self.finger_count <= 0:
            raise ValueError("finger_count must be a positive integer")
        self.target_faces = _target_faces(target_faces, self.finger_count)
        self.settings = settings or RollingSlipSettings()
        self._states = [_initial_finger_state() for _ in range(self.finger_count)]
        self._previous_time_s: float | None = None
        self._next_track_id = 0

    def reset(self) -> None:
        self._states = [_initial_finger_state() for _ in range(self.finger_count)]
        self._previous_time_s = None
        self._next_track_id = 0

    def _accepted_samples(
        self, samples: Sequence[ContactPatchKinematics], finger: int
    ) -> list[ContactPatchKinematics]:
        face = self.target_faces[finger]
        axis, sign = FACE_AXIS_AND_SIGN[face]
        accepted = []
        for sample in samples:
            if _finger_index(sample.finger_index, self.finger_count) != finger:
                continue
            alignment = sign * float(sample.outward_normal_cube_local[axis])
            if (
                sample.normal_force_n
                + _EPSILON
                >= self.settings.minimum_sample_normal_force_n
                and alignment + _EPSILON >= self.settings.minimum_normal_alignment
            ):
                accepted.append(sample)
        accepted.sort(
            key=lambda value: (
                value.patch_identity,
                tuple(float(item) for item in value.witness_cube_local_m),
                value.source_contact_index,
            )
        )
        return accepted

    def _match(
        self,
        previous: Sequence[_TrackedPatch],
        current: Sequence[ContactPatchKinematics],
        tangent_axes: tuple[int, int],
    ) -> tuple[dict[int, _TrackedPatch], set[int]]:
        candidates: list[tuple[float, int, int, int]] = []
        for current_index, sample in enumerate(current):
            for previous_index, track in enumerate(previous):
                if track.patch_identity != sample.patch_identity:
                    continue
                delta = (
                    sample.witness_cube_local_m[list(tangent_axes)]
                    - track.witness_cube_local_m[list(tangent_axes)]
                )
                distance = float(np.linalg.norm(delta))
                if distance <= self.settings.maximum_patch_match_distance_m + _EPSILON:
                    candidates.append(
                        (distance, track.track_id, current_index, previous_index)
                    )
        matches: dict[int, _TrackedPatch] = {}
        used_previous: set[int] = set()
        for _, _, current_index, previous_index in sorted(candidates):
            if current_index in matches or previous_index in used_previous:
                continue
            matches[current_index] = previous[previous_index]
            used_previous.add(previous_index)
        return matches, used_previous

    def update(
        self,
        time_s: float,
        samples: Sequence[ContactPatchKinematics],
    ) -> RollingAwareSlipEstimate:
        """Consume one post-step contact frame and return current estimates."""

        time_value = float(time_s)
        if not math.isfinite(time_value) or time_value < 0.0:
            raise ValueError("time_s must be finite and nonnegative")
        if self._previous_time_s is not None and time_value <= self._previous_time_s:
            raise ValueError("time_s must increase strictly between updates")
        resolved_samples = tuple(samples)
        if any(
            not isinstance(value, ContactPatchKinematics)
            for value in resolved_samples
        ):
            raise TypeError("samples must contain ContactPatchKinematics")
        delta_time = (
            0.0
            if self._previous_time_s is None
            else time_value - self._previous_time_s
        )
        time_is_continuous = bool(
            self._previous_time_s is not None
            and delta_time <= self.settings.maximum_time_gap_s + _EPSILON
        )

        signed = np.zeros((self.finger_count, 2), dtype=np.float64)
        cumulative = np.zeros(self.finger_count, dtype=np.float64)
        velocity = np.zeros((self.finger_count, 2), dtype=np.float64)
        force = np.zeros(self.finger_count, dtype=np.float64)
        valid = np.zeros(self.finger_count, dtype=bool)
        continuous = np.zeros(self.finger_count, dtype=bool)
        rolling = np.zeros(self.finger_count, dtype=bool)
        rolling_fraction = np.zeros(self.finger_count, dtype=np.float64)
        patch_switch = np.zeros(self.finger_count, dtype=bool)
        centroid_jump = np.zeros(self.finger_count, dtype=bool)
        centroid_step = np.zeros(self.finger_count, dtype=np.float64)
        matched_count = np.zeros(self.finger_count, dtype=np.int64)
        new_count = np.zeros(self.finger_count, dtype=np.int64)
        dropped_count = np.zeros(self.finger_count, dtype=np.int64)
        switch_count = np.zeros(self.finger_count, dtype=np.int64)

        for finger_index, state in enumerate(self._states):
            axes = _tangent_axes(self.target_faces[finger_index])
            current = self._accepted_samples(resolved_samples, finger_index)
            total_force = float(sum(value.normal_force_n for value in current))
            current_valid = bool(
                current
                and total_force + _EPSILON
                >= self.settings.minimum_total_normal_force_n
            )
            matches, used_previous = self._match(state.tracks, current, axes)
            matched_count[finger_index] = len(matches)
            new_count[finger_index] = len(current) - len(matches)
            dropped_count[finger_index] = len(state.tracks) - len(used_previous)

            if current_valid:
                weights = np.asarray(
                    [sample.normal_force_n for sample in current], dtype=np.float64
                )
                tangent_velocities = np.asarray(
                    [
                        sample.relative_velocity_cube_local_m_s[list(axes)]
                        for sample in current
                    ],
                    dtype=np.float64,
                )
                current_velocity = np.average(
                    tangent_velocities, axis=0, weights=weights
                )
                scalar_speed = float(
                    np.average(
                        np.linalg.norm(tangent_velocities, axis=1), weights=weights
                    )
                )
                centroid = np.average(
                    np.asarray(
                        [sample.witness_cube_local_m for sample in current]
                    ),
                    axis=0,
                    weights=weights,
                )
            else:
                current_velocity = np.zeros(2, dtype=np.float64)
                scalar_speed = 0.0
                centroid = np.zeros(3, dtype=np.float64)

            frame_continuous = bool(
                time_is_continuous and state.valid and current_valid
            )
            continuous[finger_index] = frame_continuous
            if frame_continuous:
                state.signed_displacement_m += (
                    0.5
                    * (state.tangent_velocity_m_s + current_velocity)
                    * delta_time
                )
                state.cumulative_slip_m += (
                    0.5
                    * (state.scalar_slip_speed_m_s + scalar_speed)
                    * delta_time
                )
                tangent_delta = centroid[list(axes)] - state.centroid_cube_local_m[
                    list(axes)
                ]
                centroid_step[finger_index] = float(np.linalg.norm(tangent_delta))
                centroid_jump[finger_index] = bool(
                    centroid_step[finger_index]
                    > self.settings.centroid_jump_threshold_m + _EPSILON
                )

                rolling_force = 0.0
                for current_index, previous_track in matches.items():
                    sample = current[current_index]
                    witness_speed = float(
                        np.linalg.norm(
                            sample.witness_cube_local_m[list(axes)]
                            - previous_track.witness_cube_local_m[list(axes)]
                        )
                        / delta_time
                    )
                    sample_slip_speed = float(
                        np.linalg.norm(
                            sample.relative_velocity_cube_local_m_s[list(axes)]
                        )
                    )
                    if (
                        witness_speed + _EPSILON
                        >= self.settings.rolling_witness_speed_min_m_s
                        and sample_slip_speed
                        <= self.settings.rolling_relative_slip_speed_max_m_s
                        + _EPSILON
                    ):
                        rolling_force += sample.normal_force_n
                rolling_fraction[finger_index] = (
                    rolling_force / total_force if total_force > 0.0 else 0.0
                )
                rolling[finger_index] = bool(
                    rolling_fraction[finger_index] + _EPSILON
                    >= self.settings.rolling_force_fraction_min
                )

            switched = bool(
                current_valid
                and state.ever_valid
                and (
                    not state.valid
                    or new_count[finger_index] > 0
                    or dropped_count[finger_index] > 0
                )
            )
            if switched:
                state.patch_switch_count += 1
            patch_switch[finger_index] = switched

            next_tracks: list[_TrackedPatch] = []
            for current_index, sample in enumerate(current):
                previous_track = matches.get(current_index)
                if previous_track is None:
                    track_id = self._next_track_id
                    self._next_track_id += 1
                else:
                    track_id = previous_track.track_id
                next_tracks.append(
                    _TrackedPatch(
                        track_id=track_id,
                        patch_identity=sample.patch_identity,
                        witness_cube_local_m=np.array(
                            sample.witness_cube_local_m, copy=True
                        ),
                        tangent_velocity_m_s=np.array(
                            sample.relative_velocity_cube_local_m_s[list(axes)],
                            copy=True,
                        ),
                        normal_force_n=sample.normal_force_n,
                    )
                )
            next_tracks.sort(key=lambda value: value.track_id)
            state.tracks = next_tracks if current_valid else []
            state.valid = current_valid
            state.ever_valid = state.ever_valid or current_valid
            state.tangent_velocity_m_s = np.array(current_velocity, copy=True)
            state.scalar_slip_speed_m_s = scalar_speed
            state.centroid_cube_local_m = np.array(centroid, copy=True)

            signed[finger_index] = state.signed_displacement_m
            cumulative[finger_index] = state.cumulative_slip_m
            velocity[finger_index] = current_velocity
            force[finger_index] = total_force
            valid[finger_index] = current_valid
            switch_count[finger_index] = state.patch_switch_count

        self._previous_time_s = time_value
        return RollingAwareSlipEstimate(
            time_s=time_value,
            target_faces=self.target_faces,
            signed_tangent_displacement_m=signed,
            cumulative_irrecoverable_slip_m=cumulative,
            relative_tangent_velocity_m_s=velocity,
            normal_force_n=force,
            valid=valid,
            continuous=continuous,
            rolling_detected=rolling,
            rolling_force_fraction=rolling_fraction,
            patch_switch=patch_switch,
            centroid_jump=centroid_jump,
            centroid_tangent_step_m=centroid_step,
            matched_patch_count=matched_count,
            new_patch_count=new_count,
            dropped_patch_count=dropped_count,
            patch_switch_count=switch_count,
        )


def _point_velocity_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    point_world_m: np.ndarray,
    body_id: int,
) -> np.ndarray:
    jacobian = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacobian, None, point_world_m, body_id)
    return jacobian @ np.asarray(data.qvel, dtype=np.float64)


def mujoco_contact_patch_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    cube_geom_id: int,
    finger_geom_to_index: Mapping[int, int],
) -> tuple[ContactPatchKinematics, ...]:
    """Extract real material-point relative velocities from ``mjContact``.

    The caller supplies only distal finger collision geoms.  This makes the
    adapter independent of ``ModelInfo`` and prevents non-distal contacts from
    silently becoming valid fingertip slip evidence.
    """

    if isinstance(cube_geom_id, (bool, np.bool_)):
        raise ValueError("cube_geom_id must be an integer")
    cube_geom = int(cube_geom_id)
    if cube_geom != cube_geom_id or not 0 <= cube_geom < model.ngeom:
        raise ValueError("cube_geom_id is outside the model")
    mapping: dict[int, int] = {}
    for geom_value, finger_value in finger_geom_to_index.items():
        if isinstance(geom_value, (bool, np.bool_)):
            raise ValueError("finger geom ids must be integers")
        geom = int(geom_value)
        if geom != geom_value or not 0 <= geom < model.ngeom or geom == cube_geom:
            raise ValueError("finger geom id is outside the model or is the cube")
        finger = _finger_index(finger_value, len(ACTIVE_FINGERS))
        mapping[geom] = finger
    if not mapping:
        raise ValueError("finger_geom_to_index must not be empty")

    cube_body = int(model.geom_bodyid[cube_geom])
    cube_rotation = np.asarray(data.geom_xmat[cube_geom], dtype=np.float64).reshape(
        3, 3
    )
    cube_position = np.asarray(data.geom_xpos[cube_geom], dtype=np.float64)
    contact_force = np.zeros(6, dtype=np.float64)
    result: list[ContactPatchKinematics] = []
    for contact_index, contact in enumerate(data.contact[: data.ncon]):
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if cube_geom not in (geom1, geom2):
            continue
        other_geom = geom2 if geom1 == cube_geom else geom1
        if other_geom not in mapping or int(contact.efc_address) < 0:
            continue
        contact_force[:] = 0.0
        mujoco.mj_contactForce(model, data, contact_index, contact_force)
        normal_force = max(0.0, float(contact_force[0]))
        normal_world = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)[0]
        witness = surface_witness(
            np.asarray(contact.pos, dtype=np.float64),
            float(contact.dist),
            normal_world,
            cube_is_geom1=geom1 == cube_geom,
        )
        contact_point = np.asarray(contact.pos, dtype=np.float64)
        finger_body = int(model.geom_bodyid[other_geom])
        relative_world = _point_velocity_world(
            model, data, contact_point, finger_body
        ) - _point_velocity_world(model, data, contact_point, cube_body)
        result.append(
            ContactPatchKinematics(
                finger_index=mapping[other_geom],
                patch_identity=f"cube_geom_{cube_geom}|finger_geom_{other_geom}",
                witness_cube_local_m=cube_rotation.T
                @ (witness.position_world - cube_position),
                outward_normal_cube_local=cube_rotation.T
                @ witness.outward_normal_world,
                relative_velocity_cube_local_m_s=cube_rotation.T @ relative_world,
                normal_force_n=normal_force,
                source_contact_index=contact_index,
            )
        )
    result.sort(
        key=lambda value: (
            value.finger_index,
            value.patch_identity,
            tuple(float(item) for item in value.witness_cube_local_m),
            value.source_contact_index,
        )
    )
    return tuple(result)


def mujoco_contact_tangent_position_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    samples: Sequence[ContactPatchKinematics],
    *,
    cube_geom_id: int,
    target_faces: Sequence[str],
    active_dof_adrs: ArrayLike,
    minimum_total_normal_force_n: float = 0.05,
    minimum_normal_alignment: float = 0.95,
) -> RollingTangentJacobian:
    """Return the live signed 2-D contact map for the active finger DoFs.

    Each sample is evaluated at the exact same world contact point on the
    finger and cube.  The relative point Jacobian is rotated into the cube
    frame and projected onto the two axes tangent to that finger's configured
    target face.  Multiple manifold points are normal-force weighted; their
    creation, deletion or reordering cannot create a position error by itself.
    """

    cube_geom = int(cube_geom_id)
    if (
        isinstance(cube_geom_id, (bool, np.bool_))
        or cube_geom != cube_geom_id
        or not 0 <= cube_geom < model.ngeom
    ):
        raise ValueError("cube_geom_id is outside the model")
    faces = _target_faces(target_faces, len(ACTIVE_FINGERS))
    dofs = np.asarray(active_dof_adrs, dtype=np.int64)
    if (
        dofs.ndim != 1
        or dofs.size == 0
        or np.any(dofs < 0)
        or np.any(dofs >= model.nv)
        or np.unique(dofs).size != dofs.size
    ):
        raise ValueError("active_dof_adrs must contain unique valid DoF indices")
    minimum_force = _positive(
        minimum_total_normal_force_n, "minimum_total_normal_force_n"
    )
    alignment_minimum = float(minimum_normal_alignment)
    if not math.isfinite(alignment_minimum) or not 0.0 < alignment_minimum <= 1.0:
        raise ValueError("minimum_normal_alignment must lie in (0, 1]")

    cube_body = int(model.geom_bodyid[cube_geom])
    cube_rotation = np.asarray(
        data.geom_xmat[cube_geom], dtype=np.float64
    ).reshape(3, 3)
    weighted = np.zeros(
        (len(ACTIVE_FINGERS), 2, dofs.size), dtype=np.float64
    )
    force = np.zeros(len(ACTIVE_FINGERS), dtype=np.float64)
    for sample in samples:
        finger = _finger_index(sample.finger_index, len(ACTIVE_FINGERS))
        axis, sign = FACE_AXIS_AND_SIGN[faces[finger]]
        if (
            sample.normal_force_n <= 0.0
            or sign * float(sample.outward_normal_cube_local[axis])
            + _EPSILON
            < alignment_minimum
        ):
            continue
        contact_index = int(sample.source_contact_index)
        if not 0 <= contact_index < data.ncon:
            raise ValueError("sample source_contact_index is not active")
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if cube_geom not in (geom1, geom2):
            raise ValueError("sample source contact does not contain the cube")
        other_geom = geom2 if geom1 == cube_geom else geom1
        finger_body = int(model.geom_bodyid[other_geom])
        point_world = np.asarray(contact.pos, dtype=np.float64)
        cube_jacobian = np.zeros((3, model.nv), dtype=np.float64)
        finger_jacobian = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(
            model, data, cube_jacobian, None, point_world, cube_body
        )
        mujoco.mj_jac(
            model, data, finger_jacobian, None, point_world, finger_body
        )
        relative_local = cube_rotation.T @ (
            finger_jacobian - cube_jacobian
        )
        tangent_axes = _tangent_axes(faces[finger])
        weighted[finger] += sample.normal_force_n * relative_local[
            list(tangent_axes)
        ][:, dofs]
        force[finger] += sample.normal_force_n

    valid = force >= minimum_force - _EPSILON
    for finger in range(len(ACTIVE_FINGERS)):
        if valid[finger]:
            weighted[finger] /= force[finger]
        else:
            weighted[finger] = 0.0
    return RollingTangentJacobian(weighted, force, valid)


__all__ = [
    "ROLLING_CONTACT_SLIP_SCHEMA_VERSION",
    "ContactPatchKinematics",
    "RollingAwareContactSlipEstimator",
    "RollingAwareSlipEstimate",
    "RollingSlipSettings",
    "RollingTangentJacobian",
    "mujoco_contact_patch_kinematics",
    "mujoco_contact_tangent_position_jacobian",
]
