"""Pure NumPy contact-topology primitives for the three-finger cube task.

The MuJoCo adapter is deliberately kept outside this module.  Callers extract
``mjContact`` values and normal forces, while this module owns the geometry,
force aggregation, and time-series acceptance semantics.  This separation
makes the strict face-contact contract testable without constructing a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


SURFACE_TOLERANCE_M = 50e-6
EDGE_MARGIN_M = 0.5e-3
NORMAL_ALIGNMENT_MIN = 0.95
TARGET_FORCE_PURITY_MIN = 0.95

FINGER_ORDER = ("thumb", "index", "mid")
_NUMERIC_TOLERANCE = 1e-12


class Face(IntEnum):
    """Stable face-force array indices.

    The six physical faces intentionally precede the two non-face categories.
    Keep this order stable because it is also the last dimension of persisted
    ``face_force_n`` arrays.
    """

    X_POS = 0
    X_NEG = 1
    Y_POS = 2
    Y_NEG = 3
    Z_POS = 4
    Z_NEG = 5
    EDGE_CORNER = 6
    UNKNOWN = 7

    @property
    def is_physical(self) -> bool:
        return self.value < Face.EDGE_CORNER.value

    @property
    def axis(self) -> int:
        if not self.is_physical:
            raise ValueError(f"{self.name} is not a physical box face")
        return self.value // 2

    @property
    def sign(self) -> int:
        if not self.is_physical:
            raise ValueError(f"{self.name} is not a physical box face")
        return 1 if self.value % 2 == 0 else -1

    @property
    def outward_normal(self) -> NDArray[np.float64]:
        normal = np.zeros(3, dtype=np.float64)
        normal[self.axis] = self.sign
        return normal


FACE_ORDER = tuple(Face)
PHYSICAL_FACES = FACE_ORDER[:6]
FACE_COUNT = len(FACE_ORDER)


def opposite_face(face: Face | int) -> Face:
    """Return the opposite physical face.

    Edge/corner and unknown observations do not have an opposite and are
    rejected instead of being silently mapped to a physical face.
    """

    result = Face(face)
    if not result.is_physical:
        raise ValueError(f"{result.name} does not have an opposite face")
    return Face(result.value ^ 1)


def _vector3(values: ArrayLike, label: str) -> NDArray[np.float64]:
    vector = np.array(values, dtype=np.float64, copy=True)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain three finite values")
    return vector


def _unit_vector3(values: ArrayLike, label: str) -> NDArray[np.float64]:
    vector = _vector3(values, label)
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{label} must have non-zero length")
    return vector / norm


def _readonly(array: ArrayLike, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SurfaceWitness:
    """The closest point on the cube and its cube-outward world normal."""

    position_world: NDArray[np.float64]
    outward_normal_world: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_world",
            _readonly(_vector3(self.position_world, "position_world")),
        )
        object.__setattr__(
            self,
            "outward_normal_world",
            _readonly(
                _unit_vector3(self.outward_normal_world, "outward_normal_world")
            ),
        )


def surface_witness(
    contact_position_world: ArrayLike,
    contact_distance_m: float,
    contact_normal_geom1_to_geom2_world: ArrayLike,
    *,
    cube_is_geom1: bool,
) -> SurfaceWitness:
    """Recover the cube-side nearest point from a MuJoCo contact midpoint.

    MuJoCo stores ``contact.pos`` at the midpoint between the two nearest geom
    points and stores a unit normal directed from geom1 to geom2.  The formula
    works for separation as well as negative-distance penetration and remains
    invariant when the cube appears as geom1 or geom2.
    """

    position = _vector3(contact_position_world, "contact_position_world")
    distance = float(contact_distance_m)
    if not np.isfinite(distance):
        raise ValueError("contact_distance_m must be finite")
    contact_normal = _unit_vector3(
        contact_normal_geom1_to_geom2_world,
        "contact_normal_geom1_to_geom2_world",
    )
    outward_normal = contact_normal if cube_is_geom1 else -contact_normal
    cube_point = position - 0.5 * distance * outward_normal
    return SurfaceWitness(cube_point, outward_normal)


@dataclass(frozen=True, slots=True)
class BoxContactThresholds:
    """Geometric thresholds for clean box-face attribution."""

    surface_tolerance_m: float = SURFACE_TOLERANCE_M
    edge_margin_m: float = EDGE_MARGIN_M
    normal_alignment_min: float = NORMAL_ALIGNMENT_MIN

    def __post_init__(self) -> None:
        surface_tolerance = float(self.surface_tolerance_m)
        edge_margin = float(self.edge_margin_m)
        normal_alignment = float(self.normal_alignment_min)
        if not np.isfinite(surface_tolerance) or surface_tolerance <= 0:
            raise ValueError("surface_tolerance_m must be positive and finite")
        if not np.isfinite(edge_margin) or edge_margin <= 0:
            raise ValueError("edge_margin_m must be positive and finite")
        if surface_tolerance >= edge_margin:
            raise ValueError("surface_tolerance_m must be smaller than edge_margin_m")
        if not np.isfinite(normal_alignment) or not 0 < normal_alignment <= 1:
            raise ValueError("normal_alignment_min must be within (0, 1]")


DEFAULT_BOX_CONTACT_THRESHOLDS = BoxContactThresholds()


@dataclass(frozen=True, slots=True)
class BoxContactClassification:
    """Result of assigning one cube-side witness to a face category."""

    face: Face
    candidate_faces: tuple[Face, ...]
    surface_error_m: float
    edge_clearance_m: float
    normal_alignment: float


def classify_box_contact(
    surface_point_local: ArrayLike,
    outward_normal_local: ArrayLike,
    half_extents_m: ArrayLike,
    *,
    thresholds: BoxContactThresholds = DEFAULT_BOX_CONTACT_THRESHOLDS,
) -> BoxContactClassification:
    """Classify a cube-side witness as one clean face, an edge, or unknown.

    A force on an edge or corner is never split between adjacent faces.  A
    single candidate is also demoted to ``EDGE_CORNER`` when it lies inside the
    configured edge-exclusion band.  The normal gate is signed because
    ``surface_witness`` has already oriented the normal away from the cube.
    """

    point = _vector3(surface_point_local, "surface_point_local")
    normal = _unit_vector3(outward_normal_local, "outward_normal_local")
    half_extents = _vector3(half_extents_m, "half_extents_m")
    if np.any(half_extents <= 0):
        raise ValueError("half_extents_m must be positive")
    if thresholds.edge_margin_m >= float(np.min(half_extents)):
        raise ValueError("edge_margin_m must be smaller than every half extent")

    candidates: list[Face] = []
    candidate_errors: dict[Face, float] = {}
    absolute_point = np.abs(point)
    for face in PHYSICAL_FACES:
        axis = face.axis
        tangential_axes = [index for index in range(3) if index != axis]
        error = abs(float(point[axis]) - face.sign * float(half_extents[axis]))
        inside_tangential_bounds = bool(
            np.all(
                absolute_point[tangential_axes]
                <= half_extents[tangential_axes]
                + thresholds.surface_tolerance_m
                + _NUMERIC_TOLERANCE
            )
        )
        if (
            error <= thresholds.surface_tolerance_m + _NUMERIC_TOLERANCE
            and inside_tangential_bounds
        ):
            candidates.append(face)
            candidate_errors[face] = error

    all_plane_errors = [
        abs(float(point[face.axis]) - face.sign * float(half_extents[face.axis]))
        for face in PHYSICAL_FACES
    ]
    minimum_surface_error = float(min(all_plane_errors))
    global_clearance = float(np.min(half_extents - absolute_point))

    if len(candidates) >= 2:
        alignment = max(float(normal @ face.outward_normal) for face in candidates)
        return BoxContactClassification(
            face=Face.EDGE_CORNER,
            candidate_faces=tuple(candidates),
            surface_error_m=min(candidate_errors.values()),
            edge_clearance_m=global_clearance,
            normal_alignment=alignment,
        )
    if not candidates:
        return BoxContactClassification(
            face=Face.UNKNOWN,
            candidate_faces=(),
            surface_error_m=minimum_surface_error,
            edge_clearance_m=global_clearance,
            normal_alignment=float("nan"),
        )

    candidate = candidates[0]
    tangential_axes = [index for index in range(3) if index != candidate.axis]
    edge_clearance = float(
        np.min(half_extents[tangential_axes] - absolute_point[tangential_axes])
    )
    alignment = float(normal @ candidate.outward_normal)
    if edge_clearance + _NUMERIC_TOLERANCE < thresholds.edge_margin_m:
        category = Face.EDGE_CORNER
    elif alignment + _NUMERIC_TOLERANCE < thresholds.normal_alignment_min:
        category = Face.UNKNOWN
    else:
        category = candidate
    return BoxContactClassification(
        face=category,
        candidate_faces=(candidate,),
        surface_error_m=candidate_errors[candidate],
        edge_clearance_m=edge_clearance,
        normal_alignment=alignment,
    )


@dataclass(frozen=True, slots=True)
class ContactForceSample:
    """One active cube/controlled-finger contact in a simulation frame."""

    finger_index: int
    normal_force_n: float
    face: Face | BoxContactClassification | None = None
    distal: bool = True
    surface_witness_position_world_m: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.surface_witness_position_world_m is not None:
            object.__setattr__(
                self,
                "surface_witness_position_world_m",
                _readonly(
                    _vector3(
                        self.surface_witness_position_world_m,
                        "surface_witness_position_world_m",
                    )
                ),
            )


@dataclass(frozen=True, slots=True)
class FrameContactAggregate:
    """Fixed-shape force evidence for one simulation frame."""

    face_force_n: NDArray[np.float64]
    active_nondistal_force_n: NDArray[np.float64]
    face_position_moment_n_m: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        face_force = np.array(self.face_force_n, dtype=np.float64, copy=True)
        nondistal_force = np.array(
            self.active_nondistal_force_n, dtype=np.float64, copy=True
        )
        if face_force.ndim != 2 or face_force.shape[1] != FACE_COUNT:
            raise ValueError(f"face_force_n must have shape (fingers, {FACE_COUNT})")
        if nondistal_force.shape != (face_force.shape[0],):
            raise ValueError("active_nondistal_force_n must match the finger dimension")
        if self.face_position_moment_n_m is None:
            position_moment = np.zeros(face_force.shape + (3,), dtype=np.float64)
        else:
            position_moment = np.array(
                self.face_position_moment_n_m, dtype=np.float64, copy=True
            )
        if position_moment.shape != face_force.shape + (3,):
            raise ValueError(
                "face_position_moment_n_m must have shape "
                f"{face_force.shape + (3,)}"
            )
        if (
            not np.isfinite(face_force).all()
            or not np.isfinite(nondistal_force).all()
            or not np.isfinite(position_moment).all()
            or np.any(face_force < 0)
            or np.any(nondistal_force < 0)
        ):
            raise ValueError(
                "aggregated forces must be finite and non-negative and position "
                "moments must be finite"
            )
        object.__setattr__(self, "face_force_n", _readonly(face_force))
        object.__setattr__(
            self, "active_nondistal_force_n", _readonly(nondistal_force)
        )
        object.__setattr__(
            self, "face_position_moment_n_m", _readonly(position_moment)
        )


def aggregate_contact_frame(
    samples: Iterable[ContactForceSample],
    *,
    finger_count: int = len(FINGER_ORDER),
) -> FrameContactAggregate:
    """Sum normal force by finger and category without a per-point cutoff."""

    if isinstance(finger_count, bool) or int(finger_count) != finger_count:
        raise ValueError("finger_count must be a positive integer")
    finger_count = int(finger_count)
    if finger_count <= 0:
        raise ValueError("finger_count must be a positive integer")
    face_force = np.zeros((finger_count, FACE_COUNT), dtype=np.float64)
    position_moment = np.zeros((finger_count, FACE_COUNT, 3), dtype=np.float64)
    nondistal_force = np.zeros(finger_count, dtype=np.float64)

    for sample in samples:
        finger_index = int(sample.finger_index)
        if isinstance(sample.finger_index, bool) or finger_index != sample.finger_index:
            raise ValueError("finger_index must be an integer")
        if not 0 <= finger_index < finger_count:
            raise ValueError("finger_index is outside the configured finger range")
        force = float(sample.normal_force_n)
        if not np.isfinite(force) or force < 0:
            raise ValueError("normal_force_n must be finite and non-negative")
        if not sample.distal:
            nondistal_force[finger_index] += force
            continue
        if sample.face is None:
            raise ValueError("a distal contact requires a face classification")
        face = (
            sample.face.face
            if isinstance(sample.face, BoxContactClassification)
            else Face(sample.face)
        )
        face_force[finger_index, int(face)] += force
        if sample.surface_witness_position_world_m is not None:
            position_moment[finger_index, int(face)] += (
                force * sample.surface_witness_position_world_m
            )

    return FrameContactAggregate(face_force, nondistal_force, position_moment)


def target_face_contact_centroids(
    face_force_n: ArrayLike,
    face_position_moment_n_m: ArrayLike,
    target_faces: Sequence[Face | int],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return force-weighted world centroids for three target faces.

    Inputs describe one simulation frame.  The returned arrays have shapes
    ``(3, 3)`` and ``(3,)`` in ``FINGER_ORDER``.  A target face with zero force
    has a zero centroid and a false validity mask; a missing contact therefore
    cannot accidentally look geometrically aligned.
    """

    face_force = np.asarray(face_force_n, dtype=np.float64)
    expected_force_shape = (len(FINGER_ORDER), FACE_COUNT)
    if face_force.shape != expected_force_shape:
        raise ValueError(f"face_force_n must have shape {expected_force_shape}")
    if not np.isfinite(face_force).all() or np.any(face_force < 0):
        raise ValueError("face_force_n must be finite and non-negative")

    position_moment = np.asarray(face_position_moment_n_m, dtype=np.float64)
    expected_moment_shape = expected_force_shape + (3,)
    if position_moment.shape != expected_moment_shape:
        raise ValueError(
            "face_position_moment_n_m must have shape "
            f"{expected_moment_shape}"
        )
    if not np.isfinite(position_moment).all():
        raise ValueError("face_position_moment_n_m must be finite")

    targets = validate_three_finger_target_faces(target_faces)
    target_indices = np.fromiter(
        (int(face) for face in targets), dtype=np.intp, count=len(FINGER_ORDER)
    )
    finger_indices = np.arange(len(FINGER_ORDER), dtype=np.intp)
    target_force = face_force[finger_indices, target_indices]
    target_moment = position_moment[finger_indices, target_indices]
    valid = target_force > 0.0
    centroids = np.divide(
        target_moment,
        target_force[:, np.newaxis],
        out=np.zeros_like(target_moment),
        where=valid[:, np.newaxis],
    )
    return _readonly(centroids), _readonly(valid, dtype=np.bool_)


def three_finger_height_spread(
    contact_centroid_world_m: ArrayLike,
    contact_centroid_valid: ArrayLike,
    gravity_world_m_s2: ArrayLike,
) -> tuple[float, bool]:
    """Compute the three-centroid span along the gravity-defined up axis.

    Height is ``-(gravity / |gravity|) dot position``.  When any finger lacks
    a valid centroid, the function returns ``(0.0, False)`` rather than a
    partial-finger span.  This zero-plus-mask convention keeps persisted traces
    finite while ensuring that invalid frames cannot satisfy alignment gates.
    """

    centroids = np.asarray(contact_centroid_world_m, dtype=np.float64)
    expected_centroid_shape = (len(FINGER_ORDER), 3)
    if centroids.shape != expected_centroid_shape:
        raise ValueError(
            "contact_centroid_world_m must have shape "
            f"{expected_centroid_shape}"
        )
    if not np.isfinite(centroids).all():
        raise ValueError("contact_centroid_world_m must be finite")
    valid = np.asarray(contact_centroid_valid, dtype=np.bool_)
    if valid.shape != (len(FINGER_ORDER),):
        raise ValueError(
            f"contact_centroid_valid must have shape {(len(FINGER_ORDER),)}"
        )
    gravity = _unit_vector3(gravity_world_m_s2, "gravity_world_m_s2")
    if not bool(np.all(valid)):
        return 0.0, False
    heights = centroids @ -gravity
    return float(np.max(heights) - np.min(heights)), True


def validate_three_finger_target_faces(
    target_faces: Sequence[Face | int],
) -> tuple[Face, Face, Face]:
    """Validate ``(thumb, index, middle)`` as one opposing face pair."""

    if len(target_faces) != len(FINGER_ORDER):
        raise ValueError("target_faces must follow (thumb, index, mid)")
    thumb, index, middle = (Face(face) for face in target_faces)
    if not all(face.is_physical for face in (thumb, index, middle)):
        raise ValueError("target faces must be physical box faces")
    if index != middle:
        raise ValueError("index and middle target faces must be identical")
    if opposite_face(thumb) != index:
        raise ValueError("thumb target face must oppose index and middle")
    return thumb, index, middle


@dataclass(frozen=True, slots=True)
class FaceTraceThresholds:
    """Strict hold-window acceptance thresholds."""

    contact_force_min_n: float = 0.05
    touch_force_min_n: float = 1e-8
    target_force_purity_min: float = TARGET_FORCE_PURITY_MIN
    finger_contact_duty_min: float = 0.80
    simultaneous_contact_duty_min: float = 0.70
    material_violation_force_min_n: float = 0.05
    max_off_target_duty: float = 0.01
    max_off_target_run_s: float = 0.010
    max_nondistal_duty: float = 0.01
    max_nondistal_run_s: float = 0.010

    def __post_init__(self) -> None:
        positive = {
            "contact_force_min_n": self.contact_force_min_n,
            "material_violation_force_min_n": self.material_violation_force_min_n,
        }
        nonnegative = {
            "touch_force_min_n": self.touch_force_min_n,
            "max_off_target_run_s": self.max_off_target_run_s,
            "max_nondistal_run_s": self.max_nondistal_run_s,
        }
        proportions = {
            "target_force_purity_min": self.target_force_purity_min,
            "finger_contact_duty_min": self.finger_contact_duty_min,
            "simultaneous_contact_duty_min": self.simultaneous_contact_duty_min,
            "max_off_target_duty": self.max_off_target_duty,
            "max_nondistal_duty": self.max_nondistal_duty,
        }
        for label, value in positive.items():
            if not np.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{label} must be positive and finite")
        for label, value in nonnegative.items():
            if not np.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{label} must be non-negative and finite")
        for label, value in proportions.items():
            if not np.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{label} must be within [0, 1]")
        if self.target_force_purity_min <= 0:
            raise ValueError("target_force_purity_min must be greater than zero")


DEFAULT_FACE_TRACE_THRESHOLDS = FaceTraceThresholds()


@dataclass(frozen=True, slots=True)
class FaceTraceEvaluation:
    """Metrics, per-frame evidence, and hard checks for a face-force trace."""

    passed: bool
    failed_checks: tuple[str, ...]
    checks: Mapping[str, bool]
    target_force_n: NDArray[np.float64]
    off_target_force_n: NDArray[np.float64]
    target_force_purity: NDArray[np.float64]
    effective_contact: NDArray[np.bool_]
    material_off_target: NDArray[np.bool_]
    material_active_nondistal: NDArray[np.bool_]
    finger_target_duty: NDArray[np.float64]
    simultaneous_target_duty: float
    off_target_duty: NDArray[np.float64]
    any_off_target_duty: float
    longest_off_target_run_s: float
    nondistal_duty: NDArray[np.float64]
    any_nondistal_duty: float
    longest_nondistal_run_s: float


def _force_trace(values: ArrayLike, shape: tuple[int, ...], label: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _longest_true_run_steps(mask: NDArray[np.bool_]) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def evaluate_face_trace(
    face_force_n: ArrayLike,
    tactile_force_n: ArrayLike,
    target_faces: Sequence[Face | int],
    timestep_s: float,
    *,
    active_nondistal_force_n: ArrayLike | None = None,
    thresholds: FaceTraceThresholds = DEFAULT_FACE_TRACE_THRESHOLDS,
) -> FaceTraceEvaluation:
    """Evaluate one uniformly sampled face-contact window.

    ``face_force_n`` has shape ``(steps, 3, 8)`` in ``FACE_ORDER``.  Target
    duty is based on force, tactile evidence, and per-frame 95% target-face
    purity.  Edge/corner and unknown force remains entirely off-target.
    Material off-target and controlled-finger non-distal contacts are also
    constrained by aggregate duty and the longest consecutive run.
    """

    face_force = np.asarray(face_force_n, dtype=np.float64)
    if face_force.ndim != 3 or face_force.shape[1:] != (
        len(FINGER_ORDER),
        FACE_COUNT,
    ):
        raise ValueError(
            f"face_force_n must have shape (steps, {len(FINGER_ORDER)}, {FACE_COUNT})"
        )
    step_count = face_force.shape[0]
    if step_count <= 0:
        raise ValueError("face_force_n must contain at least one step")
    if not np.isfinite(face_force).all() or np.any(face_force < 0):
        raise ValueError("face_force_n must be finite and non-negative")
    tactile = _force_trace(
        tactile_force_n,
        (step_count, len(FINGER_ORDER)),
        "tactile_force_n",
    )
    if active_nondistal_force_n is None:
        nondistal = np.zeros((step_count, len(FINGER_ORDER)), dtype=np.float64)
    else:
        nondistal = _force_trace(
            active_nondistal_force_n,
            (step_count, len(FINGER_ORDER)),
            "active_nondistal_force_n",
        )
    timestep = float(timestep_s)
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("timestep_s must be positive and finite")
    targets = validate_three_finger_target_faces(target_faces)

    target_force = np.column_stack(
        [face_force[:, index, int(face)] for index, face in enumerate(targets)]
    )
    total_distal_force = np.sum(face_force, axis=2)
    off_target_force = total_distal_force - target_force
    target_purity = np.divide(
        target_force,
        total_distal_force,
        out=np.zeros_like(target_force),
        where=total_distal_force > 0,
    )
    effective = (
        target_force + _NUMERIC_TOLERANCE >= thresholds.contact_force_min_n
    ) & (tactile + _NUMERIC_TOLERANCE >= thresholds.touch_force_min_n) & (
        target_purity + _NUMERIC_TOLERANCE >= thresholds.target_force_purity_min
    )

    off_target_fraction = np.divide(
        off_target_force,
        total_distal_force,
        out=np.zeros_like(off_target_force),
        where=total_distal_force > 0,
    )
    material_off_target = (
        off_target_force + _NUMERIC_TOLERANCE
        >= thresholds.material_violation_force_min_n
    ) & (
        off_target_fraction
        > (1.0 - thresholds.target_force_purity_min) + _NUMERIC_TOLERANCE
    )

    total_with_nondistal = total_distal_force + nondistal
    nondistal_fraction = np.divide(
        nondistal,
        total_with_nondistal,
        out=np.zeros_like(nondistal),
        where=total_with_nondistal > 0,
    )
    material_nondistal = (
        nondistal + _NUMERIC_TOLERANCE
        >= thresholds.material_violation_force_min_n
    ) & (
        nondistal_fraction
        > (1.0 - thresholds.target_force_purity_min) + _NUMERIC_TOLERANCE
    )

    finger_target_duty = effective.mean(axis=0)
    simultaneous_target_duty = float(np.all(effective, axis=1).mean())
    off_target_duty = material_off_target.mean(axis=0)
    any_off_target = np.any(material_off_target, axis=1)
    any_off_target_duty = float(any_off_target.mean())
    longest_off_target_run_s = _longest_true_run_steps(any_off_target) * timestep
    nondistal_duty = material_nondistal.mean(axis=0)
    any_nondistal = np.any(material_nondistal, axis=1)
    any_nondistal_duty = float(any_nondistal.mean())
    longest_nondistal_run_s = _longest_true_run_steps(any_nondistal) * timestep

    checks: dict[str, bool] = {
        f"{finger}_target_face_duty": bool(
            finger_target_duty[index] + _NUMERIC_TOLERANCE
            >= thresholds.finger_contact_duty_min
        )
        for index, finger in enumerate(FINGER_ORDER)
    }
    checks.update(
        {
            "simultaneous_target_face_duty": bool(
                simultaneous_target_duty + _NUMERIC_TOLERANCE
                >= thresholds.simultaneous_contact_duty_min
            ),
            "off_target_contact_duty": bool(
                any_off_target_duty
                <= thresholds.max_off_target_duty + _NUMERIC_TOLERANCE
            ),
            "off_target_contact_run": bool(
                longest_off_target_run_s
                <= thresholds.max_off_target_run_s + _NUMERIC_TOLERANCE
            ),
            "active_nondistal_contact_duty": bool(
                any_nondistal_duty
                <= thresholds.max_nondistal_duty + _NUMERIC_TOLERANCE
            ),
            "active_nondistal_contact_run": bool(
                longest_nondistal_run_s
                <= thresholds.max_nondistal_run_s + _NUMERIC_TOLERANCE
            ),
        }
    )
    failed_checks = tuple(name for name, passed in checks.items() if not passed)
    return FaceTraceEvaluation(
        passed=not failed_checks,
        failed_checks=failed_checks,
        checks=checks,
        target_force_n=_readonly(target_force),
        off_target_force_n=_readonly(off_target_force),
        target_force_purity=_readonly(target_purity),
        effective_contact=_readonly(effective, dtype=np.bool_),
        material_off_target=_readonly(material_off_target, dtype=np.bool_),
        material_active_nondistal=_readonly(material_nondistal, dtype=np.bool_),
        finger_target_duty=_readonly(finger_target_duty),
        simultaneous_target_duty=simultaneous_target_duty,
        off_target_duty=_readonly(off_target_duty),
        any_off_target_duty=any_off_target_duty,
        longest_off_target_run_s=longest_off_target_run_s,
        nondistal_duty=_readonly(nondistal_duty),
        any_nondistal_duty=any_nondistal_duty,
        longest_nondistal_run_s=longest_nondistal_run_s,
    )


__all__ = [
    "DEFAULT_BOX_CONTACT_THRESHOLDS",
    "DEFAULT_FACE_TRACE_THRESHOLDS",
    "EDGE_MARGIN_M",
    "FACE_COUNT",
    "FACE_ORDER",
    "FINGER_ORDER",
    "NORMAL_ALIGNMENT_MIN",
    "PHYSICAL_FACES",
    "SURFACE_TOLERANCE_M",
    "TARGET_FORCE_PURITY_MIN",
    "BoxContactClassification",
    "BoxContactThresholds",
    "ContactForceSample",
    "Face",
    "FaceTraceEvaluation",
    "FaceTraceThresholds",
    "FrameContactAggregate",
    "SurfaceWitness",
    "aggregate_contact_frame",
    "classify_box_contact",
    "evaluate_face_trace",
    "opposite_face",
    "surface_witness",
    "target_face_contact_centroids",
    "three_finger_height_spread",
    "validate_three_finger_target_faces",
]
