"""Deterministic schema-v15 near-zero joint-pair campaign primitives.

The expensive MuJoCo runner deliberately lives behind these pure, auditable
primitives.  This module fixes the authenticated schema-v14 seed, the declared
search budget, local 14-variable sampling and the final contact/alignment-first
rank.  A runner may resume or shard work without changing candidate identity.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..artifacts import file_sha256
from ..config import ACTIVE_ACTUATORS
from ..grasp_pose import canonical_sha256
from ..viewer import resolve_viewer_source
from .contact_preserving_candidate_artifacts import (
    authenticate_v14_candidate_artifacts,
)


EXPERIMENT_ID = (
    "left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift"
)
SOURCE_CANDIDATE_ID = 15941124607131459
SOURCE_CAMPAIGN_ROOT = Path(
    "artifacts/left_opposed_face_palm_down_joint_pair_aligned_"
    "contact_preserving_planned_lift/tune/"
    "index_middle_joint1_near_zero_alignment_refinement_v2"
)
SOURCE_MEMBER = SOURCE_CAMPAIGN_ROOT / "final" / (
    f"candidate_{SOURCE_CANDIDATE_ID}"
)
SOURCE_CATALOG = (
    SOURCE_CAMPAIGN_ROOT / "catalogs/target_1/manipulation/catalog.json"
)
SOURCE_CONFIG_SHA256 = (
    "85e7bb0aa4bf7e5b97e5e6019f5171db829573f28c52142c1a60ea9de9703cf4"
)
SOURCE_RESULT_SHA256 = (
    "5f9c2960ddd161a82058fa0771b7adce9ba052136140dbdcd2d5c0a1dd4c851f"
)
SOURCE_TRACE_SHA256 = (
    "a4d2ad90f34f5b3f867c57ad7b5a13e07b9e155e3c6a5d4538f305440b7b0fc7"
)
SOURCE_CATALOG_SHA256 = (
    "e9776b128f5c25ed401283916ccdd9d2ed523a5a4ff037130624a7ee95316c67"
)
SEED = 20260821
VARIABLE_NAMES = (
    *ACTIVE_ACTUATORS,
    "root_delta_cube_m.x",
    "root_delta_cube_m.y",
    "root_delta_cube_m.z",
    "wrist_local_rotvec_deg.x",
    "wrist_local_rotvec_deg.y",
    "wrist_local_rotvec_deg.z",
)


@dataclass(frozen=True, slots=True)
class JointPairNearZeroBudget:
    """The exact declared campaign budget; values are not adaptive defaults."""

    seed: int = SEED
    static_start_count: int = 512
    static_retain_count: int = 64
    close_controls_per_pose: int = 6
    measured_grasp_retain_count: int = 16
    sequential_plans_per_grasp: int = 4
    feedback_candidates_per_plan: int = 16
    feedback_refine_plan_count: int = 8
    feedback_refine_per_plan: int = 128
    exact_rerun_count: int = 16
    final_candidate_count: int = 5
    perturbations_per_final: int = 16
    robustness_trials: int = 50
    robustness_required_passes: int = 45

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.static_retain_count > self.static_start_count:
            raise ValueError("static retain count exceeds sampled starts")
        if self.measured_grasp_retain_count > self.static_retain_count:
            raise ValueError("measured grasp retain count exceeds static retain count")
        if self.final_candidate_count > self.exact_rerun_count:
            raise ValueError("final count exceeds exact rerun count")
        if self.robustness_required_passes > self.robustness_trials:
            raise ValueError("robustness requirement exceeds its trial count")

    @property
    def dynamic_grasp_count(self) -> int:
        return self.static_retain_count * self.close_controls_per_pose

    @property
    def sequential_plan_count(self) -> int:
        return self.measured_grasp_retain_count * self.sequential_plans_per_grasp

    @property
    def feedback_grid_count(self) -> int:
        return self.sequential_plan_count * self.feedback_candidates_per_plan

    @property
    def feedback_refine_count(self) -> int:
        return self.feedback_refine_plan_count * self.feedback_refine_per_plan

    def as_mapping(self) -> dict[str, Any]:
        result = {
            name: int(getattr(self, name)) for name in self.__dataclass_fields__
        }
        result.update(
            {
                "dynamic_grasp_count": self.dynamic_grasp_count,
                "sequential_plan_count": self.sequential_plan_count,
                "feedback_grid_count": self.feedback_grid_count,
                "feedback_refine_count": self.feedback_refine_count,
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class AuthenticatedNearZeroSource:
    root: Path
    config_path: Path
    result_path: Path
    trace_path: Path
    catalog_path: Path
    config: dict[str, Any]
    result: dict[str, Any]

    @property
    def source_id(self) -> str:
        return canonical_sha256(
            {
                "candidate_id": SOURCE_CANDIDATE_ID,
                "config_sha256": SOURCE_CONFIG_SHA256,
                "result_sha256": SOURCE_RESULT_SHA256,
                "trace_sha256": SOURCE_TRACE_SHA256,
                "catalog_sha256": SOURCE_CATALOG_SHA256,
            }
        )


def authenticate_near_zero_source(
    repository_root: str | Path,
) -> AuthenticatedNearZeroSource:
    """Fail closed if any byte or semantic source evidence changed."""

    root = Path(repository_root).expanduser().resolve()
    member = (root / SOURCE_MEMBER).resolve()
    catalog = (root / SOURCE_CATALOG).resolve()
    bundle = authenticate_v14_candidate_artifacts(
        member,
        expected_candidate_id=SOURCE_CANDIDATE_ID,
        require_retained_trace=True,
    )
    assert bundle.trace_path is not None
    expected_hashes = {
        bundle.config_path: SOURCE_CONFIG_SHA256,
        bundle.result_path: SOURCE_RESULT_SHA256,
        bundle.trace_path: SOURCE_TRACE_SHA256,
        catalog: SOURCE_CATALOG_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"near-zero source SHA-256 mismatch: {path}")
    selected = resolve_viewer_source(
        catalog_path=catalog, trajectory="best_near_zero_grasp"
    )
    if (
        file_sha256(selected.config_path) != SOURCE_CONFIG_SHA256
        or selected.trace_path is None
        or file_sha256(selected.trace_path) != SOURCE_TRACE_SHA256
    ):
        raise RuntimeError("near-zero catalog alias changed its authenticated member")
    config = json.loads(bundle.config_path.read_text(encoding="utf-8"))
    result = copy.deepcopy(bundle.result)
    cube = config.get("cube", {})
    if not (
        math.isclose(float(cube.get("edge_m", -1.0)), 0.079, abs_tol=1e-12)
        and math.isclose(float(cube.get("mass_kg", -1.0)), 0.160, abs_tol=1e-12)
        and math.isclose(float(cube.get("friction", -1.0)), 0.8, abs_tol=1e-12)
    ):
        raise RuntimeError("near-zero source cube configuration changed")
    metrics = result.get("summary", {}).get("metrics", {})
    pair = metrics.get("index_middle_joint_pair_alignment", {})
    if (
        result.get("grasp_success") is not True
        or float(pair.get("grasp_window_angle_p95_deg", math.inf)) > 0.25
    ):
        raise RuntimeError("near-zero source lost its measured grasp evidence")
    return AuthenticatedNearZeroSource(
        root=root,
        config_path=bundle.config_path,
        result_path=bundle.result_path,
        trace_path=bundle.trace_path,
        catalog_path=catalog,
        config=config,
        result=result,
    )


@dataclass(frozen=True, slots=True)
class StaticPerturbation:
    index: int
    joint_qpos_offset_rad: tuple[float, ...]
    root_delta_cube_m: tuple[float, float, float]
    wrist_local_rotvec_deg: tuple[float, float, float]

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise ValueError("index must be non-negative")
        arrays = {
            "joint_qpos_offset_rad": (self.joint_qpos_offset_rad, 8),
            "root_delta_cube_m": (self.root_delta_cube_m, 3),
            "wrist_local_rotvec_deg": (self.wrist_local_rotvec_deg, 3),
        }
        for name, (raw, width) in arrays.items():
            values = np.asarray(raw, dtype=np.float64)
            if values.shape != (width,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain {width} finite values")
            object.__setattr__(self, name, tuple(float(value) for value in values))

    @property
    def candidate_id(self) -> int:
        digest = canonical_sha256(self.as_mapping())
        return 15_100_000_000_000_000 + int(digest[:13], 16) % 100_000_000_000_000

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "index": int(self.index),
            "joint_qpos_offset_rad": list(self.joint_qpos_offset_rad),
            "root_delta_cube_m": list(self.root_delta_cube_m),
            "wrist_local_rotvec_deg": list(self.wrist_local_rotvec_deg),
        }


def generate_static_perturbations(
    budget: JointPairNearZeroBudget = JointPairNearZeroBudget(),
) -> tuple[StaticPerturbation, ...]:
    """Generate the fixed 512-start local LHS, with the source as row zero."""

    count = budget.static_start_count
    dimensions = len(VARIABLE_NAMES)
    rng = np.random.default_rng(budget.seed)
    unit = np.empty((count - 1, dimensions), dtype=np.float64)
    for column in range(dimensions):
        permutation = rng.permutation(count - 1)
        unit[:, column] = (permutation + rng.random(count - 1)) / (count - 1)
    centered = 2.0 * unit - 1.0
    joint = 0.03 * centered[:, :8]
    translation = 0.0015 * centered[:, 8:11]
    rotvec = 1.5 * centered[:, 11:14]
    norms = np.linalg.norm(rotvec, axis=1)
    outside = norms > 2.0
    rotvec[outside] *= (2.0 / norms[outside])[:, None]
    rows = [
        StaticPerturbation(0, (0.0,) * 8, (0.0,) * 3, (0.0,) * 3)
    ]
    rows.extend(
        StaticPerturbation(
            index + 1,
            tuple(joint[index]),
            tuple(translation[index]),
            tuple(rotvec[index]),
        )
        for index in range(count - 1)
    )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class GraspControlVariant:
    """One of the declared 3-duration x 2-profile grasp controllers."""

    index: int
    close_s: float
    mode: str

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise ValueError("index must be non-negative")
        if float(self.close_s) not in (1.25, 1.5, 1.75):
            raise ValueError("close_s is not one of the registered v15 durations")
        if self.mode not in ("original", "synchronized_preload"):
            raise ValueError("unknown grasp close/preload mode")
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "close_s", float(self.close_s))

    @property
    def controller_variant_id(self) -> str:
        return canonical_sha256(self.as_mapping())

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "index": self.index,
            "close_s": self.close_s,
            "mode": self.mode,
        }


def grasp_control_variants() -> tuple[GraspControlVariant, ...]:
    """Return the canonical six grasp controllers in stable search order."""

    result: list[GraspControlVariant] = []
    for close_s in (1.25, 1.5, 1.75):
        for mode in ("original", "synchronized_preload"):
            result.append(GraspControlVariant(len(result), close_s, mode))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class JointPairFeedbackVariant:
    """One point in the declared 4x4 alignment/slip feedback grid."""

    index: int
    alignment_gain: float
    slip_recovery_gain_rad_per_m: float

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise ValueError("index must be non-negative")
        alignment = float(self.alignment_gain)
        slip = float(self.slip_recovery_gain_rad_per_m)
        if alignment not in (0.25, 0.5, 0.75, 1.0):
            raise ValueError("alignment_gain is outside the registered grid")
        if slip not in (2.0, 4.0, 6.0, 8.0):
            raise ValueError("slip recovery gain is outside the registered grid")
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "alignment_gain", alignment)
        object.__setattr__(self, "slip_recovery_gain_rad_per_m", slip)

    @property
    def feedback_variant_id(self) -> str:
        return canonical_sha256(self.as_mapping())

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "index": self.index,
            "alignment_gain": self.alignment_gain,
            "slip_recovery_gain_rad_per_m": (
                self.slip_recovery_gain_rad_per_m
            ),
        }


def joint_pair_feedback_variants() -> tuple[JointPairFeedbackVariant, ...]:
    """Return alignment-major Cartesian grid independent of worker count."""

    result: list[JointPairFeedbackVariant] = []
    for alignment in (0.25, 0.5, 0.75, 1.0):
        for slip in (2.0, 4.0, 6.0, 8.0):
            result.append(
                JointPairFeedbackVariant(len(result), alignment, slip)
            )
    return tuple(result)


def _finite_metric(record: Mapping[str, Any], path: Sequence[str], fallback: float) -> float:
    value: Any = record
    for name in path:
        if not isinstance(value, Mapping):
            return fallback
        value = value.get(name)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _finite_metric_alias(
    record: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
    fallback: float,
) -> float:
    """Read the first finite metric across planner/runtime spellings.

    The offline planning evidence uses the explicit ``*_angle_*`` names while
    the full-reset evaluator historically serializes ``operation_max_deg`` and
    ``operation_p95_deg``.  Ranking must accept both without turning genuine
    final evidence into ``inf``.
    """

    for path in paths:
        value = _finite_metric(record, path, math.nan)
        if math.isfinite(value):
            return value
    return fallback


def near_zero_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Hard pass, robustness, alignment and then contact/path deterministic rank."""

    summary = record.get("summary", record)
    metrics = summary.get("metrics", {}) if isinstance(summary, Mapping) else {}
    pair = metrics.get("joint_pair_alignment", metrics.get("index_middle_joint_pair_alignment", {}))
    contact = metrics.get("contact_preserving_planned_lift", {})
    smooth = metrics.get("motion_smoothness", {})
    duties = contact.get("target_face_effective_duty", {}) if isinstance(contact, Mapping) else {}
    minimum_duty = min(
        (_finite_metric(duties, (finger,), -math.inf) for finger in ("thumb", "index", "mid")),
        default=-math.inf,
    )
    candidate = record.get("candidate_id", "")
    candidate_key = (0, int(candidate)) if isinstance(candidate, int) and not isinstance(candidate, bool) else (1, str(candidate))
    return (
        not bool(record.get("full_success", summary.get("passed", False) if isinstance(summary, Mapping) else False)),
        -int(record.get("perturbation_pass_count", 0)),
        _finite_metric_alias(
            pair,
            (("operation_angle_max_deg",), ("operation_max_deg",)),
            math.inf,
        ),
        _finite_metric_alias(
            pair,
            (("operation_angle_p95_deg",), ("operation_p95_deg",)),
            math.inf,
        ),
        -_finite_metric_alias(
            pair,
            (
                ("operation_within_limit_duty",),
                ("operation_within_p95_limit_duty",),
            ),
            -math.inf,
        ),
        _finite_metric_alias(
            pair,
            (("operation_longest_violation_s",),),
            math.inf,
        ),
        -minimum_duty,
        _finite_metric(contact, ("simultaneous_longest_contact_loss_s",), math.inf),
        -_finite_metric(metrics, ("operation_median_lift_m",), -math.inf),
        _finite_metric(smooth, ("operation_max_lateral_displacement_m",), math.inf),
        _finite_metric(smooth, ("operation_max_orientation_drift_deg",), math.inf),
        _finite_metric(metrics, ("actuator_saturation_fraction",), math.inf),
        candidate_key,
    )


def rank_near_zero_candidates(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    materialized = [copy.deepcopy(dict(record)) for record in records]
    materialized.sort(key=near_zero_candidate_rank)
    return tuple(materialized)


__all__ = [
    "AuthenticatedNearZeroSource",
    "EXPERIMENT_ID",
    "GraspControlVariant",
    "JointPairNearZeroBudget",
    "JointPairFeedbackVariant",
    "SEED",
    "SOURCE_CANDIDATE_ID",
    "StaticPerturbation",
    "VARIABLE_NAMES",
    "authenticate_near_zero_source",
    "generate_static_perturbations",
    "grasp_control_variants",
    "joint_pair_feedback_variants",
    "near_zero_candidate_rank",
    "rank_near_zero_candidates",
]
