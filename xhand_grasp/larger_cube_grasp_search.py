"""Deterministic schema-v3 larger-cube campaign planning and orchestration.

Pure helpers expose the complete work plan for unit tests.  The MuJoCo
orchestrator below consumes those helpers through injectable screen/runner
interfaces, retaining worker-order-independent candidate IDs and rankings.
"""

from __future__ import annotations

import copy
import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import mujoco

from .config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    DISTAL_BODY_NAMES,
    validate_config,
)
from .contacts import BoxContactThresholds
from .experiment import ExperimentDefinition, OpposedFaceAssignment, resolve_experiment
from .large_cube_search import _fixed_material_config, static_candidate_advances
from .scene import build_model
from .search import v3_verify_near_miss_rank
from .v2_search import (
    KinematicScreenResult,
    _cube_position_in_root,
    _cube_world_position,
    _latin_hypercube,
    _rpy_matrix,
    _static_score,
)


EXPERIMENT_ID = "left_opposed_face_palm_down_larger_cube_grasp_then_lift"
CONFIG_SCHEMA_VERSION = 3
FINGER_FACE_OPTIONS = ("+X", "-X", "+Y", "-Y")

_STAGE_SEED_OFFSETS = {
    "coarse": 100_000_000,
    "neighbor": 200_000_000,
    "exact": 300_000_000,
    "grasp_refine": 400_000_000,
    "manipulation_refine": 500_000_000,
    "constant_density_refine": 600_000_000,
    "final_perturbation": 700_000_000,
}


Validator = Callable[[dict[str, Any]], Any]


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _edge_key(edge_m: float) -> int:
    return int(round(_finite(edge_m, "edge_m") * 1_000_000.0))


@dataclass(frozen=True)
class LargerCubeCampaignPlan:
    """Decision-complete budget and material policy for the v3 experiment."""

    schema_version: int = CONFIG_SCHEMA_VERSION
    experiment_id: str = EXPERIMENT_ID
    seed: int = 20260821
    coarse_edges_m: tuple[float, ...] = (
        0.052,
        0.054,
        0.056,
        0.058,
        0.060,
        0.062,
        0.064,
    )
    pitch_values_deg: tuple[float, ...] = (
        60.0,
        65.0,
        70.0,
        75.0,
        80.0,
        85.0,
        90.0,
    )
    candidate_faces: tuple[str, ...] = FINGER_FACE_OPTIONS
    coarse_samples_per_pitch: int = 5_000
    neighbor_samples_per_pitch: int = 10_000
    exact_samples_per_pitch: int = 50_000
    coarse_size_count: int = 3
    exact_size_count: int = 2
    exact_retain_per_size: int = 256
    grasp_refine_seed_count_per_size: int = 4
    grasp_refine_per_seed: int = 128
    manipulation_seed_count: int = 8
    manipulation_refine_per_seed: int = 128
    constant_density_max_candidates: int = 16
    density_refine_seed_count: int = 4
    density_refine_per_seed: int = 128
    finalist_count: int = 16
    perturbations_per_finalist: int = 16
    discovery_mass_kg: float = 0.020
    discovery_friction: float = 0.8
    reference_edge_m: float = 0.030
    reference_mass_kg: float = 0.020
    robustness_edge_window_count: int = 5
    robustness_edge_step_m: float = 0.001
    robustness_density_scales: tuple[float, ...] = (0.9, 1.0, 1.1)
    robustness_friction: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.2)
    fixed_mass_case_label: str = "fixed_20g_control"

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"larger-cube campaign requires schema_version {CONFIG_SCHEMA_VERSION}"
            )
        if self.experiment_id != EXPERIMENT_ID:
            raise ValueError(f"experiment_id must be {EXPERIMENT_ID!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

        edges = tuple(_finite(value, "coarse_edges_m") for value in self.coarse_edges_m)
        expected_edges = tuple(value / 1000.0 for value in range(52, 65, 2))
        if edges != expected_edges:
            raise ValueError("coarse_edges_m must be the complete 52--64 mm even grid")
        object.__setattr__(self, "coarse_edges_m", edges)

        pitches = tuple(_finite(value, "pitch_values_deg") for value in self.pitch_values_deg)
        if pitches != tuple(float(value) for value in range(60, 91, 5)):
            raise ValueError("pitch_values_deg must be 60, 65, ..., 90 degrees")
        object.__setattr__(self, "pitch_values_deg", pitches)
        if tuple(self.candidate_faces) != FINGER_FACE_OPTIONS:
            raise ValueError("candidate_faces must enumerate +X/-X/+Y/-Y in order")

        for label in (
            "coarse_samples_per_pitch",
            "neighbor_samples_per_pitch",
            "exact_samples_per_pitch",
            "coarse_size_count",
            "exact_size_count",
            "exact_retain_per_size",
            "grasp_refine_seed_count_per_size",
            "grasp_refine_per_seed",
            "manipulation_seed_count",
            "manipulation_refine_per_seed",
            "constant_density_max_candidates",
            "density_refine_seed_count",
            "density_refine_per_seed",
            "finalist_count",
            "perturbations_per_finalist",
            "robustness_edge_window_count",
        ):
            _positive_integer(getattr(self, label), label)
        if self.coarse_size_count != 3 or self.exact_size_count != 2:
            raise ValueError("the declared campaign selects three coarse and two exact sizes")
        if self.exact_retain_per_size * self.exact_size_count != 512:
            raise ValueError("the declared exact screens must provide 512 dynamic candidates")
        if self.robustness_edge_window_count != 5:
            raise ValueError("the declared robustness window contains five edges")

        for label in (
            "discovery_mass_kg",
            "discovery_friction",
            "reference_edge_m",
            "reference_mass_kg",
            "robustness_edge_step_m",
        ):
            if _finite(getattr(self, label), label) <= 0.0:
                raise ValueError(f"{label} must be positive")
        scales = tuple(
            _finite(value, "robustness_density_scales")
            for value in self.robustness_density_scales
        )
        frictions = tuple(
            _finite(value, "robustness_friction")
            for value in self.robustness_friction
        )
        if scales != (0.9, 1.0, 1.1):
            raise ValueError("robustness_density_scales must be (0.9, 1.0, 1.1)")
        if frictions != (0.4, 0.6, 0.8, 1.0, 1.2):
            raise ValueError("robustness_friction must be (0.4, ..., 1.2)")
        if self.fixed_mass_case_label != "fixed_20g_control":
            raise ValueError("fixed-mass case family must be fixed_20g_control")

    @property
    def lower_edge_m(self) -> float:
        return self.coarse_edges_m[0]

    @property
    def upper_edge_m(self) -> float:
        return self.coarse_edges_m[-1]

    @property
    def density_kg_m3(self) -> float:
        return self.reference_mass_kg / self.reference_edge_m**3

    @property
    def dynamic_candidate_count(self) -> int:
        return self.exact_size_count * self.exact_retain_per_size

    @property
    def grasp_refinement_count(self) -> int:
        return (
            self.exact_size_count
            * self.grasp_refine_seed_count_per_size
            * self.grasp_refine_per_seed
        )

    @property
    def manipulation_refinement_count(self) -> int:
        return self.manipulation_seed_count * self.manipulation_refine_per_seed

    @property
    def density_refinement_count(self) -> int:
        return self.density_refine_seed_count * self.density_refine_per_seed

    @property
    def robustness_case_count(self) -> int:
        density = (
            self.robustness_edge_window_count
            * len(self.robustness_density_scales)
            * len(self.robustness_friction)
        )
        control = self.robustness_edge_window_count * len(self.robustness_friction)
        return density + control

    def constant_density_mass_kg(self, edge_m: float) -> float:
        edge = _finite(edge_m, "edge_m")
        if edge <= 0.0:
            raise ValueError("edge_m must be positive")
        return self.reference_mass_kg * (edge / self.reference_edge_m) ** 3

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe declaration suitable for tuning reports."""

        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "coarse_edges_m": list(self.coarse_edges_m),
            "pitch_values_deg": list(self.pitch_values_deg),
            "candidate_faces": list(self.candidate_faces),
            "budgets": {
                "coarse_samples_per_pitch": self.coarse_samples_per_pitch,
                "neighbor_samples_per_pitch": self.neighbor_samples_per_pitch,
                "exact_samples_per_pitch": self.exact_samples_per_pitch,
                "coarse_size_count": self.coarse_size_count,
                "exact_size_count": self.exact_size_count,
                "exact_retain_per_size": self.exact_retain_per_size,
                "dynamic_candidate_count": self.dynamic_candidate_count,
                "grasp_refine_seed_count_per_size": self.grasp_refine_seed_count_per_size,
                "grasp_refine_per_seed": self.grasp_refine_per_seed,
                "grasp_refinement_count": self.grasp_refinement_count,
                "manipulation_seed_count": self.manipulation_seed_count,
                "manipulation_refine_per_seed": self.manipulation_refine_per_seed,
                "manipulation_refinement_count": self.manipulation_refinement_count,
                "constant_density_max_candidates": self.constant_density_max_candidates,
                "density_refine_seed_count": self.density_refine_seed_count,
                "density_refine_per_seed": self.density_refine_per_seed,
                "density_refinement_count": self.density_refinement_count,
                "finalist_count": self.finalist_count,
                "perturbations_per_finalist": self.perturbations_per_finalist,
            },
            "material_policy": {
                "discovery_mass_kg": self.discovery_mass_kg,
                "discovery_friction": self.discovery_friction,
                "reference_edge_m": self.reference_edge_m,
                "reference_mass_kg": self.reference_mass_kg,
                "density_kg_m3": self.density_kg_m3,
                "success_requires_constant_density": True,
            },
            "robustness": {
                "edge_limits_m": [self.lower_edge_m, self.upper_edge_m],
                "edge_window_count": self.robustness_edge_window_count,
                "edge_step_m": self.robustness_edge_step_m,
                "density_scales": list(self.robustness_density_scales),
                "friction": list(self.robustness_friction),
                "fixed_mass_kg": self.discovery_mass_kg,
                "fixed_mass_case_label": self.fixed_mass_case_label,
                "case_count": self.robustness_case_count,
            },
        }


DEFAULT_PLAN = LargerCubeCampaignPlan()


@dataclass(frozen=True)
class StaticScreenJob:
    """One deterministic edge/pitch screen unit."""

    stage: str
    edge_m: float
    pitch_deg: float
    samples: int
    seed: int
    face_sample_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in {"coarse", "neighbor", "exact"}:
            raise ValueError("screen job stage must be coarse, neighbor or exact")
        if _finite(self.edge_m, "edge_m") <= 0.0:
            raise ValueError("edge_m must be positive")
        _finite(self.pitch_deg, "pitch_deg")
        _positive_integer(self.samples, "samples")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        counts = {str(key): int(value) for key, value in self.face_sample_counts.items()}
        if set(counts) != set(FINGER_FACE_OPTIONS):
            raise ValueError("face_sample_counts must cover +X/-X/+Y/-Y")
        if any(value < 0 for value in counts.values()) or sum(counts.values()) != self.samples:
            raise ValueError("face_sample_counts must be non-negative and sum to samples")
        object.__setattr__(self, "face_sample_counts", counts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "edge_m": self.edge_m,
            "edge_mm": self.edge_m * 1000.0,
            "pitch_deg": self.pitch_deg,
            "samples": self.samples,
            "seed": self.seed,
            "face_sample_counts": dict(self.face_sample_counts),
        }


@dataclass(frozen=True)
class SizeScreenSummary:
    """Worker-independent aggregate used to select the next size stage."""

    edge_m: float
    sample_count: int
    eligible_count: int
    clean_three_count: int
    near_three_count: int
    best_score: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        edge = _finite(self.edge_m, "edge_m")
        object.__setattr__(self, "edge_m", edge)
        for label in (
            "sample_count",
            "eligible_count",
            "clean_three_count",
            "near_three_count",
        ):
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if self.eligible_count > self.sample_count:
            raise ValueError("eligible_count cannot exceed sample_count")
        score = tuple(_finite(value, "best_score") for value in self.best_score)
        object.__setattr__(self, "best_score", score)


def _face_sample_counts(samples: int, faces: Sequence[str]) -> dict[str, int]:
    quotient, remainder = divmod(samples, len(faces))
    return {
        face: quotient + int(index < remainder)
        for index, face in enumerate(faces)
    }


def stage_seed(
    *, stage: str, edge_m: float, pitch_deg: float, seed: int = DEFAULT_PLAN.seed
) -> int:
    """Derive a stable seed from semantic coordinates, never worker order."""

    if stage not in _STAGE_SEED_OFFSETS:
        raise ValueError(f"unknown search stage {stage!r}")
    pitch_key = int(round(_finite(pitch_deg, "pitch_deg") * 1000.0))
    return int(seed) + _STAGE_SEED_OFFSETS[stage] + _edge_key(edge_m) * 1009 + pitch_key * 9176


def _screen_jobs(
    stage: str,
    edges_m: Iterable[float],
    samples_per_pitch: int,
    *,
    plan: LargerCubeCampaignPlan,
) -> tuple[StaticScreenJob, ...]:
    samples = _positive_integer(samples_per_pitch, "samples_per_pitch")
    edges = tuple(sorted({_finite(edge, "edge_m") for edge in edges_m}))
    return tuple(
        StaticScreenJob(
            stage=stage,
            edge_m=edge,
            pitch_deg=pitch,
            samples=samples,
            seed=stage_seed(
                stage=stage, edge_m=edge, pitch_deg=pitch, seed=plan.seed
            ),
            face_sample_counts=_face_sample_counts(samples, plan.candidate_faces),
        )
        for edge in edges
        for pitch in plan.pitch_values_deg
    )


def coarse_screen_jobs(
    plan: LargerCubeCampaignPlan = DEFAULT_PLAN,
) -> tuple[StaticScreenJob, ...]:
    """Return all 49 coarse jobs; callers must not stop on an early failure."""

    return _screen_jobs(
        "coarse",
        plan.coarse_edges_m,
        plan.coarse_samples_per_pitch,
        plan=plan,
    )


def adjacent_neighbor_edges_m(
    selected_coarse_edges_m: Iterable[float],
    plan: LargerCubeCampaignPlan = DEFAULT_PLAN,
) -> tuple[float, ...]:
    """Return unique +/-1 mm neighbours of the three selected coarse edges."""

    selected = tuple(sorted({_finite(edge, "selected edge") for edge in selected_coarse_edges_m}))
    if len(selected) != plan.coarse_size_count:
        raise ValueError(
            f"neighbor stage requires exactly {plan.coarse_size_count} coarse sizes"
        )
    if any(
        not any(math.isclose(edge, declared, abs_tol=1e-12) for declared in plan.coarse_edges_m)
        for edge in selected
    ):
        raise ValueError("neighbor stage inputs must be declared coarse edges")
    result = {
        round(edge + delta, 12)
        for edge in selected
        for delta in (-0.001, 0.001)
        if plan.lower_edge_m - 1e-12 <= edge + delta <= plan.upper_edge_m + 1e-12
    }
    return tuple(sorted(result))


def neighbor_screen_jobs(
    selected_coarse_edges_m: Iterable[float],
    plan: LargerCubeCampaignPlan = DEFAULT_PLAN,
) -> tuple[StaticScreenJob, ...]:
    return _screen_jobs(
        "neighbor",
        adjacent_neighbor_edges_m(selected_coarse_edges_m, plan),
        plan.neighbor_samples_per_pitch,
        plan=plan,
    )


def exact_screen_jobs(
    selected_exact_edges_m: Iterable[float],
    plan: LargerCubeCampaignPlan = DEFAULT_PLAN,
) -> tuple[StaticScreenJob, ...]:
    edges = tuple(sorted({_finite(edge, "selected exact edge") for edge in selected_exact_edges_m}))
    if len(edges) != plan.exact_size_count:
        raise ValueError(f"exact stage requires exactly {plan.exact_size_count} sizes")
    if any(edge < plan.lower_edge_m - 1e-12 or edge > plan.upper_edge_m + 1e-12 for edge in edges):
        raise ValueError("exact stage sizes must remain inside 52--64 mm")
    return _screen_jobs(
        "exact", edges, plan.exact_samples_per_pitch, plan=plan
    )


def size_summary_rank(summary: SizeScreenSummary) -> tuple[Any, ...]:
    sample_count = max(1, summary.sample_count)
    return (
        float(summary.eligible_count > 0),
        summary.best_score,
        summary.eligible_count / sample_count,
        summary.eligible_count,
        summary.clean_three_count,
        summary.near_three_count,
        -_edge_key(summary.edge_m),
    )


def select_top_size_summaries(
    summaries: Iterable[SizeScreenSummary], *, count: int
) -> tuple[SizeScreenSummary, ...]:
    """Select sizes deterministically and reject ambiguous duplicate aggregates."""

    requested = _positive_integer(count, "count")
    materialized = tuple(summaries)
    keys = tuple(_edge_key(summary.edge_m) for summary in materialized)
    if len(set(keys)) != len(keys):
        raise ValueError("size summaries must contain one aggregate per edge")
    if len(materialized) < requested:
        raise ValueError("not enough size summaries for requested selection")
    return tuple(sorted(materialized, key=size_summary_rank, reverse=True)[:requested])


def select_coarse_edges_m(
    summaries: Iterable[SizeScreenSummary],
    plan: LargerCubeCampaignPlan = DEFAULT_PLAN,
) -> tuple[float, ...]:
    materialized = tuple(summaries)
    if {_edge_key(item.edge_m) for item in materialized} != {
        _edge_key(edge) for edge in plan.coarse_edges_m
    }:
        raise ValueError("coarse selection requires summaries for all seven sizes")
    selected = select_top_size_summaries(materialized, count=plan.coarse_size_count)
    return tuple(item.edge_m for item in selected)


def select_exact_edges_m(
    summaries: Iterable[SizeScreenSummary],
    plan: LargerCubeCampaignPlan = DEFAULT_PLAN,
) -> tuple[float, ...]:
    selected = select_top_size_summaries(summaries, count=plan.exact_size_count)
    return tuple(item.edge_m for item in selected)


def _summary(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = result.get("summary", {})
    return value if isinstance(value, Mapping) else {}


def _metrics(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _summary(result).get("metrics", {})
    return value if isinstance(value, Mapping) else {}


def _bool_metric(result: Mapping[str, Any], name: str) -> bool:
    summary = _summary(result)
    stage_status = summary.get("stage_status", {})
    value = (
        stage_status.get(name, False)
        if isinstance(stage_status, Mapping) and name in stage_status
        else summary.get(name, False)
    )
    return value is True or isinstance(value, np.bool_) and bool(value)


def grasp_succeeded(result: Mapping[str, Any]) -> bool:
    """Return only the explicit v3 grasp latch result; generic pass is insufficient."""

    return _bool_metric(result, "grasp_success")


def manipulation_succeeded(result: Mapping[str, Any]) -> bool:
    return _bool_metric(result, "manipulation_success")


def full_succeeded(result: Mapping[str, Any]) -> bool:
    return _bool_metric(result, "full_success")


def _numeric_metric(
    metrics: Mapping[str, Any], names: Sequence[str], *, default: float
) -> float:
    for name in names:
        if name not in metrics:
            continue
        try:
            number = float(metrics[name])
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return default


def _grasp_stability_margin(result: Mapping[str, Any]) -> float:
    metrics = _metrics(result)
    direct = _numeric_metric(
        metrics,
        ("grasp_stability_margin", "grasp_minimum_normalized_margin"),
        default=-math.inf,
    )
    if math.isfinite(direct):
        return direct
    margins = metrics.get("grasp_gate_margins")
    if not isinstance(margins, Mapping) or not margins:
        final_steps = _numeric_metric(
            metrics,
            (
                "verify_max_consecutive_all_gate_steps",
                "verify_max_consecutive_gate_steps",
                "grasp_gate_final_consecutive_steps",
            ),
            default=-math.inf,
        )
        required_steps = _numeric_metric(
            metrics, ("grasp_stable_window_steps",), default=-math.inf
        )
        if (
            math.isfinite(final_steps)
            and math.isfinite(required_steps)
            and required_steps > 0.0
        ):
            return (final_steps - required_steps) / required_steps
        return -math.inf
    values: list[float] = []
    for value in margins.values():
        try:
            number = float(value)
        except (TypeError, ValueError):
            return -math.inf
        if not math.isfinite(number):
            return -math.inf
        values.append(number)
    return min(values, default=-math.inf)


def _perturbation_passes(result: Mapping[str, Any]) -> int:
    for container in (result, _summary(result), _metrics(result)):
        value = container.get("perturbation_passes")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    probe = result.get("local_perturbation_probe")
    if isinstance(probe, Mapping):
        value = probe.get("passes")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0


def larger_cube_candidate_rank(
    result: Mapping[str, Any], *, perturbation_passes: int | None = None
) -> tuple[float, ...]:
    """Canonical final rank with candidate ID as the total-order tie break.

    Acquired candidates retain the campaign contract: full hard pass,
    grasp-stability margin, perturbation pass count, topology retention, lower
    contact force, lower saturation, then stable candidate ID.  Before grasp
    acquisition, raw VERIFY evidence is inserted ahead of the force penalty so
    zero-contact candidates cannot beat candidates approaching a three-finger
    grasp merely because they exert less force.
    """

    metrics = _metrics(result)
    topology = _numeric_metric(
        metrics,
        (
            "operation_target_face_simultaneous_duty",
            "target_face_simultaneous_duty",
            "grasp_target_face_simultaneous_duty",
            "topology_duty",
        ),
        default=-math.inf,
    )
    force = _numeric_metric(
        metrics,
        ("peak_total_distal_contact_force_n", "peak_grasp_contact_force_n"),
        default=math.inf,
    )
    saturation = _numeric_metric(
        metrics, ("actuator_saturation_fraction",), default=math.inf
    )
    try:
        candidate_id = int(result["candidate_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("candidate result requires an integer candidate_id") from error
    passes = (
        _perturbation_passes(result)
        if perturbation_passes is None
        else int(perturbation_passes)
    )
    if passes < -1:
        raise ValueError("perturbation_passes must be -1 or non-negative")
    if not grasp_succeeded(result):
        return (
            float(full_succeeded(result)),
            _grasp_stability_margin(result),
            float(passes),
            *v3_verify_near_miss_rank(dict(metrics)),
            topology,
            -force,
            -saturation,
            -float(candidate_id),
        )
    return (
        float(full_succeeded(result)),
        _grasp_stability_margin(result),
        float(passes),
        topology,
        -force,
        -saturation,
        -float(candidate_id),
    )


def deterministic_rank_results(
    results: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return a worker-order-independent total order and reject duplicate IDs."""

    materialized = tuple(results)
    try:
        ids = tuple(int(result["candidate_id"]) for result in materialized)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("every result requires an integer candidate_id") from error
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_id values must be unique")
    return tuple(sorted(materialized, key=larger_cube_candidate_rank, reverse=True))


def select_grasp_refinement_parents(
    results: Iterable[dict[str, Any]],
    *,
    exact_edges_m: Iterable[float],
    plan: LargerCubeCampaignPlan = DEFAULT_PLAN,
) -> tuple[dict[str, Any], ...]:
    """Select four CLOSE+VERIFY parents per exact size, including near misses."""

    ranked = deterministic_rank_results(results)
    selected: list[dict[str, Any]] = []
    for edge in exact_edges_m:
        edge_key = _edge_key(edge)
        group = [
            result
            for result in ranked
            if _edge_key(result["config"]["cube"]["edge_m"]) == edge_key
        ]
        selected.extend(group[: plan.grasp_refine_seed_count_per_size])
    return tuple(selected)


def select_manipulation_parents(
    results: Iterable[dict[str, Any]],
    plan: LargerCubeCampaignPlan = DEFAULT_PLAN,
) -> tuple[dict[str, Any], ...]:
    """Enforce the hard boundary: only a latched stable grasp may manipulate."""

    verified = [result for result in results if grasp_succeeded(result)]
    return deterministic_rank_results(verified)[: plan.manipulation_seed_count]


def _validated_ranges(
    ranges: Mapping[str, Sequence[float]], label: str
) -> dict[str, tuple[float, float]]:
    if set(ranges) != set(ACTIVE_ACTUATORS):
        raise ValueError(f"{label} must contain exactly the eight active actuators")
    result: dict[str, tuple[float, float]] = {}
    for name in ACTIVE_ACTUATORS:
        value = tuple(ranges[name])
        if len(value) != 2:
            raise ValueError(f"{label}.{name} must contain two bounds")
        lower = _finite(value[0], f"{label}.{name}[0]")
        upper = _finite(value[1], f"{label}.{name}[1]")
        if lower > upper:
            raise ValueError(f"{label}.{name} must be increasing")
        result[name] = (lower, upper)
    return result


def manipulation_targets(config: Mapping[str, Any]) -> dict[str, float]:
    """Materialize the absolute target only for command dispatch, never storage."""

    control = config.get("control")
    if not isinstance(control, Mapping):
        raise ValueError("config.control must be a mapping")
    grasp = control.get("grasp_targets_rad")
    delta = control.get("manipulation_delta_rad")
    if not isinstance(grasp, Mapping) or set(grasp) != set(ACTIVE_ACTUATORS):
        raise ValueError("grasp_targets_rad must contain exactly the active actuators")
    if not isinstance(delta, Mapping) or set(delta) != set(ACTIVE_ACTUATORS):
        raise ValueError("manipulation_delta_rad must contain exactly the active actuators")
    return {
        name: _finite(grasp[name], f"grasp_targets_rad.{name}")
        + _finite(delta[name], f"manipulation_delta_rad.{name}")
        for name in ACTIVE_ACTUATORS
    }


def sample_manipulation_delta_candidates(
    parent_config: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    delta_bounds_rad: Mapping[str, Sequence[float]],
    local_radius_rad: float | Mapping[str, float] | None = None,
    absolute_target_bounds_rad: Mapping[str, Sequence[float]] | None = None,
    validator: Validator | None = None,
) -> list[dict[str, Any]]:
    """Sample relative operation deltas while preserving the verified grasp pose.

    The effective range is the intersection of the experiment delta bounds,
    an optional local radius around the parent delta, and optional absolute
    actuator limits translated by the stored grasp target.  Empty intersections
    fail rather than silently clipping or sampling a second absolute pose.
    """

    sample_count = _positive_integer(count, "count")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    bounds = _validated_ranges(delta_bounds_rad, "delta_bounds_rad")
    absolute = (
        None
        if absolute_target_bounds_rad is None
        else _validated_ranges(absolute_target_bounds_rad, "absolute_target_bounds_rad")
    )
    control = parent_config.get("control")
    if not isinstance(control, Mapping):
        raise ValueError("parent config requires control")
    grasp = control.get("grasp_targets_rad")
    centre = control.get("manipulation_delta_rad")
    if not isinstance(grasp, Mapping) or set(grasp) != set(ACTIVE_ACTUATORS):
        raise ValueError("parent grasp_targets_rad must contain active actuators")
    if not isinstance(centre, Mapping) or set(centre) != set(ACTIVE_ACTUATORS):
        raise ValueError("parent manipulation_delta_rad must contain active actuators")

    if isinstance(local_radius_rad, Mapping):
        if set(local_radius_rad) != set(ACTIVE_ACTUATORS):
            raise ValueError("local_radius_rad must contain active actuators")
        radii = {
            name: _finite(local_radius_rad[name], f"local_radius_rad.{name}")
            for name in ACTIVE_ACTUATORS
        }
    elif local_radius_rad is None:
        radii = {name: math.inf for name in ACTIVE_ACTUATORS}
    else:
        radius = _finite(local_radius_rad, "local_radius_rad")
        radii = {name: radius for name in ACTIVE_ACTUATORS}
    if any(value <= 0.0 for value in radii.values()):
        raise ValueError("local manipulation radii must be positive")

    effective: dict[str, tuple[float, float]] = {}
    for name in ACTIVE_ACTUATORS:
        lower, upper = bounds[name]
        centre_value = _finite(centre[name], f"manipulation_delta_rad.{name}")
        radius = radii[name]
        lower = max(lower, centre_value - radius)
        upper = min(upper, centre_value + radius)
        if absolute is not None:
            grasp_value = _finite(grasp[name], f"grasp_targets_rad.{name}")
            lower = max(lower, absolute[name][0] - grasp_value)
            upper = min(upper, absolute[name][1] - grasp_value)
        if lower > upper + 1e-12:
            raise ValueError(f"empty manipulation delta range for actuator {name!r}")
        effective[name] = (lower, upper)

    rng = np.random.default_rng(seed)
    unit = np.empty((sample_count, len(ACTIVE_ACTUATORS)), dtype=np.float64)
    for dimension in range(len(ACTIVE_ACTUATORS)):
        unit[:, dimension] = (
            rng.permutation(sample_count) + rng.random(sample_count)
        ) / sample_count

    candidates: list[dict[str, Any]] = []
    for row in unit:
        candidate = copy.deepcopy(dict(parent_config))
        candidate["control"]["manipulation_delta_rad"] = {
            name: effective[name][0]
            + float(row[index]) * (effective[name][1] - effective[name][0])
            for index, name in enumerate(ACTIVE_ACTUATORS)
        }
        # Exercise the relative-target invariant before accepting the sample.
        manipulation_targets(candidate)
        if validator is not None:
            validator(candidate)
        candidates.append(candidate)
    return candidates


def build_manipulation_refinement_candidates(
    results: Iterable[dict[str, Any]],
    *,
    delta_bounds_rad: Mapping[str, Sequence[float]],
    plan: LargerCubeCampaignPlan = DEFAULT_PLAN,
    local_radius_rad: float | Mapping[str, float] | None = None,
    absolute_target_bounds_rad: Mapping[str, Sequence[float]] | None = None,
    validator: Validator | None = None,
) -> list[dict[str, Any]]:
    """Expand at most eight verified grasps into exactly 128 delta trials each."""

    candidates: list[dict[str, Any]] = []
    for parent_index, parent in enumerate(select_manipulation_parents(results, plan)):
        candidates.extend(
            sample_manipulation_delta_candidates(
                parent["config"],
                count=plan.manipulation_refine_per_seed,
                seed=(
                    plan.seed
                    + _STAGE_SEED_OFFSETS["manipulation_refine"]
                    + parent_index * 1_000_003
                ),
                delta_bounds_rad=delta_bounds_rad,
                local_radius_rad=local_radius_rad,
                absolute_target_bounds_rad=absolute_target_bounds_rad,
                validator=validator,
            )
        )
    return candidates


def constant_density_candidates(
    results: Iterable[dict[str, Any]],
    plan: LargerCubeCampaignPlan = DEFAULT_PLAN,
    *,
    validator: Validator | None = None,
) -> list[dict[str, Any]]:
    """Transition only fixed-mass full passes to the constant-density stage."""

    passing = [
        result
        for result in results
        if full_succeeded(result)
        and math.isclose(
            _finite(result["config"]["cube"]["mass_kg"], "cube.mass_kg"),
            plan.discovery_mass_kg,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            _finite(result["config"]["cube"]["friction"], "cube.friction"),
            plan.discovery_friction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    ranked = deterministic_rank_results(passing)
    candidates: list[dict[str, Any]] = []
    for parent in ranked[: plan.constant_density_max_candidates]:
        candidate = copy.deepcopy(parent["config"])
        edge_m = _finite(candidate["cube"]["edge_m"], "cube.edge_m")
        candidate["cube"]["mass_kg"] = plan.constant_density_mass_kg(edge_m)
        if validator is not None:
            validator(candidate)
        candidates.append(candidate)
    return candidates


def robustness_edge_window_m(
    nominal_edge_m: float, plan: LargerCubeCampaignPlan = DEFAULT_PLAN
) -> tuple[float, ...]:
    """Return five consecutive integer-mm edges, clamped to 52--64 mm."""

    nominal = _finite(nominal_edge_m, "nominal_edge_m")
    lower, upper = plan.lower_edge_m, plan.upper_edge_m
    if nominal < lower - 1e-12 or nominal > upper + 1e-12:
        raise ValueError("nominal edge must remain inside 52--64 mm")
    grid_count = round((upper - lower) / plan.robustness_edge_step_m) + 1
    centre = round((nominal - lower) / plan.robustness_edge_step_m)
    half = plan.robustness_edge_window_count // 2
    start = min(max(0, centre - half), grid_count - plan.robustness_edge_window_count)
    return tuple(
        round(lower + (start + index) * plan.robustness_edge_step_m, 12)
        for index in range(plan.robustness_edge_window_count)
    )


def larger_cube_robustness_cases(
    config: Mapping[str, Any],
    plan: LargerCubeCampaignPlan = DEFAULT_PLAN,
    *,
    validator: Validator | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build 75 constant-density cases followed by 25 fixed-20 g controls."""

    nominal_edge = _finite(config["cube"]["edge_m"], "cube.edge_m")
    edges = robustness_edge_window_m(nominal_edge, plan)
    cases: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []

    for edge_m in edges:
        reference_mass = plan.constant_density_mass_kg(edge_m)
        for density_scale in plan.robustness_density_scales:
            for friction in plan.robustness_friction:
                case = copy.deepcopy(dict(config))
                case["cube"]["edge_m"] = edge_m
                case["cube"]["mass_kg"] = reference_mass * density_scale
                case["cube"]["friction"] = friction
                if validator is not None:
                    validator(case)
                cases.append(case)
                metadata.append(
                    {
                        "grid_index": len(metadata),
                        "case_family": "constant_density",
                        "edge_m": edge_m,
                        "edge_mm": edge_m * 1000.0,
                        "mass_kg": case["cube"]["mass_kg"],
                        "friction": friction,
                        "density_scale": density_scale,
                        "reference_density_kg_m3": plan.density_kg_m3,
                    }
                )

    for edge_m in edges:
        for friction in plan.robustness_friction:
            case = copy.deepcopy(dict(config))
            case["cube"]["edge_m"] = edge_m
            case["cube"]["mass_kg"] = plan.discovery_mass_kg
            case["cube"]["friction"] = friction
            if validator is not None:
                validator(case)
            cases.append(case)
            metadata.append(
                {
                    "grid_index": len(metadata),
                    "case_family": plan.fixed_mass_case_label,
                    "edge_m": edge_m,
                    "edge_mm": edge_m * 1000.0,
                    "mass_kg": plan.discovery_mass_kg,
                    "friction": friction,
                    "density_scale": None,
                    "reference_density_kg_m3": plan.density_kg_m3,
                }
            )

    if len(cases) != plan.robustness_case_count or len(metadata) != len(cases):
        raise AssertionError(
            f"generated {len(cases)} robustness cases, expected "
            f"{plan.robustness_case_count}"
        )
    return cases, metadata


def campaign_manifest(
    plan: LargerCubeCampaignPlan = DEFAULT_PLAN,
) -> dict[str, Any]:
    """Return the complete declared stage graph and nominal work budgets."""

    coarse_jobs = coarse_screen_jobs(plan)
    pitch_count = len(plan.pitch_values_deg)
    coarse_sample_count = sum(job.samples for job in coarse_jobs)
    declared_neighbor_edge_slots = plan.coarse_size_count * 2
    declared_neighbor_sample_count = (
        declared_neighbor_edge_slots * pitch_count * plan.neighbor_samples_per_pitch
    )
    declared_exact_sample_count = (
        plan.exact_size_count * pitch_count * plan.exact_samples_per_pitch
    )
    return {
        "campaign": plan.as_dict(),
        "stage_order": [
            "coarse_static",
            "neighbor_static",
            "exact_static",
            "fixed_mass_close_verify",
            "fixed_mass_grasp_refinement",
            "fixed_mass_manipulation_refinement",
            "constant_density_reevaluation",
            "constant_density_refinement",
            "final_perturbation_probe",
        ],
        "hard_stage_gates": {
            "manipulation_requires_grasp_success": True,
            "constant_density_requires_fixed_mass_full_success": True,
            "campaign_success_requires_constant_density_full_success": True,
        },
        "nominal_counts": {
            "coarse_job_count": len(coarse_jobs),
            "coarse_samples_per_size": len(plan.pitch_values_deg)
            * plan.coarse_samples_per_pitch,
            "coarse_sample_count": coarse_sample_count,
            "declared_neighbor_edge_slots": declared_neighbor_edge_slots,
            "declared_neighbor_sample_count": declared_neighbor_sample_count,
            "declared_exact_sample_count": declared_exact_sample_count,
            "declared_max_static_sample_count": coarse_sample_count
            + declared_neighbor_sample_count
            + declared_exact_sample_count,
            "dynamic_candidate_count": plan.dynamic_candidate_count,
            "grasp_refinement_count": plan.grasp_refinement_count,
            "manipulation_refinement_count": plan.manipulation_refinement_count,
            "density_refinement_count": plan.density_refinement_count,
            "final_perturbation_count": plan.finalist_count
            * plan.perturbations_per_finalist,
            "robustness_case_count": plan.robustness_case_count,
        },
    }


def _static_search_budget_report(
    *,
    campaign: Any,
    pitch_count: int,
    selected_coarse_edges_m: Sequence[float],
    neighbor_edges_m: Sequence[float],
    coarse_samples_per_pitch: int,
    neighbor_samples_per_pitch: int,
    exact_samples_per_pitch: int,
    screen_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Describe the maximum declaration and this run's deduplicated budget."""

    raw_neighbor_slots = tuple(
        round(float(edge) + delta, 12)
        for edge in selected_coarse_edges_m
        for delta in (-0.001, 0.001)
    )
    lower = float(campaign.coarse_edges_m[0])
    upper = float(campaign.coarse_edges_m[-1])
    in_range_slots = tuple(
        edge for edge in raw_neighbor_slots if lower - 1e-12 <= edge <= upper + 1e-12
    )
    unique_in_range = tuple(sorted(set(in_range_slots)))
    observed_neighbors = tuple(sorted(float(edge) for edge in neighbor_edges_m))
    if observed_neighbors != unique_in_range:
        raise RuntimeError(
            "neighbor screen edges do not match the deduplicated in-range plan; "
            f"expected={unique_in_range}, observed={observed_neighbors}"
        )

    declared_neighbor_slots = int(campaign.coarse_size_count) * 2

    def stage_samples(stage: str) -> int:
        return sum(int(record["sample_count"]) for record in screen_records[stage])

    actual_by_stage = {
        "coarse": stage_samples("coarse"),
        "neighbor": stage_samples("neighbor"),
        "exact": stage_samples("exact"),
    }
    expected_actual_by_stage = {
        "coarse": len(campaign.coarse_edges_m)
        * pitch_count
        * coarse_samples_per_pitch,
        "neighbor": len(observed_neighbors)
        * pitch_count
        * neighbor_samples_per_pitch,
        "exact": int(campaign.exact_size_count)
        * pitch_count
        * exact_samples_per_pitch,
    }
    versioned_declared_by_stage = {
        "coarse": len(campaign.coarse_edges_m)
        * pitch_count
        * int(campaign.coarse_samples_per_pitch),
        "neighbor": declared_neighbor_slots
        * pitch_count
        * int(campaign.odd_samples_per_pitch),
        "exact": int(campaign.exact_size_count)
        * pitch_count
        * int(campaign.fine_samples_per_pitch),
    }
    effective_declared_by_stage = {
        "coarse": len(campaign.coarse_edges_m)
        * pitch_count
        * coarse_samples_per_pitch,
        "neighbor": declared_neighbor_slots
        * pitch_count
        * neighbor_samples_per_pitch,
        "exact": int(campaign.exact_size_count)
        * pitch_count
        * exact_samples_per_pitch,
    }
    out_of_range_count = len(raw_neighbor_slots) - len(in_range_slots)
    duplicate_count = len(in_range_slots) - len(unique_in_range)
    reasons: list[str] = []
    if duplicate_count:
        reasons.append("neighbor_edge_deduplication")
    if out_of_range_count:
        reasons.append("neighbor_range_clamping")
    if actual_by_stage != expected_actual_by_stage:
        reasons.append("screen_reported_sample_count_mismatch")
    stop_reason = (
        "completed_full_declared_static_budget"
        if not reasons
        else "completed_with_" + "_and_".join(reasons)
    )
    return {
        "versioned_declared": {
            "stage_sample_count": versioned_declared_by_stage,
            "maximum_total_sample_count": sum(
                versioned_declared_by_stage.values()
            ),
            "neighbor_edge_slots": declared_neighbor_slots,
        },
        "effective_declared": {
            "stage_sample_count": effective_declared_by_stage,
            "maximum_total_sample_count": sum(
                effective_declared_by_stage.values()
            ),
            "neighbor_edge_slots": declared_neighbor_slots,
        },
        "actual": {
            "stage_sample_count": actual_by_stage,
            "expected_stage_sample_count": expected_actual_by_stage,
            "total_sample_count": sum(actual_by_stage.values()),
            "unique_neighbor_edge_count": len(observed_neighbors),
        },
        "neighbor_selection": {
            "raw_slots_m": list(raw_neighbor_slots),
            "unique_in_range_edges_m": list(observed_neighbors),
            "duplicate_slot_count": duplicate_count,
            "out_of_range_slot_count": out_of_range_count,
        },
        "stop_reason": stop_reason,
    }


def _per_size_outcomes(
    *,
    selected_exact_edges_m: Sequence[float],
    initial_runs: Sequence[Mapping[str, Any]],
    grasp_local_runs: Sequence[Mapping[str, Any]],
    manipulation_runs: Sequence[Mapping[str, Any]],
    density_runs: Sequence[Mapping[str, Any]],
    density_local_runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return JSON-safe dynamic outcomes with explicit per-stage denominators."""

    def at_edge(
        results: Sequence[Mapping[str, Any]], edge_m: float
    ) -> list[Mapping[str, Any]]:
        key = _edge_key(edge_m)
        return [
            result
            for result in results
            if _edge_key(result["config"]["cube"]["edge_m"]) == key
        ]

    records: list[dict[str, Any]] = []
    for edge_m in sorted(float(value) for value in selected_exact_edges_m):
        initial = at_edge(initial_runs, edge_m)
        grasp_local = at_edge(grasp_local_runs, edge_m)
        grasp_trials = initial + grasp_local
        manipulation = at_edge(manipulation_runs, edge_m)
        density_reevaluation = at_edge(density_runs, edge_m)
        density_local = at_edge(density_local_runs, edge_m)
        density_trials = density_reevaluation + density_local
        stable_count = sum(grasp_succeeded(result) for result in grasp_trials)
        stable_denominator = len(grasp_trials)
        fixed_full_count = sum(full_succeeded(result) for result in manipulation)
        density_full_count = sum(full_succeeded(result) for result in density_trials)
        fixed_dynamic_count = (
            len(initial) + len(grasp_local) + len(manipulation)
        )
        total_dynamic_count = fixed_dynamic_count + len(density_trials)
        records.append(
            {
                "edge_m": edge_m,
                "edge_mm": edge_m * 1000.0,
                "initial_dynamic_trial_count": len(initial),
                "grasp_refinement_trial_count": len(grasp_local),
                "manipulation_refinement_trial_count": len(manipulation),
                "fixed_mass_dynamic_trial_count": fixed_dynamic_count,
                "density_reevaluation_trial_count": len(density_reevaluation),
                "density_refinement_trial_count": len(density_local),
                "density_dynamic_trial_count": len(density_trials),
                "total_dynamic_trial_count": total_dynamic_count,
                "stable_grasp_evaluation_count": stable_denominator,
                "stable_grasp_success_count": stable_count,
                "stable_grasp_success_rate": (
                    stable_count / stable_denominator
                    if stable_denominator
                    else 0.0
                ),
                "fixed_full_success_count": fixed_full_count,
                "density_full_success_count": density_full_count,
            }
        )
    return records


def _candidate_parameters_v3(
    row: np.ndarray,
    *,
    pitch_deg: float,
    candidate_id: int,
    definition: ExperimentDefinition,
) -> dict[str, Any]:
    """Map one 14-dimensional sample to a grasp pose only.

    Unlike the v2 large-cube screen, no unrelated terminal pose is sampled at
    this stage.  Manipulation deltas remain exactly zero until a dynamic grasp
    has passed the feedback gate.
    """

    values = np.asarray(row, dtype=np.float64)
    expected = 6 + len(ACTIVE_ACTUATORS)
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ValueError(f"v3 static sample must contain {expected} finite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("v3 static sample values must remain within [0, 1]")
    bounds = definition.search_bounds

    def scale(unit: float, limits: Sequence[float]) -> float:
        return float(limits[0] + unit * (limits[1] - limits[0]))

    cursor = 0
    roll = scale(values[cursor], bounds.hand_roll_deg)
    cursor += 1
    yaw = scale(values[cursor], bounds.hand_yaw_deg)
    cursor += 1
    cube_in_root = np.asarray(
        [
            scale(values[cursor + index], bounds.cube_position_in_root_m[axis])
            for index, axis in enumerate(("x", "y", "z"))
        ],
        dtype=np.float64,
    )
    cursor += 3
    cube_yaw = scale(values[cursor], bounds.cube_yaw_deg)
    cursor += 1
    grasp_targets = {
        name: scale(values[cursor + index], bounds.actuator_targets_rad[name])
        for index, name in enumerate(ACTIVE_ACTUATORS)
    }
    assignment = definition.candidate_faces[
        candidate_id % len(definition.candidate_faces)
    ]
    return {
        "candidate_id": int(candidate_id),
        "hand_rpy_deg": [roll, float(pitch_deg), yaw],
        "cube_in_root_m": cube_in_root,
        "cube_yaw_deg": cube_yaw,
        # _static_score intentionally consumes the historical semantic name;
        # it is only a transient parameter, never persisted in schema v3.
        "pregrasp_targets_rad": grasp_targets,
        "target_assignment": assignment,
    }


def _materialize_v3_candidate(
    base: dict[str, Any], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = copy.deepcopy(base)
    rpy = [float(value) for value in parameters["hand_rpy_deg"]]
    cube_world = _cube_world_position(candidate)
    cube_in_root = np.asarray(parameters["cube_in_root_m"], dtype=np.float64)
    candidate["hand_pose"]["translation_m"] = (
        cube_world - _rpy_matrix(rpy) @ cube_in_root
    ).tolist()
    candidate["hand_pose"]["rpy_deg"] = rpy
    candidate["cube"]["rpy_deg"] = [
        0.0,
        0.0,
        float(parameters["cube_yaw_deg"]),
    ]
    candidate["control"]["grasp_targets_rad"] = {
        name: float(parameters["pregrasp_targets_rad"][name])
        for name in ACTIVE_ACTUATORS
    }
    candidate["control"]["manipulation_delta_rad"] = {
        name: 0.0 for name in ACTIVE_ACTUATORS
    }
    assignment = parameters["target_assignment"]
    if not isinstance(assignment, OpposedFaceAssignment):
        raise TypeError("target_assignment must be OpposedFaceAssignment")
    candidate["contact_topology"]["target_faces"] = assignment.as_dict()
    validate_config(candidate)
    return candidate


def kinematic_screen_v3(
    base: dict[str, Any],
    *,
    samples_per_pitch: int,
    retain: int,
    seed: int,
    definition: ExperimentDefinition | None = None,
) -> KinematicScreenResult:
    """Screen static grasp poses without sampling a manipulation endpoint."""

    definition = resolve_experiment(base) if definition is None else definition
    if int(base.get("schema_version", 0)) != 3:
        raise ValueError("kinematic_screen_v3 requires schema version 3")
    if definition.experiment_id != EXPERIMENT_ID:
        raise ValueError("kinematic_screen_v3 received a different experiment")
    if samples_per_pitch <= 0 or retain < len(definition.candidate_faces):
        raise ValueError("static samples must be positive and retain every face")

    model, info = build_model(base)
    data = mujoco.MjData(model)
    topology = base["contact_topology"]
    thresholds = BoxContactThresholds(
        surface_tolerance_m=float(topology["surface_tolerance_m"]),
        edge_margin_m=float(topology["edge_margin_m"]),
        normal_alignment_min=float(topology["min_normal_alignment"]),
    )
    distal_geom_ids = {
        finger: tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.body_weldid[int(model.geom_bodyid[geom_id])])
            == info.distal_weld_ids[finger]
            and (
                int(model.geom_contype[geom_id]) != 0
                or int(model.geom_conaffinity[geom_id]) != 0
            )
        )
        for finger in ACTIVE_FINGERS
    }
    distal_site_ids = {
        finger: np.asarray(
            [
                site_id
                for site_id in range(model.nsite)
                if int(model.body_weldid[int(model.site_bodyid[site_id])])
                == info.distal_weld_ids[finger]
            ],
            dtype=int,
        )
        for finger in ACTIVE_FINGERS
    }
    if any(not values for values in distal_geom_ids.values()) or any(
        values.size == 0 for values in distal_site_ids.values()
    ):
        raise ValueError("active distal geometry/tactile mapping is incomplete")

    capacity = {
        assignment: retain // len(definition.candidate_faces)
        + int(index < retain % len(definition.candidate_faces))
        for index, assignment in enumerate(definition.candidate_faces)
    }
    entries: dict[OpposedFaceAssignment, list[tuple[Any, ...]]] = {
        assignment: [] for assignment in definition.candidate_faces
    }

    seed_assignment = OpposedFaceAssignment.from_mapping(
        base["contact_topology"]["target_faces"]
    )
    seed_parameters = {
        "candidate_id": -1,
        "hand_rpy_deg": [float(value) for value in base["hand_pose"]["rpy_deg"]],
        "cube_in_root_m": _cube_position_in_root(base),
        "cube_yaw_deg": float(base["cube"].get("rpy_deg", [0.0, 0.0, 0.0])[2]),
        "pregrasp_targets_rad": {
            name: float(base["control"]["grasp_targets_rad"][name])
            for name in ACTIVE_ACTUATORS
        },
        "target_assignment": seed_assignment,
    }
    seed_score, seed_diagnostic = _static_score(
        model,
        data,
        info,
        base,
        seed_parameters,
        thresholds,
        distal_geom_ids,
        distal_site_ids,
    )
    entries[seed_assignment].append(
        (seed_score, 1, seed_parameters, seed_diagnostic)
    )
    capacity[seed_assignment] -= 1

    dimensions = 6 + len(ACTIVE_ACTUATORS)
    total_samples = 0
    for pitch_index, pitch in enumerate(definition.search_bounds.palm_pitch_values_deg):
        rng = np.random.default_rng(int(seed) + pitch_index * 1_000_003)
        matrix = _latin_hypercube(samples_per_pitch, dimensions, rng)
        for row_index, row in enumerate(matrix):
            candidate_id = pitch_index * samples_per_pitch + row_index
            parameters = _candidate_parameters_v3(
                row,
                pitch_deg=pitch,
                candidate_id=candidate_id,
                definition=definition,
            )
            score, diagnostic = _static_score(
                model,
                data,
                info,
                base,
                parameters,
                thresholds,
                distal_geom_ids,
                distal_site_ids,
            )
            assignment = parameters["target_assignment"]
            heap = entries[assignment]
            item = (score, -candidate_id, parameters, diagnostic)
            face_capacity = capacity[assignment]
            # The seed occupies one slot only for its own face.
            maximum = face_capacity + int(assignment == seed_assignment)
            if len(heap) < maximum:
                heapq.heappush(heap, item)
            elif maximum and item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
            total_samples += 1

    ranked = sorted(
        [item for heap in entries.values() for item in heap],
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    candidates = tuple(_materialize_v3_candidate(base, item[2]) for item in ranked)
    diagnostics = tuple(item[3] for item in ranked)
    return KinematicScreenResult(
        seed=int(seed),
        sample_count=total_samples,
        retained_count=len(candidates),
        candidates=candidates,
        diagnostics=diagnostics,
    )


def _static_diagnostic_rank(diagnostic: Mapping[str, Any]) -> tuple[Any, ...]:
    distances = diagnostic.get("target_site_signed_distance_m", ())
    finite_distance = sum(
        abs(float(value)) if value is not None and math.isfinite(float(value)) else 1.0
        for value in distances
    )
    score = tuple(float(value) for value in diagnostic.get("score", ()))
    penetration_ok = score[2] if len(score) > 2 else 0.0
    return (
        float(not bool(diagnostic.get("forbidden_contact", True))),
        penetration_ok,
        float(diagnostic.get("clean_target_contact_count", 0)),
        float(diagnostic.get("near_target_face_count", 0)),
        -float(diagnostic.get("max_penetration_m", math.inf)),
        -finite_distance,
        *score,
        -float(diagnostic.get("candidate_id", math.inf)),
    )


def _local_grasp_candidates_v3(
    parent: dict[str, Any],
    *,
    count: int,
    seed: int,
    definition: ExperimentDefinition,
    preserve_manipulation_delta: bool = False,
) -> list[dict[str, Any]]:
    """Locally perturb pose and verified grasp targets, never a second pose."""

    if count <= 0:
        return []
    rng = np.random.default_rng(seed)
    bounds = definition.search_bounds
    parent_rpy = np.asarray(parent["hand_pose"]["rpy_deg"], dtype=np.float64)
    parent_cube_in_root = _cube_position_in_root(parent)
    parent_yaw = float(parent["cube"]["rpy_deg"][2])
    result: list[dict[str, Any]] = []
    for _ in range(count):
        candidate = copy.deepcopy(parent)
        rpy = parent_rpy + rng.uniform(-2.0, 2.0, 3)
        rpy[0] = np.clip(rpy[0], *bounds.hand_roll_deg)
        rpy[1] = np.clip(rpy[1], *bounds.palm_pitch_deg)
        rpy[2] = np.clip(rpy[2], *bounds.hand_yaw_deg)
        cube_in_root = parent_cube_in_root + rng.uniform(-0.0025, 0.0025, 3)
        cube_in_root = np.asarray(
            [
                np.clip(cube_in_root[index], *bounds.cube_position_in_root_m[axis])
                for index, axis in enumerate(("x", "y", "z"))
            ],
            dtype=np.float64,
        )
        cube_yaw = float(np.clip(parent_yaw + rng.uniform(-3.0, 3.0), *bounds.cube_yaw_deg))
        candidate["hand_pose"]["rpy_deg"] = rpy.tolist()
        candidate["cube"]["rpy_deg"] = [0.0, 0.0, cube_yaw]
        candidate["hand_pose"]["translation_m"] = (
            _cube_world_position(candidate) - _rpy_matrix(rpy) @ cube_in_root
        ).tolist()
        for name in ACTIVE_ACTUATORS:
            lower, upper = bounds.actuator_targets_rad[name]
            span = upper - lower
            value = float(parent["control"]["grasp_targets_rad"][name])
            candidate["control"]["grasp_targets_rad"][name] = float(
                np.clip(value + rng.normal(0.0, 0.02 * span), lower, upper)
            )
        if not preserve_manipulation_delta:
            candidate["control"]["manipulation_delta_rad"] = {
                name: 0.0 for name in ACTIVE_ACTUATORS
            }
        validate_config(candidate)
        result.append(candidate)
    return result


def _run_v3_stage(
    configs: Sequence[dict[str, Any]],
    *,
    next_id: int,
    workers: int,
    run_candidates: Callable[[list[tuple[int, dict[str, Any]]], int], list[dict[str, Any]]],
    stage: str,
    material_policy: str,
) -> tuple[list[dict[str, Any]], int]:
    payloads = [
        (next_id + index, copy.deepcopy(config))
        for index, config in enumerate(configs)
    ]
    submitted = {
        candidate_id: copy.deepcopy(config) for candidate_id, config in payloads
    }
    results = run_candidates(payloads, workers) if payloads else []
    expected_ids = tuple(candidate_id for candidate_id, _ in payloads)
    try:
        received_ids = tuple(int(result["candidate_id"]) for result in results)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{stage} runner returned an invalid candidate_id") from error
    if (
        len(results) != len(payloads)
        or len(set(received_ids)) != len(received_ids)
        or set(received_ids) != set(expected_ids)
    ):
        raise RuntimeError(
            f"{stage} runner result IDs must exactly match submitted IDs; "
            f"expected={expected_ids}, received={received_ids}"
        )
    for result in results:
        candidate_id = int(result["candidate_id"])
        if result.get("config") != submitted[candidate_id]:
            raise RuntimeError(
                f"{stage} runner rebound candidate_id={candidate_id} to a "
                "different configuration"
            )
        result["search_stage"] = stage
        result["material_policy"] = material_policy
    ordered = sorted(results, key=lambda result: int(result["candidate_id"]))
    return ordered, next_id + len(payloads)


def _screen_size_v3(
    config: dict[str, Any],
    *,
    edge_m: float,
    samples_per_pitch: int,
    retain: int,
    seed: int,
    definition: ExperimentDefinition,
    screen_candidates: Callable[..., KinematicScreenResult],
    stage: str,
) -> dict[str, Any]:
    campaign = definition.size_campaign
    assert campaign is not None
    base = _fixed_material_config(
        config,
        edge_m=edge_m,
        mass_kg=campaign.discovery_mass_kg,
        friction=campaign.discovery_friction,
    )
    screen = screen_candidates(
        base,
        samples_per_pitch=samples_per_pitch,
        retain=retain,
        seed=stage_seed(
            stage={"coarse": "coarse", "neighbor": "neighbor", "exact": "exact"}[stage],
            edge_m=edge_m,
            pitch_deg=0.0,
            seed=seed,
        ),
        definition=definition,
    )
    pairs = list(zip(screen.candidates, screen.diagnostics, strict=True))
    pairs.sort(key=lambda pair: _static_diagnostic_rank(pair[1]), reverse=True)
    max_penetration = float(config["acceptance"]["max_penetration_m"])
    eligible = [
        pair
        for pair in pairs
        if static_candidate_advances(pair[1], max_penetration_m=max_penetration)
    ]
    clean_three = sum(
        int(item[1].get("clean_target_contact_count", 0)) >= 3 for item in pairs
    )
    near_three = sum(
        int(item[1].get("near_target_face_count", 0)) >= 3 for item in pairs
    )
    best_score = (
        tuple(float(value) for value in pairs[0][1].get("score", ()))
        if pairs
        else ()
    )
    summary = SizeScreenSummary(
        edge_m=float(edge_m),
        sample_count=int(screen.sample_count),
        eligible_count=len(eligible),
        clean_three_count=clean_three,
        near_three_count=near_three,
        best_score=best_score,
    )
    record = {
        "stage": stage,
        "edge_m": float(edge_m),
        "edge_mm": float(edge_m) * 1000.0,
        "seed": int(screen.seed),
        "samples_per_pitch": int(samples_per_pitch),
        "sample_count": int(screen.sample_count),
        "retained_count": len(pairs),
        "eligible_count": len(eligible),
        "clean_three_count": clean_three,
        "near_three_count": near_three,
        "top_diagnostics": [pair[1] for pair in pairs[:20]],
    }
    return {
        "edge_m": float(edge_m),
        "pairs": pairs,
        "eligible": eligible,
        "summary": summary,
        "record": record,
    }


def tune_larger_cube_grasp_then_lift(
    config: dict[str, Any],
    *,
    workers: int,
    seed: int,
    run_candidates: Callable[[list[tuple[int, dict[str, Any]]], int], list[dict[str, Any]]],
    rank_candidate: Callable[[dict[str, Any]], tuple[float, ...]],
    kinematic_samples_per_pitch: int | None = None,
    dynamic_candidate_count: int | None = None,
    local_refine_seed_count: int | None = None,
    local_refine_per_seed: int | None = None,
    final_candidate_count: int | None = None,
    perturbations_per_final: int | None = None,
    fallback_physics_count: int | None = None,
    fallback_kinematic_samples_per_pitch: int | None = None,
    perturb_cases: Callable[..., list[dict[str, Any]]] | None = None,
    screen_candidates: Callable[..., KinematicScreenResult] = kinematic_screen_v3,
) -> dict[str, Any]:
    """Execute the declared grasp-first campaign with explicit hard gates."""

    validate_config(config)
    definition = resolve_experiment(config)
    if definition.experiment_id != EXPERIMENT_ID or definition.control_protocol is None:
        raise ValueError("selected experiment is not the v3 larger-cube campaign")
    campaign = definition.size_campaign
    assert campaign is not None
    coarse_samples = (
        campaign.coarse_samples_per_pitch
        if kinematic_samples_per_pitch is None
        else int(kinematic_samples_per_pitch)
    )
    neighbor_samples = (
        campaign.odd_samples_per_pitch
        if kinematic_samples_per_pitch is None
        else max(1, 2 * coarse_samples)
    )
    fine_samples = (
        campaign.fine_samples_per_pitch
        if fallback_kinematic_samples_per_pitch is None
        else int(fallback_kinematic_samples_per_pitch)
    )
    dynamic_total = (
        campaign.dynamic_candidate_count
        if dynamic_candidate_count is None
        else int(dynamic_candidate_count)
    )
    grasp_seed_total = (
        definition.search_bounds.local_refine_seed_count
        if local_refine_seed_count is None
        else int(local_refine_seed_count)
    )
    grasp_refine_count = (
        campaign.local_refine_per_seed
        if local_refine_per_seed is None
        else int(local_refine_per_seed)
    )
    finalist_count = (
        campaign.finalist_count
        if final_candidate_count is None
        else int(final_candidate_count)
    )
    perturbation_count = (
        campaign.perturbations_per_final
        if perturbations_per_final is None
        else int(perturbations_per_final)
    )
    density_max = (
        campaign.constant_density_max_candidates
        if fallback_physics_count is None
        else int(fallback_physics_count)
    )
    for label, value in (
        ("coarse_samples", coarse_samples),
        ("neighbor_samples", neighbor_samples),
        ("fine_samples", fine_samples),
        ("dynamic_total", dynamic_total),
        ("grasp_seed_total", grasp_seed_total),
        ("grasp_refine_count", grasp_refine_count),
        ("finalist_count", finalist_count),
        ("perturbation_count", perturbation_count),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    if density_max < 0:
        raise ValueError("fallback physics count must be non-negative")

    coarse_bundles = [
        _screen_size_v3(
            config,
            edge_m=edge,
            samples_per_pitch=coarse_samples,
            retain=max(64, dynamic_total // campaign.exact_size_count),
            seed=seed,
            definition=definition,
            screen_candidates=screen_candidates,
            stage="coarse",
        )
        for edge in campaign.coarse_edges_m
    ]
    selected_coarse_edges = select_coarse_edges_m(
        [bundle["summary"] for bundle in coarse_bundles],
        LargerCubeCampaignPlan(seed=seed),
    )
    neighbor_edges = adjacent_neighbor_edges_m(
        selected_coarse_edges, LargerCubeCampaignPlan(seed=seed)
    )
    neighbor_bundles = [
        _screen_size_v3(
            config,
            edge_m=edge,
            samples_per_pitch=neighbor_samples,
            retain=max(64, dynamic_total // campaign.exact_size_count),
            seed=seed,
            definition=definition,
            screen_candidates=screen_candidates,
            stage="neighbor",
        )
        for edge in neighbor_edges
    ]
    coarse_by_edge = {
        _edge_key(bundle["edge_m"]): bundle for bundle in coarse_bundles
    }
    exact_selection_pool = [
        coarse_by_edge[_edge_key(edge)]["summary"] for edge in selected_coarse_edges
    ] + [bundle["summary"] for bundle in neighbor_bundles]
    selected_exact_edges = select_exact_edges_m(
        exact_selection_pool, LargerCubeCampaignPlan(seed=seed)
    )
    per_size_dynamic = max(1, math.ceil(dynamic_total / campaign.exact_size_count))
    exact_bundles = [
        _screen_size_v3(
            config,
            edge_m=edge,
            samples_per_pitch=fine_samples,
            retain=max(len(definition.candidate_faces), per_size_dynamic),
            seed=seed,
            definition=definition,
            screen_candidates=screen_candidates,
            stage="exact",
        )
        for edge in selected_exact_edges
    ]

    dynamic_configs: list[dict[str, Any]] = []
    quotient, remainder = divmod(dynamic_total, len(exact_bundles))
    for index, bundle in enumerate(exact_bundles):
        requested = quotient + int(index < remainder)
        eligible_ids = {id(pair[0]) for pair in bundle["eligible"]}
        ordered = sorted(
            bundle["pairs"],
            key=lambda pair: (
                float(id(pair[0]) in eligible_ids),
                _static_diagnostic_rank(pair[1]),
            ),
            reverse=True,
        )
        selected = ordered[:requested]
        dynamic_configs.extend(pair[0] for pair in selected)
        bundle["record"]["dynamic_selection"] = {
            "requested_count": requested,
            "eligible_available_count": len(bundle["eligible"]),
            "selected_count": len(selected),
            "near_gate_fill_count": max(0, len(selected) - len(bundle["eligible"])),
        }
    if not dynamic_configs:
        raise RuntimeError("exact v3 screens retained no dynamic candidates")

    next_id = 0
    initial_runs, next_id = _run_v3_stage(
        dynamic_configs,
        next_id=next_id,
        workers=workers,
        run_candidates=run_candidates,
        stage="fixed_mass_close_verify",
        material_policy="fixed_20g_control",
    )

    ranked_initial = sorted(initial_runs, key=rank_candidate, reverse=True)
    grasp_parents: list[dict[str, Any]] = []
    seed_quotient, seed_remainder = divmod(grasp_seed_total, len(selected_exact_edges))
    for edge_index, edge in enumerate(selected_exact_edges):
        capacity = seed_quotient + int(edge_index < seed_remainder)
        group = [
            item
            for item in ranked_initial
            if _edge_key(item["config"]["cube"]["edge_m"]) == _edge_key(edge)
        ]
        grasp_parents.extend(group[:capacity])
    grasp_local_configs: list[dict[str, Any]] = []
    for parent_index, parent in enumerate(grasp_parents):
        grasp_local_configs.extend(
            _local_grasp_candidates_v3(
                parent["config"],
                count=grasp_refine_count,
                seed=seed + _STAGE_SEED_OFFSETS["grasp_refine"] + parent_index,
                definition=definition,
            )
        )
    grasp_local_runs, next_id = _run_v3_stage(
        grasp_local_configs,
        next_id=next_id,
        workers=workers,
        run_candidates=run_candidates,
        stage="fixed_mass_grasp_refinement",
        material_policy="fixed_20g_control",
    )
    grasp_results = initial_runs + grasp_local_runs
    grasp_passes = [item for item in grasp_results if grasp_succeeded(item)]

    manipulation_parents = sorted(
        grasp_passes, key=rank_candidate, reverse=True
    )[: campaign.manipulation_seed_count]
    manipulation_configs: list[dict[str, Any]] = []
    assert definition.search_bounds.manipulation_delta_rad is not None
    for parent_index, parent in enumerate(manipulation_parents):
        manipulation_configs.extend(
            sample_manipulation_delta_candidates(
                parent["config"],
                count=campaign.manipulation_refine_per_seed,
                seed=(
                    seed
                    + _STAGE_SEED_OFFSETS["manipulation_refine"]
                    + parent_index
                ),
                delta_bounds_rad=definition.search_bounds.manipulation_delta_rad,
                absolute_target_bounds_rad=definition.search_bounds.actuator_targets_rad,
                validator=validate_config,
            )
        )
    manipulation_runs, next_id = _run_v3_stage(
        manipulation_configs,
        next_id=next_id,
        workers=workers,
        run_candidates=run_candidates,
        stage="fixed_mass_manipulation_refinement",
        material_policy="fixed_20g_control",
    )
    fixed_results = grasp_results + manipulation_runs
    # A zero-delta CLOSE/VERIFY run is evidence of a stable grasp only.  Even if
    # incidental physics happens to satisfy the global checks, it must not skip
    # the declared operation-search stage or authorize a density transition.
    fixed_full_passes = [
        item for item in manipulation_runs if full_succeeded(item)
    ]
    best_fixed_mass = (
        copy.deepcopy(
            sorted(
                fixed_full_passes,
                key=larger_cube_candidate_rank,
                reverse=True,
            )[0]
        )
        if fixed_full_passes
        else None
    )

    density_configs: list[dict[str, Any]] = []
    for parent in sorted(
        fixed_full_passes, key=larger_cube_candidate_rank, reverse=True
    )[:density_max]:
        candidate = copy.deepcopy(parent["config"])
        edge = float(candidate["cube"]["edge_m"])
        candidate["cube"]["mass_kg"] = campaign.constant_density_mass_kg(edge)
        validate_config(candidate)
        density_configs.append(candidate)
    density_runs, next_id = _run_v3_stage(
        density_configs,
        next_id=next_id,
        workers=workers,
        run_candidates=run_candidates,
        stage="constant_density_reevaluation",
        material_policy="constant_density",
    )
    density_full_passes = [item for item in density_runs if full_succeeded(item)]
    density_local_configs: list[dict[str, Any]] = []
    if density_runs and not density_full_passes:
        density_parents = sorted(
            density_runs, key=larger_cube_candidate_rank, reverse=True
        )[: campaign.density_refine_seed_count]
        for parent_index, parent in enumerate(density_parents):
            density_local_configs.extend(
                sample_manipulation_delta_candidates(
                    parent["config"],
                    count=campaign.density_refine_per_seed,
                    seed=(
                        seed
                        + _STAGE_SEED_OFFSETS["constant_density_refine"]
                        + parent_index
                    ),
                    delta_bounds_rad=definition.search_bounds.manipulation_delta_rad,
                    local_radius_rad=0.05,
                    absolute_target_bounds_rad=definition.search_bounds.actuator_targets_rad,
                    validator=validate_config,
                )
            )
    density_local_runs, next_id = _run_v3_stage(
        density_local_configs,
        next_id=next_id,
        workers=workers,
        run_candidates=run_candidates,
        stage="constant_density_refinement",
        material_policy="constant_density",
    )
    density_results = density_runs + density_local_runs
    density_full_passes = [item for item in density_results if full_succeeded(item)]

    # Report the most advanced stage actually reached.  This prevents an
    # incidental full-looking zero-delta CLOSE/VERIFY observation from
    # outranking a genuine manipulation near miss.
    final_pool = (
        density_results
        if density_results
        else manipulation_runs
        if manipulation_runs
        else grasp_results
    )
    hard_finalists = [item for item in final_pool if full_succeeded(item)]
    finalist_source = hard_finalists if hard_finalists else final_pool
    finalists = sorted(
        finalist_source, key=larger_cube_candidate_rank, reverse=True
    )[:finalist_count]
    probe_records: list[dict[str, Any]] = []
    if perturb_cases is not None:
        for finalist_index, parent in enumerate(finalists):
            cases = perturb_cases(
                parent["config"],
                count=perturbation_count,
                seed=seed + _STAGE_SEED_OFFSETS["final_perturbation"] + finalist_index,
            )
            trials, next_id = _run_v3_stage(
                cases,
                next_id=next_id,
                workers=workers,
                run_candidates=run_candidates,
                stage="final_perturbation_probe",
                material_policy="local_material_and_pose_perturbation",
            )
            probe_records.append(
                {
                    "candidate_id": int(parent["candidate_id"]),
                    "passes": sum(full_succeeded(trial) for trial in trials),
                    "trial_count": len(trials),
                    "trials": trials,
                }
            )
    probe_lookup = {record["candidate_id"]: record for record in probe_records}

    def selection_key(item: dict[str, Any]) -> tuple[Any, ...]:
        probe = probe_lookup.get(int(item["candidate_id"]), {"passes": -1})
        return larger_cube_candidate_rank(
            item, perturbation_passes=int(probe["passes"])
        )

    ranked_final_pool = sorted(final_pool, key=selection_key, reverse=True)
    best = copy.deepcopy(ranked_final_pool[0])
    best["local_perturbation_probe"] = copy.deepcopy(
        probe_lookup.get(
            int(best["candidate_id"]),
            {
                "candidate_id": int(best["candidate_id"]),
                "passes": 0,
                "trial_count": 0,
                "trials": [],
            },
        )
    )
    fixed_mass_success = bool(fixed_full_passes)
    constant_density_success = bool(density_full_passes)
    grasp_success = bool(grasp_passes)
    if constant_density_success:
        campaign_classification = "validated_constant_density"
        stop_reason = "constant_density_full_pass_found"
    elif fixed_mass_success:
        campaign_classification = "validated_fixed_mass_ablation_only"
        stop_reason = "no_constant_density_full_pass"
    elif grasp_success:
        campaign_classification = "stable_grasp_only"
        stop_reason = "stable_grasp_found_no_fixed_mass_manipulation_pass"
    else:
        campaign_classification = "not_validated"
        stop_reason = "no_stable_grasp_acquired"
    best["config"]["experiment_status"] = {
        "classification": campaign_classification,
        "passed": constant_density_success,
        "grasp_success": grasp_success,
        "manipulation_success": bool(fixed_mass_success or constant_density_success),
        "full_success": constant_density_success,
        "fixed_mass_discovery_passed": fixed_mass_success,
        "constant_density_passed": constant_density_success,
        "stop_reason": stop_reason,
        "note": (
            "A constant-density candidate passed the stable-grasp and manipulation gates."
            if constant_density_success
            else "Only the fixed-20 g geometry ablation passed; constant-density validation failed."
            if fixed_mass_success
            else "A stable grasp was acquired, but no fixed-mass manipulation passed."
            if grasp_success
            else "No candidate acquired the declared continuous stable grasp."
        ),
    }
    if best_fixed_mass is not None:
        best_fixed_mass["config"]["experiment_status"] = {
            "classification": "validated_fixed_mass_ablation",
            "passed": False,
            "grasp_success": True,
            "manipulation_success": True,
            "full_success": True,
            "fixed_mass_discovery_passed": True,
            "constant_density_passed": False,
            "note": "This is a fixed-20 g geometry ablation, not final validation.",
        }

    screen_records = {
        "coarse": [bundle["record"] for bundle in coarse_bundles],
        "neighbor": [bundle["record"] for bundle in neighbor_bundles],
        "exact": [bundle["record"] for bundle in exact_bundles],
    }
    static_search_budget = _static_search_budget_report(
        campaign=campaign,
        pitch_count=len(definition.search_bounds.palm_pitch_values_deg),
        selected_coarse_edges_m=selected_coarse_edges,
        neighbor_edges_m=neighbor_edges,
        coarse_samples_per_pitch=coarse_samples,
        neighbor_samples_per_pitch=neighbor_samples,
        exact_samples_per_pitch=fine_samples,
        screen_records=screen_records,
    )
    per_size_outcomes = _per_size_outcomes(
        selected_exact_edges_m=selected_exact_edges,
        initial_runs=initial_runs,
        grasp_local_runs=grasp_local_runs,
        manipulation_runs=manipulation_runs,
        density_runs=density_runs,
        density_local_runs=density_local_runs,
    )

    def result_with_probe(item: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(item)
        result["local_perturbation_probe"] = copy.deepcopy(
            probe_lookup.get(
                int(item["candidate_id"]),
                {
                    "candidate_id": int(item["candidate_id"]),
                    "passes": 0,
                    "trial_count": 0,
                    "trials": [],
                },
            )
        )
        return result

    all_main_results = fixed_results + density_results
    return {
        "experiment_id": definition.experiment_id,
        "campaign_kind": "larger_cube_grasp_then_lift",
        "campaign_classification": campaign_classification,
        "campaign_status": copy.deepcopy(best["config"]["experiment_status"]),
        "seed": int(seed),
        "workers": int(workers),
        "campaign_manifest": campaign_manifest(LargerCubeCampaignPlan(seed=seed)),
        "static_search_budget": static_search_budget,
        "per_size_outcomes": per_size_outcomes,
        "size_stages": screen_records,
        "selected_coarse_edges_m": [float(value) for value in selected_coarse_edges],
        "neighbor_edges_m": [float(value) for value in neighbor_edges],
        "selected_exact_edges_m": [float(value) for value in selected_exact_edges],
        "kinematic_sample_count": sum(
            record["sample_count"]
            for records in screen_records.values()
            for record in records
        ),
        "initial_dynamic_count": len(initial_runs),
        "grasp_refinement_count": len(grasp_local_runs),
        "stable_grasp_candidate_count": len(grasp_passes),
        "manipulation_refinement_count": len(manipulation_runs),
        "fixed_mass_passing_candidates": len(fixed_full_passes),
        "constant_density_reevaluation_count": len(density_runs),
        "constant_density_refinement_count": len(density_local_runs),
        "constant_density_passing_candidates": len(density_full_passes),
        "candidate_count": len(all_main_results),
        "perturbation_probe_count": sum(
            record["trial_count"] for record in probe_records
        ),
        "simulation_count": len(all_main_results)
        + sum(record["trial_count"] for record in probe_records),
        "passing_candidates": len(density_full_passes),
        "grasp_success": grasp_success,
        "fixed_mass_success": fixed_mass_success,
        "constant_density_success": constant_density_success,
        "nominal_success": constant_density_success,
        "stop_reason": stop_reason,
        "best_fixed_mass": best_fixed_mass,
        "best": best,
        "top_candidates": [
            result_with_probe(item) for item in ranked_final_pool[:20]
        ],
        "local_perturbation_probes": probe_records,
    }


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_PLAN",
    "EXPERIMENT_ID",
    "FINGER_FACE_OPTIONS",
    "LargerCubeCampaignPlan",
    "SizeScreenSummary",
    "StaticScreenJob",
    "adjacent_neighbor_edges_m",
    "build_manipulation_refinement_candidates",
    "campaign_manifest",
    "coarse_screen_jobs",
    "constant_density_candidates",
    "deterministic_rank_results",
    "exact_screen_jobs",
    "full_succeeded",
    "grasp_succeeded",
    "larger_cube_candidate_rank",
    "larger_cube_robustness_cases",
    "manipulation_succeeded",
    "manipulation_targets",
    "neighbor_screen_jobs",
    "robustness_edge_window_m",
    "sample_manipulation_delta_candidates",
    "select_coarse_edges_m",
    "select_exact_edges_m",
    "select_grasp_refinement_parents",
    "select_manipulation_parents",
    "select_top_size_summaries",
    "size_summary_rank",
    "stage_seed",
    "kinematic_screen_v3",
    "tune_larger_cube_grasp_then_lift",
]
