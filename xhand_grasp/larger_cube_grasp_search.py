"""Deterministic planning helpers for the schema-v3 larger-cube campaign.

This module intentionally does not run MuJoCo.  It turns the versioned search
contract into deterministic work items and enforces the grasp-first stage
boundary around injected simulation results.  Keeping those operations pure
makes the expensive campaign reproducible across worker counts and lets unit
tests exercise the full orchestration budget without launching a simulation.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .config import ACTIVE_ACTUATORS


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


def larger_cube_candidate_rank(result: Mapping[str, Any]) -> tuple[float, ...]:
    """Canonical final rank with candidate ID as the total-order tie break."""

    metrics = _metrics(result)
    topology = _numeric_metric(
        metrics,
        (
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
    return (
        float(full_succeeded(result)),
        float(grasp_succeeded(result)),
        _grasp_stability_margin(result),
        float(_perturbation_passes(result)),
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
            "coarse_sample_count": sum(job.samples for job in coarse_jobs),
            "dynamic_candidate_count": plan.dynamic_candidate_count,
            "grasp_refinement_count": plan.grasp_refinement_count,
            "manipulation_refinement_count": plan.manipulation_refinement_count,
            "density_refinement_count": plan.density_refinement_count,
            "final_perturbation_count": plan.finalist_count
            * plan.perturbations_per_finalist,
            "robustness_case_count": plan.robustness_case_count,
        },
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
]
