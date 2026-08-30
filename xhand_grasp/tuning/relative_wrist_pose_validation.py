"""Validation helpers for the schema-v11 relative-wrist-pose campaign.

This module is deliberately side-effect free: it computes equal-density
masses, produces deterministic full-reset pose/friction trials, and classifies
already-computed pass flags.  It never launches a campaign or writes an
artifact.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..config import validate_config
from ..experiments.opposed_face_palm_down_larger_relative_wrist_pose_actual_contact_smooth_vertical_lift import (
    DENSITY_REVALIDATION_KG_M3,
    EDGES_M,
    EXPERIMENT_ID,
    ROBUSTNESS,
    VALIDATION_LABELS,
)


ValidationFamily = Literal["fixed_160g", "constant_density"]
ValidationStage = Literal["grasp", "manipulation", "robust"]

FIXED_MASS_KG = 0.160
PERTURBATION_COUNT = 50
REQUIRED_PERTURBATION_PASSES = 45
DEFAULT_SEED = 20260821
PERTURBATION_FAMILY = "v11_best_fixed_160g_pose_friction_50"

FIXED_160G_VALIDATION_LABELS: dict[str, str] = dict(VALIDATION_LABELS)
CONSTANT_DENSITY_VALIDATION_LABELS: dict[str, str] = {
    "grasp": "validated_constant_density_grasp",
    "manipulation": "validated_constant_density_manipulation",
    "robust": "validated_constant_density_robust_full_success",
}


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def constant_density_mass_kg(edge_m: float) -> float:
    """Return the 740.7407 kg/m^3 mass for an edge inside 85--104 mm."""

    edge = _finite(edge_m, "edge_m")
    minimum = min(EDGES_M)
    maximum = max(EDGES_M)
    if not minimum - 1e-12 <= edge <= maximum + 1e-12:
        raise ValueError(
            f"edge_m must remain inside the registered {minimum:g}--{maximum:g} m range"
        )
    return float(DENSITY_REVALIDATION_KG_M3 * edge**3)


def validation_label(
    family: ValidationFamily,
    stage: ValidationStage,
) -> str:
    """Resolve a validation label without conflating fixed and equal density."""

    tables = {
        "fixed_160g": FIXED_160G_VALIDATION_LABELS,
        "constant_density": CONSTANT_DENSITY_VALIDATION_LABELS,
    }
    if family not in tables:
        raise ValueError("family must be 'fixed_160g' or 'constant_density'")
    if stage not in {"grasp", "manipulation", "robust"}:
        raise ValueError("stage must be grasp, manipulation or robust")
    return tables[family][stage]


@dataclass(frozen=True, slots=True)
class DensityRevalidationCase:
    """One registered edge and its equal-density mass."""

    edge_m: float
    mass_kg: float
    density_kg_m3: float = DENSITY_REVALIDATION_KG_M3
    family: str = "constant_density"

    @property
    def edge_mm(self) -> float:
        return self.edge_m * 1000.0

    def as_dict(self) -> dict[str, float | str]:
        return {
            "edge_m": self.edge_m,
            "edge_mm": self.edge_mm,
            "mass_kg": self.mass_kg,
            "mass_g": self.mass_kg * 1000.0,
            "density_kg_m3": self.density_kg_m3,
            "family": self.family,
        }


def density_revalidation_cases() -> tuple[DensityRevalidationCase, ...]:
    """Return all 20 registered 85--104 mm equal-density cases."""

    return tuple(
        DensityRevalidationCase(
            edge_m=float(edge),
            mass_kg=constant_density_mass_kg(float(edge)),
        )
        for edge in EDGES_M
    )


def materialize_constant_density_revalidation(
    config: Mapping[str, Any],
    *,
    source_candidate_id: str | int,
) -> dict[str, Any]:
    """Create one full-reset equal-density revalidation configuration.

    The registered schema-v11 discovery campaign is intentionally fixed at
    160 g, so an equal-density rerun must use the validator's explicit
    ``robustness_trial`` context.  This keeps the physical override legal
    without allowing it to inherit the source catalog's nominal success
    claim.  The post-validation runner awards a density label only after the
    resulting simulation independently passes every full-success check.
    """

    base = copy.deepcopy(dict(config))
    if base.get("schema_version") != 11 or base.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(
            "constant-density revalidation requires the registered schema-v11 experiment"
        )
    if base.get("run_context") is not None:
        raise ValueError(
            "constant-density revalidation requires a canonical nominal config"
        )
    validate_config(base)
    edge_m = float(base["cube"]["edge_m"])
    source_mass_kg = float(base["cube"]["mass_kg"])
    if not math.isclose(source_mass_kg, FIXED_MASS_KG, abs_tol=1e-12):
        raise ValueError("constant-density source must use the registered fixed 160 g mass")
    density_mass_kg = constant_density_mass_kg(edge_m)
    base.pop("experiment_status", None)
    base["run_context"] = {"kind": "robustness_trial"}
    base["cube"]["mass_kg"] = density_mass_kg
    metadata = copy.deepcopy(dict(base.get("candidate_metadata", {})))
    metadata["post_validation"] = {
        "family": "constant_density_full_reset",
        "source_candidate_id": str(source_candidate_id),
        "source_mass_kg": source_mass_kg,
        "resolved_mass_kg": density_mass_kg,
        "density_kg_m3": DENSITY_REVALIDATION_KG_M3,
        "full_reset_rerun": True,
        "initial_state_source": "configured_no_contact_reset",
        "checkpoint_used": False,
    }
    base["candidate_metadata"] = metadata
    validate_config(base)
    return base


def _scale(unit: float, bounds: Sequence[float]) -> float:
    return float(bounds[0] + unit * (bounds[1] - bounds[0]))


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        values[:, dimension] = (
            rng.permutation(count) + rng.random(count)
        ) / count
    return values


@dataclass(frozen=True, slots=True)
class PoseFrictionPerturbationCase:
    """One deterministic seven-dimensional full-reset perturbation."""

    trial: int
    seed: int
    cube_center_xy_delta_m: tuple[float, float]
    cube_gap_delta_m: float
    cube_rpy_delta_deg: tuple[float, float, float]
    friction_delta: float
    family: str = PERTURBATION_FAMILY
    full_reset_rerun: bool = True
    initial_state_source: str = "configured_no_contact_reset"
    checkpoint_used: bool = False

    def resolved_perturbations(self) -> dict[str, Any]:
        return {
            "cube_center_xy_delta_m": list(self.cube_center_xy_delta_m),
            "cube_gap_delta_m": self.cube_gap_delta_m,
            "cube_rpy_delta_deg": list(self.cube_rpy_delta_deg),
            "friction_delta": self.friction_delta,
            # The registered v11 robustness campaign intentionally does not
            # perturb mass, so pose/friction robustness cannot masquerade as
            # equal-density revalidation.
            "mass_scale": 1.0,
        }


def generate_pose_friction_perturbation_cases(
    *, seed: int = DEFAULT_SEED
) -> tuple[PoseFrictionPerturbationCase, ...]:
    """Generate the registered 50-case deterministic pose/friction envelope."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    matrix = _latin_hypercube(PERTURBATION_COUNT, 7, seed)
    parameters = ROBUSTNESS
    cases: list[PoseFrictionPerturbationCase] = []
    for trial, row in enumerate(matrix):
        cases.append(
            PoseFrictionPerturbationCase(
                trial=trial,
                seed=seed,
                cube_center_xy_delta_m=(
                    _scale(float(row[0]), parameters.position_xy_delta_m),
                    _scale(float(row[1]), parameters.position_xy_delta_m),
                ),
                cube_gap_delta_m=_scale(
                    float(row[2]), parameters.z_offset_delta_m
                ),
                cube_rpy_delta_deg=tuple(
                    _scale(float(row[3 + axis]), parameters.rpy_delta_deg)
                    for axis in range(3)
                ),  # type: ignore[arg-type]
                friction_delta=_scale(
                    float(row[6]), parameters.friction_delta
                ),
            )
        )
    return tuple(cases)


def apply_pose_friction_perturbation_case(
    config: Mapping[str, Any],
    case: PoseFrictionPerturbationCase,
    *,
    source_candidate_id: str | int,
) -> dict[str, Any]:
    """Materialize one v11 robustness trial without changing its mass."""

    base = copy.deepcopy(dict(config))
    if base.get("schema_version") != 11 or base.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("pose/friction perturbations require the registered schema-v11 experiment")
    if base.get("run_context") is not None:
        raise ValueError("pose/friction perturbations require a canonical nominal config")
    validate_config(base)
    original_mass = float(base["cube"]["mass_kg"])
    xy = np.asarray(base["cube"]["center_xy_m"], dtype=np.float64)
    rpy = np.asarray(base["cube"].get("rpy_deg", (0.0, 0.0, 0.0)), dtype=np.float64)
    base.pop("experiment_status", None)
    base["run_context"] = {"kind": "robustness_trial"}
    base["cube"]["center_xy_m"] = (
        xy + np.asarray(case.cube_center_xy_delta_m, dtype=np.float64)
    ).tolist()
    base["cube"]["z_offset_m"] = float(
        base["cube"].get("z_offset_m", 0.0) + case.cube_gap_delta_m
    )
    base["cube"]["rpy_deg"] = (
        rpy + np.asarray(case.cube_rpy_delta_deg, dtype=np.float64)
    ).tolist()
    base["cube"]["friction"] = float(
        base["cube"]["friction"] + case.friction_delta
    )
    base["cube"]["mass_kg"] = original_mass
    metadata = copy.deepcopy(dict(base.get("candidate_metadata", {})))
    metadata["robustness_trial"] = {
        "family": case.family,
        "seed": case.seed,
        "trial": case.trial,
        "source_candidate_id": str(source_candidate_id),
        "resolved_perturbations": case.resolved_perturbations(),
        "full_reset_rerun": case.full_reset_rerun,
        "initial_state_source": case.initial_state_source,
        "checkpoint_used": case.checkpoint_used,
    }
    base["candidate_metadata"] = metadata
    validate_config(base)
    return base


@dataclass(frozen=True, slots=True)
class RobustnessClassification:
    family: ValidationFamily
    nominal_full_success: bool
    perturbation_count: int
    perturbation_passes: int
    required_perturbation_passes: int
    registered_budget_complete: bool
    robust_passed: bool
    validation_label: str | None
    stop_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "nominal_full_success": self.nominal_full_success,
            "perturbation_count": self.perturbation_count,
            "perturbation_passes": self.perturbation_passes,
            "required_perturbation_passes": self.required_perturbation_passes,
            "registered_budget_complete": self.registered_budget_complete,
            "robust_passed": self.robust_passed,
            "validation_label": self.validation_label,
            "stop_reason": self.stop_reason,
        }


def classify_pose_friction_robustness(
    passed: Sequence[bool],
    *,
    nominal_full_success: bool,
    family: ValidationFamily = "fixed_160g",
) -> RobustnessClassification:
    """Apply the exact 45-of-50 gate without promoting failed nominals."""

    if family not in {"fixed_160g", "constant_density"}:
        raise ValueError("family must be 'fixed_160g' or 'constant_density'")
    values = tuple(passed)
    if any(not isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("passed must contain only boolean values")
    count = len(values)
    pass_count = sum(bool(value) for value in values)
    budget_complete = count == PERTURBATION_COUNT
    robust = bool(
        nominal_full_success
        and budget_complete
        and pass_count >= REQUIRED_PERTURBATION_PASSES
    )
    if robust:
        reason = "robust_passed"
    elif not nominal_full_success:
        reason = "nominal_failed_full_success"
    elif not budget_complete:
        reason = "registered_perturbation_budget_incomplete"
    else:
        reason = "perturbation_pass_count_below_45_of_50"
    return RobustnessClassification(
        family=family,
        nominal_full_success=bool(nominal_full_success),
        perturbation_count=count,
        perturbation_passes=pass_count,
        required_perturbation_passes=REQUIRED_PERTURBATION_PASSES,
        registered_budget_complete=budget_complete,
        robust_passed=robust,
        validation_label=validation_label(family, "robust") if robust else None,
        stop_reason=reason,
    )


__all__ = [
    "CONSTANT_DENSITY_VALIDATION_LABELS",
    "DEFAULT_SEED",
    "DensityRevalidationCase",
    "FIXED_160G_VALIDATION_LABELS",
    "FIXED_MASS_KG",
    "PERTURBATION_COUNT",
    "PERTURBATION_FAMILY",
    "PoseFrictionPerturbationCase",
    "REQUIRED_PERTURBATION_PASSES",
    "RobustnessClassification",
    "apply_pose_friction_perturbation_case",
    "classify_pose_friction_robustness",
    "constant_density_mass_kg",
    "density_revalidation_cases",
    "generate_pose_friction_perturbation_cases",
    "materialize_constant_density_revalidation",
    "validation_label",
]
