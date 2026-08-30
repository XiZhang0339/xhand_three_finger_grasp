"""Small-motion manipulability prescreen for schema-v12 grasps.

This module deliberately does not claim a manipulation success.  It branches
17 short probes from an authenticated grasp checkpoint, fits the local 6-D
object response, and scores a bounded virtual +2 mm motion.  Published grasp
evidence still comes from a complete reset-to-lock simulation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ACTIVE_ACTUATORS
from .actual_contact_manipulation import (
    GraspPhysicsCheckpoint,
    ManipulationSearchBudget,
    manipulation_delta_bounds,
    prepare_grasp_checkpoint,
    run_checkpoint_probe_set,
)


@dataclass(frozen=True, slots=True)
class ContactPointManipulabilityBudget:
    """Versioned small-signal probing contract used by schema-v12."""

    probe_epsilon_rad: float = 0.02
    maximum_delta_rad: float = 0.05
    target_upward_m: float = 0.002
    manipulate_s: float = 2.0
    hold_s: float = 1.0
    ridge: float = 1e-4

    def __post_init__(self) -> None:
        for name in (
            "probe_epsilon_rad",
            "maximum_delta_rad",
            "target_upward_m",
            "manipulate_s",
            "hold_s",
            "ridge",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.probe_epsilon_rad > self.maximum_delta_rad:
            raise ValueError("probe_epsilon_rad must not exceed maximum_delta_rad")
        if self.maximum_delta_rad > 0.05 + 1e-12:
            raise ValueError("schema-v12 prescreen deltas may not exceed 0.05 rad")
        if not math.isclose(self.target_upward_m, 0.002, abs_tol=1e-12):
            raise ValueError("schema-v12 virtual target must be exactly +2 mm")


def _probe_response_matrix(
    probe_results: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if len(probe_results) != 1 + 2 * len(ACTIVE_ACTUATORS):
        raise ValueError("manipulability prescreen requires exactly 17 probes")
    zero = next(
        (
            value
            for value in probe_results
            if value.get("probe", {}).get("kind") == "zero"
        ),
        None,
    )
    if zero is None:
        raise ValueError("probe set has no zero branch")
    bias = np.asarray(zero.get("response_6d"), dtype=np.float64)
    if bias.shape != (6,) or not np.isfinite(bias).all():
        raise ValueError("zero probe response is invalid")
    matrix = np.zeros((6, len(ACTIVE_ACTUATORS)), dtype=np.float64)
    columns: list[dict[str, Any]] = []
    for column, actuator in enumerate(ACTIVE_ACTUATORS):
        sides: dict[int, tuple[float, np.ndarray]] = {}
        for record in probe_results:
            probe = record.get("probe", {})
            if probe.get("actuator") != actuator:
                continue
            direction = int(probe.get("direction", 0))
            applied = probe.get("applied_delta_rad", {})
            step = float(applied.get(actuator, 0.0)) if isinstance(applied, Mapping) else 0.0
            response = np.asarray(record.get("response_6d"), dtype=np.float64)
            if direction in (-1, 1) and abs(step) > 1e-15 and response.shape == (6,):
                if not np.isfinite(response).all():
                    raise ValueError("probe response contains non-finite values")
                sides[direction] = (step, response)
        if set(sides) != {-1, 1}:
            raise ValueError(f"probe pair for {actuator} collapsed at command limits")
        negative_step, negative = sides[-1]
        positive_step, positive = sides[1]
        denominator = positive_step - negative_step
        if denominator <= 1e-15:
            raise ValueError(f"invalid probe spacing for {actuator}")
        matrix[:, column] = (positive - negative) / denominator
        columns.append(
            {
                "actuator": actuator,
                "negative_step_rad": float(negative_step),
                "positive_step_rad": float(positive_step),
                "column_norm": float(np.linalg.norm(matrix[:, column])),
            }
        )
    return matrix, bias, columns


def fit_contact_point_manipulability(
    probe_results: Sequence[Mapping[str, Any]],
    bounds: Mapping[str, Sequence[float]],
    *,
    budget: ContactPointManipulabilityBudget = ContactPointManipulabilityBudget(),
) -> dict[str, Any]:
    """Fit and score a bounded +2 mm virtual lift without declaring success."""

    if set(bounds) != set(ACTIVE_ACTUATORS):
        raise ValueError("bounds must name exactly the eight active actuators")
    matrix, bias, columns = _probe_response_matrix(probe_results)
    target = np.asarray((0.0, 0.0, budget.target_upward_m, 0.0, 0.0, 0.0))
    scales = np.asarray(
        (
            0.002,
            0.002,
            0.002,
            math.radians(5.0),
            math.radians(5.0),
            math.radians(5.0),
        ),
        dtype=np.float64,
    )
    weighted = matrix / scales[:, None]
    desired = (target - bias) / scales
    lower = np.asarray(
        [max(float(bounds[name][0]), -budget.maximum_delta_rad) for name in ACTIVE_ACTUATORS]
    )
    upper = np.asarray(
        [min(float(bounds[name][1]), budget.maximum_delta_rad) for name in ACTIVE_ACTUATORS]
    )
    if np.any(lower > upper):
        raise ValueError("registered control limits leave no 0.05 rad prescreen range")
    hessian = weighted.T @ weighted + budget.ridge * np.eye(len(ACTIVE_ACTUATORS))
    offset = weighted.T @ desired
    try:
        solution = np.linalg.solve(hessian, offset)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(hessian, offset, rcond=None)[0]
    solution = np.clip(solution, lower, upper)
    lipschitz = max(float(np.linalg.eigvalsh(hessian)[-1]), 1e-12)
    for _ in range(512):
        updated = np.clip(solution - (hessian @ solution - offset) / lipschitz, lower, upper)
        if float(np.max(np.abs(updated - solution))) <= 1e-13:
            solution = updated
            break
        solution = updated
    predicted = bias + matrix @ solution
    normalized_residual = float(np.linalg.norm((predicted - target) / scales))
    score = float(1.0 / (1.0 + normalized_residual))
    return {
        "contact_point_manipulability_prescreen_schema_version": 1,
        "available": True,
        "success_evidence": False,
        "purpose": "rank_future_small_motion_manipulability_only",
        "probe_count": len(probe_results),
        "columns": columns,
        "jacobian_6x8": matrix.tolist(),
        "bias_response_6d": bias.tolist(),
        "target_response_6d": target.tolist(),
        "maximum_abs_delta_rad": budget.maximum_delta_rad,
        "solution_delta_rad": {
            name: float(solution[index]) for index, name in enumerate(ACTIVE_ACTUATORS)
        },
        "predicted_response_6d": predicted.tolist(),
        "matrix_rank": int(np.linalg.matrix_rank(weighted)),
        "weighted_residual_norm": normalized_residual,
        "manipulability_score": score,
    }


def run_contact_point_manipulability_prescreen(
    config: Mapping[str, Any],
    trace_or_path: Mapping[str, Any] | str | Path,
    result: Mapping[str, Any],
    *,
    budget: ContactPointManipulabilityBudget = ContactPointManipulabilityBudget(),
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Reacquire one verified grasp, run 17 branches, and return only a score."""

    grasp: GraspPhysicsCheckpoint = prepare_grasp_checkpoint(
        config, trace_or_path, result
    )
    legacy_probe_budget = ManipulationSearchBudget(
        probe_epsilon_rad=budget.probe_epsilon_rad,
        manipulate_s=budget.manipulate_s,
        hold_s=budget.hold_s,
    )
    probes = run_checkpoint_probe_set(grasp, budget=legacy_probe_budget)
    registered = manipulation_delta_bounds(grasp.model, config)
    fitted = fit_contact_point_manipulability(probes, registered, budget=budget)
    fitted["grasp_lock_step"] = int(grasp.grasp_lock_step)
    fitted["checkpoint_used_for_search_only"] = True
    fitted["published_grasp_requires_full_reset_rerun"] = True
    return fitted, probes


__all__ = [
    "ContactPointManipulabilityBudget",
    "fit_contact_point_manipulability",
    "run_contact_point_manipulability_prescreen",
]
