"""Deterministic final selection for schema-v9 grasp/manipulation evidence.

The first complete trajectory and the best final trajectory are deliberately
different concepts.  ``best_first`` is chronological evidence: once the first
full-reset hard pass is observed it must remain addressable forever.  The
remaining catalog members are selected by physical quality, subject to the
versioned five-trajectory diversity contract.

This module has no MuJoCo dependency.  It consumes the persisted config and
summary mappings, which lets the campaign stopping rule and both Viewer
catalogs use exactly the same decision procedure.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence


THUMB_BEND_ACTUATOR = "left_hand_thumb_bend_joint_actuator"
SelectionKind = Literal["grasp_pose", "manipulation"]
_DEFAULT_MINIMUM_DISTINCT_EDGES = 3
_DEFAULT_MINIMUM_THUMB_BANDS = 2


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(value: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    current: Mapping[str, Any] = value
    for name in names:
        current = _mapping(current.get(name))
    return current


def _finite(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _candidate_identifier(value: Mapping[str, Any]) -> tuple[int, int | str]:
    raw = value.get("candidate_id", value.get("candidate_index", ""))
    if isinstance(raw, bool):
        return (1, str(raw).lower())
    try:
        return (0, int(raw))
    except (TypeError, ValueError):
        return (1, str(raw))


def _summary(record: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = record.get("summary", record)
    return _mapping(summary)


def _config(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(record.get("config"))


def _stage_status(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _nested(_summary(record), "stage_status")


def grasp_success(record: Mapping[str, Any]) -> bool:
    return bool(_stage_status(record).get("grasp_success", False))


def full_success(record: Mapping[str, Any]) -> bool:
    summary = _summary(record)
    return bool(
        summary.get("passed", False)
        and _stage_status(record).get("full_success", False)
    )


def _actual_thumb_median(record: Mapping[str, Any]) -> float:
    metrics = _nested(_summary(record), "metrics")
    actual = _nested(metrics, "actual_grasp_pose")
    actual_metrics = _nested(actual, "metrics")
    return _finite(actual_metrics.get("thumb_actual_median_rad"), math.nan)


def _stability_margin(record: Mapping[str, Any]) -> float:
    metrics = _nested(_summary(record), "metrics")
    actual_metrics = _nested(metrics, "actual_grasp_pose", "metrics")
    settings = _nested(_config(record), "grasp_pose")
    error = _finite(
        actual_metrics.get("maximum_nominal_joint_error_rad"), math.inf
    )
    span = _finite(
        actual_metrics.get("maximum_joint_stability_span_rad"), math.inf
    )
    error_limit = _finite(settings.get("max_nominal_joint_error_rad"), 0.04)
    span_limit = _finite(settings.get("max_joint_stability_span_rad"), 0.03)
    if error_limit <= 0.0 or span_limit <= 0.0:
        return -math.inf
    return min(
        (error_limit - error) / error_limit,
        (span_limit - span) / span_limit,
    )


def _pose_preservation_margin(record: Mapping[str, Any]) -> float:
    metrics = _nested(_summary(record), "metrics", "pose_preservation")
    settings = _nested(_config(record), "pose_preservation")
    translation = _finite(metrics.get("max_translation_m"), math.inf)
    orientation = _finite(metrics.get("max_orientation_drift_deg"), math.inf)
    translation_limit = _finite(settings.get("max_translation_m"), 0.0005)
    orientation_limit = _finite(
        settings.get("max_orientation_drift_deg"), 1.0
    )
    if translation_limit <= 0.0 or orientation_limit <= 0.0:
        return -math.inf
    return min(
        (translation_limit - translation) / translation_limit,
        (orientation_limit - orientation) / orientation_limit,
    )


def _closure_p95(record: Mapping[str, Any]) -> float:
    closure = _nested(_summary(record), "metrics", "closure_alignment")
    phase = _mapping(closure.get("close")) or closure
    values = [
        _finite(
            phase.get("max_p95_angle_deg", phase.get("worst_p95_angle_deg")),
            math.nan,
        )
    ]
    per_finger = _mapping(phase.get("per_finger"))
    for finger in ("thumb", "index", "mid", "middle"):
        finger_metrics = _mapping(per_finger.get(finger))
        values.append(
            _finite(
                finger_metrics.get(
                    "angle_p95_deg", finger_metrics.get("p95_angle_deg")
                ),
                math.nan,
            )
        )
    finite = [value for value in values if math.isfinite(value)]
    return max(finite, default=math.inf)


def _closure_margin(record: Mapping[str, Any]) -> float:
    limit = _finite(
        _nested(_config(record), "closure_alignment").get(
            "dynamic_p95_max_angle_deg"
        ),
        30.0,
    )
    value = _closure_p95(record)
    if limit <= 0.0 or not math.isfinite(value):
        return -math.inf
    return (limit - value) / limit


def _ratio_margin(value: float, limit: float) -> float:
    if limit <= 0.0 or not math.isfinite(value):
        return -math.inf
    return (limit - value) / limit


def _vertical_and_smoothness_margins(
    record: Mapping[str, Any],
) -> tuple[float, float]:
    summary_metrics = _nested(_summary(record), "metrics")
    acceptance = _nested(_config(record), "acceptance")
    smooth = _nested(summary_metrics, "motion_smoothness")
    thresholds = _nested(_config(record), "motion_smoothness")

    median = _finite(summary_metrics.get("operation_median_lift_m"), -math.inf)
    minimum = _finite(
        summary_metrics.get("operation_minimum_lift_m"), -math.inf
    )
    median_limit = _finite(acceptance.get("median_lift_m"), 0.010)
    minimum_limit = _finite(acceptance.get("minimum_lift_m"), 0.008)
    lift_margin = min(
        (median - median_limit) / median_limit
        if median_limit > 0.0
        else -math.inf,
        (minimum - minimum_limit) / minimum_limit
        if minimum_limit > 0.0
        else -math.inf,
    )

    lateral = _finite(
        smooth.get("operation_max_lateral_displacement_m"), math.inf
    )
    orientation = _finite(
        smooth.get("operation_max_orientation_drift_deg"), math.inf
    )
    lateral_limit = _finite(
        thresholds.get("max_lateral_displacement_m"), 0.002
    )
    orientation_limit = _finite(
        thresholds.get("max_orientation_drift_deg"), 10.0
    )
    vertical_margin = min(
        lift_margin,
        _ratio_margin(lateral, lateral_limit),
        _ratio_margin(orientation, orientation_limit),
    )

    pairs = (
        (
            "operation_cumulative_height_backtrack_m",
            "max_cumulative_height_backtrack_m",
        ),
        ("operation_downward_speed_duty", "max_downward_speed_duty"),
        (
            "operation_peak_filtered_upward_speed_m_s",
            "max_peak_upward_speed_m_s",
        ),
        (
            "operation_peak_abs_filtered_acceleration_m_s2",
            "max_abs_vertical_acceleration_m_s2",
        ),
        (
            "operation_peak_abs_filtered_jerk_m_s3",
            "max_abs_vertical_jerk_m_s3",
        ),
        (
            "operation_hold_entry_linear_speed_m_s",
            "max_hold_entry_linear_speed_m_s",
        ),
    )
    smoothness_margin = min(
        (
            _ratio_margin(
                _finite(smooth.get(metric), math.inf),
                _finite(thresholds.get(threshold), -1.0),
            )
            for metric, threshold in pairs
        ),
        default=-math.inf,
    )
    return vertical_margin, smoothness_margin


def _force_imbalance(record: Mapping[str, Any]) -> float:
    metrics = _nested(_summary(record), "metrics")
    candidates = (
        _mapping(metrics.get("verify_peak_target_face_force_n")),
        _mapping(metrics.get("peak_distal_contact_force_n")),
    )
    for values in candidates:
        force = [
            _finite(values.get(name), math.nan)
            for name in ("thumb", "index", "mid")
        ]
        if not math.isfinite(force[-1]):
            force[-1] = _finite(values.get("middle"), math.nan)
        if all(math.isfinite(value) and value >= 0.0 for value in force):
            maximum = max(force)
            return (maximum - min(force)) / maximum if maximum > 0.0 else math.inf
    return math.inf


def final_candidate_rank_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-safe evidence in the exact schema-v9 final rank order."""

    thumb = _actual_thumb_median(record)
    vertical, smoothness = _vertical_and_smoothness_margins(record)
    metrics = _nested(_summary(record), "metrics")
    saturation = _finite(metrics.get("actuator_saturation_fraction"), math.inf)

    def optional(value: float) -> float | None:
        return value if math.isfinite(value) else None

    return {
        "full_success": full_success(record),
        "grasp_success": grasp_success(record),
        "actual_thumb_median_rad": optional(thumb),
        "actual_thumb_distance_from_1p50_rad": optional(abs(thumb - 1.50)),
        "grasp_stability_min_normalized_margin": optional(
            _stability_margin(record)
        ),
        "pose_preservation_min_normalized_margin": optional(
            _pose_preservation_margin(record)
        ),
        "closure_alignment_normalized_margin": optional(
            _closure_margin(record)
        ),
        "vertical_motion_min_normalized_margin": optional(vertical),
        "smoothness_min_normalized_margin": optional(smoothness),
        "contact_force_imbalance_fraction": optional(_force_imbalance(record)),
        "actuator_saturation_fraction": optional(saturation),
    }


def final_candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Worker-order-independent final rank required by the v9 experiment."""

    evidence = final_candidate_rank_evidence(record)
    return (
        not bool(evidence["full_success"]),
        _finite(evidence["actual_thumb_distance_from_1p50_rad"], math.inf),
        -_finite(
            evidence["grasp_stability_min_normalized_margin"], -math.inf
        ),
        -_finite(
            evidence["pose_preservation_min_normalized_margin"], -math.inf
        ),
        -_finite(
            evidence["closure_alignment_normalized_margin"], -math.inf
        ),
        -_finite(
            evidence["vertical_motion_min_normalized_margin"], -math.inf
        ),
        -_finite(evidence["smoothness_min_normalized_margin"], -math.inf),
        _finite(evidence["contact_force_imbalance_fraction"], math.inf),
        _finite(evidence["actuator_saturation_fraction"], math.inf),
        _candidate_identifier(record),
    )


def _edge_m(record: Mapping[str, Any]) -> float | None:
    value = _finite(_nested(_config(record), "cube").get("edge_m"), math.nan)
    return round(value, 9) if math.isfinite(value) and value > 0.0 else None


def _thumb_centers(record: Mapping[str, Any]) -> tuple[float, ...]:
    raw = _nested(_config(record), "actual_contact_grasp_pose_campaign").get(
        "thumb_actual_centers_rad", ()
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    values = tuple(_finite(value, math.nan) for value in raw)
    return values if values and all(math.isfinite(value) for value in values) else ()


def _thumb_band(record: Mapping[str, Any]) -> float | None:
    actual = _actual_thumb_median(record)
    centers = _thumb_centers(record)
    if not math.isfinite(actual) or not centers:
        return None
    return min(centers, key=lambda center: (abs(actual - center), center))


def _selection_limits(
    records: Sequence[Mapping[str, Any]], selected_count: int
) -> tuple[int, int]:
    if selected_count != 5:
        return 1, 1
    for record in records:
        selection = _nested(
            _config(record), "actual_contact_grasp_pose_campaign", "selection"
        )
        if selection:
            return (
                int(
                    selection.get(
                        "minimum_distinct_edges",
                        _DEFAULT_MINIMUM_DISTINCT_EDGES,
                    )
                ),
                int(
                    selection.get(
                        "minimum_thumb_bands", _DEFAULT_MINIMUM_THUMB_BANDS
                    )
                ),
            )
    return _DEFAULT_MINIMUM_DISTINCT_EDGES, _DEFAULT_MINIMUM_THUMB_BANDS


def _discovery_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(record.get("discovery_index", 2**63 - 1)),
        _candidate_identifier(record),
    )


def _identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return _candidate_identifier(record)


def _coverage(records: Sequence[Mapping[str, Any]]) -> tuple[set[float], set[float]]:
    edges = {value for record in records if (value := _edge_m(record)) is not None}
    bands = {
        value for record in records if (value := _thumb_band(record)) is not None
    }
    return edges, bands


def _fill_to_count(
    required: Sequence[Mapping[str, Any]],
    ranked: Sequence[Mapping[str, Any]],
    selected_count: int,
) -> tuple[Mapping[str, Any], ...]:
    selected = list(required)
    used = {_identity(value) for value in selected}
    for record in ranked:
        identity = _identity(record)
        if identity in used:
            continue
        selected.append(record)
        used.add(identity)
        if len(selected) >= selected_count:
            break
    return tuple(selected[:selected_count])


def _diverse_five(
    eligible: Sequence[Mapping[str, Any]],
    first: Mapping[str, Any],
    *,
    minimum_edges: int,
    minimum_bands: int,
) -> tuple[Mapping[str, Any], ...]:
    """Choose the lexicographically best ranked diverse set containing first.

    Diversity has only 11x5 possible category pairs.  Enumerating at most four
    category representatives is therefore bounded (C(55, 4) ~= 341k) and
    gives an exact result without making it depend on worker completion order.
    """

    ranked = tuple(sorted(eligible, key=final_candidate_rank))
    if len(ranked) < 5:
        return _fill_to_count((first,), ranked, 5)
    rank_position = {_identity(record): index for index, record in enumerate(ranked)}
    first_category = (_edge_m(first), _thumb_band(first))
    representative: dict[tuple[float | None, float | None], Mapping[str, Any]] = {}
    for record in ranked:
        category = (_edge_m(record), _thumb_band(record))
        if category == first_category:
            continue
        representative.setdefault(category, record)
    categories = tuple(
        sorted(
            representative,
            key=lambda value: final_candidate_rank(representative[value]),
        )
    )
    best_feasible: tuple[tuple[int, ...], tuple[Mapping[str, Any], ...]] | None = None
    best_fallback: tuple[
        tuple[Any, ...], tuple[Mapping[str, Any], ...]
    ] | None = None
    for count in range(0, min(4, len(categories)) + 1):
        for category_group in itertools.combinations(categories, count):
            required = (first, *(representative[value] for value in category_group))
            selected = _fill_to_count(required, ranked, 5)
            if len(selected) != 5:
                continue
            edges, bands = _coverage(selected)
            positions = tuple(
                sorted(rank_position[_identity(record)] for record in selected)
            )
            if len(edges) >= minimum_edges and len(bands) >= minimum_bands:
                score = positions
                if best_feasible is None or score < best_feasible[0]:
                    best_feasible = (score, selected)
            fallback_score = (
                -min(len(edges), minimum_edges),
                -min(len(bands), minimum_bands),
                -len(edges),
                -len(bands),
                positions,
            )
            if best_fallback is None or fallback_score < best_fallback[0]:
                best_fallback = (fallback_score, selected)
    chosen = best_feasible[1] if best_feasible is not None else best_fallback[1]
    return tuple(sorted(chosen, key=final_candidate_rank))


@dataclass(frozen=True)
class ActualContactSelection:
    selected: tuple[dict[str, Any], ...]
    eligible: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


def select_actual_contact_candidates(
    records: Iterable[Mapping[str, Any]],
    *,
    kind: SelectionKind,
    selected_count: int,
) -> ActualContactSelection:
    """Select catalog/campaign evidence with one shared deterministic rule."""

    if kind not in ("grasp_pose", "manipulation"):
        raise ValueError("kind must be grasp_pose or manipulation")
    if not 1 <= int(selected_count) <= 5:
        raise ValueError("selected_count must be between 1 and 5")
    materialized = [copy.deepcopy(dict(value)) for value in records]
    predicate = grasp_success if kind == "grasp_pose" else full_success
    eligible = [
        value
        for value in materialized
        if predicate(value) and not bool(value.get("parameter_override", False))
    ]
    identifiers = [_identity(value) for value in eligible]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("eligible candidates must have unique candidate IDs")
    eligible_by_discovery = tuple(sorted(eligible, key=_discovery_key))
    minimum_edges, minimum_bands = _selection_limits(
        materialized, int(selected_count)
    )
    if not eligible_by_discovery:
        selected: tuple[Mapping[str, Any], ...] = ()
        best_first: Mapping[str, Any] | None = None
    else:
        best_first = eligible_by_discovery[0]
        if int(selected_count) == 1:
            selected = (best_first,)
        elif int(selected_count) == 5:
            selected = _diverse_five(
                eligible_by_discovery,
                best_first,
                minimum_edges=minimum_edges,
                minimum_bands=minimum_bands,
            )
        else:
            selected = _fill_to_count(
                (best_first,),
                tuple(sorted(eligible_by_discovery, key=final_candidate_rank)),
                int(selected_count),
            )
            selected = tuple(sorted(selected, key=final_candidate_rank))
    edges, bands = _coverage(selected)
    target_reached = bool(
        len(selected) == int(selected_count)
        and (
            int(selected_count) != 5
            or (
                len(edges) >= minimum_edges
                and len(bands) >= minimum_bands
            )
        )
    )
    diversity = {
        "required_distinct_edge_count": minimum_edges,
        "required_actual_thumb_band_count": minimum_bands,
        "selected_distinct_edge_count": len(edges),
        "selected_actual_thumb_band_count": len(bands),
        "selected_edges_m": sorted(edges),
        "selected_actual_thumb_bands_rad": sorted(bands),
        "edge_deficit": max(0, minimum_edges - len(edges)),
        "actual_thumb_band_deficit": max(0, minimum_bands - len(bands)),
        "satisfied": bool(
            int(selected_count) != 5
            or (
                len(edges) >= minimum_edges
                and len(bands) >= minimum_bands
            )
        ),
    }
    selected_dicts = tuple(copy.deepcopy(dict(value)) for value in selected)
    metadata = {
        "selection_schema_version": 1,
        "kind": kind,
        "requested_success_count": int(selected_count),
        "eligible_success_count": len(eligible_by_discovery),
        "selected_success_count": len(selected_dicts),
        "target_reached": target_reached,
        "best_first_candidate_id": (
            None if best_first is None else str(best_first.get("candidate_id"))
        ),
        "best_first_discovery_index": (
            None
            if best_first is None
            else int(best_first.get("discovery_index", 2**63 - 1))
        ),
        "selected_candidate_ids_in_final_rank_order": [
            str(value.get("candidate_id")) for value in selected_dicts
        ],
        "rank_order": [
            "full_hard_pass",
            "actual_thumb_distance_from_1p50_rad",
            "grasp_stability_margin",
            "pose_preservation_margin",
            "closure_alignment_margin",
            "vertical_motion_margin",
            "smoothness_margin",
            "contact_force_balance",
            "actuator_saturation",
            "candidate_id",
        ],
        "diversity": diversity,
    }
    return ActualContactSelection(
        selected=selected_dicts,
        eligible=tuple(copy.deepcopy(dict(value)) for value in eligible_by_discovery),
        metadata=metadata,
    )


__all__ = [
    "ActualContactSelection",
    "final_candidate_rank",
    "final_candidate_rank_evidence",
    "full_success",
    "grasp_success",
    "select_actual_contact_candidates",
]
