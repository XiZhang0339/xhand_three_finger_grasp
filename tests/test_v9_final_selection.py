from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    export_actual_contact_grasp_pose_catalog,
    export_actual_contact_manipulation_catalog,
)
from xhand_grasp.actual_contact_selection import (
    final_candidate_rank,
    select_actual_contact_candidates,
)
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import load_config


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift.json"
)


def _record(
    identifier: int,
    *,
    discovery: int | None = None,
    edge_m: float = 0.060,
    thumb: float = 1.50,
    full: bool = True,
    error: float = 0.005,
    span: float = 0.005,
    translation: float = 0.0001,
    orientation: float = 0.2,
    closure: float = 10.0,
    median_lift: float = 0.012,
    minimum_lift: float = 0.010,
    lateral: float = 0.0005,
    manipulation_orientation: float = 2.0,
    jerk: float = 0.5,
    force: tuple[float, float, float] = (1.0, 1.0, 1.0),
    saturation: float = 0.01,
) -> dict:
    config = load_config(TEMPLATE)
    config["cube"]["edge_m"] = edge_m
    metrics = {
        "actual_grasp_pose": {
            "passed": True,
            "metrics": {
                "thumb_actual_median_rad": thumb,
                "maximum_nominal_joint_error_rad": error,
                "maximum_joint_stability_span_rad": span,
            },
        },
        "pose_preservation": {
            "max_translation_m": translation,
            "max_orientation_drift_deg": orientation,
        },
        "closure_alignment": {
            "worst_p95_angle_deg": closure,
            "per_finger": {
                name: {"angle_p95_deg": closure}
                for name in ("thumb", "index", "mid")
            },
        },
        "operation_median_lift_m": median_lift,
        "operation_minimum_lift_m": minimum_lift,
        "motion_smoothness": {
            "operation_max_lateral_displacement_m": lateral,
            "operation_max_orientation_drift_deg": manipulation_orientation,
            "operation_cumulative_height_backtrack_m": 0.00005,
            "operation_downward_speed_duty": 0.005,
            "operation_peak_filtered_upward_speed_m_s": 0.010,
            "operation_peak_abs_filtered_acceleration_m_s2": 0.05,
            "operation_peak_abs_filtered_jerk_m_s3": jerk,
            "operation_hold_entry_linear_speed_m_s": 0.002,
        },
        "verify_peak_target_face_force_n": {
            "thumb": force[0],
            "index": force[1],
            "mid": force[2],
        },
        "actuator_saturation_fraction": saturation,
    }
    return {
        "candidate_id": identifier,
        "discovery_index": identifier if discovery is None else discovery,
        "config": config,
        "summary": {
            "passed": full,
            "stage_status": {
                "grasp_success": True,
                "manipulation_success": full,
                "full_success": full,
            },
            "metrics": metrics,
            "checks": {},
            "failed_checks": [] if full else ["operation_median_lift_reached"],
        },
    }


@pytest.mark.parametrize(
    ("worse", "better"),
    [
        ({"full": False}, {"full": True}),
        ({"thumb": 1.52}, {"thumb": 1.50}),
        ({"error": 0.020}, {"error": 0.005}),
        ({"translation": 0.0004}, {"translation": 0.0001}),
        ({"closure": 25.0}, {"closure": 10.0}),
        (
            {"median_lift": 0.0105, "minimum_lift": 0.0085},
            {"median_lift": 0.012, "minimum_lift": 0.010},
        ),
        ({"jerk": 2.0}, {"jerk": 0.5}),
        ({"force": (2.0, 1.0, 0.5)}, {"force": (1.0, 1.0, 1.0)}),
        ({"saturation": 0.2}, {"saturation": 0.01}),
    ],
)
def test_final_rank_uses_declared_physical_priority(worse, better) -> None:
    left = _record(1, **worse)
    right = _record(2, **better)
    # Make every criterion before the parameter under test identical.
    shared = set(worse) | set(better)
    ordered_fields = (
        "full",
        "thumb",
        "error",
        "translation",
        "closure",
        "median_lift",
        "minimum_lift",
        "jerk",
        "force",
        "saturation",
    )
    first_changed = min(ordered_fields.index(value) for value in shared)
    defaults = {
        "full": True,
        "thumb": 1.50,
        "error": 0.005,
        "translation": 0.0001,
        "closure": 10.0,
        "median_lift": 0.012,
        "minimum_lift": 0.010,
        "jerk": 0.5,
        "force": (1.0, 1.0, 1.0),
        "saturation": 0.01,
    }
    for name in ordered_fields[:first_changed]:
        assert worse.get(name, defaults[name]) == better.get(name, defaults[name])
    assert final_candidate_rank(right) < final_candidate_rank(left)


def test_best_first_is_chronological_while_final_members_are_ranked() -> None:
    records = [
        _record(90, discovery=0, edge_m=0.060, thumb=1.40, closure=29.0),
        _record(1, discovery=5, edge_m=0.060, thumb=1.50),
        _record(2, discovery=4, edge_m=0.061, thumb=1.50),
        _record(3, discovery=3, edge_m=0.062, thumb=1.55),
        _record(4, discovery=2, edge_m=0.063, thumb=1.45),
        _record(5, discovery=1, edge_m=0.064, thumb=1.50),
    ]
    forward = select_actual_contact_candidates(
        records, kind="manipulation", selected_count=5
    )
    reverse = select_actual_contact_candidates(
        reversed(records), kind="manipulation", selected_count=5
    )
    assert forward.metadata == reverse.metadata
    assert forward.metadata["best_first_candidate_id"] == "90"
    assert forward.metadata["target_reached"]
    assert forward.metadata["diversity"]["selected_distinct_edge_count"] >= 3
    assert forward.metadata["diversity"]["selected_actual_thumb_band_count"] >= 2
    assert "90" in forward.metadata["selected_candidate_ids_in_final_rank_order"]
    assert [final_candidate_rank(value) for value in forward.selected] == sorted(
        final_candidate_rank(value) for value in forward.selected
    )


def test_raw_five_successes_do_not_reach_target_without_diversity() -> None:
    records = [_record(index, edge_m=0.060, thumb=1.50) for index in range(5)]
    selection = select_actual_contact_candidates(
        records, kind="manipulation", selected_count=5
    )
    assert len(selection.selected) == 5
    assert not selection.metadata["target_reached"]
    assert selection.metadata["diversity"]["edge_deficit"] == 2
    assert selection.metadata["diversity"]["actual_thumb_band_deficit"] == 1

    first = select_actual_contact_candidates(
        records, kind="manipulation", selected_count=1
    )
    assert first.metadata["target_reached"]
    assert first.metadata["best_first_candidate_id"] == "0"


def _persist_candidate(root: Path, record: dict) -> dict:
    identifier = int(record["candidate_id"])
    directory = root / f"candidate_{identifier}"
    directory.mkdir(parents=True)
    config_path = directory / "resolved_config.json"
    result_path = directory / "result.json"
    trace_path = directory / "trace.npz"
    write_json(config_path, record["config"])
    np.savez_compressed(trace_path, time=np.asarray([0.001]))
    write_json(
        result_path,
        {
            "candidate_id": identifier,
            "discovery_index": int(record["discovery_index"]),
            "summary": record["summary"],
            "artifacts": {
                "resolved_config": config_path.name,
                "trace": trace_path.name,
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "trace": file_sha256(trace_path),
                },
            },
        },
    )
    return {
        "candidate_id": identifier,
        "discovery_index": int(record["discovery_index"]),
        "config_path": config_path,
        "result_path": result_path,
        "trace_path": trace_path,
    }


def test_grasp_and_manipulation_catalogs_share_selection_and_diversity_metadata(
    tmp_path: Path,
) -> None:
    records = [
        _record(50, discovery=0, edge_m=0.060, thumb=1.40, closure=28.0),
        _record(1, discovery=5, edge_m=0.060, thumb=1.50),
        _record(2, discovery=4, edge_m=0.061, thumb=1.50),
        _record(3, discovery=3, edge_m=0.062, thumb=1.55),
        _record(4, discovery=2, edge_m=0.063, thumb=1.45),
        _record(5, discovery=1, edge_m=0.064, thumb=1.50),
    ]
    sources = [_persist_candidate(tmp_path / "sources", value) for value in records]
    grasp = export_actual_contact_grasp_pose_catalog(
        sources, tmp_path / "grasp", selected_count=5
    )
    manipulation = export_actual_contact_manipulation_catalog(
        sources, tmp_path / "manipulation", selected_count=5
    )
    assert grasp["selection"]["selected_candidate_ids_in_final_rank_order"] == (
        manipulation["selection"]["selected_candidate_ids_in_final_rank_order"]
    )
    assert grasp["selection"]["rank_order"] == manipulation["selection"][
        "rank_order"
    ]
    assert grasp["diversity"] == manipulation["diversity"]
    assert grasp["target_reached"] and manipulation["target_reached"]
    for catalog in (grasp, manipulation):
        best = next(
            value
            for value in catalog["trajectories"]
            if value["trajectory_id"] == catalog["aliases"]["best_first"]
        )
        assert best["candidate_id"] == "50"


def test_candidate_id_is_the_final_worker_independent_tie_break() -> None:
    high = _record(9, discovery=0)
    low = copy.deepcopy(high)
    low["candidate_id"] = 2
    low["discovery_index"] = 99
    assert final_candidate_rank(low) < final_candidate_rank(high)
