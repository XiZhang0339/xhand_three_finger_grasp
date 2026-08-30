from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_FINGERS
from xhand_grasp.contact_point_targeting import contact_point_plan_from_config
from xhand_grasp.contact_slip import contact_tangent_slip_from_grasp
from xhand_grasp.contacts import FACE_ORDER, Face
from xhand_grasp.controller import grasp_gate_order
from xhand_grasp.evaluation import _v12_contact_point_metrics
from xhand_grasp.scene import build_model
from xhand_grasp.simulation import (
    _allocate_traces,
    _populate_derived_v13_contact_slip_trace_fields,
)
from xhand_grasp.tuning import scaled_contact_downsize_ranking as ranking
from xhand_grasp.tuning.scaled_contact_downsize_campaign import (
    _default_manipulation_runner,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_larger_relative_wrist_pose_"
    "actual_contact_smooth_vertical_lift.json"
)


def _point_config(schema_version: int) -> dict:
    return {
        "schema_version": schema_version,
        "cube": {"edge_m": 0.08},
        "contact_topology": {
            "target_faces": {"thumb": "-X", "index": "+X", "mid": "+X"},
            "target_force_fraction": 0.95,
        },
        "contact_point_plan": {
            "schema_version": 1,
            "point_plan_id": "d" * 64,
            "cube_edge_m": 0.08,
            "coordinate_frame": "cube_local",
            "points": {
                "thumb": {"face": "-X", "yz_m": [0.0, 0.01]},
                "index": {"face": "+X", "yz_m": [-0.01, 0.01]},
                "mid": {"face": "+X", "yz_m": [0.01, 0.01]},
            },
            "target_radius_m": 0.002,
            "frozen": True,
        },
        "control_protocol": {
            "stable_window_s": 0.002,
            "grasp_gate": {
                "min_target_face_force_n": 0.05,
                "min_target_force_fraction": 0.95,
                "require_touch": True,
            },
        },
        "acceptance": {
            "touch_force_min_n": 1e-8,
            "contact_force_min_n": 0.05,
        },
    }


def _synthetic_point_trace(model, config: dict) -> dict[str, np.ndarray]:
    total_steps = 6
    traces = _allocate_traces(model, total_steps, schema_version=13)
    plan = contact_point_plan_from_config(config)
    traces["time"][:] = np.arange(1, total_steps + 1) * model.opt.timestep
    traces["cube_pos"][:] = 0.0
    traces["cube_quat"][:] = [1.0, 0.0, 0.0, 0.0]
    traces["control_state"][:] = [
        "VERIFY",
        "VERIFY",
        "MANIPULATE",
        "MANIPULATE",
        "HOLD",
        "HOLD",
    ]
    traces["grasp_acquisition_step"] = np.asarray(1, dtype=np.int64)
    traces["contact_point_plan_id"] = np.asarray(plan.point_plan_id)
    traces["target_contact_points_cube_local_m"][:] = (
        plan.target_points_cube_local_m
    )
    traces["target_contact_point_radius_m"] = np.asarray(plan.target_radius_m)
    traces["tactile_max"][:] = 1.0
    traces["grasp_gate_order"] = np.asarray(grasp_gate_order(13))
    traces["grasp_gate"] = np.ones(
        (total_steps, len(grasp_gate_order(13))), dtype=bool
    )

    offsets_y = np.asarray([0.0, 0.0004, 0.0030, 0.0040, 0.0050, 0.0060])
    forces = np.asarray([1.0, 3.0, 2.0, 2.0, 2.0, 2.0])
    target_faces = (Face.X_NEG, Face.X_POS, Face.X_POS)
    for step in range(total_steps):
        actual = plan.target_points_cube_local_m.copy()
        actual[:, 1] += offsets_y[step]
        for finger, face in enumerate(target_faces):
            face_index = FACE_ORDER.index(face)
            force = forces[step]
            traces["distal_face_force_n"][step, finger, face_index] = force
            traces["distal_face_position_moment_n_m"][
                step, finger, face_index
            ] = force * actual[finger]
        traces["target_face_contact_centroid_world_m"][step] = actual
        traces["target_face_contact_centroid_valid"][step] = True
        traces["target_face_contact_centroid_cube_local_m"][step] = actual
        traces["target_contact_point_tangent_error_m"][step] = abs(
            offsets_y[step]
        )
        within = abs(offsets_y[step]) <= plan.target_radius_m + 1e-12
        traces["target_contact_point_within_radius"][step] = within
        gate_column = list(grasp_gate_order(13)).index(
            "contact_points_within_target_regions"
        )
        traces["grasp_gate"][step, gate_column] = within

    _populate_derived_v13_contact_slip_trace_fields(model, config, traces)
    return traces


def test_force_weighted_baseline_and_face_tangent_projection() -> None:
    centroid = np.zeros((4, 3, 3), dtype=np.float64)
    centroid[0, :, :] = [0.0, 0.0, 0.0]
    centroid[1, :, :] = [0.0, 0.004, 0.0]
    centroid[2, :, :] = [0.005, 0.010, 0.012]
    centroid[3, :, :] = [0.009, 0.012, 0.016]
    force = np.ones((4, 3), dtype=np.float64)
    force[1] = 3.0
    result = contact_tangent_slip_from_grasp(
        centroid,
        np.ones((4, 3), dtype=bool),
        np.ones((4, 3), dtype=bool),
        force,
        ("-X", "+Y", "+Z"),
        acquisition_start_step=0,
        acquisition_end_step=1,
    )
    np.testing.assert_allclose(result.baseline_centroid_cube_local_m[:, 1], 0.003)
    # -X ignores normal X: sqrt((.010-.003)^2 + .012^2).
    assert result.tangent_slip_from_grasp_m[2, 0] == pytest.approx(
        np.hypot(0.007, 0.012)
    )
    # +Y ignores normal Y: sqrt(.005^2 + .012^2).
    assert result.tangent_slip_from_grasp_m[2, 1] == pytest.approx(0.013)
    # +Z ignores normal Z: sqrt(.005^2 + (.010-.003)^2).
    assert result.tangent_slip_from_grasp_m[2, 2] == pytest.approx(
        np.hypot(0.005, 0.007)
    )


def test_schema13_npz_cache_recomputes_slip_and_operation_statistics() -> None:
    with MODEL_CONFIG.open(encoding="utf-8") as handle:
        model, _ = build_model(json.load(handle))
    config = _point_config(13)
    traces = _synthetic_point_trace(model, config)
    metrics, checks = _v12_contact_point_metrics(model, config, traces)
    assert checks["v13_contact_slip_trace_matches_raw_contacts"] is True
    targeting = metrics["contact_point_targeting"]
    assert targeting["manipulate"]["sample_count"] == 2
    assert targeting["hold"]["sample_count"] == 2
    assert targeting["operation"]["sample_count"] == 4
    slip = targeting["contact_slip_from_grasp"]
    assert slip["soft_ranking_only"] is True
    assert slip["baseline"]["thumb"]["centroid_cube_local_m"][1] == pytest.approx(
        0.0003
    )
    assert slip["operation"]["per_finger"]["thumb"][
        "tangent_slip_p50_m"
    ] == pytest.approx(np.median([0.0027, 0.0037, 0.0047, 0.0057]))
    # Rolling outside the frozen 2 mm region is diagnostic during operation;
    # no operation point-region or slip threshold is added to checks.
    assert targeting["operation"]["all_points_within_radius_duty"] == 0.0
    assert not any("operation" in name and "slip" in name for name in checks)

    corrupted = copy.deepcopy(traces)
    corrupted["target_contact_tangent_slip_from_grasp_m"] = traces[
        "target_contact_tangent_slip_from_grasp_m"
    ].copy()
    corrupted["target_contact_tangent_slip_from_grasp_m"][3, 0] += 1e-4
    _, corrupted_checks = _v12_contact_point_metrics(model, config, corrupted)
    assert corrupted_checks["v13_contact_slip_trace_matches_raw_contacts"] is False


def test_schema12_metrics_and_trace_schema_remain_unchanged() -> None:
    with MODEL_CONFIG.open(encoding="utf-8") as handle:
        model, _ = build_model(json.load(handle))
    traces = _allocate_traces(model, 2, schema_version=12)
    assert "target_contact_tangent_slip_from_grasp_m" not in traces
    assert "grasp_contact_centroid_baseline_cube_local_m" not in traces


def _rank_record(candidate_id: int, p95: float, *, schema: int = 13) -> dict:
    per_finger = {
        finger: {
            "valid_duty": 1.0,
            "tangent_slip_p95_m": p95,
            "tangent_slip_max_m": p95 * 1.2,
        }
        for finger in ACTIVE_FINGERS
    }
    baseline = {finger: {"valid": True} for finger in ACTIVE_FINGERS}
    return {
        "candidate_id": candidate_id,
        "config": {"schema_version": schema},
        "summary": {
            "metrics": {
                "contact_point_targeting": {
                    "contact_slip_from_grasp": {
                        "baseline": baseline,
                        "operation": {"per_finger": per_finger},
                    }
                }
            }
        },
    }


def test_v13_soft_slip_sort_is_deterministic_and_never_overrides_hard_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hard_rank(record, *, config=None):
        del config
        return (
            not bool(record.get("full_success", True)),
            False,
            0.0,
            0.0,
            0.0,
            0,
            int(record["candidate_id"]),
        )

    monkeypatch.setattr(ranking, "manipulation_candidate_rank", hard_rank)
    low = _rank_record(20, 0.001)
    high = _rank_record(10, 0.004)
    failed_but_low = _rank_record(1, 0.0001)
    failed_but_low["full_success"] = False
    expected = [20, 10, 1]
    for values in ((high, failed_but_low, low), (failed_but_low, low, high)):
        ranked = ranking.rank_v13_manipulation_candidates(values)
        assert [value["candidate_id"] for value in ranked] == expected
        assert all("v13_contact_slip_rank_evidence" in value for value in ranked)


def test_non_v13_sort_delegates_and_v13_runner_uses_compacting_slip_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = ({"candidate_id": 99},)
    monkeypatch.setattr(
        ranking,
        "rank_manipulation_candidates",
        lambda records, *, config=None: sentinel,
    )
    assert (
        ranking.rank_v13_manipulation_candidates([_rank_record(1, 0.1, schema=12)])
        is sentinel
    )
    source = inspect.getsource(_default_manipulation_runner)
    assert "rank_v13_manipulation_candidates" in source
    assert "compact_v13_manipulation_candidate_artifacts" in source
    assert "authenticate_v13_manipulation_compaction_report" in source
    assert "retain_failure_trace_count=1" in source
    assert "target_reached(all_records)" in source
