from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import ACTIVE_ACTUATORS, load_config, validate_config
from xhand_grasp.scene import rpy_degrees_to_rotation_matrix
from xhand_grasp.tuning import actual_contact_grasp_pose as static_search
from xhand_grasp.tuning import actual_contact_size_continuation as continuation


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_larger_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)


@pytest.fixture
def template() -> dict:
    return load_config(CONFIG_PATH)


def _source(template: dict, edge_m: float = 0.067, thumb: float = 1.50) -> dict:
    config = copy.deepcopy(template)
    config["cube"]["edge_m"] = edge_m
    actual = [
        float(config["grasp_pose"]["nominal_joint_qpos_rad"][name])
        for name in ACTIVE_ACTUATORS
    ]
    actual[ACTIVE_ACTUATORS.index(continuation.THUMB_BEND_ACTUATOR)] = thumb
    return {
        "source_index": 7,
        "pose_id": "measured-source-7",
        "source_kind": "authenticated_grasp_success",
        "eligible_as_success_evidence": True,
        "actual_joint_qpos_rad": actual,
        "config": config,
    }


def test_registered_policy_and_one_mm_schedule_are_exact(template):
    policy = continuation.SizeContinuationPolicy.from_config(template)

    assert policy.step_m == pytest.approx(0.001)
    assert policy.bridge_edges_m == pytest.approx((0.068, 0.069, 0.070, 0.071))
    assert policy.published_edges_m == pytest.approx(
        tuple(value / 1000.0 for value in range(72, 91))
    )
    schedule = continuation.continuation_edge_schedule(policy, 0.067)
    assert len(schedule) == 23
    assert [value[0] for value in schedule[:5]] == pytest.approx(
        (0.068, 0.069, 0.070, 0.071, 0.072)
    )
    assert [value[1] for value in schedule[:5]] == [False, False, False, False, True]
    assert schedule[-1][0] == pytest.approx(0.090)
    assert schedule[-1][1] is True


def test_policy_rejects_bridge_publication_and_non_unit_spacing(template):
    changed = copy.deepcopy(template)
    changed["size_continuation"]["publish_bridge_results"] = True
    with pytest.raises(ValueError, match="must not be publishable"):
        continuation.SizeContinuationPolicy.from_config(changed)

    changed = copy.deepcopy(template)
    changed["size_continuation"]["bridge_edges_m"][1] = 0.0695
    with pytest.raises(ValueError, match="declared step"):
        continuation.SizeContinuationPolicy.from_config(changed)


def test_schema_v10_preflight_rejects_a_changed_continuation_policy(template):
    changed = copy.deepcopy(template)
    changed["size_continuation"]["publish_bridge_results"] = True

    with pytest.raises(ValueError, match="must not be publishable"):
        validate_config(changed)


def test_prepared_step_fixes_cube_world_pose_material_and_thumb(template):
    source = _source(template, thumb=1.70)
    source_config = source["config"]
    source_rotation = rpy_degrees_to_rotation_matrix(
        source_config["hand_pose"]["rpy_deg"]
    )
    source_cube = static_search._cube_world_position(source_config)
    source_root = np.asarray(source_config["hand_pose"]["translation_m"])
    source_local = source_rotation.T @ (source_cube - source_root)

    prepared, diagnostics = continuation.prepare_continuation_config(
        template, source, 0.068
    )

    assert prepared["cube"]["edge_m"] == pytest.approx(0.068)
    assert prepared["cube"]["mass_kg"] == pytest.approx(0.160)
    assert prepared["cube"]["friction"] == pytest.approx(0.8)
    assert prepared["cube"]["center_xy_m"] == pytest.approx((0.071, -0.027))
    assert prepared["cube"]["rpy_deg"] == pytest.approx(
        (0.0, 0.0, 27.609990189403167)
    )
    target_cube = static_search._cube_world_position(prepared)
    assert target_cube[2] == pytest.approx(
        template["scene"]["support_top_z_m"] + 0.068 / 2.0
    )
    target_rotation = rpy_degrees_to_rotation_matrix(
        prepared["hand_pose"]["rpy_deg"]
    )
    target_root = np.asarray(prepared["hand_pose"]["translation_m"])
    assert target_rotation.T @ (target_cube - target_root) == pytest.approx(
        source_local
    )
    assert prepared["grasp_pose"]["nominal_joint_qpos_rad"][
        continuation.THUMB_BEND_ACTUATOR
    ] == pytest.approx(1.60)
    assert diagnostics["thumb_actual_clipped"] is True
    assert diagnostics["cube_pose_sampled"] is False


class _FakeStaticResult:
    static_geometry_pass = False

    def as_dict(self) -> dict:
        return {
            "static_geometry_pass": False,
            "missing_target_witness_count": 3,
            "off_target_distal_penetrating_count": 0,
            "minimum_active_nondistal_gap_m": 0.001,
            "precontact_minimum_hand_gap_m": 0.001,
            "contact_height_spread_m": 0.020,
            "target_witness": {"thumb": None, "index": None, "mid": None},
            "retreat_evidence": {"thumb": None, "index": None, "mid": None},
        }


def test_continuation_never_publishes_bridge_or_inherits_success(
    template, monkeypatch
):
    def fake_step(config, *, maximum_iterations):
        assert maximum_iterations == 4
        return (
            copy.deepcopy(config),
            _FakeStaticResult(),
            {
                "method": continuation.CONTINUATION_METHOD,
                "initial_measurement_m": [0.0] * 5,
                "final_measurement_m": [0.0] * 5,
                "iterations": [],
            },
            None,
        )

    monkeypatch.setattr(continuation, "_dls_step", fake_step)
    monkeypatch.setattr(static_search, "_static_rank", lambda record: (record["candidate_id"],))

    execution = continuation.continue_actual_qpos_sources(
        template, (_source(template),)
    )

    assert len(execution.bridge_records) == 4
    assert len(execution.published_records) == 19
    assert {round(value["edge_m"] * 1000) for value in execution.bridge_records} == {
        68,
        69,
        70,
        71,
    }
    assert {round(value["edge_m"] * 1000) for value in execution.published_records} == set(
        range(72, 91)
    )
    assert all(value["cell_index"] >= 0 for value in execution.published_records)
    assert all(
        value["source_success_evidence_inherited"] is False
        for value in (*execution.bridge_records, *execution.published_records)
    )
    assert execution.report["publish_bridge_results"] is False
    assert execution.report["source_success_evidence_inherited"] is False


def test_dls_stop_reason_terminates_chain_deterministically(template, monkeypatch):
    def failed_step(config, *, maximum_iterations):
        return (
            copy.deepcopy(config),
            _FakeStaticResult(),
            {"iterations": [], "initial_measurement_m": None},
            "missing_distal_witness",
        )

    monkeypatch.setattr(continuation, "_dls_step", failed_step)
    monkeypatch.setattr(static_search, "_static_rank", lambda record: (record["candidate_id"],))

    first = continuation.continue_actual_qpos_sources(template, (_source(template),))
    second = continuation.continue_actual_qpos_sources(template, (_source(template),))

    assert not first.published_records
    assert len(first.bridge_records) == 1
    assert first.report == second.report
    assert first.report["chains"][0]["stop_reason"] == "missing_distal_witness"
    assert first.report["chains"][0]["step_count"] == 1
