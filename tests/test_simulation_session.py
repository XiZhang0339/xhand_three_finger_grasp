from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.artifacts import json_text
from xhand_grasp.config import load_config
from xhand_grasp.contact_environment import ContactEnvironmentSpec
from xhand_grasp.simulation import SimulationSession, run_simulation


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("config_name", "shorten_legacy"),
    [
        ("left_three_finger_cube.json", True),
        ("left_opposed_face_palm_down.json", True),
        ("left_opposed_face_palm_down_larger_cube_grasp_then_lift.json", False),
    ],
)
def test_incremental_session_and_batch_wrapper_are_field_exact(
    tmp_path, config_name, shorten_legacy
):
    config = load_config(ROOT / "grasp_configs" / config_name)
    if shorten_legacy:
        config["timing"] = {
            "settle_s": 0.002,
            "pregrasp_s": 0.002,
            "lift_s": 0.002,
            "hold_s": 0.002,
        }
    batch_path = tmp_path / f"batch-{config_name}.npz"
    batch_summary = run_simulation(copy.deepcopy(config), trace_path=batch_path)

    session = SimulationSession(copy.deepcopy(config))
    while not session.complete:
        session.advance_one()
    incremental_summary = session.finalize()
    with np.load(batch_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(session.traces)
        for name in archive.files:
            np.testing.assert_array_equal(session.traces[name], archive[name])
    assert json_text(incremental_summary) == json_text(batch_summary)
    session.close()


def test_session_reset_restores_controller_physics_and_trace_prefix():
    config = load_config(
        ROOT
        / "grasp_configs"
        / "left_opposed_face_palm_down_larger_cube_grasp_then_lift.json"
    )
    session = SimulationSession(config)
    for _ in range(25):
        session.advance_one()
    first_qpos = session.data.qpos.copy()
    first_prefix = {
        name: values[:25].copy()
        for name, values in session.traces.items()
        if values.ndim > 0 and len(values) == session.total_steps
    }

    session.reset()
    for _ in range(25):
        session.advance_one()

    np.testing.assert_array_equal(session.data.qpos, first_qpos)
    for name, expected in first_prefix.items():
        np.testing.assert_array_equal(session.traces[name][:25], expected)
    session.close()


def test_session_cannot_finalize_an_incomplete_trace():
    config = load_config(ROOT / "grasp_configs" / "left_three_finger_cube.json")
    session = SimulationSession(config)
    session.advance_one()

    with pytest.raises(RuntimeError, match="cannot finalize an incomplete"):
        session.finalize()
    session.close()


def test_v4_incremental_step_records_alignment_evidence_and_gate():
    config = load_config(
        ROOT
        / "grasp_configs"
        / "left_opposed_face_palm_tilted_down_aligned_contacts_grasp_then_lift.json"
    )
    session = SimulationSession(config)

    sample = session.advance_one()

    assert sample.index == 0
    assert session.traces["distal_face_position_moment_n_m"].shape == (
        session.total_steps,
        3,
        8,
        3,
    )
    assert session.traces["target_face_contact_centroid_world_m"].shape == (
        session.total_steps,
        3,
        3,
    )
    assert session.traces["grasp_gate_order"].tolist()[-1] == (
        "contact_height_aligned"
    )
    assert np.isfinite(session.traces["finger_down_tilt_deg"][0])
    assert np.isfinite(session.traces["three_contact_height_spread_m"][0])
    assert session.traces["three_contact_height_aligned"].dtype == np.bool_
    session.close()


def test_opt_in_contact_environment_is_audited_without_changing_legacy_trace():
    config = load_config(ROOT / "grasp_configs" / "left_three_finger_cube.json")
    config["timing"] = {
        "settle_s": 0.002,
        "pregrasp_s": 0.002,
        "lift_s": 0.002,
        "hold_s": 0.002,
    }
    legacy = SimulationSession(copy.deepcopy(config))
    environment = SimulationSession(
        copy.deepcopy(config),
        contact_environment=ContactEnvironmentSpec(iterations=200),
    )
    try:
        assert "contact_environment_id" not in legacy.traces
        assert environment.model.opt.iterations == 200
        assert environment.traces["contact_environment_id"].item() == (
            environment.contact_environment.environment_id
        )
        while not environment.complete:
            environment.advance_one()
        summary = environment.finalize()
        evidence = summary["contact_environment"]
        assert evidence["model_change_audit"]["passed"]
        assert evidence["compiled"]["solver"]["iterations"] == 200
        assert evidence["runtime"]["verified_contact_step_count"] > 0
        assert np.max(
            environment.traces["contact_environment_cube_contact_count"]
        ) > 0
    finally:
        legacy.close()
        environment.close()
