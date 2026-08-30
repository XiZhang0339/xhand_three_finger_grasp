from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import load_config
from xhand_grasp.simulation import SimulationSession


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_tilted_down_far_hand_fingertip_grasp_then_lift.json"
)


@pytest.mark.slow
def test_v5_complete_session_reports_distance_thumb_and_soft_pad_metrics():
    config = load_config(CONFIG)
    session = SimulationSession(config)
    try:
        while not session.complete:
            session.advance_one()
        summary = session.finalize()
        traces = session.traces
    finally:
        session.close()

    assert summary["checks"]["initial_root_cube_distance_within_range"]
    assert "palm_press_depth_within_range" not in summary["checks"]
    distance = summary["metrics"]["root_cube_center_distance_m"]
    assert 0.138 <= distance["initial"] <= 0.160
    assert summary["metrics"]["legacy_palm_press_depth_m"][
        "acceptance_role"
    ] == "diagnostic_only"

    frame_count = len(traces["time"])
    for name in (
        "root_cube_center_distance_m",
        "thumb_bend_command_rad",
        "thumb_bend_qpos_rad",
    ):
        assert traces[name].shape == (frame_count,)
        assert np.isfinite(traces[name]).all()
    for name in (
        "distal_pad_force_n",
        "distal_nonpad_force_n",
        "distal_pad_force_fraction",
        "distal_active_taxel_count",
    ):
        assert traces[name].shape == (frame_count, 3)
        assert np.isfinite(traces[name]).all()

    preference = summary["metrics"]["fingertip_contact"]
    assert preference["max_taxel_assignment_distance_m"] == pytest.approx(0.006)
    assert preference["trace_fraction_matches_raw_forces"]
    assert preference["trace_pose_and_thumb_bend_match_raw_state"]
