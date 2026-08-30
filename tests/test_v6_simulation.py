from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import load_config
from xhand_grasp.evaluation import evaluate_trace
from xhand_grasp.simulation import SimulationSession


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_pose_preserving_grasp.json"
)


def test_v6_session_uses_reset_pre_step_pose_as_immutable_baseline():
    config = load_config(CONFIG)
    session = SimulationSession(config)
    try:
        reset_position = np.asarray(
            session.data.xpos[session.info.cube_body_id], dtype=np.float64
        ).copy()
        reset_quaternion = np.asarray(
            session.data.xquat[session.info.cube_body_id], dtype=np.float64
        ).copy()
        assert np.array_equal(session.traces["initial_cube_pos_m"], reset_position)
        assert np.array_equal(
            session.traces["initial_cube_quat"], reset_quaternion
        )

        session.advance_one()
        first_position = session.traces["cube_pos"][0]
        expected_translation = float(np.linalg.norm(first_position - reset_position))
        assert expected_translation > 0.0
        assert session.traces["cube_translation_from_initial_m"][
            0
        ] == pytest.approx(expected_translation, abs=1e-15)
        # Using the first post-step trace sample as the reference would hide
        # exactly this gravity-driven first-step displacement.
        assert not np.array_equal(first_position, reset_position)
    finally:
        session.close()


@pytest.mark.slow
def test_v6_npz_round_trip_recomputes_integrity_from_raw_state(tmp_path):
    config = load_config(CONFIG)
    trace_path = tmp_path / "pose_preserving_trace.npz"
    session = SimulationSession(config)
    try:
        while not session.complete:
            session.advance_one()
        original = session.finalize(trace_path=trace_path)

        with np.load(trace_path, allow_pickle=False) as archive:
            loaded = {name: archive[name].copy() for name in archive.files}
        recomputed = evaluate_trace(
            session.model,
            session.info,
            config,
            session.phase_steps,
            loaded,
        )

        integrity_checks = (
            "v6_pose_preservation_trace_matches_raw_state",
            "v6_close_profile_trace_matches_config",
            "v6_first_distal_contact_steps_match_raw_trace",
        )
        for name in integrity_checks:
            assert original["checks"][name]
            assert recomputed["checks"][name]
        assert recomputed["metrics"]["pose_preservation"] == original[
            "metrics"
        ]["pose_preservation"]
        assert np.array_equal(
            loaded["initial_cube_pos_m"],
            session.traces["initial_cube_pos_m"],
        )
        assert np.array_equal(
            loaded["first_distal_contact_step"],
            session.traces["first_distal_contact_step"],
        )

        # Persisted helper flags are evidence, not authority: editing one must
        # be detected when the archive is evaluated again.
        loaded["pregrasp_pose_preserved_latched"][0] = False
        tampered = evaluate_trace(
            session.model,
            session.info,
            config,
            session.phase_steps,
            loaded,
        )
        assert not tampered["checks"][
            "v6_pose_preservation_trace_matches_raw_state"
        ]
    finally:
        session.close()
