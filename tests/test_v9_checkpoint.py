from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from xhand_grasp.checkpoint import (
    INTEGRATION_STATE_SPEC,
    PhysicsCheckpoint,
    capture_physics_checkpoint,
    restore_physics_checkpoint,
)
from xhand_grasp.config import load_config
from xhand_grasp.scene import build_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_high_thumb_normal_aligned_"
    "smooth_vertical_lift.json"
)


def _step_with_schedule(model: mujoco.MjModel, data: mujoco.MjData, start: int, count: int) -> None:
    for step in range(start, start + count):
        phase = 0.15 + 0.05 * np.sin(0.017 * step + np.arange(model.nu))
        data.ctrl[:] = np.clip(
            phase,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_step(model, data)


def test_integration_checkpoint_restores_deterministic_continuation():
    config = load_config(CONFIG)
    model, _ = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _step_with_schedule(model, data, 0, 120)
    checkpoint = capture_physics_checkpoint(model, data, step_index=119)

    _step_with_schedule(model, data, 120, 80)
    expected = np.empty(checkpoint.state_size, dtype=np.float64)
    mujoco.mj_getState(model, data, expected, INTEGRATION_STATE_SPEC)

    restored_model, _ = build_model(copy.deepcopy(config))
    restored_data = mujoco.MjData(restored_model)
    restore_physics_checkpoint(restored_model, restored_data, checkpoint)
    _step_with_schedule(restored_model, restored_data, 120, 80)
    actual = np.empty(checkpoint.state_size, dtype=np.float64)
    mujoco.mj_getState(
        restored_model, restored_data, actual, INTEGRATION_STATE_SPEC
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(restored_data.xpos, data.xpos, rtol=0.0, atol=1e-13)
    assert restored_data.ncon == data.ncon


def test_checkpoint_is_defensive_and_rejects_incompatible_model():
    config = load_config(CONFIG)
    model, _ = build_model(config)
    data = mujoco.MjData(model)
    checkpoint = capture_physics_checkpoint(model, data, step_index=0)
    source = checkpoint.state.copy()
    source[:] = 1.0
    assert not np.array_equal(checkpoint.state, source)

    invalid = PhysicsCheckpoint(
        state=checkpoint.state,
        nq=checkpoint.nq + 1,
        nv=checkpoint.nv,
        na=checkpoint.na,
        nu=checkpoint.nu,
        state_size=checkpoint.state_size,
        step_index=checkpoint.step_index,
    )
    with pytest.raises(ValueError, match="dimensions differ"):
        restore_physics_checkpoint(model, data, invalid)
