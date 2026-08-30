"""Deterministic MuJoCo state checkpoints used by grasp-pose search.

Search may branch many manipulation probes from one independently validated
free-body grasp.  The checkpoint therefore stores MuJoCo's complete
``mjSTATE_INTEGRATION`` vector, not a hand-written qpos/qvel subset.  Derived
kinematics, contacts and sensors are reconstructed with :func:`mj_forward`
after restoration.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


INTEGRATION_STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


@dataclass(frozen=True, slots=True)
class PhysicsCheckpoint:
    """A model-bound, pickle-free MuJoCo integration state."""

    state: np.ndarray
    nq: int
    nv: int
    na: int
    nu: int
    state_size: int
    step_index: int

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float64)
        if state.shape != (int(self.state_size),) or not np.isfinite(state).all():
            raise ValueError("checkpoint state must be a finite state_size vector")
        for name in ("nq", "nv", "na", "nu", "state_size", "step_index"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"checkpoint {name} must be non-negative")
        object.__setattr__(self, "state", state.copy())


def capture_physics_checkpoint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    step_index: int,
) -> PhysicsCheckpoint:
    """Copy every state component needed for deterministic ``mj_step``."""

    state_size = int(mujoco.mj_stateSize(model, INTEGRATION_STATE_SPEC))
    state = np.empty(state_size, dtype=np.float64)
    mujoco.mj_getState(model, data, state, INTEGRATION_STATE_SPEC)
    return PhysicsCheckpoint(
        state=state,
        nq=int(model.nq),
        nv=int(model.nv),
        na=int(model.na),
        nu=int(model.nu),
        state_size=state_size,
        step_index=int(step_index),
    )


def restore_physics_checkpoint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    checkpoint: PhysicsCheckpoint,
) -> None:
    """Restore a compatible checkpoint and rebuild all derived fields."""

    signature = (int(model.nq), int(model.nv), int(model.na), int(model.nu))
    expected = (checkpoint.nq, checkpoint.nv, checkpoint.na, checkpoint.nu)
    if signature != expected:
        raise ValueError(
            "checkpoint/model dimensions differ: "
            f"checkpoint={expected}, model={signature}"
        )
    state_size = int(mujoco.mj_stateSize(model, INTEGRATION_STATE_SPEC))
    if state_size != checkpoint.state_size:
        raise ValueError(
            "checkpoint/model integration state sizes differ: "
            f"checkpoint={checkpoint.state_size}, model={state_size}"
        )
    mujoco.mj_setState(
        model,
        data,
        np.asarray(checkpoint.state, dtype=np.float64),
        INTEGRATION_STATE_SPEC,
    )
    mujoco.mj_forward(model, data)


__all__ = [
    "INTEGRATION_STATE_SPEC",
    "PhysicsCheckpoint",
    "capture_physics_checkpoint",
    "restore_physics_checkpoint",
]
