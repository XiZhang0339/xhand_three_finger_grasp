"""Deterministic control trajectory helpers for the cube-lift task."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .config import (
    ACTIVE_ACTUATORS,
    contact_preload_targets,
    precontact_targets,
)


def smoothstep(alpha: float) -> float:
    alpha = min(1.0, max(0.0, float(alpha)))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def minimum_jerk(alpha: float) -> float:
    """Return the clamped quintic minimum-jerk position profile.

    The profile has zero velocity and acceleration at both endpoints.  It is
    intentionally separate from :func:`smoothstep`: schemas v1--v7 retain the
    cubic trajectory byte-for-byte, while schema v8 opts into this quintic
    profile through its versioned control protocol.
    """

    alpha = min(1.0, max(0.0, float(alpha)))
    return alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)


def quintic_c2_knot_derivatives(
    knot_times_s: np.ndarray,
    knot_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve shared knot derivatives for a clamped C4 quintic spline.

    The public name is retained because schema-v14 configs already identify
    this interpolation boundary as C2. A merely C2 path has a jerk jump at
    every planner knot, which is visible in the free cube even after the
    required 51 ms filter. Quintic pieces have exactly enough freedom to also
    match jerk and snap at every interior knot while fixing endpoint velocity
    and acceleration to zero. Schemas 1--13 never call this helper.
    """

    times = np.asarray(knot_times_s, dtype=np.float64)
    values = np.asarray(knot_values, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("knot_times_s must be a vector with at least two knots")
    if values.ndim < 1 or values.shape[0] != times.size:
        raise ValueError("knot_values leading axis must match knot_times_s")
    if (
        not np.isfinite(times).all()
        or not np.isfinite(values).all()
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("trajectory knots must be finite and strictly increasing")

    velocities = np.zeros_like(values, dtype=np.float64)
    accelerations = np.zeros_like(values, dtype=np.float64)
    interior_count = times.size - 2
    if interior_count <= 0:
        return velocities, accelerations

    flattened = values.reshape(times.size, -1)

    def continuity_residual(
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
    ) -> np.ndarray:
        residual = np.empty((2 * interior_count, position.shape[1]))
        for knot in range(1, times.size - 1):
            left_h = float(times[knot] - times[knot - 1])
            right_h = float(times[knot + 1] - times[knot])

            def coefficients(index: int, duration: float) -> tuple[np.ndarray, ...]:
                c0 = position[index]
                c1 = duration * velocity[index]
                c2 = 0.5 * duration**2 * acceleration[index]
                displacement = position[index + 1] - c0 - c1 - c2
                terminal_velocity = duration * velocity[index + 1] - c1 - 2.0 * c2
                terminal_acceleration = (
                    duration**2 * acceleration[index + 1] - 2.0 * c2
                )
                c3 = (
                    10.0 * displacement
                    - 4.0 * terminal_velocity
                    + 0.5 * terminal_acceleration
                )
                c4 = (
                    -15.0 * displacement
                    + 7.0 * terminal_velocity
                    - terminal_acceleration
                )
                c5 = (
                    6.0 * displacement
                    - 3.0 * terminal_velocity
                    + 0.5 * terminal_acceleration
                )
                return c0, c1, c2, c3, c4, c5

            *_, left_c3, left_c4, left_c5 = coefficients(knot - 1, left_h)
            *_, right_c3, right_c4, _ = coefficients(knot, right_h)
            left_jerk = (6.0 * left_c3 + 24.0 * left_c4 + 60.0 * left_c5) / (
                left_h**3
            )
            right_jerk = 6.0 * right_c3 / right_h**3
            left_snap = (24.0 * left_c4 + 120.0 * left_c5) / left_h**4
            right_snap = 24.0 * right_c4 / right_h**4
            row = 2 * (knot - 1)
            residual[row] = left_jerk - right_jerk
            residual[row + 1] = left_snap - right_snap
        return residual

    zeros = np.zeros_like(flattened)
    constant = continuity_residual(flattened, zeros, zeros)
    unknown_count = 2 * interior_count
    matrix = np.empty((unknown_count, unknown_count), dtype=np.float64)
    zero_position = np.zeros((times.size, 1), dtype=np.float64)
    for column in range(unknown_count):
        basis_velocity = np.zeros_like(zero_position)
        basis_acceleration = np.zeros_like(zero_position)
        interior = column // 2 + 1
        if column % 2 == 0:
            basis_velocity[interior, 0] = 1.0
        else:
            basis_acceleration[interior, 0] = 1.0
        matrix[:, column] = continuity_residual(
            zero_position, basis_velocity, basis_acceleration
        )[:, 0]
    solved = np.linalg.solve(matrix, -constant)
    for interior in range(interior_count):
        velocities[interior + 1] = solved[2 * interior].reshape(values.shape[1:])
        accelerations[interior + 1] = solved[2 * interior + 1].reshape(
            values.shape[1:]
        )
    return velocities, accelerations


def interpolate_quintic_c2(
    knot_times_s: np.ndarray,
    knot_values: np.ndarray,
    elapsed_s: float,
    *,
    knot_velocities: np.ndarray | None = None,
    knot_accelerations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Evaluate the C2 quintic Hermite path and its first two derivatives."""

    times = np.asarray(knot_times_s, dtype=np.float64)
    values = np.asarray(knot_values, dtype=np.float64)
    if knot_velocities is None or knot_accelerations is None:
        if knot_velocities is not None or knot_accelerations is not None:
            raise ValueError(
                "knot velocities and accelerations must be supplied together"
            )
        velocities, accelerations = quintic_c2_knot_derivatives(times, values)
    else:
        velocities = np.asarray(knot_velocities, dtype=np.float64)
        accelerations = np.asarray(knot_accelerations, dtype=np.float64)
        if velocities.shape != values.shape or accelerations.shape != values.shape:
            raise ValueError("knot derivative arrays must match knot_values")
        if not np.isfinite(velocities).all() or not np.isfinite(accelerations).all():
            raise ValueError("knot derivative arrays must be finite")

    elapsed = float(np.clip(float(elapsed_s), times[0], times[-1]))
    index = int(np.searchsorted(times, elapsed, side="right") - 1)
    index = min(max(index, 0), times.size - 2)
    duration = float(times[index + 1] - times[index])
    unit = (elapsed - float(times[index])) / duration

    value0 = values[index]
    value1 = values[index + 1]
    velocity0 = velocities[index]
    velocity1 = velocities[index + 1]
    acceleration0 = accelerations[index]
    acceleration1 = accelerations[index + 1]

    coefficient0 = value0
    coefficient1 = duration * velocity0
    coefficient2 = 0.5 * duration**2 * acceleration0
    displacement = value1 - coefficient0 - coefficient1 - coefficient2
    terminal_velocity = duration * velocity1 - coefficient1 - 2.0 * coefficient2
    terminal_acceleration = duration**2 * acceleration1 - 2.0 * coefficient2
    coefficient3 = (
        10.0 * displacement
        - 4.0 * terminal_velocity
        + 0.5 * terminal_acceleration
    )
    coefficient4 = (
        -15.0 * displacement
        + 7.0 * terminal_velocity
        - terminal_acceleration
    )
    coefficient5 = (
        6.0 * displacement
        - 3.0 * terminal_velocity
        + 0.5 * terminal_acceleration
    )

    value = (
        coefficient0
        + coefficient1 * unit
        + coefficient2 * unit**2
        + coefficient3 * unit**3
        + coefficient4 * unit**4
        + coefficient5 * unit**5
    )
    velocity = (
        coefficient1
        + 2.0 * coefficient2 * unit
        + 3.0 * coefficient3 * unit**2
        + 4.0 * coefficient4 * unit**3
        + 5.0 * coefficient5 * unit**4
    ) / duration
    acceleration = (
        2.0 * coefficient2
        + 6.0 * coefficient3 * unit
        + 12.0 * coefficient4 * unit**2
        + 20.0 * coefficient5 * unit**3
    ) / duration**2
    return (
        np.asarray(value, dtype=np.float64),
        np.asarray(velocity, dtype=np.float64),
        np.asarray(acceleration, dtype=np.float64),
        index,
    )


def actuator_target_vector(
    model: mujoco.MjModel, targets: dict[str, float]
) -> np.ndarray:
    vector = np.zeros(model.nu, dtype=np.float64)
    if set(targets) != set(ACTIVE_ACTUATORS):
        raise ValueError("target mapping must contain exactly the active actuators")
    for name, value in targets.items():
        actuator_id = model.actuator(name).id
        lower, upper = model.actuator_ctrlrange[actuator_id]
        value = float(value)
        if not lower <= value <= upper:
            raise ValueError(
                f"{name} target {value} is outside ctrlrange [{lower}, {upper}]"
            )
        vector[actuator_id] = value
    return vector


def preflight_config(config: dict[str, Any]) -> None:
    """Compile once and validate both target maps against real actuator limits."""

    # Local import preserves the dependency direction during staged extraction;
    # scene construction never needs to import trajectory helpers.
    from .scene import build_model

    model, _ = build_model(config)
    control = config["control"]
    if int(config.get("schema_version", 1)) >= 3:
        grasp_targets = contact_preload_targets(config)
        actuator_target_vector(model, grasp_targets)
        actuator_target_vector(model, precontact_targets(config))
        deltas = control["manipulation_delta_rad"]
        if set(deltas) != set(ACTIVE_ACTUATORS):
            raise ValueError(
                "manipulation_delta_rad must contain exactly the active actuators"
            )
        actuator_target_vector(
            model,
            {
                name: float(grasp_targets[name]) + float(deltas[name])
                for name in ACTIVE_ACTUATORS
            },
        )
    else:
        actuator_target_vector(model, control["pregrasp_targets_rad"])
        actuator_target_vector(model, control["final_targets_rad"])


def _phase_steps(model: mujoco.MjModel, config: dict[str, Any]) -> dict[str, int]:
    dt = float(model.opt.timestep)
    if int(config.get("schema_version", 1)) >= 3:
        protocol = config["control_protocol"]
        timings = (
            ("settle", "settle_s"),
            ("close", "close_s"),
            ("verify", "verify_timeout_s"),
            ("manipulate", "manipulate_s"),
            ("hold", "min_hold_s"),
        )
        steps: dict[str, int] = {}
        for phase, field in timings:
            seconds = float(protocol[field])
            count = int(round(seconds / dt))
            if count <= 0 or abs(count * dt - seconds) > 0.5 * dt + 1e-12:
                raise ValueError(
                    f"control_protocol.{field} does not align with timestep {dt}"
                )
            steps[phase] = count
        return steps

    steps: dict[str, int] = {}
    for name, seconds in config["timing"].items():
        count = int(round(float(seconds) / dt))
        if count <= 0 or abs(count * dt - float(seconds)) > 0.5 * dt + 1e-12:
            raise ValueError(f"timing.{name} does not align with timestep {dt}")
        steps[name.removesuffix("_s")] = count
    return steps
