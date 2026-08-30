"""Checkpoint-guided manipulation search for measured schema-v9 grasps.

The expensive part of a manipulation response experiment is obtaining the
same, independently verified free-body grasp for every actuator probe.  This
module runs the real controller from its initial no-contact state once,
captures MuJoCo's complete integration state on the ``grasp_lock`` sample,
and restores that state into a fresh :class:`mujoco.MjData` for every probe.

Checkpoint branching is *search evidence only*.  Every trust-region candidate
is ultimately passed to :func:`xhand_grasp.simulation.run_simulation`, which
starts from reset and must report ``stage_status.full_success`` before the
candidate can advance.
"""

from __future__ import annotations

import copy
import math
import multiprocessing
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..actual_contact_capability import (
    LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID,
    resolve_actual_contact_definition,
)
from ..checkpoint import (
    PhysicsCheckpoint,
    capture_physics_checkpoint,
    restore_physics_checkpoint,
)
from ..config import ACTIVE_ACTUATORS, contact_preload_targets, validate_config
from ..experiment import resolve_experiment
from ..scene import build_model
from ..simulation import SimulationSession, run_simulation
from ..trajectory import actuator_target_vector, minimum_jerk


EXPERIMENT_ID = LEGACY_ACTUAL_CONTACT_EXPERIMENT_ID
DEFAULT_SEED = 20260821
DEFAULT_TARGET_RESPONSE_6D = (0.0, 0.0, 0.011, 0.0, 0.0, 0.0)
_REFINEMENT_CANDIDATE_BASE = 109_000_000_000_000


@dataclass(frozen=True, slots=True)
class ManipulationSearchBudget:
    """Injectable budget for response probes and deterministic trust search."""

    probe_epsilon_rad: float = 0.02
    manipulate_s: float = 2.0
    hold_s: float = 1.0
    trust_candidate_count: int = 64
    target_upward_m: float = 0.011
    trust_radius_fraction: float = 0.15
    wide_candidate_fraction: float = 0.40
    wide_radius_fraction: float = 0.45
    ridge: float = 1e-4
    inward_preload_weight: float = 2e-3
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        finite_positive = (
            "probe_epsilon_rad",
            "manipulate_s",
            "hold_s",
            "target_upward_m",
            "trust_radius_fraction",
            "wide_radius_fraction",
            "ridge",
        )
        for name in finite_positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.trust_candidate_count, int) or isinstance(
            self.trust_candidate_count, bool
        ) or self.trust_candidate_count <= 0:
            raise ValueError("trust_candidate_count must be a positive integer")
        if not math.isfinite(float(self.inward_preload_weight)) or float(
            self.inward_preload_weight
        ) < 0.0:
            raise ValueError("inward_preload_weight must be finite and non-negative")
        if not math.isfinite(float(self.wide_candidate_fraction)) or not (
            0.0 <= float(self.wide_candidate_fraction) <= 1.0
        ):
            raise ValueError("wide_candidate_fraction must lie within [0, 1]")
        if float(self.wide_radius_fraction) < float(self.trust_radius_fraction):
            raise ValueError(
                "wide_radius_fraction must not be smaller than trust_radius_fraction"
            )
        if not 0.010 - 1e-12 <= float(self.target_upward_m) <= 0.012 + 1e-12:
            raise ValueError("target_upward_m must stay in the declared 10--12 mm band")


@dataclass(frozen=True, slots=True)
class LocalRefinementBudget:
    """Declared top-eight by 128 deterministic local refinement budget."""

    parent_count: int = 8
    candidates_per_parent: int = 128
    radius_fraction: float = 0.06
    batch_size: int = 64
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        for name in ("parent_count", "candidates_per_parent", "batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        radius = float(self.radius_fraction)
        if not math.isfinite(radius) or radius <= 0.0 or radius > 1.0:
            raise ValueError("radius_fraction must lie within (0, 1]")

    @property
    def maximum_candidate_count(self) -> int:
        return self.parent_count * self.candidates_per_parent


@dataclass(frozen=True, slots=True)
class GraspPhysicsCheckpoint:
    """A model-bound checkpoint captured on a verified grasp-lock sample."""

    model: mujoco.MjModel
    checkpoint: PhysicsCheckpoint
    cube_body_id: int
    grasp_lock_step: int
    # Stable-window median: this is the schema-v9 grasp pose persisted in NPZ.
    actual_grasp_qpos_rad: np.ndarray
    # Instantaneous qpos in the exact post-step integration state checkpoint.
    lock_sample_joint_qpos_rad: np.ndarray
    cube_position_world_m: np.ndarray
    cube_quaternion_wxyz: np.ndarray
    config: dict[str, Any]

    def __post_init__(self) -> None:
        actual = np.asarray(self.actual_grasp_qpos_rad, dtype=np.float64)
        lock_sample = np.asarray(self.lock_sample_joint_qpos_rad, dtype=np.float64)
        position = np.asarray(self.cube_position_world_m, dtype=np.float64)
        quaternion = np.asarray(self.cube_quaternion_wxyz, dtype=np.float64)
        if actual.shape != (len(ACTIVE_ACTUATORS),) or not np.isfinite(actual).all():
            raise ValueError("actual_grasp_qpos_rad must contain eight finite values")
        if lock_sample.shape != (len(ACTIVE_ACTUATORS),) or not np.isfinite(
            lock_sample
        ).all():
            raise ValueError(
                "lock_sample_joint_qpos_rad must contain eight finite values"
            )
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("cube_position_world_m must contain three finite values")
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("cube_quaternion_wxyz must contain four finite values")
        if not 0 <= int(self.cube_body_id) < int(self.model.nbody):
            raise ValueError("cube_body_id is invalid for checkpoint model")
        if int(self.grasp_lock_step) < 0:
            raise ValueError("grasp_lock_step must be non-negative")
        object.__setattr__(self, "actual_grasp_qpos_rad", actual.copy())
        object.__setattr__(self, "lock_sample_joint_qpos_rad", lock_sample.copy())
        object.__setattr__(self, "cube_position_world_m", position.copy())
        object.__setattr__(self, "cube_quaternion_wxyz", quaternion.copy())
        object.__setattr__(self, "config", copy.deepcopy(dict(self.config)))


@dataclass(frozen=True, slots=True)
class ProbeSpecification:
    kind: str
    actuator: str | None
    direction: int
    requested_delta_rad: tuple[float, ...]
    applied_delta_rad: tuple[float, ...]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "actuator": self.actuator,
            "direction": int(self.direction),
            "requested_delta_rad": {
                name: float(self.requested_delta_rad[index])
                for index, name in enumerate(ACTIVE_ACTUATORS)
            },
            "applied_delta_rad": {
                name: float(self.applied_delta_rad[index])
                for index, name in enumerate(ACTIVE_ACTUATORS)
            },
        }


def _scalar_int(trace: Mapping[str, Any], key: str) -> int:
    value = np.asarray(trace[key])
    if value.size != 1:
        raise ValueError(f"{key} must be a scalar trace field")
    return int(value.reshape(-1)[0])


def _load_trace(trace_or_path: Mapping[str, Any] | str | Path) -> tuple[Mapping[str, Any], Any]:
    if isinstance(trace_or_path, Mapping):
        return trace_or_path, None
    archive = np.load(Path(trace_or_path), allow_pickle=False)
    return archive, archive


def validate_grasp_success_source(
    config: Mapping[str, Any],
    trace_or_path: Mapping[str, Any] | str | Path,
    result: Mapping[str, Any],
) -> int:
    """Authenticate the v9 grasp result consumed by checkpoint search."""

    resolve_actual_contact_definition(
        config, context="checkpoint manipulation"
    )
    summary = result.get("summary", result)
    stage = summary.get("stage_status", {}) if isinstance(summary, Mapping) else {}
    if not isinstance(stage, Mapping) or stage.get("grasp_success") is not True:
        raise ValueError("source result is not a verified grasp-success result")
    trace, closeable = _load_trace(trace_or_path)
    try:
        lock_step = _scalar_int(trace, "grasp_lock_step")
        if lock_step < 0:
            raise ValueError("source trace has no grasp_lock event")
        if "grasp_pose_actual_qpos_rad" not in trace:
            raise ValueError("source trace has no measured grasp qpos")
        actual = np.asarray(trace["grasp_pose_actual_qpos_rad"], dtype=np.float64)
        if actual.shape != (len(ACTIVE_ACTUATORS),) or not np.isfinite(actual).all():
            raise ValueError("source measured grasp qpos is invalid")
        return lock_step
    finally:
        if closeable is not None:
            closeable.close()


def acquire_grasp_checkpoint(
    config: Mapping[str, Any],
    *,
    expected_lock_step: int | None = None,
    session_factory: Callable[[dict[str, Any]], SimulationSession] = SimulationSession,
) -> GraspPhysicsCheckpoint:
    """Full-reset rerun through grasp lock, then capture integration state.

    This is deliberately not reconstructed from a persisted NPZ.  The cube is
    free throughout the rerun and the controller must reacquire the grasp.
    """

    resolved = copy.deepcopy(dict(config))
    resolve_actual_contact_definition(
        resolved, context="grasp checkpoint acquisition"
    )
    session = session_factory(resolved)
    try:
        while not session.complete:
            event = session.advance_one()
            controller = session.controller
            if controller is not None and controller.acquired:
                lock_step = int(controller.grasp_acquisition_step)
                if lock_step != int(event.index):
                    raise RuntimeError("grasp acquisition event/sample boundary mismatch")
                if expected_lock_step is not None and lock_step != int(expected_lock_step):
                    raise RuntimeError(
                        "fresh grasp-lock step differs from source trace: "
                        f"fresh={lock_step}, source={int(expected_lock_step)}"
                    )
                checkpoint = capture_physics_checkpoint(
                    session.model,
                    session.data,
                    step_index=session.step_index,
                )
                active_ids = np.asarray(
                    [session.model.actuator(name).id for name in ACTIVE_ACTUATORS],
                    dtype=np.int64,
                )
                lock_sample = np.asarray(
                    session.data.qpos[session.info.actuator_qpos_adrs][active_ids],
                    dtype=np.float64,
                )
                actual = np.asarray(
                    controller.grasp_pose_actual_qpos_rad, dtype=np.float64
                ).copy()
                return GraspPhysicsCheckpoint(
                    model=session.model,
                    checkpoint=checkpoint,
                    cube_body_id=int(session.info.cube_body_id),
                    grasp_lock_step=lock_step,
                    actual_grasp_qpos_rad=actual,
                    lock_sample_joint_qpos_rad=lock_sample,
                    cube_position_world_m=np.asarray(
                        session.data.xpos[session.info.cube_body_id], dtype=np.float64
                    ),
                    cube_quaternion_wxyz=np.asarray(
                        session.data.xquat[session.info.cube_body_id], dtype=np.float64
                    ),
                    config=resolved,
                )
        raise RuntimeError("fresh full-dynamics rerun ended without grasp_lock")
    finally:
        session.close()


def prepare_grasp_checkpoint(
    config: Mapping[str, Any],
    trace_or_path: Mapping[str, Any] | str | Path,
    result: Mapping[str, Any],
    *,
    session_factory: Callable[[dict[str, Any]], SimulationSession] = SimulationSession,
) -> GraspPhysicsCheckpoint:
    """Validate persisted evidence and deterministically reacquire the grasp."""

    lock_step = validate_grasp_success_source(config, trace_or_path, result)
    source_trace, closeable = _load_trace(trace_or_path)
    try:
        expected_actual = np.asarray(
            source_trace["grasp_pose_actual_qpos_rad"], dtype=np.float64
        ).copy()
    finally:
        if closeable is not None:
            closeable.close()
    grasp = acquire_grasp_checkpoint(
        config,
        expected_lock_step=lock_step,
        session_factory=session_factory,
    )
    if not np.array_equal(grasp.actual_grasp_qpos_rad, expected_actual):
        raise RuntimeError(
            "fresh stable-window actual grasp qpos differs from source trace"
        )
    return grasp


def manipulation_delta_bounds(
    model: mujoco.MjModel,
    config: Mapping[str, Any],
) -> dict[str, tuple[float, float]]:
    """Intersect registered search bounds with real absolute ctrl limits."""

    preload = contact_preload_targets(copy.deepcopy(dict(config)))
    definition = resolve_experiment(config)
    registered = definition.search_bounds.manipulation_delta_rad
    result: dict[str, tuple[float, float]] = {}
    for name in ACTIVE_ACTUATORS:
        actuator_id = int(model.actuator(name).id)
        ctrl_lower, ctrl_upper = (
            float(value) for value in model.actuator_ctrlrange[actuator_id]
        )
        lower = ctrl_lower - float(preload[name])
        upper = ctrl_upper - float(preload[name])
        if registered is not None:
            lower = max(lower, float(registered[name][0]))
            upper = min(upper, float(registered[name][1]))
        if lower > upper:
            raise ValueError(f"no valid manipulation range remains for {name}")
        result[name] = (lower, upper)
    return result


def clip_manipulation_delta(
    delta: Mapping[str, float] | Sequence[float],
    bounds: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    """Clip one eight-axis relative command to deterministic named bounds."""

    if set(bounds) != set(ACTIVE_ACTUATORS):
        raise ValueError("bounds must contain exactly the eight active actuators")
    if isinstance(delta, Mapping):
        if set(delta) != set(ACTIVE_ACTUATORS):
            raise ValueError("delta must contain exactly the eight active actuators")
        values = [float(delta[name]) for name in ACTIVE_ACTUATORS]
    else:
        values = [float(value) for value in delta]
        if len(values) != len(ACTIVE_ACTUATORS):
            raise ValueError("delta sequence must contain eight values")
    if not np.isfinite(values).all():
        raise ValueError("delta values must be finite")
    return {
        name: float(np.clip(values[index], bounds[name][0], bounds[name][1]))
        for index, name in enumerate(ACTIVE_ACTUATORS)
    }


def generate_probe_specifications(
    model: mujoco.MjModel,
    config: Mapping[str, Any],
    *,
    epsilon_rad: float = 0.02,
) -> tuple[ProbeSpecification, ...]:
    """Return zero plus +/- one-axis probes, always in the canonical order."""

    if not math.isfinite(float(epsilon_rad)) or float(epsilon_rad) <= 0.0:
        raise ValueError("epsilon_rad must be positive and finite")
    bounds = manipulation_delta_bounds(model, config)
    zeros = np.zeros(len(ACTIVE_ACTUATORS), dtype=np.float64)
    specs = [
        ProbeSpecification("zero", None, 0, tuple(zeros), tuple(zeros))
    ]
    for column, name in enumerate(ACTIVE_ACTUATORS):
        for direction in (-1, 1):
            requested = zeros.copy()
            requested[column] = direction * float(epsilon_rad)
            clipped = clip_manipulation_delta(requested, bounds)
            applied = tuple(float(clipped[item]) for item in ACTIVE_ACTUATORS)
            specs.append(
                ProbeSpecification(
                    "single_actuator",
                    name,
                    direction,
                    tuple(float(value) for value in requested),
                    applied,
                )
            )
    if len(specs) != 1 + 2 * len(ACTIVE_ACTUATORS):
        raise AssertionError("probe construction no longer yields exactly 17 probes")
    return tuple(specs)


def _quat_rotation_vector_wxyz(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    sw, sx, sy, sz = start
    inverse = np.asarray((sw, -sx, -sy, -sz), dtype=np.float64)
    ew, ex, ey, ez = end
    iw, ix, iy, iz = inverse
    relative = np.asarray(
        (
            ew * iw - ex * ix - ey * iy - ez * iz,
            ew * ix + ex * iw + ey * iz - ez * iy,
            ew * iy - ex * iz + ey * iw + ez * ix,
            ew * iz + ex * iy - ey * ix + ez * iw,
        ),
        dtype=np.float64,
    )
    relative /= np.linalg.norm(relative)
    if relative[0] < 0.0:
        relative *= -1.0
    vector_norm = float(np.linalg.norm(relative[1:]))
    if vector_norm <= 1e-15:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(np.clip(relative[0], -1.0, 1.0)))
    return relative[1:] * (angle / vector_norm)


def _aligned_step_count(seconds: float, timestep: float, label: str) -> int:
    count = int(round(float(seconds) / float(timestep)))
    if count <= 0 or abs(count * float(timestep) - float(seconds)) > 0.5 * timestep + 1e-12:
        raise ValueError(f"{label} does not align with model timestep")
    return count


def run_checkpoint_probe(
    grasp: GraspPhysicsCheckpoint,
    specification: ProbeSpecification,
    *,
    manipulate_s: float = 2.0,
    hold_s: float = 1.0,
) -> dict[str, Any]:
    """Restore one independent branch and execute minimum-jerk + hold."""

    model = grasp.model
    data = mujoco.MjData(model)
    restore_physics_checkpoint(model, data, grasp.checkpoint)
    start_position = np.asarray(data.xpos[grasp.cube_body_id], dtype=np.float64).copy()
    start_quaternion = np.asarray(data.xquat[grasp.cube_body_id], dtype=np.float64).copy()
    preload = actuator_target_vector(
        model, contact_preload_targets(copy.deepcopy(grasp.config))
    )
    delta = np.zeros(model.nu, dtype=np.float64)
    for index, name in enumerate(ACTIVE_ACTUATORS):
        delta[model.actuator(name).id] = float(specification.applied_delta_rad[index])
    manipulate_steps = _aligned_step_count(manipulate_s, model.opt.timestep, "manipulate_s")
    hold_steps = _aligned_step_count(hold_s, model.opt.timestep, "hold_s")
    for local_step in range(manipulate_steps):
        alpha = minimum_jerk((local_step + 1) / manipulate_steps)
        data.ctrl[:] = preload + alpha * delta
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
    manipulation_position = np.asarray(
        data.xpos[grasp.cube_body_id], dtype=np.float64
    ).copy()
    manipulation_quaternion = np.asarray(
        data.xquat[grasp.cube_body_id], dtype=np.float64
    ).copy()
    for _ in range(hold_steps):
        data.ctrl[:] = preload + delta
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
    final_position = np.asarray(data.xpos[grasp.cube_body_id], dtype=np.float64).copy()
    final_quaternion = np.asarray(data.xquat[grasp.cube_body_id], dtype=np.float64).copy()
    translation = manipulation_position - start_position
    rotation = _quat_rotation_vector_wxyz(start_quaternion, manipulation_quaternion)
    hold_translation = final_position - manipulation_position
    hold_rotation = _quat_rotation_vector_wxyz(manipulation_quaternion, final_quaternion)
    return {
        "probe": specification.as_mapping(),
        "search_branch_source": "grasp_lock_mjstate_integration_checkpoint",
        "command_reference": "contact_preload_targets_rad",
        "manipulation_profile": "minimum_jerk_quintic",
        "initial_active_command_rad": {
            name: float(preload[model.actuator(name).id])
            for name in ACTIVE_ACTUATORS
        },
        "terminal_active_command_rad": {
            name: float(
                preload[model.actuator(name).id]
                + delta[model.actuator(name).id]
            )
            for name in ACTIVE_ACTUATORS
        },
        "checkpoint_step_index": int(grasp.checkpoint.step_index),
        "grasp_lock_step": int(grasp.grasp_lock_step),
        "manipulate_steps": int(manipulate_steps),
        "hold_steps": int(hold_steps),
        "response_6d": np.concatenate((translation, rotation)).tolist(),
        "translation_world_m": translation.tolist(),
        "rotation_vector_world_rad": rotation.tolist(),
        "hold_translation_world_m": hold_translation.tolist(),
        "hold_rotation_vector_world_rad": hold_rotation.tolist(),
        "final_cube_position_world_m": final_position.tolist(),
        "final_cube_quaternion_wxyz": final_quaternion.tolist(),
        "finite": bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.ctrl).all()
        ),
    }


def run_checkpoint_probe_set(
    grasp: GraspPhysicsCheckpoint,
    *,
    budget: ManipulationSearchBudget = ManipulationSearchBudget(),
    probe_runner: Callable[..., Mapping[str, Any]] = run_checkpoint_probe,
) -> tuple[dict[str, Any], ...]:
    """Run all 17 probes from independent restorations of one checkpoint."""

    specifications = generate_probe_specifications(
        grasp.model, grasp.config, epsilon_rad=budget.probe_epsilon_rad
    )
    return tuple(
        copy.deepcopy(
            dict(
                probe_runner(
                    grasp,
                    specification,
                    manipulate_s=budget.manipulate_s,
                    hold_s=budget.hold_s,
                )
            )
        )
        for specification in specifications
    )


def fit_response_jacobian(
    probe_results: Sequence[Mapping[str, Any]],
    bounds: Mapping[str, Sequence[float]],
    *,
    config: Mapping[str, Any],
    target_response_6d: Sequence[float] = DEFAULT_TARGET_RESPONSE_6D,
    ridge: float = 1e-4,
    inward_preload_weight: float = 2e-3,
) -> dict[str, Any]:
    """Fit a central 6x8 response Jacobian and bounded least-squares target."""

    if len(probe_results) != 1 + 2 * len(ACTIVE_ACTUATORS):
        raise ValueError("response fitting requires the complete 17-probe set")
    target = np.asarray(target_response_6d, dtype=np.float64)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("target_response_6d must contain six finite values")
    if not 0.010 - 1e-12 <= target[2] <= 0.012 + 1e-12:
        raise ValueError("vertical response target must stay in the 10--12 mm band")
    zero = next(
        (
            record
            for record in probe_results
            if record.get("probe", {}).get("kind") == "zero"
        ),
        None,
    )
    if zero is None:
        raise ValueError("probe set has no zero branch")
    bias = np.asarray(zero["response_6d"], dtype=np.float64)
    if bias.shape != (6,) or not np.isfinite(bias).all():
        raise ValueError("zero probe response is invalid")
    matrix = np.zeros((6, len(ACTIVE_ACTUATORS)), dtype=np.float64)
    columns = []
    for column, name in enumerate(ACTIVE_ACTUATORS):
        directional: dict[int, tuple[float, np.ndarray]] = {}
        for record in probe_results:
            metadata = record.get("probe", {})
            if metadata.get("actuator") != name:
                continue
            direction = int(metadata.get("direction", 0))
            applied = float(metadata.get("applied_delta_rad", {}).get(name, 0.0))
            response = np.asarray(record.get("response_6d"), dtype=np.float64)
            if direction in (-1, 1) and abs(applied) > 1e-15 and response.shape == (6,):
                directional[direction] = (applied, response)
        method = "missing"
        negative_step = 0.0
        positive_step = 0.0
        if set(directional) == {-1, 1}:
            negative_step, negative_response = directional[-1]
            positive_step, positive_response = directional[1]
            denominator = positive_step - negative_step
            if abs(denominator) > 1e-15:
                matrix[:, column] = (
                    positive_response - negative_response
                ) / denominator
                method = "central"
        elif directional:
            direction = 1 if 1 in directional else -1
            step, response = directional[direction]
            matrix[:, column] = (response - bias) / step
            method = "forward" if direction == 1 else "backward"
            if direction == 1:
                positive_step = step
            else:
                negative_step = step
        if method == "missing":
            raise ValueError(f"probe pair for {name} collapsed at command limits")
        columns.append(
            {
                "actuator": name,
                "method": method,
                "negative_step_rad": float(negative_step),
                "positive_step_rad": float(positive_step),
                "column_norm": float(np.linalg.norm(matrix[:, column])),
            }
        )

    scales = np.asarray(
        (0.002, 0.002, 0.011, math.radians(10.0), math.radians(10.0), math.radians(10.0)),
        dtype=np.float64,
    )
    weighted = matrix / scales[:, None]
    desired = (target - bias) / scales
    preload = contact_preload_targets(copy.deepcopy(dict(config)))
    nominal = config["grasp_pose"]["nominal_joint_qpos_rad"]
    inward = np.asarray(
        [
            np.clip(float(preload[name]) - float(nominal[name]), -0.01, 0.01)
            for name in ACTIVE_ACTUATORS
        ],
        dtype=np.float64,
    )
    lower = np.asarray([float(bounds[name][0]) for name in ACTIVE_ACTUATORS])
    upper = np.asarray([float(bounds[name][1]) for name in ACTIVE_ACTUATORS])
    hessian = (
        weighted.T @ weighted
        + (float(ridge) + float(inward_preload_weight))
        * np.eye(len(ACTIVE_ACTUATORS))
    )
    offset = (
        weighted.T @ desired
        + float(inward_preload_weight) * inward
    )
    try:
        solution = np.linalg.solve(hessian, offset)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(hessian, offset, rcond=None)[0]
    solution = np.clip(solution, lower, upper)
    lipschitz = max(float(np.linalg.eigvalsh(hessian)[-1]), 1e-12)
    for _ in range(512):
        updated = np.clip(solution - (hessian @ solution - offset) / lipschitz, lower, upper)
        if float(np.max(np.abs(updated - solution))) <= 1e-13:
            solution = updated
            break
        solution = updated
    predicted = bias + matrix @ solution
    return {
        "available": True,
        "probe_count": len(probe_results),
        "columns": columns,
        "jacobian_6x8": matrix.tolist(),
        "bias_response_6d": bias.tolist(),
        "target_response_6d": target.tolist(),
        "inward_preload_preference_rad": {
            name: float(inward[index]) for index, name in enumerate(ACTIVE_ACTUATORS)
        },
        "solution_delta_rad": {
            name: float(solution[index]) for index, name in enumerate(ACTIVE_ACTUATORS)
        },
        "predicted_response_6d": predicted.tolist(),
        "matrix_rank": int(np.linalg.matrix_rank(weighted)),
        "weighted_residual_norm": float(np.linalg.norm((predicted - target) / scales)),
    }


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, dimensions), dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty((count, dimensions), dtype=np.float64)
    for column in range(dimensions):
        permutation = rng.permutation(count)
        values[:, column] = (permutation + rng.random(count)) / count
    return 2.0 * values - 1.0


def generate_trust_region_deltas(
    response_model: Mapping[str, Any],
    bounds: Mapping[str, Sequence[float]],
    *,
    count: int = 64,
    seed: int = DEFAULT_SEED,
    trust_radius_fraction: float = 0.15,
    wide_candidate_fraction: float = 0.40,
    wide_radius_fraction: float = 0.45,
) -> tuple[dict[str, float], ...]:
    """Generate deterministic near-solution and wider bounded candidates."""

    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 <= float(wide_candidate_fraction) <= 1.0:
        raise ValueError("wide_candidate_fraction must lie within [0, 1]")
    if (
        not math.isfinite(float(trust_radius_fraction))
        or float(trust_radius_fraction) <= 0.0
        or not math.isfinite(float(wide_radius_fraction))
        or float(wide_radius_fraction) < float(trust_radius_fraction)
    ):
        raise ValueError("trust/wide radii must be finite, positive and ordered")
    solved = clip_manipulation_delta(response_model["solution_delta_rad"], bounds)
    solved_vector = np.asarray([solved[name] for name in ACTIVE_ACTUATORS])
    lower = np.asarray([float(bounds[name][0]) for name in ACTIVE_ACTUATORS])
    upper = np.asarray([float(bounds[name][1]) for name in ACTIVE_ACTUATORS])
    candidates: list[np.ndarray] = [solved_vector]
    for scale in (0.9, 1.1, 0.75):
        if len(candidates) < count:
            candidates.append(np.clip(scale * solved_vector, lower, upper))
    remaining = count - len(candidates)
    wide_count = (
        0
        if remaining <= 0 or float(wide_candidate_fraction) == 0.0
        else min(
            remaining,
            max(1, int(round(remaining * float(wide_candidate_fraction)))),
        )
    )
    near_count = remaining - wide_count
    near_units = _latin_hypercube(
        near_count,
        len(ACTIVE_ACTUATORS),
        int(
            np.random.SeedSequence(
                [int(seed), count, 9_000_064, 0]
            ).generate_state(1)[0]
        ),
    )
    near_radius = float(trust_radius_fraction) * (upper - lower)
    for unit in near_units:
        candidates.append(
            np.clip(solved_vector + unit * near_radius, lower, upper)
        )
    wide_units = _latin_hypercube(
        wide_count,
        len(ACTIVE_ACTUATORS),
        int(
            np.random.SeedSequence(
                [int(seed), count, 9_000_064, 1]
            ).generate_state(1)[0]
        ),
    )
    wide_radius = float(wide_radius_fraction) * (upper - lower)
    for unit in wide_units:
        candidates.append(
            np.clip(solved_vector + unit * wide_radius, lower, upper)
        )
    return tuple(
        {
            name: float(vector[index])
            for index, name in enumerate(ACTIVE_ACTUATORS)
        }
        for vector in candidates[:count]
    )


def _finite_metric(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def manipulation_candidate_rank_evidence(
    record: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract lift, topology and smoothness normalized acceptance margins."""

    summary = record.get("summary", record)
    if not isinstance(summary, Mapping):
        summary = {}
    stage = summary.get("stage_status", {})
    if not isinstance(stage, Mapping):
        stage = {}
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    resolved = config or record.get("config")
    if not isinstance(resolved, Mapping):
        raise ValueError("candidate ranking requires a resolved config")
    acceptance = resolved["acceptance"]
    median_limit = float(acceptance["median_lift_m"])
    minimum_limit = float(acceptance["minimum_lift_m"])
    median = _finite_metric(metrics.get("operation_median_lift_m"), -math.inf)
    minimum = _finite_metric(metrics.get("operation_minimum_lift_m"), -math.inf)
    lift_margin = min(
        (median - median_limit) / median_limit,
        (minimum - minimum_limit) / minimum_limit,
    )

    required_topology = float(acceptance["simultaneous_contact_duty"])
    topology = _finite_metric(
        metrics.get("operation_target_face_simultaneous_duty"), -math.inf
    )
    alignment = metrics.get("contact_alignment", {})
    operation_alignment = (
        alignment.get("operation", {}) if isinstance(alignment, Mapping) else {}
    )
    aligned_duty = _finite_metric(
        operation_alignment.get("aligned_duty")
        if isinstance(operation_alignment, Mapping)
        else None,
        topology,
    )
    topology_margin = min(
        (topology - required_topology) / required_topology,
        (aligned_duty - required_topology) / required_topology,
    )

    smooth = metrics.get("motion_smoothness", {})
    if not isinstance(smooth, Mapping):
        smooth = {}
    thresholds = resolved["motion_smoothness"]
    smooth_pairs = (
        (
            "operation_cumulative_height_backtrack_m",
            "max_cumulative_height_backtrack_m",
        ),
        ("operation_downward_speed_duty", "max_downward_speed_duty"),
        ("operation_peak_filtered_upward_speed_m_s", "max_peak_upward_speed_m_s"),
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
        ("operation_max_lateral_displacement_m", "max_lateral_displacement_m"),
        ("operation_max_orientation_drift_deg", "max_orientation_drift_deg"),
    )
    smooth_margins = []
    for metric_name, threshold_name in smooth_pairs:
        limit = float(thresholds[threshold_name])
        value = _finite_metric(smooth.get(metric_name), math.inf)
        smooth_margins.append((limit - value) / limit)
    smooth_margin = min(smooth_margins, default=-math.inf)
    failed = summary.get("failed_checks", [])
    failed_count = len(failed) if isinstance(failed, list) else 10**9
    return {
        "full_success": bool(stage.get("full_success", False)),
        "grasp_success": bool(stage.get("grasp_success", False)),
        "operation_median_lift_m": median if math.isfinite(median) else None,
        "operation_minimum_lift_m": minimum if math.isfinite(minimum) else None,
        "lift_min_normalized_margin": (
            lift_margin if math.isfinite(lift_margin) else None
        ),
        "operation_topology_duty": topology if math.isfinite(topology) else None,
        "operation_aligned_duty": (
            aligned_duty if math.isfinite(aligned_duty) else None
        ),
        "topology_min_normalized_margin": (
            topology_margin if math.isfinite(topology_margin) else None
        ),
        "smoothness_min_normalized_margin": (
            smooth_margin if math.isfinite(smooth_margin) else None
        ),
        "failed_check_count": int(failed_count),
    }


def manipulation_candidate_rank(
    record: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Hard success, then lift/topology/smoothness margin, then stable ID."""

    evidence = manipulation_candidate_rank_evidence(record, config=config)
    return (
        not evidence["full_success"],
        not evidence["grasp_success"],
        -_finite_metric(evidence["lift_min_normalized_margin"], -math.inf),
        -_finite_metric(evidence["topology_min_normalized_margin"], -math.inf),
        -_finite_metric(evidence["smoothness_min_normalized_margin"], -math.inf),
        int(evidence["failed_check_count"]),
        int(record.get("candidate_id", record.get("candidate_index", 2**63 - 1))),
    )


def rank_manipulation_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Worker-order-independent manipulation result ordering."""

    materialized = [copy.deepcopy(dict(record)) for record in records]
    materialized.sort(key=lambda record: manipulation_candidate_rank(record, config=config))
    return tuple(materialized)


def generate_local_refinement_candidates(
    parent_records: Sequence[Mapping[str, Any]],
    bounds: Mapping[str, Sequence[float]],
    *,
    config: Mapping[str, Any],
    budget: LocalRefinementBudget = LocalRefinementBudget(),
) -> tuple[dict[str, Any], ...]:
    """Generate balanced top-parent local candidates with immutable IDs."""

    ranked = rank_manipulation_candidates(parent_records, config=config)
    return _generate_ranked_local_refinement_candidates(
        ranked,
        budget=budget,
        bounds_resolver=lambda _parent: bounds,
    )


def _generate_ranked_local_refinement_candidates(
    ranked_parent_records: Sequence[Mapping[str, Any]],
    *,
    budget: LocalRefinementBudget,
    bounds_resolver: Callable[
        [Mapping[str, Any]], Mapping[str, Sequence[float]]
    ],
) -> tuple[dict[str, Any], ...]:
    """Materialize an interleaved refinement set around already-ranked parents."""

    parents = tuple(
        copy.deepcopy(dict(value))
        for value in ranked_parent_records[: budget.parent_count]
    )
    if len(parents) < budget.parent_count:
        raise ValueError(
            f"local refinement requires {budget.parent_count} parent results"
        )
    bounds_by_parent: list[tuple[np.ndarray, np.ndarray]] = []
    per_parent_units = []
    for parent_rank, parent in enumerate(parents):
        bounds = bounds_resolver(parent)
        if set(bounds) != set(ACTIVE_ACTUATORS):
            raise ValueError(
                "each refinement parent must resolve exactly eight actuator bounds"
            )
        lower = np.asarray(
            [float(bounds[name][0]) for name in ACTIVE_ACTUATORS],
            dtype=np.float64,
        )
        upper = np.asarray(
            [float(bounds[name][1]) for name in ACTIVE_ACTUATORS],
            dtype=np.float64,
        )
        if (
            lower.shape != (len(ACTIVE_ACTUATORS),)
            or upper.shape != lower.shape
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(lower > upper)
        ):
            raise ValueError("refinement parent actuator bounds are invalid")
        bounds_by_parent.append((lower, upper))
        units = _latin_hypercube(
            budget.candidates_per_parent,
            len(ACTIVE_ACTUATORS),
            int(
                np.random.SeedSequence(
                    [budget.seed, int(parent["candidate_id"]), parent_rank, 9_128_008]
                ).generate_state(1)[0]
            ),
        )
        # Preserve the exact parent as the first deterministic local anchor.
        units[0] = 0.0
        per_parent_units.append(units)
    generated = []
    # Interleave parents so every resumable batch samples all eight basins.
    for local_index in range(budget.candidates_per_parent):
        for parent_rank, parent in enumerate(parents):
            lower, upper = bounds_by_parent[parent_rank]
            radius = float(budget.radius_fraction) * (upper - lower)
            center = np.asarray(
                [float(parent["manipulation_delta_rad"][name]) for name in ACTIVE_ACTUATORS],
                dtype=np.float64,
            )
            values = np.clip(
                center + per_parent_units[parent_rank][local_index] * radius,
                lower,
                upper,
            )
            candidate_id = (
                _REFINEMENT_CANDIDATE_BASE
                + local_index * budget.parent_count
                + parent_rank
            )
            generated.append(
                {
                    "candidate_id": int(candidate_id),
                    "parent_candidate_id": int(parent["candidate_id"]),
                    "parent_rank": int(parent_rank),
                    "local_index": int(local_index),
                    "manipulation_delta_rad": {
                        name: float(values[index])
                        for index, name in enumerate(ACTIVE_ACTUATORS)
                    },
                }
            )
    return tuple(generated)


def generate_multiconfig_local_refinement_candidates(
    parent_records: Sequence[Mapping[str, Any]],
    *,
    budget: LocalRefinementBudget = LocalRefinementBudget(),
    bounds_resolver: Callable[
        [Mapping[str, Any]], Mapping[str, Sequence[float]]
    ]
    | None = None,
) -> tuple[dict[str, Any], ...]:
    """Generate global top-eight refinements when parents use different poses.

    Every parent record must carry its own resolved ``config``.  Ranking and
    actuator clipping therefore use the exact cube/hand/controller pair that
    produced that parent, instead of silently applying the first source's
    bounds to all eight basins.
    """

    ranked = rank_manipulation_candidates(parent_records)

    def registered_bounds(parent: Mapping[str, Any]) -> Mapping[str, Sequence[float]]:
        config = parent.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("multi-config refinement parents require resolved config")
        model, _ = build_model(copy.deepcopy(dict(config)))
        return manipulation_delta_bounds(model, config)

    return _generate_ranked_local_refinement_candidates(
        ranked,
        budget=budget,
        bounds_resolver=(registered_bounds if bounds_resolver is None else bounds_resolver),
    )


def materialize_manipulation_config(
    base_config: Mapping[str, Any],
    delta: Mapping[str, float],
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Apply only a relative preload manipulation delta to a v9 config."""

    candidate = copy.deepcopy(dict(base_config))
    # Compile once so clipping uses the exact resolved model.  Avoid allocating
    # a full SimulationSession/traces for this purely mechanical operation.
    model, _ = build_model(candidate)
    bounds = manipulation_delta_bounds(model, candidate)
    clipped = clip_manipulation_delta(delta, bounds)
    candidate["control"]["manipulation_delta_rad"] = clipped
    if validate:
        validate_config(candidate)
    return candidate


def _materialize_with_model(
    base_config: Mapping[str, Any],
    delta: Mapping[str, float],
    model: mujoco.MjModel,
) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(base_config))
    candidate["control"]["manipulation_delta_rad"] = clip_manipulation_delta(
        delta, manipulation_delta_bounds(model, candidate)
    )
    validate_config(candidate)
    return candidate


def _run_full_reset_candidate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Spawn-safe production execution of one full reset candidate."""

    summary = run_simulation(copy.deepcopy(dict(job["config"])))
    return {
        "candidate_id": int(job["candidate_id"]),
        "summary": summary,
    }


def run_full_reset_candidate_jobs(
    jobs: Sequence[Mapping[str, Any]], workers: int
) -> tuple[dict[str, Any], ...]:
    """Execute full reruns using deterministic ``spawn`` process semantics."""

    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not jobs:
        return ()
    if workers == 1:
        results = [_run_full_reset_candidate_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            results = list(
                pool.map(_run_full_reset_candidate_job, jobs, chunksize=1)
            )
    results.sort(key=lambda value: int(value["candidate_id"]))
    return tuple(results)


FullResetExecutor = Callable[
    [Sequence[Mapping[str, Any]], int], Sequence[Mapping[str, Any]]
]

def run_full_reset_candidates(
    base_config: Mapping[str, Any],
    deltas: Sequence[Mapping[str, float]],
    *,
    model: mujoco.MjModel | None = None,
    workers: int = 1,
    simulation_runner: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    executor: FullResetExecutor = run_full_reset_candidate_jobs,
    job_metadata: Sequence[Mapping[str, Any]] | None = None,
    candidate_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Full-rerun candidates; only explicit ``full_success`` may advance."""

    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if simulation_runner is not None and workers != 1:
        raise ValueError(
            "an injected simulation_runner is local-only; use an injected "
            "spawn-safe executor to test multiple workers"
        )
    if model is None:
        model, _ = build_model(copy.deepcopy(dict(base_config)))
    if job_metadata is not None and len(job_metadata) != len(deltas):
        raise ValueError("job_metadata must contain one mapping per delta")
    if candidate_ids is not None:
        if len(candidate_ids) != len(deltas):
            raise ValueError("candidate_ids must contain one ID per delta")
        normalized_ids = [int(value) for value in candidate_ids]
        if len(set(normalized_ids)) != len(normalized_ids) or any(
            value < 0 for value in normalized_ids
        ):
            raise ValueError("candidate_ids must be unique non-negative integers")
    else:
        normalized_ids = list(range(len(deltas)))
    jobs = []
    for candidate_index, delta in enumerate(deltas):
        candidate = _materialize_with_model(base_config, delta, model)
        jobs.append(
            {
                "candidate_id": int(normalized_ids[candidate_index]),
                "candidate_index": int(candidate_index),
                "config": candidate,
                "job_metadata": (
                    {}
                    if job_metadata is None
                    else copy.deepcopy(dict(job_metadata[candidate_index]))
                ),
            }
        )
    if simulation_runner is not None:
        executed = tuple(
            {
                "candidate_id": int(job["candidate_id"]),
                "summary": copy.deepcopy(dict(simulation_runner(job["config"]))),
            }
            for job in jobs
        )
    else:
        executed = tuple(executor(tuple(jobs), workers))
    expected = {int(job["candidate_id"]): job for job in jobs}
    observed = [int(record.get("candidate_id", -1)) for record in executed]
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise RuntimeError("full-reset executor did not preserve candidate IDs")
    records = []
    for raw in sorted(executed, key=lambda value: int(value["candidate_id"])):
        candidate_id = int(raw["candidate_id"])
        candidate_index = int(expected[candidate_id]["candidate_index"])
        candidate = expected[candidate_id]["config"]
        summary = copy.deepcopy(dict(raw["summary"]))
        stage = summary.get("stage_status", {})
        full_success = bool(
            isinstance(stage, Mapping) and stage.get("full_success") is True
        )
        records.append(
            {
                "candidate_index": int(candidate_index),
                "candidate_id": int(candidate_id),
                "manipulation_delta_rad": copy.deepcopy(
                    candidate["control"]["manipulation_delta_rad"]
                ),
                "full_reset_rerun": True,
                "initial_state_source": "configured_no_contact_reset",
                "checkpoint_used": False,
                "full_success": full_success,
                "advanced": full_success,
                "summary": summary,
                "config": candidate,
                "job_metadata": copy.deepcopy(
                    expected[candidate_id]["job_metadata"]
                ),
                "executor_metadata": {
                    key: copy.deepcopy(value)
                    for key, value in raw.items()
                    if key not in {"candidate_id", "summary"}
                },
            }
        )
    return {
        "candidate_count": len(records),
        "full_success_count": sum(record["full_success"] for record in records),
        "candidates": records,
        "advanced_candidates": [
            record for record in records if record["advanced"]
        ],
    }


def pending_local_refinement_candidates(
    generated: Sequence[Mapping[str, Any]],
    completed: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate resumable result IDs and return the deterministic remainder."""

    generated_by_id = {int(value["candidate_id"]): value for value in generated}
    if len(generated_by_id) != len(generated):
        raise ValueError("generated refinement candidate IDs must be unique")
    completed_ids = [int(value["candidate_id"]) for value in completed]
    if len(set(completed_ids)) != len(completed_ids):
        raise ValueError("completed refinement candidate IDs must be unique")
    unknown = set(completed_ids) - set(generated_by_id)
    if unknown:
        raise ValueError(f"completed refinement contains unknown IDs: {sorted(unknown)}")
    return tuple(
        copy.deepcopy(dict(value))
        for value in generated
        if int(value["candidate_id"]) not in set(completed_ids)
    )


def run_local_refinement_batch(
    base_config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    model: mujoco.MjModel | None = None,
    workers: int = 1,
    executor: FullResetExecutor = run_full_reset_candidate_jobs,
) -> dict[str, Any]:
    """Execute one resumable local-refinement batch through full resets."""

    if not candidates:
        return {
            "candidate_count": 0,
            "full_success_count": 0,
            "candidates": [],
            "advanced_candidates": [],
            "ranked_candidates": [],
        }
    result = run_full_reset_candidates(
        base_config,
        [value["manipulation_delta_rad"] for value in candidates],
        model=model,
        workers=workers,
        executor=executor,
        candidate_ids=[int(value["candidate_id"]) for value in candidates],
        job_metadata=[
            {
                "stage": "manipulation_local_refinement",
                "parent_candidate_id": int(value["parent_candidate_id"]),
                "parent_rank": int(value["parent_rank"]),
                "local_index": int(value["local_index"]),
            }
            for value in candidates
        ],
    )
    ranked = rank_manipulation_candidates(result["candidates"], config=base_config)
    return {
        **result,
        "ranked_candidates": list(ranked),
    }


def run_checkpoint_guided_manipulation(
    config: Mapping[str, Any],
    trace_or_path: Mapping[str, Any] | str | Path,
    result: Mapping[str, Any],
    *,
    budget: ManipulationSearchBudget = ManipulationSearchBudget(),
    workers: int = 1,
    session_factory: Callable[[dict[str, Any]], SimulationSession] = SimulationSession,
    probe_runner: Callable[..., Mapping[str, Any]] = run_checkpoint_probe,
    simulation_runner: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    full_reset_executor: FullResetExecutor = run_full_reset_candidate_jobs,
    full_reset_job_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute checkpoint probes, fit response, then full-reset all candidates."""

    grasp = prepare_grasp_checkpoint(
        config, trace_or_path, result, session_factory=session_factory
    )
    probes = run_checkpoint_probe_set(
        grasp, budget=budget, probe_runner=probe_runner
    )
    bounds = manipulation_delta_bounds(grasp.model, config)
    target = (0.0, 0.0, budget.target_upward_m, 0.0, 0.0, 0.0)
    response_model = fit_response_jacobian(
        probes,
        bounds,
        config=config,
        target_response_6d=target,
        ridge=budget.ridge,
        inward_preload_weight=budget.inward_preload_weight,
    )
    deltas = generate_trust_region_deltas(
        response_model,
        bounds,
        count=budget.trust_candidate_count,
        seed=budget.seed,
        trust_radius_fraction=budget.trust_radius_fraction,
        wide_candidate_fraction=budget.wide_candidate_fraction,
        wide_radius_fraction=budget.wide_radius_fraction,
    )
    reruns = run_full_reset_candidates(
        config,
        deltas,
        model=grasp.model,
        workers=workers,
        simulation_runner=simulation_runner,
        executor=full_reset_executor,
        job_metadata=full_reset_job_metadata,
    )
    return {
        "experiment_id": str(config["experiment_id"]),
        "budget": asdict(budget),
        "grasp_lock_step": int(grasp.grasp_lock_step),
        "actual_grasp_qpos_rad": grasp.actual_grasp_qpos_rad.tolist(),
        "lock_sample_joint_qpos_rad": (
            grasp.lock_sample_joint_qpos_rad.tolist()
        ),
        "probe_count": len(probes),
        "probes": list(probes),
        "response_model": response_model,
        "full_reruns": reruns,
    }


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_TARGET_RESPONSE_6D",
    "EXPERIMENT_ID",
    "FullResetExecutor",
    "GraspPhysicsCheckpoint",
    "LocalRefinementBudget",
    "ManipulationSearchBudget",
    "ProbeSpecification",
    "acquire_grasp_checkpoint",
    "clip_manipulation_delta",
    "fit_response_jacobian",
    "generate_probe_specifications",
    "generate_local_refinement_candidates",
    "generate_multiconfig_local_refinement_candidates",
    "generate_trust_region_deltas",
    "manipulation_delta_bounds",
    "manipulation_candidate_rank",
    "manipulation_candidate_rank_evidence",
    "materialize_manipulation_config",
    "prepare_grasp_checkpoint",
    "pending_local_refinement_candidates",
    "rank_manipulation_candidates",
    "run_checkpoint_guided_manipulation",
    "run_checkpoint_probe",
    "run_checkpoint_probe_set",
    "run_full_reset_candidate_jobs",
    "run_full_reset_candidates",
    "run_local_refinement_batch",
    "validate_grasp_success_source",
]
