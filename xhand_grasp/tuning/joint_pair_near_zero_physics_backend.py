"""Production MuJoCo backend for the schema-v15 near-zero campaign.

The campaign orchestrator deliberately owns only scheduling, deterministic job
identities and crash-safe stage ledgers.  This module is the corresponding
physics boundary.  Every dynamic record comes from :class:`SimulationSession`;
checkpoint rollouts are used only to construct a plan and never as success
evidence.

The implementation has two useful properties for long searches:

* candidate simulation directories use the schema-v15 atomic artifact writer;
* static DLS and sequential-planning parents compile MuJoCo once per worker
  chunk/parent, while records are returned in the scheduler's canonical order.

Nothing in this module changes the schema-v14 numerical path.
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing as mp
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..artifacts import file_sha256, json_compatible, write_json
from ..config import (
    ACTIVE_ACTUATORS,
    ACTIVE_FINGERS,
    resolved_pose_constraint_values,
    validate_config,
)
from ..experiment import (
    ContactForceTargets,
    JointPairFeedbackParameters,
    ManipulationPlanParameters,
    resolve_experiment,
)
from ..grasp_pose import canonical_sha256
from ..joint_pair_geometry import (
    measure_oriented_joint_pair_geometry,
    resolve_joint_pair,
)
from ..scene import (
    build_model,
    rpy_degrees_to_quaternion,
    rpy_degrees_to_rotation_matrix,
)
from ..simulation import _active_finger_self_collision_snapshot
from ..v15_identity import install_v15_top_level_identities
from .actual_contact_grasp_pose import (
    _cube_world_position,
    apply_precontact_solution,
    evaluate_direct_actual_contact_pose,
)
from .actual_contact_manipulation import (
    acquire_grasp_checkpoint,
    manipulation_delta_bounds,
)
from .contact_constrained_planner import ContactConstrainedPlannerSettings
from .contact_point_targeted_search import (
    ContactPointPlan,
    ContactPointSearchPolicy,
    PointTargetDLSSettings,
    PointTargetVariables,
    assert_frozen_contact_point_plan,
    materialize_point_target_candidate,
    model_active_joint_bounds,
    point_target_boundary_violations,
    point_target_static_acceptance,
    point_target_trial_evaluation,
    project_point_target_variables,
    solve_point_target_dls,
)
from .joint_pair_constrained_planner import (
    JointPairPlannerSettings,
    materialize_all_joint_pair_constrained_plan_configs,
    plan_joint_pair_constrained_sequential_trajectory,
)
from .joint_pair_near_zero_campaign import (
    EXPERIMENT_ID,
    SEED,
    GraspControlVariant,
)
from .joint_pair_near_zero_campaign_runner import (
    V15CampaignBackend,
    V15CampaignJob,
)
from .joint_pair_near_zero_candidate_artifacts import (
    V15CandidateArtifactBundle,
    run_or_resume_v15_candidate_artifacts,
)
from .joint_pair_near_zero_pose_search import (
    materialize_grasp_control_variant,
)


PHYSICS_BACKEND_SCHEMA_VERSION = 1
_EPSILON = 1e-12
_FINGER_CLOSE_GROUPS = {
    "thumb": ACTIVE_ACTUATORS[:3],
    "index": ACTIVE_ACTUATORS[3:6],
    "mid": ACTIVE_ACTUATORS[6:],
}
_PHYSICAL_PROBE_WAYPOINT_DELTA_RAD = np.asarray(
    (-0.00134, 0.00048, -0.01038, -0.00572, -0.01161, 0.00744, -0.00977, 0.00961),
    dtype=np.float64,
)
_PHYSICAL_PROBE_WAYPOINT_SCALES = (0.75, 1.0, 1.15)
_PHYSICAL_PROBE_THUMB_ROTA1_PRELOAD_RAD = (-0.003, -0.004, -0.005)
_PHYSICAL_PROBE_MID_JOINT1_PRELOAD_RAD = (0.002, 0.003, 0.004)
_PHYSICAL_PROBE_STRUCTURED_BRANCH_COUNT = (
    len(_PHYSICAL_PROBE_WAYPOINT_SCALES)
    * len(_PHYSICAL_PROBE_THUMB_ROTA1_PRELOAD_RAD)
    * len(_PHYSICAL_PROBE_MID_JOINT1_PRELOAD_RAD)
)


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON mapping: {path}")
    return value


def _candidate_directory(workspace: Path, stage: str, candidate_id: int) -> Path:
    return workspace / "physics" / stage / f"candidate_{int(candidate_id)}"


def _install_job_metadata(
    config: Mapping[str, Any],
    job: V15CampaignJob,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    metadata = resolved.setdefault("candidate_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("candidate_metadata must be a mapping")
    metadata["candidate_id"] = int(job.candidate_id)
    metadata["v15_campaign_job"] = {
        "schema_version": PHYSICS_BACKEND_SCHEMA_VERSION,
        "stage": str(job.stage),
        "index": int(job.index),
        "parent_candidate_id": job.parent_candidate_id,
        "job_sha256": canonical_sha256(job.descriptor()),
        "full_reset_success_evidence": job.stage
        in {
            "dynamic_grasp",
            "feedback_grid",
            "feedback_refinement",
            "exact_rerun",
            "local_perturbation",
            "robustness",
        },
        **copy.deepcopy(dict(evidence or {})),
    }
    install_v15_top_level_identities(resolved)
    validate_config(resolved)
    return resolved


def _static_candidate_rejection_diagnostics(
    attempted_config: Mapping[str, Any] | None,
    seed_config: Mapping[str, Any],
    *,
    phase: str,
    error: BaseException,
) -> dict[str, Any]:
    """Describe a rejected static proposal without persisting it as config.

    A DLS proposal may be numerically finite yet leave a registered pose
    envelope (the finger-down tilt boundary is a common example).  Such a
    proposal is useful negative search evidence, but an invalid
    ``resolved_config.json`` would make the atomic candidate impossible to
    authenticate or resume.  Only compact diagnostics and the proposal hash
    are retained; the artifact itself is based on the validated campaign seed.
    """

    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "phase": str(phase),
        "error_type": type(error).__name__,
        "error": str(error),
        "attempted_config_persisted": False,
        "fallback_config": "validated_seed_config",
        "seed_config_semantic_sha256": canonical_sha256(seed_config),
    }
    if attempted_config is None:
        diagnostics["attempted_config_available"] = False
        return diagnostics
    diagnostics["attempted_config_available"] = True
    try:
        diagnostics["attempted_config_semantic_sha256"] = canonical_sha256(
            attempted_config
        )
    except (TypeError, ValueError):
        # A malformed proposal still needs a resumable rejection artifact.
        diagnostics["attempted_config_semantic_sha256"] = None
    hand_pose = attempted_config.get("hand_pose")
    if isinstance(hand_pose, Mapping):
        diagnostics["attempted_hand_pose"] = copy.deepcopy(dict(hand_pose))
    grasp_pose = attempted_config.get("grasp_pose")
    if isinstance(grasp_pose, Mapping):
        nominal = grasp_pose.get("nominal_joint_qpos_rad")
        if isinstance(nominal, Mapping):
            diagnostics["attempted_nominal_joint_qpos_rad"] = {
                name: float(nominal[name])
                for name in ACTIVE_ACTUATORS
                if name in nominal
            }
    try:
        diagnostics["attempted_resolved_pose_constraints"] = json_compatible(
            resolved_pose_constraint_values(dict(attempted_config))
        )
    except (KeyError, TypeError, ValueError, ArithmeticError) as pose_error:
        diagnostics["attempted_resolved_pose_constraints_error"] = (
            f"{type(pose_error).__name__}: {pose_error}"
        )
    return diagnostics


def _record_from_bundle(
    job: V15CampaignJob,
    bundle: V15CandidateArtifactBundle,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = bundle.result
    summary = copy.deepcopy(dict(result["summary"]))
    return {
        "candidate_id": int(job.candidate_id),
        "parent_candidate_id": job.parent_candidate_id,
        "stage": str(job.stage),
        "grasp_success": bool(result["grasp_success"]),
        "full_success": bool(result["full_success"]),
        "static_pass": bool(result["grasp_success"]),
        "summary": summary,
        "config_path": str(bundle.config_path),
        "result_path": str(bundle.result_path),
        "trace_path": str(bundle.trace_path) if bundle.trace_path else None,
        "artifact_reused": bool(bundle.reused),
        **copy.deepcopy(dict(extra or {})),
    }


def _simulation_task(
    task: tuple[dict[str, Any], str, int, bool, bool, dict[str, Any]],
) -> dict[str, Any]:
    (
        config,
        destination,
        candidate_id,
        final_rerun,
        retain_grasp_trace,
        descriptor,
    ) = task
    bundle = run_or_resume_v15_candidate_artifacts(
        config,
        destination,
        candidate_id,
        final_rerun=final_rerun,
        retain_grasp_trace=retain_grasp_trace,
    )
    # Recreate the immutable job shell in the worker only for canonical output.
    job = V15CampaignJob(
        stage=str(descriptor["stage"]),
        index=int(descriptor["index"]),
        parent_candidate_id=(
            None
            if descriptor["parent_candidate_id"] is None
            else int(descriptor["parent_candidate_id"])
        ),
        payload=copy.deepcopy(dict(descriptor["payload"])),
        candidate_id=int(candidate_id),
    )
    return _record_from_bundle(job, bundle)


def _run_simulation_tasks(
    tasks: Sequence[
        tuple[dict[str, Any], str, int, bool, bool, dict[str, Any]]
    ],
    *,
    workers: int,
) -> tuple[dict[str, Any], ...]:
    if not tasks:
        return ()
    if workers <= 1:
        return tuple(_simulation_task(task) for task in tasks)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        # executor.map preserves input order, so worker completion timing never
        # changes the stage report or the downstream candidate schedule.
        return tuple(pool.map(_simulation_task, tasks, chunksize=1))


def _atomic_metadata_directory(
    destination: Path,
    *,
    input_sha256: str,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[Path, Path, bool]:
    """Write or authenticate a non-simulation candidate atomically."""

    destination = destination.resolve()
    config_path = destination / "resolved_config.json"
    result_path = destination / "result.json"
    if destination.exists():
        persisted = _load_json(result_path)
        if (
            persisted.get("complete") is not True
            or persisted.get("input_sha256") != input_sha256
            or persisted.get("config_semantic_sha256")
            != canonical_sha256(config)
            or file_sha256(config_path)
            != persisted.get("artifacts", {}).get("resolved_config_sha256")
        ):
            raise RuntimeError(f"changed or incomplete v15 metadata artifact: {destination}")
        return config_path, result_path, True
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging.", dir=destination.parent
        )
    )
    committed = False
    try:
        staged_config = staging / config_path.name
        staged_result = staging / result_path.name
        write_json(staged_config, config)
        payload = {
            **copy.deepcopy(dict(result)),
            "complete": True,
            "input_sha256": input_sha256,
            "config_semantic_sha256": canonical_sha256(config),
            "artifacts": {
                "resolved_config": staged_config.name,
                "resolved_config_sha256": file_sha256(staged_config),
            },
        }
        write_json(staged_result, payload)
        staging.rename(destination)
        committed = True
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)
    return config_path, result_path, False


def _static_policy(seed_config: Mapping[str, Any], count: int, seed: int) -> ContactPointSearchPolicy:
    plan = assert_frozen_contact_point_plan(seed_config)
    root = np.asarray(seed_config["hand_pose"]["translation_m"], dtype=np.float64)
    # The cube centre helper is intentionally not duplicated here; the root
    # distance envelope only protects the local solver and can conservatively
    # include every registered +/-1.5 mm proposal.
    distance = float(np.linalg.norm(root - _cube_world_position(seed_config)))
    points = {
        finger: plan.points[finger] for finger in ACTIVE_FINGERS
    }
    return ContactPointSearchPolicy(
        seed=int(seed),
        sample_count=max(1, int(count)),
        retain_point_plan_count=1,
        reference_points=points,
        half_width_m=0.008,
        minimum_edge_margin_m=0.0005,
        maximum_height_spread_m=0.005,
        minimum_index_middle_separation_m=0.010,
        static_target_radius_m=0.004,
        target_radius_m=float(plan.target_radius_m),
        minimum_normal_alignment=0.95,
        maximum_penetration_m=0.002,
        signed_orbit_deg=(0.0,),
        root_delta_cube_m={axis: (-0.0015, 0.0015) for axis in "xyz"},
        wrist_local_rotvec_deg={axis: (-1.5, 1.5) for axis in "xyz"},
        max_wrist_local_rotvec_norm_deg=2.0,
        root_cube_distance_m=(max(0.001, distance - 0.010), distance + 0.010),
        thumb_actual_range_rad=(1.40, 1.60),
    )


def _point_target_evaluator(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: Any,
    plan: ContactPointPlan,
):
    def evaluate(config: Mapping[str, Any]):
        result = evaluate_direct_actual_contact_pose(model, data, info, config)
        # The v15 dynamic loop has stricter forbidden/self-collision gates.
        # Static DLS needs the genuine witness geometry, but the historical
        # v11-only extra proof fields must not make every schema-v15 finite
        # difference unusable.
        return point_target_trial_evaluation(
            result,
            config,
            plan,
            minimum_normal_alignment=0.95,
            maximum_penetration_m=0.002,
            require_extended_safety_evidence=False,
        )

    return evaluate


def _static_pair_and_self_collision(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model.body_pos[info.root_body_id] = np.asarray(
        config["hand_pose"]["translation_m"], dtype=np.float64
    )
    model.body_quat[info.root_body_id] = rpy_degrees_to_quaternion(
        config["hand_pose"]["rpy_deg"]
    )
    mujoco.mj_resetData(model, data)
    active_ids = np.asarray(
        [model.actuator(name).id for name in ACTIVE_ACTUATORS], dtype=np.int64
    )
    values = np.asarray(
        [
            config["grasp_pose"]["nominal_joint_qpos_rad"][name]
            for name in ACTIVE_ACTUATORS
        ],
        dtype=np.float64,
    )
    data.qpos[np.asarray(info.actuator_qpos_adrs)[active_ids]] = values
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    binding = resolve_joint_pair(
        model, config["joint_pair_alignment"]["joint_names"]
    )
    if binding is None:  # pragma: no cover - validated v15 always resolves
        raise RuntimeError("v15 static screen could not resolve the joint pair")
    try:
        geometry = measure_oriented_joint_pair_geometry(
            data.xanchor[binding.first_joint_id],
            data.xanchor[binding.second_joint_id],
            np.asarray(data.xmat[binding.cube_body_id]).reshape(3, 3),
            minimum_separation_m=float(
                config["joint_pair_alignment"]["minimum_length_m"]
            ),
        )
        pair = {
            "valid": True,
            "vector_cube_m": np.asarray(geometry["vector_cube_m"]).tolist(),
            "signed_residual": np.asarray(
                geometry["joint_pair_signed_residual"]
            ).tolist(),
            "length_m": float(geometry["length_m"]),
            "angle_deg": float(geometry["angle_to_cube_positive_y_deg"]),
            "positive_y": bool(
                np.asarray(geometry["vector_cube_m"])[1] > 0.0
            ),
        }
    except ValueError as error:
        pair = {
            "valid": False,
            "vector_cube_m": [0.0, 0.0, 0.0],
            "signed_residual": [0.0, 0.0],
            "length_m": 0.0,
            "angle_deg": 180.0,
            "positive_y": False,
            "error": str(error),
        }
    collision = _active_finger_self_collision_snapshot(model, data, info)
    return {"joint_pair": pair, "active_finger_self_collision": collision}


def _pair_aware_secondary_dls(
    base_config: Mapping[str, Any],
    initial_variables: PointTargetVariables,
    *,
    policy: ContactPointSearchPolicy,
    settings: PointTargetDLSSettings,
    evaluator: Any,
    joint_bounds: Mapping[str, tuple[float, float]],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: Any,
    maximum_iterations: int = 4,
) -> tuple[
    dict[str, Any],
    PointTargetVariables,
    Any,
    Any,
    Any,
    dict[str, Any],
]:
    """Refine all fourteen variables with contact *and* pair residuals.

    The reusable point-target solver owns the first twelve collision-witness
    residuals.  This additive v15 pass appends the directed
    ``(v_x/v_y, v_z/v_y)`` residual and evaluates every finite difference with
    the same real MuJoCo geometry.  It therefore avoids changing the frozen
    v12/v13 numerical path while satisfying the v15 combined-DLS contract.
    """

    current = initial_variables.as_array()
    reference = current.copy()
    finite_difference = np.asarray(
        [settings.joint_finite_difference_rad] * 8
        + [settings.translation_finite_difference_m] * 3
        + [settings.rotation_finite_difference_rad] * 3,
        dtype=np.float64,
    )
    variable_scale = np.asarray(
        [settings.joint_step_rad] * 8
        + [settings.translation_step_m] * 3
        + [settings.rotation_step_rad] * 3,
        dtype=np.float64,
    )
    pair_scale = math.tan(math.radians(0.25))
    target = np.asarray(
        (0.00015,) * 3 + (0.0,) * 6 + (1.0,) * 3 + (0.0, 0.0),
        dtype=np.float64,
    )
    scale = np.asarray(
        (0.0005,) * 3
        + (policy.static_target_radius_m,) * 6
        + (0.05,) * 3
        + (pair_scale, pair_scale),
        dtype=np.float64,
    )

    def boundary(value: np.ndarray) -> tuple[str, ...]:
        return point_target_boundary_violations(
            base_config,
            PointTargetVariables.from_array(value),
            policy,
            signed_orbit_deg=0.0,
            joint_bounds=joint_bounds,
            check_pose_constraints=True,
        )

    def evaluate(value: np.ndarray):
        candidate = materialize_point_target_candidate(
            base_config,
            PointTargetVariables.from_array(value),
            signed_orbit_deg=0.0,
            synchronize_preload=True,
        )
        point = evaluator(candidate)
        if not point.safe or point.measurement is None:
            return None
        physical = _static_pair_and_self_collision(
            model, data, info, candidate
        )
        pair = physical["joint_pair"]
        collision = physical["active_finger_self_collision"]
        if not pair["valid"] or not pair["positive_y"] or collision["active"]:
            return None
        measurement = np.concatenate(
            (
                np.asarray(point.measurement, dtype=np.float64),
                np.asarray(pair["signed_residual"], dtype=np.float64),
            )
        )
        return candidate, point, physical, measurement

    observed = evaluate(current)
    if observed is None:
        raise ValueError("pair-aware DLS start has no safe combined measurement")
    current_config, current_point, current_physical, measurement = observed
    iterations: list[dict[str, Any]] = []

    def objective(value: np.ndarray, variables: np.ndarray) -> float:
        contact_pair = np.linalg.norm((target - value) / scale)
        regularization = math.sqrt(settings.regularization_weight) * np.linalg.norm(
            (variables - reference) / variable_scale
        )
        return float(math.hypot(contact_pair, regularization))

    for iteration in range(int(maximum_iterations)):
        before = objective(measurement, current)
        jacobian = np.zeros((len(target), len(current)), dtype=np.float64)
        available = np.zeros(len(current), dtype=bool)
        for column in range(len(current)):
            for direction in (1.0, -1.0):
                trial = current.copy()
                trial[column] += direction * finite_difference[column]
                if boundary(trial):
                    continue
                trial_observed = evaluate(trial)
                if trial_observed is None:
                    continue
                trial_measurement = trial_observed[3]
                jacobian[:, column] = (
                    (trial_measurement - measurement)
                    / (direction * finite_difference[column])
                    / scale
                )
                available[column] = True
                break
        normalized = (target - measurement) / scale
        scaled = jacobian * variable_scale[np.newaxis, :]
        weight = math.sqrt(settings.regularization_weight)
        system = np.vstack((scaled, weight * np.eye(len(current))))
        rhs = np.concatenate(
            (normalized, -weight * (current - reference) / variable_scale)
        )
        normal = system.T @ system + settings.damping**2 * np.eye(len(current))
        unit_step = np.linalg.solve(normal, system.T @ rhs)
        unit_step[~available] = 0.0
        maximum = float(np.max(np.abs(unit_step)))
        if maximum > 1.0:
            unit_step /= maximum
        raw_step = variable_scale * unit_step
        accepted = None
        for line_scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = project_point_target_variables(
                current + line_scale * raw_step, policy, joint_bounds
            )
            if boundary(trial):
                continue
            trial_observed = evaluate(trial)
            if trial_observed is None:
                continue
            after = objective(trial_observed[3], trial)
            if after + 1e-12 < before:
                accepted = (trial, trial_observed, after, line_scale)
                break
        iterations.append(
            {
                "iteration": iteration,
                "objective_before": before,
                "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
                "available_variable_count": int(np.count_nonzero(available)),
                "accepted": accepted is not None,
                "accepted_scale": None if accepted is None else accepted[3],
                "objective_after": None if accepted is None else accepted[2],
            }
        )
        if accepted is None:
            break
        current, observed, _, _ = accepted
        current_config, current_point, current_physical, measurement = observed

    acceptance = point_target_static_acceptance(current_point, policy)
    if acceptance.passed:
        current_config = apply_precontact_solution(
            current_config, current_point.static_result
        )
    diagnostics = {
        "method": "v15_contact_and_directed_joint_pair_14_variable_dls",
        "measurement_dimension": 14,
        "pair_residual_definition": "vx_over_vy_vz_over_vy",
        "pair_residual_scale": [pair_scale, pair_scale],
        "initial_variables": reference.tolist(),
        "final_variables": current.tolist(),
        "final_measurement": measurement.tolist(),
        "final_pair_angle_deg": float(
            current_physical["joint_pair"]["angle_deg"]
        ),
        "iterations": iterations,
    }
    return (
        current_config,
        PointTargetVariables.from_array(current),
        current_point.static_result,
        current_point,
        acceptance,
        diagnostics,
    )


def _static_stage(
    jobs: Sequence[V15CampaignJob],
    workspace: Path,
    context: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    if not jobs:
        return ()
    seed_config = _load_json(context["seed_config_path"])
    # The recovery path below intentionally persists this configuration when
    # a DLS proposal violates a registered envelope.  Authenticate that
    # fallback before doing any physics work so rejection artifacts can never
    # conceal a damaged campaign seed.
    validate_config(seed_config)
    plan = assert_frozen_contact_point_plan(seed_config)
    policy = _static_policy(seed_config, len(jobs), seed)
    model, info = build_model(copy.deepcopy(seed_config))
    data = mujoco.MjData(model)
    evaluator = _point_target_evaluator(model, data, info, plan)
    bounds = model_active_joint_bounds(model, seed_config)
    nominal = np.asarray(
        [
            seed_config["grasp_pose"]["nominal_joint_qpos_rad"][name]
            for name in ACTIVE_ACTUATORS
        ],
        dtype=np.float64,
    )
    settings = PointTargetDLSSettings(maximum_iterations=12)
    records: list[dict[str, Any]] = []
    for job in jobs:
        offsets = np.asarray(job.payload["joint_qpos_offset_rad"], dtype=np.float64)
        variables = PointTargetVariables(
            actual_joint_qpos_rad=tuple(nominal + offsets),
            root_delta_cube_m=tuple(job.payload["root_delta_cube_m"]),
            wrist_local_rotvec_rad=tuple(
                np.radians(job.payload["wrist_local_rotvec_deg"])
            ),
        )
        error: str | None = None
        rejection_diagnostics: list[dict[str, Any]] = []
        try:
            # Refine the zero perturbation as well.  The DLS regularizer keeps
            # it close to the authenticated measured pose, while the dynamic
            # ``original`` branch later restores the source preload residual.
            # This has been verified by a full-reset grasp run and gives the
            # planner a materially cleaner near-zero alignment seed.
            solved = solve_point_target_dls(
                seed_config,
                signed_orbit_deg=0.0,
                policy=policy,
                initial_variables=variables,
                settings=settings,
                evaluator=evaluator,
                joint_bounds=bounds,
                check_pose_constraints=True,
            )
            (
                resolved,
                _refined_variables,
                static_result,
                static_evaluation,
                static_acceptance,
                pair_diagnostics,
            ) = _pair_aware_secondary_dls(
                seed_config,
                solved.variables,
                policy=policy,
                settings=settings,
                evaluator=evaluator,
                joint_bounds=bounds,
                model=model,
                data=data,
                info=info,
            )
            point_distances = static_evaluation.point_distance_m or ()
            point_errors = static_evaluation.point_error_yz_m or ()
            stop_reason = solved.stop_reason
            dls_diagnostics = {
                "contact_point_dls": solved.diagnostics,
                "joint_pair_aware_dls": pair_diagnostics,
            }
            physical = _static_pair_and_self_collision(
                model, data, info, resolved
            )
            pair = physical["joint_pair"]
            collision = physical["active_finger_self_collision"]
            static_pass = bool(
                static_acceptance.passed
                and pair["valid"]
                and pair["positive_y"]
                and pair["length_m"]
                >= float(resolved["joint_pair_alignment"]["minimum_length_m"])
                - _EPSILON
                and pair["angle_deg"]
                <= float(resolved["joint_pair_alignment"]["grasp_max_deg"])
                + _EPSILON
                and not collision["active"]
            )
            static_metrics = static_result.as_dict()
            static_metrics["point_target"] = {
                "point_plan_id": plan.point_plan_id,
                "point_distance_m": list(point_distances),
                "point_error_yz_m": [
                    list(value) for value in point_errors
                ],
                "acceptance": static_acceptance.as_dict(),
                "stop_reason": stop_reason,
                "dls_diagnostics_sha256": canonical_sha256(
                    dls_diagnostics
                ),
            }
            static_metrics.update(physical)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as caught:
            # Boundary/geometry rejection is a legitimate static near miss.
            # It is persisted explicitly and never promoted as physics success.
            # Materialising the raw proposal is best-effort only: if it lies
            # outside a registered pose envelope, the validation guard below
            # replaces it with the authenticated seed before writing anything.
            error = f"{type(caught).__name__}: {caught}"
            try:
                resolved = materialize_point_target_candidate(
                    seed_config,
                    variables,
                    signed_orbit_deg=0.0,
                    synchronize_preload=False,
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as materialize_error:
                rejection_diagnostics.append(
                    _static_candidate_rejection_diagnostics(
                        None,
                        seed_config,
                        phase="static_candidate_materialization",
                        error=materialize_error,
                    )
                )
                resolved = copy.deepcopy(seed_config)
            static_pass = False
            static_metrics = {
                "point_target": {"stop_reason": "static_solver_rejected"},
                "joint_pair": {
                    "valid": False,
                    "angle_deg": 180.0,
                    "length_m": 0.0,
                    "positive_y": False,
                },
                "active_finger_self_collision": {"active": False},
            }
            dls_diagnostics = {"error": error}
        evidence: dict[str, Any] = {
            "static_filter_is_success_evidence": False,
            "real_collision_witness_screen": True,
            "orientation_aware_14_variable_dls": True,
            "directed_joint_pair_residual_in_dls": job.index != 0,
            "dls_maximum_iterations": 12,
        }
        if rejection_diagnostics:
            evidence["static_candidate_rejection_diagnostics"] = copy.deepcopy(
                rejection_diagnostics
            )
        try:
            resolved = _install_job_metadata(
                resolved,
                job,
                evidence=evidence,
            )
        except ValueError as validation_error:
            # Never persist an invalid DLS proposal.  It remains explicit
            # negative evidence, while the artifact config is a validated seed
            # plus immutable job/rejection metadata.  This makes an interrupted
            # formal workspace safely resumable at the rejected candidate.
            rejection = _static_candidate_rejection_diagnostics(
                resolved,
                seed_config,
                phase="resolved_static_candidate_validation",
                error=validation_error,
            )
            rejection_diagnostics.append(rejection)
            evidence["static_candidate_rejection_diagnostics"] = copy.deepcopy(
                rejection_diagnostics
            )
            static_pass = False
            validation_message = (
                f"{type(validation_error).__name__}: {validation_error}"
            )
            error = (
                validation_message
                if error is None
                else f"{error}; resolved_config_rejection={validation_message}"
            )
            point_target_metrics = static_metrics.setdefault("point_target", {})
            point_target_metrics["stop_reason"] = "invalid_resolved_static_candidate"
            point_target_metrics["resolved_config_rejection"] = copy.deepcopy(
                rejection
            )
            dls_diagnostics = {
                **copy.deepcopy(dict(dls_diagnostics)),
                "resolved_config_rejection": copy.deepcopy(rejection),
            }
            resolved = _install_job_metadata(
                seed_config,
                job,
                evidence=evidence,
            )
        pair_angle = float(
            static_metrics.get("joint_pair", {}).get("angle_deg", 180.0)
        )
        summary = {
            "passed": False,
            "stage_status": {
                "grasp_success": False,
                "manipulation_success": False,
                "full_success": False,
                "failure_reason": None if static_pass else "static_filter_failed",
            },
            "metrics": {
                "static_filter": static_metrics,
                # The campaign's deterministic rank consumes this common key.
                "joint_pair_alignment": {
                    "operation_angle_max_deg": pair_angle,
                    "operation_angle_p95_deg": pair_angle,
                    "operation_within_limit_duty": float(static_pass),
                    "operation_longest_violation_s": 0.0 if static_pass else 1.0,
                },
            },
        }
        result_payload = {
            "joint_pair_near_zero_static_result_schema_version": 1,
            "candidate_id": int(job.candidate_id),
            "parent_candidate_id": None,
            "stage": job.stage,
            "static_pass": static_pass,
            "error": error,
            "summary": summary,
            "dls_diagnostics": json_compatible(dls_diagnostics),
        }
        input_sha = canonical_sha256(
            {
                "job": job.descriptor(),
                "seed_config": canonical_sha256(seed_config),
                "backend_schema_version": PHYSICS_BACKEND_SCHEMA_VERSION,
            }
        )
        config_path, result_path, reused = _atomic_metadata_directory(
            _candidate_directory(workspace, job.stage, job.candidate_id),
            input_sha256=input_sha,
            config=resolved,
            result=result_payload,
        )
        records.append(
            {
                "candidate_id": int(job.candidate_id),
                "parent_candidate_id": None,
                "stage": job.stage,
                "static_pass": static_pass,
                "grasp_success": False,
                "full_success": False,
                "summary": summary,
                "config_path": str(config_path),
                "result_path": str(result_path),
                "trace_path": None,
                "artifact_reused": reused,
            }
        )
    return tuple(records)


def _parent_records(context: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    parents = context.get("parents", ())
    if not isinstance(parents, Sequence):
        raise ValueError("stage context parents must be a sequence")
    result: dict[int, Mapping[str, Any]] = {}
    for value in parents:
        if not isinstance(value, Mapping):
            raise ValueError("stage parent record must be a mapping")
        candidate_id = int(value["candidate_id"])
        if candidate_id in result:
            raise ValueError(f"duplicate stage parent {candidate_id}")
        result[candidate_id] = value
    return result


def _config_from_parent(parent: Mapping[str, Any]) -> dict[str, Any]:
    raw = parent.get("config_path")
    if raw is None:
        raise RuntimeError(f"parent {parent.get('candidate_id')} has no config_path")
    return _load_json(str(raw))


def _materialize_grasp_variant(
    parent: Mapping[str, Any],
    job: V15CampaignJob,
    *,
    reference_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = _config_from_parent(parent)
    if reference_config is None:
        # Production supplies the authenticated migrated seed.  Falling back
        # to the candidate itself keeps the private unit-level materializer
        # usable without weakening the formal runner's preload distinction.
        reference_config = config
    planning_warm_start = copy.deepcopy(config["manipulation_plan"])
    raw_variant = job.payload["controller_variant"]
    variant = GraspControlVariant(
        index=int(raw_variant["index"]),
        close_s=float(raw_variant["close_s"]),
        mode=str(raw_variant["mode"]),
    )
    # Keep the two declared controller families physically distinct after the
    # static DLS has moved the measured contact pose.  ``original`` transports
    # the authenticated source preload-minus-qpos residual to the new pose;
    # ``synchronized_preload`` commands the new nominal qpos directly.  Merely
    # changing the close profile would otherwise duplicate half of the formal
    # six-controller grid.
    config = materialize_grasp_control_variant(
        config,
        variant,
        reference_config=reference_config,
    )
    # Dynamic acquisition is evaluated independently from manipulation.  A
    # zero plan prevents an inherited near-miss endpoint from destabilising a
    # valid grasp after it has already been measured, while the later planner
    # always replaces all 21 nodes before manipulation evidence is published.
    plan = copy.deepcopy(config["manipulation_plan"])
    plan.pop("plan_id", None)
    plan["actuator_waypoints_rad"] = {
        name: [0.0] * 21 for name in ACTIVE_ACTUATORS
    }
    for field, width in (
        ("joint_pair_residual_jacobian_2x8", 2),
        ("object_response_jacobian_6x8", 6),
        ("target_force_jacobian_3x8", 3),
    ):
        plan[field] = np.zeros((21, width, len(ACTIVE_ACTUATORS))).tolist()
    plan["plan_id"] = canonical_sha256(plan)
    config["manipulation_plan"] = plan
    config["control"]["manipulation_delta_rad"] = {
        name: 0.0 for name in ACTIVE_ACTUATORS
    }
    return _install_job_metadata(
        config,
        job,
        evidence={
            "controller_variant": variant.as_mapping(),
            "dynamic_grasp_from_initial_no_contact_state": True,
            "manipulation_plan_zeroed_for_grasp_stage": True,
            "planning_warm_start_manipulation_plan": planning_warm_start,
        },
    )


def _target_force_config_from_grasp_trace(
    config: dict[str, Any], trace_path: str | Path
) -> dict[str, Any]:
    with np.load(trace_path, allow_pickle=False) as trace:
        start = int(np.asarray(trace["grasp_stable_window_start_step"]).item())
        end = int(np.asarray(trace["grasp_stable_window_end_step"]).item())
        force = np.asarray(
            trace["closure_target_contact_force_n"], dtype=np.float64
        )[start : end + 1]
        if force.shape[0] <= 0 or force.shape[1:] != (3,):
            raise RuntimeError("grasp trace has no stable three-finger force window")
        median = np.median(force, axis=0)
    target = ContactForceTargets(
        schema_version=1,
        source="verify_window_median_clamped",
        minimum_n=0.2,
        maximum_n=3.0,
        per_finger_n={
            finger: float(np.clip(median[index], 0.2, 3.0))
            for index, finger in enumerate(ACTIVE_FINGERS)
        },
    )
    resolved = copy.deepcopy(config)
    resolved["contact_force_targets_n"] = target.as_config()
    install_v15_top_level_identities(resolved)
    validate_config(resolved)
    return resolved


def _contact_settings(config: Mapping[str, Any]) -> ContactConstrainedPlannerSettings:
    targets = config["contact_force_targets_n"]["per_finger_n"]
    return ContactConstrainedPlannerSettings(
        duration_s=3.0,
        max_knot_delta_rad=0.04,
        target_object_response_6d=(0.0, 0.0, 0.011, 0.0, 0.0, 0.0),
        target_normal_force_n=tuple(
            float(targets[finger]) for finger in ACTIVE_FINGERS
        ),
        minimum_normal_force_n=(0.05, 0.05, 0.05),
        maximum_tangent_slip_m=(0.002, 0.002, 0.002),
        object_response_scale=(
            0.002,
            0.002,
            0.011,
            math.radians(0.5),
            math.radians(0.5),
            math.radians(0.5),
        ),
        # Force is enforced by the explicit >=0.05 N linear bands.  Keeping
        # it as a stiff least-squares target (the generic 0.1 N scale) caused
        # the optimizer to sacrifice essentially all vertical motion merely
        # to reproduce the VERIFY force imbalance.  The measured strong-center
        # sweep used 100 N here, leaving force as a soft balance objective
        # while preserving the hard safety floor.
        force_response_scale_n=(100.0, 100.0, 100.0),
        # The 21-knot warm start already contains the measured v14 vertical
        # response.  Real sequential probe sweeps showed that the generic
        # 1e-5 ridge collapses that seed to an almost-zero plan, whereas 1e3
        # retains about 10.85 mm predicted lift with substantially fewer
        # contact-loss segments than the more aggressive 1e4 setting.  This
        # term only anchors the optimizer; all hard force, slip, pair and
        # full-reset acceptance checks remain unchanged.
        ridge=1_000.0,
    )


def _planning_parent_task(
    task: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        str,
    ]
) -> list[dict[str, Any]]:
    parent, descriptors, destination_raw = task
    destination = Path(destination_raw).resolve()
    input_sha = canonical_sha256(
        {
            "parent": parent,
            "jobs": descriptors,
            "backend_schema_version": PHYSICS_BACKEND_SCHEMA_VERSION,
        }
    )
    complete_path = destination / "planning_report.json"
    if destination.exists():
        payload = _load_json(complete_path)
        if payload.get("complete") is not True or payload.get("input_sha256") != input_sha:
            raise RuntimeError(f"changed sequential-planning artifact: {destination}")
        return [copy.deepcopy(dict(value)) for value in payload["records"]]
    config = _config_from_parent(parent)
    trace_path = parent.get("trace_path")
    if trace_path is None:
        raise RuntimeError("measured grasp parent has no retained trace")
    job_metadata = config.get("candidate_metadata", {}).get(
        "v15_campaign_job", {}
    )
    warm_start = (
        job_metadata.get("planning_warm_start_manipulation_plan")
        if isinstance(job_metadata, Mapping)
        else None
    )
    if not isinstance(warm_start, Mapping):
        raise RuntimeError(
            "dynamic grasp parent lost its authenticated manipulation warm start"
        )
    config["manipulation_plan"] = copy.deepcopy(dict(warm_start))
    config["control"]["manipulation_delta_rad"] = {
        name: float(
            config["manipulation_plan"]["actuator_waypoints_rad"][name][-1]
        )
        for name in ACTIVE_ACTUATORS
    }
    config = _target_force_config_from_grasp_trace(config, str(trace_path))
    grasp = acquire_grasp_checkpoint(config)
    settings = _contact_settings(config)
    pair_settings = JointPairPlannerSettings.from_alignment_config(
        config["joint_pair_alignment"]
    )
    bounds = manipulation_delta_bounds(grasp.model, config)
    report = plan_joint_pair_constrained_sequential_trajectory(
        grasp,
        config["manipulation_plan"],
        bounds,
        contact_settings=settings,
        joint_pair_settings=pair_settings,
    )
    materialized = materialize_all_joint_pair_constrained_plan_configs(
        config, report, validate=False
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
    )
    committed = False
    records: list[dict[str, Any]] = []
    try:
        for descriptor, base in zip(descriptors, materialized, strict=True):
            job = V15CampaignJob(
                str(descriptor["stage"]),
                int(descriptor["index"]),
                int(descriptor["parent_candidate_id"]),
                copy.deepcopy(dict(descriptor["payload"])),
                int(descriptor["candidate_id"]),
            )
            candidate = _install_job_metadata(
                base,
                job,
                evidence={
                    "sequential_checkpoint_planning": True,
                    "checkpoint_is_search_evidence_only": True,
                    "plan_attempt_index": int(job.payload["plan_index"]),
                    "full_reset_rerun_required": True,
                },
            )
            member = staging / f"candidate_{job.candidate_id}"
            member.mkdir()
            config_path = member / "resolved_config.json"
            result_path = member / "result.json"
            write_json(config_path, candidate)
            attempt = report.attempts[int(job.payload["plan_index"])]
            result = {
                "joint_pair_near_zero_planning_result_schema_version": 1,
                "complete": True,
                "candidate_id": int(job.candidate_id),
                "parent_candidate_id": job.parent_candidate_id,
                "planning_success": bool(attempt.search_contact_and_pair_safe),
                "checkpoint_search_evidence_only": True,
                "full_reset_success_evidence": False,
                "plan_attempt": attempt.as_mapping(),
                "config_semantic_sha256": canonical_sha256(candidate),
                "artifacts": {
                    "resolved_config": config_path.name,
                    "resolved_config_sha256": file_sha256(config_path),
                },
            }
            write_json(result_path, result)
            records.append(
                {
                    "candidate_id": int(job.candidate_id),
                    "parent_candidate_id": job.parent_candidate_id,
                    "stage": job.stage,
                    "static_pass": True,
                    "grasp_success": True,
                    "full_success": False,
                    "planning_success": bool(attempt.search_contact_and_pair_safe),
                    "summary": {
                        "passed": False,
                        "stage_status": {
                            "grasp_success": True,
                            "manipulation_success": False,
                            "full_success": False,
                        },
                        "metrics": {
                            "joint_pair_alignment": {
                                "operation_angle_max_deg": float(
                                    attempt.maximum_pair_angle_deg
                                ),
                                "operation_angle_p95_deg": float(
                                    attempt.maximum_pair_angle_deg
                                ),
                                "operation_within_limit_duty": float(
                                    attempt.search_contact_and_pair_safe
                                ),
                                "operation_longest_violation_s": (
                                    0.0
                                    if attempt.search_contact_and_pair_safe
                                    else 3.0
                                ),
                            }
                        },
                    },
                    "config_path": str(
                        destination
                        / f"candidate_{job.candidate_id}"
                        / config_path.name
                    ),
                    "result_path": str(
                        destination
                        / f"candidate_{job.candidate_id}"
                        / result_path.name
                    ),
                    "trace_path": None,
                }
            )
        report_payload = {
            "joint_pair_near_zero_physical_planning_report_schema_version": 1,
            "complete": True,
            "input_sha256": input_sha,
            "parent_candidate_id": int(parent["candidate_id"]),
            "physical_probe_count_per_segment": 17,
            "segment_count": 20,
            "attempt_count": 4,
            "planner_report": report.as_mapping(),
            "records": records,
        }
        write_json(staging / complete_path.name, report_payload)
        staging.rename(destination)
        committed = True
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)
    return records


def _run_planning_tasks(
    jobs: Sequence[V15CampaignJob],
    workspace: Path,
    parents: Mapping[int, Mapping[str, Any]],
    *,
    workers: int,
) -> tuple[dict[str, Any], ...]:
    grouped: dict[int, list[V15CampaignJob]] = {}
    for job in jobs:
        assert job.parent_candidate_id is not None
        grouped.setdefault(job.parent_candidate_id, []).append(job)
    tasks = []
    for parent_id, values in grouped.items():
        if len(values) != 4 or [int(v.payload["plan_index"]) for v in values] != list(range(4)):
            raise RuntimeError("each v15 grasp must own exactly four canonical plans")
        tasks.append(
            (
                copy.deepcopy(dict(parents[parent_id])),
                [value.descriptor() for value in values],
                str(workspace / "physics" / "sequential_planning" / f"parent_{parent_id}"),
            )
        )
    if workers <= 1:
        batches = [_planning_parent_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as pool:
            batches = list(pool.map(_planning_parent_task, tasks, chunksize=1))
    by_id = {
        int(record["candidate_id"]): record
        for batch in batches
        for record in batch
    }
    return tuple(copy.deepcopy(by_id[job.candidate_id]) for job in jobs)


def _pair_feedback_mapping(
    source: Mapping[str, Any], alignment_gain: float, slip_gain: float
) -> dict[str, Any]:
    raw = copy.deepcopy(dict(source))
    raw.pop("feedback_id", None)
    raw["alignment_gain"] = float(alignment_gain)
    raw["slip_recovery_gain_rad_per_m"] = float(slip_gain)
    return JointPairFeedbackParameters(**raw).as_config()


def _materialize_feedback_variant(
    parent: Mapping[str, Any], job: V15CampaignJob
) -> dict[str, Any]:
    config = _config_from_parent(parent)
    variant = job.payload["feedback_variant"]
    config["joint_pair_feedback"] = _pair_feedback_mapping(
        config["joint_pair_feedback"],
        float(variant["alignment_gain"]),
        float(variant["slip_recovery_gain_rad_per_m"]),
    )
    return _install_job_metadata(
        config,
        job,
        evidence={
            "feedback_variant": copy.deepcopy(dict(variant)),
            "full_reset_from_initial_no_contact_state": True,
        },
    )


def _refined_config(
    parent: Mapping[str, Any], job: V15CampaignJob, *, seed: int
) -> dict[str, Any]:
    config = _config_from_parent(parent)
    fixed_state = {
        key: copy.deepcopy(config[key])
        for key in ("cube", "hand_pose", "grasp_pose")
    }
    state = np.random.SeedSequence(
        [int(seed), int(parent["candidate_id"]) & 0xFFFFFFFF, int(job.index)]
    )
    rng = np.random.default_rng(state)
    local_index = int(job.payload["local_index"])
    structured_branch = local_index < _PHYSICAL_PROBE_STRUCTURED_BRANCH_COUNT
    feedback = config["joint_pair_feedback"]
    if structured_branch:
        # These 27 candidates isolate the measured physical probe.  Preserve
        # the already validated parent controller exactly (notably the formal
        # gain=1.0/slip=2.0 branch) so preload/waypoint effects are auditable.
        alignment = float(feedback["alignment_gain"])
        slip = float(feedback["slip_recovery_gain_rad_per_m"])
    else:
        alignment = float(
            np.clip(
                float(feedback["alignment_gain"])
                + rng.uniform(-0.10, 0.10),
                0.25,
                1.0,
            )
        )
        slip = float(
            np.clip(
                float(feedback["slip_recovery_gain_rad_per_m"])
                + rng.uniform(-1.0, 1.0),
                2.0,
                8.0,
            )
        )
    config["joint_pair_feedback"] = _pair_feedback_mapping(
        feedback, alignment, slip
    )
    if structured_branch:
        combination_index, scale_index = divmod(
            local_index, len(_PHYSICAL_PROBE_WAYPOINT_SCALES)
        )
        thumb_index, mid_index = divmod(
            combination_index,
            len(_PHYSICAL_PROBE_MID_JOINT1_PRELOAD_RAD),
        )
        waypoint_scale = float(_PHYSICAL_PROBE_WAYPOINT_SCALES[scale_index])
        terminal = _PHYSICAL_PROBE_WAYPOINT_DELTA_RAD * waypoint_scale
        requested_residual = np.zeros(len(ACTIVE_ACTUATORS), dtype=np.float64)
        requested_residual[ACTIVE_ACTUATORS.index(
            "left_hand_thumb_rota_joint1_actuator"
        )] = _PHYSICAL_PROBE_THUMB_ROTA1_PRELOAD_RAD[thumb_index]
        requested_residual[ACTIVE_ACTUATORS.index(
            "left_hand_mid_joint1_actuator"
        )] = _PHYSICAL_PROBE_MID_JOINT1_PRELOAD_RAD[mid_index]
        refinement_branch: dict[str, Any] = {
            "schema_version": 1,
            "name": "structured_physical_probe",
            "structured_index": local_index,
            "thumb_rota1_preload_delta_rad": float(
                _PHYSICAL_PROBE_THUMB_ROTA1_PRELOAD_RAD[thumb_index]
            ),
            "mid_joint1_preload_delta_rad": float(
                _PHYSICAL_PROBE_MID_JOINT1_PRELOAD_RAD[mid_index]
            ),
            "waypoint_scale": waypoint_scale,
            "physical_probe_waypoint_delta_rad": {
                name: float(_PHYSICAL_PROBE_WAYPOINT_DELTA_RAD[index])
                for index, name in enumerate(ACTIVE_ACTUATORS)
            },
        }
    else:
        waypoint_scale = None
        terminal = rng.uniform(-0.003, 0.003, len(ACTIVE_ACTUATORS))
        requested_residual = rng.uniform(
            -0.006, 0.006, len(ACTIVE_ACTUATORS)
        )
        refinement_branch = {
            "schema_version": 1,
            "name": "random_local",
            "random_index": local_index
            - _PHYSICAL_PROBE_STRUCTURED_BRANCH_COUNT,
            "waypoint_scale": None,
            "physical_probe_waypoint_delta_rad": None,
        }
    # Joint-space waypoint refinement is smooth across all 21 nodes.  It does
    # not alter the hand/grasp pose, so the physical response model remains
    # bound to the same checkpoint as required by the plan.
    raw_plan = copy.deepcopy(config["manipulation_plan"])
    raw_plan.pop("plan_id", None)
    definition = resolve_experiment(config)
    registered_delta = definition.search_bounds.manipulation_delta_rad
    if registered_delta is None:
        raise RuntimeError("v15 refinement requires registered manipulation bounds")
    progress = np.asarray(raw_plan["knot_times_s"], dtype=np.float64)
    progress = progress / progress[-1]
    if structured_branch:
        if progress.shape[0] < 11:
            raise RuntimeError(
                "physical-probe refinement requires at least eleven plan nodes"
            )
        # The measured probe enters without a command discontinuity: node 5
        # receives half the vector, node 6 reaches the full vector, and the
        # full offset is held through node 10 and smoothly through HOLD at the
        # terminal command.
        blend = np.zeros_like(progress)
        blend[5] = 0.5
        blend[6:] = 1.0
    else:
        blend = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
    waypoint_applied_residual_rad: dict[str, list[float]] = {}
    for actuator_index, name in enumerate(ACTIVE_ACTUATORS):
        values = np.asarray(raw_plan["actuator_waypoints_rad"][name], dtype=np.float64)
        original_values = values.copy()
        values += terminal[actuator_index] * blend
        values = np.clip(
            values,
            float(registered_delta[name][0]),
            float(registered_delta[name][1]),
        )
        # Preserve the registered per-node trust region.
        for knot in range(1, len(values)):
            values[knot] = np.clip(
                values[knot], values[knot - 1] - 0.04, values[knot - 1] + 0.04
            )
        raw_plan["actuator_waypoints_rad"][name] = values.tolist()
        waypoint_applied_residual_rad[name] = (
            values - original_values
        ).tolist()
    raw_plan["plan_id"] = canonical_sha256(raw_plan)
    # Round-trip through the public schema-v2 parser before simulation.
    config["manipulation_plan"] = ManipulationPlanParameters.from_config(
        raw_plan
    ).as_config()
    config["control"]["manipulation_delta_rad"] = {
        name: float(config["manipulation_plan"]["actuator_waypoints_rad"][name][-1])
        for name in ACTIVE_ACTUATORS
    }

    # Refine the physical preload separately from the measured grasp pose.
    # The residual is deliberately small, deterministically seeded and clipped
    # to the experiment's registered (model-safe) command envelope.  It can
    # rebalance contact force without silently changing the hand root or the
    # nominal actual-qpos grasp definition.
    preload_bounds = definition.search_bounds.actuator_targets_rad
    if preload_bounds is None:
        raise RuntimeError("v15 refinement requires registered preload bounds")
    preload_before = copy.deepcopy(
        config["control"]["contact_preload_targets_rad"]
    )
    preload_after: dict[str, float] = {}
    applied_residual: dict[str, float] = {}
    bound_hits: list[str] = []
    for actuator_index, name in enumerate(ACTIVE_ACTUATORS):
        lower, upper = preload_bounds[name]
        requested = float(preload_before[name]) + float(
            requested_residual[actuator_index]
        )
        clipped = float(np.clip(requested, float(lower), float(upper)))
        preload_after[name] = clipped
        applied_residual[name] = clipped - float(preload_before[name])
        if not math.isclose(clipped, requested, rel_tol=0.0, abs_tol=1e-15):
            bound_hits.append(name)
    config["control"]["contact_preload_targets_rad"] = preload_after

    for key, value in fixed_state.items():
        if config[key] != value:
            raise AssertionError(f"v15 control refinement changed fixed {key}")
    for field in (
        "precontact_targets_rad",
        "contact_preload_targets_rad",
        "manipulation_delta_rad",
    ):
        if set(config["control"][field]) != set(ACTIVE_ACTUATORS):
            raise AssertionError(
                f"v15 refinement changed inactive-actuator isolation in {field}"
            )

    resolved = _install_job_metadata(
        config,
        job,
        evidence={
            "feedback_refinement": {
                "seed": int(seed),
                "local_index": local_index,
                "refinement_branch": refinement_branch,
                "waypoint_terminal_residual_rad": terminal.tolist(),
                "waypoint_node_envelope": blend.tolist(),
                "waypoint_applied_residual_rad": (
                    waypoint_applied_residual_rad
                ),
                "alignment_gain": alignment,
                "slip_recovery_gain_rad_per_m": slip,
                "contact_preload_requested_residual_rad": {
                    name: float(requested_residual[index])
                    for index, name in enumerate(ACTIVE_ACTUATORS)
                },
                "contact_preload_applied_residual_rad": applied_residual,
                "contact_preload_targets_before_rad": preload_before,
                "contact_preload_targets_after_rad": preload_after,
                "contact_preload_bounds_rad": {
                    name: [float(value[0]), float(value[1])]
                    for name, value in preload_bounds.items()
                },
                "contact_preload_bound_hits": sorted(bound_hits),
                "inactive_actuator_commands": "implicit_zero_unchanged",
                "parent_selection": copy.deepcopy(
                    job.payload.get("refinement_parent_selection", {})
                ),
                "pose_changed": False,
                "reprobe_required": False,
            },
            "full_reset_from_initial_no_contact_state": True,
        },
    )
    validate_config(resolved)
    for key, value in fixed_state.items():
        if resolved[key] != value:
            raise AssertionError(
                f"v15 identity installation changed fixed {key}"
            )
    return resolved


def _rerun_config(parent: Mapping[str, Any], job: V15CampaignJob) -> dict[str, Any]:
    return _install_job_metadata(
        _config_from_parent(parent),
        job,
        evidence={
            "exact_timestep_s": 0.001,
            "from_initial_no_contact_state": True,
            "checkpoint_used": False,
            "exact_parent_selection": copy.deepcopy(
                job.payload.get("exact_parent_selection", {})
            ),
        },
    )


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    result = np.empty((count, dimensions), dtype=np.float64)
    for axis in range(dimensions):
        result[:, axis] = (rng.permutation(count) + rng.random(count)) / count
    return result


def _perturbation_configs(
    base: Mapping[str, Any],
    jobs: Sequence[V15CampaignJob],
    *,
    family: str,
    seed: int,
) -> dict[int, dict[str, Any]]:
    if not jobs:
        return {}
    definition = resolve_experiment(base)
    limits = definition.robustness
    ordered = sorted(jobs, key=lambda value: int(value.payload.get("perturbation_index", value.payload.get("trial_index", value.index))))
    matrix = _latin_hypercube(
        len(ordered), 8,
        int(np.random.SeedSequence([seed, int(ordered[0].parent_candidate_id or 0) & 0xFFFFFFFF, 16 if family == "local" else 50]).generate_state(1)[0]),
    )
    base_xy = np.asarray(base["cube"]["center_xy_m"], dtype=np.float64)
    base_rpy = np.asarray(base["cube"].get("rpy_deg", (0.0, 0.0, 0.0)), dtype=np.float64)
    base_z = float(base["cube"].get("z_offset_m", 0.0))
    base_mass = float(base["cube"]["mass_kg"])
    base_friction = float(base["cube"]["friction"])

    def scale(value: float, bounds: Sequence[float]) -> float:
        return float(bounds[0] + value * (bounds[1] - bounds[0]))

    result: dict[int, dict[str, Any]] = {}
    for job, row in zip(ordered, matrix, strict=True):
        config = copy.deepcopy(dict(base))
        xy = np.asarray(
            [scale(row[0], limits.position_xy_delta_m), scale(row[1], limits.position_xy_delta_m)]
        )
        z = scale(row[2], limits.z_offset_delta_m)
        rpy = np.asarray(
            [scale(row[3 + axis], limits.rpy_delta_deg) for axis in range(3)]
        )
        mass = scale(row[6], limits.mass_scale)
        friction = scale(row[7], limits.friction_delta)
        config["cube"]["center_xy_m"] = (base_xy + xy).tolist()
        config["cube"]["z_offset_m"] = base_z + z
        config["cube"]["rpy_deg"] = (base_rpy + rpy).tolist()
        config["cube"]["mass_kg"] = base_mass * mass
        config["cube"]["friction"] = base_friction + friction
        config["run_context"] = {"kind": "robustness_trial"}
        result[job.candidate_id] = _install_job_metadata(
            config,
            job,
            evidence={
                "perturbation_family": family,
                "resolved_perturbations": {
                    "cube_center_xy_delta_m": xy.tolist(),
                    "cube_z_offset_delta_m": z,
                    "cube_rpy_delta_deg": rpy.tolist(),
                    "mass_scale": mass,
                    "friction_delta": friction,
                },
                "from_initial_no_contact_state": True,
            },
        )
    return result


@dataclass(slots=True)
class JointPairNearZeroPhysicsStageRunner:
    """Callable production stage runner consumed by the v15 orchestrator."""

    workers: int = 1
    seed: int = SEED

    def __post_init__(self) -> None:
        if isinstance(self.workers, bool) or int(self.workers) <= 0:
            raise ValueError("workers must be a positive integer")
        if isinstance(self.seed, bool) or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        self.workers = int(self.workers)
        self.seed = int(self.seed)

    def __call__(
        self,
        stage: str,
        jobs: Sequence[V15CampaignJob],
        workspace: Path,
        context: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        jobs = tuple(jobs)
        workspace = Path(workspace).expanduser().resolve()
        retain_grasp_trace = context.get("retain_grasp_trace", True)
        if not isinstance(retain_grasp_trace, bool):
            raise TypeError("stage context retain_grasp_trace must be a boolean")
        if stage == "static_filter":
            return _static_stage(jobs, workspace, context, seed=self.seed)
        parents = _parent_records(context)
        for job in jobs:
            if job.parent_candidate_id is None or job.parent_candidate_id not in parents:
                raise RuntimeError(f"{stage} job lost its canonical parent")
        if stage == "sequential_planning":
            return _run_planning_tasks(
                jobs, workspace, parents, workers=self.workers
            )

        tasks: list[
            tuple[dict[str, Any], str, int, bool, bool, dict[str, Any]]
        ] = []
        if stage == "dynamic_grasp":
            reference_config = _load_json(context["seed_config_path"])
            configs = {
                job.candidate_id: _materialize_grasp_variant(
                    parents[int(job.parent_candidate_id)],
                    job,
                    reference_config=reference_config,
                )
                for job in jobs
            }
        elif stage == "feedback_grid":
            configs = {
                job.candidate_id: _materialize_feedback_variant(
                    parents[int(job.parent_candidate_id)], job
                )
                for job in jobs
            }
        elif stage == "feedback_refinement":
            configs = {
                job.candidate_id: _refined_config(
                    parents[int(job.parent_candidate_id)], job, seed=self.seed
                )
                for job in jobs
            }
        elif stage == "exact_rerun":
            configs = {
                job.candidate_id: _rerun_config(
                    parents[int(job.parent_candidate_id)], job
                )
                for job in jobs
            }
        elif stage in {"local_perturbation", "robustness"}:
            configs = {}
            grouped: dict[int, list[V15CampaignJob]] = {}
            for job in jobs:
                assert job.parent_candidate_id is not None
                grouped.setdefault(job.parent_candidate_id, []).append(job)
            for parent_id, group in grouped.items():
                configs.update(
                    _perturbation_configs(
                        _config_from_parent(parents[parent_id]),
                        group,
                        family="local" if stage == "local_perturbation" else "robustness",
                        seed=self.seed,
                    )
                )
        else:
            raise ValueError(f"unsupported v15 physics stage {stage!r}")

        for job in jobs:
            tasks.append(
                (
                    configs[job.candidate_id],
                    str(_candidate_directory(workspace, stage, job.candidate_id)),
                    int(job.candidate_id),
                    stage == "exact_rerun",
                    retain_grasp_trace,
                    job.descriptor(),
                )
            )
        records = list(_run_simulation_tasks(tasks, workers=self.workers))
        # Local/robust records need their nominal parent for pass aggregation.
        for record, job in zip(records, jobs, strict=True):
            record["parent_candidate_id"] = job.parent_candidate_id
            if "plan_candidate_id" in job.payload:
                record["plan_candidate_id"] = int(
                    job.payload["plan_candidate_id"]
                )
        return records


def create_joint_pair_near_zero_campaign_backend(
    *, workers: int = 1, seed: int = SEED
) -> V15CampaignBackend:
    """Return the production backend used by ``grasp_cube.py tune``."""

    return V15CampaignBackend(
        JointPairNearZeroPhysicsStageRunner(workers=workers, seed=seed)
    )


__all__ = [
    "JointPairNearZeroPhysicsStageRunner",
    "PHYSICS_BACKEND_SCHEMA_VERSION",
    "create_joint_pair_near_zero_campaign_backend",
]
