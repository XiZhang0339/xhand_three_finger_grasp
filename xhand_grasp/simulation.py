"""Generic deterministic MuJoCo simulation loop for registered experiments."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

from xhand_tactile import TactileReader

from .config import ACTIVE_ACTUATORS, ACTIVE_FINGERS, precontact_targets
from .controller import (
    ContactPreservingPlannedLiftController,
    GraspVerifyThenManipulateController,
    InitialPoseHistoryEvidence,
    JointPairAlignedContactPreservingPlannedLiftController,
    OperationFeedback,
    RollingSlipAwareJointPairAlignedContactPreservingPlannedLiftController,
    TargetFaceEvidence,
    ControlState,
    build_grasp_controller,
    compute_grasp_gate_evidence,
    compute_target_face_evidence,
    grasp_gate_order,
    quaternion_drift_deg,
)
from .joint_pair_geometry import (
    JointPairBinding,
    joint_pair_telemetry,
    resolve_joint_pair,
)
from .contacts import (
    DEFAULT_BOX_CONTACT_THRESHOLDS,
    FACE_ORDER,
    BoxContactThresholds,
    classify_box_contact,
    surface_witness,
    target_face_contact_centroids,
    three_finger_height_spread,
)
from .contact_geometry import distal_collision_geom_ids, nearest_taxel_assignment
from .contact_environment import (
    CONTACT_ENVIRONMENT_FIELD_PATHS,
    ContactEnvironmentSpec,
    apply_to_model as apply_contact_environment_to_model,
    audit_allowed_model_changes,
    requested_environment_snapshot,
    runtime_cube_contact_snapshot,
    verify_runtime_cube_contacts,
)
from .contact_point_targeting import (
    ContactPointObservation,
    ContactPointPlan,
    contact_point_observation,
    contact_point_plan_from_config,
)
from .closure_alignment import (
    aggregate_force_weighted_closure_contacts,
    point_jacobian_command_velocity,
)
from .evaluation import evaluate_trace
from .rendering import (
    VideoRecorder,
    VideoSettings,
    expected_video_frame_count,
    probe_video,
)
from .rolling_contact_slip import (
    ROLLING_CONTACT_SLIP_SCHEMA_VERSION,
    RollingAwareContactSlipEstimator,
    RollingAwareSlipEstimate,
    RollingSlipSettings,
    RollingTangentJacobian,
    mujoco_contact_patch_kinematics,
    mujoco_contact_tangent_position_jacobian,
)
from .scene import ModelInfo, build_model, signed_finger_down_tilt_deg
from .trajectory import _phase_steps, actuator_target_vector, smoothstep
from .v14_identity import (
    V14_TOP_LEVEL_ID_FIELDS,
    v14_identity_trace_values,
)


# These labels are persisted in schema-v2 traces.  Physical labels deliberately
# match ``contact_topology.target_faces`` so a consumer can resolve axes without
# importing Python enums; the final two bins describe rejected classifications.
PERSISTED_FACE_ORDER = (
    "+X",
    "-X",
    "+Y",
    "-Y",
    "+Z",
    "-Z",
    "EDGE_CORNER",
    "UNKNOWN",
)


@dataclass(frozen=True)
class ContactSnapshot:
    finger_forces: np.ndarray
    forbidden: bool
    support: bool
    floor: bool
    max_penetration: float
    friction_error: float
    cube_contact_seen: bool
    contact_dim_ok: bool
    hand_cube_contact: bool
    distal_face_force_n: np.ndarray | None = None
    active_nondistal_force_n: np.ndarray | None = None
    distal_face_position_moment_n_m: np.ndarray | None = None
    distal_pad_force_n: np.ndarray | None = None
    distal_nonpad_force_n: np.ndarray | None = None
    distal_pad_force_fraction: np.ndarray | None = None
    distal_active_taxel_count: np.ndarray | None = None

    def legacy_tuple(self) -> tuple[np.ndarray, bool, bool, bool, float, float, bool, bool]:
        return (
            self.finger_forces,
            self.forbidden,
            self.support,
            self.floor,
            self.max_penetration,
            self.friction_error,
            self.cube_contact_seen,
            self.contact_dim_ok,
        )


@dataclass(frozen=True)
class ClosureAlignmentSnapshot:
    """One force-weighted target-face closure sample for all active fingers."""

    witness_world_m: np.ndarray
    command_velocity_world_m_s: np.ndarray
    cube_outward_normal_world: np.ndarray
    cosine: np.ndarray
    angle_deg: np.ndarray
    inward_speed_m_s: np.ndarray
    tangent_speed_m_s: np.ndarray
    valid: np.ndarray
    target_contact_force_n: np.ndarray


def _palm_down_angle_deg(
    model: mujoco.MjModel, data: mujoco.MjData, info: ModelInfo
) -> float:
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    norm = float(np.linalg.norm(gravity))
    if norm <= 1e-12:
        raise ValueError("schema v2 requires non-zero gravity for palm-down evaluation")
    if info.palm_frame_site_id < 0:
        raise ValueError("schema v2 model is missing the palm-frame site")
    rotation = data.site_xmat[info.palm_frame_site_id].reshape(3, 3)
    palm_normal_world = rotation[:, 2]
    cosine = float(np.dot(palm_normal_world, gravity / norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _quaternion_delta_rotvec_world(
    reference_wxyz: np.ndarray, value_wxyz: np.ndarray
) -> np.ndarray:
    """Return the shortest world-frame rotation vector from reference to value."""

    reference = np.asarray(reference_wxyz, dtype=np.float64).reshape(4)
    value = np.asarray(value_wxyz, dtype=np.float64).reshape(4)
    reference /= np.linalg.norm(reference)
    value /= np.linalg.norm(value)
    relative = np.empty(4, dtype=np.float64)
    conjugate = reference.copy()
    conjugate[1:] *= -1.0
    mujoco.mju_mulQuat(relative, value, conjugate)
    if relative[0] < 0.0:
        relative *= -1.0
    vector_norm = float(np.linalg.norm(relative[1:]))
    if vector_norm <= 1e-14:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, float(relative[0]))
    return np.asarray(relative[1:] * (angle / vector_norm), dtype=np.float64)


def _directed_joint_pair_observation(
    data: mujoco.MjData,
    binding: JointPairBinding,
) -> dict[str, Any]:
    """Return the schema-v15 directed pair residual in the cube frame."""

    try:
        telemetry = joint_pair_telemetry(data, binding)
    except ValueError:
        return {
            "vector_cube_m": np.zeros(3, dtype=np.float64),
            "residual": np.zeros(2, dtype=np.float64),
            "angle_deg": 180.0,
            "length_m": 0.0,
            "positive_y": False,
            "valid": False,
        }
    vector = np.asarray(telemetry["vector_cube_m"], dtype=np.float64)
    length = float(telemetry["length_m"])
    y = float(vector[1])
    valid = bool(
        np.isfinite(vector).all()
        and np.isfinite(length)
        and length > 1e-12
        and abs(y) > 1e-12
    )
    if not valid:
        return {
            "vector_cube_m": vector.copy(),
            "residual": np.zeros(2, dtype=np.float64),
            "angle_deg": 180.0,
            "length_m": max(0.0, length) if np.isfinite(length) else 0.0,
            "positive_y": False,
            "valid": False,
        }
    residual = np.asarray((vector[0] / y, vector[2] / y), dtype=np.float64)
    angle = float(
        np.degrees(np.arccos(np.clip(y / length, -1.0, 1.0)))
    )
    return {
        "vector_cube_m": vector.copy(),
        "residual": residual,
        "angle_deg": angle,
        "length_m": length,
        "positive_y": bool(y > 0.0),
        "valid": bool(np.isfinite(residual).all() and np.isfinite(angle)),
    }


def _active_finger_self_collision_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: ModelInfo,
) -> dict[str, Any]:
    """Aggregate contacts between two different active fingers for v15."""

    pairs: list[str] = []
    total_force = 0.0
    maximum_penetration = 0.0
    contact_force = np.zeros(6, dtype=np.float64)
    for contact_index, contact in enumerate(data.contact[: data.ncon]):
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        finger1 = info.hand_body_parts.get(body1)
        finger2 = info.hand_body_parts.get(body2)
        if (
            finger1 not in ACTIVE_FINGERS
            or finger2 not in ACTIVE_FINGERS
            or finger1 == finger2
        ):
            continue
        normal_force = 0.0
        if int(contact.efc_address) >= 0:
            contact_force[:] = 0.0
            mujoco.mj_contactForce(
                model, data, contact_index, contact_force
            )
            normal_force = max(0.0, float(contact_force[0]))
        penetration = max(0.0, -float(contact.dist))
        if normal_force <= 1e-8 and penetration <= 0.0:
            continue
        name1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom1
        ) or f"geom_{geom1}"
        name2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom2
        ) or f"geom_{geom2}"
        first, second = sorted((str(name1), str(name2)))
        pairs.append(f"{first}|{second}")
        total_force += normal_force
        maximum_penetration = max(maximum_penetration, penetration)
    pairs.sort()
    return {
        "active": bool(pairs),
        "contact_count": len(pairs),
        "normal_force_n": float(total_force),
        "max_penetration_m": float(maximum_penetration),
        "pairs": ";".join(pairs),
    }


def contact_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: ModelInfo,
    *,
    classify_faces: bool = False,
    box_thresholds: BoxContactThresholds = DEFAULT_BOX_CONTACT_THRESHOLDS,
    distal_taxel_site_ids: np.ndarray | None = None,
    max_taxel_assignment_distance_m: float = 0.006,
) -> ContactSnapshot:
    """Aggregate every active cube contact after ``mj_forward``.

    Legacy values are always produced.  Face arrays are allocated only for the
    v2 experiment so v1 trace keys and numerical behavior remain unchanged.
    """

    finger_forces = np.zeros(3, dtype=np.float64)
    distal_face_force = (
        np.zeros((3, len(FACE_ORDER)), dtype=np.float64) if classify_faces else None
    )
    distal_face_position_moment = (
        np.zeros((3, len(FACE_ORDER), 3), dtype=np.float64)
        if classify_faces
        else None
    )
    active_nondistal = np.zeros(3, dtype=np.float64) if classify_faces else None
    taxel_site_ids: np.ndarray | None = None
    pad_force: np.ndarray | None = None
    nonpad_force: np.ndarray | None = None
    active_taxels: list[set[int]] | None = None
    if distal_taxel_site_ids is not None:
        taxel_site_ids = np.asarray(distal_taxel_site_ids, dtype=np.int64)
        if (
            taxel_site_ids.ndim != 2
            or taxel_site_ids.shape[0] != len(ACTIVE_FINGERS)
            or taxel_site_ids.shape[1] == 0
        ):
            raise ValueError(
                "distal_taxel_site_ids must have shape (3, taxels) in active-finger order"
            )
        if np.any(taxel_site_ids < 0) or np.any(taxel_site_ids >= model.nsite):
            raise ValueError("distal_taxel_site_ids contains an invalid site id")
        assignment_distance = float(max_taxel_assignment_distance_m)
        if not np.isfinite(assignment_distance) or assignment_distance <= 0.0:
            raise ValueError(
                "max_taxel_assignment_distance_m must be positive and finite"
            )
        pad_force = np.zeros(3, dtype=np.float64)
        nonpad_force = np.zeros(3, dtype=np.float64)
        active_taxels = [set() for _ in ACTIVE_FINGERS]
    forbidden = False
    support = False
    floor = False
    max_penetration = 0.0
    friction_error = 0.0
    cube_contact_seen = False
    contact_dim_ok = True
    hand_cube_contact = False
    contact_force = np.zeros(6, dtype=np.float64)

    cube_rotation = (
        data.geom_xmat[info.cube_geom_id].reshape(3, 3) if classify_faces else None
    )
    cube_position = data.geom_xpos[info.cube_geom_id] if classify_faces else None
    half_extents = model.geom_size[info.cube_geom_id] if classify_faces else None

    for contact_index, contact in enumerate(data.contact[: data.ncon]):
        max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if info.cube_geom_id not in (geom1, geom2):
            continue
        other_geom = geom2 if geom1 == info.cube_geom_id else geom1
        if other_geom < 0:
            continue
        active = int(contact.efc_address) >= 0
        normal_force = 0.0
        if active:
            mujoco.mj_contactForce(model, data, contact_index, contact_force)
            normal_force = max(0.0, float(contact_force[0]))
        touching = float(contact.dist) <= 0.0 or normal_force > 1e-8
        cube_contact_seen = cube_contact_seen or active
        friction_error = max(
            friction_error,
            abs(float(contact.friction[0]) - info.requested_friction),
            abs(float(contact.friction[1]) - info.requested_friction),
        )
        contact_dim_ok = contact_dim_ok and int(contact.dim) == 4

        if other_geom == info.support_geom_id:
            support = support or touching
            continue
        if other_geom == info.floor_geom_id:
            floor = floor or touching
            continue

        other_body = int(model.geom_bodyid[other_geom])
        if other_body not in info.hand_body_parts:
            continue
        hand_cube_contact = hand_cube_contact or touching
        weld_id = int(model.body_weldid[other_body])
        distal_index: int | None = None
        for finger_index, finger in enumerate(ACTIVE_FINGERS):
            if weld_id == info.distal_weld_ids[finger]:
                distal_index = finger_index
                finger_forces[finger_index] += normal_force
                break

        part = info.hand_body_parts[other_body]
        if classify_faces and normal_force > 0.0 and part in ACTIVE_FINGERS:
            assert active_nondistal is not None
            if distal_index is None:
                active_nondistal[ACTIVE_FINGERS.index(part)] += normal_force
            else:
                assert distal_face_force is not None
                assert cube_rotation is not None
                assert cube_position is not None
                assert half_extents is not None
                contact_normal = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)[
                    0
                ]
                witness = surface_witness(
                    np.asarray(contact.pos, dtype=np.float64),
                    float(contact.dist),
                    contact_normal,
                    cube_is_geom1=geom1 == info.cube_geom_id,
                )
                point_local = cube_rotation.T @ (
                    witness.position_world - cube_position
                )
                normal_local = cube_rotation.T @ witness.outward_normal_world
                classification = classify_box_contact(
                    point_local,
                    normal_local,
                    half_extents,
                    thresholds=box_thresholds,
                )
                distal_face_force[
                    distal_index, FACE_ORDER.index(classification.face)
                ] += normal_force
                assert distal_face_position_moment is not None
                distal_face_position_moment[
                    distal_index, FACE_ORDER.index(classification.face)
                ] += normal_force * witness.position_world

        if distal_index is not None and normal_force > 0.0 and taxel_site_ids is not None:
            assert pad_force is not None
            assert nonpad_force is not None
            assert active_taxels is not None
            # ``contact.pos`` is the midpoint of the two nearest points.  The
            # already recovered cube witness therefore gives the exact distal
            # collision-geom witness by reflection through that midpoint.
            contact_normal = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)[
                0
            ]
            cube_witness = surface_witness(
                np.asarray(contact.pos, dtype=np.float64),
                float(contact.dist),
                contact_normal,
                cube_is_geom1=geom1 == info.cube_geom_id,
            )
            distal_witness = (
                2.0 * np.asarray(contact.pos, dtype=np.float64)
                - cube_witness.position_world
            )
            assignment = nearest_taxel_assignment(
                distal_witness,
                data.site_xpos[taxel_site_ids[distal_index]],
                max_assignment_distance_m=max_taxel_assignment_distance_m,
            )
            if assignment.assigned:
                assert assignment.taxel_index is not None
                pad_force[distal_index] += normal_force
                active_taxels[distal_index].add(assignment.taxel_index)
            else:
                nonpad_force[distal_index] += normal_force

        if part in {"palm", "ring", "pinky"}:
            forbidden = forbidden or touching

    pad_fraction = None
    active_taxel_count = None
    if pad_force is not None:
        assert nonpad_force is not None
        assert active_taxels is not None
        total_distal = pad_force + nonpad_force
        pad_fraction = np.divide(
            pad_force,
            total_distal,
            out=np.zeros_like(pad_force),
            where=total_distal > 0.0,
        )
        active_taxel_count = np.asarray(
            [len(indices) for indices in active_taxels], dtype=np.int64
        )

    return ContactSnapshot(
        finger_forces=finger_forces,
        forbidden=forbidden,
        support=support,
        floor=floor,
        max_penetration=max_penetration,
        friction_error=friction_error,
        cube_contact_seen=cube_contact_seen,
        contact_dim_ok=contact_dim_ok,
        hand_cube_contact=hand_cube_contact,
        distal_face_force_n=distal_face_force,
        active_nondistal_force_n=active_nondistal,
        distal_face_position_moment_n_m=distal_face_position_moment,
        distal_pad_force_n=pad_force,
        distal_nonpad_force_n=nonpad_force,
        distal_pad_force_fraction=pad_fraction,
        distal_active_taxel_count=active_taxel_count,
    )


def closure_alignment_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: ModelInfo,
    config: dict[str, Any],
    actuator_target_velocity_rad_s: np.ndarray,
    *,
    box_thresholds: BoxContactThresholds,
) -> ClosureAlignmentSnapshot:
    """Aggregate real target-face distal witnesses for schema-v8 CLOSE.

    MuJoCo's contact frame points from geom1 to geom2.  ``surface_witness``
    canonicalizes that convention to a cube-outward normal before the desired
    finger direction is taken as its negative.
    """

    from .evaluation import face_from_label

    velocity_command = np.asarray(
        actuator_target_velocity_rad_s, dtype=np.float64
    )
    if velocity_command.shape != (model.nu,) or not np.isfinite(
        velocity_command
    ).all():
        raise ValueError(
            "actuator_target_velocity_rad_s must have shape (model.nu,)"
        )
    per_finger_force: list[list[float]] = [[] for _ in ACTIVE_FINGERS]
    per_finger_velocity: list[list[np.ndarray]] = [
        [] for _ in ACTIVE_FINGERS
    ]
    per_finger_normal: list[list[np.ndarray]] = [
        [] for _ in ACTIVE_FINGERS
    ]
    per_finger_witness: list[list[np.ndarray]] = [
        [] for _ in ACTIVE_FINGERS
    ]
    target_faces = tuple(
        face_from_label(config["contact_topology"]["target_faces"][finger])
        for finger in ACTIVE_FINGERS
    )
    cube_rotation = data.geom_xmat[info.cube_geom_id].reshape(3, 3)
    cube_position = data.geom_xpos[info.cube_geom_id]
    half_extents = model.geom_size[info.cube_geom_id]
    contact_force = np.zeros(6, dtype=np.float64)

    for contact_index, contact in enumerate(data.contact[: data.ncon]):
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if info.cube_geom_id not in (geom1, geom2):
            continue
        if int(contact.efc_address) < 0:
            continue
        other_geom = geom2 if geom1 == info.cube_geom_id else geom1
        if other_geom < 0:
            continue
        other_body = int(model.geom_bodyid[other_geom])
        weld_id = int(model.body_weldid[other_body])
        finger_index = next(
            (
                index
                for index, finger in enumerate(ACTIVE_FINGERS)
                if weld_id == info.distal_weld_ids[finger]
            ),
            None,
        )
        if finger_index is None:
            continue
        mujoco.mj_contactForce(model, data, contact_index, contact_force)
        normal_force = max(0.0, float(contact_force[0]))
        if normal_force <= 0.0:
            continue
        contact_normal = np.asarray(
            contact.frame, dtype=np.float64
        ).reshape(3, 3)[0]
        cube_witness = surface_witness(
            np.asarray(contact.pos, dtype=np.float64),
            float(contact.dist),
            contact_normal,
            cube_is_geom1=geom1 == info.cube_geom_id,
        )
        point_local = cube_rotation.T @ (
            cube_witness.position_world - cube_position
        )
        normal_local = cube_rotation.T @ cube_witness.outward_normal_world
        classification = classify_box_contact(
            point_local,
            normal_local,
            half_extents,
            thresholds=box_thresholds,
        )
        if classification.face is not target_faces[finger_index]:
            continue
        distal_witness = (
            2.0 * np.asarray(contact.pos, dtype=np.float64)
            - cube_witness.position_world
        )
        point_velocity = point_jacobian_command_velocity(
            model,
            data,
            distal_witness,
            other_body,
            info.actuator_dof_adrs,
            velocity_command,
        )
        per_finger_force[finger_index].append(normal_force)
        per_finger_velocity[finger_index].append(point_velocity)
        per_finger_normal[finger_index].append(
            cube_witness.outward_normal_world
        )
        per_finger_witness[finger_index].append(distal_witness)

    witness = np.zeros((3, 3), dtype=np.float64)
    command_velocity = np.zeros((3, 3), dtype=np.float64)
    outward_normal = np.zeros((3, 3), dtype=np.float64)
    cosine = np.full(3, -1.0, dtype=np.float64)
    angle = np.full(3, 180.0, dtype=np.float64)
    inward = np.zeros(3, dtype=np.float64)
    tangent = np.zeros(3, dtype=np.float64)
    valid = np.zeros(3, dtype=bool)
    target_force = np.zeros(3, dtype=np.float64)
    minimum_force = float(config["closure_alignment"]["min_contact_force_n"])
    for finger_index in range(3):
        force = np.asarray(per_finger_force[finger_index], dtype=np.float64)
        velocities = np.asarray(
            per_finger_velocity[finger_index], dtype=np.float64
        ).reshape((-1, 3))
        normals = np.asarray(
            per_finger_normal[finger_index], dtype=np.float64
        ).reshape((-1, 3))
        sample = aggregate_force_weighted_closure_contacts(
            force,
            velocities,
            normals,
            minimum_total_force_n=minimum_force,
        )
        target_force[finger_index] = sample.total_normal_force_n
        command_velocity[finger_index] = sample.command_velocity_world_m_s
        outward_normal[finger_index] = sample.cube_outward_normal_world
        cosine[finger_index] = sample.cosine
        angle[finger_index] = sample.angle_deg
        inward[finger_index] = sample.inward_speed_m_s
        tangent[finger_index] = sample.tangent_speed_m_s
        valid[finger_index] = sample.valid
        if force.size and float(np.sum(force)) > 0.0:
            witness[finger_index] = np.average(
                np.asarray(
                    per_finger_witness[finger_index], dtype=np.float64
                ),
                axis=0,
                weights=force,
            )
    return ClosureAlignmentSnapshot(
        witness_world_m=witness,
        command_velocity_world_m_s=command_velocity,
        cube_outward_normal_world=outward_normal,
        cosine=cosine,
        angle_deg=angle,
        inward_speed_m_s=inward,
        tangent_speed_m_s=tangent,
        valid=valid,
        target_contact_force_n=target_force,
    )


def _contact_snapshot(
    model: mujoco.MjModel, data: mujoco.MjData, info: ModelInfo
) -> tuple[np.ndarray, bool, bool, bool, float, float, bool, bool]:
    """Compatibility adapter for the historical private helper."""

    return contact_snapshot(model, data, info).legacy_tuple()


def _allocate_traces(
    model: mujoco.MjModel,
    total_steps: int,
    *,
    schema_version: int,
    contact_feedback_schema_version: int = 1,
) -> dict[str, np.ndarray]:
    traces: dict[str, np.ndarray] = {
        "time": np.empty(total_steps),
        "cube_pos": np.empty((total_steps, 3)),
        "cube_quat": np.empty((total_steps, 4)),
        "cube_velocity": np.empty((total_steps, 6)),
        "root_pos": np.empty((total_steps, 3)),
        "root_quat": np.empty((total_steps, 4)),
        "ctrl": np.empty((total_steps, model.nu)),
        "joint_qpos": np.empty((total_steps, model.nu)),
        "joint_qvel": np.empty((total_steps, model.nu)),
        "actuator_force": np.empty((total_steps, model.nu)),
        "finger_contact_force": np.empty((total_steps, 3)),
        "tactile_max": np.empty((total_steps, 5)),
        "forbidden_contact": np.zeros(total_steps, dtype=bool),
        "support_contact": np.zeros(total_steps, dtype=bool),
        "floor_contact": np.zeros(total_steps, dtype=bool),
        "max_penetration": np.empty(total_steps),
        "friction_error": np.empty(total_steps),
        "cube_contact_seen": np.zeros(total_steps, dtype=bool),
        "contact_dim_ok": np.ones(total_steps, dtype=bool),
        "finite": np.ones(total_steps, dtype=bool),
    }
    if schema_version >= 2:
        traces.update(
            {
                # Static, self-describing axes for persisted numeric arrays.
                # Fixed-width Unicode keeps ``allow_pickle=False`` sufficient
                # when loading the archive.
                "face_order": np.asarray(PERSISTED_FACE_ORDER, dtype=np.str_),
                "finger_order": np.asarray(ACTIVE_FINGERS, dtype=np.str_),
                "actuator_order": np.asarray(
                    [model.actuator(index).name for index in range(model.nu)],
                    dtype=np.str_,
                ),
                "palm_down_angle_deg": np.empty(total_steps),
                "distal_face_force_n": np.zeros(
                    (total_steps, 3, len(FACE_ORDER)), dtype=np.float64
                ),
                "active_nondistal_force_n": np.zeros(
                    (total_steps, 3), dtype=np.float64
                ),
                "target_face_force_purity": np.zeros(
                    (total_steps, 3), dtype=np.float64
                ),
                "target_face_topology": np.zeros(total_steps, dtype=bool),
            }
        )
    if schema_version >= 3:
        gate_order = grasp_gate_order(schema_version)
        traces.update(
            {
                "control_state": np.empty(total_steps, dtype="<U10"),
                "grasp_gate_order": np.asarray(gate_order, dtype=np.str_),
                "grasp_gate": np.zeros(
                    (total_steps, len(gate_order)), dtype=bool
                ),
                "grasp_gate_consecutive_steps": np.zeros(
                    total_steps, dtype=np.int64
                ),
                "grasp_acquired": np.zeros(total_steps, dtype=bool),
                "manipulation_progress": np.zeros(total_steps, dtype=np.float64),
                "target_face_effective": np.zeros((total_steps, 3), dtype=bool),
            }
        )
    if schema_version >= 4:
        traces.update(
            {
                "finger_down_tilt_deg": np.empty(total_steps),
                "distal_face_position_moment_n_m": np.zeros(
                    (total_steps, 3, len(FACE_ORDER), 3), dtype=np.float64
                ),
                "target_face_contact_centroid_world_m": np.zeros(
                    (total_steps, 3, 3), dtype=np.float64
                ),
                "target_face_contact_centroid_valid": np.zeros(
                    (total_steps, 3), dtype=bool
                ),
                "three_contact_height_spread_m": np.zeros(
                    total_steps, dtype=np.float64
                ),
                "three_contact_height_aligned": np.zeros(
                    total_steps, dtype=bool
                ),
            }
        )
    if schema_version >= 5:
        traces.update(
            {
                "root_cube_center_distance_m": np.empty(total_steps),
                "thumb_bend_command_rad": np.empty(total_steps),
                "thumb_bend_qpos_rad": np.empty(total_steps),
                "distal_pad_force_n": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "distal_nonpad_force_n": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "distal_pad_force_fraction": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "distal_active_taxel_count": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.int64
                ),
            }
        )
    if schema_version >= 6:
        traces.update(
            {
                # The immutable reference is sampled after ``mj_forward`` at
                # reset and before the first integration step.  Keeping it as
                # a scalar axis avoids silently using post-step ``cube_pos[0]``
                # as the reference during offline recomputation.
                "initial_cube_pos_m": np.empty(3, dtype=np.float64),
                "initial_cube_quat": np.empty(4, dtype=np.float64),
                "initial_joint_qpos_rad": np.empty(
                    model.nu, dtype=np.float64
                ),
                "initialized_at_pregrasp": np.asarray(False, dtype=bool),
                "cube_translation_from_initial_m": np.empty(
                    total_steps, dtype=np.float64
                ),
                "cube_orientation_from_initial_deg": np.empty(
                    total_steps, dtype=np.float64
                ),
                "initial_pose_translation_history_stable": np.zeros(
                    total_steps, dtype=bool
                ),
                "initial_pose_orientation_history_stable": np.zeros(
                    total_steps, dtype=bool
                ),
                "pregrasp_pose_within_limit": np.zeros(
                    total_steps, dtype=bool
                ),
                "pregrasp_pose_preserved_latched": np.zeros(
                    total_steps, dtype=bool
                ),
                "pregrasp_support_retained_latched": np.zeros(
                    total_steps, dtype=bool
                ),
                "settle_hand_contact_free_latched": np.zeros(
                    total_steps, dtype=bool
                ),
                "hand_cube_contact": np.zeros(total_steps, dtype=bool),
                "close_progress": np.zeros(
                    (total_steps, model.nu), dtype=np.float64
                ),
                "pregrasp_target_rad": np.zeros(model.nu, dtype=np.float64),
                "close_start_fraction": np.zeros(model.nu, dtype=np.float64),
                "close_end_fraction": np.ones(model.nu, dtype=np.float64),
                "first_distal_contact_step": np.full(
                    len(ACTIVE_FINGERS), -1, dtype=np.int64
                ),
            }
        )
    if schema_version >= 8:
        traces.update(
            {
                "command_target_velocity_rad_s": np.zeros(
                    (total_steps, model.nu), dtype=np.float64
                ),
                "closure_witness_world_m": np.zeros(
                    (total_steps, 3, 3), dtype=np.float64
                ),
                "closure_command_velocity_world_m_s": np.zeros(
                    (total_steps, 3, 3), dtype=np.float64
                ),
                "closure_cube_outward_normal_world": np.zeros(
                    (total_steps, 3, 3), dtype=np.float64
                ),
                "closure_alignment_cosine": np.full(
                    (total_steps, 3), -1.0, dtype=np.float64
                ),
                "closure_alignment_angle_deg": np.full(
                    (total_steps, 3), 180.0, dtype=np.float64
                ),
                "closure_inward_speed_m_s": np.zeros(
                    (total_steps, 3), dtype=np.float64
                ),
                "closure_tangent_speed_m_s": np.zeros(
                    (total_steps, 3), dtype=np.float64
                ),
                "closure_alignment_valid": np.zeros(
                    (total_steps, 3), dtype=bool
                ),
                "closure_target_contact_force_n": np.zeros(
                    (total_steps, 3), dtype=np.float64
                ),
                "operation_height_filtered_m": np.zeros(
                    total_steps, dtype=np.float64
                ),
                "operation_vertical_velocity_filtered_m_s": np.zeros(
                    total_steps, dtype=np.float64
                ),
                "operation_vertical_acceleration_filtered_m_s2": np.zeros(
                    total_steps, dtype=np.float64
                ),
                "operation_vertical_jerk_filtered_m_s3": np.zeros(
                    total_steps, dtype=np.float64
                ),
                "operation_lateral_displacement_m": np.zeros(
                    total_steps, dtype=np.float64
                ),
                "operation_orientation_drift_deg": np.zeros(
                    total_steps, dtype=np.float64
                ),
                "motion_filter_valid": np.zeros(total_steps, dtype=bool),
            }
        )
    if schema_version >= 9:
        traces.update(
            {
                "grasp_pose_base_gate": np.zeros(total_steps, dtype=bool),
                "grasp_pose_actual_joint_qpos_rad": np.zeros(
                    (total_steps, len(ACTIVE_ACTUATORS)), dtype=np.float64
                ),
                "grasp_pose_thumb_actual_within_range": np.zeros(
                    total_steps, dtype=bool
                ),
                "grasp_pose_nominal_joint_qpos_rad": np.zeros(
                    len(ACTIVE_ACTUATORS), dtype=np.float64
                ),
                "precontact_target_rad": np.zeros(
                    model.nu, dtype=np.float64
                ),
            }
        )
    if schema_version >= 11:
        traces.update(
            {
                "clockwise_orbit_deg": np.asarray(0.0, dtype=np.float64),
                "initial_root_position_cube_m": np.zeros(3, dtype=np.float64),
                "initial_cube_from_root_rotation": np.eye(3, dtype=np.float64),
                "root_delta_cube_m": np.zeros(3, dtype=np.float64),
                "wrist_local_rotvec_deg": np.zeros(3, dtype=np.float64),
                "relative_pose_boundary_rejection_names": np.asarray(
                    [], dtype=np.str_
                ),
                "relative_pose_boundary_rejection_counts": np.asarray(
                    [], dtype=np.int64
                ),
            }
        )
    if schema_version >= 12:
        traces.update(
            {
                "contact_point_plan_id": np.asarray("", dtype=np.str_),
                "target_contact_points_cube_local_m": np.zeros(
                    (len(ACTIVE_FINGERS), 3), dtype=np.float64
                ),
                "target_contact_point_radius_m": np.asarray(
                    0.0, dtype=np.float64
                ),
                "target_face_contact_centroid_cube_local_m": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS), 3), dtype=np.float64
                ),
                "target_contact_point_tangent_error_m": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "target_contact_point_within_radius": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
            }
        )
    if schema_version >= 13:
        traces.update(
            {
                "grasp_contact_centroid_baseline_cube_local_m": np.zeros(
                    (len(ACTIVE_FINGERS), 3), dtype=np.float64
                ),
                "grasp_contact_centroid_baseline_valid": np.zeros(
                    len(ACTIVE_FINGERS), dtype=bool
                ),
                "grasp_contact_centroid_baseline_force_sum_n": np.zeros(
                    len(ACTIVE_FINGERS), dtype=np.float64
                ),
                "target_contact_tangent_slip_from_grasp_m": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "target_contact_tangent_slip_from_grasp_valid": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
            }
        )
    if schema_version >= 14:
        traces.update(
            {
                # Scalar, fixed-width Unicode values keep NPZ loading safe
                # with ``allow_pickle=False``.  They are populated from the
                # independently validated resolved config during reset.
                **{
                    name: np.asarray("", dtype=np.str_)
                    for name in V14_TOP_LEVEL_ID_FIELDS
                },
                "manipulation_plan_knot_times_s": np.zeros(21, dtype=np.float64),
                "manipulation_plan_waypoints_rad": np.zeros(
                    (21, model.nu), dtype=np.float64
                ),
                "manipulation_plan_desired_cube_position_delta_m": np.zeros(
                    (21, 3), dtype=np.float64
                ),
                "manipulation_plan_desired_cube_rotation_vector_rad": np.zeros(
                    (21, 3), dtype=np.float64
                ),
                "planned_feedforward_target_rad": np.zeros(
                    (total_steps, model.nu), dtype=np.float64
                ),
                "feedback_correction_rad": np.zeros(
                    (total_steps, model.nu), dtype=np.float64
                ),
                "contact_force_target_n": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "contact_force_filtered_n": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "contact_force_error_n": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "contact_force_integral_n_s": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "contact_loss_run_steps": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.int64
                ),
                "contact_progress_frozen": np.zeros(total_steps, dtype=bool),
                "contact_recovery_active": np.zeros(total_steps, dtype=bool),
                "planned_knot_index": np.zeros(total_steps, dtype=np.int64),
                "desired_cube_position_delta_m": np.zeros(
                    (total_steps, 3), dtype=np.float64
                ),
                "desired_cube_rotation_vector_rad": np.zeros(
                    (total_steps, 3), dtype=np.float64
                ),
                "actual_cube_position_delta_m": np.zeros(
                    (total_steps, 3), dtype=np.float64
                ),
                "actual_cube_rotation_vector_rad": np.zeros(
                    (total_steps, 3), dtype=np.float64
                ),
                "operation_feedback_source_step": np.full(
                    total_steps, -1, dtype=np.int64
                ),
            }
        )
        if schema_version >= 15 or int(contact_feedback_schema_version) >= 2:
            traces.update(
                {
                    # Current post-step slip observations.  Commands at t use
                    # only the values persisted at t-1.
                    "online_contact_tangent_slip_m": np.zeros(
                        (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                    ),
                    "online_contact_tangent_slip_valid": np.zeros(
                        (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                    ),
                    "contact_tangent_slip_freeze_risk": np.zeros(
                        (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                    ),
                    "contact_tangent_slip_abort_risk": np.zeros(
                        (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                    ),
                }
            )
    if schema_version >= 15:
        traces.update(
            {
                "joint_pair_alignment_id": np.asarray("", dtype=np.str_),
                "joint_pair_feedback_id": np.asarray("", dtype=np.str_),
                "joint_pair_joint_names": np.asarray(("", ""), dtype=np.str_),
                "joint_pair_vector_cube_m": np.zeros(
                    (total_steps, 3), dtype=np.float64
                ),
                "joint_pair_residual": np.zeros(
                    (total_steps, 2), dtype=np.float64
                ),
                "joint_pair_angle_deg": np.full(
                    total_steps, 180.0, dtype=np.float64
                ),
                "joint_pair_length_m": np.zeros(total_steps, dtype=np.float64),
                "joint_pair_positive_y": np.zeros(total_steps, dtype=bool),
                "joint_pair_valid": np.zeros(total_steps, dtype=bool),
                "joint_pair_alignment_safe": np.zeros(total_steps, dtype=bool),
                "joint_pair_freeze_risk": np.zeros(total_steps, dtype=bool),
                "joint_pair_abort_risk": np.zeros(total_steps, dtype=bool),
                "joint_pair_violation_run_steps": np.zeros(
                    total_steps, dtype=np.int64
                ),
                "joint_pair_progress_frozen": np.zeros(total_steps, dtype=bool),
                "joint_pair_slip_recovery_active": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
                "joint_pair_alignment_request_rad": np.zeros(
                    (total_steps, model.nu), dtype=np.float64
                ),
                "joint_pair_slip_recovery_correction_rad": np.zeros(
                    (total_steps, model.nu), dtype=np.float64
                ),
                "joint_pair_feedback_correction_rad": np.zeros(
                    (total_steps, model.nu), dtype=np.float64
                ),
                "joint_pair_feedback_velocity_rad_s": np.zeros(
                    (total_steps, model.nu), dtype=np.float64
                ),
                "joint_pair_feedback_saturated": np.zeros(
                    total_steps, dtype=bool
                ),
                "joint_pair_feedback_source_step": np.full(
                    total_steps, -1, dtype=np.int64
                ),
                "joint_pair_active_residual_jacobian_2x8": np.zeros(
                    (total_steps, 2, len(ACTIVE_ACTUATORS)), dtype=np.float64
                ),
                "joint_pair_plan_residual_jacobian_2x8": np.zeros(
                    (21, 2, len(ACTIVE_ACTUATORS)), dtype=np.float64
                ),
                "joint_pair_plan_object_jacobian_6x8": np.zeros(
                    (21, 6, len(ACTIVE_ACTUATORS)), dtype=np.float64
                ),
                "joint_pair_plan_force_jacobian_3x8": np.zeros(
                    (21, 3, len(ACTIVE_ACTUATORS)), dtype=np.float64
                ),
                "joint_pair_abort_reason_step": np.full(
                    total_steps, "", dtype="<U40"
                ),
                "active_finger_self_collision": np.zeros(
                    total_steps, dtype=bool
                ),
                "active_finger_self_collision_contact_count": np.zeros(
                    total_steps, dtype=np.int64
                ),
                "active_finger_self_collision_normal_force_n": np.zeros(
                    total_steps, dtype=np.float64
                ),
                "active_finger_self_collision_max_penetration_m": np.zeros(
                    total_steps, dtype=np.float64
                ),
                "active_finger_self_collision_pairs": np.full(
                    total_steps, "", dtype="<U512"
                ),
            }
        )
    if schema_version >= 16:
        traces.update(
            {
                "rolling_contact_slip_schema_version": np.asarray(
                    ROLLING_CONTACT_SLIP_SCHEMA_VERSION, dtype=np.int64
                ),
                "rolling_contact_target_faces": np.asarray(
                    ("", "", ""), dtype=np.str_
                ),
                "native_tactile_target_face_effective": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
                "rolling_aware_target_face_effective": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
                "rolling_signed_tangent_displacement_m": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS), 2), dtype=np.float64
                ),
                "rolling_cumulative_irrecoverable_slip_m": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "rolling_relative_tangent_velocity_m_s": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS), 2), dtype=np.float64
                ),
                "rolling_contact_normal_force_n": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "rolling_contact_valid": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
                "rolling_contact_continuous": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
                "rolling_detected": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
                "rolling_force_fraction": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "rolling_patch_switch": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
                "rolling_centroid_jump": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
                "rolling_centroid_tangent_step_m": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "rolling_matched_patch_count": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.int64
                ),
                "rolling_new_patch_count": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.int64
                ),
                "rolling_dropped_patch_count": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.int64
                ),
                "rolling_patch_switch_count": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.int64
                ),
                "rolling_tangent_jacobian_m_per_rad": np.zeros(
                    (
                        total_steps,
                        len(ACTIVE_FINGERS),
                        2,
                        len(ACTIVE_ACTUATORS),
                    ),
                    dtype=np.float64,
                ),
                "rolling_tangent_jacobian_force_n": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "rolling_tangent_jacobian_valid": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
                # Controller state is recorded separately from the raw
                # post-step observation.  Commands at t use the observation
                # at t-1, so this source axis makes causal replay auditable.
                "rolling_slip_feedback_source_step": np.full(
                    total_steps, -1, dtype=np.int64
                ),
                "rolling_slip_filtered_velocity_m_s": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS), 2), dtype=np.float64
                ),
                "rolling_slip_predicted_displacement_m": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS), 2), dtype=np.float64
                ),
                "rolling_slip_predicted_magnitude_m": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "rolling_slip_controller_cumulative_m": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.float64
                ),
                "rolling_slip_observation_valid": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
                "rolling_slip_recovery_active": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
                "rolling_slip_freeze_active": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
                "rolling_slip_abort_risk": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=bool
                ),
                "rolling_slip_exit_run_steps": np.zeros(
                    (total_steps, len(ACTIVE_FINGERS)), dtype=np.int64
                ),
                "rolling_slip_request_rad": np.zeros(
                    (total_steps, model.nu), dtype=np.float64
                ),
                "rolling_slip_correction_rad": np.zeros(
                    (total_steps, model.nu), dtype=np.float64
                ),
                "rolling_slip_correction_velocity_rad_s": np.zeros(
                    (total_steps, model.nu), dtype=np.float64
                ),
                "rolling_slip_active_jacobian_m_per_rad": np.zeros(
                    (
                        total_steps,
                        len(ACTIVE_FINGERS),
                        2,
                        len(ACTIVE_ACTUATORS),
                    ),
                    dtype=np.float64,
                ),
                "rolling_slip_feedback_saturated": np.zeros(
                    total_steps, dtype=bool
                ),
            }
        )
    return traces


def _populate_derived_v2_trace_fields(
    config: dict[str, Any], traces: dict[str, np.ndarray]
) -> None:
    from .evaluation import face_from_label

    target_indices = np.asarray(
        [
            FACE_ORDER.index(
                face_from_label(config["contact_topology"]["target_faces"][finger])
            )
            for finger in ACTIVE_FINGERS
        ],
        dtype=int,
    )
    finger_index = np.arange(3)
    face_force = traces["distal_face_force_n"]
    target_force = face_force[:, finger_index, target_indices]
    total_force = np.sum(face_force, axis=2)
    purity = np.divide(
        target_force,
        total_force,
        out=np.zeros_like(target_force),
        where=total_force > 0.0,
    )
    traces["target_face_force_purity"][:] = purity
    force_min = float(config["acceptance"]["contact_force_min_n"])
    touch_min = float(config["acceptance"]["touch_force_min_n"])
    fraction = float(config["contact_topology"]["target_force_fraction"])
    effective = (
        (target_force >= force_min)
        & (traces["tactile_max"][:, :3] >= touch_min)
        & (purity >= fraction)
    )
    traces["target_face_topology"][:] = np.all(effective, axis=1)


def _populate_derived_v13_contact_slip_trace_fields(
    model: mujoco.MjModel,
    config: dict[str, Any],
    traces: dict[str, np.ndarray],
) -> None:
    """Populate the persisted schema-v13 soft rolling/slip diagnostics."""

    from .contact_slip import contact_tangent_slip_from_grasp

    plan = contact_point_plan_from_config(config)
    total_steps = int(np.asarray(traces["time"]).shape[0])
    effective = np.zeros((total_steps, len(ACTIVE_FINGERS)), dtype=bool)
    target_force = np.zeros((total_steps, len(ACTIVE_FINGERS)), dtype=np.float64)
    for step in range(total_steps):
        evidence = compute_target_face_evidence(
            config,
            np.asarray(traces["distal_face_force_n"][step], dtype=np.float64),
            np.asarray(traces["active_nondistal_force_n"][step], dtype=np.float64),
            np.asarray(traces["tactile_max"][step, :3], dtype=np.float64),
        )
        effective[step] = evidence.target_face_effective
        target_force[step] = evidence.target_force_n

    acquisition_step = int(
        np.asarray(traces["grasp_acquisition_step"]).reshape(-1)[0]
    )
    stable_steps = int(
        round(
            float(config["control_protocol"]["stable_window_s"])
            / float(model.opt.timestep)
        )
    )
    slip = contact_tangent_slip_from_grasp(
        np.asarray(
            traces["target_face_contact_centroid_cube_local_m"],
            dtype=np.float64,
        ),
        np.asarray(traces["target_face_contact_centroid_valid"], dtype=bool),
        effective,
        target_force,
        plan.target_faces,
        acquisition_start_step=acquisition_step - stable_steps + 1,
        acquisition_end_step=acquisition_step,
    )
    traces["grasp_contact_centroid_baseline_cube_local_m"][:] = (
        slip.baseline_centroid_cube_local_m
    )
    traces["grasp_contact_centroid_baseline_valid"][:] = slip.baseline_valid
    traces["grasp_contact_centroid_baseline_force_sum_n"][:] = (
        slip.baseline_force_sum_n
    )
    traces["target_contact_tangent_slip_from_grasp_m"][:] = (
        slip.tangent_slip_from_grasp_m
    )
    traces["target_contact_tangent_slip_from_grasp_valid"][:] = (
        slip.tangent_slip_valid
    )


def _rolling_contact_slip_summary(
    config: Mapping[str, Any], traces: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Summarize schema-v16 material slip without changing legacy evaluation."""

    states = np.asarray(traces["control_state"]).astype(str)
    operation_mask = np.isin(
        states, (ControlState.MANIPULATE.value, ControlState.HOLD.value)
    )
    cumulative = np.asarray(
        traces["rolling_cumulative_irrecoverable_slip_m"], dtype=np.float64
    )
    signed = np.asarray(
        traces["rolling_signed_tangent_displacement_m"], dtype=np.float64
    )
    velocity = np.asarray(
        traces["rolling_relative_tangent_velocity_m_s"], dtype=np.float64
    )
    valid = np.asarray(traces["rolling_contact_valid"], dtype=bool)
    continuous = np.asarray(traces["rolling_contact_continuous"], dtype=bool)
    rolling = np.asarray(traces["rolling_detected"], dtype=bool)
    patch_switch = np.asarray(traces["rolling_patch_switch"], dtype=bool)
    centroid_jump = np.asarray(traces["rolling_centroid_jump"], dtype=bool)
    centroid_step = np.asarray(
        traces["rolling_centroid_tangent_step_m"], dtype=np.float64
    )
    patch_switch_count = np.asarray(
        traces["rolling_patch_switch_count"], dtype=np.int64
    )
    jacobian_valid = np.asarray(
        traces["rolling_tangent_jacobian_valid"], dtype=bool
    )
    predicted_magnitude = np.asarray(
        traces["rolling_slip_predicted_magnitude_m"], dtype=np.float64
    )
    recovery_active = np.asarray(
        traces["rolling_slip_recovery_active"], dtype=bool
    )
    freeze_active = np.asarray(
        traces["rolling_slip_freeze_active"], dtype=bool
    )
    abort_risk = np.asarray(traces["rolling_slip_abort_risk"], dtype=bool)
    feedback_saturated = np.asarray(
        traces["rolling_slip_feedback_saturated"], dtype=bool
    )
    native_effective = np.asarray(
        traces["native_tactile_target_face_effective"], dtype=bool
    )
    rolling_aware_effective = np.asarray(
        traces["rolling_aware_target_face_effective"], dtype=bool
    )
    operation_indices = np.flatnonzero(operation_mask)
    feedback = config["joint_pair_feedback"]
    freeze_threshold = float(feedback["slip_freeze_threshold_m"])
    abort_threshold = float(feedback["slip_abort_threshold_m"])

    def duty(values: np.ndarray, finger_index: int) -> float:
        if operation_indices.size == 0:
            return 0.0
        return float(np.mean(values[operation_mask, finger_index]))

    per_finger: dict[str, Any] = {}
    for finger_index, finger in enumerate(ACTIVE_FINGERS):
        if operation_indices.size:
            op_cumulative = cumulative[operation_mask, finger_index]
            op_signed = signed[operation_mask, finger_index]
            speed = np.linalg.norm(
                velocity[operation_mask, finger_index], axis=1
            )
            valid_speed = speed[valid[operation_mask, finger_index]]
            final_index = int(operation_indices[-1])
            final_cumulative = float(cumulative[final_index, finger_index])
            final_signed = signed[final_index, finger_index].tolist()
            maximum_cumulative = float(np.max(op_cumulative))
            maximum_centroid_step = float(
                np.max(centroid_step[operation_mask, finger_index])
            )
            final_switch_count = int(
                patch_switch_count[final_index, finger_index]
            )
            maximum_predicted = float(
                np.max(predicted_magnitude[operation_mask, finger_index])
            )
        else:
            valid_speed = np.asarray([], dtype=np.float64)
            final_cumulative = 0.0
            final_signed = [0.0, 0.0]
            maximum_cumulative = 0.0
            maximum_centroid_step = 0.0
            final_switch_count = 0
            maximum_predicted = 0.0
        speed_percentiles = (
            np.percentile(valid_speed, (50.0, 95.0)).tolist()
            if valid_speed.size
            else [0.0, 0.0]
        )
        maximum_speed = float(np.max(valid_speed)) if valid_speed.size else 0.0
        per_finger[finger] = {
            "target_face": str(
                np.asarray(traces["rolling_contact_target_faces"]).astype(str)[
                    finger_index
                ]
            ),
            "final_signed_tangent_displacement_m": final_signed,
            "final_cumulative_irrecoverable_slip_m": final_cumulative,
            "maximum_cumulative_irrecoverable_slip_m": maximum_cumulative,
            "relative_tangent_speed_m_s": {
                "p50": float(speed_percentiles[0]),
                "p95": float(speed_percentiles[1]),
                "maximum": maximum_speed,
            },
            "valid_duty": duty(valid, finger_index),
            "continuous_duty": duty(continuous, finger_index),
            "rolling_detected_duty": duty(rolling, finger_index),
            "patch_switch_count_in_operation": int(
                np.sum(patch_switch[operation_mask, finger_index])
            ),
            "centroid_jump_count_in_operation": int(
                np.sum(centroid_jump[operation_mask, finger_index])
            ),
            "maximum_centroid_tangent_step_m": maximum_centroid_step,
            "final_patch_switch_count": final_switch_count,
            "tangent_jacobian_valid_duty": duty(
                jacobian_valid, finger_index
            ),
            "maximum_predicted_signed_slip_magnitude_m": maximum_predicted,
            "recovery_active_duty": duty(recovery_active, finger_index),
            "freeze_active_duty": duty(freeze_active, finger_index),
            "abort_risk_duty": duty(abort_risk, finger_index),
            "native_tactile_target_face_effective_duty": duty(
                native_effective, finger_index
            ),
            "rolling_aware_target_face_effective_duty": duty(
                rolling_aware_effective, finger_index
            ),
            "physical_contact_rescue_sample_count": int(
                np.sum(
                    operation_mask
                    & ~native_effective[:, finger_index]
                    & rolling_aware_effective[:, finger_index]
                )
            ),
            "below_freeze_threshold": bool(
                maximum_cumulative <= freeze_threshold + 1e-12
            ),
            "below_abort_threshold": bool(
                maximum_cumulative <= abort_threshold + 1e-12
            ),
        }

    maxima = [
        float(value["maximum_cumulative_irrecoverable_slip_m"])
        for value in per_finger.values()
    ]
    return {
        "schema_version": ROLLING_CONTACT_SLIP_SCHEMA_VERSION,
        "measurement": "integrated_material_point_relative_tangent_velocity",
        "target_face_effective_policy": (
            "VERIFY=native_tactile; MANIPULATE/HOLD=(native_tactile OR "
            "real_distal_target_face) AND force/purity/no_offtarget/no_nondistal"
        ),
        "operation_reset": "first_MANIPULATE_or_HOLD_post_step_sample",
        "operation_sample_count": int(operation_indices.size),
        "slip_freeze_threshold_m": freeze_threshold,
        "slip_abort_threshold_m": abort_threshold,
        "maximum_cumulative_irrecoverable_slip_m": max(maxima, default=0.0),
        "all_fingers_below_freeze_threshold": bool(
            all(value["below_freeze_threshold"] for value in per_finger.values())
        ),
        "all_fingers_below_abort_threshold": bool(
            all(value["below_abort_threshold"] for value in per_finger.values())
        ),
        "feedback_saturated_duty": (
            float(np.mean(feedback_saturated[operation_mask]))
            if operation_indices.size
            else 0.0
        ),
        "per_finger": per_finger,
    }


def _v16_target_face_effective(
    config: Mapping[str, Any],
    command_state: ControlState,
    target: TargetFaceEvidence,
    rolling: RollingAwareSlipEstimate,
) -> np.ndarray:
    """Return v16 operation contact evidence while preserving VERIFY gates.

    Native tactile evidence remains authoritative before manipulation.  During
    MANIPULATE/HOLD a real distal collision patch on the requested face may
    replace a missing taxel signal, but it may not bypass force, purity,
    off-target, or active non-distal safety predicates.
    """

    native = np.asarray(target.target_face_effective, dtype=bool)
    if command_state not in (ControlState.MANIPULATE, ControlState.HOLD):
        return native.copy()
    gate = config["control_protocol"]["grasp_gate"]
    force_min = float(gate["min_target_face_force_n"])
    purity_min = float(gate["min_target_force_fraction"])
    strict_contact = (
        (np.asarray(target.target_force_n, dtype=np.float64) >= force_min)
        & (
            np.asarray(target.target_force_purity, dtype=np.float64)
            >= purity_min
        )
        & ~np.asarray(target.material_off_target, dtype=bool)
        & ~np.asarray(target.material_active_nondistal, dtype=bool)
    )
    real_distal_target_face = (
        np.asarray(rolling.valid, dtype=bool)
        & (np.asarray(rolling.normal_force_n, dtype=np.float64) >= force_min)
    )
    return strict_contact & (native | real_distal_target_face)


def _v4_alignment_snapshot(
    model: mujoco.MjModel,
    config: dict[str, Any],
    contact: ContactSnapshot,
    tactile_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Derive one schema-v4 centroid/spread gate from raw contact evidence."""

    from .evaluation import face_from_label

    assert contact.distal_face_force_n is not None
    assert contact.active_nondistal_force_n is not None
    assert contact.distal_face_position_moment_n_m is not None
    target_faces = tuple(
        face_from_label(config["contact_topology"]["target_faces"][finger])
        for finger in ACTIVE_FINGERS
    )
    centroids, valid = target_face_contact_centroids(
        contact.distal_face_force_n,
        contact.distal_face_position_moment_n_m,
        target_faces,
    )
    spread, spread_valid = three_finger_height_spread(
        centroids,
        valid,
        model.opt.gravity,
    )
    target = compute_target_face_evidence(
        config,
        contact.distal_face_force_n,
        contact.active_nondistal_force_n,
        tactile_max[:3],
    )
    aligned = bool(
        spread_valid
        and np.all(target.target_face_effective)
        and spread
        <= float(config["contact_alignment"]["max_height_spread_m"]) + 1e-12
    )
    return centroids, valid, spread, aligned


@dataclass(frozen=True)
class SimulationStep:
    """One completed physics step from :class:`SimulationSession`."""

    index: int
    time_s: float
    control_state: str | None


class SimulationSession:
    """Incremental deterministic simulation shared by batch and GUI runners.

    ``advance_one`` always performs the same sequence as the historical batch
    loops: command, ``mj_step``, ``mj_forward``, contact/gate observation, then
    trace persistence.  The public session lets the passive Viewer pace those
    exact steps without owning or mutating the physics ``MjData``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        video_path: str | Path | None = None,
        contact_environment: ContactEnvironmentSpec | Mapping[str, Any] | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.schema_version = int(self.config.get("schema_version", 1))
        if contact_environment is None:
            self.contact_environment: ContactEnvironmentSpec | None = None
        elif isinstance(contact_environment, ContactEnvironmentSpec):
            self.contact_environment = contact_environment
        else:
            self.contact_environment = ContactEnvironmentSpec.from_config(
                contact_environment
            )
        self.contact_environment_evidence: dict[str, Any] | None = None
        self._contact_environment_runtime_first: dict[str, Any] | None = None
        self._contact_environment_runtime_last: dict[str, Any] | None = None
        self._contact_environment_runtime_contact_steps = 0
        self._contact_environment_runtime_other_geoms: set[str] = set()
        if self.contact_environment is None:
            self.model, self.info = build_model(self.config)
        else:
            reference_model, reference_info = build_model(self.config)
            self.model, self.info = build_model(self.config)
            compiled = apply_contact_environment_to_model(
                self.model,
                self.info.cube_geom_id,
                self.contact_environment,
            )
            audit = audit_allowed_model_changes(
                reference_model,
                self.model,
                reference_info.cube_geom_id,
                candidate_cube_geom_id=self.info.cube_geom_id,
                allowed_environment_fields=CONTACT_ENVIRONMENT_FIELD_PATHS,
            )
            audit.assert_passed()
            self.contact_environment_evidence = {
                "requested": requested_environment_snapshot(
                    self.contact_environment
                ),
                "compiled": compiled,
                "model_change_audit": audit.as_mapping(),
            }
        self.data = mujoco.MjData(self.model)
        self.reader = TactileReader(self.model, self.data, "left")
        self.pad_assignment_distance_m = 0.006
        self.thumb_bend_actuator_id = -1
        if self.schema_version >= 5:
            fingertip_contact = self.config.get("fingertip_contact_preferences", {})
            self.pad_assignment_distance_m = float(
                fingertip_contact.get("taxel_assignment_max_distance_m", 0.006)
            )
            if (
                not np.isfinite(self.pad_assignment_distance_m)
                or self.pad_assignment_distance_m <= 0.0
            ):
                raise ValueError(
                    "fingertip_contact_preferences."
                    "taxel_assignment_max_distance_m must be positive and finite"
                )
            self.thumb_bend_actuator_id = self.model.actuator(
                "left_hand_thumb_bend_joint_actuator"
            ).id
        self.phase_steps = _phase_steps(self.model, self.config)
        self._video_path = Path(video_path) if video_path is not None else None
        self._recorder: VideoRecorder | None = None
        self._summary: dict[str, Any] | None = None
        self._closed = False

        self.box_thresholds = DEFAULT_BOX_CONTACT_THRESHOLDS
        if self.schema_version >= 2:
            topology = self.config["contact_topology"]
            self.box_thresholds = BoxContactThresholds(
                surface_tolerance_m=float(topology["surface_tolerance_m"]),
                edge_margin_m=float(topology["edge_margin_m"]),
                normal_alignment_min=float(topology["min_normal_alignment"]),
            )

        self.controller: GraspVerifyThenManipulateController | None = None
        self.contact_point_plan: ContactPointPlan | None = None
        self.joint_pair_binding: JointPairBinding | None = None
        if self.schema_version >= 12:
            self.contact_point_plan = contact_point_plan_from_config(self.config)
        if self.schema_version >= 15:
            self.joint_pair_binding = resolve_joint_pair(
                self.model,
                self.config["joint_pair_alignment"]["joint_names"],
            )
            if self.joint_pair_binding is None:
                raise ValueError("schema-v15 requires a configured joint pair")
        self.rolling_slip_estimator: RollingAwareContactSlipEstimator | None = None
        self._rolling_finger_geom_to_index: dict[int, int] = {}
        self._rolling_active_dof_adrs = np.asarray([], dtype=np.int64)
        self._rolling_target_faces: tuple[str, ...] = ()
        self._rolling_operation_started = False
        if self.schema_version >= 16:
            feedback = self.config["joint_pair_feedback"]
            if int(feedback.get("schema_version", -1)) != 2:
                raise ValueError(
                    "schema-v16 requires joint_pair_feedback schema version 2"
                )
            self._rolling_target_faces = tuple(
                str(self.config["contact_topology"]["target_faces"][finger])
                for finger in ACTIVE_FINGERS
            )
            settings = RollingSlipSettings(
                minimum_sample_normal_force_n=1e-8,
                minimum_total_normal_force_n=float(
                    self.config["acceptance"]["contact_force_min_n"]
                ),
                minimum_normal_alignment=float(
                    self.config["contact_topology"]["min_normal_alignment"]
                ),
                maximum_patch_match_distance_m=float(
                    feedback["maximum_patch_match_distance_m"]
                ),
                centroid_jump_threshold_m=float(
                    feedback["centroid_jump_diagnostic_threshold_m"]
                ),
                maximum_time_gap_s=float(
                    feedback["maximum_contact_time_gap_s"]
                ),
            )
            self.rolling_slip_estimator = RollingAwareContactSlipEstimator(
                self._rolling_target_faces, settings=settings
            )
            distal_geoms = distal_collision_geom_ids(
                self.model, self.info.distal_weld_ids
            )
            self._rolling_finger_geom_to_index = {
                int(geom_id): finger_index
                for finger_index, finger in enumerate(ACTIVE_FINGERS)
                for geom_id in distal_geoms[finger]
            }
            self._rolling_active_dof_adrs = np.asarray(
                self.info.actuator_dof_adrs[self.info.active_actuator_ids],
                dtype=np.int64,
            )
        self.pregrasp_target: np.ndarray | None = None
        self.final_target: np.ndarray | None = None
        if self.schema_version >= 3:
            self.total_steps = sum(self.phase_steps.values())
        else:
            self.total_steps = sum(self.phase_steps.values())
            self.pregrasp_target = actuator_target_vector(
                self.model, self.config["control"]["pregrasp_targets_rad"]
            )
            self.final_target = actuator_target_vector(
                self.model, self.config["control"]["final_targets_rad"]
            )

        self.step_index = 0
        self.initial_cube_z = 0.0
        self.traces: dict[str, np.ndarray] = {}
        self._initial_pose_history: InitialPoseHistoryEvidence | None = None
        self._pose_preservation_latched = True
        self._support_retained_latched = True
        self._settle_hand_contact_free_latched = True
        self._last_close_progress = np.zeros(self.model.nu, dtype=np.float64)
        self.reset()
        if self._video_path is not None:
            self._recorder = VideoRecorder(
                self.model, self._video_path, VideoSettings()
            )

    @property
    def complete(self) -> bool:
        return self.step_index >= self.total_steps

    @property
    def remaining_steps(self) -> int:
        return max(0, self.total_steps - self.step_index)

    def reset(self) -> None:
        """Restore the exact initial state and discard this run's trace."""

        if self._recorder is not None and self.step_index > 0:
            raise RuntimeError("a video-recording SimulationSession cannot be reset")
        if self._closed:
            raise RuntimeError("cannot reset a closed SimulationSession")
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        initialize_pregrasp = bool(
            self.schema_version >= 6
            and self.config["pose_preservation"][
                "initialize_active_joints_at_pregrasp"
            ]
        )
        if initialize_pregrasp:
            initial_target = actuator_target_vector(
                self.model,
                precontact_targets(self.config),
            )
            self.data.qpos[self.info.actuator_qpos_adrs] = initial_target
            self.data.ctrl[:] = initial_target
        mujoco.mj_forward(self.model, self.data)
        self.initial_cube_z = float(self.data.xpos[self.info.cube_body_id, 2])
        self.step_index = 0
        self._summary = None
        if self.schema_version >= 3:
            self.controller = build_grasp_controller(
                self.model, self.config, self.phase_steps
            )
            self.total_steps = self.controller.total_steps
        self.traces = _allocate_traces(
            self.model,
            self.total_steps,
            schema_version=self.schema_version,
            contact_feedback_schema_version=int(
                self.config.get("contact_feedback", {}).get("schema_version", 1)
            ),
        )
        if self.contact_environment is not None:
            self.traces["contact_environment_id"] = np.asarray(
                self.contact_environment.environment_id, dtype=np.str_
            )
            self.traces["contact_environment_cube_contact_count"] = np.zeros(
                self.total_steps, dtype=np.int64
            )
            self.traces[
                "contact_environment_active_distal_contact_count"
            ] = np.zeros(self.total_steps, dtype=np.int64)
            self._contact_environment_runtime_first = None
            self._contact_environment_runtime_last = None
            self._contact_environment_runtime_contact_steps = 0
            self._contact_environment_runtime_other_geoms.clear()
        self._initial_pose_history = None
        self._pose_preservation_latched = True
        self._support_retained_latched = True
        self._settle_hand_contact_free_latched = True
        self._last_close_progress = np.zeros(self.model.nu, dtype=np.float64)
        self._rolling_operation_started = False
        if self.rolling_slip_estimator is not None:
            self.rolling_slip_estimator.reset()
            self.traces["rolling_contact_target_faces"][:] = np.asarray(
                self._rolling_target_faces, dtype=np.str_
            )
        if self.schema_version >= 6:
            assert self.controller is not None
            initial_position = np.asarray(
                self.data.xpos[self.info.cube_body_id], dtype=np.float64
            ).copy()
            initial_quaternion = np.asarray(
                self.data.xquat[self.info.cube_body_id], dtype=np.float64
            ).copy()
            self.controller.latch_initial_pose(
                initial_position, initial_quaternion
            )
            self.traces["initial_cube_pos_m"][:] = initial_position
            self.traces["initial_cube_quat"][:] = initial_quaternion
            self.traces["initial_joint_qpos_rad"][:] = self.data.qpos[
                self.info.actuator_qpos_adrs
            ]
            self.traces["initialized_at_pregrasp"] = np.asarray(
                initialize_pregrasp, dtype=bool
            )
            self.traces["pregrasp_target_rad"][:] = (
                self.controller.pregrasp_target
            )
            if self.schema_version >= 9:
                self.traces["precontact_target_rad"][:] = (
                    self.controller.pregrasp_target
                )
                self.traces["grasp_pose_nominal_joint_qpos_rad"][:] = np.asarray(
                    [
                        float(
                            self.config["grasp_pose"][
                                "nominal_joint_qpos_rad"
                            ][name]
                        )
                        for name in ACTIVE_ACTUATORS
                    ],
                    dtype=np.float64,
                )
            if self.schema_version >= 11:
                relative = (
                    self.config.get("candidate_metadata", {}).get(
                        "relative_wrist_pose_search", {}
                    )
                    if isinstance(self.config.get("candidate_metadata"), dict)
                    else {}
                )
                self.traces["clockwise_orbit_deg"] = np.asarray(
                    float(relative.get("clockwise_orbit_deg", 0.0)),
                    dtype=np.float64,
                )
                self.traces["root_delta_cube_m"][:] = np.asarray(
                    relative.get("root_delta_cube_m", (0.0, 0.0, 0.0)),
                    dtype=np.float64,
                )
                self.traces["wrist_local_rotvec_deg"][:] = np.asarray(
                    relative.get("wrist_local_rotvec_deg", (0.0, 0.0, 0.0)),
                    dtype=np.float64,
                )
                cube_rotation = np.asarray(
                    self.data.xmat[self.info.cube_body_id], dtype=np.float64
                ).reshape(3, 3)
                root_rotation = np.asarray(
                    self.data.xmat[self.info.root_body_id], dtype=np.float64
                ).reshape(3, 3)
                self.traces["initial_root_position_cube_m"][:] = (
                    cube_rotation.T
                    @ (
                        np.asarray(
                            self.data.xpos[self.info.root_body_id],
                            dtype=np.float64,
                        )
                        - initial_position
                    )
                )
                self.traces["initial_cube_from_root_rotation"][:] = (
                    cube_rotation.T @ root_rotation
                )
                boundary = relative.get("boundary_rejection_counts", {})
                if isinstance(boundary, dict):
                    names = tuple(sorted(str(value) for value in boundary))
                    self.traces["relative_pose_boundary_rejection_names"] = (
                        np.asarray(names, dtype=np.str_)
                    )
                    self.traces["relative_pose_boundary_rejection_counts"] = (
                        np.asarray(
                            [int(boundary[name]) for name in names],
                            dtype=np.int64,
                        )
                    )
            if self.schema_version >= 12:
                assert self.contact_point_plan is not None
                self.traces["contact_point_plan_id"] = np.asarray(
                    self.contact_point_plan.point_plan_id, dtype=np.str_
                )
                self.traces["target_contact_points_cube_local_m"][:] = (
                    self.contact_point_plan.target_points_cube_local_m
                )
                self.traces["target_contact_point_radius_m"] = np.asarray(
                    self.contact_point_plan.target_radius_m, dtype=np.float64
                )
            if self.schema_version >= 14:
                if not isinstance(
                    self.controller, ContactPreservingPlannedLiftController
                ):
                    raise RuntimeError("schema-v14+ controller factory mismatch")
                if self.schema_version == 14:
                    for name, value in v14_identity_trace_values(
                        self.config
                    ).items():
                        self.traces[name] = np.asarray(value, dtype=np.str_)
                self.traces["manipulation_plan_knot_times_s"][:] = (
                    self.controller.knot_times_s
                )
                self.traces["manipulation_plan_waypoints_rad"][:] = (
                    self.controller.plan_waypoints
                )
                self.traces[
                    "manipulation_plan_desired_cube_position_delta_m"
                ][:] = self.controller.desired_cube_position_delta_m
                self.traces[
                    "manipulation_plan_desired_cube_rotation_vector_rad"
                ][:] = self.controller.desired_cube_rotation_vector_rad
            if self.schema_version >= 15:
                if not isinstance(
                    self.controller,
                    JointPairAlignedContactPreservingPlannedLiftController,
                ):
                    raise RuntimeError("schema-v15 controller factory mismatch")
                planned = self.controller
                self.traces["joint_pair_alignment_id"] = np.asarray(
                    planned.joint_pair_alignment_id, dtype=np.str_
                )
                self.traces["joint_pair_feedback_id"] = np.asarray(
                    planned.joint_pair_feedback_id, dtype=np.str_
                )
                self.traces["joint_pair_joint_names"][:] = np.asarray(
                    planned.joint_pair_joint_names, dtype=np.str_
                )
                self.traces["joint_pair_plan_residual_jacobian_2x8"][:] = (
                    planned.joint_pair_plan_residual_jacobian_active
                )
                self.traces["joint_pair_plan_object_jacobian_6x8"][:] = (
                    planned.joint_pair_plan_object_jacobian_active
                )
                self.traces["joint_pair_plan_force_jacobian_3x8"][:] = (
                    planned.joint_pair_plan_force_jacobian_active
                )
            self.traces["close_start_fraction"][:] = (
                self.controller.close_start_fraction
            )
            self.traces["close_end_fraction"][:] = (
                self.controller.close_end_fraction
            )

    def _rolling_contact_observation(
        self, command_state: ControlState
    ) -> tuple[RollingAwareSlipEstimate, RollingTangentJacobian]:
        """Observe schema-v16 material slip at the post-step sample boundary."""

        if self.rolling_slip_estimator is None:
            raise RuntimeError("schema-v16 rolling-slip estimator is missing")
        if (
            command_state in (ControlState.MANIPULATE, ControlState.HOLD)
            and not self._rolling_operation_started
        ):
            # Pre-contact motion is useful for diagnostics but must not enter
            # the manipulation slip budget.  The first operation observation
            # is the zero-displacement material reference.
            self.rolling_slip_estimator.reset()
            self._rolling_operation_started = True
        samples = mujoco_contact_patch_kinematics(
            self.model,
            self.data,
            cube_geom_id=self.info.cube_geom_id,
            finger_geom_to_index=self._rolling_finger_geom_to_index,
        )
        estimate = self.rolling_slip_estimator.update(
            float(self.data.time), samples
        )
        jacobian = mujoco_contact_tangent_position_jacobian(
            self.model,
            self.data,
            samples,
            cube_geom_id=self.info.cube_geom_id,
            target_faces=self._rolling_target_faces,
            active_dof_adrs=self._rolling_active_dof_adrs,
            minimum_total_normal_force_n=float(
                self.config["acceptance"]["contact_force_min_n"]
            ),
            minimum_normal_alignment=float(
                self.config["contact_topology"]["min_normal_alignment"]
            ),
        )
        return estimate, jacobian

    def _schema_v6_close_progress(
        self, step: int, state: ControlState
    ) -> np.ndarray:
        """Return the persisted per-actuator close interpolation fraction."""

        assert self.controller is not None
        if state is ControlState.SETTLE:
            progress = np.zeros(self.model.nu, dtype=np.float64)
        elif state is ControlState.CLOSE:
            fraction = (
                step - self.controller.settle_end + 1
            ) / self.phase_steps["close"]
            progress = np.zeros(self.model.nu, dtype=np.float64)
            for name in ACTIVE_ACTUATORS:
                actuator_id = self.model.actuator(name).id
                start = self.controller.close_start_fraction[actuator_id]
                end = self.controller.close_end_fraction[actuator_id]
                progress[actuator_id] = smoothstep(
                    (fraction - start) / (end - start)
                )
        elif state is ControlState.ABORT:
            progress = self._last_close_progress.copy()
        else:
            progress = np.zeros(self.model.nu, dtype=np.float64)
            progress[self.info.active_actuator_ids] = 1.0
        self._last_close_progress = progress.copy()
        return progress

    def _legacy_target(self, step: int) -> np.ndarray:
        assert self.pregrasp_target is not None
        assert self.final_target is not None
        settle_steps = self.phase_steps["settle"]
        pregrasp_steps = self.phase_steps["pregrasp"]
        lift_steps = self.phase_steps["lift"]
        if step < settle_steps:
            return np.zeros(self.model.nu)
        if step < settle_steps + pregrasp_steps:
            local_step = step - settle_steps + 1
            return smoothstep(local_step / pregrasp_steps) * self.pregrasp_target
        if step < settle_steps + pregrasp_steps + lift_steps:
            local_step = step - settle_steps - pregrasp_steps + 1
            alpha = smoothstep(local_step / lift_steps)
            return self.pregrasp_target + alpha * (
                self.final_target - self.pregrasp_target
            )
        return self.final_target

    def _finite_state(self) -> bool:
        return all(
            np.isfinite(values).all()
            for values in (
                self.data.qpos,
                self.data.qvel,
                self.data.qacc,
                self.data.ctrl,
                self.data.actuator_force,
                self.data.sensordata,
            )
        )

    def _store_common(
        self,
        step: int,
        contact: ContactSnapshot,
        tactile_max: np.ndarray,
        *,
        finite: bool,
    ) -> None:
        data = self.data
        info = self.info
        traces = self.traces
        traces["time"][step] = data.time
        traces["cube_pos"][step] = data.xpos[info.cube_body_id]
        traces["cube_quat"][step] = data.xquat[info.cube_body_id]
        traces["cube_velocity"][step] = data.qvel[
            info.cube_dof_adr : info.cube_dof_adr + 6
        ]
        traces["root_pos"][step] = data.xpos[info.root_body_id]
        traces["root_quat"][step] = data.xquat[info.root_body_id]
        traces["ctrl"][step] = data.ctrl
        traces["joint_qpos"][step] = data.qpos[info.actuator_qpos_adrs]
        traces["joint_qvel"][step] = data.qvel[info.actuator_dof_adrs]
        traces["actuator_force"][step] = data.actuator_force
        traces["finger_contact_force"][step] = contact.finger_forces
        traces["tactile_max"][step] = tactile_max
        traces["forbidden_contact"][step] = contact.forbidden
        traces["support_contact"][step] = contact.support
        traces["floor_contact"][step] = contact.floor
        traces["max_penetration"][step] = contact.max_penetration
        traces["friction_error"][step] = contact.friction_error
        traces["cube_contact_seen"][step] = contact.cube_contact_seen
        traces["contact_dim_ok"][step] = contact.contact_dim_ok
        traces["finite"][step] = finite
        if self.schema_version >= 2:
            traces["palm_down_angle_deg"][step] = _palm_down_angle_deg(
                self.model, data, info
            )
            assert contact.distal_face_force_n is not None
            assert contact.active_nondistal_force_n is not None
            traces["distal_face_force_n"][step] = contact.distal_face_force_n
            traces["active_nondistal_force_n"][step] = (
                contact.active_nondistal_force_n
            )
        if self.schema_version >= 4:
            assert contact.distal_face_position_moment_n_m is not None
            traces["distal_face_position_moment_n_m"][step] = (
                contact.distal_face_position_moment_n_m
            )
            root_rotation = data.xmat[info.root_body_id].reshape(3, 3)
            traces["finger_down_tilt_deg"][step] = (
                signed_finger_down_tilt_deg(
                    root_rotation,
                    self.model.opt.gravity,
                )
            )
        if self.schema_version >= 5:
            assert self.thumb_bend_actuator_id >= 0
            traces["root_cube_center_distance_m"][step] = float(
                np.linalg.norm(
                    data.xpos[info.cube_body_id] - data.xpos[info.root_body_id]
                )
            )
            traces["thumb_bend_command_rad"][step] = data.ctrl[
                self.thumb_bend_actuator_id
            ]
            traces["thumb_bend_qpos_rad"][step] = data.qpos[
                info.actuator_qpos_adrs[self.thumb_bend_actuator_id]
            ]
            assert contact.distal_pad_force_n is not None
            assert contact.distal_nonpad_force_n is not None
            assert contact.distal_pad_force_fraction is not None
            assert contact.distal_active_taxel_count is not None
            traces["distal_pad_force_n"][step] = contact.distal_pad_force_n
            traces["distal_nonpad_force_n"][step] = contact.distal_nonpad_force_n
            traces["distal_pad_force_fraction"][step] = (
                contact.distal_pad_force_fraction
            )
            traces["distal_active_taxel_count"][step] = (
                contact.distal_active_taxel_count
            )

    def advance_one(self) -> SimulationStep:
        """Advance exactly one configured control/physics/observation sample."""

        if self._closed:
            raise RuntimeError("cannot advance a closed SimulationSession")
        if self.complete:
            raise StopIteration("SimulationSession has completed")
        step = self.step_index
        command = None
        if self.schema_version >= 3:
            assert self.controller is not None
            command = self.controller.command(step)
            target = command.target
        else:
            target = self._legacy_target(step)
        self.data.ctrl[:] = target
        mujoco.mj_step(self.model, self.data)
        # Keep the historical sample boundary: all derived fields and sensors
        # correspond exactly to the post-step ``data.time`` persisted below.
        mujoco.mj_forward(self.model, self.data)

        if self.contact_environment is not None:
            runtime_environment = runtime_cube_contact_snapshot(
                self.model, self.data, self.info.cube_geom_id
            )
            cube_contact_count = int(runtime_environment["contact_count"])
            self.traces["contact_environment_cube_contact_count"][step] = (
                cube_contact_count
            )
            if cube_contact_count:
                verify_runtime_cube_contacts(
                    self.contact_environment,
                    runtime_environment,
                    require_contacts=True,
                )
                self._contact_environment_runtime_contact_steps += 1
                if self._contact_environment_runtime_first is None:
                    self._contact_environment_runtime_first = runtime_environment
                self._contact_environment_runtime_last = runtime_environment
                distal_weld_ids = set(self.info.distal_weld_ids.values())
                active_distal_count = 0
                for record in runtime_environment["contacts"]:
                    other_geom_id = int(record["other_geom_id"])
                    other_body_id = int(self.model.geom_bodyid[other_geom_id])
                    if int(self.model.body_weldid[other_body_id]) in distal_weld_ids:
                        active_distal_count += 1
                    self._contact_environment_runtime_other_geoms.add(
                        str(record["other_geom_name"])
                    )
                self.traces[
                    "contact_environment_active_distal_contact_count"
                ][step] = active_distal_count

        contact = contact_snapshot(
            self.model,
            self.data,
            self.info,
            classify_faces=self.schema_version >= 2,
            box_thresholds=self.box_thresholds,
            distal_taxel_site_ids=(
                self.reader.site_ids[: len(ACTIVE_FINGERS)]
                if self.schema_version >= 5
                else None
            ),
            max_taxel_assignment_distance_m=self.pad_assignment_distance_m,
        )
        active_finger_self_collision: dict[str, Any] | None = None
        if self.schema_version >= 15:
            active_finger_self_collision = (
                _active_finger_self_collision_snapshot(
                    self.model, self.data, self.info
                )
            )
        closure_snapshot = None
        if (
            self.schema_version >= 8
            and command is not None
            and command.state is ControlState.CLOSE
        ):
            closure_snapshot = closure_alignment_snapshot(
                self.model,
                self.data,
                self.info,
                self.config,
                command.target_velocity_rad_s,
                box_thresholds=self.box_thresholds,
            )
        tactile_max = self.reader.normal_forces().max(axis=1)
        finite = self._finite_state()
        alignment_snapshot = None
        if self.schema_version >= 4:
            alignment_snapshot = _v4_alignment_snapshot(
                self.model,
                self.config,
                contact,
                tactile_max,
            )
        point_observation: ContactPointObservation | None = None
        target_evidence = None
        if self.schema_version >= 12:
            assert self.contact_point_plan is not None
            assert alignment_snapshot is not None
            assert contact.distal_face_force_n is not None
            assert contact.active_nondistal_force_n is not None
            target_evidence = compute_target_face_evidence(
                self.config,
                contact.distal_face_force_n,
                contact.active_nondistal_force_n,
                tactile_max[:3],
            )
            point_observation = contact_point_observation(
                self.contact_point_plan,
                alignment_snapshot[0],
                alignment_snapshot[1],
                target_evidence.target_face_effective,
                np.asarray(
                    self.data.geom_xpos[self.info.cube_geom_id],
                    dtype=np.float64,
                ),
                np.asarray(
                    self.data.geom_xmat[self.info.cube_geom_id],
                    dtype=np.float64,
                ).reshape(3, 3),
            )

        joint_pair_observation: dict[str, Any] | None = None
        joint_pair_alignment_safe: bool | None = None
        if self.schema_version >= 15:
            if self.joint_pair_binding is None:
                raise RuntimeError("schema-v15 joint-pair binding is missing")
            joint_pair_observation = _directed_joint_pair_observation(
                self.data, self.joint_pair_binding
            )
            alignment = self.config["joint_pair_alignment"]
            joint_pair_alignment_safe = bool(
                joint_pair_observation["valid"]
                and joint_pair_observation["positive_y"]
                and float(joint_pair_observation["length_m"])
                >= float(alignment["minimum_length_m"]) - 1e-12
                and float(joint_pair_observation["angle_deg"])
                <= float(alignment["grasp_max_deg"]) + 1e-12
            )

        rolling_estimate: RollingAwareSlipEstimate | None = None
        rolling_jacobian: RollingTangentJacobian | None = None
        rolling_aware_target_face_effective: np.ndarray | None = None
        if self.schema_version >= 16:
            if command is None:
                raise RuntimeError("schema-v16 control command is missing")
            rolling_estimate, rolling_jacobian = (
                self._rolling_contact_observation(command.state)
            )
            if target_evidence is None:
                raise RuntimeError("schema-v16 target-face evidence is missing")
            rolling_aware_target_face_effective = _v16_target_face_effective(
                self.config,
                command.state,
                target_evidence,
                rolling_estimate,
            )

        if self.schema_version >= 3:
            assert self.controller is not None
            assert command is not None
            assert contact.distal_face_force_n is not None
            assert contact.active_nondistal_force_n is not None
            cube_position = np.asarray(
                self.data.xpos[self.info.cube_body_id]
            ).copy()
            cube_quaternion = np.asarray(
                self.data.xquat[self.info.cube_body_id]
            ).copy()
            cube_velocity = np.asarray(
                self.data.qvel[
                    self.info.cube_dof_adr : self.info.cube_dof_adr + 6
                ]
            ).copy()
            palm_angle = _palm_down_angle_deg(self.model, self.data, self.info)
            actuator_qpos = np.asarray(
                self.data.qpos[self.info.actuator_qpos_adrs]
            )
            limited_qpos = actuator_qpos[self.info.joint_limited]
            limited_ranges = self.info.joint_ranges[self.info.joint_limited]
            joint_limits_respected = bool(
                np.all(limited_qpos >= limited_ranges[:, 0] - 2e-3)
                and np.all(limited_qpos <= limited_ranges[:, 1] + 2e-3)
            )
            inactive_controls_zero = bool(
                np.all(self.data.ctrl[self.info.inactive_actuator_ids] == 0.0)
            )
            initial_pose_history_stable = None
            if self.schema_version >= 6:
                initial_position, initial_quaternion = (
                    self.controller.initial_pose_reference()
                )
                raw_initial_translation = float(
                    np.linalg.norm(cube_position - initial_position)
                )
                raw_initial_orientation = quaternion_drift_deg(
                    initial_quaternion, cube_quaternion
                )
                current_pose_ok = bool(
                    raw_initial_translation
                    <= float(
                        self.config["pose_preservation"][
                            "max_translation_m"
                        ]
                    )
                    + 1e-12
                    and raw_initial_orientation
                    <= float(
                        self.config["pose_preservation"][
                            "max_orientation_drift_deg"
                        ]
                    )
                    + 1e-12
                )
                # Once acquisition is latched the requested scope is complete;
                # later manipulation motion must not retroactively invalidate
                # the pre-grasp history.  The acquisition sample itself is
                # observed here before ``controller.observe`` sets the latch.
                if not self.controller.acquired:
                    self._initial_pose_history = (
                        self.controller.observe_initial_pose_history(
                            cube_position, cube_quaternion
                        )
                    )
                    self._pose_preservation_latched = bool(
                        self._pose_preservation_latched
                        and self._initial_pose_history.passed
                    )
                    if bool(
                        self.config["pose_preservation"][
                            "require_support_contact"
                        ]
                    ):
                        self._support_retained_latched = bool(
                            self._support_retained_latched and contact.support
                        )
                    if (
                        command.state is ControlState.SETTLE
                        and bool(
                            self.config["pose_preservation"][
                                "require_no_hand_cube_contact_during_settle"
                            ]
                        )
                    ):
                        self._settle_hand_contact_free_latched = bool(
                            self._settle_hand_contact_free_latched
                            and not contact.hand_cube_contact
                        )
                if self._initial_pose_history is None:
                    raise RuntimeError(
                        "schema-v6 initial pose history was not observed"
                    )
                initial_pose_history_stable = bool(
                    self._pose_preservation_latched
                    and self._support_retained_latched
                    and self._settle_hand_contact_free_latched
                )
            reference_position, reference_quaternion = (
                self.controller.stability_reference(
                    cube_position, cube_quaternion
                )
            )
            gate_evidence = compute_grasp_gate_evidence(
                self.config,
                distal_face_force_n=contact.distal_face_force_n,
                active_nondistal_force_n=contact.active_nondistal_force_n,
                tactile_force_n=tactile_max[:3],
                forbidden_contact=contact.forbidden,
                support_contact=contact.support,
                floor_contact=contact.floor,
                cube_position=cube_position,
                cube_quaternion=cube_quaternion,
                reference_position=reference_position,
                reference_quaternion=reference_quaternion,
                cube_linear_speed_m_s=float(np.linalg.norm(cube_velocity[:3])),
                early_lift_m=float(cube_position[2] - self.initial_cube_z),
                palm_down_angle_deg=palm_angle,
                max_penetration_m=contact.max_penetration,
                finite=finite,
                joint_limits_respected=joint_limits_respected,
                inactive_controls_zero=inactive_controls_zero,
                contact_height_aligned=(
                    alignment_snapshot[3]
                    if alignment_snapshot is not None
                    else None
                ),
                initial_pose_history_stable=initial_pose_history_stable,
                thumb_actual_qpos_within_range=(
                    bool(
                        float(
                            self.config["grasp_pose"][
                                "thumb_actual_range_rad"
                            ][0]
                        )
                        - 1e-12
                        <= float(
                            actuator_qpos[
                                self.model.actuator(
                                    "left_hand_thumb_bend_joint_actuator"
                                ).id
                            ]
                        )
                        <= float(
                            self.config["grasp_pose"][
                                "thumb_actual_range_rad"
                            ][1]
                        )
                        + 1e-12
                    )
                    if self.schema_version >= 9
                    else None
                ),
                contact_points_within_target_regions=(
                    point_observation.all_effective_within_radius
                    if point_observation is not None
                    else None
                ),
                joint_pair_alignment_safe=joint_pair_alignment_safe,
                active_finger_self_collision=(
                    bool(active_finger_self_collision["active"])
                    if active_finger_self_collision is not None
                    else None
                ),
            )
            operation_feedback = None
            if self.schema_version >= 14:
                if target_evidence is None or alignment_snapshot is None:
                    raise RuntimeError("schema-v14 contact observation is incomplete")
                operation_feedback = OperationFeedback(
                    target_force_n=target_evidence.target_force_n,
                    target_force_purity=target_evidence.target_force_purity,
                    # Preserve the native tactile-aware observation here.
                    # The schema-v16 controller owns the operation-only
                    # merge with strict physical rolling evidence.  Keeping
                    # the raw bit at this boundary makes substitution
                    # telemetry auditable and prevents VERIFY from ever
                    # inheriting an operation relaxation.
                    target_face_effective=target_evidence.target_face_effective,
                    material_off_target=target_evidence.material_off_target,
                    material_active_nondistal=(
                        target_evidence.material_active_nondistal
                    ),
                    tactile_force_n=tactile_max[:3],
                    contact_centroid_world_m=alignment_snapshot[0],
                    contact_centroid_valid=alignment_snapshot[1],
                    contact_centroid_cube_local_m=(
                        point_observation.centroid_cube_local_m
                        if point_observation is not None
                        else np.zeros((len(ACTIVE_FINGERS), 3), dtype=np.float64)
                    ),
                    cube_position_m=cube_position,
                    cube_quaternion_wxyz=cube_quaternion,
                    cube_velocity=cube_velocity,
                    joint_qpos_rad=actuator_qpos,
                    joint_qvel_rad_s=np.asarray(
                        self.data.qvel[self.info.actuator_dof_adrs],
                        dtype=np.float64,
                    ),
                    forbidden_contact=contact.forbidden,
                    max_penetration_m=contact.max_penetration,
                    finite=finite,
                    joint_limits_respected=joint_limits_respected,
                    inactive_controls_zero=inactive_controls_zero,
                    joint_pair_vector_cube_m=(
                        joint_pair_observation["vector_cube_m"]
                        if joint_pair_observation is not None
                        else np.zeros(3, dtype=np.float64)
                    ),
                    joint_pair_residual=(
                        joint_pair_observation["residual"]
                        if joint_pair_observation is not None
                        else np.zeros(2, dtype=np.float64)
                    ),
                    joint_pair_angle_deg=(
                        float(joint_pair_observation["angle_deg"])
                        if joint_pair_observation is not None
                        else 180.0
                    ),
                    joint_pair_length_m=(
                        float(joint_pair_observation["length_m"])
                        if joint_pair_observation is not None
                        else 0.0
                    ),
                    joint_pair_positive_y=(
                        bool(joint_pair_observation["positive_y"])
                        if joint_pair_observation is not None
                        else False
                    ),
                    joint_pair_valid=(
                        bool(joint_pair_observation["valid"])
                        if joint_pair_observation is not None
                        else False
                    ),
                    active_finger_self_collision=(
                        bool(active_finger_self_collision["active"])
                        if active_finger_self_collision is not None
                        else False
                    ),
                    rolling_signed_tangent_displacement_m=(
                        rolling_estimate.signed_tangent_displacement_m
                        if rolling_estimate is not None
                        else np.zeros((len(ACTIVE_FINGERS), 2), dtype=np.float64)
                    ),
                    rolling_cumulative_irrecoverable_slip_m=(
                        rolling_estimate.cumulative_irrecoverable_slip_m
                        if rolling_estimate is not None
                        else np.zeros(len(ACTIVE_FINGERS), dtype=np.float64)
                    ),
                    rolling_relative_tangent_velocity_m_s=(
                        rolling_estimate.relative_tangent_velocity_m_s
                        if rolling_estimate is not None
                        else np.zeros((len(ACTIVE_FINGERS), 2), dtype=np.float64)
                    ),
                    rolling_contact_valid=(
                        rolling_estimate.valid
                        if rolling_estimate is not None
                        else np.zeros(len(ACTIVE_FINGERS), dtype=bool)
                    ),
                    rolling_contact_continuous=(
                        rolling_estimate.continuous
                        if rolling_estimate is not None
                        else np.zeros(len(ACTIVE_FINGERS), dtype=bool)
                    ),
                    rolling_detected=(
                        rolling_estimate.rolling_detected
                        if rolling_estimate is not None
                        else np.zeros(len(ACTIVE_FINGERS), dtype=bool)
                    ),
                    rolling_patch_switch=(
                        rolling_estimate.patch_switch
                        if rolling_estimate is not None
                        else np.zeros(len(ACTIVE_FINGERS), dtype=bool)
                    ),
                    rolling_tangent_jacobian_m_per_rad=(
                        rolling_jacobian.position_jacobian_cube_local_m_per_rad
                        if rolling_jacobian is not None
                        else np.zeros(
                            (
                                len(ACTIVE_FINGERS),
                                2,
                                len(ACTIVE_ACTUATORS),
                            ),
                            dtype=np.float64,
                        )
                    ),
                    rolling_tangent_jacobian_valid=(
                        rolling_jacobian.valid
                        if rolling_jacobian is not None
                        else np.zeros(len(ACTIVE_FINGERS), dtype=bool)
                    ),
                )
            self.controller.observe(
                step,
                gate_evidence,
                cube_position,
                cube_quaternion,
                actual_joint_qpos_rad=(
                    actuator_qpos if self.schema_version >= 9 else None
                ),
                operation_feedback=operation_feedback,
            )

        self._store_common(
            step,
            contact,
            tactile_max,
            finite=finite,
        )
        if self.schema_version >= 3:
            assert self.controller is not None
            assert command is not None
            self.traces["control_state"][step] = command.state.value
            self.traces["grasp_gate"][step] = gate_evidence.components
            self.traces["grasp_gate_consecutive_steps"][step] = (
                self.controller.consecutive_steps
            )
            self.traces["grasp_acquired"][step] = self.controller.acquired
            self.traces["manipulation_progress"][step] = (
                command.manipulation_progress
            )
            self.traces["target_face_effective"][step] = (
                gate_evidence.target_faces.target_face_effective
            )
            if self.schema_version >= 6:
                assert self._initial_pose_history is not None
                self.traces["cube_translation_from_initial_m"][step] = (
                    raw_initial_translation
                )
                self.traces["cube_orientation_from_initial_deg"][step] = (
                    raw_initial_orientation
                )
                self.traces[
                    "initial_pose_translation_history_stable"
                ][step] = (
                    self._initial_pose_history.translation_history_stable
                )
                self.traces[
                    "initial_pose_orientation_history_stable"
                ][step] = (
                    self._initial_pose_history.orientation_history_stable
                )
                self.traces["pregrasp_pose_within_limit"][step] = (
                    current_pose_ok
                )
                self.traces["pregrasp_pose_preserved_latched"][step] = (
                    initial_pose_history_stable
                )
                self.traces["pregrasp_support_retained_latched"][step] = (
                    self._support_retained_latched
                )
                self.traces["settle_hand_contact_free_latched"][step] = (
                    self._settle_hand_contact_free_latched
                )
                self.traces["hand_cube_contact"][step] = (
                    contact.hand_cube_contact
                )
                self.traces["close_progress"][step] = (
                    self._schema_v6_close_progress(step, command.state)
                )
                first_contact = self.traces["first_distal_contact_step"]
                unseen = first_contact < 0
                newly_touching = contact.finger_forces > 0.0
                first_contact[unseen & newly_touching] = step
            if alignment_snapshot is not None:
                centroid, centroid_valid, height_spread, aligned = (
                    alignment_snapshot
                )
                self.traces["target_face_contact_centroid_world_m"][step] = (
                    centroid
                )
                self.traces["target_face_contact_centroid_valid"][step] = (
                    centroid_valid
                )
                self.traces["three_contact_height_spread_m"][step] = (
                    height_spread
                )
                self.traces["three_contact_height_aligned"][step] = aligned
            if point_observation is not None:
                self.traces[
                    "target_face_contact_centroid_cube_local_m"
                ][step] = point_observation.centroid_cube_local_m
                self.traces["target_contact_point_tangent_error_m"][step] = (
                    point_observation.tangent_error_m
                )
                self.traces["target_contact_point_within_radius"][step] = (
                    point_observation.within_radius
                )
            if self.schema_version >= 8:
                self.traces["command_target_velocity_rad_s"][step] = (
                    command.target_velocity_rad_s
                )
                if closure_snapshot is not None:
                    self.traces["closure_witness_world_m"][step] = (
                        closure_snapshot.witness_world_m
                    )
                    self.traces[
                        "closure_command_velocity_world_m_s"
                    ][step] = closure_snapshot.command_velocity_world_m_s
                    self.traces[
                        "closure_cube_outward_normal_world"
                    ][step] = closure_snapshot.cube_outward_normal_world
                    self.traces["closure_alignment_cosine"][step] = (
                        closure_snapshot.cosine
                    )
                    self.traces["closure_alignment_angle_deg"][step] = (
                        closure_snapshot.angle_deg
                    )
                    self.traces["closure_inward_speed_m_s"][step] = (
                        closure_snapshot.inward_speed_m_s
                    )
                    self.traces["closure_tangent_speed_m_s"][step] = (
                        closure_snapshot.tangent_speed_m_s
                    )
                    self.traces["closure_alignment_valid"][step] = (
                        closure_snapshot.valid
                    )
                    self.traces["closure_target_contact_force_n"][step] = (
                        closure_snapshot.target_contact_force_n
                    )
            if self.schema_version >= 9:
                active_ids = np.asarray(
                    [self.model.actuator(name).id for name in ACTIVE_ACTUATORS],
                    dtype=np.int64,
                )
                self.traces["grasp_pose_actual_joint_qpos_rad"][step] = (
                    actuator_qpos[active_ids]
                )
                thumb_id = self.model.actuator(
                    "left_hand_thumb_bend_joint_actuator"
                ).id
                lower, upper = (
                    float(value)
                    for value in self.config["grasp_pose"][
                        "thumb_actual_range_rad"
                    ]
                )
                self.traces["grasp_pose_thumb_actual_within_range"][step] = (
                    lower - 1e-12 <= actuator_qpos[thumb_id] <= upper + 1e-12
                )
            if self.schema_version >= 14:
                if not isinstance(
                    self.controller, ContactPreservingPlannedLiftController
                ):
                    raise RuntimeError("schema-v14+ controller factory mismatch")
                planned = self.controller
                self.traces["planned_feedforward_target_rad"][step] = (
                    planned.nominal_plan_target_rad
                )
                self.traces["feedback_correction_rad"][step] = (
                    planned.feedback_correction_rad
                )
                self.traces["contact_force_target_n"][step] = (
                    planned.force_targets_n
                )
                self.traces["contact_force_filtered_n"][step] = (
                    planned.filtered_force_n
                )
                self.traces["contact_force_error_n"][step] = (
                    planned.force_error_n
                )
                self.traces["contact_force_integral_n_s"][step] = (
                    planned.force_integral_n_s
                )
                self.traces["contact_loss_run_steps"][step] = (
                    planned.contact_loss_run_steps
                )
                self.traces["contact_progress_frozen"][step] = (
                    planned.progress_frozen
                )
                self.traces["contact_recovery_active"][step] = (
                    planned.recovery_active
                )
                self.traces["planned_knot_index"][step] = (
                    planned.planned_knot_index
                )
                self.traces["desired_cube_position_delta_m"][step] = (
                    planned.desired_position_delta_now_m
                )
                self.traces["desired_cube_rotation_vector_rad"][step] = (
                    planned.desired_rotation_vector_now_rad
                )
                if planned.acquired:
                    self.traces["actual_cube_position_delta_m"][step] = (
                        cube_position - planned.grasp_lock_cube_position_m
                    )
                    self.traces["actual_cube_rotation_vector_rad"][step] = (
                        _quaternion_delta_rotvec_world(
                            planned.grasp_lock_cube_quaternion_wxyz,
                            cube_quaternion,
                        )
                    )
                # This sample's post-step observation becomes the source used
                # by the next command.  Persisting ``step - 1`` on the command
                # trace makes that one-sample causality directly auditable.
                self.traces["operation_feedback_source_step"][step] = step - 1
                if planned.slip_feedback_enabled:
                    self.traces["online_contact_tangent_slip_m"][step] = (
                        planned.last_observed_tangent_slip_m
                    )
                    self.traces["online_contact_tangent_slip_valid"][step] = (
                        planned.last_observed_tangent_slip_valid
                    )
                    # ``tangent_slip_risk_active`` was computed by command()
                    # from the observation at step-1.  Abort risk is the
                    # current post-step safety observation and can only alter
                    # the command state emitted at the following sample.
                    self.traces["contact_tangent_slip_freeze_risk"][step] = (
                        planned.tangent_slip_risk_active
                    )
                    self.traces["contact_tangent_slip_abort_risk"][step] = (
                        planned.tangent_slip_abort_risk
                    )
            if self.schema_version >= 15:
                if not isinstance(
                    self.controller,
                    JointPairAlignedContactPreservingPlannedLiftController,
                ):
                    raise RuntimeError("schema-v15 controller factory mismatch")
                pair = self.controller
                self.traces["joint_pair_vector_cube_m"][step] = (
                    pair.joint_pair_vector_cube_m
                )
                self.traces["joint_pair_residual"][step] = (
                    pair.joint_pair_residual
                )
                self.traces["joint_pair_angle_deg"][step] = (
                    pair.joint_pair_angle_deg
                )
                self.traces["joint_pair_length_m"][step] = (
                    pair.joint_pair_length_m
                )
                self.traces["joint_pair_positive_y"][step] = (
                    pair.joint_pair_positive_y
                )
                self.traces["joint_pair_valid"][step] = pair.joint_pair_valid
                self.traces["joint_pair_alignment_safe"][step] = bool(
                    joint_pair_alignment_safe
                )
                self.traces["joint_pair_freeze_risk"][step] = (
                    pair.joint_pair_freeze_risk_active
                )
                self.traces["joint_pair_abort_risk"][step] = (
                    pair.joint_pair_abort_risk
                )
                self.traces["joint_pair_violation_run_steps"][step] = (
                    pair.joint_pair_violation_run_steps
                )
                self.traces["joint_pair_progress_frozen"][step] = (
                    pair.progress_frozen
                )
                self.traces["joint_pair_slip_recovery_active"][step] = (
                    pair.joint_pair_slip_recovery_active
                )
                self.traces["joint_pair_alignment_request_rad"][step] = (
                    pair.joint_pair_alignment_request_rad
                )
                self.traces[
                    "joint_pair_slip_recovery_correction_rad"
                ][step] = pair.joint_pair_slip_recovery_correction_rad
                self.traces["joint_pair_feedback_correction_rad"][step] = (
                    pair.joint_pair_feedback_correction_rad
                )
                self.traces["joint_pair_feedback_velocity_rad_s"][step] = (
                    pair.joint_pair_feedback_velocity_rad_s
                )
                self.traces["joint_pair_feedback_saturated"][step] = (
                    pair.joint_pair_feedback_saturated
                )
                self.traces["joint_pair_feedback_source_step"][step] = step - 1
                self.traces[
                    "joint_pair_active_residual_jacobian_2x8"
                ][step] = pair.joint_pair_active_residual_jacobian_2x8
                self.traces["joint_pair_abort_reason_step"][step] = (
                    pair.joint_pair_abort_reason
                )
                if active_finger_self_collision is None:
                    raise RuntimeError(
                        "schema-v15 self-collision observation is missing"
                    )
                self.traces["active_finger_self_collision"][step] = bool(
                    active_finger_self_collision["active"]
                )
                self.traces[
                    "active_finger_self_collision_contact_count"
                ][step] = int(active_finger_self_collision["contact_count"])
                self.traces[
                    "active_finger_self_collision_normal_force_n"
                ][step] = float(active_finger_self_collision["normal_force_n"])
                self.traces[
                    "active_finger_self_collision_max_penetration_m"
                ][step] = float(
                    active_finger_self_collision["max_penetration_m"]
                )
                self.traces["active_finger_self_collision_pairs"][step] = str(
                    active_finger_self_collision["pairs"]
                )
            if self.schema_version >= 16:
                if rolling_estimate is None or rolling_jacobian is None:
                    raise RuntimeError(
                        "schema-v16 rolling-slip observation is missing"
                    )
                if (
                    target_evidence is None
                    or rolling_aware_target_face_effective is None
                ):
                    raise RuntimeError(
                        "schema-v16 target-face recomputation is missing"
                    )
                self.traces[
                    "native_tactile_target_face_effective"
                ][step] = target_evidence.target_face_effective
                self.traces[
                    "rolling_aware_target_face_effective"
                ][step] = rolling_aware_target_face_effective
                self.traces[
                    "rolling_signed_tangent_displacement_m"
                ][step] = rolling_estimate.signed_tangent_displacement_m
                self.traces[
                    "rolling_cumulative_irrecoverable_slip_m"
                ][step] = rolling_estimate.cumulative_irrecoverable_slip_m
                self.traces[
                    "rolling_relative_tangent_velocity_m_s"
                ][step] = rolling_estimate.relative_tangent_velocity_m_s
                self.traces["rolling_contact_normal_force_n"][step] = (
                    rolling_estimate.normal_force_n
                )
                self.traces["rolling_contact_valid"][step] = (
                    rolling_estimate.valid
                )
                self.traces["rolling_contact_continuous"][step] = (
                    rolling_estimate.continuous
                )
                self.traces["rolling_detected"][step] = (
                    rolling_estimate.rolling_detected
                )
                self.traces["rolling_force_fraction"][step] = (
                    rolling_estimate.rolling_force_fraction
                )
                self.traces["rolling_patch_switch"][step] = (
                    rolling_estimate.patch_switch
                )
                self.traces["rolling_centroid_jump"][step] = (
                    rolling_estimate.centroid_jump
                )
                self.traces["rolling_centroid_tangent_step_m"][step] = (
                    rolling_estimate.centroid_tangent_step_m
                )
                self.traces["rolling_matched_patch_count"][step] = (
                    rolling_estimate.matched_patch_count
                )
                self.traces["rolling_new_patch_count"][step] = (
                    rolling_estimate.new_patch_count
                )
                self.traces["rolling_dropped_patch_count"][step] = (
                    rolling_estimate.dropped_patch_count
                )
                self.traces["rolling_patch_switch_count"][step] = (
                    rolling_estimate.patch_switch_count
                )
                self.traces[
                    "rolling_tangent_jacobian_m_per_rad"
                ][step] = (
                    rolling_jacobian.position_jacobian_cube_local_m_per_rad
                )
                self.traces["rolling_tangent_jacobian_force_n"][step] = (
                    rolling_jacobian.normal_force_n
                )
                self.traces["rolling_tangent_jacobian_valid"][step] = (
                    rolling_jacobian.valid
                )
                if not isinstance(
                    self.controller,
                    RollingSlipAwareJointPairAlignedContactPreservingPlannedLiftController,
                ):
                    raise RuntimeError("schema-v16 controller factory mismatch")
                rolling_controller = self.controller
                self.traces["rolling_slip_feedback_source_step"][step] = step - 1
                self.traces["rolling_slip_filtered_velocity_m_s"][step] = (
                    rolling_controller.rolling_slip_filtered_velocity_m_s
                )
                self.traces["rolling_slip_predicted_displacement_m"][step] = (
                    rolling_controller.rolling_slip_predicted_displacement_m
                )
                self.traces["rolling_slip_predicted_magnitude_m"][step] = (
                    rolling_controller.rolling_slip_predicted_magnitude_m
                )
                self.traces["rolling_slip_controller_cumulative_m"][step] = (
                    rolling_controller.rolling_slip_cumulative_m
                )
                self.traces["rolling_slip_observation_valid"][step] = (
                    rolling_controller.rolling_slip_observation_valid
                )
                self.traces["rolling_slip_recovery_active"][step] = (
                    rolling_controller.rolling_slip_recovery_active
                )
                self.traces["rolling_slip_freeze_active"][step] = (
                    rolling_controller.rolling_slip_freeze_active
                )
                self.traces["rolling_slip_abort_risk"][step] = (
                    rolling_controller.rolling_slip_abort_risk
                )
                self.traces["rolling_slip_exit_run_steps"][step] = (
                    rolling_controller.rolling_slip_exit_run_steps
                )
                self.traces["rolling_slip_request_rad"][step] = (
                    rolling_controller.rolling_slip_request_rad
                )
                self.traces["rolling_slip_correction_rad"][step] = (
                    rolling_controller.rolling_slip_correction_rad
                )
                self.traces[
                    "rolling_slip_correction_velocity_rad_s"
                ][step] = (
                    rolling_controller.rolling_slip_correction_velocity_rad_s
                )
                self.traces[
                    "rolling_slip_active_jacobian_m_per_rad"
                ][step] = (
                    rolling_controller.rolling_slip_active_jacobian_m_per_rad
                )
                self.traces["rolling_slip_feedback_saturated"][step] = (
                    rolling_controller.rolling_slip_feedback_saturated
                )

        if self._recorder is not None:
            self._recorder.maybe_record(self.data, step)
        self.step_index += 1
        return SimulationStep(
            index=step,
            time_s=float(self.data.time),
            control_state=(command.state.value if command is not None else None),
        )

    def finalize(
        self,
        *,
        trace_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Seal, evaluate and optionally persist a completed session."""

        if not self.complete:
            raise RuntimeError(
                f"cannot finalize an incomplete SimulationSession "
                f"({self.step_index}/{self.total_steps} steps)"
            )
        if self._summary is not None:
            if trace_path is not None:
                self._write_trace(trace_path)
            return self._summary

        if self.schema_version >= 3:
            assert self.controller is not None
            self.controller.finish()
            self.traces.update(self.controller.event_traces())
        if self.schema_version >= 9:
            from .grasp_pose import evaluate_actual_grasp_pose_trace

            states = np.asarray(self.traces["control_state"]).astype(str)
            base_gate = np.all(
                np.asarray(self.traces["grasp_gate"], dtype=bool), axis=1
            ) & (states == ControlState.VERIFY.value)
            self.traces["grasp_pose_base_gate"][:] = base_gate
            actual_pose = evaluate_actual_grasp_pose_trace(
                self.model,
                self.config,
                self.traces,
                base_gate_mask=base_gate,
                contact_mask=np.any(
                    np.asarray(
                        self.traces["finger_contact_force"], dtype=np.float64
                    )
                    > 1e-8,
                    axis=1,
                ),
            )
            actual_fields = actual_pose.as_trace_fields()
            # Online and offline paths intentionally implement the same
            # inclusive 250 ms event convention.  A mismatch is persisted for
            # evaluation rather than silently replacing controller evidence.
            for name, values in actual_fields.items():
                if name in self.traces:
                    existing = np.asarray(self.traces[name])
                    if existing.shape == np.asarray(values).shape:
                        self.traces[f"controller_{name}"] = existing.copy()
                self.traces[name] = np.asarray(values).copy()
        if self.schema_version >= 2:
            _populate_derived_v2_trace_fields(self.config, self.traces)
        if self.schema_version >= 13:
            _populate_derived_v13_contact_slip_trace_fields(
                self.model, self.config, self.traces
            )

        if self._recorder is not None:
            self._recorder.close()
            self.traces["video_frame_steps"] = np.asarray(
                self._recorder.frame_steps, dtype=np.int64
            )
        else:
            self.traces["video_frame_steps"] = np.asarray([], dtype=np.int64)

        summary = evaluate_trace(
            self.model,
            self.info,
            self.config,
            self.phase_steps,
            self.traces,
        )
        if self.schema_version >= 16:
            summary["rolling_contact_slip"] = _rolling_contact_slip_summary(
                self.config, self.traces
            )
        if self.contact_environment_evidence is not None:
            summary["contact_environment"] = {
                **copy.deepcopy(self.contact_environment_evidence),
                "runtime": {
                    "verified_contact_step_count": int(
                        self._contact_environment_runtime_contact_steps
                    ),
                    "other_geom_names": sorted(
                        self._contact_environment_runtime_other_geoms
                    ),
                    "first_cube_contact_snapshot": copy.deepcopy(
                        self._contact_environment_runtime_first
                    ),
                    "last_cube_contact_snapshot": copy.deepcopy(
                        self._contact_environment_runtime_last
                    ),
                },
            }
        if self.schema_version >= 11:
            names = np.asarray(
                self.traces["relative_pose_boundary_rejection_names"]
            ).astype(str)
            counts = np.asarray(
                self.traces["relative_pose_boundary_rejection_counts"],
                dtype=np.int64,
            )
            summary["relative_wrist_pose"] = {
                "clockwise_orbit_deg": float(
                    np.asarray(self.traces["clockwise_orbit_deg"]).reshape(-1)[0]
                ),
                "initial_root_position_cube_m": np.asarray(
                    self.traces["initial_root_position_cube_m"], dtype=np.float64
                ).tolist(),
                "initial_cube_from_root_rotation": np.asarray(
                    self.traces["initial_cube_from_root_rotation"],
                    dtype=np.float64,
                ).tolist(),
                "root_delta_cube_m": np.asarray(
                    self.traces["root_delta_cube_m"], dtype=np.float64
                ).tolist(),
                "wrist_local_rotvec_deg": np.asarray(
                    self.traces["wrist_local_rotvec_deg"], dtype=np.float64
                ).tolist(),
                "boundary_rejection_counts": {
                    str(name): int(count)
                    for name, count in zip(names.tolist(), counts.tolist())
                },
                "root_cube_center_distance_m": {
                    "initial": float(
                        np.asarray(
                            self.traces["root_cube_center_distance_m"],
                            dtype=np.float64,
                        )[0]
                    ),
                    "minimum": float(
                        np.min(
                            np.asarray(
                                self.traces["root_cube_center_distance_m"],
                                dtype=np.float64,
                            )
                        )
                    ),
                    "maximum": float(
                        np.max(
                            np.asarray(
                                self.traces["root_cube_center_distance_m"],
                                dtype=np.float64,
                            )
                        )
                    ),
                },
            }
        if self._video_path is not None:
            settings = VideoSettings()
            expected_frames = expected_video_frame_count(
                self.total_steps,
                float(self.model.opt.timestep),
                settings.fps,
            )
            if (
                self._recorder is None
                or len(self._recorder.frame_steps) != expected_frames
            ):
                actual = (
                    0 if self._recorder is None else len(self._recorder.frame_steps)
                )
                raise RuntimeError(
                    f"rendered {actual} frames, expected {expected_frames}"
                )
            summary["video"] = probe_video(
                self._video_path, expected_frames, settings
            )
            summary["video"]["simulation_step_indices"] = (
                self._recorder.frame_steps
            )
        self._summary = summary
        if trace_path is not None:
            self._write_trace(trace_path)
        return summary

    def _write_trace(self, trace_path: str | Path) -> None:
        output = Path(trace_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **self.traces)

    def close(self) -> None:
        """Release renderer resources; safe to call more than once."""

        if self._closed:
            return
        if self._recorder is not None:
            self._recorder.close()
        self._closed = True


def run_simulation(
    config: dict[str, Any],
    *,
    trace_path: str | Path | None = None,
    video_path: str | Path | None = None,
    contact_environment: ContactEnvironmentSpec | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one complete session through the same incremental core as Viewer."""

    session = SimulationSession(
        config,
        video_path=video_path,
        contact_environment=contact_environment,
    )
    try:
        while not session.complete:
            session.advance_one()
        return session.finalize(trace_path=trace_path)
    finally:
        session.close()


def _run_schema_v3_simulation(
    config: dict[str, Any],
    *,
    trace_path: str | Path | None = None,
    video_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for callers of the former private v3 loop."""

    if int(config.get("schema_version", 1)) < 3:
        raise ValueError("_run_schema_v3_simulation requires schema version 3+")
    return run_simulation(
        config,
        trace_path=trace_path,
        video_path=video_path,
    )


__all__ = [
    "ClosureAlignmentSnapshot",
    "ContactSnapshot",
    "PERSISTED_FACE_ORDER",
    "SimulationSession",
    "SimulationStep",
    "closure_alignment_snapshot",
    "contact_snapshot",
    "run_simulation",
]
