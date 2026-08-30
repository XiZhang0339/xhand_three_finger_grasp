"""MuJoCo scene construction for the XHAND three-finger cube task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import mujoco
import numpy as np

from .config import (
    ACTIVE_ACTUATORS,
    DISTAL_BODY_NAMES,
    INACTIVE_ACTUATORS,
    SCRIPT_DIR,
    _finite_sequence,
    validate_config,
)


# Positive finger flexion moves the straight index/middle fingertips toward
# local +X.  This is the palm-facing normal of the XHAND root frame and is a
# model convention, not user-configurable acceptance data.
PALM_NORMAL_LOCAL = (1.0, 0.0, 0.0)
PALM_FRAME_SITE_NAME = "three_finger_palm_frame"
PRESS_SOLVE_SINGULARITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PressDepthPose:
    """Rigid pose produced by an exact world-Z press-depth solve."""

    cube_in_root_x_m: float
    cube_in_root_m: tuple[float, float, float]
    root_translation_m: tuple[float, float, float]


@dataclass(frozen=True)
class ModelInfo:
    root_body_id: int
    cube_body_id: int
    cube_geom_id: int
    support_geom_id: int
    floor_geom_id: int
    cube_joint_id: int
    cube_qpos_adr: int
    cube_dof_adr: int
    active_actuator_ids: np.ndarray
    inactive_actuator_ids: np.ndarray
    actuator_joint_ids: np.ndarray
    actuator_qpos_adrs: np.ndarray
    actuator_dof_adrs: np.ndarray
    joint_ranges: np.ndarray
    joint_limited: np.ndarray
    force_limits: np.ndarray
    distal_weld_ids: dict[str, int]
    hand_body_parts: dict[int, str]
    requested_friction: float
    # Version 1 intentionally preserves the former model layout and therefore
    # has no marker site.  Version 2 stores the compiled marker's id here.
    palm_frame_site_id: int = -1


def rpy_degrees_to_quaternion(rpy_deg: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = np.radians(_finite_sequence(rpy_deg, 3, "rpy_deg")) / 2.0
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    quaternion = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def rpy_degrees_to_rotation_matrix(rpy_deg: Iterable[float]) -> np.ndarray:
    """Return the world-from-local ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` matrix."""

    roll, pitch, yaw = np.radians(_finite_sequence(rpy_deg, 3, "rpy_deg"))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _finite_matrix3(values: Iterable[Iterable[float]], label: str) -> np.ndarray:
    try:
        matrix = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite 3x3 matrix") from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    return matrix


def _unit_vector3(values: Iterable[float], label: str) -> np.ndarray:
    vector = np.asarray(_finite_sequence(values, 3, label), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{label} must have non-zero length")
    return vector / norm


def signed_finger_down_tilt_deg(
    rotation_world_from_root: Iterable[Iterable[float]],
    gravity_world_m_s2: Iterable[float],
) -> float:
    """Measure root-local +Z from the gravity-horizontal plane.

    The result is in ``[-90, 90]`` degrees.  Positive values point toward
    gravity (finger-down), negative values point away from gravity, and zero
    lies in the plane perpendicular to gravity.
    """

    rotation = _finite_matrix3(
        rotation_world_from_root, "rotation_world_from_root"
    )
    finger_axis = _unit_vector3(rotation[:, 2], "root local +Z axis")
    gravity = _unit_vector3(gravity_world_m_s2, "gravity_world_m_s2")
    gravity_component = float(finger_axis @ gravity)
    horizontal_component = float(
        np.linalg.norm(finger_axis - gravity_component * gravity)
    )
    return float(
        np.degrees(math.atan2(gravity_component, horizontal_component))
    )


def palm_plane_ground_angle_deg(
    rotation_world_from_root: Iterable[Iterable[float]],
    gravity_world_m_s2: Iterable[float],
) -> float:
    """Return the oriented palm-normal/gravity angle in ``[0, 180]`` degrees.

    Root-local +X is the palm-facing normal.  Thus zero means the palm faces
    along gravity and its plane is parallel to the ground; 180 means that the
    same parallel plane faces away from gravity.
    """

    rotation = _finite_matrix3(
        rotation_world_from_root, "rotation_world_from_root"
    )
    palm_normal = _unit_vector3(rotation[:, 0], "root local +X axis")
    gravity = _unit_vector3(gravity_world_m_s2, "gravity_world_m_s2")
    cosine = float(palm_normal @ gravity)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def solve_press_depth_pose(
    *,
    reference_root_translation_m: Iterable[float],
    press_depth_m: float,
    rotation_world_from_root: Iterable[Iterable[float]],
    cube_world_position_m: Iterable[float],
    cube_in_root_y_m: float,
    cube_in_root_z_m: float,
) -> PressDepthPose:
    """Solve local cube X and root translation for a requested world-Z press.

    Positive ``press_depth_m`` lowers root Z from the reference pose.  Local
    cube Y/Z and the rotation stay fixed.  The returned values satisfy
    ``root = cube_world - rotation @ cube_in_root`` by construction.
    """

    reference_root = np.asarray(
        _finite_sequence(
            reference_root_translation_m, 3, "reference_root_translation_m"
        ),
        dtype=np.float64,
    )
    cube_world = np.asarray(
        _finite_sequence(cube_world_position_m, 3, "cube_world_position_m"),
        dtype=np.float64,
    )
    rotation = _finite_matrix3(
        rotation_world_from_root, "rotation_world_from_root"
    )
    depth = float(press_depth_m)
    cube_y = float(cube_in_root_y_m)
    cube_z = float(cube_in_root_z_m)
    if not np.isfinite([depth, cube_y, cube_z]).all():
        raise ValueError("press depth and cube-in-root Y/Z must be finite")

    root_z = float(reference_root[2] - depth)
    coefficient = float(rotation[2, 0])
    if abs(coefficient) <= PRESS_SOLVE_SINGULARITY_TOLERANCE:
        raise ValueError(
            "press-depth solve is near-singular because root local +X has "
            "negligible world-Z component"
        )
    cube_x = float(
        (
            cube_world[2]
            - root_z
            - rotation[2, 1] * cube_y
            - rotation[2, 2] * cube_z
        )
        / coefficient
    )
    cube_in_root = np.asarray([cube_x, cube_y, cube_z], dtype=np.float64)
    root_translation = cube_world - rotation @ cube_in_root
    if not np.isfinite(root_translation).all():
        raise ValueError("press-depth solve produced a non-finite pose")
    return PressDepthPose(
        cube_in_root_x_m=cube_x,
        cube_in_root_m=tuple(float(value) for value in cube_in_root),
        root_translation_m=tuple(float(value) for value in root_translation),
    )


def cube_vertical_half_extent_m(
    edge_m: float,
    rotation_world_from_cube: Iterable[Iterable[float]],
) -> float:
    """Return a rotated cube's support extent along world Z."""

    edge = float(edge_m)
    if not math.isfinite(edge) or edge <= 0.0:
        raise ValueError("edge_m must be positive and finite")
    rotation = _finite_matrix3(
        rotation_world_from_cube, "rotation_world_from_cube"
    )
    return float((edge / 2.0) * np.sum(np.abs(rotation[2, :])))


def cube_inertia(edge_m: float, mass_kg: float) -> np.ndarray:
    moment = float(mass_kg) * float(edge_m) ** 2 / 6.0
    return np.full(3, moment, dtype=np.float64)


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    current = int(body_id)
    while current > 0:
        if current == ancestor_id:
            return True
        current = int(model.body_parentid[current])
    return False


def _body_part(name: str) -> str:
    for finger in ("thumb", "index", "mid", "ring", "pinky"):
        if f"_{finger}" in name:
            return finger
    return "palm"


def build_model(config: dict[str, Any]) -> tuple[mujoco.MjModel, ModelInfo]:
    validate_config(config)
    spec = mujoco.MjSpec.from_file(str((SCRIPT_DIR / "xhand_left.xml").resolve()))
    root = spec.body("left_hand_link")
    if root is None:
        raise ValueError("xhand_left.xml is missing left_hand_link")
    root.pos = _finite_sequence(config["hand_pose"]["translation_m"], 3, "root pos")
    root.quat = rpy_degrees_to_quaternion(config["hand_pose"]["rpy_deg"])

    schema_version = config["schema_version"]
    if schema_version >= 2:
        root.add_site(
            name=PALM_FRAME_SITE_NAME,
            zaxis=PALM_NORMAL_LOCAL,
            size=[0.001],
            group=5,
            rgba=[0.0, 0.0, 0.0, 0.0],
        )

    scene = config["scene"]
    cube_config = config["cube"]
    support_top = float(scene["support_top_z_m"])
    floor_z = float(scene.get("floor_z_m", -0.015))
    support_radius = float(scene["support_radius_m"])
    support_half_height = (support_top - floor_z) / 2.0
    support_center_z = floor_z + support_half_height

    spec.worldbody.add_geom(
        name="three_finger_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, floor_z],
        size=[0.35, 0.35, 0.02],
        contype=1,
        conaffinity=1,
        condim=4,
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.10, 0.13, 0.16, 1.0],
    )
    center_x, center_y = [float(value) for value in cube_config["center_xy_m"]]
    spec.worldbody.add_geom(
        name="three_finger_support",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[center_x, center_y, support_center_z],
        size=[support_radius, support_half_height, 0.0],
        contype=1,
        conaffinity=1,
        condim=4,
        friction=[1.2, 0.005, 0.0001],
        rgba=[0.30, 0.36, 0.42, 1.0],
    )

    edge = float(cube_config["edge_m"])
    mass = float(cube_config["mass_kg"])
    cube_rpy = cube_config.get("rpy_deg", [0.0, 0.0, 0.0])
    if schema_version >= 4:
        vertical_half_extent = cube_vertical_half_extent_m(
            edge, rpy_degrees_to_rotation_matrix(cube_rpy)
        )
        cube_z = (
            support_top
            + vertical_half_extent
            + float(cube_config.get("z_offset_m", 0.0))
        )
    else:
        # Keep the legacy arithmetic and placement exactly unchanged.  The
        # v1-v3 scenes intentionally ignore cube roll/pitch for support height.
        cube_z = (
            support_top
            + edge / 2.0
            + float(cube_config.get("z_offset_m", 0.0))
        )
    cube = spec.worldbody.add_body(
        name="three_finger_cube",
        pos=[center_x, center_y, cube_z],
        quat=rpy_degrees_to_quaternion(cube_rpy),
        mass=mass,
        ipos=[0.0, 0.0, 0.0],
        iquat=[1.0, 0.0, 0.0, 0.0],
        inertia=cube_inertia(edge, mass),
        explicitinertial=1,
    )
    cube.add_freejoint(name="three_finger_cube_free")
    cube.add_geom(
        name="three_finger_cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[edge / 2.0] * 3,
        mass=0.0,
        contype=1,
        conaffinity=1,
        condim=4,
        priority=10,
        friction=[float(cube_config["friction"]), 0.005, 0.0001],
        solref=[float(cube_config.get("solref_timeconst_s", 0.004)), 1.0],
        solimp=[0.9, 0.95, 0.001, 0.5, 2.0],
        rgba=[0.95, 0.45, 0.08, 1.0],
    )
    spec.worldbody.add_light(
        name="three_finger_key_light",
        pos=[-0.2, -0.3, 0.5],
        dir=[0.3, 0.4, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.8, 0.8, 0.8],
        ambient=[0.2, 0.2, 0.2],
    )
    spec.worldbody.add_camera(
        name="three_finger_camera",
        pos=[0.34, -0.36, 0.25],
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
        targetbody="three_finger_cube",
        fovy=42.0,
    )

    model = spec.compile()
    root_body_id = model.body("left_hand_link").id
    cube_body_id = model.body("three_finger_cube").id
    cube_geom_id = model.geom("three_finger_cube_geom").id
    support_geom_id = model.geom("three_finger_support").id
    floor_geom_id = model.geom("three_finger_floor").id
    cube_joint_id = model.joint("three_finger_cube_free").id
    palm_frame_site_id = (
        model.site(PALM_FRAME_SITE_NAME).id if schema_version >= 2 else -1
    )

    active_ids = np.array([model.actuator(name).id for name in ACTIVE_ACTUATORS], dtype=int)
    inactive_ids = np.array(
        [model.actuator(name).id for name in INACTIVE_ACTUATORS], dtype=int
    )
    actuator_joint_ids = model.actuator_trnid[:, 0].astype(int).copy()
    actuator_qpos_adrs = model.jnt_qposadr[actuator_joint_ids].astype(int).copy()
    actuator_dof_adrs = model.jnt_dofadr[actuator_joint_ids].astype(int).copy()
    force_limits = np.max(np.abs(model.actuator_forcerange), axis=1)
    hand_body_parts = {
        body_id: _body_part(model.body(body_id).name)
        for body_id in range(1, model.nbody)
        if _is_descendant(model, body_id, root_body_id)
    }
    distal_weld_ids = {
        finger: int(model.body_weldid[model.body(body_name).id])
        for finger, body_name in DISTAL_BODY_NAMES.items()
    }

    if model.body_jntnum[root_body_id] != 0 or model.body_mocapid[root_body_id] != -1:
        raise AssertionError("the hand root must remain a fixed, non-mocap body")
    if set(active_ids) & set(inactive_ids) or len(set(active_ids)) != 8:
        raise AssertionError("actuator mapping is not isolated")

    info = ModelInfo(
        root_body_id=root_body_id,
        cube_body_id=cube_body_id,
        cube_geom_id=cube_geom_id,
        support_geom_id=support_geom_id,
        floor_geom_id=floor_geom_id,
        cube_joint_id=cube_joint_id,
        cube_qpos_adr=int(model.jnt_qposadr[cube_joint_id]),
        cube_dof_adr=int(model.jnt_dofadr[cube_joint_id]),
        active_actuator_ids=active_ids,
        inactive_actuator_ids=inactive_ids,
        actuator_joint_ids=actuator_joint_ids,
        actuator_qpos_adrs=actuator_qpos_adrs,
        actuator_dof_adrs=actuator_dof_adrs,
        joint_ranges=model.jnt_range[actuator_joint_ids].copy(),
        joint_limited=model.jnt_limited[actuator_joint_ids].astype(bool).copy(),
        force_limits=force_limits,
        distal_weld_ids=distal_weld_ids,
        hand_body_parts=hand_body_parts,
        requested_friction=float(cube_config["friction"]),
        palm_frame_site_id=palm_frame_site_id,
    )
    return model, info
