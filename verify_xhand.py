#!/usr/bin/env python3
"""Deterministic compile, dynamics, actuator and tactile checks for XHAND1."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from xhand_tactile import FINGERS, TactileReader


SCRIPT_DIR = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_component(side: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCRIPT_DIR / f"xhand_{side}.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    require(model.nbody == 31, f"{side}: nbody={model.nbody}, 应为 31")
    require(model.njnt == 12 and model.nu == 12, f"{side}: 应为 12 关节/12 执行器")
    require(model.nmesh == 30, f"{side}: nmesh={model.nmesh}, 应为 30")
    require(model.nsensor == 600 and model.nsensordata == 600, f"{side}: 应为 600 路 touch")
    require(model.nsite == 600, f"{side}: nsite={model.nsite}, 应为 600")
    require(data.ncon == 0, f"{side}: 张开位不应有自穿透，实际 {data.ncon} contacts")

    close = model.key_ctrl[model.key("xhand_close").id]
    for step in range(1500):
        data.ctrl[:] = min(1.0, step / 800.0) * close
        mujoco.mj_step(model, data)
    require(np.isfinite(data.qpos).all(), f"{side}: qpos 出现 NaN/Inf")
    require(np.isfinite(data.qvel).all(), f"{side}: qvel 出现 NaN/Inf")
    require(np.linalg.norm(data.qpos) > 1.0, f"{side}: actuator 未驱动手指")
    for joint_id in range(model.njnt):
        qpos = data.qpos[model.jnt_qposadr[joint_id]]
        lower, upper = model.jnt_range[joint_id]
        require(lower - 2e-3 <= qpos <= upper + 2e-3, f"{side}: 关节越限")
    print(f"  {side} component: compile/dynamics/limits OK")


def validate_scene_and_grasp(side: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCRIPT_DIR / f"scene_{side}.xml"))
    data = mujoco.MjData(model)
    reader = TactileReader(model, data, side)
    pinch = model.key_ctrl[model.key("xhand_pinch").id]
    for step in range(2500):
        data.ctrl[:] = min(1.0, step / 1500.0) * pinch
        mujoco.mj_step(model, data)

    contact_names = {
        tuple(sorted((model.geom(contact.geom1).name, model.geom(contact.geom2).name)))
        for contact in data.contact[: data.ncon]
        if contact.geom1 >= 0 and contact.geom2 >= 0
    }
    require(
        any("thumb_rota_link2_collision" in " ".join(pair) for pair in contact_names),
        f"{side}: 捏取球未接触拇指末端",
    )
    require(
        any("index_rota_link2_collision" in " ".join(pair) for pair in contact_names),
        f"{side}: 捏取球未接触食指末端",
    )
    normal = reader.normal_forces()
    require(normal[0].max() > 0 and normal[1].max() > 0, f"{side}: 拇/食指 touch 无输出")
    reconstructed = reader.taxel_forces_link()
    require(np.linalg.norm(reconstructed[0]) > 0, f"{side}: 拇指三轴力重建失败")
    require(np.linalg.norm(reconstructed[1]) > 0, f"{side}: 食指三轴力重建失败")
    print(
        f"  {side} pinch: thumb/index contact OK, "
        f"active touch={np.count_nonzero(normal > 1e-8)}"
    )


def validate_each_finger_probe(side: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCRIPT_DIR / f"scene_{side}.xml"))
    model.opt.gravity[:] = 0
    object_joint = model.joint("grasp_object_free").id
    object_qpos = model.jnt_qposadr[object_joint]
    probe_mocap = model.body("tactile_probe").mocapid

    for finger_index, finger in enumerate(FINGERS):
        data = mujoco.MjData(model)
        data.qpos[object_qpos : object_qpos + 3] = (0, 0, -2)
        data.qpos[object_qpos + 3 : object_qpos + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(model, data)
        reader = TactileReader(model, data, side)
        point = 60
        site_id = model.site(f"{side}_tactile_{finger}_{point:03d}_site").id
        data.mocap_pos[probe_mocap] = data.site_xpos[site_id]
        for _ in range(30):
            mujoco.mj_step(model, data)

        normal = reader.normal_forces()
        forces = reader.taxel_forces_link()
        require(
            normal[finger_index, point - 1] > 0,
            f"{side}/{finger}: probe 未触发 taxel {point}",
        )
        require(
            np.linalg.norm(forces[finger_index]) > 0,
            f"{side}/{finger}: probe 未产生三轴力",
        )
    print(f"  {side} probe: all 5 fingertips / native+3D tactile OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 XHAND1 MuJoCo 交付模型")
    parser.add_argument("--side", choices=("right", "left", "both"), default="both")
    args = parser.parse_args()
    sides = ("right", "left") if args.side == "both" else (args.side,)

    print("XHAND1 verification")
    for side in sides:
        validate_component(side)
        validate_scene_and_grasp(side)
        validate_each_finger_probe(side)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
