#!/usr/bin/env python3
"""Interactive and headless control demo for the XHAND1 MuJoCo model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from xhand_tactile import TactileReader


SCRIPT_DIR = Path(__file__).resolve().parent


class XHandDemo:
    def __init__(self, side: str, pose: str, report_interval: float):
        self.side = side
        self.model = mujoco.MjModel.from_xml_path(str(SCRIPT_DIR / f"scene_{side}.xml"))
        self.data = mujoco.MjData(self.model)
        self.reader = TactileReader(self.model, self.data, side)
        self.targets = {
            name: self.model.key_ctrl[self.model.key(f"xhand_{name}").id].copy()
            for name in ("open", "pinch", "close")
        }
        self.target_name = "open"
        self.target = self.targets["open"].copy()
        self.transition_start = self.target.copy()
        self.transition_elapsed = 1.5
        self.transition_duration = 1.5
        self.report_interval = report_interval
        self.reset_requested = False
        self.taxels_visible = False
        self.contacts_visible = True
        self.reset()
        self.begin_pose(pose)

    def begin_pose(self, pose: str) -> None:
        self.target_name = pose
        self.transition_start = self.data.ctrl.copy()
        self.target = self.targets[pose].copy()
        self.transition_elapsed = 0.0

    def key_callback(self, keycode: int) -> None:
        try:
            key = chr(keycode).upper()
        except (ValueError, OverflowError):
            return
        if key in {"O", "P", "C"}:
            self.begin_pose({"O": "open", "P": "pinch", "C": "close"}[key])
            print(f"\n目标手势: {self.target_name}")
        elif key == "R":
            self.reset_requested = True
            print("\n重置手和抓取物")
        elif key == "T":
            self.taxels_visible = not self.taxels_visible
            print(f"\ntaxel 可视化: {'开' if self.taxels_visible else '关'}")
        elif key == "F":
            self.contacts_visible = not self.contacts_visible
            print(f"\n接触力可视化: {'开' if self.contacts_visible else '关'}")

    def reset(self) -> None:
        # Component keyframes intentionally contain controls only so they remain
        # valid when a scene adds a free object.  Reset qpos from model.qpos0,
        # then apply the open control explicitly.
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = self.targets["open"]
        mujoco.mj_forward(self.model, self.data)
        self.target_name = "open"
        self.target = self.targets["open"].copy()
        self.transition_start = self.target.copy()
        self.transition_elapsed = self.transition_duration
        self.reset_requested = False

    def step(self) -> None:
        if self.reset_requested:
            self.reset()
        # Synchronized 1.5 s interpolation keeps every joint below the delivered
        # velocity limits and avoids striking the supported demo object.  Only
        # actuator controls are changed; joint qpos is never teleported.
        self.transition_elapsed = min(
            self.transition_duration,
            self.transition_elapsed + self.model.opt.timestep,
        )
        alpha = self.transition_elapsed / self.transition_duration
        self.data.ctrl[:] = self.transition_start + alpha * (
            self.target - self.transition_start
        )
        mujoco.mj_step(self.model, self.data)

    def report(self) -> None:
        fields = []
        for finger, stats in self.reader.summary().items():
            net = float(np.linalg.norm(stats.net_force_link))
            fields.append(
                f"{finger}: active={stats.active_taxels:2d}, "
                f"normal_max={stats.max_normal_force:6.2f}N, net3D={net:6.2f}N"
            )
        print(f"\r[{self.target_name:5s}] " + " | ".join(fields), end="", flush=True)

    def run_headless(self, seconds: float) -> None:
        next_report = 0.0
        steps = max(1, int(seconds / self.model.opt.timestep))
        for step in range(steps):
            self.step()
            if self.report_interval > 0 and step * self.model.opt.timestep >= next_report:
                self.report()
                next_report += self.report_interval
        self.report()
        print()

    def run_viewer(self) -> None:
        import mujoco.viewer

        print(
            "交互键: O=张开  P=捏取  C=握拳  R=重置  "
            "T=显示/隐藏600个taxel  F=接触力\n"
            "MuJoCo 右侧 Control 面板也可直接拖动 12 个 actuator。"
        )
        next_report = time.monotonic()
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.key_callback
        ) as viewer:
            viewer.opt.geomgroup[3] = 0
            viewer.opt.sitegroup[4] = 0
            while viewer.is_running():
                started = time.monotonic()
                self.step()
                viewer.opt.sitegroup[4] = int(self.taxels_visible)
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                    self.contacts_visible
                )
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(
                    self.contacts_visible
                )
                if self.report_interval > 0 and started >= next_report:
                    self.report()
                    next_report = started + self.report_interval
                viewer.sync()
                remaining = self.model.opt.timestep - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="XHAND1 MuJoCo 触觉与控制 Demo")
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--pose", choices=("open", "pinch", "close"), default="open")
    parser.add_argument("--headless", action="store_true", help="无 GUI 运行")
    parser.add_argument("--seconds", type=float, default=3.0, help="无 GUI 仿真时长")
    parser.add_argument("--report-interval", type=float, default=0.5)
    args = parser.parse_args()

    demo = XHandDemo(args.side, args.pose, args.report_interval)
    if args.headless:
        demo.run_headless(args.seconds)
    else:
        demo.run_viewer()


if __name__ == "__main__":
    main()
