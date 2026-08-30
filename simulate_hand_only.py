#!/usr/bin/env python3
"""Simulate one standalone XHAND1 hand without a scene or grasp object."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import mujoco

from demo_control import XHandDemo


PROJECT_ROOT = Path(__file__).resolve().parent


def hand_model_path(side: str) -> Path:
    """Return the delivered standalone component MJCF for ``side``."""

    return PROJECT_ROOT / f"xhand_{side}.xml"


def validate_hand_only_model(model: mujoco.MjModel, side: str) -> None:
    """Reject a model containing anything beyond the fixed-base XHAND hand."""

    prefix = f"{side}_hand_"
    body_names = [model.body(index).name for index in range(1, model.nbody)]
    geom_names = [model.geom(index).name for index in range(model.ngeom)]
    if not body_names or any(not name.startswith(prefix) for name in body_names):
        raise RuntimeError("hand-only model contains a non-XHAND body")
    if not geom_names or any(not name.startswith(prefix) for name in geom_names):
        raise RuntimeError("hand-only model contains a non-XHAND geom")
    if model.nmocap != 0:
        raise RuntimeError("hand-only model must not contain mocap bodies")
    if any(
        int(joint_type) == int(mujoco.mjtJoint.mjJNT_FREE)
        for joint_type in model.jnt_type
    ):
        raise RuntimeError("hand-only model must have a fixed root")
    if model.njnt != 12 or model.nu != 12:
        raise RuntimeError(
            f"expected 12 hand joints and actuators, got {model.njnt}/{model.nu}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="仅仿真 XHAND1 手本体（无地面、支撑或抓取物）"
    )
    parser.add_argument("--side", choices=("right", "left"), default="left")
    parser.add_argument("--pose", choices=("open", "pinch", "close"), default="open")
    parser.add_argument("--headless", action="store_true", help="无 GUI 运行")
    parser.add_argument("--seconds", type=float, default=3.0, help="无 GUI 仿真时长")
    parser.add_argument("--report-interval", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = hand_model_path(args.side)
    demo = XHandDemo(
        args.side,
        args.pose,
        args.report_interval,
        model_path=path,
    )
    validate_hand_only_model(demo.model, args.side)
    print(
        f"仅仿真 {args.side} XHAND: {path.name}; "
        f"bodies={demo.model.nbody - 1}, joints={demo.model.njnt}, "
        f"actuators={demo.model.nu}, freejoints=0, mocap=0"
    )
    if args.headless:
        demo.run_headless(args.seconds)
    else:
        demo.run_viewer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
