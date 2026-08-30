from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from demo_control import XHandDemo
from simulate_hand_only import hand_model_path, validate_hand_only_model


ROOT = Path(__file__).resolve().parents[1]


def _component_demo(side: str = "left", pose: str = "open") -> XHandDemo:
    return XHandDemo(
        side,
        pose,
        report_interval=0.0,
        model_path=ROOT / f"xhand_{side}.xml",
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_hand_only_cli_resolves_the_component_xml(side: str) -> None:
    assert hand_model_path(side) == ROOT / f"xhand_{side}.xml"


def test_hand_only_validator_rejects_the_object_demo_scene() -> None:
    scene_model = mujoco.MjModel.from_xml_path(str(ROOT / "scene_left.xml"))

    with pytest.raises(RuntimeError, match="non-XHAND body"):
        validate_hand_only_model(scene_model, "left")


@pytest.mark.parametrize("side", ["left", "right"])
def test_hand_only_model_contains_only_the_selected_xhand(side: str) -> None:
    demo = _component_demo(side)
    model = demo.model

    body_names = [model.body(index).name for index in range(model.nbody)]
    geom_names = [model.geom(index).name for index in range(model.ngeom)]

    assert model.nbody == 31
    assert body_names[0] == "world"
    assert all(name.startswith(f"{side}_hand_") for name in body_names[1:])
    assert model.ngeom == 60
    assert all(name.startswith(f"{side}_hand_") for name in geom_names)
    assert not any(
        token in name
        for name in body_names + geom_names
        for token in ("grasp_object", "support", "floor", "ground", "probe")
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_hand_only_model_has_twelve_fixed_root_hinge_dofs(side: str) -> None:
    demo = _component_demo(side)
    model = demo.model

    assert (model.nq, model.nv, model.njnt, model.nu) == (12, 12, 12, 12)
    assert model.nmocap == 0
    assert np.all(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)

    root_id = model.body(f"{side}_hand_link").id
    assert model.body_parentid[root_id] == 0
    assert model.body_jntnum[root_id] == 0


def test_hand_only_root_world_pose_does_not_move() -> None:
    demo = _component_demo("left", "close")
    root_id = demo.model.body("left_hand_link").id
    initial_position = demo.data.xpos[root_id].copy()
    initial_orientation = demo.data.xmat[root_id].copy()

    for _ in range(100):
        demo.step()

    np.testing.assert_array_equal(demo.data.xpos[root_id], initial_position)
    np.testing.assert_array_equal(demo.data.xmat[root_id], initial_orientation)


@pytest.mark.parametrize("pose", ["open", "pinch", "close"])
def test_hand_only_presets_are_component_keyframe_controls(pose: str) -> None:
    demo = _component_demo("left", pose)
    model = demo.model
    expected = model.key_ctrl[model.key(f"xhand_{pose}").id]

    assert set(demo.targets) == {"open", "pinch", "close"}
    np.testing.assert_array_equal(demo.targets[pose], expected)
    assert np.all(expected >= model.actuator_ctrlrange[:, 0])
    assert np.all(expected <= model.actuator_ctrlrange[:, 1])


def test_hand_only_manual_controls_persist_for_all_twelve_actuators() -> None:
    demo = _component_demo("left", "close")
    lower = demo.model.actuator_ctrlrange[:, 0]
    upper = demo.model.actuator_ctrlrange[:, 1]
    manual = lower + np.linspace(0.2, 0.8, demo.model.nu) * (upper - lower)

    demo.data.ctrl[:] = manual
    demo.step()
    demo.step()

    assert demo.model.nu == 12
    assert demo.control_mode == "manual"
    assert demo.target_name == "manual"
    np.testing.assert_array_equal(demo.data.ctrl, manual)


def test_hand_only_headless_cli_runs_without_opening_a_viewer() -> None:
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = "osmesa"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "simulate_hand_only.py"),
            "--side",
            "left",
            "--pose",
            "pinch",
            "--headless",
            "--seconds",
            "0.003",
            "--report-interval",
            "0",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "Traceback" not in completed.stderr
    assert "pinch" in completed.stdout
