from __future__ import annotations

import numpy as np
import pytest

from demo_control import XHandDemo


def _manual_controls(demo: XHandDemo) -> np.ndarray:
    """Return a distinct, in-range value for every actuator slider."""

    lower = demo.model.actuator_ctrlrange[:, 0]
    upper = demo.model.actuator_ctrlrange[:, 1]
    fractions = np.linspace(0.23, 0.77, demo.model.nu)
    return lower + fractions * (upper - lower)


@pytest.mark.parametrize("actuator_index", range(12))
def test_each_viewer_actuator_slider_interrupts_automatic_pose(
    actuator_index: int,
) -> None:
    """Controls imported by viewer.sync must survive the next physics step."""

    demo = XHandDemo("left", "close", report_interval=0.0)
    manual = demo.data.ctrl.copy()
    manual[actuator_index] = _manual_controls(demo)[actuator_index]

    # In the passive viewer this assignment is performed by viewer.sync() after
    # the user drags one or more sliders in the right-side Control panel.
    demo.data.ctrl[:] = manual
    demo.step()

    assert demo.model.nu == 12
    assert demo.control_mode == "manual"
    assert demo.target_name == "manual"
    np.testing.assert_allclose(demo.data.ctrl, manual, rtol=0.0, atol=0.0)


def test_manual_mode_accepts_later_slider_edits() -> None:
    demo = XHandDemo("left", "close", report_interval=0.0)
    first = demo.data.ctrl.copy()
    first[0] = _manual_controls(demo)[0]
    demo.data.ctrl[:] = first
    demo.step()

    second = first.copy()
    second[-1] = _manual_controls(demo)[-1]
    demo.data.ctrl[:] = second
    demo.step()

    assert demo.control_mode == "manual"
    np.testing.assert_allclose(demo.data.ctrl, second, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("key", "pose"),
    (("O", "open"), ("P", "pinch"), ("C", "close")),
)
def test_pose_hotkeys_resume_automatic_control_after_slider_edit(
    key: str,
    pose: str,
) -> None:
    demo = XHandDemo("left", "close", report_interval=0.0)
    manual = _manual_controls(demo)
    demo.data.ctrl[:] = manual
    demo.step()
    np.testing.assert_allclose(demo.data.ctrl, manual, rtol=0.0, atol=0.0)

    demo.key_callback(ord(key))
    demo.step()

    alpha = demo.model.opt.timestep / demo.transition_duration
    expected = manual + alpha * (demo.targets[pose] - manual)
    assert demo.control_mode == "preset"
    assert demo.target_name == pose
    np.testing.assert_allclose(demo.data.ctrl, expected, rtol=0.0, atol=1e-15)
