from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import grasp_cube as legacy_grasp_cube
import xhand_grasp.simulation as simulation_module
from xhand_grasp.artifacts import json_text, write_json
from xhand_grasp.config import load_config, validate_config
from xhand_grasp.contacts import FACE_ORDER, BoxContactThresholds
from xhand_grasp.evaluation import _v2_face_metrics, evaluate_trace, face_from_label
from xhand_grasp.scene import build_model
from xhand_grasp.simulation import _palm_down_angle_deg, run_simulation
from xhand_grasp.trajectory import _phase_steps


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "grasp_configs" / "left_three_finger_cube.json"
V2_CONFIG = ROOT / "grasp_configs" / "left_opposed_face_palm_down.json"

LEGACY_TRACE_KEYS = {
    "time",
    "cube_pos",
    "cube_quat",
    "cube_velocity",
    "root_pos",
    "root_quat",
    "ctrl",
    "joint_qpos",
    "joint_qvel",
    "actuator_force",
    "finger_contact_force",
    "tactile_max",
    "forbidden_contact",
    "support_contact",
    "floor_contact",
    "max_penetration",
    "friction_error",
    "cube_contact_seen",
    "contact_dim_ok",
    "finite",
    "video_frame_steps",
}

V2_TRACE_KEYS = {
    "face_order",
    "finger_order",
    "actuator_order",
    "palm_down_angle_deg",
    "distal_face_force_n",
    "active_nondistal_force_n",
    "target_face_force_purity",
    "target_face_topology",
}


@pytest.fixture(scope="module")
def v2_config() -> dict:
    return load_config(V2_CONFIG)


@pytest.fixture(scope="module")
def v2_run(v2_config, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("v2-simulation")
    trace_path = output_dir / "trace.npz"
    summary = run_simulation(copy.deepcopy(v2_config), trace_path=trace_path)
    with np.load(trace_path, allow_pickle=False) as archive:
        traces = {key: archive[key].copy() for key in archive.files}
    return {
        "summary": summary,
        "traces": traces,
        "trace_path": trace_path,
        "output_dir": output_dir,
    }


@pytest.mark.parametrize(
    ("pitch_deg", "expected_angle_deg"),
    [(90.0, 0.0), (0.0, 90.0), (-90.0, 180.0)],
)
def test_palm_down_angle_uses_model_owned_local_positive_x_axis(
    v2_config, pitch_deg, expected_angle_deg
):
    config = copy.deepcopy(v2_config)
    config["hand_pose"]["rpy_deg"] = [0.0, pitch_deg, 0.0]
    model, info = build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert _palm_down_angle_deg(model, data, info) == pytest.approx(
        expected_angle_deg, abs=1e-10
    )


def test_palm_down_angle_rejects_zero_gravity(v2_config):
    config = copy.deepcopy(v2_config)
    config["hand_pose"]["rpy_deg"] = [0.0, 90.0, 0.0]
    model, info = build_model(config)
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with pytest.raises(ValueError, match="non-zero gravity"):
        _palm_down_angle_deg(model, data, info)


def test_v2_config_rejects_wrong_or_incomplete_contact_topology(v2_config):
    invalid = copy.deepcopy(v2_config)
    invalid["contact_topology"]["target_faces"]["mid"] = "+Y"
    with pytest.raises(ValueError, match="same|identical"):
        validate_config(invalid)

    missing_topology = copy.deepcopy(v2_config)
    del missing_topology["contact_topology"]
    with pytest.raises((KeyError, ValueError), match="contact_topology"):
        validate_config(missing_topology)

    missing_faces = copy.deepcopy(v2_config)
    del missing_faces["contact_topology"]["target_faces"]
    with pytest.raises((KeyError, ValueError), match="target_faces"):
        validate_config(missing_faces)

    missing_purity = copy.deepcopy(v2_config)
    del missing_purity["contact_topology"]["target_force_fraction"]
    with pytest.raises((KeyError, ValueError), match="target_force_fraction"):
        validate_config(missing_purity)


def test_v2_real_run_persists_finite_face_contact_arrays(v2_config, v2_run):
    traces = v2_run["traces"]
    assert set(traces) == LEGACY_TRACE_KEYS | V2_TRACE_KEYS
    steps = traces["time"].shape[0]

    assert traces["palm_down_angle_deg"].shape == (steps,)
    assert traces["distal_face_force_n"].shape == (steps, 3, len(FACE_ORDER))
    assert traces["active_nondistal_force_n"].shape == (steps, 3)
    assert traces["target_face_force_purity"].shape == (steps, 3)
    assert traces["target_face_topology"].shape == (steps,)
    assert traces["target_face_topology"].dtype == np.bool_
    assert traces["face_order"].tolist() == [
        "+X",
        "-X",
        "+Y",
        "-Y",
        "+Z",
        "-Z",
        "EDGE_CORNER",
        "UNKNOWN",
    ]
    assert traces["finger_order"].tolist() == ["thumb", "index", "mid"]
    model, _ = build_model(v2_config)
    assert traces["actuator_order"].tolist() == [
        model.actuator(index).name for index in range(model.nu)
    ]
    assert all(
        traces[key].dtype.kind == "U"
        for key in ("face_order", "finger_order", "actuator_order")
    )

    for key in V2_TRACE_KEYS - {
        "target_face_topology",
        "face_order",
        "finger_order",
        "actuator_order",
    }:
        assert np.isfinite(traces[key]).all(), key
    assert np.all(traces["distal_face_force_n"] >= 0.0)
    assert np.all(traces["active_nondistal_force_n"] >= 0.0)
    assert np.all((0.0 <= traces["target_face_force_purity"]))
    assert np.all((traces["target_face_force_purity"] <= 1.0))

    # Every positive distal contact must land in exactly one of six faces,
    # EDGE_CORNER, or UNKNOWN.  No force may be dropped or counted twice.
    np.testing.assert_allclose(
        np.sum(traces["distal_face_force_n"], axis=2),
        traces["finger_contact_force"],
        rtol=0.0,
        atol=1e-12,
    )
    assert np.any(traces["distal_face_force_n"] > 0.0)


def test_v2_derived_trace_fields_can_be_recomputed_from_persisted_evidence(
    v2_config, v2_run
):
    traces = v2_run["traces"]
    topology = v2_config["contact_topology"]
    target_indices = np.asarray(
        [
            FACE_ORDER.index(face_from_label(topology["target_faces"][finger]))
            for finger in ("thumb", "index", "mid")
        ],
        dtype=int,
    )
    target_force = traces["distal_face_force_n"][:, np.arange(3), target_indices]
    total_force = np.sum(traces["distal_face_force_n"], axis=2)
    purity = np.divide(
        target_force,
        total_force,
        out=np.zeros_like(target_force),
        where=total_force > 0.0,
    )
    np.testing.assert_array_equal(traces["target_face_force_purity"], purity)

    effective = (
        target_force >= float(v2_config["acceptance"]["contact_force_min_n"])
    ) & (
        traces["tactile_max"][:, :3]
        >= float(v2_config["acceptance"]["touch_force_min_n"])
    ) & (
        purity >= float(topology["target_force_fraction"])
    )
    np.testing.assert_array_equal(
        traces["target_face_topology"], np.all(effective, axis=1)
    )


def test_v2_json_judgment_is_exactly_recomputable_from_npz(
    v2_config, v2_run
):
    model, info = build_model(v2_config)
    phase_steps = _phase_steps(model, v2_config)
    recomputed = evaluate_trace(
        model,
        info,
        v2_config,
        phase_steps,
        v2_run["traces"],
    )

    assert json_text(recomputed) == json_text(v2_run["summary"])
    summary_path = v2_run["output_dir"] / "summary.json"
    write_json(summary_path, v2_run["summary"])
    assert json.loads(summary_path.read_text(encoding="utf-8")) == json.loads(
        json_text(recomputed)
    )
    assert v2_run["summary"]["passed"] is False


def test_schema_v1_trace_keys_and_numbers_match_legacy_implementation(tmp_path):
    config = load_config(V1_CONFIG)
    short_config = copy.deepcopy(config)
    short_config["timing"] = {
        "settle_s": 0.02,
        "pregrasp_s": 0.02,
        "lift_s": 0.02,
        "hold_s": 0.02,
    }
    legacy_trace_path = tmp_path / "legacy.npz"
    modular_trace_path = tmp_path / "modular.npz"

    legacy_summary = legacy_grasp_cube.run_simulation(
        copy.deepcopy(short_config), trace_path=legacy_trace_path
    )
    modular_summary = run_simulation(
        copy.deepcopy(short_config), trace_path=modular_trace_path
    )

    with np.load(legacy_trace_path, allow_pickle=False) as legacy_archive, np.load(
        modular_trace_path, allow_pickle=False
    ) as modular_archive:
        assert set(legacy_archive.files) == LEGACY_TRACE_KEYS
        assert set(modular_archive.files) == LEGACY_TRACE_KEYS
        for key in sorted(LEGACY_TRACE_KEYS):
            np.testing.assert_array_equal(modular_archive[key], legacy_archive[key])

    assert json_text(modular_summary) == json_text(legacy_summary)


def test_v2_simulation_passes_configured_geometry_thresholds_to_classifier(
    v2_config, monkeypatch
):
    config = copy.deepcopy(v2_config)
    topology = config["contact_topology"]
    topology["surface_tolerance_m"] = 40e-6
    topology["edge_margin_m"] = 0.7e-3
    topology["min_normal_alignment"] = 0.90
    validate_config(config)

    captured: list[BoxContactThresholds] = []
    original_classifier = simulation_module.classify_box_contact

    def capture_thresholds(
        surface_point_local,
        outward_normal_local,
        half_extents_m,
        *,
        thresholds,
    ):
        captured.append(thresholds)
        return original_classifier(
            surface_point_local,
            outward_normal_local,
            half_extents_m,
            thresholds=thresholds,
        )

    monkeypatch.setattr(
        simulation_module, "classify_box_contact", capture_thresholds
    )
    simulation_module.run_simulation(config)

    assert captured, "the real v2 run did not classify any distal cube contact"
    assert all(isinstance(value, BoxContactThresholds) for value in captured)
    assert {
        (
            value.surface_tolerance_m,
            value.edge_margin_m,
            value.normal_alignment_min,
        )
        for value in captured
    } == {(40e-6, 0.7e-3, 0.90)}


def test_active_nondistal_hard_failure_respects_configuration_switch(v2_config):
    steps = 20
    config = copy.deepcopy(v2_config)
    target_faces = config["contact_topology"]["target_faces"]
    target_indices = [
        FACE_ORDER.index(face_from_label(target_faces[finger]))
        for finger in ("thumb", "index", "mid")
    ]
    face_force = np.zeros((steps, 3, len(FACE_ORDER)))
    for finger_index, target_index in enumerate(target_indices):
        face_force[:, finger_index, target_index] = 0.1
    traces = {
        "time": np.arange(1, steps + 1, dtype=np.float64) * 0.001,
        "distal_face_force_n": face_force,
        "tactile_max": np.full((steps, 5), 0.1),
        "support_contact": np.zeros(steps, dtype=bool),
        "floor_contact": np.zeros(steps, dtype=bool),
        "active_nondistal_force_n": np.full((steps, 3), 0.1),
        "palm_down_angle_deg": np.zeros(steps),
    }
    model = SimpleNamespace(opt=SimpleNamespace(timestep=0.001))
    phase_steps = {"hold": steps}

    config["contact_topology"]["forbid_active_nondistal"] = False
    metrics_allowed, checks_allowed = _v2_face_metrics(
        model, config, phase_steps, traces
    )
    assert metrics_allowed["material_active_nondistal_duty"] == 1.0
    assert metrics_allowed["material_active_nondistal_longest_run_s"] == pytest.approx(
        0.020
    )
    assert checks_allowed["active_nondistal_contacts_within_limit"]

    config["contact_topology"]["forbid_active_nondistal"] = True
    metrics_forbidden, checks_forbidden = _v2_face_metrics(
        model, config, phase_steps, traces
    )
    assert metrics_forbidden == metrics_allowed
    assert not checks_forbidden["active_nondistal_contacts_within_limit"]
    assert all(
        passed
        for name, passed in checks_forbidden.items()
        if name != "active_nondistal_contacts_within_limit"
    )
