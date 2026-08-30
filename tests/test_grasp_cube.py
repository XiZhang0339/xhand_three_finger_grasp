from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import grasp_cube
import xhand_grasp.search as search_module
from xhand_grasp.experiment import RobustnessParameters


@pytest.fixture(scope="module")
def nominal_config():
    return grasp_cube.load_config(grasp_cube.DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def nominal_model(nominal_config):
    return grasp_cube.build_model(nominal_config)


def test_actuator_mapping_exact(nominal_model):
    model, info = nominal_model
    assert tuple(model.actuator(index).name for index in info.active_actuator_ids) == (
        grasp_cube.ACTIVE_ACTUATORS
    )
    assert tuple(model.actuator(index).name for index in info.inactive_actuator_ids) == (
        grasp_cube.INACTIVE_ACTUATORS
    )
    assert len(set(info.active_actuator_ids) | set(info.inactive_actuator_ids)) == 12
    for actuator_id in info.active_actuator_ids:
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        assert joint_id >= 0
        assert model.joint(joint_id).name in model.actuator(actuator_id).name


def test_cube_half_extent_mass_and_inertia(nominal_model):
    model, info = nominal_model
    edge = 0.03
    mass = 0.02
    np.testing.assert_allclose(model.geom_size[info.cube_geom_id], [edge / 2] * 3)
    assert model.body_mass[info.cube_body_id] == pytest.approx(mass)
    np.testing.assert_allclose(
        model.body_inertia[info.cube_body_id],
        grasp_cube.cube_inertia(edge, mass),
    )
    assert model.jnt_type[info.cube_joint_id] == mujoco.mjtJoint.mjJNT_FREE


@pytest.mark.parametrize("friction", [0.4, 0.8, 1.2])
def test_actual_contact_friction_priority(nominal_config, friction):
    config = copy.deepcopy(nominal_config)
    config["cube"]["friction"] = friction
    config["cube"]["z_offset_m"] = -0.0005
    model, info = grasp_cube.build_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    contacts = [
        contact
        for contact in data.contact[: data.ncon]
        if {int(contact.geom1), int(contact.geom2)}
        == {info.cube_geom_id, info.support_geom_id}
    ]
    assert contacts
    for contact in contacts:
        assert int(contact.dim) == 4
        np.testing.assert_allclose(
            contact.friction,
            [friction, friction, 0.005, 0.0001, 0.0001],
        )


def test_distal_weld_and_hand_ancestry_classification(nominal_model):
    model, info = nominal_model
    expected = {
        "thumb": (
            "left_hand_thumb_rota_link2",
            "left_hand_thumb_rotaback_link2",
            "left_hand_thumb_rota_tip",
        ),
        "index": (
            "left_hand_index_rota_link2",
            "left_hand_index_rotaback_link2",
            "left_hand_index_rota_tip",
        ),
        "mid": (
            "left_hand_mid_link2",
            "left_hand_midback_link2",
            "left_hand_mid_tip",
        ),
    }
    for finger, names in expected.items():
        weld_ids = {int(model.body_weldid[model.body(name).id]) for name in names}
        assert weld_ids == {info.distal_weld_ids[finger]}
    assert info.hand_body_parts[model.body("left_hand_link").id] == "palm"
    assert int(model.geom_bodyid[info.support_geom_id]) == 0
    assert 0 not in info.hand_body_parts
    assert info.hand_body_parts[model.body("left_hand_ring_link2").id] == "ring"
    assert info.hand_body_parts[model.body("left_hand_pinky_link2").id] == "pinky"


def test_smoothstep_endpoints_and_slopes():
    assert grasp_cube.smoothstep(-1) == 0
    assert grasp_cube.smoothstep(0) == 0
    assert grasp_cube.smoothstep(1) == 1
    assert grasp_cube.smoothstep(2) == 1
    epsilon = 1e-6
    assert grasp_cube.smoothstep(epsilon) / epsilon < 1e-4
    assert (1 - grasp_cube.smoothstep(1 - epsilon)) / epsilon < 1e-4


def test_config_validation_rejects_non_three_finger_targets(nominal_config):
    config = copy.deepcopy(nominal_config)
    config["control"]["final_targets_rad"]["left_hand_ring_joint1_actuator"] = 0.1
    with pytest.raises(ValueError, match="exactly the eight"):
        grasp_cube.validate_config(config)

    config = copy.deepcopy(nominal_config)
    del config["control"]["pregrasp_targets_rad"][grasp_cube.ACTIVE_ACTUATORS[0]]
    with pytest.raises(ValueError, match="exactly the eight"):
        grasp_cube.validate_config(config)


def test_run_config_is_not_restricted_to_tuning_search_bounds(nominal_config):
    config = copy.deepcopy(nominal_config)
    name = "left_hand_thumb_bend_joint_actuator"
    config["control"]["final_targets_rad"][name] = 1.10
    grasp_cube.validate_config(config)
    model, _ = grasp_cube.build_model(config)
    targets = grasp_cube.actuator_target_vector(
        model, config["control"]["final_targets_rad"]
    )
    assert targets[model.actuator(name).id] == pytest.approx(1.10)


def test_config_rejects_invalid_support_geometry(nominal_config):
    config = copy.deepcopy(nominal_config)
    config["scene"]["floor_z_m"] = config["scene"]["support_top_z_m"]
    with pytest.raises(ValueError, match="below support_top"):
        grasp_cube.validate_config(config)


def test_json_writer_emits_strict_json(tmp_path):
    path = tmp_path / "result.json"
    grasp_cube.write_json(path, {"finite": 1.0, "bad": math.inf, "array": np.array([1, 2])})
    assert json.loads(path.read_text()) == {"array": [1, 2], "bad": None, "finite": 1.0}
    assert path.stat().st_mode & 0o777 == 0o644


def _synthetic_passing_trace(model, info, config):
    phase_steps = grasp_cube._phase_steps(model, config)
    count = sum(phase_steps.values())
    settle = phase_steps["settle"]
    hold = phase_steps["hold"]
    traces = {
        "time": np.arange(1, count + 1) * model.opt.timestep,
        "cube_pos": np.zeros((count, 3)),
        "cube_quat": np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        "cube_velocity": np.zeros((count, 6)),
        "root_pos": np.zeros((count, 3)),
        "root_quat": np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        "ctrl": np.zeros((count, model.nu)),
        "joint_qpos": np.zeros((count, model.nu)),
        "joint_qvel": np.zeros((count, model.nu)),
        "actuator_force": np.zeros((count, model.nu)),
        "finger_contact_force": np.zeros((count, 3)),
        "tactile_max": np.zeros((count, 5)),
        "forbidden_contact": np.zeros(count, dtype=bool),
        "support_contact": np.zeros(count, dtype=bool),
        "floor_contact": np.zeros(count, dtype=bool),
        "max_penetration": np.full(count, 0.002),
        "friction_error": np.zeros(count),
        "cube_contact_seen": np.ones(count, dtype=bool),
        "contact_dim_ok": np.ones(count, dtype=bool),
        "finite": np.ones(count, dtype=bool),
    }
    traces["cube_pos"][:, 2] = 0.1
    traces["cube_pos"][settle:, 2] = 0.11
    traces["cube_pos"][-500, 2] = 0.108
    hold_start = count - hold
    effective = np.zeros((hold, 3), dtype=bool)
    effective[:800, 0] = True
    effective[:800, 1] = True
    effective[:700, 2] = True
    effective[900:, 2] = True
    traces["finger_contact_force"][hold_start:] = effective * config["acceptance"][
        "contact_force_min_n"
    ]
    traces["tactile_max"][hold_start:, :3] = effective * config["acceptance"][
        "touch_force_min_n"
    ]
    angle = math.radians(20.0) / 2.0
    traces["cube_quat"][-1] = [math.cos(angle), 0.0, 0.0, math.sin(angle)]
    traces["cube_velocity"][-1, 0] = 0.049999
    return phase_steps, traces


def test_metrics_accept_exact_thresholds_and_speed_is_strict(nominal_config, nominal_model):
    model, info = nominal_model
    phase_steps, traces = _synthetic_passing_trace(model, info, nominal_config)
    result = grasp_cube.evaluate_trace(model, info, nominal_config, phase_steps, traces)
    assert result["passed"], result["failed_checks"]
    assert result["metrics"]["contact_duty"] == {
        "thumb": 0.8,
        "index": 0.8,
        "mid": 0.8,
    }
    assert result["metrics"]["simultaneous_contact_duty"] == 0.7

    traces = copy.deepcopy(traces)
    traces["cube_velocity"][-1, 0] = 0.05
    result = grasp_cube.evaluate_trace(model, info, nominal_config, phase_steps, traces)
    assert not result["checks"]["end_linear_speed_is_low"]


def _ranking_result(candidate_id, passed, force=10.0, checks=20):
    return {
        "candidate_id": candidate_id,
        "summary": {
            "passed": passed,
            "checks": {str(index): index < checks for index in range(22)},
            "metrics": {
                "peak_total_distal_contact_force_n": force,
                "actuator_saturation_fraction": 0.0,
                "hold_height_span_m": 0.0,
                "median_lift_m": 0.01,
                "contact_duty": {"thumb": 1.0, "index": 1.0, "mid": 1.0},
                "forbidden_contact_steps": 0,
            },
        },
    }


def _v2_ranking_result(
    config,
    candidate_id,
    *,
    passed,
    minimum_lift_m=0.012,
    median_lift_m=0.015,
    target_face_contact_duty=None,
    target_face_simultaneous_duty=0.85,
    force=10.0,
    checks=20,
):
    target_face_contact_duty = target_face_contact_duty or {
        "thumb": 0.9,
        "index": 0.9,
        "mid": 0.9,
    }
    touch_threshold = config["acceptance"]["touch_force_min_n"]
    return {
        "candidate_id": candidate_id,
        "config": config,
        "summary": {
            "passed": passed,
            "checks": {str(index): index < checks for index in range(22)},
            "metrics": {
                "peak_total_distal_contact_force_n": force,
                "actuator_saturation_fraction": 0.0,
                "hold_height_span_m": 0.001,
                "median_lift_m": median_lift_m,
                "minimum_lift_m": minimum_lift_m,
                "orientation_drift_deg": 10.0,
                "end_linear_speed_m_s": 0.025,
                "max_penetration_m": 0.001,
                "inactive_joint_max_abs_rad": 0.01,
                "contact_duty": {"thumb": 0.9, "index": 0.9, "mid": 0.9},
                "simultaneous_contact_duty": 0.85,
                "peak_tactile_n": {
                    "thumb": 1.5 * touch_threshold,
                    "index": 1.5 * touch_threshold,
                    "mid": 1.5 * touch_threshold,
                },
                "max_palm_down_angle_deg": 15.0,
                "target_face_contact_duty": target_face_contact_duty,
                "target_face_simultaneous_duty": target_face_simultaneous_duty,
                "peak_target_face_force_n": {
                    "thumb": 1.0,
                    "index": 1.0,
                    "mid": 1.0,
                },
                "material_off_target_duty": 0.005,
                "material_off_target_longest_run_s": 0.005,
                "material_active_nondistal_duty": 0.005,
                "material_active_nondistal_longest_run_s": 0.005,
                "forbidden_contact_steps": 0,
            },
        },
    }


def test_candidate_sort_is_deterministic_and_pass_first():
    failed = _ranking_result(0, False, force=1.0, checks=21)
    high_force = _ranking_result(1, True, force=20.0)
    low_force = _ranking_result(2, True, force=10.0)
    first = sorted([failed, high_force, low_force], key=grasp_cube.candidate_rank, reverse=True)
    second = sorted([low_force, failed, high_force], key=grasp_cube.candidate_rank, reverse=True)
    assert [item["candidate_id"] for item in first] == [2, 1, 0]
    assert [item["candidate_id"] for item in second] == [2, 1, 0]


def test_candidate_rank_v3_near_grasp_beats_zero_contact():
    def failed_v3(candidate_id, *, near_contact, checks, total_force):
        value = 0.02 if near_contact else 0.0
        return {
            "candidate_id": candidate_id,
            "summary": {
                "passed": False,
                "stage_status": {
                    "grasp_success": False,
                    "manipulation_success": False,
                    "full_success": False,
                },
                "checks": {str(index): index < checks for index in range(30)},
                "metrics": {
                    "verify_effective_finger_count": 0,
                    "verify_max_simultaneous_effective_finger_count": 0,
                    "verify_target_face_effective_duty": {
                        "thumb": 0.0,
                        "index": 0.0,
                        "mid": 0.0,
                    },
                    "verify_target_face_simultaneous_duty": 0.0,
                    "verify_peak_target_face_force_n": {
                        "thumb": value,
                        "index": value,
                        "mid": value,
                    },
                    "verify_peak_tactile_n": {
                        "thumb": value,
                        "index": value,
                        "mid": value,
                    },
                    "verify_max_consecutive_all_gate_steps": 0,
                    "verify_all_gate_duty": 0.0,
                    "verify_gate_component_duty": {},
                    "material_active_nondistal_duty": 0.0,
                    "median_lift_m": 0.0,
                    "peak_total_distal_contact_force_n": total_force,
                    "actuator_saturation_fraction": 0.0,
                },
            },
        }

    # The zero-contact candidate deliberately has more generic checks, less
    # force, and a better ID: old failure ranking preferred it for all three.
    zero = failed_v3(1, near_contact=False, checks=30, total_force=0.0)
    near = failed_v3(9, near_contact=True, checks=0, total_force=10.0)
    assert grasp_cube.candidate_rank(near) > grasp_cube.candidate_rank(zero)


def test_candidate_rank_v2_passes_prefer_larger_minimum_normalized_margin():
    config = grasp_cube.load_config(
        grasp_cube.SCRIPT_DIR
        / "grasp_configs"
        / "left_opposed_face_palm_down.json"
    )
    safer = _v2_ranking_result(
        config,
        1,
        passed=True,
        minimum_lift_m=0.012,
        force=25.0,
    )
    tight = _v2_ranking_result(
        config,
        2,
        passed=True,
        minimum_lift_m=0.0088,
        force=0.1,
    )

    safer_rank = grasp_cube.candidate_rank(safer)
    tight_rank = grasp_cube.candidate_rank(tight)
    ranked = sorted([tight, safer], key=grasp_cube.candidate_rank, reverse=True)

    assert safer_rank[1] == pytest.approx(0.5)
    assert tight_rank[1] == pytest.approx(0.1)
    assert [item["candidate_id"] for item in ranked] == [1, 2]


def test_candidate_rank_v2_failures_prefer_target_face_topology_over_lift():
    config = grasp_cube.load_config(
        grasp_cube.SCRIPT_DIR
        / "grasp_configs"
        / "left_opposed_face_palm_down.json"
    )
    complete_topology = _v2_ranking_result(
        config,
        1,
        passed=False,
        median_lift_m=0.001,
        target_face_contact_duty={"thumb": 0.8, "index": 0.8, "mid": 0.8},
        target_face_simultaneous_duty=0.7,
    )
    poor_simultaneous = _v2_ranking_result(
        config,
        2,
        passed=False,
        median_lift_m=0.03,
        target_face_contact_duty={"thumb": 0.8, "index": 0.8, "mid": 0.8},
        target_face_simultaneous_duty=0.1,
    )
    missing_one_finger = _v2_ranking_result(
        config,
        3,
        passed=False,
        median_lift_m=0.03,
        target_face_contact_duty={"thumb": 0.8, "index": 0.8, "mid": 0.1},
        target_face_simultaneous_duty=0.7,
    )

    ranked = sorted(
        [missing_one_finger, poor_simultaneous, complete_topology],
        key=grasp_cube.candidate_rank,
        reverse=True,
    )

    assert [item["candidate_id"] for item in ranked] == [1, 2, 3]


def test_candidate_rank_v1_retains_legacy_tuple_and_order():
    passed = _ranking_result(7, True, force=12.0)
    more_checks = _ranking_result(4, False, checks=21)
    fewer_checks = _ranking_result(3, False, checks=20)

    assert grasp_cube.candidate_rank(passed) == (
        1.0,
        -12.0,
        -0.0,
        -0.0,
        0.01,
        -7,
    )
    assert grasp_cube.candidate_rank(more_checks) == (
        0.0,
        21.0,
        0.01,
        1.0,
        -0.0,
        -4,
    )
    ranked = sorted(
        [fewer_checks, passed, more_checks],
        key=grasp_cube.candidate_rank,
        reverse=True,
    )
    assert [item["candidate_id"] for item in ranked] == [7, 4, 3]


def test_normalized_acceptance_margin_identifies_limiting_metric(nominal_config):
    acceptance = nominal_config["acceptance"]
    metrics = {
        "median_lift_m": 0.011,
        "minimum_lift_m": 0.010,
        "hold_height_span_m": 0.0005,
        "orientation_drift_deg": 5.0,
        "end_linear_speed_m_s": 0.01,
        "max_penetration_m": 0.001,
        "inactive_joint_max_abs_rad": 0.001,
        "contact_duty": {"thumb": 1.0, "index": 1.0, "mid": 1.0},
        "simultaneous_contact_duty": 1.0,
        "peak_tactile_n": {"thumb": 1.0, "index": 1.0, "mid": 1.0},
    }
    margins = grasp_cube.normalized_acceptance_margins(metrics, acceptance)
    assert min(margins, key=margins.get) == "median_lift_m"
    assert margins["median_lift_m"] == pytest.approx(0.1)


def test_v2_normalized_margins_include_topology_duty_and_run_limits():
    config = grasp_cube.load_config(
        grasp_cube.SCRIPT_DIR
        / "grasp_configs"
        / "left_opposed_face_palm_down.json"
    )
    metrics = {
        "median_lift_m": 0.02,
        "minimum_lift_m": 0.016,
        "hold_height_span_m": 0.0,
        "orientation_drift_deg": 0.0,
        "end_linear_speed_m_s": 0.0,
        "max_penetration_m": 0.0,
        "inactive_joint_max_abs_rad": 0.0,
        "contact_duty": {"thumb": 1.0, "index": 1.0, "mid": 1.0},
        "simultaneous_contact_duty": 1.0,
        "peak_tactile_n": {"thumb": 1.0, "index": 1.0, "mid": 1.0},
        "max_palm_down_angle_deg": 0.0,
        "target_face_contact_duty": {
            "thumb": 1.0,
            "index": 1.0,
            "mid": 1.0,
        },
        "target_face_simultaneous_duty": 1.0,
        "material_off_target_duty": 0.009,
        "material_off_target_longest_run_s": 0.008,
        "material_active_nondistal_duty": 0.007,
        "material_active_nondistal_longest_run_s": 0.0095,
    }

    margins = grasp_cube.normalized_acceptance_margins(
        metrics,
        config["acceptance"],
        contact_topology=config["contact_topology"],
    )

    assert margins["material_off_target_duty"] == pytest.approx(0.1)
    assert margins["material_off_target_longest_run_s"] == pytest.approx(0.2)
    assert margins["material_active_nondistal_duty"] == pytest.approx(0.3)
    assert margins["material_active_nondistal_longest_run_s"] == pytest.approx(0.05)
    assert min(margins, key=margins.get) == (
        "material_active_nondistal_longest_run_s"
    )


def _custom_robustness_parameters() -> RobustnessParameters:
    return RobustnessParameters(
        edge_m=(0.029,),
        mass_kg=(0.015, 0.025),
        friction=(0.55,),
        perturbation_count=3,
        required_pass_count=2,
        seed=77,
        position_xy_delta_m=(0.0001, 0.0002),
        rpy_delta_deg=(0.25, 0.5),
        mass_scale=(1.02, 1.03),
        friction_delta=(0.02, 0.03),
        z_offset_delta_m=(0.0002, 0.0003),
    )


def test_robustness_cases_use_registered_grid_seed_and_perturbation_ranges(
    nominal_config, monkeypatch
):
    parameters = _custom_robustness_parameters()
    monkeypatch.setattr(
        search_module,
        "resolve_experiment",
        lambda _config: SimpleNamespace(robustness=parameters),
    )

    grid, perturbations = search_module.robustness_cases(nominal_config)
    repeated_grid, repeated = search_module.robustness_cases(nominal_config)
    _, different_seed = search_module.robustness_cases(nominal_config, seed=78)

    assert grid == repeated_grid
    assert len(grid) == parameters.grid_case_count == 2
    assert [case["cube"]["mass_kg"] for case in grid] == [0.015, 0.025]
    assert all(case["cube"]["edge_m"] == 0.029 for case in grid)
    assert all(case["cube"]["friction"] == 0.55 for case in grid)
    assert len(perturbations) == parameters.perturbation_count == 3
    assert perturbations == repeated
    assert perturbations != different_seed

    base_cube = nominal_config["cube"]
    for case in perturbations:
        cube = case["cube"]
        for axis in range(2):
            delta = cube["center_xy_m"][axis] - base_cube["center_xy_m"][axis]
            assert (
                parameters.position_xy_delta_m[0]
                <= delta
                <= parameters.position_xy_delta_m[1]
            )
        for axis in range(3):
            delta = cube["rpy_deg"][axis] - base_cube["rpy_deg"][axis]
            assert parameters.rpy_delta_deg[0] <= delta <= parameters.rpy_delta_deg[1]
        mass_scale = cube["mass_kg"] / base_cube["mass_kg"]
        assert parameters.mass_scale[0] <= mass_scale <= parameters.mass_scale[1]
        friction_delta = cube["friction"] - base_cube["friction"]
        assert (
            parameters.friction_delta[0]
            <= friction_delta
            <= parameters.friction_delta[1]
        )
        z_delta = cube["z_offset_m"] - base_cube["z_offset_m"]
        assert parameters.z_offset_delta_m[0] <= z_delta <= parameters.z_offset_delta_m[1]


@pytest.mark.parametrize(
    ("nominal_passed", "perturbation_passes", "expected"),
    [(False, 3, False), (True, 1, False), (True, 2, True)],
)
def test_robust_pass_requires_nominal_and_registered_perturbation_threshold(
    nominal_config,
    monkeypatch,
    nominal_passed,
    perturbation_passes,
    expected,
):
    parameters = RobustnessParameters(
        edge_m=(0.03,),
        mass_kg=(0.02,),
        friction=(0.8,),
        perturbation_count=3,
        required_pass_count=2,
        seed=91,
        position_xy_delta_m=(-0.0001, 0.0001),
        rpy_delta_deg=(-0.1, 0.1),
        mass_scale=(0.99, 1.01),
        friction_delta=(-0.01, 0.01),
        z_offset_delta_m=(0.0, 0.0001),
    )
    monkeypatch.setattr(
        search_module,
        "resolve_experiment",
        lambda _config: SimpleNamespace(robustness=parameters),
    )
    call_sizes = []

    def run_candidates(payload, workers):
        assert workers == 1
        call_index = len(call_sizes)
        call_sizes.append(len(payload))
        results = []
        for trial_index, (candidate_id, config) in enumerate(payload):
            if call_index == 0:
                passed = nominal_passed
            elif call_index == 1:
                passed = False
            else:
                passed = trial_index < perturbation_passes
            results.append(
                {
                    "candidate_id": candidate_id,
                    "config": config,
                    "summary": {
                        "passed": passed,
                        "failed_checks": [] if passed else ["mock_failure"],
                        "checks": {"mock_check": passed},
                        "metrics": {},
                    },
                }
            )
        return results

    monkeypatch.setattr(search_module, "_run_candidates", run_candidates)
    result = search_module.robustness(nominal_config, workers=1)

    assert call_sizes == [1, 1, 3]
    assert result["seed"] == parameters.seed
    assert result["nominal_passed"] is nominal_passed
    assert result["perturbation_passes"] == perturbation_passes
    assert result["required_perturbation_passes"] == 2
    assert result["robust_passed"] is expected


def test_robustness_case_counts_and_seed(nominal_config):
    grid_a, perturbations_a = grasp_cube.robustness_cases(nominal_config, 20260821)
    grid_b, perturbations_b = grasp_cube.robustness_cases(nominal_config, 20260821)
    assert len(grid_a) == 5 * 4 * 5 == 100
    assert len(perturbations_a) == 50
    assert grid_a == grid_b
    assert perturbations_a == perturbations_b
    for case in perturbations_a:
        assert 0.0 <= case["cube"]["z_offset_m"] <= 0.0005
        assert abs(case["cube"]["center_xy_m"][0] - nominal_config["cube"]["center_xy_m"][0]) <= 0.0015
        assert abs(case["cube"]["center_xy_m"][1] - nominal_config["cube"]["center_xy_m"][1]) <= 0.0015


@pytest.mark.slow
def test_nominal_strict_three_finger_lift(nominal_config):
    result = grasp_cube.run_simulation(nominal_config)
    assert result["passed"], result["failed_checks"]
    metrics = result["metrics"]
    # Frozen MuJoCo 3.10.0/schema-v1 numerical baseline.  This complements the
    # façade/API compatibility checks with a real golden trajectory rather than
    # comparing two names bound to the same modular implementation.
    assert metrics["median_lift_m"] == pytest.approx(
        0.012765013032722664, abs=1e-12
    )
    assert metrics["minimum_lift_m"] == pytest.approx(
        0.01275354861913279, abs=1e-12
    )
    assert metrics["hold_height_span_m"] == pytest.approx(
        0.00032726817444937717, abs=1e-12
    )
    assert metrics["orientation_drift_deg"] == pytest.approx(
        10.511144173871754, abs=1e-10
    )
    assert metrics["end_linear_speed_m_s"] == pytest.approx(
        0.001980536725516048, abs=1e-12
    )
    assert metrics["contact_duty"] == {
        "thumb": 1.0,
        "index": 1.0,
        "mid": 1.0,
    }
    assert metrics["simultaneous_contact_duty"] == 1.0
    assert metrics["max_penetration_m"] == pytest.approx(
        0.0009968384853974417, abs=1e-12
    )
    assert metrics["inactive_ctrl_max_abs"] == 0.0
    assert metrics["forbidden_contact_steps"] == 0
