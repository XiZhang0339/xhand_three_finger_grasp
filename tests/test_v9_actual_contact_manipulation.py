from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from xhand_grasp.checkpoint import capture_physics_checkpoint
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.controller import GraspVerifyThenManipulateController
from xhand_grasp.scene import build_model
from xhand_grasp.trajectory import _phase_steps, minimum_jerk
from xhand_grasp.tuning import actual_contact_manipulation as manipulation


ROOT = Path(__file__).resolve().parents[1]
V9_TEMPLATE = ROOT / "grasp_configs" / (
    "left_opposed_face_palm_down_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)


def _config() -> dict:
    return load_config(V9_TEMPLATE)


def _synthetic_model() -> mujoco.MjModel:
    joints = []
    actuators = []
    for index, name in enumerate(ACTIVE_ACTUATORS):
        joint = f"joint_{index}"
        joints.append(
            f'<body name="link_{index}" pos="{0.03 * index} 0 0.2">'
            f'<joint name="{joint}" type="hinge" axis="0 1 0" '
            'range="-2 2" damping="0.01"/>'
            '<geom type="sphere" size="0.002" mass="0.001" '
            'contype="0" conaffinity="0"/>'
            '</body>'
        )
        actuators.append(
            f'<position name="{name}" joint="{joint}" kp="10" '
            'ctrlrange="-2 2" forcerange="-100 100"/>'
        )
    xml = f"""
    <mujoco>
      <option timestep="0.001" gravity="0 0 -9.81"/>
      <worldbody>
        <body name="cube" pos="0 0 0.2">
          <freejoint/>
          <geom type="box" size="0.01 0.01 0.01" mass="0.1"/>
        </body>
        {''.join(joints)}
      </worldbody>
      <actuator>{''.join(actuators)}</actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def _synthetic_checkpoint(config: dict | None = None) -> manipulation.GraspPhysicsCheckpoint:
    config = copy.deepcopy(config or _config())
    model = _synthetic_model()
    data = mujoco.MjData(model)
    nominal = config["grasp_pose"]["nominal_joint_qpos_rad"]
    for name in ACTIVE_ACTUATORS:
        actuator_id = model.actuator(name).id
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        data.qpos[model.jnt_qposadr[joint_id]] = float(nominal[name])
    mujoco.mj_forward(model, data)
    cube_id = model.body("cube").id
    checkpoint = capture_physics_checkpoint(model, data, step_index=250)
    return manipulation.GraspPhysicsCheckpoint(
        model=model,
        checkpoint=checkpoint,
        cube_body_id=cube_id,
        grasp_lock_step=249,
        actual_grasp_qpos_rad=np.asarray(
            [float(nominal[name]) for name in ACTIVE_ACTUATORS]
        ),
        lock_sample_joint_qpos_rad=np.asarray(
            [float(nominal[name]) for name in ACTIVE_ACTUATORS]
        ),
        cube_position_world_m=np.asarray(data.xpos[cube_id]),
        cube_quaternion_wxyz=np.asarray(data.xquat[cube_id]),
        config=config,
    )


def test_source_requires_actual_grasp_evidence_not_large_command() -> None:
    config = _config()
    config["control"]["contact_preload_targets_rad"][ACTIVE_ACTUATORS[0]] = 1.59
    trace = {
        "grasp_lock_step": np.asarray(123, dtype=np.int64),
        "grasp_pose_actual_qpos_rad": np.asarray(
            [config["grasp_pose"]["nominal_joint_qpos_rad"][name] for name in ACTIVE_ACTUATORS]
        ),
    }
    assert manipulation.validate_grasp_success_source(
        config, trace, {"stage_status": {"grasp_success": True}}
    ) == 123
    trace["grasp_lock_step"] = np.asarray(-1, dtype=np.int64)
    with pytest.raises(ValueError, match="no grasp_lock"):
        manipulation.validate_grasp_success_source(
            config, trace, {"stage_status": {"grasp_success": True}}
        )


def test_probe_plan_is_zero_plus_eight_positive_negative_pairs_and_clips() -> None:
    config = _config()
    # Real ctrl upper is 2.0 in the small model; the registered v9 delta upper
    # remains the tighter source of truth for most axes.
    model = _synthetic_model()
    specs = manipulation.generate_probe_specifications(model, config, epsilon_rad=0.02)
    assert len(specs) == 17
    assert specs[0].kind == "zero"
    assert [spec.actuator for spec in specs[1::2]] == list(ACTIVE_ACTUATORS)
    assert [spec.direction for spec in specs[1:]] == [-1, 1] * 8
    bounds = manipulation.manipulation_delta_bounds(model, config)
    for spec in specs:
        for index, name in enumerate(ACTIVE_ACTUATORS):
            assert bounds[name][0] <= spec.applied_delta_rad[index] <= bounds[name][1]


def test_each_probe_independently_restores_identical_checkpoint() -> None:
    grasp = _synthetic_checkpoint()
    zero = manipulation.generate_probe_specifications(
        grasp.model, grasp.config, epsilon_rad=0.02
    )[0]
    first = manipulation.run_checkpoint_probe(
        grasp, zero, manipulate_s=0.005, hold_s=0.005
    )
    second = manipulation.run_checkpoint_probe(
        grasp, zero, manipulate_s=0.005, hold_s=0.005
    )
    np.testing.assert_array_equal(first["response_6d"], second["response_6d"])
    np.testing.assert_array_equal(
        first["final_cube_position_world_m"], second["final_cube_position_world_m"]
    )
    assert first["checkpoint_step_index"] == second["checkpoint_step_index"] == 250


def _linear_probe_results(
    specs: tuple[manipulation.ProbeSpecification, ...],
) -> tuple[dict, ...]:
    matrix = np.zeros((6, 8), dtype=np.float64)
    matrix[2, :] = np.linspace(0.02, 0.09, 8)
    matrix[0, 1] = 0.001
    bias = np.asarray((0.0, 0.0, -0.0001, 0.0, 0.0, 0.0))
    records = []
    for spec in specs:
        applied = np.asarray(spec.applied_delta_rad)
        records.append(
            {
                "probe": spec.as_mapping(),
                "response_6d": (bias + matrix @ applied).tolist(),
            }
        )
    return tuple(records)


def test_response_fit_and_64_trust_deltas_are_deterministic_and_bounded() -> None:
    config = _config()
    model = _synthetic_model()
    bounds = manipulation.manipulation_delta_bounds(model, config)
    specs = manipulation.generate_probe_specifications(model, config)
    response = manipulation.fit_response_jacobian(
        _linear_probe_results(specs), bounds, config=config
    )
    assert response["probe_count"] == 17
    assert np.asarray(response["jacobian_6x8"]).shape == (6, 8)
    first = manipulation.generate_trust_region_deltas(response, bounds, count=64)
    second = manipulation.generate_trust_region_deltas(response, bounds, count=64)
    assert first == second
    assert len(first) == 64
    for delta in first:
        for name in ACTIVE_ACTUATORS:
            assert bounds[name][0] <= delta[name] <= bounds[name][1]


def test_trust_set_keeps_solution_and_adds_deterministic_wide_exploration() -> None:
    bounds = {name: (-1.0, 1.0) for name in ACTIVE_ACTUATORS}
    response = {
        "solution_delta_rad": {name: 0.0 for name in ACTIVE_ACTUATORS}
    }
    candidates = manipulation.generate_trust_region_deltas(
        response,
        bounds,
        count=64,
        seed=20260821,
        trust_radius_fraction=0.15,
        wide_candidate_fraction=0.40,
        wide_radius_fraction=0.45,
    )
    assert len(candidates) == 64
    assert candidates[0] == response["solution_delta_rad"]
    # A near-only set would stay within 0.15 * the two-radian range = 0.3.
    assert any(
        max(abs(value) for value in candidate.values()) > 0.3 + 1e-12
        for candidate in candidates
    )
    assert candidates == manipulation.generate_trust_region_deltas(
        response,
        bounds,
        count=64,
        seed=20260821,
        trust_radius_fraction=0.15,
        wide_candidate_fraction=0.40,
        wide_radius_fraction=0.45,
    )


def test_full_reset_rerun_is_only_advancement_path(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    model = _synthetic_model()
    calls: list[dict] = []

    def fake_run_simulation(candidate: dict) -> dict:
        calls.append(copy.deepcopy(candidate))
        index_delta = candidate["control"]["manipulation_delta_rad"][
            "left_hand_index_joint1_actuator"
        ]
        if index_delta > 0.0:
            return {"passed": False, "stage_status": {"full_success": True}}
        return {"passed": True, "stage_status": {"full_success": False}}

    monkeypatch.setattr(manipulation, "run_simulation", fake_run_simulation)
    zero = {name: 0.0 for name in ACTIVE_ACTUATORS}
    positive = dict(zero)
    positive["left_hand_index_joint1_actuator"] = 0.05
    report = manipulation.run_full_reset_candidates(
        config, (zero, positive), model=model
    )
    assert len(calls) == 2
    assert all(record["full_reset_rerun"] for record in report["candidates"])
    assert report["full_success_count"] == 1
    assert [record["candidate_index"] for record in report["advanced_candidates"]] == [1]


def test_full_reset_candidate_ids_and_order_do_not_depend_on_workers() -> None:
    config = _config()
    model = _synthetic_model()
    deltas = []
    for index in range(5):
        delta = {name: 0.0 for name in ACTIVE_ACTUATORS}
        delta["left_hand_index_joint1_actuator"] = 0.01 * index
        deltas.append(delta)
    observed_workers: list[int] = []

    def order_scrambling_executor(jobs, workers):
        observed_workers.append(workers)
        records = [
            {
                "candidate_id": int(job["candidate_id"]),
                "summary": {
                    "stage_status": {
                        "full_success": int(job["candidate_id"]) % 2 == 0
                    }
                },
                "artifacts": {
                    "trace": job["job_metadata"]["trace_name"]
                },
            }
            for job in jobs
        ]
        return tuple(records if workers == 1 else reversed(records))

    serial = manipulation.run_full_reset_candidates(
        config,
        deltas,
        model=model,
        workers=1,
        executor=order_scrambling_executor,
        job_metadata=[{"trace_name": f"trace_{index}.npz"} for index in range(5)],
    )
    parallel = manipulation.run_full_reset_candidates(
        config,
        deltas,
        model=model,
        workers=3,
        executor=order_scrambling_executor,
        job_metadata=[{"trace_name": f"trace_{index}.npz"} for index in range(5)],
    )
    assert observed_workers == [1, 3]
    assert serial == parallel
    assert [record["candidate_id"] for record in serial["candidates"]] == list(
        range(5)
    )
    assert [
        record["executor_metadata"]["artifacts"]["trace"]
        for record in serial["candidates"]
    ] == [f"trace_{index}.npz" for index in range(5)]


def test_injectable_small_budget_preserves_declared_target_band() -> None:
    budget = manipulation.ManipulationSearchBudget(
        manipulate_s=0.005,
        hold_s=0.005,
        trust_candidate_count=3,
        target_upward_m=0.010,
    )
    assert budget.trust_candidate_count == 3
    with pytest.raises(ValueError, match="10--12 mm"):
        manipulation.ManipulationSearchBudget(target_upward_m=0.009)


class _SyntheticLockedSession:
    """Minimal session contract around real MuJoCo state for pipeline tests."""

    def __init__(self, config: dict) -> None:
        self.model = _synthetic_model()
        self.data = mujoco.MjData(self.model)
        qpos_addresses = np.empty(self.model.nu, dtype=np.int64)
        nominal = config["grasp_pose"]["nominal_joint_qpos_rad"]
        actual = np.empty(len(ACTIVE_ACTUATORS), dtype=np.float64)
        for index, name in enumerate(ACTIVE_ACTUATORS):
            actuator_id = self.model.actuator(name).id
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            qpos_addresses[actuator_id] = qpos_address
            actual[index] = float(nominal[name])
            self.data.qpos[qpos_address] = actual[index]
        mujoco.mj_forward(self.model, self.data)
        self.info = SimpleNamespace(
            cube_body_id=self.model.body("cube").id,
            actuator_qpos_adrs=qpos_addresses,
        )
        self.controller = SimpleNamespace(
            acquired=False,
            grasp_acquisition_step=-1,
            grasp_pose_actual_qpos_rad=actual.copy(),
        )
        self.step_index = 0
        self._complete = False
        self.closed = False

    @property
    def complete(self) -> bool:
        return self._complete

    def advance_one(self) -> SimpleNamespace:
        self.controller.acquired = True
        self.controller.grasp_acquisition_step = 0
        self.step_index = 1
        self._complete = True
        return SimpleNamespace(index=0)

    def close(self) -> None:
        self.closed = True


def test_small_budget_pipeline_contract_uses_checkpoint_then_full_reset() -> None:
    config = _config()
    actual = np.asarray(
        [config["grasp_pose"]["nominal_joint_qpos_rad"][name] for name in ACTIVE_ACTUATORS]
    )
    source_trace = {
        "grasp_lock_step": np.asarray(0, dtype=np.int64),
        "grasp_pose_actual_qpos_rad": actual,
    }
    calls: list[dict] = []

    def final_runner(candidate: dict) -> dict:
        calls.append(copy.deepcopy(candidate))
        return {
            "passed": len(calls) == 1,
            "stage_status": {"full_success": len(calls) == 1},
        }

    report = manipulation.run_checkpoint_guided_manipulation(
        config,
        source_trace,
        {"stage_status": {"grasp_success": True}},
        budget=manipulation.ManipulationSearchBudget(
            manipulate_s=0.002,
            hold_s=0.002,
            trust_candidate_count=3,
            target_upward_m=0.010,
        ),
        session_factory=_SyntheticLockedSession,
        simulation_runner=final_runner,
    )
    assert report["grasp_lock_step"] == 0
    np.testing.assert_array_equal(report["actual_grasp_qpos_rad"], actual)
    assert report["probe_count"] == 17
    assert all(
        probe["search_branch_source"]
        == "grasp_lock_mjstate_integration_checkpoint"
        for probe in report["probes"]
    )
    assert all(
        probe["command_reference"] == "contact_preload_targets_rad"
        and probe["manipulation_profile"] == "minimum_jerk_quintic"
        for probe in report["probes"]
    )
    reruns = report["full_reruns"]
    assert len(calls) == reruns["candidate_count"] == 3
    assert reruns["full_success_count"] == 1
    assert all(
        record["initial_state_source"] == "configured_no_contact_reset"
        and record["checkpoint_used"] is False
        for record in reruns["candidates"]
    )
    assert len(reruns["advanced_candidates"]) == 1


def test_v9_controller_minimum_jerk_is_relative_to_contact_preload() -> None:
    config = _config()
    actuator = "left_hand_thumb_bend_joint_actuator"
    config["control"]["manipulation_delta_rad"][actuator] = -0.05
    model, _ = build_model(config)
    phases = _phase_steps(model, config)
    controller = GraspVerifyThenManipulateController(model, config, phases)

    # Advance command sequencing to the first VERIFY sample.  Acquisition is
    # injected here because this test isolates the post-lock command contract;
    # actual gate/lock reconstruction has separate simulation/evaluator tests.
    for step in range(controller.close_end + 1):
        verify_command = controller.command(step)
    assert verify_command.state.value == "VERIFY"
    controller._acquired = True
    controller.grasp_acquisition_step = controller.close_end
    controller.manipulation_start_step = controller.close_end + 1

    command = controller.command(controller.close_end + 1)
    actuator_id = model.actuator(actuator).id
    raw = 1.0 / phases["manipulate"]
    preload = config["control"]["contact_preload_targets_rad"][actuator]
    expected = preload + minimum_jerk(raw) * -0.05
    assert command.state.value == "MANIPULATE"
    assert command.target[actuator_id] == pytest.approx(expected)
    assert command.target[actuator_id] != pytest.approx(
        config["grasp_pose"]["nominal_joint_qpos_rad"][actuator]
        + minimum_jerk(raw) * -0.05
    )


def _refinement_summary(
    *,
    full: bool,
    median: float,
    minimum: float,
    topology: float,
    smooth_scale: float = 0.5,
) -> dict:
    return {
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": full,
            "full_success": full,
        },
        "failed_checks": [] if full else ["operation_median_lift_reached"],
        "metrics": {
            "operation_median_lift_m": median,
            "operation_minimum_lift_m": minimum,
            "operation_target_face_simultaneous_duty": topology,
            "contact_alignment": {"operation": {"aligned_duty": topology}},
            "motion_smoothness": {
                "operation_cumulative_height_backtrack_m": 0.0002 * smooth_scale,
                "operation_downward_speed_duty": 0.02 * smooth_scale,
                "operation_peak_filtered_upward_speed_m_s": 0.02 * smooth_scale,
                "operation_peak_abs_filtered_acceleration_m_s2": 0.12 * smooth_scale,
                "operation_peak_abs_filtered_jerk_m_s3": 2.5 * smooth_scale,
                "operation_hold_entry_linear_speed_m_s": 0.005 * smooth_scale,
                "operation_max_lateral_displacement_m": 0.002 * smooth_scale,
                "operation_max_orientation_drift_deg": 10.0 * smooth_scale,
            },
        },
    }


def test_refinement_ranking_prioritizes_success_then_normalized_margins() -> None:
    config = _config()
    records = [
        {
            "candidate_id": 3,
            "manipulation_delta_rad": {name: 0.0 for name in ACTIVE_ACTUATORS},
            "summary": _refinement_summary(
                full=False, median=0.012, minimum=0.009, topology=0.9
            ),
        },
        {
            "candidate_id": 2,
            "manipulation_delta_rad": {name: 0.0 for name in ACTIVE_ACTUATORS},
            "summary": _refinement_summary(
                full=True, median=0.010, minimum=0.008, topology=0.7
            ),
        },
        {
            "candidate_id": 1,
            "manipulation_delta_rad": {name: 0.0 for name in ACTIVE_ACTUATORS},
            "summary": _refinement_summary(
                full=False, median=0.009, minimum=0.007, topology=0.95
            ),
        },
    ]
    ranked = manipulation.rank_manipulation_candidates(records, config=config)
    assert [record["candidate_id"] for record in ranked] == [2, 3, 1]
    evidence = manipulation.manipulation_candidate_rank_evidence(
        ranked[0], config=config
    )
    assert evidence["full_success"] is True
    assert evidence["lift_min_normalized_margin"] == pytest.approx(0.0)


def test_top_eight_by_128_refinement_is_deterministic_balanced_and_resumable() -> None:
    config = _config()
    model = _synthetic_model()
    bounds = manipulation.manipulation_delta_bounds(model, config)
    parents = []
    for candidate_id in range(8):
        delta = {name: 0.0 for name in ACTIVE_ACTUATORS}
        delta["left_hand_index_joint1_actuator"] = 0.01 * candidate_id
        parents.append(
            {
                "candidate_id": candidate_id,
                "manipulation_delta_rad": delta,
                "summary": _refinement_summary(
                    full=False,
                    median=0.001 * candidate_id,
                    minimum=0.0008 * candidate_id,
                    topology=0.5 + 0.01 * candidate_id,
                ),
            }
        )
    budget = manipulation.LocalRefinementBudget()
    first = manipulation.generate_local_refinement_candidates(
        parents, bounds, config=config, budget=budget
    )
    second = manipulation.generate_local_refinement_candidates(
        list(reversed(parents)), bounds, config=config, budget=budget
    )
    assert first == second
    assert len(first) == budget.maximum_candidate_count == 1024
    assert {value["parent_rank"] for value in first[:8]} == set(range(8))
    assert all(value["local_index"] == 0 for value in first[:8])
    assert len({value["candidate_id"] for value in first}) == 1024
    for value in first:
        for name in ACTIVE_ACTUATORS:
            assert bounds[name][0] <= value["manipulation_delta_rad"][name] <= bounds[name][1]
    pending = manipulation.pending_local_refinement_candidates(
        first, [{"candidate_id": value["candidate_id"]} for value in first[:64]]
    )
    assert len(pending) == 960
    assert {value["candidate_id"] for value in pending}.isdisjoint(
        {value["candidate_id"] for value in first[:64]}
    )


def test_multiconfig_refinement_ranks_and_clips_each_parent_independently() -> None:
    template = _config()
    parents = []
    bounds_by_id = {}
    for candidate_id in range(8):
        config = copy.deepcopy(template)
        delta = {name: 0.0 for name in ACTIVE_ACTUATORS}
        half_width = 0.01 * (candidate_id + 1)
        bounds_by_id[candidate_id] = {
            name: (-half_width, half_width) for name in ACTIVE_ACTUATORS
        }
        parents.append(
            {
                "candidate_id": candidate_id,
                "config": config,
                "manipulation_delta_rad": delta,
                "summary": _refinement_summary(
                    full=False,
                    median=0.001 * candidate_id,
                    minimum=0.0008 * candidate_id,
                    topology=0.5 + 0.01 * candidate_id,
                ),
            }
        )

    def resolve(parent):
        return bounds_by_id[int(parent["candidate_id"])]

    budget = manipulation.LocalRefinementBudget(candidates_per_parent=8)
    first = manipulation.generate_multiconfig_local_refinement_candidates(
        parents, budget=budget, bounds_resolver=resolve
    )
    second = manipulation.generate_multiconfig_local_refinement_candidates(
        list(reversed(parents)), budget=budget, bounds_resolver=resolve
    )
    assert first == second
    assert len(first) == 64
    for candidate in first:
        parent_bounds = bounds_by_id[candidate["parent_candidate_id"]]
        for name, value in candidate["manipulation_delta_rad"].items():
            assert parent_bounds[name][0] <= value <= parent_bounds[name][1]


def test_local_refinement_batch_preserves_large_ids_and_worker_order() -> None:
    config = _config()
    model = _synthetic_model()
    candidates = []
    for local_index in range(3):
        candidates.append(
            {
                "candidate_id": 109_000_000_000_000 + local_index,
                "parent_candidate_id": 62,
                "parent_rank": 0,
                "local_index": local_index,
                "manipulation_delta_rad": {
                    name: 0.0 for name in ACTIVE_ACTUATORS
                },
            }
        )

    def executor(jobs, workers):
        assert workers == 4
        return tuple(
            {
                "candidate_id": job["candidate_id"],
                "summary": _refinement_summary(
                    full=False,
                    median=0.001 * job["candidate_index"],
                    minimum=0.0008 * job["candidate_index"],
                    topology=0.5,
                ),
            }
            for job in reversed(jobs)
        )

    result = manipulation.run_local_refinement_batch(
        config, candidates, model=model, workers=4, executor=executor
    )
    assert [value["candidate_id"] for value in result["candidates"]] == [
        value["candidate_id"] for value in candidates
    ]
    assert result["ranked_candidates"][0]["candidate_id"] == candidates[2]["candidate_id"]
    assert all(
        value["job_metadata"]["parent_candidate_id"] == 62
        for value in result["candidates"]
    )
