from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import (
    ACTIVE_ACTUATORS,
    INACTIVE_ACTUATORS,
    load_config,
    validate_config,
)
from xhand_grasp.experiment import JointPairFeedbackParameters
from xhand_grasp.tuning.joint_pair_near_zero_campaign_runner import (
    V15CampaignBackend,
    V15CampaignJob,
    build_static_jobs,
)
import xhand_grasp.tuning.joint_pair_near_zero_physics_backend as physics_backend
from xhand_grasp.tuning.joint_pair_near_zero_physics_backend import (
    JointPairNearZeroPhysicsStageRunner,
    _materialize_grasp_variant,
    _materialize_feedback_variant,
    _perturbation_configs,
    _refined_config,
    create_joint_pair_near_zero_campaign_backend,
)
from xhand_grasp.v15_identity import validate_v15_top_level_identities


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "grasp_configs/left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift.json"
)


def _job(
    stage: str, candidate_id: int, parent_candidate_id: int, payload: dict
) -> V15CampaignJob:
    return V15CampaignJob(
        stage=stage,
        index=0,
        parent_candidate_id=parent_candidate_id,
        payload=payload,
        candidate_id=candidate_id,
    )


def test_production_backend_factory_and_input_validation() -> None:
    backend = create_joint_pair_near_zero_campaign_backend(
        workers=2, seed=20260821
    )
    assert isinstance(backend, V15CampaignBackend)
    assert isinstance(backend.stage_runner, JointPairNearZeroPhysicsStageRunner)
    assert backend.stage_runner.workers == 2
    with pytest.raises(ValueError, match="workers"):
        JointPairNearZeroPhysicsStageRunner(workers=0)
    with pytest.raises(ValueError, match="seed"):
        JointPairNearZeroPhysicsStageRunner(seed=-1)


def test_feedback_materialization_rebinds_feedback_and_all_v15_ids(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(config), encoding="utf-8")
    parent = {"candidate_id": 91, "config_path": str(parent_path)}
    job = _job(
        "feedback_grid",
        92,
        91,
        {
            "feedback_variant": {
                "schema_version": 1,
                "index": 0,
                "alignment_gain": 0.75,
                "slip_recovery_gain_rad_per_m": 6.0,
            }
        },
    )
    resolved = _materialize_feedback_variant(parent, job)
    feedback = JointPairFeedbackParameters.from_config(
        resolved["joint_pair_feedback"]
    )
    assert feedback.alignment_gain == 0.75
    assert feedback.slip_recovery_gain_rad_per_m == 6.0
    assert resolved["candidate_metadata"]["candidate_id"] == 92
    validate_v15_top_level_identities(resolved)
    validate_config(resolved)


def test_feedback_refinement_is_deterministic_bounded_and_pose_preserving(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    parent_path = tmp_path / "feedback_parent.json"
    parent_path.write_text(json.dumps(config), encoding="utf-8")
    parent = {"candidate_id": 9101, "config_path": str(parent_path)}
    selection = {
        "schema_version": 1,
        "plan_candidate_id": 8101,
        "mode": "eligible",
        "eligible": True,
        "requirements": {"grasp_success": True},
        "failed_requirements": [],
    }
    job = _job(
        "feedback_refinement",
        9201,
        9101,
        {
            "feedback_candidate_id": 9101,
            "plan_candidate_id": 8101,
            "local_index": 7,
            "reprobe_if_pose_changed": True,
            "refinement_parent_selection": selection,
        },
    )
    first = _refined_config(parent, job, seed=20260821)
    second = _refined_config(parent, job, seed=20260821)
    assert first == second
    for field in ("cube", "hand_pose", "grasp_pose"):
        assert first[field] == config[field]
    assert first["object_config_id"] == config["object_config_id"]
    assert first["grasp_pose_id"] == config["grasp_pose_id"]
    assert first["grasp_object_pair_id"] == config["grasp_object_pair_id"]

    before = config["control"]["contact_preload_targets_rad"]
    after = first["control"]["contact_preload_targets_rad"]
    assert set(after) == set(ACTIVE_ACTUATORS)
    assert not set(after).intersection(INACTIVE_ACTUATORS)
    bounds = physics_backend.resolve_experiment(
        first
    ).search_bounds.actuator_targets_rad
    assert bounds is not None
    assert any(abs(after[name] - before[name]) > 0.0 for name in ACTIVE_ACTUATORS)
    for name in ACTIVE_ACTUATORS:
        assert bounds[name][0] <= after[name] <= bounds[name][1]
        assert abs(after[name] - before[name]) <= 0.006 + 1e-12

    metadata = first["candidate_metadata"]["v15_campaign_job"][
        "feedback_refinement"
    ]
    assert metadata["pose_changed"] is False
    assert metadata["reprobe_required"] is False
    assert metadata["inactive_actuator_commands"] == "implicit_zero_unchanged"
    assert metadata["parent_selection"] == selection
    assert set(metadata["contact_preload_applied_residual_rad"]) == set(
        ACTIVE_ACTUATORS
    )
    validate_v15_top_level_identities(first)
    validate_config(first)


def test_feedback_refinement_covers_structured_physical_probe_grid(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    config["joint_pair_feedback"] = physics_backend._pair_feedback_mapping(
        config["joint_pair_feedback"], 1.0, 2.0
    )
    physics_backend.install_v15_top_level_identities(config)
    validate_config(config)
    parent_path = tmp_path / "structured_parent.json"
    parent_path.write_text(json.dumps(config), encoding="utf-8")
    parent = {"candidate_id": 9301, "config_path": str(parent_path)}
    expected_probe = np.asarray(
        [-0.00134, 0.00048, -0.01038, -0.00572,
         -0.01161, 0.00744, -0.00977, 0.00961],
        dtype=np.float64,
    )
    expected_thumb = {-0.003, -0.004, -0.005}
    expected_mid = {0.002, 0.003, 0.004}
    expected_scales = {0.75, 1.0, 1.15}
    observed_combinations: set[tuple[float, float]] = set()
    observed_scales: set[float] = set()
    base_waypoints = config["manipulation_plan"]["actuator_waypoints_rad"]
    preload = config["control"]["contact_preload_targets_rad"]
    bounds = physics_backend.resolve_experiment(
        config
    ).search_bounds.manipulation_delta_rad
    assert bounds is not None

    for local_index in range(27):
        job = _job(
            "feedback_refinement",
            9400 + local_index,
            9301,
            {
                "feedback_candidate_id": 9301,
                "plan_candidate_id": 8301,
                "local_index": local_index,
                "reprobe_if_pose_changed": True,
            },
        )
        resolved = _refined_config(parent, job, seed=20260821)
        metadata = resolved["candidate_metadata"]["v15_campaign_job"][
            "feedback_refinement"
        ]
        branch = metadata["refinement_branch"]
        assert branch["name"] == "structured_physical_probe"
        assert metadata["alignment_gain"] == 1.0
        assert metadata["slip_recovery_gain_rad_per_m"] == 2.0
        assert resolved["joint_pair_feedback"]["alignment_gain"] == 1.0
        assert (
            resolved["joint_pair_feedback"][
                "slip_recovery_gain_rad_per_m"
            ]
            == 2.0
        )
        observed_combinations.add(
            (
                branch["thumb_rota1_preload_delta_rad"],
                branch["mid_joint1_preload_delta_rad"],
            )
        )
        scale = float(branch["waypoint_scale"])
        observed_scales.add(scale)
        np.testing.assert_allclose(
            [
                branch["physical_probe_waypoint_delta_rad"][name]
                for name in ACTIVE_ACTUATORS
            ],
            expected_probe,
            rtol=0.0,
            atol=0.0,
        )
        envelope = np.asarray(metadata["waypoint_node_envelope"])
        np.testing.assert_array_equal(envelope[:5], 0.0)
        assert envelope[5] == 0.5
        np.testing.assert_array_equal(envelope[6:], 1.0)

        thumb_name = "left_hand_thumb_rota_joint1_actuator"
        mid_name = "left_hand_mid_joint1_actuator"
        assert resolved["control"]["contact_preload_targets_rad"][thumb_name] == pytest.approx(
            preload[thumb_name] + branch["thumb_rota1_preload_delta_rad"]
        )
        assert resolved["control"]["contact_preload_targets_rad"][mid_name] == pytest.approx(
            preload[mid_name] + branch["mid_joint1_preload_delta_rad"]
        )
        for index, name in enumerate(ACTIVE_ACTUATORS):
            values = np.asarray(
                resolved["manipulation_plan"]["actuator_waypoints_rad"][name]
            )
            original = np.asarray(base_waypoints[name])
            assert values[5] - original[5] == pytest.approx(
                0.5 * scale * expected_probe[index]
            )
            assert values[6] - original[6] == pytest.approx(
                scale * expected_probe[index]
            )
            assert values[10] - original[10] == pytest.approx(
                scale * expected_probe[index]
            )
            assert np.max(np.abs(np.diff(values))) <= 0.04 + 1e-12
            assert np.all(values >= bounds[name][0] - 1e-12)
            assert np.all(values <= bounds[name][1] + 1e-12)
        for field in ("cube", "hand_pose", "grasp_pose"):
            assert resolved[field] == config[field]
        for threshold in (
            "slip_freeze_threshold_m",
            "slip_abort_threshold_m",
        ):
            assert resolved["joint_pair_feedback"][threshold] == config[
                "joint_pair_feedback"
            ][threshold]
        validate_config(resolved)

    assert observed_combinations == {
        (thumb, mid) for thumb in expected_thumb for mid in expected_mid
    }
    assert observed_scales == expected_scales

    random_job = _job(
        "feedback_refinement",
        9500,
        9301,
        {
            "feedback_candidate_id": 9301,
            "plan_candidate_id": 8301,
            "local_index": 27,
            "reprobe_if_pose_changed": True,
        },
    )
    random_config = _refined_config(parent, random_job, seed=20260821)
    random_branch = random_config["candidate_metadata"]["v15_campaign_job"][
        "feedback_refinement"
    ]["refinement_branch"]
    assert random_branch["name"] == "random_local"
    assert random_branch["physical_probe_waypoint_delta_rad"] is None
    random_metadata = random_config["candidate_metadata"]["v15_campaign_job"][
        "feedback_refinement"
    ]
    assert (
        random_metadata["alignment_gain"],
        random_metadata["slip_recovery_gain_rad_per_m"],
    ) != (1.0, 2.0)
    validate_config(random_config)


def test_dynamic_grasp_zero_plan_retains_authenticated_planning_warm_start(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    warm_plan = config["manipulation_plan"]
    parent_path = tmp_path / "static.json"
    parent_path.write_text(json.dumps(config), encoding="utf-8")
    job = _job(
        "dynamic_grasp",
        82,
        81,
        {
            "controller_variant": {
                "schema_version": 1,
                "index": 0,
                "close_s": 1.25,
                "mode": "original",
            }
        },
    )
    resolved = _materialize_grasp_variant(
        {"candidate_id": 81, "config_path": str(parent_path)}, job
    )
    assert all(
        np.array_equal(values, np.zeros(21))
        for values in resolved["manipulation_plan"][
            "actuator_waypoints_rad"
        ].values()
    )
    persisted = resolved["candidate_metadata"]["v15_campaign_job"][
        "planning_warm_start_manipulation_plan"
    ]
    assert persisted == warm_plan
    assert any(
        abs(value) > 0.0
        for values in persisted["actuator_waypoints_rad"].values()
        for value in values
    )
    validate_config(resolved)


def test_perturbations_are_deterministic_free_cube_full_reset_configs() -> None:
    base = load_config(CONFIG)
    jobs = tuple(
        _job(
            "robustness",
            200 + index,
            101,
            {"trial_index": index, "seed": 20260821},
        )
        for index in range(4)
    )
    first = _perturbation_configs(
        base, jobs, family="robustness", seed=20260821
    )
    second = _perturbation_configs(
        base, jobs, family="robustness", seed=20260821
    )
    assert first == second
    assert len({tuple(value["cube"]["center_xy_m"]) for value in first.values()}) == 4
    for job in jobs:
        value = first[job.candidate_id]
        assert value["run_context"] == {"kind": "robustness_trial"}
        assert value["candidate_metadata"]["candidate_id"] == job.candidate_id
        assert value["cube"]["edge_m"] == 0.079
        assert value["cube"]["freejoint"] if "freejoint" in value["cube"] else True
        validate_v15_top_level_identities(value)
        validate_config(value)


def test_real_mujoco_static_zero_offset_anchor_is_physically_screened(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    job = build_static_jobs()[0]
    runner = JointPairNearZeroPhysicsStageRunner(workers=1)
    records = runner(
        "static_filter",
        (job,),
        tmp_path / "campaign",
        {"seed_config_path": str(seed_path)},
    )
    assert len(records) == 1
    record = records[0]
    assert record["candidate_id"] == job.candidate_id
    assert record["static_pass"] is True
    assert record["trace_path"] is None
    metrics = record["summary"]["metrics"]["static_filter"]
    assert metrics["static_geometry_pass"] is True
    assert metrics["point_target"]["stop_reason"] in {
        "dls_converged",
        "dls_no_safe_improvement",
        "dls_maximum_iterations",
    }
    assert metrics["joint_pair"]["positive_y"] is True
    assert metrics["joint_pair"]["angle_deg"] <= 0.5
    assert metrics["active_finger_self_collision"]["active"] is False
    persisted = load_config(record["config_path"])
    validate_v15_top_level_identities(persisted)
    resumed = runner(
        "static_filter",
        (job,),
        tmp_path / "campaign",
        {"seed_config_path": str(seed_path)},
    )
    assert resumed[0]["artifact_reused"] is True
    assert resumed[0]["config_path"] == record["config_path"]


def test_invalid_static_pose_is_a_resumable_failure_with_valid_seed_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_path = tmp_path / "seed.json"
    seed_config = load_config(CONFIG)
    seed_path.write_text(json.dumps(seed_config), encoding="utf-8")
    anchor_job, rejected_job = build_static_jobs()[0], build_static_jobs()[4]
    runner = JointPairNearZeroPhysicsStageRunner(workers=1)
    workspace = tmp_path / "formal_campaign_resume"

    # Simulate the prefix already committed before power loss/failure.
    prefix = runner(
        "static_filter",
        (anchor_job,),
        workspace,
        {"seed_config_path": str(seed_path)},
    )
    assert prefix[0]["static_pass"] is True

    original_solver = physics_backend.solve_point_target_dls

    def reject_solver(*args: object, **kwargs: object) -> object:
        variables = kwargs["initial_variables"]
        if (
            np.linalg.norm(variables.root_delta_cube_m) <= 1e-15
            and np.linalg.norm(variables.wrist_local_rotvec_rad) <= 1e-15
            and np.allclose(
                variables.actual_joint_qpos_rad,
                [
                    seed_config["grasp_pose"]["nominal_joint_qpos_rad"][name]
                    for name in ACTIVE_ACTUATORS
                ],
                rtol=0.0,
                atol=1e-15,
            )
        ):
            return original_solver(*args, **kwargs)
        raise ValueError("forced DLS rejection before persistence")

    def invalid_materialization(*args: object, **kwargs: object) -> dict:
        invalid = json.loads(json.dumps(seed_config))
        # Local +Z is vertical here, so this is outside the registered 30--40
        # degree finger-down envelope and must never be persisted as config.
        invalid["hand_pose"]["rpy_deg"] = [0.0, 0.0, 0.0]
        return invalid

    monkeypatch.setattr(physics_backend, "solve_point_target_dls", reject_solver)
    monkeypatch.setattr(
        physics_backend,
        "materialize_point_target_candidate",
        invalid_materialization,
    )
    rejected_prefix = runner(
        "static_filter",
        (rejected_job,),
        workspace,
        {"seed_config_path": str(seed_path)},
    )
    assert prefix[0]["artifact_reused"] is False
    rejected = rejected_prefix[0]
    assert rejected["static_pass"] is False
    assert rejected["artifact_reused"] is False

    persisted = load_config(rejected["config_path"])
    validate_config(persisted)
    assert persisted["hand_pose"] == seed_config["hand_pose"]
    metadata = persisted["candidate_metadata"]["v15_campaign_job"]
    diagnostics = metadata["static_candidate_rejection_diagnostics"]
    assert diagnostics[-1]["phase"] == "resolved_static_candidate_validation"
    assert "finger_down_tilt_deg" in diagnostics[-1]["error"]
    assert diagnostics[-1]["attempted_config_persisted"] is False
    assert diagnostics[-1]["attempted_resolved_pose_constraints"][
        "finger_down_tilt_deg"
    ] == pytest.approx(-90.0)
    result = json.loads(Path(rejected["result_path"]).read_text(encoding="utf-8"))
    assert result["static_pass"] is False
    assert "finger_down_tilt_deg" in result["error"]
    assert result["summary"]["metrics"]["static_filter"]["point_target"][
        "stop_reason"
    ] == "invalid_resolved_static_candidate"

    resumed_rejected = runner(
        "static_filter",
        (rejected_job,),
        workspace,
        {"seed_config_path": str(seed_path)},
    )
    assert resumed_rejected[0]["artifact_reused"] is True
    monkeypatch.undo()
    resumed_anchor = runner(
        "static_filter",
        (anchor_job,),
        workspace,
        {"seed_config_path": str(seed_path)},
    )
    assert resumed_anchor[0]["artifact_reused"] is True


def test_backend_rejects_unknown_stage(tmp_path: Path) -> None:
    runner = JointPairNearZeroPhysicsStageRunner()
    job = _job("unknown", 2, 1, {})
    with pytest.raises(ValueError, match="unsupported"):
        runner(
            "unknown",
            (job,),
            tmp_path,
            {"parents": ({"candidate_id": 1},)},
        )
