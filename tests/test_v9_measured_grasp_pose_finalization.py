from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
    bind_candidate_result_semantic_sha256,
)
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.grasp_pose import canonical_sha256, controller_id, grasp_pose_id
from xhand_grasp.tuning.actual_contact_grasp_pose_measured import (
    evaluate_measured_finalization_job,
    extract_measured_grasp_qpos,
    materialize_measured_grasp_config,
    measured_grasp_pose_succeeded,
    prepare_measured_finalization_jobs,
    run_or_resume_measured_grasp_finalization,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "grasp_configs"
    / "left_opposed_face_palm_down_actual_contact_grasp_pose_smooth_vertical_lift.json"
)
START = 25
END = 274
TOTAL = 300


def _summary(*, grasp: bool) -> dict:
    return {
        "passed": False,
        "failed_checks": ["operation_median_lift_reached"],
        "checks": {
            "v6_initial_joint_state_matches_pregrasp_config": True,
            "no_hand_cube_contact_during_settle": True,
        },
        "stage_status": {
            "grasp_success": grasp,
            "manipulation_success": False,
            "full_success": False,
        },
    }


def _write_trace(path: Path, qpos: np.ndarray, *, grasp: bool = True) -> None:
    history = np.broadcast_to(qpos, (TOTAL, len(ACTIVE_ACTUATORS))).copy()
    np.savez_compressed(
        path,
        grasp_stable_window_start_step=np.asarray(START if grasp else -1),
        grasp_stable_window_end_step=np.asarray(END if grasp else -1),
        grasp_lock_step=np.asarray(END if grasp else -1),
        grasp_pose_actual_joint_qpos_rad=history,
        grasp_pose_actual_qpos_rad=(qpos if grasp else np.zeros_like(qpos)),
        initialized_at_pregrasp=np.asarray(True),
    )


def _source(root: Path) -> tuple[Path, dict, dict, np.ndarray]:
    dynamic = root / "dynamic"
    directory = dynamic / "candidates" / "candidate_7"
    directory.mkdir(parents=True)
    config = load_config(TEMPLATE)
    actual = np.asarray(
        [config["grasp_pose"]["nominal_joint_qpos_rad"][name] for name in ACTIVE_ACTUATORS],
        dtype=np.float64,
    )
    actual[0] = 1.45
    actual[1:] += np.linspace(0.001, 0.007, 7)
    config_path = directory / "resolved_config.json"
    trace_path = directory / "trace.npz"
    result_path = directory / "result.json"
    write_json(config_path, config)
    _write_trace(trace_path, actual)
    result = bind_candidate_result_semantic_sha256({
        "candidate_result_schema_version": 1,
        "complete": True,
        "candidate_id": 7,
        "candidate_sha256": canonical_sha256(config),
        "grasp_pose_id": grasp_pose_id(config),
        "controller_id": controller_id(config),
        "grasp_success": True,
        "summary": _summary(grasp=True),
        "artifacts": {
            "resolved_config": config_path.name,
            "trace": trace_path.name,
            "sha256": {
                "resolved_config": file_sha256(config_path),
                "trace": file_sha256(trace_path),
            },
        },
    })
    write_json(result_path, result)
    record = {
        **result,
        "config": copy.deepcopy(config),
        "artifact_directory": "candidates/candidate_7",
    }
    return dynamic, record, config, actual


def _fake_runner_factory(first_actual: np.ndarray, fixed_actual: np.ndarray):
    calls: list[dict] = []

    def run(config, *, trace_path, video_path):
        assert video_path is None
        calls.append(copy.deepcopy(config))
        actual = first_actual if len(calls) == 1 else fixed_actual
        _write_trace(Path(trace_path), actual)
        return _summary(grasp=True)

    return calls, run


def test_large_preload_command_cannot_rebind_measured_grasp_pose(tmp_path):
    dynamic, record, config, actual = _source(tmp_path)
    assert config["control"]["contact_preload_targets_rad"][ACTIVE_ACTUATORS[0]] == 1.6
    measured = extract_measured_grasp_qpos(
        config,
        dynamic / record["artifact_directory"] / "trace.npz",
        record,
    )
    rebound = materialize_measured_grasp_config(
        config,
        measured,
        source_candidate_id=7,
        source_candidate_sha256=record["candidate_sha256"],
        source_trace_sha256=file_sha256(
            dynamic / record["artifact_directory"] / "trace.npz"
        ),
    )
    nominal = rebound["grasp_pose"]["nominal_joint_qpos_rad"]
    assert [nominal[name] for name in ACTIVE_ACTUATORS] == actual.tolist()
    assert nominal[ACTIVE_ACTUATORS[0]] == 1.45
    assert rebound["control"]["contact_preload_targets_rad"][ACTIVE_ACTUATORS[0]] == 1.6
    assert rebound["candidate_metadata"]["measured_grasp_pose_finalization"][
        "preload_command_used_as_pose_evidence"
    ] is False
    assert controller_id(rebound) == controller_id(config)


def test_measured_finalization_iterates_to_exact_pose_and_reacquires_from_reset(
    tmp_path,
):
    dynamic, record, source_config, source_actual = _source(tmp_path)
    source_before = copy.deepcopy(source_config)
    next_actual = source_actual.copy()
    next_actual[3] += 0.001
    calls, runner = _fake_runner_factory(next_actual, next_actual)
    job = prepare_measured_finalization_jobs((record,), dynamic)[0]
    result = evaluate_measured_finalization_job(job, simulation_runner=runner)

    assert len(calls) == 2
    assert source_config == source_before
    assert measured_grasp_pose_succeeded(result)
    assert result["fixed_point"]["converged_exactly"] is True
    assert result["fixed_point"]["iterations_executed"] == 2
    assert result["initial_state_source"] == "configured_no_contact_reset"
    assert result["checkpoint_used"] is False
    assert result["finalization_checks"]["full_reset_reacquisition"] is True
    assert result["controller_id"] == record["controller_id"]
    assert result["grasp_pose_id"] == grasp_pose_id(result["config"])
    assert result["grasp_pose_id"] != record["grasp_pose_id"]
    assert result["candidate_sha256"] == canonical_sha256(result["config"])
    assert result["candidate_sha256"] != record["candidate_sha256"]
    configured = np.asarray(
        [
            result["config"]["grasp_pose"]["nominal_joint_qpos_rad"][name]
            for name in ACTIVE_ACTUATORS
        ]
    )
    assert np.array_equal(configured, next_actual)
    persisted = Path(job["output_directory"])
    assert persisted.is_dir()
    payload = json.loads((persisted / "result.json").read_text(encoding="utf-8"))
    authenticate_candidate_result_semantic_sha256(
        payload, source=persisted / "result.json"
    )
    assert payload["source_provenance"]["source_trace_sha256"] == file_sha256(
        dynamic / record["artifact_directory"] / "trace.npz"
    )


def test_measured_finalization_resume_authenticates_and_skips_executor(tmp_path):
    dynamic, record, _config, actual = _source(tmp_path)

    def executor(jobs, workers):
        assert workers == 1
        return tuple(
            evaluate_measured_finalization_job(
                job,
                simulation_runner=_fake_runner_factory(actual, actual)[1],
            )
            for job in jobs
        )

    first = run_or_resume_measured_grasp_finalization(
        (record,), dynamic, workers=1, executor=executor
    )
    assert len(first) == 1 and measured_grasp_pose_succeeded(first[0])

    def must_not_run(_jobs, _workers):  # pragma: no cover - assertion path
        raise AssertionError("authenticated measured candidate should be reused")

    resumed = run_or_resume_measured_grasp_finalization(
        (record,),
        dynamic,
        workers=1,
        resume=True,
        executor=must_not_run,
    )
    assert resumed[0]["reused"] is True
    assert resumed[0]["candidate_sha256"] == first[0]["candidate_sha256"]


def test_failed_reacquisition_is_persisted_as_diagnostic(tmp_path):
    dynamic, record, _config, _actual = _source(tmp_path)
    job = prepare_measured_finalization_jobs((record,), dynamic)[0]

    def failed_runner(_config, *, trace_path, video_path):
        assert video_path is None
        _write_trace(Path(trace_path), np.zeros(8), grasp=False)
        return _summary(grasp=False)

    result = evaluate_measured_finalization_job(
        job, simulation_runner=failed_runner
    )
    assert not measured_grasp_pose_succeeded(result)
    assert result["classification"] == "measured_actual_contact_grasp_pose_rejected"
    assert result["fixed_point"]["iterations_executed"] == 1
    assert result["actual_grasp_pose_qpos_rad"] is None
    assert Path(job["output_directory"], "result.json").is_file()
