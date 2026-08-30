from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    bind_candidate_result_semantic_sha256,
)
from xhand_grasp.artifacts import write_json
from xhand_grasp.config import load_config
from xhand_grasp.tuning.joint_pair_near_zero_campaign_runner import (
    V15CampaignJob,
)
from xhand_grasp.tuning.joint_pair_near_zero_candidate_artifacts import (
    COMPACT_TRACE_RETENTION_POLICY,
    TRACE_RETENTION_POLICY,
    authenticate_v15_candidate_artifacts,
    run_or_resume_v15_candidate_artifacts,
)
import xhand_grasp.tuning.joint_pair_near_zero_physics_backend as backend


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "grasp_configs/left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift.json"
)


class _Session:
    def __init__(self, *, grasp: bool, full: bool) -> None:
        self._complete = False
        self._grasp = grasp
        self._full = full

    @property
    def complete(self) -> bool:
        return self._complete

    def advance_one(self) -> None:
        self._complete = True

    def finalize(self, *, trace_path: str | Path | None = None) -> dict:
        if trace_path is not None:
            np.savez_compressed(trace_path, time_s=np.asarray([0.0]))
        return {
            "passed": self._full,
            "stage_status": {
                "grasp_success": self._grasp,
                "full_success": self._full,
            },
            "metrics": {},
        }

    def close(self) -> None:
        pass


def _factory(*, grasp: bool, full: bool):
    return lambda _config: _Session(grasp=grasp, full=full)


def test_v15_default_retention_remains_the_sealed_v1_contract(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "candidate_101"
    bundle = run_or_resume_v15_candidate_artifacts(
        load_config(CONFIG),
        destination,
        101,
        session_factory=_factory(grasp=True, full=False),
    )
    assert bundle.trace_path == destination / "trace.npz"
    assert bundle.result["trace_retention"] == {
        "schema_version": 1,
        "policy": TRACE_RETENTION_POLICY,
        "retained": True,
        "reason": "grasp_success",
    }
    authenticate_v15_candidate_artifacts(
        destination, expected_retain_grasp_trace=True
    )


def test_v15_compact_v2_omits_grasp_trace_and_binds_resume_policy(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "candidate_102"
    config = load_config(CONFIG)
    bundle = run_or_resume_v15_candidate_artifacts(
        config,
        destination,
        102,
        retain_grasp_trace=False,
        session_factory=_factory(grasp=True, full=False),
    )
    assert bundle.trace_path is None
    assert not (destination / "trace.npz").exists()
    assert bundle.result["trace_retention"] == {
        "schema_version": 2,
        "policy": COMPACT_TRACE_RETENTION_POLICY,
        "retain_grasp_trace": False,
        "final_rerun": False,
        "retained": False,
        "reason": "grasp_success_summary_only",
    }
    resumed = run_or_resume_v15_candidate_artifacts(
        config,
        destination,
        102,
        retain_grasp_trace=False,
        session_factory=lambda _config: pytest.fail("compact candidate reran"),
    )
    assert resumed.reused is True
    with pytest.raises(RuntimeError, match="trace-retention policy changed"):
        run_or_resume_v15_candidate_artifacts(
            config,
            destination,
            102,
            session_factory=lambda _config: pytest.fail("policy mismatch reran"),
        )


def test_v15_compact_contract_rejects_semantically_rehashed_tampering(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "candidate_103"
    run_or_resume_v15_candidate_artifacts(
        load_config(CONFIG),
        destination,
        103,
        retain_grasp_trace=False,
        session_factory=_factory(grasp=True, full=False),
    )
    result_path = destination / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["trace_retention"]["final_rerun"] = True
    write_json(result_path, bind_candidate_result_semantic_sha256(result))
    with pytest.raises(RuntimeError, match="retention decision is inconsistent"):
        authenticate_v15_candidate_artifacts(destination)


@pytest.mark.parametrize(
    ("full", "final_rerun", "reason"),
    ((True, False, "full_success"), (False, True, "explicit_final_rerun")),
)
def test_v15_compact_policy_still_forces_success_and_final_traces(
    tmp_path: Path,
    full: bool,
    final_rerun: bool,
    reason: str,
) -> None:
    candidate_id = 104 + int(final_rerun)
    destination = tmp_path / f"candidate_{candidate_id}"
    bundle = run_or_resume_v15_candidate_artifacts(
        load_config(CONFIG),
        destination,
        candidate_id,
        final_rerun=final_rerun,
        retain_grasp_trace=False,
        session_factory=_factory(grasp=True, full=full),
    )
    assert bundle.trace_path == destination / "trace.npz"
    assert bundle.result["trace_retention"]["reason"] == reason
    assert bundle.result["trace_retention"]["retained"] is True
    authenticate_v15_candidate_artifacts(
        destination,
        require_retained_trace=True,
        expected_retain_grasp_trace=False,
    )


def test_v15_worker_task_forwards_compact_policy_and_final_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[bool, bool]] = []

    def fake_run(
        config: dict,
        destination: str,
        candidate_id: int,
        *,
        final_rerun: bool,
        retain_grasp_trace: bool,
    ) -> SimpleNamespace:
        calls.append((final_rerun, retain_grasp_trace))
        return SimpleNamespace(
            candidate_id=candidate_id,
            config_path=Path(destination) / "resolved_config.json",
            result_path=Path(destination) / "result.json",
            trace_path=(Path(destination) / "trace.npz") if final_rerun else None,
            reused=False,
            result={
                "grasp_success": True,
                "full_success": False,
                "summary": {"metrics": {}},
            },
        )

    monkeypatch.setattr(
        backend, "run_or_resume_v15_candidate_artifacts", fake_run
    )
    job = V15CampaignJob(
        stage="feedback_refinement",
        index=0,
        parent_candidate_id=1,
        payload={},
        candidate_id=201,
    )
    task_base = (
        copy.deepcopy(load_config(CONFIG)),
        str(tmp_path / "candidate_201"),
        201,
    )
    backend._simulation_task((*task_base, False, False, job.descriptor()))
    backend._simulation_task((*task_base, True, False, job.descriptor()))
    assert calls == [(False, False), (True, False)]


def test_v15_stage_context_places_compact_policy_in_worker_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple] = []
    config = load_config(CONFIG)
    config_path = tmp_path / "parent.json"
    write_json(config_path, config)
    parent = {"candidate_id": 301, "config_path": str(config_path)}
    job = V15CampaignJob(
        stage="feedback_grid",
        index=0,
        parent_candidate_id=301,
        payload={
            "feedback_variant": {
                "schema_version": 1,
                "index": 0,
                "alignment_gain": 0.25,
                "slip_recovery_gain_rad_per_m": 2.0,
            }
        },
        candidate_id=302,
    )

    def fake_tasks(tasks, *, workers):
        captured.extend(tasks)
        return ({"candidate_id": 302},)

    monkeypatch.setattr(backend, "_run_simulation_tasks", fake_tasks)
    runner = backend.JointPairNearZeroPhysicsStageRunner(workers=1)
    runner(
        "feedback_grid",
        (job,),
        tmp_path / "workspace",
        {"parents": (parent,), "retain_grasp_trace": False},
    )
    assert len(captured) == 1
    assert captured[0][3] is False  # not an exact/final rerun
    assert captured[0][4] is False  # compact grasp trace policy
    with pytest.raises(TypeError, match="retain_grasp_trace"):
        runner(
            "feedback_grid",
            (job,),
            tmp_path / "workspace",
            {"parents": (parent,), "retain_grasp_trace": "false"},
        )
