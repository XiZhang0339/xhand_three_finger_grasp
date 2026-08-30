from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
)
from xhand_grasp.artifacts import write_json
from xhand_grasp.config import load_config
from xhand_grasp.tuning.contact_preserving_candidate_artifacts import (
    TRACE_RETENTION_POLICY,
    authenticate_v14_candidate_artifacts,
    run_or_resume_v14_candidate_artifacts,
)
from xhand_grasp.tuning.contact_preserving_planned_lift_campaign import (
    _candidate_artifact_paths,
)


CONFIG = Path(
    "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


def _summary(*, grasp: bool, full: bool) -> dict[str, Any]:
    return {
        "passed": bool(full),
        "failed_checks": [] if full else ["operation_median_lift_reached"],
        "stage_status": {
            "grasp_success": bool(grasp),
            "manipulation_success": bool(full),
            "full_success": bool(full),
        },
        # Confirm strict serialization of real evaluator NumPy scalar values.
        "finite": np.bool_(True),
    }


class _FakeSession:
    def __init__(self, summary: dict[str, Any], *, fail: bool = False) -> None:
        self.summary = copy.deepcopy(summary)
        self.step = 0
        self.fail = fail
        self.trace_requests: list[Path] = []
        self.closed = False

    @property
    def complete(self) -> bool:
        return self.step >= 2

    def advance_one(self) -> None:
        self.step += 1

    def finalize(self, *, trace_path=None):
        if self.fail:
            raise RuntimeError("injected simulation failure")
        if not self.complete:
            raise RuntimeError("incomplete fake session")
        if trace_path is not None:
            path = Path(trace_path)
            self.trace_requests.append(path)
            np.savez_compressed(
                path,
                time=np.asarray((0.001, 0.002)),
                control_state=np.asarray(("VERIFY", "HOLD")),
            )
        return copy.deepcopy(self.summary)

    def close(self) -> None:
        self.closed = True


class _Factory:
    def __init__(self, summary: dict[str, Any], *, fail: bool = False) -> None:
        self.summary = summary
        self.fail = fail
        self.sessions: list[_FakeSession] = []

    def __call__(self, _config: dict[str, Any]) -> _FakeSession:
        value = _FakeSession(self.summary, fail=self.fail)
        self.sessions.append(value)
        return value


def _config() -> dict[str, Any]:
    return load_config(CONFIG)


def test_v14_near_miss_commits_only_authenticated_config_and_summary(
    tmp_path: Path,
) -> None:
    factory = _Factory(_summary(grasp=False, full=False))
    destination = tmp_path / "candidate_14001"
    bundle = run_or_resume_v14_candidate_artifacts(
        _config(),
        destination,
        14001,
        session_factory=factory,
        validator=None,
    )

    assert bundle.reused is False
    assert bundle.trace_retained is False
    assert bundle.trace_path is None
    assert not (destination / "trace.npz").exists()
    assert factory.sessions[0].trace_requests == []
    assert {value.name for value in destination.iterdir()} == {
        "resolved_config.json",
        "result.json",
    }
    assert bundle.result["trace_retention"] == {
        "schema_version": 1,
        "policy": TRACE_RETENTION_POLICY,
        "retained": False,
        "reason": "near_miss_summary_only",
        "retain_grasp_success": True,
        "trace_materialized": False,
        "summary_only": True,
    }
    assert bundle.result["artifacts"]["trace_sha256_at_evaluation"] is None
    authenticate_candidate_result_semantic_sha256(bundle.result)

    resumed = run_or_resume_v14_candidate_artifacts(
        _config(),
        destination,
        14001,
        session_factory=lambda _value: pytest.fail("resume reran physics"),
        validator=None,
    )
    assert resumed.reused is True
    assert resumed.result == bundle.result


@pytest.mark.parametrize(
    ("grasp", "full", "reason"),
    ((True, False, "grasp_success"), (True, True, "full_success")),
)
def test_v14_success_retains_same_session_trace(
    tmp_path: Path, grasp: bool, full: bool, reason: str
) -> None:
    factory = _Factory(_summary(grasp=grasp, full=full))
    destination = tmp_path / f"candidate_{reason}"
    bundle = run_or_resume_v14_candidate_artifacts(
        _config(), destination, 14002, session_factory=factory, validator=None
    )

    assert bundle.trace_retained is True
    assert bundle.trace_path == destination / "trace.npz"
    assert len(factory.sessions) == 1
    assert len(factory.sessions[0].trace_requests) == 1
    assert factory.sessions[0].closed is True
    assert bundle.trace_retention_reason == reason
    with np.load(bundle.trace_path, allow_pickle=False) as trace:
        assert trace["time"].shape == (2,)
    assert bundle.result["artifacts"]["trace_sha256_at_evaluation"] == (
        bundle.result["artifacts"]["sha256"]["trace"]
    )


def test_v14_explicit_final_rerun_retains_failed_trace(tmp_path: Path) -> None:
    factory = _Factory(_summary(grasp=False, full=False))
    bundle = run_or_resume_v14_candidate_artifacts(
        _config(),
        tmp_path / "final_rerun",
        14003,
        final_rerun=True,
        session_factory=factory,
        validator=None,
    )
    assert bundle.trace_retained is True
    assert bundle.trace_retention_reason == "explicit_final_rerun"
    assert bundle.result["classification"] == "near_miss"


def test_v14_high_volume_policy_compacts_grasp_only_but_never_full_success(
    tmp_path: Path,
) -> None:
    grasp_factory = _Factory(_summary(grasp=True, full=False))
    grasp = run_or_resume_v14_candidate_artifacts(
        _config(),
        tmp_path / "grasp_only",
        14007,
        retain_grasp_success=False,
        session_factory=grasp_factory,
        validator=None,
    )
    assert grasp.trace_retained is False
    assert grasp.trace_retention_reason == "grasp_success_summary_only"
    assert grasp_factory.sessions[0].trace_requests == []
    assert grasp.result["trace_retention"]["retain_grasp_success"] is False
    assert grasp.result["grasp_success"] is True

    full_factory = _Factory(_summary(grasp=True, full=True))
    full = run_or_resume_v14_candidate_artifacts(
        _config(),
        tmp_path / "full_success",
        14008,
        retain_grasp_success=False,
        session_factory=full_factory,
        validator=None,
    )
    assert full.trace_retained is True
    assert full.trace_retention_reason == "full_success"


def test_v14_resume_binds_requested_trace_retention_policy(tmp_path: Path) -> None:
    destination = tmp_path / "candidate"
    run_or_resume_v14_candidate_artifacts(
        _config(),
        destination,
        14009,
        retain_grasp_success=False,
        session_factory=_Factory(_summary(grasp=True, full=False)),
        validator=None,
    )
    with pytest.raises(RuntimeError, match="trace-retention policy changed"):
        run_or_resume_v14_candidate_artifacts(
            _config(),
            destination,
            14009,
            retain_grasp_success=True,
            session_factory=lambda _value: pytest.fail("must fail before physics"),
            validator=None,
        )


def test_v14_resume_rejects_summary_only_for_later_final_rerun(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "candidate"
    run_or_resume_v14_candidate_artifacts(
        _config(),
        destination,
        14004,
        session_factory=_Factory(_summary(grasp=False, full=False)),
        validator=None,
    )
    with pytest.raises(RuntimeError, match="cannot satisfy a final-rerun"):
        run_or_resume_v14_candidate_artifacts(
            _config(),
            destination,
            14004,
            final_rerun=True,
            session_factory=lambda _value: pytest.fail("must fail before physics"),
            validator=None,
        )


def test_v14_resume_authenticates_id_config_semantics_and_files(
    tmp_path: Path,
) -> None:
    config = _config()
    destination = tmp_path / "candidate"
    run_or_resume_v14_candidate_artifacts(
        config,
        destination,
        14005,
        session_factory=_Factory(_summary(grasp=True, full=False)),
        validator=None,
    )
    with pytest.raises(RuntimeError, match="wrong candidate ID"):
        authenticate_v14_candidate_artifacts(
            destination, expected_config=config, expected_candidate_id=999
        )
    changed = copy.deepcopy(config)
    changed.setdefault("candidate_metadata", {})["tampered"] = True
    with pytest.raises(RuntimeError, match="requested config changed"):
        authenticate_v14_candidate_artifacts(destination, expected_config=changed)

    result_path = destination / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["summary"]["failed_checks"] = ["changed"]
    write_json(result_path, payload)
    with pytest.raises(RuntimeError, match="semantic SHA-256 mismatch"):
        authenticate_v14_candidate_artifacts(destination)


def test_v14_authentication_rejects_candidate_directory_id_mismatch(
    tmp_path: Path,
) -> None:
    config = _config()
    destination = tmp_path / "candidate_14007"
    run_or_resume_v14_candidate_artifacts(
        config,
        destination,
        14007,
        session_factory=_Factory(_summary(grasp=True, full=False)),
        validator=None,
    )
    mismatched = tmp_path / "candidate_14008"
    destination.rename(mismatched)

    with pytest.raises(RuntimeError, match="directory ID disagrees"):
        authenticate_v14_candidate_artifacts(mismatched)


def test_v14_failed_simulation_never_publishes_partial_directory(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "candidate"
    with pytest.raises(RuntimeError, match="injected simulation failure"):
        run_or_resume_v14_candidate_artifacts(
            _config(),
            destination,
            14006,
            session_factory=_Factory(
                _summary(grasp=False, full=False), fail=True
            ),
            validator=None,
        )
    assert not destination.exists()
    assert list(tmp_path.glob(".candidate.v14-staging.*")) == []


def test_v14_helper_rejects_legacy_schema_without_writing(tmp_path: Path) -> None:
    config = _config()
    config["schema_version"] = 13
    destination = tmp_path / "legacy"
    with pytest.raises(ValueError, match="restricted to schema v14"):
        run_or_resume_v14_candidate_artifacts(
            config,
            destination,
            13001,
            session_factory=lambda _value: pytest.fail("legacy physics must not run"),
            validator=None,
        )
    assert not destination.exists()


def test_v14_candidate_artifact_paths_accepts_only_declared_search_metadata(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "candidate"
    bundle = run_or_resume_v14_candidate_artifacts(
        _config(),
        destination,
        14010,
        session_factory=_Factory(_summary(grasp=False, full=False)),
        validator=None,
    )
    wrapped = {
        **bundle.result,
        "plan_candidate_id": 9001,
        "plan_rank": 0,
        "feedback_index": 63,
        "feedback_id": "feedback_63",
        "artifact_directory": "candidates/candidate_14010",
    }

    assert _candidate_artifact_paths(destination, wrapped) == bundle.artifact_paths


def test_v14_candidate_artifact_paths_rejects_unknown_or_changed_wrapper_fields(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "candidate"
    bundle = run_or_resume_v14_candidate_artifacts(
        _config(),
        destination,
        14011,
        session_factory=_Factory(_summary(grasp=False, full=False)),
        validator=None,
    )

    unknown = {**bundle.result, "undeclared_scheduler_state": 1}
    with pytest.raises(RuntimeError, match="unknown metadata fields"):
        _candidate_artifact_paths(destination, unknown)

    changed = copy.deepcopy(bundle.result)
    changed["classification"] = "success"
    with pytest.raises(RuntimeError, match="changed persisted result fields"):
        _candidate_artifact_paths(destination, changed)

    missing = copy.deepcopy(bundle.result)
    missing.pop("summary_sha256")
    with pytest.raises(RuntimeError, match="lost persisted result fields"):
        _candidate_artifact_paths(destination, missing)
