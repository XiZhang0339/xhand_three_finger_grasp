from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from xhand_grasp.config import load_config
from xhand_grasp.tuning.contact_preserving_candidate_artifacts import (
    authenticate_v14_candidate_artifacts,
    run_or_resume_v14_candidate_artifacts,
)
import xhand_grasp.tuning.contact_preserving_lift_rescue_campaign as rescue


CONFIG = Path(
    "grasp_configs/left_opposed_face_palm_down_contact_preserving_planned_lift.json"
)


class _SummarySession:
    def __init__(self) -> None:
        self._complete = False

    @property
    def complete(self) -> bool:
        return self._complete

    def advance_one(self) -> None:
        self._complete = True

    def finalize(self, *, trace_path=None):
        if trace_path is not None:
            np.savez_compressed(trace_path, time=np.asarray([0.001]))
        return {
            "passed": False,
            "failed_checks": ["smooth_motion_jerk_within_limit"],
            "stage_status": {
                "grasp_success": True,
                "manipulation_success": False,
                "full_success": False,
            },
            "metrics": {},
        }

    def close(self) -> None:
        return None


def _fake_candidate_worker(payload):
    bundle = run_or_resume_v14_candidate_artifacts(
        payload["config"],
        payload["destination"],
        int(payload["candidate_id"]),
        retain_grasp_success=False,
        session_factory=lambda _config: _SummarySession(),
    )
    return copy.deepcopy(bundle.result)


def _phase_report(*, full_count: int) -> dict:
    records = [
        {"candidate_id": index + 1, "full_success": index < full_count}
        for index in range(rescue.PHASE_ONE_CANDIDATE_COUNT)
    ]
    return {
        "complete": True,
        "declared_candidate_count": rescue.PHASE_ONE_CANDIDATE_COUNT,
        "candidate_count": rescue.PHASE_ONE_CANDIDATE_COUNT,
        "full_success_count": full_count,
        "records": records,
    }


def test_time_warp_gate_runs_only_after_exact_zero_success_phase() -> None:
    assert rescue._time_warp_required(_phase_report(full_count=0)) is True
    assert rescue._time_warp_required(_phase_report(full_count=1)) is False
    malformed = _phase_report(full_count=0)
    malformed["records"].pop()
    with pytest.raises(RuntimeError, match="registered budget"):
        rescue._time_warp_required(malformed)


def test_candidate_batch_is_full_reset_atomic_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rescue, "_run_candidate_job", _fake_candidate_worker)
    config = load_config(CONFIG)
    jobs = (
        {
            "candidate_id": 141,
            "candidate_sha256": "a" * 64,
            "job_sequence_index": 1,
            "config": config,
            "job_metadata": {"job_sequence_index": 1, "family": "second"},
        },
        {
            "candidate_id": 140,
            "candidate_sha256": "b" * 64,
            "job_sequence_index": 0,
            "config": config,
            "job_metadata": {"job_sequence_index": 0, "family": "first"},
        },
    )
    records, artifacts = rescue._execute_candidate_jobs(
        jobs,
        tmp_path,
        phase_name="phase/candidates",
        workers=1,
        global_rank_offset=10,
    )
    assert [value["candidate_id"] for value in records] == [140, 141]
    assert [value["plan_rank"] for value in records] == [10, 11]
    assert all(value["trace_retention"]["summary_only"] for value in records)
    assert len(artifacts) == 4
    assert all(value.is_file() for value in artifacts)

    repeated, repeated_artifacts = rescue._execute_candidate_jobs(
        jobs,
        tmp_path,
        phase_name="phase/candidates",
        workers=1,
        global_rank_offset=10,
    )
    assert repeated == records
    assert repeated_artifacts == artifacts


def test_candidate_batch_rejects_duplicate_ids_before_physics(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    jobs = tuple(
        {
            "candidate_id": 140,
            "config": config,
            "job_metadata": {"job_sequence_index": index},
        }
        for index in range(2)
    )
    with pytest.raises(RuntimeError, match="not unique"):
        rescue._execute_candidate_jobs(
            jobs,
            tmp_path,
            phase_name="phase/candidates",
            workers=1,
            global_rank_offset=0,
        )


def test_authenticated_refinement_bundle_import_is_atomic_and_reusable(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    candidate_id = 140123
    source = tmp_path / "source" / f"candidate_{candidate_id}"
    source_bundle = run_or_resume_v14_candidate_artifacts(
        config,
        source,
        candidate_id,
        retain_grasp_success=False,
        session_factory=lambda _config: _SummarySession(),
    )
    job = {"candidate_id": candidate_id, "config": config}
    workspace = tmp_path / "new"
    rescue._import_reused_refinement_candidates(
        {"bundles": {candidate_id: source_bundle}}, (job,), workspace
    )
    destination = (
        workspace
        / "refinement_rescue"
        / "candidates"
        / f"candidate_{candidate_id}"
    )
    imported = authenticate_v14_candidate_artifacts(
        destination,
        expected_config=config,
        expected_candidate_id=candidate_id,
        expected_retain_grasp_success=False,
    )
    assert imported.result == source_bundle.result
    assert not list(destination.parent.glob(".candidate_*.reuse-staging.*"))
    # A resumed import authenticates the committed directory and is a no-op.
    rescue._import_reused_refinement_candidates(
        {"bundles": {candidate_id: source_bundle}}, (job,), workspace
    )


def test_rescue_catalog_selection_uses_rescue_rank_and_keeps_first_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {"candidate_id": value, "full_success": True, "plan_rank": value - 9}
        for value in range(10, 16)
    ]
    # The chronologically first success is outside the rescue top five and must
    # replace only the last slot, while the remaining order still comes from the
    # dedicated rescue ranker rather than the legacy contact-only ranker.
    records[0]["plan_rank"] = 0

    def fake_rank(values):
        return tuple(
            copy.deepcopy(dict(value))
            for value in sorted(values, key=lambda item: -int(item["candidate_id"]))
        )

    monkeypatch.setattr(rescue, "_rank_phase_one", fake_rank)
    selected = rescue._selected_rescue_catalog_records(records)
    assert [value["candidate_id"] for value in selected] == [15, 14, 13, 12, 10]


def test_time_warp_receives_only_rescue_ranked_top_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {"candidate_id": value, "config": {"schema_version": 14}}
        for value in range(8)
    ]
    ranked = tuple(reversed(records))
    monkeypatch.setattr(rescue, "_rank_phase_one", lambda _values: ranked)
    monkeypatch.setattr(
        rescue, "_records_with_configs", lambda _records, _workspace: tuple(records)
    )
    parents = rescue._select_time_warp_parent_records(records, tmp_path)
    assert [value["candidate_id"] for value in parents] == [7, 6, 5, 4]
