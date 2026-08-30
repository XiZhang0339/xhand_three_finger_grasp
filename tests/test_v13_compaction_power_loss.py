from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    bind_candidate_result_semantic_sha256,
)
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.tuning import actual_contact_grasp_pose as legacy_campaign
from xhand_grasp.tuning import scaled_contact_downsize_ranking as ranking


def _summary(success: bool) -> dict:
    return {
        "passed": success,
        "failed_checks": [] if success else ["operation_median_lift_reached"],
        "stage_status": {
            "grasp_success": True,
            "manipulation_success": success,
            "full_success": success,
        },
    }


def _candidate(root: Path, identifier: int, *, success: bool) -> dict:
    directory = root / f"candidate_{identifier}"
    directory.mkdir(parents=True)
    config = {"schema_version": 13, "candidate_metadata": {"candidate_id": identifier}}
    config_path = directory / "resolved_config.json"
    result_path = directory / "result.json"
    trace_path = directory / "trace.npz"
    write_json(config_path, config)
    trace_path.write_bytes(f"trace-{identifier}".encode())
    payload = bind_candidate_result_semantic_sha256(
        {
            "complete": True,
            "candidate_id": identifier,
            "summary": _summary(success),
            "artifacts": {
                "resolved_config": config_path.name,
                "trace": trace_path.name,
                "trace_retained": True,
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "trace": file_sha256(trace_path),
                },
            },
        }
    )
    write_json(result_path, payload)
    return {
        "candidate_id": identifier,
        "discovery_index": identifier,
        "config": config,
        "config_path": config_path,
        "result_path": result_path,
        "trace_path": trace_path,
        "summary": copy.deepcopy(payload["summary"]),
        "result_semantic_sha256": payload["result_semantic_sha256"],
        "trace_retained": True,
        "trace_sha256_at_evaluation": file_sha256(trace_path),
        "manipulation_delta_rad": {},
    }


def _install_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(directory, *, candidate_id, expected_config):
        directory = Path(directory)
        result_path = directory / "result.json"
        trace_path = directory / "trace.npz"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        artifacts = payload["artifacts"]
        retained = bool(artifacts.get("trace_retained", True))
        trace_sha = (
            artifacts["sha256"].get("trace")
            if retained
            else artifacts.get("trace_sha256_at_evaluation")
        )
        return {
            "candidate_id": int(candidate_id),
            "discovery_index": int(candidate_id),
            "config": copy.deepcopy(expected_config),
            "config_path": directory / "resolved_config.json",
            "result_path": result_path,
            "trace_path": trace_path,
            "summary": copy.deepcopy(payload["summary"]),
            "result_semantic_sha256": payload["result_semantic_sha256"],
            "trace_retained": retained,
            "trace_sha256_at_evaluation": trace_sha,
            "manipulation_delta_rad": {},
        }

    monkeypatch.setattr(
        legacy_campaign, "_load_persisted_manipulation_candidate", load
    )
    monkeypatch.setattr(
        ranking,
        "v13_manipulation_candidate_rank",
        lambda value, _config: (
            not bool(value["summary"]["stage_status"]["full_success"]),
            int(value["candidate_id"]),
        ),
    )


def _set_result_compacted(candidate: dict, *, keep_tombstone: bool) -> None:
    result_path = Path(candidate["result_path"])
    trace_path = Path(candidate["trace_path"])
    tombstone = trace_path.parent / ".trace.npz.compacting"
    observed = file_sha256(tombstone if tombstone.exists() else trace_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    artifacts = payload["artifacts"]
    artifacts["trace"] = None
    artifacts["trace_retained"] = False
    artifacts["trace_sha256_at_evaluation"] = observed
    artifacts["sha256"].pop("trace", None)
    write_json(result_path, payload)
    if not keep_tombstone:
        (tombstone if tombstone.exists() else trace_path).unlink()


@pytest.mark.parametrize(
    "interrupted_state",
    ("before_rename", "after_rename", "after_result", "after_unlink"),
)
def test_v13_compactor_recovers_each_power_loss_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_state: str,
) -> None:
    _install_loader(monkeypatch)
    records = (
        _candidate(tmp_path, 1, success=True),
        _candidate(tmp_path, 2, success=False),
        _candidate(tmp_path, 3, success=False),
    )
    victim = records[2]
    trace = Path(victim["trace_path"])
    tombstone = trace.parent / ".trace.npz.compacting"
    if interrupted_state in {"after_rename", "after_result"}:
        trace.replace(tombstone)
    if interrupted_state == "after_result":
        _set_result_compacted(victim, keep_tombstone=True)
    elif interrupted_state == "after_unlink":
        _set_result_compacted(victim, keep_tombstone=False)

    compacted, report = ranking.compact_v13_manipulation_candidate_artifacts(
        records,
        retain_failure_trace_count=1,
        report_path=tmp_path / f"{interrupted_state}.json",
    )
    assert report["complete"] is True
    assert report["actual_contact_manipulation_compaction_schema_version"] == 2
    assert [value["candidate_id"] for value in compacted] == [1, 2, 3]
    assert Path(records[0]["trace_path"]).is_file()
    assert Path(records[1]["trace_path"]).is_file()
    assert not trace.exists()
    assert not tombstone.exists()
    victim_result = json.loads(Path(victim["result_path"]).read_text())
    assert victim_result["artifacts"]["trace_retained"] is False


def test_v13_compactor_restores_retained_trace_after_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_loader(monkeypatch)
    records = (
        _candidate(tmp_path, 1, success=True),
        _candidate(tmp_path, 2, success=False),
    )
    retained = records[1]
    trace = Path(retained["trace_path"])
    expected = file_sha256(trace)
    tombstone = trace.parent / ".trace.npz.compacting"
    trace.replace(tombstone)

    ranking.compact_v13_manipulation_candidate_artifacts(
        records, retain_failure_trace_count=1
    )
    assert trace.is_file()
    assert file_sha256(trace) == expected
    assert not tombstone.exists()


def test_v13_compaction_report_authenticates_state_and_monotonic_compaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_loader(monkeypatch)
    records = (
        _candidate(tmp_path, 1, success=True),
        _candidate(tmp_path, 2, success=False),
        _candidate(tmp_path, 3, success=False),
    )
    first_report = tmp_path / "keep_two_failures.json"
    ranking.compact_v13_manipulation_candidate_artifacts(
        records, retain_failure_trace_count=2, report_path=first_report
    )
    authenticated = ranking.authenticate_v13_manipulation_compaction_report(
        records, first_report, retain_failure_trace_count=2
    )
    assert all(value["trace_retained"] for value in authenticated)

    # A wider later compaction may validly change retained=True to False while
    # preserving candidate identity, semantic evidence and evaluation trace hash.
    ranking.compact_v13_manipulation_candidate_artifacts(
        records,
        retain_failure_trace_count=1,
        report_path=tmp_path / "keep_one_failure.json",
    )
    transitioned = ranking.authenticate_v13_manipulation_compaction_report(
        records, first_report, retain_failure_trace_count=2
    )
    assert [value["trace_retained"] for value in transitioned] == [True, True, False]


def test_v13_compaction_report_rejects_stale_candidate_set_and_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_loader(monkeypatch)
    records = (
        _candidate(tmp_path, 1, success=True),
        _candidate(tmp_path, 2, success=False),
    )
    report_path = tmp_path / "compaction.json"
    ranking.compact_v13_manipulation_candidate_artifacts(
        records, retain_failure_trace_count=1, report_path=report_path
    )
    with pytest.raises(RuntimeError, match="candidate set changed"):
        ranking.authenticate_v13_manipulation_compaction_report(
            records[:1], report_path, retain_failure_trace_count=1
        )

    payload = json.loads(report_path.read_text())
    payload["candidate_records"][0]["result_semantic_sha256"] = "0" * 64
    write_json(report_path, payload)
    with pytest.raises(RuntimeError, match="candidate evidence changed"):
        ranking.authenticate_v13_manipulation_compaction_report(
            records, report_path, retain_failure_trace_count=1
        )


@pytest.mark.parametrize("tamper", ("trace", "result_semantics"))
def test_v13_compaction_report_rejects_changed_persisted_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    _install_loader(monkeypatch)
    records = (
        _candidate(tmp_path, 1, success=True),
        _candidate(tmp_path, 2, success=False),
    )
    report_path = tmp_path / "compaction.json"
    ranking.compact_v13_manipulation_candidate_artifacts(
        records, retain_failure_trace_count=1, report_path=report_path
    )
    if tamper == "trace":
        Path(records[1]["trace_path"]).write_bytes(b"changed-trace")
        match = "trace state SHA-256 mismatch"
    else:
        result_path = Path(records[1]["result_path"])
        payload = json.loads(result_path.read_text())
        payload["summary"]["failed_checks"].append("tampered")
        write_json(result_path, payload)
        match = "semantic SHA-256 mismatch"
    with pytest.raises(RuntimeError, match=match):
        ranking.authenticate_v13_manipulation_compaction_report(
            records, report_path, retain_failure_trace_count=1
        )


def test_v13_compactor_rejects_ambiguous_or_lost_trace_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_loader(monkeypatch)
    record = _candidate(tmp_path, 1, success=True)
    trace = Path(record["trace_path"])
    tombstone = trace.parent / ".trace.npz.compacting"
    tombstone.write_bytes(trace.read_bytes())
    with pytest.raises(RuntimeError, match="both exist"):
        ranking.compact_v13_manipulation_candidate_artifacts((record,))

    tombstone.unlink()
    trace.unlink()
    with pytest.raises(RuntimeError, match="disappeared"):
        ranking.compact_v13_manipulation_candidate_artifacts((record,))
