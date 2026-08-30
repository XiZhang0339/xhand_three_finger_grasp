from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    authenticate_candidate_result_semantic_sha256,
    bind_candidate_result_semantic_sha256,
    commit_campaign_stage,
)
from xhand_grasp.artifacts import file_sha256, write_json
from xhand_grasp.config import ACTIVE_ACTUATORS, load_config
from xhand_grasp.grasp_pose import controller_id, grasp_pose_id
from xhand_grasp.tuning import actual_contact_grasp_pose as campaign
from xhand_grasp.tuning.actual_contact_manipulation import LocalRefinementBudget


TEMPLATE = Path(
    "grasp_configs/left_opposed_face_palm_down_actual_contact_grasp_pose_"
    "smooth_vertical_lift.json"
)


def _summary(*, full: bool, lift_m: float) -> dict:
    return {
        "passed": full,
        "failed_checks": [] if full else ["operation_median_lift_reached"],
        "stage_status": {"grasp_success": True, "full_success": full},
        "metrics": {
            "operation_median_lift_m": lift_m,
            "operation_minimum_lift_m": lift_m,
            "operation_target_face_simultaneous_duty": 0.7,
            "contact_alignment": {"operation": {"aligned_duty": 0.7}},
        },
    }


def _write_candidate(
    directory: Path, config: dict, identifier: int, *, full: bool, lift_m: float
) -> dict:
    directory.mkdir(parents=True)
    config_path = directory / "resolved_config.json"
    trace_path = directory / "trace.npz"
    write_json(config_path, config)
    np.savez_compressed(trace_path, time=np.asarray([0.001]))
    summary = _summary(full=full, lift_m=lift_m)
    payload = bind_candidate_result_semantic_sha256(
        {
            "actual_contact_manipulation_candidate_schema_version": 1,
            "complete": True,
            "candidate_id": identifier,
            "discovery_index": identifier,
            "source_candidate_id": 1,
            "candidate_sha256": campaign.canonical_sha256(config),
            "grasp_pose_id": grasp_pose_id(config),
            "controller_id": controller_id(config),
            "initial_state_source": "configured_no_contact_reset",
            "checkpoint_used": False,
            "full_success": full,
            "summary": summary,
            "artifacts": {
                "resolved_config": "resolved_config.json",
                "trace": "trace.npz",
                "trace_retained": True,
                "sha256": {
                    "resolved_config": file_sha256(config_path),
                    "trace": file_sha256(trace_path),
                },
            },
        }
    )
    result_path = directory / "result.json"
    write_json(result_path, payload)
    return {
        "candidate_id": identifier,
        "discovery_index": identifier,
        "config": copy.deepcopy(config),
        "config_path": config_path,
        "result_path": result_path,
        "trace_path": trace_path,
        "manipulation_delta_rad": copy.deepcopy(
            config["control"]["manipulation_delta_rad"]
        ),
        "summary": summary,
    }


def test_compaction_keeps_every_success_and_only_best_failed_trace(tmp_path) -> None:
    config = load_config(TEMPLATE)
    records = [
        _write_candidate(tmp_path / f"candidate_{index}", config, index,
                         full=index == 3, lift_m=0.001 * index)
        for index in range(4)
    ]
    report_path = tmp_path / "compaction.json"
    compacted, report = campaign.compact_manipulation_candidate_artifacts(
        records, retain_failure_trace_count=1, report_path=report_path
    )
    retained = {value["candidate_id"] for value in compacted if value["trace_retained"]}
    assert retained == {2, 3}
    assert report["success_traces_always_retained"] is True
    assert report["compacted_trace_count"] == 2
    for value in compacted:
        payload = json.loads(Path(value["result_path"]).read_text())
        authenticate_candidate_result_semantic_sha256(payload)
        assert Path(value["trace_path"]).is_file() is value["trace_retained"]
    resumed, resumed_report = campaign.compact_manipulation_candidate_artifacts(
        compacted, retain_failure_trace_count=1, report_path=report_path
    )
    assert resumed_report == report
    assert [value["candidate_id"] for value in resumed] == [0, 1, 2, 3]

    expanded_retention, expanded_report = (
        campaign.compact_manipulation_candidate_artifacts(
            compacted, retain_failure_trace_count=3
        )
    )
    assert {
        value["candidate_id"]
        for value in expanded_retention
        if value["trace_retained"]
    } == {2, 3}
    assert expanded_report["retained_trace_count"] == 2


def test_committed_quick_stage_loads_after_later_global_compaction(tmp_path) -> None:
    root = tmp_path / "campaign"
    config = load_config(TEMPLATE)
    records = [
        _write_candidate(
            root / "manipulation" / "candidates" / f"candidate_{index}",
            config,
            index,
            full=index == 3,
            lift_m=0.001 * index,
        )
        for index in range(4)
    ]

    def stage_report(path: Path, values: tuple[dict, ...], stage: str) -> None:
        write_json(
            path,
                {
                    "actual_contact_manipulation_stage_schema_version": 1,
                    "complete": True,
                "target_success_count": 1,
                "candidate_count": len(values),
                "candidate_records": [
                    {
                        "candidate_id": int(value["candidate_id"]),
                        "discovery_index": int(value["discovery_index"]),
                        "config_path": str(Path(value["config_path"]).relative_to(root)),
                        "result_path": str(Path(value["result_path"]).relative_to(root)),
                        "trace_path": str(Path(value["trace_path"]).relative_to(root)),
                        "result_semantic_sha256": value["result_semantic_sha256"],
                        "trace_retained": bool(value["trace_retained"]),
                    }
                    for value in values
                ],
                "stage": stage,
            },
        )

    quick_compaction = root / "manipulation/quick_target_1_compaction.json"
    write_json(
        root / "stage_ledger.json",
        {
            "stage_ledger_schema_version": 1,
            "experiment_id": campaign.EXPERIMENT_ID,
            "campaign_input_sha256": "synthetic",
            "stages": {},
        },
    )
    quick, _ = campaign.compact_manipulation_candidate_artifacts(
        records, retain_failure_trace_count=1, report_path=quick_compaction
    )
    quick_report = root / "manipulation/quick_target_1_report.json"
    stage_report(quick_report, quick, "quick")
    commit_campaign_stage(
        root,
        "quick_manipulation_1",
        stage_input={"target": 1},
        artifacts=(quick_compaction, quick_report),
        summary={"candidate_count": 4, "target_reached": False},
    )

    expanded_compaction = root / "manipulation/expanded_target_1_compaction.json"
    expanded, _ = campaign.compact_manipulation_candidate_artifacts(
        quick, retain_failure_trace_count=0, report_path=expanded_compaction
    )
    expanded_report = root / "manipulation/expanded_target_1_report.json"
    stage_report(expanded_report, expanded, "expanded")
    commit_campaign_stage(
        root,
        "expanded_manipulation_1",
        stage_input={"target": 1},
        artifacts=(expanded_compaction, expanded_report),
        summary={"candidate_count": 4, "target_reached": False},
    )

    loaded = campaign._load_committed_manipulation_stage(
        root, stage="quick", target_success_count=1
    )
    assert loaded is not None
    assert [value["candidate_id"] for value in loaded.records] == [0, 1, 2, 3]
    assert {
        value["candidate_id"]
        for value in loaded.records
        if value["trace_retained"]
    } == {3}


def test_small_formal_refinement_is_batched_and_resumable(tmp_path, monkeypatch) -> None:
    config = load_config(TEMPLATE)
    parents = []
    for identifier in range(2):
        parents.append(
            {
                "candidate_id": identifier,
                "discovery_index": identifier,
                "candidate_sha256": campaign.canonical_sha256(config),
                "result_semantic_sha256": f"{identifier + 1:064x}",
                "config": copy.deepcopy(config),
                "manipulation_delta_rad": copy.deepcopy(
                    config["control"]["manipulation_delta_rad"]
                ),
                "summary": _summary(full=False, lift_m=0.001 * identifier),
            }
        )
    calls = []

    def fake_jobs(jobs, workers):
        calls.append((len(jobs), workers))
        output = []
        for job in jobs:
            metadata = job["job_metadata"]
            record = _write_candidate(
                Path(metadata["refinement_root"])
                / "candidates"
                / f"candidate_{job['candidate_id']}",
                job["config"],
                int(job["candidate_id"]),
                full=False,
                lift_m=0.002,
            )
            output.append({"candidate_id": job["candidate_id"], "summary": record["summary"]})
        return tuple(output)

    monkeypatch.setattr(
        campaign, "_run_persisted_manipulation_refinement_jobs", fake_jobs
    )
    budget = LocalRefinementBudget(
        parent_count=2, candidates_per_parent=2, batch_size=2
    )
    first = campaign._run_manipulation_local_refinement_stage(
        parents, tmp_path, seed=20260821, workers=3, budget=budget
    )
    assert len(first.records) == 4
    assert first.summary["declared_candidate_budget"] == 4
    assert calls == [(2, 3), (2, 3)]
    calls.clear()
    second = campaign._run_manipulation_local_refinement_stage(
        parents, tmp_path, seed=20260821, workers=3, budget=budget
    )
    assert len(second.records) == 4
    assert calls == []
