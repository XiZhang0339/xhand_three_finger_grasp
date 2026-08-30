from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import initialize_or_resume_campaign
from xhand_grasp.artifacts import file_sha256
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning import contact_preserving_micro_jerk_recovery_finalizer as recovery


def _records():
    result = []
    for index in range(recovery.MICRO_CANDIDATE_COUNT):
        physical = f"{index + 1:064x}"
        result.append(
            {
                "candidate_id": 15_100_000_000_000_000 + index,
                "artifact_directory": f"micro_jerk_refinement/candidates/candidate_{index}",
                "config_semantic_sha256": f"{index + 2:064x}",
                "full_success": False,
                "summary": {
                    "failed_checks": ["smooth_motion_jerk_within_limit"],
                    "metrics": {
                        "motion_smoothness": {
                            "operation_peak_abs_filtered_jerk_m_s3": 3.0
                            + index / 10000.0
                        }
                    },
                },
                "rescue_job": {"physical_plan_sha256": physical},
            }
        )
    return result


def _phase_report(records):
    physical = [value["rescue_job"]["physical_plan_sha256"] for value in records]
    return {
        "complete": True,
        # The atomic ledger/directory stage and the report payload predate one
        # another and intentionally use different stable names.
        "stage": recovery.SOURCE_REPORT_STAGE,
        "declared_candidate_count": recovery.MICRO_CANDIDATE_COUNT,
        "candidate_count": recovery.MICRO_CANDIDATE_COUNT,
        "full_success_count": 0,
        "physical_unique_candidate_count": recovery.MICRO_CANDIDATE_COUNT,
        "physical_duplicate_candidate_count": 0,
        "physical_plan_set_sha256": canonical_sha256(sorted(physical)),
        "source_authentication_id": "a" * 64,
        "records": records,
    }


def _fake_source(tmp_path: Path):
    tmp_path.mkdir(parents=True)
    records = tuple(_records())
    files = []
    for name in ("campaign_manifest.json", "stage_ledger.json", "report.json"):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        files.append(path)
    immutable = []
    for name in ("config.json", "model.xml", "uv.lock", "source_catalog.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        immutable.append(path)
    manifest = {
        "seed": 20260821,
        "config_path": str(immutable[0]),
        "config_sha256": file_sha256(immutable[0]),
        "model_path": str(immutable[1]),
        "model_sha256": file_sha256(immutable[1]),
        "uv_lock_path": str(immutable[2]),
        "uv_lock_sha256": file_sha256(immutable[2]),
        "actual_qpos_source_manifest_path": str(immutable[3]),
        "actual_qpos_source_manifest_sha256": file_sha256(immutable[3]),
    }
    return recovery.AuthenticatedMicroJerkExecution(
        root=tmp_path,
        manifest_path=files[0],
        ledger_path=files[1],
        report_path=files[2],
        manifest=manifest,
        ledger={},
        report=_phase_report(list(records)),
        records=records,
        artifact_paths=tuple(files),
        old_execution_source_sha256="b" * 64,
        old_execution_source_files=("xhand_grasp/old_snapshot.py",),
        execution_authentication_id="c" * 64,
    )


def test_phase_report_requires_exact_1024_unique_physical_candidates():
    records = _records()
    report = _phase_report(records)
    normalized = recovery._validate_phase_report_shape(
        report, expected_source_authentication_id="a" * 64
    )
    assert len(normalized) == 1024
    wrong_stage = copy.deepcopy(report)
    wrong_stage["stage"] = recovery.SOURCE_STAGE
    with pytest.raises(RuntimeError, match="incomplete or incompatible"):
        recovery._validate_phase_report_shape(
            wrong_stage, expected_source_authentication_id="a" * 64
        )
    damaged = copy.deepcopy(report)
    damaged["records"][1]["rescue_job"]["physical_plan_sha256"] = damaged[
        "records"
    ][0]["rescue_job"]["physical_plan_sha256"]
    with pytest.raises(RuntimeError, match="reused a physical plan"):
        recovery._validate_phase_report_shape(
            damaged, expected_source_authentication_id="a" * 64
        )


def test_recovery_manifest_binds_old_and_current_source_snapshots(tmp_path):
    source = _fake_source(tmp_path / "source")
    manifest = recovery.build_micro_jerk_recovery_manifest(source)
    assert manifest["old_execution_source_sha256"] == "b" * 64
    assert manifest["source_sha256"] == recovery.current_source_snapshot()[
        "source_sha256"
    ]
    assert manifest["source_sha256"] != manifest["old_execution_source_sha256"]
    assert manifest["selected_candidate_ids"] == [
        15_100_000_000_000_000 + value for value in range(5)
    ]
    assert manifest["campaign_input_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "campaign_input_sha256"}
    )


def test_recovery_manifest_resume_rejects_current_source_tamper(tmp_path):
    source = _fake_source(tmp_path / "source")
    manifest = recovery.build_micro_jerk_recovery_manifest(source)
    output = tmp_path / "recovery"
    initialize_or_resume_campaign(output, manifest, resume=False)
    initialize_or_resume_campaign(output, manifest, resume=True)
    changed = copy.deepcopy(manifest)
    changed["source_sha256"] = "d" * 64
    changed["campaign_input_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "campaign_input_sha256"}
    )
    with pytest.raises(RuntimeError, match="source_sha256"):
        initialize_or_resume_campaign(output, changed, resume=True)


def test_top_five_resume_authentication_rejects_summary_difference(
    tmp_path, monkeypatch
):
    selected = _records()[:5]
    persisted = [
        {
            **copy.deepcopy(value),
            "artifact_directory": f"reruns/candidate_{value['candidate_id']}",
        }
        for value in selected
    ]
    report = {
        "complete": True,
        "candidate_count": 5,
        "records": persisted,
    }

    def fake_authenticator(*args, **kwargs):
        return SimpleNamespace(result={"summary": {"changed": True}})

    monkeypatch.setattr(
        recovery, "authenticate_v14_candidate_artifacts", fake_authenticator
    )
    with pytest.raises(RuntimeError, match="summary differs"):
        recovery._validate_top_five_report(report, selected, tmp_path)


def test_manifest_selection_is_worker_order_independent(tmp_path):
    source = _fake_source(tmp_path / "source")
    reversed_source = recovery.AuthenticatedMicroJerkExecution(
        **{
            field: getattr(source, field)
            for field in source.__dataclass_fields__
            if field != "records"
        },
        records=tuple(reversed(source.records)),
    )
    first = recovery.build_micro_jerk_recovery_manifest(source)
    second = recovery.build_micro_jerk_recovery_manifest(reversed_source)
    assert first["selected_candidate_ids"] == second["selected_candidate_ids"]
    assert first["selected_records_sha256"] == second["selected_records_sha256"]
