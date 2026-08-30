from __future__ import annotations

import json
from pathlib import Path

import pytest

from xhand_grasp.actual_contact_grasp_pose_catalog import (
    initialize_or_resume_campaign,
    validate_stage_ledger,
)
from xhand_grasp.artifacts import write_json
from xhand_grasp.grasp_pose import canonical_sha256
from xhand_grasp.tuning.joint_pair_near_zero_campaign import EXPERIMENT_ID
from xhand_grasp.tuning.joint_pair_near_zero_campaign_runner import (
    V15CampaignBackend,
    V15CampaignJob,
    publish_v15_viewer_catalog,
)
from xhand_grasp.tuning.joint_pair_near_zero_refinement_continuation import (
    _commit_target_result,
    _publish_committed_catalog,
    _run_continuation_stage,
    authenticate_feedback_source,
    run_refinement_continuation,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL_SOURCE = (
    ROOT
    / "artifacts"
    / "left_opposed_face_palm_down_joint_pair_near_zero_"
    "contact_preserving_planned_lift"
    / "tune"
    / "formal_campaign_v15_4"
)


def _workspace(tmp_path: Path) -> Path:
    bound = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "uv_lock_sha256": "3" * 64,
        "actual_qpos_source_manifest_sha256": "4" * 64,
        "source_sha256": "5" * 64,
    }
    manifest = {**bound, "campaign_input_sha256": canonical_sha256(bound)}
    return initialize_or_resume_campaign(
        tmp_path / "continuation", manifest, resume=False
    )


def _catalog_record(tmp_path: Path, candidate_id: int = 15200000000000001):
    source = tmp_path / f"exact_{candidate_id}"
    source.mkdir()
    config_path = source / "resolved_config.json"
    result_path = source / "result.json"
    trace_path = source / "trace.npz"
    write_json(config_path, {"candidate_id": candidate_id, "kind": "config"})
    write_json(result_path, {"candidate_id": candidate_id, "kind": "result"})
    trace_path.write_bytes(b"locked-one-ms-trace")
    return {
        "candidate_id": candidate_id,
        "parent_candidate_id": candidate_id - 1,
        "stage": "exact_rerun",
        "grasp_success": True,
        "full_success": False,
        "summary": {"passed": False, "metrics": {}},
        "config_path": str(config_path.resolve()),
        "result_path": str(result_path.resolve()),
        "trace_path": str(trace_path.resolve()),
        "perturbation_pass_count": 0,
    }


def test_continuation_rejects_any_workspace_nested_with_its_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "immutable_source"
    source.mkdir()
    nested = source / "continuation"
    backend = V15CampaignBackend(lambda *args: ())

    with pytest.raises(ValueError, match="disjoint"):
        run_refinement_continuation(
            source,
            nested,
            config_path=tmp_path / "missing.json",
            resume=False,
            target_success_count=1,
            backend=backend,
        )
    assert not nested.exists()

    with pytest.raises(ValueError, match="disjoint"):
        run_refinement_continuation(
            source,
            tmp_path,
            config_path=tmp_path / "missing.json",
            resume=True,
            target_success_count=1,
            backend=backend,
        )


def test_continuation_stage_resume_reauthenticates_job_and_context_hashes(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    job = V15CampaignJob(
        stage="feedback_refinement",
        index=0,
        parent_candidate_id=7,
        payload={"local_index": 0},
        candidate_id=15200000000000042,
    )
    calls: list[str] = []

    def execute(stage, jobs, output, context):
        calls.append(stage)
        return tuple(
            {
                "candidate_id": value.candidate_id,
                "parent_candidate_id": value.parent_candidate_id,
                "stage": stage,
                "grasp_success": True,
                "full_success": False,
                "summary": {"passed": False, "metrics": {}},
            }
            for value in jobs
        )

    backend = V15CampaignBackend(execute)
    context = {"feedback_source_id": "source", "parents": ()}
    first = _run_continuation_stage(
        workspace, "feedback_refinement", (job,), backend, context
    )
    second = _run_continuation_stage(
        workspace, "feedback_refinement", (job,), backend, context
    )
    assert first == second
    assert calls == ["feedback_refinement"]

    with pytest.raises(RuntimeError, match="differs from current jobs/input"):
        _run_continuation_stage(
            workspace,
            "feedback_refinement",
            (job,),
            backend,
            {"feedback_source_id": "changed", "parents": ()},
        )


def test_catalog_recovers_uncommitted_staging_and_uses_ledger_as_truth(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    record = _catalog_record(tmp_path)
    catalog_root = workspace / "catalogs/target_1/manipulation"
    staging = catalog_root.parent / ".manipulation.v15-staging"
    staging.mkdir(parents=True)
    (staging / "partial").write_text("power loss", encoding="utf-8")
    # A complete rename followed by a crash before ledger commit is also
    # recoverable: the publisher authenticates and commits this directory.
    publish_v15_viewer_catalog((record,), catalog_root)

    stage_input = {"ranked_exact_sha256": canonical_sha256((record,))}
    catalog = _publish_committed_catalog(
        workspace,
        target_success_count=1,
        records=(record,),
        robust_candidate_id=None,
        stage_input=stage_input,
    )
    assert catalog.is_file()
    assert not staging.exists()
    ledger = validate_stage_ledger(workspace)
    assert "catalog_target_1" in ledger["stages"]

    before = catalog.read_bytes()
    resumed = _publish_committed_catalog(
        workspace,
        target_success_count=1,
        records=(record,),
        robust_candidate_id=None,
        stage_input=stage_input,
    )
    assert resumed == catalog
    assert catalog.read_bytes() == before


def test_partial_catalog_is_replaced_before_commit(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    record = _catalog_record(tmp_path)
    catalog_root = workspace / "catalogs/target_1/manipulation"
    catalog_root.mkdir(parents=True)
    (catalog_root / "partial").write_text("power loss", encoding="utf-8")

    path = _publish_committed_catalog(
        workspace,
        target_success_count=1,
        records=(record,),
        robust_candidate_id=None,
        stage_input={"source": "fixed"},
    )
    assert path.is_file()
    assert not (catalog_root / "partial").exists()
    assert "catalog_target_1" in validate_stage_ledger(workspace)["stages"]


def test_continuation_results_are_target_specific_and_committed(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    common = {
        "complete": True,
        "experiment_id": EXPERIMENT_ID,
        "workspace": str(workspace),
        "full_success_count": 1,
    }
    target_one = {
        **common,
        "target_success_count": 1,
        "target_reached": True,
    }
    target_five = {
        **common,
        "target_success_count": 5,
        "target_reached": False,
    }
    _commit_target_result(
        workspace,
        target_success_count=1,
        result=target_one,
        stage_input={"catalog": "one"},
    )
    _commit_target_result(
        workspace,
        target_success_count=5,
        result=target_five,
        stage_input={"catalog": "five"},
    )

    assert json.loads(
        (workspace / "continuation_result_target_1.json").read_text()
    ) == target_one
    assert json.loads(
        (workspace / "continuation_result_target_5.json").read_text()
    ) == target_five
    assert not (workspace / "continuation_result.json").exists()
    stages = validate_stage_ledger(workspace)["stages"]
    assert "continuation_result_target_1" in stages
    assert "continuation_result_target_5" in stages


def test_formal_feedback_source_reauthenticates_against_current_jobs() -> None:
    source = authenticate_feedback_source(FORMAL_SOURCE)
    assert len(source.records) == 1024
    assert source.binding["fully_reauthenticated_against_current_jobs"] is True
    assert source.binding["read_only"] is True
    assert source.binding["feedback_job_count"] == 1024
